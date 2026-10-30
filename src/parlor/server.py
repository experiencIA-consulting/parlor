"""Parlor — real-time multimodal AI (voice + vision) on a cloud LLM.

Instead of a local llama.cpp model, the stack talks to hosted providers:
the utterance is transcribed through the OpenAI speech API (see cloud.py),
the reply streams from any OpenAI-compatible chat endpoint (api.openai.com
directly, or OpenRouter), and TTS is cloud-speached sentence-by-sentence
(see tts.py). This server owns the conversation history and re-sends it
every request; the client holds it prompt-free, so a turn's cost is just
the model's own tokens. No local model downloads, no GPU needed.
"""

import asyncio
import base64
import itertools
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

# Piped stdout on Windows defaults to cp1252 ('strict'): one non-latin1
# character in ANY log line ('→' in timings, a Spanish transcript) killed
# every LLM turn mid-flight — after the reply arrived, before the decider
# ran — so the whole turn collapsed to a silent release. UTF-8 makes the
# console file byte-readable everywhere; 'replace' keeps residue harmless.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):  # pragma: no cover
        pass

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:  # raised by ws.send_text after the client goes away (uvicorn-specific)
    from uvicorn.protocols.utils import ClientDisconnected
except ImportError:  # pragma: no cover
    class ClientDisconnected(OSError):
        pass

DISCONNECT_ERRORS = (WebSocketDisconnect, ClientDisconnected)

from dotenv import load_dotenv
load_dotenv()  # before importing cloud — it reads its config at import time

from parlor import actions, cloud, personas, reasoner, tts
from parlor.modes import MODES
from parlor.pipeline import (estimate_tokens, image_part, merge_wavs,
                             release_client, run_turn, send_json,
                             speak_only, text_part, valid_audio,
                             wav_to_float32)

# Turn completeness is judged by the smart-turn audio classifier before the
# LLM is involved, so the prompt carries no FINISHED/WAIT machinery at all.
# The user's words now arrive PRE-transcribed (OpenAI speech API — see
# cloud.py) instead of being heard by the model itself, and that fact is
# stated here once so the transcript is treated as speech, not typed text.
SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user talks to you "
    "through a microphone and may show you their camera: each message starts "
    "with a transcript of what they actually SAID (spoken words, not typing). "
    "Your replies are spoken aloud, so write plain conversational text "
    "without formatting. If a message's transcript is just your own previous "
    "reply playing back (echo), don't answer it — briefly ask what they'd "
    "like to talk about."
)

# The model's replies are pure speech: it never emits control markup.
# Actions (timers, mode switches, research) are decided by a separate
# grammar-forced request — see actions.py for the measured rationale.
# The speech prompt only needs the model to KNOW the capabilities exist,
# so confirmations sound natural and never contradict what the decider
# does. This note lives in the system prompt because it changes no
# per-turn behavior — unlike the old in-band tags, nothing here has to
# fight for attention next to the audio.
CAPABILITY_NOTE = (
    " You can set countdown timers, translate everything the user says, "
    "or just listen quietly while they think out loud. When the user "
    "asks for one of these, confirm it in one short natural sentence — a "
    "separate system watches the conversation and actually performs it, "
    "so never say you can't, and never invent its results."
)

# Appended only when a reasoner endpoint is configured (.env REASONER_*):
# the model must not answer current-info questions from stale memory —
# the decider hands them to the background assistant. The closing
# "everything else, answer yourself" is load-bearing: without it, once
# history contains one "I'll find out" the model starts deferring stable
# general-knowledge questions too (both e2e failures of the first
# migration run were "capital of France" answered with a deferral).
RESEARCH_NOTE = (
    " You also have a background research assistant with web access: "
    "when the user asks you to search, look up, or research something, "
    "or asks about anything current or changing (weather, news, prices, "
    "scores), don't answer from memory — your knowledge is stale. Say in "
    "one short sentence that you'll find out; the answer arrives later "
    "and you can share it then. Everything else — greetings, chat, "
    "general knowledge, facts that don't change — answer yourself, "
    "directly and completely."
)

# The per-utterance instruction in translate mode. Exit commands are not
# its concern: the action decider judges every utterance BEFORE this
# prompt runs (see the turn loop), so the prompt stays a pure interpreter.
TRANSLATE_PROMPT = (
    "Translate what the user said into English. Output ONLY the English "
    "translation of the words above — no commentary, no answers, no "
    "opinions. If they already spoke English, restate their words in "
    "clear English."
)

# Modes whose audio turns replace the standard respond/flush instruction
# wholesale (the mode chooses what a turn IS, not just how it's phrased).
# Listen has no per-turn prompt: its content turns are transcribed and
# shown client-side WITHOUT an LLM call — the words the user thinks aloud
# are already known precisely (the STT text), so re-asking a model to
# echo them as a transcript line would only paraphrase and burn tokens.
MODE_PROMPTS = {"translate": TRANSLATE_PROMPT}

# The ring is a server-initiated turn like a delegation delivery: the
# model phrases the announcement, and if it yields nothing speakable the
# fallback IS the announcement. "System note (not user audio)" for the
# same anti-echo reason as DELIVER_PROMPT.
TIMER_PROMPT = (
    'System note (not user audio): the user\'s countdown timer "{label}" '
    "(set for {duration}) just went off. Tell them it's done in one short "
    "spoken sentence. Don't start anything new."
)
TIMER_FALLBACK = "Ding — your {label} timer is done."

# A session can only usefully run a few timers, and a runaway decision
# must not schedule endless rings; > 8h is a misread, not a timer. The
# duration itself arrives as integer seconds from the action decider —
# the model does the word-number math, in any language.
MAX_PENDING_TIMERS = 3
MAX_TIMER_S = 8 * 3600


def duration_phrase(seconds: float) -> str:
    """Exact spoken duration for timer prompts ('3 minutes', '90 seconds')
    — unlike elapsed_phrase, a timer's length is not approximate."""
    if seconds < 120:
        return f"{seconds:g} seconds"
    if seconds < 3600:
        return f"{seconds / 60:g} minutes"
    n = seconds / 3600
    return "1 hour" if n == 1 else f"{n:g} hours"

# A finished background task is delivered by the voice model, not read out
# raw: it ties the answer back to the conversation and keeps one voice.
# "System note (not user audio)" leads both prompts because the model
# occasionally misread the text-only turn as its own reply playing back
# and gave the anti-echo response instead of the answer (observed live).
DELIVER_PROMPT = (
    "System note (not user audio): your background research assistant "
    'just finished the task "{task}" (it took {took}). Its answer:\n'
    "{answer}\n\n"
    "Tell the user now: one short lead-in sentence tying it back to what "
    "they asked, then the answer itself word-for-word. Never drop or "
    "change a name, number, or place — the words above are already "
    "spoken-style. Plain spoken text."
)
DELIVER_FAILED_PROMPT = (
    "System note (not user audio): your background research assistant "
    'could not finish the task "{task}". Briefly tell the user you '
    "couldn't get that answer, and that they're welcome to ask again. "
    "Don't start new research now."
)
DELIVER_FALLBACK = "Sorry — I couldn't get an answer for that one."

# One session can only usefully consume a few pending research tasks, and
# an off-the-rails reply must not queue an HTTP call per imagined tag.
MAX_PENDING_DELEGATIONS = 3

# A "respond" turn: the user's transcript leads the message (the STT
# words), the instruction follows. With cloud STT the transcript is
# exact by construction — no leading ###TRANSCRIPT: line is asked of
# the model and none needs parsing out of the reply.
RESPOND_PROMPT = (
    "Respond to what the user said: 1-4 short sentences, spoken aloud."
    "{camera}"
)

# The clause a camera turn appends to its instruction — a named constant
# so benchmarks measure the production wording, never a retyped copy.
CAMERA_CLAUSE = " Mention what you see in their camera image if relevant."

# Spoken when a turn yields no reply at all — silence would leave the
# user hanging, and a stored transcript-only reply teaches the model to
# do it again next turn.
FLUSH_FALLBACK = "Take your time — I'm listening."
AUDIO_FALLBACK = "Sorry, I didn't catch that — could you say it again?"

# A "flush" turn: the classifier judged the utterance unfinished, the user
# then stayed silent, so answer what we have — the model decides whether it
# is answerable or needs encouragement to continue. Dedicated prompt: with
# a bolted-on suffix the model would sometimes answer as if spoken complete.
FLUSH_PROMPT = (
    "The user paused mid-thought and stayed quiet. If their words feel "
    "unfinished, write one short, warm sentence encouraging them to "
    "continue; otherwise respond to them in 1-4 short sentences, spoken "
    "aloud.{camera}"
)

# A sense of elapsed time. The model can't perceive silence between
# turns, so when a real gap precedes an utterance the turn instruction
# says so — in the per-turn tail (never the system prompt: that's the
# prefix-cache prefix), and only on audio turns (a text turn's
# instruction IS the user's typed words). Coarse buckets on purpose:
# exact numbers read as data the model parrots back awkwardly.
try:
    TIME_NOTE_MIN_S = float(os.environ.get("TIME_NOTE_MIN_S", "120"))
except ValueError:
    print("TIME_NOTE_MIN_S is not a number — using 120s")
    TIME_NOTE_MIN_S = 120.0
TIME_NOTE = " (Time note: {phrase} of quiet passed before this message.)"

# The model has no clock; the start time anchors rough what-time-is-it
# awareness. Formatted ONCE at connect — never re-formatted mid-session.
SESSION_CLOCK = " This conversation started on {clock}."


def elapsed_phrase(seconds: float) -> str:
    """'a few seconds' / 'about 20 seconds' / 'about a minute' /
    'about N minutes' / 'about an hour' / 'about N hours'."""
    if seconds < 10:
        return "a few seconds"
    if seconds < 45:
        return f"about {round(seconds / 10) * 10} seconds"
    if seconds < 90:
        return "about a minute"
    if seconds < 55 * 60:
        return f"about {round(seconds / 60)} minutes"
    if seconds < 90 * 60:
        return "about an hour"
    return f"about {round(seconds / 3600)} hours"


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content
                     if isinstance(p, dict) and p.get("type") == "text")


# Any run of ECHO_WORDS consecutive words of the last reply heard again by
# the mic is OUR TTS playing back, not the user: the browser's raised VAD
# threshold catches most of it, but a late catch — or loud speakers —
# lands as a transcript. Genuine speech doesn't reproduce an 8-word run
# verbatim (the old on-device build drew the line at 5 words for prompt
# echoes; the reply itself repeats prompt-ish phrases far more often).
ECHO_WORDS = 8


def _norm_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def looks_like_echo(transcript: str, history: list) -> bool:
    last = ""
    for m in reversed(history):
        if m.get("role") == "assistant":
            last = _message_text(m.get("content"))
            break
    if not last:
        return False
    t_words = _norm_words(transcript)
    if len(t_words) < ECHO_WORDS:
        return False
    hay = " " + " ".join(_norm_words(last)) + " "
    return any(" " + " ".join(t_words[i:i + ECHO_WORDS]) + " " in hay
               for i in range(len(t_words) - ECHO_WORDS + 1))


# Rotate history before the token budget fills. Rough estimates are fine
# here — the guard just needs to fire before the context and the bill
# get silly. Cloud models have very large contexts; the rotation limit
# is a cost/brevity dial, not a hard ceiling.
HISTORY_ROTATE_TOKENS = int(os.environ.get("HISTORY_ROTATE_TOKENS", "24000"))
CONTEXT_HEADROOM = 2000


def rotate_history(history: list) -> list:
    """Drop the oldest quarter of messages, keeping the system prompt. The
    kept slice must start on a user message: dropping a user turn while
    keeping its reply would leave an orphaned assistant message about
    words that no longer exist. A history with at most one exchange
    returns unchanged (nothing droppable — and slicing a bare [system]
    would duplicate the system prompt)."""
    if len(history) <= 3:
        return history
    keep = 1 + max(2, 3 * (len(history) - 1) // 4)
    while keep > 3 and history[-(keep - 1)].get("role") != "user":
        keep -= 1
    return [history[0]] + history[-(keep - 1):]

tts_backend = None
detector = None  # smart-turn end-of-turn classifier

# Reasoner calls block for up to REASONER_TIMEOUT — keep them off the
# default executor, which serves the latency-critical path (LLM streaming,
# TTS, the turn classifier).
REASONER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reasoner")


def load_models():
    global tts_backend, detector
    # No local LLM: there is nothing to spawn. A missing key surfaces on
    # the first request; the polite check is here so the operator sees
    # it at startup instead.
    if not cloud.enabled():
        print("WARNING: OPENAI_API_KEY is not set — voice turns will fail "
              "until it is (see .env.example).")
    from parlor.turn_detector import TurnDetector
    detector = TurnDetector()
    tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "web" / "static"), name="static")


@app.get("/")
async def root():
    html = (Path(__file__).parent / "web" / "index.html").read_text()
    return HTMLResponse(content=html.replace("{{model}}", cloud.model_label()))


@app.get("/personas")
async def personas_catalog():
    """The curated personality catalog: key/label/description per preset.
    Voice ids, style instructions and persona prompts live server-side —
    the client never ships a prompt."""
    return JSONResponse([
        {"key": p.key, "label": p.label, "description": p.description}
        for p in personas.PRESETS
    ])


def turn_instruction(msg: dict, has_image: bool, has_transcript: bool) -> str:
    if has_transcript:
        camera = CAMERA_CLAUSE if has_image else ""
        prompt = FLUSH_PROMPT if msg.get("type") == "flush" else RESPOND_PROMPT
        return prompt.format(camera=camera)
    if has_image:
        return "The user is showing you their camera. Describe what you see."
    return msg.get("text", "Hello!")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    delegation = reasoner.enabled()
    clock = time.strftime("%A at %I:%M %p").replace(" 0", " ")
    persona = personas.get("neutral")

    def system_prompt(p: personas.Persona) -> str:
        text = (SYSTEM_PROMPT + SESSION_CLOCK.format(clock=clock)
                + CAPABILITY_NOTE + (RESEARCH_NOTE if delegation else ""))
        if p.system:
            text += " " + p.system
        return text

    history: list = [{"role": "system", "content": system_prompt(persona)}]
    mode = MODES["conversation"]

    interrupted = asyncio.Event()
    active = {"stream": None}
    msg_queue = asyncio.Queue()

    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    stream = active.get("stream")
                    if stream:
                        stream.cancel()  # actually aborts generation
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except DISCONNECT_ERRORS:
            pass
        finally:
            # Always unblock the main loop, even on unexpected errors.
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    frame_image: str | None = None   # camera frame held for the current utterance
    speech_chunks: list[str] = []    # streamed-in speech of the current utterance
    held_audio: list[str] = []       # incomplete-turn segments awaiting continuation

    def remember(user_msg: dict, raw_text: str, no_speech: bool) -> None:
        """Store a finished turn verbatim. Some providers reuse a prefix
        cache when consecutive requests share a leading context, and
        byte-stable messages keep that hit alive. Two kinds of turn are
        never stored, because a degenerate message poisons every later
        request: one the model produced nothing for, and a no_words turn
        (the STT heard nothing, or heard our own reply playing back — no
        user words stand behind it; one stored echo loop came back as
        invented or copied user words on every turn after it)."""
        if not raw_text.strip() or no_speech:
            return
        history.append(user_msg)
        history.append({"role": "assistant", "content": raw_text})

    def remember_user_only(user_msg: dict) -> None:
        """A listen-mode content turn: the user's words are kept —
        thinking out loud is context the later exit question builds on —
        but there is no assistant reply to pair with them (the turn ends
        silent BY DESIGN). Consecutive user messages are fine for cloud
        chat APIs."""
        history.append(user_msg)

    delegation_ids = itertools.count(1)
    delegation_tasks: set[asyncio.Task] = set()
    ready_events: list[dict] = []  # rang/finished while the floor was busy
    timer_ids = itertools.count(1)
    pending_timers: dict[int, asyncio.Task] = {}  # id → sleeping ring task
    playing_since = {"t": 0.0}  # last spoken reply is (probably) still playing
    # When the line was last live — an exchange ending, or the user's
    # speech streaming in (so a long monologue doesn't read as a long
    # silence). The gap since it is the "quiet" the time note reports.
    last_activity = {"t": time.time()}
    prompt_tokens = {"last": 0}  # real context size, from the usage chunk

    def floor_busy() -> bool:
        """The floor is held by the user (audio held for a continuation, a
        mid-utterance chunk stream, a just-fired barge-in) or by our own
        voice: a spoken reply counts as playing until the client's 'ready'
        says playback ended — a delivery starting mid-playback would cut
        the reply off mid-sentence. The 30s staleness escape means a lost
        'ready' can only delay a result, never strand it."""
        playing = playing_since["t"] and time.time() - playing_since["t"] < 30
        return bool(held_audio or speech_chunks or interrupted.is_set() or playing)

    def drain_ready() -> None:
        """Requeue one finished background event whenever the floor frees
        up — called at every point that releases it, because msg_queue
        only wakes for client traffic and a result must not wait for one.
        Timers ring in every mode (they are user-requested alarms) while
        research results wait out translation/listening — an English
        answer must not barge into an interpreting or quiet-listening
        session — so this scans instead of popping the head: a parked
        answer must not block a timer ringing behind it."""
        if floor_busy():
            return
        for i, item in enumerate(ready_events):
            if item["type"] == "timer_done" or mode.allows_delegation:
                msg_queue.put_nowait(ready_events.pop(i))
                return

    async def switch_mode(name: str) -> bool:
        """Enter a mode by name (model tag or UI escape hatch). Unknown and
        already-current names are no-ops, so callers can pass raw values.
        Returns whether a switch happened."""
        nonlocal mode, frame_image, speech_chunks, held_audio
        name = name.strip().lower()
        if name == mode.name:
            return True  # already there — the tag "fired" in every sense
        if name not in MODES:
            # The model confirmed something out loud and then emitted junk —
            # make that diagnosable instead of a silent nothing.
            print(f"Ignoring unknown mode {name!r}")
            return False
        mode = MODES[name]
        # Start the new mode clean: a switch through the UI escape hatch can
        # fire with audio held under the OLD mode's gating, and translate
        # mode never resolves holds — stale held state would keep
        # floor_busy() true and block deliveries for the whole session.
        frame_image = None
        speech_chunks = []
        held_audio = []
        print(f"Mode → {mode.name}")
        await send_json(ws, {"type": "mode_changed", "mode": mode.name})
        drain_ready()  # leaving translation frees deferred deliveries
        return True

    async def run_delegation(task_id: int, task: str) -> None:
        """Background reasoner call; its outcome re-enters the main loop
        through msg_queue so delivery is serialized with real turns."""
        t0 = time.time()
        try:
            answer = await asyncio.get_event_loop().run_in_executor(
                REASONER_POOL, reasoner.ask, task)
            # Clamp a verbose answer: it is interpolated into the delivery
            # prompt and stored in history under a small LLAMA_CTX.
            if len(answer) > 1500:
                answer = answer[:1500].rsplit(". ", 1)[0] + "."
            outcome = {"ok": True, "answer": answer}
        except Exception as e:
            print(f"Delegation #{task_id} failed: {e}")
            outcome = {"ok": False, "answer": ""}
        await msg_queue.put({"type": "delegation_done", "id": task_id,
                             "task": task, "took_s": round(time.time() - t0, 1),
                             **outcome})

    async def spawn_delegation(task: str) -> None:
        """Hand one research task to the background reasoner. Guards stay
        from the tag era: a fragment task gives the reasoner nothing to
        work with, and a runaway decision must not queue an HTTP call per
        imagined task."""
        if not delegation or not mode.allows_delegation:
            return
        if len(task.split()) < 3:
            print(f"Delegation skipped (fragment): {task!r}")
            return
        if len(delegation_tasks) >= MAX_PENDING_DELEGATIONS:
            print(f"Delegation skipped (cap {MAX_PENDING_DELEGATIONS}): {task!r}")
            return
        task_id = next(delegation_ids)
        print(f"Delegation #{task_id}: {task!r}")
        await send_json(ws, {"type": "delegation_started",
                             "id": task_id, "task": task})
        t = asyncio.create_task(run_delegation(task_id, task))
        delegation_tasks.add(t)
        t.add_done_callback(delegation_tasks.discard)

    async def proactive_turn(prompt: str, fallback: str) -> None:
        """A server-initiated turn — a delegation delivery or a timer ring:
        the model phrases it, TTS speaks it, and history keeps it like any
        real turn. Nothing server-initiated ever acts: the action decider
        only judges real user turns."""
        user_msg = {"role": "user", "content": [text_part(prompt)]}
        raw_text, pt, _, spoke = await run_turn(
            ws, history + [user_msg], interrupted, active, tts_backend,
            expect_transcript=False, tts_voice=persona.voice or mode.tts_voice,
            tts_instructions=persona.tts_instructions,
            proactive=True, fallback=fallback, instruction=prompt)
        if pt:
            prompt_tokens["last"] = pt
        if spoke:
            playing_since["t"] = time.time()  # this reply is now playing
        remember(user_msg, raw_text, no_speech=False)
        last_activity["t"] = time.time()
        drain_ready()         # more background events may already be waiting

    async def deliver_delegation(done: dict) -> None:
        """A finished research task, spoken by the voice model. Failures are
        delivered too — a delegation must never end in silence, which is
        also why the fallback is the reasoner's own answer: if the model's
        relay yields nothing speakable (##-markup relapse, transcript-only
        reply), TTS speaks the answer directly."""
        interrupted.clear()
        await send_json(ws, {"type": "delegation_resolved",
                             "id": done["id"], "ok": done["ok"]})
        prompt = (DELIVER_PROMPT if done["ok"] else DELIVER_FAILED_PROMPT).format(
            task=done["task"], answer=done["answer"],
            took=elapsed_phrase(done.get("took_s", 0.0)))
        await proactive_turn(
            prompt, done["answer"] if done["ok"] else DELIVER_FALLBACK)

    async def run_timer(timer_id: int, label: str, seconds: float) -> None:
        """Sleep out the countdown, then re-enter the main loop through
        msg_queue so the ring is serialized with real turns. Cancellation
        during the sleep enqueues nothing."""
        await asyncio.sleep(seconds)
        await msg_queue.put({"type": "timer_done", "id": timer_id,
                             "label": label, "seconds": seconds})

    async def spawn_timer(seconds: int, label: str) -> None:
        """Schedule one countdown from the action decision."""
        if seconds > MAX_TIMER_S:
            print(f"Timer skipped (over {MAX_TIMER_S // 3600}h cap): {seconds}s")
            return
        if len(pending_timers) >= MAX_PENDING_TIMERS:
            print(f"Timer skipped (cap {MAX_PENDING_TIMERS}): {label!r}")
            return
        timer_id = next(timer_ids)
        label = label or "timer"
        print(f"Timer #{timer_id}: {seconds}s {label!r}")
        pending_timers[timer_id] = asyncio.create_task(
            run_timer(timer_id, label, seconds))
        await send_json(ws, {"type": "timer_started", "id": timer_id,
                             "label": label, "seconds": seconds})

    async def apply_decision(decision) -> None:
        """Fire what the decider found. Timers/research fire when the
        turn is conversational at EITHER end of a mode transition: a
        plain conversation turn, or a listen/translate exit whose same
        breath asks for more ("okay, I'm done — set a twenty-minute
        timer": the exit turn confirms it aloud, so dropping it would be
        a spoken promise with no action — review finding). A mention
        mid-translate/listen without an exit stays content. Ordering
        keeps spawn_delegation's allows_delegation gate live: entries
        spawn before the switch, exits switch first."""
        if mode.name == "conversation":
            if decision.timer:
                await spawn_timer(*decision.timer)
            if decision.research:
                await spawn_delegation(decision.research)
            if decision.mode:
                await switch_mode(decision.mode)
            return
        if decision.mode:
            await switch_mode(decision.mode)
        if mode.name == "conversation":  # exited just now
            if decision.timer:
                await spawn_timer(*decision.timer)
            if decision.research:
                await spawn_delegation(decision.research)

    async def deliver_timer(done: dict) -> None:
        """The ring. Its label and duration ride the event, not the pending
        entry, so popping it is only about not ringing twice."""
        if pending_timers.pop(done["id"], None) is None:
            return  # cancelled after it fired
        interrupted.clear()
        await send_json(ws, {"type": "timer_resolved", "id": done["id"]})
        prompt = TIMER_PROMPT.format(label=done["label"],
                                     duration=duration_phrase(done["seconds"]))
        await proactive_turn(prompt, TIMER_FALLBACK.format(label=done["label"]))

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            # Rotate history before the token budget fills: keep the system
            # prompt and the most recent exchanges. Cloud contexts are huge;
            # this fires early enough to keep turns (and their cost) small.
            # Double headroom because the incoming turn isn't counted yet;
            # drop a quarter (not half) so a rotation is barely noticeable.
            est = estimate_tokens(history)
            used = max(est, prompt_tokens["last"])
            if used > HISTORY_ROTATE_TOKENS - 2 * CONTEXT_HEADROOM:
                rotated = rotate_history(history)
                if len(rotated) < len(history):
                    print(f"Context near limit (est {est}, real {prompt_tokens['last']}) "
                          f"— dropping {len(history) - len(rotated)} oldest messages")
                    history = rotated
                    prompt_tokens["last"] = 0  # stale after a rotation

            if msg.get("type") == "ready":
                # The client returned to idle listening: playback finished,
                # or a false barge-in never became an utterance. A sticky
                # interrupted flag must not strand queued deliveries — and
                # the client cleared its own frame-discard flag before
                # sending this, so delivering now is safe.
                playing_since["t"] = 0.0
                last_activity["t"] = time.time()
                interrupted.clear()
                drain_ready()
                continue

            if msg.get("type") == "set_mode":
                # The UI escape hatch (mode chip's stop button): a switch
                # that must work even when the model mistranslates the
                # spoken exit command.
                await switch_mode(str(msg.get("mode", "")))
                continue

            if msg.get("type") == "set_personality":
                # Palette switch: voice + delivery + chat persona ride the
                # NEXT request wholesale — nothing in the pipeline cares
                # which model "was" speaking, so the switch is instant and
                # keeps the conversation (the system message is rewritten
                # in place: the old persona clause leaves future turns).
                p = personas.get(str(msg.get("key", "")))
                if p.key != persona.key:
                    persona = p
                    history[0] = {"role": "system",
                                  "content": system_prompt(persona)}
                    print(f"Persona → {persona.label} ({persona.voice or 'default voice'})")
                await send_json(ws, {"type": "personality_changed",
                                     "key": persona.key,
                                     "label": persona.label})
                continue

            if msg.get("type") == "voice_preview":
                # One short sample line, synthesized OUTSIDE any turn (no
                # history entry, no turn frames — distinct preview_* frames
                # keep the client's transcript/playback state untouched).
                p = personas.get(str(msg.get("key", "")))
                try:
                    pcm = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: tts_backend.generate(
                            p.sample_line,
                            voice=p.voice or mode.tts_voice,
                            instructions=p.tts_instructions))
                    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                    await send_json(ws, {"type": "voice_preview_start",
                                         "sample_rate": tts_backend.sample_rate})
                    await send_json(ws, {"type": "voice_preview_chunk",
                                         "audio": base64.b64encode(
                                             pcm_int16.tobytes()).decode()})
                    await send_json(ws, {"type": "voice_preview_end"})
                except DISCONNECT_ERRORS:
                    raise
                except Exception:
                    traceback.print_exc()  # preview failure keeps the session
                    await send_json(ws, {"type": "voice_preview_end",
                                         "failed": True})
                continue

            if msg.get("type") == "delegation_done":
                if not mode.allows_delegation:
                    # Parked until the user exits translation/listening —
                    # tell the client so its chip stops implying work in
                    # progress.
                    await send_json(ws, {"type": "delegation_parked",
                                         "id": msg["id"]})
                    ready_events.append(msg)
                    continue
                if floor_busy():
                    ready_events.append(msg)  # deliver at the next idle moment
                    continue
                try:
                    await deliver_delegation(msg)
                except DISCONNECT_ERRORS:
                    raise
                except Exception:
                    traceback.print_exc()  # keep the session alive
                    if not msg.get("redelivered"):
                        msg["redelivered"] = True  # one more try at idle
                        ready_events.append(msg)
                    await release_client(ws)
                continue

            if msg.get("type") == "timer_done":
                if msg["id"] not in pending_timers:
                    continue  # cancelled after it fired
                if floor_busy():
                    ready_events.append(msg)  # ring at the next idle moment
                    continue
                try:
                    await deliver_timer(msg)
                except DISCONNECT_ERRORS:
                    raise
                except Exception:
                    traceback.print_exc()  # keep the session alive
                    pending_timers.pop(msg["id"], None)  # no re-ring
                    await release_client(ws)
                continue

            if msg.get("type") == "cancel_timer":
                # The chip's ✕ — like set_mode, a UI action that must work
                # regardless of what the model believes.
                task = pending_timers.pop(msg.get("id"), None)
                if task:
                    task.cancel()
                    ready_events[:] = [e for e in ready_events
                                       if not (e["type"] == "timer_done"
                                               and e["id"] == msg["id"])]
                    print(f"Timer #{msg['id']} cancelled")
                    await send_json(ws, {"type": "timer_resolved",
                                         "id": msg["id"], "cancelled": True})
                continue

            if msg.get("type") == "frame":
                if msg.get("image") and mode.wants_camera:
                    frame_image = msg["image"]
                    speech_chunks = []  # a new frame opens a new utterance
                continue

            if msg.get("type") == "speech_chunk":
                if msg.get("seq") == 0:
                    speech_chunks = []
                if valid_audio(msg.get("audio")):
                    speech_chunks.append(msg["audio"])
                    last_activity["t"] = time.time()
                continue

            interrupted.clear()
            audio_b64s = held_audio + (speech_chunks if msg.get("chunked") else [])
            speech_chunks = []
            if valid_audio(msg.get("audio")):
                audio_b64s.append(msg["audio"])
            image = (msg.get("image") or frame_image) if mode.wants_camera else None
            has_audio = bool(audio_b64s)
            is_flush = msg.get("type") == "flush"

            if not has_audio and not image and not msg.get("text"):
                # Mic glitch (or a flush with nothing held) produced no usable media.
                await release_client(ws)
                drain_ready()  # the floor is provably free here
                continue

            # From here on any failure (malformed WAV included — valid_audio
            # only checks length) must release the client, not kill the
            # session loop.
            p_complete = None  # smart-turn probability, surfaced in the UI
            try:
                # The audio classifier judges completeness before the LLM is
                # involved at all. Incomplete → hold the segments (they ride
                # in the next turn's content) and wait. A flush turn skips
                # the check: the client waited out the hold and now wants an
                # answer to whatever we have. Translate mode skips it
                # entirely — an interpreter renders on a short silence, it
                # does not wait out thinking pauses.
                if has_audio and not is_flush and mode.uses_smart_turn:
                    pcm = np.concatenate([wav_to_float32(b) for b in audio_b64s])
                    t0 = time.time()
                    complete, prob = await asyncio.get_event_loop().run_in_executor(
                        None, detector.predict, pcm)
                    decision_s = round(time.time() - t0, 3)
                    p_complete = round(prob, 2)
                    if not complete and not interrupted.is_set():
                        held_audio = audio_b64s
                        # Release the client BEFORE the transcription stall:
                        # until turn_incomplete arrives it can't capture a
                        # resumed utterance, and the flush timer's
                        # live-speech guard can't see the user talking.
                        await send_json(ws, {
                            "type": "turn_incomplete",
                            "decision_s": decision_s, "p_complete": p_complete,
                        })
                        continue

                # Merge the streamed segments into one WAV and hand it to
                # the STT. Blocking; the turn is stalled on it anyway, so
                # it runs in the default executor.
                if audio_b64s:
                    t0 = time.time()
                    wav = merge_wavs(audio_b64s)
                    transcript = ""
                    if wav:
                        try:
                            transcript = (await asyncio.get_event_loop()
                                          .run_in_executor(
                                              None, cloud.transcribe, wav)
                                          ).strip()
                        except Exception as e:
                            print(f"STT failed: {e}")
                    stt_s = round(time.time() - t0, 3)
                    echo = bool(transcript) and looks_like_echo(transcript, history)
                    if not transcript or echo:
                        if echo:
                            print(f"STT heard our own reply ({stt_s}s) — "
                                  f"echo dropped: {transcript!r}")
                        else:
                            print(f"STT heard no words ({stt_s}s)")
                    else:
                        timings = {"stt_s": stt_s, "p_complete": p_complete}
                        # The transcript LEADS: the client shows what was
                        # heard while the reply still streams.
                        await send_json(ws, {"type": "transcript",
                                             "transcription": transcript,
                                             "p_complete": p_complete})
                else:
                    transcript = ""
                    echo = False
                content = ([image_part(image)] if image else []) + \
                          ([text_part(transcript)] if transcript else [])
                frame_image = None
                held_audio = []

                # When the decision runs is a latency choice. Conversation
                # and translate turns speak FIRST and decide after, hidden
                # under TTS playback — an interpreter must start rendering
                # immediately, so a translated exit command gets rendered
                # once and THEN the switch lands (the vanishing chip is
                # the confirmation). Listen turns decide BEFORE: they play
                # nothing (the ~1s only delays the on-screen transcript),
                # and an exit ("what do you think?") must be ANSWERED, not
                # silently transcribed — the decision picks the path.
                # Either way a decision FIRES only after the turn, gated
                # on no_words/interrupt like always.
                decision = actions.NONE
                pre_decided = (mode.name == "listen" and has_audio
                               and transcript)
                if pre_decided:
                    decision = await asyncio.get_event_loop().run_in_executor(
                        None, actions.decide_before, history,
                        [text_part(transcript)], mode.name)

                if has_audio and (not transcript or echo):
                    # No user words stand behind this turn (a breath once
                    # "asked" to translate everything — measured live in
                    # the on-device build). Nothing may act, nothing is
                    # stored; a silent turn would leave the user hanging,
                    # so speak the mode's fallback — except in listen
                    # mode, where silence IS the contract.
                    if mode.speaks_fallback and not interrupted.is_set():
                        line = FLUSH_FALLBACK if is_flush else AUDIO_FALLBACK
                        spoke = await speak_only(ws, line, tts_backend,
                                                 tts_voice=persona.voice or mode.tts_voice,
                                                 tts_instructions=persona.tts_instructions)
                        if spoke:
                            playing_since["t"] = time.time()
                    else:
                        await release_client(ws)
                    last_activity["t"] = time.time()
                    continue

                if pre_decided and not decision.mode:
                    # A listen content turn needs no reply and no LLM
                    # call: the exact words were transcribed above.
                    await send_json(ws, {"type": "turn_final",
                                         "transcription": None,
                                         "proactive": False,
                                         "timings": timings,
                                         "p_complete": p_complete,
                                         "spoke": False})
                    remember_user_only({"role": "user",
                                        "content": [text_part(transcript)]})
                    last_activity["t"] = time.time()
                    continue

                if pre_decided and decision.mode:
                    # Addressed to the assistant (an exit command): answer
                    # it as a normal conversation turn, not as content.
                    instruction = RESPOND_PROMPT.format(camera="")
                elif mode.name in MODE_PROMPTS and has_audio:
                    instruction = MODE_PROMPTS[mode.name]
                else:
                    instruction = turn_instruction(msg, bool(image),
                                                   has_audio)
                # Elapsed quiet is a per-mode policy — see modes.py.
                gap = time.time() - last_activity["t"]
                if has_audio and mode.wants_time_note and gap >= TIME_NOTE_MIN_S:
                    phrase = elapsed_phrase(gap)
                    instruction += TIME_NOTE.format(phrase=phrase)
                    print(f"Time note: {phrase}")
                if has_audio and decision.mode:
                    fallback = AUDIO_FALLBACK  # an exit turn must speak
                elif has_audio and not mode.speaks_fallback:
                    fallback = None       # a listen turn is MEANT to end silent
                elif is_flush:
                    fallback = FLUSH_FALLBACK
                elif has_audio and transcript:
                    fallback = AUDIO_FALLBACK
                else:
                    fallback = None
                user_msg = {"role": "user", "content": content + [text_part(instruction)]}
                raw_text, pt, no_speech, spoke = await run_turn(
                    ws, history + [user_msg], interrupted, active,
                    tts_backend, expect_transcript=False,
                    p_complete=p_complete,
                    tts_voice=persona.voice or mode.tts_voice,
                    tts_instructions=persona.tts_instructions,
                    fallback=fallback, instruction=instruction)
                if pt:
                    prompt_tokens["last"] = pt
                if spoke:
                    playing_since["t"] = time.time()  # reply now playing client-side
                if not pre_decided and not no_speech and not interrupted.is_set() \
                        and (has_audio or msg.get("text")):
                    decision = await asyncio.get_event_loop().run_in_executor(
                        None, actions.decide_after,
                        history + [user_msg,
                                   {"role": "assistant", "content": raw_text}],
                        mode.name)
                if no_speech or interrupted.is_set():
                    # No user words stand behind this turn, or the user
                    # cut it off: nothing may act.
                    decision = actions.NONE
                await apply_decision(decision)
                remember(user_msg, raw_text, no_speech)
                last_activity["t"] = time.time()
            except DISCONNECT_ERRORS:
                raise
            except Exception:
                traceback.print_exc()  # keep the session alive
                await release_client(ws)

            # A result that finished while the floor was busy delivers now.
            drain_ready()
    except DISCONNECT_ERRORS:
        print("Client disconnected")
    finally:
        recv_task.cancel()
        for t in delegation_tasks:
            t.cancel()
        for t in pending_timers.values():
            t.cancel()


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    # localhost, not 0.0.0.0: browsers treat http://localhost as a secure
    # context but not http://0.0.0.0, and without one getUserMedia (mic,
    # camera) doesn't exist.
    uvicorn.run(app, host="localhost", port=port)


if __name__ == "__main__":
    main()
