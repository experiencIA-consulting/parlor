"""The action decider: what the user asked the assistant to DO, decided by
a separate structured-JSON request instead of in-band control tags.

Why a second request (benchmarks/archbench.py in the on-device build):
in-band tags scored recall 0.955 with the miss being an
ack-without-action ("I will be quiet", no tag) — a spoken promise the
server never keeps, the worst failure a voice assistant has. The
decoupled head scored recall 1.0, its only error an unwanted-but-
cancellable timer misfire (0.062). Structurally, the head can never leak
into speech (there is nothing to excise), runs at temperature 0 while
speech keeps its sampling temperature, sees the model's own
confirmation as evidence, and does the duration math itself ("twenty
minutes" → 1200) in any language. History stays pure speech — no stored
tag can teach the model a state the server isn't in.

It runs on the same chat endpoint as speech, as one small structured
call per turn: cheap on cloud APIs, and it keeps the guarantee that a
spoken promise is always kept.
"""

from dataclasses import dataclass

from parlor import cloud

# One prompt per session mode: the head must know what "no change" means
# ("conversation" reported while already conversing is state, not an
# action) and, mid-translation/listening, what an exit command sounds
# like. The listen wording carries the clearly-ask-YOU nuance measured in
# the e2e suite: a think-aloud quasi-question ("the thing I wanted to ask
# you about is…") is still thinking, not an exit.
_HEAD_COMMON = (
    "System note (not user audio): you are the action decider. From the "
    "user's last message (the assistant's reply above may help), report "
    "what the user asked the assistant to DO, as JSON. timer_seconds: "
    "the countdown duration in seconds if they asked for a timer or a "
    "timed reminder, else 0 — asking the assistant to be quiet for some "
    "time is a mode request, never a timer. timer_label: a "
    "two-or-three-word label for it, else empty. {mode_clause} "
    "research_task: if they asked to "
    "search, look up, or research something, or asked about anything "
    "current or changing (weather, news, prices, scores, openings, "
    "\"right now\", \"today\"), the task restated to stand alone, else "
    "empty. A duration or topic merely mentioned in passing is NOT a "
    "request: report an action only when the user asked for it."
)
_MODE_CLAUSES = {
    "conversation": (
        "mode: the mode they asked to SWITCH TO — 'translate' (translate "
        "everything they say from now on), 'listen' (any ask for the "
        "assistant to just listen, stay silent or quiet, or stop "
        "responding for a while) — the session is already in normal "
        "conversation, so mode is 'none' unless they asked to change it."),
    "translate": (
        "mode: 'conversation' ONLY if their words are a command to the "
        "assistant to stop translating and go back to normal "
        "conversation (in any language); everything else they say is "
        "content being translated, so mode is 'none'."),
    "listen": (
        "mode: 'conversation' ONLY if their words clearly ask the "
        "assistant to start responding again — like \"okay, I'm done\", "
        "\"what do you think?\", \"stop listening\"; if they are still "
        "thinking out loud, even wondering about something or mentioning "
        "the assistant, mode is 'none'."),
}
HEAD_SCHEMA = {
    "type": "object",
    "properties": {
        "timer_seconds": {"type": "integer"},
        "timer_label": {"type": "string"},
        "mode": {"type": "string",
                 "enum": ["none", "translate", "listen", "conversation"]},
        "research_task": {"type": "string"},
    },
    "required": ["timer_seconds", "timer_label", "mode", "research_task"],
}


@dataclass(frozen=True)
class ActionDecision:
    """What the user asked for, typed. None/empty means 'nothing'."""
    timer: tuple[int, str] | None = None   # (seconds, label)
    mode: str | None = None                # target mode, already ≠ current
    research: str | None = None            # self-contained task

    def any(self) -> bool:
        return bool(self.timer or self.mode or self.research)


NONE = ActionDecision()


def decide_after(messages: list, current_mode: str) -> ActionDecision:
    """Judge a finished conversation turn (blocking — run in an executor).
    `messages` ends with the assistant reply, byte-identical to the turn
    request plus the reply — the whole prefix is already in the slot
    cache, so only the decider prompt pays prefill. The reply is
    evidence: a confirmation like "three minutes, I'll let you know" is
    exactly what the head keys on."""
    return _decide(lambda prompt: messages + [{"role": "user", "content": prompt}],
                   current_mode)


def decide_before(history: list, content: list, current_mode: str) -> ActionDecision:
    """Judge an utterance BEFORE any reply exists (blocking). Used in
    translate/listen, where what the reply IS depends on the decision
    (content to render, or a command to the assistant). The decider
    prompt rides the same user message as the audio — the production
    instruction shape, and it extends the primed prefix instead of
    opening a second user turn (chat templates dislike consecutive
    same-role messages)."""
    def build(prompt: str) -> list:
        text = {"type": "text", "text": prompt}
        return history + [{"role": "user", "content": content + [text]}]
    return _decide(build, current_mode)


def _decide(build, current_mode: str) -> ActionDecision:
    """One structured-JSON head call. Failures return NONE: a lost
    decision is a no-op turn, never an exception in the turn loop."""
    try:
        head_prompt = _HEAD_COMMON.format(mode_clause=_MODE_CLAUSES[current_mode])
        # max_tokens must clear the JSON skeleton (~30 tokens) PLUS a long
        # restated research task — truncated JSON fails json.loads and
        # silently drops an action the reply already promised (review
        # finding). The schema bounds the shape, so the cap is generous.
        raw = cloud.chat_blocking(build(head_prompt), max_tokens=192,
                                  temperature=0.0, json_schema=HEAD_SCHEMA)
        head = cloud.parse_json(raw)
    except Exception as e:
        print(f"Action decider failed: {e}")
        return NONE
    timer = None
    seconds = head.get("timer_seconds") or 0
    if isinstance(seconds, int) and seconds > 0:
        timer = (seconds, str(head.get("timer_label", "")).strip())
    target = head.get("mode")
    # Mid-translate/listen the only sanctioned transition is OUT — the
    # clauses ask only about exits, so any other target is a misread
    # (a phantom listen→translate jump would answer aloud mid-listen).
    allowed = (("translate", "listen", "conversation")
               if current_mode == "conversation" else ("conversation",))
    mode = target if target in allowed and target != current_mode else None
    research = str(head.get("research_task", "")).strip() or None
    decision = ActionDecision(timer=timer, mode=mode, research=research)
    if decision.any():
        print(f"Action decision: {decision}")
    return decision
