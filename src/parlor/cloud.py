"""OpenAI-compatible cloud backend: replaces the on-device llama.cpp
stack with hosted models — chat/vision (chat completions), speech-to-text
(audio transcriptions) and, together with tts.py, text-to-speech.

Any OpenAI-compatible chat endpoint works: api.openai.com directly, or
OpenRouter (set OPENAI_BASE_URL=https://openrouter.ai/api/v1 and an
OpenRouter model id). Speech (STT/TTS) still goes through the OpenAI
API — OpenRouter doesn't offer it.

The module mirrors the llama.py interface that the rest of the server
already speaks (chat_blocking, a streaming ChatStream with cancel(),
prompt_tokens): the turn loop, the action decider and the pipeline need
no protocol changes, only the media handling differs (history is plain
text + images; the utterance is transcribed before the LLM sees it).
"""

import json
import os
import re
import threading
import urllib.request
import urllib.error

from dotenv import load_dotenv

load_dotenv()  # config below is read at import time — .env must apply first

BASE_URL = os.environ.get("OPENAI_BASE_URL",
                          "https://api.openai.com/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# The conversation model. On api.openai.com a bare model id ("gpt-4o-mini");
# on OpenRouter "vendor/model" ("openai/gpt-4o-mini"). It must accept
# images for camera support.
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

STT_MODEL = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
STT_LANGUAGE = os.environ.get("OPENAI_STT_LANGUAGE", "")

try:
    TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
except ValueError:
    print("TEMPERATURE is not a number — using 0.7")
    TEMPERATURE = 0.7

_TIMEOUT = float(os.environ.get("CLOUD_TIMEOUT", "300"))


def enabled() -> bool:
    return bool(API_KEY)


def model_label() -> str:
    return MODEL


# OpenAI's reasoning models reject max_tokens (they want
# max_completion_tokens); every other OpenAI-compatible endpoint takes
# max_tokens. Only the field name varies — same reasoning as reasoner.py.
def _token_field() -> str:
    if "api.openai.com" in BASE_URL and re.match(
            r"(?:o\d|gpt-5)", MODEL.split("/")[-1]):
        return "max_completion_tokens"
    return "max_tokens"


def _headers() -> dict:
    return {"Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"}


def _chat_body(messages: list, max_tokens: int, stream: bool,
               temperature: float | None = None,
               json_schema: dict | None = None) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        _token_field(): max_tokens,
        "temperature": TEMPERATURE if temperature is None else temperature,
        "stream": stream,
    }
    if stream:
        # The final chunk carries usage — the REAL prompt size, which
        # drives history rotation (estimates drift).
        body["stream_options"] = {"include_usage": True}
    if json_schema:
        # Structured outputs: the outcome is structurally guaranteed to
        # parse (used by the action head). Some endpoints/models reject
        # the field — chat_blocking retries once without it.
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "action_decision",
                "strict": True,
                "schema": {**json_schema, "additionalProperties": False},
            },
        }
    return body


def _post_chat(body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        BASE_URL + "/chat/completions", data=json.dumps(body).encode(),
        headers=_headers(), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _strip_schema(body: dict) -> dict:
    body = dict(body)
    body.pop("response_format", None)
    return body


def chat_blocking(messages: list, max_tokens: int,
                  temperature: float | None = None,
                  json_schema: dict | None = None) -> str:
    """Non-streaming request; returns the message content."""
    body = _chat_body(messages, max_tokens, stream=False,
                      temperature=temperature, json_schema=json_schema)
    try:
        data = _post_chat(body, _TIMEOUT)
    except urllib.error.HTTPError as e:
        if json_schema and e.code == 400:
            # Endpoint or model without structured-output support —
            # retry asking for JSON in prose; the caller parses leniently.
            data = _post_chat(_strip_schema(body), _TIMEOUT)
        else:
            raise RuntimeError(f"chat endpoint HTTP {e.code}: "
                               f"{e.read()[:300]!r}") from e
    if "error" in data:
        raise RuntimeError(f"chat endpoint: {data['error']}")
    return data["choices"][0]["message"].get("content") or ""


def parse_json(raw: str) -> dict:
    """Tolerant JSON parse for a text-only fallback of the action head:
    the first {...} block wins; prose around it is ignored."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


class ChatStream:
    """Streaming chat request, driven from an executor thread. cancel() is
    thread-safe and actually aborts generation server-side (the connection
    close tears the socket down)."""

    def __init__(self, messages: list, max_tokens: int):
        self.body = _chat_body(messages, max_tokens, stream=True)
        self._client: urllib.request.Request | None = None
        self.cancelled = False
        self.prompt_tokens: int | None = None  # real count, from the usage chunk

    def run(self, on_delta):
        req = urllib.request.Request(
            BASE_URL + "/chat/completions", data=json.dumps(self.body).encode(),
            headers=_headers(), method="POST")
        try:
            self._client = urllib.request.urlopen(req, timeout=_TIMEOUT)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read()[:300].decode("utf-8", "replace")
            except OSError:
                pass
            raise RuntimeError(f"chat endpoint HTTP {e.code}: {detail!r} "
                               f"({BASE_URL}, model={MODEL})") from e
        try:
            while True:
                line = self._client.readline()
                if not line or self.cancelled:
                    break
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage and usage.get("prompt_tokens"):
                    self.prompt_tokens = usage["prompt_tokens"]
                choices = chunk.get("choices") or []
                text = choices[0].get("delta", {}).get("content") \
                    if choices else None
                if text:
                    on_delta(text)
        except Exception:
            # Truncation is normal on abort; a genuinely dead endpoint
            # surfaces on the next request.
            if not self.cancelled:
                raise
        finally:
            self._close()

    def _close(self):
        if self._client:
            try:
                self._client.close()
            except OSError:
                pass
            self._client = None

    def cancel(self):
        self.cancelled = True
        self._close()


# ── speech-to-text ────────────────────────────────────────────────────────

def transcribe(wav_bytes: bytes) -> str:
    """Transcribe a whole 16kHz s16 mono WAV. Blocking; run in an
    executor. Empty string means the model heard no words (silence,
    noise, a cough) — the caller treats that like the local stack's
    no-speech turn."""
    data = {"model": STT_MODEL}
    if STT_LANGUAGE:
        data["language"] = STT_LANGUAGE
    boundary = "----ParlorFormBoundary7MxAi1"
    parts = []
    for name, value in data.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f"name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 "name=\"file\"; filename=\"speech.wav\"\r\n"
                 "Content-Type: audio/wav\r\n\r\n".encode()
                 + wav_bytes + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        BASE_URL + "/audio/transcriptions", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Authorization": f"Bearer {API_KEY}"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            out = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"STT endpoint HTTP {e.code}: "
                           f"{e.read()[:300]!r}") from e
    return (out.get("text") or "").strip()
