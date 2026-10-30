"""Background reasoner: hands a task from the voice model to a frontier
text model on any OpenAI-compatible chat endpoint and returns a short
speakable answer.

REASONER_* config wins when set. Without it, delegation falls back to the
same OpenAI credentials the chat stack uses — on api.openai.com that
means a native web-search model (gpt-4o-search-preview), so research
works with only one key. With a dedicated OpenRouter key, the default
endpoint is OpenRouter, where REASONER_WEB_SEARCH=1 (the default)
appends the ':online' suffix so any model gets provider-side web search.
"""

import json
import os
import urllib.request

import parlor.cloud as cloud

BASE_URL = os.environ.get("REASONER_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("REASONER_API_KEY", "")
MODEL = os.environ.get("REASONER_MODEL", "")
WEB_SEARCH = os.environ.get("REASONER_WEB_SEARCH", "1") == "1"

if not (BASE_URL and API_KEY and MODEL):
    # Fall back to the chat stack's credentials: research rides the same
    # key. OpenAI offers web search natively through the -search-preview
    # models, so delegation keeps working with no second provider. (No
    # key at all → enabled() is False and Parlor never delegates.)
    BASE_URL = BASE_URL or cloud.BASE_URL
    API_KEY = API_KEY or cloud.API_KEY
    MODEL = MODEL or ("gpt-4o-search-preview"
                      if "api.openai.com" in BASE_URL else "openai/gpt-5.6-luna")

if WEB_SEARCH and "openrouter.ai" in BASE_URL and not MODEL.endswith(":online"):
    MODEL += ":online"
try:
    TIMEOUT_S = float(os.environ.get("REASONER_TIMEOUT", "90"))
except ValueError:
    print("REASONER_TIMEOUT is not a number — using 90s")
    TIMEOUT_S = 90.0

# On OpenRouter web search is requested through the model id itself, so it
# has to be baked in here; other endpoints would reject the suffix.
if (os.environ.get("REASONER_WEB_SEARCH", "1") == "1"
        and "openrouter.ai" in BASE_URL and not MODEL.endswith(":online")):
    MODEL += ":online"

# The answer is relayed aloud by the voice model near-verbatim, so it must
# already be speech-shaped when it comes back.
SYSTEM_PROMPT = (
    "You are the background research assistant behind a real-time voice "
    "AI. Answer the task directly, in plain conversational text that will "
    "be read aloud: no markdown, no lists, no headings, no URLs. Lead "
    "with the answer, keep it to 2-6 short sentences, and include the "
    "concrete facts (names, numbers, places) the task asks for."
)


def enabled() -> bool:
    return bool(API_KEY)


def ask(task: str) -> str:
    """Blocking chat-completion call — run it in an executor. Raises on
    transport/HTTP errors and empty answers; the caller turns any failure
    into a spoken apology."""
    # api.openai.com rejects max_tokens for its reasoning models and wants
    # max_completion_tokens; every other OpenAI-compatible endpoint
    # (OpenRouter, llama-server, vLLM) takes max_tokens. Only the field
    # name varies: reasoning models spend hidden thinking tokens from the
    # same budget wherever they run, and the answer stays short because
    # the system prompt asks for a few spoken sentences — the cap just
    # stops essays.
    field = ("max_completion_tokens" if "api.openai.com" in BASE_URL
             else "max_tokens")
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps({
            "model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": task}],
            field: 4096,
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        body = json.load(resp)
    answer = (body["choices"][0]["message"]["content"] or "").strip()
    if not answer:
        raise RuntimeError("reasoner returned an empty answer")
    return answer
