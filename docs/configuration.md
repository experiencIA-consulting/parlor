# Configuration

Every setting is an environment variable, set in your shell or in a `.env`
file at the repo root. `.env` is loaded at startup (a real shell variable
always wins over `.env`); changes need a server restart, because most
config is read once at import.

## Cloud LLM (conversation)

| Variable         | Default                        | Description |
| ---------------- | ------------------------------ | ----------- |
| `OPENAI_API_KEY` | (required)                     | API key for the endpoint |
| `OPENAI_BASE_URL`| `https://api.openai.com/v1`    | Any OpenAI-compatible chat endpoint (OpenRouter: `https://openrouter.ai/api/v1`) |
| `OPENAI_MODEL`   | `gpt-4o-mini`                  | Conversation model. On OpenRouter use `vendor/model` (e.g. `openai/gpt-4o-mini`). Must accept images for camera support. |

The module mirrors the OpenAI chat-completions protocol: streaming, usage
in the final chunk, and structured outputs for the action decider
(robots that reject `response_format` are retried once with prose JSON).

## Speech-to-text (OpenAI)

| Variable              | Default                  | Description |
| --------------------- | ------------------------ | ----------- |
| `OPENAI_STT_MODEL`    | `gpt-4o-mini-transcribe` | Any model the `/v1/audio/transcriptions` endpoint accepts (`whisper-1`, ...) |
| `OPENAI_STT_LANGUAGE` | (auto-detect)            | Force a language (ISO 639-1, e.g. `es`) |

## Text-to-speech (OpenAI by default)

| Variable           | Default              | Description |
| ------------------ | -------------------- | ----------- |
| `OPENAI_TTS_MODEL` | `gpt-4o-mini-tts`    | Model served by `/v1/audio/speech` |
| `OPENAI_TTS_VOICE` | `alloy`              | Voice id (alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer, ...) |
| `TTS_BACKEND`      | `openai`             | `kokoro` switches back to the local Kokoro stack (Linux/Apple Silicon installs only) |

## Server

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `PORT`   | `8000`  | Web server port (serves on `localhost` only — browsers require a secure context for mic/camera) |
| `TEMPERATURE` | `0.7` | Sampling temperature for speech (0 = deterministic; the action decider always runs at 0) |
| `HISTORY_ROTATE_TOKENS` | `24000` | Estimated context size that triggers dropping the oldest quarter of the history — a cost dial, not the model's real ceiling |
| `TIME_NOTE_MIN_S` | `120` | Seconds of quiet before the next turn tells the model how long the silence was |

## Background research

Delegation reuses the main `OPENAI_API_KEY` when `REASONER_*` is unset: on
api.openai.com that runs `gpt-4o-search-preview` (native web search), so
research works with a single key. A dedicated OpenRouter endpoint gives
you any frontier model plus the `:online` web-search mechanism.

| Variable              | Default                                     | Description |
| --------------------- | ------------------------------------------- | ----------- |
| `REASONER_API_KEY`    | (falls back to the chat key)                 | API key for the endpoint |
| `REASONER_BASE_URL`   | (falls back to `OPENAI_BASE_URL`)            | Any OpenAI-compatible chat endpoint |
| `REASONER_MODEL`      | (falls back as described above)              | Model the endpoint should run |
| `REASONER_WEB_SEARCH` | `1`                                          | On OpenRouter, append `:online` for provider-side web search |
| `REASONER_TIMEOUT`    | `90`                                         | Seconds before a background task fails (and is delivered as a spoken apology) |
