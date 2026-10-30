# Parlor

Real-time multimodal AI (voice + vision, features similar to [GPT-Live](https://openai.com/index/introducing-gpt-live/)) on a **cloud LLM** — no local model downloads, no dedicated GPU. This fork talks to **OpenAI / OpenRouter** (any OpenAI-compatible endpoint) instead of running llama.cpp on-device.

https://github.com/user-attachments/assets/0833b0ec-9e48-46dc-9fde-c3ac4a70abf9

> **Research preview.** This is an early experiment. Expect rough edges and bugs.

# Why?

I'm [self-hosting a totally free voice AI](https://www.fikrikarim.com/bule-ai-initial-release/) on my home server to help people learn speaking English. It has hundreds of monthly active users, and I spent weeks improving it until it's almost on par with ChatGPT Advanced Voice.

Then, my heart broke when OpenAI released [GPT-Live](https://openai.com/index/introducing-gpt-live/), which is miles ahead of ChatGPT Advanced Voice. GPT-Live is so good that I use it almost every day. It made my app feel obsolete. I either need to redirect my users to GPT-Live, or update the app to match it. I chose to try, with the additional constraint that it had to run 100% locally on my MacBook M3 Pro.

My first attempt was to fine-tune Gemma 4 12B to behave like a full-duplex model. Something like grafting a decision tick + speech head to the model. It failed after multiple trials. For now, I think a classic cascade system is still better. We just need to wait until a benevolent frontier AI company releases a full-duplex model that's on par with GPT-Live.

## AI disclosure

This software is developed with strong assistance from Claude and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. This disclosure is inspired by [ds4](https://github.com/antirez/ds4/).

Everything above this section is human-written, everything from here down is AI-generated.

## How it works

```
Browser (mic + camera)
    │
    │  WebSocket (audio PCM + JPEG frames)
    ▼
FastAPI server (conversation history, modes, timers — all server-owned)
    ├── smart-turn-v3 (~0.2s, tiny local classifier)    →  did you finish your thought?
    ├── OpenAI speech-to-text                          →  what did you say? (transcript first)
    ├── Any OpenAI-compatible chat endpoint            →  streams the reply (text + vision)
    │     └── action head (structured-JSON request)    →  timers · modes · research
    ├── OpenAI text-to-speech                          →  speaks sentence-by-sentence
    └── background reasoner (optional)                 →  frontier model for web research
    │
    │  WebSocket (transcript + streamed audio chunks)
    ▼
Browser (playback + transcript)
```

- **Cloud backend.** No local model downloads and no GPU needed: the LLM is any OpenAI-compatible endpoint (api.openai.com directly, or OpenRouter with any vendor's model), and speech rides OpenAI's transcription (STT) and speech (TTS) APIs.- **Background research** (optional). "Find the best pizza in Rome right now" hands the task to a frontier model while the conversation continues; the answer is spoken when it arrives. With no extra config it rides the main OpenAI key (web-search-enabled model); a dedicated `REASONER_*` config lets it run on OpenRouter plus web search.
- **Hands-free turn-taking.** Browser-side [Silero VAD](https://github.com/ricky0123/vad) with a ~200ms silence cutoff — no push-to-talk. Pipecat's [smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) then judges whether you actually finished your thought: mid-thought pauses are held silently until you continue, or answered once you stay quiet.
- **Everything streams.** The utterance is transcribed and the transcript is on screen immediately, then the reply is spoken sentence-by-sentence while the model is still generating.
- **Barge-in.** Speak over the AI to interrupt it; generation is aborted server-side (the streaming connection is torn down). Echo is handled without the browser's echo canceller — or just wear headphones.
- **Actions never ride the speech.** Timers, mode switches, and research requests are decided by a separate structured-JSON request over the same chat endpoint, hidden under TTS playback. The spoken reply stays pure speech, so control markup can never leak into TTS and a promise made aloud is always kept.
- **Timers.** "Set a timer for three minutes for the pasta" — the server owns the clock, the model announces the ring out loud in any mode, and a countdown chip with a cancel button tracks it.
- **Live translation mode.** "Translate everything I say into English" turns Parlor into a consecutive interpreter: each utterance rendered after a short silence, no conversational replies, in any language the STT hears — until you say "stop translating" or hit the stop chip.
- **Just-listen mode.** "Just listen for a while, I want to think out loud" makes it a silent scribe: every utterance transcribed on screen, nothing spoken back, until you address it again.
- **A sense of time.** The model is told how much quiet preceded a turn, how long research took, and when the session started — so "how long was I gone?" gets a real answer.

## Requirements

- Python 3.12+
- An [OpenAI API key](https://platform.openai.com/) in `.env` (or any OpenAI-compatible endpoint + a second provider for speech; see [docs/configuration.md](docs/configuration.md))
- No GPU, no model downloads: the only fetch at startup is the ~3 MB smart-turn classifier
- Works on macOS, Linux, and Windows

Voice conversation with the defaults (gpt-4o-mini + OpenAI speech APIs) costs on the order of **1–2 cents per minute** of talking, dominated by transcription and speech synthesis; bigger chat models scale that up.

## Quick start

```bash
git clone https://github.com/experiencIA-consulting/parlor.git
cd parlor

# Install uv if you don't have it (https://docs.astral.sh/uv/):
#   Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync

# One-time: put your OpenAI key in .env
cp .env.example .env      # then edit: OPENAI_API_KEY=sk-...
# OpenRouter instead:
#   OPENAI_BASE_URL=https://openrouter.ai/api/v1
#   OPENAI_MODEL=openai/gpt-4o-mini

uv run parlor
```

Open [http://localhost:8000](http://localhost:8000), grant camera and microphone access, and start talking. The only download at startup is the ~3 MB smart-turn classifier.

## Configuration

Set these in your shell or a `.env` at the repo root. The common ones:

| Variable            | Default                        | Description                                         |
| ------------------- | ------------------------------ | --------------------------------------------------- |
| `OPENAI_API_KEY`    | (required)                     | API key for the chat endpoint                       |
| `OPENAI_BASE_URL`   | `https://api.openai.com/v1`    | Any OpenAI-compatible chat endpoint (OpenRouter: `https://openrouter.ai/api/v1`) |
| `OPENAI_MODEL`      | `gpt-4o-mini`                  | Conversation model (must accept images for camera)  |
| `PORT`              | `8000`                         | Server port                                         |
| `REASONER_API_KEY`  | (falls back to the chat key)   | API key enabling background research                |

The full list — STT/TTS models, voice, Spanish-pinned STT, history rotation, reasoner, test hooks — is in [docs/configuration.md](docs/configuration.md).

## Latency

First audio = VAD silence (~200ms) + smart-turn (~200ms) + STT (~0.3-1s, grows with utterance length) + chat first token + TTS (~0.5s). Expect roughly 1.5-3s on a typical short question, depending on your distance to OpenAI and the model you point at.

## Testing

An end-to-end suite spawns the real server (cloud LLM, STT, TTS, turn
detector) and drives it over WebSocket with synthesized speech — including
degraded audio (clipped word endings, noise, other voices) that reproduces
live-mic failure modes. The reasoner endpoint is mocked locally; chat, STT
and TTS consume a real OpenAI key
(`OPENAI_API_KEY` must be set to run the full suite; unit tests run
without):

```bash
uv run pytest
```

Set `PARLOR_TEST_URL=ws://localhost:8000/ws` to run it against an
already-running server. Browser-only behavior (echo at speaker volume, VAD
feel, multilingual speech) still needs a live mic and ears.

## Project structure

```
src/parlor/
├── server.py              # FastAPI app + per-connection conversation loop
├── cloud.py               # OpenAI-compatible backend: chat, streaming, STT
├── pipeline.py            # Streaming turn pipeline (decode → sentences → TTS)
├── actions.py             # Action decider (structured JSON: timers, modes, research)
├── reasoner.py            # Background research delegation (OpenAI-compatible)
├── modes.py               # Session modes (conversation, translate, listen)
├── turn_detector.py       # smart-turn-v3 end-of-turn classifier
├── tts.py                 # Cloud TTS (OpenAI speech; Kokoro retained behind TTS_BACKEND=kokoro)
└── web/                   # Frontend (markup, styles, app logic: VAD, camera, playback)
tests/                     # End-to-end test suite (uv run pytest)
benchmarks/
├── bench.py               # End-to-end latency benchmark
├── fixtures.py            # Spoken-audio fixtures (synthesized locally)
├── compare.py             # Diff two benchmark result files
├── turnbench.py           # Turn-detection accuracy benchmark
├── archbench.py           # In-band control tags vs decoupled action head
├── camerabench.py         # Attach-every-turn vs camera-as-tool-call
└── timerprobe.py          # Why the server owns the timer clock
```

## Acknowledgments

- [OpenAI](https://platform.openai.com/docs) APIs — chat, transcription, speech
- [OpenRouter](https://openrouter.ai) — any frontier model behind one OpenAI-compatible endpoint
- [Silero VAD](https://github.com/snakers4/silero-vad) for browser voice activity detection
- [smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) end-of-turn detection by Pipecat

## License

[Apache 2.0](LICENSE)
