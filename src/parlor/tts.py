"""TTS backends: OpenAI cloud TTS by default (no local model), with the
original local Kokoro backends (mlx-audio on Apple Silicon, kokoro-onnx
elsewhere) still available via TTS_BACKEND=kokoro."""

import json
import os
import urllib.request
import urllib.error

import numpy as np

import parlor.cloud as cloud


CLOUD_TTS_TIMEOUT = float(os.environ.get("CLOUD_TTS_TIMEOUT", "120"))


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(self, text: str, voice: str = "", speed: float = 1.0,
                 instructions: str = "") -> np.ndarray:
        raise NotImplementedError


class CloudBackend(TTSBackend):
    """OpenAI /v1/audio/speech. Returns raw 16-bit PCM (@24kHz) as a
    float array; the server streams sentence calls and the client plays
    whatever sample_rate audio_start announced."""

    sample_rate: int = 24000
    default_voice: str = os.environ.get("OPENAI_TTS_VOICE", "alloy")
    _OPENAI_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable",
                      "onyx", "nova", "sage", "shimmer", "verse"}

    def _map_voice(self, voice: str) -> str:
        """Accept Kokoro voice ids too (generic — the caller might be a
        fixture or a per-mode voice): gender is the only meaningful
        signal there. af_* are the female voices, am_* the male ones;
        any unknown voice falls back via the prefix."""
        if not voice:
            return self.default_voice
        if voice in self._OPENAI_VOICES:
            return voice
        if voice.startswith("am_"):
            return "echo"
        if voice.startswith("af_"):
            return self.default_voice if self.default_voice in self._OPENAI_VOICES else "alloy"
        return self.default_voice

    def generate(self, text: str, voice: str = "", speed: float = 1.0,
                 instructions: str = "") -> np.ndarray:
        """instructions is gpt-4o-mini-tts's style prompt (tone, emotion,
        pace, accent — conversational directives, not a persona note) —
        e.g. a persona's robotic/flat delivery rides HERE, never on the
        conversation model. Ignored by tts models without the field if
        some other OPENAI_TTS_MODEL is set."""
        body = {
            "model": cloud.TTS_MODEL,
            "voice": self._map_voice(voice),
            "input": text,
            "response_format": "pcm",  # raw 16-bit LE mono @24kHz
        }
        if instructions:
            body["instructions"] = instructions
        req = urllib.request.Request(
            cloud.BASE_URL + "/audio/speech", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {cloud.API_KEY}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=CLOUD_TTS_TIMEOUT) as resp:
                pcm = np.frombuffer(resp.read(), dtype=np.int16)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"TTS endpoint HTTP {e.code}: "
                               f"{e.read()[:300]!r}") from e
        return pcm.astype(np.float32) / 32768.0


class MLXBackend(TTSBackend):
    """mlx-audio backend (Apple Silicon GPU via MLX)."""

    def __init__(self):
        from mlx_audio.tts.generate import load_model

        self._model = load_model("mlx-community/Kokoro-82M-bf16")
        self.sample_rate = self._model.sample_rate
        # Warmup: triggers pipeline init (phonemizer, spacy, etc.)
        list(self._model.generate(text="Hello", voice="af_heart", speed=1.0))

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1,
                 instructions: str = "") -> np.ndarray:  # noqa: ARG002 — no on-device style field
        results = list(self._model.generate(text=text, voice=voice, speed=speed))
        return np.concatenate([np.array(r.audio) for r in results])


class ONNXBackend(TTSBackend):
    """kokoro-onnx backend (ONNX Runtime, CPU)."""

    def __init__(self):
        import kokoro_onnx
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
        voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")

        self._model = kokoro_onnx.Kokoro(model_path, voices_path)
        self.sample_rate = 24000

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1,
                 instructions: str = "") -> np.ndarray:  # noqa: ARG002 — no on-device style field
        pcm, _sr = self._model.create(text, voice=voice, speed=speed)
        return pcm


def load() -> TTSBackend:
    """Cloud TTS by default; Kokoro (local) only when explicitly asked."""
    backend = os.environ.get("TTS_BACKEND", "").strip().lower()
    if backend == "kokoro":
        import platform
        import sys
        if sys.platform == "darwin" and platform.machine() == "arm64":
            b = MLXBackend()
            print(f"TTS: kokoro mlx-audio (Apple GPU, sample_rate={b.sample_rate})")
            return b
        b = ONNXBackend()
        print(f"TTS: kokoro-onnx (CPU, sample_rate={b.sample_rate})")
        return b
    b = CloudBackend()
    print(f"TTS: openai cloud ({cloud.TTS_MODEL}, voice={b.default_voice}, "
          f"sample_rate={b.sample_rate})")
    return b
