"""Empirical: for the long_question text, which OpenAI TTS voices classify
as COMPLETE vs HELD by the local smart-turn model? Prints a table; keeps
candidate WAVs in benchmarks/fixtures/voicetest-* so the winner can be
promoted into fixtures.py without paying for synthesis again.
"""
import wave

import numpy as np
import fixtures as fx
from parlor import tts
from parlor.turn_detector import TurnDetector

TEXT = fx.FIXTURES["long_question"][0]

def main():
    det = TurnDetector()
    backend = tts.load()
    print(f"TTS: {type(backend).__name__} sample_rate={backend.sample_rate}")
    for voice in ["alloy", "ash", "ballad", "coral", "echo", "fable",
                  "nova", "onyx", "sage", "shimmer"]:
        try:
            pcm = np.asarray(backend.generate(TEXT, voice=voice), dtype=np.float32)
            sr = backend.sample_rate
            pcm = fx._resample_linear(pcm, sr, 16000)
            complete, prob = det.predict(pcm)
            out = fx.FIXTURES_DIR / f"voicetest-{voice}.wav"
            fx._write_wav(out, pcm, 16000)
            mark = "complete" if complete else "HOLD    "
            print(f"{voice:8s} {len(pcm)/16000:5.1f}s p={prob:.2f} {mark}")
        except Exception as e:
            print(f"{voice:8s} FAILED {e}")

if __name__ == "__main__":
    main()
