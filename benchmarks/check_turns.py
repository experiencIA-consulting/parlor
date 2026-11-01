"""Local smart-turn classification of every fixture WAV.

Zero API cost, no server: feeds each fixture the same way the server does
(raw float32 PCM) and prints p(complete). Used to retune the e2e suite's
held/completed expectations after the TTS voice swap (Kokoro voices now
synthesize through the OpenAI TTS mapping, which changes prosody).

Run:  uv run python benchmarks/check_turns.py
"""
import wave

import fixtures as fx


def main():
    import numpy as np
    from parlor.turn_detector import TurnDetector

    det = TurnDetector()
    for name in [*fx.FIXTURES, *fx.VARIANTS]:
        path = fx.FIXTURES_DIR / f"{name}.wav"
        if not path.exists():
            print(f"{name:32s} missing (run fixtures.generate_all)")
            continue
        with wave.open(str(path), "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        complete, prob = det.predict(pcm.astype(np.float32) / 32768.0)
        dur = len(pcm) / 16000
        mark = "complete" if complete else "HOLD    "
        print(f"{name:32s} {dur:5.1f}s  p={prob:.2f}  {mark}")


if __name__ == "__main__":
    main()
