"""Personality catalog sanity: unique keys, valid voice ids, every preset
ready for both the selector API and the TTS path (no API calls)."""

import json
import urllib.request

import numpy as np

from parlor import personas, tts


def test_keys_unique_and_stable():
    keys = [p.key for p in personas.PRESETS]
    assert len(keys) == len(set(keys))
    assert "neutral" in keys


def test_fields_present():
    for p in personas.PRESETS:
        assert p.key and p.label and p.description and p.sample_line
        assert p.tts_instructions, p.key  # every voice styles its delivery


def test_voices_are_valid_openai_ids_or_default():
    backend = tts.load()
    for p in personas.PRESETS:
        if p.voice:
            assert p.voice in backend._OPENAI_VOICES, p.key


def test_persona_voice_and_instructions_reach_generate_body(monkeypatch):
    """The persona's voice + delivery must actually arrive at the OpenAI
    request, not be mapped away (a 'male robot' must not come back alloy-
    voiced). Capture the body by stubbing the network call."""
    captured = {}

    class _Body:
        def __init__(self, pcm):
            self.data = pcm.tobytes()

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode())
        return _Body(np.zeros(24000, dtype=np.int16))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    # A fresh backend so the module-level env defaults can't surprise.
    b = tts.CloudBackend()
    b.generate("hola", voice="onyx", instructions="flat robot")
    assert captured["body"]["voice"] == "onyx"
    assert captured["body"]["instructions"] == "flat robot"


def test_get_unknown_degrades_to_neutral():
    p = personas.get("no-existe-ni-viejito")
    assert p.key == "neutral"
    assert personas.get(" NARRADORA ").key == "narradora"