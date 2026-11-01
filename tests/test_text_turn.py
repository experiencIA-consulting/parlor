"""Typed-turn and optional-TTS coverage without live provider calls."""

import asyncio
import json

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from parlor import server
from parlor.pipeline import run_turn, speak_only


class FakeWebSocket:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.sent = []

    async def accept(self):
        pass

    async def receive_text(self):
        try:
            return json.dumps(next(self.payloads))
        except StopIteration:
            raise WebSocketDisconnect(code=1000)

    async def send_text(self, text):
        self.sent.append(json.loads(text))


@pytest.mark.parametrize(
    ("mode", "style", "expected_prompt"),
    [
        ("conversation", "chat", "written message"),
        ("translate", "translate", "written text into English"),
        ("listen", "chat", "written message"),
    ],
)
def test_text_turn_skips_stt_and_stores_message_once(
        monkeypatch, mode, style, expected_prompt):
    payloads = []
    if mode != "conversation":
        payloads.append({"type": "set_mode", "mode": mode})
    payloads.extend([
        {"type": "text_turn", "request_id": "typed-1", "text": "Hola",
         "speak": False, "response_style": style, "image": "aW1hZ2U="},
        {"type": "text_turn", "request_id": "typed-2", "text": "¿Y esto?",
         "speak": False, "response_style": style},
    ])
    ws = FakeWebSocket(payloads)
    turn_inputs = []

    async def fake_run_turn(ws, messages, interrupted, active, backend, **kwargs):
        turn_inputs.append((messages, kwargs))
        await server.send_json(ws, {
            "type": "text_delta", "text": "Respuesta.",
            "request_id": kwargs.get("request_id"),
        })
        await server.send_json(ws, {
            "type": "turn_final", "request_id": kwargs.get("request_id"),
            "transcription": None, "timings": {}, "spoke": False,
        })
        return "Respuesta.", 8, False, False

    monkeypatch.setattr(server.reasoner, "enabled", lambda: False)
    monkeypatch.setattr(server, "run_turn", fake_run_turn)
    monkeypatch.setattr(server, "tts_backend", object())
    monkeypatch.setattr(server.actions, "decide_after", lambda *_: server.actions.NONE)
    monkeypatch.setattr(server.cloud, "transcribe",
                        lambda *_: pytest.fail("typed turn called STT"))

    asyncio.run(server.websocket_endpoint(ws))

    assert len(turn_inputs) == 2
    first_messages, first_kwargs = turn_inputs[0]
    first_user = first_messages[-1]
    first_content = first_user["content"]
    assert first_user["role"] == "user"
    assert first_content[0]["type"] == "image_url"
    assert first_content[1]["text"] == "Hola"
    assert expected_prompt in first_content[2]["text"]
    assert first_kwargs["tts_enabled"] is False
    assert first_kwargs["request_id"] == "typed-1"
    assert first_kwargs["instruction"] == ""

    # The second request receives the first exchange from server-owned
    # history, with its typed input present exactly once.
    second_messages, _ = turn_inputs[1]
    prior_user = second_messages[-3]
    assert [p.get("text") for p in prior_user["content"] if p["type"] == "text"].count("Hola") == 1
    assert any(m.get("request_id") == "typed-2" for m in ws.sent)
    assert not any(m["type"] == "transcript" for m in ws.sent)


class FakeChatStream:
    def __init__(self, messages, max_tokens):
        self.prompt_tokens = 17

    def run(self, on_delta):
        on_delta("Here is the written answer. ")


class NoTTS:
    sample_rate = 24000

    def generate(self, *args, **kwargs):
        pytest.fail("TTS must not run for a text-only response")


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(json.loads(text))


def test_run_turn_can_stream_text_without_calling_tts(monkeypatch):
    monkeypatch.setattr("parlor.pipeline.cloud.ChatStream", FakeChatStream)
    ws = FakeSocket()

    result = asyncio.run(run_turn(
        ws, [{"role": "user", "content": "Question"}], asyncio.Event(),
        {}, NoTTS(), expect_transcript=False, tts_enabled=False,
        request_id="typed-no-voice"))

    assert result[0].strip() == "Here is the written answer."
    assert result[3] is False
    assert [m["type"] for m in ws.sent] == ["text_delta", "turn_final"]
    assert ws.sent[0]["request_id"] == "typed-no-voice"
    assert ws.sent[-1]["request_id"] == "typed-no-voice"
    assert ws.sent[-1]["output"] == "text"
    assert ws.sent[-1]["spoke"] is False


def test_failed_text_turn_retry_reuses_history_entry(monkeypatch):
    payload = {"type": "text_turn", "request_id": "retry-me",
               "text": "Keep this message", "speak": False}
    ws = FakeWebSocket([payload, payload])
    calls = []

    async def fail_then_succeed(ws, messages, interrupted, active, backend, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            raise RuntimeError("mock chat outage")
        await server.send_json(ws, {
            "type": "text_delta", "text": "Recovered.",
            "request_id": kwargs["request_id"],
        })
        await server.send_json(ws, {
            "type": "turn_final", "request_id": kwargs["request_id"],
            "transcription": None, "timings": {}, "spoke": False,
        })
        return "Recovered.", 5, False, False

    monkeypatch.setattr(server.reasoner, "enabled", lambda: False)
    monkeypatch.setattr(server, "run_turn", fail_then_succeed)
    monkeypatch.setattr(server, "tts_backend", object())
    monkeypatch.setattr(server.actions, "decide_after", lambda *_: server.actions.NONE)
    monkeypatch.setattr(server.cloud, "transcribe",
                        lambda *_: pytest.fail("typed retry called STT"))

    asyncio.run(server.websocket_endpoint(ws))

    user_messages = [m for m in calls[-1] if m.get("role") == "user"]
    assert len(user_messages) == 1
    assert any(m["type"] == "turn_error" for m in ws.sent)
    assert sum(m.get("request_id") == "retry-me" and m["type"] == "turn_final"
               for m in ws.sent) == 1


class FailingTTS:
    sample_rate = 24000

    def generate(self, *args, **kwargs):
        raise RuntimeError("mock TTS outage")


class SuccessfulTTS:
    sample_rate = 24000

    def generate(self, *args, **kwargs):
        return np.zeros(2400, dtype=np.float32)


def test_tts_failure_preserves_text_response(monkeypatch):
    monkeypatch.setattr("parlor.pipeline.cloud.ChatStream", FakeChatStream)
    ws = FakeSocket()

    result = asyncio.run(run_turn(
        ws, [{"role": "user", "content": "Question"}], asyncio.Event(),
        {}, FailingTTS(), expect_transcript=False, request_id="typed-tts-fail"))

    final = next(m for m in ws.sent if m["type"] == "turn_final")
    assert result[0].strip() == "Here is the written answer."
    assert result[3] is False
    assert final["tts_failed"] is True
    assert final["output"] == "text"
    assert any(m["type"] == "text_delta" for m in ws.sent)
    assert not any(m["type"] == "audio_chunk" for m in ws.sent)


def test_fallback_tts_failure_is_visible_as_text():
    ws = FakeSocket()

    spoke = asyncio.run(speak_only(ws, "Please try again.", FailingTTS()))

    assert spoke is False
    assert ws.sent[0] == {"type": "text_delta", "text": "Please try again. "}
    assert ws.sent[-1]["output"] == "text"
    assert ws.sent[-1]["tts_failed"] is True


def test_default_voice_turn_still_streams_audio(monkeypatch):
    monkeypatch.setattr("parlor.pipeline.cloud.ChatStream", FakeChatStream)
    ws = FakeSocket()

    result = asyncio.run(run_turn(
        ws, [{"role": "user", "content": "Question"}], asyncio.Event(),
        {}, SuccessfulTTS(), expect_transcript=False))

    kinds = [m["type"] for m in ws.sent]
    assert result[3] is True
    assert "audio_start" in kinds and "audio_chunk" in kinds and "audio_end" in kinds
    assert kinds.index("turn_final") < kinds.index("audio_end")
    final = next(m for m in ws.sent if m["type"] == "turn_final")
    assert final["output"] == "spoken"
    assert final["spoke"] is True
