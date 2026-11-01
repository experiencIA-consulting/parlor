"""main() is pure configuration — pin the env contract so a refactor
can't silently re-hardcode the localhost-only bind (the whole point of
PARLOR_HOST is LAN/mobile testing)."""

import pytest

import parlor.server as server


@pytest.fixture
def capture_run(monkeypatch):
    calls = {}
    monkeypatch.setattr(server.uvicorn, "run",
                        lambda app, host, port: calls.update(host=host, port=port))
    return calls


def test_default_binds_localhost(capture_run, monkeypatch):
    monkeypatch.delenv("PARLOR_HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    server.main()
    assert capture_run == {"host": "localhost", "port": 8000}


def test_parlor_host_expands_to_lan(capture_run, monkeypatch):
    monkeypatch.setenv("PARLOR_HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "8001")
    server.main()
    assert capture_run == {"host": "0.0.0.0", "port": 8001}


def test_blank_falls_back_to_localhost(capture_run, monkeypatch):
    monkeypatch.setenv("PARLOR_HOST", "   ")
    server.main()
    assert capture_run == {"host": "localhost", "port": 8000}


def test_explicit_ip_passes_through(capture_run, monkeypatch):
    monkeypatch.setenv("PARLOR_HOST", "192.168.1.10")
    server.main()
    assert capture_run == {"host": "192.168.1.10", "port": 8000}
