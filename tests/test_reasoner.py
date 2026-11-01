"""Reasoner unit tests (no network): the search models lean on markdown
and source citations that read terribly aloud — pin the cleaner."""

import json
import urllib.request

import pytest

import parlor.reasoner as reasoner


class _Resp:
    def __init__(self, content):
        self._data = json.dumps(
            {"choices": [{"message": {"content": content}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._data


@pytest.fixture
def feed(monkeypatch):
    def _feed(content):
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda req, timeout: _Resp(content))
    return _feed


def test_strips_markdown_links_urls_and_sources(feed):
    feed("Gold is about **$3,900** per ounce. ([kitco.com]"
         "(https://kitco.com/spot?utm_source=openai))")
    out = reasoner.ask("gold")
    assert out == "Gold is about $3,900 per ounce."


def test_strips_bare_domain_and_bold_only(feed):
    feed("It is **sunny** in Remsda today. (meteor.net)")
    out = reasoner.ask("weather")
    assert out == "It is sunny in Remsda today."


def test_naked_url_and_heading_markers(feed):
    feed("## Price\nsee https://example.com/x for details")
    out = reasoner.ask("price")
    assert "https://" not in out and "##" not in out


def test_empty_answer_raises(feed):
    feed("   ")
    with pytest.raises(RuntimeError, match="empty"):
        reasoner.ask("anything")
