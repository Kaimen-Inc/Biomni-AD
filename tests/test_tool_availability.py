"""Tests for biomni.tool.availability - tools a deployment must not advertise.

``advanced_web_search_claude`` calls Anthropic directly. Behind the platform's
LLM proxy that would go around the per-user metering the platform requires, so
there it is neither offered to the agent nor callable.
"""

from __future__ import annotations

import pytest
from biomni.tool import availability
from biomni.tool.availability import CLAUDE_WEB_SEARCH


@pytest.fixture(autouse=True)
def _direct_claude(monkeypatch):
    """A deployment where the Claude web search works, unless a test says otherwise."""
    from biomni.config import default_config

    for name in ("BIOMNI_LLM_PROXY_URL", "BIOMNI_LLM_PROXY_SCHEMA", "BIOMNI_LLM_PROXY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(default_config, "llm", "claude-sonnet-4-5")


def test_available_with_a_claude_model_and_a_key():
    assert availability.claude_web_search_problem() is None
    assert availability.unavailable_tools() == {}


@pytest.mark.parametrize("schema", [None, "anthropic", "not-a-schema"])
def test_withheld_behind_the_llm_proxy_even_with_a_key(monkeypatch, schema):
    monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", "https://proxy.example/v1")
    if schema:
        monkeypatch.setenv("BIOMNI_LLM_PROXY_SCHEMA", schema)
    problem = availability.unavailable_tools()[CLAUDE_WEB_SEARCH]
    assert "LLM proxy" in problem and "search_google()" in problem


def test_withheld_without_a_claude_model(monkeypatch):
    from biomni.config import default_config

    monkeypatch.setattr(default_config, "llm", "gpt-4.1")
    assert "needs a Claude model" in availability.claude_web_search_problem()


def test_withheld_without_an_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert "ANTHROPIC_API_KEY" in availability.claude_web_search_problem()


def _catalogue_names() -> set[str]:
    from biomni.utils import read_module2api

    return {api["name"] for apis in read_module2api().values() for api in apis}


def test_the_catalogue_leaves_out_what_cannot_run(monkeypatch):
    assert CLAUDE_WEB_SEARCH in _catalogue_names()
    monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", "https://proxy.example/v1")
    names = _catalogue_names()
    assert CLAUDE_WEB_SEARCH not in names
    # Only that one tool: the rest of the literature tools are still offered.
    assert {"search_google", "query_pubmed", "query_arxiv"} <= names


def test_the_tool_refuses_behind_the_proxy_without_calling_anthropic(monkeypatch):
    literature = pytest.importorskip("biomni.tool.literature")
    anthropic = pytest.importorskip("anthropic")

    def no_client(*_args, **_kwargs):
        raise AssertionError("a direct Anthropic client must not be created behind the proxy")

    monkeypatch.setattr(anthropic, "Anthropic", no_client)
    monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", "https://proxy.example/v1")
    with pytest.raises(RuntimeError, match="LLM proxy"):
        literature.advanced_web_search_claude("amyloid PET tracers")


def test_the_tool_sends_only_the_anthropic_key(monkeypatch):
    """Never the custom-endpoint key: that would hand one provider's credential to another."""
    literature = pytest.importorskip("biomni.tool.literature")
    anthropic = pytest.importorskip("anthropic")
    from biomni.config import default_config

    monkeypatch.setattr(default_config, "api_key", "custom-endpoint-key")
    created = {}

    class _Client:
        def __init__(self, api_key=None, **_kwargs):
            created["api_key"] = api_key
            self.messages = self

        def create(self, **_kwargs):
            raise RuntimeError("stop here")

    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    monkeypatch.setattr(literature.time, "sleep", lambda _s: None)
    literature.advanced_web_search_claude("amyloid PET tracers", max_retries=1)
    assert created["api_key"] == "sk-ant-test"
