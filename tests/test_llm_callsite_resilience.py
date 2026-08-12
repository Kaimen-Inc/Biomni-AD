"""Tests that previously-hardcoded LLM call sites route through ``get_llm``.

``biomni.utils.write_python_code`` and ``ToolRetriever.prompt_based_retrieval``
used to instantiate provider chat models directly (``ChatAnthropic`` /
``ChatOpenAI``), bypassing the project's LLM resilience config (max_retries /
request_timeout -> provider-SDK 429/5xx backoff). These tests pin them to the
shared ``get_llm`` factory so the resilience knobs (and their env overrides)
actually reach the call site.

We patch ``biomni.llm.get_llm`` with a recorder that returns a deterministic
``FakeListChatModel`` so no real provider package or network is needed.
"""

from __future__ import annotations

import pytest
from biomni.config import BiomniConfig
from langchain_core.language_models.fake_chat_models import FakeListChatModel


@pytest.fixture(autouse=True)
def _clear_resilience_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate from any ambient BIOMNI_LLM_* overrides on the host."""
    for k in ("BIOMNI_LLM_MAX_RETRIES", "BIOMNI_LLM_REQUEST_TIMEOUT"):
        monkeypatch.delenv(k, raising=False)


def _recording_get_llm(responses: list[str], calls: list[dict]):
    """Build a fake ``get_llm`` that records every call and returns a fake model."""

    def fake(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return FakeListChatModel(responses=list(responses))

    return fake


# ---------------------------------------------------------------------------
# write_python_code
# ---------------------------------------------------------------------------


def test_write_python_code_routes_through_get_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr("biomni.llm.get_llm", _recording_get_llm(["```python\nprint(1 + 1)\n```"], calls))

    from biomni.utils import write_python_code

    out = write_python_code("add two numbers")

    assert "print(1 + 1)" in out
    assert len(calls) == 1, "write_python_code must build its model via get_llm exactly once"
    args, kwargs = calls[0]["args"], calls[0]["kwargs"]
    assert args[0] == "claude-3-5-sonnet-20240620"
    assert kwargs["source"] == "Anthropic"
    assert isinstance(kwargs["config"], BiomniConfig)


def test_write_python_code_honors_env_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh BiomniConfig() at call time picks up BIOMNI_LLM_MAX_RETRIES."""
    monkeypatch.setenv("BIOMNI_LLM_MAX_RETRIES", "7")
    calls: list[dict] = []
    monkeypatch.setattr("biomni.llm.get_llm", _recording_get_llm(["```python\nx = 1\n```"], calls))

    from biomni.utils import write_python_code

    write_python_code("noop")

    cfg = calls[0]["kwargs"]["config"]
    assert cfg.llm_max_retries == 7


# ---------------------------------------------------------------------------
# ToolRetriever.prompt_based_retrieval
# ---------------------------------------------------------------------------


def _resources() -> dict:
    return {
        "tools": [{"name": "t0", "description": "d0"}, {"name": "t1", "description": "d1"}],
        "data_lake": ["d_a", "d_b"],
        "libraries": ["lib0", "lib1"],
    }


def test_retriever_fallback_routes_through_get_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    response = "TOOLS: [0]\nDATA_LAKE: []\nLIBRARIES: [1]"
    monkeypatch.setattr("biomni.llm.get_llm", _recording_get_llm([response], calls))

    from biomni.model.retriever import ToolRetriever

    selected = ToolRetriever().prompt_based_retrieval("find tools", _resources(), llm=None)

    assert len(calls) == 1, "the llm=None fallback must build its model via get_llm"
    args, kwargs = calls[0]["args"], calls[0]["kwargs"]
    assert args[0] == "gpt-4o"
    assert kwargs["source"] == "OpenAI"
    assert isinstance(kwargs["config"], BiomniConfig)
    # Routing through get_llm must not change selection behavior.
    assert selected["tools"] == [{"name": "t0", "description": "d0"}]
    assert selected["libraries"] == ["lib1"]
    assert selected["data_lake"] == []


def test_retriever_honors_env_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOMNI_LLM_MAX_RETRIES", "5")
    calls: list[dict] = []
    monkeypatch.setattr("biomni.llm.get_llm", _recording_get_llm(["TOOLS: []\nDATA_LAKE: []\nLIBRARIES: []"], calls))

    from biomni.model.retriever import ToolRetriever

    ToolRetriever().prompt_based_retrieval("q", _resources(), llm=None)

    assert calls[0]["kwargs"]["config"].llm_max_retries == 5


def test_retriever_uses_provided_llm_without_get_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """When an llm is supplied (the agent path), get_llm must not be called."""
    calls: list[dict] = []
    monkeypatch.setattr("biomni.llm.get_llm", _recording_get_llm(["unused"], calls))

    from biomni.model.retriever import ToolRetriever

    provided = FakeListChatModel(responses=["TOOLS: [0]\nDATA_LAKE: []\nLIBRARIES: []"])
    selected = ToolRetriever().prompt_based_retrieval("q", _resources(), llm=provided)

    assert calls == [], "get_llm must not be called when an llm is provided"
    assert selected["tools"] == [{"name": "t0", "description": "d0"}]
