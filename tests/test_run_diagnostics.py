"""Tests for the per-run diagnostics added to A1 — per-LLM-call telemetry, the
ReAct wall-clock deadline, and the run-lifecycle helpers.

These attribute the "why did a query hang" question: an ``llm_call`` event per
provider call (latency/outcome/finish reason), and a ``run_timeout`` short-circuit
when the run budget is exhausted. We exercise the real methods on a lightweight
subclass that skips A1's heavy ``__init__`` (which loads tools / know-how / a real
LLM) but keeps method binding intact.
"""

from __future__ import annotations

import io
import json
import logging
import time

import pytest
from biomni import observability as obs
from biomni.agent.a1 import A1
from biomni.config import default_config
from biomni.llm_resilience import LLMUsageTracker
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage


@pytest.fixture(autouse=True)
def _restore_root_logging():
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def _capture():
    stream = io.StringIO()
    obs.setup_logging("DEBUG", fmt="json", stream=stream, force=True)
    return stream


def _events(stream, name):
    out = []
    for line in stream.getvalue().splitlines():
        if line.strip():
            obj = json.loads(line)
            if obj.get("event") == name:
                out.append(obj)
    return out


class _DiagAgent(A1):
    """A1 with the heavy constructor bypassed — only the diagnostics-relevant state."""

    def __init__(self, llm, *, source="Anthropic", model="claude-x"):
        self.llm = llm
        self.usage_tracker = LLMUsageTracker()
        self._llm_source = source
        self._llm_model_name = model
        self._llm_prompt_caching = False
        self._react_step = 0
        self._run_started_monotonic = None
        self._run_deadline = None


class _BoomLLM:
    def invoke(self, _messages):
        raise RuntimeError("provider down")


# ---------------------------------------------------------------------------
# finish reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "meta,expected",
    [
        ({"stop_reason": "end_turn"}, "end_turn"),  # Anthropic
        ({"finish_reason": "stop"}, "stop"),  # OpenAI
        ({"stop_reason": "max_tokens", "finish_reason": "length"}, "max_tokens"),  # stop_reason wins
        ({}, None),
        (None, None),
        ("not-a-dict", None),
    ],
)
def test_extract_finish_reason(meta, expected):
    class _Resp:
        response_metadata = meta

    assert A1._extract_finish_reason(_Resp()) == expected


# ---------------------------------------------------------------------------
# _invoke_llm telemetry
# ---------------------------------------------------------------------------


def test_invoke_llm_emits_ok_event_and_records_usage():
    stream = _capture()
    agent = _DiagAgent(FakeListChatModel(responses=["hello"]))
    agent._react_step = 4
    with obs.bind_run(run_id="R1"):
        resp = agent._invoke_llm([HumanMessage(content="hi")])
    assert resp.content == "hello"
    assert agent.usage_tracker.calls == 1  # usage still recorded
    calls = _events(stream, "llm_call")
    assert len(calls) == 1
    ev = calls[0]
    assert ev["status"] == "ok"
    assert ev["model"] == "claude-x"
    assert ev["source"] == "Anthropic"
    assert ev["step"] == 4
    assert ev["run_id"] == "R1"
    assert "latency_ms" in ev
    # Token fields mirror llm_usage; legitimate zeros are kept, not dropped.
    for field in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens"):
        assert field in ev


def test_invoke_llm_emits_error_event_and_reraises():
    stream = _capture()
    agent = _DiagAgent(_BoomLLM())
    with pytest.raises(RuntimeError, match="provider down"):
        agent._invoke_llm([HumanMessage(content="hi")])
    calls = _events(stream, "llm_call")
    assert len(calls) == 1
    assert calls[0]["status"] == "error"
    assert calls[0]["error_type"] == "RuntimeError"
    assert "latency_ms" in calls[0]


# ---------------------------------------------------------------------------
# run lifecycle: begin / elapsed / deadline
# ---------------------------------------------------------------------------


def test_begin_run_arms_deadline_when_configured(monkeypatch):
    agent = _DiagAgent(FakeListChatModel(responses=["x"]))
    monkeypatch.setattr(default_config, "run_timeout_seconds", 300, raising=False)
    agent._react_step = 9
    agent._begin_run()
    assert agent._react_step == 0
    assert agent._run_started_monotonic is not None
    assert agent._run_deadline == pytest.approx(agent._run_started_monotonic + 300, abs=1.0)


def test_begin_run_no_deadline_when_unset(monkeypatch):
    agent = _DiagAgent(FakeListChatModel(responses=["x"]))
    monkeypatch.setattr(default_config, "run_timeout_seconds", None, raising=False)
    agent._begin_run()
    assert agent._run_deadline is None


def test_run_elapsed_ms_none_before_begin():
    agent = _DiagAgent(FakeListChatModel(responses=["x"]))
    assert agent._run_elapsed_ms() is None
    agent._run_started_monotonic = time.monotonic() - 1.0
    assert agent._run_elapsed_ms() == pytest.approx(1000, abs=200)


def test_enforce_deadline_false_when_unarmed_or_not_exceeded():
    agent = _DiagAgent(FakeListChatModel(responses=["x"]))
    state = {"messages": [HumanMessage(content="q")], "next_step": None}
    assert agent._enforce_run_deadline(state) is False  # no deadline
    agent._run_deadline = time.monotonic() + 100
    assert agent._enforce_run_deadline(state) is False  # not yet exceeded
    assert state["next_step"] is None
    assert len(state["messages"]) == 1


def test_enforce_deadline_true_when_exceeded_stops_run(monkeypatch):
    stream = _capture()
    monkeypatch.setattr(default_config, "run_timeout_seconds", 5, raising=False)
    agent = _DiagAgent(FakeListChatModel(responses=["x"]))
    agent._react_step = 12
    agent._run_started_monotonic = time.monotonic() - 10
    agent._run_deadline = time.monotonic() - 1  # already past
    state = {"messages": [HumanMessage(content="q")], "next_step": None}
    with obs.bind_run(run_id="R2"):
        assert agent._enforce_run_deadline(state) is True
    assert state["next_step"] == "end"
    # A user-facing solution message is appended explaining the stop.
    assert "<solution>" in state["messages"][-1].content
    assert "time budget" in state["messages"][-1].content
    evs = _events(stream, "run_timeout")
    assert len(evs) == 1
    assert evs[0]["step"] == 12
    assert evs[0]["budget_s"] == 5
    assert evs[0]["run_id"] == "R2"
