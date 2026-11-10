"""Tests for the ``trace.json`` a run leaves in its folder.

The chat UI drives the graph itself (``agent.app.stream``) and afterwards hands
the conversation to the agent through ``raw_log``; only ``go()`` and
``go_stream()`` ever set ``agent.log``. The trace used to be written straight
from ``agent.log``, so every run started from the chat produced a zero-byte
``trace.json`` and nothing but a line on stdout said so.

``A1`` is expensive to construct (it builds an LLM client and a LangGraph), so
these bind the methods under test to a lightweight stub, as
``test_agent_workspace_scope`` does. The whole module skips when the agent stack
(pandas/langchain/langgraph) is not installed.
"""

from __future__ import annotations

import json
import os

import pytest

try:
    from biomni.agent.a1 import A1
    from biomni.agent.ad1 import AD1
    from biomni.utils import pretty_print
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    HAS_AGENT = True
except Exception:  # pragma: no cover - depends on the installed extras
    HAS_AGENT = False

pytestmark = pytest.mark.skipif(not HAS_AGENT, reason="agent stack (pandas/langchain/langgraph) not installed")

PROMPT = "Plot the age distribution of cohort_1.csv"


class _StubAgent:
    """Carries only what the trace methods and ``_save_run_artifacts`` read."""

    # No class-level ``log``: an agent the chat UI drives never has one.
    data_root_dir = None

    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)

    if HAS_AGENT:
        _trace_entries = A1._trace_entries
        _save_trace = A1._save_trace
        _generate_notebook = A1._generate_notebook

    def _generate_markdown_content(self, include_images=True):
        return "# report"

    def save_conversation_history(self, *args, **kwargs):
        pass

    def _get_all_files(self, directory):
        return set()


def _conversation():
    """What the ReAct loop leaves in the final state of one chat turn."""
    return [
        HumanMessage(content=PROMPT),
        AIMessage(content="Plotting it.\n<execute>print(1 + 1)</execute>"),
        HumanMessage(content="<observation>2</observation>"),
        AIMessage(content="Done: the plot is saved as age_distribution.png."),
    ]


def _chat_driven_agent(messages=None):
    """An agent in the state ``_save_run_artifacts_for_agent`` leaves it in."""
    messages = _conversation() if messages is None else messages
    return _StubAgent(raw_log=list(messages), _conversation_state={"messages": list(messages)})


def _read_trace(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestTraceOfAChatDrivenRun:
    """The reported bug: the run folder's ``trace.json`` was empty."""

    @pytest.mark.parametrize("agent_cls", [A1, AD1] if HAS_AGENT else [], ids=["A1", "AD1"])
    def test_save_run_artifacts_writes_the_conversation(self, agent_cls, tmp_path):
        """Both writers (A1's and AD1's own) go through the real ``_save_run_artifacts``."""
        agent = _chat_driven_agent()
        agent._save_run_artifacts = agent_cls._save_run_artifacts.__get__(agent)
        run_dir = tmp_path / "run_1"
        run_dir.mkdir()

        agent._save_run_artifacts("run_1", str(run_dir), set())

        trace = run_dir / "trace.json"
        assert trace.stat().st_size > 0
        entries = _read_trace(trace)
        assert [type(e) for e in entries] == [str] * 4
        assert entries == [pretty_print(m, printout=False) for m in _conversation()]

    def test_the_run_list_can_still_read_the_prompt_from_the_first_entry(self, tmp_path):
        """``AD1``'s run list takes the prompt from the text after the first entry's banner."""
        agent = _chat_driven_agent()
        path = tmp_path / "trace.json"

        agent._save_trace(str(path))

        first = _read_trace(path)[0]
        assert "Human Message" in first
        assert first.split("\n\n")[-1] == PROMPT

    def test_tool_calls_are_kept(self, tmp_path):
        """An agent that calls tools natively leaves them in the trace, as ``go()``'s log does."""
        call = {"name": "run_python_repl", "args": {"command": "print(2)"}, "id": "call_1"}
        messages = [
            HumanMessage(content=PROMPT),
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(content="2", tool_call_id="call_1", name="run_python_repl"),
        ]
        agent = _chat_driven_agent(messages)
        path = tmp_path / "trace.json"

        agent._save_trace(str(path))

        entries = _read_trace(path)
        assert "Tool Call: run_python_repl" in entries[1]
        assert "print(2)" in entries[1]
        assert "Tool Message" in entries[2]

    def test_non_ascii_text_survives(self, tmp_path):
        text = "α-synuclein ≥ 2× in 患者 samples"
        agent = _chat_driven_agent([HumanMessage(content=text)])
        path = tmp_path / "trace.json"

        agent._save_trace(str(path))

        assert text in _read_trace(path)[0]

    def test_the_conversation_is_not_printed_again(self, capsys):
        """Saving must not dump the whole transcript into the server log a second time."""
        agent = _chat_driven_agent()

        agent._trace_entries()

        assert capsys.readouterr().out == ""


class TestTraceOfAGoRun:
    """``go()`` and ``go_stream()`` record ``log`` as they stream; that stays the source."""

    def test_the_recorded_log_is_written_as_is(self):
        log = ["=== Human Message ===\n\nhi", "=== Ai Message ===\n\nhello"]
        agent = _StubAgent(log=list(log), raw_log=_conversation())

        assert agent._trace_entries() == log

    def test_a_run_that_stopped_part_way_keeps_what_it_recorded(self):
        """``go()`` clears ``raw_log`` first, so a failure part-way leaves only ``log`` to go on."""
        log = ["=== Human Message ===\n\nhi"]
        agent = _StubAgent(log=list(log), raw_log=[])

        assert agent._trace_entries() == log

    def test_the_previous_runs_state_never_stands_in_for_this_one(self):
        """``go()`` sets ``_conversation_state`` only when the stream ends, so after a failure it is the last run's.

        Nothing was recorded for this run, and the previous run's conversation
        is not a trace of it.
        """
        agent = _StubAgent(log=[], raw_log=[], _conversation_state={"messages": _conversation()})

        assert agent._trace_entries() == []

    def test_an_agent_without_a_run_in_the_log_falls_back_to_the_conversation(self):
        agent = _StubAgent(log=[], raw_log=_conversation())

        assert len(agent._trace_entries()) == 4


class TestNothingRecorded:
    def test_an_empty_trace_is_an_empty_list_not_an_empty_file(self, tmp_path):
        """A run with no conversation still leaves a file ``json.load`` can read."""
        path = tmp_path / "trace.json"

        _StubAgent()._save_trace(str(path))

        assert _read_trace(path) == []

    def test_an_agent_that_never_ran_does_not_raise(self):
        assert _StubAgent()._trace_entries() == []


class _PdfStub(_StubAgent):
    """``_StubAgent`` with the real ``save_conversation_history`` and no PDF library to call."""

    if HAS_AGENT:
        save_conversation_history = A1.save_conversation_history

    def _convert_markdown_to_pdf(self, markdown_path, pdf_path):
        """CI has neither weasyprint nor pandoc, and the conversion is not what is under test."""


class TestPdfSummary:
    def test_a_saved_pdf_is_not_reported_as_failed(self, tmp_path, capsys):
        """The step count after the PDF was written read ``self.log``; with none, it reported a saved PDF as a failure."""
        agent = _PdfStub(raw_log=_conversation())

        agent.save_conversation_history(str(tmp_path / "report"), save_pdf=True)

        out = capsys.readouterr().out
        assert "Total steps recorded: 4" in out
        assert "Could not convert" not in out


class TestRobustness:
    def test_one_message_that_cannot_be_printed_does_not_blank_the_trace(self, tmp_path):
        """List content with a block that has no type makes ``pretty_print`` raise."""
        broken = AIMessage(content=[{"text": "no type on this block"}])
        messages = [HumanMessage(content=PROMPT), broken, AIMessage(content="Done.")]
        with pytest.raises(KeyError):
            pretty_print(broken, printout=False)
        agent = _chat_driven_agent(messages)
        path = tmp_path / "trace.json"

        agent._save_trace(str(path))

        entries = _read_trace(path)
        assert len(entries) == 3
        assert PROMPT in entries[0]
        assert "no type on this block" in entries[1]
        assert "Done." in entries[2]

    def test_a_failed_serialisation_does_not_leave_an_empty_file(self, tmp_path):
        """The file used to be opened for writing before anything was serialised."""
        agent = _StubAgent(log=[object()])
        path = tmp_path / "trace.json"

        with pytest.raises(TypeError):
            agent._save_trace(str(path))

        assert not path.exists()

    def test_a_failed_serialisation_leaves_an_existing_trace_alone(self, tmp_path):
        path = tmp_path / "trace.json"
        path.write_text('["earlier"]', encoding="utf-8")

        with pytest.raises(TypeError):
            _StubAgent(log=[object()])._save_trace(str(path))

        assert path.read_text(encoding="utf-8") == '["earlier"]'

    def test_the_file_keeps_default_permissions(self, tmp_path):
        """No mkstemp-style 0600: the run folder is shared with the people who download from it."""
        path = tmp_path / "trace.json"
        other = tmp_path / "report.md"
        other.write_text("# report", encoding="utf-8")

        _chat_driven_agent()._save_trace(str(path))

        assert os.stat(path).st_mode == os.stat(other).st_mode
