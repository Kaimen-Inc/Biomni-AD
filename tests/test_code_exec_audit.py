"""Tests for the code-execution audit log emitted by A1's execute node.

The agent runs LLM-generated code un-sandboxed, so each execution emits a
structured ``code_execution`` event (hash + size + status + timing, never the
raw source/output). We test the two pure pieces — status classification and the
audit emitter — without standing up the full LangGraph agent.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import types

import pytest
from biomni import observability as obs
from biomni.agent.a1 import A1


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


# ---------------------------------------------------------------------------
# status classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "result,expected",
    [
        ("hello world\n", "ok"),
        ("42", "ok"),
        ("", "ok"),
        ("Error: name 'x' is not defined", "error"),
        ("Error running R code:\nboom", "error"),
        ("Error running Bash script (exit code 1):\nboom", "error"),
        ("Error in execution: kaboom", "error"),
        ("ERROR: Code execution timed out after 600 seconds. Please try ...", "timeout"),
    ],
)
def test_classify_execution_status(result, expected):
    assert A1._classify_execution_status(result) == expected


# ---------------------------------------------------------------------------
# audit emitter
# ---------------------------------------------------------------------------


def _audit_stub():
    """Minimal object that satisfies A1._audit_code_execution's use of self."""
    return types.SimpleNamespace(
        _classify_execution_status=A1._classify_execution_status,
        _audit_code_execution=A1._audit_code_execution,
    )


def test_audit_emits_event_with_expected_fields():
    stream = _capture()
    stub = _audit_stub()
    code = "print('hi')"
    with obs.bind_run(session_id="S", run_id="R"):
        stub._audit_code_execution(
            stub,
            language="python",
            executed_code=code,
            result="hi\n",
            duration_ms=12.5,
            timeout_s=600,
        )
    line = json.loads(stream.getvalue().strip())
    assert line["event"] == "code_execution"
    assert line["language"] == "python"
    assert line["status"] == "ok"
    assert line["code_chars"] == len(code)
    assert line["code_sha256"] == hashlib.sha256(code.encode()).hexdigest()[:12]
    assert len(line["code_sha256"]) == 12
    assert line["duration_ms"] == 12.5
    assert line["timeout_s"] == 600
    assert line["output_chars"] == 3
    assert line["output_truncated"] is False
    # correlation ids flow through
    assert line["session_id"] == "S"
    assert line["run_id"] == "R"


def test_audit_flags_truncated_output():
    stream = _capture()
    stub = _audit_stub()
    stub._audit_code_execution(
        stub,
        language="bash",
        executed_code="echo hi",
        result="x" * 20000,
        duration_ms=1.0,
        timeout_s=600,
    )
    line = json.loads(stream.getvalue().strip())
    assert line["output_truncated"] is True
    assert line["output_chars"] == 20000


def test_audit_reports_timeout_status():
    stream = _capture()
    stub = _audit_stub()
    stub._audit_code_execution(
        stub,
        language="python",
        executed_code="while True: pass",
        result="ERROR: Code execution timed out after 600 seconds. ...",
        duration_ms=600000.0,
        timeout_s=600,
    )
    assert json.loads(stream.getvalue().strip())["status"] == "timeout"


def test_audit_does_not_log_raw_code_or_output():
    stream = _capture()
    stub = _audit_stub()
    secret_code = "patient_id = 'PHI-12345-SENSITIVE'"
    secret_output = "result for PHI-12345-SENSITIVE"
    stub._audit_code_execution(
        stub,
        language="python",
        executed_code=secret_code,
        result=secret_output,
        duration_ms=1.0,
        timeout_s=600,
    )
    out = stream.getvalue()
    assert "PHI-12345-SENSITIVE" not in out  # neither code nor output is logged verbatim


def test_audit_never_raises_on_bad_input():
    _capture()
    stub = _audit_stub()
    # Non-str result must not break the audit path.
    stub._audit_code_execution(
        stub,
        language="python",
        executed_code="x=1",
        result=None,  # type: ignore[arg-type]
        duration_ms=1.0,
        timeout_s=600,
    )
