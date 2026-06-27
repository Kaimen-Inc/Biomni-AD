"""Tests for biomni.observability — structured logging, correlation, redaction.

These pin the behaviour the AKS deployment relies on: JSON lines with
correlation ids, secret/PII scrubbing at the format chokepoint, idempotent
setup, and contextvar propagation into worker threads.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from biomni import observability as obs


@pytest.fixture(autouse=True)
def _restore_root_logging():
    """Snapshot/restore the root logger so setup_logging() can't leak between tests."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def _capture(level="DEBUG", fmt="json"):
    """Configure logging to an in-memory stream and return (stream, handler)."""
    import io

    stream = io.StringIO()
    handler = obs.setup_logging(level, fmt=fmt, stream=stream, force=True)
    return stream, handler


# ---------------------------------------------------------------------------
# Redactor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUV",
        "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX",
        "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
        "AKIAIOSFODNN7EXAMPLE",
        "AIzaSyA1234567890abcdefghijklmnopqrstuv1",
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "xoxb-123456789012-ABCDEFGHIJKL",
        "Bearer abc123.def456-ghi789",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36",
    ],
)
def test_redactor_scrubs_known_key_shapes(raw):
    out = obs.Redactor().redact(f"prefix {raw} suffix")
    assert raw not in out
    assert obs.REDACTED in out or "[REDACTED" in out


def test_redactor_scrubs_known_env_secret_values():
    r = obs.Redactor(["my-custom-deploy-token-XYZ"])
    out = r.redact("the token is my-custom-deploy-token-XYZ here")
    assert "my-custom-deploy-token-XYZ" not in out
    assert obs.REDACTED in out


def test_redactor_ignores_short_env_values():
    # Short values would cause collateral redaction of unrelated log text.
    r = obs.Redactor(["abc", ""])
    assert r.redact("abc def") == "abc def"


def test_redactor_collapses_base64_data_uri():
    blob = "data:image/png;base64," + "A" * 5000
    out = obs.Redactor().redact(f"plot={blob}")
    assert "AAAA" not in out
    assert "data:[REDACTED base64]" in out
    assert len(out) < 200


def test_redactor_emails_opt_in():
    assert obs.Redactor().redact("a@b.com") == "a@b.com"
    assert obs.Redactor(redact_emails=True).redact("a@b.com") == "[REDACTED-EMAIL]"


def test_redactor_from_environ_harvests_secret_named_vars(monkeypatch):
    monkeypatch.setenv("MY_SERVICE_TOKEN", "longsecretvalue123")
    monkeypatch.setenv("PLAIN_SETTING", "not-secret-but-longish")
    r = obs.Redactor.from_environ()
    out = r.redact("tok=longsecretvalue123 cfg=not-secret-but-longish")
    assert "longsecretvalue123" not in out
    assert "not-secret-but-longish" in out  # non-secret name untouched


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


def test_json_line_has_core_fields_and_context():
    stream, _ = _capture()
    log = logging.getLogger("biomni.test")
    with obs.bind_run(session_id="S1", run_id="R1"):
        log.info("hello")
    line = json.loads(stream.getvalue().strip())
    assert line["level"] == "INFO"
    assert line["logger"] == "biomni.test"
    assert line["msg"] == "hello"
    assert line["session_id"] == "S1"
    assert line["run_id"] == "R1"
    assert line["ts"].endswith("+00:00")  # UTC


def test_json_omits_unset_context():
    stream, _ = _capture()
    logging.getLogger("biomni.test").info("no-context")
    line = json.loads(stream.getvalue().strip())
    assert "session_id" not in line
    assert "run_id" not in line


def test_json_includes_extra_fields():
    stream, _ = _capture()
    logging.getLogger("biomni.test").info("evt", extra={"language": "python", "duration_ms": 5})
    line = json.loads(stream.getvalue().strip())
    assert line["language"] == "python"
    assert line["duration_ms"] == 5


def test_json_includes_exception_traceback():
    stream, _ = _capture()
    log = logging.getLogger("biomni.test")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("failed")
    line = json.loads(stream.getvalue().strip())
    assert "ValueError: boom" in line["exc"]


def test_json_redacts_secret_in_message(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secretkey-ABCDEFGHIJKL")
    stream, _ = _capture()
    logging.getLogger("biomni.test").warning("auth failed for sk-ant-secretkey-ABCDEFGHIJKL")
    out = stream.getvalue()
    assert "sk-ant-secretkey-ABCDEFGHIJKL" not in out
    assert obs.REDACTED in out


def test_human_format_is_not_json_but_redacts():
    stream, _ = _capture(fmt="text")
    logging.getLogger("biomni.test").info("hi sk-ant-secretkey-ABCDEFGHIJKL")
    out = stream.getvalue()
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)
    assert "sk-ant-secretkey-ABCDEFGHIJKL" not in out


# ---------------------------------------------------------------------------
# setup_logging behaviour
# ---------------------------------------------------------------------------


def test_setup_logging_is_idempotent():
    import io

    s1 = io.StringIO()
    h1 = obs.setup_logging("INFO", stream=s1, force=True)
    managed = [h for h in logging.getLogger().handlers if getattr(h, "_biomni_managed", False)]
    assert len(managed) == 1
    # Second call without force keeps the same handler (no duplicate output).
    h2 = obs.setup_logging("DEBUG", stream=io.StringIO())
    assert h2 is h1
    managed = [h for h in logging.getLogger().handlers if getattr(h, "_biomni_managed", False)]
    assert len(managed) == 1
    # But level is kept in sync.
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_level_from_env(monkeypatch):
    import io

    monkeypatch.setenv("LOG_LEVEL", "warning")
    obs.setup_logging(stream=io.StringIO(), force=True)
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_quiets_noisy_loggers():
    import io

    obs.setup_logging("DEBUG", stream=io.StringIO(), force=True)
    assert logging.getLogger("httpx").level == logging.WARNING


# ---------------------------------------------------------------------------
# Correlation context
# ---------------------------------------------------------------------------


def test_bind_run_nesting_and_reset():
    assert obs.get_context() == {}
    with obs.bind_run(session_id="S"):
        assert obs.get_context() == {"session_id": "S"}
        with obs.bind_run(run_id="R"):
            assert obs.get_context() == {"session_id": "S", "run_id": "R"}
        # inner reset restores only run_id
        assert obs.get_context() == {"session_id": "S"}
    assert obs.get_context() == {}


def test_capture_context_propagates_into_worker_thread():
    with obs.bind_run(session_id="S9", run_id="R9"):
        # Captured in the caller thread (where the ids are bound).
        ctx = obs.capture_context()
        with ThreadPoolExecutor(max_workers=1) as ex:
            # Raw submit: worker thread has its own empty context -> ids lost.
            raw = ex.submit(obs.get_context).result()
            # Replayed captured context: ids cross the thread boundary.
            replayed = ex.submit(ctx.run, obs.get_context).result()
    assert raw == {}
    assert replayed == {"session_id": "S9", "run_id": "R9"}


def test_set_run_id_and_session_id_with_token_reset():
    sid_token = obs.set_session_id("S")
    rid_token = obs.set_run_id("R")
    assert obs.get_context() == {"session_id": "S", "run_id": "R"}
    obs.run_id_var.reset(rid_token)
    obs.session_id_var.reset(sid_token)
    assert obs.get_context() == {}


def test_capture_context_inside_worker_captures_nothing():
    # Guards the bug class: copy_context() called in the worker snapshots the
    # worker's empty context, so ids do NOT propagate that way.
    with obs.bind_run(session_id="S", run_id="R"):
        with ThreadPoolExecutor(max_workers=1) as ex:
            inner = ex.submit(lambda: obs.capture_context().run(obs.get_context)).result()
    assert inner == {}


# ---------------------------------------------------------------------------
# emit_event / telemetry helpers
# ---------------------------------------------------------------------------


def test_emit_event_writes_structured_line():
    stream, _ = _capture()
    obs.emit_event("code_execution", language="r", status="ok")
    line = json.loads(stream.getvalue().strip())
    assert line["event"] == "code_execution"
    assert line["msg"] == "code_execution"
    assert line["language"] == "r"
    assert line["status"] == "ok"


def test_emit_event_sanitizes_reserved_keys():
    stream, _ = _capture()
    obs.emit_event("evt", module="x", name="y", duration_ms=1)
    line = json.loads(stream.getvalue().strip())
    # Reserved LogRecord names get a trailing underscore rather than crashing.
    assert line["module_"] == "x"
    assert line["name_"] == "y"
    assert line["duration_ms"] == 1


def test_json_formatter_never_raises_on_hostile_field():
    stream, _ = _capture()

    class Hostile:
        def __str__(self):
            raise RuntimeError("nope")

        __repr__ = __str__

    # Must not raise, and must still emit a line (degraded, flagged for triage).
    logging.getLogger("biomni.test").info("evt", extra={"bad": Hostile()})
    line = json.loads(stream.getvalue().strip())
    assert line["log_format_error"] is True
    assert line["msg"] == "evt"
    assert line["level"] == "INFO"


def test_emit_event_never_raises():
    # Unserializable value still must not blow up the caller; default=str saves it.
    obs.setup_logging("INFO", stream=__import__("io").StringIO(), force=True)
    obs.emit_event("evt", weird=object())  # should not raise


def test_diff_usage_summary_computes_per_run_delta():
    before = {
        "calls": 2,
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "total_tokens": 150,
        "cache_hit_ratio": 0.0,
    }
    after = {
        "calls": 5,
        "input_tokens": 100,
        "output_tokens": 90,
        "cache_read_tokens": 300,
        "cache_creation_tokens": 10,
        "total_tokens": 490,
        "cache_hit_ratio": 0.5,
    }
    delta = obs.diff_usage_summary(before, after)
    assert delta["calls"] == 3
    assert delta["input_tokens"] == 0
    assert delta["output_tokens"] == 40
    assert delta["cache_read_tokens"] == 300
    # ratio recomputed from delta: 300 / (300 + 0) == 1.0
    assert delta["cache_hit_ratio"] == 1.0


def test_log_llm_usage_emits_event_with_summary_fields():
    stream, _ = _capture()
    obs.log_llm_usage({"calls": 3, "total_tokens": 1234}, model="claude-sonnet-4-5", latency_ms=42)
    line = json.loads(stream.getvalue().strip())
    assert line["event"] == "llm_usage"
    assert line["calls"] == 3
    assert line["total_tokens"] == 1234
    assert line["model"] == "claude-sonnet-4-5"
    assert line["latency_ms"] == 42


# ---------------------------------------------------------------------------
# RunHeartbeat
# ---------------------------------------------------------------------------


def _heartbeats(stream):
    out = []
    for line in stream.getvalue().splitlines():
        if line.strip():
            obj = json.loads(line)
            if obj.get("event") == "run_heartbeat":
                out.append(obj)
    return out


def test_heartbeat_emits_periodically_with_elapsed_and_context():
    stream, _ = _capture()
    with obs.bind_run(session_id="S", run_id="R"), obs.RunHeartbeat(interval=0.05, status=lambda: {"step": 7}):
        time.sleep(0.22)
    beats = _heartbeats(stream)
    assert len(beats) >= 2  # several ticks within the window
    assert beats[0]["session_id"] == "S" and beats[0]["run_id"] == "R"
    assert beats[0]["step"] == 7
    assert "elapsed_ms" in beats[0]


def test_heartbeat_stops_after_exit():
    stream, _ = _capture()
    with obs.RunHeartbeat(interval=0.05):
        time.sleep(0.12)
    count_at_exit = len(_heartbeats(stream))
    time.sleep(0.2)
    assert len(_heartbeats(stream)) == count_at_exit  # no ticks after __exit__


def test_heartbeat_survives_failing_status_callback():
    stream, _ = _capture()

    def boom():
        raise RuntimeError("status failed")

    with obs.RunHeartbeat(interval=0.05, status=boom):
        time.sleep(0.12)
    # Still emits (status failure is swallowed; elapsed_ms always present).
    beats = _heartbeats(stream)
    assert beats and "elapsed_ms" in beats[0]


def test_heartbeat_zero_interval_does_not_busy_loop():
    # interval<=0 must be floored, not spin. Bounded ticks in a short window.
    stream, _ = _capture()
    with obs.RunHeartbeat(interval=0):
        time.sleep(0.12)
    assert len(_heartbeats(stream)) < 100


def test_heartbeat_is_not_reentrant():
    hb = obs.RunHeartbeat(interval=0.05)
    with hb:
        with pytest.raises(RuntimeError, match="not reentrant"):
            hb.__enter__()
