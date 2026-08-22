"""Tests for per-session isolation of the generated-code REPL.

The Chainlit app serves several chats from one process, and every chat executes
LLM-written Python in it. Before these guarantees existed, three module-level
globals were shared by the whole process, so two people asking a question at the
same moment collided: run A received run B's printed output and B's
``OUTPUT_DIR`` (sending A's generated files into B's run directory), and because
each execution restored whatever ``sys.stdout`` it found rather than the one it
replaced, a single overlap left the process writing into a discarded buffer and
silently swallowed every later print.

These lock in the isolation that makes "one deployment, many concurrent chats"
safe, and the credential scrub around ``exec``.
"""

from __future__ import annotations

import os
import sys
import threading

import pytest
from biomni.observability import bind_run
from biomni.tool import support_tools
from biomni.tool.support_tools import get_repl_namespace, run_python_repl
from biomni.utils import run_with_timeout


@pytest.fixture(autouse=True)
def _clean_sessions() -> None:
    """Every test starts from empty REPL state."""
    with support_tools._sessions_lock:
        support_tools._sessions.clear()
    yield
    with support_tools._sessions_lock:
        support_tools._sessions.clear()


# --------------------------------------------------------------------------- #
# Session scoping
# --------------------------------------------------------------------------- #


def test_namespace_persists_within_a_session() -> None:
    """The "variables persist between calls" contract still holds per chat."""
    with bind_run(session_id="chat-1"):
        run_python_repl("x = 41")
        assert run_python_repl("print(x + 1)") == "42\n"


def test_namespace_is_not_shared_across_sessions() -> None:
    with bind_run(session_id="chat-1"):
        run_python_repl("secret_var = 'chat-1 data'")

    with bind_run(session_id="chat-2"):
        out = run_python_repl("print(secret_var)")

    assert "chat-1 data" not in out
    assert "Error" in out


def test_unbound_callers_share_one_default_session() -> None:
    """Notebooks and the CLI keep the old single-namespace behaviour."""
    run_python_repl("notebook_var = 7")
    assert run_python_repl("print(notebook_var)") == "7\n"


def test_output_dir_injection_is_per_session() -> None:
    """A concurrent run must not redirect this run's generated files.

    ``A1._inject_custom_functions_to_repl`` writes ``OUTPUT_DIR`` here before
    each execution; on a shared namespace the later writer won for everyone.
    """
    with bind_run(session_id="chat-1"):
        get_repl_namespace()["OUTPUT_DIR"] = "/runs/user-one"
    with bind_run(session_id="chat-2"):
        get_repl_namespace()["OUTPUT_DIR"] = "/runs/user-two"

    with bind_run(session_id="chat-1"):
        assert run_python_repl("print(OUTPUT_DIR)") == "/runs/user-one\n"


def test_discard_drops_only_the_named_session() -> None:
    with bind_run(session_id="chat-1"):
        run_python_repl("keep_me = 1")
    with bind_run(session_id="chat-2"):
        run_python_repl("keep_me = 2")

    support_tools.discard_repl_session("chat-1")

    with bind_run(session_id="chat-2"):
        assert run_python_repl("print(keep_me)") == "2\n"
    with bind_run(session_id="chat-1"):
        assert "Error" in run_python_repl("print(keep_me)")


def test_session_registry_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A closed tab never says goodbye, so growth is capped rather than trusted."""
    monkeypatch.setattr(support_tools, "_MAX_SESSIONS", 3)

    for i in range(5):
        with bind_run(session_id=f"chat-{i}"):
            run_python_repl(f"v = {i}")

    with support_tools._sessions_lock:
        assert len(support_tools._sessions) == 3
        # Least-recently-used entries are the ones evicted.
        assert set(support_tools._sessions) == {"chat-2", "chat-3", "chat-4"}


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_sessions_do_not_cross_capture_output() -> None:
    """The regression that made this whole module necessary.

    Both runs go through ``run_with_timeout`` exactly as ``A1``'s execute node
    does, so this also covers the contextvar propagation into its worker thread
    - without it both runs would land in the shared default session.
    """
    results: dict[str, str] = {}
    started = threading.Barrier(2, timeout=10)

    def chat(name: str, out_dir: str, marker: str) -> None:
        with bind_run(session_id=f"chat-{name}", run_id=f"run-{name}"):
            get_repl_namespace()["OUTPUT_DIR"] = out_dir
            started.wait()
            results[name] = run_with_timeout(
                run_python_repl,
                [
                    "import time\n"
                    f"my_var = {marker!r}\n"
                    "time.sleep(0.3)\n"
                    "print('OUTPUT_DIR =', OUTPUT_DIR)\n"
                    "print('my_var =', my_var)\n"
                ],
                timeout=30,
            )

    threads = [
        threading.Thread(target=chat, args=("a", "/runs/user-a", "a-data")),
        threading.Thread(target=chat, args=("b", "/runs/user-b", "b-data")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert results["a"] == "OUTPUT_DIR = /runs/user-a\nmy_var = a-data\n"
    assert results["b"] == "OUTPUT_DIR = /runs/user-b\nmy_var = b-data\n"


def test_overlapping_execution_leaves_process_stdout_usable(capsys: pytest.CaptureFixture) -> None:
    """Routing, not swapping: no execution can strand ``sys.stdout``.

    The old code restored the handle it found on entry, so two overlapping runs
    left ``sys.stdout`` pointing at a dead buffer for the rest of the process.
    """
    before = sys.stdout

    def chat(name: str) -> None:
        with bind_run(session_id=f"chat-{name}"):
            run_python_repl("import time; time.sleep(0.2); print('inside')")

    threads = [threading.Thread(target=chat, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    print("visible after execution")
    assert "visible after execution" in capsys.readouterr().out
    # Either untouched, or wrapped by the router that forwards to it.
    assert sys.stdout is before or getattr(sys.stdout, "fallback", None) is before


def test_plots_are_captured_per_session() -> None:
    with bind_run(session_id="chat-1"):
        support_tools._session().plots.append("data:image/png;base64,AAAA")
        assert support_tools.get_captured_plots() == ["data:image/png;base64,AAAA"]

    with bind_run(session_id="chat-2"):
        assert support_tools.get_captured_plots() == []
        support_tools.clear_captured_plots()

    with bind_run(session_id="chat-1"):
        assert support_tools.get_captured_plots() == ["data:image/png;base64,AAAA"]


# --------------------------------------------------------------------------- #
# Credentials and error reporting
# --------------------------------------------------------------------------- #


def test_generated_code_cannot_read_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    out = run_python_repl(
        "import os\nprint(repr(os.environ.get('ANTHROPIC_API_KEY')))\nprint(os.environ['AWS_REGION'])"
    )

    assert out == "None\nus-east-1\n"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-secret"


def test_credentials_are_restored_when_generated_code_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    assert "Error" in run_python_repl("raise ValueError('boom')")
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-secret"


def test_output_printed_before_a_failure_is_kept() -> None:
    """The agent uses it to work out which step broke."""
    out = run_python_repl("print('step one done')\nraise ValueError('boom')")
    assert out == "step one done\nError: boom"


def test_logging_setup_never_binds_the_repl_router() -> None:
    """A log record emitted during a code step must not land in that code's output.

    ``setup_logging`` binds whatever ``sys.stdout`` is at the time. If that were
    the router, records emitted while generated code ran would be captured into
    the snippet's buffer - surfaced to the user as their own program's output,
    and lost from the log pipeline.
    """
    import logging

    from biomni.observability import setup_logging

    real_stdout = sys.stdout
    support_tools._ensure_stdout_router()
    assert isinstance(sys.stdout, support_tools._StdoutRouter)

    try:
        setup_logging(force=True)
        handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)]
        assert handlers, "expected setup_logging to install a StreamHandler"
        for handler in handlers:
            assert not isinstance(handler.stream, support_tools._StdoutRouter)
            assert handler.stream is real_stdout
    finally:
        sys.stdout = real_stdout
        logging.getLogger().handlers.clear()
