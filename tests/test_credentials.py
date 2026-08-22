"""Tests for biomni.credentials - keeping provider keys away from generated code.

The agent ``exec``s LLM-written Python in this process, so anything left in
``os.environ`` is readable by whoever writes the prompt. These lock in what the
scrub window hides, what it deliberately leaves alone, that it survives
overlapping executions, and that in-process consumers can still read through it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest
from biomni import credentials


@pytest.fixture(autouse=True)
def _reset_override_cache() -> None:
    """The allow/extra lists are cached on first read; start every test cold.

    They are cached deliberately - see is_credential - so that generated code
    cannot set BIOMNI_SCRUB_ENV_ALLOW inside one window and read the key in the
    next. That makes them process state, and process state has to be reset
    between tests or the first one to touch them decides for the rest.
    """
    # Snapshot the override variables too: they are deliberately *not*
    # credential-shaped (that is the hazard one test below exercises), so a test
    # simulating generated code writes them directly and monkeypatch cannot undo
    # what it did not set.
    saved = {name: os.environ.get(name) for name in (credentials._ALLOW_ENV, credentials._EXTRA_ENV)}
    credentials._name_set.cache_clear()
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    credentials._name_set.cache_clear()


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "SYNAPSE_AUTH_TOKEN",
        "PROTOCOLS_IO_ACCESS_TOKEN",
        "BIOMNI_CUSTOM_API_KEY",
        "DB_PASSWORD",
    ],
)
def test_credential_shaped_names_are_scrubbed(name: str) -> None:
    assert credentials.is_credential(name)


@pytest.mark.parametrize(
    "name",
    [
        # Endpoints and routing config, not secrets: generated analysis code has
        # a legitimate reason to read these, so a prefix-wide AWS_*/AZURE_* block
        # would be too blunt.
        "AWS_REGION",
        "ENDPOINT_URL",
        "DEPLOYMENT_NAME",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "BIOMNI_DATA_PATH",
        # Header *names* configured for the auth gateway - "AUTH" alone must not
        # trigger the classifier.
        "BIOMNI_AUTH_USER_ID_HEADER",
        "BIOMNI_AUTH_EMAIL_HEADER",
    ],
)
def test_non_secret_names_stay_visible(name: str) -> None:
    assert not credentials.is_credential(name)


def test_the_session_signing_secret_is_never_hidden() -> None:
    """Chainlit re-reads it on every JWT operation.

    Hiding it does not protect it - chainlit_ui/persistence.py writes it to
    $BIOMNI_STATE_DIR/threads/auth_secret, so generated code can read it off disk
    either way - but it does break the process: while one chat ran a code step,
    every other user reconnecting or resuming a thread hit `assert secret` and
    got a 500.
    """
    assert not credentials.is_credential("CHAINLIT_AUTH_SECRET")


def test_an_exempt_name_stays_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "signing-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    with credentials.scrubbed_environ():
        assert os.environ.get("CHAINLIT_AUTH_SECRET") == "signing-key"
        assert os.environ.get("ANTHROPIC_API_KEY") is None


def test_the_allowlist_cannot_be_re_armed_at_run_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """A snippet must not be able to disable the scrub for everyone else.

    Neither override variable is credential-shaped, so both stay writable from
    inside a window. Reading them once is what stops
    `os.environ['BIOMNI_SCRUB_ENV_ALLOW']='ANTHROPIC_API_KEY'` in one execution
    from exposing the key in the next - process-wide, for every other chat.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    assert credentials.is_credential("ANTHROPIC_API_KEY")

    # what generated code would attempt, mid-window
    with credentials.scrubbed_environ():
        os.environ["BIOMNI_SCRUB_ENV_ALLOW"] = "ANTHROPIC_API_KEY"

    assert credentials.is_credential("ANTHROPIC_API_KEY"), "the allowlist was re-armed from inside a window"
    with credentials.scrubbed_environ():
        assert os.environ.get("ANTHROPIC_API_KEY") is None


def test_a_timed_out_execution_cannot_leak_the_window_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_with_timeout abandons threads it cannot kill.

    A nested window in an abandoned thread never runs its finally, so a plain
    decrement would leave the process permanently stripped of its credentials -
    including the ones the server itself needs.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    opened = threading.Event()

    def abandoned() -> None:
        # A thread that enters a window and never leaves - what run_with_timeout
        # gives up on when a snippet blocks in C and PyThreadState_SetAsyncExc
        # cannot reach it.
        # Bound to a local, not left as a temporary: an unreferenced context
        # manager is collected as soon as __enter__ returns, and closing the
        # generator runs the very finally this test needs not to run. The frame
        # stays alive while the thread blocks, which is what keeps it open.
        window = credentials.scrubbed_environ()
        window.__enter__()
        opened.set()
        threading.Event().wait(30)
        del window

    worker = threading.Thread(target=abandoned, daemon=True)
    worker.start()
    opened.wait(5)

    assert credentials.scrub_active()
    assert os.environ.get("ANTHROPIC_API_KEY") is None

    released = credentials.release_thread(worker.ident)

    assert released == 1
    assert not credentials.scrub_active(), "window stayed open after the thread was abandoned"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-secret"


def test_releasing_one_thread_leaves_another_thread_window_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two chats, one times out: the other must stay protected.

    Resetting the global depth would have restored the credentials underneath a
    chat still executing model-written code.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    opened = threading.Event()

    def abandoned() -> None:
        # Bound to a local, not left as a temporary: an unreferenced context
        # manager is collected as soon as __enter__ returns, and closing the
        # generator runs the very finally this test needs not to run. The frame
        # stays alive while the thread blocks, which is what keeps it open.
        window = credentials.scrubbed_environ()
        window.__enter__()
        opened.set()
        threading.Event().wait(30)
        del window

    worker = threading.Thread(target=abandoned, daemon=True)
    worker.start()
    opened.wait(5)

    with credentials.scrubbed_environ():
        credentials.release_thread(worker.ident)
        assert credentials.scrub_active(), "the surviving window was closed by the other thread's cleanup"
        assert os.environ.get("ANTHROPIC_API_KEY") is None

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-secret"


def test_extra_and_allow_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOMNI_SCRUB_ENV_EXTRA", "MY_LAB_HANDLE, OTHER_ONE")
    monkeypatch.setenv("BIOMNI_SCRUB_ENV_ALLOW", "PROTOCOLS_IO_ACCESS_TOKEN")

    assert credentials.is_credential("MY_LAB_HANDLE")
    assert credentials.is_credential("other_one")  # case-insensitive
    # The allowlist wins, so a deployment whose analysis code genuinely needs a
    # token-shaped variable is not forced to disable the scrub wholesale.
    assert not credentials.is_credential("PROTOCOLS_IO_ACCESS_TOKEN")


# --------------------------------------------------------------------------- #
# The scrub window
# --------------------------------------------------------------------------- #


def test_window_hides_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    with credentials.scrubbed_environ() as hidden:
        assert "ANTHROPIC_API_KEY" in hidden
        assert os.environ.get("ANTHROPIC_API_KEY") is None
        assert os.environ.get("AWS_REGION") == "us-east-1"

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-secret"


def test_window_survives_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    with pytest.raises(RuntimeError), credentials.scrubbed_environ():
        raise RuntimeError("boom")

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-secret"


def test_assignment_inside_the_window_cannot_poison_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restoration overwrites, so a snippet cannot leave a bogus key behind."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    with credentials.scrubbed_environ():
        os.environ["ANTHROPIC_API_KEY"] = "attacker-supplied"

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-secret"


def test_nested_windows_restore_only_at_the_outermost_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    with credentials.scrubbed_environ():
        with credentials.scrubbed_environ():
            assert os.environ.get("ANTHROPIC_API_KEY") is None
        # Inner exit must not un-hide the key while the outer block runs.
        assert os.environ.get("ANTHROPIC_API_KEY") is None

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-secret"


def test_overlapping_windows_in_threads_do_not_unhide_early(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two concurrent runs share one window; the last one out restores it.

    Regression guard for the reference counting: if each thread restored on its
    own exit, the shorter run would re-expose the key while the longer one was
    still executing model-written code.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    first_open = threading.Event()
    second_done = threading.Event()
    seen: dict[str, object] = {}

    def short_run() -> None:
        first_open.wait(5)
        with credentials.scrubbed_environ():
            pass
        second_done.set()

    def long_run() -> None:
        with credentials.scrubbed_environ():
            first_open.set()
            second_done.wait(5)
            # The other thread has finished its window; the key must still be
            # hidden because this run is still inside its own.
            seen["during"] = os.environ.get("ANTHROPIC_API_KEY")

    threads = [threading.Thread(target=long_run), threading.Thread(target=short_run)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert seen["during"] is None
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-secret"


def test_no_window_leaves_environ_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    assert not credentials.scrub_active()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-secret"


# --------------------------------------------------------------------------- #
# Reading through the window
# --------------------------------------------------------------------------- #


def test_getenv_sees_through_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-process consumers (the LLM factory, the readiness probe) must not break.

    A chat opened while another chat executes code still has to build a working
    client, and a probe firing in that window must not report the pod unready.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    with credentials.scrubbed_environ():
        assert credentials.getenv("ANTHROPIC_API_KEY") == "sk-ant-secret"
        assert credentials.getenv("NOT_SET_AT_ALL", "fallback") == "fallback"

    assert credentials.getenv("ANTHROPIC_API_KEY") == "sk-ant-secret"


def test_subprocess_children_do_not_inherit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stripping the real mapping - not a filtered copy - is what covers this.

    A snippet that shells out would otherwise read the key straight out of the
    child's environment, no matter what the ``exec`` globals contained.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    script = "import os; print(os.environ.get('ANTHROPIC_API_KEY', '')); print(os.environ.get('AWS_REGION', ''))"
    with credentials.scrubbed_environ():
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout

    secret_line, region_line = out.splitlines()
    assert secret_line == ""
    assert region_line == "us-east-1"
