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
        "CHAINLIT_AUTH_SECRET",
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
