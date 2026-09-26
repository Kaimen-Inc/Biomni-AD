"""The login guard, in the real app: a cookie is only good for the user the gateway names.

Chainlit signs a browser in once, from the gateway's headers, and then believes
its cookie for 15 days. The guard (chainlit_app.gateway_checked_user) checks the
cookie against the headers on every request, so a browser someone else used, or
a cookie the previous release issued to every visitor under one shared name,
cannot open another user's settings and conversations.

Driven through Chainlit's own HTTP API by login_guard_app.py, in an interpreter
of its own. Skipped where the chainlit extra is not installed, as in CI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("chainlit")
pytest.importorskip("aiosqlite")  # the chat-history store the conversation checks use

REPO = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).with_name("login_guard_app.py")
RESULT = "LOGIN_GUARD_RESULT "


@pytest.fixture(scope="module")
def checks(tmp_path_factory: pytest.TempPathFactory) -> list[list]:
    root = tmp_path_factory.mktemp("login_guard")
    approot = root / "approot"
    shutil.copytree(REPO / ".chainlit", approot / ".chainlit")
    for name in ("workspace", "outputs", "state"):
        (root / name).mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BIOMNI_", "CHAINLIT_"))}
    env.update(
        CHAINLIT_APP_ROOT=str(approot),
        BIOMNI_TRUST_AUTH_HEADERS="true",
        BIOMNI_DATA_PATH=str(root / "workspace"),
        BIOMNI_OUTPUT_ROOT=str(root / "outputs"),
        BIOMNI_STATE_DIR=str(root / "state"),
        BIOMNI_LOG_FORMAT="text",
        LOG_LEVEL="ERROR",
    )
    done = subprocess.run(
        [sys.executable, str(DRIVER)], env=env, cwd=approot, capture_output=True, text=True, timeout=300
    )
    results = [line for line in done.stdout.splitlines() if line.startswith(RESULT)]
    assert done.returncode == 0 and results, f"the app did not run:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    return json.loads(results[-1][len(RESULT) :])["checks"]


def test_a_cookie_is_only_good_for_the_user_the_gateway_names(checks: list[list]) -> None:
    failed = [f"{label}: got {got!r}, want {want!r}" for label, got, want in checks if got != want]
    assert not failed, "\n".join(failed)
    assert len(checks) == 18, "a check stopped running"
