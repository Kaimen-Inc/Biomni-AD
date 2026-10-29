"""BIOMNI_HIDE_LOGOUT, in the real app: the page a browser receives names the stylesheet that hides the item.

Chainlit has no setting for the user menu's Logout item, so it is hidden by a
stylesheet chosen once at startup (chainlit_ui.ui_flags.custom_stylesheet). What
is checked here is the part a unit test cannot see: that the real app, started
with the setting on, tells the browser to load that stylesheet in place of the
usual one, and serves both. That the selectors still match Chainlit's markup is
a matter for a browser - see public/biomni-hide-logout.css.

Driven by ui_flags_app.py in an interpreter of its own. Skipped where the
chainlit extra is not installed, as in CI.
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
pytest.importorskip("aiosqlite")

REPO = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).with_name("ui_flags_app.py")
RESULT = "UI_FLAGS_RESULT "


@pytest.fixture(scope="module")
def checks(tmp_path_factory: pytest.TempPathFactory) -> list[list]:
    root = tmp_path_factory.mktemp("ui_flags")
    approot = root / "approot"
    shutil.copytree(REPO / ".chainlit", approot / ".chainlit")
    shutil.copytree(REPO / "public", approot / "public")
    for name in ("workspace", "outputs", "state"):
        (root / name).mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BIOMNI_", "CHAINLIT_"))}
    env.update(
        CHAINLIT_APP_ROOT=str(approot),
        BIOMNI_HIDE_LOGOUT="true",
        BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE="true",
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


def test_the_page_loads_the_logout_hiding_stylesheet(checks: list[list]) -> None:
    failed = [f"{label}: got {got!r}, want {want!r}" for label, got, want in checks if got != want]
    assert not failed, "\n".join(failed)
    assert len(checks) == 6, "a check stopped running"
