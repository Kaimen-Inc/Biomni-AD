"""Tests for chainlit_ui.planning.build_planning_system_prompt.

Only the pure function is exercised here — `interactive_planning` itself
calls into Chainlit's async UI primitives and isn't reachable outside a
running Chainlit session.

The Chainlit module is stubbed at import time so this test runs in any
environment that has the rest of `biomni` installed.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _stub_chainlit() -> object:
    """Inject a minimal `chainlit` shim so planning.py imports succeed.

    Yields and removes the stub on teardown so the entry doesn't leak
    across the rest of the pytest session — another test that wants the
    real chainlit (or no chainlit at all) shouldn't pick up our lambdas.
    """
    if "chainlit" in sys.modules:
        # Caller already provides chainlit (real or stubbed elsewhere); leave alone.
        yield None
        return

    cl = types.ModuleType("chainlit")
    for name in ("Step", "AskActionMessage", "AskUserMessage", "Action"):
        setattr(cl, name, lambda *a, **k: None)
    sys.modules["chainlit"] = cl
    try:
        yield cl
    finally:
        sys.modules.pop("chainlit", None)


def _import_planning():
    import importlib

    return importlib.import_module("chainlit_ui.planning")


class _Agent:
    """Minimal agent stub — only the attributes the function reads."""

    user_data_inventory: str | None = None
    data_root_dir: str | None = None


def test_a1_prompt_returns_general_template() -> None:
    planning = _import_planning()
    out = planning.build_planning_system_prompt(_Agent(), "a1")
    assert out.startswith("You are a biomedical research assistant")
    assert "Alzheimer" not in out


def test_ad1_prompt_returns_ad_template() -> None:
    planning = _import_planning()
    out = planning.build_planning_system_prompt(_Agent(), "ad1")
    assert "expert Alzheimer" in out
    assert "LOCAL-FIRST RULE" in out


def test_unknown_agent_type_defaults_to_general() -> None:
    planning = _import_planning()
    out = planning.build_planning_system_prompt(_Agent(), "anything-else")
    assert out.startswith("You are a biomedical research assistant")


def test_inventory_is_appended_when_present() -> None:
    planning = _import_planning()
    agent = _Agent()
    agent.user_data_inventory = "trem2.csv\napoe.csv"
    agent.data_root_dir = "/srv/user-data"

    out = planning.build_planning_system_prompt(agent, "a1")

    assert "/srv/user-data" in out
    assert "trem2.csv" in out
    assert "apoe.csv" in out


def test_empty_inventory_is_not_appended() -> None:
    planning = _import_planning()
    agent = _Agent()
    agent.user_data_inventory = ""
    agent.data_root_dir = "/data"

    out = planning.build_planning_system_prompt(agent, "a1")
    assert "user data directory" not in out


def test_missing_attributes_are_safe() -> None:
    planning = _import_planning()

    class Bare:
        """Agent with no inventory / data_root_dir attributes at all."""

    out = planning.build_planning_system_prompt(Bare(), "a1")
    # Should still return the base prompt without crashing on getattr
    assert "biomedical research assistant" in out
