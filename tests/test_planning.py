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
    across the rest of the pytest session. We also drop the cached
    `chainlit_ui.planning` module — once it's been imported its
    module-scope `cl` name is bound to the stub, so a later test that
    re-imports planning would still see the stub unless we force a
    fresh import.
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
        sys.modules.pop("chainlit_ui.planning", None)


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


# ---------------------------------------------------------------------------
# extract_planned_data_files
# ---------------------------------------------------------------------------


def _extract(text: str):
    return _import_planning().extract_planned_data_files(text)


def _heading():
    return _import_planning().DATA_FILES_HEADING


def test_extract_reads_the_declared_files():
    plan = f"1. Load data\n2. Model it\n\n{_heading()}\n- studyA/a.csv\n- studyB/b.tsv\n"
    assert _extract(plan) == ["studyA/a.csv", "studyB/b.tsv"]


def test_extract_unwraps_backticks_and_bold_heading():
    plan = f"**{_heading()}**\n- `studyA/a.csv`\n"
    assert _extract(plan) == ["studyA/a.csv"]


def test_extract_accepts_other_bullet_styles():
    plan = f"{_heading()}:\n* one.csv\n1. two.csv\n+ three.csv\n"
    assert _extract(plan) == ["one.csv", "two.csv", "three.csv"]


def test_extract_treats_none_as_no_files():
    for marker in ("none", "None", "n/a", "(none)"):
        assert _extract(f"{_heading()}\n- {marker}\n") == []


def test_extract_returns_empty_without_the_section():
    assert _extract("1. Just a plan with no file section") == []
    assert _extract("") == []


def test_extract_stops_at_the_end_of_the_bullet_block():
    plan = f"{_heading()}\n- a.csv\n\nSome trailing prose\n- not-a-file\n"
    assert _extract(plan) == ["a.csv"]


def test_extract_tolerates_a_blank_line_after_the_heading():
    plan = f"{_heading()}\n\n- a.csv\n"
    assert _extract(plan) == ["a.csv"]


def test_extract_deduplicates_preserving_order():
    plan = f"{_heading()}\n- a.csv\n- b.csv\n- a.csv\n"
    assert _extract(plan) == ["a.csv", "b.csv"]


def test_planning_prompt_requests_the_file_section():
    planning = _import_planning()

    class _Agent:
        user_data_inventory = "studyA/a.csv"
        data_root_dir = "/workspace"

    prompt = planning.build_planning_system_prompt(_Agent(), "a1")
    assert planning.DATA_FILES_HEADING in prompt
    # Must demand real files and forbid the directory/placeholder listings the
    # model produced before, rather than inviting a section that says nothing.
    assert "omit the section entirely" in prompt
    assert "Never list a directory" in prompt


def test_extract_matches_a_numbered_heading():
    """The docstring promises numbered headings work; \\W* would not match digits."""
    plan = f"4. {_heading()}\n- a.csv\n"
    assert _extract(plan) == ["a.csv"]


def test_extract_matches_a_markdown_heading():
    plan = f"### {_heading()}\n- a.csv\n"
    assert _extract(plan) == ["a.csv"]


# ---------------------------------------------------------------------------
# Placeholder filtering: a list of directories is worse than no list
# ---------------------------------------------------------------------------


def test_extract_drops_directory_entries():
    plan = f"{_heading()}\n- /ws/studyA/\n- /ws/studyB\n- /ws/studyA/real.csv\n"
    assert _extract(plan) == ["/ws/studyA/real.csv"]


def test_extract_drops_to_be_discovered_placeholders():
    plan = (
        f"{_heading()}\n"
        "- /ws/GCST90027158/ (files to be discovered in Step 1)\n"
        "- /ws/NG00105-eQTL/ (files to be discovered in Step 1)\n"
    )
    assert _extract(plan) == []


def test_extract_drops_hedged_and_glob_entries():
    plan = f"{_heading()}\n- /ws/a/NG00102.csv (if available)\n- /ws/a/*.csv\n- /ws/a/kept.tsv\n"
    assert _extract(plan) == ["/ws/a/kept.tsv"]


def test_extract_keeps_plain_concrete_paths():
    plan = f"{_heading()}\n- studyA/plasma_1.csv\n- /abs/path/data.tsv.gz\n"
    assert _extract(plan) == ["studyA/plasma_1.csv", "/abs/path/data.tsv.gz"]
