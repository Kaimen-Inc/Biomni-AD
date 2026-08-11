"""Tests for the agent side of workspace scoping and output resolution.

`A1` is expensive to construct (it builds an LLM client and a LangGraph), so
these bind the two methods under test to a lightweight stub. That is enough:
both read only attributes the UI sets on the agent.

The agent module pulls in the heavy stack (pandas, langchain, langgraph); the
whole module skips when that is not installed rather than erroring at collection.
"""

from __future__ import annotations

import os

import pytest
from biomni import fs_scan

try:
    from biomni.agent.a1 import A1

    HAS_AGENT = True
except Exception:  # pragma: no cover - depends on the installed extras
    HAS_AGENT = False

pytestmark = pytest.mark.skipif(not HAS_AGENT, reason="agent stack (pandas/langchain/langgraph) not installed")

# BIOMNI_USER_DATA_PATH / BIOMNI_DATA_PATH matter here: biomni.agent.a1 calls
# load_dotenv() at import, so a developer's .env would otherwise leak in and these
# tests would assert different things locally than in CI.
_ENV_TO_CLEAR = (
    "BIOMNI_OUTPUT_ROOT",
    "BIOMNI_DEFAULT_SCOPE_PATHS",
    "BIOMNI_USER_DATA_PATH",
    "BIOMNI_DATA_PATH",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    fs_scan.clear_cache()
    yield
    fs_scan.clear_cache()


class _StubAgent:
    """Minimal stand-in carrying only the attributes the methods read."""

    def __init__(self, **attrs):
        self.data_lake_dir = ""
        for key, value in attrs.items():
            setattr(self, key, value)

    _resolve_runs_root = A1._resolve_runs_root if HAS_AGENT else None
    _get_user_data_resources = A1._get_user_data_resources if HAS_AGENT else None


@pytest.fixture
def workspace(tmp_path):
    for folder, count in (("studyA", 2), ("bigdump", 5)):
        (tmp_path / folder).mkdir()
        for i in range(count):
            (tmp_path / folder / f"f{i}.csv").write_text("x")
    return tmp_path


# --------------------------------------------------------------------------- #
# Output directory
# --------------------------------------------------------------------------- #


def test_runs_root_prefers_what_the_ui_set(tmp_path):
    agent = _StubAgent(runs_root=str(tmp_path / "from-ui"), data_root_dir=str(tmp_path))
    assert agent._resolve_runs_root() == str(tmp_path / "from-ui")


def test_runs_root_falls_back_to_env_output_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMNI_OUTPUT_ROOT", str(tmp_path / "volume"))
    agent = _StubAgent(data_root_dir=str(tmp_path))
    assert agent._resolve_runs_root() == str(tmp_path / "volume")


def test_runs_root_falls_back_to_cwd_without_a_data_root():
    agent = _StubAgent(data_root_dir=None)
    assert agent._resolve_runs_root() == os.path.abspath(os.path.join(os.getcwd(), "runs"))


# --------------------------------------------------------------------------- #
# Retriever scope
# --------------------------------------------------------------------------- #


def test_retriever_indexes_the_whole_root_without_a_scope(workspace):
    agent = _StubAgent(data_root_dir=str(workspace), scope_roots=[])
    names = {r["name"] for r in agent._get_user_data_resources()}
    assert any("studyA" in n for n in names)
    assert any("bigdump" in n for n in names)


def test_retriever_honours_the_selected_scope(workspace):
    agent = _StubAgent(data_root_dir=str(workspace), scope_roots=[str(workspace / "studyA")])
    names = {r["name"] for r in agent._get_user_data_resources()}
    assert names
    assert all("bigdump" not in n for n in names)
    # Names stay relative to the data root so downstream path resolution is unchanged.
    assert all(n.startswith("user-data:studyA/") for n in names)


def test_retriever_ignores_a_scope_entry_that_no_longer_exists(workspace):
    agent = _StubAgent(
        data_root_dir=str(workspace),
        scope_roots=[str(workspace / "studyA"), str(workspace / "ghost")],
    )
    names = {r["name"] for r in agent._get_user_data_resources()}
    assert names and all("ghost" not in n for n in names)


def test_retriever_splits_its_budget_across_selected_folders(workspace):
    agent = _StubAgent(
        data_root_dir=str(workspace),
        scope_roots=[str(workspace / "studyA"), str(workspace / "bigdump")],
    )
    resources = agent._get_user_data_resources(max_items=4)
    # Two folders, budget 4 -> at most 2 from each, and neither is starved out.
    names = {r["name"] for r in resources}
    assert any("studyA" in n for n in names)
    assert any("bigdump" in n for n in names)
    assert len(resources) <= 4


def test_retriever_is_empty_without_a_data_root():
    assert _StubAgent(data_root_dir=None)._get_user_data_resources() == []


def test_runs_root_ignores_the_bundled_data_dir_when_no_workspace_is_configured(tmp_path, monkeypatch):
    """data_root_dir defaults to ./data; that is not a user workspace.

    Treating it as one would silently relocate a notebook user's output from
    ./runs into ./data/biomni-outputs.
    """
    monkeypatch.delenv("BIOMNI_USER_DATA_PATH", raising=False)
    monkeypatch.delenv("BIOMNI_DATA_PATH", raising=False)
    agent = _StubAgent(data_root_dir=str(tmp_path))
    assert agent._resolve_runs_root() == os.path.abspath(os.path.join(os.getcwd(), "runs"))


def test_runs_root_uses_the_workspace_once_one_is_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMNI_USER_DATA_PATH", str(tmp_path))
    agent = _StubAgent(data_root_dir=str(tmp_path))
    assert agent._resolve_runs_root() == str(tmp_path / "biomni-outputs")


def test_retriever_names_files_outside_the_data_root_by_absolute_path(tmp_path):
    """BIOMNI_USER_DATA_PATH and BIOMNI_DATA_PATH can differ; relpath would
    produce meaningless "../../" names in that normal case."""
    data_root = tmp_path / "data-root"
    data_root.mkdir()
    elsewhere = tmp_path / "mounted-study"
    elsewhere.mkdir()
    (elsewhere / "a.csv").write_text("x")

    agent = _StubAgent(data_root_dir=str(data_root), scope_roots=[str(elsewhere)])
    names = [r["name"] for r in agent._get_user_data_resources()]
    assert names == [f"user-data:{elsewhere / 'a.csv'}"]
    assert ".." not in names[0]
