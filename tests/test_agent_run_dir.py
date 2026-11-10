"""Tests for the folder a ``go()`` call writes into.

``AD1.go`` makes a run folder to save its report, trace and notebook in, and then
hands the run to ``A1.go``, which made a second one: everything the generated
code saved went into the second, and whoever opened the first found a report
with no files beside it. ``A1.go`` also took the folder an earlier run had left
on the agent for one its caller had made, so a second ``go()`` on one agent wrote
into the first run's folder and replaced its report, trace and notebook.

``A1`` is expensive to construct (it builds an LLM client and a LangGraph), so
these build the agents without ``__init__`` and give them only what ``go()``
reads. The whole module skips when the agent stack (pandas/langchain/langgraph)
is not installed.
"""

from __future__ import annotations

import itertools
import os

import pytest

try:
    from biomni.agent.a1 import A1
    from biomni.agent.ad1 import AD1
    from langchain_core.messages import AIMessage, HumanMessage

    HAS_AGENT = True
except Exception:  # pragma: no cover - depends on the installed extras
    HAS_AGENT = False

pytestmark = pytest.mark.skipif(not HAS_AGENT, reason="agent stack (pandas/langchain/langgraph) not installed")

AGENTS = [A1, AD1] if HAS_AGENT else []
AGENT_IDS = ["A1", "AD1"]


class _App:
    """Streams one reply, and saves a file where generated code is told to: the run's OUTPUT_DIR."""

    def __init__(self, agent):
        self.agent = agent
        self.calls = itertools.count(1)
        self.output_path_seen = []

    def stream(self, inputs, stream_mode, config):
        n = next(self.calls)
        # What _inject_custom_functions_to_repl hands generated code as OUTPUT_DIR.
        run_dir = self.agent._current_run_dir
        self.output_path_seen.append(os.environ.get("BIOMNI_OUTPUT_PATH"))
        with open(os.path.join(run_dir, f"plot_{n}.png"), "w") as f:
            f.write("png")
        human = HumanMessage(content="hi")
        yield {"messages": [human]}
        yield {"messages": [human, AIMessage(content="<solution>done</solution>")]}


class _Saved:
    """What ``_save_run_artifacts`` was asked to save, and where."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    @property
    def folders(self) -> list[str]:
        return [folder for _run_id, folder in self.calls]


def _agent(cls, tmp_path):
    """An agent with just what ``go()`` reads, and the saving of artifacts replaced by a record of the call."""
    agent = cls.__new__(cls)
    ids = itertools.count(1)
    agent.use_tool_retriever = False
    agent.ad_keywords = []
    agent.data_root_dir = None
    agent.run_ids_built = []

    def build_run_id(topic=None):
        run_id = f"run_{next(ids)}"
        agent.run_ids_built.append(run_id)
        return run_id

    agent._build_run_id = build_run_id
    agent._resolve_runs_root = lambda: str(tmp_path / "runs")
    agent._get_all_files = lambda directory: set()
    agent.app = _App(agent)
    saved = _Saved()

    def save(run_id, run_dir, initial_files):
        saved.calls.append((run_id, run_dir))
        with open(os.path.join(run_dir, "report.md"), "w") as f:
            f.write("# report")

    agent._save_run_artifacts = save
    return agent, saved


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    # go() writes BIOMNI_OUTPUT_PATH; setting it here makes monkeypatch undo whatever it does.
    monkeypatch.setenv("BIOMNI_OUTPUT_PATH", "")
    monkeypatch.chdir(tmp_path)


def _folders(tmp_path) -> list[str]:
    root = tmp_path / "runs"
    return sorted(os.listdir(root)) if root.exists() else []


@pytest.mark.parametrize("cls", AGENTS, ids=AGENT_IDS)
class TestEachCallGetsItsOwnFolder:
    def test_the_report_and_the_generated_files_are_in_one_folder(self, cls, tmp_path):
        agent, saved = _agent(cls, tmp_path)

        agent.go("hello")

        assert _folders(tmp_path) == ["run_1"]
        (folder,) = saved.folders
        assert sorted(os.listdir(folder)) == ["plot_1.png", "report.md"]

    def test_generated_code_is_told_the_folder_the_report_goes_to(self, cls, tmp_path):
        agent, saved = _agent(cls, tmp_path)

        agent.go("hello")

        assert agent.app.output_path_seen == saved.folders

    def test_a_second_call_writes_into_its_own_folder(self, cls, tmp_path):
        """``A1.go`` used to take the first run's folder, left on the agent, for one its caller had made."""
        agent, saved = _agent(cls, tmp_path)

        agent.go("first")
        agent.go("second")

        assert _folders(tmp_path) == ["run_1", "run_2"]
        first, second = saved.folders
        assert sorted(os.listdir(first)) == ["plot_1.png", "report.md"]
        assert sorted(os.listdir(second)) == ["plot_2.png", "report.md"]

    def test_the_run_id_is_built_once_per_call(self, cls, tmp_path):
        """Building it names the folder from the prompt, which asks the LLM for a short label."""
        agent, _saved = _agent(cls, tmp_path)

        agent.go("first")
        agent.go("second")

        assert agent.run_ids_built == ["run_1", "run_2"]

    def test_the_artifacts_are_saved_under_the_name_of_the_folder(self, cls, tmp_path):
        agent, saved = _agent(cls, tmp_path)

        agent.go("hello")

        assert saved.calls == [("run_1", str(tmp_path / "runs" / "run_1"))]


@pytest.mark.parametrize("cls", AGENTS, ids=AGENT_IDS)
class TestAFolderTheCallerChose:
    def test_the_run_writes_into_it_and_makes_no_other(self, cls, tmp_path):
        agent, saved = _agent(cls, tmp_path)
        chosen = tmp_path / "chosen" / "nested"

        agent.go("hello", run_dir=str(chosen))

        assert saved.calls == [("nested", str(chosen))]
        assert sorted(os.listdir(chosen)) == ["plot_1.png", "report.md"]
        assert agent.app.output_path_seen == [str(chosen)]
        assert _folders(tmp_path) == []
        assert agent.run_ids_built == []

    def test_a_relative_folder_is_taken_from_the_working_directory(self, cls, tmp_path):
        agent, saved = _agent(cls, tmp_path)

        agent.go("hello", run_dir="chosen/")

        assert saved.calls == [("chosen", str(tmp_path / "chosen"))]

    def test_the_next_call_does_not_reuse_it(self, cls, tmp_path):
        agent, saved = _agent(cls, tmp_path)

        agent.go("first", run_dir=str(tmp_path / "chosen"))
        agent.go("second")

        assert saved.folders == [str(tmp_path / "chosen"), str(tmp_path / "runs" / "run_1")]
        assert sorted(os.listdir(tmp_path / "chosen")) == ["plot_1.png", "report.md"]
