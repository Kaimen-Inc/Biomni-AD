"""Tests for chainlit_ui.workspace_panel - scope rendering and agent inventory.

The module is deliberately Chainlit-free so it is importable in CI (which
installs no chainlit extra); these tests exercise it directly.

The behaviour that matters: an unselected workspace is advertised without ever
being walked, and a selected one is described from the selected folders only.
"""

from __future__ import annotations

import pytest
from biomni import fs_scan
from biomni.run_registry import RunRecord
from biomni.workspace_prefs import OutputTarget, WorkspacePrefs, resolve_scope
from chainlit_ui import workspace_panel as panel


@pytest.fixture(autouse=True)
def _reset_scan_cache() -> None:
    fs_scan.clear_cache()
    yield
    fs_scan.clear_cache()


@pytest.fixture
def workspace(tmp_path):
    """A workspace with two data folders and one that should never be walked."""
    (tmp_path / "studyA").mkdir()
    (tmp_path / "studyA" / "a1.csv").write_text("x")
    (tmp_path / "studyA" / "a2.csv").write_text("x")
    (tmp_path / "studyB").mkdir()
    (tmp_path / "studyB" / "b1.tsv").write_text("x")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "__pycache__").mkdir()
    return tmp_path


# --------------------------------------------------------------------------- #
# Cheap introspection
# --------------------------------------------------------------------------- #


def test_list_top_level_dirs_skips_hidden_and_excluded(workspace):
    assert panel.list_top_level_dirs(str(workspace)) == ["studyA", "studyB"]


def test_list_top_level_dirs_ignores_files(workspace):
    (workspace / "loose.txt").write_text("x")
    assert panel.list_top_level_dirs(str(workspace)) == ["studyA", "studyB"]


def test_list_top_level_dirs_handles_missing_root(tmp_path):
    assert panel.list_top_level_dirs(None) == []
    assert panel.list_top_level_dirs(str(tmp_path / "nope")) == []


def test_list_top_level_dirs_excludes_the_output_directory(workspace):
    """The output folder lives in the workspace but is a destination, not input."""
    (workspace / "biomni-outputs").mkdir()
    names = panel.list_top_level_dirs(str(workspace), exclude_paths=[str(workspace / "biomni-outputs")])
    assert names == ["studyA", "studyB"]


def test_list_top_level_dirs_ignores_exclusions_outside_the_workspace(workspace):
    names = panel.list_top_level_dirs(str(workspace), exclude_paths=["/somewhere/else", ""])
    assert names == ["studyA", "studyB"]


def test_list_top_level_dirs_respects_limit(workspace):
    for i in range(5):
        (workspace / f"extra{i}").mkdir()
    assert len(panel.list_top_level_dirs(str(workspace), limit=3)) == 3


def test_summarize_scope_counts_only_selected_folders(workspace):
    prefs = WorkspacePrefs(scope_paths=["studyA"])
    scope = resolve_scope(prefs, str(workspace))
    summaries = panel.summarize_scope(scope, str(workspace))
    assert [(s.label, s.file_count) for s in summaries] == [("studyA", 2)]


# --------------------------------------------------------------------------- #
# Agent inventory
# --------------------------------------------------------------------------- #


def test_inventory_without_a_selection_lists_folders_and_forbids_assuming_empty(workspace):
    scope = resolve_scope(WorkspacePrefs(), str(workspace))
    text = panel.build_scope_inventory(scope, str(workspace))
    assert "NOT been indexed" in text
    assert "studyA" in text and "studyB" in text
    assert "Do not assume the workspace is empty." in text
    # No file was described, only folders.
    assert "a1.csv" not in text


def test_inventory_without_a_selection_does_not_walk_the_workspace(workspace, monkeypatch):
    """The whole point of the default path: no recursive scan on a cold session."""

    def _explode(*args, **kwargs):
        raise AssertionError("scan_directory must not be called for an unselected scope")

    monkeypatch.setattr(panel, "scan_directory", _explode)
    scope = resolve_scope(WorkspacePrefs(), str(workspace))
    assert "studyA" in panel.build_scope_inventory(scope, str(workspace))


def test_inventory_with_a_selection_describes_only_that_folder(workspace):
    scope = resolve_scope(WorkspacePrefs(scope_paths=["studyA"]), str(workspace))
    text = panel.build_scope_inventory(scope, str(workspace))
    assert "a1.csv" in text and "a2.csv" in text
    assert "b1.tsv" not in text
    assert "Prefer them over anything else on disk." in text


def test_inventory_reports_missing_selections(workspace):
    scope = resolve_scope(WorkspacePrefs(scope_paths=["studyA", "ghost"]), str(workspace))
    text = panel.build_scope_inventory(scope, str(workspace))
    assert "no longer present" in text
    assert "ghost" in text


def test_inventory_flags_a_scope_whose_folders_all_vanished(workspace):
    """A wholly missing scope must not read to the agent as "nothing selected"."""
    scope = resolve_scope(WorkspacePrefs(scope_paths=["ghost1", "ghost2"]), str(workspace))
    assert scope.roots == []
    text = panel.build_scope_inventory(scope, str(workspace))
    assert "GONE" in text
    assert "ghost1" in text and "ghost2" in text
    assert "Tell the user" in text


def test_panel_flags_a_scope_whose_folders_all_vanished(workspace):
    scope = resolve_scope(WorkspacePrefs(scope_paths=["ghost"]), str(workspace))
    text = panel.build_scope_panel(scope, str(workspace), None)
    assert "Selected but missing" in text
    assert "ghost" in text


def test_inventory_is_empty_without_a_workspace():
    scope = resolve_scope(WorkspacePrefs(), None)
    assert panel.build_scope_inventory(scope, None) == ""


def test_inventory_flags_a_bounded_scan(workspace, monkeypatch):
    monkeypatch.setenv("BIOMNI_WORKSPACE_MAX_FILES", "1")
    fs_scan.clear_cache()
    scope = resolve_scope(WorkspacePrefs(scope_paths=["studyA"]), str(workspace))
    text = panel.build_scope_inventory(scope, str(workspace))
    assert "scan bounded for responsiveness" in text
    assert "1+" in text


# --------------------------------------------------------------------------- #
# Sidebar panel
# --------------------------------------------------------------------------- #


def test_panel_without_selection_invites_a_choice(workspace):
    scope = resolve_scope(WorkspacePrefs(), str(workspace))
    text = panel.build_scope_panel(scope, str(workspace), None)
    assert "Nothing selected yet" in text
    assert "studyA" in text
    assert "Settings" in text


def test_panel_with_selection_shows_counts(workspace):
    scope = resolve_scope(WorkspacePrefs(scope_paths=["studyA"]), str(workspace))
    text = panel.build_scope_panel(scope, str(workspace), None)
    assert "`studyA/`" in text
    assert "2 files" in text


def test_panel_warns_about_ephemeral_output(workspace):
    scope = resolve_scope(WorkspacePrefs(), str(workspace))
    output = OutputTarget(path="/app/runs", source="cwd-fallback", writable=True)
    text = panel.build_scope_panel(scope, str(workspace), output)
    assert "lost when the application restarts" in text
    assert "BIOMNI_OUTPUT_ROOT" in text


def test_panel_warns_about_unwritable_output(workspace):
    scope = resolve_scope(WorkspacePrefs(), str(workspace))
    output = OutputTarget(path="/nope", source="env", writable=False, reason="no writable output location found")
    text = panel.build_scope_panel(scope, str(workspace), output)
    assert "Not writable" in text


def test_panel_stays_quiet_about_a_healthy_output(workspace):
    scope = resolve_scope(WorkspacePrefs(), str(workspace))
    output = OutputTarget(path=str(workspace / "biomni-outputs"), source="workspace", writable=True)
    text = panel.build_scope_panel(scope, str(workspace), output)
    assert "⚠️" not in text
    assert "biomni-outputs" in text


def test_panel_lists_missing_selections(workspace):
    scope = resolve_scope(WorkspacePrefs(scope_paths=["studyA", "ghost"]), str(workspace))
    text = panel.build_scope_panel(scope, str(workspace), None)
    assert "Selected but missing" in text
    assert "ghost" in text


def test_panel_without_workspace():
    scope = resolve_scope(WorkspacePrefs(), None)
    assert "No user workspace is configured" in panel.build_scope_panel(scope, None, None)


def test_panel_reports_persistence_location(workspace):
    scope = resolve_scope(WorkspacePrefs(), str(workspace))
    text = panel.build_scope_panel(scope, str(workspace), None, persistence="/state/prefs")
    assert "/state/prefs" in text


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


def test_runs_panel_is_helpful_when_empty():
    assert "No runs recorded yet" in panel.build_runs_panel([])


def test_runs_panel_lists_status_and_output():
    records = [
        RunRecord(run_id="r1", status="completed", prompt="find targets", output_dir="/out/r1"),
        RunRecord(run_id="r2", status="interrupted", prompt="long job", error="stopped"),
    ]
    text = panel.build_runs_panel(records)
    assert "find targets" in text and "/out/r1" in text
    assert "long job" in text and "stopped" in text


def test_runs_panel_respects_limit():
    records = [RunRecord(run_id=f"r{i}", prompt=f"job {i}") for i in range(10)]
    text = panel.build_runs_panel(records, limit=2)
    assert "job 0" in text and "job 1" in text
    assert "job 5" not in text


def test_previous_runs_notice_only_mentions_unfinished_work():
    records = [
        RunRecord(run_id="ok", status="completed", prompt="done job"),
        RunRecord(run_id="bad", status="interrupted", prompt="broken job", output_dir="/out/bad"),
    ]
    notice = panel.build_previous_runs_notice(records)
    assert notice is not None
    assert "broken job" in notice
    assert "done job" not in notice
    assert "/out/bad" in notice


def test_previous_runs_notice_is_none_when_all_is_well():
    assert panel.build_previous_runs_notice([RunRecord(run_id="ok", status="completed")]) is None
    assert panel.build_previous_runs_notice([]) is None


def test_previous_runs_notice_truncates_long_lists():
    records = [RunRecord(run_id=f"r{i}", status="failed", prompt=f"job {i}") for i in range(8)]
    notice = panel.build_previous_runs_notice(records)
    assert notice is not None
    assert "and 3 more" in notice


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #


def test_scope_choice_items_label_folders():
    assert panel.scope_choice_items(["a", "b"]) == {"📁 a": "a", "📁 b": "b"}


def test_tree_preview_renders_nested_paths():
    lines = panel.build_tree_preview_lines(["a/b/c.csv", "a/d.csv", "top.txt"])
    rendered = "\n".join(lines)
    assert "📁 a/" in rendered
    assert "📄 top.txt" in rendered


def test_tree_preview_respects_max_lines():
    paths = [f"dir/f{i}.csv" for i in range(50)]
    assert len(panel.build_tree_preview_lines(paths, max_lines=5)) <= 5
