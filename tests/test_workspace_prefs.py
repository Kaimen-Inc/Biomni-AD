"""Tests for biomni.workspace_prefs - scope and output-directory preferences.

The behaviours worth locking in are the degradation paths: a corrupt or stale
preferences file must fall back to defaults rather than break session start, and
output-directory resolution must always yield a usable path even when the
preferred one has become unwritable.
"""

from __future__ import annotations

import json
import os

import pytest
from biomni import workspace_prefs as wp

_ENV_TO_CLEAR = (
    "BIOMNI_DEFAULT_SCOPE_PATHS",
    "BIOMNI_OUTPUT_ROOT",
    "BIOMNI_PREFS_DIR",
    "BIOMNI_STATE_DIR",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# WorkspacePrefs
# --------------------------------------------------------------------------- #


def test_default_prefs_are_empty_scope():
    prefs = wp.default_prefs()
    assert prefs.scope_paths == []
    assert prefs.scope_is_default
    assert prefs.output_dir is None


def test_default_prefs_can_be_seeded_by_env(monkeypatch):
    monkeypatch.setenv("BIOMNI_DEFAULT_SCOPE_PATHS", "studies/one, studies/two")
    prefs = wp.default_prefs()
    assert prefs.scope_paths == ["studies/one", "studies/two"]
    assert not prefs.scope_is_default


def test_from_dict_rejects_unusable_payloads():
    assert wp.WorkspacePrefs.from_dict(None) is None
    assert wp.WorkspacePrefs.from_dict("nope") is None
    assert wp.WorkspacePrefs.from_dict({"version": wp.PREFS_VERSION + 1}) is None


def test_from_dict_filters_junk_scope_entries():
    prefs = wp.WorkspacePrefs.from_dict(
        {"version": wp.PREFS_VERSION, "scope_paths": ["ok", "", None, 42, "  "], "output_dir": 7}
    )
    assert prefs is not None
    assert prefs.scope_paths == ["ok"]
    assert prefs.output_dir is None  # non-string discarded


def test_round_trip_through_dict():
    prefs = wp.WorkspacePrefs(scope_paths=["a"], output_dir="/out", remember=False, email="x@y.org")
    restored = wp.WorkspacePrefs.from_dict(prefs.to_dict())
    assert restored is not None
    assert restored.scope_paths == ["a"]
    assert restored.output_dir == "/out"
    assert restored.remember is False
    assert restored.email == "x@y.org"


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def test_json_store_round_trip(tmp_path):
    store = wp.JsonFilePrefsStore(str(tmp_path))
    prefs = wp.WorkspacePrefs(scope_paths=["studies/x"], output_dir="out")
    assert store.save("user-abc", prefs) is True

    loaded = store.load("user-abc")
    assert loaded is not None
    assert loaded.scope_paths == ["studies/x"]
    assert loaded.output_dir == "out"
    assert loaded.updated_at  # stamped on save


def test_json_store_leaves_no_temp_files(tmp_path):
    store = wp.JsonFilePrefsStore(str(tmp_path))
    store.save("user-abc", wp.WorkspacePrefs())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["user-abc.json"]


def test_json_store_missing_user_is_none(tmp_path):
    assert wp.JsonFilePrefsStore(str(tmp_path)).load("nobody") is None


def test_corrupt_prefs_file_falls_back_to_defaults(tmp_path):
    (tmp_path / "user-abc.json").write_text("{not json", encoding="utf-8")
    store = wp.JsonFilePrefsStore(str(tmp_path))
    assert store.load("user-abc") is None
    assert wp.load_prefs(store, "user-abc").scope_is_default


def test_store_key_cannot_escape_its_directory(tmp_path):
    nested = tmp_path / "prefs"
    nested.mkdir()
    store = wp.JsonFilePrefsStore(str(nested))
    store.save("../escaped", wp.WorkspacePrefs())
    # Written inside the store directory, not beside it.
    assert not (tmp_path / "escaped.json").exists()
    assert list(nested.glob("*.json"))


def test_null_store_never_persists():
    store = wp.NullPrefsStore()
    assert store.save("k", wp.WorkspacePrefs()) is False
    assert store.load("k") is None
    assert wp.load_prefs(store, "k").scope_is_default


def test_load_prefs_returns_stored_value_when_present(tmp_path):
    store = wp.JsonFilePrefsStore(str(tmp_path))
    store.save("k", wp.WorkspacePrefs(scope_paths=["chosen"]))
    assert wp.load_prefs(store, "k").scope_paths == ["chosen"]


# --------------------------------------------------------------------------- #
# Store selection
# --------------------------------------------------------------------------- #


def test_build_store_prefers_explicit_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMNI_PREFS_DIR", str(tmp_path / "explicit"))
    monkeypatch.setenv("BIOMNI_STATE_DIR", str(tmp_path / "state"))
    store = wp.build_prefs_store(str(tmp_path / "workspace"))
    assert isinstance(store, wp.JsonFilePrefsStore)
    assert store.root == str(tmp_path / "explicit")


def test_build_store_uses_state_dir_next(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMNI_STATE_DIR", str(tmp_path / "state"))
    store = wp.build_prefs_store(None)
    assert isinstance(store, wp.JsonFilePrefsStore)
    assert store.root == str(tmp_path / "state" / "prefs")


def test_build_store_falls_back_to_writable_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = wp.build_prefs_store(str(workspace))
    assert isinstance(store, wp.JsonFilePrefsStore)
    assert store.root == str(workspace / wp.WORKSPACE_STATE_DIRNAME / "prefs")


def test_build_store_disabled_when_nothing_is_writable(tmp_path):
    # A workspace path that does not exist and cannot be created (parent missing).
    assert isinstance(wp.build_prefs_store(None), wp.NullPrefsStore)


def test_build_store_ignores_read_only_workspace(tmp_path):
    workspace = tmp_path / "ro"
    workspace.mkdir()
    workspace.chmod(0o500)
    try:
        assert isinstance(wp.build_prefs_store(str(workspace)), wp.NullPrefsStore)
    finally:
        workspace.chmod(0o700)  # so pytest can clean up


# --------------------------------------------------------------------------- #
# Writability
# --------------------------------------------------------------------------- #


def test_is_writable_dir_for_creatable_nested_path(tmp_path):
    assert wp.is_writable_dir(str(tmp_path / "does" / "not" / "exist"))


def test_is_writable_dir_rejects_empty_and_read_only(tmp_path):
    assert wp.is_writable_dir(None) is False
    assert wp.is_writable_dir("") is False
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        assert wp.is_writable_dir(str(ro / "child")) is False
    finally:
        ro.chmod(0o700)


# --------------------------------------------------------------------------- #
# Output directory resolution
# --------------------------------------------------------------------------- #


def test_output_preference_wins(tmp_path):
    prefs = wp.WorkspacePrefs(output_dir=str(tmp_path / "chosen"))
    target = wp.resolve_output_dir(prefs, workspace_root=str(tmp_path))
    assert target.source == "preference"
    assert target.path == str(tmp_path / "chosen")
    assert target.writable
    assert not target.is_ephemeral


def test_relative_output_preference_is_workspace_relative(tmp_path):
    prefs = wp.WorkspacePrefs(output_dir="results")
    target = wp.resolve_output_dir(prefs, workspace_root=str(tmp_path))
    assert target.path == str(tmp_path / "results")


def test_env_output_root_used_when_no_preference(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMNI_OUTPUT_ROOT", str(tmp_path / "volume"))
    target = wp.resolve_output_dir(wp.WorkspacePrefs(), workspace_root=str(tmp_path))
    assert target.source == "env"
    assert target.path == str(tmp_path / "volume")


def test_writable_workspace_beats_cwd_fallback(tmp_path):
    target = wp.resolve_output_dir(wp.WorkspacePrefs(), workspace_root=str(tmp_path))
    assert target.source == "workspace"
    assert target.path == str(tmp_path / wp.DEFAULT_OUTPUT_DIRNAME)


def test_read_only_workspace_falls_back_to_cwd_and_is_flagged_ephemeral(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        target = wp.resolve_output_dir(wp.WorkspacePrefs(), workspace_root=str(ro))
        assert target.source == "cwd-fallback"
        assert target.is_ephemeral
        assert target.reason and "workspace" in target.reason
    finally:
        ro.chmod(0o700)


def test_unwritable_preference_falls_through_to_next_candidate(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        prefs = wp.WorkspacePrefs(output_dir=str(ro / "nope"))
        target = wp.resolve_output_dir(prefs, workspace_root=str(tmp_path))
        assert target.source == "workspace"
        assert target.reason and "preference" in target.reason
    finally:
        ro.chmod(0o700)


def test_ensure_output_dir_creates_it(tmp_path):
    target = wp.resolve_output_dir(wp.WorkspacePrefs(), workspace_root=str(tmp_path))
    assert wp.ensure_output_dir(target) is True
    assert os.path.isdir(target.path)


def test_ensure_output_dir_refuses_unwritable_target():
    target = wp.OutputTarget(path="/definitely/not/writable", source="env", writable=False)
    assert wp.ensure_output_dir(target) is False


# --------------------------------------------------------------------------- #
# Scope resolution
# --------------------------------------------------------------------------- #


def test_scope_is_default_when_unset(tmp_path):
    resolution = wp.resolve_scope(wp.WorkspacePrefs(), str(tmp_path))
    assert resolution.is_default
    assert not resolution.has_selection


def test_scope_resolves_relative_entries(tmp_path):
    (tmp_path / "studies").mkdir()
    resolution = wp.resolve_scope(wp.WorkspacePrefs(scope_paths=["studies"]), str(tmp_path))
    assert resolution.roots == [str(tmp_path / "studies")]
    assert resolution.missing == []
    assert not resolution.is_default


def test_scope_reports_missing_entries_instead_of_dropping_them(tmp_path):
    (tmp_path / "here").mkdir()
    resolution = wp.resolve_scope(wp.WorkspacePrefs(scope_paths=["here", "gone"]), str(tmp_path))
    assert resolution.roots == [str(tmp_path / "here")]
    assert resolution.missing == ["gone"]


def test_scope_deduplicates_equivalent_entries(tmp_path):
    (tmp_path / "s").mkdir()
    prefs = wp.WorkspacePrefs(scope_paths=["s", str(tmp_path / "s")])
    assert wp.resolve_scope(prefs, str(tmp_path)).roots == [str(tmp_path / "s")]


def test_relative_label_falls_back_to_absolute_outside_workspace(tmp_path):
    assert wp.relative_label(str(tmp_path / "a" / "b"), str(tmp_path)) == os.path.join("a", "b")
    assert wp.relative_label("/elsewhere/x", str(tmp_path)) == "/elsewhere/x"


def test_normalize_scope_entries_relativizes_and_dedupes(tmp_path):
    entries = [str(tmp_path / "a"), "a", " b ", "", "b"]
    assert wp.normalize_scope_entries(entries, str(tmp_path)) == ["a", "b"]


def test_saved_prefs_survive_a_reload_cycle(tmp_path):
    """End-to-end: choose a scope, persist it, read it back in a fresh store."""
    store = wp.build_prefs_store(str(tmp_path))
    (tmp_path / "studies").mkdir()
    prefs = wp.WorkspacePrefs(scope_paths=wp.normalize_scope_entries([str(tmp_path / "studies")], str(tmp_path)))
    assert store.save("user-1", prefs)

    reloaded = wp.load_prefs(wp.build_prefs_store(str(tmp_path)), "user-1")
    assert wp.resolve_scope(reloaded, str(tmp_path)).roots == [str(tmp_path / "studies")]

    on_disk = json.loads((tmp_path / wp.WORKSPACE_STATE_DIRNAME / "prefs" / "user-1.json").read_text())
    assert on_disk["scope_paths"] == ["studies"]


# --------------------------------------------------------------------------- #
# env_flag / NullPrefsStore reason
# --------------------------------------------------------------------------- #


def test_env_flag_recognises_true_and_false_spellings(monkeypatch):
    for raw in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("BIOMNI_TEST_FLAG", raw)
        assert wp.env_flag("BIOMNI_TEST_FLAG") is True
    for raw in ("0", "false", "no", "off"):
        monkeypatch.setenv("BIOMNI_TEST_FLAG", raw)
        assert wp.env_flag("BIOMNI_TEST_FLAG") is False


def test_env_flag_falls_back_on_unset_or_garbage(monkeypatch):
    monkeypatch.delenv("BIOMNI_TEST_FLAG", raising=False)
    assert wp.env_flag("BIOMNI_TEST_FLAG") is False
    assert wp.env_flag("BIOMNI_TEST_FLAG", default=True) is True
    monkeypatch.setenv("BIOMNI_TEST_FLAG", "maybe")
    assert wp.env_flag("BIOMNI_TEST_FLAG", default=True) is True


def test_null_store_explains_why_nothing_persists():
    assert "no writable" in wp.NullPrefsStore().describe
    assert "no authenticated user" in wp.NullPrefsStore("no authenticated user").describe
