"""Tests for biomni.workspace_prefs - scope and output-directory preferences.

The behaviours worth locking in are the degradation paths: a corrupt or stale
preferences file must fall back to defaults rather than break session start, and
output-directory resolution must always yield a usable path even when the
preferred one has become unwritable.
"""

from __future__ import annotations

import json
import logging
import os
import stat

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
        assert target.reason and target.reason.startswith("The output directory you configured")
    finally:
        ro.chmod(0o700)


def test_fallback_reason_names_the_setting_and_the_path(tmp_path, monkeypatch):
    """An unwritable BIOMNI_OUTPUT_ROOT must be distinguishable from an unset one.

    Regression test for the GRIP report: the app fell back to /app/runs and said
    nothing about why, which looked identical to the env var being ignored.
    """
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        monkeypatch.setenv("BIOMNI_OUTPUT_ROOT", str(ro / "outputs"))
        target = wp.resolve_output_dir(wp.WorkspacePrefs(), workspace_root=str(tmp_path))
        assert target.source == "workspace"
        assert target.reason
        assert "BIOMNI_OUTPUT_ROOT" in target.reason
        assert str(ro / "outputs") in target.reason
    finally:
        ro.chmod(0o700)


def test_skipped_output_root_is_logged(tmp_path, monkeypatch, caplog):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        monkeypatch.setenv("BIOMNI_OUTPUT_ROOT", str(ro / "outputs"))
        with caplog.at_level(logging.WARNING, logger="biomni.workspace_prefs"):
            wp.resolve_output_dir(wp.WorkspacePrefs(), workspace_root=str(tmp_path))
        messages = [record.getMessage() for record in caplog.records]
        assert any("BIOMNI_OUTPUT_ROOT" in message and str(ro / "outputs") in message for message in messages)
    finally:
        ro.chmod(0o700)


def test_cwd_fallback_on_a_mounted_volume_is_not_ephemeral():
    """A disk mounted over the fallback directory is durable, and must not warn.

    GRIP mounts one at /app/runs - the exact path the fallback uses - so the
    "results are lost on restart" warning was false there.
    """
    on_volume = wp.OutputTarget(path="/app/runs", source="cwd-fallback", writable=True, mounted_volume=True)
    assert not on_volume.is_ephemeral

    container_local = wp.OutputTarget(path="/app/runs", source="cwd-fallback", writable=True)
    assert container_local.is_ephemeral


def test_on_mounted_volume_needs_positive_evidence(tmp_path, monkeypatch):
    """Unverifiable means "not a volume", so the UI never over-promises."""
    monkeypatch.setattr(wp, "_mount_fstype", lambda _path: None)
    assert wp.on_mounted_volume(str(tmp_path)) is False

    # A distinct mount that is RAM-backed or the container's own layer is not
    # durability either, however different its device number.
    monkeypatch.setattr(wp, "_mount_fstype", lambda _path: "tmpfs")
    assert wp.on_mounted_volume(str(tmp_path)) is False
    monkeypatch.setattr(wp, "_mount_fstype", lambda _path: "overlay")
    assert wp.on_mounted_volume(str(tmp_path)) is False


def test_mount_fstype_picks_the_longest_matching_mountpoint(tmp_path, monkeypatch):
    mounts = tmp_path / "mounts"
    mounts.write_text(
        "overlay / overlay rw 0 0\n"
        "/dev/sdh /app/runs ext4 rw 0 0\n"
        "//stg.file.core.windows.net/share /app/user-data cifs rw 0 0\n"
    )
    real_open = open
    monkeypatch.setattr(
        "builtins.open",
        lambda path, *a, **k: real_open(mounts, *a, **k) if path == "/proc/self/mounts" else real_open(path, *a, **k),
    )
    assert wp._mount_fstype("/app/runs/run_1") == "ext4"
    assert wp._mount_fstype("/app/user-data/x") == "cifs"
    assert wp._mount_fstype("/app/biomni") == "overlay"


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


def test_delete_forgets_stored_preferences(tmp_path):
    store = wp.JsonFilePrefsStore(str(tmp_path))
    store.save("k", wp.WorkspacePrefs(scope_paths=["chosen"]))
    assert store.delete("k") is True
    assert store.load("k") is None
    # Deleting what is not there is success: the desired state is "forgotten".
    assert store.delete("k") is True


def test_null_store_delete_is_a_no_op():
    assert wp.NullPrefsStore().delete("k") is False


# --------------------------------------------------------------------------- #
# Explaining an unusable directory
# --------------------------------------------------------------------------- #
#
# GRIP runs the app as uid 57439 against an Azure Files (SMB) share. What the
# process meets there is decided by the mount table and by who the process is,
# so both are substituted: a mount table written for the test, and a process
# identity that is not the owner of the test's directories.

_GRIP_UID = 57439


def _mount_table(tmp_path, monkeypatch, *lines: str) -> None:
    table = tmp_path / "mounts"
    table.write_text("".join(f"{line}\n" for line in lines))
    monkeypatch.setattr(wp, "_MOUNTS_FILE", str(table))


def _as_grip_uid(monkeypatch) -> None:
    monkeypatch.setattr(wp, "_process_ids", lambda: (_GRIP_UID, _GRIP_UID, frozenset({_GRIP_UID})))


def _share(tmp_path, mode: int = 0o755):
    share = tmp_path / "user-data"
    share.mkdir()
    share.chmod(mode)
    return share


def test_mount_info_reads_escaped_paths_and_options(tmp_path, monkeypatch):
    _mount_table(
        tmp_path,
        monkeypatch,
        "overlay / overlay rw,relatime 0 0",
        "//acct.file.core.windows.net/share /app/user\\040data cifs rw,vers=3.1.1,uid=0,gid=0,dir_mode=0777 0 0",
    )
    info = wp.mount_info("/app/user data/studyA")
    assert info is not None
    assert (info.mountpoint, info.fstype, info.source) == (
        "/app/user data",
        "cifs",
        "//acct.file.core.windows.net/share",
    )
    assert info.option("uid") == "0" and info.option("dir_mode") == "0777"
    assert not info.read_only


def test_the_last_mount_on_a_mountpoint_is_the_one_in_effect(tmp_path, monkeypatch):
    """Mounting again over the same point hides the first mount."""
    _mount_table(
        tmp_path,
        monkeypatch,
        "//acct/share /app/user-data cifs rw,uid=0,gid=0 0 0",
        "//acct/share /app/user-data cifs rw,uid=57439,gid=57439 0 0",
    )
    assert wp.mount_info("/app/user-data").option("uid") == "57439"


def test_a_mount_point_matches_whole_path_components_only(tmp_path, monkeypatch):
    """/app/user-data is not a parent of /app/user."""
    _mount_table(tmp_path, monkeypatch, "overlay / overlay rw 0 0", "//acct/share /app/user-data cifs rw 0 0")
    assert wp.mount_info("/app/user").mountpoint == "/"
    assert wp.mount_info("/app/user-data/x").mountpoint == "/app/user-data"


def test_a_symlink_onto_a_share_is_on_the_share(tmp_path, monkeypatch):
    share = _share(tmp_path)
    link = tmp_path / "workspace-link"
    link.symlink_to(share)
    _mount_table(tmp_path, monkeypatch, "overlay / overlay rw 0 0", f"//acct/share {share} cifs rw 0 0")
    assert wp.mount_info(str(link)).fstype == "cifs"
    assert wp.mount_info(str(link / "study")).mountpoint == str(share)


def test_a_mount_point_that_is_not_utf8_does_not_break_the_lookup(tmp_path, monkeypatch):
    table = tmp_path / "mounts"
    table.write_bytes(b"overlay / overlay rw 0 0\n/dev/sdc /mnt/caf\xe9 ext4 rw 0 0\n")
    monkeypatch.setattr(wp, "_MOUNTS_FILE", str(table))
    assert wp.mount_info("/app/runs").mountpoint == "/"


def test_no_mount_table_means_cannot_tell(tmp_path, monkeypatch):
    monkeypatch.setattr(wp, "_MOUNTS_FILE", str(tmp_path / "absent"))
    assert wp.mount_info("/app") is None


def test_an_smb_share_is_explained_by_the_options_it_is_actually_mounted_with(tmp_path, monkeypatch):
    """The GRIP case: new mount options set, but not the ones this container sees."""
    share = _share(tmp_path)
    _mount_table(
        tmp_path, monkeypatch, f"//acct.file.core.windows.net/share {share} cifs rw,uid=0,gid=0,dir_mode=0755 0 0"
    )
    _as_grip_uid(monkeypatch)

    reason = wp.explain_unwritable(str(share / "biomni-outputs"))
    assert f"{share} is owned by uid" in reason
    # The share, never the storage account that serves it.
    assert "on an SMB share (/share)" in reason and "acct" not in reason
    assert "which are uid=0,gid=0,dir_mode=0755 on the mount this container sees" in reason
    assert "Mount it with uid=57439,gid=57439,dir_mode=0770,file_mode=0770" in reason
    assert "mounted afresh on the node" in reason


def test_an_smb_share_with_no_ownership_options_says_they_are_unset(tmp_path, monkeypatch):
    share = _share(tmp_path)
    _mount_table(tmp_path, monkeypatch, f"//acct/share {share} cifs rw,vers=3.1.1 0 0")
    _as_grip_uid(monkeypatch)
    assert "which are unset (uid=0,gid=0)" in wp.explain_unwritable(str(share))


def test_an_smb_share_under_server_acls_is_not_blamed_on_the_mount_options(tmp_path, monkeypatch):
    share = _share(tmp_path)
    _mount_table(tmp_path, monkeypatch, f"//acct/share {share} cifs rw,cifsacl,uid=0,gid=0 0 0")
    _as_grip_uid(monkeypatch)
    reason = wp.explain_unwritable(str(share))
    assert "mounted with cifsacl, so its owner and mode come from the file server's ACLs" in reason
    assert "grant uid 57439 write access in the share's ACLs" in reason
    assert "come only from the mount options" not in reason


def test_a_share_the_process_cannot_list_is_explained_for_reading(tmp_path, monkeypatch):
    """dir_mode=0770 applied but uid/gid not: the workspace becomes unreadable, not just read-only."""
    share = _share(tmp_path, 0o770)
    try:
        _mount_table(tmp_path, monkeypatch, f"//acct/share {share} cifs rw,uid=0,gid=0,dir_mode=0770 0 0")
        _as_grip_uid(monkeypatch)
        reason = wp.explain_unreadable(str(share))
        assert "with mode 0770 on an SMB share" in reason
        assert "uid=0,gid=0,dir_mode=0770" in reason
        assert "Mount it with uid=57439,gid=57439" in reason
    finally:
        share.chmod(0o755)


def test_a_read_only_mount_is_named_as_such(tmp_path, monkeypatch):
    share = _share(tmp_path)
    _mount_table(tmp_path, monkeypatch, f"/dev/sdb {share} ext4 ro,relatime 0 0")
    _as_grip_uid(monkeypatch)

    def no_statvfs(_path):
        raise OSError("unsupported")

    # The mount table is the fallback witness when statvfs cannot answer.
    monkeypatch.setattr(os, "statvfs", no_statvfs)
    reason = wp.explain_unwritable(str(share))
    assert reason == f"{share} is on a read-only mount (ext4 mount of /dev/sdb); mount the volume read-write"


def test_a_read_only_container_image_is_not_told_to_remount(tmp_path, monkeypatch):
    """readOnlyRootFilesystem with the volume missing: there is nothing to remount."""
    image = _share(tmp_path)
    _mount_table(tmp_path, monkeypatch, f"overlay {image} overlay ro,relatime 0 0")
    _as_grip_uid(monkeypatch)
    monkeypatch.setattr(os, "statvfs", lambda _path: (_ for _ in ()).throw(OSError("unsupported")))
    reason = wp.explain_unwritable(str(image / "runs"))
    assert reason == (
        f"{image} is inside the container image, which is mounted read-only; mount a writable volume at "
        f"{image / 'runs'}"
    )


def test_an_nfs_export_is_explained_as_ignoring_fsgroup(tmp_path, monkeypatch):
    share = _share(tmp_path)
    _mount_table(tmp_path, monkeypatch, f"nfs.example:/export {share} nfs4 rw 0 0")
    _as_grip_uid(monkeypatch)
    reason = wp.explain_unwritable(str(share))
    assert "on an NFS export (/export), which ignores fsGroup" in reason
    assert "grant uid 57439 write access on the NFS server" in reason


def test_a_fuse_mount_is_not_given_the_fsgroup_remedy(tmp_path, monkeypatch):
    share = _share(tmp_path)
    _mount_table(tmp_path, monkeypatch, f"blobfuse2 {share} fuse rw,nosuid,nodev 0 0")
    _as_grip_uid(monkeypatch)
    reason = wp.explain_unwritable(str(share))
    assert "on a FUSE mount (blobfuse2), which ignores fsGroup" in reason
    assert "allow_other" in reason
    assert "securityContext.fsGroup" not in reason


def test_an_ordinary_volume_gets_the_fsgroup_remedy(tmp_path, monkeypatch):
    share = _share(tmp_path)
    _mount_table(tmp_path, monkeypatch, f"/dev/sdb {share} ext4 rw 0 0")
    _as_grip_uid(monkeypatch)
    assert wp.explain_unwritable(str(share)).endswith(
        "(ext4 mount of /dev/sdb); set securityContext.fsGroup: 57439 on the pod (it does not apply to hostPath "
        "volumes), or chown it to uid 57439"
    )


def _dir_stat(mode: int, uid: int, gid: int):
    return os.stat_result((stat.S_IFDIR | mode, 0, 0, 0, uid, gid, 0, 0, 0, 0))


def test_the_owner_class_alone_decides_for_the_owner():
    """0o577: group and others may write, the owner may not - and the owner is denied."""
    st = _dir_stat(0o577, uid=1000, gid=1000)
    assert not wp._mode_allows(st, 1000, frozenset({1000}), write=True)
    assert wp._mode_allows(st, 1000, frozenset({1000}), write=False)
    assert wp._mode_allows(st, 2000, frozenset({1000}), write=True)  # group member
    assert wp._mode_allows(st, 3000, frozenset({3000}), write=True)  # anyone else


@pytest.mark.skipif(os.geteuid() == 0, reason="root is never denied by mode bits")
def test_a_directory_that_denies_its_own_owner_is_not_answered_with_chown(tmp_path, monkeypatch):
    share = _share(tmp_path, 0o500)
    try:
        monkeypatch.setattr(wp, "_MOUNTS_FILE", str(tmp_path / "absent"))
        reason = wp.explain_unwritable(str(share))
        assert f"which denies its own owner (uid {os.getuid()})" in reason
        assert "chmod u+rwx" in reason
        assert "chown" not in reason
    finally:
        share.chmod(0o755)


def test_a_mode_that_permits_access_points_at_what_else_can_refuse_it(tmp_path, monkeypatch):
    share = _share(tmp_path)
    monkeypatch.setattr(wp, "_MOUNTS_FILE", str(tmp_path / "absent"))
    uid = os.stat(share).st_uid
    monkeypatch.setattr(wp, "_process_ids", lambda: (uid, 0, frozenset({0})))
    assert "which permits writing, so something else refuses it" in wp.explain_unwritable(str(share))


def test_storage_facts_carry_what_a_log_reader_needs(tmp_path, monkeypatch):
    share = _share(tmp_path)
    _mount_table(
        tmp_path,
        monkeypatch,
        f"//acct/share {share} cifs rw,vers=3.1.1,username=acct,addr=10.0.0.9,uid=0,gid=0,dir_mode=0755,mfsymlinks 0 0",
    )
    _as_grip_uid(monkeypatch)
    facts = wp.storage_facts(str(share / "biomni-outputs"))
    assert facts["path"] == str(share / "biomni-outputs")
    assert facts["checked"] == str(share)
    assert facts["fstype"] == "cifs" and facts["source"] == "/share"
    assert facts["mode"] == "0755"
    # Only the options that decide access - no account name, no server address.
    assert facts["options"] == "rw,vers=3.1.1,uid=0,gid=0,dir_mode=0755"
    assert not any("acct" in str(value) or "10.0.0.9" in str(value) for value in facts.values())
    assert {"readable", "writable"} <= facts.keys()


def test_storage_facts_explain_what_the_process_cannot_do(tmp_path, monkeypatch):
    share = _share(tmp_path)
    monkeypatch.setattr(wp, "_MOUNTS_FILE", str(tmp_path / "absent"))
    monkeypatch.setattr(wp, "is_writable_dir", lambda _path: False)
    monkeypatch.setattr(wp, "is_readable_dir", lambda _path: False)
    _as_grip_uid(monkeypatch)
    facts = wp.storage_facts(str(share))
    assert facts["write_reason"].startswith(f"{share} is owned by uid")
    assert facts["read_reason"].startswith(f"{share} is owned by uid")


def test_candidates_failing_on_the_same_directory_are_explained_once(tmp_path, monkeypatch):
    """BIOMNI_OUTPUT_ROOT and the workspace default usually fail together, on one mount."""
    workspace = _share(tmp_path, 0o500)
    try:
        monkeypatch.setenv("BIOMNI_OUTPUT_ROOT", str(workspace))
        target = wp.resolve_output_dir(wp.WorkspacePrefs(), workspace_root=str(workspace))
        assert target.source == "cwd-fallback"
        assert target.reason is not None
        assert target.reason.startswith(
            f"BIOMNI_OUTPUT_ROOT ({workspace}) and the workspace default ({workspace}/biomni-outputs) are not writable"
        )
        assert target.reason.count(f"{workspace} is owned by uid") == 1
    finally:
        workspace.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root is never denied by mode bits")
def test_three_locations_failing_together_read_as_a_list(tmp_path, monkeypatch):
    workspace = _share(tmp_path, 0o500)
    try:
        monkeypatch.setenv("BIOMNI_OUTPUT_ROOT", str(workspace / "root"))
        prefs = wp.WorkspacePrefs(output_dir=str(workspace / "mine"))
        target = wp.resolve_output_dir(prefs, workspace_root=str(workspace))
        assert target.reason is not None
        assert target.reason.startswith(
            f"The output directory you configured ({workspace}/mine), BIOMNI_OUTPUT_ROOT ({workspace}/root) and the "
            f"workspace default ({workspace}/biomni-outputs) are not writable"
        )
    finally:
        workspace.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root is never denied by mode bits")
def test_an_output_root_that_is_the_workspace_default_is_reported_once(tmp_path, monkeypatch):
    """The GRIP setup: BIOMNI_OUTPUT_ROOT=<workspace>/biomni-outputs."""
    workspace = _share(tmp_path, 0o500)
    try:
        monkeypatch.setenv("BIOMNI_OUTPUT_ROOT", str(workspace / "biomni-outputs"))
        target = wp.resolve_output_dir(wp.WorkspacePrefs(), workspace_root=str(workspace))
        assert target.reason is not None
        assert target.reason.startswith(f"BIOMNI_OUTPUT_ROOT ({workspace}/biomni-outputs) is not writable")
        assert "the workspace default" not in target.reason
    finally:
        workspace.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root is never denied by mode bits")
def test_each_passed_over_location_starts_a_sentence_of_its_own(tmp_path, monkeypatch):
    chosen = _share(tmp_path, 0o500)
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o500)
    try:
        monkeypatch.delenv("BIOMNI_OUTPUT_ROOT", raising=False)
        target = wp.resolve_output_dir(wp.WorkspacePrefs(output_dir=str(chosen)), workspace_root=str(workspace))
        assert target.reason is not None
        assert target.reason.startswith(f"The output directory you configured ({chosen}) is not writable")
        assert f". The workspace default ({workspace}/biomni-outputs) is not writable" in target.reason
    finally:
        chosen.chmod(0o755)
        workspace.chmod(0o755)


def test_a_directory_inside_the_image_is_answered_with_a_volume(tmp_path, monkeypatch):
    """A setting that points where no volume is mounted: opening up the image would keep nothing."""
    image = _share(tmp_path)
    _mount_table(tmp_path, monkeypatch, f"overlay {image} overlay rw,relatime 0 0")
    _as_grip_uid(monkeypatch)
    assert wp.explain_unwritable(str(image / "outputs")).endswith(
        f"inside the container image rather than on a mounted volume; mount a writable volume at {image}/outputs"
    )


def test_an_unwritable_location_is_logged_once_not_on_every_resolution(tmp_path, monkeypatch, caplog):
    ro = _share(tmp_path, 0o500)
    try:
        monkeypatch.setenv("BIOMNI_OUTPUT_ROOT", str(ro))
        with caplog.at_level(logging.WARNING, logger="biomni.workspace_prefs"):
            for _ in range(3):
                wp.resolve_output_dir(wp.WorkspacePrefs(), workspace_root=None)
        assert sum("BIOMNI_OUTPUT_ROOT" in r.getMessage() for r in caplog.records) == 1
    finally:
        ro.chmod(0o755)


def test_is_readable_dir(tmp_path):
    assert wp.is_readable_dir(str(tmp_path))
    assert not wp.is_readable_dir(None)
    assert not wp.is_readable_dir(str(tmp_path / "absent"))
    (tmp_path / "file").write_text("x")
    assert not wp.is_readable_dir(str(tmp_path / "file"))
