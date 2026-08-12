"""Tests for biomni.fs_scan - the bounded/cached workspace scanner.

These lock in the properties that keep agent bootstrap from hanging on a large
or network-backed workspace: a hard file-count cap, a wall-clock deadline, and a
short-TTL cache that collapses the several sidebar/inventory walks of one chat
start into a single traversal.
"""

from __future__ import annotations

import pytest
from biomni import fs_scan


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a cold cache and default (unset) env knobs."""
    for k in (
        "BIOMNI_WORKSPACE_MAX_FILES",
        "BIOMNI_WORKSPACE_SCAN_TIMEOUT_S",
        "BIOMNI_WORKSPACE_SCAN_TTL_S",
    ):
        monkeypatch.delenv(k, raising=False)
    fs_scan.clear_cache()
    yield
    fs_scan.clear_cache()


def _make_tree(root, n_files: int, *, subdir: str = "") -> None:
    base = root / subdir if subdir else root
    base.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (base / f"f{i}.txt").write_text("x")


# --------------------------------------------------------------------------- #
# Basic correctness
# --------------------------------------------------------------------------- #


def test_missing_path_returns_empty() -> None:
    result = fs_scan.scan_directory("/no/such/path/really")
    assert result.file_count == 0
    assert result.files == []
    assert not result.bounded


def test_none_path_returns_empty() -> None:
    assert fs_scan.scan_directory(None).file_count == 0


def test_small_tree_is_exact(tmp_path) -> None:
    _make_tree(tmp_path, 5)
    _make_tree(tmp_path, 3, subdir="nested")
    result = fs_scan.scan_directory(str(tmp_path))
    assert result.file_count == 8
    assert not result.bounded
    assert result.count_label() == "8"
    assert len(result.files) == 8
    assert "nested/f0.txt" in result.files


def test_hidden_files_and_excluded_dirs_skipped(tmp_path) -> None:
    (tmp_path / "keep.txt").write_text("x")
    (tmp_path / ".secret").write_text("x")  # hidden file
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x")  # excluded dir
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "m.pyc").write_text("x")
    result = fs_scan.scan_directory(str(tmp_path))
    assert result.file_count == 1
    assert result.files == ["keep.txt"]


def test_max_depth_prunes_deep_files(tmp_path) -> None:
    (tmp_path / "top.txt").write_text("x")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "deep.txt").write_text("x")
    shallow = fs_scan.scan_directory(str(tmp_path), max_depth=1)
    assert "top.txt" in shallow.files
    assert "a/b/c/deep.txt" not in shallow.files
    deep_scan = fs_scan.scan_directory(str(tmp_path), max_depth=10)
    assert "a/b/c/deep.txt" in deep_scan.files


def test_exclude_top_subdirs(tmp_path) -> None:
    _make_tree(tmp_path, 2, subdir="biomniAD")
    (tmp_path / "root.txt").write_text("x")
    result = fs_scan.scan_directory(str(tmp_path), exclude_top_subdirs={"biomniAD"})
    assert result.file_count == 1
    assert result.files == ["root.txt"]


def test_prune_subtrees_skips_nested_tree(tmp_path) -> None:
    (tmp_path / "user.txt").write_text("x")
    lake = tmp_path / "data" / "data_lake"
    _make_tree(lake, 4)
    result = fs_scan.scan_directory(str(tmp_path), prune_subtrees=[str(lake)])
    assert result.file_count == 1
    assert result.files == ["user.txt"]


# --------------------------------------------------------------------------- #
# Bounding: the core anti-hang guarantees
# --------------------------------------------------------------------------- #


def test_file_cap_truncates(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOMNI_WORKSPACE_MAX_FILES", "10")
    _make_tree(tmp_path, 50)
    result = fs_scan.scan_directory(str(tmp_path))
    assert result.truncated is True
    assert result.bounded is True
    assert result.file_count == 10  # a floor: stops exactly at the cap
    assert len(result.files) == 10
    assert result.count_label() == "10+"


def test_max_files_override(tmp_path) -> None:
    _make_tree(tmp_path, 50)
    result = fs_scan.scan_directory(str(tmp_path), max_files_override=7)
    assert result.truncated is True
    assert result.file_count == 7


def test_wall_clock_deadline_trips(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_tree(tmp_path, 20)

    # Deterministic clock: first reading is the scan start, every later reading
    # is well past the deadline, so the walk aborts on its first check.
    calls = {"n": 0}

    def fake_monotonic() -> float:
        calls["n"] += 1
        return 1000.0 if calls["n"] == 1 else 1_000_000.0

    monkeypatch.setattr(fs_scan.time, "monotonic", fake_monotonic)
    result = fs_scan.scan_directory(str(tmp_path))
    assert result.timed_out is True
    assert result.bounded is True


# --------------------------------------------------------------------------- #
# Caching: collapse repeated walks within one chat start
# --------------------------------------------------------------------------- #


def test_repeated_scan_is_cached(tmp_path) -> None:
    _make_tree(tmp_path, 3)
    first = fs_scan.scan_directory(str(tmp_path))
    second = fs_scan.scan_directory(str(tmp_path))
    assert first is second  # served from cache, tree walked once


def test_clear_cache_forces_rescan(tmp_path) -> None:
    _make_tree(tmp_path, 3)
    first = fs_scan.scan_directory(str(tmp_path))
    fs_scan.clear_cache()
    second = fs_scan.scan_directory(str(tmp_path))
    assert first is not second


def test_distinct_params_do_not_collide(tmp_path) -> None:
    _make_tree(tmp_path, 3)
    a = fs_scan.scan_directory(str(tmp_path), max_depth=1)
    b = fs_scan.scan_directory(str(tmp_path), max_depth=5)
    assert a is not b  # different key => independent cache entries


# --------------------------------------------------------------------------- #
# Review-fix regressions
# --------------------------------------------------------------------------- #


def test_runs_dir_is_excluded(tmp_path) -> None:
    """`runs/` (Biomni per-run output history) must not consume the scan budget."""
    (tmp_path / "dataset.csv").write_text("x")
    _make_tree(tmp_path, 20, subdir="runs")
    result = fs_scan.scan_directory(str(tmp_path))
    assert result.file_count == 1
    assert result.files == ["dataset.csv"]


@pytest.mark.parametrize(
    ("var", "getter", "default"),
    [
        ("BIOMNI_WORKSPACE_MAX_FILES", fs_scan.max_files, 10_000),
        ("BIOMNI_WORKSPACE_SCAN_TIMEOUT_S", fs_scan.scan_timeout_s, 15.0),
    ],
)
def test_nonpositive_env_falls_back_to_default(monkeypatch, var, getter, default) -> None:
    """0 / negative must NOT disable the guard - it falls back to the default."""
    for bad in ("0", "-5"):
        monkeypatch.setenv(var, bad)
        assert getter() == default
    monkeypatch.setenv(var, "garbage")
    assert getter() == default
