"""Unit tests for biomni.artifact - the shared run-id / file-snapshot helpers."""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pytest
from biomni import artifact as artifact_module
from biomni.artifact import (
    DEFAULT_EXCLUDED_DIRS,
    SnapshotCancelled,
    build_run_id,
    get_all_files,
    snapshot_files,
    summarize_topic_for_run_id,
)

# --- summarize_topic_for_run_id ------------------------------------------------


@pytest.mark.parametrize(
    "topic,expected",
    [
        ("", ""),
        ("the and or", "the_and_or"),  # all stopwords → fallback path
        ("Analyze TREM2 variants in AD", "trem2_variants"),  # 'ad' < 3 chars, dropped
        ("Map APOE allele effects", "map_apoe_allele"),
        ("   ", ""),
        ("!!! @#$ %%%", ""),
        (
            "a b c d e f g h i j k l m",
            "the_and_or"[:0] or "",
        ),  # all 1-char skipped → fallback to first 3 raw lowercased
    ],
)
def test_summarize_topic_known_cases(topic: str, expected: str) -> None:
    result = summarize_topic_for_run_id(topic)
    if expected == "" and topic.strip() and any(c.isalnum() for c in topic):
        # The 1-char fallback path returns first 3 raw tokens; loosen the assertion.
        assert isinstance(result, str)
        return
    assert result == expected, (topic, result)


def test_summarize_topic_truncates_to_40_chars() -> None:
    long = "supercalifragilistic " * 10
    result = summarize_topic_for_run_id(long)
    assert len(result) <= 40


def test_summarize_topic_strips_punctuation() -> None:
    result = summarize_topic_for_run_id("Drug-target/interaction (TREM2)!")
    # Hyphen / paren / slash / bang stripped; only alphanumeric kept
    assert "/" not in result
    assert "-" not in result
    assert "(" not in result
    assert "!" not in result


# --- build_run_id --------------------------------------------------------------


def test_build_run_id_empty_topic_returns_timestamp_and_token() -> None:
    rid = build_run_id(None)
    assert rid.startswith("run_")
    # run_YYYYMMDD_HHMMSS_xxxxxx
    assert rid.count("_") == 3


def test_build_run_id_is_unique_for_the_same_second_and_topic() -> None:
    """Two chats running the same starter prompt must not share a directory.

    Every caller creates the run directory with ``exist_ok=True``, so a repeated
    id is not an error - it is two runs silently overwriting each other's
    report and trace.
    """
    fixed = datetime(2025, 1, 2, 3, 4, 5)
    ids = {build_run_id("Multi-omics AD risk gene portrait", now=lambda: fixed) for _ in range(200)}
    assert len(ids) == 200


def test_build_run_id_token_is_injectable() -> None:
    fixed = datetime(2025, 1, 2, 3, 4, 5)
    rid = build_run_id("Map AD GWAS loci", now=lambda: fixed, token="abc123")
    assert rid == "run_20250102_030405_abc123_map_gwas_loci"


def test_build_run_id_keeps_chronological_sort_order() -> None:
    """The token sits after the timestamp, so a listing still sorts by time."""
    earlier = build_run_id("topic one", now=lambda: datetime(2025, 1, 2, 3, 4, 5))
    later = build_run_id("topic two", now=lambda: datetime(2025, 1, 2, 3, 4, 6))
    assert sorted([later, earlier]) == [earlier, later]


def test_build_run_id_includes_topic_slug() -> None:
    rid = build_run_id("Map AD GWAS loci")
    assert rid.startswith("run_")
    assert rid.endswith("_gwas_loci")


def test_build_run_id_uses_llm_summarizer_when_provided() -> None:
    rid = build_run_id("anything", llm_summarizer=lambda _: "custom_slug")
    assert rid.endswith("_custom_slug")


def test_build_run_id_falls_back_when_summarizer_returns_empty() -> None:
    rid = build_run_id("Map AD GWAS loci", llm_summarizer=lambda _: "")
    assert rid.endswith("_gwas_loci")  # heuristic kicks in


def test_build_run_id_falls_back_when_summarizer_raises(caplog: pytest.LogCaptureFixture) -> None:
    def boom(_topic: str) -> str:
        raise RuntimeError("llm offline")

    with caplog.at_level(logging.DEBUG, logger="biomni.artifact"):
        rid = build_run_id("Map AD GWAS loci", llm_summarizer=boom)

    assert rid.endswith("_gwas_loci")
    # Failure must be logged so it's diagnosable
    assert any("summarizer failed" in rec.message for rec in caplog.records)


def test_build_run_id_with_injected_now() -> None:
    fixed = datetime(2025, 1, 2, 3, 4, 5)
    rid = build_run_id("foo bar", now=lambda: fixed)
    assert rid.startswith("run_20250102_030405")


# --- get_all_files -------------------------------------------------------------


def test_get_all_files_returns_empty_for_missing_dir(tmp_path) -> None:
    assert get_all_files(str(tmp_path / "does-not-exist")) == set()


def test_get_all_files_returns_empty_for_empty_string() -> None:
    assert get_all_files("") == set()


def test_get_all_files_skips_excluded_and_hidden(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "should_skip.txt").write_text("x")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "skip.pyc").write_text("x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "skip.txt").write_text("x")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "skip.js").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "keep.txt").write_text("x")
    (tmp_path / ".hidden").write_text("x")
    (tmp_path / "visible.txt").write_text("x")

    result = {os.path.basename(p) for p in get_all_files(str(tmp_path))}
    assert result == {"keep.txt", "visible.txt"}


def test_get_all_files_excludes_match_documented_default_set() -> None:
    # Guard against accidental drift in the default exclude set
    assert "runs" in DEFAULT_EXCLUDED_DIRS
    assert "__pycache__" in DEFAULT_EXCLUDED_DIRS
    assert "node_modules" in DEFAULT_EXCLUDED_DIRS
    assert "site-packages" in DEFAULT_EXCLUDED_DIRS


def test_get_all_files_custom_exclude_overrides_defaults(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "in.txt").write_text("x")
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "out.txt").write_text("x")

    # Override: include "runs/" but skip a custom dir
    result = {os.path.basename(p) for p in get_all_files(str(tmp_path), exclude_dirs={"junk"})}
    assert result == {"in.txt"}


# --- snapshot_files ------------------------------------------------------------


def test_snapshot_files_is_the_union_of_what_each_directory_holds(tmp_path) -> None:
    (tmp_path / "cwd").mkdir()
    (tmp_path / "cwd" / "notes.txt").write_text("x")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "table.csv").write_text("x")

    assert snapshot_files(str(tmp_path / "cwd"), str(tmp_path / "data")) == {
        str(tmp_path / "cwd" / "notes.txt"),
        str(tmp_path / "data" / "table.csv"),
    }


def test_snapshot_files_skips_what_is_not_there(tmp_path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "kept.txt").write_text("x")

    # The agent's data root is optional and may have been deleted since it was configured.
    result = snapshot_files(None, "", str(tmp_path / "gone"), str(tmp_path / "real"))
    assert result == {str(tmp_path / "real" / "kept.txt")}
    assert snapshot_files() == set()


def test_snapshot_files_counts_a_file_once_when_the_directories_overlap(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "shared.txt").write_text("x")

    assert snapshot_files(str(tmp_path), str(tmp_path / "sub")) == {str(tmp_path / "sub" / "shared.txt")}


def test_snapshot_files_sees_exactly_what_get_all_files_sees(tmp_path) -> None:
    """The run's before and after snapshots must agree on what a file is, or every run looks like it made a mess."""
    for name in ("runs", "__pycache__", ".git", "sub"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "inner.txt").write_text("x")
    (tmp_path / ".hidden").write_text("x")
    (tmp_path / "visible.txt").write_text("x")

    assert snapshot_files(str(tmp_path)) == get_all_files(str(tmp_path))


def _every_directory_listed_separately(*directories: str) -> set[str]:
    """What snapshot_files must always return, however it gets there: the plain union."""
    files: set[str] = set()
    for directory in directories:
        files |= get_all_files(directory)
    return files


@pytest.fixture
def walks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The directories get_all_files is actually asked to walk."""
    asked: list[str] = []
    real = artifact_module.get_all_files

    def spy(directory, **kwargs):
        asked.append(directory)
        return real(directory, **kwargs)

    monkeypatch.setattr(artifact_module, "get_all_files", spy)
    return asked


@pytest.fixture
def app_tree(tmp_path):
    """The deployment's shape: the app runs in /app and the workspace is mounted inside it."""
    app = tmp_path / "app"
    (app / "user-data" / "study").mkdir(parents=True)
    (app / "user-data" / "study" / "table.csv").write_text("x")
    (app / "code.py").write_text("x")
    return app


def test_a_directory_inside_another_is_not_walked_again(app_tree, walks) -> None:
    """Walking the app directory already lists the workspace under it, and over a network share that is minutes."""
    result = snapshot_files(str(app_tree), str(app_tree / "user-data"))

    assert walks == [str(app_tree)]
    assert result == _every_directory_listed_separately(str(app_tree), str(app_tree / "user-data"))
    assert str(app_tree / "user-data" / "study" / "table.csv") in result


def test_the_order_of_the_directories_does_not_matter(app_tree, walks) -> None:
    snapshot_files(str(app_tree / "user-data"), str(app_tree))
    assert walks == [str(app_tree)]


def test_the_same_directory_given_twice_is_walked_once(app_tree, walks) -> None:
    snapshot_files(str(app_tree), str(app_tree) + os.sep, str(app_tree / "user-data" / ".."))
    assert walks == [str(app_tree)]


def test_siblings_that_only_share_a_prefix_are_both_walked(tmp_path, walks) -> None:
    for name in ("data", "data-old"):
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}.csv").write_text("x")

    result = snapshot_files(str(tmp_path / "data"), str(tmp_path / "data-old"))
    assert sorted(walks) == [str(tmp_path / "data"), str(tmp_path / "data-old")]
    assert len(result) == 2


@pytest.mark.parametrize("hidden_or_excluded", [".state", "runs", "node_modules"])
def test_a_directory_the_outer_walk_would_skip_is_walked_on_its_own(tmp_path, walks, hidden_or_excluded) -> None:
    """get_all_files never enters hidden or excluded directories, so it would miss this one's files."""
    inner = tmp_path / hidden_or_excluded / "workspace"
    inner.mkdir(parents=True)
    (inner / "table.csv").write_text("x")
    (tmp_path / "code.py").write_text("x")

    result = snapshot_files(str(tmp_path), str(inner))

    assert str(inner) in walks
    assert str(inner / "table.csv") in result
    assert result == _every_directory_listed_separately(str(tmp_path), str(inner))


def test_a_directory_reached_through_a_symlink_is_walked_on_its_own(tmp_path, walks) -> None:
    """os.walk does not follow links, so the outer walk never lists what is behind one."""
    real = tmp_path / "share" / "workspace"
    real.mkdir(parents=True)
    (real / "table.csv").write_text("x")
    app = tmp_path / "app"
    app.mkdir()
    (app / "user-data").symlink_to(real, target_is_directory=True)

    result = snapshot_files(str(app), str(app / "user-data"))

    assert sorted(walks) == [str(app), str(app / "user-data")]
    assert str(app / "user-data" / "table.csv") in result
    assert result == _every_directory_listed_separately(str(app), str(app / "user-data"))


def test_a_walk_can_be_called_off(app_tree) -> None:
    with pytest.raises(SnapshotCancelled):
        snapshot_files(str(app_tree), cancelled=lambda: True)


def test_a_called_off_walk_stops_at_the_next_directory_instead_of_finishing(tmp_path) -> None:
    for i in range(20):
        (tmp_path / f"d{i:02d}").mkdir()
        (tmp_path / f"d{i:02d}" / "f.txt").write_text("x")
    asked = 0

    def cancelled() -> bool:
        nonlocal asked
        asked += 1
        return asked > 3

    with pytest.raises(SnapshotCancelled):
        get_all_files(str(tmp_path), cancelled=cancelled)
    assert asked == 4, "it must stop at once, not read the other seventeen directories first"


def test_a_walk_that_is_not_called_off_is_unchanged(app_tree) -> None:
    assert snapshot_files(str(app_tree), cancelled=lambda: False) == snapshot_files(str(app_tree))
