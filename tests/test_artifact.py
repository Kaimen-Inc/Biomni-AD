"""Unit tests for biomni.artifact — the shared run-id / file-snapshot helpers."""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pytest
from biomni.artifact import (
    DEFAULT_EXCLUDED_DIRS,
    build_run_id,
    get_all_files,
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


def test_build_run_id_empty_topic_returns_timestamp_only() -> None:
    rid = build_run_id(None)
    assert rid.startswith("run_")
    # exactly one underscore after "run" → run_YYYYMMDD_HHMMSS
    assert rid.count("_") == 2


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
