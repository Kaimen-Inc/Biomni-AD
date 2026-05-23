"""Tests for chainlit_ui.datasets — AD dataset prompt filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from chainlit_ui.datasets import (
    AD_DATASET_PROMPTS,
    build_suggested_prompts_markdown,
    discover_present_dataset_ids,
)

if TYPE_CHECKING:
    from pathlib import Path

# --- discover_present_dataset_ids ---------------------------------------------


def test_discover_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    assert discover_present_dataset_ids(tmp_path / "does-not-exist") == set()


def test_discover_returns_empty_for_empty_lake(tmp_path: Path) -> None:
    assert discover_present_dataset_ids(tmp_path) == set()


def test_discover_skips_directories_with_only_readme(tmp_path: Path) -> None:
    (tmp_path / "DS1").mkdir()
    (tmp_path / "DS1" / "README.md").write_text("docs only")
    (tmp_path / "DS1" / "readme.txt").write_text("more docs")  # case-insensitive

    assert discover_present_dataset_ids(tmp_path) == set()


def test_discover_returns_ids_with_real_data(tmp_path: Path) -> None:
    (tmp_path / "DS_with_data").mkdir()
    (tmp_path / "DS_with_data" / "summary.tsv").write_text("data")
    (tmp_path / "DS_readme_only").mkdir()
    (tmp_path / "DS_readme_only" / "README.md").write_text("docs")
    (tmp_path / "loose_file.txt").write_text("not a dataset dir")  # files at root ignored

    assert discover_present_dataset_ids(tmp_path) == {"DS_with_data"}


def test_discover_ignores_files_at_lake_root(tmp_path: Path) -> None:
    """Only direct sub-directories count as datasets."""
    (tmp_path / "stray.csv").write_text("data")
    assert discover_present_dataset_ids(tmp_path) == set()


# --- build_suggested_prompts_markdown -----------------------------------------


def test_markdown_is_empty_when_lake_missing(tmp_path: Path) -> None:
    assert build_suggested_prompts_markdown(tmp_path / "missing") == ""


def test_markdown_is_empty_when_no_datasets_match(tmp_path: Path) -> None:
    # A dataset is present but its id is not in AD_DATASET_PROMPTS
    (tmp_path / "UNKNOWN_DS").mkdir()
    (tmp_path / "UNKNOWN_DS" / "data.csv").write_text("x")
    assert build_suggested_prompts_markdown(tmp_path) == ""


def test_markdown_lists_matched_datasets_grouped_by_category(tmp_path: Path) -> None:
    # Set up two datasets from different categories
    for ds_id in ("NG00126", "RADR"):
        (tmp_path / ds_id).mkdir()
        (tmp_path / ds_id / "data.tsv").write_text("x")

    md = build_suggested_prompts_markdown(tmp_path)

    assert md.startswith("**Suggested prompts based on your local data:**")
    assert "*Rare variants*" in md
    # Both should be present since they share the "Rare variants" category
    assert "NG00126" in md
    assert "TREM2 and APOE rare variants in the RADR" in md


def test_markdown_only_includes_locally_present_ids(tmp_path: Path) -> None:
    (tmp_path / "NG00052").mkdir()
    (tmp_path / "NG00052" / "summary.tsv").write_text("x")
    # All other dataset dirs absent

    md = build_suggested_prompts_markdown(tmp_path)

    assert "NG00052" in md
    # Other dataset ids should not appear at all
    for ds_id, _prompt, _cat in AD_DATASET_PROMPTS:
        if ds_id != "NG00052":
            assert ds_id not in md, f"{ds_id} should not appear when not on disk"


# --- AD_DATASET_PROMPTS constant ----------------------------------------------


@pytest.mark.parametrize("entry", AD_DATASET_PROMPTS)
def test_prompt_entries_are_well_formed(entry: tuple[str, str, str]) -> None:
    ds_id, prompt_text, category = entry
    assert isinstance(ds_id, str) and ds_id
    assert isinstance(prompt_text, str) and prompt_text
    assert isinstance(category, str) and category


def test_dataset_ids_are_unique() -> None:
    ids = [e[0] for e in AD_DATASET_PROMPTS]
    assert len(ids) == len(set(ids)), "duplicate dataset id in AD_DATASET_PROMPTS"
