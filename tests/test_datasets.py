"""Tests for chainlit_ui.datasets - AD dataset prompt filtering."""

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


def test_markdown_shows_one_example_per_category(tmp_path: Path) -> None:
    """The Readme is a page about how to use the app, not a prompt catalogue.

    Both of these datasets are "Rare variants", so only the first contributes -
    the section is showing the shape of a good question, not enumerating one
    per file on disk.
    """
    for ds_id in ("NG00126", "RADR"):
        (tmp_path / ds_id).mkdir()
        (tmp_path / ds_id / "data.tsv").write_text("x")

    md = build_suggested_prompts_markdown(tmp_path)

    assert md.startswith("**Examples, using data you actually have:**")
    assert "NG00126" in md
    assert "RADR" not in md
    assert len([line for line in md.splitlines() if line.startswith("- ")]) == 1


def test_markdown_per_category_cap_is_adjustable(tmp_path: Path) -> None:
    for ds_id in ("NG00126", "RADR"):
        (tmp_path / ds_id).mkdir()
        (tmp_path / ds_id / "data.tsv").write_text("x")

    md = build_suggested_prompts_markdown(tmp_path, per_category=2)
    assert "NG00126" in md and "RADR" in md


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


def test_markdown_caps_the_total_number_of_examples(tmp_path: Path) -> None:
    """Brevity is the point; a full lake would otherwise contribute a dozen."""
    for ds_id, _prompt, _cat in AD_DATASET_PROMPTS:
        (tmp_path / ds_id).mkdir()
        (tmp_path / ds_id / "data.tsv").write_text("x")

    md = build_suggested_prompts_markdown(tmp_path)
    assert len([line for line in md.splitlines() if line.startswith("- ")]) == 5
