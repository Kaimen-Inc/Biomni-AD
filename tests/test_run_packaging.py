"""Tests for packaging a finished run as a download.

The run directory lives on the server. Telling somebody its path only helps if
they have a shell there, which a demo attendee does not - so the archive is the
only route the report, the notebook and the generated files have off the box.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import chainlit_app
import pytest


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs" / "run_20260822_abc123_demo"
    d.mkdir(parents=True)
    (d / "report.md").write_text("# findings")
    (d / "trace.ipynb").write_text('{"cells": []}')
    (d / "plot.png").write_bytes(b"\x89PNG" + b"\0" * 100)
    sub = d / "tables"
    sub.mkdir()
    (sub / "results.csv").write_text("gene,logfc\nAPOE,2.4\n")
    return d


def test_packages_every_file_including_subdirectories(run_dir: Path) -> None:
    archive, note = chainlit_app._package_run_dir(str(run_dir), run_dir.name)

    assert archive is not None, note
    with zipfile.ZipFile(archive) as z:
        names = sorted(z.namelist())
    assert names == sorted(
        [
            f"{run_dir.name}/report.md",
            f"{run_dir.name}/trace.ipynb",
            f"{run_dir.name}/plot.png",
            f"{run_dir.name}/tables/results.csv",
        ]
    )


def test_archive_contents_survive_the_round_trip(run_dir: Path) -> None:
    archive, _ = chainlit_app._package_run_dir(str(run_dir), run_dir.name)
    with zipfile.ZipFile(archive) as z:
        assert z.read(f"{run_dir.name}/report.md").decode() == "# findings"


def test_archive_lands_outside_the_run_directory(run_dir: Path) -> None:
    """Otherwise it ends up inside the next archive of itself, and in the listing."""
    archive, _ = chainlit_app._package_run_dir(str(run_dir), run_dir.name)

    assert Path(archive).parent.name == ".packages"
    assert not str(Path(archive)).startswith(str(run_dir) + os.sep)
    assert not any(p.suffix == ".zip" for p in run_dir.rglob("*"))


def test_repackaging_is_stable(run_dir: Path) -> None:
    first, _ = chainlit_app._package_run_dir(str(run_dir), run_dir.name)
    second, _ = chainlit_app._package_run_dir(str(run_dir), run_dir.name)

    assert first == second
    with zipfile.ZipFile(second) as z:
        assert len(z.namelist()) == 4, "a re-run must not nest the previous archive"


def test_empty_run_offers_no_download(tmp_path: Path) -> None:
    d = tmp_path / "runs" / "run_empty"
    d.mkdir(parents=True)

    archive, note = chainlit_app._package_run_dir(str(d), "run_empty")

    assert archive is None
    assert "no files" in note


def test_missing_directory_is_not_fatal(tmp_path: Path) -> None:
    archive, note = chainlit_app._package_run_dir(str(tmp_path / "nope"), "run_x")
    assert archive is None
    assert "no run directory" in note


def test_oversized_run_is_reported_rather_than_zipped(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run can write a multi-GB intermediate; zipping it would stall the app."""
    monkeypatch.setattr(chainlit_app, "MAX_PACKAGE_BYTES", 10)

    archive, note = chainlit_app._package_run_dir(str(run_dir), run_dir.name)

    assert archive is None
    assert "too large" in note
    assert "MB" in note


def test_hidden_files_are_left_out(run_dir: Path) -> None:
    (run_dir / ".DS_Store").write_text("junk")
    hidden = run_dir / ".cache"
    hidden.mkdir()
    (hidden / "x.tmp").write_text("junk")

    archive, _ = chainlit_app._package_run_dir(str(run_dir), run_dir.name)

    with zipfile.ZipFile(archive) as z:
        assert not any(".DS_Store" in n or ".cache" in n for n in z.namelist())
