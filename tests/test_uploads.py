"""Tests for chainlit_ui.uploads - keeping attachments after the session ends.

The naming rules carry the weight here: the filename comes from a machine this
process does not control, it becomes a path, and it is echoed into prompts.
"""

from __future__ import annotations

import os

from chainlit_ui import uploads

# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #


def test_ordinary_names_are_left_alone():
    assert uploads.safe_filename("cohort_2024.csv") == "cohort_2024.csv"
    assert uploads.safe_filename("NG00105-eQTL.tsv.gz") == "NG00105-eQTL.tsv.gz"


def test_directory_traversal_is_stripped():
    assert uploads.safe_filename("../../etc/passwd") == "passwd"
    assert uploads.safe_filename("/etc/shadow") == "shadow"
    assert uploads.safe_filename("..") == uploads._FALLBACK_NAME


def test_windows_paths_are_reduced_to_the_filename():
    """A Windows path arrives as a single segment on POSIX, so basename alone
    would keep the drive and folders as part of the name."""
    assert uploads.safe_filename(r"C:\\Users\\kuan\\data.csv") == "data.csv"


def test_shell_and_space_characters_are_replaced():
    assert uploads.safe_filename("my data; rm -rf *.csv") == "my_data_rm_-rf_.csv"


def test_empty_and_missing_names_get_a_fallback():
    assert uploads.safe_filename("") == uploads._FALLBACK_NAME
    assert uploads.safe_filename(None) == uploads._FALLBACK_NAME
    assert uploads.safe_filename("   ") == uploads._FALLBACK_NAME


def test_very_long_names_are_truncated_but_keep_their_extension():
    name = "x" * 500 + ".parquet"
    result = uploads.safe_filename(name)
    assert len(result) <= uploads._MAX_NAME_LEN
    assert result.endswith(".parquet")


# --------------------------------------------------------------------------- #
# Collisions
# --------------------------------------------------------------------------- #


def test_a_free_name_is_used_as_is(tmp_path):
    assert uploads.unique_destination(str(tmp_path), "a.csv") == str(tmp_path / "a.csv")


def test_a_taken_name_is_numbered(tmp_path):
    (tmp_path / "a.csv").write_text("first")
    assert uploads.unique_destination(str(tmp_path), "a.csv") == str(tmp_path / "a (2).csv")
    (tmp_path / "a (2).csv").write_text("second")
    assert uploads.unique_destination(str(tmp_path), "a.csv") == str(tmp_path / "a (3).csv")


# --------------------------------------------------------------------------- #
# Copying
# --------------------------------------------------------------------------- #


def test_upload_is_copied_under_its_real_name(tmp_path):
    """The point of the whole module: Chainlit hands over a UUID path, and the
    name the user chose is usually what the file means."""
    scratch = tmp_path / ".files" / "session"
    scratch.mkdir(parents=True)
    source = scratch / "ef4edb5c-f251-465d-8bd0-9d2531dc68e2.csv"
    source.write_text("gene,score\nAPOE,0.9\n")

    out = tmp_path / "outputs" / uploads.UPLOADS_DIRNAME
    stored = uploads.store_upload(str(source), str(out), "cohort.csv")

    assert stored == str(out / "cohort.csv")
    assert (out / "cohort.csv").read_text().startswith("gene,score")
    # The original is left in place: Chainlit owns that copy and serves the
    # browser from it.
    assert source.exists()


def test_the_source_is_not_moved_or_altered(tmp_path):
    source = tmp_path / "src.csv"
    source.write_text("x")
    uploads.store_upload(str(source), str(tmp_path / "out"), "src.csv")
    assert source.read_text() == "x"


def test_a_missing_source_yields_none(tmp_path):
    assert uploads.store_upload(str(tmp_path / "gone.csv"), str(tmp_path / "out"), "gone.csv") is None
    assert uploads.store_upload("", str(tmp_path / "out")) is None


def test_an_unwritable_destination_yields_none(tmp_path):
    source = tmp_path / "src.csv"
    source.write_text("x")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        assert uploads.store_upload(str(source), str(locked / "uploads"), "src.csv") is None
    finally:
        locked.chmod(0o700)


def test_store_uploads_falls_back_per_file(tmp_path):
    """A partial failure must not cost the agent the other attachments."""
    good = tmp_path / "good.csv"
    good.write_text("x")
    missing = str(tmp_path / "missing.csv")
    out = tmp_path / "out"

    stored = uploads.store_uploads([(str(good), "good.csv"), (missing, "missing.csv")], str(out))

    assert stored[0] == str(out / "good.csv")
    assert stored[1] == missing  # unchanged, still usable for this message


def test_two_attachments_with_one_name_do_not_overwrite(tmp_path):
    first = tmp_path / "a" / "results.csv"
    second = tmp_path / "b" / "results.csv"
    for path, body in ((first, "one"), (second, "two")):
        path.parent.mkdir(parents=True)
        path.write_text(body)

    out = tmp_path / "out"
    stored = uploads.store_uploads([(str(first), "results.csv"), (str(second), "results.csv")], str(out))

    assert [os.path.basename(p) for p in stored] == ["results.csv", "results (2).csv"]
    assert (out / "results.csv").read_text() == "one"
    assert (out / "results (2).csv").read_text() == "two"


def test_uploads_dir_hangs_off_the_output_directory():
    assert uploads.uploads_dir("/data/outputs") == os.path.join("/data/outputs", "uploads")
