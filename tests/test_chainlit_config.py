"""Guards on .chainlit/config.toml - the settings that silently break the UI.

Chainlit's config is not validated against the frontend that consumes it, so a
wrong value here does not fail at startup, does not log an error, and does not
raise in the browser. It just makes a control stop working. These tests pin the
ones that have already done that.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).resolve().parent.parent / ".chainlit" / "config.toml"


@pytest.fixture(scope="module")
def config() -> dict:
    if not CONFIG_PATH.is_file():
        pytest.skip("no .chainlit/config.toml in this checkout")
    return tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def upload(config: dict) -> dict:
    return config.get("features", {}).get("spontaneous_file_upload", {})


def test_upload_accept_is_a_mapping_not_a_wildcard(upload: dict) -> None:
    """``accept = ["*/*"]`` silently disables the 📎 button.

    The chain: the frontend drops "*/*" as an invalid MIME type, which leaves
    the File System Access API options with an EMPTY accept dictionary; Chrome
    rejects that with an AbortError; and react-dropzone reads AbortError as "the
    user cancelled the file dialog" and returns quietly. The result is a button
    that does nothing at all, with no error anywhere - which is exactly how it
    was reported.
    """
    accept = upload.get("accept")
    assert isinstance(accept, dict), f"accept must be a MIME-type -> extensions mapping, got {accept!r}"
    assert accept, "an empty accept mapping is what the broken wildcard produced"


def test_upload_accept_entries_are_well_formed(upload: dict) -> None:
    """Every entry must survive the frontend's own filter.

    It drops any key that is not `type/subtype` and any value that is not a list
    of dotted extensions - each one narrowing what the picker will accept, and
    dropping all of them recreates the empty-dictionary failure.
    """
    for mime, extensions in upload["accept"].items():
        assert mime.count("/") == 1 and all(mime.split("/")), f"{mime!r} is not a MIME type"
        assert "*" not in mime, f"{mime!r}: wildcards are rejected by the frontend"
        assert isinstance(extensions, list) and extensions, f"{mime!r} needs at least one extension"
        for ext in extensions:
            assert ext.startswith(".") and len(ext) > 1, f"{ext!r} is not a file extension"


def test_upload_is_enabled(upload: dict) -> None:
    assert upload.get("enabled") is True


def test_upload_limits_are_sane(upload: dict) -> None:
    assert upload.get("max_files", 0) >= 1
    assert upload.get("max_size_mb", 0) >= 1
