"""Tests for durable chat-thread storage.

Covers the parts that decide *where* conversations are written and *whether*
they can be read back - the failure modes here are silent by nature, because
Chainlit's data layer logs and continues on a SQL error rather than raising, so
a wrong path or a missing table looks like a working app that forgets
everything.

``build_data_layer`` itself needs Chainlit installed and is skipped when it is
not (CI installs no chainlit extra).
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from chainlit_ui import persistence

_ENV_TO_CLEAR = (
    "BIOMNI_THREADS_DB_URL",
    "BIOMNI_STATE_DIR",
    "BIOMNI_THREAD_ELEMENT_MAX_BYTES",
    "CHAINLIT_AUTH_SECRET",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# Location
# --------------------------------------------------------------------------- #


def test_state_dir_is_preferred(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMNI_STATE_DIR", str(tmp_path / "state"))
    url = persistence.resolve_database_url(str(tmp_path / "workspace"))
    assert url is not None
    assert str(tmp_path / "state" / "threads" / "threads.db") in url


def test_workspace_is_the_fallback(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    url = persistence.resolve_database_url(str(workspace))
    assert url is not None
    assert str(workspace / ".biomni" / "threads") in url


def test_no_location_means_no_history(tmp_path):
    assert persistence.resolve_database_url(None) is None


def test_explicit_url_wins_and_is_used_verbatim(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMNI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("BIOMNI_THREADS_DB_URL", "postgresql+asyncpg://db/threads")
    assert persistence.resolve_database_url(str(tmp_path)) == "postgresql+asyncpg://db/threads"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("sqlite+aiosqlite:////var/lib/x.db", "/var/lib/x.db"),
        ("sqlite+aiosqlite:///relative.db", "relative.db"),
        ("sqlite:////tmp/y.db?timeout=5", "/tmp/y.db"),
        ("postgresql+asyncpg://host/db", None),
    ],
)
def test_sqlite_path_extraction(url, expected):
    assert persistence.sqlite_path_from_url(url) == expected


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_schema_is_created_and_verified(tmp_path):
    db = tmp_path / "threads.db"
    assert persistence.ensure_sqlite_schema(str(db))
    assert persistence.verify_sqlite_schema(str(db)) == []


def test_schema_creation_is_idempotent(tmp_path):
    db = tmp_path / "threads.db"
    assert persistence.ensure_sqlite_schema(str(db))
    assert persistence.ensure_sqlite_schema(str(db))
    assert persistence.verify_sqlite_schema(str(db)) == []


def test_missing_tables_are_reported(tmp_path):
    """A half-built database must be detectable.

    Chainlit's data layer would log-and-continue against this, discarding every
    conversation while the UI looked fine.
    """
    db = tmp_path / "threads.db"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE users ("id" TEXT PRIMARY KEY)')
    missing = persistence.verify_sqlite_schema(str(db))
    assert "users" not in missing
    assert "threads" in missing and "steps" in missing


def test_schema_covers_every_column_chainlit_writes(tmp_path):
    """The data layer builds its INSERT from the live step/element dicts.

    An unknown column is not an error there - `execute_sql` catches it and logs
    - so the row simply vanishes. Pin the column sets against the TypedDicts
    Chainlit actually serialises.
    """
    chainlit_step = pytest.importorskip("chainlit.step", reason="chainlit not installed")
    chainlit_element = pytest.importorskip("chainlit.element", reason="chainlit not installed")

    db = tmp_path / "threads.db"
    persistence.ensure_sqlite_schema(str(db))
    with sqlite3.connect(db) as conn:
        step_cols = {row[1] for row in conn.execute("PRAGMA table_info(steps)")}
        element_cols = {row[1] for row in conn.execute("PRAGMA table_info(elements)")}

    # `feedback` and `modes` are read back from the join, never inserted;
    # `path` is consumed to read the bytes and dropped before the insert.
    step_fields = set(chainlit_step.StepDict.__annotations__) - {"feedback", "modes"}
    element_fields = set(chainlit_element.ElementDict.__annotations__) - {"path"}

    assert step_fields <= step_cols
    assert element_fields <= element_cols


def test_a_database_from_an_older_build_gains_new_columns(tmp_path):
    """The upgrade path that CREATE TABLE IF NOT EXISTS cannot serve.

    An existing file keeps its old table definition forever, and a step dict
    with one column the table lacks fails its INSERT - which the data layer
    logs and swallows, so the conversation simply stops being recorded. The
    databases at risk are exactly the ones with history in them.
    """
    db = tmp_path / "threads.db"
    persistence.ensure_sqlite_schema(str(db))
    with sqlite3.connect(db) as conn:
        for _, column, _ in persistence.SQLITE_ADDED_COLUMNS:
            conn.execute(f'ALTER TABLE steps DROP COLUMN "{column}"')
        conn.commit()

    assert persistence.ensure_sqlite_schema(str(db)) is True

    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(steps)")}
    assert {column for _, column, _ in persistence.SQLITE_ADDED_COLUMNS} <= columns


def test_reapplying_the_schema_is_idempotent(tmp_path):
    """It runs on every boot; a second pass must not fail on its own work."""
    db = tmp_path / "threads.db"
    assert persistence.ensure_sqlite_schema(str(db)) is True
    assert persistence.ensure_sqlite_schema(str(db)) is True
    assert persistence.verify_sqlite_schema(str(db)) == []


# --------------------------------------------------------------------------- #
# Auth secret
# --------------------------------------------------------------------------- #


def test_existing_secret_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "operator-supplied")
    monkeypatch.setenv("BIOMNI_STATE_DIR", str(tmp_path))
    assert persistence.ensure_auth_secret(None) == "operator-supplied"
    assert not (tmp_path / "threads" / persistence.AUTH_SECRET_FILENAME).exists()


def test_generated_secret_is_stable_across_restarts(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMNI_STATE_DIR", str(tmp_path))
    first = persistence.ensure_auth_secret(None)
    second = persistence.ensure_auth_secret(None)
    assert first and first == second


def test_generated_secret_is_not_world_readable(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMNI_STATE_DIR", str(tmp_path))
    persistence.ensure_auth_secret(None)
    path = tmp_path / "threads" / persistence.AUTH_SECRET_FILENAME
    assert path.exists()
    assert path.stat().st_mode & 0o077 == 0


def test_secret_falls_back_to_memory_without_storage():
    secret = persistence.ensure_auth_secret(None)
    assert secret


# --------------------------------------------------------------------------- #
# Element size guard
# --------------------------------------------------------------------------- #


class _Element:
    def __init__(self, *, path=None, content=None):
        self.path = path
        self.content = content
        self.name = "thing"


def test_payload_size_from_disk(tmp_path):
    target = tmp_path / "plot.png"
    target.write_bytes(b"x" * 1234)
    assert persistence.element_payload_size(_Element(path=str(target))) == 1234


def test_payload_size_from_content():
    assert persistence.element_payload_size(_Element(content="hello")) == 5
    assert persistence.element_payload_size(_Element(content=b"hello")) == 5


def test_payload_size_of_an_unreadable_path_is_zero(tmp_path):
    assert persistence.element_payload_size(_Element(path=str(tmp_path / "gone"))) == 0


def test_max_inline_bytes_is_configurable(monkeypatch):
    assert persistence.max_inline_bytes() == 2 * 1024 * 1024
    monkeypatch.setenv("BIOMNI_THREAD_ELEMENT_MAX_BYTES", "512")
    assert persistence.max_inline_bytes() == 512
    monkeypatch.setenv("BIOMNI_THREAD_ELEMENT_MAX_BYTES", "not-a-number")
    assert persistence.max_inline_bytes() == 2 * 1024 * 1024


def test_inline_storage_returns_a_data_url():
    storage = persistence.InlineBlobStorage()
    result = asyncio.run(storage.upload_file("user/el/plot.png", b"\x89PNG", mime="image/png"))
    assert result["url"] == "data:image/png;base64,iVBORw=="


def test_inline_storage_claims_no_object_key():
    """An object key makes reopening a thread break every image.

    The data layer reads it as "the bytes are somewhere addressable", drops the
    stored URL, and asks get_read_url to rebuild one from the key - which for a
    payload that lives in the row is not possible. Reporting no key keeps the
    read on the column holding the data URL.
    """
    storage = persistence.InlineBlobStorage()
    result = asyncio.run(storage.upload_file("user/el/plot.png", b"\x89PNG", mime="image/png"))
    assert not result.get("object_key")
    assert result  # still truthy: the data layer rejects an empty result


def test_inline_storage_refuses_to_invent_a_read_url():
    """Raising is what makes conversations written by the older code readable.

    The data layer catches it and falls back to the row's own URL, so an
    element that still carries an object key renders from its stored payload.
    """
    storage = persistence.InlineBlobStorage()
    with pytest.raises(NotImplementedError):
        asyncio.run(storage.get_read_url("user/el/plot.png"))
