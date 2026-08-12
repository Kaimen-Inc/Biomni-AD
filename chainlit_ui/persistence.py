"""Durable chat threads - the conversation list users expect on the left.

Chainlit renders a thread history sidebar, and lets a user reopen and continue
an old conversation, only when two things are true: an authenticated user, and
a configured data layer. This module supplies both halves for Biomni-AD.

Storage follows the same ladder as preferences and run records, so an operator
configures one volume and everything lands on it:

    ``BIOMNI_THREADS_DB_URL`` -> ``BIOMNI_STATE_DIR/threads/threads.db``
    -> ``<workspace>/.biomni/threads/threads.db`` -> no persistence

The default is SQLite because it needs no infrastructure and a single replica is
already assumed elsewhere (file-backed preferences have the same constraint).
Point ``BIOMNI_THREADS_DB_URL`` at ``postgresql+asyncpg://...`` when scaling out;
Chainlit's data layer speaks both. The schema is only auto-created for SQLite -
for a shared database, creating tables is a migration an operator owns.

Nothing here imports Chainlit at module scope: CI installs no chainlit extra and
must still be able to import and test the pure parts (URL resolution, DDL,
secret handling).
"""

from __future__ import annotations

import base64
import logging
import os
import secrets
import sqlite3
from typing import Any

from biomni.workspace_prefs import WORKSPACE_STATE_DIRNAME, is_writable_dir

logger = logging.getLogger(__name__)

THREADS_DIRNAME = "threads"
THREADS_DB_FILENAME = "threads.db"
AUTH_SECRET_FILENAME = "auth_secret"

# Elements are persisted inline (see InlineBlobStorage). Anything larger than
# this is dropped from the archive rather than bloating the database; the file
# itself still exists in the run's output directory.
_DEFAULT_MAX_INLINE_BYTES = 2 * 1024 * 1024

# How long a blocked SQLite writer waits for the lock before giving up.
_SQLITE_BUSY_TIMEOUT_S = 30.0


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

# Chainlit's data layer issues raw SQL against these tables and quotes every
# identifier, so the camelCase column names are load-bearing. Types are kept
# permissive (TEXT) because SQLite has no native JSON/array type and the layer
# serialises those fields itself before binding them.
#
# `execute_sql` swallows every exception into a log warning, so a missing column
# does not raise - it silently drops the row. That is why the column list here
# mirrors StepDict/ElementDict exactly rather than only the fields we expect to
# see, and why `verify_schema` exists.
SQLITE_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS users (
        "id"          TEXT PRIMARY KEY,
        "identifier"  TEXT NOT NULL UNIQUE,
        "metadata"    TEXT NOT NULL,
        "createdAt"   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS threads (
        "id"             TEXT PRIMARY KEY,
        "createdAt"      TEXT,
        "name"           TEXT,
        "userId"         TEXT,
        "userIdentifier" TEXT,
        "tags"           TEXT,
        "metadata"       TEXT,
        FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS steps (
        "id"            TEXT PRIMARY KEY,
        "name"          TEXT NOT NULL,
        "type"          TEXT NOT NULL,
        "threadId"      TEXT NOT NULL,
        "parentId"      TEXT,
        "streaming"     BOOLEAN NOT NULL,
        "waitForAnswer" BOOLEAN,
        "isError"       BOOLEAN,
        "metadata"      TEXT,
        "tags"          TEXT,
        "input"         TEXT,
        "output"        TEXT,
        "createdAt"     TEXT,
        "command"       TEXT,
        "start"         TEXT,
        "end"           TEXT,
        "generation"    TEXT,
        "showInput"     TEXT,
        "language"      TEXT,
        "indent"        INTEGER,
        "defaultOpen"   BOOLEAN,
        "autoCollapse"  BOOLEAN,
        FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS elements (
        "id"           TEXT PRIMARY KEY,
        "threadId"     TEXT,
        "type"         TEXT,
        "url"          TEXT,
        "chainlitKey"  TEXT,
        "name"         TEXT NOT NULL,
        "display"      TEXT,
        "objectKey"    TEXT,
        "size"         TEXT,
        "page"         INTEGER,
        "language"     TEXT,
        "forId"        TEXT,
        "mime"         TEXT,
        "props"        TEXT,
        "autoPlay"     BOOLEAN,
        "playerConfig" TEXT,
        FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feedbacks (
        "id"       TEXT PRIMARY KEY,
        "forId"    TEXT NOT NULL,
        "threadId" TEXT NOT NULL,
        "value"    INTEGER NOT NULL,
        "comment"  TEXT
    )
    """,
    'CREATE INDEX IF NOT EXISTS ix_threads_user ON threads("userId")',
    'CREATE INDEX IF NOT EXISTS ix_steps_thread ON steps("threadId")',
    'CREATE INDEX IF NOT EXISTS ix_elements_thread ON elements("threadId")',
)

REQUIRED_TABLES = ("users", "threads", "steps", "elements", "feedbacks")

# Columns that Chainlit releases added after the CREATE TABLE statements above
# were written, applied to databases that already exist.
#
# This is not cosmetic. The data layer builds its INSERT from the keys of the
# live step dict, so one unknown column fails the whole statement - and
# `execute_sql` catches the error and logs it, so the step is lost without
# anything surfacing. A single missing column silently empties the transcript.
# `autoCollapse` arrived in chainlit 2.10, inside the ">=2.8,<3" this project
# allows, which is close enough to reach a deployment on the next `pip install`.
SQLITE_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (("steps", "autoCollapse", "BOOLEAN"),)


# --------------------------------------------------------------------------- #
# Location
# --------------------------------------------------------------------------- #


def resolve_threads_dir(workspace_root: str | None = None) -> str | None:
    """Directory for the thread database, or ``None`` when nothing is writable."""
    state_dir = os.getenv("BIOMNI_STATE_DIR", "").strip()
    if state_dir:
        candidate = os.path.join(state_dir, THREADS_DIRNAME)
        if is_writable_dir(candidate):
            return candidate
        logger.warning("BIOMNI_STATE_DIR=%s is not writable; chat history will not persist there", state_dir)

    if workspace_root:
        candidate = os.path.join(workspace_root, WORKSPACE_STATE_DIRNAME, THREADS_DIRNAME)
        if is_writable_dir(candidate):
            return candidate

    return None


def resolve_database_url(workspace_root: str | None = None) -> str | None:
    """Async SQLAlchemy URL for the thread store, or ``None`` to disable it.

    An explicit ``BIOMNI_THREADS_DB_URL`` wins and is used verbatim, so a
    deployment can point at Postgres without this module knowing anything about
    it.
    """
    explicit = os.getenv("BIOMNI_THREADS_DB_URL", "").strip()
    if explicit:
        return explicit

    directory = resolve_threads_dir(workspace_root)
    if not directory:
        return None
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        logger.warning("could not create %s; chat history will not persist", directory, exc_info=True)
        return None
    return "sqlite+aiosqlite:///" + os.path.join(directory, THREADS_DB_FILENAME)


def sqlite_path_from_url(url: str) -> str | None:
    """The on-disk file behind a SQLite URL, or ``None`` for any other backend.

    ``sqlite+aiosqlite:////var/lib/x.db`` -> ``/var/lib/x.db`` (four slashes: the
    scheme's three plus the path's own leading one), and
    ``sqlite+aiosqlite:///x.db`` -> ``x.db``, relative to the process directory.
    """
    if not url.startswith("sqlite"):
        return None
    _, separator, rest = url.partition(":///")
    if not separator or not rest:
        return None
    return os.path.normpath(rest.split("?", 1)[0])


def ensure_sqlite_schema(db_path: str) -> bool:
    """Create the Chainlit tables if they are not already there. Never raises."""
    try:
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            for statement in SQLITE_SCHEMA:
                conn.execute(statement)
            _add_missing_columns(conn)
            conn.commit()
        return True
    except Exception:
        logger.warning("could not initialise the chat history database at %s", db_path, exc_info=True)
        return False


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Bring an existing database up to :data:`SQLITE_ADDED_COLUMNS`.

    ``CREATE TABLE IF NOT EXISTS`` does nothing for a database that already has
    the table, so a column added to the schema would only ever reach fresh
    installs - and the deployments that lose their transcripts are precisely the
    ones with history to lose. Additive and idempotent: no column is dropped or
    retyped, so an older build keeps working against the same file.
    """
    for table, column, declaration in SQLITE_ADDED_COLUMNS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column in existing:
            continue
        conn.execute(f'ALTER TABLE {table} ADD COLUMN "{column}" {declaration}')
        logger.info("added the %s.%s column to the chat history database", table, column)


def verify_sqlite_schema(db_path: str) -> list[str]:
    """Names of the required tables that are missing. Empty list means healthy.

    Worth checking explicitly because Chainlit's data layer logs and continues on
    a SQL error instead of raising: a database missing a table would look like a
    working app that quietly forgets every conversation.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        present = {row[0] for row in rows}
        return [name for name in REQUIRED_TABLES if name not in present]
    except Exception:
        logger.warning("could not inspect the chat history database at %s", db_path, exc_info=True)
        return list(REQUIRED_TABLES)


# --------------------------------------------------------------------------- #
# Auth secret
# --------------------------------------------------------------------------- #


def ensure_auth_secret(workspace_root: str | None = None) -> str:
    """Return the JWT secret Chainlit needs, generating a persistent one if unset.

    Chainlit refuses to start with any auth callback registered and no
    ``CHAINLIT_AUTH_SECRET``. Generating one on the fly keeps a deployment that
    has not set it from hard-failing, but the value must be *stable*: a fresh
    secret per process invalidates every browser session on restart, and differs
    between replicas. So it is written next to the thread database and reused.

    Falls back to a process-local secret when nothing is writable.
    """
    existing = os.getenv("CHAINLIT_AUTH_SECRET", "").strip()
    if existing:
        return existing

    directory = resolve_threads_dir(workspace_root)
    if directory:
        path = os.path.join(directory, AUTH_SECRET_FILENAME)
        try:
            os.makedirs(directory, exist_ok=True)
            if os.path.isfile(path):
                stored = open(path, encoding="utf-8").read().strip()
                if stored:
                    return stored
            generated = secrets.token_urlsafe(48)
            # 0600: this is a signing key. Written via a temp file + rename so a
            # second replica starting concurrently cannot read a half-written one.
            tmp = f"{path}.{os.getpid()}.tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(generated)
            os.replace(tmp, path)
            logger.info("generated a CHAINLIT_AUTH_SECRET at %s; set it explicitly for multi-replica setups", path)
            return generated
        except OSError:
            logger.warning("could not persist an auth secret under %s", directory, exc_info=True)

    logger.warning(
        "CHAINLIT_AUTH_SECRET is not set and no writable location was found; using a process-local secret. "
        "Browser sessions will not survive a restart."
    )
    return secrets.token_urlsafe(48)


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #


def max_inline_bytes() -> int:
    raw = os.getenv("BIOMNI_THREAD_ELEMENT_MAX_BYTES", "").strip()
    try:
        return max(0, int(raw)) if raw else _DEFAULT_MAX_INLINE_BYTES
    except ValueError:
        return _DEFAULT_MAX_INLINE_BYTES


def build_data_layer(workspace_root: str | None = None) -> Any | None:
    """Chainlit data layer for durable threads, or ``None`` when disabled.

    Imports Chainlit lazily so the module stays importable in a bare test
    environment. Returns ``None`` on any failure: losing chat history is bad,
    but taking the app down with it is worse.
    """
    url = resolve_database_url(workspace_root)
    if not url:
        logger.info("no writable location for chat history; conversations will not be listed across sessions")
        return None

    db_path = sqlite_path_from_url(url)
    if db_path:
        if not ensure_sqlite_schema(db_path):
            return None
        missing = verify_sqlite_schema(db_path)
        if missing:
            logger.error("chat history database at %s is missing tables %s; disabling it", db_path, missing)
            return None
    else:
        logger.info("using an operator-provided chat history database; its schema is assumed to exist")

    # SQLite serialises writers. Two browser tabs (or two users in single-user
    # mode) both mid-conversation is not an exotic case, and the default busy
    # timeout is 5 seconds - short enough that a burst of step writes can fail
    # with "database is locked". The data layer would swallow that into a log
    # line and drop the step, so give a blocked writer room to wait instead.
    connect_args = {"timeout": _SQLITE_BUSY_TIMEOUT_S} if db_path else None

    try:
        return _inline_element_data_layer_class()(
            conninfo=url,
            connect_args=connect_args,
            storage_provider=InlineBlobStorage(),
            # Only SQLite needs the tag workaround; Postgres has a real array type.
            drop_list_tags=bool(db_path),
        )
    except Exception:
        logger.exception("could not build the chat history data layer; conversations will not persist")
        return None


def element_payload_size(element: Any) -> int:
    """Bytes an element would add to the database, or 0 when it cannot be known."""
    path = getattr(element, "path", None)
    if path:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    content = getattr(element, "content", None)
    if isinstance(content, bytes):
        return len(content)
    if isinstance(content, str):
        return len(content.encode("utf-8", errors="ignore"))
    return 0


def _inline_element_data_layer_class() -> Any:
    """Build the data-layer subclass. Imports Chainlit, so it is called lazily."""
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

    class InlineElementDataLayer(SQLAlchemyDataLayer):
        """Chainlit's SQL data layer, adapted for a SQLite-backed deployment.

        Two adjustments, both found by running the thing:

        * A size guard on inlined attachments. Base64 in a row is fine for a
          plot and ruinous for a 400 MB parquet file, and the base class has no
          notion of a cap - it reads the whole payload into memory and hands it
          straight to the storage provider, so the check has to happen first.
        * Dropping list-valued ``tags``. Chainlit tags every thread with the
          chat profile name when ``auto_tag_thread`` is on (it is, by default,
          and this app has profiles). The base layer binds that Python list
          directly, which SQLite cannot accept - and because ``execute_sql``
          turns every error into a log line, the whole statement is discarded
          in silence. That statement is the one carrying ``name`` and
          ``userId``, so the visible symptom is not a missing tag: it is a
          permanently empty conversation list, because every thread is written
          with no owner and no title.
        """

        def __init__(self, *args: Any, drop_list_tags: bool = False, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._drop_list_tags = drop_list_tags

        async def update_thread(
            self,
            thread_id: str,
            name: str | None = None,
            user_id: str | None = None,
            metadata: dict | None = None,
            tags: list[str] | None = None,
        ) -> None:
            if self._drop_list_tags and isinstance(tags, list):
                tags = None
            return await super().update_thread(thread_id, name=name, user_id=user_id, metadata=metadata, tags=tags)

        async def create_element(self, element: Any) -> None:
            limit = max_inline_bytes()
            size = element_payload_size(element)
            if limit and size > limit:
                logger.info(
                    "not archiving element %r (%d bytes > %d limit); it remains in the run output directory",
                    getattr(element, "name", "?"),
                    size,
                    limit,
                )
                return None
            return await super().create_element(element)

    return InlineElementDataLayer


class InlineBlobStorage:
    """Stores element payloads as ``data:`` URLs inside the thread database.

    Chainlit's SQL data layer keeps only element *metadata* and delegates the
    bytes to a blob store - S3, Azure, GCS. Requiring one of those to reread an
    old conversation would be an absurd amount of infrastructure for what a
    Biomni thread actually attaches: a few plots and text files. Embedding small
    payloads directly means a resumed thread renders exactly as it did live, with
    no extra service, no signed URLs and no second thing to back up.

    The cost is database size, which is why anything over
    ``BIOMNI_THREAD_ELEMENT_MAX_BYTES`` is skipped by
    :class:`InlineElementDataLayer` before it ever reaches here. Skipped
    attachments are not lost - they remain in the run's output directory.
    """

    async def upload_file(
        self,
        object_key: str,
        data: bytes | str,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ) -> dict[str, Any]:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        encoded = base64.b64encode(raw).decode("ascii")
        # No object key on purpose. The data layer treats one as "this payload
        # lives somewhere addressable" and, when reopening a thread, throws away
        # the stored URL to ask get_read_url for a fresh one - which for an
        # inline payload cannot be produced from a key alone. Every image in a
        # resumed conversation came back as a broken thumbnail because of it.
        # Returning only the URL keeps the data layer on the column that holds
        # the actual bytes.
        return {"url": f"data:{mime};base64,{encoded}"}

    async def delete_file(self, object_key: str) -> bool:
        return True  # the payload lives in the row; deleting the row deletes it

    async def get_read_url(self, object_key: str) -> str:
        """Unreachable for anything this class wrote; see :meth:`upload_file`.

        Rows written before that changed still carry an object key, and the
        data layer will ask for a URL for them. Raising is the honest answer -
        there is no such location - and it makes the caller fall back to the
        row's own ``url``, which is the payload. Those conversations repair
        themselves on the next read.
        """
        raise NotImplementedError("inline element payloads have no addressable location")

    async def close(self) -> None:
        return None
