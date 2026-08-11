"""Durable record of a user's agent runs, so closing the browser loses nothing.

A Biomni query can run for many minutes. Today the only trace of one lives in
the websocket session and in a container-local ``runs/`` directory: close the
tab and the user has no way to find out whether the job finished, whether it
failed, or where its results went. If the pod restarts mid-run, even the server
forgets.

This module keeps a small JSON record per run, written at every state change,
so that on the next session the user can be shown:

* runs that finished while they were away, and where the outputs are;
* runs that were still going when the process died, marked honestly as
  ``interrupted`` rather than left claiming to be running forever.

**Scope, stated plainly.** This persists the *record and the results* of a run.
It does not make execution survive a pod restart - the agent loop runs in the
serving process, so a restart kills the work itself. Detaching execution into a
worker that outlives the session is a larger change (a task queue and a place to
stream partial output to); this registry is the state layer such a worker would
need, and it delivers the user-visible half of the requirement now: no job
silently disappears.

Staleness is decided by heartbeat, not by status. A record saying ``running``
proves only that *something* wrote it; a heartbeat that stopped advancing proves
the writer is gone. :meth:`RunRegistry.reconcile` applies that rule on load.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from biomni.workspace_prefs import (
    WORKSPACE_STATE_DIRNAME,
    atomic_write_json,
    is_writable_dir,
    read_json,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

RUN_RECORD_VERSION = 1

# Terminal states: a run in one of these is finished and never reconciled.
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})

# A run whose heartbeat is older than this is assumed dead. Comfortably above
# the run heartbeat cadence (BIOMNI_RUN_HEARTBEAT_SECONDS, default 15s) so a
# merely slow run is never declared interrupted.
_DEFAULT_STALE_AFTER_S = 180.0

# Prompts are echoed back to the user as a run label; keep records small and
# avoid turning the registry into a second transcript store.
_PROMPT_LABEL_MAX = 300


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using default %s", name, raw, default)
        return default
    return value if value > 0 else default


def stale_after_s() -> float:
    """Heartbeat age past which a non-terminal run is considered interrupted."""
    return _float_env("BIOMNI_RUN_STALE_AFTER_S", _DEFAULT_STALE_AFTER_S)


@dataclass
class RunRecord:
    """One agent run, as it will be shown to the user on their next visit."""

    run_id: str
    status: str = "running"  # running | completed | failed | cancelled | interrupted
    prompt: str = ""
    agent_type: str | None = None
    output_dir: str | None = None
    session_id: str | None = None
    workspace_id: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    heartbeat_at: str = field(default_factory=_utc_now_iso)
    finished_at: str | None = None
    error: str | None = None
    version: int = RUN_RECORD_VERSION

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def label(self) -> str:
        """Short single-line description for a list entry."""
        text = " ".join(self.prompt.split())
        if len(text) > 80:
            text = text[:77].rstrip() + "..."
        return text or self.run_id

    def age_seconds(self, *, now: datetime | None = None) -> float | None:
        """Seconds since the last heartbeat, or ``None`` if unparseable."""
        beat = _parse_iso(self.heartbeat_at)
        if beat is None:
            return None
        return ((now or datetime.now(UTC)) - beat).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> RunRecord | None:
        """Rebuild from stored JSON, or ``None`` when the payload is unusable."""
        if not isinstance(data, dict):
            return None
        run_id = data.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return None
        version = data.get("version")
        if not isinstance(version, int) or version > RUN_RECORD_VERSION:
            logger.warning("ignoring run record %s with unsupported version %r", run_id, version)
            return None

        def _str_or_none(key: str) -> str | None:
            value = data.get(key)
            return value if isinstance(value, str) and value else None

        status = data.get("status")
        return cls(
            run_id=run_id,
            status=status if isinstance(status, str) and status else "interrupted",
            prompt=data.get("prompt") if isinstance(data.get("prompt"), str) else "",
            agent_type=_str_or_none("agent_type"),
            output_dir=_str_or_none("output_dir"),
            session_id=_str_or_none("session_id"),
            workspace_id=_str_or_none("workspace_id"),
            created_at=_str_or_none("created_at") or _utc_now_iso(),
            updated_at=_str_or_none("updated_at") or _utc_now_iso(),
            heartbeat_at=_str_or_none("heartbeat_at") or _utc_now_iso(),
            finished_at=_str_or_none("finished_at"),
            error=_str_or_none("error"),
            version=RUN_RECORD_VERSION,
        )


class RunRegistry:
    """Per-user run records stored as one JSON file each under ``root``.

    ``enabled`` is False when no writable location exists; every method then
    becomes a no-op returning empty results, so callers need no branching and a
    deployment without a volume behaves exactly as it does today.
    """

    def __init__(self, root: str | None) -> None:
        self.root = os.path.abspath(root) if root else None

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _user_dir(self, user_key: str) -> str | None:
        if not self.root:
            return None
        return os.path.join(self.root, os.path.basename(user_key))

    def _path(self, user_key: str, run_id: str) -> str | None:
        user_dir = self._user_dir(user_key)
        if user_dir is None:
            return None
        return os.path.join(user_dir, f"{os.path.basename(run_id)}.json")

    # -- writes ------------------------------------------------------------ #

    def start(
        self,
        user_key: str,
        run_id: str,
        *,
        prompt: str = "",
        agent_type: str | None = None,
        output_dir: str | None = None,
        session_id: str | None = None,
        workspace_id: str | None = None,
    ) -> RunRecord:
        """Record a run as started. Returns the record even when disabled."""
        record = RunRecord(
            run_id=run_id,
            status="running",
            prompt=" ".join(prompt.split())[:_PROMPT_LABEL_MAX],
            agent_type=agent_type,
            output_dir=output_dir,
            session_id=session_id,
            workspace_id=workspace_id,
        )
        self._write(user_key, record)
        return record

    def heartbeat(self, user_key: str, record: RunRecord) -> None:
        """Advance the liveness stamp of a still-running record."""
        if record.is_terminal:
            return
        record.heartbeat_at = _utc_now_iso()
        record.updated_at = record.heartbeat_at
        self._write(user_key, record)

    def finish(
        self,
        user_key: str,
        record: RunRecord,
        *,
        status: str = "completed",
        error: str | None = None,
        output_dir: str | None = None,
    ) -> RunRecord:
        """Close a record out in a terminal state."""
        record.status = status
        record.error = error
        if output_dir:
            record.output_dir = output_dir
        now = _utc_now_iso()
        record.updated_at = now
        record.heartbeat_at = now
        record.finished_at = now
        self._write(user_key, record)
        return record

    def _write(self, user_key: str, record: RunRecord) -> bool:
        path = self._path(user_key, record.run_id)
        if path is None:
            return False
        return atomic_write_json(path, record.to_dict())

    # -- reads ------------------------------------------------------------- #

    def list_for_user(self, user_key: str, *, limit: int = 20) -> list[RunRecord]:
        """Most recent runs first. Empty when disabled or the user is new."""
        user_dir = self._user_dir(user_key)
        if user_dir is None or not os.path.isdir(user_dir):
            return []
        try:
            names = [n for n in os.listdir(user_dir) if n.endswith(".json") and not n.startswith(".")]
        except OSError:
            logger.warning("could not list run records in %s", user_dir, exc_info=True)
            return []

        records: list[RunRecord] = []
        for name in names:
            record = RunRecord.from_dict(read_json(os.path.join(user_dir, name)))
            if record is not None:
                records.append(record)

        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]

    def reconcile(self, user_key: str, *, limit: int = 20) -> list[RunRecord]:
        """List runs, first marking abandoned ones as ``interrupted``.

        A record still claiming to run while its heartbeat has gone quiet means
        the process that owned it died. Rewriting it here (rather than only
        displaying it differently) means the correction is durable and the user
        sees the same truth on every subsequent visit.
        """
        threshold = stale_after_s()
        records = self.list_for_user(user_key, limit=limit)
        now = datetime.now(UTC)

        for record in records:
            if record.is_terminal:
                continue
            age = record.age_seconds(now=now)
            if age is None or age > threshold:
                record.status = "interrupted"
                record.error = "the application stopped while this run was in progress"
                record.finished_at = _utc_now_iso()
                record.updated_at = record.finished_at
                self._write(user_key, record)
                logger.info("marked run %s as interrupted (heartbeat age %s)", record.run_id, age)

        return records

    def unfinished(self, records: Iterable[RunRecord]) -> list[RunRecord]:
        """Records that ended without completing - worth surfacing to the user."""
        return [r for r in records if r.status in {"interrupted", "failed"}]


def build_run_registry(workspace_root: str | None = None) -> RunRegistry:
    """Pick a storage location for run records, mirroring the preferences store.

    Order: ``BIOMNI_RUNS_STATE_DIR`` → ``BIOMNI_STATE_DIR/runs`` →
    ``<workspace>/.biomni/runs`` → disabled.
    """
    explicit = os.getenv("BIOMNI_RUNS_STATE_DIR", "").strip()
    if explicit and is_writable_dir(explicit):
        return RunRegistry(explicit)

    state_dir = os.getenv("BIOMNI_STATE_DIR", "").strip()
    if state_dir:
        candidate = os.path.join(state_dir, "runs")
        if is_writable_dir(candidate):
            return RunRegistry(candidate)

    if workspace_root:
        candidate = os.path.join(workspace_root, WORKSPACE_STATE_DIRNAME, "runs")
        if is_writable_dir(candidate):
            return RunRegistry(candidate)

    logger.info("no writable location for run records; runs will not be listed across sessions")
    return RunRegistry(None)
