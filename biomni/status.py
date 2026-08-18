"""Application status for the platform's monitoring framework (``GET /status``).

The framework polls every application for one primary fact: is it *doing*
anything? An application that is genuinely idle may be reclaimed; one that is
working must not be. Everything in this module exists to answer that honestly.

**What counts as activity.** A user action - opening or resuming a chat, sending
a message, changing settings, uploading a file - and any agent run in flight.
Deliberately *not* counted: the monitoring poll itself and the Kubernetes
liveness/readiness probes. Those arrive every few seconds forever, so counting
them would pin ``active`` to ``true`` for the life of the pod and tell the
framework nothing at all.

**Why idleness is tolerated.** A researcher reads a result, thinks, then asks
the next question; in between, the process does nothing. Reporting inactive in
that gap would be true and useless. So the app stays active for a grace period
after the last activity - ``BIOMNI_STATUS_INACTIVITY_SECONDS``, default 4 hours,
which is the platform's own suggestion and comfortably longer than a working
session's think time - and reports inactive only when nothing has run and nobody
has touched it for that long.

An idle *open* session is deliberately not enough on its own. Browser tabs get
left open for days, and honouring one would make a pod that has done nothing
since Tuesday claim to be busy. A tab someone is actually using generates
activity by definition, which the window above already covers.

**The elapsed-time check uses a monotonic clock**, so a wall-clock correction
(NTP step, container clock skew) cannot make a busy app look idle for hours.
``last_activity`` is still reported as an epoch timestamp, which is what the
framework asked for.

**Counts are process-local**, because that is exactly what the endpoint means:
one pod, describing itself. With several replicas the framework sees several
statuses and the application is busy if any of them says so.

The payload carries no user data - counts, a timestamp and a schema version - so
it is safe to serve to a monitoring system that has not authenticated as anyone.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from biomni.health import register_routes_first

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from starlette.requests import Request

logger = logging.getLogger(__name__)

# Version of the payload shape below. Bump only for a breaking change, so the
# monitoring framework can tell an old pod from a new one during a rollout.
SCHEMA_VERSION = 1

# How long the app keeps reporting active after the last activity. Long enough
# to span the pauses in a working session; short enough that a genuinely
# abandoned pod is visible within a working day.
DEFAULT_INACTIVITY_SECONDS = 4 * 60 * 60


def inactivity_window_seconds() -> float:
    """The configured grace period, in seconds (``BIOMNI_STATUS_INACTIVITY_SECONDS``).

    ``0`` is a legitimate setting: it means "active only while something is
    actually running". An unparseable value falls back to the default rather
    than raising - a typo in a manifest must not break the status endpoint.
    """
    raw = os.getenv("BIOMNI_STATUS_INACTIVITY_SECONDS", "").strip()
    if not raw:
        return float(DEFAULT_INACTIVITY_SECONDS)
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "ignoring invalid BIOMNI_STATUS_INACTIVITY_SECONDS=%r; using %ss",
            raw,
            DEFAULT_INACTIVITY_SECONDS,
        )
        return float(DEFAULT_INACTIVITY_SECONDS)
    if value < 0:
        logger.warning(
            "ignoring negative BIOMNI_STATUS_INACTIVITY_SECONDS=%r; using %ss", raw, DEFAULT_INACTIVITY_SECONDS
        )
        return float(DEFAULT_INACTIVITY_SECONDS)
    return value


def _coerce_count(value: Any) -> int | None:
    """A gauge reading as a non-negative ``int``, or ``None`` if it is not one."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    count = int(value)
    return count if count >= 0 else None


class ActivityTracker:
    """Process-wide record of what this pod is doing, and when it last did it.

    Thread-safe because the callers are not all on the event loop: agent work
    runs in an executor, and the status endpoint is served by whichever worker
    picks up the request.

    Two kinds of number live here:

    * **Counters** the app maintains itself (runs in flight, sessions opened and
      closed). They are authoritative because we own the events that move them.
    * **Gauges** registered by whatever component actually owns the number
      (see :meth:`register_gauge`). A gauge is read at request time and a broken
      one is simply omitted, so the component that owns it stays free to be
      absent - Chainlit is not importable in a bare test environment, and chat
      history can be switched off entirely.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._monotonic = monotonic
        self._lock = threading.Lock()
        # Startup counts as activity: a pod that was just rolled out has not
        # been idle, it has not been *asked* anything yet, and reporting it
        # inactive would invite reclaiming it before the first user arrives.
        self._last_activity_epoch = clock()
        self._last_activity_monotonic = monotonic()
        self._background_jobs = 0
        self._open_sessions = 0
        self._gauges: dict[str, Callable[[], Any]] = {}

    # -- recording ------------------------------------------------------- #

    def record_activity(self, kind: str = "") -> None:
        """Note that something happened. ``kind`` is for the debug log only."""
        with self._lock:
            self._last_activity_epoch = self._clock()
            self._last_activity_monotonic = self._monotonic()
        if kind:
            logger.debug("activity recorded: %s", kind)

    def session_opened(self, kind: str = "session") -> None:
        with self._lock:
            self._open_sessions += 1
        self.record_activity(f"{kind}_opened")

    def session_closed(self, kind: str = "session") -> None:
        # Clamped at zero: a close without a matching open (a reload racing a
        # disconnect) must not drive the count negative and stay wrong forever.
        with self._lock:
            self._open_sessions = max(0, self._open_sessions - 1)
        self.record_activity(f"{kind}_closed")

    def job_started(self, kind: str = "run") -> None:
        with self._lock:
            self._background_jobs += 1
        self.record_activity(f"{kind}_started")

    def job_finished(self, kind: str = "run") -> None:
        with self._lock:
            self._background_jobs = max(0, self._background_jobs - 1)
        # Recorded on the way out as well, so the idle window starts when the
        # work ended rather than when it began.
        self.record_activity(f"{kind}_finished")

    @contextmanager
    def track_job(self, kind: str = "run") -> Iterator[None]:
        """Count one job for the duration of the block.

        ``finally`` matters more than it looks: an agent run is routinely
        cancelled (the Stop button) or killed by the run timeout, and a job that
        failed to decrement would leave the pod claiming to be busy forever.
        """
        self.job_started(kind)
        try:
            yield
        finally:
            self.job_finished(kind)

    # -- gauges ---------------------------------------------------------- #

    def register_gauge(self, name: str, provider: Callable[[], Any]) -> None:
        """Have ``provider`` answer for ``name`` when the status is served.

        Recognised names are ``open_sessions`` (overrides the internal counter,
        which cannot see a websocket that dropped without a clean close) and
        ``database_connections``.
        """
        with self._lock:
            self._gauges[name] = provider

    def _read_gauge(self, name: str) -> int | None:
        with self._lock:
            provider = self._gauges.get(name)
        if provider is None:
            return None
        try:
            return _coerce_count(provider())
        except Exception:
            # Monitoring must never be able to take the app down, and a missing
            # field is a documented outcome for it.
            logger.debug("status gauge %r failed; omitting it", name, exc_info=True)
            return None

    # -- reporting ------------------------------------------------------- #

    def snapshot(self) -> dict[str, Any]:
        """The ``GET /status`` payload."""
        window = inactivity_window_seconds()
        with self._lock:
            jobs = self._background_jobs
            sessions = self._open_sessions
            last_activity = self._last_activity_epoch
            idle_seconds = self._monotonic() - self._last_activity_monotonic

        reported_sessions = self._read_gauge("open_sessions")
        if reported_sessions is not None:
            sessions = reported_sessions

        payload: dict[str, Any] = {
            "active": jobs > 0 or idle_seconds <= window,
            "last_activity": int(last_activity),
            "schema_version": SCHEMA_VERSION,
            "background_jobs": jobs,
            "open_sessions": sessions,
        }

        # Only fields we can actually measure. ``active_threads`` always applies;
        # a database connection count exists only where chat history is enabled.
        internal: dict[str, Any] = {"active_threads": threading.active_count()}
        connections = self._read_gauge("database_connections")
        if connections is not None:
            internal["database_connections"] = connections
        payload["internal_processes"] = internal

        return payload

    def reset(self) -> None:
        """Drop all state. Tests only."""
        with self._lock:
            self._last_activity_epoch = self._clock()
            self._last_activity_monotonic = self._monotonic()
            self._background_jobs = 0
            self._open_sessions = 0
            self._gauges.clear()


# One per process, which is the scope the endpoint describes.
ACTIVITY = ActivityTracker()


def status_payload() -> dict[str, Any]:
    """Current status of this process, in the monitoring framework's shape."""
    return ACTIVITY.snapshot()


def register_status_route(app: Any, *, path: str = "/status") -> None:
    """Register ``GET /status`` at the front of ``app``'s router.

    Ahead of everything, for the same reason the probes are: Chainlit's
    catch-all would otherwise answer with the single-page app.
    """
    from starlette.responses import JSONResponse

    async def status(_request: Request) -> JSONResponse:
        # The poll itself is not activity - see the module docstring.
        return JSONResponse(status_payload(), status_code=200)

    register_routes_first(app, [(path, status)])


__all__ = [
    "ACTIVITY",
    "DEFAULT_INACTIVITY_SECONDS",
    "SCHEMA_VERSION",
    "ActivityTracker",
    "inactivity_window_seconds",
    "register_status_route",
    "status_payload",
]
