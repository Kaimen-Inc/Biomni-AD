"""Feed the numbers Chainlit owns into the ``/status`` payload.

:mod:`biomni.status` deliberately knows nothing about Chainlit: it is part of
the installed package, it has to import in a bare test environment, and the
counts it keeps itself (runs in flight) are ones this app actually controls.
Two numbers it cannot know are owned by Chainlit instead - how many websocket
sessions are open, and how many database connections the chat-history data layer
is holding - so they are registered here as gauges, read only when the
monitoring framework asks.

Every gauge fails soft. A gauge that raises is dropped from the payload by the
tracker, and a gauge that cannot be wired at all simply never registers, which
is the documented "field does not apply to this application" case.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from biomni.status import ActivityTracker

logger = logging.getLogger(__name__)


def count_open_sessions() -> int:
    """Websocket sessions Chainlit is currently holding.

    Chainlit's own registry is the source of truth rather than a counter this
    app increments, and the reason is its reconnect path: a websocket that drops
    and comes back restores the existing session *without* running the chat-start
    handler again, so a hand-maintained counter would decrement on every blip and
    never recover - drifting down until it reported an empty pod full of users.

    A session stays in the registry for Chainlit's ``session_timeout`` (an hour,
    by default) after a disconnect so the user can resume it. Counting it for
    that hour is the honest answer rather than a generous one: the session still
    holds its agent and workspace state in this process, which is exactly the
    internal activity the monitoring framework is asking about.
    """
    from chainlit.session import ws_sessions_id

    return len(ws_sessions_id)


def count_database_connections(data_layer: Any) -> int | None:
    """Connections currently open in the data layer's pool, if it has one.

    Counts both checked-out and idle-but-open connections: the question the
    monitoring framework is asking is how much of the database this process is
    holding, not how much of it is busy this instant.

    Returns ``None`` for a data layer with no measurable pool (an in-memory
    SQLite ``StaticPool``, or a future non-SQLAlchemy layer), which keeps the
    field out of the payload rather than reporting a made-up zero.
    """
    engine = getattr(data_layer, "engine", None)
    pool = getattr(engine, "pool", None)
    if pool is None:
        return None
    checked_out = getattr(pool, "checkedout", None)
    checked_in = getattr(pool, "checkedin", None)
    if not callable(checked_out) or not callable(checked_in):
        return None
    return int(checked_out()) + int(checked_in())


def register_session_gauge(tracker: ActivityTracker) -> None:
    """Report open websocket sessions in ``/status``."""
    try:
        count_open_sessions()  # fail here, at wiring time, not per request
    except Exception:
        logger.debug("open session count is unavailable; omitting it from /status", exc_info=True)
        return
    tracker.register_gauge("open_sessions", count_open_sessions)


def register_database_gauge(tracker: ActivityTracker, data_layer: Any) -> None:
    """Report the chat-history connection count in ``/status``.

    Called with the data layer once it exists, because there may not be one:
    chat history is off unless the deployment has a stable identity to key it by.
    """
    if data_layer is None or count_database_connections(data_layer) is None:
        return
    tracker.register_gauge("database_connections", lambda: count_database_connections(data_layer))
