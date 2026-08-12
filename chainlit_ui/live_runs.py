"""Keep a run attached to its conversation, not to the browser tab that began it.

A Biomni query runs for minutes. The user closes the laptop, comes back, clicks
the conversation in the list on the left - and the work should still be there,
still going.

Two halves make that true, and Chainlit gives us one of them for free:

* **The work continues.** Chainlit does not cancel the message task when a
  websocket drops (see ``socket.disconnect``), and every step the agent emits is
  written to the data layer as it happens. So a detached run keeps running and
  keeps filling in its own thread.
* **The reopened tab follows it.** That half is missing. Reopening a thread
  builds a *new* session; the in-flight run is still emitting to the socket of
  the old one, which nobody is listening to, so the conversation renders
  whatever was persisted before the reload and then sits there looking finished
  while the agent is still working.

This module closes that gap. A run registers the session that owns it for the
duration; when the same conversation is reopened, :meth:`LiveRuns.reattach`
points the running session's emitter at the new connection, and its remaining
output streams into the page the user is actually looking at. That redirection
is exactly what Chainlit itself does on a websocket reconnect
(``restore_existing_session``), which is why two plain attribute assignments are
enough - ``ChainlitEmitter.emit`` reads ``session.emit`` on every call.

Sessions are held by duck type (anything with ``emit``/``emit_call``), so this
module stays importable without Chainlit and the bookkeeping stays testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LiveRun:
    """An executing conversation: where its output goes, and what to cancel."""

    session: Any
    task: Any = None


class LiveRuns:
    """Which conversation is currently executing, and in whose session.

    In-memory and single-process on purpose: it answers "is this tab's
    conversation still being worked on *right now*", and a run cannot outlive
    the process that is running it. Durable run state lives in
    :mod:`biomni.run_registry`.
    """

    def __init__(self) -> None:
        self._runs: dict[str, LiveRun] = {}

    def register(self, thread_id: str | None, session: Any, task: Any = None) -> None:
        """Mark ``thread_id`` as executing in ``session``.

        ``task`` is the asyncio task doing the work. It is kept so that Stop
        remains honest from a tab that did not start the run: the session that
        owns a detached run is not the one the click arrives on, and cancelling
        the clicking session's task would stop nothing at all.
        """
        if not thread_id or session is None:
            return
        self._runs[thread_id] = LiveRun(session=session, task=task)

    def release(self, thread_id: str | None, session: Any) -> None:
        """Forget ``thread_id``, unless another session has since claimed it.

        The identity check matters: a user who reopens a finished conversation
        and asks a second question has a *new* session registered under the same
        thread id, and the first run's cleanup must not evict it.
        """
        if not thread_id:
            return
        live = self._runs.get(thread_id)
        if live is not None and live.session is session:
            del self._runs[thread_id]

    def is_live(self, thread_id: str | None) -> bool:
        return bool(thread_id) and thread_id in self._runs

    def reattach(self, thread_id: str | None, session: Any) -> bool:
        """Stream a still-running conversation into ``session``'s connection.

        Returns True when a run was redirected, so the caller can tell the user
        the page is now following live work. False means there is nothing in
        flight for this conversation, which is the ordinary case.
        """
        if not thread_id or session is None:
            return False
        live = self._runs.get(thread_id)
        if live is None or live.session is session:
            return False
        try:
            live.session.emit = session.emit
            live.session.emit_call = session.emit_call
        except Exception:
            logger.warning("could not reattach the live run for thread %s", thread_id, exc_info=True)
            return False
        logger.info("reattached a live run to the reopened conversation %s", thread_id)
        return True

    def cancel(self, thread_id: str | None) -> bool:
        """Stop the run executing in ``thread_id``. True if one was cancelled.

        Chainlit's Stop cancels the *clicking* session's task, which for a
        reopened conversation is not the task doing the work. This cancels the
        run itself, and its own cleanup then closes the record out and tells the
        page the task ended.
        """
        if not thread_id:
            return False
        live = self._runs.get(thread_id)
        task = live.task if live else None
        if task is None or task.done():
            return False
        task.cancel()
        logger.info("cancelled the run for conversation %s", thread_id)
        return True

    def clear(self) -> None:
        """Drop all bookkeeping. Tests only."""
        self._runs.clear()


# One per process; the session objects it holds are process-local anyway.
LIVE_RUNS = LiveRuns()
