"""Reading a LangGraph run from the event loop while it executes in a thread.

The agent's graph is synchronous and runs for minutes, so it gets a worker
thread of its own and hands each state back to the event loop through a queue.
Two things have to cross that boundary besides the states, and both used to be
lost on the way: the order to stop, and the exception a failed run raised. The
states themselves have to cross as they were, not as they later became.

No Chainlit import, so the stopping and failure behaviour is testable without a
running app (CI installs no chainlit extra).
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from biomni.observability import capture_context

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _snapshot(state: Any) -> Any:
    """The state as it is at this moment.

    The agent's nodes append to one shared ``messages`` list and return it, so
    every state the graph streams is the same list. Read when the event loop got
    round to it, a state already held the next node's messages: a code step went
    missing from the transcript and its observation was shown twice. A shallow
    copy is enough, since messages are appended, never changed.
    """
    if isinstance(state, dict) and isinstance(state.get("messages"), list):
        return {**state, "messages": list(state["messages"])}
    return state


@dataclass(frozen=True)
class _GraphFailed:
    """What the graph raised, on its way from the worker thread to the reader."""

    error: Exception


async def stream_langgraph(agent_app: Any, inputs: Any, config: Any) -> AsyncIterator[Any]:
    """Yield LangGraph state dicts asynchronously from a sync stream.

    The graph runs in a worker thread, which cancellation cannot reach: an
    asyncio task that is cancelled stops *reading*, and a thread that nobody
    reads goes right on calling the model. Stop therefore used to end the
    conversation while the run it stopped kept working - for as long as the
    whole plan took. The flag below is the cooperative half of the stop: the
    producer checks it between graph nodes, so the run ends after the step it
    was already in.

    An exception the graph raises is re-raised here, in the reader.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stop = threading.Event()

    def _producer():
        try:
            for state in agent_app.stream(inputs, stream_mode="values", config=config):
                asyncio.run_coroutine_threadsafe(queue.put(_snapshot(state)), loop)
                if stop.is_set():
                    break
        except Exception as exc:
            # Handed to the reader, not left on the executor's future where no
            # one would ever look: a graph that died - a model call refused, a
            # node that raised - used to read as a graph that finished, and the
            # user was told "Run complete" over a run that never produced a word.
            asyncio.run_coroutine_threadsafe(queue.put(_GraphFailed(exc)), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

    executor = ThreadPoolExecutor(max_workers=1)
    # Capture the active context here (caller thread) and replay it in the
    # producer thread, so the graph's logs carry the session_id/run_id and its
    # model calls the session's identity (the LLM proxy bills each call to it).
    ctx = capture_context()
    executor.submit(ctx.run, _producer)

    try:
        while True:
            state = await queue.get()
            if state is None:
                break
            if isinstance(state, _GraphFailed):
                raise state.error
            yield state
    finally:
        stop.set()
        # Not waited on: the worker may be mid-node, and holding the event loop
        # for it would freeze the UI. It exits at the next check.
        executor.shutdown(wait=False)


__all__ = ["stream_langgraph"]
