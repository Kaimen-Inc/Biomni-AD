"""Tests for chainlit_ui.graph_stream - a LangGraph run read from the event loop.

The graph is a stand-in with LangGraph's ``stream`` signature, so these run
without LangGraph and exercise exactly what crosses the thread boundary: states,
a failure, the order to stop, and the caller's context.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading

import pytest
from biomni.identity import UserIdentity, bound_identity, current_identity
from chainlit_ui.graph_stream import stream_langgraph


class _Graph:
    """Yields the given states, then raises ``error`` if one is given."""

    def __init__(self, states, error: Exception | None = None, gate: threading.Event | None = None) -> None:
        self.states = states
        self.error = error
        self.gate = gate
        self.produced = 0
        self.seen_identity: list[object] = []
        self.finished = threading.Event()

    def stream(self, inputs, stream_mode, config):
        assert stream_mode == "values"
        try:
            for state in self.states:
                self.seen_identity.append(current_identity())
                self.produced += 1
                yield state
                if self.gate is not None:
                    self.gate.wait(timeout=5)
            if self.error is not None:
                raise self.error
        finally:
            self.finished.set()


async def _collect(graph) -> list:
    return [state async for state in stream_langgraph(graph, {"messages": []}, {})]


def test_states_arrive_in_order():
    assert asyncio.run(_collect(_Graph([1, 2, 3]))) == [1, 2, 3]


def test_a_failed_run_raises_in_the_reader_instead_of_ending_quietly():
    """It used to end like a finished run, and the user was told "Run complete"."""
    graph = _Graph([1], error=PermissionError("refused by the LLM proxy"))
    received: list = []

    async def read():
        async for state in stream_langgraph(graph, {}, {}):
            received.append(state)

    with pytest.raises(PermissionError, match="refused by the LLM proxy"):
        asyncio.run(read())
    assert received == [1]


def test_a_run_that_fails_before_its_first_state_still_raises():
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(_collect(_Graph([], error=RuntimeError("boom"))))


def test_the_graph_runs_as_the_session_that_started_it():
    who = UserIdentity(user_id="u1", workspace_id="ws-1", source="headers")
    graph = _Graph([1, 2])

    async def read():
        with bound_identity(who):
            return await _collect(graph)

    asyncio.run(read())
    assert graph.seen_identity == [who, who]


def test_stopping_the_reader_stops_the_graph_after_its_current_step():
    """Stop cancels the reading task; the thread running the graph must follow.

    It used to keep calling the model for as long as the whole plan took.
    """
    gate = threading.Event()
    graph = _Graph(list(range(100)), gate=gate)

    async def main():
        first = asyncio.Event()

        async def read():
            async for _state in stream_langgraph(graph, {}, {}):
                first.set()

        task = asyncio.create_task(read())
        await first.wait()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        gate.set()  # let the node in progress finish
        assert await asyncio.to_thread(graph.finished.wait, 5)

    asyncio.run(main())
    # The node that was running when Stop came in, and not one more.
    assert graph.produced == 2


class _SharedListGraph:
    """Appends to one messages list in place and yields it, as A1's nodes do."""

    def stream(self, inputs, stream_mode, config):
        state = {"messages": [], "next_step": None}
        for i in range(5):
            state["messages"].append(f"message {i}")
            yield state


def test_each_state_arrives_as_it_was_when_streamed():
    """A slow reader must not see messages from nodes that ran after the state it is reading.

    It did: a code step went missing from the transcript and the observation
    after it was shown twice.
    """

    async def read_slowly():
        seen = []
        async for state in stream_langgraph(_SharedListGraph(), {}, {}):
            await asyncio.sleep(0.02)  # the graph runs ahead meanwhile
            seen.append(state["messages"][-1])
        return seen

    assert asyncio.run(read_slowly()) == [f"message {i}" for i in range(5)]
