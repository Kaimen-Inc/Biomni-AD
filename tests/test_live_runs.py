"""Tests for chainlit_ui.live_runs - a run outliving the tab that started it.

The bookkeeping is small but the failure modes are not: a conversation left
registered makes the next tab wait for work that is not running, and an eager
release hands a live run's output to nobody.
"""

from __future__ import annotations

import pytest
from chainlit_ui.live_runs import LiveRuns


class FakeSession:
    """A stand-in for chainlit's WebsocketSession - only emit matters here."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.emit = f"emit-{name}"
        self.emit_call = f"emit_call-{name}"


@pytest.fixture
def runs() -> LiveRuns:
    return LiveRuns()


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_a_registered_conversation_is_live(runs: LiveRuns) -> None:
    runs.register("t1", FakeSession("a"))
    assert runs.is_live("t1")
    assert not runs.is_live("t2")


def test_release_clears_the_conversation(runs: LiveRuns) -> None:
    session = FakeSession("a")
    runs.register("t1", session)
    runs.release("t1", session)
    assert not runs.is_live("t1")


def test_release_from_a_superseded_session_keeps_the_newer_run(runs: LiveRuns) -> None:
    """The ordering that breaks a naive dict.

    The user reopens a finished conversation and asks a second question, so a
    new session registers under the same thread id. The first run's cleanup
    must not evict it, or the second run becomes unfollowable.
    """
    first, second = FakeSession("first"), FakeSession("second")
    runs.register("t1", first)
    runs.register("t1", second)

    runs.release("t1", first)

    assert runs.is_live("t1")
    runs.release("t1", second)
    assert not runs.is_live("t1")


def test_missing_ids_and_sessions_are_ignored(runs: LiveRuns) -> None:
    runs.register(None, FakeSession("a"))
    runs.register("", FakeSession("a"))
    runs.register("t1", None)
    runs.release(None, FakeSession("a"))
    assert not runs.is_live(None)
    assert not runs.is_live("t1")


# --------------------------------------------------------------------------- #
# Reattaching
# --------------------------------------------------------------------------- #


def test_reattach_redirects_the_running_session_to_the_new_connection(runs: LiveRuns) -> None:
    """The whole point: output goes to the tab the user is looking at."""
    running, reopened = FakeSession("running"), FakeSession("reopened")
    runs.register("t1", running)

    assert runs.reattach("t1", reopened) is True
    assert running.emit == reopened.emit
    assert running.emit_call == reopened.emit_call


def test_reattach_is_false_when_nothing_is_running(runs: LiveRuns) -> None:
    assert runs.reattach("t1", FakeSession("reopened")) is False


def test_reattach_to_the_same_session_is_a_no_op(runs: LiveRuns) -> None:
    """Reconnecting the socket that already owns the run must not self-assign."""
    session = FakeSession("a")
    runs.register("t1", session)
    assert runs.reattach("t1", session) is False
    assert session.emit == "emit-a"


def test_reattach_survives_a_session_that_will_not_take_it(runs: LiveRuns) -> None:
    """A broken redirect loses live output, never the run itself."""

    class Frozen(FakeSession):
        def __setattr__(self, name: str, value: object) -> None:
            if name == "emit":
                raise RuntimeError("read-only")
            super().__setattr__(name, value)

    running = FakeSession("running")
    runs.register("t1", running)
    object.__setattr__(running, "__class__", Frozen)

    assert runs.reattach("t1", FakeSession("reopened")) is False
    assert runs.is_live("t1")


def test_reattach_needs_both_a_thread_and_a_session(runs: LiveRuns) -> None:
    runs.register("t1", FakeSession("running"))
    assert runs.reattach(None, FakeSession("reopened")) is False
    assert runs.reattach("t1", None) is False


# --------------------------------------------------------------------------- #
# Stopping
# --------------------------------------------------------------------------- #


class FakeTask:
    def __init__(self, done: bool = False) -> None:
        self._done = done
        self.cancelled = False

    def done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        self.cancelled = True


def test_cancel_stops_the_task_that_is_doing_the_work(runs: LiveRuns) -> None:
    """Stop arrives on the reopened tab; the run is in another session."""
    task = FakeTask()
    runs.register("t1", FakeSession("running"), task)

    assert runs.cancel("t1") is True
    assert task.cancelled


def test_cancel_is_false_with_nothing_to_stop(runs: LiveRuns) -> None:
    assert runs.cancel("t1") is False
    assert runs.cancel(None) is False

    runs.register("t2", FakeSession("no-task"))
    assert runs.cancel("t2") is False


def test_a_finished_task_is_not_cancelled_again(runs: LiveRuns) -> None:
    """Stop clicked as the last step lands must not report a phantom stop."""
    task = FakeTask(done=True)
    runs.register("t1", FakeSession("running"), task)

    assert runs.cancel("t1") is False
    assert not task.cancelled


def test_the_second_reader_of_a_run_wins(runs: LiveRuns) -> None:
    """Two tabs on one conversation: the most recent one gets the output.

    Same rule chainlit applies to a websocket reconnect - there is one live
    connection per session, and it is the newest.
    """
    running = FakeSession("running")
    runs.register("t1", running)
    runs.reattach("t1", FakeSession("first-tab"))
    runs.reattach("t1", FakeSession("second-tab"))
    assert running.emit == "emit-second-tab"
