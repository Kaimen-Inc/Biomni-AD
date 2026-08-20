"""Tests for chainlit_ui.status_gauges - the Chainlit-owned numbers in /status.

The point of these gauges is that they fail soft: a deployment with chat history
switched off, or a test environment with no Chainlit installed, must report a
smaller payload rather than a broken endpoint.
"""

from __future__ import annotations

import pytest
from biomni.status import ActivityTracker
from chainlit_ui import status_gauges


class FakePool:
    def __init__(self, checked_out: int, checked_in: int) -> None:
        self._out = checked_out
        self._in = checked_in

    def checkedout(self) -> int:
        return self._out

    def checkedin(self) -> int:
        return self._in


class FakeEngine:
    def __init__(self, pool) -> None:
        self.pool = pool


class FakeDataLayer:
    def __init__(self, pool) -> None:
        self.engine = FakeEngine(pool)


@pytest.fixture
def tracker() -> ActivityTracker:
    return ActivityTracker()


# --------------------------------------------------------------------------- #
# Database connections
# --------------------------------------------------------------------------- #


def test_counts_open_connections_not_just_busy_ones():
    layer = FakeDataLayer(FakePool(checked_out=2, checked_in=3))
    assert status_gauges.count_database_connections(layer) == 5


@pytest.mark.parametrize(
    "layer",
    [
        None,
        object(),  # no engine
        FakeDataLayer(pool=None),
        FakeDataLayer(pool=object()),  # a StaticPool has no checkedout()
    ],
)
def test_unmeasurable_pools_report_nothing(layer):
    assert status_gauges.count_database_connections(layer) is None


def test_database_gauge_is_registered_when_it_can_be_read(tracker):
    status_gauges.register_database_gauge(tracker, FakeDataLayer(FakePool(1, 1)))
    assert tracker.snapshot()["internal_processes"]["database_connections"] == 2


@pytest.mark.parametrize("layer", [None, FakeDataLayer(pool=None)])
def test_no_database_gauge_means_no_field(tracker, layer):
    """Chat history is off, or the layer has no pool: omit rather than report 0."""
    status_gauges.register_database_gauge(tracker, layer)
    assert "database_connections" not in tracker.snapshot()["internal_processes"]


def test_the_gauge_is_live_not_a_snapshot(tracker):
    pool = FakePool(1, 0)
    status_gauges.register_database_gauge(tracker, FakeDataLayer(pool))
    pool._out = 6
    assert tracker.snapshot()["internal_processes"]["database_connections"] == 6


# --------------------------------------------------------------------------- #
# Open sessions
# --------------------------------------------------------------------------- #


def test_session_gauge_reports_chainlits_own_registry(tracker):
    chainlit_session = pytest.importorskip("chainlit.session", reason="chainlit not installed")

    status_gauges.register_session_gauge(tracker)
    chainlit_session.ws_sessions_id["fake-session"] = object()
    try:
        assert tracker.snapshot()["open_sessions"] == len(chainlit_session.ws_sessions_id)
    finally:
        chainlit_session.ws_sessions_id.pop("fake-session", None)


def test_session_gauge_is_skipped_when_the_count_is_unavailable(tracker, monkeypatch):
    """No Chainlit (bare test env, or an embedding of the agent): fall back."""

    def unavailable() -> int:
        raise ImportError("no chainlit here")

    monkeypatch.setattr(status_gauges, "count_open_sessions", unavailable)
    tracker.session_opened()
    status_gauges.register_session_gauge(tracker)
    assert tracker.snapshot()["open_sessions"] == 1  # the tracker's own counter
