"""Tests for biomni.status - the monitoring framework's ``GET /status`` view.

The properties that matter to the platform are locked in here: a run in flight
is never reported as idle, the monitoring poll itself does not count as
activity (or the answer would always be "active"), and a broken gauge costs one
field rather than the whole endpoint.
"""

from __future__ import annotations

import pytest
from biomni import status

# The route test needs fastapi/starlette, which are optional (the `chainlit`
# extra) and absent from the minimal CI job - same guard as tests/test_health.py.
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

requires_fastapi = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi/starlette not installed (chainlit extra)")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("BIOMNI_STATUS_INACTIVITY_SECONDS", raising=False)


class FakeClock:
    """A wall clock and a monotonic clock that advance together, on command."""

    def __init__(self, epoch: float = 1_786_000_000.0) -> None:
        self.epoch = epoch
        self.elapsed = 0.0

    def time(self) -> float:
        return self.epoch + self.elapsed

    def monotonic(self) -> float:
        return self.elapsed

    def advance(self, seconds: float) -> None:
        self.elapsed += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def tracker(clock: FakeClock) -> status.ActivityTracker:
    return status.ActivityTracker(clock=clock.time, monotonic=clock.monotonic)


# --------------------------------------------------------------------------- #
# The inactivity window
# --------------------------------------------------------------------------- #


def test_default_window_is_four_hours():
    assert status.inactivity_window_seconds() == 4 * 60 * 60
    assert status.DEFAULT_INACTIVITY_SECONDS == 4 * 60 * 60


def test_window_is_configurable(monkeypatch):
    monkeypatch.setenv("BIOMNI_STATUS_INACTIVITY_SECONDS", "900")
    assert status.inactivity_window_seconds() == 900


def test_zero_window_means_active_only_while_working(monkeypatch, tracker, clock):
    monkeypatch.setenv("BIOMNI_STATUS_INACTIVITY_SECONDS", "0")
    assert tracker.snapshot()["active"] is True  # the activity is "just now"
    clock.advance(1)
    assert tracker.snapshot()["active"] is False


@pytest.mark.parametrize("raw", ["", "   ", "soon", "-60"])
def test_unparseable_window_falls_back_to_the_default(monkeypatch, raw):
    """A typo in a manifest must not break the endpoint."""
    monkeypatch.setenv("BIOMNI_STATUS_INACTIVITY_SECONDS", raw)
    assert status.inactivity_window_seconds() == status.DEFAULT_INACTIVITY_SECONDS


# --------------------------------------------------------------------------- #
# active / last_activity
# --------------------------------------------------------------------------- #


def test_a_fresh_process_is_active(tracker):
    """A pod that was just rolled out has not been idle, only unasked."""
    assert tracker.snapshot()["active"] is True


def test_inactive_once_the_window_has_passed(tracker, clock):
    clock.advance(status.DEFAULT_INACTIVITY_SECONDS + 1)
    assert tracker.snapshot()["active"] is False


def test_activity_restarts_the_window(tracker, clock):
    clock.advance(status.DEFAULT_INACTIVITY_SECONDS + 1)
    tracker.record_activity("message")
    assert tracker.snapshot()["active"] is True
    assert tracker.snapshot()["last_activity"] == int(clock.time())


def test_a_long_run_keeps_the_app_active(tracker, clock):
    """The one case that must never be wrong: work in flight, well past the window."""
    with tracker.track_job("agent_run"):
        clock.advance(status.DEFAULT_INACTIVITY_SECONDS * 10)
        snapshot = tracker.snapshot()
        assert snapshot["active"] is True
        assert snapshot["background_jobs"] == 1


def test_reading_the_status_is_not_activity(tracker, clock):
    """Otherwise the monitoring poll alone would pin the app to active forever."""
    clock.advance(status.DEFAULT_INACTIVITY_SECONDS + 1)
    assert tracker.snapshot()["active"] is False
    clock.advance(1)
    assert tracker.snapshot()["active"] is False


def test_last_activity_is_an_epoch_timestamp(tracker, clock):
    reported = tracker.snapshot()["last_activity"]
    assert isinstance(reported, int)
    assert reported == int(clock.epoch)


def test_idle_time_survives_a_wall_clock_jump(tracker, clock):
    """An NTP step must not make a busy app look idle for hours."""
    clock.epoch -= 6 * 60 * 60  # the clock is corrected backwards
    assert tracker.snapshot()["active"] is True


# --------------------------------------------------------------------------- #
# Counters
# --------------------------------------------------------------------------- #


def test_jobs_are_counted_while_they_run(tracker):
    assert tracker.snapshot()["background_jobs"] == 0
    with tracker.track_job():
        with tracker.track_job():
            assert tracker.snapshot()["background_jobs"] == 2
        assert tracker.snapshot()["background_jobs"] == 1
    assert tracker.snapshot()["background_jobs"] == 0


def test_a_failed_job_is_still_released(tracker):
    """Cancellation is routine here (the Stop button, the run timeout)."""
    with pytest.raises(RuntimeError), tracker.track_job():
        raise RuntimeError("run blew up")
    assert tracker.snapshot()["background_jobs"] == 0


def test_counters_never_go_negative(tracker):
    tracker.job_finished()
    tracker.session_closed()
    snapshot = tracker.snapshot()
    assert snapshot["background_jobs"] == 0
    assert snapshot["open_sessions"] == 0


def test_sessions_are_counted(tracker):
    tracker.session_opened()
    tracker.session_opened()
    assert tracker.snapshot()["open_sessions"] == 2
    tracker.session_closed()
    assert tracker.snapshot()["open_sessions"] == 1


def test_an_open_session_alone_does_not_mean_active(tracker, clock):
    """A tab left open for days is not work; a tab in use generates activity."""
    tracker.session_opened()
    clock.advance(status.DEFAULT_INACTIVITY_SECONDS + 1)
    snapshot = tracker.snapshot()
    assert snapshot["open_sessions"] == 1
    assert snapshot["active"] is False


# --------------------------------------------------------------------------- #
# Gauges
# --------------------------------------------------------------------------- #


def test_a_gauge_overrides_the_internal_session_count(tracker):
    tracker.session_opened()
    tracker.register_gauge("open_sessions", lambda: 7)
    assert tracker.snapshot()["open_sessions"] == 7


def test_a_failing_gauge_does_not_break_the_endpoint(tracker):
    def broken() -> int:
        raise RuntimeError("no registry")

    tracker.session_opened()
    tracker.register_gauge("open_sessions", broken)
    tracker.register_gauge("database_connections", broken)
    snapshot = tracker.snapshot()
    assert snapshot["open_sessions"] == 1  # fell back to the counter we own
    assert "database_connections" not in snapshot["internal_processes"]


@pytest.mark.parametrize("value", [None, "4", -1, True])
def test_nonsense_gauge_readings_are_dropped(tracker, value):
    tracker.register_gauge("database_connections", lambda: value)
    assert "database_connections" not in tracker.snapshot()["internal_processes"]


def test_database_connections_reported_when_available(tracker):
    tracker.register_gauge("database_connections", lambda: 4)
    assert tracker.snapshot()["internal_processes"]["database_connections"] == 4


def test_thread_count_always_applies(tracker):
    assert tracker.snapshot()["internal_processes"]["active_threads"] >= 1


# --------------------------------------------------------------------------- #
# Payload shape - this is the contract with the monitoring framework
# --------------------------------------------------------------------------- #


def test_payload_matches_the_agreed_schema(tracker):
    snapshot = tracker.snapshot()
    assert set(snapshot) == {
        "active",
        "last_activity",
        "schema_version",
        "background_jobs",
        "open_sessions",
        "internal_processes",
    }
    assert isinstance(snapshot["active"], bool)
    assert snapshot["schema_version"] == status.SCHEMA_VERSION == 1
    assert isinstance(snapshot["background_jobs"], int)
    assert isinstance(snapshot["open_sessions"], int)
    assert isinstance(snapshot["internal_processes"], dict)


def test_module_level_tracker_serves_the_payload():
    payload = status.status_payload()
    assert payload["schema_version"] == status.SCHEMA_VERSION
    assert "active" in payload


# --------------------------------------------------------------------------- #
# Route registration
# --------------------------------------------------------------------------- #


@requires_fastapi
def test_status_route_answers_with_the_payload():
    app = FastAPI()
    status.register_status_route(app)
    body = TestClient(app).get("/status").json()
    assert body["active"] in (True, False)
    assert body["schema_version"] == status.SCHEMA_VERSION


@requires_fastapi
def test_status_route_is_not_shadowed_by_a_catch_all():
    """Chainlit serves the SPA from `GET /{full_path:path}`, registered last."""

    async def spa(_request):
        return PlainTextResponse("single page app")

    app = FastAPI()
    app.router.routes.append(Route("/{full_path:path}", spa, methods=["GET"]))
    status.register_status_route(app)

    assert TestClient(app).get("/status").json()["schema_version"] == status.SCHEMA_VERSION
    assert TestClient(app).get("/anything-else").text == "single page app"


@requires_fastapi
def test_re_registration_does_not_duplicate_the_route():
    """The entry module is re-imported on dev reload."""
    app = FastAPI()
    status.register_status_route(app)
    status.register_status_route(app)
    assert sum(1 for r in app.router.routes if getattr(r, "path", None) == "/status") == 1


@requires_fastapi
def test_status_path_is_configurable():
    app = FastAPI()
    status.register_status_route(app, path="/_status")
    assert TestClient(app).get("/_status").status_code == 200
