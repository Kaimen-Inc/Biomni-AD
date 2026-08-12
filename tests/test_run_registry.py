"""Tests for biomni.run_registry - durable records of a user's agent runs.

The point of the registry is that a user who closes the browser (or whose pod
restarts) can still find out what happened to a long run. The tests below pin
the two behaviours that make that true: records survive independently of the
session, and a run abandoned by a dead process is durably re-labelled
``interrupted`` instead of claiming to run forever.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
from biomni import run_registry as rr

_ENV_TO_CLEAR = (
    "BIOMNI_RUNS_STATE_DIR",
    "BIOMNI_STATE_DIR",
    "BIOMNI_RUN_STALE_AFTER_S",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)


def _iso_ago(seconds: float) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# RunRecord
# --------------------------------------------------------------------------- #


def test_record_label_is_a_single_trimmed_line():
    record = rr.RunRecord(run_id="r1", prompt="  analyse   APOE\n  expression  ")
    assert record.label == "analyse APOE expression"


def test_record_label_truncates_long_prompts():
    record = rr.RunRecord(run_id="r1", prompt="x" * 200)
    assert len(record.label) <= 80
    assert record.label.endswith("...")


def test_record_label_falls_back_to_run_id():
    assert rr.RunRecord(run_id="r1", prompt="").label == "r1"


def test_from_dict_rejects_unusable_payloads():
    assert rr.RunRecord.from_dict(None) is None
    assert rr.RunRecord.from_dict({}) is None  # no run_id
    assert rr.RunRecord.from_dict({"run_id": "r", "version": rr.RUN_RECORD_VERSION + 1}) is None


def test_from_dict_defaults_a_missing_status_to_interrupted():
    # A record with no status was written by something that died mid-write;
    # claiming "running" would be a lie.
    record = rr.RunRecord.from_dict({"run_id": "r1", "version": rr.RUN_RECORD_VERSION})
    assert record is not None
    assert record.status == "interrupted"


def test_terminal_statuses():
    assert rr.RunRecord(run_id="r", status="completed").is_terminal
    assert rr.RunRecord(run_id="r", status="interrupted").is_terminal
    assert not rr.RunRecord(run_id="r", status="running").is_terminal


def test_age_seconds_from_heartbeat():
    record = rr.RunRecord(run_id="r", heartbeat_at=_iso_ago(60))
    age = record.age_seconds()
    assert age is not None and 55 < age < 120


def test_age_seconds_is_none_for_unparseable_stamp():
    assert rr.RunRecord(run_id="r", heartbeat_at="not-a-date").age_seconds() is None


# --------------------------------------------------------------------------- #
# Registry writes and reads
# --------------------------------------------------------------------------- #


def test_start_then_finish_round_trip(tmp_path):
    registry = rr.RunRegistry(str(tmp_path))
    record = registry.start("user-1", "run_1", prompt="find AD targets", output_dir="/out/run_1")
    assert record.status == "running"

    registry.finish("user-1", record, status="completed")

    listed = registry.list_for_user("user-1")
    assert len(listed) == 1
    assert listed[0].status == "completed"
    assert listed[0].output_dir == "/out/run_1"
    assert listed[0].finished_at


def test_records_are_isolated_per_user(tmp_path):
    registry = rr.RunRegistry(str(tmp_path))
    registry.start("user-1", "run_1")
    registry.start("user-2", "run_2")
    assert [r.run_id for r in registry.list_for_user("user-1")] == ["run_1"]
    assert [r.run_id for r in registry.list_for_user("user-2")] == ["run_2"]


def test_list_is_newest_first_and_limited(tmp_path):
    registry = rr.RunRegistry(str(tmp_path))
    for index in range(5):
        record = registry.start("user-1", f"run_{index}")
        # Stamp distinct creation times so ordering is deterministic.
        record.created_at = _iso_ago(100 - index)
        registry._write("user-1", record)

    listed = registry.list_for_user("user-1", limit=3)
    assert [r.run_id for r in listed] == ["run_4", "run_3", "run_2"]


def test_list_for_unknown_user_is_empty(tmp_path):
    assert rr.RunRegistry(str(tmp_path)).list_for_user("nobody") == []


def test_corrupt_record_is_skipped_not_fatal(tmp_path):
    registry = rr.RunRegistry(str(tmp_path))
    registry.start("user-1", "good")
    (tmp_path / "user-1" / "bad.json").write_text("{oops", encoding="utf-8")
    assert [r.run_id for r in registry.list_for_user("user-1")] == ["good"]


def test_finish_can_attach_the_output_directory(tmp_path):
    registry = rr.RunRegistry(str(tmp_path))
    record = registry.start("user-1", "run_1")
    registry.finish("user-1", record, status="failed", error="boom", output_dir="/late/path")
    listed = registry.list_for_user("user-1")[0]
    assert listed.status == "failed"
    assert listed.error == "boom"
    assert listed.output_dir == "/late/path"


def test_heartbeat_advances_liveness(tmp_path):
    registry = rr.RunRegistry(str(tmp_path))
    record = registry.start("user-1", "run_1")
    record.heartbeat_at = _iso_ago(600)
    registry._write("user-1", record)

    registry.heartbeat("user-1", record)
    age = registry.list_for_user("user-1")[0].age_seconds()
    assert age is not None and age < 30


def test_heartbeat_does_not_resurrect_a_finished_run(tmp_path):
    registry = rr.RunRegistry(str(tmp_path))
    record = registry.start("user-1", "run_1")
    registry.finish("user-1", record, status="completed")
    stamp = registry.list_for_user("user-1")[0].heartbeat_at

    registry.heartbeat("user-1", record)
    assert registry.list_for_user("user-1")[0].heartbeat_at == stamp


# --------------------------------------------------------------------------- #
# Reconciliation of abandoned runs
# --------------------------------------------------------------------------- #


def test_reconcile_marks_a_stale_run_interrupted(tmp_path):
    registry = rr.RunRegistry(str(tmp_path))
    record = registry.start("user-1", "run_1", prompt="long job")
    record.heartbeat_at = _iso_ago(rr.stale_after_s() + 60)
    registry._write("user-1", record)

    reconciled = registry.reconcile("user-1")
    assert reconciled[0].status == "interrupted"
    assert reconciled[0].error


def test_reconcile_is_durable(tmp_path):
    """The correction must be written, not just displayed."""
    registry = rr.RunRegistry(str(tmp_path))
    record = registry.start("user-1", "run_1")
    record.heartbeat_at = _iso_ago(rr.stale_after_s() + 60)
    registry._write("user-1", record)

    registry.reconcile("user-1")
    fresh = rr.RunRegistry(str(tmp_path)).list_for_user("user-1")
    assert fresh[0].status == "interrupted"


def test_reconcile_leaves_a_live_run_alone(tmp_path):
    registry = rr.RunRegistry(str(tmp_path))
    registry.start("user-1", "run_1")
    assert registry.reconcile("user-1")[0].status == "running"


def test_reconcile_respects_the_stale_threshold_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMNI_RUN_STALE_AFTER_S", "5")
    registry = rr.RunRegistry(str(tmp_path))
    record = registry.start("user-1", "run_1")
    record.heartbeat_at = _iso_ago(30)
    registry._write("user-1", record)
    assert registry.reconcile("user-1")[0].status == "interrupted"


def test_reconcile_does_not_rewrite_completed_runs(tmp_path):
    registry = rr.RunRegistry(str(tmp_path))
    record = registry.start("user-1", "run_1")
    registry.finish("user-1", record, status="completed")
    record.heartbeat_at = _iso_ago(9999)
    registry._write("user-1", record)
    assert registry.reconcile("user-1")[0].status == "completed"


def test_unfinished_selects_only_failed_and_interrupted(tmp_path):
    registry = rr.RunRegistry(str(tmp_path))
    records = [
        rr.RunRecord(run_id="a", status="completed"),
        rr.RunRecord(run_id="b", status="interrupted"),
        rr.RunRecord(run_id="c", status="failed"),
        rr.RunRecord(run_id="d", status="running"),
    ]
    assert [r.run_id for r in registry.unfinished(records)] == ["b", "c"]


# --------------------------------------------------------------------------- #
# Disabled registry / location selection
# --------------------------------------------------------------------------- #


def test_disabled_registry_is_a_no_op():
    registry = rr.RunRegistry(None)
    assert not registry.enabled
    record = registry.start("user-1", "run_1")
    assert record.run_id == "run_1"  # still usable in-memory
    registry.heartbeat("user-1", record)
    registry.finish("user-1", record)
    assert registry.list_for_user("user-1") == []
    assert registry.reconcile("user-1") == []


def test_build_registry_prefers_explicit_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMNI_RUNS_STATE_DIR", str(tmp_path / "explicit"))
    monkeypatch.setenv("BIOMNI_STATE_DIR", str(tmp_path / "state"))
    assert rr.build_run_registry(str(tmp_path)).root == str(tmp_path / "explicit")


def test_build_registry_uses_state_dir_then_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMNI_STATE_DIR", str(tmp_path / "state"))
    assert rr.build_run_registry(None).root == str(tmp_path / "state" / "runs")

    monkeypatch.delenv("BIOMNI_STATE_DIR")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    expected = os.path.join(str(workspace), ".biomni", "runs")
    assert rr.build_run_registry(str(workspace)).root == expected


def test_build_registry_disabled_without_any_writable_location():
    assert rr.build_run_registry(None).enabled is False


def test_run_ids_cannot_escape_the_user_directory(tmp_path):
    registry = rr.RunRegistry(str(tmp_path))
    registry.start("user-1", "../escape")
    assert not (tmp_path / "escape.json").exists()
    assert list((tmp_path / "user-1").glob("*.json"))


def test_list_only_opens_the_newest_records(tmp_path, monkeypatch):
    """Parsing every record on session start would be unbounded I/O on the mount."""
    registry = rr.RunRegistry(str(tmp_path))
    for index in range(40):
        registry.start("user-1", f"run_202601{index:02d}_120000_job")

    opened: list[str] = []
    real_read = rr.read_json

    def _counting_read(path):
        opened.append(path)
        return real_read(path)

    monkeypatch.setattr(rr, "read_json", _counting_read)
    listed = registry.list_for_user("user-1", limit=5)

    assert len(listed) == 5
    assert len(opened) <= 12  # a small margin above the limit, not all 40
    # Still the newest ones, despite not reading everything.
    assert listed[0].run_id == "run_20260139_120000_job"


def test_heartbeat_and_finish_are_serialised(tmp_path):
    """A beat racing finish must not resurrect a completed run as running."""
    registry = rr.RunRegistry(str(tmp_path))
    record = registry.start("user-1", "run_1")
    registry.finish("user-1", record, status="completed")

    # Simulates the heartbeat thread waking after finish() has landed.
    registry.heartbeat("user-1", record)

    reloaded = rr.RunRegistry(str(tmp_path)).list_for_user("user-1")[0]
    assert reloaded.status == "completed"
    assert reloaded.finished_at
