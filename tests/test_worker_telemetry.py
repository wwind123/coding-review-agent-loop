"""Reservation telemetry helpers (#1107)."""

from __future__ import annotations

import multiprocessing
import os

import pytest

from coding_review_agent_loop import worker_telemetry as wt
from coding_review_agent_loop.runner import run_foreground_test


def test_path_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert wt.telemetry_log_path({}) == tmp_path / "coding-review-agent-loop" / "telemetry" / "worker-reservations.jsonl"
    absolute = tmp_path / "x.jsonl"
    assert wt.telemetry_log_path({wt.LOG_ENV: str(absolute)}) == absolute
    for value in ("off", "0", "false", "No", "relative/path.jsonl", "", "   "):
        assert wt.telemetry_log_path({wt.LOG_ENV: value}) is None


def test_unwritable_log_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    wt.append_record(blocker / "sub" / "log.jsonl", {"a": 1})
    tel = wt.ReservationTelemetry(blocker / "sub" / "log.jsonl", {})
    result = run_foreground_test(
        ["python3", "-c", "print('hi')"], cwd=tmp_path, timeout_seconds=30, echo_output=False,
        reservation_telemetry=tel,
    )
    assert result.returncode == 0


def _append(path, index):
    for n in range(20):
        wt.append_record(path, {"record": "attempt", "i": index, "n": n, "pad": "x" * 200})


def test_concurrent_append_and_truncated_line(tmp_path):
    log = tmp_path / "log.jsonl"
    procs = [multiprocessing.Process(target=_append, args=(log, i)) for i in range(4)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    with open(log, "a") as stream:
        stream.write('{"record": "attempt", "trunc')
    records = list(wt.load_records(log))
    assert len(records) == 80


def _run(run_id, start, end):
    rows = [{"record": "run-start", "run_id": run_id, "at": start}]
    if end is not None:
        rows.append({"record": "run-end", "run_id": run_id, "at": end})
    return rows


def _attempt(run_id, reserved, released, **extra):
    return {"record": "attempt", "run_id": run_id, "outcome": "granted", "reserved_at": reserved,
            "released_at": released, "retained": False, **extra}


def test_duty_cycle_excludes_idle_and_wait():
    records = _run("a", 0, 100) + [
        _attempt("a", 10, 20, requested_at=0), _attempt("a", 50, 60),
    ]
    out = wt.run_duty_cycles(records)["a"]
    assert out["window_seconds"] == 100
    assert out["duty_low"] == pytest.approx(0.2) and out["duty_high"] == pytest.approx(0.2)
    assert out["incomplete"] == []


def test_retained_bounded_by_run_end_and_clipped():
    records = _run("a", 10, 100) + [
        _attempt("a", 20, 30, retained=True),
        _attempt("a", 0, 12),  # starts before run-start
        _attempt("a", 95, 500),  # released long after run-end
    ]
    out = wt.run_duty_cycles(records)["a"]
    assert out["duty_low"] < out["duty_high"] <= 1.0
    assert out["duty_high"] == pytest.approx((100 - 20 + 2) / 90)
    assert out["retained"] == 1


def test_missing_run_end_and_run_start_are_incomplete():
    records = _run("a", 0, None) + [_attempt("a", 10, 20)] + [_attempt("b", 1, 2)]
    out = wt.run_duty_cycles(records)
    assert out["a"]["incomplete"] == ["no run-end"] and out["a"]["window_seconds"] == 20
    assert out["b"]["incomplete"] == ["no run-start"] and out["b"]["duty_low"] is None


def test_unshared_uses_command_interval_and_overlap():
    records = _run("a", 0, 100) + [
        {"record": "attempt", "run_id": "a", "outcome": "unshared", "reserved_at": None,
         "target_started_at": 10, "command_seconds": 20, "released_at": 31, "retained": False},
    ]
    info = wt.run_intervals(records)["a"]
    assert info["low"] == [(10, 30)]
    assert wt.overlap_seconds(info["low"], [(25, 60)]) == 5


def test_attribution_environment_round_trip():
    env = {"AGENT_LOOP_RUN_REPO": "o/r", "AGENT_LOOP_RUN_ID": "r1", "AGENT_LOOP_RUN_ISSUE": "7"}
    got = wt.attribution_from_environment(env)
    assert got["attribution_source"] == "environment" and got["issue_number"] == 7 and got["pr_number"] is None
    assert wt.attribution_from_environment({})["attribution_source"] == "none"
    assert wt.attribution_environment({"repo": "o/r", "run_id": "r1", "issue_number": 7, "pr_number": None}) == {
        "AGENT_LOOP_RUN_REPO": "o/r", "AGENT_LOOP_RUN_ID": "r1", "AGENT_LOOP_RUN_ISSUE": "7",
    }


def test_error_outcome_after_reap_still_counts_command_interval():
    records = _run("a", 0, 100) + [
        {"record": "attempt", "run_id": "a", "outcome": "error", "reserved_at": None,
         "target_started_at": 10, "command_seconds": 20, "released_at": 31, "retained": False},
    ]
    assert wt.run_duty_cycles(records)["a"]["duty_low"] == pytest.approx(0.2)


def test_run_end_without_run_start_or_attempts_is_reported():
    out = wt.run_duty_cycles([{"record": "run-end", "run_id": "z", "at": 5}])
    assert out["z"]["incomplete"] == ["no run-start"] and out["z"]["window_seconds"] is None
