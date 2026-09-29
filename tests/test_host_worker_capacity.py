"""Host-wide shared worker capacity across concurrent loops (issue #987)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from coding_review_agent_loop.test_workers import (
    ENV_HOST_SHARING,
    ENV_WORKER_RESERVATION,
    HOST_RESERVATION_SUFFIX,
    HostCapacity,
    WorkerBudget,
    WorkerBudgetLock,
    host_capacity,
    host_sharing_enabled,
    host_worker_pool,
)

GIB = 1024 ** 3


def _cpus(count, per_worker=GIB):
    """A capacity bounded by CPUs only (plenty of memory)."""
    return HostCapacity(count, 1024 * GIB, per_worker)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="flock-based locks")


def _lock(root, invocation):
    lock, problem = WorkerBudgetLock.acquire(invocation_id=invocation, cwd=root, root=root)
    assert lock is not None, problem
    return lock


def _records(root):
    return sorted(root.glob("*" + HOST_RESERVATION_SUFFIX))


def test_single_invocation_is_granted_its_full_budget_even_above_pool(tmp_path):
    root = tmp_path / "locks"
    lock = _lock(root, "only")
    try:
        assert lock.reserve_host_workers(16, _cpus(8)) == 16
    finally:
        lock.close()
    assert _records(root) == []


def test_contending_invocations_never_exceed_the_pool(tmp_path):
    root = tmp_path / "locks"
    locks = [_lock(root, f"inv-{index}") for index in range(4)]
    try:
        grants = [lock.reserve_host_workers(5, _cpus(8)) for lock in locks]
        assert grants == [5, 3, 0, 0]
        assert sum(grants) <= 8
        # Releasing one holder frees its share for the next.
        locks[0].close()
        assert locks[2].reserve_host_workers(5, _cpus(8)) == 5
    finally:
        for lock in locks:
            lock.close()
    assert _records(root) == []


def test_reservation_of_killed_holder_is_reclaimed(tmp_path):
    root = tmp_path / "locks"
    root.mkdir(mode=0o700)
    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from coding_review_agent_loop.test_workers import HostCapacity, WorkerBudgetLock\n"
        "root = Path(sys.argv[1])\n"
        "lock, _ = WorkerBudgetLock.acquire(invocation_id='victim', cwd=root, root=root)\n"
        "lock.reserve_host_workers(8, HostCapacity(8, None, 1))\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(root)], stdout=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        other = _lock(root, "survivor")
        try:
            assert other.reserve_host_workers(4, _cpus(8)) == 0  # the victim holds the pool
            child.kill()
            child.wait(timeout=10)
            assert len(_records(root)) == 1  # leaked by the kill ...
            assert other.reserve_host_workers(4, _cpus(8)) == 4  # ... and reclaimed
            assert [json.loads(p.read_text())["workers"] for p in _records(root)] == [4]
        finally:
            other.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    assert _records(root) == []


def test_reservation_is_kept_while_a_group_member_survives(tmp_path):
    root = tmp_path / "locks"
    member = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(1.5)"], start_new_session=True,
    )
    holder = _lock(root, "holder")
    assert holder.reserve_host_workers(8, _cpus(8)) == 8
    assert holder.hold_until_group_exits(member.pid) == "watcher"
    other = _lock(root, "other")
    try:
        # The holder's process has let go, but its group still has a live
        # member: the reservation must still count.
        assert other.reserve_host_workers(2, _cpus(8)) == 0
        member.wait(timeout=10)
        deadline = time.monotonic() + 10
        granted = 0
        while time.monotonic() < deadline and not granted:
            granted = other.reserve_host_workers(2, _cpus(8))
            time.sleep(0.05)
        assert granted == 2
    finally:
        other.close()


def test_host_pool_uses_host_facts_not_operator_budget():
    budget = WorkerBudget(
        2, "operator", "clamp", "operator",
        {"cpu_available": 8, "host_usable_bytes": 30 * 1024 ** 3,
         "per_worker_bytes": 1024 ** 3, "reserve_bytes": 1024 ** 3},
    )
    assert host_worker_pool(budget) == 8
    tight = WorkerBudget(
        2, "derived", "clamp", "cpu",
        {"cpu_available": 8, "host_usable_bytes": 4 * 1024 ** 3,
         "per_worker_bytes": 1024 ** 3, "reserve_bytes": 1024 ** 3},
    )
    assert host_worker_pool(tight) == 3
    assert host_capacity(tight) == HostCapacity(8, 3 * GIB, GIB)


def test_differently_sized_loops_cannot_overcommit_shared_memory(tmp_path):
    """A 4-GiB-per-worker loop and a default 1-GiB loop on an 8-CPU host."""
    root = tmp_path / "locks"
    heavy = _lock(root, "heavy")
    light = _lock(root, "light")
    try:
        # 11 GiB usable after the reserve: the heavy loop's own pool is 2.
        assert heavy.reserve_host_workers(2, HostCapacity(8, 11 * GIB, 4 * GIB)) == 2
        # The light loop sees 8 workers alone, but 8 GiB is already reserved.
        granted = light.reserve_host_workers(8, HostCapacity(8, 11 * GIB, GIB))
        assert granted == 3
        assert 2 * 4 * GIB + granted * GIB <= 11 * GIB
        # A stricter view recorded by one holder binds every later loop.
        light.close()
        again = _lock(root, "light")
        try:
            assert again.reserve_host_workers(8, HostCapacity(8, 64 * GIB, GIB)) == 3
        finally:
            again.close()
    finally:
        heavy.close()
        light.close()


def test_reservation_survives_a_killed_wrapper_while_its_target_runs(tmp_path):
    """The wrapper dies after spawning; the orphaned target keeps its share."""
    root = tmp_path / "locks"
    root.mkdir(mode=0o700)
    script = (
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "from coding_review_agent_loop.test_workers import ENV_WORKER_RESERVATION, HostCapacity, WorkerBudgetLock\n"
        "root = Path(sys.argv[1])\n"
        "lock, _ = WorkerBudgetLock.acquire(invocation_id='victim', cwd=root, root=root)\n"
        "lock.reserve_host_workers(8, HostCapacity(8, None, 1))\n"
        "mode = sys.argv[2]\n"
        "env = dict(os.environ)\n"
        "if mode == 'token': env[ENV_WORKER_RESERVATION] = lock.reservation_token\n"
        "if mode in ('anchor', 'window'): env = {}  # like env -i: no token\n"
        "extra = {}\n"
        "if mode == 'anchor':\n"
        "    fd, preexec = lock.launch_anchor()\n"
        "    extra = dict(pass_fds=(fd,), preexec_fn=preexec)\n"
        "if mode == 'window':\n"
        "    # The target still holds the inherited lock (killed before anchoring).\n"
        "    extra = dict(pass_fds=(lock.handle.fileno(),))\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
        " start_new_session=True, env=env, **extra)\n"
        "if mode == 'pgid': lock.record_process_group(child.pid)\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    for mode in ("pgid", "token", "anchor", "window"):
        wrapper = subprocess.Popen(
            [sys.executable, "-c", script, str(root), mode], stdout=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        target = int(wrapper.stdout.readline())
        try:
            wrapper.kill()
            wrapper.wait(timeout=10)
            other = _lock(root, f"other-{mode}")
            try:
                # The lock is free but the target still runs: no capacity.
                assert other.reserve_host_workers(2, _cpus(8)) == 0, mode
                os.kill(target, 9)
                deadline = time.monotonic() + 10
                granted = 0
                while time.monotonic() < deadline and not granted:
                    granted = other.reserve_host_workers(2, _cpus(8))
                    time.sleep(0.05)
                assert granted == 2, mode
            finally:
                other.close()
        finally:
            try:
                os.kill(target, 9)
            except ProcessLookupError:
                pass
            if wrapper.poll() is None:
                wrapper.kill()
                wrapper.wait(timeout=10)
    assert _records(root) == []


def test_host_sharing_opt_out():
    assert host_sharing_enabled({}) is True
    assert host_sharing_enabled({ENV_HOST_SHARING: "on"}) is True
    for value in ("off", "0", "false", "No"):
        assert host_sharing_enabled({ENV_HOST_SHARING: value}) is False


def _budget(workers):
    return WorkerBudget(
        workers, "inherited", "clamp", "cpu",
        {"cpu_available": 4, "host_usable_bytes": 64 * GIB, "per_worker_bytes": GIB, "reserve_bytes": GIB},
        False,
    )


def _env(invocation, sharing="on"):
    return {**os.environ, "AGENT_LOOP_INVOCATION_ID": invocation, ENV_HOST_SHARING: sharing}


def test_run_foreground_test_shrinks_or_busies_on_shared_pool(tmp_path):
    from coding_review_agent_loop.runner import run_foreground_test

    root = tmp_path / "locks"
    other = _lock(root, "other-loop")
    printer = [
        sys.executable, "-c",
        "import os; print('workers=' + os.environ['AGENT_LOOP_TEST_WORKERS']); "
        f"print('token=' + str(bool(os.environ.get({ENV_WORKER_RESERVATION!r}))))",
    ]
    try:
        assert other.reserve_host_workers(3, _cpus(4)) == 3
        shrunk = run_foreground_test(
            printer, cwd=tmp_path, timeout_seconds=30, env=_env("mine"), environment_is_complete=True,
            echo_output=False, worker_budget=_budget(4), worker_lock_root=root,
        )
        assert shrunk.returncode == 0
        assert "workers=1" in shrunk.output_tail
        assert "token=True" in shrunk.output_tail
        assert other.reserve_host_workers(4, _cpus(4)) == 4  # the pool is all "other-loop" again
        busy = run_foreground_test(
            printer, cwd=tmp_path, timeout_seconds=30, env=_env("mine"), environment_is_complete=True,
            echo_output=False, worker_budget=_budget(4), worker_lock_root=root,
        )
        assert busy.outcome == "worker-budget-busy"
        assert busy.returncode == 125
        assert "other agent-loop runs on this host" in busy.output_tail
        # The opt-out restores the per-invocation behavior.
        opted_out = run_foreground_test(
            printer, cwd=tmp_path, timeout_seconds=30, env=_env("mine", "off"), environment_is_complete=True,
            echo_output=False, worker_budget=_budget(4), worker_lock_root=root,
        )
        assert opted_out.returncode == 0 and "workers=4" in opted_out.output_tail
    finally:
        other.close()
    alone = run_foreground_test(
        printer, cwd=tmp_path, timeout_seconds=30, env=_env("mine"), environment_is_complete=True,
        echo_output=False, worker_budget=_budget(4), worker_lock_root=root,
    )
    assert alone.returncode == 0 and "workers=4" in alone.output_tail
    assert _records(root) == []


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as stream:
            return stream.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except OSError:
        return False


def _eventually_granted(lock, requested, capacity, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        granted = lock.reserve_host_workers(requested, capacity)
        if granted:
            return granted
        time.sleep(0.05)
    return 0


def test_normal_exit_keeps_reservation_while_escaped_descendant_runs(tmp_path):
    from coding_review_agent_loop.runner import run_foreground_test

    root = tmp_path / "locks"
    pid_file = tmp_path / "escaped.pid"
    script = (
        "import subprocess, sys; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))"
    )
    result = run_foreground_test(
        [sys.executable, "-c", script], cwd=tmp_path, timeout_seconds=30, env=_env("escaper"),
        environment_is_complete=True, echo_output=False, worker_budget=_budget(4), worker_lock_root=root,
    )
    escaped = int(pid_file.read_text())
    try:
        assert result.returncode == 0
        assert _pid_alive(escaped)
        assert len(_records(root)) == 1  # kept: the escaped worker carries the token
        other = _lock(root, "next-loop")
        try:
            assert other.reserve_host_workers(4, _cpus(4)) == 0
            os.kill(escaped, 9)
            assert _eventually_granted(other, 4, _cpus(4)) == 4
        finally:
            other.close()
    finally:
        try:
            os.kill(escaped, 9)
        except ProcessLookupError:
            pass
    assert _records(root) == []


def test_post_spawn_failure_keeps_reservation_while_target_runs(tmp_path):
    from coding_review_agent_loop.runner import run_foreground_test

    root = tmp_path / "locks"
    started: list[int] = []

    def explode(proc):
        started.append(proc.pid)
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError):
        run_foreground_test(
            [sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path, timeout_seconds=30,
            env=_env("exploder"), environment_is_complete=True, echo_output=False,
            worker_budget=_budget(4), worker_lock_root=root, process_started=explode,
        )
    target = started[0]
    try:
        assert _pid_alive(target)
        records = _records(root)
        assert len(records) == 1
        # The child anchored its own process group at launch.
        assert records[0].with_suffix(".pgid").read_text() == str(target)
        other = _lock(root, "next-loop")
        try:
            assert other.reserve_host_workers(4, _cpus(4)) == 0
            os.kill(target, 9)
            assert _eventually_granted(other, 4, _cpus(4)) == 4
        finally:
            other.close()
    finally:
        try:
            os.killpg(target, 9)
        except ProcessLookupError:
            pass
    assert _records(root) == []


# --- reservation telemetry (#1107) -------------------------------------------------


def _telemetry(tmp_path, name="t.jsonl"):
    from coding_review_agent_loop.worker_telemetry import ReservationTelemetry

    log = tmp_path / name
    return log, ReservationTelemetry(log, {"repo": "o/r", "run_id": "r1", "attribution_source": "runner", "lane": "broker"})


def _attempts(log):
    from coding_review_agent_loop.worker_telemetry import load_records

    return [r for r in load_records(log) if r["record"] == "attempt"]


def _run(tmp_path, root, telemetry, *, budget=None, env=None, cmd=None, **extra):
    from coding_review_agent_loop.runner import run_foreground_test

    return run_foreground_test(
        cmd or [sys.executable, "-c", "print('ok')"], cwd=tmp_path, timeout_seconds=30,
        env=env or _env("mine"), environment_is_complete=True, echo_output=False,
        worker_budget=budget if budget is not None else _budget(4), worker_lock_root=root,
        reservation_telemetry=telemetry, **extra,
    )


def test_telemetry_clean_grant(tmp_path):
    root = tmp_path / "locks"
    log, tel = _telemetry(tmp_path)
    result = _run(tmp_path, root, tel)
    assert result.returncode == 0
    (clean,) = _attempts(log)
    assert clean["outcome"] == "granted" and clean["workers_granted"] == clean["workers_requested"]
    assert clean["others_count"] == 0 and clean["retained"] is False
    assert clean["command_seconds"] is not None and clean["released_at"] >= clean["reserved_at"]
    assert clean["repo"] == "o/r" and clean["attribution_source"] == "runner"
    assert _records(root) == []


def test_telemetry_degraded_grant_keeps_notice_and_lowered_budget(tmp_path):
    root = tmp_path / "locks"
    other = _lock(root, "other-loop")
    notices = []
    try:
        assert other.reserve_host_workers(3, _cpus(4)) == 3
        log, tel = _telemetry(tmp_path)
        result = _run(
            tmp_path, root, tel, output_callback=notices.append,
            cmd=[sys.executable, "-c", "import os; print('workers=' + os.environ['AGENT_LOOP_TEST_WORKERS'])"],
        )
        assert "workers=1" in result.output_tail
        (row,) = _attempts(log)
        assert row["outcome"] == "degraded"
        assert (row["workers_requested"], row["workers_granted"]) == (4, 1)
        assert (row["others_count"], row["others_workers"]) == (1, 3)
        assert "4 CPU(s)" in row["capacity"]
        assert any("other agent-loop runs on this host hold part" in text for text in notices)
    finally:
        other.close()


def test_telemetry_refused_does_not_spawn_or_leave_a_reservation(tmp_path):
    root = tmp_path / "locks"
    marker = tmp_path / "spawned"
    other = _lock(root, "other-loop")
    try:
        assert other.reserve_host_workers(4, _cpus(4)) == 4
        log, tel = _telemetry(tmp_path)
        result = _run(
            tmp_path, root, tel, cmd=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        )
        assert result.returncode == 125 and "other agent-loop runs on this host" in result.output_tail
        assert not marker.exists()
        assert len(_records(root)) == 1  # only the foreign holder
        (row,) = _attempts(log)
        assert row["outcome"] == "refused" and row["workers_granted"] == 0
        assert row["reserved_at"] is None and row["command_seconds"] is None
        assert row["target_started_at"] is None
    finally:
        other.close()
    assert _records(root) == []


def test_telemetry_overlap_rejection_writes_no_record(tmp_path):
    from coding_review_agent_loop.test_runtime import acquire_command_lane

    root = tmp_path / "locks"
    env = _env("mine")
    command = [sys.executable, "-c", "pass"]
    first = acquire_command_lane(command, cwd=tmp_path, env=env)
    assert first is not None
    try:
        log, tel = _telemetry(tmp_path)
        result = _run(tmp_path, root, tel, env=env, cmd=command)
    finally:
        first.close()
    assert result.outcome == "overlap-rejected"
    assert not log.exists()


def test_telemetry_reservation_start_excludes_wait(tmp_path, monkeypatch):
    root = tmp_path / "locks"
    log, tel = _telemetry(tmp_path)
    real = WorkerBudgetLock.reserve_host_workers

    def slow(self, requested, capacity):
        time.sleep(0.3)
        return real(self, requested, capacity)

    monkeypatch.setattr(WorkerBudgetLock, "reserve_host_workers", slow)
    _run(tmp_path, root, tel)
    (row,) = _attempts(log)
    assert row["reserved_at"] - row["requested_at"] >= 0.3


def test_telemetry_unshared_records_command_interval(tmp_path):
    root = tmp_path / "locks"
    log, tel = _telemetry(tmp_path)
    _run(tmp_path, root, tel, env=_env("mine", "off"))
    log2, tel2 = _telemetry(tmp_path, "t2.jsonl")
    _run(tmp_path, root, tel2, budget=WorkerBudget(4, "inherited", "off", "cpu", {}, False))
    log3, tel3 = _telemetry(tmp_path, "t3.jsonl")
    from coding_review_agent_loop.runner import run_foreground_test

    run_foreground_test(
        [sys.executable, "-c", "print('ok')"], cwd=tmp_path, timeout_seconds=30,
        env=_env("mine"), environment_is_complete=True, echo_output=False, reservation_telemetry=tel3,
    )
    for path in (log, log2, log3):
        (row,) = _attempts(path)
        assert row["outcome"] == "unshared" and row["reserved_at"] is None
        assert row["target_started_at"] is not None and row["command_seconds"] is not None
    assert _records(root) == []


def test_telemetry_invocation_busy_error_and_dry_run(tmp_path, monkeypatch):
    root = tmp_path / "locks"
    holder = _lock(root, "mine")
    try:
        log, tel = _telemetry(tmp_path)
        result = _run(tmp_path, root, tel)
        assert result.outcome == "worker-budget-busy"
        (row,) = _attempts(log)
        assert row["outcome"] == "invocation-busy"
    finally:
        holder.close()

    def boom(self, requested, capacity):
        raise RuntimeError("boom")

    monkeypatch.setattr(WorkerBudgetLock, "reserve_host_workers", boom)
    log2, tel2 = _telemetry(tmp_path, "t2.jsonl")
    with pytest.raises(RuntimeError, match="boom"):
        _run(tmp_path, root, tel2)
    (row,) = _attempts(log2)
    assert row["outcome"] == "error"
    assert _records(root) == []

    log3, tel3 = _telemetry(tmp_path, "t3.jsonl")
    from coding_review_agent_loop.runner import run_foreground_test

    run_foreground_test(["true"], cwd=tmp_path, timeout_seconds=5, dry_run=True, reservation_telemetry=tel3)
    assert not log3.exists()


def test_telemetry_launch_failed_probe_emits_once(tmp_path):
    root = tmp_path / "locks"
    log, tel = _telemetry(tmp_path)
    result = _run(tmp_path, root, tel, cmd=["/nonexistent/binary-1107"])
    assert result.outcome == "launch-failed"
    assert len(_attempts(log)) == 1


def test_telemetry_does_not_change_grants_or_reservation_bytes(tmp_path):
    """The reservation file the target sees is identical with telemetry on and off."""
    script = (
        "import glob, json, os, sys; "
        "d = sys.argv[1]; "
        "token = os.environ['AGENT_LOOP_WORKER_RESERVATION']; "
        "own = [f for f in glob.glob(d + '/*.reservation') if token in f]; "
        "data = json.load(open(own[0])); data.pop('token'); data.pop('pgid'); "
        "print(json.dumps({'workers': os.environ['AGENT_LOOP_TEST_WORKERS'], 'file': data}, sort_keys=True))"
    )
    seen = []
    for enabled in (True, False):
        root = tmp_path / f"locks-{enabled}"
        other = _lock(root, "other-loop")
        try:
            assert other.reserve_host_workers(2, _cpus(4)) == 2
            tel = _telemetry(tmp_path, f"x{enabled}.jsonl")[1] if enabled else None
            result = _run(tmp_path, root, tel, cmd=[sys.executable, "-c", script, str(root)])
            assert result.returncode == 0, result.output_tail
            seen.append(result.output_tail.strip().splitlines()[-1])
        finally:
            other.close()
    assert seen[0] == seen[1]
    assert json.loads(seen[0])["workers"] == "2"


def test_telemetry_flags_retained_share(tmp_path):
    root = tmp_path / "locks"
    pid_file = tmp_path / "escaped.pid"
    script = (
        "import subprocess, sys; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))"
    )
    log, tel = _telemetry(tmp_path)
    _run(tmp_path, root, tel, cmd=[sys.executable, "-c", script])
    escaped = int(pid_file.read_text())
    try:
        (row,) = _attempts(log)
        assert row["retained"] is True and row["released_at"] is not None
        assert len(_records(root)) == 1  # the watcher/carrier keeps the share, as before
        from coding_review_agent_loop.worker_telemetry import run_duty_cycles

        start, end = row["requested_at"] - 1, row["released_at"] + 100
        duty = run_duty_cycles([
            {"record": "run-start", "run_id": "r1", "at": start},
            {"record": "run-end", "run_id": "r1", "at": end},
            {**row, "run_id": "r1"},
        ])["r1"]
        assert duty["retained"] == 1 and duty["duty_low"] < duty["duty_high"] <= 1.0
    finally:
        os.kill(escaped, 9)


def test_telemetry_failed_probe_emits_once_without_spawning_and_releases(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from coding_review_agent_loop import runner as runner_module
    from coding_review_agent_loop.test_runtime import LauncherProbeResult

    root = tmp_path / "locks"
    marker = tmp_path / "spawned"
    cmd = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
    monkeypatch.setattr(
        runner_module, "probe_inner_launcher",
        lambda argv, **_kw: LauncherProbeResult(tuple(argv), "failed", diagnostic="probe failed"),
    )
    log, tel = _telemetry(tmp_path)
    result = _run(tmp_path, root, tel, cmd=cmd)
    assert result.outcome == "launch-failed" and result.inner_exec == "failed"
    assert not marker.exists()
    (row,) = _attempts(log)
    assert row["outcome"] == "granted" and row["test_outcome"] == "launch-failed"
    assert row["target_started_at"] is None and row["command_seconds"] is None
    assert row["retained"] is False
    assert _records(root) == []  # reservation released as before


def test_telemetry_configuration_refusal_writes_no_record(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from coding_review_agent_loop import runner as runner_module

    monkeypatch.setattr(
        runner_module, "apply_worker_budget",
        lambda *_a, **_k: SimpleNamespace(refused="refused by configuration", cleanup=lambda: None, notices=()),
    )
    log, tel = _telemetry(tmp_path)
    result = _run(tmp_path, tmp_path / "locks", tel)
    assert result.outcome == "worker-budget-refused"
    assert not log.exists()
