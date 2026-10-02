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
    ENV_HOST_WAIT,
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


def _env(invocation, sharing="on", wait=None):
    values = {**os.environ, "AGENT_LOOP_INVOCATION_ID": invocation, ENV_HOST_SHARING: sharing}
    if wait is not None:
        values[ENV_HOST_WAIT] = str(wait)
    return values


def test_run_foreground_test_degrades_on_timeout_and_never_busies(tmp_path):
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
            printer, cwd=tmp_path, timeout_seconds=30, env=_env("mine", wait=0.3),
            environment_is_complete=True,
            echo_output=False, worker_budget=_budget(4), worker_lock_root=root,
        )
        assert shrunk.returncode == 0
        assert "workers=1" in shrunk.output_tail
        assert "token=True" in shrunk.output_tail
        assert other.reserve_host_workers(4, _cpus(4)) == 4  # the pool is all "other-loop" again
        oversubscribed = run_foreground_test(
            printer, cwd=tmp_path, timeout_seconds=30, env=_env("mine", wait=0.3),
            environment_is_complete=True,
            echo_output=False, worker_budget=_budget(4), worker_lock_root=root,
        )
        assert oversubscribed.outcome == "passed" and oversubscribed.returncode == 0
        assert "workers=1" in oversubscribed.output_tail
        assert oversubscribed.outcome != "worker-budget-busy" and oversubscribed.returncode != 125
        assert "is busy" not in oversubscribed.output_tail
        assert "wait for them to finish" not in oversubscribed.output_tail
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
            tmp_path, root, tel, output_callback=notices.append, env=_env("mine", wait=0.3),
            cmd=[sys.executable, "-c", "import os; print('workers=' + os.environ['AGENT_LOOP_TEST_WORKERS'])"],
        )
        assert "workers=1" in result.output_tail
        (row,) = _attempts(log)
        assert row["outcome"] == "degraded"
        assert (row["workers_requested"], row["workers_granted"]) == (4, 1)
        assert (row["others_count"], row["others_workers"]) == (1, 3)
        assert "4 CPU(s)" in row["capacity"]
        assert any("this command runs with 1 worker(s) instead of 4" in text for text in notices)
        assert row["wait_timed_out"] is True and row["wait_bound_seconds"] == 0.3
        assert row["reservation_wait_seconds"] >= 0.3 and row["oversubscribed"] is False
    finally:
        other.close()


def test_telemetry_no_free_capacity_runs_one_oversubscribed_worker(tmp_path):
    root = tmp_path / "locks"
    marker = tmp_path / "spawned"
    other = _lock(root, "other-loop")
    try:
        assert other.reserve_host_workers(4, _cpus(4)) == 4
        log, tel = _telemetry(tmp_path)
        result = _run(
            tmp_path, root, tel, env=_env("mine", wait=0.3),
            cmd=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        )
        assert result.returncode == 0 and "is busy" not in result.output_tail
        assert marker.exists()
        assert len(_records(root)) == 1  # only the foreign holder once released
        (row,) = _attempts(log)
        assert row["outcome"] == "oversubscribed" and row["workers_granted"] == 1
        assert row["oversubscribed"] is True and row["wait_timed_out"] is True
        assert row["command_seconds"] is not None
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

    def slow(self, requested, capacity, **kwargs):
        time.sleep(0.3)
        return real(self, requested, capacity, **kwargs)

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

    def boom(self, requested, capacity, **kwargs):
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
            result = _run(
                tmp_path, root, tel, env=_env("mine", wait=0.2),
                cmd=[sys.executable, "-c", script, str(root)],
            )
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


def test_telemetry_release_time_is_stamped_before_post_release_work(tmp_path):
    import time

    log, tel = _telemetry(tmp_path)
    tel.begin(2)
    tel.reserved(2, None)
    tel.released()
    stamped = tel.data["released_at"]
    time.sleep(0.05)  # e.g. a slow output_callback delivering notices after close
    tel.emit()
    (rec,) = _attempts(log)
    assert rec["released_at"] == stamped
    assert time.time() - stamped >= 0.05


def test_telemetry_run_foreground_test_stamps_release_before_notices(tmp_path):
    import time

    root = tmp_path / "locks"
    log, tel = _telemetry(tmp_path)
    stamps = []

    def slow_notice(msg):
        if "late notice" in msg:
            stamps.append(tel.data["released_at"])
            time.sleep(0.2)

    from coding_review_agent_loop import runner

    real = runner.analyze_worker_report

    def with_notice(*a, **k):
        analysis = real(*a, **k)
        import dataclasses

        return dataclasses.replace(analysis, notices=("late notice",))

    orig = runner.analyze_worker_report
    runner.analyze_worker_report = with_notice
    try:
        _run(tmp_path, root, tel, output_callback=slow_notice)
    finally:
        runner.analyze_worker_report = orig
    (rec,) = _attempts(log)
    assert stamps and rec["released_at"] == stamps[0]


# --- exclusive bounded wait (#1108) ------------------------------------------------

import threading  # noqa: E402

from coding_review_agent_loop import test_workers as workers_module  # noqa: E402
from coding_review_agent_loop.test_workers import (  # noqa: E402
    DEFAULT_HOST_WAIT_SECONDS,
    MAX_HOST_WAIT_SECONDS,
    HostCapacityMutexTimeout,
    HostWaitCancelled,
    _host_capacity_mutex,
    host_wait_seconds,
)

_PRINTER = [
    sys.executable, "-c",
    "import os; print('workers=' + os.environ['AGENT_LOOP_TEST_WORKERS'])",
]


def _fast(monkeypatch):
    monkeypatch.setattr(workers_module, "HOST_WAIT_POLL_INITIAL_SECONDS", 0.05)
    monkeypatch.setattr(workers_module, "HOST_WAIT_POLL_MAX_SECONDS", 0.05)


def test_wait_config_parsing():
    assert host_wait_seconds({}) == (DEFAULT_HOST_WAIT_SECONDS, None)
    assert DEFAULT_HOST_WAIT_SECONDS == 1200.0
    assert host_wait_seconds({ENV_HOST_WAIT: "0"}) == (0.0, None)
    assert host_wait_seconds({ENV_HOST_WAIT: "-5"}) == (0.0, None)
    assert host_wait_seconds({ENV_HOST_WAIT: "7.5"}) == (7.5, None)
    assert host_wait_seconds({ENV_HOST_WAIT: "1e9"}) == (MAX_HOST_WAIT_SECONDS, None)
    for bad in ("abc", "nan", "inf"):
        value, notice = host_wait_seconds({ENV_HOST_WAIT: bad})
        assert value == DEFAULT_HOST_WAIT_SECONDS and notice and ENV_HOST_WAIT in notice


def test_heartbeat_interval_is_well_below_minimum_client_receive_timeout():
    assert workers_module.HOST_WAIT_HEARTBEAT_SECONDS * 2 < 10


def test_exclusive_admission_serialises_even_when_both_budgets_fit(tmp_path):
    root = tmp_path / "locks"
    a, b = _lock(root, "a"), _lock(root, "b")
    try:
        assert a.reserve_host_workers(4, _cpus(8)) == 4
        # The divide rule would grant 4 more; exclusive admission must not.
        assert b.reserve_host_workers(4, _cpus(8), exclusive=True) == 0
        assert b.last_free == 4 and len(_records(root)) == 1
        a.close()
        assert b.reserve_host_workers(4, _cpus(8), exclusive=True) == 4
    finally:
        a.close()
        b.close()
    assert _records(root) == []


def test_floor_grants_one_oversubscribed_worker(tmp_path):
    root = tmp_path / "locks"
    a, b = _lock(root, "a"), _lock(root, "b")
    try:
        assert a.reserve_host_workers(4, _cpus(4)) == 4
        assert b.reserve_host_workers(4, _cpus(4), floor=1) == 1
        assert b.last_free == 0 and len(_records(root)) == 2
    finally:
        a.close()
        b.close()


def test_wait_then_full_grant_and_release_is_not_blocked(tmp_path, monkeypatch):
    _fast(monkeypatch)
    root = tmp_path / "locks"
    a, b = _lock(root, "a"), _lock(root, "b")
    notices = []
    try:
        assert a.reserve_host_workers(4, _cpus(4)) == 4
        result = {}
        started = threading.Event()

        def wait():
            result["r"] = b.wait_for_host_workers(
                4, _cpus(4), wait_seconds=20, notify=notices.append,
                on_wait_start=lambda *_a: started.set(),
            )

        thread = threading.Thread(target=wait)
        thread.start()
        assert started.wait(5)
        time.sleep(0.3)
        released = time.monotonic()
        a.close()  # takes the accounting mutex; a waiter must not hold it
        assert time.monotonic() - released < 2
        thread.join(10)
        assert not thread.is_alive()
        r = result["r"]
        assert r.granted == 4 and not r.timed_out and not r.oversubscribed
        assert r.waited_seconds > 0.2
        assert len(notices) == 1 and "waiting at most" in notices[0]
    finally:
        a.close()
        b.close()


def test_uncontended_wait_is_immediate_without_notice(tmp_path):
    root = tmp_path / "locks"
    lock = _lock(root, "only")
    notices = []
    try:
        r = lock.wait_for_host_workers(4, _cpus(4), wait_seconds=20, notify=notices.append)
        assert (r.granted, r.waited_seconds, r.timed_out) == (4, 0.0, False)
        assert notices == []
    finally:
        lock.close()
    assert _records(root) == []


def test_wait_timeout_degrades_to_free_then_oversubscribes(tmp_path, monkeypatch):
    _fast(monkeypatch)
    root = tmp_path / "locks"
    a, b = _lock(root, "a"), _lock(root, "b")
    try:
        assert a.reserve_host_workers(3, _cpus(4)) == 3
        r = b.wait_for_host_workers(4, _cpus(4), wait_seconds=0.2, notify=lambda _t: None)
        assert (r.granted, r.timed_out, r.oversubscribed) == (1, True, False)
        a.close()
        b.close()
        a, b = _lock(root, "a"), _lock(root, "b")
        assert a.reserve_host_workers(4, _cpus(4)) == 4
        r = b.wait_for_host_workers(4, _cpus(4), wait_seconds=0.2, notify=lambda _t: None)
        assert (r.granted, r.timed_out, r.oversubscribed) == (1, True, True)
        b.close()
        b = _lock(root, "b")
        zero = b.wait_for_host_workers(4, _cpus(4), wait_seconds=0, notify=lambda _t: None)
        assert (zero.granted, zero.timed_out, zero.oversubscribed) == (1, False, True)
    finally:
        a.close()
        b.close()


def test_contended_run_waits_then_runs_at_full_budget(tmp_path, monkeypatch):
    _fast(monkeypatch)
    root = tmp_path / "locks"
    other = _lock(root, "other-loop")
    log, tel = _telemetry(tmp_path)
    outcome = {}
    assert other.reserve_host_workers(4, _cpus(4)) == 4

    def run():
        outcome["r"] = _run(
            tmp_path, root, tel, env=_env("mine", wait=30), cmd=_PRINTER,
        )

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.5)
    assert thread.is_alive()
    other.close()
    thread.join(20)
    assert not thread.is_alive()
    r = outcome["r"]
    assert r.returncode == 0 and "workers=4" in r.output_tail
    (row,) = _attempts(log)
    assert row["outcome"] == "granted" and row["wait_timed_out"] is False
    assert row["reservation_wait_seconds"] > 0.3 and row["workers_granted"] == 4
    assert _records(root) == []


def test_exclusive_run_does_not_overlap_when_both_would_fit(tmp_path, monkeypatch):
    _fast(monkeypatch)
    root = tmp_path / "locks"
    other = _lock(root, "other-loop")
    big = HostCapacity(8, 1024 * GIB, GIB)
    monkeypatch.setattr(workers_module, "host_capacity", lambda _b: big)
    import coding_review_agent_loop.runner as runner_module

    monkeypatch.setattr(runner_module, "host_capacity", lambda _b: big)
    assert other.reserve_host_workers(4, big) == 4
    log, tel = _telemetry(tmp_path)
    outcome = {}
    thread = threading.Thread(
        target=lambda: outcome.setdefault("r", _run(tmp_path, root, tel, env=_env("mine", wait=30), cmd=_PRINTER))
    )
    thread.start()
    time.sleep(0.4)
    assert thread.is_alive() and "r" not in outcome  # would have run beside A under the divide rule
    other.close()
    thread.join(20)
    assert "workers=4" in outcome["r"].output_tail


def test_wait_does_not_consume_the_command_timeout(tmp_path, monkeypatch):
    _fast(monkeypatch)
    root = tmp_path / "locks"
    other = _lock(root, "other-loop")
    assert other.reserve_host_workers(4, _cpus(4)) == 4
    from coding_review_agent_loop.runner import run_foreground_test

    outcome = {}

    def run():
        outcome["r"] = run_foreground_test(
            _PRINTER, cwd=tmp_path, timeout_seconds=1, env=_env("mine", wait=30),
            environment_is_complete=True, echo_output=False,
            worker_budget=_budget(4), worker_lock_root=root,
        )

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(1.6)  # longer than the command's own timeout
    other.close()
    thread.join(20)
    assert outcome["r"].outcome == "passed" and outcome["r"].elapsed_seconds < 1


def test_sharing_off_never_waits_or_writes_records(tmp_path):
    root = tmp_path / "locks"
    other = _lock(root, "other-loop")
    try:
        assert other.reserve_host_workers(4, _cpus(4)) == 4
        log, tel = _telemetry(tmp_path)
        result = _run(tmp_path, root, tel, env=_env("mine", "off"), cmd=_PRINTER)
        assert result.returncode == 0 and "workers=4" in result.output_tail
        assert len(_records(root)) == 1
        (row,) = _attempts(log)
        assert row["reservation_wait_seconds"] == 0.0
    finally:
        other.close()


def test_stale_holder_is_reclaimed_mid_wait_without_waiting_out_the_bound(tmp_path, monkeypatch):
    _fast(monkeypatch)
    root = tmp_path / "locks"
    holder = subprocess.Popen(
        [sys.executable, "-c", (
            "import sys, time; sys.path.insert(0, %r); "
            "from coding_review_agent_loop.test_workers import WorkerBudgetLock; "
            "lock, _ = WorkerBudgetLock.acquire(invocation_id='dead', cwd=%r, root=__import__('pathlib').Path(%r)); "
            "lock.reserve_host_workers(4, __import__('coding_review_agent_loop.test_workers', fromlist=['x']).HostCapacity(4, None, 1)); "
            "print('ready', flush=True); time.sleep(60)"
        ) % (str(__import__("pathlib").Path(workers_module.__file__).parents[1]), str(tmp_path), str(root))],
        stdout=subprocess.PIPE, text=True,
    )
    assert holder.stdout.readline().strip() == "ready"
    b = _lock(root, "b")
    try:
        threading.Timer(0.4, holder.kill).start()
        started = time.monotonic()
        r = b.wait_for_host_workers(4, _cpus(4), wait_seconds=30, notify=lambda _t: None)
        assert r.granted == 4 and not r.timed_out
        assert time.monotonic() - started < 10
    finally:
        holder.kill()
        holder.wait()
        b.close()


def _hold_mutex_process(root, seconds):
    code = (
        "import sys, time, fcntl, pathlib; "
        "h = open(pathlib.Path(sys.argv[1]) / 'host-capacity.mutex', 'a+'); "
        "fcntl.flock(h.fileno(), fcntl.LOCK_EX); print('held', flush=True); time.sleep(float(sys.argv[2]))"
    )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-c", code, str(root), str(seconds)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "held"
    return proc


def test_mutex_held_elsewhere_still_honours_the_wait_bound(tmp_path, monkeypatch):
    _fast(monkeypatch)
    root = tmp_path / "locks"
    holder = _hold_mutex_process(root, 30)
    try:
        log, tel = _telemetry(tmp_path)
        started = time.monotonic()
        result = _run(tmp_path, root, tel, env=_env("mine", wait=0.5), cmd=_PRINTER)
        assert time.monotonic() - started < 12
        assert result.returncode == 0 and "workers=1" in result.output_tail
        (row,) = _attempts(log)
        assert row["outcome"] == "oversubscribed" and row["accounting_unavailable"] is True
        assert row["wait_timed_out"] is True
    finally:
        holder.kill()
        holder.wait()
    assert _records(root) == []


def test_bookkeeping_is_bounded_under_a_held_mutex(tmp_path, monkeypatch):
    monkeypatch.setattr(workers_module, "HOST_MUTEX_BOOKKEEPING_GRACE_SECONDS", 0.3)
    root = tmp_path / "locks"
    lock = _lock(root, "a")
    assert lock.reserve_host_workers(2, _cpus(4)) == 2
    holder = _hold_mutex_process(root, 30)
    try:
        started = time.monotonic()
        lock.record_process_group(os.getpid())
        lock.close()
        assert time.monotonic() - started < 3
        assert len(_records(root)) == 1  # skipped release: reclaimed as stale later
    finally:
        holder.kill()
        holder.wait()
    other = _lock(root, "b")
    try:
        assert other.reserve_host_workers(4, _cpus(4), exclusive=True) == 4
    finally:
        other.close()


def test_cancel_during_wait_raises_and_leaves_no_record(tmp_path, monkeypatch):
    _fast(monkeypatch)
    root = tmp_path / "locks"
    a, b = _lock(root, "a"), _lock(root, "b")
    cancel = threading.Event()
    try:
        assert a.reserve_host_workers(4, _cpus(4)) == 4
        threading.Timer(0.3, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(HostWaitCancelled):
            b.wait_for_host_workers(4, _cpus(4), wait_seconds=30, notify=lambda _t: None, cancel=cancel)
        assert time.monotonic() - started < 5
        assert len(_records(root)) == 1
    finally:
        a.close()
        b.close()


def test_cancel_during_mutex_contention_is_serviced(tmp_path):
    root = tmp_path / "locks"
    holder = _hold_mutex_process(root, 30)
    b = _lock(root, "b")
    cancel = threading.Event()
    try:
        threading.Timer(0.3, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(HostWaitCancelled):
            b.wait_for_host_workers(4, _cpus(4), wait_seconds=30, notify=lambda _t: None, cancel=cancel)
        assert time.monotonic() - started < 5
    finally:
        holder.kill()
        holder.wait()
        b.close()


def test_heartbeats_are_sent_during_polling_and_mutex_contention(tmp_path, monkeypatch):
    _fast(monkeypatch)
    monkeypatch.setattr(workers_module, "HOST_WAIT_HEARTBEAT_SECONDS", 0.1)
    root = tmp_path / "locks"
    a, b = _lock(root, "a"), _lock(root, "b")
    beats = []
    try:
        assert a.reserve_host_workers(4, _cpus(4)) == 4
        b.wait_for_host_workers(
            4, _cpus(4), wait_seconds=0.6, notify=lambda _t: None, heartbeat=lambda: beats.append(1),
        )
        assert len(beats) >= 3
    finally:
        a.close()
        b.close()
    holder = _hold_mutex_process(root, 30)
    beats.clear()
    c = _lock(root, "c")
    try:
        c.wait_for_host_workers(
            4, _cpus(4), wait_seconds=0.6, notify=lambda _t: None, heartbeat=lambda: beats.append(1),
        )
        assert len(beats) >= 3
    finally:
        holder.kill()
        holder.wait()
        c.close()


def test_interrupt_mid_wait_records_elapsed_wait_and_cleans_up(tmp_path, monkeypatch):
    _fast(monkeypatch)
    root = tmp_path / "locks"
    other = _lock(root, "other-loop")
    assert other.reserve_host_workers(4, _cpus(4)) == 4
    log, tel = _telemetry(tmp_path)
    calls = {"n": 0}

    class Boom(Exception):
        pass

    class ExplodingEvent(threading.Event):
        def wait(self, timeout=None):
            calls["n"] += 1
            time.sleep(0.2)
            if calls["n"] >= 2:
                raise Boom()
            return False

    try:
        with pytest.raises(Boom):
            _run(tmp_path, root, tel, env=_env("mine", wait=30), host_wait_cancel=ExplodingEvent())
        (row,) = _attempts(log)
        assert row["outcome"] == "error"
        assert row["reservation_wait_seconds"] > 0.1 and row["wait_bound_seconds"] == 30.0
        assert len(_records(root)) == 1  # only the foreign holder
        probe = _lock(root, "mine")  # the per-invocation flock was released
        probe.close()
    finally:
        other.close()


def test_interrupt_during_first_mutex_acquisition_records_wait(tmp_path):
    root = tmp_path / "locks"
    holder = _hold_mutex_process(root, 30)
    log, tel = _telemetry(tmp_path)
    ticks = {"n": 0}

    class Cancel(threading.Event):
        def is_set(self):
            ticks["n"] += 1
            if ticks["n"] > 6:
                raise KeyboardInterrupt()
            return False

    try:
        with pytest.raises(KeyboardInterrupt):
            _run(tmp_path, root, tel, env=_env("mine", wait=30), host_wait_cancel=Cancel())
        (row,) = _attempts(log)
        assert row["outcome"] == "error" and row["reservation_wait_seconds"] > 0
        assert row["wait_bound_seconds"] == 30.0
        _lock(root, "mine").close()
    finally:
        holder.kill()
        holder.wait()
    assert _records(root) == []


def test_mutex_deadline_raises_timeout_and_closes_handle(tmp_path):
    holder = _hold_mutex_process(tmp_path, 30)
    try:
        with pytest.raises(HostCapacityMutexTimeout):
            with _host_capacity_mutex(tmp_path, deadline=time.monotonic() + 0.2):
                pass
    finally:
        holder.kill()
        holder.wait()
    with _host_capacity_mutex(tmp_path, deadline=time.monotonic() + 1):
        pass


def test_launch_guard_and_pre_launch_cancel_never_start_the_target(tmp_path):
    marker = tmp_path / "spawned"
    cancel = threading.Event()
    cancel.set()
    root = tmp_path / "locks"
    log, tel = _telemetry(tmp_path)
    result = _run(
        tmp_path, root, tel, env=_env("mine", "off"),
        cmd=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        host_wait_cancel=cancel,
    )
    assert result.outcome == "cancelled" and result.returncode is None
    assert not marker.exists()
    (row,) = _attempts(log)
    assert row["outcome"] == "cancelled"
