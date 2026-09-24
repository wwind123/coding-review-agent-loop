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
