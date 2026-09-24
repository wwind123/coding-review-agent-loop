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
    HOST_RESERVATION_SUFFIX,
    WorkerBudget,
    WorkerBudgetLock,
    host_sharing_enabled,
    host_worker_pool,
)

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
        assert lock.reserve_host_workers(16, 8) == 16
    finally:
        lock.close()
    assert _records(root) == []


def test_contending_invocations_never_exceed_the_pool(tmp_path):
    root = tmp_path / "locks"
    locks = [_lock(root, f"inv-{index}") for index in range(4)]
    try:
        grants = [lock.reserve_host_workers(5, 8) for lock in locks]
        assert grants == [5, 3, 0, 0]
        assert sum(grants) <= 8
        # Releasing one holder frees its share for the next.
        locks[0].close()
        assert locks[2].reserve_host_workers(5, 8) == 5
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
        "from coding_review_agent_loop.test_workers import WorkerBudgetLock\n"
        "root = Path(sys.argv[1])\n"
        "lock, _ = WorkerBudgetLock.acquire(invocation_id='victim', cwd=root, root=root)\n"
        "lock.reserve_host_workers(8, 8)\n"
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
            assert other.reserve_host_workers(4, 8) == 0  # the victim holds the pool
            child.kill()
            child.wait(timeout=10)
            assert len(_records(root)) == 1  # leaked by the kill ...
            assert other.reserve_host_workers(4, 8) == 4  # ... and reclaimed
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
    assert holder.reserve_host_workers(8, 8) == 8
    assert holder.hold_until_group_exits(member.pid) == "watcher"
    other = _lock(root, "other")
    try:
        # The holder's process has let go, but its group still has a live
        # member: the reservation must still count.
        assert other.reserve_host_workers(2, 8) == 0
        member.wait(timeout=10)
        deadline = time.monotonic() + 10
        granted = 0
        while time.monotonic() < deadline and not granted:
            granted = other.reserve_host_workers(2, 8)
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


def test_host_sharing_opt_out():
    assert host_sharing_enabled({}) is True
    assert host_sharing_enabled({ENV_HOST_SHARING: "on"}) is True
    for value in ("off", "0", "false", "No"):
        assert host_sharing_enabled({ENV_HOST_SHARING: value}) is False


def _budget(workers):
    return WorkerBudget(workers, "inherited", "clamp", "cpu", {"cpu_available": 4}, False)


def _env(invocation, sharing="on"):
    return {**os.environ, "AGENT_LOOP_INVOCATION_ID": invocation, ENV_HOST_SHARING: sharing}


def test_run_foreground_test_shrinks_or_busies_on_shared_pool(tmp_path):
    from coding_review_agent_loop.runner import run_foreground_test

    root = tmp_path / "locks"
    other = _lock(root, "other-loop")
    printer = [sys.executable, "-c", "import os; print('workers=' + os.environ['AGENT_LOOP_TEST_WORKERS'])"]
    try:
        assert other.reserve_host_workers(3, 4) == 3
        shrunk = run_foreground_test(
            printer, cwd=tmp_path, timeout_seconds=30, env=_env("mine"), environment_is_complete=True,
            echo_output=False, worker_budget=_budget(4), worker_lock_root=root,
        )
        assert shrunk.returncode == 0
        assert "workers=1" in shrunk.output_tail
        assert other.reserve_host_workers(4, 4) == 4  # the pool is all "other-loop" again
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
