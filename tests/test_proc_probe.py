"""Unit tests for the race-tolerant /proc probe helpers (issue #868)."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from _pytest.outcomes import Failed

import _proc_probe
from _proc_probe import (
    PUBLISH_SNIPPET,
    ReadinessGate,
    kill_if_same_instance,
    terminate_process_tree,
    proc_start_time,
    proc_state,
    read_pid_record,
    readiness_gate,
    wait_group_gone,
    wait_until_gone,
)


class _StubProc:
    def __init__(self, pid=None, exit_code=None):
        self.pid = pid if pid is not None else os.getpid()
        self._exit_code = exit_code

    def poll(self):
        return self._exit_code


@pytest.mark.parametrize(
    "error",
    [ProcessLookupError(3, "No such process"), FileNotFoundError(), OSError("boom")],
)
def test_unreadable_proc_entry_reads_as_gone(monkeypatch, error):
    def read_text(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(_proc_probe.Path, "read_text", read_text)
    assert proc_state(4242) is None
    assert proc_start_time(4242) is None
    assert wait_until_gone(4242, timeout=0.0) is True


@pytest.mark.parametrize("state", ["Z", "X", "x"])
def test_zombie_and_dead_states_count_as_gone(monkeypatch, state):
    fields = " ".join(str(index) for index in range(30))
    monkeypatch.setattr(
        _proc_probe.Path, "read_text", lambda self, *a, **k: f"77 (proc) {state} {fields}"
    )
    assert proc_state(77) == state
    assert wait_until_gone(77, timeout=0.0) is True


def test_pid_reuse_counts_as_gone(monkeypatch):
    fields = " ".join(str(index) for index in range(30))
    monkeypatch.setattr(
        _proc_probe.Path, "read_text", lambda self, *a, **k: f"1 (proc) S {fields}"
    )
    # starttime is field 22 of stat, the 20th field after the state letter.
    pid = os.getpid()
    assert proc_start_time(pid) == "18"
    # The same live instance is not gone; a different starttime is PID reuse.
    assert wait_until_gone(pid, start_time="18", timeout=0.0) is False
    assert wait_until_gone(pid, start_time="other", timeout=0.0) is True


def test_exited_real_child_is_gone():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert wait_until_gone(child.pid, timeout=5.0) is True


def test_live_real_child_is_not_gone():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        start_time = proc_start_time(child.pid)
        assert start_time is not None
        assert wait_until_gone(child.pid, start_time=start_time, timeout=0.2) is False
    finally:
        child.kill()
        child.wait()


def test_read_pid_record_parses_and_rejects_malformed(tmp_path):
    good = tmp_path / "good"
    good.write_text("12:34,56:78", encoding="ascii")
    assert read_pid_record(good) == [(12, "34"), (56, "78")]

    for contents in [
        "123:",
        "12,",
        "",
        "abc:12",
        ":12",
        "12:not-a-starttime",
        "12:34:56",
        "12:3 4",
        "12:34,56",
    ]:
        bad = tmp_path / "bad"
        bad.write_text(contents, encoding="ascii")
        with pytest.raises(AssertionError, match="malformed pid:starttime record"):
            read_pid_record(bad)


def test_publish_snippet_is_atomic(tmp_path):
    target = tmp_path / "record.pid"
    code = PUBLISH_SNIPPET + "import os, sys\n_publish(sys.argv[1], [os.getpid()])\n"
    completed = subprocess.run(
        [sys.executable, "-c", code, str(target)], check=False, timeout=30
    )
    assert completed.returncode == 0
    assert list(tmp_path.glob("*.tmp")) == []
    assert len(read_pid_record(target)) == 1


def test_gate_releases_once_the_record_is_published(tmp_path):
    path = tmp_path / "record.pid"
    cleanups = []
    gate = readiness_gate(path, cap=10.0, poll=0.01, cleanup=cleanups.append)
    timer = threading.Timer(0.2, lambda: path.write_text("1:2", encoding="ascii"))
    timer.start()
    try:
        gate(_StubProc())
    finally:
        timer.cancel()
    assert gate.ready is True
    assert gate.release_reason == "published"
    assert cleanups == []
    gate.assert_ready()


def test_gate_releases_when_the_process_already_exited(tmp_path):
    cleanups = []
    gate = readiness_gate(tmp_path / "missing.pid", cap=10.0, cleanup=cleanups.append)
    gate(_StubProc(exit_code=1))
    assert gate.ready is True
    assert gate.release_reason == "process-exited"
    assert cleanups == []
    gate.assert_ready()


def test_gate_cap_expiry_cleans_up_once_and_reports_failure(tmp_path):
    path = tmp_path / "missing.pid"
    cleanups = []
    gate = readiness_gate(path, cap=0.05, poll=0.01, cleanup=cleanups.append)
    proc = _StubProc()
    gate(proc)
    assert gate.ready is False
    assert gate.release_reason == "cap-expired"
    assert gate.cleanup_invocations == 1
    assert cleanups == [proc]
    assert str(path) in gate.failure_reason
    assert "0.05s readiness cap" in gate.failure_reason
    with pytest.raises(Failed, match="startup/readiness failed"):
        gate.assert_ready()


def test_gate_captures_identity_before_polling(tmp_path):
    captured = {}

    def cleanup(proc):
        captured["pid"] = gate.pid
        captured["start_time"] = gate.start_time

    gate = ReadinessGate(tmp_path / "missing.pid", cap=0.0, cleanup=cleanup)
    gate(_StubProc())
    assert gate.pid == os.getpid()
    assert gate.start_time == proc_start_time(os.getpid())
    assert captured == {"pid": gate.pid, "start_time": gate.start_time}


def test_gate_tolerates_a_raising_cleanup(tmp_path):
    def cleanup(_proc):
        raise RuntimeError("cleanup exploded")

    gate = readiness_gate(tmp_path / "missing.pid", cap=0.05, poll=0.01, cleanup=cleanup)
    gate(_StubProc())
    assert gate.ready is False
    assert gate.release_reason == "cap-expired"
    assert "cleanup exploded" in gate.failure_reason


def test_wait_group_gone_exits_early_on_process_lookup_error(monkeypatch):
    def killpg(_pgid, _sig):
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(_proc_probe.os, "killpg", killpg)
    assert wait_group_gone(4242, timeout=5.0) is True


def test_wait_group_gone_reports_a_live_group_and_a_reaped_one():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        assert wait_group_gone(child.pid, timeout=0.2) is False
    finally:
        child.kill()
        child.wait()
    assert wait_group_gone(child.pid, timeout=5.0) is True


def test_kill_if_same_instance_signals_only_a_matching_identity(monkeypatch):
    signals = []
    monkeypatch.setattr(
        _proc_probe.os, "kill", lambda pid, sig: signals.append((pid, sig))
    )
    monkeypatch.setattr(_proc_probe, "proc_start_time", lambda _pid: "1234")

    assert kill_if_same_instance(99, "1234") is True
    assert len(signals) == 1
    assert kill_if_same_instance(99, "5678") is False
    assert kill_if_same_instance(99, None) is False
    monkeypatch.setattr(_proc_probe, "proc_start_time", lambda _pid: None)
    assert kill_if_same_instance(99, "1234") is False
    assert len(signals) == 1


def test_kill_if_same_instance_tolerates_process_lookup_error(monkeypatch):
    monkeypatch.setattr(_proc_probe, "proc_start_time", lambda _pid: "1234")

    def kill(_pid, _sig):
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(_proc_probe.os, "kill", kill)
    assert kill_if_same_instance(99, "1234") is False


def test_kill_if_same_instance_kills_a_real_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        start_time = proc_start_time(child.pid)
        assert start_time is not None
        assert kill_if_same_instance(child.pid, start_time) is True
        assert wait_until_gone(child.pid, start_time=start_time, timeout=10.0) is True
    finally:
        child.kill()
        child.wait()


# A process that ignores SIGTERM only does so once its handler is installed, so
# every escalation test waits for the process to publish a readiness marker
# before signalling; otherwise SIGTERM would win the startup race and the test
# would never exercise the SIGKILL escalation it is meant to prove.
_IGNORE_TERM_AND_WAIT = (
    "import signal, sys, time; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "open(sys.argv[1], 'w').close(); "
    "time.sleep(60)"
)


def _wait_for_marker(path, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), f"{path} was never published"


def test_terminate_process_tree_escalates_for_a_sigterm_ignoring_leader(tmp_path):
    """Escalation follows group survival, not just the direct child's exit."""
    ready = tmp_path / "leader.ready"
    proc = subprocess.Popen(
        [sys.executable, "-c", _IGNORE_TERM_AND_WAIT, str(ready)], start_new_session=True
    )
    start_time = proc_start_time(proc.pid)
    try:
        _wait_for_marker(ready)
        terminate_process_tree(proc)
        assert wait_until_gone(proc.pid, start_time=start_time, timeout=10.0) is True
    finally:
        kill_if_same_instance(proc.pid, start_time)
        proc.wait()


def test_terminate_process_tree_kills_a_sigterm_ignoring_descendant(tmp_path):
    """A descendant that ignores SIGTERM dies even when the leader obeys it."""
    observed = tmp_path / "descendant.pid"
    ready = tmp_path / "descendant.ready"
    code = (
        PUBLISH_SNIPPET
        + "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {_IGNORE_TERM_AND_WAIT!r}, sys.argv[2]])\n"
        "import pathlib\n"
        "while not pathlib.Path(sys.argv[2]).exists():\n"
        "    time.sleep(0.01)\n"
        "_publish(sys.argv[1], [child.pid])\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(observed), str(ready)], start_new_session=True
    )
    descendants = []
    try:
        _wait_for_marker(observed)
        descendants = read_pid_record(observed)
        terminate_process_tree(proc)
        for child_pid, child_start in descendants:
            assert wait_until_gone(child_pid, start_time=child_start, timeout=10.0) is True
    finally:
        for child_pid, child_start in descendants:
            kill_if_same_instance(child_pid, child_start)
        proc.kill()
        proc.wait()
