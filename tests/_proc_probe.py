"""Race-tolerant /proc probes and a startup readiness gate for process tests.

The kernel returns ESRCH (``ProcessLookupError``) when a process is reaped
between ``open()`` and ``read()`` of ``/proc/<pid>/stat``, so a liveness probe
that catches only ``FileNotFoundError`` fails under host load (issue #868).
Every helper here treats a missing entry, an unreadable entry, a zombie/dead
state, and PID reuse as "already gone".
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

#: Process states that mean the instance is no longer running.
DEAD_STATES = frozenset({"Z", "X", "x"})

#: Parent-side helpers that publish a ``pid:starttime`` record atomically, so a
#: reader sees either no file or a complete record.  Prepend to a ``-c`` body.
PUBLISH_SNIPPET = (
    "import os as _os, pathlib as _pathlib\n"
    "def _starttime(pid):\n"
    "    _stat = _pathlib.Path('/proc/%d/stat' % pid).read_text(encoding='ascii')\n"
    "    return _stat[_stat.rfind(')') + 2:].split()[19]\n"
    "def _publish(path, pids):\n"
    "    _target = _pathlib.Path(path)\n"
    "    _tmp = _target.with_suffix('.tmp')\n"
    "    _tmp.write_text(','.join('%d:%s' % (pid, _starttime(pid)) for pid in pids),\n"
    "                    encoding='ascii')\n"
    "    _os.replace(_tmp, _target)\n"
)


def _stat_fields(pid: int) -> list[str] | None:
    """Return the stat fields after the command name, or None if unreadable."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        # FileNotFoundError, ProcessLookupError (ESRCH), and any other OSError
        # all mean the caller cannot observe this instance any more.
        return None
    close = stat.rfind(")")
    if close < 0:
        return None
    fields = stat[close + 2 :].split()
    return fields or None


def proc_state(pid: int) -> str | None:
    """Return the process state letter, or None when the entry is unreadable."""
    fields = _stat_fields(pid)
    if fields is None:
        return None
    return fields[0]


def proc_start_time(pid: int) -> str | None:
    """Return field 22 (starttime), or None when the entry is unreadable."""
    fields = _stat_fields(pid)
    if fields is None or len(fields) <= 19:
        return None
    return fields[19]


def _is_gone(pid: int, start_time: str | None) -> bool:
    state = proc_state(pid)
    if state is None or state in DEAD_STATES:
        return True
    if start_time is not None and proc_start_time(pid) != start_time:
        # The PID was reused by an unrelated process instance.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def wait_until_gone(
    pid: int, *, start_time: str | None = None, timeout: float = 10.0, poll: float = 0.01
) -> bool:
    """Poll until ``pid`` is gone; False only if the same instance is still live."""
    deadline = time.monotonic() + timeout
    while True:
        if _is_gone(pid, start_time):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)


def wait_group_gone(pgid: int, *, timeout: float = 10.0, poll: float = 0.01) -> bool:
    """Poll until no process remains in ``pgid``; False at the deadline."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)


def kill_if_same_instance(
    pid: int, start_time: str | None, *, sig: int = signal.SIGKILL
) -> bool:
    """Signal ``pid`` only when its current starttime still matches ``start_time``.

    A liveness or state check alone is not enough: ``proc_state`` says a PID is
    live, not that it is the same process.  The starttime is re-read
    immediately before the signal, so a reused PID is never signalled.
    """
    if start_time is None:
        return False
    current = proc_start_time(pid)
    if current is None or current != start_time:
        return False
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    return True


def read_pid_record(path: Path | str) -> list[tuple[int, str]]:
    """Parse a comma-separated ``pid:starttime`` record into typed entries.

    Both halves must be numeric and there must be exactly one separator.  A
    non-numeric start time would never equal a real /proc starttime, so a
    caller comparing it would treat a live descendant as PID-reused and report
    a survivor as already gone.
    """
    raw = Path(path).read_text(encoding="ascii")
    entries: list[tuple[int, str]] = []
    for chunk in raw.split(","):
        parts = [part.strip() for part in chunk.strip().split(":")]
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise AssertionError(f"malformed pid:starttime record {raw!r} at {path}")
        entries.append((int(parts[0]), parts[1]))
    if not entries:
        raise AssertionError(f"malformed pid:starttime record {raw!r} at {path}")
    return entries


def terminate_process_tree(
    proc, *, term_grace: float = 2.0, kill_grace: float = 2.0
) -> None:
    """Bounded TERM/KILL escalation driven by survival of the whole group.

    ``runner._terminate_process_group`` escalates on the direct child alone: if
    the parent exits on SIGTERM while a descendant ignores it, the group is
    never SIGKILLed and the descendant is orphaned.  Escalation here is decided
    by whether any member of the group is still alive, after the direct child
    has been reaped so its own zombie cannot masquerade as a survivor.
    """
    pgid = proc.pid
    _signal_group(pgid, signal.SIGTERM)
    reaped = True
    try:
        proc.wait(timeout=term_grace)
    except subprocess.TimeoutExpired:
        # A leader still holding the group is itself proof the group survived.
        reaped = False
    if not reaped or not wait_group_gone(pgid, timeout=term_grace):
        _signal_group(pgid, signal.SIGKILL)
        wait_group_gone(pgid, timeout=kill_grace)
    try:
        proc.wait(timeout=kill_grace)
    except subprocess.TimeoutExpired:
        pass


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class ReadinessGate:
    """``process_started`` hook that holds the watchdog until startup published.

    ``run_foreground_test`` computes its watchdog deadline before launch, so a
    slow start under host load could otherwise be terminated before the child
    published the PIDs the test needs.  The gate blocks in the callback until
    the record exists, the process exits, or its own cap expires.  It never
    raises, and on cap expiry it tears the tree down itself rather than leaving
    an orphan to an already-expired watchdog.
    """

    def __init__(self, path, *, cap: float = 30.0, poll: float = 0.02, cleanup=None):
        self.path = Path(path)
        self.cap = cap
        self.poll = poll
        self._cleanup = cleanup if cleanup is not None else terminate_process_tree
        self.pid: int | None = None
        self.start_time: str | None = None
        self.ready = False
        self.release_reason: str | None = None
        self.failure_reason: str | None = None
        self.waited_seconds = 0.0
        self.cleanup_invocations = 0

    def __call__(self, proc) -> None:
        # Capture the launched identity first: under start_new_session the pid
        # is also the process-group id, so identity capture never depends on
        # the cap or on a file the parent must write.
        self.pid = proc.pid
        self.start_time = proc_start_time(proc.pid)
        started = time.monotonic()
        deadline = started + self.cap
        while True:
            if self.path.exists():
                self.ready = True
                self.release_reason = "published"
                break
            if proc.poll() is not None:
                # A parent that already exited cannot be killed prematurely;
                # the call site's own record assertion reports the real problem.
                self.ready = True
                self.release_reason = "process-exited"
                break
            if time.monotonic() >= deadline:
                self.ready = False
                self.release_reason = "cap-expired"
                self.failure_reason = (
                    f"no pid:starttime record at {self.path} within the "
                    f"{self.cap:g}s readiness cap"
                )
                self._run_cleanup(proc)
                break
            time.sleep(self.poll)
        self.waited_seconds = time.monotonic() - started

    def _run_cleanup(self, proc) -> None:
        self.cleanup_invocations += 1
        try:
            self._cleanup(proc)
        except BaseException as exc:
            self.failure_reason = f"{self.failure_reason}; cleanup raised {exc!r}"

    def assert_ready(self) -> None:
        if not self.ready:
            pytest.fail(f"process startup/readiness failed: {self.failure_reason}")


def readiness_gate(path, *, cap: float = 30.0, poll: float = 0.02, cleanup=None) -> ReadinessGate:
    return ReadinessGate(path, cap=cap, poll=poll, cleanup=cleanup)
