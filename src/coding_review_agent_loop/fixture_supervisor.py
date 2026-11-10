"""Linux child-subreaper supervisor for ``run-tests --fixture`` (issue #1326).

The broker runs this file as ``sys.executable -I <this file> ...``.  It is
deliberately standard-library only: ``-I`` ignores ``PYTHONPATH`` and the
user site, so the supervisor never imports code from the checkout under test.

Protocol (one JSON object per line on stdout, commands on stdin):

* the supervisor waits for ``start`` (the broker first attaches it to the
  coder cgroup when one exists), then runs ``<fixture> setup``;
* ``inventory`` lines name every newly seen descendant ``(pid, start)``;
* a ``setup`` line reports the setup status, duration and bounded log tail;
* ``teardown`` (or stdin EOF) runs ``<fixture> teardown``; the same command
  during setup interrupts setup first;
* a ``teardown`` line, then the sweep and the run-scoped census (issue
  #1995), then one final ``cleanup`` line, which also lists processes proven
  to be in an operator-declared foreign scope (diagnostic only).

The authority proof and the census are also imported by the broker process
(``fixture_runtime``) so both use one implementation.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import select
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:  # Linux-only; importing this module must still work elsewhere (#1326).
    import fcntl
except ImportError:  # pragma: no cover - exercised by a simulated import
    fcntl = None  # type: ignore[assignment]

# The kernel's fixed inode for the initial PID namespace (PROC_PID_INIT_INO).
PROC_PID_INIT_INO = 0xEFFFFFFC
# _IO(0xb7, 0x2): NS_GET_PARENT.
NS_GET_PARENT = 0xB702
PR_SET_CHILD_SUBREAPER = 36
PR_SET_NO_NEW_PRIVS = 38
PF_KTHREAD = 0x00200000
KTHREADD_PID = 2

LOG_TAIL_BYTES = 8 * 1024
POLL_SECONDS = 0.1
MAX_ESCAPED_REPORTED = 16
MAX_CENSUS_ENTRIES = 1_000_000
CENSUS_SECONDS = 3.0
TERM_GRACE_SECONDS = 1.0
# Reads per drain call: continuous output can never starve the deadline,
# teardown-command or inventory checks of a phase loop.
MAX_DRAIN_READS = 4
MIN_TEARDOWN_SECONDS = 1.0
MIN_SWEEP_SECONDS = 1.0


class HostAuthorityError(Exception):
    """Host-wide process visibility cannot be proven."""


@dataclass(frozen=True)
class HostAuthority:
    mount_id: str


@dataclass(frozen=True)
class AuthorityProbes:
    """Injectable system calls for the host-process authority proof."""

    stat: Callable[[str], os.stat_result] = os.stat
    ns_get_parent: Callable[[str], None] | None = None
    read_text: Callable[[str], str] = lambda path: Path(path).read_text(
        encoding="utf-8", errors="replace"
    )
    getpid: Callable[[], int] = os.getpid
    subreaper_available: Callable[[], bool] | None = None


def _ns_get_parent(path: str) -> None:
    if fcntl is None:
        raise OSError(errno.ENOSYS, "ioctl is unavailable")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        fcntl.ioctl(fd, NS_GET_PARENT)
    finally:
        os.close(fd)


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _subreaper_available() -> bool:
    try:
        libc = _libc()
        value = ctypes.c_int(0)
        # PR_GET_CHILD_SUBREAPER (37) is side-effect free.
        return libc.prctl(37, ctypes.byref(value), 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False


def _proc_mount(mountinfo: str) -> tuple[str, str, str] | None:
    """Return ``(mount_id, fstype, super_options)`` of the ``/proc`` mount."""
    found: tuple[str, str, str] | None = None
    for line in mountinfo.splitlines():
        left, sep, right = line.partition(" - ")
        if not sep:
            continue
        fields = left.split()
        tail = right.split()
        if len(fields) < 5 or len(tail) < 3:
            continue
        if fields[4] == "/proc":
            # The last mount over /proc is the visible one.
            found = (fields[0], tail[0], tail[2])
    return found


def verify_host_process_authority(
    probes: AuthorityProbes | None = None, *, expected_mount_id: str | None = None,
) -> HostAuthority:
    """Prove that the caller and its ``/proc`` see the whole host process table.

    Five independent checks, all of which an unprivileged process can make.
    ``/proc/1/ns`` is never read: it is ptrace-gated for non-root callers.
    """
    probes = probes or AuthorityProbes()
    if not sys.platform.startswith("linux"):
        raise HostAuthorityError("fixture supervision requires Linux")
    # 1. The caller is in the initial PID namespace.
    try:
        pid_ns = probes.stat("/proc/self/ns/pid")
        net_ns = probes.stat("/proc/self/ns/net")
    except OSError as exc:
        raise HostAuthorityError(f"PID namespace is not inspectable: {type(exc).__name__}") from exc
    if pid_ns.st_ino != PROC_PID_INIT_INO:
        raise HostAuthorityError("caller is not in the initial PID namespace")
    if pid_ns.st_dev != net_ns.st_dev:
        raise HostAuthorityError("PID namespace link is not on the nsfs device")
    # 2. No ancestor namespace exists or is visible.
    get_parent = probes.ns_get_parent or _ns_get_parent
    try:
        get_parent("/proc/self/ns/pid")
    except OSError as exc:
        if exc.errno != errno.EPERM:
            raise HostAuthorityError(
                f"NS_GET_PARENT failed with {errno.errorcode.get(exc.errno or 0, exc.errno)}"
            ) from exc
    else:
        raise HostAuthorityError("PID namespace has a visible parent")
    # 3. The mounted procfs belongs to the caller's (initial) namespace.
    try:
        stat_text = probes.read_text("/proc/self/stat")
    except OSError as exc:
        raise HostAuthorityError("/proc/self does not resolve") from exc
    try:
        self_pid = int(stat_text.split(" ", 1)[0])
    except ValueError as exc:
        raise HostAuthorityError("/proc/self/stat is unreadable") from exc
    if self_pid != probes.getpid():
        raise HostAuthorityError("/proc/self resolves to a different PID")
    # 4. /proc is procfs and hides nothing.
    try:
        mount = _proc_mount(probes.read_text("/proc/self/mountinfo"))
    except OSError as exc:
        raise HostAuthorityError("/proc/self/mountinfo is unreadable") from exc
    if mount is None or mount[1] != "proc":
        raise HostAuthorityError("/proc is not a procfs mount")
    options = mount[2].split(",")
    if any(item.startswith("hidepid") and item not in {"hidepid=0", "hidepid=off"} for item in options):
        raise HostAuthorityError("/proc is mounted with hidepid")
    if any(item.startswith("subset=") for item in options):
        raise HostAuthorityError("/proc is mounted with subset=pid")
    if expected_mount_id is not None and mount[0] != expected_mount_id:
        raise HostAuthorityError("/proc mount changed since launch")
    # 5. The subreaper facility exists and PID 1 is visible.
    available = probes.subreaper_available or _subreaper_available
    if not available():
        raise HostAuthorityError("PR_SET_CHILD_SUBREAPER is unavailable")
    try:
        probes.read_text("/proc/1/stat")
    except OSError as exc:
        raise HostAuthorityError("/proc/1/stat is not readable") from exc
    return HostAuthority(mount_id=mount[0])


# Process table ---------------------------------------------------------------


@dataclass(frozen=True)
class ProcEntry:
    pid: int
    ppid: int
    start: int
    state: str
    flags: int
    comm: str


def parse_stat(text: str) -> ProcEntry | None:
    left = text.find("(")
    right = text.rfind(")")
    if left < 0 or right < left:
        return None
    try:
        pid = int(text[:left].strip())
        fields = text[right + 2:].split()
        return ProcEntry(
            pid=pid,
            ppid=int(fields[1]),
            start=int(fields[19]),
            state=fields[0],
            flags=int(fields[6]),
            comm=text[left + 1:right][:32],
        )
    except (ValueError, IndexError):
        return None


def read_entry(pid: int, proc_root: Path = Path("/proc")) -> ProcEntry | None:
    try:
        return parse_stat((proc_root / str(pid) / "stat").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None


def process_table(proc_root: Path = Path("/proc")) -> dict[int, ProcEntry]:
    table: dict[int, ProcEntry] = {}
    try:
        names = os.listdir(proc_root)
    except OSError:
        return table
    for name in names:
        if name.isdigit():
            entry = read_entry(int(name), proc_root)
            if entry is not None:
                table[entry.pid] = entry
    return table


def descendants(root_pid: int, table: dict[int, ProcEntry]) -> list[ProcEntry]:
    children: dict[int, list[ProcEntry]] = {}
    for entry in table.values():
        children.setdefault(entry.ppid, []).append(entry)
    found: list[ProcEntry] = []
    pending = [root_pid]
    while pending:
        for child in children.get(pending.pop(), ()):
            found.append(child)
            pending.append(child.pid)
    return found


def is_kernel_thread(entry: ProcEntry) -> bool:
    return bool(entry.flags & PF_KTHREAD) or entry.pid == KTHREADD_PID or entry.ppid == KTHREADD_PID


@dataclass
class CensusResult:
    cleanup: str
    escaped: list[dict[str, object]] = field(default_factory=list)
    reason: str = ""
    # Issue #1995: provably foreign processes (diagnostic only).
    foreign: list[dict[str, object]] = field(default_factory=list)


def _uid_from_status(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("Uid:"):
            parts = line.split()
            if len(parts) > 1 and parts[1].isdigit():
                return int(parts[1])
    return None


# Exited processes awaiting reaping (zombie or dead), as in the stat state field.
_DEAD_STATES = frozenset({"Z", "X", "x"})


def _status_field(text: str, key: str) -> str | None:
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == key:
            return parts[1] if len(parts) > 1 else None
    return None


def _status_state(text: str) -> str | None:
    # The line reads `State:\t<char> (<description>)`; only that one character is the state.
    value = _status_field(text, "State:")
    return value if value is not None and len(value) == 1 else None


def _is_reapable_dead(entry: ProcEntry, status_text: str) -> bool:
    """True only for a single-thread entry that stat and status both call dead.

    A thread-group leader that exited while other threads still run shows
    as a zombie in stat, but status then reports more than one thread, so it
    stays counted.  Any missing, unparsable or disagreeing field also keeps
    the entry counted.
    """
    if entry.state not in _DEAD_STATES:
        return False
    if _status_state(status_text) != entry.state:
        return False
    threads = _status_field(status_text, "Threads:")
    return threads is not None and threads.isdigit() and int(threads) == 1


def _cgroup(pid: int, proc_root: Path) -> str:
    try:
        return (proc_root / str(pid) / "cgroup").read_text(encoding="utf-8", errors="replace").strip()[:256]
    except OSError:
        return ""


# Run-scoped attribution (issue #1995) ----------------------------------------

MAX_ANCESTRY_DEPTH = 4096
MAX_DIAGNOSTIC_CHARS = 400
MAX_FOREIGN_REPORTED = MAX_ESCAPED_REPORTED
ATTRIBUTION_RUN = "run"
ATTRIBUTION_UNDETERMINED = "undetermined"


def unified_cgroup_path(text: str) -> str | None:
    """The cgroup-v2 path of a ``/proc/<pid>/cgroup`` text, or ``None``.

    ``None`` when there is no ``0::`` line, or the path is empty, relative or
    has a ``..`` component (outside the reader's cgroup namespace).
    """
    for line in text.splitlines():
        if line.startswith("0::"):
            path = line[3:].strip()
            if not path.startswith("/") or ".." in path.split("/") or "\x00" in path:
                return None
            return path
    return None


def read_unified_cgroup(pid: int | str, proc_root: Path = Path("/proc")) -> str | None:
    try:
        text = (proc_root / str(pid) / "cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return unified_cgroup_path(text)


def cgroup_within(path: str, root: str) -> bool:
    """Whether ``path`` is ``root`` or below it."""
    if path == root:
        return True
    prefix = root.rstrip("/") + "/"
    return path.startswith(prefix)


def _strict_ancestors(path: str) -> list[str]:
    parts = [part for part in path.split("/") if part]
    return ["/" + "/".join(parts[:index]) for index in range(len(parts))]


def effective_foreign_scopes(
    patterns: Iterable[str], run_cgroup: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(kept patterns, warnings)`` for operator-declared foreign scopes.

    A pattern matching a strict ancestor of the run cgroup covers the whole
    subtree that contains the run, so it can never be a sibling-only
    declaration and is discarded.  A pattern matching the run cgroup itself
    is kept: run attribution always takes precedence, so the run cgroup
    stays attributed.  Without a known run cgroup nothing is kept.
    """
    if run_cgroup is None:
        return (), ()
    import fnmatch

    kept: list[str] = []
    warnings: list[str] = []
    ancestors = _strict_ancestors(run_cgroup)
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern.startswith("/"):
            warnings.append(f"foreign scope {pattern!r} discarded: not an absolute cgroup path pattern")
            continue
        if any(fnmatch.fnmatchcase(ancestor, pattern) for ancestor in ancestors):
            warnings.append(
                f"foreign scope {pattern!r} discarded: it matches an ancestor of the run cgroup {run_cgroup}"
            )
            continue
        if fnmatch.fnmatchcase(run_cgroup, pattern):
            warnings.append(
                f"foreign scope {pattern!r} also matches the run cgroup {run_cgroup}; "
                "processes in the run cgroup stay attributed to the run"
            )
        kept.append(pattern)
    return tuple(dict.fromkeys(kept)), tuple(warnings)


class _CensusExpired(Exception):
    pass


def _classify_candidate(
    entry: ProcEntry,
    *,
    table: dict[int, ProcEntry],
    window_start: int,
    roots: set[tuple[int, int]],
    run_cgroup: str | None,
    foreign_scopes: tuple[str, ...],
    proc_root: Path,
    check: Callable[[], None],
) -> tuple[str, str | None]:
    """Return ``(attribution, cgroup path)``; attribution is run, undetermined or foreign.

    ``check`` raises ``_CensusExpired`` once the census deadline has passed.
    """
    import fnmatch

    check()
    cgroup = read_unified_cgroup(entry.pid, proc_root) if run_cgroup is not None else None
    if cgroup is not None and run_cgroup is not None and cgroup_within(cgroup, run_cgroup):
        return ATTRIBUTION_RUN, cgroup
    # (a) Ancestry through a run root, matched by exact (pid, start) identity.
    seen = {entry.pid}
    current = entry
    for _ in range(MAX_ANCESTRY_DEPTH):
        check()
        parent = table.get(current.ppid)
        if parent is None or parent.pid in seen:
            break
        if (parent.pid, parent.start) in roots:
            return ATTRIBUTION_RUN, cgroup
        seen.add(parent.pid)
        current = parent
    # (b) Everything else stays undetermined unless (c) a bounded,
    # self-consistent structural proof places it in a declared foreign scope.
    if run_cgroup is None or not foreign_scopes or cgroup is None:
        return ATTRIBUTION_UNDETERMINED, cgroup
    if not any(fnmatch.fnmatchcase(cgroup, pattern) for pattern in foreign_scopes):
        return ATTRIBUTION_UNDETERMINED, cgroup
    chain: list[ProcEntry] = [entry]
    seen = {entry.pid}
    current = entry
    anchor: ProcEntry | None = None
    for _ in range(MAX_ANCESTRY_DEPTH):
        check()
        if current.ppid <= 0:
            return ATTRIBUTION_UNDETERMINED, cgroup
        parent = table.get(current.ppid)
        if parent is None or parent.pid in seen:
            return ATTRIBUTION_UNDETERMINED, cgroup
        if parent.start > current.start:
            # Impossible order: the parent PID was recycled.
            return ATTRIBUTION_UNDETERMINED, cgroup
        if parent.pid == 1 or parent.ppid <= 0:
            # Reparented to init, or delegated to a unit/shim: origin unknown.
            return ATTRIBUTION_UNDETERMINED, cgroup
        check()
        if read_unified_cgroup(parent.pid, proc_root) != cgroup:
            return ATTRIBUTION_UNDETERMINED, cgroup
        chain.append(parent)
        seen.add(parent.pid)
        if parent.start < window_start:
            anchor = parent
            break
        current = parent
    if anchor is None:
        return ATTRIBUTION_UNDETERMINED, cgroup
    # Re-verify every identity of the proof after its cgroup was read.
    for item in chain:
        check()
        again = read_entry(item.pid, proc_root)
        if again is None or (again.pid, again.start) != (item.pid, item.start) or again.ppid != item.ppid:
            return ATTRIBUTION_UNDETERMINED, cgroup
    return "foreign", cgroup


def census(
    *,
    window_start: int,
    owned: Iterable[tuple[int, int]],
    self_identity: tuple[int, int],
    proc_root: Path = Path("/proc"),
    deadline: float | None = None,
    run_cgroup: str | None = None,
    run_roots: Iterable[tuple[int, int]] = (),
    foreign_scopes: Sequence[str] = (),
) -> CensusResult:
    """Flag every live non-kernel process started in the window that the run could own.

    The exclusions are the census process's own exact (pid, start) identity,
    the owned inventory, and entries that have already exited and only await
    reaping (a single-thread entry that both stat and status report as
    zombie or dead), which are not live processes.  A recycled PID with a
    different start time is still flagged.

    Issue #1995: every other in-window process is classified.  It is ``run``
    when it sits in or below ``run_cgroup`` or descends from the supervisor,
    a ``run_roots`` identity or an owned identity, and ``undetermined`` unless
    a bounded, self-consistent structural proof places it in an
    operator-declared ``foreign_scopes`` cgroup.  Only provably foreign
    processes are left out of ``escaped`` (they are listed in ``foreign``).
    Without ``run_cgroup`` the census stays host-wide: nothing is foreign.
    """
    owned_set = set(owned)
    try:
        names = os.listdir(proc_root)
    except OSError:
        return CensusResult("unconfirmed", reason="process table is unreadable")
    if len(names) > MAX_CENSUS_ENTRIES:
        return CensusResult("unconfirmed", reason="process table exceeds the census bound")
    bound_reason = "census exceeded its bound"
    table: dict[int, ProcEntry] = {}
    candidates: list[tuple[ProcEntry, int | None]] = []
    for name in names:
        if deadline is not None and time.monotonic() > deadline:
            return CensusResult("unconfirmed", reason=bound_reason)
        if not name.isdigit():
            continue
        path = proc_root / name / "stat"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            continue  # exited mid-scan
        except ProcessLookupError:
            continue
        except OSError:
            if not (proc_root / name).exists():
                continue
            return CensusResult("unconfirmed", reason=f"/proc entry {name} is unreadable")
        entry = parse_stat(text)
        if entry is None:
            if not (proc_root / name).exists():
                continue
            return CensusResult("unconfirmed", reason=f"/proc entry {name} is unparsable")
        # Both stat and status are read for every process: an unreadable
        # status of a live entry means the census cannot vouch for it.
        try:
            status_text = (proc_root / name / "status").read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, ProcessLookupError):
            continue  # exited mid-scan
        except OSError:
            if not (proc_root / name).exists():
                continue
            return CensusResult("unconfirmed", reason=f"/proc entry {name} status is unreadable")
        table[entry.pid] = entry
        if is_kernel_thread(entry):
            continue
        if _is_reapable_dead(entry, status_text):
            continue
        if entry.start < window_start:
            continue
        identity = (entry.pid, entry.start)
        if identity == tuple(self_identity) or identity in owned_set:
            continue
        candidates.append((entry, _uid_from_status(status_text)))
    scopes, _warnings = effective_foreign_scopes(foreign_scopes, run_cgroup)
    roots = {tuple(self_identity), *owned_set, *(tuple(item) for item in run_roots)}

    def check() -> None:
        if deadline is not None and time.monotonic() > deadline:
            raise _CensusExpired()

    escaped: list[dict[str, object]] = []
    foreign: list[dict[str, object]] = []
    try:
        for entry, uid in candidates:
            attribution, _path = _classify_candidate(
                entry, table=table, window_start=window_start, roots=roots,  # type: ignore[arg-type]
                run_cgroup=run_cgroup, foreign_scopes=scopes, proc_root=proc_root, check=check,
            )
            check()
            item: dict[str, object] = {
                "pid": entry.pid,
                "uid": uid,
                "comm": entry.comm,
                "cgroup": _cgroup(entry.pid, proc_root),
            }
            # The diagnostic cgroup read is a /proc read too: never let it
            # cross the deadline unnoticed, including for the last candidate.
            check()
            if attribution == "foreign":
                foreign.append(item)
            else:
                item["attribution"] = attribution
                escaped.append(item)
    except _CensusExpired:
        # Never a partial classification.
        return CensusResult("unconfirmed", reason=bound_reason)
    if escaped:
        return CensusResult("escaped", escaped=escaped, foreign=foreign)
    return CensusResult("confirmed", foreign=foreign)


# Supervisor ------------------------------------------------------------------


def _emit(payload: dict[str, object]) -> None:
    # A vanished broker must never stop teardown, the sweep or the census.
    try:
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass


class _Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.fixture = args.fixture
        # Absolute CLOCK_MONOTONIC deadlines shared with the broker process.
        self.setup_deadline = float(args.setup_deadline)
        self.request_deadline = float(args.request_deadline)
        self.teardown_timeout = max(0.1, float(args.teardown_timeout))
        self.sweep_allowance = max(1.0, float(args.sweep_allowance))
        self.mount_id = args.mount_id
        self.pid = os.getpid()
        own = read_entry(self.pid)
        self.self_start = own.start if own is not None else 0
        # Recorded before setup is forked.
        self.window_start = self.self_start
        # Issue #1995: the broker spawned this process directly, so the parent
        # is the broker; identities are always compared as (pid, start).
        parent = read_entry(os.getppid())
        self.broker_identity: tuple[int, int] | None = (
            (parent.pid, parent.start) if parent is not None and parent.pid > 1 else None
        )
        self.declared_foreign_scopes: tuple[str, ...] = tuple(getattr(args, "foreign_scope", None) or ())
        self.run_cgroup: str | None = None
        self.foreign_scopes: tuple[str, ...] = ()
        self.scope_warnings: tuple[str, ...] = ()
        self.inventory: dict[int, int] = {}
        self.teardown_requested = False
        self.stdin_closed = False
        self._stdin_buffer = b""
        self._last_inventory = 0.0

    # -- commands
    def poll_commands(self, timeout: float = 0.0) -> None:
        if self.stdin_closed:
            return
        try:
            ready, _, _ = select.select([sys.stdin.buffer], [], [], max(0.0, timeout))
        except (OSError, ValueError):
            self.stdin_closed = True
            self.teardown_requested = True
            return
        if not ready:
            return
        chunk = os.read(sys.stdin.fileno(), 4096)
        if not chunk:
            self.stdin_closed = True
            self.teardown_requested = True
            return
        self._stdin_buffer += chunk
        while b"\n" in self._stdin_buffer:
            line, self._stdin_buffer = self._stdin_buffer.split(b"\n", 1)
            if line.strip() == b"teardown":
                self.teardown_requested = True

    # -- inventory
    def refresh_inventory(self) -> None:
        table = process_table()
        for entry in descendants(self.pid, table):
            if self.inventory.get(entry.pid) == entry.start:
                continue
            self.inventory[entry.pid] = entry.start
            _emit({"event": "inventory", "pid": entry.pid, "start": entry.start})

    def reap(self) -> None:
        while True:
            try:
                pid, _status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return

    # -- phases
    def run_phase(self, phase: str, deadline: float, *, interruptible: bool) -> tuple[object, float, str]:
        """Run one phase until it exits, ``deadline`` (absolute) or teardown."""
        started = time.monotonic()
        if started >= deadline:
            return "timed_out", 0.0, ""
        tail: deque[bytes] = deque()
        tail_size = 0
        try:
            proc = subprocess.Popen(
                [self.fixture, phase],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            return 127, time.monotonic() - started, f"fixture {phase} did not start: {type(exc).__name__}"
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        os.set_blocking(fd, False)
        status: object = None
        stream_open = True

        def drain() -> None:
            nonlocal tail_size, stream_open
            for _ in range(MAX_DRAIN_READS):
                if not stream_open:
                    return
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    return
                except OSError:
                    stream_open = False
                    return
                if not chunk:
                    stream_open = False
                    return
                tail.append(chunk)
                tail_size += len(chunk)
                while tail_size > LOG_TAIL_BYTES and len(tail) > 1:
                    tail_size -= len(tail.popleft())

        while True:
            drain()
            if proc.poll() is not None:
                status = proc.returncode
                break
            if interruptible:
                self.poll_commands()
                if self.teardown_requested:
                    self._terminate(proc)
                    status = "cancelled"
                    break
            if time.monotonic() >= deadline:
                self._terminate(proc)
                status = "timed_out"
                break
            if time.monotonic() - self._last_inventory >= POLL_SECONDS:
                self.refresh_inventory()
                self._last_inventory = time.monotonic()
            try:
                select.select([fd] if stream_open else [], [], [], POLL_SECONDS)
            except (OSError, ValueError):
                time.sleep(POLL_SECONDS)
        # A daemonized child may still hold the pipe; drain one bounded batch.
        drain()
        proc.stdout.close()
        self.refresh_inventory()
        raw = b"".join(tail)[-LOG_TAIL_BYTES:]
        if isinstance(status, int) and status < 0:
            status = 128 - status
        return status, time.monotonic() - started, raw.decode("utf-8", errors="replace")

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        for sig, grace in ((signal.SIGTERM, TERM_GRACE_SECONDS), (signal.SIGKILL, TERM_GRACE_SECONDS)):
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                continue

    def sweep(self, deadline: float) -> tuple[int, bool]:
        """Kill and reap every descendant; return (swept, owned tree empty)."""
        swept: set[int] = set()
        for sig in (signal.SIGTERM, signal.SIGKILL):
            self.refresh_inventory()
            targets = descendants(self.pid, process_table())
            for entry in targets:
                swept.add(entry.pid)
                try:
                    os.kill(entry.pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            # Inventoried processes that were re-parented away still belong to
            # the fixture: signal them when their identity still matches.
            for pid, start in list(self.inventory.items()):
                entry = read_entry(pid)
                if entry is not None and entry.start == start:
                    swept.add(pid)
                    try:
                        os.kill(pid, sig)
                    except (ProcessLookupError, PermissionError):
                        pass
            grace_end = min(deadline, time.monotonic() + TERM_GRACE_SECONDS)
            while time.monotonic() < grace_end:
                self.reap()
                if not descendants(self.pid, process_table()) and not self._inventory_alive():
                    break
                time.sleep(0.05)
        while time.monotonic() < deadline:
            self.reap()
            if not descendants(self.pid, process_table()) and not self._inventory_alive():
                break
            time.sleep(0.05)
        self.reap()
        empty = not descendants(self.pid, process_table()) and not self._inventory_alive()
        return len(swept), empty

    def resolve_run_scope(self, proc_root: Path = Path("/proc")) -> None:
        """Read the run cgroup after the broker's cgroup attachment (issue #1995).

        Without a readable cgroup-v2 path or broker identity the census stays
        host-wide.  Declared foreign scopes that would cover the run's own
        subtree are discarded here, with a visible warning.
        """
        run_cgroup = read_unified_cgroup("self", proc_root)
        if self.broker_identity is None:
            run_cgroup = None
        self.run_cgroup = run_cgroup
        kept, warnings = effective_foreign_scopes(self.declared_foreign_scopes, run_cgroup)
        if self.declared_foreign_scopes and run_cgroup is None:
            warnings = (
                "run cgroup is unknown; declared foreign scopes are ignored and the census stays host-wide",
            )
        self.foreign_scopes = kept
        # Issue #1995: also carried in the cleanup event, because the broker
        # discards this process's stderr.
        self.scope_warnings = tuple(str(item)[:MAX_DIAGNOSTIC_CHARS] for item in warnings[:MAX_ESCAPED_REPORTED])
        try:
            for warning in warnings:
                sys.stderr.write(f"agent-loop fixture: warning: {warning}\n")
            sys.stderr.flush()
        except (OSError, ValueError):
            pass

    def run_census(self, deadline: float, proc_root: Path = Path("/proc")) -> CensusResult:
        return census(
            window_start=self.window_start,
            owned=self.inventory.items(),
            self_identity=(self.pid, self.self_start),
            proc_root=proc_root,
            deadline=deadline,
            run_cgroup=self.run_cgroup,
            run_roots=(() if self.broker_identity is None else (self.broker_identity,)),
            foreign_scopes=self.foreign_scopes,
        )

    def emit_cleanup(self, result: CensusResult, *, swept: int) -> None:
        hint = (
            "" if self.foreign_scopes
            else " (no foreign scope declared; see --fixture-foreign-scope)"
        )
        try:
            for item in result.escaped[:MAX_ESCAPED_REPORTED]:
                attribution = item.get("attribution", ATTRIBUTION_UNDETERMINED)
                sys.stderr.write(
                    "agent-loop fixture: escaped process pid={pid} uid={uid} comm={comm} cgroup={cgroup}".format(**item)
                    + f" attribution={attribution}"
                    + (hint if attribution == ATTRIBUTION_UNDETERMINED else "")
                    + "\n"
                )
            for item in result.foreign[:MAX_FOREIGN_REPORTED]:
                sys.stderr.write(
                    "agent-loop fixture: foreign process (declared foreign scope) "
                    "pid={pid} uid={uid} comm={comm} cgroup={cgroup}\n".format(**item)
                )
            sys.stderr.flush()
        except (OSError, ValueError):
            pass
        _emit(
            {
                "event": "cleanup",
                "cleanup": result.cleanup,
                "reason": result.reason,
                "inventoried": len(self.inventory),
                "swept": swept,
                "escaped": result.escaped[:MAX_ESCAPED_REPORTED],
                "escaped_count": len(result.escaped),
                "foreign": result.foreign[:MAX_FOREIGN_REPORTED],
                "foreign_count": len(result.foreign),
                "scope_warnings": list(self.scope_warnings),
                "foreign_scopes_declared": bool(self.foreign_scopes),
            }
        )

    def _inventory_alive(self) -> bool:
        for pid, start in self.inventory.items():
            entry = read_entry(pid)
            if entry is not None and entry.start == start and entry.state not in {"Z", "X"}:
                return True
        return False

    def run(self) -> int:
        try:
            libc = _libc()
            if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
            if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
        except OSError as exc:
            _emit({"event": "fatal", "error": f"supervisor setup failed: {exc}"})
            return 2
        _emit({"event": "ready", "pid": self.pid, "start": self.self_start, "window_start": self.window_start})
        # Wait for the broker's go-ahead (after its cgroup attachment).
        while True:
            try:
                line = sys.stdin.buffer.readline()
            except OSError:
                line = b""
            if not line:
                _emit({"event": "setup", "status": "cancelled", "duration": 0.0, "log": "", "ran": False})
                self.teardown_requested = True
                setup_status: object = "cancelled"
                break
            if line.strip() == b"teardown":
                _emit({"event": "setup", "status": "cancelled", "duration": 0.0, "log": "", "ran": False})
                self.teardown_requested = True
                setup_status = "cancelled"
                break
            if line.strip() == b"expire":
                # The setup budget ran out before the go-ahead: never started.
                _emit({"event": "setup", "status": "timed_out", "duration": 0.0, "log": "", "ran": False})
                self.teardown_requested = True
                setup_status = "timed_out"
                break
            if line.strip() == b"start":
                setup_status = None
                break
        # After the go-ahead, which follows the broker's cgroup attachment.
        self.resolve_run_scope()
        if setup_status is None and time.monotonic() >= self.setup_deadline:
            # The go-ahead was consumed after the absolute setup deadline
            # (for example the supervisor was descheduled): never start setup.
            _emit({"event": "setup", "status": "timed_out", "duration": 0.0, "log": "", "ran": False})
            self.teardown_requested = True
            setup_status = "timed_out"
        if setup_status is None:
            setup_status, duration, log = self.run_phase("setup", self.setup_deadline, interruptible=True)
            _emit({"event": "setup", "status": setup_status, "duration": round(duration, 3), "log": log, "ran": True})
            setup_ran = True
        else:
            setup_ran = False
        while not self.teardown_requested:
            self.poll_commands(POLL_SECONDS)
            self.refresh_inventory()
            self.reap()
        if setup_ran:
            # Teardown ends early enough for the sweep allowance before the
            # request deadline, but always keeps a positive bound.
            now = time.monotonic()
            bound = min(self.teardown_timeout, self.request_deadline - self.sweep_allowance - now)
            teardown_status, duration, log = self.run_phase(
                "teardown", now + max(MIN_TEARDOWN_SECONDS, bound), interruptible=False
            )
        else:
            teardown_status, duration, log = "not-run", 0.0, ""
        _emit({
            "event": "teardown", "status": teardown_status, "duration": round(duration, 3), "log": log,
            "ran": setup_ran,
        })
        now = time.monotonic()
        sweep_deadline = now + max(MIN_SWEEP_SECONDS, min(self.sweep_allowance, self.request_deadline - now))
        census_reserve = min(CENSUS_SECONDS, self.sweep_allowance / 2)
        swept, empty = self.sweep(sweep_deadline - census_reserve)
        if not empty:
            result = CensusResult("unconfirmed", reason="the owned tree is not empty after the sweep")
        else:
            try:
                verify_host_process_authority(expected_mount_id=self.mount_id)
            except HostAuthorityError as exc:
                result = CensusResult("unconfirmed", reason=f"census authority lost: {exc}")
            else:
                result = self.run_census(sweep_deadline)
        self.emit_cleanup(result, swept=swept)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fixture_supervisor")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--setup-deadline", required=True, type=float)
    parser.add_argument("--request-deadline", required=True, type=float)
    parser.add_argument("--teardown-timeout", required=True, type=float)
    parser.add_argument("--sweep-allowance", required=True, type=float)
    parser.add_argument("--mount-id", required=True)
    # Issue #1995: operator-declared foreign cgroup scopes, from orchestrator
    # config only (the broker never takes them from a run-tests client).
    parser.add_argument("--foreign-scope", action="append", default=[])
    args = parser.parse_args(argv)
    return _Supervisor(args).run()


if __name__ == "__main__":  # pragma: no cover - exercised through the broker
    raise SystemExit(main())
