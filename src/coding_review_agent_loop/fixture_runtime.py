"""Broker-side support for ``agent-loop run-tests --fixture`` (issue #1326).

The broker validates fixture provenance before anything launches, then runs
``<fixture> setup``, the unchanged recognized inner command and
``<fixture> teardown`` under one absolute request deadline.  A Linux
child-subreaper supervisor (:mod:`fixture_supervisor`) owns every fixture
descendant; cleanup is confirmed only when its owned tree is empty and the
census finds no process started in the window that the run could own
(issue #1995: processes proven to be in an operator-declared foreign scope
are reported as diagnostics only).

The fixture is never probed or recorded as a launcher: launch integrity
still authenticates only the inner command.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .errors import AgentLoopError
from .fixture_supervisor import (
    HostAuthority,
    HostAuthorityError,
    read_entry,
    verify_host_process_authority as _verify_host_process_authority,
)
from .local_test_evidence import (
    FIXTURE_MIN_TIMEOUT_SECONDS,
    FixtureRecord,
    TrackedTreeSnapshot,
    _redact_command_tokens,
    _run_git,
    _safe_text,
)

SWEEP_ALLOWANCE_SECONDS = 5.0
MIN_SETUP_AND_INNER_SECONDS = 10.0
MAX_TEARDOWN_BUDGET_SECONDS = 55.0
MIN_TEARDOWN_BUDGET_SECONDS = 5.0
LOG_EXCERPT_BYTES = 1024
SUPERVISOR_START_SECONDS = 10.0
# The supervisor stops setup this long before the inner deadline, so its
# SIGTERM/SIGKILL grace still ends by that deadline.
SETUP_TERMINATION_MARGIN_SECONDS = 2.5
# Reporting slack after the request deadline before the broker forces cleanup.
FINISH_SLACK_SECONDS = 1.0

FIXTURE_SETUP_FAILED_CAVEAT = "fixture setup failed"
FIXTURE_TEARDOWN_FAILED_CAVEAT = "fixture teardown failed"
FIXTURE_CLEANUP_ESCAPED_CAVEAT = "fixture cleanup escaped"
FIXTURE_CLEANUP_UNCONFIRMED_CAVEAT = "fixture cleanup unconfirmed"
LAUNCHER_CHANGED_CAVEAT = "launcher changed after authentication"

SUPERVISOR_PATH = Path(__file__).resolve().with_name("fixture_supervisor.py")


class FixtureRefused(AgentLoopError):
    """A fixture request refused before anything launched."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(
            f"fixture refused: {reason_code}" + (f" ({detail})" if detail else "")
        )


# Budgets -----------------------------------------------------------------------


def teardown_budget_seconds(timeout_seconds: float) -> float:
    return min(MAX_TEARDOWN_BUDGET_SECONDS, max(MIN_TEARDOWN_BUDGET_SECONDS, 0.1 * float(timeout_seconds)))


def reserve_seconds(timeout_seconds: float) -> float:
    """Teardown bound plus the separate sweep and census allowance."""
    return teardown_budget_seconds(timeout_seconds) + SWEEP_ALLOWANCE_SECONDS


def validate_fixture_timeout(timeout_seconds: float) -> None:
    if not float(timeout_seconds) >= FIXTURE_MIN_TIMEOUT_SECONDS:
        raise FixtureRefused(
            "fixture-timeout-below-floor",
            f"a --fixture run needs at least {FIXTURE_MIN_TIMEOUT_SECONDS:g}s",
        )


# Host authority ------------------------------------------------------------------


def verify_host_process_authority(**kwargs: Any) -> HostAuthority:
    """Raise ``FixtureRefused('fixture-scope-unsupported')`` without proof."""
    try:
        return _verify_host_process_authority(**kwargs)
    except HostAuthorityError as exc:
        raise FixtureRefused("fixture-scope-unsupported", str(exc)) from exc


# Provenance ---------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureProvenance:
    relative_path: str
    absolute_path: Path
    mode: str
    blob: str

    @property
    def digest(self) -> str:
        return f"{self.mode}:{self.blob}"


def validate_fixture_path(root: Path, cwd: Path, fixture: str) -> FixtureProvenance:
    """Refuse anything but an executable tracked blob identical to HEAD."""
    root = Path(root)
    lexical = Path(os.path.normpath(Path(fixture) if os.path.isabs(fixture) else Path(cwd) / fixture))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise FixtureRefused("fixture-outside-checkout") from exc
    if not relative.parts or relative.parts[0] == "..":
        raise FixtureRefused("fixture-outside-checkout")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise FixtureRefused("fixture-missing") from exc
        import stat as _stat

        if _stat.S_ISLNK(info.st_mode):
            raise FixtureRefused("fixture-symlink")
    import stat as _stat

    if not _stat.S_ISREG(os.lstat(lexical).st_mode):
        raise FixtureRefused("fixture-not-regular-file")
    rel = relative.as_posix()
    try:
        listing = _run_git(root, ("ls-files", "-s", "--error-unmatch", "--", rel))
    except AgentLoopError as exc:
        ignored = False
        try:
            _run_git(root, ("check-ignore", "-q", "--", rel))
            ignored = True
        except AgentLoopError:
            ignored = False
        raise FixtureRefused("fixture-ignored" if ignored else "fixture-untracked") from exc
    entries = [line for line in listing.decode("utf-8", errors="replace").splitlines() if line]
    if len(entries) != 1:
        raise FixtureRefused("fixture-untracked", "path is not exactly one tracked file")
    meta, _tab, path = entries[0].partition("\t")
    fields = meta.split()
    if path != rel or len(fields) != 3 or fields[2] != "0":
        raise FixtureRefused("fixture-untracked", "path is not one merged index entry")
    mode = fields[0]
    if mode == "160000":
        raise FixtureRefused("fixture-gitlink")
    if mode == "120000":
        raise FixtureRefused("fixture-symlink")
    if mode == "100644":
        raise FixtureRefused("fixture-not-executable")
    if mode != "100755":
        raise FixtureRefused("fixture-unsupported-mode")
    try:
        head_blob = _run_git(root, ("rev-parse", "--verify", "--quiet", f"HEAD:{rel}")).decode().strip()
    except AgentLoopError as exc:
        raise FixtureRefused("fixture-not-in-head") from exc
    try:
        # Raw bytes: clean/smudge, EOL and ident attributes must never make a
        # transformed worktree file hash like the HEAD blob.
        worktree_blob = _run_git(root, ("hash-object", "--no-filters", "--", rel)).decode().strip()
    except AgentLoopError as exc:
        raise FixtureRefused("fixture-unreadable") from exc
    if not head_blob or worktree_blob != head_blob:
        raise FixtureRefused("fixture-modified", "worktree bytes differ from HEAD")
    return FixtureProvenance(rel, lexical, mode, head_blob)


def require_clean_tree(before: TrackedTreeSnapshot) -> None:
    if (
        before.status_clean is not True
        or not before.complete
        or before.referenced_untracked_paths
        or before.untracked_paths
    ):
        raise FixtureRefused("fixture-tree-dirty")


# Exit-code folding ---------------------------------------------------------------


def _exit_status(value: int) -> int:
    if 1 <= value <= 255:
        return value
    if value < 0 and 128 - value <= 255:
        return 128 - value
    return 1


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def fold_exit_code(
    *,
    outcome: str,
    returncode: int | None,
    inner_exec: str,
    fixture: Mapping[str, object] | None,
    cancelled_frame: bool = False,
) -> int:
    """Total exit-code folding; 0 only for a fully clean, passing lifecycle."""
    record = fixture or {}
    setup = record.get("setup_status", "not-run")
    teardown = record.get("teardown_status", "not-run")
    cleanup = record.get("cleanup", "unconfirmed")
    if (
        outcome == "passed"
        and not cancelled_frame
        and _is_int(setup) and setup == 0
        and inner_exec == "started"
        and _is_int(returncode) and returncode == 0
        and _is_int(teardown) and teardown == 0
        and cleanup == "confirmed"
    ):
        return 0
    # 0. A stop always reports as interrupted, whatever the phase statuses.
    if outcome in {"interrupted", "cancelled"} or cancelled_frame:
        return 130
    # 1-3. Setup.
    if _is_int(setup) and setup != 0:
        return _exit_status(int(setup))
    if setup == "timed_out":
        return 124
    if setup in {"not-run", "cancelled"} or inner_exec == "not-attempted":
        return 1
    # 4. The inner command.
    if _is_int(returncode) and returncode != 0:
        return _exit_status(int(returncode))
    if outcome == "timed_out" and returncode is None:
        return 124
    # 5. Teardown.
    if _is_int(teardown) and teardown != 0:
        return _exit_status(int(teardown))
    if teardown == "timed_out":
        return 124
    # 6. Cleanup.
    if cleanup != "confirmed":
        return 1
    # 7. Anything else that missed the success condition.
    return 1


# Log tails ----------------------------------------------------------------------


def redact_log_tail(text: str, *, cwd: Path | None) -> str:
    """Redact a bounded phase log with the evidence command rules."""
    lines: list[str] = []
    for line in str(text).splitlines()[-200:]:
        tokens = line.split()
        if not tokens:
            continue
        try:
            redacted, _command, _ids, _caveats = _redact_command_tokens(tokens, cwd=cwd)
            lines.append(" ".join(redacted))
        except Exception:
            lines.append(_safe_text(line))
    return "\n".join(lines)


def log_digest(redacted: str) -> str:
    """sha256 of a completed phase's redacted tail, including an empty one."""
    return hashlib.sha256(redacted.encode("utf-8", errors="replace")).hexdigest()


def log_excerpt(redacted: str) -> str:
    raw = redacted.encode("utf-8", errors="replace")[-LOG_EXCERPT_BYTES:]
    return raw.decode("utf-8", errors="ignore")


# Supervisor handle --------------------------------------------------------------


def supervisor_scope_arguments(foreign_scopes: Sequence[str]) -> list[str]:
    """Repeated ``--foreign-scope`` arguments for valid absolute patterns."""
    arguments: list[str] = []
    for pattern in dict.fromkeys(foreign_scopes or ()):
        if isinstance(pattern, str) and pattern.startswith("/") and "\x00" not in pattern and "\n" not in pattern:
            arguments.extend(("--foreign-scope", pattern))
    return arguments


@dataclass
class SetupDecision:
    """What the runner does after the post-admission setup hook."""

    action: str  # proceed | skip | timed_out | cancel
    reason: str = ""

    @property
    def proceed(self) -> bool:
        return self.action == "proceed"


@dataclass
class FixtureSupervisor:
    """The broker's handle on one running fixture supervisor."""

    provenance: FixtureProvenance
    process: subprocess.Popen
    teardown_seconds: float
    cwd: Path | None
    inventory: dict[int, int] = field(default_factory=dict)
    setup_event: dict[str, object] | None = None
    teardown_event: dict[str, object] | None = None
    cleanup_event: dict[str, object] | None = None
    ready_event: dict[str, object] | None = None
    fatal: str | None = None
    setup_started: bool = False
    forced: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _changed: threading.Condition | None = None
    _reader: threading.Thread | None = None
    _teardown_sent: bool = False
    request_deadline: float | None = None

    @classmethod
    def launch(
        cls,
        provenance: FixtureProvenance,
        *,
        cwd: Path,
        cwd_fd: int | None,
        environment: Mapping[str, str],
        setup_deadline: float,
        request_deadline: float,
        teardown_seconds: float,
        mount_id: str,
        foreign_scopes: Sequence[str] = (),
    ) -> "FixtureSupervisor":
        """Spawn the supervisor; deadlines are absolute ``time.monotonic()`` values.

        CLOCK_MONOTONIC is system-wide on Linux, so the supervisor enforces
        the same absolute deadlines as the broker.  ``foreign_scopes`` comes
        from parent-owned orchestrator config only (issue #1995).
        """
        spawn_cwd: str | Path = cwd
        preexec: Callable[[], None] | None = None
        if cwd_fd is not None:
            spawn_cwd = "/"
            preexec = lambda fd=cwd_fd: os.fchdir(fd)
        process = subprocess.Popen(
            [
                sys.executable, "-I", str(SUPERVISOR_PATH),
                "--fixture", str(provenance.absolute_path),
                "--setup-deadline", f"{setup_deadline:.6f}",
                "--request-deadline", f"{request_deadline:.6f}",
                "--teardown-timeout", f"{max(0.1, teardown_seconds):.3f}",
                "--sweep-allowance", f"{SWEEP_ALLOWANCE_SECONDS:.3f}",
                "--mount-id", mount_id,
                *supervisor_scope_arguments(foreign_scopes),
            ],
            cwd=spawn_cwd,
            preexec_fn=preexec,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=dict(environment),
            start_new_session=True,
        )
        handle = cls(provenance, process, float(teardown_seconds), cwd)
        handle.request_deadline = float(request_deadline)
        handle._changed = threading.Condition(handle._lock)
        handle._reader = threading.Thread(target=handle._read, name="agent-loop-fixture-supervisor", daemon=True)
        handle._reader.start()
        return handle

    # -- reading
    def _read(self) -> None:
        assert self.process.stdout is not None
        for raw in self.process.stdout:
            try:
                event = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("event")
            with self._lock:
                if kind == "inventory" and len(self.inventory) < 100_000:
                    pid, start = event.get("pid"), event.get("start")
                    if isinstance(pid, int) and isinstance(start, int):
                        self.inventory[pid] = start
                elif kind == "ready":
                    self.ready_event = event
                elif kind == "setup":
                    self.setup_event = event
                elif kind == "teardown":
                    self.teardown_event = event
                elif kind == "cleanup":
                    self.cleanup_event = event
                elif kind == "fatal":
                    self.fatal = str(event.get("error", "supervisor failed"))
                assert self._changed is not None
                self._changed.notify_all()
        with self._lock:
            assert self._changed is not None
            self._changed.notify_all()

    def _wait_for(self, predicate: Callable[[], bool], deadline: float) -> bool:
        assert self._changed is not None
        with self._changed:
            while not predicate():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                if self._reader is not None and not self._reader.is_alive() and not predicate():
                    return predicate()
                self._changed.wait(min(0.1, remaining))
            return True

    # -- commands
    def _send(self, line: str) -> None:
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(line.encode("ascii") + b"\n")
            self.process.stdin.flush()
        except (OSError, ValueError):
            pass

    def start_setup(self, parent_cgroup: Path | None, *, setup_deadline: float) -> bool:
        """Release setup unless its absolute deadline already passed.

        Returns ``False`` (and tells the supervisor its setup expired, so it
        never starts) when readiness consumed the setup budget.
        """
        ready_by = min(time.monotonic() + SUPERVISOR_START_SECONDS, setup_deadline)
        if not self._wait_for(lambda: self.ready_event is not None or self.fatal is not None, ready_by):
            if time.monotonic() >= setup_deadline:
                self._send("expire")
                return False
            raise AgentLoopError("fixture supervisor did not become ready")
        if self.fatal is not None:
            raise FixtureRefused("fixture-scope-unsupported", self.fatal)
        if parent_cgroup is not None:
            (parent_cgroup / "cgroup.procs").write_text(str(self.process.pid), encoding="ascii")
        if time.monotonic() >= setup_deadline:
            self._send("expire")
            return False
        self.setup_started = True
        self._send("start")
        return True

    def wait_setup(self, deadline: float) -> dict[str, object] | None:
        self._wait_for(lambda: self.setup_event is not None, deadline)
        return self.setup_event

    def request_teardown(self) -> None:
        with self._lock:
            if self._teardown_sent:
                return
            self._teardown_sent = True
        self._send("teardown")
        try:
            assert self.process.stdin is not None
            self.process.stdin.close()
        except (OSError, ValueError):
            pass

    def finish(self, *, budget: float | None = None) -> None:
        """Request teardown, wait for cleanup, then force termination.

        Without ``budget`` (the request path) the wait ends at the request
        deadline plus reporting slack; a broker stop passes the fixture's
        reserve as a budget from the moment of the stop.
        """
        self.request_teardown()
        if budget is not None:
            wait_until = time.monotonic() + budget + FINISH_SLACK_SECONDS
        elif self.request_deadline is not None:
            wait_until = self.request_deadline + FINISH_SLACK_SECONDS
        else:
            wait_until = time.monotonic() + self.teardown_seconds + SWEEP_ALLOWANCE_SECONDS + FINISH_SLACK_SECONDS
        self._wait_for(lambda: self.cleanup_event is not None, wait_until)
        if self.cleanup_event is None:
            self.force_terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.force_terminate()
        if self._reader is not None:
            self._reader.join(timeout=2)

    def force_terminate(self) -> None:
        """Last resort: kill the supervisor group and every inventoried PID."""
        self.forced = True
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        with self._lock:
            inventory = dict(self.inventory)
        for pid, start in inventory.items():
            entry = read_entry(pid)
            if entry is not None and entry.start == start:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    # -- record
    def _phase(self, event: dict[str, object] | None) -> tuple[int | str, float, str, str]:
        if event is None:
            return "not-run", 0.0, "", ""
        status = event.get("status")
        if isinstance(status, bool) or not (isinstance(status, int) or status in {"timed_out", "cancelled", "not-run"}):
            status = "not-run"
        duration = event.get("duration")
        duration = (
            float(duration)
            if isinstance(duration, (int, float)) and not isinstance(duration, bool)
            and 0 <= duration <= 86_400 and math.isfinite(float(duration))
            else 0.0
        )
        if event.get("ran") is False:
            # Never started: no log, so no digest.
            return status, 0.0, "", ""
        redacted = redact_log_tail(str(event.get("log") or ""), cwd=self.cwd)
        return status, round(duration, 3), log_digest(redacted), log_excerpt(redacted)

    def record(self) -> tuple[FixtureRecord, str]:
        """The bounded record and the private (never projected) excerpts."""
        with self._lock:
            setup_event, teardown_event, cleanup_event = self.setup_event, self.teardown_event, self.cleanup_event
            inventoried = len(self.inventory)
        setup_status, setup_duration, setup_digest, setup_excerpt = self._phase(setup_event)
        teardown_status, teardown_duration, teardown_digest, teardown_excerpt = self._phase(teardown_event)
        cleanup = "unconfirmed"
        swept = 0
        escaped_count = 0
        escaped: list[str] = []
        if cleanup_event is not None and not self.forced:
            state = cleanup_event.get("cleanup")
            cleanup = state if state in {"confirmed", "escaped", "unconfirmed"} else "unconfirmed"
            swept = int(cleanup_event.get("swept") or 0) if isinstance(cleanup_event.get("swept"), int) else 0
            raw_escaped = cleanup_event.get("escaped") or []
            escaped_count = (
                int(cleanup_event.get("escaped_count"))
                if isinstance(cleanup_event.get("escaped_count"), int) else len(raw_escaped)
            )
            for item in list(raw_escaped)[:8]:
                if isinstance(item, Mapping):
                    escaped.append(
                        _safe_text(
                            f"pid={item.get('pid')} uid={item.get('uid')} comm={item.get('comm')} "
                            + (
                                f"attribution={item.get('attribution')} "
                                if item.get("attribution") in {"run", "undetermined"} else ""
                            )
                            + f"cgroup={item.get('cgroup')}",
                            160,
                        )
                    )
        record = FixtureRecord(
            path=self.provenance.relative_path,
            digest=self.provenance.digest,
            setup_status=setup_status,
            setup_duration_seconds=setup_duration,
            setup_log_digest=setup_digest,
            teardown_status=teardown_status,
            teardown_duration_seconds=teardown_duration,
            teardown_log_digest=teardown_digest,
            cleanup=cleanup,
            inventoried=min(inventoried, 1_000_000),
            swept=min(max(0, swept), 1_000_000),
            escaped=min(max(0, escaped_count), 1_000_000),
            escaped_processes=tuple(escaped),
        )
        excerpts = (
            f"fixture setup log (redacted):\n{setup_excerpt}\n"
            f"fixture teardown log (redacted):\n{teardown_excerpt}"
        )
        return record, excerpts


def not_run_record(provenance: FixtureProvenance) -> FixtureRecord:
    """Nothing launched: no setup, no teardown, nothing to confirm."""
    return FixtureRecord(path=provenance.relative_path, digest=provenance.digest, cleanup="unconfirmed")


def lifecycle_caveats(record: FixtureRecord) -> tuple[str, ...]:
    caveats: list[str] = []
    if record.setup_status != 0 or isinstance(record.setup_status, bool):
        caveats.append(FIXTURE_SETUP_FAILED_CAVEAT)
    elif record.teardown_status != 0 or isinstance(record.teardown_status, bool):
        caveats.append(FIXTURE_TEARDOWN_FAILED_CAVEAT)
    if record.cleanup == "escaped":
        caveats.append(FIXTURE_CLEANUP_ESCAPED_CAVEAT)
    elif record.cleanup != "confirmed":
        caveats.append(FIXTURE_CLEANUP_UNCONFIRMED_CAVEAT)
    return tuple(caveats)
