"""Subprocess execution helpers for the agent loop."""

from __future__ import annotations

import errno
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeVar

from .errors import AgentLoopError


@dataclass(frozen=True)
class ExecutableIdentity:
    path: str | None
    target: str | None
    entry_identity: tuple[int, int, int] | None
    target_identity: tuple[int, int, int] | None


@dataclass(frozen=True)
class ExecutionObservation:
    spawn_wall_time: float
    spawn_monotonic: float
    exit_monotonic: float
    elapsed_seconds: float
    before: ExecutableIdentity
    after: ExecutableIdentity
    interrupted: bool


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    cwd: Path
    stdout: str
    stderr: str
    returncode: int | None
    observation: ExecutionObservation | None = None
    capture_diagnostics: tuple[str, ...] = ()


# Matches ANSI/VT100 control sequences (CSI, OSC, charset selection) that a
# program emits when it believes it is talking to a terminal. We run some agents
# under a pseudo-terminal (see run_with_log use_pty), so we strip these from the
# captured output before parsing it.
_ANSI_RE = re.compile(r"\x1B\[[0-9;?]*[ -/]*[@-~]|\x1B\][^\x07\x1B]*(?:\x07|\x1B\\)|\x1B[()][AB0-2]")
_SPAWN_ATTEMPTS = 3
_SPAWN_RETRY_BACKOFF_SECONDS = 2
_SpawnResult = TypeVar("_SpawnResult")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text).replace("\r", "")


def tail_text(text: str, *, max_lines: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def executable_identity(command: str) -> ExecutableIdentity:
    """Capture an attempt-local identity for a bare command without caching it."""
    path = shutil.which(command) if not os.path.isabs(command) else command
    if path is None:
        return ExecutableIdentity(None, None, None, None)
    target = os.path.realpath(path)
    def identity(stat_result: os.stat_result) -> tuple[int, int, int]:
        return (stat_result.st_dev, stat_result.st_ino, stat_result.st_mtime_ns)
    try:
        entry = identity(os.lstat(path))
    except OSError:
        entry = None
    try:
        resolved = identity(os.stat(path))
    except OSError:
        resolved = None
    return ExecutableIdentity(path, target, entry, resolved)


def ensure_log_dir_ignored(log_dir: Path) -> None:
    gitignore = log_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")


class Runner:
    def __init__(self, *, dry_run: bool = False):
        self.dry_run = dry_run
        self._resolved_commands: dict[str, str] = {}
        self._command_override_flags: dict[str, str] = {}
        self._preflighted_commands: set[str] = set()
        # Parallel discuss debaters call run_with_log from worker threads (#475):
        # guard the command caches and keep a registry of live agent processes so
        # the main thread can kill them on KeyboardInterrupt.
        self._commands_lock = threading.Lock()
        self._active_procs_lock = threading.Lock()
        self._active_procs: dict[int, subprocess.Popen] = {}
        self._interrupted = False

    def remember_agent_command(
        self,
        command: str,
        resolved_path: str,
        override_flag: str,
    ) -> None:
        """Retain preflight evidence for a later spawn-time PATH race."""
        with self._commands_lock:
            self._resolved_commands[command] = resolved_path
            self._command_override_flags[command] = override_flag
            self._preflighted_commands.add(command)

    def _register_active_process(self, proc: subprocess.Popen) -> None:
        with self._active_procs_lock:
            self._active_procs[proc.pid] = proc

    def _unregister_active_process(self, proc: subprocess.Popen) -> None:
        with self._active_procs_lock:
            self._active_procs.pop(proc.pid, None)

    @staticmethod
    def _terminate_process_group(proc: subprocess.Popen) -> None:
        """killpg TERM, short grace, then killpg KILL; tolerate an exited group."""
        try:
            os.killpg(proc.pid, 15)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
            proc.wait()

    def terminate_active_processes(self) -> None:
        """Kill every registered agent process group and refuse new spawns.

        Called from the main thread on KeyboardInterrupt while parallel discuss
        debaters run in worker threads: killing the process groups unblocks the
        workers' wait loops so executor shutdown returns promptly, and the
        interrupted flag stops in-flight retry logic from spawning replacements.
        """
        self._interrupted = True
        with self._active_procs_lock:
            procs = list(self._active_procs.values())
        for proc in procs:
            if proc.poll() is None:
                self._terminate_process_group(proc)

    def _override_flag_for(self, command: str) -> str:
        override_flag = self._command_override_flags.get(command)
        if override_flag is None:
            command_name = Path(command).name
            override_flag = f"--{command_name}-cmd"
        return override_flag

    def _missing_command_error(self, command: str) -> AgentLoopError:
        override_flag = self._override_flag_for(command)
        if os.path.isabs(command):
            return AgentLoopError(
                f"{command} not found or not executable; "
                f"pass a valid executable path to {override_flag}."
            )
        return AgentLoopError(
            f"{command} CLI not found on PATH; install it or pass "
            f"{override_flag} <path>."
        )

    def _command_disappeared_after_preflight_error(self, command: str) -> AgentLoopError:
        override_flag = self._override_flag_for(command)
        return AgentLoopError(
            f"{command} CLI disappeared after successful preflight and may be updating; "
            f"it did not return on PATH after {_SPAWN_ATTEMPTS} spawn attempts. "
            f"Retry shortly or pass {override_flag} <path>."
        )

    def _command_non_executable_after_preflight_error(self, command: str) -> AgentLoopError:
        override_flag = self._override_flag_for(command)
        return AgentLoopError(
            f"{command} CLI remained temporarily non-executable after "
            f"{_SPAWN_ATTEMPTS} spawn attempts and may be updating; "
            f"retry shortly or pass {override_flag} <path>."
        )

    @staticmethod
    def _is_dangling_symlink(path: str | None) -> bool:
        return bool(path and os.path.islink(path) and not os.path.exists(path))

    def _spawn_with_retry(
        self,
        cmd: list[str],
        spawn_attempt: Callable[[], _SpawnResult],
    ) -> _SpawnResult:
        command = cmd[0]
        resolved = shutil.which(command)
        if resolved is not None:
            with self._commands_lock:
                self._resolved_commands[command] = resolved

        saw_preflight_disappearance = False
        for attempt in range(1, _SPAWN_ATTEMPTS + 1):
            try:
                return spawn_attempt()
            except FileNotFoundError as exc:
                if os.path.isabs(command):
                    raise self._missing_command_error(command) from exc
                current = shutil.which(command)
                with self._commands_lock:
                    if current is not None:
                        self._resolved_commands[command] = current
                    candidate = current or self._resolved_commands.get(command)
                    preflighted = command in self._preflighted_commands
                dangling_symlink = self._is_dangling_symlink(candidate)
                saw_preflight_disappearance |= preflighted and current is None
                if not dangling_symlink and not preflighted:
                    raise self._missing_command_error(command) from exc
                if attempt == _SPAWN_ATTEMPTS:
                    if saw_preflight_disappearance and not dangling_symlink:
                        raise self._command_disappeared_after_preflight_error(command) from exc
                    raise self._missing_command_error(command) from exc
                time.sleep(_SPAWN_RETRY_BACKOFF_SECONDS)
            except OSError as exc:
                # Claude Code can briefly leave its bare command pointing at
                # an incomplete native install during an auto-update. Treat
                # only ENOEXEC for a preflighted bare command as retryable;
                # permission errors and explicit paths remain fail-closed.
                if exc.errno != errno.ENOEXEC or os.path.isabs(command):
                    raise
                with self._commands_lock:
                    preflighted = command in self._preflighted_commands
                if not preflighted:
                    raise
                if attempt == _SPAWN_ATTEMPTS:
                    raise self._command_non_executable_after_preflight_error(command) from exc
                time.sleep(_SPAWN_RETRY_BACKOFF_SECONDS)

        raise AssertionError("spawn retry loop exited unexpectedly")

    def wait_for_executable_stability(self, command: str, *, deadline: float | None = None) -> bool:
        """Observe two matching command identities a second apart, for at most six seconds."""
        if self._interrupted:
            return False
        end = min(deadline, time.monotonic() + 6) if deadline is not None else time.monotonic() + 6
        previous: ExecutableIdentity | None = None
        while time.monotonic() < end and not self._interrupted:
            current = executable_identity(command)
            if current.path and current.entry_identity and current.target_identity and current == previous:
                return True
            previous = current
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))
        return False

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        input_text: str | None = None,
        check: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        cmd = [str(a) for a in args]
        if self.dry_run:
            print(f"[dry-run] ({cwd}) {' '.join(cmd)}")
            if input_text:
                print(input_text)
            return CommandResult(cmd, cwd, "", "", 0)
        if self._interrupted:
            raise AgentLoopError(
                "Runner is shutting down after an interrupt; refusing to start new commands."
            )

        proc = subprocess.run(
            cmd,
            cwd=cwd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env={**os.environ, **env} if env is not None else None,
        )
        result = CommandResult(cmd, cwd, proc.stdout, proc.stderr, proc.returncode)
        if check and proc.returncode != 0:
            raise AgentLoopError(
                f"Command failed with exit {proc.returncode}: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
        return result

    def run_with_log(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        log_path: Path,
        label: str,
        progress_interval_seconds: int,
        check: bool = True,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
        use_pty: bool = False,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        cmd = [str(a) for a in args]
        if self.dry_run:
            print(f"[dry-run] ({cwd}) {' '.join(cmd)}")
            return CommandResult(cmd, cwd, "", "", 0)
        if self._interrupted:
            raise AgentLoopError(
                "Runner is shutting down after an interrupt; refusing to start new commands."
            )

        log_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_log_dir_ignored(log_path.parent)
        if use_pty:
            return self._run_with_log_pty(
                cmd,
                cwd=cwd,
                log_path=log_path,
                label=label,
                progress_interval_seconds=progress_interval_seconds,
                check=check,
                env=env,
                input_text=input_text,
                timeout_seconds=timeout_seconds,
            )
        started = time.monotonic()
        deadline = started + timeout_seconds if timeout_seconds is not None else None
        next_progress = started + progress_interval_seconds
        header = f"$ {' '.join(cmd)}\n"
        if input_text is not None:
            # Preserve stdin-routed prompts in the log just as argv-routed
            # prompts are preserved in the command header.
            header += f"\n# stdin\n{input_text}\n"
        header += "\n"
        capture_diagnostics: list[str] = []
        with log_path.open("w+", encoding="utf-8") as log_file:
            log_file.write(header)
            log_file.flush()
            # stderr=subprocess.STDOUT merges stderr into the log file.
            # All agent backends (Claude, Codex, Gemini) use run_with_log,
            # so stderr capture is uniform across them (issue #266).
            # start_new_session isolates the child's process group so timeout
            # and interrupt handling can killpg the whole tree (#475).
            # A regular file keeps large prompts out of argv and avoids blocking
            # on a pipe before a child has started consuming stdin.
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as input_file:
                if input_text is not None:
                    input_file.write(input_text)
                    input_file.seek(0)

                def spawn() -> subprocess.Popen[str]:
                    if input_text is not None:
                        input_file.seek(0)
                    return subprocess.Popen(
                        cmd,
                        cwd=cwd,
                        stdin=input_file if input_text is not None else subprocess.DEVNULL,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                        env={**os.environ, **env} if env is not None else None,
                    )

                proc = self._spawn_with_retry(
                    cmd,
                    spawn,
                )
                spawn_wall_time = time.time()
                spawn_monotonic = time.monotonic()
                before_identity = executable_identity(cmd[0])
                self._register_active_process(proc)
                try:
                    while True:
                        returncode = proc.poll()
                        if returncode is not None:
                            break
                        now = time.monotonic()
                        if deadline is not None and now >= deadline:
                            # returncode=None marks the timeout for callers, matching
                            # the pty branch.
                            self._terminate_process_group(proc)
                            returncode = None
                            break
                        if now >= next_progress:
                            elapsed = int(now - started)
                            print(
                                f"[agent-loop {datetime.now().strftime('%H:%M:%S')}] "
                                f"{label} still running ({elapsed}s); log: {log_path}",
                                file=sys.stderr,
                                flush=True,
                            )
                            next_progress = now + progress_interval_seconds
                        if deadline is not None:
                            time.sleep(min(1.0, max(0.05, deadline - now)))
                        else:
                            time.sleep(1)
                except KeyboardInterrupt:
                    # The child runs in its own session and no longer receives the
                    # terminal SIGINT; kill its whole process group instead.
                    self._terminate_process_group(proc)
                    raise
                finally:
                    self._unregister_active_process(proc)

            # Read through the retained descriptor. A concurrent cleanup may
            # unlink the pathname while the child still owns this open file.
            try:
                log_file.flush()
                log_file.seek(0)
                full_output = log_file.read()
            except (OSError, UnicodeError) as exc:
                capture_diagnostics.append(f"capture_read_failed:{type(exc).__name__}:{exc}")
                full_output = ""
        output = full_output[len(header):] if full_output.startswith(header) else full_output
        exited = time.monotonic()
        result = CommandResult(cmd, cwd, output, "", returncode, ExecutionObservation(
            spawn_wall_time, spawn_monotonic,
            exited, exited - spawn_monotonic, before_identity, executable_identity(cmd[0]), self._interrupted,
        ), tuple(capture_diagnostics))
        if check and returncode != 0:
            raise AgentLoopError(
                f"Command failed with exit {returncode}: {' '.join(cmd)}\n"
                f"log: {log_path if not capture_diagnostics else '(capture pathname unavailable)'}\n\nlast output:\n{tail_text(full_output or output)}"
            )
        return result

    def _run_with_log_pty(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        log_path: Path,
        label: str,
        progress_interval_seconds: int,
        check: bool,
        env: Mapping[str, str] | None,
        input_text: str | None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        """Run a command attached to a pseudo-terminal, logging and capturing output.

        Some agents (notably Antigravity's `agy`) detect whether stdout is a TTY
        and silently drop their final response when it is not (a pipe / file /
        subprocess), see upstream antigravity-cli issue #76. Allocating a PTY makes
        the agent emit normally; we strip the resulting ANSI control sequences from
        the captured text before returning it. stdin/stdout/stderr all share the
        PTY slave, so returned stderr is always empty and stdout is the merged stream.
        """
        import pty
        import select

        if input_text is not None:
            raise ValueError("stdin text is not supported for PTY-backed commands")

        started = time.monotonic()
        deadline = started + timeout_seconds if timeout_seconds is not None else None
        next_progress = started + progress_interval_seconds
        header = f"$ {' '.join(cmd)}\n\n"
        chunks: list[bytes] = []
        with log_path.open("wb") as log_file:
            log_file.write(header.encode("utf-8"))
            log_file.flush()
            allocated_fds: tuple[int, int] | None = None

            def spawn_pty() -> subprocess.Popen[bytes]:
                nonlocal allocated_fds
                master_fd, slave_fd = pty.openpty()
                allocated_fds = (master_fd, slave_fd)
                try:
                    return subprocess.Popen(
                        cmd,
                        cwd=cwd,
                        stdin=slave_fd,
                        stdout=slave_fd,
                        stderr=slave_fd,
                        close_fds=True,
                        start_new_session=True,
                        env={**os.environ, **env} if env is not None else None,
                    )
                except BaseException:
                    os.close(master_fd)
                    os.close(slave_fd)
                    allocated_fds = None
                    raise

            proc = self._spawn_with_retry(cmd, spawn_pty)
            spawn_wall_time = time.time()
            spawn_monotonic = time.monotonic()
            before_identity = executable_identity(cmd[0])
            assert allocated_fds is not None
            master_fd, slave_fd = allocated_fds
            os.close(slave_fd)
            self._register_active_process(proc)
            try:
                timed_out = False
                while True:
                    now = time.monotonic()
                    if deadline is not None and now >= deadline and proc.poll() is None:
                        timed_out = True
                        self._terminate_process_group(proc)
                    if timed_out:
                        wait_seconds = 0.1
                    elif deadline is not None:
                        wait_seconds = min(1.0, max(0.0, deadline - now))
                    else:
                        wait_seconds = 1.0
                    ready, _, _ = select.select([master_fd], [], [], wait_seconds)
                    if master_fd in ready:
                        try:
                            data = os.read(master_fd, 65536)
                        except OSError:
                            data = b""  # EIO once the slave side has fully closed
                        if data:
                            log_file.write(data)
                            log_file.flush()
                            chunks.append(data)
                        elif proc.poll() is not None:
                            break
                    elif proc.poll() is not None:
                        break
                    now = time.monotonic()
                    if now >= next_progress:
                        elapsed = int(now - started)
                        print(
                            f"[agent-loop {datetime.now().strftime('%H:%M:%S')}] "
                            f"{label} still running ({elapsed}s); log: {log_path}",
                            file=sys.stderr,
                            flush=True,
                        )
                        next_progress = now + progress_interval_seconds
            except KeyboardInterrupt:
                # The child runs in its own session and does not receive the
                # terminal SIGINT; kill its whole process group instead.
                self._terminate_process_group(proc)
                raise
            finally:
                self._unregister_active_process(proc)
                os.close(master_fd)
            returncode = None if timed_out else proc.wait()

        raw = b"".join(chunks).decode("utf-8", errors="replace")
        output = strip_ansi(raw)
        exited = time.monotonic()
        result = CommandResult(cmd, cwd, output, "", returncode, ExecutionObservation(
            spawn_wall_time, spawn_monotonic, exited, exited - spawn_monotonic,
            before_identity, executable_identity(cmd[0]), self._interrupted,
        ))
        if check and returncode != 0:
            raise AgentLoopError(
                f"Command failed with exit {returncode}: {' '.join(cmd)}\n"
                f"log: {log_path}\n\nlast output:\n{tail_text(output)}"
            )
        return result
