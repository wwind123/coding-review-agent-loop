"""Subprocess execution helpers for the agent loop."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from .errors import AgentLoopError


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    cwd: Path
    stdout: str
    stderr: str
    returncode: int


# Matches ANSI/VT100 control sequences (CSI, OSC, charset selection) that a
# program emits when it believes it is talking to a terminal. We run some agents
# under a pseudo-terminal (see run_with_log use_pty), so we strip these from the
# captured output before parsing it.
_ANSI_RE = re.compile(r"\x1B\[[0-9;?]*[ -/]*[@-~]|\x1B\][^\x07\x1B]*(?:\x07|\x1B\\)|\x1B[()][AB0-2]")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text).replace("\r", "")


def tail_text(text: str, *, max_lines: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def ensure_log_dir_ignored(log_dir: Path) -> None:
    gitignore = log_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")


class Runner:
    def __init__(self, *, dry_run: bool = False):
        self.dry_run = dry_run

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
        use_pty: bool = False,
    ) -> CommandResult:
        cmd = [str(a) for a in args]
        if self.dry_run:
            print(f"[dry-run] ({cwd}) {' '.join(cmd)}")
            return CommandResult(cmd, cwd, "", "", 0)

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
            )
        started = time.monotonic()
        next_progress = started + progress_interval_seconds
        header = f"$ {' '.join(cmd)}\n\n"
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(header)
            log_file.flush()
            # stderr=subprocess.STDOUT merges stderr into the log file.
            # All agent backends (Claude, Codex, Gemini) use run_with_log,
            # so stderr capture is uniform across them (issue #266).
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, **env} if env is not None else None,
            )
            try:
                while True:
                    returncode = proc.poll()
                    if returncode is not None:
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
                    time.sleep(1)
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise

        full_output = log_path.read_text(encoding="utf-8")
        output = full_output[len(header):] if full_output.startswith(header) else full_output
        result = CommandResult(cmd, cwd, output, "", returncode)
        if check and returncode != 0:
            raise AgentLoopError(
                f"Command failed with exit {returncode}: {' '.join(cmd)}\n"
                f"log: {log_path}\n\nlast output:\n{tail_text(full_output)}"
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
    ) -> CommandResult:
        """Run a command attached to a pseudo-terminal, logging and capturing output.

        Some agents (notably Antigravity's `agy`) detect whether stdout is a TTY
        and silently drop their final response when it is not (a pipe / file /
        subprocess), see upstream antigravity-cli issue #76. Allocating a PTY makes
        the agent emit normally; we strip the resulting ANSI control sequences from
        the captured text before returning it.
        """
        import pty
        import select

        started = time.monotonic()
        next_progress = started + progress_interval_seconds
        header = f"$ {' '.join(cmd)}\n\n"
        master_fd, slave_fd = pty.openpty()
        chunks: list[bytes] = []
        with log_path.open("wb") as log_file:
            log_file.write(header.encode("utf-8"))
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                start_new_session=True,
                env={**os.environ, **env} if env is not None else None,
            )
            os.close(slave_fd)
            try:
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 1.0)
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
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
            finally:
                os.close(master_fd)
            returncode = proc.wait()

        raw = b"".join(chunks).decode("utf-8", errors="replace")
        output = strip_ansi(raw)
        result = CommandResult(cmd, cwd, output, "", returncode)
        if check and returncode != 0:
            raise AgentLoopError(
                f"Command failed with exit {returncode}: {' '.join(cmd)}\n"
                f"log: {log_path}\n\nlast output:\n{tail_text(output)}"
            )
        return result
