"""Subprocess execution helpers for the agent loop."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .errors import AgentLoopError


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    cwd: Path
    stdout: str
    stderr: str
    returncode: int


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
    ) -> CommandResult:
        cmd = [str(a) for a in args]
        if self.dry_run:
            print(f"[dry-run] ({cwd}) {' '.join(cmd)}")
            return CommandResult(cmd, cwd, "", "", 0)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_log_dir_ignored(log_path.parent)
        started = time.monotonic()
        next_progress = started + progress_interval_seconds
        header = f"$ {' '.join(cmd)}\n\n"
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(header)
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
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
