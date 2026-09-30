"""Best-effort record of which agent-loop commit a process runs (#1111).

The capture never raises.  It validates the Git executable's location, then
reads the tool checkout only through ``inspect_tool.run_hardened_git`` under
one overall deadline.  Movement of the checkout after capture is not detected.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from . import inspect_tool
from .agent_permissions import executable_location_refusal
from .config import AgentLoopConfig
from .logging import log

PROCESS_STARTED_AT = datetime.now(UTC).isoformat()
_PROCESS_PROVENANCE: dict | None = None

DEFAULT_DEADLINE_SECONDS = 10.0
_PROBE_TIMEOUT_SECONDS = 5.0
_DIRTY_SAMPLE_LIMIT = 10


class _Stop(Exception):
    def __init__(self, error: str) -> None:
        super().__init__(error)
        self.error = error


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _empty(package_path: str) -> dict:
    return {
        "package_path": package_path,
        "checkout_root": None,
        "commit": None,
        "dirty": None,
        "dirty_paths_sample": [],
        "error": None,
        "captured_at": _now(),
        "process_started_at": PROCESS_STARTED_AT,
    }


def _make_bounded(executor, deadline: float) -> inspect_tool.Executor:
    def bounded(argv, env, cwd, capture):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _Stop("git-timeout")
        timeout = min(_PROBE_TIMEOUT_SECONDS, remaining)
        if executor is not None:
            return executor(argv, env, cwd, capture)
        completed = subprocess.run(
            list(argv),
            env=dict(env),
            cwd=cwd,
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=capture,
            check=False,
            timeout=timeout,
        )
        return inspect_tool.ExecResult(
            completed.returncode,
            completed.stdout if capture else b"",
            completed.stderr if capture else b"",
        )

    return bounded


def capture_tool_provenance(
    config: AgentLoopConfig,
    *,
    executor: inspect_tool.Executor | None = None,
    which: Callable[[str], str | None] = shutil.which,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
) -> dict:
    """Capture the loaded package's checkout HEAD and cleanliness; never raises."""
    package_path = str(Path(__file__).resolve().parent)
    result = _empty(package_path)
    try:
        _capture(config, result, executor, which, deadline_seconds)
    except _Stop as stop:
        result["error"] = stop.error
    except subprocess.TimeoutExpired:
        result["error"] = "git-timeout"
    except inspect_tool.InspectRejected:
        result["error"] = "config-gate-refused"
        result["commit"] = None
        result["dirty"] = None
    except OSError:
        result["error"] = "git-unavailable"
    except Exception as exc:  # noqa: BLE001 - capture must never abort a run
        result["error"] = f"capture-failed: {type(exc).__name__}"
    result["captured_at"] = _now()
    return result


def _capture(config, result, executor, which, deadline_seconds) -> None:
    git = which("git")
    if not git:
        raise _Stop("git-unavailable")
    refusal = executable_location_refusal(git, config)
    if refusal:
        result["untrusted_git"] = git
        raise _Stop("git-untrusted-location")
    bounded = _make_bounded(executor, time.monotonic() + deadline_seconds)
    package_path = result["package_path"]

    top = inspect_tool.run_hardened_git(
        git, "rev-parse", ["--show-toplevel", "HEAD"], cwd=package_path, executor=bounded
    )
    lines = top.stdout.decode("utf-8", "replace").split("\n")
    if top.returncode != 0 or len(lines) < 2 or not lines[0] or not lines[1].strip():
        raise _Stop("not-a-git-checkout")
    root = lines[0].strip()
    candidate = lines[1].strip()
    result["checkout_root"] = root

    package_file = Path(__file__).resolve()
    init_file = package_file.parent / "__init__.py"
    try:
        tracked = [os.path.relpath(str(item), root) for item in (init_file, package_file)]
    except ValueError:
        raise _Stop("package-not-in-repository") from None
    member = inspect_tool.run_hardened_git(
        git, "ls-files", ["--error-unmatch", "--", *tracked], cwd=root, executor=bounded
    )
    if member.returncode != 0:
        raise _Stop("package-not-in-repository")
    result["commit"] = candidate

    try:
        status = inspect_tool.run_hardened_git(
            git, "status", ["--porcelain", "--untracked-files=normal"], cwd=root, executor=bounded
        )
    except (_Stop, subprocess.TimeoutExpired):
        result["error"] = "git-timeout"
        return
    except Exception:  # noqa: BLE001 - the commit is already verified; keep it
        result["error"] = "git-status-failed"
        return
    if status.returncode != 0:
        result["error"] = "git-status-failed"
        return
    entries = [line for line in status.stdout.decode("utf-8", "replace").split("\n") if line]
    result["dirty"] = bool(entries)
    result["dirty_paths_sample"] = entries[:_DIRTY_SAMPLE_LIMIT]
    result["dirty_count"] = len(entries)


def format_tool_provenance_line(p: dict) -> str:
    commit = p.get("commit")
    error = p.get("error") or "unknown"
    if not commit:
        line = f"agent-loop tool commit unknown ({error})"
        refused = p.get("untrusted_git")
        return f"{line}: refused {refused}" if refused else line
    root = p.get("checkout_root") or p.get("package_path")
    dirty = p.get("dirty")
    if dirty is False:
        return f"agent-loop tool commit {commit} (clean) from {root}"
    if dirty is True:
        count = p.get("dirty_count") or len(p.get("dirty_paths_sample") or [])
        return f"agent-loop tool commit {commit} (DIRTY: {count} changed paths) from {root}"
    return f"agent-loop tool commit {commit} (cleanliness unknown: {error}) from {root}"


def format_tool_commit_suffix(p: dict | None) -> str:
    commit = (p or {}).get("commit")
    if not commit:
        return "tool_commit=unknown"
    dirty = p.get("dirty")
    state = "clean" if dirty is False else "dirty" if dirty is True else "dirty=unknown"
    return f"tool_commit={commit[:7]} {state}"


def capture_process_provenance(config: AgentLoopConfig) -> dict:
    """Capture once per process, log one line (suppressed by ``--quiet``)."""
    global _PROCESS_PROVENANCE
    if _PROCESS_PROVENANCE is None:
        _PROCESS_PROVENANCE = capture_tool_provenance(config)
        log(config, format_tool_provenance_line(_PROCESS_PROVENANCE))
    return _PROCESS_PROVENANCE


def process_provenance(config: AgentLoopConfig) -> dict:
    """The cached capture, capturing (and logging) when the CLI did not."""
    return capture_process_provenance(config)
