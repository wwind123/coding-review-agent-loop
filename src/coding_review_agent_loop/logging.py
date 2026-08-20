"""Console logging helpers."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]

_LEASES: dict[Path, object] = {}
_RETENTION_SECONDS = 14 * 24 * 60 * 60
_MAX_LEASES = 32
_CAPTURE_MARKER = ".agent-loop-capture"

if TYPE_CHECKING:
    from .config import AgentLoopConfig


def datetime_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _prepare_capture_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / _CAPTURE_MARKER).touch(exist_ok=True)
    # Windows has no fcntl module.  Capture directories are still usable there;
    # only the optional cross-process lease and age pruning are unavailable.
    if fcntl is None:
        return
    if root not in _LEASES:
        lease = (root / ".lease").open("a+")
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lease.close()
        else:
            _LEASES[root] = lease
            while len(_LEASES) > _MAX_LEASES:
                old_root, old_lease = next(iter(_LEASES.items()))
                if old_root == root:
                    break
                _LEASES.pop(old_root)
                old_lease.close()
            if root.parent.parent.name == "subprocess-logs":
                _prune_capture_roots(root.parent, root)


def _prune_capture_roots(parent: Path, current: Path) -> None:
    """Best-effort age cleanup; live invocation leases are never removed."""
    if fcntl is None:
        return
    try:
        candidates = list(parent.iterdir())
    except OSError:
        return
    cutoff = time.time() - _RETENTION_SECONDS
    for candidate in candidates:
        if not candidate.is_dir() or candidate == current:
            continue
        if not (candidate / _CAPTURE_MARKER).is_file():
            continue
        try:
            if candidate.stat().st_mtime >= cutoff:
                continue
            lease = (candidate / ".lease").open("a+")
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                lease.close()
                continue
            for item in candidate.iterdir():
                if item.is_file():
                    item.unlink(missing_ok=True)
            lease.close()
            candidate.rmdir()
        except OSError:
            continue


def log(config: AgentLoopConfig, message: str) -> None:
    if config.quiet:
        return
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[agent-loop {now}] {message}", file=sys.stderr, flush=True)


def new_run_id() -> str:
    return datetime_stamp()


def agent_log_path(
    config: AgentLoopConfig,
    agent: str,
    *,
    run_id: str | None = None,
    label: str | None = None,
    attempt_suffix: str | None = None,
) -> Path:
    stamp = datetime_stamp()
    prefix = f"{run_id}-" if run_id else ""
    suffix = f"-{label}" if label else ""
    attempt = f"-{attempt_suffix}" if attempt_suffix else ""
    root = config.subprocess_log_dir or config.log_dir
    _prepare_capture_root(root)
    return root / f"{prefix}{stamp}-{agent}{suffix}{attempt}.log"


def run_usage_summary_path(config: AgentLoopConfig, run_id: str) -> Path:
    return config.log_dir / f"{run_id}-usage-summary.json"
