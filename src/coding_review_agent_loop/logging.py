"""Console logging helpers."""

from __future__ import annotations

import os
import shutil
import stat
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
_MAX_LEASES = 32
_CAPTURE_MARKER = ".agent-loop-capture"
_LEASE_NAME = ".lease"

if TYPE_CHECKING:
    from .config import AgentLoopConfig


def datetime_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _prepare_capture_root(root: Path, *, retention_days: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / _CAPTURE_MARKER).touch(exist_ok=True)
    # Windows has no fcntl module.  Capture directories are still usable there;
    # only the optional cross-process lease and age pruning are unavailable.
    if fcntl is None:
        return
    if root not in _LEASES:
        lease = (root / _LEASE_NAME).open("a+")
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
            # Retention 0 (``never``) disables pruning entirely (#1572).
            if retention_days > 0 and root.parent.parent.name == "subprocess-logs":
                _prune_capture_roots(root.parent, root, retention_days=retention_days)


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _prune_capture_roots(parent: Path, current: Path, *, retention_days: int) -> None:
    """Best-effort age cleanup; live invocation leases are never removed.

    Only real (non-symlink) directories with a regular capture marker are
    candidates, and nothing is followed through a symlink.  An expired root is
    removed whole, including subdirectories; the marker and lease go last and
    only after every other entry is gone, so a partial failure leaves the root
    marked, still expired and retried by a later sweep (#1572).
    """
    if fcntl is None or retention_days <= 0:
        return
    try:
        candidates = list(parent.iterdir())
    except OSError:
        return
    cutoff = time.time() - retention_days * 24 * 60 * 60
    for candidate in candidates:
        if candidate == current:
            continue
        try:
            st = candidate.lstat()
            if not stat.S_ISDIR(st.st_mode) or st.st_mtime >= cutoff:
                continue
            marker = candidate / _CAPTURE_MARKER
            lease_path = candidate / _LEASE_NAME
            if not _is_regular_file(marker):
                continue
            if os.path.lexists(lease_path) and not _is_regular_file(lease_path):
                continue
            lease = lease_path.open("a+")
            try:
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    continue
                if not _remove_capture_contents(candidate):
                    try:
                        os.utime(candidate, (st.st_atime, st.st_mtime), follow_symlinks=False)
                    except OSError:
                        pass
                    continue
                marker.unlink(missing_ok=True)
                lease_path.unlink(missing_ok=True)
            finally:
                lease.close()
            candidate.rmdir()
        except OSError:
            continue


def _remove_capture_contents(candidate: Path) -> bool:
    """Remove everything except the control files; True when nothing failed."""
    failed = False
    try:
        items = list(candidate.iterdir())
    except OSError:
        return False
    for item in items:
        if item.name in (_CAPTURE_MARKER, _LEASE_NAME):
            continue
        try:
            if stat.S_ISDIR(item.lstat().st_mode):
                shutil.rmtree(item)
            else:
                item.unlink(missing_ok=True)
        except OSError:
            failed = True
    return not failed


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
    _prepare_capture_root(root, retention_days=config.subprocess_log_retention_days)
    return root / f"{prefix}{stamp}-{config.active_reviewer_seat_id or agent}{suffix}{attempt}.log"


def run_usage_summary_path(config: AgentLoopConfig, run_id: str) -> Path:
    return config.log_dir / f"{run_id}-usage-summary.json"
