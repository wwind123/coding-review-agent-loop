"""Run-scoped claims on agent checkouts (#1127).

Two concurrent runs that resolve to the same agent checkout would otherwise
reset, clean and commit over each other.  Each run claims every checkout it
uses (``required_agents``) before any clone, reset, clean or dispatch.  A claim
is a ``flock`` on a lock file under a private per-user host lock root, outside
every checkout, plus atomic holder metadata.  A free ``flock`` is the only
liveness test, so a crashed holder never blocks a later run.

Claims belong to a logical-run :class:`ClaimOwner` created by the outermost
:func:`workdir_claim_scope`; nested scopes join it and only the outermost scope
releases.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import json
import os
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .errors import AgentLoopError
from .test_workers import _write_reservation, host_lock_root

CLAIM_LOCK_DIR = "workdir-claims"
_METADATA_READ_TRIES = 3
_METADATA_READ_DELAY_SECONDS = 0.05


class WorkdirClaimedError(AgentLoopError):
    """A required agent checkout is claimed by another live run."""


@dataclass(frozen=True)
class ClaimOwner:
    run_id: str
    command: str
    number: int | None
    target: str


@dataclass
class _Claim:
    owner: ClaimOwner
    handle: Any
    lock_path: Path
    meta_path: Path
    metadata: dict


_claim_root_override: Path | None = None
_registry: dict[str, _Claim] = {}
_registry_lock = threading.Lock()
_active_owner: ContextVar[ClaimOwner | None] = ContextVar("workdir_claim_owner", default=None)


def _set_claim_root_for_tests(path: Path | None) -> None:
    """Test-only override of the claim root; also clears the in-process registry."""
    global _claim_root_override
    _claim_root_override = path
    with _registry_lock:
        _registry.clear()


def claim_root() -> Path:
    if _claim_root_override is not None:
        _claim_root_override.mkdir(parents=True, exist_ok=True)
        return _claim_root_override
    return host_lock_root(CLAIM_LOCK_DIR)


def _claim_key(path: Path) -> str:
    return hashlib.sha256(os.path.realpath(path).encode("utf-8")).hexdigest()


_NUMBERED_LABELS = {"issue": "issue", "pr": "PR", "discuss": "discuss"}
_UNNUMBERED_COMMANDS = {"task", "managed-pr", "library"}


def _target_for(command: str, number: int | None, head: str | None) -> str:
    """Validate an owner identity and render its target.

    Numbered modes require a positive number; task, managed-pr and library
    require none; only managed-pr takes a head branch.
    """
    if command in _NUMBERED_LABELS:
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise AgentLoopError(f"workdir claim scope for {command!r} requires a positive number.")
    elif command in _UNNUMBERED_COMMANDS:
        if number is not None:
            raise AgentLoopError(f"workdir claim scope for {command!r} must not carry a number.")
    else:
        raise AgentLoopError(f"workdir claim scope has unknown command {command!r}.")
    if command == "managed-pr":
        if not isinstance(head, str) or not head.strip():
            raise AgentLoopError("workdir claim scope for 'managed-pr' requires a head branch.")
        return f"managed-pr from {head}"
    if head is not None:
        raise AgentLoopError(f"workdir claim scope for {command!r} must not carry a head branch.")
    if command in _NUMBERED_LABELS:
        return f"{_NUMBERED_LABELS[command]} #{number}"
    return command


def current_claim_owner() -> ClaimOwner | None:
    return _active_owner.get()


def _flock_nonblocking(handle) -> bool:
    if os.name == "nt":  # pragma: no cover - best effort, matching test_workers
        return True
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _describe_holder(path: Path, lock_path: Path, agent: str, meta: dict | None) -> str:
    where = f"Agent checkout {path} ({agent}) is already claimed by another agent-loop run"
    if not meta:
        return (
            f"{where}: unidentified holder (holder metadata missing or unreadable); "
            f"lock file: {lock_path}. Wait for it to finish, or pass an explicit --{agent}-dir."
        )
    return (
        f"{where}: pid {meta.get('pid')}, run_id {meta.get('run_id')}, repo {meta.get('repo')}, "
        f"command {meta.get('command')}, target {meta.get('target')}, "
        f"started {meta.get('started_at')}; lock file: {lock_path}. "
        f"Wait for it to finish, or pass an explicit --{agent}-dir."
    )


def _read_metadata(meta_path: Path) -> dict | None:
    for attempt in range(_METADATA_READ_TRIES):
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("pid") is not None and raw.get("run_id"):
                return raw
        except (OSError, ValueError):
            pass
        if attempt + 1 < _METADATA_READ_TRIES:
            time.sleep(_METADATA_READ_DELAY_SECONDS)
    return None


def acquire_workdir_claim(path: Path, *, agent: str, repo: str) -> bool:
    """Claim ``path`` for the active owner; return True when newly taken.

    Re-acquisition by the same owner is a no-op (False).  Any other holder,
    in this process or another, raises :class:`WorkdirClaimedError`.
    """
    owner = _active_owner.get()
    if owner is None:
        raise AgentLoopError("acquire_workdir_claim requires an active workdir_claim_scope().")
    key = _claim_key(path)
    root = claim_root()
    lock_path = root / f"{key}.lock"
    meta_path = root / f"{key}.json"
    with _registry_lock:
        existing = _registry.get(key)
        if existing is not None:
            if existing.owner == owner:
                return False
            raise WorkdirClaimedError(_describe_holder(path, lock_path, agent, existing.metadata))
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        handle = os.fdopen(descriptor, "r+")
        try:
            os.set_inheritable(descriptor, False)
        except OSError:  # pragma: no cover
            pass
        if not _flock_nonblocking(handle):
            handle.close()
            raise WorkdirClaimedError(
                _describe_holder(path, lock_path, agent, _read_metadata(meta_path))
            )
        metadata = {
            "pid": os.getpid(),
            "run_id": owner.run_id,
            "repo": repo,
            "command": owner.command,
            "number": owner.number,
            "target": owner.target,
            "agent": agent,
            "path": str(path),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            _write_reservation(meta_path, metadata)
        except OSError:
            handle.close()
            raise
        _registry[key] = _Claim(owner, handle, lock_path, meta_path, metadata)
        return True


def _release_claim(key: str) -> None:
    claim = _registry.pop(key, None)
    if claim is None:
        return
    try:
        # Metadata cleanup happens while the flock is still held so a
        # contender that acquires right after unlock keeps its own record.
        try:
            raw = json.loads(claim.meta_path.read_text(encoding="utf-8"))
            if (
                isinstance(raw, dict)
                and raw.get("pid") == os.getpid()
                and raw.get("run_id") == claim.owner.run_id
            ):
                claim.meta_path.unlink()
        except (OSError, ValueError):
            pass
        if os.name != "nt":
            import fcntl

            with contextlib.suppress(OSError):
                fcntl.flock(claim.handle.fileno(), fcntl.LOCK_UN)
    finally:
        claim.handle.close()


def release_workdir_claims(owner: ClaimOwner) -> None:
    with _registry_lock:
        for key in [k for k, c in _registry.items() if c.owner == owner]:
            _release_claim(key)


def _agent_paths(config) -> list[tuple[str, Path]]:
    from .config import required_agents

    paths = {
        "claude": config.claude_dir,
        "codex": config.codex_dir,
        "gemini": config.gemini_dir,
        "antigravity": config.antigravity_dir,
    }
    seen: set[str] = set()
    result: list[tuple[str, Path]] = []
    for agent in sorted(required_agents(config)):
        path = paths[agent]
        key = _claim_key(path)
        if key in seen:
            continue
        seen.add(key)
        result.append((agent, path))
    return result


def claim_agent_workdirs(config) -> None:
    """Claim every required agent checkout, all-or-nothing, in agent-name order."""
    taken: list[str] = []
    try:
        for agent, path in _agent_paths(config):
            if acquire_workdir_claim(path, agent=agent, repo=config.repo):
                taken.append(_claim_key(path))
    except BaseException:
        with _registry_lock:
            for key in taken:
                _release_claim(key)
        raise


@contextlib.contextmanager
def workdir_claim_scope(
    command: str | None = None, number: int | None = None, head: str | None = None
) -> Iterator[ClaimOwner]:
    """Open (or join) the logical run that owns workdir claims.

    Only the outermost scope creates the owner and releases its claims; a
    nested scope joins the active owner and ignores its own arguments.
    """
    active = _active_owner.get()
    if active is not None:
        yield active
        return
    command = command or "library"
    owner = ClaimOwner(
        run_id=uuid.uuid4().hex,
        command=command,
        number=number,
        target=_target_for(command, number, head),
    )
    token = _active_owner.set(owner)
    try:
        yield owner
    finally:
        try:
            release_workdir_claims(owner)
        finally:
            _active_owner.reset(token)


def claimed_run(command: str, number_param: str | None = None) -> Callable:
    """Decorate a ``run_*`` loop: open a scope and claim the config's workdirs first."""

    def decorator(func: Callable) -> Callable:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            number = bound.arguments.get(number_param) if number_param else None
            with workdir_claim_scope(command=command, number=number):
                claim_agent_workdirs(bound.arguments["config"])
                return func(*args, **kwargs)

        return wrapper

    return decorator
