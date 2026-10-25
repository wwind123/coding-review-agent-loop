"""PR-local dependency environments for per-run worktrees (#1652, parent #1630).

With the operator flag ``--worktree-setup-pr-local``, a per-run worktree's
``--worktree-link`` paths can be replaced by a *private* environment that the
orchestrator builds inside the worktree: every link path is removed without
being followed, then the operator ``--worktree-setup`` command runs with the
worktree as its working directory.  A coder never builds it, so the private
environment always equals setup(command, key files at HEAD).

* A writable worktree is privatized before its first agent turn, so no install
  during a turn can write through a link into the shared clone.
* A read-only worktree (sandboxed run, CLI-enforced read-only roles only)
  keeps its shared links until its committed HEAD changes a
  ``--worktree-setup-key`` digest.
* An already private environment is rebuilt whenever the HEAD digest differs
  from the digest it was built for.

Reconciling runs only at orchestrator-owned points outside agent turns and
always before the baseline capture (see ``checkout_verification``).  Setup only
ever consumes committed bytes: any uncommitted or index-hidden tracked change,
and any untracked file shadowing a key that is absent at HEAD, defers the
rebuild without removing anything.
"""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import checkout_verification as cv
from .errors import AgentLoopError, CheckoutVerificationError
from .logging import log

_DEFAULT_SETUP_TIMEOUT_SECONDS = 1800

# Lexical worktree path -> note for the next coder prompt after a rebuild.
_REBUILD_NOTES: dict[Path, str] = {}


@dataclass(frozen=True)
class Privatization:
    path: Path
    names: tuple[str, ...]
    changed_keys: tuple[str, ...]
    head: str
    digest: str
    generation: int


def pr_local_enabled(config: Any) -> bool:
    return bool(getattr(config, "worktree_setup_pr_local", False))


def _redirected(path: Path) -> AgentLoopError:
    return AgentLoopError(
        f"Assigned checkout {path} is no longer the directory agent-loop prepared: it was "
        "renamed, removed, or replaced (for example by a symlink to another checkout). "
        "Nothing was removed and no PR-local setup ran; nothing in the replacement was touched"
    )


def _validated_root(path: Path, registration: Any) -> Path:
    """The recorded canonical alias, after checking the root is still the prepared directory."""
    canonical = Path(os.path.realpath(os.fspath(path)))
    try:
        found = cv.lookup_worktree_links(Path(os.path.abspath(os.fspath(path))), canonical)
    except AgentLoopError as exc:
        raise _redirected(path) from exc
    if found is None or found.lexical != registration.lexical:
        raise _redirected(path)
    with cv._LEDGER_LOCK:
        recorded = cv._ROOT_IDENTITY.get(cv._key(path))
    if recorded is not None and cv._root_identity(path) != recorded:
        raise _redirected(path)
    return Path(registration.alias)


def _open_root(alias: Path, identity: tuple[int, int] | None, path: Path) -> int:
    try:
        root_fd = os.open(os.fspath(alias), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise _redirected(path) from exc
    info = os.fstat(root_fd)
    if identity is None or (info.st_dev, info.st_ino) != tuple(identity):
        os.close(root_fd)
        raise _redirected(path)
    return root_fd


def _remove_entries(alias: Path, identity: tuple[int, int] | None, names: list[bytes], path: Path) -> None:
    """Remove each path descriptor-relative: unlink symlinks and files, fd-rmtree directories."""
    root_fd = _open_root(alias, identity, path)
    try:
        for raw in names:
            parts = [part for part in raw.split(b"/") if part]
            opened: list[int] = []
            dir_fd = root_fd
            try:
                missing = False
                for component in parts[:-1]:
                    try:
                        dir_fd = os.open(
                            component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd
                        )
                    except FileNotFoundError:
                        missing = True
                        break
                    except OSError as exc:
                        raise AgentLoopError(
                            f"{os.fsdecode(raw)} in {path} is reached through a symlinked or "
                            "non-directory ancestor; refusing to remove it"
                        ) from exc
                    opened.append(dir_fd)
                if missing:
                    continue
                name = parts[-1]
                try:
                    info = os.lstat(name, dir_fd=dir_fd)
                except FileNotFoundError:
                    continue
                if stat.S_ISLNK(info.st_mode) or stat.S_ISREG(info.st_mode):
                    # Never follows the link: its target is left untouched.
                    os.unlink(name, dir_fd=dir_fd)
                elif stat.S_ISDIR(info.st_mode):
                    shutil.rmtree(name, dir_fd=dir_fd)
                else:
                    raise AgentLoopError(
                        f"{os.fsdecode(raw)} in {path} is an unsupported file type; refusing to remove it"
                    )
            finally:
                for fd in opened:
                    os.close(fd)
    finally:
        os.close(root_fd)


def _refuse_tracked(config: Any, runner: Any, alias: Path, names: list[bytes]) -> None:
    for raw in names:
        try:
            cv._check_untracked(config, runner, alias, os.fsdecode(raw), "worktree link")
        except cv._Incomplete as exc:
            raise AgentLoopError(f"refusing PR-local setup: {exc}") from exc


def _uncommitted_inputs(alias: Path, entries: dict, *, runner: Any) -> list[str]:
    """Every tracked change Git could hide from a HEAD-only setup, plus shadowed absent keys."""
    from .worktree_setup import _untracked_absent_keys, store_tracked_tree_dirty_paths

    dirty = store_tracked_tree_dirty_paths(alias)
    shadows = [
        f"{key} (untracked or ignored, absent at HEAD)"
        for key in _untracked_absent_keys(alias, entries, runner=runner, tracked_at_head=False)
    ]
    return sorted(dirty) + shadows


def _entry_pairs(entries: dict) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted((key, ":".join(entry) if entry else "absent") for key, entry in entries.items())
    )


def _head(runner: Any, alias: Path) -> str:
    result = runner.run(("git", "rev-parse", "--verify", "HEAD"), cwd=alias, check=False)
    sha = (result.stdout or "").strip()
    if result.returncode != 0 or not sha:
        raise AgentLoopError(f"could not resolve HEAD in {alias} for the PR-local environment")
    return sha


def reconcile_pr_local_environment(config: Any, runner: Any, path: Path) -> Privatization | None:
    """Privatize or rebuild ``path``'s environment when required; None when nothing ran.

    A no-op unless the flag is on, ``path`` is a registered per-run worktree of
    this run, and the runner is not dry-run.  Raises AgentLoopError on refusal
    or setup failure; callers poison or refuse.
    """
    if not pr_local_enabled(config) or getattr(runner, "dry_run", False):
        return None
    registration = cv.link_registration(path)
    if registration is None or not registration.pr_local:
        return None
    from .config import run_worktree_store

    store = run_worktree_store(config, Path(path))
    if store is None:
        return None
    alias = _validated_root(path, registration)
    command = config.worktree_setup
    keys = tuple(config.worktree_setup_keys)
    from .worktree_setup import run_setup_command, setup_digest

    head = _head(runner, alias)
    digest, entries = setup_digest(alias, head, command=command, keys=keys, runner=runner)
    initial = registration.writable and bool(registration.links)
    if registration.private:
        reference = sorted({value[0] for value in registration.private.values()})
        reference_digest = reference[0] if len(reference) == 1 else None
    else:
        reference_digest = registration.base_digest
    if not initial and digest == reference_digest:
        cv.set_environment_deferred(path, ())
        return None
    offending = _uncommitted_inputs(alias, entries, runner=runner)
    if offending:
        cv.set_environment_deferred(path, tuple(offending))
        log(
            config,
            f"PR-local dependency environment: deferred the rebuild in {path} because tracked "
            f"paths are uncommitted or hidden: {', '.join(offending[:8])}; nothing was removed.",
        )
        return None
    names = sorted(set(registration.links) | set(registration.private))
    _refuse_tracked(config, runner, alias, names)
    alias = _validated_root(path, registration)
    _remove_entries(alias, registration.identity, names, Path(path))
    timeout = int(
        getattr(config, "coder_test_command_timeout_seconds", 0) or _DEFAULT_SETUP_TIMEOUT_SECONDS
    )
    display = ", ".join(os.fsdecode(name) for name in names)
    log(config, f"Running --worktree-setup in the run worktree {path} at {head} for {display}")
    run_setup_command(
        alias, command, timeout_seconds=timeout,
        where=f"the run worktree {path}", consequence="no private environment was registered",
    )
    alias = _validated_root(path, registration)
    identities: dict[bytes, tuple[int, int]] = {}
    for raw in names:
        name = os.fsdecode(raw)
        info = cv.private_entry_stat(alias, raw)
        if info is None:
            raise AgentLoopError(
                f"--worktree-setup succeeded in the run worktree {path} but did not create {name} "
                "there (or it is reached through a symlink); no private environment was registered"
            )
        if stat.S_ISLNK(info.st_mode):
            raise AgentLoopError(
                f"--worktree-setup left {name} in the run worktree {path} as a symlink; a private "
                "environment must be a real directory or file, so it was refused"
            )
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise AgentLoopError(
                f"--worktree-setup left {name} in the run worktree {path} as an unsupported file type"
            )
        identities[raw] = (info.st_dev, info.st_ino)
    new_entries = _entry_pairs(entries)
    previous = dict(registration.key_entries)
    changed = tuple(key for key, value in new_entries if previous.get(key) != value)
    generation = cv.mark_private(path, identities, digest, key_entries=new_entries)
    shown_keys = ", ".join(changed) if changed else "none (initial private setup of a writable worktree)"
    log(
        config,
        f"PR-local dependency environment: rebuilt {display} privately in {path} for changed "
        f"setup keys {shown_keys} at {head} (digest {digest[:12]}); the shared clone {store} was "
        "not modified.",
    )
    with cv._LEDGER_LOCK:
        _REBUILD_NOTES[cv._key(path)] = (
            f"agent-loop rebuilt {display} privately in this worktree for HEAD {head} "
            f"(environment generation {generation})."
        )
    return Privatization(
        path=Path(path), names=tuple(os.fsdecode(name) for name in names), changed_keys=changed,
        head=head, digest=digest, generation=generation,
    )


def take_rebuild_note(path: Path) -> str | None:
    """The rebuild note for the next coder prompt, consumed once."""
    with cv._LEDGER_LOCK:
        return _REBUILD_NOTES.pop(cv._key(path), None)


def reset_rebuild_notes() -> None:
    with cv._LEDGER_LOCK:
        _REBUILD_NOTES.clear()


def worktree_writable(config: Any, agent: Any) -> bool:
    """Whether ``agent``'s worktree may be written during a turn of this run.

    Only a sandboxed run whose agent holds nothing but CLI-enforced read-only
    roles is read-only; every other worktree is writable.
    """
    from .agent_permissions import is_sandboxed

    if not is_sandboxed(config):
        return True
    coders = {getattr(config, "coder", None), getattr(config, "implementation_coder", None)}
    return agent in {value for value in coders if value is not None}


def refuse_linked_turn(config: Any, path: Path, *, role: str | None, agent: Any) -> None:
    """Pre-turn backstop: no writing turn may start while links into the shared clone remain."""
    if not pr_local_enabled(config):
        return
    registration = cv.link_registration(path)
    if registration is None or not registration.links:
        return
    from .agent_permissions import PERMISSION_CLASS_READ_ONLY, is_sandboxed, permission_class_for_role

    read_only_turn = is_sandboxed(config) and permission_class_for_role(role) == PERMISSION_CLASS_READ_ONLY
    if registration.writable or not read_only_turn:
        links = ", ".join(sorted(os.fsdecode(name) for name in registration.links))
        raise CheckoutVerificationError(
            f"Refusing to start the {agent} {role or 'agent'} turn in {path}: with "
            f"--worktree-setup-pr-local a turn that can write must not run while {links} still "
            "link into the shared clone, and agent-loop has not rebuilt them privately yet."
        )
