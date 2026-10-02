"""Per-run linked worktrees over one shared clone per repository and agent (#1162).

The repo-scoped clone ``<scratch>/OWNER-REPO/<agent>/repo`` is the shared
*store*: it owns the object store and is cloned and fetched as before.  Each
CLI run with an omitted ``--<agent>-dir`` gets its own detached
``git worktree`` under ``<scratch>/OWNER-REPO/<agent>/runs/<run-token>``,
protected by the usual #1127 run claim and removed when the run ends.

Store mutations (clone, fetch+pin, local-base fast-forward, worktree add, prune
and remove) are serialized by a short, **non-reentrant** host flock that is
never held across an agent turn.  Every worktree created here has a durable
owner record under the private host lock root; removal and pruning delete only
a directory whose record still matches its creation identity.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import threading
from pathlib import Path
from typing import Any, Iterator

from .checkout_verification import _refuse_untrusted_root, forget_checkout
from .errors import AgentLoopError, CheckoutVerificationError
from .logging import log
from .scratch import make_private_dirs
from .test_workers import _write_reservation, host_lock_root
from .workdir_claims import probe_free_claim

STORE_LOCK_DIR = "workdir-stores"
RUN_TOKEN_RE = re.compile(r"^\d{8}-\d{6}-\d{6}-[0-9a-f]{12}$")

_store_lock_root_override: Path | None = None
_held = threading.local()


def _set_store_lock_root_for_tests(path: Path | None) -> None:
    global _store_lock_root_override
    _store_lock_root_override = path


def store_lock_root() -> Path:
    if _store_lock_root_override is not None:
        _store_lock_root_override.mkdir(parents=True, exist_ok=True)
        return _store_lock_root_override
    return host_lock_root(STORE_LOCK_DIR)


def store_key(store: Path) -> str:
    return hashlib.sha256(os.path.realpath(store).encode("utf-8")).hexdigest()


def _held_keys() -> set[str]:
    keys = getattr(_held, "keys", None)
    if keys is None:
        keys = _held.keys = set()
    return keys


def _require_lock_held(store: Path) -> None:
    if store_key(store) not in _held_keys():
        raise AgentLoopError(f"Internal error: the store lock for {store} must be held here.")


@contextlib.contextmanager
def store_lock(store: Path) -> Iterator[None]:
    """Blocking host flock for one store; nested acquisition fails fast."""
    key = store_key(store)
    held = _held_keys()
    if key in held:
        raise AgentLoopError(f"Internal error: the store lock for {store} is already held (not reentrant).")
    lock_path = store_lock_root() / f"{key}.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    handle = os.fdopen(descriptor, "r+")
    try:
        try:
            os.set_inheritable(descriptor, False)
        except OSError:  # pragma: no cover
            pass
        if os.name != "nt":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        held.add(key)
        try:
            yield
        finally:
            held.discard(key)
            if os.name != "nt":
                import fcntl

                with contextlib.suppress(OSError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


# --- owner records ---------------------------------------------------------


def _records_dir(store: Path) -> Path:
    path = store_lock_root() / store_key(store)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def _record_path(store: Path, token: str) -> Path:
    return _records_dir(store) / f"{token}.json"


def _read_record(store: Path, token: str) -> dict | None:
    try:
        raw = json.loads(_record_path(store, token).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _identity(path: Path) -> list[int] | None:
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return [info.st_dev, info.st_ino]


def _admin_dir_of(store: Path, path: Path) -> Path | None:
    """Return the admin dir iff ``path/.git`` and the admin dir link to each other."""
    git_file = path / ".git"
    try:
        if not stat.S_ISREG(os.lstat(git_file).st_mode):
            return None
        text = git_file.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not text.startswith("gitdir:"):
        return None
    admin = Path(text[len("gitdir:"):].strip())
    if not admin.is_absolute():
        admin = path / admin
    worktrees = Path(os.path.realpath(store / ".git" / "worktrees"))
    if Path(os.path.realpath(admin)).parent != worktrees:
        return None
    try:
        back = (admin / "gitdir").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if os.path.realpath(back) != os.path.realpath(git_file):
        return None
    return admin


def identify_owned_worktree(store: Path, runs_root: Path, path: Path) -> dict | None:
    """Return the matching ready owner record, or None for anything not positively ours."""
    path = Path(os.path.abspath(path))
    if path.parent != Path(os.path.abspath(runs_root)) or not RUN_TOKEN_RE.match(path.name):
        return None
    try:
        if not stat.S_ISDIR(os.lstat(path).st_mode):
            return None
    except OSError:
        return None
    admin = _admin_dir_of(store, path)
    if admin is None:
        return None
    record = _read_record(store, path.name)
    if (
        record is None
        or record.get("state") != "ready"
        or record.get("run_token") != path.name
        or record.get("path") != str(path)
        or record.get("root") != _identity(path)
        or record.get("admin_name") != admin.name
        or record.get("admin") != _identity(admin)
    ):
        return None
    return record


def _git(runner: Any, store: Path, *args: str, check: bool = True):
    return runner.run(("git", *args), cwd=store, check=check)


def add_run_worktree(
    store: Path, path: Path, sha: str, *, config: Any, runner: Any
) -> None:
    """Create a detached worktree at ``sha``; the store lock must be held."""
    _require_lock_held(store)
    path = Path(os.path.abspath(path))
    token = path.name
    if not RUN_TOKEN_RE.match(token):
        raise AgentLoopError(f"Refusing to create a run worktree with a non-token name: {path}")
    make_private_dirs(path.parent)
    record_file = _record_path(store, token)
    base = {"version": 1, "run_token": token, "path": str(path), "store": str(store)}
    _write_reservation(record_file, {**base, "state": "pending"})
    _git(runner, store, "worktree", "add", "--detach", str(path), sha)
    admin = _admin_dir_of(store, path)
    if admin is None:
        raise AgentLoopError(f"git worktree add did not produce a linked worktree at {path}.")
    _write_reservation(
        record_file,
        {**base, "state": "ready", "admin_name": admin.name,
         "root": _identity(path), "admin": _identity(admin)},
    )
    from .agent_permissions import register_checkout

    register_checkout(config, path)


def _remove_owned_locked(store: Path, path: Path, *, config: Any, runner: Any) -> bool:
    """Remove one identified worktree; the store lock must be held and is never taken here."""
    _require_lock_held(store)
    path = Path(os.path.abspath(path))
    runs_root = path.parent
    if identify_owned_worktree(store, runs_root, path) is None:
        log(config, f"Leaving run worktree entry untouched (no matching owner record): {path}")
        return False
    try:
        _refuse_untrusted_root(path, agent="run worktree", purpose="worktree removal")
    except CheckoutVerificationError as exc:
        log(config, f"Leaving run worktree untouched: {exc}")
        return False
    admin_name = str(record_admin_name(store, path))
    # A doubled --force also removes a worktree locked through git.
    _git(runner, store, "worktree", "remove", "--force", "--force", str(path), check=False)
    if os.path.lexists(path):
        # Re-verify: git may have failed halfway, so never delete what is no longer ours.
        if identify_owned_worktree(store, runs_root, path) is None:
            log(config, f"Leaving run worktree untouched (identity changed during removal): {path}")
            return False
        shutil.rmtree(path)
    _git(runner, store, "worktree", "prune", check=False)
    if _admin_registered(store, admin_name):
        # Keep the owner record: it is the evidence a later startup needs to finish cleanup.
        log(
            config,
            f"Run worktree {path} was removed but git still registers it ({admin_name}); "
            "keeping its owner record for a later prune. Run `git worktree unlock` if it is locked.",
        )
        return False
    with contextlib.suppress(OSError):
        _record_path(store, path.name).unlink()
    forget_checkout(path)
    return True


def record_admin_name(store: Path, path: Path) -> str:
    record = _read_record(store, Path(path).name) or {}
    return str(record.get("admin_name") or "")


def _admin_registered(store: Path, admin_name: str) -> bool:
    """True while git still holds the worktree's admin dir (fail closed on an unknown name)."""
    if not admin_name:
        return True
    return os.path.lexists(store / ".git" / "worktrees" / admin_name)


def remove_run_worktree(store: Path, path: Path, *, config: Any, runner: Any) -> bool:
    with store_lock(store):
        return _remove_owned_locked(store, path, config=config, runner=runner)


# --- store preparation -----------------------------------------------------


def _linked_worktrees(store: Path, runner: Any) -> list[dict[str, str]]:
    out = _git(runner, store, "worktree", "list", "--porcelain", check=False)
    if out.returncode != 0:
        raise AgentLoopError(
            f"Could not list the worktrees of store {store} (git exited {out.returncode}); "
            "leaving the store untouched."
        )
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in (out.stdout or "").splitlines():
        if not line.strip():
            if current:
                entries.append(current)
            current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    if current:
        entries.append(current)
    if not entries or "worktree" not in entries[0]:
        raise AgentLoopError(
            f"Unrecognized git worktree list output for store {store}; leaving the store untouched."
        )
    return entries


def _prepare_store_locked(store: Path, *, free: bool, config: Any, runner: Any) -> None:
    from .config import _is_stale_default_workdir, _validate_repo_remote

    has_git = store.is_dir() and (store / ".git").exists()
    if has_git:
        _git(runner, store, "worktree", "prune", check=False)
    if store.is_dir() and _is_stale_default_workdir(store):
        # Without a .git there can be no linked worktrees, so skip the enumeration.
        if has_git and len(_linked_worktrees(store, runner)) > 1:
            pass  # live linked worktrees: never delete the store under them
        elif not free:
            raise AgentLoopError(_held_store_message(store, "is stale and cannot be recreated"))
        else:
            log(config, f"Stale shared store detected; recreating: {store}")
            shutil.rmtree(store)
            forget_checkout(store)
    if not store.exists():
        if not free:
            raise AgentLoopError(_held_store_message(store, "is missing and cannot be cloned"))
        try:
            make_private_dirs(store.parent)
        except OSError as exc:
            raise AgentLoopError(f"Could not create parent directory for store at {store}: {exc}") from exc
        runner.run((config.gh_cmd, "repo", "clone", config.repo, str(store)), cwd=store.parent)
    git_check = _git(runner, store, "rev-parse", "--is-inside-work-tree", check=False)
    if git_check.returncode != 0 or git_check.stdout.strip() != "true":
        raise AgentLoopError(
            f"Shared store {store} exists but is not a git checkout. Remove it so it can be re-cloned."
        )
    _validate_repo_remote(store, label="Shared store", config=config, runner=runner)
    if free:
        detach = _git(runner, store, "checkout", "--detach", check=False)
        if detach.returncode != 0:
            log(config, f"Could not detach the shared store HEAD at {store}; leaving it as is.")


def _held_store_message(store: Path, what: str) -> str:
    return (
        f"Shared store {store} {what} while another agent-loop run (an explicit "
        "--<agent>-dir run or an older agent-loop) holds its claim. Wait for that run to "
        "finish and retry."
    )


def _fast_forward_base(store: Path, base: str, sha: str, *, config: Any, runner: Any) -> None:
    ref = f"refs/heads/{base}"
    for entry in _linked_worktrees(store, runner):
        if entry.get("branch") == ref:
            log(config, f"Local {base} is checked out in {entry.get('worktree')}; skipping its fast-forward.")
            return
    exists = _git(runner, store, "rev-parse", "--verify", "--quiet", ref, check=False)
    if exists.returncode == 0:
        ancestor = _git(runner, store, "merge-base", "--is-ancestor", ref, sha, check=False)
        if ancestor.returncode != 0:
            raise AgentLoopError(
                f"Local {base} in {store} has diverged from origin/{base}; cannot fast-forward it."
            )
    _git(runner, store, "branch", "-f", base, sha)


def _fetch_and_pin(store: Path, base: str, *, config: Any, runner: Any, free: bool) -> str:
    _git(runner, store, "fetch", "origin")
    sha = _git(runner, store, "rev-parse", f"refs/remotes/origin/{base}^{{commit}}").stdout.strip()
    if free:
        _fast_forward_base(store, base, sha, config=config, runner=runner)
    return sha


def prepare_store(store: Path, *, config: Any, runner: Any) -> str:
    """Clone/validate/fetch the store under the lock and return the pinned base SHA."""
    if not config.base:
        raise AgentLoopError(
            f"Base branch for {config.repo} was not resolved. Pass --base <branch> explicitly."
        )
    with store_lock(store):
        with probe_free_claim(store) as free:
            _prepare_store_locked(store, free=free, config=config, runner=runner)
            return _fetch_and_pin(store, config.base, config=config, runner=runner, free=free)


def pin_pr_head(store: Path, pr_number: int, *, config: Any, runner: Any) -> str:
    """Fetch the PR head under the lock and return its SHA (not the mutable ref)."""
    pr_ref = f"refs/remotes/origin/pr/{pr_number}"
    with store_lock(store):
        _git(runner, store, "fetch", "origin")
        _git(runner, store, "fetch", "origin", f"+pull/{pr_number}/head:{pr_ref}")
        return _git(runner, store, "rev-parse", f"{pr_ref}^{{commit}}").stdout.strip()


# --- pruning ---------------------------------------------------------------


def prune_dead_worktrees(store: Path, runs_root: Path, own_path: Path, *, config: Any, runner: Any) -> None:
    """Remove owned worktrees whose run is gone; everything else is left untouched."""
    own = Path(os.path.abspath(own_path))
    with store_lock(store):
        _git(runner, store, "worktree", "prune", check=False)
        if runs_root.is_dir():
            for entry in sorted(runs_root.iterdir()):
                if Path(os.path.abspath(entry)) == own:
                    continue
                if identify_owned_worktree(store, runs_root, entry) is None:
                    log(config, f"Leaving unrecognized entry under {runs_root} untouched: {entry.name}")
                    continue
                with probe_free_claim(entry) as free:
                    if not free:
                        continue
                    if _remove_owned_locked(store, entry, config=config, runner=runner):
                        log(config, f"Pruned stale run worktree: {entry}")
        for record_file in _records_dir(store).glob("*.json"):
            try:
                raw = json.loads(record_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
                continue
            if os.path.lexists(raw["path"]):
                if raw.get("state") != "ready":
                    log(config, f"Run worktree {raw['path']} has a pending owner record; remove it manually.")
                continue
            if _admin_registered(store, str(raw.get("admin_name") or "")) and raw.get("state") == "ready":
                log(config, f"Run worktree {raw['path']} is gone but still registered by git; keeping its record.")
                continue
            with contextlib.suppress(OSError):
                record_file.unlink()
