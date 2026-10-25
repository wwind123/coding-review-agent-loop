"""Per-run linked worktrees over one shared clone per repository and agent (#1162).

The repo-scoped checkout ``<scratch>/OWNER-REPO/<agent>/repo`` is the shared
*store*: it owns the object store and receives verified pack imports.  Each
CLI run with an omitted ``--<agent>-dir`` gets its own detached
``git worktree`` under ``<scratch>/OWNER-REPO/<agent>/runs/<run-token>``,
protected by the usual #1127 run claim and removed when the run ends.

Store mutations (initialization, import+pin, local-base fast-forward, operator
``--worktree-setup``, pack compaction, link creation, prune and remove) are serialized by a short, **non-reentrant** host flock that is
never held across an agent turn.  Every worktree created here has a durable
owner record under the private host lock root; removal and pruning delete only
a directory whose record still matches its creation identity.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .checkout_verification import _refuse_untrusted_root, forget_checkout
from .errors import AgentLoopError, CheckoutVerificationError
from .logging import log
from .scratch import make_private_dirs
from .secure_git import local_command
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


def _create_detached_link(store: Path, path: Path, sha: str) -> tuple[Path, list[int], list[int]]:
    """Create only Git's administrative link; reset populates it separately.

    Git 2.43 execs ``update-ref`` even for ``worktree add --no-checkout``.
    Creating the three administrative files directly keeps child execution
    denied for the entire setup on that Git version and later versions.
    """
    if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", sha):
        raise AgentLoopError("Invalid pinned commit for linked worktree.")
    worktrees = store / ".git" / "worktrees"
    if not (store / ".git").is_dir() or (store / ".git").is_symlink():
        raise AgentLoopError(f"Shared store {store} has an unsafe Git directory.")
    worktrees.mkdir(mode=0o700, exist_ok=True)
    if worktrees.is_symlink():
        raise AgentLoopError(f"Shared store {store} has an unsafe worktree directory.")
    admin = worktrees / path.name
    path.mkdir(mode=0o700)
    path_identity = _identity(path)
    admin_identity = None
    gitfile_identity = None
    try:
        admin.mkdir(mode=0o700)
        admin_identity = _identity(admin)
        def exclusive(target: Path, value: str) -> None:
            nonlocal gitfile_identity
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            if target == path / ".git":
                info = os.fstat(fd)
                gitfile_identity = [info.st_dev, info.st_ino]
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())

        exclusive(admin / "HEAD", sha + "\n")
        exclusive(admin / "commondir", "../..\n")
        exclusive(admin / "gitdir", str(path / ".git") + "\n")
        exclusive(path / ".git", f"gitdir: {admin}\n")
    except Exception as original:
        try:
            if admin_identity is not None:
                if _identity(admin) != admin_identity or admin.is_symlink():
                    raise AgentLoopError(f"Fresh linked worktree cleanup refused changed identity at {admin}.")
                shutil.rmtree(admin)
            if _identity(path) != path_identity or not path.is_dir() or path.is_symlink():
                raise AgentLoopError(f"Fresh linked worktree cleanup refused changed identity at {path}.")
            gitfile = path / ".git"
            if gitfile_identity is not None:
                if _identity(gitfile) != gitfile_identity or not gitfile.is_file() or gitfile.is_symlink():
                    raise AgentLoopError(f"Fresh linked worktree cleanup refused changed identity at {gitfile}.")
                gitfile.unlink()
            path.rmdir()
        except OSError as exc:
            raise AgentLoopError(f"Fresh linked worktree cleanup failed at {path}: {exc}.") from original
        raise
    assert admin_identity is not None
    return admin, path_identity, admin_identity


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
    admin, root_identity, admin_identity = _create_detached_link(store, path, sha)
    try:
        if _admin_dir_of(store, path) != admin:
            raise AgentLoopError(f"git worktree add did not produce a linked worktree at {path}.")
        result = runner.run(("git", "reset", "--hard", sha), cwd=path)
        head = runner.run(("git", "rev-parse", "HEAD"), cwd=path).stdout.strip()
        status = runner.run(("git", "status", "--porcelain"), cwd=path).stdout.strip()
        if result.returncode or head != sha or status or _admin_dir_of(store, path) != admin:
            raise AgentLoopError(f"Fresh linked worktree {path} did not materialize cleanly.")
        from .agent_permissions import register_checkout

        register_checkout(config, path)
        _write_reservation(
            record_file,
            {**base, "state": "ready", "admin_name": admin.name,
             "root": root_identity, "admin": admin_identity},
        )
    except Exception as original:
        forget_checkout(path)
        # Git may reject a malformed administrative link, so remove only the
        # two directories whose inode identities this setup recorded.
        for target, identity in ((path, root_identity), (admin, admin_identity)):
            if _identity(target) != identity or target.is_symlink():
                raise AgentLoopError(
                    f"Fresh linked worktree cleanup refused changed identity at {target}."
                ) from original
            try:
                shutil.rmtree(target)
            except OSError as exc:
                raise AgentLoopError(
                    f"Fresh linked worktree cleanup failed at {target}: {exc}."
                ) from original
        if path.exists() or admin.exists():
            raise AgentLoopError(f"Fresh linked worktree cleanup remained incomplete at {path}.") from original
        raise


def link_source_usable(store: Path, link: str) -> bool:
    """The one predicate for link preflight and forced setup (#1650).

    ``os.path.exists`` follows symlinks, so a dangling source symlink counts as missing.
    """
    return os.path.exists(store / link)


def check_link_sources(store: Path, links: Any) -> None:
    """Fail, naming the path, before any worktree is added when a link source is missing."""
    for link in links:
        if not link_source_usable(store, link):
            raise AgentLoopError(
                f"--worktree-link '{link}': {store / link} does not exist in the shared clone {store}; "
                "pass --worktree-setup CMD to create it (for example "
                "`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`), or create the "
                "path there. Refusing to create a dangling link."
            )


def warn_unignored_links(path: Path, store: Path, links: Any, *, config: Any, runner: Any) -> list[str]:
    """Warn (never fail, never write ignore state) for links no ignore rule matches as a symlink."""
    unignored: list[str] = []
    for link in links:
        try:
            result = runner.run(("git", "check-ignore", "-q", "--no-index", "--", link), cwd=path, check=False)
        except Exception:
            continue
        if result.returncode != 1:
            continue  # 0: ignored; anything else: the probe failed, which is only a hint.
        unignored.append(link)
        log(
            config,
            f"--worktree-link '{link}': {path / link} is not matched by any ignore rule as a symlink; "
            f"add '/{link}' to {store / '.git' / 'info' / 'exclude'} (shared by every worktree) or to "
            f".gitignore without a trailing slash (a dir-only '{link}/' rule does not match a symlink).",
        )
    return unignored


def create_worktree_links(path: Path, store: Path, links: Any) -> dict[str, str]:
    """Symlink each link from the store into the worktree, descriptor-relative (never copy)."""
    mapping: dict[str, str] = {}
    root_fd = os.open(os.fspath(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for link in links:
            target = os.path.abspath(store / link)
            parts = link.split("/")
            opened: list[int] = []
            dir_fd = root_fd
            try:
                for index, component in enumerate(parts[:-1]):
                    try:
                        dir_fd = os.open(
                            component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd
                        )
                    except OSError as exc:
                        missing = "/".join(parts[: index + 1])
                        raise AgentLoopError(
                            f"--worktree-link '{link}': parent directory '{missing}' does not exist "
                            "in the run worktree; only paths whose parent directories are tracked "
                            "can be linked (link the untracked top-level directory instead)"
                        ) from exc
                    opened.append(dir_fd)
                name = parts[-1]
                try:
                    info = os.lstat(name, dir_fd=dir_fd)
                except FileNotFoundError:
                    os.symlink(target, name, dir_fd=dir_fd)
                else:
                    if not stat.S_ISLNK(info.st_mode) or os.readlink(name, dir_fd=dir_fd) != target:
                        raise AgentLoopError(
                            f"--worktree-link '{link}': {path / link} already exists in the run worktree."
                        )
            finally:
                for fd in opened:
                    os.close(fd)
            mapping[link] = target
    finally:
        os.close(root_fd)
    return mapping


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
    from .agent_permissions import (
        release_checkout_response_roots,
        release_orphaned_admin_response_roots,
        remove_checkout_tree,
    )

    # A Gemini response root inside the admin dir is non-listable (#1560);
    # git's recursive delete must be able to list it.
    release_checkout_response_roots(path)
    # A doubled --force also removes a worktree locked through git.
    _git(runner, store, "worktree", "remove", "--force", "--force", str(path), check=False)
    if os.path.lexists(path):
        # Re-verify: git may have failed halfway, so never delete what is no longer ours.
        if identify_owned_worktree(store, runs_root, path) is None:
            log(config, f"Leaving run worktree untouched (identity changed during removal): {path}")
            return False
        remove_checkout_tree(path)
    release_orphaned_admin_response_roots(store)
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

    from .agent_permissions import release_orphaned_admin_response_roots, remove_checkout_tree
    from .worktree_setup import invalidate_store_setup_state

    has_git = store.is_dir() and (store / ".git").exists()
    if has_git:
        release_orphaned_admin_response_roots(store)
        _git(runner, store, "worktree", "prune", check=False)
    if store.is_dir() and _is_stale_default_workdir(store):
        # Without a .git there can be no linked worktrees, so skip the enumeration.
        if has_git and len(_linked_worktrees(store, runner)) > 1:
            pass  # live linked worktrees: never delete the store under them
        elif has_git and getattr(config, "worktree_setup", None) and _store_tracks_paths(store, runner):
            # A real repository whose tracked files were deleted, not an
            # artifact-only directory: keep the checkout and the setup record
            # so the setup gate refuses the deletions instead of re-cloning
            # over them (#1650).
            log(config, f"Shared store {store} has tracked content deleted; keeping it for --worktree-setup.")
        elif not free:
            raise AgentLoopError(_held_store_message(store, "is stale and cannot be recreated"))
        else:
            log(config, f"Stale shared store detected; recreating: {store}")
            from .agent_permissions import release_unlistable_tree

            if (store / ".git" / "worktrees").is_dir():
                release_unlistable_tree(store / ".git" / "worktrees")
            # Before the re-creation, so a crash in between leaves no record or
            # generated-key ledger (#1650, #1649).
            invalidate_store_setup_state(store)
            remove_checkout_tree(store)
            forget_checkout(store)
    if not store.exists():
        if not free:
            raise AgentLoopError(_held_store_message(store, "is missing and cannot be cloned"))
        try:
            make_private_dirs(store.parent)
        except OSError as exc:
            raise AgentLoopError(f"Could not create parent directory for store at {store}: {exc}") from exc
        from .git_transport import default_origin

        origin = default_origin(config.repo, protocol=config.trusted_origin_protocol,
                                local_origin=config.trusted_local_origin)
        invalidate_store_setup_state(store)
        store.mkdir(mode=0o700)
        _git(runner, store, "init", "-q", "-b", config.base or "main")
        _git(runner, store, "remote", "add", "origin", origin)
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


def _store_tracks_paths(store: Path, runner: Any) -> bool:
    """True when the store holds tracked content: index entries or a non-empty HEAD tree.

    A store whose every tracked file was deleted, unstaged or staged, still
    has a HEAD commit with paths, so it is never mistaken for an artifact-only
    directory. Only an unborn HEAD with an empty index counts as untracked, and
    any inspection failure counts as tracked (fail closed).
    """
    listed = _git(runner, store, "ls-files", "-z", check=False)
    if listed.returncode != 0 or (listed.stdout or "").strip("\0"):
        return True
    head = _git(runner, store, "rev-parse", "--verify", "--quiet", "HEAD^{commit}", check=False)
    if head.returncode != 0:
        # Unborn HEAD only when the symbolic ref resolves; anything else is broken.
        symbolic = _git(runner, store, "symbolic-ref", "-q", "HEAD", check=False)
        return symbolic.returncode != 0
    tree = _git(runner, store, "ls-tree", "-z", "--name-only", head.stdout.strip(), check=False)
    return tree.returncode != 0 or bool((tree.stdout or "").strip("\0"))


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
    old = exists.stdout.strip() if exists.returncode == 0 else "0" * len(sha)
    _git(runner, store, "update-ref", ref, sha, old)


def _transport_options(config: Any) -> dict[str, Any]:
    """Operator SSH identity and agent-controlled roots for import_ref (#1380)."""
    return {
        "ssh_identity": getattr(config, "git_ssh_identity", None),
        "forbidden_roots": tuple(getattr(config, "transport_forbidden_roots", ())),
    }


def _fetch_and_pin(store: Path, base: str, *, config: Any, runner: Any, free: bool) -> str:
    from .git_transport import import_ref

    sha = import_ref(
        store, f"refs/heads/{base}", f"refs/remotes/origin/{base}",
        repo=config.repo, runner=runner, gh_cmd=config.gh_cmd,
        local_origin=config.trusted_local_origin,
        **_transport_options(config),
    )
    if free:
        _fast_forward_base(store, base, sha, config=config, runner=runner)
    return sha


def prepare_store(store: Path, *, config: Any, runner: Any) -> str:
    """Clone/validate/fetch the store under the lock and return the pinned base SHA.

    With ``--worktree-setup`` the operator setup for that same SHA runs in the
    same lock hold, so a returned pin never has an out-of-date setup (#1650).
    """
    if not config.base:
        raise AgentLoopError(
            f"Base branch for {config.repo} was not resolved. Pass --base <branch> explicitly."
        )
    from .worktree_setup import ensure_store_setup

    with store_lock(store):
        with probe_free_claim(store) as free:
            _prepare_store_locked(store, free=free, config=config, runner=runner)
            sha = _fetch_and_pin(store, config.base, config=config, runner=runner, free=free)
            ensure_store_setup(store, sha, free=free, config=config, runner=runner)
        compact_packs_if_needed(store, config=config, runner=runner)
        return sha


def pin_pr_head(store: Path, pr_number: int, *, config: Any, runner: Any) -> str:
    """Fetch the PR head under the lock and return its SHA (not the mutable ref)."""
    pr_ref = f"refs/remotes/origin/pr/{pr_number}"
    with store_lock(store):
        from .git_transport import import_ref

        if config.base:
            _fetch_and_pin(store, config.base, config=config, runner=runner, free=False)
        sha = import_ref(
            store, f"refs/pull/{pr_number}/head", pr_ref,
            repo=config.repo, runner=runner, gh_cmd=config.gh_cmd,
            local_origin=config.trusted_local_origin,
            **_transport_options(config),
        )
        compact_packs_if_needed(store, config=config, runner=runner)
        return sha


# --- pack compaction (#1578) -----------------------------------------------
#
# ``gc.auto=0`` stays in the hardened Git config, and ``git repack`` cannot run
# under the exec guard (it execs ``pack-objects``).  Every import writes one
# pack, so a long-lived checkout is compacted here instead: one confined
# ``pack-objects --stdout`` writes a replacement pack into a private scratch
# directory, Python verifies it holds every snapshot object, and only then are
# the snapshot packs removed.  Nothing is ever pruned.

PACK_COMPACTION_THRESHOLD = 50
PACK_COMPACTION_TIMEOUT_SECONDS = 900
PACK_INDEX_TIMEOUT_SECONDS = 300
_PACK_NAME_RE = re.compile(r"^pack-([0-9a-f]{40}(?:[0-9a-f]{24})?)\.pack$")
_PACK_HASH_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_PACK_SIDECARS = (".pack", ".rev", ".bitmap", ".mtimes")


class _CompactionSkip(Exception):
    """A fixed, stderr-free reason why compaction was abandoned."""


class _CompactionTimeout(_CompactionSkip):
    """A confined compaction command exceeded its timeout; starts a backoff (#1580)."""


# A store whose compaction keeps timing out would otherwise hold the store
# flock for the full timeout on every locked fetch.  A timeout records a
# private per-store marker beside the lock file, and later fetches skip the
# expensive pack commands until its deadline: 1 h, doubling per consecutive
# timeout, capped at 24 h.  Any marker that is not plainly valid fails open.
PACK_COMPACTION_BACKOFF_BASE_SECONDS = 3600
PACK_COMPACTION_BACKOFF_MAX_SECONDS = 86400
PACK_COMPACTION_BACKOFF_MAX_FAILURES = 1000
_BACKOFF_SATURATION_EXPONENT = 5  # 3600 * 2**5 already exceeds the cap.
_BACKOFF_MARKER_MAX_BYTES = 4096
_now = time.time


def _backoff_delay(failures: int) -> int:
    exponent = min(max(failures, 1) - 1, _BACKOFF_SATURATION_EXPONENT)
    return min(PACK_COMPACTION_BACKOFF_BASE_SECONDS * 2 ** exponent, PACK_COMPACTION_BACKOFF_MAX_SECONDS)


def _backoff_marker_path(checkout: Path) -> Path:
    return store_lock_root() / f"{store_key(checkout)}.compaction-backoff"


def _read_backoff(checkout: Path, now: float) -> tuple[int, float] | None:
    """``(failures, next_attempt_at)`` from a valid marker, else None (fail open)."""
    try:
        path = _backoff_marker_path(checkout)
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                return None
            raw = os.read(descriptor, _BACKOFF_MARKER_MAX_BYTES + 1)
        finally:
            os.close(descriptor)
    except OSError:
        return None
    if len(raw) > _BACKOFF_MARKER_MAX_BYTES:
        return None
    # Any failure to decode or validate the untrusted bytes, including
    # RecursionError from deep nesting and OverflowError from a huge integer
    # deadline, is an invalid marker: it must fail open, never skip compaction.
    try:
        return _parse_backoff(raw, now)
    except Exception:
        return None


def _parse_backoff(raw: bytes, now: float) -> tuple[int, float] | None:
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        return None
    failures, deadline = data.get("failures"), data.get("next_attempt_at")
    if not isinstance(failures, int) or isinstance(failures, bool):
        return None
    if not 1 <= failures <= PACK_COMPACTION_BACKOFF_MAX_FAILURES:
        return None
    if not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
        return None
    deadline = float(deadline)
    if not math.isfinite(deadline):
        return None
    # A deadline beyond the cap cannot come from a valid write (clock skew or
    # forgery); reject it rather than clamping it against a moving clock.
    if deadline > now + PACK_COMPACTION_BACKOFF_MAX_SECONDS:
        return None
    return failures, deadline


def _write_backoff(checkout: Path, failures: int, now: float) -> tuple[int, float]:
    failures = min(failures, PACK_COMPACTION_BACKOFF_MAX_FAILURES)
    deadline = now + _backoff_delay(failures)
    path = _backoff_marker_path(checkout)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "failures": failures, "next_attempt_at": deadline}, handle)
        # os.replace swaps out even a planted FIFO or symlink without opening it.
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    return failures, deadline


def _clear_backoff(checkout: Path) -> None:
    # Clearing is best effort: a leftover marker still expires within 24 h.
    with contextlib.suppress(OSError):
        _backoff_marker_path(checkout).unlink()


def _manual_repack_hint(checkout: Path) -> str:
    return (f"to compact manually run `git -C {shlex.quote(str(checkout))} repack -a -d --keep-unreachable` "
            "while no agent-loop run uses this store")


def _utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_duration(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds // 3600} h"
    return f"{seconds} s"


def _compaction_log(config: Any, message: str) -> None:
    if not getattr(config, "quiet", False):
        log(config, message)


def _is_regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _snapshot_packs(packdir: Path) -> dict[str, Path]:
    """Every non-kept, non-promisor pack with an index, keyed by pack hash."""
    packs: dict[str, Path] = {}
    for name in sorted(os.listdir(packdir)):
        match = _PACK_NAME_RE.fullmatch(name)
        if match is None:
            continue
        pack = packdir / name
        if not _is_regular(pack) or not _is_regular(pack.with_suffix(".idx")):
            continue
        if os.path.lexists(pack.with_suffix(".keep")) or os.path.lexists(pack.with_suffix(".promisor")):
            continue
        packs[match.group(1)] = pack
    return packs


def _confined_run(args: tuple[str, ...], *, checkout: Path, timeout: int, label: str,
                  stdin: Any = subprocess.DEVNULL, stdout: Any = subprocess.PIPE) -> bytes:
    argv, env, pass_fds = local_command(args, checkout=checkout)
    try:
        result = subprocess.run(argv, cwd=checkout, env=env, pass_fds=pass_fds,
                                stdin=stdin, stdout=stdout, stderr=subprocess.DEVNULL,
                                check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # subprocess.run has already killed and reaped the child.
        raise _CompactionTimeout(f"{label} timed out") from exc
    if result.returncode != 0:
        raise _CompactionSkip(f"{label} failed (exit {result.returncode})")
    return result.stdout or b""


def _pack_index_ids(idx: Path, *, checkout: Path) -> set[str]:
    """Object IDs listed by one pack index, read through builtin ``show-index``."""
    with open(idx, "rb") as handle:
        output = _confined_run(("show-index",), checkout=checkout, stdin=handle,
                               timeout=PACK_INDEX_TIMEOUT_SECONDS, label="show-index")
    ids: set[str] = set()
    for line in output.decode("ascii", "replace").splitlines():
        fields = line.split()
        if len(fields) < 2 or not _PACK_HASH_RE.fullmatch(fields[1]):
            raise _CompactionSkip("show-index output unrecognized")
        ids.add(fields[1])
    return ids


def _pack_directory(checkout: Path, runner: Any) -> Path:
    common = runner.run(("git", "rev-parse", "--git-common-dir"), cwd=checkout).stdout.strip()
    if not common:
        raise _CompactionSkip("no git directory")
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = checkout / common_path
    packdir = common_path / "objects" / "pack"
    try:
        if not stat.S_ISDIR(packdir.lstat().st_mode):
            raise _CompactionSkip("pack directory is not a directory")
    except FileNotFoundError:
        raise _CompactionSkip("no pack directory") from None
    return packdir


# A multi-pack-index written by an operator or ``git maintenance`` names the
# packs compaction deletes.  Like ``git repack -d``, compaction removes every
# MIDX artifact before it writes anything, so readers fall back to the
# per-pack indexes and never see a MIDX naming a missing pack (#1579).  The
# MIDX is never rewritten; Git rebuilds it on the next maintenance run.
_MIDX_NAME = "multi-pack-index"
_MIDX_CHAIN_DIR = "multi-pack-index.d"
_MIDX_CHAIN_FILE = "multi-pack-index-chain"
_MIDX_SIDECAR_RE = re.compile(r"^multi-pack-index-[0-9a-f]{40}(?:[0-9a-f]{24})?\.(?:bitmap|rev)$")
_MIDX_LAYER_RE = re.compile(r"^multi-pack-index-[0-9a-f]{40}(?:[0-9a-f]{24})?\.(?:midx|bitmap|rev)$")
_MIDX_REMOVAL_FAILED = "multi-pack-index could not be removed; old packs kept"


def _midx_artifacts(packdir: Path) -> list[Path]:
    """Exactly-named MIDX artifacts directly in ``packdir``, the top-level file first."""
    names = os.listdir(packdir)
    ordered = [name for name in names if name == _MIDX_NAME]
    ordered += sorted(name for name in names if _MIDX_SIDECAR_RE.fullmatch(name))
    ordered += [name for name in names if name == _MIDX_CHAIN_DIR]
    return [packdir / name for name in ordered]


def _unlink_entry(path: Path) -> None:
    # os.unlink removes a symlink itself, never its target.
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


def _remove_midx_chain(chain: Path) -> None:
    try:
        info = chain.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(info.st_mode):
        _unlink_entry(chain)
        return
    _unlink_entry(chain / _MIDX_CHAIN_FILE)
    for name in sorted(os.listdir(chain)):
        if _MIDX_LAYER_RE.fullmatch(name):
            _unlink_entry(chain / name)
    # Any unexpected entry left inside makes rmdir fail: a removal failure.
    with contextlib.suppress(FileNotFoundError):
        os.rmdir(chain)


def _remove_midx_artifacts(packdir: Path) -> None:
    """Remove every MIDX artifact without following links; raise a fixed skip on failure."""
    try:
        for path in _midx_artifacts(packdir):
            if path.name == _MIDX_CHAIN_DIR:
                _remove_midx_chain(path)
            else:
                _unlink_entry(path)
    except OSError:
        raise _CompactionSkip(_MIDX_REMOVAL_FAILED) from None


def _compact(checkout: Path, *, config: Any, runner: Any) -> None:
    packdir = _pack_directory(checkout, runner)
    snapshot = _snapshot_packs(packdir)
    if len(snapshot) <= PACK_COMPACTION_THRESHOLD:
        # At or below the threshold (for example after a manual repack): no
        # backoff should outlive the condition that caused it.
        _clear_backoff(checkout)
        return
    backoff = _read_backoff(checkout, now := _now())
    if backoff is not None and now < backoff[1]:
        failures, deadline = backoff
        raise _CompactionSkip(
            f"backoff after {failures} consecutive timeout(s); next attempt after "
            f"{_utc_iso(deadline)}; {_manual_repack_hint(checkout)}"
        )
    if _midx_artifacts(packdir):
        # Before any pack command, so a removal failure writes nothing and
        # repeats only this cheap attempt on the next locked fetch.
        _remove_midx_artifacts(packdir)
        _compaction_log(config, f"Removed stale multi-pack-index for {checkout} before compaction")
    inventory: set[str] = set()
    for pack in snapshot.values():
        inventory |= _pack_index_ids(pack.with_suffix(".idx"), checkout=checkout)
    scratch = packdir / f".agent-loop-compact-{secrets.token_hex(8)}"
    os.mkdir(scratch, 0o700)
    try:
        new_pack, new_idx, new_rev = scratch / "new.pack", scratch / "new.idx", scratch / "new.rev"
        descriptor = os.open(new_pack, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            # --no-use-bitmap-index: a bitmap walk returns before
            # --keep-unreachable adds unreferenced objects such as pinned PR
            # heads, and it overrides any pack.useBitmaps in repository config.
            _confined_run(
                ("pack-objects", "--stdout", "--no-use-bitmap-index", "--all", "--reflog",
                 "--indexed-objects", "--keep-unreachable", "--keep-true-parents",
                 "--non-empty", "--revs"),
                checkout=checkout, stdout=handle,
                timeout=PACK_COMPACTION_TIMEOUT_SECONDS, label="pack-objects",
            )
        digest = _confined_run(
            ("index-pack", "--rev-index", "-o", str(new_idx), str(new_pack)),
            checkout=checkout, timeout=PACK_COMPACTION_TIMEOUT_SECONDS, label="index-pack",
        ).decode("ascii", "replace").strip()
        if not _PACK_HASH_RE.fullmatch(digest):
            raise _CompactionSkip("index-pack output unrecognized")
        if not inventory <= _pack_index_ids(new_idx, checkout=checkout):
            raise _CompactionSkip("verification mismatch")
        target = packdir / f"pack-{digest}"
        if not os.path.lexists(target.with_suffix(".idx")):
            # Git's own order: the index last, so readers never see a pack without data.
            for source, suffix in ((new_pack, ".pack"), (new_rev, ".rev"), (new_idx, ".idx")):
                if source.exists():
                    os.chmod(source, 0o444)
                    os.replace(source, target.with_suffix(suffix))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    for pack_hash, pack in snapshot.items():
        if pack_hash == digest:
            continue
        # The index first, so new readers stop finding the pack before its data goes.
        for suffix in (".idx", *_PACK_SIDECARS):
            with contextlib.suppress(FileNotFoundError):
                pack.with_suffix(suffix).unlink()
    _clear_backoff(checkout)
    _compaction_log(
        config,
        f"Compacted {len(snapshot)} packs into one for {checkout} "
        f"({len(_snapshot_packs(packdir))} non-kept packs remain).",
    )


def _record_timeout(checkout: Path, *, config: Any, reason: str) -> None:
    """Start or extend the backoff after a timeout; a write failure is only logged."""
    # This runs inside an ``except`` clause, so nothing may escape it.
    try:
        now = _now()
        previous = _read_backoff(checkout, now)
        failures = min((previous[0] if previous else 0) + 1, PACK_COMPACTION_BACKOFF_MAX_FAILURES)
        failures, _deadline = _write_backoff(checkout, failures, now)
    except Exception:
        _compaction_log(
            config,
            f"Pack compaction for {checkout} skipped: {reason}; backoff marker not written; "
            f"{_manual_repack_hint(checkout)}",
        )
        return
    _compaction_log(
        config,
        f"Pack compaction for {checkout} skipped: {reason}; backing off "
        f"{_format_duration(_backoff_delay(failures))} (consecutive timeouts: {failures}); "
        f"{_manual_repack_hint(checkout)}",
    )


def compact_packs_if_needed(checkout: Path, *, config: Any, runner: Any) -> None:
    """Merge a checkout's packs once there are more than the threshold.

    The caller must hold ``store_lock(checkout)``; that check and the dry-run
    return run outside the log-and-continue block, so a missing lock still
    raises.  Any other failure is logged and never fails the fetch path.
    """
    _require_lock_held(checkout)
    if getattr(runner, "dry_run", False):
        return
    try:
        _compact(checkout, config=config, runner=runner)
    except _CompactionTimeout as exc:
        _record_timeout(checkout, config=config, reason=str(exc))
    except _CompactionSkip as exc:
        _compaction_log(config, f"Pack compaction for {checkout} skipped: {exc}")
    except Exception as exc:
        _compaction_log(config, f"Pack compaction for {checkout} skipped: unexpected {type(exc).__name__}")


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
