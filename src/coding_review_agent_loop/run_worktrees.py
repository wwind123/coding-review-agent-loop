"""Per-run linked worktrees over one shared clone per repository and agent (#1162).

The repo-scoped checkout ``<scratch>/OWNER-REPO/<agent>/repo`` is the shared
*store*: it owns the object store and receives verified pack imports.  Each
CLI run with an omitted ``--<agent>-dir`` gets its own detached
``git worktree`` under ``<scratch>/OWNER-REPO/<agent>/runs/<run-token>``,
protected by the usual #1127 run claim and removed when the run ends.

Store mutations (initialization, import+pin, local-base fast-forward, link creation, prune
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


def check_link_sources(store: Path, links: Any) -> None:
    """Fail, naming the path, before any worktree is added when a link source is missing."""
    for link in links:
        if not os.path.exists(store / link):
            raise AgentLoopError(
                f"--worktree-link '{link}': {store / link} does not exist in the shared checkout; "
                "refusing to create a dangling link."
            )


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
        from .git_transport import default_origin

        origin = default_origin(config.repo, protocol=config.trusted_origin_protocol,
                                local_origin=config.trusted_local_origin)
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


def _fetch_and_pin(store: Path, base: str, *, config: Any, runner: Any, free: bool) -> str:
    from .git_transport import import_ref

    sha = import_ref(
        store, f"refs/heads/{base}", f"refs/remotes/origin/{base}",
        repo=config.repo, runner=runner, gh_cmd=config.gh_cmd,
        local_origin=config.trusted_local_origin,
    )
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
        from .git_transport import import_ref

        if config.base:
            _fetch_and_pin(store, config.base, config=config, runner=runner, free=False)
        return import_ref(
            store, f"refs/pull/{pr_number}/head", pr_ref,
            repo=config.repo, runner=runner, gh_cmd=config.gh_cmd,
            local_origin=config.trusted_local_origin,
        )


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
