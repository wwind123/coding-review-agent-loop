"""Bounded structural health of shared link sources; all publication holds store_lock.

These metadata witnesses detect selected deletion, not package completeness.
No dependency file contents are read and interior symlinks are never traversed.
"""
from __future__ import annotations

import contextlib
import json
import os
import secrets
import shlex
import stat
from collections import deque
from pathlib import Path

from .errors import AgentLoopError
from .logging import log

MAX_DEPTH = 4
MAX_COUNT_DIRS = 64
CHILD_COUNT_CAP = 128
MAX_WITNESSES = 128
SCAN_PAYLOAD_MAX_BYTES = 32768
HEALTH_RECORD_MAX_BYTES = 65536
HEALTH_RECORD_NAME = "link-health.json"


def health_path(store: Path) -> Path:
    from .run_worktrees import _records_dir
    return _records_dir(store) / HEALTH_RECORD_NAME


def _size(data) -> int:
    return len(json.dumps(data, sort_keys=True).encode("utf-8"))


def _identity(info) -> list[int]:
    return [info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)]


def _relative(path: str, depth: int) -> bool:
    return isinstance(path, str) and (path == "" or (
        len(path.split("/")) <= depth and all(p not in ("", ".", "..") for p in path.split("/"))
        and not path.startswith("/") and "\0" not in path
    ))


@contextlib.contextmanager
def _root(store: Path, link: str):
    """Bind the root's lexical and resolved identities, allowing a root symlink only."""
    source = store / link
    # Configured ancestors inside the store may not be symlinks.
    parent = store
    for part in Path(link).parts[:-1]:
        parent /= part
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise OSError(f"unsafe source ancestor {parent}")
    before = source.lstat()
    resolved = os.path.realpath(source)
    target = os.stat(source)
    if not (stat.S_ISDIR(target.st_mode) or stat.S_ISREG(target.st_mode)):
        raise OSError("source is not a directory or regular file")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if stat.S_ISDIR(target.st_mode):
        flags |= os.O_DIRECTORY
    anchor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        with _directory(anchor, str(Path(resolved).parent).lstrip("/")) as parent_fd:
            fd = os.open(Path(resolved).name, flags, dir_fd=parent_fd)
    finally:
        os.close(anchor)
    try:
        binding = {"lexical": _identity(before), "target": _identity(target), "resolved": resolved}
        if _identity(os.fstat(fd)) != binding["target"]:
            raise OSError("source changed while opening")
        yield fd, binding
        if (_identity(source.lstat()) != binding["lexical"] or
                os.path.realpath(source) != resolved or _identity(os.stat(source)) != binding["target"]):
            raise OSError("source changed during inspection")
    finally:
        os.close(fd)


@contextlib.contextmanager
def _directory(root_fd: int, path: str):
    fd = os.dup(root_fd)
    held = []
    try:
        for part in path.split("/") if path else ():
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            held.append((fd, part, _identity(os.fstat(next_fd))))
            fd = next_fd
        yield fd
        for parent_fd, name, identity in held:
            if _identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != identity:
                raise OSError("probe ancestor changed during inspection")
    finally:
        os.close(fd)
        for parent_fd, _name, _identity_value in held:
            os.close(parent_fd)


def _entries(root_fd: int, path: str, *, types: bool) -> list:
    with _directory(root_fd, path) as fd:
        before = os.fstat(fd)
        found = []
        with os.scandir(fd) as iterator:
            for entry in iterator:
                found.append((entry.name, stat.S_IFMT(entry.stat(follow_symlinks=False).st_mode))
                             if types else entry.name)
                if len(found) == CHILD_COUNT_CAP:
                    break
        after = os.fstat(fd)
        if (before.st_mtime_ns, before.st_ctime_ns) != (after.st_mtime_ns, after.st_ctime_ns):
            raise OSError(f"directory {path!r} changed during enumeration")
        return found


def _witness(root_fd: int, path: str) -> int:
    parent, _, name = path.rpartition("/")
    with _directory(root_fd, parent) as fd:
        return stat.S_IFMT(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode)


def capture(store: Path, links: tuple[str, ...]) -> dict:
    """Select immutable probes breadth first, with roots reserved before optional work."""
    from .worktree_setup import _store_identity
    data = {"version": 1, "store_identity": _store_identity(store), "links": list(links),
            "generation": secrets.token_hex(16), "stale": None, "stale_link": None,
            "sources": {}, "limited": False,
            "setup_backed": False}
    queue = deque()
    counts = witnesses = 0
    for link in links:
        with _root(store, link) as (fd, binding):
            source = {"binding": binding, "counts": [], "witnesses": []}
            data["sources"][link] = source
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                counts += 1
                if counts > MAX_COUNT_DIRS:
                    raise AgentLoopError("Too many directory --worktree-link roots (maximum 64).")
                entries = _entries(fd, "", types=True)
                if not entries:
                    raise OSError(f"{link}: empty directory source")
                source["counts"].append(["", len(entries)])
                queue.append((link, "", entries))
    if _size(data) > SCAN_PAYLOAD_MAX_BYTES:
        raise AgentLoopError("--worktree-link configuration exceeds the health record byte budget.")
    while queue:
        link, parent, entries = queue.popleft()
        source = data["sources"][link]
        with _root(store, link) as (fd, binding):
            if binding != source["binding"]:
                raise OSError("source changed during baseline selection")
            for name, kind in entries:
                path = f"{parent}/{name}" if parent else name
                if len(path.split("/")) > MAX_DEPTH:
                    data["limited"] = True
                    continue
                if witnesses < MAX_WITNESSES:
                    source["witnesses"].append([path, kind])
                    if _size(data) > SCAN_PAYLOAD_MAX_BYTES:
                        source["witnesses"].pop()
                        data["limited"] = True
                    else:
                        witnesses += 1
                else:
                    data["limited"] = True
                if kind == stat.S_IFDIR and counts < MAX_COUNT_DIRS:
                    # Reserve space before doing an optional enumeration.
                    source["counts"].append([path, CHILD_COUNT_CAP])
                    if _size(data) > SCAN_PAYLOAD_MAX_BYTES:
                        source["counts"].pop()
                        data["limited"] = True
                        continue
                    children = _entries(fd, path, types=True)
                    source["counts"][-1][1] = len(children)
                    counts += 1
                    queue.append((link, path, children))
                elif kind == stat.S_IFDIR:
                    data["limited"] = True
        if len(entries) == CHILD_COUNT_CAP:
            data["limited"] = True
    return data


def _validate(data: dict, store: Path, links: tuple[str, ...]) -> None:
    from .worktree_setup import _store_identity
    if (type(data.get("version")) is not int or data.get("version") != 1 or data.get("store_identity") != _store_identity(store)
            or data.get("links") != list(links) or not isinstance(data.get("generation"), str)
            or not data["generation"] or not isinstance(data.get("sources"), dict)
            or type(data.get("limited")) is not bool or type(data.get("setup_backed")) is not bool
            or not (data.get("stale") is None or isinstance(data.get("stale"), str))
            or _size(data) > HEALTH_RECORD_MAX_BYTES):
        raise ValueError("invalid health identity/configuration")
    if data.get("stale"):
        if data.get("stale_link") not in links:
            raise ValueError("invalid stale source attribution")
    elif data.get("stale_link") is not None:
        raise ValueError("unexpected stale source attribution")
    if data.get("state") == "pending":
        if not data["stale"] or data["sources"]:
            raise ValueError("invalid pending recovery record")
        return  # A pending record can only refuse; it has no reusable baseline.
    if set(data["sources"]) != set(links) or data.get("state") is not None:
        raise ValueError("invalid source configuration")
    counts = witnesses = 0
    for source in data["sources"].values():
        binding = source["binding"]
        if (set(binding) != {"lexical", "target", "resolved"} or not isinstance(binding["resolved"], str)
                or not os.path.isabs(binding["resolved"])):
            raise ValueError("invalid root binding")
        for key in ("lexical", "target"):
            if (not isinstance(binding[key], list) or len(binding[key]) != 3
                    or any(type(n) is not int for n in binding[key])):
                raise ValueError("invalid source identity")
        for key, maximum in (("counts", MAX_DEPTH), ("witnesses", MAX_DEPTH)):
            if not isinstance(source[key], list):
                raise ValueError("invalid probes")
            seen = set()
            for probe in source[key]:
                if (not isinstance(probe, list) or len(probe) != 2 or not _relative(probe[0], maximum)
                        or type(probe[1]) is not int or probe[0] in seen):
                    raise ValueError("invalid probe")
                seen.add(probe[0])
                if key == "counts" and not 0 <= probe[1] <= CHILD_COUNT_CAP:
                    raise ValueError("invalid count")
        if binding["target"][2] == stat.S_IFDIR:
            if not source["counts"] or source["counts"][0][0] != "" or source["counts"][0][1] == 0:
                raise ValueError("missing nonempty root probe")
        elif binding["target"][2] != stat.S_IFREG or source["counts"] or source["witnesses"]:
            raise ValueError("invalid file source")
        counts += len(source["counts"])
        witnesses += len(source["witnesses"])
    if counts > MAX_COUNT_DIRS or witnesses > MAX_WITNESSES:
        raise ValueError("health probe budget exceeded")


def inspect(store: Path, links: tuple[str, ...], *, selected=None) -> tuple[dict | None, str | None, str]:
    """Unknown/malformed health is a refusal, never an implicit new baseline."""
    from .worktree_setup import _read_private_json
    link = links[0] if links else ""
    try:
        data = _read_private_json(health_path(store))
        if data is None:
            return None, "missing or malformed source-health record", link
        _validate(data, store, links)
        if data["stale"]:
            return data, data["stale"], data["stale_link"]
        if data["setup_backed"]:
            from .worktree_setup import read_setup_record
            success = read_setup_record(store)
            if (success is None or success.get("health_generation") != data["generation"]
                    or success.get("store_identity") != data["store_identity"]):
                return data, "setup success is missing for this health generation", link
        for link in links:
            if selected is not None and link not in selected:
                continue
            source = data["sources"][link]
            with _root(store, link) as (fd, binding):
                if binding != source["binding"]:
                    raise OSError("source root identity or target changed")
                for path, count in source["counts"]:
                    current = len(_entries(fd, path, types=False))
                    if current < count or (path == "" and current == 0):
                        raise OSError(f"directory {path!r} child count decreased or root is empty")
                for path, kind in source["witnesses"]:
                    if _witness(fd, path) != kind:
                        raise OSError(f"witness {path!r} changed type")
        return data, None, link
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"source health cannot be verified: {exc}", link


def publish(store: Path, links: tuple[str, ...], *, config, setup_backed: bool = False) -> dict:
    from .run_worktrees import _require_lock_held
    from .worktree_setup import _write_private_json
    _require_lock_held(store)
    path = health_path(store)
    if os.path.lexists(path) and not stat.S_ISREG(path.lstat().st_mode):
        raise AgentLoopError(f"Unsafe source-health record {path}; refusing publication.")
    data = capture(store, links)
    data["setup_backed"] = setup_backed
    _validate(data, store, links)
    _write_private_json(health_path(store), data)
    if data["limited"]:
        log(config, f"Shared source health for {store} uses bounded coverage: unselected/deeper paths, "
            "same-count substitutions and decreases remaining above the count cap are not verified.")
    return data


def refusal(store: Path, link: str, reason: str, *, turn=False) -> str:
    message = (
        f"Shared clone {str(store.absolute())!r}: --worktree-link {link!r} source "
        f"{str((store / link).absolute())!r} is damaged or stale: {reason}. "
        f"Refusing {'agent turn' if turn else 'worktree creation'}. Rerun once with --worktree-setup CMD, "
        "replacing CMD with a trusted command that reinstalls or validates this source from the shared "
        "clone root. After manual repair use a real validating command that exits nonzero if "
        "dependencies are incomplete. For writable per-run dependencies, use --worktree-setup-pr-local."
    )
    if Path(link).name == "node_modules":
        message += f" For example --worktree-setup {shlex.quote('npm ci')} runs in clone {str(store.absolute())!r}."
    return message


def begin_recovery(store: Path, links: tuple[str, ...], reason: str, *, link: str | None = None) -> None:
    """Retain valid baselines, or publish identity-bound non-authorizing history."""
    from .run_worktrees import _require_lock_held
    from .worktree_setup import _read_private_json, _write_private_json, _store_identity
    _require_lock_held(store)
    path = health_path(store)
    if os.path.lexists(path) and not stat.S_ISREG(path.lstat().st_mode):
        raise AgentLoopError(f"Unsafe source-health record {path}; refusing publication.")
    data = _read_private_json(path)
    try:
        if data is None:
            raise ValueError("missing record")
        _validate(data, store, links)
    except (ValueError, KeyError, TypeError):
        data = {"version": 1, "store_identity": _store_identity(store), "links": list(links),
                "generation": secrets.token_hex(16), "sources": {}, "limited": False,
                "state": "pending", "setup_backed": True}
    data["stale"] = reason[:2048]
    data["stale_link"] = link if link in links else links[0]
    if _size(data) > HEALTH_RECORD_MAX_BYTES:
        raise AgentLoopError("stale source-health record exceeds byte budget")
    _write_private_json(path, data)


def mark_stale(store: Path, links: tuple[str, ...], reason: str, *, link: str | None = None) -> None:
    from .worktree_setup import invalidate_setup_record, _read_private_json
    current = _read_private_json(health_path(store))
    if current is not None and current.get("links") != list(links):
        raise AgentLoopError("source configuration changed; refusing an obsolete stale notification")
    try:
        begin_recovery(store, links, reason, link=link)
    finally:
        # Never discard the generated-key ledger, including on publication failure.
        invalidate_setup_record(store)


def ensure_health(
    store: Path, links: tuple[str, ...], *, config, setup_backed=False, bootstrap=True
) -> str | None:
    """Called under the store lock, including again immediately before worktree creation."""
    if not links:
        return None
    data, reason, link = inspect(store, links)
    if reason and bootstrap and not os.path.lexists(health_path(store)) and not setup_backed:
        try:
            publish(store, links, config=config)
            log(config, f"Bootstrapped manually managed source health in {store}; historical partial "
                "damage is unknowable; use trusted reinstall if completeness is uncertain.")
            return None
        except (OSError, ValueError) as exc:
            reason = str(exc)
    if reason and data is not None:
        mark_stale(store, links, reason, link=link)
    if not reason and data and data["limited"]:
        log(config, f"Shared source health for {store} retains bounded coverage; "
            "unselected/deeper deletion and saturated counts are not fully verified.")
    return refusal(store, link, reason) if reason else None
