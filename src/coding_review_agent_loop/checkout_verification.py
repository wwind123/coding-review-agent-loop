"""Fail-closed verification of an assigned checkout before every agent turn (#1130).

Each assigned checkout has an in-process *expected fingerprint*: branch, HEAD,
staged blob identity and a content identity for every dirty path.  The ledger
is established at startup, verified before every checked turn, before every
between-turn sync and before every real test gate, and refreshed only
afterwards.  A failed post-turn capture *poisons* the entry so a later
clean-but-wrong checkout is refused instead of adopted.

Verification never mutates foreign changes.  The only bytes it removes are a
verifiable agent-loop GEMINI.md injection (see ``agents.antigravity``).
Writes made by another process *during* a turn are absorbed into the post-turn
baseline; excluding them is the workdir claim's job, not this gate.
"""

from __future__ import annotations

import hashlib
import os
import stat as stat_module
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .errors import AgentLoopError, CheckoutVerificationError
from .logging import log

# Roles whose agent runs in a tool-owned temporary directory, not the checkout.
TOOL_ISOLATED_ROLES = frozenset({"semantic-dedupe", "repair"})

# Hash budgets: a file beyond either is recorded by stat identity instead.
MAX_HASH_FILE_BYTES = 16 * 1024 * 1024
MAX_HASH_TOTAL_BYTES = 512 * 1024 * 1024
# Hard cap on dirty paths; exceeding it makes the fingerprint incomplete.
MAX_DIRTY_PATHS = 100_000
_MAX_NAMED_PATHS = 8

Descriptor = tuple


class _Incomplete(Exception):
    """The fingerprint could not be captured completely."""


@dataclass(frozen=True)
class CheckoutFingerprint:
    branch: str
    sha: str
    staged: tuple[tuple[bytes, bytes], ...]
    entries: tuple[tuple[bytes, Descriptor], ...]

    def entry_map(self) -> dict[bytes, Descriptor]:
        return dict(self.entries)


@dataclass(frozen=True)
class _Expected:
    fingerprint: CheckoutFingerprint


@dataclass(frozen=True)
class _Poisoned:
    reason: str


_LEDGER: dict[Path, _Expected | _Poisoned] = {}
_LEDGER_LOCK = threading.Lock()


def _key(path: Path) -> Path:
    # Lexical, NOT symlink-resolved: the ledger follows the path agent-loop was
    # assigned, so replacing that path with a symlink cannot redirect lookups to
    # a different checkout's (absent) entry.
    return Path(os.path.abspath(os.fspath(path)))


# The directory identity (st_dev, st_ino) each assigned path had when its entry
# was created.  A renamed root, a symlinked root or a redirected ancestor all
# change it, and verification refuses before anything touches the replacement.
_ROOT_IDENTITY: dict[Path, tuple[int, int]] = {}


def _root_identity(path: Path) -> tuple[int, int] | None:
    try:
        info = os.stat(os.fspath(path))
    except OSError:
        return None
    return (info.st_dev, info.st_ino)


def _remember_root(path: Path, *, overwrite: bool = False) -> None:
    identity = _root_identity(path)
    if identity is None:
        return
    with _LEDGER_LOCK:
        if overwrite:
            _ROOT_IDENTITY[_key(path)] = identity
        else:
            _ROOT_IDENTITY.setdefault(_key(path), identity)


def _refuse_untrusted_root(path: Path, *, agent: str, purpose: str) -> None:
    """Refuse a poisoned entry or a redirected/replaced root BEFORE any recovery or sync."""
    with _LEDGER_LOCK:
        entry = _LEDGER.get(_key(path))
        recorded = _ROOT_IDENTITY.get(_key(path))
    if entry is None:
        return
    if isinstance(entry, _Poisoned):
        raise CheckoutVerificationError(
            f"Assigned {agent} checkout {path} can no longer be trusted: {entry.reason}. "
            f"Refusing to start the {purpose}."
        )
    if recorded is not None and _root_identity(path) != recorded:
        raise CheckoutVerificationError(
            f"Assigned {agent} checkout {path} is no longer the directory agent-loop prepared: "
            "it was renamed, removed, or replaced (for example by a symlink to another "
            f"checkout). Refusing to start the {purpose}; nothing in the replacement was touched."
        )


@dataclass(frozen=True)
class _LinkRegistration:
    """Worktree links of one prepared run worktree (``--worktree-link``)."""

    lexical: str
    alias: str
    identity: tuple[int, int] | None
    links: dict[bytes, bytes]


# Held only in this process, never as git ignore state: lexical path -> registration.
_LINKS: dict[Path, _LinkRegistration] = {}


def register_worktree_links(path: Path, mapping: dict[str, str]) -> None:
    """Record the links created in ``path`` plus the root identity they are bound to."""
    registration = _LinkRegistration(
        lexical=os.fspath(_key(path)),
        alias=os.path.realpath(os.fspath(path)),
        identity=_root_identity(path),
        links={os.fsencode(link): os.fsencode(target) for link, target in mapping.items()},
    )
    with _LEDGER_LOCK:
        _LINKS[_key(path)] = registration


def registered_links(path: Path) -> dict[bytes, bytes]:
    with _LEDGER_LOCK:
        registration = _LINKS.get(_key(path))
    return dict(registration.links) if registration else {}


def lookup_worktree_links(requested: Path, canonical: Path) -> _LinkRegistration | None:
    """Match only against the fixed recorded strings; refuse a redirected or replaced root."""
    keys = {os.fspath(_key(requested)), os.fspath(canonical)}
    with _LEDGER_LOCK:
        registrations = list(_LINKS.values())
    for registration in registrations:
        if keys & {registration.lexical, registration.alias}:
            if (
                os.fspath(canonical) != registration.alias
                or _root_identity(canonical) != registration.identity
            ):
                raise AgentLoopError("registered worktree root was redirected or replaced")
            return registration
    return None


def link_intact(root: Path, link: bytes | str, target: bytes | str) -> bool:
    """True iff ``root/link`` is, without following symlinked ancestors, exactly the recorded link."""
    try:
        root_fd = os.open(os.fspath(root), os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return False
    try:
        descriptor = _classify(root_fd, os.fsencode(link), _Budget())
    except (OSError, _Incomplete):
        return False
    finally:
        os.close(root_fd)
    return descriptor == ("symlink", os.fsencode(target))


def _check_links(config, runner, path: Path, links: dict[bytes, bytes]) -> None:
    for link, target in sorted(links.items()):
        name = _disp(link)
        if not link_intact(path, link, target):
            raise _Incomplete(f"worktree link {name} was removed, retargeted or reached through a symlink")
        if _probe(config, runner, path, ("ls-files", "-z", "--full-name", "--", f":(literal){name}")):
            raise _Incomplete(f"worktree link {name} is staged in the index")
        if _probe(config, runner, path, ("ls-tree", "-z", "--full-tree", "--name-only", "HEAD", "--", name)):
            raise _Incomplete(f"worktree link {name} is owned by HEAD")


def check_worktree_links(config, runner, path: Path) -> None:
    """Raise CheckoutVerificationError unless every registered link is intact and untracked."""
    links = registered_links(path)
    if not links:
        return
    try:
        _check_links(config, runner, path, links)
    except _Incomplete as exc:
        raise CheckoutVerificationError(f"Cannot verify worktree links of {path}: {exc}.") from exc


def reset_checkout_baselines() -> None:
    """Forget every baseline (tests only)."""
    with _LEDGER_LOCK:
        _LEDGER.clear()
        _ROOT_IDENTITY.clear()
        _LINKS.clear()


def forget_checkout(path: Path) -> None:
    """Drop a path whose checkout agent-loop itself deleted and is re-cloning."""
    with _LEDGER_LOCK:
        _LEDGER.pop(_key(path), None)
        _ROOT_IDENTITY.pop(_key(path), None)
        _LINKS.pop(_key(path), None)


def has_entry(path: Path) -> bool:
    with _LEDGER_LOCK:
        return _key(path) in _LEDGER


def poison_checkout(path: Path, reason: str) -> None:
    with _LEDGER_LOCK:
        _LEDGER[_key(path)] = _Poisoned(reason)
    _remember_root(path)


def _disp(raw: bytes) -> str:
    return os.fsdecode(raw)


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------


def _probe(config, runner, path: Path, args: tuple[str, ...]) -> bytes:
    """Run one read-only git probe and return its stdout bytes."""
    from .agent_permissions import hardened_git_probe_runner_bytes, is_sandboxed

    label = "git " + " ".join(args)
    try:
        if is_sandboxed(config):
            result = hardened_git_probe_runner_bytes(config)(args, path)
        else:
            result = runner.run_binary(("git", *args), cwd=path, check=False)
    except (AgentLoopError, OSError) as exc:
        raise _Incomplete(f"probe `{label}` failed: {exc}") from exc
    if result.returncode != 0:
        raise _Incomplete(f"probe `{label}` exited {result.returncode}")
    return result.stdout


def _parse_status(raw: bytes) -> list[tuple[bytes, bytes]]:
    records: list[tuple[bytes, bytes]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise _Incomplete(f"unparseable status record {record[:40]!r}")
        xy, name = record[:2], record[3:]
        if b"R" in xy or b"C" in xy:
            raise _Incomplete(f"unexpected rename/copy status record for {_disp(name)}")
        records.append((xy, name))
    return records


def _parse_staged(raw: bytes) -> tuple[tuple[bytes, bytes], ...]:
    tokens = raw.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    staged: list[tuple[bytes, bytes]] = []
    index = 0
    while index < len(tokens):
        meta = tokens[index]
        if not meta.startswith(b":") or index + 1 >= len(tokens):
            raise _Incomplete("unparseable staged diff record")
        fields = meta[1:].split(b" ")
        if len(fields) < 5 or fields[4][:1] in (b"R", b"C"):
            raise _Incomplete("unexpected rename/copy staged diff record")
        staged.append((tokens[index + 1], b" ".join(fields)))
        index += 2
    return tuple(sorted(staged))


# --------------------------------------------------------------------------
# Confined worktree reads
# --------------------------------------------------------------------------


class _Budget:
    def __init__(self) -> None:
        self.hashed = 0


def _type_name(mode: int) -> str:
    for check, name in (
        (stat_module.S_ISFIFO, "fifo"),
        (stat_module.S_ISSOCK, "socket"),
        (stat_module.S_ISCHR, "character-device"),
        (stat_module.S_ISBLK, "block-device"),
    ):
        if check(mode):
            return name
    return "special"


def _hash_regular(dir_fd: int, name: bytes, before: os.stat_result, budget: _Budget) -> Descriptor:
    mode = stat_module.S_IMODE(before.st_mode)
    if before.st_size > MAX_HASH_FILE_BYTES or budget.hashed + before.st_size > MAX_HASH_TOTAL_BYTES:
        return (
            "file-stat", mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino,
        )
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY, dir_fd=dir_fd)
    except OSError as exc:
        raise _Incomplete(f"{_disp(name)} changed while being read: {exc}") from exc
    try:
        opened = os.fstat(fd)
        if not stat_module.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev, before.st_ino,
        ):
            raise _Incomplete(f"{_disp(name)} was swapped while being read")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(fd), "rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
                budget.hashed += len(chunk)
        return ("file", mode, digest.hexdigest())
    finally:
        os.close(fd)


def _classify(root_fd: int, path: bytes, budget: _Budget) -> Descriptor:
    parts = [part for part in path.split(b"/") if part]
    if not parts:
        raise _Incomplete("empty path in status output")
    opened: list[int] = []
    dir_fd = root_fd
    try:
        for component in parts[:-1]:
            try:
                next_fd = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd
                )
            except FileNotFoundError:
                return ("missing",)
            except OSError:
                return ("ancestor-not-directory", component)
            opened.append(next_fd)
            dir_fd = next_fd
        name = parts[-1]
        try:
            st = os.lstat(name, dir_fd=dir_fd)
        except FileNotFoundError:
            return ("missing",)
        if stat_module.S_ISREG(st.st_mode):
            return _hash_regular(dir_fd, name, st, budget)
        if stat_module.S_ISLNK(st.st_mode):
            return ("symlink", os.readlink(name, dir_fd=dir_fd))
        if stat_module.S_ISDIR(st.st_mode):
            return ("directory",)
        return ("unsupported", _type_name(st.st_mode))
    finally:
        for fd in opened:
            os.close(fd)


def capture_fingerprint(config, runner, path: Path) -> CheckoutFingerprint:
    """Capture the full fingerprint, or raise ``_Incomplete``."""
    branch = _probe(config, runner, path, ("rev-parse", "--abbrev-ref", "HEAD")).decode(
        "utf-8", "replace"
    ).strip()
    sha = _probe(config, runner, path, ("rev-parse", "HEAD")).decode("utf-8", "replace").strip()
    if not branch or not sha:
        raise _Incomplete("HEAD branch or SHA probe returned blank output")
    status_raw = _probe(
        config, runner, path,
        ("status", "--porcelain", "-z", "--untracked-files=all", "--no-renames"),
    )
    staged_raw = _probe(
        config, runner, path, ("diff", "--cached", "--raw", "--no-abbrev", "-z", "--no-renames")
    )
    records = _parse_status(status_raw)
    links = registered_links(path)
    if links:
        _check_links(config, runner, path, links)
        exact = {b"?? " + link for link in links}
        records = [(xy, name) for xy, name in records if xy + b" " + name not in exact]
    if len(records) > MAX_DIRTY_PATHS:
        tops: dict[bytes, int] = {}
        for _xy, name in records:
            top = name.split(b"/", 1)[0]
            tops[top] = tops.get(top, 0) + 1
        largest = ", ".join(
            f"{_disp(top)} ({count})"
            for top, count in sorted(tops.items(), key=lambda item: -item[1])[:5]
        )
        raise _Incomplete(
            f"{len(records)} dirty paths exceed the {MAX_DIRTY_PATHS}-path limit; "
            f"largest top-level entries: {largest}. Clean build output or untracked trees "
            "out of the assigned checkout."
        )
    staged = _parse_staged(staged_raw)
    entries: dict[bytes, Descriptor] = {}
    if records:
        budget = _Budget()
        try:
            root_fd = os.open(os.fspath(path), os.O_RDONLY | os.O_DIRECTORY)
        except OSError as exc:
            raise _Incomplete(f"checkout directory is not readable: {exc}") from exc
        try:
            for xy, name in records:
                try:
                    descriptor = _classify(root_fd, name, budget)
                except OSError as exc:
                    raise _Incomplete(
                        f"{_disp(name)} could not be read while fingerprinting: {exc}"
                    ) from exc
                kind = descriptor[0]
                if kind == "unsupported":
                    raise _Incomplete(
                        f"{_disp(name)} is an unsupported {descriptor[1]} entry; refusing to open it"
                    )
                if kind == "missing" and b"D" not in xy:
                    raise _Incomplete(
                        f"git reports {_disp(name)} present but it is missing from the worktree"
                    )
                entries[name] = (xy, *descriptor)
        finally:
            os.close(root_fd)
    return CheckoutFingerprint(branch, sha, staged, tuple(sorted(entries.items())))


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _describe_differences(expected: CheckoutFingerprint, observed: CheckoutFingerprint) -> list[str]:
    exp, obs = expected.entry_map(), observed.entry_map()
    lines: list[str] = []
    for name in sorted(set(exp) | set(obs)):
        if name not in exp:
            kind = "added"
            if obs[name][1] in {"ancestor-not-directory"}:
                kind = f"added, {obs[name][1]}"
            lines.append(f"{_disp(name)} ({kind})")
        elif name not in obs:
            lines.append(f"{_disp(name)} (removed)")
        elif exp[name] != obs[name]:
            kind = "content changed"
            if obs[name][1] == "ancestor-not-directory":
                kind = "ancestor-not-directory"
            lines.append(f"{_disp(name)} ({kind})")
    exp_staged, obs_staged = dict(expected.staged), dict(observed.staged)
    for name in sorted(set(exp_staged) | set(obs_staged)):
        if exp_staged.get(name) != obs_staged.get(name):
            lines.append(f"{_disp(name)} (staged changed)")
    return lines


def _bounded(lines: list[str]) -> str:
    shown = ", ".join(lines[:_MAX_NAMED_PATHS])
    extra = len(lines) - _MAX_NAMED_PATHS
    return shown + (f", +{extra} more" if extra > 0 else "")


def _mismatch_message(
    *, agent: str, path: Path, purpose: str, expected: CheckoutFingerprint, observed: CheckoutFingerprint
) -> str:
    parts = [
        f"Assigned {agent} checkout {path} changed outside agent-loop before the {purpose}: "
        f"expected {expected.branch} at {expected.sha}, observed {observed.branch} at {observed.sha}"
    ]
    lines = _describe_differences(expected, observed)
    if lines:
        parts.append(f"unexpected paths: {_bounded(lines)}")
    return "; ".join(parts) + ". Refusing to start the turn."


# --------------------------------------------------------------------------
# GEMINI.md recovery
# --------------------------------------------------------------------------


def recover_gemini_injection(config, runner, path: Path) -> None:
    """Remove a verified leftover agent-loop GEMINI.md injection (never anything else)."""
    if getattr(runner, "dry_run", False) or not os.path.lexists(Path(path) / "GEMINI.md"):
        return
    from .agents.antigravity import recover_stale_gemini_injection

    def is_tracked() -> bool:
        try:
            return bool(_probe(config, runner, path, ("ls-files", "-z", "--", "GEMINI.md")))
        except _Incomplete:
            return True  # never unlink on doubt

    recover_stale_gemini_injection(Path(path), is_tracked=is_tracked)


# --------------------------------------------------------------------------
# Ledger operations
# --------------------------------------------------------------------------


def _capture_or_raise(config, runner, path: Path, *, agent: str, purpose: str) -> CheckoutFingerprint:
    try:
        return capture_fingerprint(config, runner, path)
    except _Incomplete as exc:
        raise CheckoutVerificationError(
            f"Cannot verify the assigned {agent} checkout {path} before the {purpose}: {exc}. "
            "Refusing to start the turn."
        ) from exc


def _is_clean(fingerprint: CheckoutFingerprint) -> bool:
    return not fingerprint.entries and not fingerprint.staged


def establish_initial_baseline(config, runner, path: Path) -> None:
    """Startup: record the freshly prepared checkout.  Raises, creating no entry, on failure."""
    if getattr(runner, "dry_run", False):
        return
    fingerprint = _capture_or_raise(config, runner, path, agent="agent", purpose="startup baseline")
    with _LEDGER_LOCK:
        _LEDGER[_key(path)] = _Expected(fingerprint)
    _remember_root(path, overwrite=True)


def record_checkout_baseline(config, runner, path: Path, *, source: str) -> None:
    """Refresh the baseline from the live checkout; poison it if that cannot be captured."""
    if getattr(runner, "dry_run", False):
        return
    with _LEDGER_LOCK:
        # Poison is sticky: a refresh never turns an untrusted checkout back
        # into a trusted one (only a fresh startup establishment may).
        if isinstance(_LEDGER.get(_key(path)), _Poisoned):
            return
    try:
        fingerprint = capture_fingerprint(config, runner, path)
    except _Incomplete as exc:
        log(config, f"Poisoning checkout baseline for {path} after {source}: {exc}")
        poison_checkout(path, f"baseline capture after {source} failed: {exc}")
        return
    with _LEDGER_LOCK:
        if isinstance(_LEDGER.get(_key(path)), _Poisoned):
            return
        _LEDGER[_key(path)] = _Expected(fingerprint)
    _remember_root(path)  # never rewritten once recorded: a redirect stays visible


def verify_checkout(
    config, runner, *, path: Path, agent: str, purpose: str
) -> CheckoutFingerprint:
    """Refuse (CheckoutVerificationError) unless the checkout matches its baseline."""
    _refuse_untrusted_root(path, agent=agent, purpose=purpose)
    try:
        recover_gemini_injection(config, runner, path)
    except CheckoutVerificationError:
        raise
    except (AgentLoopError, OSError) as exc:
        raise CheckoutVerificationError(str(exc)) from exc
    observed = _capture_or_raise(config, runner, path, agent=agent, purpose=purpose)
    with _LEDGER_LOCK:
        entry = _LEDGER.get(_key(path))
        if entry is None and _is_clean(observed):
            _LEDGER[_key(path)] = _Expected(observed)
            self_established = True
        else:
            self_established = False
    if self_established:
        _remember_root(path)
        log(config, f"Recorded a first baseline for {agent} checkout {path} (self-established).")
        return observed
    if entry is None:
        raise CheckoutVerificationError(
            f"Assigned {agent} checkout {path} was never prepared by agent-loop in this process "
            f"and is not clean ({_bounded(_describe_differences(CheckoutFingerprint(observed.branch, observed.sha, (), ()), observed))}); "
            f"refusing to start the {purpose}."
        )
    if isinstance(entry, _Poisoned):
        raise CheckoutVerificationError(
            f"Assigned {agent} checkout {path} can no longer be trusted: {entry.reason}. "
            f"Refusing to start the {purpose}."
        )
    if observed != entry.fingerprint:
        raise CheckoutVerificationError(
            _mismatch_message(
                agent=agent, path=path, purpose=purpose,
                expected=entry.fingerprint, observed=observed,
            )
        )
    return observed


def verify_before_sync(config, runner, *, path: Path, label: str) -> None:
    """Between-turn syncs verify an existing baseline before they may reset or clean."""
    if getattr(runner, "dry_run", False) or not has_entry(path):
        return
    verify_checkout(config, runner, path=path, agent=label, purpose="checkout sync")


def refresh_after_sync(config, runner, path: Path) -> None:
    if getattr(runner, "dry_run", False):
        return
    if has_entry(path):
        record_checkout_baseline(config, runner, path, source="sync")
    else:
        establish_initial_baseline(config, runner, path)


@contextmanager
def gate_window(config, runner, *, path: Path) -> Iterator[None]:
    """Verify before a real test gate; afterwards adopt only NEW artifacts."""
    if getattr(runner, "dry_run", False):
        yield
        return
    before = verify_checkout(config, runner, path=path, agent="test gate", purpose="test gate")
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        try:
            _adopt_gate_artifacts(config, runner, path, before)
        except CheckoutVerificationError:
            if not failed:
                raise


def _adopt_gate_artifacts(config, runner, path: Path, before: CheckoutFingerprint) -> None:
    try:
        after = capture_fingerprint(config, runner, path)
    except _Incomplete as exc:
        poison_checkout(path, f"capture after the test gate failed: {exc}")
        raise CheckoutVerificationError(
            f"Cannot verify checkout {path} after the test gate: {exc}."
        ) from exc
    prior = before.entry_map()
    after_map = after.entry_map()
    changed = [name for name, value in prior.items() if after_map.get(name) != value]
    if (
        after.branch != before.branch
        or after.sha != before.sha
        or after.staged != before.staged
        or changed
    ):
        poison_checkout(path, "the test gate changed the branch, HEAD or pre-existing paths")
        raise CheckoutVerificationError(
            _mismatch_message(
                agent="test gate", path=path, purpose="test gate (after it ran)",
                expected=before, observed=after,
            ).replace("Refusing to start the turn.", "The gate must not alter existing work.")
        )
    with _LEDGER_LOCK:
        _LEDGER[_key(path)] = _Expected(after)
