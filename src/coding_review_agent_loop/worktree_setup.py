"""Operator-supplied dependency setup for shared stores (#1630, #1650).

``--worktree-setup CMD`` runs once per shared store (``<scratch>/OWNER-REPO/
<agent>/repo``) so that ``--worktree-link`` sources such as ``.venv`` or
``node_modules`` exist before per-run worktrees link them.  The setup is keyed
on a digest of the ``--worktree-setup-key`` entries at the pinned base commit
plus the command string, and runs inside ``run_worktrees.prepare_store``'s
store-lock hold, so every base pin it hands out has a matching setup.

The success record lives under the private store-lock root, never inside the
clone.  It describes the environment currently on disk: it is deleted before
any setup attempt and whenever agent-loop (re-)creates the store, and is
written only after a zero exit.  A required setup refuses any tracked
modification in the store, including initialized submodules, before it
touches the record.

A separate private generated-key ledger (``setup-generated.json``, #1649)
records the SHA-256 of each ``--worktree-setup-key`` file that is absent at the
pinned SHA but exists after a successful setup (for example the
``package-lock.json`` that ``npm install`` writes when the repository tracks
none).  Such a file is exempt from the untracked-shadow refusal on later
setups only while its bytes still match.  The ledger survives pre-setup record
invalidation and failed setups, and is deleted when agent-loop (re-)creates the
store.
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import json
import os
import re
import secrets
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from .errors import AgentLoopError
from .logging import log

SETUP_RECORD_NAME = "setup.json"
SETUP_RECORD_VERSION = 1
GENERATED_LEDGER_NAME = "setup-generated.json"
GENERATED_LEDGER_VERSION = 1
DEFAULT_WORKTREE_SETUP_TIMEOUT_SECONDS = 3600
_SETUP_RECORD_MAX_BYTES = 65536
OUTPUT_TAIL_LINES = 40
OUTPUT_TAIL_BYTES = 4096
DIRTY_PATH_LIMIT = 10
_PROBE_TIMEOUT_SECONDS = 300.0
_HASH_CHUNK_BYTES = 1 << 20
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")

SETUP_CREATED_HINT = (
    "This file may have been created by the --worktree-setup command itself (for example, "
    "npm install writes package-lock.json when the repository tracks none)"
)

MANIFEST_NAMES = frozenset({
    "pyproject.toml", "setup.py", "Pipfile.lock", "poetry.lock", "uv.lock",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
})
MANIFEST_PATTERNS = ("requirements*.txt",)

# Run tokens that already got (or were checked for) the detection hint.
_HINTED_RUNS: set[str] = set()


# --- record ----------------------------------------------------------------


def setup_record_path(store: Path) -> Path:
    from .run_worktrees import _records_dir

    return _records_dir(store) / SETUP_RECORD_NAME


def command_hash(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def _read_private_json(path: Path) -> dict | None:
    """A private operator-owned JSON object, or None for anything missing, unreadable or malformed."""
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                return None
            raw = os.read(descriptor, _SETUP_RECORD_MAX_BYTES + 1)
        finally:
            os.close(descriptor)
    except OSError:
        return None
    if len(raw) > _SETUP_RECORD_MAX_BYTES:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def read_setup_record(store: Path) -> dict | None:
    """The stored record, or None for anything missing, unreadable or malformed."""
    try:
        path = setup_record_path(store)
    except OSError:
        return None
    data = _read_private_json(path)
    if data is None or data.get("version") != SETUP_RECORD_VERSION:
        return None
    return data


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_private(path: Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def invalidate_setup_record(store: Path) -> None:
    """Delete the record (a missing one is fine); fail closed when that is impossible.

    The generated-key ledger is kept: it describes files on disk, not the
    success of a setup, so a failed retry still exempts unchanged files.
    """
    try:
        path = setup_record_path(store)
        _unlink_private(path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise AgentLoopError(
            f"Could not invalidate the --worktree-setup record for the shared clone {store}: {exc}. "
            "Refusing to continue."
        ) from exc


def invalidate_store_setup_state(store: Path) -> None:
    """Delete the record and the generated-key ledger before the store is (re-)created."""
    try:
        path = setup_record_path(store)
        _unlink_private(path)
        _unlink_private(generated_ledger_path(store))
        _fsync_directory(path.parent)
    except OSError as exc:
        raise AgentLoopError(
            f"Could not invalidate the --worktree-setup state for the shared clone {store}: {exc}. "
            "Refusing to continue."
        ) from exc


def _write_private_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    _fsync_directory(path.parent)


def _write_setup_record(store: Path, data: dict) -> None:
    _write_private_json(setup_record_path(store), data)


def _store_identity(store: Path) -> list[int] | None:
    from .run_worktrees import _identity

    return _identity(store)


def record_matches(record: dict | None, *, store: Path, digest: str, command: str) -> bool:
    return (
        record is not None
        and record.get("digest") == digest
        and record.get("command_sha256") == command_hash(command)
        # Identity is a secondary check only: deletion on store (re-)creation
        # and the missing-link-source rule are the primary invalidations.
        and record.get("store_identity") == _store_identity(store)
    )


# --- generated-key ledger (#1649) -------------------------------------------


def generated_ledger_path(store: Path) -> Path:
    from .run_worktrees import _records_dir

    return _records_dir(store) / GENERATED_LEDGER_NAME


def read_generated_ledger(store: Path, keys: tuple[str, ...] | list[str]) -> dict[str, str]:
    """Validated ``{key: sha256}`` of setup-created absent keys; empty for anything untrusted."""
    try:
        data = _read_private_json(generated_ledger_path(store))
    except OSError:
        return {}
    if data is None or data.get("version") != GENERATED_LEDGER_VERSION:
        return {}
    if data.get("store_identity") != _store_identity(store):
        return {}
    generated = data.get("generated_keys")
    if not isinstance(generated, dict):
        return {}
    allowed = set(keys)
    for key, value in generated.items():
        if key not in allowed or not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
            return {}
    return dict(generated)


def _write_generated_ledger(store: Path, generated: Mapping[str, str]) -> None:
    _write_private_json(generated_ledger_path(store), {
        "version": GENERATED_LEDGER_VERSION,
        "store_identity": _store_identity(store),
        "generated_keys": dict(sorted(generated.items())),
    })


def _regular_key_hash(store: Path, key: str) -> str | None:
    """SHA-256 of ``store/key`` when it is an operator-owned regular file inside the store, else None.

    The final component is opened without following a symlink, and the
    resolved path must stay inside the store and be the opened file.
    """
    try:
        root = os.path.realpath(store)
        path = os.path.join(root, key)
        resolved = os.path.realpath(path)
        if resolved != path or os.path.commonpath((root, resolved)) != root:
            return None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                return None
            named = os.stat(resolved, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino):
                return None
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
        finally:
            os.close(descriptor)
    except (OSError, ValueError):
        return None
    return digest.hexdigest()


def _generated_key_matches(store: Path, key: str, expected_sha256: str | None) -> bool:
    return bool(expected_sha256) and _regular_key_hash(store, key) == expected_sha256


def _generated_after_setup(store: Path, entries: dict[str, tuple[str, str, str] | None]) -> dict[str, str]:
    """Hashes of the keys absent at the SHA that exist as regular files after a successful setup.

    The pre-run checks refused every other shadow (and verified each exempt
    file), so whatever is present now was produced or rewritten by the setup.
    """
    generated: dict[str, str] = {}
    for key, entry in entries.items():
        if entry is not None:
            continue
        digest = _regular_key_hash(store, key)
        if digest is not None:
            generated[key] = digest
    return generated


# --- digest ----------------------------------------------------------------


def key_entry(store: Path, sha: str, key: str, *, runner: Any) -> tuple[str, str, str] | None:
    """``(mode, type, oid)`` of ``key`` in the commit ``sha``, or None when absent."""
    result = runner.run(
        ("git", "--literal-pathspecs", "ls-tree", "-z", sha, "--", key), cwd=store, check=False
    )
    if result.returncode != 0:
        raise AgentLoopError(
            f"Could not read --worktree-setup-key '{key}' at {sha} in the shared clone {store} "
            f"(git exited {result.returncode})."
        )
    for record in (result.stdout or "").split("\0"):
        if not record:
            continue
        metadata, separator, path = record.partition("\t")
        fields = metadata.split(" ")
        if not separator or len(fields) != 3:
            raise AgentLoopError(f"Unrecognized git ls-tree output for the shared clone {store}.")
        if path == key:
            return fields[0], fields[1], fields[2]
    return None


def setup_digest(
    store: Path, sha: str, *, command: str, keys: tuple[str, ...], runner: Any
) -> tuple[str, dict[str, tuple[str, str, str] | None]]:
    """Digest of the command and every key's Git entry at ``sha`` (never working-tree bytes)."""
    entries: dict[str, tuple[str, str, str] | None] = {}
    for key in sorted(dict.fromkeys(keys)):
        entries[key] = key_entry(store, sha, key, runner=runner)
    payload = json.dumps(
        {
            "version": SETUP_RECORD_VERSION,
            "command_sha256": command_hash(command),
            "keys": [[key, *entry] if entry else [key, "absent"] for key, entry in entries.items()],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), entries


# --- tracked-tree cleanliness ----------------------------------------------


def _porcelain_paths(raw: bytes) -> list[str]:
    paths: list[str] = []
    records = raw.split(b"\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise AgentLoopError("git status returned malformed porcelain output")
        paths.append(record[3:].decode("utf-8", errors="surrogateescape"))
        if b"R" in record[:2] or b"C" in record[:2]:
            index += 1
    return paths


def store_tracked_tree_dirty_paths(store: Path) -> list[str]:
    """Every modified, staged or deleted tracked path, including drifted or dirty submodules.

    Parent probes run through the confined Git boundary, which injects
    ``--ignore-submodules=all``; submodule state comes only from separate
    confined probes rooted at each initialized submodule.
    """
    from .local_test_evidence import (
        _dirty_populated_gitlinks,
        _git_index_entries,
        _git_nul_paths,
        _run_git,
    )

    def probe(*args: str) -> bytes:
        return _run_git(store, args, timeout=_PROBE_TIMEOUT_SECONDS)

    try:
        dirty = set(_porcelain_paths(probe("status", "--porcelain=v1", "-z", "--untracked-files=no")))
        dirty.update(_git_nul_paths(probe("diff", "--cached", "--name-only", "-z")))
        entries = _git_index_entries(probe("ls-files", "-s", "-z"))
        dirty.update(_dirty_populated_gitlinks(
            Path(os.path.realpath(store)), entries,
            timeout_seconds=_PROBE_TIMEOUT_SECONDS, cancel=None, started=time.monotonic(),
        ))
        # Status and diff trust index flags; a flagged path can hide edited bytes.
        dirty.update(
            f"{path} (assume-unchanged or skip-worktree)"
            for path in _flag_suppressed_paths(Path(os.path.realpath(store)))
        )
    except AgentLoopError as exc:
        raise AgentLoopError(
            f"Could not verify the tracked tree of the shared clone {store} before --worktree-setup: "
            f"{exc}. Refusing to run setup."
        ) from exc
    return sorted(dirty)


def _flag_suppressed_paths(
    root: Path, *, prefix: str = "", depth: int = 0, budget: list[int] | None = None
) -> list[str]:
    """Tracked paths whose index entry hides working-tree changes from status and diff.

    ``git ls-files -v`` tags assume-unchanged entries with a lowercase letter
    and skip-worktree entries with ``S``. Initialized submodules are inspected
    by separate confined probes rooted at each submodule, recursively.
    """
    from .local_test_evidence import _git_index_entries, _run_git

    if budget is None:
        budget = [128]
    raw = _run_git(root, ("ls-files", "-v", "-z"), timeout=_PROBE_TIMEOUT_SECONDS)
    flagged: list[str] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            raise AgentLoopError("git ls-files -v returned malformed output")
        tag = record[:1]
        if tag.islower() or tag == b"S":
            flagged.append(prefix + record[2:].decode("utf-8", errors="surrogateescape"))
    entries = _git_index_entries(_run_git(root, ("ls-files", "-s", "-z"), timeout=_PROBE_TIMEOUT_SECONDS))
    for relative, (mode, _object_id) in entries.items():
        if mode != "160000":
            continue
        path = root / relative
        if not os.path.lexists(path / ".git"):
            continue  # uninitialized: no worktree whose bytes could be hidden
        budget[0] -= 1
        if budget[0] < 0 or depth >= 8:
            raise AgentLoopError("gitlink inspection limit exceeded")
        if path.is_symlink() or not path.is_dir():
            raise AgentLoopError("gitlink worktree has an unsafe path")
        sub_root = path.resolve(strict=True)
        if os.path.commonpath((str(root), str(sub_root))) != str(root):
            raise AgentLoopError("gitlink worktree escapes the shared clone")
        flagged += _flag_suppressed_paths(
            sub_root, prefix=f"{prefix}{relative}/", depth=depth + 1, budget=budget
        )
    return flagged


_TRACKED_REMEDY = (
    "Tracked files in the shared clone must match the base commit; restore or remove "
    "those local changes (agent-loop never overwrites them) and rerun."
)
_ABSENT_KEY_REMEDY = (
    f"{SETUP_CREATED_HINT}. After confirming, remove it from the shared clone or track it in "
    "the repository, and rerun; agent-loop never removes it."
)
_TRACKED_KEY_REMEDY = (
    f"{SETUP_CREATED_HINT}. The base commit now tracks it and agent-loop removes such a file "
    "only while its bytes still match what a successful setup produced; after confirming, "
    "remove it from the shared clone and rerun."
)


def _refuse_dirty(store: Path, paths: list[str], *, reason: str, remedy: str = _TRACKED_REMEDY) -> None:
    shown = ", ".join(paths[:DIRTY_PATH_LIMIT])
    more = f" (and {len(paths) - DIRTY_PATH_LIMIT} more)" if len(paths) > DIRTY_PATH_LIMIT else ""
    raise AgentLoopError(
        f"--worktree-setup refused: the shared clone {store} {reason}: {shown}{more}. {remedy}"
    )


def _untracked_absent_keys(
    store: Path,
    entries: dict[str, tuple[str, str, str] | None],
    *,
    runner: Any,
    tracked_at_head: bool,
    exempt: Mapping[str, str] | None = None,
) -> list[str]:
    shadows: list[str] = []
    for key, entry in entries.items():
        if entry is not None or not os.path.lexists(store / key):
            continue
        if exempt and key in exempt and _generated_key_matches(store, key, exempt[key]):
            # Produced by an earlier successful setup and unchanged since (#1649).
            continue
        if tracked_at_head and key_entry(store, "HEAD", key, runner=runner) is not None:
            # Tracked at the older HEAD: the checkout to the pinned SHA removes it.
            continue
        shadows.append(key)
    return shadows


def _untracked_newly_tracked_keys(
    store: Path,
    entries: dict[str, tuple[str, str, str] | None],
    *,
    runner: Any,
    generated: Mapping[str, str],
    head_born: bool,
) -> tuple[list[str], list[str]]:
    """``(removable, refused)`` untracked files at keys the pinned SHA tracks but HEAD does not.

    Only a recorded generated file whose bytes still match is removable; any
    other file (ignored or not) is refused, because a non-forced checkout
    would either refuse it or silently overwrite an ignored one.
    """
    removable: list[str] = []
    refused: list[str] = []
    for key, entry in entries.items():
        if entry is None or not os.path.lexists(store / key):
            continue
        if head_born and key_entry(store, "HEAD", key, runner=runner) is not None:
            continue
        if key in generated and _generated_key_matches(store, key, generated[key]):
            removable.append(key)
        else:
            refused.append(key)
    return removable, refused


def _check_store_clean(
    store: Path,
    entries: dict[str, tuple[str, str, str] | None],
    *,
    runner: Any,
    at_sha: bool,
    generated: Mapping[str, str] | None = None,
) -> list[str]:
    """Refuse an unsafe store; before the checkout, return generated keys to remove for it."""
    generated = generated or {}
    dirty = store_tracked_tree_dirty_paths(store)
    if dirty:
        _refuse_dirty(store, dirty, reason="has locally modified, staged or deleted tracked paths")
    tracked_at_head = False
    if not at_sha:
        # An unborn HEAD (a freshly initialized store) tracks nothing yet.
        head = runner.run(("git", "rev-parse", "--verify", "--quiet", "HEAD"), cwd=store, check=False)
        tracked_at_head = head.returncode == 0
    shadows = _untracked_absent_keys(
        store, entries, runner=runner, tracked_at_head=tracked_at_head, exempt=generated
    )
    if shadows:
        _refuse_dirty(
            store, shadows,
            reason="has untracked files at --worktree-setup-key paths that are absent at the base commit",
            remedy=_ABSENT_KEY_REMEDY,
        )
    if at_sha:
        return []
    removable, refused = _untracked_newly_tracked_keys(
        store, entries, runner=runner, generated=generated, head_born=tracked_at_head
    )
    if refused:
        _refuse_dirty(
            store, refused,
            reason="has untracked files at --worktree-setup-key paths that the base commit tracks",
            remedy=_TRACKED_KEY_REMEDY,
        )
    return removable


def _remove_generated_keys(store: Path, keys: list[str], generated: Mapping[str, str]) -> None:
    """Unlink each generated key, re-verifying its bytes immediately before the unlink."""
    for key in keys:
        if not _generated_key_matches(store, key, generated.get(key)):
            _refuse_dirty(
                store, [key],
                reason="has untracked files at --worktree-setup-key paths that the base commit tracks",
                remedy=_TRACKED_KEY_REMEDY,
            )
        try:
            os.unlink(os.path.join(os.path.realpath(store), key))
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise AgentLoopError(
                f"--worktree-setup refused: could not remove the setup-created {key} in the shared "
                f"clone {store} before checking out the base commit, which now tracks it: {exc}."
            ) from exc


def _materialize(store: Path, sha: str, *, runner: Any, keys: tuple[str, ...] = ()) -> None:
    head = runner.run(("git", "rev-parse", "HEAD"), cwd=store, check=False)
    if head.returncode == 0 and head.stdout.strip() == sha:
        return
    # Non-forced, and never overwriting ignored files: Git refuses rather
    # than replacing local changes or ignored untracked content.
    moved = runner.run(
        ("git", "checkout", "-q", "--detach", "--no-overwrite-ignore", sha), cwd=store, check=False
    )
    if moved.returncode != 0:
        stderr = (moved.stderr or "").strip()
        hint = ""
        if any(key in stderr for key in keys):
            hint = (
                f" A refused --worktree-setup-key path may be an untracked file: {SETUP_CREATED_HINT}; "
                "after confirming, remove it from the shared clone and rerun."
            )
        raise AgentLoopError(
            f"--worktree-setup refused: could not check out {sha} in the shared clone {store} "
            f"(git exited {moved.returncode}). {stderr}{hint}"
        )
    head = runner.run(("git", "rev-parse", "HEAD"), cwd=store, check=False)
    if head.returncode != 0 or head.stdout.strip() != sha:
        raise AgentLoopError(
            f"--worktree-setup refused: the shared clone {store} is not at {sha} after checkout."
        )


# --- running the command ----------------------------------------------------


def output_tail(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()[-OUTPUT_TAIL_LINES:]
    tail = "\n".join(lines)
    encoded = tail.encode("utf-8")
    if len(encoded) > OUTPUT_TAIL_BYTES:
        tail = encoded[-OUTPUT_TAIL_BYTES:].decode("utf-8", errors="replace")
    return tail


def _kill_group(process: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        process.wait(timeout=10)


_TAIL_READ_BYTES = 2 * OUTPUT_TAIL_BYTES


def read_output_tail(handle: Any) -> bytes:
    """Read only a bounded suffix of the captured output, however large it grew."""
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(max(0, size - _TAIL_READ_BYTES))
    return handle.read(_TAIL_READ_BYTES)


_EXIT_POLL_SECONDS = 0.05


def _wait_for_exit_unreaped(process: subprocess.Popen, timeout_seconds: float) -> None:
    """Wait for the leader to exit but leave it a zombie, so its process group ID stays reserved.

    Raises ``subprocess.TimeoutExpired`` when the leader is still running at the deadline.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        info = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if info is not None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout_seconds)
        time.sleep(min(_EXIT_POLL_SECONDS, remaining))


def run_setup_command(
    store: Path,
    command: str,
    *,
    timeout_seconds: float,
    where: str | None = None,
    consequence: str = "no setup record was written",
) -> None:
    """Run ``command`` via ``/bin/sh -c`` in ``store`` as the operator; raise with the output tail.

    ``store`` is the shared clone, or a run worktree for a PR-local environment
    (#1652); ``where`` and ``consequence`` only select the message text.

    However the command ends, every process left in its group is killed before
    the leader is reaped, so no background child of a failed (or finished)
    setup can keep writing to the shared environment after the store lock is
    released. The leader stays an unreaped zombie until then, which keeps its
    PID, and so the group ID, from being reused by an unrelated process.
    """
    where = where or f"the shared clone {store}"
    if not hasattr(os, "waitid"):
        # Without an unreaped wait the group cannot be killed safely after the
        # shell exits, so a surviving child could outlive the store lock.
        raise AgentLoopError(
            f"--worktree-setup is not supported on this host: os.waitid is unavailable, so "
            f"agent-loop cannot stop the command's background processes safely. Nothing ran "
            f"in {where} and {consequence}."
        )
    with tempfile.TemporaryFile() as output:
        try:
            process = subprocess.Popen(
                ["/bin/sh", "-c", command], cwd=store, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            )
        except OSError as exc:
            raise AgentLoopError(
                f"--worktree-setup could not start in {where}: {exc}. "
                f"{consequence[:1].upper()}{consequence[1:]}."
            ) from exc
        try:
            _wait_for_exit_unreaped(process, timeout_seconds)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            raise AgentLoopError(
                f"--worktree-setup timed out after {timeout_seconds:g}s in {where}; "
                f"{consequence}. Raise --worktree-setup-timeout SECONDS (currently "
                f"{timeout_seconds:g}s) if the setup legitimately needs longer. "
                f"Output tail:\n{output_tail(read_output_tail(output))}"
            ) from None
        except BaseException:
            _kill_group(process)
            raise
        # The leader has exited but is not reaped yet, so the group ID is still
        # ours: stop any surviving descendants, then reap.
        _kill_group(process)
        returncode = process.wait()
        if returncode != 0:
            raise AgentLoopError(
                f"--worktree-setup exited {returncode} in {where}; "
                f"{consequence}. Output tail:\n{output_tail(read_output_tail(output))}"
            )


# --- entry point ------------------------------------------------------------


def setup_timeout_seconds(config: Any) -> int:
    """``--worktree-setup-timeout``; independent of every test-gate ceiling."""
    value = getattr(config, "worktree_setup_timeout_seconds", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return DEFAULT_WORKTREE_SETUP_TIMEOUT_SECONDS
    return value


def ensure_store_setup(store: Path, sha: str, *, free: bool, config: Any, runner: Any) -> bool:
    """Run the operator setup for ``sha`` when required; the store lock must be held.

    Returns True when the command ran in this lock hold.
    """
    command = getattr(config, "worktree_setup", None)
    if not command:
        return False
    from .run_worktrees import _held_store_message, _require_lock_held, link_source_usable

    _require_lock_held(store)
    if getattr(runner, "dry_run", False):
        log(config, f"Would run worktree setup in {store}")
        return False
    keys = tuple(getattr(config, "worktree_setup_keys", ()) or ())
    links = tuple(getattr(config, "worktree_links", ()) or ())
    digest, entries = setup_digest(store, sha, command=command, keys=keys, runner=runner)
    unusable = [link for link in links if not link_source_usable(store, link)]
    if record_matches(read_setup_record(store), store=store, digest=digest, command=command) and not unusable:
        return False
    if not free:
        raise AgentLoopError(_held_store_message(store, "needs --worktree-setup to run"))
    generated = read_generated_ledger(store, keys)
    # Refuse before the record is touched, so a refusal leaves everything as it was.
    removable = _check_store_clean(store, entries, runner=runner, at_sha=False, generated=generated)
    invalidate_setup_record(store)
    _remove_generated_keys(store, removable, generated)
    _materialize(store, sha, runner=runner, keys=keys)
    _check_store_clean(store, entries, runner=runner, at_sha=True, generated=generated)
    timeout = setup_timeout_seconds(config)
    log(config, f"Running --worktree-setup in the shared clone {store} at {sha}")
    run_setup_command(store, command, timeout_seconds=timeout)
    for link in links:
        if not link_source_usable(store, link):
            raise AgentLoopError(
                f"--worktree-link '{link}': the --worktree-setup command succeeded but did not create "
                f"{store / link} in the shared clone {store}; no setup record was written."
            )
    # Before the success record, so a crash in between only forces a rerun.
    _write_generated_ledger(store, _generated_after_setup(store, entries))
    _write_setup_record(store, {
        "version": SETUP_RECORD_VERSION,
        "digest": digest,
        "command_sha256": command_hash(command),
        "base_sha": sha,
        "store_identity": _store_identity(store),
        "completed_at": time.time(),
    })
    return True


# --- detection hint ---------------------------------------------------------


def dependency_manifests(store: Path, sha: str, *, runner: Any) -> list[str]:
    """Top-level dependency manifests in the pinned tree (no working-tree scan)."""
    result = runner.run(("git", "ls-tree", "-z", "--name-only", sha), cwd=store, check=False)
    if result.returncode != 0:
        return []
    names = [name for name in (result.stdout or "").split("\0") if name]
    return sorted(
        name for name in names
        if name in MANIFEST_NAMES or any(fnmatch.fnmatchcase(name, p) for p in MANIFEST_PATTERNS)
    )


def maybe_log_dependency_hint(store: Path, sha: str, *, config: Any, runner: Any) -> None:
    """Log one hint per run when manifests exist and no setup or link flag is given."""
    if getattr(config, "worktree_setup", None) or getattr(config, "worktree_links", ()):
        return
    token = str(getattr(config, "run_token", None) or "")
    if token in _HINTED_RUNS:
        return
    _HINTED_RUNS.add(token)
    try:
        found = dependency_manifests(store, sha, runner=runner)
    except Exception:
        return
    if not found:
        return
    log(
        config,
        f"Hint: the base tree has {', '.join(found)}, but per-run worktrees start without untracked "
        "dependencies; pass --worktree-setup CMD with --worktree-setup-key <lockfile> and "
        "--worktree-link <path> (for example .venv or node_modules) to provide them. "
        "Nothing is installed automatically.",
    )
