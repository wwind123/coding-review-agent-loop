"""Reconcile local test observations without confusing subsets for suites.

This module is deliberately separate from :mod:`evidence_reconciliation`.  The
latter reconciles discuss-mode claims; this module reconciles observations made
by the local test runner and the invocation-local test broker.

The public representation is intentionally boring: bounded strings, opaque
receipt identifiers, and explicit ``unknown`` states.  Raw environments,
snapshot records, and environment identity bytes stay in process memory only.
"""

from __future__ import annotations

import hashlib
import hmac
import select
import json
import math
import os
import re
import signal
import shlex
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import Any, Iterable, Mapping, Sequence

from .errors import AgentLoopError
from .protocol_markers import sanitize_historical_text


SCHEMA_VERSION = 1
LOCAL_TEST_EVIDENCE_SCHEMA = "local-test-evidence-v1"
LOCAL_TEST_EVIDENCE_FIELD = "local_test_evidence"

OUTCOMES = frozenset(
    {"passed", "failed", "timed_out", "interrupted", "incomplete", "overlap-rejected", "launch-failed"}
)
OUT_OF_CHECKOUT_CONTEXT_CAVEAT = "out-of-checkout context run; not evidence"
PROVENANCES = frozenset({"parent-observed", "telemetry-unverified", "self-reported"})
CLAIMS = frozenset({"current-result", "base-reproduction"})
ENVIRONMENT_STATES = frozenset(
    {"equivalent", "different", "unknown", "identity-unknown", "not-compared"}
)
WRAPPER_BOOTSTRAP_STATES = frozenset({"verified", "failed", "unknown"})
INNER_EXEC_STATES = frozenset({"not-attempted", "started", "failed"})
SUITE_START_STATES = frozenset({"not-started", "verified", "unknown"})
ATTRIBUTIONS = frozenset(
    {
        "current-head",
        "base-reproduction",
        "stale",
        "untracked-input-unverified",
        "unknown",
    }
)

MAX_SAFE_COMMAND_BYTES = 512
MAX_SAFE_ARGV_BYTES = 4 * 1024
MAX_SAFE_IDENTIFIER_BYTES = 256
MAX_SAFE_IDENTIFIERS = 8
MAX_SAFE_CAVEAT_BYTES = 256
MAX_PRIVATE_OBSERVATIONS = 64
# Issue #1182: a direct-pytest run the parent positively observed exiting
# before any test could run is recorded as a ``launch-failed`` row carrying
# this caveat first.  It is visible but is not evidence: never an
# authoritative failure and never in need of supersession.  The caveat grants
# nothing on any other outcome.
PRE_COLLECTION_LAUNCH_FAILURE_PREFIX = "pre-collection launch failure; not evidence"
# Issue #1651: result-frame field for parent-selected advisory remedy text.
BROKER_ADVISORY_FIELD = "advisory"
# Issue #1329: the reserved ``superseded_by`` value for a parent-observed
# failure stably attributed to a tracked tree other than the positively known
# current one.  It is history, not an outstanding obligation.  The hyphenated
# form cannot collide with the broker's hex receipt IDs.  The label is derived
# by each reconcile only: it is cleared from every input row and never trusted
# as reconciliation input.
TREE_CHANGE_SUPERSESSION = "superseded-by-tree-change"
# Attached only to failures.  It deliberately avoids the words the protocol's
# receipt-pass checks look for, so it never affects pass authority.
TREE_CHANGE_CAVEAT = (
    "failure recorded at an obsolete tracked tree; historical, not an outstanding obligation"
)
# Issue #1633: attached only to a parent-observed failure restored from an
# earlier run and superseded by a later qualifying pass at the same head and
# tracked tree.  Environments cannot be compared across a restart (their bytes
# are never persisted), so the relaxation is made visible on the failure.  It
# is derived by each reconcile only and never added to the passing row, whose
# receipt-pass authority rejects "environment" and "supersed" caveats.
CROSS_RESTART_SUPERSESSION_CAVEAT = "superseded across a restart; environment not compared"
# The in-process comparison verdicts.  The "differs" caveat is the only form of
# a "different" verdict that survives a restart, so it is ranked ahead of
# generic caveats and disqualifies a restored pass from cross-restart
# supersession.
ENVIRONMENT_DIFFERS_CAVEAT = "effective target environment differs"
ENVIRONMENT_NOT_COMPARED_CAVEAT = "effective target environment could not be compared"
_RESTART_COMMAND_DIGEST_RE = re.compile(r"^[0-9a-f]{32}$")
_RESTART_COMMAND_DOMAIN = b"restart-command-v2\x00"
_RESTART_ROOT_MARKER = "\x00ROOT"
MAX_PRIVATE_DIAGNOSTIC_BYTES = 8 * 1024
MAX_PRIVATE_TOTAL_BYTES = 128 * 1024
MAX_ROUND_OBSERVATIONS = 32
MAX_ROUND_BYTES = 16 * 1024
MAX_SIDECAR_BYTES = 256 * 1024
MAX_SNAPSHOT_FILES = 50_000
MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
MAX_SNAPSHOT_SECONDS = 30.0

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][AB0-2]"
)
_WINDOWS_PATH_RE = re.compile(r"(?<![\w.-])[A-Za-z]:\\[^\s`'\"|;&)<>]+")
_UNC_PATH_RE = re.compile(r"(?<![\w.-])\\\\[^\s`'\"|;&)<>]+")
_POSIX_PATH_RE = re.compile(r"(?<![\w.-])/(?:[^\s`'\"|;&)<>]+)")
_URL_USERINFO_RE = re.compile(r"(?i)(https?://)([^/@\s]+)@")
_AUTH_HEADER_RE = re.compile(r"(?i)^(authorization\s*:\s*)(.+)$")
_SECRET_OPTION_RE = re.compile(
    r"(?i)(--?(?:api[-_]?key|token|password|passwd|secret|credential|auth)(?:=|\s+))([^\s]+)"
)
_SECRET_KEY_RE = re.compile(
    r"(?i)^.*(?:password|passwd|secret|token|api[-_]?key|authorization|private[-_]?key|dsn).*$"
)
_SHELL_OPERATOR_RE = re.compile(r"(?:^|\s)(?:&&|\|\||[|;<>]|\$\(|`)")
_PARAMETER_RE = re.compile(r"\[(?:[^\]]{1,160})\]$")
_REDACTED_VALUE_RE = re.compile(r"<(?:redacted|sha256):[0-9a-f]{16}>")
_USERINFO_PLACEHOLDER_RE = re.compile(r"<userinfo:[0-9a-f]{16}>")
_REPO_RELATIVE_RE = re.compile(
    r"^(?:\.?\.?/)?(?:src|tests?|docs|helpers|lib|app|packages?)/[^\s]+$"
)

# These are exact names by design.  Do not turn this into prefix matching:
# e.g. TEST_SECRET_VALUE remains part of the effective environment identity.
ENVIRONMENT_EXCLUSIONS = frozenset(
    {
        "AGENT_LOOP_TEST_BROKER_ENDPOINT",
        "AGENT_LOOP_TEST_BROKER_CAPABILITY",
        "AGENT_LOOP_TEST_BROKER_PROTOCOL",
        "AGENT_LOOP_INVOCATION_ID",
        "AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS",
        # Orchestrator-authored shell facts (#1343): the configured test
        # interpreter is already in the argv, and a backend-specific shell cap
        # must never split identity between a Claude coder and other backends.
        "AGENT_LOOP_TEST_PYTHON",
        "AGENT_LOOP_SHELL_CAP_MS",
        "BASH_DEFAULT_TIMEOUT_MS",
        "BASH_MAX_TIMEOUT_MS",
        # Worker-budget control values and wrapper-private plugin state.
        "AGENT_LOOP_TEST_WORKERS",
        "AGENT_LOOP_TEST_WORKER_ENFORCEMENT",
        "AGENT_LOOP_WORKER_CAP_SPEC",
        "AGENT_LOOP_WORKER_CAP_NESTED",
        "AGENT_LOOP_PUBLIC_RESPONSE_BELOW",
        # Broker-owned per-request fixture state directory (#1326); the
        # fixture identity itself keys fixture rows apart.
        "AGENT_LOOP_FIXTURE_STATE_DIR",
        # Client tool identity (#1632): diagnostic only, popped by the broker.
        "AGENT_LOOP_RUN_TESTS_TOOL_COMMIT",
        "AGENT_LOOP_RUN_TESTS_TOOL_DIRTY",
        "AGENT_LOOP_RUN_TESTS_TOOL_PACKAGE",
        # Advisory per-run-worktree remedy selector (#1651): presentation
        # only, so spoofing, stripping or changing it never splits identity.
        "AGENT_LOOP_WORKTREE_DEPENDENCY_HINT",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_CODE_CHILD_SESSION",
        "CODEX_SESSION_ID",
        "CODEX_THREAD_ID",
        # Codex changes this metadata on every shell tool call (#1720).
        "CODEX_TOOL_CALL_ID",
        # Per-process and per-session agent and systemd values (#1346): they
        # change with every coder process or systemd invocation and would split
        # identity between turns running the same command, the #1329 deadlock
        # class. Exact names only; stable configuration such as CLAUDE_EFFORT
        # stays in the identity.
        "CLAUDE_PID",
        "CLAUDE_CODE_MESSAGING_SOCKET",
        "CLAUDE_CODE_MESSAGING_TOKEN",
        "CLAUDE_CODE_BRIDGE_SESSION_ID",
        "INVOCATION_ID",
        "JOURNAL_STREAM",
        "SYSTEMD_EXEC_PID",
        "SSH_CLIENT",
        "SSH_CONNECTION",
        "SSH_TTY",
        "SSH_AUTH_SOCK",
        "XDG_SESSION_ID",
        "XDG_SESSION_TYPE",
        "XDG_SESSION_CLASS",
        "XDG_SESSION_DESKTOP",
        "XDG_CURRENT_DESKTOP",
        "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS",
        "TMUX",
        "TMUX_PANE",
        "TERM",
        "TERM_PROGRAM",
        "TERM_PROGRAM_VERSION",
        "COLORTERM",
    }
)


def _bounded(value: object, limit: int) -> str:
    text = str(value)
    if len(text.encode("utf-8", errors="replace")) <= limit:
        return text
    raw = text.encode("utf-8", errors="replace")[: max(0, limit - 3)]
    return raw.decode("utf-8", errors="ignore") + "..."


def _safe_text(value: object, limit: int = MAX_SAFE_CAVEAT_BYTES) -> str:
    cleaned = _CONTROL_RE.sub("", _ANSI_RE.sub("", str(value))).replace("\r", "")
    return _bounded(sanitize_historical_text(cleaned), limit)


def _digest(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()[:16]


def _timestamp(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, str) and value.strip():
        return value
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EvidenceScope:
    """A normalized declaration of what a test observation covered."""

    kind: str = "unknown"
    selectors: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: object) -> "EvidenceScope":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            kind = str(value.get("kind", "unknown"))
            selectors_value = value.get("selectors", ())
            if isinstance(selectors_value, (list, tuple)):
                selectors = tuple(_safe_text(item, 256) for item in selectors_value)
            else:
                selectors = ()
            return cls(kind, selectors)
        if isinstance(value, str) and value.strip():
            return cls(value.strip(), ())
        return cls()

    def normalized(self) -> tuple[str, tuple[str, ...]]:
        return self.kind.strip().lower(), tuple(sorted(set(self.selectors)))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": _safe_text(self.kind, MAX_SAFE_IDENTIFIER_BYTES),
            "selectors": [_safe_text(item, MAX_SAFE_IDENTIFIER_BYTES) for item in self.selectors],
        }


@dataclass(frozen=True)
class TreeAttribution:
    """Safe repository-state attribution for one observation."""

    state: str = "unknown"
    head: str | None = None
    pre_digest: str | None = None
    post_digest: str | None = None
    tracked_digest: str | None = None
    stable: bool | None = None
    untracked_input: bool = False
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state if self.state in ATTRIBUTIONS else "unknown",
            "head": _safe_text(self.head, MAX_SAFE_IDENTIFIER_BYTES) if self.head else None,
            "pre_digest": _safe_text(self.pre_digest, MAX_SAFE_IDENTIFIER_BYTES) if self.pre_digest else None,
            "post_digest": _safe_text(self.post_digest, MAX_SAFE_IDENTIFIER_BYTES) if self.post_digest else None,
            "tracked_digest": _safe_text(self.tracked_digest, MAX_SAFE_IDENTIFIER_BYTES) if self.tracked_digest else None,
            "stable": self.stable,
            "untracked_input": self.untracked_input,
            "caveats": [_safe_text(item, MAX_SAFE_CAVEAT_BYTES) for item in self.caveats[:4]],
        }


class _EnvironmentDifference:
    """Latest names-only snapshot, never part of an evidence projection."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._names: tuple[str, ...] = ()

    def replace(self, names: tuple[str, ...]) -> None:
        with self._lock:
            self._names = names

    def snapshot(self) -> tuple[str, ...]:
        with self._lock:
            return self._names


def _environment_difference_names(left: bytes, right: bytes) -> tuple[str, ...]:
    def entries(raw: bytes) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        offset = 0
        while offset < len(raw):
            parts = []
            for _ in range(2):
                size = struct.unpack_from(">I", raw, offset)[0]
                offset += 4
                if size > len(raw) - offset:
                    raise ValueError("invalid canonical entry length")
                parts.append(raw[offset:offset + size])
                offset += size
            name = parts[0].decode("utf-8")
            parts[1].decode("utf-8")
            if name in result:
                raise ValueError("duplicate canonical entry")
            result[name] = parts[1]
        return result

    before, after = entries(left), entries(right)
    names = []
    total = 0
    for name in sorted(before.keys() | after.keys()):
        if before.get(name) == after.get(name):
            continue
        size = len(name.encode("utf-8"))
        if len(names) == 64 or total + size > 8192:
            break
        names.append(name)
        total += size
    return tuple(names)


@dataclass(frozen=True)
class EnvironmentIdentity:
    """An in-memory identity; ``canonical_bytes`` is never serialized."""

    process_token: str
    canonical_bytes: bytes = field(repr=False, compare=False)
    _difference: _EnvironmentDifference = field(
        default_factory=_EnvironmentDifference, init=False, repr=False, compare=False
    )

    def _differing_names_snapshot(self) -> tuple[str, ...]:
        return self._difference.snapshot()

    def compare(self, other: "EnvironmentIdentity | None") -> str:
        if other is None or self.process_token.split(":", 1)[0] != other.process_token.split(":", 1)[0]:
            return "unknown"
        if self.canonical_bytes == other.canonical_bytes:
            return "equivalent"
        try:
            names = _environment_difference_names(self.canonical_bytes, other.canonical_bytes)
        except (ValueError, UnicodeError, struct.error):
            # Diagnostics cannot alter verdicts or destroy an earlier snapshot.
            pass
        else:
            self._difference.replace(names)
        return "different"


class EnvironmentIdentityRegistry:
    """Memory-only environment identities scoped to one orchestrator process."""

    def __init__(self) -> None:
        self._process_token = uuid.uuid4().hex
        self._lock = Lock()

    def capture(self, environment: Mapping[str, str]) -> EnvironmentIdentity:
        canonical = canonical_environment_bytes(environment)
        return EnvironmentIdentity(f"{self._process_token}:{uuid.uuid4().hex}", canonical)

    def compare(
        self, left: EnvironmentIdentity | None, right: EnvironmentIdentity | None
    ) -> str:
        if left is None or right is None:
            return "unknown"
        return left.compare(right)


def _environment_key(name: str, *, case_insensitive: bool) -> str:
    return name.casefold() if case_insensitive else name


def canonical_environment_bytes(
    environment: Mapping[str, str], *, case_insensitive: bool | None = None
) -> bytes:
    """Encode the effective target environment using exact exclusions.

    The returned bytes are comparison material and must remain in trusted
    process memory.  Callers persisting evidence must only persist the resulting
    comparison state, never these bytes or a digest of them.
    """
    insensitive = os.name == "nt" if case_insensitive is None else case_insensitive
    entries: dict[str, tuple[str, str]] = {}
    exclusions = {
        _environment_key(name, case_insensitive=insensitive) for name in ENVIRONMENT_EXCLUSIONS
    }
    for raw_name, raw_value in environment.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            raise AgentLoopError("test broker environment names and values must be strings")
        if "\x00" in raw_name or "\x00" in raw_value:
            raise AgentLoopError("test broker environment cannot contain NUL bytes")
        key = _environment_key(raw_name, case_insensitive=insensitive)
        if key in exclusions:
            continue
        if key in entries and entries[key][0] != raw_name:
            raise AgentLoopError("test broker environment has a case-colliding variable")
        entries[key] = (raw_name, raw_value)
    output = bytearray()
    for _key, (name, value) in sorted(entries.items(), key=lambda pair: pair[0]):
        name_bytes = name.encode("utf-8")
        value_bytes = value.encode("utf-8")
        output.extend(struct.pack(">I", len(name_bytes)))
        output.extend(name_bytes)
        output.extend(struct.pack(">I", len(value_bytes)))
        output.extend(value_bytes)
    return bytes(output)


def compare_environment_identities(
    current: EnvironmentIdentity | None,
    previous: EnvironmentIdentity | None,
    *,
    registry: EnvironmentIdentityRegistry | None = None,
) -> str:
    if registry is not None:
        return registry.compare(current, previous)
    if current is None or previous is None:
        return "unknown"
    return current.compare(previous)


def environment_comparison_for_restart() -> str:
    """Restored observations cannot regain the old in-memory identity."""
    return "identity-unknown"


# Fixture lifecycle record (issue #1326) ---------------------------------------

FIXTURE_STATUS_STRINGS = frozenset({"timed_out", "not-run", "cancelled"})
FIXTURE_CLEANUP_STATES = frozenset({"confirmed", "escaped", "unconfirmed"})
FIXTURE_UNREADABLE_CAVEAT = "fixture record unreadable"
MAX_FIXTURE_PATH_BYTES = 512
MAX_FIXTURE_ESCAPED_REPORTED = 8
MAX_FIXTURE_ESCAPED_BYTES = 160
MAX_FIXTURE_COUNT = 1_000_000
_FIXTURE_DIGEST_RE = re.compile(r"^(?:100755|100644):[0-9a-f]{40,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FIXTURE_RECORD_FIELDS = frozenset(
    {
        "path", "digest",
        "setup_status", "setup_duration_seconds", "setup_log_digest",
        "teardown_status", "teardown_duration_seconds", "teardown_log_digest",
        "cleanup", "inventoried", "swept", "escaped", "escaped_processes",
    }
)


def _fixture_status(value: object) -> int | str:
    if isinstance(value, bool):
        raise ValueError("fixture status is a boolean")
    if isinstance(value, int):
        if not -255 <= value <= 255:
            raise ValueError("fixture status is out of range")
        return value
    if isinstance(value, str) and value in FIXTURE_STATUS_STRINGS:
        return value
    raise ValueError("unknown fixture status")


def _fixture_duration(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("fixture duration is not a number")
    # Range-check before float(): an arbitrarily large int would overflow.
    if not 0 <= value <= 86_400:
        raise ValueError("fixture duration is out of range")
    return round(float(value), 3)


def _fixture_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_FIXTURE_COUNT:
        raise ValueError("fixture count is out of range")
    return value


def _fixture_log_digest(value: object) -> str:
    if value == "" or (isinstance(value, str) and _SHA256_RE.match(value)):
        return str(value)
    raise ValueError("fixture log digest is malformed")


@dataclass(frozen=True)
class FixtureRecord:
    """Bounded lifecycle of one ``run-tests --fixture`` run.

    The redacted log excerpts never live here; they stay in the private
    journal row's ``diagnostic`` field and are never projected.
    """

    path: str
    digest: str
    setup_status: int | str = "not-run"
    setup_duration_seconds: float = 0.0
    setup_log_digest: str = ""
    teardown_status: int | str = "not-run"
    teardown_duration_seconds: float = 0.0
    teardown_log_digest: str = ""
    cleanup: str = "unconfirmed"
    inventoried: int = 0
    swept: int = 0
    escaped: int = 0
    escaped_processes: tuple[str, ...] = ()

    @property
    def identity(self) -> tuple[str, str]:
        return (self.path, self.digest)

    @property
    def clean(self) -> bool:
        return (
            self.setup_status == 0
            and not isinstance(self.setup_status, bool)
            and self.teardown_status == 0
            and not isinstance(self.teardown_status, bool)
            and self.cleanup == "confirmed"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": _bounded(self.path, MAX_FIXTURE_PATH_BYTES),
            "digest": self.digest,
            "setup_status": self.setup_status,
            "setup_duration_seconds": self.setup_duration_seconds,
            "setup_log_digest": self.setup_log_digest,
            "teardown_status": self.teardown_status,
            "teardown_duration_seconds": self.teardown_duration_seconds,
            "teardown_log_digest": self.teardown_log_digest,
            "cleanup": self.cleanup,
            "inventoried": self.inventoried,
            "swept": self.swept,
            "escaped": self.escaped,
            "escaped_processes": list(self.escaped_processes[:MAX_FIXTURE_ESCAPED_REPORTED]),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "FixtureRecord":
        """Strictly decode a record; any deviation raises ``ValueError``."""
        if not isinstance(value, Mapping) or set(value) != FIXTURE_RECORD_FIELDS:
            raise ValueError("fixture record has unknown or missing fields")
        path = value["path"]
        if (
            not isinstance(path, str) or not path or "\x00" in path
            or len(path.encode("utf-8")) > MAX_FIXTURE_PATH_BYTES
            or path.startswith("/") or ".." in Path(path).parts
        ):
            raise ValueError("fixture path is malformed")
        digest = value["digest"]
        if not isinstance(digest, str) or not _FIXTURE_DIGEST_RE.match(digest):
            raise ValueError("fixture digest is malformed")
        cleanup = value["cleanup"]
        if cleanup not in FIXTURE_CLEANUP_STATES:
            raise ValueError("fixture cleanup state is unknown")
        escaped_processes = value["escaped_processes"]
        if (
            not isinstance(escaped_processes, list)
            or len(escaped_processes) > MAX_FIXTURE_ESCAPED_REPORTED
            or any(
                not isinstance(item, str) or len(item.encode("utf-8")) > MAX_FIXTURE_ESCAPED_BYTES
                for item in escaped_processes
            )
        ):
            raise ValueError("fixture escaped list is malformed")
        return cls(
            path=path,
            digest=digest,
            setup_status=_fixture_status(value["setup_status"]),
            setup_duration_seconds=_fixture_duration(value["setup_duration_seconds"]),
            setup_log_digest=_fixture_log_digest(value["setup_log_digest"]),
            teardown_status=_fixture_status(value["teardown_status"]),
            teardown_duration_seconds=_fixture_duration(value["teardown_duration_seconds"]),
            teardown_log_digest=_fixture_log_digest(value["teardown_log_digest"]),
            cleanup=str(cleanup),
            inventoried=_fixture_count(value["inventoried"]),
            swept=_fixture_count(value["swept"]),
            escaped=_fixture_count(value["escaped"]),
            escaped_processes=tuple(_safe_text(item, MAX_FIXTURE_ESCAPED_BYTES) for item in escaped_processes),
        )


def unreadable_fixture_record(value: object = None) -> FixtureRecord:
    """A fail-closed stand-in for a malformed or oversized persisted record."""
    path = ""
    if isinstance(value, Mapping) and isinstance(value.get("path"), str):
        path = _safe_text(value["path"], MAX_FIXTURE_PATH_BYTES)
    return FixtureRecord(path=path or "<unreadable>", digest="unreadable", cleanup="unconfirmed")


def decode_fixture_record(value: object) -> tuple[FixtureRecord, tuple[str, ...]]:
    """Decode defensively: a bad value is unconfirmed with a caveat, never raises."""
    try:
        return FixtureRecord.from_mapping(value), ()
    except Exception:
        # Any malformed value (including arithmetic overflow) fails closed.
        return unreadable_fixture_record(value), (FIXTURE_UNREADABLE_CAVEAT,)


TOOL_COMMIT_ENV = "AGENT_LOOP_RUN_TESTS_TOOL_COMMIT"
TOOL_DIRTY_ENV = "AGENT_LOOP_RUN_TESTS_TOOL_DIRTY"
TOOL_PACKAGE_ENV = "AGENT_LOOP_RUN_TESTS_TOOL_PACKAGE"
TOOL_IDENTITY_ENVS = (TOOL_COMMIT_ENV, TOOL_DIRTY_ENV, TOOL_PACKAGE_ENV)
TOOL_IDENTITY_STATUSES = frozenset({"match", "mismatch", "unidentified", "unknown"})
TOOL_MISMATCH_DIAGNOSTIC = "run-tests-tool-mismatch"
TOOL_UNIDENTIFIED_DIAGNOSTIC = "run-tests-tool-unidentified"
_TOOL_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_TOOL_DIGEST_RE = re.compile(r"^[0-9a-f]{12}$")


def _tool_commit(value: object) -> str | None:
    if isinstance(value, str) and _TOOL_COMMIT_RE.match(value.strip().lower()):
        return value.strip().lower()
    return None


def _tool_digest(value: object) -> str | None:
    return value if isinstance(value, str) and _TOOL_DIGEST_RE.match(value) else None


def _tool_dirty(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {"true": True, "false": False}.get(value.strip().lower())
    return None


@dataclass(frozen=True)
class ToolIdentity:
    """The ``run-tests`` client's agent-loop identity against the orchestrator's (#1632).

    Unauthenticated diagnostic data: it never changes provenance,
    supersession or the evidence gate.  Raw package paths are never stored;
    only short digests are.
    """

    status: str
    client_commit: str | None = None
    client_dirty: bool | None = None
    client_package_digest: str | None = None
    orchestrator_commit: str | None = None
    orchestrator_dirty: bool | None = None
    orchestrator_package_digest: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "client_commit": self.client_commit,
            "client_dirty": self.client_dirty,
            "client_package_digest": self.client_package_digest,
            "orchestrator_commit": self.orchestrator_commit,
            "orchestrator_dirty": self.orchestrator_dirty,
            "orchestrator_package_digest": self.orchestrator_package_digest,
        }

    @classmethod
    def from_value(cls, value: object) -> "ToolIdentity | None":
        """Decode defensively; any malformed value yields ``None``."""
        if isinstance(value, ToolIdentity):
            return value
        if not isinstance(value, Mapping):
            return None
        try:
            status = value.get("status")
            # A non-string (possibly unhashable) status is malformed.
            if not isinstance(status, str) or status not in TOOL_IDENTITY_STATUSES:
                return None
            return cls(
                status=str(status),
                client_commit=_tool_commit(value.get("client_commit")),
                client_dirty=_tool_dirty(value.get("client_dirty")),
                client_package_digest=_tool_digest(value.get("client_package_digest")),
                orchestrator_commit=_tool_commit(value.get("orchestrator_commit")),
                orchestrator_dirty=_tool_dirty(value.get("orchestrator_dirty")),
                orchestrator_package_digest=_tool_digest(value.get("orchestrator_package_digest")),
            )
        except Exception:  # noqa: BLE001 - malformed input never raises
            return None


def _package_path_digest(path: object) -> str | None:
    if not isinstance(path, str) or not path.strip():
        return None
    from .agent_executable import package_digest

    return package_digest(path.strip())


def build_tool_identity(
    client: Mapping[str, str], orchestrator: Mapping[str, object]
) -> ToolIdentity:
    """Compare the client's reported identity with the orchestrator's capture."""
    reported = any(name in client for name in TOOL_IDENTITY_ENVS)
    client_commit = _tool_commit(client.get(TOOL_COMMIT_ENV))
    client_digest = _package_path_digest(client.get(TOOL_PACKAGE_ENV))
    orchestrator_commit = _tool_commit(orchestrator.get("commit"))
    orchestrator_digest = _package_path_digest(orchestrator.get("package_path"))
    if not reported:
        status = "unidentified"
    elif client_commit and orchestrator_commit:
        status = "match" if client_commit == orchestrator_commit else "mismatch"
    elif client_digest and orchestrator_digest and client_digest != orchestrator_digest:
        status = "mismatch"
    else:
        status = "unknown"
    return ToolIdentity(
        status=status,
        client_commit=client_commit,
        client_dirty=_tool_dirty(client.get(TOOL_DIRTY_ENV)),
        client_package_digest=client_digest,
        orchestrator_commit=orchestrator_commit,
        orchestrator_dirty=_tool_dirty(orchestrator.get("dirty")),
        orchestrator_package_digest=orchestrator_digest,
    )


def _orchestrator_tool_provenance() -> Mapping[str, object]:
    from .tool_provenance import orchestrator_tool_provenance

    return orchestrator_tool_provenance()


def _tool_label(commit: str | None, dirty: bool | None, digest: str | None) -> str:
    if commit:
        return commit[:7] + (" dirty" if dirty else "")
    if digest:
        return f"package {digest}"
    return "unknown"


def tool_identity_diagnostic(row: object) -> str | None:
    """The ``run-tests-tool-*`` diagnostic for a row's tool identity, or None."""
    identity = getattr(row, "tool_identity", None)
    if identity is None and isinstance(row, Mapping):
        identity = ToolIdentity.from_value(row.get("tool_identity"))
    if not isinstance(identity, ToolIdentity):
        return None
    if identity.status == "mismatch":
        if identity.client_commit and identity.orchestrator_commit:
            client = _tool_label(identity.client_commit, identity.client_dirty, None)
            orchestrator = _tool_label(identity.orchestrator_commit, identity.orchestrator_dirty, None)
        else:
            client = _tool_label(None, None, identity.client_package_digest)
            orchestrator = _tool_label(None, None, identity.orchestrator_package_digest)
        return (
            f"{TOOL_MISMATCH_DIAGNOSTIC}: receipt produced by agent-loop {client}, "
            f"orchestrator is {orchestrator}"
        )
    if identity.status == "unidentified":
        return (
            f"{TOOL_UNIDENTIFIED_DIAGNOSTIC}: run-tests client did not report its agent-loop "
            "identity (likely an older install)"
        )
    return None


@dataclass(frozen=True)
class LocalTestObservation:
    command: tuple[str, ...]
    outcome: str
    provenance: str
    scope: EvidenceScope = field(default_factory=EvidenceScope)
    receipt_id: str | None = None
    # Invocation-local selector. It is intentionally omitted from the durable
    # public projection; only the live broker/catalog may resolve it.
    execution_ref: str | None = field(default=None, repr=False, compare=False)
    turn_id: str | None = None
    timestamp: str = ""
    cwd: str | None = None
    normalized_command: str | None = None
    returncode: int | None = None
    attribution: TreeAttribution = field(default_factory=TreeAttribution)
    environment_state: str = "unknown"
    environment_identity: EnvironmentIdentity | None = field(
        default=None, repr=False, compare=False
    )
    caveats: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    diagnostic: str | None = field(default=None, repr=False, compare=False)
    claim: str | None = None
    superseded_by: str | None = None
    wrapper_bootstrap: str = "unknown"
    inner_exec: str = "not-attempted"
    suite_start: str = "not-started"
    # Live-only exact spawned target of an adopted package-script run (issue
    # #1294), after worker policy and launcher rebinding.  Never projected;
    # restored rows carry none, so a package-manager row fails closed.
    executed_command: tuple[str, ...] = field(default=(), repr=False, compare=False)
    # ``True`` only while every projected command token equals the raw token
    # byte for byte; ``None`` is a fresh live row that was never redacted.
    command_verbatim: bool | None = field(default=None, compare=False)
    # Issue #1326: the ``run-tests --fixture`` lifecycle; ``None`` for every
    # fixture-less row, which keeps the legacy projection unchanged.
    fixture: FixtureRecord | None = None
    # Issue #1633: the broker-computed command identity that survives a
    # restart (see ``_restart_command_digest``).  ``None`` for every row the
    # broker did not create, which never takes part in cross-restart
    # supersession.
    restart_command_digest: str | None = field(default=None, repr=False, compare=False)
    # Issue #1632: the run-tests client's tool identity, set by the broker on
    # every journaled row.  Diagnostic only; ``None`` keeps legacy rows and
    # projections unchanged.
    tool_identity: ToolIdentity | None = field(default=None, compare=False)
    # Issue #1633: set only by ``decode_bounded_evidence``.  Never projected,
    # so a persisted payload cannot claim it.
    restored: bool = field(default=False, repr=False, compare=False)
    # Issue #1661: why a launched run's suite start stayed ``unknown``.
    # Diagnostic only: equality, the observation key, admission and
    # citability never read it.  Anything outside the closed set, or beside
    # another suite start, is normalized to ``None``.
    suite_start_reason: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.suite_start_reason is not None:
            object.__setattr__(
                self, "suite_start_reason",
                coerce_suite_start_reason(self.suite_start_reason, self.suite_start),
            )
        if self.outcome not in OUTCOMES:
            raise AgentLoopError(f"unknown local test outcome: {self.outcome}")
        if self.provenance not in PROVENANCES:
            raise AgentLoopError(f"unknown local test provenance: {self.provenance}")
        if self.environment_state not in ENVIRONMENT_STATES:
            raise AgentLoopError(f"unknown environment comparison: {self.environment_state}")
        if self.wrapper_bootstrap not in WRAPPER_BOOTSTRAP_STATES:
            raise AgentLoopError(f"unknown wrapper bootstrap state: {self.wrapper_bootstrap}")
        if self.inner_exec not in INNER_EXEC_STATES:
            raise AgentLoopError(f"unknown inner exec state: {self.inner_exec}")
        if self.suite_start not in SUITE_START_STATES:
            raise AgentLoopError(f"unknown suite start state: {self.suite_start}")

    @property
    def is_failure(self) -> bool:
        return self.outcome in {"failed", "timed_out", "interrupted", "incomplete", "launch-failed"}

    @property
    def is_non_evidence_launch_failure(self) -> bool:
        return is_non_evidence_launch_failure(self)

    @property
    def is_out_of_checkout_context(self) -> bool:
        return OUT_OF_CHECKOUT_CONTEXT_CAVEAT in self.caveats

    def public_projection(self) -> dict[str, object]:
        source = _apply_command_parse_guard(self)
        tokens, command, _, command_caveats = _redact_command_tokens(
            source.command,
            cwd=Path(source.cwd) if source.cwd else None,
        )
        caveats = list(source.caveats)
        outcome = source.outcome
        argv: list[str] | None = None
        if _TRUNCATED_CAVEAT in command_caveats:
            caveats.append(_TRUNCATED_CAVEAT)
            if _ARGV_OMITTED_CAVEAT not in caveats:
                fits = sum(len(t.encode()) for t in tokens) <= MAX_SAFE_ARGV_BYTES
                if not fits:
                    caveats.extend((_CAPTURE_LIMITED_CAVEAT, _ARGV_OMITTED_CAVEAT))
                    if outcome == "passed":
                        outcome = "incomplete"
                elif not _is_downgraded(caveats):
                    argv = list(tokens)
        verbatim = (
            _tokens_verbatim(source.command, tokens, command_caveats)
            and self.command_verbatim is not False
        )
        projection = {
            "command_verbatim": verbatim,
            "command": _bounded(command, MAX_SAFE_COMMAND_BYTES),
            "receipt_id": _safe_text(self.receipt_id or "", MAX_SAFE_IDENTIFIER_BYTES),
            "turn_id": _safe_text(self.turn_id or "", MAX_SAFE_IDENTIFIER_BYTES),
            "timestamp": _bounded(_safe_text(self.timestamp, 64), 64),
            "outcome": outcome,
            "provenance": self.provenance,
            "scope": self.scope.to_dict(),
            "attribution": self.attribution.to_dict(),
            "environment": self.environment_state,
            "returncode": self.returncode,
            "claim": self.claim if self.claim in CLAIMS else None,
            "superseded_by": _safe_text(self.superseded_by or "", MAX_SAFE_IDENTIFIER_BYTES)
            if self.superseded_by
            else None,
            "caveats": [
                _safe_text(item, MAX_SAFE_CAVEAT_BYTES)
                for item in _prioritized_caveats(caveats)[:4]
            ],
            "identifiers": [
                _safe_text(item, MAX_SAFE_IDENTIFIER_BYTES)
                for item in self.identifiers[:MAX_SAFE_IDENTIFIERS]
            ],
        }
        if argv is not None:
            projection["argv"] = argv
        # Keep the legacy wire size stable for restored rows while carrying
        # explicit state whenever a live runner supplied it.
        if (self.wrapper_bootstrap, self.inner_exec, self.suite_start) != (
            "unknown", "not-attempted", "not-started"
        ):
            projection.update(
                wrapper_bootstrap=self.wrapper_bootstrap,
                inner_exec=self.inner_exec,
                suite_start=self.suite_start,
            )
        reason = coerce_suite_start_reason(self.suite_start_reason, self.suite_start)
        if reason is not None:
            # Only rows that carry one grow; legacy projections are unchanged.
            projection["suite_start_reason"] = reason
        if self.fixture is not None:
            projection["fixture"] = self.fixture.to_dict()
        if self.restart_command_digest is not None and _RESTART_COMMAND_DIGEST_RE.match(
            self.restart_command_digest
        ):
            projection["restart_command_digest"] = self.restart_command_digest
        if isinstance(self.tool_identity, ToolIdentity):
            # Its own field, outside the four-caveat cap (#1632).
            projection["tool_identity"] = self.tool_identity.to_dict()
        return projection

    def to_dict(self) -> dict[str, object]:
        return self.public_projection()

    def live_projection(self) -> dict[str, object]:
        """Return the bounded live catalog view, including the ephemeral selector."""
        projection = self.public_projection()
        if self.execution_ref:
            projection["execution_ref"] = _safe_text(self.execution_ref, MAX_SAFE_IDENTIFIER_BYTES)
        return projection


def mark_out_of_checkout_context(
    observations: Iterable[LocalTestObservation], *, assigned_workdir: Path
) -> tuple[LocalTestObservation, ...]:
    """Label live runs whose test operands escape the assigned checkout (#991).

    Only live observations carry the raw argv this needs; restored rows keep
    the persisted caveat instead.  The caveat is placed first so the bounded
    public projection, which keeps only the leading caveats, retains it.
    """
    from .workdir_guard import command_targets_outside_workdir

    marked: list[LocalTestObservation] = []
    for row in observations:
        if not row.is_out_of_checkout_context and command_targets_outside_workdir(
            row.executed_command or row.command, assigned_workdir=assigned_workdir
        ):
            row = replace(row, caveats=(OUT_OF_CHECKOUT_CONTEXT_CAVEAT, *row.caveats))
        marked.append(row)
    return tuple(marked)


_TRUNCATED_CAVEAT = "safe command truncated"
_ARGV_OMITTED_CAVEAT = "safe command truncated; argv omitted"
_CAPTURE_LIMITED_CAVEAT = "capture-limited"
_UNPARSABLE_PREFIX = "unparsable command:"


def _split_command_text(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a command string; on failure return a single token and caveats."""
    try:
        return tuple(shlex.split(text)), ()
    except ValueError as exc:
        return (text,), (
            _CAPTURE_LIMITED_CAVEAT,
            f"{_UNPARSABLE_PREFIX} {_safe_text(exc)}",
        )


def _apply_command_parse_guard(observation: LocalTestObservation) -> LocalTestObservation:
    """Downgrade a string command that cannot be parsed instead of raising (#1139)."""
    if not isinstance(observation.command, str):
        return observation
    argv, fallback = _split_command_text(observation.command)
    if not fallback:
        return replace(observation, command=argv)
    return replace(
        observation,
        command=argv,
        outcome="incomplete",
        caveats=tuple(dict.fromkeys((*fallback, *observation.caveats))),
    )


# A bare flag never carries an attached value: one letter after a single dash
# (``-pS3CRET`` is a value) or a long ``--name``.
_SAFE_FLAG_RE = re.compile(r"-[A-Za-z]|--[A-Za-z][A-Za-z0-9_-]*")
_PLACEHOLDER = r"<(?:redacted|sha256|arg-sha256|path-sha256|userinfo):[0-9a-f]{16}>"
_PLACEHOLDER_TOKEN_RE = re.compile(
    rf"(?:(?:--[A-Za-z][A-Za-z0-9_-]*|[A-Za-z_][A-Za-z0-9_]*)=)?{_PLACEHOLDER}"
)


def _tail_token_is_safe(token: str) -> bool:
    return bool(
        _SAFE_FLAG_RE.fullmatch(token)
        or _PLACEHOLDER_TOKEN_RE.fullmatch(token)
        or (_REPO_RELATIVE_RE.match(token) and "=" not in token)
    )


def _seal_unverified_tail(tokens: Sequence[str]) -> list[str]:
    """Digest every token the bounded display does not fully show unless safe.

    The durable argv extends past the 512-byte display, and redaction leaves
    unrecognised ``-`` options and short positionals verbatim.  Tokens that
    are not fully displayed are kept only when they are a bare flag, a
    whole-token digest placeholder (optionally behind a ``name=`` prefix) or a
    repo-relative path; anything else, including a token straddling the
    cutoff, becomes an ``<arg-sha256:...>`` placeholder.  Replacements shorten
    or lengthen tokens and so move the cutoff, so this runs to a fixed point,
    which keeps re-redaction idempotent.
    """
    result = list(tokens)
    # _bounded keeps limit - 3 bytes and appends "...", so a token is fully
    # displayed only when it ends within that prefix.
    limit = MAX_SAFE_COMMAND_BYTES - 3
    while True:
        offset = -1  # shlex.join separates tokens with one space
        offsets = []
        for token in result:
            offset += 1 + len(shlex.quote(token).encode())
            offsets.append(offset)
        if not offsets or offsets[-1] <= MAX_SAFE_COMMAND_BYTES:
            return result  # no truncation: the display shows every token
        for index, token in enumerate(result):
            if offsets[index] <= limit or _tail_token_is_safe(token):
                continue
            result[index] = f"<arg-sha256:{_digest(token)}>"
            break
        else:
            return result


def _is_downgraded(caveats: Iterable[str]) -> bool:
    return any(
        item == _CAPTURE_LIMITED_CAVEAT or item.startswith(_UNPARSABLE_PREFIX)
        for item in caveats
    )


def is_non_evidence_launch_failure(row: object) -> bool:
    """Whether a row (observation, projection or mapping) is a non-evidence launch failure."""
    if isinstance(row, Mapping):
        outcome, caveats = row.get("outcome"), row.get("caveats")
    else:
        outcome, caveats = getattr(row, "outcome", None), getattr(row, "caveats", None)
    if outcome != "launch-failed" or not isinstance(caveats, (list, tuple)):
        return False
    return any(
        isinstance(item, str) and item.startswith(PRE_COLLECTION_LAUNCH_FAILURE_PREFIX)
        for item in caveats
    )


def _prioritized_caveats(caveats: Iterable[str]) -> list[str]:
    unique = list(dict.fromkeys(caveats))
    has_omitted = _ARGV_OMITTED_CAVEAT in unique
    first_unparsable = next((c for c in unique if c.startswith(_UNPARSABLE_PREFIX)), None)

    def rank(item: str) -> int:
        if item.startswith(PRE_COLLECTION_LAUNCH_FAILURE_PREFIX):
            return 0
        if item == OUT_OF_CHECKOUT_CONTEXT_CAVEAT:
            return 1
        if item == _CAPTURE_LIMITED_CAVEAT:
            return 2
        if item == first_unparsable:
            return 3
        if item in {_TRUNCATED_CAVEAT, _ARGV_OMITTED_CAVEAT}:
            return 4
        if item in {ENVIRONMENT_DIFFERS_CAVEAT, CROSS_RESTART_SUPERSESSION_CAVEAT}:
            # Issue #1633: verdicts that must survive repeated transport.
            return 5
        return 6

    kept = [c for c in unique if not (has_omitted and c == _TRUNCATED_CAVEAT)]
    return sorted(kept, key=rank)


def _coerce_command(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            return tuple(shlex.split(value))
        except ValueError:
            return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _coerce_attribution(value: object) -> TreeAttribution:
    if isinstance(value, TreeAttribution):
        return value
    if not isinstance(value, Mapping):
        return TreeAttribution()
    return TreeAttribution(
        state=str(value.get("state", "unknown")),
        head=str(value["head"]) if value.get("head") is not None else None,
        pre_digest=str(value["pre_digest"]) if value.get("pre_digest") is not None else None,
        post_digest=str(value["post_digest"]) if value.get("post_digest") is not None else None,
        tracked_digest=(
            str(value["tracked_digest"]) if value.get("tracked_digest") is not None else None
        ),
        stable=value.get("stable") if isinstance(value.get("stable"), bool) else None,
        untracked_input=bool(value.get("untracked_input", False)),
        caveats=tuple(_safe_text(item) for item in value.get("caveats", ()) if item is not None),
    )


def observation_from_mapping(
    value: Mapping[str, object],
    *,
    cwd: Path | None = None,
    registry: EnvironmentIdentityRegistry | None = None,
    default_provenance: str = "telemetry-unverified",
) -> LocalTestObservation:
    raw_command = value.get("argv", value.get("command", ()))
    fallback_caveats: tuple[str, ...] = ()
    if isinstance(raw_command, str):
        command, fallback_caveats = _split_command_text(raw_command)
    else:
        command = _coerce_command(raw_command)
    from .test_runtime import normalize_test_command

    base = cwd or Path.cwd()
    try:
        normalized = str(value.get("normalized_command") or normalize_test_command(command, cwd=base))
    except Exception:
        normalized = _safe_text(value.get("command", ""), MAX_SAFE_COMMAND_BYTES)
    environment_identity = value.get("environment_identity")
    if not isinstance(environment_identity, EnvironmentIdentity):
        environment_identity = None
    environment_state = str(value.get("environment_state", value.get("environment", "unknown")))
    if environment_state not in ENVIRONMENT_STATES:
        environment_state = "unknown"
    fixture: FixtureRecord | None = None
    fixture_caveats: tuple[str, ...] = ()
    restart_digest = value.get("restart_command_digest")
    if not isinstance(restart_digest, str) or not _RESTART_COMMAND_DIGEST_RE.match(restart_digest):
        restart_digest = None
    if "fixture" in value:
        raw_fixture = value.get("fixture")
        if isinstance(raw_fixture, FixtureRecord):
            fixture = raw_fixture
        else:
            fixture, fixture_caveats = decode_fixture_record(raw_fixture)
    return LocalTestObservation(
        command=command,
        outcome="incomplete" if fallback_caveats else str(value.get("outcome", "incomplete")),
        provenance=str(value.get("provenance", default_provenance)),
        scope=EvidenceScope.from_value(value.get("scope")),
        receipt_id=str(value["receipt_id"]) if value.get("receipt_id") is not None else None,
        execution_ref=(
            str(value["execution_ref"])
            if value.get("execution_ref") is not None else None
        ),
        turn_id=str(value["turn_id"]) if value.get("turn_id") is not None else None,
        timestamp=_timestamp(value.get("timestamp")),
        cwd=str(value["cwd"]) if value.get("cwd") is not None else None,
        normalized_command=normalized,
        returncode=(int(value["returncode"]) if isinstance(value.get("returncode"), int) else None),
        attribution=_coerce_attribution(value.get("attribution", value.get("tree"))),
        environment_state=environment_state,
        environment_identity=environment_identity,
        caveats=tuple(
            dict.fromkeys(
                (
                    *fallback_caveats,
                    *fixture_caveats,
                    *(_safe_text(item) for item in value.get("caveats", ()) if item is not None),
                )
            )
        ),
        identifiers=tuple(_safe_text(item, MAX_SAFE_IDENTIFIER_BYTES) for item in value.get("identifiers", ()) if item is not None),
        diagnostic=str(value["diagnostic"]) if value.get("diagnostic") is not None else None,
        claim=str(value["claim"]) if value.get("claim") is not None else None,
        wrapper_bootstrap=str(value.get("wrapper_bootstrap", "unknown")),
        inner_exec=str(value.get("inner_exec", "not-attempted")),
        suite_start=str(value.get("suite_start", "not-started")),
        suite_start_reason=coerce_suite_start_reason(
            value.get("suite_start_reason"), value.get("suite_start", "not-started")
        ),
        # A restored row cannot prove its tokens were never altered.
        command_verbatim=value.get("command_verbatim") is True,
        fixture=fixture,
        restart_command_digest=restart_digest,
        tool_identity=ToolIdentity.from_value(value.get("tool_identity")),
    )


def _observation_key(observation: LocalTestObservation) -> tuple[object, ...]:
    return (
        observation.receipt_id,
        observation.normalized_command,
        observation.outcome,
        observation.provenance,
        observation.scope.normalized(),
        observation.attribution.to_dict(),
        observation.environment_state,
        observation.returncode,
        # Issue #1326: a fixture run never pools with another fixture or none.
        observation.fixture.identity if observation.fixture is not None else None,
    )


def _fixture_identity(observation: LocalTestObservation) -> tuple[str, str] | None:
    return observation.fixture.identity if observation.fixture is not None else None


def _same_receipt(left: LocalTestObservation, right: LocalTestObservation) -> bool:
    return left.receipt_id is not None and left.receipt_id == right.receipt_id


def _restart_command_digest(argv: Sequence[str], cwd: Path, root: Path) -> str | None:
    """Return the restart-stable command identity, or ``None`` (issue #1633).

    The digest is an injective function of what ran: the checkout-relative
    cwd and the raw argv, modulo only the location of the checkout root.  The
    single rewrite substitutes the exact root string by a NUL marker, which is
    invertible because argv and paths cannot contain NUL; length prefixes fix
    the field boundaries.  Nothing is resolved or normalized, and nothing is
    derived from the deliberately lossy redaction output.  Any input that
    cannot be encoded that way, or whose redacted argv would exceed the argv
    cap (and so be downgraded on transport), gets no digest and fails closed.
    """
    if not isinstance(root, Path):
        raise TypeError("restart command digest root must be a Path")
    root_text = str(root)
    cwd_text = str(cwd)
    if cwd_text == root_text:
        relative = "."
    elif cwd_text.startswith(root_text + os.sep):
        relative = cwd_text[len(root_text) + 1 :]
    else:
        return None
    tokens = tuple(argv)
    if any(not isinstance(token, str) or "\x00" in token for token in tokens) or "\x00" in relative:
        return None
    redacted, _command, _ids, _caveats = _redact_command_tokens(tokens, cwd=cwd)
    if sum(len(token.encode()) for token in redacted) > MAX_SAFE_ARGV_BYTES:
        return None
    digest = hashlib.sha256(_RESTART_COMMAND_DOMAIN)
    try:
        for text in (relative, *(token.replace(root_text, _RESTART_ROOT_MARKER) for token in tokens)):
            encoded = text.encode("utf-8", errors="surrogateescape")
            digest.update(struct.pack(">I", len(encoded)))
            digest.update(encoded)
    except UnicodeEncodeError:
        return None
    return digest.hexdigest()[:32]


def _snapshot_positively_known(
    current_snapshot: "TrackedTreeSnapshot | None", current_head: str | None
) -> bool:
    return bool(
        current_head
        and current_snapshot is not None
        and current_snapshot.complete
        and current_snapshot.stable is True
        and current_snapshot.status_clean is True
        and current_snapshot.tracked_digest
        and current_snapshot.head == current_head
    )


def _restart_fixture_key(
    observation: LocalTestObservation,
) -> tuple[bool, tuple[str, str] | None]:
    """Transport-normalized fixture identity; ``(False, None)`` fails closed."""
    fixture = observation.fixture
    if fixture is None:
        return True, None
    path_bytes = len(fixture.path.encode("utf-8", errors="replace"))
    if (
        path_bytes > MAX_FIXTURE_PATH_BYTES
        # A transported path at the bound may be a truncation of a longer
        # one, so it can never prove identity either.
        or (path_bytes >= MAX_FIXTURE_PATH_BYTES - 6 and fixture.path.endswith("..."))
        or not _FIXTURE_DIGEST_RE.match(fixture.digest)
    ):
        return False, None
    return True, (fixture.path, fixture.digest)


def _restart_scope_key(observation: LocalTestObservation) -> tuple[str, tuple[str, ...]]:
    return EvidenceScope.from_value(observation.scope.to_dict()).normalized()


@dataclass(frozen=True)
class _CrossRestartContext:
    """The reconcile's validated current state for cross-restart supersession."""

    current_head: str | None
    current_snapshot: "TrackedTreeSnapshot | None"
    snapshot_head_matches: bool
    refreshed_restored: frozenset[int]
    # Each row's own head as it entered this reconcile, before the handoff
    # refresh rewrites a live row's head to the current one.  Eligibility is
    # judged on these, never on a rewritten attribution.
    entry_heads: tuple[str | None, ...] = ()
    base_reproduction_rows: frozenset[int] = frozenset()


def _can_supersede_across_restart(
    failure: LocalTestObservation,
    passing: LocalTestObservation,
    *,
    context: _CrossRestartContext | None,
    failure_index: int,
    passing_index: int,
) -> tuple[bool, str | None]:
    """The gated branch for a failure restored from an earlier run (#1633).

    It never compares redaction-rewritten fields (``normalized_command``,
    ``command``, ``identifiers``); command identity is the broker digest.  A
    persisted state or caveat never satisfies a check alone: restored rows
    qualify only through the attribution refresh of this reconcile.
    """
    if failure.provenance != "parent-observed":
        return False, None
    if (
        context is None
        or not context.snapshot_head_matches
        or not _snapshot_positively_known(context.current_snapshot, context.current_head)
    ):
        return False, "current head or tracked tree is not positively known"
    assert context.current_snapshot is not None
    if passing.environment_identity is None and not passing.restored:
        return False, "later pass has no live or restored identity"
    unstable_failure = failure.attribution.stable is False
    if unstable_failure and (
        passing.timestamp <= failure.timestamp
        or failure.claim == "base-reproduction"
        or failure.attribution.state == "base-reproduction"
        or passing.claim == "base-reproduction"
        or passing_index in context.base_reproduction_rows
    ):
        return False, None
    for row, index in ((failure, failure_index), (passing, passing_index)):
        # An explicitly unstable failure cannot establish tree authority.
        # Only its side is waived; the pass must still be freshly confirmed.
        if row is failure and unstable_failure:
            continue
        if row.restored and index not in context.refreshed_restored:
            return False, "restored attribution was not confirmed against the current head"
        if not 0 <= index < len(context.entry_heads) or (
            context.entry_heads[index] != context.current_head
        ):
            return False, "row was not recorded at the current head"
        attribution = row.attribution
        if (
            attribution.state != "current-head"
            or attribution.head != context.current_head
            or attribution.stable is not True
            or attribution.tracked_digest != context.current_snapshot.tracked_digest
            or attribution.untracked_input
        ):
            return False, "attribution is not stable current-head at the current tracked digest"
    if (
        failure.restart_command_digest is None
        or passing.restart_command_digest is None
        or failure.restart_command_digest != passing.restart_command_digest
    ):
        return False, "command differs; subset/superset or command disagreement remains visible"
    failure_fixture_ok, failure_fixture = _restart_fixture_key(failure)
    passing_fixture_ok, passing_fixture = _restart_fixture_key(passing)
    if not failure_fixture_ok or not passing_fixture_ok or failure_fixture != passing_fixture:
        return False, "fixture identity differs; a run with another fixture or none cannot supersede it"
    if _restart_scope_key(failure) != _restart_scope_key(passing):
        return False, "scope differs; subset/superset result cannot supersede the failure"
    if (
        passing.environment_state == "different"
        or ENVIRONMENT_DIFFERS_CAVEAT in passing.caveats
    ):
        return False, "environment identity is different"
    if (
        failure.is_out_of_checkout_context
        or passing.is_out_of_checkout_context
        or failure.is_non_evidence_launch_failure
    ):
        return False, None
    return True, None


def _can_supersede(
    failure: LocalTestObservation,
    passing: LocalTestObservation,
    *,
    registry: EnvironmentIdentityRegistry | None,
    context: _CrossRestartContext | None = None,
    failure_index: int = -1,
    passing_index: int = -1,
) -> tuple[bool, str | None]:
    if not failure.is_failure or passing.outcome != "passed":
        return False, None
    if passing.provenance != "parent-observed":
        return False, "later pass was not parent-observed"
    if failure.restored:
        return _can_supersede_across_restart(
            failure, passing, context=context,
            failure_index=failure_index, passing_index=passing_index,
        )
    # A live failure keeps the strict in-process rule: a restored pass has no
    # identity, so the comparison below can never be "equivalent".
    if failure.normalized_command != passing.normalized_command:
        return False, "command differs; subset/superset or command disagreement remains visible"
    if _fixture_identity(failure) != _fixture_identity(passing):
        return False, "fixture identity differs; a run with another fixture or none cannot supersede it"
    if failure.scope.normalized() != passing.scope.normalized():
        return False, "scope differs; subset/superset result cannot supersede the failure"
    if failure.attribution.stable is False:
        if (
            failure.provenance != "parent-observed"
            or passing.timestamp <= failure.timestamp
            or failure.claim == "base-reproduction"
            or failure.attribution.state == "base-reproduction"
            or passing.claim == "base-reproduction"
            or context is not None and passing_index in context.base_reproduction_rows
        ):
            return False, None
        if context is None or not _snapshot_positively_known(
            context.current_snapshot, context.current_head
        ):
            return False, "current head or tracked tree is not positively known"
        assert context.current_snapshot is not None
        attribution = passing.attribution
        if (
            attribution.state != "current-head"
            or attribution.stable is not True
            or attribution.head != context.current_head
            or attribution.tracked_digest != context.current_snapshot.tracked_digest
            or attribution.untracked_input
        ):
            return False, "later pass is not stable current-head at the current tracked digest"
        comparison = compare_environment_identities(
            passing.environment_identity, failure.environment_identity, registry=registry
        )
        return comparison == "equivalent", (
            None if comparison == "equivalent" else f"environment identity is {comparison}"
        )
    if failure.attribution.state != passing.attribution.state:
        return False, "tracked-tree attribution differs"
    if failure.attribution.state != "current-head":
        return False, "failure does not have stable current-head attribution"
    if not failure.attribution.stable or not passing.attribution.stable:
        return False, "tracked-tree snapshot was not stable"
    if failure.attribution.tracked_digest != passing.attribution.tracked_digest:
        return False, "tracked tree changed"
    comparison = compare_environment_identities(
        passing.environment_identity, failure.environment_identity, registry=registry
    )
    if comparison != "equivalent":
        return False, f"environment identity is {comparison}"
    return True, None


@dataclass(frozen=True)
class LocalTestEvidence:
    observations: tuple[LocalTestObservation, ...]
    caveats: tuple[str, ...] = ()
    capture_incomplete: bool = False
    authoritative_failures: tuple[str, ...] = ()
    legacy_capture_limited: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": LOCAL_TEST_EVIDENCE_SCHEMA,
            "observations": [item.to_dict() for item in self.observations],
            "caveats": [_safe_text(item, MAX_SAFE_CAVEAT_BYTES) for item in self.caveats],
            "capture_incomplete": self.capture_incomplete,
            "authoritative_failures": [
                _safe_text(item, MAX_SAFE_IDENTIFIER_BYTES)
                for item in self.authoritative_failures
            ],
            "legacy_capture_limited": self.legacy_capture_limited,
        }

    def public_projection(self) -> dict[str, object]:
        return self.to_dict()


def _clear_tree_change_label(observation: LocalTestObservation) -> LocalTestObservation:
    """Drop an earlier reconcile's tree-change label and caveat; keep genuine receipts.

    The caveat is stripped even when the label is already gone: mapping
    coercion and the default bounded decode drop ``superseded_by`` but keep
    caveats, and a stale caveat must never decorate a blocking failure.
    """
    label = (
        None if observation.superseded_by == TREE_CHANGE_SUPERSESSION
        else observation.superseded_by
    )
    # Issue #1633: the cross-restart caveat is likewise derived per reconcile;
    # a persisted or forged copy is never input.
    caveats = tuple(
        item
        for item in observation.caveats
        if item not in {TREE_CHANGE_CAVEAT, CROSS_RESTART_SUPERSESSION_CAVEAT}
    )
    if label == observation.superseded_by and caveats == observation.caveats:
        return observation
    return replace(observation, superseded_by=label, caveats=caveats)


@dataclass(frozen=True)
class _TreeChangeEntryFacts:
    base_reproduction: bool
    stable: bool | None
    tracked_digest: str | None

    @classmethod
    def of(cls, row: LocalTestObservation) -> "_TreeChangeEntryFacts":
        return cls(
            base_reproduction=(
                row.attribution.state == "base-reproduction"
                or row.claim == "base-reproduction"
            ),
            stable=row.attribution.stable,
            tracked_digest=row.attribution.tracked_digest,
        )


def _classify_tree_change_failures(
    rows: list[LocalTestObservation],
    entry_facts: Sequence[_TreeChangeEntryFacts],
    *,
    current_snapshot: "TrackedTreeSnapshot | None",
    snapshot_head_matches: bool,
) -> None:
    """Label failures recorded at an obsolete tracked tree as history (#1329).

    Runs only against a positively known current digest: a complete, stable,
    clean, head-matched snapshot.  Every uncertain case leaves the failure
    blocking.  Attribution is never rewritten, and base-reproduction rows (as
    recorded at entry) are never labelled.
    """
    if len(entry_facts) != len(rows):
        raise AgentLoopError("tree-change entry facts lost their row association")
    if (
        current_snapshot is None
        or not current_snapshot.complete
        or current_snapshot.stable is not True
        or current_snapshot.status_clean is not True
        or not current_snapshot.tracked_digest
        or not snapshot_head_matches
    ):
        return
    current_digest = current_snapshot.tracked_digest
    for index, row in enumerate(rows):
        facts = entry_facts[index]
        if (
            not row.is_failure
            or row.provenance != "parent-observed"
            or row.superseded_by
            or row.is_out_of_checkout_context
            or row.is_non_evidence_launch_failure
            or facts.base_reproduction
            or facts.stable is not True
            or not facts.tracked_digest
            or facts.tracked_digest == current_digest
        ):
            continue
        rows[index] = replace(
            row,
            superseded_by=TREE_CHANGE_SUPERSESSION,
            caveats=(*row.caveats, TREE_CHANGE_CAVEAT),
        )


def reconcile_test_observations(
    observations: Iterable[LocalTestObservation | Mapping[str, object]],
    *,
    current_head: str | None = None,
    current_snapshot: "TrackedTreeSnapshot | None" = None,
    registry: EnvironmentIdentityRegistry | None = None,
    legacy_tests_run: Sequence[str] | None = None,
    cwd: Path | None = None,
) -> LocalTestEvidence:
    """Chronologically reconcile measured observations.

    Supersession is intentionally strict.  A passing subset, a result from a
    changed head, or a result whose environment cannot be compared remains in
    the history and receives a caveat rather than erasing a failure.
    """
    rows: list[LocalTestObservation] = []
    for index, item in enumerate(observations):
        if isinstance(item, LocalTestObservation):
            rows.append(_clear_tree_change_label(item))
        elif isinstance(item, Mapping):
            rows.append(
                _clear_tree_change_label(
                    observation_from_mapping(item, cwd=cwd, registry=registry)
                )
            )
        else:
            raise AgentLoopError(f"local test observation {index} is not an object")
    legacy_rows: list[LocalTestObservation] = []
    if legacy_tests_run:
        legacy_rows = list(parse_legacy_tests_run(legacy_tests_run, cwd=cwd or Path.cwd()))
        rows.extend(legacy_rows)
    rows.sort(key=lambda item: (item.timestamp, item.receipt_id or ""))
    caveats: list[str] = []
    seen_receipts: dict[str, LocalTestObservation] = {}
    filtered: list[LocalTestObservation] = []
    for row in rows:
        if row.receipt_id:
            previous = seen_receipts.get(row.receipt_id)
            if previous is not None:
                if _observation_key(previous) == _observation_key(row):
                    caveats.append(f"idempotent duplicate receipt {row.receipt_id}")
                    continue
                caveats.append(f"conflicting receipt {row.receipt_id} rejected")
                filtered.append(
                    replace(
                        row,
                        outcome="incomplete",
                        provenance="telemetry-unverified",
                        caveats=(*row.caveats, "conflicting receipt rejected"),
                    )
                )
                continue
            seen_receipts[row.receipt_id] = row
        filtered.append(row)
    rows = filtered
    # Entry facts for the tree-change pass (#1329), captured once the rows
    # have their final order and before any loop rewrites attribution.  From
    # here on rows are never reordered, inserted or removed: every later loop
    # assigns ``rows[index]`` in place, so ``entry_facts[index]`` stays bound
    # to its observation.
    entry_facts = [_TreeChangeEntryFacts.of(row) for row in rows]
    entry_heads = tuple(row.attribution.head for row in rows)

    # Make live comparisons visible without persisting the environment bytes.
    # A first observation has no comparison partner; later observations with
    # the same command and scope expose divergence or unavailable identity.
    # Restored rows have no identity, so they are neither compared nor used
    # as partners (issue #1633): a live row whose only same-key predecessors
    # are restored keeps "not-compared".
    for index, row in enumerate(rows):
        if row.environment_state == "identity-unknown" or row.restored:
            continue
        comparable = next(
            (
                previous
                for previous in reversed(rows[:index])
                if not previous.restored
                and previous.normalized_command == row.normalized_command
                and previous.scope.normalized() == row.scope.normalized()
                and _fixture_identity(previous) == _fixture_identity(row)
            ),
            None,
        )
        if comparable is None:
            continue
        comparison = compare_environment_identities(
            row.environment_identity,
            comparable.environment_identity,
            registry=registry,
        )
        if comparison == "equivalent":
            rows[index] = replace(row, environment_state="equivalent")
        elif comparison == "different":
            rows[index] = replace(
                row,
                environment_state="different",
                caveats=(*row.caveats, ENVIRONMENT_DIFFERS_CAVEAT),
            )
        else:
            rows[index] = replace(
                row,
                environment_state="unknown",
                caveats=(*row.caveats, ENVIRONMENT_NOT_COMPARED_CAVEAT),
            )

    # Compare live observations with the clean eventual tree at handoff. This
    # lets a stable pre-commit test become attributable after the coder commits.
    snapshot_head_matches = (
        current_head is None
        or current_snapshot is not None and current_snapshot.head == current_head
    )
    refreshed_restored: set[int] = set()
    if (
        current_snapshot is not None
        and current_snapshot.complete
        and current_snapshot.stable is True
        and current_snapshot.status_clean is True
        and current_snapshot.tracked_digest
    ):
        for index, row in enumerate(rows):
            if row.environment_identity is None:
                # Issue #1633: a parent-observed row restored from an earlier
                # run is refreshed from its own persisted head and digest.
                # Anything else that lacks a live identity is left untouched,
                # and the later stale and tree-change passes still own every
                # restored row this does not confirm.
                attribution = row.attribution
                if (
                    row.restored
                    and row.provenance == "parent-observed"
                    and snapshot_head_matches
                    and attribution.stable is True
                    and attribution.tracked_digest == current_snapshot.tracked_digest
                    and attribution.head
                    and attribution.head == current_head
                ):
                    state = (
                        "untracked-input-unverified"
                        if attribution.untracked_input
                        or attribution.state == "untracked-input-unverified"
                        else "current-head"
                    )
                    rows[index] = replace(row, attribution=replace(attribution, state=state))
                    refreshed_restored.add(index)
                continue
            attribution = row.attribution
            if attribution.stable is not True or not attribution.tracked_digest:
                continue
            if not snapshot_head_matches:
                rows[index] = replace(
                    row,
                    attribution=replace(attribution, state="stale"),
                    caveats=(*row.caveats, "assigned checkout HEAD differs from the authenticated PR head"),
                )
            elif attribution.tracked_digest == current_snapshot.tracked_digest:
                state = (
                    "untracked-input-unverified"
                    if attribution.untracked_input
                    or attribution.state == "untracked-input-unverified"
                    else "current-head"
                )
                promoted_head = current_head or current_snapshot.head
                rows[index] = replace(
                    row,
                    attribution=replace(attribution, state=state, head=promoted_head),
                    # Issue #1633: a row whose recorded head is rewritten
                    # here was not run at the promoted head.  It keeps the
                    # in-process handoff promotion but loses the restart
                    # command digest, so the rewritten head can never make it
                    # cross-restart eligible after transport.
                    restart_command_digest=(
                        row.restart_command_digest
                        if attribution.head and attribution.head == promoted_head
                        else None
                    ),
                )
            else:
                rows[index] = replace(
                    row,
                    attribution=replace(attribution, state="stale"),
                    caveats=(*row.caveats, "tested tracked tree differs from the eventual PR tree"),
                )

    restart_context = _CrossRestartContext(
        current_head=current_head,
        current_snapshot=current_snapshot,
        snapshot_head_matches=snapshot_head_matches,
        refreshed_restored=frozenset(refreshed_restored),
        entry_heads=entry_heads,
        base_reproduction_rows=frozenset(
            index for index, facts in enumerate(entry_facts) if facts.base_reproduction
        ),
    )
    for failure_index, failure in enumerate(rows):
        if (
            not failure.is_failure
            or failure.superseded_by
            or failure.is_out_of_checkout_context
            or failure.is_non_evidence_launch_failure
        ):
            continue
        for later_index in range(failure_index + 1, len(rows)):
            later = rows[later_index]
            if later.outcome != "passed" or later.is_out_of_checkout_context:
                continue
            allowed, reason = _can_supersede(
                failure, later, registry=registry, context=restart_context,
                failure_index=failure_index, passing_index=later_index,
            )
            if allowed:
                receipt = later.receipt_id or f"observation-{rows.index(later)}"
                rows[failure_index] = replace(
                    failure,
                    superseded_by=receipt,
                    # Issue #1633: only the failure carries the caveat.
                    caveats=(
                        (*failure.caveats, CROSS_RESTART_SUPERSESSION_CAVEAT)
                        if failure.restored else failure.caveats
                    ),
                )
                break
            if reason and (
                "environment" in reason
                or "subset" in reason
                or "scope" in reason
                or "tracked tree" in reason
            ):
                caveats.append(
                    f"{failure.receipt_id or 'observation'} retained: {reason}"
                )

    # A current-head pass cannot retroactively make a changed-head failure
    # current, and restored metadata has no live identity with which to prove
    # equivalence.
    for index, row in enumerate(rows):
        updated = row
        if current_head and row.attribution.head and row.attribution.head != current_head:
            updated = replace(
                updated,
                attribution=replace(row.attribution, state="stale"),
                caveats=(*row.caveats, "observation head differs from current head"),
            )
        if row.environment_state == "not-compared" and row.environment_identity is None:
            updated = replace(
                updated,
                environment_state=environment_comparison_for_restart(),
                caveats=(*updated.caveats, "restored observation has identity-unknown environment"),
            )
        rows[index] = updated

    _classify_tree_change_failures(
        rows,
        entry_facts,
        current_snapshot=current_snapshot,
        snapshot_head_matches=snapshot_head_matches,
    )

    authoritative_failures = tuple(
        row.receipt_id or f"observation-{index}"
        for index, row in enumerate(rows)
        if row.is_failure
        and row.provenance == "parent-observed"
        and not row.superseded_by
        and not row.is_out_of_checkout_context
        and not row.is_non_evidence_launch_failure
    )
    legacy_limited = bool(legacy_rows)
    if legacy_limited:
        caveats.append("legacy tests_run is self-reported; direct shell capture is limited")
    return LocalTestEvidence(
        observations=tuple(rows),
        caveats=tuple(dict.fromkeys(_safe_text(item) for item in caveats)),
        capture_incomplete=any(row.outcome == "incomplete" for row in rows),
        authoritative_failures=authoritative_failures,
        legacy_capture_limited=legacy_limited,
    )


def reconcile_local_test_evidence(*args: Any, **kwargs: Any) -> LocalTestEvidence:
    return reconcile_test_observations(*args, **kwargs)


def reconcile_test_evidence(*args: Any, **kwargs: Any) -> LocalTestEvidence:
    return reconcile_test_observations(*args, **kwargs)


def parse_legacy_tests_run(
    tests_run: Sequence[str], *, cwd: Path, turn_id: str | None = None
) -> tuple[LocalTestObservation, ...]:
    """Convert the old display-only command list into visibly limited evidence."""
    from .test_runtime import normalize_test_command

    rows: list[LocalTestObservation] = []
    for index, declaration in enumerate(tests_run):
        command_text = str(declaration)
        caveats: list[str] = ["legacy tests_run declaration; no parent receipt"]
        argv, fallback_caveats = _split_command_text(command_text)
        if fallback_caveats:
            caveats.extend(fallback_caveats)
            outcome = "incomplete"
        else:
            if not argv or _SHELL_OPERATOR_RE.search(command_text):
                caveats.append("capture-limited shell-composed declaration")
                outcome = "incomplete"
            else:
                outcome = "passed"
        try:
            normalized = normalize_test_command(argv, cwd=cwd)
        except Exception:
            normalized = _safe_text(command_text, MAX_SAFE_COMMAND_BYTES)
        rows.append(
            LocalTestObservation(
                command=argv,
                outcome=outcome,
                provenance="self-reported",
                scope=EvidenceScope("unknown", ()),
                receipt_id=f"legacy-{index + 1}",
                turn_id=turn_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                cwd=str(cwd),
                normalized_command=normalized,
                attribution=TreeAttribution(state="unknown", caveats=("legacy declaration",)),
                environment_state="identity-unknown",
                caveats=tuple(caveats),
            )
        )
    return tuple(rows)


def _path_digest(path: str) -> str:
    return f"<path-sha256:{hashlib.sha256(path.encode('utf-8', errors='replace')).hexdigest()[:16]}>"


def _relativize_token(token: str, cwd: Path) -> str:
    candidate = Path(token).expanduser()
    if candidate.is_absolute() or _WINDOWS_PATH_RE.fullmatch(token) or _UNC_PATH_RE.fullmatch(token):
        try:
            return candidate.resolve(strict=False).relative_to(cwd.resolve()).as_posix()
        except (ValueError, OSError):
            return _path_digest(token)
    return token


def _redact_command_tokens(
    argv: Sequence[str] | str, *, cwd: Path | None = None, identifiers: Sequence[str] = ()
) -> tuple[tuple[str, ...], str, tuple[str, ...], tuple[str, ...]]:
    """Return redacted tokens plus the bounded command, identifiers and caveats."""
    base = cwd or Path.cwd()
    tokens = _coerce_command(argv)
    result: list[str] = []
    found_identifiers: list[str] = list(identifiers)
    caveats: list[str] = []
    redact_next = False
    for index, raw in enumerate(tokens):
        token = _safe_text(raw, 2048)
        if redact_next:
            result.append(
                token
                if _REDACTED_VALUE_RE.fullmatch(token)
                else f"<redacted:{_digest(token)}>"
            )
            redact_next = False
            continue
        if _SECRET_OPTION_RE.match(token):
            match = _SECRET_OPTION_RE.match(token)
            assert match is not None
            operand = match.group(2)
            safe_operand = (
                operand
                if _REDACTED_VALUE_RE.fullmatch(operand)
                else f"<redacted:{_digest(operand)}>"
            )
            result.append(match.group(1) + safe_operand)
            caveats.append("credential option redacted")
            continue
        if token.lower() in {"--token", "--password", "--secret", "--api-key"}:
            result.append(token)
            redact_next = True
            caveats.append("credential option redacted")
            continue
        assignment = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", token)
        if assignment:
            name, value = assignment.groups()
            if re.fullmatch(r"<(?:redacted|sha256):[0-9a-f]{16}>", value):
                safe_value = value
            elif _SECRET_KEY_RE.match(name):
                safe_value = f"<redacted:{_digest(value)}>"
                caveats.append("secret-like environment assignment redacted")
            else:
                safe_value = f"<sha256:{hashlib.sha256(value.encode()).hexdigest()[:16]}>"
            result.append(f"{name}={safe_value}")
            continue
        token = _URL_USERINFO_RE.sub(
            lambda match: match.group(1)
            + (
                match.group(2)
                if _USERINFO_PLACEHOLDER_RE.fullmatch(match.group(2))
                else f"<userinfo:{_digest(match.group(2))}>"
            ),
            token,
        )
        # Fix the deliberately compact lambda output without ever retaining
        # userinfo in a public string.
        token = re.sub(r"<userinfo:([^>]+)>}", r"<userinfo:\1>", token)
        if "@" in token and re.search(r"(?i)(?:postgres|mysql|mongodb|redis)://", token):
            token = _path_digest(token)
            caveats.append("DSN redacted")
        token = _WINDOWS_PATH_RE.sub(lambda match: _relativize_token(match.group(0), base), token)
        token = _UNC_PATH_RE.sub(lambda match: _relativize_token(match.group(0), base), token)
        token = _POSIX_PATH_RE.sub(lambda match: _relativize_token(match.group(0), base), token)
        auth_header = _AUTH_HEADER_RE.match(token)
        if auth_header:
            token = auth_header.group(1) + (
                auth_header.group(2)
                if _REDACTED_VALUE_RE.fullmatch(auth_header.group(2))
                else f"<redacted:{_digest(auth_header.group(2))}>"
            )
            caveats.append("authorization header redacted")
        parameter = _PARAMETER_RE.search(token)
        if parameter:
            base_token = token[: parameter.start()]
            if re.fullmatch(r"\[param-sha256:[0-9a-f]{16}\]", parameter.group(0)):
                token = f"{base_token}{parameter.group(0)}"
            else:
                safe_base = (
                    base_token
                    if _REPO_RELATIVE_RE.match(base_token) or base_token.startswith(".")
                    else _path_digest(base_token)
                )
                token = f"{safe_base}[param-sha256:{_digest(parameter.group(0))}]"
                found_identifiers.append(_digest(parameter.group(0)))
                caveats.append("parametrized test identifier redacted")
        if index > 0 and not token.startswith("-") and not parameter and not (
            _REPO_RELATIVE_RE.match(token) or token.startswith(".")
        ) and len(token) > 32:
            found_identifiers.append(_path_digest(token))
            token = f"<arg-sha256:{_digest(token)}>"
        result.append(token)
    result = _seal_unverified_tail(result)
    command = shlex.join(result)
    if len(command.encode()) > MAX_SAFE_COMMAND_BYTES:
        command = _bounded(command, MAX_SAFE_COMMAND_BYTES)
        caveats.append(_TRUNCATED_CAVEAT)
    safe_ids = tuple(_bounded(item, MAX_SAFE_IDENTIFIER_BYTES) for item in found_identifiers[:MAX_SAFE_IDENTIFIERS])
    return tuple(result), command, safe_ids, tuple(
        dict.fromkeys(_bounded(item, MAX_SAFE_CAVEAT_BYTES) for item in caveats)
    )


def _tokens_verbatim(
    raw: Sequence[str] | str, tokens: Sequence[str], caveats: Sequence[str]
) -> bool:
    """Whether redaction left every projected token identical to its raw token."""
    return (
        _TRUNCATED_CAVEAT not in caveats
        and _coerce_command(raw) == tuple(tokens)
    )


def redact_test_command(
    argv: Sequence[str] | str, *, cwd: Path | None = None, identifiers: Sequence[str] = ()
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return a bounded safe command, identifiers, and redaction caveats."""
    _, command, safe_ids, caveats = _redact_command_tokens(
        argv, cwd=cwd, identifiers=identifiers
    )
    return command, safe_ids, caveats


def redact_observation(observation: LocalTestObservation) -> LocalTestObservation:
    observation = _apply_command_parse_guard(observation)
    tokens, command, identifiers, command_caveats = _redact_command_tokens(
        observation.command, cwd=Path(observation.cwd) if observation.cwd else None,
        identifiers=observation.identifiers,
    )
    return replace(
        observation,
        command_verbatim=(
            _tokens_verbatim(observation.command, tokens, command_caveats)
            and observation.command_verbatim is not False
        ),
        normalized_command=command,
        # Take the redacted argv directly: the bounded display may end inside
        # a quoted token and must never be re-parsed (#1139).
        command=tokens,
        identifiers=identifiers,
        caveats=tuple(dict.fromkeys((*observation.caveats, *command_caveats))),
        diagnostic=None,
    )


def _row_has_evidence_downgrade(caveats: object) -> bool:
    return isinstance(caveats, list) and any(
        isinstance(item, str)
        and (item.startswith(_UNPARSABLE_PREFIX) or item == _ARGV_OMITTED_CAVEAT)
        for item in caveats
    )


def _with_bounded_classification(
    detail: dict[str, object], source: Mapping[str, object]
) -> dict[str, object]:
    """Carry the source row's supersession label into the transported detail.

    The label is display data only (#1329): retention and display consumers
    read it, while reconciliation input decodes without it.
    """
    label = source.get("superseded_by")
    if isinstance(label, str) and label:
        detail["superseded_by"] = _safe_text(label, MAX_SAFE_IDENTIFIER_BYTES)
    return detail


def bounded_evidence_for_round(evidence: LocalTestEvidence | Mapping[str, object]) -> str:
    """Encode a bounded canonical metadata field, degrading before transport."""
    source = evidence.to_dict() if isinstance(evidence, LocalTestEvidence) else dict(evidence)
    raw_rows = source.get("observations")
    rows = list(raw_rows) if isinstance(raw_rows, list) else []
    all_details = [
        _with_bounded_classification(
            redact_observation(observation_from_mapping(row)).to_dict(), row
        )
        for row in rows
        if isinstance(row, Mapping) and row.get("outcome") in OUTCOMES
    ]
    # Retain unresolved failures before any class of pass, then prefer newer
    # rows within a class. Restore chronological order for rendering.
    def retention_priority(row: Mapping[str, object]) -> int:
        if row.get("superseded_by") or is_non_evidence_launch_failure(row):
            return 0
        caveats = row.get("caveats")
        if isinstance(caveats, list) and OUT_OF_CHECKOUT_CONTEXT_CAVEAT in caveats:
            # Context is never evidence: under the count or byte cap it must
            # yield to every in-checkout row, including unresolved failures.
            return 1
        if row.get("outcome") in {
            "failed",
            "timed_out",
            "interrupted",
            "incomplete",
        }:
            return 5
        if row.get("outcome") != "passed":
            return 4
        attribution = row.get("attribution")
        state = attribution.get("state") if isinstance(attribution, Mapping) else "unknown"
        if state == "stale":
            return 2
        return 3

    indexed = list(enumerate(all_details))
    selected = sorted(
        sorted(indexed, key=lambda item: (retention_priority(item[1]), item[0]), reverse=True)[
            :MAX_ROUND_OBSERVATIONS
        ],
        key=lambda item: item[0],
    )
    details = [row for _, row in selected]
    count_truncated = len(details) != len(all_details)
    source_caveats = source.get("caveats", ())
    if not isinstance(source_caveats, (list, tuple)):
        source_caveats = ()
    source_failures = source.get("authoritative_failures", ())
    if not isinstance(source_failures, (list, tuple)):
        source_failures = ()
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": LOCAL_TEST_EVIDENCE_SCHEMA,
        "observations": details,
        "caveats": [
            _safe_text(item, MAX_SAFE_CAVEAT_BYTES)
            for item in source_caveats
            if item is not None
        ],
        "capture_incomplete": bool(source.get("capture_incomplete", False))
        or any(_row_has_evidence_downgrade(row.get("caveats")) for row in all_details),
        "authoritative_failures": [
            _safe_text(item, MAX_SAFE_IDENTIFIER_BYTES)
            for item in source_failures
            if item is not None
        ],
        "legacy_capture_limited": bool(source.get("legacy_capture_limited", False)),
    }

    def encode(candidate: list[Mapping[str, object]], *, truncated: bool = False) -> str:
        value = dict(payload)
        value["observations"] = candidate
        if truncated:
            kept_ids = {id(row) for row in candidate}
            # Issue #1632: a dropped row's tool diagnostic stays countable.
            dropped_tool = sum(
                1 for row in all_details
                if id(row) not in kept_ids and tool_identity_diagnostic(row) is not None
            )
            value["caveats"] = list(
                dict.fromkeys(
                    [
                        *value.get("caveats", []),
                        "local evidence details truncated before transport",
                        *(
                            [
                                f"{dropped_tool} dropped row(s) carried a run-tests tool "
                                "identity diagnostic"
                            ]
                            if dropped_tool
                            else []
                        ),
                    ]
                )
            )
            value["capture_incomplete"] = True
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    encoded = encode(details, truncated=count_truncated)
    if len(encoded.encode()) > MAX_ROUND_BYTES:
        retained = list(enumerate(details))
        while retained and len(
            encode([row for _, row in retained], truncated=True).encode()
        ) > MAX_ROUND_BYTES:
            drop_index, _ = min(
                retained,
                key=lambda item: (retention_priority(item[1]), item[0]),
            )
            retained = [item for item in retained if item[0] != drop_index]
        encoded = encode([row for _, row in retained], truncated=True)
    if len(encoded.encode()) > MAX_ROUND_BYTES:
        redacted = json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": LOCAL_TEST_EVIDENCE_SCHEMA,
                "observations": [],
                "counts": {
                    "total": len(all_details),
                    "failed": sum(row.get("outcome") in {"failed", "timed_out", "interrupted", "incomplete"} for row in all_details),
                },
                "caveats": ["local evidence details dropped; capture incomplete"],
                "capture_incomplete": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        encoded = redacted[:MAX_ROUND_BYTES]
    return encoded


def canonicalize_bounded_evidence(value: object) -> str | None:
    """Tolerantly normalize a legacy/malformed metadata value."""
    if isinstance(value, LocalTestEvidence):
        return bounded_evidence_for_round(value)
    if isinstance(value, Mapping):
        return bounded_evidence_for_round(value)
    if isinstance(value, str):
        parsed = decode_bounded_evidence(value, restore_classification=True)
        return bounded_evidence_for_round(parsed) if parsed is not None else None
    return None


def decode_bounded_evidence(
    value: object, *, restore_classification: bool = False
) -> LocalTestEvidence | None:
    """Decode bounded round evidence.

    By default supersession labels are dropped, so a persisted label can
    never act as reconciliation input.  ``restore_classification=True`` is a
    display-only mode for comment rendering, the unchanged-head stop sentence
    and canonicalization (#1329).
    """
    if not isinstance(value, str):
        return None
    try:
        payload = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    rows = payload.get("observations")
    if not isinstance(rows, list):
        rows = []
    observations = tuple(
        replace(
            redact_observation(observation_from_mapping(row)),
            # No persisted value can reconstitute the process-private canonical
            # bytes needed for equality. Every restored row degrades visibly.
            environment_state=environment_comparison_for_restart(),
            # Execution selectors are invocation-local and must never become
            # durable receipt authority after a restart.
            execution_ref=None,
            # Issue #1633: the only place a row is marked restored.
            restored=True,
            superseded_by=(
                _safe_text(row["superseded_by"], MAX_SAFE_IDENTIFIER_BYTES)
                if restore_classification
                and isinstance(row.get("superseded_by"), str)
                and row["superseded_by"]
                else None
            ),
        )
        for row in rows
        if isinstance(row, Mapping)
        and row.get("outcome") in OUTCOMES
        and row.get("provenance") in PROVENANCES
    )
    return LocalTestEvidence(
        observations=observations,
        caveats=tuple(_safe_text(item) for item in payload.get("caveats", ()) if item is not None),
        capture_incomplete=bool(payload.get("capture_incomplete", False))
        or any(
            item.startswith(_UNPARSABLE_PREFIX)
            for observation in observations
            for item in observation.caveats
        ),
        authoritative_failures=tuple(str(item) for item in payload.get("authoritative_failures", ()) if item),
        legacy_capture_limited=bool(payload.get("legacy_capture_limited", False)),
    )


@dataclass(frozen=True)
class TrackedTreeSnapshot:
    """Bounded, read-only snapshot used for attribution decisions."""

    root: str
    head: str | None
    digest: str | None
    tracked_digest: str | None
    status_clean: bool | None
    complete: bool
    stable: bool | None
    untracked_paths: tuple[str, ...] = ()
    referenced_untracked_paths: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    @property
    def known(self) -> bool:
        return self.complete and self.stable is True and self.digest is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "head": self.head,
            "digest": self.digest,
            "tracked_digest": self.tracked_digest,
            "status_clean": self.status_clean,
            "complete": self.complete,
            "stable": self.stable,
            "untracked_paths": list(self.untracked_paths),
            "referenced_untracked_paths": list(self.referenced_untracked_paths),
            "caveats": list(self.caveats),
        }


class SnapshotCancelled(Exception):
    """A tracked-tree snapshot was cancelled because the broker is stopping.

    Deliberately not an ``AgentLoopError``/``OSError``: it must pass through the
    incomplete-snapshot handlers and reach the broker's shutdown-completion path.
    """


SNAPSHOT_CANCEL_POLL_SECONDS = 0.05


def _snapshot_checkpoint(started: float, timeout_seconds: float, cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise SnapshotCancelled()
    if time.monotonic() - started > timeout_seconds:
        raise AgentLoopError("git snapshot time limit exceeded")


def _run_git(
    root: Path, args: Sequence[str], *, timeout: float = 10.0, cancel: Event | None = None
) -> bytes:
    from .secure_git import local_command

    command, env, pass_fds = local_command(args, checkout=root)
    if cancel is None:
        try:
            result = subprocess.run(
                command,
                cwd=root,
                env=env,
                pass_fds=pass_fds,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AgentLoopError(f"git snapshot failed: {type(exc).__name__}") from exc
        if result.returncode != 0:
            raise AgentLoopError("git snapshot command failed")
        return result.stdout
    # Cancellable: poll the child, kill it when cancelled or out of time.
    if cancel.is_set():
        raise SnapshotCancelled()
    try:
        process = subprocess.Popen(
            command, cwd=root, env=env, pass_fds=pass_fds,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
    except OSError as exc:
        raise AgentLoopError(f"git snapshot failed: {type(exc).__name__}") from exc
    deadline = time.monotonic() + timeout
    try:
        while True:
            if cancel.is_set():
                raise SnapshotCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentLoopError("git snapshot failed: TimeoutExpired")
            try:
                stdout, _stderr = process.communicate(
                    timeout=min(SNAPSHOT_CANCEL_POLL_SECONDS, remaining)
                )
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.communicate(timeout=2)
        except Exception:
            pass
        raise
    if process.returncode != 0:
        raise AgentLoopError("git snapshot command failed")
    return stdout


def _git_nul_paths(raw: bytes) -> tuple[str, ...]:
    try:
        return tuple(item.decode("utf-8", errors="surrogateescape") for item in raw.split(b"\0") if item)
    except UnicodeError as exc:
        raise AgentLoopError("git snapshot contained invalid path data") from exc


def _git_index_entries(raw: bytes) -> dict[str, tuple[str, str]]:
    """Parse stage-zero index modes and object IDs without losing odd paths."""
    entries: dict[str, tuple[str, str]] = {}
    try:
        for record in raw.split(b"\0"):
            if not record:
                continue
            metadata, separator, path_bytes = record.partition(b"\t")
            fields = metadata.split(b" ")
            if not separator or len(fields) != 3 or fields[2] != b"0":
                raise AgentLoopError("git snapshot index has unresolved stages")
            mode = fields[0].decode("ascii", errors="strict")
            object_id = fields[1].decode("ascii", errors="strict")
            relative = path_bytes.decode("utf-8", errors="surrogateescape")
            if relative in entries:
                raise AgentLoopError("git snapshot index contains duplicate paths")
            entries[relative] = (mode, object_id)
    except UnicodeError as exc:
        raise AgentLoopError("git snapshot contained invalid index data") from exc
    return entries


def _dirty_gitlink_paths(
    status_raw: bytes, index_entries: Mapping[str, tuple[str, str]]
) -> tuple[str, ...]:
    """Return gitlinks whose checked-out worktree state is not clean."""
    dirty: list[str] = []
    records = status_raw.split(b"\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise AgentLoopError("git snapshot contained malformed status data")
        state = record[:2]
        relative = record[3:].decode("utf-8", errors="surrogateescape")
        if index_entries.get(relative, (None, None))[0] == "160000":
            dirty.append(relative)
        # In porcelain v1 -z, rename/copy entries carry one extra path.
        if b"R" in state or b"C" in state:
            if index >= len(records) or not records[index]:
                raise AgentLoopError("git snapshot contained malformed rename status")
            index += 1
    return tuple(sorted(dirty))


def _dirty_populated_gitlinks(
    root: Path, index_entries: Mapping[str, tuple[str, str]], *,
    timeout_seconds: float, cancel: Event | None, started: float,
    depth: int = 0, budget: list[int] | None = None,
) -> tuple[str, ...]:
    """Inspect populated submodules through separate confined Git processes."""
    dirty: list[str] = []
    if budget is None:
        budget = [128]
    gitlinks = [(relative, object_id) for relative, (mode, object_id) in index_entries.items()
                if mode == "160000"]
    if len(gitlinks) > budget[0] or (depth >= 8 and gitlinks):
        raise AgentLoopError("gitlink inspection limit exceeded")
    for relative, object_id in gitlinks:
        budget[0] -= 1
        _snapshot_checkpoint(started, timeout_seconds, cancel)
        path = root / relative
        if not os.path.lexists(path):
            # An uninitialized gitlink without its placeholder directory has
            # no submodule worktree, so no unattributable content exists.
            continue
        if path.is_symlink() or not path.is_dir():
            raise AgentLoopError("gitlink worktree has an unsafe path")
        if not os.path.lexists(path / ".git"):
            # An uninitialized submodule is an empty placeholder directory.
            # Any content there is outside the index and cannot be attributed.
            if any(path.iterdir()):
                dirty.append(relative)
            continue
        sub_root = path.resolve(strict=True)
        if os.path.commonpath((str(root), str(sub_root))) != str(root):
            raise AgentLoopError("gitlink worktree escapes snapshot root")
        def probe(args: tuple[str, ...]) -> bytes:
            _snapshot_checkpoint(started, timeout_seconds, cancel)
            remaining = max(0.1, timeout_seconds - (time.monotonic() - started))
            return _run_git(sub_root, args, timeout=remaining, cancel=cancel)

        top = probe(("rev-parse", "--show-toplevel"))
        if Path(top.decode("utf-8", "strict").strip()).resolve(strict=True) != sub_root:
            raise AgentLoopError("gitlink worktree does not have its own Git root")
        head = probe(("rev-parse", "HEAD"))
        status = probe(("status", "--porcelain=v1", "-z", "--untracked-files=normal"))
        nested_entries = _git_index_entries(
            probe(("ls-files", "-s", "-z"))
        )
        nested_dirty = _dirty_populated_gitlinks(
            sub_root, nested_entries, timeout_seconds=timeout_seconds, cancel=cancel,
            started=started, depth=depth + 1, budget=budget,
        )
        if head.decode("ascii", "strict").strip() != object_id or status.strip() or nested_dirty:
            dirty.append(relative)
    return tuple(sorted(dirty))


def _path_is_referenced(root: Path, token: str) -> str | None:
    if not token or token.startswith("-") or "=" in token and token.split("=", 1)[0].startswith("-"):
        return None
    candidate = Path(token).expanduser()
    try:
        resolved = candidate if candidate.is_absolute() else root / candidate
        relative = resolved.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (ValueError, OSError):
        return None
    return relative.as_posix()


def _referenced_paths(root: Path, argv: Sequence[str]) -> set[str]:
    inner = argv
    try:
        from .test_runtime import parse_managed_test_invocation

        parsed = parse_managed_test_invocation(argv, allow_command_name_launcher=True)
        if parsed is not None:
            inner = parsed.inner_argv
    except Exception:
        pass
    return {
        path
        for token in inner
        if (path := _path_is_referenced(root, str(token))) is not None
    }


def _hash_snapshot_summaries(
    records: Sequence[tuple[str, int, bytes]],
) -> str:
    """Hash fixed-size summaries of records whose payloads were streamed."""
    digest = hashlib.sha256()
    for name, payload_size, payload_digest in sorted(
        records, key=lambda item: item[0].encode("utf-8", errors="surrogateescape")
    ):
        name_bytes = name.encode("utf-8", errors="surrogateescape")
        digest.update(struct.pack(">I", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack(">Q", payload_size))
        digest.update(payload_digest)
    return digest.hexdigest()


def _stream_snapshot_record(
    path: Path,
    *,
    name: str,
    prefix: bytes,
    remaining: list[int],
    started: float,
    timeout_seconds: float,
    cancel: Event | None = None,
) -> tuple[str, int, bytes]:
    """Hash one file without reading beyond the aggregate byte/time budget."""
    name_size = len(name.encode("utf-8", errors="surrogateescape")) + 16
    before = path.stat(follow_symlinks=False)
    payload_size = len(prefix) + before.st_size
    required = name_size + payload_size
    if required > remaining[0]:
        raise AgentLoopError("git snapshot byte limit exceeded")
    _snapshot_checkpoint(started, timeout_seconds, cancel)
    remaining[0] -= required
    digest = hashlib.sha256(prefix)
    read_size = 0
    with path.open("rb") as stream:
        while True:
            _snapshot_checkpoint(started, timeout_seconds, cancel)
            chunk = stream.read(min(1024 * 1024, before.st_size - read_size + 1))
            if not chunk:
                break
            read_size += len(chunk)
            if read_size > before.st_size:
                raise AgentLoopError("repository file changed during snapshot")
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    if read_size != before.st_size or (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    ) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    ):
        raise AgentLoopError("repository file changed during snapshot")
    return name, payload_size, digest.digest()


def _bytes_snapshot_record(
    name: str, payload: bytes, *, remaining: list[int]
) -> tuple[str, int, bytes]:
    required = len(name.encode("utf-8", errors="surrogateescape")) + len(payload) + 16
    if required > remaining[0]:
        raise AgentLoopError("git snapshot byte limit exceeded")
    remaining[0] -= required
    return name, len(payload), hashlib.sha256(payload).digest()


def capture_tracked_tree_snapshot(
    root: Path,
    *,
    argv: Sequence[str] = (),
    max_files: int = MAX_SNAPSHOT_FILES,
    max_bytes: int = MAX_SNAPSHOT_BYTES,
    timeout_seconds: float = MAX_SNAPSHOT_SECONDS,
    cancel: Event | None = None,
) -> TrackedTreeSnapshot:
    """Capture Git/index/tracked-content/unignored-untracked state.

    Errors and limit exhaustion return ``complete=False`` rather than a clean
    snapshot.  No cleanup or checkout mutation is attempted.
    """
    started = time.monotonic()
    try:
        requested_root = root.expanduser()
        canonical_root = requested_root.resolve(strict=True)
        if not canonical_root.is_dir():
            raise AgentLoopError("snapshot root is not a directory")
        from . import checkout_verification as _cv

        registration = _cv.lookup_worktree_links(
            Path(os.path.abspath(requested_root)), canonical_root
        )
        link_names: set[str] = set()
        private_names: set[str] = set()
        if registration is not None:
            for link_b, target_b in sorted(registration.links.items()):
                if not _cv.link_intact(canonical_root, link_b, target_b):
                    raise AgentLoopError("registered worktree link was removed or retargeted")
                link_names.add(os.fsdecode(link_b))
            # Validate every private environment before exempting it (#1652).
            for name_b, (_env_digest, identity) in sorted(registration.private.items()):
                if not _cv.private_path_intact(canonical_root, name_b, identity):
                    raise AgentLoopError("registered private environment was removed or replaced")
                private_names.add(os.fsdecode(name_b))
        top = _run_git(canonical_root, ("rev-parse", "--show-toplevel"), timeout=timeout_seconds, cancel=cancel)
        git_root = Path(top.decode("utf-8", errors="strict").strip()).resolve(strict=True)
        if git_root != canonical_root:
            raise AgentLoopError("snapshot root does not equal Git top-level")
        head = _run_git(canonical_root, ("rev-parse", "HEAD"), timeout=timeout_seconds, cancel=cancel).decode().strip()
        index_raw = _run_git(canonical_root, ("ls-files", "-s", "-z"), timeout=timeout_seconds, cancel=cancel)
        index_entries = _git_index_entries(index_raw)
        tracked = _git_nul_paths(_run_git(canonical_root, ("ls-files", "-z"), timeout=timeout_seconds, cancel=cancel))
        if set(index_entries) != set(tracked):
            raise AgentLoopError("git snapshot index and tracked paths disagree")
        untracked = _git_nul_paths(
            _run_git(canonical_root, ("ls-files", "--others", "--exclude-standard", "-z"), timeout=timeout_seconds, cancel=cancel)
        )
        if private_names:
            for name in private_names:
                if any(path == name or path.startswith(f"{name}/") for path in tracked) or _run_git(
                    canonical_root, ("ls-tree", "-z", "--full-tree", "--name-only", "HEAD", "--", name),
                    timeout=timeout_seconds, cancel=cancel,
                ):
                    raise AgentLoopError("registered private environment is tracked or staged")
        if link_names or private_names:
            untracked = [
                name for name in untracked
                if name not in link_names
                and not any(name == p or name.startswith(f"{p}/") for p in private_names)
            ]
        if len(tracked) + len(untracked) > max_files:
            raise AgentLoopError("git snapshot file limit exceeded")
        remaining = [max_bytes]
        index_records = [
            _bytes_snapshot_record("HEAD", head.encode(), remaining=remaining),
            _bytes_snapshot_record("INDEX", index_raw, remaining=remaining),
        ]
        if registration is not None:
            for link_b, target_b in sorted(registration.links.items()):
                index_records.append(
                    _bytes_snapshot_record(
                        "\0worktree-link\0" + os.fsdecode(link_b), target_b, remaining=remaining
                    )
                )
        tracked_records: list[tuple[str, int, bytes]] = []
        # Bind evidence to one concrete dependency environment (#1652); None
        # (flag off or unregistered) appends nothing, keeping digests unchanged.
        environment = _cv.registration_environment_record(registration)
        if environment is not None:
            tracked_records.append(
                _bytes_snapshot_record("\0worktree-env\0", environment.encode(), remaining=remaining)
            )
        for relative in tracked:
            _snapshot_checkpoint(started, timeout_seconds, cancel)
            path = canonical_root / relative
            try:
                index_mode, object_id = index_entries[relative]
                if index_mode == "160000":
                    tracked_records.append(
                        _bytes_snapshot_record(
                            relative,
                            b"gitlink\0" + object_id.encode("ascii"),
                            remaining=remaining,
                        )
                    )
                    continue
                info = path.lstat()
                if path.is_symlink():
                    payload = b"symlink\0" + os.readlink(path).encode("utf-8", errors="surrogateescape")
                    tracked_records.append(
                        _bytes_snapshot_record(
                            relative,
                            str(info.st_mode).encode() + b"\0" + payload,
                            remaining=remaining,
                        )
                    )
                elif not path.is_file():
                    raise AgentLoopError("special tracked input is not snapshot-safe")
                else:
                    tracked_records.append(
                        _stream_snapshot_record(
                            path,
                            name=relative,
                            prefix=str(info.st_mode).encode() + b"\0file\0",
                            remaining=remaining,
                            started=started,
                            timeout_seconds=timeout_seconds,
                            cancel=cancel,
                        )
                    )
            except (OSError, UnicodeError) as exc:
                raise AgentLoopError("tracked file could not be read") from exc
        untracked_records: list[tuple[str, int, bytes]] = []
        for relative in untracked:
            _snapshot_checkpoint(started, timeout_seconds, cancel)
            path = canonical_root / relative
            try:
                if path.is_symlink():
                    payload = b"symlink\0" + os.readlink(path).encode("utf-8", errors="surrogateescape")
                    untracked_records.append(
                        _bytes_snapshot_record(relative, payload, remaining=remaining)
                    )
                elif path.is_file():
                    untracked_records.append(
                        _stream_snapshot_record(
                            path,
                            name=relative,
                            prefix=b"file\0",
                            remaining=remaining,
                            started=started,
                            timeout_seconds=timeout_seconds,
                            cancel=cancel,
                        )
                    )
                else:
                    raise AgentLoopError("special untracked input is not snapshot-safe")
            except (OSError, UnicodeError) as exc:
                raise AgentLoopError("untracked file could not be read") from exc
        tracked_status = _run_git(
            canonical_root,
            ("status", "--porcelain=v1", "-z", "--untracked-files=no"),
            timeout=timeout_seconds,
            cancel=cancel,
        )
        dirty_gitlinks = tuple(sorted(set(
            _dirty_gitlink_paths(tracked_status, index_entries)
        ) | set(_dirty_populated_gitlinks(
            canonical_root, index_entries, timeout_seconds=timeout_seconds,
            cancel=cancel, started=started,
        ))))
        # A gitlink object ID does not describe the checked-out submodule
        # worktree, so dirty gitlinks cannot be compared with an eventual tree.
        tracked_digest = (
            None if dirty_gitlinks else _hash_snapshot_summaries(tracked_records)
        )
        digest = _hash_snapshot_summaries(index_records + tracked_records + untracked_records)
        referenced = _referenced_paths(canonical_root, argv)
        referenced_untracked = tuple(sorted(referenced.intersection(untracked)))
        return TrackedTreeSnapshot(
            root=str(canonical_root),
            head=head,
            digest=digest,
            tracked_digest=tracked_digest,
            status_clean=not bool(tracked_status.strip()) and not dirty_gitlinks,
            complete=True,
            stable=True,
            untracked_paths=tuple(sorted(untracked)),
            referenced_untracked_paths=referenced_untracked,
            caveats=(
                ("dirty submodule worktree prevents eventual-tree attribution",)
                if dirty_gitlinks
                else ()
            ),
        )
    except (AgentLoopError, OSError, UnicodeError, ValueError) as exc:
        return TrackedTreeSnapshot(
            root=str(root), head=None, digest=None, tracked_digest=None,
            status_clean=None, complete=False, stable=None,
            caveats=(f"snapshot unknown: {_safe_text(exc)}",),
        )


def stable_tracked_tree_snapshot(
    root: Path,
    *,
    argv: Sequence[str] = (),
    max_files: int = MAX_SNAPSHOT_FILES,
    max_bytes: int = MAX_SNAPSHOT_BYTES,
    timeout_seconds: float = MAX_SNAPSHOT_SECONDS,
    cancel: Event | None = None,
) -> TrackedTreeSnapshot:
    before = capture_tracked_tree_snapshot(
        root, argv=argv, max_files=max_files, max_bytes=max_bytes,
        timeout_seconds=timeout_seconds, cancel=cancel,
    )
    if not before.complete:
        return before
    if cancel is not None and cancel.is_set():
        raise SnapshotCancelled()
    after = capture_tracked_tree_snapshot(
        root, argv=argv, max_files=max_files, max_bytes=max_bytes,
        timeout_seconds=timeout_seconds, cancel=cancel,
    )
    if not after.complete:
        return after
    if before.digest != after.digest or before.tracked_digest != after.tracked_digest or before.head != after.head:
        return replace(
            after,
            stable=False,
            caveats=("repository changed during snapshot",),
        )
    return replace(
        after,
        stable=True,
        caveats=tuple(dict.fromkeys((*before.caveats, *after.caveats))),
    )


def attribute_current_head(
    before: TrackedTreeSnapshot,
    after: TrackedTreeSnapshot,
    *,
    current_head: str | None,
    argv: Sequence[str] = (),
) -> TreeAttribution:
    caveats = list((*before.caveats, *after.caveats))
    if not before.complete or not after.complete or before.stable is not True or after.stable is not True:
        return TreeAttribution(state="unknown", head=after.head or before.head, stable=False, caveats=tuple(caveats + ["stable repository snapshot unavailable"]))
    if before.digest != after.digest or before.tracked_digest != after.tracked_digest:
        return TreeAttribution(state="unknown", head=after.head, pre_digest=before.digest, post_digest=after.digest, tracked_digest=after.tracked_digest, stable=False, caveats=tuple(caveats + ["repository changed while the test command ran"]))
    if before.head != after.head or (current_head and after.head != current_head):
        return TreeAttribution(state="stale", head=after.head, pre_digest=before.digest, post_digest=after.digest, tracked_digest=after.tracked_digest, stable=True, caveats=tuple(caveats + ["head changed or does not match current head"]))
    referenced = set(before.referenced_untracked_paths) | set(after.referenced_untracked_paths)
    if referenced:
        return TreeAttribution(state="untracked-input-unverified", head=after.head, pre_digest=before.digest, post_digest=after.digest, tracked_digest=after.tracked_digest, stable=True, untracked_input=True, caveats=tuple(caveats + ["referenced non-ignored untracked input is not attributable"]))
    untracked_present = bool(set(before.untracked_paths) | set(after.untracked_paths))
    if before.status_clean is not True or after.status_clean is not True:
        return TreeAttribution(state="unknown", head=after.head, pre_digest=before.digest, post_digest=after.digest, tracked_digest=after.tracked_digest, stable=True, untracked_input=untracked_present, caveats=tuple(caveats + ["tracked worktree awaits comparison with the eventual PR tree"] + (["non-ignored untracked content may affect test discovery or configuration"] if untracked_present else [])))
    if untracked_present:
        return TreeAttribution(
            state="untracked-input-unverified",
            head=after.head,
            pre_digest=before.digest,
            post_digest=after.digest,
            tracked_digest=after.tracked_digest,
            stable=True,
            untracked_input=False,
            caveats=tuple(caveats + ["non-ignored untracked content may affect test discovery or configuration"]),
        )
    return TreeAttribution(state="current-head", head=after.head, pre_digest=before.digest, post_digest=after.digest, tracked_digest=after.tracked_digest, stable=True, untracked_input=False, caveats=tuple(caveats))


def attribute_base_reproduction(
    before: TrackedTreeSnapshot,
    after: TrackedTreeSnapshot,
    *,
    base_commit: str,
    dangerous_checkout_mutation: bool = False,
    live_service_execution: bool = False,
) -> TreeAttribution:
    if dangerous_checkout_mutation or live_service_execution:
        return TreeAttribution(state="unknown", head=after.head, stable=False, caveats=("base reproduction requires no dangerous checkout or live-service execution",))
    if not before.complete or not after.complete or before.stable is not True or after.stable is not True:
        return TreeAttribution(state="unknown", head=after.head, stable=False, caveats=("complete stable base snapshot unavailable",))
    if before.head != base_commit or after.head != base_commit or before.digest != after.digest:
        return TreeAttribution(state="unknown", head=after.head, pre_digest=before.digest, post_digest=after.digest, tracked_digest=after.tracked_digest, stable=False, caveats=("base reproduction was not clean and stable",))
    if before.status_clean is not True or after.status_clean is not True or before.untracked_paths or after.untracked_paths:
        return TreeAttribution(state="unknown", head=after.head, stable=False, caveats=("base reproduction requires empty staged, unstaged, and non-ignored-untracked status",))
    return TreeAttribution(state="base-reproduction", head=base_commit, pre_digest=before.digest, post_digest=after.digest, tracked_digest=after.tracked_digest, stable=True)


# Invocation-local test broker -------------------------------------------------

BROKER_ENDPOINT_ENV = "AGENT_LOOP_TEST_BROKER_ENDPOINT"
BROKER_CAPABILITY_ENV = "AGENT_LOOP_TEST_BROKER_CAPABILITY"
BROKER_PROTOCOL_ENV = "AGENT_LOOP_TEST_BROKER_PROTOCOL"
BROKER_PROTOCOL = "local-test-broker-v1"
BROKER_FRAME_LIMIT = 128 * 1024
from .test_runtime import (
    EXECUTED_ARGV_MAX_ITEM_BYTES as BROKER_MAX_ARG_BYTES,
    EXECUTED_ARGV_MAX_ITEMS as BROKER_MAX_ARGV,
    EXECUTED_ARGV_MAX_TOTAL_BYTES as BROKER_MAX_ARGV_TOTAL,
    coerce_suite_start_reason,
)
BROKER_MAX_CWD_BYTES = 4 * 1024
BROKER_MAX_ENV = 256
BROKER_MAX_ENV_TOTAL = 64 * 1024
BROKER_MAX_ENV_NAME_BYTES = 128
BROKER_MAX_ENV_VALUE_BYTES = 8 * 1024
BROKER_REQUEST_FIELDS = frozenset(
    {"turn_id", "nonce", "argv", "timeout_seconds", "cwd", "environment"}
)
# Issue #1326: the only optional request field.  A request carries either the
# exact legacy field set or that set plus ``fixture``.
BROKER_FIXTURE_FIELD = "fixture"
FIXTURE_STATE_DIR_ENV = "AGENT_LOOP_FIXTURE_STATE_DIR"
FIXTURE_MIN_TIMEOUT_SECONDS = 30.0
BROKER_RESPONSE_FIELDS = frozenset(
    {"type", "receipt_id", "outcome", "returncode", "elapsed_seconds", "output_tail", "error"}
)


class BrokerProtocolError(AgentLoopError):
    """A malformed, unauthenticated, or bounded-out broker request."""


class BrokerNotDispatched(BrokerProtocolError):
    """Connect or send failed before the request frame was fully written."""


class BrokerResultUnknown(BrokerProtocolError):
    """The request frame was sent but no terminal response was received."""


class ExecutionReferenceRegistry:
    """Keep invocation-local execution namespaces unique for one runner.

    A selector is ephemeral, but its authority remains live for as long as a
    runner retains the corresponding turn catalog. Namespace registration is
    therefore deliberately monotonic: a later broker cannot reuse a namespace
    from an earlier turn, even after that broker has stopped and its journal
    has been copied into the runner's retained observations.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._namespaces: set[str] = set()

    def reserve_namespace(self, namespace: str) -> None:
        namespace = str(namespace)
        if not namespace or len(namespace.encode("utf-8", errors="replace")) > 256:
            raise BrokerProtocolError("execution namespace is empty or oversized")
        with self._lock:
            if namespace in self._namespaces:
                raise BrokerProtocolError(
                    "execution namespace collides with a retained test turn"
                )
            self._namespaces.add(namespace)


def _json_no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BrokerProtocolError("duplicate JSON object key")
        result[key] = value
    return result


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise BrokerProtocolError("truncated broker frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(connection: socket.socket) -> dict[str, object]:
    header = _recv_exact(connection, 4)
    (length,) = struct.unpack(">I", header)
    if length < 2 or length > BROKER_FRAME_LIMIT:
        raise BrokerProtocolError("broker frame exceeds 128 KiB limit")
    try:
        value = json.loads(
            _recv_exact(connection, length).decode("utf-8", errors="strict"),
            object_pairs_hook=_json_no_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrokerProtocolError("broker frame is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise BrokerProtocolError("broker frame must be a JSON object")
    return value


def _send_frame(connection: socket.socket, payload: Mapping[str, object]) -> None:
    raw = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > BROKER_FRAME_LIMIT:
        raise BrokerProtocolError("broker response exceeds 128 KiB limit")
    connection.sendall(struct.pack(">I", len(raw)) + raw)


BROKER_STREAM_SEND_GRACE_SECONDS = 10.0
BROKER_TERMINAL_SEND_GRACE_SECONDS = 2.0
BROKER_SEND_CANCEL_POLL_SECONDS = 0.05
# Issue #1653: request-lifetime client liveness.  A client announces that it
# acknowledges heartbeats through this reserved environment entry (popped by
# the broker like the tool-identity entries, so the request schema is
# unchanged).  A client with no acknowledged progress for the timeout is lost.
BROKER_CLIENT_ACKS_ENV = "AGENT_LOOP_BROKER_CLIENT_ACKS"
BROKER_CLIENT_ACK_TIMEOUT_SECONDS = 30.0
BROKER_ACK_FRAME_LIMIT = 256
BROKER_MONITOR_POLL_SECONDS = 0.1
BROKER_MONITOR_JOIN_SECONDS = 30.0
BROKER_CLIENT_LOST_CAVEAT = "run-tests client disconnected; target stopped by the broker"
BROKER_NO_TEST_OUTCOMES = frozenset(
    {"overlap-rejected", "worker-budget-busy", "worker-budget-refused", "cancelled"}
)
BROKER_REFUSAL_DIAGNOSTIC_BYTES = 2048


BROKER_CLIENT_LOST_DIAGNOSTIC = "test command cancelled before launch (its run-tests client was lost)"


def _cancelled_frame(diagnostic: str = "test command cancelled before launch (the broker is stopping)") -> dict[str, object]:
    return {
        "type": "result", "receipt_id": "", "execution_ref": None, "outcome": "cancelled",
        "returncode": None, "elapsed_seconds": 0.0, "output_tail": "",
        "suite_start": "not-started",
        "diagnostic": diagnostic,
    }


def _legacy_client_reason(environment: Mapping[str, object]) -> str:
    """The visible no-test reason for a client without ack support (#1653)."""
    pinned = environment.get("AGENT_LOOP_EXECUTABLE")
    named = (
        _safe_text(pinned, 512) if isinstance(pinned, str) and pinned.strip()
        else "$AGENT_LOOP_EXECUTABLE"
    )
    return (
        "agent-loop: this run-tests client predates broker liveness acknowledgements; nothing "
        f"was run. Re-run with this run's pinned agent-loop executable {named}."
    )


def _legacy_refusal_frame(reason: str) -> dict[str, object]:
    """A terminal no-test frame an older client already treats as not run."""
    return {
        **_cancelled_frame(),
        "returncode": 125,
        "inner_exec": "not-attempted",
        "diagnostic": _safe_text(reason, BROKER_REFUSAL_DIAGNOSTIC_BYTES),
    }


class _AckReader:
    """Bounded, non-blocking reader of client heartbeat acknowledgements (#1653).

    Only ``{"type": "ack"}`` frames may follow a request.  A partial frame is
    buffered across polls; EOF, a reset or any other frame means the client
    is lost.
    """

    def __init__(self, connection: socket.socket) -> None:
        self._connection = connection
        self._buffer = b""
        self.reason = ""

    def _lost(self, reason: str) -> str:
        self.reason = reason
        return "lost"

    def poll(self, timeout: float, *, max_reads: int = 16) -> str | None:
        """Return ``"ack"``, ``"lost"`` or ``None`` (nothing new)."""
        acked = False
        for attempt in range(max_reads):
            try:
                ready, _, _ = select.select(
                    [self._connection], [], [], max(0.0, timeout) if attempt == 0 else 0.0,
                )
            except (OSError, ValueError):
                return self._lost("the client connection closed")
            if not ready:
                break
            try:
                chunk = self._connection.recv(4096, socket.MSG_DONTWAIT)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                return self._lost("the client connection was reset")
            if not chunk:
                return self._lost("the client closed its connection")
            self._buffer += chunk
            while len(self._buffer) >= 4:
                (length,) = struct.unpack(">I", self._buffer[:4])
                if length < 2 or length > BROKER_ACK_FRAME_LIMIT:
                    return self._lost("the client sent an unexpected frame")
                if len(self._buffer) < 4 + length:
                    break
                raw, self._buffer = self._buffer[4:4 + length], self._buffer[4 + length:]
                try:
                    value = json.loads(raw.decode("utf-8", errors="strict"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return self._lost("the client sent an unexpected frame")
                if value != {"type": "ack"}:
                    return self._lost("the client sent an unexpected frame")
                acked = True
        return "ack" if acked else None


def _drain_acks(connection: socket.socket, seconds: float) -> None:
    """Discard pending ack bytes before close so the queued result is not reset."""
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        remaining = deadline - time.monotonic()
        try:
            ready, _, _ = select.select([connection], [], [], max(0.0, remaining))
            if not ready:
                return
            if not connection.recv(4096, socket.MSG_DONTWAIT):
                return
        except (BlockingIOError, InterruptedError):
            pass
        except (OSError, ValueError):
            return
        if remaining <= 0:
            return


class _RequestCancel(Event):
    """One owned request's cancel token (issue #1653).

    A real ``Event`` (``is_set`` and a bounded ``wait``), also set while the
    broker's own stop event is set, so every existing stop check holds.
    """

    def __init__(self, stop: Event) -> None:
        super().__init__()
        self._broker_stop = stop

    def is_set(self) -> bool:
        return super().is_set() or self._broker_stop.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            if self.is_set():
                return True
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return self.is_set()
            slice_seconds = BROKER_SEND_CANCEL_POLL_SECONDS if remaining is None else min(
                remaining, BROKER_SEND_CANCEL_POLL_SECONDS
            )
            if super().wait(slice_seconds):
                return True


class _BrokerRequest:
    """Liveness, cancellation and lock-holder state of one owned request (#1653)."""

    def __init__(self, stop: Event, *, command: str, normalized: str) -> None:
        self.cancel = _RequestCancel(stop)
        self.done = Event()
        self.command = command
        self.normalized = normalized
        self.phase = "preparing"
        self.started = time.monotonic()
        self.last_progress = self.started
        self.client_lost = False
        self.loss_reason = ""
        self.cancel_started = False
        self.completing = False
        self.processes: dict[int, Any] = {}
        self.fixtures: dict[int, tuple[Any, float]] = {}
        # The locks this request really holds (kind -> exact key, if any),
        # reported by the runner at acquisition and release.
        self.holds: dict[str, str | None] = {}
        self.cwd_label = "."


def _host_heartbeat_grace() -> float:
    from .test_workers import HOST_WAIT_HEARTBEAT_SEND_GRACE_SECONDS

    return HOST_WAIT_HEARTBEAT_SEND_GRACE_SECONDS


class _ConnectionSender:
    """Bounded, cancellation-aware writer for one broker connection (#1108).

    Every send is non-blocking and gives up at its grace.  A frame that cannot
    be written completely (zero or some bytes) wedges the connection: it is
    shut down and every later send returns at once, so no byte ever follows a
    truncated frame and a client that stopped reading can only stall its own
    handler, never the admission bound or broker shutdown.
    """

    def __init__(self, connection: socket.socket, cancel: Event) -> None:
        self._connection = connection
        self._cancel = cancel
        self._lock = Lock()
        self.wedged = False

    def bind_cancel(self, cancel: Event) -> None:
        """Use an owned request's cancel token (#1653)."""
        self._cancel = cancel

    def _wedge(self) -> None:
        self.wedged = True
        try:
            self._connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def send(
        self,
        payload: Mapping[str, object],
        *,
        grace: float,
        cancellable: bool = True,
        skip_if_busy: bool = False,
        lock_wait: float = 0.0,
    ) -> bool:
        """Send one frame; return whether it was delivered or deliberately skipped-safe.

        ``skip_if_busy`` (heartbeats) skips the beat, without wedging, when the
        lock is busy or nothing at all could be written.  ``lock_wait`` lets a
        skippable beat wait that long for the lock first (#1653), so sustained
        streaming cannot starve liveness probes.
        """
        raw = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(raw) > BROKER_FRAME_LIMIT:
            raise BrokerProtocolError("broker response exceeds 128 KiB limit")
        frame = struct.pack(">I", len(raw)) + raw
        if self.wedged:
            return False
        # One deadline covers lock acquisition and writing.
        deadline = time.monotonic() + max(0.0, grace)
        if skip_if_busy:
            acquired = (
                self._lock.acquire(timeout=lock_wait) if lock_wait > 0
                else self._lock.acquire(blocking=False)
            )
            if not acquired:
                return False
        else:
            while not self._lock.acquire(
                timeout=max(0.0, min(BROKER_SEND_CANCEL_POLL_SECONDS, deadline - time.monotonic()))
            ):
                if time.monotonic() >= deadline or (cancellable and self._cancel.is_set()):
                    # Another send holds the lock mid-frame: fail this send
                    # and wedge, so no later frame can interleave.
                    self._wedge()
                    return False
        try:
            if self.wedged:
                return False
            view = memoryview(frame)
            sent = 0
            while sent < len(frame):
                try:
                    count = self._connection.send(view[sent:], socket.MSG_DONTWAIT)
                except (BlockingIOError, InterruptedError):
                    if sent == 0 and skip_if_busy:
                        return False
                    now = time.monotonic()
                    if now >= deadline or (cancellable and self._cancel.is_set()):
                        self._wedge()
                        return False
                    try:
                        select.select(
                            [], [self._connection],
                            [], min(BROKER_SEND_CANCEL_POLL_SECONDS, deadline - now),
                        )
                    except (OSError, ValueError):
                        self._wedge()
                        return False
                    continue
                except OSError:
                    self._wedge()
                    return False
                sent += count
            return True
        finally:
            self._lock.release()


def _validate_broker_request(
    request: Mapping[str, object], *, turn_id: str, capability: str, root: Path, ceiling: float
) -> dict[str, object]:
    fields = set(request)
    if fields != BROKER_REQUEST_FIELDS and fields != BROKER_REQUEST_FIELDS | {BROKER_FIXTURE_FIELD}:
        raise BrokerProtocolError("broker request has unknown or missing fields")
    fixture: str | None = None
    if BROKER_FIXTURE_FIELD in request:
        raw_fixture = request[BROKER_FIXTURE_FIELD]
        if (
            not isinstance(raw_fixture, str) or not raw_fixture or "\x00" in raw_fixture
            or len(raw_fixture.encode("utf-8")) > BROKER_MAX_CWD_BYTES
        ):
            raise BrokerProtocolError("broker fixture must be a bounded non-empty path")
        fixture = raw_fixture
    supplied_turn = request.get("turn_id")
    if not isinstance(supplied_turn, str) or supplied_turn != turn_id or "\x00" in supplied_turn or not (1 <= len(supplied_turn) <= 128):
        raise BrokerProtocolError("broker request is bound to a different turn")
    nonce = request.get("nonce")
    if not isinstance(nonce, str) or not (16 <= len(nonce) <= 256) or "\x00" in nonce:
        raise BrokerProtocolError("broker nonce is invalid")
    nonce_parts = nonce.split(".", 1)
    if len(nonce_parts) != 2 or not hmac.compare_digest(
        nonce_parts[1],
        hmac.new(capability.encode("ascii"), nonce_parts[0].encode("ascii"), hashlib.sha256).hexdigest(),
    ):
        raise BrokerProtocolError("broker nonce authentication failed")
    argv = request.get("argv")
    if not isinstance(argv, list) or not argv or len(argv) > BROKER_MAX_ARGV:
        raise BrokerProtocolError("broker argv exceeds its bound")
    argv_values: list[str] = []
    total = 0
    for item in argv:
        if not isinstance(item, str) or "\x00" in item:
            raise BrokerProtocolError("broker argv contains invalid text")
        size = len(item.encode("utf-8"))
        if size > BROKER_MAX_ARG_BYTES:
            raise BrokerProtocolError("broker argv entry exceeds its bound")
        total += size
        argv_values.append(item)
    if total > BROKER_MAX_ARGV_TOTAL:
        raise BrokerProtocolError("broker argv exceeds its total bound")
    timeout = request.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or float(timeout) <= 0 or float(timeout) > ceiling:
        raise BrokerProtocolError("broker timeout is outside the configured ceiling")
    if fixture is not None and float(timeout) < FIXTURE_MIN_TIMEOUT_SECONDS:
        raise BrokerProtocolError(
            f"fixture refused: fixture-timeout-below-floor (a --fixture run needs at least "
            f"{FIXTURE_MIN_TIMEOUT_SECONDS:g}s)"
        )
    cwd = request.get("cwd")
    if not isinstance(cwd, str) or "\x00" in cwd or len(cwd.encode("utf-8")) > BROKER_MAX_CWD_BYTES or not os.path.isabs(cwd):
        raise BrokerProtocolError("broker cwd must be an absolute bounded path")
    environment = request.get("environment")
    if not isinstance(environment, dict) or len(environment) > BROKER_MAX_ENV:
        raise BrokerProtocolError("broker environment exceeds its bound")
    total_environment = 0
    for name, value in environment.items():
        if not isinstance(name, str) or not isinstance(value, str) or "\x00" in name or "\x00" in value:
            raise BrokerProtocolError("broker environment contains invalid text")
        if len(name.encode("utf-8")) > BROKER_MAX_ENV_NAME_BYTES or len(value.encode("utf-8")) > BROKER_MAX_ENV_VALUE_BYTES:
            raise BrokerProtocolError("broker environment entry exceeds its bound")
        total_environment += len(name.encode("utf-8")) + len(value.encode("utf-8"))
    if total_environment > BROKER_MAX_ENV_TOTAL:
        raise BrokerProtocolError("broker environment exceeds its total bound")
    requested = Path(cwd)
    try:
        canonical = requested.resolve(strict=True)
        canonical_root = root.resolve(strict=True)
        if not canonical.is_dir() or os.path.commonpath((str(canonical_root), str(canonical))) != str(canonical_root):
            raise BrokerProtocolError("broker cwd is outside the assigned checkout")
        if requested.is_symlink():
            raise BrokerProtocolError("broker cwd may not be a final symlink")
    except (OSError, ValueError) as exc:
        raise BrokerProtocolError("broker cwd could not be validated") from exc
    validated: dict[str, object] = {
        "turn_id": turn_id,
        "nonce": nonce,
        "argv": tuple(argv_values),
        "timeout_seconds": float(timeout),
        "cwd": canonical,
        "environment": dict(environment),
    }
    if fixture is not None:
        validated[BROKER_FIXTURE_FIELD] = fixture
    return validated


@dataclass(frozen=True)
class BrokerRunResult:
    receipt_id: str
    execution_ref: str | None
    outcome: str
    returncode: int | None
    elapsed_seconds: float
    output_tail: str = ""
    error: str | None = None
    wrapper_bootstrap: str = "unknown"
    inner_exec: str = "not-attempted"
    suite_start: str = "not-started"
    diagnostic: str = ""
    executed_argv: tuple[str, ...] = ()
    # Issue #1661: diagnostic reason beside ``suite_start=unknown``; ``None``
    # from an older broker or whenever the suite start is not unknown.
    suite_start_reason: str | None = None
    workers_cohort: str | None = None
    worker_enforcement: str | None = None
    worker_caveats: tuple[str, ...] = ()
    worker_environment: Mapping[str, str] = field(default_factory=dict)
    # Issue #1326: the projected fixture record, decoded defensively; ``None``
    # for fixture-less responses.
    fixture: Mapping[str, object] | None = None
    # Issue #1651: parent-selected advisory remedy text for a launch failure.
    # Presentation only; never journaled, never part of the diagnostic.
    advisory: str = ""


def _bounded_argv(values: Sequence[object]) -> list[str]:
    """Apply the request argv caps to a response argv."""
    bounded: list[str] = []
    total = 0
    for item in list(values)[:BROKER_MAX_ARGV]:
        text = str(item)
        size = len(text.encode("utf-8", errors="replace"))
        if size > BROKER_MAX_ARG_BYTES or total + size > BROKER_MAX_ARGV_TOTAL:
            break
        total += size
        bounded.append(text)
    return bounded


@dataclass
class _ReplayReservation:
    digest: str
    ready: Event = field(default_factory=Event)
    response: dict[str, object] | None = None


class _FixtureRun:
    """Broker-side state of one ``run-tests --fixture`` request (issue #1326).

    The absolute request deadline is fixed at acceptance (after the before
    snapshot): the runner gets ``inner_deadline``, and teardown plus the
    sweep keep the reserve after it.
    """

    def __init__(self, *, provenance: Any, mount_id: str, timeout_seconds: float) -> None:
        from .fixture_runtime import reserve_seconds

        self.provenance = provenance
        self.mount_id = mount_id
        self.timeout_seconds = float(timeout_seconds)
        self.reserve = reserve_seconds(timeout_seconds)
        self.request_deadline = time.monotonic() + self.timeout_seconds
        self.inner_deadline = self.request_deadline - self.reserve
        self.supervisor: Any | None = None
        self.decision: Any | None = None
        self.state_dir = Path(tempfile.mkdtemp(prefix="agent-loop-fixture-state-"))
        os.chmod(self.state_dir, 0o700)
        self._finished = False
        self.request: _BrokerRequest | None = None

    def close(self) -> None:
        import shutil

        shutil.rmtree(self.state_dir, ignore_errors=True)

    def finish(self, server: "TestBrokerServer") -> None:
        if self._finished:
            return
        self._finished = True
        try:
            if self.supervisor is not None:
                # Bounded by the original request deadline, not a fresh reserve.
                self.supervisor.finish()
                with server._journal_lock:
                    server._active_fixtures.pop(id(self.supervisor), None)
                    if self.request is not None:
                        self.request.fixtures.pop(id(self.supervisor), None)
        finally:
            self.close()

    def fold(
        self, outcome: str, *, inner_exec: str, shutdown: bool, diagnostic: str,
    ) -> tuple[FixtureRecord, str, str, tuple[str, ...]]:
        """Return (record, private excerpts, folded outcome, caveats)."""
        from .fixture_runtime import LAUNCHER_CHANGED_CAVEAT, lifecycle_caveats, not_run_record

        if self.supervisor is not None:
            record, excerpts = self.supervisor.record()
        else:
            record, excerpts = not_run_record(self.provenance), ""
        caveats: list[str] = []
        if diagnostic == LAUNCHER_CHANGED_CAVEAT:
            caveats.append(LAUNCHER_CHANGED_CAVEAT)
        action = getattr(self.decision, "action", None)
        if self.supervisor is not None and (shutdown or action == "cancel"):
            # A stop after the supervisor launched is never ``passed``.
            outcome = "interrupted"
        elif action == "timed_out":
            outcome = "timed_out"
        elif action == "skip":
            outcome = "failed"
        elif outcome == "passed" and not record.clean:
            outcome = "failed"
        if self.supervisor is not None or action is not None:
            caveats.extend(lifecycle_caveats(record))
        return record, excerpts, outcome, tuple(dict.fromkeys(caveats))


class TestBrokerServer:
    """A single-turn AF_UNIX broker owned by the parent orchestrator."""

    def __init__(
        self,
        *,
        root: Path,
        turn_id: str | None = None,
        execution_namespace: str | None = None,
        execution_reference_registry: ExecutionReferenceRegistry | None = None,
        timeout_ceiling: float = 1800,
        containment_policy: object | None = None,
        execute: Any | None = None,
        environment_registry: EnvironmentIdentityRegistry | None = None,
        telemetry_attribution: dict | None = None,
        dependency_remedy: str | None = None,
    ) -> None:
        if root.is_symlink():
            raise BrokerProtocolError("broker root may not be a symlink")
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise BrokerProtocolError("broker root is not a directory")
        self.turn_id = turn_id or uuid.uuid4().hex
        # Issue #1651: chosen by the parent from its own config when it starts
        # the broker; attached to launch-failure frames as presentation only.
        self.dependency_remedy = dependency_remedy or None
        self.capability = uuid.uuid4().hex + uuid.uuid4().hex
        self.timeout_ceiling = float(timeout_ceiling)
        self.containment_policy = containment_policy
        self._execute = execute
        self._runtime_dir: Path | None = None
        self._socket: socket.socket | None = None
        self._thread: Thread | None = None
        self._handler_threads: set[Thread] = set()
        self._stop = False
        self._send_lock = Lock()
        # Issue #1108: pending admission is cancelled by ``_stop_event``; the
        # launch lock synchronises stop() with target launch + registration.
        self._stop_event = Event()
        self._launch_lock = Lock()
        self._deferred_close = False
        self._resource_lock = Lock()
        self._journal: list[LocalTestObservation] = []
        # Reservations are retained for the broker lifetime. New work fails
        # closed at the bound instead of making an old authenticated nonce
        # executable again.
        self._receipts: dict[str, _ReplayReservation] = {}
        self._journal_lock = Lock()
        self._environment_registry = environment_registry or EnvironmentIdentityRegistry()
        self._execution_namespace = execution_namespace or (uuid.uuid4().hex + uuid.uuid4().hex)
        self._execution_reference_registry = (
            execution_reference_registry or ExecutionReferenceRegistry()
        )
        self._execution_reference_registry.reserve_namespace(self._execution_namespace)
        self._next_execution_ordinal = 0
        self._execution_refs: set[str] = set()
        self._parent_containment_handle: Any | None = None
        self._process_started: Any | None = None
        self._process_finished: Any | None = None
        self._active_processes: dict[int, Any] = {}
        # Issue #1326: fixture supervisors are never sent the ordinary
        # SIGTERM/SIGKILL; stop() hands them a graceful teardown first.
        self._active_fixtures: dict[int, tuple[Any, float]] = {}
        # Issue #1653: owned live requests, each with its own cancel token.
        self._requests: dict[int, _BrokerRequest] = {}
        self._pinned_root: Any | None = None
        self._worker_budget: Any | None = None
        self._worker_lock_root: Path | None = None
        # Runner-supplied at turn start; never read from a coder request (#1107).
        self._telemetry_attribution = dict(telemetry_attribution) if telemetry_attribution else None
        self._tool_mismatch_logged: set[tuple[str | None, str | None]] = set()

    def set_execution_context(
        self,
        *,
        containment_handle: Any | None,
        process_started: Any,
        process_finished: Any,
        worker_budget: Any | None = None,
    ) -> None:
        """Bind broker children to the live requesting turn before requests run.

        ``worker_budget`` is the parent-owned, post-admission budget.  It is
        authoritative: a client may only lower it or tighten its mode.
        """
        self._parent_containment_handle = containment_handle
        self._process_started = process_started
        self._process_finished = process_finished
        self._worker_budget = worker_budget

    def effective_worker_budget(self, environment: Mapping[str, str]) -> Any | None:
        """Stricter of the parent budget and the client's requested values."""
        parent = self._worker_budget
        if parent is None:
            return None
        from dataclasses import replace as _replace

        from .test_workers import (
            ENV_TEST_WORKER_ENFORCEMENT,
            ENV_TEST_WORKERS,
            WorkerBudgetError,
            parse_enforcement,
            parse_worker_count,
            stricter_mode,
        )

        workers = parent.workers
        mode = parent.enforcement
        try:
            requested = parse_worker_count(environment.get(ENV_TEST_WORKERS, ""), name=ENV_TEST_WORKERS)
        except WorkerBudgetError:
            requested = None
        if requested is not None and requested < workers:
            workers = requested
        try:
            requested_mode = parse_enforcement(
                environment.get(ENV_TEST_WORKER_ENFORCEMENT), name=ENV_TEST_WORKER_ENFORCEMENT
            )
        except WorkerBudgetError:
            requested_mode = None
        if requested_mode is not None:
            mode = stricter_mode(mode, requested_mode)
        return _replace(parent, workers=workers, enforcement=mode, source="inherited")

    @property
    def endpoint(self) -> str:
        if self._runtime_dir is None:
            raise BrokerProtocolError("test broker is not started")
        return str(self._runtime_dir / "broker.sock")

    @property
    def environment(self) -> dict[str, str]:
        return {
            BROKER_ENDPOINT_ENV: self.endpoint,
            BROKER_CAPABILITY_ENV: self.capability,
            BROKER_PROTOCOL_ENV: BROKER_PROTOCOL,
        }

    @property
    def journal(self) -> tuple[LocalTestObservation, ...]:
        with self._journal_lock:
            return tuple(self._journal)

    def start(self) -> "TestBrokerServer":
        if self._socket is not None:
            return self
        from .containment import PinnedCheckoutRoot

        self._pinned_root = PinnedCheckoutRoot.open(self.root)
        self._runtime_dir = Path(tempfile.mkdtemp(prefix="agent-loop-test-broker-"))
        os.chmod(self._runtime_dir, 0o700)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(self.endpoint)
            os.chmod(self.endpoint, 0o600)
            server.listen(8)
            server.settimeout(0.2)
        except BaseException:
            server.close()
            self.stop()
            raise
        self._socket = server
        self._thread = Thread(target=self._serve, name="agent-loop-test-broker", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop = True
        server, self._socket = self._socket, None
        if server is not None:
            server.close()
        # Either the runner's guarded launch completes first and its process is
        # in the snapshot below, or it sees the event and launches nothing.
        with self._launch_lock:
            with self._journal_lock:
                # Shared with evidence publication: a journal append either
                # precedes this point or observes the stop and is unattributed.
                self._stop_event.set()
                for request in self._requests.values():
                    request.cancel.set()
                active = list(self._active_processes.values())
                fixtures = list(self._active_fixtures.values())
        from .runner import terminate_process_group_fully

        for proc in active:
            # The whole group, including a descendant that ignored SIGTERM
            # after its leader exited (#1653); idempotent with a client-loss
            # cancel of the same request.
            terminate_process_group_fully(proc.pid, proc=proc)
        # Teardown interrupts a running setup; each supervisor then gets its
        # reserve before it (and every inventoried PID) is forcibly killed.
        for supervisor, _reserve in fixtures:
            supervisor.request_teardown()
        for supervisor, reserve in fixtures:
            supervisor.finish(budget=reserve)
        if self._thread is not None and self._thread is not current_thread():
            self._thread.join(timeout=2)
        self._thread = None
        with self._journal_lock:
            handlers = list(self._handler_threads)
        for handler in handlers:
            if handler is not current_thread():
                handler.join(timeout=5)
        with self._journal_lock:
            still_running = any(
                handler.is_alive() and handler is not current_thread()
                for handler in self._handler_threads
            )
            if still_running:
                # A handler (e.g. in one stalled file read) must never run
                # against closed resources; the last one to exit closes them.
                self._deferred_close = True
        if not still_running:
            self._close_resources()

    def _close_resources(self) -> None:
        with self._resource_lock:
            runtime_dir, self._runtime_dir = self._runtime_dir, None
            pinned_root, self._pinned_root = self._pinned_root, None
        if runtime_dir is not None:
            try:
                for item in runtime_dir.iterdir():
                    item.unlink(missing_ok=True)
                runtime_dir.rmdir()
            except OSError:
                pass
        if pinned_root is not None:
            pinned_root.close()

    def snapshot_journal(self) -> tuple[LocalTestObservation, ...]:
        return self.journal

    def live_execution_catalog(self) -> tuple[dict[str, object], ...]:
        """Return selectors for the current turn without making them durable."""
        with self._journal_lock:
            return tuple(item.live_projection() for item in self._journal)

    def _serve(self) -> None:
        server = self._socket
        if server is None:
            return
        while not self._stop:
            try:
                connection, _address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            handler = Thread(target=self._handle, args=(connection,), daemon=True)
            with self._journal_lock:
                self._handler_threads.add(handler)
            handler.start()

    def _handle(self, connection: socket.socket) -> None:
        validated: Mapping[str, object] | None = None
        sender = _ConnectionSender(connection, self._stop_event)
        terminal = BROKER_TERMINAL_SEND_GRACE_SECONDS
        try:
            with connection:
                try:
                    request = _recv_frame(connection)
                    validated = _validate_broker_request(
                        request,
                        turn_id=self.turn_id,
                        capability=self.capability,
                        root=self.root,
                        ceiling=self.timeout_ceiling,
                    )
                    if dict(validated["environment"]).get(BROKER_CLIENT_ACKS_ENV) != "1":  # type: ignore[call-overload]
                        # Issue #1653: a client that cannot acknowledge
                        # heartbeats cannot be proven live, so nothing is
                        # taken or launched.  The reason is streamed (an older
                        # client prints output frames) and the terminal frame
                        # is a no-test result, never an error frame that would
                        # trigger the client's local fallback.
                        reason = _legacy_client_reason(validated["environment"])  # type: ignore[arg-type]
                        sender.send({"type": "output", "data": reason + "\n"}, grace=terminal, cancellable=False)
                        sender.send(_legacy_refusal_frame(reason), grace=terminal, cancellable=False)
                        _drain_acks(connection, BROKER_SEND_CANCEL_POLL_SECONDS)
                        return
                    nonce = str(validated["nonce"])
                    digest = hashlib.sha256(json.dumps(validated, sort_keys=True, default=str).encode()).hexdigest()
                    with self._journal_lock:
                        reservation = self._receipts.get(nonce)
                        owns_reservation = reservation is None
                        if reservation is None:
                            if len(self._receipts) >= MAX_PRIVATE_OBSERVATIONS:
                                raise BrokerProtocolError("broker replay capacity is exhausted")
                            reservation = _ReplayReservation(digest=digest)
                            self._receipts[nonce] = reservation
                        elif reservation.digest != digest:
                            raise BrokerProtocolError("conflicting replay for nonce")
                    if not owns_reservation:
                        from .test_workers import HOST_WAIT_HEARTBEAT_SECONDS

                        last_frame = time.monotonic()
                        acks = _AckReader(connection)
                        while not reservation.ready.wait(BROKER_SEND_CANCEL_POLL_SECONDS):
                            # Issue #1653: a follower consumes its own acks so
                            # they never back up; losing it ends only this
                            # follower, never the owner's reservation.
                            if acks.poll(0.0) == "lost":
                                return
                            if self._stop_event.is_set():
                                # Cancel this follower alone: the owner's
                                # reservation is left untouched.
                                sender.send(_cancelled_frame(), grace=terminal, cancellable=False)
                                return
                            if time.monotonic() - last_frame >= HOST_WAIT_HEARTBEAT_SECONDS:
                                last_frame = time.monotonic()
                                sender.send(
                                    {"type": "output", "data": ""},
                                    grace=_host_heartbeat_grace(), skip_if_busy=True,
                                )
                        if reservation.response is None:
                            raise BrokerProtocolError("broker replay did not complete")
                        sender.send(reservation.response, grace=terminal, cancellable=False)
                        _drain_acks(connection, BROKER_SEND_CANCEL_POLL_SECONDS)
                        return
                    state = self._register_request(validated)
                    sender.bind_cancel(state.cancel)
                    monitor = Thread(
                        target=self._monitor_client, args=(connection, sender, state),
                        name="agent-loop-test-broker-client-monitor", daemon=True,
                    )
                    monitor.start()
                    try:
                        response = self._execute_request(validated, connection, sender, state)
                    except (BrokerProtocolError, AgentLoopError, OSError, ValueError) as exc:
                        if isinstance(exc, AgentLoopError):
                            self._record_capture_failure(validated, exc)
                        response = {"type": "error", "error": _safe_text(exc)}
                    except Exception as exc:
                        self._record_capture_failure(validated, exc)
                        response = {
                            "type": "error",
                            "error": f"broker execution failed: {type(exc).__name__}",
                        }
                    finally:
                        # The monitor stops before the terminal send, so a
                        # heartbeat never follows the result frame.
                        state.done.set()
                        monitor.join(BROKER_MONITOR_JOIN_SECONDS)
                        with self._journal_lock:
                            self._requests.pop(id(state), None)
                            reservation.response = response
                            reservation.ready.set()
                    sender.send(response, grace=terminal, cancellable=False)
                    _drain_acks(connection, BROKER_SEND_CANCEL_POLL_SECONDS)
                except (BrokerProtocolError, AgentLoopError, OSError, ValueError) as exc:
                    if (
                        isinstance(exc, AgentLoopError)
                        and not isinstance(exc, BrokerProtocolError)
                        and validated is not None
                    ):
                        self._record_capture_failure(validated, exc)
                    try:
                        sender.send(
                            {"type": "error", "error": _safe_text(exc)},
                            grace=terminal, cancellable=False,
                        )
                    except (OSError, BrokerProtocolError):
                        pass
        finally:
            with self._journal_lock:
                self._handler_threads.discard(current_thread())
                close_now = self._deferred_close and not self._handler_threads
            if close_now:
                self._close_resources()

    def _register_request(self, validated: Mapping[str, object]) -> _BrokerRequest:
        """Register an owned request; a stop already in progress cancels it."""
        argv = tuple(str(item) for item in validated.get("argv", ()))  # type: ignore[union-attr]
        try:
            command = redact_test_command(argv)[0]
        except Exception:  # noqa: BLE001 - a diagnostic label only
            command = "<unprintable command>"
        try:
            from .test_runtime import normalize_test_command

            normalized = normalize_test_command(argv, cwd=Path(str(validated.get("cwd"))))
        except Exception:  # noqa: BLE001
            normalized = ""
        state = _BrokerRequest(self._stop_event, command=command, normalized=normalized)
        try:
            relative = os.path.relpath(str(validated.get("cwd")), str(self.root))
            state.cwd_label = "." if relative == "." else relative
        except (TypeError, ValueError):
            state.cwd_label = "."
        with self._journal_lock:
            self._requests[id(state)] = state
            if self._stop_event.is_set():
                state.cancel.set()
        return state

    def _monitor_client(
        self, connection: socket.socket, sender: _ConnectionSender, state: _BrokerRequest,
    ) -> None:
        """Watch one owned request's client for its whole lifetime (#1653).

        The sole reader of the connection after the request frame.  EOF, a
        reset, an unexpected frame, a wedged send, or no acknowledged progress
        for ``BROKER_CLIENT_ACK_TIMEOUT_SECONDS`` since the request started or
        the last ack (whether or not a heartbeat could be delivered) is loss.
        """
        from .test_workers import HOST_WAIT_HEARTBEAT_SECONDS

        reader = _AckReader(connection)
        last_probe = time.monotonic()
        while not state.done.is_set():
            reason = ""
            if sender.wedged:
                reason = "the client stopped reading its output"
            elif time.monotonic() - state.last_progress > BROKER_CLIENT_ACK_TIMEOUT_SECONDS:
                reason = (
                    f"no heartbeat acknowledgement for {BROKER_CLIENT_ACK_TIMEOUT_SECONDS:g} s"
                )
            else:
                status = reader.poll(BROKER_MONITOR_POLL_SECONDS)
                if status == "lost":
                    reason = reader.reason
                elif status == "ack":
                    state.last_progress = time.monotonic()
            if reason:
                if not state.done.is_set():
                    self._cancel_request(state, reason)
                return
            if time.monotonic() - last_probe >= HOST_WAIT_HEARTBEAT_SECONDS:
                last_probe = time.monotonic()
                grace = _host_heartbeat_grace()
                sender.send(
                    {"type": "output", "data": ""},
                    grace=grace, cancellable=False, skip_if_busy=True, lock_wait=grace,
                )

    def _cancel_request(self, state: _BrokerRequest, reason: str) -> None:
        """Cancel one request whose client was lost; idempotent (#1653).

        Setting the token under the launch lock means a guarded launch either
        sees it and runs nothing, or is already registered here for
        termination.  The runner releases the lane only after the group is
        confirmed empty.
        """
        with self._launch_lock:
            with self._journal_lock:
                if state.cancel_started or state.completing:
                    return
                state.cancel_started = True
                if not self._stop_event.is_set():
                    state.client_lost = True
                    state.loss_reason = reason
                state.cancel.set()
                processes = list(state.processes.values())
                fixtures = list(state.fixtures.values())
        if state.client_lost:
            sys.stderr.write(
                f"agent-loop: run-tests client lost ({reason}); cancelled its request "
                f"{state.command} ({state.phase})\n"
            )
        from .runner import terminate_process_group_fully

        for proc in processes:
            terminate_process_group_fully(proc.pid, proc=proc)
        for supervisor, _reserve in fixtures:
            supervisor.request_teardown()
        for supervisor, reserve in fixtures:
            supervisor.finish(budget=reserve)

    def _describe_holder(
        self, state: _BrokerRequest, kind: str, key: str | None = None,
    ) -> str | None:
        """Name the request of this broker that really holds the refused lock (#1653).

        Ownership is reported by the runner at acquisition and release, so a
        request still in its before snapshot, or one whose locks are already
        released, is never named.  A refused lane is matched by its exact key
        (cwd, lane identity and invocation), never by a similar command; the
        worker budget has a single holder per invocation.  ``None`` when no
        request here holds that lock.
        """
        with self._journal_lock:
            holders = [
                request for request in self._requests.values()
                if request is not state and kind in request.holds
                and (kind != "lane" or (key is not None and request.holds[kind] == key))
            ]
        if not holders:
            return None
        holder = min(holders, key=lambda request: request.started)
        return (
            f"`{_bounded(holder.command, 200)}` in `{_bounded(holder.cwd_label, 200)}` ({holder.phase}, "
            f"{time.monotonic() - holder.started:.0f} s); it frees when that command finishes "
            "or is cancelled, not by killing a run-tests client process, and a killed or "
            f"non-reading client is cancelled within about {BROKER_CLIENT_ACK_TIMEOUT_SECONDS:g} s"
        )

    def _record_capture_failure(
        self,
        request: Mapping[str, object] | None,
        error: BaseException,
    ) -> None:
        """Retain a bounded caveat when authoritative execution cannot start."""
        argv = tuple(str(item) for item in request.get("argv", ())) if request else ()
        cwd = str(request.get("cwd")) if request and request.get("cwd") else str(self.root)
        try:
            from .test_runtime import normalize_test_command

            normalized = normalize_test_command(argv, cwd=Path(cwd))
        except Exception:
            normalized = ""
        observation = LocalTestObservation(
            command=argv,
            outcome="incomplete",
            provenance="telemetry-unverified",
            scope=EvidenceScope("unknown", ()),
            receipt_id=uuid.uuid4().hex,
            turn_id=self.turn_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cwd=cwd,
            normalized_command=normalized,
            attribution=TreeAttribution(
                state="unknown",
                stable=False,
                caveats=("broker context or infrastructure failure",),
            ),
            environment_state="identity-unknown",
            caveats=(f"local test capture incomplete: {type(error).__name__}",),
        )
        with self._journal_lock:
            self._append_journal_locked(observation)

    def _new_execution_ref_locked(self) -> str:
        """Mint a collision-checked opaque selector for one observation."""
        while True:
            self._next_execution_ordinal += 1
            candidate = f"{self._execution_namespace}:observation-{self._next_execution_ordinal}"
            if candidate not in self._execution_refs:
                self._execution_refs.add(candidate)
                return candidate

    def _append_journal_locked(self, observation: LocalTestObservation) -> LocalTestObservation | None:
        """Append within the bound while retaining measured failures longest.

        The returned object is the exact observation submitted to the journal,
        including its minted selector.  Keep that newest observation retained
        when it is a managed observation so a completed test cannot be paired
        with an older journal entry when the bounded catalog is saturated.
        Telemetry-only capture failures may be evicted immediately.
        """
        if observation.execution_ref is None:
            observation = replace(
                observation,
                execution_ref=self._new_execution_ref_locked(),
            )
        self._journal.append(observation)
        while len(self._journal) > MAX_PRIVATE_OBSERVATIONS:
            # A completed managed observation must remain addressable long
            # enough for its caller to receive the selector minted above.
            # Telemetry-only capture failures do not have that requirement;
            # when the journal is saturated they must not evict measured
            # failures merely because they are newest.
            candidate_rows = (
                self._journal[:-1]
                if observation.provenance != "telemetry-unverified"
                else self._journal
            )
            discard = next(
                (
                    index
                    for index, row in enumerate(candidate_rows)
                    if row.provenance == "telemetry-unverified"
                ),
                None,
            )
            if discard is None:
                discard = next(
                    (
                        index
                        for index, row in enumerate(candidate_rows)
                        if row.is_non_evidence_launch_failure
                    ),
                    None,
                )
            if discard is None:
                discard = next(
                    (
                        index
                        for index, row in enumerate(candidate_rows)
                        if not row.is_failure
                    ),
                    0,
                )
            del self._journal[discard]
        return observation if any(
            row.execution_ref == observation.execution_ref for row in self._journal
        ) else None

    def _execute_request(
        self,
        request: Mapping[str, object],
        connection: socket.socket,
        sender: "_ConnectionSender | None" = None,
        state: "_BrokerRequest | None" = None,
    ) -> dict[str, object]:
        from .containment import open_confined_cwd

        if sender is None:
            sender = _ConnectionSender(connection, self._stop_event)
        if state is None:
            state = _BrokerRequest(self._stop_event, command="", normalized="")

        argv = tuple(request["argv"])  # type: ignore[arg-type]
        requested_cwd = Path(request["cwd"])  # type: ignore[arg-type]
        environment = dict(request["environment"])  # type: ignore[arg-type]
        # The server, not the client, decides the actual target environment.
        # Control variables are never forwarded to the target.  The server
        # owns the authenticated turn identity that remains in its context.
        for name in (
            BROKER_ENDPOINT_ENV, BROKER_CAPABILITY_ENV, BROKER_PROTOCOL_ENV, FIXTURE_STATE_DIR_ENV,
            BROKER_CLIENT_ACKS_ENV,
        ):
            environment.pop(name, None)
        # Issue #1632: the client's tool identity is diagnostic only; it never
        # reaches the target or its environment identity.
        client_tool = {
            name: str(environment.pop(name)) for name in TOOL_IDENTITY_ENVS if name in environment
        }
        tool_identity = self._tool_identity(client_tool)
        environment["AGENT_LOOP_INVOCATION_ID"] = self.turn_id
        pinned_root = self._pinned_root
        if pinned_root is None:
            raise BrokerProtocolError("test broker root is not pinned")
        fixture_request = request.get(BROKER_FIXTURE_FIELD)
        fixture_run: Any | None = None
        with open_confined_cwd(pinned_root, requested_cwd) as confined:
            cwd = confined.path
            if fixture_request is not None:
                from .fixture_runtime import (
                    validate_fixture_path,
                    validate_fixture_timeout,
                    verify_host_process_authority,
                )

                # Issue #1326: every refusal happens before anything launches.
                validate_fixture_timeout(float(request["timeout_seconds"]))
                authority = verify_host_process_authority()
                provenance = validate_fixture_path(self.root, cwd, str(fixture_request))
            before = stable_tracked_tree_snapshot(
                self.root,
                argv=argv,
                timeout_seconds=min(MAX_SNAPSHOT_SECONDS, self.timeout_ceiling),
            )
            if state.cancel.is_set():
                # Issue #1653: the client was lost (or the broker stopped)
                # during the bounded before snapshot; nothing launches.
                return _cancelled_frame(BROKER_CLIENT_LOST_DIAGNOSTIC) if state.client_lost else _cancelled_frame()
            if fixture_request is not None:
                from .fixture_runtime import require_clean_tree

                require_clean_tree(before)
                fixture_run = _FixtureRun(
                    provenance=provenance,
                    mount_id=authority.mount_id,
                    timeout_seconds=float(request["timeout_seconds"]),
                )
                fixture_run.request = state
                environment[FIXTURE_STATE_DIR_ENV] = str(fixture_run.state_dir)

            def bounded_grace(grace: float) -> float:
                # Issue #1326: in a fixture run no send may outlive the inner
                # watchdog cutoff, so a client that stopped reading can never
                # push termination, teardown or the sweep past the deadline.
                if fixture_run is None:
                    return grace
                from .runner import DEADLINE_TERMINATION_ALLOWANCE_SECONDS

                cutoff = fixture_run.inner_deadline - DEADLINE_TERMINATION_ALLOWANCE_SECONDS
                return max(0.0, min(grace, cutoff - time.monotonic()))

            def stream(chunk: str) -> None:
                safe = chunk.encode("utf-8", errors="replace")[:16 * 1024].decode("utf-8", errors="ignore")
                try:
                    # Issue #1653: a send that wedges marks the client lost;
                    # the monitor then cancels the request and stops its target.
                    sender.send(
                        {"type": "output", "data": safe},
                        grace=bounded_grace(BROKER_STREAM_SEND_GRACE_SECONDS),
                    )
                except OSError:
                    pass

            def heartbeat() -> None:
                # Only the host-capacity wait calls this before launch.
                if state.phase == "preparing":
                    state.phase = "waiting for host capacity"
                sender.send(
                    {"type": "output", "data": ""},
                    grace=bounded_grace(_host_heartbeat_grace()), skip_if_busy=True,
                )

            def wait_notice(text: str) -> None:
                if state.phase == "preparing" and text.startswith("agent-loop worker budget:"):
                    state.phase = "waiting for host capacity"
                sender.send(
                    {"type": "output", "data": text + "\n"},
                    grace=bounded_grace(_host_heartbeat_grace()),
                )

            parent_cgroup = None
            if self._execute is None:
                handle = self._parent_containment_handle
                if handle is not None and getattr(handle, "managed", False):
                    report = handle.refresh_report()
                    if report is None or not report.target_started or handle.cgroup_path is None:
                        if fixture_run is not None:
                            fixture_run.close()
                        raise BrokerProtocolError("coder containment scope is not ready for broker tests")
                    parent_cgroup = handle.cgroup_path
            setup_hook = (
                self._fixture_setup_hook(
                    fixture_run, cwd=cwd, cwd_fd=confined.fd,
                    environment=environment, parent_cgroup=parent_cgroup, state=state,
                )
                if fixture_run is not None else None
            )
            try:
                if self._execute is not None:
                    result = self._execute_injected(
                        argv, cwd, float(request["timeout_seconds"]), environment, stream,
                        fixture_run=fixture_run, setup_hook=setup_hook,
                    )
                else:
                    result = self._run_foreground(
                        argv, cwd=cwd, cwd_fd=confined.fd,
                        timeout_seconds=float(request["timeout_seconds"]),
                        environment=environment, stream=stream, heartbeat=heartbeat,
                        wait_notice=wait_notice, parent_cgroup=parent_cgroup,
                        fixture_run=fixture_run, setup_hook=setup_hook, state=state,
                    )
            finally:
                if fixture_run is not None:
                    # Teardown, the sweep and the census run on success,
                    # failure, exceptions and stop alike, before the after
                    # snapshot (issue #1326).
                    fixture_run.finish(self)
            with self._journal_lock:
                # From here a lost client no longer cancels anything: the
                # target is gone and its locks are released.
                state.completing = True
                state.phase = "completing"
            return self._complete_request(
                argv=argv, cwd=cwd, environment=environment, before=before,
                result=result, fixture_run=fixture_run, root=pinned_root.path,
                tool_identity=tool_identity, client_lost=state.client_lost,
            )

    def _tool_identity(self, client: Mapping[str, str]) -> ToolIdentity | None:
        """Compare a request's client identity with this orchestrator; never raises."""
        try:
            orchestrator = _orchestrator_tool_provenance()
            identity = build_tool_identity(client, orchestrator)
        except Exception:  # noqa: BLE001 - a diagnostic must never refuse a run
            return None
        if identity.status == "mismatch":
            # Raw package paths go only to the orchestrator's own log.
            key = (client.get(TOOL_COMMIT_ENV), client.get(TOOL_PACKAGE_ENV))
            if key not in self._tool_mismatch_logged:
                self._tool_mismatch_logged.add(key)
                sys.stderr.write(
                    "agent-loop: Warning: run-tests receipt produced by agent-loop "
                    f"{client.get(TOOL_COMMIT_ENV) or 'unknown'} from "
                    f"{client.get(TOOL_PACKAGE_ENV) or '<unknown package>'}; orchestrator is "
                    f"{orchestrator.get('commit') or 'unknown'} from "
                    f"{orchestrator.get('package_path') or '<unknown package>'}\n"
                )
        return identity

    def _run_foreground(
        self, argv: tuple[str, ...], *, cwd: Path, cwd_fd: int, timeout_seconds: float,
        environment: dict[str, str], stream: Any, heartbeat: Any, wait_notice: Any,
        parent_cgroup: Path | None, fixture_run: "_FixtureRun | None", setup_hook: Any | None,
        state: "_BrokerRequest | None" = None,
    ) -> Any:
        from .runner import run_foreground_test

        if state is None:
            state = _BrokerRequest(self._stop_event, command="", normalized="")
        request_state = state

        def started(proc: Any) -> None:
            with self._journal_lock:
                self._active_processes[proc.pid] = proc
                request_state.processes[proc.pid] = proc
                request_state.phase = "running"
            if self._process_started is not None:
                self._process_started(proc)

        def lock_observer(kind: str, held: bool, key: str | None = None) -> None:
            with self._journal_lock:
                if held:
                    request_state.holds[kind] = key
                else:
                    request_state.holds.pop(kind, None)

        def finished(proc: Any) -> None:
            with self._journal_lock:
                self._active_processes.pop(proc.pid, None)
                request_state.processes.pop(proc.pid, None)
            if self._process_finished is not None:
                self._process_finished(proc)

        from .worker_telemetry import ReservationTelemetry, telemetry_log_path

        telemetry = ReservationTelemetry(
            telemetry_log_path(),
            self._telemetry_attribution
            or {"attribution_source": "none", "lane": "broker"},
        )
        from .test_runtime import resolve_adopted_package_script

        # One static package.json read per request (issue #1294).
        package_script = resolve_adopted_package_script(
            argv, cwd=cwd, environment=environment
        )
        return run_foreground_test(
            argv,
            package_script=package_script,
            reservation_telemetry=telemetry,
            worker_budget=self.effective_worker_budget(environment),
            worker_lock_root=self._worker_lock_root,
            host_wait_cancel=request_state.cancel,
            host_wait_heartbeat=heartbeat,
            host_wait_notify=wait_notice,
            host_launch_guard=lambda: self._launch_lock,
            lock_holder_describer=lambda kind, key=None: self._describe_holder(request_state, kind, key),
            lock_observer=lock_observer,
            # Issue #1653: a fixture's teardown and sweep finish before the
            # lane or worker budget is released, on every path that reached
            # the setup hook (cancel, failure or a completed target).
            before_lock_release=(
                (lambda: fixture_run.finish(self)) if fixture_run is not None else None
            ),
            cwd=cwd,
            cwd_fd=cwd_fd,
            timeout_seconds=timeout_seconds,
            env=environment,
            # Broker children either attach to the requesting managed
            # coder scope above or use their registered process group.
            # They must never infer containment from the broker
            # parent's own cgroup.
            containment_policy=None,
            containment_role="test-gate",
            environment_is_complete=True,
            output_callback=stream,
            echo_output=False,
            parent_cgroup_path=parent_cgroup,
            process_started=started,
            process_finished=finished,
            wrapper_bootstrap="verified",
            health_provenance="broker-parent",
            classify_pre_collection=True,
            # Issue #1326: both stay None for fixture-less requests.
            absolute_deadline=(fixture_run.inner_deadline if fixture_run is not None else None),
            post_admission_setup=setup_hook,
        )

    def _execute_injected(
        self, argv: tuple[str, ...], cwd: Path, timeout: float, environment: dict[str, str],
        stream: Any, *, fixture_run: "_FixtureRun | None", setup_hook: Any | None,
    ) -> Any:
        """The test-injected executor, with the fixture hook applied around it."""
        assert self._execute is not None
        if setup_hook is not None and fixture_run is not None:
            decision = setup_hook(fixture_run.inner_deadline)
            if not decision.proceed:
                from types import SimpleNamespace

                return SimpleNamespace(
                    outcome=(
                        "cancelled" if decision.action == "cancel"
                        else "timed_out" if decision.action == "timed_out" else "failed"
                    ),
                    returncode=None, elapsed_seconds=0.0, output_tail=decision.reason,
                    wrapper_bootstrap="verified", inner_exec="not-attempted",
                    suite_start="not-started", diagnostic=decision.reason, args=list(argv),
                )
        return self._execute(argv, cwd, timeout, environment, stream)

    def _fixture_setup_hook(
        self, fixture_run: "_FixtureRun", *, cwd: Path, cwd_fd: int,
        environment: Mapping[str, str], parent_cgroup: Path | None,
        state: "_BrokerRequest | None" = None,
    ) -> Any:
        """The runner's post-admission hook: launch the supervisor, run setup.

        Issue #1653: every cancel check reads the request's own token, so a
        lost client cancels this request alone.
        """
        cancel: Event = state.cancel if state is not None else self._stop_event
        from .fixture_runtime import (
            MIN_SETUP_AND_INNER_SECONDS,
            SETUP_TERMINATION_MARGIN_SECONDS,
            FixtureSupervisor,
            SetupDecision,
            teardown_budget_seconds,
        )

        def hook(deadline: float) -> Any:
            if deadline - time.monotonic() < MIN_SETUP_AND_INNER_SECONDS:
                fixture_run.decision = SetupDecision(
                    "timed_out", "fixture setup budget exhausted before setup could start",
                )
                return fixture_run.decision
            # The host_launch_guard handshake: a stop either sees this
            # registered supervisor or this launch sees the stop.
            with self._launch_lock:
                if cancel.is_set():
                    fixture_run.decision = SetupDecision(
                        "cancel",
                        "broker stopping before the fixture launch" if self._stop_event.is_set()
                        else "run-tests client lost before the fixture launch",
                    )
                    return fixture_run.decision
                if state is not None:
                    state.phase = "fixture setup"
                # Setup stops early enough for its termination grace to end
                # by the inner deadline; both deadlines are absolute.
                setup_deadline = deadline - SETUP_TERMINATION_MARGIN_SECONDS
                supervisor = FixtureSupervisor.launch(
                    fixture_run.provenance, cwd=cwd, cwd_fd=cwd_fd, environment=environment,
                    setup_deadline=setup_deadline,
                    request_deadline=fixture_run.request_deadline,
                    teardown_seconds=teardown_budget_seconds(fixture_run.timeout_seconds),
                    mount_id=fixture_run.mount_id,
                )
                fixture_run.supervisor = supervisor
                with self._journal_lock:
                    self._active_fixtures[id(supervisor)] = (supervisor, fixture_run.reserve)
                    if state is not None:
                        state.fixtures[id(supervisor)] = (supervisor, fixture_run.reserve)
            try:
                released = supervisor.start_setup(parent_cgroup, setup_deadline=setup_deadline)
            except (AgentLoopError, OSError) as exc:
                fixture_run.decision = SetupDecision(
                    "skip", f"fixture supervisor failed: {_safe_text(exc)}",
                )
                return fixture_run.decision
            if not released:
                fixture_run.decision = SetupDecision(
                    "timed_out", "fixture setup budget expired before the supervisor was ready",
                )
                return fixture_run.decision
            # The supervisor stops setup at its deadline; the report must
            # arrive by the inner deadline, never inside the teardown reserve.
            event = supervisor.wait_setup(deadline)
            status = event.get("status") if event is not None else None
            if cancel.is_set() or status == "cancelled":
                fixture_run.decision = SetupDecision(
                    "cancel",
                    "run-tests client lost during fixture setup"
                    if state is not None and state.client_lost
                    else "broker stopping during fixture setup",
                )
            elif status == 0 and not isinstance(status, bool):
                fixture_run.decision = SetupDecision("proceed")
            else:
                fixture_run.decision = SetupDecision(
                    "skip",
                    "fixture setup failed (status "
                    + (str(status) if status is not None else "unknown") + ")",
                )
            return fixture_run.decision

        return hook

    def _complete_request(
        self, *, argv: tuple[str, ...], cwd: Path, environment: Mapping[str, str],
        before: "TrackedTreeSnapshot", result: Any, fixture_run: "_FixtureRun | None",
        root: Path | None = None, tool_identity: ToolIdentity | None = None,
        client_lost: bool = False,
    ) -> dict[str, object]:
        """Snapshot, attribute, fold, journal and build the result frame.

        ``client_lost`` (#1653): the broker stopped this target because its
        client was lost; its row is journaled as non-evidence.
        """
        launched_fixture = fixture_run is not None and fixture_run.supervisor is not None
        if str(getattr(result, "outcome", "")) == "cancelled" and not launched_fixture:
            # Nothing launched: no post-run snapshot, attribution or evidence.
            return _cancelled_frame(BROKER_CLIENT_LOST_DIAGNOSTIC) if client_lost else _cancelled_frame()
        shutdown = self._stop_event.is_set()
        after = None
        if not shutdown and not client_lost:
            try:
                after = stable_tracked_tree_snapshot(
                    self.root,
                    argv=argv,
                    timeout_seconds=min(MAX_SNAPSHOT_SECONDS, self.timeout_ceiling),
                    cancel=self._stop_event,
                )
            except SnapshotCancelled:
                shutdown = True
        if shutdown or after is None:
            # The target ran before the broker stopped: record it, but never as
            # attributable (passing) evidence.
            attribution = TreeAttribution(
                state="unknown",
                stable=False,
                caveats=("unattributed: broker shutdown",),
            )
        else:
            attribution = attribute_current_head(before, after, current_head=after.head, argv=argv)
        identity = self._environment_registry.capture(environment)
        from .test_runtime import normalize_test_command
        # Keep the shared registry private to the broker process; the bytes are
        # not present in the journal's public projection.
        receipt_id = uuid.uuid4().hex
        execution_ref: str | None = None
        suite_start = str(getattr(result, "suite_start", "unknown"))
        worker_enforcement = getattr(result, "worker_enforcement", None)
        worker_caveats = tuple(str(item) for item in getattr(result, "worker_caveats", ()) or ())
        journal_caveats: tuple[str, ...] = ("output stream was broker-forwarded",)
        if worker_enforcement is not None and (
            worker_enforcement in {"clamped", "unverified", "exceeded", "refused-in-command"}
            or worker_caveats
        ):
            journal_caveats += (
                "worker budget (top-level runner): "
                + str(worker_enforcement)
                + (f"; {', '.join(worker_caveats)}" if worker_caveats else "")
                + "; executed argv: "
                + " ".join(str(item) for item in getattr(result, "args", argv))[:512],
            )
        pre_collection = getattr(result, "pre_collection_launch_failure", None)
        result_outcome = str(result.outcome)
        result_diagnostic = getattr(result, "diagnostic", "")
        if isinstance(pre_collection, str) and pre_collection and result_outcome == "failed":
            # Issue #1182: parent-observed non-evidence launch failure.  The
            # row stays visible with the caveat first; the frame reports it
            # as ``launch-failed``/``not-started`` so no timing or health is
            # recorded for it.
            caveat = _safe_text(
                f"{PRE_COLLECTION_LAUNCH_FAILURE_PREFIX}: {pre_collection}", MAX_SAFE_CAVEAT_BYTES,
            )
            journal_caveats = (caveat, *journal_caveats)
            result_outcome = "launch-failed"
            suite_start = "not-started"
            result_diagnostic = caveat
        else:
            pre_collection = None
        inner_exec = str(getattr(result, "inner_exec", "not-attempted"))
        suite_start_reason = coerce_suite_start_reason(
            getattr(result, "suite_start_reason", None), suite_start
        )
        fixture_record: FixtureRecord | None = None
        fixture_diagnostic = ""
        journal_fixture_row = False
        if fixture_run is not None:
            fixture_record, fixture_diagnostic, result_outcome, fixture_caveats = fixture_run.fold(
                # A lost client after the supervisor launched is an
                # interruption like a stop, so the lifecycle is journaled.
                result_outcome, inner_exec=inner_exec,
                shutdown=self._stop_event.is_set() or client_lost,
                diagnostic=str(result_diagnostic or ""),
            )
            journal_caveats = (*fixture_caveats, *journal_caveats)
            # Fixture rows bypass the suite_start gate: a setup failure, a
            # deadline before setup or a stop is still a journaled lifecycle.
            journal_fixture_row = result_outcome not in {
                "overlap-rejected", "worker-budget-busy", "worker-budget-refused", "cancelled",
            } and (launched_fixture or fixture_run.decision is not None)
        if (
            journal_fixture_row
            or (suite_start != "not-started" or pre_collection)
            and str(getattr(result, "outcome", "")) not in {
                "overlap-rejected", "worker-budget-busy", "worker-budget-refused", "cancelled",
            }
        ):
            observation = LocalTestObservation(
                command=argv,
                outcome=result_outcome,
                provenance="parent-observed",
                scope=EvidenceScope("unknown", ()),
                receipt_id=receipt_id,
                turn_id=self.turn_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                cwd=str(cwd),
                normalized_command=normalize_test_command(argv, cwd=cwd),
                returncode=result.returncode,
                attribution=attribution,
                environment_state="not-compared",
                environment_identity=identity,
                caveats=journal_caveats,
                diagnostic=(
                    "\n".join(item for item in (getattr(result, "output_tail", None), fixture_diagnostic) if item)
                    if fixture_run is not None else getattr(result, "output_tail", None)
                ),
                wrapper_bootstrap=str(getattr(result, "wrapper_bootstrap", "unknown")),
                inner_exec=inner_exec,
                suite_start=suite_start,
                suite_start_reason=suite_start_reason,
                executed_command=tuple(
                    str(item) for item in (getattr(result, "args", ()) or ())
                ),
                fixture=fixture_record,
                # Issue #1633: computed over the raw argv and the confined cwd
                # relative to the pinned root, never from redacted output.
                restart_command_digest=(
                    _restart_command_digest(argv, cwd, root) if root is not None else None
                ),
                tool_identity=tool_identity,
            )
            if client_lost:
                # Applied after fixture folding, so folding can never restore
                # an authoritative outcome; the receipt stays visible.
                observation = replace(
                    observation,
                    outcome="incomplete",
                    provenance="telemetry-unverified",
                    attribution=TreeAttribution(
                        state="unknown", stable=False, caveats=(BROKER_CLIENT_LOST_CAVEAT,),
                    ),
                    caveats=(BROKER_CLIENT_LOST_CAVEAT, *observation.caveats),
                )
            with self._journal_lock:
                if self._stop_event.is_set() and "unattributed: broker shutdown" not in observation.attribution.caveats:
                    observation = replace(
                        observation,
                        attribution=TreeAttribution(
                            state="unknown", stable=False,
                            caveats=("unattributed: broker shutdown",),
                        ),
                    )
                retained = self._append_journal_locked(observation)
                execution_ref = retained.execution_ref if retained is not None else None
        if fixture_record is not None and result_outcome in {"interrupted", "cancelled"}:
            # The client still receives the cancelled frame, now carrying the
            # fixture lifecycle and the interrupted row's receipt.
            return {
                **_cancelled_frame(),
                "receipt_id": receipt_id if execution_ref is not None else "",
                "execution_ref": execution_ref,
                "diagnostic": (
                    "fixture run interrupted (its run-tests client was lost)" if client_lost
                    else "fixture run interrupted (the broker is stopping)"
                ),
                BROKER_FIXTURE_FIELD: fixture_record.to_dict(),
            }
        response: dict[str, object] = {
            "type": "result",
            "receipt_id": receipt_id,
            "execution_ref": execution_ref,
            "outcome": result_outcome,
            "returncode": result.returncode,
            "elapsed_seconds": float(result.elapsed_seconds),
            "output_tail": _safe_text(getattr(result, "output_tail", ""), 8 * 1024),
            "wrapper_bootstrap": str(getattr(result, "wrapper_bootstrap", "unknown")),
            "inner_exec": inner_exec,
            "suite_start": suite_start,
            # A no-test refusal's reason (and its lock holder, #1653) is the
            # client's only explanation, so it keeps a larger bound.
            "diagnostic": _safe_text(
                result_diagnostic,
                BROKER_REFUSAL_DIAGNOSTIC_BYTES if result_outcome in BROKER_NO_TEST_OUTCOMES
                else MAX_SAFE_CAVEAT_BYTES,
            ),
            "executed_argv": _bounded_argv(getattr(result, "args", argv) or argv),
            "workers_cohort": getattr(result, "workers_cohort", None),
            "worker_enforcement": worker_enforcement,
            "worker_caveats": list(worker_caveats)[:16],
            "worker_environment": self._worker_environment_summary(environment),
        }
        if suite_start_reason is not None:
            # Issue #1661: present only beside ``suite_start=unknown``, so
            # every other frame is unchanged for older clients.
            response["suite_start_reason"] = suite_start_reason
        if fixture_record is not None:
            response[BROKER_FIXTURE_FIELD] = fixture_record.to_dict()
        if self.dependency_remedy and (result_outcome == "launch-failed" or inner_exec == "failed"):
            response[BROKER_ADVISORY_FIELD] = _safe_text(self.dependency_remedy, 1024)
        return response

    def _worker_environment_summary(self, environment: Mapping[str, str]) -> dict[str, str]:
        """Effective controlled values, without PYTHONPATH/PYTEST_PLUGINS contents."""
        budget = self.effective_worker_budget(environment)
        if budget is None:
            return {}
        from .test_workers import ENV_TEST_WORKER_ENFORCEMENT, ENV_TEST_WORKERS

        summary = {
            ENV_TEST_WORKERS: str(budget.workers),
            ENV_TEST_WORKER_ENFORCEMENT: str(budget.enforcement),
            "plugin": "present" if budget.enforcement != "off" else "absent",
        }
        return summary


class TestBrokerClient:
    """Client-side ``run-tests`` adapter using the startup environment snapshot."""

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        values = dict(environment or os.environ)
        self._environment = values
        self.endpoint = values.get(BROKER_ENDPOINT_ENV)
        self.capability = values.get(BROKER_CAPABILITY_ENV)
        self.protocol = values.get(BROKER_PROTOCOL_ENV)
        self.turn_id = values.get("AGENT_LOOP_INVOCATION_ID")

    @property
    def available(self) -> bool:
        return bool(self.endpoint and self.capability and self.protocol == BROKER_PROTOCOL and self.turn_id)

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        cwd: Path | None = None,
        environment_overrides: Mapping[str, str] | None = None,
        fixture: str | None = None,
    ) -> BrokerRunResult:
        """Run one command through the broker.

        ``BrokerNotDispatched`` means the request frame was never fully
        written, so nothing ran; ``BrokerResultUnknown`` means it was sent and
        the broker owns whatever happened.  Both subclass
        ``BrokerProtocolError`` so fixture-less callers see no change.
        """
        if not self.available:
            raise BrokerNotDispatched("test broker is unavailable")
        assert self.endpoint is not None and self.capability is not None and self.turn_id is not None
        snapshot_cwd = str(cwd or Path.cwd())
        snapshot_environment = dict(self._environment)
        # Exact protocol variables are transport-only. Do not wildcard-delete
        # test variables or virtual-environment/PATH state.
        for name in (BROKER_ENDPOINT_ENV, BROKER_CAPABILITY_ENV, BROKER_PROTOCOL_ENV):
            snapshot_environment.pop(name, None)
        snapshot_environment["AGENT_LOOP_INVOCATION_ID"] = self.turn_id
        # Issue #1653: this client acknowledges every heartbeat.
        snapshot_environment[BROKER_CLIENT_ACKS_ENV] = "1"
        if environment_overrides:
            # The client's requested worker values; the broker only honours
            # them when they are stricter than the parent-owned budget.
            snapshot_environment.update({str(k): str(v) for k, v in environment_overrides.items()})
        request = {
            "turn_id": self.turn_id,
                "nonce": (
                    (raw_nonce := uuid.uuid4().hex)
                    + "."
                    + hmac.new(
                        self.capability.encode("ascii"), raw_nonce.encode("ascii"), hashlib.sha256
                    ).hexdigest()
                ),
            "argv": [str(item) for item in argv],
            "timeout_seconds": timeout_seconds,
            "cwd": snapshot_cwd,
            "environment": snapshot_environment,
        }
        if fixture is not None:
            # Omitted when absent so the fixture-less request (and its replay
            # digest) stays byte-identical.
            request[BROKER_FIXTURE_FIELD] = str(fixture)
            # Declared overhead: the before/after tree snapshots run outside
            # the request deadline (issue #1326).
            socket_bound = float(timeout_seconds) + 2 * MAX_SNAPSHOT_SECONDS + 10.0
        else:
            socket_bound = max(5.0, min(float(timeout_seconds) + 10.0, 1800.0))
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(socket_bound)
            try:
                connection.connect(self.endpoint)
                # The capability is an out-of-band secret carried by a protected
                # environment and turned into the exact request binding. It is
                # intentionally not part of the request schema or any response.
                request["turn_id"] = self.turn_id
                _send_frame(connection, request)
            except BrokerProtocolError as exc:
                # Only an oversized frame, refused before any byte was sent.
                raise BrokerNotDispatched(str(exc)) from exc
            except (OSError, ValueError) as exc:
                raise BrokerNotDispatched(
                    f"broker request was not dispatched: {type(exc).__name__}"
                ) from exc
            acks_disabled = False
            while True:
                try:
                    response = _recv_frame(connection)
                except BrokerProtocolError as exc:
                    raise BrokerResultUnknown(str(exc)) from exc
                except (OSError, ValueError) as exc:
                    raise BrokerResultUnknown(
                        f"broker response lost after dispatch: {type(exc).__name__}"
                    ) from exc
                if response.get("type") == "output":
                    data = response.get("data")
                    if isinstance(data, str):
                        print(data, end="", flush=True)
                    if data == "" and not acks_disabled:
                        # Acknowledged only after every earlier frame was
                        # handled.  Best effort: a failed or slow ack (the
                        # broker may already have sent its result and closed)
                        # disables acks and the client keeps reading.
                        acks_disabled = not _send_ack(connection, socket_bound)
                    continue
                if response.get("type") == "error":
                    raise BrokerProtocolError(str(response.get("error", "broker request failed")))
                if response.get("type") != "result":
                    raise BrokerProtocolError("unknown broker response")
                return BrokerRunResult(
                    receipt_id=str(response.get("receipt_id", "")),
                    execution_ref=(
                        str(response["execution_ref"])
                        if response.get("execution_ref") is not None else None
                    ),
                    outcome=str(response.get("outcome", "incomplete")),
                    returncode=(int(response["returncode"]) if isinstance(response.get("returncode"), int) else None),
                    elapsed_seconds=float(response.get("elapsed_seconds", 0.0)),
                    output_tail=str(response.get("output_tail", "")),
                    wrapper_bootstrap=str(response.get("wrapper_bootstrap", "unknown")),
                    inner_exec=str(response.get("inner_exec", "not-attempted")),
                    suite_start=str(response.get("suite_start", "not-started")),
                    suite_start_reason=coerce_suite_start_reason(
                        response.get("suite_start_reason"), response.get("suite_start", "not-started")
                    ),
                    diagnostic=str(response.get("diagnostic", "")),
                    executed_argv=tuple(
                        str(item) for item in (response.get("executed_argv") or ())
                        if isinstance(item, str)
                    ),
                    workers_cohort=(
                        str(response["workers_cohort"])
                        if isinstance(response.get("workers_cohort"), str) else None
                    ),
                    worker_enforcement=(
                        str(response["worker_enforcement"])
                        if isinstance(response.get("worker_enforcement"), str) else None
                    ),
                    worker_caveats=tuple(
                        str(item) for item in (response.get("worker_caveats") or ())
                        if isinstance(item, str)
                    ),
                    worker_environment={
                        str(key): str(value)
                        for key, value in dict(response.get("worker_environment") or {}).items()
                    },
                    fixture=(
                        decode_fixture_record(response[BROKER_FIXTURE_FIELD])[0].to_dict()
                        if BROKER_FIXTURE_FIELD in response else None
                    ),
                    advisory=(
                        str(response[BROKER_ADVISORY_FIELD])
                        if isinstance(response.get(BROKER_ADVISORY_FIELD), str) else ""
                    ),
                )


def _send_ack(connection: socket.socket, restore_timeout: float) -> bool:
    """Send one bounded heartbeat acknowledgement; False when it failed."""
    from .test_workers import HOST_WAIT_HEARTBEAT_SEND_GRACE_SECONDS

    try:
        connection.settimeout(HOST_WAIT_HEARTBEAT_SEND_GRACE_SECONDS)
        _send_frame(connection, {"type": "ack"})
        return True
    except (OSError, BrokerProtocolError):
        return False
    finally:
        try:
            connection.settimeout(restore_timeout)
        except OSError:
            pass


def broker_client_from_environment(environment: Mapping[str, str] | None = None) -> TestBrokerClient | None:
    client = TestBrokerClient(environment)
    return client if client.available else None


# Friendly aliases for callers and integrations.
TestObservation = LocalTestObservation
TestScope = EvidenceScope
EnvironmentRegistry = EnvironmentIdentityRegistry
reconcile = reconcile_test_observations
parse_legacy_test_commands = parse_legacy_tests_run
safe_public_observation = LocalTestObservation.public_projection


def unsuperseded_receipts_sentence(value: object) -> str:
    """One bounded sentence naming unsuperseded authoritative failure receipts.

    Used by the unchanged-head stop (issue #1182) so a loop that stops while
    the only open item is evidence bookkeeping is not read as the coder
    failing to act.  Empty when there are no such receipts.
    """
    evidence = decode_bounded_evidence(value, restore_classification=True) if value else None
    if evidence is None:
        return ""
    failing = [
        row for row in evidence.observations
        if row.is_failure
        and row.provenance == "parent-observed"
        and not row.superseded_by
        and not row.is_out_of_checkout_context
        and not row.is_non_evidence_launch_failure
    ]
    if not failing:
        return ""
    named = []
    for row in failing[:3]:
        command, _ids, _caveats = redact_test_command(row.command)
        named.append(f"`{_bounded(command, 120)}` ({row.outcome})")
    more = f" and {len(failing) - 3} more" if len(failing) > 3 else ""
    return (
        " The latest local test evidence still holds unsuperseded failure receipts: "
        + ", ".join(named) + more
        + ". This stop may be evidence bookkeeping rather than a missing code change; an operator "
        "can verify the command and decide whether the receipts are still meaningful."
    )


def tool_identity_clause(value: object) -> str:
    """One bounded clause naming the first run-tests tool diagnostic (#1632), or ``""``.

    Accepts decoded evidence or a bounded evidence string.
    """
    evidence = value
    if not isinstance(value, LocalTestEvidence):
        evidence = decode_bounded_evidence(value, restore_classification=True) if value else None
    if evidence is None:
        return ""
    diagnostics = [
        diagnostic
        for row in evidence.observations
        if (diagnostic := tool_identity_diagnostic(row)) is not None
    ]
    if not diagnostics:
        return ""
    more = f" ({len(diagnostics) - 1} more row(s) flagged)" if len(diagnostics) > 1 else ""
    return (
        f" Run-tests tool identity: {_bounded(diagnostics[0], 240)}{more}; receipts are kept and "
        "judged as before."
    )


def superseded_by_current_turn(
    evidence: LocalTestEvidence | None, current_turn_receipt_ids: Iterable[str]
) -> frozenset[str]:
    """Receipts from this coder turn that superseded an authoritative failure (#1626).

    Reads the live, unbounded reconciliation, never the bounded round
    projection: a failure that merely fell out of the byte-capped projection
    never counts.  Only failure rows that ``unsuperseded_receipts_sentence``
    would otherwise name (parent-observed, not out-of-checkout, not a
    non-evidence launch failure) contribute, and only a ``superseded_by``
    value in ``current_turn_receipt_ids`` is returned, so the tree-change
    label or an earlier turn's receipt never does.
    """
    if evidence is None:
        return frozenset()
    current = frozenset(
        receipt for receipt in current_turn_receipt_ids if isinstance(receipt, str) and receipt
    )
    return frozenset(
        row.superseded_by
        for row in evidence.observations
        if row.is_failure
        and row.provenance == "parent-observed"
        and not row.is_out_of_checkout_context
        and not row.is_non_evidence_launch_failure
        and row.superseded_by
        and row.superseded_by != TREE_CHANGE_SUPERSESSION
        and row.superseded_by in current
    )
