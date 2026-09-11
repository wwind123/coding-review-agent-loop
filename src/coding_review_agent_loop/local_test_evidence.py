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
import json
import math
import os
import re
import signal
import shlex
import socket
import struct
import subprocess
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
MAX_SAFE_IDENTIFIER_BYTES = 256
MAX_SAFE_IDENTIFIERS = 8
MAX_SAFE_CAVEAT_BYTES = 256
MAX_PRIVATE_OBSERVATIONS = 64
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
        "AGENT_LOOP_PUBLIC_RESPONSE_BELOW",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_CODE_CHILD_SESSION",
        "CODEX_SESSION_ID",
        "CODEX_THREAD_ID",
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


@dataclass(frozen=True)
class EnvironmentIdentity:
    """An in-memory identity; ``canonical_bytes`` is never serialized."""

    process_token: str
    canonical_bytes: bytes = field(repr=False, compare=False)

    def compare(self, other: "EnvironmentIdentity | None") -> str:
        if other is None or self.process_token.split(":", 1)[0] != other.process_token.split(":", 1)[0]:
            return "unknown"
        return "equivalent" if self.canonical_bytes == other.canonical_bytes else "different"


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


@dataclass(frozen=True)
class LocalTestObservation:
    command: tuple[str, ...]
    outcome: str
    provenance: str
    scope: EvidenceScope = field(default_factory=EvidenceScope)
    receipt_id: str | None = None
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

    def __post_init__(self) -> None:
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

    def public_projection(self) -> dict[str, object]:
        command, _, _ = redact_test_command(
            self.command,
            cwd=Path(self.cwd) if self.cwd else None,
        )
        projection = {
            "command": _bounded(command, MAX_SAFE_COMMAND_BYTES),
            "receipt_id": _safe_text(self.receipt_id or "", MAX_SAFE_IDENTIFIER_BYTES),
            "turn_id": _safe_text(self.turn_id or "", MAX_SAFE_IDENTIFIER_BYTES),
            "timestamp": _bounded(_safe_text(self.timestamp, 64), 64),
            "outcome": self.outcome,
            "provenance": self.provenance,
            "scope": self.scope.to_dict(),
            "attribution": self.attribution.to_dict(),
            "environment": self.environment_state,
            "returncode": self.returncode,
            "claim": self.claim if self.claim in CLAIMS else None,
            "superseded_by": _safe_text(self.superseded_by or "", MAX_SAFE_IDENTIFIER_BYTES)
            if self.superseded_by
            else None,
            "caveats": [_safe_text(item, MAX_SAFE_CAVEAT_BYTES) for item in self.caveats[:4]],
            "identifiers": [
                _safe_text(item, MAX_SAFE_IDENTIFIER_BYTES)
                for item in self.identifiers[:MAX_SAFE_IDENTIFIERS]
            ],
        }
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
        return projection

    def to_dict(self) -> dict[str, object]:
        return self.public_projection()


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
    command = _coerce_command(value.get("argv", value.get("command", ())))
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
    return LocalTestObservation(
        command=command,
        outcome=str(value.get("outcome", "incomplete")),
        provenance=str(value.get("provenance", default_provenance)),
        scope=EvidenceScope.from_value(value.get("scope")),
        receipt_id=str(value["receipt_id"]) if value.get("receipt_id") is not None else None,
        turn_id=str(value["turn_id"]) if value.get("turn_id") is not None else None,
        timestamp=_timestamp(value.get("timestamp")),
        cwd=str(value["cwd"]) if value.get("cwd") is not None else None,
        normalized_command=normalized,
        returncode=(int(value["returncode"]) if isinstance(value.get("returncode"), int) else None),
        attribution=_coerce_attribution(value.get("attribution", value.get("tree"))),
        environment_state=environment_state,
        environment_identity=environment_identity,
        caveats=tuple(_safe_text(item) for item in value.get("caveats", ()) if item is not None),
        identifiers=tuple(_safe_text(item, MAX_SAFE_IDENTIFIER_BYTES) for item in value.get("identifiers", ()) if item is not None),
        diagnostic=str(value["diagnostic"]) if value.get("diagnostic") is not None else None,
        claim=str(value["claim"]) if value.get("claim") is not None else None,
        wrapper_bootstrap=str(value.get("wrapper_bootstrap", "unknown")),
        inner_exec=str(value.get("inner_exec", "not-attempted")),
        suite_start=str(value.get("suite_start", "not-started")),
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
    )


def _same_receipt(left: LocalTestObservation, right: LocalTestObservation) -> bool:
    return left.receipt_id is not None and left.receipt_id == right.receipt_id


def _can_supersede(
    failure: LocalTestObservation,
    passing: LocalTestObservation,
    *,
    registry: EnvironmentIdentityRegistry | None,
) -> tuple[bool, str | None]:
    if not failure.is_failure or passing.outcome != "passed":
        return False, None
    if passing.provenance != "parent-observed":
        return False, "later pass was not parent-observed"
    if failure.normalized_command != passing.normalized_command:
        return False, "command differs; subset/superset or command disagreement remains visible"
    if failure.scope.normalized() != passing.scope.normalized():
        return False, "scope differs; subset/superset result cannot supersede the failure"
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
            rows.append(item)
        elif isinstance(item, Mapping):
            rows.append(observation_from_mapping(item, cwd=cwd, registry=registry))
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

    # Make live comparisons visible without persisting the environment bytes.
    # A first observation has no comparison partner; later observations with
    # the same command and scope expose divergence or unavailable identity.
    for index, row in enumerate(rows):
        if row.environment_state == "identity-unknown":
            continue
        comparable = next(
            (
                previous
                for previous in reversed(rows[:index])
                if previous.normalized_command == row.normalized_command
                and previous.scope.normalized() == row.scope.normalized()
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
                caveats=(*row.caveats, "effective target environment differs"),
            )
        else:
            rows[index] = replace(
                row,
                environment_state="unknown",
                caveats=(*row.caveats, "effective target environment could not be compared"),
            )

    # Compare live observations with the clean eventual tree at handoff. This
    # lets a stable pre-commit test become attributable after the coder commits.
    if (
        current_snapshot is not None
        and current_snapshot.complete
        and current_snapshot.stable is True
        and current_snapshot.status_clean is True
        and current_snapshot.tracked_digest
    ):
        for index, row in enumerate(rows):
            if row.environment_identity is None:
                continue
            attribution = row.attribution
            if attribution.stable is not True or not attribution.tracked_digest:
                continue
            if attribution.tracked_digest == current_snapshot.tracked_digest:
                state = (
                    "untracked-input-unverified"
                    if attribution.untracked_input
                    or attribution.state == "untracked-input-unverified"
                    else "current-head"
                )
                rows[index] = replace(
                    row,
                    attribution=replace(
                        attribution,
                        state=state,
                        head=current_head or current_snapshot.head,
                    ),
                )
            else:
                rows[index] = replace(
                    row,
                    attribution=replace(attribution, state="stale"),
                    caveats=(*row.caveats, "tested tracked tree differs from the eventual PR tree"),
                )

    for failure_index, failure in enumerate(rows):
        if not failure.is_failure or failure.superseded_by:
            continue
        for later in rows[failure_index + 1 :]:
            if later.outcome != "passed":
                continue
            allowed, reason = _can_supersede(failure, later, registry=registry)
            if allowed:
                receipt = later.receipt_id or f"observation-{rows.index(later)}"
                rows[failure_index] = replace(failure, superseded_by=receipt)
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

    authoritative_failures = tuple(
        row.receipt_id or f"observation-{index}"
        for index, row in enumerate(rows)
        if row.is_failure and row.provenance == "parent-observed" and not row.superseded_by
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
        try:
            argv = tuple(shlex.split(command_text))
        except ValueError as exc:
            argv = (command_text,)
            caveats.extend(("capture-limited", f"unparsable command: {_safe_text(exc)}"))
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


def redact_test_command(
    argv: Sequence[str] | str, *, cwd: Path | None = None, identifiers: Sequence[str] = ()
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return a bounded safe command, identifiers, and redaction caveats."""
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
            lambda match: match.group(1) + f"<userinfo:{_digest(match.group(2))}>", token
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
            token = auth_header.group(1) + f"<redacted:{_digest(auth_header.group(2))}>"
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
    command = shlex.join(result)
    if len(command.encode()) > MAX_SAFE_COMMAND_BYTES:
        command = _bounded(command, MAX_SAFE_COMMAND_BYTES)
        caveats.append("safe command truncated")
    safe_ids = tuple(_bounded(item, MAX_SAFE_IDENTIFIER_BYTES) for item in found_identifiers[:MAX_SAFE_IDENTIFIERS])
    return command, safe_ids, tuple(
        dict.fromkeys(_bounded(item, MAX_SAFE_CAVEAT_BYTES) for item in caveats)
    )


def redact_observation(observation: LocalTestObservation) -> LocalTestObservation:
    command, identifiers, command_caveats = redact_test_command(
        observation.command, cwd=Path(observation.cwd) if observation.cwd else None,
        identifiers=observation.identifiers,
    )
    return replace(
        observation,
        normalized_command=command,
        # Preserve argv boundaries so repeated durable projections are stable.
        command=tuple(shlex.split(command)),
        identifiers=identifiers,
        caveats=tuple(dict.fromkeys((*observation.caveats, *command_caveats))),
        diagnostic=None,
    )


def bounded_evidence_for_round(evidence: LocalTestEvidence | Mapping[str, object]) -> str:
    """Encode a bounded canonical metadata field, degrading before transport."""
    source = evidence.to_dict() if isinstance(evidence, LocalTestEvidence) else dict(evidence)
    raw_rows = source.get("observations")
    rows = list(raw_rows) if isinstance(raw_rows, list) else []
    all_details = [
        redact_observation(observation_from_mapping(row)).to_dict()
        for row in rows
        if isinstance(row, Mapping) and row.get("outcome") in OUTCOMES
    ]
    # Retain unresolved failures before any class of pass, then prefer newer
    # rows within a class. Restore chronological order for rendering.
    def retention_priority(row: Mapping[str, object]) -> int:
        if row.get("superseded_by"):
            return 0
        if row.get("outcome") in {
            "failed",
            "timed_out",
            "interrupted",
            "incomplete",
        }:
            return 4
        if row.get("outcome") != "passed":
            return 3
        attribution = row.get("attribution")
        state = attribution.get("state") if isinstance(attribution, Mapping) else "unknown"
        if state == "stale":
            return 1
        return 2

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
        "capture_incomplete": bool(source.get("capture_incomplete", False)),
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
            value["caveats"] = list(
                dict.fromkeys(
                    [
                        *value.get("caveats", []),
                        "local evidence details truncated before transport",
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
        parsed = decode_bounded_evidence(value)
        return bounded_evidence_for_round(parsed) if parsed is not None else None
    return None


def decode_bounded_evidence(value: object) -> LocalTestEvidence | None:
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
        )
        for row in rows
        if isinstance(row, Mapping)
        and row.get("outcome") in OUTCOMES
        and row.get("provenance") in PROVENANCES
    )
    return LocalTestEvidence(
        observations=observations,
        caveats=tuple(_safe_text(item) for item in payload.get("caveats", ()) if item is not None),
        capture_incomplete=bool(payload.get("capture_incomplete", False)),
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


def _run_git(root: Path, args: Sequence[str], *, timeout: float = 10.0) -> bytes:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *args),
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

        parsed = parse_managed_test_invocation(argv)
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
) -> tuple[str, int, bytes]:
    """Hash one file without reading beyond the aggregate byte/time budget."""
    name_size = len(name.encode("utf-8", errors="surrogateescape")) + 16
    before = path.stat(follow_symlinks=False)
    payload_size = len(prefix) + before.st_size
    required = name_size + payload_size
    if required > remaining[0]:
        raise AgentLoopError("git snapshot byte limit exceeded")
    if time.monotonic() - started > timeout_seconds:
        raise AgentLoopError("git snapshot time limit exceeded")
    remaining[0] -= required
    digest = hashlib.sha256(prefix)
    read_size = 0
    with path.open("rb") as stream:
        while True:
            if time.monotonic() - started > timeout_seconds:
                raise AgentLoopError("git snapshot time limit exceeded")
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
        top = _run_git(canonical_root, ("rev-parse", "--show-toplevel"), timeout=timeout_seconds)
        git_root = Path(top.decode("utf-8", errors="strict").strip()).resolve(strict=True)
        if git_root != canonical_root:
            raise AgentLoopError("snapshot root does not equal Git top-level")
        head = _run_git(canonical_root, ("rev-parse", "HEAD"), timeout=timeout_seconds).decode().strip()
        index_raw = _run_git(canonical_root, ("ls-files", "-s", "-z"), timeout=timeout_seconds)
        index_entries = _git_index_entries(index_raw)
        tracked = _git_nul_paths(_run_git(canonical_root, ("ls-files", "-z"), timeout=timeout_seconds))
        if set(index_entries) != set(tracked):
            raise AgentLoopError("git snapshot index and tracked paths disagree")
        untracked = _git_nul_paths(
            _run_git(canonical_root, ("ls-files", "--others", "--exclude-standard", "-z"), timeout=timeout_seconds)
        )
        if len(tracked) + len(untracked) > max_files:
            raise AgentLoopError("git snapshot file limit exceeded")
        remaining = [max_bytes]
        index_records = [
            _bytes_snapshot_record("HEAD", head.encode(), remaining=remaining),
            _bytes_snapshot_record("INDEX", index_raw, remaining=remaining),
        ]
        tracked_records: list[tuple[str, int, bytes]] = []
        for relative in tracked:
            if time.monotonic() - started > timeout_seconds:
                raise AgentLoopError("git snapshot time limit exceeded")
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
                        )
                    )
            except (OSError, UnicodeError) as exc:
                raise AgentLoopError("tracked file could not be read") from exc
        untracked_records: list[tuple[str, int, bytes]] = []
        for relative in untracked:
            if time.monotonic() - started > timeout_seconds:
                raise AgentLoopError("git snapshot time limit exceeded")
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
        )
        dirty_gitlinks = _dirty_gitlink_paths(tracked_status, index_entries)
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
            status_clean=not bool(tracked_status.strip()),
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
) -> TrackedTreeSnapshot:
    before = capture_tracked_tree_snapshot(
        root, argv=argv, max_files=max_files, max_bytes=max_bytes, timeout_seconds=timeout_seconds
    )
    if not before.complete:
        return before
    after = capture_tracked_tree_snapshot(
        root, argv=argv, max_files=max_files, max_bytes=max_bytes, timeout_seconds=timeout_seconds
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
BROKER_MAX_ARGV = 256
BROKER_MAX_ARGV_TOTAL = 32 * 1024
BROKER_MAX_ARG_BYTES = 8 * 1024
BROKER_MAX_CWD_BYTES = 4 * 1024
BROKER_MAX_ENV = 256
BROKER_MAX_ENV_TOTAL = 64 * 1024
BROKER_MAX_ENV_NAME_BYTES = 128
BROKER_MAX_ENV_VALUE_BYTES = 8 * 1024
BROKER_REQUEST_FIELDS = frozenset(
    {"turn_id", "nonce", "argv", "timeout_seconds", "cwd", "environment"}
)
BROKER_RESPONSE_FIELDS = frozenset(
    {"type", "receipt_id", "outcome", "returncode", "elapsed_seconds", "output_tail", "error"}
)


class BrokerProtocolError(AgentLoopError):
    """A malformed, unauthenticated, or bounded-out broker request."""


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


def _validate_broker_request(
    request: Mapping[str, object], *, turn_id: str, capability: str, root: Path, ceiling: float
) -> dict[str, object]:
    if set(request) != BROKER_REQUEST_FIELDS:
        raise BrokerProtocolError("broker request has unknown or missing fields")
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
    return {
        "turn_id": turn_id,
        "nonce": nonce,
        "argv": tuple(argv_values),
        "timeout_seconds": float(timeout),
        "cwd": canonical,
        "environment": dict(environment),
    }


@dataclass(frozen=True)
class BrokerRunResult:
    receipt_id: str
    outcome: str
    returncode: int | None
    elapsed_seconds: float
    output_tail: str = ""
    error: str | None = None
    wrapper_bootstrap: str = "unknown"
    inner_exec: str = "not-attempted"
    suite_start: str = "not-started"
    diagnostic: str = ""


@dataclass
class _ReplayReservation:
    digest: str
    ready: Event = field(default_factory=Event)
    response: dict[str, object] | None = None


class TestBrokerServer:
    """A single-turn AF_UNIX broker owned by the parent orchestrator."""

    def __init__(
        self,
        *,
        root: Path,
        turn_id: str | None = None,
        timeout_ceiling: float = 1800,
        containment_policy: object | None = None,
        execute: Any | None = None,
        environment_registry: EnvironmentIdentityRegistry | None = None,
    ) -> None:
        if root.is_symlink():
            raise BrokerProtocolError("broker root may not be a symlink")
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise BrokerProtocolError("broker root is not a directory")
        self.turn_id = turn_id or uuid.uuid4().hex
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
        self._journal: list[LocalTestObservation] = []
        # Reservations are retained for the broker lifetime. New work fails
        # closed at the bound instead of making an old authenticated nonce
        # executable again.
        self._receipts: dict[str, _ReplayReservation] = {}
        self._journal_lock = Lock()
        self._environment_registry = environment_registry or EnvironmentIdentityRegistry()
        self._parent_containment_handle: Any | None = None
        self._process_started: Any | None = None
        self._process_finished: Any | None = None
        self._active_processes: dict[int, Any] = {}
        self._pinned_root: Any | None = None

    def set_execution_context(
        self,
        *,
        containment_handle: Any | None,
        process_started: Any,
        process_finished: Any,
    ) -> None:
        """Bind broker children to the live requesting turn before requests run."""
        self._parent_containment_handle = containment_handle
        self._process_started = process_started
        self._process_finished = process_finished

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
        with self._journal_lock:
            active = list(self._active_processes.values())
        for proc in active:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=2)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
        if self._thread is not None and self._thread is not current_thread():
            self._thread.join(timeout=2)
        self._thread = None
        with self._journal_lock:
            handlers = list(self._handler_threads)
        for handler in handlers:
            if handler is not current_thread():
                handler.join(timeout=5)
        if self._runtime_dir is not None:
            try:
                for item in self._runtime_dir.iterdir():
                    item.unlink(missing_ok=True)
                self._runtime_dir.rmdir()
            except OSError:
                pass
            self._runtime_dir = None
        if self._pinned_root is not None:
            self._pinned_root.close()
            self._pinned_root = None

    def snapshot_journal(self) -> tuple[LocalTestObservation, ...]:
        return self.journal

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
                        reservation.ready.wait()
                        if reservation.response is None:
                            raise BrokerProtocolError("broker replay did not complete")
                        _send_frame(connection, reservation.response)
                        return
                    try:
                        response = self._execute_request(validated, connection)
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
                        with self._journal_lock:
                            reservation.response = response
                            reservation.ready.set()
                    _send_frame(connection, response)
                except (BrokerProtocolError, AgentLoopError, OSError, ValueError) as exc:
                    if (
                        isinstance(exc, AgentLoopError)
                        and not isinstance(exc, BrokerProtocolError)
                        and validated is not None
                    ):
                        self._record_capture_failure(validated, exc)
                    try:
                        _send_frame(connection, {"type": "error", "error": _safe_text(exc)})
                    except OSError:
                        pass
        finally:
            with self._journal_lock:
                self._handler_threads.discard(current_thread())

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

    def _append_journal_locked(self, observation: LocalTestObservation) -> None:
        """Append within the bound while retaining measured failures longest."""
        self._journal.append(observation)
        while len(self._journal) > MAX_PRIVATE_OBSERVATIONS:
            discard = next(
                (
                    index
                    for index, row in enumerate(self._journal)
                    if row.provenance == "telemetry-unverified"
                ),
                None,
            )
            if discard is None:
                discard = next(
                    (
                        index
                        for index, row in enumerate(self._journal)
                        if not row.is_failure
                    ),
                    0,
                )
            del self._journal[discard]

    def _execute_request(self, request: Mapping[str, object], connection: socket.socket) -> dict[str, object]:
        from .containment import open_confined_cwd

        argv = tuple(request["argv"])  # type: ignore[arg-type]
        requested_cwd = Path(request["cwd"])  # type: ignore[arg-type]
        environment = dict(request["environment"])  # type: ignore[arg-type]
        # The server, not the client, decides the actual target environment.
        # Control variables are never forwarded to the target.  The server
        # owns the authenticated turn identity that remains in its context.
        for name in (BROKER_ENDPOINT_ENV, BROKER_CAPABILITY_ENV, BROKER_PROTOCOL_ENV):
            environment.pop(name, None)
        environment["AGENT_LOOP_INVOCATION_ID"] = self.turn_id
        pinned_root = self._pinned_root
        if pinned_root is None:
            raise BrokerProtocolError("test broker root is not pinned")
        with open_confined_cwd(pinned_root, requested_cwd) as confined:
            cwd = confined.path
            before = stable_tracked_tree_snapshot(
                self.root,
                argv=argv,
                timeout_seconds=min(MAX_SNAPSHOT_SECONDS, self.timeout_ceiling),
            )

            def stream(chunk: str) -> None:
                safe = chunk.encode("utf-8", errors="replace")[:16 * 1024].decode("utf-8", errors="ignore")
                try:
                    with self._send_lock:
                        _send_frame(connection, {"type": "output", "data": safe})
                except OSError:
                    # The target remains bounded and the evidence result can
                    # still be returned; a disconnected client is a capture caveat.
                    pass

            if self._execute is not None:
                result = self._execute(argv, cwd, float(request["timeout_seconds"]), environment, stream)
            else:
                from .runner import run_foreground_test

                parent_cgroup = None
                handle = self._parent_containment_handle
                if handle is not None and getattr(handle, "managed", False):
                    report = handle.refresh_report()
                    if report is None or not report.target_started or handle.cgroup_path is None:
                        raise BrokerProtocolError("coder containment scope is not ready for broker tests")
                    parent_cgroup = handle.cgroup_path

                def started(proc: Any) -> None:
                    with self._journal_lock:
                        self._active_processes[proc.pid] = proc
                    if self._process_started is not None:
                        self._process_started(proc)

                def finished(proc: Any) -> None:
                    with self._journal_lock:
                        self._active_processes.pop(proc.pid, None)
                    if self._process_finished is not None:
                        self._process_finished(proc)

                result = run_foreground_test(
                    argv,
                    cwd=cwd,
                    cwd_fd=confined.fd,
                    timeout_seconds=float(request["timeout_seconds"]),
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
                )
            after = stable_tracked_tree_snapshot(
                self.root,
                argv=argv,
                timeout_seconds=min(MAX_SNAPSHOT_SECONDS, self.timeout_ceiling),
            )
        attribution = attribute_current_head(before, after, current_head=after.head, argv=argv)
        identity = self._environment_registry.capture(environment)
        from .test_runtime import normalize_test_command
        # Keep the shared registry private to the broker process; the bytes are
        # not present in the journal's public projection.
        receipt_id = uuid.uuid4().hex
        suite_start = str(getattr(result, "suite_start", "unknown"))
        if suite_start != "not-started" and str(getattr(result, "outcome", "")) != "overlap-rejected":
            observation = LocalTestObservation(
                command=argv,
                outcome=str(result.outcome),
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
                caveats=("output stream was broker-forwarded",),
                diagnostic=getattr(result, "output_tail", None),
                wrapper_bootstrap=str(getattr(result, "wrapper_bootstrap", "unknown")),
                inner_exec=str(getattr(result, "inner_exec", "not-attempted")),
                suite_start=suite_start,
            )
            with self._journal_lock:
                self._append_journal_locked(observation)
        return {
            "type": "result",
            "receipt_id": receipt_id,
            "outcome": str(result.outcome),
            "returncode": result.returncode,
            "elapsed_seconds": float(result.elapsed_seconds),
            "output_tail": _safe_text(getattr(result, "output_tail", ""), 8 * 1024),
            "wrapper_bootstrap": str(getattr(result, "wrapper_bootstrap", "unknown")),
            "inner_exec": str(getattr(result, "inner_exec", "not-attempted")),
            "suite_start": suite_start,
            "diagnostic": _safe_text(getattr(result, "diagnostic", ""), MAX_SAFE_CAVEAT_BYTES),
        }


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
        self, argv: Sequence[str], *, timeout_seconds: float, cwd: Path | None = None
    ) -> BrokerRunResult:
        if not self.available:
            raise BrokerProtocolError("test broker is unavailable")
        assert self.endpoint is not None and self.capability is not None and self.turn_id is not None
        snapshot_cwd = str(cwd or Path.cwd())
        snapshot_environment = dict(self._environment)
        # Exact protocol variables are transport-only. Do not wildcard-delete
        # test variables or virtual-environment/PATH state.
        for name in (BROKER_ENDPOINT_ENV, BROKER_CAPABILITY_ENV, BROKER_PROTOCOL_ENV):
            snapshot_environment.pop(name, None)
        snapshot_environment["AGENT_LOOP_INVOCATION_ID"] = self.turn_id
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
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(max(5.0, min(float(timeout_seconds) + 10.0, 1800.0)))
            connection.connect(self.endpoint)
            # The capability is an out-of-band secret carried by a protected
            # environment and turned into the exact request binding. It is
            # intentionally not part of the request schema or any response.
            request["turn_id"] = self.turn_id
            _send_frame(connection, request)
            while True:
                response = _recv_frame(connection)
                if response.get("type") == "output":
                    data = response.get("data")
                    if isinstance(data, str):
                        print(data, end="", flush=True)
                    continue
                if response.get("type") == "error":
                    raise BrokerProtocolError(str(response.get("error", "broker request failed")))
                if response.get("type") != "result":
                    raise BrokerProtocolError("unknown broker response")
                return BrokerRunResult(
                    receipt_id=str(response.get("receipt_id", "")),
                    outcome=str(response.get("outcome", "incomplete")),
                    returncode=(int(response["returncode"]) if isinstance(response.get("returncode"), int) else None),
                    elapsed_seconds=float(response.get("elapsed_seconds", 0.0)),
                    output_tail=str(response.get("output_tail", "")),
                    wrapper_bootstrap=str(response.get("wrapper_bootstrap", "unknown")),
                    inner_exec=str(response.get("inner_exec", "not-attempted")),
                    suite_start=str(response.get("suite_start", "not-started")),
                    diagnostic=str(response.get("diagnostic", "")),
                )


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
