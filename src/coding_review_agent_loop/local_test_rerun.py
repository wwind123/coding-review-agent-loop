"""Exact ``run-tests --rerun`` replay for unsuperseded failures (issue #1999).

The runner owns one invocation-scoped :class:`RerunCatalog`; every per-turn
test broker resolves ``--rerun <receipt-id>`` from it and replays the
recorded argv, cwd, fixture and environment.  The catalog is memory-only and
never projected, journaled into bounded evidence, logged, prompted or
persisted.  This module also renders the local-only coder-prompt block that
names each unsuperseded failure with its redacted command, its exact rerun
handle and the latest rejection reason.  Supersession authority is unchanged:
a replay supersedes a failure only through ``_can_supersede``.
"""

from __future__ import annotations

import os
import shlex
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Mapping, Sequence

from .local_test_evidence import (
    BROKER_FIXTURE_FIELD,
    BROKER_REFUSAL_DIAGNOSTIC_BYTES,
    BROKER_RERUN_FIELD,
    ENVIRONMENT_EXCLUSIONS,
    FIXTURE_MIN_TIMEOUT_SECONDS,
    MAX_PRIVATE_OBSERVATIONS,
    MAX_SAFE_IDENTIFIER_BYTES,
    _SAFE_RECEIPT_ID_RE,
    LocalTestEvidence,
    LocalTestObservation,
    _bounded,
    _cancelled_frame,
    _safe_text,
    redact_test_command,
)


RERUN_REFUSAL_PREFIX = "rerun refused"
RERUN_CATALOG_UNAVAILABLE = "rerun-catalog-unavailable"
RERUN_RECEIPT_UNKNOWN = "rerun-receipt-unknown"
RERUN_CWD_INVALID = "rerun-cwd-invalid"
RERUN_FIXTURE_TIMEOUT_BELOW_FLOOR = "rerun-fixture-timeout-below-floor"
_RERUN_REFUSAL_TEXT = {
    RERUN_CATALOG_UNAVAILABLE: "this broker has no rerun catalog",
    RERUN_RECEIPT_UNKNOWN: (
        "the receipt is not a parent-observed failure recorded in this checkout during this "
        "invocation (unknown, evicted, restored from an earlier invocation, passing, a launch "
        "failure or out of checkout)"
    ),
    RERUN_CWD_INVALID: "the recorded working directory is no longer valid in the checkout",
    RERUN_FIXTURE_TIMEOUT_BELOW_FLOOR: (
        f"the recorded fixture needs a timeout of at least {FIXTURE_MIN_TIMEOUT_SECONDS:g}s"
    ),
}
MAX_RERUN_CATALOG_ENTRIES = MAX_PRIVATE_OBSERVATIONS
# Broker-owned transport, turn and acknowledgement controls, plus every name
# excluded from the environment identity: a replay takes these from the
# current request, never from the recorded one.
_RERUN_REFRESHED_ENVIRONMENT = ENVIRONMENT_EXCLUSIONS | {"AGENT_LOOP_BROKER_CLIENT_ACKS"}


def rerun_refusal_text(code: str) -> str:
    """The fixed not-evidence refusal line for a closed-set rerun code."""
    reason = _RERUN_REFUSAL_TEXT.get(code, _RERUN_REFUSAL_TEXT[RERUN_RECEIPT_UNKNOWN])
    return f"{RERUN_REFUSAL_PREFIX}: {code} ({reason}); nothing was run (not evidence)"


@dataclass(frozen=True)
class RerunEntry:
    """One private replay recipe; every field except the receipt is secret."""

    receipt_id: str
    argv: tuple[str, ...] = field(repr=False)
    cwd_relative: str = field(repr=False)
    fixture: str | None = field(repr=False)
    environment: Mapping[str, str] = field(repr=False)
    root: str = field(repr=False)
    restart_command_digest: str | None = field(default=None, repr=False)


def _canonical_root(root: Path | str) -> str:
    try:
        return str(Path(root).resolve(strict=True))
    except (OSError, ValueError, RuntimeError):
        return ""


class RerunCatalog:
    """Invocation-scoped private replay catalog (issue #1999).

    Owned by the runner, so every per-turn broker of one invocation shares
    it, and bound to the canonical checkout root an entry was recorded
    under.  It holds the raw argv, cwd, fixture and validated request
    environment of parent-observed failures, in memory only: never
    projected, journaled into bounded evidence, logged, prompted or
    persisted.  Bounded to ``MAX_RERUN_CATALOG_ENTRIES``, oldest evicted.
    """

    def __init__(self, *, limit: int = MAX_RERUN_CATALOG_ENTRIES) -> None:
        self._lock = Lock()
        self._limit = max(1, int(limit))
        self._entries: "OrderedDict[str, RerunEntry]" = OrderedDict()

    def __repr__(self) -> str:
        return f"RerunCatalog(<{len(self)} private entries>)"

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def record(
        self,
        *,
        receipt_id: str,
        argv: Sequence[str],
        cwd: Path,
        root: Path,
        fixture: str | None,
        environment: Mapping[str, str],
        restart_command_digest: str | None = None,
    ) -> bool:
        if not isinstance(receipt_id, str) or not _SAFE_RECEIPT_ID_RE.fullmatch(receipt_id):
            return False
        canonical_root = _canonical_root(root)
        if not canonical_root:
            return False
        cwd_text = str(cwd)
        if cwd_text == canonical_root:
            relative = "."
        elif cwd_text.startswith(canonical_root + os.sep):
            relative = cwd_text[len(canonical_root) + 1 :]
        else:
            return False
        entry = RerunEntry(
            receipt_id=receipt_id,
            argv=tuple(str(item) for item in argv),
            cwd_relative=relative,
            fixture=str(fixture) if fixture is not None else None,
            environment=dict(environment),
            root=canonical_root,
            restart_command_digest=restart_command_digest,
        )
        with self._lock:
            self._entries.pop(receipt_id, None)
            self._entries[receipt_id] = entry
            while len(self._entries) > self._limit:
                self._entries.popitem(last=False)
        return True

    def resolve(self, receipt_id: object, root: Path | str) -> RerunEntry | None:
        if not isinstance(receipt_id, str):
            return None
        canonical_root = _canonical_root(root)
        with self._lock:
            entry = self._entries.get(receipt_id)
        if entry is None or not canonical_root or entry.root != canonical_root:
            return None
        return entry

    def contains(self, receipt_id: object, root: Path | str) -> bool:
        return self.resolve(receipt_id, root) is not None


def _rerun_cwd(root: Path, relative: str) -> Path | None:
    """Revalidate a recorded checkout-relative cwd with the request checks."""
    requested = root if relative == "." else root / relative
    try:
        canonical = requested.resolve(strict=True)
        canonical_root = root.resolve(strict=True)
        if (
            not canonical.is_dir()
            or os.path.commonpath((str(canonical_root), str(canonical))) != str(canonical_root)
            or requested.is_symlink()
        ):
            return None
    except (OSError, ValueError):
        return None
    return canonical


def resolve_rerun_request(
    validated: Mapping[str, object],
    *,
    catalog: "RerunCatalog | None",
    root: Path,
) -> tuple[dict[str, object] | None, str | None]:
    """Rebuild a validated rerun request from the private catalog (#1999).

    Returns ``(request, None)`` or ``(None, refusal code)``; a refusal runs
    nothing and records nothing.  Only the broker-owned controls and the
    names excluded from the environment identity come from the current
    request; the argv, cwd, fixture and environment are the recorded ones.
    """
    if catalog is None:
        return None, RERUN_CATALOG_UNAVAILABLE
    entry = catalog.resolve(validated.get(BROKER_RERUN_FIELD), root)
    if entry is None:
        return None, RERUN_RECEIPT_UNKNOWN
    cwd = _rerun_cwd(Path(entry.root), entry.cwd_relative)
    if cwd is None:
        return None, RERUN_CWD_INVALID
    timeout = float(validated["timeout_seconds"])  # type: ignore[arg-type]
    if entry.fixture is not None and timeout < FIXTURE_MIN_TIMEOUT_SECONDS:
        return None, RERUN_FIXTURE_TIMEOUT_BELOW_FLOOR
    current = dict(validated.get("environment") or {})  # type: ignore[call-overload]
    environment = {
        name: value for name, value in entry.environment.items()
        if name not in _RERUN_REFRESHED_ENVIRONMENT
    }
    environment.update(
        {name: value for name, value in current.items() if name in _RERUN_REFRESHED_ENVIRONMENT}
    )
    rebuilt: dict[str, object] = {
        "turn_id": validated["turn_id"],
        "nonce": validated["nonce"],
        "argv": tuple(entry.argv),
        "timeout_seconds": timeout,
        "cwd": cwd,
        "environment": environment,
        BROKER_RERUN_FIELD: entry.receipt_id,
    }
    if entry.fixture is not None:
        rebuilt[BROKER_FIXTURE_FIELD] = entry.fixture
    return rebuilt, None


def _rerun_refusal_frame(code: str) -> dict[str, object]:
    """A terminal no-test frame: the client never falls back to a local run."""
    return {
        **_cancelled_frame(),
        "returncode": 2,
        "inner_exec": "not-attempted",
        "diagnostic": _safe_text(rerun_refusal_text(code), BROKER_REFUSAL_DIAGNOSTIC_BYTES),
        "error_code": code,
    }


MAX_RERUN_BLOCK_DETAILED = 8
RERUN_HANDLE_FALLBACK_PREFIX = '"${AGENT_LOOP_EXECUTABLE:-agent-loop}" run-tests'
IDENTICAL_RERUN_RULE = (
    "Only an identical rerun clears a failure: it must pass, be parent-observed, run at the "
    "current head and tracked tree, and run in an equivalent environment. Broader, narrower or "
    "otherwise different selections never supersede it, however many tests they pass. Use the "
    "exact rerun handle, which replays the recorded command, fixture and environment."
)
NO_RERUN_HANDLE_REASON = (
    "no exact rerun handle in this invocation: raw argv and environment are never persisted, "
    "and this failure was recorded by an earlier invocation or was evicted from the bounded "
    "rerun catalog; fix the cause and ask an operator to verify it"
)


def unsuperseded_failure_rerun_block(
    evidence: LocalTestEvidence | None,
    catalog: RerunCatalog | None,
    root: Path | str | None,
    wrapper_prefix: Sequence[str] | None = None,
) -> str:
    """Local coder-prompt block for every unsuperseded failure (issue #1999).

    Built from the live reconciliation and the runner's private catalog; it
    is never posted.  It carries only receipt ids, redacted commands, exact
    ``--rerun`` handles and the latest rejection reason, never raw argv or
    environment values.  A handle is printed only when the catalog can
    resolve it for ``root``.
    """
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
    prefix = shlex.join(wrapper_prefix) if wrapper_prefix else RERUN_HANDLE_FALLBACK_PREFIX

    def handle(row: LocalTestObservation) -> str | None:
        receipt = row.receipt_id
        if (
            catalog is None or root is None or not receipt
            or not _SAFE_RECEIPT_ID_RE.fullmatch(receipt)
            or not catalog.contains(receipt, root)
        ):
            return None
        return f"{prefix} --rerun {receipt}"

    lines = [
        "## Unsuperseded local test failures (local prompt only; never post this block)",
        "",
        "The parent journal still holds these unsuperseded failure receipts. "
        + IDENTICAL_RERUN_RULE,
        "",
    ]
    for row in failing[:MAX_RERUN_BLOCK_DETAILED]:
        receipt = _safe_text(row.receipt_id or "(no receipt)", MAX_SAFE_IDENTIFIER_BYTES)
        command, _ids, _caveats = redact_test_command(row.command)
        lines.append(f"- Receipt `{receipt}` ({row.outcome}): `{_bounded(command, 240)}`")
        exact = handle(row)
        lines.append(
            f"  - Exact rerun: `{exact}`" if exact is not None
            else f"  - {NO_RERUN_HANDLE_REASON}."
        )
        rejection = evidence.latest_rejection_for(row.receipt_id)
        if rejection is not None:
            pass_receipt, reason, count = rejection
            lines.append(
                f"  - Latest rejected later pass `{_safe_text(pass_receipt, MAX_SAFE_IDENTIFIER_BYTES)}`: "
                f"{reason} ({count} later pass(es) rejected in total)."
            )
        else:
            lines.append("  - No later pass has been evaluated against it yet.")
    lines.extend(["", "Receipt-to-handle map (every unsuperseded failure):"])
    for row in failing:
        receipt = _safe_text(row.receipt_id or "(no receipt)", MAX_SAFE_IDENTIFIER_BYTES)
        exact = handle(row)
        lines.append(
            f"- `{receipt}` -> `{exact}`" if exact is not None
            else f"- `{receipt}` -> no exact rerun handle in this invocation"
        )
    lines.extend(
        [
            "",
            "Cite a replay as `agent-loop run-tests --rerun <receipt-id>` with the new receipt it "
            "prints. Never copy the redacted command back as a rerun: a redacted value is not "
            "the original argument.",
        ]
    )
    return "\n".join(lines)
