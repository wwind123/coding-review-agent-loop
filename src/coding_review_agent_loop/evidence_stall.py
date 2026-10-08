"""Pure evidence-only stall classification (#1324, stage 1 of #1317).

A PR can spend many rounds on evidence bookkeeping rather than code: every open
blocker is a request for admissible citation evidence the coder cannot capture,
and nothing else is wrong.  This module decides, from orchestrator-written
state only, whether a review round is *evidence-only* and whether K consecutive
such rounds with an unchanged unsatisfied-row set should stop for a human
decision.

Nothing here performs I/O and nothing is taken from agent prose.  Reviewer
tags (``evidence_row_ids``) are verified against the canonical risk-matrix
evidence the orchestrator persisted for the current head and the currently
approved matrix identity; a tag naming a satisfied row is an ordinary finding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .errors import AgentLoopError
from .protocol import (
    EVIDENCE_OBLIGATION_KIND,
    UnresolvedReviewItem,
    parse_risk_test_matrix_evidence,
)

# Closed set of reasons a round does not qualify.
STALL_REASONS = frozenset(
    {
        "no-applicable-matrix",
        "evidence-head-unbound",
        "evidence-absent",
        "evidence-matrix-mismatch",
        "no-unsatisfied-rows",
        "no-open-items",
        "untagged-finding",
        "tag-names-satisfied-row",
        "machine-obligation-open",
        "checks-failing",
        "checks-unavailable",
        "evidence-freeze-active",
    }
)

CANONICAL_STATUS_PREFIX = "canonical-status:"


@dataclass(frozen=True)
class CheckBoardSummary:
    """Names of failing and infrastructure-stalled checks on the review head."""

    failing: tuple[str, ...] = ()
    infrastructure_stalls: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnsatisfiedRows:
    """Result of deriving the unsatisfied-row set from canonical evidence.

    ``reason`` is ``evidence-absent`` or ``evidence-matrix-mismatch`` when the
    evidence cannot be used; ``rows`` is then empty and must not be read as
    "every row unverified".
    """

    rows: Mapping[str, str] = field(default_factory=dict)
    reason: str | None = None

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.rows))


@dataclass(frozen=True)
class StallRoundSnapshot:
    review_round: int
    review_head: str
    matrix_identity: str
    qualifies: bool
    unsatisfied_row_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "review_round": self.review_round,
            "review_head": self.review_head,
            "matrix_identity": self.matrix_identity,
            "qualifies": self.qualifies,
            "unsatisfied_row_ids": sorted(self.unsatisfied_row_ids),
            "reasons": sorted(self.reasons),
        }


@dataclass(frozen=True)
class StallWindow:
    stop: bool
    length: int


def unsatisfied_rows(
    evidence: object,
    diagnostics: Iterable[Mapping[str, object]] | None,
    enforceable_row_ids: Sequence[str],
    approved_identity: str,
) -> UnsatisfiedRows:
    """Derive ``row_id -> explanation`` for enforceable rows that are not verified.

    ``evidence`` is the persisted ``risk_test_matrix_evidence`` payload of the
    coder record bound to the current head, or ``None`` when that record has no
    matrix evidence.  Every explanation starts with the row's own canonical
    status code and then lists the sorted row-specific diagnostic codes, so no
    unsatisfied row has an empty explanation and no code is ever invented.
    """
    if evidence is None:
        return UnsatisfiedRows(reason="evidence-absent")
    try:
        parsed = parse_risk_test_matrix_evidence(
            evidence,
            expected_identity=approved_identity,
            expected_row_ids=tuple(enforceable_row_ids),
        )
    except AgentLoopError:
        return UnsatisfiedRows(reason="evidence-matrix-mismatch")
    codes: dict[str, set[str]] = {}
    for diagnostic in diagnostics or ():
        row_id = diagnostic.get("row_id") if isinstance(diagnostic, Mapping) else None
        code = diagnostic.get("code") if isinstance(diagnostic, Mapping) else None
        if isinstance(row_id, str) and isinstance(code, str) and code.strip():
            codes.setdefault(row_id, set()).add(code.strip())
    enforceable = set(enforceable_row_ids)
    rows: dict[str, str] = {}
    for row in sorted(parsed.rows, key=lambda r: r.row_id):
        if row.row_id not in enforceable or row.status == "verified":
            continue
        explanation = [f"{CANONICAL_STATUS_PREFIX}{row.status}", *sorted(codes.get(row.row_id, ()))]
        rows[row.row_id] = ", ".join(explanation)
    return UnsatisfiedRows(rows=rows)


def classify_round(
    *,
    review_round: int,
    review_head: str,
    approved_identity: str | None,
    open_items: Sequence[UnresolvedReviewItem],
    unsatisfied: UnsatisfiedRows | None,
    checks: CheckBoardSummary | None,
    freeze_active: bool,
) -> StallRoundSnapshot:
    """Classify one review round; it qualifies only when no reason applies.

    ``unsatisfied`` is ``None`` when no coder record is bound to the current
    head.  ``open_items`` is the reconciled ledger after CI reconciliation;
    only blocking and same-PR items are mandatory.
    """
    reasons: set[str] = set()
    row_ids: tuple[str, ...] = ()
    if approved_identity is None:
        reasons.add("no-applicable-matrix")
    elif unsatisfied is None:
        reasons.add("evidence-head-unbound")
    elif unsatisfied.reason is not None:
        reasons.add(unsatisfied.reason)
    elif not unsatisfied.rows:
        reasons.add("no-unsatisfied-rows")
    else:
        row_ids = unsatisfied.row_ids
    usable = set(row_ids)
    mandatory = [item for item in open_items if item.status in {"blocking", "same-pr"}]
    reviewer_items = 0
    for item in mandatory:
        if item.is_machine_obligation:
            if item.obligation_kind != EVIDENCE_OBLIGATION_KIND:
                reasons.add("machine-obligation-open")
            continue
        reviewer_items += 1
        if not item.evidence_row_ids:
            reasons.add("untagged-finding")
        elif not set(item.evidence_row_ids) <= usable:
            reasons.add("tag-names-satisfied-row")
    if reviewer_items == 0:
        reasons.add("no-open-items")
    if checks is None:
        reasons.add("checks-unavailable")
    elif checks.failing or checks.infrastructure_stalls:
        reasons.add("checks-failing")
    if freeze_active:
        reasons.add("evidence-freeze-active")
    return StallRoundSnapshot(
        review_round=review_round,
        review_head=review_head,
        matrix_identity=approved_identity or "",
        qualifies=not reasons,
        unsatisfied_row_ids=row_ids,
        reasons=tuple(sorted(reasons)),
    )


def stall_snapshot_is_consistent(snapshot: Mapping[str, object]) -> bool:
    """Whether a persisted stall snapshot is internally consistent (#1324).

    ``qualifies`` must hold exactly when ``reasons`` is empty, every reason
    must belong to the closed :data:`STALL_REASONS` set, row IDs and reasons
    must be unique strings, and a qualifying snapshot must name its head,
    matrix identity, and a non-empty unsatisfied-row set.  Anything else is
    corrupt history and counts as non-qualifying.
    """
    qualifies = snapshot.get("qualifies")
    reasons = snapshot.get("reasons")
    row_ids = snapshot.get("unsatisfied_row_ids")
    if not isinstance(qualifies, bool) or not isinstance(reasons, (list, tuple)):
        return False
    if not isinstance(row_ids, (list, tuple)):
        return False
    if any(not isinstance(reason, str) or reason not in STALL_REASONS for reason in reasons):
        return False
    if any(not isinstance(row_id, str) or not row_id.strip() for row_id in row_ids):
        return False
    if len(set(reasons)) != len(reasons) or len(set(row_ids)) != len(row_ids):
        return False
    if qualifies != (not reasons):
        return False
    if qualifies:
        head = snapshot.get("review_head")
        identity = snapshot.get("matrix_identity")
        if not row_ids or not isinstance(head, str) or not head.strip():
            return False
        if not isinstance(identity, str) or not identity.strip():
            return False
    return True


def stall_window(
    current: StallRoundSnapshot,
    prior_snapshots_by_review_round: Mapping[int, Mapping[str, object] | None],
    k: int,
) -> StallWindow:
    """Count consecutive qualifying rounds ending at ``current``.

    The walk ends at a missing, invalid, inconsistent, or non-qualifying
    snapshot (see :func:`stall_snapshot_is_consistent`), or one
    whose row set or matrix identity differs from the current snapshot's.  With
    ``k <= 0`` the detector never stops.
    """
    if not current.qualifies:
        return StallWindow(stop=False, length=0)
    length = 1
    review_round = current.review_round - 1
    expected_rows = sorted(current.unsatisfied_row_ids)
    while review_round >= 1:
        snapshot = prior_snapshots_by_review_round.get(review_round)
        if (
            not isinstance(snapshot, Mapping)
            or snapshot.get("invalid")
            or not stall_snapshot_is_consistent(snapshot)
            or snapshot.get("review_round") != review_round
            or snapshot.get("qualifies") is not True
            or snapshot.get("matrix_identity") != current.matrix_identity
            or sorted(snapshot.get("unsatisfied_row_ids") or ()) != expected_rows
        ):
            break
        length += 1
        review_round -= 1
    return StallWindow(stop=k > 0 and length >= k, length=length)


def approved_matrix_row_ids(approved_plan_context: object | None) -> tuple[str, ...] | None:
    """Enforceable approved-matrix row IDs, or ``None`` without an applicable matrix."""
    if approved_plan_context is None or not getattr(approved_plan_context, "matrix_available", False):
        return None
    row_ids = getattr(approved_plan_context, "risk_test_matrix_expected_row_ids", None)
    return tuple(row_ids) if row_ids else None
