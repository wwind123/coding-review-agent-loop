"""Pure risk-matrix evidence conversion (#1325, stage 2 of #1317).

A risk-matrix row whose evidence the coder demonstrably cannot capture is
converted into a ``human-exact-head-evidence`` obligation and answered through
the existing exact-head evidence freeze.  This module decides, from
orchestrator-owned state only:

* the per-attempt capture classification persisted on coder records, built by
  ``protocol.derive_risk_test_matrix_evidence`` from structured observation
  fields and never from diagnostic codes;
* which rows convert automatically after N consecutive capture-limited coder
  attempts, each paired with the review snapshot that judged it;
* whether an operator's ``--pr-convert-evidence-rows`` list is acceptable;
* the conversion audit record that rebuilds the conversion after a restart;
* live-head reconciliation of converted obligations (reopen, restore, or
  supersede by tool-captured evidence).

Nothing here performs I/O.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from .errors import AgentLoopError
from .evidence_stall import StallRoundSnapshot, stall_snapshot_is_consistent
from .protocol_risk import (
    CAPTURE_DISQUALIFIERS,
    CAPTURE_LIMITED,
    CAPTURE_OBSERVATION_CLASSES,
    NOT_CAPTURE_LIMITED,
    DerivedRiskEvidenceResult,
    SignedEvidenceReference,
    disqualify_capture_classification,
)
from .protocol import (
    EVIDENCE_OBLIGATION_KIND,
    MACHINE_AUTHORITY,
    UnresolvedReviewItem,
)
from .unresolved_items import (
    converted_evidence_obligation_identity,
    converted_evidence_obligation_text,
)

EVIDENCE_CONVERSION_SOURCES = frozenset({"auto", "operator"})
_SHA_RE = re.compile(r"[^\s`]{1,128}")
_IDENTITY_RE = re.compile(r"[0-9a-f]{64}")
_ITEM_ID_RE = re.compile(r"item-[1-9][0-9]*")


# ---------------------------------------------------------------------------
# Per-attempt capture classification


def capture_classification_payload(
    derived: DerivedRiskEvidenceResult | None,
    *,
    head_sha: str | None,
) -> dict[str, object] | None:
    """The replayable per-attempt record for one coder record's evidence."""
    if derived is None or not head_sha or not derived.row_capture_classifications:
        return None
    return {
        "version": 1,
        "matrix_identity": derived.evidence.matrix_identity,
        "head": head_sha,
        "rows": {
            row_id: {
                "class": entry["class"],
                "observation_classes": list(entry["observation_classes"]),  # type: ignore[arg-type]
                "disqualifiers": list(entry["disqualifiers"]),  # type: ignore[arg-type]
            }
            for row_id, entry in sorted(derived.row_capture_classifications.items())
        },
    }


def decode_capture_classification(
    raw: object,
    *,
    subject: str | None,
    evidence_identity: str | None,
) -> Mapping[str, object] | None:
    """Strictly decode a persisted classification; anything else is ``None``.

    The record must be bound to its own coder record: ``head`` equals the
    record subject and ``matrix_identity`` equals the identity of the
    record's own canonical evidence.  ``None`` counts as a non-qualifying
    attempt and never as capture-limited.
    """
    if not isinstance(raw, Mapping) or raw.get("version") != 1:
        return None
    if set(raw) != {"version", "matrix_identity", "head", "rows"}:
        return None
    head = raw.get("head")
    identity = raw.get("matrix_identity")
    rows = raw.get("rows")
    if (
        not isinstance(head, str)
        or not _SHA_RE.fullmatch(head)
        or head != subject
        or not isinstance(identity, str)
        or not _IDENTITY_RE.fullmatch(identity)
        or identity != evidence_identity
        or not isinstance(rows, Mapping)
        or not rows
    ):
        return None
    decoded: dict[str, dict[str, object]] = {}
    for row_id, entry in rows.items():
        if not isinstance(row_id, str) or not row_id.strip() or not isinstance(entry, Mapping):
            return None
        if set(entry) != {"class", "observation_classes", "disqualifiers"}:
            return None
        row_class = entry.get("class")
        classes = entry.get("observation_classes")
        disqualifiers = entry.get("disqualifiers")
        if not isinstance(row_class, str) or row_class not in {CAPTURE_LIMITED, NOT_CAPTURE_LIMITED}:
            return None
        if not _closed_string_list(classes, CAPTURE_OBSERVATION_CLASSES):
            return None
        if not _closed_string_list(disqualifiers, CAPTURE_DISQUALIFIERS):
            return None
        # Capture-limited holds exactly when no disqualifier applies.
        if (row_class == CAPTURE_LIMITED) != (not disqualifiers):
            return None
        decoded[row_id] = {
            "class": row_class,
            "observation_classes": list(classes),
            "disqualifiers": list(disqualifiers),
        }
    return {"version": 1, "matrix_identity": identity, "head": head, "rows": decoded}


def _closed_string_list(value: object, vocabulary: frozenset[str]) -> bool:
    """A sorted, duplicate-free list of strings from ``vocabulary``; never raises."""
    return (
        isinstance(value, list)
        and all(isinstance(entry, str) and entry in vocabulary for entry in value)
        and value == sorted(set(value))
    )


def attempt_counts_for_row(
    classification: Mapping[str, object] | None,
    row_id: str,
    *,
    matrix_identity: str,
    head_sha: str,
) -> bool:
    """Whether the bound record's classification makes ``row_id`` capture-limited."""
    if not isinstance(classification, Mapping):
        return False
    if classification.get("matrix_identity") != matrix_identity or classification.get("head") != head_sha:
        return False
    rows = classification.get("rows")
    entry = rows.get(row_id) if isinstance(rows, Mapping) else None
    return isinstance(entry, Mapping) and entry.get("class") == CAPTURE_LIMITED


# ---------------------------------------------------------------------------
# Automatic conversion decision


def auto_conversion_rows(
    current_snapshot: StallRoundSnapshot,
    prior_snapshots_by_review_round: Mapping[int, Mapping[str, object] | None],
    classifications_by_record_round: Mapping[int, Mapping[str, object] | None],
    n: int,
    matrix_identity: str,
    already_converted: Iterable[str] = (),
) -> dict[str, tuple[int, ...]]:
    """Rows that convert now, with the coder record rounds that counted.

    The walk mirrors :func:`evidence_stall.stall_window` backward from the
    current review: it stops at the first missing, invalid, non-qualifying,
    attempt-unknown, or other-matrix snapshot, and at a repeated bound attempt
    (the same coder record judged twice).  A row converts only when each of
    the first ``n`` distinct attempts counts for it: the pairing review
    qualified, the row was in that review's unsatisfied set, and the bound
    record's classification is capture-limited for the same matrix and head.
    An earlier verified, absent, or not-capture-limited attempt ends the run,
    so the count restarts.  ``n <= 0`` disables automatic conversion.
    """
    if n <= 0:
        return {}
    if (
        not current_snapshot.qualifies
        or current_snapshot.bound_attempt_round is None
        or current_snapshot.matrix_identity != matrix_identity
    ):
        return {}
    walk: list[tuple[str, int, frozenset[str]]] = [
        (
            current_snapshot.review_head,
            current_snapshot.bound_attempt_round,
            frozenset(current_snapshot.unsatisfied_row_ids),
        )
    ]
    seen = {current_snapshot.bound_attempt_round}
    review_round = current_snapshot.review_round - 1
    while len(walk) < n and review_round >= 1:
        snapshot = prior_snapshots_by_review_round.get(review_round)
        if (
            not isinstance(snapshot, Mapping)
            or snapshot.get("invalid")
            or not stall_snapshot_is_consistent(snapshot)
            or snapshot.get("review_round") != review_round
            or snapshot.get("qualifies") is not True
            or snapshot.get("matrix_identity") != matrix_identity
        ):
            break
        bound = snapshot.get("bound_attempt_round")
        if isinstance(bound, bool) or not isinstance(bound, int) or bound in seen:
            break
        seen.add(bound)
        walk.append(
            (
                str(snapshot.get("review_head")),
                bound,
                frozenset(snapshot.get("unsatisfied_row_ids") or ()),  # type: ignore[arg-type]
            )
        )
        review_round -= 1
    if len(walk) < n:
        return {}
    skip = set(already_converted)
    result: dict[str, tuple[int, ...]] = {}
    for row_id in sorted(current_snapshot.unsatisfied_row_ids):
        if row_id in skip:
            continue
        rounds: list[int] = []
        for head_sha, bound, rows in walk[:n]:
            if row_id not in rows or not attempt_counts_for_row(
                classifications_by_record_round.get(bound),
                row_id,
                matrix_identity=matrix_identity,
                head_sha=head_sha,
            ):
                break
            rounds.append(bound)
        if len(rounds) == n:
            result[row_id] = tuple(sorted(rounds))
    return result


def validate_operator_conversion_rows(
    requested: Sequence[str],
    *,
    unsatisfied_row_ids: Sequence[str] | None,
    already_converted: Iterable[str] = (),
    reason: str | None = None,
) -> tuple[str, ...]:
    """All-or-nothing validation of ``--pr-convert-evidence-rows`` (#1325).

    Every named row must be in the current unsatisfied set minus rows
    already converted under the same matrix identity (pending or cleared).
    Any other ID, a duplicate, or an unavailable set raises before any agent
    runs, naming the rejected IDs, and nothing is converted.
    """
    names = [str(row).strip() for row in requested]
    if not names or any(not name for name in names):
        raise AgentLoopError("--pr-convert-evidence-rows needs at least one non-empty row ID.")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise AgentLoopError(
            "--pr-convert-evidence-rows names a row more than once: " + ", ".join(duplicates)
            + ". Nothing was converted and no agent was run."
        )
    if unsatisfied_row_ids is None:
        raise AgentLoopError(
            "--pr-convert-evidence-rows cannot be applied: "
            + (reason or "no unsatisfied risk-matrix row set is bound to the current PR head")
            + f". Rejected row(s): {', '.join(names)}. Nothing was converted and no agent was run."
        )
    allowed = set(unsatisfied_row_ids) - set(already_converted)
    rejected = [name for name in names if name not in allowed]
    if rejected:
        raise AgentLoopError(
            "--pr-convert-evidence-rows accepts only rows that are currently unsatisfied at the "
            f"live head and not already converted; rejected: {', '.join(rejected)}. Currently "
            "convertible: "
            + (", ".join(sorted(allowed)) or "(none)")
            + ". Nothing was converted and no agent was run."
        )
    return tuple(names)


# ---------------------------------------------------------------------------
# Conversion audit record


@dataclass(frozen=True)
class ConversionAuditRow:
    row_id: str
    obligation_identity: str
    obligation_item_id: str
    owners: tuple[str, ...]
    converted_item_ids: tuple[str, ...]
    attempt_record_rounds: tuple[int, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "obligation_identity": self.obligation_identity,
            "obligation_item_id": self.obligation_item_id,
            "owners": list(self.owners),
            "converted_item_ids": list(self.converted_item_ids),
            "attempt_record_rounds": list(self.attempt_record_rounds),
        }


@dataclass(frozen=True)
class EvidenceConversionAudit:
    """Orchestrator audit of one conversion, posted once before any dispatch."""

    source: str
    matrix_identity: str
    head: str
    rows: tuple[ConversionAuditRow, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "source": self.source,
            "matrix_identity": self.matrix_identity,
            "head": self.head,
            "rows": [row.to_payload() for row in self.rows],
        }

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(row.row_id for row in self.rows)

    @classmethod
    def from_payload(cls, value: object) -> "EvidenceConversionAudit | None":
        """Strict decode; an invalid record is ignored (treated as unconverted)."""
        if not isinstance(value, Mapping) or value.get("version") != 1:
            return None
        if set(value) != {"version", "source", "matrix_identity", "head", "rows"}:
            return None
        source = value.get("source")
        identity = value.get("matrix_identity")
        head = value.get("head")
        raw_rows = value.get("rows")
        if (
            not isinstance(source, str)
            or source not in EVIDENCE_CONVERSION_SOURCES
            or not isinstance(identity, str)
            or not _IDENTITY_RE.fullmatch(identity)
            or not isinstance(head, str)
            or not _SHA_RE.fullmatch(head)
            or not isinstance(raw_rows, list)
            or not raw_rows
        ):
            return None
        rows: list[ConversionAuditRow] = []
        for raw in raw_rows:
            if not isinstance(raw, Mapping) or set(raw) != {
                "row_id", "obligation_identity", "obligation_item_id", "owners",
                "converted_item_ids", "attempt_record_rounds",
            }:
                return None
            row_id = raw.get("row_id")
            item_id = raw.get("obligation_item_id")
            owners = raw.get("owners")
            converted = raw.get("converted_item_ids")
            attempts = raw.get("attempt_record_rounds")
            if (
                not isinstance(row_id, str)
                or not row_id.strip()
                or raw.get("obligation_identity")
                != converted_evidence_obligation_identity(row_id, identity)
                or not isinstance(item_id, str)
                or not _ITEM_ID_RE.fullmatch(item_id)
                or not isinstance(owners, list)
                or not owners
                or any(not isinstance(owner, str) or not owner.strip() for owner in owners)
                or len(set(owners)) != len(owners)
                or not isinstance(converted, list)
                or any(not isinstance(entry, str) or not _ITEM_ID_RE.fullmatch(entry) for entry in converted)
                or not isinstance(attempts, list)
                or any(isinstance(entry, bool) or not isinstance(entry, int) or entry < 1 for entry in attempts)
                or (source == "auto" and not attempts)
            ):
                return None
            rows.append(
                ConversionAuditRow(
                    row_id=row_id,
                    obligation_identity=str(raw["obligation_identity"]),
                    obligation_item_id=item_id,
                    owners=tuple(owners),
                    converted_item_ids=tuple(converted),
                    attempt_record_rounds=tuple(attempts),
                )
            )
        if len({row.row_id for row in rows}) != len(rows):
            return None
        return cls(source=str(source), matrix_identity=identity, head=head, rows=tuple(rows))


def converted_rows(
    unresolved_items: Sequence[UnresolvedReviewItem], *, matrix_identity: str
) -> frozenset[str]:
    """Rows with a row-converted obligation (pending or tombstoned) for this matrix."""
    return frozenset(
        item.converted_row_id
        for item in unresolved_items
        if item.converted_row_id is not None
        and item.converted_matrix_identity == matrix_identity
    )


# ---------------------------------------------------------------------------
# Live-head reconciliation


def _reopen(item: UnresolvedReviewItem) -> UnresolvedReviewItem:
    return replace(
        item,
        status="blocking",
        source_status="blocking",
        lifecycle="evidence_deferred",
        candidate_head_sha=None,
        owner_states=tuple((owner, "pending") for owner in item.resolution_owners),
    )


def _tombstone(item: UnresolvedReviewItem, *, head_sha: str, note: str) -> UnresolvedReviewItem:
    notes = item.notes if note in item.notes else (*item.notes, note)
    return replace(
        item,
        status="resolved",
        lifecycle="cleared",
        candidate_head_sha=head_sha,
        owner_states=tuple((owner, "cleared") for owner in item.resolution_owners),
        notes=notes,
    )


def tool_superseded_note(head_sha: str) -> str:
    return f"Superseded by tool-captured evidence verified at head {head_sha}."


def _drop_unbacked_conversions(
    ledger: Sequence[UnresolvedReviewItem],
    admitted_audits: Sequence[EvidenceConversionAudit],
    *,
    matrix_identity: str,
) -> list[UnresolvedReviewItem]:
    """Conversion state counts only when an admitted audit records it (#1325).

    A converted obligation whose (identity, item ID) no admitted audit names
    is dropped, and a finding's converted rows not named by an admitted audit
    for the current matrix return to its residual demand (a `converted`
    finding with residual rows reopens as blocking).  So a ledger adopted from
    a record with a rejected or foreign audit grants no conversion.
    """
    backed_obligations = {
        (row.obligation_identity, row.obligation_item_id)
        for audit in admitted_audits
        for row in audit.rows
    }
    backed_rows = {
        row.row_id
        for audit in admitted_audits
        if audit.matrix_identity == matrix_identity
        for row in audit.rows
    }
    result: list[UnresolvedReviewItem] = []
    for item in ledger:
        if item.converted_row_id is not None:
            if (item.obligation_identity, item.item_id) in backed_obligations:
                result.append(item)
            continue
        if item.converted_row_ids:
            kept = tuple(row for row in item.converted_row_ids if row in backed_rows)
            if kept != item.converted_row_ids:
                status = item.status
                if status == "converted" and set(kept) != set(item.evidence_row_ids):
                    status = "blocking"
                item = replace(item, converted_row_ids=kept, status=status)
        result.append(item)
    return result


def reconcile_converted_obligations(
    ledger: Sequence[UnresolvedReviewItem],
    *,
    live_head: str | None,
    matrix_identity: str | None,
    admitted_audits: Sequence[EvidenceConversionAudit] = (),
    admitted_references: Sequence[SignedEvidenceReference] = (),
    tool_verified_rows_at_live_head: Iterable[str] = (),
    drop_unbacked: bool = True,
) -> list[UnresolvedReviewItem]:
    """Bind every row-converted obligation to the live head (#1325).

    0. Drop conversion state no admitted audit backs.
    1. Restore an obligation recorded in an admitted audit but missing from
       the ledger, with the audit's item ID and owners.
    2. A converted obligation is satisfied at the live head only by an
       admitted reference for (row, matrix identity, live head) or by admitted
       tool-captured evidence bound to the live head that verifies the row
       (tombstoned as tool-superseded).  Any other tombstone, or a freeze at
       another head, reopens in place as blocking/deferred with the same item
       ID, owners and identity.  A ledger tombstone alone never satisfies a row.
    3. References are left untouched; the overlay ignores other heads.

    Pure and idempotent, so a resume replays it identically.
    """
    if not live_head or not matrix_identity:
        return list(ledger)
    # Unbacked conversion state is dropped only when the authority history
    # was fully read; unavailable history keeps the established barrier.
    result = (
        _drop_unbacked_conversions(ledger, admitted_audits, matrix_identity=matrix_identity)
        if drop_unbacked
        else list(ledger)
    )
    present_identities = {
        item.obligation_identity for item in result if item.obligation_identity
    }
    present_ids = {item.item_id for item in result}
    for audit in admitted_audits:
        if audit.matrix_identity != matrix_identity:
            continue
        for row in audit.rows:
            if row.obligation_identity in present_identities or row.obligation_item_id in present_ids:
                continue
            result.append(
                UnresolvedReviewItem(
                    item_id=row.obligation_item_id,
                    reviewer="Orchestrator",
                    source_round=1,
                    text=converted_evidence_obligation_text(
                        row_id=row.row_id,
                        matrix_identity=matrix_identity,
                        head_sha=audit.head,
                        explanation="restored from the conversion audit record",
                    ),
                    status="blocking",
                    source_status="blocking",
                    resolution_owners=row.owners,
                    owner_states=tuple((owner, "pending") for owner in row.owners),
                    authority=MACHINE_AUTHORITY,
                    obligation_kind=EVIDENCE_OBLIGATION_KIND,
                    lifecycle="evidence_deferred",
                    obligation_identity=row.obligation_identity,
                    converted_row_id=row.row_id,
                    converted_matrix_identity=matrix_identity,
                )
            )
            present_identities.add(row.obligation_identity)
            present_ids.add(row.obligation_item_id)
    signed = {
        (reference.row_id, reference.matrix_identity, reference.head)
        for reference in admitted_references
    }
    tool_verified = set(tool_verified_rows_at_live_head)
    for index, item in enumerate(result):
        if item.converted_row_id is None or item.converted_matrix_identity != matrix_identity:
            continue
        row_id = item.converted_row_id
        signed_here = (row_id, matrix_identity, live_head) in signed
        tool_here = row_id in tool_verified
        if item.lifecycle == "evidence_frozen":
            if item.candidate_head_sha != live_head:
                result[index] = _reopen(item)
            continue
        if item.lifecycle == "cleared" and item.candidate_head_sha == live_head and (
            signed_here or tool_here
        ):
            continue
        if tool_here:
            result[index] = _tombstone(item, head_sha=live_head, note=tool_superseded_note(live_head))
        elif signed_here:
            result[index] = _tombstone(
                item,
                head_sha=live_head,
                note=f"Cleared at head {live_head} by an admitted signed-evidence reference.",
            )
        elif item.lifecycle == "cleared" or item.status not in {"blocking", "same-pr"}:
            result[index] = _reopen(item)
    return result
