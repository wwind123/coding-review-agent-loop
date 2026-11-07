"""Round metadata persistence and resume helpers."""

from __future__ import annotations
from . import round_metadata
from .qualification_checkpoint import (
    _latest_qualification_checkpoint,
    _latest_qualification_checkpoint_record,
)
from .review_scope import authority_records

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .review_scope import is_complete_code_review, scope_present
from .review_approval import _contradictory_evidence_approval
from .agents.base import AgentName
from .agents.registry import agent_display_name, agent_signature
from .errors import AgentLoopError
from .round_transport import ROUND_RESUME_MARKER_RE, decode_mapping, hydrate_mapping
from .comment_rendering import (
    EXECUTION_RECOMMENDATION_MARKER_RE,
    decode_execution_recommendation_marker,
    RISK_TEST_MATRIX_MARKER_RE,
    decode_risk_test_matrix_marker,
)
from .workdir_guard import validate_checkout_inspected_evidence
from .protocol_risk import (
    parse_risk_test_matrix,
    parse_risk_test_matrix_changes,
    risk_test_matrix_identity,
)
from .protocol import (
    ReviewItemDisposition,
    UnresolvedReviewItem,
    EVIDENCE_OBLIGATION_KIND,
    UNKNOWN_MACHINE_AUTHORITY,
    failed_discuss_review_placeholder,
    failed_discuss_answer_placeholder,
    parse_plan_revision_patch,
)
from .protocol_common import HTML_COMMENT_RE, SIGNATURE_RE
from .protocol_discuss_types import (
    ParsedDiscussAgenda,
    ParsedDiscussFinalSynthesis,
    ParsedDiscussRoundSynthesis,
    ParsedDiscussResponse,
)
from .protocol_discuss import (
    parse_structured_discuss_agenda,
    parse_canonical_discuss_final_synthesis,
    parse_canonical_discuss_round_synthesis,
    parse_structured_discuss_review,
    parse_structured_discuss_answer,
    parse_legacy_structured_discuss_answer,
)
from .plan_assembly import decode_assembled_plan_sidecar, rendered_plan_identity
from .unresolved_items import (
    _apply_unresolved_item_dispositions,
    _is_machine_obligation,
    release_evidence_freeze,
)


logger = logging.getLogger(__name__)


from .legacy_machine_items import (
    _promote_legacy_machine_items,
    _demote_plan_machine_items,
    recover_pr_acknowledgement_aliases,
)


def _extract_round_metadata_records(comments: Sequence[object], *, flow: str) -> tuple[round_metadata.PostedRoundRecord, ...]:
    records: list[round_metadata.PostedRoundRecord] = []
    bodies = tuple(body for comment in comments if isinstance((body := getattr(comment, "body", None)), str))
    for index, comment in enumerate(comments):
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        matches = list(ROUND_RESUME_MARKER_RE.finditer(body))
        if not matches:
            continue
        payload, missing = hydrate_mapping(decode_mapping(matches[-1].group("payload")), bodies)
        # Synthesis is advisory. A missing synthesis sidecar must not make a
        # durable vote or legacy agenda unrecoverable; those fields simply
        # decode as None. Existing PR/review spill fields remain strict.
        discuss_optional_missing = (
            {
                "round_synthesis", "final_synthesis", "raw_synthesis_response",
                "analyzer_response", "final_analyzer_response",
            }
            if flow == "discuss"
            else set()
        )
        required_missing = missing - discuss_optional_missing
        if required_missing:
            raise AgentLoopError(
                "Incomplete round metadata: "
                f"{', '.join(sorted(required_missing))} sidecars are unavailable; "
                "restore sidecars or remove the incomplete anchor and rerun."
            )
        metadata = round_metadata._decode_round_metadata_mapping(payload)
        if metadata.flow != flow:
            continue
        records.append(
            round_metadata.PostedRoundRecord(
                index=index,
                metadata=metadata,
                body=round_metadata._strip_round_metadata(body),
            )
        )
    if flow == "plan":
        # Legacy machine promotion exists for PR ledgers, whose machine gates
        # have clearance paths.  A plan ledger has none (#1005).
        return _demote_plan_machine_items(tuple(records))
    return _promote_legacy_machine_items(recover_pr_acknowledgement_aliases(records))


def _scheduler_item_ledger_signature(items):
    """Code scheduling ignores findings owned by verified visual turns."""
    from .review_scope import is_visual_item
    return _prior_item_ledger_signature(tuple(item for item in items if not is_visual_item(item)))


def _prior_item_ledger_signature(items: Sequence[UnresolvedReviewItem]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            item.item_id,
            item.reviewer,
            item.source_round,
            item.text,
            item.status,
            item.source_status,
            item.notes,
            item.fix_scope,
            item.resolution_owners,
            item.owner_states,
            item.owner_evidence,
            item.owner_dispositions,
            item.authority,
            item.obligation_kind,
            item.lifecycle,
            item.failed_head_sha,
            item.candidate_head_sha,
            item.obligation_identity,
            *((("evidence_aliases", item.alias_item_ids, item.alias_obligation_identities),)
              if item.alias_item_ids or item.alias_obligation_identities else ()),
            # Preserve legacy digests while distinguishing visual ownership.
            *((('origin_scope', item.origin_scope),) if item.origin_scope is not None else ()),
            *((('screenshot_request', repr(item.screenshot_request)),) if item.screenshot_request is not None else ()),
            # Appended only for items with sub-items, so a legacy item keeps
            # its pre-#958 signature tuple exactly.
            *(
                (
                    tuple(
                        (sub.sub_item_id, sub.text, sub.status, sub.resolved_round)
                        for sub in item.sub_items
                    ),
                )
                if item.sub_items
                else ()
            ),
        )
        for item in items
    )


def _select_current_round_records(
    records: Sequence[round_metadata.PostedRoundRecord],
    *,
    subject: str,
) -> round_metadata.ResumedRoundSelection | None:
    records = authority_records(records)
    subject_records = [record for record in records if record.metadata.subject == subject]
    if not subject_records:
        return None
    anchor_record = select_plan_round_anchor(subject_records)
    anchor_metadata = anchor_record.metadata
    prior_items_signature = _prior_item_ledger_signature(anchor_metadata.prior_items)
    current_round_records = tuple(
        record
        for record in subject_records
        if record.metadata.round_number == anchor_metadata.round_number
        and (
            _prior_item_ledger_signature(record.metadata.prior_items) == prior_items_signature
            or (anchor_metadata.flow == "plan" and record.metadata.role == "reviewer"
                and record.index > anchor_record.index)
        )
    )
    # A head-review recovery handoff for this round is a boundary: records the
    # same head, round and ledger published before it are never part of the
    # recovered round (it carries only the ledger, budget and handoff).  Uses only
    # fields that always decode, so it applies to every caller, live or retired.
    boundary = next(
        (
            record
            for record in reversed(subject_records)
            if (
                record.metadata.phase == round_metadata.HEAD_REVIEW_RECOVERY_PHASE
                or _is_acknowledgement_failure_boundary(record.metadata)
            )
            and record.metadata.round_number == anchor_metadata.round_number
        ),
        None,
    )
    if boundary is not None:
        current_round_records = tuple(
            record for record in current_round_records if record.index >= boundary.index
        )
    latest_coder_record = next(
        (
            record
            for record in reversed(current_round_records)
            if record.metadata.role == "coder"
        ),
        None,
    )
    if latest_coder_record is not None:
        current_round_records = tuple(
            record for record in current_round_records if record.index >= latest_coder_record.index
        )
    return round_metadata.ResumedRoundSelection(
        anchor_record=anchor_record,
        current_round_records=current_round_records,
    )


from .plan_round_closure import (
    _plan_round_is_reconciled,
    project_plan_history,
    select_plan_round_anchor,
)


PR_PRE_RECONCILIATION_PHASES = frozenset(
    {
        "scheduler-prelaunch",
        "acknowledgement-repair-failed",
        round_metadata.CODER_DISPATCH_PHASE,
        round_metadata.CODER_FOLLOWUP_REJECTED_PHASE,
        round_metadata.HEAD_REVIEW_RECOVERY_PHASE,
    }
)


def _is_acknowledgement_failure_boundary(metadata: round_metadata.PostedRoundMetadata) -> bool:
    return (metadata.flow == "pr" and metadata.phase == "acknowledgement-repair-failed"
            and metadata.role == "summary" and metadata.agent == "Orchestrator")


# Records that reopen a reconciled PR round at the same head and round number:
# conversion, head-review recovery, and acknowledgement repair failure. A
# scheduler checkpoint after one of them belongs to the reopened round.
_PR_ROUND_REOPENING_PHASES = frozenset({
    round_metadata.EVIDENCE_CONVERSION_PHASE, round_metadata.HEAD_REVIEW_RECOVERY_PHASE, "acknowledgement-repair-failed",
})


def inert_post_reconciliation_checkpoint_indexes(
    records: Sequence[round_metadata.PostedRoundRecord],
) -> frozenset[int]:
    """Indexes of PR scheduler checkpoints posted into an already closed round (#1553).

    A PR round is closed by its reconciliation-phase summary.  A later
    ``scheduler-prelaunch`` summary with the same round number and subject
    (a stray checkpoint written by an older tool version, or a signed
    board-amendment activation) is inert for resume anchor selection and for
    every scheduler reader.  Only the amendment-lineage reader, which reads
    the comments itself, still sees it, and the startup force-full latch
    still reads an amendment activation (#1559, see
    ``force_full_latch_records``).  Nothing is deleted.
    """
    closed: set[tuple[int, str]] = set()
    inert: set[int] = set()
    for record in sorted(records, key=lambda item: item.index):
        metadata = record.metadata
        key = (metadata.round_number, metadata.subject)
        if metadata.role == "summary" and metadata.phase == "reconciliation" and not scope_present(metadata):
            closed.add(key)
        elif metadata.phase in _PR_ROUND_REOPENING_PHASES:
            closed.discard(key)
        elif (
            metadata.role == "summary"
            and metadata.phase == "scheduler-prelaunch"
            and key in closed
        ):
            inert.add(record.index)
    return frozenset(inert)


def _is_inert_post_reconciliation_checkpoint(
    record: round_metadata.PostedRoundRecord, inert_indexes: frozenset[int]
) -> bool:
    return record.index in inert_indexes


def drop_inert_post_reconciliation_checkpoints(
    records: Sequence[round_metadata.PostedRoundRecord],
) -> tuple[round_metadata.PostedRoundRecord, ...]:
    """``records`` without the inert post-reconciliation checkpoints (#1553)."""
    inert = inert_post_reconciliation_checkpoint_indexes(records)
    if not inert:
        return tuple(records)
    return tuple(
        record for record in records
        if not _is_inert_post_reconciliation_checkpoint(record, inert)
    )


def post_reconciliation_amendment_activation_indexes(
    records: Sequence[round_metadata.PostedRoundRecord],
) -> frozenset[int]:
    """Inert checkpoint indexes that are lineage-visible amendment activations (#1559).

    An inert post-reconciliation checkpoint is an activation when it carries a
    non-null reviewer-board amendment digest and is the earliest record, of
    any role, carrying that digest: the record the amendment lineage counts
    as the amendment's first use.  A stray copied from an earlier checkpoint
    has a null or repeated digest and is not an activation.
    """
    inert = inert_post_reconciliation_checkpoint_indexes(records)
    if not inert:
        return frozenset()
    seen: set[str] = set()
    activations: set[int] = set()
    for record in sorted(records, key=lambda item: item.index):
        digest = record.metadata.reviewer_board_amendment_digest
        if digest is None or digest in seen:
            continue
        seen.add(digest)
        if record.index in inert:
            activations.add(record.index)
    return frozenset(activations)


def force_full_latch_records(
    records: Sequence[round_metadata.PostedRoundRecord],
) -> tuple[round_metadata.PostedRoundRecord, ...]:
    """Records the startup force-full latch reconstruction reads (#1559).

    The non-inert records plus every inert amendment activation, in input
    order.  A run interrupted right after an activation-only record keeps an
    operator latch recorded only there; strays stay fully inert.
    """
    inert = inert_post_reconciliation_checkpoint_indexes(records)
    if not inert:
        return tuple(records)
    activations = post_reconciliation_amendment_activation_indexes(records)
    return tuple(
        record for record in records
        if record.index not in inert or record.index in activations
    )


def startup_force_full_latch(
    records: Sequence[round_metadata.PostedRoundRecord], *, requires_primary: bool
) -> tuple[bool, bool]:
    """``(operator, automatic)`` force-full latches restored at PR startup.

    An operator-sourced latch is always restored.  Any other latch (automatic
    or legacy unsourced) is restored only for a policy that does not require
    a primary; the staged policy re-derives it from qualified panel evidence.
    """
    operator = automatic = False
    for record in force_full_latch_records(records):
        if not scope_present(record.metadata) and record.metadata.scheduler_force_full:
            if record.metadata.scheduler_force_full_source == "operator":
                operator = True
            elif not requires_primary:
                automatic = True
    return operator, automatic


def _pr_round_is_reconciled(records: Sequence[round_metadata.PostedRoundRecord]) -> bool:
    """True when the PR round holds an actual reconciliation summary.

    The scheduler-prelaunch summary the PR loop posts before review (also the
    secondary-audit panel opening) is a pre-reviewer checkpoint, exactly as it
    is for plans (#905); legacy unlabeled summaries still count.
    """
    return any(
        record.metadata.role == "summary"
        and not scope_present(record.metadata)
        and record.metadata.phase not in PR_PRE_RECONCILIATION_PHASES
        for record in records
    )


def _max_unresolved_item_number_from_records(records: Sequence[round_metadata.PostedRoundRecord]) -> int:
    max_number = 0
    for record in records:
        max_number = max(max_number, record.original_item_number_high_water)
        for item in (*record.metadata.prior_items, *record.metadata.new_items):
            for item_id in (item.item_id, *item.alias_item_ids):
                if match := re.fullmatch(r"item-(\d+)", item_id):
                    max_number = max(max_number, int(match.group(1)))
    return max_number


def _active_pr_items(items: Sequence[UnresolvedReviewItem]) -> list[UnresolvedReviewItem]:
    # Inert converted findings and row tombstones (#1325) are retained so a
    # resume keeps provenance and can reopen a tombstone at a new head.
    return [
        item for item in items
        if item.status in {"blocking", "same-pr"} or item.is_inert_ledger_record
    ]


def _append_active_pr_new_items(
    active_items: list[UnresolvedReviewItem],
    records: Sequence[round_metadata.PostedRoundRecord],
) -> None:
    seen_item_ids = {item.item_id for item in active_items}
    for record in records:
        if scope_present(record.metadata):
            continue
        for item in record.metadata.new_items:
            if item.status not in {"blocking", "same-pr"} or item.item_id in seen_item_ids:
                continue
            active_items.append(item)
            seen_item_ids.add(item.item_id)


from .native_evidence import _append_recorded_evidence_requests


def _aggregate_record_dispositions(
    records: Sequence[round_metadata.PostedRoundRecord],
) -> dict[str, list[ReviewItemDisposition]]:
    dispositions_by_item: dict[str, list[ReviewItemDisposition]] = {}
    for record in records:
        if scope_present(record.metadata):
            continue
        for disposition in record.metadata.dispositions:
            dispositions_by_item.setdefault(disposition.item_id, []).append(disposition)
    return dispositions_by_item


def _canonically_resolved_history_item_ids(
    records: Sequence[round_metadata.PostedRoundRecord],
    *,
    reconciliation_mode: str,
    same_status: str,
    current_carried_ids: Sequence[str] = (),
) -> frozenset[str]:
    """Return item IDs that recorded history proves were canonically cleared.

    Round metadata is replayed group by group (subject, round number) through
    the same reconciler that the live loop uses, so owner-scoped partial or
    non-owner clearance keeps an item active exactly as it did live.  An item
    counts as resolved only if the reconciler dropped it and every disposition
    for it in the clearing round was ``resolved``; a later carry or
    re-introduction makes it active again.  Machine obligations are never
    returned.  The result is a whitelist for removing no-op ``resolved``
    dispositions, so every ambiguity must exclude the ID.
    """
    groups: dict[tuple[str, int], list[round_metadata.PostedRoundRecord]] = {}
    for record in records:
        key = (record.metadata.subject, record.metadata.round_number)
        groups.setdefault(key, []).append(record)
    active: dict[str, UnresolvedReviewItem] = {}
    resolved: set[str] = set()
    for group in groups.values():
        for record in group:
            for item in record.metadata.prior_items:
                active[item.item_id] = item
                resolved.discard(item.item_id)
        dispositions_by_item = _aggregate_record_dispositions(
            [record for record in group if is_complete_code_review(record.metadata)]
        )
        candidates = [
            item for item in active.values() if item.item_id in dispositions_by_item
        ]
        if candidates:
            kept, future = _apply_unresolved_item_dispositions(
                candidates,
                dispositions_by_item,
                same_status=same_status,
                retain_future=True,
                reconciliation_mode=reconciliation_mode,
            )
            remaining_ids = {item.item_id for item in (*kept, *future)}
            for item in candidates:
                if item.item_id in remaining_ids:
                    continue
                del active[item.item_id]
                if _is_machine_obligation(item):
                    continue
                if all(
                    disposition.disposition == "resolved"
                    for disposition in dispositions_by_item[item.item_id]
                ):
                    resolved.add(item.item_id)
        for record in group:
            for item in record.metadata.new_items:
                active[item.item_id] = item
                resolved.discard(item.item_id)
    return frozenset(resolved - set(active) - set(current_carried_ids))


def _replay_history_items(
    records: Sequence[round_metadata.PostedRoundRecord],
    *,
    reconciliation_mode: str,
    same_status: str,
) -> tuple[dict[str, str], dict[str, UnresolvedReviewItem]]:
    """Shared group-by-(subject, round) replay for the history helpers.

    Returns the legacy per-item ``outcomes`` map and, tracked independently,
    the items currently eligible for deferred publication (#1389).  The
    outcomes map is never derived from, nor fed into, the deferred map.
    """
    groups: dict[tuple[str, int], list[round_metadata.PostedRoundRecord]] = {}
    for record in records:
        key = (record.metadata.subject, record.metadata.round_number)
        groups.setdefault(key, []).append(record)
    active: dict[str, UnresolvedReviewItem] = {}
    outcomes: dict[str, str] = {}
    deferred: dict[str, UnresolvedReviewItem] = {}

    def appear(item: UnresolvedReviewItem) -> None:
        if _is_machine_obligation(item):
            return
        # A future item still carried in the stored ledger (full context mode, planner)
        # stays a reconciliation candidate, so a later authoritative `resolved`
        # clears it; compact mode never stores it, so absence keeps it deferred.
        active[item.item_id] = item
        outcomes[item.item_id] = "deferred" if item.status == "future" else "active"
        if item.status == "future":
            deferred[item.item_id] = item
        else:
            deferred.pop(item.item_id, None)

    for group in groups.values():
        for record in group:
            for item in record.metadata.prior_items:
                appear(item)
        dispositions_by_item = _aggregate_record_dispositions(
            [record for record in group if is_complete_code_review(record.metadata)]
        )
        candidates = [
            item for item in active.values() if item.item_id in dispositions_by_item
        ]
        if candidates:
            kept, future = _apply_unresolved_item_dispositions(
                candidates,
                dispositions_by_item,
                same_status=same_status,
                retain_future=False,
                reconciliation_mode=reconciliation_mode,
            )
            for item in future:
                active.pop(item.item_id, None)
                outcomes[item.item_id] = "deferred"
                deferred[item.item_id] = item
            for item in kept:
                active[item.item_id] = item
                if item.status == "future":
                    deferred[item.item_id] = item
                else:
                    # Re-promoted to mandatory work: no longer publishable.
                    deferred.pop(item.item_id, None)
            returned = {item.item_id for item in (*kept, *future)}
            for item in candidates:
                if item.item_id in returned:
                    continue
                # The reconciler drops a non-machine item only when every
                # obligation is cleared, including a clearance it derived from
                # sub-item completion, so the drop itself is authoritative.
                del active[item.item_id]
                outcomes[item.item_id] = "resolved"
                deferred.pop(item.item_id, None)
        for record in group:
            for item in record.metadata.new_items:
                appear(item)
    return outcomes, deferred


def canonical_history_item_outcomes(
    records: Sequence[round_metadata.PostedRoundRecord],
    *,
    reconciliation_mode: str,
    same_status: str,
) -> dict[str, str]:
    """Replay already-extracted records into per-item history outcomes (#1273).

    Read-only sibling of ``_canonically_resolved_history_item_ids``: the same
    group-by-(subject, round) replay through the live reconciler, but called
    with ``retain_future=False`` (the compact-PR call shape).  Every item whose
    reconciled status is ``future`` therefore lands in the SECOND returned
    collection, which is where a deferral is read; the FIRST collection holds
    only still-mandatory items and carries their updated state forward.

    Returns ``item_id -> "active" | "deferred" | "resolved"``.  ``resolved`` is
    any candidate the reconciler dropped (literal or derived clearance), so a
    retained future item cleared by sub-item completion is not left deferred.
    A later carry or reintroduction as a mandatory item resets the outcome to
    ``active``; machine obligations never get an outcome.
    """
    outcomes, _deferred = _replay_history_items(
        records, reconciliation_mode=reconciliation_mode, same_status=same_status
    )
    return outcomes


def canonical_deferred_history_items(
    records: Sequence[round_metadata.PostedRoundRecord],
    *,
    reconciliation_mode: str,
    same_status: str,
) -> tuple[UnresolvedReviewItem, ...]:
    """Return the items whose latest replayed state is a pending deferral (#1389).

    An item is eligible when it last appeared (as a prior or new item) with
    status ``future`` or the reconciler deferred it.  A later appearance or
    reconciliation with a mandatory status (re-promotion), or a reconciler
    drop (resolution), removes it.  Machine obligations are ignored.  This
    feeds only approval-time follow-up publication; it is never required work.
    """
    _outcomes, deferred = _replay_history_items(
        records, reconciliation_mode=reconciliation_mode, same_status=same_status
    )
    return tuple(deferred.values())


def _live_round_resolved_item_ids(
    *,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions_by_item: Mapping[str, Sequence[ReviewItemDisposition]],
    carried_item_ids: Sequence[str],
) -> frozenset[str]:
    """Return IDs the round still in progress canonically resolved (#874).

    The posted-comment snapshot a loop holds is fetched before its review
    turns, so replaying it through ``_canonically_resolved_history_item_ids``
    can never see the round that is still running: the IDs it just cleared stay
    active in that history and are excluded from the whitelist.  A turn that
    runs after the round's dispositions were applied -- the plan revision --
    therefore needs this in-process proof too.  The dispositions and the
    post-round carried set are both authenticated in memory, and the rule is
    the same one applied to recorded history: the reconciler dropped the item,
    every disposition it received in this round was ``resolved``, and machine
    obligations never count.
    """
    carried = set(carried_item_ids)
    resolved: set[str] = set()
    for item in prior_items:
        if item.item_id in carried or _is_machine_obligation(item):
            continue
        dispositions = tuple(dispositions_by_item.get(item.item_id, ()))
        if not dispositions:
            continue
        if all(disposition.disposition == "resolved" for disposition in dispositions):
            resolved.add(item.item_id)
    return frozenset(resolved)


def _drop_unterminated_evidence_response_records(
    records: Sequence[round_metadata.PostedRoundRecord],
) -> tuple[round_metadata.PostedRoundRecord, ...]:
    """Ignore an interrupted evidence-response pass (#1068).

    Every evidence-response or refresh pass ends with exactly one terminal
    freeze or release record at the same round and head.  Reviewer records of
    a pass with no following terminal record are a partial transcript: they
    must neither anchor recovery nor clear evidence, so the rerun restores the
    prior state and repeats the whole pass.
    """
    terminal: list[tuple[int, str, int]] = [
        (record.index, record.metadata.subject, record.metadata.round_number)
        for record in records
        if record.metadata.phase in round_metadata.EVIDENCE_TERMINAL_PHASES
    ]
    kept: list[round_metadata.PostedRoundRecord] = []
    for record in records:
        metadata = record.metadata
        if metadata.phase == round_metadata.EVIDENCE_RESPONSE_PHASE and not any(
            index > record.index
            and subject == metadata.subject
            and round_number == metadata.round_number
            for index, subject, round_number in terminal
        ):
            continue
        kept.append(record)
    return tuple(kept)


def _invalid_evidence_record_item(
    items: Sequence[UnresolvedReviewItem],
    *,
    records: Sequence[round_metadata.PostedRoundRecord],
    round_number: int,
    reason: str,
) -> UnresolvedReviewItem:
    """The non-bypassable blocker for a malformed evidence record."""
    return UnresolvedReviewItem(
        item_id=f"item-{_max_unresolved_item_number_from_records(records) + 1}",
        reviewer="Orchestrator",
        source_round=round_number,
        text=(
            f"The persisted exact-head evidence record is malformed or contradictory ({reason}). "
            "No approval, qualification, or merge is permitted until an operator repairs it."
        ),
        status="blocking",
        source_status="blocking",
        authority=UNKNOWN_MACHINE_AUTHORITY,
        obligation_kind="unknown",
        lifecycle="repair_required",
        obligation_identity="invalid-evidence-record",
    )


def _evidence_boundary_ledger(
    record: round_metadata.PostedRoundRecord,
    *,
    records: Sequence[round_metadata.PostedRoundRecord],
) -> tuple[UnresolvedReviewItem, ...]:
    """Return a terminal record's ledger, failing closed on contradictions."""
    metadata = record.metadata
    ledger = tuple(metadata.prior_items)
    reason: str | None = None
    if metadata.phase == round_metadata.EVIDENCE_FREEZE_PHASE:
        freeze = metadata.evidence_freeze
        if freeze is None or not freeze.valid:
            reason = "the freeze payload is missing or malformed"
        else:
            frozen = {
                item.obligation_identity
                for item in ledger
                if item.is_machine_obligation
                and item.obligation_kind == EVIDENCE_OBLIGATION_KIND
                and item.lifecycle == "evidence_frozen"
                and item.candidate_head_sha == freeze.frozen_head
            }
            if freeze.frozen_head != metadata.subject:
                reason = "the frozen head does not match the record head"
            elif frozen != set(freeze.evidence_identities):
                reason = "the frozen identities do not match the frozen ledger"
    else:
        release = metadata.evidence_release
        if release is None or not release.valid:
            reason = "the release payload is missing or malformed"
        elif release.released_head != metadata.subject:
            reason = "the released head does not match the record head"
        elif any(
            item.is_machine_obligation
            and item.obligation_kind == EVIDENCE_OBLIGATION_KIND
            and item.lifecycle == "evidence_frozen"
            for item in ledger
        ):
            reason = "a released ledger still holds frozen evidence"
    if reason is None:
        return ledger
    return (
        *ledger,
        _invalid_evidence_record_item(
            ledger, records=records, round_number=metadata.round_number, reason=reason
        ),
    )


class _RecoveryRefusal(AgentLoopError):
    """A refusal ``--review-unrecorded-head`` is allowed to override (#1292)."""


def _strip_bot_suffix(login: object) -> object:
    if isinstance(login, str) and login.endswith("[bot]"):
        return login[: -len("[bot]")]
    return login


def _recovery_record_admitted(
    record: round_metadata.PostedRoundRecord,
    comments: Sequence[object],
    trusted_actor: tuple[str, int] | None,
) -> bool:
    """True only when the authenticated GitHub actor authored ``record``.

    Round metadata is not authenticated in general; this check applies only to
    the record kinds and anchor branch added by #1292.
    """
    if trusted_actor is None:
        return False
    login, actor_id = trusted_actor
    if not 0 <= record.index < len(comments):
        return False
    comment = comments[record.index]
    author_id = getattr(comment, "author_id", None)
    return (
        isinstance(author_id, int)
        and not isinstance(author_id, bool)
        and author_id == actor_id
        and _strip_bot_suffix(getattr(comment, "author", None)) == _strip_bot_suffix(login)
    )


def record_carries_evidence_authority(metadata: round_metadata.PostedRoundMetadata) -> bool:
    """Whether a record carries #1325 evidence-conversion authority.

    True for the conversion phase, any signed-evidence reference, a capture
    classification, or a stall snapshot that pairs a review with a coder
    attempt.  Such records are admitted only from the authenticated actor.
    """
    stall = metadata.evidence_stall
    return (
        bool(metadata.post_merge_registry)
        or metadata.post_merge_publication is not None
        or metadata.phase == round_metadata.EVIDENCE_CONVERSION_PHASE
        or bool(metadata.signed_evidence_references)
        or metadata.risk_test_matrix_capture_classification is not None
        or metadata.evidence_conversion is not None
        or metadata.conversion_stall_snapshot is not None
        or (isinstance(stall, Mapping) and "bound_attempt_round" in stall)
    )


def strip_evidence_authority(metadata: round_metadata.PostedRoundMetadata) -> round_metadata.PostedRoundMetadata:
    """The record as an unauthenticated reader may use it (#1325).

    References, the audit, the classification and the attempt pairing are
    removed; the stall snapshot itself keeps the stage-1 trust model.
    """
    stall = metadata.evidence_stall
    if isinstance(stall, Mapping) and "bound_attempt_round" in stall:
        stall = {key: value for key, value in stall.items() if key != "bound_attempt_round"}
    return replace(
        metadata,
        post_merge_authenticated=False,
        post_merge_registry=(),
        post_merge_publication=None,
        risk_test_matrix_capture_classification=None,
        evidence_conversion=None,
        conversion_stall_snapshot=None,
        conversion_review_round=None,
        signed_evidence_references=(),
        evidence_stall=stall,
    )


def admit_evidence_authority_records(
    records: Sequence[round_metadata.PostedRoundRecord],
    comments: Sequence[object],
    trusted_actor: tuple[str, int] | None,
) -> tuple[round_metadata.PostedRoundRecord, ...]:
    """Fail-closed authorship check for #1325 authority carriers.

    A record whose comment was not authored by the authenticated actor (or
    when the actor is unknown) keeps its ordinary content but loses every
    conversion and signed-evidence field.  Strict decoding and the hashed
    reference ID prove consistency, not authorship.
    """
    admitted: list[round_metadata.PostedRoundRecord] = []
    rejected = 0
    for record in records:
        if not record_carries_evidence_authority(record.metadata) or _recovery_record_admitted(
            record, comments, trusted_actor
        ):
            if record.metadata.post_merge_registry:
                record = replace(record, metadata=replace(record.metadata, post_merge_authenticated=True))
            admitted.append(record)
            continue
        rejected += 1
        admitted.append(replace(record, metadata=strip_evidence_authority(record.metadata)))
    if rejected:
        logger.warning(
            "Ignoring evidence-conversion authority on %d record(s) not authored by the "
            "authenticated orchestrator actor", rejected,
        )
    return tuple(admitted)


def _admit_recovery_records(
    records: Sequence[round_metadata.PostedRoundRecord],
    comments: Sequence[object],
    trusted_actor: tuple[str, int] | None,
) -> tuple[round_metadata.PostedRoundRecord, ...]:
    """Drop new-phase records the authenticated actor did not author.

    A conversion record (#1325) is a ledger handoff too, so a foreign one is
    dropped rather than restored; every other record loses only its
    evidence-conversion authority fields.
    """
    return admit_evidence_authority_records(
        tuple(
            record
            for record in records
            if (
                record.metadata.phase not in round_metadata.RECOVERY_RECORD_PHASES
                and record.metadata.phase != round_metadata.EVIDENCE_CONVERSION_PHASE
            )
            or (
                _recovery_record_admitted(record, comments, trusted_actor)
                # A conversion handoff whose audit was rejected is not a
                # conversion: recovery reconstructs the preceding ledger.
                and (
                    record.metadata.phase != round_metadata.EVIDENCE_CONVERSION_PHASE
                    or record.metadata.evidence_conversion is not None
                )
            )
        ),
        comments,
        trusted_actor,
    )


def _is_budget_candidate(metadata: round_metadata.PostedRoundMetadata) -> bool:
    """Whether the operator override may take a round budget from this record.

    One predicate shared by budget selection and REST coverage, so the records
    that can supply a budget are exactly the records whose authorship must be
    proven: any record bearing a qualification checkpoint, or a recovery record.
    """
    return metadata.qualification_checkpoint is not None or (
        metadata.phase in round_metadata.RECOVERY_RECORD_PHASES and metadata.recovery_round_budget is not None
    )


def unauthenticated_recovery_record_indexes(
    comments: Sequence[object],
    head_sha: str | None = None,
    review_unrecorded_head: bool = False,
) -> tuple[int, ...]:
    """Indexes of records recovery depends on whose comment has no author identity.

    After the REST identity merge every matched comment has a numeric author ID.
    A needed record left without one was not covered by the REST read (a page
    omitted it), so it can be neither admitted nor ruled out as another
    author's.  Needed records are: every recovery-phase record; the legacy
    qualification-checkpoint anchor of a head that has no records of its own;
    and, under ``--review-unrecorded-head``, every checkpoint (a budget source).
    """
    records = authority_records(_extract_round_metadata_records(comments, flow="pr"))
    needed: set[int] = set()
    for record in records:
        phase = record.metadata.phase
        if phase in round_metadata.RECOVERY_RECORD_PHASES or record_carries_evidence_authority(record.metadata):
            needed.add(record.index)
        elif review_unrecorded_head and _is_budget_candidate(record.metadata):
            needed.add(record.index)
    if head_sha and records and not any(r.metadata.subject == head_sha for r in records):
        prior = [r for r in records if r.metadata.phase != round_metadata.CODER_DISPATCH_PHASE]
        if prior and prior[-1].metadata.phase == "qualification-checkpoint":
            needed.add(prior[-1].index)
    return tuple(
        sorted(
            index for index in needed
            if getattr(comments[index], "author_id", None) is None
        )
    )


def pr_resume_needs_author_admission(
    comments: Sequence[object],
    head_sha: str | None,
    review_unrecorded_head: bool = False,
) -> bool:
    """Whether resuming needs REST identities and the authenticated actor."""
    if review_unrecorded_head:
        return True
    records = authority_records(_extract_round_metadata_records(comments, flow="pr"))
    if any(
        record.metadata.phase in round_metadata.RECOVERY_RECORD_PHASES
        or record_carries_evidence_authority(record.metadata)
        for record in records
    ):
        return True
    return bool(
        records
        and head_sha
        and not any(record.metadata.subject == head_sha for record in records)
    )


def _coder_handoff_is_live(
    records: Sequence[round_metadata.PostedRoundRecord], handoff: round_metadata.PostedRoundRecord
) -> bool:
    """A coder-recovery handoff is retired by any later coder, reviewer or review handoff."""
    for record in records:
        if record.index <= handoff.index:
            continue
        metadata = record.metadata
        if scope_present(metadata):
            continue
        if metadata.role in {"coder", "reviewer"} or metadata.phase == round_metadata.HEAD_REVIEW_RECOVERY_PHASE:
            return False
    return True


def _head_review_handoff_is_live(
    records: Sequence[round_metadata.PostedRoundRecord], handoff: round_metadata.PostedRoundRecord
) -> bool:
    """A head-review handoff is retired by a later coder/reviewer record or reconciliation on its subject."""
    for record in records:
        if record.index <= handoff.index or record.metadata.subject != handoff.metadata.subject:
            continue
        metadata = record.metadata
        if scope_present(metadata):
            continue
        if metadata.role in {"coder", "reviewer"}:
            return False
        if (
            metadata.role == "summary"
            and metadata.round_number == handoff.metadata.round_number
            and (
                metadata.phase in {"reconciliation", "authoritative"}
                or metadata.phase in round_metadata.EVIDENCE_TERMINAL_PHASES
            )
        ):
            return False
    return True


def _live_recovery_handoff(
    records: Sequence[round_metadata.PostedRoundRecord], subject: str, *, head_moved: bool
) -> round_metadata.PostedRoundRecord | None:
    """The newest live recovery handoff on ``subject``, validated before any field is read.

    Candidacy and liveness use only fields that always decode (index, phase,
    subject, role, core round).  A structurally invalid live candidate raises
    before any agent call.  Only a valid first-attempt dispatch on the current
    head is skipped, because ordinary resume must behave as it did before.
    """
    candidates: list[round_metadata.PostedRoundRecord] = []
    for record in records:
        metadata = record.metadata
        if metadata.subject != subject:
            continue
        if metadata.phase in {round_metadata.CODER_DISPATCH_PHASE, round_metadata.CODER_FOLLOWUP_REJECTED_PHASE}:
            if _coder_handoff_is_live(records, record):
                candidates.append(record)
        elif metadata.phase == round_metadata.HEAD_REVIEW_RECOVERY_PHASE:
            if _head_review_handoff_is_live(records, record):
                candidates.append(record)
    for record in reversed(candidates):
        problems = round_metadata._recovery_record_problems(record.metadata)
        if problems:
            raise _RecoveryRefusal(
                f"Recovery record at comment index {record.index} "
                f"(phase {record.metadata.phase}) is malformed: {', '.join(problems)}. "
                "Rerun with --review-unrecorded-head to review the current head."
            )
        metadata = record.metadata
        if (
            not head_moved
            and metadata.phase == round_metadata.CODER_DISPATCH_PHASE
            and metadata.dispatch_attempt == 1
            and metadata.recovery_dispatch is False
        ):
            continue
        return record
    return None


def _consume_recovery_handoff(
    records: Sequence[round_metadata.PostedRoundRecord],
    handoff: round_metadata.PostedRoundRecord,
    *,
    head_moved: bool,
) -> round_metadata.ResumedReviewRound | None:
    metadata = handoff.metadata
    next_number = _max_unresolved_item_number_from_records(records) + 1
    ledger = tuple(_active_pr_items(metadata.prior_items))
    if metadata.phase == round_metadata.HEAD_REVIEW_RECOVERY_PHASE:
        if not ledger:
            return None
        return round_metadata.ResumedReviewRound(
            round_number=metadata.round_number,
            prior_items=ledger,
            coder_output=None,
            completed_reviews=(),
            next_unresolved_item_number=next_number,
            ledger_may_be_incomplete=True,
            qualification_checkpoint=None,
            recovery_round_budget=metadata.recovery_round_budget,
            head_review_recovery=metadata.head_review_recovery_source,
            head_review_recovery_post_required=head_moved,
        )
    reason = metadata.rejected_coder_followup_reason or round_metadata.UNRECORDED_DISPATCH_REASON
    carried = (*metadata.carried_rejection_reasons, reason)[-round_metadata.MAX_REJECTED_DISPATCH_ATTEMPTS:]
    attempt = metadata.dispatch_attempt or 1
    if attempt >= round_metadata.MAX_REJECTED_DISPATCH_ATTEMPTS:
        listed = "; ".join(f"({number}) {text}" for number, text in enumerate(carried, 1))
        raise _RecoveryRefusal(
            f"Coder recovery stopped after {attempt} consecutive rejected or interrupted "
            f"follow-up attempts at round {metadata.dispatch_round}. Recorded reasons: {listed}. "
            + round_metadata.REVIEW_UNRECORDED_HEAD_HINT
        )
    if not ledger:
        return None
    return round_metadata.ResumedReviewRound(
        round_number=int(metadata.dispatch_round or metadata.round_number),
        prior_items=ledger,
        coder_output=None,
        completed_reviews=(),
        next_unresolved_item_number=next_number,
        ledger_may_be_incomplete=False,
        unrecorded_head_advance=True,
        qualification_checkpoint=None,
        rejected_coder_followup_reason=reason,
        rejected_coder_followup_from_head=metadata.dispatch_head,
        dispatch_attempt=attempt,
        recovery_round_budget=metadata.recovery_round_budget,
        carried_rejection_reasons=carried,
    )


def _legacy_checkpoint_head_review(
    anchor: round_metadata.PostedRoundRecord,
    records: Sequence[round_metadata.PostedRoundRecord],
    *,
    comments: Sequence[object],
    trusted_actor: tuple[str, int] | None,
) -> round_metadata.ResumedReviewRound | None:
    """Recover a PR stranded on a pre-handoff qualification checkpoint (#1292).

    The checkpoint may have preceded a coder dispatch or a required head review;
    legacy records cannot tell which, so both become an ordinary review round of
    the current head at the anchor's own round.  Only the bounded budget carries.
    """
    if not _is_legacy_checkpoint_anchor(anchor, comments, trusted_actor):
        return None
    metadata = anchor.metadata
    ledger = tuple(_active_pr_items(metadata.prior_items))
    if not ledger:
        return None
    return round_metadata.ResumedReviewRound(
        round_number=metadata.round_number,
        prior_items=ledger,
        coder_output=None,
        completed_reviews=(),
        next_unresolved_item_number=_max_unresolved_item_number_from_records(records) + 1,
        ledger_may_be_incomplete=True,
        unrecorded_head_advance=False,
        qualification_checkpoint=None,
        recovery_round_budget=round_metadata.RecoveryRoundBudget.from_checkpoint(
            metadata.qualification_checkpoint
        ),
        head_review_recovery="legacy-checkpoint",
        head_review_recovery_post_required=True,
    )


def _operator_head_review_resume(
    walk_records: Sequence[round_metadata.PostedRoundRecord],
    records: Sequence[round_metadata.PostedRoundRecord],
    *,
    comments: Sequence[object] = (),
    trusted_actor: tuple[str, int] | None = None,
) -> round_metadata.ResumedReviewRound | None:
    """``--review-unrecorded-head``: an ordinary review round of the current head."""
    records = authority_records(records)
    walk_records = authority_records(walk_records)
    if not records:
        return None
    latest = records[-1].metadata
    ledger = _active_pr_items(latest.prior_items)
    if latest.role in {"reviewer", "summary"}:
        _append_active_pr_new_items(ledger, (records[-1],))
    budget: round_metadata.RecoveryRoundBudget | None = None
    for record in reversed(walk_records):
        metadata = record.metadata
        # Every budget source is authenticated: recovery records were admitted
        # already, and a qualification checkpoint (or any other record) only
        # counts when the authenticated actor authored it.  Only recovery
        # phases may carry a recovery budget.
        if not _is_budget_candidate(metadata):
            continue
        if not _recovery_record_admitted(record, comments, trusted_actor):
            continue
        if round_metadata._recovery_record_problems(metadata):
            continue
        if (
            metadata.phase in round_metadata.RECOVERY_RECORD_PHASES
            and metadata.recovery_round_budget is not None
            and metadata.recovery_round_budget.valid
        ):
            budget = metadata.recovery_round_budget
            break
        if metadata.qualification_checkpoint is not None:
            candidate = round_metadata.RecoveryRoundBudget.from_checkpoint(metadata.qualification_checkpoint)
            if candidate is not None and candidate.valid:
                budget = candidate
                break
    if not ledger:
        return None
    return round_metadata.ResumedReviewRound(
        round_number=latest.round_number,
        prior_items=tuple(ledger),
        coder_output=None,
        completed_reviews=(),
        next_unresolved_item_number=_max_unresolved_item_number_from_records(walk_records) + 1,
        ledger_may_be_incomplete=True,
        qualification_checkpoint=None,
        recovery_round_budget=budget,
        head_review_recovery="operator",
        head_review_recovery_post_required=True,
    )


def _recover_unrecorded_pr_head_advance(
    records: Sequence[round_metadata.PostedRoundRecord],
    *,
    head_sha: str,
    reconciliation_mode: str = "aggregate",
    walk_records: Sequence[round_metadata.PostedRoundRecord] | None = None,
    comments: Sequence[object] = (),
    trusted_actor: tuple[str, int] | None = None,
) -> round_metadata.ResumedReviewRound | None:
    """Recover a head that carries no records of its own.

    ``records`` is the ordinary view (coder-dispatch records excluded);
    ``walk_records`` additionally holds the admitted dispatch records the
    recovery-handoff walk inspects.
    """
    records = authority_records(records)
    if walk_records is not None:
        walk_records = authority_records(walk_records)
    records = _drop_unterminated_evidence_response_records(records)
    walk_view = (
        records
        if walk_records is None
        else _drop_unterminated_evidence_response_records(walk_records)
    )
    prior_walk_records = [
        record for record in walk_view if record.metadata.subject != head_sha
    ]
    if prior_walk_records:
        handoff = _live_recovery_handoff(
            walk_view, prior_walk_records[-1].metadata.subject, head_moved=True
        )
        if handoff is not None:
            return _consume_recovery_handoff(walk_view, handoff, head_moved=True)
    prior_records = [record for record in records if record.metadata.subject != head_sha]
    if not prior_records:
        return None
    latest_prior_subject = prior_records[-1].metadata.subject
    latest_prior_record = prior_records[-1]
    if (
        latest_prior_record.metadata.phase == round_metadata.EVIDENCE_CONVERSION_PHASE
        and latest_prior_record.metadata.evidence_conversion is not None
    ):
        # A push after a conversion record (#1325): its post-conversion ledger
        # is the handoff, and live-head reconciliation rebinds the converted
        # obligations to the new head.
        recovered = _active_pr_items(release_evidence_freeze(latest_prior_record.metadata.prior_items))
        if not recovered:
            return None
        return round_metadata.ResumedReviewRound(
            round_number=latest_prior_record.metadata.round_number,
            prior_items=tuple(recovered),
            coder_output=None,
            completed_reviews=(),
            next_unresolved_item_number=_max_unresolved_item_number_from_records(records) + 1,
            ledger_may_be_incomplete=False,
            unrecorded_head_advance=True,
        )
    if latest_prior_record.metadata.phase in round_metadata.EVIDENCE_TERMINAL_PHASES:
        # A freeze or release was the last handoff before someone pushed a
        # new head.  The old head's evidence does not carry forward: frozen
        # evidence returns to deferred and the new head needs full review.
        boundary = latest_prior_record.metadata
        ledger = _evidence_boundary_ledger(latest_prior_record, records=records)
        from .native_evidence import reopen_cleared_requests
        ledger = reopen_cleared_requests(ledger, latest_prior_record, records, head_sha,
            admitted_records=[r for r in records if _recovery_record_admitted(r, comments, trusted_actor)])
        broken_head = (
            boundary.evidence_freeze.frozen_head
            if boundary.phase == round_metadata.EVIDENCE_FREEZE_PHASE
            and boundary.evidence_freeze is not None
            and boundary.evidence_freeze.valid
            else (boundary.subject if boundary.phase == round_metadata.EVIDENCE_FREEZE_PHASE else None)
        )
        recovered = _active_pr_items(release_evidence_freeze(ledger))
        if not recovered:
            return None
        return round_metadata.ResumedReviewRound(
            round_number=boundary.round_number,
            prior_items=tuple(recovered),
            coder_output=None,
            completed_reviews=(),
            next_unresolved_item_number=_max_unresolved_item_number_from_records(records) + 1,
            ledger_may_be_incomplete=False,
            unrecorded_head_advance=True,
            evidence_boundary=boundary,
            broken_evidence_freeze_head=broken_head,
        )
    selection = _select_current_round_records(records, subject=latest_prior_subject)
    if selection is None:
        return None

    current_round_records = selection.current_round_records
    anchor_metadata = selection.anchor_record.metadata
    latest_coder_record = next(
        (
            record
            for record in reversed(current_round_records)
            if record.metadata.role == "coder"
        ),
        None,
    )
    reviewer_records_after_coder = tuple(
        record
        for record in current_round_records
        if is_complete_code_review(record.metadata)
        and (latest_coder_record is None or record.index > latest_coder_record.index)
    )
    new_item_records_after_coder = tuple(
        record
        for record in current_round_records
        if record.metadata.role in {"reviewer", "summary"}
        and (latest_coder_record is None or record.index > latest_coder_record.index)
    )
    all_reviewer_records = tuple(
        record for record in current_round_records if is_complete_code_review(record.metadata)
    )
    all_new_item_records = tuple(
        record
        for record in current_round_records
        if record.metadata.role in {"reviewer", "summary"}
    )

    if latest_coder_record is not None and not reviewer_records_after_coder:
        round_number = latest_coder_record.metadata.round_number
        recovered_items = _active_pr_items(latest_coder_record.metadata.prior_items)
        coder_output = latest_coder_record.metadata.raw_structured_coder_response or latest_coder_record.body
        compact_prior_summaries = latest_coder_record.metadata.compact_prior_summaries
    elif latest_coder_record is not None:
        round_number = latest_coder_record.metadata.round_number
        recovered_items, _future_items = _apply_unresolved_item_dispositions(
            latest_coder_record.metadata.prior_items,
            _aggregate_record_dispositions(reviewer_records_after_coder),
            retain_future=False,
            reconciliation_mode=reconciliation_mode,
            round_number=round_number,
        )
        recovered_items = _active_pr_items(recovered_items)
        _append_active_pr_new_items(recovered_items, new_item_records_after_coder)
        _append_recorded_evidence_requests(
            recovered_items, reviewer_records_after_coder, records=records,
            admitted_records=[r for r in records if _recovery_record_admitted(r, comments, trusted_actor)]
        )
        coder_output = latest_coder_record.metadata.raw_structured_coder_response or latest_coder_record.body
        compact_prior_summaries = latest_coder_record.metadata.compact_prior_summaries
    elif all_reviewer_records:
        round_number = anchor_metadata.round_number
        recovered_items, _future_items = _apply_unresolved_item_dispositions(
            anchor_metadata.prior_items,
            _aggregate_record_dispositions(all_reviewer_records),
            retain_future=False,
            reconciliation_mode=reconciliation_mode,
            round_number=round_number,
        )
        recovered_items = _active_pr_items(recovered_items)
        _append_active_pr_new_items(recovered_items, all_new_item_records)
        _append_recorded_evidence_requests(
            recovered_items, all_reviewer_records, records=records,
            admitted_records=[r for r in records if _recovery_record_admitted(r, comments, trusted_actor)]
        )
        coder_output = None
        compact_prior_summaries = ()
    else:
        boundary = next((record for record in reversed(current_round_records)
                         if _is_acknowledgement_failure_boundary(record.metadata)), None)
        if boundary is not None:
            return round_metadata.ResumedReviewRound(
                round_number=boundary.metadata.round_number,
                prior_items=tuple(_active_pr_items(boundary.metadata.prior_items)),
                coder_output=None, completed_reviews=(),
                next_unresolved_item_number=_max_unresolved_item_number_from_records(walk_view) + 1,
                ledger_may_be_incomplete=False, unrecorded_head_advance=True,
            )
        return _legacy_checkpoint_head_review(
            selection.anchor_record,
            records,
            comments=comments,
            trusted_actor=trusted_actor,
        )

    if not recovered_items:
        return None

    return round_metadata.ResumedReviewRound(
        round_number=round_number,
        prior_items=tuple(recovered_items),
        coder_output=coder_output,
        completed_reviews=(),
        next_unresolved_item_number=_max_unresolved_item_number_from_records(records) + 1,
        ledger_may_be_incomplete=True,
        compact_prior_summaries=compact_prior_summaries,
        unrecorded_head_advance=True,
        coder_metadata=latest_coder_record.metadata if latest_coder_record else None,
        coder_comment_body=latest_coder_record.body if latest_coder_record else None,
        local_test_evidence=(
            latest_coder_record.metadata.local_test_evidence
            if latest_coder_record is not None
            else None
        ),
        qualification_checkpoint=_latest_qualification_checkpoint(records, head_sha=head_sha),
    )


def _is_legacy_checkpoint_anchor(
    record: round_metadata.PostedRoundRecord,
    comments: Sequence[object],
    trusted_actor: tuple[str, int] | None,
) -> bool:
    metadata = record.metadata
    return (
        metadata.role == "summary"
        and metadata.phase == "qualification-checkpoint"
        and _recovery_record_admitted(record, comments, trusted_actor)
    )


def _latest_prior_pr_subject_is_coherent(
    records: Sequence[round_metadata.PostedRoundRecord],
    *,
    head_sha: str,
    comments: Sequence[object] = (),
    trusted_actor: tuple[str, int] | None = None,
) -> bool:
    records = authority_records(records)
    prior_records = [record for record in records if record.metadata.subject != head_sha]
    if not prior_records:
        return True
    if prior_records[-1].metadata.phase in round_metadata.EVIDENCE_TERMINAL_PHASES:
        # A terminal evidence record is a complete handoff even when its
        # released ledger holds no active item.
        return True
    latest_prior_subject = prior_records[-1].metadata.subject
    selection = _select_current_round_records(records, subject=latest_prior_subject)
    if selection is None:
        return False
    if any(
        (record.metadata.role == "coder" or is_complete_code_review(record.metadata)
         or _is_acknowledgement_failure_boundary(record.metadata))
        and not scope_present(record.metadata)
        for record in selection.current_round_records
    ):
        return True
    # An admitted pre-handoff qualification checkpoint is a recoverable anchor
    # (#1292): the new head gets an ordinary review round of its own.
    return _is_legacy_checkpoint_anchor(selection.anchor_record, comments, trusted_actor)


def _approved_plan_hash(text: str) -> str:
    # Keep this local to avoid coupling the transport layer to decomposition.
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def _plan_declarations(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract bounded scope/deferred declarations without rewriting plan text."""
    sections: dict[str, list[str]] = {"scope": [], "deferred": []}
    active: str | None = None
    for line in text.splitlines():
        heading = re.match(r"^\s{0,3}#{2,6}\s+(.+?)\s*$", line)
        if heading:
            title = heading.group(1).strip().casefold()
            if any(token in title for token in ("deferred", "future", "out of scope", "non-goal")):
                active = "deferred"
            elif any(token in title for token in ("scope", "in scope", "current work")):
                active = "scope"
            else:
                active = None
            continue
        if active is not None and line.strip():
            sections[active].append(line.strip())
    return tuple(sections["scope"]), tuple(sections["deferred"])


def _approved_matrix_channel(
    *,
    contract_version: int | None,
    payload: object,
    changes_payload: object,
    identity: str | None,
    source_locator: str | None,
    diagnostic: str | None = None,
    boundary_digest: str | None = None,
    canonical_text: str | None = None,
) -> dict[str, object]:
    if contract_version is None and payload is None and changes_payload in (None, (), []):
        return {
            "risk_test_matrix_availability": "not-planned",
            "risk_test_matrix_diagnostic": None,
            "risk_test_matrix_contract_version": None,
            "risk_test_matrix_identity": None,
            "risk_test_matrix_payload": None,
            "risk_test_matrix_changes_payload": (),
            "risk_test_matrix_source_locator": source_locator,
            "risk_test_matrix_boundary_digest": boundary_digest,
        }
    if diagnostic:
        return {
            "risk_test_matrix_availability": "unavailable",
            "risk_test_matrix_diagnostic": diagnostic,
            "risk_test_matrix_contract_version": contract_version,
            "risk_test_matrix_identity": None,
            "risk_test_matrix_payload": None,
            "risk_test_matrix_changes_payload": (),
            "risk_test_matrix_source_locator": source_locator,
            "risk_test_matrix_boundary_digest": boundary_digest,
        }
    try:
        if contract_version != 1:
            raise ValueError("unsupported risk matrix contract version")
        matrix = parse_risk_test_matrix(payload, context="approved risk_test_matrix")
        changes = parse_risk_test_matrix_changes(
            changes_payload if changes_payload is not None else (),
            context="approved risk_test_matrix_changes",
        )
        actual_identity = risk_test_matrix_identity(matrix, changes, contract_version=contract_version)
        if not identity or identity != actual_identity:
            raise ValueError("risk matrix identity mismatch")
        if boundary_digest != actual_identity:
            raise ValueError("risk matrix payload-bound section boundary mismatch")
        if canonical_text and not round_metadata._canonical_matrix_boundary_matches(
            canonical_text, actual_identity
        ):
            raise ValueError("canonical plan risk matrix section boundary mismatch")
        return {
            "risk_test_matrix_availability": "available",
            "risk_test_matrix_diagnostic": None,
            "risk_test_matrix_contract_version": contract_version,
            "risk_test_matrix_identity": actual_identity,
            "risk_test_matrix_payload": matrix.to_payload(),
            "risk_test_matrix_changes_payload": tuple(change.to_payload() for change in changes),
            "risk_test_matrix_source_locator": source_locator,
            "risk_test_matrix_boundary_digest": boundary_digest,
        }
    except (AgentLoopError, TypeError, ValueError) as exc:
        return {
            "risk_test_matrix_availability": "unavailable",
            "risk_test_matrix_diagnostic": f"Matrix unavailable: {exc}",
            "risk_test_matrix_contract_version": contract_version,
            "risk_test_matrix_identity": None,
            "risk_test_matrix_payload": None,
            "risk_test_matrix_changes_payload": (),
            "risk_test_matrix_source_locator": source_locator,
            "risk_test_matrix_boundary_digest": boundary_digest,
        }


def make_approved_plan_context(
    text: str | None,
    *,
    source_locator: str | None = None,
    expected_hash: str | None = None,
    expected_subject: str | None = None,
    risk_test_matrix_contract_version: int | None = None,
    risk_test_matrix_payload: object = None,
    risk_test_matrix_changes_payload: object = None,
    risk_test_matrix_identity: str | None = None,
    risk_test_matrix_source_locator: str | None = None,
    risk_test_matrix_diagnostic: str | None = None,
    risk_test_matrix_boundary_digest: str | None = None,
) -> round_metadata.ApprovedPlanContext:
    """Build and validate a plan context from raw canonical text.

    Hash and subject validation happen before any historical-text sanitizing or
    prompt truncation.  Callers can therefore safely pass the returned context
    through compact and full prompt builders.
    """
    matrix_fields = _approved_matrix_channel(
        contract_version=risk_test_matrix_contract_version,
        payload=risk_test_matrix_payload,
        changes_payload=risk_test_matrix_changes_payload,
        identity=risk_test_matrix_identity,
        source_locator=risk_test_matrix_source_locator or source_locator,
        diagnostic=risk_test_matrix_diagnostic,
        boundary_digest=risk_test_matrix_boundary_digest,
        canonical_text=text,
    )
    if (
        matrix_fields["risk_test_matrix_availability"] == "not-planned"
        and text
        and not risk_test_matrix_contract_version
    ):
        marker = RISK_TEST_MATRIX_MARKER_RE.search(text)
        if marker is not None:
            try:
                marker_payload = decode_risk_test_matrix_marker(marker.group("payload"))
                matrix_fields = _approved_matrix_channel(
                    contract_version=int(marker_payload["contract_version"]),
                    # Keep the marker's validated JSON-compatible payloads
                    # intact.  The channel parser accepts raw mappings; a
                    # parsed RiskTestMatrixChange is intentionally not a
                    # mapping and would make marker-only recovery fail for
                    # the normal non-empty draft audit case.
                    payload=marker_payload["matrix"],
                    changes_payload=marker_payload["changes"],
                    identity=str(marker_payload["identity"]),
                    source_locator=risk_test_matrix_source_locator or source_locator,
                    boundary_digest=str(marker_payload["identity"]),
                    canonical_text=text,
                )
            except (AgentLoopError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                matrix_fields = _approved_matrix_channel(
                    contract_version=1,
                    payload=None,
                    changes_payload=None,
                    identity=None,
                    source_locator=risk_test_matrix_source_locator or source_locator,
                    diagnostic=f"Matrix unavailable: {exc}",
                )
    if not text or not text.strip():
        if matrix_fields["risk_test_matrix_availability"] == "available":
            return round_metadata.ApprovedPlanContext(
                plan_hash=expected_hash,
                plan_subject=expected_subject,
                source_locator=source_locator,
                availability="available",
                has_matching_candidate=True,
                diagnostic="Canonical approved plan prose omitted; authenticated structured matrix remains available.",
                **matrix_fields,
            )
        return round_metadata.ApprovedPlanContext(
            plan_hash=expected_hash,
            plan_subject=expected_subject,
            source_locator=source_locator,
            availability="unavailable",
            diagnostic="The approved plan text is unavailable; recover the canonical plan record before reviewing.",
            **matrix_fields,
        )
    raw = text.strip()
    actual_hash = _approved_plan_hash(raw)
    actual_subject = round_metadata._plan_subject(raw)
    if expected_hash is not None and actual_hash != expected_hash:
        return round_metadata.ApprovedPlanContext(
            canonical_text=raw,
            plan_hash=actual_hash,
            plan_subject=actual_subject,
            source_locator=source_locator,
            availability="mismatched",
            has_matching_candidate=False,
            diagnostic=(
                f"Recovered plan hash {actual_hash} does not match handoff hash {expected_hash}."
            ),
            **matrix_fields,
        )
    if expected_subject is not None and actual_subject != expected_subject:
        return round_metadata.ApprovedPlanContext(
            canonical_text=raw,
            plan_hash=actual_hash,
            plan_subject=actual_subject,
            source_locator=source_locator,
            availability="mismatched",
            has_matching_candidate=True,
            diagnostic=(
                f"Recovered plan subject {actual_subject} does not match handoff subject {expected_subject}."
            ),
            **matrix_fields,
        )
    scope, deferred = _plan_declarations(raw)
    return round_metadata.ApprovedPlanContext(
        canonical_text=raw,
        plan_hash=actual_hash,
        plan_subject=actual_subject,
        source_locator=source_locator,
        scope=scope,
        deferred_work=deferred,
        availability="available",
        has_matching_candidate=True,
        **matrix_fields,
    )


def _legacy_freeform_plan_candidates(record: round_metadata.PostedRoundRecord) -> tuple[str, ...]:
    """Return raw-text candidates for pre-canonical free-form plan comments.

    Older plan comments stored only the rendered body.  The renderer may have
    replaced a trailing provider-only signature with a model-qualified one, so
    migration recovery considers the visible body, the body without that
    generated signature, and the provider signatures that could have been
    replaced.  The caller still validates the short hash and full subject
    against the durable handoff before accepting any candidate.
    """
    visible = record.body.strip()
    if not visible:
        return ()
    candidates: list[str] = [visible]
    lines = visible.splitlines()
    tail = len(lines)
    while tail > 0 and (
        not lines[tail - 1].strip() or HTML_COMMENT_RE.match(lines[tail - 1])
    ):
        tail -= 1
    if tail <= 0 or not SIGNATURE_RE.match(lines[tail - 1]):
        return tuple(dict.fromkeys(candidates))

    signature_index = tail - 1
    suffix = lines[tail:]
    without_signature = "\n".join([*lines[:signature_index], *suffix]).strip()
    if without_signature:
        candidates.append(without_signature)

    for agent in ("claude", "codex", "gemini", "antigravity"):
        if record.metadata.agent != agent_display_name(agent):
            continue
        signatures = {
            agent_signature(agent),
            agent_signature(agent, model_used=record.metadata.model_used),
        }
        if record.metadata.configured_model:
            configured_label = record.metadata.configured_model
            if record.metadata.configured_effort:
                configured_label = f"{configured_label} ({record.metadata.configured_effort})"
            signatures.add(agent_signature(agent, model_used=configured_label))
        for signature in signatures:
            reconstructed = "\n".join(
                [*lines[:signature_index], f"-- {signature}", *suffix]
            ).strip()
            if reconstructed:
                candidates.append(reconstructed)
        break
    spaced_candidates: list[str] = []
    for candidate in candidates:
        spaced_candidates.append(candidate)
        candidate_lines = candidate.splitlines()
        with_comment_spacing: list[str] = []
        for line in candidate_lines:
            if (
                HTML_COMMENT_RE.match(line)
                and with_comment_spacing
                and with_comment_spacing[-1].strip()
            ):
                with_comment_spacing.append("")
            with_comment_spacing.append(line)
        spaced = "\n".join(with_comment_spacing).strip()
        if spaced:
            spaced_candidates.append(spaced)
    return tuple(dict.fromkeys(spaced_candidates))


@dataclass(frozen=True)
class ApprovedPlanComment:
    """The planner comment that carries an approved plan, for presentation."""

    comment: object
    # Structured steps from the authenticated plan sidecar, or ``None`` when
    # the record has no sidecar bound to the approved rendered plan.
    plan_steps: tuple[str, ...] | None


def _sidecar_plan_steps(metadata: round_metadata.PostedRoundMetadata, rendered_plan: str) -> tuple[str, ...] | None:
    if metadata.assembled_plan_sidecar is None:
        return None
    try:
        sidecar = decode_assembled_plan_sidecar(metadata.assembled_plan_sidecar)
    except AgentLoopError:
        return None
    if (
        sidecar.rendered_plan_identity is not None
        and sidecar.rendered_plan_identity != rendered_plan_identity(rendered_plan)
    ):
        return None
    steps = sidecar.canonical_json.get("plan_steps")
    if not isinstance(steps, list) or not steps or not all(isinstance(step, str) for step in steps):
        return None
    return tuple(steps)


def find_approved_plan_comment(
    comments: Sequence[object],
    *,
    expected_hash: str,
) -> ApprovedPlanComment | None:
    """Return the latest planner comment carrying the approved plan.

    Presentation only (#941): the approval announcement links here and uses
    the structured steps to summarize them instead of repeating them.  Any
    lookup failure yields ``None`` so the announcement degrades to the full
    plan text rather than failing.
    """
    try:
        records = _extract_round_metadata_records(comments, flow="plan")
    except AgentLoopError:
        return None
    for record in reversed(records):
        if record.metadata.role != "coder":
            continue
        raw = record.metadata.canonical_plan or record.metadata.raw_structured_coder_response
        candidates = (raw,) if raw is not None else _legacy_freeform_plan_candidates(record)
        matched = next(
            (
                candidate.strip()
                for candidate in candidates
                if candidate and _approved_plan_hash(candidate) == expected_hash
            ),
            None,
        )
        if matched is None:
            continue
        return ApprovedPlanComment(
            comment=comments[record.index],
            plan_steps=_sidecar_plan_steps(record.metadata, matched),
        )
    return None


def recover_approved_plan_context(
    comments: Sequence[object],
    *,
    expected_hash: str,
    expected_subject: str | None = None,
    expected_matrix_identity: str | None = None,
) -> round_metadata.ApprovedPlanContext:
    """Recover exactly the plan selected by a durable implementation handoff."""
    records = _extract_round_metadata_records(comments, flow="plan")
    candidates: list[tuple[int, str]] = []
    observed_hashes: set[str] = set()
    observed_subjects: set[str] = set()
    subject_rejected = False
    for record in records:
        raw = record.metadata.canonical_plan
        if raw is None:
            # Legacy freeform plan records have no canonical sidecar. Their
            # visible body is only a safe fallback when its raw hash and plan
            # subject match. normalize_freeform_signature may have replaced a
            # provider-only trailing signature, so try the body with that
            # generated signature removed/reconstructed as well.
            raw = record.metadata.raw_structured_coder_response
        raw_candidates = (raw,) if raw is not None else _legacy_freeform_plan_candidates(record)
        # A canonical sidecar is already authoritative; only the migration
        # fallback needs the legacy metadata subject as a second identity
        # check. Existing canonical records may use arbitrary subjects from
        # older metadata versions and must remain recoverable by handoff hash.
        required_subject = (
            expected_subject
            if raw is not None
            else expected_subject or record.metadata.subject
        )
        for candidate in raw_candidates:
            if candidate is None or not candidate.strip():
                continue
            candidate = candidate.strip()
            actual_hash = _approved_plan_hash(candidate)
            observed_hashes.add(actual_hash)
            actual_subject = round_metadata._plan_subject(candidate)
            observed_subjects.add(actual_subject)
            if actual_hash != expected_hash:
                continue
            if required_subject and actual_subject != required_subject:
                subject_rejected = True
                continue
            candidates.append((record.index, candidate))
    if not candidates:
        available = ", ".join(sorted(observed_hashes)) or "none"
        subject_detail = ""
        if subject_rejected:
            subject_detail = (
                " The handoff hash was observed, but no candidate also matched the "
                f"expected plan subject; recovered subjects: "
                f"{', '.join(sorted(observed_subjects)) or 'none'}."
            )
        return round_metadata.ApprovedPlanContext(
            plan_hash=expected_hash,
            plan_subject=expected_subject,
            availability="mismatched" if observed_hashes else "unavailable",
            has_matching_candidate=False,
            diagnostic=(
                f"No canonical approved plan matches handoff hash {expected_hash}; "
                f"recovered plan hashes: {available}.{subject_detail}"
            ),
        )
    unique_texts = {raw for _index, raw in candidates}
    if len(unique_texts) != 1:
        return round_metadata.ApprovedPlanContext(
            plan_hash=expected_hash,
            plan_subject=expected_subject,
            availability="mismatched",
            has_matching_candidate=True,
            diagnostic=(
                f"Multiple divergent canonical plan records match handoff hash {expected_hash}."
            ),
        )
    matching_indices = {index for index, _raw in candidates}
    matching_records = [record for record in records if record.index in matching_indices]
    matrix_presence = [
        round_metadata._risk_test_matrix_metadata_present(record.metadata)
        for record in matching_records
    ]
    matching_matrix_identities = {
        record.metadata.risk_test_matrix_identity
        for record, present in zip(matching_records, matrix_presence)
        if present
    }
    matching_matrix_identity_missing = any(
        present and record.metadata.risk_test_matrix_identity is None
        for record, present in zip(matching_records, matrix_presence)
    )
    matrix_conflict = bool(matching_matrix_identities) and (
        len(matching_matrix_identities) > 1
        or not all(matrix_presence)
        or matching_matrix_identity_missing
    )
    index, raw = candidates[-1]
    matching_record = next((record for record in reversed(records) if record.index == index), None)
    matrix_fields: dict[str, object] = {}
    if matching_record is not None:
        matrix_fields = {
            "risk_test_matrix_contract_version": matching_record.metadata.risk_test_matrix_contract_version,
            "risk_test_matrix_payload": matching_record.metadata.risk_test_matrix_payload,
            "risk_test_matrix_changes_payload": matching_record.metadata.risk_test_matrix_changes_payload,
            "risk_test_matrix_identity": matching_record.metadata.risk_test_matrix_identity,
            "risk_test_matrix_boundary_digest": matching_record.metadata.risk_test_matrix_boundary_digest,
            "risk_test_matrix_source_locator": f"issue comment index {index}",
            "risk_test_matrix_diagnostic": matching_record.metadata.risk_test_matrix_diagnostic,
        }
        if matrix_conflict:
            # Canonical plan identity is still unambiguous. Close only the
            # semantic matrix channel when matching records disagree; do not
            # turn a matrix-only conflict into a whole-plan mismatch.
            matrix_fields.update(
                {
                    "risk_test_matrix_diagnostic": (
                        f"Multiple divergent risk matrix records match approved plan {expected_hash}."
                    ),
                    "risk_test_matrix_payload": None,
                    "risk_test_matrix_changes_payload": (),
                    "risk_test_matrix_identity": None,
                    "risk_test_matrix_boundary_digest": None,
                }
            )
        if expected_matrix_identity is not None and matching_record.metadata.risk_test_matrix_identity != expected_matrix_identity:
            matrix_fields["risk_test_matrix_diagnostic"] = (
                f"Recovered matrix identity does not match handoff identity {expected_matrix_identity}."
            )
            matrix_fields["risk_test_matrix_payload"] = None
            matrix_fields["risk_test_matrix_changes_payload"] = ()
            matrix_fields["risk_test_matrix_identity"] = None
    return make_approved_plan_context(
        raw,
        source_locator=f"issue comment index {index}",
        expected_hash=expected_hash,
        expected_subject=expected_subject,
        **matrix_fields,
    )


def _resume_pr_round(
    comments: Sequence[object],
    *,
    head_sha: str | None,
    configured_reviewers: Sequence[AgentName],
    reconciliation_mode: str = "aggregate",
    trusted_actor: tuple[str, int] | None = None,
    review_unrecorded_head: bool = False,
) -> round_metadata.ResumedReviewRound | None:
    if not head_sha:
        return None
    all_records = _drop_unterminated_evidence_response_records(
        _extract_round_metadata_records(comments, flow="pr")
    )
    if not all_records:
        return None
    # Only authenticated-actor recovery records are admitted; the walk sees the
    # admitted coder-dispatch records, ordinary selection never does.
    walk_records = _admit_recovery_records(all_records, comments, trusted_actor)
    records = tuple(
        record for record in walk_records if record.metadata.phase != round_metadata.CODER_DISPATCH_PHASE
    )
    try:
        return _resume_pr_round_from_records(
            comments,
            walk_records=walk_records,
            records=records,
            head_sha=head_sha,
            configured_reviewers=configured_reviewers,
            reconciliation_mode=reconciliation_mode,
            trusted_actor=trusted_actor,
        )
    except _RecoveryRefusal:
        if not review_unrecorded_head:
            raise
        return _operator_head_review_resume(
            walk_records, records, comments=comments, trusted_actor=trusted_actor
        )


def _conversion_handoff_resume(
    conversion_record: round_metadata.PostedRoundRecord,
    *,
    current_head_records: Sequence[round_metadata.PostedRoundRecord],
    walk_records: Sequence[round_metadata.PostedRoundRecord],
) -> round_metadata.ResumedReviewRound:
    """Resume at a conversion record before any repeat-pass review (#1325)."""
    conversion = conversion_record.metadata
    latest_coder = next(
        (record for record in reversed(current_head_records) if record.metadata.role == "coder"),
        None,
    )
    return round_metadata.ResumedReviewRound(
        round_number=conversion.round_number,
        prior_items=tuple(conversion.prior_items),
        coder_output=(
            latest_coder.metadata.raw_structured_coder_response or latest_coder.body
            if latest_coder is not None
            else None
        ),
        coder_metadata=latest_coder.metadata if latest_coder is not None else None,
        coder_comment_body=latest_coder.body if latest_coder is not None else None,
        completed_reviews=(),
        next_unresolved_item_number=_max_unresolved_item_number_from_records(
            list(walk_records)
        )
        + 1,
        reconciled=True,
        qualification_checkpoint=None,
    )


def _resume_pr_round_from_records(
    comments: Sequence[object],
    *,
    walk_records: Sequence[round_metadata.PostedRoundRecord],
    records: Sequence[round_metadata.PostedRoundRecord],
    head_sha: str,
    configured_reviewers: Sequence[AgentName],
    reconciliation_mode: str,
    trusted_actor: tuple[str, int] | None,
) -> round_metadata.ResumedReviewRound | None:
    records = authority_records(records)
    walk_records = authority_records(walk_records)
    handoff = _live_recovery_handoff(walk_records, head_sha, head_moved=False)
    if handoff is not None:
        return _consume_recovery_handoff(walk_records, handoff, head_moved=False)
    # A checkpoint posted into a closed round never anchors the resume, so a
    # later ledger cannot displace the reconciled transcript (#1553).
    records = drop_inert_post_reconciliation_checkpoints(records)
    from .capture_dispatch import resume_capture_round
    scoped_resume = resume_capture_round(records, comments=comments, trusted_actor=trusted_actor, head=head_sha)
    if scoped_resume is not None:
        return scoped_resume
    current_head_records = [record for record in records if record.metadata.subject == head_sha]
    if current_head_records and current_head_records[-1].metadata.phase in round_metadata.EVIDENCE_TERMINAL_PHASES:
        # The latest current-head record is the single freeze or release
        # comment of #1068: it is the handoff boundary, carrying the ledger,
        # round number, budget and signed-input baseline.
        boundary_record = current_head_records[-1]
        boundary = boundary_record.metadata
        return round_metadata.ResumedReviewRound(
            round_number=boundary.round_number,
            prior_items=_evidence_boundary_ledger(boundary_record, records=records),
            coder_output=None,
            completed_reviews=(),
            next_unresolved_item_number=_max_unresolved_item_number_from_records(records) + 1,
            reconciled=True,
            qualification_checkpoint=None,
            evidence_boundary=boundary,
            evidence_clearances=boundary.evidence_clearances,
        )
    if (
        current_head_records
        and current_head_records[-1].metadata.phase == round_metadata.EVIDENCE_CONVERSION_PHASE
        and current_head_records[-1].metadata.evidence_conversion is not None
    ):
        # The conversion record (#1325) was posted before the same-head repeat
        # started any reviewer: it is the handoff boundary carrying the
        # post-conversion ledger and round.  Its repeat re-runs every reviewer.
        return _conversion_handoff_resume(
            current_head_records[-1],
            current_head_records=current_head_records,
            walk_records=walk_records,
        )
    selection = _select_current_round_records(records, subject=head_sha)
    if selection is None:
        recovered = _recover_unrecorded_pr_head_advance(
            records,
            head_sha=head_sha,
            reconciliation_mode=reconciliation_mode,
            walk_records=walk_records,
            comments=comments,
            trusted_actor=trusted_actor,
        )
        if recovered is not None:
            return recovered
        if _latest_prior_pr_subject_is_coherent(
            records, head_sha=head_sha, comments=comments, trusted_actor=trusted_actor
        ):
            return None
        latest_prior_subject = records[-1].metadata.subject
        raise _RecoveryRefusal(
            "PR head advanced without a recorded coder follow-up and the metadata-backed "
            "handoff could not be recovered safely. "
            f"Current head: {head_sha}. Latest recorded metadata subject: {latest_prior_subject}. "
            + round_metadata.REVIEW_UNRECORDED_HEAD_HINT
            + " Do not hand-author a coder follow-up in the coder's name."
        )
    current_round_records = selection.current_round_records
    anchor_metadata = selection.anchor_record.metadata
    review_handoff = next(
        (
            record
            for record in reversed(records)
            if record.metadata.subject == head_sha
            and record.metadata.phase == round_metadata.HEAD_REVIEW_RECOVERY_PHASE
            and not round_metadata._recovery_record_problems(record.metadata)
        ),
        None,
    )
    # A head-review handoff supersedes every earlier checkpoint on this head.
    superseded_below = review_handoff.index if review_handoff is not None else -1
    if review_handoff is not None:
        # Head-review recovery carries only the ledger, budget and handoff: records
        # published before the handoff (reviewers, coder, reconciliation) that share
        # its head, round and ledger are never replayed as part of the new round.
        current_round_records = tuple(
            record for record in current_round_records if record.index >= superseded_below
        )
    checkpoint_record = _latest_qualification_checkpoint_record(
        records, head_sha=head_sha, min_index=superseded_below
    )
    latest_coder_record = next(
        (
            record
            for record in reversed(current_round_records)
            if record.metadata.role == "coder"
        ),
        None,
    )
    # An admitted conversion record (#1325) reopens this round at the same head.
    # Reviews and the reconciliation before it belong to the pre-conversion
    # pass: they never saw the converted ledger, so the repeat resumes only its
    # own records (#1664).  The round's coder record is kept.
    conversion_boundary = next(
        (
            record
            for record in reversed(records)
            if record.metadata.subject == head_sha
            and record.metadata.phase == round_metadata.EVIDENCE_CONVERSION_PHASE
            and record.metadata.round_number == anchor_metadata.round_number
            and record.metadata.evidence_conversion is not None
            and record.index > superseded_below
        ),
        None,
    )
    reopened_below = conversion_boundary.index if conversion_boundary is not None else -1
    reviewer_records: dict[str, round_metadata.PostedRoundRecord] = {}
    reconciliation_record: round_metadata.PostedRoundRecord | None = None
    configured_reviewer_names = {agent_display_name(agent) for agent in configured_reviewers}
    for record in current_round_records:
        metadata = record.metadata
        if (
            metadata.role == "summary"
            and metadata.phase == "reconciliation"
            and metadata.round_number == anchor_metadata.round_number
            and metadata.subject == head_sha
            and record.index > reopened_below
        ):
            reconciliation_record = record
    for record in current_round_records:
        metadata = record.metadata
        if not is_complete_code_review(metadata) or metadata.agent not in configured_reviewer_names:
            continue
        if record.index <= reopened_below:
            continue
        if _contradictory_evidence_approval(metadata):
            reviewer_records.pop(metadata.agent, None)
        else:
            reviewer_records[metadata.agent] = record
    if conversion_boundary is not None and latest_coder_record is None and not reviewer_records:
        return _conversion_handoff_resume(
            conversion_boundary,
            current_head_records=[r for r in records if r.metadata.subject == head_sha],
            walk_records=walk_records,
        )
    if latest_coder_record is None and not reviewer_records and not any(
        _contradictory_evidence_approval(r.metadata) for r in current_round_records
    ):
        open_prelaunch = (
            anchor_metadata.phase == "scheduler-prelaunch"
            and anchor_metadata.round_number > 1
            and anchor_metadata.scheduler_metadata_status == "valid"
        )
        open_ack_failure = any(
            _is_acknowledgement_failure_boundary(record.metadata)
            for record in current_round_records
        )
        if not (open_prelaunch or open_ack_failure):
            if checkpoint_record is None:
                return None
            # Preserve first-round startup and qualification recovery. A later
            # open prelaunch instead retains its own round, ledger and spool.
            anchor_metadata = checkpoint_record.metadata
            current_round_records = (checkpoint_record,)
            reconciliation_record = None
    prior_items = anchor_metadata.prior_items
    round_number = anchor_metadata.round_number
    ledger_may_be_incomplete = (
        len(anchor_metadata.prior_items) == 0
        and any(
            record.metadata.new_items
            for record in records
            if record.metadata.subject == anchor_metadata.subject
            and record.metadata.round_number < anchor_metadata.round_number
        )
    )
    return round_metadata.ResumedReviewRound(
        round_number=round_number,
        prior_items=prior_items,
        coder_output=(
            latest_coder_record.metadata.raw_structured_coder_response
            or latest_coder_record.body
            if latest_coder_record is not None
            else None
        ),
        coder_metadata=latest_coder_record.metadata if latest_coder_record else None,
        coder_comment_body=latest_coder_record.body if latest_coder_record else None,
        completed_reviews=tuple(reviewer_records[agent_display_name(agent)] for agent in configured_reviewers if agent_display_name(agent) in reviewer_records),
        # Item IDs are never reused: keep the global high-water mark across heads
        # (recovery rounds, and the rounds that follow a recovery push, resume on
        # a head whose own records may carry only low, still-active numbers).
        next_unresolved_item_number=_max_unresolved_item_number_from_records(
            list(walk_records)
        )
        + 1,
        ledger_may_be_incomplete=ledger_may_be_incomplete,
        compact_prior_summaries=(
            latest_coder_record.metadata.compact_prior_summaries
            if latest_coder_record is not None
            else ()
        ),
        reconciled=_pr_round_is_reconciled(
            tuple(
                record
                for record in current_round_records
                if record.index >= max(superseded_below, reopened_below)
            )
            + ((conversion_boundary,) if conversion_boundary is not None else ())
        ),
        local_test_evidence=(
            latest_coder_record.metadata.local_test_evidence
            if latest_coder_record is not None
            else None
        ),
        qualification_checkpoint=_latest_qualification_checkpoint(
            records, head_sha=head_sha, min_index=superseded_below
        ),
        recovery_round_budget=(
            review_handoff.metadata.recovery_round_budget
            if review_handoff is not None
            else None
        ),
        head_review_recovery=(
            review_handoff.metadata.head_review_recovery_source
            if review_handoff is not None
            else None
        ),
        reconciliation_record=reconciliation_record,
    )


def _resume_plan_round(
    comments: Sequence[object],
    *,
    configured_reviewers: Sequence[AgentName],
    min_round_number: int = 1,
) -> tuple[str, round_metadata.ResumedReviewRound] | None:
    records = project_plan_history(_extract_round_metadata_records(comments, flow="plan"))
    if min_round_number > 1:
        # Planning-scope projection (#1390): records of an earlier planning
        # scope never select the resumed plan or its carried ledger.
        records = tuple(
            record for record in records if record.metadata.round_number >= min_round_number
        )
    if not records:
        return None
    latest_coder_record = next((record for record in reversed(records) if record.metadata.role == "coder"), None)
    if latest_coder_record is None:
        return None
    selection = _select_current_round_records(records, subject=latest_coder_record.metadata.subject)
    if selection is None:
        return None
    current_round_records = selection.current_round_records
    anchor_metadata = selection.anchor_record.metadata
    current_plan = latest_coder_record.metadata.canonical_plan or latest_coder_record.body
    coder_output = latest_coder_record.metadata.raw_structured_coder_response or current_plan
    metadata_version = latest_coder_record.metadata.execution_strategy_contract_version
    matrix_metadata_version = latest_coder_record.metadata.risk_test_matrix_contract_version
    semantic_response_form = latest_coder_record.metadata.response_form
    semantic_sidecar_authoritative = semantic_response_form == "semantic-patch-v1"
    if semantic_response_form in {"fresh-plan-state", "legacy-full-state"}:
        # Fresh and legacy full-state publications seed the next semantic
        # revision.  Their authenticated sidecar must bind to the exact
        # canonical Markdown persisted with the round; a subject check alone
        # is insufficient because both the Markdown and subject can be
        # replaced while leaving the sidecar untouched.
        metadata = latest_coder_record.metadata
        if metadata.assembled_plan_sidecar is None or metadata.canonical_plan is None:
            raise AgentLoopError(
                "Authenticated full-state planning metadata is incomplete: "
                "authenticated assembled state and canonical Markdown are both required "
                "for restart."
            )
        try:
            sidecar = decode_assembled_plan_sidecar(metadata.assembled_plan_sidecar)
            if sidecar.response_form != semantic_response_form:
                raise AgentLoopError("full-state sidecar response form mismatch")
            if sidecar.round_number != metadata.round_number:
                raise AgentLoopError("full-state sidecar round mismatch")
            if metadata.aggregate_plan_identity != sidecar.aggregate_identity:
                raise AgentLoopError("full-state sidecar aggregate identity mismatch")
            if (
                sidecar.rendered_plan_identity is None
                or sidecar.rendered_plan_identity
                != rendered_plan_identity(metadata.canonical_plan)
            ):
                raise AgentLoopError(
                    "full-state sidecar rendered-plan identity does not match canonical Markdown"
                )
        except (AgentLoopError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise AgentLoopError(
                "Authenticated full-state planning state is missing or contradictory; "
                f"refusing to resume from an unbound canonical Markdown: {exc}"
            ) from exc
    if semantic_sidecar_authoritative:
        # Semantic rounds resume from the authenticated assembled sidecar. The
        # raw model patch is provenance only and is never reparsed into the
        # canonical plan used by prompts or reviewers.
        metadata = latest_coder_record.metadata
        if metadata.assembled_plan_sidecar is None or metadata.canonical_plan is None:
            raise AgentLoopError(
                "Semantic planning metadata is incomplete: authenticated assembled state "
                "and canonical Markdown are both required for restart."
            )
        try:
            sidecar = decode_assembled_plan_sidecar(metadata.assembled_plan_sidecar)
            if sidecar.response_form != "semantic-patch-v1":
                raise AgentLoopError("semantic sidecar response form mismatch")
            if sidecar.round_number != metadata.round_number:
                raise AgentLoopError("semantic sidecar round mismatch")
            if metadata.aggregate_plan_identity != sidecar.aggregate_identity:
                raise AgentLoopError("semantic sidecar aggregate identity mismatch")
            patch = parse_plan_revision_patch(metadata.raw_patch_provenance or {})
            if patch.base_round_number != metadata.base_round_number:
                raise AgentLoopError("semantic patch base round mismatch")
            if patch.base_state_identity != metadata.base_state_identity:
                raise AgentLoopError("semantic patch base identity mismatch")
            if sidecar.raw_patch != patch.to_payload():
                raise AgentLoopError("semantic sidecar patch provenance mismatch")
            if (
                sidecar.rendered_plan_identity is None
                or sidecar.rendered_plan_identity
                != rendered_plan_identity(metadata.canonical_plan)
            ):
                raise AgentLoopError(
                    "semantic sidecar rendered-plan identity does not match canonical Markdown"
                )
        except (AgentLoopError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise AgentLoopError(
                "Authenticated semantic planning state is missing or contradictory; "
                f"refusing to prompt from raw patch text: {exc}"
            ) from exc
        if round_metadata._plan_subject(metadata.canonical_plan) != metadata.subject:
            raise AgentLoopError(
                "Semantic planning metadata subject does not match canonical Markdown."
            )
    if metadata_version == 1 and not semantic_sidecar_authoritative:
        # A generation-1 plan is identified by its canonical rendered text.
        # Never resume a record whose subject was computed from a different
        # representation (for example, raw response JSON versus rendered plan
        # markdown); doing so would make the next invocation fork the
        # same plan into a new round.
        if latest_coder_record.metadata.canonical_plan is None:
            raise AgentLoopError(
                "Generation-1 planning metadata has no canonical plan text; "
                "repair the handoff or start a new plan round."
            )
        if round_metadata._plan_subject(current_plan) != latest_coder_record.metadata.subject:
            raise AgentLoopError(
                "Generation-1 planning metadata subject does not match its canonical plan; "
                "repair the handoff or start a new plan round."
            )
    if semantic_response_form != "semantic-patch-v1":
        all_bodies = tuple(
            body for comment in comments if isinstance((body := getattr(comment, "body", None)), str)
        )
        fresh_artifact = False
        fresh_matrix_artifact = False
        try:
            raw_payload, _ = json.JSONDecoder().raw_decode(coder_output.lstrip())
            fresh_artifact = (
                isinstance(raw_payload, dict)
                and (
                    "execution_strategy_contract_version" in raw_payload
                    or "execution_recommendation" in raw_payload
                )
            ) or "AGENT_EXECUTION_RECOMMENDATION" in latest_coder_record.body
            fresh_matrix_artifact = (
                isinstance(raw_payload, dict)
                and (
                    "risk_test_matrix_contract_version" in raw_payload
                    or "risk_test_matrix" in raw_payload
                )
            ) or RISK_TEST_MATRIX_MARKER_RE.search(latest_coder_record.body) is not None
        except (AttributeError, json.JSONDecodeError):
            fresh_artifact = "AGENT_EXECUTION_RECOMMENDATION" in latest_coder_record.body
            fresh_matrix_artifact = RISK_TEST_MATRIX_MARKER_RE.search(latest_coder_record.body) is not None
        if fresh_artifact and metadata_version != 1:
            raise AgentLoopError(
                "Fresh generation-1 planning data is present but its round metadata is "
                "missing or not generation 1; restart the planning handoff instead of "
                "downgrading it to legacy-undecided."
            )
        if fresh_matrix_artifact and matrix_metadata_version != 1:
            raise AgentLoopError(
                "Fresh generation-1 risk matrix data is present but its round metadata is "
                "missing or not generation 1; restart the planning handoff instead of "
                "downgrading it to legacy-undecided."
            )
    if metadata_version == 1 and semantic_response_form != "semantic-patch-v1":
        # A generation-1 round is never downgraded to legacy on resume.  The
        # raw response is the provenance source; the visible canonical plan is
        # intentionally markdown and cannot substitute for it.
        from .protocol import (
            validate_structured_plan_revision,
            validate_structured_plan_state,
        )

        try:
            raw_payload, _ = json.JSONDecoder().raw_decode(coder_output.lstrip())
            response_kind = raw_payload.get("kind") if isinstance(raw_payload, dict) else None
            if response_kind == "plan_revision":
                parsed_revision = validate_structured_plan_revision(
                    coder_output,
                    architecture_status_mode="legacy",
                    require_execution_strategy_contract=1,
                    require_risk_test_matrix_contract=(1 if matrix_metadata_version == 1 else 0),
                )
                if parsed_revision is None:
                    raise AgentLoopError(
                        "Generation-1 planning metadata has no recoverable structured response."
                    )
                parsed_recommendation = parsed_revision.execution_recommendation
            elif response_kind == "plan_state":
                parsed = validate_structured_plan_state(
                    coder_output,
                    architecture_status_mode="legacy",
                    require_execution_strategy_contract=1,
                    require_risk_test_matrix_contract=(1 if matrix_metadata_version == 1 else 0),
                )
                if parsed is None:
                    raise AgentLoopError(
                        "Generation-1 planning metadata has no recoverable structured response."
                    )
                parsed_recommendation = parsed.execution_recommendation
            else:
                raise AgentLoopError(
                    "Generation-1 planning metadata has an unknown structured response kind."
                )
            if parsed_recommendation is None:
                raise AgentLoopError(
                    "Generation-1 planning metadata has no execution recommendation."
                )

            # The canonical plan is the public, hashed rendering.  Recover the
            # recommendation sidecar from it (including transport sidecars) and
            # compare it with both the raw response and durable metadata.  This
            # prevents a restart from accepting a changed recommendation that
            # retained only the old short topology summary.
            marker_match = EXECUTION_RECOMMENDATION_MARKER_RE.search(current_plan)
            if marker_match is None:
                marker_match = EXECUTION_RECOMMENDATION_MARKER_RE.search(
                    latest_coder_record.body
                )
            if marker_match is None:
                raise AgentLoopError(
                    "Generation-1 planning metadata has no canonical execution recommendation sidecar."
                )
            from .protocol import parse_execution_recommendation_payload

            sidecar_payload = decode_execution_recommendation_marker(
                marker_match.group("payload"), bodies=all_bodies
            )
            sidecar_recommendation = parse_execution_recommendation_payload(
                sidecar_payload, context="canonical execution_recommendation"
            )
            if (
                not isinstance(latest_coder_record.metadata.execution_strategy_identity, dict)
                or latest_coder_record.metadata.execution_strategy_identity
                != parsed_recommendation.identity()
                or sidecar_recommendation.identity() != parsed_recommendation.identity()
            ):
                raise AgentLoopError(
                    "Generation-1 planning metadata has a missing or mismatched complete "
                    "strategy/topology identity across raw response, canonical sidecar, and metadata."
                )
            if matrix_metadata_version == 1:
                parsed_matrix = (
                    parsed_revision.risk_test_matrix
                    if response_kind == "plan_revision"
                    else parsed.risk_test_matrix
                )
                parsed_changes = (
                    parsed_revision.risk_test_matrix_changes
                    if response_kind == "plan_revision"
                    else parsed.risk_test_matrix_changes
                )
                if parsed_matrix is None or not latest_coder_record.metadata.risk_test_matrix_identity:
                    raise AgentLoopError("Generation-1 planning metadata has no complete risk matrix identity.")
                if risk_test_matrix_identity(parsed_matrix, parsed_changes) != latest_coder_record.metadata.risk_test_matrix_identity:
                    raise AgentLoopError("Generation-1 planning metadata has a mismatched risk matrix identity.")
                matrix_marker_match = RISK_TEST_MATRIX_MARKER_RE.search(current_plan)
                if matrix_marker_match is None:
                    raise AgentLoopError("Generation-1 planning metadata has no canonical risk matrix sidecar.")
                matrix_marker = decode_risk_test_matrix_marker(matrix_marker_match.group("payload"))
                if matrix_marker["identity"] != latest_coder_record.metadata.risk_test_matrix_identity:
                    raise AgentLoopError("Generation-1 canonical risk matrix sidecar does not match metadata.")
        except (AgentLoopError, AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AgentLoopError(
                "Generation-1 planning round metadata is missing or has a malformed "
                "execution strategy contract; repair the handoff or start a new plan round."
            ) from exc
    ledger_may_be_incomplete = (
        len(anchor_metadata.prior_items) == 0
        and any(
            record.metadata.new_items
            for record in records
            if record.metadata.subject == anchor_metadata.subject
            and record.metadata.round_number < anchor_metadata.round_number
        )
    )
    reviewer_records: dict[str, round_metadata.PostedRoundRecord] = {}
    configured_reviewer_names = {agent_display_name(agent) for agent in configured_reviewers}
    for record in current_round_records:
        metadata = record.metadata
        if not is_complete_code_review(metadata) or metadata.agent not in configured_reviewer_names:
            continue
        reviewer_records[metadata.agent] = record
    settled_new_items: list[UnresolvedReviewItem] = []
    settled_item_ids: set[str] = set()
    for record in current_round_records:
        for item in record.metadata.new_items:
            if item.item_id in settled_item_ids:
                continue
            settled_item_ids.add(item.item_id)
            settled_new_items.append(item)
    return (
        current_plan,
        round_metadata.ResumedReviewRound(
            round_number=anchor_metadata.round_number,
            prior_items=anchor_metadata.prior_items,
            coder_output=coder_output,
            completed_reviews=tuple(reviewer_records[agent_display_name(agent)] for agent in configured_reviewers if agent_display_name(agent) in reviewer_records),
            next_unresolved_item_number=_max_unresolved_item_number_from_records(
                [record for record in records if record.metadata.subject == anchor_metadata.subject]
            )
            + 1,
            ledger_may_be_incomplete=ledger_may_be_incomplete,
            compact_prior_summaries=latest_coder_record.metadata.compact_prior_summaries,
            reconciled=_plan_round_is_reconciled(current_round_records),
            coder_metadata=latest_coder_record.metadata,
            local_test_evidence=latest_coder_record.metadata.local_test_evidence,
            current_round_new_items=tuple(settled_new_items),
        ),
    )


@dataclass(frozen=True)
class ResumedDiscussState:
    done: bool
    round_history: tuple[tuple[ParsedDiscussResponse, ...], ...]
    next_round_number: int
    prior_round_agenda: tuple[str, ...] = ()
    prior_analyzer_agenda: ParsedDiscussAgenda | None = None
    prior_round_synthesis: ParsedDiscussRoundSynthesis | None = None
    final_synthesis: ParsedDiscussFinalSynthesis | None = None
    synthesis_provenance: dict | None = None
    in_progress_votes: dict[str, ParsedDiscussResponse] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.in_progress_votes is None:
            object.__setattr__(self, "in_progress_votes", {})


def _decode_analyzer_agenda(raw: str | None) -> ParsedDiscussAgenda | None:
    """Reparse a persisted analyzer agenda; corrupt/legacy payloads resume in plain mode."""
    if not raw:
        return None
    try:
        return parse_structured_discuss_agenda(raw)
    except AgentLoopError:
        return None


def _decode_round_synthesis(raw: str | None) -> ParsedDiscussRoundSynthesis | None:
    """Decode optional canonical synthesis without poisoning a valid resume."""
    if not raw or len(raw.encode("utf-8")) > 16_000:
        return None
    try:
        return parse_canonical_discuss_round_synthesis(raw)
    except (AgentLoopError, TypeError, ValueError):
        return None


def _decode_final_synthesis(raw: str | None) -> ParsedDiscussFinalSynthesis | None:
    """Decode an optional canonical final synthesis without poisoning resume."""
    if not raw or len(raw.encode("utf-8")) > 16_000:
        return None
    try:
        return parse_canonical_discuss_final_synthesis(raw)
    except (AgentLoopError, TypeError, ValueError):
        return None


def _decode_discuss_vote(
    record: round_metadata.PostedRoundRecord, *, round_number: int, reviewer_workdirs: Mapping[str, Path]
) -> ParsedDiscussResponse:
    text = record.metadata.raw_structured_coder_response or record.body
    if record.metadata.result_mode == "answer":
        try:
            vote = parse_structured_discuss_answer(
                text, reviewer=record.metadata.agent, round_number=round_number
            )
        except AgentLoopError:
            # Only persisted round metadata gets the explicitly opt-in legacy
            # decoder. Its exact-key validation still rejects malformed/mixed
            # payloads rather than hiding a corrupt transcript.
            vote = parse_legacy_structured_discuss_answer(
                text, reviewer=record.metadata.agent, round_number=round_number
            )
    else:
        vote = parse_structured_discuss_review(text, reviewer=record.metadata.agent, round_number=round_number)
    if vote is None:
        raise AgentLoopError(
            f"Could not decode discuss response metadata for {record.metadata.agent} "
            f"(round {round_number}); the posted comment's structured payload is missing "
            "or invalid."
        )
    workdir = reviewer_workdirs.get(record.metadata.agent)
    if workdir is None:
        raise AgentLoopError(
            f"Discuss round metadata references reviewer {record.metadata.agent!r} "
            "with no known assigned checkout (it is not in the currently configured "
            "reviewers); a persisted vote from it cannot be checkout-validated."
        )
    validate_checkout_inspected_evidence(vote.evidence_claims, assigned_workdir=workdir)
    return vote


def _resume_discuss_round(
    comments: Sequence[object],
    *,
    subject: str,
    configured_reviewers: Sequence[AgentName],
    reviewer_workdirs: Mapping[str, Path],
    result_mode: str = "triage",
) -> ResumedDiscussState | None:
    records = _extract_round_metadata_records(comments, flow="discuss")
    subject_records = [record for record in records if record.metadata.subject == subject]
    if not subject_records:
        return None
    stored_modes = {record.metadata.result_mode for record in subject_records}
    if stored_modes != {result_mode}:
        raise AgentLoopError(
            "Discuss transcript result mode conflicts with the requested mode; "
            "use the same --discuss-result-mode used to create the transcript."
        )
    configured_reviewer_names = [agent_display_name(agent) for agent in configured_reviewers]
    rounds: dict[int, list[round_metadata.PostedRoundRecord]] = {}
    for record in subject_records:
        rounds.setdefault(record.metadata.round_number, []).append(record)

    round_history: list[tuple[ParsedDiscussResponse, ...]] = []
    prior_round_agenda: tuple[str, ...] = ()
    prior_analyzer_agenda: ParsedDiscussAgenda | None = None
    prior_round_synthesis: ParsedDiscussRoundSynthesis | None = None
    synthesis_provenance: dict | None = None
    next_round_number = 1
    in_progress_votes: dict[str, ParsedDiscussResponse] = {}
    for round_number in sorted(rounds):
        round_records = rounds[round_number]
        summary_record = next(
            (record for record in round_records if record.metadata.role == "summary"), None
        )
        debater_records: dict[str, round_metadata.PostedRoundRecord] = {}
        for record in round_records:
            if record.metadata.role == "debater":
                debater_records[record.metadata.agent] = record
        if summary_record is not None:
            # Debaters recorded as failed in a partial round (#475) post no
            # comment; their summary metadata accounts for the gap.
            failed_by_name = dict(summary_record.metadata.failed_debaters)
            missing = [
                name
                for name in configured_reviewer_names
                if name not in debater_records and name not in failed_by_name
            ]
            if missing:
                raise AgentLoopError(
                    "Discuss round metadata is inconsistent: round "
                    f"{round_number} has a summary comment but is missing debater "
                    f"comments for: {', '.join(missing)}."
                )
            votes = tuple(
                _decode_discuss_vote(
                    debater_records[name], round_number=round_number, reviewer_workdirs=reviewer_workdirs
                )
                if name in debater_records
                else (
                    failed_discuss_answer_placeholder(name, failed_by_name[name])
                    if result_mode == "answer"
                    else failed_discuss_review_placeholder(name, failed_by_name[name])
                )
                for name in configured_reviewer_names
            )
            round_history.append(votes)
            prior_round_agenda = summary_record.metadata.agenda
            prior_analyzer_agenda = _decode_analyzer_agenda(summary_record.metadata.analyzer_response)
            prior_round_synthesis = _decode_round_synthesis(summary_record.metadata.round_synthesis)
            final_synthesis = _decode_final_synthesis(summary_record.metadata.final_synthesis)
            synthesis_provenance = summary_record.metadata.synthesis_provenance
            if summary_record.metadata.is_final:
                return ResumedDiscussState(
                    done=True,
                    round_history=tuple(round_history),
                    next_round_number=round_number,
                    prior_round_agenda=prior_round_agenda,
                    prior_analyzer_agenda=prior_analyzer_agenda,
                    prior_round_synthesis=prior_round_synthesis,
                    final_synthesis=final_synthesis,
                    synthesis_provenance=synthesis_provenance,
                )
            next_round_number = round_number + 1
        else:
            next_round_number = round_number
            for name, record in debater_records.items():
                in_progress_votes[name] = _decode_discuss_vote(
                    record, round_number=round_number, reviewer_workdirs=reviewer_workdirs
                )
            break
    return ResumedDiscussState(
        done=False,
        round_history=tuple(round_history),
        next_round_number=next_round_number,
        prior_round_agenda=prior_round_agenda,
        prior_analyzer_agenda=prior_analyzer_agenda,
        prior_round_synthesis=prior_round_synthesis,
        final_synthesis=None,
        synthesis_provenance=synthesis_provenance,
        in_progress_votes=in_progress_votes,
    )
