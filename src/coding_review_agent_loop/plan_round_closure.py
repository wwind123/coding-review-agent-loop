"""Closed planning history projections; raw records remain available for lineage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .round_state import PostedRoundRecord


# Planning summary records that are posted *before* reviewer reconciliation.
# Staged planning writes a `scheduler-prelaunch` record at the top of every
# round and a `plan-phase-advance` record before a reviewer-only round, so an
# interruption at either checkpoint must not be read as a settled round
# (#905, from #841).
PRE_RECONCILIATION_PLAN_SUMMARY_PHASES = frozenset(
    {"scheduler-prelaunch", "plan-phase-advance"}
)


def _plan_round_is_reconciled(records: Sequence[PostedRoundRecord]) -> bool:
    """True when the round holds an actual planning reconciliation checkpoint.

    The reconciliation record carries `phase="reconciliation"`.  A legacy
    summary record written before the phase labels existed carries the default
    `authoritative` phase and still counts, so historical rounds resume exactly
    as they did before staged planning existed.
    """
    return any(
        record.metadata.flow == "plan"
        and record.metadata.role == "summary"
        and record.metadata.phase in {"reconciliation", "authoritative"}
        for record in records
    )


def plan_round_key(record: PostedRoundRecord) -> tuple[str, int]:
    return record.metadata.subject, record.metadata.round_number


def project_plan_history(records: Sequence[PostedRoundRecord]) -> tuple[PostedRoundRecord, ...]:
    """Drop only idle planning checkpoints after settlement, keeping used openings."""
    closed: set[tuple[str, int]] = set()
    idle: set[int] = set()
    ordered = sorted(records, key=lambda record: record.index)
    for position, record in enumerate(ordered):
        metadata = record.metadata
        if metadata.flow != "plan":
            continue
        key = plan_round_key(record)
        if _plan_round_is_reconciled((record,)):
            closed.add(key)
        elif metadata.role == "summary" and metadata.phase == "scheduler-prelaunch" and key in closed:
            if not any(
                later.metadata.flow == "plan"
                and plan_round_key(later) == key
                and later.metadata.role == "reviewer"
                for later in ordered[position + 1:]
            ):
                idle.add(record.index)
    return tuple(record for record in records if record.index not in idle)


def neutral_plan_rollover_indices(records: Sequence[PostedRoundRecord]) -> frozenset[int]:
    """Identify authority-free staged advances, including pre-fix persisted ones.

    This is a lineage compatibility rule only. Retained work in the settled
    predecessor round still needs its own board and scheduler validation; it
    does not widen the separate amendment activation boundary.
    """
    ordered = sorted(
        (record for record in project_plan_history(records) if record.metadata.flow == "plan"),
        key=lambda record: record.index,
    )
    eligible: set[int] = set()
    for position, record in enumerate(ordered):
        meta = record.metadata
        if not (
            meta.role == "summary" and meta.agent == "Orchestrator"
            and meta.phase == "plan-phase-advance"
            and meta.scheduler_metadata_status == "absent"
            and meta.scheduler_contract is None
            and meta.scheduler_issue_digest is None
            and not meta.scheduler_stall_reset and not meta.scheduler_step_back_deferral
            and meta.scheduler_execution_mode is None
            and meta.plan_board_root is None
            and meta.plan_board_identity is None
            and meta.plan_board_binding_init is None
            and meta.reviewer_board_amendment_digest is None
            and meta.reviewer_board_change_digest is None
            and not meta.new_items and not meta.dispositions
            and not meta.evidence_clearances
            and not meta.evidence_requests
            and meta.evidence_freeze is None and meta.evidence_release is None
        ):
            continue
        settled = []
        for previous in reversed(ordered[:position]):
            if plan_round_key(previous) != (meta.subject, meta.round_number - 1):
                break
            settled.append(previous)
        if not _plan_round_is_reconciled(settled):
            continue
        if meta.plan_candidate_key is not None and any(
            previous.metadata.plan_candidate_key is not None
            and previous.metadata.plan_candidate_key != meta.plan_candidate_key
            for previous in settled
        ):
            continue
        eligible.add(record.index)
    return frozenset(eligible)


def select_plan_round_anchor(records: Sequence[PostedRoundRecord]) -> PostedRoundRecord:
    """Reconciliation owns the ledger even when a later used checkpoint exists."""
    latest = records[-1]
    if latest.metadata.flow == "plan":
        for record in reversed(records):
            if plan_round_key(record) == plan_round_key(latest) and _plan_round_is_reconciled((record,)):
                return record
    return latest


def plan_amendment_activation_round(records, lineage, execution_round: int) -> int:
    """Recognize only a neutral reviewer-only advance immediately after closed N.

    Callers supply admitted, scope-projected history. A pending amendment must
    precede the advance; scheduler/reviewer work after settlement invalidates
    this narrow restart boundary. Normal activation validation stays unchanged.
    """
    pending = lineage.pending_amendments
    if not pending or execution_round <= 1 or not records:
        return execution_round
    ordered = sorted(project_plan_history(records), key=lambda record: record.index)
    advance = ordered[-1]
    meta = advance.metadata
    if not (
        meta.flow == "plan" and meta.role == "summary"
        and meta.phase == "plan-phase-advance"
        and meta.round_number == execution_round
        and meta.scheduler_contract is None
        and meta.reviewer_board_amendment_digest is None
        and all(amendment.comment_index < advance.index for amendment in pending)
    ):
        return execution_round
    predecessor = next((
        record for record in reversed(ordered[:-1])
        if record.metadata.flow == "plan"
    ), None)
    if predecessor is None or not (
        predecessor.metadata.subject == meta.subject
        and predecessor.metadata.round_number == execution_round - 1
        and _plan_round_is_reconciled((predecessor,))
        and all(amendment.effective_from_round == execution_round - 1 for amendment in pending)
    ):
        return execution_round
    return execution_round - 1


def attach_plan_rollover_metadata(body, metadata, *, staged: bool):
    """A staged advance is not an amendment use or a scheduler decision."""
    from .round_state import _attach_round_metadata, set_plan_board_stamp, reset_plan_board_stamp

    if not staged:
        return _attach_round_metadata(body, metadata)
    token = set_plan_board_stamp(None)
    try:
        return _attach_round_metadata(body, metadata)
    finally:
        reset_plan_board_stamp(token)
