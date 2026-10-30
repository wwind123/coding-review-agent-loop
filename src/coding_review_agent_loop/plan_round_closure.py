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
