"""Same-round reviewer visibility, shared by the launch guards and recovery (#1156).

Panel independence is about what a launching reviewer can *read*, not about
which scheduler phase a record was published under.  The selector here returns
every same-round reviewer publication already public; one narrowly defined
record class (the primary's approved review that gates a panel launch) is
excluded because the panel is designed to read it.

A trusted record that reopens the round at the same head (an admitted
evidence conversion, #1664) may be passed as ``reopening_boundary_index``:
publications at or before it belong to the earlier pass, which the reopened
pass legitimately reads, so they are not same-round peers.  The index is
always an index into the same ``records`` history that is passed in. Failed
acknowledgement repairs automatically bound visibility using resume's predicate.
"""

from __future__ import annotations

from collections.abc import Sequence

from .round_state import PostedRoundRecord, _is_acknowledgement_failure_boundary

# Scheduler phases whose launch is gated on the primary's approved review.
GATED_LAUNCH_PHASES = frozenset({"secondary-audit", "final-secondary-sweep"})
PRIMARY_APPROVAL_OPENING = "primary-approval"


def acknowledgement_failure_boundary_index(
    records: Sequence[PostedRoundRecord], *, flow: str, round_number: int, subject: str
) -> int | None:
    """The latest failed-repair boundary in this exact history, as used by resume."""
    return max(
        (record.index for record in records
         if record.metadata.flow == flow
         and record.metadata.round_number == round_number
         and record.metadata.subject == subject
         and _is_acknowledgement_failure_boundary(record.metadata)),
        default=None,
    )


def same_round_public_publications(
    records: Sequence[PostedRoundRecord],
    *,
    flow: str,
    round_number: int,
    subject: str,
    reviewer_names: Sequence[str],
    reopening_boundary_index: int | None = None,
) -> list[PostedRoundRecord]:
    """Every published same-round reviewer record after the round's latest coder record.

    Reads the complete extracted history: no ledger-signature filter, no
    last-record-per-reviewer dedupe, no ``scheduler_phase`` filter, and no
    reconciliation boundary.  A ``reopening_boundary_index`` raises the lower
    bound; coder, failed acknowledgement repair and reopening boundaries combine
    by max. Failed-repair boundaries use the same predicate as resume.
    """
    coder_boundary = max(
        (
            record.index for record in records
            if record.metadata.flow == flow
            and record.metadata.role == "coder"
            and record.metadata.round_number == round_number
            and record.metadata.subject == subject
        ),
        default=-1,
    )
    failure_boundary = acknowledgement_failure_boundary_index(
        records, flow=flow, round_number=round_number, subject=subject,
    )
    lower_bound = max(
        coder_boundary,
        failure_boundary if failure_boundary is not None else -1,
        reopening_boundary_index if reopening_boundary_index is not None else -1,
    )
    return [
        record for record in records
        if record.metadata.flow == flow
        and record.metadata.round_number == round_number
        and record.metadata.subject == subject
        and record.metadata.agent in reviewer_names
        and record.metadata.role not in {"summary", "coder"}
        and record.metadata.phase == "publication"
        and record.index > lower_bound
    ]


def primary_gate_records(
    records: Sequence[PostedRoundRecord],
    *,
    opening_source: str | None,
    flow: str,
    round_number: int,
    subject: str,
    primary_reviewer: str | None,
    launch_phase: str | None,
    launching: Sequence[str],
    checkpoint_index: int | None,
    reopening_boundary_index: int | None = None,
) -> list[PostedRoundRecord]:
    """The primary's approved review that gates the current panel launch.

    Excluded from the peer set only when the primary-approval opening is in
    force, the launch is a secondary-audit or final-secondary-sweep batch that
    does not include the primary, and the review was posted before this
    invocation's own scheduler checkpoint.
    """
    if (
        primary_reviewer is None
        or checkpoint_index is None
        or opening_source != PRIMARY_APPROVAL_OPENING
        or launch_phase not in GATED_LAUNCH_PHASES
        or primary_reviewer in launching
    ):
        return []
    return [
        record
        for record in same_round_public_publications(
            records, flow=flow, round_number=round_number, subject=subject,
            reviewer_names=(primary_reviewer,),
            reopening_boundary_index=reopening_boundary_index,
        )
        if record.metadata.state == "approved" and record.index < checkpoint_index
    ]


def latest_round_checkpoint_index(
    records: Sequence[PostedRoundRecord], *, flow: str, round_number: int, subject: str
) -> int | None:
    """Index of the round's latest scheduler-prelaunch checkpoint, if any."""
    from .plan_round_closure import project_plan_history

    records = project_plan_history(records)
    indexes = [
        record.index for record in records
        if record.metadata.flow == flow
        and record.metadata.role == "summary"
        and record.metadata.phase == "scheduler-prelaunch"
        and record.metadata.round_number == round_number
        and record.metadata.subject == subject
    ]
    return max(indexes) if indexes else None


def visible_peer_names(
    records: Sequence[PostedRoundRecord],
    *,
    opening_source: str | None,
    flow: str,
    round_number: int,
    subject: str,
    reviewer_names: Sequence[str],
    primary_reviewer: str | None,
    launch_phase: str | None,
    launching: Sequence[str],
    checkpoint_index: int | None,
    reopening_boundary_index: int | None = None,
) -> set[str]:
    """Reviewers with a same-round public publication, minus the gating record."""
    publications = same_round_public_publications(
        records, flow=flow, round_number=round_number, subject=subject,
        reviewer_names=reviewer_names,
        reopening_boundary_index=reopening_boundary_index,
    )
    gates = {
        record.index for record in primary_gate_records(
            records, opening_source=opening_source, flow=flow,
            round_number=round_number, subject=subject,
            primary_reviewer=primary_reviewer, launch_phase=launch_phase,
            launching=launching, checkpoint_index=checkpoint_index,
            reopening_boundary_index=reopening_boundary_index,
        )
    }
    return {
        record.metadata.agent for record in publications if record.index not in gates
    }
