"""Same-round reviewer visibility, shared by the launch guards and recovery (#1156).

Panel independence is about what a launching reviewer can *read*, not about
which scheduler phase a record was published under.  The selector here returns
every same-round reviewer publication already public; one narrowly defined
record class (the primary's approved review that gates a panel launch) is
excluded because the panel is designed to read it.
"""

from __future__ import annotations

from collections.abc import Sequence

from .round_state import PostedRoundRecord

# Scheduler phases whose launch is gated on the primary's approved review.
GATED_LAUNCH_PHASES = frozenset({"secondary-audit", "final-secondary-sweep"})
PRIMARY_APPROVAL_OPENING = "primary-approval"


def same_round_public_publications(
    records: Sequence[PostedRoundRecord],
    *,
    flow: str,
    round_number: int,
    subject: str,
    reviewer_names: Sequence[str],
) -> list[PostedRoundRecord]:
    """Every published same-round reviewer record after the round's latest coder record.

    Reads the complete extracted history: no ledger-signature filter, no
    last-record-per-reviewer dedupe, no ``scheduler_phase`` filter, and no
    reconciliation boundary.
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
    return [
        record for record in records
        if record.metadata.flow == flow
        and record.metadata.round_number == round_number
        and record.metadata.subject == subject
        and record.metadata.agent in reviewer_names
        and record.metadata.role not in {"summary", "coder"}
        and record.metadata.phase == "publication"
        and record.index > coder_boundary
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
        )
        if record.metadata.state == "approved" and record.index < checkpoint_index
    ]


def latest_round_checkpoint_index(
    records: Sequence[PostedRoundRecord], *, flow: str, round_number: int, subject: str
) -> int | None:
    """Index of the round's latest scheduler-prelaunch checkpoint, if any."""
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
) -> set[str]:
    """Reviewers with a same-round public publication, minus the gating record."""
    publications = same_round_public_publications(
        records, flow=flow, round_number=round_number, subject=subject,
        reviewer_names=reviewer_names,
    )
    gates = {
        record.index for record in primary_gate_records(
            records, opening_source=opening_source, flow=flow,
            round_number=round_number, subject=subject,
            primary_reviewer=primary_reviewer, launch_phase=launch_phase,
            launching=launching, checkpoint_index=checkpoint_index,
        )
    }
    return {
        record.metadata.agent for record in publications if record.index not in gates
    }
