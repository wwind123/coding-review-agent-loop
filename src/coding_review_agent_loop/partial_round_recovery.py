"""Operator recovery list for a partially published review round (#1142).

When the independence guard refuses to run a reviewer beside a public
same-round peer, the operator must delete the round's already-posted records so
the whole round runs again.  This module works out *which* comments, from the
same facts resume and the guard read, and verifies the answer by simulating
resume on the history that would remain.  It never changes the guard's
decision and never deletes anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .errors import AgentLoopError
from .github import IssueComment, strip_bot_login_suffix
from .round_state import (
    PRE_RECONCILIATION_PLAN_SUMMARY_PHASES,
    PostedRoundRecord,
    _extract_round_metadata_records,
)
from .round_transport import (
    ROUND_RESUME_MARKER_RE,
    attachment_keys,
    is_round_transport_sidecar,
    referenced_attachment_keys,
)


@dataclass(frozen=True)
class RecoveryTarget:
    label: str
    comment_id: int | None
    url: str | None
    author: str | None
    created_at: str | None


@dataclass(frozen=True)
class PartialRoundRecovery:
    verified: bool
    targets: tuple[RecoveryTarget, ...] = ()
    uncertain: tuple[RecoveryTarget, ...] = ()
    reason: str | None = None


def _target(label: str, comment: IssueComment) -> RecoveryTarget:
    return RecoveryTarget(
        label=label, comment_id=comment.comment_id, url=comment.url,
        author=comment.author, created_at=comment.created_at,
    )


def _comment_key(comment: IssueComment) -> tuple[str | None, str | None, str | None]:
    return (strip_bot_login_suffix(comment.author), comment.created_at, comment.body)


def _snapshot_covered(
    snapshot: Sequence[IssueComment], rest: Sequence[IssueComment]
) -> bool:
    """The REST history is non-empty and holds every snapshot round record."""
    if not rest:
        return False
    keys = {_comment_key(comment) for comment in rest}
    return all(
        _comment_key(comment) in keys
        for comment in snapshot
        if isinstance(comment.body, str) and ROUND_RESUME_MARKER_RE.search(comment.body)
    )


def _publication_records(
    records: Sequence[PostedRoundRecord],
    *,
    flow: str,
    round_number: int,
    subject: str,
    scheduler_phase: str | None,
    reviewer_names: Sequence[str],
) -> list[PostedRoundRecord]:
    return [
        record for record in records
        if record.metadata.flow == flow
        and record.metadata.round_number == round_number
        and record.metadata.subject == subject
        and record.metadata.agent in reviewer_names
        and record.metadata.role != "summary"
        and record.metadata.role != "coder"
        and record.metadata.phase == "publication"
        and (flow != "pr" or record.metadata.scheduler_phase == scheduler_phase)
    ]


def _reconciling_summaries(
    records: Sequence[PostedRoundRecord], *, flow: str, round_number: int, subject: str
) -> list[PostedRoundRecord]:
    return [
        record for record in records
        if record.metadata.flow == flow
        and record.metadata.role == "summary"
        and record.metadata.round_number == round_number
        and record.metadata.subject == subject
        and record.metadata.phase not in PRE_RECONCILIATION_PLAN_SUMMARY_PHASES
    ]


def _plan(
    comments: Sequence[IssueComment],
    *,
    flow: str,
    round_number: int,
    subject: str,
    scheduler_phase: str | None,
    reviewer_names: Sequence[str],
    widen: bool,
) -> tuple[dict[int, str], dict[int, str], str | None]:
    """Return (targets, uncertain attachments, uncertainty reason) by comment index."""
    records = _extract_round_metadata_records(comments, flow=flow)
    publications = _publication_records(
        records, flow=flow, round_number=round_number, subject=subject,
        scheduler_phase=scheduler_phase, reviewer_names=reviewer_names,
    )
    targets: dict[int, str] = {
        record.index: f"{record.metadata.agent} review (round {round_number})"
        for record in publications
    }
    for record in _reconciling_summaries(
        records, flow=flow, round_number=round_number, subject=subject
    ):
        targets[record.index] = f"round {round_number} reconciliation"
    if widen:
        for record in records:
            if (
                record.metadata.round_number == round_number
                and record.metadata.subject == subject
                and record.metadata.role != "coder"
                and record.metadata.phase not in PRE_RECONCILIATION_PLAN_SUMMARY_PHASES
                and (record.metadata.phase == "publication" or record.metadata.role == "summary")
                and (
                    record.metadata.scheduler_phase == scheduler_phase
                    or record.metadata.role == "summary"
                )
            ):
                targets.setdefault(record.index, f"round {round_number} record")
    if not targets:
        return {}, {}, None

    # The last durable record before the round began bounds orphan attribution.
    round_indexes = {
        record.index for record in records
        if record.metadata.flow == flow
        and record.metadata.round_number == round_number
        and record.metadata.subject == subject
    }
    first_in_round = min(round_indexes | set(targets))
    earlier = [
        record.index for record in records
        if record.index < first_in_round and record.index not in round_indexes
    ]
    lower_bound = max(earlier) if earlier else None
    if scheduler_phase is not None:
        openings = [
            record.index for record in records
            if record.metadata.role == "summary"
            and record.metadata.phase == "scheduler-prelaunch"
            and record.metadata.round_number == round_number
            and record.metadata.subject == subject
            and record.index < min(targets)
        ]
        if openings:
            lower_bound = max(openings)

    attachments = [
        index for index, comment in enumerate(comments)
        if isinstance(comment.body, str) and is_round_transport_sidecar(comment.body)
    ]
    attachment_set = set(attachments)
    retained_refs: set[tuple[str, str]] = set()
    targeted_refs: set[tuple[str, str]] = set()
    for index, comment in enumerate(comments):
        if index in attachment_set or not isinstance(comment.body, str):
            continue
        refs = referenced_attachment_keys(comment.body)
        (targeted_refs if index in targets else retained_refs).update(refs)

    def referenced(keys: set[tuple[str, str]], item: tuple[str, str]) -> bool:
        return item in keys or (item[0], "*") in keys

    round_authors = {
        strip_bot_login_suffix(comments[i].author) for i in targets
    }
    uncertain: dict[int, str] = {}
    for index in attachments:
        keys = attachment_keys(comments[index].body or "")
        if any(referenced(retained_refs, key) for key in keys):
            continue
        if any(referenced(targeted_refs, key) for key in keys):
            targets[index] = "attachment of a listed comment"
            continue
        # Unreferenced: attribute by the round interval and author.
        if lower_bound is not None:
            if index > lower_bound and strip_bot_login_suffix(comments[index].author) in round_authors:
                targets[index] = "unreferenced attachment"
            elif index > lower_bound:
                uncertain[index] = "unreferenced attachment (author differs)"
        elif index < first_in_round:
            uncertain[index] = "unreferenced attachment (possibly from this round, not verified)"
        elif strip_bot_login_suffix(comments[index].author) in round_authors:
            targets[index] = "unreferenced attachment"
    reason = "unreferenced attachments could not be attributed to this round" if uncertain else None
    return targets, uncertain, reason


def compute_partial_round_recovery(
    *,
    snapshot: Sequence[IssueComment],
    read_rest: Callable[[], Sequence[IssueComment]],
    flow: str,
    round_number: int,
    subject: str,
    scheduler_phase: str | None,
    reviewer_names: Sequence[str],
    resume: Callable[[Sequence[IssueComment]], object],
) -> PartialRoundRecovery:
    """The comments whose deletion lets the round run again, verified or provisional."""
    complete = True
    try:
        comments: Sequence[IssueComment] = tuple(read_rest())
        if not _snapshot_covered(snapshot, comments):
            raise AgentLoopError("incomplete comment history")
    except AgentLoopError:
        comments = tuple(snapshot)
        complete = False
    try:
        return _compute(
            comments, complete=complete, flow=flow, round_number=round_number,
            subject=subject, scheduler_phase=scheduler_phase,
            reviewer_names=reviewer_names, resume=resume,
        )
    except AgentLoopError:
        return PartialRoundRecovery(
            verified=False,
            reason="the round's comments could not be analysed",
        )


def _compute(
    comments: Sequence[IssueComment],
    *,
    complete: bool,
    flow: str,
    round_number: int,
    subject: str,
    scheduler_phase: str | None,
    reviewer_names: Sequence[str],
    resume: Callable[[Sequence[IssueComment]], object],
) -> PartialRoundRecovery:
    def check(targets: dict[int, str]) -> bool:
        if not targets:
            return False
        remaining = [c for i, c in enumerate(comments) if i not in targets]
        try:
            resume(remaining)
            records = _extract_round_metadata_records(remaining, flow=flow)
        except AgentLoopError:
            return False
        if _publication_records(
            records, flow=flow, round_number=round_number, subject=subject,
            scheduler_phase=scheduler_phase, reviewer_names=reviewer_names,
        ):
            return False
        return not _reconciling_summaries(
            records, flow=flow, round_number=round_number, subject=subject
        )

    verified = False
    targets: dict[int, str] = {}
    uncertain: dict[int, str] = {}
    reason: str | None = None
    for widen in (False, True):
        targets, uncertain, reason = _plan(
            comments, flow=flow, round_number=round_number, subject=subject,
            scheduler_phase=scheduler_phase, reviewer_names=reviewer_names, widen=widen,
        )
        if check(targets):
            verified = True
            break
    if not verified:
        reason = reason or "resume could not be confirmed on the history that would remain"
    if uncertain:
        verified = False
    if not complete:
        verified = False
        reason = "comment ids could not be read completely"
    if any(comments[i].comment_id is None or not comments[i].url for i in targets):
        verified = False
        reason = reason or "a required comment has no id or url"
    return PartialRoundRecovery(
        verified=verified,
        targets=tuple(_target(label, comments[i]) for i, label in sorted(targets.items())),
        uncertain=tuple(_target(label, comments[i]) for i, label in sorted(uncertain.items())),
        reason=None if verified else reason,
    )


__all__ = [
    "PartialRoundRecovery", "RecoveryTarget", "compute_partial_round_recovery",
]
