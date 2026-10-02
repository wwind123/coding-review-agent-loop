"""Operator recovery list for a partially published review round (#1142).

When the independence guard refuses to run a reviewer beside a public
same-round peer, the operator must delete the round's already-posted records so
the whole round runs again.  This module works out *which* comments, from the
same facts resume and the guard read, and verifies the answer by simulating
resume on the history that would remain.  It never changes the guard's
decision and never deletes anything.
"""

from __future__ import annotations

from collections import Counter
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
    """The REST history is non-empty and holds every relevant snapshot comment.

    Relevant comments are round-metadata records and transport attachments;
    they are compared with multiplicity so duplicates cannot collapse.
    """
    if not rest:
        return False
    available = Counter(_comment_key(comment) for comment in rest)
    needed = Counter(
        _comment_key(comment)
        for comment in snapshot
        if isinstance(comment.body, str)
        and (
            ROUND_RESUME_MARKER_RE.search(comment.body)
            or is_round_transport_sidecar(comment.body)
        )
    )
    return all(available[key] >= count for key, count in needed.items())


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
) -> tuple[dict[int, str], dict[int, str], str | None, int | None]:
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
        return {}, {}, None, None

    # The last durable record before the round began bounds orphan attribution.
    round_indexes = {
        record.index for record in records
        if record.metadata.flow == flow
        and record.metadata.round_number == round_number
        and record.metadata.subject == subject
    }
    first_in_round = min(round_indexes | set(targets))
    # Durable records of any flow (plan, pr, discuss) on the surface count.
    earlier = [
        index for index, comment in enumerate(comments)
        if index < first_in_round
        and index not in round_indexes
        and isinstance(comment.body, str)
        and ROUND_RESUME_MARKER_RE.search(comment.body)
        and not is_round_transport_sidecar(comment.body)
    ]
    lower_bound = max(earlier) if earlier else None
    if scheduler_phase is not None:
        # A later scheduler phase starts at its qualified opening: the first
        # checkpoint after the last earlier-phase record of this round.  A
        # later checkpoint (posted by a rerun) must not move the bound up and
        # hide an orphan from the first attempt.
        earlier_phase = [
            record.index for record in records
            if record.metadata.flow == flow
            and record.metadata.round_number == round_number
            and record.metadata.subject == subject
            and record.metadata.role == "reviewer"
            and record.metadata.phase == "publication"
            and record.metadata.scheduler_phase != scheduler_phase
            and record.index < min(targets)
        ]
        if earlier_phase:
            last_earlier = max(earlier_phase)
            openings = [
                record.index for record in records
                if record.metadata.role == "summary"
                and record.metadata.phase == "scheduler-prelaunch"
                and record.metadata.round_number == round_number
                and record.metadata.subject == subject
                and last_earlier < record.index < min(targets)
            ]
            lower_bound = min(openings) if openings else last_earlier

    attachments = [
        index for index, comment in enumerate(comments)
        if isinstance(comment.body, str) and is_round_transport_sidecar(comment.body)
    ]
    attachment_set = set(attachments)
    undecodable = False
    retained_refs: set[tuple[str, str]] = set()
    targeted_refs: set[tuple[str, str]] = set()
    for index, comment in enumerate(comments):
        if index in attachment_set or not isinstance(comment.body, str):
            continue
        try:
            refs = referenced_attachment_keys(comment.body)
        except AgentLoopError:
            # Keep the known targets; no attachment may be called unreferenced.
            undecodable = True
            continue
        (targeted_refs if index in targets else retained_refs).update(refs)

    def referenced(keys: set[tuple[str, str]], item: tuple[str, str]) -> bool:
        return item in keys or (item[0], "*") in keys

    round_authors = {
        strip_bot_login_suffix(comments[i].author)
        for i in targets
        if comments[i].author is not None
    }

    def same_author(index: int) -> bool:
        author = comments[index].author
        return author is not None and strip_bot_login_suffix(author) in round_authors

    uncertain: dict[int, str] = {}
    for index in attachments:
        keys = attachment_keys(comments[index].body or "")
        if any(referenced(retained_refs, key) for key in keys):
            continue
        if any(referenced(targeted_refs, key) for key in keys):
            targets[index] = "attachment of a listed comment"
            continue
        if undecodable:
            uncertain[index] = "attachment (references could not be decoded)"
            continue
        # Unreferenced: attribute by the round interval and author.
        if lower_bound is not None:
            if index > lower_bound and same_author(index):
                targets[index] = "unreferenced attachment"
            elif index > lower_bound:
                uncertain[index] = "unreferenced attachment (author differs)"
        elif index < first_in_round:
            uncertain[index] = "unreferenced attachment (possibly from this round, not verified)"
        elif same_author(index):
            targets[index] = "unreferenced attachment"
        else:
            uncertain[index] = "unreferenced attachment (author unknown or differs)"
    reason = "unreferenced attachments could not be attributed to this round" if uncertain else None
    if undecodable:
        reason = "attachment references could not be fully decoded"
    return targets, uncertain, reason, lower_bound


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
    fingerprint: Callable[[Sequence[IssueComment]], object] | None = None,
) -> PartialRoundRecovery:
    """The comments whose deletion lets the round run again, verified or provisional.

    ``fingerprint`` summarizes state the retained records must keep (for
    example the qualified panel opening); it must be equal before and after.
    """
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
            reviewer_names=reviewer_names, resume=resume, fingerprint=fingerprint,
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
    fingerprint: Callable[[Sequence[IssueComment]], object] | None = None,
) -> PartialRoundRecovery:
    def state_of(result: object) -> object:
        return result[1] if isinstance(result, tuple) and len(result) == 2 else result

    try:
        before = state_of(resume(comments))
        before_print = fingerprint(comments) if fingerprint is not None else None
    except AgentLoopError:
        before = None
        before_print = None

    def check(targets: dict[int, str], lower_bound: int | None) -> bool:
        if not targets:
            return False
        remaining = [c for i, c in enumerate(comments) if i not in targets]
        try:
            resumed = resume(remaining)
            records = _extract_round_metadata_records(remaining, flow=flow)
            refs: set[tuple[str, str]] = set()
            held: set[tuple[str, str]] = set()
            attachment_positions: list[tuple[int, tuple[tuple[str, str], ...]]] = []
            for position, comment in enumerate(remaining):
                body = comment.body if isinstance(comment.body, str) else ""
                if is_round_transport_sidecar(body):
                    keys = attachment_keys(body)
                    held.update(keys)
                    attachment_positions.append((position, keys))
                else:
                    refs.update(referenced_attachment_keys(body))
        except AgentLoopError:
            return False
        if _publication_records(
            records, flow=flow, round_number=round_number, subject=subject,
            scheduler_phase=scheduler_phase, reviewer_names=reviewer_names,
        ):
            return False
        if _reconciling_summaries(
            records, flow=flow, round_number=round_number, subject=subject
        ):
            return False
        # Every attachment a retained comment references must still exist.
        def present(key: tuple[str, str]) -> bool:
            if key[1] == "*":
                return any(held_key[0] == key[0] for held_key in held)
            return key in held

        if not all(present(key) for key in refs):
            return False
        # No unreferenced attachment may remain inside the round's interval.
        original_index = [i for i in range(len(comments)) if i not in targets]
        for position, keys in attachment_positions:
            if lower_bound is None or original_index[position] <= lower_bound:
                continue
            if not any(
                key in refs or (key[0], "*") in refs for key in keys
            ):
                return False
        state = state_of(resumed)
        anchored = any(
            record.metadata.flow == flow
            and record.metadata.role == "coder"
            and record.metadata.round_number == round_number
            and record.metadata.subject == subject
            for record in records
        )
        if getattr(state, "round_number", None) == round_number:
            if getattr(state, "reconciled", False):
                return False
            if any(
                record.metadata.agent in reviewer_names
                and (scheduler_phase is None or record.metadata.scheduler_phase == scheduler_phase)
                for record in getattr(state, "completed_reviews", ())
            ):
                return False
        elif anchored:
            # A round anchored by a retained coder record must still resume as
            # this round; only an unanchored round may fall back legitimately.
            return False
        if before is not None and getattr(before, "round_number", None) == round_number:
            if state is not None and getattr(state, "round_number", None) == round_number:
                # Retained artifacts must hydrate exactly as before deletion.
                for name in ("coder_output", "coder_metadata", "compact_prior_summaries"):
                    if getattr(state, name, None) != getattr(before, name, None):
                        return False
        if fingerprint is not None:
            try:
                if fingerprint(remaining) != before_print:
                    return False
            except AgentLoopError:
                return False
        return True

    verified = False
    targets: dict[int, str] = {}
    uncertain: dict[int, str] = {}
    reason: str | None = None
    for widen in (False, True):
        targets, uncertain, reason, bound = _plan(
            comments, flow=flow, round_number=round_number, subject=subject,
            scheduler_phase=scheduler_phase, reviewer_names=reviewer_names, widen=widen,
        )
        if check(targets, bound):
            verified = True
            break
    if not verified:
        reason = reason or "resume could not be confirmed on the history that would remain"
    if uncertain or reason == "attachment references could not be fully decoded":
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
