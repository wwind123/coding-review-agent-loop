"""Failed reviewer seats in a blocked review round (#2129).

A reviewer seat *fails* in a round when its response is still rejected after
its re-asks (a deterministic failure), when it is unavailable (transient or
agent-unavailable), or when it reports an incomplete review.  When another
seat in the same round blocked, the PR and plan loops continue the round to
the coder or planner instead of stopping.  The failed seat's carried items are
frozen (see ``unresolved_items.preserve_failed_seat_items``), the seat is named
in the round's reconciliation record (``failed_reviewer_seats``), it is never
counted as approving, and it is invoked again next round.

The existing stop is kept for single-reviewer boards, when no healthy seat
blocked, for quota-reset errors, in evidence passes, in the final allowed
round, and when no must-fix item remains.  These helpers are pure so both
loops share one rule.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import replace

from .errors import QuotaResetExceededError

FAILED_SEAT_SCHEDULER_REASON = "failed seat re-invoked after a continued round"


def failed_seat_continuation_allowed(
    *,
    configured_reviewer_count: int,
    blocking_seats: Iterable[str],
    failed_seats: Collection[str],
    failures: Iterable[BaseException],
    round_number: int,
    allowed_rounds: int,
    evidence_pass: object = None,
) -> bool:
    """Whether a round with failed seats may continue to the coder/planner.

    Every condition must hold: a multi-reviewer board, at least one blocking
    seat that did not fail, no evidence pass, a round below the final allowed
    round, and no quota-reset failure.  The callers additionally require at
    least one must-fix item before they actually continue.
    """
    failed = set(failed_seats)
    if not failed:
        return False
    failure_list = list(failures)
    return (
        configured_reviewer_count > 1
        and any(seat not in failed for seat in blocking_seats)
        and evidence_pass is None
        and round_number < allowed_rounds
        and not any(isinstance(error, QuotaResetExceededError) for error in failure_list)
    )


def failure_category(error: BaseException) -> str:
    """The failure category a seat failure is reported under."""
    return getattr(error, "failure_category", None) or "deterministic"


def describe_failed_seats(
    failed: Mapping[str, BaseException],
    *,
    prior_items: Sequence[object],
    frozen_item_ids: Collection[str],
) -> str:
    """One bounded sentence naming each failed seat, its category and frozen items."""
    if not failed:
        return ""
    frozen = set(frozen_item_ids)
    parts = []
    for seat, error in failed.items():
        owned = [
            item.item_id
            for item in prior_items
            if item.item_id in frozen
            and seat in (tuple(item.resolution_owners) or (item.reviewer,))
        ]
        parts.append(
            f"{seat} ({failure_category(error)}; frozen carried items: "
            f"{', '.join(owned) if owned else 'none'})"
        )
    return "Failed reviewer seat(s): " + "; ".join(parts) + "."


def finalization_wording(*, stops: bool, continuation_allowed: bool, failed: Collection[str]) -> str:
    """The reconciliation comment's finalization clause."""
    if continuation_allowed and failed:
        return (
            "continues to the coder if must-fix items remain, with failed seat(s) "
            + ", ".join(failed)
        )
    return "stops" if stops else "continues"


def force_select_failed_seats(decision, seats: Collection[str], required: Sequence[str]):
    """Return ``decision`` with every required failed seat selected and unpaused."""
    forced = {seat for seat in seats if seat in set(required)}
    if decision is None or not forced:
        return decision
    selected = set(decision.selected_reviewers) | forced
    return replace(
        decision,
        selected_reviewers=tuple(name for name in required if name in selected),
        paused_reviewers=tuple(
            (name, reason) for name, reason in decision.paused_reviewers if name not in forced
        ),
        reason=f"{decision.reason}; {FAILED_SEAT_SCHEDULER_REASON}: {', '.join(sorted(forced))}",
    )


def peer_blocked_settlement(
    *,
    parse,
    blocks,
    name,
    configured_reviewer_count: int,
    round_number: int,
    allowed_rounds: int,
    evidence_pass: object = None,
    retry_types: tuple[type, ...] = (),
):
    """A ``settle_failures`` hook for ``_launch_reviewer_turns``.

    Failed seats settle in the round (instead of withholding the healthy
    peers' outcomes for an independent rerun) only when continuation is
    allowed and a healthy peer returned a completed blocking review (``blocks``
    mirrors ordinary settlement: new findings, a carried item kept blocking,
    or a substantive blocking summary all count; an incomplete review does
    not).  ``retry_types`` (for example settings-lock waits) always keep
    the withheld rerun.
    """

    def settle(settled: Mapping, retry_errors: Mapping) -> bool:
        if any(isinstance(error, retry_types) for error in retry_errors.values()):
            return False
        blockers = []
        for reviewer, turn in settled.items():
            if reviewer in retry_errors:
                continue
            parsed = parse(turn)
            if parsed is not None and blocks(parsed):
                blockers.append(name(reviewer))
        return failed_seat_continuation_allowed(
            configured_reviewer_count=configured_reviewer_count,
            blocking_seats=blockers,
            failed_seats=[name(reviewer) for reviewer in retry_errors],
            failures=retry_errors.values(),
            round_number=round_number,
            allowed_rounds=allowed_rounds,
            evidence_pass=evidence_pass,
        )

    return settle


def plan_failed_seat_retry_state(current_resume, *, round_reviewers, resumed, records, subject):
    """``(retry, unsettled)`` for a reconciled plan round with failed seats.

    The round's records are split at its latest reconciliation record.  That
    record names the failed seats ``F``; reviews published after it belong to
    an unfinished replacement settlement (a retry), reviews before it to the
    original round.  ``retry`` is true when every round reviewer without a
    resumed review is in ``F`` and either a failed seat is still missing or a
    failed seat already published after the record.  A round whose original
    reviews (before the record) were published in parallel never retries: a
    same-round retry would let the failed seat read its peers' published
    verdicts.  ``unsettled`` names the reviewers whose reviews were published
    after the record, so their findings are minted exactly once by the
    replacement settlement.  ``records`` is called lazily.
    """
    if current_resume is None or not current_resume.reconciled:
        return False, frozenset()
    missing = {name for name in round_reviewers if name not in resumed}
    round_records = [
        record for record in records()
        if record.metadata.round_number == current_resume.round_number
        and record.metadata.subject == subject
    ]
    latest = None
    for position, record in enumerate(round_records):
        metadata = record.metadata
        if metadata.role == "summary" and metadata.phase == "reconciliation":
            latest = position
    if latest is None:
        return False, frozenset()
    if any(r.metadata.role == "reviewer" and r.metadata.phase == "publication"
           for r in round_records[:latest]):
        return False, frozenset()
    failed = set(round_records[latest].metadata.failed_reviewer_seats)
    if not failed or not missing <= failed:
        return False, frozenset()
    reviewed_after = frozenset(
        record.metadata.agent for record in round_records[latest + 1:]
        if record.metadata.role == "reviewer"
    )
    retry = bool(missing) or bool(failed & reviewed_after)
    return retry, (reviewed_after if retry else frozenset())


def plan_round_retries_failed_seats(current_resume, *, round_reviewers, resumed, records, subject) -> bool:
    """Whether a reconciled plan round still owes a failed-seat settlement (see above)."""
    return plan_failed_seat_retry_state(
        current_resume, round_reviewers=round_reviewers, resumed=resumed, records=records, subject=subject,
    )[0]


def pending_failed_seat_retries(records) -> frozenset[str]:
    """Failed seats still owed a retry, reconstructed from durable records.

    The latest PR or plan reconciliation summary names the seats that failed in its
    round.  A named seat is still pending until it publishes a reviewer
    record after that summary, so a restart (or a closed-round rollover)
    keeps force-selecting it under selective policies.  Selecting an extra
    reviewer is always safe, so this hint never removes a seat.
    """
    records = list(records)  # history order
    latest = None
    for position, record in enumerate(records):
        metadata = record.metadata
        if metadata.role == "summary" and metadata.phase == "reconciliation":
            latest = position
    if latest is None or not records[latest].metadata.failed_reviewer_seats:
        return frozenset()
    reviewed_after = {
        record.metadata.agent for record in records[latest + 1:]
        if record.metadata.role == "reviewer"
    }
    return frozenset(seat for seat in records[latest].metadata.failed_reviewer_seats if seat not in reviewed_after)
