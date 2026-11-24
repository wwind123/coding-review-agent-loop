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
    allowed and a healthy peer returned a blocking review with actionable
    content.  ``retry_types`` (for example settings-lock waits) always keep
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
