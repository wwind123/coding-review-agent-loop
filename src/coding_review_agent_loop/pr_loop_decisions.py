"""Pure reviewer, ledger and CI plumbing decisions for the PR loop."""
from __future__ import annotations

from collections.abc import Callable, Collection, Sequence

from .ci_health import WatchFailureClass
from .github import observation_is_stale
from .protocol import ReviewItemDisposition, UnresolvedReviewItem, human_requirements_resolved
from .response_validation import _reviewer_requirement_coverage_matches
from .round_state import PostedRoundRecord
from .unresolved_items import _apply_unresolved_item_dispositions


def _classify_pr_reviewer_prelaunch(
    reviewer_name: str,
    *,
    unavailable: bool,
    resumed: bool,
    selective_policy: bool,
    selected_reviewer_names: Collection[str],
    prior_approval: PostedRoundRecord | None,
    full_board_pass: bool,
    round_number: int,
    human_requirements: Sequence[object],
) -> str:
    """Pure pre-launch classification: ``skip``, ``resumed``, ``carried`` or ``turn``.

    Shared by the launch path and the closed-round gate (#1553) so the two
    can never disagree about whether a reviewer needs a fresh turn.
    """
    if unavailable:
        return "skip"
    if resumed:
        return "resumed"
    if selective_policy and reviewer_name not in selected_reviewer_names:
        return "skip"
    if (
        not full_board_pass
        and prior_approval is not None
        and prior_approval.metadata.round_number < round_number
        and (
            not human_requirements
            or (
                human_requirements_resolved(prior_approval.body)
                and _reviewer_requirement_coverage_matches(
                    human_requirements,
                    prior_approval.metadata.surfaced_reviewer_requirement_ids,
                )
            )
        )
    ):
        return "carried"
    return "turn"


def _settle_pr_round_ledger(
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions_by_item: dict[str, list[ReviewItemDisposition]],
    round_new_items: Sequence[UnresolvedReviewItem],
    *,
    compact: bool,
    reconciliation_mode: str,
    ledger_view: Callable[[Sequence[UnresolvedReviewItem]], Sequence[UnresolvedReviewItem]],
    **evidence_kwargs: object,
) -> tuple[list[UnresolvedReviewItem], list[UnresolvedReviewItem]]:
    """The PR round-end ledger settlement, shared with closed-round rollover (#1553).

    Returns the settled ledger and the carried items that left it as future
    follow-ups (only in compact context, where they are publication-only).
    Inert converted findings and tombstones keep their identities.
    """
    if compact:
        settled, future_from_prior_items = _apply_unresolved_item_dispositions(
            ledger_view(prior_items),
            dispositions_by_item,
            retain_future=False,
            reconciliation_mode=reconciliation_mode,
            **evidence_kwargs,
        )
    else:
        settled, _future_items = _apply_unresolved_item_dispositions(
            ledger_view(prior_items),
            dispositions_by_item,
            reconciliation_mode=reconciliation_mode,
            **evidence_kwargs,
        )
        future_from_prior_items = []
    return list(ledger_view([*settled, *round_new_items])), future_from_prior_items


def _route_post_approval_ci_failure(
    classification: WatchFailureClass | None, *, rerun_used: bool
) -> str:
    """Route one classified post-approval failure.

    Returns ``coder`` (existing obligation path), ``rerun_and_watch`` (the
    single plumbing rerun for this head), ``hold`` (keep watching) or
    ``infra_stop`` (no coder obligation, no merge).
    """
    if classification == "plumbing-only":
        return "infra_stop" if rerun_used else "rerun_and_watch"
    if classification == "stuck-infrastructure":
        return "infra_stop"
    if classification == "plumbing-awaiting-tests":
        return "hold"
    return "coder"


def _plumbing_rerun_decision(
    observations, rerun_state, *, filter_stale: bool, budget_left: bool
) -> str:
    """Decide whether one plumbing-only failure may spend the head's single rerun.

    Every rerun request, from the watcher or the approval snapshot, passes
    through here (#1477).  A spent budget never issues another rerun: evidence
    from before that rerun keeps watching, fresh evidence stops on
    infrastructure.  An unspent budget reruns only while the shared watch
    budget can still observe the result.
    """
    if rerun_state is not None:
        fresh = [
            check for check in observations
            if not (filter_stale and observation_is_stale(check, rerun_state, None))
        ]
        return "watch" if not fresh else "infra_stop"
    return "rerun" if budget_left else "budget_exhausted"


def _plumbing_observation_lines(failed, stuck) -> list[str]:
    lines = [
        f"{check.name} ({check.status.lower()})"
        + (f" — {check.url.strip()}" if check.url and check.url.strip() else "")
        for check in failed
    ]
    lines.extend(
        f"{check.name} (stuck pending after its producing run completed)"
        + (f" — {check.url.strip()}" if check.url and check.url.strip() else "")
        for check in stuck
    )
    return lines
