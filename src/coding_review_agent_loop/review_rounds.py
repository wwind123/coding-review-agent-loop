"""Review outcome classification, round-ledger helpers, reviewer-turn launch, spool replay and partial-round refusal.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1195); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import contextvars
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace as dataclasses_replace
from pathlib import Path
from .agents.base import AgentName
from .agents.registry import agent_display_name, get_backend
from .config import AgentLoopConfig, reviewers
from .board_amendment import ReviewerBoardAmendment, amend_contract
from .errors import AgentInvocationError, AgentLoopError, QuotaResetExceededError
from .github import PullRequestChecks
from .logging import log
from .protocol import (
    ParsedPlanReview,
    ParsedReview,
    ReviewItemDisposition,
    UnresolvedReviewItem,
)
from .runner import Runner
from .ci_health import (
    StalledCheck,
    is_canonical_stall_only_text,
    is_canonical_pending_only_text,
    is_wholly_infrastructure_blocked,
)
from .round_state import (
    ResumedReviewRound,
    _canonically_resolved_history_item_ids,
    _live_round_resolved_item_ids,
    _extract_round_metadata_records,
)
from .partial_round_recovery import PartialRoundRecovery, RecoveryTarget
from .publication_resume import PublicationResumeStop
from .review_spool import ReviewRoundSpool, review_spool_root
from .agent_failure import ValidatedAgentResponse
from .architecture_contract import (
    _with_validation_context,
    _accept_candidate,
    _accepted_validated_response,
)
from .panel_evidence import _board_amendment_template, _PR_AMENDMENT_RERUN_CLAUSE


def _is_pending_ci_only_review(parsed_review: ParsedReview, pr_checks: PullRequestChecks) -> bool:
    """Detect a blocking review whose only content restates pending/unavailable
    GitHub check status rather than an actionable code-level finding.

    This is a defense-in-depth backstop: the reviewer prompt already instructs
    reviewers not to use pending/unavailable checks as the sole reason to
    block, but a reviewer may still do so. Any other content (a distinct
    blocking item, or a Same-PR follow-up) causes this to return False so
    mixed responses still route back to the coder normally. Whole-statement
    matching is deliberately conservative: an unfamiliar phrasing must not
    silently turn a blocking code review into an approval.
    """
    if pr_checks.state not in {"pending", "unavailable"}:
        return False
    if parsed_review.followups.same_pr:
        return False
    if any(item.disposition in {"blocking", "same-pr"} for item in parsed_review.dispositions):
        return False
    candidate_texts = [
        item.text for item in parsed_review.blocking_items if item.text and item.text.strip()
    ]
    if not candidate_texts and parsed_review.summary and parsed_review.summary.strip():
        candidate_texts = [parsed_review.summary]
    if not candidate_texts:
        return False
    check_names = tuple(check.name for check in pr_checks.pending) + pr_checks.missing_required
    if not all(is_canonical_pending_only_text(text, check_names=check_names) for text in candidate_texts):
        return False
    summary = (parsed_review.summary or "").strip()
    return (
        not summary
        or summary in _BOILERPLATE_REVIEW_SUMMARIES
        or is_canonical_pending_only_text(summary, check_names=check_names)
    )


def _normalize_approval_gated_managed_ci_review(
    parsed_review: ParsedReview,
    *,
    prior_items: Sequence[UnresolvedReviewItem],
    pr_checks: PullRequestChecks,
    current_head_sha: str | None,
) -> ParsedReview:
    """Prevent a reviewer from deadlocking post-approval managed CI.

    A repaired managed-CI head must first receive every required reviewer
    approval. Only then can the orchestrator dispatch exact-head
    qualification. Reviewers still disposition the durable machine record,
    but a blocking disposition against that wait is not a code finding.
    """
    if parsed_review.state != "blocking" or pr_checks.state not in {
        "passing",
        "no_checks",
        "pending",
        "unavailable",
    }:
        return parsed_review
    if parsed_review.blocking_items or parsed_review.followups.same_pr:
        return parsed_review

    active_dispositions = tuple(
        disposition
        for disposition in parsed_review.dispositions
        if disposition.disposition in {"blocking", "same-pr"}
    )
    if not active_dispositions:
        return parsed_review

    items_by_id = {item.item_id: item for item in prior_items}
    for disposition in active_dispositions:
        item = items_by_id.get(disposition.item_id)
        if not (
            item is not None
            and item.is_machine_obligation
            and item.obligation_kind == "managed-exact-head-ci"
            and item.lifecycle == "awaiting_current_head_review"
            and bool(item.candidate_head_sha)
            and item.candidate_head_sha == current_head_sha
            and item.failed_head_sha != current_head_sha
        ):
            return parsed_review

    active_ids = {disposition.item_id for disposition in active_dispositions}
    dispositions = tuple(
        dataclasses_replace(disposition, disposition="resolved")
        if disposition.item_id in active_ids
        else disposition
        for disposition in parsed_review.dispositions
    )
    return dataclasses_replace(parsed_review, state="approved", dispositions=dispositions)


def _coder_infrastructure_stall_notice(stalls: Sequence[StalledCheck]) -> str:
    """Prepended to a coder follow-up review when checks are wholly
    infrastructure-blocked (#602), so the coder never has to discover this
    itself by waiting on the stalled check.
    """
    if not stalls:
        return ""
    bullets = "\n".join(f"- {stall.describe()}" for stall in stalls)
    return (
        "External CI infrastructure is currently blocking the following GitHub "
        "checks; this is not a code defect and there is no fix for it in this PR:\n"
        f"{bullets}\n\n"
        "Do not wait for these checks to leave queued state and do not attempt to "
        "retrigger them. Fix only the genuine review items below. If your terminal "
        "response needs to mention CI status, name the affected check/run above and "
        "say that work should resume once GitHub Actions runners recover.\n\n"
    )


_BOILERPLATE_REVIEW_SUMMARIES = {"Review complete.", "Plan review complete."}


def _is_infrastructure_ci_only_review(parsed_review: ParsedReview, pr_checks: PullRequestChecks) -> bool:
    """Detect a blocking review whose only content is a canonical restatement of
    an external CI infrastructure stall (a queued check that never started a
    job, or one cancelled before execution because a hosted runner was
    unavailable), rather than an actionable code-level finding.

    Like the pending-only filter, this fails closed on ambiguous prose. It requires the
    whole check board to already be classified `is_wholly_infrastructure_blocked`
    and every blocking item (and non-boilerplate summary) to pass the closed-
    vocabulary `is_canonical_stall_only_text` check. Any failure aborts the
    downgrade for the whole review, so a mixed item that names the stalled run
    and then describes an unrelated code defect reaches the coder unchanged.
    """
    if not is_wholly_infrastructure_blocked(pr_checks):
        return False
    if parsed_review.followups.same_pr:
        return False
    if any(item.disposition in {"blocking", "same-pr"} for item in parsed_review.dispositions):
        return False
    candidate_texts = [
        item.text for item in parsed_review.blocking_items if item.text and item.text.strip()
    ]
    if not candidate_texts:
        return False
    stalls = pr_checks.infrastructure_stalls
    if not all(is_canonical_stall_only_text(text, stalls=stalls) for text in candidate_texts):
        return False
    summary = (parsed_review.summary or "").strip()
    if (
        summary
        and summary not in _BOILERPLATE_REVIEW_SUMMARIES
        and not is_canonical_stall_only_text(summary, stalls=stalls)
    ):
        return False
    return True


def _should_record_new_blocking_item(
    summary: str,
    *,
    had_prior_items: bool,
    had_dispositions: bool,
    has_active_carried_disposition: bool = False,
) -> bool:
    if not summary:
        return False
    if summary.strip() in {"Review complete.", "Plan review complete."}:
        return False
    if not had_prior_items or not had_dispositions:
        return True
    if has_active_carried_disposition:
        return False
    non_empty_lines = [line.strip() for line in summary.splitlines() if line.strip()]
    if len(non_empty_lines) > 1:
        return True
    return len(non_empty_lines[0]) >= 80


_INCOMPLETE_REVIEW_RE = re.compile(
    r"\breview\s+(?:is\s+)?incomplete\b|"
    r"\b(?:resolution|finding)\s+could\s+not\s+be\s+confirmed\b|"
    r"\b(?:could\s+not|cannot|can't|unable\s+to)\s+"
    r"(?:complete|confirm|verify|assess|review)\b",
    re.IGNORECASE,
)


def _is_incomplete_pr_review(parsed_review: ParsedReview) -> bool:
    """Identify a reviewer failure reported as a blocking verdict.

    A reviewer occasionally returns a syntactically valid blocking response that
    only says it could not inspect the diff or confirm a prior item. That is
    not actionable feedback for the coder. Keep this deliberately narrow so
    substantive freeform blocking summaries retain their existing behavior.
    """
    if parsed_review.state != "blocking":
        return False
    if parsed_review.blocking_items or parsed_review.followups.same_pr:
        return False
    if any(disposition.disposition in {"blocking", "same-pr"} for disposition in parsed_review.dispositions):
        return False
    candidate_texts = [parsed_review.summary]
    candidate_texts.extend(disposition.note for disposition in parsed_review.dispositions)
    return any(text and _INCOMPLETE_REVIEW_RE.search(text) for text in candidate_texts)


def _describe_pr_review_outcome(parsed_review: ParsedReview, *, has_blocking_summary: bool) -> str:
    if parsed_review.state == "approved":
        return "approved"
    has_same_pr = bool(parsed_review.followups.same_pr)
    has_blocking_findings = bool(parsed_review.blocking_items) or has_blocking_summary
    if has_blocking_findings and has_same_pr:
        return "blocking with blocking findings and same-PR follow-ups"
    if has_same_pr:
        return "blocking with same-PR follow-ups"
    return "blocking with blocking findings"


def _format_incomplete_pr_review_comment(
    *,
    pr_number: int,
    unavailable_reviewers: Mapping[AgentName, AgentInvocationError],
    approved_reviewer_names: Sequence[str],
    detection_round: int | None = None,
    blocking_reviewer_names: Sequence[str] = (),
    open_must_fix_count: int | None = None,
    remedy_lines: Sequence[str] = (),
) -> str:
    lines = [
        "**Review status: Incomplete**",
        "",
        "After the unavailable reviewer(s) were detected"
        + (f" in round {detection_round}" if detection_round is not None else "")
        + ", no coder follow-up, CI wait, qualification, or merge was started. "
        "Their failure is not a code finding and does not count as approval.",
        "",
        "### Missing required reviewer input",
    ]
    for reviewer, failure in unavailable_reviewers.items():
        category = failure.failure_category or "unknown"
        lines.append(f"- {agent_display_name(reviewer)}: {category}")
    if approved_reviewer_names:
        lines.extend(
            [
                "",
                "### Healthy reviewer approvals",
                *[f"- {name}" for name in approved_reviewer_names],
            ]
        )
    if blocking_reviewer_names or open_must_fix_count is not None:
        lines.extend(["", "### Open findings"])
        if blocking_reviewer_names:
            lines.append("- Blocking reviewers this round: " + ", ".join(blocking_reviewer_names))
        if open_must_fix_count is not None:
            lines.append(f"- Open must-fix items (including carried): {open_must_fix_count}")
    if remedy_lines:
        lines.extend(["", "### What to do", *remedy_lines])
    lines.extend(
        [
            "",
            (
                f"Resolve the reviewer problem as described above before merging PR #{pr_number}."
                if remedy_lines
                else f"Resolve the reviewer problem before merging PR #{pr_number}."
            ),
        ]
    )
    return "\n".join(lines)


def _unavailable_reviewer_remedy(
    *,
    route: str,
    surface: str,
    round_number: int | None = None,
    in_evidence_pass: bool = False,
) -> list[str]:
    """Operator remedy prose for an unavailable-reviewer stop (#1129).

    ``route`` is ``amendment`` (scheduler contract, amendment validated),
    ``amendment-lookup-failed``, ``flags`` (all-reviewers), ``restore-only``
    (strict managed-CI all-reviewers), or ``rejected`` (the amendment would be
    refused).  The text carries no amendment JSON or record grammar.
    """
    lines = [
        "- Restore the reviewer backend and rerun unchanged; nothing else needs to change.",
    ]
    degraded = " This is a degraded review mode, not a neutral substitution."
    if route == "amendment":
        lines.append(
            "- Or, for this run only, a human operator may remove the reviewer with a signed "
            f"reviewer-board amendment posted on {surface}, effective from round {round_number}."
            + _PR_AMENDMENT_RERUN_CLAUSE + degraded
        )
    elif route == "amendment-lookup-failed":
        lines.append(
            "- Or, for this run only, a human operator may remove the reviewer with a signed "
            f"reviewer-board amendment posted on {surface}, effective from the round the "
            "rerun resumes into." + _PR_AMENDMENT_RERUN_CLAUSE + degraded
        )
    elif route == "flags":
        lines.append(
            "- Or rerun with the unavailable reviewer removed from the --reviewer flags."
            + degraded
        )
    elif route == "restore-only":
        lines.append(
            "- Restoring the backend is the only in-run option: this is an issue-created strict "
            "managed-CI PR whose approved plan re-verifies the configured reviewer board."
        )
    else:
        lines.append(
            "- The reviewer board cannot be reduced for this run (the unavailable reviewer is "
            "the primary or the last secondary); restore the backend, or start a fresh run "
            "with a different reviewer board."
        )
    if in_evidence_pass:
        lines.append(
            "- The stopped exact-head evidence pass is repeated in full with the remaining "
            "board on rerun."
        )
    return lines


def _unavailable_reviewer_amendment_advisory(
    *,
    pr_number: int,
    contract: object,
    lineage: object,
    removed: Sequence[str],
    fetch_start_round: Callable[[], int],
) -> tuple[str, int | None, str | None]:
    """Validate an amendment removing ``removed`` before it is advertised (#1129).

    Returns ``(outcome, round, template)`` where outcome is ``validated``,
    ``rejected`` or ``lookup-failed``.  Never raises.
    """
    try:
        round_number = fetch_start_round()
        amendment = ReviewerBoardAmendment(
            flow="pr",
            issue=None,
            pr_number=pr_number,
            original_required_reviewers=tuple(getattr(contract, "required_reviewers")),
            policy=str(getattr(contract, "policy")),
            primary_reviewer=getattr(contract, "primary_reviewer"),
            removed_reviewers=tuple(removed),
            effective_from_round=round_number,
            reason="reviewer unavailable",
            rationale="",
            digest="",
            comment_locator="(dry run)",
            comment_index=0,
        )
        try:
            amend_contract(
                contract,
                amendment,
                base_board=tuple(getattr(getattr(lineage, "contracts")[0], "required_reviewers")),
                previously_removed=tuple(getattr(lineage, "removed_reviewers", ())),
            )
        except AgentLoopError:
            return "rejected", round_number, None
        template = _board_amendment_template(
            flow="pr",
            issue_number=None,
            pr_number=pr_number,
            persisted=contract,
            removed=removed,
            start_round_number=round_number,
        )
        return "validated", round_number, template
    except Exception:  # noqa: BLE001 - advisory text only; the stop stays authoritative
        return "lookup-failed", None, None


def _is_incomplete_plan_review(parsed_review: ParsedPlanReview) -> bool:
    """Identify a plan reviewer failure reported as a blocking verdict.

    Mirrors `_is_incomplete_pr_review` for the plan review flow: a reviewer
    occasionally returns a syntactically valid blocking response that only
    says it could not inspect the plan or confirm a prior item. That is not
    actionable feedback for the coder. Keep this deliberately narrow so
    substantive freeform blocking summaries retain their existing behavior.
    """
    if parsed_review.state != "blocking":
        return False
    if parsed_review.items.blocking or parsed_review.items.same_plan:
        return False
    if any(disposition.disposition in {"blocking", "same-plan"} for disposition in parsed_review.dispositions):
        return False
    candidate_texts = [parsed_review.summary]
    candidate_texts.extend(disposition.note for disposition in parsed_review.dispositions)
    return any(text and _INCOMPLETE_REVIEW_RE.search(text) for text in candidate_texts)


def _describe_plan_review_outcome(parsed_review: ParsedPlanReview) -> str:
    if parsed_review.state == "approved":
        return "approved"
    has_blocking = bool(parsed_review.items.blocking)
    has_same_plan = bool(parsed_review.items.same_plan)
    if has_blocking and has_same_plan:
        return "blocking with blocking plan issues and same-plan follow-ups"
    if has_same_plan:
        return "blocking with same-plan follow-ups"
    return "blocking with blocking plan issues"


def _round_ledger_may_be_incomplete(
    *,
    current_resume: ResumedReviewRound | None,
    prior_unresolved_items: Sequence[UnresolvedReviewItem],
    comments: Sequence[object],
    flow: str,
    current_subject: str,
    accounted_item_ids: Sequence[str] = (),
) -> bool:
    """Whether the active finding ledger may be missing a recorded item.

    ``accounted_item_ids`` holds item IDs that are demonstrably part of the
    reconstructed ledger even when the durable record that introduced them
    names an earlier plan subject: the ones this run carried or minted, plus
    the ones recorded history proves were canonically cleared.  The cleared
    half has to come from recorded history rather than the in-process set
    alone, because a reviewer-only phase advance persists an empty carried
    ledger, so a restart on that seam would otherwise rediscover a cleared
    cross-subject item and read the ledger as unreconstructible.  Callers that
    pass nothing keep the previous conservative reading, where any
    cross-subject item at all makes the ledger unreconstructible; that is what
    the PR flow and the compatibility-default full-board planning path both
    do, so only staged planning changes behavior here (#905, from #841).
    """
    same_subject_incomplete = (
        current_resume.ledger_may_be_incomplete
        if current_resume is not None
        else False
    )
    if prior_unresolved_items:
        return same_subject_incomplete
    records = _extract_round_metadata_records(comments, flow=flow)
    accounted = set(accounted_item_ids)
    cross_subject_incomplete = any(
        item.item_id not in accounted
        for record in records
        if record.metadata.subject != current_subject
        for item in record.metadata.new_items
    )
    return same_subject_incomplete or cross_subject_incomplete


def _round_resolved_history_item_ids(
    *,
    prior_unresolved_items: Sequence[UnresolvedReviewItem],
    comments: Sequence[object],
    flow: str,
    reconciliation_mode: str,
    same_status: str,
) -> tuple[str, ...]:
    """IDs whose recorded history proves canonical resolution (#862).

    It lets the deterministic strip remove a reviewer's no-op ``resolved``
    disposition of an item cleared in an earlier round.  Computed for every
    round, not only incomplete-ledger ones, because a lossless semantic patch
    needs the same proof even when the ledger is complete (#872).  Undecodable
    metadata yields no history, so recovery stays fail-closed.
    """
    try:
        records = _extract_round_metadata_records(comments, flow=flow)
    except AgentLoopError:
        return ()
    return tuple(
        sorted(
            _canonically_resolved_history_item_ids(
                records,
                reconciliation_mode=reconciliation_mode,
                same_status=same_status,
                current_carried_ids=tuple(item.item_id for item in prior_unresolved_items),
            )
        )
    )


def _post_round_resolved_history_item_ids(
    *,
    prior_unresolved_items: Sequence[UnresolvedReviewItem],
    dispositions_by_item: Mapping[str, Sequence[ReviewItemDisposition]],
    carried_items: Sequence[UnresolvedReviewItem],
    comments: Sequence[object],
    flow: str,
    reconciliation_mode: str,
    same_status: str,
) -> tuple[str, ...]:
    """Resolved-history proof for a turn that runs after this round's dispositions (#874).

    The plan revision is issued at the end of a round, so a planner that echoes
    an item the round just cleared needs that item in the whitelist.  The
    recorded history alone cannot supply it: ``comments`` is the snapshot taken
    before the review turns, and nothing refetches it when a reviewer
    disposition is posted, so the replay never sees the clearing round.  The
    round's own authenticated dispositions and post-round carried set do, and
    both proofs apply the same canonical resolution rule, so their union stays
    fail-closed.
    """
    carried_ids = tuple(item.item_id for item in carried_items)
    recorded = _round_resolved_history_item_ids(
        prior_unresolved_items=carried_items,
        comments=comments,
        flow=flow,
        reconciliation_mode=reconciliation_mode,
        same_status=same_status,
    )
    live = _live_round_resolved_item_ids(
        prior_items=prior_unresolved_items,
        dispositions_by_item=dispositions_by_item,
        carried_item_ids=carried_ids,
    )
    return tuple(sorted(set(recorded) | live))


@dataclass(frozen=True)
class _ReviewerTurnResult:
    """Outcome of one plan/PR reviewer turn: a validated response or a captured failure.

    Worker threads in the --review-parallel path return these instead of
    raising, so exceptions never cross the thread boundary; once every launched
    turn has returned, the main thread publishes each validated review in
    completion order (never mid-round, #1025) and then performs
    configured-order aggregation.
    """

    reviewer_name: str
    response: ValidatedAgentResponse | None = None
    error: AgentLoopError | None = None


def _review_round_spool(
    config: AgentLoopConfig, *, surface: str, number: int, round_number: int, subject: str
) -> ReviewRoundSpool:
    return ReviewRoundSpool(
        root=review_spool_root(config.agent_memory_dir),
        repo=config.repo,
        surface=surface,
        number=number,
        round_number=round_number,
        subject=subject,
    )


def _replay_spooled_review(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    reviewer: AgentName,
    fields: dict[str, object],
    validators: dict[str, object],
) -> _ReviewerTurnResult | None:
    """Rebuild a spooled reviewer response through its round's own validator.

    The spooled text is the accepted text of the original invocation, so the
    same validate/strict re-parse chain yields the same accepted response.
    ``None`` means the text no longer validates in this run's context and
    the caller must fall back to a fresh reviewer turn.
    """
    reviewer_name = agent_display_name(reviewer)
    text = fields.get("text")
    validate = validators["validate"]
    strict_revalidate = validators.get("strict_revalidate")
    assert isinstance(text, str) and callable(validate)
    try:
        marker_value = _with_validation_context(validate, runner=runner, acquisition=None)(text)
        candidate = _accept_candidate(
            text,
            marker_value,
            strict_revalidate=strict_revalidate,  # type: ignore[arg-type]
            runner=runner,
            acquisition=None,
        )
    except AgentLoopError as exc:
        log(
            config,
            f"{reviewer_name}: withheld same-round review no longer validates ({exc}); "
            "invoking a fresh review turn instead",
        )
        return None
    identity = {
        name: fields.get(name)
        for name in (
            "session_id", "model_used", "provider", "role", "configured_model",
            "configured_effort", "effort_source", "observed_model", "observed_effort",
            "observation_provenance", "acquisition_returncode",
        )
    }
    acquisition_outcome = fields.get("acquisition_outcome")
    if acquisition_outcome not in {"success", "accepted_nonzero_exit", "accepted_timeout"}:
        acquisition_outcome = "success"
    log(config, f"{reviewer_name}: replaying its withheld same-round review instead of re-invoking it")
    return _ReviewerTurnResult(
        reviewer_name=reviewer_name,
        response=_accepted_validated_response(
            candidate, acquisition_outcome=acquisition_outcome, **identity
        ),
    )


def _spooled_response_fields(response: ValidatedAgentResponse) -> dict[str, object]:
    return {
        "text": response.text,
        "session_id": response.session_id,
        "model_used": response.model_used,
        "provider": response.provider,
        "role": response.role,
        "configured_model": response.configured_model,
        "configured_effort": response.configured_effort,
        "effort_source": response.effort_source,
        "observed_model": response.observed_model,
        "observed_effort": response.observed_effort,
        "observation_provenance": response.observation_provenance,
        "acquisition_outcome": response.acquisition_outcome,
        "acquisition_returncode": response.acquisition_returncode,
    }


def _incomplete_plan_review_error(reviewer_name: str) -> AgentLoopError:
    return AgentLoopError(
        f"{reviewer_name} did not complete plan review and reported no actionable "
        "blocking plan issues or Same-Plan follow-ups. This is a reviewer-internal error; "
        "agent-loop stopped before a coder follow-up. Rerun or switch the reviewer/model "
        "after resolving the reviewer environment."
    )


def _incomplete_pr_review_error(reviewer_name: str) -> AgentLoopError:
    return AgentLoopError(
        f"{reviewer_name} did not complete PR review and reported no actionable "
        "blocking items or Same-PR follow-ups. This is a reviewer-internal error; "
        "agent-loop stopped before a coder follow-up. Rerun or switch the reviewer/model "
        "after resolving the reviewer environment."
    )


class PartialReviewRoundError(AgentLoopError):
    """A reviewer would have to run while a same-round peer's body is public (#1025)."""


def _replay_unavailable_cause(
    spool: ReviewRoundSpool,
    reviewer_name: str,
    *,
    own_posted: bool = False,
    loaded_unreplayable: bool = False,
) -> str:
    """Why this reviewer's withheld outcome cannot be replayed (#1142)."""
    if own_posted:
        return (
            f"{reviewer_name}'s own review for this round is already posted but no longer "
            "accepted, so it needs a fresh turn and its withheld outcome cannot be replayed."
        )
    if loaded_unreplayable:
        return f"{reviewer_name}'s spooled outcome no longer validates against this run."
    if not spool.has_records():
        return f"No review spool for this round exists on this host (looked under {spool.directory})."
    if not spool.record_exists(reviewer_name):
        return f"The round's review spool at {spool.directory} holds no outcome for {reviewer_name}."
    if spool.load(reviewer_name) is None:
        return (
            f"{reviewer_name}'s spool file {spool._path(reviewer_name)} exists but is unreadable "
            "or does not belong to this round."
        )
    return f"{reviewer_name}'s spooled outcome no longer validates against this run."


def _format_recovery_target(target: RecoveryTarget) -> str:
    if target.comment_id is None:
        # A projection can carry a permalink without a numeric id; never print
        # "comment None".
        url = f" {target.url}" if target.url else ""
        return (
            f"- {target.label}: by {target.author or 'unknown author'}, created "
            f"{target.created_at or 'unknown time'} (id could not be determined){url}"
        )
    return f"- {target.label}: comment {target.comment_id} {target.url or ''}".rstrip()


def _render_recovery(recovery: PartialRoundRecovery) -> str:
    lines: list[str] = []
    if recovery.verified:
        lines.append(
            f"Delete these {len(recovery.targets)} comments so the whole round runs again "
            "independently, then rerun:"
        )
        lines.extend(_format_recovery_target(t) for t in recovery.targets)
        lines.append(
            "Other comments (such as the incomplete-status notice) carry no round state for "
            "this round and need not be deleted."
        )
        lines.append(
            "Deleting qualification checkpoint records also removes the managed-CI "
            "authorization binding; the next resume may then need fresh managed-CI "
            "authorization (`--managed-ci-fresh`)."
        )
    else:
        lines.append(
            "The comments below are a provisional list that could not be verified as a "
            f"sufficient deletion set ({recovery.reason or 'unverified'}); inspect the "
            "round's records before deleting."
        )
        lines.extend(_format_recovery_target(t) for t in recovery.targets)
        if recovery.uncertain:
            lines.append("Unreferenced attachments possibly from this round, not verified:")
            lines.extend(_format_recovery_target(t) for t in recovery.uncertain)
    return "\n".join(lines)


def _partial_round_refusal(
    *,
    surface: str,
    number: int,
    round_number: int,
    reviewer_name: str,
    public_peers: Sequence[str],
    cause: str = "",
    recovery: Callable[[], PartialRoundRecovery] | None = None,
) -> PartialReviewRoundError:
    target = f"PR #{number}" if surface == "pr" else f"issue #{number}"
    message = (
        f"Review round {round_number} on {target} is partially published "
        f"({', '.join(public_peers)} already posted) but {reviewer_name}'s withheld "
        "same-round outcome is missing or no longer validates, so invoking it now would "
        "let it read its peers' findings."
    )
    if cause:
        message += f" {cause}"
    if recovery is not None:
        try:
            listing = _render_recovery(recovery())
        except Exception:  # the refusal type must never change
            listing = (
                "The round's comment list is unavailable; inspect the round's reviewer "
                "verdicts, reconciliation records and attachments before deleting them."
            )
        message += "\n" + listing
    return PartialReviewRoundError(message)


def _replay_spooled_failure(fields: dict[str, object], reviewer_name: str) -> AgentInvocationError | None:
    failure = fields.get("failure")
    if not isinstance(failure, dict):
        return None
    return AgentInvocationError(
        f"{reviewer_name} (replayed from its withheld same-round outcome): {failure.get('message')}",
        failure_category=failure.get("failure_category"),  # type: ignore[arg-type]
    )


def _refuse_partial_round_before_sequential_turns(
    *,
    spool: ReviewRoundSpool,
    fresh_turn_reviewers: Sequence[AgentName],
    public_peers: Sequence[AgentName],
    already_posted: Sequence[AgentName] = (),
    recovery: Callable[[], PartialRoundRecovery] | None = None,
) -> None:
    """Stop before any sequential turn when one of them would face a public peer.

    A sequential round only reaches this with no spooled outcomes (a round
    holding them is finished by the withhold-then-publish launcher), so a
    reviewer needing a fresh turn beside a public same-batch peer could never
    be replayed.  Checking the whole round first avoids re-running one
    reviewer and then stopping at the next.
    """
    for reviewer in fresh_turn_reviewers:
        peers = sorted(agent_display_name(peer) for peer in public_peers if peer != reviewer)
        if peers:
            raise _partial_round_refusal(
                surface=spool.surface, number=spool.number, round_number=spool.round_number,
                reviewer_name=agent_display_name(reviewer), public_peers=peers,
                cause=_replay_unavailable_cause(
                    spool, agent_display_name(reviewer), own_posted=reviewer in already_posted
                ),
                recovery=recovery,
            )


def _same_round_replay_or_invoke(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    reviewer: AgentName,
    spool: ReviewRoundSpool,
    public_peers: Sequence[AgentName],
    validators: dict[str, object],
    invoke: Callable[[], ValidatedAgentResponse],
    already_posted: bool = False,
    recovery: Callable[[], PartialRoundRecovery] | None = None,
) -> ValidatedAgentResponse:
    """Sequential-mode seam: never invoke a reviewer against public same-round peers.

    A parallel round interrupted between publications may be resumed without
    ``--review-parallel``.  The unpublished reviewer's withheld outcome is then
    replayed exactly as in parallel mode; without a valid one the run stops
    instead of invoking the reviewer.
    """
    reviewer_name = agent_display_name(reviewer)
    peers = [agent_display_name(peer) for peer in public_peers if peer != reviewer]
    if not peers:
        return invoke()
    fields = None if already_posted else spool.load(reviewer_name)
    if fields is not None:
        failure = _replay_spooled_failure(fields, reviewer_name)
        if failure is not None:
            raise failure
        replayed = _replay_spooled_review(
            runner, config=config, reviewer=reviewer, fields=fields, validators=validators
        )
        if replayed is not None and replayed.response is not None:
            return replayed.response
    raise _partial_round_refusal(
        surface=spool.surface, number=spool.number, round_number=spool.round_number,
        reviewer_name=reviewer_name, public_peers=sorted(peers),
        cause=_replay_unavailable_cause(
            spool, reviewer_name, own_posted=already_posted,
            loaded_unreplayable=fields is not None,
        ),
        recovery=recovery,
    )


def _launch_reviewer_turns(
    runner: Runner,
    pending: Sequence[AgentName],
    *,
    thread_name_prefix: str,
    run_turn: Callable[[AgentName], _ReviewerTurnResult],
    on_completion: Callable[[AgentName, _ReviewerTurnResult], bool | None] | None = None,
    spool: ReviewRoundSpool | None = None,
    replay_turn: Callable[[AgentName, dict[str, object]], _ReviewerTurnResult | None] | None = None,
    public_peers: Sequence[AgentName] = (),
    retry_bound: Callable[[AgentName, _ReviewerTurnResult], AgentLoopError | None] | None = None,
    prelaunch_failures: dict[AgentName, _ReviewerTurnResult] | None = None,
    configured_order: Sequence[AgentName] = (),
    config: AgentLoopConfig | None = None,
    already_posted: Sequence[AgentName] = (),
    max_workers: int | None = None,
    prepare_fallback: Callable[[AgentName], None] | None = None,
    recovery: Callable[[], PartialRoundRecovery] | None = None,
) -> dict[AgentName, _ReviewerTurnResult]:
    """Run workers concurrently, then deliver results in completion order.

    ``run_turn`` must never let an exception escape; it is responsible for
    capturing any failure into the returned ``_ReviewerTurnResult`` so the
    thread pool never needs to propagate a worker exception. On
    KeyboardInterrupt, active agent processes are killed so worker wait loops
    return promptly before the interrupt is re-raised.

    ``on_completion`` (publication) runs only after every worker has returned
    (#1025).  Publishing a finished reviewer's body while a peer is still
    running would put that body on the PR/issue the peer can read mid-turn,
    so a late reviewer could echo it and panel agreement would stop being
    independent corroboration.  Completion order is still preserved for the
    publications themselves.

    With a ``spool``, no reviewer is ever invoked while a same-round peer's
    body is public:

    * every settled outcome (validated response, or a failure that settles the
      reviewer as unavailable) is persisted privately before the first post,
      and a rerun replays it instead of re-invoking the reviewer, so an
      interruption between posts is harmless;
    * when ``retry_bound`` reports that some reviewer must be re-invoked (a
      fatal failure or an incomplete review), nothing is published: the
      healthy outcomes stay in the spool and that failure is raised, so the
      retried reviewer later runs with no same-round peer body visible;
    * a reviewer still needing a fresh turn while ``public_peers`` is
      non-empty -- a lost or invalid spool record -- stops the run.

    A reviewer in ``already_posted`` has its own same-round record on the
    surface that resume rejected; it is re-invoked, never replayed.

    Callers may skip per-reviewer launch preparation for reviewers expected to
    replay.  ``prepare_fallback`` runs for any of them whose replay fell back
    to a fresh turn; an ``AgentLoopError`` it raises becomes that reviewer's
    turn failure instead of a launch.

    The round's spool is discarded only after every publication succeeded.
    """
    results: dict[AgentName, _ReviewerTurnResult] = {}
    replayed: set[AgentName] = set()
    replay_fallbacks: list[AgentName] = []
    if spool is not None and replay_turn is not None:
        for reviewer in pending:
            reviewer_name = agent_display_name(reviewer)
            if reviewer in already_posted:
                spool.remove(reviewer_name)
                continue
            fields = spool.load(reviewer_name)
            if fields is None:
                continue
            replay_fallbacks.append(reviewer)
            failure = _replay_spooled_failure(fields, reviewer_name)
            result = (
                _ReviewerTurnResult(reviewer_name=reviewer_name, error=failure)
                if failure is not None
                else replay_turn(reviewer, fields)
            )
            if result is None and spool.publication_state(reviewer_name)[0] in {"carrier", "malformed"}:
                # A frozen publication may already be partly public: a fresh
                # turn would read its own sidecars, and an unvalidated body must
                # never be published.  Stop with the record kept (#1258).
                raise PublicationResumeStop(
                    f"{reviewer_name}'s frozen same-round publication no longer validates in "
                    "this run's context, so it cannot be resumed. The spool record was kept, "
                    "nothing was posted, and no reviewer was launched; repair the round with "
                    "the partial-round recovery list (#1142)."
                )
            if result is not None:
                results[reviewer] = result
                replayed.add(reviewer)
    to_launch = [reviewer for reviewer in pending if reviewer not in replayed]
    if spool is not None:
        for reviewer in to_launch:
            peers = sorted(agent_display_name(peer) for peer in public_peers if peer != reviewer)
            if peers:
                raise _partial_round_refusal(
                    surface=spool.surface, number=spool.number, round_number=spool.round_number,
                    reviewer_name=agent_display_name(reviewer), public_peers=peers,
                    cause=_replay_unavailable_cause(
                        spool, agent_display_name(reviewer),
                        own_posted=reviewer in already_posted,
                        loaded_unreplayable=reviewer in replay_fallbacks,
                    ),
                    recovery=recovery,
                )
    if prepare_fallback is not None:
        for reviewer in [r for r in replay_fallbacks if r in to_launch]:
            try:
                prepare_fallback(reviewer)
            except AgentLoopError as exc:
                results[reviewer] = _ReviewerTurnResult(
                    reviewer_name=agent_display_name(reviewer), error=exc
                )
                to_launch.remove(reviewer)
    if to_launch:
        # A sequential run finishing a held round launches one reviewer at a
        # time (reviewer workdirs may be shared); publication is still withheld
        # until all of them return.
        executor = ThreadPoolExecutor(
            max_workers=min(len(to_launch), max_workers or len(to_launch)),
            thread_name_prefix=thread_name_prefix,
        )
        try:
            # copy_context: reviewer threads must see the run owner (Antigravity quota memory).
            futures = {
                executor.submit(contextvars.copy_context().run, run_turn, reviewer): reviewer
                for reviewer in to_launch
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        except KeyboardInterrupt:
            runner.terminate_active_processes()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    if spool is None:
        if on_completion is not None:
            for reviewer, result in results.items():
                on_completion(reviewer, result)
        return results

    settled: dict[AgentName, _ReviewerTurnResult] = {**(prelaunch_failures or {}), **results}
    retry_errors: dict[AgentName, AgentLoopError] = {}
    if retry_bound is not None:
        for reviewer, result in settled.items():
            error = retry_bound(reviewer, result)
            if error is not None:
                retry_errors[reviewer] = error
    for reviewer, result in settled.items():
        if reviewer in replayed or reviewer in retry_errors:
            continue
        reviewer_name = agent_display_name(reviewer)
        if result.error is not None:
            spool.store_failure(
                reviewer_name,
                message=str(result.error),
                failure_category=getattr(result.error, "failure_category", None),
            )
        elif result.response is not None:
            spool.store(reviewer_name, _spooled_response_fields(result.response))
    withheld = [
        reviewer for reviewer, result in results.items()
        if reviewer not in retry_errors and result.error is None and result.response is not None
    ]
    if retry_errors and withheld:
        order = {reviewer: index for index, reviewer in enumerate(configured_order or pending)}
        failed = sorted(retry_errors, key=lambda reviewer: order.get(reviewer, len(order)))
        if config is not None:
            log(
                config,
                "Withholding the same-round review(s) of "
                f"{', '.join(agent_display_name(r) for r in withheld)} until "
                f"{', '.join(agent_display_name(r) for r in failed)} completes an independent "
                "turn; rerun to retry",
            )
        for reviewer in failed:
            if isinstance(retry_errors[reviewer], QuotaResetExceededError):
                raise retry_errors[reviewer]
        raise retry_errors[failed[0]]
    if on_completion is not None:
        for reviewer, result in results.items():
            if on_completion(reviewer, result):
                # Published now: a rerun resumes (or, if the posted record is
                # rejected, re-invokes) it rather than replaying a stale outcome.
                spool.remove(agent_display_name(reviewer))
    spool.discard()
    return results


def _ensure_parallel_reviewer_workdirs(
    config: AgentLoopConfig, *, flag_name: str, role_label: str
) -> None:
    """Reject shared workdirs among concurrently scheduled reviewers (#475, #594).

    Deliberately NOT bypassed by --allow-shared-dir: concurrent git/tool
    activity from two agents in one worktree can race and corrupt it. A
    later, non-concurrent role (the discuss analyzer, the plan/PR coder) may
    still share a reviewer's directory because it only runs after the
    reviewer synchronization point.
    """
    seen: dict[Path, AgentName] = {}
    for reviewer in reviewers(config):
        path = get_backend(reviewer).workdir(config).resolve()
        other = seen.get(path)
        if other is not None:
            raise AgentLoopError(
                f"{flag_name} requires a distinct workdir per {role_label}: "
                f"{agent_display_name(other)} and {agent_display_name(reviewer)} "
                f"both resolve to {path}. Use separate clones/worktrees per {role_label} "
                f"or drop {flag_name}; --allow-shared-dir does not lift this "
                "requirement."
            )
        seen[path] = reviewer


def _ensure_parallel_discuss_workdirs(config: AgentLoopConfig) -> None:
    _ensure_parallel_reviewer_workdirs(config, flag_name="--discuss-parallel", role_label="debater")
