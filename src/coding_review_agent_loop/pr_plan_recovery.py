"""Recover an independently approved issue plan before entering PR review."""
from __future__ import annotations
from .config import AgentLoopConfig, reviewers
from .github import IssueContext
from .runner import Runner
from .plan_verification import plan_verification_inputs
from .round_state import _resume_plan_round
from .panel_evidence import _require_complete_canonical_plan_approval
from .execution_policy import ApprovedPlanContext, make_approved_plan_context
from .decomposition import approved_plan_hash

def _admitted_plan_comments(
    runner: Runner | None, config: AgentLoopConfig, issue_context: IssueContext,
) -> tuple[tuple[object, ...], tuple[str, int] | None]:
    """Plan history with plan board-change links admitted (#1377)."""
    from .plan_board_state import admitted_issue_comments

    return admitted_issue_comments(
        runner, config=config, issue_number=issue_context.number,
        comments=issue_context.comments,
    )


def _canonical_plan_without_handoff(
    config: AgentLoopConfig,
    issue_context: IssueContext,
    *,
    source_locator: str,
    error_message: str,
    runner: Runner | None = None,
) -> ApprovedPlanContext | None:
    """The completely approved canonical plan of an issue with no handoff record.

    Approval is verified against the approving plan board, never the PR
    invocation's board (#1373), resolving any recorded plan board change
    lineage (#1377).  ``None`` means the issue has no plan round; incomplete
    approval, including an open carried obligation, raises ``error_message``.
    """
    comments, trusted_actor = _admitted_plan_comments(runner, config, issue_context)
    plan_config, plan_comments = plan_verification_inputs(
        config, comments, issue_number=issue_context.number, trusted_actor=trusted_actor,
    )
    resumed_plan = _resume_plan_round(plan_comments, configured_reviewers=reviewers(plan_config))
    if resumed_plan is None:
        return None
    plan_text, resumed_plan_round = resumed_plan
    _require_complete_canonical_plan_approval(
        plan_comments,
        config=plan_config,
        plan_text=plan_text,
        plan_round=resumed_plan_round,
        human_requirements=issue_context.human_requirements,
        error_message=error_message,
    )
    # The shared approval classification: requirement binding (issue and
    # authoritative parent), a reconstructible ledger, inherited
    # admissibility and signed supersession (#1377).
    from .plan_verification import require_recoverable_plan_approval

    require_recoverable_plan_approval(
        runner, config=config, issue_context=issue_context, comments=comments,
        trusted_actor=trusted_actor, error_message=error_message,
    )
    return make_approved_plan_context(
        plan_text, source_locator=source_locator, expected_hash=approved_plan_hash(plan_text),
    )
