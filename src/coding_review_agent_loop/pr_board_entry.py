"""Read-only PR entry board-floor check (moved from pr_loop)."""
from __future__ import annotations
from typing import Sequence
from .agents.registry import agent_display_name
from .config import AgentLoopConfig, reviewers
from .board_amendment import collect_reviewer_board_amendments
from .reviewer_floor import board_floor_enabled
from .board_change import board_reentry_round, comments_bear_board_change_links
from .pr_board_state import resolve_pr_board_state
from .github import PullRequestReviewContext
from .runner import Runner
from .review_scheduling import make_contract, policy_capabilities
from .panel_evidence import _board_amendment_route_clause
from .pr_loop_support import _resume_pr_round_admitted, _pr_contract_drift_error, _pr_amendment_start_round


def _enforce_pr_board_floor(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_context: PullRequestReviewContext,
    pr_number: int,
    handoff_authorizations: Sequence[object] = (),
) -> None:
    """Resolve the PR board lineage read-only, then check the effective-board floor.

    Malformed, unauthenticated or not-yet-activatable lineage history (signed
    amendments and board-change links) fails closed here, and a board-change
    candidate is floor-checked against the lineage that would exist after
    posting it, so every caller refuses before any write (#1376).  Without a
    floor, signed amendments or link records there is nothing to refuse
    early: startup resolves the lineage before any agent invocation.
    """
    if not (
        board_floor_enabled(config)
        or comments_bear_board_change_links(pr_context.comments)
        or collect_reviewer_board_amendments(
            pr_context.comments, flow="pr", pr_number=pr_number, ignored_sink=[],
        )
    ):
        return
    capabilities = policy_capabilities(config.pr_review_policy)
    configured = reviewers(config)
    configured_contract = make_contract(
        tuple(agent_display_name(reviewer) for reviewer in configured),
        config.pr_review_policy,
        config.pr_review_broad_rules,
        (
            agent_display_name(config.primary_reviewer)
            if config.primary_reviewer is not None
            else None
        ),
    )

    def resume_round(effective_config: object, _lineage: object) -> int:
        resumed = _resume_pr_round_admitted(
            runner,
            config=config,
            pr_number=pr_number,
            comments=pr_context.comments,
            head_sha=pr_context.metadata.head_sha,
            configured_reviewers=reviewers(effective_config),
            reconciliation_mode=(
                "owner-scoped" if capabilities.owner_scoped_reconciliation else "aggregate"
            ),
        )
        return board_reentry_round(
            pr_context.comments, resumed.round_number if resumed is not None else None,
            signed_amendment=True,
        )

    resolve_pr_board_state(
        runner,
        config=config,
        pr_number=pr_number,
        comments=pr_context.comments,
        drift_error=lambda persisted, detail: _pr_contract_drift_error(
            persisted,
            detail,
            pr_number=pr_number,
            configured=configured_contract,
            start_round_number=lambda: _pr_amendment_start_round(
                pr_context, configured, capabilities, runner=runner, config=config,
            ),
            amendments_recognized=True,
        ),
        handoff_authorizations=handoff_authorizations,
        resume_round=resume_round,
        floor_refusal_hint=lambda persisted, refused: _board_amendment_route_clause(
            flow="pr",
            issue_number=None,
            pr_number=pr_number,
            persisted=persisted,
            configured=refused,
            start_round_number=lambda: _pr_amendment_start_round(
                pr_context, configured, capabilities, runner=runner, config=config,
            ),
            amendments_recognized=True,
        ),
    )
