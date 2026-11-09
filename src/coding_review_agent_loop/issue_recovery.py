"""Shared issue execution recovery, independent of lifecycle entry points."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable
from .pr_head_sync import get_observed_pr_review_context
from dataclasses import replace as dataclasses_replace
from .agents.registry import agent_display_name
from .config import AgentLoopConfig, reviewers
from .decomposition import approved_plan_hash, normalize_execution_recommendation
from .errors import AgentLoopError
from .expected_closure import reject_parent_from_contract, resolve_issue_contract
from .github import (
    IssueContext,
    post_trusted_pr_comment,
    validate_pr_body_does_not_close_issue,
    validate_pr_expected_closing_issues,
)
from .issue_pr_handoff import (
    find_latest_issue_pr_handoff,
    post_issue_pr_handoff_comment,
    require_pr_metadata_for_handoff,
    resolve_canonical_pr_for_issue,
)
from .issue_pr_provenance import IssuePrProvenanceScope
from .pr_contract import format_pr_contract_comment, make_pr_contract
from .logging import log
from .memory import prepare_agent_memory
from .runner import Runner
from .usage import RunUsageContext
from .round_metadata import ApprovedPlanContext, _plan_subject
from .round_state import (
    _resume_plan_round,
    make_approved_plan_context,
    recover_approved_plan_context,
)
from .protocol_markers import TrustedBody
from .response_validation import ResolvedExecution
from .panel_evidence import _plan_growth_verdict_for_hash
from .execution_policy import (
    _FreshChildProvenance,
    _extract_current_expected_closing_issue_ids,
    _current_execution_recommendation,
    _resolve_execution_policy,
    _print_dry_run_execution_preview,
    _print_execution_resolution_summary,
    _persist_execution_decision_if_needed,
    _preflight_fresh_staged_topology,
    _preflight_fresh_one_shot_recovery,
)
from .child_plan_binding import (
    _inherited_matrix_binding,
    _PlanSupersessionBinding,
    verified_retired_child_plan_hashes,
    _route_child_plan_handoff,
)
from .pr_loop import run_pr_loop


@dataclass(frozen=True)
class RecoveryContinuation:
    """Validated state from recovery when fresh work may continue."""

    config: AgentLoopConfig
    issue_context: IssueContext


def recover_issue_execution(
    runner: Runner, *, config: AgentLoopConfig, issue_number: int,
    issue_context: IssueContext, parent_issue_context: IssueContext | None,
    staged_parent_issue: int | None, fresh_child: _FreshChildProvenance | None,
    plan_first: bool, requested_policy: str,
    implement_after_approval: bool, usage_context: RunUsageContext,
    run_plan_first_loop: Callable[..., int],
) -> int | RecoveryContinuation:
    """Recover before fresh work; an integer (including zero) is a handled result.

    Entry points retain startup, telemetry and workdir claims. This helper owns
    plan selection, child admissibility, execution history and canonical PR
    recovery for both issue entry and parent-driven child resume.
    """
    recovered_plan_hash: str | None = None
    recovered_plan_additions: tuple[int, ...] | None = None
    recovered_plan_context: ApprovedPlanContext | None = None
    recorded_plan_handoff = None
    if plan_first:
        # Prefer the plan hash recorded by the issue-side handoff. A later
        # planning round may be unrelated to the PR already handed off, so
        # resuming the newest plan would silently change the implementation
        # contract. Fall back to the latest reconstructable round only when
        # no approved-plan handoff has selected a plan yet.
        recorded_plan_handoff = find_latest_issue_pr_handoff(
            issue_context.comments,
            issue_number=issue_number,
            repo=config.repo,
        )
        if (
            recorded_plan_handoff is not None
            and recorded_plan_handoff.flow == "approved-plan-implementation"
            and recorded_plan_handoff.plan_hash
        ):
            recovered_plan_hash = recorded_plan_handoff.plan_hash
            recovered_plan_context = recover_approved_plan_context(
                issue_context.comments,
                expected_hash=recovered_plan_hash,
            )
            if recovered_plan_context.is_available:
                recovered_plan_additions = _extract_current_expected_closing_issue_ids(
                    recovered_plan_context.canonical_text or ""
                )
        else:
            # This is a comment-only reconstruction. It must happen before
            # memory preparation or any agent invocation so an existing
            # plan can be checked without re-planning.
            from .plan_issue_scope import comments_bear_scope_boundary, resolve_plan_scope

            # Only the current planning scope's plan is reconstructed (#1390).
            # Admit explicitly, and only when a boundary-shaped record is
            # present: a history without one needs no GitHub read (#1987).
            scope_comments: tuple[object, ...] = tuple(issue_context.comments)
            scope_actor: tuple[str, int] | None = None
            if comments_bear_scope_boundary(scope_comments):
                from .plan_board_state import admitted_issue_comments

                scope_comments, scope_actor = admitted_issue_comments(
                    runner, config=config, issue_number=issue_number,
                    comments=scope_comments,
                )
            recovered_plan_state = _resume_plan_round(
                issue_context.comments,
                configured_reviewers=reviewers(config),
                min_round_number=resolve_plan_scope(
                    scope_comments, scope_actor, issue_number=issue_number,
                ),
            )
            if recovered_plan_state is not None:
                recovered_plan_hash = approved_plan_hash(recovered_plan_state[0])
                recovered_plan_context = make_approved_plan_context(
                    recovered_plan_state[0],
                    source_locator=f"issue #{issue_number} reconstructed plan round",
                    expected_hash=recovered_plan_hash,
                )
                recovered_plan_additions = _extract_current_expected_closing_issue_ids(
                    recovered_plan_state[0]
                )

    # A planning child's handed-off plan is judged against its inherited
    # parent rows before the canonical PR is resolved (#936).  A matching
    # signed supersession record reopens planning whether or not the plan
    # is admissible (#985); without one, an inadmissible plan fails closed
    # and an admissible one resumes its PR.  A same-PR plan replacement
    # must verify.
    plan_supersession: _PlanSupersessionBinding | None = None
    if (
        plan_first
        and fresh_child is not None
        and fresh_child.route.is_planning
        and recorded_plan_handoff is not None
        and recorded_plan_handoff.flow == "approved-plan-implementation"
        and recovered_plan_context is not None
        and recovered_plan_context.is_available
    ):
        plan_supersession = _route_child_plan_handoff(
            runner,
            config=config,
            issue_number=issue_number,
            issue_context=issue_context,
            fresh_child=fresh_child,
            handoff=recorded_plan_handoff,
            child_plan_context=recovered_plan_context,
        )
        if plan_supersession is not None and config.dry_run:
            print(
                f"Issue #{issue_number}: dry run; approved child plan "
                f"{plan_supersession.superseded_hash} would be re-planned under its signed "
                f"supersession record and PR #{plan_supersession.pr_number} rebound."
            )
            return 0
    if plan_supersession is not None:
        memory = prepare_agent_memory(runner, config)
        return run_plan_first_loop(
            runner,
            issue_number=issue_number,
            config=config,
            memory=memory,
            issue_context=issue_context,
            requested_policy=requested_policy,
            implement_after_approval=implement_after_approval,
            usage_context=usage_context,
            inherited_matrix_binding=_inherited_matrix_binding(
                parent_issue=fresh_child.parent_issue,
                stage_id=fresh_child.stage_id,
                parent_plan_context=fresh_child.parent_plan_context,
            ),
            plan_supersession=plan_supersession,
        )
    # A verified signed re-plan retires the execution decision recorded
    # under each superseded plan (#988); the resumed run then records the
    # decision for the rebound plan instead of failing on the old one.
    retired_plan_hashes: frozenset[str] = frozenset()
    if (
        plan_first
        and fresh_child is not None
        and fresh_child.route.is_planning
        and recorded_plan_handoff is not None
        and recorded_plan_handoff.flow == "approved-plan-implementation"
    ):
        retired_plan_hashes = verified_retired_child_plan_hashes(
            issue_context.comments,
            repo=config.repo,
            parent_plan_context=fresh_child.parent_plan_context,
            child_issue=issue_number,
            parent_issue=fresh_child.parent_issue,
            stage_id=fresh_child.stage_id,
            pr_number=recorded_plan_handoff.pr_number,
        )

    # Rebuild durable retirement before open-PR recovery too: reopened
    # history must not hide behind a current-plan canonical handoff.
    if plan_first and recovered_plan_hash is not None:
        from .decomposition import live_execution_decisions
        from .execution_retirement import assess_execution_history
        from .execution_policy import _approved_one_shot_split_keys
        if any(decision.plan_hash == recovered_plan_hash and decision.retires_plan_hashes
               for decision in live_execution_decisions(issue_context.comments, parent_issue=issue_number)):
            history = assess_execution_history(
                runner, config=config, issue_number=issue_number,
                approved_split_keys=_approved_one_shot_split_keys,
                comments=issue_context.comments, plan_hash=recovered_plan_hash, newly_retired=(),
            )
            if history is not None:
                config = dataclasses_replace(config, execution_history=history)
                issue_context = history.view(issue_context)

    # Resolve the canonical AGENT_ISSUE_PR_HANDOFF record (or, failing
    # that, the legacy exactly-one-open-PR search) before invoking a
    # coder in either direct or plan-first mode, so a rerun after an
    # interrupted PR review resumes that PR instead of creating a
    # duplicate (#589).
    resolved_pr = resolve_canonical_pr_for_issue(
        runner,
        config=config,
        issue_number=issue_number,
        issue_context=issue_context,
        defer_terminal=plan_first,
        expected_fallback_scope=(
            None
            if plan_first and recovered_plan_hash is None
            else IssuePrProvenanceScope(
                repository=config.repo,
                issue_number=issue_number,
                flow="approved" if plan_first else "direct",
                approved_plan_hash=recovered_plan_hash if plan_first else None,
            )
        ),
    )
    if resolved_pr is not None:
        recovered_execution: ResolvedExecution | None = None
        recovered_topology = None
        if plan_first:
            recovered_recommendation = None
            if recovered_plan_context is not None and recovered_plan_context.canonical_text:
                recovered_recommendation = _current_execution_recommendation(
                    recovered_plan_context.canonical_text,
                    issue_context.comments,
                )
            # Existing explicit modes retain their historical recovery for
            # legacy plans.  A fresh recommendation, however, is an
            # approval-bound contract and must pass the same policy and
            # topology checks as the initial implementation route.
            recovered_execution = _resolve_execution_policy(
                config,
                requested_policy=requested_policy,
                recommendation=recovered_recommendation,
            )
            if (
                recovered_execution.recommendation is not None
                and recovered_execution.strategy == "staged"
            ):
                if recovered_plan_context is None or not recovered_plan_context.canonical_text:
                    raise AgentLoopError(
                        "Fresh staged execution recovery has no reconstructable approved plan; "
                        "repair the handoff or rerun plan-first planning before resuming."
                    )
                recovered_topology = normalize_execution_recommendation(
                    recovered_execution.recommendation,
                    approved_plan=recovered_plan_context.canonical_text or "",
                    plan_subject=_plan_subject(recovered_plan_context.canonical_text or ""),
                )
                _preflight_fresh_staged_topology(
                    runner,
                    issue_number=issue_number,
                    approved_plan=recovered_plan_context.canonical_text or "",
                    config=config,
                    issue_context=issue_context,
                    mode=recovered_execution.action,
                    normalized_topology=recovered_topology,
                    retired_plan_hashes=retired_plan_hashes,
                )
            elif (
                recovered_execution.recommendation is not None
                and recovered_execution.strategy == "one-shot"
            ):
                if recovered_plan_context is None or not recovered_plan_context.canonical_text:
                    raise AgentLoopError(
                        "Fresh one-shot execution recovery has no reconstructable approved plan; "
                        "repair the handoff or rerun plan-first planning before resuming."
                    )
                _preflight_fresh_one_shot_recovery(
                    runner,
                    issue_number=issue_number,
                    approved_plan=recovered_plan_context.canonical_text or "",
                    config=config,
                    issue_context=issue_context,
                    recommendation=recovered_execution.recommendation,
                    retired_plan_hashes=retired_plan_hashes,
                )
        closing_contract = resolve_issue_contract(
            primary_issue=issue_number,
            cli_additions=config.expected_closing_issue_ids,
            plan_additions=recovered_plan_additions,
            recovered=(
                resolved_pr.metadata.expected_closing_issue_ids
                if resolved_pr.metadata is not None
                else (issue_number,)
            ),
            supersede=config.supersede_expected_closing_contract,
        )
        reject_parent_from_contract(closing_contract, parent_issue=staged_parent_issue)
        config = dataclasses_replace(
            config,
            expected_closing_issue_ids=closing_contract.issue_ids,
            expected_closing_contract_resolved=True,
        )
        resolved_metadata = resolved_pr.metadata
        if (
            plan_first
            and resolved_pr.source == "canonical"
            and resolved_metadata is not None
            and resolved_metadata.flow == "approved-plan-implementation"
            and recovered_plan_hash is not None
            and resolved_metadata.plan_hash != recovered_plan_hash
        ):
            raise AgentLoopError(
                f"Canonical approved-plan handoff for issue #{issue_number} points to "
                f"PR #{resolved_pr.pr_number} with plan hash {resolved_metadata.plan_hash}, "
                f"but the reconstructable approved plan has hash {recovered_plan_hash}. "
                f"Review the recorded PR with `agent-loop pr {resolved_pr.pr_number}` or "
                "remove the stale handoff marker before rerunning issue mode."
            )
        if recovered_execution is not None:
            if recovered_execution.recommendation is not None:
                _print_execution_resolution_summary(
                    issue_number=issue_number,
                    resolved=recovered_execution,
                    normalized_topology=recovered_topology,
                )
            if config.dry_run:
                _print_dry_run_execution_preview(
                    issue_number=issue_number,
                    resolved=recovered_execution,
                    normalized_topology=recovered_topology,
                )
                return 0
            if recovered_plan_context is not None and recovered_plan_context.canonical_text:
                _persist_execution_decision_if_needed(
                    runner,
                    config=config,
                    issue_number=issue_number,
                    current_plan=recovered_plan_context.canonical_text,
                    issue_comments=issue_context.comments,
                    recommendation=recovered_execution.recommendation,
                    requested_policy=recovered_execution.requested_policy,
                    resolved_execution=recovered_execution,
                    retired_plan_hashes=retired_plan_hashes,
                )
        if plan_first and resolved_pr.source == "canonical" and resolved_metadata is not None:
            if resolved_metadata.flow == "approved-plan-implementation" and recovered_plan_hash is None:
                log(
                    config,
                    f"WARNING: issue #{issue_number} is resuming canonical approved-plan PR "
                    f"#{resolved_pr.pr_number} using recorded plan hash {resolved_metadata.plan_hash}; "
                    "no reconstructable prior plan round was found.",
                )
        if plan_first and resolved_pr.source == "legacy-closing-reference" and recovered_plan_hash is None:
            raise AgentLoopError(
                f"Found unique legacy PR #{resolved_pr.pr_number} with strong closing evidence "
                f"for issue #{issue_number}, but no approved plan round is reconstructable. "
                "Issue-mode plan-first recovery cannot invent plan provenance; review the PR "
                f"directly with `agent-loop pr {resolved_pr.pr_number}` or rerun direct issue mode."
            )
        if (
            recovered_execution is not None
            and recovered_execution.recommendation is not None
            and recovered_execution.action == "plan-only"
        ):
            print(
                f"Issue #{issue_number} plan-first recovery resolved to plan-only; "
                f"PR #{resolved_pr.pr_number} review was not started."
            )
            return 0
        log(
            config,
            f"Issue #{issue_number}: resuming PR #{resolved_pr.pr_number} review instead of "
            f"invoking {agent_display_name(config.coder)}.",
        )
        if config.triage and fresh_child is None:
            log(
                config,
                f"Issue #{issue_number}: --triage skipped; the recorded route of existing "
                f"PR #{resolved_pr.pr_number} wins.",
            )
        log(
            config,
            f"Issue #{issue_number}: PR #{resolved_pr.pr_number} association source="
            f"{resolved_pr.source}, evidence={resolved_pr.evidence_summary}",
        )
        if staged_parent_issue is not None:
            validate_pr_body_does_not_close_issue(
                runner,
                config=config,
                pr_number=resolved_pr.pr_number,
                issue_number=staged_parent_issue,
            )
        # Keep rejected issue-implementation evidence rejected during an
        # explicit managed recovery. The PR loop authenticates the
        # authorization record independently of this legacy association.
        if resolved_pr.source == "legacy-closing-reference":
            pr_context = get_observed_pr_review_context(runner, config=config, pr_number=resolved_pr.pr_number)
            validate_pr_expected_closing_issues(
                runner,
                config=config,
                pr_number=resolved_pr.pr_number,
                expected_issue_ids=closing_contract.issue_ids,
                body=pr_context.metadata.body,
                reject_unexpected=config.managed_ci,
            )
            pr_url, pr_head_sha = require_pr_metadata_for_handoff(pr_context.metadata)
            pr_contract = make_pr_contract(
                repository=config.repo,
                pr_number=resolved_pr.pr_number,
                origin_flow=(
                    "approved-plan-implementation"
                    if plan_first and recovered_plan_hash is not None
                    else "issue-implementation"
                ),
                primary_issue_number=issue_number,
                expected_closing_issue_ids=closing_contract.issue_ids,
                supersedes_hash=closing_contract.supersedes_hash,
            )
            post_trusted_pr_comment(
                runner,
                config=config,
                pr_number=resolved_pr.pr_number,
                body=TrustedBody.canonical(
                    format_pr_contract_comment(pr_contract),
                    expected_tokens=("AGENT_PR_EXPECTED_CLOSING_ISSUES",),
                ),
            )
            if not config.managed_ci:
                post_issue_pr_handoff_comment(
                    runner,
                    config=config,
                    issue_number=issue_number,
                    pr_number=resolved_pr.pr_number,
                    pr_url=pr_url,
                    pr_head_sha=pr_head_sha,
                    flow=(
                        "approved-plan-implementation"
                        if plan_first and recovered_plan_hash is not None
                        else "issue-implementation"
                    ),
                    plan_hash=recovered_plan_hash if plan_first else None,
                    expected_closing_issue_ids=closing_contract.issue_ids,
                    supersedes_hash=closing_contract.supersedes_hash,
                    plan_growth_verdict=(
                        _plan_growth_verdict_for_hash(
                            config,
                            plan_hash=recovered_plan_hash,
                            comment_sources=(issue_context.comments,),
                        )
                        if plan_first and recovered_plan_hash is not None
                        else None
                    ),
                )
        return run_pr_loop(
            runner,
            pr_number=resolved_pr.pr_number,
            config=config,
            issue_context=issue_context,
            approved_plan_context=recovered_plan_context,
            parent_issue_context=parent_issue_context,
            usage_context=usage_context,
            managed_ci_issue_number=issue_number,
        )

    return RecoveryContinuation(config, issue_context)
