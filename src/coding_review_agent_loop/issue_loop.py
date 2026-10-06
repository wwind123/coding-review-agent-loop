"""The issue and task loop entry points.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1203); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import sys
from dataclasses import replace as dataclasses_replace
from .agents.registry import agent_display_name
from .workdir_claims import claimed_run
from .config import (
    AgentLoopConfig,
    ensure_agent_workdirs,
    resolve_base_branch,
    reviewers,
    sync_coder_base_before_implementation,
)
from .decomposition import (
    approved_plan_hash,
    normalize_execution_recommendation,
    risk_matrix_row_ids_for_owner,
)
from .protocol import EXECUTION_DISPOSITION_PLANNING
from .errors import (
    AgentLoopError,
)
from .expected_closure import (
    reject_parent_from_contract,
    resolve_issue_contract,
)
from .github import (
    deduplicate_human_requirements,
    get_issue_context,
    get_pr_review_context,
    post_pr_comment,
    post_trusted_pr_contract_record,
    post_trusted_pr_comment,
    reset_authenticated_github_actor,
    reject_forged_protocol_markers,
    validate_open_issue,
    validate_open_pr,
    validate_pr_body_does_not_close_issue,
    validate_pr_expected_closing_issues,
    validate_pr_references_issue,
)
from .issue_pr_handoff import (
    find_latest_issue_pr_handoff,
    post_issue_pr_handoff_comment,
    require_pr_metadata_for_handoff,
    resolve_canonical_pr_for_issue,
)
from .integration_close import reconcile_merged_integration_child
from .issue_pr_provenance import IssuePrProvenanceScope
from .pr_contract import (
    format_pr_contract_comment,
    make_pr_contract,
)
from .logging import log
from .memory import prepare_agent_memory
from .managed_ci import (
    AuthenticatedIssueCreatedHandoff,
    authenticate_issue_created_handoff,
    preflight_managed_ci_creation,
)
from .prompts import (
    build_issue_prompt,
    build_task_clarification_prompt,
    build_task_prompt,
    render_coder_human_requirements_prompt_context,
)
from .protocol import (
    StructuredIssueImplementation,
    StructuredTaskResult,
)
from .runner import Runner
from .salvage import (
    SalvageContext,
    latest_salvage_context,
)
from .usage import RunUsageContext
from .workdirs import active_workdir
from .workdir_guard import (
    validate_assigned_head_advanced,
    validate_test_observation_citations_within_workdir,
)
from .comment_rendering import (
    normalize_freeform_signature,
    render_public_agent_comment,
)
from .round_state import (
    ApprovedPlanContext,
    PostedRoundMetadata,
    _attach_round_metadata,
    _plan_subject,
    _resume_plan_round,
    make_approved_plan_context,
    recover_approved_plan_context,
)
from .protocol_markers import (
    TrustedBody,
)
from .agent_failure import (
    ISSUE_IMPLEMENTATION_SALVAGE_SCOPE,
    TASK_IMPLEMENTATION_SALVAGE_SCOPE,
    _metadata_identity_fields,
)
from .architecture_contract import (
    _freeze_prompt_architecture,
    _architecture_metadata_fields,
    _test_observation_degradation_fields,
    _TerminalNoPrImplementation,
    _TerminalIssueImplementationConflict,
    _architecture_mode_validators,
)
from .validated_agent import (
    _new_usage_context,
    _begin_run_telemetry,
    _end_run_telemetry,
    _persist_usage_summary,
    CompletionRecoveryPolicy,
    _run_validated_agent,
)
from .response_validation import (
    ResolvedExecution,
    _validate_issue_implementation_response,
    _derive_authenticated_risk_evidence_for_coder,
    _post_no_pr_implementation_terminal_comment,
    _post_structured_issue_implementation_terminal_comment,
    _require_task_implementation_result,
    _validate_response_tests_with_post_pr_context,
    _degrade_out_of_checkout_tests,
    _validate_structured_response_tests_with_post_pr_context,
    _validate_structured_response_observations_with_post_pr_context,
)
from .panel_evidence import (
    _plan_growth_verdict_for_hash,
)
from .execution_policy import (
    _extract_current_expected_closing_issue_ids,
    _current_execution_recommendation,
    _normalize_requested_execution_policy,
    _resolve_execution_policy,
    _resolve_fresh_child_provenance,
    _child_resume_hint,
    _refuse_plain_mode_over_planning,
    _post_child_planning_handoff,
    _print_dry_run_execution_preview,
    _print_execution_resolution_summary,
    _persist_execution_decision_if_needed,
    _preflight_fresh_staged_topology,
    _preflight_fresh_one_shot_recovery,
    _infer_staged_parent_issue,
)
from .child_plan_binding import (
    _inherited_matrix_binding,
    _PlanSupersessionBinding,
    verified_retired_child_plan_hashes,
    _route_child_plan_handoff,
)
from .pr_loop_support import (
    _print_unprotected_managed_ci_warning,
    _read_assigned_workdir_head,
)
from .pr_loop import run_pr_loop
from .issue_implementation import (
    _embed_pr_contract_marker,
    _advisory_issue_pr_provenance,
    _publish_issue_authorization_with_recovery,
)
from .plan_first_loop import (
    _dispatch_decomposition_child,
    _run_plan_first_loop,
)


@claimed_run("issue", "issue_number")
def run_issue_loop(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    plan_first: bool = False,
    implement_after_approval: bool = False,
    requested_policy: str | None = None,
    usage_context: RunUsageContext | None = None,
) -> int:
    owned_usage_context = usage_context is None
    usage_context = usage_context or _new_usage_context(config)
    reset_authenticated_github_actor(runner)
    telemetry_token = _begin_run_telemetry(
        runner, config, usage_context, owned_usage_context, issue_number=issue_number
    )
    try:
        requested_policy = _normalize_requested_execution_policy(
            config,
            requested_policy=requested_policy,
            implement_after_approval=implement_after_approval,
        )
        if config.plan_execution_mode != requested_policy:
            # Keep prompt construction and all pre-approval reads aligned with
            # the one normalized requested policy.  The reviewed recommendation
            # is still resolved only at the approval boundary.
            config = dataclasses_replace(config, plan_execution_mode=requested_policy)
        config = resolve_base_branch(config, runner)
        ensure_agent_workdirs(config, runner)
        config = _freeze_prompt_architecture(runner, config)
        log(config, f"Validating issue #{issue_number}")
        validate_open_issue(runner, config=config, issue_number=issue_number)
        issue_context = get_issue_context(runner, config=config, issue_number=issue_number)
        # Before any routing (including the staged direct-child dispatch): an
        # interrupted closure after a confirmed merge into an integration base
        # is finished here, without a coder, review or merge replay.
        if reconcile_merged_integration_child(
            runner, config=config, issue_number=issue_number, issue_context=issue_context
        ):
            return 0
        staged_parent_issue = _infer_staged_parent_issue(issue_context)
        parent_issue_context = (
            get_issue_context(runner, config=config, issue_number=staged_parent_issue)
            if staged_parent_issue is not None
            else None
        )

        # Direct child-issue entry (#808): a materialized fresh decomposition
        # child is routed through the same seam as parent dispatch.  CLI flags
        # never switch a recorded or declared route.
        fresh_child = _resolve_fresh_child_provenance(
            issue_context=issue_context,
            parent_issue_context=parent_issue_context,
        )
        if fresh_child is not None:
            route = fresh_child.route
            log(
                config,
                f"Issue #{issue_number}: fresh decomposition child of #{fresh_child.parent_issue} "
                f"stage `{fresh_child.stage_id}` resolved route `{route.disposition}` "
                f"(origin={route.origin})",
            )
            if route.is_human:
                raise AgentLoopError(
                    f"Issue #{issue_number} is stage `{fresh_child.stage_id}` of parent "
                    f"#{fresh_child.parent_issue} with automation "
                    f"`{fresh_child.created.phase.automation}`; human-owned stages are never "
                    "implemented or planned by agent-loop."
                )
            if route.is_direct:
                if plan_first:
                    raise AgentLoopError(
                        f"Issue #{issue_number} is stage `{fresh_child.stage_id}` of parent "
                        f"#{fresh_child.parent_issue} with "
                        f"{'recorded' if fresh_child.handoff is not None else 'reviewed'} "
                        "disposition `direct-implementation`; `--plan-first` cannot switch it to "
                        f"child planning. Rerun `agent-loop issue {issue_number}` without "
                        "`--plan-first`, or (before dispatch) post a signed "
                        "child-execution-disposition-override record on the parent issue."
                    )
                memory = prepare_agent_memory(runner, config)
                return _dispatch_decomposition_child(
                    runner,
                    config=config,
                    memory=memory,
                    usage_context=usage_context,
                    parent_issue=fresh_child.parent_issue,
                    approved_plan=fresh_child.approved_plan,
                    plan_hash=fresh_child.plan_hash,
                    plan_subject=fresh_child.plan_subject,
                    recommendation=fresh_child.recommendation,
                    approved_plan_context=fresh_child.parent_plan_context,
                    created=fresh_child.created,
                    phase_index=fresh_child.phase_index,
                    route=route,
                    child_issue_context=issue_context,
                    parent_issue_context=parent_issue_context,
                    coder_session_id=None,
                    existing_handoff=fresh_child.handoff,
                )
            if not plan_first:
                raise AgentLoopError(
                    f"Issue #{issue_number} is stage `{fresh_child.stage_id}` of parent "
                    f"#{fresh_child.parent_issue} with "
                    f"{'recorded' if fresh_child.handoff is not None else 'reviewed'} "
                    "disposition `requires-child-planning`; plain issue mode cannot switch it to "
                    "direct implementation. Rerun "
                    f"`{_child_resume_hint(issue_number, EXECUTION_DISPOSITION_PLANNING)}`, or "
                    "(before dispatch) post a signed child-execution-disposition-override record "
                    "on the parent issue."
                )
            if fresh_child.handoff is None and not config.dry_run:
                # Post the planning handoff before any planning agent runs,
                # using the same identity and override-digest rules as
                # parent dispatch; a rerun finds it and posts nothing new.
                _post_child_planning_handoff(
                    runner,
                    config=config,
                    parent_issue=fresh_child.parent_issue,
                    plan_hash=fresh_child.plan_hash,
                    plan_subject=fresh_child.plan_subject,
                    phase_index=fresh_child.phase_index,
                    created=fresh_child.created,
                    recommendation=fresh_child.recommendation,
                    inherited_matrix_row_ids=risk_matrix_row_ids_for_owner(
                        fresh_child.parent_plan_context.risk_test_matrix_payload
                        if fresh_child.parent_plan_context.matrix_available else None,
                        fresh_child.stage_id,
                    ),
                    override_digest=route.override_digest,
                )
                parent_issue_context = get_issue_context(
                    runner, config=config, issue_number=fresh_child.parent_issue
                )

        if not plan_first:
            # A decomposition child was routed above; this is the structurally
            # identical top-level case, which must fail closed too (#1088).
            _refuse_plain_mode_over_planning(
                runner,
                config=config,
                issue_number=issue_number,
                projection_comments=issue_context.comments,
            )

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
                recovered_plan_state = _resume_plan_round(
                    issue_context.comments, configured_reviewers=reviewers(config)
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
            return _run_plan_first_loop(
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
                pr_context = get_pr_review_context(runner, config=config, pr_number=resolved_pr.pr_number)
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

        memory = prepare_agent_memory(runner, config)
        if plan_first:
            return _run_plan_first_loop(
                runner,
                issue_number=issue_number,
                config=config,
                memory=memory,
                issue_context=issue_context,
                requested_policy=requested_policy,
                implement_after_approval=implement_after_approval,
                usage_context=usage_context,
                inherited_matrix_binding=(
                    _inherited_matrix_binding(
                        parent_issue=fresh_child.parent_issue,
                        stage_id=fresh_child.stage_id,
                        parent_plan_context=fresh_child.parent_plan_context,
                    )
                    if fresh_child is not None and fresh_child.route.is_planning
                    else None
                ),
            )

        closing_contract = resolve_issue_contract(
            primary_issue=issue_number,
            cli_additions=config.expected_closing_issue_ids,
            plan_additions=None,
            recovered=None,
            supersede=config.supersede_expected_closing_contract,
        )
        reject_parent_from_contract(closing_contract, parent_issue=staged_parent_issue)
        config = dataclasses_replace(
            config,
            expected_closing_issue_ids=closing_contract.issue_ids,
            expected_closing_contract_resolved=True,
        )

        # The first issue snapshot was used for validation and provenance; the
        # implementation handoff must use fresh target and parent snapshots.
        issue_context = get_issue_context(runner, config=config, issue_number=issue_number)
        # A concurrent plan-first run may have approved since the first
        # snapshot; recheck the one the coder is dispatched from (#1088).
        _refuse_plain_mode_over_planning(
            runner,
            config=config,
            issue_number=issue_number,
            projection_comments=issue_context.comments,
        )
        if parent_issue_context is not None:
            parent_issue_context = get_issue_context(
                runner, config=config, issue_number=parent_issue_context.number
            )
        implementation_requirements = deduplicate_human_requirements(
            [
                *(parent_issue_context.human_requirements if parent_issue_context is not None else ()),
                *issue_context.human_requirements,
            ]
        )
        implementation_human_requirements_context = render_coder_human_requirements_prompt_context(
            implementation_requirements,
        )
        sync_coder_base_before_implementation(config, runner)
        config = _freeze_prompt_architecture(runner, config)
        managed_ci_creation_intent = None
        if config.managed_ci:
            # Direct issue mode is intentionally the only new creation path.
            # A plain `issue --auto-merge` invocation keeps its historical
            # ordinary opening behavior unless it uses plan-first.
            managed_ci_creation_intent = preflight_managed_ci_creation(
                runner, config=config, issue_number=issue_number
            )
            if managed_ci_creation_intent is not None and managed_ci_creation_intent.audit_nonce:
                _print_unprotected_managed_ci_warning(managed_ci_creation_intent.protection_mode)
        assigned_head_before = _read_assigned_workdir_head(runner, config)
        salvage_summary = latest_salvage_context(
            config.log_dir,
            issue_context.comments,
            repo=config.repo,
            issue_number=issue_number,
            scope=ISSUE_IMPLEMENTATION_SALVAGE_SCOPE,
        )
        coder_response = _run_validated_agent(
            runner,
            agent=config.coder,
            config=config,
            prompt=build_issue_prompt(
                issue_number,
                config,
                memory,
                issue_context=issue_context,
                salvage_summary=salvage_summary,
                staged_parent_issue=staged_parent_issue,
                managed_ci_creation_intent=managed_ci_creation_intent,
                parent_issue_context=parent_issue_context,
            ),
            marker_description="structured issue_implementation result, blocking, or clarification",
            require_architecture_impact_contract=True,
            **_architecture_mode_validators(lambda mode: lambda text: _validate_issue_implementation_response(
                text,
                human_requirements=implementation_requirements,
                require_architecture_impact=True, architecture_status_mode=mode,
            )),
            usage_context=usage_context,
            # Without the exact coder role a sandboxed run hands this committing
            # turn the fail-closed read-only grant (#1077).
            role="coder",
            use_repair=True,
            repair_expected_kind="issue_implementation",
            repair_surfaced_requirement_ids=implementation_human_requirements_context.surfaced_requirement_ids,
            repair_requires_direct_discussion_ack=implementation_human_requirements_context.requires_direct_discussion_ack,
            salvage_context=SalvageContext(
                repo=config.repo,
                issue_number=issue_number,
                scope=ISSUE_IMPLEMENTATION_SALVAGE_SCOPE,
                agent=config.coder,
                run_id=usage_context.run_id,
            ),
            operation_description="issue implementation",
            completion_recovery=CompletionRecoveryPolicy(
                issue_number=issue_number,
                issue_context=issue_context,
                approved_plan_context=None,
                parent_issue_context=parent_issue_context,
                human_requirements=implementation_requirements,
            ),
            managed_ci_recovery_protection=(
                managed_ci_creation_intent.protection_mode
                if managed_ci_creation_intent is not None else None
            ),
        )
        coder_output = coder_response.text
        coder_session_id = coder_response.session_id
        implementation_result = coder_response.marker_value
        if isinstance(implementation_result, _TerminalIssueImplementationConflict):
            implementation_result = _TerminalIssueImplementationConflict(
                _degrade_out_of_checkout_tests(implementation_result.parsed, config=config)
            )
            validate_test_observation_citations_within_workdir(
                implementation_result.parsed.test_observations,
                assigned_workdir=active_workdir(config),
            )
            _post_structured_issue_implementation_terminal_comment(
                runner,
                config=config,
                issue_number=issue_number,
                parsed=implementation_result.parsed,
                model_used=coder_response.model_used,
            )
            raise AgentLoopError(
                "Coder implementation result was not accepted for handoff because a signed "
                "human requirement is blocked."
            )
        if isinstance(implementation_result, StructuredIssueImplementation):
            if implementation_result.pr_number is None:
                implementation_result = _degrade_out_of_checkout_tests(
                    implementation_result, config=config
                )
                validate_test_observation_citations_within_workdir(
                    implementation_result.test_observations,
                    assigned_workdir=active_workdir(config),
                )
                _post_structured_issue_implementation_terminal_comment(
                    runner,
                    config=config,
                    issue_number=issue_number,
                    parsed=implementation_result,
                    model_used=coder_response.model_used,
                )
                raise AgentLoopError(
                    "Coder did not create a valid PR; implementation is blocking."
                )
            pr_number = implementation_result.pr_number
        else:
            # Clarification remains the legacy terminal alternative.
            if isinstance(implementation_result, _TerminalNoPrImplementation):
                _post_no_pr_implementation_terminal_comment(
                    runner,
                    config=config,
                    issue_number=issue_number,
                    coder_response=coder_response,
                )
                raise AgentLoopError(
                    "Coder did not create a valid PR; implementation is "
                    + implementation_result.state
                    + "."
                )
            raise AgentLoopError("Issue implementation validator returned an unknown result type.")
        validate_assigned_head_advanced(
            before_head=assigned_head_before,
            after_head=_read_assigned_workdir_head(runner, config),
            assigned_workdir=active_workdir(config),
        )
        log(config, f"{agent_display_name(config.coder)} reported PR #{pr_number}; validating it is open")
        validate_open_pr(runner, config=config, pr_number=pr_number)
        initial_pr_context = get_pr_review_context(runner, config=config, pr_number=pr_number)
        managed_ci_handoff: AuthenticatedIssueCreatedHandoff | None = None
        if managed_ci_creation_intent is not None:
            managed_ci_handoff = authenticate_issue_created_handoff(
                runner,
                config=config,
                intent=managed_ci_creation_intent,
                issue_number=issue_number,
                pr_number=pr_number,
                metadata=initial_pr_context.metadata,
            )
            if managed_ci_handoff.override_nonce is not None:
                config = dataclasses_replace(
                    config,
                    managed_ci_expected_override_nonce=managed_ci_handoff.override_nonce,
                )
            managed_ci_handoff = _publish_issue_authorization_with_recovery(
                runner,
                config=config,
                handoff=managed_ci_handoff,
                metadata=initial_pr_context.metadata,
                issue_number=issue_number,
            )
        else:
            reject_forged_protocol_markers(
                initial_pr_context.metadata.body or "",
                surface=f"pull-request #{pr_number} body",
            )
        if isinstance(implementation_result, StructuredIssueImplementation):
            implementation_result = _validate_structured_response_tests_with_post_pr_context(
                implementation_result,
                runner=runner,
                config=config,
                pr_number=pr_number,
            )
            _validate_structured_response_observations_with_post_pr_context(
                implementation_result.test_observations,
                runner=runner,
                config=config,
                pr_number=pr_number,
            )
        initial_pr_metadata = initial_pr_context.metadata
        validate_pr_references_issue(
            runner,
            config=config,
            pr_number=pr_number,
            issue_number=issue_number,
            staged_parent_issue=staged_parent_issue,
            body=initial_pr_metadata.body,
        )
        validate_pr_expected_closing_issues(
            runner,
            config=config,
            pr_number=pr_number,
            expected_issue_ids=closing_contract.issue_ids,
            body=initial_pr_metadata.body,
        )
        _advisory_issue_pr_provenance(
            runner,
            config=config,
            pr_number=pr_number,
            expected_scope=IssuePrProvenanceScope(
                repository=config.repo,
                issue_number=issue_number,
                flow="direct",
            ),
        )
        initial_pr_url, initial_pr_head_sha = require_pr_metadata_for_handoff(initial_pr_metadata)
        pr_contract = make_pr_contract(
            repository=config.repo,
            pr_number=pr_number,
            origin_flow="issue-implementation",
            primary_issue_number=issue_number,
            expected_closing_issue_ids=closing_contract.issue_ids,
        )
        post_trusted_pr_contract_record(
            runner,
            config=config,
            pr_number=pr_number,
            body=TrustedBody.canonical(
                format_pr_contract_comment(pr_contract),
                expected_tokens=("AGENT_PR_EXPECTED_CLOSING_ISSUES",),
            ),
        )
        post_issue_pr_handoff_comment(
            runner,
            config=config,
            issue_number=issue_number,
            pr_number=pr_number,
            pr_url=initial_pr_url,
            pr_head_sha=initial_pr_head_sha,
            flow="issue-implementation",
            plan_hash=None,
            expected_closing_issue_ids=closing_contract.issue_ids,
        )
        implementation_result, _initial_derived_risk_evidence = _derive_authenticated_risk_evidence_for_coder(
            implementation_result,
            approved_plan_context=None,
            runner=runner,
            assigned_workdir=active_workdir(config),
            head_sha=initial_pr_metadata.head_sha,
            config=config,
            session_id=coder_response.session_id,
            invocation_id=coder_response.acquisition_test_turn_id,
            _closed_execution_catalog=coder_response.acquisition_test_observations,
            _journal_observations=coder_response.acquisition_test_observations,
            reauthenticate_head=lambda: get_pr_review_context(
                runner, config=config, pr_number=pr_number
            ).metadata.head_sha,
        )
        initial_local_test_evidence = runner.render_local_test_evidence(
            current_head=initial_pr_metadata.head_sha,
            legacy_tests_run=implementation_result.tests_run,
            cwd=active_workdir(config),
        )
        initial_coder_body = _attach_round_metadata(
            render_public_agent_comment(
                kind="issue_implementation",
                parsed=implementation_result,
                agent=config.coder,
                config=config,
                model_used=coder_response.model_used,
                local_test_evidence=initial_local_test_evidence,
                current_test_turn_id=coder_response.acquisition_test_turn_id,
            ),
            PostedRoundMetadata(
                flow="pr",
                role="coder",
                agent=agent_display_name(config.coder),
                round_number=1,
                subject=str(initial_pr_metadata.head_sha or "unknown"),
                prior_items=(),
                raw_structured_coder_response=coder_output,
                local_test_evidence=initial_local_test_evidence,
                risk_test_matrix_evidence=(
                    implementation_result.risk_test_matrix_evidence.to_payload()
                    if implementation_result.risk_test_matrix_evidence is not None
                    else None
                ),
                # The establishing comment renders the full row list (#959).
                risk_test_matrix_evidence_full_round=(
                    1 if implementation_result.risk_test_matrix_evidence is not None else None
                ),
                risk_test_matrix_diagnostics=tuple(
                    diagnostic.to_payload()
                    for diagnostic in implementation_result.risk_test_matrix_diagnostics
                ),
                model_used=coder_response.model_used,
                **_metadata_identity_fields(coder_response),
                acquisition_outcome=coder_response.acquisition_outcome,
                acquisition_returncode=coder_response.acquisition_returncode,
                **_test_observation_degradation_fields(implementation_result),
                **_architecture_metadata_fields(config, result=implementation_result),
            ),
        )
        post_trusted_pr_comment(
            runner,
            config=config,
            pr_number=pr_number,
            body=_embed_pr_contract_marker(initial_coder_body, pr_contract),
        )
        return run_pr_loop(
            runner,
            pr_number=pr_number,
            config=config,
            coder_session_id=coder_session_id,
            issue_context=issue_context,
            workdirs_ready=True,
            usage_context=usage_context,
            pre_review_test_pending=True,
            managed_ci_handoff=managed_ci_handoff,
        )
    finally:
        _end_run_telemetry(runner, telemetry_token)
        if owned_usage_context:
            _persist_usage_summary(config, usage_context)


def _read_clarification_from_stdin() -> str:
    print(
        "\nProvide clarification (one entry per line; finish with a single '.' line or Ctrl+D):",
        file=sys.stderr,
        flush=True,
    )
    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip() == ".":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


@claimed_run("task")
def run_task_loop(
    runner: Runner,
    *,
    task_text: str,
    config: AgentLoopConfig,
    interactive: bool = False,
    max_clarification_rounds: int = 3,
    clarification_input=None,
    usage_context: RunUsageContext | None = None,
) -> int:
    owned_usage_context = usage_context is None
    usage_context = usage_context or _new_usage_context(config)
    telemetry_token = _begin_run_telemetry(runner, config, usage_context, owned_usage_context)
    try:
        if not task_text.strip():
            raise AgentLoopError("Task text is empty; provide a non-empty description.")
        if max_clarification_rounds < 0:
            raise AgentLoopError("--max-clarification-rounds must be zero or positive.")
        config = resolve_base_branch(config, runner)
        ensure_agent_workdirs(config, runner)
        memory = prepare_agent_memory(runner, config)

        history: list[tuple[str, str]] = []
        read_clarification = clarification_input or _read_clarification_from_stdin
        coder_name = agent_display_name(config.coder)
        session_id: str | None = None

        for attempt in range(max_clarification_rounds + 1):
            if attempt == 0:
                sync_coder_base_before_implementation(config, runner)
                config = _freeze_prompt_architecture(runner, config)
                prompt = build_task_prompt(task_text, config, memory)
            assigned_head_before = _read_assigned_workdir_head(runner, config)
            log(config, f"Task attempt {attempt + 1}: invoking {coder_name}")
            coder_response = _run_validated_agent(
                runner,
                agent=config.coder,
                config=config,
                prompt=prompt,
                session_id=session_id,
                marker_description="structured task_result JSON, blocking, or clarification outcome",
                require_architecture_impact_contract=True,
                **_architecture_mode_validators(lambda mode: lambda text: _require_task_implementation_result(
                    text,
                    # This is a fresh task turn even when architecture context
                    # is disabled or unavailable; legacy decoding is resume-only.
                    required_architecture_impact_contract=1, architecture_status_mode=mode,
                )),
                usage_context=usage_context,
                role="coder",
                salvage_context=SalvageContext(
                    repo=config.repo,
                    issue_number=None,
                    scope=TASK_IMPLEMENTATION_SALVAGE_SCOPE,
                    agent=config.coder,
                    run_id=usage_context.run_id,
                ),
                operation_description="task implementation",
            )
            coder_output = coder_response.text
            session_id = coder_response.session_id

            structured_task = (
                coder_response.marker_value
                if isinstance(coder_response.marker_value, StructuredTaskResult)
                else None
            )

            if isinstance(coder_response.marker_value, _TerminalNoPrImplementation):
                raise AgentLoopError(
                    "Coder did not create a valid PR; task implementation is "
                    f"{coder_response.marker_value.state}.\n\n{coder_output}"
                )

            if isinstance(coder_response.marker_value, int) or (
                structured_task is not None and structured_task.outcome == "opened_pr"
            ):
                pr_number = (
                    structured_task.pr_number
                    if structured_task is not None
                    else coder_response.marker_value
                )
                assert isinstance(pr_number, int)
                _validate_response_tests_with_post_pr_context(
                    coder_output,
                    runner=runner,
                    config=config,
                    pr_number=pr_number,
                )
                validate_assigned_head_advanced(
                    before_head=assigned_head_before,
                    after_head=_read_assigned_workdir_head(runner, config),
                    assigned_workdir=active_workdir(config),
                )
                log(config, f"{coder_name} reported PR #{pr_number}; validating it is open")
                validate_open_pr(runner, config=config, pr_number=pr_number)
                initial_pr_metadata = get_pr_review_context(runner, config=config, pr_number=pr_number).metadata
                post_pr_comment(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    body=_attach_round_metadata(
                        normalize_freeform_signature(coder_output, agent=config.coder, config=config, model_used=coder_response.model_used),
                        PostedRoundMetadata(
                            flow="pr",
                            role="coder",
                            agent=coder_name,
                            round_number=1,
                            subject=str(initial_pr_metadata.head_sha or "unknown"),
                            prior_items=(),
                            model_used=coder_response.model_used,
                            **_metadata_identity_fields(coder_response),
                            acquisition_outcome=coder_response.acquisition_outcome,
                            acquisition_returncode=coder_response.acquisition_returncode,
                            **_architecture_metadata_fields(config, result=structured_task),
                        ),
                    ),
                )
                return run_pr_loop(
                    runner,
                    pr_number=pr_number,
                    config=config,
                    coder_session_id=session_id,
                    workdirs_ready=True,
                    usage_context=usage_context,
                    pre_review_test_pending=True,
                )

            if structured_task is not None and structured_task.outcome == "blocking":
                raise AgentLoopError(
                    "Coder returned a structured task blocking result without a PR.\n\n"
                    + coder_output
                )

            if not interactive:
                raise AgentLoopError(
                    f"{coder_name} requested clarification but the loop is non-interactive. "
                    "Add the missing details to the task text or rerun with --interactive.\n\n"
                    f"{coder_name}'s questions:\n{coder_output}"
                )

            if attempt >= max_clarification_rounds:
                raise AgentLoopError(
                    f"{coder_name} still requested clarification after "
                    f"{max_clarification_rounds} rounds; "
                    "human intervention required."
                )

            log(config, f"{coder_name} requested clarification (round {attempt + 1}); awaiting user input")
            print(coder_output, flush=True)
            answers = read_clarification()
            if not answers.strip():
                raise AgentLoopError("Empty clarification reply; aborting task.")
            history.append((coder_output, answers))
            prompt = build_task_clarification_prompt(task_text, history, config, memory)

        raise AgentLoopError("run_task_loop exited unexpectedly without producing a PR.")
    finally:
        _end_run_telemetry(runner, telemetry_token)
        if owned_usage_context:
            _persist_usage_summary(config, usage_context)
