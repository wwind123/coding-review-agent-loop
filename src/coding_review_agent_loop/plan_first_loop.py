"""The plan-first loop together with staged child dispatch.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1202); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace as dataclasses_replace
from typing import Literal
from .agents.base import AgentName
from .agents.registry import agent_display_name
from .config import (
    AgentLoopConfig,
    reviewers,
)
from .board_amendment import (
    ContractLineage,
    amendment_audit_already_posted,
    amendment_summary_line,
    apply_board_amendment_to_ledger,
    collect_reviewer_board_amendments,
    render_amendment_audit_comment,
    require_amendment_activation,
    resolve_contract_lineage,
)
from .decomposition import (
    CreatedPhaseIssue,
    approved_plan_hash,
    find_existing_one_shot_impl_handoff,
    find_latest_one_shot_impl_handoff,
    find_existing_phase_implementation_handoff,
    post_phase_implementation_handoff_comment,
    normalize_execution_recommendation,
    validate_separately_planned_child_matrix,
    InheritedMatrixBinding,
    InheritedRowDifference,
    inherited_matrix_reviewed_deltas,
    risk_matrix_row_ids_for_owner,
    EXECUTION_TOPOLOGY_SOURCE,
    ChildDispositionOverride,
    PhaseImplementationHandoffMetadata,
    collect_child_disposition_overrides,
)
from .protocol import EXECUTION_DISPOSITION_DIRECT
from .child_topology import (
    NeedsHumanDecision,
    NestedTopologyDecision,
)
from .errors import (
    AgentInvocationError,
    AgentLoopError,
    DeterministicPlanValidationExhaustion,
    QuotaResetExceededError,
)
from .github import (
    IssueContext,
    get_issue_context,
    get_pr_review_context,
    get_pr_state,
    post_issue_comment,
    read_rest_issue_comments,
    post_frozen_round_bodies,
    round_publication,
    validate_pr_body_does_not_close_issue,
)
from .publication_resume import context_digest as publication_context_digest
from .issue_pr_handoff import (
    post_issue_pr_handoff_comment,
    require_pr_metadata_for_handoff,
    resolve_canonical_pr_for_issue,
)
from .issue_pr_provenance import IssuePrProvenanceScope
from .phase_progress import (
    STATUS_HUMAN_PENDING,
    StagedTopologyOutcome,
    record_staged_completion,
    resolve_staged_phase_progress,
    select_current_phase,
)
from .split_materialization import (
    find_existing_split_materialization,
    find_existing_split_stage_handoff,
    post_split_stage_handoff_comment,
    resolve_selected_stage_child,
)
from .logging import log
from .finding_history import (
    PHASE_PLAN as _HISTORY_PHASE_PLAN,
    FindingHistoryLedger,
    log_declared_generalization,
)
from .prompts import (
    CompactPlanTailContext,
    CompactPriorContext,
    INHERITED_COVERAGE_DELTA_MAX_BYTES,
    INHERITED_OBLIGATIONS_ENFORCEABLE_MAX_BYTES,
    build_issue_plan_prompt,
    inherited_coverage_delta_size,
    inherited_obligations_enforceable_size,
    build_plan_review_prompt,
    build_plan_revision_prompt,
    format_agent_list,
    render_coder_human_requirements_prompt_context,
)
from .protocol import (
    ParsedPlanReview,
    ExecutionStrategyRecommendation,
    ParsedReview,
    ReviewItemDisposition,
    StructuredPlanState,
    StructuredPlanRevision,
    PlanRevisionPatch,
    UnresolvedReviewItem,
    human_requirements_resolved,
    is_clarification_request,
    parse_plan_review_items,
    parse_plan_state,
    parse_structured_plan_review,
    review_freeform_summary_text,
    validate_structured_plan_state,
    validate_risk_test_matrix_revision,
    risk_test_matrix_identity,
)
from .runner import Runner
from .usage import RunUsageContext
from .comment_rendering import (
    render_plan_phase_advance,
    render_plan_scheduling_audit,
    same_model_panel_note,
    RISK_TEST_MATRIX_MARKER_RE,
    decode_risk_test_matrix_marker,
    normalize_freeform_signature,
    render_public_agent_comment,
    render_canonical_plan_revision,
    render_canonical_plan_state,
)
from .followups import (
    _plan_followup_source_from_unresolved_item,
    _publish_plan_approved_followups,
    FollowupSourceContext,
)
from .config import phased_delivery_guard_active
from .round_state import (
    prior_plan_execution_mode,
    ApprovedPlanContext,
    PostedRoundMetadata,
    PostedRoundRecord,
    ResumedReviewRound,
    _attach_round_metadata,
    _extract_round_metadata_records,
    _plan_subject,
    _prior_item_ledger_signature,
    _resume_plan_round,
    make_approved_plan_context,
    scope_approved_plan_matrix,
    PlanValidationDiagnosticTransport,
    sanitize_plan_validation_diagnostic,
)
from .plan_assembly import (
    AssembledPlanSidecar,
    AuthenticatedPlanState,
    assemble_authenticated_plan_revision,
    decode_assembled_plan_sidecar,
    hydrate_authenticated_plan_state,
    make_assembled_plan_sidecar,
)
from .plan_growth import (
    PlanGrowthThresholds,
    assess_plan_growth,
    check_growth_justification,
    check_scope_ledger_preservation,
    plan_growth_approval_verdict,
    plan_growth_gate_enforced,
    plan_strategy,
    render_growth_measurements,
    render_growth_notice,
)
from .plan_review_scheduling import (
    PLAN_HISTORY_CONTRADICTORY_KEY,
    PLAN_HISTORY_INTACT,
    PlanCandidateKey,
    PlanCrossCuttingContracts,
    PlanPrePanelSafetyError,
    PlanReviewSchedulingContract,
    PlanSchedulerSnapshot,
    classify_plan_transition,
    make_plan_contract,
    plan_history_fallback_reason,
    plan_policy_capabilities,
    plan_undecodable_history_message,
    select_plan_reviewers,
)
from .partial_round_recovery import (
    PartialRoundRecovery,
    RecoveryVisibilityContext,
    compute_partial_round_recovery,
)
from .round_visibility import latest_round_checkpoint_index, visible_peer_names
from .unresolved_items import (
    _apply_unresolved_item_dispositions,
    _collect_prior_compact_summaries,
    bound_compact_prior_summaries,
    _next_unresolved_item,
    _record_prior_item_disposition,
    _validate_plan_review_response,
)
from .agent_failure import (
    ValidatedAgentResponse,
    _metadata_identity_fields,
)
from .architecture_contract import (
    _architecture_metadata_fields,
    _architecture_mode_validators,
    _resumed_review_architecture,
    _acknowledgement_repair_forbids_assessment,
    _pin_acknowledgement_repair,
)
from .validated_agent import (
    _run_structured_repair,
    _log_repair_attempts,
    _run_validated_agent,
)
from .response_validation import (
    _require_plan_state_or_clarification,
    _validate_response_with_human_requirements,
    _current_plan_has_complete_human_requirement_dispositions,
    _describe_requirement_set_change,
    _surfaced_reviewer_requirement_ids,
    _validate_plan_revision_response,
    _validate_plan_revision_patch_response,
    _validate_plan_revision_patch_payload,
    _drop_repeated_carried_plan_future_followups,
)
from .panel_evidence import (
    _superseded_prepanel_plan_review,
    _scheduler_recorded_force_full,
    PlanPanelEvidence,
    _plan_candidate_key_for,
    _outstanding_plan_phase,
    _board_amendment_template,
    _board_amendment_route_clause,
    _stale_amendment_repost_clause,
    _append_board_amendment_note,
    _keep_reused_amendment_round_reviews,
    _plan_contract_or_none,
    _derive_plan_panel_evidence,
    _carried_plan_approvals,
    _classify_staged_plan_history,
    plan_issue_text_digest,
    plan_primary_blocking_streak_detail,
    plan_primary_stall_message,
    _planner_candidate_rounds,
    _plan_growth_candidate_count,
    _plan_growth_gate_violation,
    _plan_cross_cutting_contracts,
    _resumed_plan_transition_inputs,
    _plan_revision_descriptor,
)
from .review_step_back import (
    BLOCKING_CLASSES,
    DISPOSITION_DEFER_EPISODE,
    DISPOSITION_DEFER_PENDING,
    DISPOSITION_ESCALATED,
    PlanStepBackAnchor,
    PlanStepBackContext,
    StallStepBackDisposition,
    derive_plan_step_back_state,
    entry_payload_for_plan,
    mandatory_plan_findings_since,
    plan_growth_crossing_round,
    plan_stall_step_back_disposition,
    plan_step_back_anchor,
    plan_step_back_candidate_rounds,
    reintroduced_dissolved_items,
    render_step_back_human_decision,
    step_back_alternative_summary,
)
from .review_rounds import (
    _is_incomplete_plan_review,
    _describe_plan_review_outcome,
    _round_ledger_may_be_incomplete,
    _round_resolved_history_item_ids,
    _post_round_resolved_history_item_ids,
    _ReviewerTurnResult,
    _review_round_spool,
    _replay_spooled_review,
    _preflight_spooled_publications,
    _incomplete_plan_review_error,
    _refuse_partial_round_before_sequential_turns,
    _same_round_replay_or_invoke,
    _launch_reviewer_turns,
    _ensure_parallel_reviewer_workdirs,
)
from .execution_policy import (
    _extract_current_deferred_stages,
    _extract_current_expected_closing_issue_ids,
    _extract_current_child_stages,
    _plan_first_line,
    _current_execution_recommendation,
    _resolve_execution_policy,
    staged_plan_mode_conflict_message,
    ChildExecutionRoute,
    resolve_child_execution_route,
    _fresh_phase_marker_payload,
    _child_resume_hint,
    _post_child_planning_handoff,
    _print_staged_phase_progress,
    _print_staged_terminal_report,
    _print_dry_run_execution_preview,
    _print_execution_resolution_summary,
    _persist_execution_decision_if_needed,
    _retirable_execution_decision_hashes,
    _preflight_fresh_staged_topology,
    _preflight_fresh_one_shot_recovery,
    _handle_plan_first_split_scope,
    _infer_staged_parent_issue,
)
from .child_plan_binding import (
    MAX_INHERITED_MATRIX_REPLANS,
    _InheritedReplanDiagnostic,
    _inherited_matrix_binding,
    _PlanSupersessionBinding,
    _require_authorized_replan_state,
    _rebind_superseded_child_plan,
    _inadmissible_plan_audit_line,
    _resumed_inherited_replan_force_full,
    _recover_current_plan_validation_diagnostic,
    _persist_exhausted_plan_validation_diagnostic,
    _assemble_structured_plan_round_body,
    _post_plan_coder_round_comment,
)
from .pr_loop_support import (
    _scheduler_obligations,
    _visibility_snapshot,
)
from .pr_loop import run_pr_loop
from .issue_implementation import (
    _implement_approved_issue,
    _decompose_approved_plan,
)


def _run_child_planning_cycle(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    memory,
    usage_context: RunUsageContext,
    parent_issue: int,
    child_issue_number: int,
    inherited_matrix_binding: InheritedMatrixBinding | None = None,
) -> int:
    """Run the child's own plan/review cycle (policy ``auto``) after its handoff."""
    # The child inherits the operator's run-wide plan-review policy and primary
    # plan reviewer (#929), but no parent scheduling state crosses the
    # boundary: the child plan is a fresh artifact, so the force-full latch is
    # reset and the execution mode is ``auto`` (#905, from #841).
    child_config = dataclasses_replace(
        config,
        plan_execution_mode="auto",
        plan_review_force_full=False,
        plan_reset_stall_streak=False,
    )
    if config.plan_reset_stall_streak:
        log(
            config,
            f"Issue #{parent_issue}: child #{child_issue_number} plan review does not "
            "inherit --plan-reset-stall-streak",
        )
    if config.plan_review_force_full:
        log(
            config,
            f"Issue #{parent_issue}: child #{child_issue_number} plan review does not "
            "inherit --plan-review-force-full; the child plan starts under "
            f"--plan-review-policy {config.plan_review_policy} without the "
            "parent's full-board override",
        )
    child_issue_context = get_issue_context(
        runner, config=child_config, issue_number=child_issue_number
    )
    log(
        config,
        f"Issue #{parent_issue}: child #{child_issue_number} requires its own reviewed plan; "
        "starting the child plan-first cycle with policy auto before any implementation coder",
    )
    return _run_plan_first_loop(
        runner,
        issue_number=child_issue_number,
        config=child_config,
        memory=memory,
        issue_context=child_issue_context,
        requested_policy="auto",
        usage_context=usage_context,
        inherited_matrix_binding=inherited_matrix_binding,
    )


def _dispatch_decomposition_child(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    memory,
    usage_context: RunUsageContext,
    parent_issue: int,
    approved_plan: str,
    plan_hash: str,
    plan_subject: str,
    recommendation: ExecutionStrategyRecommendation | None,
    approved_plan_context: ApprovedPlanContext,
    created: CreatedPhaseIssue,
    phase_index: int,
    route: ChildExecutionRoute,
    child_issue_context: IssueContext,
    parent_issue_context: IssueContext,
    coder_session_id: str | None,
    existing_handoff: PhaseImplementationHandoffMetadata | None,
) -> int:
    """Dispatch one materialized child along its resolved route.

    Shared by the parent implement-by-phase path and the direct child-issue
    entry.  The handoff is persisted before any coder or planning work; when
    a matching handoff already exists it is reused, never re-posted.
    """
    if created.issue_number is None:
        raise AgentLoopError(
            "Cannot implement decomposed phase because its child issue number "
            "was not available from GitHub CLI output."
        )
    if route.is_human:
        raise AgentLoopError(
            f"Issue #{created.issue_number} is a {created.phase.automation} stage; no agent "
            "implementation or planning is dispatched for human-owned stages."
        )
    stage_id = (
        getattr(created.phase, "stage_id", None)
        or str(getattr(created.phase, "position", None) or phase_index)
    )
    inherited_matrix_row_ids = risk_matrix_row_ids_for_owner(
        approved_plan_context.risk_test_matrix_payload
        if approved_plan_context.matrix_available else None,
        stage_id,
    )
    if route.is_planning:
        if recommendation is None:
            raise AgentLoopError(
                "Child planning requires a fresh approved-plan-v1 topology; legacy topologies "
                "cannot route a child to planning."
            )
        if existing_handoff is None:
            # Persistence before any planning agent runs.
            _post_child_planning_handoff(
                runner,
                config=config,
                parent_issue=parent_issue,
                plan_hash=plan_hash,
                plan_subject=plan_subject,
                phase_index=phase_index,
                created=created,
                recommendation=recommendation,
                inherited_matrix_row_ids=inherited_matrix_row_ids,
                override_digest=route.override_digest,
            )
        return _run_child_planning_cycle(
            runner,
            config=config,
            memory=memory,
            usage_context=usage_context,
            parent_issue=parent_issue,
            child_issue_number=created.issue_number,
            inherited_matrix_binding=_inherited_matrix_binding(
                parent_issue=parent_issue,
                stage_id=stage_id,
                parent_plan_context=approved_plan_context,
            ),
        )
    if existing_handoff is None:
        # Persist the parent-owned assignment before child execution so
        # PR validation and crash recovery see the same phase identity.
        post_phase_implementation_handoff_comment(
            runner,
            config=config,
            parent_issue=parent_issue,
            mode="implement-by-phase",
            plan_hash=plan_hash,
            phase_index=phase_index,
            created=created,
            strategy=recommendation.strategy if recommendation is not None else None,
            topology_source=EXECUTION_TOPOLOGY_SOURCE if recommendation is not None else None,
            execution_strategy_contract_version=1 if recommendation is not None else None,
            recommendation_digest=(
                str(recommendation.identity()["recommendation_sha256"])
                if recommendation is not None else None
            ),
            plan_subject=plan_subject,
            inherited_matrix_row_ids=inherited_matrix_row_ids,
            execution_disposition=(
                EXECUTION_DISPOSITION_DIRECT if recommendation is not None else None
            ),
            override_digest=route.override_digest if recommendation is not None else None,
        )
    child_plan_context = make_approved_plan_context(
        approved_plan,
        source_locator=f"issue #{parent_issue} topology checkpoint phase {phase_index}",
        expected_hash=plan_hash,
        expected_subject=plan_subject,
    )
    child_plan_context = scope_approved_plan_matrix(
        child_plan_context,
        execution_owner=stage_id,
        valid_stage_ids=tuple(
            phase.stage_id or str(phase.position)
            for phase in recommendation.child_stages
        ) if recommendation is not None else (),
    )
    return _implement_approved_issue(
        runner,
        issue_number=created.issue_number,
        approved_plan=created.phase.parent_context or approved_plan,
        config=config,
        memory=memory,
        issue_context=child_issue_context,
        approved_plan_context=child_plan_context,
        parent_issue_context=parent_issue_context,
        coder_session_id=coder_session_id,
        usage_context=usage_context,
        execution_recommendation=recommendation,
    )


def _dispatch_current_decomposition_phase(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    memory,
    usage_context: RunUsageContext,
    issue_number: int,
    current_plan: str,
    plan_subject: str,
    outcome: StagedTopologyOutcome,
    recommendation: ExecutionStrategyRecommendation | None,
    approved_plan_context: ApprovedPlanContext,
    issue_context: IssueContext,
    mode: str,
    coder_session_id: str | None,
) -> int:
    """Route and dispatch the *current* phase of an implement-by-phase topology.

    The current phase is the first phase that is not complete, authenticated
    from live child state (#918).  A completed stage-1 therefore advances to
    stage-2 with its real ``phase_index``, and a fully delivered topology
    reports a terminal state instead of pointing the operator at a finished
    child.
    """
    created = outcome.created
    if not created:
        raise AgentLoopError("Plan decomposition produced no phases.")
    plan_hash = outcome.plan_hash
    fresh = recommendation is not None
    topology_stage_ids = outcome.stage_ids

    if config.dry_run:
        # Dry run resolves no child state and issues no GitHub reads: it
        # reports the first phase's declared disposition, exactly as today.
        first_phase = created[0]
        first_stage_id = topology_stage_ids[0]
        if first_phase.phase.automation != "agent-pr":
            print(
                f"Issue #{issue_number} approved plan decomposed; first phase requires human work "
                f"({first_phase.phase.automation}), so implementation is stopping."
            )
            return 0
        declared = getattr(first_phase.phase, "execution_disposition", None) or "legacy-ambiguous"
        print(
            f"Issue #{issue_number} dry-run decomposed the approved plan; "
            "phase implementation is not started. "
            f"First phase `{first_stage_id}` declared disposition: {declared}."
        )
        return 0

    parent_issue_context = get_issue_context(
        runner, config=config, issue_number=issue_number
    )
    progress = resolve_staged_phase_progress(
        runner,
        config=config,
        parent_issue=issue_number,
        parent_comments=parent_issue_context.comments,
        outcome=outcome,
    )
    _print_staged_phase_progress(issue_number, progress)
    selected = select_current_phase(progress)
    if selected is None:
        _print_staged_terminal_report(
            issue_number=issue_number, progress=progress, outcome=outcome
        )
        try:
            recorded = record_staged_completion(
                runner,
                config=config,
                parent_issue=issue_number,
                progress=progress,
                outcome=outcome,
            )
        except AgentLoopError as exc:
            # The delivery report above stands; only the write-back is
            # withheld, and nothing incomplete was published.
            log(config, f"Issue #{issue_number} completion record not written: {exc}")
            print(
                f"No completion record was written to issue #{issue_number} ({exc}). "
                "Rerun the parent once that is resolved to record it."
            )
            return 0
        if recorded:
            print(f"Recorded the staged completion on issue #{issue_number}.")
        return 0

    phase_index = selected.phase_index
    stage_id = selected.stage_id
    selected_created = selected.created
    child_issue_number = selected.child_issue_number
    if child_issue_number is None:
        raise AgentLoopError(
            "Cannot implement decomposed phase because its child issue number "
            "was not available from GitHub CLI output."
        )

    if selected.status == STATUS_HUMAN_PENDING:
        if fresh:
            # Stage-scoped validation: an override naming the human stage is a
            # human-decision error; a valid override for another stage is
            # neither an error nor an input here.
            child_comments = get_issue_context(
                runner, config=config, issue_number=child_issue_number
            ).comments
            overrides = collect_child_disposition_overrides(
                parent_comments=issue_context.comments,
                child_comments=child_comments,
                parent_issue=issue_number,
                plan_hash=plan_hash,
                topology_stage_ids=topology_stage_ids,
                routed_stage_id=stage_id,
                child_stage_id=stage_id,
                child_issue_number=child_issue_number,
            )
            resolve_child_execution_route(
                selected_created.phase,
                topology_source=EXECUTION_TOPOLOGY_SOURCE,
                recorded_handoff=None,
                overrides=overrides,
            )
        print(
            f"Issue #{issue_number} approved plan decomposed; phase {phase_index} "
            f"(`{stage_id}`) requires human work ({selected.automation}) on child issue "
            f"#{child_issue_number}, so implementation is stopping."
        )
        return 0

    handoff = find_existing_phase_implementation_handoff(
        parent_issue_context.comments,
        parent_issue=issue_number,
        plan_hash=plan_hash,
        mode=mode,
        phase_index=phase_index,
        child_issue_number=child_issue_number,
    )
    if not fresh:
        if handoff is not None:
            # Only an in-progress phase reaches here: a completed child is
            # never advertised as a resume target (#918).
            print(
                f"Issue #{issue_number} approved plan already handed off to child issue "
                f"#{handoff.child_issue_number}; resume directly with "
                f"`agent-loop issue {handoff.child_issue_number}`."
            )
            return 0
        route = ChildExecutionRoute(EXECUTION_DISPOSITION_DIRECT, None, "legacy-topology")
        overrides: tuple[ChildDispositionOverride, ...] = ()
    else:
        child_issue_context = get_issue_context(
            runner, config=config, issue_number=child_issue_number
        )
        overrides = collect_child_disposition_overrides(
            parent_comments=parent_issue_context.comments,
            child_comments=child_issue_context.comments,
            parent_issue=issue_number,
            plan_hash=plan_hash,
            topology_stage_ids=topology_stage_ids,
            routed_stage_id=stage_id,
            child_stage_id=stage_id,
            child_issue_number=child_issue_number,
        )
        route = resolve_child_execution_route(
            selected_created.phase,
            topology_source=EXECUTION_TOPOLOGY_SOURCE,
            recorded_handoff=handoff,
            overrides=overrides,
        )
        if handoff is not None:
            # Reconciliation ran inside the seam; the hint follows the
            # recorded route so it never guides into the conflicting form.
            print(
                f"Issue #{issue_number} approved plan already handed off to child issue "
                f"#{handoff.child_issue_number} with disposition `{route.disposition}`; "
                f"resume directly with `{_child_resume_hint(handoff.child_issue_number, route.disposition)}`."
            )
            return 0
    if not fresh:
        child_issue_context = get_issue_context(
            runner, config=config, issue_number=child_issue_number
        )
    return _dispatch_decomposition_child(
        runner,
        config=config,
        memory=memory,
        usage_context=usage_context,
        parent_issue=issue_number,
        approved_plan=current_plan,
        plan_hash=plan_hash,
        plan_subject=plan_subject,
        recommendation=recommendation,
        approved_plan_context=approved_plan_context,
        created=selected_created,
        phase_index=phase_index,
        route=route,
        child_issue_context=child_issue_context,
        parent_issue_context=parent_issue_context,
        coder_session_id=coder_session_id,
        existing_handoff=handoff,
    )


def _typed_deferred_work_titles(sidecar: object | None) -> tuple[str, ...]:
    """Titles of the typed ``deferred_work`` in the canonical plan, best effort (#1268)."""
    canonical = getattr(sidecar, "canonical_json", None)
    if not isinstance(canonical, dict):
        return ()
    typed = canonical.get("typed_stages")
    for container in (canonical, typed if isinstance(typed, dict) else {}):
        items = container.get("deferred_work")
        if isinstance(items, list):
            titles = tuple(
                str(item["title"]) for item in items
                if isinstance(item, dict) and isinstance(item.get("title"), str)
            )
            if titles:
                return titles
    return ()


def _run_plan_first_loop(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    memory,
    issue_context: IssueContext,
    requested_policy: str | None = None,
    implement_after_approval: bool = False,
    usage_context: RunUsageContext,
    inherited_matrix_binding: InheritedMatrixBinding | None = None,
    plan_supersession: _PlanSupersessionBinding | None = None,
) -> int:
    if config.review_parallel:
        _ensure_parallel_reviewer_workdirs(config, flag_name="--review-parallel", role_label="reviewer")
    staged_parent_number = _infer_staged_parent_issue(issue_context)
    parent_issue_context = (
        get_issue_context(runner, config=config, issue_number=staged_parent_number)
        if staged_parent_number is not None
        else None
    )
    coder_name = agent_display_name(config.coder)
    configured_reviewers = reviewers(config)

    # --- Staged planning scheduling (#905, from #841) -------------------
    plan_reviewer_names = tuple(agent_display_name(name) for name in configured_reviewers)
    plan_capabilities = plan_policy_capabilities(config.plan_review_policy)
    staged_planning = plan_capabilities.scheduler_enabled
    plan_scheduler_contract = (
        make_plan_contract(
            plan_reviewer_names,
            config.plan_review_policy,
            (
                agent_display_name(config.primary_plan_reviewer)
                if config.primary_plan_reviewer is not None
                else None
            ),
        )
        if staged_planning
        else None
    )
    plan_primary_name = (
        plan_scheduler_contract.primary_reviewer if plan_scheduler_contract is not None else None
    )
    plan_operator_force_full = bool(config.plan_review_force_full)
    plan_issue_digest = plan_issue_text_digest(issue_context)
    plan_reset_checkpoint_written = False
    plan_automatic_force_full = False
    plan_scheduler_calls_avoided = 0
    plan_phase_advance_pending = False

    def stop_plan_pre_panel(message: str, *, round_number: int | None) -> None:
        """Log and post the planning diagnostic, then stop before any turn."""
        label = f"round {round_number}" if round_number is not None else "startup"
        log(config, f"Planning {label}: {message}")
        # Plain audit text only: no round metadata that a later resume could
        # mistake for a planning scheduler checkpoint or a panel opening.
        post_issue_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=f"Plan review scheduling diagnostic ({label}): {message}",
        )
        raise PlanPrePanelSafetyError(message)

    def plan_step_back_measurements(assessment: object | None, text: str | None) -> str:
        if text:
            return text
        describe = getattr(assessment, "describe", None)
        return f"Plan-growth measurements: {describe()}." if callable(describe) else ""

    def plan_step_back_state(
        records: Sequence[PostedRoundRecord], *, panel_opening_index: int | None
    ):
        """The primary's step-back state, or ``None`` when degraded or disabled (#1251)."""
        if config.plan_step_back_rounds <= 0 or plan_primary_name is None:
            return None
        state = derive_plan_step_back_state(
            records,
            primary=plan_primary_name,
            panel_opening_index=panel_opening_index,
            current_issue_digest=plan_issue_digest,
        )
        if state.degraded:
            log(
                config,
                "Planning step-back suppressed: a malformed step-back history entry "
                "degrades the planning history",
            )
            return None
        return state

    def plan_step_back_escalation(
        records: Sequence[PostedRoundRecord],
        *,
        panel_opening_index: int | None,
        assessment: object | None,
        measurements: str | None,
        post_review: bool = False,
    ) -> str | None:
        """Human-decision message once the primary rejected the step-back, else ``None``."""
        state = plan_step_back_state(records, panel_opening_index=panel_opening_index)
        if (
            state is None
            or state.episode is None
            or state.escalation_count < config.plan_step_back_escalation_rounds
        ):
            return None
        anchor = plan_step_back_anchor(records, state.episode)
        return render_step_back_human_decision(
            phase="plan",
            measurements=plan_step_back_measurements(assessment, measurements),
            step_back_round=state.episode.candidate_round,
            blocks=state.escalation_count,
            threshold=config.plan_step_back_escalation_rounds,
            alternative=step_back_alternative_summary(records, state.episode),
            dissolved=anchor.dissolved_ids,
            reintroduced=reintroduced_dissolved_items(records, state.episode, anchor),
            post_review=post_review,
        )

    def _stall_step_back_disposition() -> StallStepBackDisposition:
        """What the reached stall limit does about the plan step-back (#1275)."""
        crossing = (
            plan_growth_crossing_round(
                plan_records,
                config=config,
                current_round=current_plan_sidecar.round_number,
            )
            if current_plan_sidecar is not None
            else None
        )
        return plan_stall_step_back_disposition(
            plan_step_back_state(
                plan_records, panel_opening_index=plan_panel_evidence.opening_index
            ),
            round_number=round_number,
            crossing_round=crossing,
            growth_gate_enforced=plan_growth_gate_enforced(config),
            step_back_rounds=config.plan_step_back_rounds,
            escalation_rounds=config.plan_step_back_escalation_rounds,
        )

    if inherited_matrix_binding is not None:
        inherited_measured = inherited_obligations_enforceable_size(inherited_matrix_binding)
        if inherited_measured > INHERITED_OBLIGATIONS_ENFORCEABLE_MAX_BYTES:
            # Fail closed before any planner or reviewer turn: the block is
            # never truncated, and the retention check is never relaxed.
            stop_plan_pre_panel(
                f"inherited parent matrix obligations for stage "
                f"`{inherited_matrix_binding.stage_id}` measure {inherited_measured} bytes, above "
                f"the permitted {INHERITED_OBLIGATIONS_ENFORCEABLE_MAX_BYTES} bytes for a "
                "lossless planning prompt; reduce or split the rows the approved plan on "
                f"parent issue #{inherited_matrix_binding.parent_issue} allocates to this stage, "
                "then rerun child planning.",
                round_number=None,
            )

    def check_inherited_candidate(
        child_matrix: object,
    ) -> tuple[InheritedRowDifference, ...]:
        """Mechanical inherited-row check plus the fail-closed delta cap."""
        assert inherited_matrix_binding is not None
        validate_separately_planned_child_matrix(
            inherited_matrix_binding.parent_matrix,
            child_matrix,  # type: ignore[arg-type]
            execution_owner=inherited_matrix_binding.stage_id,
        )
        deltas = inherited_matrix_reviewed_deltas(
            inherited_matrix_binding.parent_matrix,
            child_matrix,  # type: ignore[arg-type]
            execution_owner=inherited_matrix_binding.stage_id,
        )
        delta_measured = inherited_coverage_delta_size(deltas)
        if delta_measured > INHERITED_COVERAGE_DELTA_MAX_BYTES:
            raise AgentLoopError(
                f"Separately planned child departs from its inherited parent matrix rows by "
                f"{delta_measured} bytes of reviewed deltas, above the permitted "
                f"{INHERITED_COVERAGE_DELTA_MAX_BYTES} bytes that reviewers can be shown without "
                "truncation. Make fewer or smaller departures from the inherited text: keep "
                "inherited rows closer to the parent values and move new coverage into "
                "child-local rows."
            )
        return deltas

    def inherited_review_context(
        plan_text: str,
    ) -> tuple[tuple[InheritedRowDifference, ...], str | None]:
        """Reviewed deltas recomputed from the canonical plan; never persisted."""
        if inherited_matrix_binding is None:
            return (), None
        matrix_match = RISK_TEST_MATRIX_MARKER_RE.search(plan_text)
        try:
            child_matrix = (
                decode_risk_test_matrix_marker(matrix_match.group("payload"))["matrix"]
                if matrix_match is not None
                else None
            )
            return check_inherited_candidate(child_matrix), None
        except AgentLoopError as exc:
            # Only a historical plan published before this check can reach a
            # reviewer in this state; reviewers are told to block it so the
            # revision turn re-enters the enforced replan path.
            return (), sanitize_plan_validation_diagnostic(str(exc))

    def run_inherited_checked_planner_turn(
        invoke: Callable[[object | None], ValidatedAgentResponse],
        *,
        derive_matrix: Callable[[ValidatedAgentResponse], object],
        initial_diagnostic: object | None,
        candidate_kind: Literal["plan_state", "plan_revision"],
        target_coder_round: int,
        prior_plan_subject: str | None,
    ) -> ValidatedAgentResponse:
        """Orchestrator-owned bounded replan over unpublished candidates.

        Deliberately outside ``_run_validated_agent``: the envelope-only repair
        model never sees an inherited-row rejection.  A rejected candidate
        stays in local memory only, so the authenticated base is untouched.
        """
        diagnostic = initial_diagnostic
        for replan_attempt in range(MAX_INHERITED_MATRIX_REPLANS + 1):
            try:
                response = invoke(diagnostic)
            except AgentInvocationError as exc:
                # A semantic-patch payload rejection is unsatisfiable by the
                # envelope-only repair model; it loses this candidate, not
                # the run (#979).
                if exc.bounded_replan_rejection is None:
                    raise
                candidate_text = exc.bounded_replan_rejection.candidate_text
                rejection = sanitize_plan_validation_diagnostic(
                    exc.bounded_replan_rejection.diagnostic
                )
                failure_description = "fail semantic patch payload validation"
            else:
                if is_clarification_request(response.text):
                    return response
                # Candidate assembly runs on every plan-first run, not only
                # child cycles: a deterministic assembly failure (for example a
                # row-bound overflow) becomes a bounded-replan diagnostic, not
                # a crash.
                try:
                    child_matrix = derive_matrix(response)
                    if inherited_matrix_binding is not None:
                        check_inherited_candidate(child_matrix)
                    return response
                except AgentLoopError as exc:
                    rejection = sanitize_plan_validation_diagnostic(str(exc))
                candidate_text = response.text
                failure_description = (
                    "weaken inherited parent matrix rows"
                    if inherited_matrix_binding is not None
                    else "fail deterministic plan assembly"
                )
            candidate_digest = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()
            if replan_attempt >= MAX_INHERITED_MATRIX_REPLANS:
                exhaustion = DeterministicPlanValidationExhaustion(
                    candidate_kind=candidate_kind,
                    candidate_text=candidate_text,
                    diagnostic=rejection,
                    candidate_digest=candidate_digest,
                )
                error = AgentInvocationError(
                    f"{coder_name} produced {MAX_INHERITED_MATRIX_REPLANS + 1} consecutive plan "
                    f"candidates that {failure_description}; stopping before any "
                    f"reviewer or implementation turn.\n{rejection}",
                    failure_category="deterministic",
                    plan_validation_exhaustion=exhaustion,
                )
                _persist_exhausted_plan_validation_diagnostic(
                    runner,
                    config=config,
                    issue_context=issue_context,
                    issue_number=issue_number,
                    original_error=error,
                    exhaustion=exhaustion,
                    target_coder_round=target_coder_round,
                    prior_plan_subject=prior_plan_subject,
                    candidate_kind=candidate_kind,
                    require_execution_strategy_contract=require_fresh_execution_contract,
                    require_risk_test_matrix_contract=require_fresh_matrix_contract,
                )
                raise error
            log(
                config,
                f"Planning issue #{issue_number}: unpublished candidate failed deterministic "
                f"plan validation; bounded replan {replan_attempt + 1} of "
                f"{MAX_INHERITED_MATRIX_REPLANS}",
            )
            diagnostic = _InheritedReplanDiagnostic(
                diagnostic=rejection,
                failure_attempt=replan_attempt + 1,
                candidate_digest=candidate_digest,
            )
        raise AssertionError("unreachable inherited replan state")

    def plan_history_records(
        *, refresh: bool = False, round_number: int | None = None
    ) -> tuple[PostedRoundRecord, ...]:
        """Planning round-metadata records, or the class-D stop.

        Checked at startup and again at every round boundary: a record set that
        cannot be extracted at all makes panel state, finding ownership, and
        exact-plan approvals unknowable, and the operator override cannot
        authorize a board over it.  Round boundaries re-read the durable
        history so records this run posted are part of it.
        """
        comments = issue_context.comments
        if refresh:
            comments = get_issue_context(
                runner, config=config, issue_number=issue_number
            ).comments
        try:
            return _extract_round_metadata_records(comments, flow="plan")
        except AgentLoopError as exc:
            stop_plan_pre_panel(
                plan_undecodable_history_message(exc), round_number=round_number
            )
            raise

    def planning_contract_drift_records() -> tuple[PostedRoundRecord, ...]:
        """Planning records inspected for an in-flight contract change.

        Drift detection runs under *both* policies: restarting a run that
        already persisted a staged planning contract with the compatibility
        default would otherwise silently continue on a different contract.  The
        default path keeps its historical behavior for an unreadable record set,
        which ``_resume_plan_round`` already reports; only staged planning turns
        that into the class-D diagnostic stop.
        """
        if staged_planning:
            return plan_history_records()
        try:
            return _extract_round_metadata_records(issue_context.comments, flow="plan")
        except AgentLoopError:
            return ()

    def plan_contract_drift_error(
        persisted_contract: PlanReviewSchedulingContract, detail: str
    ) -> AgentLoopError:
        def template_round() -> int | None:
            resumed = _resume_plan_round(
                issue_context.comments, configured_reviewers=configured_reviewers
            )
            return resumed[1].round_number if resumed is not None else 1

        return AgentLoopError(
            "Plan review scheduler contract changed during resume; the required "
            "reviewer board, the planning policy, and the primary plan reviewer "
            "must remain immutable for the run. This run is configured with "
            f"--plan-review-policy {config.plan_review_policy} and primary "
            f"{plan_primary_name or '(none)'}, but issue #{issue_number} already "
            "carries a planning scheduler contract for policy "
            f"{persisted_contract.policy} with primary "
            f"{persisted_contract.primary_reviewer or '(none)'} and reviewer board "
            f"{', '.join(persisted_contract.required_reviewers)} ({detail}). Rerun with the "
            "persisted planning policy, primary, and reviewer board."
            + _stale_amendment_repost_clause(detail, template_round)
            + _board_amendment_route_clause(
                flow="plan",
                issue_number=issue_number,
                pr_number=None,
                persisted=persisted_contract,
                configured=plan_scheduler_contract,
                start_round_number=template_round,
                amendments_recognized=bool(plan_board_amendments),
            )
        )

    # Signed reviewer-board amendments (#943) are read from the same issue
    # comment list the plan round records live in, so comment order is
    # always comparable.  Without an amendment this is exactly the historical
    # immutability rule.
    plan_amendment_diagnostics: list[str] = []
    plan_board_amendments = collect_reviewer_board_amendments(
        issue_context.comments,
        flow="plan",
        issue_number=issue_number,
        ignored_sink=plan_amendment_diagnostics,
    )
    for diagnostic in plan_amendment_diagnostics:
        log(config, f"Planning issue #{issue_number}: {diagnostic}")
    plan_drift_records = planning_contract_drift_records()
    plan_contract_lineage: ContractLineage = resolve_contract_lineage(
        plan_drift_records,
        plan_board_amendments,
        plan_scheduler_contract,
        contract_from_metadata=_plan_contract_or_none,
        drift_error=plan_contract_drift_error,
    )
    plan_amendment_digest = plan_contract_lineage.active_digest
    for record in plan_drift_records:
        metadata = record.metadata
        if not staged_planning:
            continue
        if (
            metadata.scheduler_force_full
            and metadata.scheduler_force_full_source == "operator"
        ):
            plan_operator_force_full = True
        if metadata.scheduler_calls_avoided is not None:
            plan_scheduler_calls_avoided = max(
                plan_scheduler_calls_avoided, metadata.scheduler_calls_avoided
            )
    require_fresh_execution_contract = bool(
        getattr(config, "execution_strategy_contract_required", False)
    )
    # Generation-1 plans use the same fresh-contract gate as the execution
    # recommendation.  The default config enables both; test/legacy callers
    # that explicitly disable fresh planning contracts remain compatible.
    require_fresh_matrix_contract = require_fresh_execution_contract
    coder_session_id: str | None = None
    reviewer_session_ids: dict[AgentName, str | None] = {}
    unresolved_items: list[UnresolvedReviewItem] = []
    compact_prior_summaries: list[str] = []
    next_unresolved_item_number = 1
    current_plan_sidecar: AssembledPlanSidecar | None = None
    current_response_form: str | None = None
    # Authenticated revision provenance for the planning transition
    # classifier.  Both stay ``None`` across a resume boundary, so a key change
    # observed only after a restart classifies broad rather than being assumed
    # narrow from unauthenticated state.
    current_plan_patch: PlanRevisionPatch | None = None
    previous_plan_contracts: PlanCrossCuttingContracts | None = None
    # Item IDs this run has carried or minted.  A durable record that named an
    # earlier plan subject no longer makes the ledger look unreconstructible
    # once the run itself has accounted for that item, which is what a resumed
    # run does for every item the resumed round carried.
    plan_accounted_item_ids: set[str] = set()
    resume_state = _resume_plan_round(issue_context.comments, configured_reviewers=configured_reviewers)
    # Amendment activation (#943) runs right after resume reconstruction and
    # before any agent invocation or comment post: an amendment no
    # digest-bound record has used must start at the round this resume
    # re-enters.
    plan_amendment_start_round = resume_state[1].round_number if resume_state is not None else 1
    require_amendment_activation(
        plan_contract_lineage,
        start_round_number=plan_amendment_start_round,
        template=lambda amendment, round_number: _board_amendment_template(
            flow="plan",
            issue_number=issue_number,
            pr_number=None,
            persisted=plan_contract_lineage.contracts[
                plan_contract_lineage.amendments.index(amendment)
            ],
            removed=amendment.removed_reviewers,
            restored=amendment.restored_reviewers,
            start_round_number=round_number,
        ),
    )
    plan_amendment_reassignments = ()
    plan_amendment_note: str | None = None
    if plan_contract_lineage.active_amendment is not None:
        plan_amended_contract = plan_contract_lineage.contracts[-1]
        _view, plan_amendment_reassignments = apply_board_amendment_to_ledger(
            (
                (*resume_state[1].prior_items, *resume_state[1].current_round_new_items)
                if resume_state is not None
                else ()
            ),
            removed_reviewers=plan_contract_lineage.removed_reviewers,
            remaining_reviewers=plan_amended_contract.required_reviewers,
            primary_reviewer=plan_amended_contract.primary_reviewer,
        )
        plan_amendment_note = amendment_summary_line(
            plan_contract_lineage, plan_amendment_reassignments
        )
        log(config, f"Planning issue #{issue_number}: {plan_amendment_note}")
        if not amendment_audit_already_posted(
            issue_context.comments, plan_contract_lineage.active_amendment.digest
        ):
            post_issue_comment(
                runner,
                config=config,
                issue_number=issue_number,
                body=render_amendment_audit_comment(
                    plan_contract_lineage,
                    start_round_number=plan_amendment_start_round,
                    reassignments=plan_amendment_reassignments,
                ),
            )

    def plan_ledger_view(
        items: Sequence[UnresolvedReviewItem],
    ) -> tuple[UnresolvedReviewItem, ...]:
        """Derived ledger with removed reviewers' ownership reassigned (#943).

        Persisted ``prior_items`` inside a round are never rewritten; this
        view feeds scheduler obligations, dispositions, and completion.
        """
        if plan_contract_lineage.active_amendment is None:
            return tuple(items)
        amended = plan_contract_lineage.contracts[-1]
        view, _reassignments = apply_board_amendment_to_ledger(
            items,
            removed_reviewers=plan_contract_lineage.removed_reviewers,
            remaining_reviewers=amended.required_reviewers,
            primary_reviewer=amended.primary_reviewer,
        )
        return view

    if plan_supersession is not None:
        # Classify the latest reconstructable plan round before any agent
        # turn (#936): only the superseded plan itself, or a plan produced by
        # the digest-bound re-plan, may be revised, approved, or rebound.
        _require_authorized_replan_state(
            issue_context.comments,
            issue_number=issue_number,
            plan_supersession=plan_supersession,
            latest_plan=resume_state[0] if resume_state is not None else None,
            latest_round=resume_state[1].round_number if resume_state is not None else None,
        )
    # Set by the approval guard when an approved plan fails the inherited
    # check; the revision turn then runs with no reviewer item.
    inherited_guard_revision: str | None = None
    # Plan-growth gate (#886): authenticated planner candidates by round, and
    # the orchestrator growth notice that replaces an approval of a
    # non-compliant candidate with a planner revision.
    planner_candidate_rounds = _planner_candidate_rounds(issue_context.comments)
    growth_guard_revision: str | None = None

    def check_candidate_growth(
        candidate: StructuredPlanState | StructuredPlanRevision,
        *,
        canonical_text: str,
        prior_payload: Mapping[str, object] | None,
        target_round: int,
    ) -> None:
        """Post-assembly self-check of an unpublished candidate (#886).

        Measures the candidate itself: its own canonical text and the
        planner-candidate count including it.  Every failure is a correctable
        deterministic validation failure, fed back before any reviewer turn.
        """
        check_scope_ledger_preservation(prior_payload, candidate)
        if not plan_growth_gate_enforced(config):
            return
        check_growth_justification(
            candidate,
            assess_plan_growth(
                candidate,
                rendered_chars=len(canonical_text),
                revision_count=_plan_growth_candidate_count(
                    planner_candidate_rounds, target_round
                ),
                thresholds=PlanGrowthThresholds.from_config(config),
            ),
        )
    plan_validation_diagnostic: PlanValidationDiagnosticTransport | None = None
    if resume_state is None:
        plan_validation_diagnostic = _recover_current_plan_validation_diagnostic(
            runner,
            config=config,
            issue_context=issue_context,
            issue_number=issue_number,
            target_coder_round=1,
            prior_plan_subject=None,
            candidate_kind="plan_state",
            require_execution_strategy_contract=require_fresh_execution_contract,
            require_risk_test_matrix_contract=require_fresh_matrix_contract,
        )
        log(config, f"Planning issue #{issue_number}: invoking {coder_name} (context mode: full)")
        plan_human_requirements_context = render_coder_human_requirements_prompt_context(
            issue_context.human_requirements,
            requirement_scope="planning requirements",
            full_omission_fallback="Fetch the issue discussion directly before finalizing the plan.",
        )
        def invoke_fresh_planner(turn_diagnostic: object | None) -> ValidatedAgentResponse:
            return _run_validated_agent(
                runner,
                agent=config.coder,
                config=config,
                prompt=build_issue_plan_prompt(
                    issue_number,
                    config,
                    memory,
                    issue_context=issue_context,
                    plan_validation_diagnostic=turn_diagnostic,
                    inherited_matrix_binding=inherited_matrix_binding,
                ),
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking --> or <!-- AGENT_CLARIFY -->",
                require_architecture_impact_contract=True,
                **_architecture_mode_validators(lambda mode: lambda text, human_requirements=issue_context.human_requirements: _validate_response_with_human_requirements(
                    text,
                    marker_validator=lambda text: _require_plan_state_or_clarification(
                        text,
                        # Fresh planner turns always use the v1 impact contract.
                        # Document availability controls prompt material, not the
                        # response protocol or the assessment requirement.
                        required_architecture_impact_contract=1,
                        require_execution_strategy_contract=(
                            1 if require_fresh_execution_contract else 0
                        ),
                        require_risk_test_matrix_contract=(
                            1 if require_fresh_matrix_contract else 0
                        ), architecture_status_mode=mode,
                    ),
                    human_requirements=human_requirements,
                    requirement_scope="planning requirements",
                    full_omission_fallback="Fetch the issue discussion directly before finalizing the plan.",
                )),
                usage_context=usage_context,
                use_repair=True,
                repair_expected_kind="plan_state",
                repair_surfaced_requirement_ids=plan_human_requirements_context.surfaced_requirement_ids,
                repair_requires_direct_discussion_ack=plan_human_requirements_context.requires_direct_discussion_ack,
                require_execution_strategy_contract=require_fresh_execution_contract,
                require_risk_test_matrix_contract=require_fresh_matrix_contract,
                operation_description="planning",
                plan_validation_failure_handler=lambda exhaustion, error: _persist_exhausted_plan_validation_diagnostic(
                    runner,
                    config=config,
                    issue_context=issue_context,
                    issue_number=issue_number,
                    original_error=error,
                    exhaustion=exhaustion,
                    target_coder_round=1,
                    prior_plan_subject=None,
                    candidate_kind="plan_state",
                    require_execution_strategy_contract=require_fresh_execution_contract,
                    require_risk_test_matrix_contract=require_fresh_matrix_contract,
                ),
            )

        def fresh_candidate_matrix(response: ValidatedAgentResponse) -> object:
            candidate = validate_structured_plan_state(
                response.text,
                architecture_status_mode="legacy",
                require_execution_strategy_contract=(
                    1 if require_fresh_execution_contract else 0
                ),
                require_risk_test_matrix_contract=(
                    1 if require_fresh_matrix_contract else 0
                ),
            )
            if isinstance(candidate, StructuredPlanState):
                check_candidate_growth(
                    candidate,
                    canonical_text=(
                        render_canonical_plan_state(candidate, config)
                        if candidate.execution_recommendation is not None
                        else response.text
                    ),
                    prior_payload=None,
                    target_round=1,
                )
            return getattr(candidate, "risk_test_matrix", None)

        plan_response = run_inherited_checked_planner_turn(
            invoke_fresh_planner,
            derive_matrix=fresh_candidate_matrix,
            initial_diagnostic=plan_validation_diagnostic,
            candidate_kind="plan_state",
            target_coder_round=1,
            prior_plan_subject=None,
        )
        plan_output = plan_response.text
        coder_session_id = plan_response.session_id
        if is_clarification_request(plan_output):
            raise AgentLoopError(
                f"{coder_name} requested clarification during planning; human intervention required.\n\n"
                f"{coder_name}'s questions:\n{plan_output}"
            )
        current_plan = plan_output
        public_plan_output = plan_output
        raw_structured_coder_response: str | None = None
        canonical_plan: str | None = None
        structured_plan = validate_structured_plan_state(
            plan_output,
            architecture_status_mode="legacy",
            require_execution_strategy_contract=(
                1 if require_fresh_execution_contract else 0
            ),
            require_risk_test_matrix_contract=(
                1 if require_fresh_matrix_contract else 0
            ),
        )
        if isinstance(structured_plan, StructuredPlanState):
            raw_structured_coder_response = plan_output
            if structured_plan.execution_recommendation is not None:
                canonical_plan = render_canonical_plan_state(structured_plan, config)
            else:
                canonical_plan = plan_output
            current_plan = canonical_plan
            public_plan_output = render_public_agent_comment(
                kind="plan_state",
                parsed=structured_plan,
                agent=config.coder,
                config=config,
                model_used=plan_response.model_used,
            )
        else:
            # Preserve the exact free-form response as the canonical plan.
            # The public comment may have a normalized signature, but plan
            # recovery must hash the raw text selected by the handoff.
            canonical_plan = current_plan
            public_plan_output = normalize_freeform_signature(
                plan_output, agent=config.coder, config=config, model_used=plan_response.model_used
            )
        if isinstance(structured_plan, StructuredPlanState):
            # Publication-time bootstrap: the first eligible revision must
            # bind to an authenticated complete state, never to rendered text
            # or a model-echoed copy of the plan.
            current_plan_sidecar = make_assembled_plan_sidecar(
                structured_plan,
                round_number=1,
                response_form="fresh-plan-state",
                rendered_plan=canonical_plan,
            )
            current_response_form = "fresh-plan-state"
        try:
            plan_round_metadata = PostedRoundMetadata(
                    flow="plan",
                    role="coder",
                    plan_execution_mode=config.plan_execution_mode,
                    agent=coder_name,
                    round_number=1,
                    subject=_plan_subject(current_plan),
                    prior_plan_subject=None,
                    prior_items=(),
                    canonical_plan=canonical_plan,
                    raw_structured_coder_response=raw_structured_coder_response,
                    compact_prior_summaries=tuple(compact_prior_summaries),
                    model_used=plan_response.model_used,
                    **_metadata_identity_fields(plan_response),
                    acquisition_outcome=plan_response.acquisition_outcome,
                    acquisition_returncode=plan_response.acquisition_returncode,
                    **_architecture_metadata_fields(
                        config, result=plan_response.marker_value
                    ),
                    execution_strategy_contract_version=(
                        1
                        if structured_plan is not None
                        and structured_plan.execution_recommendation is not None
                        else None
                    ),
                    execution_strategy_identity=(
                        structured_plan.execution_recommendation.identity()
                        if structured_plan is not None
                        and structured_plan.execution_recommendation is not None
                        else None
                    ),
                    risk_test_matrix_contract_version=(
                        structured_plan.risk_test_matrix_contract_version
                        if structured_plan is not None else None
                    ),
                    risk_test_matrix_payload=(
                        structured_plan.risk_test_matrix.to_payload()
                        if structured_plan is not None and structured_plan.risk_test_matrix is not None
                        else None
                    ),
                    risk_test_matrix_changes_payload=(
                        tuple(change.to_payload() for change in structured_plan.risk_test_matrix_changes)
                        if structured_plan is not None else ()
                    ),
                    risk_test_matrix_identity=(
                        risk_test_matrix_identity(
                            structured_plan.risk_test_matrix,
                            structured_plan.risk_test_matrix_changes,
                        )
                        if structured_plan is not None and structured_plan.risk_test_matrix is not None
                        else None
                    ),
                    risk_test_matrix_boundary_digest=(
                        risk_test_matrix_identity(
                            structured_plan.risk_test_matrix,
                            structured_plan.risk_test_matrix_changes,
                        )
                        if structured_plan is not None and structured_plan.risk_test_matrix is not None
                        else None
                    ),
                    response_form=(
                        current_response_form if current_plan_sidecar is not None else None
                    ),
                    aggregate_plan_identity=(
                        current_plan_sidecar.aggregate_identity
                        if current_plan_sidecar is not None else None
                    ),
                    assembled_plan_sidecar=(
                        current_plan_sidecar.to_payload()
                        if current_plan_sidecar is not None else None
                    ),
            )
        except ValueError as exc:
            # A contradictory metadata record must not escape as a bare
            # dataclass ValueError with no diagnostic (#879).
            raise AgentLoopError(
                "Could not record the plan round metadata for round 1 "
                f"(response form {current_response_form or 'free-form'}): {exc}"
            ) from exc
        if isinstance(structured_plan, StructuredPlanState):
            plan_round_body = _assemble_structured_plan_round_body(
                config=config,
                issue_number=issue_number,
                kind="plan_state",
                parsed_plan=structured_plan,
                full_comment=public_plan_output,
                metadata=plan_round_metadata,
                raw_text=plan_output,
                prior_items=(),
                model_used=plan_response.model_used,
                surfaced_requirement_ids=plan_human_requirements_context.surfaced_requirement_ids,
                requires_direct_discussion_ack=(
                    plan_human_requirements_context.requires_direct_discussion_ack
                ),
            )
        else:
            plan_round_body = _attach_round_metadata(public_plan_output, plan_round_metadata)
        if _post_plan_coder_round_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=plan_round_body,
            diagnostic=plan_validation_diagnostic,
            target_coder_round=1,
            prior_plan_subject=None,
            candidate_kind="plan_state",
            require_execution_strategy_contract=require_fresh_execution_contract,
            require_risk_test_matrix_contract=require_fresh_matrix_contract,
        ):
            plan_validation_diagnostic = None
        if current_plan_sidecar is not None:
            planner_candidate_rounds.add(1)
        start_round_number = 1
        resumed_round: ResumedReviewRound | None = None
        current_coder_output = plan_output
    else:
        current_plan, resumed_round = resume_state
        current_coder_output = resumed_round.coder_output
        unresolved_items = list(resumed_round.prior_items)
        compact_prior_summaries = list(
            bound_compact_prior_summaries(resumed_round.compact_prior_summaries)
        )
        next_unresolved_item_number = resumed_round.next_unresolved_item_number
        start_round_number = resumed_round.round_number
        if resumed_round.coder_metadata is not None and resumed_round.coder_metadata.assembled_plan_sidecar is not None:
            current_plan_sidecar = decode_assembled_plan_sidecar(
                resumed_round.coder_metadata.assembled_plan_sidecar
            )
            current_response_form = resumed_round.coder_metadata.response_form
        # Rebuild the authenticated classifier inputs from durable records, so
        # a restart right after a remediation coder turn classifies the same
        # plan-step revision narrow instead of latching the complete board.
        current_plan_patch, previous_plan_contracts = _resumed_plan_transition_inputs(
            issue_context.comments,
            coder_metadata=resumed_round.coder_metadata,
        )
        plan_accounted_item_ids.update(
            item.item_id
            for item in (*resumed_round.prior_items, *resumed_round.current_round_new_items)
        )
        log(config, f"Planning issue #{issue_number}: resuming round {start_round_number}")
        if inherited_matrix_binding is not None and _resumed_inherited_replan_force_full(
            issue_context.comments
        ):
            # The revision of an inadmissible approved plan gets the complete
            # board even when the run restarted right after that round (#936).
            plan_automatic_force_full = True
        # A resumed round carries its planning-generation discriminator in
        # durable coder metadata. Historical rounds intentionally have no
        # discriminator and must remain legacy-undecided; applying the fresh
        # gate here would force an old plan to fabricate a matrix on its next
        # revision.
        require_fresh_matrix_contract = bool(
            require_fresh_matrix_contract
            and resumed_round.coder_metadata is not None
            and resumed_round.coder_metadata.risk_test_matrix_contract_version == 1
        )
        plan_validation_diagnostic = _recover_current_plan_validation_diagnostic(
            runner,
            config=config,
            issue_context=issue_context,
            issue_number=issue_number,
            target_coder_round=start_round_number + 1,
            prior_plan_subject=_plan_subject(current_plan),
            candidate_kind="plan_revision",
            require_execution_strategy_contract=require_fresh_execution_contract,
            require_risk_test_matrix_contract=require_fresh_matrix_contract,
        )

    narrow_directive_delivered = False
    try:
        mode_history_records = _extract_round_metadata_records(
            issue_context.comments, flow="plan"
        )
    except AgentLoopError:
        mode_history_records = ()
    # Advisory per-run finding history (#1273), seeded once from the planning
    # records already extracted above; an undecodable authoritative record set
    # keeps its existing stop and never reaches this point with history work.
    plan_finding_history = FindingHistoryLedger(
        _HISTORY_PHASE_PLAN, log=lambda message: log(config, message)
    )
    plan_finding_history.seed_from_records(
        mode_history_records, reconciliation_mode="aggregate", same_status="same-plan"
    )
    prior_execution_mode = prior_plan_execution_mode(mode_history_records)
    execution_mode_history_present = bool(mode_history_records)
    if prior_execution_mode is not None and prior_execution_mode != config.plan_execution_mode:
        log(
            config,
            f"Planning: execution mode changed from {prior_execution_mode} to "
            f"{config.plan_execution_mode} since the last recorded round",
        )

    for round_number in range(start_round_number, config.max_rounds + 1):
        current_resume = resumed_round if resumed_round is not None and round_number == resumed_round.round_number else None
        prior_unresolved_items = current_resume.prior_items if current_resume is not None else tuple(unresolved_items)
        plan_finding_history.note_carried_ledger(prior_unresolved_items)
        prior_dispositions: dict[str, list[ReviewItemDisposition]] = {
            item.item_id: [] for item in prior_unresolved_items
        }
        round_new_unresolved_items: list[UnresolvedReviewItem] = list(
            current_resume.current_round_new_items
            if current_resume is not None and current_resume.reconciled
            else ()
        )
        current_plan_subject = _plan_subject(current_plan)
        round_resolved_history_item_ids = _round_resolved_history_item_ids(
            prior_unresolved_items=prior_unresolved_items,
            comments=issue_context.comments,
            flow="plan",
            reconciliation_mode="aggregate",
            same_status="same-plan",
        )
        plan_accounted_item_ids.update(
            item.item_id
            for item in (*prior_unresolved_items, *round_new_unresolved_items)
        )
        plan_accounted_item_ids.update(round_resolved_history_item_ids)
        round_ledger_incomplete = _round_ledger_may_be_incomplete(
            current_resume=current_resume,
            prior_unresolved_items=prior_unresolved_items,
            comments=issue_context.comments,
            flow="plan",
            current_subject=current_plan_subject,
            # Compatibility default: full-board planning keeps the previous
            # conservative reading, where any cross-subject item at all makes
            # the ledger unreconstructible, so its context-mode selection and
            # posted bodies are unchanged by staged planning (#905).
            accounted_item_ids=(
                tuple(sorted(plan_accounted_item_ids)) if staged_planning else ()
            ),
        )
        plan_hr_ids = _surfaced_reviewer_requirement_ids(
            issue_context.human_requirements,
            requirement_scope="planning requirements",
        )
        current_plan_key = _plan_candidate_key_for(
            plan_subject=current_plan_subject,
            sidecar=current_plan_sidecar,
            surfaced_requirement_ids=plan_hr_ids,
        )
        plan_growth_assessment, plan_growth_violation = _plan_growth_gate_violation(
            config,
            plan_payload=(
                current_plan_sidecar.canonical_json if current_plan_sidecar is not None else None
            ),
            plan_text=current_plan,
            # Counted up to the round that published this candidate, never the
            # loop round: a reviewer-only phase-advance round keeps the count.
            revision_count=_plan_growth_candidate_count(
                planner_candidate_rounds,
                current_plan_sidecar.round_number
                if current_plan_sidecar is not None
                else round_number,
            ),
        )
        if plan_growth_assessment is not None:
            log(
                config,
                f"Planning round {round_number}: plan growth measurements: "
                f"{plan_growth_assessment.describe()}; crossed="
                f"{', '.join(plan_growth_assessment.crossed) or 'none'}; gate="
                f"{config.plan_growth_gate}",
            )
        # Only a candidate recovered from history can reach reviewers in this
        # state: fresh candidates are rejected by the self-check.
        plan_growth_notice = (
            render_growth_notice(
                plan_growth_assessment,
                violation=plan_growth_violation,
                strategy=(
                    plan_strategy(current_plan_sidecar.canonical_json)
                    if current_plan_sidecar is not None
                    else None
                ),
            )
            if plan_growth_violation is not None and plan_growth_assessment is not None
            else None
        )
        # Reviewers see the measurements of every crossed one-shot candidate,
        # justified or not; the corrective notice is only for non-compliant ones.
        plan_growth_measurements = (
            render_growth_measurements(plan_growth_assessment)
            if plan_growth_assessment is not None
            and plan_growth_assessment.crossed
            and current_plan_sidecar is not None
            and plan_strategy(current_plan_sidecar.canonical_json) == "one-shot"
            else None
        )
        plan_scheduler_decision = None
        stall_deferred_for_step_back = False
        step_back_candidate_review = False
        step_back_review_anchor: PlanStepBackAnchor | None = None
        pre_round_step_back_episode = None
        step_back_history_intact = False
        step_back_history_class = "unclassified"
        plan_posted_checkpoint: PostedRoundMetadata | None = None
        plan_panel_evidence = PlanPanelEvidence()
        plan_qualifying_approvals: tuple[str, ...] = ()
        plan_previous_key: PlanCandidateKey | None = None
        # One private spool per round, shared by the parallel launcher and the
        # sequential resume seam (#1025).
        plan_round_spool = _review_round_spool(
            config, surface="plan", number=issue_number,
            round_number=round_number, subject=current_plan_subject,
        )

        def _plan_review_validators(reviewer_name: str) -> dict[str, object]:
            return _architecture_mode_validators(lambda mode: lambda text, reviewer_name=reviewer_name: _validate_plan_review_response(
                text,
                reviewer=reviewer_name,
                unresolved_items=prior_unresolved_items,
                # Never share the mutable round_new_unresolved_items list
                # with concurrent workers (#594): it only enriches the
                # UnknownPriorItemDispositionError message, so an empty
                # tuple here changes no validation outcome.
                current_round_items=(),
                surfaced_requirement_ids=_surfaced_reviewer_requirement_ids(
                    issue_context.human_requirements,
                    requirement_scope="planning requirements",
                ), architecture_status_mode=mode,
            ))

        def _plan_publication(reviewer_name: str):
            return round_publication(
                runner, config=config, spool=plan_round_spool,
                reviewer_name=reviewer_name, flow="plan", round_number=round_number,
                subject=current_plan_subject, surface_kind="issue", number=issue_number,
                validation_context=publication_context_digest(
                    candidate=(
                        current_plan_key.as_dict() if staged_planning else current_plan_subject
                    ),
                    surfaced=sorted(str(item) for item in plan_hr_ids),
                ),
            )

        plan_spool_preflight_done = [False]

        def _plan_spool_preflight() -> None:
            """Check every spool record before any checkpoint or reviewer launch (#1258)."""
            if plan_spool_preflight_done[0]:
                return
            plan_spool_preflight_done[0] = True
            published = [
                record.metadata.agent
                for record in (current_resume.completed_reviews if current_resume is not None else ())
            ]
            _preflight_spooled_publications(
                runner, config=config, spool=plan_round_spool, reviewers=configured_reviewers,
                validators_for=_plan_review_validators, publication_for=_plan_publication,
                published_names=published,
                post_frozen=lambda plan: post_frozen_round_bodies(
                    runner, config=config, surface_kind="issue", number=issue_number, plan=plan
                ),
                selected_reviewers=round_reviewers,
            )

        round_reviewers = tuple(configured_reviewers)
        # Staged candidate under a guard-active execution mode (#1268).  The
        # phased-delivery guard can never be satisfied there, so stop for a
        # human decision before any reviewer or revision turn (and before the
        # growth seam and the scheduler block, so every review policy is
        # covered) unless --plan-narrow-staged owns this round's revision.
        staged_mode_revision = False
        if phased_delivery_guard_active(config.plan_execution_mode):
            try:
                round_recommendation = _current_execution_recommendation(
                    current_plan, issue_context.comments
                )
            except AgentLoopError:
                round_recommendation = None
            if round_recommendation is not None and round_recommendation.strategy == "staged":
                if config.plan_narrow_staged and not narrow_directive_delivered:
                    staged_mode_revision = True
                else:
                    stop_plan_pre_panel(
                        staged_plan_mode_conflict_message(config.plan_execution_mode),
                        round_number=round_number,
                    )
        # Under primary-then-panel, a non-compliant candidate the primary has
        # already approved can never pass, so the panel is not scheduled on it:
        # this round invokes no reviewer and the gate below starts a planner
        # revision instead of a reviewer-only phase advance (#886).
        growth_skip_panel = False
        if (
            staged_planning
            and (plan_growth_notice is not None or staged_mode_revision)
            and plan_primary_name is not None
        ):
            plan_records = plan_history_records(refresh=True, round_number=round_number)
            growth_skip_panel = plan_primary_name in _carried_plan_approvals(
                plan_records,
                current_key=current_plan_key,
                required_reviewers=plan_reviewer_names,
                surfaced_requirement_ids=plan_hr_ids,
                panel_evidence=_derive_plan_panel_evidence(
                    plan_records,
                    primary_reviewer=plan_primary_name,
                    required_reviewers=plan_reviewer_names,
                ),
                primary_reviewer=plan_primary_name,
            )
        if growth_skip_panel:
            round_reviewers = ()
            log(
                config,
                f"Planning round {round_number}: the primary approved a candidate that "
                + (
                    "must be narrowed (--plan-narrow-staged)"
                    if staged_mode_revision and plan_growth_notice is None
                    else "fails the plan-growth gate"
                )
                + "; no panel review is scheduled on it",
            )
        if staged_planning and not growth_skip_panel:
            assert plan_scheduler_contract is not None
            incomplete_key = current_plan_key.incompleteness_reason()
            if incomplete_key is not None:
                raise AgentLoopError(
                    "--plan-review-policy primary-then-panel requires a generation-1 plan: "
                    f"{incomplete_key}. Re-plan the issue so the canonical plan carries an "
                    "execution-strategy recommendation and a risk matrix, or rerun with the "
                    "compatibility default --plan-review-policy all-reviewers."
                )
            plan_records = plan_history_records(refresh=True, round_number=round_number)
            plan_panel_evidence = _derive_plan_panel_evidence(
                plan_records,
                primary_reviewer=plan_primary_name,
                required_reviewers=plan_reviewer_names,
            )
            plan_qualifying_approvals = _carried_plan_approvals(
                plan_records,
                current_key=current_plan_key,
                required_reviewers=plan_reviewer_names,
                surfaced_requirement_ids=plan_hr_ids,
                panel_evidence=plan_panel_evidence,
                primary_reviewer=plan_primary_name,
                restoration_rounds=plan_contract_lineage.restoration_rounds,
            )
            plan_history = _classify_staged_plan_history(
                plan_records, current_key=current_plan_key
            )
            latest_scheduler_record = plan_history.latest_scheduler_record
            plan_previous_key = plan_history.previous_key
            plan_history_class = plan_history.history_class
            step_back_history_intact = plan_history_class == PLAN_HISTORY_INTACT
            step_back_history_class = str(plan_history_class)
            if config.plan_step_back_rounds > 0 and plan_primary_name is not None:
                step_back_candidate_review = (
                    current_plan_sidecar.round_number
                    if current_plan_sidecar is not None
                    else round_number
                ) in plan_step_back_candidate_rounds(plan_records)
            if plan_history_class == PLAN_HISTORY_CONTRADICTORY_KEY:
                # A contradictory persisted key can supply neither an approval
                # nor a panel opening.
                plan_qualifying_approvals = ()
            plan_reset_round = (
                config.plan_reset_stall_streak
                and not plan_reset_checkpoint_written
                and plan_scheduler_contract is not None
            )
            if plan_reset_round and config.plan_primary_stall_rounds > 0:
                log(
                    config,
                    f"Planning round {round_number}: stall streak reset by operator "
                    "(--plan-reset-stall-streak)",
                )
            if (
                config.plan_step_back_rounds > 0
                and plan_primary_name is not None
                and step_back_history_intact
                and not plan_panel_evidence.opened
            ):
                pre_round_state = plan_step_back_state(
                    plan_records, panel_opening_index=plan_panel_evidence.opening_index
                )
                if pre_round_state is not None and pre_round_state.episode is not None:
                    pre_round_step_back_episode = pre_round_state.episode
                    # A reset on this round closes the episode before reviewers run.
                    if not plan_reset_round:
                        step_back_review_anchor = plan_step_back_anchor(
                            plan_records, pre_round_state.episode
                        )
            if (
                config.plan_step_back_rounds > 0
                and not staged_mode_revision
                and not plan_reset_round
                and plan_primary_name is not None
                and not plan_panel_evidence.opened
                and not plan_operator_force_full
                and plan_primary_name not in plan_qualifying_approvals
                and not (
                    current_resume is not None
                    and any(
                        record.metadata.agent == plan_primary_name
                        for record in current_resume.completed_reviews
                    )
                )
            ):
                # Independent of --plan-primary-stall-rounds: only
                # --plan-step-back-rounds 0 disables the step-back stop.  It runs
                # before the stall stop, which a pending or active step-back defers.
                if not step_back_history_intact:
                    log(
                        config,
                        f"Planning round {round_number}: step-back stop suppressed: degraded "
                        f"planning history ({step_back_history_class})",
                    )
                else:
                    step_back_stop = plan_step_back_escalation(
                        plan_records,
                        panel_opening_index=plan_panel_evidence.opening_index,
                        assessment=plan_growth_assessment,
                        measurements=plan_growth_measurements,
                    )
                    if step_back_stop is not None:
                        stop_plan_pre_panel(step_back_stop, round_number=round_number)
            if (
                config.plan_primary_stall_rounds > 0
                and not staged_mode_revision
                and not plan_reset_round
                and plan_primary_name is not None
                and not plan_panel_evidence.opened
                and not plan_operator_force_full
                and plan_primary_name not in plan_qualifying_approvals
                and not (
                    current_resume is not None
                    and any(
                        record.metadata.agent == plan_primary_name
                        for record in current_resume.completed_reviews
                    )
                )
            ):
                if plan_history_class != PLAN_HISTORY_INTACT:
                    log(
                        config,
                        f"Planning round {round_number}: stall stop suppressed: degraded "
                        f"planning history ({plan_history_class})",
                    )
                else:
                    plan_primary_streak = plan_primary_blocking_streak_detail(
                        plan_records,
                        primary=plan_primary_name,
                        panel_opening_index=plan_panel_evidence.opening_index,
                        current_issue_digest=plan_issue_digest,
                        current_execution_mode=config.plan_execution_mode,
                    )
                    if plan_primary_streak.count >= config.plan_primary_stall_rounds:
                        disposition = _stall_step_back_disposition()
                        if disposition.kind == DISPOSITION_DEFER_PENDING:
                            stall_deferred_for_step_back = True
                            log(
                                config,
                                f"Planning round {round_number}: stall stop deferred: "
                                f"pending plan step-back ({disposition.reason})",
                            )
                        elif disposition.kind == DISPOSITION_DEFER_EPISODE:
                            log(
                                config,
                                f"Planning round {round_number}: stall stop deferred: "
                                f"{disposition.reason}",
                            )
                        else:
                            # Before the prelaunch checkpoint and every agent turn,
                            # so the stop writes no record a resume could read as
                            # a checkpoint, an approval, or a panel opening.
                            escalation = (
                                plan_step_back_escalation(
                                    plan_records,
                                    panel_opening_index=plan_panel_evidence.opening_index,
                                    assessment=plan_growth_assessment,
                                    measurements=plan_growth_measurements,
                                )
                                if disposition.kind == DISPOSITION_ESCALATED
                                else None
                            )
                            stop_plan_pre_panel(
                                escalation
                                if escalation is not None
                                else plan_primary_stall_message(
                                    streak=plan_primary_streak.count,
                                    threshold=config.plan_primary_stall_rounds,
                                    plan_chars=len(current_plan),
                                    legacy_undigested=plan_primary_streak.edit_cannot_clear(
                                        config.plan_primary_stall_rounds
                                    ),
                                    step_back_status=disposition.reason,
                                ),
                                round_number=round_number,
                            )
            plan_classification = classify_plan_transition(
                plan_previous_key,
                current_plan_key,
                _plan_revision_descriptor(
                    response_form=current_response_form,
                    sidecar=current_plan_sidecar,
                    patch=current_plan_patch,
                ),
                previous_contracts=previous_plan_contracts,
                current_contracts=_plan_cross_cutting_contracts(current_plan_sidecar),
                ledger_reconstructible=not round_ledger_incomplete,
            )
            if plan_panel_evidence.opened and plan_panel_evidence.post_opening_automatic_latch:
                plan_automatic_force_full = True
            plan_snapshot = PlanSchedulerSnapshot(
                contract=plan_scheduler_contract,
                previous_key=plan_previous_key,
                current_key=current_plan_key,
                obligations=_scheduler_obligations(
                    plan_ledger_view(prior_unresolved_items),
                    required_reviewers=plan_reviewer_names,
                    active_statuses=frozenset({"blocking", "same-plan"}),
                ),
                force_full=plan_automatic_force_full,
                force_full_source="automatic" if plan_automatic_force_full else None,
                operator_force_full=plan_operator_force_full,
                panel_evidence=plan_panel_evidence.opened,
                phase=(
                    latest_scheduler_record.metadata.scheduler_phase
                    if latest_scheduler_record is not None
                    else None
                ),
                degraded_history_class=plan_history_class,
                premature_secondary_reviews=(
                    () if plan_panel_evidence.opened
                    else plan_panel_evidence.premature_secondary_reviews
                ),
            )
            try:
                plan_scheduler_decision = select_plan_reviewers(
                    plan_snapshot,
                    plan_classification,
                    qualifying_approvals=plan_qualifying_approvals,
                    phase=plan_snapshot.phase,
                )
            except PlanPrePanelSafetyError as exc:
                stop_plan_pre_panel(str(exc), round_number=round_number)
                raise
            plan_scheduler_decision = _keep_reused_amendment_round_reviews(
                plan_scheduler_decision,
                lineage=plan_contract_lineage,
                round_number=round_number,
                current_resume=current_resume,
                eligible=lambda name: (
                    name == plan_primary_name or plan_panel_evidence.opened
                ),
            )
            if plan_scheduler_decision.latches_force_full:
                plan_automatic_force_full = True
            plan_scheduler_calls_avoided += plan_scheduler_decision.calls_avoided
            plan_recorded_force_full, plan_recorded_force_full_source = (
                _scheduler_recorded_force_full(
                    operator=plan_operator_force_full,
                    automatic=plan_automatic_force_full,
                )
            )
            plan_selected_names = set(plan_scheduler_decision.selected_reviewers)
            round_reviewers = tuple(
                reviewer
                for reviewer in configured_reviewers
                if agent_display_name(reviewer) in plan_selected_names
            )
            log(
                config,
                f"Planning round {round_number}: {plan_scheduler_contract.policy} plan scheduler "
                f"phase={plan_scheduler_decision.phase} {plan_scheduler_decision.reason}; "
                f"selected={', '.join(plan_scheduler_decision.selected_reviewers) or 'none'}; "
                "paused="
                f"{', '.join(f'{name} ({why})' for name, why in plan_scheduler_decision.paused_reviewers) or 'none'}; "
                f"primary={plan_primary_name or 'none'}; active_owners="
                f"{', '.join(plan_scheduler_decision.active_owners) or 'none'}; "
                f"panel_evidence={plan_panel_evidence.opened}; "
                f"force_full={plan_recorded_force_full} "
                f"(source: {plan_recorded_force_full_source or 'none'}); "
                f"degraded_history={plan_history_class}; "
                f"calls_avoided_cumulative={plan_scheduler_calls_avoided}"
                + (f"; {plan_amendment_note}" if plan_amendment_note else ""),
            )
            _plan_spool_preflight()
            plan_posted_checkpoint = PostedRoundMetadata(
                flow="plan",
                role="summary",
                agent="Orchestrator",
                round_number=round_number,
                subject=current_plan_subject,
                prior_items=prior_unresolved_items,
                phase="scheduler-prelaunch",
                scheduler_contract=plan_scheduler_contract.as_dict(),
                reviewer_board_amendment_digest=plan_amendment_digest,
                scheduler_obligation_digest=hashlib.sha256(
                    repr(_prior_item_ledger_signature(prior_unresolved_items)).encode("utf-8")
                ).hexdigest()[:16],
                scheduler_selected_reviewers=plan_scheduler_decision.selected_reviewers,
                scheduler_paused_reviewers=plan_scheduler_decision.paused_reviewers,
                scheduler_reasons=(
                    plan_scheduler_decision.reason,
                    plan_classification.reason,
                ),
                scheduler_final_sweep=plan_scheduler_decision.final_sweep,
                scheduler_force_full=plan_recorded_force_full,
                scheduler_force_full_source=plan_recorded_force_full_source,
                scheduler_calls_avoided=plan_scheduler_calls_avoided,
                scheduler_phase=plan_scheduler_decision.phase,
                scheduler_primary_reviewer=plan_primary_name,
                scheduler_approved_reviewers=tuple(plan_qualifying_approvals),
                scheduler_active_owners=plan_scheduler_decision.active_owners,
                plan_candidate_key=current_plan_key.as_dict(),
                scheduler_plan_previous_key=(
                    plan_previous_key.as_dict() if plan_previous_key is not None else None
                ),
                scheduler_issue_digest=(
                    plan_issue_digest
                    if plan_scheduler_decision.phase == "primary"
                    else None
                ),
                scheduler_stall_reset=(
                    plan_reset_round and plan_scheduler_decision.phase == "primary"
                ),
                scheduler_step_back_deferral=(
                    stall_deferred_for_step_back
                    and plan_scheduler_decision.phase == "primary"
                ),
                scheduler_execution_mode=config.plan_execution_mode,
                **_architecture_metadata_fields(config),
            )
            post_issue_comment(
                runner,
                config=config,
                issue_number=issue_number,
                body=_attach_round_metadata(
                    _append_board_amendment_note(render_plan_scheduling_audit(
                        phase=plan_scheduler_decision.phase,
                        reason=plan_scheduler_decision.reason,
                        selected=plan_scheduler_decision.selected_reviewers,
                        paused=plan_scheduler_decision.paused_reviewers,
                        primary=plan_primary_name,
                        active_owners=plan_scheduler_decision.active_owners,
                        plan_subject=current_plan_subject,
                        panel_evidence=plan_panel_evidence.opened,
                        degraded_history_class=plan_history_class,
                        degraded_history_reason=plan_history_fallback_reason(plan_history_class),
                        force_full=plan_recorded_force_full,
                        force_full_source=plan_recorded_force_full_source,
                        calls_avoided=plan_scheduler_calls_avoided,
                        unqualified_artifacts=(
                            () if plan_panel_evidence.opened
                            else plan_panel_evidence.unqualified_artifacts
                        ),
                    ), plan_amendment_note),
                    plan_posted_checkpoint,
                ),
            )
            if plan_reset_round and plan_scheduler_decision.phase == "primary":
                plan_reset_checkpoint_written = True
        _plan_spool_preflight()
        use_compact_context = (
            config.planning_context_mode == "compact"
            and round_number >= 2
            and not round_ledger_incomplete
        )
        context_mode = "compact" if use_compact_context else "full"
        context_reason = " (ledger incomplete)" if (
            config.planning_context_mode == "compact"
            and round_number >= 2
            and round_ledger_incomplete
        ) else ""
        blocking_reviews: list[tuple[str, str]] = []
        approved_review_outputs: list[tuple[str, str]] = []
        # The accepted carrier beside each approved text, keyed by reviewer, so
        # an acknowledgement repair can pin its assessment and records (#925).
        accepted_review_carriers: dict[str, ParsedPlanReview | ParsedReview] = {}
        all_approved = True
        # Actual models of reviewers that reviewed in THIS round (fresh or resumed
        # same-round records), for the same-model panel note (#1236).
        round_review_models: dict[str, str | None] = {}
        resumed_by_name = {
            record.metadata.agent: record for record in (current_resume.completed_reviews if current_resume is not None else ())
        }
        # Superseded pre-panel plan reviews, replayed to their own author only
        # as non-authoritative context under the operator planning override.
        superseded_plan_prepanel: dict[str, PostedRoundRecord] = {}
        # Item IDs claimed only by superseded pre-panel reviews; excluded from
        # the ledger even when the record itself is not replayed as context.
        superseded_prepanel_item_ids: set[str] = set()
        if staged_planning and plan_primary_name is not None:
            def _is_unqualified_prepanel(record: PostedRoundRecord) -> bool:
                return not (
                    plan_panel_evidence.opening_index is not None
                    and record.index > plan_panel_evidence.opening_index
                )

            if plan_operator_force_full:
                # The override authorizes the complete board over exactly the
                # artifacts the pre-panel diagnostic would otherwise stop on,
                # so each superseded secondary gets its own earlier claims back
                # as context, and never as findings, ownership, or approval.
                # A reviewer that already produced a qualified post-opening
                # review has had its fresh turn, so it needs no replay.
                already_freshly_invoked = {
                    record.metadata.agent
                    for record in plan_records
                    if record.metadata.role == "reviewer"
                    and not _is_unqualified_prepanel(record)
                }
                for record in sorted(plan_records, key=lambda item: item.index):
                    metadata = record.metadata
                    if (
                        metadata.role == "reviewer"
                        and metadata.agent
                        and metadata.agent != plan_primary_name
                        and metadata.agent in plan_reviewer_names
                        and metadata.agent not in already_freshly_invoked
                        and _is_unqualified_prepanel(record)
                    ):
                        superseded_plan_prepanel[metadata.agent] = record
                        superseded_prepanel_item_ids.update(
                            item.item_id for item in metadata.new_items
                        )
            # A secondary plan review recorded before any qualified panel
            # opening is an unqualified artifact: it is never resumed as
            # settled work, never an approval, and never ownership.  The
            # reviewer is freshly invoked once a qualified opening exists.
            for name in [
                name
                for name, record in resumed_by_name.items()
                if name != plan_primary_name and _is_unqualified_prepanel(record)
            ]:
                log(
                    config,
                    f"Planning round {round_number}: {name}'s plan review predates any "
                    "qualified panel opening; it is superseded, non-authoritative "
                    "context and is not resumed as settled work",
                )
                if plan_operator_force_full:
                    superseded_plan_prepanel[name] = resumed_by_name[name]
                superseded_prepanel_item_ids.update(
                    item.item_id for item in resumed_by_name[name].metadata.new_items
                )
                resumed_by_name.pop(name, None)
            if superseded_prepanel_item_ids:
                # A reconciled resume rehydrates every current-round item,
                # including the superseded secondary's own claims (and their
                # duplicates on the reconciliation summary).  Those claims are
                # excluded from finding and ownership accounting, so they must
                # not survive as must-fix obligations once the same secondary
                # is freshly invoked.
                rehydrated = [
                    item
                    for item in round_new_unresolved_items
                    if item.item_id in superseded_prepanel_item_ids
                ]
                if rehydrated:
                    log(
                        config,
                        f"Planning round {round_number}: dropping superseded pre-panel plan "
                        "item(s) "
                        + ", ".join(sorted({item.item_id for item in rehydrated}))
                        + " from the current-round ledger; they establish no obligation",
                    )
                    round_new_unresolved_items[:] = [
                        item
                        for item in round_new_unresolved_items
                        if item.item_id not in superseded_prepanel_item_ids
                    ]
            if superseded_plan_prepanel:
                log(
                    config,
                    f"Planning round {round_number}: replaying superseded pre-panel plan "
                    "review context to "
                    + ", ".join(sorted(superseded_plan_prepanel))
                    + " as non-authoritative context only",
                )

        inherited_review_deltas, inherited_review_failure = inherited_review_context(
            current_plan
        )

        def _build_plan_review_prompt(reviewer: AgentName) -> str:
            # Built once per reviewer from pre-round state only, so the same
            # prompt is produced regardless of sequential or parallel launch.
            return build_plan_review_prompt(
                issue_number,
                round_number,
                current_plan,
                config,
                reviewer=reviewer,
                memory=memory,
                issue_context=issue_context,
                unresolved_items=prior_unresolved_items,
                compact_context=use_compact_context,
                compact_prior=CompactPriorContext(tuple(compact_prior_summaries)),
                compact_tail=CompactPlanTailContext(
                    subject=current_plan_subject,
                    action=(
                        "Review the current plan for correctness, architecture fit, "
                        "missing edge cases, test strategy, and ambiguity."
                    ),
                ),
                # Each reviewer sees only its own superseded pre-panel review.
                superseded_prepanel_review=_superseded_prepanel_plan_review(
                    superseded_plan_prepanel.get(agent_display_name(reviewer))
                ),
                inherited_matrix_binding=inherited_matrix_binding,
                inherited_reviewed_deltas=inherited_review_deltas,
                inherited_check_failure=inherited_review_failure,
                plan_growth_notice=plan_growth_notice,
                plan_growth_measurements=plan_growth_measurements,
                step_back_review_notice=step_back_candidate_review,
                step_back_anchor=step_back_review_anchor,
                prior_execution_mode=prior_execution_mode,
                execution_mode_history_present=execution_mode_history_present,
            )

        plan_fatal_errors: list[tuple[str, AgentLoopError]] = []
        plan_turn_results: dict[AgentName, _ReviewerTurnResult] = {}
        early_published_plan_reviewers: set[AgentName] = {
            reviewer
            for reviewer in round_reviewers
            if (
                (record := resumed_by_name.get(agent_display_name(reviewer))) is not None
                and record.metadata.phase == "publication"
            )
        }

        def _post_plan_reviewer_comment(
            reviewer_name: str,
            parsed: ParsedPlanReview,
            *,
            review_output: str,
            model_used: str | None,
            identity: ValidatedAgentResponse | None = None,
            acquisition_outcome: str = "success",
            acquisition_returncode: int | None = None,
            new_items: tuple[UnresolvedReviewItem, ...] = (),
            phase: str = "authoritative",
        ) -> None:
            """Post one plan review using the same rendering and durable record."""
            # A spooled reviewer's bodies are frozen before the first post so a
            # rerun after an outage resumes them verbatim (#1258).
            publication_hook = (
                {"publication": _plan_publication(reviewer_name)}
                if phase == "publication" and identity is not None
                else {}
            )
            post_issue_comment(
                runner, config=config, issue_number=issue_number, **publication_hook,
                body=_attach_round_metadata(
                    render_public_agent_comment(
                        kind="plan_review", parsed=parsed, agent=reviewer_name,
                        prior_items=prior_unresolved_items, dispositions=parsed.dispositions,
                        human_requirements_resolved_flag=human_requirements_resolved(review_output),
                        config=config, model_used=model_used,
                    ),
                    PostedRoundMetadata(
                        flow="plan", role="reviewer", agent=reviewer_name,
                        round_number=round_number, subject=_plan_subject(current_plan),
                        plan_execution_mode=config.plan_execution_mode,
                        prior_items=prior_unresolved_items, dispositions=parsed.dispositions,
                        new_items=new_items, state=parsed.state,
                        # Staged planning only: a full-board planning run keeps
                        # writing exactly today's record shape.  The surfaced
                        # planning-requirement IDs are persisted only when this
                        # review actually carried HUMAN_REQUIREMENTS_RESOLVED,
                        # so a later round cannot satisfy the signed-requirement
                        # gate vacuously through a carried approval.
                        plan_candidate_key=(
                            current_plan_key.as_dict() if staged_planning else None
                        ),
                        surfaced_reviewer_requirement_ids=(
                            tuple(plan_hr_ids)
                            if staged_planning and human_requirements_resolved(review_output)
                            else ()
                        ),
                        compact_prior_summaries=tuple(compact_prior_summaries),
                        model_used=model_used, phase=phase,
                        **(_metadata_identity_fields(identity) if identity is not None else {}),
                        acquisition_outcome=acquisition_outcome,
                        acquisition_returncode=acquisition_returncode,
                        **_architecture_metadata_fields(config, result=parsed),
                        canonical_reviewer_response=(review_output if phase == "publication" else None),
                    ),
                ),
            )

        # A round that holds withheld outcomes began as a parallel round; finish
        # it through the same withhold-then-publish launcher even when this run
        # is sequential, so no retried reviewer sees a same-round peer's body.
        plan_round_parallel = config.review_parallel or plan_round_spool.has_records()
        plan_today_peers = tuple(
            reviewer for reviewer in round_reviewers
            if (record := resumed_by_name.get(agent_display_name(reviewer))) is not None
            and record.metadata.phase == "publication"
        )
        plan_round_public_peers = plan_today_peers
        plan_recovery_context: RecoveryVisibilityContext | None = None
        if staged_planning and plan_scheduler_decision is not None:
            def _derive_plan_visibility_evidence(
                records: Sequence[PostedRoundRecord],
            ) -> PlanPanelEvidence:
                return _derive_plan_panel_evidence(
                    records,
                    primary_reviewer=plan_primary_name,
                    required_reviewers=plan_reviewer_names,
                )

            plan_recovery_context = RecoveryVisibilityContext(
                primary_reviewer=plan_primary_name,
                launch_phase=plan_scheduler_decision.phase,
                launching=tuple(sorted(plan_selected_names)),
                derive_evidence=_derive_plan_visibility_evidence,
            )
            if plan_posted_checkpoint is not None:
                # Same visibility rule as the PR guard (#1156): every same-round
                # review already public counts, whatever its scheduler phase and
                # whether or not its author was selected again, including
                # records resume stripped as unqualified pre-opening.
                try:
                    plan_fresh_comments = get_issue_context(
                        runner, config=config, issue_number=issue_number
                    ).comments
                except AgentLoopError:
                    plan_fresh_records = None
                else:
                    try:
                        plan_fresh_records = _extract_round_metadata_records(
                            plan_fresh_comments, flow="plan"
                        )
                    except AgentLoopError as exc:
                        stop_plan_pre_panel(
                            plan_undecodable_history_message(exc), round_number=round_number
                        )
                        raise
                plan_visibility_records, _plan_checkpoint_present = _visibility_snapshot(
                    fresh_records=plan_fresh_records,
                    base_records=plan_records,
                    base_length=len(issue_context.comments),
                    checkpoint=plan_posted_checkpoint,
                )
                plan_visible_names = visible_peer_names(
                    plan_visibility_records,
                    opening_source=_derive_plan_visibility_evidence(
                        plan_visibility_records
                    ).opening_source,
                    flow="plan", round_number=round_number, subject=current_plan_subject,
                    reviewer_names=[agent_display_name(r) for r in configured_reviewers],
                    primary_reviewer=plan_primary_name,
                    launch_phase=plan_scheduler_decision.phase,
                    launching=plan_recovery_context.launching,
                    checkpoint_index=latest_round_checkpoint_index(
                        plan_visibility_records, flow="plan",
                        round_number=round_number, subject=current_plan_subject,
                    ),
                )
                plan_round_public_peers = tuple(
                    reviewer for reviewer in configured_reviewers
                    if reviewer in plan_today_peers
                    or agent_display_name(reviewer) in plan_visible_names
                )
        def _plan_round_recovery() -> PartialRoundRecovery:
            return compute_partial_round_recovery(
                snapshot=issue_context.comments,
                read_rest=lambda: read_rest_issue_comments(
                    runner, config=config, issue_number=issue_number,
                    purpose="the partial review round's comment ids cannot be listed",
                    reject_empty_output=True,
                ),
                flow="plan", round_number=round_number, subject=current_plan_subject,
                context=plan_recovery_context,
                reviewer_names=[agent_display_name(r) for r in configured_reviewers],
                resume=lambda remaining: _resume_plan_round(
                    remaining, configured_reviewers=configured_reviewers
                ),
                # The resumed plan subject must not shift after deletion.
                fingerprint=lambda remaining: (
                    lambda resumed: resumed[0] if resumed is not None else None
                )(_resume_plan_round(remaining, configured_reviewers=configured_reviewers)),
            )

        if not plan_round_parallel:
            _refuse_partial_round_before_sequential_turns(
                spool=plan_round_spool,
                fresh_turn_reviewers=[
                    reviewer for reviewer in round_reviewers
                    if resumed_by_name.get(agent_display_name(reviewer)) is None
                ],
                public_peers=plan_round_public_peers,
                recovery=_plan_round_recovery,
            )
        if plan_round_parallel:
            pending_plan_reviewers = [
                reviewer for reviewer in round_reviewers
                if resumed_by_name.get(agent_display_name(reviewer)) is None
            ]
            if pending_plan_reviewers:
                plan_prompts = {
                    reviewer: _build_plan_review_prompt(reviewer) for reviewer in pending_plan_reviewers
                }
                pending_plan_names = [agent_display_name(reviewer) for reviewer in pending_plan_reviewers]
                log(
                    config,
                    f"Planning round {round_number}: invoking {', '.join(pending_plan_names)} "
                    f"in parallel on issue #{issue_number}",
                )

                def _plan_reviewer_worker(reviewer: AgentName) -> _ReviewerTurnResult:
                    reviewer_name = agent_display_name(reviewer)
                    try:
                        response = _run_validated_agent(
                            runner,
                            agent=reviewer,
                            config=config,
                            prompt=plan_prompts[reviewer],
                            session_id=reviewer_session_ids.get(reviewer),
                            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                            **_plan_review_validators(reviewer_name),
                            usage_context=usage_context,
                            use_repair=True,
                            repair_expected_kind="plan_review",
                            repair_reviewer_requirement_ids=_surfaced_reviewer_requirement_ids(
                                issue_context.human_requirements,
                                requirement_scope="planning requirements",
                            ),
                            repair_allowed_prior_item_ids=tuple(
                                item.item_id for item in prior_unresolved_items
                            ),
                            ledger_incomplete=round_ledger_incomplete,
                            repair_resolved_history_item_ids=round_resolved_history_item_ids,
                            role="reviewer",
                            operation_description="plan review",
                            reask_on_missing_judgement_field=True,
                        )
                    except AgentLoopError as exc:
                        # Includes QuotaResetExceededError: captured here and
                        # re-raised on the main thread with priority.
                        return _ReviewerTurnResult(reviewer_name=reviewer_name, error=exc)
                    return _ReviewerTurnResult(reviewer_name=reviewer_name, response=response)

                def _plan_publication_parsed(turn: _ReviewerTurnResult) -> ParsedPlanReview | None:
                    if turn.error is not None or turn.response is None:
                        return None
                    parsed = turn.response.marker_value
                    assert isinstance(parsed, ParsedPlanReview)
                    return dataclasses_replace(
                        parsed,
                        items=_drop_repeated_carried_plan_future_followups(
                            parsed.items, prior_items=prior_unresolved_items,
                            dispositions=parsed.dispositions,
                        ),
                    )

                def _plan_retry_bound(reviewer: AgentName, turn: _ReviewerTurnResult) -> AgentLoopError | None:
                    """Every plan-review failure or incomplete review is retried by a rerun."""
                    if turn.error is not None:
                        return turn.error
                    parsed = _plan_publication_parsed(turn)
                    if parsed is not None and _is_incomplete_plan_review(parsed):
                        return _incomplete_plan_review_error(agent_display_name(reviewer))
                    return None

                def _publish_plan_completion(reviewer: AgentName, turn: _ReviewerTurnResult) -> bool:
                    """Publish a validated reviewer response without mutating round state.

                    Numbering and ledger mutations stay below the settlement barrier.
                    The raw validated response in metadata makes this checkpoint
                    resumable even though its ``new_items`` are provisional.
                    """
                    parsed = _plan_publication_parsed(turn)
                    if parsed is None or _is_incomplete_plan_review(parsed):
                        return False
                    assert turn.response is not None
                    reviewer_name = agent_display_name(reviewer)
                    _post_plan_reviewer_comment(
                        reviewer_name, parsed, review_output=turn.response.text,
                        model_used=turn.response.model_used,
                        identity=turn.response,
                        acquisition_outcome=turn.response.acquisition_outcome,
                        acquisition_returncode=turn.response.acquisition_returncode,
                        phase="publication",
                    )
                    early_published_plan_reviewers.add(reviewer)
                    return True

                plan_turn_results = _launch_reviewer_turns(
                    runner,
                    pending_plan_reviewers,
                    thread_name_prefix=f"plan-review-r{round_number}",
                    run_turn=_plan_reviewer_worker,
                    on_completion=_publish_plan_completion,
                    spool=plan_round_spool,
                    replay_turn=lambda reviewer, fields: _replay_spooled_review(
                        runner, config=config, reviewer=reviewer, fields=fields,
                        validators=_plan_review_validators(agent_display_name(reviewer)),
                    ),
                    public_peers=plan_round_public_peers,
                    recovery=_plan_round_recovery,
                    retry_bound=_plan_retry_bound,
                    max_workers=None if config.review_parallel else 1,
                    configured_order=round_reviewers,
                    config=config,
                )

        for reviewer in round_reviewers:
            reviewer_name = agent_display_name(reviewer)
            resumed_record = resumed_by_name.get(reviewer_name)
            if resumed_record is not None:
                review_output = resumed_record.metadata.canonical_reviewer_response or resumed_record.body
                review_model_used = resumed_record.metadata.model_used
                review_acquisition_outcome = resumed_record.metadata.acquisition_outcome
                review_acquisition_returncode = resumed_record.metadata.acquisition_returncode
                structured_review = parse_structured_plan_review(
                    review_output,
                    architecture_status_mode="legacy",
                    reviewer=reviewer_name,
                    surfaced_requirement_ids=_surfaced_reviewer_requirement_ids(
                        issue_context.human_requirements,
                        requirement_scope="planning requirements",
                    ),
                )
                resumed_impact, resumed_records = _resumed_review_architecture(
                    resumed_record.metadata,
                    structured_review.architecture_impact if structured_review is not None else None,
                )
                parsed_review = ParsedPlanReview(
                    state=resumed_record.metadata.state or parse_plan_state(review_output),
                    summary=(
                        structured_review.summary
                        if structured_review is not None
                        else review_freeform_summary_text(review_output)
                    ),
                    items=(
                        structured_review.items
                        if structured_review is not None
                        else parse_plan_review_items(review_output, reviewer=reviewer_name)
                    ),
                    dispositions=resumed_record.metadata.dispositions,
                    architecture_impact=resumed_impact,
                    architecture_impact_degradations=resumed_records,
                )
                review_state = parsed_review.state
                log(config, f"Planning round {round_number}: resuming {reviewer_name}'s completed review")
                reviewer_new_unresolved_items = list(resumed_record.metadata.new_items)
            elif plan_round_parallel:
                turn = plan_turn_results[reviewer]
                if turn.error is not None:
                    category = getattr(turn.error, "failure_category", None) or "error"
                    log(
                        config,
                        f"Planning round {round_number}: {reviewer_name} failed ({category}); "
                        "will raise after the remaining reviewers in this round are applied",
                    )
                    plan_fatal_errors.append((reviewer_name, turn.error))
                    continue
                review_response = turn.response
                assert review_response is not None
                review_output = review_response.text
                review_model_used = review_response.model_used
                review_acquisition_outcome = review_response.acquisition_outcome
                review_acquisition_returncode = review_response.acquisition_returncode
                reviewer_session_ids[reviewer] = review_response.session_id
                parsed_review = review_response.marker_value
                assert isinstance(parsed_review, ParsedPlanReview)
                parsed_review = dataclasses_replace(
                    parsed_review,
                    items=_drop_repeated_carried_plan_future_followups(
                        parsed_review.items,
                        prior_items=prior_unresolved_items,
                        dispositions=parsed_review.dispositions,
                    ),
                )
                review_state = parsed_review.state
                reviewer_new_unresolved_items = []
            else:
                log(
                    config,
                    f"Planning round {round_number}: {reviewer_name} reviewing issue #{issue_number} "
                    f"(context mode: {context_mode}{context_reason})",
                )
                sequential_plan_validators = _architecture_mode_validators(lambda mode: lambda text, reviewer_name=reviewer_name, items=prior_unresolved_items: _validate_plan_review_response(
                    text,
                    reviewer=reviewer_name,
                    unresolved_items=items,
                    current_round_items=round_new_unresolved_items,
                    surfaced_requirement_ids=_surfaced_reviewer_requirement_ids(
                        issue_context.human_requirements,
                        requirement_scope="planning requirements",
                    ), architecture_status_mode=mode,
                ))
                review_response = _same_round_replay_or_invoke(
                    runner,
                    config=config,
                    reviewer=reviewer,
                    spool=plan_round_spool,
                    public_peers=plan_round_public_peers,
                    recovery=_plan_round_recovery,
                    validators=sequential_plan_validators,
                    invoke=lambda reviewer=reviewer, validators=sequential_plan_validators: _run_validated_agent(
                        runner,
                        agent=reviewer,
                        config=config,
                        prompt=_build_plan_review_prompt(reviewer),
                        session_id=reviewer_session_ids.get(reviewer),
                        marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                        **validators,
                        usage_context=usage_context,
                        use_repair=True,
                        repair_expected_kind="plan_review",
                        repair_reviewer_requirement_ids=_surfaced_reviewer_requirement_ids(
                            issue_context.human_requirements,
                            requirement_scope="planning requirements",
                        ),
                        repair_allowed_prior_item_ids=tuple(item.item_id for item in prior_unresolved_items),
                        ledger_incomplete=round_ledger_incomplete,
                        repair_resolved_history_item_ids=round_resolved_history_item_ids,
                        role="reviewer",
                        operation_description="plan review",
                        reask_on_missing_judgement_field=True,
                    ),
                )
                review_output = review_response.text
                review_model_used = review_response.model_used
                review_acquisition_outcome = review_response.acquisition_outcome
                review_acquisition_returncode = review_response.acquisition_returncode
                reviewer_session_ids[reviewer] = review_response.session_id
                parsed_review = review_response.marker_value
                assert isinstance(parsed_review, ParsedPlanReview)
                parsed_review = dataclasses_replace(
                    parsed_review,
                    items=_drop_repeated_carried_plan_future_followups(
                        parsed_review.items,
                        prior_items=prior_unresolved_items,
                        dispositions=parsed_review.dispositions,
                    ),
                )
                review_state = parsed_review.state
                reviewer_new_unresolved_items = []

            if _is_incomplete_plan_review(parsed_review):
                log(
                    config,
                    f"Planning round {round_number}: {reviewer_name} did not complete its plan review "
                    "and reported no actionable blocking plan issues or Same-Plan follow-ups; "
                    "stopping without a coder follow-up",
                )
                incomplete_review_error = _incomplete_plan_review_error(reviewer_name)
                if plan_round_parallel:
                    plan_fatal_errors.append((reviewer_name, incomplete_review_error))
                    continue
                raise incomplete_review_error

            log(
                config,
                "Planning round "
                f"{round_number}: {reviewer_name} outcome is {_describe_plan_review_outcome(parsed_review)}",
            )
            round_review_models[reviewer_name] = review_model_used
            for disposition in parsed_review.dispositions:
                _record_prior_item_disposition(
                    prior_dispositions,
                    disposition,
                    flow="plan",
                    round_number=round_number,
                    subject=current_plan_subject,
                    reviewer_name=reviewer_name,
                )
            if review_state == "blocking":
                all_approved = False
                blocking_reviews.append((reviewer_name, review_output))
            else:
                approved_review_outputs.append((reviewer_name, review_output))
                accepted_review_carriers[reviewer_name] = parsed_review
            if resumed_record is None or (
                resumed_record.metadata.phase == "publication"
                and not (current_resume is not None and current_resume.reconciled)
            ):
                for item in parsed_review.items.blocking:
                    tracked_item = _next_unresolved_item(
                        item_number=next_unresolved_item_number,
                        reviewer=item.reviewer,
                        source_round=round_number,
                        text=item.text,
                        status="blocking",
                    )
                    round_new_unresolved_items.append(tracked_item)
                    reviewer_new_unresolved_items.append(tracked_item)
                    next_unresolved_item_number += 1
                for item in parsed_review.items.same_plan:
                    tracked_item = _next_unresolved_item(
                        item_number=next_unresolved_item_number,
                        reviewer=item.reviewer,
                        source_round=round_number,
                        text=item.text,
                        status="same-plan",
                    )
                    round_new_unresolved_items.append(tracked_item)
                    reviewer_new_unresolved_items.append(tracked_item)
                    next_unresolved_item_number += 1
                for item in parsed_review.items.future:
                    tracked_item = _next_unresolved_item(
                        item_number=next_unresolved_item_number,
                        reviewer=item.reviewer,
                        source_round=round_number,
                        text=item.text,
                        status="future",
                    )
                    round_new_unresolved_items.append(tracked_item)
                    reviewer_new_unresolved_items.append(tracked_item)
                    next_unresolved_item_number += 1
                if reviewer not in early_published_plan_reviewers:
                    _post_plan_reviewer_comment(
                        reviewer_name, parsed_review, review_output=review_output,
                        model_used=review_model_used,
                        identity=(review_response if resumed_record is None else None),
                        acquisition_outcome=review_acquisition_outcome,
                        acquisition_returncode=review_acquisition_returncode,
                        new_items=tuple(reviewer_new_unresolved_items),
                    )
            else:
                round_new_unresolved_items.extend(reviewer_new_unresolved_items)

        if not plan_fatal_errors:
            # Every same-round outcome is now published; a sequential resume
            # that replayed withheld reviews no longer needs them.
            plan_round_spool.discard()
        same_model_note = same_model_panel_note(round_review_models)
        if same_model_note is not None:
            log(config, f"Planning round {round_number}: {same_model_note}")
        if (
            plan_round_parallel or same_model_note is not None
        ) and not (current_resume is not None and current_resume.reconciled):
            settled = ", ".join(agent_display_name(reviewer) for reviewer in round_reviewers)
            post_issue_comment(
                runner, config=config, issue_number=issue_number,
                body=_attach_round_metadata(
                    f"Plan review round {round_number} reconciliation: settled reviewers: {settled or 'none'}. "
                    f"Finalization {'stops' if plan_fatal_errors else 'continues'} after reconciliation."
                    + (f" {same_model_note}." if same_model_note else ""),
                    PostedRoundMetadata(
                        flow="plan", role="summary", agent="Orchestrator", round_number=round_number,
                        subject=_plan_subject(current_plan), prior_items=prior_unresolved_items,
                        dispositions=tuple(
                            disposition for values in prior_dispositions.values() for disposition in values
                        ), new_items=tuple(round_new_unresolved_items), phase="reconciliation",
                        **_architecture_metadata_fields(config),
                    ),
                ),
            )

        if plan_fatal_errors:
            # Every healthy reviewer above was already applied (comment
            # posted, items numbered) in configured order, so a rerun resumes
            # them instead of re-invoking (#594). Raise only now: quota resets
            # take priority, otherwise the first configured-order failure.
            for _reviewer_name, error in plan_fatal_errors:
                if isinstance(error, QuotaResetExceededError):
                    raise error
            raise plan_fatal_errors[0][1]

        unresolved_items, _ = _apply_unresolved_item_dispositions(
            plan_ledger_view(prior_unresolved_items),
            prior_dispositions,
            same_status="same-plan",
            retain_future=True,
        )
        compact_prior_summaries = list(
            bound_compact_prior_summaries(
                [
                    *compact_prior_summaries,
                    *_collect_prior_compact_summaries(
                        plan_ledger_view(prior_unresolved_items),
                        unresolved_items,
                        prior_dispositions,
                    ),
                ]
            )
        )
        # The ledger handed to the next round is built from the amended view,
        # so from the next round on persisted ``prior_items`` carry explicit
        # ownership that excludes removed reviewers (#943).
        unresolved_items = list(
            plan_ledger_view([*unresolved_items, *round_new_unresolved_items])
        )
        plan_finding_history.observe_reconciled(unresolved_items)
        # Items minted and cleared inside one round never reappear as prior
        # items, so record them here too.
        plan_accounted_item_ids.update(item.item_id for item in unresolved_items)
        plan_accounted_item_ids.update(
            item.item_id for item in round_new_unresolved_items
        )
        must_fix_items = [item for item in unresolved_items if item.status in {"blocking", "same-plan"}]
        if all_approved and not must_fix_items and issue_context.human_requirements:
            hr_ids = _surfaced_reviewer_requirement_ids(
                issue_context.human_requirements,
                requirement_scope="planning requirements",
            )
            coder_dispositions_complete = _current_plan_has_complete_human_requirement_dispositions(
                current_coder_output,
                surfaced_requirement_ids=hr_ids,
                inherited_dispositions=(
                    hydrate_authenticated_plan_state(current_plan_sidecar).plan.human_requirement_dispositions
                    if current_response_form == "semantic-patch-v1"
                    and current_plan_sidecar is not None
                    else None
                ),
            )
            missing_acknowledgements = (
                [reviewer_name for reviewer_name, _review_output in approved_review_outputs]
                if not coder_dispositions_complete
                else [
                    reviewer_name
                    for reviewer_name, review_output in approved_review_outputs
                    if not human_requirements_resolved(review_output)
                ]
            )
            if missing_acknowledgements:
                if not coder_dispositions_complete:
                    log(
                        config,
                        f"Planning round {round_number}: current coder plan lacks complete "
                        "human requirement dispositions; re-injecting as blocking plan item",
                    )
                    synthetic_review = (
                        "Orchestrator plan review:\n\n"
                        "The current canonical coder plan lacks complete structured dispositions "
                        "for the signed human requirements. Coder must provide one valid "
                        "disposition with evidence for every surfaced requirement before a "
                        "reviewer approval marker can be accepted."
                    )
                    blocking_reviews.append(("Orchestrator", synthetic_review))
                    round_new_unresolved_items.append(
                        _next_unresolved_item(
                            item_number=next_unresolved_item_number,
                            reviewer="Orchestrator",
                            source_round=round_number,
                            text=(
                                "The current canonical coder plan lacks complete structured "
                                "dispositions for the signed human requirements. Coder must provide "
                                "one valid disposition with evidence for every surfaced requirement "
                                "before a reviewer approval marker can be accepted."
                            ),
                            status="blocking",
                        )
                    )
                    next_unresolved_item_number += 1
                    unresolved_items = [*unresolved_items, round_new_unresolved_items[-1]]
                    must_fix_items = [
                        item for item in unresolved_items if item.status in {"blocking", "same-plan"}
                    ]
                    all_approved = False
                    # A reviewer cannot repair a missing coder attestation.
                    missing_acknowledgements = []
                still_missing = []
                repaired_plan_approvals: dict[str, str] = {}
                for reviewer_name, review_output in approved_review_outputs:
                    if human_requirements_resolved(review_output):
                        continue
                    log(
                        config,
                        f"Planning round {round_number}: {reviewer_name} approved without "
                        "HUMAN_REQUIREMENTS_RESOLVED; attempting repair",
                    )
                    repaired_text, repaired_validated, repair_attempts = _run_structured_repair(
                        review_output,
                        runner=runner,
                        config=config,
                        usage_context=usage_context,
                        validate=lambda candidate, reviewer_name=reviewer_name: _validate_plan_review_response(
                            candidate,
                            reviewer=reviewer_name,
                            unresolved_items=prior_unresolved_items,
                            current_round_items=round_new_unresolved_items,
                            surfaced_requirement_ids=hr_ids, architecture_status_mode="strict",
                        ),
                        repair_kwargs={
                            "expected_kind": "plan_review",
                            "reviewer_requirement_ids": hr_ids,
                            "allowed_prior_item_ids": tuple(
                                item.item_id for item in prior_unresolved_items
                            ),
                        },
                        forbid_architecture_impact=_acknowledgement_repair_forbids_assessment(
                            review_output
                        ),
                    )
                    repaired_validated = _pin_acknowledgement_repair(
                        accepted_review_carriers.get(reviewer_name),
                        repaired_validated,
                        config=config,
                        reviewer_name=reviewer_name,
                    )
                    _log_repair_attempts(
                        config, f"Planning round {round_number}: {reviewer_name}", repair_attempts
                    )
                    if repaired_validated is not None:
                        repaired_parsed = repaired_validated
                        if (
                            repaired_parsed.state == "approved"
                            and human_requirements_resolved(repaired_text)
                        ):
                            log(
                                config,
                                f"Planning round {round_number}: repair recovered "
                                f"HUMAN_REQUIREMENTS_RESOLVED for {reviewer_name}",
                            )
                            repaired_plan_approvals[reviewer_name] = repaired_text
                            if staged_planning:
                                # The record posted before the repair stores no
                                # surfaced requirement IDs, because the original
                                # text carried no acknowledgement.  Without an
                                # amended record the next round would reject this
                                # approval as unacknowledged and re-invoke the
                                # reviewer instead of advancing the phase, so the
                                # repaired result is persisted here; a later
                                # record for the same reviewer supersedes the
                                # earlier one (#905).
                                log(
                                    config,
                                    f"Planning round {round_number}: persisting the repaired "
                                    f"{reviewer_name} plan approval for the exact-plan carry",
                                )
                                _post_plan_reviewer_comment(
                                    reviewer_name,
                                    repaired_parsed,
                                    review_output=repaired_text,
                                    model_used=None,
                                )
                            continue
                        if repaired_parsed.state == "blocking":
                            log(
                                config,
                                f"Planning round {round_number}: repair returned blocking for "
                                f"{reviewer_name}; treating as reviewer blocking",
                            )
                            blocking_reviews.append((reviewer_name, repaired_text))
                            for item in repaired_parsed.items.blocking:
                                new_item = _next_unresolved_item(
                                    item_number=next_unresolved_item_number,
                                    reviewer=item.reviewer,
                                    source_round=round_number,
                                    text=item.text,
                                    status="blocking",
                                )
                                round_new_unresolved_items.append(new_item)
                                unresolved_items = [*unresolved_items, new_item]
                                next_unresolved_item_number += 1
                            for item in repaired_parsed.items.same_plan:
                                new_item = _next_unresolved_item(
                                    item_number=next_unresolved_item_number,
                                    reviewer=item.reviewer,
                                    source_round=round_number,
                                    text=item.text,
                                    status="same-plan",
                                )
                                round_new_unresolved_items.append(new_item)
                                unresolved_items = [*unresolved_items, new_item]
                                next_unresolved_item_number += 1
                            all_approved = False
                            must_fix_items = [
                                item for item in unresolved_items
                                if item.status in {"blocking", "same-plan"}
                            ]
                            continue
                    still_missing.append(reviewer_name)
                if still_missing:
                    log(
                        config,
                        f"Planning round {round_number}: reviewer(s) {', '.join(still_missing)} "
                        "approved without acknowledging signed human requirements; "
                        "re-injecting as blocking plan item",
                    )
                    synthetic_review = (
                        "Orchestrator plan review:\n\n"
                        f"Reviewer(s) {', '.join(still_missing)} approved without "
                        "acknowledging the signed human requirements. Coder must address the "
                        "human requirements and ensure the reviewer explicitly resolves them "
                        "before plan approval."
                    )
                    blocking_reviews.append(("Orchestrator", synthetic_review))
                    round_new_unresolved_items.append(
                        _next_unresolved_item(
                            item_number=next_unresolved_item_number,
                            reviewer="Orchestrator",
                            source_round=round_number,
                            text=(
                                f"Reviewer(s) {', '.join(still_missing)} approved without "
                                "acknowledging the signed human requirements. Coder must address the "
                                "human requirements and ensure the reviewer explicitly resolves them "
                                "before plan approval."
                            ),
                            status="blocking",
                        )
                    )
                    next_unresolved_item_number += 1
                    unresolved_items = [*unresolved_items, round_new_unresolved_items[-1]]
                    must_fix_items = [
                        item for item in unresolved_items if item.status in {"blocking", "same-plan"}
                    ]
                    all_approved = False
                if repaired_plan_approvals:
                    # Later gates and the approval carry must read the repaired
                    # text, not the output that lacked the acknowledgement.
                    approved_review_outputs = [
                        (name, repaired_plan_approvals.get(name, output))
                        for name, output in approved_review_outputs
                    ]

        # A machine obligation has no planning clearance path: reviewer
        # dispositions are evidence only, so after a unanimous approval it
        # would drive another revision every round forever (#1005).  Recovery
        # demotes only the recognized legacy promotion; a malformed or
        # unrecognized machine record stays fail-closed here, stopping with a
        # diagnostic naming the items instead of revising again.
        unclearable_plan_items = [item for item in must_fix_items if item.is_machine_obligation]
        if all_approved and unclearable_plan_items:
            raise AgentLoopError(
                f"Planning round {round_number}: every reviewer approved, but plan item(s) "
                + ", ".join(
                    f"{item.item_id} (owner {item.reviewer or 'unknown'}, kind "
                    f"{item.obligation_kind or 'unknown'})"
                    for item in unclearable_plan_items
                )
                + " are machine obligations that no planning participant can clear. "
                "Stopping instead of revising an approved plan indefinitely; inspect the "
                "item's round metadata and rerun."
            )
        # The final gate evaluates every required plan reviewer, carried and
        # current alike.  A paused reviewer counts only through a qualifying
        # exact-key carried approval; a reviewer that blocked this round loses
        # any carry it held.
        plan_missing_approvals: tuple[str, ...] = ()
        if staged_planning:
            blocked_this_round = {name for name, _output in blocking_reviews}
            approved_this_round = {name for name, _output in approved_review_outputs}
            plan_round_qualifying = (
                set(plan_qualifying_approvals) | approved_this_round
            ) - blocked_this_round
            plan_missing_approvals = tuple(
                name for name in plan_reviewer_names if name not in plan_round_qualifying
            )
        plan_phase_advance_pending = bool(
            staged_planning
            and all_approved
            and not must_fix_items
            and plan_missing_approvals
        )
        growth_guard_revision = None
        if plan_growth_notice is not None and all_approved and not must_fix_items:
            # A non-compliant candidate can never pass, so neither a
            # reviewer-only panel advance nor an approval is spent on it: the
            # next round is a planner revision carrying the growth notice.
            if plan_phase_advance_pending:
                log(
                    config,
                    f"Planning round {round_number}: skipping the reviewer-only plan phase "
                    "advance; the candidate fails the plan-growth gate",
                )
            plan_phase_advance_pending = False
            growth_guard_revision = plan_growth_notice
        # --plan-narrow-staged (#1268): an orchestrator-owned revision
        # obligation, like the growth notice.  It blocks approval and any
        # reviewer-only panel advance, and is delivered once per invocation.
        narrow_revision: str | None = None
        if staged_mode_revision:
            narrow_revision = (
                "Operator directive (orchestrator, not a reviewer finding): this "
                f"invocation's execution mode is {config.plan_execution_mode} and cannot "
                "deliver a staged plan. Revise to a one-shot plan whose single "
                "deliverable is complete and useful on its own and fits one PR. Move "
                "independent remaining scope into the typed `deferred_work` category; "
                "that category is recorded only and is never materialized, so the "
                "operator must file any follow-up issues for it manually."
            )
            if plan_phase_advance_pending:
                log(
                    config,
                    f"Planning round {round_number}: skipping the reviewer-only plan phase "
                    "advance; the staged candidate must be narrowed (--plan-narrow-staged)",
                )
            plan_phase_advance_pending = False
        # The outstanding phase, not the one that just ran: both the durable
        # phase-advance record and the round-budget diagnostic must name the
        # round that is still pending.
        plan_outstanding_phase = (
            _outstanding_plan_phase(
                plan_snapshot,
                decision=plan_scheduler_decision,
                current_key=current_plan_key,
                obligations=_scheduler_obligations(
                    unresolved_items,
                    required_reviewers=plan_reviewer_names,
                    active_statuses=frozenset({"blocking", "same-plan"}),
                ),
                qualifying_approvals=tuple(sorted(plan_round_qualifying)),
                panel_evidence=(
                    plan_panel_evidence.opened
                    or (
                        plan_scheduler_decision is not None
                        and plan_scheduler_decision.records_panel_opening
                    )
                ),
                force_full=plan_automatic_force_full,
                force_full_source=(
                    "automatic" if plan_automatic_force_full else None
                ),
            )
            if plan_phase_advance_pending
            else None
        )

        inherited_guard_revision = None
        # The signed re-plan has not produced a revision yet: the current plan
        # is still the one the human authorized replacing.
        supersession_revision_pending = (
            plan_supersession is not None
            and approved_plan_hash(current_plan) == plan_supersession.superseded_hash
        )
        if (
            all_approved
            and not must_fix_items
            and not plan_missing_approvals
            and inherited_review_failure is not None
        ):
            # Only a historical candidate published before the planning-time
            # check can reach this guard.  It is revised through the enforced
            # replan path instead of dead-ending the run (#936).
            inadmissible_hash = approved_plan_hash(current_plan)
            if round_number == config.max_rounds:
                raise AgentLoopError(
                    f"Approved child plan {inadmissible_hash} on issue #{issue_number} is "
                    "inadmissible under the inherited-matrix contract and must be revised, but "
                    f"the planning round budget ({config.max_rounds}) is exhausted. Raise "
                    "--max-rounds and rerun; re-planning continues the existing round "
                    f"numbering.\n{inherited_review_failure}"
                )
            audit_line = _inadmissible_plan_audit_line(inadmissible_hash)
            if not any(
                isinstance(getattr(comment, "body", None), str)
                and audit_line in comment.body
                for comment in issue_context.comments
            ):
                # Plain audit text only: no round metadata a resume could read.
                post_issue_comment(
                    runner,
                    config=config,
                    issue_number=issue_number,
                    body=(
                        f"{audit_line}\n\nNo reviewer reported a blocker; the orchestrator's "
                        "mechanical inherited-row check rejected the approved plan. The planner "
                        "revises it next, the revision is rechecked before publication, and the "
                        "complete reviewer board reviews the result.\n\n"
                        f"{inherited_review_failure}\n-- Orchestrator"
                    ),
                )
            inherited_guard_revision = inherited_review_failure
        elif (
            all_approved
            and not must_fix_items
            and not plan_missing_approvals
            and supersession_revision_pending
        ):
            # An admissible plan under a signed supersession (#985): reviewer
            # approval of the superseded plan is not approval of its
            # replacement, so the authorized revision runs instead of
            # approving and rebinding the plan the human asked to replace.
            if round_number == config.max_rounds:
                raise AgentLoopError(
                    f"Approved child plan {plan_supersession.superseded_hash} on issue "
                    f"#{issue_number} must be re-planned under its signed supersession, but "
                    f"the planning round budget ({config.max_rounds}) is exhausted. Raise "
                    "--max-rounds and rerun; re-planning continues the existing round "
                    "numbering."
                )
        elif narrow_revision is not None and growth_guard_revision is None:
            if all_approved and not must_fix_items:
                log(
                    config,
                    f"Planning round {round_number}: reviewers approved, but the staged plan "
                    f"cannot be delivered under {config.plan_execution_mode}; starting a "
                    "narrowing revision (--plan-narrow-staged)",
                )
        elif growth_guard_revision is not None:
            log(
                config,
                f"Planning round {round_number}: reviewers approved, but the plan fails the "
                f"plan-growth gate ({plan_growth_violation}); starting a planner revision",
            )
        elif all_approved and not must_fix_items and not plan_missing_approvals:
            if plan_amendment_note:
                log(
                    config,
                    f"Planning issue #{issue_number}: plan approved on a reduced board. "
                    f"{plan_amendment_note}",
                )
            # Re-read both sides at the approval-to-implementation boundary so
            # a human instruction posted during planning cannot be hidden by
            # the original snapshot. New signed IDs require a fresh planning
            # acknowledgement rather than being silently folded into the plan.
            refreshed_issue_context = get_issue_context(
                runner, config=config, issue_number=issue_number
            )
            previous_requirement_ids = {
                requirement.requirement_id for requirement in issue_context.human_requirements
            }
            refreshed_requirement_ids = {
                requirement.requirement_id
                for requirement in refreshed_issue_context.human_requirements
            }
            if refreshed_requirement_ids != previous_requirement_ids:
                # Any set inequality is a changed requirement set, not only an
                # addition.  A withdrawal is equally disqualifying: every plan
                # review and every carried exact-key approval was bound to the
                # earlier surfaced requirement digest, so a withdrawn ID leaves
                # the approvals acknowledging a requirement that no longer
                # exists (#905, from #841).
                raise AgentLoopError(
                    f"Issue #{issue_number} signed human requirement(s) changed after plan "
                    "approval: "
                    f"{_describe_requirement_set_change(previous_requirement_ids, refreshed_requirement_ids)}. "
                    "Re-run planning so the current signed requirements receive explicit "
                    "acknowledgement."
                )
            issue_context = refreshed_issue_context
            if parent_issue_context is not None:
                previous_parent_requirement_ids = {
                    requirement.requirement_id
                    for requirement in parent_issue_context.human_requirements
                }
                refreshed_parent_context = get_issue_context(
                    runner, config=config, issue_number=parent_issue_context.number
                )
                refreshed_parent_requirement_ids = {
                    requirement.requirement_id
                    for requirement in refreshed_parent_context.human_requirements
                }
                if refreshed_parent_requirement_ids != previous_parent_requirement_ids:
                    raise AgentLoopError(
                        f"Authoritative parent issue #{parent_issue_context.number} signed human "
                        "requirement(s) changed after plan approval: "
                        f"{_describe_requirement_set_change(previous_parent_requirement_ids, refreshed_parent_requirement_ids)}"
                        ". Re-run planning so the current signed parent requirements receive "
                        "explicit acknowledgement."
                    )
                parent_issue_context = refreshed_parent_context
            approved_future_followup_sources = [
                _plan_followup_source_from_unresolved_item(item)
                for item in unresolved_items
                if item.status == "future"
            ]
            plan_hash = approved_plan_hash(current_plan)
            plan_subject = _plan_subject(current_plan)
            # The gate's own judgement of the candidate it just let through,
            # recorded in the handoff so resume re-validates against it (#1074).
            approval_growth_verdict = plan_growth_approval_verdict(
                config,
                current_plan_sidecar.canonical_json if current_plan_sidecar is not None else None,
                plan_growth_assessment,
            )
            approved_plan_context = make_approved_plan_context(
                current_plan,
                source_locator=f"issue #{issue_number} approved-plan round",
                expected_hash=plan_hash,
                expected_subject=plan_subject,
            )
            if approved_plan_context.matrix_available:
                # Cross the approval boundary through the same immutable
                # validator used by recovery. There is no prior approved
                # baseline to diff on a first approval, so comparing the
                # accepted payload with itself specifically asserts that the
                # boundary cannot mutate it while it is being bound.
                validate_risk_test_matrix_revision(
                    approved_plan_context.risk_test_matrix_payload or {},
                    approved_plan_context.risk_test_matrix_payload or {},
                    approved_plan_context.risk_test_matrix_changes_payload,
                    approved=True,
                )
            recommendation = _current_execution_recommendation(
                current_plan, issue_context.comments
            )
            resolved_execution = _resolve_execution_policy(
                config,
                requested_policy=requested_policy,
                implement_after_approval=implement_after_approval,
                recommendation=recommendation,
            )
            mode = resolved_execution.action
            canonical_strategy = resolved_execution.strategy
            if plan_supersession is not None:
                # Signed re-plan (#936): the existing PR is rebound to the
                # approved revision.  No fresh implementation turn runs and no
                # second PR is opened.
                if mode == "plan-only":
                    print(
                        f"Issue #{issue_number} re-planned child plan {plan_hash} approved by "
                        f"{format_agent_list(configured_reviewers)}; rerun without plan-only to "
                        f"rebind PR #{plan_supersession.pr_number}."
                    )
                    return 0
                _rebind_superseded_child_plan(
                    runner,
                    config=config,
                    issue_number=issue_number,
                    issue_context=issue_context,
                    plan_supersession=plan_supersession,
                    plan_hash=plan_hash,
                )
                issue_context = get_issue_context(
                    runner, config=config, issue_number=issue_number
                )
                if parent_issue_context is not None:
                    validate_pr_body_does_not_close_issue(
                        runner,
                        config=config,
                        pr_number=plan_supersession.pr_number,
                        issue_number=parent_issue_context.number,
                    )
                return run_pr_loop(
                    runner,
                    pr_number=plan_supersession.pr_number,
                    config=config,
                    issue_context=issue_context,
                    approved_plan_context=approved_plan_context,
                    parent_issue_context=parent_issue_context,
                    usage_context=usage_context,
                    managed_ci_issue_number=issue_number,
                )
            if (
                canonical_strategy == "staged"
                and mode != "plan-only"
                and staged_parent_number is not None
                and _fresh_phase_marker_payload(issue_context) is not None
            ):
                # Nested-topology guard (#808): a planning child whose own
                # reviewed recommendation is staged stops for a human decision
                # before any decision record, checkpoint, summary, child
                # issue, handoff, or coder work.  Hierarchical execution is
                # tracked in #720.
                nested = NestedTopologyDecision(
                    parent_issue=staged_parent_number,
                    child_issue=issue_number,
                    plan_hash=plan_hash,
                    requested_policy=resolved_execution.requested_policy,
                )
                log(config, str(nested))
                print(json.dumps(nested.as_dict(), sort_keys=True))
                return 2
            # A re-approval under a new hash supersedes a decision nothing has
            # acted on yet (#1087); a realized one still fails closed here.
            retired_decision_hashes: frozenset[str] = frozenset()
            retiring_decision_hashes: tuple[str, ...] = ()
            if recommendation is not None and mode != "plan-only":
                (
                    retired_decision_hashes,
                    retiring_decision_hashes,
                ) = _retirable_execution_decision_hashes(
                    runner,
                    config=config,
                    issue_number=issue_number,
                    comments=issue_context.comments,
                    plan_hash=plan_hash,
                )
            normalized_topology = None
            if recommendation is not None and canonical_strategy == "staged":
                normalized_topology = normalize_execution_recommendation(
                    recommendation,
                    approved_plan=current_plan,
                    plan_subject=plan_subject,
                )
            if (
                normalized_topology is not None
                and canonical_strategy == "staged"
                and mode != "plan-only"
            ):
                staged_preflight = _preflight_fresh_staged_topology(
                    runner,
                    issue_number=issue_number,
                    approved_plan=current_plan,
                    config=config,
                    issue_context=issue_context,
                    mode=mode,
                    normalized_topology=normalized_topology,
                    retired_plan_hashes=retired_decision_hashes,
                )
                if isinstance(staged_preflight, NeedsHumanDecision):
                    print(json.dumps(staged_preflight.as_dict(), sort_keys=True))
                    return 2
            elif recommendation is not None and canonical_strategy == "one-shot" and mode != "plan-only":
                _preflight_fresh_one_shot_recovery(
                    runner,
                    issue_number=issue_number,
                    approved_plan=current_plan,
                    config=config,
                    issue_context=issue_context,
                    recommendation=recommendation,
                    retired_plan_hashes=retired_decision_hashes,
                )
            plan_additions = _extract_current_expected_closing_issue_ids(current_plan)
            split_topology = bool(
                mode in {"decompose-only", "implement-by-phase"}
                or config.materialize_split_issues
                or _extract_current_child_stages(current_plan)
                or _extract_current_deferred_stages(current_plan)
            )
            if split_topology and (config.expected_closing_issue_ids or plan_additions):
                raise AgentLoopError(
                    "Additional expected closing issue IDs are single-PR-only and cannot be "
                    "carried through split/decomposition materialization. Invoke the actual "
                    "child issue with a child-scoped --expected-closing-issue declaration."
                )
            if recommendation is not None:
                _print_execution_resolution_summary(
                    issue_number=issue_number,
                    resolved=resolved_execution,
                    normalized_topology=normalized_topology,
                    # The staged implement-by-phase dispatch path is the only
                    # path that prints a progress-derived replacement for this
                    # line; every other path keeps it.
                    defer_child_work=(
                        normalized_topology is not None
                        and mode == "implement-by-phase"
                        and not config.dry_run
                    ),
                )
            if config.dry_run and resolved_execution.is_automatic:
                _print_dry_run_execution_preview(
                    issue_number=issue_number,
                    resolved=resolved_execution,
                    normalized_topology=normalized_topology,
                )
                return 0
            _persist_execution_decision_if_needed(
                runner,
                config=config,
                issue_number=issue_number,
                current_plan=current_plan,
                issue_comments=issue_context.comments,
                recommendation=recommendation,
                requested_policy=resolved_execution.requested_policy,
                resolved_execution=resolved_execution,
                retired_plan_hashes=retired_decision_hashes,
                retires_plan_hashes=retiring_decision_hashes,
            )
            _publish_plan_approved_followups(
                runner,
                config=config,
                issue_number=issue_number,
                approved_plan=current_plan,
                plan_hash=plan_hash,
                plan_subject=plan_subject,
                issue_comments=issue_context.comments,
                sources=approved_future_followup_sources,
                source_context=FollowupSourceContext(
                    repo=config.repo,
                    source_kind="plan",
                    source_number=issue_number,
                    source_identity=plan_hash,
                    parent_issue_numbers=(issue_number,),
                ),
                allow_issue_filing=mode in {"implement-one-shot", "implement-by-phase"},
                usage_context=usage_context,
            )
            split_scope_materialized = _handle_plan_first_split_scope(
                runner,
                issue_number=issue_number,
                config=config,
                current_plan=current_plan,
                plan_subject=plan_subject,
                issue_context=issue_context,
                execution_mode=mode,
                resolved_execution=resolved_execution,
            )
            if isinstance(split_scope_materialized, NeedsHumanDecision):
                print(json.dumps(split_scope_materialized.as_dict(), sort_keys=True))
                return 2
            if split_scope_materialized:
                # Refetch so downstream logic (selected-stage resolution below,
                # decomposition, etc.) sees the `AGENT_DISCUSS_SPLIT` comment
                # this call may have just posted, instead of the stale
                # pre-materialization snapshot (#492 review).
                issue_context = get_issue_context(runner, config=config, issue_number=issue_number)
            if mode == "plan-only":
                print(
                    f"Issue #{issue_number} plan approved by {format_agent_list(configured_reviewers)}."
                    + (f" {plan_amendment_note}" if plan_amendment_note else "")
                )
                deferred_titles = _typed_deferred_work_titles(current_plan_sidecar)
                if deferred_titles:
                    print(
                        "Deferred work recorded only (not tracked or materialized; file "
                        "follow-up issues manually): " + "; ".join(deferred_titles)
                    )
                return 0

            if mode in {"decompose-only", "implement-by-phase"}:
                staged_outcome = _decompose_approved_plan(
                    runner,
                    issue_number=issue_number,
                    approved_plan=current_plan,
                    config=config,
                    memory=memory,
                    issue_context=issue_context,
                    mode=mode,
                    coder_session_id=coder_session_id,
                    usage_context=usage_context,
                    execution_recommendation=recommendation,
                    normalized_topology=normalized_topology,
                )
                if isinstance(staged_outcome, NeedsHumanDecision):
                    print(json.dumps(staged_outcome.as_dict(), sort_keys=True))
                    return 2
                if normalized_topology is not None:
                    final_integration = normalized_topology[0].final_integration_work
                    print(
                        "Final integration work: "
                        f"{final_integration.status}; deliverables: "
                        f"{', '.join(final_integration.deliverables) or 'none'}; "
                        "acceptance criteria: "
                        f"{', '.join(final_integration.acceptance_criteria) or 'none'}; "
                        "covered scope items: "
                        f"{', '.join(final_integration.covered_scope_item_ids) or 'none'}"
                    )
                    if mode == "decompose-only":
                        # `decompose-only` never enters the phase dispatcher,
                        # so it keeps today's topology-only stage listing.
                        remaining_stage_ids = tuple(
                            phase.stage_id or str(phase.position)
                            for phase in normalized_topology[0].phases[1:]
                        )
                        print(
                            "Remaining child work after the first phase: "
                            f"{', '.join(remaining_stage_ids) or 'none'}"
                        )
                if mode == "decompose-only":
                    print(f"Issue #{issue_number} approved plan decomposed into child issues.")
                    return 0
                return _dispatch_current_decomposition_phase(
                    runner,
                    config=config,
                    memory=memory,
                    usage_context=usage_context,
                    issue_number=issue_number,
                    current_plan=current_plan,
                    plan_subject=plan_subject,
                    outcome=staged_outcome,
                    recommendation=recommendation,
                    approved_plan_context=approved_plan_context,
                    issue_context=issue_context,
                    mode=mode,
                    coder_session_id=coder_session_id,
                )

            if mode == "implement-one-shot":
                plan_hash = approved_plan_hash(current_plan)
                plan_subject = _plan_subject(current_plan)

                # Selected-stage handoff (#476): when the parent's split proposals were
                # already fully materialized into child issues (discuss `split` or a prior
                # plan-first run), implementation must target the child the approved plan
                # actually covers instead of treating the whole parent as solved.
                #
                # This does NOT apply when the approved plan itself structurally declares
                # `deferred_stages`: that plan keeps its own primary scope on the parent and
                # only files the *remainder* as children (`_handle_plan_first_split_scope`),
                # so the parent is never one of the materialized children. Skipping stage
                # resolution here also fixes a rerun/resume regression (#492 review): once
                # such a plan's own one-shot handoff is posted on the parent, a later rerun
                # would otherwise see the freshly materialized children, find no stage-handoff
                # marker, and fail to match the plan's own title against a sibling stage's
                # title, raising instead of resuming the already-handed-off parent PR.
                target_issue_number = issue_number
                target_issue_context = issue_context
                staged_parent_issue: int | None = None
                if (
                    staged_parent_number is not None
                    and parent_issue_context is not None
                    and _fresh_phase_marker_payload(issue_context) is not None
                ):
                    # A separately planned decomposition child (#808)
                    # implements its own one-shot recommendation with the
                    # parent recorded as the staged parent.
                    staged_parent_issue = staged_parent_number
                # Explicit child stages are the parent plan's bounded remainder;
                # legacy deferred entries preserve the same historical parent-owned
                # behavior. Dependencies/actions alone must not suppress handoff.
                current_plan_declares_own_deferred_stages = bool(
                    _extract_current_child_stages(current_plan)
                    or _extract_current_deferred_stages(current_plan)
                )
                split_metadata = find_existing_split_materialization(
                    issue_context.comments, parent_issue=issue_number
                )
                if (
                    split_metadata is not None
                    and split_metadata.children
                    and not current_plan_declares_own_deferred_stages
                ):
                    stage_handoff = find_existing_split_stage_handoff(
                        issue_context.comments, parent_issue=issue_number, plan_hash=plan_hash
                    )
                    if stage_handoff is not None:
                        selected_child = next(
                            (
                                child
                                for child in split_metadata.children
                                if child.number == stage_handoff.child_issue_number
                            ),
                            None,
                        )
                        if selected_child is None or selected_child.number is None:
                            raise AgentLoopError(
                                f"Recorded split-stage handoff for issue #{issue_number} references "
                                "a child issue that is no longer in the materialized split metadata; "
                                "manual recovery required."
                            )
                    else:
                        selected_child = resolve_selected_stage_child(
                            split_metadata.children,
                            parent_issue=issue_number,
                            plan_title_or_subject=_plan_first_line(current_plan),
                            split_stage_flag=config.split_stage,
                        )
                        post_split_stage_handoff_comment(
                            runner,
                            config=config,
                            parent_issue=issue_number,
                            plan_hash=plan_hash,
                            child=selected_child,
                        )
                    target_issue_number = selected_child.number
                    target_issue_context = get_issue_context(
                        runner, config=config, issue_number=target_issue_number
                    )
                    staged_parent_issue = issue_number
                    parent_issue_context = get_issue_context(
                        runner, config=config, issue_number=issue_number
                    )

                # Canonical handoff resolution runs before the (older,
                # plan-hash-scoped) one-shot handoff lookup below, so a stale
                # canonical record fails safely instead of being bypassed by
                # it (#589). It is scoped to the selected target issue, since
                # a split-stage plan implements a child, not the parent.
                resolved_pr = resolve_canonical_pr_for_issue(
                    runner,
                    config=config,
                    issue_number=target_issue_number,
                    issue_context=target_issue_context,
                    expected_fallback_scope=IssuePrProvenanceScope(
                        repository=config.repo,
                        issue_number=target_issue_number,
                        flow="approved",
                        approved_plan_hash=plan_hash,
                    ),
                )
                if resolved_pr is not None:
                    if (
                        resolved_pr.source == "canonical"
                        and resolved_pr.metadata is not None
                        and resolved_pr.metadata.flow == "approved-plan-implementation"
                        and resolved_pr.metadata.plan_hash != plan_hash
                    ):
                        raise AgentLoopError(
                            f"Canonical approved-plan handoff for issue #{target_issue_number} "
                            f"points to PR #{resolved_pr.pr_number} with plan hash "
                            f"{resolved_pr.metadata.plan_hash}, but the current approved plan "
                            f"has hash {plan_hash}. Review it with `agent-loop pr "
                            f"{resolved_pr.pr_number}` or remove the stale handoff marker."
                        )
                    log(
                        config,
                        f"Issue #{target_issue_number}: resuming PR #{resolved_pr.pr_number} review for "
                        f"already-handed-off plan (source={resolved_pr.source}, "
                        f"evidence={resolved_pr.evidence_summary})",
                    )
                    if staged_parent_issue is not None:
                        validate_pr_body_does_not_close_issue(
                            runner,
                            config=config,
                            pr_number=resolved_pr.pr_number,
                            issue_number=staged_parent_issue,
                        )
                    if resolved_pr.source == "legacy-closing-reference":
                        resumed_pr_context = get_pr_review_context(
                            runner, config=config, pr_number=resolved_pr.pr_number
                        )
                        pr_url, pr_head_sha = require_pr_metadata_for_handoff(resumed_pr_context.metadata)
                        post_issue_pr_handoff_comment(
                            runner,
                            config=config,
                            issue_number=target_issue_number,
                            pr_number=resolved_pr.pr_number,
                            pr_url=pr_url,
                            pr_head_sha=pr_head_sha,
                            flow="approved-plan-implementation",
                            plan_hash=plan_hash,
                            plan_growth_verdict=approval_growth_verdict,
                        )
                    return run_pr_loop(
                        runner,
                        pr_number=resolved_pr.pr_number,
                        config=config,
                        issue_context=target_issue_context,
                        approved_plan_context=approved_plan_context,
                        parent_issue_context=(
                            parent_issue_context if staged_parent_issue is not None else None
                        ),
                        usage_context=usage_context,
                        managed_ci_issue_number=target_issue_number,
                    )

                existing_handoff = find_existing_one_shot_impl_handoff(
                    target_issue_context.comments,
                    parent_issue=target_issue_number,
                    plan_hash=plan_hash,
                    mode="implement-one-shot",
                )
                any_one_shot_handoff = find_latest_one_shot_impl_handoff(
                    target_issue_context.comments,
                    parent_issue=target_issue_number,
                    mode="implement-one-shot",
                )
                if (
                    existing_handoff is None
                    and any_one_shot_handoff is not None
                    and any_one_shot_handoff.plan_hash != plan_hash
                ):
                    try:
                        older_state = get_pr_state(
                            runner, config=config, pr_number=any_one_shot_handoff.pr_number
                        )
                    except AgentLoopError as exc:
                        raise AgentLoopError(
                            f"Older one-shot handoff for PR #{any_one_shot_handoff.pr_number} "
                            f"cannot be validated ({exc}). Review it directly with `agent-loop pr "
                            f"{any_one_shot_handoff.pr_number}` or remove the stale handoff."
                        ) from exc
                    if older_state == "OPEN":
                        raise AgentLoopError(
                            f"Open one-shot handoff for PR #{any_one_shot_handoff.pr_number} has "
                            f"older plan hash {any_one_shot_handoff.plan_hash}, but the current "
                            f"approved plan has hash {plan_hash}. Review the recorded PR with "
                            f"`agent-loop pr {any_one_shot_handoff.pr_number}` or remove the "
                            "stale handoff before creating another implementation PR."
                        )
                if existing_handoff is not None:
                    try:
                        pr_state = get_pr_state(
                            runner, config=config, pr_number=existing_handoff.pr_number
                        )
                    except AgentLoopError:
                        raise AgentLoopError(
                            f"PR #{existing_handoff.pr_number} recorded in the one-shot handoff "
                            f"for issue #{target_issue_number} cannot be found in {config.repo}. "
                            f"Verify the PR exists and rerun "
                            f"`agent-loop pr {existing_handoff.pr_number}` directly to continue, "
                            "or remove the handoff comment from the issue and rerun to re-implement."
                        )
                    if pr_state == "OPEN":
                        log(
                            config,
                            f"Issue #{target_issue_number}: resuming PR #{existing_handoff.pr_number} "
                            "review for already-handed-off plan",
                        )
                        if staged_parent_issue is not None:
                            validate_pr_body_does_not_close_issue(
                                runner,
                                config=config,
                                pr_number=existing_handoff.pr_number,
                                issue_number=staged_parent_issue,
                            )
                        return run_pr_loop(
                            runner,
                            pr_number=existing_handoff.pr_number,
                            config=config,
                            issue_context=target_issue_context,
                            approved_plan_context=approved_plan_context,
                            parent_issue_context=(
                                parent_issue_context if staged_parent_issue is not None else None
                            ),
                            usage_context=usage_context,
                            managed_ci_issue_number=target_issue_number,
                        )
                    else:
                        print(
                            f"Issue #{target_issue_number} approved plan was handed off to "
                            f"PR #{existing_handoff.pr_number}, which is "
                            f"{pr_state.lower()}. Nothing to resume."
                        )
                        return 0
                return _implement_approved_issue(
                    runner,
                    issue_number=target_issue_number,
                    approved_plan=current_plan,
                    config=config,
                    memory=memory,
                    issue_context=target_issue_context,
                    approved_plan_context=approved_plan_context,
                    parent_issue_context=(
                        parent_issue_context if staged_parent_issue is not None else None
                    ),
                    coder_session_id=coder_session_id,
                    usage_context=usage_context,
                    one_shot_parent_issue=target_issue_number,
                    plan_subject=plan_subject,
                    staged_parent_issue=staged_parent_issue,
                    execution_recommendation=recommendation,
                    plan_growth_verdict=approval_growth_verdict,
                )
            raise AgentLoopError(f"Unknown plan execution mode: {mode}")

        # The round's reviews are already posted, so refresh the durable history:
        # the pre-round comment snapshot cannot hold them, and the trigger and the
        # stop must land on exactly the K-th and M-th block.  The stop is checked
        # before the generic max-rounds exit so the M-th block at the limit still
        # posts the human-decision diagnostic.
        step_back_records: tuple[PostedRoundRecord, ...] | None = None
        if (
            config.plan_step_back_rounds > 0
            and staged_planning
            and plan_primary_name is not None
            and current_plan_sidecar is not None
            and not plan_panel_evidence.opened
            and not plan_phase_advance_pending
            and inherited_guard_revision is None
            and growth_guard_revision is None
            and narrow_revision is None
            and not supersession_revision_pending
        ):
            if not step_back_history_intact:
                log(
                    config,
                    f"Planning round {round_number}: step-back suppressed: degraded "
                    f"planning history ({step_back_history_class})",
                )
            else:
                step_back_records = plan_history_records(
                    refresh=True, round_number=round_number
                )
                step_back_stop = plan_step_back_escalation(
                    step_back_records,
                    panel_opening_index=plan_panel_evidence.opening_index,
                    assessment=plan_growth_assessment,
                    measurements=plan_growth_measurements,
                    post_review=True,
                )
                if step_back_stop is not None:
                    # Before the planner turn, so no further planner turn runs.
                    stop_plan_pre_panel(step_back_stop, round_number=round_number)

        if round_number == config.max_rounds:
            if growth_guard_revision is not None:
                raise AgentLoopError(
                    f"Reached max planning rounds ({config.max_rounds}) for issue #{issue_number} "
                    "while the plan-growth gate still blocked approval: the one-shot plan "
                    f"crosses plan-growth threshold(s) without a covering justification "
                    f"({plan_growth_violation}). Raise --max-rounds and rerun so a revision "
                    "restructures the plan as staged or justifies one-shot, or rerun with "
                    "--plan-growth-gate off."
                )
            if plan_phase_advance_pending:
                # Distinct from the blocking-plan-issues message: no reviewer
                # reported a blocker, the run simply ran out of rounds while a
                # reviewer-only phase advance was still outstanding.
                raise AgentLoopError(
                    f"Reached max planning rounds ({config.max_rounds}) for issue #{issue_number} "
                    "while a reviewer-only plan phase advance was still pending. Outstanding "
                    f"phase: {plan_outstanding_phase or 'secondary-audit'}; "
                    "reviewer(s) still missing an exact-plan approval: "
                    f"{', '.join(plan_missing_approvals)}. No reviewer reported blocking plan "
                    "issues. Staged planning spends one round per phase advance; raise "
                    "--max-rounds and rerun."
                )
            raise AgentLoopError(
                f"One or more reviewers still reported blocking plan issues after "
                f"round {round_number}; human review required."
            )

        if plan_phase_advance_pending:
            # Reviewer-only round: the candidate plan stays byte-identical, no
            # planner turn runs, and the posted phase-advance record makes the
            # advance auditable and resumable.
            log(
                config,
                f"Planning round {round_number}: advancing to a reviewer-only plan round; "
                f"reviewer(s) still missing an exact-plan approval: "
                f"{', '.join(plan_missing_approvals)}; no planner turn is invoked",
            )
            post_issue_comment(
                runner,
                config=config,
                issue_number=issue_number,
                body=_attach_round_metadata(
                    render_plan_phase_advance(
                        next_round_number=round_number + 1,
                        plan_subject=current_plan_subject,
                        missing_reviewers=plan_missing_approvals,
                        phase=plan_outstanding_phase or "secondary-audit",
                    ),
                    PostedRoundMetadata(
                        flow="plan",
                        role="summary",
                        agent="Orchestrator",
                        round_number=round_number + 1,
                        subject=current_plan_subject,
                        prior_items=tuple(unresolved_items),
                        phase="plan-phase-advance",
                        plan_candidate_key=current_plan_key.as_dict(),
                        **_architecture_metadata_fields(config),
                    ),
                ),
            )
            resumed_round = None
            continue

        combined_review = "\n\n".join(f"{name} plan review:\n\n{review}" for name, review in blocking_reviews)
        revision_initial_diagnostic: object | None = plan_validation_diagnostic
        if inherited_guard_revision is not None:
            # Attributed to the orchestrator; no synthetic reviewer item exists.
            combined_review = (
                "Orchestrator inherited-matrix check (not a reviewer finding):\n\n"
                f"{inherited_guard_revision}"
            )
            revision_initial_diagnostic = _InheritedReplanDiagnostic(
                diagnostic=inherited_guard_revision,
                failure_attempt=1,
                candidate_digest=hashlib.sha256(current_plan.encode("utf-8")).hexdigest(),
            )
            # The revised subject gets the complete board and carries no
            # approval from the superseded one.
            plan_automatic_force_full = True
        if growth_guard_revision is not None:
            # Attributed to the orchestrator; no reviewer item ID is minted.
            combined_review = (
                f"{combined_review}\n\n{growth_guard_revision}"
                if combined_review
                else growth_guard_revision
            )
        if narrow_revision is not None:
            combined_review = (
                f"{combined_review}\n\n{narrow_revision}" if combined_review else narrow_revision
            )
            narrow_directive_delivered = True
        if supersession_revision_pending:
            # The signed authorization is the revision's instruction; the
            # planner must actually replace the plan, not return it (#985).
            authorization = (
                "Signed child-plan supersession (human authorization, not a reviewer "
                f"finding): approved plan {plan_supersession.superseded_hash} must be "
                f"replaced. Human rationale:\n\n{plan_supersession.rationale}"
            )
            combined_review = (
                f"{authorization}\n\n{combined_review}" if combined_review else authorization
            )
            plan_automatic_force_full = True
        step_back_context: PlanStepBackContext | None = None
        if step_back_records is not None and current_plan_sidecar is not None:
            step_back_state = plan_step_back_state(
                step_back_records, panel_opening_index=plan_panel_evidence.opening_index
            )
            if (
                step_back_state is not None
                and step_back_state.episode is None
                and step_back_state.reviews
                and step_back_state.reviews[-1].round_number == round_number
            ):
                crossing_round = plan_growth_crossing_round(
                    step_back_records,
                    config=config,
                    current_round=current_plan_sidecar.round_number,
                )
                if crossing_round is not None:
                    before = step_back_state.streak_before(crossing_round, round_number)
                    streak = max(step_back_state.streak_since(crossing_round), before)
                    # A step-back pending at this round (judged from earlier rounds)
                    # is granted on any blocking primary review, repeat-only too.
                    pending = (
                        before >= config.plan_step_back_rounds
                        and step_back_state.reviews[-1].classification in BLOCKING_CLASSES
                    )
                    if pending or (
                        step_back_state.streak_since(crossing_round)
                        >= config.plan_step_back_rounds
                    ):
                        step_back_context = PlanStepBackContext(
                            measurements=plan_step_back_measurements(
                                plan_growth_assessment, plan_growth_measurements
                            ),
                            findings=mandatory_plan_findings_since(
                                step_back_records,
                                primary=plan_primary_name,
                                first_round=crossing_round,
                            ),
                            execution_mode=config.plan_execution_mode,
                            streak=streak,
                        )
                        log(
                            config,
                            f"Planning round {round_number}: step-back revision: the plan "
                            f"crossed a growth signal at round {crossing_round} and the "
                            f"primary blocked {streak} consecutive round(s) on new findings",
                        )
        step_back_anchor: PlanStepBackAnchor | None = None
        if (
            step_back_context is None
            and config.plan_step_back_rounds > 0
            and plan_primary_name is not None
            and step_back_history_intact
            and not plan_panel_evidence.opened
            and pre_round_step_back_episode is not None
        ):
            # Independent of eligibility for a new step-back turn: narrowing, guard
            # and supersession revisions inside an episode carry the anchor too.
            anchor_records = (
                step_back_records
                if step_back_records is not None
                else plan_history_records(refresh=True, round_number=round_number)
            )
            anchor_state = plan_step_back_state(
                anchor_records, panel_opening_index=plan_panel_evidence.opening_index
            )
            if anchor_state is not None and anchor_state.episode is not None:
                step_back_anchor = plan_step_back_anchor(anchor_records, anchor_state.episode)
                log(
                    config,
                    f"Planning round {round_number}: step-back anchor: revision "
                    "constrained to the simplified design from round "
                    f"{anchor_state.episode.candidate_round} "
                    f"({len(step_back_anchor.dissolved_ids)} dissolved item(s))",
                )
        log(
            config,
            f"Planning round {round_number}: {coder_name} revising the plan "
            f"(context mode: {context_mode}{context_reason})",
        )
        plan_revision_human_requirements_context = render_coder_human_requirements_prompt_context(
            issue_context.human_requirements,
            requirement_scope="planning requirements",
            full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
        )
        semantic_revision = (
            current_plan_sidecar is not None
            and current_response_form in {
                "fresh-plan-state", "legacy-full-state", "semantic-patch-v1"
            }
            and isinstance(current_plan_sidecar.canonical_json.get("risk_test_matrix"), dict)
        )
        semantic_base: AuthenticatedPlanState | None = None
        if semantic_revision:
            # Hydration is a pre-prompt gate. Missing or conflicting durable
            # authority must not be papered over with the visible Markdown.
            semantic_base = hydrate_authenticated_plan_state(current_plan_sidecar)
        def invoke_revision_planner(turn_diagnostic: object | None) -> ValidatedAgentResponse:
            return _run_validated_agent(
                runner,
                agent=config.coder,
                config=config,
                prompt=build_plan_revision_prompt(
                    issue_number,
                    round_number,
                    current_plan,
                    combined_review,
                    config,
                    memory,
                    issue_context=issue_context,
                    unresolved_items=must_fix_items,
                    compact_context=use_compact_context,
                    compact_prior=CompactPriorContext(tuple(compact_prior_summaries)),
                    compact_tail=CompactPlanTailContext(
                        subject=current_plan_subject,
                        action=(
                            "Step back: propose a materially simpler alternative or a "
                            "re-scope instead of patching the newest finding."
                            if step_back_context is not None
                            else "Revise the implementation plan to address the blocking plan review."
                        ),
                    ),
                    require_risk_test_matrix_contract=require_fresh_matrix_contract,
                    plan_validation_diagnostic=turn_diagnostic,
                    inherited_matrix_binding=inherited_matrix_binding,
                    response_form=("semantic-patch-v1" if semantic_revision else None),
                    base_round_number=(semantic_base.round_number if semantic_base is not None else None),
                    base_state_identity=(semantic_base.state_identity if semantic_base is not None else None),
                    plan_growth_notice=plan_growth_notice,
                    step_back_context=step_back_context,
                    step_back_anchor=step_back_anchor,
                    generalization_guidance=True,
                    finding_history=plan_finding_history.view(round_number),
                    prior_execution_mode=prior_execution_mode,
                    execution_mode_history_present=execution_mode_history_present,
                ),
                session_id=coder_session_id,
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                # A semantic patch carries no assessment of its own; the assembled
                # plan inherits the base or a strict patch replace.
                require_architecture_impact_contract=not semantic_revision,
                **_architecture_mode_validators(lambda mode: (
                    (lambda text, human_requirements=issue_context.human_requirements, items=tuple(must_fix_items): _validate_plan_revision_patch_response(
                        text,
                        unresolved_items=items,
                        human_requirements=human_requirements,
                        inherited_human_requirement_dispositions=(
                            semantic_base.plan.human_requirement_dispositions
                            if semantic_base is not None else None
                        ),
                    ))
                    if semantic_revision
                    else (lambda text, human_requirements=issue_context.human_requirements, items=tuple(must_fix_items): _validate_response_with_human_requirements(
                        text,
                        marker_validator=lambda revised_text: _validate_plan_revision_response(
                            revised_text,
                            unresolved_items=items,
                            require_architecture_impact=True,
                            require_execution_strategy_contract=require_fresh_execution_contract,
                            require_risk_test_matrix_contract=require_fresh_matrix_contract,
                            reject_unsolicited_risk_test_matrix_contract=(
                                not require_fresh_matrix_contract
                            ), architecture_status_mode=mode,
                        ),
                        human_requirements=human_requirements,
                        requirement_scope="planning requirements",
                        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
                    ))
                )),
                usage_context=usage_context,
                use_repair=True,
                repair_expected_kind=("plan_revision_patch" if semantic_revision else "plan_revision"),
                repair_surfaced_requirement_ids=(
                    plan_revision_human_requirements_context.surfaced_requirement_ids
                ),
                repair_requires_direct_discussion_ack=(
                    plan_revision_human_requirements_context.requires_direct_discussion_ack
                ),
                require_execution_strategy_contract=(
                    False if semantic_revision else require_fresh_execution_contract
                ),
                require_risk_test_matrix_contract=(
                    False if semantic_revision else require_fresh_matrix_contract
                ),
                reject_unsolicited_risk_test_matrix_contract=(
                    False if semantic_revision else not require_fresh_matrix_contract
                ),
                repair_allowed_prior_item_ids=tuple(item.item_id for item in must_fix_items),
                ledger_incomplete=round_ledger_incomplete,
                # The revision runs after this round's dispositions were applied,
                # so its proof must cover the items this round resolved. The
                # pre-round value still carries them, and the pre-loop comment
                # snapshot cannot replay the round that cleared them, so the proof
                # also reads this round's in-process dispositions (#874).
                repair_resolved_history_item_ids=_post_round_resolved_history_item_ids(
                    prior_unresolved_items=prior_unresolved_items,
                    dispositions_by_item=prior_dispositions,
                    carried_items=unresolved_items,
                    comments=issue_context.comments,
                    flow="plan",
                    reconciliation_mode="aggregate",
                    same_status="same-plan",
                ),
                operation_description="plan revision",
                semantic_patch_payload_validator=(
                    (lambda payload, human_requirements=issue_context.human_requirements, items=tuple(must_fix_items): _validate_plan_revision_patch_payload(
                        payload,
                        unresolved_items=items,
                        human_requirements=human_requirements,
                        inherited_human_requirement_dispositions=(
                            semantic_base.plan.human_requirement_dispositions
                            if semantic_base is not None else None
                        ),
                    ))
                    if semantic_revision
                    else None
                ),
                plan_validation_failure_handler=(
                    None if semantic_revision else lambda exhaustion, error: _persist_exhausted_plan_validation_diagnostic(
                        runner,
                        config=config,
                        issue_context=issue_context,
                        issue_number=issue_number,
                        original_error=error,
                        exhaustion=exhaustion,
                        target_coder_round=round_number + 1,
                        prior_plan_subject=current_plan_subject,
                        candidate_kind="plan_revision",
                        require_execution_strategy_contract=require_fresh_execution_contract,
                        require_risk_test_matrix_contract=require_fresh_matrix_contract,
                    )
                ),
            )

        def revision_candidate_matrix(response: ValidatedAgentResponse) -> object:
            if semantic_revision:
                if not isinstance(response.marker_value, PlanRevisionPatch):
                    raise AgentLoopError(
                        "Semantic planning response crossed the pinned response form."
                    )
                assert semantic_base is not None
                # Assembled against the unchanged authenticated base; the
                # result stays local until the candidate passes.
                assembled_candidate, _ = assemble_authenticated_plan_revision(
                    semantic_base,
                    response.marker_value,
                    result_round_number=round_number + 1,
                )
                check_candidate_growth(
                    assembled_candidate,
                    canonical_text=render_canonical_plan_revision(
                        assembled_candidate, must_fix_items, config
                    ),
                    prior_payload=semantic_base.canonical_payload,
                    target_round=round_number + 1,
                )
                return assembled_candidate.risk_test_matrix
            if isinstance(response.marker_value, StructuredPlanRevision):
                check_candidate_growth(
                    response.marker_value,
                    canonical_text=render_canonical_plan_revision(
                        response.marker_value, must_fix_items, config
                    ),
                    prior_payload=(
                        current_plan_sidecar.canonical_json
                        if current_plan_sidecar is not None
                        else None
                    ),
                    target_round=round_number + 1,
                )
            return getattr(response.marker_value, "risk_test_matrix", None)

        plan_response = run_inherited_checked_planner_turn(
            invoke_revision_planner,
            derive_matrix=revision_candidate_matrix,
            initial_diagnostic=revision_initial_diagnostic,
            candidate_kind="plan_revision",
            target_coder_round=round_number + 1,
            prior_plan_subject=current_plan_subject,
        )
        canonical_plan: str | None = None
        public_comment = plan_response.text
        raw_structured_coder_response: str | None = None
        # Observed before the sidecar is replaced: the classifier compares the
        # cross-cutting contracts across exactly this transition.
        previous_plan_contracts = _plan_cross_cutting_contracts(current_plan_sidecar)
        current_plan_patch = (
            plan_response.marker_value
            if semantic_revision and isinstance(plan_response.marker_value, PlanRevisionPatch)
            else None
        )
        if semantic_revision:
            if not isinstance(plan_response.marker_value, PlanRevisionPatch):
                raise AgentLoopError(
                    "Semantic planning response crossed the pinned response form."
                )
            assert semantic_base is not None
            assembled_plan, assembled_sidecar = assemble_authenticated_plan_revision(
                semantic_base,
                plan_response.marker_value,
                result_round_number=round_number + 1,
            )
            raw_structured_coder_response = plan_response.text
            canonical_plan = render_canonical_plan_revision(
                assembled_plan, must_fix_items, config
            )
            assembled_sidecar = make_assembled_plan_sidecar(
                assembled_plan,
                round_number=round_number + 1,
                response_form="semantic-patch-v1",
                raw_patch=assembled_sidecar.raw_patch,
                rendered_plan=canonical_plan,
            )
            current_plan = canonical_plan
            current_coder_output = plan_response.text
            current_plan_sidecar = assembled_sidecar
            current_response_form = "semantic-patch-v1"
            public_comment = render_public_agent_comment(
                kind="plan_revision",
                parsed=assembled_plan,
                agent=config.coder,
                prior_items=must_fix_items,
                raw_text=plan_response.text,
                config=config,
                model_used=plan_response.model_used,
            )
        elif isinstance(plan_response.marker_value, StructuredPlanRevision):
            raw_structured_coder_response = plan_response.text
            previous_matrix_match = RISK_TEST_MATRIX_MARKER_RE.search(current_plan)
            if previous_matrix_match is not None and plan_response.marker_value.risk_test_matrix is None:
                raise AgentLoopError(
                    "Plan revision omitted the approved draft risk matrix; preserve its rows or "
                    "record an explicit matrix revision instead of silently removing it."
                )
            if previous_matrix_match is not None and plan_response.marker_value.risk_test_matrix is not None:
                previous_matrix_payload = decode_risk_test_matrix_marker(
                    previous_matrix_match.group("payload")
                )
                validate_risk_test_matrix_revision(
                    previous_matrix_payload["matrix"],
                    plan_response.marker_value.risk_test_matrix,
                    plan_response.marker_value.risk_test_matrix_changes,
                    historical_changes=previous_matrix_payload.get("changes", ()),
                )
            canonical_plan = render_canonical_plan_revision(
                plan_response.marker_value, must_fix_items, config
            )
            current_plan = canonical_plan
            current_coder_output = plan_response.text
            current_plan_sidecar = make_assembled_plan_sidecar(
                plan_response.marker_value,
                round_number=round_number + 1,
                response_form="legacy-full-state",
                rendered_plan=canonical_plan,
            )
            current_response_form = "legacy-full-state"
            public_comment = render_public_agent_comment(
                kind="plan_revision",
                parsed=plan_response.marker_value,
                agent=config.coder,
                prior_items=must_fix_items,
                raw_text=plan_response.text,
                config=config,
                model_used=plan_response.model_used,
            )
        else:
            current_plan = plan_response.text
            current_coder_output = plan_response.text
            current_plan_sidecar = None
            current_response_form = None
            # Free-form revisions also need a lossless canonical sidecar. The
            # rendered signature is presentation only and is not plan identity.
            canonical_plan = current_plan
            public_comment = normalize_freeform_signature(
                plan_response.text, agent=config.coder, config=config, model_used=plan_response.model_used
            )
        metadata_plan = (
            assembled_plan
            if semantic_revision
            else plan_response.marker_value
            if isinstance(plan_response.marker_value, StructuredPlanRevision)
            else None
        )
        coder_session_id = plan_response.session_id
        plan_finding_history.record_fix(
            plan_response.marker_value,
            published_round=round_number + 1,
            agent=coder_name,
        )
        log_declared_generalization(
            plan_response.marker_value,
            log=lambda message: log(config, message),
            round_number=round_number,
            agent=coder_name,
            step_back_directed=step_back_context is not None,
        )
        try:
            plan_round_metadata = PostedRoundMetadata(
                    flow="plan",
                    role="coder",
                    plan_execution_mode=config.plan_execution_mode,
                    agent=coder_name,
                    round_number=round_number + 1,
                    subject=_plan_subject(current_plan),
                    prior_plan_subject=current_plan_subject,
                    prior_items=tuple(unresolved_items),
                    canonical_plan=canonical_plan,
                    raw_structured_coder_response=raw_structured_coder_response,
                    compact_prior_summaries=tuple(compact_prior_summaries),
                    model_used=plan_response.model_used,
                    **_metadata_identity_fields(plan_response),
                    execution_strategy_contract_version=(
                        1
                        if metadata_plan is not None
                        and metadata_plan.execution_recommendation is not None
                        else None
                    ),
                    execution_strategy_identity=(
                        metadata_plan.execution_recommendation.identity()
                        if metadata_plan is not None
                        and metadata_plan.execution_recommendation is not None
                        else None
                    ),
                    risk_test_matrix_contract_version=(
                        metadata_plan.risk_test_matrix_contract_version
                        if metadata_plan is not None else None
                    ),
                    risk_test_matrix_payload=(
                        metadata_plan.risk_test_matrix.to_payload()
                        if metadata_plan is not None
                        and metadata_plan.risk_test_matrix is not None else None
                    ),
                    risk_test_matrix_changes_payload=(
                        tuple(change.to_payload() for change in metadata_plan.risk_test_matrix_changes)
                        if metadata_plan is not None else ()
                    ),
                    risk_test_matrix_identity=(
                        risk_test_matrix_identity(
                            metadata_plan.risk_test_matrix,
                            metadata_plan.risk_test_matrix_changes,
                        )
                        if metadata_plan is not None
                        and metadata_plan.risk_test_matrix is not None else None
                    ),
                    risk_test_matrix_boundary_digest=(
                        risk_test_matrix_identity(
                            metadata_plan.risk_test_matrix,
                            metadata_plan.risk_test_matrix_changes,
                        )
                        if metadata_plan is not None
                        and metadata_plan.risk_test_matrix is not None else None
                    ),
                    response_form=(
                        current_response_form if current_plan_sidecar is not None else None
                    ),
                    base_round_number=(
                        plan_response.marker_value.base_round_number
                        if semantic_revision and isinstance(plan_response.marker_value, PlanRevisionPatch)
                        else None
                    ),
                    base_state_identity=(
                        plan_response.marker_value.base_state_identity
                        if semantic_revision and isinstance(plan_response.marker_value, PlanRevisionPatch)
                        else None
                    ),
                    aggregate_plan_identity=(
                        current_plan_sidecar.aggregate_identity
                        if current_plan_sidecar is not None else None
                    ),
                    raw_patch_provenance=(
                        current_plan_sidecar.raw_patch
                        if semantic_revision and current_plan_sidecar is not None else None
                    ),
                    assembled_plan_sidecar=(
                        current_plan_sidecar.to_payload()
                        if current_plan_sidecar is not None else None
                    ),
                    **_architecture_metadata_fields(
                        config, result=metadata_plan
                    ),
                    acquisition_outcome=plan_response.acquisition_outcome,
                    acquisition_returncode=plan_response.acquisition_returncode,
                    # Every coder round of a signed re-plan carries the
                    # authorization digest (#936); absent otherwise.
                    plan_supersession_digest=(
                        plan_supersession.digest if plan_supersession is not None else None
                    ),
                    plan_supersession_superseded_hash=(
                        plan_supersession.superseded_hash
                        if plan_supersession is not None else None
                    ),
                    # Orchestrator-written, reviewer-owned, never from agent output.
                    step_back_entries=(
                        (entry_payload_for_plan(
                            reviewer=plan_primary_name, trigger_round=round_number
                        ),)
                        if step_back_context is not None and plan_primary_name is not None
                        else ()
                    ),
            )
        except ValueError as exc:
            # A contradictory metadata record must not escape as a bare
            # dataclass ValueError with no diagnostic (#879).
            raise AgentLoopError(
                "Could not record the plan round metadata for round "
                f"{round_number + 1} (response form "
                f"{current_response_form or 'free-form'}): {exc}"
            ) from exc
        if metadata_plan is not None:
            plan_round_body = _assemble_structured_plan_round_body(
                config=config,
                issue_number=issue_number,
                kind="plan_revision",
                parsed_plan=metadata_plan,
                full_comment=public_comment,
                metadata=plan_round_metadata,
                raw_text=plan_response.text,
                prior_items=must_fix_items,
                model_used=plan_response.model_used,
                surfaced_requirement_ids=(
                    plan_revision_human_requirements_context.surfaced_requirement_ids
                ),
                requires_direct_discussion_ack=(
                    plan_revision_human_requirements_context.requires_direct_discussion_ack
                ),
            )
        else:
            plan_round_body = _attach_round_metadata(public_comment, plan_round_metadata)
        if _post_plan_coder_round_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=plan_round_body,
            diagnostic=plan_validation_diagnostic,
            target_coder_round=round_number + 1,
            prior_plan_subject=current_plan_subject,
            candidate_kind="plan_revision",
            require_execution_strategy_contract=require_fresh_execution_contract,
            require_risk_test_matrix_contract=require_fresh_matrix_contract,
        ):
            plan_validation_diagnostic = None
        if current_plan_sidecar is not None:
            planner_candidate_rounds.add(round_number + 1)
        resumed_round = None

    raise AgentLoopError(
        f"Reached max planning rounds ({config.max_rounds}) for issue #{issue_number}; "
        "human review required."
    )
