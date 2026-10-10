"""The PR review loop entry point.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1200); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import hashlib
import shlex
import sys
import time
from collections.abc import Sequence
from dataclasses import replace as dataclasses_replace
from .agents.base import AgentName
from .agents.registry import agent_display_name
from .workdir_claims import claimed_run
from .config import (
    AgentLoopConfig,
    ensure_agent_workdirs,
    github_bootstrap_cwd,
    resolve_base_branch,
    reviewers,
    resolve_invocation,
    sync_coder_pr_before_validation,
    sync_reviewer_pr_before_review,
)
from .board_amendment import (
    ContractLineage,
    amendment_audit_already_posted,
    amendment_summary_line,
    apply_board_amendment_to_ledger,
    collect_reviewer_board_amendments,
    predates_restoration,
    reject_misplaced_pr_amendments,
    render_amendment_audit_comment,
    require_amendment_activation,
    resolve_contract_lineage,
)
from .decomposition import (
    _decode_json_payload,
    approved_plan_hash,
    find_existing_decomposition,
    find_latest_one_shot_impl_handoff,
    normalize_execution_recommendation,
    risk_matrix_row_ids_for_owner,
    recover_execution_recommendation,
    EXECUTION_TOPOLOGY_SOURCE,
    find_existing_topology_checkpoint,
    find_phase_implementation_handoffs_for_parent,
    PHASE_IDENTITY_MARKER_RE,
    phase_identity,
    collect_child_disposition_overrides,
    reconcile_handoff_disposition,
)
from .protocol import (
    EXECUTION_DISPOSITION_DIRECT,
    EXECUTION_DISPOSITION_PLANNING,
)
from .evidence_stall import approved_matrix_row_ids
from .errors import (
    AgentInvocationError,
    AgentLoopError,
    HumanDecisionRequiredError,
    QuotaResetExceededError,
)
from .expected_closure import (
    reject_parent_from_contract,
    resolve_direct_contract,
    resolve_issue_contract,
)
from .github import (
    IssueContext,
    PullRequestMetadata,
    PullRequestChecks,
    PullRequestMergeability,
    PullRequestReviewContext,
    get_issue_context,
    get_pr_mergeability,
    parse_linked_issue_numbers,
    get_pr_checks,
    get_pr_review_context,
    post_pr_comment,
    post_trusted_pr_comment,
    read_rest_issue_comments,
    post_frozen_round_bodies,
    round_publication,
    reject_forged_protocol_markers,
    validate_open_issue,
    validate_open_pr,
    validate_pr_expected_closing_issues,
    watch_pr_checks,
)
from .publication_resume import context_digest as publication_context_digest
from .issue_pr_handoff import (
    find_latest_issue_pr_handoff,
    post_issue_pr_handoff_comment,
    require_pr_metadata_for_handoff,
)
from .pr_contract import (
    find_latest_pr_contract,
    format_pr_contract_comment,
    make_pr_contract,
)
from .split_materialization import (
    SPLIT_CHILD_MARKER_RE,
    find_existing_split_stage_handoff,
)
from .logging import log
from .memory import prepare_agent_memory
from .managed_ci import (
    AuthenticatedManagedResume,
    AuthenticatedIssueCreatedHandoff,
    FINAL_CONTEXT,
    MANAGED_LABEL,
    OrdinaryRecoveryCapability,
    activate_managed_ci,
    managed_ci_attachment_matches_head,
    release_stale_managed_ci_attachment,
    authorize_fresh_issue_created_resume,
    authenticate_source_managed_resume,
    dispatch_final_qualification,
    intermediate_managed_checks,
    managed_label_present,
    find_actor_round_metadata_comment_ids,
    publish_issue_created_continuity_authorization,
    publish_manual_v2_qualification,
    prepare_v2_merge,
    publish_round_readiness,
    release_adopted_managed_ci,
    release_retained_managed_label,
    revalidate_adopted_managed_ci,
    revalidate_issue_created_handoff,
    recover_issue_created_handoff,
    render_managed_ci_resume_command,
    wait_for_final_qualification,
)
from .managed_pr import (
    recover_managed_pr_origin,
    validate_managed_pr_body,
)
from .migrations import validate_pr_migration_topology
from .finding_history import (
    PHASE_PR as _HISTORY_PHASE_PR,
    FindingHistoryLedger,
    log_declared_generalization,
)
from .prompts import (
    CompactPriorContext,
    CompactPrReviewTailContext,
    build_followup_prompt,
    build_merge_conflict_prompt,
    build_review_prompt,
    build_same_pr_followup_prompt,
    format_agent_list,
    render_coder_human_requirements_prompt_context,
)
from .protocol import (
    ParsedPlanReview,
    ParsedReview,
    ReviewItemDisposition,
    StructuredCoderFollowup,
    UnresolvedReviewItem,
    human_requirements_resolved,
    parse_agent_state,
    parse_structured_pr_review,
    review_freeform_summary_text,
)
from .protocol import parse_review
from .runner import Runner
from .salvage import SalvageContext
from .usage import RunUsageContext
from .workdirs import active_workdir
from .workdir_guard import (
    validate_response_tests_within_workdir,
    validate_test_observation_citations_within_workdir,
)
from .checks import (
    _ci_infrastructure_details,
    _format_ci_infrastructure_comment,
    _format_pr_checks_comment,
    _ci_infrastructure_stop_message,
    _pending_ci_status_summary,
    _pending_ci_stop_guidance,
    _pending_ci_stop_message,
    _pr_check_blocking_review,
    _pr_check_details,
    run_optional_tests,
    run_pre_review_tests,
)
from .ci_health import (
    CiInfrastructureStall,
    is_wholly_infrastructure_blocked,
)
from .local_test_evidence import unsuperseded_receipts_sentence
from .comment_rendering import (
    add_coder_followup_head_unchanged_notice,
    normalize_freeform_signature,
    same_model_panel_note,
    render_public_agent_comment,
    resolve_matrix_evidence_render,
)
from .followups import (
    _approved_followup_from_unresolved_item,
    _publish_approved_followups,
)
from .round_state import (
    ApprovedPlanContext,
    EVIDENCE_FREEZE_PHASE,
    EVIDENCE_RELEASE_PHASE,
    EVIDENCE_RESPONSE_PHASE,
    QualificationCheckpoint,
    PostedRoundMetadata,
    _is_followup_dispatch_head,
    PostedRoundRecord,
    ResumedReviewRound,
    _attach_round_metadata,
    _extract_round_metadata_records,
    _latest_pr_approved_reviews_for_head,
    _prior_item_ledger_signature,
    _resume_plan_round,
    make_approved_plan_context,
    scope_approved_plan_matrix,
    recover_approved_plan_context,
    _resume_pr_round,
    RecoveryRoundBudget,
)
from .protocol_markers import TrustedBody
from .review_scheduling import (
    PrePanelSafetyError,
    SchedulerSnapshot,
    TransitionClassification,
    make_contract,
    policy_capabilities,
    pre_panel_safety_message,
    select_reviewers,
    undecodable_history_message,
)
from .partial_round_recovery import (
    PartialRoundRecovery,
    RecoveryVisibilityContext,
    compute_partial_round_recovery,
)
from .round_visibility import (
    latest_round_checkpoint_index,
    visible_peer_names,
)
from .unresolved_items import (
    ClearedItemProgress,
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    MERGE_CONFLICT_ITEM_ID,
    _apply_dispute_evidence,
    _apply_unresolved_item_dispositions,
    _collect_prior_compact_summaries,
    bound_compact_prior_summaries,
    _clear_human_requirements_ack_item,
    _format_same_pr_unresolved_items,
    format_coder_followup_context,
    _next_unresolved_item,
    _advance_machine_obligations_for_head,
    _clear_machine_obligations,
    _is_machine_obligation,
    _machine_obligation_is_revalidation_candidate,
    _set_machine_obligation_lifecycle,
    _upsert_machine_obligation,
    _reconcile_merge_conflict_item,
    _record_prior_item_disposition,
    _reconcile_human_requirements_ack_item,
    _raise_if_maintained_disputed_items,
    select_coder_followup_items,
    _frozen_evidence_obligations,
    _is_evidence_obligation,
    _pending_evidence_obligations,
    _upsert_evidence_obligation,
    release_evidence_freeze,
    _validate_coder_followup_response,
    _validate_review_response,
)
from .agent_failure import (
    PR_FOLLOWUP_SALVAGE_SCOPE,
    ValidatedAgentResponse,
    _metadata_identity_fields,
    _capture_agent_invocation,
)
from .architecture_contract import (
    _freeze_prompt_architecture,
    _architecture_metadata_fields,
    _test_observation_degradation_fields,
    _latest_pr_architecture_observation,
    _architecture_mode_validators,
    _resumed_review_architecture,
    _acknowledgement_repair_forbids_assessment,
    _pin_acknowledgement_repair,
)
from .validated_agent import (
    _new_usage_context,
    _begin_run_telemetry,
    _end_run_telemetry,
    _persist_usage_summary,
    _run_structured_repair,
    _log_repair_attempts,
    _run_validated_agent,
)
from .risk_coverage_map import (
    coverage_map_applies,
    extract_coverage_map_section,
    render_coverage_map,
)
from .response_validation import (
    _current_test_turn_observations,
    _derive_authenticated_risk_evidence_for_coder,
    final_risk_coverage_assessment,
    _build_requirements_context,
    _surfaced_reviewer_requirement_ids,
    _reviewer_requirement_identity_ids,
    _reviewer_requirement_coverage_matches,
    _resumed_pr_reviewer_matches_requirements,
    _drop_repeated_carried_future_followups,
    _degrade_out_of_checkout_tests,
)
from .panel_evidence import (
    PrPanelEvidence,
    _derive_pr_panel_evidence,
    _pr_record_is_panel_qualified,
    _describe_superseded_prepanel_review,
    _superseded_prepanel_review,
    _scheduler_recorded_force_full,
    _board_amendment_template,
    _require_handoff_plan_growth_verdict,
    _require_complete_canonical_plan_approval,
)
from .review_rounds import (
    _is_pending_ci_only_review,
    _normalize_approval_gated_managed_ci_review,
    _coder_infrastructure_stall_notice,
    _is_infrastructure_ci_only_review,
    _should_record_new_blocking_item,
    _is_incomplete_pr_review,
    _describe_pr_review_outcome,
    _format_incomplete_pr_review_comment,
    _unavailable_reviewer_remedy,
    _unavailable_reviewer_amendment_advisory,
    _round_ledger_may_be_incomplete,
    _round_resolved_history_item_ids,
    _ReviewerTurnResult,
    _review_round_spool,
    _replay_spooled_review,
    _preflight_spooled_publications,
    _incomplete_pr_review_error,
    _refuse_partial_round_before_sequential_turns,
    _same_round_replay_or_invoke,
    _launch_reviewer_turns,
    _ensure_parallel_reviewer_workdirs,
)
from .execution_policy import (
    _extract_current_expected_closing_issue_ids,
    _child_resume_hint,
    _record_staged_parent_completion_after_merge,
    _infer_staged_parent_issue,
)
from .child_plan_binding import (
    _PlanningChildBinding,
    _managed_ci_retired_plan_hashes,
    _require_admissible_pr_child_plan,
    _reject_pending_child_plan_supersession,
)
from .pr_loop_support import (
    ExactHeadCiProof,
    _pr_step_back_carried_entries,
    _pr_step_back_decision,
    _pr_step_back_sweep_contexts,
    _pr_step_back_tracked_reviewers,
    _merge_with_exact_head_proof,
    _read_assigned_workdir_head,
    MAX_UNCHANGED_HEAD_CODER_TURNS,
    _UnchangedHeadTracker,
    _coder_followup_head_log,
    _coder_followup_review_context,
    _reviewer_summary_context,
    _stop_on_terminal_without_status,
    _pr_followup_source_context,
    _stop_after_ci_watch_timeout,
    _scheduler_obligations,
    _partition_unresolved_items,
    _log_coder_followup_dispatch,
    _machine_obligation_checkpoint,
    _qualification_digest,
    _qualification_checkpoint_review_identity_matches,
    approved_pr_reopen_hint,
    _publish_sub_item_progress,
    _sub_item_progress_block,
    _round_limit_diagnostic,
    _single_line_diagnostic,
    _ensure_finalization_ready,
    _RepeatableRoundSequence,
    _evidence_barrier_note,
    _append_evidence_barrier_note,
    _evidence_freeze_diagnostic,
    _publish_evidence_freeze,
    _pr_evidence_stall_decision,
    _followup_reports_code_changes,
    _evidence_stall_citation_rows,
    _publish_evidence_release,
    _evidence_freeze_gate,
    _refuse_dispatch_while_evidence_frozen,
    _is_evidence_only_blocking_review,
    _evidence_review_context,
    _finalize_ordinary_recovery_checked,
    _mergeability_for_unreadable_protection,
    _ordinary_snapshot_nonauthority_reason,
    _ordinary_checks_snapshot_is_authoritative,
    _managed_success_supersedes_ordinary_checks,
    _persist_qualification_checkpoint,
    _persist_coder_dispatch,
    _persist_head_review_recovery,
    _record_coder_followup_rejection,
    _resume_pr_round_admitted,
    _cached_trusted_actor,
    _visibility_snapshot,
    _latest_pr_reviewer_records,
    _reviewer_needs_fresh_context,
    _reviewer_history_is_reconstructible,
    _returning_reviewer_context,
    _observe_pr_transition,
    _scheduler_contract_from_metadata,
    _pr_contract_drift_error,
    _post_reduced_board_completion_note,
    _pr_amendment_start_round,
    _managed_binding_retired_plan_hashes,
    _managed_binding_protection_mode,
    _pr_amendment_plan_board_hint,
    _fresh_pr_qualification_snapshot,
    _preserve_issue_created_managed_suppression,
    _recover_managed_ci_approved_plan,
)


def unchanged_head_stop_message(
    *, pr, coder_name, previous_head, turns, round_number, evidence, route
) -> str:
    """The unchanged-head stop text; receipts (issue #1182) go before the route."""
    return (
        f"PR #{pr}: {coder_name} left head {previous_head} unchanged in "
        f"{turns} consecutive follow-up rounds, so another "
        "review of the same diff cannot change the verdict. Stopping before round "
        f"{round_number + 1}; human review required."
        f"{unsuperseded_receipts_sentence(evidence)}"
        f"{route}"
    )


@claimed_run("pr", "pr_number")
def run_pr_loop(
    runner: Runner,
    *,
    pr_number: int,
    config: AgentLoopConfig,
    coder_session_id: str | None = None,
    reviewer_session_id: str | None = None,
    issue_context: IssueContext | None = None,
    approved_plan_context: ApprovedPlanContext | None = None,
    parent_issue_context: IssueContext | None = None,
    workdirs_ready: bool = False,
    usage_context: RunUsageContext | None = None,
    pre_review_test_pending: bool = False,
    managed_pr_origin: tuple[str, str, str, str | None] | None = None,
    managed_ci_handoff: AuthenticatedIssueCreatedHandoff | None = None,
    initial_coverage_map: str | None = None,
    managed_ci_issue_number: int | None = None,
) -> int:
    owned_usage_context = usage_context is None
    usage_context = usage_context or _new_usage_context(config)
    managed_ci = None
    ordinary_recovery: OrdinaryRecoveryCapability | None = None
    ordinary_recovery_selected = False
    managed_ci_qualified = False
    managed_pr_recovered = False
    authenticated_managed_resume: AuthenticatedManagedResume | None = None
    # Captured once by the provenance block for a fresh planning child and
    # passed to every qualification snapshot (#936); ``None`` otherwise.
    planning_child_binding: _PlanningChildBinding | None = None

    def recheck_pending_child_plan_supersession() -> None:
        """Refetch the child issue before any approval or merge (#985).

        A signed supersession posted while reviewers ran must stop the run on
        every finalization path, not only those that take a qualification
        snapshot.
        """
        if planning_child_binding is None or approved_plan_context is None:
            return
        fresh_child_issue = get_issue_context(
            runner, config=config, issue_number=planning_child_binding.child_issue
        )
        _reject_pending_child_plan_supersession(
            fresh_child_issue.comments,
            binding=planning_child_binding,
            plan_hash=approved_plan_context.plan_hash,
            pr_number=pr_number,
            stopped="no approval or merge was attempted",
        )

    unchanged_head_tracker = _UnchangedHeadTracker()
    telemetry_token = _begin_run_telemetry(
        runner, config, usage_context, owned_usage_context, pr_number=pr_number
    )
    try:
        bootstrap_cwd = github_bootstrap_cwd(config)
        initial_pr_context = get_pr_review_context(
            runner,
            config=config,
            pr_number=pr_number,
            cwd=bootstrap_cwd,
        )
        # Resolve the live PR base before any managed-CI authentication or
        # workdir setup.  An inherited repository default is provenance, not
        # permission to reinterpret a PR targeting another branch.
        config = resolve_base_branch(
            config,
            runner,
            pr_metadata=initial_pr_context.metadata,
            cwd=bootstrap_cwd,
        )
        # A successful manual qualification retains the managed label on the
        # ready PR.  Release it before any managed-CI authentication so every
        # downstream path sees the ready/unlabeled state it already handles.
        if release_retained_managed_label(
            runner, config=config, pr_number=pr_number, cwd=bootstrap_cwd,
        ):
            initial_pr_context = get_pr_review_context(
                runner,
                config=config,
                pr_number=pr_number,
                cwd=bootstrap_cwd,
            )
        issue_context_refreshed = False
        parent_issue_context_refreshed = False
        # A caller-provided issue snapshot may predate plan approval. Refresh
        # both the child and the authoritative in-process parent snapshot
        # before any managed-CI canonical plan recovery reads their comments,
        # so a stale snapshot cannot defeat the guarded parent fallback.  Each
        # issue is fetched at most once per run; the later refresh block is
        # guarded by these flags.
        if issue_context is not None:
            issue_context = get_issue_context(
                runner, config=config, issue_number=issue_context.number
            )
            issue_context_refreshed = True
        if parent_issue_context is not None:
            parent_issue_context = get_issue_context(
                runner, config=config, issue_number=parent_issue_context.number
            )
            parent_issue_context_refreshed = True
        if config.managed_ci_fresh_authorization:
            fresh_issue_number = managed_ci_issue_number or config.managed_ci_issue_number
            if fresh_issue_number is None:
                raise AgentLoopError(
                    "Managed-CI fresh authorization requires an explicit issue scope in PR mode."
                )
            config = dataclasses_replace(
                config,
                managed_ci_issue_number=fresh_issue_number,
            )
            if issue_context is None:
                validate_open_issue(
                    runner, config=config, issue_number=fresh_issue_number
                )
                issue_context = get_issue_context(
                    runner, config=config, issue_number=fresh_issue_number
                )
                issue_context_refreshed = True
            elif issue_context.number != fresh_issue_number:
                raise AgentLoopError(
                    "Managed-CI fresh authorization issue scope does not match the "
                    "authenticated issue context."
                )
            canonical_handoff = find_latest_issue_pr_handoff(
                issue_context.comments,
                issue_number=fresh_issue_number,
                repo=config.repo,
            )
            fresh_retired_plan_hashes: frozenset[str] = frozenset()
            if canonical_handoff is not None:
                if canonical_handoff.pr_number != pr_number:
                    raise AgentLoopError(
                        "Managed-CI fresh authorization issue scope is already bound to "
                        "a different canonical PR."
                    )
                if canonical_handoff.flow == "approved-plan-implementation":
                    if not canonical_handoff.plan_hash:
                        raise AgentLoopError(
                            "Managed-CI fresh authorization found an approved-plan handoff "
                            "without a canonical plan identity."
                        )
                    _require_handoff_plan_growth_verdict(
                        config,
                        canonical_handoff,
                        context="Managed-CI fresh authorization",
                    )
                    recovered = _recover_managed_ci_approved_plan(
                        issue_context.comments,
                        expected_hash=canonical_handoff.plan_hash,
                        parent_comments=(
                            parent_issue_context.comments
                            if parent_issue_context is not None
                            else None
                        ),
                    )
                    if not recovered.is_available:
                        raise AgentLoopError(
                            "Managed-CI fresh authorization could not recover the canonical "
                            "approved plan for the explicit issue scope."
                        )
                    if (
                        approved_plan_context is not None
                        and approved_plan_context.plan_hash != recovered.plan_hash
                    ):
                        raise AgentLoopError(
                            "Managed-CI fresh authorization approved-plan scope does not "
                            "match the canonical issue plan."
                        )
                    approved_plan_context = recovered
                    (
                        fresh_retired_plan_hashes,
                        fetched_parent_issue_context,
                    ) = _managed_ci_retired_plan_hashes(
                        runner,
                        config=config,
                        issue_context=issue_context,
                        parent_issue_context=parent_issue_context,
                        pr_number=pr_number,
                    )
                    if fetched_parent_issue_context is not parent_issue_context:
                        parent_issue_context = fetched_parent_issue_context
                        parent_issue_context_refreshed = True
            else:
                resumed_plan = _resume_plan_round(
                    issue_context.comments,
                    configured_reviewers=reviewers(config),
                )
                if resumed_plan is not None:
                    plan_text, resumed_plan_round = resumed_plan
                    _require_complete_canonical_plan_approval(
                        issue_context.comments,
                        config=config,
                        plan_text=plan_text,
                        plan_round=resumed_plan_round,
                        human_requirements=issue_context.human_requirements,
                        error_message=(
                            "Managed-CI fresh authorization found planning state without "
                            "a complete canonical reviewer approval."
                            + _pr_amendment_plan_board_hint(
                                initial_pr_context.comments,
                                pr_number=pr_number,
                                supplied_reviewers=reviewers(config),
                            )
                        ),
                    )
                    recovered_plan_context = make_approved_plan_context(
                        plan_text,
                        source_locator=(
                            f"issue #{fresh_issue_number} canonical approved plan"
                        ),
                        expected_hash=approved_plan_hash(plan_text),
                    )
                    if (
                        approved_plan_context is not None
                        and approved_plan_context.plan_hash
                        != recovered_plan_context.plan_hash
                    ):
                        raise AgentLoopError(
                            "Managed-CI fresh authorization approved-plan scope does not "
                            "match the canonical issue plan."
                        )
                    approved_plan_context = recovered_plan_context
                elif approved_plan_context is not None or _extract_round_metadata_records(
                    issue_context.comments, flow="plan"
                ):
                    raise AgentLoopError(
                        "Managed-CI fresh authorization could not prove a complete canonical "
                        "approved plan for the explicit issue scope."
                    )
            managed_ci_handoff = authorize_fresh_issue_created_resume(
                runner,
                config=config,
                pr_number=pr_number,
                issue_number=fresh_issue_number,
                metadata=initial_pr_context.metadata,
                approved_plan_hash=(
                    approved_plan_context.plan_hash
                    if approved_plan_context is not None else None
                ),
                retired_plan_hashes=fresh_retired_plan_hashes,
            )
            authenticated_managed_resume = AuthenticatedManagedResume(
                origin="issue-created",
                lifecycle=managed_ci_handoff.lifecycle,
                issue_created_handoff=managed_ci_handoff,
                override_nonce=managed_ci_handoff.override_nonce,
            )
        if managed_ci_handoff is not None:
            managed_ci_handoff = revalidate_issue_created_handoff(
                runner,
                config=config,
                handoff=managed_ci_handoff,
                metadata=initial_pr_context.metadata,
            )
            authenticated_managed_resume = AuthenticatedManagedResume(
                origin="issue-created",
                lifecycle=managed_ci_handoff.lifecycle,
                issue_created_handoff=managed_ci_handoff,
                override_nonce=managed_ci_handoff.override_nonce,
            )
        if managed_pr_origin is None:
            recovered_origin = recover_managed_pr_origin(
                initial_pr_context.metadata.body or "",
                fetched_head_branch=initial_pr_context.metadata.head_branch,
            )
            if recovered_origin is not None:
                managed_pr_origin = recovered_origin
                managed_pr_recovered = True
            elif managed_ci_handoff is None:
                managed_ci_handoff = recover_issue_created_handoff(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    metadata=initial_pr_context.metadata,
                    issue_number=managed_ci_issue_number,
                )
                if managed_ci_handoff is None:
                    reject_forged_protocol_markers(
                        initial_pr_context.metadata.body or "",
                        surface=f"pull-request #{pr_number} body",
                    )
                else:
                    # Ordinary PR recovery must bind authorization records to
                    # the canonical server-side issue/plan scope just as the
                    # explicit fresh path does.  The record's own plan field
                    # is never allowed to define that scope.
                    if issue_context is None:
                        validate_open_issue(
                            runner,
                            config=config,
                            issue_number=managed_ci_handoff.issue_number,
                        )
                        issue_context = get_issue_context(
                            runner,
                            config=config,
                            issue_number=managed_ci_handoff.issue_number,
                        )
                        issue_context_refreshed = True
                    canonical_handoff = find_latest_issue_pr_handoff(
                        issue_context.comments,
                        issue_number=managed_ci_handoff.issue_number,
                        repo=config.repo,
                    )
                    recovered_scope: ApprovedPlanContext | None = None
                    if (
                        canonical_handoff is not None
                        and canonical_handoff.flow == "approved-plan-implementation"
                        and canonical_handoff.plan_hash
                    ):
                        _require_handoff_plan_growth_verdict(
                            config,
                            canonical_handoff,
                            context="Managed-CI ordinary resume",
                        )
                        candidate_scope = _recover_managed_ci_approved_plan(
                            issue_context.comments,
                            expected_hash=canonical_handoff.plan_hash,
                            parent_comments=(
                                parent_issue_context.comments
                                if parent_issue_context is not None
                                else None
                            ),
                        )
                        if not candidate_scope.is_available:
                            raise AgentLoopError(
                                "Managed-CI ordinary resume could not recover the canonical approved plan."
                            )
                        recovered_scope = candidate_scope
                    elif canonical_handoff is None:
                        resumed_plan = _resume_plan_round(
                            issue_context.comments,
                            configured_reviewers=reviewers(config),
                        )
                        if resumed_plan is not None:
                            plan_text, resumed_plan_round = resumed_plan
                            _require_complete_canonical_plan_approval(
                                issue_context.comments,
                                config=config,
                                plan_text=plan_text,
                                plan_round=resumed_plan_round,
                                human_requirements=issue_context.human_requirements,
                                error_message=(
                                    "Managed-CI ordinary resume found incomplete canonical "
                                    "plan approval."
                                    + _pr_amendment_plan_board_hint(
                                        initial_pr_context.comments,
                                        pr_number=pr_number,
                                        supplied_reviewers=reviewers(config),
                                    )
                                ),
                            )
                            recovered_scope = make_approved_plan_context(
                                plan_text,
                                source_locator=(
                                    f"issue #{managed_ci_handoff.issue_number} canonical approved plan"
                                ),
                                expected_hash=approved_plan_hash(plan_text),
                            )
                    if recovered_scope is not None:
                        if (
                            approved_plan_context is not None
                            and approved_plan_context.plan_hash != recovered_scope.plan_hash
                        ):
                            raise AgentLoopError(
                                "Managed-CI ordinary resume approved-plan scope does not match "
                                "the canonical issue plan."
                            )
                        approved_plan_context = recovered_scope
                        ordinary_retired: frozenset[str] = frozenset()
                        if canonical_handoff is not None:
                            # A verified signed rebind leaves grants under the
                            # superseded plan on the PR as history (#993).
                            (
                                ordinary_retired,
                                fetched_parent_issue_context,
                            ) = _managed_ci_retired_plan_hashes(
                                runner,
                                config=config,
                                issue_context=issue_context,
                                parent_issue_context=parent_issue_context,
                                pr_number=pr_number,
                            )
                            if fetched_parent_issue_context is not parent_issue_context:
                                parent_issue_context = fetched_parent_issue_context
                                parent_issue_context_refreshed = True
                        managed_ci_handoff = dataclasses_replace(
                            managed_ci_handoff,
                            approved_plan_hash=recovered_scope.plan_hash,
                            retired_plan_hashes=(
                                frozenset()
                                if recovered_scope.plan_hash in ordinary_retired
                                else ordinary_retired
                            ),
                        )
                    managed_ci_handoff = revalidate_issue_created_handoff(
                        runner,
                        config=config,
                        handoff=managed_ci_handoff,
                        metadata=initial_pr_context.metadata,
                    )
                    authenticated_managed_resume = AuthenticatedManagedResume(
                        origin="issue-created",
                        lifecycle=managed_ci_handoff.lifecycle,
                        issue_created_handoff=managed_ci_handoff,
                        override_nonce=managed_ci_handoff.override_nonce,
                    )
        if managed_pr_origin is not None:
            source_branch, source_sha, managed_branch, override_nonce = managed_pr_origin
            validate_managed_pr_body(
                initial_pr_context.metadata.body or "",
                source_branch=source_branch,
                source_sha=source_sha,
                managed_branch=managed_branch,
                override_nonce=override_nonce,
                fetched_head_sha=initial_pr_context.metadata.head_sha,
                fetched_head_branch=initial_pr_context.metadata.head_branch,
                fetched_base_branch=initial_pr_context.metadata.base_branch,
                expected_base_branch=config.base,
                require_creation_head_sha=not managed_pr_recovered,
            )
            if managed_pr_recovered and authenticated_managed_resume is None and config.effective_managed_ci:
                authenticated_managed_resume = authenticate_source_managed_resume(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    source_branch=source_branch,
                    source_sha=source_sha,
                    managed_branch=managed_branch,
                    override_nonce=override_nonce,
                )
        recorded_pr_contract = find_latest_pr_contract(
            initial_pr_context.comments,
            repository=config.repo,
            pr_number=pr_number,
        )
        # A caller-provided issue snapshot may predate plan approval. Refresh
        # it before deriving requirements or handoff provenance.  The managed-CI
        # recovery block above already refreshed whatever it read, so these
        # fetches are flag-guarded to keep each issue fetched once per run.
        if issue_context is not None and not issue_context_refreshed:
            issue_context = get_issue_context(
                runner, config=config, issue_number=issue_context.number
            )
            issue_context_refreshed = True
        if parent_issue_context is not None and not parent_issue_context_refreshed:
            parent_issue_context = get_issue_context(
                runner, config=config, issue_number=parent_issue_context.number
            )
            parent_issue_context_refreshed = True
        # Public PR-mode recovery has no issue-mode configuration bit carrying
        # the immutable closing contract.  Once the issue-created managed
        # tuple has been authenticated, derive that contract from the
        # server-backed issue/approved-plan scope before activation.  PR body
        # references remain validation evidence only and never participate in
        # this resolution.
        if (
            config.managed_ci
            and config.expected_closing_issue_ids is None
            and authenticated_managed_resume is not None
            and authenticated_managed_resume.origin == "issue-created"
            and issue_context is not None
        ):
            issue_scope_handoff = find_latest_issue_pr_handoff(
                issue_context.comments,
                issue_number=issue_context.number,
                repo=config.repo,
            )
            if issue_scope_handoff is not None and issue_scope_handoff.pr_number != pr_number:
                issue_scope_handoff = None
            plan_additions = (
                _extract_current_expected_closing_issue_ids(
                    approved_plan_context.canonical_text
                )
                if approved_plan_context is not None
                and approved_plan_context.canonical_text
                else None
            )
            if (
                recorded_pr_contract is not None
                and issue_scope_handoff is not None
                and tuple(recorded_pr_contract.expected_closing_issue_ids)
                != tuple(issue_scope_handoff.expected_closing_issue_ids)
            ):
                raise AgentLoopError(
                    "Authenticated issue-side and PR-side expected closing contracts diverge; "
                    "no managed activation was performed."
                )
            closing_contract = resolve_issue_contract(
                primary_issue=issue_context.number,
                cli_additions=None,
                plan_additions=plan_additions,
                recovered=(
                    issue_scope_handoff.expected_closing_issue_ids
                    if issue_scope_handoff is not None
                    else (
                        recorded_pr_contract.expected_closing_issue_ids
                        if recorded_pr_contract is not None
                        else None
                    )
                ),
                supersede=config.supersede_expected_closing_contract,
            )
            reject_parent_from_contract(
                closing_contract,
                parent_issue=(
                    parent_issue_context.number
                    if parent_issue_context is not None
                    else None
                ),
            )
            config = dataclasses_replace(
                config,
                expected_closing_issue_ids=closing_contract.issue_ids,
                expected_closing_contract_resolved=True,
            )
        if config.expected_closing_contract_resolved:
            assert config.expected_closing_issue_ids is not None
            closing_contract = make_pr_contract(
                repository=config.repo,
                pr_number=pr_number,
                origin_flow=(
                    recorded_pr_contract.origin_flow
                    if recorded_pr_contract is not None
                    else (
                        config.pr_origin_flow
                        if issue_context is None
                        else "issue-implementation"
                    )
                ),
                primary_issue_number=(
                    recorded_pr_contract.primary_issue_number
                    if recorded_pr_contract is not None
                    else None if issue_context is None else issue_context.number
                ),
                expected_closing_issue_ids=config.expected_closing_issue_ids,
                supersedes_hash=(
                    recorded_pr_contract.supersedes_hash
                    if recorded_pr_contract is not None
                    else None
                ),
            )
            if recorded_pr_contract is not None and tuple(
                recorded_pr_contract.expected_closing_issue_ids
            ) != tuple(closing_contract.expected_closing_issue_ids):
                closing_contract = resolve_direct_contract(
                    explicit=closing_contract.expected_closing_issue_ids,
                    recovered=recorded_pr_contract.expected_closing_issue_ids,
                    supersede=config.supersede_expected_closing_contract,
                )
                assert closing_contract is not None
                closing_contract = make_pr_contract(
                    repository=config.repo,
                    pr_number=pr_number,
                    origin_flow=recorded_pr_contract.origin_flow,
                    primary_issue_number=recorded_pr_contract.primary_issue_number,
                    expected_closing_issue_ids=closing_contract.issue_ids,
                    supersedes_hash=closing_contract.supersedes_hash,
                )
        else:
            resolved_contract = resolve_direct_contract(
                explicit=config.expected_closing_issue_ids,
                recovered=(
                    recorded_pr_contract.expected_closing_issue_ids
                    if recorded_pr_contract is not None
                    else None
                ),
                supersede=config.supersede_expected_closing_contract,
            )
            closing_contract = (
                None
                if resolved_contract is None
                else make_pr_contract(
                    repository=config.repo,
                    pr_number=pr_number,
                    origin_flow=(
                        recorded_pr_contract.origin_flow
                        if recorded_pr_contract is not None
                        else (
                            config.pr_origin_flow
                            if issue_context is None
                            else "issue-implementation"
                        )
                    ),
                    primary_issue_number=(
                        recorded_pr_contract.primary_issue_number
                        if recorded_pr_contract is not None
                        else None if issue_context is None else issue_context.number
                    ),
                    expected_closing_issue_ids=resolved_contract.issue_ids,
                    supersedes_hash=(
                        recorded_pr_contract.supersedes_hash
                        if recorded_pr_contract is not None
                        and tuple(resolved_contract.issue_ids)
                        == tuple(recorded_pr_contract.expected_closing_issue_ids)
                        else resolved_contract.supersedes_hash
                    ),
                )
            )
        legacy_closing_validation_only = (
            issue_context is not None
            and recorded_pr_contract is None
            and config.expected_closing_contract_resolved
        )
        if legacy_closing_validation_only:
            if config.managed_ci:
                # A pre-contract canonical handoff is a legacy recovery
                # record. Do not retroactively turn its old Refs-only body
                # into a new durable contract. Keep the resolved contract for
                # closing-reference validation, but do not persist a new
                # PR-side contract solely from the invocation's recovery
                # context.
                log(
                    config,
                    f"PR #{pr_number}: no PR-side expected-closing record found; retaining legacy "
                    "handoff recovery without persisting a contract from prose",
                )
            else:
                # Preserve the ordinary legacy issue-resume compatibility
                # path: a pre-contract Refs-only PR is not retroactively
                # upgraded into an affirmative closing contract.
                closing_contract = None
        contract_needs_persisting = (
            closing_contract is not None
            and (recorded_pr_contract is None or recorded_pr_contract != closing_contract)
            and not legacy_closing_validation_only
        )
        issue_handoff_to_update = None
        if issue_context is not None and recorded_pr_contract is not None:
            issue_handoff_to_update = find_latest_issue_pr_handoff(
                issue_context.comments,
                issue_number=issue_context.number,
                repo=config.repo,
            )
            if (
                contract_needs_persisting
                and recorded_pr_contract.origin_flow
                in {"issue-implementation", "approved-plan-implementation"}
                and issue_handoff_to_update is not None
                and tuple(issue_handoff_to_update.expected_closing_issue_ids)
                != tuple(recorded_pr_contract.expected_closing_issue_ids)
            ):
                raise AgentLoopError(
                    "Issue-side and PR-side expected closing contracts disagree before "
                    "supersession; no durable metadata changed."
                )
        if issue_context is None:
            contract_primary_issue = (
                recorded_pr_contract.primary_issue_number
                if recorded_pr_contract is not None
                else None
            )
            linked_issue_numbers = parse_linked_issue_numbers(
                initial_pr_context.metadata.body,
                repo=config.repo,
            )
            if contract_primary_issue is not None:
                linked_issue_number = contract_primary_issue
                log(
                    config,
                    f"PR #{pr_number} contract selects primary issue #{linked_issue_number}; "
                    "using it for approved-plan provenance even if the PR body references other issues",
                )
                issue_context = get_issue_context(
                    runner, config=config, issue_number=linked_issue_number
                )
                issue_context_refreshed = True
            elif len(linked_issue_numbers) == 1:
                linked_issue_number = linked_issue_numbers[0]
                log(
                    config,
                    f"PR #{pr_number} references issue #{linked_issue_number}; "
                    "including linked issue context in review prompts",
                )
                issue_context = get_issue_context(
                    runner, config=config, issue_number=linked_issue_number
                )
                issue_context_refreshed = True
            elif linked_issue_numbers:
                candidates = ", ".join(f"#{number}" for number in linked_issue_numbers)
                log(
                    config,
                    f"PR #{pr_number} references multiple issues ({candidates}); "
                    "linked issue context is ambiguous and will not be included",
                )
            else:
                log(config, f"PR #{pr_number} has no linked issue context to include in review prompts")

        # Recover an approved plan only from the exact issue-side handoff that
        # names this PR.  Newer planning comments cannot replace that binding.
        if issue_context is not None:
            staged_parent_number = _infer_staged_parent_issue(issue_context)
            if staged_parent_number is not None and staged_parent_number != issue_context.number:
                if parent_issue_context is None:
                    parent_issue_context = get_issue_context(
                        runner, config=config, issue_number=staged_parent_number
                    )
                    parent_issue_context_refreshed = True
            issue_handoff = find_latest_issue_pr_handoff(
                issue_context.comments,
                issue_number=issue_context.number,
                repo=config.repo,
            )
            # These values are populated only for a validated decomposition
            # phase marker.  Keep ordinary approved-plan resumes on the normal
            # full-matrix path without relying on branch-local state.
            fresh_phase = False
            phase_handoff = None
            handoff_disposition = None
            stable_stage_id = None
            normalized = None
            planning_child_binding = None
            if issue_handoff is not None and issue_handoff.pr_number != pr_number:
                plan_bound_handoff = (
                    issue_handoff.flow == "approved-plan-implementation"
                    or (
                        recorded_pr_contract is not None
                        and recorded_pr_contract.origin_flow == "approved-plan-implementation"
                    )
                    or approved_plan_context is not None
                )
                if plan_bound_handoff:
                    raise AgentLoopError(
                        f"Issue #{issue_context.number} handoff selects PR #{issue_handoff.pr_number}, "
                        f"not PR #{pr_number}; review the recorded PR directly or repair the handoff."
                    )
                log(
                    config,
                    f"Issue #{issue_context.number} has an older direct implementation handoff "
                    f"for PR #{issue_handoff.pr_number}; it is unrelated to PR #{pr_number} "
                    "and will not gate this ordinary direct review.",
                )
                issue_handoff = None
            if issue_handoff is not None:
                if recorded_pr_contract is not None and (
                    recorded_pr_contract.primary_issue_number != issue_handoff.issue_number
                    or recorded_pr_contract.origin_flow != issue_handoff.flow
                ):
                    raise AgentLoopError(
                        "Issue-side and PR-side handoff provenance disagree on primary issue or flow."
                    )
                if issue_handoff.flow == "approved-plan-implementation":
                    if not issue_handoff.plan_hash:
                        raise AgentLoopError(
                            f"Approved-plan handoff for issue #{issue_context.number} has no plan hash."
                        )
                    _require_handoff_plan_growth_verdict(
                        config,
                        issue_handoff,
                        context=f"PR #{pr_number} recovery",
                    )
                    if approved_plan_context is not None and approved_plan_context.plan_hash != issue_handoff.plan_hash:
                        raise AgentLoopError(
                            f"PR #{pr_number} received approved plan {approved_plan_context.plan_hash}, "
                            f"but the issue-side handoff requires {issue_handoff.plan_hash}."
                        )
                    if parent_issue_context is not None:
                        child_bodies = [issue_context.body or ""] + [
                            comment.body or "" for comment in issue_context.comments
                        ]
                        split_child = next(
                            (SPLIT_CHILD_MARKER_RE.search(body) for body in child_bodies if SPLIT_CHILD_MARKER_RE.search(body)),
                            None,
                        )
                        if split_child is not None:
                            split_parent = int(split_child.group("parent"))
                            if split_parent != parent_issue_context.number:
                                raise AgentLoopError(
                                    "Validated split-child marker names a different parent issue."
                                )
                            stage_handoff = find_existing_split_stage_handoff(
                                parent_issue_context.comments,
                                parent_issue=parent_issue_context.number,
                                plan_hash=issue_handoff.plan_hash,
                            )
                            if stage_handoff is None or stage_handoff.child_issue_number != issue_context.number:
                                raise AgentLoopError(
                                    "Split child has no matching parent handoff selecting this exact child."
                                )
                        phase_marker = next(
                            (PHASE_IDENTITY_MARKER_RE.search(body) for body in child_bodies if PHASE_IDENTITY_MARKER_RE.search(body)),
                            None,
                        )
                        if phase_marker is not None:
                            phase_payload = _decode_json_payload(
                                phase_marker.group("payload"),
                                marker_name="AGENT_PLAN_PHASE_IDENTITY",
                            )
                            # A separately planned child has its own implementation
                            # hash; phase membership remains bound to the parent plan.
                            phase_plan_hash = phase_payload.get("plan_hash")
                            phase_source = phase_payload.get("source")
                            fresh_phase = phase_source == EXECUTION_TOPOLOGY_SOURCE
                            if fresh_phase:
                                phase_index = phase_payload.get("phase_index")
                                stable_stage_id = phase_payload.get("stage_id")
                                phase_strategy = phase_payload.get("strategy")
                                phase_contract = phase_payload.get(
                                    "execution_strategy_contract_version"
                                )
                                phase_digest = phase_payload.get("recommendation_digest")
                                if (
                                    not isinstance(phase_index, int)
                                    or isinstance(phase_index, bool)
                                    or phase_index < 1
                                    or not isinstance(stable_stage_id, str)
                                    or not stable_stage_id.strip()
                                    or phase_strategy != "staged"
                                    or phase_contract != 1
                                    or not isinstance(phase_digest, str)
                                    or not phase_digest
                                ):
                                    raise AgentLoopError(
                                        "Fresh decomposition child phase identity must include a valid "
                                        "ordinal, stable stage ID, staged strategy, contract version, and digest."
                                    )
                            else:
                                phase_index = phase_payload.get("stage_id")
                                stable_stage_id = None
                            if (
                                phase_payload.get("parent_issue") != parent_issue_context.number
                                or not isinstance(phase_plan_hash, str)
                                or not phase_plan_hash
                                or not isinstance(phase_index, int)
                                or isinstance(phase_index, bool)
                            ):
                                raise AgentLoopError(
                                    "Decomposition child phase identity must name the validated parent issue, "
                                    "a non-empty plan hash, and a valid phase position."
                                )
                            if fresh_phase:
                                parent_plan_context = recover_approved_plan_context(
                                    parent_issue_context.comments,
                                    expected_hash=phase_plan_hash,
                                )
                                if not parent_plan_context.is_available or not parent_plan_context.canonical_text:
                                    raise AgentLoopError(
                                        "Fresh decomposition child phase cannot recover the approved parent plan."
                                    )
                                recommendation = recover_execution_recommendation(
                                    parent_issue_context.comments,
                                    expected_digest=phase_digest,
                                )
                                normalized, _retained = normalize_execution_recommendation(
                                    recommendation,
                                    approved_plan=parent_plan_context.canonical_text,
                                    plan_subject=parent_plan_context.plan_subject or "",
                                )
                                if phase_index > len(normalized.phases):
                                    raise AgentLoopError(
                                        "Fresh decomposition child phase is outside the approved topology."
                                    )
                                phase = normalized.phases[phase_index - 1]
                                if phase.stage_id != stable_stage_id:
                                    raise AgentLoopError(
                                        "Fresh decomposition child phase stable ID disagrees with its ordinal."
                                    )
                                parent_matrix_row_ids = risk_matrix_row_ids_for_owner(
                                    parent_plan_context.risk_test_matrix_payload
                                    if parent_plan_context.matrix_available else None,
                                    stable_stage_id,
                                )
                                marker_row_ids = phase_payload.get(
                                    "inherited_matrix_row_ids", []
                                )
                                if not isinstance(marker_row_ids, list) or any(
                                    not isinstance(item, str) for item in marker_row_ids
                                ) or tuple(marker_row_ids) != parent_matrix_row_ids:
                                    raise AgentLoopError(
                                        "Fresh decomposition child phase inherited matrix rows "
                                        "do not match the approved parent owner allocation."
                                    )
                                expected_identity = phase_identity(
                                    parent_issue=parent_issue_context.number,
                                    plan_hash=phase_plan_hash,
                                    topology_source=EXECUTION_TOPOLOGY_SOURCE,
                                    phase_index=phase_index,
                                    phase=phase,
                                    stage_id=stable_stage_id,
                                    execution_strategy_contract_version=1,
                                )
                                if (
                                    phase_payload.get("identity") != expected_identity
                                    or phase_payload.get("recommendation_digest")
                                    != normalized.recommendation_digest
                                ):
                                    raise AgentLoopError(
                                        "Fresh decomposition child phase identity does not match the approved parent topology."
                                    )
                                summary = find_existing_decomposition(
                                    parent_issue_context.comments,
                                    parent_issue=parent_issue_context.number,
                                    plan_hash=phase_plan_hash,
                                    strategy="staged",
                                    topology_source=EXECUTION_TOPOLOGY_SOURCE,
                                    recommendation_digest=normalized.recommendation_digest,
                                    plan_subject=parent_plan_context.plan_subject,
                                )
                                if summary is None or summary.mode not in {
                                    "decompose-only", "implement-by-phase"
                                }:
                                    raise AgentLoopError(
                                        "Fresh decomposition child phase has no matching canonical parent topology summary."
                                    )
                                if (
                                    len(summary.phase_identities) != len(normalized.phases)
                                    or summary.phase_identities[phase_index - 1] != expected_identity
                                    or summary.stage_ids[phase_index - 1] != stable_stage_id
                                ):
                                    raise AgentLoopError(
                                        "Fresh decomposition child phase disagrees with the parent topology summary."
                                    )
                                # The decomposition summary records how the
                                # topology was first materialized and is
                                # intentionally not rewritten when a later
                                # implement-by-phase invocation dispatches a
                                # child. Inventory parent handoffs by phase
                                # instead of using that immutable mode.
                                phase_handoffs = tuple(
                                    handoff
                                    for handoff in find_phase_implementation_handoffs_for_parent(
                                        parent_issue_context.comments,
                                        parent_issue=parent_issue_context.number,
                                    )
                                    if (
                                        handoff.phase_index == phase_index
                                        or handoff.stage_id == stable_stage_id
                                        or handoff.child_issue_number == issue_context.number
                                    )
                                )
                                if len(phase_handoffs) > 1:
                                    raise AgentLoopError(
                                        "Fresh decomposition child phase has multiple parent "
                                        "implementation handoffs for the same phase."
                                    )
                                phase_handoff = phase_handoffs[0] if phase_handoffs else None
                                if phase_handoff is not None and tuple(
                                    phase_handoff.inherited_matrix_row_ids
                                ) != parent_matrix_row_ids:
                                    raise AgentLoopError(
                                        "Fresh decomposition child implementation handoff has an "
                                        "unbound or incomplete parent matrix assignment."
                                    )
                                if phase_handoff is not None:
                                    # Shared reconciliation rule (#808): the
                                    # PR binds to the reconciled handoff
                                    # disposition, never to the phase alone.
                                    stage_overrides = collect_child_disposition_overrides(
                                        parent_comments=parent_issue_context.comments,
                                        child_comments=issue_context.comments,
                                        parent_issue=parent_issue_context.number,
                                        plan_hash=phase_plan_hash,
                                        topology_stage_ids=tuple(
                                            item.stage_id or "" for item in normalized.phases
                                        ),
                                        routed_stage_id=stable_stage_id,
                                        child_stage_id=stable_stage_id,
                                        child_issue_number=issue_context.number,
                                    )
                                    handoff_disposition = reconcile_handoff_disposition(
                                        phase, phase_handoff, stage_overrides
                                    )
                                if phase_handoff is not None and (
                                    phase_handoff.plan_hash != phase_plan_hash
                                    or phase_handoff.mode != "implement-by-phase"
                                    or phase_handoff.child_issue_number != issue_context.number
                                    or phase_handoff.phase_index != phase_index
                                    or (
                                        handoff_disposition == EXECUTION_DISPOSITION_DIRECT
                                        and issue_handoff is not None
                                        and issue_handoff.plan_hash != phase_plan_hash
                                    )
                                    or phase_handoff.strategy != "staged"
                                    or phase_handoff.topology_source != EXECUTION_TOPOLOGY_SOURCE
                                    or phase_handoff.execution_strategy_contract_version != 1
                                    or phase_handoff.recommendation_digest != normalized.recommendation_digest
                                    or phase_handoff.stage_id != stable_stage_id
                                    or phase_handoff.plan_subject != parent_plan_context.plan_subject
                                    or phase_handoff.phase_title != phase.title
                                    or phase_handoff.automation != phase.automation
                                ):
                                    raise AgentLoopError(
                                        "Fresh decomposition child phase implementation handoff "
                                        "disagrees with the parent topology."
                                    )
                                if summary.mode == "implement-by-phase" and phase_handoff is None:
                                    raise AgentLoopError(
                                        "Fresh decomposition child phase has no matching canonical "
                                        "implementation handoff."
                                    )
                                if handoff_disposition == EXECUTION_DISPOSITION_PLANNING:
                                    # A planning child's PR binds to its
                                    # reviewed child plan hash; the parent
                                    # identity, summary, and inherited rows
                                    # were validated above.
                                    if (
                                        issue_handoff is None
                                        or not issue_handoff.plan_hash
                                        or issue_handoff.plan_hash == phase_plan_hash
                                    ):
                                        raise AgentLoopError(
                                            "Fresh decomposition child phase was routed to child "
                                            "planning, but its issue-to-PR handoff copies the parent "
                                            "phase hash instead of a reviewed child plan hash; a "
                                            "parent-hash copy is never child provenance."
                                        )
                                    child_plan_context = recover_approved_plan_context(
                                        issue_context.comments,
                                        expected_hash=issue_handoff.plan_hash,
                                    )
                                    if (
                                        not child_plan_context.is_available
                                        or not child_plan_context.canonical_text
                                    ):
                                        raise AgentLoopError(
                                            "Fresh decomposition child phase was routed to child "
                                            f"planning, but no reviewed approved child plan "
                                            f"{issue_handoff.plan_hash} is recoverable on the child "
                                            "issue; repair the child plan round or the issue-to-PR "
                                            "provenance."
                                        )
                                    planning_child_binding = _PlanningChildBinding(
                                        child_issue=issue_context.number,
                                        parent_issue=parent_issue_context.number,
                                        stage_id=stable_stage_id,
                                        parent_plan_context=parent_plan_context,
                                    )
                                    _require_admissible_pr_child_plan(
                                        issue_context.comments,
                                        config=config,
                                        binding=planning_child_binding,
                                        child_plan_context=child_plan_context,
                                        pr_number=pr_number,
                                    )
                                    approved_plan_context = child_plan_context
                                if phase_handoff is None:
                                    # A fresh child phase without a parent-owned
                                    # implementation handoff is independently
                                    # planned.  Its issue-to-PR handoff must
                                    # therefore bind to an approved plan on the
                                    # child issue itself.  Do not let the later
                                    # parent-plan recovery fallback turn a
                                    # copied parent hash into provenance.
                                    child_plan_context = recover_approved_plan_context(
                                        issue_context.comments,
                                        expected_hash=(
                                            issue_handoff.plan_hash
                                            if issue_handoff is not None
                                            else None
                                        ),
                                    )
                                    if (
                                        not child_plan_context.is_available
                                        or not (
                                            child_plan_context.canonical_text
                                            or child_plan_context.matrix_available
                                        )
                                    ):
                                        raise AgentLoopError(
                                            "Fresh decomposition child phase has no parent "
                                            "implementation handoff and no recoverable approved "
                                            "child plan; repair the child issue-to-PR provenance."
                                        )
                                    planning_child_binding = _PlanningChildBinding(
                                        child_issue=issue_context.number,
                                        parent_issue=parent_issue_context.number,
                                        stage_id=stable_stage_id,
                                        parent_plan_context=parent_plan_context,
                                    )
                                    _require_admissible_pr_child_plan(
                                        issue_context.comments,
                                        config=config,
                                        binding=planning_child_binding,
                                        child_plan_context=child_plan_context,
                                        pr_number=pr_number,
                                    )
                                    # The child has its own approved plan, so
                                    # its matrix owners are authoritative for
                                    # this PR. The parent stage ID is only the
                                    # binding used to validate inherited row
                                    # semantics; using it to scope the child
                                    # payload would make a valid one-shot child
                                    # (whose rows are owned by `one-shot`)
                                    # unenforceable. All applicable rows in
                                    # the separately approved child plan,
                                    # including inherited and child-local rows,
                                    # remain enforceable here.
                                    approved_plan_context = child_plan_context
                            else:
                                checkpoint = None
                                for topology_mode in ("decompose-only", "implement-by-phase"):
                                    checkpoint = find_existing_topology_checkpoint(
                                        parent_issue_context.comments,
                                        parent_issue=parent_issue_context.number,
                                        plan_hash=phase_plan_hash,
                                        mode=topology_mode,
                                    )
                                    if checkpoint is not None:
                                        break
                                stage_id = phase_index
                                if checkpoint is None or not 1 <= stage_id <= len(checkpoint.phases):
                                    raise AgentLoopError(
                                        "Decomposition child phase is not covered by a matching parent topology checkpoint."
                                    )
                                expected_identity = phase_identity(
                                    parent_issue=checkpoint.parent_issue,
                                    plan_hash=checkpoint.plan_hash,
                                    topology_source=checkpoint.topology_source,
                                    phase_index=stage_id,
                                    phase=checkpoint.phases[stage_id - 1],
                                )
                                if phase_payload.get("identity") != expected_identity:
                                    raise AgentLoopError(
                                        "Decomposition child phase identity does not match the parent topology checkpoint."
                                    )
                    if approved_plan_context is None or not approved_plan_context.is_available:
                        expected_plan_subject = None
                        one_shot_record = find_latest_one_shot_impl_handoff(
                            issue_context.comments,
                            parent_issue=issue_context.number,
                            mode="implement-one-shot",
                        )
                        if (
                            one_shot_record is not None
                            and one_shot_record.plan_hash == issue_handoff.plan_hash
                        ):
                            expected_plan_subject = one_shot_record.plan_subject or None
                        if approved_plan_context is None or not approved_plan_context.is_available:
                            approved_plan_context = recover_approved_plan_context(
                                issue_context.comments,
                                expected_hash=issue_handoff.plan_hash,
                                expected_subject=expected_plan_subject,
                            )
                            if (
                                not approved_plan_context.is_available
                                and not approved_plan_context.has_matching_candidate
                                and parent_issue_context is not None
                            ):
                                parent_candidate = recover_approved_plan_context(
                                    parent_issue_context.comments,
                                    expected_hash=issue_handoff.plan_hash,
                                    expected_subject=expected_plan_subject,
                                )
                                if parent_candidate.is_available:
                                    approved_plan_context = parent_candidate
                    if (
                        fresh_phase
                        and phase_handoff is not None
                        and handoff_disposition == EXECUTION_DISPOSITION_DIRECT
                        and approved_plan_context.matrix_available
                    ):
                        approved_plan_context = scope_approved_plan_matrix(
                            approved_plan_context,
                            execution_owner=stable_stage_id,
                            valid_stage_ids=tuple(stage.stage_id for stage in normalized.phases),
                        )
                    if not approved_plan_context.is_available:
                        raise AgentLoopError(
                            f"PR #{pr_number} is bound to approved plan {issue_handoff.plan_hash}, "
                            f"but the canonical plan could not be recovered: "
                            f"{approved_plan_context.diagnostic or 'no diagnostic available'} "
                            "Restore the plan round metadata or resume after repairing the issue handoff."
                        )
                    if (
                        approved_plan_context.is_available
                        and approved_plan_context.plan_hash != issue_handoff.plan_hash
                    ):
                        raise AgentLoopError(
                            "Recovered approved plan hash does not match the issue-side handoff."
                        )
            elif (
                recorded_pr_contract is not None
                and recorded_pr_contract.origin_flow == "approved-plan-implementation"
                and (
                    approved_plan_context is None
                    or not approved_plan_context.is_available
                )
            ):
                raise AgentLoopError(
                    f"PR #{pr_number} declares approved-plan provenance but issue #{issue_context.number} "
                    "has no matching issue-side handoff. Reconcile the handoff before reviewing."
                )
        if (
            closing_contract is not None
            and contract_needs_persisting
            and recorded_pr_contract is not None
            and recorded_pr_contract.origin_flow
            in {"issue-implementation", "approved-plan-implementation"}
            and issue_context is not None
            and issue_handoff_to_update is None
        ):
            issue_handoff_to_update = find_latest_issue_pr_handoff(
                issue_context.comments,
                issue_number=issue_context.number,
                repo=config.repo,
            )
            if (
                issue_handoff_to_update is not None
                and tuple(issue_handoff_to_update.expected_closing_issue_ids)
                != tuple(recorded_pr_contract.expected_closing_issue_ids)
            ):
                raise AgentLoopError(
                    "Issue-side and PR-side expected closing contracts disagree before "
                    "supersession; no durable metadata changed."
                )
        if (
            config.managed_ci
            and authenticated_managed_resume is not None
            and authenticated_managed_resume.origin == "issue-created"
            and closing_contract is not None
        ):
            # Reject an unapproved same-repository closing reference before
            # activation can apply the managed suppression label.  The later
            # per-round check remains necessary because the body can change
            # after this initial authentication boundary.
            validate_pr_expected_closing_issues(
                runner,
                config=config,
                pr_number=pr_number,
                expected_issue_ids=closing_contract.expected_closing_issue_ids,
                body=initial_pr_context.metadata.body,
                reject_unexpected=True,
            )
        if not workdirs_ready:
            ensure_agent_workdirs(config, runner)
        config = _freeze_prompt_architecture(
            runner,
            config,
            target_revision=initial_pr_context.metadata.base_branch,
            candidate_revision=initial_pr_context.metadata.head_sha,
            pr_pair=True,
        )
        if config.review_parallel:
            _ensure_parallel_reviewer_workdirs(config, flag_name="--review-parallel", role_label="reviewer")
        # Select managed activation before any PR-side contract/comment write.
        # In particular, an authenticated ready/unlabeled re-entry reached by
        # implicit auto-merge must be able to stop with the PR untouched.
        activation = activate_managed_ci(
            runner,
            config=config,
            pr_number=pr_number,
            metadata=initial_pr_context.metadata,
            managed_resume=authenticated_managed_resume,
            resume_origin=(
                "source-managed"
                if managed_pr_origin is not None and authenticated_managed_resume is None
                else None
            ),
        )
        if activation is not None and activation.activation_path == "ordinary_fallback":
            ordinary_recovery_selected = True
            ordinary_recovery = activation.ordinary_recovery
            managed_ci = None
            log(
                config,
                f"PR #{pr_number}: ordinary recovery selected; "
                "the previous managed activation is not being resumed"
                + (f". {activation.state_report}" if activation.state_report else ""),
            )
            if ordinary_recovery is None:
                if activation.state_report:
                    # #1067: after a mutation by this run, print the measured
                    # state instead of asserting that the PR is still draft.
                    print(
                        f"PR #{pr_number} was not merged because managed recovery provenance or "
                        f"an unlabeled CI route is unavailable. {activation.state_report}"
                    )
                    return 0
                command = render_managed_ci_resume_command(
                    config, pr_number=pr_number, managed_ci=True,
                )
                print(
                    f"PR #{pr_number} remains draft and unmerged because managed recovery provenance or "
                    f"an unlabeled CI route is unavailable. Resume with `{command}`."
                )
                return 0
        else:
            managed_ci = activation
        log(config, f"Validating PR #{pr_number}")
        validate_open_pr(runner, config=config, pr_number=pr_number)
        if closing_contract is not None:
            validate_pr_expected_closing_issues(
                runner,
                config=config,
                pr_number=pr_number,
                expected_issue_ids=closing_contract.expected_closing_issue_ids,
                body=initial_pr_context.metadata.body,
                reject_unexpected=config.managed_ci and issue_context is not None,
            )
            if contract_needs_persisting:
                post_trusted_pr_comment(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    body=TrustedBody.canonical(
                        format_pr_contract_comment(closing_contract),
                        expected_tokens=("AGENT_PR_EXPECTED_CLOSING_ISSUES",),
                    ),
                )
            if (
                contract_needs_persisting
                and recorded_pr_contract is not None
                and recorded_pr_contract.origin_flow
                in {"issue-implementation", "approved-plan-implementation"}
                and issue_context is not None
                and issue_handoff_to_update is not None
            ):
                if issue_context.number != recorded_pr_contract.primary_issue_number:
                    raise AgentLoopError(
                        "The linked issue context does not match the issue-origin PR contract; "
                        "resume with the authoritative issue or PR metadata."
                    )
                pr_url, pr_head_sha = require_pr_metadata_for_handoff(initial_pr_context.metadata)
                post_issue_pr_handoff_comment(
                    runner,
                    config=config,
                    issue_number=issue_context.number,
                    pr_number=pr_number,
                    pr_url=pr_url,
                    pr_head_sha=pr_head_sha,
                    flow=recorded_pr_contract.origin_flow,
                    plan_hash=issue_handoff_to_update.plan_hash,
                    expected_closing_issue_ids=closing_contract.expected_closing_issue_ids,
                    supersedes_hash=closing_contract.supersedes_hash,
                    # A closing-ID superset keeps the plan, so it keeps the
                    # plan's approval-time verdict (or its legacy absence).
                    plan_growth_verdict=(
                        issue_handoff_to_update.plan_growth_verdict
                        if recorded_pr_contract.origin_flow == "approved-plan-implementation"
                        else None
                    ),
                )
        def managed_ci_active(metadata: PullRequestMetadata) -> bool:
            """Drop adopted filtering immediately when its live handshake changes."""
            nonlocal managed_ci
            if managed_ci is None:
                return False
            if revalidate_adopted_managed_ci(
                runner, config=config, pr_number=pr_number, metadata=metadata, contract=managed_ci
            ):
                if managed_ci.origin in {"issue-created", "source-managed"} or managed_ci.issue_created_pr:
                    label_state = managed_label_present(
                        runner, config=config, pr_number=pr_number,
                    )
                    if label_state is True:
                        return True
                    if config.managed_ci:
                        raise AgentLoopError(
                            f"PR #{pr_number} lost its authenticated `{MANAGED_LABEL}` suppression label; "
                            "managed qualification is cancelled and the head is not qualified."
                        )
                    managed_ci = None
                    return False
                return True
            if managed_ci.adopted_existing_pr:
                if config.managed_ci:
                    raise AgentLoopError(
                        f"PR #{pr_number} managed-CI adoption provenance changed; "
                        "managed qualification is cancelled and the head is not qualified."
                    )
                if not release_adopted_managed_ci(
                    runner, config=config, pr_number=pr_number, contract=managed_ci
                ):
                    log(
                        config,
                        f"PR #{pr_number}: unable to release invocation-owned managed-CI label"
                        + managed_ci.release_diagnostic,
                    )
            log(config, f"PR #{pr_number}: managed-CI adoption provenance changed; using ordinary CI")
            managed_ci = None
            return False
        memory = prepare_agent_memory(runner, config)
        from .reviewer_seats import reviewer_seat_binding, validate_pr_seat_bindings
        seat_binding = reviewer_seat_binding(config)

        def bound_pr_metadata(**fields: object) -> PostedRoundMetadata:
            metadata = PostedRoundMetadata(**fields)
            return (
                dataclasses_replace(metadata, seat_binding=seat_binding)
                if seat_binding is not None and metadata.flow == "pr"
                else metadata
            )
        reviewer_session_ids: dict[AgentName, str | None] = {}
        unavailable_reviewer_failures: dict[AgentName, AgentInvocationError] = {}
        configured_reviewers = reviewers(config)
        scheduler_capabilities = policy_capabilities(config.pr_review_policy)
        # Historical variable name retained for compatibility with helpers and
        # tests; it now means any scheduler-enabled PR policy.
        selective_policy = scheduler_capabilities.scheduler_enabled
        scheduler_contract = make_contract(
            tuple(agent_display_name(reviewer) for reviewer in configured_reviewers),
            config.pr_review_policy,
            config.pr_review_broad_rules,
            (
                agent_display_name(config.primary_reviewer)
                if config.primary_reviewer is not None
                else None
            ),
        )
        reviewer_acquisition_contract: dict[str, tuple[object, ...]] = {}
        from .reviewer_seats import SeatAgent, seat_backend
        for reviewer in configured_reviewers:
            invocation = resolve_invocation(config, provider=reviewer, role="reviewer")
            reviewer_acquisition_contract[agent_display_name(reviewer)] = (
                reviewer.model_chain if isinstance(reviewer, SeatAgent) else invocation.configured_model,
                invocation.resolved_effort,
                seat_backend(reviewer),
            )
        # Two separate full-board latches (#840).  The operator latch is the
        # explicit ``--pr-review-force-full`` authorization (or a persisted
        # operator-sourced record); it is durable for the run and every resume.
        # ``scheduler_force_full`` is the automatic recovery latch.  Under the
        # staged policy it is restored and raised only after a qualified panel
        # opening; before one, automatic fallbacks re-invoke only the primary.
        scheduler_operator_force_full = bool(config.pr_review_force_full)
        scheduler_force_full = False
        # Automatic fallback reasons raised outside the scheduler block.  The
        # staged policy consumes them per decision; other policies latch.
        pending_automatic_fallback_reasons: list[str] = []
        scheduler_calls_avoided = 0
        final_sweep_pending = False
        def stop_pre_panel(message: str, *, round_number: int | None) -> None:
            """Log and post the pre-panel diagnostic, then stop before any reviewer."""
            label = f"round {round_number}" if round_number is not None else "startup"
            log(config, f"{label.capitalize()}: {message}")
            # Plain audit text only: no round metadata that could later be
            # mistaken for a scheduler checkpoint or a panel opening.
            post_pr_comment(
                runner,
                config=config,
                pr_number=pr_number,
                body=f"PR review scheduling diagnostic ({label}): {message}",
            )
            raise PrePanelSafetyError(message)

        # A scheduler contract written by an earlier run is immutable. Missing
        # legacy scheduler fields intentionally mean "use the full board".
        try:
            startup_records = _extract_round_metadata_records(initial_pr_context.comments, flow="pr")
        except AgentLoopError as exc:
            if not scheduler_capabilities.requires_primary:
                raise
            # Staged policy: panel state is unknowable.  Stop through the same
            # no-reviewer diagnostic path as an in-round decode failure.
            stop_pre_panel(undecodable_history_message(exc), round_number=None)
            raise
        changed_seat_models = validate_pr_seat_bindings(startup_records, config)
        if changed_seat_models:
            log(config, "PR reviewer model-chain reconfiguration requires fresh review: " + ", ".join(sorted(changed_seat_models)))
        # Advisory per-run finding history (#1273), seeded once from the records
        # just extracted through the authoritative path above.
        pr_finding_history = FindingHistoryLedger(
            _HISTORY_PHASE_PR, log=lambda message: log(config, message)
        )
        pr_finding_history.seed_from_records(
            startup_records,
            reconciliation_mode=(
                "owner-scoped"
                if scheduler_capabilities.owner_scoped_reconciliation
                else "aggregate"
            ),
            same_status="same-pr",
        )
        # Signed reviewer-board amendments (#943) live on the PR itself, in
        # the same comment list as the PR scheduler records, for issue-mode
        # and standalone runs alike.  A PR amendment on the owning issue is
        # never ordered against PR comments; it fails closed.
        if issue_context is not None:
            reject_misplaced_pr_amendments(
                issue_context.comments, issue_number=issue_context.number
            )
        pr_amendment_diagnostics: list[str] = []
        pr_board_amendments = collect_reviewer_board_amendments(
            initial_pr_context.comments,
            flow="pr",
            pr_number=pr_number,
            ignored_sink=pr_amendment_diagnostics,
        )
        for diagnostic in pr_amendment_diagnostics:
            log(config, f"PR #{pr_number}: {diagnostic}")
        from .reviewer_seats import validate_pr_backend_outage_amendments
        validate_pr_backend_outage_amendments(pr_board_amendments, config)
        pr_contract_lineage: ContractLineage = resolve_contract_lineage(
            startup_records,
            pr_board_amendments,
            scheduler_contract,
            accept_base_configured=True,
            contract_from_metadata=_scheduler_contract_from_metadata,
            drift_error=lambda persisted, detail: _pr_contract_drift_error(
                persisted,
                detail,
                pr_number=pr_number,
                configured=scheduler_contract,
                start_round_number=lambda: _pr_amendment_start_round(
                    initial_pr_context, configured_reviewers, scheduler_capabilities,
                    runner=runner, config=config,
                ),
                amendments_recognized=bool(pr_board_amendments),
            ),
        )
        # The operator may keep the original board configured and let the signed
        # amendment do the removing (#1133).  Rebind every later board read to
        # the amended board so scheduling, prompts, and the gate agree.
        operator_reviewers = reviewers(config)
        pr_effective_contract = pr_contract_lineage.contracts[-1]
        if pr_effective_contract != scheduler_contract:
            effective_names = set(pr_effective_contract.required_reviewers)
            effective_reviewers = tuple(
                reviewer
                for reviewer in operator_reviewers
                if agent_display_name(reviewer) in effective_names
            )
            log(
                config,
                f"PR #{pr_number}: configured reviewer board "
                f"{', '.join(agent_display_name(r) for r in operator_reviewers)} differs from "
                "the signed amended board "
                f"{', '.join(pr_effective_contract.required_reviewers)}; using the amended board.",
            )
            config = dataclasses_replace(
                config, reviewer=effective_reviewers,
                pr_seat_binding_override=seat_binding,
            )
            configured_reviewers = reviewers(config)
            scheduler_contract = pr_effective_contract
            reviewer_acquisition_contract = {
                name: value
                for name, value in reviewer_acquisition_contract.items()
                if name in effective_names
            }
        pr_amendment_digest = pr_contract_lineage.active_digest
        pr_amendment_checkpoint_pending = bool(pr_contract_lineage.pending_amendments)
        for record in startup_records:
            if record.metadata.scheduler_force_full:
                if record.metadata.scheduler_force_full_source == "operator":
                    scheduler_operator_force_full = True
                elif not scheduler_capabilities.requires_primary:
                    scheduler_force_full = True
            if record.metadata.scheduler_calls_avoided is not None:
                scheduler_calls_avoided = max(
                    scheduler_calls_avoided, record.metadata.scheduler_calls_avoided
                )
        def request_automatic_scheduler_fallback(reason: str) -> None:
            """Record an automatic full-board request (#840).

            Non-staged policies keep the historical durable latch.  The staged
            policy defers the decision to the scheduler block, which latches
            only after a qualified panel opening and otherwise re-invokes just
            the primary with full context.
            """
            nonlocal scheduler_force_full
            if scheduler_capabilities.requires_primary:
                if reason not in pending_automatic_fallback_reasons:
                    pending_automatic_fallback_reasons.append(reason)
            else:
                scheduler_force_full = True

        unresolved_items: list[UnresolvedReviewItem] = []
        pr_compact_prior_summaries: list[str] = []
        latest_coder_output: str | None = None
        latest_coder_coverage_map: str | None = None
        latest_coder_metadata: PostedRoundMetadata | None = None
        next_unresolved_item_number = 1
        start_round_number = 1
        resumed_round: ResumedReviewRound | None = None
        qualification_checkpoint: QualificationCheckpoint | None = None
        if reviewer_session_id is not None and configured_reviewers:
            # Backward-compatible single-reviewer resume support: older callers
            # pass one reviewer session, so attach it to the first configured reviewer.
            reviewer_session_ids[configured_reviewers[0]] = reviewer_session_id
        prefetched_pr_context: PullRequestReviewContext | None = None
        # Bounded-progress guard for merge-conflict rounds (#606): if the coder
        # is dispatched to resolve a conflict and the PR head is still exactly
        # the same head the next time a conflict dispatch is about to happen,
        # the coder round made no progress -- stop cleanly instead of looping.
        conflict_dispatch_head_sha: str | None = None
        latest_mergeability: PullRequestMergeability | None = None
        resumed_round = _resume_pr_round_admitted(
            runner,
            config=config,
            pr_number=pr_number,
            comments=initial_pr_context.comments,
            head_sha=initial_pr_context.metadata.head_sha,
            configured_reviewers=configured_reviewers,
            reconciliation_mode=(
                "owner-scoped"
                if scheduler_capabilities.owner_scoped_reconciliation
                else "aggregate"
            ),
        )
        pr_amendment_start_round = (
            resumed_round.round_number if resumed_round is not None else 1
        )
        require_amendment_activation(
            pr_contract_lineage,
            start_round_number=pr_amendment_start_round,
            template=lambda amendment, round_number: _board_amendment_template(
                flow="pr",
                issue_number=None,
                pr_number=pr_number,
                persisted=pr_contract_lineage.contracts[
                    pr_contract_lineage.amendments.index(amendment)
                ],
                removed=amendment.removed_reviewers,
                restored=amendment.restored_reviewers,
                start_round_number=round_number,
            ),
        )
        pr_amendment_note: str | None = None
        if pr_contract_lineage.active_amendment is not None:
            pr_amended_contract = pr_contract_lineage.contracts[-1]
            _pr_view, pr_amendment_reassignments = apply_board_amendment_to_ledger(
                (
                    (*resumed_round.prior_items, *resumed_round.current_round_new_items)
                    if resumed_round is not None
                    else ()
                ),
                removed_reviewers=pr_contract_lineage.removed_reviewers,
                remaining_reviewers=pr_amended_contract.required_reviewers,
                primary_reviewer=pr_amended_contract.primary_reviewer,
            )
            pr_amendment_note = amendment_summary_line(
                pr_contract_lineage, pr_amendment_reassignments
            )
            log(config, f"PR #{pr_number}: {pr_amendment_note}")
            if not amendment_audit_already_posted(
                initial_pr_context.comments, pr_contract_lineage.active_amendment.digest
            ):
                post_pr_comment(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    body=render_amendment_audit_comment(
                        pr_contract_lineage,
                        start_round_number=pr_amendment_start_round,
                        reassignments=pr_amendment_reassignments,
                    ),
                )

        def announce_reduced_board_completion() -> str:
            """Durable reduced-board completion note (#943); returns stdout suffix."""
            if (
                pr_amendment_note is None
                or pr_contract_lineage.active_amendment is None
                # A board restored back to C0 is not reduced (#984).
                or not pr_contract_lineage.removed_reviewers
            ):
                return ""
            _post_reduced_board_completion_note(
                runner,
                config=config,
                pr_number=pr_number,
                digest=pr_contract_lineage.active_amendment.digest,
                note=pr_amendment_note,
            )
            return f" {pr_amendment_note}"

        def pr_ledger_view(
            items: Sequence[UnresolvedReviewItem],
        ) -> tuple[UnresolvedReviewItem, ...]:
            """Derived ledger with removed reviewers' ownership reassigned (#943)."""
            if pr_contract_lineage.active_amendment is None:
                return tuple(items)
            amended = pr_contract_lineage.contracts[-1]
            view, _reassignments = apply_board_amendment_to_ledger(
                items,
                removed_reviewers=pr_contract_lineage.removed_reviewers,
                remaining_reviewers=amended.required_reviewers,
                primary_reviewer=amended.primary_reviewer,
            )
            return view

        if resumed_round is not None:
            unresolved_items = list(resumed_round.prior_items)
            pr_compact_prior_summaries = list(
                bound_compact_prior_summaries(resumed_round.compact_prior_summaries)
            )
            latest_coder_output = resumed_round.coder_output
            # A stored comment may predate the rendering boundary, so the map is
            # restored only for an applicable matrix and only when unambiguous.
            latest_coder_coverage_map = (
                initial_coverage_map
                if initial_coverage_map
                and resumed_round.coder_metadata is not None
                and resumed_round.coder_metadata.round_number == 1
                else None
            ) or (
                extract_coverage_map_section(
                    resumed_round.coder_comment_body,
                    expected_revision=(
                        resumed_round.coder_metadata.subject
                        if resumed_round.coder_metadata is not None else None
                    ),
                    # The coder's persisted response must not name the heading:
                    # otherwise the stored section may be coder-authored prose.
                    coder_response=(
                        resumed_round.coder_metadata.raw_structured_coder_response
                        if resumed_round.coder_metadata is not None else None
                    ),
                    require_coder_clean=True,
                )
                if coverage_map_applies(approved_plan_context)
                else None
            )
            latest_coder_metadata = resumed_round.coder_metadata
            qualification_checkpoint = resumed_round.qualification_checkpoint
            next_unresolved_item_number = resumed_round.next_unresolved_item_number
            start_round_number = resumed_round.round_number
            if changed_seat_models:
                # A published peer cannot be re-invoked within its old
                # parallel round. Begin a new round at the same head while
                # carrying both the prior ledger and this round's verified
                # new findings. The latter have not yet been reconciled into
                # prior_items when a run stopped immediately after review.
                carried_by_id = {item.item_id: item for item in unresolved_items}
                for review_record in resumed_round.completed_reviews:
                    for item in review_record.metadata.new_items:
                        if item.reviewer != review_record.metadata.agent:
                            raise AgentLoopError(
                                "PR seat reconfiguration cannot verify the owner of a carried finding."
                            )
                        existing = carried_by_id.get(item.item_id)
                        if existing is not None and existing != item:
                            raise AgentLoopError(
                                "PR seat reconfiguration found conflicting provenance for a carried finding."
                            )
                        carried_by_id[item.item_id] = item
                unresolved_items = list(carried_by_id.values())
                start_round_number += 1
                qualification_checkpoint = None
                resumed_round = None
                final_sweep_pending = True
            log(config, f"PR #{pr_number}: resuming round {start_round_number}")
            if latest_coder_metadata is not None and any(
                "external/unrecorded head advance" in reason
                for reason in latest_coder_metadata.scheduler_reasons
            ):
                # The first post-recovery review must be full-board even when
                # the recovered coder checkpoint is the only current-head
                # record.  This is an audit decision, not the durable
                # operator force-full latch.
                final_sweep_pending = True
        watch_failure_extension_used = False
        watch_head_extension_used = False
        watch_deadline: float | None = None
        watch_attempts_remaining: int | None = None
        # One extra slot is only activated by a watcher-discovered failure on
        # the configured final round; ordinary review failures retain the cap.
        allowed_rounds = config.max_rounds
        if qualification_checkpoint is not None:
            if qualification_checkpoint.valid:
                if config.max_rounds <= qualification_checkpoint.allowed_rounds <= config.max_rounds + 2:
                    allowed_rounds = qualification_checkpoint.allowed_rounds
                    watch_failure_extension_used = qualification_checkpoint.watch_failure_extension_used
                    watch_head_extension_used = qualification_checkpoint.watch_head_extension_used
                else:
                    # A persisted ceiling outside this invocation's bounded
                    # repair budget is not an authorization to mint rounds.
                    qualification_checkpoint = QualificationCheckpoint.invalid(
                        "checkpoint budget is outside the configured bound"
                    )
                    request_automatic_scheduler_fallback(
                        "qualification checkpoint budget is outside the configured bound"
                    )
                    final_sweep_pending = True
            else:
                request_automatic_scheduler_fallback("qualification checkpoint is invalid")
                final_sweep_pending = True
        # Bounded budget carried by a recovery record (#1292), restored under
        # the same bound.  It never carries lifecycle, approvals or identities,
        # and no recovery path grants an extension of its own.
        recovery_budget = (
            resumed_round.recovery_round_budget if resumed_round is not None else None
        )
        if recovery_budget is not None and qualification_checkpoint is None:
            if (
                recovery_budget.valid
                and config.max_rounds <= recovery_budget.allowed_rounds <= config.max_rounds + 2
            ):
                allowed_rounds = recovery_budget.allowed_rounds
                watch_failure_extension_used = recovery_budget.watch_failure_extension_used
                watch_head_extension_used = recovery_budget.watch_head_extension_used
            else:
                request_automatic_scheduler_fallback(
                    "recovery record budget is invalid or out of bound"
                )
                final_sweep_pending = True
        if resumed_round is not None and resumed_round.head_review_recovery is not None:
            log(
                config,
                f"PR #{pr_number}: head-review recovery ({resumed_round.head_review_recovery}) "
                f"runs an ordinary review round {resumed_round.round_number} of the current head",
            )
        # Exact-head evidence state (#1068).  A freeze or release record is a
        # handoff boundary that carries the round budget; restore it under the
        # same bound as a qualification checkpoint before the pre-round guard.
        evidence_boundary = (
            resumed_round.evidence_boundary if resumed_round is not None else None
        )
        evidence_clearances: list[tuple[str, str]] = list(
            resumed_round.evidence_clearances if resumed_round is not None else ()
        )
        evidence_budget_correction_pending = False
        if evidence_boundary is not None:
            boundary_payload = (
                evidence_boundary.evidence_freeze
                if evidence_boundary.phase == EVIDENCE_FREEZE_PHASE
                else evidence_boundary.evidence_release
            )
            if (
                boundary_payload is not None
                and boundary_payload.valid
                and boundary_payload.budget_valid
                and boundary_payload.allowed_rounds is not None
                and config.max_rounds <= boundary_payload.allowed_rounds <= config.max_rounds + 2
            ):
                allowed_rounds = boundary_payload.allowed_rounds
                watch_failure_extension_used = bool(boundary_payload.watch_failure_extension_used)
                watch_head_extension_used = bool(boundary_payload.watch_head_extension_used)
            else:
                # Invalid-checkpoint defaults: the persisted ceiling is never an
                # authorization to mint rounds.
                allowed_rounds = config.max_rounds
                watch_failure_extension_used = False
                watch_head_extension_used = False
                request_automatic_scheduler_fallback("evidence record budget is invalid or out of bound")
                final_sweep_pending = True
                evidence_budget_correction_pending = bool(
                    evidence_boundary.phase == EVIDENCE_FREEZE_PHASE
                    and boundary_payload is not None
                    and boundary_payload.valid
                    and resumed_round is not None
                    and resumed_round.broken_evidence_freeze_head is None
                )
        # Set by the finalization gate to re-run the board at the same head
        # inside the current round: ``response`` when frozen evidence can be
        # answered, ``refresh`` for signed input that arrived before a freeze.
        pending_evidence_pass: str | None = None
        # The signed-requirement IDs surfaced to the reviewers whose verdict the
        # next finalization relies on; ``None`` is an unverifiable baseline.
        evidence_surfaced_baseline: tuple[str, ...] | None = None
        evidence_revalidation_required = False
        if (
            evidence_boundary is not None
            and evidence_boundary.phase == EVIDENCE_RELEASE_PHASE
            and resumed_round is not None
            and resumed_round.broken_evidence_freeze_head is None
            and not resumed_round.unrecorded_head_advance
        ):
            release_payload = evidence_boundary.evidence_release
            evidence_revalidation_required = True
            evidence_surfaced_baseline = (
                release_payload.signed_requirement_ids_surfaced
                if release_payload is not None and release_payload.valid
                else None
            )

        def collect_live_requirement_ids(context: PullRequestReviewContext) -> tuple[str, ...]:
            """Re-collect signed requirement identities for the final revalidation."""
            live_issue = (
                get_issue_context(runner, config=config, issue_number=issue_context.number)
                if issue_context is not None
                else None
            )
            live_parent = (
                get_issue_context(runner, config=config, issue_number=parent_issue_context.number)
                if parent_issue_context is not None
                else None
            )
            return _reviewer_requirement_identity_ids(
                _build_requirements_context(
                    target_issue_context=live_issue,
                    pr_context=context,
                    parent_issue_context=live_parent,
                ).effective_requirements
            )

        # The two independent one-shot watcher allowances below can extend the
        # effective ceiling by two rounds in one invocation. Keep the static
        # range large enough to reach both; ``allowed_rounds`` remains the
        # authoritative guard and prevents either slot from being used unless
        # its corresponding watcher transition grants it.  An evidence pass
        # re-runs the current round number without advancing it.
        round_sequence = _RepeatableRoundSequence(start_round_number, config.max_rounds + 3)

        def stage_same_head_evidence_pass(
            context: PullRequestReviewContext, *, current_round: int
        ) -> None:
            """Re-run the board at the same head inside ``current_round`` (#1068)."""
            nonlocal initial_pr_context, prefetched_pr_context, resumed_round
            nonlocal pending_evidence_pass, issue_context_refreshed
            nonlocal parent_issue_context_refreshed
            pending_evidence_pass = (
                "response"
                if any(
                    item.candidate_head_sha == context.metadata.head_sha
                    for item in _frozen_evidence_obligations(unresolved_items)
                )
                else "refresh"
            )
            resumed_round = None
            issue_context_refreshed = False
            parent_issue_context_refreshed = False
            if current_round == start_round_number:
                initial_pr_context = context
            else:
                prefetched_pr_context = context
            log(
                config,
                f"Round {current_round}: new signed human input reached PR #{pr_number} at "
                f"head {context.metadata.head_sha}; re-running the full review board at the "
                f"same head as an evidence {pending_evidence_pass} pass without advancing the round",
            )
            round_sequence.repeat_current()

        def stage_evidence_head_change(context: PullRequestReviewContext) -> None:
            """Hand a head that moved before the freeze to the new-head path."""
            nonlocal unresolved_items, prefetched_pr_context, resumed_round
            nonlocal final_sweep_pending
            unresolved_items = _advance_machine_obligations_for_head(
                release_evidence_freeze(unresolved_items),
                current_head_sha=context.metadata.head_sha,
            )
            prefetched_pr_context = context
            resumed_round = None
            final_sweep_pending = True
            log(
                config,
                f"PR #{pr_number} head changed to {context.metadata.head_sha} before the evidence "
                "gate; no freeze was published and the new head needs review",
            )

        # Step-back entries recorded by the previous iteration's coder turn (#1251).
        pending_step_back_sweeps: dict = {}
        for round_number in round_sequence:
            evidence_pass = pending_evidence_pass
            pending_evidence_pass = None
            if round_number > allowed_rounds:
                raise AgentLoopError(
                    _round_limit_diagnostic(
                        pr_number=pr_number,
                        round_number=allowed_rounds,
                        items=unresolved_items,
                        current_head_sha=(
                            prefetched_pr_context.metadata.head_sha
                            if prefetched_pr_context is not None
                            else initial_pr_context.metadata.head_sha
                        ),
                        sub_item_stall_rounds=config.sub_item_stall_rounds,
                    )
                )
            coder_name = agent_display_name(config.coder)
            pre_review_tests_passed = False
            if pre_review_test_pending:
                run_pre_review_tests(runner, config)
                pre_review_tests_passed = bool(config.pre_review_tests and config.test_command)
                pre_review_test_pending = False
            if round_number == start_round_number:
                pr_context = initial_pr_context
            elif prefetched_pr_context is not None:
                pr_context = prefetched_pr_context
                prefetched_pr_context = None
            else:
                pr_context = get_pr_review_context(runner, config=config, pr_number=pr_number)
            initial_pr_context = pr_context
            pr_metadata = pr_context.metadata
            if qualification_checkpoint is not None and qualification_checkpoint.valid:
                checkpoint_base = pr_metadata.base_branch or config.base
                checkpoint_item = next(
                    (
                        item for item in unresolved_items
                        if _is_machine_obligation(item)
                        and item.obligation_kind == qualification_checkpoint.obligation_kind
                        and (item.obligation_identity or item.item_id)
                        == qualification_checkpoint.obligation_identity
                    ),
                    None,
                )
                checkpoint_head_matches = (
                    (
                        qualification_checkpoint.lifecycle == "repair_required"
                        and qualification_checkpoint.failed_head_sha == pr_metadata.head_sha
                    )
                    or (
                        qualification_checkpoint.lifecycle != "repair_required"
                        and qualification_checkpoint.candidate_head_sha == pr_metadata.head_sha
                        and qualification_checkpoint.failed_head_sha != pr_metadata.head_sha
                    )
                )
                checkpoint_stale = (
                    checkpoint_item is None
                    or not checkpoint_head_matches
                    or qualification_checkpoint.base_branch != checkpoint_base
                )
                if checkpoint_stale:
                    # Readiness is a cache, never authority.  A changed head,
                    # base, or obligation identity forces safe reconstruction
                    # and a full reviewer board before another qualification.
                    qualification_checkpoint = QualificationCheckpoint.invalid(
                        "checkpoint no longer matches the live qualification inputs"
                    )
                    request_automatic_scheduler_fallback(
                        "qualification checkpoint no longer matches the live inputs"
                    )
                    final_sweep_pending = True
            if issue_context is not None and not (
                round_number == start_round_number and issue_context_refreshed
            ):
                issue_context = get_issue_context(
                    runner, config=config, issue_number=issue_context.number
                )
            if issue_context is not None:
                issue_context_refreshed = True
            if parent_issue_context is not None and not (
                round_number == start_round_number and parent_issue_context_refreshed
            ):
                parent_issue_context = get_issue_context(
                    runner, config=config, issue_number=parent_issue_context.number
                )
            if parent_issue_context is not None:
                parent_issue_context_refreshed = True
            pr_comments = pr_context.comments
            pr_posted_checkpoint: PostedRoundMetadata | None = None
            # A review round is a fresh acquisition boundary. Rebind the
            # candidate/base pair before constructing any reviewer prompt so
            # a coder push or retarget cannot be reviewed with stale
            # architecture prose. The approved reviewer identity is the
            # invalidation anchor; target-tip movement alone is intentionally
            # ignored by ArchitecturePair.identity().
            if config.architecture_context_enabled:
                stored_architecture_identity = _latest_pr_architecture_observation(
                    pr_comments, head_sha=pr_metadata.head_sha
                )
                round_architecture_config = _freeze_prompt_architecture(
                    runner,
                    config,
                    target_revision=pr_metadata.base_branch,
                    candidate_revision=pr_metadata.head_sha,
                    pr_pair=True,
                )
                round_architecture = round_architecture_config.architecture_context
                round_architecture_identity = (
                    round_architecture.identity()
                    if hasattr(round_architecture, "identity")
                    else None
                )
                if (
                    stored_architecture_identity is not None
                    and round_architecture_identity != stored_architecture_identity
                ):
                    # The scheduler will launch a full current-head review;
                    # persistence of that review makes this transition
                    # exact-once on resume.
                    final_sweep_pending = True
                    request_automatic_scheduler_fallback("architecture identity changed")
                config = round_architecture_config
            followup_source_context = _pr_followup_source_context(
                config=config,
                pr_number=pr_number,
                pr_metadata=pr_metadata,
                issue_context=issue_context,
            )
            if closing_contract is not None:
                validate_pr_expected_closing_issues(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    expected_issue_ids=closing_contract.expected_closing_issue_ids,
                    body=pr_metadata.body,
                    reject_unexpected=config.managed_ci and issue_context is not None,
                )
            if managed_ci_active(pr_metadata) and pre_review_tests_passed and pr_metadata.head_sha:
                publish_round_readiness(
                    runner,
                    config=config,
                    head_sha=pr_metadata.head_sha,
                )
            requirements_context = _build_requirements_context(
                target_issue_context=issue_context,
                pr_context=pr_context,
                parent_issue_context=parent_issue_context,
            )
            human_requirements = requirements_context.effective_requirements
            if qualification_checkpoint is not None and qualification_checkpoint.valid:
                checkpoint_plan = (
                    approved_plan_context.plan_hash
                    if approved_plan_context is not None else None
                )
                checkpoint_inputs_stale = (
                    qualification_checkpoint.plan_digest is not None
                    and qualification_checkpoint.plan_digest != checkpoint_plan
                ) or (
                    qualification_checkpoint.requirements_digest is not None
                    and qualification_checkpoint.requirements_digest != _qualification_digest(
                        tuple(requirement.requirement_id for requirement in human_requirements)
                    )
                ) or (
                    qualification_checkpoint.acquisition_digest is not None
                    and qualification_checkpoint.acquisition_digest != _qualification_digest(
                        reviewer_acquisition_contract
                    )
                )
                if checkpoint_inputs_stale:
                    qualification_checkpoint = QualificationCheckpoint.invalid(
                        "qualification inputs changed during resume"
                    )
                    request_automatic_scheduler_fallback("qualification inputs changed during resume")
                    final_sweep_pending = True
            current_resume = resumed_round if resumed_round is not None and round_number == resumed_round.round_number else None
            if (
                current_resume is not None
                and current_resume.head_review_recovery is not None
                and current_resume.head_review_recovery_post_required
            ):
                # The handoff is written before any reviewer is dispatched so an
                # interruption resumes with this ledger, round and budget.
                _persist_head_review_recovery(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    round_number=current_resume.round_number,
                    head_sha=pr_metadata.head_sha,
                    ledger=current_resume.prior_items,
                    budget=RecoveryRoundBudget(
                        allowed_rounds,
                        watch_failure_extension_used,
                        watch_head_extension_used,
                    ),
                    source=current_resume.head_review_recovery,
                )
                resumed_round = dataclasses_replace(
                    current_resume, head_review_recovery_post_required=False
                )
                current_resume = resumed_round
            unresolved_items = _reconcile_human_requirements_ack_item(
                current_resume.prior_items if current_resume is not None else unresolved_items,
                coder_output=latest_coder_output,
                human_requirements=human_requirements,
                source_round=round_number,
            )
            # A CI obligation is bound to the head that failed.  Only a
            # strictly different live head may become a qualification
            # candidate; reviewer dispositions never perform this transition.
            unresolved_items = _advance_machine_obligations_for_head(
                unresolved_items,
                current_head_sha=pr_metadata.head_sha,
            )
            # GitHub mergeability gate (#606): fetch and evaluate before starting
            # a review round so a confirmed conflict routes straight to the coder
            # instead of spending reviewer time (or later a full CI wait) on a
            # branch that cannot merge. `unknown` is left alone here so a
            # transient GitHub mergeability computation window never triggers an
            # unnecessary coder round.
            round_start_mergeability = get_pr_mergeability(runner, config=config, pr_number=pr_number)
            latest_mergeability = round_start_mergeability
            unresolved_items = _reconcile_merge_conflict_item(
                unresolved_items,
                mergeability=round_start_mergeability,
                source_round=round_number,
                current_head_sha=pr_metadata.head_sha,
            )
            # A confirmed conflict from this probe is pending; so is a
            # preserved blocker from an earlier confirmed conflict that this
            # probe merely returned `unknown` for on the same head (a probe
            # hiccup, not evidence the conflict resolved) -- checking the
            # ledger, not the raw probe state, is what keeps the coder from
            # being bypassed by a transient GitHub mergeability failure.
            conflict_pending = any(
                item.item_id == MERGE_CONFLICT_ITEM_ID for item in unresolved_items
            )
            if conflict_pending:
                log(
                    config,
                    f"Round {round_number}: PR #{pr_number} has a merge conflict with "
                    f"{round_start_mergeability.base_branch or config.base or 'the base branch'}; "
                    f"skipping reviewers and routing to {agent_display_name(config.coder)}",
                )
            # Exact-head evidence recovery (#1068).  A freeze or release record
            # is the handoff boundary; an external push breaks a freeze.
            evidence_skip_reviewers = False
            broken_freeze_head = (
                current_resume.broken_evidence_freeze_head
                if current_resume is not None
                else None
            )
            if broken_freeze_head is not None:
                post_pr_comment(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    body=(
                        "## Exact-head evidence freeze broken\n\n"
                        f"The evidence freeze at `{broken_freeze_head}` was broken by an external "
                        f"push: the PR head is now `{pr_metadata.head_sha}`. Evidence recorded for "
                        "the old head does not carry forward. The requested evidence returns to "
                        "deferred and the new head gets a full review."
                        + _evidence_barrier_note(unresolved_items)
                        + "\n\n-- coding-review-agent-loop"
                    ),
                )
                log(
                    config,
                    f"Round {round_number}: evidence freeze at {broken_freeze_head} was broken by "
                    f"an external push to {pr_metadata.head_sha}; reviewing the new head in full",
                )
                final_sweep_pending = True
                request_automatic_scheduler_fallback("an external push broke an evidence freeze")
            live_evidence_boundary = (
                current_resume.evidence_boundary
                if current_resume is not None
                and broken_freeze_head is None
                and not current_resume.unrecorded_head_advance
                and current_resume.evidence_boundary is not None
                and current_resume.evidence_boundary.subject == pr_metadata.head_sha
                else None
            )
            if live_evidence_boundary is not None:
                if any(
                    item.obligation_identity == "invalid-evidence-record"
                    for item in unresolved_items
                ):
                    raise AgentLoopError(
                        f"PR #{pr_number} has a malformed or contradictory persisted exact-head "
                        "evidence record at head "
                        f"{pr_metadata.head_sha}; it is kept as a non-bypassable blocker. No "
                        "coder, qualification, or merge was attempted. "
                        + _round_limit_diagnostic(
                            pr_number=pr_number,
                            round_number=round_number,
                            items=unresolved_items,
                            current_head_sha=pr_metadata.head_sha,
                            sub_item_stall_rounds=config.sub_item_stall_rounds,
                        )
                    )
                freeze_payload = live_evidence_boundary.evidence_freeze
                if (
                    live_evidence_boundary.phase == EVIDENCE_FREEZE_PHASE
                    and freeze_payload is not None
                    and _frozen_evidence_obligations(unresolved_items)
                ):
                    # Frozen rerun: one check snapshot and one mergeability
                    # observation (the round-start probe above) at H, with no
                    # CI wait and no qualification dispatch.
                    rerun_checks = get_pr_checks(runner, config=config, metadata=pr_metadata)
                    if managed_ci_active(pr_metadata):
                        rerun_checks = intermediate_managed_checks(rerun_checks)
                    rerun_stalled = {
                        (check.kind, check.name) for check in rerun_checks.infrastructure_stalls
                    }
                    rerun_failures = tuple(
                        check for check in rerun_checks.failing
                        if (check.kind, check.name) not in rerun_stalled
                    )
                    gate_reasons: list[str] = []
                    if rerun_failures:
                        failure_snapshot = dataclasses_replace(
                            rerun_checks, state="failing", failing=rerun_failures,
                            pending=(), missing_required=(), infrastructure_stalls=(),
                            required_checks=(), branch_protection_note=None,
                        )
                        details = _pr_check_details(failure_snapshot)
                        details.append(f"Frozen head: {pr_metadata.head_sha}")
                        had_ordinary_obligation = any(
                            _is_machine_obligation(item)
                            and item.obligation_kind == "github-pr-checks"
                            for item in unresolved_items
                        )
                        unresolved_items = _upsert_machine_obligation(
                            unresolved_items,
                            item_number=next_unresolved_item_number,
                            kind="github-pr-checks",
                            source_round=round_number,
                            text=_pr_check_blocking_review(pr_number, "failing", details),
                            failed_head_sha=pr_metadata.head_sha,
                        )
                        if not had_ordinary_obligation:
                            next_unresolved_item_number += 1
                        post_pr_comment(
                            runner, config=config, pr_number=pr_number,
                            body=_format_pr_checks_comment(pr_number, "failing", details),
                        )
                        gate_reasons.append("failing GitHub checks")
                    if conflict_pending:
                        gate_reasons.append("a merge conflict with the base branch")
                    if gate_reasons:
                        # Release before any budget check or dispatch; the
                        # repair then runs as round R+1 under the ordinary guard.
                        unresolved_items = _publish_evidence_release(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            round_number=round_number,
                            head_sha=str(pr_metadata.head_sha),
                            items=unresolved_items,
                            reason="machine-gate",
                            surfaced_requirement_ids=freeze_payload.signed_requirement_ids_at_freeze,
                            allowed_rounds=allowed_rounds,
                            watch_failure_extension_used=watch_failure_extension_used,
                            watch_head_extension_used=watch_head_extension_used,
                            clearances=evidence_clearances,
                            detail="; ".join(gate_reasons),
                        )
                        log(
                            config,
                            f"Round {round_number}: evidence freeze released at "
                            f"{pr_metadata.head_sha}: {'; '.join(gate_reasons)}",
                        )
                        evidence_skip_reviewers = True
                    else:
                        if evidence_budget_correction_pending:
                            # Only the budget was invalid: keep the frozen
                            # ledger and re-persist one corrected record.
                            _publish_evidence_freeze(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                round_number=round_number,
                                head_sha=str(pr_metadata.head_sha),
                                items=unresolved_items,
                                signed_requirement_ids=freeze_payload.signed_requirement_ids_at_freeze,
                                allowed_rounds=allowed_rounds,
                                watch_failure_extension_used=watch_failure_extension_used,
                                watch_head_extension_used=watch_head_extension_used,
                                clearances=evidence_clearances,
                                still_frozen=True,
                            )
                            evidence_budget_correction_pending = False
                        # New human input is detected by signed-requirement
                        # identity: an edit yields a new ID, a deletion none.
                        new_signed_ids = set(
                            _reviewer_requirement_identity_ids(human_requirements)
                        ) - set(freeze_payload.signed_requirement_ids_at_freeze)
                        if not new_signed_ids:
                            log(
                                config,
                                f"Round {round_number}: PR #{pr_number} remains frozen at "
                                f"{pr_metadata.head_sha}; no new signed human input",
                            )
                            raise HumanDecisionRequiredError(
                                _evidence_freeze_diagnostic(
                                    pr_number=pr_number,
                                    head_sha=pr_metadata.head_sha,
                                    items=unresolved_items,
                                )
                            )
                        evidence_pass = "response"
                        log(
                            config,
                            f"Round {round_number}: new signed human input at frozen head "
                            f"{pr_metadata.head_sha}; re-invoking the full review board",
                        )
                elif live_evidence_boundary.phase == EVIDENCE_RELEASE_PHASE:
                    # The releasing pass already reviewed this head; resume at
                    # the boundary without re-invoking reviewers.
                    evidence_skip_reviewers = True
            evidence_response_head = (
                pr_metadata.head_sha
                if evidence_pass is not None
                and any(
                    item.candidate_head_sha == pr_metadata.head_sha
                    for item in _frozen_evidence_obligations(unresolved_items)
                )
                else None
            )
            if evidence_pass is not None:
                evidence_skip_reviewers = False
            prior_unresolved_items = tuple(unresolved_items)
            pr_finding_history.note_carried_ledger(prior_unresolved_items)
            prior_dispositions: dict[str, list[ReviewItemDisposition]] = {
                item.item_id: [] for item in prior_unresolved_items
            }
            round_new_unresolved_items: list[UnresolvedReviewItem] = []
            # (reviewer, requests) pairs from this round's reviews (#1068).
            round_evidence_requests: list[tuple[str, tuple[str, ...]]] = []
            current_pr_subject = str(pr_metadata.head_sha or "unknown")
            round_ledger_incomplete = _round_ledger_may_be_incomplete(
                current_resume=current_resume,
                prior_unresolved_items=prior_unresolved_items,
                comments=pr_comments,
                flow="pr",
                current_subject=current_pr_subject,
            )
            round_resolved_history_item_ids = _round_resolved_history_item_ids(
                prior_unresolved_items=prior_unresolved_items,
                comments=pr_comments,
                flow="pr",
                reconciliation_mode=(
                    "owner-scoped"
                    if scheduler_capabilities.owner_scoped_reconciliation
                    else "aggregate"
                ),
                same_status="same-pr",
            )
            use_compact_pr_context = (
                config.pr_review_context_mode == "compact"
                and round_number >= 2
                and not round_ledger_incomplete
            )
            coder_followup_context = _coder_followup_review_context(
                latest_coder_output,
                latest_coder_metadata,
                head_sha=pr_metadata.head_sha,
                assigned_workdir=active_workdir(config),
                coverage_map=latest_coder_coverage_map,
            ) + _evidence_review_context(
                prior_unresolved_items, response_head=evidence_response_head
            )
            # Step-back sweeps (#1251) bind to the step-back coder record's
            # resulting head and this round only; rebuilt from durable records.
            pr_step_back_sweeps = _pr_step_back_sweep_contexts(
                config,
                pr_comments,
                round_number=round_number,
                head_sha=pr_metadata.head_sha,
                carried=pending_step_back_sweeps,
            )
            pending_step_back_sweeps = {}
            # Persist the same digest identities surfaced in reviewer prompts.
            # An edited signed comment must not inherit the old approval.
            surfaced_reviewer_requirement_ids = _reviewer_requirement_identity_ids(
                human_requirements
            )
            if not evidence_skip_reviewers:
                evidence_surfaced_baseline = surfaced_reviewer_requirement_ids
            approved_review_outputs: list[tuple[str, str]] = []
            round_blocking_reviewer_names: list[str] = []
            # The accepted carrier beside each approved text, keyed by reviewer, so
            # an acknowledgement repair can pin its assessment and records (#925).
            accepted_review_carriers: dict[str, ParsedPlanReview | ParsedReview] = {}
            completed_by_name = {
                record.metadata.agent: record
                for record in (current_resume.completed_reviews if current_resume is not None else ())
            }
            resumed_by_name = {
                name: record
                for name, record in completed_by_name.items()
                if _resumed_pr_reviewer_matches_requirements(
                    record,
                    human_requirements,
                    approved_plan_context,
                    reviewer_acquisition_contract=(
                        reviewer_acquisition_contract if selective_policy or config.reviewer_seats else None
                    ),
                )
                and (seat_binding is None or record.metadata.seat_binding == seat_binding)
            }
            # Staged-policy panel evidence (#840).  Derived from comment-ordered
            # history before any resume, approval, or scheduling decision uses
            # reviewer records, so premature secondary work cannot count.
            panel_evidence: PrPanelEvidence | None = None
            superseded_prepanel_reviews: dict[str, PostedRoundRecord] = {}
            if scheduler_capabilities.requires_primary:
                panel_history_error: AgentLoopError | None = None
                try:
                    panel_history = _extract_round_metadata_records(pr_comments, flow="pr")
                except AgentLoopError as exc:
                    panel_history = None
                    panel_history_error = exc
                if panel_history is None:
                    # Resume, approval, ledger, and qualification accounting all
                    # need this history, so the operator override cannot make an
                    # undecodable ledger safe either; stop in both flag states.
                    stop_pre_panel(
                        undecodable_history_message(panel_history_error),
                        round_number=round_number,
                    )
                else:
                    panel_evidence = _derive_pr_panel_evidence(
                        panel_history,
                        primary_reviewer=scheduler_contract.primary_reviewer,
                        required_reviewers=scheduler_contract.required_reviewers,
                    )
                assert panel_evidence is not None
                unqualified_blocking: list[PostedRoundRecord] = []
                # Classify every completed review of the interrupted round, not
                # only the resume-eligible ones: a premature blocking review
                # whose requirement, plan, or acquisition identity is now stale
                # must still reach the diagnostic or the audited supersession
                # rather than being dropped silently.
                for resumed_name, resumed_candidate in completed_by_name.items():
                    if _pr_record_is_panel_qualified(
                        resumed_candidate,
                        panel_evidence,
                        primary_reviewer=scheduler_contract.primary_reviewer,
                        operator_force_full=scheduler_operator_force_full,
                    ):
                        continue
                    # Not resumed, not a summary, not an approval, never early
                    # published: the reviewer needs a fresh post-opening turn.
                    resumed_by_name.pop(resumed_name, None)
                    if (
                        resumed_candidate.metadata.state == "approved"
                        and not resumed_candidate.metadata.new_items
                    ):
                        log(
                            config,
                            f"Round {round_number}: ignoring {resumed_name}'s review recorded before "
                            "any qualified panel opening; it gets a fresh post-opening turn",
                        )
                    elif scheduler_operator_force_full:
                        # Superseded, not consumed or discarded: excluded from
                        # the ledger and ownership, listed in the operator
                        # opening record, and replayed only as non-authoritative
                        # context for the reviewer's fresh turn.
                        superseded_prepanel_reviews[resumed_name] = resumed_candidate
                    else:
                        unqualified_blocking.append(resumed_candidate)
                if unqualified_blocking:
                    stop_pre_panel(
                        pre_panel_safety_message(
                            "premature panel review(s) "
                            + "; ".join(
                                _describe_superseded_prepanel_review(record)
                                for record in unqualified_blocking
                            )
                            + " were recorded before any qualified panel opening and are "
                            "blocking or carry new items,"
                        ),
                        round_number=round_number,
                    )
            # Summaries are review-level context, not new findings or substitutes
            # for an item's immutable claim. Seed from saved reviews for recovery.
            reviewer_summaries = {
                name: context
                for name, record in resumed_by_name.items()
                if (
                    context := _reviewer_summary_context(
                        name,
                        review_freeform_summary_text(record.body),
                        round_number=record.metadata.round_number,
                        head_sha=record.metadata.subject,
                    )
                )
            }
            unchanged_head_approvals = _latest_pr_approved_reviews_for_head(
                pr_comments,
                head_sha=pr_metadata.head_sha,
                configured_reviewers=configured_reviewers,
                approved_plan_context=approved_plan_context,
                human_requirements=human_requirements,
                require_architecture_contract=config.architecture_context_enabled,
                reviewer_acquisition_contract=(
                    reviewer_acquisition_contract if selective_policy or config.reviewer_seats else None
                ),
            )
            if seat_binding is not None:
                unchanged_head_approvals = {
                    name: record for name, record in unchanged_head_approvals.items()
                    if record.metadata.seat_binding == seat_binding
                }
            if panel_evidence is not None:
                # Causal approval eligibility: a secondary approval counts only
                # when recorded after the first qualified panel opening, so a
                # premature approval never opens the panel, shrinks the first
                # audit, is carried, or satisfies the exact-head barrier.
                unchanged_head_approvals = {
                    name: record
                    for name, record in unchanged_head_approvals.items()
                    if _pr_record_is_panel_qualified(
                        record,
                        panel_evidence,
                        primary_reviewer=scheduler_contract.primary_reviewer,
                        operator_force_full=scheduler_operator_force_full,
                    )
                }
            # A restored reviewer (#984) must re-approve the current head: an
            # approval from before its restoration round is never carried.
            pr_restoration_rounds = pr_contract_lineage.restoration_rounds
            unchanged_head_approvals = {
                name: record
                for name, record in unchanged_head_approvals.items()
                if not predates_restoration(
                    pr_restoration_rounds, name, record.metadata.round_number
                )
            }
            checkpoint_expected_plan_digest = (
                approved_plan_context.plan_hash
                if approved_plan_context is not None
                else None
            )
            checkpoint_expected_requirements_digest = _qualification_digest(
                tuple(requirement.requirement_id for requirement in human_requirements)
            )
            checkpoint_expected_acquisition_digest = _qualification_digest(
                reviewer_acquisition_contract
            )
            checkpoint_expected_attempt_id: str | None = None
            if (
                qualification_checkpoint is not None
                and qualification_checkpoint.lifecycle == "qualifying"
            ):
                if qualification_checkpoint.obligation_kind == "github-pr-checks":
                    checkpoint_expected_attempt_id = (
                        f"github-checks:{pr_metadata.head_sha}"
                        if pr_metadata.head_sha
                        else None
                    )
                elif (
                    qualification_checkpoint.obligation_kind == "managed-exact-head-ci"
                    and managed_ci is not None
                    and managed_ci_attachment_matches_head(managed_ci, pr_metadata.head_sha)
                ):
                    checkpoint_expected_attempt_id = (
                        f"{managed_ci.attached_run_id}/{managed_ci.run_attempt}"
                    )
            if (
                qualification_checkpoint is not None
                and qualification_checkpoint.valid
                and qualification_checkpoint.lifecycle
                in {"qualification_ready", "qualifying"}
                and not _qualification_checkpoint_review_identity_matches(
                    qualification_checkpoint,
                    configured_reviewers=configured_reviewers,
                    current_approvals=unchanged_head_approvals,
                    unresolved_items=unresolved_items,
                    expected_plan_digest=checkpoint_expected_plan_digest,
                    expected_requirements_digest=checkpoint_expected_requirements_digest,
                    expected_acquisition_digest=checkpoint_expected_acquisition_digest,
                    expected_qualification_attempt_id=checkpoint_expected_attempt_id,
                )
            ):
                # A persisted readiness marker is only a cache.  Missing
                # reviewer records, a changed approval set, or a changed
                # authority/ledger signature must never suppress the board.
                qualification_checkpoint = QualificationCheckpoint.invalid(
                    "qualification checkpoint review identities are incomplete or stale"
                )
                request_automatic_scheduler_fallback(
                    "qualification checkpoint review identities are incomplete or stale"
                )
                final_sweep_pending = True
            skip_reviewers_for_recovery = bool(
                current_resume is not None
                and (
                    (
                        current_resume.unrecorded_head_advance
                        # A broken freeze, or a recovered ledger holding only
                        # evidence, has nothing for the coder: review in full.
                        and broken_freeze_head is None
                        and any(
                            not _is_evidence_obligation(item)
                            for item in current_resume.prior_items
                        )
                    )
                    or (
                        qualification_checkpoint is not None
                        and qualification_checkpoint.valid
                        and qualification_checkpoint.lifecycle
                        in {"repair_required", "qualification_ready", "qualifying"}
                    )
                )
            )
            if skip_reviewers_for_recovery:
                if current_resume is not None and current_resume.unrecorded_head_advance:
                    if current_resume.rejected_coder_followup_reason:
                        log(
                            config,
                            f"Round {round_number}: coder recovery (attempt "
                            f"{(current_resume.dispatch_attempt or 1) + 1}) after a rejected "
                            f"coder follow-up dispatched on "
                            f"{current_resume.rejected_coder_followup_from_head}; routing the "
                            f"carried items through {coder_name} before review",
                        )
                    else:
                        log(
                            config,
                            f"Round {round_number}: PR head advanced to {current_pr_subject} "
                            "without current-head coder metadata; routing recovered prior items "
                            f"through {coder_name} before review",
                        )
                else:
                    log(
                        config,
                        f"Round {round_number}: resuming the persisted machine-obligation "
                        "checkpoint without redispatching completed reviewer work",
                    )
            scheduler_previous_sha: str | None = None
            scheduler_diff_context = ""
            latest_reviewer_records: dict[str, PostedRoundRecord] = {}
            external_recovery_full_board = (
                skip_reviewers_for_recovery or broken_freeze_head is not None
            )
            if evidence_skip_reviewers and not skip_reviewers_for_recovery:
                log(
                    config,
                    f"Round {round_number}: resuming at the persisted exact-head evidence "
                    "record without redispatching completed reviewer work",
                )
            skip_reviewers_for_recovery = skip_reviewers_for_recovery or evidence_skip_reviewers
            # One private spool per round, shared by the parallel launcher and
            # the sequential resume seam (#1025).
            pr_round_spool = _review_round_spool(
                config, surface="pr", number=pr_number,
                round_number=round_number, subject=current_pr_subject,
            )
            if seat_binding is not None:
                for reviewer in configured_reviewers:
                    name = agent_display_name(reviewer)
                    saved = pr_round_spool.load(name)
                    if saved is None or saved.get("failure") is not None or saved.get("seat_binding") == seat_binding:
                        continue
                    state, _carrier = pr_round_spool.publication_state(name)
                    if state == "fresh":
                        pr_round_spool.remove(name)
                        log(config, f"Discarded unpublished {name} response checkpoint after seat reconfiguration")
                    else:
                        raise AgentLoopError(
                            f"{name} has a published or unprovable response checkpoint under another seat binding; refusing recovery."
                        )

            def _pr_review_validators(reviewer_name: str) -> dict[str, object]:
                return _architecture_mode_validators(lambda mode: lambda text, reviewer_name=reviewer_name: _validate_review_response(
                    text,
                    reviewer=reviewer_name,
                    unresolved_items=prior_unresolved_items,
                    # Never share the mutable round_new_unresolved_items
                    # list with concurrent workers (#594): it only
                    # enriches the UnknownPriorItemDispositionError
                    # message, so an empty tuple changes no outcome.
                    current_round_items=(), architecture_status_mode=mode,
                    approved_matrix_row_ids=approved_matrix_row_ids(approved_plan_context),
                ))

            def _pr_publication(reviewer_name: str):
                return round_publication(
                    runner, config=config, spool=pr_round_spool,
                    reviewer_name=reviewer_name, flow="pr", round_number=round_number,
                    subject=current_pr_subject, surface_kind="pr", number=pr_number,
                    validation_context=publication_context_digest(
                        head=current_pr_subject,
                        surfaced=sorted(str(item) for item in surfaced_reviewer_requirement_ids),
                        **({"seat_binding": seat_binding} if seat_binding is not None else {}),
                    ),
                )

            pr_spool_preflight_done = [False]

            def _pr_spool_preflight(selected_names: set[str]) -> None:
                """Check every spool record before any checkpoint or reviewer launch (#1258)."""
                if (
                    pr_spool_preflight_done[0]
                    or skip_reviewers_for_recovery
                    or conflict_pending
                    or evidence_pass is not None
                ):
                    return
                pr_spool_preflight_done[0] = True
                published = [
                    agent_display_name(reviewer) for reviewer in configured_reviewers
                    if resumed_by_name.get(agent_display_name(reviewer)) is not None
                ]
                _preflight_spooled_publications(
                    runner, config=config, spool=pr_round_spool, reviewers=configured_reviewers,
                    validators_for=_pr_review_validators, publication_for=_pr_publication,
                    published_names=published,
                    post_frozen=lambda plan: post_frozen_round_bodies(
                        runner, config=config, surface_kind="pr", number=pr_number, plan=plan
                    ),
                    selected_reviewers=[
                        reviewer for reviewer in configured_reviewers
                        if agent_display_name(reviewer) in selected_names
                    ],
                )

            scheduler_metadata_recovery_full_board = False
            metadata_recovery_reasons: list[str] = []
            if selective_policy:
                try:
                    historical_records = _extract_round_metadata_records(pr_comments, flow="pr")
                except AgentLoopError:
                    historical_records = ()
                    scheduler_metadata_recovery_full_board = True
                    metadata_recovery_reasons.append("scheduler history could not be decoded")
                latest_reviewer_records = _latest_pr_reviewer_records(
                    historical_records, configured_reviewers
                )
                current_head_records = [
                    record for record in historical_records
                    if record.metadata.subject == current_pr_subject
                ]
                scheduler_records = [
                    record
                    for record in historical_records
                    if record.metadata.scheduler_metadata_status != "absent"
                ]
                latest_scheduler_record = scheduler_records[-1] if scheduler_records else None
                if historical_records and (
                    not scheduler_records
                    or latest_scheduler_record.metadata.scheduler_metadata_status != "valid"
                ):
                    # Under selective-intermediate this recovery override
                    # applies to the current decision only; the staged policy
                    # latches it durably below only after a qualified panel
                    # opening (#840).
                    scheduler_metadata_recovery_full_board = True
                    metadata_recovery_reasons.append("latest scheduler metadata is missing or invalid")
                current_scheduler_records = [
                    record
                    for record in current_head_records
                    if record.metadata.scheduler_metadata_status != "absent"
                ]
                if current_head_records and not current_scheduler_records:
                    # Do not infer a same-head selective decision from legacy
                    # reviewer/coder records without a scheduler checkpoint.
                    scheduler_metadata_recovery_full_board = True
                    metadata_recovery_reasons.append("current-head records lack scheduler metadata")
                if any(
                    record.metadata.scheduler_metadata_status == "invalid"
                    for record in current_head_records
                ):
                    scheduler_metadata_recovery_full_board = True
                    metadata_recovery_reasons.append("current-head scheduler metadata is invalid")
                if scheduler_contract.primary_reviewer is not None and any(
                    record.metadata.scheduler_metadata_status == "valid"
                    and (
                        record.metadata.scheduler_phase is None
                        or record.metadata.scheduler_primary_reviewer is None
                    )
                    for record in historical_records
                ):
                    # A pre-staged scheduler payload may decode as valid while
                    # lacking the phase-aware authority introduced with the
                    # primary policy. It remains compatible data, but cannot
                    # authorize a reduced primary/panel selection.
                    scheduler_metadata_recovery_full_board = True
                    metadata_recovery_reasons.append("scheduler metadata lacks phase authority")
                if any(
                    record.metadata.scheduler_metadata_status == "valid"
                    and record.metadata.scheduler_current_sha != record.metadata.subject
                    for record in historical_records
                ):
                    # A decoded checkpoint can still be contradictory with its
                    # enclosing audit record's subject. Never derive a
                    # previous transition from that state.
                    scheduler_metadata_recovery_full_board = True
                    metadata_recovery_reasons.append("a scheduler checkpoint contradicts its record subject")
                automatic_fallback_reasons: list[str] = list(pending_automatic_fallback_reasons)
                if scheduler_metadata_recovery_full_board and scheduler_capabilities.recovery_latches_force_full:
                    automatic_fallback_reasons.extend(
                        f"scheduler metadata recovery: {reason}"
                        for reason in dict.fromkeys(metadata_recovery_reasons)
                    )
                # The durable phase checkpoint is the latest valid scheduler
                # record that carries phase authority.  It only reports whether
                # the secondary panel has been opened; approvals stay exact-head.
                scheduler_checkpoint_phase: str | None = None
                if scheduler_capabilities.phase_aware and not scheduler_metadata_recovery_full_board:
                    scheduler_checkpoint_phase = next(
                        (
                            record.metadata.scheduler_phase
                            for record in reversed(historical_records)
                            if record.metadata.scheduler_metadata_status == "valid"
                            and record.metadata.scheduler_phase is not None
                        ),
                        None,
                    )
                current_coder_record = next(
                    (
                        record for record in reversed(current_head_records)
                        if record.metadata.role == "coder"
                        and record.metadata.scheduler_metadata_status == "valid"
                        and record.metadata.scheduler_current_sha == current_pr_subject
                    ),
                    None,
                )
                coder_checkpoint_is_current = bool(
                    current_coder_record is not None
                    and not any(
                        record.index > current_coder_record.index
                        and record.metadata.role in {"reviewer", "summary"}
                        for record in current_head_records
                    )
                )
                if coder_checkpoint_is_current and current_coder_record is not None:
                    scheduler_previous_sha = current_coder_record.metadata.scheduler_previous_sha
                elif current_head_records:
                    # Reviewer/audit records after a coder checkpoint mean the
                    # candidate is already in a same-head reconciliation
                    # round.  Never reuse the previous coder transition for a
                    # final sweep on this unchanged head.
                    scheduler_previous_sha = current_pr_subject
                else:
                    scheduler_previous_sha = next(
                        (
                            record.metadata.scheduler_current_sha
                            for record in reversed(historical_records)
                            if record.metadata.scheduler_metadata_status == "valid"
                            and record.metadata.scheduler_current_sha
                        ),
                        None,
                    )
                obligations = _scheduler_obligations(
                    pr_ledger_view(prior_unresolved_items),
                    required_reviewers=tuple(
                        agent_display_name(reviewer) for reviewer in configured_reviewers
                    ),
                )
                current_obligation_digest = hashlib.sha256(
                    repr(_prior_item_ledger_signature(prior_unresolved_items)).encode("utf-8")
                ).hexdigest()[:16]
                # The coder checkpoint is the conservative recovery latch for
                # every ledger, including reviewer-only ledgers. Qualification
                # and reconciliation summaries may add a newer digest for a
                # machine obligation, but must not narrow the existing
                # reviewer-only safety behavior.
                persisted_obligation_digests: list[str] = []
                if coder_checkpoint_is_current and current_coder_record is not None:
                    coder_digest = current_coder_record.metadata.scheduler_obligation_digest
                    if coder_digest is not None:
                        persisted_obligation_digests.append(coder_digest)
                if any(_is_machine_obligation(item) for item in prior_unresolved_items):
                    current_head_digest_records = tuple(
                        record
                        for record in current_head_records
                        if (
                            record.metadata.scheduler_metadata_status == "valid"
                            and record.metadata.scheduler_obligation_digest is not None
                        )
                    )
                    if current_head_digest_records:
                        persisted_obligation_digests.append(
                            current_head_digest_records[-1].metadata.scheduler_obligation_digest
                        )
                if any(
                    persisted_digest != current_obligation_digest
                    for persisted_digest in persisted_obligation_digests
                ):
                    if scheduler_capabilities.requires_primary:
                        automatic_fallback_reasons.append(
                            "scheduler obligation digest changed during recovery"
                        )
                    else:
                        scheduler_force_full = True
                    log(
                        config,
                        f"Round {round_number}: scheduler obligation digest changed during recovery; "
                        + (
                            "requesting an automatic full-board fallback"
                            if scheduler_capabilities.requires_primary
                            else "forcing the full reviewer board"
                        ),
                    )
                active_scopes = tuple(
                    sorted({path for obligation in obligations for path in (obligation.scope or ())})
                )
                if external_recovery_full_board:
                    classification = TransitionClassification(
                        "broad",
                        "external/unrecorded head advance requires full board",
                    )
                elif scheduler_metadata_recovery_full_board:
                    classification = TransitionClassification(
                        "broad",
                        (
                            "scheduler metadata is missing or invalid"
                            if scheduler_capabilities.requires_primary
                            else "scheduler metadata is missing or invalid; full board required"
                        ),
                    )
                elif scheduler_previous_sha is None:
                    classification = TransitionClassification("broad", "initial candidate requires the full board")
                elif scheduler_previous_sha == current_pr_subject:
                    if any(obligation.scope is None for obligation in obligations):
                        classification = TransitionClassification(
                            "broad", "the active obligation ledger is not reconstructible"
                        )
                    else:
                        classification = TransitionClassification(
                            "narrow", "same exact candidate head; recover missing reviewer work"
                        )
                else:
                    classification = _observe_pr_transition(
                        runner,
                        checkout=active_workdir(config),
                        previous_sha=scheduler_previous_sha,
                        current_sha=current_pr_subject,
                        scopes=active_scopes,
                        broad_rules=scheduler_contract.broad_rules,
                        obligations=obligations,
                    )
                final_sweep = final_sweep_pending or (
                    scheduler_previous_sha is not None
                    and not obligations
                ) or external_recovery_full_board
                if classification.narrow and scheduler_previous_sha != current_pr_subject:
                    # A reviewer may miss several narrow turns. Keep that
                    # transition narrow when the reviewer's own older span is
                    # observable; only missing/unavailable history is broad.
                    unreconstructible_history = [
                        name
                        for name in scheduler_contract.required_reviewers
                        # Under the staged policy a secondary that has never
                        # reviewed is not "returning": its first invocation
                        # always receives the complete base-to-head diff, so
                        # only reviewers with a prior record need span history.
                        if not (
                            scheduler_capabilities.phase_aware
                            and latest_reviewer_records.get(name) is None
                        )
                        and not _reviewer_history_is_reconstructible(
                            runner,
                            checkout=active_workdir(config),
                            record=latest_reviewer_records.get(name),
                            current_head_sha=current_pr_subject,
                        )
                    ]
                    if unreconstructible_history:
                        classification = TransitionClassification(
                            "broad",
                            "a returning reviewer's history could not be reconstructed",
                            tuple(sorted(unreconstructible_history)),
                        )
                scheduler_fallback_notes: tuple[str, ...] = ()
                if scheduler_capabilities.requires_primary:
                    assert panel_evidence is not None
                    if panel_evidence.opened:
                        # Post-panel: keep the conservative monotonic latch,
                        # restored only from automatic/legacy records after the
                        # qualified opening and raised by any automatic reason.
                        if panel_evidence.post_opening_automatic_latch:
                            scheduler_force_full = True
                        if automatic_fallback_reasons and not scheduler_force_full:
                            scheduler_force_full = True
                            log(
                                config,
                                f"Round {round_number}: post-panel automatic fallback raised the durable "
                                "force-full latch (source: automatic); the complete board is selected "
                                "for the rest of the run: " + "; ".join(automatic_fallback_reasons),
                            )
                        scheduler_fallback_notes = tuple(automatic_fallback_reasons)
                        snapshot_force_full = scheduler_force_full
                    else:
                        # Strict pre-panel: automatic reasons apply to this
                        # decision only and re-invoke just the primary with full
                        # context; nothing is latched.
                        notes = list(automatic_fallback_reasons)
                        if panel_evidence.prepanel_latch and not scheduler_operator_force_full:
                            notes.append(
                                "an unattributed or automatic force-full latch recorded before any "
                                "qualified panel opening was not honored; rerun with "
                                "--pr-review-force-full to restore the complete board"
                            )
                        if panel_evidence.unqualified_artifacts and not scheduler_operator_force_full:
                            notes.append(
                                "unqualified pre-approval panel history was ignored: "
                                + ", ".join(panel_evidence.unqualified_artifacts[:6])
                                + (" ..." if len(panel_evidence.unqualified_artifacts) > 6 else "")
                            )
                        scheduler_fallback_notes = tuple(notes)
                        snapshot_force_full = bool(automatic_fallback_reasons)
                        if automatic_fallback_reasons:
                            log(
                                config,
                                f"Round {round_number}: automatic fallback before any qualified panel "
                                "opening applies to this decision only (no latch): "
                                + "; ".join(automatic_fallback_reasons),
                            )
                else:
                    snapshot_force_full = (
                        scheduler_force_full
                        or scheduler_operator_force_full
                        or scheduler_metadata_recovery_full_board
                    )
                scheduler_snapshot = SchedulerSnapshot(
                    previous_sha=scheduler_previous_sha,
                    current_sha=current_pr_subject,
                    contract=scheduler_contract,
                    obligations=obligations,
                    force_full=snapshot_force_full,
                    phase=scheduler_checkpoint_phase,
                    operator_force_full=scheduler_operator_force_full,
                    panel_evidence=bool(panel_evidence is not None and panel_evidence.opened),
                    fallback_reasons=scheduler_fallback_notes,
                )
                try:
                    scheduler_decision = select_reviewers(
                        scheduler_snapshot,
                        classification,
                        qualifying_approvals=tuple(unchanged_head_approvals),
                        unavailable_reviewers=tuple(
                            agent_display_name(reviewer) for reviewer in unavailable_reviewer_failures
                        ),
                        final_sweep=final_sweep,
                        phase=scheduler_checkpoint_phase,
                    )
                except PrePanelSafetyError as exc:
                    stop_pre_panel(str(exc), round_number=round_number)
                    raise
                if not (skip_reviewers_for_recovery or conflict_pending):
                    pending_automatic_fallback_reasons.clear()
                if (
                    scheduler_capabilities.requires_primary
                    and panel_evidence is not None
                    and panel_evidence.opened
                    and scheduler_decision.phase == "full-board"
                    and not scheduler_operator_force_full
                    and not scheduler_force_full
                ):
                    # Post-panel fallback is monotonic: once an unsafe broad or
                    # ambiguous transition reactivates the complete board after a
                    # qualified opening, later heads keep it (source automatic)
                    # instead of dropping back to owner-scoped remediation.
                    scheduler_force_full = True
                    log(
                        config,
                        f"Round {round_number}: post-panel full-board fallback raised the durable "
                        "force-full latch (source: automatic): " + scheduler_decision.reason,
                    )
                scheduler_recorded_force_full, scheduler_recorded_force_full_source = (
                    _scheduler_recorded_force_full(
                        operator=scheduler_operator_force_full,
                        automatic=scheduler_force_full,
                    )
                )
                superseded_audit_text = ""
                if superseded_prepanel_reviews:
                    superseded_audit_text = (
                        " Superseded premature panel reviews (not resumed, not approvals, not "
                        "ledger items; replayed only as non-authoritative context): "
                        + "; ".join(
                            _describe_superseded_prepanel_review(record)
                            for record in superseded_prepanel_reviews.values()
                        )
                        + "."
                    )
                scheduler_diff_context = (
                    "\nScheduler audit record: the orchestrator classified the transition as "
                    f"{classification.kind} ({classification.reason}). Previous reviewed SHA: "
                    f"{scheduler_previous_sha or '(none)'}; current SHA: {current_pr_subject}. "
                    "Changed paths observed since that review: "
                    f"{', '.join(classification.changed_paths) or '(unavailable)'}. "
                    f"Selected phase: {scheduler_decision.phase}; primary: "
                    f"{scheduler_contract.primary_reviewer or '(none)'}; active owners: "
                    f"{', '.join(scheduler_decision.active_owners) or '(none)'}. "
                    f"Scheduling reason: {scheduler_decision.reason}. "
                    "Inspect the complete base-to-head diff independently; this summary is not a substitute.\n"
                )
                scheduler_calls_avoided += scheduler_decision.calls_avoided
                final_sweep_pending = False
                selected_reviewer_names = set(scheduler_decision.selected_reviewers)
                if evidence_pass is not None:
                    # Evidence-response and refresh passes always run the
                    # complete board at the same head (#1068).
                    selected_reviewer_names = {
                        agent_display_name(reviewer) for reviewer in configured_reviewers
                    }
                log(
                    config,
                    f"Round {round_number}: {scheduler_contract.policy} scheduler phase="
                    f"{scheduler_decision.phase} {scheduler_decision.reason}; "
                    f"selected={', '.join(scheduler_decision.selected_reviewers) or 'none'}; "
                    f"paused={', '.join(name for name, _reason in scheduler_decision.paused_reviewers) or 'none'}; "
                    f"force_full={scheduler_recorded_force_full} "
                    f"(source: {scheduler_recorded_force_full_source or 'none'}); "
                    f"panel_evidence={bool(panel_evidence is not None and panel_evidence.opened)}; "
                    f"checkpoint_phase={scheduler_checkpoint_phase or 'none'}; head={current_pr_subject}; "
                    f"primary={scheduler_contract.primary_reviewer or 'none'}; active_owners="
                    f"{', '.join(scheduler_decision.active_owners) or 'none'}"
                    + (f"; {pr_amendment_note}" if pr_amendment_note else ""),
                )
                # The amendment's activation round always persists a fresh
                # digest-bound scheduler decision (#943), even when every
                # remaining reviewer's review is reused and nobody is invoked.
                _pr_spool_preflight(selected_reviewer_names)
                amendment_checkpoint_due = bool(
                    pr_amendment_checkpoint_pending
                    and pr_contract_lineage.active_amendment is not None
                    and round_number == pr_contract_lineage.active_amendment.effective_from_round
                )
                if (
                    (scheduler_decision.selected_reviewers or amendment_checkpoint_due)
                    and not skip_reviewers_for_recovery
                    and not conflict_pending
                    # An evidence pass writes only its reviewer records and
                    # one terminal record, so an interrupted pass is ignorable.
                    and evidence_pass is None
                ):
                    pr_amendment_checkpoint_pending = False
                    pr_posted_checkpoint = bound_pr_metadata(
                        flow="pr", role="summary", agent="Orchestrator",
                        round_number=round_number, subject=current_pr_subject,
                        prior_items=prior_unresolved_items, phase="scheduler-prelaunch",
                        scheduler_contract=scheduler_contract.as_dict(),
                        reviewer_board_amendment_digest=pr_amendment_digest,
                        scheduler_previous_sha=scheduler_previous_sha,
                        scheduler_current_sha=current_pr_subject,
                        scheduler_obligation_digest=hashlib.sha256(
                            repr(_prior_item_ledger_signature(prior_unresolved_items)).encode("utf-8")
                        ).hexdigest()[:16],
                        scheduler_selected_reviewers=scheduler_decision.selected_reviewers,
                        scheduler_paused_reviewers=scheduler_decision.paused_reviewers,
                        scheduler_reasons=(scheduler_decision.reason, classification.reason),
                        scheduler_final_sweep=final_sweep,
                        scheduler_force_full=scheduler_recorded_force_full,
                        scheduler_force_full_source=scheduler_recorded_force_full_source,
                        scheduler_calls_avoided=scheduler_calls_avoided,
                        scheduler_phase=scheduler_decision.phase,
                        scheduler_primary_reviewer=scheduler_contract.primary_reviewer,
                        scheduler_approved_reviewers=tuple(sorted(unchanged_head_approvals)),
                        scheduler_active_owners=scheduler_decision.active_owners,
                        scheduler_scope_digest=hashlib.sha256(
                            repr(classification.changed_paths).encode("utf-8")
                        ).hexdigest()[:16],
                        **_architecture_metadata_fields(config),
                    )
                    post_pr_comment(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        body=_attach_round_metadata(
                            f"PR review scheduling audit: selected {', '.join(scheduler_decision.selected_reviewers) or 'none'}; "
                            f"paused {', '.join(name for name, _reason in scheduler_decision.paused_reviewers) or 'none'}; "
                            f"reason: {scheduler_decision.reason}; phase: {scheduler_decision.phase}; "
                            f"head: {current_pr_subject}; primary: "
                            f"{scheduler_contract.primary_reviewer or '(none)'}; active owners: "
                            f"{', '.join(scheduler_decision.active_owners) or '(none)'}; "
                            f"force-full: {scheduler_recorded_force_full} "
                            f"(source: {scheduler_recorded_force_full_source or 'none'}); "
                            "scheduler-policy calls avoided cumulatively: "
                            f"{scheduler_calls_avoided}.{superseded_audit_text}"
                            + (f" {pr_amendment_note}" if pr_amendment_note else ""),
                            pr_posted_checkpoint,
                        ),
                    )
            else:
                scheduler_previous_sha = None
                scheduler_decision = None
                scheduler_diff_context = ""
                selected_reviewer_names = {agent_display_name(reviewer) for reviewer in configured_reviewers}
                final_sweep = False
                scheduler_recorded_force_full, scheduler_recorded_force_full_source = False, None
                superseded_audit_text = ""
                _pr_spool_preflight(selected_reviewer_names)
            skip_reviewers_this_round = skip_reviewers_for_recovery or conflict_pending

            pr_fatal_errors: list[tuple[str, AgentLoopError]] = []
            pr_prep_failures: dict[AgentName, AgentLoopError] = {}
            pr_turn_results: dict[AgentName, _ReviewerTurnResult] = {}
            early_published_pr_reviewers: set[AgentName] = {
                reviewer
                for reviewer in configured_reviewers
                if (
                    (record := resumed_by_name.get(agent_display_name(reviewer))) is not None
                    and record.metadata.phase == "publication"
                )
            }
            shared_reviewer_pr_checks: PullRequestChecks | None = None
            reviewer_agents_by_name = {
                agent_display_name(reviewer): reviewer for reviewer in configured_reviewers
            }

            def _post_pr_reviewer_comment(
                reviewer_name: str,
                parsed: ParsedReview,
                *,
                review_output: str,
                model_used: str | None,
                identity: ValidatedAgentResponse | None = None,
                acquisition_outcome: str = "success",
                acquisition_returncode: int | None = None,
                new_items: tuple[UnresolvedReviewItem, ...] = (),
                phase: str | None = None,
            ) -> None:
                """Post one PR review using the same rendering and durable record."""
                if phase is None:
                    phase = (
                        EVIDENCE_RESPONSE_PHASE if evidence_pass is not None else "authoritative"
                    )
                # A spooled reviewer's bodies are frozen before the first post so
                # a rerun after an outage resumes them verbatim (#1258).
                publication_hook = (
                    {"publication": _pr_publication(reviewer_name)}
                    if phase == "publication" and identity is not None
                    else {}
                )
                post_pr_comment(
                    runner, config=config, pr_number=pr_number, **publication_hook,
                    body=_attach_round_metadata(
                        render_public_agent_comment(
                            kind="pr_review", parsed=parsed,
                            agent=reviewer_agents_by_name[reviewer_name],
                            human_requirements_resolved_flag=human_requirements_resolved(review_output),
                            prior_items=prior_unresolved_items, dispositions=parsed.dispositions,
                            config=config, model_used=model_used,
                        ),
                        bound_pr_metadata(
                            flow="pr", role="reviewer", agent=reviewer_name,
                            round_number=round_number, subject=current_pr_subject,
                            prior_items=prior_unresolved_items, dispositions=parsed.dispositions,
                            new_items=new_items, state=parsed.state, model_used=model_used,
                            sub_item_degradations=tuple(parsed.sub_item_degradations),
                            **(_metadata_identity_fields(identity) if identity is not None else {}),
                            acquisition_outcome=acquisition_outcome,
                            acquisition_returncode=acquisition_returncode,
                            surfaced_reviewer_requirement_ids=surfaced_reviewer_requirement_ids,
                            **_architecture_metadata_fields(config, result=parsed),
                            approved_plan_hash=(
                                approved_plan_context.plan_hash
                                if approved_plan_context is not None
                                else None
                            ),
                            approved_plan_subject=(
                                approved_plan_context.plan_subject
                                if approved_plan_context is not None
                                else None
                            ),
                            phase=phase,
                            evidence_requests=parsed.exact_head_evidence_requests,
                            canonical_reviewer_response=(review_output if phase == "publication" else None),
                            scheduler_contract=(scheduler_contract.as_dict() if selective_policy else None),
                            reviewer_board_amendment_digest=(pr_amendment_digest if selective_policy else None),
                            scheduler_previous_sha=(scheduler_previous_sha if selective_policy else None),
                            scheduler_current_sha=(current_pr_subject if selective_policy else None),
                            scheduler_obligation_digest=(
                                hashlib.sha256(
                                    repr(_prior_item_ledger_signature(prior_unresolved_items)).encode("utf-8")
                                ).hexdigest()[:16]
                                if selective_policy else None
                            ),
                            scheduler_selected_reviewers=(
                                tuple(sorted(selected_reviewer_names)) if selective_policy else ()
                            ),
                            scheduler_paused_reviewers=(
                                scheduler_decision.paused_reviewers
                                if selective_policy and scheduler_decision is not None else ()
                            ),
                            scheduler_reasons=(
                                (scheduler_decision.reason, classification.reason)
                                if selective_policy and scheduler_decision is not None else ()
                            ),
                            scheduler_final_sweep=(final_sweep if selective_policy else None),
                            scheduler_force_full=(scheduler_recorded_force_full if selective_policy else None),
                            scheduler_force_full_source=(
                                scheduler_recorded_force_full_source if selective_policy else None
                            ),
                            scheduler_calls_avoided=(scheduler_calls_avoided if selective_policy else None),
                            scheduler_phase=(scheduler_decision.phase if selective_policy and scheduler_decision is not None else None),
                            scheduler_primary_reviewer=(scheduler_contract.primary_reviewer if selective_policy else None),
                            scheduler_approved_reviewers=(tuple(sorted(unchanged_head_approvals)) if selective_policy else ()),
                            scheduler_active_owners=(scheduler_decision.active_owners if selective_policy and scheduler_decision is not None else ()),
                            scheduler_scope_digest=(hashlib.sha256(repr(classification.changed_paths).encode("utf-8")).hexdigest()[:16] if selective_policy else None),
                        ),
                    ),
                )

            def _pr_reviewer_prelaunch_kind(reviewer: AgentName) -> str:
                # Pure classification, no agent calls: mirrors the resumed/
                # carried-approval branching below so the pre-round parallel
                # launch list matches what the per-reviewer loop would decide.
                reviewer_name = agent_display_name(reviewer)
                if reviewer in unavailable_reviewer_failures:
                    return "skip"
                if resumed_by_name.get(reviewer_name) is not None:
                    return "resumed"
                if selective_policy and reviewer_name not in selected_reviewer_names:
                    return "skip"
                prior_approval = unchanged_head_approvals.get(reviewer_name)
                if (
                    evidence_pass is None
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

            reviewer_fresh_contexts = {
                reviewer: _returning_reviewer_context(
                    runner,
                    reviewer=reviewer,
                    checkout=active_workdir(config),
                    current_head_sha=current_pr_subject,
                    current_round=round_number,
                    latest_reviewer_records=latest_reviewer_records,
                )
                if _reviewer_needs_fresh_context(
                    reviewer,
                    selective_policy=selective_policy,
                    current_head_sha=current_pr_subject,
                    current_round=round_number,
                    latest_reviewer_records=latest_reviewer_records,
                )
                else ""
                for reviewer in configured_reviewers
            }

            pr_launch_phase = (
                scheduler_decision.phase if selective_policy and scheduler_decision is not None else None
            )
            # Today's peer set, built from every posted same-round record
            # including records resume rejected: their bodies are just as
            # public.  It is only ever widened below, never narrowed.
            pr_today_peers = tuple(
                reviewer for reviewer in configured_reviewers
                if (record := completed_by_name.get(agent_display_name(reviewer))) is not None
                and record.metadata.phase == "publication"
                and record.metadata.scheduler_phase == pr_launch_phase
            )
            pr_round_public_peers = pr_today_peers
            pr_recovery_context: RecoveryVisibilityContext | None = None
            if selective_policy and scheduler_decision is not None:
                def _derive_pr_visibility_evidence(records: Sequence[PostedRoundRecord]) -> PrPanelEvidence:
                    return _derive_pr_panel_evidence(
                        records,
                        primary_reviewer=scheduler_contract.primary_reviewer,
                        required_reviewers=scheduler_contract.required_reviewers,
                    )

                pr_recovery_context = RecoveryVisibilityContext(
                    primary_reviewer=(
                        scheduler_contract.primary_reviewer
                        if scheduler_capabilities.requires_primary else None
                    ),
                    launch_phase=pr_launch_phase,
                    launching=tuple(sorted(selected_reviewer_names)),
                    derive_evidence=_derive_pr_visibility_evidence,
                    target_opening=scheduler_capabilities.requires_primary,
                    removed_opening_fingerprint=(None, None, None),
                )
            if (
                pr_recovery_context is not None
                and evidence_pass is None
                and pr_posted_checkpoint is not None
            ):
                # Panel independence is about what a launching reviewer can
                # read, not the scheduler phase a record was published under
                # (#1156).  Visibility is read from the complete history with
                # this invocation's own checkpoint included, so opening
                # evidence is never stale.
                # A failed transport read falls back to the pre-post history plus
                # the posted checkpoint; history that was read but cannot be
                # decoded makes visibility unknowable, so stop before any
                # reviewer launches (as the round-start decode failure does).
                pr_fresh_records: tuple[PostedRoundRecord, ...] | None = None
                try:
                    pr_fresh_comments = get_pr_review_context(
                        runner, config=config, pr_number=pr_number
                    ).comments
                except AgentLoopError:
                    pr_fresh_comments = None
                try:
                    if pr_fresh_comments is not None:
                        pr_fresh_records = _extract_round_metadata_records(
                            pr_fresh_comments, flow="pr"
                        )
                    pr_base_records = _extract_round_metadata_records(pr_comments, flow="pr")
                except AgentLoopError as exc:
                    if scheduler_capabilities.requires_primary:
                        stop_pre_panel(
                            undecodable_history_message(exc), round_number=round_number
                        )
                    raise
                pr_visibility_records, _pr_checkpoint_present = _visibility_snapshot(
                    fresh_records=pr_fresh_records,
                    base_records=pr_base_records,
                    base_length=len(pr_comments),
                    checkpoint=pr_posted_checkpoint,
                )
                pr_visibility_evidence = (
                    _derive_pr_visibility_evidence(pr_visibility_records)
                    if scheduler_capabilities.requires_primary else None
                )
                pr_visible_names = visible_peer_names(
                    pr_visibility_records,
                    opening_source=(
                        pr_visibility_evidence.opening_source
                        if pr_visibility_evidence is not None else None
                    ),
                    flow="pr", round_number=round_number, subject=current_pr_subject,
                    reviewer_names=[agent_display_name(r) for r in configured_reviewers],
                    primary_reviewer=pr_recovery_context.primary_reviewer,
                    launch_phase=pr_launch_phase,
                    launching=pr_recovery_context.launching,
                    checkpoint_index=latest_round_checkpoint_index(
                        pr_visibility_records, flow="pr",
                        round_number=round_number, subject=current_pr_subject,
                    ),
                )
                pr_round_public_peers = tuple(
                    reviewer for reviewer in configured_reviewers
                    if reviewer in pr_today_peers
                    or agent_display_name(reviewer) in pr_visible_names
                )
            # A round that holds withheld outcomes began as a parallel round;
            # finish it through the same withhold-then-publish launcher even
            # when this run is sequential (#1025).
            pr_round_parallel = (
                config.review_parallel or pr_round_spool.has_records()
            ) and evidence_pass is None
            def _pr_recovery_fingerprint(remaining: Sequence[object]) -> object:
                """The qualified panel opening, identified by its comment, not its index."""
                if not scheduler_capabilities.requires_primary:
                    return None
                evidence = _derive_pr_panel_evidence(
                    _extract_round_metadata_records(remaining, flow="pr"),
                    primary_reviewer=scheduler_contract.primary_reviewer,
                    required_reviewers=scheduler_contract.required_reviewers,
                )
                opening = (
                    remaining[evidence.opening_index]
                    if evidence.opening_index is not None
                    else None
                )
                return (
                    evidence.opening_source,
                    getattr(opening, "created_at", None),
                    getattr(opening, "body", None),
                )

            def _pr_round_recovery() -> PartialRoundRecovery:
                return compute_partial_round_recovery(
                    snapshot=pr_comments,
                    read_rest=lambda: read_rest_issue_comments(
                        runner, config=config, issue_number=pr_number,
                        purpose="the partial review round's comment ids cannot be listed",
                        reject_empty_output=True,
                    ),
                    flow="pr", round_number=round_number, subject=current_pr_subject,
                    context=pr_recovery_context,
                    reviewer_names=[agent_display_name(r) for r in configured_reviewers],
                    resume=lambda remaining: _resume_pr_round(
                        remaining,
                        head_sha=pr_metadata.head_sha,
                        configured_reviewers=configured_reviewers,
                        reconciliation_mode=(
                            "owner-scoped"
                            if scheduler_capabilities.owner_scoped_reconciliation
                            else "aggregate"
                        ),
                        trusted_actor=_cached_trusted_actor(runner),
                        review_unrecorded_head=config.review_unrecorded_head,
                    ),
                    fingerprint=_pr_recovery_fingerprint,
                )

            if not pr_round_parallel and not skip_reviewers_this_round and evidence_pass is None:
                _refuse_partial_round_before_sequential_turns(
                    spool=pr_round_spool,
                    fresh_turn_reviewers=[
                        reviewer for reviewer in configured_reviewers
                        if _pr_reviewer_prelaunch_kind(reviewer) == "turn"
                    ],
                    public_peers=pr_round_public_peers,
                    already_posted=tuple(
                        reviewer for reviewer in configured_reviewers
                        if agent_display_name(reviewer) in completed_by_name
                    ),
                    recovery=_pr_round_recovery,
                )
            if pr_round_parallel and not skip_reviewers_this_round:
                pending_pr_reviewers = [
                    reviewer for reviewer in configured_reviewers
                    if _pr_reviewer_prelaunch_kind(reviewer) == "turn"
                ]
                if pending_pr_reviewers:
                    launchable_pr_reviewers: list[AgentName] = []
                    for reviewer in pending_pr_reviewers:
                        reviewer_name = agent_display_name(reviewer)
                        if (
                            reviewer_name not in completed_by_name
                            and pr_round_spool.load(reviewer_name) is not None
                        ):
                            # A withheld same-round outcome is replayed, not re-run.
                            launchable_pr_reviewers.append(reviewer)
                            continue
                        try:
                            sync_reviewer_pr_before_review(config, runner, reviewer, pr_number, pr_metadata)
                        except AgentLoopError as exc:
                            log(
                                config,
                                f"Round {round_number}: {reviewer_name} failed pre-review PR sync "
                                f"({getattr(exc, 'failure_category', None) or 'error'}); skipping its "
                                "launch while the remaining reviewers still run",
                            )
                            pr_prep_failures[reviewer] = exc
                            continue
                        launchable_pr_reviewers.append(reviewer)
                    if launchable_pr_reviewers:
                        # One shared checks snapshot for the whole round (#594),
                        # instead of one fetch per reviewer as sequential mode
                        # does, so every concurrently launched prompt and the
                        # post-turn pending-CI-only downgrade agree.
                        shared_reviewer_pr_checks = get_pr_checks(
                            runner, config=config, metadata=pr_metadata
                        )
                        if managed_ci_active(pr_metadata):
                            shared_reviewer_pr_checks = intermediate_managed_checks(
                                shared_reviewer_pr_checks
                            )
                        pr_parallel_compact_tail = (
                            CompactPrReviewTailContext(
                                head_sha=pr_metadata.head_sha,
                                round_number=round_number,
                            )
                            if use_compact_pr_context
                            else None
                        )
                        pr_prompts = {
                            reviewer: build_review_prompt(
                                pr_number,
                                round_number,
                                config,
                                reviewer=reviewer,
                                pr_metadata=pr_metadata,
                                pr_checks=shared_reviewer_pr_checks,
                                memory=memory,
                                issue_context=issue_context,
                                human_requirements=human_requirements,
                                unresolved_items=prior_unresolved_items,
                                compact_context=(
                                    use_compact_pr_context
                                    and not reviewer_fresh_contexts[reviewer]
                                ),
                                compact_prior=(
                                    CompactPriorContext(tuple(pr_compact_prior_summaries))
                                    if use_compact_pr_context
                                    else None
                                ),
                                compact_tail=pr_parallel_compact_tail,
                                coder_followup_context=(
                                    coder_followup_context
                                    + scheduler_diff_context
                                    + reviewer_fresh_contexts[reviewer]
                                ),
                                approved_plan_context=approved_plan_context,
                                parent_issue_context=parent_issue_context,
                                superseded_prepanel_review_context=_superseded_prepanel_review(
                                    superseded_prepanel_reviews.get(agent_display_name(reviewer))
                                ),
                                sweep_context=pr_step_back_sweeps.get(
                                    agent_display_name(reviewer), ""
                                ),
                            )
                            for reviewer in launchable_pr_reviewers
                        }
                        launchable_pr_names = [
                            agent_display_name(reviewer) for reviewer in launchable_pr_reviewers
                        ]
                        log(
                            config,
                            f"Round {round_number}: invoking {', '.join(launchable_pr_names)} "
                            f"in parallel on PR #{pr_number}",
                        )

                        def _pr_reviewer_worker(reviewer: AgentName) -> _ReviewerTurnResult:
                            reviewer_name = agent_display_name(reviewer)
                            try:
                                response = _run_validated_agent(
                                    runner,
                                    agent=reviewer,
                                    config=config,
                                    prompt=pr_prompts[reviewer],
                                    session_id=(
                                        None
                                        if use_compact_pr_context or reviewer_fresh_contexts[reviewer]
                                        else reviewer_session_ids.get(reviewer)
                                    ),
                                    marker_description="<!-- AGENT_STATE: approved|blocking -->",
                                    **_pr_review_validators(reviewer_name),
                                    usage_context=usage_context,
                                    use_repair=True,
                                    repair_expected_kind="pr_review",
                                    repair_reviewer_requirement_ids=_surfaced_reviewer_requirement_ids(
                                        human_requirements,
                                        requirement_scope="PR requirements",
                                    ),
                                    repair_allowed_prior_item_ids=tuple(
                                        item.item_id for item in prior_unresolved_items
                                    ),
                                    repair_evidence_row_ids=approved_matrix_row_ids(approved_plan_context) or (),
                                    ledger_incomplete=round_ledger_incomplete,
                                    repair_resolved_history_item_ids=round_resolved_history_item_ids,
                                    role="reviewer",
                                    operation_description="PR review",
                                    reask_on_prior_disposition_omission=True,
                                    reask_on_missing_judgement_field=True,
                                )
                            except AgentLoopError as exc:
                                # Includes QuotaResetExceededError: captured here
                                # and re-raised on the main thread with priority.
                                return _ReviewerTurnResult(reviewer_name=reviewer_name, error=exc)
                            return _ReviewerTurnResult(reviewer_name=reviewer_name, response=response)

                        def _pr_publication_parsed(turn: _ReviewerTurnResult) -> ParsedReview | None:
                            if turn.error is not None or turn.response is None:
                                return None
                            parsed = turn.response.marker_value
                            assert isinstance(parsed, ParsedReview)
                            parsed = dataclasses_replace(
                                parsed,
                                followups=_drop_repeated_carried_future_followups(
                                    parsed.followups, prior_items=prior_unresolved_items,
                                    dispositions=parsed.dispositions,
                                ),
                            )
                            if parsed.state == "blocking" and _is_pending_ci_only_review(parsed, shared_reviewer_pr_checks):
                                parsed = dataclasses_replace(parsed, state="approved", blocking_items=())
                            if parsed.state == "blocking" and _is_infrastructure_ci_only_review(parsed, shared_reviewer_pr_checks):
                                parsed = dataclasses_replace(parsed, state="approved", blocking_items=())
                            if parsed.state == "blocking":
                                parsed = _normalize_approval_gated_managed_ci_review(
                                    parsed,
                                    prior_items=prior_unresolved_items,
                                    pr_checks=shared_reviewer_pr_checks,
                                    current_head_sha=current_pr_subject,
                                )
                            if _is_evidence_only_blocking_review(parsed, prior_unresolved_items):
                                parsed = dataclasses_replace(parsed, state="approved")
                            return parsed

                        def _pr_failure_is_fatal(error: AgentLoopError) -> bool:
                            # Mirrors the settlement below: fatal failures stop the
                            # round and are re-invoked by a rerun; others settle
                            # the reviewer as unavailable for this round.
                            return len(configured_reviewers) == 1 or getattr(
                                error, "failure_category", None
                            ) in {None, "deterministic"}

                        def _pr_retry_bound(reviewer: AgentName, turn: _ReviewerTurnResult) -> AgentLoopError | None:
                            if turn.error is not None:
                                return turn.error if _pr_failure_is_fatal(turn.error) else None
                            parsed = _pr_publication_parsed(turn)
                            if (
                                parsed is not None
                                and _is_incomplete_pr_review(parsed)
                                and len(configured_reviewers) == 1
                            ):
                                return _incomplete_pr_review_error(agent_display_name(reviewer))
                            return None

                        def _publish_pr_completion(reviewer: AgentName, turn: _ReviewerTurnResult) -> bool:
                            """Publish a PR review after the round's workers return; settlement remains below."""
                            parsed = _pr_publication_parsed(turn)
                            if parsed is None or _is_incomplete_pr_review(parsed):
                                return False
                            assert turn.response is not None
                            reviewer_name = agent_display_name(reviewer)
                            _post_pr_reviewer_comment(
                                reviewer_name, parsed, review_output=turn.response.text,
                                model_used=turn.response.model_used,
                                identity=turn.response,
                                acquisition_outcome=turn.response.acquisition_outcome,
                                acquisition_returncode=turn.response.acquisition_returncode,
                                phase="publication",
                            )
                            early_published_pr_reviewers.add(reviewer)
                            return True

                        pr_turn_results = _launch_reviewer_turns(
                            runner,
                            launchable_pr_reviewers,
                            thread_name_prefix=f"pr-review-r{round_number}",
                            run_turn=_pr_reviewer_worker,
                            on_completion=_publish_pr_completion,
                            spool=pr_round_spool,
                            replay_turn=lambda reviewer, fields: _replay_spooled_review(
                                runner, config=config, reviewer=reviewer, fields=fields,
                                validators=_pr_review_validators(agent_display_name(reviewer)),
                            ),
                            public_peers=pr_round_public_peers,
                            recovery=_pr_round_recovery,
                            retry_bound=_pr_retry_bound,
                            max_workers=None if config.review_parallel else 1,
                            # A spooled reviewer skipped the pre-review sync
                            # above; sync it if its replay falls back to a turn.
                            prepare_fallback=lambda reviewer: sync_reviewer_pr_before_review(
                                config, runner, reviewer, pr_number, pr_metadata
                            ),
                            already_posted=tuple(
                                reviewer for reviewer in configured_reviewers
                                if agent_display_name(reviewer) in completed_by_name
                            ),
                            prelaunch_failures={
                                reviewer: _ReviewerTurnResult(
                                    reviewer_name=agent_display_name(reviewer), error=error
                                )
                                for reviewer, error in pr_prep_failures.items()
                            },
                            configured_order=configured_reviewers,
                            config=config,
                        )
                    else:
                        log(
                            config,
                            f"Round {round_number}: every pending reviewer failed pre-review "
                            "PR sync; nothing to launch in parallel this round",
                        )

            # Actual models of reviewers that reviewed in THIS round (fresh or
            # same-round resumed); carried prior-round approvals are excluded (#1236).
            round_review_models: dict[str, str | None] = {}
            for reviewer in (() if skip_reviewers_this_round else configured_reviewers):
                reviewer_name = agent_display_name(reviewer)
                reviewer_pr_checks = shared_reviewer_pr_checks
                if (
                    selective_policy
                    and reviewer_name not in selected_reviewer_names
                    and reviewer_name not in resumed_by_name
                ):
                    log(
                        config,
                        f"Round {round_number}: pausing {reviewer_name}; "
                        "its prior approval is historical for this candidate",
                    )
                    continue
                if reviewer in unavailable_reviewer_failures:
                    log(
                        config,
                        f"Round {round_number}: skipping unavailable reviewer {reviewer_name}; "
                        "it already exhausted its retry budget in this run",
                    )
                    continue
                resumed_record = resumed_by_name.get(reviewer_name)
                carried_approval_record: PostedRoundRecord | None = None
                if resumed_record is not None:
                    canonical_review_output = resumed_record.metadata.canonical_reviewer_response
                    review_output = canonical_review_output or resumed_record.body
                    review_model_used = resumed_record.metadata.model_used
                    review_acquisition_outcome = resumed_record.metadata.acquisition_outcome
                    review_acquisition_returncode = resumed_record.metadata.acquisition_returncode
                    structured_review = (
                        parse_structured_pr_review(review_output, reviewer=reviewer_name, architecture_status_mode="legacy")
                        if canonical_review_output is not None
                        else None
                    )
                    reparsed_review = structured_review or parse_review(
                        review_output, reviewer=reviewer_name
                    )
                    resumed_impact, resumed_records = _resumed_review_architecture(
                        resumed_record.metadata,
                        structured_review.architecture_impact if structured_review is not None else None,
                    )
                    parsed_review = ParsedReview(
                        state=resumed_record.metadata.state or parse_agent_state(review_output),
                        summary=reparsed_review.summary,
                        blocking_items=reparsed_review.blocking_items,
                        followups=reparsed_review.followups,
                        dispositions=resumed_record.metadata.dispositions,
                        architecture_impact=resumed_impact,
                        architecture_impact_degradations=resumed_records,
                        exact_head_evidence_requests=resumed_record.metadata.evidence_requests,
                        sub_item_degradations=resumed_record.metadata.sub_item_degradations,
                    )
                    review_state = parsed_review.state
                    reviewer_new_unresolved_items = list(resumed_record.metadata.new_items)
                    log(config, f"Round {round_number}: resuming {reviewer_name}'s completed review")
                elif (
                    evidence_pass is None
                    and (prior_approval := unchanged_head_approvals.get(reviewer_name)) is not None
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
                    carried_approval_record = prior_approval
                    review_output = prior_approval.body
                    review_model_used = prior_approval.metadata.model_used
                    review_acquisition_outcome = prior_approval.metadata.acquisition_outcome
                    review_acquisition_returncode = prior_approval.metadata.acquisition_returncode
                    reparsed_review = parse_review(review_output, reviewer=reviewer_name)
                    parsed_review = ParsedReview(
                        state="approved",
                        summary=review_freeform_summary_text(review_output),
                        blocking_items=reparsed_review.blocking_items,
                        followups=reparsed_review.followups,
                        dispositions=(),
                    )
                    review_state = parsed_review.state
                    reviewer_new_unresolved_items = []
                    log(
                        config,
                        f"Round {round_number}: skipping {reviewer_name}; it approved unchanged "
                        f"PR head {current_pr_subject} in round {prior_approval.metadata.round_number}",
                    )
                elif pr_round_parallel:
                    review_failure: AgentInvocationError | AgentLoopError | None = (
                        pr_prep_failures.get(reviewer)
                    )
                    turn = pr_turn_results.get(reviewer) if review_failure is None else None
                    if turn is not None and turn.error is not None:
                        review_failure = turn.error
                    if review_failure is not None:
                        if (
                            len(configured_reviewers) == 1
                            or getattr(review_failure, "failure_category", None) in {None, "deterministic"}
                        ):
                            pr_fatal_errors.append((reviewer_name, review_failure))
                            continue
                        unavailable_reviewer_failures[reviewer] = review_failure
                        category = getattr(review_failure, "failure_category", None) or "unknown"
                        log(
                            config,
                            f"Round {round_number}: {reviewer_name} became unavailable "
                            f"({category}); finishing this round's remaining reviews, then stopping",
                        )
                        continue
                    assert turn is not None and turn.response is not None
                    review_response = turn.response
                    reviewer_pr_checks = shared_reviewer_pr_checks
                    review_output = review_response.text
                    review_model_used = review_response.model_used
                    review_acquisition_outcome = review_response.acquisition_outcome
                    review_acquisition_returncode = review_response.acquisition_returncode
                    reviewer_session_ids[reviewer] = review_response.session_id
                    parsed_review = review_response.marker_value
                    assert isinstance(parsed_review, ParsedReview)
                    parsed_review = dataclasses_replace(
                        parsed_review,
                        followups=_drop_repeated_carried_future_followups(
                            parsed_review.followups,
                            prior_items=prior_unresolved_items,
                            dispositions=parsed_review.dispositions,
                        ),
                    )
                    review_state = parsed_review.state
                    reviewer_new_unresolved_items = []
                else:
                    reviewer_fresh_context = bool(reviewer_fresh_contexts[reviewer])
                    reviewer_compact_context = use_compact_pr_context and not reviewer_fresh_context
                    context_mode = "compact" if reviewer_compact_context else "full"
                    log(
                        config,
                        f"Round {round_number}: {reviewer_name} reviewing PR #{pr_number} "
                        f"(context mode: {context_mode})",
                    )
                    sync_reviewer_pr_before_review(config, runner, reviewer, pr_number, pr_metadata)
                    reviewer_pr_checks = get_pr_checks(
                        runner, config=config, metadata=pr_metadata
                    )
                    if managed_ci_active(pr_metadata):
                        reviewer_pr_checks = intermediate_managed_checks(reviewer_pr_checks)
                    compact_tail = (
                        CompactPrReviewTailContext(
                            head_sha=pr_metadata.head_sha,
                            round_number=round_number,
                        )
                        if use_compact_pr_context
                        else None
                    )
                    sequential_pr_validators = _architecture_mode_validators(lambda mode: lambda text, reviewer_name=reviewer_name, items=prior_unresolved_items: _validate_review_response(
                        text,
                        reviewer=reviewer_name,
                        unresolved_items=items,
                        current_round_items=round_new_unresolved_items, architecture_status_mode=mode,
                        approved_matrix_row_ids=approved_matrix_row_ids(approved_plan_context),
                    ))
                    review_response, review_failure = _capture_agent_invocation(
                        lambda: _same_round_replay_or_invoke(
                            runner,
                            config=config,
                            reviewer=reviewer,
                            spool=pr_round_spool,
                            public_peers=pr_round_public_peers,
                            recovery=_pr_round_recovery,
                            validators=sequential_pr_validators,
                            already_posted=reviewer_name in completed_by_name,
                            invoke=lambda: _run_validated_agent(
                            runner,
                            agent=reviewer,
                            config=config,
                            prompt=build_review_prompt(
                                pr_number,
                                round_number,
                                config,
                                reviewer=reviewer,
                                pr_metadata=pr_metadata,
                                pr_checks=reviewer_pr_checks,
                                memory=memory,
                                issue_context=issue_context,
                                human_requirements=human_requirements,
                                unresolved_items=prior_unresolved_items,
                                compact_context=reviewer_compact_context,
                                compact_prior=(
                                    CompactPriorContext(tuple(pr_compact_prior_summaries))
                                    if use_compact_pr_context
                                    else None
                                ),
                                compact_tail=compact_tail,
                                coder_followup_context=(
                                    coder_followup_context
                                    + scheduler_diff_context
                                    + reviewer_fresh_contexts[reviewer]
                                ),
                                approved_plan_context=approved_plan_context,
                                parent_issue_context=parent_issue_context,
                                superseded_prepanel_review_context=_superseded_prepanel_review(
                                    superseded_prepanel_reviews.get(reviewer_name)
                                ),
                                sweep_context=pr_step_back_sweeps.get(reviewer_name, ""),
                            ),
                            session_id=(
                                None
                                if reviewer_compact_context or reviewer_fresh_context
                                else reviewer_session_ids.get(reviewer)
                            ),
                            marker_description="<!-- AGENT_STATE: approved|blocking -->",
                            **sequential_pr_validators,
                            usage_context=usage_context,
                            use_repair=True,
                            repair_expected_kind="pr_review",
                            repair_reviewer_requirement_ids=_surfaced_reviewer_requirement_ids(
                                human_requirements,
                                requirement_scope="PR requirements",
                            ),
                            repair_allowed_prior_item_ids=tuple(
                                item.item_id for item in prior_unresolved_items
                            ),
                            repair_evidence_row_ids=approved_matrix_row_ids(approved_plan_context) or (),
                            ledger_incomplete=round_ledger_incomplete,
                            repair_resolved_history_item_ids=round_resolved_history_item_ids,
                            role="reviewer",
                            operation_description="PR review",
                            reask_on_prior_disposition_omission=True,
                            reask_on_missing_judgement_field=True,
                            ),
                        )
                    )
                    if review_failure is not None:
                        if (
                            len(configured_reviewers) == 1
                            or review_failure.failure_category in {None, "deterministic"}
                        ):
                            raise review_failure
                        unavailable_reviewer_failures[reviewer] = review_failure
                        category = review_failure.failure_category or "unknown"
                        log(
                            config,
                            f"Round {round_number}: {reviewer_name} became unavailable "
                            f"({category}); finishing this round's remaining reviews, then stopping",
                        )
                        continue
                    assert review_response is not None
                    review_output = review_response.text
                    review_model_used = review_response.model_used
                    review_acquisition_outcome = review_response.acquisition_outcome
                    review_acquisition_returncode = review_response.acquisition_returncode
                    reviewer_session_ids[reviewer] = review_response.session_id
                    parsed_review = review_response.marker_value
                    assert isinstance(parsed_review, ParsedReview)
                    parsed_review = dataclasses_replace(
                        parsed_review,
                        followups=_drop_repeated_carried_future_followups(
                            parsed_review.followups,
                            prior_items=prior_unresolved_items,
                            dispositions=parsed_review.dispositions,
                        ),
                    )
                    review_state = parsed_review.state
                    reviewer_new_unresolved_items = []

                if carried_approval_record is None:
                    round_review_models[reviewer_name] = review_model_used

                if (
                    resumed_record is None
                    and review_state == "blocking"
                    and _is_pending_ci_only_review(parsed_review, reviewer_pr_checks)
                ):
                    log(
                        config,
                        f"Round {round_number}: {reviewer_name} blocking review only restates "
                        f"GitHub check status ({reviewer_pr_checks.state}); treating as approved instead "
                        "of starting a new coder follow-up round",
                    )
                    parsed_review = dataclasses_replace(parsed_review, state="approved", blocking_items=())
                    review_state = parsed_review.state

                if (
                    resumed_record is None
                    and review_state == "blocking"
                    and _is_infrastructure_ci_only_review(parsed_review, reviewer_pr_checks)
                ):
                    log(
                        config,
                        f"Round {round_number}: {reviewer_name} blocking review only restates "
                        "an external CI infrastructure stall; treating as approved instead of "
                        "starting a new coder follow-up round",
                    )
                    parsed_review = dataclasses_replace(parsed_review, state="approved", blocking_items=())
                    review_state = parsed_review.state

                # Intentionally include resumed records: their durable blocking
                # state can represent only the approval-gated managed-CI wait.
                # Normalizing that state in memory lets settlement continue to
                # qualification without re-posting a duplicate reviewer comment.
                if review_state == "blocking":
                    if reviewer_pr_checks is None:
                        reviewer_pr_checks = get_pr_checks(
                            runner, config=config, metadata=pr_metadata
                        )
                        if managed_ci_active(pr_metadata):
                            reviewer_pr_checks = intermediate_managed_checks(
                                reviewer_pr_checks
                            )
                    normalized_review = _normalize_approval_gated_managed_ci_review(
                        parsed_review,
                        prior_items=prior_unresolved_items,
                        pr_checks=reviewer_pr_checks,
                        current_head_sha=current_pr_subject,
                    )
                    if normalized_review.state == "approved":
                        log(
                            config,
                            f"Round {round_number}: {reviewer_name} blocking review only restates "
                            "an approval-gated managed exact-head wait; treating the code review "
                            "as approved so the orchestrator can dispatch qualification",
                        )
                        parsed_review = normalized_review
                        review_state = parsed_review.state

                if review_state == "blocking" and _is_evidence_only_blocking_review(
                    parsed_review, prior_unresolved_items
                ):
                    # Missing human evidence is not a code blocker (#1068): the
                    # code review approves and the evidence stays in the ledger.
                    log(
                        config,
                        f"Round {round_number}: {reviewer_name} blocks only on human-only "
                        "exact-head evidence; treating the code review as approved and keeping "
                        "the evidence as a deferred final barrier",
                    )
                    parsed_review = dataclasses_replace(parsed_review, state="approved")
                    review_state = parsed_review.state

                if _is_incomplete_pr_review(parsed_review):
                    if len(configured_reviewers) == 1:
                        incomplete_pr_review_error = _incomplete_pr_review_error(reviewer_name)
                        if pr_round_parallel:
                            pr_fatal_errors.append((reviewer_name, incomplete_pr_review_error))
                            continue
                        raise incomplete_pr_review_error
                    unavailable_reviewer_failures[reviewer] = AgentInvocationError(
                        f"{reviewer_name} reported an incomplete PR review without actionable "
                        "blocking items or Same-PR follow-ups.",
                        failure_category="agent-unavailable",
                    )
                    log(
                        config,
                        f"Round {round_number}: {reviewer_name} did not complete its PR review "
                        "and reported no actionable blocking items or Same-PR follow-ups; "
                        "finishing this round's remaining reviews, then stopping",
                    )
                    continue

                if parsed_review.summary.strip():
                    summary_record = resumed_record or carried_approval_record
                    summary_round = summary_record.metadata.round_number if summary_record else round_number
                    summary_head = summary_record.metadata.subject if summary_record else current_pr_subject
                    reviewer_summaries[reviewer_name] = _reviewer_summary_context(
                        reviewer_name,
                        parsed_review.summary,
                        round_number=summary_round,
                        head_sha=summary_head,
                    )
                for disposition in parsed_review.dispositions:
                    _record_prior_item_disposition(
                        prior_dispositions,
                        disposition,
                        flow="pr",
                        round_number=round_number,
                        subject=current_pr_subject,
                        reviewer_name=reviewer_name,
                    )
                if carried_approval_record is None and parsed_review.exact_head_evidence_requests:
                    round_evidence_requests.append(
                        (reviewer_name, parsed_review.exact_head_evidence_requests)
                    )
                blocking_summary = parsed_review.summary
                has_structured_blocking_content = bool(
                    parsed_review.blocking_items or parsed_review.followups.same_pr
                )
                has_active_carried_disposition = any(
                    disposition.disposition in {"blocking", "same-pr"}
                    for disposition in parsed_review.dispositions
                )
                has_blocking_summary = not has_structured_blocking_content and _should_record_new_blocking_item(
                    blocking_summary,
                    had_prior_items=bool(prior_unresolved_items),
                    had_dispositions=bool(parsed_review.dispositions),
                    has_active_carried_disposition=has_active_carried_disposition,
                )
                log(
                    config,
                    f"Round {round_number}: {reviewer_name} outcome is "
                    f"{_describe_pr_review_outcome(parsed_review, has_blocking_summary=has_blocking_summary)}",
                )
                if review_state == "blocking":
                    round_blocking_reviewer_names.append(reviewer_name)
                    if (
                        (resumed_record is None or resumed_record.metadata.phase == "publication")
                        and carried_approval_record is None
                    ):
                        if parsed_review.blocking_items:
                            for blocking_item in parsed_review.blocking_items:
                                tracked_item = _next_unresolved_item(
                                    item_number=next_unresolved_item_number,
                                    reviewer=blocking_item.reviewer,
                                    source_round=round_number,
                                    text=blocking_item.text,
                                    status="blocking",
                                    fix_scope=blocking_item.fix_scope,
                                    sub_items=blocking_item.sub_items,
                                    evidence_row_ids=blocking_item.evidence_row_ids,
                                )
                                round_new_unresolved_items.append(tracked_item)
                                reviewer_new_unresolved_items.append(tracked_item)
                                next_unresolved_item_number += 1
                        elif has_blocking_summary:
                            tracked_item = _next_unresolved_item(
                                item_number=next_unresolved_item_number,
                                reviewer=reviewer_name,
                                source_round=round_number,
                                text=blocking_summary,
                                status="blocking",
                            )
                            round_new_unresolved_items.append(tracked_item)
                            reviewer_new_unresolved_items.append(tracked_item)
                            next_unresolved_item_number += 1
                        if parsed_review.followups.same_pr:
                            # Once a review is blocking, its same-PR findings
                            # belong in the mandatory fix round regardless of
                            # the policy for optional follow-ups on an approved
                            # review. The mode only controls publication of the
                            # latter.
                            for followup in parsed_review.followups.same_pr:
                                tracked_item = _next_unresolved_item(
                                    item_number=next_unresolved_item_number,
                                    reviewer=followup.reviewer,
                                    source_round=round_number,
                                    text=followup.text,
                                    status="same-pr",
                                    fix_scope=followup.fix_scope,
                                    sub_items=followup.sub_items,
                                )
                                round_new_unresolved_items.append(tracked_item)
                                reviewer_new_unresolved_items.append(tracked_item)
                                next_unresolved_item_number += 1
                        if reviewer not in early_published_pr_reviewers:
                            _post_pr_reviewer_comment(
                                reviewer_name, parsed_review, review_output=review_output,
                                model_used=review_model_used,
                                identity=(review_response if resumed_record is None else None),
                                acquisition_outcome=review_acquisition_outcome,
                                acquisition_returncode=review_acquisition_returncode,
                                new_items=tuple(reviewer_new_unresolved_items),
                            )
                    else:
                        if resumed_record is not None:
                            round_new_unresolved_items.extend(reviewer_new_unresolved_items)
                    continue

                approved_review_outputs.append((reviewer_name, review_output))
                accepted_review_carriers[reviewer_name] = parsed_review
                if (
                    (resumed_record is None or resumed_record.metadata.phase == "publication")
                    and carried_approval_record is None
                ):
                    if config.approved_followups != "ignore":
                        for followup in parsed_review.followups.future:
                            tracked_item = _next_unresolved_item(
                                item_number=next_unresolved_item_number,
                                reviewer=followup.reviewer,
                                source_round=round_number,
                                text=followup.text,
                                status="future",
                            )
                            round_new_unresolved_items.append(tracked_item)
                            reviewer_new_unresolved_items.append(tracked_item)
                            next_unresolved_item_number += 1
                    if reviewer not in early_published_pr_reviewers:
                        _post_pr_reviewer_comment(
                            reviewer_name, parsed_review, review_output=review_output,
                            model_used=review_model_used,
                            identity=(review_response if resumed_record is None else None),
                            acquisition_outcome=review_acquisition_outcome,
                            acquisition_returncode=review_acquisition_returncode,
                            new_items=tuple(reviewer_new_unresolved_items),
                        )
                else:
                    if resumed_record is not None:
                        round_new_unresolved_items.extend(reviewer_new_unresolved_items)

            if not pr_fatal_errors and not skip_reviewers_this_round:
                # Every same-round outcome is now published; a sequential
                # resume that replayed withheld reviews no longer needs them.
                pr_round_spool.discard()
            same_model_note = same_model_panel_note(round_review_models)
            if same_model_note is not None:
                log(config, f"Round {round_number}: {same_model_note}")
            if (
                (
                    pr_round_parallel
                    or selective_policy
                    or unavailable_reviewer_failures
                    or same_model_note is not None
                )
                and not skip_reviewers_this_round
                and not (current_resume is not None and current_resume.reconciled)
                and evidence_pass is None
            ):
                settled_reviewers = (
                    tuple(sorted(selected_reviewer_names))
                    if selective_policy
                    else tuple(agent_display_name(reviewer) for reviewer in configured_reviewers)
                )
                settled = ", ".join(settled_reviewers)
                post_pr_comment(
                    runner, config=config, pr_number=pr_number,
                    body=_attach_round_metadata(
                        f"PR review round {round_number} reconciliation: settled reviewers: {settled or 'none'}. "
                        f"Finalization {'stops' if pr_fatal_errors or unavailable_reviewer_failures else 'continues'} after reconciliation. "
                        f"Historical approvals remain exact-head-bound; scheduler-policy calls avoided "
                        f"cumulatively: {scheduler_calls_avoided}. Phase: "
                        f"{scheduler_decision.phase if scheduler_decision is not None else 'full-board'}; "
                        f"force-full: {scheduler_recorded_force_full} "
                        f"(source: {scheduler_recorded_force_full_source or 'none'})."
                        + (f" {same_model_note}." if same_model_note else ""),
                        bound_pr_metadata(
                            flow="pr", role="summary", agent="Orchestrator", round_number=round_number,
                            subject=current_pr_subject, prior_items=prior_unresolved_items,
                            dispositions=tuple(
                                disposition for values in prior_dispositions.values() for disposition in values
                            ), new_items=tuple(round_new_unresolved_items), phase="reconciliation",
                            scheduler_contract=(scheduler_contract.as_dict() if selective_policy else None),
                            reviewer_board_amendment_digest=(pr_amendment_digest if selective_policy else None),
                            scheduler_previous_sha=(scheduler_previous_sha if selective_policy else None),
                            scheduler_current_sha=(current_pr_subject if selective_policy else None),
                            scheduler_obligation_digest=(
                                hashlib.sha256(
                                    repr(_prior_item_ledger_signature(prior_unresolved_items)).encode("utf-8")
                                ).hexdigest()[:16]
                                if selective_policy else None
                            ),
                            scheduler_selected_reviewers=settled_reviewers if selective_policy else (),
                            scheduler_paused_reviewers=(
                                scheduler_decision.paused_reviewers
                                if selective_policy and scheduler_decision is not None else ()
                            ),
                            scheduler_reasons=(
                                (scheduler_decision.reason, classification.reason)
                                if selective_policy and scheduler_decision is not None else ()
                            ),
                            scheduler_final_sweep=(final_sweep if selective_policy else None),
                            scheduler_force_full=(scheduler_recorded_force_full if selective_policy else None),
                            scheduler_force_full_source=(
                                scheduler_recorded_force_full_source if selective_policy else None
                            ),
                            scheduler_calls_avoided=(scheduler_calls_avoided if selective_policy else None),
                            scheduler_phase=(scheduler_decision.phase if selective_policy and scheduler_decision is not None else None),
                            scheduler_primary_reviewer=(scheduler_contract.primary_reviewer if selective_policy else None),
                            scheduler_approved_reviewers=(
                                tuple(sorted({
                                    *unchanged_head_approvals,
                                    *(name for name, _output in approved_review_outputs),
                                })) if selective_policy else ()
                            ),
                            scheduler_active_owners=(scheduler_decision.active_owners if selective_policy and scheduler_decision is not None else ()),
                            scheduler_scope_digest=(hashlib.sha256(repr(classification.changed_paths).encode("utf-8")).hexdigest()[:16] if selective_policy else None),
                        ),
                    ),
                )

            if pr_fatal_errors:
                # Every healthy reviewer above was already applied (comment
                # posted, items numbered, unavailable failures recorded) in
                # configured order, so a rerun resumes them instead of
                # re-invoking (#594). Raise only now: quota resets take
                # priority, otherwise the first configured-order failure.
                for _reviewer_name, error in pr_fatal_errors:
                    if isinstance(error, QuotaResetExceededError):
                        raise error
                raise pr_fatal_errors[0][1]

            round_evidence_clearances: list[tuple[str, str]] = []
            round_cleared_sub_items: list[ClearedItemProgress] = []
            round_sub_item_notes: list[str] = []
            evidence_reconciliation_kwargs = {
                "evidence_response_head": evidence_response_head,
                "configured_reviewers": tuple(
                    agent_display_name(reviewer) for reviewer in configured_reviewers
                ),
                "evidence_clearances": round_evidence_clearances,
                "round_number": round_number,
                "cleared_items_progress": round_cleared_sub_items,
                "sub_item_degradations": round_sub_item_notes,
            }
            if use_compact_pr_context:
                unresolved_items, future_from_prior_items = _apply_unresolved_item_dispositions(
                    pr_ledger_view(prior_unresolved_items),
                    prior_dispositions,
                    retain_future=False,
                    reconciliation_mode=(
                        "owner-scoped"
                        if scheduler_capabilities.owner_scoped_reconciliation
                        else "aggregate"
                    ),
                    **evidence_reconciliation_kwargs,
                )
                pr_compact_prior_summaries = list(
                    bound_compact_prior_summaries(
                        [
                            *pr_compact_prior_summaries,
                            *_collect_prior_compact_summaries(
                                prior_unresolved_items,
                                unresolved_items,
                                prior_dispositions,
                            ),
                        ]
                    )
                )
            else:
                unresolved_items, _future_items = _apply_unresolved_item_dispositions(
                    pr_ledger_view(prior_unresolved_items),
                    prior_dispositions,
                    reconciliation_mode=(
                        "owner-scoped"
                        if scheduler_capabilities.owner_scoped_reconciliation
                        else "aggregate"
                    ),
                    **evidence_reconciliation_kwargs,
                )
                future_from_prior_items = []
            unresolved_items = list(
                pr_ledger_view([*unresolved_items, *round_new_unresolved_items])
            )
            pr_finding_history.observe_reconciled(unresolved_items, future_from_prior_items)
            # Confirmed sub-item outcome of this round (#958): published after
            # reconciliation and before any coder turn, approval, budget check
            # or exit, so a terminal approval round is never silent.
            _publish_sub_item_progress(
                runner,
                config=config,
                pr_number=pr_number,
                round_number=round_number,
                items=unresolved_items,
                cleared=round_cleared_sub_items,
                notes=round_sub_item_notes,
            )
            # Dispositions first, then newly emitted evidence requests, so a
            # same-response re-emission of a just-cleared request hits its
            # head-scoped clearance entry instead of recreating it (#1068).
            evidence_clearances.extend(
                clearance
                for clearance in round_evidence_clearances
                if clearance not in evidence_clearances
            )
            for requesting_reviewer, requests in round_evidence_requests:
                for request_text in requests:
                    unresolved_items, consumed_item_number = _upsert_evidence_obligation(
                        unresolved_items,
                        item_number=next_unresolved_item_number,
                        reviewer=requesting_reviewer,
                        text=request_text,
                        source_round=round_number,
                        current_head_sha=pr_metadata.head_sha,
                        clearances=evidence_clearances,
                    )
                    if consumed_item_number:
                        next_unresolved_item_number += 1
            if round_evidence_clearances:
                evidence_revalidation_required = True
                log(
                    config,
                    f"Round {round_number}: exact-head evidence cleared at "
                    f"{pr_metadata.head_sha} by its requesting reviewer(s): "
                    + ", ".join(identity for identity, _head in round_evidence_clearances),
                )
            # Human-requirement acknowledgement is a structured reviewer/coder
            # contract, not a reviewer-item ownership decision. Once every
            # required reviewer has emitted the explicit acknowledgement on
            # this round, clear the durable acknowledgement obligation here;
            # ordinary `resolved` dispositions alone still cannot do so.
            if human_requirements and any(
                item.item_id == HUMAN_REQUIREMENTS_ACK_ITEM_ID
                for item in unresolved_items
            ):
                required_reviewer_names = {
                    agent_display_name(reviewer) for reviewer in configured_reviewers
                }
                acknowledged_reviewer_names = {
                    reviewer_name
                    for reviewer_name, review_output in approved_review_outputs
                    if human_requirements_resolved(review_output)
                }
                if required_reviewer_names <= acknowledged_reviewer_names:
                    unresolved_items = _clear_human_requirements_ack_item(unresolved_items)
            future_followups = [
                _approved_followup_from_unresolved_item(item)
                for item in [*unresolved_items, *future_from_prior_items]
                if item.status == "future"
            ]
            migration_validation = None
            if not unavailable_reviewer_failures and any(
                _is_machine_obligation(item)
                and item.obligation_kind == "alembic-migration"
                for item in unresolved_items
            ):
                # A carried migration obligation must be re-probed by its own
                # authority before it can continue to the coder/finalization
                # partition. This is deliberately independent of reviewer
                # dispositions and permits a fresh successful validation to
                # clear the obligation on the current candidate head.
                sync_coder_pr_before_validation(config, runner, pr_number, pr_metadata)
                migration_validation = validate_pr_migration_topology(
                    runner,
                    config=config,
                    checkout=active_workdir(config),
                    pr_metadata=pr_metadata,
                )
                if migration_validation.ok:
                    unresolved_items = _clear_machine_obligations(
                        unresolved_items, kind="alembic-migration"
                    )
            item_partitions = _partition_unresolved_items(
                unresolved_items,
                current_head_sha=pr_metadata.head_sha,
            )
            reviewer_blockers = list(item_partitions["reviewer_blockers"])
            repair_required_machine_obligations = list(
                item_partitions["repair_required_machine_obligations"]
            )
            revalidation_candidates = list(item_partitions["revalidation_candidates"])
            finalization_blockers = list(item_partitions["finalization_blockers"])
            # Only human findings and machine obligations that explicitly need
            # repair go to the coder.  A current, fully reviewed CI candidate
            # proceeds to source-authoritative qualification instead.
            must_fix_items = list(item_partitions["coder_blockers"])
            if unavailable_reviewer_failures:
                # Decide at the point of detection (#1129): a required reviewer
                # that cannot be reached can never be satisfied by continuing.
                approved_reviewer_names = [
                    reviewer_name for reviewer_name, _review_output in approved_review_outputs
                ]
                unavailable_names = [
                    agent_display_name(reviewer) for reviewer in unavailable_reviewer_failures
                ]
                # A host/account outage affects every seat of that backend.
                # Model access and one seat's fallback exhaustion remain local.
                outage_backends = {
                    seat_backend(reviewer)
                    for reviewer, failure in unavailable_reviewer_failures.items()
                    if any(phrase in str(failure).lower() for phrase in (
                        "backend outage", "provider outage", "settings lock", "cli not found"
                    ))
                }
                if outage_backends:
                    unavailable_names = [
                        agent_display_name(reviewer) for reviewer in configured_reviewers
                        if reviewer in unavailable_reviewer_failures
                        or seat_backend(reviewer) in outage_backends
                    ]
                pr_surface = f"PR #{pr_number}"
                advisory_template: str | None = None
                advisory_round: int | None = None
                if selective_policy:
                    advisory_outcome, advisory_round, advisory_template = (
                        _unavailable_reviewer_amendment_advisory(
                            pr_number=pr_number,
                            contract=pr_effective_contract,
                            lineage=pr_contract_lineage,
                            removed=unavailable_names,
                            seat_local_failure=any(
                                seat_backend(reviewer) not in outage_backends
                                for reviewer in unavailable_reviewer_failures
                            ),
                            shared_outage_backends=frozenset(outage_backends),
                            seat_binding_config=config,
                            fetch_start_round=lambda: _pr_amendment_start_round(
                                get_pr_review_context(runner, config=config, pr_number=pr_number),
                                configured_reviewers,
                                scheduler_capabilities,
                                runner=runner, config=config,
                            ),
                        )
                    )
                    remedy_route = {
                        "validated": "amendment",
                        "rejected": "rejected",
                    }.get(advisory_outcome, "amendment-lookup-failed")
                elif _managed_binding_protection_mode(managed_ci_handoff) == "strict":
                    remedy_route = "restore-only"
                else:
                    remedy_route = "flags"
                remedy_lines = _unavailable_reviewer_remedy(
                    route=remedy_route,
                    surface=pr_surface,
                    round_number=advisory_round,
                    in_evidence_pass=evidence_pass is not None,
                )
                post_pr_comment(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    body=_format_incomplete_pr_review_comment(
                        pr_number=pr_number,
                        unavailable_reviewers=unavailable_reviewer_failures,
                        approved_reviewer_names=approved_reviewer_names,
                        detection_round=round_number,
                        blocking_reviewer_names=round_blocking_reviewer_names,
                        open_must_fix_count=len(must_fix_items),
                        remedy_lines=remedy_lines,
                    ),
                )
                categories = "; ".join(
                    f"{agent_display_name(reviewer)}: {failure.failure_category or 'unknown'}"
                    for reviewer, failure in unavailable_reviewer_failures.items()
                )
                healthy = ", ".join(approved_reviewer_names) or "(none)"
                blockers = ", ".join(round_blocking_reviewer_names) or "(none)"
                raise AgentLoopError(
                    f"PR #{pr_number} review incomplete: missing required input from "
                    f"{', '.join(unavailable_names)} ({categories}). Detected in round "
                    f"{round_number}. Healthy reviewers approved: {healthy}. Blocking reviewers "
                    f"this round: {blockers}. Open must-fix items (including carried): "
                    f"{len(must_fix_items)}. After detection in round {round_number}, no coder "
                    "follow-up, CI wait, qualification, or merge was started.\n"
                    + "\n".join(remedy_lines)
                    + (
                        "\n\nSigned amendment template (replace the rationale placeholder):\n\n"
                        + advisory_template
                        if advisory_template
                        else ""
                    )
                )
            if evidence_pass is not None:
                # Every evidence-response or refresh pass ends with exactly one
                # terminal record at this round and head (#1068).
                pass_head = str(pr_metadata.head_sha)
                if must_fix_items:
                    unresolved_items = _publish_evidence_release(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        round_number=round_number,
                        head_sha=pass_head,
                        items=unresolved_items,
                        reason="findings",
                        surfaced_requirement_ids=surfaced_reviewer_requirement_ids,
                        allowed_rounds=allowed_rounds,
                        watch_failure_extension_used=watch_failure_extension_used,
                        watch_head_extension_used=watch_head_extension_used,
                        clearances=evidence_clearances,
                    )
                    log(
                        config,
                        f"Round {round_number}: evidence {evidence_pass} pass at {pass_head} "
                        "raised findings; the freeze is released and the coder repairs them",
                    )
                elif not _pending_evidence_obligations(unresolved_items):
                    unresolved_items = _publish_evidence_release(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        round_number=round_number,
                        head_sha=pass_head,
                        items=unresolved_items,
                        reason=("evidence-cleared" if round_evidence_clearances else "refresh-clean"),
                        surfaced_requirement_ids=surfaced_reviewer_requirement_ids,
                        allowed_rounds=allowed_rounds,
                        watch_failure_extension_used=watch_failure_extension_used,
                        watch_head_extension_used=watch_head_extension_used,
                        clearances=evidence_clearances,
                    )
                    evidence_revalidation_required = True
                elif evidence_response_head is not None:
                    # Evidence kept at the frozen head: no CI wait or
                    # qualification; revalidate and re-persist the freeze.
                    recheck_pending_child_plan_supersession()
                    gate_outcome = _evidence_freeze_gate(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        round_number=round_number,
                        head_sha=pr_metadata.head_sha,
                        items=unresolved_items,
                        surfaced_requirement_ids=surfaced_reviewer_requirement_ids,
                        collect_requirement_ids=collect_live_requirement_ids,
                        allowed_rounds=allowed_rounds,
                        watch_failure_extension_used=watch_failure_extension_used,
                        watch_head_extension_used=watch_head_extension_used,
                        clearances=evidence_clearances,
                    )
                    assert gate_outcome.context is not None
                    if gate_outcome.action == "head_changed":
                        stage_evidence_head_change(gate_outcome.context)
                        continue
                    if gate_outcome.action == "refresh":
                        stage_same_head_evidence_pass(
                            gate_outcome.context, current_round=round_number
                        )
                        continue
                    raise AgentLoopError(
                        f"PR #{pr_number} evidence gate proceeded while evidence is pending; "
                        "no approval or merge was attempted."
                    )
                else:
                    # A pre-freeze refresh pass with the evidence still
                    # deferred also ends with one terminal record, so a clean
                    # stop for pending checks afterwards keeps the pass (its
                    # reviews, requests and refreshed signed baseline) instead
                    # of leaving it as an interrupted transcript.
                    unresolved_items = _publish_evidence_release(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        round_number=round_number,
                        head_sha=pass_head,
                        items=unresolved_items,
                        reason="refresh-clean",
                        surfaced_requirement_ids=surfaced_reviewer_requirement_ids,
                        allowed_rounds=allowed_rounds,
                        watch_failure_extension_used=watch_failure_extension_used,
                        watch_head_extension_used=watch_head_extension_used,
                        clearances=evidence_clearances,
                    )
                    evidence_revalidation_required = True

            if must_fix_items:
                try:
                    _raise_if_maintained_disputed_items(
                        must_fix_items,
                        prior_items=prior_unresolved_items,
                    )
                except HumanDecisionRequiredError as exc:
                    post_pr_comment(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        body=(
                            "## Human decision required\n\n"
                            f"{exc}\n\n"
                            "Post a PR comment that states the decision and required action, "
                            "and end it with the standalone signature `-- Human Reviewer`. "
                            "Then rerun this PR; no coder, CI, qualification, or merge was "
                            "started from this decision boundary.\n\n"
                            "-- coding-review-agent-loop"
                        ),
                    )
                    raise

            pr_checks: PullRequestChecks | None = None
            if not must_fix_items:
                if human_requirements:
                    missing_acknowledgements = [
                        reviewer_name
                        for reviewer_name, review_output in approved_review_outputs
                        if not human_requirements_resolved(review_output)
                    ]
                    if missing_acknowledgements:
                        hr_ids = _surfaced_reviewer_requirement_ids(
                            human_requirements,
                            requirement_scope="PR requirements",
                        )
                        still_missing = []
                        for reviewer_name, review_output in approved_review_outputs:
                            if human_requirements_resolved(review_output):
                                continue
                            log(
                                config,
                                f"Round {round_number}: {reviewer_name} approved without "
                                "HUMAN_REQUIREMENTS_RESOLVED; attempting repair",
                            )
                            repaired_text, repaired_validated, repair_attempts = _run_structured_repair(
                                review_output,
                                runner=runner,
                                config=config,
                                usage_context=usage_context,
                                validate=lambda candidate, reviewer_name=reviewer_name: _validate_review_response(
                                    candidate,
                                    reviewer=reviewer_name,
                                    unresolved_items=prior_unresolved_items,
                                    current_round_items=round_new_unresolved_items, architecture_status_mode="strict",
                                    approved_matrix_row_ids=approved_matrix_row_ids(approved_plan_context),
                                ),
                                repair_kwargs={
                                    "expected_kind": "pr_review",
                                    "reviewer_requirement_ids": hr_ids,
                                    "allowed_prior_item_ids": tuple(
                                        item.item_id for item in prior_unresolved_items
                                    ),
                                    "evidence_row_id_universe": approved_matrix_row_ids(approved_plan_context) or (),
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
                                config, f"Round {round_number}: {reviewer_name}", repair_attempts
                            )
                            if repaired_validated is not None:
                                repaired_parsed = repaired_validated
                                if repaired_parsed.summary.strip():
                                    reviewer_summaries[reviewer_name] = _reviewer_summary_context(
                                        reviewer_name,
                                        repaired_parsed.summary,
                                        round_number=round_number,
                                        head_sha=current_pr_subject,
                                    )
                                if (
                                    repaired_parsed.state == "approved"
                                    and human_requirements_resolved(repaired_text)
                                ):
                                    log(
                                        config,
                                        f"Round {round_number}: repair recovered "
                                        f"HUMAN_REQUIREMENTS_RESOLVED for {reviewer_name}",
                                    )
                                    continue
                                if repaired_parsed.state == "blocking":
                                    log(
                                        config,
                                        f"Round {round_number}: repair returned blocking for "
                                        f"{reviewer_name}; treating as reviewer blocking",
                                    )
                                    for item in repaired_parsed.blocking_items:
                                        new_item = _next_unresolved_item(
                                            item_number=next_unresolved_item_number,
                                            reviewer=item.reviewer,
                                            source_round=round_number,
                                            text=item.text,
                                            status="blocking",
                                            sub_items=item.sub_items,
                                            evidence_row_ids=item.evidence_row_ids,
                                        )
                                        round_new_unresolved_items.append(new_item)
                                        unresolved_items.append(new_item)
                                        next_unresolved_item_number += 1
                                    for item in repaired_parsed.followups.same_pr:
                                        new_item = _next_unresolved_item(
                                            item_number=next_unresolved_item_number,
                                            reviewer=item.reviewer,
                                            source_round=round_number,
                                            text=item.text,
                                            status="same-pr",
                                            sub_items=item.sub_items,
                                        )
                                        round_new_unresolved_items.append(new_item)
                                        unresolved_items.append(new_item)
                                        next_unresolved_item_number += 1
                                    all_approved = False
                                    must_fix_items = [
                                        item for item in unresolved_items
                                        if item.status in {"blocking", "same-pr"}
                                    ]
                                    continue
                            still_missing.append(reviewer_name)
                        if still_missing:
                            log(
                                config,
                                f"Round {round_number}: reviewer(s) {', '.join(still_missing)} "
                                "approved without acknowledging signed human requirements; "
                                "re-injecting as blocking item",
                            )
                            unresolved_items.append(
                                _next_unresolved_item(
                                    item_number=next_unresolved_item_number,
                                    reviewer="Orchestrator",
                                    source_round=round_number,
                                    text=(
                                        f"Reviewer(s) {', '.join(still_missing)} approved without "
                                        "acknowledging the signed human requirements. Coder must address the "
                                        "human requirements and ensure the reviewer explicitly resolves them "
                                        "before approval."
                                    ),
                                    status="blocking",
                                )
                            )
                            next_unresolved_item_number += 1
                            must_fix_items = [
                                item for item in unresolved_items if item.status in {"blocking", "same-pr"}
                            ]
                if not must_fix_items and selective_policy:
                    fresh_context, fresh_requirement_ids, fresh_plan_context, config = _fresh_pr_qualification_snapshot(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        issue_context=issue_context,
                        parent_issue_context=parent_issue_context,
                        approved_plan_context=approved_plan_context,
                        scheduler_contract=scheduler_contract if selective_policy else None,
                        allow_plan_handoff_change=True,
                        planning_child_binding=planning_child_binding,
                        managed_protection_mode=_managed_binding_protection_mode(managed_ci_handoff),
                        managed_retired_plan_hashes=_managed_binding_retired_plan_hashes(managed_ci_handoff),
                        plan_binding_reviewers=operator_reviewers,
                    )
                    if fresh_context.metadata.head_sha != pr_metadata.head_sha or fresh_context.architecture_identity_changed:
                        log(
                            config,
                            f"Round {round_number}: PR head changed during reviewer reconciliation; "
                            "restarting exact-head scheduling",
                        )
                        unresolved_items = _advance_machine_obligations_for_head(
                            unresolved_items,
                            current_head_sha=fresh_context.metadata.head_sha,
                        )
                        qualification_checkpoint = _machine_obligation_checkpoint(
                            unresolved_items,
                            current_head_sha=fresh_context.metadata.head_sha,
                            base_branch=fresh_context.metadata.base_branch or config.base,
                            allowed_rounds=allowed_rounds,
                            watch_failure_extension_used=watch_failure_extension_used,
                            watch_head_extension_used=watch_head_extension_used,
                            lifecycle="awaiting_current_head_review",
                        )
                        _persist_qualification_checkpoint(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            round_number=round_number + 1,
                            head_sha=fresh_context.metadata.head_sha,
                            unresolved_items=unresolved_items,
                            checkpoint=qualification_checkpoint,
                            message=(
                                f"PR #{pr_number} qualification checkpoint: head changed during "
                                "review reconciliation; current-head review is required."
                            ),
                        )
                        prefetched_pr_context = fresh_context
                        final_sweep_pending = False
                        continue
                    plan_identity_changed = fresh_plan_context != approved_plan_context
                    if plan_identity_changed:
                        approved_plan_context = fresh_plan_context
                        log(
                            config,
                            f"Round {round_number}: approved-plan/handoff identity changed; "
                            "invalidating prior approvals for a fresh final sweep",
                        )
                    if plan_identity_changed or set(fresh_requirement_ids) != {
                        requirement.requirement_id for requirement in human_requirements
                    }:
                        # Signed input that reaches a PR carrying exact-head evidence is
                        # answered at the same head inside this round (#1068).
                        if not plan_identity_changed and (
                            evidence_revalidation_required
                            or _pending_evidence_obligations(unresolved_items)
                        ):
                            stage_same_head_evidence_pass(fresh_context, current_round=round_number)
                            continue
                        log(
                            config,
                            f"Round {round_number}: signed human-requirement identity changed; "
                            "invalidating prior approvals for a fresh final sweep",
                        )
                        final_sweep_pending = True
                        prefetched_pr_context = fresh_context
                        continue
                    qualifying_names = set(unchanged_head_approvals)
                    qualifying_names.update(
                        reviewer_name for reviewer_name, _output in approved_review_outputs
                    )
                    missing_names = [
                        agent_display_name(reviewer)
                        for reviewer in configured_reviewers
                        if agent_display_name(reviewer) not in qualifying_names
                    ]
                    if missing_names:
                        final_sweep_pending = True
                        log(
                            config,
                            f"Round {round_number}: exact-head gate requires final sweep for "
                            f"{', '.join(missing_names)}; no coder dispatch",
                        )
                        if round_number == allowed_rounds:
                            raise AgentLoopError(
                                f"PR #{pr_number} exact-head final sweep is missing reviewer approval "
                                f"from {', '.join(missing_names)}. No coder follow-up or merge was attempted."
                            )
                        continue

                for obligation_kind in {
                    item.obligation_kind
                    for item in revalidation_candidates
                    if item.obligation_kind is not None
                }:
                    unresolved_items = _set_machine_obligation_lifecycle(
                        unresolved_items,
                        kind=obligation_kind,
                        lifecycle="qualification_ready",
                    )
                sync_coder_pr_before_validation(config, runner, pr_number, pr_metadata)
                if migration_validation is None:
                    migration_validation = validate_pr_migration_topology(
                        runner,
                        config=config,
                        checkout=active_workdir(config),
                        pr_metadata=pr_metadata,
                    )
                if not migration_validation.ok:
                    log(config, f"Round {round_number}: Alembic migration validation blocked approval")
                    had_migration_obligation = any(
                        _is_machine_obligation(item)
                        and item.obligation_kind == "alembic-migration"
                        for item in unresolved_items
                    )
                    unresolved_items = _upsert_machine_obligation(
                        unresolved_items,
                        item_number=next_unresolved_item_number,
                        kind="alembic-migration",
                        source_round=round_number,
                        text=(
                            "Alembic migration validation failed: "
                            + (migration_validation.message or "Migration validation failed.")
                        ),
                        failed_head_sha=pr_metadata.head_sha,
                    )
                    if not had_migration_obligation:
                        next_unresolved_item_number += 1
                else:
                    unresolved_items = _clear_machine_obligations(
                        unresolved_items, kind="alembic-migration"
                    )
                must_fix_items = list(
                    _partition_unresolved_items(
                        unresolved_items,
                        current_head_sha=pr_metadata.head_sha,
                    )["coder_blockers"]
                )

                # Re-evaluate mergeability before CI checks / merge (#606): reviews
                # can take a while, so the branch may have gone conflicted since
                # the round-start probe. A confirmed conflict here skips the checks
                # fetch entirely and blocks the merge branch below via must_fix_items.
                merge_gate_mergeability = get_pr_mergeability(runner, config=config, pr_number=pr_number)
                latest_mergeability = merge_gate_mergeability
                unresolved_items = _reconcile_merge_conflict_item(
                    unresolved_items,
                    mergeability=merge_gate_mergeability,
                    source_round=round_number,
                    current_head_sha=pr_metadata.head_sha,
                )
                must_fix_items = list(
                    _partition_unresolved_items(
                        unresolved_items,
                        current_head_sha=pr_metadata.head_sha,
                    )["coder_blockers"]
                )
                finalization_observations: dict[str, str] = {}
                merge_gate_conflict_pending = any(
                    item.item_id == MERGE_CONFLICT_ITEM_ID for item in unresolved_items
                )
                if merge_gate_conflict_pending:
                    log(
                        config,
                        f"Round {round_number}: PR #{pr_number} became conflicted with "
                        f"{merge_gate_mergeability.base_branch or config.base or 'the base branch'} "
                        "before merge; skipping CI checks and merge this round",
                    )
                    pr_checks = None
                else:
                    pr_checks = get_pr_checks(runner, config=config, metadata=pr_metadata)
                    if managed_ci_active(pr_metadata):
                        pr_checks = intermediate_managed_checks(pr_checks)
                if pr_checks is not None and not must_fix_items and is_wholly_infrastructure_blocked(pr_checks):
                    # Every remaining blocking/pending signal is external GitHub
                    # Actions infrastructure (a queued check that never started a
                    # job, or one cancelled before execution because a hosted
                    # runner was unavailable): stop cleanly and resumably instead
                    # of a synthetic blocking item, a coder round, or a merge.
                    stall = CiInfrastructureStall(checks=pr_checks.infrastructure_stalls)
                    _publish_approved_followups(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        head_sha=pr_metadata.head_sha,
                        pr_comments=pr_comments,
                        followups=future_followups,
                        source_context=followup_source_context,
                        usage_context=usage_context,
                    )
                    post_pr_comment(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        body=_format_ci_infrastructure_comment(pr_number, stall),
                    )
                    post_pr_comment(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        body=_append_evidence_barrier_note(
                            _ci_infrastructure_stop_message(pr_number, stall, []),
                            unresolved_items,
                        ),
                    )
                    log(
                        config,
                        f"Round {round_number}: reviewers approved PR #{pr_number}; GitHub checks "
                        "are wholly blocked by external CI infrastructure; stopping without a "
                        "coder follow-up round or merge",
                    )
                    print(
                        f"PR #{pr_number} was approved by {format_agent_list(configured_reviewers)}, "
                        "but external GitHub Actions infrastructure is blocking CI "
                        f"({'; '.join(_ci_infrastructure_details(stall))}). No code change is "
                        "required and no merge was attempted; rerun the same command once GitHub "
                        "Actions runners recover."
                    )
                    return 0
                if not must_fix_items and ordinary_recovery_selected and not managed_ci_active(pr_metadata):
                    if ordinary_recovery is None:
                        raise AgentLoopError(
                            f"PR #{pr_number} was released to ordinary CI, but recovery provenance "
                            "could not be correlated; no merge attempted."
                        )
                    # A signed child-plan supersession stops the run before any freeze
                    # can ask a human for evidence under a plan authorized for replacement.
                    recheck_pending_child_plan_supersession()
                    # Exact-head evidence gate (#1068): the last reads before any
                    # finalization side effect; publishes the single freeze when evidence
                    # is pending, and otherwise revalidates head and signed input.
                    if evidence_revalidation_required or _pending_evidence_obligations(unresolved_items):
                        gate_outcome = _evidence_freeze_gate(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            round_number=round_number,
                            head_sha=pr_metadata.head_sha,
                            items=unresolved_items,
                            surfaced_requirement_ids=evidence_surfaced_baseline,
                            collect_requirement_ids=collect_live_requirement_ids,
                            allowed_rounds=allowed_rounds,
                            watch_failure_extension_used=watch_failure_extension_used,
                            watch_head_extension_used=watch_head_extension_used,
                            clearances=evidence_clearances,
                        )
                        assert gate_outcome.context is not None
                        if gate_outcome.action == "head_changed":
                            stage_evidence_head_change(gate_outcome.context)
                            continue
                        if gate_outcome.action == "refresh":
                            stage_same_head_evidence_pass(gate_outcome.context, current_round=round_number)
                            continue
                    merged = _finalize_ordinary_recovery_checked(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        round_number=round_number,
                        items=unresolved_items,
                        current_head_sha=pr_metadata.head_sha,
                        capability=ordinary_recovery,
                    )
                    if merged:
                        print(f"PR #{pr_number} merged after deliberate ordinary recovery.")
                        unresolved_items = _clear_machine_obligations(
                            unresolved_items, kind="github-pr-checks"
                        )
                        _record_staged_parent_completion_after_merge(
                            runner, config=config, issue_context=issue_context,
                            pr_number=pr_number,
                        )
                    return 0
                if (
                    not must_fix_items
                    and (config.auto_merge or config.watch_pending_ci)
                    and not managed_ci_active(pr_metadata)
                    # A failure already visible in the approval snapshot is
                    # existing actionable reviewer feedback; preserve the
                    # normal coder-routing path. The watcher handles failures
                    # discovered by a fresh poll after an otherwise clean
                    # approval snapshot.
                    and (pr_checks is None or pr_checks.state != "failing")
                ):
                    unresolved_items = _set_machine_obligation_lifecycle(
                        unresolved_items,
                        kind="github-pr-checks",
                        lifecycle="qualifying",
                    )
                    qualification_checkpoint = _machine_obligation_checkpoint(
                        unresolved_items,
                        current_head_sha=pr_metadata.head_sha,
                        base_branch=pr_metadata.base_branch or config.base,
                        allowed_rounds=allowed_rounds,
                        watch_failure_extension_used=watch_failure_extension_used,
                        watch_head_extension_used=watch_head_extension_used,
                        approval_digest=_qualification_digest(
                            tuple(sorted({
                                *unchanged_head_approvals,
                                *(name for name, _output in approved_review_outputs),
                            }))
                        ),
                        plan_digest=(
                            approved_plan_context.plan_hash
                            if approved_plan_context is not None else None
                        ),
                        requirements_digest=_qualification_digest(
                            tuple(requirement.requirement_id for requirement in human_requirements)
                        ),
                        acquisition_digest=_qualification_digest(reviewer_acquisition_contract),
                        scheduler_digest=_qualification_digest(
                            _prior_item_ledger_signature(unresolved_items)
                        ),
                        qualification_attempt_id=(
                            f"github-checks:{pr_metadata.head_sha}"
                            if pr_metadata.head_sha else None
                        ),
                    )
                    if qualification_checkpoint is not None:
                        post_pr_comment(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            body=_attach_round_metadata(
                                f"PR #{pr_number} qualification checkpoint: full-board checks are being watched.",
                                bound_pr_metadata(
                                    flow="pr",
                                    role="summary",
                                    agent="Orchestrator",
                                    round_number=round_number,
                                    subject=pr_metadata.head_sha,
                                    prior_items=tuple(unresolved_items),
                                    phase="qualification-checkpoint",
                                    qualification_checkpoint=qualification_checkpoint,
                                    **_architecture_metadata_fields(config),
                                ),
                            ),
                        )
                    if watch_deadline is None:
                        watch_deadline = time.monotonic() + config.ci_timeout_seconds
                        watch_attempts_remaining = max(
                            1,
                            config.ci_timeout_seconds // config.ci_poll_interval_seconds,
                        )
                    assert watch_attempts_remaining is not None
                    if watch_attempts_remaining <= 0:
                        return _stop_after_ci_watch_timeout(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            round_number=round_number,
                            head_sha=pr_metadata.head_sha,
                            pr_comments=pr_comments,
                            followups=future_followups,
                            source_context=followup_source_context,
                            usage_context=usage_context,
                            details=[
                                "The shared CI watch budget was exhausted by earlier watcher rounds; "
                                "no fresh CI poll was performed."
                            ],
                            reason="budget_exhausted",
                        )
                    watching_message = (
                        f"Reviewers approved PR #{pr_number}; watching GitHub checks "
                        "in the foreground. "
                        "No coder or reviewer agents will run while checks remain pending."
                    )
                    log(config, f"Round {round_number}: {watching_message}")
                    post_pr_comment(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        body=watching_message + "\n\n-- coding-review-agent-loop",
                    )
                    watch_outcome = watch_pr_checks(
                        runner,
                        config,
                        pr_number,
                        metadata=pr_metadata,
                        deadline=watch_deadline,
                        attempts=watch_attempts_remaining,
                    )
                    watch_attempts_remaining -= watch_outcome.attempts_used
                    if watch_outcome.status == "dry_run":
                        print(f"PR #{pr_number} approval found; dry-run preview did not perform live CI watching.")
                        return 0
                    if watch_outcome.status == "passed":
                        _publish_approved_followups(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            head_sha=pr_metadata.head_sha,
                            pr_comments=pr_comments,
                            followups=future_followups,
                            source_context=followup_source_context,
                            usage_context=usage_context,
                        )
                        run_optional_tests(runner, config)
                        if selective_policy:
                            fresh_context, fresh_requirement_ids, fresh_plan_context, config = _fresh_pr_qualification_snapshot(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                issue_context=issue_context,
                                parent_issue_context=parent_issue_context,
                                approved_plan_context=approved_plan_context,
                                scheduler_contract=scheduler_contract if selective_policy else None,
                                allow_plan_handoff_change=True,
                                planning_child_binding=planning_child_binding,
                                managed_protection_mode=_managed_binding_protection_mode(managed_ci_handoff),
                                managed_retired_plan_hashes=_managed_binding_retired_plan_hashes(managed_ci_handoff),
                                plan_binding_reviewers=operator_reviewers,
                            )
                            if fresh_context.metadata.head_sha != pr_metadata.head_sha or fresh_context.architecture_identity_changed:
                                unresolved_items = _advance_machine_obligations_for_head(
                                    unresolved_items,
                                    current_head_sha=fresh_context.metadata.head_sha,
                                )
                                qualification_checkpoint = _machine_obligation_checkpoint(
                                    unresolved_items,
                                    current_head_sha=fresh_context.metadata.head_sha,
                                    base_branch=fresh_context.metadata.base_branch or config.base,
                                    allowed_rounds=allowed_rounds,
                                    watch_failure_extension_used=watch_failure_extension_used,
                                    watch_head_extension_used=watch_head_extension_used,
                                    lifecycle="awaiting_current_head_review",
                                )
                                _persist_qualification_checkpoint(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    round_number=round_number + 1,
                                    head_sha=fresh_context.metadata.head_sha,
                                    unresolved_items=unresolved_items,
                                    checkpoint=qualification_checkpoint,
                                    message=(
                                        f"PR #{pr_number} qualification checkpoint: head changed after "
                                        "ordinary CI success; current-head review is required."
                                    ),
                                )
                                prefetched_pr_context = fresh_context
                                final_sweep_pending = False
                                continue
                            plan_identity_changed = fresh_plan_context != approved_plan_context
                            if plan_identity_changed:
                                approved_plan_context = fresh_plan_context
                            if plan_identity_changed or set(fresh_requirement_ids) != {
                                requirement.requirement_id for requirement in human_requirements
                            }:
                                # Signed input that reaches a PR carrying exact-head evidence is
                                # answered at the same head inside this round (#1068).
                                if not plan_identity_changed and (
                                    evidence_revalidation_required
                                    or _pending_evidence_obligations(unresolved_items)
                                ):
                                    stage_same_head_evidence_pass(fresh_context, current_round=round_number)
                                    continue
                                final_sweep_pending = True
                                prefetched_pr_context = fresh_context
                                continue
                        if watch_outcome.head_sha != pr_metadata.head_sha:
                            raise AgentLoopError(
                                f"PR #{pr_number} full-board CI passed for a stale head; no merge attempted."
                            )
                        # The watcher status is only an aggregate. Apply the same
                        # source-specific proof predicate used by the review-only
                        # snapshot path before clearing or merging.
                        if not _ordinary_checks_snapshot_is_authoritative(
                            watch_outcome.pr_checks,
                            watch_outcome.mergeability,
                            head_sha=watch_outcome.head_sha,
                        ):
                            details = (
                                _pr_check_details(watch_outcome.pr_checks)
                                if watch_outcome.pr_checks is not None
                                else ["No authoritative current-head check snapshot was available."]
                            )
                            return _stop_after_ci_watch_timeout(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                round_number=round_number,
                                head_sha=pr_metadata.head_sha,
                                pr_comments=pr_comments,
                                followups=future_followups,
                                source_context=followup_source_context,
                                usage_context=usage_context,
                                details=details,
                                reason="non_authoritative",
                            )
                        unresolved_items = _clear_machine_obligations(
                            unresolved_items, kind="github-pr-checks"
                        )
                        # A signed child-plan supersession stops the run before any freeze
                        # can ask a human for evidence under a plan authorized for replacement.
                        recheck_pending_child_plan_supersession()
                        # Exact-head evidence gate (#1068): the last reads before any
                        # finalization side effect; publishes the single freeze when evidence
                        # is pending, and otherwise revalidates head and signed input.
                        if evidence_revalidation_required or _pending_evidence_obligations(unresolved_items):
                            gate_outcome = _evidence_freeze_gate(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                round_number=round_number,
                                head_sha=pr_metadata.head_sha,
                                items=unresolved_items,
                                surfaced_requirement_ids=evidence_surfaced_baseline,
                                collect_requirement_ids=collect_live_requirement_ids,
                                allowed_rounds=allowed_rounds,
                                watch_failure_extension_used=watch_failure_extension_used,
                                watch_head_extension_used=watch_head_extension_used,
                                clearances=evidence_clearances,
                            )
                            assert gate_outcome.context is not None
                            if gate_outcome.action == "head_changed":
                                stage_evidence_head_change(gate_outcome.context)
                                continue
                            if gate_outcome.action == "refresh":
                                stage_same_head_evidence_pass(gate_outcome.context, current_round=round_number)
                                continue
                        _ensure_finalization_ready(
                            pr_number=pr_number,
                            round_number=round_number,
                            items=unresolved_items,
                            current_head_sha=pr_metadata.head_sha,
                            sub_item_stall_rounds=config.sub_item_stall_rounds,
                            config=config,
                        )
                        if config.auto_merge:
                            if not watch_outcome.head_sha:
                                raise AgentLoopError(
                                    f"PR #{pr_number} full-board CI passed without a current-head "
                                    "proof; no merge attempted."
                                )
                            _merge_with_exact_head_proof(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                proof=ExactHeadCiProof(
                                    head_sha=watch_outcome.head_sha,
                                    source="full-board",
                                ),
                            )
                            print(
                                f"PR #{pr_number} merged after CI watch completed."
                                + announce_reduced_board_completion()
                            )
                            _record_staged_parent_completion_after_merge(
                                runner, config=config, issue_context=issue_context,
                                pr_number=pr_number,
                            )
                        else:
                            print(
                                f"PR #{pr_number} is merge-ready after CI watch completed."
                                + announce_reduced_board_completion()
                            )
                        return 0
                    if watch_outcome.status == "not_started":
                        command = render_managed_ci_resume_command(
                            config, pr_number=pr_number, managed_ci=False,
                        )
                        log(
                            config,
                            f"Round {round_number}: PR #{pr_number} has no materialized CI board within "
                            "the startup window; no merge attempted",
                        )
                        print(
                            f"PR #{pr_number} has no materialized current-head CI board. No merge was "
                            f"attempted; resume with `{command}`."
                        )
                        if config.auto_merge:
                            raise AgentLoopError(
                                f"PR #{pr_number} full-board CI did not start within the bounded "
                                "startup window; no merge attempted."
                            )
                        return 0
                    if watch_outcome.status == "infrastructure_stall":
                        _publish_approved_followups(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            head_sha=pr_metadata.head_sha,
                            pr_comments=pr_comments,
                            followups=future_followups,
                            source_context=followup_source_context,
                            usage_context=usage_context,
                        )
                        assert watch_outcome.stall is not None
                        post_pr_comment(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            body=_format_ci_infrastructure_comment(
                                pr_number, watch_outcome.stall
                            ),
                        )
                        post_pr_comment(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            body=_append_evidence_barrier_note(
                                _ci_infrastructure_stop_message(
                                    pr_number, watch_outcome.stall, []
                                ),
                                unresolved_items,
                            ),
                        )
                        print(f"PR #{pr_number} CI watch stopped: external CI infrastructure is stalled.")
                        return 0
                    if watch_outcome.status == "timeout":
                        details = (
                            _pr_check_details(watch_outcome.pr_checks)
                            if watch_outcome.pr_checks
                            else ["No reliable check snapshot was available."]
                        )
                        return _stop_after_ci_watch_timeout(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            round_number=round_number,
                            head_sha=pr_metadata.head_sha,
                            pr_comments=pr_comments,
                            followups=future_followups,
                            source_context=followup_source_context,
                            usage_context=usage_context,
                            details=details,
                            reason="timeout",
                        )
                    if watch_outcome.status == "protection_unreadable":
                        details = (
                            _pr_check_details(watch_outcome.pr_checks)
                            if watch_outcome.pr_checks
                            else ["No reliable check snapshot was available."]
                        )
                        merge_state = (
                            watch_outcome.mergeability.merge_state_raw
                            if watch_outcome.mergeability is not None
                            else None
                        )
                        details.append(
                            f"GitHub merge state for the current head: {merge_state or 'unavailable'} "
                            "(CLEAN is required when branch protection is unreadable)."
                        )
                        return _stop_after_ci_watch_timeout(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            round_number=round_number,
                            head_sha=pr_metadata.head_sha,
                            pr_comments=pr_comments,
                            followups=future_followups,
                            source_context=followup_source_context,
                            usage_context=usage_context,
                            details=details,
                            reason="protection_unreadable",
                        )
                    if watch_outcome.status == "head_changed":
                        log(config, f"PR #{pr_number} head changed while watching; re-review is required")
                        # Consume a fresh review round without dispatching the
                        # coder: prior-head approvals and resume state are stale.
                        if round_number == allowed_rounds and not watch_head_extension_used:
                            allowed_rounds += 1
                            watch_head_extension_used = True
                        resumed_round = None
                        current_resume = None
                        prefetched_pr_context = get_pr_review_context(
                            runner, config=config, pr_number=pr_number
                        )
                        unresolved_items = _advance_machine_obligations_for_head(
                            unresolved_items,
                            current_head_sha=prefetched_pr_context.metadata.head_sha,
                        )
                        qualification_checkpoint = _machine_obligation_checkpoint(
                            unresolved_items,
                            current_head_sha=prefetched_pr_context.metadata.head_sha,
                            base_branch=prefetched_pr_context.metadata.base_branch or config.base,
                            allowed_rounds=allowed_rounds,
                            watch_failure_extension_used=watch_failure_extension_used,
                            watch_head_extension_used=watch_head_extension_used,
                            lifecycle="awaiting_current_head_review",
                        )
                        _persist_qualification_checkpoint(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            round_number=round_number + 1,
                            head_sha=prefetched_pr_context.metadata.head_sha,
                            unresolved_items=unresolved_items,
                            checkpoint=qualification_checkpoint,
                            message=(
                                f"PR #{pr_number} qualification checkpoint: head changed; "
                                "current-head review is required before revalidation."
                            ),
                        )
                        continue
                    elif watch_outcome.status == "merge_conflict":
                        unresolved_items = _reconcile_merge_conflict_item(
                            unresolved_items,
                            mergeability=watch_outcome.mergeability,
                            source_round=round_number,
                            current_head_sha=pr_metadata.head_sha,
                        )
                    elif watch_outcome.status == "failed":
                        details = (
                            _pr_check_details(watch_outcome.pr_checks)
                            if watch_outcome.pr_checks
                            else ["GitHub checks failed."]
                        )
                        log(config, f"Round {round_number}: CI watch failed; resuming coder")
                        post_pr_comment(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            body=_format_pr_checks_comment(pr_number, "failing", details),
                        )
                        had_ordinary_obligation = any(
                            _is_machine_obligation(item)
                            and item.obligation_kind == "github-pr-checks"
                            for item in unresolved_items
                        )
                        unresolved_items = _upsert_machine_obligation(
                            unresolved_items,
                            item_number=next_unresolved_item_number,
                            kind="github-pr-checks",
                            source_round=round_number,
                            text=_pr_check_blocking_review(
                                pr_number, "failing", details
                            ),
                            failed_head_sha=watch_outcome.head_sha or pr_metadata.head_sha,
                        )
                        if not had_ordinary_obligation:
                            next_unresolved_item_number += 1
                        if round_number == allowed_rounds and not watch_failure_extension_used:
                            allowed_rounds += 1
                            watch_failure_extension_used = True
                    must_fix_items = list(
                        _partition_unresolved_items(
                            unresolved_items,
                            current_head_sha=pr_metadata.head_sha,
                        )["coder_blockers"]
                    )
                if not must_fix_items:
                    if pr_checks.state in {"pending", "unavailable"}:
                        details = _pr_check_details(pr_checks)
                        if not config.effective_managed_ci and not managed_ci_active(pr_metadata):
                            # Pending/unavailable checks are an external wait, not
                            # actionable coder feedback: stop cleanly instead of
                            # erroring or spending another coder/reviewer round.
                            _publish_approved_followups(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                head_sha=pr_metadata.head_sha,
                                pr_comments=pr_comments,
                                followups=future_followups,
                                source_context=followup_source_context,
                                usage_context=usage_context,
                            )
                            post_pr_comment(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                body=_format_pr_checks_comment(pr_number, pr_checks.state, details),
                            )
                            post_pr_comment(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                body=_append_evidence_barrier_note(
                                    _pending_ci_stop_message(pr_number, pr_checks.state, details),
                                    unresolved_items,
                                ),
                            )
                            log(
                                config,
                                f"Round {round_number}: reviewers approved PR #{pr_number}; "
                                f"GitHub checks are {pr_checks.state}; stopping without a "
                                "coder follow-up round",
                            )
                            print(
                                f"PR #{pr_number} was approved by "
                                f"{format_agent_list(configured_reviewers)}, but "
                                f"{_pending_ci_status_summary(pr_checks.state)}. "
                                f"{_pending_ci_stop_guidance(pr_checks.state)}"
                                + _evidence_barrier_note(unresolved_items)
                            )
                            return 0
                        # Managed qualification and --auto-merge: post the
                        # informational comment, then fall through to wait for
                        # the final gate before merging or publishing a manual
                        # result.
                        post_pr_comment(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            body=_format_pr_checks_comment(pr_number, pr_checks.state, details),
                        )
                    elif pr_checks.state == "failing":
                        details = _pr_check_details(pr_checks)
                        post_pr_comment(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            body=_format_pr_checks_comment(pr_number, pr_checks.state, details),
                        )
                        log(
                            config,
                            f"Round {round_number}: GitHub PR checks blocked approval ({pr_checks.state})",
                        )
                        had_ordinary_obligation = any(
                            _is_machine_obligation(item)
                            and item.obligation_kind == "github-pr-checks"
                            for item in unresolved_items
                        )
                        unresolved_items = _upsert_machine_obligation(
                            unresolved_items,
                            item_number=next_unresolved_item_number,
                            kind="github-pr-checks",
                            source_round=round_number,
                            text=_pr_check_blocking_review(pr_number, pr_checks.state, details),
                            failed_head_sha=pr_metadata.head_sha,
                        )
                        if not had_ordinary_obligation:
                            next_unresolved_item_number += 1
                    elif (
                        not managed_ci_active(pr_metadata)
                        and not ordinary_recovery_selected
                        and any(
                            _is_machine_obligation(item)
                            and item.obligation_kind == "github-pr-checks"
                            for item in unresolved_items
                        )
                    ):
                        snapshot_mergeability = _mergeability_for_unreadable_protection(
                            runner, config=config, pr_number=pr_number, checks=pr_checks,
                        )
                        if _ordinary_checks_snapshot_is_authoritative(
                            pr_checks,
                            snapshot_mergeability,
                            head_sha=pr_metadata.head_sha,
                        ):
                            # In review-only mode the foreground watcher is
                            # intentionally disabled. A fresh, correlated,
                            # full-board passing snapshot is still the ordinary CI
                            # authority named by the lifecycle contract and must
                            # clear the carried obligation here.
                            unresolved_items = _clear_machine_obligations(
                                unresolved_items, kind="github-pr-checks"
                            )
                        elif pr_checks.state == "passing":
                            reason = _single_line_diagnostic(
                                _ordinary_snapshot_nonauthority_reason(
                                    pr_checks,
                                    snapshot_mergeability,
                                    head_sha=pr_metadata.head_sha,
                                )
                            )
                            for item in unresolved_items:
                                if (
                                    _is_machine_obligation(item)
                                    and item.obligation_kind == "github-pr-checks"
                                ):
                                    finalization_observations[item.item_id] = (
                                        f"GitHub checks read passing at {pr_metadata.head_sha} "
                                        f"in this run, but the snapshot was not authoritative "
                                        f"({reason}), so this obligation was not cleared"
                                    )
                    must_fix_items = list(
                        _partition_unresolved_items(
                            unresolved_items,
                            current_head_sha=pr_metadata.head_sha,
                        )["coder_blockers"]
                    )
                if not must_fix_items:
                    _publish_approved_followups(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        head_sha=pr_metadata.head_sha,
                        pr_comments=pr_comments,
                        followups=future_followups,
                        source_context=followup_source_context,
                        usage_context=usage_context,
                    )
                    run_optional_tests(runner, config)
                    if selective_policy:
                        fresh_context, fresh_requirement_ids, fresh_plan_context, config = _fresh_pr_qualification_snapshot(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            issue_context=issue_context,
                            parent_issue_context=parent_issue_context,
                            approved_plan_context=approved_plan_context,
                            scheduler_contract=scheduler_contract if selective_policy else None,
                            allow_plan_handoff_change=True,
                            planning_child_binding=planning_child_binding,
                            managed_protection_mode=_managed_binding_protection_mode(managed_ci_handoff),
                            managed_retired_plan_hashes=_managed_binding_retired_plan_hashes(managed_ci_handoff),
                            plan_binding_reviewers=operator_reviewers,
                        )
                        if fresh_context.metadata.head_sha != pr_metadata.head_sha or fresh_context.architecture_identity_changed:
                            prefetched_pr_context = fresh_context
                            final_sweep_pending = False
                            continue
                        plan_identity_changed = fresh_plan_context != approved_plan_context
                        if plan_identity_changed:
                            approved_plan_context = fresh_plan_context
                        if plan_identity_changed or set(fresh_requirement_ids) != {
                            requirement.requirement_id for requirement in human_requirements
                        }:
                            # Signed input that reaches a PR carrying exact-head evidence is
                            # answered at the same head inside this round (#1068).
                            if not plan_identity_changed and (
                                evidence_revalidation_required
                                or _pending_evidence_obligations(unresolved_items)
                            ):
                                stage_same_head_evidence_pass(fresh_context, current_round=round_number)
                                continue
                            final_sweep_pending = True
                            prefetched_pr_context = fresh_context
                            continue
                    if config.auto_merge or managed_ci_active(pr_metadata):
                        fresh_context, fresh_requirement_ids, fresh_plan_context, config = _fresh_pr_qualification_snapshot(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            issue_context=issue_context,
                            parent_issue_context=parent_issue_context,
                            approved_plan_context=approved_plan_context,
                            scheduler_contract=scheduler_contract if selective_policy else None,
                            allow_plan_handoff_change=True,
                            planning_child_binding=planning_child_binding,
                            managed_protection_mode=_managed_binding_protection_mode(managed_ci_handoff),
                            managed_retired_plan_hashes=_managed_binding_retired_plan_hashes(managed_ci_handoff),
                            plan_binding_reviewers=operator_reviewers,
                        )
                        if fresh_context.metadata.head_sha != pr_metadata.head_sha or fresh_context.architecture_identity_changed:
                            log(
                                config,
                                f"Round {round_number}: PR head changed before managed qualification; "
                                "restarting review scheduling",
                            )
                            unresolved_items = _advance_machine_obligations_for_head(
                                unresolved_items,
                                current_head_sha=fresh_context.metadata.head_sha,
                            )
                            qualification_checkpoint = _machine_obligation_checkpoint(
                                unresolved_items,
                                current_head_sha=fresh_context.metadata.head_sha,
                                base_branch=fresh_context.metadata.base_branch or config.base,
                                allowed_rounds=allowed_rounds,
                                watch_failure_extension_used=watch_failure_extension_used,
                                watch_head_extension_used=watch_head_extension_used,
                                lifecycle="awaiting_current_head_review",
                            )
                            _persist_qualification_checkpoint(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                round_number=round_number + 1,
                                head_sha=fresh_context.metadata.head_sha,
                                unresolved_items=unresolved_items,
                                checkpoint=qualification_checkpoint,
                                message=(
                                    f"PR #{pr_number} qualification checkpoint: head changed before "
                                    "managed qualification; current-head review is required."
                                ),
                            )
                            prefetched_pr_context = fresh_context
                            final_sweep_pending = False
                            continue
                        plan_identity_changed = fresh_plan_context != approved_plan_context
                        if plan_identity_changed:
                            approved_plan_context = fresh_plan_context
                        if plan_identity_changed or set(fresh_requirement_ids) != {
                            requirement.requirement_id for requirement in human_requirements
                        }:
                            # Signed input that reaches a PR carrying exact-head evidence is
                            # answered at the same head inside this round (#1068).
                            if not plan_identity_changed and (
                                evidence_revalidation_required
                                or _pending_evidence_obligations(unresolved_items)
                            ):
                                stage_same_head_evidence_pass(fresh_context, current_round=round_number)
                                continue
                            log(
                                config,
                                f"Round {round_number}: qualification contract changed before managed CI; "
                                "restarting final sweep",
                            )
                            prefetched_pr_context = fresh_context
                            final_sweep_pending = True
                            continue
                        if managed_ci_active(pr_metadata):
                            assert pr_metadata.head_sha is not None
                            assert pr_metadata.head_branch is not None
                            unresolved_items = _set_machine_obligation_lifecycle(
                                unresolved_items,
                                kind="managed-exact-head-ci",
                                lifecycle="qualifying",
                            )
                            qualification_checkpoint = _machine_obligation_checkpoint(
                                unresolved_items,
                                current_head_sha=pr_metadata.head_sha,
                                base_branch=pr_metadata.base_branch or config.base,
                                allowed_rounds=allowed_rounds,
                                watch_failure_extension_used=watch_failure_extension_used,
                                watch_head_extension_used=watch_head_extension_used,
                                approval_digest=_qualification_digest(
                                    tuple(sorted({
                                        *unchanged_head_approvals,
                                        *(name for name, _output in approved_review_outputs),
                                    }))
                                ),
                                plan_digest=(
                                    approved_plan_context.plan_hash
                                    if approved_plan_context is not None else None
                                ),
                                requirements_digest=_qualification_digest(
                                    tuple(requirement.requirement_id for requirement in human_requirements)
                                ),
                                acquisition_digest=_qualification_digest(reviewer_acquisition_contract),
                                scheduler_digest=_qualification_digest(
                                    _prior_item_ledger_signature(unresolved_items)
                                ),
                            )
                            if qualification_checkpoint is not None:
                                post_pr_comment(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    body=_attach_round_metadata(
                                        (
                                            f"PR #{pr_number} qualification checkpoint: "
                                            "authoritative exact-head validation is in progress."
                                        ),
                                        bound_pr_metadata(
                                            flow="pr",
                                            role="summary",
                                            agent="Orchestrator",
                                            round_number=round_number,
                                            subject=pr_metadata.head_sha,
                                            prior_items=tuple(unresolved_items),
                                            phase="qualification-checkpoint",
                                            qualification_checkpoint=qualification_checkpoint,
                                            **_architecture_metadata_fields(config),
                                        ),
                                    ),
                                )
                            release_stale_managed_ci_attachment(
                                managed_ci,
                                head_sha=pr_metadata.head_sha,
                                config=config,
                                pr_number=pr_number,
                            )
                            attached_attempt = (
                                qualification_checkpoint is not None
                                and qualification_checkpoint.valid
                                and qualification_checkpoint.lifecycle == "qualifying"
                                and qualification_checkpoint.candidate_head_sha == pr_metadata.head_sha
                                and managed_ci_attachment_matches_head(
                                    managed_ci, pr_metadata.head_sha
                                )
                            )
                            if attached_attempt:
                                log(
                                    config,
                                    f"PR #{pr_number}: resuming attached qualification attempt "
                                    f"{managed_ci.attached_run_id}/{managed_ci.run_attempt}",
                                )
                            else:
                                dispatch_final_qualification(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    expected_head_sha=pr_metadata.head_sha,
                                    head_ref=pr_metadata.head_branch,
                                    contract=managed_ci,
                                )
                            if (
                                qualification_checkpoint is not None
                                and qualification_checkpoint.valid
                                and managed_ci.attached_run_id is not None
                                and managed_ci.run_attempt is not None
                            ):
                                qualification_checkpoint = dataclasses_replace(
                                    qualification_checkpoint,
                                    qualification_attempt_id=(
                                        f"{managed_ci.attached_run_id}/{managed_ci.run_attempt}"
                                    ),
                                )
                                post_pr_comment(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    body=_attach_round_metadata(
                                        f"PR #{pr_number} qualification checkpoint: attached attempt "
                                        f"{qualification_checkpoint.qualification_attempt_id}.",
                                        bound_pr_metadata(
                                            flow="pr",
                                            role="summary",
                                            agent="Orchestrator",
                                            round_number=round_number,
                                            subject=pr_metadata.head_sha,
                                            prior_items=tuple(unresolved_items),
                                            phase="qualification-checkpoint",
                                            qualification_checkpoint=qualification_checkpoint,
                                        ),
                                    ),
                                )
                            if managed_ci.activation_path == "ordinary_fallback":
                                ordinary_recovery_selected = True
                                ordinary_recovery = managed_ci.ordinary_recovery
                                managed_ci = None
                                if ordinary_recovery is None:
                                    raise AgentLoopError(
                                        f"PR #{pr_number} managed resume could not be correlated to ordinary "
                                        "recovery CI; no merge attempted."
                                    )
                                # A signed child-plan supersession stops the run before any freeze
                                # can ask a human for evidence under a plan authorized for replacement.
                                recheck_pending_child_plan_supersession()
                                # Exact-head evidence gate (#1068): the last reads before any
                                # finalization side effect; publishes the single freeze when evidence
                                # is pending, and otherwise revalidates head and signed input.
                                if evidence_revalidation_required or _pending_evidence_obligations(unresolved_items):
                                    gate_outcome = _evidence_freeze_gate(
                                        runner,
                                        config=config,
                                        pr_number=pr_number,
                                        round_number=round_number,
                                        head_sha=pr_metadata.head_sha,
                                        items=unresolved_items,
                                        surfaced_requirement_ids=evidence_surfaced_baseline,
                                        collect_requirement_ids=collect_live_requirement_ids,
                                        allowed_rounds=allowed_rounds,
                                        watch_failure_extension_used=watch_failure_extension_used,
                                        watch_head_extension_used=watch_head_extension_used,
                                        clearances=evidence_clearances,
                                    )
                                    assert gate_outcome.context is not None
                                    if gate_outcome.action == "head_changed":
                                        stage_evidence_head_change(gate_outcome.context)
                                        continue
                                    if gate_outcome.action == "refresh":
                                        stage_same_head_evidence_pass(gate_outcome.context, current_round=round_number)
                                        continue
                                merged = _finalize_ordinary_recovery_checked(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    round_number=round_number,
                                    items=unresolved_items,
                                    current_head_sha=pr_metadata.head_sha,
                                    capability=ordinary_recovery,
                                )
                                if merged:
                                    print(f"PR #{pr_number} merged after deliberate ordinary recovery.")
                                    unresolved_items = _clear_machine_obligations(
                                        unresolved_items, kind="github-pr-checks"
                                    )
                                    _record_staged_parent_completion_after_merge(
                                        runner, config=config, issue_context=issue_context,
                                        pr_number=pr_number,
                                    )
                                return 0
                            managed_outcome = wait_for_final_qualification(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                metadata=pr_metadata,
                                contract=managed_ci,
                            )
                            if managed_outcome.status == "passed":
                                if managed_outcome.head_sha != pr_metadata.head_sha:
                                    log(
                                        config,
                                        f"PR #{pr_number} managed CI reported success for a stale or "
                                        "unidentified head; requiring re-review",
                                    )
                                    resumed_round = None
                                    current_resume = None
                                    prefetched_pr_context = get_pr_review_context(
                                        runner, config=config, pr_number=pr_number
                                    )
                                    unresolved_items = _advance_machine_obligations_for_head(
                                        unresolved_items,
                                        current_head_sha=prefetched_pr_context.metadata.head_sha,
                                    )
                                    qualification_checkpoint = _machine_obligation_checkpoint(
                                        unresolved_items,
                                        current_head_sha=prefetched_pr_context.metadata.head_sha,
                                        base_branch=prefetched_pr_context.metadata.base_branch or config.base,
                                        allowed_rounds=allowed_rounds,
                                        watch_failure_extension_used=watch_failure_extension_used,
                                        watch_head_extension_used=watch_head_extension_used,
                                        lifecycle="awaiting_current_head_review",
                                    )
                                    _persist_qualification_checkpoint(
                                        runner,
                                        config=config,
                                        pr_number=pr_number,
                                        round_number=round_number + 1,
                                        head_sha=prefetched_pr_context.metadata.head_sha,
                                        unresolved_items=unresolved_items,
                                        checkpoint=qualification_checkpoint,
                                        message=(
                                            f"PR #{pr_number} qualification checkpoint: managed success was "
                                            "stale; current-head review is required."
                                        ),
                                    )
                                    continue
                                if selective_policy:
                                    fresh_context, fresh_requirement_ids, fresh_plan_context, config = _fresh_pr_qualification_snapshot(
                                        runner,
                                        config=config,
                                        pr_number=pr_number,
                                        issue_context=issue_context,
                                        parent_issue_context=parent_issue_context,
                                        approved_plan_context=approved_plan_context,
                                        scheduler_contract=scheduler_contract if selective_policy else None,
                                        allow_plan_handoff_change=True,
                                        planning_child_binding=planning_child_binding,
                                        managed_protection_mode=_managed_binding_protection_mode(managed_ci_handoff),
                                        managed_retired_plan_hashes=_managed_binding_retired_plan_hashes(managed_ci_handoff),
                                        plan_binding_reviewers=operator_reviewers,
                                    )
                                    if fresh_context.metadata.head_sha != pr_metadata.head_sha or fresh_context.architecture_identity_changed:
                                        prefetched_pr_context = fresh_context
                                        final_sweep_pending = False
                                        continue
                                    plan_identity_changed = fresh_plan_context != approved_plan_context
                                    if plan_identity_changed:
                                        approved_plan_context = fresh_plan_context
                                    if plan_identity_changed or set(fresh_requirement_ids) != {
                                        requirement.requirement_id for requirement in human_requirements
                                    }:
                                        # Signed input that reaches a PR carrying exact-head evidence is
                                        # answered at the same head inside this round (#1068).
                                        if not plan_identity_changed and (
                                            evidence_revalidation_required
                                            or _pending_evidence_obligations(unresolved_items)
                                        ):
                                            stage_same_head_evidence_pass(fresh_context, current_round=round_number)
                                            continue
                                        prefetched_pr_context = fresh_context
                                        final_sweep_pending = True
                                        continue
                                unresolved_items = _clear_machine_obligations(
                                    unresolved_items, kind="managed-exact-head-ci"
                                )
                                supersession_items = [
                                    item for item in unresolved_items
                                    if item.obligation_kind == "github-pr-checks"
                                    and _machine_obligation_is_revalidation_candidate(
                                        item, current_head_sha=pr_metadata.head_sha
                                    )
                                ]
                                supersession_verdict, supersession_predicate = (
                                    _managed_success_supersedes_ordinary_checks(
                                        unresolved_items,
                                        outcome=managed_outcome,
                                        current_head_sha=pr_metadata.head_sha,
                                        mergeability=(
                                            managed_outcome.mergeability
                                            or (
                                                _mergeability_for_unreadable_protection(
                                                    runner,
                                                    config=config,
                                                    pr_number=pr_number,
                                                    checks=managed_outcome.checks,
                                                )
                                                if supersession_items
                                                else None
                                            )
                                        ),
                                    )
                                )
                                if supersession_verdict == "cleared":
                                    unresolved_items = _clear_machine_obligations(
                                        unresolved_items, kind="github-pr-checks"
                                    )
                                    log(
                                        config,
                                        f"PR #{pr_number}: exact-head qualification at "
                                        f"{pr_metadata.head_sha} superseded carried github-pr-checks "
                                        f"({', '.join(item.item_id for item in supersession_items)})",
                                    )
                                elif supersession_verdict == "unqualified":
                                    raise AgentLoopError(
                                        f"PR #{pr_number} cannot finalize: carried github-pr-checks "
                                        f"({', '.join(item.item_id for item in supersession_items)}) "
                                        "was not superseded by exact-head qualification at "
                                        f"{pr_metadata.head_sha}: {supersession_predicate}. "
                                        "No approval or merge was attempted."
                                    )
                                # A signed child-plan supersession stops the run before any freeze
                                # can ask a human for evidence under a plan authorized for replacement.
                                recheck_pending_child_plan_supersession()
                                # Exact-head evidence gate (#1068): the last reads before any
                                # finalization side effect; publishes the single freeze when evidence
                                # is pending, and otherwise revalidates head and signed input.
                                if evidence_revalidation_required or _pending_evidence_obligations(unresolved_items):
                                    gate_outcome = _evidence_freeze_gate(
                                        runner,
                                        config=config,
                                        pr_number=pr_number,
                                        round_number=round_number,
                                        head_sha=pr_metadata.head_sha,
                                        items=unresolved_items,
                                        surfaced_requirement_ids=evidence_surfaced_baseline,
                                        collect_requirement_ids=collect_live_requirement_ids,
                                        allowed_rounds=allowed_rounds,
                                        watch_failure_extension_used=watch_failure_extension_used,
                                        watch_head_extension_used=watch_head_extension_used,
                                        clearances=evidence_clearances,
                                    )
                                    assert gate_outcome.context is not None
                                    if gate_outcome.action == "head_changed":
                                        stage_evidence_head_change(gate_outcome.context)
                                        continue
                                    if gate_outcome.action == "refresh":
                                        stage_same_head_evidence_pass(gate_outcome.context, current_round=round_number)
                                        continue
                                _ensure_finalization_ready(
                                    pr_number=pr_number,
                                    round_number=round_number,
                                    items=unresolved_items,
                                    current_head_sha=pr_metadata.head_sha,
                                    sub_item_stall_rounds=config.sub_item_stall_rounds,
                                    config=config,
                                )
                                if config.auto_merge:
                                    managed_ci_qualified = True
                                    prepare_v2_merge(
                                        runner,
                                        config=config,
                                        pr_number=pr_number,
                                        expected_head_sha=pr_metadata.head_sha,
                                        contract=managed_ci,
                                    )
                                    _merge_with_exact_head_proof(
                                        runner,
                                        config=config,
                                        pr_number=pr_number,
                                        proof=ExactHeadCiProof(
                                            head_sha=pr_metadata.head_sha or "",
                                            source="managed exact-head",
                                            base_ref=managed_ci.base_ref,
                                            repository=managed_ci.repository or config.repo,
                                        ),
                                    )
                                    print(
                                        f"PR #{pr_number} approved by "
                                        f"{format_agent_list(configured_reviewers)}."
                                        + announce_reduced_board_completion()
                                    )
                                    _record_staged_parent_completion_after_merge(
                                        runner, config=config, issue_context=issue_context,
                                        pr_number=pr_number,
                                    )
                                else:
                                    qualified_head = publish_manual_v2_qualification(
                                        runner,
                                        config=config,
                                        pr_number=pr_number,
                                        expected_head_sha=pr_metadata.head_sha,
                                        contract=managed_ci,
                                        reviewers=tuple(str(reviewer) for reviewer in configured_reviewers),
                                    )
                                    managed_ci_qualified = True
                                    merge_command = (
                                        f"gh pr merge {pr_number} --repo {shlex.quote(config.repo)} --merge "
                                        f"--match-head-commit {shlex.quote(qualified_head)}"
                                    )
                                    risk = (
                                        " GitHub cannot force a human or other automation to use that SHA."
                                        if managed_ci.protection_mode != "strict"
                                        else ""
                                    )
                                    print(
                                        f"PR #{pr_number} approved and qualified; manual merge required. "
                                        f"Qualified head: {qualified_head}. Run `{merge_command}` after "
                                        f"confirming the live head.{risk}"
                                        + announce_reduced_board_completion()
                                        + approved_pr_reopen_hint(pr_number)
                                    )
                                return 0
                            if managed_outcome.status == "head_changed":
                                log(
                                    config,
                                    f"PR #{pr_number} head changed during managed CI; re-review is required",
                                )
                                if round_number == allowed_rounds and not watch_head_extension_used:
                                    allowed_rounds += 1
                                    watch_head_extension_used = True
                                resumed_round = None
                                current_resume = None
                                prefetched_pr_context = get_pr_review_context(
                                    runner, config=config, pr_number=pr_number
                                )
                                unresolved_items = _advance_machine_obligations_for_head(
                                    unresolved_items,
                                    current_head_sha=prefetched_pr_context.metadata.head_sha,
                                )
                                qualification_checkpoint = _machine_obligation_checkpoint(
                                    unresolved_items,
                                    current_head_sha=prefetched_pr_context.metadata.head_sha,
                                    base_branch=prefetched_pr_context.metadata.base_branch or config.base,
                                    allowed_rounds=allowed_rounds,
                                    watch_failure_extension_used=watch_failure_extension_used,
                                    watch_head_extension_used=watch_head_extension_used,
                                    lifecycle="awaiting_current_head_review",
                                )
                                _persist_qualification_checkpoint(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    round_number=round_number + 1,
                                    head_sha=prefetched_pr_context.metadata.head_sha,
                                    unresolved_items=unresolved_items,
                                    checkpoint=qualification_checkpoint,
                                    message=(
                                        f"PR #{pr_number} qualification checkpoint: head changed during "
                                        "managed validation; current-head review is required."
                                    ),
                                )
                                continue
                            if managed_outcome.status == "merge_conflict":
                                assert managed_outcome.mergeability is not None
                                latest_mergeability = managed_outcome.mergeability
                                unresolved_items = _reconcile_merge_conflict_item(
                                    unresolved_items,
                                    mergeability=managed_outcome.mergeability,
                                    source_round=round_number,
                                    current_head_sha=pr_metadata.head_sha,
                                )
                            elif managed_outcome.status == "infrastructure_stall":
                                assert managed_outcome.stall is not None
                                post_pr_comment(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    body=_format_ci_infrastructure_comment(
                                        pr_number, managed_outcome.stall
                                    ),
                                )
                                post_pr_comment(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    body=_append_evidence_barrier_note(
                                        _ci_infrastructure_stop_message(
                                            pr_number, managed_outcome.stall, []
                                        ),
                                        unresolved_items,
                                    ),
                                )
                                log(
                                    config,
                                    f"Round {round_number}: PR #{pr_number} managed CI wait "
                                    "stopped on external infrastructure blocking; no merge attempted",
                                )
                                print(
                                    f"PR #{pr_number} was approved by "
                                    f"{format_agent_list(configured_reviewers)}, but external GitHub "
                                    "Actions infrastructure is blocking managed exact-head CI "
                                    f"({'; '.join(_ci_infrastructure_details(managed_outcome.stall))}). "
                                    "No merge was attempted; rerun the same command once GitHub Actions "
                                    "runners recover."
                                )
                                return 0
                            elif managed_outcome.status == "terminal_without_status":
                                return _stop_on_terminal_without_status(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    round_number=round_number,
                                    outcome=managed_outcome,
                                )
                            elif managed_outcome.status == "failed":
                                details = (
                                    list(managed_outcome.failure_details)
                                    if managed_outcome.failure_details
                                    else _pr_check_details(managed_outcome.checks)
                                    if managed_outcome.checks
                                    else ["Managed exact-head CI failed."]
                                )
                                post_pr_comment(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    body=_format_pr_checks_comment(pr_number, "failing", details),
                                )
                                had_managed_obligation = any(
                                    _is_machine_obligation(item)
                                    and item.obligation_kind == "managed-exact-head-ci"
                                    for item in unresolved_items
                                )
                                unresolved_items = _upsert_machine_obligation(
                                    unresolved_items,
                                    item_number=next_unresolved_item_number,
                                    kind="managed-exact-head-ci",
                                    source_round=round_number,
                                    text=_pr_check_blocking_review(
                                        pr_number, "failing", details
                                    ),
                                    failed_head_sha=managed_outcome.head_sha or pr_metadata.head_sha,
                                )
                                if not had_managed_obligation:
                                    next_unresolved_item_number += 1
                                if round_number == allowed_rounds and not watch_failure_extension_used:
                                    allowed_rounds += 1
                                    watch_failure_extension_used = True
                            else:
                                legacy_timeout_message = (
                                    f"Managed exact-head CI for PR #{pr_number} did not pass within "
                                    f"{config.ci_timeout_seconds}s."
                                )
                                timeout_board = managed_outcome.checks
                                if (
                                    timeout_board is None
                                    or not pr_metadata.head_sha
                                    or managed_outcome.head_sha != pr_metadata.head_sha
                                    or timeout_board.check_query_status != "ok"
                                    or timeout_board.check_query_errors
                                ):
                                    raise AgentLoopError(legacy_timeout_message)
                                stalled_names = {
                                    stalled.name for stalled in timeout_board.infrastructure_stalls
                                }
                                repairable = tuple(
                                    check for check in timeout_board.failing
                                    if check.name != FINAL_CONTEXT and check.name not in stalled_names
                                )
                                if not repairable:
                                    final_observed = [
                                        check for check in (
                                            *timeout_board.passing, *timeout_board.pending, *timeout_board.failing
                                        )
                                        if check.name == FINAL_CONTEXT
                                    ]
                                    pending_names = [
                                        check.name for check in timeout_board.pending
                                        if check.name != FINAL_CONTEXT
                                    ]
                                    if not pending_names and not timeout_board.missing_required:
                                        raise AgentLoopError(legacy_timeout_message)
                                    raise AgentLoopError(
                                        f"{legacy_timeout_message} Exact-head board at "
                                        f"{pr_metadata.head_sha}: final context "
                                        f"{final_observed[0].status.lower() if final_observed else 'not reporting'}; "
                                        f"pending: {', '.join(pending_names) or 'none'}; "
                                        "required not reporting: "
                                        f"{', '.join(timeout_board.missing_required) or 'none'}."
                                    )
                                details = _pr_check_details(
                                    dataclasses_replace(timeout_board, failing=repairable)
                                )
                                post_pr_comment(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    body=_format_pr_checks_comment(pr_number, "failing", details),
                                )
                                had_checks_obligation = any(
                                    _is_machine_obligation(item)
                                    and item.obligation_kind == "github-pr-checks"
                                    for item in unresolved_items
                                )
                                unresolved_items = _upsert_machine_obligation(
                                    unresolved_items,
                                    item_number=next_unresolved_item_number,
                                    kind="github-pr-checks",
                                    source_round=round_number,
                                    text=_pr_check_blocking_review(pr_number, "failing", details),
                                    failed_head_sha=pr_metadata.head_sha,
                                )
                                if not had_checks_obligation:
                                    next_unresolved_item_number += 1
                                if round_number == allowed_rounds and not watch_failure_extension_used:
                                    allowed_rounds += 1
                                    watch_failure_extension_used = True
                        elif ordinary_recovery_selected:
                            if ordinary_recovery is None:
                                raise AgentLoopError(
                                    f"PR #{pr_number} ordinary recovery provenance is unavailable; "
                                    "no merge attempted."
                                )
                            # A signed child-plan supersession stops the run before any freeze
                            # can ask a human for evidence under a plan authorized for replacement.
                            recheck_pending_child_plan_supersession()
                            # Exact-head evidence gate (#1068): the last reads before any
                            # finalization side effect; publishes the single freeze when evidence
                            # is pending, and otherwise revalidates head and signed input.
                            if evidence_revalidation_required or _pending_evidence_obligations(unresolved_items):
                                gate_outcome = _evidence_freeze_gate(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    round_number=round_number,
                                    head_sha=pr_metadata.head_sha,
                                    items=unresolved_items,
                                    surfaced_requirement_ids=evidence_surfaced_baseline,
                                    collect_requirement_ids=collect_live_requirement_ids,
                                    allowed_rounds=allowed_rounds,
                                    watch_failure_extension_used=watch_failure_extension_used,
                                    watch_head_extension_used=watch_head_extension_used,
                                    clearances=evidence_clearances,
                                )
                                assert gate_outcome.context is not None
                                if gate_outcome.action == "head_changed":
                                    stage_evidence_head_change(gate_outcome.context)
                                    continue
                                if gate_outcome.action == "refresh":
                                    stage_same_head_evidence_pass(gate_outcome.context, current_round=round_number)
                                    continue
                            merged = _finalize_ordinary_recovery_checked(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                round_number=round_number,
                                items=unresolved_items,
                                current_head_sha=pr_metadata.head_sha,
                                capability=ordinary_recovery,
                            )
                            if merged:
                                print(f"PR #{pr_number} merged after deliberate ordinary recovery.")
                            if merged:
                                unresolved_items = _clear_machine_obligations(
                                    unresolved_items, kind="github-pr-checks"
                                )
                                _record_staged_parent_completion_after_merge(
                                    runner, config=config, issue_context=issue_context,
                                    pr_number=pr_number,
                                )
                            return 0
                        elif config.auto_merge:
                            raise AgentLoopError(
                                f"PR #{pr_number} reached the ordinary auto-merge path without "
                                "consuming a full-board CI watcher result; no merge attempted."
                            )
                        must_fix_items = list(
                            _partition_unresolved_items(
                                unresolved_items,
                                current_head_sha=pr_metadata.head_sha,
                            )["coder_blockers"]
                        )
                    if not must_fix_items:
                        # A signed child-plan supersession stops the run before any freeze
                        # can ask a human for evidence under a plan authorized for replacement.
                        recheck_pending_child_plan_supersession()
                        # Exact-head evidence gate (#1068): the last reads before any
                        # finalization side effect; publishes the single freeze when evidence
                        # is pending, and otherwise revalidates head and signed input.
                        if evidence_revalidation_required or _pending_evidence_obligations(unresolved_items):
                            gate_outcome = _evidence_freeze_gate(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                round_number=round_number,
                                head_sha=pr_metadata.head_sha,
                                items=unresolved_items,
                                surfaced_requirement_ids=evidence_surfaced_baseline,
                                collect_requirement_ids=collect_live_requirement_ids,
                                allowed_rounds=allowed_rounds,
                                watch_failure_extension_used=watch_failure_extension_used,
                                watch_head_extension_used=watch_head_extension_used,
                                clearances=evidence_clearances,
                            )
                            assert gate_outcome.context is not None
                            if gate_outcome.action == "head_changed":
                                stage_evidence_head_change(gate_outcome.context)
                                continue
                            if gate_outcome.action == "refresh":
                                stage_same_head_evidence_pass(gate_outcome.context, current_round=round_number)
                                continue
                        _ensure_finalization_ready(
                            pr_number=pr_number,
                            round_number=round_number,
                            items=unresolved_items,
                            current_head_sha=pr_metadata.head_sha,
                            sub_item_stall_rounds=config.sub_item_stall_rounds,
                            observations=finalization_observations,
                            config=config,
                        )
                        print(
                            f"PR #{pr_number} approved by {format_agent_list(configured_reviewers)}."
                            + announce_reduced_board_completion()
                            + approved_pr_reopen_hint(pr_number)
                        )
                        return 0
            # A clustered sibling after a step-back stops for a human decision
            # here, before the round-limit exit and any coder invocation; K clustered blocks otherwise
            # turn this follow-up into one generalize-the-class step-back turn.
            step_back_decision = _pr_step_back_decision(
                runner,
                config,
                pr_number=pr_number,
                round_number=round_number,
                snapshot_comments=pr_comments,
                tracked=_pr_step_back_tracked_reviewers(config, configured_reviewers),
                head_sha=pr_metadata.head_sha,
            )
            if round_number == allowed_rounds:
                raise AgentLoopError(
                    _round_limit_diagnostic(
                        pr_number=pr_number,
                        round_number=round_number,
                        items=unresolved_items,
                        current_head_sha=pr_metadata.head_sha,
                        sub_item_stall_rounds=config.sub_item_stall_rounds,
                    )
                )

            has_merge_conflict_item = any(
                item.item_id == MERGE_CONFLICT_ITEM_ID for item in unresolved_items
            )

            if has_merge_conflict_item:
                if (
                    conflict_dispatch_head_sha is not None
                    and conflict_dispatch_head_sha == (pr_metadata.head_sha or "")
                ):
                    # The previous conflict-resolution round was dispatched from
                    # this exact head and the head still has not moved: another
                    # coder round would just repeat the same conflict. Stop
                    # cleanly and resumably instead of looping (#606).
                    conflict_base = (
                        (latest_mergeability.base_branch if latest_mergeability else None)
                        or pr_metadata.base_branch
                        or config.base
                        or "its base branch"
                    )
                    post_pr_comment(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        body=(
                            f"PR #{pr_number} is still conflicted with `{conflict_base}` and the head "
                            f"did not change after the last conflict-resolution round. Push a resolved "
                            "head to continue, then rerun the same command.\n\n"
                            f"<!-- AGENT_STATE: blocking -->\n-- Orchestrator"
                        ),
                    )
                    log(
                        config,
                        f"Round {round_number}: PR #{pr_number} is still conflicted with "
                        f"`{conflict_base}` and the head did not advance; stopping cleanly",
                    )
                    print(
                        f"PR #{pr_number} is still conflicted with `{conflict_base}` and the head "
                        "did not change after the last conflict-resolution round. Push a resolved "
                        "head to continue, then rerun the same command."
                    )
                    return 0
                conflict_dispatch_head_sha = pr_metadata.head_sha or ""

            # Structural backstop (#602): even when the reviewer downgrade above
            # correctly declines (a mixed item, a genuine defect alongside a
            # stalled check), a coder round must never be able to wait
            # indefinitely on external CI infrastructure. Always confirm the
            # freshest check board before starting the round and, when it is
            # wholly infrastructure-blocked, prepend the stall context so the
            # coder fixes only the genuine items and returns a bounded terminal
            # response instead of chasing the stalled check. Skipped entirely
            # while conflicted (#606): a conflicted branch must not generate any
            # check-run, commit-status, or branch-protection API calls.
            if pr_checks is None and not has_merge_conflict_item:
                pr_checks = get_pr_checks(runner, config=config, metadata=pr_metadata)
                if managed_ci_active(pr_metadata):
                    pr_checks = intermediate_managed_checks(pr_checks)
            if pr_checks is not None and not has_merge_conflict_item:
                stalled = {(check.kind, check.name) for check in pr_checks.infrastructure_stalls}
                failures = tuple(
                    check for check in pr_checks.failing
                    if (check.kind, check.name) not in stalled
                )
                # This is a single post-review snapshot, not a CI wait. Only
                # observed failures become work; missing/pending checks do not.
                if failures:
                    failure_snapshot = dataclasses_replace(
                        pr_checks, state="failing", failing=failures,
                        pending=(), missing_required=(), infrastructure_stalls=(),
                        required_checks=(), branch_protection_note=None,
                    )
                    details = _pr_check_details(failure_snapshot)
                    details.append(f"Reviewed head: {pr_metadata.head_sha}")
                    text = _pr_check_blocking_review(pr_number, "failing", details) + (
                        "\nInspect the linked failure logs and address these failures alongside "
                        "the reviewer findings. Run relevant local regression tests. Do not wait "
                        "for queued or running CI checks before returning your follow-up."
                    )
                    had_ordinary_obligation = any(
                        _is_machine_obligation(item)
                        and item.obligation_kind == "github-pr-checks"
                        for item in unresolved_items
                    )
                    unresolved_items = _upsert_machine_obligation(
                        unresolved_items,
                        item_number=next_unresolved_item_number,
                        kind="github-pr-checks",
                        source_round=round_number,
                        text=text,
                        failed_head_sha=pr_metadata.head_sha,
                    )
                    if not had_ordinary_obligation:
                        next_unresolved_item_number += 1
                    log(config, f"Round {round_number}: including available CI failures in coder follow-up")
                    post_pr_comment(
                        runner, config=config, pr_number=pr_number,
                        body=_format_pr_checks_comment(pr_number, "failing", details),
                    )
            # Evidence-only stall (#1324): decided after the check snapshot and
            # CI reconciliation above, before any coder invocation.  Not
            # evaluated on the merge-conflict path, which makes no check calls.
            evidence_stall_snapshot = (
                None
                if has_merge_conflict_item
                else _pr_evidence_stall_decision(
                    runner,
                    config,
                    pr_number=pr_number,
                    round_number=round_number,
                    head_sha=pr_metadata.head_sha,
                    approved_plan_context=approved_plan_context,
                    open_items=unresolved_items,
                    pr_checks=pr_checks,
                )
            )
            stall_context = (
                _coder_infrastructure_stall_notice(pr_checks.infrastructure_stalls)
                if pr_checks is not None and is_wholly_infrastructure_blocked(pr_checks)
                else ""
            )

            same_pr_items = [item for item in unresolved_items if item.status == "same-pr"]
            blocking_items = [item for item in unresolved_items if item.status == "blocking"]
            summary_context = (
                "Latest reviewer summaries (review-level context):\n"
                "These summaries supplement the item ledger; they do not create new item IDs "
                "or replace Original claims. Do not treat one reviewer's summary as another's "
                "item-specific evidence.\n\n"
                + "\n\n".join(reviewer_summaries.values()) + "\n\n"
                if reviewer_summaries else ""
            )
            if (
                current_resume is not None
                and current_resume.unrecorded_head_advance
                and current_resume.rejected_coder_followup_reason
            ):
                # The orchestrator rejected the previous coder turn (#1292).
                summary_context = (
                    f"Recovery context: the previous coder follow-up (attempt "
                    f"{current_resume.dispatch_attempt or 1}) was dispatched on "
                    f"`{current_resume.rejected_coder_followup_from_head or 'unknown'}`; the "
                    f"current head is `{pr_metadata.head_sha or 'unknown'}`. The orchestrator "
                    "did not accept its response: "
                    f"{current_resume.rejected_coder_followup_reason} Nothing from that attempt "
                    "was reviewed or recorded. Check each recovered item against the current "
                    "head, fix the cause of the rejection, and report only valid in-checkout "
                    "test evidence. List an item in addressed_items only after confirming the "
                    "current head satisfies it.\n\n"
                    + summary_context
                )
            elif current_resume is not None and current_resume.unrecorded_head_advance:
                # The external head may or may not contain the fixes; the
                # coder must check each recovered item against it (#1034).
                summary_context = (
                    f"Recovery context: the PR head `{pr_metadata.head_sha or 'unknown'}` was "
                    "advanced by a commit that carries no coder metadata (for example a manual "
                    "push). That commit may or may not satisfy the recovered items below. Check "
                    "each recovered item against the current head. List an item in "
                    "addressed_items only after confirming the current head satisfies it; keep "
                    "any unsatisfied item in remaining_items and fix it with a commit if "
                    "possible. Your notes must describe what the current head actually "
                    "contains, not restate the reviewer's suggestion or assume the external "
                    "commit's intent.\n\n"
                    + summary_context
                )
            # Machine obligations inserted since reconciliation (CI failures).
            pr_finding_history.observe_reconciled(unresolved_items)
            finding_history_view = pr_finding_history.view(round_number)
            step_back_guidance = (
                step_back_decision.coder_guidance() if not has_merge_conflict_item else ""
            )
            step_back_entries = (
                step_back_decision.entry_payloads() if step_back_guidance else ()
            )
            if has_merge_conflict_item:
                other_items = [
                    item for item in unresolved_items if item.item_id != MERGE_CONFLICT_ITEM_ID
                ]
                coder_followup_items = select_coder_followup_items(other_items)
                combined_review = summary_context + format_coder_followup_context(other_items)
                coder_human_requirements_context = render_coder_human_requirements_prompt_context(
                    human_requirements
                )
                resolved_base_branch = (
                    (latest_mergeability.base_branch if latest_mergeability else None)
                    or pr_metadata.base_branch
                    or config.base
                    or "the base branch"
                )
                resolved_head_sha = (
                    (latest_mergeability.head_sha if latest_mergeability else None) or pr_metadata.head_sha
                )
                merge_state_detail = (
                    f"mergeable={latest_mergeability.mergeable_raw or 'unknown'}, "
                    f"mergeStateStatus={latest_mergeability.merge_state_raw or 'unknown'}"
                    if latest_mergeability is not None
                    else "mergeable=unknown, mergeStateStatus=unknown"
                )
                followup_prompt = build_merge_conflict_prompt(
                    pr_number,
                    round_number,
                    combined_review,
                    config,
                    memory,
                    issue_context=issue_context,
                    human_requirements=human_requirements,
                    base_branch=resolved_base_branch,
                    head_sha=resolved_head_sha,
                    merge_state_detail=merge_state_detail,
                    human_requirements_context=coder_human_requirements_context,
                    approved_plan_context=approved_plan_context,
                    parent_issue_context=parent_issue_context,
                )
                log(config, f"Round {round_number}: {coder_name} resolving merge conflict")
            elif same_pr_items and not blocking_items:
                # This dispatch intentionally omits retained future work. Keep
                # the validator and repair pass on the exact same visible
                # classifiable item set as the same-PR prompt.
                coder_followup_items = select_coder_followup_items(same_pr_items)
                combined_review = stall_context + summary_context + _format_same_pr_unresolved_items(same_pr_items)
                coder_human_requirements_context = render_coder_human_requirements_prompt_context(
                    human_requirements
                )
                followup_prompt = build_same_pr_followup_prompt(
                    pr_number,
                    round_number,
                    combined_review,
                    config,
                    memory,
                    issue_context=issue_context,
                    human_requirements=human_requirements,
                    human_requirements_context=coder_human_requirements_context,
                    approved_plan_context=approved_plan_context,
                    parent_issue_context=parent_issue_context,
                    step_back_context=step_back_guidance,
                    generalization_guidance=True,
                    finding_history=finding_history_view,
                )
                _log_coder_followup_dispatch(config, round_number, coder_name, coder_followup_items)
            else:
                coder_followup_items = select_coder_followup_items(unresolved_items)
                combined_review = stall_context + summary_context + format_coder_followup_context(unresolved_items)
                coder_human_requirements_context = render_coder_human_requirements_prompt_context(
                    human_requirements
                )
                followup_prompt = build_followup_prompt(
                    pr_number,
                    round_number,
                    combined_review,
                    config,
                    memory,
                    issue_context=issue_context,
                    human_requirements=human_requirements,
                    human_requirements_context=coder_human_requirements_context,
                    approved_plan_context=approved_plan_context,
                    parent_issue_context=parent_issue_context,
                    step_back_context=step_back_guidance,
                    generalization_guidance=True,
                    finding_history=finding_history_view,
                )
                _log_coder_followup_dispatch(config, round_number, coder_name, coder_followup_items)
            repair_unresolved_item_ids = tuple(
                item.item_id for item in coder_followup_items
            )
            if (
                evidence_pass is not None
                and _frozen_evidence_obligations(unresolved_items)
            ):
                # A machine gate failed after the pass settled: the pass still
                # ends with exactly one terminal record, released before any
                # head-changing dispatch (#1068).
                unresolved_items = _publish_evidence_release(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    round_number=round_number,
                    head_sha=str(pr_metadata.head_sha),
                    items=unresolved_items,
                    reason="machine-gate",
                    surfaced_requirement_ids=surfaced_reviewer_requirement_ids,
                    allowed_rounds=allowed_rounds,
                    watch_failure_extension_used=watch_failure_extension_used,
                    watch_head_extension_used=watch_head_extension_used,
                    clearances=evidence_clearances,
                )
            _refuse_dispatch_while_evidence_frozen(
                unresolved_items,
                pr_number=pr_number,
                operation=(
                    "a merge-conflict resolution"
                    if has_merge_conflict_item
                    else "a coder follow-up or CI repair"
                ),
            )
            # Persist the exact machine state and any consumed watcher budget
            # before invoking the coder. If the agent process is interrupted
            # during repair, resume must not forget the failed head or mint the
            # same extension again.
            qualification_checkpoint = _machine_obligation_checkpoint(
                unresolved_items,
                current_head_sha=pr_metadata.head_sha,
                base_branch=pr_metadata.base_branch or config.base,
                allowed_rounds=allowed_rounds,
                watch_failure_extension_used=watch_failure_extension_used,
                watch_head_extension_used=watch_head_extension_used,
                approval_digest=_qualification_digest(
                    tuple(sorted({
                        *unchanged_head_approvals,
                        *(name for name, _output in approved_review_outputs),
                    }))
                ),
                plan_digest=(
                    approved_plan_context.plan_hash
                    if approved_plan_context is not None else None
                ),
                requirements_digest=_qualification_digest(
                    tuple(requirement.requirement_id for requirement in human_requirements)
                ),
                acquisition_digest=_qualification_digest(reviewer_acquisition_contract),
                scheduler_digest=_qualification_digest(
                    _prior_item_ledger_signature(unresolved_items)
                ),
            )
            _persist_qualification_checkpoint(
                runner,
                config=config,
                pr_number=pr_number,
                round_number=round_number + 1,
                head_sha=pr_metadata.head_sha,
                unresolved_items=unresolved_items,
                checkpoint=qualification_checkpoint,
                message=(
                    f"PR #{pr_number} qualification checkpoint: machine obligation state "
                    "persisted before coder handoff."
                ),
            )
            # Record the slot, ledger, budget and attempt before the coder can
            # push, so a rejected or interrupted turn is resumable (#1292).  If
            # the record cannot be written, the coder is not dispatched.
            pre_turn_ledger = tuple(unresolved_items)
            pre_turn_head = str(pr_metadata.head_sha or "")
            dispatch_budget = RecoveryRoundBudget(
                allowed_rounds, watch_failure_extension_used, watch_head_extension_used
            )
            coder_recovery_resume = (
                current_resume
                if current_resume is not None
                and current_resume.unrecorded_head_advance
                and current_resume.dispatch_attempt is not None
                else None
            )
            dispatch_attempt = (
                coder_recovery_resume.dispatch_attempt + 1
                if coder_recovery_resume is not None
                else 1
            )
            dispatch_carried_reasons = (
                coder_recovery_resume.carried_rejection_reasons
                if coder_recovery_resume is not None
                else ()
            )
            _persist_coder_dispatch(
                runner,
                config=config,
                pr_number=pr_number,
                dispatch_round=round_number,
                dispatch_head=pre_turn_head,
                ledger=pre_turn_ledger,
                budget=dispatch_budget,
                attempt=dispatch_attempt,
                carried_reasons=dispatch_carried_reasons,
            )
            try:
                coder_response = _run_validated_agent(
                    runner,
                    agent=config.coder,
                    config=config,
                    prompt=followup_prompt,
                    session_id=coder_session_id,
                    marker_description="<!-- AGENT_STATE: approved|blocking -->",
                    require_architecture_impact_contract=True,
                    **_architecture_mode_validators(lambda mode: lambda text, items=tuple(coder_followup_items), human_requirements=human_requirements: _validate_coder_followup_response(
                        text,
                        unresolved_items=items,
                        human_requirements=human_requirements,
                        # Every new coder-followup response is v1. A missing
                        # document must not silently downgrade the protocol.
                        required_architecture_impact_contract=1,
                        delivered_risk_test_matrix=(
                            approved_plan_context.risk_test_matrix_payload
                            if approved_plan_context is not None and approved_plan_context.matrix_available
                            else None
                        ),
                        delivered_risk_test_matrix_identity=(
                            approved_plan_context.risk_test_matrix_identity
                            if approved_plan_context is not None and approved_plan_context.matrix_available
                            else None
                        ),
                        required_risk_test_matrix_contract=(
                            1 if approved_plan_context is not None and approved_plan_context.matrix_available
                            else 0
                        ),
                        authoritative_test_observations=_current_test_turn_observations(runner),
                        execution_catalog=_current_test_turn_observations(runner),
                        delivered_risk_test_matrix_row_ids=(
                            approved_plan_context.risk_test_matrix_expected_row_ids
                            if approved_plan_context is not None and approved_plan_context.matrix_available
                            else None
                        ), architecture_status_mode=mode,
                    )),
                    usage_context=usage_context,
                    role="coder",
                    use_repair=True,
                    repair_expected_kind="coder_followup",
                    repair_unresolved_item_ids=repair_unresolved_item_ids,
                    repair_surfaced_requirement_ids=coder_human_requirements_context.surfaced_requirement_ids,
                    repair_requires_direct_discussion_ack=coder_human_requirements_context.requires_direct_discussion_ack,
                    salvage_context=SalvageContext(
                        repo=config.repo,
                        issue_number=None if issue_context is None else issue_context.number,
                        scope=PR_FOLLOWUP_SALVAGE_SCOPE,
                        agent=config.coder,
                        run_id=usage_context.run_id,
                    ),
                    operation_description="PR feedback follow-up",
                )
                coder_output = coder_response.text
                coder_session_id = coder_response.session_id
                latest_coder_output = coder_output
                public_comment = coder_output
                raw_structured_coder_response: str | None = None
                if isinstance(coder_response.marker_value, StructuredCoderFollowup):
                    coder_response = dataclasses_replace(
                        coder_response,
                        marker_value=_degrade_out_of_checkout_tests(
                            coder_response.marker_value, config=config
                        ),
                    )
                    validate_test_observation_citations_within_workdir(
                        coder_response.marker_value.test_observations,
                        assigned_workdir=active_workdir(config),
                    )
                    raw_structured_coder_response = coder_output
                    if coder_response.marker_value.disputed_items:
                        unresolved_items = _apply_dispute_evidence(
                            unresolved_items,
                            disputed_items=coder_response.marker_value.disputed_items,
                            dispute_evidence=coder_response.marker_value.dispute_evidence,
                        )
                        disputed_names = ", ".join(coder_response.marker_value.disputed_items)
                        log(
                            config,
                            f"Round {round_number}: {coder_name} disputed item(s) {disputed_names} "
                            "with counter-evidence; will surface to human if reviewer still blocks",
                        )
                    public_comment = render_public_agent_comment(
                        kind="coder_followup",
                        parsed=coder_response.marker_value,
                        agent=config.coder,
                        prior_items=tuple(unresolved_items),
                        config=config,
                        model_used=coder_response.model_used,
                    )
                else:
                    validate_response_tests_within_workdir(
                        coder_output,
                        assigned_workdir=active_workdir(config),
                    )
                    public_comment = normalize_freeform_signature(
                        coder_output, agent=config.coder, config=config, model_used=coder_response.model_used
                    )

                unresolved_items = _reconcile_human_requirements_ack_item(
                    unresolved_items,
                    coder_output=coder_output,
                    human_requirements=human_requirements,
                    source_round=round_number,
                )
                updated_pr_context = get_pr_review_context(runner, config=config, pr_number=pr_number)
                unresolved_items = _advance_machine_obligations_for_head(
                    unresolved_items,
                    current_head_sha=updated_pr_context.metadata.head_sha,
                )
                # Reconcile the coder's assigned checkout with the freshly fetched
                # PR head before deriving any canonical evidence. The builder also
                # takes an independent stable snapshot, so a mismatch is retained
                # as non-verified evidence rather than discarding the PR handoff.
                assigned_worktree_head_after_followup = _read_assigned_workdir_head(
                    runner, config
                )
                if isinstance(coder_response.marker_value, StructuredCoderFollowup):
                    # A citation-only follow-up may leave the head unchanged;
                    # any other addressed item expects a pushed change (#1338).
                    # Classified against the ledger and unsatisfied rows fixed
                    # before the coder turn, never against agent prose.
                    head_change_expected = _followup_reports_code_changes(
                        coder_response.marker_value,
                        prior_items=pre_turn_ledger,
                        citation_row_ids=_evidence_stall_citation_rows(
                            evidence_stall_snapshot
                        ),
                    )
                    derived_followup, _followup_derived_risk_evidence = (
                        _derive_authenticated_risk_evidence_for_coder(
                            coder_response.marker_value,
                            approved_plan_context=approved_plan_context,
                            runner=runner,
                            assigned_workdir=active_workdir(config),
                            head_sha=updated_pr_context.metadata.head_sha,
                            predecessor_head=pr_metadata.head_sha,
                            head_change_expected=head_change_expected,
                            config=config,
                            session_id=coder_response.session_id,
                            invocation_id=coder_response.acquisition_test_turn_id,
                            _closed_execution_catalog=coder_response.acquisition_test_observations,
                            _journal_observations=coder_response.acquisition_test_observations,
                            assigned_worktree_head=assigned_worktree_head_after_followup,
                            reauthenticate_head=lambda: get_pr_review_context(
                                runner, config=config, pr_number=pr_number
                            ).metadata.head_sha,
                        )
                    )
                    coder_response = dataclasses_replace(
                        coder_response,
                        marker_value=derived_followup,
                    )
                local_test_evidence = runner.render_local_test_evidence(
                    current_head=updated_pr_context.metadata.head_sha,
                    legacy_tests_run=(
                        coder_response.marker_value.tests_run
                        if isinstance(coder_response.marker_value, StructuredCoderFollowup)
                        else None
                    ),
                    cwd=active_workdir(config),
                    prior_local_test_evidence=(
                        latest_coder_metadata.local_test_evidence
                        if latest_coder_metadata is not None
                        else None
                    ),
                )
                # The coder metadata record posted below is numbered one past the
                # loop round; the matrix-evidence anchor must use that number.
                coder_record_round = round_number + 1
                if isinstance(coder_response.marker_value, StructuredCoderFollowup):
                    pr_finding_history.record_fix(
                        coder_response.marker_value,
                        published_round=coder_record_round,
                        agent=coder_name,
                    )
                    log_declared_generalization(
                        coder_response.marker_value,
                        log=lambda message: log(config, message),
                        round_number=round_number,
                        agent=coder_name,
                        step_back_directed=bool(step_back_guidance),
                    )
                matrix_evidence_render_decision = None
                if (
                    isinstance(coder_response.marker_value, StructuredCoderFollowup)
                    and coder_response.marker_value.risk_test_matrix_evidence is not None
                ):
                    matrix_evidence_render_decision = resolve_matrix_evidence_render(
                        coder_response.marker_value.risk_test_matrix_evidence,
                        latest_coder_metadata,
                        coder_record_round,
                    )
                # The head this follow-up was dispatched against; persisted so the
                # head-unchanged framing below survives resume (#1034).
                followup_dispatch_head = (
                    pr_metadata.head_sha
                    if _is_followup_dispatch_head(pr_metadata.head_sha)
                    else None
                )
                head_unchanged_sha = (
                    followup_dispatch_head
                    if followup_dispatch_head is not None
                    and updated_pr_context.metadata.head_sha == followup_dispatch_head
                    else None
                )
                followup_bound_head = (
                    _followup_derived_risk_evidence.bound_head_sha
                    if isinstance(coder_response.marker_value, StructuredCoderFollowup)
                    and _followup_derived_risk_evidence is not None
                    else None
                )
                followup_coverage_assessment = (
                    final_risk_coverage_assessment(
                        coder_response.marker_value,
                        approved_plan_context=approved_plan_context,
                        workdir=active_workdir(config),
                        initial_head_sha=updated_pr_context.metadata.head_sha,
                        derived=_followup_derived_risk_evidence,
                    )
                    if isinstance(coder_response.marker_value, StructuredCoderFollowup)
                    else None
                )
                if isinstance(coder_response.marker_value, StructuredCoderFollowup):
                    public_comment = render_public_agent_comment(
                        kind="coder_followup",
                        parsed=coder_response.marker_value,
                        agent=config.coder,
                        prior_items=tuple(unresolved_items),
                        config=config,
                        model_used=coder_response.model_used,
                        local_test_evidence=local_test_evidence,
                        current_test_turn_id=coder_response.acquisition_test_turn_id,
                        matrix_evidence_render_decision=matrix_evidence_render_decision,
                        head_unchanged_sha=head_unchanged_sha,
                        coverage_assessment=followup_coverage_assessment,
                    )
                elif head_unchanged_sha is not None:
                    public_comment = add_coder_followup_head_unchanged_notice(
                        public_comment, head_unchanged_sha
                    )
                # Carry the orchestrator's own rendering, never text re-parsed from a
                # comment that also holds coder prose (#1290).
                latest_coder_coverage_map = (
                    render_coverage_map(followup_coverage_assessment) or None
                )

                qualification_checkpoint = _machine_obligation_checkpoint(
                    unresolved_items,
                    current_head_sha=updated_pr_context.metadata.head_sha,
                    base_branch=updated_pr_context.metadata.base_branch or config.base,
                    allowed_rounds=allowed_rounds,
                    watch_failure_extension_used=watch_failure_extension_used,
                    watch_head_extension_used=watch_head_extension_used,
                    plan_digest=(
                        approved_plan_context.plan_hash
                        if approved_plan_context is not None else None
                    ),
                    requirements_digest=_qualification_digest(
                        tuple(requirement.requirement_id for requirement in human_requirements)
                    ),
                    acquisition_digest=_qualification_digest(reviewer_acquisition_contract),
                    scheduler_digest=_qualification_digest(
                        _prior_item_ledger_signature(unresolved_items)
                    ),
                )
                latest_coder_metadata = bound_pr_metadata(
                    flow="pr",
                    role="coder",
                    agent=coder_name,
                    round_number=coder_record_round,
                    subject=str(
                        followup_bound_head
                        or updated_pr_context.metadata.head_sha
                        or "unknown"
                    ),
                    prior_items=tuple(unresolved_items),
                    raw_structured_coder_response=raw_structured_coder_response,
                    local_test_evidence=local_test_evidence,
                    risk_test_matrix_evidence=(
                        coder_response.marker_value.risk_test_matrix_evidence.to_payload()
                        if isinstance(coder_response.marker_value, StructuredCoderFollowup)
                        and coder_response.marker_value.risk_test_matrix_evidence is not None
                        else None
                    ),
                    evidence_stall=evidence_stall_snapshot,
                    risk_test_matrix_evidence_full_round=(
                        matrix_evidence_render_decision.anchor_round
                        if matrix_evidence_render_decision is not None
                        else None
                    ),
                    risk_test_matrix_diagnostics=(
                        tuple(
                            diagnostic.to_payload()
                            for diagnostic in coder_response.marker_value.risk_test_matrix_diagnostics
                        )
                        if isinstance(coder_response.marker_value, StructuredCoderFollowup)
                        else ()
                    ),
                    compact_prior_summaries=tuple(pr_compact_prior_summaries),
                    model_used=coder_response.model_used,
                    acquisition_outcome=coder_response.acquisition_outcome,
                    acquisition_returncode=coder_response.acquisition_returncode,
                    scheduler_contract=(scheduler_contract.as_dict() if selective_policy else None),
                    reviewer_board_amendment_digest=(pr_amendment_digest if selective_policy else None),
                    scheduler_previous_sha=(pr_metadata.head_sha if selective_policy else None),
                    scheduler_current_sha=(
                        str(updated_pr_context.metadata.head_sha or "unknown")
                        if selective_policy else None
                    ),
                    scheduler_obligation_digest=(
                        hashlib.sha256(
                            repr(_prior_item_ledger_signature(unresolved_items)).encode("utf-8")
                        ).hexdigest()[:16]
                        if selective_policy else None
                    ),
                    scheduler_selected_reviewers=(
                        tuple(sorted(selected_reviewer_names))
                        if selective_policy else ()
                    ),
                    scheduler_paused_reviewers=(
                        scheduler_decision.paused_reviewers
                        if selective_policy and scheduler_decision is not None else ()
                    ),
                    scheduler_reasons=(
                        (
                            (scheduler_decision.reason, classification.reason)
                            if scheduler_decision is not None else ()
                        )
                        + (
                            ("external/unrecorded head advance requires full board",)
                            if external_recovery_full_board else ()
                        )
                    ) if selective_policy else (),
                    scheduler_final_sweep=(final_sweep if selective_policy else None),
                    scheduler_force_full=(scheduler_recorded_force_full if selective_policy else None),
                    scheduler_force_full_source=(
                        scheduler_recorded_force_full_source if selective_policy else None
                    ),
                    scheduler_calls_avoided=(scheduler_calls_avoided if selective_policy else None),
                    scheduler_phase=(scheduler_decision.phase if selective_policy and scheduler_decision is not None else None),
                    scheduler_primary_reviewer=(scheduler_contract.primary_reviewer if selective_policy else None),
                    scheduler_approved_reviewers=(
                        tuple(sorted(unchanged_head_approvals)) if selective_policy else ()
                    ),
                    scheduler_active_owners=(scheduler_decision.active_owners if selective_policy and scheduler_decision is not None else ()),
                    scheduler_scope_digest=(hashlib.sha256(repr(classification.changed_paths).encode("utf-8")).hexdigest()[:16] if selective_policy else None),
                    qualification_checkpoint=qualification_checkpoint,
                    followup_dispatch_head=followup_dispatch_head,
                    step_back_entries=step_back_entries,
                    **_test_observation_degradation_fields(coder_response.marker_value),
                    **_architecture_metadata_fields(
                        config, result=coder_response.marker_value
                    ),
                )
            except AgentLoopError as coder_followup_error:
                # A coder that pushed and was then rejected leaves a head with no
                # coder or reviewer record; persist why, then surface the original
                # error unchanged.
                _record_coder_followup_rejection(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    error=coder_followup_error,
                    dispatch_round=round_number,
                    pre_turn_head=pre_turn_head,
                    ledger=pre_turn_ledger,
                    budget=dispatch_budget,
                    attempt=dispatch_attempt,
                    recovery_dispatch=dispatch_attempt > 1,
                    carried_reasons=dispatch_carried_reasons,
                )
                raise
            post_pr_comment(
                runner,
                config=config,
                pr_number=pr_number,
                body=_attach_round_metadata(
                    public_comment,
                    latest_coder_metadata,
                ),
            )
            if managed_ci_handoff is not None and managed_ci is not None and managed_ci.issue_created_pr:
                predecessor_head = pr_metadata.head_sha
                new_head = updated_pr_context.metadata.head_sha
                if predecessor_head and new_head and predecessor_head != new_head:
                    round_comment_ids = find_actor_round_metadata_comment_ids(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        actor_login=managed_ci_handoff.trusted_actor_login,
                        actor_id=managed_ci_handoff.trusted_actor_id,
                        predecessor_head=predecessor_head,
                        new_head=new_head,
                        round_number=round_number,
                        after_comment_id=(
                            managed_ci_handoff.authorization_comment_id or 0
                        ),
                    )
                    managed_ci_handoff = publish_issue_created_continuity_authorization(
                        runner,
                        config=config,
                        handoff=managed_ci_handoff,
                        predecessor_head=predecessor_head,
                        new_head=new_head,
                        round_comment_ids=round_comment_ids,
                    )
                    authenticated_managed_resume = AuthenticatedManagedResume(
                        origin="issue-created",
                        lifecycle=managed_ci_handoff.lifecycle,
                        issue_created_handoff=managed_ci_handoff,
                        override_nonce=managed_ci_handoff.override_nonce,
                    )
            pending_step_back_sweeps = _pr_step_back_carried_entries(
                step_back_entries,
                coder_round=coder_record_round,
                head_sha=updated_pr_context.metadata.head_sha,
            )
            previous_head = pr_metadata.head_sha
            unchanged_head_coder_turns = unchanged_head_tracker.observe(
                previous_head, updated_pr_context.metadata.head_sha
            )
            if unchanged_head_coder_turns >= MAX_UNCHANGED_HEAD_CODER_TURNS:
                # Re-reviewing an identical diff reaches the same verdict every
                # round; a finding a PR-mode coder turn cannot satisfy (such as
                # a required re-plan) must stop with a route, not consume the
                # round budget (#985).
                route = (
                    " If the blocking finding requires re-planning the child plan, post the "
                    "signed child-plan supersession record on child issue "
                    f"#{planning_child_binding.child_issue} and rerun "
                    f"`{_child_resume_hint(planning_child_binding.child_issue, EXECUTION_DISPOSITION_PLANNING)}`."
                    if planning_child_binding is not None
                    else ""
                )
                raise AgentLoopError(
                    unchanged_head_stop_message(
                        pr=pr_number,
                        coder_name=coder_name,
                        previous_head=previous_head,
                        turns=unchanged_head_coder_turns,
                        round_number=round_number,
                        evidence=(
                            latest_coder_metadata.local_test_evidence
                            if latest_coder_metadata is not None
                            else None
                        ),
                        route=route,
                    )
                )
            log(
                config,
                _coder_followup_head_log(
                    round_number,
                    coder_name,
                    previous_head,
                    updated_pr_context.metadata.head_sha,
                    unchanged_head_coder_turns,
                ),
            )
            pre_review_test_pending = True
            if external_recovery_full_board:
                # The recovered external head was not reviewed by this run;
                # keep the next candidate's full-board review requirement
                # across an interruption without latching normal broad paths.
                final_sweep_pending = True
            resumed_round = None
            prefetched_pr_context = updated_pr_context

        raise AgentLoopError(
            f"Reached max rounds ({config.max_rounds}) for PR #{pr_number}; human review required."
            + _sub_item_progress_block(
                unresolved_items,
                round_number=config.max_rounds,
                window=config.sub_item_stall_rounds,
            )
        )
    finally:
        cleanup_failure: AgentLoopError | None = None
        active_exception = sys.exc_info()[1]
        preserve_issue_created_suppression = _preserve_issue_created_managed_suppression(
            managed_ci,
            active_exception=active_exception,
        )
        if preserve_issue_created_suppression:
            log(
                config,
                f"PR #{pr_number}: retaining `{MANAGED_LABEL}` after interrupted managed run; "
                "resume the exact PR after correcting the reported condition",
            )
        try:
            if (
                managed_ci is not None
                and not managed_ci_qualified
                and not preserve_issue_created_suppression
            ):
                should_release = (
                    managed_ci.adopted_existing_pr
                    or managed_ci.origin in {"issue-created", "source-managed"}
                    or config.managed_ci
                )
                if should_release and not release_adopted_managed_ci(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    contract=managed_ci,
                    force=config.managed_ci,
                ):
                    message = f"PR #{pr_number}: unable to release invocation-owned managed-CI label"
                    diagnostic = managed_ci.release_diagnostic
                    if config.managed_ci:
                        cleanup_failure = AgentLoopError(
                            message + "; the PR remains suppressed and requires manual label removal."
                            + diagnostic
                        )
                    log(config, message + diagnostic)
        finally:
            _end_run_telemetry(runner, telemetry_token)
        if owned_usage_context:
            _persist_usage_summary(config, usage_context)
        if cleanup_failure is not None:
            if active_exception is not None:
                # Preserve the original failure while making cleanup failure
                # visible in its traceback. Usage accounting above must run
                # even when label cleanup also fails.
                active_exception.add_note(str(cleanup_failure))
            else:
                raise cleanup_failure
