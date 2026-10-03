"""High-level issue, task, and PR orchestration loops."""

from __future__ import annotations

import collections
import datetime
import dataclasses
import functools
import hashlib
import json
import re
import shlex
import sys
import time
import unicodedata
import urllib.parse
import zoneinfo
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass, replace as dataclasses_replace
from pathlib import Path
from typing import Literal, TypeVar

from .agents.base import AgentName, AgentResult
from .agents.antigravity import AntigravityAttemptState
from .agents.registry import agent_display_name, agent_signature, get_backend, run_agent_result
from .architecture_context import architecture_material, freeze_architecture_context
from .workdir_claims import claimed_run
from .config import (
    AgentLoopConfig,
    configured_model_for,
    ensure_agent_workdirs,
    github_bootstrap_cwd,
    resolve_base_branch,
    reviewers,
    resolve_invocation,
    sync_coder_base_before_implementation,
    sync_coder_pr_before_validation,
    sync_reviewer_pr_before_review,
)
from .board_amendment import (
    ReviewerBoardAmendment,
    amend_contract,
    ContractLineage,
    added_in_config,
    amendment_audit_already_posted,
    amendment_summary_line,
    apply_board_amendment_to_ledger,
    collect_reviewer_board_amendments,
    format_reviewer_board_amendment_comment,
    missing_from_config,
    predates_restoration,
    reject_misplaced_pr_amendments,
    render_amendment_audit_comment,
    require_amendment_activation,
    resolve_contract_lineage,
    restoration_rounds_from_comments,
    stale_unread_amendment_locator,
)
from .decomposition import (
    _decode_json_payload,
    EXECUTION_DECISION_MARKER_RE,
    issue_has_execution_decision,
    CreatedPhaseIssue,
    PlanDecomposition,
    RecordedPhase,
    approved_plan_hash,
    create_decomposition_child_issues,
    find_existing_decomposition,
    find_existing_one_shot_impl_handoff,
    find_one_shot_impl_handoffs,
    find_latest_one_shot_impl_handoff,
    find_existing_phase_implementation_handoff,
    parse_plan_decomposition,
    post_decomposition_parent_summary,
    post_one_shot_impl_handoff_comment,
    post_phase_implementation_handoff_comment,
    adapt_typed_child_stages,
    normalize_execution_recommendation,
    validate_risk_matrix_ownership,
    validate_separately_planned_child_matrix,
    AuthorizedReplanLineage,
    ChildPlanRebindRecord,
    authorized_replan_lineage,
    collect_child_plan_supersessions,
    find_child_plan_rebind_records,
    format_child_plan_rebind_section,
    format_child_plan_supersession_comment,
    InheritedMatrixBinding,
    InheritedRowDifference,
    inherited_matrix_reviewed_deltas,
    risk_matrix_row_ids_for_owner,
    recover_execution_recommendation,
    ExecutionDecision,
    EXECUTION_TOPOLOGY_SOURCE,
    find_existing_execution_decision,
    live_execution_decisions,
    reject_legacy_topology_collision,
    post_execution_decision,
    find_existing_topology_checkpoint,
    find_topology_checkpoints_for_parent,
    find_decompositions_for_parent,
    find_phase_implementation_handoffs_for_parent,
    PHASE_IDENTITY_MARKER_RE,
    phase_identity,
    post_topology_checkpoint,
    TopologyCheckpoint,
    ChildDispositionOverride,
    PhaseImplementationHandoffMetadata,
    collect_child_disposition_overrides,
    phase_direct_readiness_problems,
    reconcile_handoff_disposition,
    retained_parent_scope_matches,
)
from .protocol import EXECUTION_DISPOSITION_DIRECT, EXECUTION_DISPOSITION_PLANNING
from .child_topology import NeedsHumanDecision, NestedTopologyDecision, parent_child_search_queries
from .errors import (
    AgentInvocationError,
    CheckoutVerificationError,
    AgentLoopError,
    DeterministicPlanValidationExhaustion,
    FreshContractIntegrityError,
    HumanDecisionRequiredError,
    IssueImplementationConflictError,
    PreservedUnsatisfiedResponse,
    NonRepairableEvidenceRejection,
    QuotaResetExceededError,
    ReviewSubstanceIntegrityError,
    SemanticPatchPayloadRejection,
    MissingPriorItemDispositionError,
    SemanticPatchUnknownPriorItemDispositionError,
    UnknownPriorItemDispositionError,
)
from .expected_closure import (
    ExpectedClosingContract,
    reject_parent_from_contract,
    resolve_direct_contract,
    resolve_issue_contract,
)
from .github import (
    _REST_ISSUE_COMMENT_PAGE_SIZE,
    _classify_check_status,
    CiWatchOutcome,
    strip_bot_login_suffix,
    IssueContext,
    PullRequestMetadata,
    PullRequestChecks,
    PullRequestMergeability,
    PullRequestReviewContext,
    HumanReviewRequirement,
    board_protection_is_reliable,
    protection_awaits_readiness,
    deduplicate_human_requirements,
    get_pr_head_sha,
    get_issue_context,
    get_pr_mergeability,
    parse_linked_issue_numbers,
    parse_strong_issue_reference_evidence,
    get_pr_checks,
    get_pr_review_context,
    get_pr_state,
    merge_pr,
    post_issue_comment,
    post_pr_comment,
    post_trusted_pr_contract_record,
    post_trusted_issue_comment,
    post_trusted_pr_comment,
    post_verified_trusted_issue_round_comment,
    post_verified_trusted_issue_protocol_comment,
    note_host_footer_observed,
    reset_authenticated_github_actor,
    reset_host_footer_log_latch,
    read_rest_issue_comments,
    resolve_authenticated_github_actor,
    reject_forged_protocol_markers,
    search_issues,
    validate_open_issue,
    validate_open_pr,
    validate_pr_body_does_not_close_issue,
    validate_pr_expected_closing_issues,
    validate_pr_references_issue,
    validate_pull_request_provenance,
    watch_pr_checks,
)
from .issue_pr_handoff import (
    AGENT_ISSUE_PR_HANDOFF_RE,
    IssuePrHandoffMetadata,
    decode_issue_pr_handoff_record,
    find_latest_issue_pr_handoff,
    authenticate_canonical_issue_pr,
    format_issue_pr_handoff_comment,
    post_issue_pr_handoff_comment,
    resolve_issue_pr_handoff_lineage,
    require_pr_metadata_for_handoff,
    resolve_canonical_pr_for_issue,
)
from .issue_pr_provenance import IssuePrProvenanceScope
from .phase_progress import (
    STATUS_HUMAN_PENDING,
    PhaseProgress,
    StagedTopologyOutcome,
    record_staged_completion,
    render_phase_status_line,
    resolve_staged_phase_progress,
    select_current_phase,
)
from .pr_contract import (
    PR_EXPECTED_CLOSING_MARKER_RE,
    PrExpectedClosingContract,
    find_latest_pr_contract,
    format_pr_contract_comment,
    make_pr_contract,
    render_pr_contract_marker,
)
from .split_materialization import (
    DISCUSS_SPLIT_MARKER_RE,
    SPLIT_CHILD_MARKER_RE,
    SPLIT_STAGE_HANDOFF_MARKER_RE,
    UNFILED_SPLIT_WARNING_MARKER_RE,
    MaterializedSplitChild,
    SplitStageProposal,
    dedupe_split_stage_proposals,
    find_existing_split_materialization,
    find_existing_split_stage_handoff,
    has_unfiled_split_warning,
    materialize_split_proposals,
    post_split_stage_handoff_comment,
    post_unfiled_split_warning,
    resolve_selected_stage_child,
    split_stage_proposal_from_deferred_stage,
    split_stage_proposal_from_text,
)
from . import tool_provenance
from .logging import log, new_run_id, run_usage_summary_path
from .evidence_reconciliation import (
    bounded_reconciliation_candidates,
    collect_evidence_observations,
    reconcile_evidence,
)
from .memory import AgentMemoryContext, prepare_agent_memory
from .managed_ci import (
    AuthenticatedManagedResume,
    AuthenticatedIssueCreatedHandoff,
    FINAL_CONTEXT,
    MANAGED_LABEL,
    ManagedCiContract,
    ManagedCiOutcome,
    OrdinaryRecoveryCapability,
    activate_managed_ci,
    managed_ci_attachment_matches_head,
    release_stale_managed_ci_attachment,
    authorize_fresh_issue_created_resume,
    authenticate_source_managed_resume,
    authenticate_issue_created_handoff,
    dispatch_final_qualification,
    intermediate_managed_checks,
    managed_label_present,
    preflight_managed_ci_creation,
    find_actor_round_metadata_comment_ids,
    publish_issue_created_continuity_authorization,
    publish_issue_created_authorization,
    publish_manual_v2_qualification,
    prepare_v2_merge,
    publish_round_readiness,
    refresh_ordinary_recovery_capability,
    release_adopted_managed_ci,
    release_retained_managed_label,
    revalidate_adopted_managed_ci,
    revalidate_issue_created_handoff,
    recover_issue_created_handoff,
    render_managed_ci_resume_command,
    validate_ordinary_recovery_capability,
    verify_managed_pr_plan_binding,
    wait_for_ordinary_recovery,
    wait_for_final_qualification,
    waivable_protection_states,
    waiver_flags_for_protection,
)
from .managed_pr import recover_managed_pr_origin, validate_managed_pr_body
from .migrations import validate_pr_migration_topology
from .prompts import (
    CompactPlanTailContext,
    CompactPriorContext,
    CompactPrReviewTailContext,
    build_discuss_agenda_prompt,
    build_discuss_round_synthesis_prompt,
    build_discuss_final_synthesis_prompt,
    build_discuss_final_analysis_prompt,
    build_discuss_evidence_reconciliation_prompt,
    build_discuss_answer_confirmation_prompt,
    build_discuss_semantic_comparison_prompt,
    build_discuss_review_prompt,
    build_completion_recovery_prompt,
    build_followup_prompt,
    build_issue_implementation_prompt,
    INHERITED_COVERAGE_DELTA_MAX_BYTES,
    INHERITED_OBLIGATIONS_ENFORCEABLE_MAX_BYTES,
    build_issue_plan_prompt,
    inherited_coverage_delta_size,
    inherited_obligations_enforceable_size,
    build_issue_prompt,
    build_plan_decomposition_prompt,
    build_plan_review_prompt,
    build_plan_revision_prompt,
    build_merge_conflict_prompt,
    build_review_prompt,
    SupersededPrepanelReview,
    build_same_pr_followup_prompt,
    build_task_clarification_prompt,
    build_task_prompt,
    format_agent_list,
    render_coder_human_requirements_prompt_context,
)
from .protocol import (
    AgentUnavailable,
    ApprovedFollowup,
    ApprovedFollowups,
    DISCUSS_FAILED_OUTCOME,
    DISCUSS_RESEARCH_TARGET_VALUES,
    DISCUSS_SYNTHESIS_MAX_ENTRIES,
    DISCUSS_SYNTHESIS_MAX_TEXT_BYTES,
    ChildStage,
    DeferredStage,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
    ParsedDiscussAgenda,
    ParsedDiscussEvidenceReconciliation,
    ParsedDiscussAnswer,
    ParsedDiscussFinalSynthesis,
    ParsedDiscussRoundSynthesis,
    DiscussSynthesisConsensus,
    DiscussSynthesisDisagreement,
    DiscussSynthesisPosition,
    DiscussSynthesisResponseReference,
    DiscussUnresolvedItem,
    ParsedDiscussSemanticComparison,
    ParsedDiscussResponse,
    ParsedDiscussReview,
    failed_discuss_review_category,
    failed_discuss_review_placeholder,
    failed_discuss_answer_placeholder,
    is_failed_discuss_response,
    ParsedPlanReview,
    ExecutionStrategyRecommendation,
    PlanReviewItems,
    ParsedReview,
    PUBLIC_RESPONSE_MARKER,
    ReviewItemDisposition,
    StructuredCoderFollowup,
    StructuredIssueImplementation,
    DerivedRiskEvidenceResult,
    PostAuthClaimDiagnostic,
    SemanticRiskCoverageClaims,
    DEGRADED_ROW_CLAIM_DIAGNOSTIC,
    UNAPPROVED_ROW_CLAIM_DIAGNOSTIC,
    StructuredPlanState,
    StructuredPlanRevision,
    PlanRevisionPatch,
    parse_plan_revision_patch,
    StructuredTaskResult,
    UnresolvedReviewItem,
    CI_MACHINE_OBLIGATION_KINDS,
    MACHINE_OBLIGATION_KINDS,
    human_requirements_resolved,
    is_clarification_request,
    parse_human_requirements_acknowledgement,
    parse_agent_state,
    parse_agent_unavailable,
    parse_plan_review,
    parse_plan_review_items,
    parse_plan_state,
    parse_structured_plan_review,
    parse_structured_pr_review,
    parse_pr_number,
    review_freeform_summary_text,
    normalize_response_file_structured_text,
    validate_human_requirements_acknowledgement,
    validate_human_requirement_dispositions,
    validate_structured_coder_followup,
    validate_structured_human_requirements_acknowledgement,
    validate_structured_issue_implementation,
    parse_historical_structured_coder_followup,
    parse_historical_structured_issue_implementation,
    validate_structured_plan_state,
    validate_structured_plan_revision,
    validate_structured_plan_revision_patch,
    validate_risk_test_matrix_revision,
    validate_structured_task_result,
    derive_risk_test_matrix_evidence,
    semantic_risk_claim_schema_text,
    risk_test_matrix_identity,
    validate_structured_discuss_agenda,
    parse_structured_discuss_final_synthesis,
    validate_structured_discuss_final_synthesis,
    validate_structured_discuss_round_synthesis,
    validate_structured_discuss_review,
    validate_structured_discuss_answer,
    validate_structured_discuss_answer_confirmation,
    validate_structured_discuss_evidence_reconciliation,
    validate_structured_discuss_semantic_comparison,
    serialize_discuss_round_synthesis,
    serialize_discuss_final_synthesis,
    parse_architecture_impact,
    sanitize_architecture_impact,
    ARCHITECTURE_IMPACT_DECLARED_STATUSES,
    ARCHITECTURE_IMPACT_UNDETERMINED,
    ArchitectureImpact,
    ParseDegradation,
)
from .protocol import parse_review
from .repair import (
    CandidateDecision,
    RepairAttemptResult,
    attempt_envelope_normalization,
    attempt_semantic_patch_disposition_normalization,
    attempt_repair,
    execute_repair,
    strip_unknown_prior_item_dispositions,
    unknown_dispositions_are_resolved_history,
    require_recoverable_fresh_execution_contract,
    require_recoverable_fresh_risk_test_matrix_contract,
    require_recoverable_review_substance,
)
from .repair_preservation import (
    canonicalize_architecture_near_miss_text,
    normalize_architecture_impact_near_miss,
    recover_payload,
    require_recoverable_semantic_patch,
    require_repair_architecture_impact_absent,
    validate_repair_preservation,
)
from .runner import Runner

from .local_test_evidence import (
    reconcile_test_observations,
    stable_tracked_tree_snapshot,
)
from .salvage import (
    SalvageArtifacts,
    SalvageContext,
    capture_salvage_artifacts,
    latest_salvage_context,
    post_salvage_comment,
)
from .transient import (
    NON_RETRYABLE_AGENT_OUTPUT_RE,
    TRANSIENT_AGENT_OUTPUT_RE,
    classify_antigravity_capacity,
    is_transient_agent_output,
    looks_like_backgrounded_completion,
)
from .usage import RunUsageContext, UsageMetadata, estimate_usage
from .worker_telemetry import append_record, run_record, telemetry_log_path
from .workdirs import active_workdir
from .workdir_guard import (
    read_workdir_head,
    validate_assigned_head_advanced,
    validate_checkout_inspected_evidence,
    validate_response_tests_within_workdir,
    command_is_admissible_evidence,
    command_targets_outside_workdir,
    partition_reported_tests_by_workdir,
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
    _unreadable_protection_stop_message,
    run_optional_tests,
    run_pre_review_tests,
)
from .ci_health import (
    CiInfrastructureStall,
    StalledCheck,
    is_canonical_stall_only_text,
    is_canonical_pending_only_text,
    is_wholly_infrastructure_blocked,
)
from .comment_rendering import (
    _extract_plan_human_requirements_block,
    DEFERRED_STAGES_MARKER_RE,
    render_decomposition_degradation_comment,
    render_refused_decomposition_comment,
    render_plan_phase_advance,
    render_plan_scheduling_audit,
    render_sub_item_progress_comment,
    sub_item_progress_digest,
    sub_item_progress_record_keys,
    EXECUTION_RECOMMENDATION_MARKER_RE,
    RISK_TEST_MATRIX_MARKER_RE,
    ITEM_SUMMARY_LIMIT,
    _append_before_trailing_metadata,
    _format_unresolved_item_label,
    _extract_plan_revision_human_requirements_block,
    _item_label_status,
    _normalize_item_summary,
    _public_reviewer_name,
    _render_disposition_status,
    _render_prior_dispositions_section,
    _render_public_review_comment,
    _replace_structured_section,
    _review_freeform_summary_text,
    add_coder_followup_head_unchanged_notice,
    decode_deferred_stages_marker,
    decode_execution_recommendation_marker,
    decode_risk_test_matrix_marker,
    normalize_freeform_signature,
    render_discuss_round_summary_comment,
    render_public_agent_comment,
    resolve_matrix_evidence_render,
    render_agent_unavailable_comment,
    render_canonical_plan_revision,
    render_canonical_plan_state,
    render_canonical_plan_steps,
    PLAN_EXPECTED_CLOSING_MARKER_RE,
    decode_expected_closing_issue_declaration,
    _render_discuss_agenda_lines,
)
from .followups import (
    APPROVED_FOLLOWUP_MARKER_RE,
    PLAN_APPROVED_FOLLOWUP_MARKER_RE,
    approved_plan_hashes_for_issue,
    GroupedApprovedFollowup,
    MAX_APPROVED_FOLLOWUP_ISSUES,
    _append_approved_followups_marker,
    _approved_followup_from_unresolved_item,
    _approved_followups_marker,
    _dedupe_approved_followups,
    _followup_heading_key,
    _followup_issue_body,
    _followup_issue_title,
    _format_approved_followup_summary,
    _format_same_pr_followups,
    _has_approved_followups_marker,
    _plan_followup_source_from_unresolved_item,
    _publish_plan_approved_followups,
    _normalize_followup_key,
    _publish_approved_followups,
    FollowupSourceContext,
)
from .round_state import (
    ApprovedPlanContext,
    EVIDENCE_FREEZE_PHASE,
    EVIDENCE_RELEASE_PHASE,
    EVIDENCE_RESPONSE_PHASE,
    EvidenceFreezeRecord,
    EvidenceReleaseRecord,
    QualificationCheckpoint,
    PostedRoundMetadata,
    _is_followup_dispatch_head,
    followup_head_unchanged_sha,
    rebuild_resumed_coder_carrier,
    PostedRoundRecord,
    ROUND_RESUME_MARKER_RE,
    ResumedRoundSelection,
    ResumedReviewRound,
    _attach_round_metadata,
    _decode_discuss_vote,
    _decode_round_metadata,
    _deserialize_disposition,
    _deserialize_unresolved_item,
    _encode_round_metadata,
    _canonically_resolved_history_item_ids,
    _live_round_resolved_item_ids,
    _extract_round_metadata_records,
    _latest_pr_approved_reviews_for_head,
    _max_unresolved_item_number_from_records,
    _plan_subject,
    _prior_item_ledger_signature,
    _resume_discuss_round,
    _resume_plan_round,
    make_approved_plan_context,
    scope_approved_plan_matrix,
    recover_approved_plan_context,
    RequirementsContext,
    _resume_pr_round,
    _select_current_round_records,
    _serialize_disposition,
    _serialize_unresolved_item,
    _strip_round_metadata,
    PlanValidationDiagnosticPayload,
    PlanValidationDiagnosticTransport,
    encode_plan_validation_diagnostic_body,
    has_plan_validation_diagnostic_marker,
    recover_plan_validation_diagnostic,
    sanitize_plan_validation_diagnostic,
)
from .round_transport import (
    decode_mapping,
    is_round_transport_sidecar,
    prepare_round_comment,
    round_comment_fits,
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
    STRUCTURAL_SIGNALS,
    PlanGrowthApprovalVerdict,
    PlanGrowthAssessment,
    PlanGrowthThresholds,
    assess_plan_growth,
    check_growth_justification,
    check_scope_ledger_preservation,
    growth_justification_violation,
    handoff_growth_verdict_violation,
    plan_growth_approval_verdict,
    plan_growth_gate_enforced,
    plan_justification,
    plan_strategy,
    render_growth_measurements,
    render_growth_notice,
)
from .plan_review_scheduling import (
    PLAN_HISTORY_CONTRADICTORY_KEY,
    PLAN_HISTORY_INTACT,
    PLAN_HISTORY_TRANSPORT_FAILURE,
    PlanCandidateKey,
    PlanCrossCuttingContracts,
    PlanPrePanelSafetyError,
    PlanRevisionDescriptor,
    PlanReviewSchedulingContract,
    PlanSchedulerSnapshot,
    PlanSchedulingDecision,
    classify_plan_history,
    classify_plan_transition,
    execution_recommendation_contract_identity,
    execution_recommendation_contract_projection,
    make_plan_contract,
    plan_history_fallback_reason,
    plan_policy_capabilities,
    plan_undecodable_history_message,
    select_plan_reviewers,
    surfaced_requirement_id_digest,
)
from .protocol_markers import (
    TrustedBody,
    is_complete_marker_occurrence,
    sanitize_historical_text,
    scan_reserved_markers,
)
from .review_scheduling import (
    PANEL_OPENED_PHASES,
    GitChange,
    PrePanelSafetyError,
    ReviewObligation,
    ReviewSchedulingContract,
    SchedulerSnapshot,
    TransitionClassification,
    classify_transition,
    make_contract,
    policy_capabilities,
    pre_panel_safety_message,
    select_reviewers,
    undecodable_history_message,
)
from .partial_round_recovery import (
    PartialRoundRecovery,
    RecoveryTarget,
    RecoveryVisibilityContext,
    compute_partial_round_recovery,
)
from .round_visibility import latest_round_checkpoint_index, visible_peer_names
from .review_spool import ReviewRoundSpool, review_spool_root
from .unresolved_items import (
    ALL_RESOLVED_PROSE_RE,
    ClearedItemProgress,
    newly_stalled_items,
    render_sub_item_progress_summary,
    sub_item_progress,
    CODER_DISPUTE_NOTE_PREFIX,
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    MERGE_CONFLICT_ITEM_ID,
    _apply_dispute_evidence,
    _apply_unresolved_item_dispositions,
    _collect_prior_compact_summaries,
    bound_compact_prior_summaries,
    _clear_human_requirements_ack_item,
    _clear_merge_conflict_item,
    _format_same_pr_unresolved_items,
    format_coder_followup_context,
    _maybe_fill_resolved_dispositions_from_prose,
    _next_unresolved_item,
    _advance_machine_obligations_for_head,
    _clear_machine_obligations,
    _is_machine_obligation,
    _machine_obligation_is_ci,
    _machine_obligation_is_revalidation_candidate,
    _machine_obligation_requires_repair,
    _set_machine_obligation_lifecycle,
    _upsert_machine_obligation,
    _normalize_disposition_section_prose,
    _reconcile_merge_conflict_item,
    _record_prior_item_disposition,
    _reconcile_human_requirements_ack_item,
    _raise_if_maintained_disputed_items,
    select_coder_followup_items,
    coder_followup_is_ci_repair,
    _frozen_evidence_obligations,
    _is_evidence_obligation,
    _pending_evidence_obligations,
    _upsert_evidence_obligation,
    freeze_evidence_obligations,
    release_evidence_freeze,
    _upsert_human_requirements_ack_item,
    _validate_coder_followup_response,
    _validate_plan_review_response,
    _validate_review_response,
    _validate_structured_coder_followup_items,
)
from .agent_failure import (
    NEAR_MISS_AGENT_MARKER_RE,
    PUBLIC_RESPONSE_ARTIFACT_PREFIX_RE,
    STRUCTURED_PUBLIC_RESPONSE_KINDS,
    PLAN_REVISION_FOOTER_RE,
    STRUCTURED_FENCE_RE,
    PUBLIC_RESPONSE_TRANSIENT_DIAGNOSTIC_RE,
    UNSUPPORTED_MODEL_DIRECT_RE,
    INVALID_REQUEST_RE,
    MODEL_SUPPORT_OR_AVAILABILITY_RE,
    MODEL_TOKEN_RE,
    MODEL_PARENTHESES_SUFFIX_RE,
    FAILURE_CLASSIFICATION_TEXT_LIMIT,
    ISSUE_IMPLEMENTATION_SALVAGE_SCOPE,
    APPROVED_PLAN_IMPLEMENTATION_SALVAGE_SCOPE,
    TASK_IMPLEMENTATION_SALVAGE_SCOPE,
    PR_FOLLOWUP_SALVAGE_SCOPE,
    LONG_RESET_THRESHOLD_SECONDS,
    _PROVIDER_DEFINITIVE_FAILURE_CATEGORIES,
    _QUOTA_RATE_LIMIT_RE,
    _RETRY_AFTER_SECONDS_RE,
    _TRY_AGAIN_IN_RE,
    _RESET_IN_RE,
    _ISO_TIMESTAMP_RE,
    _ABSOLUTE_RESET_TIME_RE,
    ValidatedAgentResponse,
    _response_identity_fields,
    _metadata_identity_fields,
    _AgentUnavailableResponse,
    _capture_agent_invocation,
    _UnsupportedModelDiagnostic,
    _parse_absolute_reset_seconds,
    _parse_rate_limit_reset_seconds,
    _format_reset_duration,
    _format_reset_at_utc,
    _agent_log_context,
    _is_transient_agent_output,
    _decode_public_response_json_prefix,
    _recognized_structured_public_response_kind,
    _is_error_shaped_json_payload,
    _bounded_failure_classification_text,
    _json_error_payload_text,
    _first_json_error_payload_text,
    _has_transient_availability_signal,
    _looks_like_unsupported_model_text,
    _looks_like_unsupported_effort_text,
    _unsupported_model_classification_text,
    _is_transient_public_response,
    _is_retryable_marker_near_miss,
    _STRUCTURED_SCHEMA_REJECTION_RE,
    _SEMANTIC_EVIDENCE_REJECTION_CLASSIFICATION,
    _is_structured_schema_rejection,
    _failure_category,
    _response_file_structured_status,
    _candidate_source_texts,
    _unfence_structured_json_blocks,
    _neutralize_untrusted_markers,
    _structured_response_candidates,
    _recover_valid_structured_candidate,
    _HumanRequirementsRecoveryContext,
    _split_reconstructable_plan_revision_response,
    _plan_revision_missing_human_acknowledgement,
    _human_requirements_acknowledgement_blocks,
    _recover_plan_revision_human_requirements_acknowledgement,
    _retry_delay,
    _clean_diagnostic_fragment,
    _model_flag_value,
    _parse_model_from_provider_text,
    _extract_provider_auth_context,
    _extract_unsupported_model_reason,
    _configured_requested_model,
    _resolve_requested_model,
    _known_unsupported_model_fallback,
    _build_unsupported_model_diagnostic,
    _unsupported_model_suggestion,
    _executable_replacement_failure_detail,
    _failure_suggestion,
    _format_unsupported_model_agent_response_error,
    _format_invalid_agent_response_error,
    _compact_failure_reason,
    _PatchSalvageDiagnostic,
    _FailedRunDiagnostics,
    _best_effort_failed_run_status,
    _status_is_untracked_only,
    _capture_failed_run_salvage_diagnostic,
    _operation_description_from_context,
    _failed_response_recording_reason,
    _public_response_file_diagnostic,
    _failed_run_diagnostics,
    _agent_failure_classification_text,
)
from .architecture_contract import (
    _VALIDATION_TEST_TURN_CONTEXT,
    _freeze_prompt_architecture,
    _architecture_metadata_fields,
    _test_observation_degradation_fields,
    _latest_pr_approval_architecture_identity,
    _latest_pr_architecture_observation,
    _revalidate_pr_architecture_identity,
    _TerminalNoPrImplementation,
    _TerminalIssueImplementationConflict,
    _ARCHITECTURE_CONTRACT_CARRIERS,
    _ARCHITECTURE_RECORD_CARRIERS,
    _TERMINAL_PAYLOAD_WRAPPERS,
    _carrier_payload,
    _with_carrier_payload,
    _is_contract_free_result,
    _unwrap_architecture_result,
    _architecture_result_fields,
    architecture_impact_contract_unsatisfied,
    _architecture_contract_diagnostic,
    _attach_architecture_degradations,
    _merge_degradation_records,
    _architecture_mode_validators,
    _canonical_comparison_projection,
    _AcceptedCandidate,
    _AcceptedTextCanonicalizationError,
    _with_validation_context,
    _accept_candidate,
    _accepted_validated_response,
    _ARCHITECTURE_CONTRACT_RETRY_SECTION_CHARS,
    _architecture_contract_retry_prompt,
    _OMISSION_REASK_MAX_IDS,
    _prior_disposition_omission_reask_prompt,
    _surface_decomposition_degradations,
    _architecture_impact_from_metadata,
    _resumed_review_architecture,
    _acknowledgement_repair_forbids_assessment,
    _pin_acknowledgement_repair,
    _surface_refused_decomposition,
    _ArchitectureImpactContractUnsatisfied,
)
from .validated_agent import (
    _new_usage_context,
    _begin_run_telemetry,
    _end_run_telemetry,
    _resolve_usage_metadata,
    _persist_usage_summary,
    _ORIGINAL_ATTEMPT_REPAIR,
    _run_structured_repair,
    CompletionRecoveryPolicy,
    _CompletionRecoveryOutcome,
    _post_completion_recovery_terminal_comment,
    _synthesized_completion_recovery_unavailable,
    _attempt_claude_completion_recovery,
    _refused_contract_attempt,
    _log_repair_attempts,
    _capture_terminal_plan_repair_rejection,
    _semantic_patch_payload_rejection,
    _history_strip_reason,
    _history_strip_allowed,
    _run_validated_agent,
)
from .response_validation import (
    ResolvedExecution,
    _validate_issue_implementation_contract,
    _validate_issue_implementation_response,
    _current_test_turn_observations,
    _admissible_evidence_observations,
    _safe_execution_handle_catalog,
    _post_auth_correction_prompt,
    _parse_fresh_correction_claims,
    _NON_ACTIONABLE_RISK_DIAGNOSTICS,
    _derive_authenticated_risk_evidence_for_coder,
    _post_no_pr_implementation_terminal_comment,
    _post_structured_issue_implementation_terminal_comment,
    _require_pr_number,
    _require_pr_number_or_clarification,
    _require_task_implementation_result,
    _require_plan_state_or_clarification,
    _validate_response_with_human_requirements,
    _current_plan_has_complete_human_requirement_dispositions,
    _effective_plan_revision_patch_dispositions,
    _merge_human_requirements,
    _build_requirements_context,
    _describe_requirement_set_change,
    _surfaced_reviewer_requirement_ids,
    _reviewer_requirement_identity_ids,
    _reviewer_requirement_coverage_matches,
    _resumed_pr_reviewer_matches_requirements,
    _validate_plan_revision_response,
    _validate_plan_revision_patch_response,
    _validate_plan_revision_patch_payload,
    _check_plan_revision_patch_ledger,
    _check_plan_revision_patch_human_requirement_dispositions,
    _drop_repeated_carried_future_followups,
    _drop_repeated_carried_plan_future_followups,
    _validate_tests_with_post_pr_context,
    _validate_response_tests_with_post_pr_context,
    _StructuredTestReport,
    _degrade_out_of_checkout_tests,
    _validate_structured_response_tests_with_post_pr_context,
    _validate_structured_response_observations_with_post_pr_context,
)
from .panel_evidence import (
    PrPanelEvidence,
    _derive_pr_panel_evidence,
    _pr_record_is_panel_qualified,
    _describe_superseded_prepanel_review,
    _superseded_prepanel_review,
    _superseded_prepanel_plan_review,
    _scheduler_recorded_force_full,
    PLAN_PANEL_OPENED_PHASES,
    PlanPanelEvidence,
    _plan_identity_digest,
    _plan_candidate_key_for,
    _outstanding_plan_phase,
    _board_amendment_template,
    _PR_AMENDMENT_RERUN_CLAUSE,
    _board_amendment_route_clause,
    _stale_amendment_repost_clause,
    _append_board_amendment_note,
    _keep_reused_amendment_round_reviews,
    _plan_contract_or_none,
    _plan_scheduler_contract_from_metadata,
    _plan_key_from_payload,
    _derive_plan_panel_evidence,
    _carried_plan_approvals,
    _StagedPlanHistory,
    _classify_staged_plan_history,
    PlanPrimaryStreak,
    plan_issue_text_digest,
    plan_primary_blocking_streak,
    plan_primary_blocking_streak_detail,
    plan_primary_stall_message,
    _planner_candidate_rounds,
    _plan_growth_candidate_count,
    _plan_growth_gate_violation,
    _require_plan_growth_compliance,
    _plan_growth_verdict_for_hash,
    _require_handoff_plan_growth_verdict,
    _require_complete_canonical_plan_approval,
    _plan_cross_cutting_contracts,
    _resumed_plan_transition_inputs,
    _plan_revision_descriptor,
)
from .review_rounds import (
    _is_pending_ci_only_review,
    _normalize_approval_gated_managed_ci_review,
    _coder_infrastructure_stall_notice,
    _BOILERPLATE_REVIEW_SUMMARIES,
    _is_infrastructure_ci_only_review,
    _should_record_new_blocking_item,
    _INCOMPLETE_REVIEW_RE,
    _is_incomplete_pr_review,
    _describe_pr_review_outcome,
    _format_incomplete_pr_review_comment,
    _unavailable_reviewer_remedy,
    _unavailable_reviewer_amendment_advisory,
    _is_incomplete_plan_review,
    _describe_plan_review_outcome,
    _round_ledger_may_be_incomplete,
    _round_resolved_history_item_ids,
    _post_round_resolved_history_item_ids,
    _ReviewerTurnResult,
    _review_round_spool,
    _replay_spooled_review,
    _spooled_response_fields,
    _incomplete_plan_review_error,
    _incomplete_pr_review_error,
    PartialReviewRoundError,
    _replay_unavailable_cause,
    _format_recovery_target,
    _render_recovery,
    _partial_round_refusal,
    _replay_spooled_failure,
    _refuse_partial_round_before_sequential_turns,
    _same_round_replay_or_invoke,
    _launch_reviewer_turns,
    _ensure_parallel_reviewer_workdirs,
    _ensure_parallel_discuss_workdirs,
)
from .discuss_loop import (
    DISCUSS_CONSENSUS_MARKER_RE,
    _is_bot_authored_discuss_comment,
    _discuss_subject,
    _merge_discuss_split_proposals,
    _detect_discuss_consensus,
    _normalize_discuss_answer,
    _detect_discuss_answer_consensus,
    _aggregate_discuss_unresolved_items,
    _discuss_has_material_items,
    _final_discuss_answer_item_outcome,
    _handle_discuss_split_outcome,
    _recover_final_discuss_split_proposals,
    _DISCUSS_AGENDA_SUPPORT_STOP_WORDS,
    _normalize_discuss_agenda_phrase,
    _tokenize_discuss_agenda_support,
    _DiscussAgendaSupportCorpus,
    _build_discuss_agenda_support_corpus,
    _discuss_agenda_text_has_support,
    _validate_discuss_analyzer_agenda_fidelity,
    _disc_synthesis_vote_text,
    _disc_synthesis_text_supported_by_vote,
    _validate_discuss_round_synthesis_fidelity,
    _mechanical_discuss_final_classification,
    _validate_discuss_final_synthesis_fidelity,
    _run_discuss_analyzer,
    _validate_discuss_final_analyzer_fidelity,
    _run_discuss_final_analyzer,
    _run_discuss_evidence_reconciler,
    _DiscussDebaterTurnResult,
    _validate_structured_discuss_vote_with_evidence,
    _run_discuss_debater_turn,
    _run_discuss_semantic_finalization,
    _adapt_discuss_final_synthesis,
    _safe_discuss_synthesis_serialization,
    _post_discuss_debater_comment,
    _validate_discuss_evidence_update_targets,
    _run_discuss_loop,
    run_discuss_loop,
)
from .execution_policy import (
    _DEFERRED_STAGES_SECTION_RE,
    _NEXT_HEADING_RE,
    _PLAN_NARROWING_PHRASE_RE,
    _extract_current_deferred_stages,
    _extract_current_expected_closing_issue_ids,
    _extract_current_child_stages,
    _log_typed_plan_stage_dispositions,
    _prior_discuss_split_proposals,
    _plan_text_suggests_narrowing,
    _plan_first_line,
    _current_execution_recommendation,
    _normalize_requested_execution_policy,
    _resolve_execution_policy,
    CHILD_ROUTE_HUMAN,
    ChildExecutionRoute,
    resolve_child_execution_route,
    _fresh_phase_marker_payload,
    _FreshChildProvenance,
    _resolve_fresh_child_provenance,
    _child_resume_hint,
    _projection_may_carry_planning_record,
    _refuse_plain_mode_over_planning,
    _post_child_planning_handoff,
    _print_staged_phase_progress,
    _print_parent_obligation,
    _print_staged_terminal_report,
    _recorded_staged_outcome_for_child,
    _record_staged_parent_completion_after_merge,
    _print_dry_run_execution_preview,
    _print_execution_resolution_summary,
    _persist_execution_decision_if_needed,
    _REALIZATION_LISTING_LIMIT,
    _strict_gh_listing,
    _execution_decision_realization_evidence,
    _retirable_execution_decision_hashes,
    _preflight_fresh_staged_topology,
    _preflight_fresh_split_topology,
    _approved_one_shot_split_keys,
    _preflight_fresh_one_shot_recovery,
    _handle_plan_first_split_scope,
    _infer_staged_parent_issue,
    _resolved_stage_ids,
)
from .child_plan_binding import (
    _plan_validation_contract_versions,
    MAX_INHERITED_MATRIX_REPLANS,
    _InheritedReplanDiagnostic,
    _inherited_matrix_binding,
    _PlanningChildBinding,
    _PlanSupersessionBinding,
    _child_plan_admissibility_failure,
    _child_plan_supersession_route,
    verify_child_plan_rebind,
    verified_retired_child_plan_hashes,
    _managed_ci_retired_plan_hashes,
    _require_authorized_replan_state,
    _PlanContractShape,
    _REBIND_EXPANSION_LIST_LIMIT,
    _REBIND_EXPANSION_ITEM_CHARS,
    _REBIND_EXPANSION_HEADING,
    _CONTRACT_ELEMENT_ID_KEYS,
    _flatten_contract_values,
    _recommendation_commitments,
    _matrix_row_values,
    _plan_contract_shape,
    _REBIND_EXPANSION_HEAD_CHARS,
    _REBIND_EXPANSION_LEAD_CHARS,
    _flatten_contract_text,
    _common_prefix_length,
    _clip_contract_item,
    _added_contract_items,
    _plan_contract_expansion,
    _render_contract_expansion_notice,
    _rebind_contract_expansion_notice,
    _REBIND_GROWTH_HEADING,
    _REBIND_GROWTH_RATIONALE_CHARS,
    _render_rebind_growth_advisory,
    _rebind_growth_advisory,
    _rebind_superseded_child_plan,
    _route_child_plan_handoff,
    _require_admissible_pr_child_plan,
    _reject_pending_child_plan_supersession,
    _inadmissible_plan_audit_line,
    _resumed_inherited_replan_force_full,
    _recover_current_plan_validation_diagnostic,
    _persist_exhausted_plan_validation_diagnostic,
    _assemble_structured_plan_round_body,
    _COMPACT_DIGEST_DISPOSITION_LINE_RE,
    _compact_digest_disposition_ids,
    _compact_digest_acknowledgement_text,
    _post_plan_coder_round_comment,
)
from .pr_loop_support import (
    ExactHeadCiProof,
    _render_ci_rerun_command,
    _print_unprotected_managed_ci_warning,
    _merge_with_exact_head_proof,
    _read_assigned_workdir_head,
    MAX_UNCHANGED_HEAD_CODER_TURNS,
    _UnchangedHeadTracker,
    _coder_followup_head_log,
    _coder_followup_review_context,
    _reviewer_summary_context,
    _extract_structured_coder_summary,
    _extract_structured_coder_tests_run,
    _finalize_ordinary_recovery_merge,
    _require_clean_merge_state_after_ready,
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
    _sub_item_progress_already_posted,
    _publish_sub_item_progress,
    _sub_item_progress_block,
    _round_limit_diagnostic,
    _round_limit_blocker_message,
    _single_line_diagnostic,
    _finalization_obligation_predicate,
    _finalization_obligation_detail,
    _finalization_obligation_lines,
    _ensure_finalization_ready,
    _RepeatableRoundSequence,
    _evidence_barrier_note,
    _append_evidence_barrier_note,
    _evidence_item_lines,
    _evidence_freeze_diagnostic,
    _render_evidence_freeze_notice,
    _EVIDENCE_RELEASE_MESSAGES,
    _publish_evidence_freeze,
    _publish_evidence_release,
    _EvidenceGateOutcome,
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
    _visibility_snapshot,
    _latest_pr_reviewer_records,
    _reviewer_needs_fresh_context,
    _reviewer_diff_summary,
    _reviewer_history_is_reconstructible,
    _returning_reviewer_context,
    _all_pending_resolution_owners_unavailable,
    _observe_pr_transition,
    _scheduler_contract_from_metadata,
    _pr_contract_drift_error,
    REDUCED_BOARD_COMPLETION_HEADING,
    _post_reduced_board_completion_note,
    _pr_amendment_start_round,
    _is_completed_full_board_scheduler_record,
    _managed_binding_retired_plan_hashes,
    _managed_binding_protection_mode,
    _PR_AMENDMENT_PLAN_BOARD_HINT,
    _pr_amendment_plan_board_hint,
    _verify_strict_managed_plan_binding,
    _fresh_pr_qualification_snapshot,
    _preserve_issue_created_managed_suppression,
    _recover_managed_ci_approved_plan,
)
from .pr_loop import run_pr_loop


def _embed_pr_contract_marker(body: str | TrustedBody, contract: PrExpectedClosingContract) -> TrustedBody:
    marker = render_pr_contract_marker(contract)
    body_text = str(body)
    if "\n-- " in body_text:
        prefix, signature = body_text.rsplit("\n-- ", 1)
        rendered = f"{prefix}\n{marker}\n-- {signature}"
    else:
        rendered = f"{body_text.rstrip()}\n{marker}"
    expected = tuple(item.definition.token for item in scan_reserved_markers(rendered))
    return TrustedBody.canonical(rendered, expected_tokens=expected)


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


def _advisory_issue_pr_provenance(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    expected_scope: IssuePrProvenanceScope,
) -> None:
    """Warn on missing provenance without blocking a newly created PR handoff."""
    if config.dry_run:
        return
    try:
        validate_pull_request_provenance(
            runner,
            config=config,
            pr_number=pr_number,
            expected_scope=expected_scope,
        )
    except AgentLoopError as exc:
        log(
            config,
            f"WARNING: PR #{pr_number} did not prove expected issue commit provenance "
            f"(repository={expected_scope.repository}, issue=#{expected_scope.issue_number}, "
            f"flow={expected_scope.flow}, plan={expected_scope.approved_plan_hash or 'none'}): {exc}. "
            "Do not rewrite or force-push solely to satisfy this warning. If execution is "
            "interrupted before handoff, resume the PR directly with "
            f"`agent-loop pr {pr_number}`.",
        )


def _publish_issue_authorization_with_recovery(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff,
    metadata: PullRequestMetadata,
    issue_number: int,
    approved_plan_hash_value: str | None = None,
) -> AuthenticatedIssueCreatedHandoff:
    """Publish the durable checkpoint or give an authority-changing remedy."""
    try:
        return publish_issue_created_authorization(
            runner,
            config=config,
            handoff=handoff,
            metadata=metadata,
            approved_plan_hash=approved_plan_hash_value,
        )
    except AgentLoopError as exc:
        command = render_managed_ci_resume_command(
            config,
            pr_number=handoff.pr_number,
            issue_number=issue_number,
            managed_ci=True,
            fresh_authorization=True,
            fresh_issue_number=issue_number,
        )
        raise AgentLoopError(
            f"{exc}\n\nManaged-CI authorization publication was interrupted and no "
            "handoff or qualification was claimed. After verifying the PR, create an "
            f"explicit new operator authorization with `{command}`."
        ) from exc


def _approved_implementation_config(config: AgentLoopConfig) -> tuple[AgentLoopConfig, bool]:
    """Return the config and session-reuse policy for approved plan implementation."""
    implementation_coder = config.implementation_coder or config.coder
    updates: dict[str, object] = {"coder": implementation_coder}
    reuse_session = implementation_coder == config.coder

    model = config.implementation_coder_model.strip()
    if model:
        reuse_session = False
        if implementation_coder == "claude":
            updates["claude_model"] = model
        elif implementation_coder == "codex":
            updates["codex_model"] = model
        elif implementation_coder == "gemini":
            updates["gemini_model"] = model
        elif implementation_coder == "antigravity":
            updates["antigravity_model"] = None
            updates["antigravity_models"] = (model,)

    effort = config.implementation_codex_reasoning_effort.strip()
    if effort:
        reuse_session = False
        updates["implementation_effort_active"] = True

    claude_effort = config.implementation_claude_effort.strip()
    if claude_effort:
        reuse_session = False
        updates["implementation_effort_active"] = True

    if updates == {"coder": config.coder}:
        return config, True
    return dataclasses_replace(config, **updates), reuse_session


def _implement_approved_issue(
    runner: Runner,
    *,
    issue_number: int,
    approved_plan: str,
    config: AgentLoopConfig,
    memory,
    issue_context: IssueContext,
    coder_session_id: str | None,
    usage_context: RunUsageContext,
    one_shot_parent_issue: int | None = None,
    plan_subject: str | None = None,
    staged_parent_issue: int | None = None,
    approved_plan_context: ApprovedPlanContext | None = None,
    parent_issue_context: IssueContext | None = None,
    execution_recommendation=None,
    plan_growth_verdict: PlanGrowthApprovalVerdict | None = None,
) -> int:
    implementation_config, reuse_planning_session = _approved_implementation_config(config)
    coder_name = agent_display_name(implementation_config.coder)
    implementation_session_id = coder_session_id if reuse_planning_session else None
    plan_hash = (
        approved_plan_context.plan_hash
        if approved_plan_context is not None and approved_plan_context.plan_hash
        else approved_plan_hash(approved_plan)
    )
    if plan_growth_verdict is None:
        # Callers without the planning loop's own approval-time assessment
        # measure the approved candidate from its authenticated plan round.
        plan_growth_verdict = _plan_growth_verdict_for_hash(
            config,
            plan_hash=plan_hash,
            comment_sources=(
                issue_context.comments,
                parent_issue_context.comments if parent_issue_context is not None else None,
            ),
        )
    execution_identity = (
        execution_recommendation.identity()
        if execution_recommendation is not None else None
    )
    if approved_plan_context is None:
        approved_plan_context = make_approved_plan_context(
            approved_plan,
            source_locator=f"issue #{issue_number} approved-plan implementation",
            expected_hash=plan_hash,
            expected_subject=plan_subject,
        )
    if execution_recommendation is not None and approved_plan_context.matrix_available:
        validate_risk_matrix_ownership(
            approved_plan_context.risk_test_matrix_payload,
            execution_recommendation,
        )
    plan_additions = _extract_current_expected_closing_issue_ids(approved_plan)
    implementation_requirements = deduplicate_human_requirements(
        [
            *(parent_issue_context.human_requirements if parent_issue_context is not None else ()),
            *issue_context.human_requirements,
        ]
    )
    implementation_human_requirements_context = render_coder_human_requirements_prompt_context(
        implementation_requirements,
    )

    # A prior implementation attempt may have created a PR and then aborted
    # before recording any handoff marker/comment (e.g. the #493 test-report
    # false positive, which produced the duplicate PR #494 for #492). Resolve
    # the canonical AGENT_ISSUE_PR_HANDOFF record first, falling back to the
    # legacy exactly-one-open-PR GitHub search when no record exists yet
    # (#495, #589).
    resolved_pr = resolve_canonical_pr_for_issue(
        runner,
        config=config,
        issue_number=issue_number,
        issue_context=issue_context,
        expected_fallback_scope=IssuePrProvenanceScope(
            repository=config.repo,
            issue_number=issue_number,
            flow="approved",
            approved_plan_hash=plan_hash,
        ),
    )
    recovered_contract_ids = (
        resolved_pr.metadata.expected_closing_issue_ids
        if resolved_pr is not None and resolved_pr.metadata is not None
        else (issue_number,) if resolved_pr is not None else None
    )
    closing_contract = resolve_issue_contract(
        primary_issue=issue_number,
        cli_additions=config.expected_closing_issue_ids,
        plan_additions=plan_additions,
        recovered=recovered_contract_ids,
        supersede=config.supersede_expected_closing_contract,
    )
    reject_parent_from_contract(closing_contract, parent_issue=staged_parent_issue)
    implementation_config = dataclasses_replace(
        implementation_config,
        expected_closing_issue_ids=closing_contract.issue_ids,
        expected_closing_contract_resolved=True,
    )
    if resolved_pr is not None:
        existing_pr_number = resolved_pr.pr_number
        if (
            resolved_pr.source == "canonical"
            and resolved_pr.metadata is not None
            and resolved_pr.metadata.flow == "approved-plan-implementation"
            and resolved_pr.metadata.plan_hash != plan_hash
        ):
            raise AgentLoopError(
                f"Canonical approved-plan handoff for issue #{issue_number} points to PR "
                f"#{existing_pr_number} with plan hash {resolved_pr.metadata.plan_hash}, "
                f"but the current approved plan has hash {plan_hash}. Review the recorded PR "
                f"with `agent-loop pr {existing_pr_number}` or remove the stale handoff marker."
            )
        log(
            config,
            f"Existing implementation PR #{existing_pr_number} found for issue #{issue_number} "
            f"/ approved plan {plan_hash}; resuming PR review instead of invoking {coder_name} "
            f"(source={resolved_pr.source}, evidence={resolved_pr.evidence_summary}).",
        )
        if staged_parent_issue is not None:
            validate_pr_body_does_not_close_issue(
                runner,
                config=implementation_config,
                pr_number=existing_pr_number,
                issue_number=staged_parent_issue,
            )
        resumed_pr_context = None
        if resolved_pr.source == "legacy-closing-reference" or one_shot_parent_issue is not None:
            resumed_pr_context = get_pr_review_context(
                runner, config=implementation_config, pr_number=existing_pr_number
            )
        # Explicit managed recovery may be resuming a PR whose implementation
        # report was rejected before the canonical handoff checkpoint. Its
        # durable authorization is sufficient to enter the PR loop, but must
        # not launder that rejected report into a canonical handoff.
        if resolved_pr.source == "legacy-closing-reference":
            pr_url, pr_head_sha = require_pr_metadata_for_handoff(resumed_pr_context.metadata)
            validate_pr_expected_closing_issues(
                runner,
                config=implementation_config,
                pr_number=existing_pr_number,
                expected_issue_ids=closing_contract.issue_ids,
                body=resumed_pr_context.metadata.body,
                reject_unexpected=implementation_config.managed_ci,
            )
            pr_contract = make_pr_contract(
                repository=implementation_config.repo,
                pr_number=existing_pr_number,
                origin_flow="approved-plan-implementation",
                primary_issue_number=issue_number,
                expected_closing_issue_ids=closing_contract.issue_ids,
                supersedes_hash=closing_contract.supersedes_hash,
            )
            post_trusted_pr_comment(
                runner,
                config=implementation_config,
                pr_number=existing_pr_number,
                body=TrustedBody.canonical(
                    format_pr_contract_comment(pr_contract),
                    expected_tokens=("AGENT_PR_EXPECTED_CLOSING_ISSUES",),
                ),
            )
            if not implementation_config.managed_ci:
                post_issue_pr_handoff_comment(
                    runner,
                    config=implementation_config,
                    issue_number=issue_number,
                    pr_number=existing_pr_number,
                    pr_url=pr_url,
                    pr_head_sha=pr_head_sha,
                    flow="approved-plan-implementation",
                    plan_hash=plan_hash,
                    expected_closing_issue_ids=closing_contract.issue_ids,
                    supersedes_hash=closing_contract.supersedes_hash,
                    plan_growth_verdict=plan_growth_verdict,
                )
        if one_shot_parent_issue is not None:
            post_one_shot_impl_handoff_comment(
                runner,
                config=implementation_config,
                parent_issue=one_shot_parent_issue,
                mode="implement-one-shot",
                plan_hash=plan_hash,
                plan_subject=plan_subject or "",
                pr_number=existing_pr_number,
                pr_head_sha=resumed_pr_context.metadata.head_sha,
                strategy=(execution_recommendation.strategy if execution_recommendation is not None else None),
                topology_source=(
                    str(execution_identity["topology_source"])
                    if execution_identity is not None else None
                ),
                execution_strategy_contract_version=(
                    1 if execution_recommendation is not None else None
                ),
                recommendation_digest=(
                    str(execution_identity["recommendation_sha256"])
                    if execution_identity is not None else None
                ),
            )
        return run_pr_loop(
            runner,
            pr_number=existing_pr_number,
            config=implementation_config,
            issue_context=issue_context,
            approved_plan_context=approved_plan_context,
            parent_issue_context=parent_issue_context,
            usage_context=usage_context,
            managed_ci_issue_number=issue_number,
        )

    salvage_summary = latest_salvage_context(
        implementation_config.log_dir,
        issue_context.comments,
        repo=implementation_config.repo,
        issue_number=issue_number,
        scope=APPROVED_PLAN_IMPLEMENTATION_SALVAGE_SCOPE,
        approved_plan_hash=plan_hash,
    )
    sync_coder_base_before_implementation(implementation_config, runner)
    implementation_config = _freeze_prompt_architecture(runner, implementation_config)
    managed_ci_creation_intent = preflight_managed_ci_creation(
        runner, config=implementation_config, issue_number=issue_number
    )
    if managed_ci_creation_intent is not None and managed_ci_creation_intent.audit_nonce:
        _print_unprotected_managed_ci_warning(managed_ci_creation_intent.protection_mode)
    log(config, f"Planning approved; invoking {coder_name} to implement issue #{issue_number}")
    assigned_head_before = _read_assigned_workdir_head(runner, implementation_config)
    coder_response = _run_validated_agent(
        runner,
        agent=implementation_config.coder,
        config=implementation_config,
        prompt=build_issue_implementation_prompt(
            issue_number,
            approved_plan,
            implementation_config,
            memory,
            issue_context=issue_context,
            salvage_summary=salvage_summary,
            staged_parent_issue=staged_parent_issue,
            managed_ci_creation_intent=managed_ci_creation_intent,
            approved_plan_context=approved_plan_context,
            parent_issue_context=parent_issue_context,
        ),
        session_id=implementation_session_id,
        marker_description="structured issue_implementation result, blocking, or clarification",
        require_architecture_impact_contract=True,
        **_architecture_mode_validators(lambda mode: lambda text: _validate_issue_implementation_response(
            text,
            human_requirements=implementation_requirements,
            require_architecture_impact=True,
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
            require_risk_test_matrix_contract=(
                approved_plan_context is not None and approved_plan_context.matrix_available
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
        repair_expected_kind="issue_implementation",
        repair_surfaced_requirement_ids=implementation_human_requirements_context.surfaced_requirement_ids,
        repair_requires_direct_discussion_ack=implementation_human_requirements_context.requires_direct_discussion_ack,
        salvage_context=SalvageContext(
            repo=implementation_config.repo,
            issue_number=issue_number,
            scope=APPROVED_PLAN_IMPLEMENTATION_SALVAGE_SCOPE,
            agent=implementation_config.coder,
            run_id=usage_context.run_id,
            approved_plan_hash=plan_hash,
        ),
        operation_description="approved-plan implementation",
        completion_recovery=CompletionRecoveryPolicy(
            issue_number=issue_number,
            issue_context=issue_context,
            approved_plan_context=approved_plan_context,
            parent_issue_context=parent_issue_context,
            human_requirements=implementation_requirements,
        ),
        managed_ci_recovery_protection=(
            managed_ci_creation_intent.protection_mode
            if managed_ci_creation_intent is not None else None
        ),
    )
    coder_output = coder_response.text
    implementation_result = coder_response.marker_value
    if isinstance(implementation_result, _TerminalIssueImplementationConflict):
        implementation_result = _TerminalIssueImplementationConflict(
            _degrade_out_of_checkout_tests(
                implementation_result.parsed, config=implementation_config
            )
        )
        _post_structured_issue_implementation_terminal_comment(
            runner,
            config=implementation_config,
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
                implementation_result, config=implementation_config
            )
            _post_structured_issue_implementation_terminal_comment(
                runner,
                config=implementation_config,
                issue_number=issue_number,
                parsed=implementation_result,
                model_used=coder_response.model_used,
            )
            raise AgentLoopError(
                "Coder did not create a valid PR; implementation is blocking."
            )
        pr_number = implementation_result.pr_number
    elif isinstance(implementation_result, _TerminalNoPrImplementation):
        _post_no_pr_implementation_terminal_comment(
            runner,
            config=implementation_config,
            issue_number=issue_number,
            coder_response=coder_response,
        )
        raise AgentLoopError(
            "Coder did not create a valid PR; implementation is " + implementation_result.state + "."
        )
    else:
        raise AgentLoopError("Issue implementation validator returned an unknown result type.")
    validate_assigned_head_advanced(
        before_head=assigned_head_before,
        after_head=_read_assigned_workdir_head(runner, implementation_config),
        assigned_workdir=active_workdir(implementation_config),
    )
    log(config, f"{coder_name} reported PR #{pr_number}; validating it is open")
    validate_open_pr(runner, config=implementation_config, pr_number=pr_number)
    initial_pr_context = get_pr_review_context(runner, config=implementation_config, pr_number=pr_number)
    managed_ci_handoff: AuthenticatedIssueCreatedHandoff | None = None
    if managed_ci_creation_intent is not None:
        managed_ci_handoff = authenticate_issue_created_handoff(
            runner,
            config=implementation_config,
            intent=managed_ci_creation_intent,
            issue_number=issue_number,
            pr_number=pr_number,
            metadata=initial_pr_context.metadata,
        )
        if managed_ci_handoff.override_nonce is not None:
            # Install the expected nonce before any PR/issue publication.  It
            # remains runtime-only and is revalidated at run_pr_loop entry.
            implementation_config = dataclasses_replace(
                implementation_config,
                managed_ci_expected_override_nonce=managed_ci_handoff.override_nonce,
            )
        if managed_ci_handoff is not None:
            managed_ci_handoff = _publish_issue_authorization_with_recovery(
                runner,
                config=implementation_config,
                handoff=managed_ci_handoff,
                metadata=initial_pr_context.metadata,
                issue_number=issue_number,
                approved_plan_hash_value=plan_hash,
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
            config=implementation_config,
            pr_number=pr_number,
        )
        _validate_structured_response_observations_with_post_pr_context(
            implementation_result.test_observations,
            runner=runner,
            config=implementation_config,
            pr_number=pr_number,
        )
    validate_pr_references_issue(
        runner,
        config=implementation_config,
        pr_number=pr_number,
        issue_number=issue_number,
        staged_parent_issue=staged_parent_issue,
        body=initial_pr_context.metadata.body,
    )
    validate_pr_expected_closing_issues(
        runner,
        config=implementation_config,
        pr_number=pr_number,
        expected_issue_ids=closing_contract.issue_ids,
        body=initial_pr_context.metadata.body,
    )
    _advisory_issue_pr_provenance(
        runner,
        config=implementation_config,
        pr_number=pr_number,
        expected_scope=IssuePrProvenanceScope(
            repository=implementation_config.repo,
            issue_number=issue_number,
            flow="approved",
            approved_plan_hash=plan_hash,
        ),
    )
    initial_pr_url, initial_pr_head_sha = require_pr_metadata_for_handoff(initial_pr_context.metadata)
    pr_contract = make_pr_contract(
        repository=implementation_config.repo,
        pr_number=pr_number,
        origin_flow="approved-plan-implementation",
        primary_issue_number=issue_number,
        expected_closing_issue_ids=closing_contract.issue_ids,
        supersedes_hash=closing_contract.supersedes_hash,
    )
    post_trusted_pr_contract_record(
        runner,
        config=implementation_config,
        pr_number=pr_number,
        body=TrustedBody.canonical(
            format_pr_contract_comment(pr_contract),
            expected_tokens=("AGENT_PR_EXPECTED_CLOSING_ISSUES",),
        ),
    )
    post_issue_pr_handoff_comment(
        runner,
        config=implementation_config,
        issue_number=issue_number,
        pr_number=pr_number,
        pr_url=initial_pr_url,
        pr_head_sha=initial_pr_head_sha,
        flow="approved-plan-implementation",
        plan_hash=plan_hash,
        expected_closing_issue_ids=closing_contract.issue_ids,
        supersedes_hash=closing_contract.supersedes_hash,
        plan_growth_verdict=plan_growth_verdict,
    )
    if one_shot_parent_issue is not None:
        post_one_shot_impl_handoff_comment(
            runner,
            config=implementation_config,
            parent_issue=one_shot_parent_issue,
            mode="implement-one-shot",
            plan_hash=plan_hash,
            plan_subject=plan_subject or "",
            pr_number=pr_number,
            pr_head_sha=initial_pr_context.metadata.head_sha,
            strategy=(execution_recommendation.strategy if execution_recommendation is not None else None),
            topology_source=(
                str(execution_identity["topology_source"])
                if execution_identity is not None else None
            ),
            execution_strategy_contract_version=(
                1 if execution_recommendation is not None else None
            ),
            recommendation_digest=(
                str(execution_identity["recommendation_sha256"])
                if execution_identity is not None else None
            ),
        )
    implementation_result, _initial_derived_risk_evidence = _derive_authenticated_risk_evidence_for_coder(
        implementation_result,
        approved_plan_context=approved_plan_context,
        runner=runner,
        assigned_workdir=active_workdir(implementation_config),
        head_sha=initial_pr_context.metadata.head_sha,
        config=implementation_config,
        session_id=coder_response.session_id,
        invocation_id=coder_response.acquisition_test_turn_id,
        _closed_execution_catalog=coder_response.acquisition_test_observations,
        _journal_observations=coder_response.acquisition_test_observations,
        reauthenticate_head=lambda: get_pr_review_context(
            runner, config=implementation_config, pr_number=pr_number
        ).metadata.head_sha,
    )
    initial_local_test_evidence = runner.render_local_test_evidence(
        current_head=initial_pr_context.metadata.head_sha,
        legacy_tests_run=implementation_result.tests_run,
        cwd=active_workdir(implementation_config),
    )
    initial_coder_body = _attach_round_metadata(
        render_public_agent_comment(
            kind="issue_implementation",
            parsed=implementation_result,
            agent=implementation_config.coder,
            config=implementation_config,
            model_used=coder_response.model_used,
            local_test_evidence=initial_local_test_evidence,
            current_test_turn_id=coder_response.acquisition_test_turn_id,
        ),
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent=coder_name,
            round_number=1,
            subject=str(initial_pr_context.metadata.head_sha or "unknown"),
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
            **_test_observation_degradation_fields(implementation_result),
            **_architecture_metadata_fields(
                implementation_config,
                result=implementation_result,
            ),
            acquisition_outcome=coder_response.acquisition_outcome,
            acquisition_returncode=coder_response.acquisition_returncode,
        ),
    )
    post_trusted_pr_comment(
        runner,
        config=implementation_config,
        pr_number=pr_number,
        body=_embed_pr_contract_marker(initial_coder_body, pr_contract),
    )
    return run_pr_loop(
        runner,
        pr_number=pr_number,
        config=implementation_config,
        coder_session_id=coder_response.session_id,
        issue_context=issue_context,
        approved_plan_context=approved_plan_context,
        parent_issue_context=parent_issue_context,
        workdirs_ready=True,
        usage_context=usage_context,
        pre_review_test_pending=True,
        managed_ci_handoff=managed_ci_handoff,
    )


def _decompose_approved_plan(
    runner: Runner,
    *,
    issue_number: int,
    approved_plan: str,
    config: AgentLoopConfig,
    memory,
    issue_context: IssueContext,
    mode: str,
    coder_session_id: str | None,
    usage_context: RunUsageContext,
    execution_recommendation=None,
    normalized_topology=None,
) -> StagedTopologyOutcome | NeedsHumanDecision:
    plan_hash = approved_plan_hash(approved_plan)
    plan_subject = _plan_subject(approved_plan)
    if execution_recommendation is not None and normalized_topology is None:
        normalized_topology = normalize_execution_recommendation(
            execution_recommendation,
            approved_plan=approved_plan,
            plan_subject=plan_subject,
        )
    if execution_recommendation is not None:
        parent_matrix_context = make_approved_plan_context(
            approved_plan,
            source_locator=f"issue #{issue_number} approved-plan topology",
            expected_hash=plan_hash,
            expected_subject=plan_subject,
        )
        if parent_matrix_context.matrix_available:
            risk_matrix_payload = parent_matrix_context.risk_test_matrix_payload
            validate_risk_matrix_ownership(
                parent_matrix_context.risk_test_matrix_payload,
                execution_recommendation,
            )
        else:
            risk_matrix_payload = None
    else:
        risk_matrix_payload = None
    if normalized_topology is not None:
        decomposition, retained_parent_scope = normalized_topology
        reject_legacy_topology_collision(
            issue_context.comments,
            parent_issue=issue_number,
            plan_hash=plan_hash,
        )
        topology_source = EXECUTION_TOPOLOGY_SOURCE
        canonical_strategy = decomposition.strategy
        recommendation_digest = decomposition.recommendation_digest
        execution_contract_version = decomposition.execution_strategy_contract_version
        existing = find_existing_decomposition(
            issue_context.comments,
            parent_issue=issue_number,
            plan_hash=plan_hash,
            strategy=canonical_strategy,
            topology_source=topology_source,
            recommendation_digest=recommendation_digest,
            plan_subject=plan_subject,
        )
    else:
        canonical_strategy = None
        recommendation_digest = None
        execution_contract_version = None
    existing = find_existing_decomposition(
        issue_context.comments,
        parent_issue=issue_number,
        plan_hash=plan_hash,
        mode=mode,
    ) if normalized_topology is None else existing
    if existing is not None:
        log(config, f"Plan decomposition already exists for issue #{issue_number} ({mode}); not recreating children")
        adopted = tuple(
            CreatedPhaseIssue(
                phase=(
                    decomposition.phases[index]
                    if normalized_topology is not None
                    and index < len(decomposition.phases)
                    else RecordedPhase(title=title, automation=automation)
                ),
                issue_url=url,
                issue_number=number,
            )
            for index, ((title, url, number), automation) in enumerate(
                zip(existing.children, existing.automation, strict=False)
            )
        )
        # The recovered summary is the only source of stage identity and of
        # the parent's own obligations on the adopted and legacy paths, so it
        # is carried forward here instead of being discarded.
        return StagedTopologyOutcome(
            created=adopted,
            stage_ids=_resolved_stage_ids(
                adopted,
                normalized_topology=normalized_topology,
                recorded_stage_ids=existing.stage_ids,
            ),
            automations=tuple(item.phase.automation for item in adopted),
            plan_hash=plan_hash,
            mode=mode,
            topology_source=existing.topology_source,
            retained_parent_scope=(
                retained_parent_scope
                if normalized_topology is not None
                else existing.retained_parent_scope
            ),
            final_integration_work=(
                decomposition.final_integration_work
                if normalized_topology is not None
                else existing.final_integration_work
            ),
        )

    checkpoint = None
    if normalized_topology is None:
        checkpoint = find_existing_topology_checkpoint(
            issue_context.comments,
            parent_issue=issue_number,
            plan_hash=plan_hash,
            mode=mode,
        )
        retained_parent_scope = None
        topology_source = "model"
    if checkpoint is not None:
        # A checkpoint is the normalized model/typed output.  Reuse it before
        # invoking a coder so a create-before-summary failure is resumable.
        decomposition = PlanDecomposition(
            phases=checkpoint.phases,
            architecture_impact=(
                parse_architecture_impact(
                    # The checkpoint decoder restores lists as tuples; give the
                    # parser its JSON-array wire shape back.  Stored text keeps
                    # the explicit legacy decode, exactly as before #925.
                    sanitize_architecture_impact(checkpoint.architecture_impact),
                    context="checkpoint.architecture_impact",
                    architecture_status_mode="legacy",
                )
                if checkpoint.architecture_impact is not None else None
            ),
        )
        topology_source = checkpoint.topology_source
        retained_parent_scope = checkpoint.retained_parent_scope
    elif normalized_topology is None and mode == "decompose-only":
        typed_stages = _extract_current_child_stages(approved_plan)
        if typed_stages:
            decomposition, retained_parent_scope = adapt_typed_child_stages(
                typed_stages,
                approved_plan=approved_plan,
                plan_subject=_plan_subject(approved_plan),
            )
            topology_source = "typed"

    if normalized_topology is None and checkpoint is None and topology_source == "model":
        coder_name = agent_display_name(config.coder)
        log(config, f"Planning approved; invoking {coder_name} to decompose issue #{issue_number}")
        try:
            decomposition_response = _run_validated_agent(
                runner,
                agent=config.coder,
                config=config,
                prompt=build_plan_decomposition_prompt(
                    issue_number,
                    approved_plan,
                    config,
                    memory,
                    issue_context=issue_context,
                ),
                session_id=coder_session_id,
                marker_description="plan decomposition JSON",
                **_architecture_mode_validators(lambda mode: lambda text: parse_plan_decomposition(
                    text, required_architecture_impact_contract=1, architecture_status_mode=mode
                )),
                usage_context=usage_context,
                operation_description="plan decomposition",
                require_architecture_impact_contract=True,
            )
        except AgentInvocationError as exc:
            _surface_refused_decomposition(
                runner, config=config, issue_number=issue_number, error=exc
            )
            raise
        decomposition = decomposition_response.marker_value
        _surface_decomposition_degradations(
            runner, config=config, issue_number=issue_number, decomposition=decomposition
        )
    if topology_source == EXECUTION_TOPOLOGY_SOURCE and risk_matrix_payload is None:
        recovered_matrix_context = make_approved_plan_context(
            approved_plan,
            source_locator=f"issue #{issue_number} approved-plan topology",
            expected_hash=plan_hash,
            expected_subject=plan_subject,
        )
        if recovered_matrix_context.matrix_available:
            risk_matrix_payload = recovered_matrix_context.risk_test_matrix_payload
    created = create_decomposition_child_issues(
        runner,
        config=config,
        parent_issue=issue_number,
        approved_plan=approved_plan,
        decomposition=decomposition,
        topology_source=topology_source,
        issue_comments=issue_context.comments,
        mode=mode,
        retained_parent_scope=retained_parent_scope,
        strategy=canonical_strategy,
        execution_strategy_contract_version=execution_contract_version,
        recommendation_digest=recommendation_digest,
        plan_subject=plan_subject,
        risk_test_matrix=risk_matrix_payload,
    )
    if isinstance(created, NeedsHumanDecision):
        return created
    summary_allocation_kwargs = (
        {"final_integration_work": decomposition.final_integration_work}
        if topology_source == EXECUTION_TOPOLOGY_SOURCE
        else {}
    )
    post_decomposition_parent_summary(
        runner,
        config=config,
        parent_issue=issue_number,
        mode=mode,
        plan_hash=plan_hash,
        created=created,
        topology_source=topology_source,
        retained_parent_scope=retained_parent_scope,
        strategy=canonical_strategy,
        execution_strategy_contract_version=execution_contract_version,
        recommendation_digest=recommendation_digest,
        plan_subject=plan_subject,
        **summary_allocation_kwargs,
    )
    return StagedTopologyOutcome(
        created=tuple(created),
        stage_ids=_resolved_stage_ids(created, normalized_topology=normalized_topology),
        automations=tuple(item.phase.automation for item in created),
        plan_hash=plan_hash,
        mode=mode,
        topology_source=topology_source,
        retained_parent_scope=retained_parent_scope,
        final_integration_work=(
            decomposition.final_integration_work
            if topology_source == EXECUTION_TOPOLOGY_SOURCE
            else None
        ),
    )


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

    for round_number in range(start_round_number, config.max_rounds + 1):
        current_resume = resumed_round if resumed_round is not None and round_number == resumed_round.round_number else None
        prior_unresolved_items = current_resume.prior_items if current_resume is not None else tuple(unresolved_items)
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
        plan_posted_checkpoint: PostedRoundMetadata | None = None
        plan_panel_evidence = PlanPanelEvidence()
        plan_qualifying_approvals: tuple[str, ...] = ()
        plan_previous_key: PlanCandidateKey | None = None
        round_reviewers = tuple(configured_reviewers)
        # Under primary-then-panel, a non-compliant candidate the primary has
        # already approved can never pass, so the panel is not scheduled on it:
        # this round invokes no reviewer and the gate below starts a planner
        # revision instead of a reviewer-only phase advance (#886).
        growth_skip_panel = False
        if staged_planning and plan_growth_notice is not None and plan_primary_name is not None:
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
                f"Planning round {round_number}: the primary approved a candidate that fails "
                "the plan-growth gate; no panel review is scheduled on it",
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
                config.plan_primary_stall_rounds > 0
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
                    )
                    if plan_primary_streak.count >= config.plan_primary_stall_rounds:
                        # Before the prelaunch checkpoint and every agent turn,
                        # so the stop writes no record a resume could read as
                        # a checkpoint, an approval, or a panel opening.
                        stop_plan_pre_panel(
                            plan_primary_stall_message(
                                streak=plan_primary_streak.count,
                                threshold=config.plan_primary_stall_rounds,
                                plan_chars=len(current_plan),
                                legacy_undigested=plan_primary_streak.edit_cannot_clear(
                                    config.plan_primary_stall_rounds
                                ),
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
            post_issue_comment(
                runner, config=config, issue_number=issue_number,
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

        # One private spool per round, shared by the parallel launcher and the
        # sequential resume seam (#1025).
        plan_round_spool = _review_round_spool(
            config, surface="plan", number=issue_number,
            round_number=round_number, subject=current_plan_subject,
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
        if plan_round_parallel and not (current_resume is not None and current_resume.reconciled):
            settled = ", ".join(agent_display_name(reviewer) for reviewer in round_reviewers)
            post_issue_comment(
                runner, config=config, issue_number=issue_number,
                body=_attach_round_metadata(
                    f"Plan review round {round_number} reconciliation: settled reviewers: {settled or 'none'}. "
                    f"Finalization {'stops' if plan_fatal_errors else 'continues'} after reconciliation.",
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
                        action="Revise the implementation plan to address the blocking plan review.",
                    ),
                    require_risk_test_matrix_contract=require_fresh_matrix_contract,
                    plan_validation_diagnostic=turn_diagnostic,
                    inherited_matrix_binding=inherited_matrix_binding,
                    response_form=("semantic-patch-v1" if semantic_revision else None),
                    base_round_number=(semantic_base.round_number if semantic_base is not None else None),
                    base_state_identity=(semantic_base.state_identity if semantic_base is not None else None),
                    plan_growth_notice=plan_growth_notice,
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
        try:
            plan_round_metadata = PostedRoundMetadata(
                    flow="plan",
                    role="coder",
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
