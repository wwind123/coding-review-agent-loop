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
from .issue_implementation import (
    _embed_pr_contract_marker,
    _advisory_issue_pr_provenance,
    _publish_issue_authorization_with_recovery,
    _approved_implementation_config,
    _implement_approved_issue,
    _decompose_approved_plan,
)
from .plan_first_loop import (
    _run_child_planning_cycle,
    _dispatch_decomposition_child,
    _dispatch_current_decomposition_phase,
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
