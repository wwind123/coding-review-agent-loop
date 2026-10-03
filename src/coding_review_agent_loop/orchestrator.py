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


@dataclass(frozen=True)
class ExactHeadCiProof:
    """Non-empty current-head CI evidence required by automated merge paths."""

    head_sha: str
    source: str


def _render_ci_rerun_command(config: AgentLoopConfig, *, pr_number: int) -> str:
    """Render local CI recovery guidance through the shared token builder."""
    return render_managed_ci_resume_command(
        config,
        pr_number=pr_number,
        managed_ci=config.managed_ci,
        preserve_managed_options=(
            config.managed_ci
            or config.managed_ci_trusted_actor is not None
            or config.allow_unprotected_managed_ci
            or config.allow_unreadable_protection
            or config.managed_ci_adopt_existing_pr
        ),
        include_context=False,
    )


def _print_unprotected_managed_ci_warning(protection_mode: str) -> None:
    message = (
        "WARNING: --allow-unprotected-managed-ci is active for this invocation. GitHub cannot "
        "prevent a manual merge, other automation, a compromised credential, or an agent-loop "
        "defect from bypassing the voluntary final-ci/exact-head gate."
    )
    if protection_mode == "unreadable":
        message += (
            " --allow-unreadable-protection is also active: classic branch protection could not "
            "be read by this token, so the exact-head gate is treated as voluntary for this "
            "invocation."
        )
    print(message)


def _merge_with_exact_head_proof(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    proof: ExactHeadCiProof,
) -> None:
    """Make the live-head read the final remote operation before merging."""
    # Fetch a fresh, minimal head as the final normal remote read. If GitHub
    # serves an inconsistent GraphQL projection while it is converging, fetch
    # the full live PR tuple and require that authoritative view to agree with
    # the proof too; neither cached review metadata nor an old check board is
    # accepted.
    live_head = get_pr_head_sha(runner, config, pr_number)
    if live_head != proof.head_sha:
        live_head = get_pr_review_context(
            runner, config=config, pr_number=pr_number
        ).metadata.head_sha
    if live_head != proof.head_sha:
        raise AgentLoopError(
            f"PR #{pr_number} head changed after {proof.source} CI proof; no merge attempted."
        )
    merge_pr(runner, config, pr_number, expected_head_sha=proof.head_sha)


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


def _read_assigned_workdir_head(runner: Runner, config: AgentLoopConfig) -> str | None:
    # The PR handoff guard intentionally keeps strict exception behavior: a
    # runner/tooling exception must not be silently converted into evidence
    # that the coder advanced the assigned checkout. Ordinary Git failures
    # and blank HEAD output remain an unavailable (None) observation.
    probe = read_workdir_head(runner, active_workdir(config))
    return probe.value if probe.available else None


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


# Consecutive PR follow-up coder turns allowed to leave the head unchanged
# before the loop stops instead of re-reviewing an identical diff (#985).
MAX_UNCHANGED_HEAD_CODER_TURNS = 2


class _UnchangedHeadTracker:
    """Count consecutive coder follow-ups that left one PR head unchanged (#985).

    The count belongs to a single head: a follow-up on any other head,
    including one advanced externally between rounds, starts a fresh count,
    and a follow-up that moves the head clears it.
    """

    def __init__(self) -> None:
        self.head_sha: str | None = None
        self.count = 0

    def observe(self, reviewed_head: str | None, head_after_followup: str | None) -> int:
        if not reviewed_head or head_after_followup != reviewed_head:
            self.head_sha, self.count = None, 0
        elif reviewed_head == self.head_sha:
            self.count += 1
        else:
            self.head_sha, self.count = reviewed_head, 1
        return self.count


def _coder_followup_head_log(
    round_number: int,
    coder_name: str,
    dispatch_head: str | None,
    observed_head: str | None,
    unchanged_count: int,
) -> str:
    """Describe a PR coder follow-up by its observed head change (#1034).

    The refetch cannot prove who moved the head, so an advance is reported
    actor-neutrally, and an unknown head makes no push or unchanged claim.
    """
    if not dispatch_head or not observed_head:
        return (
            f"Round {round_number}: {coder_name} follow-up complete; "
            "PR head change could not be determined"
        )
    if dispatch_head != observed_head:
        return (
            f"Round {round_number}: PR head advanced {dispatch_head[:12]}..{observed_head[:12]} "
            f"during {coder_name} follow-up; re-reviewing"
        )
    return (
        f"Round {round_number}: {coder_name} follow-up left PR head {observed_head[:12]} "
        f"unchanged ({unchanged_count}/{MAX_UNCHANGED_HEAD_CODER_TURNS}); re-reviewing"
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


def _coder_followup_review_context(
    text: str | None,
    metadata: PostedRoundMetadata | None,
    *,
    head_sha: str | None,
    assigned_workdir: Path | None = None,
) -> str:
    if not text or metadata is None:
        return ""
    if not head_sha or metadata.subject != head_sha:
        return (
            "Latest coder explanation omitted: its recorded head does not match "
            "the current PR head. Do not treat earlier fix claims as current evidence.\n"
        )
    summary = _extract_structured_coder_summary(text)
    tests = _extract_structured_coder_tests_run(text)
    out_of_checkout_tests: tuple[str, ...] = ()
    if tests and assigned_workdir is not None:
        # The persisted response keeps the coder's raw list; reapply the same
        # classification the public comment used so the reviewer never sees
        # an out-of-checkout baseline as an ordinary test run (#991).
        try:
            partition = partition_reported_tests_by_workdir(
                tests, assigned_workdir=assigned_workdir
            )
        except AgentLoopError:
            tests, out_of_checkout_tests = (), tuple(tests)
        else:
            tests, out_of_checkout_tests = partition.in_checkout, partition.out_of_checkout
    # Dropped-citation records are restored from the round metadata (#927).
    parsed = rebuild_resumed_coder_carrier(text, metadata)
    payload: dict[str, object] = {
        "summary": summary,
        "tests_run": tests,
        "local_test_evidence": metadata.local_test_evidence,
    }
    if out_of_checkout_tests:
        payload["out_of_checkout_context_runs_not_evidence"] = out_of_checkout_tests
    # Reviewers receive the orchestrator-derived authority carried by the
    # same round metadata as the public comment. Do not reconstruct it from
    # the coder's fresh response or from the cumulative local journal.
    if metadata.risk_test_matrix_evidence is not None:
        payload["risk_test_matrix_evidence"] = metadata.risk_test_matrix_evidence
    if metadata.risk_test_matrix_diagnostics:
        payload["risk_test_matrix_diagnostics"] = [
            dict(item) for item in metadata.risk_test_matrix_diagnostics
        ]
    if isinstance(parsed, StructuredCoderFollowup):
        payload.update(
            addressed_items=parsed.addressed_items,
            addressed_item_notes=parsed.addressed_item_notes,
            remaining_items=parsed.remaining_items,
            remaining_item_notes=parsed.remaining_item_notes,
            disputed_items=parsed.disputed_items,
            dispute_evidence=parsed.dispute_evidence,
            test_observations=[
                {
                    "command": item.command,
                    "receipt_id": item.receipt_id,
                    "claim": item.claim,
                }
                for item in parsed.test_observations
            ],
        )
    if (
        isinstance(parsed, (StructuredCoderFollowup, StructuredIssueImplementation))
        and parsed.test_observation_degradations
    ):
        # A dropped citation supports nothing; reviewers see that it was dropped.
        payload["test_observation_degradations"] = [
            record.to_payload() for record in parsed.test_observation_degradations
        ]
    if isinstance(parsed, StructuredCoderFollowup):
        if parsed.addressed_sub_items:
            # Advisory claims only: the reviewer verifies each with a sub-item
            # disposition; nothing here changes persisted sub-item status.
            payload["claimed_addressed_sub_items_unverified"] = list(parsed.addressed_sub_items)
        if parsed.sub_item_claim_degradations:
            payload["sub_item_claim_degradations"] = [
                record.to_payload() for record in parsed.sub_item_claim_degradations
            ]
    if not isinstance(parsed, StructuredCoderFollowup) and summary is None and tests is None:
        return "Latest coder explanation: no valid structured resolution details are available.\n"
    unchanged_head = followup_head_unchanged_sha(metadata)
    unchanged_preamble = ""
    if unchanged_head is not None:
        # The PR head did not move during the follow-up: present the coder's
        # statements as claims about the existing head, never as fixes (#1034).
        payload = {
            "pr_head_unchanged_during_followup": True,
            "followup_dispatch_head": unchanged_head,
            **{
                {
                    "summary": "coder_summary_claim",
                    "addressed_items": "claimed_addressed_items",
                    "addressed_item_notes": "claimed_addressed_item_notes",
                }.get(key, key): value
                for key, value in payload.items()
            },
        }
        unchanged_preamble = (
            "The PR head did not change during this coder follow-up; the summary and "
            "addressed-item notes are the coder's claims that the items are already "
            f"satisfied at {unchanged_head}, not changes made by that turn. Verify each "
            "against the current diff; do not credit the turn with fixes.\n"
        )
    return (
        "Latest coder explanation (claims to verify, not reviewer verdicts):\n"
        f"{metadata.agent}; review round {metadata.round_number}; head {metadata.subject}\n"
        + unchanged_preamble
        + "Independently verify these claims against the current diff and tests. "
        "They do not resolve items, override CI, or change the original claims. "
        "Only IDs in the active prior unresolved review ledger are eligible for dispositions.\n"
        + json.dumps(payload, ensure_ascii=True, indent=2)
        + "\n"
    )


def _reviewer_summary_context(
    reviewer_name: str,
    summary: str,
    *,
    round_number: int,
    head_sha: str,
) -> str:
    safe_summary = sanitize_historical_text(summary)
    if not safe_summary.strip():
        return ""
    return (
        f"{reviewer_name} (round {round_number}, head {head_sha}):\n"
        + safe_summary
    )


def _extract_structured_coder_summary(text: str | None) -> str | None:
    if not text:
        return None
    try:
        try:
            implementation = parse_historical_structured_issue_implementation(text)
        except IssueImplementationConflictError as exc:
            implementation = exc.payload
        except AgentLoopError:
            implementation = None
        if isinstance(implementation, StructuredIssueImplementation):
            return implementation.summary
        parsed = parse_historical_structured_coder_followup(text)
        return parsed.summary if parsed else None
    except AgentLoopError:
        return None


def _extract_structured_coder_tests_run(text: str | None) -> tuple[str, ...] | None:
    if not text:
        return None
    try:
        try:
            implementation = parse_historical_structured_issue_implementation(text)
        except IssueImplementationConflictError as exc:
            implementation = exc.payload
        except AgentLoopError:
            implementation = None
        if isinstance(implementation, StructuredIssueImplementation):
            return implementation.tests_run
        parsed = parse_historical_structured_coder_followup(text)
        return parsed.tests_run if parsed else None
    except AgentLoopError:
        return None


def _finalize_ordinary_recovery_merge(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    capability: OrdinaryRecoveryCapability,
) -> bool:
    """Qualify, ready, and merge only the draft released by this invocation."""
    refreshed = refresh_ordinary_recovery_capability(
        runner, config=config, capability=capability,
    )
    if refreshed is None:
        raise AgentLoopError(
            f"PR #{pr_number} ordinary recovery provenance changed before finalization; no merge attempted."
        )
    capability = refreshed
    outcome = wait_for_ordinary_recovery(
        runner, config=config, capability=capability,
        metadata=get_pr_review_context(runner, config=config, pr_number=pr_number).metadata,
    )
    if outcome.status != "passed":
        if outcome.status == "not_started":
            command = render_managed_ci_resume_command(
                config, pr_number=pr_number, managed_ci=False,
            )
            log(
                config,
                f"PR #{pr_number}: ordinary recovery CI did not start within the bounded startup window; "
                "leaving the PR draft and unmerged",
            )
            print(
                f"PR #{pr_number} remains draft and unmerged because ordinary recovery CI did not "
                f"materialize for the current head. Resume with `{command}`."
            )
            return False
        if outcome.status == "protection_unreadable":
            merge_state = (
                outcome.mergeability.merge_state_raw
                if outcome.mergeability is not None
                else None
            )
            log(
                config,
                f"PR #{pr_number}: ordinary recovery board is green but branch protection is "
                f"unreadable and the merge state is {merge_state or 'unavailable'}; "
                "leaving the PR draft and unmerged",
            )
            print(
                f"PR #{pr_number} remains draft and unmerged: ordinary recovery CI passed for "
                f"{capability.expected_head_sha}, but the current GitHub token cannot read branch "
                "protection (HTTP 403) and GitHub reports merge state "
                f"{merge_state or 'unavailable'} (DRAFT or CLEAN for the same head is required). "
                "Grant the token administration read access or resolve the merge state, then "
                "rerun agent-loop."
            )
            raise AgentLoopError(
                f"PR #{pr_number} ordinary recovery could not confirm merge readiness: branch "
                "protection is unreadable and GitHub's merge state is neither DRAFT nor CLEAN "
                "for the exact head; the draft was left unmerged."
            )
        raise AgentLoopError(
            f"PR #{pr_number} ordinary recovery did not qualify the exact head "
            f"({outcome.status}); the draft was left unmerged."
        )
    clean_required_after_ready = outcome.checks is not None and protection_awaits_readiness(
        outcome.checks, outcome.mergeability, head_sha=outcome.head_sha,
    )
    if not _ordinary_checks_snapshot_is_authoritative(
        outcome.checks,
        outcome.mergeability,
        head_sha=outcome.head_sha,
        defer_unreadable_protection=clean_required_after_ready,
    ):
        details = (
            _pr_check_details(outcome.checks)
            if outcome.checks is not None
            else ["No authoritative current-head check snapshot was available."]
        )
        log(
            config,
            f"PR #{pr_number}: ordinary recovery reported aggregate passing status "
            "without an authoritative success-only current-head check snapshot; "
            "leaving the PR draft and unmerged",
        )
        print(
            f"PR #{pr_number} remains draft and unmerged because ordinary recovery "
            "did not produce an authoritative success-only current-head check board "
            f"({'; '.join(details)})."
        )
        return False
    if not validate_ordinary_recovery_capability(runner, config=config, capability=capability):
        raise AgentLoopError(
            f"PR #{pr_number} ordinary recovery provenance changed before readiness; no merge attempted."
        )
    ready = runner.run(
        [config.gh_cmd, "pr", "ready", str(pr_number), "--repo", config.repo],
        cwd=active_workdir(config), check=False,
    )
    if ready.returncode != 0:
        raise AgentLoopError(f"Unable to mark recovered PR #{pr_number} ready for review.")
    if not validate_ordinary_recovery_capability(
        runner, config=config, capability=capability, require_draft=None,
    ):
        raise AgentLoopError(
            f"PR #{pr_number} head or provenance changed after `gh pr ready`; "
            "the PR remains ready and was not merged."
        )
    if clean_required_after_ready:
        assert outcome.checks is not None
        _require_clean_merge_state_after_ready(
            runner,
            config=config,
            pr_number=pr_number,
            checks=outcome.checks,
            head_sha=capability.expected_head_sha,
        )
    try:
        _merge_with_exact_head_proof(
            runner,
            config=config,
            pr_number=pr_number,
            proof=ExactHeadCiProof(
                head_sha=capability.expected_head_sha,
                source="ordinary recovery",
            ),
        )
    except Exception:
        # Do not convert a successfully readied PR back into a draft. A safe
        # rerun can now inspect the ready exact head and retry the merge gate.
        log(config, f"PR #{pr_number}: merge failed after ordinary recovery readiness; PR remains ready")
        raise
    return True


def _require_clean_merge_state_after_ready(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    checks: PullRequestChecks,
    head_sha: str,
) -> None:
    """Require GitHub's CLEAN merge state after a draft recovery is readied.

    Under unreadable (403) classic protection, the draft could only report
    ``DRAFT``.  Once ready, GitHub recomputes the merge state from the real
    protection; poll it for the bounded startup window and refuse to merge
    unless it is ``CLEAN`` for the same exact head.
    """
    attempts = max(
        1,
        (config.ci_startup_timeout_seconds + config.ci_poll_interval_seconds - 1)
        // config.ci_poll_interval_seconds,
    )
    mergeability: PullRequestMergeability | None = None
    for attempt in range(attempts):
        mergeability = get_pr_mergeability(runner, config=config, pr_number=pr_number)
        if board_protection_is_reliable(checks, mergeability, head_sha=head_sha):
            return
        if mergeability.state == "conflicted" or (
            mergeability.head_sha is not None and mergeability.head_sha != head_sha
        ):
            break
        if attempt < attempts - 1:
            runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
    merge_state = mergeability.merge_state_raw if mergeability is not None else None
    log(
        config,
        f"PR #{pr_number}: branch protection is unreadable and the readied PR's merge state "
        f"is {merge_state or 'unavailable'}, not CLEAN; PR remains ready and unmerged",
    )
    print(
        f"PR #{pr_number} was marked ready after ordinary recovery CI passed, but the current "
        "GitHub token cannot read branch protection (HTTP 403) and GitHub reports merge state "
        f"{merge_state or 'unavailable'} (CLEAN is required) for {head_sha}. Satisfy the remaining "
        "protection rules or grant the token administration read access, then merge manually "
        f"with `--match-head-commit {head_sha}` or rerun agent-loop."
    )
    raise AgentLoopError(
        f"PR #{pr_number} branch protection is unreadable and GitHub's merge state is not CLEAN "
        "after readiness; the PR remains ready and was not merged."
    )


def _stop_on_terminal_without_status(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    outcome: ManagedCiOutcome,
) -> int:
    conclusion = outcome.workflow_conclusion or "unknown"
    attempt_text = (
        f"run `{outcome.run_id}` attempt `{outcome.run_attempt}`"
        if outcome.run_id is not None
        else "the correlated managed-CI attempt"
    )
    body = (
        f"PR #{pr_number} managed exact-head CI stopped because {attempt_text} "
        f"reached terminal workflow state `{conclusion}` without publishing a "
        f"correlated `{FINAL_CONTEXT}` status. No terminal status was synthesized "
        "and no merge was attempted.\n\n"
        "The round is resumable: for the unchanged head, rerun the command after "
        "a legitimate GitHub rerun creates a higher attempt, or rerun it to dispatch "
        "a fresh eligible same-nonce run. If the head was corrected, restart exact-head "
        "review so a new ledger is created/used."
    )
    post_pr_comment(runner, config=config, pr_number=pr_number, body=body)
    log(
        config,
        f"Round {round_number}: managed CI reached terminal state without "
        "publishing the correlated exact-head status; no merge attempted",
    )
    print(
        f"PR #{pr_number} managed exact-head CI reached terminal workflow state "
        f"`{conclusion}` without publishing its correlated status. No merge was "
        "attempted; rerun after a legitimate GitHub rerun or fresh same-nonce "
        "dispatch (or restart review if the head changed)."
    )
    return 0


def _pr_followup_source_context(
    *,
    config: AgentLoopConfig,
    pr_number: int,
    pr_metadata: PullRequestMetadata,
    issue_context: IssueContext | None,
) -> FollowupSourceContext:
    linked = parse_linked_issue_numbers(pr_metadata.body, repo=config.repo)
    parent_numbers = (issue_context.number,) if issue_context is not None else ()
    related = tuple(number for number in linked if number not in parent_numbers)
    if issue_context is None and len(linked) > 1:
        log(
            config,
            f"PR #{pr_number} has multiple linked issue references; preserving them as related context instead of inventing a parent",
        )
    if issue_context is None and not linked:
        log(config, f"Unable to resolve a parent issue for PR #{pr_number} follow-up lookup; using PR and topic context")
    return FollowupSourceContext(
        repo=config.repo,
        source_kind="pr",
        source_number=pr_number,
        source_identity=pr_metadata.head_sha,
        parent_issue_numbers=parent_numbers,
        related_issue_numbers=related,
        related_pr_numbers=(pr_number,),
    )


def _stop_after_ci_watch_timeout(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    head_sha: str | None,
    pr_comments: Sequence[object],
    followups: list[ApprovedFollowup],
    details: list[str],
    reason: Literal["budget_exhausted", "timeout", "non_authoritative", "protection_unreadable"],
    source_context: FollowupSourceContext,
    usage_context: RunUsageContext | None = None,
) -> int:
    """Publish resumable guidance for a watch that cannot continue or finish."""
    _publish_approved_followups(
        runner,
        config=config,
        pr_number=pr_number,
        head_sha=head_sha,
        pr_comments=pr_comments,
        followups=followups,
        source_context=source_context,
        usage_context=usage_context,
    )
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=(
            _unreadable_protection_stop_message(pr_number, details)
            if reason == "protection_unreadable"
            else _pending_ci_stop_message(pr_number, "pending", details)
        ),
    )
    rerun = _render_ci_rerun_command(config, pr_number=pr_number)
    note = (
        ""
        if config.invocation_argv
        else " (deterministic fallback; original invocation unavailable)"
    )
    if reason == "budget_exhausted":
        log(
            config,
            f"Round {round_number}: PR #{pr_number} shared CI watch budget was "
            "exhausted before a fresh poll; no merge attempted",
        )
        print(
            f"PR #{pr_number} CI watch budget was exhausted by earlier rounds; "
            f"no fresh poll was performed and no merge was attempted. "
            f"Rerun: {rerun}{note}"
        )
        if config.auto_merge:
            raise AgentLoopError(
                f"PR #{pr_number} full-board CI watch budget was exhausted before "
                "a fresh poll; no merge attempted."
            )
    elif reason == "timeout":
        print(
            f"PR #{pr_number} CI watch timed out: {'; '.join(details)}. "
            f"Rerun: {rerun}{note}"
        )
        if config.auto_merge:
            raise AgentLoopError(
                f"PR #{pr_number} full-board CI watch did not pass within "
                f"{config.ci_timeout_seconds}s; no merge attempted."
            )
    elif reason == "protection_unreadable":
        log(
            config,
            f"Round {round_number}: PR #{pr_number} branch protection is unreadable and "
            "GitHub did not report a CLEAN merge state for the green board; no merge attempted",
        )
        print(
            f"PR #{pr_number} CI watch stopped: every observed check passed, but the current "
            "GitHub token cannot read branch protection (HTTP 403) and GitHub did not report "
            f"a CLEAN merge state for the current head: {'; '.join(details)}. "
            "Grant the token administration read access, or satisfy the remaining protection "
            f"rules (for example required reviews), then rerun: {rerun}{note}"
        )
        if config.auto_merge:
            raise AgentLoopError(
                f"PR #{pr_number} branch protection is unreadable and GitHub's merge state "
                "is not CLEAN; no merge attempted."
            )
    else:
        log(
            config,
            f"Round {round_number}: ordinary CI watcher returned a non-authoritative "
            "passing-looking board; no merge attempted",
        )
        print(
            f"PR #{pr_number} CI watch did not produce an authoritative success for the "
            f"current head: {'; '.join(details)}. No merge was attempted; rerun: {rerun}{note}"
        )
        if config.auto_merge:
            raise AgentLoopError(
                f"PR #{pr_number} full-board CI watch returned a non-authoritative "
                "passing-looking result; no merge attempted."
            )
    return 0


def _scheduler_obligations(
    items: Sequence[UnresolvedReviewItem],
    *,
    required_reviewers: Sequence[str] = (),
    active_statuses: frozenset[str] = frozenset({"blocking", "same-pr"}),
) -> tuple[ReviewObligation, ...]:
    """Build scheduler obligations from the canonical finding ledger.

    ``active_statuses`` defaults to the PR statuses.  The planning flow retains
    its follow-up findings as ``same-plan`` (#905, from #841), so it passes that
    status instead; otherwise a post-panel narrow revision for a ``same-plan``
    finding would lose its durable owner and the scheduler would fall through to
    a final sweep instead of invoking that owner plus the primary.
    """
    obligations: list[ReviewObligation] = []
    required = set(required_reviewers)
    for item in items:
        if item.status not in active_statuses:
            continue
        if _is_evidence_obligation(item):
            # No coder round can resolve human-only evidence, and its
            # unscoped owner would otherwise make every later code transition
            # broad (#1068).  Evidence passes run the full board by design.
            continue
        owners = item.resolution_owners or (item.reviewer,)
        raw_states = item.owner_states
        states = dict(raw_states or ((owner, "pending") for owner in owners))
        reconstructible = (
            bool(owners)
            and len(set(owners)) == len(owners)
            and all(isinstance(owner, str) and owner for owner in owners)
            and (not raw_states or (len(raw_states) == len(owners) and set(states) == set(owners)))
            and all(state in {"pending", "cleared"} for state in states.values())
            and item.reviewer in required
            and not any(note.startswith(CODER_DISPUTE_NOTE_PREFIX) for note in item.notes)
        )
        # A synthetic or orchestrator-owned item has no reviewer-authored
        # objective scope, so it must conservatively force a full board.
        scope = item.fix_scope if reconstructible else None
        if item.reviewer == "Orchestrator" or item.reviewer not in required:
            scope = None
        obligations.append(
            ReviewObligation(
                item_id=item.item_id,
                status=item.status,
                scope=scope,
                resolution_owners=tuple(owners),
                pending_owners=tuple(owner for owner in owners if states.get(owner) != "cleared"),
            )
        )
    return tuple(obligations)


def _partition_unresolved_items(
    items: Sequence[UnresolvedReviewItem],
    *,
    current_head_sha: str | None,
) -> dict[str, tuple[UnresolvedReviewItem, ...]]:
    """Separate human review work from machine qualification work.

    ``must_fix_items`` historically mixed these categories and made a
    synthetic CI reviewer look like a human owner.  The partitions are kept
    explicit so a reviewed repair head may qualify while every machine gate
    remains pending and visible until its own authority clears it.
    """
    reviewer_blockers: list[UnresolvedReviewItem] = []
    repair_required: list[UnresolvedReviewItem] = []
    revalidation_candidates: list[UnresolvedReviewItem] = []
    finalization_blockers: list[UnresolvedReviewItem] = []
    evidence_obligations: list[UnresolvedReviewItem] = []
    for item in items:
        if item.status not in {"blocking", "same-pr"}:
            continue
        finalization_blockers.append(item)
        if _is_evidence_obligation(item):
            # Human-only exact-head evidence is a final barrier (#1068): it
            # blocks finalization but is never coder or repair work.
            evidence_obligations.append(item)
        elif not _is_machine_obligation(item):
            reviewer_blockers.append(item)
        elif _machine_obligation_requires_repair(item, current_head_sha=current_head_sha):
            repair_required.append(item)
        elif _machine_obligation_is_revalidation_candidate(
            item, current_head_sha=current_head_sha
        ):
            revalidation_candidates.append(item)
        else:
            repair_required.append(item)
    coder_blockers = [*reviewer_blockers, *repair_required]
    return {
        "reviewer_blockers": tuple(reviewer_blockers),
        "repair_required_machine_obligations": tuple(repair_required),
        "revalidation_candidates": tuple(revalidation_candidates),
        "finalization_blockers": tuple(finalization_blockers),
        "coder_blockers": tuple(coder_blockers),
        "evidence_obligations": tuple(evidence_obligations),
    }


def _log_coder_followup_dispatch(
    config: AgentLoopConfig,
    round_number: int,
    coder_name: str,
    coder_followup_items: Sequence[UnresolvedReviewItem],
) -> None:
    """Announce a coder round by what actually routed it (#1024)."""
    if coder_followup_is_ci_repair(coder_followup_items):
        log(config, f"Round {round_number}: {coder_name} repairing failed CI")
    else:
        log(config, f"Round {round_number}: {coder_name} addressing reviewer feedback")


def _machine_obligation_checkpoint(
    items: Sequence[UnresolvedReviewItem],
    *,
    current_head_sha: str | None,
    base_branch: str | None,
    allowed_rounds: int,
    watch_failure_extension_used: bool,
    watch_head_extension_used: bool,
    lifecycle: str | None = None,
    qualification_attempt_id: str | None = None,
    approval_digest: str | None = None,
    plan_digest: str | None = None,
    requirements_digest: str | None = None,
    acquisition_digest: str | None = None,
    scheduler_digest: str | None = None,
) -> QualificationCheckpoint | None:
    """Create a bounded checkpoint from the active source-specific obligation."""
    candidates = [
        item for item in items
        if _is_machine_obligation(item)
        and item.status in {"blocking", "same-pr"}
        and item.obligation_kind in CI_MACHINE_OBLIGATION_KINDS
    ]
    item = next(
        (
            candidate for candidate in candidates
            if _machine_obligation_is_revalidation_candidate(
                candidate, current_head_sha=current_head_sha
            )
        ),
        None,
    ) or (candidates[0] if candidates else None)
    if item is None:
        return None
    effective_lifecycle = lifecycle or item.lifecycle or "repair_required"
    # Head advancement is deliberately fail-closed.  A force-push revert to a
    # previously failed head leaves the obligation in ``repair_required`` with
    # no candidate.  Several callers request the normal awaiting-review state
    # after a head change, but that request is invalid for this transition and
    # must not leak QualificationCheckpoint's ValueError out of the loop.
    if (
        not item.candidate_head_sha
        or item.candidate_head_sha == item.failed_head_sha
        or item.lifecycle == "repair_required"
    ):
        effective_lifecycle = "repair_required"
    return QualificationCheckpoint(
        obligation_kind=item.obligation_kind or "unknown",
        obligation_identity=item.obligation_identity or item.item_id,
        lifecycle=effective_lifecycle,
        failed_head_sha=item.failed_head_sha,
        candidate_head_sha=item.candidate_head_sha,
        base_branch=base_branch,
        approval_digest=approval_digest,
        plan_digest=plan_digest,
        requirements_digest=requirements_digest,
        acquisition_digest=acquisition_digest,
        scheduler_digest=scheduler_digest,
        qualification_attempt_id=qualification_attempt_id,
        watch_failure_extension_used=watch_failure_extension_used,
        watch_head_extension_used=watch_head_extension_used,
        allowed_rounds=allowed_rounds,
    )


def _qualification_digest(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:16]


def _qualification_checkpoint_review_identity_matches(
    checkpoint: QualificationCheckpoint,
    *,
    configured_reviewers: Sequence[AgentName],
    current_approvals: Mapping[str, object],
    unresolved_items: Sequence[UnresolvedReviewItem],
    expected_plan_digest: str | None,
    expected_requirements_digest: str,
    expected_acquisition_digest: str,
    expected_qualification_attempt_id: str | None = None,
) -> bool:
    """Check the durable proof needed to skip a resumed reviewer board.

    Repair handoffs do not claim that the candidate has been reviewed.  Once a
    checkpoint says that the candidate is ready for, or is already in, final
    qualification, however, it may suppress reviewer execution only when the
    persisted identities still describe the live transcript.  In particular,
    a summary-only checkpoint or a checkpoint with a changed ledger must force
    the normal full-board recovery path.
    """
    if checkpoint.lifecycle not in {"qualification_ready", "qualifying"}:
        return True
    expected_reviewers = {
        agent_display_name(reviewer) for reviewer in configured_reviewers
    }
    observed_reviewers = set(current_approvals)
    if observed_reviewers != expected_reviewers:
        return False
    if (
        not checkpoint.approval_digest
        or not checkpoint.requirements_digest
        or not checkpoint.acquisition_digest
        or not checkpoint.scheduler_digest
    ):
        return False
    current_approval_digest = _qualification_digest(tuple(sorted(observed_reviewers)))
    current_scheduler_digest = _qualification_digest(
        _prior_item_ledger_signature(unresolved_items)
    )
    return (
        checkpoint.approval_digest == current_approval_digest
        and checkpoint.plan_digest == expected_plan_digest
        and checkpoint.requirements_digest == expected_requirements_digest
        and checkpoint.acquisition_digest == expected_acquisition_digest
        and checkpoint.scheduler_digest == current_scheduler_digest
        and (
            checkpoint.lifecycle != "qualifying"
            or (
                expected_qualification_attempt_id is not None
                and checkpoint.qualification_attempt_id == expected_qualification_attempt_id
            )
        )
    )


def approved_pr_reopen_hint(pr_number: int) -> str:
    """Name the operator path for new instructions on an approved PR (#1020).

    An ordinary PR comment is not a requirement, so an approved head exits
    without dispatching an agent. A signed human requirement invalidates the
    carried approvals and re-invokes reviewers against it at the same head.
    """
    return (
        f" To add instructions this PR must still satisfy, post a PR comment that ends "
        f"with a line containing exactly `-- Human Reviewer`, then rerun `agent-loop pr "
        f"{pr_number}`. Unsigned comments are not read as requirements. Only a human may "
        "sign; an agent relaying an operator decision must disclose the relay in the body."
    )


def _sub_item_progress_already_posted(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    digest: str,
) -> bool:
    """Whether a trusted orchestrator comment already carries this round's record.

    Only comments written by the authenticated actor count; the same record in
    anyone else's comment is ordinary text and grants nothing (#958).
    """
    try:
        actor_login, actor_id = resolve_authenticated_github_actor(runner, config=config)
    except AgentLoopError:
        return False
    comments = get_pr_review_context(runner, config=config, pr_number=pr_number).comments
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        author_id = getattr(comment, "author_id", None)
        if author_id is not None:
            if author_id != actor_id:
                continue
        elif getattr(comment, "author", None) != actor_login:
            continue
        if (pr_number, round_number, digest) in sub_item_progress_record_keys(body):
            return True
    return False


def _publish_sub_item_progress(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    items: Sequence[UnresolvedReviewItem],
    cleared: Sequence[ClearedItemProgress],
    notes: Sequence[str] = (),
) -> None:
    """Log the sub-item signal and post the once-per-round confirmed outcome (#958).

    Advisory only: nothing here changes the round budget.  The comment is a
    pure function of the reconciled ledger, so a resumed round recomputes the
    same digest and the trusted-record check keeps the post exactly-once.
    """
    for note in notes:
        log(config, f"Round {round_number}: sub-item note: {note}")
    window = config.sub_item_stall_rounds
    for entry in sub_item_progress(items, (), current_round=round_number, window=window):
        if entry.classification == "stalled":
            log(
                config,
                f"Round {round_number}: WARNING: {entry.item_id} has "
                f"{entry.resolved}/{entry.total} sub-items resolved and none closed in the "
                f"last {entry.window} rounds (stalled; advisory, the round budget is unchanged).",
            )
    stalled = newly_stalled_items(items, current_round=round_number, window=window)
    if not cleared and not stalled:
        return
    digest = sub_item_progress_digest(cleared, stalled)
    if _sub_item_progress_already_posted(
        runner, config=config, pr_number=pr_number, round_number=round_number, digest=digest
    ):
        return
    body = render_sub_item_progress_comment(
        pr_number=pr_number, round_number=round_number, cleared=cleared, stalled=stalled
    )
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=TrustedBody.canonical(body, expected_tokens=("AGENT_SUB_ITEM_PROGRESS",)),
    )


def _sub_item_progress_block(
    items: Sequence[UnresolvedReviewItem], *, round_number: int, window: int
) -> str:
    """Per-item converging/stalled summary for a budget-exit message (#958).

    Empty when no carried item has sub-items, so every diagnostic stays
    byte-identical for findings without them.
    """
    lines = render_sub_item_progress_summary(
        sub_item_progress(items, (), current_round=round_number, window=window)
    )
    if not lines:
        return ""
    return "\nSub-item progress:\n" + "\n".join(f"- {line}" for line in lines)


def _round_limit_diagnostic(
    *,
    pr_number: int,
    round_number: int,
    items: Sequence[UnresolvedReviewItem],
    current_head_sha: str | None,
    sub_item_stall_rounds: int | None = None,
) -> str:
    """Describe the blocker plus, when items have sub-items, their progress (#958)."""
    message = _round_limit_blocker_message(
        pr_number=pr_number,
        round_number=round_number,
        items=items,
        current_head_sha=current_head_sha,
    )
    if sub_item_stall_rounds is None:
        return message
    return message + _sub_item_progress_block(
        items, round_number=round_number, window=sub_item_stall_rounds
    )


def _round_limit_blocker_message(
    *,
    pr_number: int,
    round_number: int,
    items: Sequence[UnresolvedReviewItem],
    current_head_sha: str | None,
) -> str:
    """Describe the actual terminal/resumable blocker, not its display owner."""
    partitions = _partition_unresolved_items(items, current_head_sha=current_head_sha)
    reviewer_blockers = partitions["reviewer_blockers"]
    repair = partitions["repair_required_machine_obligations"]
    candidates = partitions["revalidation_candidates"]
    evidence = partitions["evidence_obligations"]
    # Code and machine blockers are named first; deferred human-only evidence
    # is described as the remaining final barrier, never as the blocker.
    evidence_suffix = _evidence_barrier_note(evidence)
    if reviewer_blockers:
        names = ", ".join(
            f"{item.reviewer} ({item.item_id})" for item in reviewer_blockers
        )
        return (
            f"PR #{pr_number} still reported blocking issues after round {round_number}: "
            f"reviewer-owned findings: {names}. "
            "The named reviewer/owner must provide actionable resolution evidence."
            + evidence_suffix
        )
    if repair:
        details = ", ".join(
            f"{item.obligation_kind or 'unknown'} ({item.item_id})"
            for item in repair
        )
        return (
            f"PR #{pr_number} has blocking issues after round {round_number}: machine obligation(s) "
            "awaiting a new repair head: "
            f"{details}. Reviewer approval cannot clear them; push a strictly "
            "different corrected head."
            + evidence_suffix
        )
    if candidates:
        details = ", ".join(
            f"{item.obligation_kind or 'unknown'} ({item.item_id})"
            for item in candidates
        )
        return (
            f"PR #{pr_number} has a unanimously reviewed correction awaiting authoritative "
            f"qualification after round {round_number}: {details}."
            + evidence_suffix
        )
    if evidence and len(evidence) == len(
        [item for item in items if item.status in {"blocking", "same-pr"}]
    ):
        return (
            f"PR #{pr_number} has no open code or machine findings after round {round_number}; "
            "the remaining barrier is human-only exact-head evidence: "
            + ", ".join(f"{item.reviewer} ({item.item_id})" for item in evidence)
            + ". It is requested only at the clean head through an evidence freeze."
        )
    active = [item for item in items if item.status in {"blocking", "same-pr"}]
    if active:
        return (
            f"PR #{pr_number} has an unknown or unreconstructible persisted obligation after "
            f"round {round_number}; no qualification or merge is permitted."
        )
    return f"Reached the review budget after round {round_number} for PR #{pr_number}; human review required."


def _single_line_diagnostic(value: object) -> str:
    """Escape every control character so a value renders on exactly one line."""
    out: list[str] = []
    for ch in str(value):
        if unicodedata.category(ch)[0] == "C" or unicodedata.category(ch) in {"Zl", "Zp"}:
            code = ord(ch)
            out.append(f"\\x{code:02x}" if code <= 0xFF else f"\\u{code:04x}")
        else:
            out.append(ch)
    return "".join(out)


def _finalization_obligation_predicate(
    item: UnresolvedReviewItem, *, current_head_sha: str | None
) -> str:
    """Describe, from ledger fields only, what keeps one obligation unsatisfied.

    Never claims success that was not observed: no wording here says a source
    passed or succeeded, and ``failed at`` appears only where a failed head is
    recorded.
    """
    kind = item.obligation_kind
    lifecycle = item.lifecycle
    cur = current_head_sha or "none"
    candidate = item.candidate_head_sha
    failed = item.failed_head_sha
    if kind not in MACHINE_OBLIGATION_KINDS or kind == "unknown":
        return "unknown or unreconstructible obligation; no qualification or merge is permitted"
    if _is_evidence_obligation(item):
        frozen = f" at {candidate}" if candidate else ""
        return f"human-only exact-head evidence pending{frozen}"
    if kind == "human-requirements-acknowledgement":
        return (
            "signed human requirements have not been validly acknowledged; "
            "a response acknowledging them is required"
        )
    if kind == "merge-conflict":
        head = (
            f"merge conflict with the base branch confirmed at {failed}"
            if failed
            else "merge conflict with the base branch reported; head not confirmed"
        )
        return f"{head}; resolve the conflict on a new head"
    if kind == "alembic-migration":
        head = (
            f"migration validation failed at {failed}"
            if failed
            else "migration validation has no recorded clearance"
        )
        return (
            f"{head}; it is re-probed each round and clears only when "
            f"validation of the current head {cur} succeeds"
        )
    if kind in CI_MACHINE_OBLIGATION_KINDS:
        if lifecycle == "repair_required":
            if failed and failed == current_head_sha:
                return (
                    f"authoritative source failed at {failed}, which is the current head; "
                    "a strictly different head is required"
                )
            if failed:
                return (
                    f"authoritative source failed at {failed}; current head {cur} "
                    "has not been bound as a revalidation candidate"
                )
            return (
                "authoritative source reported a failure at an unrecorded head; "
                "a corrected head is required"
            )
        if lifecycle in {"awaiting_current_head_review", "qualification_ready", "qualifying"}:
            if not candidate:
                return f"lifecycle {lifecycle} has no recorded candidate head; failing closed"
            if failed and failed == current_head_sha:
                return (
                    f"lifecycle {lifecycle}, but the current head {cur} is the recorded "
                    "failed head; a strictly different head is required"
                )
            if candidate != current_head_sha:
                return (
                    f"lifecycle {lifecycle} is bound to candidate head {candidate}, "
                    f"not the current head {cur}; the current head must be reviewed "
                    "before qualification"
                )
            if lifecycle == "awaiting_current_head_review":
                return f"awaiting unanimous reviewer approval at candidate head {candidate}"
            if lifecycle == "qualification_ready":
                return f"approved at {candidate}; authoritative qualification not yet dispatched"
            return (
                f"qualification in progress at {candidate}; "
                "no authoritative success recorded for this source in this run"
            )
    return f"{kind} in lifecycle {lifecycle or 'none'}; no clearance recorded; failing closed"


def _finalization_obligation_detail(
    items: Sequence[UnresolvedReviewItem],
    *,
    current_head_sha: str | None,
    observations: Mapping[str, str] | None,
) -> str:
    """Render one escaped line per machine-obligation blocker (#1119)."""
    lines = _finalization_obligation_lines(
        items, current_head_sha=current_head_sha, observations=observations
    )
    if not lines:
        return ""
    return "\nBlocking obligations:\n" + "\n".join(f"- {line}" for line in lines)


def _finalization_obligation_lines(
    items: Sequence[UnresolvedReviewItem],
    *,
    current_head_sha: str | None,
    observations: Mapping[str, str] | None,
) -> list[str]:
    partitions = _partition_unresolved_items(items, current_head_sha=current_head_sha)
    ordered = (
        *partitions["repair_required_machine_obligations"],
        *partitions["revalidation_candidates"],
        *partitions["evidence_obligations"],
    )
    esc = _single_line_diagnostic
    lines: list[str] = []
    for item in ordered:
        if item.item_id in (observations or {}):
            predicate = f"observed in this run: {esc(observations[item.item_id])}"
        else:
            predicate = esc(
                _finalization_obligation_predicate(item, current_head_sha=current_head_sha)
            )
        lines.append(
            f"{esc(item.obligation_kind or 'unknown')} ({esc(item.item_id)}): "
            f"lifecycle={esc(item.lifecycle or 'none')}, "
            f"candidate_head={esc(item.candidate_head_sha or 'none')}, "
            f"failed_head={esc(item.failed_head_sha or 'none')}; {predicate}"
        )
    return lines


def _ensure_finalization_ready(
    *,
    pr_number: int,
    round_number: int,
    items: Sequence[UnresolvedReviewItem],
    current_head_sha: str | None,
    ignored_machine_kinds: frozenset[str] = frozenset(),
    sub_item_stall_rounds: int | None = None,
    observations: Mapping[str, str] | None = None,
    config: AgentLoopConfig | None = None,
) -> None:
    """Fail closed unless every non-ignored obligation is actually cleared.

    A coder-blocker-free partition is sufficient to start source-specific
    qualification, but it is never sufficient to approve or merge.  The
    authoritative success path may explicitly ignore the one source it is
    about to validate (ordinary recovery); all other ledger obligations must
    be gone before a finalization side effect.
    """
    blockers = tuple(
        item
        for item in _partition_unresolved_items(
            items, current_head_sha=current_head_sha
        )["finalization_blockers"]
        if not (
            _is_machine_obligation(item)
            and item.obligation_kind in ignored_machine_kinds
        )
    )
    if blockers:
        diagnostic = _round_limit_diagnostic(
            pr_number=pr_number,
            round_number=round_number,
            items=blockers,
            current_head_sha=current_head_sha,
            sub_item_stall_rounds=sub_item_stall_rounds,
        )
        detail = _finalization_obligation_detail(
            blockers, current_head_sha=current_head_sha, observations=observations
        )
        if config is not None:
            # Operator diagnostics only: logged, never posted (#1119).
            for item in blockers:
                if _is_machine_obligation(item):
                    continue
                log(
                    config,
                    f"{_single_line_diagnostic(item.reviewer)} "
                    f"({_single_line_diagnostic(item.item_id)}): reviewer-owned finding",
                )
            for line in _finalization_obligation_lines(
                blockers, current_head_sha=current_head_sha, observations=observations
            ):
                log(config, f"PR #{pr_number} blocking obligation {line}")
        raise AgentLoopError(
            f"PR #{pr_number} cannot finalize: {diagnostic} No approval or merge was attempted."
            + detail
        )


class _RepeatableRoundSequence:
    """Round numbers for the PR loop; the current round may run again (#1068).

    Evidence-response and same-head refresh passes re-run the review board
    inside the current round, so they never advance ``round_number`` and the
    pre-round budget guard cannot reject them.  Each repeat is triggered only
    by a signed requirement ID no reviewer has seen, and a hard cap backs that
    bound so a misbehaving source cannot spin the loop.
    """

    MAX_REPEATS_PER_ROUND = 8

    def __init__(self, start: int, stop: int) -> None:
        self._next = start
        self._stop = stop
        self._current: int | None = None
        self._repeat = False
        self._repeats = 0

    def __iter__(self) -> "_RepeatableRoundSequence":
        return self

    def __next__(self) -> int:
        if self._repeat and self._current is not None:
            self._repeat = False
            return self._current
        if self._next >= self._stop:
            raise StopIteration
        self._current = self._next
        self._next += 1
        self._repeats = 0
        return self._current

    def repeat_current(self) -> None:
        self._repeats += 1
        if self._repeats > self.MAX_REPEATS_PER_ROUND:
            raise HumanDecisionRequiredError(
                f"Round {self._current} was re-run {self.MAX_REPEATS_PER_ROUND} times for new "
                "signed human input without settling; human review required."
            )
        self._repeat = True


def _evidence_barrier_note(items: Sequence[UnresolvedReviewItem]) -> str:
    """Name pending human-only evidence as the remaining final barrier."""
    evidence = _pending_evidence_obligations(items)
    if not evidence:
        return ""
    return (
        " Human-only exact-head evidence remains the final barrier ("
        + ", ".join(f"{item.item_id} from {item.reviewer}" for item in evidence)
        + "); it is requested only once every code finding and machine gate is clean."
    )


def _append_evidence_barrier_note(body: str, items: Sequence[UnresolvedReviewItem]) -> str:
    """Add the evidence note to a clean-stop comment, before its signature."""
    note = _evidence_barrier_note(items).strip()
    if not note:
        return body
    text = str(body)
    if "\n-- " in text:
        prefix, signature = text.rsplit("\n-- ", 1)
        return f"{prefix.rstrip()}\n\n{note}\n\n-- {signature}"
    return f"{text.rstrip()}\n\n{note}"


def _evidence_item_lines(items: Sequence[UnresolvedReviewItem]) -> list[str]:
    return [
        f"- [{item.item_id}] requested by {item.reviewer}: {item.text}"
        for item in items
    ]


def _evidence_freeze_diagnostic(
    *, pr_number: int, head_sha: str | None, items: Sequence[UnresolvedReviewItem]
) -> str:
    evidence = _pending_evidence_obligations(items)
    return (
        f"PR #{pr_number} is frozen at head {head_sha or 'unknown'} awaiting human-only "
        "exact-head evidence: "
        + "; ".join(f"{item.item_id} from {item.reviewer}" for item in evidence)
        + ". No coder, CI qualification, or merge was started. Supply the evidence for "
        "exactly this head, or withdraw the request, in a PR comment whose last line is "
        f"exactly `-- Human Reviewer`, then rerun `agent-loop pr {pr_number}`. Pushing a new "
        "commit breaks the freeze and the new head needs full review and fresh evidence."
    )


def _render_evidence_freeze_notice(
    *,
    pr_number: int,
    head_sha: str,
    items: Sequence[UnresolvedReviewItem],
    still_frozen: bool,
) -> str:
    evidence = _pending_evidence_obligations(items)
    heading = (
        "## Exact-head evidence freeze (still in effect)"
        if still_frozen
        else "## Exact-head evidence freeze"
    )
    return "\n".join(
        [
            heading,
            "",
            f"Every reviewer reports no code findings at head `{head_sha}` and every machine "
            "gate is clean. The remaining barrier is evidence that an agent session cannot "
            "produce:",
            "",
            *_evidence_item_lines(evidence),
            "",
            f"Evidence requested at `{head_sha}`. No further code changes will be accepted "
            "until the authenticated live evidence for this head is supplied or the request "
            "is withdrawn.",
            "",
            "To respond, post a PR comment that supplies the evidence for exactly this head "
            "(or withdraws the request) and ends with a line containing exactly "
            f"`-- Human Reviewer`, then rerun `agent-loop pr {pr_number}`. The requesting "
            "reviewer re-reviews this same head. Pushing a new commit breaks the freeze; the "
            "new head then needs full review and fresh evidence.",
            "",
            "-- coding-review-agent-loop",
        ]
    )


_EVIDENCE_RELEASE_MESSAGES = {
    "machine-gate": (
        "a machine gate failed at the frozen head ({detail}). The requested evidence returns "
        "to deferred; the failure is repaired first and the evidence is requested again at "
        "the next clean head."
    ),
    "findings": (
        "a reviewer raised new findings at the frozen head. The requested evidence returns "
        "to deferred and is requested again at the next clean head; evidence for this head "
        "does not carry forward."
    ),
    "evidence-cleared": (
        "the requesting reviewer(s) accepted the supplied evidence or its withdrawal. The PR "
        "proceeds to final validation at the same head."
    ),
    "refresh-clean": (
        "a same-head refresh for new signed input found no findings. The PR proceeds to "
        "final validation at the same head."
    ),
}


def _publish_evidence_freeze(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    head_sha: str,
    items: Sequence[UnresolvedReviewItem],
    signed_requirement_ids: Sequence[str],
    allowed_rounds: int,
    watch_failure_extension_used: bool,
    watch_head_extension_used: bool,
    clearances: Sequence[tuple[str, str]] = (),
    still_frozen: bool = False,
) -> list[UnresolvedReviewItem]:
    """Publish and persist a freeze with exactly one comment (#1068).

    The frozen ledger exists only in memory until this single write; the
    comment body is the human notice and its round metadata is the only
    persistence.  A failed write therefore leaves no freeze behind.
    """
    frozen = freeze_evidence_obligations(items, head_sha=head_sha)
    identities = tuple(
        item.obligation_identity or item.item_id for item in _frozen_evidence_obligations(frozen)
    )
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=_attach_round_metadata(
            _render_evidence_freeze_notice(
                pr_number=pr_number,
                head_sha=head_sha,
                items=frozen,
                still_frozen=still_frozen,
            ),
            PostedRoundMetadata(
                flow="pr",
                role="summary",
                agent="Orchestrator",
                round_number=round_number,
                subject=head_sha,
                prior_items=tuple(frozen),
                state="blocking",
                phase=EVIDENCE_FREEZE_PHASE,
                evidence_freeze=EvidenceFreezeRecord(
                    frozen_head=head_sha,
                    evidence_identities=identities,
                    signed_requirement_ids_at_freeze=tuple(sorted(set(signed_requirement_ids))),
                    allowed_rounds=allowed_rounds,
                    watch_failure_extension_used=watch_failure_extension_used,
                    watch_head_extension_used=watch_head_extension_used,
                ),
                evidence_clearances=tuple(clearances),
                **_architecture_metadata_fields(config),
            ),
        ),
    )
    return frozen


def _publish_evidence_release(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    head_sha: str,
    items: Sequence[UnresolvedReviewItem],
    reason: str,
    surfaced_requirement_ids: Sequence[str],
    allowed_rounds: int,
    watch_failure_extension_used: bool,
    watch_head_extension_used: bool,
    clearances: Sequence[tuple[str, str]] = (),
    detail: str = "",
) -> list[UnresolvedReviewItem]:
    """Post the single terminal release comment carrying the released ledger."""
    released = release_evidence_freeze(items)
    message = _EVIDENCE_RELEASE_MESSAGES[reason].format(detail=detail or "see the checks comment")
    heading = {
        "evidence-cleared": "## Exact-head evidence accepted",
        "refresh-clean": "## Same-head refresh complete",
    }.get(reason, "## Exact-head evidence freeze released")
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=_attach_round_metadata(
            f"{heading}\n\nAt head `{head_sha}`: {message}"
            + _evidence_barrier_note(released)
            + "\n\n-- coding-review-agent-loop",
            PostedRoundMetadata(
                flow="pr",
                role="summary",
                agent="Orchestrator",
                round_number=round_number,
                subject=head_sha,
                prior_items=tuple(released),
                state="blocking",
                phase=EVIDENCE_RELEASE_PHASE,
                evidence_release=EvidenceReleaseRecord(
                    released_head=head_sha,
                    reason=reason,
                    signed_requirement_ids_surfaced=tuple(sorted(set(surfaced_requirement_ids))),
                    allowed_rounds=allowed_rounds,
                    watch_failure_extension_used=watch_failure_extension_used,
                    watch_head_extension_used=watch_head_extension_used,
                ),
                evidence_clearances=tuple(clearances),
                **_architecture_metadata_fields(config),
            ),
        ),
    )
    return released


@dataclass(frozen=True)
class _EvidenceGateOutcome:
    """What the finalization-point evidence gate decided (#1068)."""

    action: str  # "proceed" | "head_changed" | "refresh"
    context: PullRequestReviewContext | None = None


def _evidence_freeze_gate(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    head_sha: str | None,
    items: Sequence[UnresolvedReviewItem],
    surfaced_requirement_ids: Sequence[str] | None,
    collect_requirement_ids: Callable[[PullRequestReviewContext], Sequence[str]],
    allowed_rounds: int,
    watch_failure_extension_used: bool,
    watch_head_extension_used: bool,
    clearances: Sequence[tuple[str, str]] = (),
) -> _EvidenceGateOutcome:
    """Final revalidation and, when evidence is pending, the single freeze write.

    Runs immediately before ``_ensure_finalization_ready`` and any merge, after
    every machine gate.  The two reads are the last before the write:

    * a live head other than ``head_sha`` returns ``head_changed`` without
      posting anything about evidence;
    * a live signed requirement ID outside the set surfaced to this round's
      reviewers returns ``refresh`` so the board re-runs at the same head
      before any freeze or approval (a missing baseline counts as empty).

    With both matching and evidence pending, the freeze is published and the
    run stops at the human decision boundary.  With nothing pending the
    caller proceeds to finalization.
    """
    fresh = get_pr_review_context(runner, config=config, pr_number=pr_number)
    if not head_sha or fresh.metadata.head_sha != head_sha:
        return _EvidenceGateOutcome("head_changed", fresh)
    live_ids = tuple(collect_requirement_ids(fresh))
    if set(live_ids) - set(surfaced_requirement_ids or ()):
        return _EvidenceGateOutcome("refresh", fresh)
    if not _pending_evidence_obligations(items):
        return _EvidenceGateOutcome("proceed", fresh)
    still_frozen = bool(_frozen_evidence_obligations(items))
    frozen = _publish_evidence_freeze(
        runner,
        config=config,
        pr_number=pr_number,
        round_number=round_number,
        head_sha=head_sha,
        items=items,
        signed_requirement_ids=live_ids,
        allowed_rounds=allowed_rounds,
        watch_failure_extension_used=watch_failure_extension_used,
        watch_head_extension_used=watch_head_extension_used,
        clearances=clearances,
        still_frozen=still_frozen,
    )
    raise HumanDecisionRequiredError(
        _evidence_freeze_diagnostic(pr_number=pr_number, head_sha=head_sha, items=frozen)
    )


def _refuse_dispatch_while_evidence_frozen(
    items: Sequence[UnresolvedReviewItem], *, pr_number: int, operation: str
) -> None:
    """Choke point before any head-changing dispatch (#1068).

    Every legitimate path releases a freeze before dispatching, so this only
    catches bugs: a frozen head must never move under a human evidence run.
    """
    frozen = _frozen_evidence_obligations(items)
    if not frozen:
        return
    raise AgentLoopError(
        f"Refusing to dispatch {operation} while exact-head evidence is frozen. "
        + _evidence_freeze_diagnostic(
            pr_number=pr_number, head_sha=frozen[0].candidate_head_sha, items=frozen
        )
    )


def _is_evidence_only_blocking_review(
    parsed: ParsedReview, prior_items: Sequence[UnresolvedReviewItem]
) -> bool:
    """A blocking review whose only open items are evidence requests.

    Missing human evidence is not a code blocker, so such a review approves
    the code while the evidence obligations stay in the ledger.
    """
    if parsed.state != "blocking" or parsed.blocking_items or parsed.followups.same_pr:
        return False
    evidence_ids = {item.item_id for item in prior_items if _is_evidence_obligation(item)}
    active = [
        disposition
        for disposition in parsed.dispositions
        if disposition.disposition in {"blocking", "same-pr"}
    ]
    if any(disposition.item_id not in evidence_ids for disposition in active):
        return False
    return bool(active) or bool(parsed.exact_head_evidence_requests)


def _evidence_review_context(
    items: Sequence[UnresolvedReviewItem], *, response_head: str | None
) -> str:
    """Reviewer context for carried evidence obligations and response passes."""
    evidence = _pending_evidence_obligations(items)
    if not evidence:
        return ""
    lines = ["", "Human-only exact-head evidence obligations (not code findings):"]
    for item in evidence:
        state = (
            f"frozen at head `{item.candidate_head_sha}`"
            if item.lifecycle == "evidence_frozen"
            else "deferred until every code finding and machine gate is clean"
        )
        lines.append(f"- [{item.item_id}] requested by {item.reviewer}; {state}: {item.text}")
    lines.append(
        "These are machine-owned records. Disposition each one: `blocking` with a short "
        "note keeps a request that is still needed; `resolved` is only a note unless the "
        "response rule below applies. A review whose only open items are kept evidence "
        "requests uses `state: blocking` with empty `blocking_items` and is treated as "
        "approving the code. Never list missing human evidence in `blocking_items`."
    )
    if response_head:
        lines.append(
            f"This is an evidence-response re-review at frozen head `{response_head}` after "
            "new signed human input. If you requested an item, dispose it `resolved` only "
            "when the signed input supplies adequate evidence for exactly this head or "
            "withdraws the request; otherwise keep it `blocking` with a note. Other "
            "reviewers' dispositions on your item are notes only. Do not re-emit a request "
            "you just resolved. Report any code defect the evidence reveals in "
            "`blocking_items`."
        )
    return "\n".join(lines) + "\n"


def _finalize_ordinary_recovery_checked(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    items: Sequence[UnresolvedReviewItem],
    current_head_sha: str | None,
    capability: OrdinaryRecoveryCapability,
) -> bool:
    """Run ordinary recovery only after every unrelated obligation is clear."""
    _ensure_finalization_ready(
        pr_number=pr_number,
        round_number=round_number,
        items=items,
        current_head_sha=current_head_sha,
        ignored_machine_kinds=frozenset({"github-pr-checks"}),
        config=config,
    )
    return _finalize_ordinary_recovery_merge(
        runner,
        config=config,
        pr_number=pr_number,
        capability=capability,
    )


def _mergeability_for_unreadable_protection(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    checks: PullRequestChecks | None,
) -> PullRequestMergeability | None:
    """Fetch GitHub's merge state only when classic protection returned 403."""
    if checks is None or checks.branch_protection_status != "forbidden":
        return None
    return get_pr_mergeability(runner, config=config, pr_number=pr_number)


def _ordinary_snapshot_nonauthority_reason(
    checks: PullRequestChecks | None,
    mergeability: PullRequestMergeability | None = None,
    *,
    head_sha: str | None = None,
    defer_unreadable_protection: bool = False,
) -> str:
    """Name the first unmet conjunct of ``_ordinary_checks_snapshot_is_authoritative``.

    Empty exactly when that predicate is True. Edit both functions together.
    """
    if checks is None:
        return "the check board is unavailable"
    if checks.state != "passing":
        return f"the aggregate state is {checks.state}"
    if checks.check_query_status != "ok":
        return f"the check query status is {checks.check_query_status}"
    if not (
        board_protection_is_reliable(checks, mergeability, head_sha=head_sha)
        or (defer_unreadable_protection and checks.branch_protection_status == "forbidden")
    ):
        merge_state = mergeability.merge_state_raw if mergeability is not None else None
        return (
            f"branch protection is not reliable for the head (protection status "
            f"{checks.branch_protection_status}, merge state {merge_state or 'unknown'})"
        )
    if checks.pending:
        return "checks are still pending: " + ", ".join(c.name for c in checks.pending)
    if checks.missing_required:
        return "required checks are missing: " + ", ".join(checks.missing_required)
    successful = [c for c in checks.passing if c.status.strip().lower() == "success"]
    if not successful:
        return "no check has a real success conclusion (only skipped or neutral)"
    required = set(checks.required_checks)
    success_names = {c.name for c in successful if c.name in required}
    if success_names != required:
        return "required checks without a success conclusion: " + ", ".join(
            sorted(required - success_names)
        )
    for check in checks.passing:
        if check.name in required and check.status.strip().lower() != "success":
            return (
                f"required check {check.name} has a non-success {check.kind} "
                f"observation ({check.status})"
            )
    return ""


def _ordinary_checks_snapshot_is_authoritative(
    checks: PullRequestChecks | None,
    mergeability: PullRequestMergeability | None = None,
    *,
    head_sha: str | None = None,
    defer_unreadable_protection: bool = False,
) -> bool:
    """Return whether a fresh ordinary-check snapshot can clear its ledger.

    ``get_pr_checks`` is queried against the current PR head immediately before
    this decision. Requiring both API surfaces and a known branch-protection
    result keeps a passing-looking partial, absent, or unavailable snapshot
    from becoming a final gate. An unreadable (403) classic protection is
    accepted only with GitHub's ``CLEAN`` merge state for ``head_sha``, unless
    ``defer_unreadable_protection`` says the caller checks ``CLEAN`` itself
    after readiness and before merging (a draft reports ``DRAFT``).

    Keep ``_ordinary_snapshot_nonauthority_reason`` in step with this predicate.
    """
    # ``PullRequestChecks.state`` deliberately treats neutral and skipped
    # conclusions as passing for ordinary status reporting.  That aggregate is
    # useful for display, but it is not evidence that a final gate actually
    # ran.  A clearing snapshot must contain a real success conclusion; when
    # branch protection names required checks, each of those checks must also
    # have that conclusion.
    successful_checks = tuple(
        check for check in (checks.passing if checks is not None else ())
        if check.status.strip().lower() == "success"
    )
    required_names = set(checks.required_checks) if checks is not None else set()
    required_success_names = {
        check.name for check in successful_checks if check.name in required_names
    }
    required_observations = (
        check for check in (checks.passing if checks is not None else ())
        if check.name in required_names
    )
    return bool(
        checks is not None
        and checks.state == "passing"
        and checks.check_query_status == "ok"
        and (
            board_protection_is_reliable(checks, mergeability, head_sha=head_sha)
            or (
                defer_unreadable_protection
                and checks.branch_protection_status == "forbidden"
            )
        )
        and not checks.pending
        and not checks.missing_required
        and successful_checks
        and required_success_names == required_names
        and all(
            check.status.strip().lower() == "success"
            for check in required_observations
        )
    )


def _managed_success_supersedes_ordinary_checks(
    items: Sequence[UnresolvedReviewItem],
    *,
    outcome: ManagedCiOutcome,
    current_head_sha: str | None,
    mergeability: PullRequestMergeability | None,
) -> tuple[str, str]:
    """Decide whether a correlated managed success retires carried ordinary checks (#1117).

    Returns ``(verdict, predicate)`` with verdict ``not_applicable``, ``cleared``
    or ``unqualified``. Only a ``github-pr-checks`` revalidation candidate at the
    qualified head is considered; the judgement is made from the full,
    unfiltered exact-head board and never from obligation text.
    """
    candidates = [
        item
        for item in items
        if item.obligation_kind == "github-pr-checks"
        and _machine_obligation_is_revalidation_candidate(item, current_head_sha=current_head_sha)
    ]
    if not candidates:
        return "not_applicable", ""
    checks = outcome.checks
    if checks is None or not current_head_sha or outcome.head_sha != current_head_sha:
        return "unqualified", "the qualified exact-head check board is unavailable or not for the current head"
    if checks.check_query_status != "ok" or checks.check_query_errors:
        errors = "; ".join(checks.check_query_errors) or checks.check_query_status
        return "unqualified", f"the exact-head check board had query errors ({errors})"
    if not checks.listing_complete:
        return "unqualified", "the exact-head check listing is incomplete (total_count mismatch or unparsed entries)"
    final = [check for check in (*checks.passing, *checks.pending, *checks.failing) if check.name == FINAL_CONTEXT]
    if not any(check.status.strip().lower() == "success" for check in final):
        return "unqualified", f"`{FINAL_CONTEXT}` has no success observation"
    if checks.failing or checks.pending or checks.missing_required:
        names = [
            check.name for check in (*checks.failing, *checks.pending)
        ] + list(checks.missing_required)
        return "unqualified", f"checks are failing, pending or missing at the qualified head: {', '.join(names)}"
    required = set(checks.required_checks)
    for check in checks.shadowed:
        state = _classify_check_status(check.status)
        if state in {"failing", "pending"}:
            return "unqualified", f"a same-name observation of `{check.name}` is {check.status.lower()}"
        if check.name in required and check.status.strip().lower() != "success":
            return "unqualified", (
                f"required check `{check.name}` has a same-name observation that is {check.status.lower()}"
            )
    if not _ordinary_checks_snapshot_is_authoritative(checks, mergeability, head_sha=current_head_sha):
        if protection_awaits_readiness(checks, mergeability, head_sha=current_head_sha):
            return "unqualified", (
                f"branch protection is unreadable (HTTP 403) and GitHub reports DRAFT for draft PR at "
                f"{current_head_sha}, so required contexts cannot be verified before readiness; grant the "
                "token read access to branch protection (administration: read), then resume. Do not mark "
                "the PR ready manually: a ready PR that still carries the managed label is a mixed "
                "lifecycle that managed resume refuses"
            )
        if checks.branch_protection_status == "forbidden":
            state = mergeability.merge_state_raw if mergeability is not None else None
            return "unqualified", (
                "branch protection is unreadable (HTTP 403) and GitHub's merge state is "
                f"{state or 'unavailable'} rather than CLEAN for {current_head_sha}"
            )
        for observed in checks.passing:
            if observed.name in required and observed.status.strip().lower() != "success":
                return "unqualified", (
                    f"required check `{observed.name}` is {observed.status.lower()} rather than a "
                    "real success"
                )
        for name in sorted(required):
            if not any(
                observed.name == name and observed.status.strip().lower() == "success"
                for observed in checks.passing
            ):
                return "unqualified", f"required check `{name}` has no real success at the qualified head"
        return "unqualified", "the exact-head board is not an authoritative success-only snapshot"
    return "cleared", ""


def _persist_qualification_checkpoint(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    head_sha: str | None,
    unresolved_items: Sequence[UnresolvedReviewItem],
    checkpoint: QualificationCheckpoint | None,
    message: str,
) -> None:
    """Write a resumable round-metadata checkpoint before risky continuation."""
    if checkpoint is None:
        return
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=_attach_round_metadata(
            message,
            PostedRoundMetadata(
                flow="pr",
                role="summary",
                agent="Orchestrator",
                round_number=round_number,
                subject=str(head_sha or "unknown"),
                prior_items=tuple(unresolved_items),
                state="blocking",
                phase="qualification-checkpoint",
                qualification_checkpoint=checkpoint,
                **_architecture_metadata_fields(config),
            ),
        ),
    )


def _visibility_snapshot(
    *,
    fresh_records: Sequence[PostedRoundRecord] | None,
    base_records: Sequence[PostedRoundRecord],
    base_length: int,
    checkpoint: PostedRoundMetadata | None,
) -> tuple[tuple[PostedRoundRecord, ...], bool]:
    """History as visible once this invocation's own scheduler checkpoint is posted (#1156).

    Prefers the refreshed history when it holds the checkpoint; otherwise the
    pre-post history plus a synthetic record for it, never the stale evidence
    alone.  Returns the records and whether the checkpoint is present.
    """
    if checkpoint is None:
        return tuple(base_records), False
    base_top = max((record.index for record in base_records), default=-1)
    if fresh_records is None:
        history: tuple[PostedRoundRecord, ...] = tuple(base_records)
    else:
        history = tuple(fresh_records)
        if any(
            record.index > base_top
            and record.metadata.flow == checkpoint.flow
            and record.metadata.role == "summary"
            and record.metadata.phase == "scheduler-prelaunch"
            and record.metadata.round_number == checkpoint.round_number
            and record.metadata.subject == checkpoint.subject
            for record in history
        ):
            return history, True
    # The refreshed history (when readable) keeps every publication it
    # exposed; only the checkpoint this invocation just posted is added.
    top = max((record.index for record in history), default=-1)
    synthetic = PostedRoundRecord(
        index=max(base_length, base_top + 1, top + 1), metadata=checkpoint, body=""
    )
    return (*history, synthetic), True


def _latest_pr_reviewer_records(
    records: Sequence[PostedRoundRecord],
    configured_reviewers: Sequence[AgentName],
) -> dict[str, PostedRoundRecord]:
    """Return the latest persisted review for every configured reviewer.

    This intentionally searches all candidate subjects.  The exact-head
    approval helper cannot be used here because a stale-head record is the
    evidence needed to decide whether a returning reviewer needs a fresh
    context and session.
    """
    configured_names = {agent_display_name(reviewer) for reviewer in configured_reviewers}
    latest: dict[str, PostedRoundRecord] = {}
    for record in reversed(records):
        if record.metadata.role != "reviewer" or record.metadata.agent not in configured_names:
            continue
        latest.setdefault(record.metadata.agent, record)
    return latest


def _reviewer_needs_fresh_context(
    reviewer: AgentName,
    *,
    selective_policy: bool,
    current_head_sha: str,
    current_round: int,
    latest_reviewer_records: Mapping[str, PostedRoundRecord],
) -> bool:
    """Identify a reviewer that must receive a new full-context turn.

    A reviewer with no persisted review, a review of an older head, or a
    review that missed an intervening round cannot safely reuse a compact
    session.  In particular, the lookup must not be limited to exact-head
    approvals: those records intentionally exclude the stale record we need
    to inspect.
    """
    if not selective_policy:
        return False
    record = latest_reviewer_records.get(agent_display_name(reviewer))
    if record is None:
        return True
    return (
        record.metadata.subject != current_head_sha
        or record.metadata.round_number < current_round - 1
    )


def _reviewer_diff_summary(
    runner: Runner,
    *,
    checkout: Path,
    last_reviewed_sha: str | None,
    current_head_sha: str,
) -> str:
    """Render a bounded, orchestrator-observed diff summary for a returner."""
    if not last_reviewed_sha:
        return "No prior reviewer SHA was recorded; inspect the complete base-to-head diff."
    if last_reviewed_sha == current_head_sha:
        return "The last reviewed SHA is the current head; inspect the complete base-to-head diff for any missed context."
    try:
        result = runner.run(
            ["git", "diff", "--name-status", "--find-renames", "--find-copies", last_reviewed_sha, current_head_sha],
            cwd=checkout,
            check=False,
        )
        if result.returncode != 0:
            return (
                f"The diff from {last_reviewed_sha} to {current_head_sha} was unavailable; "
                "inspect the complete base-to-head diff independently."
            )
        paths = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        if not paths:
            return (
                f"The diff from {last_reviewed_sha} to {current_head_sha} had no name-status output; "
                "inspect the complete base-to-head diff independently."
            )
        displayed = paths[:32]
        suffix = "; ..." if len(paths) > len(displayed) else ""
        return (
            f"Observed diff from {last_reviewed_sha} to {current_head_sha}: "
            + "; ".join(displayed)
            + suffix
            + ". Inspect the complete base-to-head diff independently."
        )
    except (OSError, AttributeError, TypeError):
        return (
            f"The diff from {last_reviewed_sha} to {current_head_sha} could not be observed; "
            "inspect the complete base-to-head diff independently."
        )


def _reviewer_history_is_reconstructible(
    runner: Runner,
    *,
    checkout: Path,
    record: PostedRoundRecord | None,
    current_head_sha: str,
) -> bool:
    """Check that a returning reviewer's own history can be reconstructed.

    A reviewer may legitimately miss several narrow coder rounds. Their older
    reviewed SHA is still sufficient when Git can observe that SHA as an
    ancestor and can produce the complete span diff. This check is intentionally
    independent of the scheduler's previous transition SHA: trusting that SHA
    here would turn normal selective pauses into alternating full-board rounds.
    """
    if record is None:
        return False
    last_sha = record.metadata.subject
    if not last_sha or last_sha == current_head_sha:
        return bool(last_sha)
    try:
        ancestry = runner.run(
            ["git", "merge-base", "--is-ancestor", last_sha, current_head_sha],
            cwd=checkout,
            check=False,
        )
        if ancestry.returncode != 0:
            return False
        diff = runner.run(
            [
                "git", "diff", "--name-status", "--find-renames", "--find-copies",
                last_sha, current_head_sha,
            ],
            cwd=checkout,
            check=False,
        )
        return diff.returncode == 0
    except (OSError, AttributeError, TypeError):
        return False


def _returning_reviewer_context(
    runner: Runner,
    *,
    reviewer: AgentName,
    checkout: Path,
    current_head_sha: str,
    current_round: int,
    latest_reviewer_records: Mapping[str, PostedRoundRecord],
) -> str:
    record = latest_reviewer_records.get(agent_display_name(reviewer))
    last_sha = record.metadata.subject if record is not None else None
    if record is None:
        history = "No prior review record is available for this reviewer."
    else:
        missed_rounds = max(0, current_round - record.metadata.round_number - 1)
        history = (
            f"Last reviewed SHA: {record.metadata.subject}; "
            f"last attended round: {record.metadata.round_number}."
            + (
                f" Missed {missed_rounds} intervening review round(s); reconstruct context from the current history."
                if missed_rounds else ""
            )
        )
    return (
        "\nReturning reviewer handoff context (orchestrator-derived):\n"
        f"{history}\n"
        + _reviewer_diff_summary(
            runner,
            checkout=checkout,
            last_reviewed_sha=last_sha,
            current_head_sha=current_head_sha,
        )
        + "\nYour review must inspect the complete current base-to-head diff; the coder's summary is not a substitute.\n"
    )


def _all_pending_resolution_owners_unavailable(
    item: UnresolvedReviewItem,
    unavailable_names: set[str],
) -> bool:
    """Return true only when a non-empty pending owner set is unavailable."""
    owners = item.resolution_owners or (item.reviewer,)
    states = dict(item.owner_states or ((owner, "pending") for owner in owners))
    pending = [owner for owner in owners if states.get(owner) != "cleared"]
    return bool(pending) and all(owner in unavailable_names for owner in pending)


def _observe_pr_transition(
    runner: Runner,
    *,
    checkout: Path,
    previous_sha: str | None,
    current_sha: str | None,
    scopes: Sequence[str],
    broad_rules: Sequence[str],
    obligations: Sequence[ReviewObligation] = (),
) -> TransitionClassification:
    """Collect repository-observed ancestry and exact diff facts."""
    if not previous_sha or not current_sha or previous_sha == current_sha:
        return TransitionClassification("broad", "missing or unchanged review SHA")
    try:
        ancestry = runner.run(
            ["git", "merge-base", "--is-ancestor", previous_sha, current_sha],
            cwd=checkout,
            check=False,
        )
        if ancestry.returncode != 0:
            return TransitionClassification("broad", "candidate history is not an available ancestor")
        names = runner.run(
            [
                "git", "diff", "--name-status", "-z", "--find-renames", "--find-copies",
                previous_sha, current_sha,
            ],
            cwd=checkout,
            check=False,
        )
        if names.returncode != 0:
            return TransitionClassification("broad", "complete exact Git diff was unavailable")
        raw = names.stdout
        if not raw:
            return TransitionClassification("broad", "transition contained no observable diff")
        tokens = raw.split("\0")
        changes: list[GitChange] = []
        index = 0
        while index < len(tokens):
            status = tokens[index]
            index += 1
            if not status:
                continue
            code = status[:1].upper()
            if code in {"R", "C"}:
                # NUL-formatted rename/copy entries carry old and new paths.
                if index + 1 >= len(tokens):
                    return TransitionClassification("broad", "diff contained an incomplete rename/copy record")
                old_path, new_path = tokens[index], tokens[index + 1]
                index += 2
                changes.append(GitChange(old_path, status=status))
                changes.append(GitChange(new_path, status=status))
            else:
                if index >= len(tokens):
                    return TransitionClassification("broad", "diff contained an incomplete name-status record")
                changes.append(GitChange(tokens[index], status=status))
                index += 1
        # A binary check is separate from name-status because a binary file can
        # otherwise look exactly like an ordinary modification.
        numstat = runner.run(
            ["git", "diff", "--numstat", previous_sha, current_sha],
            cwd=checkout,
            check=False,
        )
        if numstat.returncode != 0:
            return TransitionClassification("broad", "binary/text diff classification was unavailable")
        binary_paths = {
            line.rsplit("\t", 1)[-1]
            for line in numstat.stdout.splitlines()
            if line.startswith("-\t-\t")
        }
        mode = runner.run(
            ["git", "diff", "--summary", previous_sha, current_sha],
            cwd=checkout,
            check=False,
        )
        if mode.returncode != 0:
            return TransitionClassification("broad", "diff mode-change classification was unavailable")
        added_paths = {
            change.path
            for change in changes
            if change.status[:1].upper() == "A"
        }
        unsafe_summary = False
        for summary_line in mode.stdout.splitlines():
            lowered = summary_line.lower()
            create_marker = "create mode 100644 "
            create_index = lowered.find(create_marker)
            if create_index >= 0:
                # Git reports the mode of an ordinary newly-added text file in
                # the summary.  The name-status record is authoritative for
                # distinguishing that safe A entry from an executable,
                # symlink, or other mode/type change.
                created_path = summary_line[create_index + len(create_marker):].strip()
                if created_path in added_paths:
                    continue
            if any(
                token in lowered
                for token in ("mode change", "create mode", "delete mode", "submodule", "rename", "copy")
            ):
                unsafe_summary = True
                break
        if binary_paths or unsafe_summary:
            return TransitionClassification("broad", "diff contained binary or mode changes")
        return classify_transition(
            previous_sha,
            current_sha,
            changes,
            scopes=scopes,
            broad_rules=broad_rules,
            obligations=obligations,
        )
    except (OSError, AttributeError, TypeError):
        return TransitionClassification("broad", "repository history or diff observation failed")


def _scheduler_contract_from_metadata(
    metadata: PostedRoundMetadata,
) -> ReviewSchedulingContract | None:
    if metadata.scheduler_contract is None:
        return None
    from .review_scheduling import ReviewSchedulingContract as _Contract

    return _Contract.from_mapping(metadata.scheduler_contract)


def _pr_contract_drift_error(
    persisted: ReviewSchedulingContract,
    detail: str,
    *,
    pr_number: int,
    configured: ReviewSchedulingContract | None,
    start_round_number: Callable[[], int | None],
    during: str = "resume",
    amendments_recognized: bool = False,
) -> AgentLoopError:
    """The fail-closed PR contract-drift error, with the amendment route (#943)."""
    return AgentLoopError(
        f"PR review scheduler contract changed during {during}; "
        + ("no qualification or merge is permitted; " if during == "qualification" else "")
        + "required reviewers, policy, and broad-path rules must remain immutable. PR "
        f"#{pr_number} carries a scheduler contract for policy {persisted.policy} with "
        f"primary {persisted.primary_reviewer or '(none)'} and reviewer board "
        f"{', '.join(persisted.required_reviewers)} ({detail})."
        + (
            ""
            if during == "qualification"
            else _stale_amendment_repost_clause(detail, start_round_number)
            + _board_amendment_route_clause(
                flow="pr",
                issue_number=None,
                pr_number=pr_number,
                persisted=persisted,
                configured=configured,
                start_round_number=start_round_number,
                amendments_recognized=amendments_recognized,
            )
        )
    )


REDUCED_BOARD_COMPLETION_HEADING = "Review completed on a reduced reviewer board."


def _post_reduced_board_completion_note(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    digest: str,
    note: str,
) -> None:
    """Post one plain completion note per amendment digest (no round metadata)."""
    try:
        comments = get_pr_review_context(runner, config=config, pr_number=pr_number).comments
    except AgentLoopError:
        comments = ()
    if any(
        REDUCED_BOARD_COMPLETION_HEADING in (getattr(comment, "body", "") or "")
        and digest in (getattr(comment, "body", "") or "")
        for comment in comments
    ):
        return
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=(
            f"{REDUCED_BOARD_COMPLETION_HEADING}\n\n- {note}\n"
            f"- Signed amendment digest: `{digest}`\n\n-- Orchestrator"
        ),
    )


def _pr_amendment_start_round(
    pr_context: PullRequestReviewContext,
    configured_reviewers: Sequence[AgentName],
    scheduler_capabilities: object,
) -> int:
    """The round a PR resume would re-enter; only used to fill an error template."""
    resumed = _resume_pr_round(
        pr_context.comments,
        head_sha=pr_context.metadata.head_sha,
        configured_reviewers=configured_reviewers,
        reconciliation_mode=(
            "owner-scoped"
            if getattr(scheduler_capabilities, "owner_scoped_reconciliation", False)
            else "aggregate"
        ),
    )
    return resumed.round_number if resumed is not None else 1


def _is_completed_full_board_scheduler_record(
    record: PostedRoundRecord,
    *,
    scheduler_contract: ReviewSchedulingContract,
) -> bool:
    """Return whether ``record`` is a completed conservative recovery point.

    Invalid scheduler metadata must force the next scheduling decision to the
    full board.  Once that decision has completed, however, the invalid record
    is historical state and must not permanently veto qualification on every
    resume.  The reconciliation checkpoint is the durable completion marker;
    a prelaunch checkpoint or an individual reviewer record is not sufficient.
    """
    metadata = record.metadata
    if (
        metadata.scheduler_metadata_status != "valid"
        or metadata.role != "summary"
        or metadata.phase != "reconciliation"
        or metadata.scheduler_current_sha != metadata.subject
        or set(metadata.scheduler_selected_reviewers)
        != set(scheduler_contract.required_reviewers)
        or metadata.scheduler_paused_reviewers
    ):
        return False
    try:
        return _scheduler_contract_from_metadata(metadata) == scheduler_contract
    except AgentLoopError:
        return False


def _managed_binding_retired_plan_hashes(
    handoff: AuthenticatedIssueCreatedHandoff | None,
) -> frozenset[str]:
    """Plans a verified signed rebind retired for this managed handoff (#993)."""
    return handoff.retired_plan_hashes if handoff is not None else frozenset()


def _managed_binding_protection_mode(
    handoff: AuthenticatedIssueCreatedHandoff | None,
) -> str | None:
    """Return ``strict`` only for a handoff that publishes no PR-side record.

    Authorization records are skipped exactly when the authenticated handoff
    carries no override nonce; any other handoff must bind through them.
    """
    if handoff is not None and handoff.protection_mode == "strict" and handoff.override_nonce is None:
        return "strict"
    return None


_PR_AMENDMENT_PLAN_BOARD_HINT = (
    " Rerun with the original reviewer board configured; the signed PR amendment still "
    "removes the reviewer from PR review."
)


def _pr_amendment_plan_board_hint(
    pr_comments: Sequence[object],
    *,
    pr_number: int,
    supplied_reviewers: Sequence[AgentName],
) -> str:
    """Rerun hint when a PR-only amendment explains a reduced supplied board (#1133)."""
    try:
        amendments = collect_reviewer_board_amendments(
            pr_comments, flow="pr", pr_number=pr_number
        )
    except AgentLoopError:
        return ""
    supplied = {agent_display_name(reviewer) for reviewer in supplied_reviewers}
    for amendment in amendments:
        original = set(amendment.original_required_reviewers)
        if any(name not in supplied for name in amendment.removed_reviewers) and supplied <= original:
            return _PR_AMENDMENT_PLAN_BOARD_HINT
    return ""


def _verify_strict_managed_plan_binding(
    *,
    config: AgentLoopConfig,
    pr_number: int,
    issue_context: IssueContext,
    metadata: PullRequestMetadata,
    expected_plan_hash: str,
    pr_comments: Sequence[object] = (),
) -> None:
    """Bind a strict-protection managed PR to the issue's canonical plan.

    A strictly protected base never publishes a PR-side authorization record,
    and managed recovery never synthesizes the issue-side handoff.  The durable
    binding is therefore the one the managed resume itself used: the reserved
    managed branch for this issue and the issue's canonical, completely
    approved plan, whose hash must still be the plan the reviewers were bound
    to.  GitHub's exact-head protection independently gates the merge.
    """

    def fail(reason: str) -> AgentLoopError:
        return AgentLoopError(
            "Approved-plan/handoff identity changed or disappeared during PR qualification; "
            f"the strict managed-CI binding does not tie PR #{pr_number} to approved plan "
            f"{expected_plan_hash} ({reason}). Stale approvals cannot be used for this head."
            + _pr_amendment_plan_board_hint(
                pr_comments, pr_number=pr_number, supplied_reviewers=reviewers(config)
            )
        )

    if metadata.head_branch != f"agent-loop/managed-{issue_context.number}" or not metadata.head_sha:
        raise fail(f"the PR is not the reserved managed branch for issue #{issue_context.number}")
    resumed_plan = _resume_plan_round(
        issue_context.comments,
        configured_reviewers=reviewers(config),
    )
    if resumed_plan is None:
        raise fail("the issue carries no canonical approved plan")
    plan_text, plan_round = resumed_plan
    _require_complete_canonical_plan_approval(
        issue_context.comments,
        config=config,
        plan_text=plan_text,
        plan_round=plan_round,
        human_requirements=issue_context.human_requirements,
        error_message=str(fail("the canonical plan is not completely approved")),
    )
    if approved_plan_hash(plan_text) != expected_plan_hash:
        raise fail("the issue's canonical approved plan changed")


def _fresh_pr_qualification_snapshot(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    issue_context: IssueContext | None,
    parent_issue_context: IssueContext | None,
    approved_plan_context: ApprovedPlanContext | None = None,
    scheduler_contract: ReviewSchedulingContract | None = None,
    allow_plan_handoff_change: bool = False,
    planning_child_binding: _PlanningChildBinding | None = None,
    managed_protection_mode: str | None = None,
    managed_retired_plan_hashes: frozenset[str] = frozenset(),
    plan_binding_reviewers: Sequence[AgentName] | None = None,
) -> tuple[PullRequestReviewContext, tuple[str, ...], ApprovedPlanContext | None, AgentLoopConfig]:
    """Refetch the PR-side qualification inputs immediately before a gate.

    ``plan_binding_reviewers`` is the operator-supplied board the issue plan is
    re-verified against; it differs from ``config.reviewer`` when a signed PR
    amendment reduced the effective PR board (#1133).
    """
    staged_owner = (
        approved_plan_context.risk_test_matrix_execution_owner
        if approved_plan_context is not None
        else None
    )
    context = get_pr_review_context(runner, config=config, pr_number=pr_number)
    approved_identity = _latest_pr_approval_architecture_identity(
        context.comments, head_sha=context.metadata.head_sha
    )
    stored_identity = _latest_pr_architecture_observation(
        context.comments, head_sha=context.metadata.head_sha
    )
    if stored_identity is None:
        # A reviewer approval is the durable resume authority when no
        # qualification checkpoint has recorded a newer observation. Never
        # invent an identity for legacy records.
        stored_identity = approved_identity
    if stored_identity is None:
        # There may be no current-head approval yet. Retain the latest durable
        # observation as a compatibility fallback, but never use raw prose.
        stored_identity = next(
            (
                record.metadata.architecture_identity
                for record in reversed(_extract_round_metadata_records(context.comments, flow="pr"))
                if isinstance(record.metadata.architecture_identity, dict)
            ),
            None,
        )
    fresh_architecture, architecture_changed = _revalidate_pr_architecture_identity(
        runner, config=config, metadata=context.metadata, stored_identity=stored_identity
    )
    if fresh_architecture is not None:
        # AgentLoopConfig is frozen and hashable.  Rebind a replacement config
        # instead of mutating the captured instance during a qualification gate.
        config = dataclasses_replace(config, architecture_context=fresh_architecture)
    if architecture_changed and fresh_architecture is not None:
        context = dataclasses_replace(context, architecture_identity_changed=True)
    if scheduler_contract is not None:
        # Scheduler metadata is an optimization over the immutable required
        # reviewer contract. A fresh qualification read must never silently
        # accept malformed or conflicting audit state, including a record
        # posted while managed CI was running.
        fresh_scheduler_records = _extract_round_metadata_records(
            context.comments, flow="pr"
        )
        superseded_invalid_indexes: set[int] = set()
        for record_position, record in enumerate(fresh_scheduler_records):
            status = record.metadata.scheduler_metadata_status
            if status == "invalid":
                if any(
                    candidate.index > record.index
                    and _is_completed_full_board_scheduler_record(
                        candidate,
                        scheduler_contract=scheduler_contract,
                    )
                    for candidate in fresh_scheduler_records[record_position + 1 :]
                ):
                    # Resume recovery deliberately selected the full board.
                    # A later completed reconciliation checkpoint supersedes
                    # this historical malformed optimization record.
                    superseded_invalid_indexes.add(record.index)
                    continue
                raise AgentLoopError(
                    "Malformed or contradictory PR review scheduler metadata was observed "
                    "during qualification; no qualification or merge is permitted."
                )
            if status != "valid":
                continue
            try:
                _scheduler_contract_from_metadata(record.metadata)
            except AgentLoopError as exc:
                raise AgentLoopError(
                    "Malformed PR review scheduler contract was observed during qualification; "
                    "no qualification or merge is permitted."
                ) from exc
            if record.metadata.scheduler_current_sha != record.metadata.subject:
                raise AgentLoopError(
                    "Contradictory PR review scheduler head metadata was observed during "
                    "qualification; no qualification or merge is permitted."
                )
        # Amendments are re-read from this same fresh PR comment fetch (#943),
        # so the gate never relies on a stale or separately refreshed source.
        # Pre-amendment contracts are accepted only by the lineage rules, and
        # a post-amendment record only with the exact amendment digest.
        try:
            # Every fresh record takes part, contract-neutral ones included,
            # so a digest on a coder or reviewer record posted during managed
            # CI fails closed; only an invalid optimization record already
            # superseded by the recovery rule above is left out.
            fresh_amendment_diagnostics: list[str] = []
            fresh_amendments = collect_reviewer_board_amendments(
                context.comments,
                flow="pr",
                pr_number=pr_number,
                ignored_sink=fresh_amendment_diagnostics,
            )
            for diagnostic in fresh_amendment_diagnostics:
                log(config, f"PR #{pr_number}: {diagnostic}")
            resolve_contract_lineage(
                tuple(
                    record
                    for record in fresh_scheduler_records
                    if record.index not in superseded_invalid_indexes
                ),
                fresh_amendments,
                scheduler_contract,
                contract_from_metadata=_scheduler_contract_from_metadata,
                drift_error=lambda persisted, detail: _pr_contract_drift_error(
                    persisted,
                    detail,
                    pr_number=pr_number,
                    configured=scheduler_contract,
                    start_round_number=lambda: None,
                    during="qualification",
                ),
            )
        except AgentLoopError as exc:
            if "no qualification or merge is permitted" in str(exc):
                raise
            raise AgentLoopError(
                f"{exc} No qualification or merge is permitted."
            ) from exc
    fresh_issue = issue_context
    fresh_parent = parent_issue_context
    fresh_approved_plan_context = approved_plan_context
    if issue_context is not None:
        fresh_issue = get_issue_context(runner, config=config, issue_number=issue_context.number)
        if scheduler_contract is not None:
            try:
                reject_misplaced_pr_amendments(
                    fresh_issue.comments, issue_number=fresh_issue.number
                )
            except AgentLoopError as exc:
                raise AgentLoopError(
                    f"{exc} No qualification or merge is permitted."
                ) from exc
    if parent_issue_context is not None:
        fresh_parent = get_issue_context(
            runner, config=config, issue_number=parent_issue_context.number
        )
    if approved_plan_context is not None:
        if fresh_issue is None or not approved_plan_context.plan_hash:
            raise AgentLoopError(
                "Approved-plan/handoff identity is missing during PR qualification; "
                "stale approvals cannot be used for this head."
            )
        fresh_handoff = find_latest_issue_pr_handoff(
            fresh_issue.comments,
            issue_number=fresh_issue.number,
            repo=config.repo,
        )
        if fresh_handoff is None and config.managed_ci:
            # Managed-CI recovery deliberately does not synthesize the
            # issue-side handoff (#966).  A voluntary or plan-limited base binds
            # through the trusted PR-side authorization chain; a strictly
            # protected base publishes no such record, so it is bound the way
            # its resume was: reserved branch plus the issue's canonical plan.
            if managed_protection_mode == "strict":
                _verify_strict_managed_plan_binding(
                    config=(
                        config
                        if plan_binding_reviewers is None
                        else dataclasses_replace(
                            config, reviewer=tuple(plan_binding_reviewers)
                        )
                    ),
                    pr_number=pr_number,
                    issue_context=fresh_issue,
                    metadata=context.metadata,
                    expected_plan_hash=approved_plan_context.plan_hash,
                    pr_comments=context.comments,
                )
            else:
                verify_managed_pr_plan_binding(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    issue_number=fresh_issue.number,
                    live_head=context.metadata.head_sha,
                    approved_plan_hash=approved_plan_context.plan_hash,
                    retired_plan_hashes=managed_retired_plan_hashes,
                )
        elif (
            fresh_handoff is None
            or fresh_handoff.pr_number != pr_number
            or fresh_handoff.flow != "approved-plan-implementation"
        ):
            raise AgentLoopError(
                "Approved-plan/handoff identity changed or disappeared during PR qualification; "
                "stale approvals cannot be used for this head."
            )
        if fresh_handoff is not None and fresh_handoff.plan_hash != approved_plan_context.plan_hash:
            if not allow_plan_handoff_change:
                raise AgentLoopError(
                    "Approved-plan/handoff identity changed or disappeared during PR qualification; "
                    "stale approvals cannot be used for this head."
                )
            replacement_plan: ApprovedPlanContext | None = None
            plan_candidates = [fresh_issue.comments]
            if fresh_parent is not None:
                plan_candidates.append(fresh_parent.comments)
            for comments in plan_candidates:
                candidate = recover_approved_plan_context(
                    comments,
                    expected_hash=fresh_handoff.plan_hash,
                )
                if candidate.is_available:
                    replacement_plan = candidate
                    break
            if replacement_plan is None:
                raise AgentLoopError(
                    "Approved-plan/handoff identity changed during PR qualification, but the "
                    "replacement approved plan could not be recovered; stale approvals cannot "
                    "be used for this head."
                )
            if planning_child_binding is not None:
                # Mid-run adoption (#936): a planning child's replacement
                # plan passes the same rebind verifier and admissibility rule
                # as the entry paths before it can reach a final sweep, merge,
                # or managed-CI gate.  Non-child PRs keep today's behavior.
                verified_replacement = verify_child_plan_rebind(
                    fresh_issue.comments,
                    repo=config.repo,
                    parent_plan_context=planning_child_binding.parent_plan_context,
                    child_issue=planning_child_binding.child_issue,
                    parent_issue=planning_child_binding.parent_issue,
                    stage_id=planning_child_binding.stage_id,
                    pr_number=pr_number,
                )
                if (
                    verified_replacement is None
                    or verified_replacement.plan_hash != fresh_handoff.plan_hash
                ):
                    raise AgentLoopError(
                        f"Human repair required: the approved-plan handoff for child issue "
                        f"#{planning_child_binding.child_issue} changed to plan "
                        f"{fresh_handoff.plan_hash} while PR #{pr_number} was under review, but "
                        "it is not a verified same-PR plan replacement with a rebind audit "
                        "record; no final sweep, merge, or managed-CI gate ran."
                    )
            fresh_approved_plan_context = replacement_plan
        # The handoff hash alone is not enough: recover the canonical plan
        # again from the freshly fetched issue/parent comments and require the
        # same hash and subject that the reviewers were bound to.
        plan_candidates = [fresh_issue.comments]
        if fresh_parent is not None:
            plan_candidates.append(fresh_parent.comments)
        recovered_plan = next(
            (
                candidate
                for comments in plan_candidates
                if (
                    candidate := recover_approved_plan_context(
                        comments,
                        expected_hash=fresh_approved_plan_context.plan_hash,
                        expected_subject=(
                            fresh_approved_plan_context.plan_subject
                            if fresh_approved_plan_context.plan_hash == approved_plan_context.plan_hash
                            else None
                        ),
                    )
                ).is_available
            ),
            None,
        )
        if (
            recovered_plan is None
            or recovered_plan.plan_hash != fresh_approved_plan_context.plan_hash
            or (
                fresh_approved_plan_context.plan_hash == approved_plan_context.plan_hash
                and recovered_plan.plan_subject != approved_plan_context.plan_subject
            )
        ):
            raise AgentLoopError(
                "Approved plan identity changed or disappeared during PR qualification; "
                "stale approvals cannot be used for this head."
            )
        if planning_child_binding is not None:
            # A signed supersession posted while reviewers ran must not be
            # bypassed by qualifying the plan it authorizes replacing (#985).
            _reject_pending_child_plan_supersession(
                fresh_issue.comments,
                binding=planning_child_binding,
                plan_hash=fresh_approved_plan_context.plan_hash,
                pr_number=pr_number,
                stopped="no final sweep, merge, or managed-CI gate ran",
            )
    if (
        staged_owner
        and fresh_approved_plan_context is not None
        and fresh_approved_plan_context.matrix_available
    ):
        fresh_approved_plan_context = scope_approved_plan_matrix(
            fresh_approved_plan_context,
            execution_owner=staged_owner,
        )
    requirements = _build_requirements_context(
        target_issue_context=fresh_issue,
        pr_context=context,
        parent_issue_context=fresh_parent,
    )
    result = (
        requirement.requirement_id
        for requirement in requirements.effective_requirements
    )
    requirement_ids = tuple(result)
    return context, requirement_ids, fresh_approved_plan_context, config


def _preserve_issue_created_managed_suppression(
    contract: ManagedCiContract | None,
    *,
    active_exception: BaseException | None,
) -> bool:
    """Keep tool-created PRs suppressed when orchestration is interrupted."""

    return bool(
        active_exception is not None
        and contract is not None
        and (
            contract.issue_created_pr
            or contract.origin in {"issue-created", "source-managed"}
        )
    )


def _recover_managed_ci_approved_plan(
    child_comments: Sequence[object],
    *,
    expected_hash: str,
    parent_comments: Sequence[object] | None = None,
) -> ApprovedPlanContext:
    """Recover the canonical approved plan for a managed-CI resume.

    A staged decomposition child carries only its issue-to-PR handoff record;
    the approved plan round lives on the authoritative parent issue named by
    the child's authenticated fresh-phase identity.  The fallback predicate is
    exactly what the recovery model reports: the child recovery is unavailable
    *and* ``has_matching_candidate`` is False, i.e. no candidate survived.
    Only then are the in-process parent comments consulted, as the ordinary
    PR-mode recovery path does.

    That predicate is deliberately broader than "the child carries no record
    with ``expected_hash``".  No ``expected_subject`` is passed here, but
    ``recover_approved_plan_context`` still derives a required subject for a
    legacy free-form record from its own metadata, so such a record can carry
    the handoff hash, be rejected on that derived subject, and still leave
    ``has_matching_candidate`` False.  The fallback is permitted in that case:
    the subject-rejected record is never adopted, and the canonical handoff
    plan hash remains the sole binding on whatever the parent yields.

    Only divergent accepted candidates — several records matching the hash
    that disagree on the plan text — set ``has_matching_candidate`` True, and
    those keep failing closed: the parent is never consulted.  The parent
    identity is only ever the in-process context supplied by the staged-child
    dispatch; nothing is inferred from PR body text or from the handoff
    record's own fields.
    """

    candidate = recover_approved_plan_context(
        child_comments,
        expected_hash=expected_hash,
    )
    if candidate.is_available or candidate.has_matching_candidate:
        return candidate
    if parent_comments is None:
        return candidate
    parent_candidate = recover_approved_plan_context(
        parent_comments,
        expected_hash=expected_hash,
    )
    if parent_candidate.is_available:
        return parent_candidate
    return candidate


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
                    log(config, f"PR #{pr_number}: unable to release invocation-owned managed-CI label")
            log(config, f"PR #{pr_number}: managed-CI adoption provenance changed; using ordinary CI")
            managed_ci = None
            return False
        memory = prepare_agent_memory(runner, config)
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
        for reviewer in configured_reviewers:
            invocation = resolve_invocation(config, provider=reviewer, role="reviewer")
            reviewer_acquisition_contract[agent_display_name(reviewer)] = (
                invocation.configured_model,
                invocation.resolved_effort,
                reviewer,
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
                    initial_pr_context, configured_reviewers, scheduler_capabilities
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
            config = dataclasses_replace(config, reviewer=effective_reviewers)
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
        resumed_round = _resume_pr_round(
            initial_pr_context.comments,
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
            latest_coder_metadata = resumed_round.coder_metadata
            qualification_checkpoint = resumed_round.qualification_checkpoint
            next_unresolved_item_number = resumed_round.next_unresolved_item_number
            start_round_number = resumed_round.round_number
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
            ) + _evidence_review_context(
                prior_unresolved_items, response_head=evidence_response_head
            )
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
                        reviewer_acquisition_contract if selective_policy else None
                    ),
                )
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
                    reviewer_acquisition_contract if selective_policy else None
                ),
            )
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
                    pr_posted_checkpoint = PostedRoundMetadata(
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
                post_pr_comment(
                    runner, config=config, pr_number=pr_number,
                    body=_attach_round_metadata(
                        render_public_agent_comment(
                            kind="pr_review", parsed=parsed, agent=reviewer_name,
                            human_requirements_resolved_flag=human_requirements_resolved(review_output),
                            prior_items=prior_unresolved_items, dispositions=parsed.dispositions,
                            config=config, model_used=model_used,
                        ),
                        PostedRoundMetadata(
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

            # One private spool per round, shared by the parallel launcher and
            # the sequential resume seam (#1025).
            pr_round_spool = _review_round_spool(
                config, surface="pr", number=pr_number,
                round_number=round_number, subject=current_pr_subject,
            )
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
                            ))

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
                                    ledger_incomplete=round_ledger_incomplete,
                                    repair_resolved_history_item_ids=round_resolved_history_item_ids,
                                    role="reviewer",
                                    operation_description="PR review",
                                    reask_on_prior_disposition_omission=True,
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
                            ledger_incomplete=round_ledger_incomplete,
                            repair_resolved_history_item_ids=round_resolved_history_item_ids,
                            role="reviewer",
                            operation_description="PR review",
                            reask_on_prior_disposition_omission=True,
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
            if (
                (pr_round_parallel or selective_policy or unavailable_reviewer_failures)
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
                        f"(source: {scheduler_recorded_force_full_source or 'none'}).",
                        PostedRoundMetadata(
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
                            fetch_start_round=lambda: _pr_amendment_start_round(
                                get_pr_review_context(runner, config=config, pr_number=pr_number),
                                configured_reviewers,
                                scheduler_capabilities,
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
                                ),
                                repair_kwargs={
                                    "expected_kind": "pr_review",
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
                                PostedRoundMetadata(
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
                                        PostedRoundMetadata(
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
                                        PostedRoundMetadata(
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
            if current_resume is not None and current_resume.unrecorded_head_advance:
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
                derived_followup, _followup_derived_risk_evidence = (
                    _derive_authenticated_risk_evidence_for_coder(
                        coder_response.marker_value,
                        approved_plan_context=approved_plan_context,
                        runner=runner,
                        assigned_workdir=active_workdir(config),
                        head_sha=updated_pr_context.metadata.head_sha,
                        predecessor_head=pr_metadata.head_sha,
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
                )
            elif head_unchanged_sha is not None:
                public_comment = add_coder_followup_head_unchanged_notice(
                    public_comment, head_unchanged_sha
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
            latest_coder_metadata = PostedRoundMetadata(
                flow="pr",
                role="coder",
                agent=coder_name,
                round_number=coder_record_round,
                subject=str(updated_pr_context.metadata.head_sha or "unknown"),
                prior_items=tuple(unresolved_items),
                raw_structured_coder_response=raw_structured_coder_response,
                local_test_evidence=local_test_evidence,
                risk_test_matrix_evidence=(
                    coder_response.marker_value.risk_test_matrix_evidence.to_payload()
                    if isinstance(coder_response.marker_value, StructuredCoderFollowup)
                    and coder_response.marker_value.risk_test_matrix_evidence is not None
                    else None
                ),
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
                **_test_observation_degradation_fields(coder_response.marker_value),
                **_architecture_metadata_fields(
                    config, result=coder_response.marker_value
                ),
            )
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
                    f"PR #{pr_number}: {coder_name} left head {previous_head} unchanged in "
                    f"{unchanged_head_coder_turns} consecutive follow-up rounds, so another "
                    "review of the same diff cannot change the verdict. Stopping before round "
                    f"{round_number + 1}; human review required.{route}"
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
                    if config.managed_ci:
                        cleanup_failure = AgentLoopError(
                            message + "; the PR remains suppressed and requires manual label removal."
                        )
                    log(config, message)
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
