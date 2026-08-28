"""High-level issue, task, and PR orchestration loops."""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import shlex
import sys
import time
import zoneinfo
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace as dataclasses_replace
from pathlib import Path
from typing import Literal

from .agents.base import AgentName, AgentResult
from .agents.antigravity import AntigravityAttemptState
from .agents.registry import agent_display_name, agent_signature, get_backend, run_agent_result
from .config import (
    AgentLoopConfig,
    ensure_agent_workdirs,
    github_bootstrap_cwd,
    resolve_base_branch,
    reviewers,
    sync_coder_base_before_implementation,
    sync_coder_pr_before_validation,
    sync_reviewer_pr_before_review,
)
from .decomposition import (
    _decode_json_payload,
    CreatedPhaseIssue,
    RecordedPhase,
    approved_plan_hash,
    create_decomposition_child_issues,
    find_existing_decomposition,
    find_existing_one_shot_impl_handoff,
    find_latest_one_shot_impl_handoff,
    find_existing_phase_implementation_handoff,
    parse_plan_decomposition,
    post_decomposition_parent_summary,
    post_one_shot_impl_handoff_comment,
    post_phase_implementation_handoff_comment,
)
from .errors import (
    AgentInvocationError,
    AgentLoopError,
    IssueImplementationConflictError,
    QuotaResetExceededError,
    UnknownPriorItemDispositionError,
)
from .expected_closure import (
    ExpectedClosingContract,
    reject_parent_from_contract,
    resolve_direct_contract,
    resolve_issue_contract,
)
from .github import (
    CiWatchOutcome,
    IssueContext,
    PullRequestChecks,
    PullRequestMergeability,
    PullRequestReviewContext,
    get_pr_head_sha,
    get_issue_context,
    get_pr_mergeability,
    parse_linked_issue_numbers,
    get_pr_checks,
    get_pr_review_context,
    get_pr_state,
    merge_pr,
    post_issue_comment,
    post_pr_comment,
    post_trusted_pr_contract_record,
    post_trusted_pr_comment,
    reject_forged_protocol_markers,
    validate_open_issue,
    validate_open_pr,
    validate_pr_body_does_not_close_issue,
    validate_pr_expected_closing_issues,
    validate_pr_references_issue,
    validate_pull_request_provenance,
    wait_for_ci,
    watch_pr_checks,
)
from .issue_pr_handoff import (
    find_latest_issue_pr_handoff,
    post_issue_pr_handoff_comment,
    require_pr_metadata_for_handoff,
    resolve_canonical_pr_for_issue,
)
from .issue_pr_provenance import IssuePrProvenanceScope
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
from .logging import log, new_run_id, run_usage_summary_path
from .evidence_reconciliation import (
    bounded_reconciliation_candidates,
    collect_evidence_observations,
    reconcile_evidence,
)
from .memory import AgentMemoryContext, prepare_agent_memory
from .managed_ci import (
    AuthenticatedIssueCreatedHandoff,
    FINAL_CONTEXT,
    MANAGED_LABEL,
    ManagedCiOutcome,
    OrdinaryRecoveryCapability,
    activate_managed_ci,
    authenticate_issue_created_handoff,
    dispatch_final_qualification,
    intermediate_managed_checks,
    managed_label_present,
    preflight_managed_ci_creation,
    publish_manual_v2_qualification,
    prepare_v2_merge,
    publish_round_readiness,
    refresh_ordinary_recovery_capability,
    release_adopted_managed_ci,
    revalidate_adopted_managed_ci,
    revalidate_issue_created_handoff,
    recover_issue_created_handoff,
    render_managed_ci_resume_command,
    validate_ordinary_recovery_capability,
    wait_for_ordinary_recovery,
    wait_for_final_qualification,
)
from .managed_pr import recover_managed_pr_origin, validate_managed_pr_body
from .migrations import validate_pr_migration_topology
from .prompts import (
    CompactPlanTailContext,
    CompactPriorContext,
    CompactPrReviewTailContext,
    build_discuss_agenda_prompt,
    build_discuss_final_analysis_prompt,
    build_discuss_evidence_reconciliation_prompt,
    build_discuss_answer_confirmation_prompt,
    build_discuss_semantic_comparison_prompt,
    build_discuss_review_prompt,
    build_completion_recovery_prompt,
    build_followup_prompt,
    build_issue_implementation_prompt,
    build_issue_plan_prompt,
    build_issue_prompt,
    build_plan_decomposition_prompt,
    build_plan_review_prompt,
    build_plan_revision_prompt,
    build_merge_conflict_prompt,
    build_review_prompt,
    build_same_pr_followup_prompt,
    build_task_clarification_prompt,
    build_task_prompt,
    format_agent_list,
    render_coder_human_requirements_prompt_context,
)
from .protocol import (
    AgentUnavailable,
    ApprovedFollowups,
    DISCUSS_FAILED_OUTCOME,
    DISCUSS_RESEARCH_TARGET_VALUES,
    ChildStage,
    DeferredStage,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
    ParsedDiscussAgenda,
    ParsedDiscussEvidenceReconciliation,
    ParsedDiscussAnswer,
    DiscussUnresolvedItem,
    ParsedDiscussSemanticComparison,
    ParsedDiscussResponse,
    ParsedDiscussReview,
    failed_discuss_review_category,
    failed_discuss_review_placeholder,
    failed_discuss_answer_placeholder,
    is_failed_discuss_response,
    ParsedPlanReview,
    PlanReviewItems,
    ParsedReview,
    PUBLIC_RESPONSE_MARKER,
    ReviewItemDisposition,
    StructuredCoderFollowup,
    StructuredIssueImplementation,
    StructuredPlanState,
    StructuredPlanRevision,
    UnresolvedReviewItem,
    human_requirements_resolved,
    is_clarification_request,
    parse_human_requirements_acknowledgement,
    parse_agent_state,
    parse_agent_unavailable,
    parse_plan_review,
    parse_plan_review_items,
    parse_plan_state,
    parse_structured_plan_review,
    parse_pr_number,
    review_freeform_summary_text,
    normalize_response_file_structured_text,
    validate_human_requirements_acknowledgement,
    validate_human_requirement_dispositions,
    validate_structured_coder_followup,
    validate_structured_human_requirements_acknowledgement,
    validate_structured_issue_implementation,
    validate_structured_plan_state,
    validate_structured_plan_revision,
    validate_structured_discuss_agenda,
    validate_structured_discuss_review,
    validate_structured_discuss_answer,
    validate_structured_discuss_answer_confirmation,
    validate_structured_discuss_evidence_reconciliation,
    validate_structured_discuss_semantic_comparison,
)
from .protocol import parse_review
from .repair import (
    RepairAttemptResult,
    attempt_envelope_normalization,
    attempt_repair,
    execute_repair,
    strip_unknown_prior_item_dispositions,
)
from .runner import Runner
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
from .workdirs import active_workdir
from .workdir_guard import (
    read_workdir_head,
    validate_assigned_head_advanced,
    validate_checkout_inspected_evidence,
    validate_response_tests_within_workdir,
    validate_test_commands_within_workdir,
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
    StalledCheck,
    is_canonical_stall_only_text,
    is_wholly_infrastructure_blocked,
)
from .comment_rendering import (
    DEFERRED_STAGES_MARKER_RE,
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
    decode_deferred_stages_marker,
    normalize_freeform_signature,
    render_discuss_round_summary_comment,
    render_public_agent_comment,
    render_agent_unavailable_comment,
    render_canonical_plan_revision,
    render_canonical_plan_steps,
    PLAN_EXPECTED_CLOSING_MARKER_RE,
    decode_expected_closing_issue_declaration,
    _render_discuss_agenda_lines,
)
from .followups import (
    APPROVED_FOLLOWUP_MARKER_RE,
    GroupedApprovedFollowup,
    MAX_APPROVED_FOLLOWUP_ISSUES,
    _append_approved_followups_marker,
    _approved_followup_from_unresolved_item,
    _approved_followups_marker,
    _create_approved_followup_issues,
    _dedupe_approved_followups,
    _followup_heading_key,
    _followup_issue_body,
    _followup_issue_title,
    _format_approved_followup_summary,
    _format_created_followup_issue_summary,
    _format_same_pr_followups,
    _has_approved_followups_marker,
    _plan_followup_source_from_unresolved_item,
    _publish_plan_approved_followups,
    _normalize_followup_key,
    _publish_approved_followups,
)
from .round_state import (
    PostedRoundMetadata,
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
    _extract_round_metadata_records,
    _latest_pr_approved_reviews_for_head,
    _max_unresolved_item_number_from_records,
    _plan_subject,
    _prior_item_ledger_signature,
    _resume_discuss_round,
    _resume_plan_round,
    _resume_pr_round,
    _select_current_round_records,
    _serialize_disposition,
    _serialize_unresolved_item,
    _strip_round_metadata,
)
from .round_transport import is_round_transport_sidecar
from .protocol_markers import TrustedBody, scan_reserved_markers
from .unresolved_items import (
    ALL_RESOLVED_PROSE_RE,
    CODER_DISPUTE_NOTE_PREFIX,
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    MERGE_CONFLICT_ITEM_ID,
    _apply_dispute_evidence,
    _apply_unresolved_item_dispositions,
    _collect_prior_compact_summaries,
    _clear_human_requirements_ack_item,
    _clear_merge_conflict_item,
    _format_same_pr_unresolved_items,
    _format_unresolved_items_for_coder,
    _is_disputed_item,
    _maybe_fill_resolved_dispositions_from_prose,
    _next_unresolved_item,
    _normalize_disposition_section_prose,
    _reconcile_merge_conflict_item,
    _record_prior_item_disposition,
    _reconcile_human_requirements_ack_item,
    _upsert_human_requirements_ack_item,
    _validate_coder_followup_response,
    _validate_plan_review_response,
    _validate_review_response,
    _validate_structured_coder_followup_items,
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


# TRANSIENT_AGENT_OUTPUT_RE / NON_RETRYABLE_AGENT_OUTPUT_RE / is_transient_agent_output
# now live in .transient (imported above) so dependency-light callers such as
# helpers.run_external can reuse them without importing this module.
NEAR_MISS_AGENT_MARKER_RE = re.compile(
    r"(?m)^[ \t]*AGENT_(?:PLAN_)?STATE:[ \t]*(?:approved|blocking)[ \t.]*$",
    re.I,
)
PUBLIC_RESPONSE_ARTIFACT_PREFIX_RE = re.compile(
    r"\A\s*(?:={3,}\s*AGENT_LOOP_PUBLIC_RESPONSE_BELOW\s*={3,}\s*)+",
    re.I,
)
STRUCTURED_PUBLIC_RESPONSE_KINDS = frozenset(
    {"plan_state", "plan_review", "pr_review", "coder_followup", "issue_implementation", "plan_revision", "discuss_review", "discuss_answer", "discuss_semantic_comparison", "discuss_answer_confirmation"}
)
PLAN_REVISION_FOOTER_RE = re.compile(r"(?m)^<!--\s*AGENT_PLAN_STATE:\s*(approved|blocking)\s*-->\s*$")
STRUCTURED_FENCE_RE = re.compile(
    r"```(?:json)?[ \t]*\n(?P<body>\s*\{.*?\}\s*)```",
    re.I | re.S,
)
PUBLIC_RESPONSE_TRANSIENT_DIAGNOSTIC_RE = re.compile(
    r"\A\s*(?:\[[^\]]*(?:error|fatal)[^\]]*\]\s*)?"
    r"(?:"
    r"invalid stream|empty response|malformed tool call|"
    r"(?:http|status)\s*[:=]?\s*429\b|429\s+too many requests\b|too many requests\b|"
    r"rate.?limit(?:ed)?\b|quota\b.{0,40}\b(?:exceeded|exhausted)\b|"
    r"resource[_-]?exhausted\b|ratelimitexceeded\b|retry[- ]after\b|retry[_-]?delay\b|"
    r"no capacity available\b|model_capacity_exhausted\b|"
    r"capacity\b.{0,80}\b(?:unavailable|exceeded|exhausted)\b|"
    r"(?:gemini|claude|codex|provider|cli)\b.{0,120}"
    r"(?:429|rate.?limit|resource.?exhausted|no capacity|overloaded)"
    r")",
    re.I | re.S,
)
UNSUPPORTED_MODEL_DIRECT_RE = re.compile(
    r"\bmodel\b.{0,80}\b(?:is\s+)?not\s+(?:supported|available)\b|"
    r"\bmodel\b.{0,80}\bunavailable\b|"
    r"\bunsupported[_-]?\s*model\b|"
    r"\bmodel[_-]?not[_-]?(?:supported|available)\b|"
    r"\bmodel[_-]?unavailable\b",
    re.I | re.S,
)
INVALID_REQUEST_RE = re.compile(r"\binvalid_request_error\b", re.I)
MODEL_SUPPORT_OR_AVAILABILITY_RE = re.compile(
    r"\b(?:model|deployment)\b.{0,120}\b"
    r"(?:not\s+(?:supported|available)|unsupported|unavailable)\b|"
    r"\b(?:not\s+(?:supported|available)|unsupported|unavailable)\b"
    r".{0,120}\b(?:model|deployment)\b",
    re.I | re.S,
)
MODEL_TOKEN_RE = re.compile(
    r"(?:['\"`](?P<quoted>[A-Za-z0-9][A-Za-z0-9._:/+-]{1,})['\"`]\s+model\b)|"
    r"(?:\bmodel\s*(?:name)?\s*(?:is|:)?\s*['\"`]?(?P<after>[A-Za-z0-9][A-Za-z0-9._:/+-]{1,})['\"`]?)",
    re.I,
)
MODEL_PARENTHESES_SUFFIX_RE = re.compile(r"\s+\([^)]*\)\s*$")
FAILURE_CLASSIFICATION_TEXT_LIMIT = 12000
ISSUE_IMPLEMENTATION_SALVAGE_SCOPE = "issue-implementation"
APPROVED_PLAN_IMPLEMENTATION_SALVAGE_SCOPE = "approved-plan-implementation"
TASK_IMPLEMENTATION_SALVAGE_SCOPE = "task-implementation"
PR_FOLLOWUP_SALVAGE_SCOPE = "pr-followup"

# Threshold above which a rate-limit reset time causes an immediate exit
# rather than a silent wait (5 minutes).
LONG_RESET_THRESHOLD_SECONDS = 300

# Subset of TRANSIENT_AGENT_OUTPUT_RE patterns that specifically signal quota / rate-limit errors
# and where a reset time might be present in the error text.
_QUOTA_RATE_LIMIT_RE = re.compile(
    r"\b429\b|rate[- ]?limit(?:ed)?|"
    r"session[- ]?limit|too many sessions|"
    r"resource[- ]?exhausted|\bquota\b|"
    r"no capacity available|capacity.*(?:unavailable|exceeded)|"
    r"overloaded",
    re.I,
)
# Parse "Retry-After: N" (HTTP header) or "retry after N" or "retryDelay: Ns" (gRPC).
_RETRY_AFTER_SECONDS_RE = re.compile(
    r"\bretry[- ]after[:\s]+(\d+)\b"
    r"|\bretry[_-]?delay[:\s]+['\"]?(\d+)s['\"]?",
    re.I,
)
# Parse "try again in Xh Ym Zs".
_TRY_AGAIN_IN_RE = re.compile(
    r"\btry\s+again\s+in\s+"
    r"(?:(?P<h>\d+)\s*h(?:r|ours?)?\s*)?"
    r"(?:(?P<m>\d+)\s*m(?:in(?:utes?)?)?\s*)?"
    r"(?:(?P<s>\d+)\s*s(?:ec(?:onds?)?)?)?",
    re.I,
)
# Parse "reset in Xh Ym" / "resets in X hours".
_RESET_IN_RE = re.compile(
    r"\brese(?:t|ts)\s+in\s+"
    r"(?:(?P<h>\d+)\s*h(?:r|ours?)?\s*)?"
    r"(?:(?P<m>\d+)\s*m(?:in(?:utes?)?)?\s*)?"
    r"(?:(?P<s>\d+)\s*s(?:ec(?:onds?)?)?)?",
    re.I,
)
# Parse ISO 8601 timestamps (used to compute reset delta from now).
_ISO_TIMESTAMP_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)"
)
# Parse Claude Code session-limit messages such as
# "resets 1:30am (America/Los_Angeles)".
_ABSOLUTE_RESET_TIME_RE = re.compile(
    r"\brese(?:t|ts)(?:\s+at)?\s+"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<ampm>a\.?m\.?|p\.?m\.?)\s*"
    r"\((?P<timezone>[A-Za-z0-9_./+-]+)\)",
    re.I,
)


@dataclass(frozen=True)
class ExactHeadCiProof:
    """Non-empty current-head CI evidence required by automated merge paths."""

    head_sha: str
    source: str


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


@dataclass(frozen=True)
class ValidatedAgentResponse:
    text: str
    session_id: str | None
    marker_value: object
    usage: UsageMetadata | None = None
    # Model the agent actually ran, for the dynamic signature (#332). Carried from
    # AgentResult.model_used so the orchestrator render sites can stamp it.
    model_used: str | None = None
    acquisition_outcome: Literal["success", "accepted_nonzero_exit", "accepted_timeout"] = "success"
    acquisition_returncode: int | None = None


class _AgentUnavailableResponse(AgentLoopError):
    """Internal control flow for a validated agent-unavailable envelope."""

    def __init__(self, unavailable: AgentUnavailable) -> None:
        self.unavailable = unavailable
        super().__init__(unavailable.summary)


def _capture_agent_invocation(
    invoke: Callable[[], ValidatedAgentResponse],
) -> tuple[ValidatedAgentResponse | None, AgentInvocationError | None]:
    try:
        return invoke(), None
    except AgentInvocationError as exc:
        return None, exc


@dataclass(frozen=True)
class _UnsupportedModelDiagnostic:
    agent: AgentName | None
    agent_name: str
    role: str | None
    requested_model: str | None
    provider_auth_context: str | None
    fallback_flag: str | None
    fallback_value: str | None
    reason: str | None

    @property
    def role_qualified_agent(self) -> str:
        role = (self.role or "").strip().lower()
        if role in {"coder", "reviewer", "debater", "analyzer", "summary"}:
            return f"{self.agent_name} {role}"
        return self.agent_name


def _parse_absolute_reset_seconds(
    text: str,
    *,
    now_utc: datetime.datetime | None = None,
) -> int | None:
    m = _ABSOLUTE_RESET_TIME_RE.search(text)
    if not m:
        return None

    try:
        tz = zoneinfo.ZoneInfo(m.group("timezone"))
    except zoneinfo.ZoneInfoNotFoundError:
        return None

    if now_utc is None:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=datetime.timezone.utc)
    else:
        now_utc = now_utc.astimezone(datetime.timezone.utc)

    hour = int(m.group("hour"))
    minute = int(m.group("minute") or 0)
    ampm = m.group("ampm").lower().replace(".", "")
    if not 1 <= hour <= 12 or not 0 <= minute <= 59:
        return None
    if ampm == "am":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12

    now_local = now_utc.astimezone(tz)
    reset_local = now_local.replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )
    if reset_local <= now_local:
        reset_local += datetime.timedelta(days=1)

    return int((reset_local.astimezone(datetime.timezone.utc) - now_utc).total_seconds())


def _parse_rate_limit_reset_seconds(
    text: str,
    *,
    now_utc: datetime.datetime | None = None,
) -> int | None:
    """Extract the reset wait time in seconds from a rate-limit error message.

    Returns None if the reset time cannot be reliably parsed.
    """
    m = _RETRY_AFTER_SECONDS_RE.search(text)
    if m:
        val = m.group(1) or m.group(2)
        if val:
            return int(val)

    m = _TRY_AGAIN_IN_RE.search(text)
    if m and any(m.group(g) for g in ("h", "m", "s")):
        return (
            int(m.group("h") or 0) * 3600
            + int(m.group("m") or 0) * 60
            + int(m.group("s") or 0)
        )

    m = _RESET_IN_RE.search(text)
    if m and any(m.group(g) for g in ("h", "m", "s")):
        return (
            int(m.group("h") or 0) * 3600
            + int(m.group("m") or 0) * 60
            + int(m.group("s") or 0)
        )

    m = _ISO_TIMESTAMP_RE.search(text)
    if m:
        try:
            ts_str = m.group(1).replace(" ", "T")
            if not ts_str.endswith("Z"):
                ts_str += "Z"
            ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            base_now = now_utc or datetime.datetime.now(datetime.timezone.utc)
            if base_now.tzinfo is None:
                base_now = base_now.replace(tzinfo=datetime.timezone.utc)
            else:
                base_now = base_now.astimezone(datetime.timezone.utc)
            delta = int((ts - base_now).total_seconds())
            if delta > 0:
                return delta
        except (ValueError, OverflowError):
            pass

    reset_secs = _parse_absolute_reset_seconds(text, now_utc=now_utc)
    if reset_secs is not None:
        return reset_secs

    return None


def _format_reset_duration(seconds: int) -> str:
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)


def _format_reset_at_utc(seconds: int) -> str:
    reset_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)
    return reset_time.strftime("%H:%M UTC")

def _agent_log_context(log_paths: Sequence[object]) -> str:
    paths = [str(path) for path in log_paths if path is not None]
    if not paths:
        return ""
    return "\nAttempt logs:\n" + "\n".join(f"- {path}" for path in paths)


# Backwards-compatible alias: the implementation moved to .transient.
_is_transient_agent_output = is_transient_agent_output


def _decode_public_response_json_prefix(text: str) -> object | None:
    stripped = text.lstrip()
    stripped = PUBLIC_RESPONSE_ARTIFACT_PREFIX_RE.sub("", stripped)
    try:
        payload, _end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    return payload


def _is_error_shaped_json_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    error_keys = {"error", "errors", "code", "status", "message", "type"}
    return bool(error_keys.intersection(payload))


def _bounded_failure_classification_text(text: str) -> str:
    if len(text) <= FAILURE_CLASSIFICATION_TEXT_LIMIT:
        return text
    head_limit = FAILURE_CLASSIFICATION_TEXT_LIMIT // 3
    tail_limit = FAILURE_CLASSIFICATION_TEXT_LIMIT - head_limit
    return f"{text[:head_limit]}\n... [truncated] ...\n{text[-tail_limit:]}"


def _json_error_payload_text(payload: object) -> str:
    parts: list[str] = []

    def collect(value: object) -> None:
        if len(parts) >= 50:
            return
        if isinstance(value, dict):
            preferred_keys = (
                "error",
                "errors",
                "type",
                "code",
                "status",
                "message",
                "detail",
                "details",
            )
            seen: set[object] = set()
            for key in preferred_keys:
                if key in value:
                    seen.add(key)
                    collect(value[key])
            for key, item in value.items():
                if key not in seen and isinstance(item, (str, int, float)):
                    collect(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value[:20]:
                collect(item)
            return
        if isinstance(value, (str, int, float)):
            text = str(value).strip()
            if text:
                parts.append(text)

    collect(payload)
    return "\n".join(parts)


def _first_json_error_payload_text(text: str) -> str | None:
    candidates = [text.lstrip()]
    candidates.extend(
        line.strip()
        for line in text.splitlines()[:80]
        if line.strip().startswith("{")
    )
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        payload = _decode_public_response_json_prefix(candidate)
        if not _is_error_shaped_json_payload(payload):
            continue
        payload_text = _json_error_payload_text(payload)
        if payload_text:
            return payload_text
    return None


def _has_transient_availability_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:429|rate[- ]?limit(?:ed)?|quota|resource[-_ ]?exhausted|"
            r"no capacity|capacity|overloaded|internal server error|bad gateway|"
            r"service unavailable|gateway timeout|model_capacity_exhausted)\b",
            text,
            re.I,
        )
    )


def _looks_like_unsupported_model_text(text: str) -> bool:
    if not text.strip():
        return False
    if _has_transient_availability_signal(text):
        return False
    if UNSUPPORTED_MODEL_DIRECT_RE.search(text):
        return True
    return bool(
        INVALID_REQUEST_RE.search(text)
        and MODEL_SUPPORT_OR_AVAILABILITY_RE.search(text)
    )


def _unsupported_model_classification_text(
    text: str,
    *,
    public_response: bool = False,
    repair_expected_kind: str | None = None,
) -> str | None:
    """Return the bounded diagnostic text when it names an unsupported model."""
    bounded = _bounded_failure_classification_text(text)
    if public_response:
        payload = _decode_public_response_json_prefix(bounded)
        if not isinstance(payload, dict):
            return None
        kind = payload.get("kind")
        if (
            kind in STRUCTURED_PUBLIC_RESPONSE_KINDS
            and (
                repair_expected_kind is None
                or repair_expected_kind in STRUCTURED_PUBLIC_RESPONSE_KINDS
            )
        ):
            return None
        if not _is_error_shaped_json_payload(payload):
            return None
        payload_text = _json_error_payload_text(payload)
        if _looks_like_unsupported_model_text(payload_text):
            return payload_text
        return None

    payload_text = _first_json_error_payload_text(bounded)
    if payload_text and _looks_like_unsupported_model_text(payload_text):
        return payload_text
    if _looks_like_unsupported_model_text(bounded):
        return bounded
    return None


def _is_transient_public_response(text: str, *, repair_expected_kind: str | None = None) -> bool:
    """Classify extracted public responses without matching transient terms in content."""
    if NON_RETRYABLE_AGENT_OUTPUT_RE.search(text):
        return False

    payload = _decode_public_response_json_prefix(text)
    if isinstance(payload, dict):
        kind = payload.get("kind")
        if (
            kind in STRUCTURED_PUBLIC_RESPONSE_KINDS
            and (
                repair_expected_kind is None
                or repair_expected_kind in STRUCTURED_PUBLIC_RESPONSE_KINDS
            )
        ):
            return False
        if _is_error_shaped_json_payload(payload):
            return _is_transient_agent_output(json.dumps(payload, sort_keys=True))

    stripped = text.strip()
    return bool(PUBLIC_RESPONSE_TRANSIENT_DIAGNOSTIC_RE.search(stripped))


def _is_retryable_marker_near_miss(text: str) -> bool:
    return bool(NEAR_MISS_AGENT_MARKER_RE.search(text)) and not bool(
        NON_RETRYABLE_AGENT_OUTPUT_RE.search(text)
    )


def _failure_category(
    text: str,
    *,
    public_response: bool = False,
    repair_expected_kind: str | None = None,
) -> str:
    """Classify a failure for logging: helps users decide whether to rerun or fix config/code."""
    if not text.strip():
        return "empty-response"
    if _unsupported_model_classification_text(
        text,
        public_response=public_response,
        repair_expected_kind=repair_expected_kind,
    ):
        return "unsupported_model"  # requested model is incompatible with provider/auth mode
    if NON_RETRYABLE_AGENT_OUTPUT_RE.search(text):
        return "non-retryable"  # auth/billing — fix configuration
    if public_response:
        if _is_transient_public_response(text, repair_expected_kind=repair_expected_kind):
            return "transient"  # extracted provider diagnostic — rerun may help
        return "deterministic"  # public response protocol/content issue
    if TRANSIENT_AGENT_OUTPUT_RE.search(text):
        return "transient"  # rate-limit/infra — rerun may help
    return "deterministic"  # no transient signal — may need code fix


def _response_file_structured_status(text: str) -> str:
    normalized, status = normalize_response_file_structured_text(text)
    if status is not None:
        return status
    if normalized.lstrip().startswith("{"):
        return "structured-prefix"
    if normalized.lstrip().startswith("```"):
        return "fenced-or-markdown"
    return "markdown-or-prose"


def _candidate_source_texts(result: AgentResult) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    if result.message_text:
        sources.append(("message_text", result.message_text))
    if result.raw_output:
        raw = result.raw_output
        if PUBLIC_RESPONSE_MARKER in raw:
            sources.append(("stdout_marker", raw.rsplit(PUBLIC_RESPONSE_MARKER, 1)[1].lstrip()))
        sources.append(("raw_output", raw))
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for source, text in sources:
        key = (source, text)
        if key in seen:
            continue
        seen.add(key)
        unique.append((source, text))
    return unique


def _unfence_structured_json_blocks(text: str) -> str:
    return STRUCTURED_FENCE_RE.sub(lambda match: match.group("body").strip(), text)


def _structured_response_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    decoder = json.JSONDecoder()
    for variant in (text, _unfence_structured_json_blocks(text)):
        for match in re.finditer(r"\{", variant):
            start = match.start()
            try:
                payload, end = decoder.raw_decode(variant[start:])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            absolute_end = start + end
            trailing = variant[absolute_end:]
            for signature in re.finditer(r"(?m)^--\s+\S[^\n]*(?:\n)?", trailing):
                candidate = variant[start : absolute_end + signature.end()]
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
                break
    return candidates


def _recover_valid_structured_candidate(
    result: AgentResult,
    *,
    validate: Callable[[str], object],
    expected_kind: str | None,
    config: AgentLoopConfig,
    agent_name: str,
) -> tuple[str, object] | None:
    if expected_kind not in STRUCTURED_PUBLIC_RESPONSE_KINDS:
        return None
    valid: list[tuple[str, str, object]] = []
    invalid_count = 0
    for source, text in _candidate_source_texts(result):
        for candidate in _structured_response_candidates(text):
            try:
                marker_value = validate(candidate)
            except AgentLoopError:
                invalid_count += 1
                continue
            valid.append((source, candidate, marker_value))
    unique_valid: list[tuple[str, str, object]] = []
    seen_candidates: set[str] = set()
    for item in valid:
        if item[1] in seen_candidates:
            continue
        seen_candidates.add(item[1])
        unique_valid.append(item)
    if len(unique_valid) == 1:
        source, candidate, marker_value = unique_valid[0]
        log(
            config,
            f"{agent_name}: public response file was not structured; recovered valid "
            f"{expected_kind} from {source}",
        )
        return candidate, marker_value
    if len(unique_valid) > 1:
        log(
            config,
            f"{agent_name}: refused stdout/result recovery because multiple structured "
            f"{expected_kind} candidates were present",
        )
    elif invalid_count:
        log(
            config,
            f"{agent_name}: stdout/result contained structured-looking output, but no "
            f"candidate passed {expected_kind} validation",
        )
    else:
        log(
            config,
            f"{agent_name}: stdout/result did not contain a recoverable structured "
            f"{expected_kind} response",
        )
    return None


@dataclass(frozen=True)
class _HumanRequirementsRecoveryContext:
    surfaced_requirement_ids: tuple[str, ...]
    requires_direct_discussion_ack: bool


def _split_reconstructable_plan_revision_response(text: str) -> tuple[str, str] | None:
    normalized, _status = normalize_response_file_structured_text(text)
    decoder = json.JSONDecoder()
    stripped = normalized.strip()
    try:
        payload, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("kind") != "plan_revision":
        return None
    json_prefix = stripped[:end].rstrip()
    trailing = stripped[end:].lstrip()
    footer_match = PLAN_REVISION_FOOTER_RE.search(trailing)
    if footer_match is None:
        return None
    before_footer = trailing[: footer_match.start()].strip()
    if before_footer:
        return None
    footer_and_signature = trailing[footer_match.start() :].strip()
    return json_prefix, footer_and_signature


def _plan_revision_missing_human_acknowledgement(
    text: str,
    *,
    context: _HumanRequirementsRecoveryContext,
) -> bool:
    if not context.surfaced_requirement_ids and not context.requires_direct_discussion_ack:
        return False
    if _split_reconstructable_plan_revision_response(text) is None:
        return False
    parsed = parse_human_requirements_acknowledgement(text)
    return not parsed.marker_present or not parsed.section_present


def _human_requirements_acknowledgement_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    for index, line in enumerate(lines):
        if HUMAN_REQUIREMENTS_ADDRESSED_MARKER not in line:
            continue
        block_lines = [line.strip()]
        section_seen = False
        for next_line in lines[index + 1 :]:
            if section_seen and (
                PLAN_REVISION_FOOTER_RE.match(next_line.strip())
                or re.match(r"^--\s+\S", next_line.strip())
                or next_line.lstrip().startswith("{")
            ):
                break
            block_lines.append(next_line.rstrip())
            if re.match(r"^\s*###\s+Human requirements\s*$", next_line, re.I):
                section_seen = True
        blocks.append("\n".join(block_lines).strip())
    return blocks


def _recover_plan_revision_human_requirements_acknowledgement(
    result: AgentResult,
    *,
    text: str | None = None,
    validate: Callable[[str], object],
    context: _HumanRequirementsRecoveryContext,
    config: AgentLoopConfig,
    agent_name: str,
) -> tuple[str, object] | None:
    split = _split_reconstructable_plan_revision_response(text if text is not None else result.text)
    if split is None:
        log(
            config,
            f"{agent_name}: refused plan_revision human-requirements recovery because "
            "the public response file is not a reconstructable structured plan revision",
        )
        return None

    valid_blocks: list[tuple[str, str, tuple[object, ...]]] = []
    invalid_count = 0
    incomplete_count = 0
    for source, source_text in _candidate_source_texts(result):
        for block in _human_requirements_acknowledgement_blocks(source_text):
            parsed = parse_human_requirements_acknowledgement(block)
            if not parsed.marker_present or not parsed.section_present:
                incomplete_count += 1
                continue
            try:
                validate_human_requirements_acknowledgement(
                    block,
                    surfaced_requirement_ids=context.surfaced_requirement_ids,
                    requires_direct_discussion_ack=context.requires_direct_discussion_ack,
                )
                source_plan = validate_structured_plan_revision(source_text)
                if source_plan is None:
                    raise AgentLoopError("captured response was not a structured plan revision")
                validate_human_requirement_dispositions(
                    source_plan.human_requirement_dispositions,
                    surfaced_requirement_ids=context.surfaced_requirement_ids,
                    context="plan_revision.human_requirement_dispositions",
                )
            except AgentLoopError:
                invalid_count += 1
                continue
            valid_blocks.append((source, block, source_plan.human_requirement_dispositions))

    unique_valid: list[tuple[str, str, tuple[object, ...]]] = []
    seen_blocks: set[str] = set()
    for source, block, dispositions in valid_blocks:
        if block in seen_blocks:
            continue
        seen_blocks.add(block)
        unique_valid.append((source, block, dispositions))

    if len(unique_valid) != 1:
        if len(unique_valid) > 1:
            reason = "multiple distinct valid acknowledgement blocks were present"
        elif invalid_count:
            reason = "captured acknowledgement evidence failed human-requirements validation"
        elif incomplete_count:
            reason = "captured acknowledgement evidence lacked the marker or section"
        else:
            reason = "no captured acknowledgement evidence was present"
        log(config, f"{agent_name}: refused plan_revision human-requirements recovery because {reason}")
        return None

    json_prefix, footer_and_signature = split
    source, block, dispositions = unique_valid[0]
    payload = json.loads(json_prefix)
    payload["human_requirement_dispositions"] = [
        {
            "requirement_id": item.requirement_id,
            "disposition": item.disposition,
            "evidence": item.evidence,
        }
        for item in dispositions
    ]
    json_prefix = json.dumps(payload)
    recovered_text = f"{json_prefix}\n{block}\n{footer_and_signature}"
    try:
        marker_value = validate(recovered_text)
    except AgentLoopError as exc:
        log(
            config,
            f"{agent_name}: refused plan_revision human-requirements recovery because "
            f"the reconstructed response did not validate ({exc})",
        )
        return None
    log(
        config,
        f"{agent_name}: recovered plan_revision human-requirements acknowledgement from {source}",
    )
    return recovered_text, marker_value


def _retry_delay(config: AgentLoopConfig, retry_index: int) -> int:
    delays = config.agent_retry_backoff_seconds
    if not delays:
        return 1
    return delays[min(retry_index - 1, len(delays) - 1)]


def _clean_diagnostic_fragment(text: str, *, limit: int = 800) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" \t\r\n.;")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _model_flag_value(model: str | None) -> str | None:
    if model is None:
        return None
    stripped = model.strip()
    if not stripped:
        return None
    return MODEL_PARENTHESES_SUFFIX_RE.sub("", stripped).strip() or stripped


def _parse_model_from_provider_text(text: str) -> str | None:
    for match in MODEL_TOKEN_RE.finditer(text):
        model = match.group("quoted") or match.group("after")
        if not model:
            continue
        lowered = model.lower()
        if lowered in {"is", "not", "unsupported", "available", "unavailable", "supported"}:
            continue
        return model.strip(".,;:")
    return None


def _extract_provider_auth_context(text: str) -> str | None:
    patterns = (
        r"\bwhen using (?P<context>[^.\n;]+)",
        r"\bwhen authenticated (?:as|with) (?P<context>[^.\n;]+)",
        r"\bfor (?P<context>[^.\n;]*(?:account|auth|authentication|provider|"
        r"api key|subscription|project|tenant|workspace|organization)[^.\n;]*)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            context = _clean_diagnostic_fragment(match.group("context"), limit=240)
            return context or None
    return None


def _extract_unsupported_model_reason(text: str) -> str | None:
    for line in text.splitlines():
        if _looks_like_unsupported_model_text(line):
            return _clean_diagnostic_fragment(line)
    if _looks_like_unsupported_model_text(text):
        return _clean_diagnostic_fragment(text)
    return None


def _configured_requested_model(
    agent: AgentName | None,
    config: AgentLoopConfig | None,
) -> str | None:
    if config is None:
        return None
    if agent == "codex":
        return config.codex_model.strip() if config.codex_model else None
    if agent == "antigravity":
        return config.antigravity_models[0].strip() if config.antigravity_models else None
    if agent == "gemini":
        return config.gemini_model.strip() if config.gemini_model else None
    if agent == "claude":
        return config.claude_model.strip() if config.claude_model else None
    return None


def _resolve_requested_model(
    *,
    agent: AgentName | None,
    config: AgentLoopConfig | None,
    result: AgentResult | None,
    classification_text: str,
) -> str | None:
    result_model = result.model_used.strip() if result is not None and result.model_used else None
    config_model = _configured_requested_model(agent, config)
    parsed_model = _parse_model_from_provider_text(classification_text)
    return result_model or config_model or parsed_model


def _known_unsupported_model_fallback(
    *,
    agent: AgentName | None,
    requested_model: str | None,
    provider_auth_context: str | None,
    reason: str | None,
) -> tuple[str | None, str | None]:
    model = _model_flag_value(requested_model)
    combined = f"{provider_auth_context or ''} {reason or ''}"
    if (
        agent == "codex"
        and model is not None
        and model.lower() == "gpt-5.5-pro"
        and re.search(r"\bchatgpt\b", combined, re.I)
    ):
        return "--codex-model", "gpt-5.5"
    return None, None


def _build_unsupported_model_diagnostic(
    *,
    agent: AgentName | None,
    agent_name: str,
    config: AgentLoopConfig | None,
    role: str | None,
    result: AgentResult | None,
    classification_text: str,
) -> _UnsupportedModelDiagnostic:
    unsupported_text = (
        _unsupported_model_classification_text(classification_text)
        or _bounded_failure_classification_text(classification_text)
    )
    requested_model = _resolve_requested_model(
        agent=agent,
        config=config,
        result=result,
        classification_text=unsupported_text,
    )
    provider_auth_context = _extract_provider_auth_context(unsupported_text)
    reason = _extract_unsupported_model_reason(unsupported_text)
    fallback_flag, fallback_value = _known_unsupported_model_fallback(
        agent=agent,
        requested_model=requested_model,
        provider_auth_context=provider_auth_context,
        reason=reason,
    )
    return _UnsupportedModelDiagnostic(
        agent=agent,
        agent_name=agent_name,
        role=role,
        requested_model=requested_model,
        provider_auth_context=provider_auth_context,
        fallback_flag=fallback_flag,
        fallback_value=fallback_value,
        reason=reason,
    )


def _unsupported_model_suggestion(diagnostic: _UnsupportedModelDiagnostic | None) -> str:
    if diagnostic is None:
        return (
            "Suggestion: choose a model supported by the configured provider/auth mode, "
            "or switch to a provider/auth configuration where the requested model is available."
        )
    requested = _model_flag_value(diagnostic.requested_model)
    requested_ref = f"`{requested}`" if requested else "the requested model"
    if diagnostic.fallback_flag and diagnostic.fallback_value:
        return (
            f"Suggestion: try a compatible {diagnostic.agent_name} model, for example:\n"
            f"  {diagnostic.fallback_flag} {diagnostic.fallback_value}\n"
            f"Alternatively, use a provider/auth configuration where {requested_ref} is "
            "available, if supported by your setup."
        )
    return (
        f"Suggestion: choose a model compatible with {diagnostic.agent_name}'s "
        f"configured provider/auth mode, or use a provider/auth configuration where "
        f"{requested_ref} is available. The orchestrator will not change the requested "
        "model automatically."
    )


def _executable_replacement_failure_detail(
    *,
    provider: AgentName | None,
    reason: str | None,
    stability_error: str | None,
) -> str:
    """Render sticky replacement evidence without losing the terminal cause."""
    if provider is None:
        return ""
    if stability_error:
        return stability_error
    if provider == "claude":
        label = "Claude self-update"
    elif provider == "codex":
        label = "Codex executable replacement"
    elif provider == "gemini":
        label = "Gemini executable replacement"
    elif provider == "antigravity":
        label = "Antigravity executable replacement"
    else:
        label = "agent executable replacement"
    detail = f"likely {label} interruption"
    if reason:
        detail += f" ({reason})"
    return f"{detail}; dedicated replay and ordinary retries were exhausted."


def _failure_suggestion(
    category: str | None,
    reason: str,
    agent_name: str,
    *,
    classification_text: str = "",
    unsupported_model_diagnostic: _UnsupportedModelDiagnostic | None = None,
) -> str:
    """Return a one-line actionable suggestion to append to an agent failure message."""
    combined = f"{reason} {classification_text}"
    if category == "unsupported_model":
        return _unsupported_model_suggestion(unsupported_model_diagnostic)
    if category == "agent-unavailable":
        return (
            "Suggestion: resolve the reported agent environment/provider/tooling problem, "
            "or switch that agent/model before re-running."
        )
    if category == "self-update-interruption":
        if agent_name.lower() == "claude":
            return "Suggestion: wait for the Claude Code self-update to finish, then re-run."
        return f"Suggestion: wait for the {agent_name} executable replacement to finish, then re-run."
    if category == "executable-replacement":
        return f"Suggestion: wait for the {agent_name} executable replacement to finish, then re-run."
    if category == "transient":
        if _QUOTA_RATE_LIMIT_RE.search(combined):
            return (
                "Suggestion: wait for quota reset or rate-limit window to pass, "
                "then re-run the same command to resume."
            )
        return (
            "Suggestion: re-run the same command — "
            "this is a transient failure and a retry may succeed."
        )
    if category == "non-retryable":
        if re.search(r"\b(?:credit|billing)\b", combined, re.I):
            return "Suggestion: check your API billing / credit balance, then re-run."
        if re.search(r"\bdirty\b", combined, re.I):
            return "Suggestion: clean up the dirty working tree or workdir, then re-run."
        return f"Suggestion: check that {agent_name} is installed and authenticated, then re-run."
    if category == "deterministic":
        if "repair invocation failure" in reason and "invalid_output" in reason:
            return (
                "Suggestion: re-run the same command — "
                "the round is resumable and a retry may succeed."
            )
        return "Suggestion: inspect the log above, fix the underlying issue, then re-run."
    return ""


def _format_unsupported_model_agent_response_error(
    *,
    diagnostic: _UnsupportedModelDiagnostic,
    marker_description: str,
    reason: str,
    exit_context: str,
    log_context: str,
    suggestion: str,
) -> str:
    if diagnostic.requested_model:
        model_phrase = f"requested model `{diagnostic.requested_model}`"
    else:
        model_phrase = "the requested model"
    context_phrase = (
        f" when using {diagnostic.provider_auth_context}"
        if diagnostic.provider_auth_context
        else ""
    )
    provider_line = (
        f"\nProvider diagnostic: {diagnostic.reason}"
        if diagnostic.reason
        else ""
    )
    suggestion_line = f"\n{suggestion}" if suggestion else ""
    return (
        f"{diagnostic.role_qualified_agent} failed because {model_phrase} is not "
        f"supported{context_phrase}. No successful agent result was recorded. "
        f"Required marker: {marker_description}. Reason: {reason}.{exit_context} "
        "Failure category: unsupported_model (choose a compatible model or "
        f"provider/auth mode).{provider_line}"
        f"{log_context}"
        f"{suggestion_line}"
    )


def _format_invalid_agent_response_error(
    *,
    agent_name: str,
    marker_description: str,
    reason: str,
    result: AgentResult | None,
    log_paths: Sequence[object],
    category: str | None = None,
    agent: AgentName | None = None,
    config: AgentLoopConfig | None = None,
    role: str | None = None,
    classification_text: str = "",
) -> str:
    exit_context = ""
    if result is not None and result.returncode not in (0, None):
        exit_context = f" Agent exit code: {result.returncode}."
    log_context = _agent_log_context(log_paths)
    category_hint = ""
    if category == "transient":
        category_hint = " Failure category: transient (rerun may succeed)."
    elif category == "non-retryable":
        category_hint = " Failure category: non-retryable (check credentials or billing)."
    elif category == "deterministic":
        category_hint = " Failure category: deterministic (may require a code fix)."
    elif category == "timeout":
        category_hint = " Failure category: timeout (the agent exceeded the configured time limit)."
    elif category == "self-update-interruption":
        if agent_name.lower() == "claude":
            category_hint = " Failure category: self-update-interruption (Claude Code updated during startup)."
        else:
            category_hint = f" Failure category: self-update-interruption ({agent_name} executable changed during invocation)."
    elif category == "executable-replacement":
        category_hint = f" Failure category: executable-replacement ({agent_name} changed during invocation)."
    elif category == "agent-unavailable":
        category_hint = " Failure category: agent-unavailable (the agent explicitly could not continue)."
    if not classification_text:
        classification_text = (result.raw_output or result.text or "") if result is not None else ""
    unsupported_model_diagnostic = None
    if category == "unsupported_model":
        unsupported_model_diagnostic = _build_unsupported_model_diagnostic(
            agent=agent,
            agent_name=agent_name,
            config=config,
            role=role,
            result=result,
            classification_text=classification_text,
        )
    suggestion = _failure_suggestion(
        category,
        reason,
        agent_name,
        classification_text=classification_text,
        unsupported_model_diagnostic=unsupported_model_diagnostic,
    )
    if category == "unsupported_model" and unsupported_model_diagnostic is not None:
        return _format_unsupported_model_agent_response_error(
            diagnostic=unsupported_model_diagnostic,
            marker_description=marker_description,
            reason=reason,
            exit_context=exit_context,
            log_context=log_context,
            suggestion=suggestion,
        )
    suggestion_line = f"\n{suggestion}" if suggestion else ""
    return (
        f"{agent_name} failed before producing a valid public response. "
        "No review result was recorded. "
        f"Required marker: {marker_description}. Reason: {reason}.{exit_context}"
        f"{category_hint}"
        f"{log_context}"
        f"{suggestion_line}"
    )


def _compact_failure_reason(reason: str, classification_text: str) -> str:
    detail = classification_text.strip()
    if not detail or detail == reason:
        return reason
    lines = detail.splitlines()
    if len(lines) > 20:
        detail = "\n".join(lines[-20:])
    if len(detail) > 4000:
        detail = detail[-4000:]
    return f"{reason}; diagnostic:\n{detail}"


@dataclass(frozen=True)
class _PatchSalvageDiagnostic:
    artifacts: SalvageArtifacts | None
    line: str


@dataclass(frozen=True)
class _FailedRunDiagnostics:
    patch_salvage: _PatchSalvageDiagnostic
    response_line: str

    def format_for_error(self) -> str:
        return f"\n{self.patch_salvage.line}\n{self.response_line}"


def _best_effort_failed_run_status(
    runner: Runner,
    config: AgentLoopConfig,
) -> str | None:
    try:
        result = runner.run(
            ("git", "status", "--short"),
            cwd=active_workdir(config),
            check=False,
        )
    except (AgentLoopError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _status_is_untracked_only(status_text: str | None) -> bool:
    lines = [line for line in (status_text or "").splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("?? ") for line in lines)


def _capture_failed_run_salvage_diagnostic(
    *,
    runner: Runner,
    config: AgentLoopConfig,
    agent_name: str,
    salvage_context: SalvageContext | None,
    operation_description: str,
    failure_category: str,
    failure_reason: str,
    classification_text: str,
    marker_description: str,
    result: AgentResult | None,
) -> _PatchSalvageDiagnostic:
    if salvage_context is None:
        return _PatchSalvageDiagnostic(
            artifacts=None,
            line=(
                "No implementation salvage was attempted because this was "
                f"{operation_description}, not a mutating implementation attempt."
            ),
        )
    compacted_failure_reason = _compact_failure_reason(failure_reason, classification_text)
    try:
        artifacts = capture_salvage_artifacts(
            runner,
            checkout=active_workdir(config),
            log_dir=config.log_dir,
            context=salvage_context,
            failure_category=failure_category,
            failure_reason=compacted_failure_reason,
            required_marker=marker_description,
            result=result,
        )
    except (AgentLoopError, OSError) as exc:
        log(
            config,
            f"{agent_name}: salvage capture failed ({exc}); preserving original agent failure",
        )
        return _PatchSalvageDiagnostic(
            artifacts=None,
            line=(
                "Implementation salvage was attempted for "
                f"{operation_description}, but capture failed ({exc}); "
                "preserving the original agent failure."
            ),
        )
    if artifacts is not None:
        log(config, f"{agent_name}: salvage artifacts written to {artifacts.directory}")
        comment_posted = post_salvage_comment(
            runner,
            config=config,
            artifacts=artifacts,
            context=salvage_context,
            failure_category=failure_category,
            failure_reason=compacted_failure_reason,
        )
        comment_note = (
            f" A GitHub salvage comment was posted to issue #{salvage_context.issue_number}."
            if comment_posted
            else " No GitHub salvage comment was posted."
        )
        return _PatchSalvageDiagnostic(
            artifacts=artifacts,
            line=(
                "Implementation salvage artifacts were written to "
                f"{artifacts.summary_path}; patch: {artifacts.patch_path}.{comment_note}"
            ),
        )

    status_text = _best_effort_failed_run_status(runner, config)
    if _status_is_untracked_only(status_text):
        line = (
            "Implementation salvage was attempted for "
            f"{operation_description}, but only untracked files were present; "
            "no tracked/staged `git diff HEAD --binary` existed, so no patch "
            "artifacts were created."
        )
    else:
        line = (
            "Implementation salvage was attempted for "
            f"{operation_description}, but no tracked/staged "
            "`git diff HEAD --binary` existed, so no patch artifacts were created."
        )
    return _PatchSalvageDiagnostic(artifacts=None, line=line)


def _operation_description_from_context(
    *,
    salvage_context: SalvageContext | None,
    repair_expected_kind: str | None,
    role: str | None,
    label: str | None,
    marker_description: str,
) -> str:
    if salvage_context is not None:
        if salvage_context.scope == ISSUE_IMPLEMENTATION_SALVAGE_SCOPE:
            return "issue implementation"
        if salvage_context.scope == APPROVED_PLAN_IMPLEMENTATION_SALVAGE_SCOPE:
            return "approved-plan implementation"
        if salvage_context.scope == TASK_IMPLEMENTATION_SALVAGE_SCOPE:
            return "task implementation"
        if salvage_context.scope == PR_FOLLOWUP_SALVAGE_SCOPE:
            return "PR feedback follow-up"
        return salvage_context.scope.replace("-", " ")
    if repair_expected_kind == "plan_review":
        return "plan review"
    if repair_expected_kind == "plan_revision":
        return "plan revision"
    if repair_expected_kind == "pr_review":
        return "PR review"
    if repair_expected_kind == "coder_followup":
        return "structured PR feedback follow-up repair"
    if repair_expected_kind == "issue_implementation":
        return "issue implementation response repair"
    if repair_expected_kind == "discuss_review":
        return "discuss review"
    if repair_expected_kind == "discuss_agenda":
        return "discuss analyzer"
    if label and label.startswith("discuss-analyzer"):
        return "discuss analyzer"
    if label and label.startswith("discuss-r"):
        return "discuss review"
    if marker_description == "plan decomposition JSON":
        return "plan decomposition"
    if "AGENT_PR" in marker_description:
        return "implementation"
    if "AGENT_PLAN_STATE" in marker_description and "CLARIFY" in marker_description:
        return "planning"
    if role == "reviewer":
        return "review"
    return "agent operation"


def _failed_response_recording_reason(
    *,
    result: AgentResult | None,
    failure_reason: str,
    classification_text: str,
) -> str:
    if result is None:
        return "the agent run failed before a response path was available"
    if result.returncode is None:
        return "the agent command timed out"
    if result.returncode != 0:
        combined = f"{failure_reason}\n{classification_text}\n{result.raw_output}\n{result.text}"
        if _QUOTA_RATE_LIMIT_RE.search(combined):
            return "the agent command exited with quota/session-limit status"
        return f"the agent command exited with failing status {result.returncode}"
    if not result.text.strip():
        return "the agent response was empty"
    return f"the public response failed validation ({failure_reason})"


def _public_response_file_diagnostic(
    *,
    result: AgentResult | None,
    failure_reason: str,
    classification_text: str,
) -> str:
    reason = _failed_response_recording_reason(
        result=result,
        failure_reason=failure_reason,
        classification_text=classification_text,
    )
    if result is None:
        return (
            "No public response file was produced (response path unavailable); "
            f"no result was recorded because {reason}."
        )

    response_path = result.response_file_path
    response_file_text = result.response_file_text
    if response_file_text is None and response_path is not None:
        try:
            response_file_text = response_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            response_file_text = None

    if response_file_text:
        if response_path is None:
            return (
                "A public response was present, but its file path is unavailable; "
                f"no result was recorded because {reason}."
            )
        return (
            f"A public response file exists at {response_path}, but no result was "
            f"recorded because {reason}."
        )
    if response_path is not None:
        return (
            f"No non-empty public response file was produced at expected path "
            f"{response_path}; no result was recorded because {reason}."
        )
    return (
        "No public response file was produced (response path unavailable); "
        f"no result was recorded because {reason}."
    )


def _failed_run_diagnostics(
    *,
    runner: Runner,
    config: AgentLoopConfig,
    agent_name: str,
    salvage_context: SalvageContext | None,
    operation_description: str,
    failure_category: str,
    failure_reason: str,
    classification_text: str,
    marker_description: str,
    result: AgentResult | None,
) -> _FailedRunDiagnostics:
    patch_salvage = _capture_failed_run_salvage_diagnostic(
        runner=runner,
        config=config,
        agent_name=agent_name,
        salvage_context=salvage_context,
        operation_description=operation_description,
        failure_category=failure_category,
        failure_reason=failure_reason,
        classification_text=classification_text,
        marker_description=marker_description,
        result=result,
    )
    response_line = _public_response_file_diagnostic(
        result=result,
        failure_reason=failure_reason,
        classification_text=classification_text,
    )
    return _FailedRunDiagnostics(
        patch_salvage=patch_salvage,
        response_line=response_line,
    )


def _agent_failure_classification_text(
    result: AgentResult,
    *,
    phase: str,
) -> str:
    """Choose the text that matches the failure being classified."""
    if phase in {"command", "empty"}:
        return result.raw_output or result.text
    return result.text


def _new_usage_context(config: AgentLoopConfig) -> RunUsageContext:
    run_id = new_run_id()
    return RunUsageContext(run_id=run_id, summary_path=run_usage_summary_path(config, run_id))


def _resolve_usage_metadata(
    *,
    config: AgentLoopConfig,
    prompt: str,
    result: AgentResult,
) -> UsageMetadata | None:
    if result.usage is not None:
        return result.usage.with_io_sizes(prompt=prompt, response=result.text)
    if config.dry_run:
        return None
    return estimate_usage(prompt, result.text)


def _persist_usage_summary(config: AgentLoopConfig, usage_context: RunUsageContext) -> None:
    usage_context.write_summary()
    totals = usage_context.totals()
    log(
        config,
        "Usage summary written to "
        f"{usage_context.summary_path} "
        f"(calls={totals.call_count}, exact={totals.exact_calls}, "
        f"partial={totals.partial_calls}, estimated={totals.estimated_calls})",
    )


_ORIGINAL_ATTEMPT_REPAIR = attempt_repair


def _run_structured_repair(
    raw: str,
    *,
    runner: Runner,
    config: AgentLoopConfig,
    usage_context: RunUsageContext | None,
    validate: Callable[[str], object],
    repair_kwargs: dict[str, object],
) -> tuple[str | None, object | None, list[RepairAttemptResult]]:
    """Run configured repair, retaining compatibility with patched legacy test hooks."""
    if attempt_repair is not _ORIGINAL_ATTEMPT_REPAIR:
        try:
            repaired = attempt_repair(raw, config.gemini_cmd, **repair_kwargs)
        except TypeError as exc:
            # Keep older test/integration hooks callable while the reviewer-ID
            # context is rolled out. The real repair API accepts this keyword.
            if "reviewer_requirement_ids" not in str(exc):
                raise
            legacy_kwargs = dict(repair_kwargs)
            legacy_kwargs.pop("reviewer_requirement_ids", None)
            repaired = attempt_repair(raw, config.gemini_cmd, **legacy_kwargs)
        if repaired is None:
            return None, None, []
        try:
            parsed = validate(repaired)
        except AgentLoopError as exc:
            return repaired, None, [
                RepairAttemptResult(
                    backend="gemini",
                    model="legacy-test-hook",
                    prompt="",
                    output=repaired,
                    returncode=0,
                    outcome="invalid_output",
                    diagnostic=str(exc),
                    log_path=None,
                    fallback_planned=False,
                )
            ]
        return repaired, parsed, []
    return execute_repair(
        raw,
        runner=runner,
        config=config,
        run_id=usage_context.run_id if usage_context is not None else None,
        usage_context=usage_context,
        validate=validate,
        **repair_kwargs,
    )


@dataclass(frozen=True)
class CompletionRecoveryPolicy:
    """Explicit opt-in for the bounded same-session completion-recovery pass (#588).

    Passed only by the direct issue-implementation call sites that validate
    structured `issue_implementation` results; every other
    `_run_validated_agent` caller (planning, plan/PR review, discuss, task,
    and the coder follow-up/PR loop) leaves this `None` and is therefore
    ineligible by construction -- eligibility is never inferred from the
    agent name or response text alone.
    """

    issue_number: int
    issue_context: IssueContext | None = None


@dataclass(frozen=True)
class _CompletionRecoveryOutcome:
    validated: ValidatedAgentResponse | None
    result: AgentResult
    error: str
    classification_text: str
    failure_category: str
    # Protocol-valid text already persisted to the recovery attempt's own
    # response file and posted to the GitHub issue; set only when validated
    # is None.
    terminal_public_response: str | None


def _post_completion_recovery_terminal_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    completion_recovery: CompletionRecoveryPolicy,
    recovery_result: AgentResult,
    terminal_text: str,
) -> None:
    if recovery_result.response_file_path is not None:
        recovery_result.response_file_path.write_text(terminal_text, encoding="utf-8")
    post_issue_comment(
        runner,
        config=config,
        issue_number=completion_recovery.issue_number,
        body=terminal_text,
    )


def _synthesized_completion_recovery_unavailable(
    *, config: AgentLoopConfig, recovery_result: AgentResult, category: str, summary: str
) -> tuple[AgentUnavailable, str]:
    unavailable = AgentUnavailable(
        schema_version=1,
        kind="agent_unavailable",
        retryable=False,
        category=category,
        summary=summary,
        suggested_action=(
            "Inspect the completion-recovery log and salvage artifacts, "
            "then retry the implementation manually."
        ),
    )
    signature = agent_signature("claude", config, model_used=recovery_result.model_used)
    return unavailable, render_agent_unavailable_comment(unavailable, signature=signature)


def _attempt_claude_completion_recovery(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    completion_recovery: CompletionRecoveryPolicy,
    session_id: str,
    validate: Callable[[str], object],
    usage_context: RunUsageContext | None,
    run_id: str | None,
    role: str | None,
    label: str | None,
    timeout_seconds: float | None,
) -> _CompletionRecoveryOutcome:
    """One bounded ``claude --resume`` completion-recovery pass (#588).

    Entirely self-contained and attempt-local: every variable here is scoped
    to this one recovery call, never the caller's original (failed) attempt.
    The caller's original ``AgentResult`` is not even passed in, so isolation
    is structural rather than merely asserted. Four outcomes:

    - a valid PR/blocking/clarify per ``validate()`` -> success, returned as
      a normal ``ValidatedAgentResponse`` subject to ordinary PR validation;
      this includes a current invocation response-file artifact that validates
      after a timeout or nonzero exit, with its acquisition diagnostic retained;
    - the resume turn itself declares a protocol-valid ``AGENT_UNAVAILABLE``
      -> terminal, posted/persisted verbatim, never passed to ``validate()``,
      and always terminal regardless of its own ``retryable`` flag (the
      bounded one-recovery-attempt policy overrides the agent's preference);
    - transport failures, or text that still fails ``validate()`` -> terminal
      with a synthesized non-retryable ``AGENT_UNAVAILABLE`` rendered,
      persisted, and posted; except a second background-wait-only response,
      which is a deterministic protocol failure rather than an operational
      unavailability.

    In every terminal case there is exactly one ``--resume`` call total.
    """
    recovery_prompt = build_completion_recovery_prompt(
        config,
        issue_context=completion_recovery.issue_context,
    )
    recovery_result = run_agent_result(
        runner,
        agent="claude",
        config=config,
        prompt=recovery_prompt,
        session_id=session_id,
        run_id=run_id,
        role=role,
        label=label,
        timeout_seconds=timeout_seconds,
    )
    recovery_usage_record = None
    if usage_context is not None:
        recovery_usage = _resolve_usage_metadata(
            config=config, prompt=recovery_prompt, result=recovery_result
        )
        if recovery_usage is not None:
            recovery_usage_record = usage_context.add_record(
                agent="claude",
                session_id=recovery_result.session_id,
                returncode=recovery_result.returncode,
                usage=recovery_usage,
                raw_backend_usage=recovery_result.raw_usage,
                role="completion-recovery",
            )
    else:
        recovery_usage = None

    def _terminal(category: str, summary: str, *, error: str) -> _CompletionRecoveryOutcome:
        _unavailable, rendered = _synthesized_completion_recovery_unavailable(
            config=config, recovery_result=recovery_result, category=category, summary=summary
        )
        _post_completion_recovery_terminal_comment(
            runner,
            config=config,
            completion_recovery=completion_recovery,
            recovery_result=recovery_result,
            terminal_text=rendered,
        )
        return _CompletionRecoveryOutcome(
            validated=None,
            result=recovery_result,
            error=error,
            classification_text=recovery_result.raw_output or recovery_result.text,
            failure_category="agent-unavailable",
            terminal_public_response=rendered,
        )

    # A response-file artifact is authoritative for this invocation even when
    # the CLI reports a failed exit.  Do not consult stdout here: it may only
    # contain diagnostics.  Invalid artifacts intentionally fall through to
    # the existing terminal transport handling below.
    recovery_artifact = recovery_result.response_file_text
    if recovery_artifact:
        try:
            unavailable = parse_agent_unavailable(recovery_artifact)
        except AgentLoopError:
            unavailable = None
        if unavailable is not None:
            _post_completion_recovery_terminal_comment(
                runner, config=config, completion_recovery=completion_recovery,
                recovery_result=recovery_result, terminal_text=recovery_artifact,
            )
            return _CompletionRecoveryOutcome(
                validated=None, result=recovery_result,
                error=("agent explicitly reported it cannot continue after completion "
                       f"recovery ({unavailable.category}): {unavailable.summary}"),
                classification_text=recovery_artifact, failure_category="agent-unavailable",
                terminal_public_response=recovery_artifact,
            )
        try:
            marker_value = validate(recovery_artifact)
        except AgentLoopError:
            pass
        else:
            accepted_outcome = (
                "accepted_timeout" if recovery_result.returncode is None
                else "accepted_nonzero_exit" if recovery_result.returncode != 0 else "success"
            )
            if recovery_usage_record is not None:
                recovery_usage_record.validation_status = "validated"
                recovery_usage_record.outcome = accepted_outcome
            if accepted_outcome != "success":
                log(config, "claude completion-recovery accepted a valid response-file artifact "
                    f"despite returncode={recovery_result.returncode!r}")
            return _CompletionRecoveryOutcome(
                validated=ValidatedAgentResponse(
                    text=recovery_artifact, session_id=recovery_result.session_id,
                    marker_value=marker_value, usage=recovery_usage,
                    model_used=recovery_result.model_used,
                    acquisition_outcome=accepted_outcome,
                    acquisition_returncode=recovery_result.returncode,
                ),
                result=recovery_result, error="", classification_text="", failure_category="",
                terminal_public_response=None,
            )
    if recovery_result.returncode is None:
        limit = f" after {timeout_seconds:g}s" if timeout_seconds is not None else ""
        return _terminal(
            "environment",
            "The bounded claude --resume completion-recovery pass timed out.",
            error=f"completion-recovery resume timed out{limit}",
        )
    if recovery_result.returncode != 0:
        return _terminal(
            "tooling",
            "The bounded claude --resume completion-recovery pass exited with a non-zero status.",
            error=f"completion-recovery resume exited with {recovery_result.returncode}",
        )
    recovery_text = recovery_result.text
    if not recovery_text.strip():
        return _terminal(
            "tooling",
            "The bounded claude --resume completion-recovery pass produced no output.",
            error="completion-recovery resume produced no output",
        )

    try:
        unavailable = parse_agent_unavailable(recovery_text)
    except AgentLoopError:
        unavailable = None
    if unavailable is not None:
        # Agent-declared: post/persist verbatim, never validate()'d, and
        # always terminal regardless of unavailable.retryable.
        _post_completion_recovery_terminal_comment(
            runner,
            config=config,
            completion_recovery=completion_recovery,
            recovery_result=recovery_result,
            terminal_text=recovery_text,
        )
        return _CompletionRecoveryOutcome(
            validated=None,
            result=recovery_result,
            error=(
                "agent explicitly reported it cannot continue after completion "
                f"recovery ({unavailable.category}): {unavailable.summary}"
            ),
            classification_text=recovery_text,
            failure_category="agent-unavailable",
            terminal_public_response=recovery_text,
        )

    try:
        marker_value = validate(recovery_text)
    except AgentLoopError as exc:
        if looks_like_backgrounded_completion(recovery_text):
            # A completed CLI turn that again says it is waiting for
            # background work is not an environment/provider outage. Do not
            # overwrite the real diagnostic with AGENT_UNAVAILABLE or post a
            # misleading operational-failure comment (#593).
            return _CompletionRecoveryOutcome(
                validated=None,
                result=recovery_result,
                error=(
                    "completion-recovery resume again deferred to background work "
                    f"without a terminal response: {exc}"
                ),
                classification_text=recovery_text,
                failure_category="deterministic",
                terminal_public_response=None,
            )
        _unavailable, rendered = _synthesized_completion_recovery_unavailable(
            config=config,
            recovery_result=recovery_result,
            category="tooling",
            summary=(
                "The bounded claude --resume completion-recovery pass did not "
                f"produce a valid terminal response: {exc}"
            ),
        )
        _post_completion_recovery_terminal_comment(
            runner,
            config=config,
            completion_recovery=completion_recovery,
            recovery_result=recovery_result,
            terminal_text=rendered,
        )
        return _CompletionRecoveryOutcome(
            validated=None,
            result=recovery_result,
            error=str(exc),
            classification_text=recovery_text,
            failure_category="agent-unavailable",
            terminal_public_response=rendered,
        )

    return _CompletionRecoveryOutcome(
        validated=ValidatedAgentResponse(
            text=recovery_text,
            session_id=recovery_result.session_id,
            marker_value=marker_value,
            usage=recovery_usage,
            model_used=recovery_result.model_used,
        ),
        result=recovery_result,
        error="",
        classification_text="",
        failure_category="",
        terminal_public_response=None,
    )


def _log_repair_attempts(config: AgentLoopConfig, prefix: str, attempts: Sequence[RepairAttemptResult]) -> None:
    for attempt in attempts:
        diagnostic = attempt.diagnostic or "(none)"
        log(
            config,
            f"{prefix}: repair backend={attempt.backend} model={attempt.model} "
            f"outcome={attempt.outcome} returncode="
            f"{attempt.returncode if attempt.returncode is not None else 'none'}; "
            f"diagnostic={diagnostic}; log={attempt.log_path or '(none)'}; "
            f"fallback_planned={'yes' if attempt.fallback_planned else 'no'}",
        )


def _run_validated_agent(
    runner: Runner,
    *,
    agent: AgentName,
    config: AgentLoopConfig,
    prompt: str,
    marker_description: str,
    validate: Callable[[str], object],
    session_id: str | None = None,
    usage_context: RunUsageContext | None = None,
    use_repair: bool = False,
    repair_expected_kind: str | None = None,
    repair_unresolved_item_ids: Sequence[str] | None = None,
    repair_surfaced_requirement_ids: Sequence[str] | None = None,
    repair_reviewer_requirement_ids: Sequence[str] | None = None,
    repair_requires_direct_discussion_ack: bool = False,
    repair_allowed_prior_item_ids: Sequence[str] | None = None,
    ledger_incomplete: bool = False,
    role: str | None = None,
    label: str | None = None,
    timeout_seconds: float | None = None,
    salvage_context: SalvageContext | None = None,
    operation_description: str | None = None,
    completion_recovery: CompletionRecoveryPolicy | None = None,
) -> ValidatedAgentResponse:
    # Agent responses are current untrusted visible text.  Keep this guard in
    # the validation seam so every artifact recovery and repair path receives
    # the same provenance check before it can be accepted.
    response_validator = validate

    def validate(text: str) -> object:
        TrustedBody.current_untrusted_visible(text)
        return response_validator(text)

    agent_name = agent_display_name(agent)
    operation_description = operation_description or _operation_description_from_context(
        salvage_context=salvage_context,
        repair_expected_kind=repair_expected_kind,
        role=role,
        label=label,
        marker_description=marker_description,
    )
    log_paths: list[object] = []
    antigravity_attempts = (
        AntigravityAttemptState.from_config(config, config.agent_max_retries)
        if agent == "antigravity"
        else None
    )
    # Each fallback retains an initial attempt; retry allowance is shared. The
    # provider-specific replacement replay gets one explicit extra slot.
    max_attempts = (
        len(config.antigravity_models) + config.agent_max_retries + 1
        if antigravity_attempts is not None
        else config.agent_max_retries + 2
    )
    last_error = f"{agent_name} produced no output."
    last_result: AgentResult | None = None
    last_classification_text = ""
    last_failure_category = "empty-response"
    # Set only by an exhausted completion-recovery attempt (#588): the
    # protocol-valid text already persisted to the recovery attempt's own
    # response file and posted to the GitHub issue, attached to the final
    # AgentInvocationError so callers/tests can assert on it without
    # re-parsing the message.
    terminal_public_response: str | None = None
    completion_recovery_attempted = False
    # Keep the detection guard separate from the replay marker: a failed
    # stability check must not make the next ordinary retry look like a replay.
    executable_replacement_considered = False
    executable_replacement_replay_pending = False
    executable_replacement_provider: AgentName | None = None
    executable_replacement_reason: str | None = None
    ordinary_retries_used = 0
    self_update_deadline: float | None = None
    self_update_stability_error: str | None = None
    # Refusals are diagnostic-only context. Keep only the latest one, without
    # treating it as accepted executable-replacement evidence.
    latest_replay_refusal_detail: str | None = None
    next_timeout_seconds = timeout_seconds
    marker_safety_repair_attempted = False
    executable_replacement_policies: dict[AgentName, tuple[str, str, str, bool]] = {
        "claude": (
            config.claude_cmd,
            "Claude self-update",
            "self-update-attempt2",
            True,
        ),
        "codex": (
            config.codex_cmd,
            "Codex executable replacement",
            "executable-replacement-attempt2",
            False,
        ),
        "gemini": (
            config.gemini_cmd,
            "Gemini executable replacement",
            "executable-replacement-attempt2",
            False,
        ),
        "antigravity": (
            config.antigravity_cmd,
            "Antigravity executable replacement",
            "executable-replacement-attempt2",
            False,
        ),
    }

    for attempt in range(1, max_attempts + 1):
        replacement_stability_failed = False
        attempt_config = (
            antigravity_attempts.singleton_config(config)
            if antigravity_attempts is not None
            else config
        )
        invocation_kwargs: dict[str, object] = {}
        is_executable_replacement_replay = executable_replacement_replay_pending
        if agent == "claude":
            invocation_kwargs["attempt_suffix"] = (
                "self-update-attempt2"
                if is_executable_replacement_replay
                else f"attempt{ordinary_retries_used + 1}"
            )
        elif is_executable_replacement_replay:
            invocation_kwargs["attempt_suffix"] = executable_replacement_policies[agent][2]
        result = run_agent_result(
            runner,
            agent=agent,
            config=attempt_config,
            prompt=prompt,
            session_id=session_id,
            run_id=usage_context.run_id if usage_context is not None else None,
            role=role,
            label=label,
            timeout_seconds=next_timeout_seconds,
            **invocation_kwargs,
        )
        # The bounded deadline belongs only to the interrupted invocation and
        # its dedicated replay. A later ordinary retry has its normal budget.
        if is_executable_replacement_replay:
            executable_replacement_replay_pending = False
            next_timeout_seconds = timeout_seconds
        last_result = result
        if result.log_path is not None:
            log_paths.append(result.log_path)
        text = result.text
        usage = _resolve_usage_metadata(config=config, prompt=prompt, result=result)
        usage_record = None
        if usage_context is not None and usage is not None:
            usage_record = usage_context.add_record(
                agent=agent,
                session_id=result.session_id,
                returncode=result.returncode,
                usage=usage,
                raw_backend_usage=result.raw_usage,
            )

        # A backend always loads the uniquely assigned public response file.
        # On a timeout/nonzero exit, that artifact can still be a complete
        # response; stdout is diagnostics only and must never be salvaged.
        artifact = result.response_file_text
        artifact_unavailable = None
        # Preserve the normal zero-exit path below, including its marker
        # recovery diagnostic. Failed exits alone may be salvaged from the
        # per-invocation response-file artifact.
        if artifact and result.returncode != 0:
            try:
                artifact_unavailable = parse_agent_unavailable(artifact)
            except AgentLoopError:
                artifact_unavailable = None
            if artifact_unavailable is None:
                try:
                    artifact_marker_value = validate(artifact)
                except AgentLoopError:
                    pass
                else:
                    acquisition_outcome = (
                        "accepted_timeout" if result.returncode is None
                        else "accepted_nonzero_exit" if result.returncode != 0 else "success"
                    )
                    if usage_record is not None:
                        usage_record.validation_status = "validated"
                        usage_record.outcome = acquisition_outcome
                    if acquisition_outcome != "success":
                        log(
                            config,
                            f"{agent_name}: accepted valid response-file artifact despite "
                            f"returncode={result.returncode!r}",
                        )
                    return ValidatedAgentResponse(
                        text=artifact,
                        session_id=result.session_id,
                        marker_value=artifact_marker_value,
                        usage=usage,
                        model_used=result.model_used,
                        acquisition_outcome=acquisition_outcome,
                        acquisition_returncode=result.returncode,
                    )
            else:
                # Let the existing agent-unavailable policy handle the valid
                # envelope, even when the command itself failed.
                text = artifact

        if result.self_update_replay_refusal_kind is not None:
            refusal_detail = result.self_update_replay_refusal_detail or (
                f"{agent_name} replay refused ({result.self_update_replay_refusal_kind})"
            )
            latest_replay_refusal_detail = refusal_detail
            refusal_outcome = (
                "self_update_replay_refused_changed_workdir"
                if result.self_update_replay_refusal_kind in {"changed-head", "changed-status"}
                else "self_update_replay_refused_unavailable_workdir"
            )
            if usage_record is not None:
                usage_record.outcome = refusal_outcome
                usage_record.log_path = str(result.log_path) if result.log_path else None
            log(
                config,
                f"{agent_name} attempt replay refusal: {refusal_detail}; "
                "continuing with provider-derived retry classification",
            )

        # A configured provider executable can be replaced after spawning. Each
        # backend supplies its own evidence gate, while this branch owns the
        # bounded stability wait, one replay, and retry accounting. A workdir
        # refusal deliberately remains diagnostic-only and cannot enter it.
        if (
            agent in executable_replacement_policies
            and result.self_update_reason is not None
            and result.self_update_replay_refusal_kind is None
            and not executable_replacement_considered
            and result.command_result is not None
        ):
            executable_replacement_considered = True
            executable_replacement_provider = agent
            executable_replacement_reason = result.self_update_reason
            observation = result.command_result.observation
            command, provider_label, _suffix, uses_remaining_deadline = (
                executable_replacement_policies[agent]
            )
            if uses_remaining_deadline:
                self_update_deadline = (
                    observation.spawn_monotonic + timeout_seconds
                    if timeout_seconds is not None and observation is not None
                    else None
                )
                stable = runner.wait_for_executable_stability(
                    config.claude_cmd, deadline=self_update_deadline
                )
                remaining = (
                    self_update_deadline - time.monotonic()
                    if self_update_deadline is not None else None
                )
                replay_timeout = remaining
                deadline_exhausted = remaining is not None and remaining <= 0
            else:
                # Codex, Gemini, and Antigravity use a bounded six-second
                # stability observation. Their replay is fresh and receives the
                # complete configured timeout, if any.
                stable = runner.wait_for_executable_stability(
                    command, deadline=None
                )
                replay_timeout = timeout_seconds
                deadline_exhausted = False
            if not stable or deadline_exhausted:
                replacement_stability_failed = True
                deadline_label = (
                    "within the invocation deadline"
                    if uses_remaining_deadline
                    else "within the bounded stability window"
                )
                self_update_stability_error = (
                    f"likely {provider_label} interruption ({result.self_update_reason}); "
                    f"executable did not stabilize {deadline_label}"
                )
                if usage_record is not None:
                    usage_record.outcome = (
                        "self_update_interruption"
                        if uses_remaining_deadline
                        else "executable_replacement_interruption"
                    )
                    usage_record.log_path = str(result.log_path) if result.log_path else None
                # Preserve the ordinary retry allowance: a failed stability
                # observation does not make provider errors final.
            else:
                executable_replacement_replay_pending = True
                if usage_record is not None:
                    usage_record.outcome = (
                        "self_update_interruption"
                        if uses_remaining_deadline
                        else "executable_replacement_interruption"
                    )
                    usage_record.log_path = str(result.log_path) if result.log_path else None
                next_timeout_seconds = replay_timeout
                log(
                    config,
                    f"{agent_name}: {result.self_update_reason}; replaying once after executable stability",
                )
                continue
        should_retry = False
        provider_capacity = False
        if result.returncode is None and artifact_unavailable is None:
            # Timed out (returncode=None from Runner.run_with_log). Detected
            # before transient classification: a kill deadline is not a
            # provider hiccup, so retrying or repairing would only waste the
            # same wall-clock budget again (#475).
            limit = f" after {timeout_seconds:g}s" if timeout_seconds is not None else ""
            last_error = f"agent command timed out{limit}"
            last_classification_text = ""
            last_failure_category = "timeout"
            break
        if result.returncode != 0 and artifact_unavailable is None:
            last_error = f"agent command exited with {result.returncode}"
            classification_text = _agent_failure_classification_text(result, phase="command")
            if result.command_result is not None and result.command_result.capture_diagnostics:
                classification_text += "\nsubprocess capture unavailable; retryable tooling failure"
            last_classification_text = classification_text
            should_retry = (
                replacement_stability_failed
                or bool(result.command_result and result.command_result.capture_diagnostics)
                or _is_transient_agent_output(classification_text)
            )
            last_failure_category = _failure_category(classification_text)
            capacity = classify_antigravity_capacity(
                classification_text,
                returncode=result.returncode,
                empty_response=False,
                signatures=config.antigravity_quota_signatures,
            ) if agent == "antigravity" else None
            provider_capacity = bool(capacity and capacity.is_capacity)
            if provider_capacity:
                should_retry = True
                last_failure_category = "transient"
        elif not text.strip():
            last_error = "agent response was empty"
            classification_text = _agent_failure_classification_text(result, phase="empty")
            last_classification_text = classification_text
            should_retry = replacement_stability_failed or _is_transient_agent_output(
                classification_text
            )
            last_failure_category = _failure_category(classification_text)
            capacity = classify_antigravity_capacity(
                classification_text,
                returncode=result.returncode,
                empty_response=True,
                signatures=config.antigravity_quota_signatures,
            ) if agent == "antigravity" else None
            provider_capacity = bool(capacity and capacity.is_capacity)
            if provider_capacity:
                should_retry = True
                last_failure_category = "transient"
        else:
            response_file_pre_status = (
                _response_file_structured_status(result.response_file_text)
                if result.response_file_text
                else None
            )
            try:
                unavailable = parse_agent_unavailable(text)
                if unavailable is not None:
                    raise _AgentUnavailableResponse(unavailable)
                marker_value = validate(text)
                if response_file_pre_status == "leading-public-response-marker-recovered":
                    log(
                        config,
                        f"{agent_name}: response file contained stdout filtering marker and "
                        "validated after stripping it",
                    )
            except _AgentUnavailableResponse as exc:
                unavailable = exc.unavailable
                last_error = (
                    f"agent explicitly reported it cannot continue ({unavailable.category}): "
                    f"{unavailable.summary}. Suggested action: {unavailable.suggested_action}"
                )
                classification_text = text
                last_classification_text = classification_text
                last_failure_category = "agent-unavailable"
                should_retry = unavailable.retryable
                log(
                    config,
                    f"{agent_name}: explicitly reported agent-unavailable "
                    f"({unavailable.category}, retryable={'yes' if unavailable.retryable else 'no'})",
                )
            except AgentLoopError as exc:
                last_error = str(exc)
                marker_safety_failure = "Current untrusted GitHub text contains reserved protocol marker(s):" in str(exc)
                classification_text = _agent_failure_classification_text(result, phase="validation")
                last_classification_text = classification_text
                public_text_is_transient = _is_transient_public_response(
                    classification_text,
                    repair_expected_kind=repair_expected_kind,
                )
                last_failure_category = _failure_category(
                    classification_text,
                    public_response=True,
                    repair_expected_kind=repair_expected_kind,
                )
                if result.command_result is not None and result.command_result.capture_diagnostics:
                    last_failure_category = "transient"
                    public_text_is_transient = True
                if (
                    completion_recovery is not None
                    and not completion_recovery_attempted
                    and agent == "claude"
                    and result.session_id
                    and looks_like_backgrounded_completion(classification_text)
                ):
                    # At most one same-session resume, ever, for this call
                    # (#588): mark it attempted before invoking so a failure
                    # cannot loop back into this branch on a later attempt.
                    completion_recovery_attempted = True
                    recovery_outcome = _attempt_claude_completion_recovery(
                        runner,
                        config=config,
                        completion_recovery=completion_recovery,
                        session_id=result.session_id,
                        validate=validate,
                        usage_context=usage_context,
                        run_id=usage_context.run_id if usage_context is not None else None,
                        role=role,
                        label=label,
                        timeout_seconds=timeout_seconds,
                    )
                    if recovery_outcome.validated is not None:
                        return recovery_outcome.validated
                    last_result = recovery_outcome.result
                    last_error = recovery_outcome.error
                    last_classification_text = recovery_outcome.classification_text
                    last_failure_category = recovery_outcome.failure_category
                    terminal_public_response = recovery_outcome.terminal_public_response
                    should_retry = False
                    break
                response_failure_is_unsupported = last_failure_category == "unsupported_model"
                # Marker near-misses are a separate first-attempt nudge for common footer typos;
                # structured JSON protocol drift still remains repairable when retries are exhausted.
                should_retry = replacement_stability_failed or public_text_is_transient or (
                    not response_failure_is_unsupported
                    and attempt == 1
                    and _is_retryable_marker_near_miss(classification_text)
                )
                if (
                    result.raw_output
                    and result.raw_output != classification_text
                    and _is_transient_agent_output(result.raw_output)
                    and not public_text_is_transient
                ):
                    log(
                        config,
                        f"{agent_name}: transient diagnostics were present outside the public response",
                    )
                response_file_status = None
                if result.response_file_text:
                    response_file_status = response_file_pre_status
                    if response_file_status == "leading-public-response-marker-not-recoverable":
                        log(
                            config,
                            f"{agent_name}: response file contained stdout filtering marker but "
                            "the remainder was not recoverable",
                        )
                    elif response_file_status in {"markdown-or-prose", "fenced-or-markdown"}:
                        log(
                            config,
                            f"{agent_name}: public response file was not structured "
                            f"({response_file_status})",
                        )
                response_file_not_structured = response_file_status in {
                    "leading-public-response-marker-not-recoverable",
                    "markdown-or-prose",
                    "fenced-or-markdown",
                }
                if (
                    result.response_file_text
                    and response_file_not_structured
                    and not response_failure_is_unsupported
                ):
                    recovered = _recover_valid_structured_candidate(
                        result,
                        validate=validate,
                        expected_kind=repair_expected_kind,
                        config=config,
                        agent_name=agent_name,
                    )
                    if recovered is not None:
                        recovered_text, marker_value = recovered
                        if usage_record is not None:
                            usage_record.validation_status = "validated"
                        return ValidatedAgentResponse(
                            text=recovered_text,
                            session_id=result.session_id,
                            marker_value=marker_value,
                            usage=usage,
                            model_used=result.model_used,
                        )
                if (
                    not response_failure_is_unsupported
                    and repair_expected_kind == "plan_revision"
                    and result.response_file_text
                    and not isinstance(exc, UnknownPriorItemDispositionError)
                    and _plan_revision_missing_human_acknowledgement(
                        result.text,
                        context=_HumanRequirementsRecoveryContext(
                            surfaced_requirement_ids=tuple(
                                repair_surfaced_requirement_ids or ()
                            ),
                            requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                        ),
                    )
                ):
                    recovered = _recover_plan_revision_human_requirements_acknowledgement(
                        result,
                        validate=validate,
                        context=_HumanRequirementsRecoveryContext(
                            surfaced_requirement_ids=tuple(
                                repair_surfaced_requirement_ids or ()
                            ),
                            requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                        ),
                        config=config,
                        agent_name=agent_name,
                    )
                    if recovered is not None:
                        recovered_text, marker_value = recovered
                        if usage_record is not None:
                            usage_record.validation_status = "validated"
                        return ValidatedAgentResponse(
                            text=recovered_text,
                            session_id=result.session_id,
                            marker_value=marker_value,
                            usage=usage,
                            model_used=result.model_used,
                        )
                normalized: str | None = None
                if (
                    use_repair
                    and not public_text_is_transient
                    and not response_failure_is_unsupported
                    and repair_expected_kind in STRUCTURED_PUBLIC_RESPONSE_KINDS
                    and not (
                        isinstance(exc, UnknownPriorItemDispositionError)
                        and ledger_incomplete
                    )
                ):
                    normalized = attempt_envelope_normalization(
                        text,
                        expected_kind=repair_expected_kind,
                    )
                    if normalized is not None:
                        try:
                            marker_value = validate(normalized)
                        except UnknownPriorItemDispositionError as norm_exc:
                            # Combined fix (issue #274): envelope normalization removed the
                            # trailing defect but the normalized candidate still has unknown
                            # prior dispositions. Try stripping them from the normalized text
                            # so both defects are resolved in one deterministic pass.
                            # Only apply when the original error was structural; when it was
                            # already UnknownPriorItemDispositionError, block 2 handles it.
                            if (
                                not isinstance(exc, UnknownPriorItemDispositionError)
                                and not ledger_incomplete
                                and repair_expected_kind in {"pr_review", "plan_review", "plan_revision"}
                            ):
                                stripped_from_normalized = strip_unknown_prior_item_dispositions(
                                    normalized,
                                    allowed_ids=frozenset(norm_exc.allowed_ids),
                                    expected_kind=repair_expected_kind,
                                )
                                if stripped_from_normalized is not None:
                                    try:
                                        marker_value = validate(stripped_from_normalized)
                                    except AgentLoopError:
                                        if (
                                            repair_expected_kind == "plan_revision"
                                            and result.response_file_text
                                            and _plan_revision_missing_human_acknowledgement(
                                                stripped_from_normalized,
                                                context=_HumanRequirementsRecoveryContext(
                                                    surfaced_requirement_ids=tuple(
                                                        repair_surfaced_requirement_ids or ()
                                                    ),
                                                    requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                                                ),
                                            )
                                        ):
                                            recovered = _recover_plan_revision_human_requirements_acknowledgement(
                                                result,
                                                text=stripped_from_normalized,
                                                validate=validate,
                                                context=_HumanRequirementsRecoveryContext(
                                                    surfaced_requirement_ids=tuple(
                                                        repair_surfaced_requirement_ids or ()
                                                    ),
                                                    requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                                                ),
                                                config=config,
                                                agent_name=agent_name,
                                            )
                                            if recovered is not None:
                                                recovered_text, marker_value = recovered
                                                if usage_record is not None:
                                                    usage_record.validation_status = "validated"
                                                return ValidatedAgentResponse(
                                                    text=recovered_text,
                                                    session_id=result.session_id,
                                                    marker_value=marker_value,
                                                    usage=usage,
                                                    model_used=result.model_used,
                                                )
                                    else:
                                        removed = ", ".join(sorted(norm_exc.unknown_ids))
                                        allowed_str = ", ".join(sorted(norm_exc.allowed_ids)) or "(none)"
                                        log(
                                            config,
                                            f"{agent_name}: combined envelope normalization and "
                                            f"deterministic strip recovered malformed response; "
                                            f"removed prior-item ID(s) {removed}; "
                                            f"allowed carried prior IDs: {allowed_str}",
                                        )
                                        if usage_record is not None:
                                            usage_record.validation_status = "validated"
                                        return ValidatedAgentResponse(
                                            text=stripped_from_normalized,
                                            session_id=result.session_id,
                                            marker_value=marker_value,
                                            usage=usage,
                                            model_used=result.model_used,
                                        )
                        except AgentLoopError:
                            pass
                        else:
                            log(
                                config,
                                f"{agent_name}: envelope normalization recovered malformed response",
                            )
                            if usage_record is not None:
                                usage_record.validation_status = "validated"
                            return ValidatedAgentResponse(
                                text=normalized,
                                session_id=result.session_id,
                                marker_value=marker_value,
                                usage=usage,
                                model_used=result.model_used,
                            )
                if (
                    use_repair
                    and not public_text_is_transient
                    and not response_failure_is_unsupported
                    and isinstance(exc, UnknownPriorItemDispositionError)
                    and not ledger_incomplete
                    and repair_expected_kind in {"pr_review", "plan_review", "plan_revision"}
                ):
                    stripped_text = strip_unknown_prior_item_dispositions(
                        text,
                        allowed_ids=frozenset(exc.allowed_ids),
                        expected_kind=repair_expected_kind,
                    )
                    if stripped_text is not None:
                        try:
                            marker_value = validate(stripped_text)
                        except AgentLoopError:
                            if (
                                repair_expected_kind == "plan_revision"
                                and result.response_file_text
                                and _plan_revision_missing_human_acknowledgement(
                                    stripped_text,
                                    context=_HumanRequirementsRecoveryContext(
                                        surfaced_requirement_ids=tuple(
                                            repair_surfaced_requirement_ids or ()
                                        ),
                                        requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                                    ),
                                )
                            ):
                                recovered = _recover_plan_revision_human_requirements_acknowledgement(
                                    result,
                                    text=stripped_text,
                                    validate=validate,
                                    context=_HumanRequirementsRecoveryContext(
                                        surfaced_requirement_ids=tuple(
                                            repair_surfaced_requirement_ids or ()
                                        ),
                                        requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                                    ),
                                    config=config,
                                    agent_name=agent_name,
                                )
                                if recovered is not None:
                                    recovered_text, marker_value = recovered
                                    if usage_record is not None:
                                        usage_record.validation_status = "validated"
                                    return ValidatedAgentResponse(
                                        text=recovered_text,
                                        session_id=result.session_id,
                                        marker_value=marker_value,
                                        usage=usage,
                                        model_used=result.model_used,
                                    )
                        else:
                            removed = ", ".join(sorted(exc.unknown_ids))
                            allowed_str = ", ".join(sorted(exc.allowed_ids)) or "(none)"
                            log(
                                config,
                                f"{agent_name}: deterministically removed unknown prior-item "
                                f"disposition ID(s) {removed}; allowed carried prior IDs: {allowed_str}",
                            )
                            if usage_record is not None:
                                usage_record.validation_status = "validated"
                            return ValidatedAgentResponse(
                                text=stripped_text,
                                session_id=result.session_id,
                                marker_value=marker_value,
                                usage=usage,
                                model_used=result.model_used,
                            )
                if (
                    use_repair
                    and not public_text_is_transient
                    and not response_failure_is_unsupported
                    and (not marker_safety_failure or not marker_safety_repair_attempted)
                    and not (
                        isinstance(exc, UnknownPriorItemDispositionError)
                        and ledger_incomplete
                    )
                ):
                    log(config, f"{agent_name}: schema validation failed ({exc}); attempting repair pass")
                    repair_kwargs: dict[str, object] = {"expected_kind": repair_expected_kind}
                    if repair_unresolved_item_ids is not None:
                        repair_kwargs["unresolved_item_ids"] = tuple(repair_unresolved_item_ids)
                    if repair_expected_kind in {"issue_implementation", "plan_state", "plan_revision"}:
                        repair_kwargs["surfaced_requirement_ids"] = tuple(
                            repair_surfaced_requirement_ids or ()
                        )
                        repair_kwargs["requires_direct_discussion_ack"] = (
                            repair_requires_direct_discussion_ack
                        )
                    elif (
                        repair_expected_kind == "coder_followup"
                        and (
                            repair_surfaced_requirement_ids is not None
                            or repair_requires_direct_discussion_ack
                        )
                    ):
                        repair_kwargs["surfaced_requirement_ids"] = tuple(repair_surfaced_requirement_ids or ())
                        repair_kwargs["requires_direct_discussion_ack"] = repair_requires_direct_discussion_ack
                    elif (
                        repair_expected_kind in {"plan_review", "pr_review"}
                        and repair_reviewer_requirement_ids is not None
                    ):
                        repair_kwargs["reviewer_requirement_ids"] = tuple(
                            repair_reviewer_requirement_ids
                        )
                    if isinstance(exc, UnknownPriorItemDispositionError):
                        repair_kwargs["allowed_prior_item_ids"] = exc.allowed_ids
                        repair_kwargs["unknown_prior_item_ids"] = exc.unknown_ids
                        repair_kwargs["same_round_context"] = exc.same_round_description
                    elif repair_allowed_prior_item_ids is not None:
                        repair_kwargs["allowed_prior_item_ids"] = tuple(repair_allowed_prior_item_ids)
                    original_validation_error = str(exc)
                    if marker_safety_failure:
                        marker_safety_repair_attempted = True
                    repaired, repaired_marker, repair_attempts = _run_structured_repair(
                        normalized if normalized is not None else text,
                        runner=runner,
                        config=config,
                        usage_context=usage_context,
                        validate=validate,
                        repair_kwargs=repair_kwargs,
                    )
                    _log_repair_attempts(config, agent_name, repair_attempts)
                    if repaired is not None:
                        if repaired_marker is None:
                            repair_detail = (
                                repair_attempts[-1].diagnostic
                                if repair_attempts
                                else "repair output failed validation"
                            )
                            last_error = (
                                f"{original_validation_error}; repair failure: {repair_detail}"
                            )
                            log(
                                config,
                                f"{agent_name}: repair pass produced invalid output ({repair_detail})",
                            )
                        else:
                            marker_value = repaired_marker
                            if isinstance(exc, UnknownPriorItemDispositionError):
                                removed = ", ".join(sorted(exc.unknown_ids))
                                allowed = ", ".join(sorted(exc.allowed_ids)) or "(none)"
                                log(
                                    config,
                                    f"{agent_name}: repair pass removed unknown prior-item "
                                    f"disposition ID(s) {removed}; allowed carried prior IDs: {allowed}",
                                )
                            else:
                                log(config, f"{agent_name}: repair pass recovered malformed response")
                            if usage_record is not None:
                                usage_record.validation_status = "validated"
                            return ValidatedAgentResponse(
                                text=repaired,
                                session_id=result.session_id,
                                marker_value=marker_value,
                                usage=usage,
                                model_used=result.model_used,
                            )
                    elif repair_attempts:
                        details = "; ".join(
                            f"{attempt.backend}/{attempt.model}: {attempt.outcome}"
                            + (f" ({attempt.diagnostic})" if attempt.diagnostic else "")
                            for attempt in repair_attempts
                        )
                        last_error = f"{original_validation_error}; repair invocation failure: {details}"
            else:
                if usage_record is not None:
                    usage_record.validation_status = "validated"
                return ValidatedAgentResponse(
                    text=text,
                    session_id=result.session_id,
                    marker_value=marker_value,
                    usage=usage,
                    model_used=result.model_used,
                )

        if should_retry:
            if _QUOTA_RATE_LIMIT_RE.search(classification_text):
                reset_secs = _parse_rate_limit_reset_seconds(classification_text)
                if reset_secs is not None and reset_secs > LONG_RESET_THRESHOLD_SECONDS:
                    duration_str = _format_reset_duration(reset_secs)
                    at_str = _format_reset_at_utc(reset_secs)
                    message = (
                        f"{agent_name} quota exhausted. Reset in {duration_str} (at {at_str}). "
                        "Rerun when quota resets, or switch to a different API key / model."
                    )
                    replacement_detail = _executable_replacement_failure_detail(
                        provider=executable_replacement_provider,
                        reason=executable_replacement_reason,
                        stability_error=self_update_stability_error,
                    )
                    diagnostic_classification_text = classification_text
                    if replacement_detail:
                        message += f" {replacement_detail}"
                        diagnostic_classification_text = (
                            f"{classification_text}\n{replacement_detail}"
                        ).strip()
                    if latest_replay_refusal_detail:
                        message += f" {latest_replay_refusal_detail}"
                        diagnostic_classification_text = (
                            f"{diagnostic_classification_text}\n{latest_replay_refusal_detail}"
                        ).strip()
                    diagnostics = _failed_run_diagnostics(
                        runner=runner,
                        config=config,
                        agent_name=agent_name,
                        salvage_context=salvage_context,
                        operation_description=operation_description,
                        failure_category=last_failure_category,
                        failure_reason=message,
                        classification_text=diagnostic_classification_text,
                        marker_description=marker_description,
                        result=last_result,
                    )
                    message += diagnostics.format_for_error()
                    raise QuotaResetExceededError(message)
            transition = (
                antigravity_attempts.next_after_failure(
                    retryable=should_retry, provider_capacity=provider_capacity
                )
                if antigravity_attempts is not None
                else ("retry" if ordinary_retries_used < config.agent_max_retries else "stop")
            )
            if transition == "retry":
                if antigravity_attempts is None:
                    ordinary_retries_used += 1
                delay = _retry_delay(config, attempt)
                category = last_failure_category
                retry_attempt = (
                    ordinary_retries_used + 1
                    if antigravity_attempts is None
                    else attempt + 1
                )
                retry_budget = (
                    config.agent_max_retries + 1
                    if antigravity_attempts is None
                    else max_attempts
                )
                log(
                    config,
                    f"{agent_name}: {category} failure ({last_error}); "
                    f"retrying in {delay}s (attempt {retry_attempt}/{retry_budget})",
                )
                runner.run(("sleep", str(delay)), cwd=active_workdir(config))
                continue
            if transition == "fallback":
                log(
                    config,
                    f"{agent_name}: provider capacity exhausted for "
                    f"{result.model_used}; trying {antigravity_attempts.models[antigravity_attempts.model_index]} "
                    "without additional delay",
                )
                continue
        break

    if executable_replacement_provider is not None:
        replacement_detail = _executable_replacement_failure_detail(
            provider=executable_replacement_provider,
            reason=executable_replacement_reason,
            stability_error=self_update_stability_error,
        )
        context_details = [replacement_detail]
        if latest_replay_refusal_detail:
            context_details.append(latest_replay_refusal_detail)
        last_error = f"{'; '.join(context_details)}; final failure: {last_error}"
        classification_parts = [last_classification_text, replacement_detail]
        if latest_replay_refusal_detail:
            classification_parts.append(latest_replay_refusal_detail)
        last_classification_text = "\n".join(
            part for part in classification_parts if part
        ).strip()
        last_failure_category = (
            "self-update-interruption"
            if executable_replacement_provider == "claude"
            else "executable-replacement"
        )
    elif latest_replay_refusal_detail:
        # Refusal context is added only after all retry decisions. In
        # particular, the provider-derived category remains untouched.
        last_error = f"{latest_replay_refusal_detail}; final failure: {last_error}"
        last_classification_text = "\n".join(
            part for part in (last_classification_text, latest_replay_refusal_detail) if part
        ).strip()
    diagnostics = _failed_run_diagnostics(
        runner=runner,
        config=config,
        agent_name=agent_name,
        salvage_context=salvage_context,
        operation_description=operation_description,
        failure_category=last_failure_category,
        failure_reason=last_error,
        classification_text=last_classification_text,
        marker_description=marker_description,
        result=last_result,
    )
    message = _format_invalid_agent_response_error(
        agent_name=agent_name,
        marker_description=marker_description,
        reason=last_error,
        result=last_result,
        log_paths=log_paths,
        category=last_failure_category,
        agent=agent,
        config=config,
        role=role,
        classification_text=last_classification_text,
    )
    message += diagnostics.format_for_error()
    raise AgentInvocationError(
        message,
        failure_category=last_failure_category,
        terminal_public_response=terminal_public_response,
    )


@dataclass(frozen=True)
class _TerminalNoPrImplementation:
    state: str


@dataclass(frozen=True)
class _TerminalIssueImplementationConflict:
    """A valid implementation payload rejected from handoff by semantics."""

    parsed: StructuredIssueImplementation


def _validate_issue_implementation_contract(
    parsed: StructuredIssueImplementation,
    *,
    human_requirements,
) -> None:
    prompt_context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="implementation requirements",
        full_omission_fallback="Fetch the issue discussion directly before implementing.",
    )
    validate_human_requirement_dispositions(
        parsed.human_requirement_dispositions,
        surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
        context="issue_implementation.human_requirement_dispositions",
    )
    validate_structured_human_requirements_acknowledgement(
        parsed.human_requirements.addressed_ids,
        dispositions=parsed.human_requirement_dispositions,
        checked_discussion_directly=parsed.human_requirements.checked_discussion_directly,
        surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
        requires_direct_discussion_ack=prompt_context.requires_direct_discussion_ack,
    )


def _validate_issue_implementation_response(
    text: str,
    *,
    human_requirements,
) -> StructuredIssueImplementation | _TerminalNoPrImplementation | _TerminalIssueImplementationConflict:
    """Validate an implementation result and isolate the terminal conflict path."""
    if is_clarification_request(text):
        return _TerminalNoPrImplementation("clarification")
    try:
        parsed = validate_structured_issue_implementation(text)
    except IssueImplementationConflictError as exc:
        parsed = exc.payload
        if not isinstance(parsed, StructuredIssueImplementation):
            raise AgentLoopError("Issue implementation conflict did not retain a typed payload.") from exc
        # A conflict is terminal only after the normal acknowledgement and
        # exact-ledger contract has been proven valid for this issue.
        _validate_issue_implementation_contract(parsed, human_requirements=human_requirements)
        return _TerminalIssueImplementationConflict(parsed)
    if parsed is None:
        raise AgentLoopError(
            "Issue implementation response must use the required structured `issue_implementation` format."
        )
    _validate_issue_implementation_contract(parsed, human_requirements=human_requirements)
    return parsed


def _post_no_pr_implementation_terminal_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    coder_response: ValidatedAgentResponse,
) -> None:
    """Post a genuine coder-declared no-PR blocking/clarify result to GitHub (#588).

    Matches how a PR-success implementation result is already posted
    (post_pr_comment / post_issue_pr_handoff_comment): the coder's own text
    is the actionable public record, whether or not a PR was created.
    """
    post_issue_comment(
        runner,
        config=config,
        issue_number=issue_number,
        body=normalize_freeform_signature(
            coder_response.text,
            agent=config.coder,
            config=config,
            model_used=coder_response.model_used,
        ),
    )


def _post_structured_issue_implementation_terminal_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    parsed: StructuredIssueImplementation,
    model_used: str | None,
) -> None:
    """Publish a typed no-PR or rejected-conflict implementation result."""
    post_issue_comment(
        runner,
        config=config,
        issue_number=issue_number,
        body=render_public_agent_comment(
            kind="issue_implementation",
            parsed=parsed,
            agent=config.coder,
            config=config,
            model_used=model_used,
        ),
    )


def _require_pr_number(text: str) -> int:
    pr_number = parse_pr_number(text)
    if pr_number is None:
        raise AgentLoopError("Agent response did not include a valid positive PR marker or PR URL.")
    return pr_number


def _require_pr_number_or_clarification(text: str) -> int | str:
    pr_number = parse_pr_number(text)
    if pr_number is not None:
        return pr_number
    if is_clarification_request(text):
        return "clarification"
    raise AgentLoopError(
        "Agent response did not include a PR marker, PR URL, or clarification marker."
    )


def _require_task_implementation_result(text: str) -> int | str | _TerminalNoPrImplementation:
    """Accept a positive PR, a clarification request, or a terminal no-PR blocking result (#604)."""
    pr_number = parse_pr_number(text)
    if pr_number is not None:
        return pr_number
    if is_clarification_request(text):
        return "clarification"
    try:
        state = parse_agent_state(text)
    except AgentLoopError:
        state = None
    if state == "blocking":
        return _TerminalNoPrImplementation("blocking")
    raise AgentLoopError(
        "Agent response did not include a PR marker, PR URL, clarification marker, "
        "or a terminal blocking marker."
    )


def _require_plan_state_or_clarification(text: str) -> StructuredPlanState | str:
    if is_clarification_request(text):
        return "clarification"
    structured_plan = validate_structured_plan_state(text)
    if structured_plan is None:
        raise AgentLoopError(
            "Initial planning response must include a structured `plan_state` JSON object."
        )
    return structured_plan
def _validate_response_with_human_requirements(
    text: str,
    *,
    marker_validator: Callable[[str], object],
    human_requirements,
    requirement_scope: str,
    full_omission_fallback: str,
) -> object:
    marker_value = marker_validator(text)
    prompt_context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope=requirement_scope,
        full_omission_fallback=full_omission_fallback,
    )
    validate_human_requirements_acknowledgement(
        text,
        surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
        requires_direct_discussion_ack=prompt_context.requires_direct_discussion_ack,
    )
    if hasattr(marker_value, "human_requirement_dispositions"):
        validate_human_requirement_dispositions(
            marker_value.human_requirement_dispositions,
            surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
            context=f"{getattr(marker_value, 'kind', 'structured')}.human_requirement_dispositions",
        )
    return marker_value


def _current_plan_has_complete_human_requirement_dispositions(
    coder_output: str | None,
    *,
    surfaced_requirement_ids: Sequence[str],
) -> bool:
    """Return whether the current coder plan has the required attestation.

    The raw structured coder response is retained in round metadata specifically
    so resume can apply this same gate to a canonical markdown revision.
    """
    if coder_output is None:
        return False
    try:
        try:
            parsed = validate_structured_plan_state(coder_output)
        except AgentLoopError:
            parsed = validate_structured_plan_revision(coder_output)
        if parsed is None:
            return False
        validate_human_requirement_dispositions(
            parsed.human_requirement_dispositions,
            surfaced_requirement_ids=surfaced_requirement_ids,
            context=f"{parsed.kind}.human_requirement_dispositions",
        )
    except AgentLoopError:
        return False
    return True


def _merge_human_requirements(
    issue_context: IssueContext | None,
    pr_context: PullRequestReviewContext,
):
    combined = list(issue_context.human_requirements if issue_context is not None else ())
    combined.extend(pr_context.human_requirements)
    return tuple(sorted(combined, key=lambda requirement: requirement.created_at or ""))


def _surfaced_reviewer_requirement_ids(
    human_requirements: Sequence,
    *,
    requirement_scope: str,
) -> tuple[str, ...]:
    return render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope=requirement_scope,
    ).surfaced_requirement_ids


def _validate_plan_revision_response(
    text: str,
    *,
    unresolved_items: Sequence[UnresolvedReviewItem] = (),
) -> StructuredPlanRevision | str:
    parsed = validate_structured_plan_revision(text)
    if parsed is not None:
        allowed_ids = {item.item_id for item in unresolved_items}
        unknown = {
            disposition.item_id
            for disposition in parsed.prior_plan_item_dispositions
        } - allowed_ids
        if unknown:
            raise UnknownPriorItemDispositionError(
                unknown_ids=tuple(sorted(unknown)),
                allowed_ids=tuple(sorted(allowed_ids)),
                same_round_description=(
                    "Same-round findings are informational only and must not be "
                    "dispositioned as prior carried items."
                ),
            )
        return parsed
    raise AgentLoopError("Plan revision did not use the required structured format.")


_PENDING_CI_TEXT_KEYWORDS = (
    "pending",
    "in progress",
    "in_progress",
    "queued",
    "still running",
    "not yet report",
    "unavailable",
    "check status",
    "github check",
    "ci check",
)


def _drop_repeated_carried_future_followups(
    followups: ApprovedFollowups,
    *,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
) -> ApprovedFollowups:
    """Keep carried future work in its original ledger item.

    A reviewer records a carried item's status through `prior_item_dispositions`.
    Repeating that same concern in `future_followups` used to allocate another
    item ID, even though final follow-up publishing later grouped the two.
    """
    future_ids = {
        disposition.item_id
        for disposition in dispositions
        if disposition.disposition == "future"
    }
    carried = [
        _approved_followup_from_unresolved_item(item)
        for item in prior_items
        if item.item_id in future_ids
    ]
    if not carried or not followups.future:
        return followups

    carried_group_count = len(_dedupe_approved_followups(carried))
    retained = tuple(
        followup
        for followup in followups.future
        if len(_dedupe_approved_followups([*carried, followup])) > carried_group_count
    )
    if retained == followups.future:
        return followups
    return ApprovedFollowups(same_pr=followups.same_pr, future=retained)


def _drop_repeated_carried_plan_future_followups(
    items: PlanReviewItems,
    *,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
) -> PlanReviewItems:
    """Keep repeated carried plan follow-ups in their original ledger items."""
    followups = _drop_repeated_carried_future_followups(
        ApprovedFollowups(same_pr=items.same_plan, future=items.future),
        prior_items=prior_items,
        dispositions=dispositions,
    )
    if followups.future == items.future:
        return items
    return PlanReviewItems(
        blocking=items.blocking,
        same_plan=items.same_plan,
        future=followups.future,
    )


def _is_pending_ci_only_review(parsed_review: ParsedReview, pr_checks: PullRequestChecks) -> bool:
    """Detect a blocking review whose only content restates pending/unavailable
    GitHub check status rather than an actionable code-level finding.

    This is a defense-in-depth backstop: the reviewer prompt already instructs
    reviewers not to use pending/unavailable checks as the sole reason to
    block, but a reviewer may still do so. Any other content (a distinct
    blocking item, or a Same-PR follow-up) causes this to return False so
    mixed responses still route back to the coder normally.
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
    check_names = {check.name.lower() for check in pr_checks.pending}
    check_names.update(name.lower() for name in pr_checks.missing_required)
    for text in candidate_texts:
        lowered = text.lower()
        mentions_check_name = any(name in lowered for name in check_names)
        mentions_ci_keyword = any(keyword in lowered for keyword in _PENDING_CI_TEXT_KEYWORDS)
        if not (mentions_check_name or mentions_ci_keyword):
            return False
    return True


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

    Unlike `_is_pending_ci_only_review`'s keyword heuristic, this requires the
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


_INCOMPLETE_PR_REVIEW_RE = re.compile(
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
    return any(text and _INCOMPLETE_PR_REVIEW_RE.search(text) for text in candidate_texts)


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
) -> str:
    lines = [
        "**Review status: Incomplete**",
        "",
        "No coder follow-up was started for the unavailable reviewer(s). "
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
    lines.extend(
        [
            "",
            "Resolve the reviewer problem or rerun with a replacement reviewer/model "
            f"before merging PR #{pr_number}.",
        ]
    )
    return "\n".join(lines)


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
    return any(text and _INCOMPLETE_PR_REVIEW_RE.search(text) for text in candidate_texts)


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


_DEFERRED_STAGES_SECTION_RE = re.compile(
    r"^###\s*Deferred stages \(not in this plan\)\s*$",
    re.M,
)
_NEXT_HEADING_RE = re.compile(r"^#{1,6}\s", re.M)
_PLAN_NARROWING_PHRASE_RE = re.compile(
    r"\b(stage \d+ of|first stage|out of scope|separate issue|follow-up issue|future issue)\b",
    re.I,
)


def _extract_current_deferred_stages(current_plan: str) -> tuple[DeferredStage, ...]:
    """Recover declared `deferred_stages` from the current plan text (#476).

    `current_plan` is either the raw structured `plan_state` JSON response (the
    initial round, never revised) or the canonical revision markdown (after a
    `plan_revision` round), so both forms are checked.
    """
    try:
        structured = validate_structured_plan_state(current_plan)
    except AgentLoopError:
        structured = None
    if structured is not None:
        return structured.deferred_stages
    # The canonical markdown carries an AGENT_DEFERRED_STAGES marker with the
    # exact structured title/summary pairs (#492 review): a title containing
    # its own colon (e.g. "Stage 2: API follow-up") would corrupt the
    # human-readable `- {title}: {summary}` bullets if split on the first
    # colon, so the marker is authoritative and the prose is parsed only as a
    # fallback for text that predates it.
    marker_match = DEFERRED_STAGES_MARKER_RE.search(current_plan)
    if marker_match:
        return decode_deferred_stages_marker(marker_match.group("payload"))
    match = _DEFERRED_STAGES_SECTION_RE.search(current_plan)
    if not match:
        return ()
    section = current_plan[match.end():]
    next_heading = _NEXT_HEADING_RE.search(section)
    if next_heading:
        section = section[: next_heading.start()]
    stages: list[DeferredStage] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        title_and_summary = stripped[2:]
        title, _sep, summary = title_and_summary.partition(":")
        stages.append(DeferredStage(title=title.strip(), summary=summary.strip()))
    return tuple(stages)


def _extract_current_expected_closing_issue_ids(
    current_plan: str,
) -> tuple[int, ...] | None:
    """Recover the optional plan declaration from JSON or canonical Markdown."""
    for validator in (validate_structured_plan_state, validate_structured_plan_revision):
        try:
            structured = validator(current_plan)
        except AgentLoopError:
            structured = None
        if structured is not None:
            return structured.additional_closing_issue_ids
    marker = PLAN_EXPECTED_CLOSING_MARKER_RE.search(current_plan)
    if marker is None:
        return None
    return decode_expected_closing_issue_declaration(marker.group("payload"))


def _extract_current_child_stages(current_plan: str) -> tuple[ChildStage, ...]:
    """Return only explicitly typed child stages; legacy entries are record-only."""
    try:
        structured = validate_structured_plan_state(current_plan)
    except AgentLoopError:
        structured = None
    if structured is not None:
        return structured.typed_stages.child_stages
    marker = re.search(r"<!--\s*AGENT_TYPED_PLAN_STAGES:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", current_plan, re.I)
    if not marker:
        return ()
    try:
        payload = _decode_json_payload(marker.group("payload"), marker_name="AGENT_TYPED_PLAN_STAGES")
        children = payload.get("child_stages", [])
        if not isinstance(children, list):
            return ()
        return tuple(
            ChildStage(str(item["title"]), str(item["summary"]))
            for item in children
            if isinstance(item, dict)
            and isinstance(item.get("title"), str)
            and isinstance(item.get("summary"), str)
        )
    except AgentLoopError:
        return ()


def _log_typed_plan_stage_dispositions(current_plan: str, *, config: AgentLoopConfig) -> None:
    """Make the record-only typed categories visible in CLI output (#585)."""
    try:
        structured = validate_structured_plan_state(current_plan)
    except AgentLoopError:
        structured = None
    if structured is not None:
        for entry in structured.deferred_stages:
            log(config, f"Plan scope: recorded-only legacy deferred stage: {entry.title}.")
        categories = (
            ("linked dependency", structured.typed_stages.external_dependencies),
            ("recorded-only deferred work", structured.typed_stages.deferred_work),
            ("recorded-only plan action", structured.typed_stages.plan_actions),
        )
        for disposition, entries in categories:
            for entry in entries:
                log(config, f"Plan scope: {disposition}: {entry.title}.")
        return
    for entry in _extract_current_deferred_stages(current_plan):
        log(config, f"Plan scope: recorded-only legacy deferred stage: {entry.title}.")
    marker = re.search(
        r"<!--\s*AGENT_TYPED_PLAN_STAGES:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
        current_plan,
        re.I,
    )
    if not marker:
        return
    try:
        payload = _decode_json_payload(
            marker.group("payload"), marker_name="AGENT_TYPED_PLAN_STAGES"
        )
    except AgentLoopError:
        return
    for field, disposition in (
        ("external_dependencies", "linked dependency"),
        ("deferred_work", "recorded-only deferred work"),
        ("plan_actions", "recorded-only plan action"),
    ):
        entries = payload.get(field, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("title"), str):
                log(config, f"Plan scope: {disposition}: {entry['title']}.")


def _prior_discuss_split_proposals(
    issue_context: IssueContext, *, config: AgentLoopConfig
) -> list[str]:
    """Recover proposals from a prior discuss `split` consensus on this issue (#476).

    Used by plan-first narrowing so a plan built on top of an earlier discuss
    split still files (or warns about) the stages the plan itself doesn't
    cover, even when discuss-mode materialization was never run.
    """
    records = _extract_round_metadata_records(issue_context.comments, flow="discuss")
    final_summaries = [
        record for record in records if record.metadata.role == "summary" and record.metadata.is_final
    ]
    if not final_summaries:
        return []
    latest = final_summaries[-1]
    if latest.metadata.split_proposals:
        return list(latest.metadata.split_proposals)
    configured_reviewers = reviewers(config)
    reviewer_workdirs = {
        agent_display_name(agent): get_backend(agent).workdir(config) for agent in configured_reviewers
    }
    recovered = _recover_final_discuss_split_proposals(
        issue_context,
        subject=latest.metadata.subject,
        configured_reviewers=configured_reviewers,
        reviewer_workdirs=reviewer_workdirs,
    )
    return list(recovered[0]) if recovered else []


def _plan_text_suggests_narrowing(plan_text: str) -> bool:
    return bool(_PLAN_NARROWING_PHRASE_RE.search(plan_text))


def _plan_first_line(plan_text: str) -> str:
    """A title-like string for `plan_text`, used for split-stage title matching.

    For the initial (never-revised) `plan_state` round, `plan_text` is the raw
    structured JSON response rather than rendered markdown, so its natural
    "title" is the structured `summary` field, not its literal first line.
    """
    try:
        structured = validate_structured_plan_state(plan_text)
    except AgentLoopError:
        structured = None
    if structured is not None:
        return structured.summary
    for line in plan_text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped
    return plan_text.strip()


def _handle_plan_first_split_scope(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    current_plan: str,
    plan_subject: str,
    issue_context: IssueContext,
) -> bool:
    """Materialize (or warn about) split/deferred stages before implementation
    handoff (#476), so a plan-first run that narrows scope to one stage cannot
    silently leave the rest unfiled (the #467/#474 gap).

    Returns True when this call may have posted a fresh `AGENT_DISCUSS_SPLIT`
    materialization comment on the parent (#492 review): the caller must then
    refetch `issue_context` before any downstream logic (e.g. implement-one-shot
    selected-stage resolution) reads issue comments, since the in-memory
    `issue_context.comments` snapshot predates this call and would otherwise
    look stale and hide children materialized moments earlier in this same run.
    """
    current_deferred_stages = _extract_current_deferred_stages(current_plan)
    current_child_stages = _extract_current_child_stages(current_plan)
    _log_typed_plan_stage_dispositions(current_plan, config=config)
    prior_discuss_proposals = _prior_discuss_split_proposals(issue_context, config=config)
    if current_child_stages or current_deferred_stages:
        # This plan structurally declares its own deferred_stages, so it keeps
        # a primary scope on the parent (the implement-one-shot branch below
        # never hands that scope off to a child in this case). If a prior
        # discuss split proposal names that same primary scope, it must be
        # excluded here rather than filed as a duplicate child issue for work
        # the parent PR is about to implement and close directly (#492 review).
        plan_own_key = split_stage_proposal_from_text(_plan_first_line(current_plan)).key
        prior_discuss_proposals = [
            proposal
            for proposal in prior_discuss_proposals
            if split_stage_proposal_from_text(proposal).key != plan_own_key
        ]
    remaining_proposals = dedupe_split_stage_proposals(
        [split_stage_proposal_from_deferred_stage(stage) for stage in current_child_stages]
        + [split_stage_proposal_from_text(proposal) for proposal in prior_discuss_proposals]
    )
    if remaining_proposals:
        if config.materialize_split_issues:
            materialize_split_proposals(
                runner,
                config=config,
                parent_issue=issue_number,
                subject=plan_subject,
                proposals=remaining_proposals,
                issue_comments=issue_context.comments,
            )
            return True
        log(
            config,
            f"Planning issue #{issue_number}: split follow-ups remain unfiled; rerun with "
            "--materialize-split-issues or file them manually.",
        )
        if not has_unfiled_split_warning(
            issue_context.comments, issue_number=issue_number, subject=plan_subject
        ):
            post_unfiled_split_warning(
                runner,
                config=config,
                issue_number=issue_number,
                subject=plan_subject,
                proposals=remaining_proposals,
            )
        return False
    if current_deferred_stages or prior_discuss_proposals:
        return False
    if not _plan_text_suggests_narrowing(current_plan):
        return False
    log(
        config,
        f"Planning issue #{issue_number}: approved plan text suggests scope narrowing "
        "but no `deferred_stages` or discuss split proposals were declared or filed.",
    )
    if has_unfiled_split_warning(issue_context.comments, issue_number=issue_number, subject=plan_subject):
        return False
    post_issue_comment(
        runner,
        config=config,
        issue_number=issue_number,
        body="\n".join(
            [
                "### Possible unfiled scope narrowing",
                "",
                "The approved plan's text appears to narrow scope (mentions a stage, "
                "follow-up issue, or out-of-scope work), but no `deferred_stages` were "
                "structurally declared and no discuss split proposals exist for this issue. "
                "If this plan intentionally defers work, declare it via `deferred_stages` in a "
                "future revision or file the follow-up issue(s) manually. This is a heuristic "
                "warning only; the orchestrator never auto-creates issues from prose.",
                "",
                f"<!-- AGENT_SPLIT_UNFILED_WARNING: issue={issue_number} subject={plan_subject} -->",
                "-- coding-review-agent-loop",
            ]
        ),
    )
    return False


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


def _validate_tests_with_post_pr_context(
    validate_tests: Callable[[], None],
    *,
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
    report_description: str,
) -> None:
    try:
        validate_tests()
    except AgentLoopError as exc:
        try:
            validate_open_pr(runner, config=config, pr_number=pr_number)
        except Exception as pr_exc:
            raise AgentLoopError(
                f"{exc}\n\n"
                f"The coder reported PR #{pr_number}, but the orchestrator could not confirm it is open. "
                f"The handoff/reviewer comments were not posted because the {report_description} was invalid. "
                "Inspect the PR state on GitHub before deciding whether to resume the existing PR or rerun "
                "implementation."
            ) from pr_exc
        raise AgentLoopError(
            f"{exc}\n\n"
            f"PR #{pr_number} was confirmed open, but the handoff/reviewer comments were not posted because "
            f"the {report_description} was invalid. Correct the PR/comment if needed, then continue safely with "
            f"`agent-loop pr {pr_number}` instead of rerunning implementation and creating a duplicate PR."
        ) from exc


def _validate_response_tests_with_post_pr_context(
    text: str,
    *,
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
) -> None:
    _validate_tests_with_post_pr_context(
        lambda: validate_response_tests_within_workdir(
            text, assigned_workdir=active_workdir(config)
        ),
        runner=runner,
        config=config,
        pr_number=pr_number,
        report_description="test report",
    )


def _validate_structured_response_tests_with_post_pr_context(
    tests_run: Sequence[str] | None,
    *,
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
) -> None:
    """Validate structured test commands with the same confirmed-PR diagnostic."""
    _validate_tests_with_post_pr_context(
        lambda: validate_test_commands_within_workdir(
            tests_run,
            assigned_workdir=active_workdir(config),
        ),
        runner=runner,
        config=config,
        pr_number=pr_number,
        report_description="structured test report",
    )


def _round_ledger_may_be_incomplete(
    *,
    current_resume: ResumedReviewRound | None,
    prior_unresolved_items: Sequence[UnresolvedReviewItem],
    comments: Sequence[object],
    flow: str,
    current_subject: str,
) -> bool:
    same_subject_incomplete = (
        current_resume.ledger_may_be_incomplete
        if current_resume is not None
        else False
    )
    if prior_unresolved_items:
        return same_subject_incomplete
    records = _extract_round_metadata_records(comments, flow=flow)
    cross_subject_incomplete = any(
        record.metadata.new_items
        for record in records
        if record.metadata.subject != current_subject
    )
    return same_subject_incomplete or cross_subject_incomplete


def _infer_staged_parent_issue(issue_context: IssueContext) -> int | None:
    """Read only generated child-issue markers for direct staged safety checks."""
    candidates: set[int] = set()
    issue_body = issue_context.body or ""
    bodies = [issue_body]
    bodies.extend(comment.body or "" for comment in issue_context.comments)
    for body in bodies:
        for match in SPLIT_CHILD_MARKER_RE.finditer(body):
            candidates.add(int(match.group("parent")))
    first_line = issue_body.splitlines()[0].strip() if issue_body.splitlines() else ""
    decomposition_match = re.fullmatch(
        r"Child phase issue for parent #(?P<parent>\d+)\b.*",
        first_line,
        re.IGNORECASE,
    )
    if decomposition_match:
        candidates.add(int(decomposition_match.group("parent")))
    if len(candidates) > 1:
        joined = ", ".join(f"#{number}" for number in sorted(candidates))
        raise AgentLoopError(
            f"Issue #{issue_context.number} contains conflicting generated staged-parent "
            f"markers ({joined}); resolve the child issue metadata before running issue mode."
        )
    return next(iter(candidates), None)


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
        updates["codex_reasoning_effort"] = effort

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
) -> int:
    implementation_config, reuse_planning_session = _approved_implementation_config(config)
    coder_name = agent_display_name(implementation_config.coder)
    implementation_session_id = coder_session_id if reuse_planning_session else None
    plan_hash = approved_plan_hash(approved_plan)
    plan_additions = _extract_current_expected_closing_issue_ids(approved_plan)
    implementation_human_requirements_context = render_coder_human_requirements_prompt_context(
        issue_context.human_requirements,
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
        if resolved_pr.source == "legacy-closing-reference":
            pr_url, pr_head_sha = require_pr_metadata_for_handoff(resumed_pr_context.metadata)
            validate_pr_expected_closing_issues(
                runner,
                config=implementation_config,
                pr_number=existing_pr_number,
                expected_issue_ids=closing_contract.issue_ids,
                body=resumed_pr_context.metadata.body,
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
            )
        return run_pr_loop(
            runner,
            pr_number=existing_pr_number,
            config=implementation_config,
            issue_context=issue_context,
            usage_context=usage_context,
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
    managed_ci_creation_intent = preflight_managed_ci_creation(
        runner, config=implementation_config, issue_number=issue_number
    )
    if managed_ci_creation_intent is not None and managed_ci_creation_intent.audit_nonce:
        print(
            "WARNING: --allow-unprotected-managed-ci is active for this invocation. GitHub cannot "
            "prevent a manual merge, other automation, a compromised credential, or an agent-loop "
            "defect from bypassing the voluntary final-ci/exact-head gate."
        )
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
        ),
        session_id=implementation_session_id,
        marker_description="structured issue_implementation result, blocking, or clarification",
        validate=lambda text: _validate_issue_implementation_response(
            text,
            human_requirements=issue_context.human_requirements,
        ),
        usage_context=usage_context,
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
        ),
    )
    coder_output = coder_response.text
    implementation_result = coder_response.marker_value
    if isinstance(implementation_result, _TerminalIssueImplementationConflict):
        validate_test_commands_within_workdir(
            implementation_result.parsed.tests_run,
            assigned_workdir=active_workdir(implementation_config),
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
            validate_test_commands_within_workdir(
                implementation_result.tests_run,
                assigned_workdir=active_workdir(implementation_config),
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
        _validate_structured_response_tests_with_post_pr_context(
            implementation_result.tests_run,
            runner=runner,
            config=implementation_config,
            pr_number=implementation_result.pr_number,
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
    else:
        reject_forged_protocol_markers(initial_pr_context.metadata.body or "")
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
        )
    initial_coder_body = _attach_round_metadata(
        render_public_agent_comment(
            kind="issue_implementation",
            parsed=implementation_result,
            agent=implementation_config.coder,
            config=implementation_config,
            model_used=coder_response.model_used,
        ),
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent=coder_name,
            round_number=1,
            subject=str(initial_pr_context.metadata.head_sha or "unknown"),
            prior_items=(),
            raw_structured_coder_response=coder_output,
            model_used=coder_response.model_used,
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
) -> tuple[CreatedPhaseIssue, ...]:
    plan_hash = approved_plan_hash(approved_plan)
    existing = find_existing_decomposition(
        issue_context.comments,
        parent_issue=issue_number,
        plan_hash=plan_hash,
        mode=mode,
    )
    if existing is not None:
        log(config, f"Plan decomposition already exists for issue #{issue_number} ({mode}); not recreating children")
        return tuple(
            CreatedPhaseIssue(
                phase=RecordedPhase(title=title, automation=automation),
                issue_url=url,
                issue_number=number,
            )
            for (title, url, number), automation in zip(existing.children, existing.automation, strict=False)
        )

    coder_name = agent_display_name(config.coder)
    log(config, f"Planning approved; invoking {coder_name} to decompose issue #{issue_number}")
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
        validate=parse_plan_decomposition,
        usage_context=usage_context,
        operation_description="plan decomposition",
    )
    decomposition = decomposition_response.marker_value
    created = create_decomposition_child_issues(
        runner,
        config=config,
        parent_issue=issue_number,
        approved_plan=approved_plan,
        decomposition=decomposition,
    )
    post_decomposition_parent_summary(
        runner,
        config=config,
        parent_issue=issue_number,
        mode=mode,
        plan_hash=plan_hash,
        created=created,
    )
    return created


@dataclass(frozen=True)
class _ReviewerTurnResult:
    """Outcome of one plan/PR reviewer turn: a validated response or a captured failure.

    Worker threads in the --review-parallel path return these instead of
    raising, so exceptions never cross the thread boundary; the main thread
    delivers completed turns to the main thread in completion order.  The
    caller may publish an individual validated review immediately, but must
    retain configured-order aggregation until every launched turn settles.
    """

    reviewer_name: str
    response: ValidatedAgentResponse | None = None
    error: AgentLoopError | None = None


def _launch_reviewer_turns(
    runner: Runner,
    pending: Sequence[AgentName],
    *,
    thread_name_prefix: str,
    run_turn: Callable[[AgentName], _ReviewerTurnResult],
    on_completion: Callable[[AgentName, _ReviewerTurnResult], None] | None = None,
) -> dict[AgentName, _ReviewerTurnResult]:
    """Run workers concurrently, delivering completion-order results before settlement.

    ``run_turn`` must never let an exception escape; it is responsible for
    capturing any failure into the returned ``_ReviewerTurnResult`` so the
    thread pool never needs to propagate a worker exception. On
    KeyboardInterrupt, active agent processes are killed so worker wait loops
    return promptly before the interrupt is re-raised.
    """
    executor = ThreadPoolExecutor(max_workers=len(pending), thread_name_prefix=thread_name_prefix)
    try:
        futures = {executor.submit(run_turn, reviewer): reviewer for reviewer in pending}
        results: dict[AgentName, _ReviewerTurnResult] = {}
        for future in as_completed(futures):
            reviewer = futures[future]
            result = future.result()
            results[reviewer] = result
            if on_completion is not None:
                try:
                    on_completion(reviewer, result)
                except BaseException:
                    # A publication failure must not leave reviewer subprocesses
                    # running or allow a later reconciliation/coder turn.
                    for other in futures:
                        other.cancel()
                    runner.terminate_active_processes()
                    raise
        return results
    except KeyboardInterrupt:
        runner.terminate_active_processes()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _run_plan_first_loop(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    memory,
    issue_context: IssueContext,
    implement_after_approval: bool,
    usage_context: RunUsageContext,
) -> int:
    if config.review_parallel:
        _ensure_parallel_reviewer_workdirs(config, flag_name="--review-parallel", role_label="reviewer")
    coder_name = agent_display_name(config.coder)
    configured_reviewers = reviewers(config)
    coder_session_id: str | None = None
    reviewer_session_ids: dict[AgentName, str | None] = {}
    unresolved_items: list[UnresolvedReviewItem] = []
    compact_prior_summaries: list[str] = []
    next_unresolved_item_number = 1
    resume_state = _resume_plan_round(issue_context.comments, configured_reviewers=configured_reviewers)
    if resume_state is None:
        log(config, f"Planning issue #{issue_number}: invoking {coder_name} (context mode: full)")
        plan_human_requirements_context = render_coder_human_requirements_prompt_context(
            issue_context.human_requirements,
            requirement_scope="planning requirements",
            full_omission_fallback="Fetch the issue discussion directly before finalizing the plan.",
        )
        plan_response = _run_validated_agent(
            runner,
            agent=config.coder,
            config=config,
            prompt=build_issue_plan_prompt(issue_number, config, memory, issue_context=issue_context),
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking --> or <!-- AGENT_CLARIFY -->",
            validate=lambda text, human_requirements=issue_context.human_requirements: _validate_response_with_human_requirements(
                text,
                marker_validator=_require_plan_state_or_clarification,
                human_requirements=human_requirements,
                requirement_scope="planning requirements",
                full_omission_fallback="Fetch the issue discussion directly before finalizing the plan.",
            ),
            usage_context=usage_context,
            use_repair=True,
            repair_expected_kind="plan_state",
            repair_surfaced_requirement_ids=plan_human_requirements_context.surfaced_requirement_ids,
            repair_requires_direct_discussion_ack=plan_human_requirements_context.requires_direct_discussion_ack,
            operation_description="planning",
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
        structured_plan = validate_structured_plan_state(plan_output)
        if isinstance(structured_plan, StructuredPlanState):
            raw_structured_coder_response = plan_output
            canonical_plan = plan_output
            public_plan_output = render_public_agent_comment(
                kind="plan_state",
                parsed=structured_plan,
                agent=config.coder,
                config=config,
                model_used=plan_response.model_used,
            )
        else:
            public_plan_output = normalize_freeform_signature(
                plan_output, agent=config.coder, config=config, model_used=plan_response.model_used
            )
        post_issue_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=_attach_round_metadata(
                public_plan_output,
                PostedRoundMetadata(
                    flow="plan",
                    role="coder",
                    agent=coder_name,
                    round_number=1,
                    subject=_plan_subject(current_plan),
                    prior_items=(),
                    canonical_plan=canonical_plan,
                    raw_structured_coder_response=raw_structured_coder_response,
                    compact_prior_summaries=tuple(compact_prior_summaries),
                    model_used=plan_response.model_used,
                    acquisition_outcome=plan_response.acquisition_outcome,
                    acquisition_returncode=plan_response.acquisition_returncode,
                ),
            ),
        )
        start_round_number = 1
        resumed_round: ResumedReviewRound | None = None
        current_coder_output = plan_output
    else:
        current_plan, resumed_round = resume_state
        current_coder_output = resumed_round.coder_output
        unresolved_items = list(resumed_round.prior_items)
        compact_prior_summaries = list(resumed_round.compact_prior_summaries)
        next_unresolved_item_number = resumed_round.next_unresolved_item_number
        start_round_number = resumed_round.round_number
        log(config, f"Planning issue #{issue_number}: resuming round {start_round_number}")

    for round_number in range(start_round_number, config.max_rounds + 1):
        current_resume = resumed_round if resumed_round is not None and round_number == resumed_round.round_number else None
        prior_unresolved_items = current_resume.prior_items if current_resume is not None else tuple(unresolved_items)
        prior_dispositions: dict[str, list[ReviewItemDisposition]] = {
            item.item_id: [] for item in prior_unresolved_items
        }
        round_new_unresolved_items: list[UnresolvedReviewItem] = []
        current_plan_subject = _plan_subject(current_plan)
        round_ledger_incomplete = _round_ledger_may_be_incomplete(
            current_resume=current_resume,
            prior_unresolved_items=prior_unresolved_items,
            comments=issue_context.comments,
            flow="plan",
            current_subject=current_plan_subject,
        )
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
        all_approved = True
        resumed_by_name = {
            record.metadata.agent: record for record in (current_resume.completed_reviews if current_resume is not None else ())
        }

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
            )

        plan_fatal_errors: list[tuple[str, AgentLoopError]] = []
        plan_turn_results: dict[AgentName, _ReviewerTurnResult] = {}
        early_published_plan_reviewers: set[AgentName] = {
            reviewer
            for reviewer in configured_reviewers
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
                        compact_prior_summaries=tuple(compact_prior_summaries),
                        model_used=model_used, phase=phase,
                        acquisition_outcome=acquisition_outcome,
                        acquisition_returncode=acquisition_returncode,
                        canonical_reviewer_response=(review_output if phase == "publication" else None),
                    ),
                ),
            )

        if config.review_parallel:
            pending_plan_reviewers = [
                reviewer for reviewer in configured_reviewers
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
                            validate=lambda text, reviewer_name=reviewer_name: _validate_plan_review_response(
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
                                ),
                            ),
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
                            role="reviewer",
                            operation_description="plan review",
                        )
                    except AgentLoopError as exc:
                        # Includes QuotaResetExceededError: captured here and
                        # re-raised on the main thread with priority.
                        return _ReviewerTurnResult(reviewer_name=reviewer_name, error=exc)
                    return _ReviewerTurnResult(reviewer_name=reviewer_name, response=response)

                def _publish_plan_completion(reviewer: AgentName, turn: _ReviewerTurnResult) -> None:
                    """Publish a validated reviewer response without mutating round state.

                    Numbering and ledger mutations stay below the settlement barrier.
                    The raw validated response in metadata makes this checkpoint
                    resumable even though its ``new_items`` are provisional.
                    """
                    if turn.error is not None or turn.response is None:
                        return
                    reviewer_name = agent_display_name(reviewer)
                    parsed = turn.response.marker_value
                    assert isinstance(parsed, ParsedPlanReview)
                    parsed = dataclasses_replace(
                        parsed,
                        items=_drop_repeated_carried_plan_future_followups(
                            parsed.items, prior_items=prior_unresolved_items,
                            dispositions=parsed.dispositions,
                        ),
                    )
                    if _is_incomplete_plan_review(parsed):
                        return
                    _post_plan_reviewer_comment(
                        reviewer_name, parsed, review_output=turn.response.text,
                        model_used=turn.response.model_used,
                        acquisition_outcome=turn.response.acquisition_outcome,
                        acquisition_returncode=turn.response.acquisition_returncode,
                        phase="publication",
                    )
                    early_published_plan_reviewers.add(reviewer)

                plan_turn_results = _launch_reviewer_turns(
                    runner,
                    pending_plan_reviewers,
                    thread_name_prefix=f"plan-review-r{round_number}",
                    run_turn=_plan_reviewer_worker,
                    on_completion=_publish_plan_completion,
                )

        for reviewer in configured_reviewers:
            reviewer_name = agent_display_name(reviewer)
            resumed_record = resumed_by_name.get(reviewer_name)
            if resumed_record is not None:
                review_output = resumed_record.metadata.canonical_reviewer_response or resumed_record.body
                review_model_used = resumed_record.metadata.model_used
                review_acquisition_outcome = resumed_record.metadata.acquisition_outcome
                review_acquisition_returncode = resumed_record.metadata.acquisition_returncode
                structured_review = parse_structured_plan_review(
                    review_output,
                    reviewer=reviewer_name,
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
                )
                review_state = parsed_review.state
                log(config, f"Planning round {round_number}: resuming {reviewer_name}'s completed review")
                reviewer_new_unresolved_items = list(resumed_record.metadata.new_items)
            elif config.review_parallel:
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
                review_response = _run_validated_agent(
                    runner,
                    agent=reviewer,
                    config=config,
                    prompt=_build_plan_review_prompt(reviewer),
                    session_id=reviewer_session_ids.get(reviewer),
                    marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                    validate=lambda text, reviewer_name=reviewer_name, items=prior_unresolved_items: _validate_plan_review_response(
                        text,
                        reviewer=reviewer_name,
                        unresolved_items=items,
                        current_round_items=round_new_unresolved_items,
                        surfaced_requirement_ids=_surfaced_reviewer_requirement_ids(
                            issue_context.human_requirements,
                            requirement_scope="planning requirements",
                        ),
                    ),
                    usage_context=usage_context,
                    use_repair=True,
                    repair_expected_kind="plan_review",
                    repair_reviewer_requirement_ids=_surfaced_reviewer_requirement_ids(
                        issue_context.human_requirements,
                        requirement_scope="planning requirements",
                    ),
                    repair_allowed_prior_item_ids=tuple(item.item_id for item in prior_unresolved_items),
                    ledger_incomplete=round_ledger_incomplete,
                    role="reviewer",
                    operation_description="plan review",
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
                incomplete_review_error = AgentLoopError(
                    f"{reviewer_name} did not complete plan review and reported no actionable "
                    "blocking plan issues or Same-Plan follow-ups. This is a reviewer-internal error; "
                    "agent-loop stopped before a coder follow-up. Rerun or switch the reviewer/model "
                    "after resolving the reviewer environment."
                )
                if config.review_parallel:
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
            if resumed_record is None or resumed_record.metadata.phase == "publication":
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
                        acquisition_outcome=review_acquisition_outcome,
                        acquisition_returncode=review_acquisition_returncode,
                        new_items=tuple(reviewer_new_unresolved_items),
                    )
            else:
                round_new_unresolved_items.extend(reviewer_new_unresolved_items)

        if config.review_parallel and not (current_resume is not None and current_resume.reconciled):
            settled = ", ".join(agent_display_name(reviewer) for reviewer in configured_reviewers)
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
            prior_unresolved_items,
            prior_dispositions,
            same_status="same-plan",
            retain_future=True,
        )
        compact_prior_summaries.extend(
            _collect_prior_compact_summaries(
                prior_unresolved_items,
                unresolved_items,
                prior_dispositions,
            )
        )
        unresolved_items = [*unresolved_items, *round_new_unresolved_items]
        must_fix_items = [item for item in unresolved_items if item.status in {"blocking", "same-plan"}]
        if all_approved and not must_fix_items and issue_context.human_requirements:
            hr_ids = _surfaced_reviewer_requirement_ids(
                issue_context.human_requirements,
                requirement_scope="planning requirements",
            )
            coder_dispositions_complete = _current_plan_has_complete_human_requirement_dispositions(
                current_coder_output,
                surfaced_requirement_ids=hr_ids,
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
                            surfaced_requirement_ids=hr_ids,
                        ),
                        repair_kwargs={
                            "expected_kind": "plan_review",
                            "reviewer_requirement_ids": hr_ids,
                            "allowed_prior_item_ids": tuple(
                                item.item_id for item in prior_unresolved_items
                            ),
                        },
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

        if all_approved and not must_fix_items:
            approved_future_followup_sources = [
                _plan_followup_source_from_unresolved_item(item)
                for item in unresolved_items
                if item.status == "future"
            ]
            plan_hash = approved_plan_hash(current_plan)
            plan_subject = _plan_subject(current_plan)
            mode = config.plan_execution_mode
            if implement_after_approval:
                mode = "implement-one-shot"
            plan_additions = _extract_current_expected_closing_issue_ids(current_plan)
            split_topology = bool(
                mode in {"decompose-only", "implement-by-phase"}
                or config.materialize_split_issues
                or _extract_current_child_stages(current_plan)
                or _extract_current_deferred_stages(current_plan)
            )
            if split_topology and (config.expected_closing_issue_ids is not None or plan_additions is not None):
                raise AgentLoopError(
                    "Additional expected closing issue IDs are single-PR-only and cannot be "
                    "carried through split/decomposition materialization. Invoke the actual "
                    "child issue with a child-scoped --expected-closing-issue declaration."
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
                allow_issue_filing=mode in {"implement-one-shot", "implement-by-phase"},
            )
            split_scope_materialized = _handle_plan_first_split_scope(
                runner,
                issue_number=issue_number,
                config=config,
                current_plan=current_plan,
                plan_subject=plan_subject,
                issue_context=issue_context,
            )
            if split_scope_materialized:
                # Refetch so downstream logic (selected-stage resolution below,
                # decomposition, etc.) sees the `AGENT_DISCUSS_SPLIT` comment
                # this call may have just posted, instead of the stale
                # pre-materialization snapshot (#492 review).
                issue_context = get_issue_context(runner, config=config, issue_number=issue_number)
            if mode == "plan-only":
                print(
                    f"Issue #{issue_number} plan approved by {format_agent_list(configured_reviewers)}."
                )
                return 0

            if mode in {"decompose-only", "implement-by-phase"}:
                created = _decompose_approved_plan(
                    runner,
                    issue_number=issue_number,
                    approved_plan=current_plan,
                    config=config,
                    memory=memory,
                    issue_context=issue_context,
                    mode=mode,
                    coder_session_id=coder_session_id,
                    usage_context=usage_context,
                )
                if mode == "decompose-only":
                    print(f"Issue #{issue_number} approved plan decomposed into child issues.")
                    return 0
                first_agent_phase = next(
                    (item for item in created if item.phase.automation == "agent-pr"),
                    None,
                )
                first_phase = created[0] if created else None
                if first_phase is None:
                    raise AgentLoopError("Plan decomposition produced no phases.")
                if first_phase.phase.automation != "agent-pr":
                    print(
                        f"Issue #{issue_number} approved plan decomposed; first phase requires human work "
                        f"({first_phase.phase.automation}), so implementation is stopping."
                    )
                    return 0
                if first_agent_phase is None or first_agent_phase.issue_number is None:
                    raise AgentLoopError(
                        "Cannot implement first decomposed phase because its child issue number "
                        "was not available from GitHub CLI output."
                    )
                plan_hash = approved_plan_hash(current_plan)
                handoff = find_existing_phase_implementation_handoff(
                    issue_context.comments,
                    parent_issue=issue_number,
                    plan_hash=plan_hash,
                    mode=mode,
                    phase_index=1,
                    child_issue_number=first_agent_phase.issue_number,
                )
                if handoff is not None:
                    print(
                        f"Issue #{issue_number} approved plan already handed off to child issue "
                        f"#{handoff.child_issue_number}; resume directly with "
                        f"`agent-loop issue {handoff.child_issue_number}`."
                    )
                    return 0
                child_issue_context = get_issue_context(
                    runner,
                    config=config,
                    issue_number=first_agent_phase.issue_number,
                )
                phase_parent_context = first_agent_phase.phase.parent_context or current_plan
                implementation_result = _implement_approved_issue(
                    runner,
                    issue_number=first_agent_phase.issue_number,
                    approved_plan=phase_parent_context,
                    config=config,
                    memory=memory,
                    issue_context=child_issue_context,
                    coder_session_id=coder_session_id,
                    usage_context=usage_context,
                )
                post_phase_implementation_handoff_comment(
                    runner,
                    config=config,
                    parent_issue=issue_number,
                    mode=mode,
                    plan_hash=plan_hash,
                    phase_index=1,
                    created=first_agent_phase,
                )
                return implementation_result

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
                        )
                    return run_pr_loop(
                        runner,
                        pr_number=resolved_pr.pr_number,
                        config=config,
                        issue_context=target_issue_context,
                        usage_context=usage_context,
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
                            usage_context=usage_context,
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
                    coder_session_id=coder_session_id,
                    usage_context=usage_context,
                    one_shot_parent_issue=target_issue_number,
                    plan_subject=plan_subject,
                    staged_parent_issue=staged_parent_issue,
                )
            raise AgentLoopError(f"Unknown plan execution mode: {mode}")

        if round_number == config.max_rounds:
            raise AgentLoopError(
                f"One or more reviewers still reported blocking plan issues after "
                f"round {round_number}; human review required."
            )

        combined_review = "\n\n".join(f"{name} plan review:\n\n{review}" for name, review in blocking_reviews)
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
        plan_response = _run_validated_agent(
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
            ),
            session_id=coder_session_id,
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=lambda text, human_requirements=issue_context.human_requirements, items=tuple(must_fix_items): _validate_response_with_human_requirements(
                text,
                marker_validator=lambda revised_text: _validate_plan_revision_response(
                    revised_text,
                    unresolved_items=items,
                ),
                human_requirements=human_requirements,
                requirement_scope="planning requirements",
                full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
            ),
            usage_context=usage_context,
            use_repair=True,
            repair_expected_kind="plan_revision",
            repair_surfaced_requirement_ids=(
                plan_revision_human_requirements_context.surfaced_requirement_ids
            ),
            repair_requires_direct_discussion_ack=(
                plan_revision_human_requirements_context.requires_direct_discussion_ack
            ),
            repair_allowed_prior_item_ids=tuple(item.item_id for item in must_fix_items),
            ledger_incomplete=round_ledger_incomplete,
            operation_description="plan revision",
        )
        canonical_plan: str | None = None
        public_comment = plan_response.text
        raw_structured_coder_response: str | None = None
        if isinstance(plan_response.marker_value, StructuredPlanRevision):
            raw_structured_coder_response = plan_response.text
            canonical_plan = render_canonical_plan_revision(
                plan_response.marker_value, must_fix_items, config
            )
            current_plan = canonical_plan
            current_coder_output = plan_response.text
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
            public_comment = normalize_freeform_signature(
                plan_response.text, agent=config.coder, config=config, model_used=plan_response.model_used
            )
        coder_session_id = plan_response.session_id
        post_issue_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=_attach_round_metadata(
                public_comment,
                PostedRoundMetadata(
                    flow="plan",
                    role="coder",
                    agent=coder_name,
                    round_number=round_number + 1,
                    subject=_plan_subject(current_plan),
                    prior_items=tuple(unresolved_items),
                    canonical_plan=canonical_plan,
                    raw_structured_coder_response=raw_structured_coder_response,
                    compact_prior_summaries=tuple(compact_prior_summaries),
                    model_used=plan_response.model_used,
                    acquisition_outcome=plan_response.acquisition_outcome,
                    acquisition_returncode=plan_response.acquisition_returncode,
                ),
            ),
        )
        resumed_round = None

    raise AgentLoopError(
        f"Reached max planning rounds ({config.max_rounds}) for issue #{issue_number}; "
        "human review required."
    )


def run_issue_loop(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    plan_first: bool = False,
    implement_after_approval: bool = False,
    usage_context: RunUsageContext | None = None,
) -> int:
    owned_usage_context = usage_context is None
    usage_context = usage_context or _new_usage_context(config)
    try:
        config = resolve_base_branch(config, runner)
        ensure_agent_workdirs(config, runner)
        log(config, f"Validating issue #{issue_number}")
        validate_open_issue(runner, config=config, issue_number=issue_number)
        issue_context = get_issue_context(runner, config=config, issue_number=issue_number)
        staged_parent_issue = _infer_staged_parent_issue(issue_context)

        recovered_plan_hash: str | None = None
        recovered_plan_additions: tuple[int, ...] | None = None
        if plan_first:
            # This is a comment-only reconstruction.  It must happen before
            # memory preparation or any agent invocation so an existing
            # approved-plan handoff can be checked without re-planning.
            recovered_plan_state = _resume_plan_round(
                issue_context.comments, configured_reviewers=reviewers(config)
            )
            if recovered_plan_state is not None:
                recovered_plan_hash = approved_plan_hash(recovered_plan_state[0])
                recovered_plan_additions = _extract_current_expected_closing_issue_ids(
                    recovered_plan_state[0]
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
            if plan_first and resolved_pr.source == "canonical" and resolved_metadata is not None:
                if (
                    resolved_metadata.flow == "approved-plan-implementation"
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
            if resolved_pr.source == "legacy-closing-reference":
                pr_context = get_pr_review_context(runner, config=config, pr_number=resolved_pr.pr_number)
                validate_pr_expected_closing_issues(
                    runner,
                    config=config,
                    pr_number=resolved_pr.pr_number,
                    expected_issue_ids=closing_contract.issue_ids,
                    body=pr_context.metadata.body,
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
                )
            return run_pr_loop(
                runner,
                pr_number=resolved_pr.pr_number,
                config=config,
                issue_context=issue_context,
                usage_context=usage_context,
            )

        memory = prepare_agent_memory(runner, config)
        if plan_first:
            return _run_plan_first_loop(
                runner,
                issue_number=issue_number,
                config=config,
                memory=memory,
                issue_context=issue_context,
                implement_after_approval=implement_after_approval,
                usage_context=usage_context,
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

        implementation_human_requirements_context = render_coder_human_requirements_prompt_context(
            issue_context.human_requirements,
        )
        sync_coder_base_before_implementation(config, runner)
        managed_ci_creation_intent = None
        if config.managed_ci:
            # Direct issue mode is intentionally the only new creation path.
            # A plain `issue --auto-merge` invocation keeps its historical
            # ordinary opening behavior unless it uses plan-first.
            managed_ci_creation_intent = preflight_managed_ci_creation(
                runner, config=config, issue_number=issue_number
            )
            if managed_ci_creation_intent is not None and managed_ci_creation_intent.audit_nonce:
                print(
                    "WARNING: --allow-unprotected-managed-ci is active for this invocation. GitHub cannot "
                    "prevent a manual merge, other automation, a compromised credential, or an agent-loop "
                    "defect from bypassing the voluntary final-ci/exact-head gate."
                )
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
            ),
            marker_description="structured issue_implementation result, blocking, or clarification",
            validate=lambda text: _validate_issue_implementation_response(
                text,
                human_requirements=issue_context.human_requirements,
            ),
            usage_context=usage_context,
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
            ),
        )
        coder_output = coder_response.text
        coder_session_id = coder_response.session_id
        implementation_result = coder_response.marker_value
        if isinstance(implementation_result, _TerminalIssueImplementationConflict):
            validate_test_commands_within_workdir(
                implementation_result.parsed.tests_run,
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
                validate_test_commands_within_workdir(
                    implementation_result.tests_run,
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
            _validate_structured_response_tests_with_post_pr_context(
                implementation_result.tests_run,
                runner=runner,
                config=config,
                pr_number=implementation_result.pr_number,
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
        else:
            reject_forged_protocol_markers(initial_pr_context.metadata.body or "")
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
        initial_coder_body = _attach_round_metadata(
            render_public_agent_comment(
                kind="issue_implementation",
                parsed=implementation_result,
                agent=config.coder,
                config=config,
                model_used=coder_response.model_used,
            ),
            PostedRoundMetadata(
                flow="pr",
                role="coder",
                agent=agent_display_name(config.coder),
                round_number=1,
                subject=str(initial_pr_metadata.head_sha or "unknown"),
                prior_items=(),
                raw_structured_coder_response=coder_output,
                model_used=coder_response.model_used,
                acquisition_outcome=coder_response.acquisition_outcome,
                acquisition_returncode=coder_response.acquisition_returncode,
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
    try:
        if not task_text.strip():
            raise AgentLoopError("Task text is empty; provide a non-empty description.")
        if max_clarification_rounds < 0:
            raise AgentLoopError("--max-clarification-rounds must be zero or positive.")
        config = resolve_base_branch(config, runner)
        ensure_agent_workdirs(config, runner)
        memory = prepare_agent_memory(runner, config)

        history: list[tuple[str, str]] = []
        prompt = build_task_prompt(task_text, config, memory)
        read_clarification = clarification_input or _read_clarification_from_stdin
        coder_name = agent_display_name(config.coder)
        session_id: str | None = None

        for attempt in range(max_clarification_rounds + 1):
            if attempt == 0:
                sync_coder_base_before_implementation(config, runner)
            assigned_head_before = _read_assigned_workdir_head(runner, config)
            log(config, f"Task attempt {attempt + 1}: invoking {coder_name}")
            coder_response = _run_validated_agent(
                runner,
                agent=config.coder,
                config=config,
                prompt=prompt,
                session_id=session_id,
                marker_description="<!-- AGENT_PR: <number> -->, PR URL, blocking, or <!-- AGENT_CLARIFY -->",
                validate=_require_task_implementation_result,
                usage_context=usage_context,
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

            if isinstance(coder_response.marker_value, _TerminalNoPrImplementation):
                raise AgentLoopError(
                    "Coder did not create a valid PR; task implementation is "
                    f"{coder_response.marker_value.state}.\n\n{coder_output}"
                )

            if isinstance(coder_response.marker_value, int):
                pr_number = coder_response.marker_value
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
                            acquisition_outcome=coder_response.acquisition_outcome,
                            acquisition_returncode=coder_response.acquisition_returncode,
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
        if owned_usage_context:
            _persist_usage_summary(config, usage_context)


def _extract_structured_coder_summary(text: str | None) -> str | None:
    if not text:
        return None
    try:
        try:
            implementation = validate_structured_issue_implementation(text)
        except IssueImplementationConflictError as exc:
            implementation = exc.payload
        except AgentLoopError:
            implementation = None
        if isinstance(implementation, StructuredIssueImplementation):
            return implementation.summary
        parsed = validate_structured_coder_followup(text)
        return parsed.summary if parsed else None
    except AgentLoopError:
        return None


def _extract_structured_coder_tests_run(text: str | None) -> tuple[str, ...] | None:
    if not text:
        return None
    try:
        try:
            implementation = validate_structured_issue_implementation(text)
        except IssueImplementationConflictError as exc:
            implementation = exc.payload
        except AgentLoopError:
            implementation = None
        if isinstance(implementation, StructuredIssueImplementation):
            return implementation.tests_run
        parsed = validate_structured_coder_followup(text)
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
        raise AgentLoopError(
            f"PR #{pr_number} ordinary recovery did not qualify the exact head "
            f"({outcome.status}); the draft was left unmerged."
        )
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


def run_pr_loop(
    runner: Runner,
    *,
    pr_number: int,
    config: AgentLoopConfig,
    coder_session_id: str | None = None,
    reviewer_session_id: str | None = None,
    issue_context: IssueContext | None = None,
    workdirs_ready: bool = False,
    usage_context: RunUsageContext | None = None,
    pre_review_test_pending: bool = False,
    managed_pr_origin: tuple[str, str, str, str | None] | None = None,
    managed_ci_handoff: AuthenticatedIssueCreatedHandoff | None = None,
) -> int:
    owned_usage_context = usage_context is None
    usage_context = usage_context or _new_usage_context(config)
    managed_ci = None
    ordinary_recovery: OrdinaryRecoveryCapability | None = None
    ordinary_recovery_selected = False
    managed_ci_qualified = False
    managed_pr_recovered = False
    try:
        bootstrap_cwd = github_bootstrap_cwd(config)
        initial_pr_context = get_pr_review_context(
            runner,
            config=config,
            pr_number=pr_number,
            cwd=bootstrap_cwd,
        )
        if managed_ci_handoff is not None:
            managed_ci_handoff = revalidate_issue_created_handoff(
                runner,
                config=config,
                handoff=managed_ci_handoff,
                metadata=initial_pr_context.metadata,
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
                )
                if managed_ci_handoff is None:
                    reject_forged_protocol_markers(initial_pr_context.metadata.body or "")
                else:
                    managed_ci_handoff = revalidate_issue_created_handoff(
                        runner,
                        config=config,
                        handoff=managed_ci_handoff,
                        metadata=initial_pr_context.metadata,
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
        recorded_pr_contract = find_latest_pr_contract(
            initial_pr_context.comments,
            repository=config.repo,
            pr_number=pr_number,
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
        if issue_context is not None and recorded_pr_contract is None and config.expected_closing_contract_resolved:
            # A pre-contract canonical handoff is a legacy recovery record. Do
            # not retroactively turn its old Refs-only body into a new durable
            # contract; only newly persisted PR-side records activate the gate.
            log(
                config,
                f"PR #{pr_number}: no PR-side expected-closing record found; retaining legacy "
                "handoff recovery without inferring a contract from prose",
            )
            closing_contract = None
        contract_needs_persisting = (
            closing_contract is not None
            and (recorded_pr_contract is None or recorded_pr_contract != closing_contract)
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
            linked_issue_numbers = parse_linked_issue_numbers(
                initial_pr_context.metadata.body,
                repo=config.repo,
            )
            if len(linked_issue_numbers) == 1:
                linked_issue_number = linked_issue_numbers[0]
                log(
                    config,
                    f"PR #{pr_number} references issue #{linked_issue_number}; "
                    "including linked issue context in review prompts",
                )
                issue_context = get_issue_context(
                    runner, config=config, issue_number=linked_issue_number
                )
            elif linked_issue_numbers:
                candidates = ", ".join(f"#{number}" for number in linked_issue_numbers)
                log(
                    config,
                    f"PR #{pr_number} references multiple issues ({candidates}); "
                    "linked issue context is ambiguous and will not be included",
                )
            else:
                log(config, f"PR #{pr_number} has no linked issue context to include in review prompts")
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
        config = resolve_base_branch(
            config,
            runner,
            pr_metadata=initial_pr_context.metadata,
            cwd=bootstrap_cwd,
        )
        if not workdirs_ready:
            ensure_agent_workdirs(config, runner)
        if config.review_parallel:
            _ensure_parallel_reviewer_workdirs(config, flag_name="--review-parallel", role_label="reviewer")
        log(config, f"Validating PR #{pr_number}")
        validate_open_pr(runner, config=config, pr_number=pr_number)
        if closing_contract is not None:
            validate_pr_expected_closing_issues(
                runner,
                config=config,
                pr_number=pr_number,
                expected_issue_ids=closing_contract.expected_closing_issue_ids,
                body=initial_pr_context.metadata.body,
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
                )
        activation = activate_managed_ci(
            runner,
            config=config,
            pr_number=pr_number,
            metadata=initial_pr_context.metadata,
        )
        if activation is not None and activation.activation_path == "ordinary_fallback":
            ordinary_recovery_selected = True
            ordinary_recovery = activation.ordinary_recovery
            managed_ci = None
            log(
                config,
                f"PR #{pr_number}: ordinary recovery selected; "
                "the previous managed activation is not being resumed",
            )
            if ordinary_recovery is None:
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

        def managed_ci_active(metadata: PullRequestMetadata) -> bool:
            """Drop adopted filtering immediately when its live handshake changes."""
            nonlocal managed_ci
            if managed_ci is None:
                return False
            if revalidate_adopted_managed_ci(
                runner, config=config, pr_number=pr_number, metadata=metadata, contract=managed_ci
            ):
                if managed_ci.issue_created_pr:
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
        unresolved_items: list[UnresolvedReviewItem] = []
        pr_compact_prior_summaries: list[str] = []
        latest_coder_output: str | None = None
        next_unresolved_item_number = 1
        start_round_number = 1
        resumed_round: ResumedReviewRound | None = None
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
        )
        if resumed_round is not None:
            unresolved_items = list(resumed_round.prior_items)
            pr_compact_prior_summaries = list(resumed_round.compact_prior_summaries)
            latest_coder_output = resumed_round.coder_output
            next_unresolved_item_number = resumed_round.next_unresolved_item_number
            start_round_number = resumed_round.round_number
            log(config, f"PR #{pr_number}: resuming round {start_round_number}")
        watch_failure_extension_used = False
        watch_head_extension_used = False
        watch_deadline: float | None = None
        watch_attempts_remaining: int | None = None
        # One extra slot is only activated by a watcher-discovered failure on
        # the configured final round; ordinary review failures retain the cap.
        allowed_rounds = config.max_rounds
        # The two independent one-shot watcher allowances below can extend the
        # effective ceiling by two rounds in one invocation. Keep the static
        # range large enough to reach both; ``allowed_rounds`` remains the
        # authoritative guard and prevents either slot from being used unless
        # its corresponding watcher transition grants it.
        for round_number in range(start_round_number, config.max_rounds + 3):
            if round_number > allowed_rounds:
                raise AgentLoopError(
                    f"One or more reviewers still reported blocking issues after round {allowed_rounds}; human review required."
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
            pr_comments = pr_context.comments
            if closing_contract is not None:
                validate_pr_expected_closing_issues(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    expected_issue_ids=closing_contract.expected_closing_issue_ids,
                    body=pr_metadata.body,
                )
            if managed_ci_active(pr_metadata) and pre_review_tests_passed and pr_metadata.head_sha:
                publish_round_readiness(
                    runner,
                    config=config,
                    head_sha=pr_metadata.head_sha,
                )
            human_requirements = _merge_human_requirements(issue_context, pr_context)
            current_resume = resumed_round if resumed_round is not None and round_number == resumed_round.round_number else None
            unresolved_items = _reconcile_human_requirements_ack_item(
                current_resume.prior_items if current_resume is not None else unresolved_items,
                coder_output=latest_coder_output,
                human_requirements=human_requirements,
                source_round=round_number,
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
            prior_unresolved_items = tuple(unresolved_items)
            prior_dispositions: dict[str, list[ReviewItemDisposition]] = {
                item.item_id: [] for item in prior_unresolved_items
            }
            round_new_unresolved_items: list[UnresolvedReviewItem] = []
            current_pr_subject = str(pr_metadata.head_sha or "unknown")
            round_ledger_incomplete = _round_ledger_may_be_incomplete(
                current_resume=current_resume,
                prior_unresolved_items=prior_unresolved_items,
                comments=pr_comments,
                flow="pr",
                current_subject=current_pr_subject,
            )
            use_compact_pr_context = (
                config.pr_review_context_mode == "compact"
                and round_number >= 2
                and not round_ledger_incomplete
            )
            compact_coder_summary = (
                _extract_structured_coder_summary(latest_coder_output)
                if use_compact_pr_context
                else None
            )
            compact_coder_tests_run = (
                _extract_structured_coder_tests_run(latest_coder_output)
                if use_compact_pr_context
                else None
            )
            surfaced_reviewer_requirement_ids = _surfaced_reviewer_requirement_ids(
                human_requirements,
                requirement_scope="PR requirements",
            )
            approved_review_outputs: list[tuple[str, str]] = []
            resumed_by_name = {
                record.metadata.agent: record for record in (current_resume.completed_reviews if current_resume is not None else ())
            }
            unchanged_head_approvals = _latest_pr_approved_reviews_for_head(
                pr_comments,
                head_sha=pr_metadata.head_sha,
                configured_reviewers=configured_reviewers,
            )
            skip_reviewers_for_recovery = bool(
                current_resume is not None and current_resume.unrecorded_head_advance
            )
            if skip_reviewers_for_recovery:
                log(
                    config,
                    f"Round {round_number}: PR head advanced to {current_pr_subject} "
                    "without current-head coder metadata; routing recovered prior items "
                    f"through {coder_name} before review",
                )
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
                acquisition_outcome: str = "success",
                acquisition_returncode: int | None = None,
                new_items: tuple[UnresolvedReviewItem, ...] = (),
                phase: str = "authoritative",
            ) -> None:
                """Post one PR review using the same rendering and durable record."""
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
                            acquisition_outcome=acquisition_outcome,
                            acquisition_returncode=acquisition_returncode,
                            surfaced_reviewer_requirement_ids=surfaced_reviewer_requirement_ids,
                            phase=phase,
                            canonical_reviewer_response=(review_output if phase == "publication" else None),
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
                prior_approval = unchanged_head_approvals.get(reviewer_name)
                if (
                    prior_approval is not None
                    and prior_approval.metadata.round_number < round_number
                    and (
                        not human_requirements
                        or (
                            human_requirements_resolved(prior_approval.body)
                            and set(surfaced_reviewer_requirement_ids).issubset(
                                prior_approval.metadata.surfaced_reviewer_requirement_ids
                            )
                        )
                    )
                ):
                    return "carried"
                return "turn"

            if config.review_parallel and not skip_reviewers_this_round:
                pending_pr_reviewers = [
                    reviewer for reviewer in configured_reviewers
                    if _pr_reviewer_prelaunch_kind(reviewer) == "turn"
                ]
                if pending_pr_reviewers:
                    launchable_pr_reviewers: list[AgentName] = []
                    for reviewer in pending_pr_reviewers:
                        reviewer_name = agent_display_name(reviewer)
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
                                compact_context=use_compact_pr_context,
                                compact_prior=(
                                    CompactPriorContext(tuple(pr_compact_prior_summaries))
                                    if use_compact_pr_context
                                    else None
                                ),
                                compact_tail=pr_parallel_compact_tail,
                                compact_coder_summary=compact_coder_summary,
                                compact_coder_tests_run=compact_coder_tests_run,
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
                                        None if use_compact_pr_context else reviewer_session_ids.get(reviewer)
                                    ),
                                    marker_description="<!-- AGENT_STATE: approved|blocking -->",
                                    validate=lambda text, reviewer_name=reviewer_name: _validate_review_response(
                                        text,
                                        reviewer=reviewer_name,
                                        unresolved_items=prior_unresolved_items,
                                        # Never share the mutable round_new_unresolved_items
                                        # list with concurrent workers (#594): it only
                                        # enriches the UnknownPriorItemDispositionError
                                        # message, so an empty tuple changes no outcome.
                                        current_round_items=(),
                                    ),
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
                                    role="reviewer",
                                    operation_description="PR review",
                                )
                            except AgentLoopError as exc:
                                # Includes QuotaResetExceededError: captured here
                                # and re-raised on the main thread with priority.
                                return _ReviewerTurnResult(reviewer_name=reviewer_name, error=exc)
                            return _ReviewerTurnResult(reviewer_name=reviewer_name, response=response)

                        def _publish_pr_completion(reviewer: AgentName, turn: _ReviewerTurnResult) -> None:
                            """Publish a completion-order PR review; settlement remains below."""
                            if turn.error is not None or turn.response is None:
                                return
                            reviewer_name = agent_display_name(reviewer)
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
                            if _is_incomplete_pr_review(parsed):
                                return
                            _post_pr_reviewer_comment(
                                reviewer_name, parsed, review_output=turn.response.text,
                                model_used=turn.response.model_used,
                                acquisition_outcome=turn.response.acquisition_outcome,
                                acquisition_returncode=turn.response.acquisition_returncode,
                                phase="publication",
                            )
                            early_published_pr_reviewers.add(reviewer)

                        pr_turn_results = _launch_reviewer_turns(
                            runner,
                            launchable_pr_reviewers,
                            thread_name_prefix=f"pr-review-r{round_number}",
                            run_turn=_pr_reviewer_worker,
                            on_completion=_publish_pr_completion,
                        )
                    else:
                        log(
                            config,
                            f"Round {round_number}: every pending reviewer failed pre-review "
                            "PR sync; nothing to launch in parallel this round",
                        )

            for reviewer in (() if skip_reviewers_this_round else configured_reviewers):
                reviewer_name = agent_display_name(reviewer)
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
                    review_output = resumed_record.metadata.canonical_reviewer_response or resumed_record.body
                    review_model_used = resumed_record.metadata.model_used
                    review_acquisition_outcome = resumed_record.metadata.acquisition_outcome
                    review_acquisition_returncode = resumed_record.metadata.acquisition_returncode
                    reparsed_review = parse_review(review_output, reviewer=reviewer_name)
                    parsed_review = ParsedReview(
                        state=resumed_record.metadata.state or parse_agent_state(review_output),
                        summary=review_freeform_summary_text(review_output),
                        blocking_items=reparsed_review.blocking_items,
                        followups=reparsed_review.followups,
                        dispositions=resumed_record.metadata.dispositions,
                    )
                    review_state = parsed_review.state
                    reviewer_new_unresolved_items = list(resumed_record.metadata.new_items)
                    log(config, f"Round {round_number}: resuming {reviewer_name}'s completed review")
                elif (
                    (prior_approval := unchanged_head_approvals.get(reviewer_name)) is not None
                    and prior_approval.metadata.round_number < round_number
                    and (
                        not human_requirements
                        or (
                            human_requirements_resolved(prior_approval.body)
                            and set(surfaced_reviewer_requirement_ids).issubset(
                                prior_approval.metadata.surfaced_reviewer_requirement_ids
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
                elif config.review_parallel:
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
                            f"({category}); continuing with the remaining reviewers",
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
                    context_mode = "compact" if use_compact_pr_context else "full"
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
                    review_response, review_failure = _capture_agent_invocation(
                        lambda: _run_validated_agent(
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
                                compact_context=use_compact_pr_context,
                                compact_prior=(
                                    CompactPriorContext(tuple(pr_compact_prior_summaries))
                                    if use_compact_pr_context
                                    else None
                                ),
                                compact_tail=compact_tail,
                                compact_coder_summary=compact_coder_summary,
                                compact_coder_tests_run=compact_coder_tests_run,
                            ),
                            session_id=(
                                None
                                if use_compact_pr_context
                                else reviewer_session_ids.get(reviewer)
                            ),
                            marker_description="<!-- AGENT_STATE: approved|blocking -->",
                            validate=lambda text, reviewer_name=reviewer_name, items=prior_unresolved_items: _validate_review_response(
                                text,
                                reviewer=reviewer_name,
                                unresolved_items=items,
                                current_round_items=round_new_unresolved_items,
                            ),
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
                            role="reviewer",
                            operation_description="PR review",
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
                            f"({category}); continuing with the remaining reviewers",
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

                if _is_incomplete_pr_review(parsed_review):
                    if len(configured_reviewers) == 1:
                        incomplete_pr_review_error = AgentLoopError(
                            f"{reviewer_name} did not complete PR review and reported no actionable "
                            "blocking items or Same-PR follow-ups. This is a reviewer-internal error; "
                            "agent-loop stopped before a coder follow-up. Rerun or switch the reviewer/model "
                            "after resolving the reviewer environment."
                        )
                        if config.review_parallel:
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
                        "continuing with the remaining reviewers",
                    )
                    continue

                for disposition in parsed_review.dispositions:
                    _record_prior_item_disposition(
                        prior_dispositions,
                        disposition,
                        flow="pr",
                        round_number=round_number,
                        subject=current_pr_subject,
                        reviewer_name=reviewer_name,
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
                            if config.approved_followups.startswith("fix-and-"):
                                for followup in parsed_review.followups.same_pr:
                                    tracked_item = _next_unresolved_item(
                                        item_number=next_unresolved_item_number,
                                        reviewer=followup.reviewer,
                                        source_round=round_number,
                                        text=followup.text,
                                        status="same-pr",
                                    )
                                    round_new_unresolved_items.append(tracked_item)
                                    reviewer_new_unresolved_items.append(tracked_item)
                                    next_unresolved_item_number += 1
                            else:
                                tracked_item = _next_unresolved_item(
                                    item_number=next_unresolved_item_number,
                                    reviewer=reviewer_name,
                                    source_round=round_number,
                                    text="\n".join(
                                        [
                                            "Blocking review included Same-PR follow-ups, "
                                            f"but --approved-followups={config.approved_followups} "
                                            "does not enable a same-PR fix path.",
                                            "",
                                            _format_same_pr_followups(parsed_review.followups.same_pr),
                                        ]
                                    ),
                                    status="blocking",
                                )
                                round_new_unresolved_items.append(tracked_item)
                                reviewer_new_unresolved_items.append(tracked_item)
                                next_unresolved_item_number += 1
                        if reviewer not in early_published_pr_reviewers:
                            _post_pr_reviewer_comment(
                                reviewer_name, parsed_review, review_output=review_output,
                                model_used=review_model_used,
                                acquisition_outcome=review_acquisition_outcome,
                                acquisition_returncode=review_acquisition_returncode,
                                new_items=tuple(reviewer_new_unresolved_items),
                            )
                    else:
                        if resumed_record is not None:
                            round_new_unresolved_items.extend(reviewer_new_unresolved_items)
                    continue

                approved_review_outputs.append((reviewer_name, review_output))
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
                            acquisition_outcome=review_acquisition_outcome,
                            acquisition_returncode=review_acquisition_returncode,
                            new_items=tuple(reviewer_new_unresolved_items),
                        )
                else:
                    if resumed_record is not None:
                        round_new_unresolved_items.extend(reviewer_new_unresolved_items)

            if (
                config.review_parallel
                and not skip_reviewers_this_round
                and not (current_resume is not None and current_resume.reconciled)
            ):
                settled = ", ".join(agent_display_name(reviewer) for reviewer in configured_reviewers)
                post_pr_comment(
                    runner, config=config, pr_number=pr_number,
                    body=_attach_round_metadata(
                        f"PR review round {round_number} reconciliation: settled reviewers: {settled or 'none'}. "
                        f"Finalization {'stops' if pr_fatal_errors else 'continues'} after reconciliation.",
                        PostedRoundMetadata(
                            flow="pr", role="summary", agent="Orchestrator", round_number=round_number,
                            subject=current_pr_subject, prior_items=prior_unresolved_items,
                            dispositions=tuple(
                                disposition for values in prior_dispositions.values() for disposition in values
                            ), new_items=tuple(round_new_unresolved_items), phase="reconciliation",
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

            if use_compact_pr_context:
                unresolved_items, future_from_prior_items = _apply_unresolved_item_dispositions(
                    prior_unresolved_items,
                    prior_dispositions,
                    retain_future=False,
                )
                pr_compact_prior_summaries.extend(
                    _collect_prior_compact_summaries(
                        prior_unresolved_items,
                        unresolved_items,
                        prior_dispositions,
                    )
                )
            else:
                unresolved_items, _future_items = _apply_unresolved_item_dispositions(
                    prior_unresolved_items,
                    prior_dispositions,
                )
                future_from_prior_items = []
            unresolved_items = [*unresolved_items, *round_new_unresolved_items]
            future_followups = [
                _approved_followup_from_unresolved_item(item)
                for item in [*unresolved_items, *future_from_prior_items]
                if item.status == "future"
            ]
            must_fix_items = [item for item in unresolved_items if item.status in {"blocking", "same-pr"}]

            if must_fix_items:
                prior_item_ids = {item.item_id for item in prior_unresolved_items}
                disputed_still_blocking = [
                    item for item in must_fix_items
                    if item.item_id in prior_item_ids
                    and _is_disputed_item(item)
                ]
                if disputed_still_blocking:
                    item_summaries = "\n".join(
                        "\n".join(
                            [
                                f"- [{item.item_id}] from {item.reviewer}: {item.text[:300]}",
                                *[f"  Update/evidence: {note}" for note in item.notes],
                            ]
                        )
                        for item in disputed_still_blocking
                    )
                    raise AgentLoopError(
                        f"Reviewer did not resolve {len(disputed_still_blocking)} disputed item(s) "
                        "after seeing coder counter-evidence. Human review required to resolve the "
                        f"disagreement.\n\nDisputed items still unresolved:\n{item_summaries}"
                    )

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
                                    current_round_items=round_new_unresolved_items,
                                ),
                                repair_kwargs={
                                    "expected_kind": "pr_review",
                                    "reviewer_requirement_ids": hr_ids,
                                    "allowed_prior_item_ids": tuple(
                                        item.item_id for item in prior_unresolved_items
                                    ),
                                },
                            )
                            _log_repair_attempts(
                                config, f"Round {round_number}: {reviewer_name}", repair_attempts
                            )
                            if repaired_validated is not None:
                                repaired_parsed = repaired_validated
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
                if not must_fix_items and unavailable_reviewer_failures:
                    approved_reviewer_names = [
                        reviewer_name for reviewer_name, _review_output in approved_review_outputs
                    ]
                    post_pr_comment(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        body=_format_incomplete_pr_review_comment(
                            pr_number=pr_number,
                            unavailable_reviewers=unavailable_reviewer_failures,
                            approved_reviewer_names=approved_reviewer_names,
                        ),
                    )
                    missing = ", ".join(
                        agent_display_name(reviewer)
                        for reviewer in unavailable_reviewer_failures
                    )
                    healthy = ", ".join(approved_reviewer_names) or "(none)"
                    raise AgentLoopError(
                        f"PR #{pr_number} review incomplete: missing required input from {missing}. "
                        f"Healthy reviewers approved: {healthy}. No coder follow-up or merge was attempted."
                    )

                sync_coder_pr_before_validation(config, runner, pr_number, pr_metadata)
                migration_validation = validate_pr_migration_topology(
                    runner,
                    config=config,
                    checkout=active_workdir(config),
                    pr_metadata=pr_metadata,
                )
                if not migration_validation.ok:
                    log(config, f"Round {round_number}: Alembic migration validation blocked approval")
                    unresolved_items.append(
                        _next_unresolved_item(
                            item_number=next_unresolved_item_number,
                            reviewer="Alembic migration validation",
                            source_round=round_number,
                            text=migration_validation.message or "Migration validation failed.",
                            status="blocking",
                        )
                    )
                    next_unresolved_item_number += 1
                    must_fix_items = [item for item in unresolved_items if item.status in {"blocking", "same-pr"}]

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
                must_fix_items = [item for item in unresolved_items if item.status in {"blocking", "same-pr"}]
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
                        body=_ci_infrastructure_stop_message(pr_number, stall, []),
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
                    merged = _finalize_ordinary_recovery_merge(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        capability=ordinary_recovery,
                    )
                    if merged:
                        print(f"PR #{pr_number} merged after deliberate ordinary recovery.")
                    return 0
                if not must_fix_items and config.watch_pending_ci and not managed_ci_active(pr_metadata):
                    if watch_deadline is None:
                        watch_deadline = time.monotonic() + config.ci_timeout_seconds
                        watch_attempts_remaining = max(
                            1,
                            config.ci_timeout_seconds // config.ci_poll_interval_seconds,
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
                    assert watch_attempts_remaining is not None
                    if watch_attempts_remaining <= 0:
                        watch_outcome = CiWatchOutcome(status="timeout")
                    else:
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
                        )
                        run_optional_tests(runner, config)
                        if config.auto_merge:
                            _merge_with_exact_head_proof(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                proof=ExactHeadCiProof(
                                    head_sha=pr_metadata.head_sha or "",
                                    source="full-board",
                                ),
                            )
                            print(f"PR #{pr_number} merged after CI watch completed.")
                        else:
                            print(f"PR #{pr_number} is merge-ready after CI watch completed.")
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
                        return 0
                    # Compatibility fail-safe for an older watcher/provider
                    # that still emits the retired outcome. It is never a
                    # merge permit; unreadable managed suppression remains a
                    # hard error rather than silently becoming ordinary CI.
                    if watch_outcome.status == "no_checks":
                        label_live = managed_label_present(
                            runner, config=config, pr_number=pr_number
                        )
                        if label_live is not False:
                            raise AgentLoopError(
                                f"PR #{pr_number} has no ordinary CI checks while `{MANAGED_LABEL}` "
                                "is present or unreadable; remove the label and wait for real CI before merging."
                            )
                        command = render_managed_ci_resume_command(
                            config, pr_number=pr_number, managed_ci=False,
                        )
                        print(
                            f"PR #{pr_number} has an empty CI board. No merge was attempted; "
                            f"resume with `{command}`."
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
                            body=_ci_infrastructure_stop_message(
                                pr_number, watch_outcome.stall, []
                            ),
                        )
                        print(f"PR #{pr_number} CI watch stopped: external CI infrastructure is stalled.")
                        return 0
                    if watch_outcome.status == "timeout":
                        _publish_approved_followups(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            head_sha=pr_metadata.head_sha,
                            pr_comments=pr_comments,
                            followups=future_followups,
                        )
                        details = (
                            _pr_check_details(watch_outcome.pr_checks)
                            if watch_outcome.pr_checks
                            else ["No reliable check snapshot was available."]
                        )
                        post_pr_comment(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            body=_pending_ci_stop_message(pr_number, "pending", details),
                        )
                        rerun = (
                            shlex.join(config.invocation_argv)
                            if config.invocation_argv
                            else f"agent-loop pr {pr_number} --watch-pending-ci"
                        )
                        note = (
                            ""
                            if config.invocation_argv
                            else " (deterministic fallback; original invocation unavailable)"
                        )
                        print(
                            f"PR #{pr_number} CI watch timed out: {'; '.join(details)}. "
                            f"Rerun: {rerun}{note}"
                        )
                        return 0
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
                        unresolved_items.append(
                            _next_unresolved_item(
                                item_number=next_unresolved_item_number,
                                reviewer="GitHub PR checks",
                                source_round=round_number,
                                text=_pr_check_blocking_review(
                                    pr_number, "failing", details
                                ),
                                status="blocking",
                            )
                        )
                        next_unresolved_item_number += 1
                        if round_number == allowed_rounds and not watch_failure_extension_used:
                            allowed_rounds += 1
                            watch_failure_extension_used = True
                    must_fix_items = [item for item in unresolved_items if item.status in {"blocking", "same-pr"}]
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
                                body=_pending_ci_stop_message(pr_number, pr_checks.state, details),
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
                        unresolved_items.append(
                            _next_unresolved_item(
                                item_number=next_unresolved_item_number,
                                reviewer="GitHub PR checks",
                                source_round=round_number,
                                text=_pr_check_blocking_review(pr_number, pr_checks.state, details),
                                status="blocking",
                            )
                        )
                        next_unresolved_item_number += 1
                        must_fix_items = [
                            item for item in unresolved_items if item.status in {"blocking", "same-pr"}
                        ]
                if not must_fix_items:
                    _publish_approved_followups(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        head_sha=pr_metadata.head_sha,
                        pr_comments=pr_comments,
                        followups=future_followups,
                    )
                    run_optional_tests(runner, config)
                    if config.auto_merge or managed_ci_active(pr_metadata):
                        if managed_ci_active(pr_metadata):
                            assert pr_metadata.head_sha is not None
                            assert pr_metadata.head_branch is not None
                            dispatch_final_qualification(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                expected_head_sha=pr_metadata.head_sha,
                                head_ref=pr_metadata.head_branch,
                                contract=managed_ci,
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
                                merged = _finalize_ordinary_recovery_merge(
                                    runner,
                                    config=config,
                                    pr_number=pr_number,
                                    capability=ordinary_recovery,
                                )
                                if merged:
                                    print(f"PR #{pr_number} merged after deliberate ordinary recovery.")
                                return 0
                            managed_outcome = wait_for_final_qualification(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                metadata=pr_metadata,
                                contract=managed_ci,
                            )
                            if managed_outcome.status == "passed":
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
                                    body=_ci_infrastructure_stop_message(
                                        pr_number, managed_outcome.stall, []
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
                                unresolved_items.append(
                                    _next_unresolved_item(
                                        item_number=next_unresolved_item_number,
                                        reviewer="GitHub managed exact-head CI",
                                        source_round=round_number,
                                        text=_pr_check_blocking_review(
                                            pr_number, "failing", details
                                        ),
                                        status="blocking",
                                    )
                                )
                                next_unresolved_item_number += 1
                                if round_number == allowed_rounds and not watch_failure_extension_used:
                                    allowed_rounds += 1
                                    watch_failure_extension_used = True
                            else:
                                raise AgentLoopError(
                                    f"Managed exact-head CI for PR #{pr_number} did not pass within "
                                    f"{config.ci_timeout_seconds}s."
                                )
                            must_fix_items = [
                                item
                                for item in unresolved_items
                                if item.status in {"blocking", "same-pr"}
                            ]
                            wait_outcome = None
                        elif ordinary_recovery_selected:
                            if ordinary_recovery is None:
                                raise AgentLoopError(
                                    f"PR #{pr_number} ordinary recovery provenance is unavailable; "
                                    "no merge attempted."
                                )
                            merged = _finalize_ordinary_recovery_merge(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                capability=ordinary_recovery,
                            )
                            if merged:
                                print(f"PR #{pr_number} merged after deliberate ordinary recovery.")
                            return 0
                        else:
                            wait_outcome = wait_for_ci(
                                runner, config, pr_number, metadata=pr_metadata
                            )
                        if wait_outcome is None:
                            pass
                        elif wait_outcome.status == "infrastructure_stall":
                            assert wait_outcome.stall is not None
                            post_pr_comment(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                body=_format_ci_infrastructure_comment(pr_number, wait_outcome.stall),
                            )
                            post_pr_comment(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                body=_ci_infrastructure_stop_message(pr_number, wait_outcome.stall, []),
                            )
                            log(
                                config,
                                f"Round {round_number}: PR #{pr_number} CI wait stopped on external "
                                "infrastructure blocking; no merge attempted",
                            )
                            print(
                                f"PR #{pr_number} was approved by "
                                f"{format_agent_list(configured_reviewers)}, but external GitHub "
                                "Actions infrastructure is blocking CI "
                                f"({'; '.join(_ci_infrastructure_details(wait_outcome.stall))}). "
                                "No merge was attempted; rerun the same command once GitHub Actions "
                                "runners recover."
                            )
                            return 0
                        elif wait_outcome.status == "merge_conflict":
                            assert wait_outcome.mergeability is not None
                            latest_mergeability = wait_outcome.mergeability
                            unresolved_items = _reconcile_merge_conflict_item(
                                unresolved_items,
                                mergeability=wait_outcome.mergeability,
                                source_round=round_number,
                                current_head_sha=pr_metadata.head_sha,
                            )
                            must_fix_items = [
                                item for item in unresolved_items if item.status in {"blocking", "same-pr"}
                            ]
                            log(
                                config,
                                f"Round {round_number}: PR #{pr_number} became conflicted with "
                                f"{wait_outcome.mergeability.base_branch or config.base or 'the base branch'} "
                                "during the CI wait; no merge attempted",
                            )
                        else:
                            _merge_with_exact_head_proof(
                                runner,
                                config=config,
                                pr_number=pr_number,
                                proof=ExactHeadCiProof(
                                    head_sha=pr_metadata.head_sha or "",
                                    source="configured CI",
                                ),
                            )
                    if not must_fix_items:
                        print(f"PR #{pr_number} approved by {format_agent_list(configured_reviewers)}.")
                        return 0
            if round_number == allowed_rounds:
                raise AgentLoopError(
                    f"One or more reviewers still reported blocking issues after round {round_number}; "
                    "human review required."
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
            stall_context = (
                _coder_infrastructure_stall_notice(pr_checks.infrastructure_stalls)
                if pr_checks is not None and is_wholly_infrastructure_blocked(pr_checks)
                else ""
            )

            same_pr_items = [item for item in unresolved_items if item.status == "same-pr"]
            blocking_items = [item for item in unresolved_items if item.status == "blocking"]
            if has_merge_conflict_item:
                other_items = [
                    item for item in unresolved_items if item.item_id != MERGE_CONFLICT_ITEM_ID
                ]
                combined_review = _format_unresolved_items_for_coder(other_items)
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
                )
                log(config, f"Round {round_number}: {coder_name} resolving merge conflict")
            elif same_pr_items and not blocking_items:
                combined_review = stall_context + _format_same_pr_unresolved_items(same_pr_items)
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
                )
                log(config, f"Round {round_number}: {coder_name} addressing reviewer feedback")
            else:
                combined_review = stall_context + _format_unresolved_items_for_coder(unresolved_items)
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
                )
                log(config, f"Round {round_number}: {coder_name} addressing reviewer feedback")
            repair_unresolved_item_ids = tuple(
                item.item_id
                for item in unresolved_items
                if item.item_id not in {HUMAN_REQUIREMENTS_ACK_ITEM_ID, MERGE_CONFLICT_ITEM_ID}
            )
            coder_response = _run_validated_agent(
                runner,
                agent=config.coder,
                config=config,
                prompt=followup_prompt,
                session_id=coder_session_id,
                marker_description="<!-- AGENT_STATE: approved|blocking -->",
                validate=lambda text, items=tuple(unresolved_items), human_requirements=human_requirements: _validate_coder_followup_response(
                    text,
                    unresolved_items=items,
                    human_requirements=human_requirements,
                ),
                usage_context=usage_context,
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
                validate_test_commands_within_workdir(
                    coder_response.marker_value.tests_run,
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

            post_pr_comment(
                runner,
                config=config,
                pr_number=pr_number,
                body=_attach_round_metadata(
                    public_comment,
                    PostedRoundMetadata(
                        flow="pr",
                        role="coder",
                        agent=coder_name,
                        round_number=round_number + 1,
                        subject=str(updated_pr_context.metadata.head_sha or "unknown"),
                        prior_items=tuple(unresolved_items),
                        raw_structured_coder_response=raw_structured_coder_response,
                        compact_prior_summaries=tuple(pr_compact_prior_summaries),
                        model_used=coder_response.model_used,
                        acquisition_outcome=coder_response.acquisition_outcome,
                        acquisition_returncode=coder_response.acquisition_returncode,
                    ),
                ),
            )
            log(config, f"Round {round_number}: {coder_name} pushed updates for re-review")
            pre_review_test_pending = True
            resumed_round = None
            prefetched_pr_context = updated_pr_context

        raise AgentLoopError(
            f"Reached max rounds ({config.max_rounds}) for PR #{pr_number}; human review required."
        )
    finally:
        cleanup_failure: AgentLoopError | None = None
        if (
            managed_ci is not None
            and not managed_ci_qualified
        ):
            should_release = managed_ci.adopted_existing_pr or config.managed_ci
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
        if owned_usage_context:
            _persist_usage_summary(config, usage_context)
        if cleanup_failure is not None:
            active_exception = sys.exc_info()[1]
            if active_exception is not None:
                # Preserve the original failure while making cleanup failure
                # visible in its traceback. Usage accounting above must run
                # even when label cleanup also fails.
                active_exception.add_note(str(cleanup_failure))
            else:
                raise cleanup_failure


DISCUSS_CONSENSUS_MARKER_RE = re.compile(
    r"<!--\s*AGENT_DISCUSS_CONSENSUS:\s*([0-9a-f]+)\s*-->",
    re.I,
)


def _is_bot_authored_discuss_comment(body: str) -> bool:
    if is_round_transport_sidecar(body):
        return True
    if DISCUSS_CONSENSUS_MARKER_RE.search(body):
        return True
    # Split-materialization comments (#476) can land on the same issue a
    # discuss run is evaluating (the parent and the discuss subject are the
    # same issue); they must not perturb subject hashing or be forwarded to
    # debaters as fresh human discussion on a later rerun.
    if (
        DISCUSS_SPLIT_MARKER_RE.search(body)
        or UNFILED_SPLIT_WARNING_MARKER_RE.search(body)
        or SPLIT_STAGE_HANDOFF_MARKER_RE.search(body)
    ):
        return True
    match = ROUND_RESUME_MARKER_RE.search(body)
    if match is None:
        return False
    try:
        metadata = _decode_round_metadata(match.group("payload"))
    except AgentLoopError:
        return False
    return metadata.flow == "discuss"


def _discuss_subject(issue_context: IssueContext) -> str:
    text = (issue_context.title or "") + "\n\n" + (issue_context.body or "")
    non_bot_bodies = [
        c.body
        for c in issue_context.comments
        if c.body and not _is_bot_authored_discuss_comment(c.body)
    ]
    if non_bot_bodies:
        text += "\n\n" + "\n\n".join(non_bot_bodies)
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _merge_discuss_split_proposals(votes: Sequence[ParsedDiscussReview]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for vote in votes:
        for proposal in vote.split_proposals:
            if proposal not in seen:
                seen.add(proposal)
                merged.append(proposal)
    return merged


def _detect_discuss_consensus(
    votes: list[ParsedDiscussReview],
) -> tuple[str, list[str]] | None:
    if not votes:
        return None
    outcome = votes[0].outcome
    if any(vote.outcome != outcome for vote in votes):
        return None
    split_proposals = _merge_discuss_split_proposals(votes) if outcome == "split" else []
    return outcome, split_proposals


def _normalize_discuss_answer(answer: str) -> str:
    return " ".join(answer.strip().split()).casefold()


def _detect_discuss_answer_consensus(
    responses: Sequence[ParsedDiscussAnswer], *, partial: bool = False
) -> tuple[str, list[str]] | None:
    if partial or not responses:
        return None
    if _discuss_has_material_items(responses):
        # A final round applies explicit outcome precedence. Before then,
        # classified material prevents normalized-text convergence.
        if not all(response.position == "needs-human" for response in responses):
            return None
    if all(response.position == "needs-human" for response in responses):
        return "needs-human", []
    answers = [response.answer for response in responses]
    if all(answer is not None for answer in answers):
        normalized = {_normalize_discuss_answer(answer or "") for answer in answers}
        if len(normalized) == 1:
            return "answer", []
    return None


def _aggregate_discuss_unresolved_items(
    responses: Sequence[ParsedDiscussAnswer],
) -> tuple[DiscussUnresolvedItem, ...]:
    """Return current-round classified items, deduplicated within a status.

    We deliberately retain identical text under different statuses: a blocker
    must not disappear merely because another debater calls it a follow-up.
    """
    seen: set[tuple[str, str]] = set()
    merged: list[DiscussUnresolvedItem] = []
    for response in responses:
        for item in response.unresolved_items:
            key = (item.status, item.text)
            if key not in seen:
                seen.add(key)
                merged.append(item)
    return tuple(merged)


def _discuss_has_material_items(responses: Sequence[ParsedDiscussAnswer]) -> bool:
    return any(
        item.status in {"blocker", "human-decision"}
        for item in _aggregate_discuss_unresolved_items(responses)
    )


def _final_discuss_answer_item_outcome(
    responses: Sequence[ParsedDiscussAnswer],
) -> str | None:
    statuses = {item.status for item in _aggregate_discuss_unresolved_items(responses)}
    if "human-decision" in statuses:
        return "needs-human"
    if "blocker" in statuses:
        return "deadlock"
    return None


def _handle_discuss_split_outcome(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    subject: str,
    split_proposals: Sequence[str],
    final_votes: Sequence[ParsedDiscussReview],
    issue_comments: Sequence[object],
    post_warning_comment: bool,
) -> None:
    """Materialize (or warn about) a discuss `split` consensus's proposed sub-issues (#476).

    Called both when the final summary is freshly posted and on a resumed/
    already-final run, so enabling `--materialize-split-issues` on a rerun
    still files the proposals instead of leaving them stuck in comments.
    Idempotent: `materialize_split_proposals` finds prior children via the
    parent marker and creates nothing when they already cover every proposal.
    """
    if not split_proposals:
        return
    proposals = dedupe_split_stage_proposals(
        [split_stage_proposal_from_text(proposal) for proposal in split_proposals]
    )
    if config.materialize_split_issues:
        rationale = tuple(
            (vote.reviewer, vote.rationale) for vote in final_votes if vote.outcome == "split"
        )
        materialize_split_proposals(
            runner,
            config=config,
            parent_issue=issue_number,
            subject=subject,
            proposals=proposals,
            rationale=rationale,
            issue_comments=issue_comments,
        )
        return
    log(
        config,
        f"discuss: split follow-ups are NOT filed as issues for #{issue_number}; rerun with "
        "--materialize-split-issues or file them manually.",
    )
    if post_warning_comment and not has_unfiled_split_warning(
        issue_comments, issue_number=issue_number, subject=subject
    ):
        post_unfiled_split_warning(
            runner,
            config=config,
            issue_number=issue_number,
            subject=subject,
            proposals=proposals,
        )


def _recover_final_discuss_split_proposals(
    issue_context: IssueContext,
    *,
    subject: str,
    configured_reviewers: Sequence[AgentName],
    reviewer_workdirs: Mapping[str, Path],
) -> tuple[list[str], list[ParsedDiscussReview]] | None:
    """Legacy fallback (#476): reconstruct final-round votes and merged split
    proposals from debater comment metadata when `PostedRoundMetadata` has no
    `split_proposals` recorded on the final summary (comments posted before
    this field existed)."""
    records = _extract_round_metadata_records(issue_context.comments, flow="discuss")
    subject_records = [record for record in records if record.metadata.subject == subject]
    if not subject_records:
        return None
    final_round_number = max(record.metadata.round_number for record in subject_records)
    debater_records = [
        record
        for record in subject_records
        if record.metadata.round_number == final_round_number and record.metadata.role == "debater"
    ]
    if not debater_records:
        return None
    configured_reviewer_names = [agent_display_name(agent) for agent in configured_reviewers]
    by_name = {record.metadata.agent: record for record in debater_records}
    final_votes: list[ParsedDiscussReview] = []
    for name in configured_reviewer_names:
        record = by_name.get(name)
        if record is None:
            continue
        vote = _decode_discuss_vote(
            record,
            round_number=final_round_number,
            reviewer_workdirs=reviewer_workdirs,
        )
        # This is triage-only legacy recovery. A mixed or malformed transcript
        # must not be interpreted as a split consensus.
        if not isinstance(vote, ParsedDiscussReview):
            return None
        final_votes.append(vote)
    if not final_votes:
        return None
    consensus = _detect_discuss_consensus(final_votes)
    if consensus is None or consensus[0] != "split":
        return None
    return consensus[1], final_votes


_DISCUSS_AGENDA_SUPPORT_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "this",
        "to",
        "use",
        "was",
        "we",
        "what",
        "when",
        "whether",
        "which",
        "while",
        "with",
        "would",
        # Low-signal discuss/analyzer boilerplate. These terms are common in
        # agenda summaries and should not validate invented content alone.
        "approach",
        "change",
        "custom",
        "existing",
        "implementation",
        "issue",
        "next",
        "objection",
        "question",
        "round",
        "scope",
        "strategy",
    }
)


def _normalize_discuss_agenda_phrase(text: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _tokenize_discuss_agenda_support(
    text: str,
    *,
    ignored_names: Sequence[str] = (),
) -> tuple[str, ...]:
    ignored_tokens = {
        token
        for name in ignored_names
        for token in re.findall(r"[A-Za-z0-9_]+", name.lower())
    }
    tokens: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
        if len(token) <= 2 and not token.isdigit():
            continue
        if token in ignored_tokens or token in _DISCUSS_AGENDA_SUPPORT_STOP_WORDS:
            continue
        tokens.append(token)
    return tuple(tokens)


@dataclass(frozen=True)
class _DiscussAgendaSupportCorpus:
    phrase_segments: tuple[str, ...]
    tokens: frozenset[str]


def _build_discuss_agenda_support_corpus(
    *,
    issue_context: IssueContext,
    round_history: Sequence[Sequence[ParsedDiscussResponse]],
    prior_agenda: ParsedDiscussAgenda | None,
    configured_reviewers: Sequence[AgentName],
    analyzer: AgentName,
) -> _DiscussAgendaSupportCorpus:
    segments: list[str] = []
    phrase_segments: list[str] = []

    def add(value: object, *, phrase_support: bool = False) -> None:
        if isinstance(value, str) and value.strip():
            segments.append(value)
            if phrase_support:
                phrase_segments.append(value)

    add(issue_context.title, phrase_support=True)
    add(issue_context.body, phrase_support=True)
    for comment in issue_context.comments:
        add(comment.author)
        add(comment.created_at)
        add(comment.body, phrase_support=True)
    for reviewer in configured_reviewers:
        add(agent_display_name(reviewer))
    for round_index, votes in enumerate(round_history, start=1):
        add(f"Round {round_index}")
        for vote in votes:
            add(vote.reviewer)
            if isinstance(vote, ParsedDiscussAnswer):
                add(vote.position)
                add(vote.answer, phrase_support=True)
                add(vote.rationale, phrase_support=True)
                for item in vote.unresolved_items:
                    add(item.status)
                    add(item.text, phrase_support=True)
            elif isinstance(vote, ParsedDiscussReview):
                add(vote.outcome)
                add(vote.rationale, phrase_support=True)
                for proposal in vote.split_proposals:
                    add(proposal, phrase_support=True)
            else:
                add("failed")
                add(vote.category, phrase_support=True)
            add(getattr(vote, "rebuttal", None), phrase_support=True)
            add(getattr(vote, "analyzer_framing", None))
            add(getattr(vote, "framing_note", None), phrase_support=True)
            add(getattr(vote, "research_status", None))
            add(getattr(vote, "research_target", None))
            for question in getattr(vote, "research_questions", ()):
                add(question, phrase_support=True)
            for fact in getattr(vote, "sourced_facts", ()):
                add(fact.fact, phrase_support=True)
                add(fact.source, phrase_support=True)
    if prior_agenda is not None:
        for point in prior_agenda.consensus:
            add(point, phrase_support=True)
        for disagreement in prior_agenda.disagreements:
            add(disagreement.topic, phrase_support=True)
            for name, position in disagreement.positions:
                add(name)
                add(position, phrase_support=True)
            add(disagreement.question_for_next_round, phrase_support=True)
        for fact in prior_agenda.missing_facts:
            add(fact, phrase_support=True)
        for question in prior_agenda.research_questions:
            add(question, phrase_support=True)
        for target in prior_agenda.research_question_targets:
            add(target)

    ignored_names = [
        agent_display_name(analyzer),
        *(agent_display_name(r) for r in configured_reviewers),
    ]
    tokens = frozenset(
        token
        for segment in segments
        for token in _tokenize_discuss_agenda_support(segment, ignored_names=ignored_names)
    )
    return _DiscussAgendaSupportCorpus(
        phrase_segments=tuple(
            normalized
            for segment in phrase_segments
            if (normalized := _normalize_discuss_agenda_phrase(segment))
        ),
        tokens=tokens,
    )


def _discuss_agenda_text_has_support(
    text: str,
    *,
    corpus: _DiscussAgendaSupportCorpus,
    ignored_names: Sequence[str],
) -> bool:
    tokens = _tokenize_discuss_agenda_support(text, ignored_names=ignored_names)
    if tokens:
        supported = sum(1 for token in set(tokens) if token in corpus.tokens)
        required = 1 if len(set(tokens)) <= 4 else 2
        return supported >= required
    normalized = _normalize_discuss_agenda_phrase(text)
    return bool(
        normalized
        and any(
            normalized == segment or normalized in segment
            for segment in corpus.phrase_segments
        )
    )


def _validate_discuss_analyzer_agenda_fidelity(
    agenda: ParsedDiscussAgenda,
    *,
    issue_context: IssueContext,
    round_history: Sequence[Sequence[ParsedDiscussResponse]],
    prior_agenda: ParsedDiscussAgenda | None,
    configured_reviewers: Sequence[AgentName],
    analyzer: AgentName,
) -> None:
    allowed_names = {agent_display_name(reviewer) for reviewer in configured_reviewers}
    targets = agenda.research_question_targets
    if targets and len(targets) != len(agenda.research_questions):
        raise AgentLoopError(
            "analyzer agenda research_question_targets must align one-to-one with research_questions."
        )
    invalid_targets = sorted(set(targets) - DISCUSS_RESEARCH_TARGET_VALUES)
    if invalid_targets:
        raise AgentLoopError(
            "analyzer agenda used invalid research target(s): " + ", ".join(invalid_targets)
        )
    unknown_names = sorted(
        {name for disagreement in agenda.disagreements for name, _position in disagreement.positions}
        - allowed_names
    )
    if unknown_names:
        raise AgentLoopError(
            "analyzer agenda used unknown debater name(s): " + ", ".join(unknown_names)
        )

    ignored_names = [agent_display_name(analyzer), *sorted(allowed_names)]
    corpus = _build_discuss_agenda_support_corpus(
        issue_context=issue_context,
        round_history=round_history,
        prior_agenda=prior_agenda,
        configured_reviewers=configured_reviewers,
        analyzer=analyzer,
    )

    fields: list[tuple[str, str]] = []
    fields.extend(("consensus", point) for point in agenda.consensus)
    for disagreement in agenda.disagreements:
        fields.append(("disagreement topic", disagreement.topic))
        fields.extend(("position", position) for _name, position in disagreement.positions)
        fields.append(("question_for_next_round", disagreement.question_for_next_round))
    fields.extend(("missing_fact", fact) for fact in agenda.missing_facts)
    fields.extend(("research_question", question) for question in agenda.research_questions)

    for field, text in fields:
        if not _discuss_agenda_text_has_support(text, corpus=corpus, ignored_names=ignored_names):
            raise AgentLoopError(f"analyzer agenda {field} lacks transcript support: {text}")


def _run_discuss_analyzer(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    analyzer: AgentName,
    memory: AgentMemoryContext | None,
    issue_context: IssueContext,
    round_number: int,
    round_history: Sequence[Sequence[ParsedDiscussResponse]],
    prior_agenda: ParsedDiscussAgenda | None,
    configured_reviewers: Sequence[AgentName],
    usage_context: RunUsageContext,
) -> tuple[ParsedDiscussAgenda | None, str | None]:
    """Run the optional discuss analyzer after a non-final round.

    Returns (parsed_agenda, raw_response). Any analyzer failure that survives
    the repair pass falls back to (None, None) so the round closes with the
    mechanical agenda and the next debate round sees the full prior positions;
    only a quota-reset stop propagates because the whole run must pause.
    """
    analyzer_name = agent_display_name(analyzer)
    log(
        config,
        f"discuss: invoking analyzer {analyzer_name} on issue #{issue_number} "
        f"(after round {round_number})",
    )
    try:
        response = _run_validated_agent(
            runner,
            agent=analyzer,
            config=config,
            prompt=build_discuss_agenda_prompt(
                issue_number,
                config,
                analyzer=analyzer,
                memory=memory,
                issue_context=issue_context,
                round_number=round_number,
                round_history=round_history,
                prior_agenda=prior_agenda,
                research_mode=config.discuss_research,
            ),
            marker_description="<!-- AGENT_PLAN_STATE: approved -->",
            validate=validate_structured_discuss_agenda,
            usage_context=usage_context,
            use_repair=True,
            repair_expected_kind="discuss_agenda",
            role="reviewer",
            label=f"discuss-analyzer-r{round_number}",
            operation_description="discuss analyzer",
        )
    except QuotaResetExceededError:
        raise
    except AgentLoopError as exc:
        log(
            config,
            f"discuss: analyzer {analyzer_name} failed ({exc}); falling back to the "
            f"mechanical agenda for round {round_number + 1}",
        )
        return None, None
    parsed = response.marker_value
    assert isinstance(parsed, ParsedDiscussAgenda)
    try:
        _validate_discuss_analyzer_agenda_fidelity(
            parsed,
            issue_context=issue_context,
            round_history=round_history,
            prior_agenda=prior_agenda,
            configured_reviewers=configured_reviewers,
            analyzer=analyzer,
        )
    except AgentLoopError as exc:
        log(
            config,
            f"discuss: analyzer {analyzer_name} agenda rejected ({exc}); falling back to the "
            f"mechanical agenda for round {round_number + 1}",
        )
        return None, None
    return parsed, response.text


def _validate_discuss_final_analyzer_fidelity(
    agenda: ParsedDiscussAgenda,
    *,
    final_votes: Sequence[ParsedDiscussResponse],
    configured_reviewers: Sequence[AgentName],
    analyzer: AgentName,
) -> None:
    """Validate an advisory final pass against final debater text only (#529)."""
    if not final_votes or any(is_failed_discuss_response(vote) for vote in final_votes):
        raise AgentLoopError("final analyzer requires successful final-round responses.")
    if agenda.research_required or agenda.research_questions or agenda.research_question_targets:
        raise AgentLoopError("final analyzer output must not include next-round research fields.")
    allowed_names = {agent_display_name(reviewer) for reviewer in configured_reviewers}
    final_names = {vote.reviewer for vote in final_votes}
    unknown_names = sorted(
        {name for disagreement in agenda.disagreements for name, _position in disagreement.positions}
        - allowed_names.intersection(final_names)
    )
    if unknown_names:
        raise AgentLoopError("final analyzer used unknown or absent debater name(s): " + ", ".join(unknown_names))
    empty_context = IssueContext(0, "", None, None, None, ())
    corpus = _build_discuss_agenda_support_corpus(
        issue_context=empty_context,
        round_history=(tuple(final_votes),),
        prior_agenda=None,
        configured_reviewers=configured_reviewers,
        analyzer=analyzer,
    )
    ignored_names = [agent_display_name(analyzer), *sorted(allowed_names)]
    fields: list[tuple[str, str]] = [("consensus", point) for point in agenda.consensus]
    for disagreement in agenda.disagreements:
        fields.append(("disagreement topic", disagreement.topic))
        fields.extend(("position", position) for _name, position in disagreement.positions)
        fields.append(("question_for_next_round", disagreement.question_for_next_round))
    fields.extend(("missing_fact", fact) for fact in agenda.missing_facts)
    for field, text in fields:
        if not _discuss_agenda_text_has_support(text, corpus=corpus, ignored_names=ignored_names):
            raise AgentLoopError(f"final analyzer {field} lacks final-round support: {text}")


def _run_discuss_final_analyzer(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    analyzer: AgentName | None,
    round_number: int,
    final_votes: Sequence[ParsedDiscussResponse],
    configured_reviewers: Sequence[AgentName],
    usage_context: RunUsageContext,
) -> tuple[ParsedDiscussAgenda | None, str | None]:
    """Best-effort final-only analyzer pass; failures never affect finalization."""
    if analyzer is None or not final_votes or any(is_failed_discuss_response(vote) for vote in final_votes):
        return None, None


    analyzer_name = agent_display_name(analyzer)
    try:
        response = _run_validated_agent(
            runner, agent=analyzer, config=config,
            prompt=build_discuss_final_analysis_prompt(
                issue_number, config, analyzer=analyzer, round_number=round_number,
                final_votes=final_votes,
            ),
            marker_description="<!-- AGENT_PLAN_STATE: approved -->",
            validate=validate_structured_discuss_agenda, usage_context=usage_context,
            use_repair=True, repair_expected_kind="discuss_agenda", role="reviewer",
            label=f"discuss-final-analyzer-r{round_number}",
            operation_description="final discuss analyzer",
        )
        parsed = response.marker_value
        assert isinstance(parsed, ParsedDiscussAgenda)
        _validate_discuss_final_analyzer_fidelity(
            parsed, final_votes=final_votes, configured_reviewers=configured_reviewers, analyzer=analyzer,
        )
        return parsed, response.text
    except QuotaResetExceededError:
        raise
    except Exception as exc:
        log(config, f"discuss: final analyzer {analyzer_name} unavailable ({exc}); omitting advisory observations")
        return None, None


def _run_discuss_evidence_reconciler(
    runner: Runner, *, issue_number: int, config: AgentLoopConfig, analyzer: AgentName | None,
    subject: str, round_history: Sequence[Sequence[ParsedDiscussResponse]], usage_context: RunUsageContext,
) -> tuple[tuple[tuple[str, ...], ...], str | None]:
    """Best-effort semantic grouping, isolated from the final agenda analyzer."""
    if analyzer is None:
        return (), None
    observations, updates = collect_evidence_observations(subject, round_history)
    candidates = bounded_reconciliation_candidates(observations, updates)
    candidate_ids = {str(candidate["id"]) for candidate in candidates}
    statuses = {item.observation_id: item.status for item in observations if item.observation_id in candidate_ids}
    if len(candidates) < 2:
        return (), None
    try:
        response = _run_validated_agent(
            runner, agent=analyzer, config=config,
            prompt=build_discuss_evidence_reconciliation_prompt(issue_number, config, analyzer=analyzer, candidates=candidates),
            marker_description="<!-- AGENT_PLAN_STATE: approved -->",
            validate=lambda text: validate_structured_discuss_evidence_reconciliation(text, observation_ids=tuple(statuses), observation_statuses=statuses),
            usage_context=usage_context, use_repair=True,
            repair_expected_kind="discuss_evidence_reconciliation", role="analyzer",
            label="discuss-evidence-reconciliation", operation_description="evidence reconciliation",
        )
        parsed = response.marker_value
        assert isinstance(parsed, ParsedDiscussEvidenceReconciliation)
        return parsed.groups, response.text
    except QuotaResetExceededError:
        raise
    except Exception as exc:
        log(config, f"discuss: evidence reconciler unavailable ({exc}); using exact-match ledger")
        return (), None


@dataclass(frozen=True)
class _DiscussDebaterTurnResult:
    """Outcome of one debater turn: a validated response or a captured failure.

    Worker threads in the parallel path return these instead of raising, so
    exceptions never cross the thread boundary; the main thread applies the
    failure policy after all futures settle (#475).
    """

    reviewer_name: str
    response: ValidatedAgentResponse | None = None
    error: AgentLoopError | None = None

    @property
    def failure_category(self) -> str:
        return getattr(self.error, "failure_category", None) or "error"


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


def _validate_structured_discuss_vote_with_evidence(
    text: str,
    *,
    reviewer_name: str,
    round_number: int,
    config: AgentLoopConfig,
    assigned_workdir: Path,
) -> ParsedDiscussResponse:
    """Parse a debater's structured vote, then reject any checkout-inspected
    evidence claim whose path:line reference does not resolve inside the
    reviewer's own assigned checkout right now (#541)."""
    parsed = (
        validate_structured_discuss_answer(
            text, reviewer=reviewer_name, round_number=round_number, research_mode=config.discuss_research
        )
        if config.discuss_result_mode == "answer"
        else validate_structured_discuss_review(
            text, reviewer=reviewer_name, round_number=round_number, research_mode=config.discuss_research
        )
    )
    validate_checkout_inspected_evidence(parsed.evidence_claims, assigned_workdir=assigned_workdir)
    return parsed


def _run_discuss_debater_turn(
    runner: Runner,
    *,
    reviewer: AgentName,
    reviewer_name: str,
    config: AgentLoopConfig,
    prompt: str,
    round_number: int,
    usage_context: RunUsageContext,
) -> ValidatedAgentResponse:
    """Run one debater turn from a prebuilt prompt.

    Never posts to GitHub or mutates shared round state, so it is safe to call
    from a worker thread; the caller posts the comment after the round's
    synchronization point.
    """
    assigned_workdir = get_backend(reviewer).workdir(config)
    return _run_validated_agent(
        runner,
        agent=reviewer,
        config=config,
        prompt=prompt,
        marker_description="<!-- AGENT_PLAN_STATE: approved -->",
        validate=lambda text, r=reviewer_name, rn=round_number, wd=assigned_workdir: (
            _validate_structured_discuss_vote_with_evidence(
                text, reviewer_name=r, round_number=rn, config=config, assigned_workdir=wd
            )
        ),
        usage_context=usage_context,
        use_repair=True,
        repair_expected_kind="discuss_answer" if config.discuss_result_mode == "answer" else "discuss_review",
        role="reviewer",
        label=f"discuss-r{round_number}",
        timeout_seconds=config.discuss_debater_timeout,
        operation_description="discuss review",
    )


def _run_discuss_semantic_finalization(
    runner: Runner, *, issue_number: int, config: AgentLoopConfig, answers: Sequence[ParsedDiscussAnswer],
    configured_reviewers: Sequence[AgentName], usage_context: RunUsageContext,
) -> tuple[str, str, dict[str, object] | None]:
    """Fail-closed semantic answer evaluation. Exact equality is intentionally outside this helper."""
    analyzer = config.discuss_analyzer
    if analyzer is None or len(answers) != len(configured_reviewers):
        return "deadlock", "deadlock", None
    if any(item.position != "answer" or not item.answer for item in answers) or _discuss_has_material_items(answers):
        return "deadlock", "deadlock", None
    names = [item.reviewer for item in answers]
    try:
        response = _run_validated_agent(
            runner, agent=analyzer, config=config,
            prompt=build_discuss_semantic_comparison_prompt(issue_number, config, answers=answers),
            marker_description="<!-- AGENT_PLAN_STATE: approved -->",
            validate=lambda text: validate_structured_discuss_semantic_comparison(text, reviewers=names),
            usage_context=usage_context, use_repair=True,
            repair_expected_kind="discuss_semantic_comparison", role="analyzer",
            label="discuss-semantic-comparison", operation_description="semantic answer comparison",
        )
        comparison = response.marker_value
        assert isinstance(comparison, ParsedDiscussSemanticComparison)
    except (AgentLoopError, AgentInvocationError):
        # Preserve an explicit audit record even when the comparator itself
        # fails. This distinguishes its fail-closed deadlock from a purely
        # textual disagreement in both the public summary and round metadata.
        return "deadlock", "semantic-comparison-failed", {
            "classification": "failed",
            "analyzer": agent_display_name(analyzer),
        }
    audit: dict[str, object] = {
        "classification": comparison.classification,
        "shared_recommendation": comparison.shared_recommendation,
        "remaining_decisions": comparison.remaining_decisions,
        "evidence": comparison.evidence,
        "analyzer": agent_display_name(analyzer),
    }
    if comparison.classification == "equivalent":
        return "answer", "semantic-equivalent", audit
    if comparison.classification == "material_conflict":
        return "deadlock", "material-conflict", audit
    confirmations = []
    try:
        for reviewer in configured_reviewers:
            reviewer_name = agent_display_name(reviewer)
            response = _run_validated_agent(
                runner, agent=reviewer, config=config,
                prompt=build_discuss_answer_confirmation_prompt(
                    issue_number, config, reviewer=reviewer,
                    shared_recommendation=comparison.shared_recommendation,
                    remaining_decisions=comparison.remaining_decisions),
                marker_description="<!-- AGENT_PLAN_STATE: approved -->",
                validate=lambda text, name=reviewer_name: validate_structured_discuss_answer_confirmation(text, reviewer=name),
                usage_context=usage_context, use_repair=True,
                repair_expected_kind="discuss_answer_confirmation", role="reviewer",
                label="discuss-answer-confirmation", timeout_seconds=config.discuss_debater_timeout,
                operation_description="semantic answer confirmation",
            )
            confirmations.append(response.marker_value)
    except (AgentLoopError, AgentInvocationError):
        return "deadlock", "confirmation-failed", audit
    effective = [comparison.shared_recommendation if item.decision == "confirm" else item.answer for item in confirmations]
    audit["confirmations"] = tuple(confirmations)
    if all(answer and _normalize_discuss_answer(answer) == _normalize_discuss_answer(effective[0] or "") for answer in effective):
        audit["confirmed_answer"] = effective[0]
        return "answer", "debater-confirmed", audit
    return "deadlock", "confirmation-disagreement", audit


def _post_discuss_debater_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    reviewer_name: str,
    parsed: ParsedDiscussResponse,
    response: ValidatedAgentResponse,
    round_number: int,
    subject: str,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=issue_number,
        body=_attach_round_metadata(
            render_public_agent_comment(
                kind="discuss_answer" if config.discuss_result_mode == "answer" else "discuss_review",
                parsed=parsed,
                agent=reviewer_name,
                config=config,
                model_used=response.model_used,
                round_number=round_number,
            ),
            PostedRoundMetadata(
                flow="discuss",
                role="debater",
                agent=reviewer_name,
                round_number=round_number,
                subject=subject,
                raw_structured_coder_response=response.text,
                model_used=response.model_used,
                research_mode=config.discuss_research,
                result_mode=config.discuss_result_mode,
            ),
        ),
    )


def _validate_discuss_evidence_update_targets(
    parsed: ParsedDiscussResponse, *, subject: str, round_history: Sequence[Sequence[ParsedDiscussResponse]],
) -> None:
    """Reject active malformed updates; legacy replay merely audits them."""
    if not isinstance(parsed, (ParsedDiscussReview, ParsedDiscussAnswer)):
        return
    observations, _updates = collect_evidence_observations(subject, round_history)
    allowed = {item.observation_id for item in observations}
    for update in parsed.evidence_updates:
        if not update.target_observation_id.startswith(f"{subject}-") or update.target_observation_id not in allowed:
            raise AgentLoopError(f"evidence update targets unknown or cross-subject observation: {update.target_observation_id}")


def _run_discuss_loop(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    usage_context: RunUsageContext,
    discuss_max_rounds: int = 2,
) -> int:
    from .config import reviewers as _reviewers
    if discuss_max_rounds < 0:
        raise AgentLoopError("--discuss-max-rounds must be zero or greater.")
    issue_context = get_issue_context(runner, config=config, issue_number=issue_number)
    subject = _discuss_subject(issue_context)
    configured_reviewers = list(_reviewers(config))
    reviewer_workdirs = {
        agent_display_name(agent): get_backend(agent).workdir(config) for agent in configured_reviewers
    }
    resume_state = _resume_discuss_round(
        issue_context.comments, subject=subject, configured_reviewers=configured_reviewers,
        reviewer_workdirs=reviewer_workdirs, result_mode=config.discuss_result_mode,
    )

    def _resolve_final_split_proposals() -> tuple[list[str], Sequence[ParsedDiscussReview]] | None:
        if config.discuss_result_mode == "answer":
            return None
        # Resuming (or already-final) reruns must materialize a split consensus
        # instead of silently skipping (#476): prefer the split proposals
        # recorded directly on the final summary metadata, and fall back to
        # reconstructing them from debater comment metadata for legacy
        # summaries that predate the `split_proposals` field.
        if resume_state is not None and resume_state.done and resume_state.round_history:
            final_votes = resume_state.round_history[-1]
            consensus = _detect_discuss_consensus(list(final_votes))
            if consensus is not None and consensus[0] == "split" and consensus[1]:
                return consensus[1], final_votes
        return _recover_final_discuss_split_proposals(
            issue_context, subject=subject, configured_reviewers=configured_reviewers,
            reviewer_workdirs=reviewer_workdirs,
        )

    already_final = False
    for comment in issue_context.comments or []:
        body = comment.body or ""
        m = DISCUSS_CONSENSUS_MARKER_RE.search(body)
        if m and m.group(1).lower() == subject:
            already_final = True
            break
    if already_final or (resume_state is not None and resume_state.done):
        log(config, f"discuss: found matching consensus for issue #{issue_number}; skipping debate")
        recovered = _resolve_final_split_proposals()
        if recovered is not None:
            split_proposals, final_votes = recovered
            _handle_discuss_split_outcome(
                runner,
                issue_number=issue_number,
                config=config,
                subject=subject,
                split_proposals=split_proposals,
                final_votes=final_votes,
                issue_comments=issue_context.comments,
                post_warning_comment=False,
            )
        elif config.materialize_split_issues:
            log(
                config,
                f"discuss: issue #{issue_number} has a final consensus comment but no "
                "recoverable split-proposal metadata; nothing to materialize (or this was not "
                "a `split` consensus).",
            )
        return 0
    memory = prepare_agent_memory(runner, config)
    analyzer = config.discuss_analyzer
    analyzer_name = agent_display_name(analyzer) if analyzer is not None else None
    prompt_issue_context = issue_context
    if analyzer is not None:
        # Agenda-focused analyzer mode: prior rounds reach debaters only through
        # the structured agenda (or the analyzer via round_history), so strip
        # discuss-flow bot comments from the prompt-facing issue context. Plain
        # mode keeps the full context unchanged.
        prompt_issue_context = dataclasses_replace(
            issue_context,
            comments=tuple(
                comment
                for comment in (issue_context.comments or ())
                if not (comment.body and _is_bot_authored_discuss_comment(comment.body))
            ),
        )
    if analyzer is not None and discuss_max_rounds == 0:
        log(
            config,
            "discuss: --discuss-analyzer is set but --discuss-max-rounds=0 leaves no "
            "non-final round, so the analyzer will not run.",
        )
    if resume_state is not None:
        round_history: list[list[ParsedDiscussReview]] = [list(votes) for votes in resume_state.round_history]
        start_round_number = resume_state.next_round_number
        prior_round_agenda: list[str] = list(resume_state.prior_round_agenda)
        prior_analyzer_agenda: ParsedDiscussAgenda | None = (
            resume_state.prior_analyzer_agenda if analyzer is not None else None
        )
        in_progress_votes: dict[str, ParsedDiscussReview] = dict(resume_state.in_progress_votes)
        if round_history or in_progress_votes:
            log(config, f"discuss: resuming issue #{issue_number} at round {start_round_number}")
    else:
        round_history = []
        start_round_number = 1
        prior_round_agenda = []
        prior_analyzer_agenda = None
        in_progress_votes = {}
    max_round_number = discuss_max_rounds + 1
    if start_round_number > max_round_number:
        if not round_history:
            raise AgentLoopError(
                "discuss: resumed state expects round "
                f"{start_round_number} but --discuss-max-rounds={discuss_max_rounds} allows only "
                f"{max_round_number} round(s), and no completed round was found to finalize from. "
                "Rerun with a --discuss-max-rounds at least as large as the rounds already posted "
                "on the issue, or repair the discuss transcript."
            )
        log(
            config,
            f"discuss: resumed round {start_round_number} exceeds --discuss-max-rounds="
            f"{discuss_max_rounds} (allows {max_round_number} round(s)); finalizing from the last "
            "completed round instead of starting a new one.",
        )
        final_round_number = len(round_history)
        final_votes = round_history[-1]
        # A resumed partial round (#475) may carry placeholder votes; keep the
        # vote table to real positions and surface the rest as failures.
        final_successful_votes = [
            vote for vote in final_votes if vote.outcome != DISCUSS_FAILED_OUTCOME
        ]
        final_failed_debaters = tuple(
            (vote.reviewer, failed_discuss_review_category(vote))
            for vote in final_votes
            if vote.outcome == DISCUSS_FAILED_OUTCOME
        )
        final_analyzer_agenda, final_analyzer_response_raw = _run_discuss_final_analyzer(
            runner,
            issue_number=issue_number,
            config=config,
            analyzer=analyzer,
            round_number=final_round_number,
            final_votes=final_successful_votes,
            configured_reviewers=configured_reviewers,
            usage_context=usage_context,
        )
        evidence_groups, evidence_reconciler_raw = _run_discuss_evidence_reconciler(
            runner, issue_number=issue_number, config=config, analyzer=analyzer, subject=f"issue-{issue_number}",
            round_history=round_history, usage_context=usage_context,
        )
        evidence_reconciliation = reconcile_evidence(f"issue-{issue_number}", round_history, evidence_groups)
        evidence_reconciliation["raw_evidence_reconciler_response"] = evidence_reconciler_raw
        summary_body = render_discuss_round_summary_comment(
            round_number=final_round_number,
            reviewer_votes=final_successful_votes,
            is_final=True,
            subject=subject,
            outcome="needs-human",
            consensus_kind="deadlock",
            round_history=round_history,
            split_proposals=[],
            prior_analyzer_agenda=prior_analyzer_agenda,
            final_analyzer_agenda=final_analyzer_agenda,
            analyzer_name=analyzer_name,
            research_mode=config.discuss_research,
            failed_debaters=final_failed_debaters,
            result_mode=config.discuss_result_mode,
            evidence_reconciliation=evidence_reconciliation,
        )
        post_issue_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=_attach_round_metadata(
                summary_body,
                PostedRoundMetadata(
                    flow="discuss",
                    role="summary",
                    agent="Orchestrator",
                    round_number=final_round_number,
                    subject=subject,
                    is_final=True,
                    consensus_kind="deadlock",
                    agenda=(),
                    final_analyzer_response=final_analyzer_response_raw,
                    research_mode=config.discuss_research,
                    failed_debaters=final_failed_debaters,
                    evidence_reconciliation=evidence_reconciliation,
                ),
            ),
        )
        log(
            config,
            f"discuss: posted final summary comment for issue #{issue_number} "
            "(outcome: needs-human; kind: deadlock)",
        )
        return 0
    for round_number in range(start_round_number, max_round_number + 1):
        prior_round_votes = round_history[-1] if round_history else []
        votes_by_name: dict[str, ParsedDiscussReview] = {}
        failures_by_name: dict[str, _DiscussDebaterTurnResult] = {}
        pending: list[AgentName] = []
        for reviewer in configured_reviewers:
            reviewer_name = agent_display_name(reviewer)
            resumed_vote = (
                in_progress_votes.get(reviewer_name) if round_number == start_round_number else None
            )
            if resumed_vote is not None:
                log(
                    config,
                    f"discuss: resuming {reviewer_name}'s posted round {round_number} position "
                    f"on issue #{issue_number}",
                )
                votes_by_name[reviewer_name] = resumed_vote
            else:
                pending.append(reviewer)

        def _build_debater_prompt(reviewer: AgentName) -> str:
            return build_discuss_review_prompt(
                issue_number,
                config,
                reviewer=reviewer,
                memory=memory,
                issue_context=prompt_issue_context,
                round_number=round_number,
                prior_round_votes=prior_round_votes,
                prior_round_agenda=prior_round_agenda,
                analyzer_agenda=prior_analyzer_agenda,
                research_mode=config.discuss_research,
            )

        if config.discuss_parallel and pending:
            # Same-round debaters run concurrently; prompts are built up front
            # from shared pre-round state and comments are posted only after
            # every future settles, so no debater can see a co-debater's
            # in-progress round-N output. Zero-pending resumes skip this branch
            # entirely (no executor is constructed).
            prompts = {
                agent_display_name(reviewer): _build_debater_prompt(reviewer)
                for reviewer in pending
            }
            pending_names = [agent_display_name(reviewer) for reviewer in pending]
            log(
                config,
                f"discuss: invoking {', '.join(pending_names)} in parallel on "
                f"issue #{issue_number} (round {round_number})",
            )

            def _debater_worker(reviewer: AgentName, reviewer_name: str) -> _DiscussDebaterTurnResult:
                try:
                    response = _run_discuss_debater_turn(
                        runner,
                        reviewer=reviewer,
                        reviewer_name=reviewer_name,
                        config=config,
                        prompt=prompts[reviewer_name],
                        round_number=round_number,
                        usage_context=usage_context,
                    )
                    parsed = response.marker_value
                    assert isinstance(parsed, (ParsedDiscussReview, ParsedDiscussAnswer))
                    _validate_discuss_evidence_update_targets(
                        parsed, subject=f"issue-{issue_number}", round_history=round_history
                    )
                except AgentLoopError as exc:
                    # Includes QuotaResetExceededError: captured here and
                    # re-raised on the main thread with priority.
                    return _DiscussDebaterTurnResult(reviewer_name=reviewer_name, error=exc)
                return _DiscussDebaterTurnResult(reviewer_name=reviewer_name, response=response)

            executor = ThreadPoolExecutor(
                max_workers=len(pending), thread_name_prefix=f"discuss-r{round_number}"
            )
            try:
                futures = {
                    agent_display_name(reviewer): executor.submit(
                        _debater_worker, reviewer, agent_display_name(reviewer)
                    )
                    for reviewer in pending
                }
                # The analyzer synchronization point: wait for every debater.
                turn_results = {name: future.result() for name, future in futures.items()}
            except KeyboardInterrupt:
                # Workers never receive the terminal SIGINT; kill their agent
                # process groups so their wait loops return and the shutdown
                # below completes promptly, then propagate the interrupt.
                runner.terminate_active_processes()
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            for name in pending_names:
                turn = turn_results[name]
                if turn.error is None:
                    parsed = turn.response.marker_value
                    assert isinstance(parsed, (ParsedDiscussReview, ParsedDiscussAnswer))
                    _post_discuss_debater_comment(
                        runner,
                        config=config,
                        issue_number=issue_number,
                        reviewer_name=name,
                        parsed=parsed,
                        response=turn.response,
                        round_number=round_number,
                        subject=subject,
                    )
                    votes_by_name[name] = parsed
                else:
                    failures_by_name[name] = turn
            # Successful votes are posted above even when the round is about
            # to abort, so a rerun resumes them instead of re-invoking.
            for name in pending_names:
                turn = failures_by_name.get(name)
                if turn is not None and isinstance(turn.error, QuotaResetExceededError):
                    raise turn.error
            if failures_by_name and config.discuss_on_debater_failure == "fail":
                raise next(iter(failures_by_name.values())).error
        else:
            for reviewer in pending:
                reviewer_name = agent_display_name(reviewer)
                log(
                    config,
                    f"discuss: invoking {reviewer_name} on issue #{issue_number} "
                    f"(round {round_number})",
                )
                try:
                    response = _run_discuss_debater_turn(
                        runner,
                        reviewer=reviewer,
                        reviewer_name=reviewer_name,
                        config=config,
                        prompt=_build_debater_prompt(reviewer),
                        round_number=round_number,
                        usage_context=usage_context,
                    )
                except QuotaResetExceededError:
                    raise
                except AgentLoopError as exc:
                    if config.discuss_on_debater_failure == "fail":
                        raise
                    failures_by_name[reviewer_name] = _DiscussDebaterTurnResult(
                        reviewer_name=reviewer_name, error=exc
                    )
                    log(
                        config,
                        f"discuss: {reviewer_name} failed round {round_number} "
                        f"({failures_by_name[reviewer_name].failure_category}); continuing "
                        "per --discuss-on-debater-failure=partial",
                    )
                    continue
                parsed = response.marker_value
                assert isinstance(parsed, (ParsedDiscussReview, ParsedDiscussAnswer))
                _validate_discuss_evidence_update_targets(
                    parsed, subject=f"issue-{issue_number}", round_history=round_history
                )
                _post_discuss_debater_comment(
                    runner,
                    config=config,
                    issue_number=issue_number,
                    reviewer_name=reviewer_name,
                    parsed=parsed,
                    response=response,
                    round_number=round_number,
                    subject=subject,
                )
                votes_by_name[reviewer_name] = parsed

        failed_debaters: list[tuple[str, str]] = []
        reviewer_votes: list[ParsedDiscussResponse] = []
        for reviewer in configured_reviewers:
            reviewer_name = agent_display_name(reviewer)
            vote = votes_by_name.get(reviewer_name)
            if vote is not None:
                reviewer_votes.append(vote)
                continue
            failure = failures_by_name[reviewer_name]
            category = failure.failure_category
            failed_debaters.append((reviewer_name, category))
            reviewer_votes.append(
                failed_discuss_answer_placeholder(reviewer_name, category)
                if config.discuss_result_mode == "answer"
                else failed_discuss_review_placeholder(reviewer_name, category)
            )
        if failed_debaters:
            # Reached only under the "partial" policy: continue when at least
            # two debaters produced votes; a partial round can never declare
            # final consensus because the placeholder outcome differs.
            if len(votes_by_name) < 2:
                log(
                    config,
                    "discuss: --discuss-on-debater-failure=partial requires at least two "
                    f"successful debater votes in round {round_number}, got {len(votes_by_name)}",
                )
                raise next(iter(failures_by_name.values())).error
            log(
                config,
                f"discuss: continuing round {round_number} with partial results; failed "
                "debater(s): "
                + ", ".join(f"{name} ({category})" for name, category in failed_debaters),
            )
        successful_votes = [
            vote for vote in reviewer_votes
            if not is_failed_discuss_response(vote)
        ]
        round_history.append(reviewer_votes)
        if config.discuss_result_mode == "answer":
            answer_votes = [vote for vote in successful_votes if isinstance(vote, ParsedDiscussAnswer)]
            consensus = _detect_discuss_answer_consensus(answer_votes, partial=bool(failed_debaters))
        else:
            consensus = _detect_discuss_consensus(reviewer_votes)  # type: ignore[arg-type]
        is_final = consensus is not None or round_number == max_round_number
        # The completed current round is authoritative: a later round can
        # clear/reclassify earlier items. At the final round, classified
        # material selects a fail-closed outcome before text comparison.
        if (
            is_final and config.discuss_result_mode == "answer" and not failed_debaters
            and len(answer_votes) == len(configured_reviewers)
        ):
            material_outcome = _final_discuss_answer_item_outcome(answer_votes)
            if material_outcome is not None:
                consensus = (material_outcome, [])
        semantic_comparison: dict[str, object] | None = None
        semantic_finalization_ran = False
        # Keep normalized equality as the zero-call fast path.  Only a complete,
        # final all-answer round is eligible for the configured independent analyzer.
        if (
            is_final and consensus is None and config.discuss_result_mode == "answer"
            and not failed_debaters
            and len(successful_votes) == len(configured_reviewers)
            and all(isinstance(vote, ParsedDiscussAnswer) and vote.position == "answer" and vote.answer for vote in successful_votes)
            and not _discuss_has_material_items(
                [vote for vote in successful_votes if isinstance(vote, ParsedDiscussAnswer)]
            )
        ):
            semantic_finalization_ran = True
            outcome, semantic_kind, semantic_comparison = _run_discuss_semantic_finalization(
                runner, issue_number=issue_number, config=config,
                answers=[vote for vote in successful_votes if isinstance(vote, ParsedDiscussAnswer)],
                configured_reviewers=configured_reviewers, usage_context=usage_context,
            )
            consensus_kind = semantic_kind
            consensus = (outcome, []) if outcome == "answer" else None
        if consensus is None:
            outcome = "deadlock" if config.discuss_result_mode == "answer" else "needs-human"
            round_split_proposals: list[str] = (
                [] if config.discuss_result_mode == "answer" else _merge_discuss_split_proposals(successful_votes)  # type: ignore[arg-type]
            )
            consensus_kind = None if not is_final else (
                semantic_kind if semantic_finalization_ran else "deadlock"
            )
        else:
            outcome, round_split_proposals = consensus
            if semantic_comparison is None:
                consensus_kind = ("unanimous" if len(round_history) == 1 else "converged") if is_final else None
        next_analyzer_agenda: ParsedDiscussAgenda | None = None
        analyzer_response_raw: str | None = None
        final_analyzer_agenda: ParsedDiscussAgenda | None = None
        final_analyzer_response_raw: str | None = None
        if not is_final and analyzer is not None:
            next_analyzer_agenda, analyzer_response_raw = _run_discuss_analyzer(
                runner,
                issue_number=issue_number,
                config=config,
                analyzer=analyzer,
                memory=memory,
                issue_context=prompt_issue_context,
                round_number=round_number,
                round_history=round_history,
                prior_agenda=prior_analyzer_agenda,
                configured_reviewers=configured_reviewers,
                usage_context=usage_context,
            )
        elif is_final and consensus_kind != "debater-confirmed":
            # Semantic finalization already used the configured analyzer to
            # compare the completed final-round answers. When every debater
            # confirms that recommendation, keep that audit as the final
            # analyzer record rather than invoking the same analyzer again
            # for advisory observations over identical input.
            final_analyzer_agenda, final_analyzer_response_raw = _run_discuss_final_analyzer(
                runner,
                issue_number=issue_number,
                config=config,
                analyzer=analyzer,
                round_number=round_number,
                final_votes=successful_votes,
                configured_reviewers=configured_reviewers,
                usage_context=usage_context,
            )
        evidence_reconciliation = None
        if is_final:
            evidence_groups, evidence_reconciler_raw = _run_discuss_evidence_reconciler(
                runner, issue_number=issue_number, config=config, analyzer=analyzer, subject=f"issue-{issue_number}",
                round_history=round_history, usage_context=usage_context,
            )
            evidence_reconciliation = reconcile_evidence(f"issue-{issue_number}", round_history, evidence_groups)
            evidence_reconciliation["raw_evidence_reconciler_response"] = evidence_reconciler_raw
        summary_body = render_discuss_round_summary_comment(
            round_number=round_number,
            # The vote table and agenda draw from real positions only; failed
            # debaters surface in the dedicated failures section instead.
            reviewer_votes=successful_votes,
            is_final=is_final,
            subject=subject,
            outcome=outcome if is_final else None,
            consensus_kind=consensus_kind,
            round_history=round_history if is_final else None,
            split_proposals=round_split_proposals,
            analyzer_agenda=next_analyzer_agenda,
            prior_analyzer_agenda=prior_analyzer_agenda if is_final else None,
            final_analyzer_agenda=final_analyzer_agenda,
            analyzer_name=analyzer_name,
            research_mode=config.discuss_research,
            failed_debaters=tuple(failed_debaters),
            result_mode=config.discuss_result_mode,
            semantic_comparison=semantic_comparison,
            evidence_reconciliation=evidence_reconciliation,
        )
        agenda = () if is_final else tuple(_render_discuss_agenda_lines(successful_votes))
        post_issue_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=_attach_round_metadata(
                summary_body,
                PostedRoundMetadata(
                    flow="discuss",
                    role="summary",
                    agent="Orchestrator",
                    round_number=round_number,
                    subject=subject,
                    is_final=is_final,
                    consensus_kind=consensus_kind,
                    agenda=agenda,
                    analyzer_response=analyzer_response_raw,
                    final_analyzer_response=final_analyzer_response_raw,
                    research_mode=config.discuss_research,
                    failed_debaters=tuple(failed_debaters),
                    split_proposals=tuple(round_split_proposals) if is_final else (),
                    result_mode=config.discuss_result_mode,
                    evidence_reconciliation=evidence_reconciliation,
                ),
            ),
        )
        if is_final:
            log(
                config,
                f"discuss: posted final summary comment for issue #{issue_number} "
                f"(outcome: {outcome}; kind: {consensus_kind})",
            )
            if outcome == "split":
                _handle_discuss_split_outcome(
                    runner,
                    issue_number=issue_number,
                    config=config,
                    subject=subject,
                    split_proposals=round_split_proposals,
                    final_votes=successful_votes,
                    issue_comments=issue_context.comments,
                    post_warning_comment=True,
                )
            return 0
        log(
            config,
            f"discuss: posted round {round_number} summary comment for issue #{issue_number}; "
            f"continuing to round {round_number + 1}",
        )
        prior_round_agenda = list(agenda)
        prior_analyzer_agenda = next_analyzer_agenda
    return 0


def run_discuss_loop(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    discuss_max_rounds: int = 2,
    usage_context: RunUsageContext | None = None,
) -> int:
    owned_usage_context = usage_context is None
    usage_context = usage_context or _new_usage_context(config)
    try:
        if config.discuss_parallel:
            _ensure_parallel_discuss_workdirs(config)
        config = resolve_base_branch(config, runner)
        ensure_agent_workdirs(config, runner)
        log(config, f"discuss: validating issue #{issue_number}")
        validate_open_issue(runner, config=config, issue_number=issue_number)
        return _run_discuss_loop(
            runner,
            issue_number=issue_number,
            config=config,
            usage_context=usage_context,
            discuss_max_rounds=discuss_max_rounds,
        )
    finally:
        if owned_usage_context:
            _persist_usage_summary(config, usage_context)
