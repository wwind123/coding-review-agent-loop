"""High-level issue, task, and PR orchestration loops."""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import sys
import zoneinfo
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace as dataclasses_replace
from pathlib import Path

from .agents.base import AgentName, AgentResult
from .agents.registry import agent_display_name, get_backend, run_agent_result
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
    CreatedPhaseIssue,
    RecordedPhase,
    approved_plan_hash,
    create_decomposition_child_issues,
    find_existing_decomposition,
    find_existing_one_shot_impl_handoff,
    find_existing_phase_implementation_handoff,
    parse_plan_decomposition,
    post_decomposition_parent_summary,
    post_one_shot_impl_handoff_comment,
    post_phase_implementation_handoff_comment,
)
from .errors import (
    AgentInvocationError,
    AgentLoopError,
    QuotaResetExceededError,
    UnknownPriorItemDispositionError,
)
from .github import (
    IssueContext,
    PullRequestChecks,
    PullRequestReviewContext,
    find_open_pr_referencing_issue,
    get_issue_context,
    parse_linked_issue_numbers,
    get_pr_checks,
    get_pr_review_context,
    get_pr_state,
    merge_pr,
    post_issue_comment,
    post_pr_comment,
    validate_open_issue,
    validate_open_pr,
    validate_pr_body_does_not_close_issue,
    validate_pr_references_issue,
    wait_for_ci,
)
from .split_materialization import (
    DISCUSS_SPLIT_MARKER_RE,
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
    build_followup_prompt,
    build_issue_implementation_prompt,
    build_issue_plan_prompt,
    build_issue_prompt,
    build_plan_decomposition_prompt,
    build_plan_review_prompt,
    build_plan_revision_prompt,
    build_review_prompt,
    build_same_pr_followup_prompt,
    build_task_clarification_prompt,
    build_task_prompt,
    format_agent_list,
    render_coder_human_requirements_prompt_context,
)
from .protocol import (
    DISCUSS_FAILED_OUTCOME,
    DISCUSS_RESEARCH_TARGET_VALUES,
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
    ParsedReview,
    PUBLIC_RESPONSE_MARKER,
    ReviewItemDisposition,
    StructuredCoderFollowup,
    StructuredPlanState,
    StructuredPlanRevision,
    UnresolvedReviewItem,
    human_requirements_resolved,
    is_clarification_request,
    parse_human_requirements_acknowledgement,
    parse_agent_state,
    parse_plan_review,
    parse_plan_review_items,
    parse_plan_state,
    parse_structured_plan_review,
    parse_pr_number,
    review_freeform_summary_text,
    normalize_response_file_structured_text,
    validate_human_requirements_acknowledgement,
    validate_structured_coder_followup,
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
    is_transient_agent_output,
)
from .usage import RunUsageContext, UsageMetadata, estimate_usage
from .workdirs import active_workdir
from .workdir_guard import (
    validate_assigned_head_advanced,
    validate_checkout_inspected_evidence,
    validate_response_tests_within_workdir,
    validate_test_commands_within_workdir,
)
from .checks import (
    _format_pr_checks_comment,
    _pending_ci_status_summary,
    _pending_ci_stop_guidance,
    _pending_ci_stop_message,
    _pr_check_blocking_review,
    _pr_check_details,
    run_optional_tests,
    run_pre_review_tests,
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
    render_canonical_plan_revision,
    render_canonical_plan_steps,
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
from .unresolved_items import (
    ALL_RESOLVED_PROSE_RE,
    CODER_DISPUTE_NOTE_PREFIX,
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    _apply_dispute_evidence,
    _apply_unresolved_item_dispositions,
    _collect_prior_compact_summaries,
    _clear_human_requirements_ack_item,
    _format_same_pr_unresolved_items,
    _format_unresolved_items_for_coder,
    _is_disputed_item,
    _maybe_fill_resolved_dispositions_from_prose,
    _next_unresolved_item,
    _normalize_disposition_section_prose,
    _record_prior_item_disposition,
    _reconcile_human_requirements_ack_item,
    _upsert_human_requirements_ack_item,
    _validate_coder_followup_response,
    _validate_plan_review_response,
    _validate_review_response,
    _validate_structured_coder_followup_items,
)


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
    {"plan_state", "plan_review", "pr_review", "coder_followup", "plan_revision", "discuss_review", "discuss_answer", "discuss_semantic_comparison", "discuss_answer_confirmation"}
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
class ValidatedAgentResponse:
    text: str
    session_id: str | None
    marker_value: object
    usage: UsageMetadata | None = None
    # Model the agent actually ran, for the dynamic signature (#332). Carried from
    # AgentResult.model_used so the orchestrator render sites can stamp it.
    model_used: str | None = None


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

    valid_blocks: list[tuple[str, str]] = []
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
            except AgentLoopError:
                invalid_count += 1
                continue
            valid_blocks.append((source, block))

    unique_valid: list[tuple[str, str]] = []
    seen_blocks: set[str] = set()
    for source, block in valid_blocks:
        if block in seen_blocks:
            continue
        seen_blocks.add(block)
        unique_valid.append((source, block))

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
    source, block = unique_valid[0]
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
        repaired = attempt_repair(raw, config.gemini_cmd, **repair_kwargs)
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
    repair_requires_direct_discussion_ack: bool = False,
    repair_allowed_prior_item_ids: Sequence[str] | None = None,
    ledger_incomplete: bool = False,
    role: str | None = None,
    label: str | None = None,
    timeout_seconds: float | None = None,
    salvage_context: SalvageContext | None = None,
    operation_description: str | None = None,
) -> ValidatedAgentResponse:
    agent_name = agent_display_name(agent)
    operation_description = operation_description or _operation_description_from_context(
        salvage_context=salvage_context,
        repair_expected_kind=repair_expected_kind,
        role=role,
        label=label,
        marker_description=marker_description,
    )
    log_paths: list[object] = []
    max_attempts = config.agent_max_retries + 1
    last_error = f"{agent_name} produced no output."
    last_result: AgentResult | None = None
    last_classification_text = ""
    last_failure_category = "empty-response"

    for attempt in range(1, max_attempts + 1):
        result = run_agent_result(
            runner,
            agent=agent,
            config=config,
            prompt=prompt,
            session_id=session_id,
            run_id=usage_context.run_id if usage_context is not None else None,
            role=role,
            label=label,
            timeout_seconds=timeout_seconds,
        )
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

        should_retry = False
        if result.returncode is None:
            # Timed out (returncode=None from Runner.run_with_log). Detected
            # before transient classification: a kill deadline is not a
            # provider hiccup, so retrying or repairing would only waste the
            # same wall-clock budget again (#475).
            limit = f" after {timeout_seconds:g}s" if timeout_seconds is not None else ""
            last_error = f"agent command timed out{limit}"
            last_classification_text = ""
            last_failure_category = "timeout"
            break
        if result.returncode != 0:
            last_error = f"agent command exited with {result.returncode}"
            classification_text = _agent_failure_classification_text(result, phase="command")
            last_classification_text = classification_text
            should_retry = _is_transient_agent_output(classification_text)
            last_failure_category = _failure_category(classification_text)
        elif not text.strip():
            last_error = "agent response was empty"
            classification_text = _agent_failure_classification_text(result, phase="empty")
            last_classification_text = classification_text
            should_retry = _is_transient_agent_output(classification_text)
            last_failure_category = _failure_category(classification_text)
        else:
            response_file_pre_status = (
                _response_file_structured_status(result.response_file_text)
                if result.response_file_text
                else None
            )
            try:
                marker_value = validate(text)
                if response_file_pre_status == "leading-public-response-marker-recovered":
                    log(
                        config,
                        f"{agent_name}: response file contained stdout filtering marker and "
                        "validated after stripping it",
                    )
            except AgentLoopError as exc:
                last_error = str(exc)
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
                response_failure_is_unsupported = last_failure_category == "unsupported_model"
                # Marker near-misses are a separate first-attempt nudge for common footer typos;
                # structured JSON protocol drift still remains repairable when retries are exhausted.
                should_retry = public_text_is_transient or (
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
                    and not (
                        isinstance(exc, UnknownPriorItemDispositionError)
                        and ledger_incomplete
                    )
                ):
                    log(config, f"{agent_name}: schema validation failed ({exc}); attempting repair pass")
                    repair_kwargs: dict[str, object] = {"expected_kind": repair_expected_kind}
                    if repair_unresolved_item_ids is not None:
                        repair_kwargs["unresolved_item_ids"] = tuple(repair_unresolved_item_ids)
                    if (
                        repair_expected_kind == "coder_followup"
                        and repair_surfaced_requirement_ids is not None
                    ):
                        repair_kwargs["surfaced_requirement_ids"] = tuple(repair_surfaced_requirement_ids)
                    if isinstance(exc, UnknownPriorItemDispositionError):
                        repair_kwargs["allowed_prior_item_ids"] = exc.allowed_ids
                        repair_kwargs["unknown_prior_item_ids"] = exc.unknown_ids
                        repair_kwargs["same_round_context"] = exc.same_round_description
                    elif repair_allowed_prior_item_ids is not None:
                        repair_kwargs["allowed_prior_item_ids"] = tuple(repair_allowed_prior_item_ids)
                    original_validation_error = str(exc)
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
                    diagnostics = _failed_run_diagnostics(
                        runner=runner,
                        config=config,
                        agent_name=agent_name,
                        salvage_context=salvage_context,
                        operation_description=operation_description,
                        failure_category=last_failure_category,
                        failure_reason=message,
                        classification_text=classification_text,
                        marker_description=marker_description,
                        result=last_result,
                    )
                    message += diagnostics.format_for_error()
                    raise QuotaResetExceededError(message)
            if attempt < max_attempts:
                delay = _retry_delay(config, attempt)
                category = last_failure_category
                log(
                    config,
                    f"{agent_name}: {category} failure ({last_error}); "
                    f"retrying in {delay}s (attempt {attempt + 1}/{max_attempts})",
                )
                runner.run(("sleep", str(delay)), cwd=active_workdir(config))
                continue
        break

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
    raise AgentInvocationError(message, failure_category=last_failure_category)


def _require_pr_number(text: str) -> int:
    pr_number = parse_pr_number(text)
    if pr_number is None:
        raise AgentLoopError("Agent response did not include a PR marker or PR URL.")
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


def _require_plan_state_or_clarification(text: str) -> str:
    if is_clarification_request(text):
        return "clarification"
    structured_plan = validate_structured_plan_state(text)
    if structured_plan is None:
        raise AgentLoopError(
            "Initial planning response must include a structured `plan_state` JSON object."
        )
    return structured_plan.state
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
    return marker_value


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


def _should_record_new_blocking_item(summary: str, *, had_prior_items: bool, had_dispositions: bool) -> bool:
    if not summary:
        return False
    if summary.strip() in {"Review complete.", "Plan review complete."}:
        return False
    if not had_prior_items or not had_dispositions:
        return True
    non_empty_lines = [line.strip() for line in summary.splitlines() if line.strip()]
    if len(non_empty_lines) > 1:
        return True
    return len(non_empty_lines[0]) >= 80


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
    prior_discuss_proposals = _prior_discuss_split_proposals(issue_context, config=config)
    if current_deferred_stages:
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
        [split_stage_proposal_from_deferred_stage(stage) for stage in current_deferred_stages]
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
    result = runner.run(
        ("git", "rev-parse", "HEAD"),
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _validate_response_tests_with_post_pr_context(
    text: str,
    *,
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
) -> None:
    try:
        validate_response_tests_within_workdir(text, assigned_workdir=active_workdir(config))
    except AgentLoopError as exc:
        try:
            validate_open_pr(runner, config=config, pr_number=pr_number)
        except Exception as pr_exc:
            raise AgentLoopError(
                f"{exc}\n\n"
                f"The coder reported PR #{pr_number}, but the orchestrator could not confirm it is open. "
                "The handoff/reviewer comments were not posted because the test report was invalid. "
                "Inspect the PR state on GitHub before deciding whether to resume the existing PR or rerun "
                "implementation."
            ) from pr_exc
        raise AgentLoopError(
            f"{exc}\n\n"
            f"PR #{pr_number} was confirmed open, but the handoff/reviewer comments were not posted because "
            "the test report was invalid. Correct the PR/comment if needed, then continue safely with "
            f"`agent-loop pr {pr_number}` instead of rerunning implementation and creating a duplicate PR."
        ) from exc


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

    # A prior implementation attempt may have created a PR and then aborted
    # before recording any handoff marker/comment (e.g. the #493 test-report
    # false positive, which produced the duplicate PR #494 for #492). Check
    # GitHub directly for an already-open PR referencing this issue before
    # invoking the coder again (#495).
    existing_pr_number = find_open_pr_referencing_issue(
        runner, config=config, issue_number=issue_number
    )
    if existing_pr_number is not None:
        log(
            config,
            f"Existing implementation PR #{existing_pr_number} found for issue #{issue_number} "
            f"/ approved plan {plan_hash}; resuming PR review instead of invoking {coder_name}.",
        )
        validate_pr_references_issue(
            runner,
            config=implementation_config,
            pr_number=existing_pr_number,
            issue_number=issue_number,
        )
        if staged_parent_issue is not None:
            validate_pr_body_does_not_close_issue(
                runner,
                config=implementation_config,
                pr_number=existing_pr_number,
                issue_number=staged_parent_issue,
            )
        if one_shot_parent_issue is not None:
            resumed_pr_context = get_pr_review_context(
                runner, config=implementation_config, pr_number=existing_pr_number
            )
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
        ),
        session_id=implementation_session_id,
        marker_description="<!-- AGENT_PR: <number> --> or PR URL",
        validate=lambda text, human_requirements=issue_context.human_requirements: _validate_response_with_human_requirements(
            text,
            marker_validator=_require_pr_number,
            human_requirements=human_requirements,
            requirement_scope="implementation requirements",
            full_omission_fallback="Fetch the issue discussion directly before implementing.",
        ),
        usage_context=usage_context,
        salvage_context=SalvageContext(
            repo=implementation_config.repo,
            issue_number=issue_number,
            scope=APPROVED_PLAN_IMPLEMENTATION_SALVAGE_SCOPE,
            agent=implementation_config.coder,
            run_id=usage_context.run_id,
            approved_plan_hash=plan_hash,
        ),
        operation_description="approved-plan implementation",
    )
    coder_output = coder_response.text
    pr_number = int(coder_response.marker_value)
    _validate_response_tests_with_post_pr_context(
        coder_output,
        runner=runner,
        config=implementation_config,
        pr_number=pr_number,
    )
    validate_assigned_head_advanced(
        before_head=assigned_head_before,
        after_head=_read_assigned_workdir_head(runner, implementation_config),
        assigned_workdir=active_workdir(implementation_config),
    )
    log(config, f"{coder_name} reported PR #{pr_number}; validating it is open")
    validate_open_pr(runner, config=implementation_config, pr_number=pr_number)
    validate_pr_references_issue(
        runner,
        config=implementation_config,
        pr_number=pr_number,
        issue_number=issue_number,
    )
    if staged_parent_issue is not None:
        validate_pr_body_does_not_close_issue(
            runner,
            config=implementation_config,
            pr_number=pr_number,
            issue_number=staged_parent_issue,
        )
    initial_pr_context = get_pr_review_context(runner, config=implementation_config, pr_number=pr_number)
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
    post_pr_comment(
        runner,
        config=implementation_config,
        pr_number=pr_number,
        body=_attach_round_metadata(
            normalize_freeform_signature(
                coder_output,
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
                model_used=coder_response.model_used,
            ),
        ),
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
                ),
            ),
        )
        start_round_number = 1
        resumed_round: ResumedReviewRound | None = None
    else:
        current_plan, resumed_round = resume_state
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
        for reviewer in configured_reviewers:
            reviewer_name = agent_display_name(reviewer)
            resumed_record = resumed_by_name.get(reviewer_name)
            if resumed_record is not None:
                review_output = resumed_record.body
                review_model_used = resumed_record.metadata.model_used
                structured_review = parse_structured_plan_review(
                    review_output,
                    reviewer=reviewer_name,
                )
                parsed_review = ParsedPlanReview(
                    state=resumed_record.metadata.state or parse_plan_state(review_output),
                    summary=review_freeform_summary_text(review_output),
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
                    prompt=build_plan_review_prompt(
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
                    ),
                    session_id=reviewer_session_ids.get(reviewer),
                    marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                    validate=lambda text, reviewer_name=reviewer_name, items=prior_unresolved_items: _validate_plan_review_response(
                        text,
                        reviewer=reviewer_name,
                        unresolved_items=items,
                        current_round_items=round_new_unresolved_items,
                    ),
                    usage_context=usage_context,
                    use_repair=True,
                    repair_expected_kind="plan_review",
                    repair_allowed_prior_item_ids=tuple(item.item_id for item in prior_unresolved_items),
                    ledger_incomplete=round_ledger_incomplete,
                    role="reviewer",
                    operation_description="plan review",
                )
                review_output = review_response.text
                review_model_used = review_response.model_used
                reviewer_session_ids[reviewer] = review_response.session_id
                parsed_review = review_response.marker_value
                assert isinstance(parsed_review, ParsedPlanReview)
                review_state = parsed_review.state
                reviewer_new_unresolved_items = []
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
            if resumed_record is None:
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
                post_issue_comment(
                    runner,
                    config=config,
                    issue_number=issue_number,
                    body=_attach_round_metadata(
                        render_public_agent_comment(
                            kind="plan_review",
                            parsed=parsed_review,
                            agent=reviewer_name,
                            prior_items=prior_unresolved_items,
                            dispositions=parsed_review.dispositions,
                            human_requirements_resolved_flag=human_requirements_resolved(
                                review_output
                            ),
                            config=config,
                            model_used=review_model_used,
                        ),
                        PostedRoundMetadata(
                            flow="plan",
                            role="reviewer",
                            agent=reviewer_name,
                            round_number=round_number,
                            subject=_plan_subject(current_plan),
                            prior_items=prior_unresolved_items,
                            dispositions=parsed_review.dispositions,
                            new_items=tuple(reviewer_new_unresolved_items),
                            state=review_state,
                            compact_prior_summaries=tuple(compact_prior_summaries),
                            model_used=review_model_used,
                        ),
                    ),
                )
            else:
                round_new_unresolved_items.extend(reviewer_new_unresolved_items)

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
            missing_acknowledgements = [
                reviewer_name
                for reviewer_name, review_output in approved_review_outputs
                if not human_requirements_resolved(review_output)
            ]
            if missing_acknowledgements:
                hr_ids = _surfaced_reviewer_requirement_ids(
                    issue_context.human_requirements,
                    requirement_scope="planning requirements",
                )
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
                post_phase_implementation_handoff_comment(
                    runner,
                    config=config,
                    parent_issue=issue_number,
                    mode=mode,
                    plan_hash=plan_hash,
                    phase_index=1,
                    created=first_agent_phase,
                )
                child_issue_context = get_issue_context(
                    runner,
                    config=config,
                    issue_number=first_agent_phase.issue_number,
                )
                phase_parent_context = first_agent_phase.phase.parent_context or current_plan
                return _implement_approved_issue(
                    runner,
                    issue_number=first_agent_phase.issue_number,
                    approved_plan=phase_parent_context,
                    config=config,
                    memory=memory,
                    issue_context=child_issue_context,
                    coder_session_id=coder_session_id,
                    usage_context=usage_context,
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
                current_plan_declares_own_deferred_stages = bool(
                    _extract_current_deferred_stages(current_plan)
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

                existing_handoff = find_existing_one_shot_impl_handoff(
                    target_issue_context.comments,
                    parent_issue=target_issue_number,
                    plan_hash=plan_hash,
                    mode="implement-one-shot",
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
                        validate_pr_references_issue(
                            runner,
                            config=config,
                            pr_number=existing_handoff.pr_number,
                            issue_number=target_issue_number,
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

        sync_coder_base_before_implementation(config, runner)
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
            ),
            marker_description="<!-- AGENT_PR: <number> --> or PR URL",
            validate=lambda text, human_requirements=issue_context.human_requirements: _validate_response_with_human_requirements(
                text,
                marker_validator=_require_pr_number,
                human_requirements=human_requirements,
                requirement_scope="implementation requirements",
                full_omission_fallback="Fetch the issue discussion directly before implementing.",
            ),
            usage_context=usage_context,
            salvage_context=SalvageContext(
                repo=config.repo,
                issue_number=issue_number,
                scope=ISSUE_IMPLEMENTATION_SALVAGE_SCOPE,
                agent=config.coder,
                run_id=usage_context.run_id,
            ),
            operation_description="issue implementation",
        )
        coder_output = coder_response.text
        coder_session_id = coder_response.session_id
        pr_number = int(coder_response.marker_value)
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
        log(config, f"{agent_display_name(config.coder)} reported PR #{pr_number}; validating it is open")
        validate_open_pr(runner, config=config, pr_number=pr_number)
        validate_pr_references_issue(
            runner,
            config=config,
            pr_number=pr_number,
            issue_number=issue_number,
        )
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
                    agent=agent_display_name(config.coder),
                    round_number=1,
                    subject=str(initial_pr_metadata.head_sha or "unknown"),
                    prior_items=(),
                    model_used=coder_response.model_used,
                ),
            ),
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
                marker_description="<!-- AGENT_PR: <number> -->, PR URL, or <!-- AGENT_CLARIFY -->",
                validate=_require_pr_number_or_clarification,
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
        parsed = validate_structured_coder_followup(text)
        return parsed.summary if parsed else None
    except AgentLoopError:
        return None


def _extract_structured_coder_tests_run(text: str | None) -> tuple[str, ...] | None:
    if not text:
        return None
    try:
        parsed = validate_structured_coder_followup(text)
        return parsed.tests_run if parsed else None
    except AgentLoopError:
        return None


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
) -> int:
    owned_usage_context = usage_context is None
    usage_context = usage_context or _new_usage_context(config)
    try:
        bootstrap_cwd = github_bootstrap_cwd(config)
        initial_pr_context = get_pr_review_context(
            runner,
            config=config,
            pr_number=pr_number,
            cwd=bootstrap_cwd,
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
        config = resolve_base_branch(
            config,
            runner,
            pr_metadata=initial_pr_context.metadata,
            cwd=bootstrap_cwd,
        )
        if not workdirs_ready:
            ensure_agent_workdirs(config, runner)
        log(config, f"Validating PR #{pr_number}")
        validate_open_pr(runner, config=config, pr_number=pr_number)
        memory = prepare_agent_memory(runner, config)
        reviewer_session_ids: dict[AgentName, str | None] = {}
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
        for round_number in range(start_round_number, config.max_rounds + 1):
            coder_name = agent_display_name(config.coder)
            if pre_review_test_pending:
                run_pre_review_tests(runner, config)
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
            human_requirements = _merge_human_requirements(issue_context, pr_context)
            current_resume = resumed_round if resumed_round is not None and round_number == resumed_round.round_number else None
            unresolved_items = _reconcile_human_requirements_ack_item(
                current_resume.prior_items if current_resume is not None else unresolved_items,
                coder_output=latest_coder_output,
                human_requirements=human_requirements,
                source_round=round_number,
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
            approved_review_outputs: list[tuple[str, str]] = []
            pr_checks = get_pr_checks(runner, config=config, metadata=pr_metadata)
            resumed_by_name = {
                record.metadata.agent: record for record in (current_resume.completed_reviews if current_resume is not None else ())
            }
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
            for reviewer in (() if skip_reviewers_for_recovery else configured_reviewers):
                reviewer_name = agent_display_name(reviewer)
                resumed_record = resumed_by_name.get(reviewer_name)
                if resumed_record is not None:
                    review_output = resumed_record.body
                    review_model_used = resumed_record.metadata.model_used
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
                else:
                    context_mode = "compact" if use_compact_pr_context else "full"
                    log(
                        config,
                        f"Round {round_number}: {reviewer_name} reviewing PR #{pr_number} "
                        f"(context mode: {context_mode})",
                    )
                    sync_reviewer_pr_before_review(config, runner, reviewer, pr_number, pr_metadata)
                    compact_tail = (
                        CompactPrReviewTailContext(
                            head_sha=pr_metadata.head_sha,
                            round_number=round_number,
                        )
                        if use_compact_pr_context
                        else None
                    )
                    review_response = _run_validated_agent(
                        runner,
                        agent=reviewer,
                        config=config,
                        prompt=build_review_prompt(
                            pr_number,
                            round_number,
                            config,
                            reviewer=reviewer,
                            pr_metadata=pr_metadata,
                            pr_checks=pr_checks,
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
                        session_id=None if use_compact_pr_context else reviewer_session_ids.get(reviewer),
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
                        repair_allowed_prior_item_ids=tuple(item.item_id for item in prior_unresolved_items),
                        ledger_incomplete=round_ledger_incomplete,
                        role="reviewer",
                        operation_description="PR review",
                    )
                    review_output = review_response.text
                    review_model_used = review_response.model_used
                    reviewer_session_ids[reviewer] = review_response.session_id
                    parsed_review = review_response.marker_value
                    assert isinstance(parsed_review, ParsedReview)
                    review_state = parsed_review.state
                    reviewer_new_unresolved_items = []

                if (
                    resumed_record is None
                    and review_state == "blocking"
                    and _is_pending_ci_only_review(parsed_review, pr_checks)
                ):
                    log(
                        config,
                        f"Round {round_number}: {reviewer_name} blocking review only restates "
                        f"GitHub check status ({pr_checks.state}); treating as approved instead "
                        "of starting a new coder follow-up round",
                    )
                    parsed_review = dataclasses_replace(parsed_review, state="approved", blocking_items=())
                    review_state = parsed_review.state

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
                has_blocking_summary = not has_structured_blocking_content and _should_record_new_blocking_item(
                    blocking_summary,
                    had_prior_items=bool(prior_unresolved_items),
                    had_dispositions=bool(parsed_review.dispositions),
                )
                log(
                    config,
                    f"Round {round_number}: {reviewer_name} outcome is "
                    f"{_describe_pr_review_outcome(parsed_review, has_blocking_summary=has_blocking_summary)}",
                )
                if review_state == "blocking":
                    if resumed_record is None:
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
                        post_pr_comment(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            body=_attach_round_metadata(
                                render_public_agent_comment(
                                    kind="pr_review",
                                    parsed=parsed_review,
                                    agent=reviewer_name,
                                    human_requirements_resolved_flag=human_requirements_resolved(
                                        review_output
                                    ),
                                    prior_items=prior_unresolved_items,
                                    dispositions=parsed_review.dispositions,
                                    config=config,
                                    model_used=review_model_used,
                                ),
                                PostedRoundMetadata(
                                    flow="pr",
                                    role="reviewer",
                                    agent=reviewer_name,
                                    round_number=round_number,
                                    subject=current_pr_subject,
                                    prior_items=prior_unresolved_items,
                                    dispositions=parsed_review.dispositions,
                                    new_items=tuple(reviewer_new_unresolved_items),
                                    state=review_state,
                                    model_used=review_model_used,
                                ),
                            ),
                        )
                    else:
                        round_new_unresolved_items.extend(reviewer_new_unresolved_items)
                    continue

                approved_review_outputs.append((reviewer_name, review_output))
                if resumed_record is None:
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
                    post_pr_comment(
                        runner,
                        config=config,
                        pr_number=pr_number,
                        body=_attach_round_metadata(
                            render_public_agent_comment(
                                kind="pr_review",
                                parsed=parsed_review,
                                agent=reviewer_name,
                                human_requirements_resolved_flag=human_requirements_resolved(
                                    review_output
                                ),
                                prior_items=prior_unresolved_items,
                                dispositions=parsed_review.dispositions,
                                config=config,
                                model_used=review_model_used,
                            ),
                            PostedRoundMetadata(
                                flow="pr",
                                role="reviewer",
                                agent=reviewer_name,
                                round_number=round_number,
                                    subject=current_pr_subject,
                                prior_items=prior_unresolved_items,
                                dispositions=parsed_review.dispositions,
                                new_items=tuple(reviewer_new_unresolved_items),
                                state=review_state,
                                model_used=review_model_used,
                            ),
                        ),
                    )
                else:
                    round_new_unresolved_items.extend(reviewer_new_unresolved_items)

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
                        f"- [{item.item_id}] from {item.reviewer}: {item.text[:300]}"
                        for item in disputed_still_blocking
                    )
                    raise AgentLoopError(
                        f"Reviewer did not resolve {len(disputed_still_blocking)} disputed item(s) "
                        "after seeing coder counter-evidence. Human review required to resolve the "
                        f"disagreement.\n\nDisputed items still unresolved:\n{item_summaries}"
                    )

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
                if not must_fix_items:
                    if pr_checks.state in {"pending", "unavailable"}:
                        details = _pr_check_details(pr_checks)
                        if not config.auto_merge:
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
                        # --auto-merge: post the informational comment, then fall
                        # through to wait for CI before merging, as today.
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
                    if config.auto_merge:
                        wait_for_ci(runner, config, pr_number)
                        merge_pr(runner, config, pr_number)
                    print(f"PR #{pr_number} approved by {format_agent_list(configured_reviewers)}.")
                    return 0
            if round_number == config.max_rounds:
                raise AgentLoopError(
                    f"One or more reviewers still reported blocking issues after round {round_number}; "
                    "human review required."
                )

            same_pr_items = [item for item in unresolved_items if item.status == "same-pr"]
            blocking_items = [item for item in unresolved_items if item.status == "blocking"]
            if same_pr_items and not blocking_items:
                combined_review = _format_same_pr_unresolved_items(same_pr_items)
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
            else:
                combined_review = _format_unresolved_items_for_coder(unresolved_items)
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
                if item.item_id != HUMAN_REQUIREMENTS_ACK_ITEM_ID
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
        if owned_usage_context:
            _persist_usage_summary(config, usage_context)


DISCUSS_CONSENSUS_MARKER_RE = re.compile(
    r"<!--\s*AGENT_DISCUSS_CONSENSUS:\s*([0-9a-f]+)\s*-->",
    re.I,
)


def _is_bot_authored_discuss_comment(body: str) -> bool:
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


def _ensure_parallel_discuss_workdirs(config: AgentLoopConfig) -> None:
    """Reject shared workdirs among concurrently scheduled debaters (#475).

    Deliberately NOT bypassed by --allow-shared-dir: concurrent git/tool
    activity from two agents in one worktree can race and corrupt it. The
    analyzer (or the coder) may still share a debater's directory because it
    runs only after the debater synchronization point.
    """
    from .config import reviewers as _reviewers

    seen: dict[Path, AgentName] = {}
    for reviewer in _reviewers(config):
        path = get_backend(reviewer).workdir(config).resolve()
        other = seen.get(path)
        if other is not None:
            raise AgentLoopError(
                "--discuss-parallel requires a distinct workdir per debater: "
                f"{agent_display_name(other)} and {agent_display_name(reviewer)} "
                f"both resolve to {path}. Use separate clones/worktrees per debater "
                "or drop --discuss-parallel; --allow-shared-dir does not lift this "
                "requirement."
            )
        seen[path] = reviewer


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
