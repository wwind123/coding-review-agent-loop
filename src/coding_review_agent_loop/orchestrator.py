"""High-level issue, task, and PR orchestration loops."""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import sys
import zoneinfo
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .agents.base import AgentName, AgentResult
from .agents.registry import agent_display_name, run_agent_result
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
from .errors import AgentLoopError, QuotaResetExceededError, UnknownPriorItemDispositionError
from .github import (
    IssueContext,
    PullRequestReviewContext,
    get_issue_context,
    get_pr_checks,
    get_pr_review_context,
    get_pr_state,
    merge_pr,
    post_issue_comment,
    post_pr_comment,
    validate_open_issue,
    validate_open_pr,
    validate_pr_references_issue,
    wait_for_ci,
)
from .logging import log, new_run_id, run_usage_summary_path
from .memory import prepare_agent_memory
from .migrations import validate_pr_migration_topology
from .prompts import (
    CompactPlanTailContext,
    CompactPriorContext,
    CompactPrReviewTailContext,
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
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
    ParsedDiscussReview,
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
    validate_structured_discuss_review,
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
from .transient import (
    NON_RETRYABLE_AGENT_OUTPUT_RE,
    TRANSIENT_AGENT_OUTPUT_RE,
    is_transient_agent_output,
)
from .usage import RunUsageContext, UsageMetadata, estimate_usage
from .workdirs import active_workdir
from .workdir_guard import (
    validate_assigned_head_advanced,
    validate_response_tests_within_workdir,
    validate_test_commands_within_workdir,
)
from .checks import (
    _format_pr_checks_comment,
    _pr_check_blocking_review,
    _pr_check_details,
    run_optional_tests,
    run_pre_review_tests,
)
from .comment_rendering import (
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
    normalize_freeform_signature,
    render_discuss_consensus_comment,
    render_public_agent_comment,
    render_canonical_plan_revision,
    render_canonical_plan_steps,
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
    _decode_round_metadata,
    _deserialize_disposition,
    _deserialize_unresolved_item,
    _encode_round_metadata,
    _extract_round_metadata_records,
    _max_unresolved_item_number_from_records,
    _plan_subject,
    _prior_item_ledger_signature,
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
    {"plan_review", "pr_review", "coder_followup", "plan_revision", "discuss_review"}
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


def _failure_suggestion(
    category: str | None,
    reason: str,
    agent_name: str,
    *,
    classification_text: str = "",
) -> str:
    """Return a one-line actionable suggestion to append to an agent failure message."""
    combined = f"{reason} {classification_text}"
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


def _format_invalid_agent_response_error(
    *,
    agent_name: str,
    marker_description: str,
    reason: str,
    result: AgentResult | None,
    log_paths: Sequence[object],
    category: str | None = None,
) -> str:
    exit_context = ""
    if result is not None and result.returncode != 0:
        exit_context = f" Agent exit code: {result.returncode}."
    log_context = _agent_log_context(log_paths)
    category_hint = ""
    if category == "transient":
        category_hint = " Failure category: transient (rerun may succeed)."
    elif category == "non-retryable":
        category_hint = " Failure category: non-retryable (check credentials or billing)."
    elif category == "deterministic":
        category_hint = " Failure category: deterministic (may require a code fix)."
    classification_text = (result.raw_output or result.text or "") if result is not None else ""
    suggestion = _failure_suggestion(category, reason, agent_name, classification_text=classification_text)
    suggestion_line = f"\n{suggestion}" if suggestion else ""
    return (
        f"{agent_name} failed before producing a valid public response. "
        "No review result was recorded. "
        f"Required marker: {marker_description}. Reason: {reason}.{exit_context}"
        f"{category_hint}"
        f"{log_context}"
        f"{suggestion_line}"
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
) -> ValidatedAgentResponse:
    agent_name = agent_display_name(agent)
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
                # Marker near-misses are a separate first-attempt nudge for common footer typos;
                # structured JSON protocol drift still remains repairable when retries are exhausted.
                should_retry = public_text_is_transient or (
                    attempt == 1 and _is_retryable_marker_near_miss(classification_text)
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
                if result.response_file_text and response_file_not_structured:
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
                    repair_expected_kind == "plan_revision"
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
                    raise QuotaResetExceededError(
                        f"{agent_name} quota exhausted. Reset in {duration_str} (at {at_str}). "
                        "Rerun when quota resets, or switch to a different API key / model."
                    )
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

    raise AgentLoopError(
        _format_invalid_agent_response_error(
            agent_name=agent_name,
            marker_description=marker_description,
            reason=last_error,
            result=last_result,
            log_paths=log_paths,
            category=last_failure_category,
        )
    )


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
    return parse_plan_state(text)
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
    if has_blocking_summary and has_same_pr:
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


def _read_assigned_workdir_head(runner: Runner, config: AgentLoopConfig) -> str | None:
    result = runner.run(
        ("git", "rev-parse", "HEAD"),
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


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
) -> int:
    coder_name = agent_display_name(config.coder)
    sync_coder_base_before_implementation(config, runner)
    log(config, f"Planning approved; invoking {coder_name} to implement issue #{issue_number}")
    assigned_head_before = _read_assigned_workdir_head(runner, config)
    coder_response = _run_validated_agent(
        runner,
        agent=config.coder,
        config=config,
        prompt=build_issue_implementation_prompt(
            issue_number,
            approved_plan,
            config,
            memory,
            issue_context=issue_context,
        ),
        session_id=coder_session_id,
        marker_description="<!-- AGENT_PR: <number> --> or PR URL",
        validate=lambda text, human_requirements=issue_context.human_requirements: _validate_response_with_human_requirements(
            text,
            marker_validator=_require_pr_number,
            human_requirements=human_requirements,
            requirement_scope="implementation requirements",
            full_omission_fallback="Fetch the issue discussion directly before implementing.",
        ),
        usage_context=usage_context,
    )
    coder_output = coder_response.text
    validate_response_tests_within_workdir(coder_output, assigned_workdir=active_workdir(config))
    validate_assigned_head_advanced(
        before_head=assigned_head_before,
        after_head=_read_assigned_workdir_head(runner, config),
        assigned_workdir=active_workdir(config),
    )
    pr_number = int(coder_response.marker_value)
    log(config, f"{coder_name} reported PR #{pr_number}; validating it is open")
    validate_open_pr(runner, config=config, pr_number=pr_number)
    validate_pr_references_issue(
        runner,
        config=config,
        pr_number=pr_number,
        issue_number=issue_number,
    )
    initial_pr_context = get_pr_review_context(runner, config=config, pr_number=pr_number)
    if one_shot_parent_issue is not None:
        post_one_shot_impl_handoff_comment(
            runner,
            config=config,
            parent_issue=one_shot_parent_issue,
            mode="implement-one-shot",
            plan_hash=approved_plan_hash(approved_plan),
            plan_subject=plan_subject or "",
            pr_number=pr_number,
            pr_head_sha=initial_pr_context.metadata.head_sha,
        )
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
                subject=str(initial_pr_context.metadata.head_sha or "unknown"),
                prior_items=(),
                model_used=coder_response.model_used,
            ),
        ),
    )
    return run_pr_loop(
        runner,
        pr_number=pr_number,
        config=config,
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
                existing_handoff = find_existing_one_shot_impl_handoff(
                    issue_context.comments,
                    parent_issue=issue_number,
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
                            f"for issue #{issue_number} cannot be found in {config.repo}. "
                            f"Verify the PR exists and rerun "
                            f"`agent-loop pr {existing_handoff.pr_number}` directly to continue, "
                            "or remove the handoff comment from the issue and rerun to re-implement."
                        )
                    if pr_state == "OPEN":
                        log(
                            config,
                            f"Issue #{issue_number}: resuming PR #{existing_handoff.pr_number} "
                            "review for already-handed-off plan",
                        )
                        validate_pr_references_issue(
                            runner,
                            config=config,
                            pr_number=existing_handoff.pr_number,
                            issue_number=issue_number,
                        )
                        return run_pr_loop(
                            runner,
                            pr_number=existing_handoff.pr_number,
                            config=config,
                            issue_context=issue_context,
                            usage_context=usage_context,
                        )
                    else:
                        print(
                            f"Issue #{issue_number} approved plan was handed off to "
                            f"PR #{existing_handoff.pr_number}, which is "
                            f"{pr_state.lower()}. Nothing to resume."
                        )
                        return 0
                return _implement_approved_issue(
                    runner,
                    issue_number=issue_number,
                    approved_plan=current_plan,
                    config=config,
                    memory=memory,
                    issue_context=issue_context,
                    coder_session_id=coder_session_id,
                    usage_context=usage_context,
                    one_shot_parent_issue=issue_number,
                    plan_subject=plan_subject,
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
        coder_response = _run_validated_agent(
            runner,
            agent=config.coder,
            config=config,
            prompt=build_issue_prompt(issue_number, config, memory, issue_context=issue_context),
            marker_description="<!-- AGENT_PR: <number> --> or PR URL",
            validate=lambda text, human_requirements=issue_context.human_requirements: _validate_response_with_human_requirements(
                text,
                marker_validator=_require_pr_number,
                human_requirements=human_requirements,
                requirement_scope="implementation requirements",
                full_omission_fallback="Fetch the issue discussion directly before implementing.",
            ),
            usage_context=usage_context,
        )
        coder_output = coder_response.text
        coder_session_id = coder_response.session_id
        validate_response_tests_within_workdir(coder_output, assigned_workdir=active_workdir(config))
        validate_assigned_head_advanced(
            before_head=assigned_head_before,
            after_head=_read_assigned_workdir_head(runner, config),
            assigned_workdir=active_workdir(config),
        )
        pr_number = int(coder_response.marker_value)
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
            )
            coder_output = coder_response.text
            session_id = coder_response.session_id

            if isinstance(coder_response.marker_value, int):
                validate_response_tests_within_workdir(coder_output, assigned_workdir=active_workdir(config))
                validate_assigned_head_advanced(
                    before_head=assigned_head_before,
                    after_head=_read_assigned_workdir_head(runner, config),
                    assigned_workdir=active_workdir(config),
                )
                pr_number = coder_response.marker_value
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
                    )
                    review_output = review_response.text
                    review_model_used = review_response.model_used
                    reviewer_session_ids[reviewer] = review_response.session_id
                    parsed_review = review_response.marker_value
                    assert isinstance(parsed_review, ParsedReview)
                    review_state = parsed_review.state
                    reviewer_new_unresolved_items = []

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
                has_blocking_summary = _should_record_new_blocking_item(
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
                        if has_blocking_summary:
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
                        _publish_approved_followups(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            head_sha=pr_metadata.head_sha,
                            pr_comments=pr_comments,
                            followups=future_followups,
                        )
                    if pr_checks.state in {"failing", "pending", "unavailable"}:
                        details = _pr_check_details(pr_checks)
                        post_pr_comment(
                            runner,
                            config=config,
                            pr_number=pr_number,
                            body=_format_pr_checks_comment(pr_number, pr_checks.state, details),
                        )
                        if pr_checks.state == "failing":
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
                        else:
                            raise AgentLoopError(
                                f"GitHub PR checks for PR #{pr_number} are {pr_checks.state}; "
                                "wait for CI or investigate GitHub API access before treating the PR as approved."
                            )
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


def _discuss_subject(issue_context: IssueContext) -> str:
    text = (issue_context.title or "") + "\n\n" + (issue_context.body or "")
    non_consensus_bodies = [
        c.body
        for c in issue_context.comments
        if c.body and not DISCUSS_CONSENSUS_MARKER_RE.search(c.body)
    ]
    if non_consensus_bodies:
        text += "\n\n" + "\n\n".join(non_consensus_bodies)
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
    for comment in issue_context.comments or []:
        body = comment.body or ""
        m = DISCUSS_CONSENSUS_MARKER_RE.search(body)
        if m and m.group(1).lower() == subject:
            log(config, f"discuss: found matching consensus comment for issue #{issue_number}; skipping")
            return 0
    memory = prepare_agent_memory(runner, config)
    round_history: list[list[ParsedDiscussReview]] = []
    consensus: tuple[str, list[str]] | None = None
    max_round_number = discuss_max_rounds + 1
    for round_number in range(1, max_round_number + 1):
        prior_round_votes = round_history[-1] if round_history else []
        reviewer_votes: list[ParsedDiscussReview] = []
        for reviewer in _reviewers(config):
            reviewer_name = agent_display_name(reviewer)
            log(
                config,
                f"discuss: invoking {reviewer_name} on issue #{issue_number} "
                f"(round {round_number})",
            )
            response = _run_validated_agent(
                runner,
                agent=reviewer,
                config=config,
                prompt=build_discuss_review_prompt(
                    issue_number,
                    config,
                    reviewer=reviewer,
                    memory=memory,
                    issue_context=issue_context,
                    round_number=round_number,
                    prior_round_votes=prior_round_votes,
                ),
                marker_description="<!-- AGENT_PLAN_STATE: approved -->",
                validate=lambda text, r=reviewer_name, rn=round_number: validate_structured_discuss_review(
                    text, reviewer=r, round_number=rn
                ),
                usage_context=usage_context,
                use_repair=True,
                repair_expected_kind="discuss_review",
                role="reviewer",
            )
            parsed = response.marker_value
            assert isinstance(parsed, ParsedDiscussReview)
            reviewer_votes.append(parsed)
        round_history.append(reviewer_votes)
        consensus = _detect_discuss_consensus(reviewer_votes)
        if consensus is not None:
            break
    final_votes = round_history[-1]
    if consensus is None:
        outcome = "needs-human"
        split_proposals = []
        consensus_kind = "deadlock"
    else:
        outcome, split_proposals = consensus
        consensus_kind = "unanimous" if len(round_history) == 1 else "converged"
    body = render_discuss_consensus_comment(
        outcome=outcome,
        consensus_kind=consensus_kind,
        round_number=len(round_history),
        reviewer_votes=final_votes,
        round_history=round_history,
        split_proposals=split_proposals,
        subject=subject,
        config=config,
    )
    post_issue_comment(runner, config=config, issue_number=issue_number, body=body)
    log(
        config,
        f"discuss: posted consensus comment for issue #{issue_number} "
        f"(outcome: {outcome}; kind: {consensus_kind})",
    )
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
