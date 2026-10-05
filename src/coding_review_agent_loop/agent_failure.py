"""Agent response types and agent-failure classification and diagnostics.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1190); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import datetime
import json
import re
import zoneinfo
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from .agents.base import AgentName, AgentResult
from .config import AgentLoopConfig, configured_model_for
from .errors import AgentInvocationError, AgentLoopError
from .logging import log
from .protocol import (
    AgentUnavailable,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
    PUBLIC_RESPONSE_MARKER,
    parse_human_requirements_acknowledgement,
    normalize_response_file_structured_text,
    validate_human_requirements_acknowledgement,
    validate_human_requirement_dispositions,
    validate_structured_plan_revision,
)
from .reset_parsing import (
    _ABSOLUTE_RESET_TIME_RE,
    _ISO_TIMESTAMP_RE,
    _RESET_IN_RE,
    _RETRY_AFTER_SECONDS_RE,
    _TRY_AGAIN_IN_RE,
    _parse_absolute_reset_seconds,
    _parse_rate_limit_reset_seconds,
)
from .runner import Runner
from .salvage import (
    SalvageArtifacts,
    SalvageContext,
    capture_salvage_artifacts,
    post_salvage_comment,
)
from .transient import (
    NON_RETRYABLE_AGENT_OUTPUT_RE,
    TRANSIENT_AGENT_OUTPUT_RE,
    ProviderFailureVerdict,
    classify_codex_failure,
    is_transient_agent_output,
)
from .usage import UsageMetadata
from .workdirs import active_workdir
from .protocol_markers import (
    is_complete_marker_occurrence,
    sanitize_historical_text,
    scan_reserved_markers,
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
    {"plan_state", "plan_review", "pr_review", "coder_followup", "issue_implementation", "plan_revision", "plan_revision_patch", "discuss_review", "discuss_answer", "discuss_semantic_comparison", "discuss_answer_confirmation"}
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
    r"(?:selected\s+)?model\s+is\s+(?:currently\s+)?at\s+capacity\b|"
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

# Failure categories that already name a definitive provider, credential, or
# host condition extracted from the agent's own diagnostics. A review-substance
# repair refusal (#871) must never reclassify one of these into a retryable
# reviewer unavailability: the reviewer turn failed for a reason a rerun cannot
# fix, and the operator needs the original suggestion.
_PROVIDER_DEFINITIVE_FAILURE_CATEGORIES = frozenset(
    {
        "non-retryable",
        "unsupported_model",
        "unsupported_effort",
        "resource-exhausted",
        "containment-indeterminate",
    }
)

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
@dataclass(frozen=True)
class ValidatedAgentResponse:
    text: str
    session_id: str | None
    marker_value: object
    usage: UsageMetadata | None = None
    # Model the agent actually ran, for the dynamic signature (#332). Carried from
    # AgentResult.model_used so the orchestrator render sites can stamp it.
    model_used: str | None = None
    provider: AgentName | None = None
    role: str | None = None
    configured_model: str | None = None
    configured_effort: str | None = None
    effort_source: str | None = None
    observed_model: str | None = None
    observed_effort: str | None = None
    observation_provenance: str | None = None
    acquisition_outcome: Literal["success", "accepted_nonzero_exit", "accepted_timeout"] = "success"
    acquisition_returncode: int | None = None
    # Ephemeral coder acquisition authority. These values survive format
    # repair in memory but are intentionally excluded from durable metadata.
    acquisition_test_turn_id: str | None = None
    acquisition_test_observations: tuple[object, ...] = ()


def _response_identity_fields(
    result: AgentResult | object,
    *,
    acquisition_result: AgentResult | object | None = None,
) -> dict[str, object]:
    acquisition = acquisition_result if acquisition_result is not None else result
    return {
        "provider": getattr(result, "provider", None),
        "role": getattr(result, "role", None),
        "configured_model": getattr(result, "configured_model", None),
        "configured_effort": getattr(result, "configured_effort", None),
        "effort_source": getattr(result, "effort_source", None),
        "observed_model": getattr(result, "observed_model", None),
        "observed_effort": getattr(result, "observed_effort", None),
        "observation_provenance": getattr(result, "observation_provenance", None),
        "acquisition_test_turn_id": getattr(
            acquisition, "test_turn_id", getattr(acquisition, "acquisition_test_turn_id", None)
        ),
        "acquisition_test_observations": getattr(
            acquisition,
            "test_turn_observations",
            getattr(acquisition, "acquisition_test_observations", ()),
        ),
    }


def _metadata_identity_fields(response: object) -> dict[str, object]:
    """Identity fields safe to expand into PostedRoundMetadata."""
    fields = _response_identity_fields(response)  # type: ignore[arg-type]
    fields.pop("role", None)
    fields.pop("acquisition_test_turn_id", None)
    fields.pop("acquisition_test_observations", None)
    return fields


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


def _recognized_structured_public_response_kind(text: str) -> str | None:
    """Return a trusted response kind without interpreting its free-form values."""
    payload = _decode_public_response_json_prefix(text)
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    return kind if isinstance(kind, str) and kind in STRUCTURED_PUBLIC_RESPONSE_KINDS else None


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


def _looks_like_unsupported_effort_text(text: str) -> bool:
    """Recognize a CLI rejecting the tool-owned effort flag."""
    return bool(
        re.search(
            r"(?:unknown|unrecognized|invalid|unsupported)\s+"
            r"(?:option|flag|argument).*--effort|"
            r"--effort.*(?:unknown|unrecognized|invalid|unsupported|not supported)",
            text,
            re.I,
        )
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
    if (
        _recognized_structured_public_response_kind(text) is not None
        and (repair_expected_kind is None or repair_expected_kind in STRUCTURED_PUBLIC_RESPONSE_KINDS)
    ):
        return False
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


_STRUCTURED_SCHEMA_REJECTION_RE = re.compile(
    r"^structured [a-z_]+ (?:response|repair) failed trusted validation$"
)


# Classification text for a semantic evidence rejection that skipped repair
# (#990). The terminal category names this rejection, not a repair symptom.
_SEMANTIC_EVIDENCE_REJECTION_CLASSIFICATION = (
    "structured response failed semantic evidence validation"
)


def _is_structured_schema_rejection(classification_text: str) -> bool:
    """True when a recognized structured envelope failed schema validation (#957).

    The rejection is deterministic for that output, but the output came from a
    stochastic model, so the operator guidance must not discourage a rerun.
    """
    return any(
        _STRUCTURED_SCHEMA_REJECTION_RE.match(line.strip())
        for line in (classification_text or "").splitlines()
    )


def _failure_category(
    text: str,
    *,
    public_response: bool = False,
    repair_expected_kind: str | None = None,
    provider_verdict: ProviderFailureVerdict | None = None,
) -> str:
    """Classify a failure for logging: helps users decide whether to rerun or fix config/code."""
    if not text.strip():
        return "empty-response"
    if provider_verdict is not None and provider_verdict.source == "structured":
        if _looks_like_unsupported_effort_text(text):
            return "unsupported_effort"
        if _unsupported_model_classification_text(
            text, public_response=public_response, repair_expected_kind=repair_expected_kind
        ):
            return "unsupported_model"
        return provider_verdict.category
    if (
        public_response
        and _recognized_structured_public_response_kind(text) is not None
        and (repair_expected_kind is None or repair_expected_kind in STRUCTURED_PUBLIC_RESPONSE_KINDS)
    ):
        # Structured response values are agent-authored content, not provider
        # diagnostics. A rejected envelope remains deterministic even if its
        # prose happens to contain auth, billing, credit, timeout, or dirty-tree
        # vocabulary.
        return "deterministic"
    lowered = text.lower()
    if _looks_like_unsupported_effort_text(text):
        return "unsupported_effort"
    if "resource-exhausted" in lowered or "resource exhausted" in lowered:
        return "resource-exhausted"
    if "containment-indeterminate" in lowered or "cleanup-failed" in lowered:
        return "containment-indeterminate"
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


def _neutralize_untrusted_markers(
    text: str, *, config: AgentLoopConfig, agent_name: str
) -> str:
    """Defang reserved markers an agent merely named in its prose (#891).

    An agent describing protocol code legitimately writes a token such as a
    split-warning record name.  Refusing the whole response makes any work on
    the protocol itself unreviewable, so neutralize the span into its stable
    label instead.  The markers still carry no authority: the sanitized text
    cannot be parsed as a durable record, and tool-owned publications keep
    their own fail-closed check.
    """
    if not isinstance(text, str) or not text:
        return text
    occurrences = scan_reserved_markers(text)
    if not occurrences:
        return text
    # Emitting a complete, parseable record is a forgery attempt and keeps the
    # existing fail-closed behavior.  Only bare names in prose are defanged.
    if any(is_complete_marker_occurrence(item) for item in occurrences):
        return text
    sanitized = sanitize_historical_text(text)
    names = ", ".join(sorted({item.definition.token for item in occurrences}))
    log(
        config,
        f"{agent_name}: neutralized reserved protocol marker(s) named in the response "
        f"prose: {names}",
    )
    return sanitized


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
                source_plan = validate_structured_plan_revision(source_text, architecture_status_mode="legacy")
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
    *,
    role: str | None = None,
) -> str | None:
    if config is None or agent is None:
        return None
    model = configured_model_for(config, agent, role=role)
    return model.strip() if model else None


def _resolve_requested_model(
    *,
    agent: AgentName | None,
    config: AgentLoopConfig | None,
    result: AgentResult | None,
    classification_text: str,
    role: str | None = None,
) -> str | None:
    result_model = (
        _model_flag_value(result.model_used)
        if result is not None and result.model_used
        else None
    )
    config_model = _configured_requested_model(agent, config, role=role)
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
        role=role,
    )
    provider_auth_context = _extract_provider_auth_context(unsupported_text)
    reason = _extract_unsupported_model_reason(unsupported_text)
    fallback_flag, fallback_value = _known_unsupported_model_fallback(
        agent=agent,
        requested_model=requested_model,
        provider_auth_context=provider_auth_context,
        reason=reason,
    )
    if (
        fallback_flag == "--codex-model"
        and role == "reviewer"
        and config is not None
        and config.reviewer_codex_model
    ):
        fallback_flag = "--reviewer-codex-model"
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
    if category == "unsupported_effort":
        return (
            "Suggestion: use the dedicated effort option and upgrade the provider CLI "
            "to a release that accepts its explicit effort flag; agent-loop will not "
            "retry without the requested setting."
        )
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
        if re.search(
            r"\b(?:credit|billing)\b|insufficient_quota|check your plan|payment required|\b402\b",
            combined,
            re.I,
        ):
            return "Suggestion: check your API billing / credit balance, then re-run."
        if re.search(r"\bdirty\b", combined, re.I):
            return "Suggestion: clean up the dirty working tree or workdir, then re-run."
        return f"Suggestion: check that {agent_name} is installed and authenticated, then re-run."
    if category == "deterministic":
        if _is_structured_schema_rejection(classification_text):
            return (
                "Suggestion: re-run the same command — the agent's structured response "
                "failed schema validation, and model output varies between runs, so a "
                "retry may succeed. If the same rejection recurs, inspect the log above."
            )
        if "repair invocation failure" in reason and "invalid_output" in reason:
            return (
                "Suggestion: re-run the same command — "
                "the round is resumable and a retry may succeed."
            )
        return "Suggestion: inspect the log above, fix the underlying issue, then re-run."
    if category == "repair-provider-failure":
        if "transient_provider_error" in reason:
            return (
                "Suggestion: re-run the same command — the Antigravity repair model "
                "reported a transient model-access failure; the round is resumable "
                "and a retry may succeed."
            )
        return ""
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
    elif (
        category == "deterministic"
        and classification_text == _SEMANTIC_EVIDENCE_REJECTION_CLASSIFICATION
    ):
        category_hint = (
            " Failure category: semantic-evidence-rejection (a risk-matrix claim selected a "
            "test observation that cannot carry authority; repair was skipped because "
            "reformatting cannot change it)."
        )
    elif category == "deterministic" and _is_structured_schema_rejection(classification_text):
        category_hint = (
            " Failure category: schema-validation (the agent's structured response did not "
            "match the schema; model output varies between runs, so a rerun may succeed)."
        )
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
    elif category == "unsupported_effort":
        category_hint = (
            " Failure category: unsupported_effort (the provider CLI rejected the "
            "explicit --effort setting; no retry omitted it)."
        )
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
        f"Reason: {reason}. Required marker: {marker_description}.{exit_context}"
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
    if repair_expected_kind == "discuss_round_synthesis":
        return "discuss round synthesis"
    if repair_expected_kind == "discuss_final_synthesis":
        return "discuss final synthesis"
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


def _agent_failure_classification(
    result: AgentResult,
    *,
    phase: str,
) -> tuple[str, ProviderFailureVerdict | None]:
    """Choose the classification text and, for Codex, the provider verdict (#1269)."""
    if phase in {"command", "empty"}:
        if result.provider == "codex":
            verdict = classify_codex_failure(result.raw_output or "")
            if verdict is not None:
                return verdict.text, verdict
        return result.raw_output or result.text, None
    return result.text, None


def _agent_failure_classification_text(
    result: AgentResult,
    *,
    phase: str,
) -> str:
    """Choose the text that matches the failure being classified."""
    return _agent_failure_classification(result, phase=phase)[0]
