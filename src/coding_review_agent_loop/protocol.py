"""Parsing for agent response markers."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass, field

from .errors import AgentLoopError

PUBLIC_RESPONSE_MARKER = "=== AGENT_LOOP_PUBLIC_RESPONSE_BELOW ==="

HUMAN_REQUIREMENTS_ADDRESSED_MARKER = "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->"
HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK = (
    "checked the relevant GitHub discussion directly before responding"
)

STATE_RE = re.compile(r"<!--\s*AGENT_STATE:\s*(approved|blocking)\s*-->", re.I)
PLAN_STATE_RE = re.compile(r"<!--\s*AGENT_PLAN_STATE:\s*(approved|blocking)\s*-->", re.I)
PR_RE = re.compile(r"<!--\s*AGENT_PR:\s*(\d+)\s*-->", re.I)
GH_PR_URL_RE = re.compile(r"/pull/(\d+)(?:\b|$)")
CLARIFY_RE = re.compile(r"<!--\s*AGENT_CLARIFY\s*-->", re.I)
# Standalone variants: marker must occupy its own line.
_STANDALONE_CLARIFY_RE = re.compile(r"(?m)^\s*<!--\s*AGENT_CLARIFY\s*-->\s*$", re.I)
_STANDALONE_STATE_RE = re.compile(r"(?m)^\s*<!--\s*AGENT_STATE:\s*(approved|blocking)\s*-->\s*$", re.I)
_STANDALONE_PLAN_STATE_RE = re.compile(
    r"(?m)^\s*<!--\s*AGENT_PLAN_STATE:\s*(approved|blocking)\s*-->\s*$", re.I
)
_STANDALONE_PR_RE = re.compile(r"(?m)^\s*<!--\s*AGENT_PR:\s*(\d+)\s*-->\s*$", re.I)
# Matches the opening or closing line of a fenced code block (``` or ~~~, 3+ chars).
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})", re.M)
HUMAN_REVIEWER_SIGNATURE_RE = re.compile(r"^\s*--\s*Human Reviewer\s*$", re.I | re.M)
HUMAN_REQUIREMENTS_RESOLVED_RE = re.compile(
    r"<!--\s*HUMAN_REQUIREMENTS_RESOLVED\s*-->",
    re.I,
)
HUMAN_REQUIREMENTS_ADDRESSED_RE = re.compile(
    re.escape(HUMAN_REQUIREMENTS_ADDRESSED_MARKER),
    re.I,
)
HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK_RE = re.compile(
    re.escape(HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK),
    re.I,
)


def _followup_heading_re(title: str) -> re.Pattern[str]:
    title_with_punctuation = rf"{title}[:.]?"
    return re.compile(
        rf"^\s*#{{2,6}}\s+"
        rf"(?:{title_with_punctuation}|\*\*{title}\*\*[:.]?|\*\*{title_with_punctuation}\*\*)"
        rf"\s*$",
        re.I,
    )


SAME_PR_FOLLOWUP_HEADING_RE = _followup_heading_re(r"same[- ]pr follow[- ]ups")
FUTURE_FOLLOWUP_HEADING_RE = _followup_heading_re(r"future follow[- ]ups")
LEGACY_FOLLOWUP_HEADING_RE = _followup_heading_re(r"non[- ]blocking follow[- ]ups")
PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE = _followup_heading_re(
    r"prior unresolved item dispositions"
)
BLOCKING_ISSUES_HEADING_RE = _followup_heading_re(r"blocking issues")
BLOCKING_PLAN_ISSUES_HEADING_RE = _followup_heading_re(r"blocking plan issues")
SAME_PLAN_FOLLOWUP_HEADING_RE = _followup_heading_re(r"same[- ]plan follow[- ]ups")
HUMAN_REQUIREMENTS_HEADING_RE = _followup_heading_re(r"human requirements")
PRIOR_UNRESOLVED_PLAN_ITEM_DISPOSITIONS_HEADING_RE = _followup_heading_re(
    r"prior unresolved plan item dispositions"
)
ANY_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")
HTML_COMMENT_RE = re.compile(r"^\s*<!--.*-->\s*$")
SIGNATURE_RE = re.compile(r"^\s*--\s+\S")
BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(?P<text>.+?)\s*$")
HEADING_LEVEL_RE = re.compile(r"^\s*(#{1,6})\s+\S")
THEMATIC_BREAK_RE = re.compile(r"^\s*(?:([-*_])\s*){3,}\s*$")
ITEM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _empty_placeholder_re(*phrases: str) -> re.Pattern[str]:
    joined = "|".join(phrases)
    return re.compile(rf"^\(?\s*(?:none|n/a|{joined})\s*\)?\.?$", re.I)


EMPTY_FOLLOWUP_RE = _empty_placeholder_re(
    r"no follow[- ]?ups?",
    r"no same[- ]pr follow[- ]?ups?",
    r"no future follow[- ]?ups?",
)
EMPTY_PLAN_SECTION_RE = _empty_placeholder_re(
    r"no blocking plan issues?",
    r"no same[- ]plan follow[- ]?ups?",
    r"no future follow[- ]?ups?",
)


@dataclass(frozen=True)
class ApprovedFollowup:
    reviewer: str
    text: str


@dataclass(frozen=True)
class ApprovedFollowups:
    same_pr: tuple[ApprovedFollowup, ...]
    future: tuple[ApprovedFollowup, ...]


@dataclass(frozen=True)
class ReviewItemDisposition:
    item_id: str
    reviewer: str
    disposition: str
    note: str | None = None


@dataclass(frozen=True)
class UnresolvedReviewItem:
    item_id: str
    reviewer: str
    source_round: int
    text: str
    status: str
    source_status: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedReview:
    state: str
    summary: str
    blocking_items: tuple[ApprovedFollowup, ...]
    followups: ApprovedFollowups
    dispositions: tuple[ReviewItemDisposition, ...]
    raw_dispositions_text: str = ""


@dataclass(frozen=True)
class PlanReviewItems:
    blocking: tuple[ApprovedFollowup, ...]
    same_plan: tuple[ApprovedFollowup, ...]
    future: tuple[ApprovedFollowup, ...]


@dataclass(frozen=True)
class ParsedPlanReview:
    state: str
    summary: str
    items: PlanReviewItems
    dispositions: tuple[ReviewItemDisposition, ...]
    raw_dispositions_text: str = ""


@dataclass(frozen=True)
class StructuredPrReview:
    schema_version: int
    kind: str
    state: str
    summary: str
    blocking_items: tuple[str, ...]
    same_pr_followups: tuple[str, ...]
    future_followups: tuple[str, ...]
    prior_item_dispositions: tuple[ReviewItemDisposition, ...]


@dataclass(frozen=True)
class StructuredPlanReview:
    schema_version: int
    kind: str
    state: str
    summary: str
    blocking_plan_issues: tuple[str, ...]
    same_plan_followups: tuple[str, ...]
    future_followups: tuple[str, ...]
    prior_plan_item_dispositions: tuple[ReviewItemDisposition, ...]


@dataclass(frozen=True)
class StructuredHumanRequirementsPayload:
    addressed_ids: tuple[str, ...]
    checked_discussion_directly: bool


@dataclass(frozen=True)
class StructuredCoderFollowup:
    schema_version: int
    kind: str
    state: str
    summary: str
    addressed_items: tuple[str, ...]
    remaining_items: tuple[str, ...]
    human_requirements: StructuredHumanRequirementsPayload
    addressed_item_notes: dict[str, str]
    remaining_item_notes: dict[str, str]
    tests_run: tuple[str, ...] | None = None
    disputed_items: tuple[str, ...] = ()
    dispute_evidence: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredPlanRevision:
    schema_version: int
    kind: str
    state: str
    summary: str
    prior_plan_item_dispositions: tuple[ReviewItemDisposition, ...]
    plan_steps: tuple[str, ...]


@dataclass(frozen=True)
class StructuredPlanState:
    schema_version: int
    kind: str
    state: str
    summary: str
    plan_steps: tuple[str, ...]


@dataclass(frozen=True)
class StructuredDiscussReview:
    schema_version: int
    kind: str
    outcome: str
    rationale: str
    split_proposals: tuple[str, ...]
    rebuttal: str | None = None


@dataclass(frozen=True)
class ParsedDiscussReview:
    outcome: str
    rationale: str
    split_proposals: tuple[str, ...]
    reviewer: str
    rebuttal: str | None = None


DISCUSS_OUTCOME_VALUES = frozenset({"implement", "do-not-implement", "needs-human", "split"})


@dataclass(frozen=True)
class ParsedHumanRequirementsAcknowledgement:
    marker_present: bool
    section_present: bool
    addressed_ids: tuple[str, ...]
    section_text: str


def parse_agent_state(text: str) -> str:
    matches = STATE_RE.findall(text)
    if not matches:
        raise AgentLoopError("Agent response did not include <!-- AGENT_STATE: approved|blocking -->")
    # Use the final marker as authoritative; responses may quote earlier review markers.
    return matches[-1].lower()


def parse_plan_state(text: str) -> str:
    matches = PLAN_STATE_RE.findall(text)
    if not matches:
        raise AgentLoopError(
            "Agent response did not include <!-- AGENT_PLAN_STATE: approved|blocking -->"
        )
    return matches[-1].lower()


def parse_pr_number(text: str) -> int | None:
    # Use the final marker as authoritative, consistent with parse_agent_state.
    markers = list(PR_RE.finditer(text))
    if markers:
        return int(markers[-1].group(1))
    urls = list(GH_PR_URL_RE.finditer(text))
    if urls:
        return int(urls[-1].group(1))
    return None


def _fenced_code_block_ranges(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character ranges for each complete fenced code block."""
    ranges: list[tuple[int, int]] = []
    open_char: str | None = None
    open_len: int = 0
    open_start: int = 0
    for m in _FENCE_RE.finditer(text):
        fence_chars = m.group(1)
        char = fence_chars[0]
        length = len(fence_chars)
        if open_char is None:
            open_char, open_len, open_start = char, length, m.start()
        elif char == open_char and length >= open_len:
            line_end = text.find("\n", m.start())
            end = (line_end + 1) if line_end != -1 else len(text)
            ranges.append((open_start, end))
            open_char = None
    return ranges


def is_clarification_request(text: str) -> bool:
    # Only standalone AGENT_CLARIFY (own line) counts; inline examples are ignored.
    clarify_matches = list(_STANDALONE_CLARIFY_RE.finditer(text))
    if not clarify_matches:
        return False
    # Exclude matches inside fenced code blocks.
    code_ranges = _fenced_code_block_ranges(text)

    def _in_code_block(pos: int) -> bool:
        return any(start <= pos < end for start, end in code_ranges)

    active_clarify = [m for m in clarify_matches if not _in_code_block(m.start())]
    if not active_clarify:
        return False
    last_clarify_pos = active_clarify[-1].start()
    # A standalone AGENT_STATE / AGENT_PLAN_STATE / AGENT_PR marker that is NOT inside a
    # code block AND appears AFTER the last active AGENT_CLARIFY takes precedence.
    # Markers appearing before the final AGENT_CLARIFY (e.g. from an earlier round's footer
    # quoted in prose, or a plan state preceding an appendix question) do not suppress it.
    # Inline markers in prose (non-standalone) are also ignored.
    for regex in (_STANDALONE_STATE_RE, _STANDALONE_PLAN_STATE_RE, _STANDALONE_PR_RE):
        for m in regex.finditer(text):
            if not _in_code_block(m.start()) and m.start() > last_clarify_pos:
                return False
    # GH_PR_URL is likewise positional: a non-code-block PR URL appearing after the last
    # active AGENT_CLARIFY also takes precedence.
    if any(
        not _in_code_block(m.start()) and m.start() > last_clarify_pos
        for m in GH_PR_URL_RE.finditer(text)
    ):
        return False
    # AGENT_CLARIFY must be the final content: only blank lines and/or
    # signature lines (``-- Name``) may follow the last active marker.
    last_m = active_clarify[-1]
    after_clarify = text[last_m.end():]
    for line in after_clarify.splitlines():
        if not line.strip():
            continue
        if SIGNATURE_RE.match(line):
            continue
        return False
    return True


def parse_signed_human_requirement_body(text: str | None) -> str | None:
    """Return comment body before a standalone ``-- Human Reviewer`` signature."""
    if not text:
        return None
    match = HUMAN_REVIEWER_SIGNATURE_RE.search(text)
    if not match:
        return None
    body = text[: match.start()].strip()
    return body or None


def human_requirements_resolved(text: str) -> bool:
    return bool(HUMAN_REQUIREMENTS_RESOLVED_RE.search(text))


def _normalize_requirement_label(text: str) -> str:
    match = re.fullmatch(r"\s*Requirement\s+(\d+)\s*", text, re.I)
    if not match:
        raise AgentLoopError(
            f"Invalid human requirement label: {text}. Only exact surfaced signed labels like "
            "`Requirement 1` are valid; issue acceptance criteria, reviewer item IDs, reviewer "
            "comments, and arbitrary labels are not signed human requirements."
        )
    return f"Requirement {match.group(1)}"


def parse_human_requirements_acknowledgement(text: str) -> ParsedHumanRequirementsAcknowledgement:
    marker_present = bool(HUMAN_REQUIREMENTS_ADDRESSED_RE.search(text))
    section_present = False
    section_lines: list[str] = []
    addressed_ids: list[str] = []
    active = False

    for line in text.splitlines():
        if HUMAN_REQUIREMENTS_HEADING_RE.match(line):
            section_present = True
            active = True
            continue
        if not active:
            continue
        if ANY_HEADING_RE.match(line) or HTML_COMMENT_RE.match(line) or SIGNATURE_RE.match(line):
            active = False
            continue
        section_lines.append(line)
        bullet = BULLET_RE.match(line)
        if not bullet:
            continue
        for match in re.finditer(r"\bRequirement\s+\d+\b", bullet.group("text"), re.I):
            addressed_ids.append(_normalize_requirement_label(match.group(0)))

    return ParsedHumanRequirementsAcknowledgement(
        marker_present=marker_present,
        section_present=section_present,
        addressed_ids=tuple(addressed_ids),
        section_text="\n".join(section_lines).strip(),
    )


def validate_human_requirements_acknowledgement(
    text: str,
    *,
    surfaced_requirement_ids: Sequence[str],
    requires_direct_discussion_ack: bool,
) -> None:
    parsed = parse_human_requirements_acknowledgement(text)
    validate_structured_human_requirements_acknowledgement(
        parsed.addressed_ids,
        checked_discussion_directly=HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK_RE.search(
            parsed.section_text
        )
        is not None,
        surfaced_requirement_ids=surfaced_requirement_ids,
        requires_direct_discussion_ack=requires_direct_discussion_ack,
        marker_present=parsed.marker_present,
        section_present=parsed.section_present,
    )


def validate_structured_human_requirements_acknowledgement(
    addressed_ids: Sequence[str],
    *,
    checked_discussion_directly: bool,
    surfaced_requirement_ids: Sequence[str],
    requires_direct_discussion_ack: bool,
    marker_present: bool = True,
    section_present: bool = True,
) -> None:
    if not surfaced_requirement_ids and not requires_direct_discussion_ack:
        if addressed_ids:
            raise AgentLoopError(
                "Coder response listed signed human requirement IDs even though no signed human "
                "requirements were surfaced. Use `human_requirements.addressed_ids: []`; issue "
                "acceptance criteria, reviewer item IDs, reviewer comments, and arbitrary labels "
                "are not signed human requirements."
            )
        return

    if not marker_present:
        raise AgentLoopError(
            "Coder response missing required signed human requirements marker "
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}."
        )
    if not section_present:
        raise AgentLoopError("Coder response missing required `### Human requirements` section.")

    normalized_addressed_ids = [_normalize_requirement_label(item_id) for item_id in addressed_ids]
    duplicates = sorted(
        {
            item_id
            for item_id in normalized_addressed_ids
            if normalized_addressed_ids.count(item_id) > 1
        }
    )
    if duplicates:
        raise AgentLoopError(
            "Coder response listed signed human requirement IDs more than once: "
            + ", ".join(duplicates)
        )

    expected_ids = tuple(_normalize_requirement_label(item_id) for item_id in surfaced_requirement_ids)
    unknown = sorted(set(normalized_addressed_ids) - set(expected_ids))
    if unknown:
        raise AgentLoopError(
            "Coder response referenced unknown signed human requirement IDs: "
            + ", ".join(unknown)
        )

    if expected_ids:
        missing = sorted(set(expected_ids) - set(normalized_addressed_ids))
        if missing:
            raise AgentLoopError(
                "Coder response did not address all surfaced signed human requirement IDs: "
                + ", ".join(missing)
            )
        return

    if not checked_discussion_directly:
        raise AgentLoopError(
            "Coder response must acknowledge that the prompt omitted the detailed signed human requirements "
            f"and that it {HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK}."
        )


def review_freeform_summary_text(text: str) -> str:
    lines: list[str] = []
    skip_structured_section = False
    structured_heading_res = (
        BLOCKING_ISSUES_HEADING_RE,
        BLOCKING_PLAN_ISSUES_HEADING_RE,
        SAME_PLAN_FOLLOWUP_HEADING_RE,
        SAME_PR_FOLLOWUP_HEADING_RE,
        FUTURE_FOLLOWUP_HEADING_RE,
        LEGACY_FOLLOWUP_HEADING_RE,
        HUMAN_REQUIREMENTS_HEADING_RE,
        PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE,
        PRIOR_UNRESOLVED_PLAN_ITEM_DISPOSITIONS_HEADING_RE,
    )
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(
            r"^\*\*Review verdict:\*\*\s+(Approved|Blocking)\s*$",
            stripped,
            re.IGNORECASE,
        ):
            continue
        if any(pattern.match(line) for pattern in structured_heading_res):
            skip_structured_section = True
            continue
        if skip_structured_section and stripped.startswith("### "):
            skip_structured_section = False
        if skip_structured_section:
            continue
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if stripped.startswith("-- "):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _expect_object(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AgentLoopError(f"{context} must be a JSON object.")
    return value


def _expect_exact_keys(
    value: dict[str, object],
    *,
    context: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    if missing:
        raise AgentLoopError(f"{context} is missing required field(s): {', '.join(missing)}")
    unknown = sorted(keys - required - optional)
    if unknown:
        raise AgentLoopError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _expect_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentLoopError(f"{context} must be an integer.")
    return value


def _expect_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise AgentLoopError(f"{context} must be a boolean.")
    return value


def _expect_non_empty_string(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise AgentLoopError(f"{context} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise AgentLoopError(f"{context} must be a non-empty string.")
    return normalized


def _expect_string_list(
    value: object,
    *,
    context: str,
    item_context: str,
    min_length: int = 0,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")
    rendered = tuple(
        _expect_non_empty_string(item, context=f"{item_context} at index {index}")
        for index, item in enumerate(value)
    )
    if len(rendered) < min_length:
        raise AgentLoopError(f"{context} must contain at least {min_length} item(s).")
    return rendered


def _expect_optional_string_list(
    payload: dict[str, object],
    field_name: str,
    *,
    context: str,
    item_context: str,
    min_length: int = 0,
) -> tuple[str, ...]:
    value = payload.get(field_name, [])
    return _expect_string_list(
        value,
        context=context,
        item_context=item_context,
        min_length=min_length,
    )


def _expect_state(value: object, *, context: str) -> str:
    state = _expect_non_empty_string(value, context=context)
    if state not in {"approved", "blocking"}:
        raise AgentLoopError(f"{context} must be `approved` or `blocking`.")
    return state


def _expect_item_id(value: object, *, context: str) -> str:
    item_id = _expect_non_empty_string(value, context=context)
    if not ITEM_ID_RE.fullmatch(item_id):
        raise AgentLoopError(
            f"{context} must match `[A-Za-z0-9][A-Za-z0-9._-]*`."
        )
    return item_id


def _expect_item_id_list(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")
    return tuple(
        _expect_item_id(item, context=f"{context} item at index {index}")
        for index, item in enumerate(value)
    )


def _expect_item_note_map(
    value: object,
    *,
    context: str,
    allowed_item_ids: set[str],
    allowed_context: str,
) -> dict[str, str]:
    note_payload = _expect_object(value, context=context)
    rendered: dict[str, str] = {}
    for raw_item_id, raw_note in note_payload.items():
        item_id = _expect_item_id(raw_item_id, context=f"{context} key")
        if item_id not in allowed_item_ids:
            raise AgentLoopError(f"{context} key `{item_id}` is not listed in {allowed_context}.")
        rendered[item_id] = _expect_non_empty_string(raw_note, context=f"{context}.{item_id}")
    return rendered


def _expect_requirement_id_list(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")
    rendered: list[str] = []
    for index, item in enumerate(value):
        label = _expect_non_empty_string(item, context=f"{context} item at index {index}")
        rendered.append(_normalize_requirement_label(label))
    return tuple(rendered)


def _extract_structured_response_object(text: str) -> dict[str, object] | None:
    stripped = text.strip()
    if not stripped:
        return None
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    if stripped[end:].strip():
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def normalize_response_file_structured_text(text: str) -> tuple[str, str | None]:
    """Recover a public-response stdout marker only when it leaked at the front.

    The public response file is already authoritative, so the stdout filtering
    marker is invalid there.  This helper is intentionally narrow: it only strips
    a leading marker after whitespace when the remaining body begins like the
    required structured response.
    """
    stripped = text.lstrip()
    if not stripped.startswith(PUBLIC_RESPONSE_MARKER):
        return text, None
    remainder = stripped[len(PUBLIC_RESPONSE_MARKER) :].lstrip()
    if remainder.startswith("{"):
        return remainder, "leading-public-response-marker-recovered"
    return text, "leading-public-response-marker-not-recoverable"


def _extract_json_object_prefix(text: str) -> tuple[dict[str, object], str] | None:
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return None
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise AgentLoopError("Structured response must begin with one top-level JSON object.") from exc
    if not isinstance(payload, dict):
        raise AgentLoopError("Structured response must begin with a JSON object.")
    return payload, stripped[end:]


def _consume_structured_footer_and_signature(
    *,
    payload: dict[str, object],
    trailing: str,
    state_re: re.Pattern[str],
    state_marker_name: str,
    context_label: str,
    allow_human_requirements_prefix: bool = False,
    allow_human_requirements_marker_after_footer: bool = False,
) -> dict[str, object]:
    trailing = trailing.lstrip()
    state_match = state_re.search(trailing)
    if state_match is None:
        raise AgentLoopError(
            f"{context_label} must place <!-- {state_marker_name}: approved|blocking --> after the JSON object."
        )

    before_footer = trailing[: state_match.start()].strip()
    if before_footer:
        if not allow_human_requirements_prefix:
            raise AgentLoopError(
                f"{context_label} may not include prose between the JSON object and the {state_marker_name} footer."
            )
        parsed_human_requirements = parse_human_requirements_acknowledgement(before_footer)
        if not parsed_human_requirements.marker_present or not parsed_human_requirements.section_present:
            raise AgentLoopError(
                f"{context_label} may only include a signed human requirements acknowledgement before the {state_marker_name} footer."
            )

    trailing = trailing[state_match.end() :].lstrip()
    if allow_human_requirements_marker_after_footer and HUMAN_REQUIREMENTS_RESOLVED_RE.match(trailing):
        marker_match = HUMAN_REQUIREMENTS_RESOLVED_RE.match(trailing)
        assert marker_match is not None
        trailing = trailing[marker_match.end() :].lstrip()
    signature_match = re.match(r"^--\s+\S[^\n]*(?:\n)?$", trailing)
    if signature_match is None or trailing[signature_match.end() :].strip():
        raise AgentLoopError(
            f"{context_label} may not include trailing prose after the JSON footer and signature."
        )

    footer_state = state_match.group(1).lower()
    payload_state = payload.get("state")
    if isinstance(payload_state, str) and payload_state.strip():
        if payload_state.strip() != footer_state:
            raise AgentLoopError(
                f"{context_label} footer {state_marker_name} must match the payload state."
            )
    return payload


def _extract_structured_pr_review_payload(text: str) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)
    if extracted is None:
        return None
    payload, trailing = extracted
    trailing = trailing.lstrip()
    if HUMAN_REQUIREMENTS_RESOLVED_RE.match(trailing):
        marker_match = HUMAN_REQUIREMENTS_RESOLVED_RE.match(trailing)
        assert marker_match is not None
        trailing = trailing[marker_match.end() :]
    return _consume_structured_footer_and_signature(
        payload=payload,
        trailing=trailing,
        state_re=STATE_RE,
        state_marker_name="AGENT_STATE",
        context_label="Structured PR review",
        allow_human_requirements_marker_after_footer=True,
    )


def _extract_structured_plan_review_payload(text: str) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)
    if extracted is None:
        return None
    payload, trailing = extracted
    trailing = trailing.lstrip()
    if HUMAN_REQUIREMENTS_RESOLVED_RE.match(trailing):
        marker_match = HUMAN_REQUIREMENTS_RESOLVED_RE.match(trailing)
        assert marker_match is not None
        trailing = trailing[marker_match.end() :]
    return _consume_structured_footer_and_signature(
        payload=payload,
        trailing=trailing,
        state_re=PLAN_STATE_RE,
        state_marker_name="AGENT_PLAN_STATE",
        context_label="Structured plan review",
    )


def _extract_structured_plan_revision_payload(text: str) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)
    if extracted is None:
        return None
    payload, trailing = extracted
    return _consume_structured_footer_and_signature(
        payload=payload,
        trailing=trailing,
        state_re=PLAN_STATE_RE,
        state_marker_name="AGENT_PLAN_STATE",
        context_label="Structured plan revision",
        allow_human_requirements_prefix=True,
    )


def _extract_structured_plan_state_payload(text: str) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)
    if extracted is None:
        return None
    payload, trailing = extracted
    return _consume_structured_footer_and_signature(
        payload=payload,
        trailing=trailing,
        state_re=PLAN_STATE_RE,
        state_marker_name="AGENT_PLAN_STATE",
        context_label="Structured plan state",
    )


def _extract_structured_discuss_review_payload(text: str) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)
    if extracted is None:
        return None
    payload, trailing = extracted
    result = _consume_structured_footer_and_signature(
        payload=payload,
        trailing=trailing,
        state_re=PLAN_STATE_RE,
        state_marker_name="AGENT_PLAN_STATE",
        context_label="Structured discuss review",
    )
    if result is None:
        return None
    footer_match = PLAN_STATE_RE.search(trailing)
    if footer_match is not None:
        footer_state = footer_match.group(1).lower()
        if footer_state != "approved":
            raise AgentLoopError(
                f"Structured discuss review footer AGENT_PLAN_STATE must be `approved`; got `{footer_state}`."
            )
    return result


def _extract_structured_coder_followup_payload(text: str) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)
    if extracted is None:
        return None
    payload, trailing = extracted
    return _consume_structured_footer_and_signature(
        payload=payload,
        trailing=trailing,
        state_re=STATE_RE,
        state_marker_name="AGENT_STATE",
        context_label="Structured coder follow-up",
    )


def _require_supported_schema_version(payload: dict[str, object]) -> None:
    if "schema_version" not in payload:
        raise AgentLoopError("Structured response is missing required field: schema_version")
    version = _expect_int(payload["schema_version"], context="schema_version")
    if version != 1:
        raise AgentLoopError(f"Unsupported structured response schema_version: {version}")


def _parse_review_item_disposition_payload(
    value: object,
    *,
    field_name: str,
    reviewer: str,
    allowed_same_status: str,
    is_plan_review: bool,
) -> ReviewItemDisposition:
    payload = _expect_object(value, context=field_name)
    _expect_exact_keys(
        payload,
        context=field_name,
        required={"item_id", "disposition"},
        optional={"note"},
    )
    item_id = _expect_item_id(payload["item_id"], context=f"{field_name}.item_id")
    disposition = _expect_non_empty_string(payload["disposition"], context=f"{field_name}.disposition")
    allowed_statuses = {"resolved", "blocking", allowed_same_status, "future"}
    if disposition not in allowed_statuses:
        rendered = ", ".join(sorted(allowed_statuses))
        raise AgentLoopError(f"{field_name}.disposition must be one of: {rendered}")
    note_value = payload.get("note")
    note = None
    if note_value is not None:
        note = _expect_non_empty_string(note_value, context=f"{field_name}.note")
    if _active_disposition_has_empty_note(
        disposition,
        note,
        same_status=allowed_same_status,
        is_plan_review=is_plan_review,
    ):
        raise AgentLoopError(
            f"{field_name}.note cannot be an empty placeholder for active disposition `{disposition}`."
        )
    return ReviewItemDisposition(
        item_id=item_id,
        reviewer=reviewer,
        disposition=disposition,
        note=note,
    )


def _expect_disposition_list(
    value: object,
    *,
    context: str,
    reviewer: str,
    allowed_same_status: str,
    is_plan_review: bool,
) -> tuple[ReviewItemDisposition, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")
    return tuple(
        _parse_review_item_disposition_payload(
            item,
            field_name=f"{context}[{index}]",
            reviewer=reviewer,
            allowed_same_status=allowed_same_status,
            is_plan_review=is_plan_review,
        )
        for index, item in enumerate(value)
    )


def _finalize_parsed_review(
    *,
    state: str,
    summary: str,
    blocking_items: tuple[ApprovedFollowup, ...],
    followups: ApprovedFollowups,
    dispositions: tuple[ReviewItemDisposition, ...],
    raw_dispositions_text: str = "",
) -> ParsedReview:
    if state == "blocking" and followups.future:
        followups = ApprovedFollowups(same_pr=followups.same_pr, future=())
    if state == "blocking" and any(item.disposition == "future" for item in dispositions):
        raise AgentLoopError(
            "Blocking reviews may not downgrade prior unresolved items to Future follow-ups."
        )
    if state == "approved":
        active_dispositions = [
            item.disposition for item in dispositions if item.disposition in {"blocking", "same-pr"}
        ]
        if blocking_items or followups.same_pr or active_dispositions:
            raise AgentLoopError(
                "Approved reviews must be fully complete for this round. Do not use "
                "`approved` when blocking issues, Same-PR follow-ups, or any prior unresolved "
                "item stays `still blocking` or `same-pr`."
            )
    return ParsedReview(
        state=state,
        summary=summary,
        blocking_items=blocking_items,
        followups=followups,
        dispositions=dispositions,
        raw_dispositions_text=raw_dispositions_text,
    )


def _finalize_parsed_plan_review(
    *,
    state: str,
    summary: str,
    items: PlanReviewItems,
    dispositions: tuple[ReviewItemDisposition, ...],
    raw_dispositions_text: str = "",
) -> ParsedPlanReview:
    if state == "blocking" and items.future:
        items = PlanReviewItems(blocking=items.blocking, same_plan=items.same_plan, future=())
    if state == "blocking" and any(item.disposition == "future" for item in dispositions):
        raise AgentLoopError(
            "Blocking plan reviews may not downgrade prior unresolved plan items to Future follow-ups."
        )
    if state == "approved":
        active_dispositions = [
            item.disposition for item in dispositions if item.disposition in {"blocking", "same-plan"}
        ]
        if items.blocking or items.same_plan or active_dispositions:
            raise AgentLoopError(
                "Approved plan reviews must be fully complete for this planning round. "
                "Do not use `approved` when blocking plan issues, Same-plan follow-ups, "
                "or carried-forward plan items remain active."
            )
    return ParsedPlanReview(
        state=state,
        summary=summary,
        items=items,
        dispositions=dispositions,
        raw_dispositions_text=raw_dispositions_text,
    )


def _structured_followups(items: tuple[str, ...], *, reviewer: str) -> tuple[ApprovedFollowup, ...]:
    return tuple(ApprovedFollowup(reviewer=reviewer, text=item) for item in items)


def _normalized_review_item_text(text: str) -> str:
    normalized = text.strip()
    normalized = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)+", "", normalized)
    normalized = normalized.strip("`*_ \t\r\n")
    normalized = re.sub(r"`([^`]+)`", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip("`*_ \t\r\n.,;:!?-")
    return normalized.casefold()


def _dedupe_followups_against(
    items: tuple[ApprovedFollowup, ...],
    *authoritative_groups: tuple[ApprovedFollowup, ...],
) -> tuple[ApprovedFollowup, ...]:
    authoritative_texts = {
        normalized
        for group in authoritative_groups
        for item in group
        if (normalized := _normalized_review_item_text(item.text))
    }
    if not authoritative_texts:
        return items
    return tuple(
        item for item in items if _normalized_review_item_text(item.text) not in authoritative_texts
    )


def _dedupe_plan_review_items(items: PlanReviewItems) -> PlanReviewItems:
    same_plan = _dedupe_followups_against(items.same_plan, items.blocking)
    future = _dedupe_followups_against(items.future, items.blocking, same_plan)
    if same_plan is items.same_plan and future is items.future:
        return items
    return PlanReviewItems(blocking=items.blocking, same_plan=same_plan, future=future)


def _dedupe_pr_review_items(
    blocking_items: tuple[ApprovedFollowup, ...],
    followups: ApprovedFollowups,
) -> ApprovedFollowups:
    same_pr = _dedupe_followups_against(followups.same_pr, blocking_items)
    future = _dedupe_followups_against(followups.future, blocking_items, same_pr)
    if same_pr is followups.same_pr and future is followups.future:
        return followups
    return ApprovedFollowups(same_pr=same_pr, future=future)


def parse_structured_pr_review(text: str, *, reviewer: str) -> ParsedReview | None:
    payload = _extract_structured_pr_review_payload(text)
    if payload is None:
        return None
    _require_supported_schema_version(payload)
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "pr_review":
        raise AgentLoopError("Structured response kind mismatch: expected `pr_review`.")
    _expect_exact_keys(
        payload,
        context="pr_review",
        required={
            "schema_version",
            "kind",
            "state",
            "summary",
            "prior_item_dispositions",
        },
        optional={"blocking_items", "same_pr_followups", "future_followups"},
    )
    state = _expect_state(payload["state"], context="pr_review.state")
    summary = review_freeform_summary_text(
        _expect_non_empty_string(payload["summary"], context="pr_review.summary")
    )
    blocking_items = _expect_optional_string_list(
        payload,
        "blocking_items",
        context="pr_review.blocking_items",
        item_context="pr_review.blocking_items",
    )
    same_pr_followups = _expect_optional_string_list(
        payload,
        "same_pr_followups",
        context="pr_review.same_pr_followups",
        item_context="pr_review.same_pr_followups",
    )
    future_followups = _expect_optional_string_list(
        payload,
        "future_followups",
        context="pr_review.future_followups",
        item_context="pr_review.future_followups",
    )
    dispositions = _expect_disposition_list(
        payload["prior_item_dispositions"],
        context="pr_review.prior_item_dispositions",
        reviewer=reviewer,
        allowed_same_status="same-pr",
        is_plan_review=False,
    )
    structured_blocking_items = _structured_followups(blocking_items, reviewer=reviewer)
    followups = _dedupe_pr_review_items(
        structured_blocking_items,
        ApprovedFollowups(
            same_pr=_structured_followups(same_pr_followups, reviewer=reviewer),
            future=_structured_followups(future_followups, reviewer=reviewer),
        ),
    )
    return _finalize_parsed_review(
        state=state,
        summary=summary,
        blocking_items=structured_blocking_items,
        followups=followups,
        dispositions=dispositions,
    )


def parse_structured_plan_review(text: str, *, reviewer: str) -> ParsedPlanReview | None:
    payload = _extract_structured_plan_review_payload(text)
    if payload is None:
        return None
    _require_supported_schema_version(payload)
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "plan_review":
        raise AgentLoopError("Structured response kind mismatch: expected `plan_review`.")
    _expect_exact_keys(
        payload,
        context="plan_review",
        required={
            "schema_version",
            "kind",
            "state",
            "summary",
            "prior_plan_item_dispositions",
        },
        optional={"blocking_plan_issues", "same_plan_followups", "future_followups"},
    )
    state = _expect_state(payload["state"], context="plan_review.state")
    summary = review_freeform_summary_text(
        _expect_non_empty_string(payload["summary"], context="plan_review.summary")
    )
    blocking_items = _expect_optional_string_list(
        payload,
        "blocking_plan_issues",
        context="plan_review.blocking_plan_issues",
        item_context="plan_review.blocking_plan_issues",
    )
    same_plan_followups = _expect_optional_string_list(
        payload,
        "same_plan_followups",
        context="plan_review.same_plan_followups",
        item_context="plan_review.same_plan_followups",
    )
    future_followups = _expect_optional_string_list(
        payload,
        "future_followups",
        context="plan_review.future_followups",
        item_context="plan_review.future_followups",
    )
    dispositions = _expect_disposition_list(
        payload["prior_plan_item_dispositions"],
        context="plan_review.prior_plan_item_dispositions",
        reviewer=reviewer,
        allowed_same_status="same-plan",
        is_plan_review=True,
    )
    items = _dedupe_plan_review_items(
        PlanReviewItems(
            blocking=_structured_followups(blocking_items, reviewer=reviewer),
            same_plan=_structured_followups(same_plan_followups, reviewer=reviewer),
            future=_structured_followups(future_followups, reviewer=reviewer),
        )
    )
    return _finalize_parsed_plan_review(
        state=state,
        summary=summary,
        items=items,
        dispositions=dispositions,
    )


def validate_structured_coder_followup(text: str) -> StructuredCoderFollowup | None:
    payload = _extract_structured_coder_followup_payload(text)
    if payload is None:
        return None
    _require_supported_schema_version(payload)
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "coder_followup":
        raise AgentLoopError("Structured response kind mismatch: expected `coder_followup`.")
    _expect_exact_keys(
        payload,
        context="coder_followup",
        required={
            "schema_version",
            "kind",
            "state",
            "summary",
            "addressed_items",
            "remaining_items",
            "human_requirements",
        },
        optional={
            "addressed_item_notes",
            "remaining_item_notes",
            "tests_run",
            "disputed_items",
            "dispute_evidence",
        },
    )
    human_requirements_payload = _expect_object(
        payload["human_requirements"],
        context="coder_followup.human_requirements",
    )
    _expect_exact_keys(
        human_requirements_payload,
        context="coder_followup.human_requirements",
        required={"addressed_ids", "checked_discussion_directly"},
    )
    tests_run_value = payload.get("tests_run")
    tests_run = (
        _expect_string_list(
            tests_run_value,
            context="coder_followup.tests_run",
            item_context="coder_followup.tests_run",
        )
        if tests_run_value is not None
        else None
    )
    addressed_items = _expect_item_id_list(
        payload["addressed_items"],
        context="coder_followup.addressed_items",
    )
    remaining_items = _expect_item_id_list(
        payload["remaining_items"],
        context="coder_followup.remaining_items",
    )
    disputed_items = _expect_item_id_list(
        payload.get("disputed_items", []),
        context="coder_followup.disputed_items",
    )
    addressed_item_notes = _expect_item_note_map(
        payload.get("addressed_item_notes", {}),
        context="coder_followup.addressed_item_notes",
        allowed_item_ids=set(addressed_items),
        allowed_context="coder_followup.addressed_items",
    )
    remaining_item_notes = _expect_item_note_map(
        payload.get("remaining_item_notes", {}),
        context="coder_followup.remaining_item_notes",
        allowed_item_ids=set(remaining_items),
        allowed_context="coder_followup.remaining_items",
    )
    dispute_evidence = _expect_item_note_map(
        payload.get("dispute_evidence", {}),
        context="coder_followup.dispute_evidence",
        allowed_item_ids=set(disputed_items),
        allowed_context="coder_followup.disputed_items",
    )
    missing_evidence = sorted(item_id for item_id in disputed_items if not dispute_evidence.get(item_id))
    if missing_evidence:
        raise AgentLoopError(
            "Coder dispute must include non-empty evidence for each disputed item. "
            "Missing evidence for: " + ", ".join(missing_evidence)
        )
    all_classified = [*addressed_items, *remaining_items, *disputed_items]
    duplicates = sorted({item_id for item_id in all_classified if all_classified.count(item_id) > 1})
    if duplicates:
        raise AgentLoopError(
            "Coder follow-up listed unresolved reviewer item IDs more than once: "
            + ", ".join(duplicates)
        )
    return StructuredCoderFollowup(
        schema_version=1,
        kind="coder_followup",
        state=_expect_state(payload["state"], context="coder_followup.state"),
        summary=_expect_non_empty_string(payload["summary"], context="coder_followup.summary"),
        addressed_items=addressed_items,
        remaining_items=remaining_items,
        human_requirements=StructuredHumanRequirementsPayload(
            addressed_ids=_expect_requirement_id_list(
                human_requirements_payload["addressed_ids"],
                context="coder_followup.human_requirements.addressed_ids",
            ),
            checked_discussion_directly=_expect_bool(
                human_requirements_payload["checked_discussion_directly"],
                context="coder_followup.human_requirements.checked_discussion_directly",
            ),
        ),
        addressed_item_notes=addressed_item_notes,
        remaining_item_notes=remaining_item_notes,
        tests_run=tests_run,
        disputed_items=disputed_items,
        dispute_evidence=dispute_evidence,
    )


def validate_structured_plan_revision(text: str) -> StructuredPlanRevision | None:
    payload = _extract_structured_plan_revision_payload(text)
    if payload is None:
        return None
    _require_supported_schema_version(payload)
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "plan_revision":
        raise AgentLoopError("Structured response kind mismatch: expected `plan_revision`.")
    _expect_exact_keys(
        payload,
        context="plan_revision",
        required={
            "schema_version",
            "kind",
            "state",
            "summary",
            "prior_plan_item_dispositions",
            "plan_steps",
        },
    )
    state = _expect_non_empty_string(payload["state"], context="plan_revision.state")
    if state != "blocking":
        raise AgentLoopError("plan_revision.state must be `blocking`.")
    summary = _expect_non_empty_string(payload["summary"], context="plan_revision.summary")
    dispositions = _expect_disposition_list(
        payload["prior_plan_item_dispositions"],
        context="plan_revision.prior_plan_item_dispositions",
        reviewer="coder",
        allowed_same_status="same-plan",
        is_plan_review=True,
    )
    plan_steps = _expect_string_list(
        payload["plan_steps"],
        context="plan_revision.plan_steps",
        item_context="plan_revision.plan_steps",
        min_length=1,
    )
    return StructuredPlanRevision(
        schema_version=1,
        kind="plan_revision",
        state="blocking",
        summary=summary,
        prior_plan_item_dispositions=dispositions,
        plan_steps=plan_steps,
    )


def validate_structured_plan_state(text: str) -> StructuredPlanState | None:
    payload = _extract_structured_plan_state_payload(text)
    if payload is None:
        return None
    if "schema_version" in payload:
        _require_supported_schema_version(payload)
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "plan_state":
        raise AgentLoopError("Structured response kind mismatch: expected `plan_state`.")
    _expect_exact_keys(
        payload,
        context="plan_state",
        required={"kind", "summary", "plan_steps"},
        optional={"schema_version", "state"},
    )
    if "state" in payload:
        _expect_state(payload["state"], context="plan_state.state")
    return StructuredPlanState(
        schema_version=int(payload.get("schema_version", 1)),
        kind="plan_state",
        state=parse_plan_state(text),
        summary=_expect_non_empty_string(payload["summary"], context="plan_state.summary"),
        plan_steps=_expect_string_list(
            payload["plan_steps"],
            context="plan_state.plan_steps",
            item_context="plan_state.plan_steps",
            min_length=1,
        ),
    )


def _collect_section_items(
    text: str,
    *,
    sections: Sequence[tuple[re.Pattern[str], list[ApprovedFollowup]]],
    empty_item_re: re.Pattern[str],
    reviewer: str,
) -> None:
    def heading_level(line: str) -> int | None:
        match = HEADING_LEVEL_RE.match(line)
        if not match:
            return None
        return len(match.group(1))

    def normalize_item_text(lines: list[str]) -> str:
        trimmed = list(lines)
        while trimmed and not trimmed[0].strip():
            trimmed.pop(0)
        while trimmed and not trimmed[-1].strip():
            trimmed.pop()
        if not trimmed:
            return ""
        common_indent = min(
            len(line) - len(line.lstrip(" "))
            for line in trimmed
            if line.strip()
        )
        if common_indent:
            trimmed = [line[common_indent:] if line.strip() else "" for line in trimmed]

        rendered: list[str] = []
        paragraph: list[str] = []
        fence_marker: str | None = None

        def flush_paragraph() -> None:
            if paragraph:
                rendered.append(" ".join(part.strip() for part in paragraph))
                paragraph.clear()

        for raw_line in trimmed:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                flush_paragraph()
                if rendered and rendered[-1] != "":
                    rendered.append("")
                continue

            fence_match = re.match(r"^\s*(```+|~~~+)", line)
            if fence_match:
                flush_paragraph()
                rendered.append(line)
                marker = fence_match.group(1)
                if fence_marker == marker:
                    fence_marker = None
                elif fence_marker is None:
                    fence_marker = marker
                continue
            if fence_marker is not None:
                rendered.append(line)
                continue

            if HEADING_LEVEL_RE.match(line):
                flush_paragraph()
                rendered.append(line)
                continue

            if re.match(r"^\s{2,}(?:[-*+]\s+|\d+[.)]\s+)", line) or stripped.startswith(">"):
                flush_paragraph()
                rendered.append(line)
                continue

            paragraph.append(stripped)

        flush_paragraph()
        while rendered and rendered[-1] == "":
            rendered.pop()
        return "\n".join(rendered).strip()

    def section_bucket(line: str) -> list[ApprovedFollowup] | None:
        for pattern, bucket in sections:
            if pattern.match(line):
                return bucket
        return None

    active: list[ApprovedFollowup] | None = None
    current: list[str] = []
    active_heading_level: int | None = None

    def flush_current() -> None:
        if active is not None and current:
            item = normalize_item_text(current)
            if item and not empty_item_re.match(item):
                active.append(ApprovedFollowup(reviewer=reviewer, text=item))
            current.clear()

    for line in text.splitlines():
        next_active = section_bucket(line)
        if next_active is not None:
            flush_current()
            active = next_active
            active_heading_level = heading_level(line)
            continue
        if active is None:
            continue

        next_heading_level = heading_level(line)
        if next_heading_level is not None and active_heading_level is not None and next_heading_level <= active_heading_level:
            flush_current()
            active = None
            active_heading_level = None
            continue
        if HTML_COMMENT_RE.match(line) or SIGNATURE_RE.match(line):
            flush_current()
            active = None
            active_heading_level = None
            continue
        if THEMATIC_BREAK_RE.match(line):
            flush_current()
            continue
        bullet = BULLET_RE.match(line)
        if bullet:
            flush_current()
            current.append(bullet.group("text"))
            continue
        if next_heading_level is not None:
            flush_current()
            current.append(line.rstrip())
            continue
        if current or line.strip():
            current.append(line.rstrip())

    flush_current()


def _extract_section_text(text: str, *, heading_re: re.Pattern[str]) -> str:
    active = False
    lines: list[str] = []
    for line in text.splitlines():
        if heading_re.match(line):
            active = True
            continue
        if not active:
            continue
        if ANY_HEADING_RE.match(line) or HTML_COMMENT_RE.match(line) or SIGNATURE_RE.match(line):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def parse_approved_followups(text: str, *, reviewer: str) -> ApprovedFollowups:
    """Extract same-PR and future follow-ups from an approved review."""
    same_pr: list[ApprovedFollowup] = []
    future: list[ApprovedFollowup] = []
    _collect_section_items(
        text,
        sections=(
            (SAME_PR_FOLLOWUP_HEADING_RE, same_pr),
            (FUTURE_FOLLOWUP_HEADING_RE, future),
            (LEGACY_FOLLOWUP_HEADING_RE, future),
        ),
        empty_item_re=EMPTY_FOLLOWUP_RE,
        reviewer=reviewer,
    )
    return ApprovedFollowups(same_pr=tuple(same_pr), future=tuple(future))


def parse_pr_blocking_items(text: str, *, reviewer: str) -> tuple[ApprovedFollowup, ...]:
    blocking: list[ApprovedFollowup] = []
    _collect_section_items(
        text,
        sections=((BLOCKING_ISSUES_HEADING_RE, blocking),),
        empty_item_re=EMPTY_FOLLOWUP_RE,
        reviewer=reviewer,
    )
    return tuple(blocking)


def parse_plan_review_items(text: str, *, reviewer: str) -> PlanReviewItems:
    blocking: list[ApprovedFollowup] = []
    same_plan: list[ApprovedFollowup] = []
    future: list[ApprovedFollowup] = []
    _collect_section_items(
        text,
        sections=(
            (BLOCKING_PLAN_ISSUES_HEADING_RE, blocking),
            (SAME_PLAN_FOLLOWUP_HEADING_RE, same_plan),
            (FUTURE_FOLLOWUP_HEADING_RE, future),
        ),
        empty_item_re=EMPTY_PLAN_SECTION_RE,
        reviewer=reviewer,
    )
    return _dedupe_plan_review_items(
        PlanReviewItems(
            blocking=tuple(blocking),
            same_plan=tuple(same_plan),
            future=tuple(future),
        )
    )


def _disposition_re(*same_statuses: str) -> re.Pattern[str]:
    same_pattern = "|".join(same_statuses)
    status_pattern = (
        r"resolved|"
        r"(?:still\s+)?blocking|"
        rf"(?:still\s+)?(?:{same_pattern})|"
        r"(?:downgraded\s+to\s+)?future follow[- ]up"
    )
    return re.compile(
        r"^\s*\[?(?P<item_id>[A-Za-z0-9][A-Za-z0-9._-]*)\]?\s*"
        r"(?:"
        r"(?:(?P<label>.+?)\s*->\s*)"
        r"|"
        r"(?:(?:->|:)?\s*)"
        r")"
        rf"(?P<status>{status_pattern})"
        r"(?:\s*:\s*(?P<note>.+))?\s*$",
        re.I,
    )


def _normalize_disposition(status: str, *, same_status: str) -> str:
    normalized = " ".join(status.lower().split())
    normalized = normalized.replace("same pr", "same-pr").replace("same plan", "same-plan")
    normalized = normalized.replace("follow up", "follow-up")
    if normalized == "resolved":
        return "resolved"
    if normalized.endswith("blocking"):
        return "blocking"
    if normalized.endswith(same_status):
        return same_status
    if normalized.endswith("future follow-up"):
        return "future"
    raise AgentLoopError(f"Unsupported unresolved item disposition: {status}")


def _active_disposition_has_empty_note(
    disposition: str,
    note: str | None,
    *,
    same_status: str,
    is_plan_review: bool,
) -> bool:
    if disposition == "resolved" or not note:
        return False

    same_status_pattern = re.escape(same_status).replace(r"\-", "[- ]")
    blocking_phrases = [r"no blocking issues?"]
    if is_plan_review:
        blocking_phrases.append(r"no blocking plan issues?")

    empty_note_res = {
        "blocking": _empty_placeholder_re(*blocking_phrases),
        same_status: _empty_placeholder_re(rf"no {same_status_pattern} follow[- ]?ups?"),
        "future": _empty_placeholder_re(r"no future follow[- ]?ups?", r"no follow[- ]?ups?"),
    }
    empty_note_re = empty_note_res.get(disposition)
    return bool(empty_note_re and empty_note_re.match(note))


def _parse_unresolved_item_dispositions(
    text: str,
    *,
    reviewer: str,
    heading_re: re.Pattern[str],
    empty_item_re: re.Pattern[str],
    disposition_re: re.Pattern[str],
    same_status: str,
    error_message: str,
    is_plan_review: bool,
) -> tuple[ReviewItemDisposition, ...]:
    dispositions: list[ReviewItemDisposition] = []
    active = False
    section_heading: str | None = None

    for line_number, line in enumerate(text.splitlines(), start=1):
        if heading_re.match(line):
            active = True
            section_heading = line.strip()
            continue
        if not active:
            continue
        if ANY_HEADING_RE.match(line) or HTML_COMMENT_RE.match(line) or SIGNATURE_RE.match(line):
            active = False
            continue
        if not line.strip():
            continue
        bullet = BULLET_RE.match(line)
        if not bullet:
            continue
        entry = bullet.group("text")
        if empty_item_re.match(entry):
            continue
        match = disposition_re.match(entry)
        if not match:
            raise AgentLoopError(
                f"{error_message} In section `{section_heading or 'unknown section'}`, "
                f"line {line_number}: `{entry}`."
            )
        note = match.group("note")
        normalized_note = note.strip() if note else None
        disposition = _normalize_disposition(match.group("status"), same_status=same_status)
        if _active_disposition_has_empty_note(
            disposition,
            normalized_note,
            same_status=same_status,
            is_plan_review=is_plan_review,
        ):
            raise AgentLoopError(
                f"{error_message} In section `{section_heading or 'unknown section'}`, "
                f"line {line_number}: `{entry}`."
            )
        dispositions.append(
            ReviewItemDisposition(
                item_id=match.group("item_id"),
                reviewer=reviewer,
                disposition=disposition,
                note=normalized_note,
            )
        )

    return tuple(dispositions)


def parse_unresolved_item_dispositions(text: str, *, reviewer: str) -> tuple[ReviewItemDisposition, ...]:
    """Extract structured prior-item dispositions from a review."""
    return _parse_unresolved_item_dispositions(
        text,
        reviewer=reviewer,
        heading_re=PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE,
        empty_item_re=EMPTY_FOLLOWUP_RE,
        disposition_re=_disposition_re(r"same[- ]pr"),
        same_status="same-pr",
        error_message=(
            "Invalid prior unresolved item disposition. Use bullets like "
            "`- [item-1] resolved`, `- [item-2] still blocking`, "
            "`- [item-3] same-pr`, or `- [item-4] future follow-up: reason`. "
            "Active unresolved dispositions must describe remaining work; use "
            "`resolved` when nothing remains."
        ),
        is_plan_review=False,
    )


def parse_plan_item_dispositions(text: str, *, reviewer: str) -> tuple[ReviewItemDisposition, ...]:
    """Extract structured prior plan-item dispositions from a plan review."""
    return _parse_unresolved_item_dispositions(
        text,
        reviewer=reviewer,
        heading_re=PRIOR_UNRESOLVED_PLAN_ITEM_DISPOSITIONS_HEADING_RE,
        empty_item_re=EMPTY_PLAN_SECTION_RE,
        disposition_re=_disposition_re(r"same[- ]plan"),
        same_status="same-plan",
        error_message=(
            "Invalid prior unresolved plan item disposition. Use bullets like "
            "`- [item-1] resolved`, `- [item-2] still blocking`, "
            "`- [item-3] same-plan`, or `- [item-4] future follow-up: reason`. "
            "Active unresolved dispositions must describe remaining work; use "
            "`resolved` when nothing remains."
        ),
        is_plan_review=True,
    )


def parse_review(text: str, *, reviewer: str) -> ParsedReview:
    """Parse a review, including state, follow-ups, and prior-item dispositions."""
    state = parse_agent_state(text)
    summary = review_freeform_summary_text(text)
    blocking_items = parse_pr_blocking_items(text, reviewer=reviewer)
    followups = parse_approved_followups(text, reviewer=reviewer)
    followups = _dedupe_pr_review_items(blocking_items, followups)
    dispositions = parse_unresolved_item_dispositions(text, reviewer=reviewer)
    return _finalize_parsed_review(
        state=state,
        summary=summary,
        blocking_items=blocking_items,
        followups=followups,
        dispositions=dispositions,
        raw_dispositions_text=_extract_section_text(
            text, heading_re=PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE
        ),
    )


def parse_pr_review(text: str, *, reviewer: str) -> ParsedReview:
    parsed = parse_structured_pr_review(text, reviewer=reviewer)
    if parsed is not None:
        return parsed
    raise AgentLoopError("Agent response did not use the required structured format.")


def parse_plan_review(text: str, *, reviewer: str) -> ParsedPlanReview:
    """Parse a plan review, including state, structured plan items, and dispositions."""
    parsed = parse_structured_plan_review(text, reviewer=reviewer)
    if parsed is not None:
        return parsed
    raise AgentLoopError("Agent response did not use the required structured format.")


def parse_non_blocking_followups(text: str, *, reviewer: str) -> list[ApprovedFollowup]:
    """Extract legacy non-blocking follow-ups as future follow-ups."""
    return list(parse_approved_followups(text, reviewer=reviewer).future)


def parse_structured_discuss_review(
    text: str, *, reviewer: str, round_number: int = 1
) -> ParsedDiscussReview | None:
    payload = _extract_structured_discuss_review_payload(text)
    if payload is None:
        return None
    _require_supported_schema_version(payload)
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "discuss_review":
        raise AgentLoopError("Structured response kind mismatch: expected `discuss_review`.")
    _expect_exact_keys(
        payload,
        context="discuss_review",
        required={"schema_version", "kind", "outcome", "rationale"},
        optional={"split_proposals", "rebuttal"},
    )
    outcome = _expect_non_empty_string(payload["outcome"], context="discuss_review.outcome")
    if outcome not in DISCUSS_OUTCOME_VALUES:
        rendered = ", ".join(sorted(DISCUSS_OUTCOME_VALUES))
        raise AgentLoopError(f"discuss_review.outcome must be one of: {rendered}")
    rationale = _expect_non_empty_string(payload["rationale"], context="discuss_review.rationale")
    split_proposals = _expect_optional_string_list(
        payload,
        "split_proposals",
        context="discuss_review.split_proposals",
        item_context="discuss_review.split_proposals",
    )
    if outcome == "split" and not split_proposals:
        raise AgentLoopError(
            "discuss_review.split_proposals must be non-empty when outcome is `split`."
        )
    rebuttal = None
    if "rebuttal" in payload:
        rebuttal = _expect_non_empty_string(payload["rebuttal"], context="discuss_review.rebuttal")
    if round_number > 1 and rebuttal is None:
        raise AgentLoopError("discuss_review.rebuttal is required for debate rounds.")
    return ParsedDiscussReview(
        outcome=outcome,
        rationale=rationale,
        split_proposals=split_proposals,
        reviewer=reviewer,
        rebuttal=rebuttal,
    )


def validate_structured_discuss_review(
    text: str, *, reviewer: str, round_number: int = 1
) -> ParsedDiscussReview:
    parsed = parse_structured_discuss_review(text, reviewer=reviewer, round_number=round_number)
    if parsed is not None:
        return parsed
    raise AgentLoopError("Discuss review did not use the required structured format.")
