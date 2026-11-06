"""Shared response validation and envelope primitives."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from .errors import AgentLoopError
from .protocol_markers import sanitize_historical_text


# Preserve the existing diagnostic channel across the ownership move.
_logger = logging.getLogger("coding_review_agent_loop.protocol")

PUBLIC_RESPONSE_MARKER = "=== AGENT_LOOP_PUBLIC_RESPONSE_BELOW ==="

HUMAN_REQUIREMENTS_ADDRESSED_MARKER = "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->"

PLAN_STATE_RE = re.compile(r"<!--\s*AGENT_PLAN_STATE:\s*(approved|blocking)\s*-->", re.I)

HUMAN_REQUIREMENTS_RESOLVED_RE = re.compile(
    r"<!--\s*HUMAN_REQUIREMENTS_RESOLVED\s*-->",
    re.I,
)

HUMAN_REQUIREMENTS_ADDRESSED_RE = re.compile(
    re.escape(HUMAN_REQUIREMENTS_ADDRESSED_MARKER),
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

HUMAN_REQUIREMENTS_HEADING_RE = _followup_heading_re(r"human requirements")

ANY_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")

HTML_COMMENT_RE = re.compile(r"^\s*<!--.*-->\s*$")

SIGNATURE_RE = re.compile(r"^\s*--\s+\S")

BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(?P<text>.+?)\s*$")

HUMAN_REQUIREMENT_STABLE_ID_RE = re.compile(r"^hr-[0-9a-f]{64}$", re.I)

PARSE_DEGRADATION_FIELD_CHARS = 200

PARSE_DEGRADATION_PREVIEW_CHARS = 120

PARSE_DEGRADATION_OUTCOMES = frozenset(
    {
        "normalized-to-changed",
        "degraded-to-undetermined",
        "claim-dropped",
        "citation-dropped",
        "citations-decoded-from-string",
    }
)

# Records beyond this count are summarized rather than rendered; the
# rendering module re-exports it under the same name.
PARSE_DEGRADATION_RENDER_LIMIT = 8

def _bounded_single_line(text: str, limit: int) -> str:
    # Records are rendered into round comments, so no angle bracket survives:
    # an HTML comment opener can never reach a body as a protocol record.
    flattened = " ".join(
        sanitize_historical_text(text)  # shape-check: fatal:authentication-or-forgery
        .replace("`", "'")
        .replace("<", "\u2039")
        .replace(">", "\u203a")
        .split()
    )
    if len(flattened) <= limit:
        return flattened
    return flattened[:limit] + "\u2026"

@dataclass(frozen=True)
class ParseDegradation:
    """A bounded, parser-derived record of one degraded element (#924).

    Records are produced only by parser code from the element it reads; no
    agent schema carries them, so an agent can never author one.
    """

    element_path: str
    rule: str
    observed_preview: str
    outcome: str

    @classmethod
    def build(cls, *, element_path: str, rule: str, observed: object, outcome: str) -> "ParseDegradation":
        observed_text = observed if isinstance(observed, str) else json.dumps(observed, sort_keys=True, default=str)
        return cls(
            element_path=_bounded_single_line(element_path, PARSE_DEGRADATION_FIELD_CHARS),  # shape-check: fatal:authentication-or-forgery
            rule=_bounded_single_line(rule, PARSE_DEGRADATION_FIELD_CHARS),  # shape-check: fatal:authentication-or-forgery
            observed_preview=_bounded_single_line(observed_text, PARSE_DEGRADATION_PREVIEW_CHARS),  # shape-check: fatal:authentication-or-forgery
            outcome=_bounded_single_line(outcome, PARSE_DEGRADATION_FIELD_CHARS),  # shape-check: fatal:authentication-or-forgery
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "element_path": sanitize_historical_text(self.element_path),  # shape-check: fatal:authentication-or-forgery
            "rule": sanitize_historical_text(self.rule),  # shape-check: fatal:authentication-or-forgery
            "observed_preview": sanitize_historical_text(self.observed_preview),  # shape-check: fatal:authentication-or-forgery
            "outcome": sanitize_historical_text(self.outcome),  # shape-check: fatal:authentication-or-forgery
        }

def parse_degradation_payload(value: object) -> ParseDegradation:
    """Strictly decode one orchestrator-written record for metadata rehydration."""
    if not isinstance(value, Mapping) or set(value) != {"element_path", "rule", "observed_preview", "outcome"}:
        raise AgentLoopError("Parse degradation record must have exactly the documented keys.")  # shape-check: fatal:orchestrator-authored
    fields: dict[str, str] = {}
    for key, limit in (
        ("element_path", PARSE_DEGRADATION_FIELD_CHARS),
        ("rule", PARSE_DEGRADATION_FIELD_CHARS),
        ("observed_preview", PARSE_DEGRADATION_PREVIEW_CHARS),
        ("outcome", PARSE_DEGRADATION_FIELD_CHARS),
    ):
        item = value[key]
        if not isinstance(item, str) or not item.strip() or len(item) > limit + 1:
            raise AgentLoopError(f"Parse degradation record field {key} is invalid.")  # shape-check: fatal:orchestrator-authored
        fields[key] = _bounded_single_line(item, limit)  # shape-check: fatal:authentication-or-forgery
    if fields["outcome"] not in PARSE_DEGRADATION_OUTCOMES:
        raise AgentLoopError("Parse degradation record outcome is invalid.")  # shape-check: fatal:orchestrator-authored
    return ParseDegradation(**fields)

@dataclass(frozen=True)
class TestObservationCitation:
    """A historical/display citation for a broker or parent test receipt.

    Fresh risk-matrix evidence is never selected with this type.  It remains
    readable for already-published comments and for the ordinary test-report
    display channel, while the deterministic builder creates authoritative
    citations from invocation-local execution handles.
    """

    command: str
    receipt_id: str
    claim: str

@dataclass(frozen=True)
class ParsedHumanRequirementsAcknowledgement:
    marker_present: bool
    section_present: bool
    addressed_ids: tuple[str, ...]
    section_text: str

def _normalize_requirement_label(text: str) -> str:
    stable = HUMAN_REQUIREMENT_STABLE_ID_RE.fullmatch(text.strip())
    if stable:
        return stable.group(0).lower()
    legacy = re.fullmatch(r"\s*Requirement\s+(\d+)\s*", text, re.I)
    if not legacy:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            f"Invalid human requirement label: {text}. Only exact surfaced signed labels like "
            "`hr-<64 hexadecimal characters>` are valid; issue acceptance criteria, reviewer item IDs, reviewer "
            "comments, and arbitrary labels are not signed human requirements."
        )
    return f"Requirement {legacy.group(1)}"

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
        for match in re.finditer(
            r"\b(?:Requirement\s+)?hr-[0-9a-f]{64}\b|\bRequirement\s+\d+\b",
            bullet.group("text"),
            re.I,
        ):
            label = match.group(0)
            stable = re.search(r"hr-[0-9a-f]{64}", label, re.I)
            addressed_ids.append(
                _normalize_requirement_label(stable.group(0) if stable else label)  # shape-check: fatal:no-conservative-reading
            )

    return ParsedHumanRequirementsAcknowledgement(
        marker_present=marker_present,
        section_present=section_present,
        addressed_ids=tuple(addressed_ids),
        section_text="\n".join(section_lines).strip(),
    )

def _expect_object(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AgentLoopError(f"{context} must be a JSON object.")  # shape-check: delegated
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
        raise AgentLoopError(f"{context} is missing required field(s): {', '.join(missing)}")  # shape-check: delegated
    unknown = sorted(keys - required - optional)
    if unknown:
        raise AgentLoopError(f"{context} has unknown field(s): {', '.join(unknown)}")  # shape-check: delegated

def _expect_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentLoopError(f"{context} must be an integer.")  # shape-check: delegated
    return value

def _expect_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise AgentLoopError(f"{context} must be a boolean.")  # shape-check: delegated
    return value

def _expect_non_empty_string(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise AgentLoopError(f"{context} must be a string.")  # shape-check: delegated
    normalized = value.strip()
    if not normalized:
        raise AgentLoopError(f"{context} must be a non-empty string.")  # shape-check: delegated
    return normalized

def _flat_object_defect(value: Mapping[object, object]) -> tuple[str, str] | None:
    """Return `(key, description)` for the first value that cannot render losslessly.

    A renderable object is non-empty, has only non-empty string keys, and every
    value is a non-blank string or a non-empty list of non-blank strings.
    Anything else (nested objects, nested lists, numbers, booleans, null, empty
    strings, empty lists) would change meaning when rendered, so it is a defect.
    """
    if not value:
        return "", "an empty object"
    for key, child in value.items():
        if not isinstance(key, str) or not key.strip():
            return str(key), "a non-string or empty key"
        if isinstance(child, str):
            if not child.strip():
                return key, "an empty string"
            continue
        if isinstance(child, list):
            if not child:
                return key, "an empty list"
            for entry in child:
                if isinstance(entry, bool) or not isinstance(entry, str):
                    return key, "a list with a non-string entry"
                if not entry.strip():
                    return key, "a list with an empty string entry"
            continue
        if isinstance(child, Mapping):
            return key, "a nested object"
        if child is None:
            return key, "null"
        if isinstance(child, bool):
            return key, "a boolean"
        if isinstance(child, (int, float)):
            return key, "a number"
        return key, f"a {type(child).__name__} value"
    return None

def _render_flat_object(value: Mapping[object, object]) -> str:
    """Render an object already known to have no `_flat_object_defect`."""
    pairs: list[str] = []
    for key, child in value.items():
        text = child.strip() if isinstance(child, str) else ", ".join(
            entry.strip() for entry in child
        )
        pairs.append(f"{str(key).strip()}: {text}")
    return "; ".join(pairs)

def _flatten_labelled_object(value: Mapping[object, object], *, context: str) -> str:
    """Render a flat object item to text, or raise naming the offending key."""
    defect = _flat_object_defect(value)
    if defect is not None:
        key, description = defect
        raise AgentLoopError(  # shape-check: delegated
            f"{context} object item must contain only non-empty string or "
            f"string-list values; got {description} for key {key!r}."
        )
    return _render_flat_object(value)

def _expect_string_list(
    value: object,
    *,
    context: str,
    item_context: str,
    min_length: int = 0,
    flatten_object_items: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: delegated
    items: list[object] = []
    for index, item in enumerate(value):
        if flatten_object_items and isinstance(item, Mapping):
            # Opt-in for the architecture_impact list fields only (#1330):
            # flatten before validation so a structured item never needs a
            # model repair pass, and leave an audit line for the normalization.
            item = _flatten_labelled_object(item, context=f"{item_context} at index {index}")  # shape-check: delegated
            _logger.warning(
                "%s at index %d: flattened architecture-impact object item to text",
                item_context,
                index,
            )
        items.append(item)
    rendered = tuple(
        _expect_non_empty_string(item, context=f"{item_context} at index {index}")  # shape-check: delegated
        for index, item in enumerate(items)
    )
    if len(rendered) < min_length:
        raise AgentLoopError(f"{context} must contain at least {min_length} item(s).")  # shape-check: delegated
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
    return _expect_string_list(  # shape-check: delegated
        value,
        context=context,
        item_context=item_context,
        min_length=min_length,
    )

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
        raise AgentLoopError("Structured response must begin with one top-level JSON object.") from exc  # shape-check: fatal:unparseable-envelope
    if not isinstance(payload, dict):
        raise AgentLoopError("Structured response must begin with a JSON object.")  # shape-check: fatal:unparseable-envelope
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
        raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
            f"{context_label} must place <!-- {state_marker_name}: approved|blocking --> after the JSON object."
        )

    before_footer = trailing[: state_match.start()].strip()
    if before_footer:
        if not allow_human_requirements_prefix:
            raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
                f"{context_label} may not include prose between the JSON object and the {state_marker_name} footer."
            )
        parsed_human_requirements = parse_human_requirements_acknowledgement(before_footer)  # shape-check: fatal:no-conservative-reading
        if not parsed_human_requirements.marker_present or not parsed_human_requirements.section_present:
            raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
                f"{context_label} may only include a signed human requirements acknowledgement before the {state_marker_name} footer."
            )

    trailing = trailing[state_match.end() :].lstrip()
    if allow_human_requirements_marker_after_footer and HUMAN_REQUIREMENTS_RESOLVED_RE.match(trailing):
        marker_match = HUMAN_REQUIREMENTS_RESOLVED_RE.match(trailing)
        assert marker_match is not None
        trailing = trailing[marker_match.end() :].lstrip()
    signature_match = re.match(r"^--\s+\S[^\n]*(?:\n)?$", trailing)
    if signature_match is None or trailing[signature_match.end() :].strip():
        raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
            f"{context_label} may not include trailing prose after the JSON footer and signature."
        )

    footer_state = state_match.group(1).lower()
    payload_state = payload.get("state")
    if isinstance(payload_state, str) and payload_state.strip():
        if payload_state.strip() != footer_state:
            raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
                f"{context_label} footer {state_marker_name} must match the payload state."
            )
    return payload

def _require_supported_schema_version(payload: dict[str, object]) -> None:
    if "schema_version" not in payload:
        raise AgentLoopError("Structured response is missing required field: schema_version")  # shape-check: fatal:kind-or-version-mismatch
    version = _expect_int(payload["schema_version"], context="schema_version")  # shape-check: fatal:kind-or-version-mismatch
    if version != 1:
        raise AgentLoopError(f"Unsupported structured response schema_version: {version}")  # shape-check: fatal:kind-or-version-mismatch
