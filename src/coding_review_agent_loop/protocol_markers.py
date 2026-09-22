"""Central registry and provenance-aware carriers for durable agent markers.

This module intentionally has no dependency on a marker owner.  Producers own
their schemas, while this leaf owns the outer grammar, canonical wire form,
surface policy, and the distinction between current untrusted prose and
historical text that the orchestrator is deliberately re-rendering.
"""

from __future__ import annotations

import ast
import base64
import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from .errors import AgentLoopError

MarkerStrictness = Literal["well-formed-only", "name-bearing-line", "bare-substring"]

ISSUE_COMMENT_SURFACE = "issue_comment"
PR_COMMENT_SURFACE = "pr_comment"
ISSUE_BODY_SURFACE = "issue_body"
PR_BODY_SURFACE = "pr_body"
REST_CREATE_SURFACE = "rest_create"
REST_PATCH_SURFACE = "rest_patch"
ANY_COMMENT_SURFACE = "any_comment"


def _b64_json(value: object) -> str:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_b64_json(value: str) -> object:
    return json.loads(base64.urlsafe_b64decode(value.encode("ascii")).decode("utf-8"))


def _canonical_b64_json(match: re.Match[str], *, prefix: str) -> str:
    value = _decode_b64_json(match.group("payload"))
    encoded = _b64_json(value)
    return f"<!-- {prefix}: {encoded} -->"


def _canonical_source(match: re.Match[str], *, token: str) -> str:
    value = _decode_b64_json(match.group("payload"))
    return f"<!-- {token} {_b64_json(value)} -->"


def _canonical_compressed_mapping(match: re.Match[str], *, token: str) -> str:
    encoded = match.group("payload")
    packed = base64.urlsafe_b64decode(encoded[3:].encode("ascii")) if encoded.startswith("v1_") else base64.urlsafe_b64decode(encoded.encode("ascii"))
    value = json.loads(zlib.decompress(packed).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mapping required")
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    canonical = "v1_" + base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii")
    return f"<!-- {token}: {canonical} -->"


def _canonical_plain_or_compressed(match: re.Match[str], *, token: str) -> str:
    """Accept records published in the plain form before the compact one."""
    encoded = match.group("payload")
    if not encoded.startswith("v1_"):
        value = _decode_b64_json(encoded)
        return f"<!-- {token}: {_b64_json(value)} -->"
    return _canonical_compressed_mapping(match, token=token)


def _canonical_checkpoint(match: re.Match[str]) -> str:
    return _canonical_plain_or_compressed(match, token="AGENT_PLAN_TOPOLOGY_CHECKPOINT")


def _canonical_raw_json(match: re.Match[str], *, token: str) -> str:
    value = json.loads(match.group("payload"))
    return f"<!-- {token} {json.dumps(value, separators=(',', ':'), sort_keys=True)} -->"


def _canonical_hex(match: re.Match[str], *, token: str) -> str:
    return f"<!-- {token}: {match.group('value').lower()} -->"


def _canonical_split_child(match: re.Match[str]) -> str:
    return f"<!-- AGENT_SPLIT_CHILD: parent={int(match.group('parent'))} key={match.group('key').lower()} -->"


def _canonical_key_value_comment(match: re.Match[str], *, token: str, colon: bool = True) -> str:
    raw = match.group("fields").strip()
    fields: list[tuple[str, str]] = []
    for item in raw.split():
        if "=" not in item:
            raise ValueError("key/value field required")
        key, value = item.split("=", 1)
        if not key or not value:
            raise ValueError("key/value field required")
        fields.append((key, value))
    token_orders = {
        "AGENT_APPROVED_FOLLOWUPS": ("pr", "head", "mode"),
        "AGENT_PLAN_APPROVED_FOLLOWUPS": ("issue", "plan", "mode"),
        "AGENT_SPLIT_UNFILED_WARNING": ("issue", "subject"),
        "AGENT_LOOP_MANAGED_CI_QUALIFIED_V2": (
            "repo", "pr", "base", "protocol", "qualified_head", "reviewers",
            "protection", "nonce", "run_id", "attempt", "generation",
        ),
    }
    ordered_keys = token_orders.get(token)
    if ordered_keys is not None:
        order = {key: index for index, key in enumerate(ordered_keys)}
        fields.sort(key=lambda item: (order.get(item[0], 100), item[0]))
        separator = ": " if colon else " "
        return f"<!-- {token}{separator}" + " ".join(f"{key}={value}" for key, value in fields) + " -->"
    preferred = {
        "issue": 0,
        "subject": 1,
        "pr": 0,
        "head": 1,
        "mode": 2,
        "repo": 1,
        "base": 2,
        "protocol": 3,
        "qualified_head": 4,
        "reviewers": 5,
        "protection": 6,
        "nonce": 7,
        "run_id": 8,
        "attempt": 9,
        "generation": 10,
    }
    fields.sort(key=lambda item: (preferred.get(item[0], 100), item[0]))
    separator = ": " if colon else " "
    return f"<!-- {token}{separator}" + " ".join(f"{key}={value}" for key, value in fields) + " -->"


def _canonical_line_key_value(match: re.Match[str], *, token: str) -> str:
    raw = match.group("fields").strip()
    fields: list[tuple[str, str]] = []
    for item in raw.split():
        if "=" not in item:
            raise ValueError("key/value field required")
        key, value = item.split("=", 1)
        if not key or not value:
            raise ValueError("key/value field required")
        fields.append((key, value))
    token_orders = {
        "AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1": (
            "nonce", "repo", "base", "head", "protection",
            "active_label_event_id", "resume_from", "provenance_head", "generation",
        ),
    }
    order = {key: index for index, key in enumerate(token_orders.get(token, ())) }
    preferred = {"nonce": 0, "repo": 1, "pr": 2, "base": 3, "head": 4, "label": 5, "protocol": 6, "generation": 7, "run_id": 8, "attempt": 9}
    if order:
        preferred = order
    fields.sort(key=lambda item: (preferred.get(item[0], 100), item[0]))
    return token + " " + " ".join(f"{key}={value}" for key, value in fields)


def _b64_definition(token: str, *, surfaces: frozenset[str], codec: str = "b64-json") -> "MarkerDefinition":
    pattern = re.compile(rf"<!--\s*{re.escape(token)}:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", re.I)
    return MarkerDefinition(
        token=token,
        pattern=pattern,
        strictness="well-formed-only",
        codec=codec,
        surfaces=surfaces,
        safe_label=f"[{token.replace('AGENT_', 'protocol ')} record]",
        canonicalizer=lambda match, token=token, codec=codec: (
            _canonical_compressed_mapping(match, token=token)
            if codec == "compressed-mapping"
            else _canonical_plain_or_compressed(match, token=token)
            if codec == "plain-or-compressed-mapping"
            else _canonical_b64_json(match, prefix=token)
        ),
    )


@dataclass(frozen=True)
class MarkerDefinition:
    token: str
    pattern: re.Pattern[str]
    strictness: MarkerStrictness
    codec: str
    surfaces: frozenset[str]
    safe_label: str
    canonicalizer: Callable[[re.Match[str]], str]


_COMMENT = frozenset({ISSUE_COMMENT_SURFACE, PR_COMMENT_SURFACE, ANY_COMMENT_SURFACE})
_ISSUE = frozenset({ISSUE_COMMENT_SURFACE, ISSUE_BODY_SURFACE, REST_CREATE_SURFACE, ANY_COMMENT_SURFACE})
_PR = frozenset({PR_COMMENT_SURFACE, PR_BODY_SURFACE, REST_CREATE_SURFACE, REST_PATCH_SURFACE, ANY_COMMENT_SURFACE})
_ISSUE_ONLY = frozenset({ISSUE_COMMENT_SURFACE, ANY_COMMENT_SURFACE})


def _key_value_definition(token: str, *, surfaces: frozenset[str], strictness: MarkerStrictness = "name-bearing-line") -> MarkerDefinition:
    pattern = re.compile(rf"<!--\s*{re.escape(token)}:\s*(?P<fields>[^\r\n]*?)\s*-->", re.I)
    return MarkerDefinition(
        token=token,
        pattern=pattern,
        strictness=strictness,
        codec="key-value",
        surfaces=surfaces,
        safe_label=f"[{token.replace('AGENT_', 'protocol ')} record]",
        canonicalizer=lambda match, token=token: _canonical_key_value_comment(match, token=token),
    )


def _line_definition(token: str, *, surfaces: frozenset[str], strictness: MarkerStrictness = "bare-substring") -> MarkerDefinition:
    pattern = re.compile(rf"(?m)^[ \t]*{re.escape(token)}[ \t]+(?P<fields>[^\r\n]+?)[ \t]*$", re.I)
    return MarkerDefinition(
        token=token,
        pattern=pattern,
        strictness=strictness,
        codec="key-value-line",
        surfaces=surfaces,
        safe_label=f"[{token.replace('AGENT_', 'protocol ')} audit]",
        canonicalizer=lambda match, token=token: _canonical_line_key_value(match, token=token),
    )


def _make_registry() -> tuple[MarkerDefinition, ...]:
    entries: list[MarkerDefinition] = [
        _b64_definition("AGENT_ISSUE_PR_HANDOFF", surfaces=_ISSUE_ONLY),
        _b64_definition("AGENT_PR_EXPECTED_CLOSING_ISSUES", surfaces=_COMMENT),
        _b64_definition("AGENT_PLAN_EXPECTED_CLOSING_ISSUES", surfaces=_ISSUE_ONLY),
        _b64_definition("AGENT_LOOP_META", surfaces=_COMMENT, codec="compressed-mapping"),
        _b64_definition("AGENT_LOOP_SIDECAR", surfaces=_COMMENT),
        _b64_definition(
            "AGENT_PLAN_VALIDATION_DIAGNOSTIC",
            surfaces=_ISSUE_ONLY,
            codec="compressed-mapping",
        ),
        _b64_definition("AGENT_TYPED_PLAN_STAGES", surfaces=_ISSUE_ONLY),
        _b64_definition("AGENT_EXECUTION_RECOMMENDATION", surfaces=_ISSUE_ONLY),
        _b64_definition("AGENT_RISK_TEST_MATRIX", surfaces=_ISSUE_ONLY),
        _b64_definition("AGENT_DEFERRED_STAGES", surfaces=_ISSUE_ONLY),
        # Compressed since #909; summaries posted earlier stay plain base64.
        _b64_definition(
            "AGENT_PLAN_DECOMPOSITION",
            surfaces=_ISSUE_ONLY,
            codec="plain-or-compressed-mapping",
        ),
        _b64_definition("AGENT_PLAN_EXECUTION_DECISION", surfaces=_ISSUE_ONLY),
        MarkerDefinition(
            token="AGENT_PLAN_TOPOLOGY_CHECKPOINT",
            pattern=re.compile(
                r"<!--\s*AGENT_PLAN_TOPOLOGY_CHECKPOINT:\s*"
                r"(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
                re.I,
            ),
            strictness="well-formed-only",
            codec="compressed-mapping",
            surfaces=_ISSUE_ONLY,
            safe_label="[protocol topology checkpoint record]",
            canonicalizer=_canonical_checkpoint,
        ),
        _b64_definition("AGENT_PLAN_PHASE_IDENTITY", surfaces=_ISSUE),
        _b64_definition("AGENT_PLAN_PHASE_IMPLEMENTATION", surfaces=_ISSUE_ONLY),
        _b64_definition("AGENT_PLAN_ONE_SHOT_IMPL", surfaces=_ISSUE_ONLY),
        _b64_definition("AGENT_CHILD_PLAN_REBIND", surfaces=_ISSUE_ONLY),
        _b64_definition("AGENT_DISCUSS_SPLIT", surfaces=_ISSUE_ONLY),
        MarkerDefinition(
            token="AGENT_DISCUSS_CONSENSUS",
            pattern=re.compile(r"<!--\s*AGENT_DISCUSS_CONSENSUS:\s*(?P<value>[0-9a-f]+)\s*-->", re.I),
            strictness="well-formed-only",
            codec="bare-hex",
            surfaces=_ISSUE_ONLY,
            safe_label="[protocol consensus record]",
            canonicalizer=lambda match: _canonical_hex(match, token="AGENT_DISCUSS_CONSENSUS"),
        ),
        _key_value_definition("AGENT_APPROVED_FOLLOWUPS", surfaces=_PR),
        _key_value_definition("AGENT_PLAN_APPROVED_FOLLOWUPS", surfaces=_ISSUE_ONLY),
        _b64_definition("AGENT_SALVAGE", surfaces=_ISSUE_ONLY),
        MarkerDefinition(
            token="AGENT_MANAGED_CI_INTENT_V2",
            pattern=re.compile(r"<!--\s*AGENT_MANAGED_CI_INTENT_V2\s+(?P<payload>.*?)\s*-->", re.I),
            strictness="well-formed-only",
            codec="raw-sorted-json",
            surfaces=_PR,
            safe_label="[protocol managed-CI intent record]",
            canonicalizer=lambda match: _canonical_raw_json(match, token="AGENT_MANAGED_CI_INTENT_V2"),
        ),
        MarkerDefinition(
            token="AGENT_LOOP_MANAGED_CI_QUALIFIED_V2",
            pattern=re.compile(r"<!--\s*AGENT_LOOP_MANAGED_CI_QUALIFIED_V2\s+(?P<fields>[^\r\n]*?)\s*-->", re.I),
            strictness="name-bearing-line",
            codec="key-value",
            surfaces=_PR,
            safe_label="[protocol managed-CI qualification audit]",
            canonicalizer=lambda match: _canonical_key_value_comment(match, token="AGENT_LOOP_MANAGED_CI_QUALIFIED_V2", colon=False),
        ),
        _line_definition("AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1", surfaces=_PR),
        _b64_definition(
            "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1",
            surfaces=frozenset({PR_COMMENT_SURFACE}),
        ),
        # Cross-surface workflow transaction record (#827).  Prepared and
        # terminal records live only in trusted PR comments.
        MarkerDefinition(
            token="AGENT_WORKFLOW_TRANSACTION",
            pattern=re.compile(
                r"<!--\s*AGENT_WORKFLOW_TRANSACTION:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
                re.I,
            ),
            strictness="well-formed-only",
            codec="b64-json",
            surfaces=frozenset({PR_COMMENT_SURFACE}),
            safe_label="[protocol workflow transaction record]",
            canonicalizer=lambda match: _canonical_b64_json(
                match, prefix="AGENT_WORKFLOW_TRANSACTION"
            ),
        ),
        MarkerDefinition(
            token="AGENT_MANAGED_PR_SOURCE_V1",
            pattern=re.compile(r"<!--\s*AGENT_MANAGED_PR_SOURCE_V1\s+(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", re.I),
            strictness="bare-substring",
            codec="source-b64-json",
            surfaces=_PR,
            safe_label="[protocol managed-PR origin record]",
            canonicalizer=lambda match: _canonical_source(match, token="AGENT_MANAGED_PR_SOURCE_V1"),
        ),
        MarkerDefinition(
            token="AGENT_SPLIT_CHILD",
            pattern=re.compile(r"<!--\s*AGENT_SPLIT_CHILD:\s*parent=(?P<parent>\d+)\s+key=(?P<key>[0-9a-f]{64})\s*-->", re.I),
            strictness="well-formed-only",
            codec="hex/key-value",
            surfaces=frozenset({ISSUE_BODY_SURFACE, REST_CREATE_SURFACE}),
            safe_label="[protocol split-child record]",
            canonicalizer=_canonical_split_child,
        ),
        _b64_definition("AGENT_SPLIT_STAGE_HANDOFF", surfaces=_ISSUE_ONLY),
        MarkerDefinition(
            token="AGENT_SPLIT_UNFILED_WARNING",
            pattern=re.compile(r"<!--\s*AGENT_SPLIT_UNFILED_WARNING:\s*issue=(?P<issue>\d+)\s+subject=(?P<subject>\S+)\s*-->", re.I),
            strictness="name-bearing-line",
            codec="key-value",
            surfaces=_ISSUE_ONLY,
            safe_label="[protocol split-warning record]",
            canonicalizer=lambda match: f"<!-- AGENT_SPLIT_UNFILED_WARNING: issue={int(match.group('issue'))} subject={match.group('subject')} -->",
        ),
    ]
    return tuple(entries)


RESERVED_MARKER_REGISTRY: tuple[MarkerDefinition, ...] = _make_registry()
MARKER_BY_TOKEN = {entry.token: entry for entry in RESERVED_MARKER_REGISTRY}

# These strings are deliberately not durable protocol records.  They are
# public response grammar, configuration/environment names, workflow feature
# probes, or ordinary state markers and therefore remain outside this writer
# boundary.
EXCLUDED_PROTOCOL_LITERALS = frozenset(
    {
        "AGENT_LOOP_MANAGED_CI_V2_PR_ADOPTION",
        "AGENT_LOOP_MANAGED_CI_V2",
        "AGENT_LOOP_MANAGED_CI_UNLABELED_RECOVERY_V1",
        "AGENT_LOOP_PUBLIC_RESPONSE_BELOW",
        "AGENT_LOOP_MANAGED_ACTOR",
        "AGENT_LOOP_WORKDIR",
        "AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS",
        "AGENT_LOOP_INVOCATION_ID",
        "AGENT_LOOP_TEST_BROKER_ENDPOINT",
        "AGENT_LOOP_TEST_BROKER_CAPABILITY",
        "AGENT_LOOP_TEST_BROKER_PROTOCOL",
        "AGENT_STATE",
        "AGENT_PLAN_STATE",
        "AGENT_PR",
        "AGENT_CLARIFY",
        "AGENT_UNAVAILABLE",
        "AGENT_MARKER_RE",
        "AGENT_OUTPUT_RE",
        "AGENT_UNAVAILABLE_CATEGORIES",
    }
)


@dataclass(frozen=True)
class MarkerOccurrence:
    definition: MarkerDefinition
    start: int
    end: int
    text: str


def _all_occurrences(text: str) -> tuple[MarkerOccurrence, ...]:
    found: list[MarkerOccurrence] = []
    fallback: list[MarkerOccurrence] = []
    for definition in RESERVED_MARKER_REGISTRY:
        for match in definition.pattern.finditer(text):
            found.append(MarkerOccurrence(definition, match.start(), match.end(), match.group(0)))
        if definition.strictness == "bare-substring":
            for match in re.finditer(re.escape(definition.token), text, re.I):
                fallback.append(
                    MarkerOccurrence(definition, match.start(), match.end(), match.group(0))
                )
        elif definition.strictness == "name-bearing-line":
            line_pattern = re.compile(rf"(?m)^.*{re.escape(definition.token)}.*$", re.I)
            for match in line_pattern.finditer(text):
                fallback.append(
                    MarkerOccurrence(definition, match.start(), match.end(), match.group(0))
                )
    # Prefer the complete outer record where a bare-substring token overlaps
    # it.  This is important for deterministic historical replacement.
    found.sort(key=lambda item: (item.start, -(item.end - item.start), item.definition.token))
    selected: list[MarkerOccurrence] = []
    for occurrence in found:
        if any(occurrence.start < old.end and old.start < occurrence.end for old in selected):
            continue
        selected.append(occurrence)
    # Fallback spans provide strictness-specific rejection for malformed
    # records, but a complete regular match always wins over its containing
    # line/token fallback.
    for occurrence in sorted(fallback, key=lambda item: (item.start, -(item.end - item.start), item.definition.token)):
        overlaps = [
            old
            for old in selected
            if occurrence.start < old.end and old.start < occurrence.end
        ]
        if not overlaps:
            selected.append(occurrence)
            continue
        if occurrence.definition.strictness != "name-bearing-line":
            continue
        # A name-bearing-line fallback normally replaces the whole line so a
        # malformed record cannot survive. If a valid record already occupies
        # part of that line, retain the strict fallback for each token mention
        # outside the valid span instead of leaking the mention after the
        # valid record is sanitized.
        for match in re.finditer(re.escape(occurrence.definition.token), occurrence.text, re.I):
            token_occurrence = MarkerOccurrence(
                occurrence.definition,
                occurrence.start + match.start(),
                occurrence.start + match.end(),
                match.group(0),
            )
            if any(
                token_occurrence.start < old.end and old.start < token_occurrence.end
                for old in selected
            ):
                continue
            selected.append(token_occurrence)
    return tuple(sorted(selected, key=lambda item: item.start))


def scan_reserved_markers(text: str) -> tuple[MarkerOccurrence, ...]:
    """Return non-overlapping reserved occurrences in source order."""
    return _all_occurrences(text)


def is_complete_marker_occurrence(occurrence: MarkerOccurrence) -> bool:
    """Whether this span is a complete, parseable record rather than a mention.

    Callers use it to separate an agent naming a token in prose, which is safe
    to neutralize, from an attempt to emit a durable record (#891).
    """
    return _is_complete_historical_occurrence(occurrence)


def named_reserved_marker_tokens(text: str) -> tuple[str, ...]:
    """Return the reserved token names that ``text`` mentions, in sorted order.

    Callers use it to log once that untrusted GitHub prose named a protocol
    record, without granting the mention any authority (#891).  A
    ``well-formed-only`` entry has no bare-name fallback in the scanner, so
    match the literal registry names rather than relying on occurrences.
    """
    if not isinstance(text, str) or not text:
        return ()
    lowered = text.lower()
    return tuple(
        sorted(
            {
                definition.token
                for definition in RESERVED_MARKER_REGISTRY
                if definition.token.lower() in lowered
            }
        )
    )


def sanitize_untrusted_prose(text: str) -> str:
    """Neutralize every reserved name in untrusted GitHub text (#891).

    :func:`sanitize_historical_text` replaces registered *occurrences*, which
    for a ``well-formed-only`` entry means a complete record only.  Untrusted
    prose that merely names such a record would otherwise keep the reserved
    spelling, so replace any remaining literal registry name with the same
    stable label.  This path feeds prompts only; complete-record parsing and
    tool-owned publication validation are unchanged.
    """
    if not isinstance(text, str) or not text:
        return text
    safe = sanitize_historical_text(text)
    # Longest name first so a shorter name can never split a longer one.
    for definition in sorted(
        RESERVED_MARKER_REGISTRY, key=lambda item: len(item.token), reverse=True
    ):
        safe = re.sub(
            re.escape(definition.token), definition.safe_label, safe, flags=re.I
        )
    return safe


def _record_shaped_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Spans that claim reserved record syntax, well-formed or not.

    Every reserved record is an HTML comment except the line-form audit
    trailer, so a token inside `<!-- ... -->` — or at the start of a
    line-form trailer — is a record claim rather than prose naming the token.
    """
    spans: list[tuple[int, int]] = []
    for definition in RESERVED_MARKER_REGISTRY:
        token = re.escape(definition.token)
        shapes = [rf"<!--\s*{token}\b[^\r\n]*?-->"]
        if definition.codec == "key-value-line":
            shapes.append(rf"(?m)^[ \t]*{token}[ \t]+[^\r\n]+$")
        for shape in shapes:
            for match in re.finditer(shape, text, re.I):
                spans.append((match.start(), match.end()))
    return tuple(spans)


def record_shaped_untrusted_markers(text: str) -> tuple[MarkerOccurrence, ...]:
    """Occurrences that claim record syntax rather than merely naming a token.

    Untrusted GitHub prose may legitimately name a reserved token when the
    work concerns the protocol itself, so naming one is neutralized rather
    than refused.  A span that claims the record grammar is still a forgery
    attempt and keeps the fail-closed behavior (#891).
    """
    if not isinstance(text, str) or not text:
        return ()
    occurrences = _all_occurrences(text)
    if not occurrences:
        return ()
    spans = _record_shaped_spans(text)
    shaped: list[MarkerOccurrence] = []
    for occurrence in occurrences:
        if _is_complete_historical_occurrence(occurrence):
            shaped.append(occurrence)
            continue
        token = re.escape(occurrence.definition.token)
        for match in re.finditer(token, occurrence.text, re.I):
            start = occurrence.start + match.start()
            end = occurrence.start + match.end()
            if any(low <= start and end <= high for low, high in spans):
                shaped.append(occurrence)
                break
    return tuple(shaped)


def _is_complete_historical_occurrence(occurrence: MarkerOccurrence) -> bool:
    match = occurrence.definition.pattern.fullmatch(occurrence.text)
    if match is None:
        return False
    if occurrence.definition.codec != "key-value-line":
        return True
    try:
        occurrence.definition.canonicalizer(match)
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError, zlib.error):
        return False
    return True


def _historical_replacement_occurrences(text: str) -> tuple[MarkerOccurrence, ...]:
    """Return safe replacement spans without discarding surrounding prose.

    A name-bearing-line fallback is deliberately broad while scanning so a
    malformed marker cannot pass unnoticed. Historical text still needs to
    retain the prose around that mention, though, so replace the token itself
    unless the complete marker grammar matched the occurrence.
    """
    occurrences = _all_occurrences(text)
    complete_occurrences = tuple(
        occurrence for occurrence in occurrences
        if _is_complete_historical_occurrence(occurrence)
    )
    replacements: list[MarkerOccurrence] = []
    for occurrence in occurrences:
        definition = occurrence.definition
        line_match = definition.pattern.fullmatch(occurrence.text)
        invalid_line_record = False
        if definition.codec == "key-value-line" and line_match is not None:
            try:
                definition.canonicalizer(line_match)
            except (ValueError, TypeError, UnicodeError, json.JSONDecodeError, zlib.error):
                invalid_line_record = True
        if (
            definition.strictness == "name-bearing-line"
            and definition.pattern.fullmatch(occurrence.text) is None
        ) or invalid_line_record:
            for match in re.finditer(re.escape(definition.token), occurrence.text, re.I):
                replacements.append(
                    MarkerOccurrence(
                        definition,
                        occurrence.start + match.start(),
                        occurrence.start + match.end(),
                        match.group(0),
                    )
                )
        else:
            replacements.append(occurrence)

    # The scanner deliberately returns non-overlapping spans, so a broad
    # name-bearing-line fallback can hide another strict token on the same
    # line. Historical text must neutralize both mentions unless a complete
    # record already covers the nested token.
    seen = {(item.definition.token, item.start, item.end) for item in replacements}
    for definition in RESERVED_MARKER_REGISTRY:
        if definition.strictness == "well-formed-only":
            continue
        for match in re.finditer(re.escape(definition.token), text, re.I):
            if any(
                complete.start <= match.start() and match.end() <= complete.end
                for complete in complete_occurrences
            ):
                continue
            key = (definition.token, match.start(), match.end())
            if key in seen:
                continue
            replacements.append(
                MarkerOccurrence(definition, match.start(), match.end(), match.group(0))
            )
            seen.add(key)
    return tuple(sorted(replacements, key=lambda item: item.start))


def sanitize_historical_text(text: str) -> str:
    """Replace reserved spans with stable labels, preserving surrounding prose."""
    occurrences = _historical_replacement_occurrences(text)
    if not occurrences:
        return text
    pieces: list[str] = []
    cursor = 0
    for occurrence in occurrences:
        pieces.append(text[cursor : occurrence.start])
        pieces.append(occurrence.definition.safe_label)
        cursor = occurrence.end
    pieces.append(text[cursor:])
    safe = "".join(pieces)
    if _all_occurrences(safe):
        raise AgentLoopError("Historical marker neutralization produced a reserved marker.")
    return safe


def historical_replacement_labels(text: str) -> tuple[str, ...]:
    """Safe labels for every span `sanitize_historical_text` would replace.

    Provenance checks need the SAME occurrence set the stripping path uses, not
    the non-overlapping scan: a malformed name-bearing-line fallback spans a
    whole line and can hide another reserved token inside it, which
    :func:`scan_reserved_markers` never reports separately while the historical
    pass neutralizes both.  Returning one label per replaced span keeps identity
    and occurrence count aligned with what stripping removes.
    """
    if not isinstance(text, str):
        raise TypeError("historical text must be a string")
    return tuple(
        occurrence.definition.safe_label
        for occurrence in _historical_replacement_occurrences(text)
    )


def historical_text_fragments(text: str) -> tuple[str, ...]:
    """Return substantive text outside spans that historical repair may replace.

    Preservation checks must not require a repair backend to reproduce this
    registry's label or punctuation around a quoted marker.  They do still
    need to retain the prose on either side of the marker.  Complete records
    and strict fallback mentions use the same replacement spans as
    :func:`sanitize_historical_text`, so malformed name-bearing lines retain
    their surrounding prose as well.
    """
    if not isinstance(text, str):
        raise TypeError("historical text must be a string")
    occurrences = _historical_replacement_occurrences(text)
    if not occurrences:
        return (text,) if text.strip() else ()

    fragments: list[str] = []
    cursor = 0
    for occurrence in occurrences:
        fragment = text[cursor:occurrence.start]
        if text[occurrence.start - 1:occurrence.start] in {"`", "'", '"'}:
            fragment = fragment[:-1]
        if fragment.strip() and re.search(r"\w", fragment):
            fragments.append(fragment)
        cursor = occurrence.end
        if text[cursor:cursor + 1] in {"`", "'", '"'}:
            cursor += 1
    fragment = text[cursor:]
    if fragment.strip() and re.search(r"\w", fragment):
        fragments.append(fragment)
    return tuple(fragments)


@dataclass(frozen=True)
class _BodySegment:
    text: str
    token: str | None = None


class TrustedBody(str):
    """Immutable body text with explicit authorization for every marker span."""

    _segments: tuple[_BodySegment, ...]

    def __new__(cls, text: str, segments: tuple[_BodySegment, ...] = ()) -> "TrustedBody":
        instance = str.__new__(cls, text)
        instance._segments = segments or (_BodySegment(text),)
        return instance

    @property
    def segments(self) -> tuple[tuple[str, str | None], ...]:
        return tuple((segment.text, segment.token) for segment in self._segments)

    @classmethod
    def current_untrusted_visible(cls, text: str) -> "TrustedBody":
        if not isinstance(text, str):
            raise TypeError("TrustedBody text must be a string")
        occurrences = scan_reserved_markers(text)
        if occurrences:
            names = ", ".join(sorted({item.definition.token for item in occurrences}))
            raise AgentLoopError(
                "Current untrusted GitHub text contains reserved protocol marker(s): "
                + names
            )
        return cls(text, (_BodySegment(text),))

    @classmethod
    def historical_visible(cls, text: str) -> "TrustedBody":
        return cls.current_untrusted_visible(sanitize_historical_text(text))

    @classmethod
    def canonical(
        cls,
        text: str,
        *,
        surface: str | None = None,
        expected_tokens: tuple[str, ...] | list[str] | set[str] | None = None,
    ) -> "TrustedBody":
        occurrences = scan_reserved_markers(text)
        expected = set(expected_tokens or ())
        if expected and {item.definition.token for item in occurrences} != expected:
            raise AgentLoopError(
                "Trusted body marker set mismatch: expected "
                f"{sorted(expected)}, found {sorted({item.definition.token for item in occurrences})}."
            )
        if len({item.definition.token for item in occurrences}) != len(occurrences):
            raise AgentLoopError("Trusted body contains duplicate reserved protocol markers.")
        segments: list[_BodySegment] = []
        cursor = 0
        for occurrence in occurrences:
            definition = occurrence.definition
            if surface is not None and surface not in definition.surfaces:
                raise AgentLoopError(
                    f"{definition.token} is not allowed on the {surface} surface."
                )
            try:
                match = definition.pattern.search(occurrence.text)
                if match is None:
                    raise ValueError("reserved marker does not match its declared grammar")
                canonical = definition.canonicalizer(
                    match
                )
            except (ValueError, TypeError, UnicodeError, json.JSONDecodeError, zlib.error) as exc:
                raise AgentLoopError(f"Invalid {definition.token} protocol record.") from exc
            if canonical != occurrence.text:
                raise AgentLoopError(f"{definition.token} protocol record is not canonical.")
            if occurrence.start > cursor:
                segments.append(_BodySegment(text[cursor : occurrence.start]))
            segments.append(_BodySegment(text=occurrence.text, token=definition.token))
            cursor = occurrence.end
        if cursor < len(text) or not segments:
            segments.append(_BodySegment(text=text[cursor:]))
        return cls(text, tuple(segment for segment in segments if segment.text or len(segments) == 1))

    @classmethod
    def marker(cls, token: str, text: str) -> "TrustedBody":
        if token not in MARKER_BY_TOKEN:
            raise AgentLoopError(f"Unknown reserved protocol marker {token}.")
        body = cls.canonical(text, expected_tokens=(token,))
        return body

    @classmethod
    def join(cls, *parts: "TrustedBody | str") -> "TrustedBody":
        converted: list[TrustedBody] = []
        for part in parts:
            converted.append(part if isinstance(part, TrustedBody) else cls.current_untrusted_visible(part))
        return cls("".join(str(part) for part in converted), tuple(segment for part in converted for segment in part._segments))

    def append(self, *parts: "TrustedBody | str") -> "TrustedBody":
        return self.join(self, *parts)

    def prepend(self, *parts: "TrustedBody | str") -> "TrustedBody":
        return self.join(*parts, self)

    def replace_text(self, old: str, replacement: "TrustedBody | str") -> "TrustedBody":
        index = str(self).find(old)
        if index < 0:
            raise AgentLoopError("Trusted body replacement target was not found.")
        left = str(self)[:index]
        right = str(self)[index + len(old) :]
        return self.join(self.current_untrusted_visible(left), replacement, self.current_untrusted_visible(right))

    def validate_for_surface(self, surface: str) -> None:
        if not isinstance(self, TrustedBody):
            raise AgentLoopError("Trusted GitHub writers require a TrustedBody carrier.")
        rebuilt = "".join(segment.text for segment in self._segments)
        if rebuilt != str(self):
            raise AgentLoopError("Trusted body segment provenance does not match its text.")
        seen: set[str] = set()
        for segment in self._segments:
            occurrences = scan_reserved_markers(segment.text)
            if segment.token is None:
                if occurrences:
                    raise AgentLoopError("Reserved marker appears in an unauthorized body segment.")
                continue
            if len(occurrences) != 1 or occurrences[0].start != 0 or occurrences[0].end != len(segment.text):
                raise AgentLoopError("Trusted marker segment is not a single exact marker span.")
            occurrence = occurrences[0]
            if occurrence.definition.token != segment.token:
                raise AgentLoopError("Trusted marker segment provenance is inconsistent.")
            if segment.token in seen:
                raise AgentLoopError("Trusted body contains duplicate authorized marker segments.")
            seen.add(segment.token)
            if surface not in occurrence.definition.surfaces:
                raise AgentLoopError(f"{segment.token} is not allowed on the {surface} surface.")


def trusted_body(
    text: str,
    *,
    surface: str | None = None,
    expected_tokens: tuple[str, ...] | list[str] | set[str] | None = None,
) -> TrustedBody:
    """Authorize a canonical producer result at the composition seam."""
    return TrustedBody.canonical(text, surface=surface, expected_tokens=expected_tokens)


def source_protocol_literals(root: Path) -> set[str]:
    """Collect string-literal protocol names for the source-inventory guard."""
    found: set[str] = set()
    for path in sorted((root / "src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            found.update(re.findall(r"\bAGENT_[A-Z0-9_]+\b", node.value))
    return found


def assert_source_inventory(root: Path) -> None:
    found = source_protocol_literals(root)
    unknown = found - set(MARKER_BY_TOKEN) - EXCLUDED_PROTOCOL_LITERALS
    if unknown:
        raise AgentLoopError(
            "Unregistered AGENT_* protocol-looking literal(s): " + ", ".join(sorted(unknown))
        )


PROTOCOL_RECORD_LABEL_MAX_CHARS = 320
_RECORD_LABEL_FIELD_MAX_CHARS = 64
_RECORD_LABEL_FIELD_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._/-]*\Z")

# Closed, trusted vocabulary.  Labels are composed only from these phrases and
# from already-authenticated identity fragments, never from agent or operator
# text, so every label is a pure function of the record it introduces.
_RECORD_LABEL_NAMES: dict[tuple[str, str], str] = {
    ("managed_ci_authorization", "creation"): "managed-CI authorization record",
    ("managed_ci_authorization", "fresh"): "managed-CI fresh re-authorization record",
    ("managed_ci_authorization", "continuity"): "managed-CI authorization continuity record",
    ("managed_ci_intent", ""): "managed exact-head CI intent record",
    ("managed_ci_override_audit", ""): "managed-CI unprotected-override audit record",
    ("managed_ci_resume_audit", ""): "managed-CI resume provenance audit record",
    ("managed_ci_qualified_head", ""): "managed-CI qualified-head record",
    ("plan_validation_diagnostic", ""): "plan-validation diagnostic record",
}
_RECORD_LABEL_FAMILIES = frozenset(family for family, _ in _RECORD_LABEL_NAMES)


def _record_label_field(value: str | None, *, field: str) -> str | None:
    """Validate one bounded identity fragment, or report it as unknown."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise AgentLoopError(f"Protocol record label {field} must be text.")
    if not value:
        return None
    if not value.isascii() or len(value) > _RECORD_LABEL_FIELD_MAX_CHARS:
        raise AgentLoopError(
            f"Protocol record label {field} is not bounded ASCII text."
        )
    if _RECORD_LABEL_FIELD_RE.fullmatch(value) is None:
        raise AgentLoopError(f"Protocol record label {field} is not a plain identifier.")
    return value


def _record_label_number(value: int | None) -> int | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _record_label_issue(issue_number: int | None) -> str:
    number = _record_label_number(issue_number)
    return f"issue #{number}" if number is not None else "the originating issue"


def _record_label_pr(pr_number: int | None) -> str:
    number = _record_label_number(pr_number)
    return f"pull request #{number}" if number is not None else "this pull request"


def protocol_record_label(
    family: str,
    *,
    kind: str | None = None,
    issue_number: int | None = None,
    pr_number: int | None = None,
    head_sha: str | None = None,
    state: str | None = None,
    attempt: int | None = None,
) -> str:
    """Render one bounded, deterministic visible label for a tool-owned record.

    Tool-owned protocol comments carry their payload in a hidden marker, which
    GitHub renders as "No description provided.".  The label names the record
    and its role in one line placed *outside* the marker span, so the marker
    grammar, payload, parsing, and recovery are untouched.  Repeated calls with
    the same inputs return byte-identical text, which keeps retries and
    read-after-write verification exact.
    """
    if family not in _RECORD_LABEL_FAMILIES:
        raise AgentLoopError(f"Unknown tool-owned protocol record family: {family}.")
    name = _RECORD_LABEL_NAMES.get((family, kind or ""))
    if name is None:
        raise AgentLoopError(
            f"Unknown {family} protocol record kind: {kind or '(none)'}."
        )
    head = _record_label_field(head_sha, field="head")
    head_text = f" at head {head[:7]}" if head is not None else ""
    if family == "managed_ci_authorization":
        role = (
            f"Binds {_record_label_issue(issue_number)} to "
            f"{_record_label_pr(pr_number)}{head_text}."
        )
    elif family == "managed_ci_intent":
        lifecycle = _record_label_field(state, field="state")
        state_text = f" in state {lifecycle}" if lifecycle is not None else ""
        role = (
            f"Tracks the exact-head CI intent for {_record_label_pr(pr_number)}"
            f"{head_text}{state_text}."
        )
    elif family == "managed_ci_override_audit":
        role = (
            "Records the unprotected-override waiver used to activate managed CI for "
            f"{_record_label_pr(pr_number)}{head_text}."
        )
    elif family == "managed_ci_resume_audit":
        role = (
            "Records resume provenance for managed CI on "
            f"{_record_label_pr(pr_number)}{head_text}."
        )
    elif family == "managed_ci_qualified_head":
        role = (
            f"Records the qualified merge head for {_record_label_pr(pr_number)}"
            f"{head_text}."
        )
    else:
        number = _record_label_number(attempt)
        attempt_text = f" (failure attempt {number})" if number is not None else ""
        role = (
            "Records deterministic plan-validation exhaustion for "
            f"{_record_label_issue(issue_number)}{attempt_text}."
        )
    label = f"Agent-loop {name} (machine-readable). {role} Not an agent response; keep this comment."
    if not label.isascii() or len(label) > PROTOCOL_RECORD_LABEL_MAX_CHARS:
        raise AgentLoopError("Protocol record label exceeds its bounded ASCII budget.")
    if scan_reserved_markers(label) or any(token in label for token in MARKER_BY_TOKEN):
        raise AgentLoopError("Protocol record label must not contain reserved marker text.")
    return label
