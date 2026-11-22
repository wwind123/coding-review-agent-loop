"""Deterministic, model-free normalization of structured agent responses (#2066).

Every function here rewrites only mechanically recoverable defects (envelope
material, lossless aliases, lone-string matrix list fields, unknown prior-item
dispositions proven to be resolved history, architecture status near misses)
and never calls a model.  The caller re-validates every rewritten candidate.
They used to live in ``repair.py`` / ``repair_preservation.py``; they now run
for every validated turn, independent of the retired repair flag.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .errors import AgentLoopError
from .protocol import STATE_RE, classify_architecture_status_near_miss
from .protocol_common import (
    HUMAN_REQUIREMENTS_RESOLVED_RE,
    PLAN_STATE_RE,
    ParseDegradation,
    _extract_json_object_prefix,
    normalize_response_file_structured_text,
    parse_human_requirements_acknowledgement,
)
from .protocol_risk import normalize_risk_test_matrix_string_lists

_SIGNATURE_LINE_RE = re.compile(r"(?m)^--\s+\S[^\n]*")

_FENCED_JSON_PREFIX_RE = re.compile(
    r"\A[ \t]{0,3}(?P<fence>`{3,}|~{3,})[ \t]*(?:json)?[ \t]*\r?\n",
    re.IGNORECASE,
)

def _payload_and_trailing(text: str) -> tuple[dict | None, str]:
    """Split *text* into its recovered JSON object and the prose that follows.

    The trailing remainder is the only part of a source that can carry freeform
    reviewer prose: the recovered object's own fields are structured data, not
    findings.
    """
    text, _ = normalize_response_file_structured_text(text)
    stripped = text.lstrip()
    fence = _FENCED_JSON_PREFIX_RE.match(stripped)
    if fence:
        fence_char = re.escape(fence.group("fence")[0])
        fence_length = len(fence.group("fence"))
        closing_fence = re.search(
            rf"(?m)^[ \t]{{0,3}}{fence_char}{{{fence_length},}}[ \t]*(?:\r?\n|\Z)",
            stripped[fence.end():],
        )
        if closing_fence:
            stripped = stripped[fence.end():fence.end() + closing_fence.start()]
        else:
            stripped = stripped[fence.end():]
        text = stripped
    try:
        parsed = _extract_json_object_prefix(text)
    except AgentLoopError:
        return None, text
    if not parsed:
        return None, text
    return parsed[0], parsed[1]


def _payload(text: str) -> dict | None:
    return _payload_and_trailing(text)[0]


def recover_payload(text: str) -> dict | None:
    """Return the JSON object mechanically recoverable from *text*, if any.

    Shared with the repair admission gate so the gate and the grounding guard
    agree on exactly which sources carry a recoverable structured payload.
    """
    return _payload(text)



def strip_unknown_prior_item_dispositions(
    raw: str,
    *,
    allowed_ids: frozenset[str],
    expected_kind: str | None,
) -> str | None:
    """Remove disposition entries whose item_id is not in allowed_ids.

    Returns the normalized text (JSON + trailing footer/signature) on success,
    or None when the input cannot be parsed or nothing needs to be removed.
    The caller must re-validate the returned text before using it.
    """
    if expected_kind not in {"pr_review", "plan_review", "plan_revision", "plan_revision_patch"}:
        return None
    disposition_field = (
        "prior_item_dispositions"
        if expected_kind == "pr_review"
        else "prior_plan_item_dispositions"
    )
    stripped = raw.lstrip()
    if not stripped.startswith("{"):
        return None
    try:
        payload, json_end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("kind") != expected_kind:
        return None
    dispositions = payload.get(disposition_field)
    if not isinstance(dispositions, list):
        return None
    filtered = [
        e for e in dispositions if isinstance(e, dict) and e.get("item_id") in allowed_ids
    ]
    if len(filtered) == len(dispositions):
        return None
    json_str = json.dumps({**payload, disposition_field: filtered}, ensure_ascii=False)
    tail = stripped[json_end:]
    if tail and not tail.startswith("\n"):
        tail = "\n" + tail
    return json_str + tail


def unknown_dispositions_are_resolved_history(
    raw: str,
    *,
    unknown_ids: Sequence[str],
    resolved_history_ids: Sequence[str],
    expected_kind: str | None,
) -> bool:
    """Return True when every unknown disposition is a no-op on resolved history.

    Used to allow the deterministic strip when the round ledger may be
    incomplete: each unknown ID must be one that recorded history proves was
    canonically resolved, and every entry for it must itself be ``resolved``.
    Any other shape (active disposition, unparseable text, wrong kind, an ID
    outside history) returns False so the caller keeps failing closed.
    """
    if expected_kind not in {"pr_review", "plan_review", "plan_revision", "plan_revision_patch"}:
        return False
    unknown = frozenset(unknown_ids)
    if not unknown or not unknown <= frozenset(resolved_history_ids):
        return False
    disposition_field = (
        "prior_item_dispositions"
        if expected_kind == "pr_review"
        else "prior_plan_item_dispositions"
    )
    stripped = raw.lstrip()
    if not stripped.startswith("{"):
        return False
    try:
        payload, _json_end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict) or payload.get("kind") != expected_kind:
        return False
    dispositions = payload.get(disposition_field)
    if not isinstance(dispositions, list):
        return False
    seen: set[str] = set()
    for entry in dispositions:
        if not isinstance(entry, dict):
            return False
        item_id = entry.get("item_id")
        if item_id not in unknown:
            continue
        if entry.get("disposition") != "resolved":
            return False
        seen.add(item_id)
    return seen == unknown


def attempt_risk_test_matrix_string_list_normalization(
    raw: str, *, expected_kind: str | None
) -> tuple[str, tuple[str, ...]] | None:
    """Wrap lone-string risk matrix list fields, preserving all other bytes.

    Semantic patches are excluded: they must be preserved byte-for-byte.
    """
    if expected_kind not in {"plan_state", "plan_revision"}:
        return None
    start = len(raw) - len(raw.lstrip())
    brace = raw.find("{", start)
    if brace < 0:
        return None
    try:
        payload, json_end = json.JSONDecoder().raw_decode(raw[brace:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    normalized, paths = normalize_risk_test_matrix_string_lists(payload)
    if not paths:
        return None
    new_json = json.dumps(normalized, ensure_ascii=False)
    return raw[:brace] + new_json + raw[brace + json_end :], paths


def attempt_envelope_normalization(raw: str, *, expected_kind: str | None) -> str | None:
    """Trim envelope-only trailing material without changing structured JSON."""
    if expected_kind == 'screenshot-coverage':
        value = recover_payload(raw)
        if value is None or value.get('kind') != expected_kind:
            return None
        return json.dumps(value)
    if expected_kind not in {"plan_state", "pr_review", "plan_review", "plan_revision", "plan_revision_patch", "coder_followup", "issue_implementation", "discuss_review", "discuss_answer", "discuss_agenda", "discuss_round_synthesis", "discuss_final_synthesis", "discuss_semantic_comparison", "discuss_answer_confirmation", "discuss_evidence_reconciliation"}:
        return None

    stripped = raw.lstrip()
    if not stripped.startswith("{"):
        # Deterministic recovery for the common JSON-fence/leading-envelope
        # shape. The source object is still parsed and validated below; this
        # only removes material outside the structured response.
        brace_index = stripped.find("{")
        if brace_index < 0:
            return None
        stripped = stripped[brace_index:]
    try:
        payload, json_end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    json_text = stripped[:json_end].rstrip()
    trailing = stripped[json_end:]
    state_re = PLAN_STATE_RE if expected_kind in {"plan_state", "plan_review", "plan_revision", "plan_revision_patch", "discuss_review", "discuss_answer", "discuss_agenda", "discuss_round_synthesis", "discuss_final_synthesis", "discuss_semantic_comparison", "discuss_answer_confirmation"} else STATE_RE
    state_match = state_re.search(trailing)
    if state_match is None:
        return None

    before_footer = re.sub(
        r"(?im)^\s*(?:`{3,}|~{3,})(?:json)?\s*$\n?",
        "",
        trailing[: state_match.start()],
    )
    after_footer_raw = trailing[state_match.end() :]
    footer = trailing[state_match.start() : state_match.end()].strip()

    # Normalize reversed-order envelope: JSON + signature + footer → JSON + footer + signature.
    # Applies when before_footer is exactly one signature line and nothing follows the footer.
    if not after_footer_raw.strip():
        before_stripped = before_footer.strip()
        sig_candidate = _SIGNATURE_LINE_RE.search(before_stripped)
        if sig_candidate is not None and before_stripped == before_stripped[sig_candidate.start() : sig_candidate.end()].strip():
            footer_state = state_match.group(1).lower()
            payload_state = payload.get("state")
            state_ok = not (
                isinstance(payload_state, str)
                and payload_state.strip()
                and payload_state.strip() != footer_state
            )
            if state_ok:
                return f"{json_text}\n{footer}\n{before_stripped}"

    preserved_before = ""
    if expected_kind in {"pr_review", "plan_review"}:
        before_lstripped = before_footer.lstrip()
        marker_match = HUMAN_REQUIREMENTS_RESOLVED_RE.match(before_lstripped)
        if marker_match is not None and before_lstripped[marker_match.end() :].strip() == "":
            preserved_before = before_lstripped[: marker_match.end()].strip()
        elif before_footer.strip():
            return None
    elif expected_kind in {"plan_state", "plan_revision", "plan_revision_patch"} and before_footer.strip():
        parsed_human_requirements = parse_human_requirements_acknowledgement(before_footer)
        if (
            parsed_human_requirements.marker_present
            and parsed_human_requirements.section_present
        ):
            preserved_before = before_footer.strip()
        else:
            return None
    elif before_footer.strip():
        return None

    after_footer_lstripped = after_footer_raw.lstrip()
    preserved_after = ""
    if expected_kind == "pr_review":
        marker_match = HUMAN_REQUIREMENTS_RESOLVED_RE.match(after_footer_lstripped)
        if marker_match is not None:
            preserved_after = after_footer_lstripped[: marker_match.end()].strip()
            after_footer_lstripped = after_footer_lstripped[marker_match.end() :]
        signature_match = _SIGNATURE_LINE_RE.match(after_footer_lstripped.lstrip())
    else:
        signature_match = _SIGNATURE_LINE_RE.search(after_footer_lstripped)

    if signature_match is None:
        return None
    signature_source = (
        after_footer_lstripped.lstrip()
        if expected_kind == "pr_review"
        else after_footer_lstripped
    )
    signature = signature_source[signature_match.start() : signature_match.end()].strip()

    parts = [json_text]
    if preserved_before:
        parts.append(preserved_before)
    parts.append(footer)
    if preserved_after:
        parts.append(preserved_after)
    parts.append(signature)
    return "\n".join(parts)


def attempt_semantic_patch_disposition_normalization(raw: str) -> str | None:
    """Rename the lossless ``rationale`` disposition alias to ``note``.

    This is deliberately narrower than model repair: it neither selects nor
    changes a disposition, and it refuses ambiguous rows that contain both
    names. The strict semantic-patch parser remains authoritative afterward.
    """
    stripped = raw.lstrip()
    if not stripped.startswith("{"):
        return None
    try:
        payload, json_end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("kind") != "plan_revision_patch":
        return None
    dispositions = payload.get("prior_plan_item_dispositions")
    if not isinstance(dispositions, list):
        return None

    normalized_dispositions: list[object] = []
    changed = False
    for entry in dispositions:
        if not isinstance(entry, dict):
            return None
        normalized_entry = dict(entry)
        if "rationale" in normalized_entry:
            if "note" in normalized_entry:
                return None
            rationale = normalized_entry.pop("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                return None
            normalized_entry["note"] = rationale
            changed = True
        normalized_dispositions.append(normalized_entry)
    if not changed:
        return None

    normalized_payload = {**payload, "prior_plan_item_dispositions": normalized_dispositions}
    tail = stripped[json_end:]
    return json.dumps(normalized_payload, ensure_ascii=False) + tail


@dataclass(frozen=True)
class ArchitectureNearMissNormalization:
    """Pre-repair normalization result; the record stays out of band."""

    raw: str
    record: ParseDegradation | None
    forbid_architecture_impact: bool


def _rewrite_architecture_near_miss(raw: str) -> tuple[str, ParseDegradation | None, dict | None, bool]:
    """Deterministically rewrite a top-level near miss to its wire-valid form.

    A corroborated `modified` becomes `changed`; every other near miss has its
    optional object removed.  Only the JSON object span is rewritten; the
    footer and signature bytes that follow it are kept.
    """
    payload, trailing = _payload_and_trailing(raw)
    impact = payload.get("architecture_impact") if isinstance(payload, dict) else None
    if not isinstance(impact, dict):
        return raw, None, payload, False
    kind = payload.get("kind") if isinstance(payload.get("kind"), str) else "response"
    near_miss = classify_architecture_status_near_miss(
        impact, context=f"{kind}.architecture_impact"
    )
    if near_miss is None:
        return raw, None, payload, False
    resolved, record = near_miss
    rewritten = dict(payload)
    removed = resolved != "changed"
    if removed:
        rewritten.pop("architecture_impact")
    else:
        rewritten["architecture_impact"] = {**impact, "status": resolved}
    normalized = json.dumps(rewritten, ensure_ascii=False) + (
        trailing if trailing.startswith(("\n", "\r")) else "\n" + trailing.lstrip()
    )
    return normalized, record, rewritten, removed


def canonicalize_architecture_near_miss_text(text: str) -> str:
    """Return accepted text with any near-miss status in its wire-valid form."""
    return _rewrite_architecture_near_miss(text)[0]


def normalize_architecture_impact_near_miss(
    raw: str, *, required_contract: bool, expected_kind: str | None = None
) -> ArchitectureNearMissNormalization:
    """Resolve the status near miss in the raw payload before repair (#924).

    Deterministic normalization runs before the LLM repair pass.  A
    corroborated `modified` becomes a wire-valid `changed` that preservation
    then pins; every other near miss has its whole optional object removed,
    which is exactly what the parser-only `undetermined` status stands for.
    Any other payload, including an unparseable one, is returned unchanged.

    The absence pin has two triggers.  A normalization removal is pinned for
    every kind, including optional-assessment reviews, so repair can never put
    back a stronger claim.  Under a required contract, a recoverable source of
    the expected kind that simply omits the assessment is pinned too.
    """
    normalized, record, payload, removed = _rewrite_architecture_near_miss(raw)
    forbid = removed or bool(
        required_contract
        and isinstance(payload, dict)
        and "architecture_impact" not in payload
        and (expected_kind is None or payload.get("kind") == expected_kind)
    )
    return ArchitectureNearMissNormalization(
        raw=normalized, record=record, forbid_architecture_impact=forbid
    )


_REVIEW_KINDS = frozenset({"plan_review", "pr_review"})
_REVIEW_KIND_UNIQUE_FIELDS = {
    "plan_review": frozenset({
        "blocking_plan_issues", "same_plan_followups", "prior_plan_item_dispositions",
    }),
    "pr_review": frozenset({
        "blocking_items", "same_pr_followups", "prior_item_dispositions",
    }),
}
_REVIEW_ENVELOPE_ONLY_KEYS = frozenset({"schema_version", "kind", "state"})


def review_substance_missing(raw: str, *, expected_kind: str | None) -> str | None:
    """Why a reviewer response carries no review substance, or ``None`` when it does.

    Narration, tool-use diagnostics, an envelope-only payload or a bare state
    footer carry no verdict and no finding.  Such a response is a reviewer
    availability failure, not a verdict (#2029): once the same-agent re-asks
    are spent the turn is classified ``agent-unavailable`` so a multi-reviewer
    round can continue and an Antigravity chain can fall back (#2066).  A
    payload counts as substance only when it is mechanically recoverable and
    either declares the expected kind with more than envelope keys or, with no
    ``kind`` at all, carries a field unique to that review schema.
    """
    if expected_kind not in _REVIEW_KINDS:
        return None
    payload = recover_payload(raw)
    if not isinstance(payload, dict):
        return f"the `{expected_kind}` response carries no recoverable review payload"
    if "kind" in payload:
        kind = payload["kind"]
        if kind != expected_kind:
            return f"the response declares kind {kind!r}, not the expected `{expected_kind}`"
        if set(payload) <= _REVIEW_ENVELOPE_ONLY_KEYS:
            return f"the `{expected_kind}` response carries only envelope keys"
        return None
    if not (_REVIEW_KIND_UNIQUE_FIELDS[expected_kind] & set(payload)):
        return f"the kindless response carries no field unique to the `{expected_kind}` schema"
    return None
