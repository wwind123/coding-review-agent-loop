"""Bounded loss checks for parseable review and implementation repair inputs."""

from collections.abc import Sequence
import re

from .errors import AgentLoopError
from .protocol import (
    _extract_json_object_prefix,
    _normalize_requirement_label,
    normalize_response_file_structured_text,
)
from .protocol_markers import scan_reserved_markers


_KINDS = {
    "pr_review", "plan_review", "plan_revision", "plan_state",
    "coder_followup", "issue_implementation",
}

_FENCED_JSON_PREFIX_RE = re.compile(
    r"\A[ \t]{0,3}(?P<fence>`{3,}|~{3,})[ \t]*(?:json)?[ \t]*\r?\n",
    re.IGNORECASE,
)
_FINDING_METADATA_KEYS = {
    "id", "item_id", "severity", "category", "state", "disposition", "verdict",
}
_HUMAN_REQUIREMENT_DISPOSITIONS = {"addressed", "blocked", "not-applicable"}


def _payload(text: str) -> dict | None:
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
        return None
    return parsed[0] if parsed else None


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _normalized_requirement_id(text: str) -> str | None:
    try:
        return _normalize_requirement_label(text)
    except AgentLoopError:
        return None


def _fragments(value: object) -> list[str]:
    if isinstance(value, str):
        # Reserved protocol syntax must be removable; its safety validator wins.
        return [value] if value.strip() and not scan_reserved_markers(value) else []
    if isinstance(value, list):
        return [text for child in value for text in _fragments(child)]
    if isinstance(value, dict):
        return [
            text for key, child in value.items()
            if key not in _FINDING_METADATA_KEYS
            for text in _fragments(child)
        ]
    return []


def validate_repair_preservation(
    raw: str,
    repaired: str,
    *,
    unresolved_item_ids: Sequence[str] | None = None,
    surfaced_requirement_ids: Sequence[str] | None = None,
) -> None:
    """Reject observable losses, not certify semantic equivalence.

    Invalid JSON and unsupported schemas remain on the ordinary repair path.
    Do not heuristically parse broken JSON or interpret prose as item ledgers.
    """
    source, target = _payload(raw), _payload(repaired)
    if (not source or not target or not isinstance(source.get("kind"), str)
            or source["kind"] not in _KINDS):
        return
    if source.get("kind") != target.get("kind"):
        return  # Kind selection belongs to the caller's schema/context validator.

    def require(condition: bool, field: str) -> None:
        if not condition:
            raise AgentLoopError(
                f"Repair content preservation failed for {field}: preserve original "
                "wording, evidence, and test caveats; do not summarize or omit content."
            )

    summary = source.get("summary")
    if isinstance(summary, str) and _fragments(summary):
        require(
            isinstance(target.get("summary"), str)
            and _normalized(summary) in _normalized(target["summary"]),
            "summary",
        )

    fields = set()
    if source["kind"] in {"coder_followup", "issue_implementation"}:
        fields.add("tests_run")
    if source["kind"] in {"plan_state", "plan_revision"}:
        fields.add("plan_steps")
    for field in fields:
        entries = source.get(field)
        if isinstance(entries, list) and all(isinstance(e, str) for e in entries):
            required = [_normalized(e) for e in entries if _fragments(e)]
            actual = target.get(field, [])
            require(isinstance(actual, list), field)
            available = [_normalized(e) for e in actual if isinstance(e, str)]
            for entry in required:
                require(entry in available, field)
                available.remove(entry)

    if source["kind"] == "coder_followup":
        allowed_item_ids = (
            set(unresolved_item_ids) if unresolved_item_ids is not None else None
        )
        for field in ("addressed_item_notes", "remaining_item_notes"):
            notes = source.get(field)
            if not isinstance(notes, dict):
                continue
            for item_id, note in notes.items():
                if not isinstance(note, str) or not _fragments(note):
                    continue
                # Disposition normalization may move a note to the other bucket.
                candidates = [
                    target[bucket].get(item_id)
                    for bucket in ("addressed_item_notes", "remaining_item_notes")
                    if isinstance(target.get(bucket), dict)
                ]
                require(any(isinstance(c, str) and _normalized(note) in _normalized(c)
                            for c in candidates), field)

        disputed_items = source.get("disputed_items")
        if isinstance(disputed_items, list) and all(
            isinstance(item_id, str) for item_id in disputed_items
        ):
            target_disputed_items = target.get("disputed_items", [])
            require(isinstance(target_disputed_items, list), "disputed_items")
            for item_id in disputed_items:
                if allowed_item_ids is not None and item_id not in allowed_item_ids:
                    continue
                require(item_id in target_disputed_items, "disputed_items")

        dispute_evidence = source.get("dispute_evidence")
        if isinstance(dispute_evidence, dict):
            target_evidence = target.get("dispute_evidence", {})
            require(isinstance(target_evidence, dict), "dispute_evidence")
            for item_id, evidence in dispute_evidence.items():
                if allowed_item_ids is not None and item_id not in allowed_item_ids:
                    continue
                if not isinstance(evidence, str) or not _fragments(evidence):
                    continue
                candidate = target_evidence.get(item_id)
                require(
                    isinstance(candidate, str)
                    and _normalized(evidence) in _normalized(candidate),
                    "dispute_evidence",
                )

    # A format repair must not adjudicate signed requirements. Preserve every
    # already-valid requirement ID/disposition pair and its substantive
    # evidence; malformed rows remain available for schema-required repair.
    dispositions = source.get("human_requirement_dispositions")
    if isinstance(dispositions, list):
        allowed_requirement_ids = None
        if surfaced_requirement_ids is not None:
            allowed_requirement_ids = {
                _normalize_requirement_label(item) for item in surfaced_requirement_ids
            }
        target_dispositions = target.get("human_requirement_dispositions", [])
        require(isinstance(target_dispositions, list), "human_requirement_dispositions")
        for entry in dispositions:
            if not isinstance(entry, dict):
                continue
            requirement_id = entry.get("requirement_id")
            disposition = entry.get("disposition")
            if (not isinstance(requirement_id, str) or not requirement_id.strip()
                    or disposition not in _HUMAN_REQUIREMENT_DISPOSITIONS):
                continue
            normalized_requirement_id = _normalized_requirement_id(requirement_id)
            if normalized_requirement_id is None:
                # Reviewer item IDs and arbitrary labels are not signed requirements.
                continue
            if (allowed_requirement_ids is not None
                    and normalized_requirement_id not in allowed_requirement_ids):
                continue
            match = next(
                (
                    candidate for candidate in target_dispositions
                    if isinstance(candidate, dict)
                    and isinstance(candidate.get("requirement_id"), str)
                    and _normalized_requirement_id(candidate["requirement_id"])
                    == normalized_requirement_id
                    and candidate.get("disposition") == disposition
                ),
                None,
            )
            require(match is not None, "human_requirement_dispositions")
            evidence = entry.get("evidence")
            if isinstance(evidence, str) and _fragments(evidence):
                candidate_evidence = match.get("evidence")
                require(
                    isinstance(candidate_evidence, str)
                    and _normalized(evidence) in _normalized(candidate_evidence),
                    "human_requirement_dispositions",
                )

    # A finding may move between current blocking/same-scope buckets, but it
    # must remain a separate finding carrying every original text fragment.
    finding_fields = {
        "pr_review": ("blocking_items", "same_pr_followups"),
        "plan_review": ("blocking_plan_issues", "same_plan_followups"),
    }.get(source["kind"], ())
    available = [
        _normalized(entry)
        for field in finding_fields
        for entry in target.get(field, [])
        if isinstance(entry, str)
    ]
    available.sort(key=len)
    for field in finding_fields:
        entries = source.get(field)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            fragments = [_normalized(text) for text in _fragments(entry)]
            if not fragments:
                continue
            match = next(
                (i for i, candidate in enumerate(available)
                 if all(text in candidate for text in fragments)), None,
            )
            require(match is not None, field)
            available.pop(match)
