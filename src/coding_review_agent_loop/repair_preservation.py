"""Bounded loss checks for parseable review and implementation repair inputs."""

from .errors import AgentLoopError
from .protocol import _extract_json_object_prefix, normalize_response_file_structured_text
from .protocol_markers import scan_reserved_markers


_KINDS = {
    "pr_review", "plan_review", "plan_revision", "plan_state",
    "coder_followup", "issue_implementation",
}


def _payload(text: str) -> dict | None:
    text, _ = normalize_response_file_structured_text(text)
    try:
        parsed = _extract_json_object_prefix(text)
    except AgentLoopError:
        return None
    return parsed[0] if parsed else None


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _fragments(value: object) -> list[str]:
    if isinstance(value, str):
        # Reserved protocol syntax must be removable; its safety validator wins.
        return [value] if value.strip() and not scan_reserved_markers(value) else []
    if isinstance(value, list):
        return [text for child in value for text in _fragments(child)]
    if isinstance(value, dict):
        return [
            text for key, child in value.items()
            if key not in {"id", "item_id"}
            for text in _fragments(child)
        ]
    return []


def validate_repair_preservation(raw: str, repaired: str) -> None:
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
