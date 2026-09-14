"""Bounded loss checks for parseable review and implementation repair inputs."""

from collections.abc import Callable, Sequence
import re

from .errors import AgentLoopError
from .protocol import (
    _extract_json_object_prefix,
    _normalize_requirement_label,
    normalize_response_file_structured_text,
)
from .protocol_markers import historical_text_fragments


_KINDS = {
    "pr_review", "plan_review", "plan_revision", "plan_state",
    "coder_followup", "issue_implementation",
    "task_result",
}

_FENCED_JSON_PREFIX_RE = re.compile(
    r"\A[ \t]{0,3}(?P<fence>`{3,}|~{3,})[ \t]*(?:json)?[ \t]*\r?\n",
    re.IGNORECASE,
)
_FINDING_METADATA_KEYS = {
    "id", "item_id", "severity", "category", "state", "disposition", "verdict",
}
_HUMAN_REQUIREMENT_DISPOSITIONS = {"addressed", "blocked", "not-applicable"}
_ARCHITECTURE_IMPACT_KEYS = frozenset({
    "status", "rationale", "affected_components", "dependencies",
    "execution_data_flows", "execution_flows", "data_flows", "persistence",
    "public_contracts", "security_boundaries", "canonical_document_action",
    "canonical_document_path", "canonical_document_rationale", "uncertainty",
})
_ARCHITECTURE_LIST_KEYS = frozenset({
    "affected_components", "dependencies", "execution_data_flows",
    "execution_flows", "data_flows", "persistence", "public_contracts",
    "security_boundaries", "uncertainty",
})
_ARCHITECTURE_FLOW_ALIAS_KEYS = ("execution_flows", "data_flows")


def _schema_valid_architecture_entry(value: object) -> bool:
    """Return whether one architecture-list entry can pass schema validation."""
    return isinstance(value, str) and bool(value.strip())


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
        # A repair may use any safe neutralization. Require the substantive
        # prose around a marker, not the registry's particular replacement
        # label, and allow marker-only text to be replaced freely.
        return list(historical_text_fragments(value))
    if isinstance(value, list):
        return [text for child in value for text in _fragments(child)]
    if isinstance(value, dict):
        return [
            text for key, child in value.items()
            if key not in _FINDING_METADATA_KEYS
            for text in _fragments(child)
        ]
    return []


def _contains_fragments(candidate: object, fragments: Sequence[str]) -> bool:
    if not isinstance(candidate, str):
        return False
    normalized = _normalized(candidate)
    return all(_normalized(fragment) in normalized for fragment in fragments)


def _schema_valid_architecture_field(key: str, value: object) -> bool:
    """Return whether a raw architecture field can be preserved safely.

    Repair preservation also sees malformed source JSON.  Only pin fields whose
    source value could have passed the architecture schema; otherwise the repair
    must be allowed to correct that field into a schema-valid representation.
    """
    if key not in _ARCHITECTURE_IMPACT_KEYS:
        return False
    if key == "status":
        return isinstance(value, str) and value.strip() in {"changed", "unchanged"}
    if key in {"rationale", "canonical_document_action"}:
        return isinstance(value, str) and bool(value.strip())
    if key == "canonical_document_path":
        return value is None or (isinstance(value, str) and bool(value.strip()))
    if key == "canonical_document_rationale":
        # The protocol permits an omitted/empty rationale as its default.  A
        # raw null is normalized to that default and is therefore not pinned.
        return isinstance(value, str) and (not value or bool(value.strip()))
    if key in _ARCHITECTURE_LIST_KEYS:
        # A partially malformed list still contains recoverable content.  Pin
        # the field when it has at least one schema-valid entry, while allowing
        # invalid entries to be removed or corrected by the ordinary schema
        # repair.  Keep empty lists pinned because an empty list is valid.
        return isinstance(value, list) and (
            not value or any(_schema_valid_architecture_entry(item) for item in value)
        )
    return False


def _preserve_architecture_list(
    source: list[object],
    target: object,
    *,
    field: str,
    require: Callable[[bool, str], None],
) -> None:
    """Match every architecture entry to a distinct repaired entry.

    Marker-only entries have no prose fragments to compare, but their presence
    is still content.  Treat them as wildcards in the matching graph while
    retaining the source list's cardinality and one-to-one correspondence.
    The augmenting-path matcher avoids making the result depend on source
    ordering when one entry's fragments are a subset of another's.
    """
    require(isinstance(target, list), field)
    valid_source = [
        entry for entry in source if _schema_valid_architecture_entry(entry)
    ]
    # Valid source entries must survive one-for-one.  Invalid source entries
    # may be corrected into a schema-valid entry or removed, so the repaired
    # list may be shorter than the source but cannot grow beyond it.
    require(len(valid_source) <= len(target) <= len(source), field)

    matched_source_by_target: dict[int, int] = {}

    def can_match(source_entry: str, target_entry: object) -> bool:
        fragments = _fragments(source_entry)
        return not fragments or _contains_fragments(target_entry, fragments)

    def augment(source_index: int, visited_targets: set[int]) -> bool:
        for target_index, target_entry in enumerate(target):
            if target_index in visited_targets:
                continue
            if not can_match(valid_source[source_index], target_entry):
                continue
            visited_targets.add(target_index)
            previous_source_index = matched_source_by_target.get(target_index)
            if (previous_source_index is None
                    or augment(previous_source_index, visited_targets)):
                matched_source_by_target[target_index] = source_index
                return True
        return False

    for source_index in range(len(valid_source)):
        require(augment(source_index, set()), field)


def validate_repair_preservation(
    raw: str,
    repaired: str,
    *,
    unresolved_item_ids: Sequence[str] | None = None,
    surfaced_requirement_ids: Sequence[str] | None = None,
    reviewer_requirement_ids: Sequence[str] | None = None,
    allow_legacy_matrix_removal: bool = False,
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

    impact = source.get("architecture_impact")
    if isinstance(impact, dict):
        valid_impact_fields = [
            (key, value) for key, value in impact.items()
            if _schema_valid_architecture_field(key, value)
        ]
        target_impact = target.get("architecture_impact")
        require(
            not valid_impact_fields or isinstance(target_impact, dict),
            "architecture_impact",
        )
        if not valid_impact_fields:
            target_impact = {}
        else:
            assert isinstance(target_impact, dict)

        # The parser treats execution_flows/data_flows as aliases for the
        # canonical execution_data_flows list when the latter is omitted.  A
        # repair may normalize either pair, so compare missing aliases as one
        # list instead of requiring each raw spelling to survive.
        aliased_source_flows: list[object] = []
        aliased_flow_keys: set[str] = set()
        for key in _ARCHITECTURE_FLOW_ALIAS_KEYS:
            value = impact.get(key)
            if not _schema_valid_architecture_field(key, value):
                continue
            if key in target_impact:
                if "execution_data_flows" not in target_impact or all(
                    other_key in target_impact
                    for other_key in _ARCHITECTURE_FLOW_ALIAS_KEYS
                    if _schema_valid_architecture_field(other_key, impact.get(other_key))
                ):
                    _preserve_architecture_list(
                        value,
                        target_impact[key],
                        field=f"architecture_impact.{key}",
                        require=require,
                    )
                    aliased_flow_keys.add(key)
                    continue
            if "execution_data_flows" in target_impact:
                aliased_source_flows.extend(value)
                aliased_flow_keys.add(key)
        if aliased_source_flows:
            _preserve_architecture_list(
                aliased_source_flows,
                target_impact["execution_data_flows"],
                field="architecture_impact.execution_data_flows",
                require=require,
            )

        for key, value in valid_impact_fields:
            if key in aliased_flow_keys:
                continue
            field = f"architecture_impact.{key}"
            if key == "execution_data_flows" and key not in target_impact and any(
                alias in target_impact for alias in _ARCHITECTURE_FLOW_ALIAS_KEYS
            ):
                target_flow_aliases: list[object] = []
                for alias in _ARCHITECTURE_FLOW_ALIAS_KEYS:
                    alias_value = target_impact.get(alias, [])
                    require(isinstance(alias_value, list), field)
                    target_flow_aliases.extend(alias_value)
                _preserve_architecture_list(
                    value,
                    target_flow_aliases,
                    field=field,
                    require=require,
                )
                continue
            require(key in target_impact, field)
            candidate = target_impact[key]
            fragments = _fragments(value)
            if isinstance(value, str):
                if fragments:
                    exact = key in {"status", "canonical_document_action"} and len(fragments) == 1
                    require(
                        (
                            isinstance(candidate, str)
                            and _normalized(fragments[0]) == _normalized(candidate)
                            if exact
                            else _contains_fragments(candidate, fragments)
                        ),
                        field,
                    )
                elif not value:
                    require(candidate == value, field)
                else:
                    # Marker-only prose may be neutralized, but it must not
                    # change the architecture field's scalar type.
                    require(isinstance(candidate, str), field)
            elif isinstance(value, list):
                _preserve_architecture_list(
                    value,
                    candidate,
                    field=field,
                    require=require,
                )
            else:
                # Preserve schema-valid scalar values such as null exactly.
                require(type(candidate) is type(value) and candidate == value, field)

    if source["kind"] in {"plan_state", "plan_revision"} and (
        "execution_strategy_contract_version" in source
        or "execution_recommendation" in source
    ):
        require(source.get("execution_strategy_contract_version") == 1, "execution_strategy_contract_version")
        require(target.get("execution_strategy_contract_version") == 1, "execution_strategy_contract_version")
        source_recommendation = source.get("execution_recommendation")
        target_recommendation = target.get("execution_recommendation")
        require(isinstance(source_recommendation, dict), "execution_recommendation")
        require(isinstance(target_recommendation, dict), "execution_recommendation")

        def preserve_execution_value(source_value: object, target_value: object, field: str) -> None:
            if isinstance(source_value, str):
                require(
                    isinstance(target_value, str)
                    and _normalized(source_value) == _normalized(target_value),
                    field,
                )
                return
            if isinstance(source_value, list):
                require(isinstance(target_value, list) and len(source_value) == len(target_value), field)
                for index, (source_item, target_item) in enumerate(zip(source_value, target_value)):
                    preserve_execution_value(source_item, target_item, f"{field}[{index}]")
                return
            if isinstance(source_value, dict):
                require(isinstance(target_value, dict) and set(source_value) == set(target_value), field)
                for key, child in source_value.items():
                    preserve_execution_value(child, target_value[key], f"{field}.{key}")
                return
            require(source_value == target_value, field)

        preserve_execution_value(
            source_recommendation,
            target_recommendation,
            "execution_recommendation",
        )

    if source["kind"] in {"plan_state", "plan_revision"} and (
        "risk_test_matrix_contract_version" in source
        or "risk_test_matrix" in source
        or "risk_test_matrix_changes" in source
    ):
        legacy_matrix_removal = allow_legacy_matrix_removal and source["kind"] == "plan_revision"
        if legacy_matrix_removal:
            if any(
                field in target
                for field in (
                    "risk_test_matrix_contract_version",
                    "risk_test_matrix",
                    "risk_test_matrix_changes",
                )
            ):
                raise AgentLoopError(
                    "Repair content preservation failed for legacy risk test matrix: "
                    "unsolicited matrix fields must be removed together."
                )
        else:
            for field in (
                "risk_test_matrix_contract_version",
                "risk_test_matrix",
                "risk_test_matrix_changes",
            ):
                require(field in target, field)

        if not legacy_matrix_removal:
            def preserve_matrix_value(source_value: object, target_value: object, field: str) -> None:
                if isinstance(source_value, str):
                    require(isinstance(target_value, str) and _normalized(source_value) == _normalized(target_value), field)
                elif isinstance(source_value, list):
                    require(isinstance(target_value, list) and len(source_value) == len(target_value), field)
                    for index, (source_item, target_item) in enumerate(zip(source_value, target_value)):
                        preserve_matrix_value(source_item, target_item, f"{field}[{index}]")
                elif isinstance(source_value, dict):
                    require(isinstance(target_value, dict) and set(source_value) == set(target_value), field)
                    for key, child in source_value.items():
                        preserve_matrix_value(child, target_value[key], f"{field}.{key}")
                else:
                    require(type(source_value) is type(target_value) and source_value == target_value, field)

            preserve_matrix_value(source.get("risk_test_matrix_contract_version"), target.get("risk_test_matrix_contract_version"), "risk_test_matrix_contract_version")
            preserve_matrix_value(source.get("risk_test_matrix"), target.get("risk_test_matrix"), "risk_test_matrix")
            preserve_matrix_value(source.get("risk_test_matrix_changes"), target.get("risk_test_matrix_changes"), "risk_test_matrix_changes")

    summary = source.get("summary")
    summary_fragments = _fragments(summary)
    if isinstance(summary, str) and summary_fragments:
        require(
            _contains_fragments(target.get("summary"), summary_fragments),
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
            actual = target.get(field, [])
            require(isinstance(actual, list), field)
            available = list(actual)
            for entry in entries:
                fragments = _fragments(entry)
                if not fragments:
                    continue
                match = next(
                    (index for index, candidate in enumerate(available)
                     if _contains_fragments(candidate, fragments)),
                    None,
                )
                require(match is not None, field)
                available.pop(match)

    if source["kind"] in {"coder_followup", "issue_implementation"}:
        source_observations = source.get("test_observations")
        if isinstance(source_observations, list):
            target_observations = target.get("test_observations", [])
            require(isinstance(target_observations, list), "test_observations")
            for entry in source_observations:
                if isinstance(entry, dict):
                    require(entry in target_observations, "test_observations")
        if "risk_test_matrix_evidence" in source:
            require(
                target.get("risk_test_matrix_evidence") == source.get("risk_test_matrix_evidence"),
                "risk_test_matrix_evidence",
            )

    if source["kind"] == "coder_followup":
        allowed_item_ids = (
            set(unresolved_item_ids) if unresolved_item_ids is not None else None
        )
        for field in ("addressed_item_notes", "remaining_item_notes"):
            notes = source.get(field)
            if not isinstance(notes, dict):
                continue
            for item_id, note in notes.items():
                if allowed_item_ids is not None and item_id not in allowed_item_ids:
                    continue
                if not isinstance(note, str) or not _fragments(note):
                    continue
                # Disposition normalization may move a note to the other bucket.
                candidates = [
                    target[bucket].get(item_id)
                    for bucket in ("addressed_item_notes", "remaining_item_notes")
                    if isinstance(target.get(bucket), dict)
                ]
                require(any(_contains_fragments(c, _fragments(note)) for c in candidates), field)

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
                    _contains_fragments(candidate, _fragments(evidence)),
                    "dispute_evidence",
                )

    # A format repair must not adjudicate surfaced signed requirements. Preserve
    # their already-valid disposition/evidence while allowing the authoritative
    # context validator to remove fabricated rows. PR reviews prohibit this
    # field entirely, so no source row is authoritative for that kind.
    dispositions = source.get("human_requirement_dispositions")
    if isinstance(dispositions, list):
        requirement_context = surfaced_requirement_ids
        if source["kind"] == "pr_review":
            requirement_context = ()
        elif source["kind"] == "plan_review":
            requirement_context = reviewer_requirement_ids
        allowed_requirement_ids = None
        if requirement_context is not None:
            allowed_requirement_ids = {
                _normalize_requirement_label(item) for item in requirement_context
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
                    _contains_fragments(candidate_evidence, _fragments(evidence)),
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
            fragments = _fragments(entry)
            if not fragments:
                continue
            match = next(
                (i for i, candidate in enumerate(available)
                 if _contains_fragments(candidate, fragments)), None,
            )
            require(match is not None, field)
            available.pop(match)
