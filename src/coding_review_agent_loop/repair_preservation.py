"""Bounded loss checks for parseable review and implementation repair inputs."""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import re

from .errors import AgentLoopError
from .protocol import (
    ARCHITECTURE_IMPACT_NEAR_MISS_RULE,
    ARCHITECTURE_IMPACT_STATUS_ALIASES,
    ParseDegradation,
    _extract_json_object_prefix,
    architecture_impact_near_miss_corroborated,
    _normalize_requirement_label,
    normalize_response_file_structured_text,
    parse_plan_revision_patch,
)
from .protocol_markers import (
    RESERVED_MARKER_REGISTRY,
    historical_replacement_labels,
    historical_text_fragments,
)


_KINDS = {
    "pr_review", "plan_review", "plan_revision", "plan_revision_patch", "plan_state",
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


REVIEW_KINDS = frozenset({"plan_review", "pr_review"})

# Bucket names carried by each review kind.  Order is (blocking, same-scope,
# future); the first two are the current-scope buckets.
REVIEW_FINDING_BUCKETS = {
    "plan_review": ("blocking_plan_issues", "same_plan_followups", "future_followups"),
    "pr_review": ("blocking_items", "same_pr_followups", "future_followups"),
}
REVIEW_DISPOSITION_FIELD = {
    "plan_review": "prior_plan_item_dispositions",
    "pr_review": "prior_item_dispositions",
}
# The kind's non-resolving active disposition values.  A carried item left in
# one of these states is still open work.
REVIEW_ACTIVE_DISPOSITIONS = {
    "plan_review": frozenset({"blocking", "same-plan"}),
    "pr_review": frozenset({"blocking", "same-pr"}),
}
# Fields unique to one review schema.  A kindless payload is admitted to repair
# only on the strength of one of these; `summary`, `state`, `schema_version`,
# `future_followups`, `human_requirement_dispositions`, and
# `architecture_impact` are shared with other response schemas and are never
# admission evidence.
REVIEW_KIND_UNIQUE_FIELDS = {
    "plan_review": frozenset({
        "blocking_plan_issues", "same_plan_followups", "prior_plan_item_dispositions",
    }),
    "pr_review": frozenset({
        "blocking_items", "same_pr_followups", "prior_item_dispositions",
    }),
}

# The repair prompt's `### Invalid enum values:` normalization table.  A drift
# assertion derives these pairs from `_REPAIR_PROMPT` itself.
DISPOSITION_VALUE_ALIASES = {
    "still blocking": "blocking",
    "still same-pr": "same-pr",
    "still same-plan": "same-plan",
}

# (A) A literal closed stop list.  Pinned verbatim by a test.
GROUNDING_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "no", "not", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "will", "with",
})

# (B) Schema vocabulary derived from the two review parsers' key names and
# enum values.  Repair may supply these words while rewriting an envelope.
REVIEW_SCHEMA_VOCABULARY = frozenset({
    "blocking_plan_issues", "same_plan_followups", "blocking_items",
    "same_pr_followups", "future_followups", "prior_plan_item_dispositions",
    "prior_item_dispositions", "summary", "state", "kind", "schema_version",
    "item_id", "disposition", "note", "reviewer", "architecture_impact",
    "human_requirement_dispositions",
    "approved", "blocking", "same-plan", "same-pr", "future", "resolved",
})

# (C) Carried item IDs are structural identifiers, not reviewer prose.
ITEM_ID_PATTERN = re.compile(r"(?i)\bitem[-_ ]?\d+\b")

# Polarity and limiting qualifiers whose deletion, addition, or substitution
# inverts a matched finding's meaning.  Exempt from both the stop list and the
# minimum-length rule so they stay visible to the comparison.
SEMANTIC_MODIFIERS = frozenset({
    "no", "not", "never", "none", "neither", "nor", "cannot", "without",
    "unless", "except", "only", "always", "must", "should", "may", "optional",
    "required", "all", "any", "every", "some", "most", "least", "more", "less",
    "fewer", "before", "after", "until",
})

# Applied BEFORE punctuation is stripped: the tokenizer splits on
# non-alphanumerics, so a contracted negation would otherwise be destroyed and
# could never be counted.  Both apostrophe forms are normalized.
CONTRACTION_NORMALIZATIONS = (
    ("can't", "cannot"),
    ("cannot", "cannot"),
    ("won't", "will not"),
    ("shan't", "shall not"),
)
_APOSTROPHE_FORMS = ("'", "\u2019")
_GENERAL_CONTRACTED_NEGATION_RE = re.compile(r"(?i)\b([a-z]+)n['\u2019]t\b")

COVERAGE_PHRASES = (
    "already covered",
    "is covered by",
    "covers this",
    "addressed by the current plan",
    "addressed by the current pr",
    "handled by the current plan",
    "handled by the current pr",
)
NEGATION_MARKERS = (
    "not", "never", "nor", "cannot", "without", "fails to", "yet to be",
    "rather than",
)
_PINNED_STILL_OPEN_PHRASES = (
    "still open", "still missing", "still blocking", "remains open",
    "remains unresolved",
)
# Generated from the coverage list so every coverage phrase automatically
# carries its negated counterparts.
STILL_OPEN_PHRASES = frozenset(_PINNED_STILL_OPEN_PHRASES) | frozenset(
    f"{marker} {phrase}"
    for marker in NEGATION_MARKERS
    if " " not in marker
    for phrase in COVERAGE_PHRASES
)

_SENTENCE_SPLIT_RE = re.compile(r"[.;\n]|(?:^|\s)[-*\u2022]\s")


def normalize_contractions(text: str) -> str:
    """Rewrite contracted negatives to their canonical words."""
    for contraction, replacement in CONTRACTION_NORMALIZATIONS:
        for apostrophe in _APOSTROPHE_FORMS:
            spelling = contraction.replace("'", apostrophe)
            text = re.compile(re.escape(spelling), re.IGNORECASE).sub(replacement, text)
    return _GENERAL_CONTRACTED_NEGATION_RE.sub(r"\1 not", text)


def _neutralization_label_tokens() -> frozenset[str]:
    """(D) Neutralization labels derived from the marker registry."""
    tokens: set[str] = set()
    for definition in RESERVED_MARKER_REGISTRY:
        for token in re.split(r"[^0-9a-z]+", definition.safe_label.casefold()):
            if token:
                tokens.add(token)
    return frozenset(tokens)


NEUTRALIZATION_LABEL_TOKENS = _neutralization_label_tokens()

_SCHEMA_VOCABULARY_TOKENS = frozenset(
    token
    for entry in REVIEW_SCHEMA_VOCABULARY
    for token in re.split(r"[^0-9a-z]+", entry.casefold())
    if token
)


def _raw_tokens(text: str) -> list[str]:
    normalized = ITEM_ID_PATTERN.sub(" ", normalize_contractions(text)).casefold()
    return [token for token in re.split(r"[^0-9a-z]+", normalized) if token]


def _content_tokens(text: str) -> list[str]:
    """Tokens that must be supported by the source."""
    kept: list[str] = []
    for token in _raw_tokens(text):
        if token in SEMANTIC_MODIFIERS:
            kept.append(token)
            continue
        if len(token) < 3 or token.isdigit():
            continue
        if token in GROUNDING_STOP_WORDS or token in _SCHEMA_VOCABULARY_TOKENS:
            continue
        if token in NEUTRALIZATION_LABEL_TOKENS:
            continue
        kept.append(token)
    return kept


def _modifier_counts(text: str) -> Counter:
    return Counter(token for token in _raw_tokens(text) if token in SEMANTIC_MODIFIERS)


def _joined_text(value: object) -> str:
    return " ".join(fragment for fragment in _fragments(value) if fragment)


def _raw_string_values(value: object) -> list[str]:
    """Substantive strings of *value* WITHOUT reserved-marker stripping."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for child in value for text in _raw_string_values(child)]
    if isinstance(value, dict):
        return [
            text for key, child in value.items()
            if key not in _FINDING_METADATA_KEYS
            for text in _raw_string_values(child)
        ]
    return []


def _normalized_label(text: str) -> str:
    return " ".join(text.casefold().split())


def _authorized_neutralization_labels(value: object) -> Counter:
    """Safe labels that may legally replace the markers *value* embeds.

    `_joined_text` strips markers, so a marker-only source finding and a
    genuinely empty one such as `{}` both flatten to the empty string. Marker
    provenance is therefore read from the unstripped strings, and the identity AND
    the occurrence count of the markers found are kept: the documented exception
    replaces EACH source marker with ITS OWN authorized safe label, so an empty
    result means the entry is not marker-only, a different family is unsupported,
    and two occurrences may not collapse into one (#871).

    Occurrences come from the registry's historical replacement spans — the same
    set the stripping pass uses — rather than the non-overlapping scan, so a
    malformed name-bearing-line fallback cannot hide a second reserved token on
    its own line and let the repair drop that occurrence unnoticed.
    """
    counts: Counter = Counter()
    for text in _raw_string_values(value):
        for label in historical_replacement_labels(text):
            counts[_normalized_label(label)] += 1
    return counts


def _consume_neutralization_labels(
    text: str, labels: Sequence[str]
) -> tuple[Counter, str]:
    """Split *text* into the *labels* it spells out and the remaining prose."""
    counts: Counter = Counter()
    remaining = _normalized_label(text)
    for label in sorted(labels, key=len, reverse=True):
        while label and label in remaining:
            counts[label] += 1
            remaining = " ".join(remaining.replace(label, " ", 1).split())
    return counts, remaining


_PROTOCOL_RECORD_LINE_RE = re.compile(r"\A(?:<!--.*-->|--\s+\S.*)\Z", re.DOTALL)
_LIST_ITEM_RE = re.compile(r"\A[-*\u2022]\s+(?P<body>.*)\Z", re.DOTALL)


def _freeform_finding_candidates(text: str) -> list[str]:
    """Non-overlapping reviewer-prose candidates drawn from freeform text.

    Each concern must appear exactly once. A paragraph contributes EITHER its
    individual lines (when it is a bulleted block) OR its joined prose, never
    both: emitting a line and then the paragraph containing it would give one
    trailing reviewer statement two equivalent candidates, which would let
    repair duplicate it into two findings, match each copy injectively, and
    raise the correspondence ceiling (#871).

    Protocol footer and signature lines are tool-owned structural records, not
    reviewer prose, so they are never candidates.
    """
    candidates: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in block.splitlines()]
        lines = [
            line for line in lines
            if line and not _PROTOCOL_RECORD_LINE_RE.match(line)
        ]
        if not lines:
            continue
        if any(_LIST_ITEM_RE.match(line) for line in lines):
            # A bulleted block states one concern per list ITEM, not per
            # physical line: a wrapped bullet's continuation lines join the item
            # they belong to, so one wrapped bullet stays one candidate and
            # cannot be matched twice (#871).
            items: list[list[str]] = []
            for line in lines:
                match = _LIST_ITEM_RE.match(line)
                if match is not None:
                    items.append([match.group("body").strip()])
                elif items:
                    items[-1].append(line)
                # A lead-in line such as `Review concerns:` before the first
                # bullet introduces the list. It is structure, not a concern, so
                # it is dropped entirely: keeping it as its own candidate let
                # repair emit the heading as a second ledger finding, and
                # prepending it to the first item let a repaired finding match on
                # the heading's tokens alone while omitting the real concern
                # (#871).
            for item in items:
                joined_item = " ".join(" ".join(item).split())
                if joined_item:
                    candidates.append(joined_item)
            continue
        joined = " ".join(" ".join(lines).split())
        if joined:
            candidates.append(joined)
    return candidates


def coverage_predicate(text: str) -> bool:
    """Negation-safe test for `the current plan/PR already covers this`."""
    if not isinstance(text, str) or not text.strip():
        return False
    normalized = " ".join(normalize_contractions(text).casefold().split())
    if any(phrase in normalized for phrase in STILL_OPEN_PHRASES):
        return False
    for sentence in _SENTENCE_SPLIT_RE.split(normalized):
        if not sentence:
            continue
        for phrase in COVERAGE_PHRASES:
            index = sentence.find(phrase)
            while index != -1:
                prefix = sentence[:index]
                if not any(
                    re.search(rf"\b{re.escape(marker)}\b", prefix)
                    for marker in NEGATION_MARKERS
                ):
                    return True
                index = sentence.find(phrase, index + 1)
    return False


def coverage_predicate_for_item(text: str, item_id: object) -> bool:
    """Run the coverage predicate over only the segments naming *item_id*."""
    if not isinstance(text, str) or not isinstance(item_id, str) or not item_id.strip():
        return False
    needle = item_id.casefold()
    segments = [
        segment
        for segment in _SENTENCE_SPLIT_RE.split(
            " ".join(normalize_contractions(text).casefold().split())
        )
        if needle in segment
    ]
    return any(coverage_predicate(segment) for segment in segments)


def _normalized_disposition(value: object) -> object:
    if isinstance(value, str):
        return DISPOSITION_VALUE_ALIASES.get(" ".join(value.split()).casefold(), value)
    return value


def _validate_review_grounding(
    raw: str,
    source: dict | None,
    target: dict,
    *,
    target_kind: str,
    allowed_prior_item_ids: Sequence[str] | None,
) -> None:
    """Require every repaired reviewer verdict to be supported by the source.

    Token coverage bounds added vocabulary and matched-pair modifier counts
    bound polarity inversion.  This is a bounded support check, not a
    certification that the reviewer's finding is correct.
    """

    def reject(detail: str) -> None:
        raise AgentLoopError(
            f"Repair content preservation failed for {target_kind} grounding: {detail}. "
            "A repaired review must be supported by the reviewer's own source text."
        )

    if not isinstance(source, dict):
        reject("the source carries no mechanically recoverable review payload")
        return
    if "kind" in source and source["kind"] != target_kind:
        # A present-but-invalid source kind (empty string, null, any non-string)
        # is not the repaired kind either, so it fails closed exactly like an
        # explicit mismatch instead of slipping past a string-only comparison.
        reject(f"the source payload declares kind {source['kind']!r}")

    source_tokens = set(_content_tokens(raw))

    def supported(value: object) -> bool:
        if value is None:
            return True
        if not isinstance(value, str):
            return False
        return set(_content_tokens(value)) <= source_tokens

    buckets = REVIEW_FINDING_BUCKETS[target_kind]
    current_scope_buckets = buckets[:2]

    def bucket_entries(payload: dict, names: Sequence[str]) -> list[object]:
        entries: list[object] = []
        for name in names:
            value = payload.get(name)
            if isinstance(value, list):
                entries.extend(value)
        return entries

    source_findings = bucket_entries(source, buckets)
    source_candidates: list[str] = []
    # Parallel to source_candidates: for a source finding whose prose is empty
    # BECAUSE it was nothing but reserved markers, the safe labels those markers
    # authorize; empty for every other candidate.
    candidate_marker_labels: list[Counter] = []
    for entry in source_findings:
        entry_text = _joined_text(entry)
        if _content_tokens(entry_text) or _modifier_counts(entry_text):
            source_candidates.append(entry_text)
            candidate_marker_labels.append(Counter())
        elif (marker_labels := _authorized_neutralization_labels(entry)):
            source_candidates.append(entry_text)
            candidate_marker_labels.append(marker_labels)
        # A genuinely empty entry such as `{}` carries no reviewer content at
        # all. It is dropped rather than kept: keeping it would both hand repair
        # a wildcard for an exempt-token-only finding and raise the
        # correspondence ceiling on the strength of nothing (#871).
    if not source_findings:
        # The payload declares no finding in any bucket. Only freeform prose
        # OUTSIDE the recovered JSON object can be a reviewer finding here: the
        # object's own fields are structured data, and `summary` in particular is
        # not a finding. Splitting the serialized payload into prose segments
        # would let an approved source's summary be copied into a current-scope
        # blocking finding and then ground the inverted verdict (#871).
        source_candidates = _freeform_finding_candidates(_payload_and_trailing(raw)[1])
        candidate_marker_labels = [Counter() for _ in source_candidates]

    # Each target finding keeps the safe labels of the markers IT embeds, so a
    # raw marker in the repaired text can be compared against the source
    # candidate's own marker families rather than merely stripping to nothing.
    target_findings: list[tuple[str, str, Counter]] = []
    for name in buckets:
        value = target.get(name)
        if not isinstance(value, list):
            continue
        for entry in value:
            target_findings.append((
                name, _joined_text(entry), _authorized_neutralization_labels(entry),
            ))

    # The ceiling applies to the freeform fallback too, so a payload declaring no
    # finding cannot gain one from a shorter list of prose segments.
    if len(target_findings) > len(source_candidates):
        reject(
            f"the repaired review carries {len(target_findings)} findings while the "
            f"source carries {len(source_candidates)} corresponding candidates"
        )

    candidate_tokens = [set(_content_tokens(text)) for text in source_candidates]
    candidate_modifiers = [_modifier_counts(text) for text in source_candidates]

    matched_candidate_by_finding: dict[int, int] = {}
    matched_finding_by_candidate: dict[int, int] = {}

    def can_match(finding_index: int, candidate_index: int) -> bool:
        _name, text, target_labels = target_findings[finding_index]
        tokens = set(_content_tokens(text))
        if not candidate_tokens[candidate_index] and not candidate_modifiers[candidate_index]:
            # A candidate with no prose corresponds to nothing unless it is a
            # genuine marker-only source finding, and then ONLY to that entry's
            # own authorized neutralization — the marker kept verbatim (stripped
            # to nothing) or replaced by one of its own safe labels. Accepting
            # any exempt-only target instead would fabricate review substance:
            # schema vocabulary, stop words and every registry safe label are
            # exempt, so a target finding of `blocking`, an unrelated marker's
            # label, or stop-word-only prose would carry no content tokens and
            # match, and whole-source coverage cannot tell a finding apart from
            # the rest of the source text (#871).
            labels = candidate_marker_labels[candidate_index]
            if not labels or _modifier_counts(text):
                return False
            # Marker IDENTITY and CARDINALITY both have to hold, so the target
            # must represent the COMPLETE source marker multiset: each occurrence
            # either kept verbatim — which `_joined_text` strips, so it is counted
            # from the target's own raw markers — or replaced by its own safe
            # label, which survives as text. An unrelated family, a dropped
            # occurrence, a duplicated one, and any leftover prose are all
            # refused, because every safe label is an exempt token and would
            # otherwise match vacuously (#871).
            spelled, remaining = _consume_neutralization_labels(text, labels)
            if remaining:
                return False
            return target_labels + spelled == labels
        if not tokens and not _modifier_counts(text):
            # The candidate is substantive, so the repaired finding must retain
            # substantive content of its own. An exempt-only target — schema
            # vocabulary such as `blocking`, or stop-word-only prose — has an
            # empty content-token set, which is trivially a subset of any
            # candidate, and whole-source coverage is vacuous for it, so without
            # this guard repair could replace a real reviewer finding with a
            # fabricated placeholder and still ground a blocking verdict. A
            # modifier-only candidate is bounded by the equality rule below,
            # which forces the target to carry those same modifiers (#871).
            return False
        if not tokens <= candidate_tokens[candidate_index]:
            return False
        # Modifier-count equality applies to EVERY matched pair, including a
        # freeform fallback candidate. A trailing-prose finding is still a
        # matched source/target pair, and subset coverage alone cannot detect a
        # deleted negation or limiting qualifier, so exempting the fallback
        # would let `This path is not exploitable` be repaired into `This path
        # is exploitable` (#871).
        return _modifier_counts(text) == candidate_modifiers[candidate_index]

    def augment(finding_index: int, visited: set[int]) -> bool:
        for candidate_index in range(len(source_candidates)):
            if candidate_index in visited or not can_match(finding_index, candidate_index):
                continue
            visited.add(candidate_index)
            previous = matched_finding_by_candidate.get(candidate_index)
            if previous is None or augment(previous, visited):
                matched_finding_by_candidate[candidate_index] = finding_index
                matched_candidate_by_finding[finding_index] = candidate_index
                return True
        return False

    for finding_index, (name, text, _labels) in enumerate(target_findings):
        if not supported(text):
            reject(f"`{name}` carries content absent from the source")
        if not augment(finding_index, set()):
            reject(
                f"`{name}` has no distinct corresponding source finding with the same "
                "negations and limiting qualifiers"
            )

    summary = target.get("summary")
    if isinstance(summary, str) and not supported(summary):
        reject("`summary` carries content absent from the source")

    disposition_field = REVIEW_DISPOSITION_FIELD[target_kind]
    active_values = REVIEW_ACTIVE_DISPOSITIONS[target_kind]
    source_disposition_entries = source.get(disposition_field)
    source_dispositions: dict[object, dict] = {}
    if isinstance(source_disposition_entries, list):
        for entry in source_disposition_entries:
            if isinstance(entry, dict) and entry.get("item_id") is not None:
                source_dispositions.setdefault(entry["item_id"], entry)
    source_has_active_disposition = any(
        _normalized_disposition(entry.get("disposition")) in active_values
        for entry in source_dispositions.values()
    )

    allowed_ids = set(allowed_prior_item_ids or ())
    target_disposition_entries = target.get(disposition_field)
    # Only an active target disposition that PRESERVES an active SOURCE
    # disposition can ground a blocking verdict.  A disposition completed from
    # `allowed_prior_item_ids`, or one promoted out of a source `future`, is
    # supplied by the repair context or by the target's own state and is not
    # reviewer-authored evidence of open work; counting it would let an
    # approved, finding-free source be repaired into a blocking review.
    target_preserves_active_source_disposition = False
    if isinstance(target_disposition_entries, list):
        for entry in target_disposition_entries:
            if not isinstance(entry, dict):
                continue
            item_id = entry.get("item_id")
            disposition = entry.get("disposition")
            note = entry.get("note")
            source_entry = source_dispositions.get(item_id)
            if source_entry is None:
                # Completion of a carried ID supplied by the repair context.
                if item_id not in allowed_ids:
                    reject(
                        f"`{disposition_field}` carries item `{item_id}`, which the source "
                        "never dispositioned and the repair context never allowed"
                    )
                if disposition == "resolved":
                    if not coverage_predicate_for_item(raw, item_id):
                        reject(
                            f"`{disposition_field}` resolves carried item `{item_id}` without "
                            "source support"
                        )
                elif disposition not in active_values:
                    reject(
                        f"`{disposition_field}` completes carried item `{item_id}` with "
                        f"disposition `{disposition}` instead of an active disposition"
                    )
                if isinstance(note, str) and note.strip() and not supported(note):
                    reject(
                        f"`{disposition_field}` note for `{item_id}` is absent from the source"
                    )
                continue
            source_disposition = _normalized_disposition(source_entry.get("disposition"))
            source_note = source_entry.get("note")
            source_entry_text = _joined_text(source_entry)
            if disposition in active_values and source_disposition in active_values:
                target_preserves_active_source_disposition = True
            if disposition != source_disposition:
                authorized_resolution = (
                    disposition == "resolved"
                    and source_disposition in active_values
                    and coverage_predicate(source_note if isinstance(source_note, str) else "")
                )
                # The schema forbids a `future` disposition in a blocking
                # review while still requiring every carried item to appear, so
                # the repair prompt must re-state it.  Permit only the
                # non-resolving active value, which keeps the item open.
                schema_mandated_future_change = (
                    source_disposition == "future"
                    and target.get("state") == "blocking"
                    and disposition in active_values
                )
                if not (authorized_resolution or schema_mandated_future_change):
                    reject(
                        f"`{disposition_field}` changes item `{item_id}` from "
                        f"`{source_disposition}` to `{disposition}` without source support"
                    )
            if isinstance(note, str) and note.strip():
                entry_tokens = set(_content_tokens(source_entry_text))
                if not set(_content_tokens(note)) <= entry_tokens:
                    reject(
                        f"`{disposition_field}` note for `{item_id}` is absent from the "
                        "source entry"
                    )
                if isinstance(source_note, str) and source_note.strip() and (
                    _modifier_counts(note) != _modifier_counts(source_note)
                ):
                    reject(
                        f"`{disposition_field}` note for `{item_id}` drops, adds, or "
                        "substitutes a negation or limiting qualifier"
                    )

    source_state = source.get("state")
    source_state = source_state if isinstance(source_state, str) else None
    target_state = target.get("state")
    if not isinstance(target_state, str):
        # A repaired review with no declared verdict carries nothing to ground;
        # the strict schema validator rejects it on its own.
        return
    source_current_findings = bool(bucket_entries(source, current_scope_buckets))
    target_matched_current = any(
        name in current_scope_buckets and index in matched_candidate_by_finding
        for index, (name, _text, _labels) in enumerate(target_findings)
    )
    if target_state == "blocking":
        if not (
            source_state == "blocking"
            or target_matched_current
            or target_preserves_active_source_disposition
        ):
            reject(
                "`state: blocking` is supported by no source blocking state, preserved "
                "source finding, or active carried disposition"
            )
    elif target_state == "approved":
        if source_state != "approved":
            reject("`state: approved` is not supported by an approved source state")
        if source_current_findings or source_has_active_disposition:
            reject(
                "`state: approved` cannot be manufactured while the source itself carries "
                "current-scope findings or active carried dispositions"
            )
    else:
        reject(f"`state` value {target_state!r} is not a grounded review verdict")


def _schema_valid_architecture_entry(value: object) -> bool:
    """Return whether one architecture-list entry can pass schema validation."""
    return isinstance(value, str) and bool(value.strip())


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


def require_recoverable_semantic_patch(raw: str) -> None:
    """Require a complete semantic patch before any model repair is attempted.

    A semantic patch is a bounded decision set, not a prose response that a
    repair model may reconstruct.  Envelope-only repair is safe once this
    complete payload has been recovered; a missing or malformed payload must
    return to the planner instead.
    """
    payload = _payload(raw)
    if not isinstance(payload, dict) or payload.get("kind") != "plan_revision_patch":
        raise AgentLoopError(
            "Semantic patch repair requires a mechanically recoverable complete patch payload."
        )
    try:
        parse_plan_revision_patch(payload)
    except AgentLoopError as exc:
        raise AgentLoopError(
            "Semantic patch repair requires a mechanically recoverable complete patch payload."
        ) from exc


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
    allowed_prior_item_ids: Sequence[str] | None = None,
    allow_legacy_matrix_removal: bool = False,
    forbid_architecture_impact: bool = False,
) -> None:
    """Reject observable losses, not certify semantic equivalence.

    Invalid JSON and unsupported schemas remain on the ordinary repair path.
    Do not heuristically parse broken JSON or interpret prose as item ledgers.
    """
    source, target = _payload(raw), _payload(repaired)
    if forbid_architecture_impact:
        require_repair_architecture_impact_absent(repaired)
    # Reviewer grounding is triggered by the repaired TARGET kind, so it also
    # covers a legacy repair entry point and an absent or wrong-kind source,
    # neither of which reaches the loss checks below.
    if isinstance(target, dict) and target.get("kind") in REVIEW_KINDS:
        _validate_review_grounding(
            raw,
            source,
            target,
            target_kind=target["kind"],
            allowed_prior_item_ids=allowed_prior_item_ids,
        )
    if (not source or not target or not isinstance(source.get("kind"), str)
            or source["kind"] not in _KINDS):
        return
    if source.get("kind") != target.get("kind"):
        return  # Kind selection belongs to the caller's schema/context validator.

    # A semantic patch is the complete bounded decision set. Repair can fix
    # only its envelope/footer; it must not invent or rewrite operations,
    # rationales, or authenticated base bindings.
    if source["kind"] == "plan_revision_patch":
        if source != target:
            raise AgentLoopError(
                "Repair content preservation failed for plan_revision_patch: "
                "semantic operations and base binding must be preserved exactly."
            )
        return

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
        if "risk_test_matrix_claims" in source:
            source_claims = source.get("risk_test_matrix_claims")
            target_claims = target.get("risk_test_matrix_claims")
            require(isinstance(source_claims, list), "risk_test_matrix_claims")
            require(isinstance(target_claims, list), "risk_test_matrix_claims")
            # Claims remain semantic, but a repair may remove an invalid row or
            # selector after the validator names the defect. Preserve every
            # substantive selector/fact from claims that are retained.
            for source_claim in source_claims:
                if not isinstance(source_claim, dict):
                    continue
                source_row = source_claim.get("row_id")
                target_matches = [
                    candidate for candidate in target_claims
                    if isinstance(candidate, dict) and candidate.get("row_id") == source_row
                ]
                if not target_matches:
                    continue
                candidate = target_matches[0]
                for field in (
                    "execution_refs", "test_identifiers", "test_locations",
                    "workflow_path_claim", "outcome_assertions",
                    "forbidden_effect_assertions",
                ):
                    if field in source_claim:
                        require(field in candidate, f"risk_test_matrix_claims.{field}")
                        if field == "execution_refs":
                            # The schema/catalog validator decides which
                            # selectors are invalid. Permit repair to remove
                            # those selectors, including one copy of a
                            # duplicate, but never let it add or replace a
                            # selector that was absent from the source claim.
                            source_refs = source_claim[field]
                            target_refs = candidate[field]
                            require(isinstance(source_refs, list), f"risk_test_matrix_claims.{field}")
                            require(isinstance(target_refs, list), f"risk_test_matrix_claims.{field}")
                            remaining_refs = list(source_refs)
                            for target_ref in target_refs:
                                require(
                                    target_ref in remaining_refs,
                                    f"risk_test_matrix_claims.{field}",
                                )
                                remaining_refs.remove(target_ref)
                            continue
                        source_fragments = _fragments(source_claim[field])
                        target_fragments = _fragments(candidate[field])
                        require(
                            all(fragment in target_fragments for fragment in source_fragments),
                            f"risk_test_matrix_claims.{field}",
                        )
        if "risk_test_matrix_evidence" in source:
            # Canonical evidence is orchestrator-owned. Fresh semantic repair
            # may remove this legacy field, but historical compatibility still
            # permits an exact unchanged copy in a replayed response. Never
            # permit a model to rewrite it into a different canonical record.
            if "risk_test_matrix_evidence" in target:
                require(
                    target["risk_test_matrix_evidence"]
                    == source["risk_test_matrix_evidence"],
                    "risk_test_matrix_evidence",
                )
        elif "risk_test_matrix_evidence" in target:
            # A fresh repair may remove a legacy canonical field that was
            # present in its source, but it may never synthesize that field
            # when the source did not contain it.
            raise AgentLoopError(
                "repair cannot invent risk_test_matrix_evidence for a fresh coder response"
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


def require_repair_architecture_impact_absent(repaired: str) -> None:
    """Pin the absence of a required assessment through repair (#925).

    When a required-contract source carries no assessment -- genuinely
    omitted, or removed by pre-repair near-miss normalization -- a repair
    model must not supply one: a fabricated `unchanged` would launder an
    unsupplied assessment into a satisfied contract.
    """
    target = _payload(repaired)
    if isinstance(target, dict) and "architecture_impact" in target:
        raise AgentLoopError(
            "Repair content preservation failed for architecture_impact: the source "
            "carried no assessment, so repair must not introduce one."
        )


@dataclass(frozen=True)
class ArchitectureNearMissNormalization:
    """Pre-repair normalization result; the record stays out of band."""

    raw: str
    record: ParseDegradation | None
    forbid_architecture_impact: bool


def normalize_architecture_impact_near_miss(
    raw: str, *, required_contract: bool, expected_kind: str | None = None
) -> ArchitectureNearMissNormalization:
    """Resolve the status near miss in the raw payload before repair (#924).

    Deterministic normalization runs before the LLM repair pass.  A
    corroborated `modified` becomes a wire-valid `changed` that preservation
    then pins; an uncorroborated one has its whole optional object removed,
    which is exactly what the parser-only `undetermined` status stands for.
    Any other payload, including an unparseable one, is returned unchanged.

    The absence pin applies to a recoverable source of the expected kind.  A
    wrong-kind source is a kind-selection defect that preservation already
    leaves to the caller's validator, exactly as before.
    """
    payload, trailing = _payload_and_trailing(raw)
    record: ParseDegradation | None = None
    normalized = raw
    impact = payload.get("architecture_impact") if isinstance(payload, dict) else None
    status = impact.get("status") if isinstance(impact, dict) else None
    if isinstance(payload, dict) and isinstance(status, str) and status in ARCHITECTURE_IMPACT_STATUS_ALIASES:
        kind = payload.get("kind") if isinstance(payload.get("kind"), str) else "response"
        corroborated = architecture_impact_near_miss_corroborated(impact)
        rewritten = dict(payload)
        if corroborated:
            rewritten["architecture_impact"] = {
                **impact, "status": ARCHITECTURE_IMPACT_STATUS_ALIASES[status],
            }
        else:
            rewritten.pop("architecture_impact")
        record = ParseDegradation.build(
            element_path=f"{kind}.architecture_impact.status",
            rule=ARCHITECTURE_IMPACT_NEAR_MISS_RULE,
            observed=status,
            outcome="normalized-to-changed" if corroborated else "degraded-to-undetermined",
        )
        normalized = json.dumps(rewritten, ensure_ascii=False) + (
            trailing if trailing.startswith(("\n", "\r")) else "\n" + trailing.lstrip()
        )
        payload = rewritten
    forbid = bool(
        required_contract
        and isinstance(payload, dict)
        and "architecture_impact" not in payload
        and (expected_kind is None or payload.get("kind") == expected_kind)
    )
    return ArchitectureNearMissNormalization(
        raw=normalized, record=record, forbid_architecture_impact=forbid
    )
