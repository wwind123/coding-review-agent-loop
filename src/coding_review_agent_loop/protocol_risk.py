"""Risk matrix contracts, semantic claims, and canonical evidence parsing."""

from __future__ import annotations

import copy
import json
import hashlib
import re
import shlex
from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass, field
from .errors import (
    AgentLoopError,
    NonRepairableEvidenceRejection,
    TestObservationsShapeError,
)
from .protocol_markers import sanitize_historical_text
from .test_runtime import (
    CITABLE_TEST_RUNNERS_SUMMARY,
    managed_test_wrapper_inner_command as _managed_test_wrapper_inner_command,
)
from .protocol_common import (
    PARSE_DEGRADATION_RENDER_LIMIT,
    ParseDegradation,
    TestObservationCitation,
    _expect_exact_keys,
    _expect_int,
    _expect_non_empty_string,
    _expect_object,
    _expect_string_list,
    _logger,
)

# Each dropped follow-up citation keeps its own record, so the drop count is
# bounded (fatal beyond it) to keep every record stored and rendered (#927).
CITATION_DEGRADATION_MAX_DROPS = PARSE_DEGRADATION_RENDER_LIMIT
CITATION_DROPPED_OUTCOME = "citation-dropped"
# A test_observations string that decoded to a JSON array (#1626).  At most
# one per response; it renders outside the drop-record limit.
CITATIONS_DECODED_FROM_STRING_OUTCOME = "citations-decoded-from-string"

SEMANTIC_RISK_CLAIMS_MAX_ROWS = 24
SEMANTIC_RISK_CLAIMS_MAX_EXECUTION_REFS = 8
SEMANTIC_RISK_CLAIMS_MAX_FIELD_BYTES = 1_024
SEMANTIC_RISK_CLAIMS_MAX_CAVEATS = 16
# Catalog-aware parsing drops selectors that cannot resolve to a current-turn
# broker handle (#859).  Refs up to this hard cap are dropped as a claim
# defect; anything larger is pathological input and still rejects.
SEMANTIC_RISK_CLAIMS_MAX_DROPPED_REF_BYTES = 16_384
# Each dropped ref is named in the claim caveat by this bounded prefix.
SEMANTIC_RISK_CLAIMS_DROPPED_REF_PREVIEW_CHARS = 120
# Bound on any raw value a parser drops with a record (#927), measured as the
# UTF-8 length of its compact JSON serialization.  An over-bound value
# rejects under payload-bound even when its defect would otherwise degrade.
DROPPED_VALUE_MAX_BYTES = SEMANTIC_RISK_CLAIMS_MAX_DROPPED_REF_BYTES

# Machine-owned claim-row schema shared by the parser, the fresh coder
# prompts, the post-auth correction prompt, and repair.  Keeping one source
# prevents models from having to guess exact keys (#849).
SEMANTIC_RISK_CLAIM_REQUIRED_KEYS = ("row_id", "execution_refs")
SEMANTIC_RISK_CLAIM_FACT_KEYS = (
    "test_identifiers",
    "test_locations",
    "workflow_path_claim",
    "outcome_assertions",
    "forbidden_effect_assertions",
)
SEMANTIC_RISK_CLAIM_OPTIONAL_KEYS = ("caveats",)
# Coverage-map keys (#1290) are accepted by the parser but only named in the
# producer text when the approved-plan matrix is applicable, so ungated prompts
# and the default schema text stay unchanged.
SEMANTIC_RISK_CLAIM_COVERAGE_KEYS = ("test_level",)
SEMANTIC_RISK_TEST_LEVELS = ("unit", "integration", "workflow")
SEMANTIC_RISK_COVERAGE_GAP_KEYS = ("row_id", "reason", "proposed_correction")
SEMANTIC_RISK_COVERAGE_GAPS_FIELD = "risk_test_matrix_coverage_gaps"
SEMANTIC_RISK_CLAIM_KEYS = (
    SEMANTIC_RISK_CLAIM_REQUIRED_KEYS
    + SEMANTIC_RISK_CLAIM_FACT_KEYS
    + SEMANTIC_RISK_CLAIM_OPTIONAL_KEYS
)


def semantic_risk_claim_example(
    *,
    row_id: str = "example-row-id",
    execution_ref: str = "exec-ref-from-this-turn-catalog",
    include_coverage_map: bool = False,
) -> dict[str, object]:
    """Return one complete example claim row containing every schema key."""
    example: dict[str, object] = {
        "row_id": row_id,
        "execution_refs": [execution_ref],
        "test_identifiers": ["tests/test_example.py::test_behavior"],
        "test_locations": ["tests/test_example.py"],
        "workflow_path_claim": "issue mode / fresh implementation parse",
        "outcome_assertions": ["The response parses and the row keeps its selector."],
        "forbidden_effect_assertions": ["No envelope rejection is raised."],
        "caveats": [],
    }
    if include_coverage_map:
        example["test_level"] = "unit"
    return example


def semantic_risk_claim_example_json(
    *,
    row_id: str = "example-row-id",
    execution_ref: str = "exec-ref-from-this-turn-catalog",
    include_coverage_map: bool = False,
) -> str:
    """Render the shared complete claim-row example as compact JSON."""
    return json.dumps(
        semantic_risk_claim_example(
            row_id=row_id,
            execution_ref=execution_ref,
            include_coverage_map=include_coverage_map,
        ),
        separators=(", ", ": "),
    )


def semantic_risk_claim_schema_text(
    *,
    include_coverage_map: bool = False,
    preserve_optional_keys: bool = False,
) -> str:
    """Describe the exact claim-row keys for prompt and repair surfaces.

    ``include_coverage_map`` adds the ``test_level`` key to the exact-key list
    and example (#1290).  ``preserve_optional_keys`` keeps the default example
    but describes ``test_level`` as an optional, preserve-only key, for repair
    instructions that must neither add nor strip it.
    """
    keys = SEMANTIC_RISK_CLAIM_KEYS
    if include_coverage_map:
        keys = keys + SEMANTIC_RISK_CLAIM_COVERAGE_KEYS
    text = _semantic_risk_claim_schema_text_base(keys, include_coverage_map)
    if preserve_optional_keys and not include_coverage_map:
        text += (
            " A claim may also carry an optional `test_level` key "
            "(`unit`, `integration`, or `workflow`); it is preserve-only: carry "
            "it exactly as the source has it and never add, change, or drop it."
        )
    return text


def _semantic_risk_claim_schema_text_base(
    keys: tuple[str, ...], include_coverage_map: bool
) -> str:
    return (
        "Each `risk_test_matrix_claims` row uses exactly these keys: "
        + ", ".join(f"`{key}`" for key in keys)
        + ". `row_id` and a non-empty `execution_refs` list are mandatory. "
        "`execution_refs` holds only the opaque selectors printed by "
        "`agent-loop run-tests` in this turn, never command strings or test "
        "identifiers; tests run outside the wrapper cannot verify a row (list "
        "them in `tests_run` only), and with no selector you omit the claim, so "
        "the row stays unverified, rather than inventing one. "
        "One selector may appear in several rows when that run executed each "
        "row's tests; list a selector at most once within a row. "
        f"`execution_refs` holds at most {SEMANTIC_RISK_CLAIMS_MAX_EXECUTION_REFS} "
        "selectors; cite only the runs that executed this row's tests, not every "
        f"run. A list over {SEMANTIC_RISK_CLAIMS_MAX_EXECUTION_REFS} keeps its "
        f"first {SEMANTIC_RISK_CLAIMS_MAX_EXECUTION_REFS} distinct selectors and "
        "records a caveat naming the rest, which cannot verify the row; a "
        "repeated selector in a list of "
        f"{SEMANTIC_RISK_CLAIMS_MAX_EXECUTION_REFS} or fewer still drops the claim. "
        "Every fact key ("
        + ", ".join(f"`{key}`" for key in SEMANTIC_RISK_CLAIM_FACT_KEYS)
        + ") must be present and non-empty for the row to verify; a missing, "
        "null, or empty fact leaves the row unverified. Each fact list holds at "
        f"most {RISK_MATRIX_MAX_LIST_ITEMS} items; a longer list keeps its "
        f"first {RISK_MATRIX_MAX_LIST_ITEMS} entries and the row records a "
        "caveat naming the loss, so split broader coverage across additional "
        "approved rows instead of overflowing one list. `caveats` is optional. "
        + (
            "`test_level` is the level the cited tests actually exercise: one of "
            "`unit`, `integration`, or `workflow` (workflow means driving the "
            "real orchestrator paths). "
            if include_coverage_map
            else ""
        )
        + "Complete example row: "
        + semantic_risk_claim_example_json(include_coverage_map=include_coverage_map)
    )


@dataclass(frozen=True)
class SemanticRiskCoverageClaim:
    """Model-authored semantic coverage for one approved matrix row.

    ``execution_refs`` are invocation-local selectors issued by the managed
    test broker.  They are intentionally not receipt IDs and are never
    persisted as evidence authority.

    ``truncated_fact_fields`` names fact lists that overflowed the
    ``RISK_MATRIX_MAX_LIST_ITEMS`` bound and were truncated before the PR was
    authenticated (#913).  Truncation is accepted content loss, so derivation
    treats the row as a claim defect rather than promoting it to ``verified``.

    ``dropped_execution_refs`` records model-supplied refs that the
    catalog-aware parser could not resolve to a current-turn handle (#859).
    They are never selectable or citable; derivation turns each one into an
    ``unknown-execution-ref`` diagnostic, and ``to_payload`` exposes them only
    through the bounded caveat the parser added.
    """

    row_id: str
    execution_refs: tuple[str, ...]
    test_identifiers: tuple[str, ...]
    test_locations: tuple[str, ...]
    workflow_path_claim: str
    outcome_assertions: tuple[str, ...]
    forbidden_effect_assertions: tuple[str, ...]
    caveats: tuple[str, ...] = ()
    dropped_execution_refs: tuple[str, ...] = ()
    # Selectors removed by over-bound normalisation (#1300); like
    # ``dropped_execution_refs`` they are disclosed only through the caveat.
    overflow_execution_refs: tuple[str, ...] = ()
    truncated_fact_fields: tuple[str, ...] = ()
    # Declared level the cited tests exercise (#1290); None when absent or invalid.
    test_level: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "row_id": sanitize_historical_text(self.row_id),  # shape-check: fatal:authentication-or-forgery
            "execution_refs": [sanitize_historical_text(item) for item in self.execution_refs],  # shape-check: fatal:authentication-or-forgery
            "test_identifiers": [sanitize_historical_text(item) for item in self.test_identifiers],  # shape-check: fatal:authentication-or-forgery
            "test_locations": [sanitize_historical_text(item) for item in self.test_locations],  # shape-check: fatal:authentication-or-forgery
            "workflow_path_claim": sanitize_historical_text(self.workflow_path_claim),  # shape-check: fatal:authentication-or-forgery
            "outcome_assertions": [sanitize_historical_text(item) for item in self.outcome_assertions],  # shape-check: fatal:authentication-or-forgery
            "forbidden_effect_assertions": [sanitize_historical_text(item) for item in self.forbidden_effect_assertions],  # shape-check: fatal:authentication-or-forgery
            "caveats": [sanitize_historical_text(item) for item in self.caveats],  # shape-check: fatal:authentication-or-forgery
        }
        if self.test_level is not None:
            payload["test_level"] = self.test_level
        return payload


@dataclass(frozen=True)
class SemanticRiskCoverageClaims:
    """Bounded shared semantic-claims carrier for both coder response kinds.

    ``dropped_row_ids`` records claims whose ``row_id`` was not in the turn's
    approved enforceable set (#920).  Those claims are discarded before
    authentication rather than rejecting the whole envelope: a dropped claim
    asserts nothing, so its row stays pending, and derivation reports each
    dropped id as an ``unapproved-row-claim`` diagnostic.

    ``degradations`` holds one parser-derived record per dropped claim of any
    row-ID defect -- unapproved, duplicated, malformed, absent or mistyped
    (#926).  It is an accessor beside the claims, never part of
    ``to_payload``, so an agent cannot author one.  Derivation turns the
    records that ``dropped_row_ids`` does not already cover into
    ``degraded-row-claim`` diagnostics.
    """

    claims: tuple[SemanticRiskCoverageClaim, ...] = ()
    dropped_row_ids: tuple[str, ...] = ()
    degradations: tuple[ParseDegradation, ...] = ()
    # Exact row ID of each unapproved-claim record, paired with the record
    # itself, because the record's observed preview is bounded.
    unapproved_claim_row_ids: tuple[tuple[ParseDegradation, str], ...] = ()

    @property
    def rows(self) -> tuple[SemanticRiskCoverageClaim, ...]:
        """Compatibility/readability alias for callers treating claims as rows."""
        return self.claims

    def to_payload(self) -> list[dict[str, object]]:
        return [claim.to_payload() for claim in self.claims]  # shape-check: fatal:authentication-or-forgery


@dataclass(frozen=True)
class SemanticRiskCoverageGap:
    """One coder-declared, evidence-backed untestable-as-specified row (#1290)."""

    row_id: str
    reason: str
    proposed_correction: str

    def to_payload(self) -> dict[str, object]:
        return {
            "row_id": sanitize_historical_text(self.row_id),  # shape-check: fatal:authentication-or-forgery
            "reason": sanitize_historical_text(self.reason),  # shape-check: fatal:authentication-or-forgery
            "proposed_correction": sanitize_historical_text(self.proposed_correction),  # shape-check: fatal:authentication-or-forgery
        }


@dataclass(frozen=True)
class SemanticRiskCoverageGaps:
    """Typed carrier for ``risk_test_matrix_coverage_gaps``.

    Never persisted as authority.  ``degradations`` records dropped entries and
    is an accessor beside the gaps, never part of ``to_payload``.
    """

    gaps: tuple[SemanticRiskCoverageGap, ...] = ()
    degradations: tuple[ParseDegradation, ...] = ()

    def to_payload(self) -> list[dict[str, object]]:
        return [gap.to_payload() for gap in self.gaps]  # shape-check: fatal:authentication-or-forgery


# The matrix is deliberately a structured, generation-gated contract.  It is
# not a reviewer finding ledger: row IDs live in their own namespace and are
# never accepted as carried review-item IDs.
RISK_TEST_MATRIX_CONTRACT_VERSION = 1
RISK_MATRIX_MAX_ROWS = 24
# A complete-scope audit entry must name the union of prior and revised row
# IDs. Each side is independently bounded by RISK_MATRIX_MAX_ROWS, so a full
# replacement revision can legitimately need twice that many IDs.
RISK_MATRIX_MAX_AUDIT_ROW_IDS = 2 * RISK_MATRIX_MAX_ROWS
RISK_MATRIX_MAX_EXCLUSIONS = 16
RISK_MATRIX_MAX_CHANGES = 32
RISK_MATRIX_MAX_LIST_ITEMS = 12
# A row may cite several authoritative receipts. Their attribution and
# execution caveats are merged into the row and need more room than ordinary
# bounded agent-authored lists, without making any list unbounded.
RISK_MATRIX_MAX_CAVEATS = 48
RISK_MATRIX_MAX_FIELD_BYTES = 1_024
RISK_MATRIX_MAX_PAYLOAD_BYTES = 96_000
RISK_MATRIX_APPLICABILITY = frozenset({"applicable", "not-applicable"})
RISK_MATRIX_CHANGE_OPERATIONS = frozenset({"add", "change", "retire", "split", "merge"})
# These ordered tuples are the wire contract used by both strict parsing and
# agent-facing schema guidance. Keep the optional rationale separate so prompt
# examples can show the exact accepted shape without weakening validation.
RISK_TEST_MATRIX_REQUIRED_KEYS = (
    "applicability",
    "rows",
    "important_exclusions",
)
RISK_TEST_MATRIX_OPTIONAL_KEYS = ("not_applicable_rationale",)
RISK_TEST_MATRIX_ROW_KEYS = (
    "row_id",
    "label",
    "entry_path_or_mode",
    "initial_state",
    "event",
    "expected_outcome",
    "forbidden_side_effects",
    "proposed_test_level",
    "proposed_test_location",
    "applicability",
    "related_scope_item_ids",
    "execution_owner",
)
RISK_TEST_MATRIX_CHANGE_KEYS = ("operation", "row_ids", "rationale")
RISK_TEST_MATRIX_CHANGE_CONTAINER_KEYS = ("changes",)
RISK_MATRIX_OWNER_NAMES = frozenset({"one-shot", "retained-parent", "final-integration"})
RISK_MATRIX_EVIDENCE_STATUSES = frozenset(
    {
        "verified", "missing", "not-run", "blocked", "failed", "timed-out",
        "stale/unverified", "incomplete",
    }
)
# Row-ID syntax is ASCII and case-sensitive; the reserved namespaces below are
# a separate, case-insensitive rule layered on top of it.
_RISK_ROW_ID_SYNTAX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# The single declaration of the identifier namespaces a matrix row ID may not
# imitate (#1016): reviewer finding IDs and signed-requirement IDs.  Both the
# row-ID validator and the diagnostic-preview neutralizer consume it, so they
# cannot disagree.  Within an identifier-like span, a finding token is reserved
# at the span start or right after ``-``/``_``/``.`` when an optional separator
# and a digit follow; the requirement prefix is reserved only at the span start.
RESERVED_FINDING_NAMESPACE_TOKENS = ("item", "finding", "review", "blocker")
RESERVED_REQUIREMENT_PREFIX = "hr-"
_RESERVED_NAMESPACE_SEPARATORS = "-_."
_RESERVED_FINDING_AT_RE = re.compile(
    r"(?:" + "|".join(RESERVED_FINDING_NAMESPACE_TOKENS) + r")[-_.]?(?=\d)",
    re.IGNORECASE | re.ASCII,
)
_RESERVED_REQUIREMENT_AT_RE = re.compile(re.escape(RESERVED_REQUIREMENT_PREFIX), re.IGNORECASE | re.ASCII)
# Identifier-like spans in free text: maximal runs of the row-ID character class.
_IDENTIFIER_LIKE_SPAN_RE = re.compile(r"[A-Za-z0-9._-]+")


def _reserved_namespace_match_at(span: str, pos: int, *, span_start: int) -> re.Match[str] | None:
    """Return the reserved-namespace match at ``pos`` of ``span``, if any."""
    if pos == span_start:
        match = _RESERVED_REQUIREMENT_AT_RE.match(span, pos)
        if match is not None:
            return match
    elif span[pos - 1] not in _RESERVED_NAMESPACE_SEPARATORS:
        return None
    return _RESERVED_FINDING_AT_RE.match(span, pos)


def _has_reserved_namespace(span: str) -> bool:
    """Whether ``span``, read as one identifier-like span, imitates a reserved ID."""
    return any(
        _reserved_namespace_match_at(span, pos, span_start=0) is not None
        for pos in range(len(span))
    )


def _is_valid_risk_row_id(row_id: str) -> bool:
    return bool(_RISK_ROW_ID_SYNTAX_RE.fullmatch(row_id)) and not _has_reserved_namespace(row_id)


# Stage IDs are approved-plan identifiers, not a second, narrower namespace.
# Topology validation checks membership in the approved recommendation; this
# parser must accept valid IDs such as ``api`` and ``s1`` as well as the older
# ``stage-*``/``child-*`` spellings.
_RISK_OWNER_STAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _risk_bounded_string(value: object, *, context: str, max_bytes: int = RISK_MATRIX_MAX_FIELD_BYTES) -> str:
    rendered = _expect_non_empty_string(value, context=context)  # shape-check: delegated
    if len(rendered.encode("utf-8")) > max_bytes:
        raise AgentLoopError(f"{context} exceeds the {max_bytes}-byte bound.")  # shape-check: delegated
    return rendered


def _risk_bounded_string_list(
    value: object,
    *,
    context: str,
    max_items: int = RISK_MATRIX_MAX_LIST_ITEMS,
    min_items: int = 0,
) -> tuple[str, ...]:
    rendered = _expect_string_list(value, context=context, item_context=context)  # shape-check: delegated
    if len(rendered) < min_items:
        raise AgentLoopError(f"{context} must contain at least {min_items} item(s).")  # shape-check: delegated
    if len(rendered) > max_items:
        raise AgentLoopError(f"{context} exceeds the {max_items}-item bound.")  # shape-check: delegated
    for index, item in enumerate(rendered):
        if len(item.encode("utf-8")) > RISK_MATRIX_MAX_FIELD_BYTES:
            raise AgentLoopError(f"{context}[{index}] exceeds the {RISK_MATRIX_MAX_FIELD_BYTES}-byte bound.")  # shape-check: delegated
    return rendered


@dataclass(frozen=True)
class RiskTestMatrixRow:
    row_id: str
    label: str
    entry_path_or_mode: str
    initial_state: str
    event: str
    expected_outcome: str
    forbidden_side_effects: tuple[str, ...]
    proposed_test_level: str
    proposed_test_location: str
    applicability: str
    related_scope_item_ids: tuple[str, ...]
    execution_owner: str

    def to_payload(self) -> dict[str, object]:
        return {
            "row_id": sanitize_historical_text(self.row_id),  # shape-check: fatal:authentication-or-forgery
            "label": sanitize_historical_text(self.label),  # shape-check: fatal:authentication-or-forgery
            "entry_path_or_mode": sanitize_historical_text(self.entry_path_or_mode),  # shape-check: fatal:authentication-or-forgery
            "initial_state": sanitize_historical_text(self.initial_state),  # shape-check: fatal:authentication-or-forgery
            "event": sanitize_historical_text(self.event),  # shape-check: fatal:authentication-or-forgery
            "expected_outcome": sanitize_historical_text(self.expected_outcome),  # shape-check: fatal:authentication-or-forgery
            "forbidden_side_effects": [sanitize_historical_text(item) for item in self.forbidden_side_effects],  # shape-check: fatal:authentication-or-forgery
            "proposed_test_level": sanitize_historical_text(self.proposed_test_level),  # shape-check: fatal:authentication-or-forgery
            "proposed_test_location": sanitize_historical_text(self.proposed_test_location),  # shape-check: fatal:authentication-or-forgery
            "applicability": sanitize_historical_text(self.applicability),  # shape-check: fatal:authentication-or-forgery
            "related_scope_item_ids": [sanitize_historical_text(item) for item in self.related_scope_item_ids],  # shape-check: fatal:authentication-or-forgery
            "execution_owner": sanitize_historical_text(self.execution_owner),  # shape-check: fatal:authentication-or-forgery
        }


@dataclass(frozen=True)
class RiskTestMatrix:
    applicability: str
    rows: tuple[RiskTestMatrixRow, ...] = ()
    important_exclusions: tuple[str, ...] = ()
    not_applicable_rationale: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "applicability": self.applicability,
            "rows": [row.to_payload() for row in self.rows],  # shape-check: fatal:authentication-or-forgery
            "important_exclusions": [sanitize_historical_text(item) for item in self.important_exclusions],  # shape-check: fatal:authentication-or-forgery
        }
        if self.not_applicable_rationale is not None:
            payload["not_applicable_rationale"] = sanitize_historical_text(self.not_applicable_rationale)  # shape-check: fatal:authentication-or-forgery
        return payload

    @property
    def is_applicable(self) -> bool:
        return self.applicability == "applicable"


@dataclass(frozen=True)
class RiskTestMatrixChange:
    operation: str
    row_ids: tuple[str, ...]
    rationale: str

    def to_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "row_ids": list(self.row_ids),
            "rationale": sanitize_historical_text(self.rationale),  # shape-check: fatal:authentication-or-forgery
        }


def _validate_risk_row_id(value: object, *, context: str) -> str:
    row_id = _risk_bounded_string(value, context=context, max_bytes=128)  # shape-check: fatal:no-conservative-reading
    if not _is_valid_risk_row_id(row_id):
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            f"{context} must be a matrix-specific identifier and may not resemble a reviewer finding ID."
        )
    return row_id


def _validate_risk_owner(value: object, *, context: str) -> str:
    owner = _risk_bounded_string(value, context=context, max_bytes=128)  # shape-check: fatal:no-conservative-reading
    if owner not in RISK_MATRIX_OWNER_NAMES and not _RISK_OWNER_STAGE_RE.fullmatch(owner):
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            f"{context} must be one of one-shot, retained-parent, final-integration, or a reviewed stage ID."
        )
    return owner


def _parse_risk_test_matrix(value: object, *, context: str = "risk_test_matrix") -> RiskTestMatrix:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload,
        context=context,
        required=set(RISK_TEST_MATRIX_REQUIRED_KEYS),
        optional=set(RISK_TEST_MATRIX_OPTIONAL_KEYS),
    )
    applicability = _risk_bounded_string(payload["applicability"], context=f"{context}.applicability", max_bytes=64)  # shape-check: fatal:no-conservative-reading
    if applicability not in RISK_MATRIX_APPLICABILITY:
        raise AgentLoopError(f"{context}.applicability must be `applicable` or `not-applicable`.")  # shape-check: fatal:no-conservative-reading
    rows_payload = payload["rows"]
    if not isinstance(rows_payload, list):
        raise AgentLoopError(f"{context}.rows must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    if len(rows_payload) > RISK_MATRIX_MAX_ROWS:
        raise AgentLoopError(f"{context}.rows exceeds the {RISK_MATRIX_MAX_ROWS}-row bound; consolidate scenarios explicitly.")  # shape-check: fatal:payload-bound
    rows: list[RiskTestMatrixRow] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(rows_payload):
        row_context = f"{context}.rows[{index}]"
        row = _expect_object(raw_row, context=row_context)  # shape-check: fatal:no-conservative-reading
        _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
            row,
            context=row_context,
            required=set(RISK_TEST_MATRIX_ROW_KEYS),
        )
        row_id = _validate_risk_row_id(row["row_id"], context=f"{row_context}.row_id")  # shape-check: fatal:no-conservative-reading
        if row_id in seen:
            raise AgentLoopError(f"{context} contains duplicate row ID `{row_id}`.")  # shape-check: fatal:no-conservative-reading
        seen.add(row_id)
        row_applicability = _risk_bounded_string(row["applicability"], context=f"{row_context}.applicability", max_bytes=64)  # shape-check: fatal:no-conservative-reading
        if row_applicability not in {"applicable", "required", "not-applicable"}:
            raise AgentLoopError(f"{row_context}.applicability is invalid.")  # shape-check: fatal:no-conservative-reading
        rows.append(
            RiskTestMatrixRow(
                row_id=row_id,
                label=_risk_bounded_string(row["label"], context=f"{row_context}.label"),  # shape-check: fatal:no-conservative-reading
                entry_path_or_mode=_risk_bounded_string(row["entry_path_or_mode"], context=f"{row_context}.entry_path_or_mode"),  # shape-check: fatal:no-conservative-reading
                initial_state=_risk_bounded_string(row["initial_state"], context=f"{row_context}.initial_state"),  # shape-check: fatal:no-conservative-reading
                event=_risk_bounded_string(row["event"], context=f"{row_context}.event"),  # shape-check: fatal:no-conservative-reading
                expected_outcome=_risk_bounded_string(row["expected_outcome"], context=f"{row_context}.expected_outcome"),  # shape-check: fatal:no-conservative-reading
                forbidden_side_effects=_risk_bounded_string_list(row["forbidden_side_effects"], context=f"{row_context}.forbidden_side_effects"),  # shape-check: fatal:no-conservative-reading
                proposed_test_level=_risk_bounded_string(row["proposed_test_level"], context=f"{row_context}.proposed_test_level"),  # shape-check: fatal:no-conservative-reading
                proposed_test_location=_risk_bounded_string(row["proposed_test_location"], context=f"{row_context}.proposed_test_location"),  # shape-check: fatal:no-conservative-reading
                applicability=row_applicability,
                related_scope_item_ids=_risk_bounded_string_list(row["related_scope_item_ids"], context=f"{row_context}.related_scope_item_ids"),  # shape-check: fatal:no-conservative-reading
                execution_owner=_validate_risk_owner(row["execution_owner"], context=f"{row_context}.execution_owner"),  # shape-check: fatal:no-conservative-reading
            )
        )
    exclusions = _risk_bounded_string_list(  # shape-check: fatal:no-conservative-reading
        payload["important_exclusions"], context=f"{context}.important_exclusions", max_items=RISK_MATRIX_MAX_EXCLUSIONS
    )
    rationale_value = payload.get("not_applicable_rationale")
    rationale = None if rationale_value is None else _risk_bounded_string(  # shape-check: fatal:no-conservative-reading
        rationale_value, context=f"{context}.not_applicable_rationale", max_bytes=2_048
    )
    if applicability == "not-applicable":
        if rows:
            raise AgentLoopError(f"{context} not-applicable matrices must contain no rows.")  # shape-check: fatal:no-conservative-reading
        if not rationale:
            raise AgentLoopError(f"{context} not-applicable matrices require a non-empty rationale.")  # shape-check: fatal:no-conservative-reading
    elif not rows:
        raise AgentLoopError(f"{context} applicable matrices require at least one row.")  # shape-check: fatal:no-conservative-reading
    elif not any(row.applicability in {"applicable", "required"} for row in rows):
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            f"{context} applicable matrices require at least one applicable or required row; "
            "use a not-applicable matrix with a proportionate rationale when no scenario is enforceable."
        )
    return RiskTestMatrix(
        applicability=applicability,
        rows=tuple(rows),
        important_exclusions=exclusions,
        not_applicable_rationale=rationale,
    )


def parse_risk_test_matrix(value: object, *, context: str = "risk_test_matrix") -> RiskTestMatrix:
    """Validate the bounded generation-1 matrix payload."""
    if isinstance(value, RiskTestMatrix):
        matrix = value
        encoded = json.dumps(matrix.to_payload(), separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")  # shape-check: fatal:authentication-or-forgery
        if len(encoded) > RISK_MATRIX_MAX_PAYLOAD_BYTES:
            raise AgentLoopError(f"{context} exceeds the {RISK_MATRIX_MAX_PAYLOAD_BYTES}-byte payload bound.")  # shape-check: fatal:payload-bound
        return matrix
    matrix = _parse_risk_test_matrix(value, context=context)  # shape-check: fatal:no-conservative-reading
    encoded = json.dumps(matrix.to_payload(), separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")  # shape-check: fatal:authentication-or-forgery
    if len(encoded) > RISK_MATRIX_MAX_PAYLOAD_BYTES:
        raise AgentLoopError(f"{context} exceeds the {RISK_MATRIX_MAX_PAYLOAD_BYTES}-byte payload bound.")  # shape-check: fatal:payload-bound
    return matrix


def _parse_risk_test_matrix_changes(value: object, *, context: str = "risk_test_matrix_changes") -> tuple[RiskTestMatrixChange, ...]:
    if isinstance(value, dict):
        _expect_exact_keys(value, context=context, required=set(RISK_TEST_MATRIX_CHANGE_CONTAINER_KEYS))  # shape-check: fatal:no-conservative-reading
        value = value["changes"]
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array or an object containing `changes`.")  # shape-check: fatal:no-conservative-reading
    if len(value) > RISK_MATRIX_MAX_CHANGES:
        raise AgentLoopError(f"{context} exceeds the {RISK_MATRIX_MAX_CHANGES}-change bound.")  # shape-check: fatal:payload-bound
    changes: list[RiskTestMatrixChange] = []
    for index, raw_change in enumerate(value):
        change_context = f"{context}[{index}]"
        change = _expect_object(raw_change, context=change_context)  # shape-check: fatal:no-conservative-reading
        _expect_exact_keys(change, context=change_context, required=set(RISK_TEST_MATRIX_CHANGE_KEYS))  # shape-check: fatal:no-conservative-reading
        operation = _risk_bounded_string(change["operation"], context=f"{change_context}.operation", max_bytes=32)  # shape-check: fatal:no-conservative-reading
        if operation not in RISK_MATRIX_CHANGE_OPERATIONS:
            raise AgentLoopError(f"{change_context}.operation is invalid.")  # shape-check: fatal:no-conservative-reading
        row_ids = tuple(_validate_risk_row_id(item, context=f"{change_context}.row_ids[{i}]") for i, item in enumerate(  # shape-check: fatal:no-conservative-reading
            _risk_bounded_string_list(  # shape-check: fatal:no-conservative-reading
                change["row_ids"],
                context=f"{change_context}.row_ids",
                max_items=RISK_MATRIX_MAX_AUDIT_ROW_IDS,
            )
        ))
        if not row_ids:
            raise AgentLoopError(f"{change_context}.row_ids must not be empty.")  # shape-check: fatal:no-conservative-reading
        changes.append(RiskTestMatrixChange(operation=operation, row_ids=row_ids, rationale=_risk_bounded_string(change["rationale"], context=f"{change_context}.rationale", max_bytes=2_048)))  # shape-check: fatal:no-conservative-reading
    return tuple(changes)


def parse_risk_test_matrix_changes(value: object, *, context: str = "risk_test_matrix_changes") -> tuple[RiskTestMatrixChange, ...]:
    return _parse_risk_test_matrix_changes(value, context=context)  # shape-check: fatal:no-conservative-reading


# Explicit, lossless table of risk-test-matrix list-of-strings fields (same
# spirit as PLAN_REVIEW_FINDING_KEY_ALIASES, #1169). A lone non-empty string
# has exactly one correct reading: a one-element list. Strict parsers stay
# strict because they also validate stored and authenticated matrices.
RISK_TEST_MATRIX_ROW_STRING_LIST_KEYS = ("forbidden_side_effects", "related_scope_item_ids")
RISK_TEST_MATRIX_STRING_LIST_KEYS = ("important_exclusions",)
RISK_TEST_MATRIX_CHANGE_STRING_LIST_KEYS = ("row_ids",)


def _wrap_lone_string(container: dict, key: str, path: str, changed: list[str]) -> None:
    value = container.get(key)
    if isinstance(value, str) and value.strip():
        container[key] = [value]
        changed.append(path)


def normalize_risk_test_matrix_string_lists(payload: dict) -> tuple[dict, tuple[str, ...]]:
    """Return a copy of a plan payload with lone matrix strings wrapped in lists."""
    result = copy.deepcopy(payload)
    changed: list[str] = []
    matrix = result.get("risk_test_matrix")
    if isinstance(matrix, dict):
        rows = matrix.get("rows")
        if isinstance(rows, list):
            for index, row in enumerate(rows):
                if isinstance(row, dict):
                    for key in RISK_TEST_MATRIX_ROW_STRING_LIST_KEYS:
                        _wrap_lone_string(row, key, f"risk_test_matrix.rows[{index}].{key}", changed)
        for key in RISK_TEST_MATRIX_STRING_LIST_KEYS:
            _wrap_lone_string(matrix, key, f"risk_test_matrix.{key}", changed)
    changes = result.get("risk_test_matrix_changes")
    if isinstance(changes, dict):
        changes = changes.get("changes")
    if isinstance(changes, list):
        for index, change in enumerate(changes):
            if isinstance(change, dict):
                for key in RISK_TEST_MATRIX_CHANGE_STRING_LIST_KEYS:
                    _wrap_lone_string(change, key, f"risk_test_matrix_changes[{index}].{key}", changed)
    return result, tuple(changed)


def risk_test_matrix_prompt_examples() -> dict[str, object]:
    """Return minimal fresh-matrix payloads accepted by the strict parser.

    Prompt layers use these fresh objects instead of copying a second schema
    literal. Tests parse the returned examples so a prompt contract change
    cannot silently diverge from the validator.
    """
    return {
        "applicable": {
            "applicability": "applicable",
            "rows": [{
                "row_id": "fresh-row",
                "label": "Fresh workflow row",
                "entry_path_or_mode": "issue plan-first / fresh plan_state",
                "initial_state": "No canonical plan exists.",
                "event": "The planner emits the applicable scenario.",
                "expected_outcome": "Strict validation accepts the response.",
                "forbidden_side_effects": ["Do not omit the scenario."],
                "proposed_test_level": "protocol unit",
                "proposed_test_location": "tests/test_protocol.py",
                "applicability": "required",
                "related_scope_item_ids": ["scope-1"],
                "execution_owner": "one-shot",
            }],
            "important_exclusions": ["Planned tests are not evidence."],
        },
        "not_applicable": {
            "applicability": "not-applicable",
            "rows": [],
            "important_exclusions": [],
            "not_applicable_rationale": "This narrow change has no meaningful transition surface.",
        },
        "changes": [{
            "operation": "change",
            "row_ids": ["fresh-row"],
            "rationale": "The current revision changes the scenario semantics.",
        }],
    }


def risk_test_matrix_identity(
    matrix: RiskTestMatrix | Mapping[str, object],
    changes: Sequence[RiskTestMatrixChange | Mapping[str, object]] = (),
    *,
    contract_version: int = RISK_TEST_MATRIX_CONTRACT_VERSION,
) -> str:
    parsed_matrix = matrix if isinstance(matrix, RiskTestMatrix) else parse_risk_test_matrix(matrix)  # shape-check: fatal:authentication-or-forgery
    parsed_changes = tuple(
        item if isinstance(item, RiskTestMatrixChange) else _parse_risk_test_matrix_changes([item])[0]  # shape-check: fatal:authentication-or-forgery
        for item in changes
    )
    payload = {
        "contract_version": contract_version,
        "matrix": parsed_matrix.to_payload(),  # shape-check: fatal:authentication-or-forgery
        "changes": [item.to_payload() for item in parsed_changes],  # shape-check: fatal:authentication-or-forgery
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def sanitize_risk_test_matrix(value: RiskTestMatrix | Mapping[str, object] | None) -> dict[str, object] | None:
    if value is None:
        return None
    parsed = value if isinstance(value, RiskTestMatrix) else parse_risk_test_matrix(value)  # shape-check: fatal:orchestrator-authored
    return parsed.to_payload()  # shape-check: fatal:authentication-or-forgery


def validate_risk_test_matrix_revision(
    previous: RiskTestMatrix | Mapping[str, object],
    current: RiskTestMatrix | Mapping[str, object],
    changes: Sequence[RiskTestMatrixChange | Mapping[str, object]],
    *,
    approved: bool = False,
    historical_changes: Sequence[RiskTestMatrixChange | Mapping[str, object]] = (),
) -> tuple[RiskTestMatrixChange, ...]:
    """Require an explicit audit operation for every semantic draft change.

    Revisions are commonly rendered from the previous canonical plan, which
    can cause an agent to repeat an already recorded audit entry. Exact
    entries supplied through ``historical_changes`` are ignored for the
    current diff, but any new semantic change still needs a fresh operation
    and any unrelated operation is still rejected.
    """
    old = previous if isinstance(previous, RiskTestMatrix) else parse_risk_test_matrix(previous, context="previous risk_test_matrix")  # shape-check: fatal:no-conservative-reading
    new = current if isinstance(current, RiskTestMatrix) else parse_risk_test_matrix(current, context="current risk_test_matrix")  # shape-check: fatal:no-conservative-reading
    parsed_changes = tuple(
        item if isinstance(item, RiskTestMatrixChange) else _parse_risk_test_matrix_changes([item])[0]  # shape-check: fatal:no-conservative-reading
        for item in changes
    )
    parsed_historical_changes = [
        item if isinstance(item, RiskTestMatrixChange) else _parse_risk_test_matrix_changes([item])[0]  # shape-check: fatal:no-conservative-reading
        for item in historical_changes
    ]
    current_changes = list(parsed_changes)
    for historical_change in parsed_historical_changes:
        try:
            current_changes.remove(historical_change)
        except ValueError:
            continue
    semantic_changed = old.to_payload() != new.to_payload()  # shape-check: fatal:authentication-or-forgery
    if approved and semantic_changed:
        raise AgentLoopError("Approved risk matrix is immutable; substantive changes require explicit replanning.")  # shape-check: fatal:no-conservative-reading
    # The approval boundary receives the accepted draft plus the audit carried
    # by that draft, rather than a prior approved baseline.  The audit is a
    # historical record of how the draft got here, so its subjects are not
    # required to be a diff against the identical accepted payload.  Draft
    # revisions (the non-approved path) still run the complete coverage and
    # no-extra-subject checks below.
    if approved and not semantic_changed:
        return parsed_changes
    if semantic_changed and not current_changes:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "Risk matrix changes omit audit operations; semantic changes require explicit "
            "review-visible audit operations."
        )
    old_by_id = {row.row_id: row for row in old.rows}
    new_by_id = {row.row_id: row for row in new.rows}
    old_ids = set(old_by_id)
    new_ids = set(new_by_id)
    changed_ids = {
        row_id for row_id in old_ids | new_ids
        if old_by_id.get(row_id) != new_by_id.get(row_id)
    }
    added_ids = new_ids - old_ids
    retired_ids = old_ids - new_ids
    retained_changed_ids = {
        row_id for row_id in old_ids & new_ids
        if old_by_id[row_id] != new_by_id[row_id]
    }
    changed_matrix_fields = {
        field
        for field in ("applicability", "important_exclusions", "not_applicable_rationale")
        if getattr(old, field) != getattr(new, field)
    }
    # A matrix-level operation is a separate audit subject from row
    # transitions. Keep the complete-scope representation and its established
    # change/split/merge operation allowance for compatibility with existing
    # rendered audits. A complete-scope ``change`` also describes retained-row
    # changes when the row set itself is unchanged; otherwise a draft that
    # edits one or more rows while changing a matrix-level field would have no
    # expressible audit operation. Structural row transitions still require
    # their own add/retire/split/merge operations.
    matrix_scope = (old_ids | new_ids) or {"matrix"}
    for change in current_changes:
        if len(set(change.row_ids)) != len(change.row_ids):
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"Risk matrix audit operation `{change.operation}` repeats a row ID; "
                "each row may be audited only once."
            )
    matrix_change_indexes: set[int] = set()
    if changed_matrix_fields:
        matrix_change_indexes = {
            index for index, change in enumerate(current_changes)
            if change.operation in {"change", "split", "merge"}
            and set(change.row_ids) == matrix_scope
        }
        if len(matrix_change_indexes) != 1:
            expected_scope = ", ".join(sorted(matrix_scope))
            changed_fields = ", ".join(sorted(changed_matrix_fields))
            if not matrix_change_indexes:
                match_detail = "found none"
            else:
                match_detail = (
                    f"found {len(matrix_change_indexes)}; remove duplicate complete-scope entries"
                )
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                "Risk matrix-level changes require one review-visible audit operation "
                "with operation `change`, `split`, or `merge` covering the complete matrix scope. "
                f"{match_detail}. Changed matrix fields: {changed_fields}. "
                f"Required row_ids: [{expected_scope}]."
            )

    # Validate each operation against the actual old/new transition. Merely
    # covering the changed IDs is insufficient: an ``add`` must contain only
    # new IDs, ``retire`` only removed IDs, and ``change`` only retained rows
    # whose row payload changed. Split/merge groups use old-side/new-side
    # membership cardinality so the audit records the direction of the
    # transition rather than accepting an arbitrary relabeling of the same ID
    # set. A retained ID is valid for a split/merge only when its row payload
    # actually changed.
    covered: set[str] = set()
    # Compute complete-scope row coverage before validating individual row
    # operations. A matrix-level audit and a precise retained-row audit are
    # separate subjects, so either ordering must be accepted when both are
    # present in the same draft revision.
    matrix_transition_covered: set[str] = set()
    for index in matrix_change_indexes:
        change = current_changes[index]
        if change.operation == "change" and not added_ids and not retired_ids:
            matrix_transition_covered.update(retained_changed_ids)
    seen_subjects: set[str] = set()
    for index, change in enumerate(current_changes):
        if index in matrix_change_indexes:
            continue
        subject = set(change.row_ids)
        # Complete-scope matrix coverage may intentionally overlap a
        # row-specific audit; only two non-matrix operations are duplicate
        # row subjects and should be rejected.
        overlap = sorted(subject & seen_subjects)
        if overlap:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                "Risk matrix audit operations overlap on row IDs: " + ", ".join(overlap)
            )
        unknown = sorted(subject - (old_ids | new_ids))
        if unknown:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                "Risk matrix audit operation names unknown row IDs: " + ", ".join(unknown)
            )

        removed = subject & retired_ids
        added = subject & added_ids
        retained = subject & (old_ids & new_ids)
        changed_retained = retained & retained_changed_ids
        unchanged_retained = retained - retained_changed_ids
        operation = change.operation
        valid = False
        expected = ""
        if operation == "add":
            valid = bool(subject) and subject == added
            expected = "only newly introduced row IDs"
        elif operation == "retire":
            valid = bool(subject) and subject == removed
            expected = "only removed row IDs"
        elif operation == "change":
            valid = bool(subject) and subject == changed_retained
            expected = "only retained row IDs whose row payload changed"
        elif operation == "split":
            old_members = subject & old_ids
            new_members = subject & new_ids
            valid = (
                len(old_members) == 1
                and len(new_members) >= 2
                and retained == changed_retained
            )
            expected = "exactly one prior row and at least two current rows"
        elif operation == "merge":
            old_members = subject & old_ids
            new_members = subject & new_ids
            valid = (
                len(old_members) >= 2
                and len(new_members) == 1
                and retained == changed_retained
            )
            expected = "at least two prior rows and exactly one current row"
        if not valid:
            details = []
            if unchanged_retained:
                details.append("unchanged retained=" + ",".join(sorted(unchanged_retained)))
            if added and operation not in {"add", "split", "merge"}:
                details.append("new=" + ",".join(sorted(added)))
            if removed and operation not in {"retire", "split", "merge"}:
                details.append("removed=" + ",".join(sorted(removed)))
            detail = f" ({'; '.join(details)})" if details else ""
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"Risk matrix audit operation `{operation}` has an invalid transition{detail}; "
                f"it must name {expected}."
            )
        seen_subjects.update(subject)
        covered.update(subject)

    covered.update(matrix_transition_covered)
    missing = sorted(changed_ids - covered)
    if missing:
        raise AgentLoopError("Risk matrix changes omit audit operations for: " + ", ".join(missing))  # shape-check: fatal:no-conservative-reading
    if approved and changed_ids:
        raise AgentLoopError("Approved risk matrix rows cannot be removed, reassigned, or weakened without a newly reviewed baseline.")  # shape-check: fatal:no-conservative-reading
    return parsed_changes


EVIDENCE_SOURCE_TOOL_CAPTURED = "tool-captured"
EVIDENCE_SOURCE_OPERATOR_SIGNED = "operator-signed"
_SHA_HEX_RE = re.compile(r"[^\s`]{1,128}")


@dataclass(frozen=True)
class SignedEvidenceReference:
    """Orchestrator-owned proof that a converted row was answered (#1325).

    Minted only when the owning reviewer(s) clear a row-converted evidence
    obligation at its frozen head after newly surfaced signed human input.
    ``reference_id`` is a SHA-256 over the canonical JSON of every other field,
    so a decoded reference is internally consistent; authorship is checked
    separately by the round-state admission predicate.
    """

    matrix_identity: str
    row_id: str
    head: str
    obligation_identity: str
    obligation_item_id: str
    owners: tuple[str, ...]
    signed_requirement_ids: tuple[str, ...]
    clearing_reviewers: tuple[str, ...]
    cleared_round: int
    reference_id: str

    @staticmethod
    def _digest(fields: Mapping[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    @classmethod
    def build(
        cls,
        *,
        matrix_identity: str,
        row_id: str,
        head: str,
        obligation_identity: str,
        obligation_item_id: str,
        owners: Sequence[str],
        signed_requirement_ids: Sequence[str],
        clearing_reviewers: Sequence[str],
        cleared_round: int,
    ) -> "SignedEvidenceReference":
        fields = {
            "version": 1,
            "matrix_identity": matrix_identity,
            "row_id": row_id,
            "head": head,
            "obligation_identity": obligation_identity,
            "obligation_item_id": obligation_item_id,
            "owners": sorted(set(owners)),
            "signed_requirement_ids": sorted(set(signed_requirement_ids)),
            "clearing_reviewers": sorted(set(clearing_reviewers)),
            "cleared_round": cleared_round,
        }
        return cls(
            matrix_identity=matrix_identity,
            row_id=row_id,
            head=head,
            obligation_identity=obligation_identity,
            obligation_item_id=obligation_item_id,
            owners=tuple(fields["owners"]),  # type: ignore[arg-type]
            signed_requirement_ids=tuple(fields["signed_requirement_ids"]),  # type: ignore[arg-type]
            clearing_reviewers=tuple(fields["clearing_reviewers"]),  # type: ignore[arg-type]
            cleared_round=cleared_round,
            reference_id=cls._digest(fields),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "matrix_identity": self.matrix_identity,
            "row_id": self.row_id,
            "head": self.head,
            "obligation_identity": self.obligation_identity,
            "obligation_item_id": self.obligation_item_id,
            "owners": list(self.owners),
            "signed_requirement_ids": list(self.signed_requirement_ids),
            "clearing_reviewers": list(self.clearing_reviewers),
            "cleared_round": self.cleared_round,
            "reference_id": self.reference_id,
        }

    @classmethod
    def from_payload(cls, value: object) -> "SignedEvidenceReference | None":
        """Strictly decode one reference; anything malformed is ``None``."""
        if not isinstance(value, Mapping) or value.get("version") != 1:
            return None
        expected = {
            "version", "matrix_identity", "row_id", "head", "obligation_identity",
            "obligation_item_id", "owners", "signed_requirement_ids",
            "clearing_reviewers", "cleared_round", "reference_id",
        }
        if set(value) != expected:
            return None
        strings = ("matrix_identity", "row_id", "head", "obligation_identity", "obligation_item_id", "reference_id")
        if any(not isinstance(value[key], str) or not value[key].strip() for key in strings):
            return None
        if not re.fullmatch(r"[0-9a-f]{64}", value["matrix_identity"]):
            return None
        if not _SHA_HEX_RE.fullmatch(value["head"]):
            return None
        lists = ("owners", "signed_requirement_ids", "clearing_reviewers")
        for key in lists:
            entries = value[key]
            if (
                not isinstance(entries, list)
                or any(not isinstance(entry, str) or not entry.strip() for entry in entries)
                or entries != sorted(set(entries))
            ):
                return None
        if not value["signed_requirement_ids"] or not value["clearing_reviewers"]:
            return None
        cleared_round = value["cleared_round"]
        if isinstance(cleared_round, bool) or not isinstance(cleared_round, int) or cleared_round < 1:
            return None
        rebuilt = cls.build(
            matrix_identity=value["matrix_identity"],
            row_id=value["row_id"],
            head=value["head"],
            obligation_identity=value["obligation_identity"],
            obligation_item_id=value["obligation_item_id"],
            owners=value["owners"],
            signed_requirement_ids=value["signed_requirement_ids"],
            clearing_reviewers=value["clearing_reviewers"],
            cleared_round=cleared_round,
        )
        if rebuilt.reference_id != value["reference_id"]:
            return None
        return rebuilt


@dataclass(frozen=True)
class RiskTestMatrixEvidenceRow:
    row_id: str
    status: str
    test_identifiers: tuple[str, ...]
    test_locations: tuple[str, ...]
    workflow_path_claim: str
    outcome_assertions: tuple[str, ...]
    forbidden_effect_assertions: tuple[str, ...]
    evidence_citations: tuple[TestObservationCitation, ...]
    caveats: tuple[str, ...] = ()
    # Provenance of a verified row (#1325).  ``operator-signed`` rows are
    # produced only by :func:`apply_signed_evidence_references` from an
    # orchestrator-owned reference; agents can never emit either field.
    evidence_source: str = EVIDENCE_SOURCE_TOOL_CAPTURED
    signed_evidence_reference: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "row_id": sanitize_historical_text(self.row_id),  # shape-check: fatal:authentication-or-forgery
            "status": sanitize_historical_text(self.status),  # shape-check: fatal:authentication-or-forgery
            "test_identifiers": [sanitize_historical_text(item) for item in self.test_identifiers],  # shape-check: fatal:authentication-or-forgery
            "test_locations": [sanitize_historical_text(item) for item in self.test_locations],  # shape-check: fatal:authentication-or-forgery
            "workflow_path_claim": sanitize_historical_text(self.workflow_path_claim),  # shape-check: fatal:authentication-or-forgery
            "outcome_assertions": [sanitize_historical_text(item) for item in self.outcome_assertions],  # shape-check: fatal:authentication-or-forgery
            "forbidden_effect_assertions": [sanitize_historical_text(item) for item in self.forbidden_effect_assertions],  # shape-check: fatal:authentication-or-forgery
            "evidence_citations": [
                {"command": sanitize_historical_text(item.command),  # shape-check: fatal:authentication-or-forgery
                 "receipt_id": sanitize_historical_text(item.receipt_id),  # shape-check: fatal:authentication-or-forgery
                 "claim": sanitize_historical_text(item.claim)}  # shape-check: fatal:authentication-or-forgery
                for item in self.evidence_citations
            ],
            "caveats": [sanitize_historical_text(item) for item in self.caveats],  # shape-check: fatal:authentication-or-forgery
        }
        # Tool-captured payloads stay byte-identical to the pre-#1325 shape.
        if self.evidence_source == EVIDENCE_SOURCE_OPERATOR_SIGNED:
            payload["evidence_source"] = EVIDENCE_SOURCE_OPERATOR_SIGNED
            payload["signed_evidence_reference"] = self.signed_evidence_reference
        return payload


@dataclass(frozen=True)
class RiskTestMatrixEvidence:
    matrix_identity: str
    rows: tuple[RiskTestMatrixEvidenceRow, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "matrix_identity": sanitize_historical_text(self.matrix_identity),  # shape-check: fatal:authentication-or-forgery
            "rows": [row.to_payload() for row in self.rows],  # shape-check: fatal:authentication-or-forgery
        }


@dataclass(frozen=True)
class PostAuthClaimDiagnostic:
    """Bounded diagnostic emitted after PR/head authentication."""

    row_id: str
    code: str
    message: str

    def to_payload(self) -> dict[str, str]:
        return {
            "row_id": sanitize_historical_text(self.row_id),  # shape-check: fatal:authentication-or-forgery
            "code": sanitize_historical_text(self.code),  # shape-check: fatal:authentication-or-forgery
            "message": sanitize_historical_text(self.message),  # shape-check: fatal:authentication-or-forgery
        }


@dataclass(frozen=True)
class DerivedRiskEvidenceResult:
    """Complete canonical evidence plus non-terminal post-auth diagnostics."""

    evidence: RiskTestMatrixEvidence
    diagnostics: tuple[PostAuthClaimDiagnostic, ...] = ()
    # The exact PR head SHA this derivation bound to (#1290); in-memory only.
    bound_head_sha: str | None = None
    # Per-row structured capture classification (#1325), built in the same
    # loop that emits ``diagnostics`` and never read back from their codes:
    # ``row_id -> {"class", "observation_classes", "disqualifiers"}``.
    row_capture_classifications: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    @property
    def risk_test_matrix_evidence(self) -> RiskTestMatrixEvidence:
        return self.evidence


# Closed classification vocabulary for automatic evidence conversion (#1325).
CAPTURE_LIMITED = "capture-limited"
NOT_CAPTURE_LIMITED = "not-capture-limited"
CAPTURE_LIMITED_OBSERVATION_CLASSES = frozenset(
    {
        # incomplete, interrupted, launch-failed or overlap-rejected outcome
        "outcome-incomplete",
        # passed, but not an authoritative receipt (launch integrity or
        # attribution unverified, agent-reported, superseded)
        "pass-unauthoritative",
        # an authoritative pass without a tool-owned receipt ID
        "pass-missing-receipt",
        # an unknown, cross-turn or dropped execution selector
        "unknown-ref",
        # a fully admissible pass is neutral: it never limits nor disqualifies
        "admissible-pass",
    }
)
CAPTURE_OBSERVATION_CLASSES = CAPTURE_LIMITED_OBSERVATION_CLASSES | frozenset(
    {"failed", "timed-out", "unclassified"}
)
CAPTURE_DISQUALIFIERS = frozenset(
    {
        "verified",
        "missing-claim",
        "incomplete-semantic-claim",
        "truncated-semantic-claim",
        "authoritative-failure",
        "authoritative-timeout",
        "unclassified-observation",
        "head-mismatch",
        "tree-mismatch",
        "checkout-head-mismatch",
        "checkout-tree-unavailable",
        "journal-failure",
        "head-changed-during-correction",
        # a passing receipt whose only launch defect is an unverified suite
        # start (#1659): the coder can capture it as a pytest test, so it is
        # never handed to a human by automatic evidence conversion
        "suite-start-unverified",
    }
)


def disqualify_capture_classification(
    derived: DerivedRiskEvidenceResult, reason: str
) -> DerivedRiskEvidenceResult:
    """Mark every row not capture-limited (e.g. the head raced a correction)."""
    rows = {
        row_id: {
            "class": NOT_CAPTURE_LIMITED,
            "observation_classes": list(entry.get("observation_classes", ())),  # type: ignore[call-overload]
            "disqualifiers": sorted({*entry.get("disqualifiers", ()), reason}),  # type: ignore[misc]
        }
        for row_id, entry in derived.row_capture_classifications.items()
    }
    return dataclasses.replace(derived, row_capture_classifications=rows)


def capture_observation_class(observation: object) -> str:
    """Closed-set class of one selected observation, from its structured fields."""
    semantics, _rich = _observation_semantics(observation)
    outcome = semantics["outcome"]
    if outcome in {"incomplete", "interrupted", "launch-failed", "overlap-rejected"}:
        return "outcome-incomplete"
    if outcome == "failed":
        return "failed"
    if outcome == "timed_out":
        return "timed-out"
    if outcome == "passed":
        if not _authoritative_receipt_passes(observation, claim="current-result"):
            return "pass-unauthoritative"
        receipt_id = _observation_value(observation, "receipt_id")
        if not isinstance(receipt_id, str) or not receipt_id.strip():
            return "pass-missing-receipt"
        return "admissible-pass"
    return "unclassified"


def _journal_disqualifies(observations: Sequence[object]) -> bool:
    """An unsuperseded, parent-observed authoritative failure or timeout.

    Narrower than derive's ``unsuperseded_failures``: incomplete, interrupted
    and launch-failed journal entries are capture limitations, not failures.
    """
    return any(
        _observation_value(observation, "provenance") == "parent-observed"
        and _observation_value(observation, "outcome") in {"failed", "timed_out"}
        and not _observation_value(observation, "superseded_by")
        and not _is_non_evidence_launch_failure(observation)
        for observation in observations
    )


def _row_capture_classification(
    *,
    status: str,
    observation_classes: Sequence[str],
    disqualifiers: set[str],
) -> dict[str, object]:
    disqualifiers = set(disqualifiers)
    if status == "verified":
        disqualifiers.add("verified")
    for observation_class in observation_classes:
        if observation_class == "failed":
            disqualifiers.add("authoritative-failure")
        elif observation_class == "timed-out":
            disqualifiers.add("authoritative-timeout")
        elif observation_class not in CAPTURE_LIMITED_OBSERVATION_CLASSES:
            disqualifiers.add("unclassified-observation")
    return {
        "class": NOT_CAPTURE_LIMITED if disqualifiers else CAPTURE_LIMITED,
        "observation_classes": sorted(set(observation_classes)),
        "disqualifiers": sorted(disqualifiers),
    }


def _is_non_evidence_launch_failure(observation: object) -> bool:
    from .local_test_evidence import is_non_evidence_launch_failure

    return is_non_evidence_launch_failure(observation)


def _observation_value(observation: object, name: str, default: object = None) -> object:
    if isinstance(observation, Mapping):
        return observation.get(name, default)
    return getattr(observation, name, default)


def _observation_attribution(observation: object) -> Mapping[str, object]:
    value = _observation_value(observation, "attribution")
    if value is None:
        value = _observation_value(observation, "tree")
    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    return {
        name: getattr(value, name, None)
        for name in ("state", "head", "tracked_digest", "stable", "untracked_input", "caveats")
    }


def _observation_command_text(observation: object) -> str:
    command = _observation_value(observation, "command")
    if isinstance(command, (tuple, list)) and all(isinstance(item, str) for item in command):
        return shlex.join(command)
    normalized = _observation_value(observation, "normalized_command")
    if isinstance(normalized, str) and normalized.strip():
        return normalized.strip()
    return str(command or "local test execution")


UNAPPROVED_ROW_CLAIM_DIAGNOSTIC = "unapproved-row-claim"
DEGRADED_ROW_CLAIM_DIAGNOSTIC = "degraded-row-claim"

# Rules recorded on claim-scope row-ID degradations (#926).  The unapproved
# rule keeps its historical ``unapproved-row-claim`` diagnostic, derived from
# ``dropped_row_ids``; every other rule becomes ``degraded-row-claim``.
CLAIM_ROW_ID_UNAPPROVED_RULE = "row_id is outside the approved enforceable set"
CLAIM_ROW_ID_DUPLICATE_RULE = "row_id is claimed more than once; every copy is dropped"
CLAIM_ROW_ID_MALFORMED_RULE = "row_id is not a valid matrix-specific identifier"
CLAIM_ROW_ID_ABSENT_RULE = "row_id key is absent"
CLAIM_ROW_ID_MISTYPED_RULE = "row_id is not a string"
CLAIM_DROPPED_OUTCOME = "claim-dropped"
COVERAGE_GAP_DROPPED_OUTCOME = "gap-dropped"
# Claim-scope rules converted from envelope rejections by the stage-3 audit
# (#927).  Each drops the one claim, or every claim for a non-array field.
CLAIM_NOT_OBJECT_RULE = "claim is not a JSON object"
CLAIMS_FIELD_NOT_ARRAY_RULE = "risk_test_matrix_claims is not a JSON array; no claim is kept"
CLAIM_EXECUTION_REFS_ABSENT_RULE = "execution_refs key is absent"
CLAIM_EXECUTION_REFS_EMPTY_RULE = "execution_refs contains no selector"
CLAIM_EXECUTION_REFS_MISTYPED_RULE = "execution_refs is not an array of non-blank strings"
CLAIM_EXECUTION_REFS_NO_CATALOG_OVERSIZE_RULE = "selector exceeds the field bound without a catalog"
CLAIM_EXECUTION_REFS_REPEATED_RULE = "execution_refs selects the same selector more than once"
CLAIM_UNKNOWN_KEYS_RULE = "claim carries unknown keys"
CLAIM_FACT_MISTYPED_RULE = "semantic fact is ill-typed, blank or duplicated"
CLAIM_FACT_OVERSIZE_RULE = "semantic fact exceeds the field bound"
# Unknown claim keys that name orchestrator-owned verification authority.
# ``status`` and ``evidence_citations`` are canonical evidence-row keys that
# assert an outcome and receipt-backed support; ``receipt_id``, ``command``
# and ``claim`` are receipt-citation keys that assert a broker receipt.
# ``evidence_source`` and ``signed_evidence_reference`` assert operator-signed
# verification, which only the orchestrator may grant (#1325).  A
# model-authored claim must never carry one, so these stay fatal (#891
# forged-record family).  Matching is exact: every other unknown key degrades.
CLAIM_RESERVED_AUTHORITY_KEYS = frozenset(
    {
        "status", "evidence_citations", "receipt_id", "command", "claim",
        "evidence_source", "signed_evidence_reference",
    }
)


def _neutralize_identifier_like_span(span: str) -> str:
    """Break every reserved-namespace match in one identifier-like span.

    One left-to-right scan: each match is replaced by its token (trailing
    separator stripped) plus a middle dot, and matching restarts right after
    it as a fresh span start, so a chained ``hr-hr-2`` becomes ``hr\u00b7hr\u00b72``
    in one pass and no span of the output is still reserved (#1016).
    """
    pieces: list[str] = []
    emitted = span_start = pos = 0
    while pos < len(span):
        match = _reserved_namespace_match_at(span, pos, span_start=span_start)
        if match is None:
            pos += 1
            continue
        pieces.append(span[emitted:pos])
        pieces.append(match.group(0).rstrip(_RESERVED_NAMESPACE_SEPARATORS) + "\u00b7")
        emitted = span_start = pos = match.end()
    pieces.append(span[emitted:])
    return "".join(pieces)


def _neutralize_identifier_like(text: str) -> str:
    """Break finding- and requirement-like tokens so a preview cannot read as one (#926).

    Applies the single reserved-namespace declaration that row-ID validation
    refuses (``RESERVED_FINDING_NAMESPACE_TOKENS``/``RESERVED_REQUIREMENT_PREFIX``)
    to each identifier-like span of free text (#1016).
    """
    return _IDENTIFIER_LIKE_SPAN_RE.sub(lambda match: _neutralize_identifier_like_span(match.group(0)), text)


def _degraded_row_claim_message(record: ParseDegradation) -> str:
    observed = _neutralize_identifier_like(_dropped_ref_preview(record.observed_preview))  # shape-check: fatal:authentication-or-forgery
    return (
        f"The semantic coverage claim at `{_dropped_ref_preview(record.element_path)}` was dropped "  # shape-check: fatal:authentication-or-forgery
        f"before authentication: {_dropped_ref_preview(record.rule)} "  # shape-check: fatal:authentication-or-forgery
        f"(observed `{observed}`). "
        "The claim asserted no coverage."
    )


def _unapproved_row_claim_message(
    row_id: str,
    *,
    row_owner: str | None,
    execution_owner: str | None,
    record: ParseDegradation | None = None,
) -> str:
    """Name the dropped row, the scope it violated, and its real owner."""
    scope = (
        f"execution owner `{_dropped_ref_preview(execution_owner)}`"  # shape-check: fatal:authentication-or-forgery
        if execution_owner
        else "this turn"
    )
    if row_owner is None:
        origin = "The row is not in the approved matrix."
    else:
        origin = f"The row belongs to execution owner `{_dropped_ref_preview(row_owner)}`."  # shape-check: fatal:authentication-or-forgery
    return (
        f"A semantic coverage claim for row "
        f"`{_neutralize_identifier_like(_dropped_ref_preview(row_id))}` was dropped "  # shape-check: fatal:authentication-or-forgery
        f"because the row is not in the approved enforceable matrix set for {scope}. "
        f"{origin} The claim asserted no coverage."
        + (
            f" Claim `{_dropped_ref_preview(record.element_path)}`: "  # shape-check: fatal:authentication-or-forgery
            f"{_dropped_ref_preview(record.rule)} (outcome `{_dropped_ref_preview(record.outcome)}`)."  # shape-check: fatal:authentication-or-forgery
            if record is not None
            else ""
        )
    )


def _claims_value(
    claims: SemanticRiskCoverageClaims | Sequence[SemanticRiskCoverageClaim] | None,
) -> tuple[SemanticRiskCoverageClaim, ...]:
    if claims is None:
        return ()
    if isinstance(claims, SemanticRiskCoverageClaims):
        return claims.claims
    return tuple(claims)


def derive_risk_test_matrix_evidence(
    *,
    matrix: RiskTestMatrix | Mapping[str, object],
    claims: SemanticRiskCoverageClaims | Sequence[SemanticRiskCoverageClaim] | None,
    observations: Sequence[object],
    execution_catalog: Sequence[object] | None = None,
    invocation_id: str | None = None,
    current_head: str | None = None,
    current_tree_digest: str | None = None,
    authenticated_checkout_head: str | None = None,
    authenticated_tree_clean: bool | None = None,
    expected_identity: str | None = None,
    execution_owner: str | None = None,
) -> DerivedRiskEvidenceResult:
    """Derive canonical evidence from trusted matrix/journal/head inputs.

    Model claims can select only live execution handles.  Receipt IDs,
    canonical rows, statuses, ordering, and mappings are all produced here.
    Any post-authentication mismatch produces complete non-verified evidence
    and a bounded diagnostic, preserving the authenticated handoff.

    Admissibility is the checkout-head, clean-tree, tree-digest,
    turn-catalog, and receipt-head-attribution proofs only.  It is
    independent of whether the follow-up moved the PR head (#1627, which
    supersedes the #1338 predecessor clause): unpushed commits and dirty
    trees are still caught by those proofs, and whether evidence alone
    answers a finding is the reviewer's judgement in the next round.
    """
    parsed_matrix = parse_risk_test_matrix(matrix)  # shape-check: fatal:orchestrator-authored
    identity = expected_identity or risk_test_matrix_identity(parsed_matrix)  # shape-check: fatal:authentication-or-forgery
    if not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise AgentLoopError("derived risk evidence matrix identity must be a SHA-256 digest")  # shape-check: fatal:orchestrator-authored
    # ``observations`` is the authoritative journal used to derive aggregate
    # failure status.  Selector resolution is deliberately a separate input:
    # a cumulative journal may retain observations from earlier coder turns,
    # but a fresh semantic claim may select only the closed catalog for the
    # current turn.  Keeping these inputs distinct prevents a historical
    # execution_ref from becoming a valid citation merely because it remains
    # inside the bounded journal.
    selector_observations = (
        tuple(execution_catalog)
        if execution_catalog is not None
        else tuple(observations)
    )
    observation_by_ref: dict[str, object] = {}
    for observation in selector_observations:
        execution_ref = _semantic_execution_ref(observation)
        if execution_ref is None:
            continue
        if execution_ref in observation_by_ref:
            raise AgentLoopError(  # shape-check: fatal:orchestrator-authored
                f"execution catalog contains colliding execution_ref `{execution_ref}`"
            )
        observation_by_ref[execution_ref] = observation

    claim_values = _claims_value(claims)
    if len(claim_values) > SEMANTIC_RISK_CLAIMS_MAX_ROWS:
        raise AgentLoopError(  # shape-check: fatal:orchestrator-authored
            f"semantic risk coverage exceeds the {SEMANTIC_RISK_CLAIMS_MAX_ROWS}-row bound"
        )
    enforceable_row_ids = {
        row.row_id for row in parsed_matrix.rows
        if row.applicability in {"applicable", "required"}
    }
    claim_by_row: dict[str, SemanticRiskCoverageClaim] = {}
    for claim in claim_values:
        if not isinstance(claim, SemanticRiskCoverageClaim):
            raise AgentLoopError("semantic risk coverage contains an invalid claim type")  # shape-check: fatal:orchestrator-authored
        if claim.row_id not in enforceable_row_ids:
            raise AgentLoopError(  # shape-check: fatal:orchestrator-authored
                f"semantic risk coverage names unknown or non-enforceable row `{claim.row_id}`"
            )
        if claim.row_id in claim_by_row:
            raise AgentLoopError(  # shape-check: fatal:orchestrator-authored
                f"semantic risk coverage contains duplicate claim for row `{claim.row_id}`"
            )
        claim_by_row[claim.row_id] = claim
        # A shared run may evidence several rows (#865); only reject repeats
        # within one row's selection.
        if len(set(claim.execution_refs)) != len(claim.execution_refs):
            raise AgentLoopError(  # shape-check: fatal:orchestrator-authored
                f"semantic risk coverage for row `{claim.row_id}` selects an execution_ref more than once"
            )
    diagnostics: list[PostAuthClaimDiagnostic] = []
    # A canonical evidence builder is a post-authentication operation. Keep
    # these parameters optional for historical/unit callers, but never let an
    # omitted authentication proof upgrade a selected receipt to ``verified``.
    # Matching the receipt's recorded tree is insufficient when the assigned
    # checkout is on another commit.  Unpushed commits make the checkout head
    # differ from the fetched PR head, and uncommitted edits fail the
    # clean-tree and tree-digest proofs.  Whether the follow-up moved the head
    # is deliberately not an admissibility input (#1627): a receipt that passes
    # these proofs is a true test of the current PR head.
    checkout_head_mismatch = (
        current_head is None
        or authenticated_checkout_head != current_head
    )
    checkout_tree_unavailable = authenticated_tree_clean is not True
    unsuperseded_failures = [
        observation for observation in observations
        if _observation_value(observation, "provenance") == "parent-observed"
        and _observation_value(observation, "outcome") in {
            "failed", "timed_out", "interrupted", "incomplete", "launch-failed"
        }
        and not _observation_value(observation, "superseded_by")
        and not _is_non_evidence_launch_failure(observation)
    ]
    if isinstance(claims, SemanticRiskCoverageClaims):
        owner_by_row = {row.row_id: row.execution_owner for row in parsed_matrix.rows}
        # One diagnostic per dropped claim (#926).  Each unapproved record is
        # paired with its exact row ID at parse time, because the record's
        # preview is bounded and two records can compare equal; iterating the
        # pairs keeps every drop, even equal-looking ones, attributed.
        covered_dropped: set[str] = set()
        paired_records: dict[ParseDegradation, int] = {}
        for record, dropped_row_id in claims.unapproved_claim_row_ids:
            paired_records[record] = paired_records.get(record, 0) + 1
            covered_dropped.add(dropped_row_id)
            diagnostics.append(PostAuthClaimDiagnostic(
                dropped_row_id,
                UNAPPROVED_ROW_CLAIM_DIAGNOSTIC,
                _unapproved_row_claim_message(  # shape-check: fatal:authentication-or-forgery
                    dropped_row_id,
                    row_owner=owner_by_row.get(dropped_row_id),
                    execution_owner=execution_owner,
                    record=record,
                ),
            ))
        for record in claims.degradations:
            if record.rule != CLAIM_ROW_ID_UNAPPROVED_RULE:
                continue
            if paired_records.get(record, 0):
                paired_records[record] -= 1
                continue
            # A record with no exact ID cannot be attributed to a row, so it
            # is keyed by its element path like other drops.
            diagnostics.append(PostAuthClaimDiagnostic(
                record.element_path,
                UNAPPROVED_ROW_CLAIM_DIAGNOSTIC,
                _degraded_row_claim_message(record),  # shape-check: fatal:authentication-or-forgery
            ))
        for dropped_row_id in claims.dropped_row_ids:
            if dropped_row_id in covered_dropped:
                continue
            # A carrier built without parser records (historical callers)
            # still reports each dropped ID once (#920).
            diagnostics.append(PostAuthClaimDiagnostic(
                dropped_row_id,
                UNAPPROVED_ROW_CLAIM_DIAGNOSTIC,
                _unapproved_row_claim_message(  # shape-check: fatal:authentication-or-forgery
                    dropped_row_id,
                    row_owner=owner_by_row.get(dropped_row_id),
                    execution_owner=execution_owner,
                ),
            ))
        for record in claims.degradations:
            if record.rule == CLAIM_ROW_ID_UNAPPROVED_RULE:
                continue
            # Dropped before authentication (#926).  Keyed by the claim's
            # element path, never by agent text; a duplicated approved row
            # additionally reads ``missing`` below.
            diagnostics.append(PostAuthClaimDiagnostic(
                record.element_path,
                DEGRADED_ROW_CLAIM_DIAGNOSTIC,
                _degraded_row_claim_message(record),  # shape-check: fatal:authentication-or-forgery
            ))
    result_rows: list[RiskTestMatrixEvidenceRow] = []
    capture_classifications: dict[str, dict[str, object]] = {}
    journal_disqualifies = _journal_disqualifies(observations)
    for row in parsed_matrix.rows:
        if row.applicability not in {"applicable", "required"}:
            continue
        claim = claim_by_row.get(row.row_id)
        caveats: list[str] = list(claim.caveats if claim is not None else ())
        citations: list[TestObservationCitation] = []
        valid_selected = True
        selected: list[object] = []
        # Structured capture classification (#1325), recorded beside each
        # diagnostic below and never derived from diagnostic codes.
        capture_classes: list[str] = []
        capture_disqualifiers: set[str] = {"journal-failure"} if journal_disqualifies else set()
        if claim is None:
            capture_disqualifiers.add("missing-claim")
            status = "missing"
            caveats.append("No semantic coverage claim was supplied for this enforceable row.")
            diagnostics.append(PostAuthClaimDiagnostic(row.row_id, "missing-claim", "No semantic row claim was supplied."))
        else:
            # Checkout authentication disqualifies the attempt independently of
            # any selector (#1325): a claim selecting only unknown or dropped
            # refs from a wrong or unauthenticated checkout never converts.
            if checkout_head_mismatch:
                capture_disqualifiers.add("checkout-head-mismatch")
            if checkout_tree_unavailable:
                capture_disqualifiers.add("checkout-tree-unavailable")
            candidate_citations: list[TestObservationCitation] = []
            missing_semantic_facts = [
                field_name
                for field_name in SEMANTIC_RISK_CLAIM_FACT_KEYS
                if not getattr(claim, field_name)
            ]
            if missing_semantic_facts:
                capture_disqualifiers.add("incomplete-semantic-claim")
                valid_selected = False
                diagnostics.append(PostAuthClaimDiagnostic(
                    row.row_id,
                    "incomplete-semantic-claim",
                    "Semantic coverage is missing required facts: "
                    + ", ".join(missing_semantic_facts)
                    + ".",
                ))
                caveats.append(
                    "The semantic coverage claim did not provide all facts required for verified evidence."
                )
            if claim.truncated_fact_fields:
                # Truncation before authentication is accepted content loss
                # (#913), so the retained prefix cannot stand in for the whole
                # claim: the row stays unverified and says why.
                capture_disqualifiers.add("truncated-semantic-claim")
                valid_selected = False
                diagnostics.append(PostAuthClaimDiagnostic(
                    row.row_id,
                    "truncated-semantic-claim",
                    f"Semantic coverage exceeded the {RISK_MATRIX_MAX_LIST_ITEMS}-item "
                    "list bound and was truncated: "
                    + ", ".join(claim.truncated_fact_fields)
                    + ".",
                ))
                caveats.append(
                    "The semantic coverage claim lost listed facts to the per-list bound, "
                    "so it cannot be verified."
                )
            for dropped_ref in claim.dropped_execution_refs:
                # Dropped before authentication (#859): never selectable, so
                # they stay out of ``selected`` and cannot yield citations.
                capture_classes.append("unknown-ref")
                valid_selected = False
                diagnostics.append(PostAuthClaimDiagnostic(
                    row.row_id, "unknown-execution-ref",
                    f"Execution selector `{_dropped_ref_preview(dropped_ref)}` was not present "  # shape-check: fatal:authentication-or-forgery
                    "in the authenticated turn catalog.",
                ))
                caveats.append("A claimed execution selector was unknown or cross-turn.")
            for execution_ref in claim.execution_refs:
                observation = observation_by_ref.get(execution_ref)
                if observation is None:
                    capture_classes.append("unknown-ref")
                    valid_selected = False
                    diagnostics.append(PostAuthClaimDiagnostic(
                        row.row_id, "unknown-execution-ref",
                        f"Execution selector `{execution_ref}` was not present in the authenticated turn catalog.",
                    ))
                    caveats.append("A claimed execution selector was unknown or cross-turn.")
                    continue
                selected.append(observation)
                selected_observation_valid = True
                semantics, _rich = _observation_semantics(observation)
                capture_classes.append(capture_observation_class(observation))
                if invocation_id is not None and _observation_value(observation, "turn_id") != invocation_id:
                    capture_classes.append("unknown-ref")
                    valid_selected = False
                    selected_observation_valid = False
                    diagnostics.append(PostAuthClaimDiagnostic(
                        row.row_id, "cross-turn-execution-ref",
                        f"Execution selector `{execution_ref}` was not bound to the current invocation.",
                    ))
                attribution = _observation_attribution(observation)
                if checkout_head_mismatch:
                    capture_disqualifiers.add("checkout-head-mismatch")
                    valid_selected = False
                    selected_observation_valid = False
                    diagnostics.append(PostAuthClaimDiagnostic(
                        row.row_id, "checkout-head-mismatch",
                        "The assigned checkout was not authenticated at the exact current PR head.",
                    ))
                if checkout_tree_unavailable:
                    capture_disqualifiers.add("checkout-tree-unavailable")
                    valid_selected = False
                    selected_observation_valid = False
                    diagnostics.append(PostAuthClaimDiagnostic(
                        row.row_id, "checkout-tree-unavailable",
                        "The assigned checkout was not authenticated as a stable clean tracked tree.",
                    ))
                if current_head is not None and attribution.get("head") not in {None, current_head}:
                    capture_disqualifiers.add("head-mismatch")
                    valid_selected = False
                    selected_observation_valid = False
                    diagnostics.append(PostAuthClaimDiagnostic(
                        row.row_id, "head-mismatch",
                        f"Execution selector `{execution_ref}` was attributed to a different head.",
                    ))
                if current_tree_digest is not None and attribution.get("tracked_digest") not in {None, current_tree_digest}:
                    capture_disqualifiers.add("tree-mismatch")
                    valid_selected = False
                    selected_observation_valid = False
                    diagnostics.append(PostAuthClaimDiagnostic(
                        row.row_id, "tree-mismatch",
                        f"Execution selector `{execution_ref}` was attributed to a different tracked tree.",
                    ))
                if not _authoritative_receipt_passes(observation, claim="current-result"):
                    valid_selected = False
                    selected_observation_valid = False
                    expected = _receipt_expected_status(observation, claim="current-result") or "stale/unverified"
                    message = f"Execution selector `{execution_ref}` is not an admissible passing receipt ({expected})."
                    if _only_suite_start_unverified(observation):
                        capture_disqualifiers.add("suite-start-unverified")
                        message += " " + suite_start_unverified_diagnostic(observation)
                        message += _node_test_hint(observation, message)  # shape-check: fatal:authentication-or-forgery
                    diagnostics.append(PostAuthClaimDiagnostic(row.row_id, "inadmissible-execution", message))
                else:
                    receipt_id = _observation_value(observation, "receipt_id")
                    if not isinstance(receipt_id, str) or not receipt_id.strip():
                        valid_selected = False
                        selected_observation_valid = False
                        diagnostics.append(PostAuthClaimDiagnostic(
                            row.row_id, "missing-receipt", "An admissible execution had no tool-owned receipt ID."
                        ))
                    else:
                        if selected_observation_valid:
                            candidate_citations.append(TestObservationCitation(
                                command=_observation_command_text(observation),
                                receipt_id=receipt_id,
                                claim="current-result",
                            ))
            if not selected:
                valid_selected = False
            # A row is verified only when every selected observation is
            # admissible.  Do not leak a citation for an otherwise passing
            # subset when any selected execution failed invocation, checkout,
            # head, tree, launch, or receipt authority checks.
            if valid_selected:
                citations.extend(candidate_citations)
            status = "verified" if valid_selected and citations else "stale/unverified"
            if not valid_selected:
                caveats.append("One or more selected executions failed post-authentication authority checks.")
            if valid_selected and unsuperseded_failures:
                status = "incomplete"
                caveats.append("An unsuperseded authoritative failure or timeout remains in the complete test journal.")
                diagnostics.append(PostAuthClaimDiagnostic(
                    row.row_id, "unsuperseded-journal-failure",
                    "A passing subset cannot erase an unsuperseded authoritative journal failure or timeout.",
                ))
        # Canonical assertions fall back to the approved row when no semantic
        # claim is available, so every enforceable row remains explicit.
        test_identifiers = claim.test_identifiers if claim is not None else ()
        test_locations = claim.test_locations if claim is not None else ()
        workflow_path = claim.workflow_path_claim if claim is not None else row.entry_path_or_mode
        outcome_assertions = claim.outcome_assertions if claim is not None else (row.expected_outcome,)
        forbidden_assertions = claim.forbidden_effect_assertions if claim is not None else row.forbidden_side_effects
        result_rows.append(RiskTestMatrixEvidenceRow(
            row_id=row.row_id,
            status=status,
            test_identifiers=test_identifiers,
            test_locations=test_locations,
            workflow_path_claim=workflow_path,
            outcome_assertions=outcome_assertions,
            forbidden_effect_assertions=forbidden_assertions,
            evidence_citations=tuple(citations),
            caveats=tuple(dict.fromkeys(caveats)),
        ))
        capture_classifications[row.row_id] = _row_capture_classification(
            status=status,
            observation_classes=capture_classes,
            disqualifiers=capture_disqualifiers,
        )
    return DerivedRiskEvidenceResult(
        evidence=RiskTestMatrixEvidence(identity, tuple(result_rows)),
        diagnostics=tuple(diagnostics),
        row_capture_classifications=capture_classifications,
    )


# Public aliases used by integrations and tests that describe the operation as
# either building or deriving the canonical object.
build_risk_test_matrix_evidence = derive_risk_test_matrix_evidence


def _parse_risk_evidence_citations(value: object, *, context: str) -> tuple[TestObservationCitation, ...]:
    return _expect_test_observations(value, context=context)  # shape-check: fatal:authentication-or-forgery


def parse_risk_test_matrix_evidence(
    value: object,
    *,
    matrix: RiskTestMatrix | Mapping[str, object] | None = None,
    expected_identity: str | None = None,
    authoritative_test_observations: Sequence[object] | None = None,
    expected_row_ids: Sequence[str] | None = None,
    context: str = "risk_test_matrix_evidence",
    trusted_signed_references: Sequence[SignedEvidenceReference] | None = None,
    expected_signed_head: str | None = None,
) -> RiskTestMatrixEvidence:
    """Parse canonical evidence.

    ``trusted_signed_references`` (#1325) are orchestrator-decoded, admitted
    references.  Without them any ``evidence_source`` or
    ``signed_evidence_reference`` key is rejected, which covers every agent
    output path.  With them, a receipt-free verified row is accepted only when
    its reference resolves to a trusted reference for the same row, matrix
    identity and ``expected_signed_head``.
    """
    payload = _expect_object(value, context=context)  # shape-check: fatal:authentication-or-forgery
    _expect_exact_keys(payload, context=context, required={"matrix_identity", "rows"})  # shape-check: fatal:authentication-or-forgery
    identity = _risk_bounded_string(payload["matrix_identity"], context=f"{context}.matrix_identity", max_bytes=128)  # shape-check: fatal:authentication-or-forgery
    if not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise AgentLoopError(f"{context}.matrix_identity must be a SHA-256 digest.")  # shape-check: fatal:authentication-or-forgery
    if expected_identity is not None and identity != expected_identity:
        raise AgentLoopError(f"{context}.matrix_identity does not match the delivered approved matrix.")  # shape-check: fatal:authentication-or-forgery
    if matrix is not None and expected_identity is None and identity != risk_test_matrix_identity(matrix):  # shape-check: fatal:authentication-or-forgery
        raise AgentLoopError(f"{context}.matrix_identity does not match the delivered matrix payload.")  # shape-check: fatal:authentication-or-forgery
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list):
        raise AgentLoopError(f"{context}.rows must be a JSON array.")  # shape-check: fatal:authentication-or-forgery
    if len(raw_rows) > RISK_MATRIX_MAX_ROWS:
        raise AgentLoopError(f"{context}.rows exceeds the {RISK_MATRIX_MAX_ROWS}-row bound.")  # shape-check: fatal:payload-bound
    parsed_matrix = parse_risk_test_matrix(matrix) if matrix is not None else None  # shape-check: fatal:authentication-or-forgery
    expected_ids = (
        set(expected_row_ids)
        if expected_row_ids is not None
        else (
            {
                row.row_id
                for row in parsed_matrix.rows
                if row.applicability in {"applicable", "required"}
            }
            if parsed_matrix is not None and parsed_matrix.is_applicable
            else None
        )
    )
    result: list[RiskTestMatrixEvidenceRow] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(raw_rows):
        row_context = f"{context}.rows[{index}]"
        row = _expect_object(raw_row, context=row_context)  # shape-check: fatal:authentication-or-forgery
        _expect_exact_keys(  # shape-check: fatal:authentication-or-forgery
            row,
            context=row_context,
            required={
                "row_id", "status", "test_identifiers", "test_locations", "workflow_path_claim",
                "outcome_assertions", "forbidden_effect_assertions", "evidence_citations",
            },
            optional={"caveats", "evidence_source", "signed_evidence_reference"},
        )
        row_id = _validate_risk_row_id(row["row_id"], context=f"{row_context}.row_id")  # shape-check: fatal:no-conservative-reading
        if row_id in seen:
            raise AgentLoopError(f"{context} maps row `{row_id}` more than once.")  # shape-check: fatal:authentication-or-forgery
        seen.add(row_id)
        status = _risk_bounded_string(row["status"], context=f"{row_context}.status", max_bytes=64)  # shape-check: fatal:authentication-or-forgery
        if status not in RISK_MATRIX_EVIDENCE_STATUSES:
            raise AgentLoopError(f"{row_context}.status is invalid.")  # shape-check: fatal:authentication-or-forgery
        parsed_row = RiskTestMatrixEvidenceRow(
            row_id=row_id,
            status=status,
            test_identifiers=_risk_bounded_string_list(row["test_identifiers"], context=f"{row_context}.test_identifiers"),  # shape-check: fatal:authentication-or-forgery
            test_locations=_risk_bounded_string_list(row["test_locations"], context=f"{row_context}.test_locations"),  # shape-check: fatal:authentication-or-forgery
            workflow_path_claim=_risk_bounded_string(row["workflow_path_claim"], context=f"{row_context}.workflow_path_claim"),  # shape-check: fatal:authentication-or-forgery
            outcome_assertions=_risk_bounded_string_list(row["outcome_assertions"], context=f"{row_context}.outcome_assertions"),  # shape-check: fatal:authentication-or-forgery
            forbidden_effect_assertions=_risk_bounded_string_list(row["forbidden_effect_assertions"], context=f"{row_context}.forbidden_effect_assertions"),  # shape-check: fatal:authentication-or-forgery
            evidence_citations=_parse_risk_evidence_citations(row["evidence_citations"], context=f"{row_context}.evidence_citations"),  # shape-check: fatal:authentication-or-forgery
            caveats=_risk_bounded_string_list(  # shape-check: fatal:authentication-or-forgery
                row.get("caveats", []),
                context=f"{row_context}.caveats",
                max_items=RISK_MATRIX_MAX_CAVEATS,
            ),
        )
        if "evidence_source" in row or "signed_evidence_reference" in row:
            parsed_row = _parse_operator_signed_row(  # shape-check: fatal:authentication-or-forgery
                row,
                parsed_row,
                row_context=row_context,
                matrix_identity=identity,
                trusted_signed_references=trusted_signed_references,
                expected_signed_head=expected_signed_head,
            )
            result.append(parsed_row)
            continue
        if status == "verified" and (
            not parsed_row.test_identifiers
            or not parsed_row.test_locations
            or not parsed_row.outcome_assertions
            or not parsed_row.forbidden_effect_assertions
            or not parsed_row.evidence_citations
        ):
            raise AgentLoopError(  # shape-check: fatal:authentication-or-forgery
                f"{row_context} with status `verified` must include test identifiers, locations, "
                "outcome and forbidden-effect assertions, and evidence citations."
            )
        if authoritative_test_observations is not None:
            invalid: list[str] = []
            matched_observations: list[object] = []
            expected_statuses: set[str] = set()
            for citation in parsed_row.evidence_citations:
                citation_matches = [
                    observation
                    for observation in authoritative_test_observations
                    if _citation_matches_observation(citation, observation)
                ]
                matched_observations.extend(citation_matches)
                if not citation_matches or (
                    status == "verified"
                    and not any(
                        _authoritative_receipt_passes(observation, claim=citation.claim)
                        for observation in citation_matches
                    )
                ):
                    invalid.append(citation.receipt_id)
                expected_statuses.update(
                    expected
                    for observation in citation_matches
                    if (expected := _receipt_expected_status(observation, claim=citation.claim)) is not None
                )
            if invalid:
                raise AgentLoopError(  # shape-check: fatal:authentication-or-forgery
                    f"{row_context} contains citations without matching authoritative "
                    + ("passing " if status == "verified" else "")
                    + "test receipts: " + ", ".join(invalid)
                )
            if expected_statuses and (
                (len(expected_statuses) == 1 and status not in expected_statuses)
                or (len(expected_statuses) > 1 and status != "incomplete")
            ):
                raise AgentLoopError(  # shape-check: fatal:authentication-or-forgery
                    f"{row_context} status `{status}` contradicts the authoritative receipt "
                    f"outcome; expected {sorted(expected_statuses)} or `incomplete` for mixed receipts."
                )
            if matched_observations:
                # Receipt and attribution caveats are authoritative evidence,
                # not optional coder prose. Carry every one into the row so a
                # repair or renderer cannot turn a caveated result into a
                # clean-looking verification. The bounded validator rejects
                # overflow rather than silently truncating caveats; caveats
                # have their own allowance because several receipts can each
                # contribute authoritative attribution and broker caveats.
                all_caveats = list(parsed_row.caveats)
                for observation in matched_observations:
                    semantics, _rich = _observation_semantics(observation)
                    for caveat in (*semantics["attribution_caveats"], *semantics["caveats"]):
                        rendered_caveat = _risk_bounded_string(  # shape-check: fatal:authentication-or-forgery
                            caveat,
                            context=f"{row_context}.authoritative_caveat",
                        )
                        if rendered_caveat not in all_caveats:
                            all_caveats.append(rendered_caveat)
                parsed_row = dataclasses.replace(
                    parsed_row,
                    caveats=_risk_bounded_string_list(  # shape-check: fatal:authentication-or-forgery
                        all_caveats,
                        context=f"{row_context}.caveats",
                        max_items=RISK_MATRIX_MAX_CAVEATS,
                    ),
                )
        result.append(parsed_row)
    if expected_ids is not None and {row.row_id for row in result} != expected_ids:
        missing = sorted(expected_ids - {row.row_id for row in result})
        extra = sorted({row.row_id for row in result} - expected_ids)
        raise AgentLoopError(f"{context} must map every delivered row exactly once (missing={missing}, extra={extra}).")  # shape-check: fatal:authentication-or-forgery
    return RiskTestMatrixEvidence(matrix_identity=identity, rows=tuple(result))


def _parse_operator_signed_row(
    row: Mapping[str, object],
    parsed_row: RiskTestMatrixEvidenceRow,
    *,
    row_context: str,
    matrix_identity: str,
    trusted_signed_references: Sequence[SignedEvidenceReference] | None,
    expected_signed_head: str | None,
) -> RiskTestMatrixEvidenceRow:
    """Accept an operator-signed row only through a trusted bound reference (#1325)."""
    if trusted_signed_references is None:
        raise AgentLoopError(  # shape-check: fatal:authentication-or-forgery
            f"{row_context} carries evidence provenance; only the orchestrator may label a "
            "row operator-signed."
        )
    source = row.get("evidence_source")
    reference_id = row.get("signed_evidence_reference")
    if source != EVIDENCE_SOURCE_OPERATOR_SIGNED or not isinstance(reference_id, str):
        raise AgentLoopError(  # shape-check: fatal:authentication-or-forgery
            f"{row_context} has invalid evidence provenance."
        )
    reference = next(
        (ref for ref in trusted_signed_references if ref.reference_id == reference_id), None
    )
    if (
        reference is None
        or expected_signed_head is None
        or reference.row_id != parsed_row.row_id
        or reference.matrix_identity != matrix_identity
        or reference.head != expected_signed_head
    ):
        raise AgentLoopError(  # shape-check: fatal:authentication-or-forgery
            f"{row_context} names a signed-evidence reference that is not bound to this row, "
            "matrix and head."
        )
    if (
        parsed_row.status != "verified"
        or parsed_row.evidence_citations
        or not parsed_row.outcome_assertions
        or not parsed_row.forbidden_effect_assertions
    ):
        raise AgentLoopError(  # shape-check: fatal:authentication-or-forgery
            f"{row_context} operator-signed rows must be verified, receipt-free, and carry "
            "outcome and forbidden-effect assertions."
        )
    return dataclasses.replace(
        parsed_row,
        evidence_source=EVIDENCE_SOURCE_OPERATOR_SIGNED,
        signed_evidence_reference=reference_id,
    )


def apply_signed_evidence_references(
    evidence: RiskTestMatrixEvidence,
    references: Sequence[SignedEvidenceReference],
    *,
    matrix_identity: str,
    current_head: str | None,
    approved_matrix: RiskTestMatrix | Mapping[str, object] | None = None,
) -> RiskTestMatrixEvidence:
    """The only producer of operator-signed rows (#1325).

    A row whose canonical status is not ``verified`` becomes verified with
    ``evidence_source: operator-signed`` when an admitted reference matches
    its row, the matrix identity and ``current_head``.  The persisted coder
    evidence is never rewritten; callers overlay it for decisions and display.
    """
    if not current_head or evidence.matrix_identity != matrix_identity:
        return evidence
    by_row = {
        reference.row_id: reference
        for reference in references
        if reference.matrix_identity == matrix_identity and reference.head == current_head
    }
    if not by_row:
        return evidence
    approved_rows: dict[str, RiskTestMatrixRow] = {}
    if approved_matrix is not None:
        approved_rows = {row.row_id: row for row in parse_risk_test_matrix(approved_matrix).rows}  # shape-check: fatal:orchestrator-authored
    rows: list[RiskTestMatrixEvidenceRow] = []
    for row in evidence.rows:
        reference = by_row.get(row.row_id)
        if reference is None or row.status == "verified":
            rows.append(row)
            continue
        approved = approved_rows.get(row.row_id)
        outcome = (approved.expected_outcome,) if approved is not None else row.outcome_assertions
        forbidden = approved.forbidden_side_effects if approved is not None else row.forbidden_effect_assertions
        rows.append(dataclasses.replace(
            row,
            status="verified",
            outcome_assertions=tuple(outcome) or ("Operator-signed exact-head evidence.",),
            forbidden_effect_assertions=tuple(forbidden) or ("No forbidden effect was reported.",),
            evidence_citations=(),
            caveats=(f"Verified by operator-signed evidence for head {current_head}.",),
            evidence_source=EVIDENCE_SOURCE_OPERATOR_SIGNED,
            signed_evidence_reference=reference.reference_id,
        ))
    return dataclasses.replace(evidence, rows=tuple(rows))


def _citation_matches_observation(
    citation: TestObservationCitation,
    observation: object,
) -> bool:
    """Match a coder citation to a live broker observation, not just its shape."""
    projected_value: Mapping[str, object] | None = None
    projected = getattr(observation, "public_projection", None)
    if callable(projected):
        try:
            value = projected()
        except Exception:  # pragma: no cover - defensive provider boundary
            value = None
        if isinstance(value, Mapping):
            projected_value = value
    elif isinstance(observation, Mapping):
        projected_value = observation

    def observed(name: str, default: object = None) -> object:
        value = getattr(observation, name, default)
        if value is not default:
            return value
        if projected_value is not None:
            return projected_value.get(name, default)
        return default

    observed_claim = observed("claim")
    if (
        observed("receipt_id") != citation.receipt_id
        or (
            observed_claim is not None
            and observed_claim != citation.claim
        )
        or (observed_claim is None and citation.claim != "current-result")
    ):
        return False
    commands: set[str] = set()
    if projected_value is not None and isinstance(projected_value.get("command"), str):
        commands.add(projected_value["command"])
    normalized = observed("normalized_command")
    if isinstance(normalized, str):
        commands.add(normalized)
    command = observed("command")
    if isinstance(command, (tuple, list)) and all(isinstance(item, str) for item in command):
        commands.add(shlex.join(command))
    if citation.command in commands:
        return True
    wrapped_command = _managed_test_wrapper_inner_command(citation.command)
    return wrapped_command is not None and wrapped_command in commands


def _observation_semantics(observation: object) -> tuple[dict[str, object], bool]:
    """Return receipt semantics and whether this is a rich broker receipt.

    Older tests and historical records may expose only outcome/provenance and
    a command projection. They remain matchable, while current broker
    observations are required to pass the complete attribution and caveat
    checks before a row can claim ``verified``.
    """
    projected_value: Mapping[str, object] | None = None
    projected = getattr(observation, "public_projection", None)
    if callable(projected):
        try:
            value = projected()
        except Exception:  # pragma: no cover - defensive provider boundary
            value = None
        if isinstance(value, Mapping):
            projected_value = value
    elif isinstance(observation, Mapping):
        projected_value = observation

    def value(name: str, default: object = None) -> object:
        marker = object()
        actual = getattr(observation, name, marker)
        if actual is not marker:
            return actual
        return projected_value.get(name, default) if projected_value is not None else default

    attribution = value("attribution")
    if attribution is None and projected_value is not None:
        attribution = projected_value.get("tree")
    if attribution is None:
        attribution = value("tree")
    if isinstance(attribution, Mapping):
        attribution_values = dict(attribution)
    else:
        attribution_values = {
            name: getattr(attribution, name, None)
            for name in ("state", "stable", "untracked_input", "caveats")
        } if attribution is not None else {}
    environment = value("environment_state")
    if environment is None and projected_value is not None:
        environment = projected_value.get("environment")
    semantics = {
        "outcome": value("outcome"),
        "provenance": value("provenance"),
        "attribution_state": attribution_values.get("state"),
        "attribution_stable": attribution_values.get("stable"),
        "untracked_input": attribution_values.get("untracked_input", False),
        "attribution_caveats": tuple(attribution_values.get("caveats") or ()),
        "environment": environment,
        "superseded_by": value("superseded_by"),
        "caveats": tuple(value("caveats", ()) or ()),
        "wrapper_bootstrap": value("wrapper_bootstrap"),
        "inner_exec": value("inner_exec"),
        "suite_start": value("suite_start"),
        # Issue #1661: diagnostic only; never part of ``rich`` or any
        # admission predicate.
        "suite_start_reason": value("suite_start_reason"),
        # Issue #1326: absent for fixture-less rows; any present value must
        # decode as a clean lifecycle before the row can be verified.
        "fixture_present": (
            getattr(observation, "fixture", None) is not None
            or (projected_value is not None and "fixture" in projected_value)
        ),
        "fixture": value("fixture"),
    }
    rich = isinstance(attribution, (Mapping,)) or attribution is not None or any(
        name in (projected_value or {})
        for name in (
            "environment", "superseded_by", "caveats", "wrapper_bootstrap", "inner_exec",
            "suite_start", "fixture",
        )
    )
    return semantics, rich


def _fixture_lifecycle_is_clean(value: object) -> bool:
    """Whether a fixture record proves setup 0, teardown 0 and confirmed cleanup."""
    from .local_test_evidence import FixtureRecord

    if isinstance(value, FixtureRecord):
        record = value
    else:
        try:
            record = FixtureRecord.from_mapping(value)
        except Exception:
            # Fail closed on any malformed record, including overflow.
            return False
    return record.clean


def _authoritative_receipt_passes(
    observation: object,
    *,
    claim: str,
) -> bool:
    semantics, rich = _observation_semantics(observation)
    if semantics["outcome"] != "passed" or semantics["provenance"] != "parent-observed":
        return False
    if not rich:
        return True
    expected_attribution = "base-reproduction" if claim == "base-reproduction" else "current-head"
    if semantics["attribution_state"] != expected_attribution:
        return False
    if semantics["attribution_stable"] is not True or semantics["untracked_input"]:
        return False
    if semantics["environment"] not in {None, "equivalent", "not-compared"}:
        return False
    if semantics["superseded_by"]:
        return False
    if semantics["fixture_present"] and not _fixture_lifecycle_is_clean(semantics["fixture"]):
        return False
    if (
        semantics["wrapper_bootstrap"] != "verified"
        or semantics["inner_exec"] != "started"
        or semantics["suite_start"] != "verified"
    ):
        return False
    caveats = " ".join(
        str(item).casefold()
        for item in (*semantics["attribution_caveats"], *semantics["caveats"])
    )
    return not any(
        token in caveats
        for token in (
            "stale", "untracked", "environment", "changed", "mismatch", "supersed",
            "timeout", "timed out", "incomplete", "unknown", "disagreement",
        )
    )


_LAUNCH_FAILURE_PHRASES: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "wrapper_bootstrap",
        {
            "failed": "the run-tests wrapper bootstrap failed (wrapper_bootstrap=failed)",
            "unknown": "the wrapper bootstrap was never verified, e.g. an unrecognized launcher (wrapper_bootstrap=unknown)",
        },
    ),
    (
        "inner_exec",
        {
            "not-attempted": "the inner test command was never executed (inner_exec=not-attempted)",
            "failed": "the inner test command failed to exec (inner_exec=failed)",
        },
    ),
    (
        "suite_start",
        {
            "not-started": "the test suite never started (suite_start=not-started)",
            "unknown": "suite start was never verified (suite_start=unknown)",
        },
    ),
)
_LAUNCH_PASSING_STATES = {
    "wrapper_bootstrap": "verified",
    "inner_exec": "started",
    "suite_start": "verified",
}


def _launch_integrity_failure_reasons(observation: object) -> tuple[str, ...]:
    """Plain-language phrases for each failing launch condition, in fixed order."""
    semantics, _rich = _observation_semantics(observation)
    reasons: list[str] = []
    for name, phrases in _LAUNCH_FAILURE_PHRASES:
        value = semantics[name]
        if value == _LAUNCH_PASSING_STATES[name]:
            continue
        if value is None:
            reasons.append(f"{name} was not recorded")
        elif isinstance(value, str) and value in phrases:
            reasons.append(phrases[value])
        else:
            reasons.append(f"{name} has unrecognized state")
    return tuple(reasons)


def _safe_label(text: str, limit: int) -> str:
    cleaned = "".join(" " if ch.isspace() or ord(ch) < 32 or ch == "\x7f" else ch for ch in text)
    cleaned = cleaned.replace("`", "'").strip()
    return cleaned.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _observation_projection_field(observation: object, name: str) -> object:
    if isinstance(observation, Mapping):
        return observation.get(name)
    projected = getattr(observation, "public_projection", None)
    if callable(projected):
        try:
            value = projected()
        except Exception:  # pragma: no cover - defensive provider boundary
            return None
        if isinstance(value, Mapping):
            return value.get(name)
    return None


_NODE_HINT_FILE_RE = re.compile(r"[A-Za-z0-9 ._/@+,=-]+")
_NODE_HINT_EXTENSIONS = (".js", ".mjs", ".cjs", ".ts")


EVIDENCE_REASK_MAX_CHARS = 1500


def render_evidence_rejection_detail(detail: str) -> str:
    """The exact text the evidence re-ask prompt quotes for a rejection detail.

    Single definition shared with the re-ask renderer, so anything validated
    against it reaches the coder unaltered: markers neutralized, whitespace runs
    collapsed, and the whole detail capped.
    """
    neutralized = sanitize_historical_text(detail)  # shape-check: fatal:authentication-or-forgery
    quoted = " ".join(neutralized.replace("<", "(").replace(">", ")").split())
    if len(quoted) > EVIDENCE_REASK_MAX_CHARS:
        quoted = quoted[: EVIDENCE_REASK_MAX_CHARS - 3] + "..."
    return quoted


def _node_test_hint(observation: object, message: str = "") -> str:
    """Name the verifiable ``node --test <file>`` spelling, or nothing (#1294).

    Derived only from the redacted projection, and only when the projection
    signals that redaction left every token byte-for-byte intact, so an
    altered, truncated or private path is never turned into a rerun command.
    """
    from .local_test_evidence import MAX_SAFE_COMMAND_BYTES, _UNPARSABLE_PREFIX
    from .test_runtime import _is_node_test_runner

    if _observation_projection_field(observation, "command_verbatim") is not True:
        return ""
    caveats = _observation_projection_field(observation, "caveats")
    if isinstance(caveats, (list, tuple)) and any(
        isinstance(item, str) and item.startswith(_UNPARSABLE_PREFIX) for item in caveats
    ):
        return ""
    command = _observation_projection_field(observation, "command")
    if isinstance(command, str):
        try:
            tokens = shlex.split(command)
        except ValueError:
            return ""
    elif isinstance(command, (list, tuple)):
        tokens = [str(item) for item in command]
    else:
        return ""
    if len(tokens) < 2 or tokens[0] not in {"node", "nodejs"}:
        return ""
    flags, file = tokens[1:-1], tokens[-1]
    if "--test" in flags or not _is_node_test_runner(["--test", *flags, file]):
        return ""
    parts = file.split("/")
    if (
        not _NODE_HINT_FILE_RE.fullmatch(file)
        or file.startswith(("/", "-"))
        or ".." in parts
        or not file.endswith(_NODE_HINT_EXTENSIONS)
        or any(part == "" for part in parts)
    ):
        return ""
    base = parts[-1].lower()
    if not (
        parts[0] in {"test", "tests", "e2e"}
        or any(word in base for word in ("test", "spec", "e2e"))
    ):
        return ""
    rerun = shlex.join(["node", "--test", *flags, file])
    sentence = (
        "To make it citable, re-run it through run-tests as: "
        f"{rerun} (a standalone script counts as one runner test that passes only if it exits 0)."
    )
    # Emit only a sentence that reaches the coder unaltered: it must survive the
    # label sanitizer AND the re-ask rendering (whitespace collapsing and the
    # whole-message cap) after the message that precedes it, so an altered,
    # collapsed or truncated rerun command is never shown.
    if _safe_label(sentence, MAX_SAFE_COMMAND_BYTES * 2) != sentence:
        return ""
    rendered = render_evidence_rejection_detail(message + " " + sentence)  # shape-check: fatal:authentication-or-forgery
    if not rendered.endswith(sentence):
        return ""
    return " " + sentence


def _observation_command_label(observation: object) -> str:
    """Redacted, bounded command text; never reads raw argv attributes."""
    from .local_test_evidence import _UNPARSABLE_PREFIX, MAX_SAFE_COMMAND_BYTES, redact_test_command

    command = _observation_projection_field(observation, "command")
    caveats = _observation_projection_field(observation, "caveats")
    unparsable = isinstance(caveats, (list, tuple)) and any(
        isinstance(item, str) and item.startswith(_UNPARSABLE_PREFIX) for item in caveats
    )
    if unparsable:
        return "command not recorded (unparsable)"
    if isinstance(observation, Mapping) and isinstance(command, str):
        # The redactor keeps an unparsable string as one opaque token, which
        # can hide a secret assignment; fail closed rather than echo it.
        # Only raw mapping strings are checked: a projection's display may be
        # truncated inside a quote while its source parsed fine.
        try:
            shlex.split(command)
        except ValueError:
            return "command not recorded (unparsable)"
    if isinstance(observation, Mapping) and command:
        try:
            if isinstance(command, (str, list, tuple)):
                command = redact_test_command(
                    [str(item) for item in command] if not isinstance(command, str) else command
                )[0]
            else:
                command = None
        except Exception:  # pragma: no cover - defensive redactor boundary
            command = None
    label = _safe_label(command, MAX_SAFE_COMMAND_BYTES) if isinstance(command, str) else ""
    return f"`{label}`" if label else "command not recorded"


def _observation_timestamp_label(observation: object) -> str:
    timestamp = _observation_projection_field(observation, "timestamp")
    label = _safe_label(timestamp, 64) if isinstance(timestamp, str) else ""
    return f"`{label}`" if label else "timestamp not recorded"


def _observation_outcome_failure_reasons(observation: object) -> tuple[str, ...]:
    from .local_test_evidence import OUTCOMES, PROVENANCES

    semantics, _rich = _observation_semantics(observation)
    reasons: list[str] = []
    outcome = semantics["outcome"]
    if outcome != "passed":
        shown = outcome if isinstance(outcome, str) and outcome in OUTCOMES else "unrecognized"
        reasons.append(f"outcome is {shown}, not passed")
    provenance = semantics["provenance"]
    if provenance != "parent-observed":
        shown = provenance if isinstance(provenance, str) and provenance in PROVENANCES else "unrecognized"
        reasons.append(f"provenance is {shown}, not parent-observed")
    return tuple(reasons)


def _known_launch_integrity_passes(observation: object) -> bool:
    """Reject explicit broker launch failures before PR/head authentication.

    Head and tree attribution are intentionally checked by the post-authentication
    builder. Launch-boundary states, however, are known as soon as the broker
    closes the current turn and must not be deferred into that phase. Older
    lightweight projections may omit these fields, so absence remains a
    compatibility case rather than being treated as a known failure.
    """
    launch_fields = ("wrapper_bootstrap", "inner_exec", "suite_start")
    if isinstance(observation, Mapping):
        supplied = any(field in observation for field in launch_fields)
    else:
        supplied = any(hasattr(observation, field) for field in launch_fields)
        if not supplied:
            projected = getattr(observation, "public_projection", None)
            if callable(projected):
                try:
                    value = projected()
                except Exception:  # pragma: no cover - defensive provider boundary
                    value = None
                supplied = isinstance(value, Mapping) and any(
                    field in value for field in launch_fields
                )
    if not supplied:
        return True
    semantics, _rich = _observation_semantics(observation)
    return not _launch_integrity_failure_reasons(observation)


SUITE_START_UNVERIFIED_DIAGNOSTIC = (
    "Suite start was never verified (suite_start=unknown; an unrecognized or unverified "
    "test launcher), so this receipt cannot be cited. Citable runners: "
    f"{CITABLE_TEST_RUNNERS_SUMMARY}. Wrap the check in a pytest test to make it citable."
)


SUITE_START_UNRECOGNIZED_DIAGNOSTIC = (
    "Suite start was never verified (suite_start=unknown): the test launcher is not "
    "recognized, so this receipt cannot be cited. Citable runners: "
    f"{CITABLE_TEST_RUNNERS_SUMMARY}. Wrap the check in a pytest test to make it citable."
)
SUITE_START_PROBE_UNAVAILABLE_DIAGNOSTIC = (
    "Suite start was never verified (suite_start=unknown): the test launcher is recognized, "
    "but its bootstrap probe could not verify it in this run (for example probe containment "
    "was unavailable), so this receipt cannot be cited. Rerunning the same command through "
    "`agent-loop run-tests` may make it citable."
)


def suite_start_unverified_diagnostic(observation: object) -> str:
    """The suite-start wording selected by the persisted reason (#1661).

    Presentation only.  A row without a reason (legacy, or an older broker)
    keeps the generic #1659 text.
    """
    from .test_runtime import (
        SUITE_START_REASON_PROBE_UNAVAILABLE,
        SUITE_START_REASON_UNRECOGNIZED,
        coerce_suite_start_reason,
    )

    semantics, _rich = _observation_semantics(observation)
    reason = coerce_suite_start_reason(semantics["suite_start_reason"], semantics["suite_start"])
    if reason == SUITE_START_REASON_UNRECOGNIZED:
        return SUITE_START_UNRECOGNIZED_DIAGNOSTIC
    if reason == SUITE_START_REASON_PROBE_UNAVAILABLE:
        return SUITE_START_PROBE_UNAVAILABLE_DIAGNOSTIC
    return SUITE_START_UNVERIFIED_DIAGNOSTIC


def _only_suite_start_unverified(observation: object) -> bool:
    """A passing parent-observed receipt whose only launch defect is suite_start=unknown.

    Issue #1659: such a selector is not an authority decision before
    authentication.  It reaches post-authentication derivation, where
    ``_authoritative_receipt_passes`` still refuses it non-terminally.  Every
    launch field must be explicitly supplied.
    """
    semantics, _rich = _observation_semantics(observation)
    return (
        semantics["outcome"] == "passed"
        and semantics["provenance"] == "parent-observed"
        and semantics["wrapper_bootstrap"] == "verified"
        and semantics["inner_exec"] == "started"
        and semantics["suite_start"] == "unknown"
    )


def _receipt_expected_status(observation: object, *, claim: str) -> str | None:
    semantics, rich = _observation_semantics(observation)
    if not rich:
        return None
    outcome = semantics["outcome"]
    if outcome == "passed":
        return "verified" if _authoritative_receipt_passes(observation, claim=claim) else "stale/unverified"
    return {
        "failed": "failed",
        "timed_out": "timed-out",
        "interrupted": "blocked",
        "launch-failed": "blocked",
        "overlap-rejected": "blocked",
        "incomplete": "incomplete",
    }.get(str(outcome))


def _parse_risk_test_matrix_contract_fields(
    payload: dict[str, object], *, context: str, required: bool = False,
    reject_unsolicited: bool = False,
) -> tuple[int | None, RiskTestMatrix | None, tuple[RiskTestMatrixChange, ...]]:
    present = {
        name: name in payload
        for name in ("risk_test_matrix_contract_version", "risk_test_matrix", "risk_test_matrix_changes")
    }
    if any(present.values()) and not all(present.values()):
        raise AgentLoopError(  # shape-check: fatal:kind-or-version-mismatch
            f"{context} risk matrix fields must include risk_test_matrix_contract_version, "
            "risk_test_matrix, and risk_test_matrix_changes together."
        )
    if not any(present.values()):
        if required:
            raise AgentLoopError(  # shape-check: fatal:kind-or-version-mismatch
                f"Fresh {context} responses require risk_test_matrix_contract_version: 1 "
                "and a complete risk_test_matrix contract."
            )
        return None, None, ()
    if reject_unsolicited:
        raise AgentLoopError(  # shape-check: fatal:kind-or-version-mismatch
            f"historical matrix-less {context} responses must not introduce a risk test "
            "matrix without an explicit fresh generation-1 planning gate."
        )
    version = _expect_int(  # shape-check: fatal:kind-or-version-mismatch
        payload["risk_test_matrix_contract_version"],
        context=f"{context}.risk_test_matrix_contract_version",
    )
    if version != RISK_TEST_MATRIX_CONTRACT_VERSION:
        raise AgentLoopError(f"Unsupported {context} risk_test_matrix_contract_version: {version}.")  # shape-check: fatal:kind-or-version-mismatch
    matrix = parse_risk_test_matrix(payload["risk_test_matrix"], context=f"{context}.risk_test_matrix")  # shape-check: fatal:kind-or-version-mismatch
    changes = parse_risk_test_matrix_changes(  # shape-check: fatal:no-conservative-reading
        payload["risk_test_matrix_changes"], context=f"{context}.risk_test_matrix_changes"
    )
    return version, matrix, changes

def _expect_test_observations(
    value: object, *, context: str
) -> tuple[TestObservationCitation, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: fatal:authentication-or-forgery
    result: list[TestObservationCitation] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        payload = _expect_object(item, context=item_context)  # shape-check: fatal:authentication-or-forgery
        _expect_exact_keys(  # shape-check: fatal:authentication-or-forgery
            payload,
            context=item_context,
            required={"command", "receipt_id", "claim"},
        )
        claim = _expect_non_empty_string(payload["claim"], context=f"{item_context}.claim")  # shape-check: fatal:authentication-or-forgery
        if claim not in {"current-result", "base-reproduction"}:
            raise AgentLoopError(  # shape-check: fatal:authentication-or-forgery
                f"{item_context}.claim must be `current-result` or `base-reproduction`."
            )
        result.append(
            TestObservationCitation(
                command=_expect_non_empty_string(payload["command"], context=f"{item_context}.command"),  # shape-check: fatal:authentication-or-forgery
                receipt_id=_expect_non_empty_string(payload["receipt_id"], context=f"{item_context}.receipt_id"),  # shape-check: fatal:authentication-or-forgery
                claim=claim,
            )
        )
    return tuple(result)


TEST_OBSERVATION_CITATION_KEYS = frozenset({"command", "receipt_id", "claim"})
TEST_OBSERVATION_CITATION_CLAIMS = frozenset({"current-result", "base-reproduction"})
CITATION_NOT_OBJECT_RULE = "citation is not a JSON object"
CITATION_KEYS_RULE = "citation keys are not exactly command, receipt_id and claim"
CITATION_BLANK_FIELD_RULE = "citation command or receipt_id is not a non-blank string"
CITATION_CLAIM_RULE = "citation claim is not current-result or base-reproduction"
TEST_OBSERVATIONS_NOT_ARRAY_RULE = "test_observations is not a JSON array; no citation is kept"
TEST_OBSERVATIONS_NOT_ARRAY_AFTER_REPAIR_RULE = (
    "test_observations is not a JSON array; format repair was attempted and did not "
    "produce one; no citation is kept"
)
TEST_OBSERVATIONS_NOT_ARRAY_RULES = frozenset(
    {TEST_OBSERVATIONS_NOT_ARRAY_RULE, TEST_OBSERVATIONS_NOT_ARRAY_AFTER_REPAIR_RULE}
)
TEST_OBSERVATIONS_DECODED_FROM_STRING_RULE = (
    "test_observations was a string holding a JSON array; the decoded array was used"
)
# How a fresh or historical parse treats a non-array test_observations (#1626).
# ``degrade`` is today's #927 drop and the default for every historical,
# resume and legacy re-parse; ``repair-first`` raises a repairable
# TestObservationsShapeError; ``degrade-after-repair`` is the fallback drop
# once repair failed, recorded with a rule saying so.
TEST_OBSERVATIONS_MODES = frozenset({"repair-first", "degrade", "degrade-after-repair"})


def _test_observation_item_defect(item: object) -> tuple[TestObservationCitation | None, str | None]:
    """Classify one model-authored citation without raising (#927).

    Returns the parsed citation, normalized exactly as the strict parser
    normalizes it, or the rule the item violates.
    """
    if not isinstance(item, dict):
        return None, CITATION_NOT_OBJECT_RULE
    if set(item) != TEST_OBSERVATION_CITATION_KEYS:
        return None, CITATION_KEYS_RULE
    command = item["command"]
    receipt_id = item["receipt_id"]
    claim = item["claim"]
    if not (isinstance(command, str) and command.strip()):
        return None, CITATION_BLANK_FIELD_RULE
    if not (isinstance(receipt_id, str) and receipt_id.strip()):
        return None, CITATION_BLANK_FIELD_RULE
    if not isinstance(claim, str) or claim.strip() not in TEST_OBSERVATION_CITATION_CLAIMS:
        return None, CITATION_CLAIM_RULE
    return TestObservationCitation(
        command=command.strip(), receipt_id=receipt_id.strip(), claim=claim.strip()
    ), None


def _citation_defect_observed(item: object, rule: str) -> str:
    """Observed text for a dropped citation: shape and key names, never values."""
    if rule == CITATION_KEYS_RULE and isinstance(item, dict):
        return _neutralize_identifier_like(
            _dropped_ref_preview("keys " + (", ".join(sorted(str(key) for key in item)) or "none"))  # shape-check: fatal:authentication-or-forgery
        )
    if rule == CITATION_CLAIM_RULE and isinstance(item, dict) and isinstance(item.get("claim"), str):
        return _neutralize_identifier_like(_dropped_ref_preview(item["claim"])) or "blank"  # shape-check: fatal:authentication-or-forgery
    return _json_type_label(item)


def _decode_test_observations_string(value: object, *, context: str) -> list | None:
    """Decode a test_observations string that holds a JSON array (#1626).

    The string is bounded like any dropped value before it is decoded.
    Returns the decoded list, or None when ``value`` is not a string or does
    not decode to a list.  Every parse mode and repair preservation share it,
    so strict, fallback and resume re-parses keep the same citations.
    """
    if not isinstance(value, str):
        return None
    _check_dropped_value_bound(value, context=context)  # shape-check: delegated
    try:
        decoded = json.loads(value)
    except ValueError:
        return None
    return decoded if isinstance(decoded, list) else None


def _degradable_test_observations(
    value: object, *, context: str, mode: str = "degrade"
) -> tuple[tuple[TestObservationCitation, ...], tuple[ParseDegradation, ...]]:
    """Parse model-authored follow-up citations, dropping malformed items (#927).

    Used only for coder follow-ups and issue implementations.  Canonical
    ``risk_test_matrix_evidence`` citations stay on the strict
    ``_expect_test_observations`` path.  Each dropped citation keeps its own
    ``citation-dropped`` record; more than ``CITATION_DEGRADATION_MAX_DROPS``
    drops reject the whole response, so every retained record is stored and
    rendered in full.  Every dropped raw value is bounded before its record
    is built.

    A string holding a JSON array is decoded first in every mode, with one
    ``citations-decoded-from-string`` record (#1626).  ``mode`` decides only
    what happens to a value that is still not an array: see
    ``TEST_OBSERVATIONS_MODES``.
    """
    if mode not in TEST_OBSERVATIONS_MODES:
        raise ValueError(f"Unknown test_observations mode: {mode}")
    leading: tuple[ParseDegradation, ...] = ()
    if isinstance(value, str):
        _check_dropped_value_bound(value, context=context)  # shape-check: fatal:payload-bound
        decoded = _decode_test_observations_string(value, context=context)  # shape-check: fatal:payload-bound
        if decoded is not None:
            value = decoded
            leading = (ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
                element_path=context,
                rule=TEST_OBSERVATIONS_DECODED_FROM_STRING_RULE,
                observed=_json_type_label(""),
                outcome=CITATIONS_DECODED_FROM_STRING_OUTCOME,
            ),)
    if not isinstance(value, list):
        # A non-list value counts as one drop at the field path.
        _check_dropped_value_bound(value, context=context)  # shape-check: fatal:payload-bound
        if mode == "repair-first":
            # Dropping the field would silently discard the turn's evidence
            # citations, so the structured repair pass gets it first.
            raise TestObservationsShapeError(  # shape-check: fatal:no-conservative-reading
                f"{context} must be a JSON array of citation objects, not "
                f"{_json_type_label(value)}.",
                field_path=context,
                type_label=_json_type_label(value),
                fallback_degradable=True,
            )
        return (), (ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
            element_path=context,
            rule=(
                TEST_OBSERVATIONS_NOT_ARRAY_AFTER_REPAIR_RULE
                if mode == "degrade-after-repair"
                else TEST_OBSERVATIONS_NOT_ARRAY_RULE
            ),
            observed=_json_type_label(value),
            outcome=CITATION_DROPPED_OUTCOME,
        ),)
    classified = [_test_observation_item_defect(item) for item in value]
    drop_count = sum(1 for _citation, rule in classified if rule is not None)
    if drop_count > CITATION_DEGRADATION_MAX_DROPS:
        message = (
            f"{context} has {drop_count} malformed citations, beyond the "
            f"{CITATION_DEGRADATION_MAX_DROPS}-citation drop bound."
        )
        if mode == "repair-first":
            # Repairable, but the terminal outcome stays fatal: no fallback.
            raise TestObservationsShapeError(  # shape-check: fatal:payload-bound
                message,
                field_path=context,
                type_label=_json_type_label(value),
                fallback_degradable=False,
            )
        raise AgentLoopError(message)  # shape-check: fatal:payload-bound
    citations: list[TestObservationCitation] = []
    records: list[ParseDegradation] = []
    for index, (citation, rule) in enumerate(classified):
        if rule is None:
            assert citation is not None
            citations.append(citation)
            continue
        item_context = f"{context}[{index}]"
        _check_dropped_value_bound(value[index], context=item_context)  # shape-check: fatal:payload-bound
        records.append(ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
            element_path=item_context,
            rule=rule,
            observed=_citation_defect_observed(value[index], rule),  # shape-check: fatal:authentication-or-forgery
            outcome=CITATION_DROPPED_OUTCOME,
        ))
    return tuple(citations), leading + tuple(records)


def _semantic_execution_ref(observation: object) -> str | None:
    if isinstance(observation, Mapping):
        value = observation.get("execution_ref")
    else:
        value = getattr(observation, "execution_ref", None)
    return value if isinstance(value, str) and value.strip() else None


def _hard_capped_claim_string(value: str, *, context: str) -> None:
    """Reject a claim string beyond the dropped-value hard cap: unbounded input."""
    if len(value.encode("utf-8")) > SEMANTIC_RISK_CLAIMS_MAX_DROPPED_REF_BYTES:
        raise AgentLoopError(  # shape-check: fatal:payload-bound
            f"{context} exceeds the {SEMANTIC_RISK_CLAIMS_MAX_DROPPED_REF_BYTES}-byte bound."
        )


def _json_type_label(value: object) -> str:
    """Name a JSON value's type alone, so a dropped value is never rendered."""
    if value is None:
        return "type null"
    if isinstance(value, bool):
        return "type boolean"
    if isinstance(value, (int, float)):
        return "type number"
    if isinstance(value, str):
        return "type string"
    if isinstance(value, list):
        return "type array"
    if isinstance(value, dict):
        return "type object"
    return f"type {type(value).__name__}"


def _optional_semantic_fact_string(value: object, *, context: str) -> tuple[str, str | None]:
    """Normalize an absent/null/blank semantic fact string to ``""``.

    A wrong type or a value over the field bound is a claim defect: the rule
    is returned and the caller drops the claim.  Only a value beyond the
    dropped-value hard cap still raises.
    """
    if value is None:
        return "", None
    if not isinstance(value, str):
        return "", CLAIM_FACT_MISTYPED_RULE
    # Raw length, before trimming: padding never hides unbounded input.
    _hard_capped_claim_string(value, context=context)  # shape-check: fatal:payload-bound
    normalized = value.strip()
    if not normalized:
        return "", None
    if len(normalized.encode("utf-8")) > RISK_MATRIX_MAX_FIELD_BYTES:
        return "", CLAIM_FACT_OVERSIZE_RULE
    return normalized, None


def _optional_semantic_fact_list(
    value: object,
    *,
    context: str,
    field: str | None = None,
    truncations: list[tuple[str, str]] | None = None,
    max_items: int = RISK_MATRIX_MAX_LIST_ITEMS,
    unique: bool = True,
) -> tuple[tuple[str, ...], str | None]:
    """Normalize an absent/null/empty semantic fact list to ``()``.

    A row that genuinely covers many tests is ordinary, so a list longer than
    ``max_items`` keeps its first entries and reports the overflow through
    ``truncations`` instead of rejecting the envelope before the PR is
    authenticated (#913).  The caller records the truncated field on the
    claim, so derivation can refuse to verify a row that lost content.

    A non-list value, non-string or blank items, duplicates (including in a
    discarded tail) and an item over the field bound are claim defects: the
    rule is returned and the caller drops the claim (#927).  Only an item
    beyond the dropped-value hard cap still raises.
    """
    if value is None:
        return (), None
    if not isinstance(value, list):
        return (), CLAIM_FACT_MISTYPED_RULE
    for index, item in enumerate(value):
        if isinstance(item, str):
            _hard_capped_claim_string(item, context=f"{context}[{index}]")  # shape-check: fatal:payload-bound
    if not all(isinstance(item, str) and item.strip() for item in value):
        return (), CLAIM_FACT_MISTYPED_RULE
    # Validate the complete list before truncating, so a malformed or
    # repeated entry anywhere -- including in the discarded tail -- stays a
    # defect rather than something truncation can hide.
    rendered = tuple(item.strip() for item in value)
    if unique and len(set(rendered)) != len(rendered):
        return (), CLAIM_FACT_MISTYPED_RULE
    if any(len(item.encode("utf-8")) > RISK_MATRIX_MAX_FIELD_BYTES for item in rendered):
        return (), CLAIM_FACT_OVERSIZE_RULE
    if len(rendered) > max_items:
        if truncations is not None:
            truncations.append((
                field or context,
                f"{context} listed {len(rendered)} items; the first "
                f"{max_items} are retained.",
            ))
        return rendered[:max_items], None
    return rendered, None


def _semantic_execution_ref_list(
    value: object, *, context: str
) -> tuple[tuple[tuple[int, str, str], ...], str | None]:
    """Validate ``execution_refs`` structure without the field bound.

    Returns the well-typed items as ``(index, raw, stripped)`` and the
    claim-scope rule, if any.  Refs over the 1,024-byte field bound must reach
    the parser's drop branch (#859), so only the explicit hard cap rejects; a
    list over the eight-ref bound is normalised by the caller (#1300).
    A non-list value, a non-string or blank item and an empty list are claim
    defects (#927); the well-typed items are still returned, so authority
    decisions on real broker handles run on a claim that is being dropped.
    """
    if not isinstance(value, list):
        return (), CLAIM_EXECUTION_REFS_MISTYPED_RULE
    rendered: list[tuple[int, str, str]] = []
    rule: str | None = None
    for index, item in enumerate(value):
        if isinstance(item, str):
            # The hard cap applies to the raw value, before trimming, so
            # whitespace padding can never smuggle unbounded input past it.
            _hard_capped_claim_string(item, context=f"{context}[{index}]")  # shape-check: fatal:payload-bound
        if not isinstance(item, str) or not item.strip():
            rule = CLAIM_EXECUTION_REFS_MISTYPED_RULE
            continue
        rendered.append((index, item, item.strip()))
    if rule is None and not rendered:
        rule = CLAIM_EXECUTION_REFS_EMPTY_RULE
    return tuple(rendered), rule


def _normalize_overlong_execution_refs(
    items: Sequence[tuple[int, str, str]],
) -> tuple[
    list[tuple[int, str, str]], list[tuple[int, str, str]], list[tuple[int, str, str]]
]:
    """De-duplicate by stripped value, then keep the first bound distinct items.

    Returns ``(kept, discarded_duplicates, discarded_overflow)`` (#1300).
    """
    distinct: list[tuple[int, str, str]] = []
    duplicates: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for item in items:
        if item[2] in seen:
            duplicates.append(item)
        else:
            seen.add(item[2])
            distinct.append(item)
    bound = SEMANTIC_RISK_CLAIMS_MAX_EXECUTION_REFS
    return distinct[:bound], duplicates, distinct[bound:]


def _truncate_utf8(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _dropped_ref_preview(ref: str) -> str:
    """Render one dropped ref as sanitized, single-line, bounded text."""
    flattened = " ".join(sanitize_historical_text(ref).replace("`", "'").split())  # shape-check: fatal:authentication-or-forgery
    if len(flattened) <= SEMANTIC_RISK_CLAIMS_DROPPED_REF_PREVIEW_CHARS:
        return flattened
    return flattened[:SEMANTIC_RISK_CLAIMS_DROPPED_REF_PREVIEW_CHARS] + "\u2026"


def _dropped_execution_refs_caveat(dropped_refs: Sequence[str]) -> str:
    """Name dropped refs in one caveat within the semantic field bound."""
    return _named_refs_caveat(  # shape-check: fatal:authentication-or-forgery
        "Dropped execution_refs that are not current-turn `agent-loop run-tests` "
        "selectors (they cannot verify this row): ",
        dropped_refs,
    )


def _overflow_execution_refs_caveat(
    duplicates: Sequence[str], overflow: Sequence[str]
) -> str:
    """Disclose over-bound normalisation of ``execution_refs`` (#1300)."""
    # Name the distinct over-bound selectors first, then the removed
    # duplicates; the shared helper bounds the text and adds ``(+N more)``.
    names = list(dict.fromkeys((*overflow, *duplicates)))
    counts = (
        f"{len(overflow)} over-bound selector(s) and "
        f"{len(duplicates)} duplicate occurrence(s) removed"
    )
    return _named_refs_caveat(  # shape-check: fatal:authentication-or-forgery
        f"Over-long execution_refs were trimmed to the first "
        f"{SEMANTIC_RISK_CLAIMS_MAX_EXECUTION_REFS} distinct selectors "
        f"({counts}; they cannot verify this row): ",
        names,
    )


def _named_refs_caveat(prefix: str, dropped_refs: Sequence[str]) -> str:
    """Name refs after ``prefix`` in one caveat within the semantic field bound."""
    budget = SEMANTIC_RISK_CLAIMS_MAX_FIELD_BYTES
    parts: list[str] = []
    for index, ref in enumerate(dropped_refs):
        candidate = prefix + ", ".join((*parts, f"`{_dropped_ref_preview(ref)}`")) + "."  # shape-check: fatal:authentication-or-forgery
        remaining = len(dropped_refs) - index - 1
        suffix = f" (+{remaining} more)" if remaining else ""
        if len((candidate + suffix).encode("utf-8")) > budget:
            omitted = len(dropped_refs) - index
            text = prefix + ", ".join(parts) + f"{', ' if parts else ''}(+{omitted} more)."
            return _truncate_utf8(text, budget)
        parts.append(f"`{_dropped_ref_preview(ref)}`")  # shape-check: fatal:authentication-or-forgery
    return _truncate_utf8(prefix + ", ".join(parts) + ".", budget)


def _truncated_fact_lists_caveat(entries: Sequence[str]) -> str:
    """Name every truncated fact list in one bounded caveat (#913)."""
    prefix = (
        "Truncated over-long semantic fact lists to the "
        f"{RISK_MATRIX_MAX_LIST_ITEMS}-item bound: "
    )
    return _truncate_utf8(prefix + " ".join(entries), SEMANTIC_RISK_CLAIMS_MAX_FIELD_BYTES)


def _dropped_value_bytes(value: object) -> int:
    """Size of a dropped raw value: UTF-8 length of its compact JSON form."""
    return len(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    )


def _check_dropped_value_bound(value: object, *, context: str) -> None:
    """Refuse to accept an unbounded raw value by dropping it (#927).

    Structured JSON extraction has no whole-response size cap, so every value
    a parser drops is measured before its record is built.  An over-bound
    value rejects the response even when its defect would otherwise degrade.
    """
    size = _dropped_value_bytes(value)
    if size > DROPPED_VALUE_MAX_BYTES:
        raise AgentLoopError(  # shape-check: fatal:payload-bound
            f"{context} is {size} bytes, beyond the {DROPPED_VALUE_MAX_BYTES}-byte bound "
            "for a dropped value."
        )


def _parse_semantic_risk_coverage_claims(
    value: object,
    *,
    context: str,
    expected_row_ids: Sequence[str] | None = None,
    execution_catalog: Sequence[object] | None = None,
) -> SemanticRiskCoverageClaims:
    """Parse the bounded, model-owned side of risk evidence.

    This parser deliberately has no receipt-ID or canonical-row fields.  When
    a live catalog is supplied, selectors are checked against that one
    invocation before the PR/head authentication phase begins.

    Missing, null, or empty semantic facts are accepted here and normalized
    to empty defaults: they are recoverable after authentication, where
    derivation records the row as unverified with an
    ``incomplete-semantic-claim`` diagnostic.

    A fact list longer than ``RISK_MATRIX_MAX_LIST_ITEMS`` is likewise a
    claim defect rather than an envelope defect (#913): the first entries are
    retained, the truncated fields are recorded on ``truncated_fact_fields``
    and named in one bounded caveat, and derivation records the row as
    unverified with a ``truncated-semantic-claim`` diagnostic.

    With a live catalog, selectors that cannot resolve to a current-turn
    broker handle are a claim defect, not an envelope defect (#859).  Command
    strings, cross-turn or restored handles, and refs between the 1,024-byte
    field bound and the 16,384-byte hard cap are dropped in order, recorded
    on ``dropped_execution_refs``, and named in one bounded caveat.  They
    carry no authority because they cannot resolve to a receipt, so the PR
    can still be authenticated and the row derives as unverified with an
    ``unknown-execution-ref`` diagnostic.

    A claim whose well-formed ``row_id`` is outside the approved enforceable
    set (for example a sibling phase's row) is dropped whole and its id is
    recorded on ``dropped_row_ids`` (#920).  The rest of the envelope is kept;
    derivation reports the dropped id as an ``unapproved-row-claim``
    diagnostic and the row's own coverage stays pending.  ``None`` alone
    means no approved set was supplied; an empty sequence is authoritative
    and admits no claim.

    Every other claim-scope defect drops that claim only (#926, #927), with
    one bounded ``ParseDegradation`` on ``degradations`` (observed values only
    through the sanitized preview; a non-string is recorded by type name
    alone).  Derivation reports it as a ``degraded-row-claim`` diagnostic and
    the approved row reads ``missing``.  The dropped defects are: an absent
    ``row_id`` key, a non-string ``row_id``, a string that fails the row-ID
    format (including finding- or requirement-like forms), and a row claimed
    more than once -- where every copy is dropped, so no arbitrary copy
    becomes the row's coverage; a claim element that is not an object; an
    absent, empty or ill-typed ``execution_refs`` (#855: a dropped claim
    needs no authority); an admissible selector repeated within one claim
    (#865: the same admissible selector may still be cited by several rows);
    unknown keys outside ``CLAIM_RESERVED_AUTHORITY_KEYS``; an ill-typed fact
    or one over the field bound; and, without a catalog, a selector over the
    field bound.  A ``risk_test_matrix_claims`` value that is not an array
    yields no claims and one record at the field path.  When one claim has
    several defects exactly one record is kept: a non-object is decided
    first, then the row-ID rules, then selectors, unknown keys and facts.

    Still fatal, because they forge or decide authority or are unbounded
    input, and always evaluated before any degradation so a degradable
    defect never masks them: a key in ``CLAIM_RESERVED_AUTHORITY_KEYS``;
    more than 24 claims or 16 caveats; any string beyond the
    16,384-byte hard cap; a dropped value whose compact JSON exceeds
    ``DROPPED_VALUE_MAX_BYTES``; catalog collisions, checked before the
    non-array degradation; and in-catalog selectors that are not passing
    parent-observed observations, or whose supplied launch-integrity state
    is failing or unknown (any of wrapper bootstrap, inner exec, or suite
    start not authoritative) -- these are real broker handles, so selecting
    one is an authority decision, not a format defect (#990).  The one
    exception (#1659) is a passing selector whose only launch defect is
    ``suite_start=unknown``: it is kept, and post-authentication derivation
    refuses it non-terminally with an ``inadmissible-execution`` diagnostic.  Without a
    catalog (historical and unit callers) selectors are kept verbatim.

    An ``execution_refs`` list with more than eight well-typed items is
    de-duplicated and trimmed to its first eight distinct selectors (#1300),
    even when the claim is also dropped for a mistyped item, so no selector
    past the bound reaches an authority check.  The discarded raw occurrences
    stay bounded by ``DROPPED_VALUE_MAX_BYTES``.
    """
    # ``None`` means no approved set was delivered (historical and unit
    # callers).  An explicitly empty set is a real scope -- a stage that owns
    # no enforceable rows -- so every claim is outside it (#920).
    allowed_rows = None if expected_row_ids is None else set(expected_row_ids)
    catalog_by_ref: dict[str, object] = {}
    if execution_catalog is not None:
        # The catalog is broker-supplied, so a collision is checked before
        # any field-scope degradation can return.
        for observation in execution_catalog:
            execution_ref = _semantic_execution_ref(observation)
            if execution_ref is None:
                continue
            if execution_ref in catalog_by_ref:
                raise NonRepairableEvidenceRejection(  # shape-check: fatal:orchestrator-authored
                    f"{context} cannot validate a colliding execution_ref `{execution_ref}`.",
                    reason="catalog-collision",
                )
            catalog_by_ref[execution_ref] = observation
    degradations: list[ParseDegradation] = []
    if not isinstance(value, list):
        # Field-scope defect: no claim survives, so every row reads missing.
        _check_dropped_value_bound(value, context=context)  # shape-check: fatal:payload-bound
        degradations.append(ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
            element_path=context,
            rule=CLAIMS_FIELD_NOT_ARRAY_RULE,
            observed=_json_type_label(value),
            outcome=CLAIM_DROPPED_OUTCOME,
        ))
        return SemanticRiskCoverageClaims((), degradations=tuple(degradations))
    if len(value) > SEMANTIC_RISK_CLAIMS_MAX_ROWS:
        raise AgentLoopError(  # shape-check: fatal:payload-bound
            f"{context} exceeds the {SEMANTIC_RISK_CLAIMS_MAX_ROWS}-row bound."
        )
    result: list[SemanticRiskCoverageClaim] = []
    dropped_row_ids: list[str] = []
    unapproved_claim_row_ids: list[tuple[ParseDegradation, str]] = []
    # Count admissible row IDs first, so a duplicated row drops every copy
    # rather than keeping whichever happened to come first (#926).
    row_counts: dict[str, int] = {}
    for raw_claim in value:
        candidate = raw_claim.get("row_id") if isinstance(raw_claim, dict) else None
        if not isinstance(candidate, str):
            continue
        # Same normalization as ``_expect_non_empty_string``.
        candidate = candidate.strip()
        if allowed_rows is None or candidate in allowed_rows:
            row_counts[candidate] = row_counts.get(candidate, 0) + 1
    for index, raw_claim in enumerate(value):
        claim_context = f"{context}[{index}]"
        if not isinstance(raw_claim, dict):
            # Object shape is decided first: no row-ID, selector or fact
            # logic runs on a non-object, and only its type is recorded.
            _check_dropped_value_bound(raw_claim, context=claim_context)  # shape-check: fatal:payload-bound
            degradations.append(ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
                element_path=claim_context,
                rule=CLAIM_NOT_OBJECT_RULE,
                observed=_json_type_label(raw_claim),
                outcome=CLAIM_DROPPED_OUTCOME,
            ))
            continue
        payload = raw_claim
        # Fatal checks first, on every value that can be evaluated, so no
        # degradable defect on the same claim can mask one.
        reserved = sorted(key for key in payload if key in CLAIM_RESERVED_AUTHORITY_KEYS)
        if reserved:
            raise AgentLoopError(  # shape-check: fatal:authentication-or-forgery
                f"{claim_context} names orchestrator-owned authority field(s): "
                + ", ".join(reserved)
                + "."
            )
        row_id_context = f"{claim_context}.row_id"
        row_id, row_defect = _claim_row_id_or_degradation(payload, context=row_id_context)  # shape-check: fatal:payload-bound
        refs_context = f"{claim_context}.execution_refs"
        if "execution_refs" in payload:
            ref_items, refs_rule = _semantic_execution_ref_list(  # shape-check: fatal:payload-bound
                payload["execution_refs"], context=refs_context
            )
        else:
            ref_items, refs_rule = (), CLAIM_EXECUTION_REFS_ABSENT_RULE
        overflow_duplicates: list[str] = []
        overflow_refs: list[str] = []
        if len(ref_items) > SEMANTIC_RISK_CLAIMS_MAX_EXECUTION_REFS:
            # Normalise whatever ``refs_rule`` says, so no selector past the
            # bound ever reaches an authority check (#1300).
            kept_items, dup_items, over_items = _normalize_overlong_execution_refs(ref_items)
            # The discarded raw occurrences (padding included) are bounded
            # before any degradation decision can mask them.
            _check_dropped_value_bound(  # shape-check: fatal:payload-bound
                [raw for _i, raw, _s in (*dup_items, *over_items)],
                context=refs_context,
            )
            overflow_duplicates = [s for _i, _r, s in dup_items]
            overflow_refs = [s for _i, _r, s in over_items]
            _logger.warning(
                "%s listed %d execution_refs; kept %d distinct, discarded %d "
                "duplicate and %d over-bound selector(s).",
                refs_context,
                len(ref_items),
                len(kept_items),
                len(dup_items),
                len(over_items),
            )
            ref_items = tuple(kept_items)
        raw_refs = tuple(stripped for _i, _r, stripped in ref_items)
        caveats_value = payload.get("caveats", [])
        if (
            isinstance(caveats_value, list)
            and len(caveats_value) > SEMANTIC_RISK_CLAIMS_MAX_CAVEATS
        ):
            raise AgentLoopError(  # shape-check: fatal:payload-bound
                f"{claim_context}.caveats exceeds the {SEMANTIC_RISK_CLAIMS_MAX_CAVEATS}-item bound."
            )
        truncated_facts: list[tuple[str, str]] = []
        fact_lists: dict[str, tuple[str, ...]] = {}
        fact_defects: list[tuple[str, str, object]] = []
        for field_name in SEMANTIC_RISK_CLAIM_FACT_KEYS:
            field_context = f"{claim_context}.{field_name}"
            raw_fact = payload.get(field_name)
            if field_name == "workflow_path_claim":
                workflow_path_claim, fact_rule = _optional_semantic_fact_string(  # shape-check: fatal:payload-bound
                    raw_fact, context=field_context
                )
            else:
                fact_lists[field_name], fact_rule = _optional_semantic_fact_list(  # shape-check: fatal:payload-bound
                    raw_fact,
                    context=field_context,
                    field=field_name,
                    truncations=truncated_facts,
                )
            if fact_rule is not None:
                fact_defects.append((field_context, fact_rule, _fact_defect_observed(raw_fact, fact_rule)))
        caveats, caveats_rule = _optional_semantic_fact_list(  # shape-check: fatal:payload-bound
            caveats_value,
            context=f"{claim_context}.caveats",
            max_items=SEMANTIC_RISK_CLAIMS_MAX_CAVEATS,
            unique=False,
        ) if caveats_value is not None else ((), CLAIM_FACT_MISTYPED_RULE)
        if caveats_rule is not None:
            fact_defects.append((
                f"{claim_context}.caveats",
                caveats_rule,
                _fact_defect_observed(caveats_value, caveats_rule),
            ))
        admissible_refs: list[str] = []
        dropped_refs: list[str] = []
        repeated_ref: str | None = None
        oversize_ref: str | None = None
        # One passing run may cover several rows (#865), so selectors are
        # unique within a claim, not across claims.
        seen_refs: set[str] = set()
        for ref in raw_refs:
            if execution_catalog is None:
                if len(ref.encode("utf-8")) > SEMANTIC_RISK_CLAIMS_MAX_FIELD_BYTES:
                    # Without a catalog there is no admissibility decision
                    # that could leave the claim well-founded, so the whole
                    # claim is dropped rather than just the selector.
                    oversize_ref = oversize_ref or ref
                    continue
            elif (
                len(ref.encode("utf-8")) > SEMANTIC_RISK_CLAIMS_MAX_FIELD_BYTES
                or ref not in catalog_by_ref
            ):
                # Oversize refs are dropped before any catalog lookup.
                if ref not in dropped_refs:
                    dropped_refs.append(ref)
                continue
            if ref in seen_refs:
                repeated_ref = repeated_ref or ref
                continue
            seen_refs.add(ref)
            admissible_refs.append(ref)
            if execution_catalog is not None:
                observation = catalog_by_ref[ref]
                semantics, _rich = _observation_semantics(observation)
                # Authority decisions over real broker handles: repair cannot
                # satisfy them, so they must not route to it (#990).
                if semantics["outcome"] != "passed" or semantics["provenance"] != "parent-observed":
                    raise NonRepairableEvidenceRejection(  # shape-check: fatal:authority-decision
                        f"{refs_context} selector `{ref}` is not an admissible passing observation: "
                        f"{'; '.join(_observation_outcome_failure_reasons(observation))}; "
                        f"command: {_observation_command_label(observation)}; "
                        f"observed at {_observation_timestamp_label(observation)}.",
                        reason="non-passing-selector",
                    )
                if not _known_launch_integrity_passes(observation) and not _only_suite_start_unverified(observation):
                    launch_integrity_message = (
                        f"{refs_context} selector `{ref}` has known non-authoritative "
                        "launch-integrity state and cannot be selected before authentication: "
                        f"{'; '.join(_launch_integrity_failure_reasons(observation))}; "
                        f"command: {_observation_command_label(observation)}; "
                        f"observed at {_observation_timestamp_label(observation)}. "
                        "Cite a different observation whose launch was fully verified."
                    )
                    node_test_hint = _node_test_hint(observation, launch_integrity_message)  # shape-check: fatal:authentication-or-forgery
                    raise NonRepairableEvidenceRejection(  # shape-check: fatal:authority-decision
                        launch_integrity_message + node_test_hint,
                        reason="launch-integrity",
                    )
        # Then the first degradable defect wins, in a fixed order, so every
        # dropped claim yields exactly one record.  The #926 row-ID rules come
        # first so an unapproved row keeps its ``unapproved-row-claim`` pairing.
        defect: tuple[str, str, object] | None = row_defect
        unapproved_row_id: str | None = None
        if defect is None and allowed_rows is not None and row_id not in allowed_rows:
            # A claim for a row outside this turn's approved enforceable set
            # (for example a sibling phase's row) asserts nothing this turn
            # can own.  Drop just that claim (#920): the row is simply not
            # claimed, which is stricter than losing every other valid claim.
            assert row_id is not None
            unapproved_row_id = row_id
            # A valid row ID never matches a reserved namespace, so this is
            # the identity and keeps exact attribution (#1016); it stays as
            # defence in depth.
            defect = (row_id_context, CLAIM_ROW_ID_UNAPPROVED_RULE, _neutralize_identifier_like(row_id))
        elif defect is None and row_counts.get(row_id, 0) > 1:
            defect = (row_id_context, CLAIM_ROW_ID_DUPLICATE_RULE, row_id)
        if defect is None and refs_rule is not None:
            observed_refs = "absent" if refs_rule == CLAIM_EXECUTION_REFS_ABSENT_RULE else (
                "empty array" if refs_rule == CLAIM_EXECUTION_REFS_EMPTY_RULE
                else _json_type_label(payload["execution_refs"])
            )
            defect = (refs_context, refs_rule, observed_refs)
        if defect is None and oversize_ref is not None:
            defect = (
                refs_context,
                CLAIM_EXECUTION_REFS_NO_CATALOG_OVERSIZE_RULE,
                f"{len(oversize_ref.encode('utf-8'))} bytes",
            )
        if defect is None and repeated_ref is not None:
            defect = (
                refs_context,
                CLAIM_EXECUTION_REFS_REPEATED_RULE,
                _neutralize_identifier_like(_dropped_ref_preview(repeated_ref)),  # shape-check: fatal:authentication-or-forgery
            )
        unknown_keys = sorted(
            key for key in payload
            if key not in SEMANTIC_RISK_CLAIM_KEYS
            and key not in SEMANTIC_RISK_CLAIM_COVERAGE_KEYS
        )
        if defect is None and unknown_keys:
            defect = (
                claim_context,
                CLAIM_UNKNOWN_KEYS_RULE,
                _neutralize_identifier_like(_dropped_ref_preview(", ".join(unknown_keys))),  # shape-check: fatal:authentication-or-forgery
            )
        # A mistyped row ID is an arbitrary JSON value with no bound of its
        # own, so it keeps the whole-claim bound.
        row_id_only_defect = defect is not None and not (
            (row_defect is not None and row_defect[1] == CLAIM_ROW_ID_MISTYPED_RULE)
            or refs_rule is not None
            or oversize_ref is not None
            or repeated_ref is not None
            or unknown_keys
            or fact_defects
        )
        if defect is None and fact_defects:
            defect = fact_defects[0]
        if defect is not None:
            element_path, rule, observed = defect
            # The whole claim is the dropped value, so it is bounded before
            # its record is built -- unless its only defects are row-ID
            # rules.  Such a claim passed every per-field bound an accepted
            # claim must pass, so it is already bounded, and a less valid
            # claim must not be rejected more strongly than an accepted one
            # (#920, #926).
            if not row_id_only_defect:
                _check_dropped_value_bound(payload, context=claim_context)  # shape-check: fatal:payload-bound
            else:
                # The one part of such a claim no per-field bound covers is
                # the discarded tail of an over-long fact list (#913), so the
                # tails are bounded together before the record is built.
                _check_dropped_value_bound(  # shape-check: fatal:payload-bound
                    {
                        field: payload[field][RISK_MATRIX_MAX_LIST_ITEMS:]
                        for field, _message in truncated_facts
                    },
                    context=claim_context,
                )
            drop_record = ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
                element_path=element_path,
                rule=rule,
                observed=observed,
                outcome=CLAIM_DROPPED_OUTCOME,
            )
            degradations.append(drop_record)
            if unapproved_row_id is not None:
                if unapproved_row_id not in dropped_row_ids:
                    dropped_row_ids.append(unapproved_row_id)
                unapproved_claim_row_ids.append((drop_record, unapproved_row_id))
            continue
        bookkeeping: list[str] = []
        test_level: str | None = None
        if "test_level" in payload and payload["test_level"] is not None:
            raw_level = payload["test_level"]
            normalized_level = raw_level.strip().lower() if isinstance(raw_level, str) else None
            if normalized_level in SEMANTIC_RISK_TEST_LEVELS:
                test_level = normalized_level
            else:
                # Degrade, not reject: the claim stays, the level reads absent.
                bookkeeping.append(
                    "test_level was not one of unit, integration, or workflow "
                    f"({_json_type_label(raw_level)}); treated as undeclared."
                )
        if dropped_refs:
            bookkeeping.append(_dropped_execution_refs_caveat(dropped_refs))  # shape-check: fatal:authentication-or-forgery
        if overflow_duplicates or overflow_refs:
            bookkeeping.append(_overflow_execution_refs_caveat(overflow_duplicates, overflow_refs))  # shape-check: fatal:authentication-or-forgery
        if truncated_facts:
            bookkeeping.append(
                _truncated_fact_lists_caveat([message for _, message in truncated_facts])
            )
        if bookkeeping:
            # Reserve room for every bookkeeping caveat at once, so neither one
            # overwrites the other and accepted content loss stays disclosed
            # (#913).  The row's own caveats keep the remaining slots.
            keep = max(SEMANTIC_RISK_CLAIMS_MAX_CAVEATS - len(bookkeeping), 0)
            caveats = (*caveats[:keep], *bookkeeping)
        assert row_id is not None
        claim = SemanticRiskCoverageClaim(
            row_id=row_id,
            execution_refs=tuple(admissible_refs),
            test_identifiers=fact_lists["test_identifiers"],
            test_locations=fact_lists["test_locations"],
            workflow_path_claim=workflow_path_claim,
            outcome_assertions=fact_lists["outcome_assertions"],
            forbidden_effect_assertions=fact_lists["forbidden_effect_assertions"],
            caveats=caveats,
            dropped_execution_refs=tuple(dropped_refs),
            overflow_execution_refs=tuple((*overflow_refs, *overflow_duplicates)),
            truncated_fact_fields=tuple(field for field, _ in truncated_facts),
            test_level=test_level,
        )
        result.append(claim)
    return SemanticRiskCoverageClaims(
        tuple(result),
        dropped_row_ids=tuple(dropped_row_ids),
        degradations=tuple(degradations),
        unapproved_claim_row_ids=tuple(unapproved_claim_row_ids),
    )


def _gap_degradations(drops: Sequence[tuple[str, str, object]]) -> tuple[ParseDegradation, ...]:
    return tuple(
        ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
            element_path=path, rule=rule, observed=observed, outcome=COVERAGE_GAP_DROPPED_OUTCOME,
        )
        for path, rule, observed in drops
    )


def _parse_semantic_risk_coverage_gaps(
    payload: Mapping[str, object],
    *,
    context: str,
    expected_row_ids: Sequence[str] | None = None,
) -> SemanticRiskCoverageGaps | None:
    """Parse ``risk_test_matrix_coverage_gaps`` with drop-not-reject semantics (#1290).

    Returns ``None`` when the field is absent so payloads stay unchanged.
    Malformed entries, unknown or non-enforceable row IDs, and duplicate row
    IDs are dropped with one bounded ``ParseDegradation`` each; the envelope is
    never rejected for a gap defect.  The array is bounded to
    ``SEMANTIC_RISK_CLAIMS_MAX_ROWS`` entries and each string to
    ``SEMANTIC_RISK_CLAIMS_MAX_FIELD_BYTES``; entries past the row bound are
    dropped with a record.
    """
    if "risk_test_matrix_coverage_gaps" not in payload:
        return None
    value = payload["risk_test_matrix_coverage_gaps"]
    drops: list[tuple[str, str, object]] = []
    if not isinstance(value, list):
        drops.append((context, "risk_test_matrix_coverage_gaps is not a JSON array; no gap is kept", _json_type_label(value)))
        return SemanticRiskCoverageGaps((), _gap_degradations(drops))  # shape-check: fatal:authentication-or-forgery
    allowed = None if expected_row_ids is None else set(expected_row_ids)
    counts: dict[str, int] = {}
    for raw in value[:SEMANTIC_RISK_CLAIMS_MAX_ROWS]:
        candidate = raw.get("row_id") if isinstance(raw, dict) else None
        if isinstance(candidate, str):
            counts[candidate.strip()] = counts.get(candidate.strip(), 0) + 1
    gaps: list[SemanticRiskCoverageGap] = []
    for index, raw in enumerate(value):
        path = f"{context}[{index}]"
        if index >= SEMANTIC_RISK_CLAIMS_MAX_ROWS:
            drops.append((path, f"gap exceeds the {SEMANTIC_RISK_CLAIMS_MAX_ROWS}-entry bound", "over-bound entry"))
            continue
        if not isinstance(raw, dict):
            drops.append((path, "gap is not a JSON object", _json_type_label(raw)))
            continue
        if set(raw) != set(SEMANTIC_RISK_COVERAGE_GAP_KEYS):
            drops.append((path, "gap must have exactly row_id, reason, and proposed_correction", _neutralize_identifier_like(_dropped_ref_preview(", ".join(sorted(str(k) for k in raw))))))  # shape-check: fatal:authentication-or-forgery
            continue
        fields: dict[str, str] = {}
        bad: str | None = None
        for key in SEMANTIC_RISK_COVERAGE_GAP_KEYS:
            item = raw[key]
            if not isinstance(item, str) or not item.strip():
                bad = f"{key} is not a non-blank string"
                break
            if len(item.encode("utf-8")) > SEMANTIC_RISK_CLAIMS_MAX_FIELD_BYTES:
                bad = f"{key} exceeds the {SEMANTIC_RISK_CLAIMS_MAX_FIELD_BYTES}-byte bound"
                break
            fields[key] = item.strip()
        if bad is not None:
            drops.append((path, f"gap {bad}", "invalid gap field"))
            continue
        row_id = fields["row_id"]
        if allowed is not None and row_id not in allowed:
            drops.append((f"{path}.row_id", "gap row_id is not an enforceable approved row", _neutralize_identifier_like(_dropped_ref_preview(row_id))))  # shape-check: fatal:authentication-or-forgery
            continue
        if counts.get(row_id, 0) > 1:
            drops.append((f"{path}.row_id", "gap row_id is declared more than once", _neutralize_identifier_like(_dropped_ref_preview(row_id))))  # shape-check: fatal:authentication-or-forgery
            continue
        gaps.append(SemanticRiskCoverageGap(
            row_id=row_id,
            reason=fields["reason"],
            proposed_correction=fields["proposed_correction"],
        ))
    return SemanticRiskCoverageGaps(tuple(gaps), _gap_degradations(drops))  # shape-check: fatal:authentication-or-forgery


def _fact_defect_observed(value: object, rule: str) -> str:
    """Observed text for a dropped fact: its type or its size, never its content."""
    if rule == CLAIM_FACT_OVERSIZE_RULE:
        return f"{_dropped_value_bytes(value)} bytes"
    return _json_type_label(value)


def _claim_row_id_or_degradation(
    payload: Mapping[str, object], *, context: str
) -> tuple[str | None, tuple[str, str, object] | None]:
    """Return a valid row ID, or the (path, rule, observed) defect (#926).

    The caller bounds the dropped claim before building its record.  Only a
    string beyond the dropped-value hard cap still raises: that is unbounded
    input rather than a defect of one claim.
    """
    if "row_id" not in payload:
        return None, (context, CLAIM_ROW_ID_ABSENT_RULE, "absent")
    value = payload["row_id"]
    if not isinstance(value, str):
        # Record the type alone, so no non-string value is ever rendered.
        return None, (
            context,
            CLAIM_ROW_ID_MISTYPED_RULE,
            f"type {'null' if value is None else type(value).__name__}",
        )
    if len(value.encode("utf-8")) > SEMANTIC_RISK_CLAIMS_MAX_DROPPED_REF_BYTES:
        raise AgentLoopError(  # shape-check: fatal:payload-bound
            f"{context} exceeds the {SEMANTIC_RISK_CLAIMS_MAX_DROPPED_REF_BYTES}-byte bound."
        )
    try:
        return _validate_risk_row_id(value, context=context), None  # shape-check: handled
    except AgentLoopError:
        return None, (
            context,
            CLAIM_ROW_ID_MALFORMED_RULE,
            _neutralize_identifier_like(_dropped_ref_preview(value)) or "blank",  # shape-check: fatal:authentication-or-forgery
        )


@dataclass(frozen=True)
class RiskTestMatrixMetadata:
    """The model-authored metadata portion of a matrix replacement."""

    applicability: str
    important_exclusions: tuple[str, ...]
    not_applicable_rationale: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "applicability": self.applicability,
            "important_exclusions": [
                sanitize_historical_text(item) for item in self.important_exclusions  # shape-check: fatal:authentication-or-forgery
            ],
        }
        if self.not_applicable_rationale is not None:
            payload["not_applicable_rationale"] = sanitize_historical_text(  # shape-check: fatal:authentication-or-forgery
                self.not_applicable_rationale
            )
        return payload


def _parse_complete_risk_test_matrix_row(value: object, *, context: str) -> RiskTestMatrixRow:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(payload, context=context, required=set(RISK_TEST_MATRIX_ROW_KEYS))  # shape-check: fatal:no-conservative-reading
    return RiskTestMatrixRow(
        row_id=_validate_risk_row_id(payload["row_id"], context=f"{context}.row_id"),  # shape-check: fatal:no-conservative-reading
        label=_risk_bounded_string(payload["label"], context=f"{context}.label"),  # shape-check: fatal:no-conservative-reading
        entry_path_or_mode=_risk_bounded_string(payload["entry_path_or_mode"], context=f"{context}.entry_path_or_mode"),  # shape-check: fatal:no-conservative-reading
        initial_state=_risk_bounded_string(payload["initial_state"], context=f"{context}.initial_state"),  # shape-check: fatal:no-conservative-reading
        event=_risk_bounded_string(payload["event"], context=f"{context}.event"),  # shape-check: fatal:no-conservative-reading
        expected_outcome=_risk_bounded_string(payload["expected_outcome"], context=f"{context}.expected_outcome"),  # shape-check: fatal:no-conservative-reading
        forbidden_side_effects=_risk_bounded_string_list(  # shape-check: fatal:no-conservative-reading
            payload["forbidden_side_effects"], context=f"{context}.forbidden_side_effects"
        ),
        proposed_test_level=_risk_bounded_string(payload["proposed_test_level"], context=f"{context}.proposed_test_level"),  # shape-check: fatal:no-conservative-reading
        proposed_test_location=_risk_bounded_string(payload["proposed_test_location"], context=f"{context}.proposed_test_location"),  # shape-check: fatal:no-conservative-reading
        applicability=_risk_bounded_string(payload["applicability"], context=f"{context}.applicability", max_bytes=64),  # shape-check: fatal:no-conservative-reading
        related_scope_item_ids=_risk_bounded_string_list(  # shape-check: fatal:no-conservative-reading
            payload["related_scope_item_ids"], context=f"{context}.related_scope_item_ids"
        ),
        execution_owner=_validate_risk_owner(payload["execution_owner"], context=f"{context}.execution_owner"),  # shape-check: fatal:no-conservative-reading
    )


def parse_risk_test_matrix_row(value: object, *, context: str = "risk_test_matrix_row") -> RiskTestMatrixRow:
    """Parse one complete row for semantic matrix operations."""
    row = _parse_complete_risk_test_matrix_row(value, context=context)  # shape-check: fatal:no-conservative-reading
    if row.applicability not in {"applicable", "required", "not-applicable"}:
        raise AgentLoopError(f"{context}.applicability is invalid.")  # shape-check: fatal:no-conservative-reading
    return row

def _parse_matrix_metadata(value: object, *, context: str) -> RiskTestMatrixMetadata:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload,
        context=context,
        required={"applicability", "important_exclusions"},
        optional={"not_applicable_rationale"},
    )
    applicability = _risk_bounded_string(payload["applicability"], context=f"{context}.applicability", max_bytes=64)  # shape-check: fatal:no-conservative-reading
    if applicability not in RISK_MATRIX_APPLICABILITY:
        raise AgentLoopError(f"{context}.applicability must be `applicable` or `not-applicable`.")  # shape-check: fatal:no-conservative-reading
    rationale_value = payload.get("not_applicable_rationale")
    rationale = None if rationale_value is None else _risk_bounded_string(  # shape-check: fatal:no-conservative-reading
        rationale_value, context=f"{context}.not_applicable_rationale", max_bytes=2_048
    )
    if applicability == "not-applicable" and not rationale:
        raise AgentLoopError(f"{context}.not_applicable_rationale must be non-empty for a not-applicable matrix.")  # shape-check: fatal:no-conservative-reading
    return RiskTestMatrixMetadata(
        applicability=applicability,
        important_exclusions=_risk_bounded_string_list(  # shape-check: fatal:no-conservative-reading
            payload["important_exclusions"],
            context=f"{context}.important_exclusions",
            max_items=RISK_MATRIX_MAX_EXCLUSIONS,
        ),
        not_applicable_rationale=rationale,
    )
