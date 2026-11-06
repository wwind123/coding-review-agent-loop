"""Parsing for agent response markers."""

from __future__ import annotations

import json
import hashlib
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
import dataclasses
from dataclasses import dataclass, field

from .post_merge_obligations import ObligationRequest
from .errors import (
    AgentLoopError,
    IssueImplementationConflictError,
    MissingJudgementFieldError,
)
from .protocol_discuss_types import (
    ParsedDiscussAnswer,
    ParsedFailedDiscussResponse,
    ParsedDiscussReview,
    ParsedDiscussResponse,
    DISCUSS_FAILED_OUTCOME,
)
from .protocol_markers import sanitize_historical_text
from .review_scheduling import normalize_fix_scope

from .protocol_common import (
    ANY_HEADING_RE,
    BULLET_RE,
    HTML_COMMENT_RE,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
    HUMAN_REQUIREMENTS_HEADING_RE,
    HUMAN_REQUIREMENTS_RESOLVED_RE,
    HUMAN_REQUIREMENT_STABLE_ID_RE,
    PARSE_DEGRADATION_RENDER_LIMIT,
    PLAN_STATE_RE,
    ParseDegradation,
    SIGNATURE_RE,
    TestObservationCitation,
    _bounded_single_line,
    _consume_structured_footer_and_signature,
    _expect_bool,
    _expect_exact_keys,
    _expect_int,
    _expect_non_empty_string,
    _expect_object,
    _expect_string_list,
    _extract_json_object_prefix,
    _flat_object_defect,
    _followup_heading_re,
    _logger,
    _normalize_requirement_label,
    _render_flat_object,
    _require_supported_schema_version,
    normalize_response_file_structured_text,
    parse_human_requirements_acknowledgement,
)


# Risk dependencies used by the response-composition adapters.
from .protocol_risk import (
    PostAuthClaimDiagnostic,
    RISK_MATRIX_MAX_CHANGES,
    RiskTestMatrix,
    RiskTestMatrixChange,
    RiskTestMatrixEvidence,
    RiskTestMatrixMetadata,
    RiskTestMatrixRow,
    SemanticRiskCoverageClaims,
    SemanticRiskCoverageGaps,
    _degradable_test_observations,
    _parse_matrix_metadata,
    _parse_risk_test_matrix_contract_fields,
    _parse_semantic_risk_coverage_claims,
    _parse_semantic_risk_coverage_gaps,
    _risk_bounded_string,
    _validate_risk_row_id,
    parse_risk_test_matrix_evidence,
    parse_risk_test_matrix_row,
)

HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK = (
    "checked the relevant GitHub discussion directly before responding"
)

STATE_RE = re.compile(r"<!--\s*AGENT_STATE:\s*(approved|blocking)\s*-->", re.I)
_PR_MARKER_VALUE_RE = re.compile(r"(?m)^\s*<!--\s*AGENT_PR:\s*(.*?)\s*-->\s*$", re.I)
GH_PR_URL_RE = re.compile(r"/pull/(\d+)(?:\b|$)")
CLARIFY_RE = re.compile(r"<!--\s*AGENT_CLARIFY\s*-->", re.I)
AGENT_UNAVAILABLE_RE = re.compile(r"<!--\s*AGENT_UNAVAILABLE\s*-->", re.I)
# Standalone variants: marker must occupy its own line.
_STANDALONE_CLARIFY_RE = re.compile(r"(?m)^\s*<!--\s*AGENT_CLARIFY\s*-->\s*$", re.I)
_STANDALONE_AGENT_UNAVAILABLE_RE = re.compile(
    r"(?m)^\s*<!--\s*AGENT_UNAVAILABLE\s*-->\s*$", re.I
)
_STANDALONE_STATE_RE = re.compile(r"(?m)^\s*<!--\s*AGENT_STATE:\s*(approved|blocking)\s*-->\s*$", re.I)
_STANDALONE_PLAN_STATE_RE = re.compile(
    r"(?m)^\s*<!--\s*AGENT_PLAN_STATE:\s*(approved|blocking)\s*-->\s*$", re.I
)
_STANDALONE_PR_RE = re.compile(r"(?m)^\s*<!--\s*AGENT_PR:\s*(\d+)\s*-->\s*$", re.I)
# Matches the opening or closing line of a fenced code block (``` or ~~~, 3+ chars).
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})", re.M)
HUMAN_REVIEWER_SIGNATURE_RE = re.compile(r"^\s*--\s*Human Reviewer\s*$", re.I | re.M)
HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK_RE = re.compile(
    re.escape(HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK),
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
PRIOR_UNRESOLVED_PLAN_ITEM_DISPOSITIONS_HEADING_RE = _followup_heading_re(
    r"prior unresolved plan item dispositions"
)
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
    # Optional reviewer-authored exact paths used only by selective PR
    # scheduling.  None means the finding is not classifiable as narrow.
    fix_scope: tuple[str, ...] | None = None
    # Optional ordered statements of a conjunctive finding (#958).  Empty for
    # every single-obligation finding.
    sub_items: tuple[str, ...] = ()
    # Optional approved risk-matrix row IDs for a finding whose sole demand is
    # admissible citation evidence for those rows (#1324).  Reviewer-declared;
    # only ``blocking_items`` accept it, and the orchestrator verifies it.
    evidence_row_ids: tuple[str, ...] = ()
    # Optional reviewer self-report on a plan review finding (#1505): whether
    # the defect was introduced by a change since the reviewer's previous plan
    # review or was already present.  Advisory telemetry only; never part of
    # the finding text and never an input to classification or approval.
    provenance: str | None = None

# Allowed values of the optional plan review finding ``provenance`` key (#1505).
PROVENANCE_REVISION_INTRODUCED = "revision_introduced"
PROVENANCE_PRE_EXISTING = "pre_existing"
PLAN_REVIEW_PROVENANCE_VALUES = frozenset({PROVENANCE_REVISION_INTRODUCED, PROVENANCE_PRE_EXISTING})
PLAN_REVIEW_FINDING_PROVENANCE_FIELD = "provenance"

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
    sub_item_dispositions: tuple[tuple[str, str], ...] = ()
    evidence_requirement_ids: tuple[str, ...] = ()

# Machine-owned obligations are deliberately distinct from reviewer findings.
# These values are persisted in round metadata, so changing one is a protocol
# migration rather than a display-only change.
MACHINE_AUTHORITY = "machine"
UNKNOWN_MACHINE_AUTHORITY = "unknown"
MACHINE_OBLIGATION_KINDS = frozenset(
    {
        "managed-exact-head-ci",
        "github-pr-checks",
        "alembic-migration",
        "merge-conflict",
        "human-requirements-acknowledgement",
        "human-exact-head-evidence",
        "unknown",
    }
)
EVIDENCE_OBLIGATION_KIND = "human-exact-head-evidence"
EVIDENCE_LIFECYCLE_STATES = frozenset({"evidence_deferred", "evidence_frozen"})
MACHINE_LIFECYCLE_STATES = frozenset(
    {
        "repair_required",
        "awaiting_current_head_review",
        "qualification_ready",
        "qualifying",
        "cleared",
        *EVIDENCE_LIFECYCLE_STATES,
    }
)
MACHINE_OBLIGATION_FIELDS = frozenset(
    {
        "authority",
        "obligation_kind",
        "lifecycle",
        "failed_head_sha",
        "candidate_head_sha",
        "obligation_identity",
    }
)
CI_MACHINE_OBLIGATION_KINDS = frozenset(
    {"managed-exact-head-ci", "github-pr-checks"}
)

SUB_ITEM_MIN_COUNT = 2
SUB_ITEM_MAX_COUNT = 12
SUB_ITEM_STATUSES = frozenset({"open", "resolved"})
SUB_ITEM_DISPOSITIONS = frozenset({"resolved", "unresolved"})

@dataclass(frozen=True)
class ReviewSubItem:
    """One separately checkable obligation of a conjunctive finding (#958)."""

    sub_item_id: str
    text: str
    status: str = "open"
    resolved_round: int | None = None

@dataclass(frozen=True)
class UnresolvedReviewItem:
    item_id: str
    reviewer: str
    source_round: int
    text: str
    status: str
    source_status: str | None = None
    notes: tuple[str, ...] = ()
    # Reviewer-provided exact paths, if present. None is deliberately distinct
    # from an empty tuple: absent/invalid scope must force conservative review.
    fix_scope: tuple[str, ...] | None = None
    # Canonical ownership state used by selective PR reconciliation. Legacy
    # records leave these empty and resolve to the source reviewer at runtime.
    resolution_owners: tuple[str, ...] = ()
    owner_states: tuple[tuple[str, str], ...] = ()
    owner_evidence: tuple[tuple[str, str], ...] = ()
    # Durable disposition outcome for each owner.  This is separate from
    # ``owner_states`` because a cleared owner may have cleared via
    # ``resolved`` or ``future``; the latter must survive until all owners
    # have cleared so a later round can retain the future reclassification.
    owner_dispositions: tuple[tuple[str, str], ...] = ()
    # Authority-aware machine-obligation fields. ``None`` retains the legacy
    # reviewer-finding representation; recovery promotes only trusted
    # orchestrator records to a machine authority. Unknown/invalid machine
    # records are represented as the non-bypassable ``unknown`` kind.
    authority: str | None = None
    obligation_kind: str | None = None
    lifecycle: str | None = None
    failed_head_sha: str | None = None
    candidate_head_sha: str | None = None
    obligation_identity: str | None = None
    alias_item_ids: tuple[str, ...] = ()
    alias_obligation_identities: tuple[str, ...] = ()
    # Reviewer-declared sub-items; never carried by machine obligations.
    sub_items: tuple[ReviewSubItem, ...] = ()
    # Evidence-only tag fixed when the orchestrator mints the item (#1324).
    # Reviewer findings only; legacy and untagged items carry ().
    evidence_row_ids: tuple[str, ...] = ()
    # Orchestrator-written subset of ``evidence_row_ids`` whose rows were
    # converted to the operator evidence freeze (#1325).  The residual demand
    # is ``evidence_row_ids`` minus these; reviewer findings only.
    converted_row_ids: tuple[str, ...] = ()
    # Row-converted evidence obligations (#1325) name the approved-matrix row
    # and matrix identity they stand for.  Both or neither; evidence only.
    converted_row_id: str | None = None
    converted_matrix_identity: str | None = None
    provenance: str | None = None
    repeat_of: str | None = None
    origin_scope: str | None = None
    screenshot_request: dict | None = None

    def __post_init__(self) -> None:
        from .review_images_coverage import validate_item
        validate_item(self)
        if self.evidence_row_ids and (
            self.authority is not None
            or self.obligation_kind is not None
            or self.lifecycle is not None
        ):
            raise ValueError("machine obligations cannot carry evidence_row_ids")
        if self.converted_row_ids and not set(self.converted_row_ids) <= set(self.evidence_row_ids):
            raise ValueError("converted_row_ids must be a subset of evidence_row_ids")
        if len(set(self.converted_row_ids)) != len(self.converted_row_ids):
            raise ValueError("converted_row_ids must be unique")
        if (self.converted_row_id is None) != (self.converted_matrix_identity is None):
            raise ValueError("converted row obligations need both a row ID and a matrix identity")
        if self.converted_row_id is not None and self.obligation_kind != EVIDENCE_OBLIGATION_KIND:
            raise ValueError("only evidence obligations may name a converted row")
        if self.status == "converted" and (
            not self.converted_row_ids or set(self.converted_row_ids) != set(self.evidence_row_ids)
        ):
            raise ValueError("a converted finding needs every tagged row converted")
        if self.sub_items:
            ids = [sub.sub_item_id for sub in self.sub_items]
            if len(set(ids)) != len(ids):
                raise ValueError("sub-item IDs must be unique")
            if any(sub.status not in SUB_ITEM_STATUSES for sub in self.sub_items):
                raise ValueError("sub-item status must be open or resolved")
        authority = self.authority
        kind = self.obligation_kind
        lifecycle = self.lifecycle
        # Any populated machine field makes this a machine record.  Do not
        # let an invalid authority, a partial lifecycle/head tuple, or a
        # machine identity without a kind fall back to an ordinary finding.
        machine = any(
            value is not None
            for value in (
                authority,
                kind,
                lifecycle,
                self.failed_head_sha,
                self.candidate_head_sha,
                self.obligation_identity,
            )
        )
        if not machine:
            return
        if authority not in {MACHINE_AUTHORITY, UNKNOWN_MACHINE_AUTHORITY}:
            raise ValueError("machine obligations require a known authority value")
        if kind not in MACHINE_OBLIGATION_KINDS:
            raise ValueError("machine obligations require a known obligation kind")
        if lifecycle not in MACHINE_LIFECYCLE_STATES:
            raise ValueError("machine obligations require a known lifecycle")
        if kind == EVIDENCE_OBLIGATION_KIND:
            if lifecycle not in EVIDENCE_LIFECYCLE_STATES | {"cleared"}:
                raise ValueError("evidence obligations require an evidence lifecycle")
            if lifecycle == "evidence_frozen" and not self.candidate_head_sha:
                raise ValueError("a frozen evidence obligation needs its frozen head")
            if not self.obligation_identity:
                raise ValueError("evidence obligations need a stable identity")
        elif lifecycle in EVIDENCE_LIFECYCLE_STATES:
            raise ValueError("evidence lifecycles are valid only for evidence obligations")
        if (
            lifecycle == "repair_required"
            and not self.failed_head_sha
            and kind in CI_MACHINE_OBLIGATION_KINDS
        ):
            raise ValueError("repair-required machine obligations need a failed head")
        if self.candidate_head_sha and self.failed_head_sha == self.candidate_head_sha:
            raise ValueError("a qualification candidate must differ from the failed head")
        if self.sub_items:
            raise ValueError("machine obligations cannot carry sub-items")

    @property
    def is_machine_obligation(self) -> bool:
        return self.authority in {MACHINE_AUTHORITY, UNKNOWN_MACHINE_AUTHORITY} or self.obligation_kind is not None

    @property
    def residual_evidence_row_ids(self) -> tuple[str, ...]:
        """Tagged rows that still demand evidence (#1325): tags minus converted rows."""
        converted = set(self.converted_row_ids)
        return tuple(row_id for row_id in self.evidence_row_ids if row_id not in converted)

    @property
    def is_inert_ledger_record(self) -> bool:
        """A retained record no reviewer or coder acts on (#1325).

        A converted finding and a cleared row-converted evidence tombstone stay
        in the ledger only for provenance and head-change reopening.
        """
        if self.status == "converted":
            return True
        return (
            self.converted_row_id is not None
            and self.obligation_kind == EVIDENCE_OBLIGATION_KIND
            and self.lifecycle == "cleared"
        )

    @property
    def authority_kind(self) -> str | None:
        """Compatibility alias for authority-aware callers."""
        return self.authority

    @property
    def machine_authority(self) -> str | None:
        return self.authority

    @property
    def machine_obligation_kind(self) -> str | None:
        return self.obligation_kind


@dataclass(frozen=True)
class ParsedReview:
    state: str
    summary: str
    blocking_items: tuple[ApprovedFollowup, ...]
    followups: ApprovedFollowups
    dispositions: tuple[ReviewItemDisposition, ...]
    raw_dispositions_text: str = ""
    architecture_impact: ArchitectureImpact | None = None
    architecture_impact_degradations: tuple[ParseDegradation, ...] = ()
    screenshot_requests: tuple[dict, ...] = ()
    screenshot_coverage_acceptances: tuple[dict, ...] = ()
    exact_head_evidence_requests: tuple[str, ...] = ()
    sub_item_degradations: tuple["ParseDegradation", ...] = ()
    post_merge_obligations: tuple[ObligationRequest, ...] = ()


@dataclass(frozen=True)
class AgentUnavailable:
    """An agent-declared inability to complete the assigned operation."""

    schema_version: int
    kind: str
    retryable: bool
    category: str
    summary: str
    suggested_action: str


AGENT_UNAVAILABLE_CATEGORIES = frozenset(
    {"environment", "permissions", "provider", "tooling", "unknown"}
)


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
    human_requirement_dispositions: tuple["HumanRequirementDisposition", ...] = ()
    architecture_impact: ArchitectureImpact | None = None
    architecture_impact_degradations: tuple[ParseDegradation, ...] = ()
    # Finding provenance tags that degraded to untagged (#1505).
    provenance_degradations: tuple[ParseDegradation, ...] = ()


@dataclass(frozen=True)
class StructuredHumanRequirementsPayload:
    addressed_ids: tuple[str, ...]
    checked_discussion_directly: bool


@dataclass(frozen=True)
class HumanRequirementDisposition:
    requirement_id: str
    disposition: str
    evidence: str


@dataclass(frozen=True)
class ArchitectureImpact:
    """Caller-owned architecture-impact assessment, separate from approval."""

    status: str
    rationale: str
    affected_components: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    execution_data_flows: tuple[str, ...] = ()
    execution_flows: tuple[str, ...] = ()
    data_flows: tuple[str, ...] = ()
    persistence: tuple[str, ...] = ()
    public_contracts: tuple[str, ...] = ()
    security_boundaries: tuple[str, ...] = ()
    canonical_document_action: str = "no-change"
    canonical_document_path: str | None = None
    canonical_document_rationale: str = ""
    uncertainty: tuple[str, ...] = ()


_ARCHITECTURE_CHANGED_REQUIRED_KEYS = frozenset({
    "affected_components", "dependencies", "execution_data_flows",
    "persistence", "public_contracts", "security_boundaries",
    "canonical_document_action", "canonical_document_path",
    "canonical_document_rationale",
})

# Deterministic near-miss vocabulary for the closed ``status`` enum (#916).
# Reviewers repeatedly wrote ``modified`` for ``changed``; rejecting the whole
# round for one off-vocabulary word discarded otherwise valid reviewed work.
# Keys are compared after lowercasing and folding ``_``/spaces to ``-``.  Any
# value outside this explicit table still fails closed.
_ARCHITECTURE_STATUS_SYNONYMS: Mapping[str, str] = {
    "changed": "changed",
    "change": "changed",
    "changes": "changed",
    "modified": "changed",
    "modifies": "changed",
    "modify": "changed",
    "updated": "changed",
    "unchanged": "unchanged",
    "no-change": "unchanged",
    "no-changes": "unchanged",
    "not-changed": "unchanged",
    "unmodified": "unchanged",
    "none": "unchanged",
    "same": "unchanged",
    "no-impact": "unchanged",
}


def _normalize_architecture_status(
    status: str, payload: Mapping[str, object], *, context: str
) -> tuple[str, str | None]:
    """Map a near-miss status onto the closed enum, returning an audit note.

    A synonym for ``changed`` is accepted only when the payload is corroborated
    by the complete ``changed``-only required field set; otherwise the value is
    too ambiguous to guess and the original enum error is raised.
    """
    if status in {"changed", "unchanged"}:
        return status, None
    enum_error = AgentLoopError(f"{context}.status must be `changed` or `unchanged`.")
    key = architecture_status_near_miss_key(status)
    canonical = _ARCHITECTURE_STATUS_SYNONYMS.get(key)
    if canonical is None:
        raise enum_error  # shape-check: fatal:no-conservative-reading
    if canonical == "changed" and not _ARCHITECTURE_CHANGED_REQUIRED_KEYS <= set(payload):
        raise enum_error  # shape-check: fatal:no-conservative-reading
    shown = sanitize_historical_text(status.strip())[:40]  # shape-check: fatal:authentication-or-forgery
    note = (
        f"agent-loop normalized {context}.status from `{shown}` to "
        f"`{canonical}` (deterministic closed-enum synonym)."
    )
    return canonical, note


ARCHITECTURE_IMPACT_DECLARED_STATUSES = frozenset({"changed", "unchanged"})
# Parser-only status for a degraded assessment.  It has no wire form: the
# response parsers reject it like any unknown status, strict decoders never
# produce it, and approved-state writers refuse it.
ARCHITECTURE_IMPACT_UNDETERMINED = "undetermined"
# The only closed-enum near miss resolved deterministically (#916).  Every
# other unknown status keeps raising exactly as before.
ARCHITECTURE_IMPACT_STATUS_ALIASES = {"modified": "changed"}
ARCHITECTURE_IMPACT_NEAR_MISS_RULE = "architecture_impact.status-closed-enum-near-miss"


@dataclass(frozen=True)
class ArchitectureImpactContract:
    """Whether this turn required an assessment, and whether one was supplied.

    A required contract is satisfied only by a declared `changed`/`unchanged`
    status.  An omitted, normalization-removed, or `undetermined` assessment is
    an unsatisfied result that the orchestration seam refuses (#925).
    """

    required: bool = False
    satisfied: bool = False


def architecture_impact_contract_for(
    impact: ArchitectureImpact | None, *, required: bool
) -> ArchitectureImpactContract:
    return ArchitectureImpactContract(
        required=required,
        satisfied=impact is not None and impact.status in ARCHITECTURE_IMPACT_DECLARED_STATUSES,
    )


def _non_blank_string_entries(value: object) -> bool:
    """Positive evidence: at least one entry the payload parser would accept.

    A renderable flat object counts exactly when `_expect_string_list` would
    flatten it (#1330), so classification and the parse that follows agree.
    """
    return isinstance(value, list) and any(
        render_architecture_list_entry(item) is not None for item in value
    )


def _near_miss_predicate_failure(payload: Mapping[str, object]) -> str | None:
    """Return the first literal key that fails the positive-evidence predicate."""
    # Literal keys only: the flow aliases do not satisfy the `changed`
    # required-key check, so counting them here would reject the envelope.
    for key in (
        "affected_components", "dependencies", "execution_data_flows",
        "persistence", "public_contracts", "security_boundaries",
    ):
        if not _non_blank_string_entries(payload.get(key)):
            return key
    action = payload.get("canonical_document_action")
    if not isinstance(action, str) or not action.strip() or action.strip() == "no-change":
        return "canonical_document_action"
    path = payload.get("canonical_document_path")
    if not isinstance(path, str) or not path.strip():
        return "canonical_document_path"
    rationale = payload.get("canonical_document_rationale")
    if not (isinstance(rationale, str) and rationale.strip()):
        return "canonical_document_rationale"
    return None


def architecture_impact_near_miss_corroborated(payload: Mapping[str, object]) -> bool:
    """Positive-evidence predicate: the payload itself settles `changed`.

    Key presence is not evidence.  Every changed-only list must hold a
    non-blank entry and the canonical-document fields must describe a real
    document action; anything less leaves the status undetermined.  Passing
    implies every key the `changed` required-key check demands is present.
    """
    return _near_miss_predicate_failure(payload) is None


ARCHITECTURE_IMPACT_CLOSED_ENUM_RULE = "status-not-in-closed-enum"


def architecture_status_near_miss_key(status: str) -> str:
    """The legacy synonym-table key: trimmed, lower-cased, separators as `-`."""
    return re.sub(r"[\s_]+", "-", status.strip().lower())


def classify_architecture_status_near_miss(
    impact: Mapping[str, object], *, context: str
) -> tuple[str, ParseDegradation] | None:
    """Classify a closed-enum near miss, or return None for any other status.

    Exactly one alias is honored: a corroborated `modified` resolves to
    `changed`.  Every other spelling of the legacy synonym table, and an
    uncorroborated `modified`, resolves to the parser-only `undetermined` --
    never to `unchanged`.  Declared and unknown values are not near misses.
    """
    status_value = impact.get("status")
    if not isinstance(status_value, str) or status_value in ARCHITECTURE_IMPACT_DECLARED_STATUSES:
        return None
    # Look the table up exactly as the legacy normalization did, so a case or
    # separator variant (`Unchanged`, `Modified`, `no_change`) degrades one
    # field instead of rejecting the envelope.
    key = architecture_status_near_miss_key(status_value)
    if key in ARCHITECTURE_IMPACT_STATUS_ALIASES:
        failure = _near_miss_predicate_failure(impact)
        if failure is None:
            resolved = ARCHITECTURE_IMPACT_STATUS_ALIASES[key]
            rule = ARCHITECTURE_IMPACT_NEAR_MISS_RULE
            outcome = "normalized-to-changed"
        else:
            resolved = ARCHITECTURE_IMPACT_UNDETERMINED
            rule = f"{ARCHITECTURE_IMPACT_NEAR_MISS_RULE}; positive evidence missing: {failure}"
            outcome = "degraded-to-undetermined"
    elif key in _ARCHITECTURE_STATUS_SYNONYMS:
        # Includes case variants of the declared values: only the exact
        # spellings are declared, so `Unchanged` is never read as `unchanged`.
        resolved = ARCHITECTURE_IMPACT_UNDETERMINED
        rule = ARCHITECTURE_IMPACT_CLOSED_ENUM_RULE
        outcome = "degraded-to-undetermined"
    else:
        return None
    return resolved, ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
        element_path=f"{context}.status", rule=rule, observed=status_value, outcome=outcome,
    )


def _parse_architecture_impact_degradable(
    value: object, *, context: str
) -> tuple[ArchitectureImpact, ParseDegradation | None]:
    """Parse an agent-supplied assessment, degrading only the status near miss."""
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    near_miss = classify_architecture_status_near_miss(payload, context=context)  # shape-check: fatal:authentication-or-forgery
    if near_miss is None:
        return _parse_architecture_impact_payload(payload, context=context), None  # shape-check: fatal:no-conservative-reading
    resolved, record = near_miss
    return _parse_architecture_impact_payload(payload, context=context, status=resolved), record  # shape-check: fatal:no-conservative-reading


def parse_architecture_impact_degradable(
    value: object, *, context: str = "architecture_impact"
) -> tuple[ArchitectureImpact, ParseDegradation | None]:
    """Degradable parse for fresh agent responses outside protocol envelopes."""
    return _parse_architecture_impact_degradable(value, context=context)  # shape-check: fatal:no-conservative-reading


ARCHITECTURE_STATUS_MODES = frozenset({"strict", "legacy", "degradable"})


def _parse_architecture_impact(
    value: object, *, context: str, architecture_status_mode: str = "strict"
) -> ArchitectureImpact:
    """Strict by default; `legacy` keeps the #916 synonym decode for stored text."""
    if architecture_status_mode not in ARCHITECTURE_STATUS_MODES:
        raise AgentLoopError(f"Unknown architecture_status_mode {architecture_status_mode!r}.")  # shape-check: fatal:no-conservative-reading
    if architecture_status_mode == "degradable":
        raise AgentLoopError("Degradable parsing returns a record; use the response parsers.")  # shape-check: fatal:no-conservative-reading
    return _parse_architecture_impact_payload(  # shape-check: fatal:no-conservative-reading
        _expect_object(value, context=context),  # shape-check: fatal:no-conservative-reading
        context=context,
        legacy=architecture_status_mode == "legacy",
    )


def _status_is_sole_defect(payload: dict[str, object], *, context: str) -> bool:
    """True when the assessment parses once its status is made valid (#1185)."""
    probe = dict(payload)
    probe["status"] = "unchanged"
    try:
        _parse_architecture_impact_payload(probe, context=context, legacy=False)  # shape-check: handled
    except AgentLoopError:
        return False
    return True


def _missing_status_error(
    message: str, payload: dict[str, object], *, context: str
) -> AgentLoopError:
    """Typed only when the status is the sole defect; otherwise the plain error."""
    if not _status_is_sole_defect(payload, context=context):
        return AgentLoopError(message)  # shape-check: fatal:no-conservative-reading
    if "status" not in payload:
        preview = None
    else:
        observed = payload["status"]
        try:
            rendered = observed if isinstance(observed, str) else json.dumps(observed)
        except (TypeError, ValueError):
            rendered = repr(observed)
        preview = _bounded_single_line(rendered, 40)  # shape-check: fatal:authentication-or-forgery
    return MissingJudgementFieldError(  # shape-check: fatal:no-conservative-reading
        message,
        field_path=f"{context}.status",
        allowed_values=("changed", "unchanged"),
        observed_preview=preview,
    )


def _parse_architecture_impact_payload(
    payload: dict[str, object], *, context: str, status: str | None = None, legacy: bool = False
) -> ArchitectureImpact:
    if status is None and not legacy and "status" not in payload:
        required_missing = sorted({"status", "rationale"} - set(payload))
        if required_missing == ["status"]:
            error = _missing_status_error(  # shape-check: fatal:no-conservative-reading
                f"{context} is missing required field(s): status", payload, context=context
            )
            if isinstance(error, MissingJudgementFieldError):
                raise error  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload,
        context=context,
        required={"status", "rationale"},
        optional={
            "affected_components", "dependencies", "execution_data_flows",
            "execution_flows", "data_flows", "persistence", "public_contracts",
            "security_boundaries", "canonical_document_action",
            "canonical_document_path", "canonical_document_rationale", "uncertainty",
        },
    )
    normalization_note: str | None = None
    if status is None:
        if not legacy:
            observed_status = payload["status"]
            if not isinstance(observed_status, str) or not observed_status.strip():
                kind = "must be a string." if not isinstance(observed_status, str) else "must be a non-empty string."
                raise _missing_status_error(  # shape-check: fatal:no-conservative-reading
                    f"{context}.status {kind}", payload, context=context
                )
        raw_status = _expect_non_empty_string(payload["status"], context=f"{context}.status")  # shape-check: fatal:no-conservative-reading
        if legacy:
            # Compatibility decode of text stored or posted before #925 (#916).
            status, normalization_note = _normalize_architecture_status(  # shape-check: fatal:no-conservative-reading
                raw_status, payload, context=context
            )
        else:
            status = raw_status
            if status not in ARCHITECTURE_IMPACT_DECLARED_STATUSES:
                raise _missing_status_error(  # shape-check: fatal:no-conservative-reading
                    f"{context}.status must be `changed` or `unchanged`.", payload, context=context
                )
    action = _expect_non_empty_string(  # shape-check: fatal:no-conservative-reading
        payload.get("canonical_document_action", "no-change"),
        context=f"{context}.canonical_document_action",
    )
    rationale = _expect_non_empty_string(payload["rationale"], context=f"{context}.rationale")  # shape-check: fatal:no-conservative-reading
    if status == "changed":
        missing = sorted(_ARCHITECTURE_CHANGED_REQUIRED_KEYS - set(payload))
        if missing:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{context} changed assessments must include: {', '.join(missing)}."
            )
    path_value = payload.get("canonical_document_path")
    if path_value is not None and (not isinstance(path_value, str) or not path_value.strip()):
        raise AgentLoopError(f"{context}.canonical_document_path must be a non-empty string or null.")  # shape-check: fatal:no-conservative-reading
    combined = _expect_string_list(payload.get("execution_data_flows", []), context=f"{context}.execution_data_flows", item_context=context, flatten_object_items=True)  # shape-check: fatal:no-conservative-reading
    execution = _expect_string_list(payload.get("execution_flows", []), context=f"{context}.execution_flows", item_context=context, flatten_object_items=True)  # shape-check: fatal:no-conservative-reading
    data = _expect_string_list(payload.get("data_flows", []), context=f"{context}.data_flows", item_context=context, flatten_object_items=True)  # shape-check: fatal:no-conservative-reading
    if not combined and (execution or data):
        combined = (*execution, *data)
    return ArchitectureImpact(
        status=status,
        rationale=rationale,
        affected_components=_expect_string_list(payload.get("affected_components", []), context=f"{context}.affected_components", item_context=context, flatten_object_items=True),  # shape-check: fatal:no-conservative-reading
        dependencies=_expect_string_list(payload.get("dependencies", []), context=f"{context}.dependencies", item_context=context, flatten_object_items=True),  # shape-check: fatal:no-conservative-reading
        execution_data_flows=combined,
        execution_flows=execution,
        data_flows=data,
        persistence=_expect_string_list(payload.get("persistence", []), context=f"{context}.persistence", item_context=context, flatten_object_items=True),  # shape-check: fatal:no-conservative-reading
        public_contracts=_expect_string_list(payload.get("public_contracts", []), context=f"{context}.public_contracts", item_context=context, flatten_object_items=True),  # shape-check: fatal:no-conservative-reading
        security_boundaries=_expect_string_list(payload.get("security_boundaries", []), context=f"{context}.security_boundaries", item_context=context, flatten_object_items=True),  # shape-check: fatal:no-conservative-reading
        canonical_document_action=action,
        canonical_document_path=path_value,
        canonical_document_rationale=(
            _expect_non_empty_string(  # shape-check: fatal:no-conservative-reading
                payload["canonical_document_rationale"],
                context=f"{context}.canonical_document_rationale",
            )
            if payload.get("canonical_document_rationale") not in (None, "")
            else ""
        ),
        uncertainty=(
            *_expect_string_list(payload.get("uncertainty", []), context=f"{context}.uncertainty", item_context=context, flatten_object_items=True),  # shape-check: fatal:no-conservative-reading
            *((normalization_note,) if normalization_note else ()),
        ),
    )


def parse_architecture_impact(
    value: object,
    *,
    context: str = "architecture_impact",
    architecture_status_mode: str = "strict",
) -> ArchitectureImpact:
    """Validate an impact object for protocol extensions outside response envelopes."""
    return _parse_architecture_impact(  # shape-check: fatal:no-conservative-reading
        value, context=context, architecture_status_mode=architecture_status_mode
    )


def sanitize_architecture_impact(value: object | None) -> dict[str, object] | None:
    """Return a marker-safe JSON payload for an agent-supplied impact assessment.

    The parsed dataclass and the JSON sidecar both contain untrusted prose.  Keep
    the transport shape intact while neutralizing reserved protocol markers in
    every string before the value can reach durable metadata, handoffs, or
    host-request artifacts.
    """
    if value is None:
        return None
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if not isinstance(value, Mapping):
        raise AgentLoopError("architecture_impact must be a mapping or parsed impact object.")  # shape-check: fatal:orchestrator-authored

    def clean(item: object) -> object:
        if isinstance(item, str):
            return sanitize_historical_text(item)  # shape-check: fatal:authentication-or-forgery
        if isinstance(item, Mapping):
            return {str(key): clean(child) for key, child in item.items()}  # shape-check: fatal:orchestrator-authored
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]  # shape-check: fatal:orchestrator-authored
        return item

    sanitized = clean(value)  # shape-check: fatal:orchestrator-authored
    return sanitized if isinstance(sanitized, dict) else None


HUMAN_REQUIREMENT_DISPOSITION_VALUES = frozenset(
    {"addressed", "blocked", "not-applicable"}
)


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
    human_requirement_dispositions: tuple[HumanRequirementDisposition, ...] = ()
    test_observations: tuple[TestObservationCitation, ...] = ()
    architecture_impact: ArchitectureImpact | None = None
    # Fresh turns own only semantic coverage claims. The old canonical field is
    # retained below for historical response parsing and is ignored by the
    # deterministic builder.
    risk_test_matrix_claims: SemanticRiskCoverageClaims | None = None
    risk_test_matrix_evidence: "RiskTestMatrixEvidence | None" = None
    risk_test_matrix_diagnostics: tuple[PostAuthClaimDiagnostic, ...] = ()
    architecture_impact_contract: ArchitectureImpactContract = ArchitectureImpactContract()
    architecture_impact_degradations: tuple[ParseDegradation, ...] = ()
    # Parser-derived records of dropped malformed citations (#927).
    test_observation_degradations: tuple[ParseDegradation, ...] = ()
    # Orchestrator-derived, never parsed: reported runs outside the assigned
    # checkout (e.g. a clean-base baseline). Visible context, never evidence.
    out_of_checkout_tests_run: tuple[str, ...] = ()
    # Advisory sub-item claims (#958): never change persisted sub-item status.
    addressed_sub_items: tuple[str, ...] = ()
    sub_item_claim_degradations: tuple[ParseDegradation, ...] = ()
    # Coder-declared untestable rows (#1290); never persisted as authority.
    risk_test_matrix_coverage_gaps: SemanticRiskCoverageGaps | None = None
    # Orchestrator-derived, never parsed: addressed_items IDs that ledger
    # history proves were already resolved, dropped by validation (#1388).
    dropped_resolved_addressed_items: tuple[str, ...] = ()
    dropped_excluded_items: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructuredIssueImplementation:
    """The strict terminal result emitted by an issue implementation coder."""

    schema_version: int
    kind: str
    state: str
    summary: str
    pr_number: int | None
    human_requirements: StructuredHumanRequirementsPayload
    human_requirement_dispositions: tuple[HumanRequirementDisposition, ...]
    tests_run: tuple[str, ...] | None = None
    test_observations: tuple[TestObservationCitation, ...] = ()
    architecture_impact: ArchitectureImpact | None = None
    risk_test_matrix_claims: SemanticRiskCoverageClaims | None = None
    risk_test_matrix_evidence: "RiskTestMatrixEvidence | None" = None
    risk_test_matrix_diagnostics: tuple[PostAuthClaimDiagnostic, ...] = ()
    architecture_impact_contract: ArchitectureImpactContract = ArchitectureImpactContract()
    architecture_impact_degradations: tuple[ParseDegradation, ...] = ()
    # Parser-derived records of dropped malformed citations (#927).
    test_observation_degradations: tuple[ParseDegradation, ...] = ()
    # Orchestrator-derived, never parsed: see StructuredCoderFollowup.
    out_of_checkout_tests_run: tuple[str, ...] = ()
    # Coder-declared untestable rows (#1290); never persisted as authority.
    risk_test_matrix_coverage_gaps: SemanticRiskCoverageGaps | None = None


@dataclass(frozen=True)
class StructuredTaskResult:
    schema_version: int
    kind: str
    state: str
    outcome: str
    summary: str
    pr_number: int | None = None
    clarification: tuple[str, ...] = ()
    architecture_impact: ArchitectureImpact | None = None
    architecture_impact_contract: ArchitectureImpactContract = ArchitectureImpactContract()
    architecture_impact_degradations: tuple[ParseDegradation, ...] = ()


@dataclass(frozen=True)
class DeferredStage:
    """A stage the plan intentionally leaves out of the current scope (#476).

    Declared by the coder in structured plan responses so scope narrowing is a
    mechanical signal instead of prose the orchestrator would have to guess at.
    """

    title: str
    summary: str


@dataclass(frozen=True)
class ChildStage:
    """A bounded implementation stage that may be filed as a child issue."""

    title: str
    summary: str


@dataclass(frozen=True)
class TypedPlanStages:
    """The four non-overlapping scope categories used by approved plans (#585)."""

    child_stages: tuple[ChildStage, ...] = ()
    external_dependencies: tuple[DeferredStage, ...] = ()
    deferred_work: tuple[DeferredStage, ...] = ()
    plan_actions: tuple[DeferredStage, ...] = ()


# Generation-1 planning is intentionally a separate wire model.  The legacy
# ``ChildStage`` type above is still used for historical, unversioned plans and
# must not silently acquire fields that would alter decomposition semantics.
EXECUTION_STRATEGY_CONTRACT_VERSION = 1
EXECUTION_TOPOLOGY_SOURCE = "approved-plan-v1"
# Keys the tool stamps next to a recommendation after acceptance (the rendered
# ``topology_source`` line, the recommendation digest, and the contract
# version).  A planner reading rendered context may copy them back into a
# fresh or patched ``execution_recommendation`` object; they are dropped before
# validation and logged, because the tool recomputes every one of them (#1333).
# Any other unknown key is still rejected.  Add a newly stamped field here
# explicitly rather than accepting arbitrary extra keys.
TOOL_OWNED_EXECUTION_RECOMMENDATION_KEYS = frozenset({
    "topology_source",
    "recommendation_digest",
    "execution_strategy_contract_version",
})
EXECUTION_AUTOMATION_CLASSES = frozenset({"agent-pr", "human-action", "manual-close"})
# Per-child execution dispositions (#808).  The planner makes the semantic
# readiness decision, plan reviewers approve or block it, and the orchestrator
# validates only the typed fields, their consistency, and provenance.
EXECUTION_DISPOSITION_DIRECT = "direct-implementation"
EXECUTION_DISPOSITION_PLANNING = "requires-child-planning"
EXECUTION_DISPOSITION_HUMAN = "human-owned"
EXECUTION_DISPOSITION_VALUES = frozenset(
    {EXECUTION_DISPOSITION_DIRECT, EXECUTION_DISPOSITION_PLANNING, EXECUTION_DISPOSITION_HUMAN}
)
EXECUTION_DISPOSITIONS_FOR_AUTOMATION: dict[str, frozenset[str]] = {
    "agent-pr": frozenset({EXECUTION_DISPOSITION_DIRECT, EXECUTION_DISPOSITION_PLANNING}),
    "human-action": frozenset({EXECUTION_DISPOSITION_HUMAN}),
    "manual-close": frozenset({EXECUTION_DISPOSITION_HUMAN}),
}


def validate_direct_readiness(
    *,
    non_goals: Sequence[str],
    compatibility_constraints: Sequence[str],
    dependency_notes: str | None,
    rationale: str | None,
    unresolved_design_decisions: Sequence[str],
) -> tuple[str, ...]:
    """Return the direct-readiness invariants violated by reviewed stage fields.

    This is the single shared validator used by the planner-turn validator,
    the routing seam, and the signed-override path.  It judges only the
    reviewed stage/phase fields it is given; callers must never feed override
    metadata into it.  An empty result means the stage is direct-ready.
    """
    problems: list[str] = []
    if tuple(unresolved_design_decisions):
        problems.append("unresolved_design_decisions must be empty")
    if not any(str(item).strip() for item in non_goals):
        problems.append("non_goals must be non-empty")
    if not any(str(item).strip() for item in compatibility_constraints):
        problems.append("compatibility_constraints must be non-empty")
    if not isinstance(dependency_notes, str) or not dependency_notes.strip():
        problems.append("dependency_notes must be non-empty")
    if not isinstance(rationale, str) or not rationale.strip():
        problems.append("execution_disposition.rationale must be non-empty")
    return tuple(problems)


@dataclass(frozen=True)
class ExecutionScopeItem:
    scope_item_id: str
    requirement: str
    acceptance_criteria: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionCouplingConstraint:
    constraint_id: str
    scope_item_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class ExecutionAllocation:
    status: str
    deliverables: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    covered_scope_item_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionDisposition:
    """The reviewed per-child execution disposition (#808)."""

    disposition: str
    rationale: str
    unresolved_design_decisions: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        clean = sanitize_historical_text  # shape-check: fatal:authentication-or-forgery
        return {
            "disposition": clean(self.disposition),
            "rationale": clean(self.rationale),
            "unresolved_design_decisions": [
                clean(value) for value in self.unresolved_design_decisions
            ],
        }


@dataclass(frozen=True)
class ExecutionChildStage:
    stage_id: str
    position: int
    title: str
    summary: str
    deliverables: tuple[str, ...]
    non_goals: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    depends_on_stage_ids: tuple[str, ...]
    dependency_notes: str
    automation: str
    rollout_risk: str
    compatibility_constraints: tuple[str, ...]
    covered_scope_item_ids: tuple[str, ...]
    # ``None`` only for recovery parsing of plans approved before #808
    # (legacy-ambiguous).  Fresh planner and revision responses must declare it.
    execution_disposition: ExecutionDisposition | None = None


@dataclass(frozen=True)
class ExecutionOneShotDelivery:
    deliverables: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    covered_scope_item_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionStrategyRecommendation:
    strategy: str
    rationale: str
    staging_feasibility: str
    scope_items: tuple[ExecutionScopeItem, ...]
    coupling_constraints: tuple[ExecutionCouplingConstraint, ...]
    one_shot_delivery: ExecutionOneShotDelivery | None
    child_stages: tuple[ExecutionChildStage, ...]
    retained_parent_work: ExecutionAllocation
    final_integration_work: ExecutionAllocation
    caveats: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        """Return the exact v1 wire shape for canonical rendering/storage."""
        clean = sanitize_historical_text  # shape-check: fatal:authentication-or-forgery

        def allocation(value: ExecutionAllocation) -> dict[str, object]:
            return {
                "status": clean(value.status),
                "deliverables": [clean(item) for item in value.deliverables],
                "acceptance_criteria": [clean(item) for item in value.acceptance_criteria],
                "covered_scope_item_ids": [clean(item) for item in value.covered_scope_item_ids],
            }

        payload: dict[str, object] = {
            "strategy": clean(self.strategy),
            "rationale": clean(self.rationale),
            "staging_feasibility": clean(self.staging_feasibility),
            "scope_items": [
                {
                    "scope_item_id": clean(item.scope_item_id),
                    "requirement": clean(item.requirement),
                    "acceptance_criteria": [clean(value) for value in item.acceptance_criteria],
                }
                for item in self.scope_items
            ],
            "coupling_constraints": [
                {
                    "constraint_id": clean(item.constraint_id),
                    "scope_item_ids": [clean(value) for value in item.scope_item_ids],
                    "rationale": clean(item.rationale),
                }
                for item in self.coupling_constraints
            ],
            "child_stages": [
                {
                    "stage_id": clean(stage.stage_id),
                    "position": stage.position,
                    "title": clean(stage.title),
                    "summary": clean(stage.summary),
                    "deliverables": [clean(value) for value in stage.deliverables],
                    "non_goals": [clean(value) for value in stage.non_goals],
                    "acceptance_criteria": [clean(value) for value in stage.acceptance_criteria],
                    "depends_on_stage_ids": [clean(value) for value in stage.depends_on_stage_ids],
                    "dependency_notes": clean(stage.dependency_notes),
                    "automation": clean(stage.automation),
                    "rollout_risk": clean(stage.rollout_risk),
                    "compatibility_constraints": [clean(value) for value in stage.compatibility_constraints],
                    "covered_scope_item_ids": [clean(value) for value in stage.covered_scope_item_ids],
                    # Emitted only when declared so digests of plans approved
                    # before the disposition contract remain byte-stable.
                    **(
                        {"execution_disposition": stage.execution_disposition.to_payload()}  # shape-check: fatal:authentication-or-forgery
                        if stage.execution_disposition is not None else {}
                    ),
                }
                for stage in self.child_stages
            ],
            "retained_parent_work": allocation(self.retained_parent_work),
            "final_integration_work": allocation(self.final_integration_work),
            "caveats": [clean(value) for value in self.caveats],
        }
        if self.one_shot_delivery is not None:
            payload["one_shot_delivery"] = {
                "deliverables": [clean(value) for value in self.one_shot_delivery.deliverables],
                "acceptance_criteria": [clean(value) for value in self.one_shot_delivery.acceptance_criteria],
                "covered_scope_item_ids": [clean(value) for value in self.one_shot_delivery.covered_scope_item_ids],
            }
        return payload

    def identity(self) -> dict[str, object]:
        """Stable mode-independent identity for the complete recommendation.

        The digest deliberately covers every approval-relevant field, including
        prose, coverage, dependencies, automation, compatibility constraints,
        and caveats.  Short topology summaries are useful diagnostics but are
        not sufficient to prove that a resumed response is the approved one.
        """
        canonical = json.dumps(
            self.to_payload(), separators=(",", ":"), sort_keys=True, ensure_ascii=False  # shape-check: fatal:authentication-or-forgery
        ).encode("utf-8")
        return {
            "contract_version": EXECUTION_STRATEGY_CONTRACT_VERSION,
            "strategy": self.strategy,
            "topology_source": EXECUTION_TOPOLOGY_SOURCE,
            "recommendation_sha256": hashlib.sha256(canonical).hexdigest(),
            "scope_item_ids": [item.scope_item_id for item in self.scope_items],
            "stage_ids": [stage.stage_id for stage in self.child_stages],
            "retained_parent_status": self.retained_parent_work.status,
            "final_integration_status": self.final_integration_work.status,
        }


# Shared classification rules for typed plan validation and materialization.
ISSUE_REFERENCE_RE = re.compile(
    r"(?:\B#[1-9]\d*\b|github\.com/[^/\s]+/[^/\s]+/issues/[1-9]\d*\b|[\w.-]+/[\w.-]+#[1-9]\d*\b)",
    re.I,
)
TRACKER_ACTION_TITLE_RE = re.compile(
    r"^(?:post|after)[ -]?approval.*(?:\b(?:creat|fil|materializ).*\bchild issue|\bchild issue.*\bmaterializ)",
    re.I,
)


# Structural plan-growth signals (#886).  The closed set is shared by the
# typed justification parser and ``plan_growth``'s deterministic assessment.
PLAN_GROWTH_SIGNALS = ("rendered-size", "scope-items", "matrix-rows", "revision-count")
PLAN_GROWTH_JUSTIFICATION_MAX_RATIONALE_CHARS = 2_000


@dataclass(frozen=True)
class OneShotGrowthJustification:
    """A reviewed claim that a grown one-shot plan should stay one-shot (#886).

    Shape only: which signals must be named is decided by the post-assembly
    growth self-check, which can measure the candidate's canonical rendering.
    """

    crossed_signals: tuple[str, ...]
    rationale: str

    def to_payload(self) -> dict[str, object]:
        return {"crossed_signals": list(self.crossed_signals), "rationale": self.rationale}


def _expect_growth_justification(value: object, *, context: str) -> OneShotGrowthJustification:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(payload, context=context, required={"crossed_signals", "rationale"})  # shape-check: fatal:no-conservative-reading
    signals_value = payload["crossed_signals"]
    if not isinstance(signals_value, list) or not signals_value:
        raise AgentLoopError(f"{context}.crossed_signals must be a non-empty JSON array.")  # shape-check: fatal:no-conservative-reading
    signals: list[str] = []
    for index, item in enumerate(signals_value):
        if not isinstance(item, str) or item not in PLAN_GROWTH_SIGNALS:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{context}.crossed_signals[{index}] must be one of: "
                + ", ".join(PLAN_GROWTH_SIGNALS)
                + "."
            )
        if item in signals:
            raise AgentLoopError(f"{context}.crossed_signals must not repeat `{item}`.")  # shape-check: fatal:no-conservative-reading
        signals.append(item)
    rationale = _expect_non_empty_string(payload["rationale"], context=f"{context}.rationale")  # shape-check: fatal:no-conservative-reading
    if len(rationale) > PLAN_GROWTH_JUSTIFICATION_MAX_RATIONALE_CHARS:
        raise AgentLoopError(  # shape-check: fatal:payload-bound
            f"{context}.rationale must be at most "
            f"{PLAN_GROWTH_JUSTIFICATION_MAX_RATIONALE_CHARS} characters."
        )
    return OneShotGrowthJustification(crossed_signals=tuple(signals), rationale=rationale)


def _expect_optional_growth_justification(
    payload: Mapping[str, object], *, context: str
) -> OneShotGrowthJustification | None:
    if "one_shot_growth_justification" not in payload:
        return None
    return _expect_growth_justification(  # shape-check: fatal:no-conservative-reading
        payload["one_shot_growth_justification"],
        context=f"{context}.one_shot_growth_justification",
    )


@dataclass(frozen=True)
class StructuredPlanRevision:
    schema_version: int
    kind: str
    state: str
    summary: str
    prior_plan_item_dispositions: tuple[ReviewItemDisposition, ...]
    plan_steps: tuple[str, ...]
    # None means the plan made no declaration. An empty tuple is an explicit
    # declaration that the single PR completes no additional issues.
    additional_closing_issue_ids: tuple[int, ...] | None = None
    deferred_stages: tuple[DeferredStage, ...] = ()
    typed_stages: TypedPlanStages = TypedPlanStages()
    human_requirement_dispositions: tuple[HumanRequirementDisposition, ...] = ()
    architecture_impact: ArchitectureImpact | None = None
    execution_strategy_contract_version: int | None = None
    execution_recommendation: ExecutionStrategyRecommendation | None = None
    risk_test_matrix_contract_version: int | None = None
    risk_test_matrix: "RiskTestMatrix | None" = None
    risk_test_matrix_changes: tuple["RiskTestMatrixChange", ...] = ()
    architecture_impact_contract: ArchitectureImpactContract = ArchitectureImpactContract()
    architecture_impact_degradations: tuple[ParseDegradation, ...] = ()
    # Absent unless the planner justified keeping a grown plan one-shot (#886).
    one_shot_growth_justification: OneShotGrowthJustification | None = None


@dataclass(frozen=True)
class StructuredPlanState:
    schema_version: int
    kind: str
    state: str
    summary: str
    plan_steps: tuple[str, ...]
    # Presence is preserved so an explicit empty declaration cannot be confused
    # with an omitted declaration during contract reconciliation.
    additional_closing_issue_ids: tuple[int, ...] | None = None
    deferred_stages: tuple[DeferredStage, ...] = ()
    typed_stages: TypedPlanStages = TypedPlanStages()
    human_requirement_dispositions: tuple[HumanRequirementDisposition, ...] = ()
    architecture_impact: ArchitectureImpact | None = None
    execution_strategy_contract_version: int | None = None
    execution_recommendation: ExecutionStrategyRecommendation | None = None
    risk_test_matrix_contract_version: int | None = None
    risk_test_matrix: "RiskTestMatrix | None" = None
    risk_test_matrix_changes: tuple["RiskTestMatrixChange", ...] = ()
    architecture_impact_contract: ArchitectureImpactContract = ArchitectureImpactContract()
    architecture_impact_degradations: tuple[ParseDegradation, ...] = ()
    # Absent unless the planner justified keeping a grown plan one-shot (#886).
    one_shot_growth_justification: OneShotGrowthJustification | None = None

def is_failed_discuss_response(response: ParsedDiscussResponse) -> bool:
    return isinstance(response, ParsedFailedDiscussResponse) or (
        isinstance(response, ParsedDiscussReview) and response.outcome == DISCUSS_FAILED_OUTCOME
    )


def discuss_response_position(response: ParsedDiscussResponse) -> str | None:
    if isinstance(response, ParsedDiscussAnswer):
        return response.position
    if isinstance(response, ParsedDiscussReview):
        return response.outcome
    return None


def failed_discuss_review_placeholder(reviewer: str, category: str) -> ParsedDiscussReview:
    return ParsedDiscussReview(
        outcome=DISCUSS_FAILED_OUTCOME,
        rationale=f"did not respond this round ({category})",
        split_proposals=(),
        reviewer=reviewer,
    )


def failed_discuss_review_category(vote: ParsedDiscussReview) -> str:
    """Recover the failure category from a placeholder vote's rationale."""
    match = re.fullmatch(r"did not respond this round \((.+)\)", vote.rationale)
    return match.group(1) if match else vote.rationale


def failed_discuss_answer_placeholder(reviewer: str, category: str) -> ParsedFailedDiscussResponse:
    return ParsedFailedDiscussResponse(reviewer=reviewer, category=category, result_mode="answer")


@dataclass(frozen=True)
class PrReference:
    """Classify the PR reference supplied in an agent response.

    Explicit markers are authoritative, including invalid ones, so an
    incidental URL cannot turn ``AGENT_PR: 0`` into a real PR handoff.
    """

    kind: str
    number: int | None = None

    @property
    def is_valid(self) -> bool:
        return self.kind == "valid"


def parse_agent_state(text: str) -> str:
    matches = STATE_RE.findall(text)
    if not matches:
        raise AgentLoopError("Agent response did not include <!-- AGENT_STATE: approved|blocking -->")  # shape-check: fatal:unparseable-envelope
    # Use the final marker as authoritative; responses may quote earlier review markers.
    return matches[-1].lower()


def parse_plan_state(text: str) -> str:
    matches = PLAN_STATE_RE.findall(text)
    if not matches:
        raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
            "Agent response did not include <!-- AGENT_PLAN_STATE: approved|blocking -->"
        )
    return matches[-1].lower()


def classify_pr_reference(text: str) -> PrReference:
    """Return whether a response has an absent, valid, or invalid PR reference.

    Only standalone, non-fenced protocol markers count. The final explicit
    marker wins and is never rescued by a URL elsewhere in the response.
    """
    code_ranges = _fenced_code_block_ranges(text)

    def active(matches):
        return [m for m in matches if not any(start <= m.start() < end for start, end in code_ranges)]

    markers = active(list(_PR_MARKER_VALUE_RE.finditer(text)))
    if markers:
        value = markers[-1].group(1).strip()
        if re.fullmatch(r"\d+", value):
            number = int(value)
            if number > 0:
                return PrReference("valid", number)
        return PrReference("invalid")

    urls = active(list(GH_PR_URL_RE.finditer(text)))
    if urls:
        number = int(urls[-1].group(1))
        if number > 0:
            return PrReference("valid", number)
        return PrReference("invalid")
    return PrReference("absent")


def parse_pr_number(text: str) -> int | None:
    """Compatibility parser that returns only a positive PR number."""
    reference = classify_pr_reference(text)
    return reference.number if reference.is_valid else None


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


def _reject_legacy_requirement_labels(
    labels: Sequence[str],
    *,
    surfaced_requirement_ids: Sequence[str],
) -> None:
    """Require a fresh acknowledgement when current prompts use stable IDs.

    Positional labels remain parseable solely so historical records can produce
    this actionable migration diagnostic. They are never mapped onto the
    currently surfaced requirement tuple.
    """
    if not any(
        HUMAN_REQUIREMENT_STABLE_ID_RE.fullmatch(item.strip())
        for item in surfaced_requirement_ids
    ):
        return
    legacy = sorted({item for item in labels if re.fullmatch(r"Requirement \d+", item)})
    if legacy:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "Coder response uses legacy positional signed-requirement label(s) "
            f"{', '.join(legacy)}. Their meaning cannot be inferred from the current requirement "
            "set; provide a fresh acknowledgement using the exact surfaced stable hr-... IDs."
        )


def validate_human_requirements_acknowledgement(
    text: str,
    *,
    surfaced_requirement_ids: Sequence[str],
    requires_direct_discussion_ack: bool,
) -> None:
    parsed = parse_human_requirements_acknowledgement(text)  # shape-check: fatal:no-conservative-reading
    validate_structured_human_requirements_acknowledgement(  # shape-check: fatal:no-conservative-reading
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
    dispositions: Sequence[HumanRequirementDisposition] | None = None,
    checked_discussion_directly: bool,
    surfaced_requirement_ids: Sequence[str],
    requires_direct_discussion_ack: bool,
    marker_present: bool = True,
    section_present: bool = True,
) -> None:
    if not surfaced_requirement_ids and not requires_direct_discussion_ack:
        if addressed_ids:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                "Coder response listed signed human requirement IDs even though no signed human "
                "requirements were surfaced. Use `human_requirements.addressed_ids: []`; issue "
                "acceptance criteria, reviewer item IDs, reviewer comments, and arbitrary labels "
                "are not signed human requirements."
            )
        if checked_discussion_directly:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                "Coder response must set `human_requirements.checked_discussion_directly` to false "
                "when no signed human requirements are surfaced and direct discussion acknowledgement is not required."
            )
        return

    if not marker_present:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "Coder response missing required signed human requirements marker "
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}."
        )
    if not section_present:
        raise AgentLoopError("Coder response missing required `### Human requirements` section.")  # shape-check: fatal:no-conservative-reading

    normalized_addressed_ids = [_normalize_requirement_label(item_id) for item_id in addressed_ids]  # shape-check: fatal:no-conservative-reading
    _reject_legacy_requirement_labels(  # shape-check: fatal:no-conservative-reading
        normalized_addressed_ids,
        surfaced_requirement_ids=surfaced_requirement_ids,
    )
    duplicates = sorted(
        {
            item_id
            for item_id in normalized_addressed_ids
            if normalized_addressed_ids.count(item_id) > 1
        }
    )
    if duplicates:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "Coder response listed signed human requirement IDs more than once: "
            + ", ".join(duplicates)
        )

    expected_ids = tuple(_normalize_requirement_label(item_id) for item_id in surfaced_requirement_ids)  # shape-check: fatal:no-conservative-reading
    unknown = sorted(set(normalized_addressed_ids) - set(expected_ids))
    if unknown:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "Coder response referenced unknown signed human requirement IDs: "
            + ", ".join(unknown)
        )

    if expected_ids:
        if dispositions is None:
            expected_addressed_ids = set(expected_ids)
            missing_message = "Coder response did not address all surfaced signed human requirement IDs: "
        else:
            expected_addressed_ids = {
                _normalize_requirement_label(item.requirement_id)  # shape-check: fatal:no-conservative-reading
                for item in dispositions
                if item.disposition == "addressed"
            }
            missing_message = (
                "Coder response human_requirements.addressed_ids must contain every requirement "
                "with an `addressed` disposition: "
            )
        actual_addressed_ids = set(normalized_addressed_ids)
        missing = sorted(expected_addressed_ids - actual_addressed_ids)
        unexpected = sorted(actual_addressed_ids - expected_addressed_ids)
        if missing:
            raise AgentLoopError(missing_message + ", ".join(missing))  # shape-check: fatal:no-conservative-reading
        if unexpected:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                "Coder response human_requirements.addressed_ids may contain only requirements "
                "with an `addressed` disposition: "
                + ", ".join(unexpected)
            )
        return

    if not checked_discussion_directly:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
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


def render_architecture_list_entry(value: object) -> str | None:
    """Losslessly render one architecture-impact list entry, or None (#1330).

    A non-blank string renders as its stripped text.  A flat object with only
    string or string-list values renders one `key: value` pair per item in the
    object's own key order, pairs joined by `; ` and list values joined by
    `, `.  Every other value, including an object with a defect reported by
    `_flat_object_defect`, returns None.  The payload parser, the near-miss
    corroboration predicate and repair preservation all decide renderability
    through this one helper.
    """
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, Mapping) and _flat_object_defect(value) is None:
        return _render_flat_object(value)
    return None



# Structured plan-review finding objects (#957).  Reviewers asked for grounded,
# itemised findings often emit objects instead of strings.  Each object is
# flattened mechanically, in this fixed order, into one finding string that
# keeps every supplied prose value verbatim, so no model repair is needed and
# no qualifier can be lost.  Identifier keys are dropped: they are the
# reviewer's local labels, not orchestrator item IDs.  Reference keys (#1230)
# are the exception: the plan renderer shows scope/stage/constraint/row IDs to
# reviewers, and they name which plan requirement a finding concerns, so they
# are folded into the text as labelled lines instead of dropped.  Unknown keys
# (#1328) whose value is a non-empty string or a non-empty list of non-empty
# strings are folded too, as `key: value` parts in the reviewer's own key
# order, after the reference lines and before `Sub-items:`: a reviewer may copy
# any field name from the PR or plan under review, so no allow-list can
# anticipate them.  Unknown keys with any other value, keys in
# REVIEW_FINDING_RESERVED_KEYS, and extras beyond the fold bounds are still
# rejected.  Repair-side handling of folded extras is unchanged (#1337).
PLAN_REVIEW_FINDING_TEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("title", ""),
    ("text", ""),
    ("issue", ""),
    ("finding", ""),
    ("description", ""),
    ("summary", ""),
    ("location", "Location"),
    ("evidence", "Evidence"),
    ("rationale", "Rationale"),
    ("impact", "Impact"),
    ("required_change", "Required change"),
    ("recommendation", "Recommendation"),
    ("suggested_fix", "Suggested fix"),
)
PLAN_REVIEW_FINDING_ID_FIELDS = frozenset({"item_id", "id"})
PLAN_REVIEW_FINDING_REFERENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("scope_item_ids", "Scope items"),
    ("scope_item_id", "Scope items"),
    ("covered_scope_item_ids", "Covered scope items"),
    ("stage_ids", "Stages"),
    ("stage_id", "Stages"),
    ("constraint_ids", "Coupling constraints"),
    ("constraint_id", "Coupling constraints"),
    ("row_ids", "Matrix rows"),
    ("row_id", "Matrix rows"),
)
# Rendered plan identifier keys deliberately not accepted on findings.
PLAN_REVIEW_FINDING_EXCLUDED_RENDERED_KEYS = frozenset(
    {
        # A stage-to-stage plan relation, not a finding reference; use stage_ids.
        "depends_on_stage_ids",
    }
)
# Explicit, unambiguous key aliases (#1169): each alias is renamed to its one
# canonical key before validation, so a synonym spelling needs no model repair.
# Extend this table when a new model adopts a new spelling; there is no fuzzy
# matching.
PLAN_REVIEW_FINDING_KEY_ALIASES: Mapping[str, str] = MappingProxyType(
    {"requested_change": "required_change"}
)
# Keys never folded into a finding's text (#1328): review envelope fields,
# disposition fields, the scheduler field `fix_scope`, alias spellings and
# rendered identifiers deliberately excluded from findings.  Folding one would
# turn protocol meaning into prose, so it is rejected as a collision instead.
# A parser's own accepted keys are skipped before this check, so the PR parser
# keeps its typed `fix_scope` handling.  `summary` is deliberately absent: it
# is a documented plan-finding text field.  Add future envelope or disposition
# keys here.
REVIEW_FINDING_RESERVED_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "kind",
        "state",
        "blocking_items",
        "same_pr_followups",
        "future_followups",
        "blocking_plan_issues",
        "same_plan_followups",
        "prior_item_dispositions",
        "prior_plan_item_dispositions",
        "human_requirement_dispositions",
        "architecture_impact",
        "exact_head_evidence_requests",
    }
    | {"disposition", "note", "sub_item_dispositions", "requirement_id", "requirement_label"}
    | {"fix_scope"}
    | set(PLAN_REVIEW_FINDING_KEY_ALIASES)
    | PLAN_REVIEW_FINDING_EXCLUDED_RENDERED_KEYS
)
REVIEW_FINDING_FOLD_MAX_KEYS = 16
REVIEW_FINDING_FOLD_MAX_KEY_CHARS = 64
REVIEW_FINDING_FOLD_MAX_ENTRIES = 32
REVIEW_FINDING_FOLD_MAX_VALUE_CHARS = 2000


# #1241: plan reviewers sometimes key a disposition `requirement_label`.  The
# alias is admitted only for an exact surfaced stable ID (see
# `_expect_human_requirement_dispositions`).
HUMAN_REQUIREMENT_DISPOSITION_KEY_ALIASES: Mapping[str, str] = MappingProxyType(
    {"requirement_label": "requirement_id"}
)


def _normalize_plan_review_finding_key_aliases(raw: dict, *, item_context: str) -> dict:
    """Return a copy of ``raw`` with alias keys renamed; values are untouched.

    An alias is left in place when its canonical key is also present, so the
    unknown-key check rejects the collision.  ``raw`` is never mutated.
    """
    normalized = dict(raw)
    for alias, canonical in PLAN_REVIEW_FINDING_KEY_ALIASES.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized.pop(alias)
            _logger.warning(
                "%s: normalized finding key alias %s -> %s", item_context, alias, canonical
            )
    return normalized


def _fold_unknown_finding_keys(
    raw: Mapping[str, object],
    *,
    accepted: frozenset[str],
    reserved: frozenset[str],
    item_context: str,
) -> tuple[list[str], list[str], list[str]]:
    """Fold string-valued unknown finding keys into labelled text (#1328).

    Returns ``(folded_parts, folded_keys, remaining_unknown)`` in the object's
    own key order.  Foldability and rendering reuse #1330's flat-object
    helpers on a singleton mapping, so a key folds exactly when its value is a
    non-empty string or a non-empty list of non-empty strings.  A reserved key
    raises a collision; a non-foldable key is returned for the caller's own
    unknown-key error.  Nothing is logged here.
    """
    folded_parts: list[str] = []
    folded_keys: list[str] = []
    remaining_unknown: list[str] = []
    for key, value in raw.items():
        if key in accepted:
            continue
        if key in reserved:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{item_context}.{key} collides with a protocol or disposition field "
                "and cannot be used as a finding key."
            )
        single = {key: value}
        if _flat_object_defect(single) is not None:
            remaining_unknown.append(str(key))
            continue
        if len(key) > REVIEW_FINDING_FOLD_MAX_KEY_CHARS:
            raise AgentLoopError(  # shape-check: fatal:payload-bound
                f"{item_context} finding key {key[:REVIEW_FINDING_FOLD_MAX_KEY_CHARS]!r}... "
                f"exceeds the {REVIEW_FINDING_FOLD_MAX_KEY_CHARS}-character key bound."
            )
        entries = [value] if isinstance(value, str) else value
        if len(entries) > REVIEW_FINDING_FOLD_MAX_ENTRIES:
            raise AgentLoopError(  # shape-check: fatal:payload-bound
                f"{item_context}.{key} exceeds the {REVIEW_FINDING_FOLD_MAX_ENTRIES}-entry bound."
            )
        if any(len(entry) > REVIEW_FINDING_FOLD_MAX_VALUE_CHARS for entry in entries):
            raise AgentLoopError(  # shape-check: fatal:payload-bound
                f"{item_context}.{key} exceeds the "
                f"{REVIEW_FINDING_FOLD_MAX_VALUE_CHARS}-character value bound."
            )
        folded_keys.append(key)
        if len(folded_keys) > REVIEW_FINDING_FOLD_MAX_KEYS:
            raise AgentLoopError(  # shape-check: fatal:payload-bound
                f"{item_context} has more than {REVIEW_FINDING_FOLD_MAX_KEYS} "
                f"unknown finding keys to fold; {key} exceeds the bound."
            )
        folded_parts.append(_render_flat_object(single))
    return folded_parts, folded_keys, remaining_unknown


def _log_folded_finding_keys(item_context: str, keys: list[str]) -> None:
    if keys:
        _logger.warning(
            "%s: folded unknown finding key(s) %s into the finding text",
            item_context,
            ", ".join(keys),
        )


def _flatten_plan_review_finding(raw: object, *, item_context: str) -> str:
    if isinstance(raw, str):
        return _expect_non_empty_string(raw, context=item_context)  # shape-check: fatal:no-conservative-reading
    if not isinstance(raw, dict):
        raise AgentLoopError(f"{item_context} must be a string or a finding object.")  # shape-check: fatal:no-conservative-reading
    raw = _normalize_plan_review_finding_key_aliases(raw, item_context=item_context)
    text_fields = {name for name, _label in PLAN_REVIEW_FINDING_TEXT_FIELDS}
    reference_fields = {name for name, _label in PLAN_REVIEW_FINDING_REFERENCE_FIELDS}
    # ``provenance`` (#1505) is read separately by the parser and never folded
    # into the finding text.
    accepted = (
        text_fields | reference_fields | PLAN_REVIEW_FINDING_ID_FIELDS
        | {"sub_items", PLAN_REVIEW_FINDING_PROVENANCE_FIELD}
    )
    folded_parts, folded_keys, unknown = _fold_unknown_finding_keys(  # shape-check: fatal:payload-bound
        raw,
        accepted=frozenset(accepted),
        reserved=REVIEW_FINDING_RESERVED_KEYS,
        item_context=item_context,
    )
    if unknown:
        allowed = ", ".join(sorted(accepted))
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            f"{item_context} has unsupported finding key(s) {', '.join(sorted(unknown))}; "
            f"use a string or an object with only: {allowed}; unknown keys with a "
            "non-empty string or string-list value are folded into the text."
        )
    for name in sorted(PLAN_REVIEW_FINDING_ID_FIELDS & set(raw)):
        if not isinstance(raw[name], str):
            raise AgentLoopError(f"{item_context}.{name} must be a string.")  # shape-check: fatal:no-conservative-reading
    parts: list[str] = []
    for name, label in PLAN_REVIEW_FINDING_TEXT_FIELDS:
        if name not in raw:
            continue
        value = _expect_non_empty_string(raw[name], context=f"{item_context}.{name}").strip()  # shape-check: fatal:no-conservative-reading
        parts.append(f"{label}: {value}" if label else value)
    if not parts:
        raise AgentLoopError(f"{item_context} finding object has no text field.")  # shape-check: fatal:no-conservative-reading
    for name, label in PLAN_REVIEW_FINDING_REFERENCE_FIELDS:
        if name not in raw:
            continue
        reference = raw[name]
        if isinstance(reference, str):
            references = [_expect_non_empty_string(reference, context=f"{item_context}.{name}").strip()]  # shape-check: fatal:no-conservative-reading
        elif isinstance(reference, list) and reference:
            references = [
                _expect_non_empty_string(entry, context=f"{item_context}.{name}[{position}]").strip()  # shape-check: fatal:no-conservative-reading
                for position, entry in enumerate(reference)
            ]
        else:
            raise AgentLoopError(f"{item_context}.{name} must be a non-empty string or a non-empty JSON array of strings.")  # shape-check: fatal:no-conservative-reading
        parts.append(f"{label}: {', '.join(references)}")
    parts.extend(folded_parts)
    if "sub_items" in raw:
        # Plan reviews get no ledger structure (#958): well-formed statements
        # are flattened into the finding text with nothing lost.
        sub_items = raw["sub_items"]
        if not isinstance(sub_items, list) or not sub_items:
            raise AgentLoopError(f"{item_context}.sub_items must be a non-empty JSON array of strings.")  # shape-check: fatal:no-conservative-reading
        statements = [
            _expect_non_empty_string(entry, context=f"{item_context}.sub_items[{position}]").strip()  # shape-check: fatal:no-conservative-reading
            for position, entry in enumerate(sub_items)
        ]
        parts.append(
            "Sub-items: "
            + " ".join(f"({number}) {statement}" for number, statement in enumerate(statements, 1))
        )
    _log_folded_finding_keys(item_context, folded_keys)
    return " ".join(parts)


def _expect_plan_review_finding_list(
    payload: dict[str, object],
    field_name: str,
    *,
    context: str,
) -> tuple[str, ...]:
    """Accept plan-review findings as strings or flattenable objects (#957)."""
    value = payload.get(field_name, [])
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    return tuple(
        _flatten_plan_review_finding(raw, item_context=f"{context} at index {index}")  # shape-check: fatal:no-conservative-reading
        for index, raw in enumerate(value)
    )


def _plan_review_finding_provenances(
    payload: dict[str, object], field_name: str, *, context: str
) -> tuple[tuple[str | None, ...], tuple[ParseDegradation, ...]]:
    """Read each finding's optional ``provenance`` tag, degrading a bad tag (#1505).

    The tag is advisory telemetry: an unknown value or a wrong type drops only
    the tag (the finding stays, untagged) and never rejects the review.
    """
    value = payload.get(field_name, [])
    if not isinstance(value, list):
        return (), ()
    tags: list[str | None] = []
    degradations: list[ParseDegradation] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or PLAN_REVIEW_FINDING_PROVENANCE_FIELD not in raw:
            tags.append(None)
            continue
        tag = raw[PLAN_REVIEW_FINDING_PROVENANCE_FIELD]
        if isinstance(tag, str) and tag in PLAN_REVIEW_PROVENANCE_VALUES:
            tags.append(tag)
            continue
        tags.append(None)
        degradations.append(ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
            element_path=f"{context}[{index}].provenance",
            rule="provenance-not-an-allowed-value",
            observed=tag,
            outcome="tag-dropped",
        ))
    return tuple(tags), tuple(degradations)


def _degradable_sub_items(
    value: object, *, context: str
) -> tuple[tuple[str, ...], ParseDegradation | None]:
    """Read optional finding sub-items, degrading any defect to a plain finding (#958)."""
    rule: str | None = None
    statements: list[str] = []
    if not isinstance(value, list):
        rule = "sub_items-not-an-array"
    elif len(value) < SUB_ITEM_MIN_COUNT:
        rule = "sub_items-needs-at-least-two-entries"
    elif len(value) > SUB_ITEM_MAX_COUNT:
        rule = "sub_items-more-than-twelve-entries"
    else:
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                rule = "sub_items-entry-not-a-non-empty-string"
                break
            normalized = _normalized_review_item_text(entry)
            if not normalized or normalized in seen:
                rule = "sub_items-duplicate-or-blank-statement"
                break
            seen.add(normalized)
            statements.append(entry.strip())
    if rule is None:
        return tuple(statements), None
    return (), ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
        element_path=f"{context}.sub_items",
        rule=rule,
        observed=value,
        outcome="claim-dropped",
    )


REVIEW_FINDING_PR_ACCEPTED_KEYS = frozenset({"text", "fix_scope", "sub_items", "evidence_row_ids"})


def _expect_review_finding_list(
    payload: dict[str, object],
    field_name: str,
    *,
    context: str,
    reviewer: str,
    degradations: list[ParseDegradation] | None = None,
    allow_sub_items: bool = True,
    allow_evidence_row_ids: bool = False,
) -> tuple[ApprovedFollowup, ...]:
    """Accept legacy strings and the scoped PR finding representation."""
    value = payload.get(field_name, [])
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    findings: list[ApprovedFollowup] = []
    for index, raw in enumerate(value):
        item_context = f"{context} at index {index}"
        sub_items: tuple[str, ...] = ()
        evidence_row_ids: tuple[str, ...] = ()
        if isinstance(raw, str):
            text = _expect_non_empty_string(raw, context=item_context)  # shape-check: fatal:no-conservative-reading
            scope = None
        else:
            item = _expect_object(raw, context=item_context)  # shape-check: fatal:no-conservative-reading
            if "text" not in item:
                raise AgentLoopError(f"{item_context} is missing required field(s): text")  # shape-check: fatal:no-conservative-reading
            folded_parts, folded_keys, unknown = _fold_unknown_finding_keys(  # shape-check: fatal:payload-bound
                item,
                accepted=REVIEW_FINDING_PR_ACCEPTED_KEYS,
                reserved=REVIEW_FINDING_RESERVED_KEYS | PLAN_REVIEW_FINDING_ID_FIELDS,
                item_context=item_context,
            )
            if unknown:
                raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                    f"{item_context} has unknown field(s): {', '.join(sorted(unknown))}"
                )
            if "evidence_row_ids" in item:
                if not allow_evidence_row_ids:
                    raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                        f"{item_context}.evidence_row_ids is accepted only on `blocking_items`."
                    )
                evidence_row_ids = _expect_evidence_row_ids(  # shape-check: fatal:no-conservative-reading
                    item["evidence_row_ids"], context=f"{item_context}.evidence_row_ids"
                )
            text = _expect_non_empty_string(item["text"], context=f"{item_context}.text")  # shape-check: fatal:no-conservative-reading
            if folded_parts:
                text = " ".join([text, *folded_parts])
            try:
                scope = normalize_fix_scope(item.get("fix_scope")) if "fix_scope" in item else None  # shape-check: fatal:no-conservative-reading
            except AgentLoopError as exc:
                raise AgentLoopError(f"{item_context}.fix_scope is invalid: {exc}") from exc  # shape-check: fatal:no-conservative-reading
            if "sub_items" in item:
                if allow_sub_items:
                    sub_items, record = _degradable_sub_items(  # shape-check: fatal:authentication-or-forgery
                        item["sub_items"], context=item_context
                    )
                else:
                    sub_items = ()
                    record = ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
                        element_path=f"{item_context}.sub_items",
                        rule="sub_items-not-supported-on-this-finding-list",
                        observed=item["sub_items"],
                        outcome="claim-dropped",
                    )
                if record is not None and degradations is not None:
                    degradations.append(record)
            _log_folded_finding_keys(item_context, folded_keys)
        findings.append(
            ApprovedFollowup(
                reviewer=reviewer,
                text=text,
                fix_scope=scope,
                sub_items=sub_items,
                evidence_row_ids=evidence_row_ids,
            )
        )
    return tuple(findings)


EVIDENCE_ROW_IDS_MAX = 32


def _expect_evidence_row_ids(value: object, *, context: str) -> tuple[str, ...]:
    """Strictly parse a finding's ``evidence_row_ids`` tag (#1324)."""
    if not isinstance(value, list) or not value:
        raise AgentLoopError(f"{context} must be a non-empty JSON array of row ID strings.")  # shape-check: fatal:no-conservative-reading
    if len(value) > EVIDENCE_ROW_IDS_MAX:
        raise AgentLoopError(  # shape-check: fatal:payload-bound
            f"{context} exceeds the {EVIDENCE_ROW_IDS_MAX}-row bound."
        )
    row_ids: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str) or not raw.strip() or len(raw) > 128:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{context} at index {index} must be a non-empty row ID string."
            )
        if raw in row_ids:
            raise AgentLoopError(f"{context} repeats row ID `{raw}`.")  # shape-check: fatal:no-conservative-reading
        row_ids.append(raw)
    return tuple(row_ids)


def _expect_optional_issue_id_list(
    payload: dict[str, object],
    field_name: str,
    *,
    context: str,
) -> tuple[int, ...] | None:
    if field_name not in payload:
        return None
    value = payload[field_name]
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    ids: set[int] = set()
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{context} item at index {index} must be a unique positive integer (not a bool)."
            )
        if item in ids:
            raise AgentLoopError(f"{context} contains duplicate issue ID #{item}.")  # shape-check: fatal:no-conservative-reading
        ids.add(item)
    return tuple(sorted(ids))


def _expect_deferred_stage_list(
    payload: dict[str, object],
    field_name: str,
    *,
    context: str,
) -> tuple[DeferredStage, ...]:
    value = payload.get(field_name, [])
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    stages: list[DeferredStage] = []
    seen_title_keys: set[str] = set()
    for index, item in enumerate(value):
        item_context = f"{context} at index {index}"
        item_payload = _expect_object(item, context=item_context)  # shape-check: fatal:no-conservative-reading
        _expect_exact_keys(item_payload, context=item_context, required={"title", "summary"})  # shape-check: fatal:no-conservative-reading
        title = _expect_non_empty_string(item_payload["title"], context=f"{item_context}.title")  # shape-check: fatal:no-conservative-reading
        title_key = " ".join(title.lower().split())
        if title_key in seen_title_keys:
            raise AgentLoopError(f"{context} has a duplicate stage title: {title!r}.")  # shape-check: fatal:no-conservative-reading
        seen_title_keys.add(title_key)
        stages.append(
            DeferredStage(
                title=title,
                summary=_expect_non_empty_string(  # shape-check: fatal:no-conservative-reading
                    item_payload["summary"], context=f"{item_context}.summary"
                ),
            )
        )
    return tuple(stages)


def _expect_typed_plan_stages(payload: dict[str, object], *, context: str) -> TypedPlanStages:
    """Parse #585's explicit plan categories and retain old payloads safely.

    Legacy ``deferred_stages`` are deliberately *not* promoted to children:
    old plans may contain dependencies and tracker actions.
    """
    categories = ("child_stages", "external_dependencies", "deferred_work", "plan_actions")
    present = [name for name in categories if name in payload]
    _expect_deferred_stage_list(payload, "deferred_stages", context=f"{context}.deferred_stages")  # shape-check: fatal:no-conservative-reading
    if not present:
        return TypedPlanStages()

    def parse_child_stages() -> tuple[ChildStage, ...]:
        value = payload.get("child_stages", [])
        if not isinstance(value, list):
            raise AgentLoopError(f"{context}.child_stages must be a JSON array.")  # shape-check: fatal:no-conservative-reading
        result: list[ChildStage] = []
        for index, item in enumerate(value):
            item_context = f"{context}.child_stages at index {index}"
            obj = _expect_object(item, context=item_context)  # shape-check: fatal:no-conservative-reading
            _expect_exact_keys(obj, context=item_context, required={"title", "summary"})  # shape-check: fatal:no-conservative-reading
            title = _expect_non_empty_string(obj["title"], context=f"{item_context}.title")  # shape-check: fatal:no-conservative-reading
            summary = _expect_non_empty_string(obj["summary"], context=f"{item_context}.summary")  # shape-check: fatal:no-conservative-reading
            if ISSUE_REFERENCE_RE.search(title + "\n" + summary):
                raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                    f"{item_context} references an existing issue; "
                    "use external_dependencies."
                )
            if TRACKER_ACTION_TITLE_RE.match(" ".join(title.split())):
                raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                    f"{item_context} is a tracker action; use plan_actions."
                )
            result.append(ChildStage(title, summary))
        return tuple(result)

    def recorded_stages(name: str) -> tuple[DeferredStage, ...]:
        value = payload.get(name, [])
        if not isinstance(value, list):
            raise AgentLoopError(f"{context}.{name} must be a JSON array.")  # shape-check: fatal:no-conservative-reading
        result: list[DeferredStage] = []
        for index, item in enumerate(value):
            item_context = f"{context}.{name} at index {index}"
            obj = _expect_object(item, context=item_context)  # shape-check: fatal:no-conservative-reading
            _expect_exact_keys(obj, context=item_context, required={"title", "summary"})  # shape-check: fatal:no-conservative-reading
            title = _expect_non_empty_string(obj["title"], context=f"{item_context}.title")  # shape-check: fatal:no-conservative-reading
            summary = _expect_non_empty_string(obj["summary"], context=f"{item_context}.summary")  # shape-check: fatal:no-conservative-reading
            result.append(DeferredStage(title, summary))
        return tuple(result)

    child_stages = parse_child_stages()  # shape-check: fatal:no-conservative-reading
    dependencies = recorded_stages("external_dependencies")  # shape-check: fatal:no-conservative-reading
    deferred = recorded_stages("deferred_work")  # shape-check: fatal:no-conservative-reading
    actions = recorded_stages("plan_actions")  # shape-check: fatal:no-conservative-reading
    seen: dict[str, str] = {}
    for category, entries in (
        ("child_stages", child_stages),
        ("external_dependencies", dependencies),
        ("deferred_work", deferred),
        ("plan_actions", actions),
    ):
        for entry in entries:
            key = " ".join(entry.title.casefold().split())
            if key in seen:
                raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                    f"{context} has a duplicate title in {seen[key]} and {category}: "
                    f"{entry.title!r}."
                )
            seen[key] = category
    return TypedPlanStages(child_stages, dependencies, deferred, actions)


def _expect_execution_allocation(
    value: object, *, context: str
) -> ExecutionAllocation:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload,
        context=context,
        required={"status", "deliverables", "acceptance_criteria", "covered_scope_item_ids"},
    )
    status = _expect_non_empty_string(payload["status"], context=f"{context}.status")  # shape-check: fatal:no-conservative-reading
    if status not in {"none", "required"}:
        raise AgentLoopError(f"{context}.status must be `none` or `required`.")  # shape-check: fatal:no-conservative-reading
    deliverables = _expect_string_list(  # shape-check: fatal:no-conservative-reading
        payload["deliverables"], context=f"{context}.deliverables",
        item_context=f"{context}.deliverables",
    )
    criteria = _expect_string_list(  # shape-check: fatal:no-conservative-reading
        payload["acceptance_criteria"], context=f"{context}.acceptance_criteria",
        item_context=f"{context}.acceptance_criteria",
    )
    covered = _expect_item_id_list(  # shape-check: fatal:no-conservative-reading
        payload["covered_scope_item_ids"], context=f"{context}.covered_scope_item_ids"
    )
    if status == "none" and (deliverables or criteria or covered):
        raise AgentLoopError(f"{context} with status `none` must have empty arrays.")  # shape-check: fatal:no-conservative-reading
    if status == "required" and (not deliverables or not criteria or not covered):
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            f"{context} with status `required` needs non-empty deliverables, "
            "acceptance_criteria, and covered_scope_item_ids."
        )
    return ExecutionAllocation(status, deliverables, criteria, covered)


def _expect_execution_disposition(
    value: object,
    *,
    context: str,
    automation: str,
    non_goals: Sequence[str],
    compatibility_constraints: Sequence[str],
    dependency_notes: str,
) -> ExecutionDisposition:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload,
        context=context,
        required={"disposition", "rationale", "unresolved_design_decisions"},
    )
    disposition = _expect_non_empty_string(payload["disposition"], context=f"{context}.disposition")  # shape-check: fatal:no-conservative-reading
    if disposition not in EXECUTION_DISPOSITION_VALUES:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            f"{context}.disposition must be one of: {', '.join(sorted(EXECUTION_DISPOSITION_VALUES))}."
        )
    allowed = EXECUTION_DISPOSITIONS_FOR_AUTOMATION.get(automation, frozenset())
    if disposition not in allowed:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            f"{context}.disposition `{disposition}` contradicts automation `{automation}`; "
            f"`{automation}` stages must declare {' or '.join(sorted(allowed))}."
        )
    rationale = _expect_non_empty_string(payload["rationale"], context=f"{context}.rationale")  # shape-check: fatal:no-conservative-reading
    unresolved = _expect_string_list(  # shape-check: fatal:no-conservative-reading
        payload["unresolved_design_decisions"],
        context=f"{context}.unresolved_design_decisions",
        item_context=f"{context}.unresolved_design_decisions",
    )
    if disposition == EXECUTION_DISPOSITION_DIRECT:
        problems = validate_direct_readiness(
            non_goals=non_goals,
            compatibility_constraints=compatibility_constraints,
            dependency_notes=dependency_notes,
            rationale=rationale,
            unresolved_design_decisions=unresolved,
        )
        if problems:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{context} declares direct-implementation but the stage is not direct-ready: "
                + "; ".join(problems)
                + ". Declare requires-child-planning or complete the stage contract."
            )
    return ExecutionDisposition(
        disposition=disposition,
        rationale=rationale,
        unresolved_design_decisions=unresolved,
    )


def _expect_execution_recommendation(
    value: object, *, context: str, require_child_dispositions: bool = False
) -> ExecutionStrategyRecommendation:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    tool_owned = [key for key in payload if key in TOOL_OWNED_EXECUTION_RECOMMENDATION_KEYS]
    if tool_owned:
        # #1333: a planner copies the tool-stamped keys from rendered context.
        # Strip only these audited top-level keys before the exact-key check;
        # the tool sets each of them itself after acceptance, so the drop is
        # lossless.  Normalizing inside the parser keeps the response text
        # untouched, which repair preservation pins byte-for-byte for patches.
        payload = {key: item for key, item in payload.items() if key not in TOOL_OWNED_EXECUTION_RECOMMENDATION_KEYS}
        for key in tool_owned:
            _logger.warning("%s: dropped tool-owned key `%s` before validation", context, key)
    required = {
        "strategy", "rationale", "staging_feasibility", "scope_items",
        "coupling_constraints", "child_stages", "retained_parent_work",
        "final_integration_work", "caveats",
    }
    _expect_exact_keys(payload, context=context, required=required, optional={"one_shot_delivery"})  # shape-check: fatal:no-conservative-reading
    strategy = _expect_non_empty_string(payload["strategy"], context=f"{context}.strategy")  # shape-check: fatal:no-conservative-reading
    if strategy not in {"one-shot", "staged"}:
        raise AgentLoopError(f"{context}.strategy must be `one-shot` or `staged`.")  # shape-check: fatal:no-conservative-reading
    rationale = _expect_non_empty_string(payload["rationale"], context=f"{context}.rationale")  # shape-check: fatal:no-conservative-reading
    feasibility = _expect_non_empty_string(  # shape-check: fatal:no-conservative-reading
        payload["staging_feasibility"], context=f"{context}.staging_feasibility"
    )
    if feasibility not in {"safe", "inseparable"}:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            f"{context}.staging_feasibility must be `safe` or `inseparable`."
        )

    scope_payload = payload["scope_items"]
    if not isinstance(scope_payload, list) or not scope_payload:
        raise AgentLoopError(f"{context}.scope_items must be a non-empty JSON array.")  # shape-check: fatal:no-conservative-reading
    scope_items: list[ExecutionScopeItem] = []
    scope_ids: set[str] = set()
    for index, raw_item in enumerate(scope_payload):
        item_context = f"{context}.scope_items[{index}]"
        item = _expect_object(raw_item, context=item_context)  # shape-check: fatal:no-conservative-reading
        _expect_exact_keys(item, context=item_context, required={"scope_item_id", "requirement", "acceptance_criteria"})  # shape-check: fatal:no-conservative-reading
        item_id = _expect_item_id(item["scope_item_id"], context=f"{item_context}.scope_item_id")  # shape-check: fatal:no-conservative-reading
        if item_id in scope_ids:
            raise AgentLoopError(f"{context}.scope_items has duplicate ID `{item_id}`.")  # shape-check: fatal:no-conservative-reading
        criteria = _expect_string_list(  # shape-check: fatal:no-conservative-reading
            item["acceptance_criteria"], context=f"{item_context}.acceptance_criteria",
            item_context=f"{item_context}.acceptance_criteria", min_length=1,
        )
        scope_ids.add(item_id)
        scope_items.append(
            ExecutionScopeItem(
                scope_item_id=item_id,
                requirement=_expect_non_empty_string(item["requirement"], context=f"{item_context}.requirement"),  # shape-check: fatal:no-conservative-reading
                acceptance_criteria=criteria,
            )
        )

    coupling_payload = payload["coupling_constraints"]
    if not isinstance(coupling_payload, list):
        raise AgentLoopError(f"{context}.coupling_constraints must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    coupling_constraints: list[ExecutionCouplingConstraint] = []
    coupling_ids: set[str] = set()
    for index, raw_constraint in enumerate(coupling_payload):
        item_context = f"{context}.coupling_constraints[{index}]"
        item = _expect_object(raw_constraint, context=item_context)  # shape-check: fatal:no-conservative-reading
        _expect_exact_keys(item, context=item_context, required={"constraint_id", "scope_item_ids", "rationale"})  # shape-check: fatal:no-conservative-reading
        constraint_id = _expect_item_id(item["constraint_id"], context=f"{item_context}.constraint_id")  # shape-check: fatal:no-conservative-reading
        if constraint_id in coupling_ids:
            raise AgentLoopError(f"{context}.coupling_constraints has duplicate ID `{constraint_id}`.")  # shape-check: fatal:no-conservative-reading
        ids = _expect_item_id_list(item["scope_item_ids"], context=f"{item_context}.scope_item_ids")  # shape-check: fatal:no-conservative-reading
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise AgentLoopError(f"{item_context}.scope_item_ids must contain at least two unique IDs.")  # shape-check: fatal:no-conservative-reading
        unknown = sorted(set(ids) - scope_ids)
        if unknown:
            raise AgentLoopError(f"{item_context}.scope_item_ids contains unknown IDs: {', '.join(unknown)}.")  # shape-check: fatal:no-conservative-reading
        coupling_ids.add(constraint_id)
        coupling_constraints.append(
            ExecutionCouplingConstraint(
                constraint_id=constraint_id,
                scope_item_ids=ids,
                rationale=_expect_non_empty_string(item["rationale"], context=f"{item_context}.rationale"),  # shape-check: fatal:no-conservative-reading
            )
        )

    child_payload = payload["child_stages"]
    if not isinstance(child_payload, list):
        raise AgentLoopError(f"{context}.child_stages must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    child_stages: list[ExecutionChildStage] = []
    stage_ids: set[str] = set()
    for index, raw_stage in enumerate(child_payload):
        item_context = f"{context}.child_stages[{index}]"
        item = _expect_object(raw_stage, context=item_context)  # shape-check: fatal:no-conservative-reading
        _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
            item,
            context=item_context,
            required={
                "stage_id", "position", "title", "summary", "deliverables", "non_goals",
                "acceptance_criteria", "depends_on_stage_ids", "dependency_notes", "automation",
                "rollout_risk", "compatibility_constraints", "covered_scope_item_ids",
            },
            optional={"execution_disposition"},
        )
        stage_id = _expect_item_id(item["stage_id"], context=f"{item_context}.stage_id")  # shape-check: fatal:no-conservative-reading
        if stage_id in stage_ids:
            raise AgentLoopError(f"{context}.child_stages has duplicate stage ID `{stage_id}`.")  # shape-check: fatal:no-conservative-reading
        position = _expect_int(item["position"], context=f"{item_context}.position")  # shape-check: fatal:no-conservative-reading
        if position != index + 1:
            raise AgentLoopError(f"{item_context}.position must be contiguous and ordered starting at 1.")  # shape-check: fatal:no-conservative-reading
        depends = _expect_item_id_list(item["depends_on_stage_ids"], context=f"{item_context}.depends_on_stage_ids")  # shape-check: fatal:no-conservative-reading
        unknown_dependencies = sorted(set(depends) - stage_ids)
        if unknown_dependencies:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{item_context}.depends_on_stage_ids must reference earlier stages; "
                f"unknown/later IDs: {', '.join(unknown_dependencies)}."
            )
        automation = _expect_non_empty_string(item["automation"], context=f"{item_context}.automation")  # shape-check: fatal:no-conservative-reading
        if automation not in EXECUTION_AUTOMATION_CLASSES:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{item_context}.automation must be one of: {', '.join(sorted(EXECUTION_AUTOMATION_CLASSES))}."
            )
        covered = _expect_item_id_list(item["covered_scope_item_ids"], context=f"{item_context}.covered_scope_item_ids")  # shape-check: fatal:no-conservative-reading
        if not covered:
            raise AgentLoopError(f"{item_context}.covered_scope_item_ids must be non-empty.")  # shape-check: fatal:no-conservative-reading
        unknown_coverage = sorted(set(covered) - scope_ids)
        if unknown_coverage:
            raise AgentLoopError(f"{item_context}.covered_scope_item_ids contains unknown IDs: {', '.join(unknown_coverage)}.")  # shape-check: fatal:no-conservative-reading
        stage_ids.add(stage_id)
        stage_non_goals = _expect_string_list(item["non_goals"], context=f"{item_context}.non_goals", item_context=f"{item_context}.non_goals")  # shape-check: fatal:no-conservative-reading
        stage_dependency_notes = _expect_non_empty_string(item["dependency_notes"], context=f"{item_context}.dependency_notes")  # shape-check: fatal:no-conservative-reading
        stage_compatibility = _expect_string_list(item["compatibility_constraints"], context=f"{item_context}.compatibility_constraints", item_context=f"{item_context}.compatibility_constraints")  # shape-check: fatal:no-conservative-reading
        execution_disposition: ExecutionDisposition | None = None
        if "execution_disposition" in item:
            execution_disposition = _expect_execution_disposition(  # shape-check: fatal:no-conservative-reading
                item["execution_disposition"],
                context=f"{item_context}.execution_disposition",
                automation=automation,
                non_goals=stage_non_goals,
                compatibility_constraints=stage_compatibility,
                dependency_notes=stage_dependency_notes,
            )
        elif require_child_dispositions:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{item_context}.execution_disposition is required for every child stage in a "
                "fresh generation-1 response: declare `direct-implementation` or "
                "`requires-child-planning` for agent-pr stages and `human-owned` for "
                "human-action/manual-close stages."
            )
        child_stages.append(
            ExecutionChildStage(
                stage_id=stage_id,
                position=position,
                title=_expect_non_empty_string(item["title"], context=f"{item_context}.title"),  # shape-check: fatal:no-conservative-reading
                summary=_expect_non_empty_string(item["summary"], context=f"{item_context}.summary"),  # shape-check: fatal:no-conservative-reading
                deliverables=_expect_string_list(item["deliverables"], context=f"{item_context}.deliverables", item_context=f"{item_context}.deliverables", min_length=1),  # shape-check: fatal:no-conservative-reading
                non_goals=stage_non_goals,
                acceptance_criteria=_expect_string_list(item["acceptance_criteria"], context=f"{item_context}.acceptance_criteria", item_context=f"{item_context}.acceptance_criteria", min_length=1),  # shape-check: fatal:no-conservative-reading
                depends_on_stage_ids=depends,
                dependency_notes=stage_dependency_notes,
                automation=automation,
                rollout_risk=_expect_non_empty_string(item["rollout_risk"], context=f"{item_context}.rollout_risk"),  # shape-check: fatal:no-conservative-reading
                compatibility_constraints=stage_compatibility,
                covered_scope_item_ids=covered,
                execution_disposition=execution_disposition,
            )
        )

    retained = _expect_execution_allocation(payload["retained_parent_work"], context=f"{context}.retained_parent_work")  # shape-check: fatal:no-conservative-reading
    final = _expect_execution_allocation(payload["final_integration_work"], context=f"{context}.final_integration_work")  # shape-check: fatal:no-conservative-reading
    caveats = _expect_string_list(payload["caveats"], context=f"{context}.caveats", item_context=f"{context}.caveats")  # shape-check: fatal:no-conservative-reading
    one_shot: ExecutionOneShotDelivery | None = None
    if strategy == "one-shot":
        if "one_shot_delivery" not in payload:
            raise AgentLoopError(f"{context}.one_shot_delivery is required for one-shot recommendations.")  # shape-check: fatal:no-conservative-reading
        if child_stages:
            raise AgentLoopError("one-shot recommendations must not contain child stages.")  # shape-check: fatal:no-conservative-reading
        if retained.status != "none" or final.status != "none":
            raise AgentLoopError("one-shot recommendations require none retained-parent and final-integration work.")  # shape-check: fatal:no-conservative-reading
        delivery = _expect_object(payload["one_shot_delivery"], context=f"{context}.one_shot_delivery")  # shape-check: fatal:no-conservative-reading
        _expect_exact_keys(delivery, context=f"{context}.one_shot_delivery", required={"deliverables", "acceptance_criteria", "covered_scope_item_ids"})  # shape-check: fatal:no-conservative-reading
        one_shot = ExecutionOneShotDelivery(
            deliverables=_expect_string_list(delivery["deliverables"], context=f"{context}.one_shot_delivery.deliverables", item_context=f"{context}.one_shot_delivery.deliverables", min_length=1),  # shape-check: fatal:no-conservative-reading
            acceptance_criteria=_expect_string_list(delivery["acceptance_criteria"], context=f"{context}.one_shot_delivery.acceptance_criteria", item_context=f"{context}.one_shot_delivery.acceptance_criteria", min_length=1),  # shape-check: fatal:no-conservative-reading
            covered_scope_item_ids=_expect_item_id_list(delivery["covered_scope_item_ids"], context=f"{context}.one_shot_delivery.covered_scope_item_ids"),  # shape-check: fatal:no-conservative-reading
        )
        allocations = [one_shot.covered_scope_item_ids]
    else:
        if "one_shot_delivery" in payload:
            raise AgentLoopError("staged recommendations must not contain one_shot_delivery.")  # shape-check: fatal:no-conservative-reading
        if feasibility != "safe":
            raise AgentLoopError("staged recommendations require staging_feasibility `safe`.")  # shape-check: fatal:no-conservative-reading
        if not child_stages:
            raise AgentLoopError("staged recommendations require at least one child stage.")  # shape-check: fatal:no-conservative-reading
        required_allocations = sum(
            allocation.status == "required" for allocation in (retained, final)
        ) + sum(bool(stage.covered_scope_item_ids) for stage in child_stages)
        if required_allocations < 2:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                "staged recommendations require at least two real delivery allocations; "
                "recommend `one-shot` when one child has no retained or final integration work."
            )
        allocations = [stage.covered_scope_item_ids for stage in child_stages]
        allocations.extend(
            allocation.covered_scope_item_ids
            for allocation in (retained, final)
            if allocation.status == "required"
        )

    allocation_owner: dict[str, int] = {}
    for allocation_index, covered_ids in enumerate(allocations):
        for scope_id in covered_ids:
            if scope_id not in scope_ids:
                raise AgentLoopError(f"{context} covers unknown scope item `{scope_id}`.")  # shape-check: fatal:no-conservative-reading
            if scope_id in allocation_owner:
                raise AgentLoopError(f"{context} covers scope item `{scope_id}` more than once.")  # shape-check: fatal:no-conservative-reading
            allocation_owner[scope_id] = allocation_index
    missing = sorted(scope_ids - set(allocation_owner))
    if missing:
        raise AgentLoopError(f"{context} leaves scope items uncovered: {', '.join(missing)}.")  # shape-check: fatal:no-conservative-reading
    for constraint in coupling_constraints:
        owners = {allocation_owner[item_id] for item_id in constraint.scope_item_ids}
        if len(owners) != 1:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{context}.{constraint.constraint_id} splits coupled scope items across allocations."
            )
    return ExecutionStrategyRecommendation(
        strategy=strategy,
        rationale=rationale,
        staging_feasibility=feasibility,
        scope_items=tuple(scope_items),
        coupling_constraints=tuple(coupling_constraints),
        one_shot_delivery=one_shot,
        child_stages=tuple(child_stages),
        retained_parent_work=retained,
        final_integration_work=final,
        caveats=caveats,
    )


def _parse_execution_contract_fields(
    payload: dict[str, object],
    *,
    context: str,
    required: bool,
    require_child_dispositions: bool = False,
) -> tuple[int | None, ExecutionStrategyRecommendation | None]:
    has_version = "execution_strategy_contract_version" in payload
    has_recommendation = "execution_recommendation" in payload
    if has_version != has_recommendation:
        raise AgentLoopError(  # shape-check: fatal:kind-or-version-mismatch
            f"{context} must include execution_strategy_contract_version and "
            "execution_recommendation together."
        )
    if not has_version:
        if required:
            raise AgentLoopError(  # shape-check: fatal:kind-or-version-mismatch
                f"Fresh {context} responses require execution_strategy_contract_version: 1 "
                "and a complete execution_recommendation."
            )
        return None, None
    version = _expect_int(payload["execution_strategy_contract_version"], context=f"{context}.execution_strategy_contract_version")  # shape-check: fatal:kind-or-version-mismatch
    if version != EXECUTION_STRATEGY_CONTRACT_VERSION:
        raise AgentLoopError(f"{context}.execution_strategy_contract_version must be 1.")  # shape-check: fatal:kind-or-version-mismatch
    recommendation = _expect_execution_recommendation(  # shape-check: fatal:no-conservative-reading
        payload["execution_recommendation"],
        context=f"{context}.execution_recommendation",
        require_child_dispositions=require_child_dispositions,
    )
    # Generation 1 has one reviewed topology. The legacy top-level
    # ``child_stages`` category is executable only for unversioned historical
    # plans; allowing it beside a v1 recommendation would create two
    # competing topologies and let a legacy splitter mutate a fresh plan.
    # Keep an explicitly empty legacy category harmless for callers that still
    # emit the shared optional key.
    if "child_stages" in payload and payload["child_stages"] != []:
        raise AgentLoopError(  # shape-check: fatal:kind-or-version-mismatch
            f"{context} cannot combine a generation-1 execution recommendation "
            "with non-empty top-level legacy child_stages; use only the reviewed "
            "execution_recommendation topology."
        )
    return version, recommendation


def parse_execution_recommendation_payload(
    value: object, *, context: str = "execution_recommendation"
) -> ExecutionStrategyRecommendation:
    """Validate a recovered recommendation before any bounded repair.

    Recovery parsing tolerates an absent per-child ``execution_disposition``
    (legacy-ambiguous) so plans approved before #808 keep their recorded
    recommendation digest; fresh planner turns require it.
    """
    return _expect_execution_recommendation(value, context=context)  # shape-check: fatal:no-conservative-reading


def _expect_state(value: object, *, context: str) -> str:
    state = _expect_non_empty_string(value, context=context)  # shape-check: fatal:no-conservative-reading
    if state not in {"approved", "blocking"}:
        raise AgentLoopError(f"{context} must be `approved` or `blocking`.")  # shape-check: fatal:no-conservative-reading
    return state


def _expect_item_id(value: object, *, context: str) -> str:
    item_id = _expect_non_empty_string(value, context=context)  # shape-check: fatal:no-conservative-reading
    if not ITEM_ID_RE.fullmatch(item_id):
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            f"{context} must match `[A-Za-z0-9][A-Za-z0-9._-]*`."
        )
    return item_id


def _expect_item_id_list(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    return tuple(
        _expect_item_id(item, context=f"{context} item at index {index}")  # shape-check: fatal:no-conservative-reading
        for index, item in enumerate(value)
    )


def _expect_item_note_map(
    value: object,
    *,
    context: str,
    allowed_item_ids: set[str],
    allowed_context: str,
) -> dict[str, str]:
    note_payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    rendered: dict[str, str] = {}
    for raw_item_id, raw_note in note_payload.items():
        item_id = _expect_item_id(raw_item_id, context=f"{context} key")  # shape-check: fatal:no-conservative-reading
        if item_id not in allowed_item_ids:
            raise AgentLoopError(f"{context} key `{item_id}` is not listed in {allowed_context}.")  # shape-check: fatal:no-conservative-reading
        rendered[item_id] = _expect_non_empty_string(raw_note, context=f"{context}.{item_id}")  # shape-check: fatal:no-conservative-reading
    return rendered


def _expect_requirement_id_list(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    rendered: list[str] = []
    for index, item in enumerate(value):
        label = _expect_non_empty_string(item, context=f"{context} item at index {index}")  # shape-check: fatal:no-conservative-reading
        rendered.append(_normalize_requirement_label(label))  # shape-check: fatal:no-conservative-reading
    return tuple(rendered)


def _expect_human_requirement_dispositions(
    value: object,
    *,
    context: str,
    alias_surfaced_requirement_ids: Sequence[str] | None = None,
) -> tuple[HumanRequirementDisposition, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    alias_ids = frozenset(alias_surfaced_requirement_ids or ())
    result: list[HumanRequirementDisposition] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        payload = _expect_object(item, context=item_context)  # shape-check: fatal:no-conservative-reading
        if alias_ids:
            for alias, canonical in HUMAN_REQUIREMENT_DISPOSITION_KEY_ALIASES.items():
                alias_value = payload.get(alias)
                if (
                    alias in payload
                    and canonical not in payload
                    and isinstance(alias_value, str)
                    and alias_value in alias_ids
                    and HUMAN_REQUIREMENT_STABLE_ID_RE.fullmatch(alias_value)
                ):
                    payload = {
                        (canonical if key == alias else key): val
                        for key, val in payload.items()
                    }
                    _logger.warning(
                        "%s: normalized disposition key alias %s -> %s",
                        item_context,
                        alias,
                        canonical,
                    )
        _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
            payload,
            context=item_context,
            required={"requirement_id", "disposition", "evidence"},
        )
        requirement_id = _normalize_requirement_label(  # shape-check: fatal:no-conservative-reading
            _expect_non_empty_string(payload["requirement_id"], context=f"{item_context}.requirement_id")  # shape-check: fatal:no-conservative-reading
        )
        disposition = _expect_non_empty_string(  # shape-check: fatal:no-conservative-reading
            payload["disposition"], context=f"{item_context}.disposition"
        )
        if disposition not in HUMAN_REQUIREMENT_DISPOSITION_VALUES:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{item_context}.disposition must be one of: addressed, blocked, not-applicable"
            )
        evidence = _expect_non_empty_string(  # shape-check: fatal:no-conservative-reading
            payload["evidence"], context=f"{item_context}.evidence"
        )
        result.append(HumanRequirementDisposition(requirement_id, disposition, evidence))
    return tuple(result)


def validate_human_requirement_dispositions(
    dispositions: Sequence[HumanRequirementDisposition],
    *,
    surfaced_requirement_ids: Sequence[str],
    context: str = "human_requirement_dispositions",
) -> None:
    expected = tuple(_normalize_requirement_label(item) for item in surfaced_requirement_ids)  # shape-check: fatal:no-conservative-reading
    actual = [item.requirement_id for item in dispositions]
    _reject_legacy_requirement_labels(actual, surfaced_requirement_ids=surfaced_requirement_ids)  # shape-check: fatal:no-conservative-reading
    duplicates = sorted({item for item in actual if actual.count(item) > 1})
    if duplicates:
        raise AgentLoopError(f"{context} contains duplicate requirement ID(s): {', '.join(duplicates)}")  # shape-check: fatal:no-conservative-reading
    unknown = sorted(set(actual) - set(expected))
    if unknown:
        raise AgentLoopError(f"{context} contains unknown requirement ID(s): {', '.join(unknown)}")  # shape-check: fatal:no-conservative-reading
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise AgentLoopError(f"{context} is missing requirement ID(s): {', '.join(missing)}")  # shape-check: fatal:no-conservative-reading
    if not expected and actual:
        raise AgentLoopError(f"{context} must be empty when no signed human requirements are surfaced.")  # shape-check: fatal:no-conservative-reading


def _consume_agent_unavailable_footer_and_signature(
    *, payload: dict[str, object], trailing: str
) -> dict[str, object]:
    trailing = trailing.lstrip()
    marker_match = _STANDALONE_AGENT_UNAVAILABLE_RE.search(trailing)
    if marker_match is None:
        raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
            "Structured agent-unavailable response must place <!-- AGENT_UNAVAILABLE --> "
            "after the JSON object."
        )
    if trailing[: marker_match.start()].strip():
        raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
            "Structured agent-unavailable response may not include prose between the JSON object "
            "and the AGENT_UNAVAILABLE footer."
        )
    signature = trailing[marker_match.end() :].lstrip()
    signature_match = re.match(r"^--\s+\S[^\n]*(?:\n)?$", signature)
    if signature_match is None or signature[signature_match.end() :].strip():
        raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
            "Structured agent-unavailable response may not include trailing prose after "
            "the footer and signature."
        )
    return payload


def parse_agent_unavailable(text: str) -> AgentUnavailable | None:
    """Parse the shared, terminal agent-unavailable response envelope."""
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)  # shape-check: fatal:unparseable-envelope
    if extracted is None:
        return None
    payload, trailing = extracted
    if payload.get("kind") != "agent_unavailable":
        return None
    _consume_agent_unavailable_footer_and_signature(payload=payload, trailing=trailing)  # shape-check: fatal:unparseable-envelope
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    _expect_exact_keys(  # shape-check: fatal:unparseable-envelope
        payload,
        context="agent_unavailable",
        required={
            "schema_version",
            "kind",
            "retryable",
            "category",
            "summary",
            "suggested_action",
        },
    )
    category = _expect_non_empty_string(  # shape-check: fatal:unparseable-envelope
        payload["category"], context="agent_unavailable.category"
    )
    if category not in AGENT_UNAVAILABLE_CATEGORIES:
        allowed = ", ".join(sorted(AGENT_UNAVAILABLE_CATEGORIES))
        raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
            f"agent_unavailable.category must be one of: {allowed}."
        )
    return AgentUnavailable(
        schema_version=_expect_int(payload["schema_version"], context="schema_version"),  # shape-check: fatal:unparseable-envelope
        kind="agent_unavailable",
        retryable=_expect_bool(payload["retryable"], context="agent_unavailable.retryable"),  # shape-check: fatal:unparseable-envelope
        category=category,
        summary=_expect_non_empty_string(payload["summary"], context="agent_unavailable.summary"),  # shape-check: fatal:unparseable-envelope
        suggested_action=_expect_non_empty_string(  # shape-check: fatal:unparseable-envelope
            payload["suggested_action"], context="agent_unavailable.suggested_action"
        ),
    )


def _extract_structured_pr_review_payload(text: str) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)  # shape-check: fatal:unparseable-envelope
    if extracted is None:
        return None
    payload, trailing = extracted
    trailing = trailing.lstrip()
    if HUMAN_REQUIREMENTS_RESOLVED_RE.match(trailing):
        marker_match = HUMAN_REQUIREMENTS_RESOLVED_RE.match(trailing)
        assert marker_match is not None
        trailing = trailing[marker_match.end() :]
    return _consume_structured_footer_and_signature(  # shape-check: fatal:unparseable-envelope
        payload=payload,
        trailing=trailing,
        state_re=STATE_RE,
        state_marker_name="AGENT_STATE",
        context_label="Structured PR review",
        allow_human_requirements_marker_after_footer=True,
    )


def _extract_structured_plan_review_payload(text: str) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)  # shape-check: fatal:unparseable-envelope
    if extracted is None:
        return None
    payload, trailing = extracted
    trailing = trailing.lstrip()
    if HUMAN_REQUIREMENTS_RESOLVED_RE.match(trailing):
        marker_match = HUMAN_REQUIREMENTS_RESOLVED_RE.match(trailing)
        assert marker_match is not None
        trailing = trailing[marker_match.end() :]
    return _consume_structured_footer_and_signature(  # shape-check: fatal:unparseable-envelope
        payload=payload,
        trailing=trailing,
        state_re=PLAN_STATE_RE,
        state_marker_name="AGENT_PLAN_STATE",
        context_label="Structured plan review",
    )


def _extract_structured_plan_revision_payload(text: str) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)  # shape-check: fatal:unparseable-envelope
    if extracted is None:
        return None
    payload, trailing = extracted
    return _consume_structured_footer_and_signature(  # shape-check: fatal:unparseable-envelope
        payload=payload,
        trailing=trailing,
        state_re=PLAN_STATE_RE,
        state_marker_name="AGENT_PLAN_STATE",
        context_label="Structured plan revision",
        allow_human_requirements_prefix=True,
    )


def _extract_structured_plan_state_payload(text: str) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)  # shape-check: fatal:unparseable-envelope
    if extracted is None:
        return None
    payload, trailing = extracted
    return _consume_structured_footer_and_signature(  # shape-check: fatal:unparseable-envelope
        payload=payload,
        trailing=trailing,
        state_re=PLAN_STATE_RE,
        state_marker_name="AGENT_PLAN_STATE",
        context_label="Structured plan state",
        allow_human_requirements_prefix=True,
    )


def _extract_structured_coder_followup_payload(text: str) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)  # shape-check: fatal:unparseable-envelope
    if extracted is None:
        return None
    payload, trailing = extracted
    return _consume_structured_footer_and_signature(  # shape-check: fatal:unparseable-envelope
        payload=payload,
        trailing=trailing,
        state_re=STATE_RE,
        state_marker_name="AGENT_STATE",
        context_label="Structured coder follow-up",
    )


def _extract_structured_issue_implementation_payload(
    text: str,
) -> dict[str, object] | None:
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)  # shape-check: fatal:unparseable-envelope
    if extracted is None:
        return None
    payload, trailing = extracted
    return _consume_structured_footer_and_signature(  # shape-check: fatal:unparseable-envelope
        payload=payload,
        trailing=trailing,
        state_re=STATE_RE,
        state_marker_name="AGENT_STATE",
        context_label="Structured issue implementation",
    )


def _parse_review_item_disposition_payload(
    value: object,
    *,
    field_name: str,
    reviewer: str,
    allowed_same_status: str,
    is_plan_review: bool,
    degradations: list[ParseDegradation] | None = None,
) -> ReviewItemDisposition:
    payload = _expect_object(value, context=field_name)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload,
        context=field_name,
        required={"item_id", "disposition"},
        optional={"note"} if is_plan_review else {"note", "sub_item_dispositions", "evidence_requirement_ids"},
    )
    item_id = _expect_item_id(payload["item_id"], context=f"{field_name}.item_id")  # shape-check: fatal:no-conservative-reading
    disposition = _expect_non_empty_string(payload["disposition"], context=f"{field_name}.disposition")  # shape-check: fatal:no-conservative-reading
    allowed_statuses = {"resolved", "blocking", allowed_same_status, "future"}
    if disposition not in allowed_statuses:
        rendered = ", ".join(sorted(allowed_statuses))
        raise AgentLoopError(f"{field_name}.disposition must be one of: {rendered}")  # shape-check: fatal:no-conservative-reading
    note_value = payload.get("note")
    note = None
    if note_value is not None:
        note = _expect_non_empty_string(note_value, context=f"{field_name}.note")  # shape-check: fatal:no-conservative-reading
    if _active_disposition_has_empty_note(
        disposition,
        note,
        same_status=allowed_same_status,
        is_plan_review=is_plan_review,
    ):
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            f"{field_name}.note cannot be an empty placeholder for active disposition `{disposition}`."
        )
    sub_item_dispositions: tuple[tuple[str, str], ...] = ()
    if "sub_item_dispositions" in payload:
        sub_item_dispositions = _degradable_sub_item_dispositions(  # shape-check: fatal:authentication-or-forgery
            payload["sub_item_dispositions"],
            context=f"{field_name}.sub_item_dispositions",
            degradations=degradations,
        )
    from .native_evidence import citation_ids
    return ReviewItemDisposition(
        evidence_requirement_ids=citation_ids(payload.get("evidence_requirement_ids", [])),
        item_id=item_id,
        reviewer=reviewer,
        disposition=disposition,
        note=note,
        sub_item_dispositions=sub_item_dispositions,
    )


def _degradable_sub_item_dispositions(
    value: object,
    *,
    context: str,
    degradations: list[ParseDegradation] | None,
) -> tuple[tuple[str, str], ...]:
    """Read a sub-item disposition map, dropping malformed pairs (#958)."""
    pairs: list[tuple[str, str]] = []
    defects: list[tuple[str, object, str]] = []
    if not isinstance(value, dict):
        defects.append(("sub_item_dispositions-not-an-object", value, context))
    else:
        for key, entry in value.items():
            if not isinstance(key, str) or not key.strip():
                defects.append(("sub_item_dispositions-key-not-a-string", key, context))
            elif not isinstance(entry, str) or entry not in SUB_ITEM_DISPOSITIONS:
                defects.append(
                    ("sub_item_dispositions-value-not-resolved-or-unresolved", entry, f"{context}.{key}")
                )
            else:
                pairs.append((key.strip(), entry))
    if degradations is not None:
        for rule, observed, path in defects[:PARSE_DEGRADATION_RENDER_LIMIT]:
            degradations.append(
                ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
                    element_path=path, rule=rule, observed=observed, outcome="claim-dropped"
                )
            )
    return tuple(pairs)


def _expect_disposition_list(
    value: object,
    *,
    context: str,
    reviewer: str,
    allowed_same_status: str,
    is_plan_review: bool,
    degradations: list[ParseDegradation] | None = None,
) -> tuple[ReviewItemDisposition, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")  # shape-check: fatal:no-conservative-reading
    return tuple(
        _parse_review_item_disposition_payload(  # shape-check: fatal:no-conservative-reading
            item,
            field_name=f"{context}[{index}]",
            reviewer=reviewer,
            allowed_same_status=allowed_same_status,
            is_plan_review=is_plan_review,
            degradations=degradations,
        )
        for index, item in enumerate(value)
    )


def _check_architecture_status_mode(mode: str) -> str:
    if mode not in ARCHITECTURE_STATUS_MODES:
        raise AgentLoopError(f"Unknown architecture_status_mode {mode!r}.")  # shape-check: fatal:orchestrator-authored
    return mode


def _degradable_response_impact(
    payload: Mapping[str, object], *, context: str, mode: str = "strict"
) -> tuple[ArchitectureImpact | None, tuple[ParseDegradation, ...]]:
    """Read an optional response assessment in the caller's explicit mode.

    Only `degradable` can produce a record or `undetermined`; `strict` accepts
    exactly `changed`/`unchanged`, and `legacy` is the stored-text decode.
    """
    _check_architecture_status_mode(mode)  # shape-check: fatal:orchestrator-authored
    if "architecture_impact" not in payload:
        return None, ()
    if mode != "degradable":
        return _parse_architecture_impact(  # shape-check: fatal:no-conservative-reading
            payload["architecture_impact"], context=context, architecture_status_mode=mode
        ), ()
    impact, record = _parse_architecture_impact_degradable(payload["architecture_impact"], context=context)  # shape-check: fatal:no-conservative-reading
    return impact, (() if record is None else (record,))


def _finalize_parsed_review(
    *,
    state: str,
    summary: str,
    blocking_items: tuple[ApprovedFollowup, ...],
    followups: ApprovedFollowups,
    dispositions: tuple[ReviewItemDisposition, ...],
    raw_dispositions_text: str = "",
    architecture_impact: ArchitectureImpact | None = None,
    architecture_impact_degradations: tuple[ParseDegradation, ...] = (),
    sub_item_degradations: tuple[ParseDegradation, ...] = (),
) -> ParsedReview:
    # Issue #1389: blocking reviews may carry off-scope Future follow-ups and
    # downgrade carried items to `future`; those never block and are published
    # at approval exactly like Future follow-ups from approved reviews.
    if state == "approved":
        active_dispositions = [
            item.disposition for item in dispositions if item.disposition in {"blocking", "same-pr"}
        ]
        if blocking_items or followups.same_pr or active_dispositions:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
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
        architecture_impact=architecture_impact,
        architecture_impact_degradations=architecture_impact_degradations,
        sub_item_degradations=sub_item_degradations,
    )


def _finalize_parsed_plan_review(
    *,
    state: str,
    summary: str,
    items: PlanReviewItems,
    dispositions: tuple[ReviewItemDisposition, ...],
    raw_dispositions_text: str = "",
    human_requirement_dispositions: tuple[HumanRequirementDisposition, ...] = (),
    architecture_impact: ArchitectureImpact | None = None,
    architecture_impact_degradations: tuple[ParseDegradation, ...] = (),
) -> ParsedPlanReview:
    # Issue #1389: blocking plan reviews keep off-scope Future follow-ups and
    # may downgrade carried items to `future`.
    if state == "approved":
        active_dispositions = [
            item.disposition for item in dispositions if item.disposition in {"blocking", "same-plan"}
        ]
        if items.blocking or items.same_plan or active_dispositions:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
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
        human_requirement_dispositions=human_requirement_dispositions,
        architecture_impact=architecture_impact,
        architecture_impact_degradations=architecture_impact_degradations,
    )


def _structured_followups(
    items: tuple[str, ...],
    *,
    reviewer: str,
    provenances: Sequence[str | None] = (),
) -> tuple[ApprovedFollowup, ...]:
    return tuple(
        ApprovedFollowup(
            reviewer=reviewer,
            text=item,
            provenance=provenances[index] if index < len(provenances) else None,
        )
        for index, item in enumerate(items)
    )


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


def parse_structured_pr_review(
    text: str, *, reviewer: str, architecture_status_mode: str = "strict"
) -> ParsedReview | None:
    payload = _extract_structured_pr_review_payload(text)  # shape-check: fatal:unparseable-envelope
    if payload is None:
        return None
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "pr_review":
        raise AgentLoopError("Structured response kind mismatch: expected `pr_review`.")  # shape-check: fatal:unparseable-envelope
    _expect_exact_keys(  # shape-check: fatal:unparseable-envelope
        payload,
        context="pr_review",
        required={
            "schema_version",
            "kind",
            "state",
            "summary",
            "prior_item_dispositions",
        },
        optional={
            "blocking_items",
            "same_pr_followups",
            "future_followups",
            "architecture_impact",
            "exact_head_evidence_requests", "screenshot_requests", "screenshot_coverage_acceptances",
            "post_merge_obligations",
        },
    )
    state = _expect_state(payload["state"], context="pr_review.state")  # shape-check: fatal:no-conservative-reading
    summary = review_freeform_summary_text(
        _expect_non_empty_string(payload["summary"], context="pr_review.summary")  # shape-check: fatal:no-conservative-reading
    )
    sub_item_degradations: list[ParseDegradation] = []
    blocking_items = _expect_review_finding_list(  # shape-check: fatal:no-conservative-reading
        payload, "blocking_items", context="pr_review.blocking_items", reviewer=reviewer,
        degradations=sub_item_degradations, allow_evidence_row_ids=True,
    )
    same_pr_followups = _expect_review_finding_list(  # shape-check: fatal:no-conservative-reading
        payload, "same_pr_followups", context="pr_review.same_pr_followups", reviewer=reviewer,
        degradations=sub_item_degradations,
    )
    future_followups = _expect_review_finding_list(  # shape-check: fatal:no-conservative-reading
        payload, "future_followups", context="pr_review.future_followups", reviewer=reviewer,
        degradations=sub_item_degradations, allow_sub_items=False,
    )
    dispositions = _expect_disposition_list(  # shape-check: fatal:no-conservative-reading
        payload["prior_item_dispositions"],
        context="pr_review.prior_item_dispositions",
        reviewer=reviewer,
        allowed_same_status="same-pr",
        is_plan_review=False,
        degradations=sub_item_degradations,
    )
    architecture_impact, architecture_impact_degradations = _degradable_response_impact(  # shape-check: fatal:no-conservative-reading
        payload, mode=architecture_status_mode, context="pr_review.architecture_impact"
    )
    evidence_requests = _expect_evidence_request_list(  # shape-check: fatal:no-conservative-reading
        payload, context="pr_review.exact_head_evidence_requests"
    )
    _reject_evidence_request_overlap(  # shape-check: fatal:no-conservative-reading
        evidence_requests,
        blocking_items=blocking_items,
        same_pr_followups=same_pr_followups,
        future_followups=future_followups,
    )
    structured_blocking_items = blocking_items
    followups = _dedupe_pr_review_items(
        structured_blocking_items,
        ApprovedFollowups(
            same_pr=same_pr_followups,
            future=future_followups,
        ),
    )
    parsed = _finalize_parsed_review(  # shape-check: fatal:no-conservative-reading
        state=state,
        summary=summary,
        blocking_items=structured_blocking_items,
        followups=followups,
        dispositions=dispositions,
        architecture_impact=architecture_impact,
        architecture_impact_degradations=architecture_impact_degradations,
        sub_item_degradations=tuple(sub_item_degradations),
    )
    from .post_merge_obligations import parse_review_fields
    obligations = parse_review_fields(payload)
    from .review_images_coverage import parse_fields
    typed, votes = parse_fields(payload, legacy=evidence_requests, findings=(*blocking_items, *same_pr_followups, *future_followups))
    return dataclasses.replace(parsed, exact_head_evidence_requests=evidence_requests,
                               screenshot_requests=typed, screenshot_coverage_acceptances=votes,
                               post_merge_obligations=obligations)


def _expect_evidence_request_list(
    payload: dict[str, object], *, context: str
) -> tuple[str, ...]:
    """Strictly parse the optional human-only exact-head evidence list (#1068)."""
    value = payload.get("exact_head_evidence_requests", [])
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array of strings.")  # shape-check: fatal:no-conservative-reading
    requests: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, str):
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{context} at index {index} must be a non-empty string."
            )
        text = _expect_non_empty_string(raw, context=f"{context} at index {index}")  # shape-check: fatal:no-conservative-reading
        normalized = _normalized_review_item_text(text)
        if not normalized:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{context} at index {index} must be a non-empty string."
            )
        if normalized in seen:
            continue
        seen.add(normalized)
        requests.append(text.strip())
    return tuple(requests)


def _reject_evidence_request_overlap(
    evidence_requests: tuple[str, ...],
    *,
    blocking_items: tuple[ApprovedFollowup, ...],
    same_pr_followups: tuple[ApprovedFollowup, ...],
    future_followups: tuple[ApprovedFollowup, ...],
) -> None:
    """Reject a request that is also listed as a coder-routed finding.

    Silently keeping either copy would be wrong: dropping the typed request
    loses the final barrier, and dropping the finding hides a code defect.
    The bounded format repair asks the reviewer to choose exactly one field.
    """
    if not evidence_requests:
        return
    finding_texts = {
        _normalized_review_item_text(item.text): field
        for field, group in (
            ("future_followups", future_followups),
            ("same_pr_followups", same_pr_followups),
            ("blocking_items", blocking_items),
        )
        for item in group
    }
    for request in evidence_requests:
        field = finding_texts.get(_normalized_review_item_text(request))
        if field is not None:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                "pr_review.exact_head_evidence_requests repeats an entry that is also listed "
                f"in `{field}`. List human-only exact-head evidence only in "
                "`exact_head_evidence_requests` and code defects only in the finding lists; "
                "each concern belongs in exactly one field."
            )


def parse_structured_plan_review(
    text: str,
    *,
    reviewer: str,
    architecture_status_mode: str = "strict",
    surfaced_requirement_ids: Sequence[str] | None = None,
) -> ParsedPlanReview | None:
    payload = _extract_structured_plan_review_payload(text)  # shape-check: fatal:unparseable-envelope
    if payload is None:
        return None
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "plan_review":
        raise AgentLoopError("Structured response kind mismatch: expected `plan_review`.")  # shape-check: fatal:unparseable-envelope
    _expect_exact_keys(  # shape-check: fatal:unparseable-envelope
        payload,
        context="plan_review",
        required={
            "schema_version",
            "kind",
            "state",
            "summary",
            "prior_plan_item_dispositions",
        },
        optional={"blocking_plan_issues", "same_plan_followups", "future_followups", "human_requirement_dispositions", "architecture_impact"},
    )
    state = _expect_state(payload["state"], context="plan_review.state")  # shape-check: fatal:no-conservative-reading
    summary = review_freeform_summary_text(
        _expect_non_empty_string(payload["summary"], context="plan_review.summary")  # shape-check: fatal:no-conservative-reading
    )
    blocking_items = _expect_plan_review_finding_list(  # shape-check: fatal:no-conservative-reading
        payload,
        "blocking_plan_issues",
        context="plan_review.blocking_plan_issues",
    )
    same_plan_followups = _expect_plan_review_finding_list(  # shape-check: fatal:no-conservative-reading
        payload,
        "same_plan_followups",
        context="plan_review.same_plan_followups",
    )
    future_followups = _expect_plan_review_finding_list(  # shape-check: fatal:no-conservative-reading
        payload,
        "future_followups",
        context="plan_review.future_followups",
    )
    dispositions = _expect_disposition_list(  # shape-check: fatal:no-conservative-reading
        payload["prior_plan_item_dispositions"],
        context="plan_review.prior_plan_item_dispositions",
        reviewer=reviewer,
        allowed_same_status="same-plan",
        is_plan_review=True,
    )
    human_requirement_dispositions = _expect_human_requirement_dispositions(  # shape-check: fatal:no-conservative-reading
        payload.get("human_requirement_dispositions", []),
        context="plan_review.human_requirement_dispositions",
        alias_surfaced_requirement_ids=surfaced_requirement_ids,
    )
    architecture_impact, architecture_impact_degradations = _degradable_response_impact(  # shape-check: fatal:no-conservative-reading
        payload, mode=architecture_status_mode, context="plan_review.architecture_impact"
    )
    provenances: dict[str, tuple[str | None, ...]] = {}
    provenance_degradations: list[ParseDegradation] = []
    for field_name in ("blocking_plan_issues", "same_plan_followups", "future_followups"):
        tags, records = _plan_review_finding_provenances(  # shape-check: fatal:authentication-or-forgery
            payload, field_name, context=f"plan_review.{field_name}"
        )
        provenances[field_name] = tags
        provenance_degradations.extend(records)
    for record in provenance_degradations:
        _logger.warning(
            "%s: dropped a malformed finding provenance tag (%s)",
            record.element_path,
            record.rule,
        )
    items = _dedupe_plan_review_items(
        PlanReviewItems(
            blocking=_structured_followups(
                blocking_items, reviewer=reviewer,
                provenances=provenances["blocking_plan_issues"],
            ),
            same_plan=_structured_followups(
                same_plan_followups, reviewer=reviewer,
                provenances=provenances["same_plan_followups"],
            ),
            future=_structured_followups(
                future_followups, reviewer=reviewer,
                provenances=provenances["future_followups"],
            ),
        )
    )
    parsed = _finalize_parsed_plan_review(  # shape-check: fatal:no-conservative-reading
        state=state,
        summary=summary,
        items=items,
        dispositions=dispositions,
        human_requirement_dispositions=human_requirement_dispositions,
        architecture_impact=architecture_impact,
        architecture_impact_degradations=architecture_impact_degradations,
    )
    if provenance_degradations:
        parsed = dataclasses.replace(
            parsed, provenance_degradations=tuple(provenance_degradations)
        )
    return parsed


def validate_structured_coder_followup(
    text: str, *, required_architecture_impact_contract: int = 0,
    delivered_risk_test_matrix: RiskTestMatrix | Mapping[str, object] | None = None,
    delivered_risk_test_matrix_identity: str | None = None,
    required_risk_test_matrix_contract: int = 0,
    authoritative_test_observations: Sequence[object] | None = None,
    delivered_risk_test_matrix_row_ids: Sequence[str] | None = None,
    execution_catalog: Sequence[object] | None = None,
    allow_historical_canonical_evidence: bool = False,
    architecture_status_mode: str = "strict",
    expected_sub_item_ids: Mapping[str, Sequence[str]] | None = None,
    test_observations_mode: str = "degrade",
) -> StructuredCoderFollowup | None:
    payload = _extract_structured_coder_followup_payload(text)  # shape-check: fatal:unparseable-envelope
    if payload is None:
        return None
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "coder_followup":
        raise AgentLoopError("Structured response kind mismatch: expected `coder_followup`.")  # shape-check: fatal:unparseable-envelope
    optional_fields = {
        "addressed_item_notes",
        "remaining_item_notes",
        "tests_run",
        "test_observations",
        "risk_test_matrix_claims",
        "risk_test_matrix_coverage_gaps",
        "disputed_items",
        "dispute_evidence",
        "architecture_impact",
        "addressed_sub_items",
    }
    if allow_historical_canonical_evidence:
        optional_fields.add("risk_test_matrix_evidence")
    _expect_exact_keys(  # shape-check: fatal:unparseable-envelope
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
            "human_requirement_dispositions",
        },
        optional=optional_fields,
    )
    human_requirements_payload = _expect_object(  # shape-check: fatal:no-conservative-reading
        payload["human_requirements"],
        context="coder_followup.human_requirements",
    )
    human_requirement_dispositions = _expect_human_requirement_dispositions(  # shape-check: fatal:no-conservative-reading
        payload["human_requirement_dispositions"],
        context="coder_followup.human_requirement_dispositions",
    )
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        human_requirements_payload,
        context="coder_followup.human_requirements",
        required={"addressed_ids", "checked_discussion_directly"},
    )
    tests_run_value = payload.get("tests_run")
    tests_run = (
        _expect_string_list(  # shape-check: fatal:no-conservative-reading
            tests_run_value,
            context="coder_followup.tests_run",
            item_context="coder_followup.tests_run",
        )
        if tests_run_value is not None
        else None
    )
    test_observations, test_observation_degradations = _degradable_test_observations(  # shape-check: fatal:payload-bound
        payload.get("test_observations", []),
        context="coder_followup.test_observations",
        mode=test_observations_mode,
    )
    risk_claims = None
    if "risk_test_matrix_claims" in payload:
        risk_claims = _parse_semantic_risk_coverage_claims(  # shape-check: fatal:authority-decision
            payload["risk_test_matrix_claims"],
            context="coder_followup.risk_test_matrix_claims",
            expected_row_ids=delivered_risk_test_matrix_row_ids,
            execution_catalog=execution_catalog,
        )
    coverage_gaps = _parse_semantic_risk_coverage_gaps(  # shape-check: fatal:payload-bound
        payload,
        context="coder_followup.risk_test_matrix_coverage_gaps",
        expected_row_ids=delivered_risk_test_matrix_row_ids,
    )
    risk_evidence = None
    if allow_historical_canonical_evidence and "risk_test_matrix_evidence" in payload:
        risk_evidence = parse_risk_test_matrix_evidence(  # shape-check: fatal:authentication-or-forgery
            payload["risk_test_matrix_evidence"],
            matrix=delivered_risk_test_matrix,
            expected_identity=delivered_risk_test_matrix_identity,
            authoritative_test_observations=authoritative_test_observations,
            expected_row_ids=delivered_risk_test_matrix_row_ids,
            context="coder_followup.risk_test_matrix_evidence",
        )
    # A missing semantic claim set is recoverable after PR authentication: the
    # builder will emit complete non-verified rows.  Do not make absence hide a
    # PR or force a repair model to invent authority.
    # An absent or undetermined required assessment is a field-scope defect:
    # return the parsed result with an unsatisfied contract instead of
    # rejecting the envelope.  The orchestration seam refuses it (#925).
    architecture_impact, architecture_impact_degradations = _degradable_response_impact(  # shape-check: fatal:no-conservative-reading
        payload, mode=architecture_status_mode, context="coder_followup.architecture_impact"
    )
    architecture_impact_contract = architecture_impact_contract_for(
        architecture_impact, required=required_architecture_impact_contract == 1
    )
    addressed_items = _expect_item_id_list(  # shape-check: fatal:no-conservative-reading
        payload["addressed_items"],
        context="coder_followup.addressed_items",
    )
    remaining_items = _expect_item_id_list(  # shape-check: fatal:no-conservative-reading
        payload["remaining_items"],
        context="coder_followup.remaining_items",
    )
    disputed_items = _expect_item_id_list(  # shape-check: fatal:no-conservative-reading
        payload.get("disputed_items", []),
        context="coder_followup.disputed_items",
    )
    addressed_item_notes = _expect_item_note_map(  # shape-check: fatal:no-conservative-reading
        payload.get("addressed_item_notes", {}),
        context="coder_followup.addressed_item_notes",
        allowed_item_ids=set(addressed_items),
        allowed_context="coder_followup.addressed_items",
    )
    remaining_item_notes = _expect_item_note_map(  # shape-check: fatal:no-conservative-reading
        payload.get("remaining_item_notes", {}),
        context="coder_followup.remaining_item_notes",
        allowed_item_ids=set(remaining_items),
        allowed_context="coder_followup.remaining_items",
    )
    dispute_evidence = _expect_item_note_map(  # shape-check: fatal:no-conservative-reading
        payload.get("dispute_evidence", {}),
        context="coder_followup.dispute_evidence",
        allowed_item_ids=set(disputed_items),
        allowed_context="coder_followup.disputed_items",
    )
    missing_evidence = sorted(item_id for item_id in disputed_items if not dispute_evidence.get(item_id))
    if missing_evidence:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "Coder dispute must include non-empty evidence for each disputed item. "
            "Missing evidence for: " + ", ".join(missing_evidence)
        )
    all_classified = [*addressed_items, *remaining_items, *disputed_items]
    duplicates = sorted({item_id for item_id in all_classified if all_classified.count(item_id) > 1})
    if duplicates:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "Coder follow-up listed unresolved reviewer item IDs more than once: "
            + ", ".join(duplicates)
        )
    sub_item_claims, sub_item_claim_degradations = _degradable_sub_item_claims(  # shape-check: fatal:authentication-or-forgery
        payload.get("addressed_sub_items", []),
        remaining_items=remaining_items,
        expected_sub_item_ids=expected_sub_item_ids,
    )
    state = _expect_state(payload["state"], context="coder_followup.state")  # shape-check: fatal:no-conservative-reading
    if state == "approved" and any(
        item.disposition == "blocked" for item in human_requirement_dispositions
    ):
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            "coder_followup.state must be `blocking` when a signed human requirement is blocked."
        )
    return StructuredCoderFollowup(
        schema_version=1,
        kind="coder_followup",
        state=state,
        summary=_expect_non_empty_string(payload["summary"], context="coder_followup.summary"),  # shape-check: fatal:no-conservative-reading
        addressed_items=addressed_items,
        remaining_items=remaining_items,
        human_requirements=StructuredHumanRequirementsPayload(
            addressed_ids=_expect_requirement_id_list(  # shape-check: fatal:no-conservative-reading
                human_requirements_payload["addressed_ids"],
                context="coder_followup.human_requirements.addressed_ids",
            ),
            checked_discussion_directly=_expect_bool(  # shape-check: fatal:no-conservative-reading
                human_requirements_payload["checked_discussion_directly"],
                context="coder_followup.human_requirements.checked_discussion_directly",
            ),
        ),
        human_requirement_dispositions=human_requirement_dispositions,
        addressed_item_notes=addressed_item_notes,
        remaining_item_notes=remaining_item_notes,
        tests_run=tests_run,
        disputed_items=disputed_items,
        dispute_evidence=dispute_evidence,
        test_observations=test_observations,
        test_observation_degradations=test_observation_degradations,
        architecture_impact=architecture_impact,
        risk_test_matrix_claims=risk_claims,
        risk_test_matrix_coverage_gaps=coverage_gaps,
        risk_test_matrix_evidence=risk_evidence,
        architecture_impact_contract=architecture_impact_contract,
        architecture_impact_degradations=architecture_impact_degradations,
        addressed_sub_items=sub_item_claims,
        sub_item_claim_degradations=sub_item_claim_degradations,
    )


def sub_item_parent_id(sub_item_id: str) -> str:
    """The item ID a minted sub-item ID (``item-N.sK``) belongs to."""
    return sub_item_id.rsplit(".s", 1)[0] if ".s" in sub_item_id else sub_item_id


def _degradable_sub_item_claims(
    value: object,
    *,
    remaining_items: Sequence[str],
    expected_sub_item_ids: Mapping[str, Sequence[str]] | None,
) -> tuple[tuple[str, ...], tuple[ParseDegradation, ...]]:
    """Read advisory coder sub-item claims; every defect drops the claim (#958)."""
    context = "coder_followup.addressed_sub_items"
    if not isinstance(value, list):
        return (), (
            ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
                element_path=context,
                rule="addressed_sub_items-not-an-array",
                observed=value,
                outcome="claim-dropped",
            ),
        )
    claims: list[str] = []
    records: list[ParseDegradation] = []
    remaining = set(remaining_items)
    for index, entry in enumerate(value):
        rule: str | None = None
        if not isinstance(entry, str) or not entry.strip():
            rule = "addressed_sub_items-entry-not-a-non-empty-string"
        else:
            entry = entry.strip()
            parent = sub_item_parent_id(entry)
            if entry in claims:
                continue
            if parent not in remaining:
                rule = "addressed_sub_items-parent-not-in-remaining_items"
            elif expected_sub_item_ids is not None and entry not in set(
                expected_sub_item_ids.get(parent, ())
            ):
                rule = "addressed_sub_items-unknown-or-already-resolved-sub-item"
        if rule is not None:
            if len(records) < PARSE_DEGRADATION_RENDER_LIMIT:
                records.append(
                    ParseDegradation.build(  # shape-check: fatal:authentication-or-forgery
                        element_path=f"{context}[{index}]",
                        rule=rule,
                        observed=entry,
                        outcome="claim-dropped",
                    )
                )
            continue
        claims.append(entry)
    return tuple(claims), tuple(records)


def validate_structured_issue_implementation(
    text: str,
    *,
    required_architecture_impact_contract: int = 0,
    delivered_risk_test_matrix: RiskTestMatrix | Mapping[str, object] | None = None,
    delivered_risk_test_matrix_identity: str | None = None,
    required_risk_test_matrix_contract: int = 0,
    authoritative_test_observations: Sequence[object] | None = None,
    delivered_risk_test_matrix_row_ids: Sequence[str] | None = None,
    execution_catalog: Sequence[object] | None = None,
    allow_historical_canonical_evidence: bool = False,
    architecture_status_mode: str = "strict",
    test_observations_mode: str = "degrade",
) -> StructuredIssueImplementation | None:
    """Parse and validate the strict issue-implementation result envelope.

    Context-sensitive requirement coverage is applied by the orchestrator,
    because only the caller knows which signed labels were surfaced.  Basic
    payload shape is validated here so malformed responses remain repairable.
    """
    payload = _extract_structured_issue_implementation_payload(text)  # shape-check: fatal:unparseable-envelope
    if payload is None:
        return None
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    if payload.get("kind") != "issue_implementation":
        raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
            "Structured response kind mismatch: expected `issue_implementation`."
        )
    optional_fields = {
        "tests_run", "test_observations", "architecture_impact",
        "risk_test_matrix_claims",
        "risk_test_matrix_coverage_gaps",
    }
    if allow_historical_canonical_evidence:
        optional_fields.add("risk_test_matrix_evidence")
    _expect_exact_keys(  # shape-check: fatal:unparseable-envelope
        payload,
        context="issue_implementation",
        required={
            "schema_version",
            "kind",
            "state",
            "summary",
            "pr_number",
            "human_requirements",
            "human_requirement_dispositions",
        },
        optional=optional_fields,
    )
    state = _expect_non_empty_string(payload["state"], context="issue_implementation.state")  # shape-check: fatal:no-conservative-reading
    if state != "blocking":
        raise AgentLoopError("issue_implementation.state must be `blocking`.")  # shape-check: fatal:no-conservative-reading
    summary = _expect_non_empty_string(  # shape-check: fatal:no-conservative-reading
        payload["summary"], context="issue_implementation.summary"
    )
    pr_value = payload["pr_number"]
    if pr_value is None:
        pr_number = None
    else:
        pr_number = _expect_int(pr_value, context="issue_implementation.pr_number")  # shape-check: fatal:no-conservative-reading
        if pr_number <= 0:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                "issue_implementation.pr_number must be a positive integer or null."
            )
    human_payload = _expect_object(  # shape-check: fatal:no-conservative-reading
        payload["human_requirements"], context="issue_implementation.human_requirements"
    )
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        human_payload,
        context="issue_implementation.human_requirements",
        required={"addressed_ids", "checked_discussion_directly"},
    )
    dispositions = _expect_human_requirement_dispositions(  # shape-check: fatal:no-conservative-reading
        payload["human_requirement_dispositions"],
        context="issue_implementation.human_requirement_dispositions",
    )
    tests_value = payload.get("tests_run")
    tests_run = (
        _expect_string_list(  # shape-check: fatal:no-conservative-reading
            tests_value,
            context="issue_implementation.tests_run",
            item_context="issue_implementation.tests_run",
        )
        if tests_value is not None
        else None
    )
    test_observations, test_observation_degradations = _degradable_test_observations(  # shape-check: fatal:payload-bound
        payload.get("test_observations", []),
        context="issue_implementation.test_observations",
        mode=test_observations_mode,
    )
    risk_claims = None
    if "risk_test_matrix_claims" in payload:
        risk_claims = _parse_semantic_risk_coverage_claims(  # shape-check: fatal:authority-decision
            payload["risk_test_matrix_claims"],
            context="issue_implementation.risk_test_matrix_claims",
            expected_row_ids=delivered_risk_test_matrix_row_ids,
            execution_catalog=execution_catalog,
        )
    coverage_gaps = _parse_semantic_risk_coverage_gaps(  # shape-check: fatal:payload-bound
        payload,
        context="issue_implementation.risk_test_matrix_coverage_gaps",
        expected_row_ids=delivered_risk_test_matrix_row_ids,
    )
    risk_evidence = None
    if allow_historical_canonical_evidence and "risk_test_matrix_evidence" in payload:
        risk_evidence = parse_risk_test_matrix_evidence(  # shape-check: fatal:authentication-or-forgery
            payload["risk_test_matrix_evidence"],
            matrix=delivered_risk_test_matrix,
            expected_identity=delivered_risk_test_matrix_identity,
            authoritative_test_observations=authoritative_test_observations,
            expected_row_ids=delivered_risk_test_matrix_row_ids,
            context="issue_implementation.risk_test_matrix_evidence",
        )
    # Missing claims remain a bounded, complete non-verified result at the
    # post-head builder boundary; they are not a reason to discard a PR.
    architecture_impact, architecture_impact_degradations = _degradable_response_impact(  # shape-check: fatal:no-conservative-reading
        payload, mode=architecture_status_mode, context="issue_implementation.architecture_impact"
    )
    architecture_impact_contract = architecture_impact_contract_for(
        architecture_impact, required=required_architecture_impact_contract == 1
    )
    parsed = StructuredIssueImplementation(
        schema_version=1,
        kind="issue_implementation",
        state=state,
        summary=summary,
        pr_number=pr_number,
        human_requirements=StructuredHumanRequirementsPayload(
            addressed_ids=_expect_requirement_id_list(  # shape-check: fatal:no-conservative-reading
                human_payload["addressed_ids"],
                context="issue_implementation.human_requirements.addressed_ids",
            ),
            checked_discussion_directly=_expect_bool(  # shape-check: fatal:no-conservative-reading
                human_payload["checked_discussion_directly"],
                context="issue_implementation.human_requirements.checked_discussion_directly",
            ),
        ),
        human_requirement_dispositions=dispositions,
        tests_run=tests_run,
        test_observations=test_observations,
        test_observation_degradations=test_observation_degradations,
        architecture_impact=architecture_impact,
        risk_test_matrix_claims=risk_claims,
        risk_test_matrix_evidence=risk_evidence,
        architecture_impact_contract=architecture_impact_contract,
        architecture_impact_degradations=architecture_impact_degradations,
        risk_test_matrix_coverage_gaps=coverage_gaps,
    )
    # An unsatisfied required contract outranks the semantic conflict, as the
    # former raise did: return it so the seam refuses it rather than letting a
    # conflict wrapper hide the missing assessment.
    if architecture_impact_contract.required and not architecture_impact_contract.satisfied:
        return parsed
    # Keep this semantic contradiction visible to callers as a dedicated error
    # with the typed payload attached.  The orchestration adapter first
    # re-applies caller-specific requirement coverage before treating it as a
    # terminal conflict.
    if parsed.pr_number is not None and any(
        item.disposition == "blocked" for item in parsed.human_requirement_dispositions
    ):
        raise IssueImplementationConflictError(parsed)  # shape-check: fatal:no-conservative-reading
    return parsed


def parse_historical_structured_coder_followup(
    text: str,
    **kwargs: object,
) -> StructuredCoderFollowup | None:
    """Read an already-persisted follow-up without making it a fresh contract."""
    # Stored text decodes in the explicit legacy mode, exactly as before #925.
    kwargs.setdefault("architecture_status_mode", "legacy")
    return validate_structured_coder_followup(  # shape-check: fatal:no-conservative-reading
        text, allow_historical_canonical_evidence=True, **kwargs
    )


def parse_historical_structured_issue_implementation(
    text: str,
    **kwargs: object,
) -> StructuredIssueImplementation | None:
    """Read an already-persisted implementation without making it a fresh contract."""
    # Stored text decodes in the explicit legacy mode, exactly as before #925.
    kwargs.setdefault("architecture_status_mode", "legacy")
    return validate_structured_issue_implementation(  # shape-check: fatal:no-conservative-reading
        text, allow_historical_canonical_evidence=True, **kwargs
    )


def validate_structured_task_result(
    text: str,
    *,
    required_architecture_impact_contract: int = 0,
    architecture_status_mode: str = "strict",
) -> StructuredTaskResult | None:
    """Validate the versioned task terminal envelope, when present."""
    normalized, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(normalized)  # shape-check: fatal:unparseable-envelope
    if extracted is None:
        return None
    payload, trailing = extracted
    payload = _consume_structured_footer_and_signature(  # shape-check: fatal:unparseable-envelope
        payload=payload, trailing=trailing, state_re=STATE_RE,
        state_marker_name="AGENT_STATE", context_label="Structured task result",
    )
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    if payload.get("kind") != "task_result":
        raise AgentLoopError("Structured response kind mismatch: expected `task_result`.")  # shape-check: fatal:unparseable-envelope
    _expect_exact_keys(  # shape-check: fatal:unparseable-envelope
        payload, context="task_result",
        required={"schema_version", "kind", "state", "outcome", "summary"},
        optional={"pr_number", "clarification", "architecture_impact"},
    )
    state = _expect_non_empty_string(payload["state"], context="task_result.state")  # shape-check: fatal:no-conservative-reading
    if state != "blocking":
        raise AgentLoopError("task_result.state must be `blocking`.")  # shape-check: fatal:no-conservative-reading
    outcome = _expect_non_empty_string(payload["outcome"], context="task_result.outcome")  # shape-check: fatal:no-conservative-reading
    if outcome not in {"opened_pr", "blocking", "clarification"}:
        raise AgentLoopError("task_result.outcome must be opened_pr, blocking, or clarification.")  # shape-check: fatal:no-conservative-reading
    pr_value = payload.get("pr_number")
    pr_number = None if pr_value is None else _expect_int(pr_value, context="task_result.pr_number")  # shape-check: fatal:no-conservative-reading
    if outcome == "opened_pr" and (pr_number is None or pr_number <= 0):
        raise AgentLoopError("task_result.opened_pr requires a positive pr_number.")  # shape-check: fatal:no-conservative-reading
    if outcome != "opened_pr" and pr_number is not None:
        raise AgentLoopError("task_result.pr_number is only valid for opened_pr.")  # shape-check: fatal:no-conservative-reading
    questions = ()
    if "clarification" in payload:
        questions = _expect_string_list(payload["clarification"], context="task_result.clarification", item_context="task_result.clarification")  # shape-check: fatal:no-conservative-reading
    if outcome == "clarification" and not questions:
        raise AgentLoopError("task_result.clarification requires at least one question.")  # shape-check: fatal:no-conservative-reading
    if outcome != "clarification" and questions:
        raise AgentLoopError("task_result.clarification is only valid for clarification.")  # shape-check: fatal:no-conservative-reading
    architecture_impact, architecture_impact_degradations = _degradable_response_impact(  # shape-check: fatal:no-conservative-reading
        payload, mode=architecture_status_mode, context="task_result.architecture_impact"
    )
    return StructuredTaskResult(
        schema_version=1, kind="task_result", state=state, outcome=outcome,
        summary=_expect_non_empty_string(payload["summary"], context="task_result.summary"),  # shape-check: fatal:no-conservative-reading
        pr_number=pr_number, clarification=questions,
        architecture_impact=architecture_impact,
        architecture_impact_contract=architecture_impact_contract_for(
            architecture_impact, required=required_architecture_impact_contract == 1
        ),
        architecture_impact_degradations=architecture_impact_degradations,
    )


def validate_structured_plan_revision(
    text: str,
    *,
    required_architecture_impact_contract: int = 0,
    require_execution_strategy_contract: int = 0,
    require_risk_test_matrix_contract: int = 0,
    reject_unsolicited_risk_test_matrix_contract: bool = False,
    require_child_dispositions: bool = False,
    architecture_status_mode: str = "strict",
) -> StructuredPlanRevision | None:
    payload = _extract_structured_plan_revision_payload(text)  # shape-check: fatal:unparseable-envelope
    if payload is None:
        return None
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "plan_revision":
        raise AgentLoopError("Structured response kind mismatch: expected `plan_revision`.")  # shape-check: fatal:unparseable-envelope
    _expect_exact_keys(  # shape-check: fatal:unparseable-envelope
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
        optional={
            "additional_closing_issue_ids",
            "deferred_stages",
            "child_stages",
            "external_dependencies",
            "deferred_work",
            "plan_actions",
            "human_requirement_dispositions",
            "architecture_impact",
            "execution_strategy_contract_version",
            "execution_recommendation",
            "risk_test_matrix_contract_version",
            "risk_test_matrix",
            "risk_test_matrix_changes",
            "one_shot_growth_justification",
        },
    )
    execution_version, execution_recommendation = _parse_execution_contract_fields(  # shape-check: fatal:no-conservative-reading
        payload,
        context="plan_revision",
        required=require_execution_strategy_contract == 1,
        require_child_dispositions=require_child_dispositions,
    )
    risk_version, risk_matrix, risk_changes = _parse_risk_test_matrix_contract_fields(  # shape-check: fatal:no-conservative-reading
        payload,
        context="plan_revision",
        required=require_risk_test_matrix_contract == 1,
        reject_unsolicited=reject_unsolicited_risk_test_matrix_contract,
    )
    state = _expect_non_empty_string(payload["state"], context="plan_revision.state")  # shape-check: fatal:no-conservative-reading
    if state != "blocking":
        raise AgentLoopError("plan_revision.state must be `blocking`.")  # shape-check: fatal:no-conservative-reading
    summary = _expect_non_empty_string(payload["summary"], context="plan_revision.summary")  # shape-check: fatal:no-conservative-reading
    dispositions = _expect_disposition_list(  # shape-check: fatal:no-conservative-reading
        payload["prior_plan_item_dispositions"],
        context="plan_revision.prior_plan_item_dispositions",
        reviewer="coder",
        allowed_same_status="same-plan",
        is_plan_review=True,
    )
    plan_steps = _expect_string_list(  # shape-check: fatal:no-conservative-reading
        payload["plan_steps"],
        context="plan_revision.plan_steps",
        item_context="plan_revision.plan_steps",
        min_length=1,
    )
    architecture_impact, architecture_impact_degradations = _degradable_response_impact(  # shape-check: fatal:no-conservative-reading
        payload, mode=architecture_status_mode, context="plan_revision.architecture_impact"
    )
    additional_closing_issue_ids = _expect_optional_issue_id_list(  # shape-check: fatal:no-conservative-reading
        payload,
        "additional_closing_issue_ids",
        context="plan_revision.additional_closing_issue_ids",
    )
    deferred_stages = _expect_deferred_stage_list(  # shape-check: fatal:no-conservative-reading
        payload, "deferred_stages", context="plan_revision.deferred_stages"
    )
    human_requirement_dispositions = _expect_human_requirement_dispositions(  # shape-check: fatal:no-conservative-reading
        payload.get("human_requirement_dispositions", []),
        context="plan_revision.human_requirement_dispositions",
    )
    return StructuredPlanRevision(
        schema_version=1,
        kind="plan_revision",
        state="blocking",
        summary=summary,
        prior_plan_item_dispositions=dispositions,
        plan_steps=plan_steps,
        additional_closing_issue_ids=additional_closing_issue_ids,
        deferred_stages=deferred_stages,
        typed_stages=_expect_typed_plan_stages(payload, context="plan_revision"),  # shape-check: fatal:no-conservative-reading
        human_requirement_dispositions=human_requirement_dispositions,
        architecture_impact=architecture_impact,
        execution_strategy_contract_version=execution_version,
        execution_recommendation=execution_recommendation,
        risk_test_matrix_contract_version=risk_version,
        risk_test_matrix=risk_matrix,
        risk_test_matrix_changes=risk_changes,
        architecture_impact_contract=architecture_impact_contract_for(
            architecture_impact, required=required_architecture_impact_contract == 1
        ),
        architecture_impact_degradations=architecture_impact_degradations,
        one_shot_growth_justification=_expect_optional_growth_justification(  # shape-check: fatal:no-conservative-reading
            payload, context="plan_revision"
        ),
    )


def validate_structured_plan_state(
    text: str,
    *,
    required_architecture_impact_contract: int = 0,
    require_execution_strategy_contract: int = 0,
    require_risk_test_matrix_contract: int = 0,
    require_child_dispositions: bool = False,
    architecture_status_mode: str = "strict",
) -> StructuredPlanState | None:
    payload = _extract_structured_plan_state_payload(text)  # shape-check: fatal:unparseable-envelope
    if payload is None:
        return None
    _require_supported_schema_version(payload)  # shape-check: fatal:kind-or-version-mismatch
    kind = payload.get("kind")
    if kind != "plan_state":
        raise AgentLoopError("Structured response kind mismatch: expected `plan_state`.")  # shape-check: fatal:unparseable-envelope
    _expect_exact_keys(  # shape-check: fatal:unparseable-envelope
        payload,
        context="plan_state",
        required={"schema_version", "kind", "state", "summary", "plan_steps"},
        optional={
            "additional_closing_issue_ids",
            "deferred_stages",
            "child_stages",
            "external_dependencies",
            "deferred_work",
            "plan_actions",
            "human_requirement_dispositions",
            "architecture_impact",
            "execution_strategy_contract_version",
            "execution_recommendation",
            "risk_test_matrix_contract_version",
            "risk_test_matrix",
            "risk_test_matrix_changes",
            "one_shot_growth_justification",
        },
    )
    execution_version, execution_recommendation = _parse_execution_contract_fields(  # shape-check: fatal:no-conservative-reading
        payload,
        context="plan_state",
        required=require_execution_strategy_contract == 1,
        require_child_dispositions=require_child_dispositions,
    )
    risk_version, risk_matrix, risk_changes = _parse_risk_test_matrix_contract_fields(  # shape-check: fatal:no-conservative-reading
        payload,
        context="plan_state",
        required=require_risk_test_matrix_contract == 1,
    )
    state = _expect_state(payload["state"], context="plan_state.state")  # shape-check: fatal:no-conservative-reading
    if state != "blocking":
        raise AgentLoopError("plan_state.state must be `blocking`.")  # shape-check: fatal:no-conservative-reading
    architecture_impact, architecture_impact_degradations = _degradable_response_impact(  # shape-check: fatal:no-conservative-reading
        payload, mode=architecture_status_mode, context="plan_state.architecture_impact"
    )
    return StructuredPlanState(
        schema_version=int(payload.get("schema_version", 1)),
        kind="plan_state",
        state=parse_plan_state(text),  # shape-check: fatal:unparseable-envelope
        summary=_expect_non_empty_string(payload["summary"], context="plan_state.summary"),  # shape-check: fatal:no-conservative-reading
        plan_steps=_expect_string_list(  # shape-check: fatal:no-conservative-reading
            payload["plan_steps"],
            context="plan_state.plan_steps",
            item_context="plan_state.plan_steps",
            min_length=1,
        ),
        additional_closing_issue_ids=_expect_optional_issue_id_list(  # shape-check: fatal:no-conservative-reading
            payload,
            "additional_closing_issue_ids",
            context="plan_state.additional_closing_issue_ids",
        ),
        deferred_stages=_expect_deferred_stage_list(  # shape-check: fatal:no-conservative-reading
            payload, "deferred_stages", context="plan_state.deferred_stages"
        ),
        typed_stages=_expect_typed_plan_stages(payload, context="plan_state"),  # shape-check: fatal:no-conservative-reading
        human_requirement_dispositions=_expect_human_requirement_dispositions(  # shape-check: fatal:no-conservative-reading
            payload.get("human_requirement_dispositions", []),
            context="plan_state.human_requirement_dispositions",
        ),
        architecture_impact=architecture_impact,
        execution_strategy_contract_version=execution_version,
        execution_recommendation=execution_recommendation,
        risk_test_matrix_contract_version=risk_version,
        risk_test_matrix=risk_matrix,
        risk_test_matrix_changes=risk_changes,
        architecture_impact_contract=architecture_impact_contract_for(
            architecture_impact, required=required_architecture_impact_contract == 1
        ),
        architecture_impact_degradations=architecture_impact_degradations,
        one_shot_growth_justification=_expect_optional_growth_justification(  # shape-check: fatal:no-conservative-reading
            payload, context="plan_state"
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
    raise AgentLoopError(f"Unsupported unresolved item disposition: {status}")  # shape-check: fatal:no-conservative-reading


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
    parent_indent: int | None = None

    for line_number, line in enumerate(text.splitlines(), start=1):
        if heading_re.match(line):
            active = True
            section_heading = line.strip()
            parent_indent = None
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
        indent = len(line.expandtabs()) - len(line.expandtabs().lstrip())
        # Canonical nested history describes the parent, not a new disposition.
        if (
            parent_indent is not None
            and indent > parent_indent
            and entry.startswith(("Original finding:", "Sub-items (per this review only"))
        ):
            continue
        if empty_item_re.match(entry):
            continue
        match = disposition_re.match(entry)
        if not match:
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{error_message} In section `{section_heading or 'unknown section'}`, "
                f"line {line_number}: `{entry}`."
            )
        note = match.group("note")
        normalized_note = note.strip() if note else None
        disposition = _normalize_disposition(match.group("status"), same_status=same_status)  # shape-check: fatal:no-conservative-reading
        if _active_disposition_has_empty_note(
            disposition,
            normalized_note,
            same_status=same_status,
            is_plan_review=is_plan_review,
        ):
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
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
        parent_indent = indent

    return tuple(dispositions)


def parse_unresolved_item_dispositions(text: str, *, reviewer: str) -> tuple[ReviewItemDisposition, ...]:
    """Extract structured prior-item dispositions from a review."""
    return _parse_unresolved_item_dispositions(  # shape-check: fatal:no-conservative-reading
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
    return _parse_unresolved_item_dispositions(  # shape-check: fatal:no-conservative-reading
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
    state = parse_agent_state(text)  # shape-check: fatal:unparseable-envelope
    summary = review_freeform_summary_text(text)
    blocking_items = parse_pr_blocking_items(text, reviewer=reviewer)
    followups = parse_approved_followups(text, reviewer=reviewer)
    followups = _dedupe_pr_review_items(blocking_items, followups)
    dispositions = parse_unresolved_item_dispositions(text, reviewer=reviewer)  # shape-check: fatal:no-conservative-reading
    return _finalize_parsed_review(  # shape-check: fatal:no-conservative-reading
        state=state,
        summary=summary,
        blocking_items=blocking_items,
        followups=followups,
        dispositions=dispositions,
        raw_dispositions_text=_extract_section_text(
            text, heading_re=PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE
        ),
    )


def parse_pr_review(
    text: str, *, reviewer: str, architecture_status_mode: str = "strict"
) -> ParsedReview:
    parsed = parse_structured_pr_review(  # shape-check: fatal:unparseable-envelope
        text, reviewer=reviewer, architecture_status_mode=architecture_status_mode
    )
    if parsed is not None:
        return parsed
    raise AgentLoopError("Agent response did not use the required structured format.")  # shape-check: fatal:unparseable-envelope


def parse_plan_review(
    text: str,
    *,
    reviewer: str,
    architecture_status_mode: str = "strict",
    surfaced_requirement_ids: Sequence[str] | None = None,
) -> ParsedPlanReview:
    """Parse a plan review, including state, structured plan items, and dispositions."""
    parsed = parse_structured_plan_review(  # shape-check: fatal:unparseable-envelope
        text,
        reviewer=reviewer,
        architecture_status_mode=architecture_status_mode,
        surfaced_requirement_ids=surfaced_requirement_ids,
    )
    if parsed is not None:
        return parsed
    raise AgentLoopError("Agent response did not use the required structured format.")  # shape-check: fatal:unparseable-envelope


def parse_non_blocking_followups(text: str, *, reviewer: str) -> list[ApprovedFollowup]:
    """Extract legacy non-blocking follow-ups as future follow-ups."""
    return list(parse_approved_followups(text, reviewer=reviewer).future)


# Semantic plan revisions -------------------------------------------------
#
# These types intentionally live beside the generation-1 wire validators.  A
# semantic patch is a bounded input to the deterministic assembler; it is not
# itself a replacement for the generation-1 ``plan_revision`` object.
PLAN_REVISION_PATCH_SCHEMA_VERSION = 1
SEMANTIC_PATCH_CONTRACT_VERSION = 1
PLAN_REVISION_PATCH_KIND = "plan_revision_patch"
PLAN_REVISION_PATCH_OPERATION_KEYS = frozenset(
    {
        "replace",
        "matrix_add",
        "matrix_edit",
        "matrix_retire",
        "matrix_split",
        "matrix_merge",
        "matrix_metadata_replace",
    }
)
PLAN_REVISION_PATCH_REPLACEABLE_FIELDS = frozenset(
    {
        "summary",
        "plan_steps",
        "architecture_impact",
        "additional_closing_issue_ids",
        "human_requirement_dispositions",
        "execution_recommendation",
        "external_dependencies",
        "deferred_work",
        "plan_actions",
        "deferred_stages",
        "one_shot_growth_justification",
    }
)
# Whole-field replaces whose ``null`` value removes the key (#886).
PLAN_REVISION_PATCH_CLEARABLE_FIELDS = frozenset({"one_shot_growth_justification"})
# Fields that cannot take a wholesale ``replace`` but do have a write path
# through dedicated per-row operations.  Diagnostics name these operations so
# an author who picked the wrong operation can recover on the next attempt.
PLAN_REVISION_PATCH_DEDICATED_FIELD_OPERATIONS: dict[str, tuple[str, ...]] = {
    "risk_test_matrix": (
        "matrix_add",
        "matrix_edit",
        "matrix_retire",
        "matrix_split",
        "matrix_merge",
        "matrix_metadata_replace",
    ),
}
_PLAN_REVISION_PATCH_IDENTITY_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PlanRevisionPatchOperation:
    """One exact-key semantic operation.

    A single tagged representation keeps parsing and serialization uniform,
    while the assembler validates which attributes are meaningful for each
    ``op``.  Parsed row values are complete generation-1 rows, never partial
    deltas.
    """

    op: str
    field: str | None = None
    value: object | None = None
    row: RiskTestMatrixRow | None = None
    row_id: str | None = None
    final_position: int | None = None
    source_row_id: str | None = None
    target_rows: tuple[RiskTestMatrixRow, ...] = ()
    source_row_ids: tuple[str, ...] = ()
    target_row: RiskTestMatrixRow | None = None
    audit_operation: str | None = None
    rationale: str | None = None

    def to_payload(self) -> dict[str, object]:
        return _json_array_mapping(self._raw_payload())  # shape-check: fatal:authentication-or-forgery

    def _raw_payload(self) -> dict[str, object]:
        if self.op == "replace":
            return {"op": self.op, "field": self.field, "value": _patch_value_payload(self.value)}  # shape-check: fatal:authentication-or-forgery
        if self.op == "matrix_add":
            return {
                "op": self.op,
                "row": self.row.to_payload() if self.row is not None else None,  # shape-check: fatal:authentication-or-forgery
                "final_position": self.final_position,
                "rationale": self.rationale,
            }
        if self.op in {"matrix_edit", "matrix_retire"}:
            payload: dict[str, object] = {
                "op": self.op,
                "row_id": self.row_id,
                "rationale": self.rationale,
            }
            if self.op == "matrix_edit":
                payload["row"] = self.row.to_payload() if self.row is not None else None  # shape-check: fatal:authentication-or-forgery
            return payload
        if self.op == "matrix_split":
            return {
                "op": self.op,
                "source_row_id": self.source_row_id,
                "target_rows": [row.to_payload() for row in self.target_rows],  # shape-check: fatal:authentication-or-forgery
                "rationale": self.rationale,
            }
        if self.op == "matrix_merge":
            return {
                "op": self.op,
                "source_row_ids": list(self.source_row_ids),
                "target_row": self.target_row.to_payload() if self.target_row is not None else None,  # shape-check: fatal:authentication-or-forgery
                "rationale": self.rationale,
            }
        return {
            "op": self.op,
            "value": self.value.to_payload() if isinstance(self.value, RiskTestMatrixMetadata) else self.value,  # shape-check: fatal:authentication-or-forgery
            "audit_operation": self.audit_operation,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class PlanRevisionPatch:
    schema_version: int
    kind: str
    semantic_patch_contract_version: int
    state: str
    summary: str
    prior_plan_item_dispositions: tuple[ReviewItemDisposition, ...]
    base_round_number: int
    base_state_identity: str
    operations: tuple[PlanRevisionPatchOperation, ...]

    def to_payload(self) -> dict[str, object]:
        return _json_array_mapping({
            "schema_version": self.schema_version,
            "kind": self.kind,
            "semantic_patch_contract_version": self.semantic_patch_contract_version,
            "state": self.state,
            "summary": sanitize_historical_text(self.summary),  # shape-check: fatal:authentication-or-forgery
            "prior_plan_item_dispositions": [
                {
                    "item_id": item.item_id,
                    "disposition": item.disposition,
                    **({"note": item.note} if item.note is not None else {}),
                }
                for item in self.prior_plan_item_dispositions
            ],
            "base_round_number": self.base_round_number,
            "base_state_identity": self.base_state_identity,
            "operations": [operation.to_payload() for operation in self.operations],  # shape-check: fatal:authentication-or-forgery
        })


def _json_array_payload(value: object) -> object:
    """Return ``value`` with every sequence rendered as a JSON array.

    ``dataclasses.asdict`` preserves tuples for immutable fields such as
    ``ArchitectureImpact.execution_data_flows``.  Patch payloads are stored as
    provenance, re-parsed with the wire parsers, and compared for equality
    against the stored JSON copy, so a tuple here both fails the parsers and
    breaks ``list != tuple`` equality on restart (#879).
    """
    if isinstance(value, Mapping):
        return {key: _json_array_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_array_payload(item) for item in value]
    return value


def _json_array_mapping(payload: dict[str, object]) -> dict[str, object]:
    return {key: _json_array_payload(item) for key, item in payload.items()}


def _patch_value_payload(value: object) -> object:
    if isinstance(value, ArchitectureImpact):
        return dataclasses.asdict(value)
    if isinstance(value, ExecutionStrategyRecommendation):
        return value.to_payload()  # shape-check: fatal:authentication-or-forgery
    if isinstance(value, HumanRequirementDisposition):
        return dataclasses.asdict(value)
    if isinstance(value, tuple):
        return [_patch_value_payload(item) for item in value]  # shape-check: fatal:authentication-or-forgery
    if isinstance(value, list):
        return [_patch_value_payload(item) for item in value]  # shape-check: fatal:authentication-or-forgery
    if isinstance(value, DeferredStage):
        return {"title": value.title, "summary": value.summary}
    if isinstance(value, TypedPlanStages):
        return {
            "child_stages": [dataclasses.asdict(item) for item in value.child_stages],
            "external_dependencies": [_patch_value_payload(item) for item in value.external_dependencies],  # shape-check: fatal:authentication-or-forgery
            "deferred_work": [_patch_value_payload(item) for item in value.deferred_work],  # shape-check: fatal:authentication-or-forgery
            "plan_actions": [_patch_value_payload(item) for item in value.plan_actions],  # shape-check: fatal:authentication-or-forgery
        }
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return value


def _parse_plan_patch_field_value(field_name: str, value: object, *, context: str) -> object:
    if field_name == "summary":
        return _expect_non_empty_string(value, context=context)  # shape-check: fatal:no-conservative-reading
    if field_name == "plan_steps":
        return _expect_string_list(value, context=context, item_context=context, min_length=1)  # shape-check: fatal:no-conservative-reading
    if field_name == "architecture_impact":
        # A patch writes approved plan state, so it stays strict: a degraded
        # status must never persist.  Name the route forward instead of the
        # generic enum message so the planner does not retry blindly (#924).
        status_value = value.get("status") if isinstance(value, Mapping) else None
        if (
            isinstance(status_value, str)
            and status_value not in ARCHITECTURE_IMPACT_DECLARED_STATUSES
            and architecture_status_near_miss_key(status_value) in _ARCHITECTURE_STATUS_SYNONYMS
        ):
            raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                f"{context}.status must be `changed` or `unchanged`; "
                f"`{status_value}` is not accepted in a patch. Replace architecture_impact "
                "with a declared `changed` or `unchanged` assessment."
            )
        return _parse_architecture_impact(value, context=context)  # shape-check: fatal:no-conservative-reading
    if field_name == "additional_closing_issue_ids":
        return _expect_optional_issue_id_list({field_name: value}, field_name, context=context)  # shape-check: fatal:no-conservative-reading
    if field_name == "human_requirement_dispositions":
        return _expect_human_requirement_dispositions(value, context=context)  # shape-check: fatal:no-conservative-reading
    if field_name == "execution_recommendation":
        return _expect_execution_recommendation(value, context=context)  # shape-check: fatal:no-conservative-reading
    if field_name in {"external_dependencies", "deferred_work", "plan_actions", "deferred_stages"}:
        return _expect_deferred_stage_list({field_name: value}, field_name, context=context)  # shape-check: fatal:no-conservative-reading
    if field_name == "one_shot_growth_justification":
        if value is None:
            return None
        return _expect_growth_justification(value, context=context)  # shape-check: fatal:no-conservative-reading
    raise AgentLoopError(f"{context} is not a writable semantic plan field.")  # shape-check: fatal:no-conservative-reading



def _format_operation_names(names: Sequence[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " or " + names[-1]


def _ordered_patch_operation_keys() -> tuple[str, ...]:
    matrix_ops = PLAN_REVISION_PATCH_DEDICATED_FIELD_OPERATIONS["risk_test_matrix"]
    return ("replace", *matrix_ops, *sorted(PLAN_REVISION_PATCH_OPERATION_KEYS - {"replace", *matrix_ops}))


def _parse_plan_revision_patch_operation(value: object, *, context: str) -> PlanRevisionPatchOperation:
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    op = _expect_non_empty_string(payload.get("op"), context=f"{context}.op")  # shape-check: fatal:no-conservative-reading
    if op not in PLAN_REVISION_PATCH_OPERATION_KEYS:
        raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
            f"{context}.op is unknown: {op!r}; valid operations are "
            f"{_format_operation_names(_ordered_patch_operation_keys())}."
        )
    if op == "replace":
        _expect_exact_keys(payload, context=context, required={"op", "field", "value"})  # shape-check: fatal:no-conservative-reading
        field_name = _expect_non_empty_string(payload["field"], context=f"{context}.field")  # shape-check: fatal:no-conservative-reading
        if field_name not in PLAN_REVISION_PATCH_REPLACEABLE_FIELDS:
            dedicated = PLAN_REVISION_PATCH_DEDICATED_FIELD_OPERATIONS.get(field_name)
            if dedicated:
                raise AgentLoopError(  # shape-check: fatal:no-conservative-reading
                    f"{context}: `{field_name}` cannot be revised with `replace`; "
                    f"use {_format_operation_names(dedicated)}."
                )
            raise AgentLoopError(f"{context}.field `{field_name}` is derived or not writable.")  # shape-check: fatal:no-conservative-reading
        return PlanRevisionPatchOperation(
            op=op,
            field=field_name,
            value=_parse_plan_patch_field_value(field_name, payload["value"], context=f"{context}.value"),  # shape-check: fatal:no-conservative-reading
        )
    if op == "matrix_add":
        _expect_exact_keys(payload, context=context, required={"op", "row", "final_position", "rationale"})  # shape-check: fatal:no-conservative-reading
        position = _expect_int(payload["final_position"], context=f"{context}.final_position")  # shape-check: fatal:no-conservative-reading
        if position < 0:
            raise AgentLoopError(f"{context}.final_position must be non-negative.")  # shape-check: fatal:no-conservative-reading
        return PlanRevisionPatchOperation(
            op=op,
            row=parse_risk_test_matrix_row(payload["row"], context=f"{context}.row"),  # shape-check: fatal:no-conservative-reading
            final_position=position,
            rationale=_risk_bounded_string(payload["rationale"], context=f"{context}.rationale", max_bytes=2_048),  # shape-check: fatal:no-conservative-reading
        )
    if op == "matrix_edit":
        _expect_exact_keys(payload, context=context, required={"op", "row_id", "row", "rationale"})  # shape-check: fatal:no-conservative-reading
        return PlanRevisionPatchOperation(
            op=op,
            row_id=_validate_risk_row_id(payload["row_id"], context=f"{context}.row_id"),  # shape-check: fatal:no-conservative-reading
            row=parse_risk_test_matrix_row(payload["row"], context=f"{context}.row"),  # shape-check: fatal:no-conservative-reading
            rationale=_risk_bounded_string(payload["rationale"], context=f"{context}.rationale", max_bytes=2_048),  # shape-check: fatal:no-conservative-reading
        )
    if op == "matrix_retire":
        _expect_exact_keys(payload, context=context, required={"op", "row_id", "rationale"})  # shape-check: fatal:no-conservative-reading
        return PlanRevisionPatchOperation(
            op=op,
            row_id=_validate_risk_row_id(payload["row_id"], context=f"{context}.row_id"),  # shape-check: fatal:no-conservative-reading
            rationale=_risk_bounded_string(payload["rationale"], context=f"{context}.rationale", max_bytes=2_048),  # shape-check: fatal:no-conservative-reading
        )
    if op == "matrix_split":
        _expect_exact_keys(payload, context=context, required={"op", "source_row_id", "target_rows", "rationale"})  # shape-check: fatal:no-conservative-reading
        source = _validate_risk_row_id(payload["source_row_id"], context=f"{context}.source_row_id")  # shape-check: fatal:no-conservative-reading
        targets_payload = payload["target_rows"]
        if not isinstance(targets_payload, list) or len(targets_payload) < 2:
            raise AgentLoopError(f"{context}.target_rows must contain at least two complete rows.")  # shape-check: fatal:no-conservative-reading
        targets = tuple(
            parse_risk_test_matrix_row(item, context=f"{context}.target_rows[{index}]")  # shape-check: fatal:no-conservative-reading
            for index, item in enumerate(targets_payload)
        )
        if len({row.row_id for row in targets}) != len(targets):
            raise AgentLoopError(f"{context}.target_rows contains duplicate row IDs.")  # shape-check: fatal:no-conservative-reading
        return PlanRevisionPatchOperation(
            op=op,
            source_row_id=source,
            target_rows=targets,
            rationale=_risk_bounded_string(payload["rationale"], context=f"{context}.rationale", max_bytes=2_048),  # shape-check: fatal:no-conservative-reading
        )
    if op == "matrix_merge":
        _expect_exact_keys(payload, context=context, required={"op", "source_row_ids", "target_row", "rationale"})  # shape-check: fatal:no-conservative-reading
        sources_payload = payload["source_row_ids"]
        if not isinstance(sources_payload, list) or len(sources_payload) < 2:
            raise AgentLoopError(f"{context}.source_row_ids must contain at least two row IDs.")  # shape-check: fatal:no-conservative-reading
        sources = tuple(
            _validate_risk_row_id(item, context=f"{context}.source_row_ids[{index}]")  # shape-check: fatal:no-conservative-reading
            for index, item in enumerate(sources_payload)
        )
        if len(set(sources)) != len(sources):
            raise AgentLoopError(f"{context}.source_row_ids must contain distinct row IDs.")  # shape-check: fatal:no-conservative-reading
        return PlanRevisionPatchOperation(
            op=op,
            source_row_ids=sources,
            target_row=parse_risk_test_matrix_row(payload["target_row"], context=f"{context}.target_row"),  # shape-check: fatal:no-conservative-reading
            rationale=_risk_bounded_string(payload["rationale"], context=f"{context}.rationale", max_bytes=2_048),  # shape-check: fatal:no-conservative-reading
        )
    _expect_exact_keys(payload, context=context, required={"op", "value", "audit_operation", "rationale"})  # shape-check: fatal:no-conservative-reading
    audit_operation = _risk_bounded_string(payload["audit_operation"], context=f"{context}.audit_operation", max_bytes=32)  # shape-check: fatal:no-conservative-reading
    if audit_operation not in {"change", "split", "merge"}:
        raise AgentLoopError(f"{context}.audit_operation must be change, split, or merge.")  # shape-check: fatal:no-conservative-reading
    return PlanRevisionPatchOperation(
        op=op,
        value=_parse_matrix_metadata(payload["value"], context=f"{context}.value"),  # shape-check: fatal:no-conservative-reading
        audit_operation=audit_operation,
        rationale=_risk_bounded_string(payload["rationale"], context=f"{context}.rationale", max_bytes=2_048),  # shape-check: fatal:no-conservative-reading
    )


def parse_plan_revision_patch(value: object, *, context: str = "plan_revision_patch") -> PlanRevisionPatch:
    """Strictly parse a revision-only semantic patch contract v1."""
    if isinstance(value, PlanRevisionPatch):
        return value
    payload = _expect_object(value, context=context)  # shape-check: fatal:no-conservative-reading
    _expect_exact_keys(  # shape-check: fatal:no-conservative-reading
        payload,
        context=context,
        required={
            "schema_version", "kind", "semantic_patch_contract_version", "state",
            "summary", "prior_plan_item_dispositions", "base_round_number",
            "base_state_identity", "operations",
        },
    )
    schema_version = _expect_int(payload["schema_version"], context=f"{context}.schema_version")  # shape-check: fatal:no-conservative-reading
    if schema_version != PLAN_REVISION_PATCH_SCHEMA_VERSION:
        raise AgentLoopError(f"{context}.schema_version must be 1.")  # shape-check: fatal:kind-or-version-mismatch
    if payload["kind"] != PLAN_REVISION_PATCH_KIND:
        raise AgentLoopError(f"{context}.kind must be `{PLAN_REVISION_PATCH_KIND}`.")  # shape-check: fatal:kind-or-version-mismatch
    contract_version = _expect_int(  # shape-check: fatal:no-conservative-reading
        payload["semantic_patch_contract_version"],
        context=f"{context}.semantic_patch_contract_version",
    )
    if contract_version != SEMANTIC_PATCH_CONTRACT_VERSION:
        raise AgentLoopError(f"{context}.semantic_patch_contract_version must be 1.")  # shape-check: fatal:kind-or-version-mismatch
    state = _expect_non_empty_string(payload["state"], context=f"{context}.state")  # shape-check: fatal:no-conservative-reading
    if state != "blocking":
        raise AgentLoopError(f"{context}.state must be `blocking`.")  # shape-check: fatal:no-conservative-reading
    summary = _expect_non_empty_string(payload["summary"], context=f"{context}.summary")  # shape-check: fatal:no-conservative-reading
    dispositions = _expect_disposition_list(  # shape-check: fatal:no-conservative-reading
        payload["prior_plan_item_dispositions"],
        context=f"{context}.prior_plan_item_dispositions",
        reviewer="coder",
        allowed_same_status="same-plan",
        is_plan_review=True,
    )
    round_number = _expect_int(payload["base_round_number"], context=f"{context}.base_round_number")  # shape-check: fatal:no-conservative-reading
    if round_number < 0:
        raise AgentLoopError(f"{context}.base_round_number must be non-negative.")  # shape-check: fatal:no-conservative-reading
    identity = _expect_non_empty_string(payload["base_state_identity"], context=f"{context}.base_state_identity")  # shape-check: fatal:no-conservative-reading
    if not _PLAN_REVISION_PATCH_IDENTITY_RE.fullmatch(identity):
        raise AgentLoopError(f"{context}.base_state_identity must be a lowercase 64-hex identity.")  # shape-check: fatal:authentication-or-forgery
    operations_payload = payload["operations"]
    if not isinstance(operations_payload, list) or not operations_payload:
        raise AgentLoopError(f"{context}.operations must be a non-empty JSON array.")  # shape-check: fatal:no-conservative-reading
    if len(operations_payload) > RISK_MATRIX_MAX_CHANGES * 2:
        raise AgentLoopError(f"{context}.operations exceeds the bounded operation limit.")  # shape-check: fatal:payload-bound
    operations = tuple(
        _parse_plan_revision_patch_operation(item, context=f"{context}.operations[{index}]")  # shape-check: fatal:no-conservative-reading
        for index, item in enumerate(operations_payload)
    )
    return PlanRevisionPatch(
        schema_version=schema_version,
        kind=PLAN_REVISION_PATCH_KIND,
        semantic_patch_contract_version=contract_version,
        state=state,
        summary=summary,
        prior_plan_item_dispositions=dispositions,
        base_round_number=round_number,
        base_state_identity=identity,
        operations=operations,
    )


def validate_structured_plan_revision_patch(text: str) -> PlanRevisionPatch | None:
    """Validate a patch response with the normal plan-state footer."""
    payload = _extract_structured_plan_revision_payload(text)  # shape-check: fatal:unparseable-envelope
    if payload is None:
        return None
    if payload.get("kind") != PLAN_REVISION_PATCH_KIND:
        raise AgentLoopError(  # shape-check: fatal:unparseable-envelope
            f"Structured response kind mismatch: expected `{PLAN_REVISION_PATCH_KIND}`."
        )
    return parse_plan_revision_patch(payload)  # shape-check: fatal:no-conservative-reading
