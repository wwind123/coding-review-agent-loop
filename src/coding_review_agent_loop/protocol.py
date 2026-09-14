"""Parsing for agent response markers."""

from __future__ import annotations

import json
import hashlib
import re
import shlex
from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass, field

from .errors import AgentLoopError, IssueImplementationConflictError
from .protocol_markers import sanitize_historical_text
from .review_scheduling import normalize_fix_scope

PUBLIC_RESPONSE_MARKER = "=== AGENT_LOOP_PUBLIC_RESPONSE_BELOW ==="

HUMAN_REQUIREMENTS_ADDRESSED_MARKER = "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->"
HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK = (
    "checked the relevant GitHub discussion directly before responding"
)

STATE_RE = re.compile(r"<!--\s*AGENT_STATE:\s*(approved|blocking)\s*-->", re.I)
PLAN_STATE_RE = re.compile(r"<!--\s*AGENT_PLAN_STATE:\s*(approved|blocking)\s*-->", re.I)
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
HUMAN_REQUIREMENT_STABLE_ID_RE = re.compile(r"^hr-[0-9a-f]{64}$", re.I)


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
        "unknown",
    }
)
MACHINE_LIFECYCLE_STATES = frozenset(
    {
        "repair_required",
        "awaiting_current_head_review",
        "qualification_ready",
        "qualifying",
        "cleared",
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

    def __post_init__(self) -> None:
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
        if (
            lifecycle == "repair_required"
            and not self.failed_head_sha
            and kind in CI_MACHINE_OBLIGATION_KINDS
        ):
            raise ValueError("repair-required machine obligations need a failed head")
        if self.candidate_head_sha and self.failed_head_sha == self.candidate_head_sha:
            raise ValueError("a qualification candidate must differ from the failed head")

    @property
    def is_machine_obligation(self) -> bool:
        return self.authority in {MACHINE_AUTHORITY, UNKNOWN_MACHINE_AUTHORITY} or self.obligation_kind is not None

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
    architecture_impact: ArchitectureImpact | None = None


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
    human_requirement_dispositions: tuple["HumanRequirementDisposition", ...] = ()
    architecture_impact: ArchitectureImpact | None = None


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


def _parse_architecture_impact(value: object, *, context: str) -> ArchitectureImpact:
    payload = _expect_object(value, context=context)
    _expect_exact_keys(
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
    status = _expect_non_empty_string(payload["status"], context=f"{context}.status")
    if status not in {"changed", "unchanged"}:
        raise AgentLoopError(f"{context}.status must be `changed` or `unchanged`.")
    action = _expect_non_empty_string(
        payload.get("canonical_document_action", "no-change"),
        context=f"{context}.canonical_document_action",
    )
    rationale = _expect_non_empty_string(payload["rationale"], context=f"{context}.rationale")
    if status == "changed":
        required_changed = {
            "affected_components", "dependencies", "execution_data_flows",
            "persistence", "public_contracts", "security_boundaries",
            "canonical_document_action", "canonical_document_path",
            "canonical_document_rationale",
        }
        missing = sorted(required_changed - set(payload))
        if missing:
            raise AgentLoopError(
                f"{context} changed assessments must include: {', '.join(missing)}."
            )
    path_value = payload.get("canonical_document_path")
    if path_value is not None and (not isinstance(path_value, str) or not path_value.strip()):
        raise AgentLoopError(f"{context}.canonical_document_path must be a non-empty string or null.")
    combined = _expect_string_list(payload.get("execution_data_flows", []), context=f"{context}.execution_data_flows", item_context=context)
    execution = _expect_string_list(payload.get("execution_flows", []), context=f"{context}.execution_flows", item_context=context)
    data = _expect_string_list(payload.get("data_flows", []), context=f"{context}.data_flows", item_context=context)
    if not combined and (execution or data):
        combined = (*execution, *data)
    return ArchitectureImpact(
        status=status,
        rationale=rationale,
        affected_components=_expect_string_list(payload.get("affected_components", []), context=f"{context}.affected_components", item_context=context),
        dependencies=_expect_string_list(payload.get("dependencies", []), context=f"{context}.dependencies", item_context=context),
        execution_data_flows=combined,
        execution_flows=execution,
        data_flows=data,
        persistence=_expect_string_list(payload.get("persistence", []), context=f"{context}.persistence", item_context=context),
        public_contracts=_expect_string_list(payload.get("public_contracts", []), context=f"{context}.public_contracts", item_context=context),
        security_boundaries=_expect_string_list(payload.get("security_boundaries", []), context=f"{context}.security_boundaries", item_context=context),
        canonical_document_action=action,
        canonical_document_path=path_value,
        canonical_document_rationale=(
            _expect_non_empty_string(
                payload["canonical_document_rationale"],
                context=f"{context}.canonical_document_rationale",
            )
            if payload.get("canonical_document_rationale") not in (None, "")
            else ""
        ),
        uncertainty=_expect_string_list(payload.get("uncertainty", []), context=f"{context}.uncertainty", item_context=context),
    )


def parse_architecture_impact(value: object, *, context: str = "architecture_impact") -> ArchitectureImpact:
    """Validate an impact object for protocol extensions outside response envelopes."""
    return _parse_architecture_impact(value, context=context)


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
        raise AgentLoopError("architecture_impact must be a mapping or parsed impact object.")

    def clean(item: object) -> object:
        if isinstance(item, str):
            return sanitize_historical_text(item)
        if isinstance(item, Mapping):
            return {str(key): clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        return item

    sanitized = clean(value)
    return sanitized if isinstance(sanitized, dict) else None


HUMAN_REQUIREMENT_DISPOSITION_VALUES = frozenset(
    {"addressed", "blocked", "not-applicable"}
)


@dataclass(frozen=True)
class TestObservationCitation:
    """A coder-authored citation for a broker or parent test receipt."""

    command: str
    receipt_id: str
    claim: str


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
    risk_test_matrix_evidence: "RiskTestMatrixEvidence | None" = None


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
    risk_test_matrix_evidence: "RiskTestMatrixEvidence | None" = None


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
EXECUTION_AUTOMATION_CLASSES = frozenset({"agent-pr", "human-action", "manual-close"})


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
        clean = sanitize_historical_text

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
            self.to_payload(), separators=(",", ":"), sort_keys=True, ensure_ascii=False
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


# The matrix is deliberately a structured, generation-gated contract.  It is
# not a reviewer finding ledger: row IDs live in their own namespace and are
# never accepted as carried review-item IDs.
RISK_TEST_MATRIX_CONTRACT_VERSION = 1
RISK_MATRIX_MAX_ROWS = 24
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
RISK_MATRIX_OWNER_NAMES = frozenset({"one-shot", "retained-parent", "final-integration"})
RISK_MATRIX_EVIDENCE_STATUSES = frozenset(
    {
        "verified", "missing", "not-run", "blocked", "failed", "timed-out",
        "stale/unverified", "incomplete",
    }
)
_RISK_ROW_ID_RE = re.compile(r"^(?!.*(?:^|[-_.])(item|finding|review|blocker)[-_.]?\d)(?!hr-)[A-Za-z0-9][A-Za-z0-9._-]*$")
# Stage IDs are approved-plan identifiers, not a second, narrower namespace.
# Topology validation checks membership in the approved recommendation; this
# parser must accept valid IDs such as ``api`` and ``s1`` as well as the older
# ``stage-*``/``child-*`` spellings.
_RISK_OWNER_STAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _risk_bounded_string(value: object, *, context: str, max_bytes: int = RISK_MATRIX_MAX_FIELD_BYTES) -> str:
    rendered = _expect_non_empty_string(value, context=context)
    if len(rendered.encode("utf-8")) > max_bytes:
        raise AgentLoopError(f"{context} exceeds the {max_bytes}-byte bound.")
    return rendered


def _risk_bounded_string_list(value: object, *, context: str, max_items: int = RISK_MATRIX_MAX_LIST_ITEMS) -> tuple[str, ...]:
    rendered = _expect_string_list(value, context=context, item_context=context)
    if len(rendered) > max_items:
        raise AgentLoopError(f"{context} exceeds the {max_items}-item bound.")
    for index, item in enumerate(rendered):
        if len(item.encode("utf-8")) > RISK_MATRIX_MAX_FIELD_BYTES:
            raise AgentLoopError(f"{context}[{index}] exceeds the {RISK_MATRIX_MAX_FIELD_BYTES}-byte bound.")
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
            "row_id": sanitize_historical_text(self.row_id),
            "label": sanitize_historical_text(self.label),
            "entry_path_or_mode": sanitize_historical_text(self.entry_path_or_mode),
            "initial_state": sanitize_historical_text(self.initial_state),
            "event": sanitize_historical_text(self.event),
            "expected_outcome": sanitize_historical_text(self.expected_outcome),
            "forbidden_side_effects": [sanitize_historical_text(item) for item in self.forbidden_side_effects],
            "proposed_test_level": sanitize_historical_text(self.proposed_test_level),
            "proposed_test_location": sanitize_historical_text(self.proposed_test_location),
            "applicability": sanitize_historical_text(self.applicability),
            "related_scope_item_ids": [sanitize_historical_text(item) for item in self.related_scope_item_ids],
            "execution_owner": sanitize_historical_text(self.execution_owner),
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
            "rows": [row.to_payload() for row in self.rows],
            "important_exclusions": [sanitize_historical_text(item) for item in self.important_exclusions],
        }
        if self.not_applicable_rationale is not None:
            payload["not_applicable_rationale"] = sanitize_historical_text(self.not_applicable_rationale)
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
            "rationale": sanitize_historical_text(self.rationale),
        }


def _validate_risk_row_id(value: object, *, context: str) -> str:
    row_id = _risk_bounded_string(value, context=context, max_bytes=128)
    if not _RISK_ROW_ID_RE.fullmatch(row_id):
        raise AgentLoopError(
            f"{context} must be a matrix-specific identifier and may not resemble a reviewer finding ID."
        )
    return row_id


def _validate_risk_owner(value: object, *, context: str) -> str:
    owner = _risk_bounded_string(value, context=context, max_bytes=128)
    if owner not in RISK_MATRIX_OWNER_NAMES and not _RISK_OWNER_STAGE_RE.fullmatch(owner):
        raise AgentLoopError(
            f"{context} must be one of one-shot, retained-parent, final-integration, or a reviewed stage ID."
        )
    return owner


def _parse_risk_test_matrix(value: object, *, context: str = "risk_test_matrix") -> RiskTestMatrix:
    payload = _expect_object(value, context=context)
    _expect_exact_keys(
        payload,
        context=context,
        required={"applicability", "rows", "important_exclusions"},
        optional={"not_applicable_rationale"},
    )
    applicability = _risk_bounded_string(payload["applicability"], context=f"{context}.applicability", max_bytes=64)
    if applicability not in RISK_MATRIX_APPLICABILITY:
        raise AgentLoopError(f"{context}.applicability must be `applicable` or `not-applicable`.")
    rows_payload = payload["rows"]
    if not isinstance(rows_payload, list):
        raise AgentLoopError(f"{context}.rows must be a JSON array.")
    if len(rows_payload) > RISK_MATRIX_MAX_ROWS:
        raise AgentLoopError(f"{context}.rows exceeds the {RISK_MATRIX_MAX_ROWS}-row bound; consolidate scenarios explicitly.")
    rows: list[RiskTestMatrixRow] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(rows_payload):
        row_context = f"{context}.rows[{index}]"
        row = _expect_object(raw_row, context=row_context)
        _expect_exact_keys(
            row,
            context=row_context,
            required={
                "row_id", "label", "entry_path_or_mode", "initial_state", "event",
                "expected_outcome", "forbidden_side_effects", "proposed_test_level",
                "proposed_test_location", "applicability", "related_scope_item_ids",
                "execution_owner",
            },
        )
        row_id = _validate_risk_row_id(row["row_id"], context=f"{row_context}.row_id")
        if row_id in seen:
            raise AgentLoopError(f"{context} contains duplicate row ID `{row_id}`.")
        seen.add(row_id)
        row_applicability = _risk_bounded_string(row["applicability"], context=f"{row_context}.applicability", max_bytes=64)
        if row_applicability not in {"applicable", "required", "not-applicable"}:
            raise AgentLoopError(f"{row_context}.applicability is invalid.")
        rows.append(
            RiskTestMatrixRow(
                row_id=row_id,
                label=_risk_bounded_string(row["label"], context=f"{row_context}.label"),
                entry_path_or_mode=_risk_bounded_string(row["entry_path_or_mode"], context=f"{row_context}.entry_path_or_mode"),
                initial_state=_risk_bounded_string(row["initial_state"], context=f"{row_context}.initial_state"),
                event=_risk_bounded_string(row["event"], context=f"{row_context}.event"),
                expected_outcome=_risk_bounded_string(row["expected_outcome"], context=f"{row_context}.expected_outcome"),
                forbidden_side_effects=_risk_bounded_string_list(row["forbidden_side_effects"], context=f"{row_context}.forbidden_side_effects"),
                proposed_test_level=_risk_bounded_string(row["proposed_test_level"], context=f"{row_context}.proposed_test_level"),
                proposed_test_location=_risk_bounded_string(row["proposed_test_location"], context=f"{row_context}.proposed_test_location"),
                applicability=row_applicability,
                related_scope_item_ids=_risk_bounded_string_list(row["related_scope_item_ids"], context=f"{row_context}.related_scope_item_ids"),
                execution_owner=_validate_risk_owner(row["execution_owner"], context=f"{row_context}.execution_owner"),
            )
        )
    exclusions = _risk_bounded_string_list(
        payload["important_exclusions"], context=f"{context}.important_exclusions", max_items=RISK_MATRIX_MAX_EXCLUSIONS
    )
    rationale_value = payload.get("not_applicable_rationale")
    rationale = None if rationale_value is None else _risk_bounded_string(
        rationale_value, context=f"{context}.not_applicable_rationale", max_bytes=2_048
    )
    if applicability == "not-applicable":
        if rows:
            raise AgentLoopError(f"{context} not-applicable matrices must contain no rows.")
        if not rationale:
            raise AgentLoopError(f"{context} not-applicable matrices require a non-empty rationale.")
    elif not rows:
        raise AgentLoopError(f"{context} applicable matrices require at least one row.")
    elif not any(row.applicability in {"applicable", "required"} for row in rows):
        raise AgentLoopError(
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
        encoded = json.dumps(matrix.to_payload(), separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
        if len(encoded) > RISK_MATRIX_MAX_PAYLOAD_BYTES:
            raise AgentLoopError(f"{context} exceeds the {RISK_MATRIX_MAX_PAYLOAD_BYTES}-byte payload bound.")
        return matrix
    matrix = _parse_risk_test_matrix(value, context=context)
    encoded = json.dumps(matrix.to_payload(), separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    if len(encoded) > RISK_MATRIX_MAX_PAYLOAD_BYTES:
        raise AgentLoopError(f"{context} exceeds the {RISK_MATRIX_MAX_PAYLOAD_BYTES}-byte payload bound.")
    return matrix


def _parse_risk_test_matrix_changes(value: object, *, context: str = "risk_test_matrix_changes") -> tuple[RiskTestMatrixChange, ...]:
    if isinstance(value, dict):
        _expect_exact_keys(value, context=context, required={"changes"})
        value = value["changes"]
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array or an object containing `changes`.")
    if len(value) > RISK_MATRIX_MAX_CHANGES:
        raise AgentLoopError(f"{context} exceeds the {RISK_MATRIX_MAX_CHANGES}-change bound.")
    changes: list[RiskTestMatrixChange] = []
    for index, raw_change in enumerate(value):
        change_context = f"{context}[{index}]"
        change = _expect_object(raw_change, context=change_context)
        _expect_exact_keys(change, context=change_context, required={"operation", "row_ids", "rationale"})
        operation = _risk_bounded_string(change["operation"], context=f"{change_context}.operation", max_bytes=32)
        if operation not in RISK_MATRIX_CHANGE_OPERATIONS:
            raise AgentLoopError(f"{change_context}.operation is invalid.")
        row_ids = tuple(_validate_risk_row_id(item, context=f"{change_context}.row_ids[{i}]") for i, item in enumerate(
            _risk_bounded_string_list(change["row_ids"], context=f"{change_context}.row_ids")
        ))
        if not row_ids:
            raise AgentLoopError(f"{change_context}.row_ids must not be empty.")
        changes.append(RiskTestMatrixChange(operation=operation, row_ids=row_ids, rationale=_risk_bounded_string(change["rationale"], context=f"{change_context}.rationale", max_bytes=2_048)))
    return tuple(changes)


def parse_risk_test_matrix_changes(value: object, *, context: str = "risk_test_matrix_changes") -> tuple[RiskTestMatrixChange, ...]:
    return _parse_risk_test_matrix_changes(value, context=context)


def risk_test_matrix_identity(
    matrix: RiskTestMatrix | Mapping[str, object],
    changes: Sequence[RiskTestMatrixChange | Mapping[str, object]] = (),
    *,
    contract_version: int = RISK_TEST_MATRIX_CONTRACT_VERSION,
) -> str:
    parsed_matrix = matrix if isinstance(matrix, RiskTestMatrix) else parse_risk_test_matrix(matrix)
    parsed_changes = tuple(
        item if isinstance(item, RiskTestMatrixChange) else _parse_risk_test_matrix_changes([item])[0]
        for item in changes
    )
    payload = {
        "contract_version": contract_version,
        "matrix": parsed_matrix.to_payload(),
        "changes": [item.to_payload() for item in parsed_changes],
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def sanitize_risk_test_matrix(value: RiskTestMatrix | Mapping[str, object] | None) -> dict[str, object] | None:
    if value is None:
        return None
    parsed = value if isinstance(value, RiskTestMatrix) else parse_risk_test_matrix(value)
    return parsed.to_payload()


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
    old = previous if isinstance(previous, RiskTestMatrix) else parse_risk_test_matrix(previous, context="previous risk_test_matrix")
    new = current if isinstance(current, RiskTestMatrix) else parse_risk_test_matrix(current, context="current risk_test_matrix")
    parsed_changes = tuple(
        item if isinstance(item, RiskTestMatrixChange) else _parse_risk_test_matrix_changes([item])[0]
        for item in changes
    )
    parsed_historical_changes = [
        item if isinstance(item, RiskTestMatrixChange) else _parse_risk_test_matrix_changes([item])[0]
        for item in historical_changes
    ]
    current_changes = list(parsed_changes)
    for historical_change in parsed_historical_changes:
        try:
            current_changes.remove(historical_change)
        except ValueError:
            continue
    semantic_changed = old.to_payload() != new.to_payload()
    if approved and semantic_changed:
        raise AgentLoopError("Approved risk matrix is immutable; substantive changes require explicit replanning.")
    # The approval boundary receives the accepted draft plus the audit carried
    # by that draft, rather than a prior approved baseline.  The audit is a
    # historical record of how the draft got here, so its subjects are not
    # required to be a diff against the identical accepted payload.  Draft
    # revisions (the non-approved path) still run the complete coverage and
    # no-extra-subject checks below.
    if approved and not semantic_changed:
        return parsed_changes
    if semantic_changed and not current_changes:
        raise AgentLoopError(
            "Risk matrix changes omit audit operations; semantic changes require explicit "
            "review-visible audit operations."
        )
    old_by_id = {row.row_id: row for row in old.rows}
    new_by_id = {row.row_id: row for row in new.rows}
    changed_ids = {
        row_id for row_id in set(old_by_id) | set(new_by_id)
        if old_by_id.get(row_id) != new_by_id.get(row_id)
    }
    changed_matrix_fields = {
        field
        for field in ("applicability", "important_exclusions", "not_applicable_rationale")
        if getattr(old, field) != getattr(new, field)
    }
    covered = {row_id for change in current_changes for row_id in change.row_ids}
    if changed_matrix_fields:
        # Matrix-level semantics need a matrix-level audit operation. A row
        # operation that happens to mention one changed row cannot authorize a
        # rewrite of exclusions, applicability, or the not-applicable reason.
        # For a zero-row not-applicable matrix, ``matrix`` is the explicit
        # matrix-level audit subject (and is not a row ID).
        matrix_scope = set(old_by_id) | set(new_by_id) or {"matrix"}
        if not any(
            change.operation in {"change", "split", "merge"}
            and set(change.row_ids) == matrix_scope
            for change in current_changes
        ):
            raise AgentLoopError(
                "Risk matrix-level changes require one review-visible audit operation "
                "covering the complete matrix scope."
            )
    missing = sorted(changed_ids - covered)
    if missing:
        raise AgentLoopError("Risk matrix changes omit audit operations for: " + ", ".join(missing))
    matrix_audit_scope = (
        (set(old_by_id) | set(new_by_id)) or {"matrix"}
        if changed_matrix_fields
        else set()
    )
    extra = sorted(covered - changed_ids - matrix_audit_scope)
    if extra:
        raise AgentLoopError(
            "Risk matrix changes contain audit subjects with no corresponding semantic change: "
            + ", ".join(extra)
        )
    if approved and changed_ids:
        raise AgentLoopError("Approved risk matrix rows cannot be removed, reassigned, or weakened without a newly reviewed baseline.")
    return parsed_changes


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


@dataclass(frozen=True)
class RiskTestMatrixEvidence:
    matrix_identity: str
    rows: tuple[RiskTestMatrixEvidenceRow, ...]


def _parse_risk_evidence_citations(value: object, *, context: str) -> tuple[TestObservationCitation, ...]:
    return _expect_test_observations(value, context=context)


def parse_risk_test_matrix_evidence(
    value: object,
    *,
    matrix: RiskTestMatrix | Mapping[str, object] | None = None,
    expected_identity: str | None = None,
    authoritative_test_observations: Sequence[object] | None = None,
    expected_row_ids: Sequence[str] | None = None,
    context: str = "risk_test_matrix_evidence",
) -> RiskTestMatrixEvidence:
    payload = _expect_object(value, context=context)
    _expect_exact_keys(payload, context=context, required={"matrix_identity", "rows"})
    identity = _risk_bounded_string(payload["matrix_identity"], context=f"{context}.matrix_identity", max_bytes=128)
    if not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise AgentLoopError(f"{context}.matrix_identity must be a SHA-256 digest.")
    if expected_identity is not None and identity != expected_identity:
        raise AgentLoopError(f"{context}.matrix_identity does not match the delivered approved matrix.")
    if matrix is not None and expected_identity is None and identity != risk_test_matrix_identity(matrix):
        raise AgentLoopError(f"{context}.matrix_identity does not match the delivered matrix payload.")
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list):
        raise AgentLoopError(f"{context}.rows must be a JSON array.")
    if len(raw_rows) > RISK_MATRIX_MAX_ROWS:
        raise AgentLoopError(f"{context}.rows exceeds the {RISK_MATRIX_MAX_ROWS}-row bound.")
    parsed_matrix = parse_risk_test_matrix(matrix) if matrix is not None else None
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
        row = _expect_object(raw_row, context=row_context)
        _expect_exact_keys(
            row,
            context=row_context,
            required={
                "row_id", "status", "test_identifiers", "test_locations", "workflow_path_claim",
                "outcome_assertions", "forbidden_effect_assertions", "evidence_citations",
            },
            optional={"caveats"},
        )
        row_id = _validate_risk_row_id(row["row_id"], context=f"{row_context}.row_id")
        if row_id in seen:
            raise AgentLoopError(f"{context} maps row `{row_id}` more than once.")
        seen.add(row_id)
        status = _risk_bounded_string(row["status"], context=f"{row_context}.status", max_bytes=64)
        if status not in RISK_MATRIX_EVIDENCE_STATUSES:
            raise AgentLoopError(f"{row_context}.status is invalid.")
        parsed_row = RiskTestMatrixEvidenceRow(
            row_id=row_id,
            status=status,
            test_identifiers=_risk_bounded_string_list(row["test_identifiers"], context=f"{row_context}.test_identifiers"),
            test_locations=_risk_bounded_string_list(row["test_locations"], context=f"{row_context}.test_locations"),
            workflow_path_claim=_risk_bounded_string(row["workflow_path_claim"], context=f"{row_context}.workflow_path_claim"),
            outcome_assertions=_risk_bounded_string_list(row["outcome_assertions"], context=f"{row_context}.outcome_assertions"),
            forbidden_effect_assertions=_risk_bounded_string_list(row["forbidden_effect_assertions"], context=f"{row_context}.forbidden_effect_assertions"),
            evidence_citations=_parse_risk_evidence_citations(row["evidence_citations"], context=f"{row_context}.evidence_citations"),
            caveats=_risk_bounded_string_list(
                row.get("caveats", []),
                context=f"{row_context}.caveats",
                max_items=RISK_MATRIX_MAX_CAVEATS,
            ),
        )
        if status == "verified" and (
            not parsed_row.test_identifiers
            or not parsed_row.test_locations
            or not parsed_row.outcome_assertions
            or not parsed_row.forbidden_effect_assertions
            or not parsed_row.evidence_citations
        ):
            raise AgentLoopError(
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
                raise AgentLoopError(
                    f"{row_context} contains citations without matching authoritative "
                    + ("passing " if status == "verified" else "")
                    + "test receipts: " + ", ".join(invalid)
                )
            if expected_statuses and (
                (len(expected_statuses) == 1 and status not in expected_statuses)
                or (len(expected_statuses) > 1 and status != "incomplete")
            ):
                raise AgentLoopError(
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
                        rendered_caveat = _risk_bounded_string(
                            caveat,
                            context=f"{row_context}.authoritative_caveat",
                        )
                        if rendered_caveat not in all_caveats:
                            all_caveats.append(rendered_caveat)
                parsed_row = dataclasses.replace(
                    parsed_row,
                    caveats=_risk_bounded_string_list(
                        all_caveats,
                        context=f"{row_context}.caveats",
                        max_items=RISK_MATRIX_MAX_CAVEATS,
                    ),
                )
        result.append(parsed_row)
    if expected_ids is not None and {row.row_id for row in result} != expected_ids:
        missing = sorted(expected_ids - {row.row_id for row in result})
        extra = sorted({row.row_id for row in result} - expected_ids)
        raise AgentLoopError(f"{context} must map every delivered row exactly once (missing={missing}, extra={extra}).")
    return RiskTestMatrixEvidence(matrix_identity=identity, rows=tuple(result))


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
    return citation.command in commands


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
    }
    rich = isinstance(attribution, (Mapping,)) or attribution is not None or any(
        name in (projected_value or {})
        for name in ("environment", "superseded_by", "caveats", "wrapper_bootstrap", "inner_exec", "suite_start")
    )
    return semantics, rich


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
    payload: dict[str, object], *, context: str, required: bool = False
) -> tuple[int | None, RiskTestMatrix | None, tuple[RiskTestMatrixChange, ...]]:
    present = {
        name: name in payload
        for name in ("risk_test_matrix_contract_version", "risk_test_matrix", "risk_test_matrix_changes")
    }
    if any(present.values()) and not all(present.values()):
        raise AgentLoopError(
            f"{context} risk matrix fields must include risk_test_matrix_contract_version, "
            "risk_test_matrix, and risk_test_matrix_changes together."
        )
    if not any(present.values()):
        if required:
            raise AgentLoopError(
                f"Fresh {context} responses require risk_test_matrix_contract_version: 1 "
                "and a complete risk_test_matrix contract."
            )
        return None, None, ()
    version = _expect_int(
        payload["risk_test_matrix_contract_version"],
        context=f"{context}.risk_test_matrix_contract_version",
    )
    if version != RISK_TEST_MATRIX_CONTRACT_VERSION:
        raise AgentLoopError(f"Unsupported {context} risk_test_matrix_contract_version: {version}.")
    matrix = parse_risk_test_matrix(payload["risk_test_matrix"], context=f"{context}.risk_test_matrix")
    changes = parse_risk_test_matrix_changes(
        payload["risk_test_matrix_changes"], context=f"{context}.risk_test_matrix_changes"
    )
    return version, matrix, changes

@dataclass(frozen=True)
class StructuredDiscussReview:
    schema_version: int
    kind: str
    outcome: str
    rationale: str
    split_proposals: tuple[str, ...]
    rebuttal: str | None = None
    analyzer_framing: str | None = None
    framing_note: str | None = None


@dataclass(frozen=True)
class DiscussUnresolvedItem:
    """A classified unresolved concern from an answer-mode discuss vote.

    This is intentionally distinct from ``UnresolvedReviewItem``, which is the
    PR/plan-review ledger type and has unrelated lifecycle semantics.
    """

    status: str
    text: str


DISCUSS_UNRESOLVED_ITEM_STATUS_VALUES = frozenset(
    {"blocker", "human-decision", "follow-up"}
)


@dataclass(frozen=True)
class StructuredDiscussAnswer:
    schema_version: int
    kind: str
    position: str
    rationale: str
    confidence: str
    unresolved_items: tuple[DiscussUnresolvedItem, ...]
    answer: str | None = None
    rebuttal: str | None = None
    analyzer_framing: str | None = None
    framing_note: str | None = None


DISCUSS_EVIDENCE_STATUS_VALUES = frozenset({"verified", "reported-but-unverified", "missing"})
DISCUSS_EVIDENCE_UPDATE_ACTIONS = frozenset({"retract", "supersede"})
DISCUSS_VERIFICATION_BASES = frozenset({"external-source-inspected", "checkout-inspected"})


@dataclass(frozen=True)
class DiscussEvidenceClaim:
    """A debater-authored observation.  IDs are assigned by the orchestrator."""

    fact: str
    status: str
    source: str | None = None
    verification_basis: str | None = None


@dataclass(frozen=True)
class DiscussEvidenceUpdate:
    action: str
    target_observation_id: str
    reason: str
    replacement_claim_index: int | None = None


@dataclass(frozen=True)
class ParsedDiscussAnswer:
    position: str
    rationale: str
    confidence: str
    unresolved_items: tuple[DiscussUnresolvedItem, ...]
    reviewer: str
    answer: str | None = None
    rebuttal: str | None = None
    analyzer_framing: str | None = None
    framing_note: str | None = None
    research_status: str | None = None
    sourced_facts: tuple[DiscussSourcedFact, ...] = ()
    research_target: str | None = None
    research_questions: tuple[str, ...] = ()
    evidence_claims: tuple[DiscussEvidenceClaim, ...] = ()
    evidence_updates: tuple[DiscussEvidenceUpdate, ...] = ()


@dataclass(frozen=True)
class DiscussSemanticEvidence:
    reviewer: str
    supports: str


@dataclass(frozen=True)
class ParsedDiscussSemanticComparison:
    classification: str
    shared_recommendation: str
    remaining_decisions: tuple[str, ...]
    evidence: tuple[DiscussSemanticEvidence, ...]


@dataclass(frozen=True)
class ParsedDiscussAnswerConfirmation:
    reviewer: str
    decision: str
    rationale: str
    answer: str | None = None


@dataclass(frozen=True)
class ParsedDiscussEvidenceReconciliation:
    groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ParsedFailedDiscussResponse:
    reviewer: str
    category: str
    result_mode: str = "triage"


@dataclass(frozen=True)
class DiscussSourcedFact:
    fact: str
    source: str


@dataclass(frozen=True)
class ParsedDiscussReview:
    outcome: str
    rationale: str
    split_proposals: tuple[str, ...]
    reviewer: str
    rebuttal: str | None = None
    analyzer_framing: str | None = None
    framing_note: str | None = None
    # Research policy fields (#477). None means the response carried no
    # `research` object (legacy transcripts and `none`-mode responses).
    research_status: str | None = None
    sourced_facts: tuple[DiscussSourcedFact, ...] = ()
    research_target: str | None = None
    research_questions: tuple[str, ...] = ()
    evidence_claims: tuple[DiscussEvidenceClaim, ...] = ()
    evidence_updates: tuple[DiscussEvidenceUpdate, ...] = ()


ParsedDiscussResponse = ParsedDiscussReview | ParsedDiscussAnswer | ParsedFailedDiscussResponse


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


DISCUSS_OUTCOME_VALUES = frozenset({"implement", "do-not-implement", "needs-human", "split"})
DISCUSS_ANALYZER_FRAMING_VALUES = frozenset({"accurate", "misframed"})
DISCUSS_RESEARCH_STATUS_VALUES = frozenset({"sourced", "not-needed", "unavailable", "inconclusive"})
DISCUSS_RESEARCH_TARGET_VALUES = frozenset(
    {
        "example-validation",
        "solution-design",
        "cost-latency",
        "implementation-feasibility",
        "policy/legal/current-facts",
    }
)

# Internal-only outcome for debaters that failed or timed out in a partial
# round (#475). Deliberately NOT in DISCUSS_OUTCOME_VALUES: real agent
# responses cannot claim it. Placeholders are built only by the orchestrator
# and by resume; their presence in a round blocks false-positive consensus.
DISCUSS_FAILED_OUTCOME = "failed"


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
class DiscussAgendaDisagreement:
    topic: str
    positions: tuple[tuple[str, str], ...]
    question_for_next_round: str


DISCUSS_SYNTHESIS_MAX_ENTRIES = 8
DISCUSS_SYNTHESIS_MAX_NEXT_ROUND_FOCUS = 5
DISCUSS_SYNTHESIS_MAX_TEXT_BYTES = 512
DISCUSS_SYNTHESIS_MAX_CANONICAL_BYTES = 16_000
DISCUSS_SYNTHESIS_CHANGE_KINDS = frozenset(
    {"introduced", "resolved", "reopened", "retracted", "refined"}
)
DISCUSS_SYNTHESIS_CLASSIFICATIONS = frozenset(
    {"consensus", "near_consensus", "material_deadlock"}
)


@dataclass(frozen=True)
class DiscussSynthesisResponseReference:
    """A response that independently supports a synthesized statement."""

    reviewer: str
    round: int


@dataclass(frozen=True)
class DiscussSynthesisConsensus:
    text: str
    references: tuple[DiscussSynthesisResponseReference, ...]


@dataclass(frozen=True)
class DiscussSynthesisPosition:
    reviewers: tuple[str, ...]
    position: str


@dataclass(frozen=True)
class DiscussSynthesisDisagreement:
    topic: str
    positions: tuple[DiscussSynthesisPosition, ...]
    decision_needed: str


@dataclass(frozen=True)
class DiscussSynthesisChange:
    kind: str
    topic: str
    text: str
    references: tuple[DiscussSynthesisResponseReference, ...]


@dataclass(frozen=True)
class ParsedDiscussRoundSynthesis:
    """Validated cumulative state for one completed answer-mode round."""

    consensus: tuple[DiscussSynthesisConsensus, ...]
    disagreements: tuple[DiscussSynthesisDisagreement, ...]
    changes: tuple[DiscussSynthesisChange, ...]
    missing_facts: tuple[str, ...]
    next_round_focus: tuple[str, ...]
    responding_reviewers: tuple[str, ...]


@dataclass(frozen=True)
class ParsedDiscussFinalSynthesis:
    """Validated executive conclusion for an answer-mode final result."""

    classification: str
    agreed_conclusions: tuple[DiscussSynthesisConsensus, ...]
    remaining_disagreements: tuple[DiscussSynthesisDisagreement, ...]
    next_action: str


# Short aliases keep the public protocol vocabulary easy to discover while
# retaining the explicit response-reference name in serialized models.
DiscussSynthesisReference = DiscussSynthesisResponseReference
DiscussRoundSynthesis = ParsedDiscussRoundSynthesis
DiscussFinalSynthesis = ParsedDiscussFinalSynthesis


@dataclass(frozen=True)
class ParsedDiscussAgenda:
    consensus: tuple[str, ...]
    disagreements: tuple[DiscussAgendaDisagreement, ...]
    missing_facts: tuple[str, ...]
    # Shared research brief (#477). Emitted by the analyzer when a research
    # policy is active; forwarded to the next round's debaters so parallel
    # turns do not duplicate work.
    research_required: bool = False
    research_questions: tuple[str, ...] = ()
    research_question_targets: tuple[str, ...] = ()
    # Optional answer-mode presentation state. Legacy agendas deliberately
    # remain valid and have no synthesis value.
    round_synthesis: ParsedDiscussRoundSynthesis | None = None


@dataclass(frozen=True)
class ParsedHumanRequirementsAcknowledgement:
    marker_present: bool
    section_present: bool
    addressed_ids: tuple[str, ...]
    section_text: str


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


def _normalize_requirement_label(text: str) -> str:
    stable = HUMAN_REQUIREMENT_STABLE_ID_RE.fullmatch(text.strip())
    if stable:
        return stable.group(0).lower()
    legacy = re.fullmatch(r"\s*Requirement\s+(\d+)\s*", text, re.I)
    if not legacy:
        raise AgentLoopError(
            f"Invalid human requirement label: {text}. Only exact surfaced signed labels like "
            "`hr-<64 hexadecimal characters>` are valid; issue acceptance criteria, reviewer item IDs, reviewer "
            "comments, and arbitrary labels are not signed human requirements."
        )
    return f"Requirement {legacy.group(1)}"


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
        raise AgentLoopError(
            "Coder response uses legacy positional signed-requirement label(s) "
            f"{', '.join(legacy)}. Their meaning cannot be inferred from the current requirement "
            "set; provide a fresh acknowledgement using the exact surfaced stable hr-... IDs."
        )


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
                _normalize_requirement_label(stable.group(0) if stable else label)
            )

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
    dispositions: Sequence[HumanRequirementDisposition] | None = None,
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
        if checked_discussion_directly:
            raise AgentLoopError(
                "Coder response must set `human_requirements.checked_discussion_directly` to false "
                "when no signed human requirements are surfaced and direct discussion acknowledgement is not required."
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
    _reject_legacy_requirement_labels(
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
        if dispositions is None:
            expected_addressed_ids = set(expected_ids)
            missing_message = "Coder response did not address all surfaced signed human requirement IDs: "
        else:
            expected_addressed_ids = {
                _normalize_requirement_label(item.requirement_id)
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
            raise AgentLoopError(missing_message + ", ".join(missing))
        if unexpected:
            raise AgentLoopError(
                "Coder response human_requirements.addressed_ids may contain only requirements "
                "with an `addressed` disposition: "
                + ", ".join(unexpected)
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


def _expect_test_observations(
    value: object, *, context: str
) -> tuple[TestObservationCitation, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")
    result: list[TestObservationCitation] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        payload = _expect_object(item, context=item_context)
        _expect_exact_keys(
            payload,
            context=item_context,
            required={"command", "receipt_id", "claim"},
        )
        claim = _expect_non_empty_string(payload["claim"], context=f"{item_context}.claim")
        if claim not in {"current-result", "base-reproduction"}:
            raise AgentLoopError(
                f"{item_context}.claim must be `current-result` or `base-reproduction`."
            )
        result.append(
            TestObservationCitation(
                command=_expect_non_empty_string(payload["command"], context=f"{item_context}.command"),
                receipt_id=_expect_non_empty_string(payload["receipt_id"], context=f"{item_context}.receipt_id"),
                claim=claim,
            )
        )
    return tuple(result)


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


def _expect_review_finding_list(
    payload: dict[str, object],
    field_name: str,
    *,
    context: str,
    reviewer: str,
) -> tuple[ApprovedFollowup, ...]:
    """Accept legacy strings and the scoped PR finding representation."""
    value = payload.get(field_name, [])
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")
    findings: list[ApprovedFollowup] = []
    for index, raw in enumerate(value):
        item_context = f"{context} at index {index}"
        if isinstance(raw, str):
            text = _expect_non_empty_string(raw, context=item_context)
            scope = None
        else:
            item = _expect_object(raw, context=item_context)
            _expect_exact_keys(item, context=item_context, required={"text"}, optional={"fix_scope"})
            text = _expect_non_empty_string(item["text"], context=f"{item_context}.text")
            try:
                scope = normalize_fix_scope(item.get("fix_scope")) if "fix_scope" in item else None
            except AgentLoopError as exc:
                raise AgentLoopError(f"{item_context}.fix_scope is invalid: {exc}") from exc
        findings.append(ApprovedFollowup(reviewer=reviewer, text=text, fix_scope=scope))
    return tuple(findings)


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
        raise AgentLoopError(f"{context} must be a JSON array.")
    ids: set[int] = set()
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise AgentLoopError(
                f"{context} item at index {index} must be a unique positive integer (not a bool)."
            )
        if item in ids:
            raise AgentLoopError(f"{context} contains duplicate issue ID #{item}.")
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
        raise AgentLoopError(f"{context} must be a JSON array.")
    stages: list[DeferredStage] = []
    seen_title_keys: set[str] = set()
    for index, item in enumerate(value):
        item_context = f"{context} at index {index}"
        item_payload = _expect_object(item, context=item_context)
        _expect_exact_keys(item_payload, context=item_context, required={"title", "summary"})
        title = _expect_non_empty_string(item_payload["title"], context=f"{item_context}.title")
        title_key = " ".join(title.lower().split())
        if title_key in seen_title_keys:
            raise AgentLoopError(f"{context} has a duplicate stage title: {title!r}.")
        seen_title_keys.add(title_key)
        stages.append(
            DeferredStage(
                title=title,
                summary=_expect_non_empty_string(
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
    _expect_deferred_stage_list(payload, "deferred_stages", context=f"{context}.deferred_stages")
    if not present:
        return TypedPlanStages()

    def parse_child_stages() -> tuple[ChildStage, ...]:
        value = payload.get("child_stages", [])
        if not isinstance(value, list):
            raise AgentLoopError(f"{context}.child_stages must be a JSON array.")
        result: list[ChildStage] = []
        for index, item in enumerate(value):
            item_context = f"{context}.child_stages at index {index}"
            obj = _expect_object(item, context=item_context)
            _expect_exact_keys(obj, context=item_context, required={"title", "summary"})
            title = _expect_non_empty_string(obj["title"], context=f"{item_context}.title")
            summary = _expect_non_empty_string(obj["summary"], context=f"{item_context}.summary")
            if ISSUE_REFERENCE_RE.search(title + "\n" + summary):
                raise AgentLoopError(
                    f"{item_context} references an existing issue; "
                    "use external_dependencies."
                )
            if TRACKER_ACTION_TITLE_RE.match(" ".join(title.split())):
                raise AgentLoopError(
                    f"{item_context} is a tracker action; use plan_actions."
                )
            result.append(ChildStage(title, summary))
        return tuple(result)

    def recorded_stages(name: str) -> tuple[DeferredStage, ...]:
        value = payload.get(name, [])
        if not isinstance(value, list):
            raise AgentLoopError(f"{context}.{name} must be a JSON array.")
        result: list[DeferredStage] = []
        for index, item in enumerate(value):
            item_context = f"{context}.{name} at index {index}"
            obj = _expect_object(item, context=item_context)
            _expect_exact_keys(obj, context=item_context, required={"title", "summary"})
            title = _expect_non_empty_string(obj["title"], context=f"{item_context}.title")
            summary = _expect_non_empty_string(obj["summary"], context=f"{item_context}.summary")
            result.append(DeferredStage(title, summary))
        return tuple(result)

    child_stages = parse_child_stages()
    dependencies = recorded_stages("external_dependencies")
    deferred = recorded_stages("deferred_work")
    actions = recorded_stages("plan_actions")
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
                raise AgentLoopError(
                    f"{context} has a duplicate title in {seen[key]} and {category}: "
                    f"{entry.title!r}."
                )
            seen[key] = category
    return TypedPlanStages(child_stages, dependencies, deferred, actions)


def _expect_execution_allocation(
    value: object, *, context: str
) -> ExecutionAllocation:
    payload = _expect_object(value, context=context)
    _expect_exact_keys(
        payload,
        context=context,
        required={"status", "deliverables", "acceptance_criteria", "covered_scope_item_ids"},
    )
    status = _expect_non_empty_string(payload["status"], context=f"{context}.status")
    if status not in {"none", "required"}:
        raise AgentLoopError(f"{context}.status must be `none` or `required`.")
    deliverables = _expect_string_list(
        payload["deliverables"], context=f"{context}.deliverables",
        item_context=f"{context}.deliverables",
    )
    criteria = _expect_string_list(
        payload["acceptance_criteria"], context=f"{context}.acceptance_criteria",
        item_context=f"{context}.acceptance_criteria",
    )
    covered = _expect_item_id_list(
        payload["covered_scope_item_ids"], context=f"{context}.covered_scope_item_ids"
    )
    if status == "none" and (deliverables or criteria or covered):
        raise AgentLoopError(f"{context} with status `none` must have empty arrays.")
    if status == "required" and (not deliverables or not criteria or not covered):
        raise AgentLoopError(
            f"{context} with status `required` needs non-empty deliverables, "
            "acceptance_criteria, and covered_scope_item_ids."
        )
    return ExecutionAllocation(status, deliverables, criteria, covered)


def _expect_execution_recommendation(
    value: object, *, context: str
) -> ExecutionStrategyRecommendation:
    payload = _expect_object(value, context=context)
    required = {
        "strategy", "rationale", "staging_feasibility", "scope_items",
        "coupling_constraints", "child_stages", "retained_parent_work",
        "final_integration_work", "caveats",
    }
    _expect_exact_keys(payload, context=context, required=required, optional={"one_shot_delivery"})
    strategy = _expect_non_empty_string(payload["strategy"], context=f"{context}.strategy")
    if strategy not in {"one-shot", "staged"}:
        raise AgentLoopError(f"{context}.strategy must be `one-shot` or `staged`.")
    rationale = _expect_non_empty_string(payload["rationale"], context=f"{context}.rationale")
    feasibility = _expect_non_empty_string(
        payload["staging_feasibility"], context=f"{context}.staging_feasibility"
    )
    if feasibility not in {"safe", "inseparable"}:
        raise AgentLoopError(
            f"{context}.staging_feasibility must be `safe` or `inseparable`."
        )

    scope_payload = payload["scope_items"]
    if not isinstance(scope_payload, list) or not scope_payload:
        raise AgentLoopError(f"{context}.scope_items must be a non-empty JSON array.")
    scope_items: list[ExecutionScopeItem] = []
    scope_ids: set[str] = set()
    for index, raw_item in enumerate(scope_payload):
        item_context = f"{context}.scope_items[{index}]"
        item = _expect_object(raw_item, context=item_context)
        _expect_exact_keys(item, context=item_context, required={"scope_item_id", "requirement", "acceptance_criteria"})
        item_id = _expect_item_id(item["scope_item_id"], context=f"{item_context}.scope_item_id")
        if item_id in scope_ids:
            raise AgentLoopError(f"{context}.scope_items has duplicate ID `{item_id}`.")
        criteria = _expect_string_list(
            item["acceptance_criteria"], context=f"{item_context}.acceptance_criteria",
            item_context=f"{item_context}.acceptance_criteria", min_length=1,
        )
        scope_ids.add(item_id)
        scope_items.append(
            ExecutionScopeItem(
                scope_item_id=item_id,
                requirement=_expect_non_empty_string(item["requirement"], context=f"{item_context}.requirement"),
                acceptance_criteria=criteria,
            )
        )

    coupling_payload = payload["coupling_constraints"]
    if not isinstance(coupling_payload, list):
        raise AgentLoopError(f"{context}.coupling_constraints must be a JSON array.")
    coupling_constraints: list[ExecutionCouplingConstraint] = []
    coupling_ids: set[str] = set()
    for index, raw_constraint in enumerate(coupling_payload):
        item_context = f"{context}.coupling_constraints[{index}]"
        item = _expect_object(raw_constraint, context=item_context)
        _expect_exact_keys(item, context=item_context, required={"constraint_id", "scope_item_ids", "rationale"})
        constraint_id = _expect_item_id(item["constraint_id"], context=f"{item_context}.constraint_id")
        if constraint_id in coupling_ids:
            raise AgentLoopError(f"{context}.coupling_constraints has duplicate ID `{constraint_id}`.")
        ids = _expect_item_id_list(item["scope_item_ids"], context=f"{item_context}.scope_item_ids")
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise AgentLoopError(f"{item_context}.scope_item_ids must contain at least two unique IDs.")
        unknown = sorted(set(ids) - scope_ids)
        if unknown:
            raise AgentLoopError(f"{item_context}.scope_item_ids contains unknown IDs: {', '.join(unknown)}.")
        coupling_ids.add(constraint_id)
        coupling_constraints.append(
            ExecutionCouplingConstraint(
                constraint_id=constraint_id,
                scope_item_ids=ids,
                rationale=_expect_non_empty_string(item["rationale"], context=f"{item_context}.rationale"),
            )
        )

    child_payload = payload["child_stages"]
    if not isinstance(child_payload, list):
        raise AgentLoopError(f"{context}.child_stages must be a JSON array.")
    child_stages: list[ExecutionChildStage] = []
    stage_ids: set[str] = set()
    for index, raw_stage in enumerate(child_payload):
        item_context = f"{context}.child_stages[{index}]"
        item = _expect_object(raw_stage, context=item_context)
        _expect_exact_keys(
            item,
            context=item_context,
            required={
                "stage_id", "position", "title", "summary", "deliverables", "non_goals",
                "acceptance_criteria", "depends_on_stage_ids", "dependency_notes", "automation",
                "rollout_risk", "compatibility_constraints", "covered_scope_item_ids",
            },
        )
        stage_id = _expect_item_id(item["stage_id"], context=f"{item_context}.stage_id")
        if stage_id in stage_ids:
            raise AgentLoopError(f"{context}.child_stages has duplicate stage ID `{stage_id}`.")
        position = _expect_int(item["position"], context=f"{item_context}.position")
        if position != index + 1:
            raise AgentLoopError(f"{item_context}.position must be contiguous and ordered starting at 1.")
        depends = _expect_item_id_list(item["depends_on_stage_ids"], context=f"{item_context}.depends_on_stage_ids")
        unknown_dependencies = sorted(set(depends) - stage_ids)
        if unknown_dependencies:
            raise AgentLoopError(
                f"{item_context}.depends_on_stage_ids must reference earlier stages; "
                f"unknown/later IDs: {', '.join(unknown_dependencies)}."
            )
        automation = _expect_non_empty_string(item["automation"], context=f"{item_context}.automation")
        if automation not in EXECUTION_AUTOMATION_CLASSES:
            raise AgentLoopError(
                f"{item_context}.automation must be one of: {', '.join(sorted(EXECUTION_AUTOMATION_CLASSES))}."
            )
        covered = _expect_item_id_list(item["covered_scope_item_ids"], context=f"{item_context}.covered_scope_item_ids")
        if not covered:
            raise AgentLoopError(f"{item_context}.covered_scope_item_ids must be non-empty.")
        unknown_coverage = sorted(set(covered) - scope_ids)
        if unknown_coverage:
            raise AgentLoopError(f"{item_context}.covered_scope_item_ids contains unknown IDs: {', '.join(unknown_coverage)}.")
        stage_ids.add(stage_id)
        child_stages.append(
            ExecutionChildStage(
                stage_id=stage_id,
                position=position,
                title=_expect_non_empty_string(item["title"], context=f"{item_context}.title"),
                summary=_expect_non_empty_string(item["summary"], context=f"{item_context}.summary"),
                deliverables=_expect_string_list(item["deliverables"], context=f"{item_context}.deliverables", item_context=f"{item_context}.deliverables", min_length=1),
                non_goals=_expect_string_list(item["non_goals"], context=f"{item_context}.non_goals", item_context=f"{item_context}.non_goals"),
                acceptance_criteria=_expect_string_list(item["acceptance_criteria"], context=f"{item_context}.acceptance_criteria", item_context=f"{item_context}.acceptance_criteria", min_length=1),
                depends_on_stage_ids=depends,
                dependency_notes=_expect_non_empty_string(item["dependency_notes"], context=f"{item_context}.dependency_notes"),
                automation=automation,
                rollout_risk=_expect_non_empty_string(item["rollout_risk"], context=f"{item_context}.rollout_risk"),
                compatibility_constraints=_expect_string_list(item["compatibility_constraints"], context=f"{item_context}.compatibility_constraints", item_context=f"{item_context}.compatibility_constraints"),
                covered_scope_item_ids=covered,
            )
        )

    retained = _expect_execution_allocation(payload["retained_parent_work"], context=f"{context}.retained_parent_work")
    final = _expect_execution_allocation(payload["final_integration_work"], context=f"{context}.final_integration_work")
    caveats = _expect_string_list(payload["caveats"], context=f"{context}.caveats", item_context=f"{context}.caveats")
    one_shot: ExecutionOneShotDelivery | None = None
    if strategy == "one-shot":
        if "one_shot_delivery" not in payload:
            raise AgentLoopError(f"{context}.one_shot_delivery is required for one-shot recommendations.")
        if child_stages:
            raise AgentLoopError("one-shot recommendations must not contain child stages.")
        if retained.status != "none" or final.status != "none":
            raise AgentLoopError("one-shot recommendations require none retained-parent and final-integration work.")
        delivery = _expect_object(payload["one_shot_delivery"], context=f"{context}.one_shot_delivery")
        _expect_exact_keys(delivery, context=f"{context}.one_shot_delivery", required={"deliverables", "acceptance_criteria", "covered_scope_item_ids"})
        one_shot = ExecutionOneShotDelivery(
            deliverables=_expect_string_list(delivery["deliverables"], context=f"{context}.one_shot_delivery.deliverables", item_context=f"{context}.one_shot_delivery.deliverables", min_length=1),
            acceptance_criteria=_expect_string_list(delivery["acceptance_criteria"], context=f"{context}.one_shot_delivery.acceptance_criteria", item_context=f"{context}.one_shot_delivery.acceptance_criteria", min_length=1),
            covered_scope_item_ids=_expect_item_id_list(delivery["covered_scope_item_ids"], context=f"{context}.one_shot_delivery.covered_scope_item_ids"),
        )
        allocations = [one_shot.covered_scope_item_ids]
    else:
        if "one_shot_delivery" in payload:
            raise AgentLoopError("staged recommendations must not contain one_shot_delivery.")
        if feasibility != "safe":
            raise AgentLoopError("staged recommendations require staging_feasibility `safe`.")
        if not child_stages:
            raise AgentLoopError("staged recommendations require at least one child stage.")
        required_allocations = sum(
            allocation.status == "required" for allocation in (retained, final)
        ) + sum(bool(stage.covered_scope_item_ids) for stage in child_stages)
        if required_allocations < 2:
            raise AgentLoopError(
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
                raise AgentLoopError(f"{context} covers unknown scope item `{scope_id}`.")
            if scope_id in allocation_owner:
                raise AgentLoopError(f"{context} covers scope item `{scope_id}` more than once.")
            allocation_owner[scope_id] = allocation_index
    missing = sorted(scope_ids - set(allocation_owner))
    if missing:
        raise AgentLoopError(f"{context} leaves scope items uncovered: {', '.join(missing)}.")
    for constraint in coupling_constraints:
        owners = {allocation_owner[item_id] for item_id in constraint.scope_item_ids}
        if len(owners) != 1:
            raise AgentLoopError(
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
    payload: dict[str, object], *, context: str, required: bool
) -> tuple[int | None, ExecutionStrategyRecommendation | None]:
    has_version = "execution_strategy_contract_version" in payload
    has_recommendation = "execution_recommendation" in payload
    if has_version != has_recommendation:
        raise AgentLoopError(
            f"{context} must include execution_strategy_contract_version and "
            "execution_recommendation together."
        )
    if not has_version:
        if required:
            raise AgentLoopError(
                f"Fresh {context} responses require execution_strategy_contract_version: 1 "
                "and a complete execution_recommendation."
            )
        return None, None
    version = _expect_int(payload["execution_strategy_contract_version"], context=f"{context}.execution_strategy_contract_version")
    if version != EXECUTION_STRATEGY_CONTRACT_VERSION:
        raise AgentLoopError(f"{context}.execution_strategy_contract_version must be 1.")
    recommendation = _expect_execution_recommendation(
        payload["execution_recommendation"], context=f"{context}.execution_recommendation"
    )
    # Generation 1 has one reviewed topology. The legacy top-level
    # ``child_stages`` category is executable only for unversioned historical
    # plans; allowing it beside a v1 recommendation would create two
    # competing topologies and let a legacy splitter mutate a fresh plan.
    # Keep an explicitly empty legacy category harmless for callers that still
    # emit the shared optional key.
    if "child_stages" in payload and payload["child_stages"] != []:
        raise AgentLoopError(
            f"{context} cannot combine a generation-1 execution recommendation "
            "with non-empty top-level legacy child_stages; use only the reviewed "
            "execution_recommendation topology."
        )
    return version, recommendation


def parse_execution_recommendation_payload(
    value: object, *, context: str = "execution_recommendation"
) -> ExecutionStrategyRecommendation:
    """Validate a recovered recommendation before any bounded repair."""
    return _expect_execution_recommendation(value, context=context)


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


def _expect_human_requirement_dispositions(
    value: object,
    *,
    context: str,
) -> tuple[HumanRequirementDisposition, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")
    result: list[HumanRequirementDisposition] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        payload = _expect_object(item, context=item_context)
        _expect_exact_keys(
            payload,
            context=item_context,
            required={"requirement_id", "disposition", "evidence"},
        )
        requirement_id = _normalize_requirement_label(
            _expect_non_empty_string(payload["requirement_id"], context=f"{item_context}.requirement_id")
        )
        disposition = _expect_non_empty_string(
            payload["disposition"], context=f"{item_context}.disposition"
        )
        if disposition not in HUMAN_REQUIREMENT_DISPOSITION_VALUES:
            raise AgentLoopError(
                f"{item_context}.disposition must be one of: addressed, blocked, not-applicable"
            )
        evidence = _expect_non_empty_string(
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
    expected = tuple(_normalize_requirement_label(item) for item in surfaced_requirement_ids)
    actual = [item.requirement_id for item in dispositions]
    _reject_legacy_requirement_labels(actual, surfaced_requirement_ids=surfaced_requirement_ids)
    duplicates = sorted({item for item in actual if actual.count(item) > 1})
    if duplicates:
        raise AgentLoopError(f"{context} contains duplicate requirement ID(s): {', '.join(duplicates)}")
    unknown = sorted(set(actual) - set(expected))
    if unknown:
        raise AgentLoopError(f"{context} contains unknown requirement ID(s): {', '.join(unknown)}")
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise AgentLoopError(f"{context} is missing requirement ID(s): {', '.join(missing)}")
    if not expected and actual:
        raise AgentLoopError(f"{context} must be empty when no signed human requirements are surfaced.")


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


def _consume_agent_unavailable_footer_and_signature(
    *, payload: dict[str, object], trailing: str
) -> dict[str, object]:
    trailing = trailing.lstrip()
    marker_match = _STANDALONE_AGENT_UNAVAILABLE_RE.search(trailing)
    if marker_match is None:
        raise AgentLoopError(
            "Structured agent-unavailable response must place <!-- AGENT_UNAVAILABLE --> "
            "after the JSON object."
        )
    if trailing[: marker_match.start()].strip():
        raise AgentLoopError(
            "Structured agent-unavailable response may not include prose between the JSON object "
            "and the AGENT_UNAVAILABLE footer."
        )
    signature = trailing[marker_match.end() :].lstrip()
    signature_match = re.match(r"^--\s+\S[^\n]*(?:\n)?$", signature)
    if signature_match is None or signature[signature_match.end() :].strip():
        raise AgentLoopError(
            "Structured agent-unavailable response may not include trailing prose after "
            "the footer and signature."
        )
    return payload


def parse_agent_unavailable(text: str) -> AgentUnavailable | None:
    """Parse the shared, terminal agent-unavailable response envelope."""
    text, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(text)
    if extracted is None:
        return None
    payload, trailing = extracted
    if payload.get("kind") != "agent_unavailable":
        return None
    _consume_agent_unavailable_footer_and_signature(payload=payload, trailing=trailing)
    _require_supported_schema_version(payload)
    _expect_exact_keys(
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
    category = _expect_non_empty_string(
        payload["category"], context="agent_unavailable.category"
    )
    if category not in AGENT_UNAVAILABLE_CATEGORIES:
        allowed = ", ".join(sorted(AGENT_UNAVAILABLE_CATEGORIES))
        raise AgentLoopError(
            f"agent_unavailable.category must be one of: {allowed}."
        )
    return AgentUnavailable(
        schema_version=_expect_int(payload["schema_version"], context="schema_version"),
        kind="agent_unavailable",
        retryable=_expect_bool(payload["retryable"], context="agent_unavailable.retryable"),
        category=category,
        summary=_expect_non_empty_string(payload["summary"], context="agent_unavailable.summary"),
        suggested_action=_expect_non_empty_string(
            payload["suggested_action"], context="agent_unavailable.suggested_action"
        ),
    )


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
        allow_human_requirements_prefix=True,
    )


def _extract_structured_discuss_review_payload(
    text: str, *, context_label: str = "Structured discuss review"
) -> dict[str, object] | None:
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
        context_label=context_label,
    )
    if result is None:
        return None
    footer_match = PLAN_STATE_RE.search(trailing)
    if footer_match is not None:
        footer_state = footer_match.group(1).lower()
        if footer_state != "approved":
            raise AgentLoopError(
                f"{context_label} footer AGENT_PLAN_STATE must be `approved`; got `{footer_state}`."
            )
    return result


def _extract_structured_discuss_agenda_payload(text: str) -> dict[str, object] | None:
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
        context_label="Structured discuss agenda",
    )
    if result is None:
        return None
    footer_match = PLAN_STATE_RE.search(trailing)
    if footer_match is not None:
        footer_state = footer_match.group(1).lower()
        if footer_state != "approved":
            raise AgentLoopError(
                f"Structured discuss agenda footer AGENT_PLAN_STATE must be `approved`; got `{footer_state}`."
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


def _extract_structured_issue_implementation_payload(
    text: str,
) -> dict[str, object] | None:
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
        context_label="Structured issue implementation",
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
    architecture_impact: ArchitectureImpact | None = None,
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
        architecture_impact=architecture_impact,
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
        human_requirement_dispositions=human_requirement_dispositions,
        architecture_impact=architecture_impact,
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
        optional={"blocking_items", "same_pr_followups", "future_followups", "architecture_impact"},
    )
    state = _expect_state(payload["state"], context="pr_review.state")
    summary = review_freeform_summary_text(
        _expect_non_empty_string(payload["summary"], context="pr_review.summary")
    )
    blocking_items = _expect_review_finding_list(
        payload, "blocking_items", context="pr_review.blocking_items", reviewer=reviewer
    )
    same_pr_followups = _expect_review_finding_list(
        payload, "same_pr_followups", context="pr_review.same_pr_followups", reviewer=reviewer
    )
    future_followups = _expect_review_finding_list(
        payload, "future_followups", context="pr_review.future_followups", reviewer=reviewer
    )
    dispositions = _expect_disposition_list(
        payload["prior_item_dispositions"],
        context="pr_review.prior_item_dispositions",
        reviewer=reviewer,
        allowed_same_status="same-pr",
        is_plan_review=False,
    )
    architecture_impact = (
        _parse_architecture_impact(payload["architecture_impact"], context="pr_review.architecture_impact")
        if "architecture_impact" in payload else None
    )
    structured_blocking_items = blocking_items
    followups = _dedupe_pr_review_items(
        structured_blocking_items,
        ApprovedFollowups(
            same_pr=same_pr_followups,
            future=future_followups,
        ),
    )
    return _finalize_parsed_review(
        state=state,
        summary=summary,
        blocking_items=structured_blocking_items,
        followups=followups,
        dispositions=dispositions,
        architecture_impact=architecture_impact,
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
        optional={"blocking_plan_issues", "same_plan_followups", "future_followups", "human_requirement_dispositions", "architecture_impact"},
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
    human_requirement_dispositions = _expect_human_requirement_dispositions(
        payload.get("human_requirement_dispositions", []),
        context="plan_review.human_requirement_dispositions",
    )
    architecture_impact = (
        _parse_architecture_impact(payload["architecture_impact"], context="plan_review.architecture_impact")
        if "architecture_impact" in payload else None
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
        human_requirement_dispositions=human_requirement_dispositions,
        architecture_impact=architecture_impact,
    )


def validate_structured_coder_followup(
    text: str, *, required_architecture_impact_contract: int = 0,
    delivered_risk_test_matrix: RiskTestMatrix | Mapping[str, object] | None = None,
    delivered_risk_test_matrix_identity: str | None = None,
    required_risk_test_matrix_contract: int = 0,
    authoritative_test_observations: Sequence[object] | None = None,
    delivered_risk_test_matrix_row_ids: Sequence[str] | None = None,
) -> StructuredCoderFollowup | None:
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
            "human_requirement_dispositions",
        },
        optional={
            "addressed_item_notes",
            "remaining_item_notes",
            "tests_run",
            "test_observations",
            "risk_test_matrix_evidence",
            "disputed_items",
            "dispute_evidence",
            "architecture_impact",
        },
    )
    human_requirements_payload = _expect_object(
        payload["human_requirements"],
        context="coder_followup.human_requirements",
    )
    human_requirement_dispositions = _expect_human_requirement_dispositions(
        payload["human_requirement_dispositions"],
        context="coder_followup.human_requirement_dispositions",
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
    test_observations = _expect_test_observations(
        payload.get("test_observations", []),
        context="coder_followup.test_observations",
    )
    risk_evidence = None
    if "risk_test_matrix_evidence" in payload:
        risk_evidence = parse_risk_test_matrix_evidence(
            payload["risk_test_matrix_evidence"],
            matrix=delivered_risk_test_matrix,
            expected_identity=delivered_risk_test_matrix_identity,
            authoritative_test_observations=authoritative_test_observations,
            expected_row_ids=delivered_risk_test_matrix_row_ids,
            context="coder_followup.risk_test_matrix_evidence",
        )
    elif required_risk_test_matrix_contract and delivered_risk_test_matrix is not None and parse_risk_test_matrix(delivered_risk_test_matrix).is_applicable:
        raise AgentLoopError("coder_followup must include risk_test_matrix_evidence for the delivered matrix.")
    architecture_impact = (
        _parse_architecture_impact(payload["architecture_impact"], context="coder_followup.architecture_impact")
        if "architecture_impact" in payload else None
    )
    if required_architecture_impact_contract == 1 and architecture_impact is None:
        raise AgentLoopError(
            "coder_followup must include architecture_impact for this fresh contract turn."
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
    state = _expect_state(payload["state"], context="coder_followup.state")
    if state == "approved" and any(
        item.disposition == "blocked" for item in human_requirement_dispositions
    ):
        raise AgentLoopError(
            "coder_followup.state must be `blocking` when a signed human requirement is blocked."
        )
    return StructuredCoderFollowup(
        schema_version=1,
        kind="coder_followup",
        state=state,
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
        human_requirement_dispositions=human_requirement_dispositions,
        addressed_item_notes=addressed_item_notes,
        remaining_item_notes=remaining_item_notes,
        tests_run=tests_run,
        disputed_items=disputed_items,
        dispute_evidence=dispute_evidence,
        test_observations=test_observations,
        architecture_impact=architecture_impact,
        risk_test_matrix_evidence=risk_evidence,
    )


def validate_structured_issue_implementation(
    text: str,
    *,
    required_architecture_impact_contract: int = 0,
    delivered_risk_test_matrix: RiskTestMatrix | Mapping[str, object] | None = None,
    delivered_risk_test_matrix_identity: str | None = None,
    required_risk_test_matrix_contract: int = 0,
    authoritative_test_observations: Sequence[object] | None = None,
    delivered_risk_test_matrix_row_ids: Sequence[str] | None = None,
) -> StructuredIssueImplementation | None:
    """Parse and validate the strict issue-implementation result envelope.

    Context-sensitive requirement coverage is applied by the orchestrator,
    because only the caller knows which signed labels were surfaced.  Basic
    payload shape is validated here so malformed responses remain repairable.
    """
    payload = _extract_structured_issue_implementation_payload(text)
    if payload is None:
        return None
    _require_supported_schema_version(payload)
    if payload.get("kind") != "issue_implementation":
        raise AgentLoopError(
            "Structured response kind mismatch: expected `issue_implementation`."
        )
    _expect_exact_keys(
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
        optional={"tests_run", "test_observations", "architecture_impact", "risk_test_matrix_evidence"},
    )
    state = _expect_non_empty_string(payload["state"], context="issue_implementation.state")
    if state != "blocking":
        raise AgentLoopError("issue_implementation.state must be `blocking`.")
    summary = _expect_non_empty_string(
        payload["summary"], context="issue_implementation.summary"
    )
    pr_value = payload["pr_number"]
    if pr_value is None:
        pr_number = None
    else:
        pr_number = _expect_int(pr_value, context="issue_implementation.pr_number")
        if pr_number <= 0:
            raise AgentLoopError(
                "issue_implementation.pr_number must be a positive integer or null."
            )
    human_payload = _expect_object(
        payload["human_requirements"], context="issue_implementation.human_requirements"
    )
    _expect_exact_keys(
        human_payload,
        context="issue_implementation.human_requirements",
        required={"addressed_ids", "checked_discussion_directly"},
    )
    dispositions = _expect_human_requirement_dispositions(
        payload["human_requirement_dispositions"],
        context="issue_implementation.human_requirement_dispositions",
    )
    tests_value = payload.get("tests_run")
    tests_run = (
        _expect_string_list(
            tests_value,
            context="issue_implementation.tests_run",
            item_context="issue_implementation.tests_run",
        )
        if tests_value is not None
        else None
    )
    test_observations = _expect_test_observations(
        payload.get("test_observations", []),
        context="issue_implementation.test_observations",
    )
    risk_evidence = None
    if "risk_test_matrix_evidence" in payload:
        risk_evidence = parse_risk_test_matrix_evidence(
            payload["risk_test_matrix_evidence"],
            matrix=delivered_risk_test_matrix,
            expected_identity=delivered_risk_test_matrix_identity,
            authoritative_test_observations=authoritative_test_observations,
            expected_row_ids=delivered_risk_test_matrix_row_ids,
            context="issue_implementation.risk_test_matrix_evidence",
        )
    elif required_risk_test_matrix_contract and delivered_risk_test_matrix is not None and parse_risk_test_matrix(delivered_risk_test_matrix).is_applicable:
        raise AgentLoopError("issue_implementation must include risk_test_matrix_evidence for the delivered matrix.")
    architecture_impact = (
        _parse_architecture_impact(
            payload["architecture_impact"], context="issue_implementation.architecture_impact"
        )
        if "architecture_impact" in payload
        else None
    )
    if required_architecture_impact_contract == 1 and architecture_impact is None:
        raise AgentLoopError(
            "issue_implementation must include architecture_impact for this fresh contract turn."
        )
    parsed = StructuredIssueImplementation(
        schema_version=1,
        kind="issue_implementation",
        state=state,
        summary=summary,
        pr_number=pr_number,
        human_requirements=StructuredHumanRequirementsPayload(
            addressed_ids=_expect_requirement_id_list(
                human_payload["addressed_ids"],
                context="issue_implementation.human_requirements.addressed_ids",
            ),
            checked_discussion_directly=_expect_bool(
                human_payload["checked_discussion_directly"],
                context="issue_implementation.human_requirements.checked_discussion_directly",
            ),
        ),
        human_requirement_dispositions=dispositions,
        tests_run=tests_run,
        test_observations=test_observations,
        architecture_impact=architecture_impact,
        risk_test_matrix_evidence=risk_evidence,
    )
    # Keep this semantic contradiction visible to callers as a dedicated error
    # with the typed payload attached.  The orchestration adapter first
    # re-applies caller-specific requirement coverage before treating it as a
    # terminal conflict.
    if parsed.pr_number is not None and any(
        item.disposition == "blocked" for item in parsed.human_requirement_dispositions
    ):
        raise IssueImplementationConflictError(parsed)
    return parsed


def validate_structured_task_result(
    text: str,
    *,
    required_architecture_impact_contract: int = 0,
) -> StructuredTaskResult | None:
    """Validate the versioned task terminal envelope, when present."""
    normalized, _status = normalize_response_file_structured_text(text)
    extracted = _extract_json_object_prefix(normalized)
    if extracted is None:
        return None
    payload, trailing = extracted
    payload = _consume_structured_footer_and_signature(
        payload=payload, trailing=trailing, state_re=STATE_RE,
        state_marker_name="AGENT_STATE", context_label="Structured task result",
    )
    _require_supported_schema_version(payload)
    if payload.get("kind") != "task_result":
        raise AgentLoopError("Structured response kind mismatch: expected `task_result`.")
    _expect_exact_keys(
        payload, context="task_result",
        required={"schema_version", "kind", "state", "outcome", "summary"},
        optional={"pr_number", "clarification", "architecture_impact"},
    )
    state = _expect_non_empty_string(payload["state"], context="task_result.state")
    if state != "blocking":
        raise AgentLoopError("task_result.state must be `blocking`.")
    outcome = _expect_non_empty_string(payload["outcome"], context="task_result.outcome")
    if outcome not in {"opened_pr", "blocking", "clarification"}:
        raise AgentLoopError("task_result.outcome must be opened_pr, blocking, or clarification.")
    pr_value = payload.get("pr_number")
    pr_number = None if pr_value is None else _expect_int(pr_value, context="task_result.pr_number")
    if outcome == "opened_pr" and (pr_number is None or pr_number <= 0):
        raise AgentLoopError("task_result.opened_pr requires a positive pr_number.")
    if outcome != "opened_pr" and pr_number is not None:
        raise AgentLoopError("task_result.pr_number is only valid for opened_pr.")
    questions = ()
    if "clarification" in payload:
        questions = _expect_string_list(payload["clarification"], context="task_result.clarification", item_context="task_result.clarification")
    if outcome == "clarification" and not questions:
        raise AgentLoopError("task_result.clarification requires at least one question.")
    if outcome != "clarification" and questions:
        raise AgentLoopError("task_result.clarification is only valid for clarification.")
    architecture_impact = (
        _parse_architecture_impact(payload["architecture_impact"], context="task_result.architecture_impact")
        if "architecture_impact" in payload else None
    )
    if required_architecture_impact_contract == 1 and architecture_impact is None:
        raise AgentLoopError(
            "task_result must include architecture_impact for this fresh contract turn."
        )
    return StructuredTaskResult(
        schema_version=1, kind="task_result", state=state, outcome=outcome,
        summary=_expect_non_empty_string(payload["summary"], context="task_result.summary"),
        pr_number=pr_number, clarification=questions,
        architecture_impact=architecture_impact,
    )


def validate_structured_plan_revision(
    text: str,
    *,
    required_architecture_impact_contract: int = 0,
    require_execution_strategy_contract: int = 0,
    require_risk_test_matrix_contract: int = 0,
) -> StructuredPlanRevision | None:
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
        },
    )
    execution_version, execution_recommendation = _parse_execution_contract_fields(
        payload,
        context="plan_revision",
        required=require_execution_strategy_contract == 1,
    )
    risk_version, risk_matrix, risk_changes = _parse_risk_test_matrix_contract_fields(
        payload,
        context="plan_revision",
        required=require_risk_test_matrix_contract == 1,
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
    architecture_impact = (
        _parse_architecture_impact(payload["architecture_impact"], context="plan_revision.architecture_impact")
        if "architecture_impact" in payload else None
    )
    if required_architecture_impact_contract == 1 and architecture_impact is None:
        raise AgentLoopError("plan_revision must include architecture_impact for this fresh contract turn.")
    additional_closing_issue_ids = _expect_optional_issue_id_list(
        payload,
        "additional_closing_issue_ids",
        context="plan_revision.additional_closing_issue_ids",
    )
    deferred_stages = _expect_deferred_stage_list(
        payload, "deferred_stages", context="plan_revision.deferred_stages"
    )
    human_requirement_dispositions = _expect_human_requirement_dispositions(
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
        typed_stages=_expect_typed_plan_stages(payload, context="plan_revision"),
        human_requirement_dispositions=human_requirement_dispositions,
        architecture_impact=architecture_impact,
        execution_strategy_contract_version=execution_version,
        execution_recommendation=execution_recommendation,
        risk_test_matrix_contract_version=risk_version,
        risk_test_matrix=risk_matrix,
        risk_test_matrix_changes=risk_changes,
    )


def validate_structured_plan_state(
    text: str,
    *,
    required_architecture_impact_contract: int = 0,
    require_execution_strategy_contract: int = 0,
    require_risk_test_matrix_contract: int = 0,
) -> StructuredPlanState | None:
    payload = _extract_structured_plan_state_payload(text)
    if payload is None:
        return None
    _require_supported_schema_version(payload)
    kind = payload.get("kind")
    if kind != "plan_state":
        raise AgentLoopError("Structured response kind mismatch: expected `plan_state`.")
    _expect_exact_keys(
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
        },
    )
    execution_version, execution_recommendation = _parse_execution_contract_fields(
        payload,
        context="plan_state",
        required=require_execution_strategy_contract == 1,
    )
    risk_version, risk_matrix, risk_changes = _parse_risk_test_matrix_contract_fields(
        payload,
        context="plan_state",
        required=require_risk_test_matrix_contract == 1,
    )
    state = _expect_state(payload["state"], context="plan_state.state")
    if state != "blocking":
        raise AgentLoopError("plan_state.state must be `blocking`.")
    architecture_impact = (
        _parse_architecture_impact(payload["architecture_impact"], context="plan_state.architecture_impact")
        if "architecture_impact" in payload else None
    )
    if required_architecture_impact_contract == 1 and architecture_impact is None:
        raise AgentLoopError("plan_state must include architecture_impact for this fresh contract turn.")
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
        additional_closing_issue_ids=_expect_optional_issue_id_list(
            payload,
            "additional_closing_issue_ids",
            context="plan_state.additional_closing_issue_ids",
        ),
        deferred_stages=_expect_deferred_stage_list(
            payload, "deferred_stages", context="plan_state.deferred_stages"
        ),
        typed_stages=_expect_typed_plan_stages(payload, context="plan_state"),
        human_requirement_dispositions=_expect_human_requirement_dispositions(
            payload.get("human_requirement_dispositions", []),
            context="plan_state.human_requirement_dispositions",
        ),
        architecture_impact=architecture_impact,
        execution_strategy_contract_version=execution_version,
        execution_recommendation=execution_recommendation,
        risk_test_matrix_contract_version=risk_version,
        risk_test_matrix=risk_matrix,
        risk_test_matrix_changes=risk_changes,
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
            and entry.startswith("Original finding:")
        ):
            continue
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
        parent_indent = indent

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


def _parse_discuss_research(
    value: object,
) -> tuple[str, tuple[DiscussSourcedFact, ...], str | None, tuple[str, ...]]:
    payload = _expect_object(value, context="discuss_review.research")
    _expect_exact_keys(
        payload,
        context="discuss_review.research",
        required={"status"},
        optional={"sourced_facts", "target", "questions"},
    )
    status = _expect_non_empty_string(payload["status"], context="discuss_review.research.status")
    if status not in DISCUSS_RESEARCH_STATUS_VALUES:
        rendered = ", ".join(sorted(DISCUSS_RESEARCH_STATUS_VALUES))
        raise AgentLoopError(f"discuss_review.research.status must be one of: {rendered}")
    facts_value = payload.get("sourced_facts", [])
    if not isinstance(facts_value, list):
        raise AgentLoopError("discuss_review.research.sourced_facts must be a JSON array.")
    sourced_facts: list[DiscussSourcedFact] = []
    for index, item in enumerate(facts_value):
        context = f"discuss_review.research.sourced_facts at index {index}"
        fact_payload = _expect_object(item, context=context)
        _expect_exact_keys(fact_payload, context=context, required={"fact", "source"})
        sourced_facts.append(
            DiscussSourcedFact(
                fact=_expect_non_empty_string(fact_payload["fact"], context=f"{context}.fact"),
                source=_expect_non_empty_string(
                    fact_payload["source"], context=f"{context}.source"
                ),
            )
        )
    if status == "sourced" and not sourced_facts:
        raise AgentLoopError(
            "discuss_review.research.sourced_facts must be non-empty when status is `sourced`."
        )
    if status != "sourced" and sourced_facts:
        raise AgentLoopError(
            "discuss_review.research.sourced_facts requires status `sourced`."
        )
    has_target = "target" in payload
    has_questions = "questions" in payload
    if has_target != has_questions:
        raise AgentLoopError(
            "discuss_review.research.target and research.questions must be supplied together."
        )
    target: str | None = None
    questions: tuple[str, ...] = ()
    if has_target:
        target = _expect_non_empty_string(payload["target"], context="discuss_review.research.target")
        if target not in DISCUSS_RESEARCH_TARGET_VALUES:
            rendered = ", ".join(sorted(DISCUSS_RESEARCH_TARGET_VALUES))
            raise AgentLoopError(f"discuss_review.research.target must be one of: {rendered}")
        questions = _expect_string_list(
            payload["questions"],
            context="discuss_review.research.questions",
            item_context="discuss_review.research.questions",
        )
        if not questions:
            raise AgentLoopError("discuss_review.research.questions must be non-empty when research intent is supplied.")
        if status == "not-needed":
            raise AgentLoopError("discuss_review.research intent requires an active research status.")
    return status, tuple(sourced_facts), target, questions


def _parse_discuss_evidence(
    value: object,
) -> tuple[tuple[DiscussEvidenceClaim, ...], tuple[DiscussEvidenceUpdate, ...]]:
    """Parse explicit evidence without trying to infer meaning from prose.

    This deliberately stays optional so persisted pre-#535 transcript responses
    can still be replayed.  Legacy ``research.sourced_facts`` are projected by
    the reconciliation layer as reported observations.
    """
    payload = _expect_object(value, context="discuss evidence")
    _expect_exact_keys(payload, context="discuss evidence", required={"claims", "updates"})
    claims_value = payload["claims"]
    updates_value = payload["updates"]
    if not isinstance(claims_value, list) or not isinstance(updates_value, list):
        raise AgentLoopError("discuss evidence.claims and evidence.updates must be JSON arrays.")
    claims: list[DiscussEvidenceClaim] = []
    for index, item in enumerate(claims_value):
        context = f"discuss evidence.claims at index {index}"
        claim = _expect_object(item, context=context)
        _expect_exact_keys(
            claim, context=context, required={"fact", "status"},
            optional={"source", "verification_basis"},
        )
        fact = _expect_non_empty_string(claim["fact"], context=f"{context}.fact")
        status = _expect_non_empty_string(claim["status"], context=f"{context}.status")
        if status not in DISCUSS_EVIDENCE_STATUS_VALUES:
            raise AgentLoopError("discuss evidence claim status must be verified, reported-but-unverified, or missing.")
        source = (_expect_non_empty_string(claim["source"], context=f"{context}.source")
                  if "source" in claim else None)
        basis = (_expect_non_empty_string(claim["verification_basis"], context=f"{context}.verification_basis")
                 if "verification_basis" in claim else None)
        if status == "missing" and (source is not None or basis is not None):
            raise AgentLoopError("missing evidence claims cannot carry a source or verification basis.")
        if status == "verified":
            if basis not in DISCUSS_VERIFICATION_BASES or source is None:
                raise AgentLoopError("verified evidence requires source and verification_basis external-source-inspected or checkout-inspected.")
            if basis == "checkout-inspected" and not re.fullmatch(r"[^\s:][^:]*:\d+", source):
                raise AgentLoopError("checkout-inspected verified evidence requires a repository-relative path:line source.")
        elif basis is not None:
            raise AgentLoopError("only verified evidence may carry verification_basis.")
        claims.append(DiscussEvidenceClaim(fact=fact, status=status, source=source, verification_basis=basis))
    updates: list[DiscussEvidenceUpdate] = []
    for index, item in enumerate(updates_value):
        context = f"discuss evidence.updates at index {index}"
        update = _expect_object(item, context=context)
        _expect_exact_keys(
            update, context=context, required={"action", "target_observation_id", "reason"},
            optional={"replacement_claim_index"},
        )
        action = _expect_non_empty_string(update["action"], context=f"{context}.action")
        if action not in DISCUSS_EVIDENCE_UPDATE_ACTIONS:
            raise AgentLoopError("discuss evidence update action must be retract or supersede.")
        replacement = update.get("replacement_claim_index")
        if replacement is not None and (not isinstance(replacement, int) or isinstance(replacement, bool) or replacement < 0 or replacement >= len(claims)):
            raise AgentLoopError("evidence replacement_claim_index must reference a claim in the same response.")
        updates.append(DiscussEvidenceUpdate(
            action=action,
            target_observation_id=_expect_non_empty_string(update["target_observation_id"], context=f"{context}.target_observation_id"),
            reason=_expect_non_empty_string(update["reason"], context=f"{context}.reason"),
            replacement_claim_index=replacement,
        ))
    return tuple(claims), tuple(updates)


def parse_structured_discuss_review(
    text: str, *, reviewer: str, round_number: int = 1, research_mode: str | None = None
) -> ParsedDiscussReview | None:
    payload = _extract_structured_discuss_review_payload(
        text, context_label="Structured discuss review"
    )
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
        optional={"split_proposals", "rebuttal", "analyzer_framing", "framing_note", "research", "evidence"},
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
    analyzer_framing = None
    if "analyzer_framing" in payload:
        analyzer_framing = _expect_non_empty_string(
            payload["analyzer_framing"], context="discuss_review.analyzer_framing"
        )
        if analyzer_framing not in DISCUSS_ANALYZER_FRAMING_VALUES:
            rendered = ", ".join(sorted(DISCUSS_ANALYZER_FRAMING_VALUES))
            raise AgentLoopError(f"discuss_review.analyzer_framing must be one of: {rendered}")
    framing_note = None
    if "framing_note" in payload:
        framing_note = _expect_non_empty_string(
            payload["framing_note"], context="discuss_review.framing_note"
        )
    if analyzer_framing == "misframed" and framing_note is None:
        raise AgentLoopError(
            "discuss_review.framing_note is required when analyzer_framing is `misframed`."
        )
    if framing_note is not None and analyzer_framing is None:
        raise AgentLoopError(
            "discuss_review.framing_note requires analyzer_framing to be set."
        )
    research_status: str | None = None
    sourced_facts: tuple[DiscussSourcedFact, ...] = ()
    research_target: str | None = None
    research_questions: tuple[str, ...] = ()
    if "research" in payload:
        research_status, sourced_facts, research_target, research_questions = _parse_discuss_research(payload["research"])
    evidence_claims, evidence_updates = ((), ())
    if "evidence" in payload:
        evidence_claims, evidence_updates = _parse_discuss_evidence(payload["evidence"])
    if research_mode == "required":
        if research_status is None:
            raise AgentLoopError(
                "discuss_review.research is required when the research policy is `required`."
            )
        if research_status == "not-needed":
            raise AgentLoopError(
                "discuss_review.research.status must not be `not-needed` when the "
                "research policy is `required`; use `sourced`, `unavailable`, or "
                "`inconclusive`."
            )
    return ParsedDiscussReview(
        outcome=outcome,
        rationale=rationale,
        split_proposals=split_proposals,
        reviewer=reviewer,
        rebuttal=rebuttal,
        analyzer_framing=analyzer_framing,
        framing_note=framing_note,
        research_status=research_status,
        sourced_facts=sourced_facts,
        research_target=research_target,
        research_questions=research_questions,
        evidence_claims=evidence_claims,
        evidence_updates=evidence_updates,
    )


def validate_structured_discuss_review(
    text: str, *, reviewer: str, round_number: int = 1, research_mode: str | None = None
) -> ParsedDiscussReview:
    parsed = parse_structured_discuss_review(
        text, reviewer=reviewer, round_number=round_number, research_mode=research_mode
    )
    if parsed is not None:
        return parsed
    raise AgentLoopError("Discuss review did not use the required structured format.")


DISCUSS_ANSWER_POSITION_VALUES = frozenset({"answer", "needs-human"})
DISCUSS_ANSWER_CONFIDENCE_VALUES = frozenset({"low", "medium", "high"})


def parse_structured_discuss_answer(
    text: str, *, reviewer: str, round_number: int = 1, research_mode: str | None = None
) -> ParsedDiscussAnswer | None:
    return _parse_structured_discuss_answer(
        text, reviewer=reviewer, round_number=round_number, research_mode=research_mode,
        allow_legacy_open_questions=False,
    )


def parse_legacy_structured_discuss_answer(
    text: str, *, reviewer: str, round_number: int = 1, research_mode: str | None = None
) -> ParsedDiscussAnswer | None:
    """Decode only persisted pre-#536 answer votes.

    New agent output must use ``unresolved_items``.  Legacy untyped questions
    are conservatively mapped to blockers for asserted answers, and to human
    decisions for escalations, so a resumed run cannot silently proceed.
    """
    return _parse_structured_discuss_answer(
        text, reviewer=reviewer, round_number=round_number, research_mode=research_mode,
        allow_legacy_open_questions=True,
    )


def _parse_discuss_unresolved_items(value: object, *, context: str) -> tuple[DiscussUnresolvedItem, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")
    items: list[DiscussUnresolvedItem] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        if not isinstance(item, dict):
            raise AgentLoopError(f"{item_context} must be an object.")
        _expect_exact_keys(item, context=item_context, required={"status", "text"})
        status = _expect_non_empty_string(item["status"], context=f"{item_context}.status")
        if status not in DISCUSS_UNRESOLVED_ITEM_STATUS_VALUES:
            raise AgentLoopError(
                f"{item_context}.status must be one of: blocker, human-decision, follow-up."
            )
        items.append(DiscussUnresolvedItem(
            status=status,
            text=_expect_non_empty_string(item["text"], context=f"{item_context}.text"),
        ))
    return tuple(items)


def _parse_structured_discuss_answer(
    text: str, *, reviewer: str, round_number: int, research_mode: str | None,
    allow_legacy_open_questions: bool,
) -> ParsedDiscussAnswer | None:
    payload = _extract_structured_discuss_review_payload(
        text, context_label="Structured discuss answer"
    )
    if payload is None:
        return None
    _require_supported_schema_version(payload)
    if payload.get("kind") != "discuss_answer":
        raise AgentLoopError("Structured response kind mismatch: expected `discuss_answer`.")
    item_key = "open_questions" if allow_legacy_open_questions else "unresolved_items"
    _expect_exact_keys(
        payload,
        context="discuss_answer",
        required={"schema_version", "kind", "position", "rationale", "confidence", item_key},
        optional={"answer", "rebuttal", "analyzer_framing", "framing_note", "research", "evidence"},
    )
    position = _expect_non_empty_string(payload["position"], context="discuss_answer.position")
    if position not in DISCUSS_ANSWER_POSITION_VALUES:
        raise AgentLoopError("discuss_answer.position must be `answer` or `needs-human`.")
    rationale = _expect_non_empty_string(payload["rationale"], context="discuss_answer.rationale")
    confidence = _expect_non_empty_string(payload["confidence"], context="discuss_answer.confidence")
    if confidence not in DISCUSS_ANSWER_CONFIDENCE_VALUES:
        raise AgentLoopError("discuss_answer.confidence must be one of: low, medium, high.")
    if allow_legacy_open_questions:
        legacy_questions = _expect_string_list(
            payload["open_questions"], context="discuss_answer.open_questions",
            item_context="discuss_answer.open_questions",
        )
        legacy_status = "human-decision" if position == "needs-human" else "blocker"
        unresolved_items = tuple(
            DiscussUnresolvedItem(status=legacy_status, text=question)
            for question in legacy_questions
        )
    else:
        unresolved_items = _parse_discuss_unresolved_items(
            payload["unresolved_items"], context="discuss_answer.unresolved_items"
        )
    answer = None
    if "answer" in payload:
        answer = _expect_non_empty_string(payload["answer"], context="discuss_answer.answer")
    if position == "answer" and answer is None:
        raise AgentLoopError("discuss_answer.answer is required when position is `answer`.")
    if position == "needs-human":
        if answer is not None:
            raise AgentLoopError("discuss_answer.answer must be omitted when position is `needs-human`.")
        if not any(item.status == "human-decision" for item in unresolved_items):
            raise AgentLoopError(
                "discuss_answer.unresolved_items must include a human-decision for `needs-human`."
            )
    rebuttal = None
    if "rebuttal" in payload:
        rebuttal = _expect_non_empty_string(payload["rebuttal"], context="discuss_answer.rebuttal")
    if round_number > 1 and rebuttal is None:
        raise AgentLoopError("discuss_answer.rebuttal is required for debate rounds.")
    analyzer_framing = payload.get("analyzer_framing")
    if analyzer_framing is not None:
        analyzer_framing = _expect_non_empty_string(analyzer_framing, context="discuss_answer.analyzer_framing")
        if analyzer_framing not in DISCUSS_ANALYZER_FRAMING_VALUES:
            raise AgentLoopError("discuss_answer.analyzer_framing must be `accurate` or `misframed`.")
    framing_note = payload.get("framing_note")
    if framing_note is not None:
        framing_note = _expect_non_empty_string(framing_note, context="discuss_answer.framing_note")
    if analyzer_framing == "misframed" and framing_note is None:
        raise AgentLoopError("discuss_answer.framing_note is required when analyzer_framing is `misframed`.")
    if framing_note is not None and analyzer_framing is None:
        raise AgentLoopError("discuss_answer.framing_note requires analyzer_framing to be set.")
    research_status, sourced_facts = (None, ())
    research_target: str | None = None
    research_questions: tuple[str, ...] = ()
    if "research" in payload:
        research_status, sourced_facts, research_target, research_questions = _parse_discuss_research(payload["research"])
    evidence_claims, evidence_updates = ((), ())
    if "evidence" in payload:
        evidence_claims, evidence_updates = _parse_discuss_evidence(payload["evidence"])
    if research_mode == "required" and (research_status is None or research_status == "not-needed"):
        raise AgentLoopError("discuss_answer.research with sourced, unavailable, or inconclusive status is required.")
    return ParsedDiscussAnswer(
        position=position, rationale=rationale, confidence=confidence,
        unresolved_items=unresolved_items, reviewer=reviewer, answer=answer,
        rebuttal=rebuttal, analyzer_framing=analyzer_framing, framing_note=framing_note,
        research_status=research_status, sourced_facts=sourced_facts,
        research_target=research_target, research_questions=research_questions,
        evidence_claims=evidence_claims, evidence_updates=evidence_updates,
    )


def validate_structured_discuss_answer(
    text: str, *, reviewer: str, round_number: int = 1, research_mode: str | None = None
) -> ParsedDiscussAnswer:
    parsed = parse_structured_discuss_answer(text, reviewer=reviewer, round_number=round_number, research_mode=research_mode)
    if parsed is None:
        raise AgentLoopError("Discuss answer did not use the required structured format.")
    return parsed


def validate_structured_discuss_evidence_reconciliation(
    text: str, *, observation_ids: Sequence[str], observation_statuses: dict[str, str],
) -> ParsedDiscussEvidenceReconciliation:
    payload = _extract_structured_discuss_review_payload(text, context_label="Evidence reconciliation")
    if payload is None:
        raise AgentLoopError("Evidence reconciliation did not use the required structured format.")
    _require_supported_schema_version(payload)
    _expect_exact_keys(payload, context="discuss_evidence_reconciliation", required={"schema_version", "kind", "groups"})
    if payload.get("kind") != "discuss_evidence_reconciliation" or not isinstance(payload["groups"], list):
        raise AgentLoopError("Expected discuss_evidence_reconciliation groups.")
    known = set(observation_ids)
    used: set[str] = set()
    groups: list[tuple[str, ...]] = []
    for index, value in enumerate(payload["groups"]):
        if not isinstance(value, list) or len(value) < 2 or not all(isinstance(item, str) and item for item in value):
            raise AgentLoopError(f"evidence reconciliation group {index} must contain at least two observation IDs.")
        group = tuple(value)
        if any(item not in known for item in group) or any(item in used for item in group):
            raise AgentLoopError("evidence reconciliation groups must use each supplied ID at most once.")
        if len({observation_statuses[item] for item in group}) != 1:
            raise AgentLoopError("evidence reconciliation can group only compatible active statuses.")
        used.update(group)
        groups.append(group)
    return ParsedDiscussEvidenceReconciliation(groups=tuple(groups))


DISCUSS_SEMANTIC_CLASSIFICATIONS = frozenset({
    "equivalent", "compatible_with_residual_decisions", "material_conflict"
})


def validate_structured_discuss_semantic_comparison(
    text: str, *, reviewers: Sequence[str]
) -> ParsedDiscussSemanticComparison:
    payload = _extract_structured_discuss_review_payload(text, context_label="Semantic comparison")
    if payload is None:
        raise AgentLoopError("Semantic comparison did not use the required structured format.")
    _require_supported_schema_version(payload)
    _expect_exact_keys(payload, context="discuss_semantic_comparison",
        required={"schema_version", "kind", "classification", "shared_recommendation", "remaining_decisions", "evidence"})
    if payload.get("kind") != "discuss_semantic_comparison":
        raise AgentLoopError("Structured response kind mismatch: expected `discuss_semantic_comparison`.")
    classification = _expect_non_empty_string(payload["classification"], context="discuss_semantic_comparison.classification")
    if classification not in DISCUSS_SEMANTIC_CLASSIFICATIONS:
        raise AgentLoopError("Unsupported semantic comparison classification.")
    shared = _expect_non_empty_string(payload["shared_recommendation"], context="discuss_semantic_comparison.shared_recommendation")
    decisions = _expect_string_list(payload["remaining_decisions"], context="discuss_semantic_comparison.remaining_decisions", item_context="discuss_semantic_comparison.remaining_decisions")
    if len(set(item.casefold() for item in decisions)) != len(decisions):
        raise AgentLoopError("Semantic comparison remaining_decisions must be deduplicated.")
    if classification == "equivalent" and decisions:
        raise AgentLoopError("Equivalent semantic comparisons cannot have remaining decisions.")
    if classification == "compatible_with_residual_decisions" and not decisions:
        raise AgentLoopError("Compatible semantic comparisons require remaining decisions.")
    evidence_payload = payload["evidence"]
    if not isinstance(evidence_payload, list) or not evidence_payload:
        raise AgentLoopError("Semantic comparison evidence must be a non-empty list.")
    evidence: list[DiscussSemanticEvidence] = []
    for index, item in enumerate(evidence_payload):
        item_payload = _expect_object(item, context=f"discuss_semantic_comparison.evidence[{index}]")
        _expect_exact_keys(item_payload, context=f"discuss_semantic_comparison.evidence[{index}]", required={"reviewer", "supports"})
        evidence.append(DiscussSemanticEvidence(
            reviewer=_expect_non_empty_string(item_payload["reviewer"], context="semantic evidence reviewer"),
            supports=_expect_non_empty_string(item_payload["supports"], context="semantic evidence supports"),
        ))
    expected = set(reviewers)
    actual = {item.reviewer for item in evidence}
    if actual != expected or len(evidence) != len(expected):
        raise AgentLoopError("Semantic comparison evidence must cover exactly every final-round reviewer.")
    return ParsedDiscussSemanticComparison(classification, shared, decisions, tuple(evidence))


def validate_structured_discuss_answer_confirmation(text: str, *, reviewer: str) -> ParsedDiscussAnswerConfirmation:
    payload = _extract_structured_discuss_review_payload(text, context_label="Answer confirmation")
    if payload is None:
        raise AgentLoopError("Answer confirmation did not use the required structured format.")
    _require_supported_schema_version(payload)
    _expect_exact_keys(payload, context="discuss_answer_confirmation",
        required={"schema_version", "kind", "decision", "rationale", "reviewer"}, optional={"answer"})
    if payload.get("kind") != "discuss_answer_confirmation":
        raise AgentLoopError("Structured response kind mismatch: expected `discuss_answer_confirmation`.")
    response_reviewer = _expect_non_empty_string(payload["reviewer"], context="discuss_answer_confirmation.reviewer")
    if response_reviewer != reviewer:
        raise AgentLoopError("Answer confirmation reviewer does not match the debater.")
    decision = _expect_non_empty_string(payload["decision"], context="discuss_answer_confirmation.decision")
    if decision not in {"confirm", "refine"}:
        raise AgentLoopError("Answer confirmation decision must be `confirm` or `refine`.")
    answer = payload.get("answer")
    if decision == "confirm" and answer is not None:
        raise AgentLoopError("Confirm responses must not include an answer.")
    if decision == "refine":
        answer = _expect_non_empty_string(answer, context="discuss_answer_confirmation.answer")
    return ParsedDiscussAnswerConfirmation(response_reviewer, decision,
        _expect_non_empty_string(payload["rationale"], context="discuss_answer_confirmation.rationale"), answer)


def _parse_discuss_agenda_disagreement(
    value: object, *, context: str
) -> DiscussAgendaDisagreement:
    payload = _expect_object(value, context=context)
    _expect_exact_keys(
        payload,
        context=context,
        required={"topic", "positions", "question_for_next_round"},
    )
    positions_payload = _expect_object(payload["positions"], context=f"{context}.positions")
    if not positions_payload:
        raise AgentLoopError(f"{context}.positions must not be empty.")
    positions = tuple(
        (
            _expect_non_empty_string(name, context=f"{context}.positions key"),
            _expect_non_empty_string(position, context=f"{context}.positions[{name!r}]"),
        )
        for name, position in positions_payload.items()
    )
    return DiscussAgendaDisagreement(
        topic=_expect_non_empty_string(payload["topic"], context=f"{context}.topic"),
        positions=positions,
        question_for_next_round=_expect_non_empty_string(
            payload["question_for_next_round"],
            context=f"{context}.question_for_next_round",
        ),
    )


def _bounded_discuss_synthesis_text(value: object, *, context: str) -> str:
    text = _expect_non_empty_string(value, context=context)
    if len(text.encode("utf-8")) > DISCUSS_SYNTHESIS_MAX_TEXT_BYTES:
        raise AgentLoopError(
            f"{context} exceeds {DISCUSS_SYNTHESIS_MAX_TEXT_BYTES} UTF-8 bytes."
        )
    return text


def _bounded_discuss_synthesis_list(
    value: object,
    *,
    context: str,
    item_context: str,
    maximum: int = DISCUSS_SYNTHESIS_MAX_ENTRIES,
) -> tuple[str, ...]:
    values = _expect_string_list(value, context=context, item_context=item_context)
    if len(values) > maximum:
        raise AgentLoopError(f"{context} may contain at most {maximum} item(s).")
    bounded = tuple(
        _bounded_discuss_synthesis_text(item, context=f"{item_context} at index {index}")
        for index, item in enumerate(values)
    )
    if len({item.casefold() for item in bounded}) != len(bounded):
        raise AgentLoopError(f"{context} must not contain duplicate entries.")
    return bounded


def _parse_discuss_synthesis_reference(
    value: object, *, context: str
) -> DiscussSynthesisResponseReference:
    payload = _expect_object(value, context=context)
    _expect_exact_keys(payload, context=context, required={"reviewer", "round"})
    reviewer = _bounded_discuss_synthesis_text(
        payload["reviewer"], context=f"{context}.reviewer"
    )
    round_number = _expect_int(payload["round"], context=f"{context}.round")
    if round_number < 1:
        raise AgentLoopError(f"{context}.round must be at least 1.")
    return DiscussSynthesisResponseReference(reviewer=reviewer, round=round_number)


def _parse_discuss_synthesis_references(
    value: object, *, context: str
) -> tuple[DiscussSynthesisResponseReference, ...]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")
    if not value:
        raise AgentLoopError(f"{context} must not be empty.")
    if len(value) > DISCUSS_SYNTHESIS_MAX_ENTRIES:
        raise AgentLoopError(
            f"{context} may contain at most {DISCUSS_SYNTHESIS_MAX_ENTRIES} item(s)."
        )
    references = tuple(
        _parse_discuss_synthesis_reference(item, context=f"{context}[{index}]")
        for index, item in enumerate(value)
    )
    keys = [(item.reviewer.casefold(), item.round) for item in references]
    if len(set(keys)) != len(keys):
        raise AgentLoopError(f"{context} must not contain duplicate references.")
    return references


def _parse_discuss_synthesis_consensus(
    value: object, *, context: str
) -> DiscussSynthesisConsensus:
    payload = _expect_object(value, context=context)
    _expect_exact_keys(payload, context=context, required={"text", "references"})
    return DiscussSynthesisConsensus(
        text=_bounded_discuss_synthesis_text(payload["text"], context=f"{context}.text"),
        references=_parse_discuss_synthesis_references(
            payload["references"], context=f"{context}.references"
        ),
    )


def _parse_discuss_synthesis_position(
    value: object, *, context: str
) -> DiscussSynthesisPosition:
    payload = _expect_object(value, context=context)
    _expect_exact_keys(payload, context=context, required={"reviewers", "position"})
    reviewers = _bounded_discuss_synthesis_list(
        payload["reviewers"],
        context=f"{context}.reviewers",
        item_context=f"{context}.reviewers",
        maximum=DISCUSS_SYNTHESIS_MAX_ENTRIES,
    )
    return DiscussSynthesisPosition(
        reviewers=reviewers,
        position=_bounded_discuss_synthesis_text(
            payload["position"], context=f"{context}.position"
        ),
    )


def _parse_discuss_synthesis_disagreement(
    value: object, *, context: str
) -> DiscussSynthesisDisagreement:
    payload = _expect_object(value, context=context)
    _expect_exact_keys(
        payload, context=context, required={"topic", "positions", "decision_needed"}
    )
    positions_value = payload["positions"]
    if not isinstance(positions_value, list):
        raise AgentLoopError(f"{context}.positions must be a JSON array.")
    if not positions_value:
        raise AgentLoopError(f"{context}.positions must not be empty.")
    if len(positions_value) > DISCUSS_SYNTHESIS_MAX_ENTRIES:
        raise AgentLoopError(
            f"{context}.positions may contain at most {DISCUSS_SYNTHESIS_MAX_ENTRIES} item(s)."
        )
    positions = tuple(
        _parse_discuss_synthesis_position(item, context=f"{context}.positions[{index}]")
        for index, item in enumerate(positions_value)
    )
    all_reviewers = [reviewer for item in positions for reviewer in item.reviewers]
    if len({reviewer.casefold() for reviewer in all_reviewers}) != len(all_reviewers):
        raise AgentLoopError(f"{context}.positions must not repeat a reviewer.")
    return DiscussSynthesisDisagreement(
        topic=_bounded_discuss_synthesis_text(payload["topic"], context=f"{context}.topic"),
        positions=positions,
        decision_needed=_bounded_discuss_synthesis_text(
            payload["decision_needed"], context=f"{context}.decision_needed"
        ),
    )


def _parse_discuss_synthesis_change(
    value: object, *, context: str
) -> DiscussSynthesisChange:
    payload = _expect_object(value, context=context)
    _expect_exact_keys(
        payload, context=context, required={"kind", "topic", "text", "references"}
    )
    kind = _bounded_discuss_synthesis_text(payload["kind"], context=f"{context}.kind")
    if kind not in DISCUSS_SYNTHESIS_CHANGE_KINDS:
        raise AgentLoopError(
            f"{context}.kind must be one of: "
            + ", ".join(sorted(DISCUSS_SYNTHESIS_CHANGE_KINDS))
        )
    return DiscussSynthesisChange(
        kind=kind,
        topic=_bounded_discuss_synthesis_text(payload["topic"], context=f"{context}.topic"),
        text=_bounded_discuss_synthesis_text(payload["text"], context=f"{context}.text"),
        references=_parse_discuss_synthesis_references(
            payload["references"], context=f"{context}.references"
        ),
    )


def _parse_round_synthesis_payload(
    payload: dict[str, object], *, context: str = "discuss_round_synthesis"
) -> ParsedDiscussRoundSynthesis:
    _require_supported_schema_version(payload)
    _expect_exact_keys(
        payload,
        context=context,
        required={
            "schema_version", "kind", "consensus", "disagreements", "changes",
            "missing_facts", "next_round_focus", "responding_reviewers",
        },
    )
    if payload.get("kind") != "discuss_round_synthesis":
        raise AgentLoopError(
            "Structured response kind mismatch: expected `discuss_round_synthesis`."
        )
    consensus = tuple(
        _parse_discuss_synthesis_consensus(item, context=f"{context}.consensus[{index}]")
        for index, item in enumerate(
            _bounded_synthesis_object_list(payload["consensus"], context=f"{context}.consensus")
        )
    )
    disagreements = tuple(
        _parse_discuss_synthesis_disagreement(
            item, context=f"{context}.disagreements[{index}]"
        )
        for index, item in enumerate(
            _bounded_synthesis_object_list(payload["disagreements"], context=f"{context}.disagreements")
        )
    )
    changes = tuple(
        _parse_discuss_synthesis_change(item, context=f"{context}.changes[{index}]")
        for index, item in enumerate(
            _bounded_synthesis_object_list(payload["changes"], context=f"{context}.changes")
        )
    )
    topics = [item.topic.casefold() for item in disagreements]
    if len(set(topics)) != len(topics):
        raise AgentLoopError(f"{context}.disagreements must not contain duplicate topics.")
    missing_facts = _bounded_discuss_synthesis_list(
        payload["missing_facts"], context=f"{context}.missing_facts", item_context=f"{context}.missing_facts"
    )
    next_focus = _bounded_discuss_synthesis_list(
        payload["next_round_focus"], context=f"{context}.next_round_focus", item_context=f"{context}.next_round_focus",
        maximum=DISCUSS_SYNTHESIS_MAX_NEXT_ROUND_FOCUS,
    )
    responding_reviewers = _bounded_discuss_synthesis_list(
        payload["responding_reviewers"], context=f"{context}.responding_reviewers", item_context=f"{context}.responding_reviewers"
    )
    return ParsedDiscussRoundSynthesis(
        consensus=consensus, disagreements=disagreements, changes=changes,
        missing_facts=missing_facts, next_round_focus=next_focus,
        responding_reviewers=responding_reviewers,
    )


def _bounded_synthesis_object_list(value: object, *, context: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise AgentLoopError(f"{context} must be a JSON array.")
    if len(value) > DISCUSS_SYNTHESIS_MAX_ENTRIES:
        raise AgentLoopError(
            f"{context} may contain at most {DISCUSS_SYNTHESIS_MAX_ENTRIES} item(s)."
        )
    return [_expect_object(item, context=f"{context}[{index}]") for index, item in enumerate(value)]


def _parse_final_synthesis_payload(
    payload: dict[str, object], *, context: str = "discuss_final_synthesis"
) -> ParsedDiscussFinalSynthesis:
    _require_supported_schema_version(payload)
    _expect_exact_keys(
        payload,
        context=context,
        required={
            "schema_version", "kind", "classification", "agreed_conclusions",
            "remaining_disagreements", "next_action",
        },
    )
    if payload.get("kind") != "discuss_final_synthesis":
        raise AgentLoopError(
            "Structured response kind mismatch: expected `discuss_final_synthesis`."
        )
    classification = _bounded_discuss_synthesis_text(
        payload["classification"], context=f"{context}.classification"
    )
    if classification not in DISCUSS_SYNTHESIS_CLASSIFICATIONS:
        raise AgentLoopError(f"{context}.classification is not supported.")
    agreed = tuple(
        _parse_discuss_synthesis_consensus(item, context=f"{context}.agreed_conclusions[{index}]")
        for index, item in enumerate(
            _bounded_synthesis_object_list(payload["agreed_conclusions"], context=f"{context}.agreed_conclusions")
        )
    )
    disagreements = tuple(
        _parse_discuss_synthesis_disagreement(
            item, context=f"{context}.remaining_disagreements[{index}]"
        )
        for index, item in enumerate(
            _bounded_synthesis_object_list(payload["remaining_disagreements"], context=f"{context}.remaining_disagreements")
        )
    )
    topics = [item.topic.casefold() for item in disagreements]
    if len(set(topics)) != len(topics):
        raise AgentLoopError(f"{context}.remaining_disagreements must not contain duplicate topics.")
    if classification == "consensus" and disagreements:
        raise AgentLoopError("A consensus final synthesis cannot contain remaining disagreements.")
    if classification == "consensus" and not agreed:
        raise AgentLoopError("A consensus final synthesis must include an agreed conclusion.")
    if classification == "near_consensus" and not disagreements:
        raise AgentLoopError(
            "A near-consensus final synthesis must include a remaining disagreement."
        )
    if classification == "material_deadlock" and not disagreements and not agreed:
        raise AgentLoopError("A material-deadlock final synthesis must describe the residual state.")
    return ParsedDiscussFinalSynthesis(
        classification=classification,
        agreed_conclusions=agreed,
        remaining_disagreements=disagreements,
        next_action=_bounded_discuss_synthesis_text(
            payload["next_action"], context=f"{context}.next_action"
        ),
    )


def _disc_synthesis_reference_payload(
    reference: DiscussSynthesisResponseReference,
) -> dict[str, object]:
    return {"reviewer": reference.reviewer, "round": reference.round}


def _disc_synthesis_consensus_payload(item: DiscussSynthesisConsensus) -> dict[str, object]:
    return {
        "text": item.text,
        "references": [_disc_synthesis_reference_payload(ref) for ref in item.references],
    }


def _disc_synthesis_disagreement_payload(item: DiscussSynthesisDisagreement) -> dict[str, object]:
    return {
        "topic": item.topic,
        "positions": [
            {"reviewers": list(position.reviewers), "position": position.position}
            for position in item.positions
        ],
        "decision_needed": item.decision_needed,
    }


def _disc_synthesis_change_payload(item: DiscussSynthesisChange) -> dict[str, object]:
    return {
        "kind": item.kind,
        "topic": item.topic,
        "text": item.text,
        "references": [_disc_synthesis_reference_payload(ref) for ref in item.references],
    }


def discuss_round_synthesis_payload(synthesis: ParsedDiscussRoundSynthesis) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "discuss_round_synthesis",
        "consensus": [_disc_synthesis_consensus_payload(item) for item in synthesis.consensus],
        "disagreements": [_disc_synthesis_disagreement_payload(item) for item in synthesis.disagreements],
        "changes": [_disc_synthesis_change_payload(item) for item in synthesis.changes],
        "missing_facts": list(synthesis.missing_facts),
        "next_round_focus": list(synthesis.next_round_focus),
        "responding_reviewers": list(synthesis.responding_reviewers),
    }


def discuss_final_synthesis_payload(synthesis: ParsedDiscussFinalSynthesis) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "discuss_final_synthesis",
        "classification": synthesis.classification,
        "agreed_conclusions": [_disc_synthesis_consensus_payload(item) for item in synthesis.agreed_conclusions],
        "remaining_disagreements": [_disc_synthesis_disagreement_payload(item) for item in synthesis.remaining_disagreements],
        "next_action": synthesis.next_action,
    }


def serialize_discuss_round_synthesis(synthesis: ParsedDiscussRoundSynthesis) -> str:
    payload = discuss_round_synthesis_payload(synthesis)
    _parse_round_synthesis_payload(payload)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(serialized.encode("utf-8")) > DISCUSS_SYNTHESIS_MAX_CANONICAL_BYTES:
        raise AgentLoopError("Canonical discuss round synthesis exceeds 16,000 UTF-8 bytes.")
    return serialized


def serialize_discuss_final_synthesis(synthesis: ParsedDiscussFinalSynthesis) -> str:
    payload = discuss_final_synthesis_payload(synthesis)
    _parse_final_synthesis_payload(payload)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(serialized.encode("utf-8")) > DISCUSS_SYNTHESIS_MAX_CANONICAL_BYTES:
        raise AgentLoopError("Canonical discuss final synthesis exceeds 16,000 UTF-8 bytes.")
    return serialized


def _parse_canonical_discuss_synthesis(text: str, *, kind: str) -> dict[str, object] | None:
    if not isinstance(text, str) or len(text.encode("utf-8")) > DISCUSS_SYNTHESIS_MAX_CANONICAL_BYTES:
        return None
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("kind") != kind:
        return None
    return value


def parse_structured_discuss_round_synthesis(text: str) -> ParsedDiscussRoundSynthesis | None:
    payload = _extract_structured_discuss_review_payload(
        text, context_label="Structured discuss round synthesis"
    )
    if payload is None:
        return None
    if payload.get("kind") != "discuss_round_synthesis":
        return None
    return _parse_round_synthesis_payload(payload)


def validate_structured_discuss_round_synthesis(text: str) -> ParsedDiscussRoundSynthesis:
    parsed = parse_structured_discuss_round_synthesis(text)
    if parsed is None:
        raise AgentLoopError("Discuss round synthesis did not use the required structured format.")
    return parsed


def parse_structured_discuss_final_synthesis(text: str) -> ParsedDiscussFinalSynthesis | None:
    payload = _extract_structured_discuss_review_payload(
        text, context_label="Structured discuss final synthesis"
    )
    if payload is None:
        return None
    if payload.get("kind") != "discuss_final_synthesis":
        return None
    return _parse_final_synthesis_payload(payload)


def validate_structured_discuss_final_synthesis(text: str) -> ParsedDiscussFinalSynthesis:
    parsed = parse_structured_discuss_final_synthesis(text)
    if parsed is None:
        raise AgentLoopError("Discuss final synthesis did not use the required structured format.")
    return parsed


def parse_canonical_discuss_round_synthesis(text: str) -> ParsedDiscussRoundSynthesis | None:
    payload = _parse_canonical_discuss_synthesis(text, kind="discuss_round_synthesis")
    if payload is None:
        return None
    return _parse_round_synthesis_payload(payload)


def parse_canonical_discuss_final_synthesis(text: str) -> ParsedDiscussFinalSynthesis | None:
    payload = _parse_canonical_discuss_synthesis(text, kind="discuss_final_synthesis")
    if payload is None:
        return None
    return _parse_final_synthesis_payload(payload)


def parse_structured_discuss_agenda(text: str) -> ParsedDiscussAgenda | None:
    payload = _extract_structured_discuss_agenda_payload(text)
    if payload is None:
        return None
    _require_supported_schema_version(payload)
    kind = payload.get("kind")
    if isinstance(kind, str) and kind != "discuss_agenda":
        raise AgentLoopError("Structured response kind mismatch: expected `discuss_agenda`.")
    _expect_exact_keys(
        payload,
        context="discuss_agenda",
        required={"schema_version", "kind", "consensus", "disagreements"},
        optional={
            "missing_facts", "research_required", "research_questions",
            "research_question_targets", "round_synthesis",
        },
    )
    consensus = _expect_string_list(
        payload["consensus"],
        context="discuss_agenda.consensus",
        item_context="discuss_agenda.consensus",
    )
    disagreements_value = payload["disagreements"]
    if not isinstance(disagreements_value, list):
        raise AgentLoopError("discuss_agenda.disagreements must be a JSON array.")
    disagreements = tuple(
        _parse_discuss_agenda_disagreement(
            item, context=f"discuss_agenda.disagreements at index {index}"
        )
        for index, item in enumerate(disagreements_value)
    )
    missing_facts = _expect_optional_string_list(
        payload,
        "missing_facts",
        context="discuss_agenda.missing_facts",
        item_context="discuss_agenda.missing_facts",
    )
    research_required = False
    if "research_required" in payload:
        research_required = _expect_bool(
            payload["research_required"], context="discuss_agenda.research_required"
        )
    research_questions = _expect_optional_string_list(
        payload,
        "research_questions",
        context="discuss_agenda.research_questions",
        item_context="discuss_agenda.research_questions",
    )
    research_question_targets = _expect_optional_string_list(
        payload,
        "research_question_targets",
        context="discuss_agenda.research_question_targets",
        item_context="discuss_agenda.research_question_targets",
    )
    for target in research_question_targets:
        if target not in DISCUSS_RESEARCH_TARGET_VALUES:
            rendered = ", ".join(sorted(DISCUSS_RESEARCH_TARGET_VALUES))
            raise AgentLoopError(f"discuss_agenda.research_question_targets must use only: {rendered}")
    if research_question_targets and len(research_question_targets) != len(research_questions):
        raise AgentLoopError(
            "discuss_agenda.research_question_targets must align one-to-one with research_questions."
        )
    if research_required and not research_questions:
        raise AgentLoopError(
            "discuss_agenda.research_questions must be non-empty when "
            "research_required is true."
        )
    if research_questions and not research_required:
        raise AgentLoopError(
            "discuss_agenda.research_questions requires research_required to be true."
        )
    round_synthesis = None
    if "round_synthesis" in payload:
        round_synthesis_payload = _expect_object(
            payload["round_synthesis"], context="discuss_agenda.round_synthesis"
        )
        round_synthesis = _parse_round_synthesis_payload(
            round_synthesis_payload, context="discuss_agenda.round_synthesis"
        )
    return ParsedDiscussAgenda(
        consensus=consensus,
        disagreements=disagreements,
        missing_facts=missing_facts,
        research_required=research_required,
        research_questions=research_questions,
        research_question_targets=research_question_targets,
        round_synthesis=round_synthesis,
    )


def validate_structured_discuss_agenda(text: str) -> ParsedDiscussAgenda:
    parsed = parse_structured_discuss_agenda(text)
    if parsed is not None:
        return parsed
    raise AgentLoopError("Discuss agenda did not use the required structured format.")
