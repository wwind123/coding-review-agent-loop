"""Approved-plan decomposition parsing and publishing helpers."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import re
import zlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .config import AgentLoopConfig
from .child_topology import (
    NeedsHumanDecision,
    merge_found_issues,
    parent_child_search_queries,
    preflight_flat_child_count,
)
from .errors import AgentLoopError
from .github import FoundIssue, create_issue, post_issue_comment, search_issues
from .runner import Runner
from .protocol_markers import TrustedBody, decompress_record_payload, sanitize_historical_text
from .protocol import (
    ArchitectureImpact,
    ChildStage,
    EXECUTION_AUTOMATION_CLASSES,
    EXECUTION_DISPOSITION_DIRECT,
    EXECUTION_DISPOSITION_HUMAN,
    EXECUTION_DISPOSITION_PLANNING,
    EXECUTION_DISPOSITION_VALUES,
    EXECUTION_STRATEGY_CONTRACT_VERSION,
    EXECUTION_TOPOLOGY_SOURCE,
    ExecutionAllocation,
    ExecutionChildStage,
    ExecutionCouplingConstraint,
    ExecutionDisposition,
    ExecutionScopeItem,
    ExecutionStrategyRecommendation,
    RiskTestMatrix,
    RiskTestMatrixRow,
    parse_architecture_impact,
    parse_risk_test_matrix,
    parse_execution_recommendation_payload,
    parse_signed_human_requirement_body,
    sanitize_architecture_impact,
    validate_direct_readiness,
)
from .issue_body_limits import (
    BODY_SAFETY_MARGIN,
    BoundedSection,
    bounded_text_present,
    fit_github_body,
    is_bounded_form,
    shortened_section,
)
from .round_transport import MAX_GITHUB_BODY_CHARS, MAX_PLAN_VALIDATION_DIAGNOSTIC_CHARS

AUTOMATION_CLASSES = set(EXECUTION_AUTOMATION_CLASSES)
DECOMPOSITION_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_DECOMPOSITION:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
PHASE_IMPLEMENTATION_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_PHASE_IMPLEMENTATION:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
TOPOLOGY_CHECKPOINT_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_TOPOLOGY_CHECKPOINT:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
EXECUTION_DECISION_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_EXECUTION_DECISION:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
PHASE_IDENTITY_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_PHASE_IDENTITY:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
LEGACY_SPLIT_IDENTITY_RE = re.compile(
    r"<!--\s*AGENT_SPLIT_CHILD:\s*parent=(?P<parent>\d+)\s+key=(?P<key>[0-9a-f]{64})\s*-->",
    re.I,
)
ONE_SHOT_IMPL_HANDOFF_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_ONE_SHOT_IMPL:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
ISSUE_NUMBER_RE = re.compile(r"/issues/(\d+)(?:\b|$)|#(\d+)\b")


@dataclass(frozen=True)
class PlanPhase:
    title: str
    scope: str
    non_goals: str
    dependency_notes: str
    rollout_risk: str
    validation: str
    parent_context: str
    automation: str
    depends_on: tuple[str, ...] = ()
    # Generation-1 reviewed fields.  The historical fields above remain the
    # legacy wire model; these fields are only populated for approved-plan-v1
    # phases and are included in their source-specific identity material.
    stage_id: str | None = None
    position: int | None = None
    deliverables: tuple[str, ...] = ()
    non_goals_items: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    depends_on_stage_ids: tuple[str, ...] = ()
    compatibility_constraints: tuple[str, ...] = ()
    covered_scope_item_ids: tuple[str, ...] = ()
    # Reviewed per-child execution disposition (#808).  ``None`` means the
    # approved plan predates the contract (legacy-ambiguous) and the routing
    # seam must fail closed to planning or a human decision.
    execution_disposition: str | None = None
    disposition_rationale: str | None = None
    unresolved_design_decisions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanDecomposition:
    phases: tuple[PlanPhase, ...]
    architecture_impact: ArchitectureImpact | None = None
    strategy: str | None = None
    topology_source: str = "model"
    execution_strategy_contract_version: int | None = None
    recommendation_digest: str | None = None
    scope_items: tuple[ExecutionScopeItem, ...] = ()
    coupling_constraints: tuple[ExecutionCouplingConstraint, ...] = ()
    retained_parent_work: ExecutionAllocation | None = None
    final_integration_work: ExecutionAllocation | None = None
    rationale: str | None = None
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordedPhase:
    title: str
    automation: str
    rollout_risk: str = "recorded"
    parent_context: str | None = None


@dataclass(frozen=True)
class CreatedPhaseIssue:
    phase: PlanPhase | RecordedPhase
    issue_url: str | None
    issue_number: int | None
    origin: str = "created"


@dataclass(frozen=True)
class RetainedParentScope:
    plan_subject: str
    plan_hash: str
    excerpt: str
    status: str = "required"
    deliverables: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    covered_scope_item_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecompositionMetadata:
    parent_issue: int
    plan_hash: str
    mode: str
    phase_count: int
    phase_titles: tuple[str, ...]
    automation: tuple[str, ...]
    children: tuple[tuple[str, str | None, int | None], ...]
    topology_source: str = "model"
    retained_parent_scope: RetainedParentScope | None = None
    final_integration_work: ExecutionAllocation | None = None
    strategy: str | None = None
    execution_strategy_contract_version: int | None = None
    recommendation_digest: str | None = None
    plan_subject: str | None = None
    stage_ids: tuple[str, ...] = ()
    phase_identities: tuple[str, ...] = ()
    # Per-phase reviewed dispositions, encoded only when at least one is set.
    dispositions: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class TopologyCheckpoint:
    parent_issue: int
    plan_hash: str
    mode: str
    topology_source: str
    phases: tuple[PlanPhase, ...]
    retained_parent_scope: RetainedParentScope | None = None
    architecture_identity: dict | None = None
    architecture_impact: dict | None = None
    architecture_contract_version: int | None = None
    strategy: str | None = None
    execution_strategy_contract_version: int | None = None
    recommendation_digest: str | None = None
    plan_subject: str | None = None


@dataclass(frozen=True)
class PhaseImplementationHandoffMetadata:
    parent_issue: int
    plan_hash: str
    mode: str
    phase_index: int
    phase_title: str
    automation: str
    child_issue_number: int
    child_issue_url: str | None
    strategy: str | None = None
    topology_source: str | None = None
    execution_strategy_contract_version: int | None = None
    recommendation_digest: str | None = None
    stage_id: str | None = None
    plan_subject: str | None = None
    inherited_matrix_row_ids: tuple[str, ...] = ()
    # Effective disposition recorded at dispatch (#808).  A handoff without
    # the field is a legacy handoff, which resume and PR validation treat as
    # ``direct-implementation`` with no override.
    execution_disposition: str | None = None
    override_digest: str | None = None


def handoff_effective_disposition(handoff: PhaseImplementationHandoffMetadata) -> str:
    """Return the disposition a recorded phase handoff binds (legacy = direct)."""
    return handoff.execution_disposition or EXECUTION_DISPOSITION_DIRECT


@dataclass(frozen=True)
class OneShotImplementationHandoffMetadata:
    parent_issue: int
    plan_hash: str
    plan_subject: str
    mode: str
    pr_number: int
    pr_head_sha: str | None
    strategy: str | None = None
    topology_source: str | None = None
    execution_strategy_contract_version: int | None = None
    recommendation_digest: str | None = None


def approved_plan_hash(approved_plan: str) -> str:
    return hashlib.sha256(approved_plan.strip().encode("utf-8")).hexdigest()[:16]


def _plan_subject_from_text(approved_plan: str) -> str:
    return hashlib.sha256(approved_plan.strip().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NormalizedExecutionTopology:
    """The single executable topology derived from a reviewed recommendation."""

    decomposition: PlanDecomposition
    retained_parent_scope: RetainedParentScope
    identity: dict[str, object]


@dataclass(frozen=True)
class ExecutionDecision:
    """A bounded approval-bound decision record.

    The identity fields are the only recovery authority.  The remaining fields
    are intentionally small diagnostics; the complete recommendation remains
    in the approved plan round and its existing bounded transport sidecars.
    """

    parent_issue: int
    plan_hash: str
    plan_subject: str
    execution_strategy_contract_version: int
    strategy: str
    topology_source: str
    recommendation_digest: str
    requested_policy: str
    current_action: str
    stage_ids: tuple[str, ...] = ()
    scope_item_ids: tuple[str, ...] = ()
    retained_parent_status: str = "none"
    final_integration_status: str = "none"
    architecture_identity: dict | None = None
    architecture_impact: dict | None = None

    def identity(self) -> dict[str, object]:
        return {
            "parent_issue": self.parent_issue,
            "plan_hash": self.plan_hash,
            "plan_subject": self.plan_subject,
            "execution_strategy_contract_version": self.execution_strategy_contract_version,
            "strategy": self.strategy,
            "topology_source": self.topology_source,
            "recommendation_digest": self.recommendation_digest,
        }

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            **self.identity(),
            "requested_policy": self.requested_policy,
            "current_action": self.current_action,
            "stage_ids": list(self.stage_ids),
            "scope_item_ids": list(self.scope_item_ids),
            "retained_parent_status": self.retained_parent_status,
            "final_integration_status": self.final_integration_status,
            "architecture_identity": self.architecture_identity,
            "architecture_impact": sanitize_architecture_impact(self.architecture_impact),
        }
        return payload


def normalize_execution_recommendation(
    recommendation: ExecutionStrategyRecommendation,
    *,
    approved_plan: str,
    plan_subject: str,
    execution_strategy_contract_version: int = EXECUTION_STRATEGY_CONTRACT_VERSION,
    topology_source: str = EXECUTION_TOPOLOGY_SOURCE,
) -> tuple[PlanDecomposition, RetainedParentScope]:
    """Normalize one validated v1 recommendation into the executable model.

    This function deliberately does not infer, normalize, or coerce reviewed
    values.  Stable IDs and exact automation classes are copied verbatim;
    dependency titles are only a display projection, while
    ``depends_on_stage_ids`` remains the authoritative reviewed dependency
    field.
    """
    if not isinstance(recommendation, ExecutionStrategyRecommendation):
        raise AgentLoopError("Execution topology normalization requires a validated recommendation.")
    if execution_strategy_contract_version != EXECUTION_STRATEGY_CONTRACT_VERSION:
        raise AgentLoopError("Unsupported execution strategy contract version.")
    if topology_source != EXECUTION_TOPOLOGY_SOURCE:
        raise AgentLoopError("Fresh execution topology must use source approved-plan-v1.")

    by_id = {stage.stage_id: stage for stage in recommendation.child_stages}
    phases: list[PlanPhase] = []
    for stage in recommendation.child_stages:
        # Validation in protocol.py has already established earlier-only
        # dependencies. Keep the stable IDs and expose titles only for the
        # legacy dependency-link renderer.
        dependency_titles = tuple(by_id[item].title for item in stage.depends_on_stage_ids)
        phases.append(
            PlanPhase(
                title=stage.title,
                scope=stage.summary,
                non_goals="\n".join(stage.non_goals),
                dependency_notes=stage.dependency_notes,
                rollout_risk=stage.rollout_risk,
                validation="\n".join(stage.acceptance_criteria),
                parent_context=sanitize_historical_text(approved_plan.strip()),
                automation=stage.automation,
                depends_on=dependency_titles,
                stage_id=stage.stage_id,
                position=stage.position,
                deliverables=stage.deliverables,
                non_goals_items=stage.non_goals,
                acceptance_criteria=stage.acceptance_criteria,
                depends_on_stage_ids=stage.depends_on_stage_ids,
                compatibility_constraints=stage.compatibility_constraints,
                covered_scope_item_ids=stage.covered_scope_item_ids,
                execution_disposition=(
                    stage.execution_disposition.disposition
                    if stage.execution_disposition is not None else None
                ),
                disposition_rationale=(
                    stage.execution_disposition.rationale
                    if stage.execution_disposition is not None else None
                ),
                unresolved_design_decisions=(
                    stage.execution_disposition.unresolved_design_decisions
                    if stage.execution_disposition is not None else ()
                ),
            )
        )

    retained = recommendation.retained_parent_work
    retained_scope = RetainedParentScope(
        plan_subject=sanitize_historical_text(plan_subject),
        plan_hash=approved_plan_hash(approved_plan),
        excerpt=sanitize_historical_text(approved_plan.strip()),
        status=retained.status,
        deliverables=retained.deliverables,
        acceptance_criteria=retained.acceptance_criteria,
        covered_scope_item_ids=retained.covered_scope_item_ids,
    )
    decomposition = PlanDecomposition(
        phases=tuple(phases),
        strategy=recommendation.strategy,
        topology_source=topology_source,
        execution_strategy_contract_version=execution_strategy_contract_version,
        recommendation_digest=str(recommendation.identity()["recommendation_sha256"]),
        scope_items=recommendation.scope_items,
        coupling_constraints=recommendation.coupling_constraints,
        retained_parent_work=recommendation.retained_parent_work,
        final_integration_work=recommendation.final_integration_work,
        rationale=recommendation.rationale,
        caveats=recommendation.caveats,
    )
    return decomposition, retained_scope


def validate_risk_matrix_ownership(
    matrix: RiskTestMatrix | dict[str, object] | None,
    recommendation: ExecutionStrategyRecommendation,
) -> None:
    """Ensure every matrix owner is a real owner in the approved topology."""
    if matrix is None:
        return
    parsed = matrix if isinstance(matrix, RiskTestMatrix) else parse_risk_test_matrix(matrix)
    stage_ids = {stage.stage_id for stage in recommendation.child_stages}
    allowed = {"one-shot", "retained-parent", "final-integration", *stage_ids}
    unknown = sorted({row.execution_owner for row in parsed.rows} - allowed)
    if unknown:
        raise AgentLoopError(
            "Risk matrix rows name execution owners absent from the approved topology: "
            + ", ".join(unknown)
        )
    if recommendation.strategy == "one-shot":
        invalid = sorted(
            {
                row.execution_owner
                for row in parsed.rows
                if row.execution_owner != "one-shot"
            }
        )
        if invalid:
            raise AgentLoopError(
                "One-shot recommendations may assign matrix rows only to `one-shot`: "
                + ", ".join(invalid)
            )
    if recommendation.strategy == "staged":
        invalid = sorted(
            {
                row.execution_owner
                for row in parsed.rows
                if row.execution_owner == "one-shot"
            }
        )
        if invalid:
            raise AgentLoopError(
                "Staged recommendations must assign matrix rows to a reviewed stage, "
                "retained-parent, or final-integration owner."
            )
    if "retained-parent" in {row.execution_owner for row in parsed.rows} and recommendation.retained_parent_work.status == "none":
        raise AgentLoopError(
            "Risk matrix rows cannot be owned by retained-parent when retained parent work is not approved."
        )
    if "final-integration" in {row.execution_owner for row in parsed.rows} and recommendation.final_integration_work.status == "none":
        raise AgentLoopError(
            "Risk matrix rows cannot be owned by final-integration when final integration work is not approved."
        )


def risk_matrix_row_ids_for_owner(
    matrix: RiskTestMatrix | dict[str, object] | None, owner: str
) -> tuple[str, ...]:
    """Return applicable parent obligations assigned to one execution owner."""
    if matrix is None:
        return ()
    parsed = matrix if isinstance(matrix, RiskTestMatrix) else parse_risk_test_matrix(matrix)
    return tuple(
        row.row_id
        for row in parsed.rows
        if row.execution_owner == owner and row.applicability in {"applicable", "required"}
    )


INHERITED_APPLICABILITY_RANK = {"not-applicable": 0, "applicable": 1, "required": 2}
# Scenario coverage fields: downstream evidence construction falls back to
# them, so a child keeps the parent text verbatim and may only extend it.
INHERITED_SCENARIO_FIELDS = (
    "entry_path_or_mode",
    "initial_state",
    "event",
    "expected_outcome",
)
INHERITED_REVIEWED_PROPOSAL_FIELDS = ("proposed_test_level", "proposed_test_location")
INHERITED_MATRIX_ROUTE_FORWARD = (
    "Route forward: revise the child plan so each inherited row keeps its row ID, "
    "never lowers applicability (not-applicable < applicable < required), copies every "
    "parent forbidden side effect exactly (case and whitespace included; additions are "
    "allowed), and keeps each scenario field (entry_path_or_mode, initial_state, event, "
    "expected_outcome) containing the parent text verbatim with refinements added after "
    "it; a parent scenario field already at the size bound admits only the identical value."
    " An already-approved child plan is revised by rerunning child planning; once it is "
    "bound to an implementation PR, post a signed child-plan-supersession record on the "
    "child issue first (see 'Re-planning an approved child plan' in the README)."
)
_INHERITED_DIAGNOSTIC_VALUE_CHARS = 96
# Entries are emitted whole under this budget so the storage sanitizer's
# bound never cuts one mid-way and the route-forward sentence always fits.
_INHERITED_DIAGNOSTIC_ENTRY_BUDGET = (
    MAX_PLAN_VALIDATION_DIAGNOSTIC_CHARS - len(INHERITED_MATRIX_ROUTE_FORWARD) - 512
)


@dataclass(frozen=True)
class InheritedMatrixBinding:
    """Binds a child plan-first cycle to the parent rows allocated to its stage."""

    parent_issue: int
    stage_id: str
    parent_matrix: dict[str, object]

    def inherited_rows(self) -> tuple[RiskTestMatrixRow, ...]:
        inherited = set(risk_matrix_row_ids_for_owner(self.parent_matrix, self.stage_id))
        if not inherited:
            return ()
        parsed = parse_risk_test_matrix(self.parent_matrix)
        return tuple(row for row in parsed.rows if row.row_id in inherited)


@dataclass(frozen=True)
class InheritedRowDifference:
    """One field-level difference between a parent row and the child's copy."""

    row_id: str
    field: str
    parent_value: str
    child_value: str
    reason: str


@dataclass(frozen=True)
class InheritedRowComparison:
    """Mechanically rejected weakenings and review-only admissible deltas."""

    weakenings: tuple[InheritedRowDifference, ...] = ()
    reviewed_deltas: tuple[InheritedRowDifference, ...] = ()


def compare_inherited_row(
    parent_row: RiskTestMatrixRow, child_row: RiskTestMatrixRow
) -> InheritedRowComparison:
    """Classify every field of one inherited row as preserved, reviewed, or weakened.

    Comparison uses the exact sanitized strings ``to_payload()`` produces: no
    case folding and no whitespace collapsing, because identifiers, paths,
    commands, and values can be case- or whitespace-sensitive.
    """
    parent = parent_row.to_payload()
    child = child_row.to_payload()
    row_id = str(parent["row_id"])
    weakenings: list[InheritedRowDifference] = []
    deltas: list[InheritedRowDifference] = []

    parent_applicability = str(parent["applicability"])
    child_applicability = str(child["applicability"])
    parent_rank = INHERITED_APPLICABILITY_RANK.get(parent_applicability)
    child_rank = INHERITED_APPLICABILITY_RANK.get(child_applicability)
    if parent_applicability != child_applicability:
        difference = InheritedRowDifference(
            row_id=row_id,
            field="applicability",
            parent_value=parent_applicability,
            child_value=child_applicability,
            reason="applicability raised",
        )
        if parent_rank is None or child_rank is None or child_rank < parent_rank:
            weakenings.append(dataclasses.replace(difference, reason="applicability weakened"))
        else:
            deltas.append(difference)

    parent_effects = [str(item) for item in parent["forbidden_side_effects"]]  # type: ignore[union-attr]
    child_effects = [str(item) for item in child["forbidden_side_effects"]]  # type: ignore[union-attr]
    for effect in parent_effects:
        if effect not in child_effects:
            weakenings.append(
                InheritedRowDifference(
                    row_id=row_id,
                    field="forbidden_side_effects",
                    parent_value=effect,
                    child_value="",
                    reason="forbidden side effect dropped",
                )
            )
    seen_added: set[str] = set()
    for effect in child_effects:
        if effect not in parent_effects and effect not in seen_added:
            seen_added.add(effect)
            deltas.append(
                InheritedRowDifference(
                    row_id=row_id,
                    field="forbidden_side_effects",
                    parent_value="",
                    child_value=effect,
                    reason="forbidden side effect added",
                )
            )

    for field in INHERITED_SCENARIO_FIELDS:
        parent_text = str(parent[field])
        child_text = str(child[field])
        if parent_text == child_text:
            continue
        difference = InheritedRowDifference(
            row_id=row_id,
            field=field,
            parent_value=parent_text,
            child_value=child_text,
            reason="coverage text extended",
        )
        if parent_text in child_text:
            deltas.append(difference)
        else:
            weakenings.append(dataclasses.replace(difference, reason="coverage text replaced"))

    for field in INHERITED_REVIEWED_PROPOSAL_FIELDS:
        if parent[field] != child[field]:
            deltas.append(
                InheritedRowDifference(
                    row_id=row_id,
                    field=field,
                    parent_value=str(parent[field]),
                    child_value=str(child[field]),
                    reason="proposed test changed",
                )
            )
    # `label` and `related_scope_item_ids` are free, and `execution_owner` is
    # excluded: a separately approved child owns its own scope and owner
    # namespaces.
    return InheritedRowComparison(tuple(weakenings), tuple(deltas))


def _inherited_row_pairs(
    parent_matrix: RiskTestMatrix | dict[str, object] | None,
    child_matrix: RiskTestMatrix | dict[str, object] | None,
    *,
    execution_owner: str,
) -> tuple[tuple[str, ...], tuple[tuple[RiskTestMatrixRow, RiskTestMatrixRow], ...]]:
    inherited = risk_matrix_row_ids_for_owner(parent_matrix, execution_owner)
    if not inherited:
        return (), ()
    if child_matrix is None:
        raise AgentLoopError(
            "Separately planned child omitted the approved parent risk matrix; "
            "repair the child plan and link inherited row IDs before implementation."
        )
    parent = parent_matrix if isinstance(parent_matrix, RiskTestMatrix) else parse_risk_test_matrix(parent_matrix)
    child = child_matrix if isinstance(child_matrix, RiskTestMatrix) else parse_risk_test_matrix(child_matrix)
    parent_by_id = {row.row_id: row for row in parent.rows}
    child_by_id = {row.row_id: row for row in child.rows}
    missing = sorted(set(inherited) - set(child_by_id))
    if missing:
        raise AgentLoopError(
            "Separately planned child is missing inherited parent matrix row IDs: "
            + ", ".join(missing)
        )
    return inherited, tuple((parent_by_id[row_id], child_by_id[row_id]) for row_id in inherited)


def _inherited_display_value(value: str) -> str:
    """Bounded, marker-safe, single-line rendering for diagnostics only."""
    text = sanitize_historical_text(value)
    text = "".join(character if ord(character) >= 32 else " " for character in text)
    if len(text) > _INHERITED_DIAGNOSTIC_VALUE_CHARS:
        text = text[: _INHERITED_DIAGNOSTIC_VALUE_CHARS - 1].rstrip() + "…"
    return f'"{text}"'


def _inherited_weakening_entry(difference: InheritedRowDifference) -> str:
    row_id = sanitize_historical_text(difference.row_id)
    if difference.field == "applicability":
        return (
            f"{row_id}: applicability weakened (parent={difference.parent_value}, "
            f"child={difference.child_value})"
        )
    if difference.field == "forbidden_side_effects":
        return (
            f"{row_id}: forbidden_side_effects dropped "
            f"{_inherited_display_value(difference.parent_value)} "
            "(no exactly equal child entry)"
        )
    return (
        f"{row_id}: {difference.field} replaced "
        f"(parent={_inherited_display_value(difference.parent_value)}, "
        f"child={_inherited_display_value(difference.child_value)}); the parent text must "
        "be kept verbatim and refinements added after it"
    )


def format_inherited_matrix_weakening_diagnostic(
    weakenings: Sequence[InheritedRowDifference],
) -> str:
    """Render weakenings whole within the plan-validation diagnostic bound."""
    header = "Separately planned child weakened inherited parent matrix rows:"
    lines = [header]
    used = len(header)
    emitted = 0
    for difference in weakenings:
        entry = "- " + _inherited_weakening_entry(difference)
        if used + 1 + len(entry) > _INHERITED_DIAGNOSTIC_ENTRY_BUDGET:
            break
        lines.append(entry)
        used += 1 + len(entry)
        emitted += 1
    remaining = list(weakenings[emitted:])
    if remaining:
        rows = len({difference.row_id for difference in remaining})
        lines.append(f"- and {len(remaining)} more weakened fields in {rows} rows")
    lines.append(INHERITED_MATRIX_ROUTE_FORWARD)
    return "\n".join(lines)


def inherited_matrix_reviewed_deltas(
    parent_matrix: RiskTestMatrix | dict[str, object] | None,
    child_matrix: RiskTestMatrix | dict[str, object] | None,
    *,
    execution_owner: str,
) -> tuple[InheritedRowDifference, ...]:
    """Admissible inherited-row differences that reviewers must judge.

    Deterministic in the two matrices, so a resumed review round recomputes
    the same deltas and nothing about them is persisted.
    """
    _, pairs = _inherited_row_pairs(
        parent_matrix, child_matrix, execution_owner=execution_owner
    )
    deltas: list[InheritedRowDifference] = []
    for parent_row, child_row in pairs:
        deltas.extend(compare_inherited_row(parent_row, child_row).reviewed_deltas)
    return tuple(deltas)


def validate_separately_planned_child_matrix(
    parent_matrix: RiskTestMatrix | dict[str, object] | None,
    child_matrix: RiskTestMatrix | dict[str, object] | None,
    *,
    execution_owner: str,
) -> tuple[str, ...]:
    """Bind a separately approved child plan to its inherited parent rows.

    A child plan may add child-local rows, but it cannot omit or weaken rows
    allocated to this stage. Weakening is judged per field class, on the exact
    sanitized strings with no case or whitespace normalization:

    - ``applicability`` is ordered (``not-applicable`` < ``applicable`` <
      ``required``) and may only stay or rise;
    - every parent ``forbidden_side_effects`` entry must survive as an exactly
      equal child entry; reordering and additions are accepted;
    - ``entry_path_or_mode``, ``initial_state``, ``event``, and
      ``expected_outcome`` must contain the parent text verbatim and may only
      extend it;
    - ``proposed_test_level`` and ``proposed_test_location`` may change but are
      always surfaced through :func:`inherited_matrix_reviewed_deltas`;
    - ``label`` and ``related_scope_item_ids`` are free, and
      ``execution_owner`` is excluded.

    Raises, strengthenings, extensions, and test placement changes are
    reviewed deltas and never raise here. The returned IDs are later used as
    the provenance-bound discharge scope for the child turn.
    """
    inherited, pairs = _inherited_row_pairs(
        parent_matrix, child_matrix, execution_owner=execution_owner
    )
    weakenings: list[InheritedRowDifference] = []
    for parent_row, child_row in pairs:
        weakenings.extend(compare_inherited_row(parent_row, child_row).weakenings)
    if weakenings:
        raise AgentLoopError(format_inherited_matrix_weakening_diagnostic(weakenings))
    return inherited


# Descriptive alias used by callers that want to emphasize the reviewed
# recommendation rather than the legacy typed-stage adapter.
adapt_execution_strategy_recommendation = normalize_execution_recommendation


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^\s*```(?:json)?\s*", "", stripped, count=1, flags=re.I)
        stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise AgentLoopError(f"Invalid plan decomposition JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise AgentLoopError("Invalid plan decomposition JSON: top-level payload must be an object.")
    trailing = stripped[end:].strip()
    if trailing and not trailing.startswith("<!--"):
        raise AgentLoopError("Invalid plan decomposition JSON: unexpected text after JSON payload.")
    return payload


def _required_text(payload: dict[str, object], key: str, *, phase_title: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgentLoopError(f"Invalid plan decomposition: phase {phase_title!r} is missing `{key}`.")
    return value.strip()


def parse_plan_decomposition(
    text: str, *, required_architecture_impact_contract: int = 0
) -> PlanDecomposition:
    payload = _extract_json_object(text)
    if payload.get("kind") not in (None, "plan_decomposition"):
        raise AgentLoopError("Invalid plan decomposition: `kind` must be `plan_decomposition`.")
    impact = (
        parse_architecture_impact(payload["architecture_impact"], context="plan_decomposition.architecture_impact")
        if "architecture_impact" in payload else None
    )
    if required_architecture_impact_contract == 1 and impact is None:
        raise AgentLoopError(
            "plan_decomposition must include architecture_impact for this fresh contract turn."
        )
    phases_payload = payload.get("phases")
    if not isinstance(phases_payload, list) or not phases_payload:
        raise AgentLoopError("Invalid plan decomposition: `phases` must be a non-empty list.")
    title_keys = [
        " ".join(phase_payload["title"].lower().split())
        for phase_payload in phases_payload
        if isinstance(phase_payload, dict)
        and isinstance(phase_payload.get("title"), str)
        and phase_payload["title"].strip()
    ]
    all_title_keys = set(title_keys)
    phases: list[PlanPhase] = []
    seen_titles: set[str] = set()
    for index, phase_payload in enumerate(phases_payload, start=1):
        if not isinstance(phase_payload, dict):
            raise AgentLoopError(f"Invalid plan decomposition: phase {index} must be an object.")
        title = _required_text(phase_payload, "title", phase_title=f"#{index}")
        title_key = " ".join(title.lower().split())
        if title_key in seen_titles:
            raise AgentLoopError(f"Invalid plan decomposition: duplicate phase title {title!r}.")
        seen_titles.add(title_key)
        automation = _required_text(phase_payload, "automation", phase_title=title)
        if automation not in AUTOMATION_CLASSES:
            raise AgentLoopError(
                f"Invalid plan decomposition: phase {title!r} has invalid automation {automation!r}."
            )
        depends_on_payload = phase_payload.get("depends_on", [])
        if depends_on_payload is None:
            depends_on_payload = []
        if not isinstance(depends_on_payload, list) or not all(
            isinstance(value, str) and value.strip() for value in depends_on_payload
        ):
            raise AgentLoopError(
                f"Invalid plan decomposition: phase {title!r} `depends_on` must be a list of titles."
            )
        for dependency in depends_on_payload:
            dependency_key = " ".join(dependency.lower().split())
            if dependency_key == title_key:
                raise AgentLoopError(
                    f"Invalid plan decomposition: phase {title!r} cannot depend on itself."
                )
            if dependency_key not in all_title_keys:
                raise AgentLoopError(
                    f"Invalid plan decomposition: phase {title!r} depends on unknown phase {dependency!r}."
                )
            if dependency_key not in seen_titles:
                raise AgentLoopError(
                    f"Invalid plan decomposition: phase {title!r} depends on {dependency!r}, "
                    "but dependencies must reference an earlier phase."
                )
        phases.append(
            PlanPhase(
                title=title,
                scope=_required_text(phase_payload, "scope", phase_title=title),
                non_goals=_required_text(phase_payload, "non_goals", phase_title=title),
                dependency_notes=_required_text(phase_payload, "dependency_notes", phase_title=title),
                rollout_risk=_required_text(phase_payload, "rollout_risk", phase_title=title),
                validation=_required_text(phase_payload, "validation", phase_title=title),
                parent_context=_required_text(phase_payload, "parent_context", phase_title=title),
                automation=automation,
                depends_on=tuple(value.strip() for value in depends_on_payload),
            )
        )
    return PlanDecomposition(phases=tuple(phases), architecture_impact=impact)


def _issue_number_from_url(issue_url: str | None) -> int | None:
    if not issue_url:
        return None
    match = ISSUE_NUMBER_RE.search(issue_url)
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def _phase_issue_title(parent_issue: int, index: int, phase: PlanPhase) -> str:
    prefix = "[Human] " if phase.automation in {"human-action", "manual-close"} else ""
    return f"{prefix}Phase {index}: {phase.title} (from #{parent_issue})"[:120]


def _phase_payload(phase: PlanPhase) -> dict[str, object]:
    """Historical phase serializer; keep this key set byte-stable."""
    return {
        "title": phase.title,
        "scope": phase.scope,
        "non_goals": phase.non_goals,
        "dependency_notes": phase.dependency_notes,
        "rollout_risk": phase.rollout_risk,
        "validation": phase.validation,
        "parent_context": phase.parent_context,
        "automation": phase.automation,
        "depends_on": list(phase.depends_on or ()),
    }


def _fresh_phase_payload(phase: PlanPhase) -> dict[str, object]:
    """The enriched source/version-specific serializer for fresh phases."""
    if not phase.stage_id or phase.position is None:
        raise AgentLoopError("Fresh phase identity requires a stable stage ID and ordinal.")
    return {
        "stage_id": phase.stage_id,
        "position": phase.position,
        "title": phase.title,
        "summary": phase.scope,
        "deliverables": list(phase.deliverables),
        "non_goals": list(phase.non_goals_items),
        "acceptance_criteria": list(phase.acceptance_criteria),
        "depends_on_stage_ids": list(phase.depends_on_stage_ids),
        "dependency_notes": phase.dependency_notes,
        "automation": phase.automation,
        "rollout_risk": phase.rollout_risk,
        "compatibility_constraints": list(phase.compatibility_constraints),
        "covered_scope_item_ids": list(phase.covered_scope_item_ids),
        "parent_context": phase.parent_context,
        # Emitted only when declared so identities and checkpoints of plans
        # approved before #808 remain byte-identical.
        **(
            {
                "execution_disposition": {
                    "disposition": phase.execution_disposition,
                    "rationale": phase.disposition_rationale or "",
                    "unresolved_design_decisions": list(phase.unresolved_design_decisions),
                }
            }
            if phase.execution_disposition is not None else {}
        ),
    }


def phase_direct_readiness_problems(phase: PlanPhase) -> tuple[str, ...]:
    """Judge direct readiness on the persisted reviewed phase fields only."""
    return validate_direct_readiness(
        non_goals=phase.non_goals_items,
        compatibility_constraints=phase.compatibility_constraints,
        dependency_notes=phase.dependency_notes,
        rationale=phase.disposition_rationale,
        unresolved_design_decisions=phase.unresolved_design_decisions,
    )


def phase_identity(
    *,
    parent_issue: int,
    plan_hash: str,
    topology_source: str,
    phase_index: int,
    phase: PlanPhase | RecordedPhase,
    stage_id: str | None = None,
    execution_strategy_contract_version: int | None = None,
) -> str:
    """Return a stable identity independent of the display title truncation."""
    fresh = topology_source == EXECUTION_TOPOLOGY_SOURCE
    if fresh:
        if not isinstance(phase, PlanPhase):
            raise AgentLoopError("Fresh phase identity requires a reviewed phase model.")
        stable_stage_id = stage_id or phase.stage_id
        if not isinstance(stable_stage_id, str) or not stable_stage_id:
            raise AgentLoopError("Fresh phase identity requires a stable string stage ID.")
        if execution_strategy_contract_version not in (None, EXECUTION_STRATEGY_CONTRACT_VERSION):
            raise AgentLoopError("Fresh phase identity has an invalid contract version.")
        material = {
            "parent_issue": parent_issue,
            "plan_hash": plan_hash,
            "source": topology_source,
            "contract_version": execution_strategy_contract_version or EXECUTION_STRATEGY_CONTRACT_VERSION,
            "phase_index": phase_index,
            "stage_id": stable_stage_id,
            "phase": _fresh_phase_payload(phase),
        }
    else:
        # Do not add fresh fields or a contract discriminator to this branch:
        # old child issues must recompute their historical digest exactly.
        legacy_phase = phase
        if isinstance(phase, RecordedPhase):
            # Summary-only recovery records predate PlanPhase's detailed
            # fields.  Materialize the same empty historical values explicitly
            # before using the byte-stable PlanPhase serializer.
            legacy_phase = PlanPhase(
                title=phase.title,
                scope="",
                non_goals="",
                dependency_notes="",
                rollout_risk=phase.rollout_risk,
                validation="",
                parent_context=phase.parent_context,
                automation=phase.automation,
                depends_on=(),
            )
        material = {
            "parent_issue": parent_issue,
            "plan_hash": plan_hash,
            "source": topology_source,
            "stage_id": phase_index,
            "phase": _phase_payload(legacy_phase),
        }
    encoded = json.dumps(material, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phase_identity_marker(
    identity: str,
    *,
    parent_issue: int,
    plan_hash: str,
    source: str,
    index: int,
    stage_id: str | None = None,
    strategy: str | None = None,
    recommendation_digest: str | None = None,
    execution_strategy_contract_version: int | None = None,
    inherited_matrix_row_ids: Sequence[str] = (),
) -> str:
    payload: dict[str, object] = {
        "identity": identity,
        "parent_issue": parent_issue,
        "plan_hash": plan_hash,
        "source": source,
    }
    if source == EXECUTION_TOPOLOGY_SOURCE:
        if not stage_id:
            raise AgentLoopError("Fresh phase marker requires a stable stage ID.")
        payload.update(
            {
                "phase_index": index,
                "stage_id": stage_id,
                "strategy": strategy,
                "recommendation_digest": recommendation_digest,
                "execution_strategy_contract_version": (
                    execution_strategy_contract_version or EXECUTION_STRATEGY_CONTRACT_VERSION
                ),
            }
        )
        if inherited_matrix_row_ids:
            payload["inherited_matrix_row_ids"] = list(inherited_matrix_row_ids)
    else:
        # Historical marker shape is intentionally retained for old child
        # issues and their exact legacy lookup rules.
        payload["stage_id"] = index
    return f"<!-- AGENT_PLAN_PHASE_IDENTITY: {_encode_json_payload(payload)} -->"


def adapt_typed_child_stages(
    stages: Sequence[object],
    *,
    approved_plan: str,
    plan_subject: str,
) -> tuple[PlanDecomposition, RetainedParentScope]:
    """Adapt the two-field typed remainder into the decomposition contract."""
    if any(not isinstance(stage, ChildStage) for stage in stages):
        raise AgentLoopError(
            "Generation-1 execution child stages cannot reach the legacy typed-stage adapter; "
            "Stage 2 must consume the reviewed recommendation directly."
        )
    excerpt = sanitize_historical_text(approved_plan.strip())
    retained = RetainedParentScope(
        plan_subject=sanitize_historical_text(plan_subject),
        plan_hash=approved_plan_hash(approved_plan),
        excerpt=excerpt,
    )
    phases = tuple(
        PlanPhase(
            title=str(stage.title),
            scope=str(stage.summary),
            non_goals=(
                "No stage-specific non-goals were declared; refer to the neutralized "
                "parent constraints above."
            ),
            dependency_notes=(
                "No stage-specific dependency notes were declared; this typed stage "
                "has no explicit inter-stage dependency."
            ),
            rollout_risk=(
                "No stage-specific rollout risk was declared; follow the neutralized "
                "parent constraints above."
            ),
            validation=(
                "No stage-specific validation state was declared; follow the parent "
                "plan's validation requirements."
            ),
            parent_context=excerpt,
            automation="agent-pr",
            depends_on=(),
        )
        for stage in stages
    )
    return PlanDecomposition(phases=phases), retained


def _checkpoint_payload(checkpoint: TopologyCheckpoint) -> dict[str, object]:
    contexts = tuple(phase.parent_context for phase in checkpoint.phases)
    shared_context = (
        contexts[0]
        if contexts and all(context == contexts[0] for context in contexts)
        else None
    )
    phase_payloads: list[dict[str, object]] = []
    for phase in checkpoint.phases:
        payload = (
            _fresh_phase_payload(phase)
            if checkpoint.topology_source == EXECUTION_TOPOLOGY_SOURCE
            else _phase_payload(phase)
        )
        if shared_context is not None:
            payload.pop("parent_context", None)
        phase_payloads.append(payload)
    retained_payload = None
    if checkpoint.retained_parent_scope is not None:
        retained_payload = {
            "plan_subject": checkpoint.retained_parent_scope.plan_subject,
            "plan_hash": checkpoint.retained_parent_scope.plan_hash,
            "excerpt": checkpoint.retained_parent_scope.excerpt,
        }
        if checkpoint.topology_source == EXECUTION_TOPOLOGY_SOURCE:
            retained_payload.update(
                {
                    "status": checkpoint.retained_parent_scope.status,
                    "deliverables": list(checkpoint.retained_parent_scope.deliverables),
                    "acceptance_criteria": list(checkpoint.retained_parent_scope.acceptance_criteria),
                    "covered_scope_item_ids": list(
                        checkpoint.retained_parent_scope.covered_scope_item_ids
                    ),
                }
            )
        # Typed phases and the retained parent scope deliberately share this
        # excerpt. Keep it in one place in the checkpoint marker.
        if (
            shared_context is not None
            and checkpoint.retained_parent_scope.excerpt == shared_context
        ):
            retained_payload.pop("excerpt")
    # Checkpoint comments are durable host-visible transport. Impact prose is
    # agent-supplied and must be marker-safe even when a caller constructed a
    # TopologyCheckpoint directly rather than going through the orchestrator.
    architecture_impact = sanitize_architecture_impact(checkpoint.architecture_impact)
    payload: dict[str, object] = {
        "parent_issue": checkpoint.parent_issue,
        "plan_hash": checkpoint.plan_hash,
        "mode": checkpoint.mode,
        "topology_source": checkpoint.topology_source,
        "shared_parent_context": shared_context,
        "phases": phase_payloads,
        "retained_parent_scope": retained_payload,
        "architecture_identity": checkpoint.architecture_identity,
        "architecture_impact": architecture_impact,
        "architecture_contract_version": checkpoint.architecture_contract_version,
    }
    if checkpoint.topology_source == EXECUTION_TOPOLOGY_SOURCE:
        payload.update(
            {
                "strategy": checkpoint.strategy,
                "execution_strategy_contract_version": (
                    checkpoint.execution_strategy_contract_version
                    or EXECUTION_STRATEGY_CONTRACT_VERSION
                ),
                "recommendation_digest": checkpoint.recommendation_digest,
                "plan_subject": checkpoint.plan_subject,
            }
        )
    return payload


def _phase_from_payload(
    payload: object,
    *,
    shared_parent_context: str | None = None,
    fresh: bool = False,
) -> PlanPhase:
    if not isinstance(payload, dict):
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
    if fresh:
        required = (
            "stage_id", "position", "title", "summary", "deliverables", "non_goals",
            "acceptance_criteria", "depends_on_stage_ids", "dependency_notes",
            "automation", "rollout_risk", "compatibility_constraints", "covered_scope_item_ids",
        )
        if any(key not in payload for key in required):
            raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
        values = {
            "title": payload["title"],
            "scope": payload["summary"],
            "non_goals": "\n".join(payload["non_goals"])
            if isinstance(payload["non_goals"], list)
            else None,
            "dependency_notes": payload["dependency_notes"],
            "rollout_risk": payload["rollout_risk"],
            "validation": "\n".join(payload["acceptance_criteria"])
            if isinstance(payload["acceptance_criteria"], list)
            else None,
            "automation": payload["automation"],
            "parent_context": payload.get("parent_context", shared_parent_context),
        }
        if (
            not isinstance(payload["stage_id"], str)
            or not isinstance(payload["position"], int)
            or isinstance(payload["position"], bool)
            or not isinstance(payload["deliverables"], list)
            or not isinstance(payload["non_goals"], list)
            or not isinstance(payload["acceptance_criteria"], list)
            or not isinstance(payload["depends_on_stage_ids"], list)
            or not isinstance(payload["compatibility_constraints"], list)
            or not isinstance(payload["covered_scope_item_ids"], list)
            or any(not isinstance(item, str) for key in (
                "deliverables", "non_goals", "acceptance_criteria", "depends_on_stage_ids",
                "compatibility_constraints", "covered_scope_item_ids",
            ) for item in payload[key])
            or any(not isinstance(values[key], str) for key in (
                "title", "scope", "dependency_notes", "rollout_risk", "automation", "parent_context",
            ))
            or values["non_goals"] is None
            or values["validation"] is None
        ):
            raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
        disposition_payload = payload.get("execution_disposition")
        execution_disposition: str | None = None
        disposition_rationale: str | None = None
        unresolved_design_decisions: tuple[str, ...] = ()
        if disposition_payload is not None:
            if (
                not isinstance(disposition_payload, dict)
                or disposition_payload.get("disposition") not in EXECUTION_DISPOSITION_VALUES
                or not isinstance(disposition_payload.get("rationale"), str)
                or not isinstance(disposition_payload.get("unresolved_design_decisions"), list)
                or any(
                    not isinstance(item, str)
                    for item in disposition_payload["unresolved_design_decisions"]
                )
            ):
                raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
            execution_disposition = str(disposition_payload["disposition"])
            disposition_rationale = str(disposition_payload["rationale"])
            unresolved_design_decisions = tuple(disposition_payload["unresolved_design_decisions"])
        return PlanPhase(
            **values,
            depends_on=(),
            stage_id=payload["stage_id"],
            position=payload["position"],
            deliverables=tuple(payload["deliverables"]),
            non_goals_items=tuple(payload["non_goals"]),
            acceptance_criteria=tuple(payload["acceptance_criteria"]),
            depends_on_stage_ids=tuple(payload["depends_on_stage_ids"]),
            compatibility_constraints=tuple(payload["compatibility_constraints"]),
            covered_scope_item_ids=tuple(payload["covered_scope_item_ids"]),
            execution_disposition=execution_disposition,
            disposition_rationale=disposition_rationale,
            unresolved_design_decisions=unresolved_design_decisions,
        )

    depends = payload.get("depends_on", [])
    if not isinstance(depends, list):
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
    values = {}
    for key in (
        "title",
        "scope",
        "non_goals",
        "dependency_notes",
        "rollout_risk",
        "validation",
        "automation",
    ):
        value = payload.get(key)
        if not isinstance(value, str):
            raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
        values[key] = value
    parent_context = payload.get("parent_context", shared_parent_context)
    if not isinstance(parent_context, str):
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
    values["parent_context"] = parent_context
    return PlanPhase(**values, depends_on=tuple(str(item) for item in depends))


def _decode_checkpoint(encoded: str) -> TopologyCheckpoint:
    # Accepts the compact form and the original uncompressed representation of
    # checkpoints posted before the compact payload format was introduced.
    payload = _decode_json_payload(encoded, marker_name="AGENT_PLAN_TOPOLOGY_CHECKPOINT")
    phases_payload = payload.get("phases")
    if not isinstance(phases_payload, list) or not phases_payload:
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
    retained_payload = payload.get("retained_parent_scope")
    shared_parent_context = payload.get("shared_parent_context")
    if shared_parent_context is not None and not isinstance(shared_parent_context, str):
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
    fresh = str(payload.get("topology_source")) == EXECUTION_TOPOLOGY_SOURCE
    retained = None
    if retained_payload is not None:
        if not isinstance(retained_payload, dict):
            raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
        retained = RetainedParentScope(
            plan_subject=str(retained_payload.get("plan_subject") or ""),
            plan_hash=str(retained_payload.get("plan_hash") or ""),
            excerpt=str(retained_payload.get("excerpt") or shared_parent_context or ""),
            status=(
                str(retained_payload.get("status") or "required")
                if fresh else "required"
            ),
            deliverables=(
                tuple(str(item) for item in retained_payload.get("deliverables", []))
                if fresh and isinstance(retained_payload.get("deliverables", []), list)
                else ()
            ),
            acceptance_criteria=(
                tuple(str(item) for item in retained_payload.get("acceptance_criteria", []))
                if fresh and isinstance(retained_payload.get("acceptance_criteria", []), list)
                else ()
            ),
            covered_scope_item_ids=(
                tuple(str(item) for item in retained_payload.get("covered_scope_item_ids", []))
                if fresh and isinstance(retained_payload.get("covered_scope_item_ids", []), list)
                else ()
            ),
        )
    try:
        architecture_identity = payload.get("architecture_identity")
        if architecture_identity is not None and not isinstance(architecture_identity, dict):
            raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
        raw_impact = payload.get("architecture_impact")
        architecture_impact = (
            asdict(parse_architecture_impact(raw_impact, context="checkpoint.architecture_impact"))
            if raw_impact is not None else None
        )
        raw_contract = payload.get("architecture_contract_version")
        if raw_contract is not None and raw_contract != 1:
            raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT architecture contract.")
        checkpoint = TopologyCheckpoint(
            parent_issue=int(payload["parent_issue"]),
            plan_hash=str(payload["plan_hash"]),
            mode=str(payload["mode"]),
            topology_source=str(payload["topology_source"]),
            phases=tuple(
                _phase_from_payload(
                    item,
                    shared_parent_context=shared_parent_context,
                    fresh=fresh,
                )
                for item in phases_payload
            ),
            retained_parent_scope=retained,
            architecture_identity=architecture_identity,
            architecture_impact=architecture_impact,
            architecture_contract_version=raw_contract,
            strategy=(
                str(payload["strategy"])
                if payload.get("strategy") is not None else None
            ),
            execution_strategy_contract_version=(
                int(payload["execution_strategy_contract_version"])
                if payload.get("execution_strategy_contract_version") is not None else None
            ),
            recommendation_digest=(
                str(payload["recommendation_digest"])
                if payload.get("recommendation_digest") is not None else None
            ),
            plan_subject=(
                str(payload["plan_subject"])
                if payload.get("plan_subject") is not None else None
            ),
        )
        if fresh and (
            checkpoint.strategy not in {"one-shot", "staged"}
            or checkpoint.execution_strategy_contract_version != EXECUTION_STRATEGY_CONTRACT_VERSION
            or not checkpoint.recommendation_digest
            or not checkpoint.plan_subject
        ):
            raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT fresh identity.")
        return checkpoint
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.") from exc


def find_existing_topology_checkpoint(
    comments: Sequence[object],
    *,
    parent_issue: int,
    plan_hash: str,
    mode: str | None = None,
    strategy: str | None = None,
    topology_source: str | None = None,
    recommendation_digest: str | None = None,
    plan_subject: str | None = None,
) -> TopologyCheckpoint | None:
    found: TopologyCheckpoint | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in TOPOLOGY_CHECKPOINT_MARKER_RE.finditer(body):
            checkpoint = _decode_checkpoint(match.group("payload"))
            matches = (
                checkpoint.parent_issue == parent_issue
                and checkpoint.plan_hash == plan_hash
                and (mode is None or checkpoint.mode == mode)
                and (strategy is None or checkpoint.strategy == strategy)
                and (topology_source is None or checkpoint.topology_source == topology_source)
                and (
                    recommendation_digest is None
                    or checkpoint.recommendation_digest == recommendation_digest
                )
                and (plan_subject is None or checkpoint.plan_subject == plan_subject)
            )
            if matches:
                if found is not None and _checkpoint_payload(found) != _checkpoint_payload(checkpoint):
                    raise AgentLoopError(
                        "Ambiguous topology recovery: multiple divergent checkpoint records match "
                        "the approved plan identity."
                    )
                found = checkpoint
    return found


def find_topology_checkpoints_for_parent(
    comments: Sequence[object], *, parent_issue: int
) -> tuple[TopologyCheckpoint, ...]:
    """Return every checkpoint for a parent, regardless of approved plan.

    Fresh execution recovery must inventory the parent before publishing a new
    decision.  The normal lookup is deliberately plan-scoped for legacy
    resume, while this helper closes the interrupted-run window where an older
    checkpoint would otherwise be invisible to a new approved plan.
    """
    found: list[TopologyCheckpoint] = []
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in TOPOLOGY_CHECKPOINT_MARKER_RE.finditer(body):
            checkpoint = _decode_checkpoint(match.group("payload"))
            if checkpoint.parent_issue == parent_issue:
                found.append(checkpoint)
    return tuple(found)


def format_topology_checkpoint(checkpoint: TopologyCheckpoint) -> str:
    encoded = _encode_compressed_json_payload(_checkpoint_payload(checkpoint))
    body = "\n".join(
        [
            f"Topology checkpoint recorded for issue #{checkpoint.parent_issue}.",
            "",
            f"Source: {checkpoint.topology_source}",
            f"Mode: {checkpoint.mode}",
            f"Stages: {len(checkpoint.phases)}",
            "",
            f"<!-- AGENT_PLAN_TOPOLOGY_CHECKPOINT: {encoded} -->",
            "-- coding-review-agent-loop",
        ]
    )
    if len(body) > MAX_GITHUB_BODY_CHARS:
        raise AgentLoopError(
            "Topology checkpoint exceeds the GitHub comment size limit after compact encoding; "
            "shorten the approved plan or use a smaller flat topology."
        )
    return body


def post_topology_checkpoint(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    checkpoint: TopologyCheckpoint,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=checkpoint.parent_issue,
        body=TrustedBody.canonical(
            format_topology_checkpoint(checkpoint),
            expected_tokens=("AGENT_PLAN_TOPOLOGY_CHECKPOINT",),
        ),
    )


def _dependency_lines(
    phase: PlanPhase,
    created: Sequence[CreatedPhaseIssue],
    *,
    placeholders: bool = False,
) -> list[str]:
    if not phase.depends_on:
        return ["- None."]
    by_title = {
        " ".join(item.phase.title.lower().split()): item
        for item in created
    }
    lines: list[str] = []
    for dependency in phase.depends_on:
        created_issue = by_title.get(" ".join(dependency.lower().split()))
        if created_issue and created_issue.issue_number is not None:
            lines.append(f"- depends on #{created_issue.issue_number}: {dependency}")
        elif created_issue and created_issue.issue_url:
            lines.append(f"- depends on {created_issue.issue_url}: {dependency}")
        elif created_issue:
            if placeholders:
                lines.append(f"- depends on __ORCHESTRATOR_ISSUE_NUMBER__: {dependency}")
            else:
                lines.append(f"- depends on previously created phase with unavailable issue URL: {dependency}")
        else:
            lines.append(f"- depends on prior phase: {dependency}")
    return lines


def _unresolved_dependencies(
    phase: PlanPhase,
    created: Sequence[CreatedPhaseIssue],
) -> tuple[str, ...]:
    by_title = {
        " ".join(item.phase.title.lower().split()): item
        for item in created
    }
    return tuple(
        dependency
        for dependency in phase.depends_on
        if (
            (item := by_title.get(" ".join(dependency.lower().split()))) is None
            or (item.issue_number is None and not item.issue_url)
        )
    )


def format_phase_issue_body(
    *,
    repo: str,
    parent_issue: int,
    approved_plan: str,
    phase: PlanPhase,
    created_so_far: Sequence[CreatedPhaseIssue],
    phase_identity_value: str | None = None,
    dependency_placeholders: bool = False,
    topology_source: str = "model",
    phase_index: int = 0,
    phase_plan_hash: str | None = None,
    strategy: str | None = None,
    recommendation_digest: str | None = None,
    execution_strategy_contract_version: int | None = None,
    inherited_matrix_row_ids: Sequence[str] = (),
) -> str:
    parent_url = f"https://github.com/{repo}/issues/{parent_issue}"
    disposition = getattr(phase, "execution_disposition", None)
    if phase.automation == "agent-pr" and disposition == EXECUTION_DISPOSITION_PLANNING:
        execution = (
            "This child requires its own reviewed plan before any implementation: the "
            "approved parent plan does not resolve every design decision for this stage. "
            "Run `agent-loop issue <this issue number> --plan-first --plan-execution-mode auto` "
            "so a child plan is produced and reviewed first; plain issue mode fails closed "
            "for this stage. Keep the eventual PR scoped to this phase."
        )
    elif phase.automation == "agent-pr":
        execution = (
            "Run `agent-loop issue <this issue number>` to implement this phase in its own PR. "
            "Keep the PR scoped to this phase."
        )
        if disposition == EXECUTION_DISPOSITION_DIRECT:
            execution += (
                " The reviewed parent stage is a complete implementation contract "
                "(implementation-ready); do not run it with `--plan-first`."
            )
    elif phase.automation == "human-action":
        execution = (
            "This phase requires human action before agent implementation continues. A human should perform "
            "the work, add a remark/update describing the result, and close this issue."
        )
    else:
        execution = (
            "This phase is a manual closure/checkpoint. A human should add the required remark/update and "
            "close this issue when the checkpoint is satisfied."
        )
    parent_excerpt = sanitize_historical_text(phase.parent_context)
    body = "\n".join(
        [
            f"Child phase issue for parent #{parent_issue}: {parent_url}",
            "",
            PARENT_EXCERPT_HEADING,
            parent_excerpt,
            "",
            "## Scope",
            phase.scope,
            "",
            "## Non-goals",
            phase.non_goals,
            "",
            "## Constraints and invariants from the parent plan",
            "The linked parent issue is the source of truth for the complete approved-plan constraints and invariants; the excerpt above is the phase-specific context supplied to this issue.",
            "",
            "## Dependency notes",
            phase.dependency_notes,
            "",
            "## Dependency links",
            *_dependency_lines(phase, created_so_far, placeholders=dependency_placeholders),
            "",
            "## Rollout risk",
            phase.rollout_risk,
            "",
            "## Validation / soak requirement",
            phase.validation,
            "",
            "## Automation classification",
            phase.automation,
            "",
            "## Execution instructions",
            execution,
        ]
    )
    if topology_source == EXECUTION_TOPOLOGY_SOURCE:
        reviewed_lines = ["", "## Reviewed stage allocation"]
        reviewed_lines.extend(
            [
                "Stable stage ID: " + sanitize_historical_text(getattr(phase, "stage_id", None) or ""),
                f"Position: {phase.position or phase_index}",
                "Deliverables:",
                *(f"- {sanitize_historical_text(value)}" for value in phase.deliverables),
                "Acceptance criteria:",
                *(f"- {sanitize_historical_text(value)}" for value in phase.acceptance_criteria),
                "Compatibility constraints:",
                *(
                    [f"- {sanitize_historical_text(value)}" for value in phase.compatibility_constraints]
                    or ["- None."]
                ),
                "Covered scope items: " + ", ".join(phase.covered_scope_item_ids),
            ]
        )
        body += "\n" + "\n".join(reviewed_lines)
        if disposition is not None:
            body += "\n\n" + "\n".join(_phase_disposition_lines(phase))
        if inherited_matrix_row_ids:
            body += "\n\n## Inherited parent risk-matrix obligations\n" + "\n".join(
                f"- {sanitize_historical_text(row_id)}" for row_id in inherited_matrix_row_ids
            )
    if phase_identity_value is not None:
        body += "\n\n" + _phase_identity_marker(
            phase_identity_value,
            parent_issue=parent_issue,
            plan_hash=phase_plan_hash or approved_plan_hash(approved_plan),
            source=topology_source,
            index=phase_index,
            stage_id=getattr(phase, "stage_id", None),
            strategy=strategy,
            recommendation_digest=recommendation_digest,
            execution_strategy_contract_version=execution_strategy_contract_version,
            inherited_matrix_row_ids=inherited_matrix_row_ids,
        )
    # Every other section is bounded by its own contract, so an oversized child
    # body is the inherited plan excerpt.  Keep the stage contract, the markers
    # and the identity intact, and point at the parent's canonical plan (#902).
    return fit_github_body(
        body,
        sections=(_parent_excerpt_section(parent_excerpt, parent_issue=parent_issue),),
        surface=f"Child phase issue body for parent #{parent_issue}",
    )


PARENT_EXCERPT_HEADING = "## Approved parent-plan excerpt for this phase"


def _parent_excerpt_section(excerpt: str, *, parent_issue: int) -> BoundedSection:
    """Describe the inherited plan excerpt embedded in a child phase body.

    The renderer and the fresh-recovery content check share this description so
    a published body that was shortened still matches its reviewed phase.
    """
    return BoundedSection(
        name="approved parent-plan excerpt",
        text=excerpt,
        pointer=(
            f"issue #{parent_issue}'s canonical plan comment and its "
            "machine-readable attachments"
        ),
    )


def _phase_disposition_lines(phase: PlanPhase) -> list[str]:
    """Render the reviewed execution disposition section of a fresh child body."""
    disposition = getattr(phase, "execution_disposition", None)
    if disposition is None:
        return []
    if disposition == EXECUTION_DISPOSITION_DIRECT:
        meaning = "Implementation-ready: the approved parent-plan slice is a complete implementation contract."
    elif disposition == EXECUTION_DISPOSITION_PLANNING:
        meaning = (
            "Requires its own reviewed plan: run this child with `--plan-first` before any "
            "implementation coder."
        )
    else:
        meaning = "Human-owned stage: no agent implementation or planning is dispatched."
    lines = [
        "## Execution disposition",
        sanitize_historical_text(disposition),
        meaning,
        "Rationale: " + sanitize_historical_text(phase.disposition_rationale or ""),
        "Unresolved design decisions:",
        *(
            [f"- {sanitize_historical_text(value)}" for value in phase.unresolved_design_decisions]
            or ["- None."]
        ),
    ]
    return lines


def _fresh_phase_content_matches(
    candidate: FoundIssue,
    *,
    parent_issue: int,
    phase: PlanPhase,
    inherited_matrix_row_ids: Sequence[str] = (),
) -> bool:
    """Check the reviewed content around a fresh phase identity marker."""
    if candidate.title != _phase_issue_title(parent_issue, phase.position or 0, phase):
        return False
    body = candidate.body
    if not isinstance(body, str):
        return False
    excerpt = _parent_excerpt_section(
        sanitize_historical_text(phase.parent_context), parent_issue=parent_issue
    )
    if not bounded_text_present(body, excerpt, after=PARENT_EXCERPT_HEADING):
        return False
    fragments = [
        f"Child phase issue for parent #{parent_issue}:",
        PARENT_EXCERPT_HEADING,
        "## Scope",
        phase.scope,
        "## Non-goals",
        phase.non_goals,
        "## Dependency notes",
        phase.dependency_notes,
        "## Rollout risk",
        phase.rollout_risk,
        "## Validation / soak requirement",
        phase.validation,
        "## Automation classification",
        phase.automation,
        "Stable stage ID: " + sanitize_historical_text(phase.stage_id or ""),
        f"Position: {phase.position}",
        "Deliverables:",
        *(f"- {sanitize_historical_text(value)}" for value in phase.deliverables),
        "Acceptance criteria:",
        *(f"- {sanitize_historical_text(value)}" for value in phase.acceptance_criteria),
        "Compatibility constraints:",
        *(
            [f"- {sanitize_historical_text(value)}" for value in phase.compatibility_constraints]
            or ["- None."]
        ),
        "Covered scope items: " + ", ".join(phase.covered_scope_item_ids),
        *_phase_disposition_lines(phase),
    ]
    if inherited_matrix_row_ids:
        fragments.extend(
            [
                "## Inherited parent risk-matrix obligations",
                *inherited_matrix_row_ids,
            ]
        )
    return all(fragment in body for fragment in fragments)


def create_decomposition_child_issues(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    approved_plan: str,
    decomposition: PlanDecomposition,
    topology_source: str = "model",
    issue_comments: Sequence[object] = (),
    mode: str = "decompose-only",
    retained_parent_scope: RetainedParentScope | None = None,
    strategy: str | None = None,
    execution_strategy_contract_version: int | None = None,
    recommendation_digest: str | None = None,
    plan_subject: str | None = None,
    preflight_only: bool = False,
    risk_test_matrix: RiskTestMatrix | dict[str, object] | None = None,
) -> tuple[CreatedPhaseIssue, ...] | NeedsHumanDecision:
    """Preflight, recover, and create one immutable decomposition topology."""
    plan_hash = approved_plan_hash(approved_plan)
    phases = tuple(decomposition.phases)
    if not phases:
        raise AgentLoopError("Plan decomposition produced no phases.")
    strategy = strategy or decomposition.strategy
    execution_strategy_contract_version = (
        execution_strategy_contract_version
        or decomposition.execution_strategy_contract_version
    )
    recommendation_digest = recommendation_digest or decomposition.recommendation_digest
    plan_subject = plan_subject or _plan_subject_from_text(approved_plan)
    fresh = topology_source == EXECUTION_TOPOLOGY_SOURCE
    parsed_matrix = (
        risk_test_matrix
        if isinstance(risk_test_matrix, RiskTestMatrix) or risk_test_matrix is None
        else parse_risk_test_matrix(risk_test_matrix)
    )
    if fresh and (
        strategy not in {"one-shot", "staged"}
        or execution_strategy_contract_version != EXECUTION_STRATEGY_CONTRACT_VERSION
        or not recommendation_digest
    ):
        raise AgentLoopError("Fresh decomposition is missing its canonical execution identity.")

    # Search is read-only and intentionally includes every issue state.  It
    # closes the create-before-summary crash window without trusting authorship.
    found = merge_found_issues(
        search_issues(
            runner,
            config=config,
            search=query,
            state="all",
        )
        for query in parent_child_search_queries(parent_issue)
    )
    expected_ids = {
        index: phase_identity(
            parent_issue=parent_issue,
            plan_hash=plan_hash,
            topology_source=topology_source,
            phase_index=index,
            phase=phase,
            stage_id=getattr(phase, "stage_id", None),
            execution_strategy_contract_version=execution_strategy_contract_version,
        )
        for index, phase in enumerate(phases, start=1)
    }
    exact: dict[str, FoundIssue] = {}
    recognized: set[str] = set()
    for candidate in found:
        marker = PHASE_IDENTITY_MARKER_RE.search(candidate.body or "")
        candidate_identity: str | None = None
        candidate_parent: int | None = None
        candidate_plan_hash: str | None = None
        candidate_source: str | None = None
        candidate_stage_id: int | str | None = None
        candidate_phase_index: int | None = None
        candidate_digest: str | None = None
        candidate_strategy: str | None = None
        candidate_contract: int | None = None
        candidate_inherited_matrix_row_ids: tuple[str, ...] = ()
        if marker:
            payload = _decode_json_payload(marker.group("payload"), marker_name="AGENT_PLAN_PHASE_IDENTITY")
            if isinstance(payload.get("identity"), str):
                candidate_identity = payload["identity"]
            if isinstance(payload.get("parent_issue"), int) and not isinstance(payload.get("parent_issue"), bool):
                candidate_parent = payload["parent_issue"]
            if isinstance(payload.get("plan_hash"), str):
                candidate_plan_hash = payload["plan_hash"]
            if isinstance(payload.get("source"), str):
                candidate_source = payload["source"]
            if isinstance(payload.get("stage_id"), (int, str)) and not isinstance(payload.get("stage_id"), bool):
                candidate_stage_id = payload["stage_id"]
            if isinstance(payload.get("phase_index"), int) and not isinstance(payload.get("phase_index"), bool):
                candidate_phase_index = payload["phase_index"]
            if isinstance(payload.get("recommendation_digest"), str):
                candidate_digest = payload["recommendation_digest"]
            if isinstance(payload.get("strategy"), str):
                candidate_strategy = payload["strategy"]
            if isinstance(payload.get("execution_strategy_contract_version"), int):
                candidate_contract = payload["execution_strategy_contract_version"]
            inherited_payload = payload.get("inherited_matrix_row_ids", [])
            if isinstance(inherited_payload, list) and all(
                isinstance(item, str) for item in inherited_payload
            ):
                candidate_inherited_matrix_row_ids = tuple(inherited_payload)
        if candidate_identity is not None and candidate_parent == parent_issue:
            recognized.add(candidate_identity)
            matched_expected = False
            for index, expected in expected_ids.items():
                if candidate_identity == expected:
                    matched_expected = True
                    expected_phase = phases[index - 1]
                    expected_stage_id = expected_phase.stage_id if fresh else index
                    metadata_matches = (
                        candidate_plan_hash == plan_hash
                        and candidate_source == topology_source
                        and candidate_stage_id == expected_stage_id
                    )
                    if fresh:
                        metadata_matches = metadata_matches and (
                            candidate_phase_index == index
                            and candidate_digest == recommendation_digest
                            and candidate_strategy == strategy
                            and candidate_contract == execution_strategy_contract_version
                            and candidate_inherited_matrix_row_ids
                            == risk_matrix_row_ids_for_owner(
                                parsed_matrix, expected_phase.stage_id or ""
                            )
                        )
                    if not metadata_matches:
                        raise AgentLoopError(
                            f"Invalid decomposition recovery identity metadata for phase {index}."
                        )
                    if fresh and not _fresh_phase_content_matches(
                        candidate,
                        parent_issue=parent_issue,
                        phase=expected_phase,
                        inherited_matrix_row_ids=risk_matrix_row_ids_for_owner(
                            parsed_matrix, expected_phase.stage_id or ""
                        ),
                    ):
                        raise AgentLoopError(
                            f"Fresh decomposition recovery content does not match phase {index}."
                        )
                    if expected in exact:
                        raise AgentLoopError(
                            f"Ambiguous decomposition recovery: multiple child issues carry identity {expected}."
                        )
                    exact[expected] = candidate
            if not matched_expected:
                # A generated phase for this parent with a different plan,
                # source, or stable stage identity must not be silently
                # adopted through title-shaped discovery.
                if fresh or candidate_source == EXECUTION_TOPOLOGY_SOURCE:
                    raise AgentLoopError(
                        "Decomposition recovery found a conflicting fresh phase identity "
                        "for the approved parent topology."
                    )
        else:
            legacy = LEGACY_SPLIT_IDENTITY_RE.search(candidate.body or "")
            if legacy and int(legacy.group("parent")) == parent_issue:
                if fresh:
                    raise AgentLoopError(
                        "Fresh decomposition recovery conflicts with an existing split child; "
                        "repair or resume the legacy split topology before creating approved phases."
                    )
                recognized.add("legacy:" + legacy.group("key").lower())
            elif candidate.body and f"#{parent_issue}" in candidate.body and candidate.title:
                # Count a parent-linked canonical child from another workflow
                # toward the parent budget, without adopting it as a desired
                # decomposition phase.
                recognized.add("linked:" + " ".join(candidate.title.casefold().split()))
        if not fresh and not candidate.body and candidate.title:
            # Some GitHub search responses omit bodies.  The generated parent
            # prefixed title is a canonical recovery key in that narrow case.
            for index, phase in enumerate(phases, start=1):
                if candidate.title == _phase_issue_title(parent_issue, index, phase):
                    identity = expected_ids[index]
                    if identity in exact:
                        raise AgentLoopError(
                            f"Ambiguous decomposition recovery: multiple title matches for phase {index}."
                        )
                    exact[identity] = candidate
                    recognized.add(identity)

    count = preflight_flat_child_count(
        parent_issue=parent_issue,
        source=topology_source,
        desired_keys=expected_ids.values(),
        recognized_keys=recognized,
        configured_limit=config.flat_child_limit,
    )
    if isinstance(count, NeedsHumanDecision):
        return count

    # Validate every title and body before any checkpoint or create.  Dependency
    # slots are orchestrator-owned placeholders and can only be replaced later
    # by adopted/created issue references.
    empty_prior = [
        CreatedPhaseIssue(phase=phase, issue_url=None, issue_number=None)
        for phase in phases
    ]
    for index, phase in enumerate(phases, start=1):
        title = _phase_issue_title(parent_issue, index, phase)
        TrustedBody.current_untrusted_visible(title)
        draft = format_phase_issue_body(
            repo=config.repo,
            parent_issue=parent_issue,
            approved_plan=approved_plan,
            phase=phase,
            created_so_far=empty_prior[: index - 1],
            phase_identity_value=expected_ids[index],
            dependency_placeholders=True,
            topology_source=topology_source,
            phase_index=index,
            phase_plan_hash=plan_hash,
            strategy=strategy,
            recommendation_digest=recommendation_digest,
            execution_strategy_contract_version=execution_strategy_contract_version,
            inherited_matrix_row_ids=(
                risk_matrix_row_ids_for_owner(parsed_matrix, phase.stage_id or "")
                if fresh else ()
            ),
        )
        TrustedBody.canonical(draft, expected_tokens=("AGENT_PLAN_PHASE_IDENTITY",))

    if preflight_only:
        # The approval boundary uses this read-only result to validate all
        # existing identities, content, and child-cap accounting before the
        # canonical execution decision is published.  Missing phases are
        # represented as planned placeholders; the normal call below will
        # create them after the decision is durable.
        return tuple(
            CreatedPhaseIssue(
                phase=phase,
                issue_url=(exact[expected_ids[index]].url if expected_ids[index] in exact else None),
                issue_number=(
                    exact[expected_ids[index]].number
                    or _issue_number_from_url(exact[expected_ids[index]].url)
                    if expected_ids[index] in exact
                    else None
                ),
                origin="adopted" if expected_ids[index] in exact else "planned",
            )
            for index, phase in enumerate(phases, start=1)
        )

    # Legacy topologies need their historical full checkpoint for recovery.
    # Fresh v1 recommendations already have a lossless approved-plan record
    # and bounded transport sidecars, so the durable checkpoint is replaced by
    # the compact execution decision published by the orchestrator.
    if (
        not config.dry_run
        and not fresh
        and find_existing_topology_checkpoint(
        issue_comments,
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        mode=mode,
        ) is None
    ):
        post_topology_checkpoint(
            runner,
            config=config,
            checkpoint=TopologyCheckpoint(
                parent_issue=parent_issue,
                plan_hash=plan_hash,
                mode=mode,
                topology_source=topology_source,
                phases=phases,
                retained_parent_scope=retained_parent_scope,
                architecture_identity=(
                    config.architecture_context.identity()
                    if hasattr(config.architecture_context, "identity") else None
                ),
                architecture_impact=(
                    sanitize_architecture_impact(asdict(decomposition.architecture_impact))
                    if decomposition.architecture_impact is not None else None
                ),
                architecture_contract_version=(
                    1 if decomposition.architecture_impact is not None else None
                ),
                strategy=strategy,
                execution_strategy_contract_version=execution_strategy_contract_version,
                recommendation_digest=recommendation_digest,
                plan_subject=plan_subject,
            ),
        )

    created: list[CreatedPhaseIssue] = []
    for index, phase in enumerate(phases, start=1):
        identity = expected_ids[index]
        found = exact.get(identity)
        if found is not None:
            created.append(
                CreatedPhaseIssue(
                    phase=phase,
                    issue_url=found.url,
                    issue_number=found.number or _issue_number_from_url(found.url),
                    origin="adopted",
                )
            )
            continue
        unresolved = () if config.dry_run else _unresolved_dependencies(phase, created)
        if unresolved:
            raise AgentLoopError(
                "Cannot create decomposition phase because dependency issue references are unavailable: "
                + ", ".join(unresolved)
            )
        title = _phase_issue_title(parent_issue, index, phase)
        body = format_phase_issue_body(
            repo=config.repo,
            parent_issue=parent_issue,
            approved_plan=approved_plan,
            phase=phase,
            created_so_far=created,
            phase_identity_value=identity,
            topology_source=topology_source,
            phase_index=index,
            phase_plan_hash=plan_hash,
            strategy=strategy,
            recommendation_digest=recommendation_digest,
            execution_strategy_contract_version=execution_strategy_contract_version,
            inherited_matrix_row_ids=(
                risk_matrix_row_ids_for_owner(parsed_matrix, phase.stage_id or "")
                if fresh else ()
            ),
        )
        if "__ORCHESTRATOR_ISSUE_NUMBER__" in body:
            raise AgentLoopError(
                "Cannot create decomposition phase because a dependency reference was not resolved."
            )
        issue_url = create_issue(
            runner,
            config=config,
            title=title,
            body=TrustedBody.canonical(
                body,
                expected_tokens=("AGENT_PLAN_PHASE_IDENTITY",),
            ),
        )
        if config.dry_run:
            # A test double may return a URL even though the real dry-run
            # Runner intentionally returns no remote issue reference.
            issue_url = None
        created.append(
            CreatedPhaseIssue(
                phase=phase,
                issue_url=issue_url,
                issue_number=_issue_number_from_url(issue_url),
                origin="created",
            )
        )
    return tuple(created)


# Prefix of a zlib-compressed record payload.  A plain payload is the base64 of
# a JSON object, so it always starts with `ey` and can never carry this prefix.
COMPRESSED_PAYLOAD_PREFIX = "v1_"


def _encode_json_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _encode_compressed_json_payload(payload: dict[str, object]) -> str:
    """Encode a record the way the topology checkpoint does (#909).

    Plan-derived text compresses by orders of magnitude, and `ensure_ascii=False`
    keeps a non-ASCII character at its UTF-8 size instead of a `\\uXXXX` escape.
    """
    raw = json.dumps(
        payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return COMPRESSED_PAYLOAD_PREFIX + base64.urlsafe_b64encode(
        zlib.compress(raw, 9)
    ).decode("ascii")


def _decode_json_payload(encoded: str, *, marker_name: str) -> dict[str, object]:
    """Decode a record in either the compressed or the original plain form.

    Records published before a producer switched to compression stay plain
    base64, so the form is detected from the prefix rather than assumed.
    """
    try:
        if encoded.startswith(COMPRESSED_PAYLOAD_PREFIX):
            raw = decompress_record_payload(
                base64.urlsafe_b64decode(
                    encoded[len(COMPRESSED_PAYLOAD_PREFIX):].encode("ascii")
                )
            )
        else:
            raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, zlib.error) as exc:
        raise AgentLoopError(f"Invalid {marker_name} payload.") from exc
    if not isinstance(payload, dict):
        raise AgentLoopError(f"Invalid {marker_name} payload.")
    return payload


def _decode_execution_decision(encoded: str) -> ExecutionDecision:
    payload = _decode_json_payload(encoded, marker_name="AGENT_PLAN_EXECUTION_DECISION")
    try:
        architecture_identity = payload.get("architecture_identity")
        architecture_impact = payload.get("architecture_impact")
        if architecture_identity is not None and not isinstance(architecture_identity, dict):
            raise ValueError("architecture_identity must be an object")
        if architecture_impact is not None and not isinstance(architecture_impact, dict):
            raise ValueError("architecture_impact must be an object")
        return ExecutionDecision(
            parent_issue=int(payload["parent_issue"]),
            plan_hash=str(payload["plan_hash"]),
            plan_subject=str(payload["plan_subject"]),
            execution_strategy_contract_version=int(
                payload["execution_strategy_contract_version"]
            ),
            strategy=str(payload["strategy"]),
            topology_source=str(payload["topology_source"]),
            recommendation_digest=str(payload["recommendation_digest"]),
            requested_policy=str(payload["requested_policy"]),
            current_action=str(payload["current_action"]),
            stage_ids=tuple(str(item) for item in payload.get("stage_ids", [])),
            scope_item_ids=tuple(str(item) for item in payload.get("scope_item_ids", [])),
            retained_parent_status=str(payload.get("retained_parent_status", "none")),
            final_integration_status=str(payload.get("final_integration_status", "none")),
            architecture_identity=architecture_identity,
            architecture_impact=architecture_impact,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentLoopError("Invalid AGENT_PLAN_EXECUTION_DECISION payload.") from exc


def format_execution_decision(decision: ExecutionDecision) -> str:
    """Render the compact approval-bound decision record."""
    encoded = _encode_json_payload(decision.to_payload())
    body = "\n".join(
        (
            f"Execution decision recorded for issue #{decision.parent_issue}.",
            "",
            f"Canonical strategy: {decision.strategy}",
            f"Execution action: {decision.current_action}",
            f"Recommendation digest: {decision.recommendation_digest}",
            "",
            f"<!-- AGENT_PLAN_EXECUTION_DECISION: {encoded} -->",
            "-- coding-review-agent-loop",
        )
    )
    if len(body) > MAX_GITHUB_BODY_CHARS:
        raise AgentLoopError("Execution decision record exceeds the GitHub body limit.")
    return body


def find_existing_execution_decision(
    comments: Sequence[object],
    *,
    parent_issue: int,
    plan_hash: str,
    plan_subject: str,
    strategy: str,
    recommendation_digest: str,
) -> ExecutionDecision | None:
    found: ExecutionDecision | None = None
    expected_identity = {
        "parent_issue": parent_issue,
        "plan_hash": plan_hash,
        "plan_subject": plan_subject,
        "execution_strategy_contract_version": EXECUTION_STRATEGY_CONTRACT_VERSION,
        "strategy": strategy,
        "topology_source": EXECUTION_TOPOLOGY_SOURCE,
        "recommendation_digest": recommendation_digest,
    }
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in EXECUTION_DECISION_MARKER_RE.finditer(body):
            decision = _decode_execution_decision(match.group("payload"))
            if decision.parent_issue != parent_issue:
                continue
            if decision.plan_hash != plan_hash:
                # A decision is durable approval-bound state, not a cache keyed
                # only by the currently visible plan.  If approval changed
                # after a crash, publishing a second decision would allow the
                # same parent to acquire two competing execution topologies.
                raise AgentLoopError(
                    "Conflicting execution decision exists for this parent under a different "
                    f"approved plan hash ({decision.plan_hash} vs {plan_hash}); repair or "
                    "resume the recorded plan before publishing a new execution decision."
                )
            if decision.identity() != expected_identity:
                raise AgentLoopError(
                    "Conflicting execution decision identity exists for the approved plan; "
                    "refusing to publish or adopt a different topology."
                )
            # Requested policy and current action are diagnostics only.
            # Explicit staged actions may resume the same canonical
            # decision, so do not fork identity on those fields.
            found = decision
    return found


def post_execution_decision(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    decision: ExecutionDecision,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=decision.parent_issue,
        body=TrustedBody.canonical(
            format_execution_decision(decision),
            expected_tokens=("AGENT_PLAN_EXECUTION_DECISION",),
        ),
    )


def recover_execution_recommendation(
    comments: Sequence[object],
    *,
    expected_digest: str | None = None,
) -> ExecutionStrategyRecommendation:
    """Hydrate a complete recommendation from plan comments and sidecars."""
    bodies = tuple(
        body for comment in comments if isinstance((body := getattr(comment, "body", None)), str)
    )
    found: ExecutionStrategyRecommendation | None = None
    for body in bodies:
        for marker in re.finditer(
            r"<!--\s*AGENT_EXECUTION_RECOMMENDATION:\s*"
            r"(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
            body,
            re.I,
        ):
            from .comment_rendering import decode_execution_recommendation_marker

            recommendation = parse_execution_recommendation_payload(
                decode_execution_recommendation_marker(marker.group("payload"), bodies=bodies),
                context="approved execution_recommendation",
            )
            digest = str(recommendation.identity()["recommendation_sha256"])
            if expected_digest is not None and digest != expected_digest:
                continue
            if found is not None and found.to_payload() != recommendation.to_payload():
                raise AgentLoopError(
                    "Ambiguous execution recovery: multiple divergent recommendations match "
                    "the approved plan."
                )
            found = recommendation
    if found is None:
        raise AgentLoopError(
            "Approved fresh execution recommendation is unavailable or its transport sidecars are incomplete."
        )
    return found


def _encode_metadata(metadata: DecompositionMetadata) -> str:
    payload: dict[str, object] = {
        "parent_issue": metadata.parent_issue,
        "plan_hash": metadata.plan_hash,
        "mode": metadata.mode,
        "phase_count": metadata.phase_count,
        "phase_titles": list(metadata.phase_titles),
        "automation": list(metadata.automation),
        "children": [
            {"title": title, "url": url, "number": number}
            for title, url, number in metadata.children
        ],
        "topology_source": metadata.topology_source,
        "retained_parent_scope": (
            {
                "plan_subject": metadata.retained_parent_scope.plan_subject,
                "plan_hash": metadata.retained_parent_scope.plan_hash,
                "excerpt": metadata.retained_parent_scope.excerpt,
            }
            if metadata.retained_parent_scope is not None
            else None
        ),
    }
    if metadata.topology_source == EXECUTION_TOPOLOGY_SOURCE:
        payload.update(
            {
                "strategy": metadata.strategy,
                "execution_strategy_contract_version": metadata.execution_strategy_contract_version,
                "recommendation_digest": metadata.recommendation_digest,
                "plan_subject": metadata.plan_subject,
                "stage_ids": list(metadata.stage_ids),
                "phase_identities": list(metadata.phase_identities),
            }
        )
        if any(item is not None for item in metadata.dispositions):
            payload["dispositions"] = list(metadata.dispositions)
        if metadata.retained_parent_scope is not None:
            retained = payload["retained_parent_scope"]
            assert isinstance(retained, dict)
            retained.update(
                {
                    "status": metadata.retained_parent_scope.status,
                    "deliverables": list(metadata.retained_parent_scope.deliverables),
                    "acceptance_criteria": list(metadata.retained_parent_scope.acceptance_criteria),
                    "covered_scope_item_ids": list(metadata.retained_parent_scope.covered_scope_item_ids),
                }
            )
        final_integration = metadata.final_integration_work or ExecutionAllocation(
            "none", (), (), ()
        )
        payload["final_integration_work"] = {
            "status": final_integration.status,
            "deliverables": list(final_integration.deliverables),
            "acceptance_criteria": list(final_integration.acceptance_criteria),
            "covered_scope_item_ids": list(final_integration.covered_scope_item_ids),
        }
    # Compressed so an embedded retained-parent excerpt costs a fraction of its
    # plain size; `_decode_metadata` still reads plain summaries (#909).
    return _encode_compressed_json_payload(payload)


def _decode_metadata(encoded: str) -> DecompositionMetadata:
    payload = _decode_json_payload(encoded, marker_name="AGENT_PLAN_DECOMPOSITION")
    children_payload = payload.get("children")
    if not isinstance(children_payload, list):
        raise AgentLoopError("Invalid AGENT_PLAN_DECOMPOSITION payload.")
    children: list[tuple[str, str | None, int | None]] = []
    for child in children_payload:
        if not isinstance(child, dict) or not isinstance(child.get("title"), str):
            raise AgentLoopError("Invalid AGENT_PLAN_DECOMPOSITION payload.")
        url = child.get("url")
        number = child.get("number")
        children.append(
            (
                child["title"],
                url if isinstance(url, str) else None,
                number if isinstance(number, int) else None,
            )
        )
    try:
        fresh = str(payload.get("topology_source")) == EXECUTION_TOPOLOGY_SOURCE
        retained_payload = payload.get("retained_parent_scope")
        retained = None
        if isinstance(retained_payload, dict):
            retained = RetainedParentScope(
                plan_subject=str(retained_payload.get("plan_subject") or ""),
                plan_hash=str(retained_payload.get("plan_hash") or ""),
                excerpt=str(retained_payload.get("excerpt") or ""),
                status=str(retained_payload.get("status") or "required"),
                deliverables=tuple(
                    str(item) for item in retained_payload.get("deliverables", [])
                    if isinstance(item, str)
                ),
                acceptance_criteria=tuple(
                    str(item) for item in retained_payload.get("acceptance_criteria", [])
                    if isinstance(item, str)
                ),
                covered_scope_item_ids=tuple(
                    str(item) for item in retained_payload.get("covered_scope_item_ids", [])
                    if isinstance(item, str)
                ),
            )
        final_payload = payload.get("final_integration_work")
        final_integration_work = None
        if isinstance(final_payload, dict):
            final_integration_work = ExecutionAllocation(
                status=str(final_payload.get("status") or "none"),
                deliverables=tuple(
                    str(item) for item in final_payload.get("deliverables", [])
                    if isinstance(item, str)
                ),
                acceptance_criteria=tuple(
                    str(item) for item in final_payload.get("acceptance_criteria", [])
                    if isinstance(item, str)
                ),
                covered_scope_item_ids=tuple(
                    str(item) for item in final_payload.get("covered_scope_item_ids", [])
                    if isinstance(item, str)
                ),
            )
        elif fresh:
            final_integration_work = ExecutionAllocation("none", (), (), ())
        metadata = DecompositionMetadata(
            parent_issue=int(payload["parent_issue"]),
            plan_hash=str(payload["plan_hash"]),
            mode=str(payload["mode"]),
            phase_count=int(payload["phase_count"]),
            phase_titles=tuple(str(value) for value in payload["phase_titles"]),
            automation=tuple(str(value) for value in payload["automation"]),
            children=tuple(children),
            topology_source=str(payload.get("topology_source") or "model"),
            retained_parent_scope=retained,
            final_integration_work=final_integration_work,
            strategy=(str(payload["strategy"]) if payload.get("strategy") is not None else None),
            execution_strategy_contract_version=(
                int(payload["execution_strategy_contract_version"])
                if payload.get("execution_strategy_contract_version") is not None else None
            ),
            recommendation_digest=(
                str(payload["recommendation_digest"])
                if payload.get("recommendation_digest") is not None else None
            ),
            plan_subject=(str(payload["plan_subject"]) if payload.get("plan_subject") is not None else None),
            stage_ids=tuple(str(item) for item in payload.get("stage_ids", [])),
            phase_identities=tuple(str(item) for item in payload.get("phase_identities", [])),
            dispositions=tuple(
                (str(item) if item is not None else None)
                for item in payload.get("dispositions", [])
            ),
        )
        if metadata.dispositions and (
            len(metadata.dispositions) != metadata.phase_count
            or any(
                item is not None and item not in EXECUTION_DISPOSITION_VALUES
                for item in metadata.dispositions
            )
        ):
            raise AgentLoopError("Invalid AGENT_PLAN_DECOMPOSITION disposition metadata.")
        if metadata.topology_source == EXECUTION_TOPOLOGY_SOURCE:
            if (
                metadata.strategy not in {"one-shot", "staged"}
                or metadata.execution_strategy_contract_version != EXECUTION_STRATEGY_CONTRACT_VERSION
                or not metadata.recommendation_digest
                or not metadata.plan_subject
                or len(metadata.stage_ids) != metadata.phase_count
                or len(metadata.phase_identities) != metadata.phase_count
            ):
                raise AgentLoopError(
                    "Invalid AGENT_PLAN_DECOMPOSITION fresh topology identity."
                )
        return metadata
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentLoopError("Invalid AGENT_PLAN_DECOMPOSITION payload.") from exc


def _encode_phase_implementation_handoff_metadata(
    metadata: PhaseImplementationHandoffMetadata,
) -> str:
    payload: dict[str, object] = {
        "parent_issue": metadata.parent_issue,
        "plan_hash": metadata.plan_hash,
        "mode": metadata.mode,
        "phase_index": metadata.phase_index,
        "phase_title": metadata.phase_title,
        "automation": metadata.automation,
        "child_issue_number": metadata.child_issue_number,
        "child_issue_url": metadata.child_issue_url,
    }
    if metadata.topology_source == EXECUTION_TOPOLOGY_SOURCE:
        payload.update(
            {
                "strategy": metadata.strategy,
                "topology_source": metadata.topology_source,
                "execution_strategy_contract_version": metadata.execution_strategy_contract_version,
                "recommendation_digest": metadata.recommendation_digest,
                "stage_id": metadata.stage_id,
                "plan_subject": metadata.plan_subject,
                **(
                    {"inherited_matrix_row_ids": list(metadata.inherited_matrix_row_ids)}
                    if metadata.inherited_matrix_row_ids else {}
                ),
                **(
                    {"execution_disposition": metadata.execution_disposition}
                    if metadata.execution_disposition is not None else {}
                ),
                **(
                    {"override_digest": metadata.override_digest}
                    if metadata.override_digest is not None else {}
                ),
            }
        )
    return _encode_json_payload(payload)


def _decode_phase_implementation_handoff_metadata(encoded: str) -> PhaseImplementationHandoffMetadata:
    payload = _decode_json_payload(encoded, marker_name="AGENT_PLAN_PHASE_IMPLEMENTATION")
    try:
        child_issue_url = payload.get("child_issue_url")
        metadata = PhaseImplementationHandoffMetadata(
            parent_issue=int(payload["parent_issue"]),
            plan_hash=str(payload["plan_hash"]),
            mode=str(payload["mode"]),
            phase_index=int(payload["phase_index"]),
            phase_title=str(payload["phase_title"]),
            automation=str(payload["automation"]),
            child_issue_number=int(payload["child_issue_number"]),
            child_issue_url=child_issue_url if isinstance(child_issue_url, str) else None,
            strategy=(str(payload["strategy"]) if payload.get("strategy") is not None else None),
            topology_source=(
                str(payload["topology_source"])
                if payload.get("topology_source") is not None else None
            ),
            execution_strategy_contract_version=(
                int(payload["execution_strategy_contract_version"])
                if payload.get("execution_strategy_contract_version") is not None else None
            ),
            recommendation_digest=(
                str(payload["recommendation_digest"])
                if payload.get("recommendation_digest") is not None else None
            ),
            stage_id=str(payload["stage_id"]) if payload.get("stage_id") is not None else None,
            plan_subject=(str(payload["plan_subject"]) if payload.get("plan_subject") is not None else None),
            inherited_matrix_row_ids=tuple(
                str(item) for item in payload.get("inherited_matrix_row_ids", [])
            ),
            execution_disposition=(
                str(payload["execution_disposition"])
                if payload.get("execution_disposition") is not None else None
            ),
            override_digest=(
                str(payload["override_digest"])
                if payload.get("override_digest") is not None else None
            ),
        )
        if metadata.execution_disposition is not None and metadata.execution_disposition not in {
            EXECUTION_DISPOSITION_DIRECT, EXECUTION_DISPOSITION_PLANNING,
        }:
            raise AgentLoopError("Invalid execution disposition in phase handoff.")
        if metadata.override_digest is not None and not re.fullmatch(
            r"[0-9a-f]{64}", metadata.override_digest
        ):
            raise AgentLoopError("Invalid override digest in phase handoff.")
        fresh_fields = {
            "strategy", "topology_source", "execution_strategy_contract_version",
            "recommendation_digest", "stage_id", "plan_subject",
            "execution_disposition", "override_digest",
        }
        inherited_ids = payload.get("inherited_matrix_row_ids", [])
        if not isinstance(inherited_ids, list) or any(
            not isinstance(item, str) or not item.strip() for item in inherited_ids
        ) or len(set(inherited_ids)) != len(inherited_ids):
            raise AgentLoopError("Invalid inherited risk matrix row IDs in phase handoff.")
        if metadata.topology_source != EXECUTION_TOPOLOGY_SOURCE and inherited_ids:
            raise AgentLoopError("Legacy phase handoffs cannot carry fresh matrix ownership metadata.")
        if metadata.topology_source == EXECUTION_TOPOLOGY_SOURCE:
            if (
                metadata.strategy != "staged"
                or metadata.execution_strategy_contract_version != EXECUTION_STRATEGY_CONTRACT_VERSION
                or not metadata.recommendation_digest
                or not metadata.stage_id
                or not metadata.plan_subject
            ):
                raise AgentLoopError("Invalid AGENT_PLAN_PHASE_IMPLEMENTATION fresh identity.")
        elif any(key in payload for key in fresh_fields):
            raise AgentLoopError("Invalid AGENT_PLAN_PHASE_IMPLEMENTATION source identity.")
        return metadata
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentLoopError("Invalid AGENT_PLAN_PHASE_IMPLEMENTATION payload.") from exc


def find_existing_decomposition(
    comments: Sequence[object],
    *,
    parent_issue: int,
    plan_hash: str | None,
    mode: str | None = None,
    strategy: str | None = None,
    topology_source: str | None = None,
    recommendation_digest: str | None = None,
    plan_subject: str | None = None,
) -> DecompositionMetadata | None:
    found: DecompositionMetadata | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in DECOMPOSITION_MARKER_RE.finditer(body):
            metadata = _decode_metadata(match.group("payload"))
            matches = (
                metadata.parent_issue == parent_issue
                and (plan_hash is None or metadata.plan_hash == plan_hash)
                and (mode is None or metadata.mode == mode)
                and (strategy is None or metadata.strategy == strategy)
                and (topology_source is None or metadata.topology_source == topology_source)
                and (
                    recommendation_digest is None
                    or metadata.recommendation_digest == recommendation_digest
                )
                and (plan_subject is None or metadata.plan_subject == plan_subject)
            )
            if matches:
                if found is not None and _encode_metadata(found) != _encode_metadata(metadata):
                    raise AgentLoopError(
                        "Ambiguous decomposition recovery: multiple divergent summaries match "
                        "the approved topology identity."
                    )
                found = metadata
    if found is None:
        return None
    if found.phase_count != len(found.children):
        known = ", ".join(url or f"#{number}" for _title, url, number in found.children if url or number)
        raise AgentLoopError(
            "Existing plan decomposition metadata is incomplete; manual recovery required before rerun. "
            f"Known child issues: {known or 'none'}."
        )
    return found


def find_decompositions_for_parent(
    comments: Sequence[object], *, parent_issue: int
) -> tuple[DecompositionMetadata, ...]:
    """Return every decomposition summary for a parent, across plan hashes."""
    found: list[DecompositionMetadata] = []
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in DECOMPOSITION_MARKER_RE.finditer(body):
            metadata = _decode_metadata(match.group("payload"))
            if metadata.parent_issue == parent_issue:
                found.append(metadata)
    return tuple(found)


def reject_legacy_topology_collision(
    comments: Sequence[object],
    *,
    parent_issue: int,
    plan_hash: str,
) -> None:
    """Fail closed when a fresh plan would reuse a historical topology.

    A fresh recommendation is keyed by its canonical strategy/source/digest.
    A same-plan legacy checkpoint or summary is not evidence for that identity
    and must not be bypassed by title-shaped child discovery.
    """
    # This inventory intentionally ignores the current plan hash.  A crash can
    # leave only a legacy summary/checkpoint; filtering by the newly approved
    # hash would make that state look absent and permit a second topology.
    for summary in find_decompositions_for_parent(comments, parent_issue=parent_issue):
        if (
            summary.plan_hash != plan_hash
            or summary.topology_source != EXECUTION_TOPOLOGY_SOURCE
        ):
            raise AgentLoopError(
                "Fresh execution topology conflicts with an existing decomposition summary "
                f"for plan {summary.plan_hash}; repair or resume the recorded topology before rerunning."
            )
    for checkpoint in find_topology_checkpoints_for_parent(comments, parent_issue=parent_issue):
        if (
            checkpoint.plan_hash != plan_hash
            or checkpoint.topology_source != EXECUTION_TOPOLOGY_SOURCE
        ):
            raise AgentLoopError(
                "Fresh execution topology conflicts with an existing topology checkpoint "
                f"for plan {checkpoint.plan_hash}; repair or resume the recorded topology before rerunning."
            )


def find_existing_phase_implementation_handoff(
    comments: Sequence[object],
    *,
    parent_issue: int,
    plan_hash: str,
    mode: str,
    phase_index: int,
    child_issue_number: int,
) -> PhaseImplementationHandoffMetadata | None:
    found: PhaseImplementationHandoffMetadata | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in PHASE_IMPLEMENTATION_MARKER_RE.finditer(body):
            metadata = _decode_phase_implementation_handoff_metadata(match.group("payload"))
            if (
                metadata.parent_issue == parent_issue
                and metadata.plan_hash == plan_hash
                and metadata.mode == mode
                and metadata.phase_index == phase_index
                and metadata.child_issue_number == child_issue_number
            ):
                found = metadata
    return found


def find_phase_implementation_handoffs(
    comments: Sequence[object],
    *,
    parent_issue: int,
    plan_hash: str,
) -> tuple[PhaseImplementationHandoffMetadata, ...]:
    """Return every phase handoff for a parent/plan identity.

    Recovery must inspect mismatched mode/source/stage records too; the
    single-phase lookup intentionally cannot do that because it is used by
    legacy resume paths.  Keeping this inventory beside the marker decoder
    gives CLI and skill preflight the same fail-closed view.
    """
    found: list[PhaseImplementationHandoffMetadata] = []
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in PHASE_IMPLEMENTATION_MARKER_RE.finditer(body):
            metadata = _decode_phase_implementation_handoff_metadata(match.group("payload"))
            if metadata.parent_issue == parent_issue and metadata.plan_hash == plan_hash:
                found.append(metadata)
    return tuple(found)


def find_phase_implementation_handoffs_for_parent(
    comments: Sequence[object], *, parent_issue: int
) -> tuple[PhaseImplementationHandoffMetadata, ...]:
    """Return every phase handoff for a parent, across plan hashes."""
    found: list[PhaseImplementationHandoffMetadata] = []
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in PHASE_IMPLEMENTATION_MARKER_RE.finditer(body):
            metadata = _decode_phase_implementation_handoff_metadata(match.group("payload"))
            if metadata.parent_issue == parent_issue:
                found.append(metadata)
    return tuple(found)


def _render_decomposition_parent_summary(
    *,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    created: Sequence[CreatedPhaseIssue],
    topology_source: str = "model",
    retained_parent_scope: RetainedParentScope | None = None,
    final_integration_work: ExecutionAllocation | None = None,
    strategy: str | None = None,
    execution_strategy_contract_version: int | None = None,
    recommendation_digest: str | None = None,
    plan_subject: str | None = None,
    inherited_matrix_row_ids: Sequence[str] = (),
) -> str:
    """Render the summary verbatim, without regard for the GitHub body limit."""
    phase_identities = tuple(
        phase_identity(
            parent_issue=parent_issue,
            plan_hash=plan_hash,
            topology_source=topology_source,
            phase_index=index,
            phase=item.phase,
            stage_id=getattr(item.phase, "stage_id", None),
            execution_strategy_contract_version=execution_strategy_contract_version,
        )
        for index, item in enumerate(created, start=1)
    )
    metadata = DecompositionMetadata(
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        mode=mode,
        phase_count=len(created),
        phase_titles=tuple(item.phase.title for item in created),
        automation=tuple(item.phase.automation for item in created),
        children=tuple(
            (item.phase.title, item.issue_url, item.issue_number)
            for item in created
        ),
        topology_source=topology_source,
        retained_parent_scope=retained_parent_scope,
        final_integration_work=(
            final_integration_work
            if topology_source == EXECUTION_TOPOLOGY_SOURCE
            else None
        ),
        strategy=strategy,
        execution_strategy_contract_version=execution_strategy_contract_version,
        recommendation_digest=recommendation_digest,
        plan_subject=plan_subject,
        stage_ids=tuple(
            getattr(item.phase, "stage_id", None) or str(index)
            for index, item in enumerate(created, start=1)
        ),
        phase_identities=phase_identities,
        dispositions=(
            tuple(getattr(item.phase, "execution_disposition", None) for item in created)
            if topology_source == EXECUTION_TOPOLOGY_SOURCE else ()
        ),
    )
    lines = [
        f"Approved plan decomposed for issue #{parent_issue}.",
        "",
        f"Mode: {mode}",
        f"Topology source: {topology_source}",
        "",
    ]
    if topology_source == EXECUTION_TOPOLOGY_SOURCE:
        lines.insert(3, f"Canonical strategy: {strategy}")
        lines.insert(4, f"Recommendation digest: {recommendation_digest}")
    if retained_parent_scope is not None:
        lines.extend(
            [
                "## Retained parent scope",
                f"Plan subject: {sanitize_historical_text(retained_parent_scope.plan_subject)}",
                f"Plan hash: {retained_parent_scope.plan_hash}",
                "The approved plan's primary scope remains owned by the parent; "
                "the typed stages below are its declared remainder.",
                "",
                retained_parent_scope.excerpt,
                "",
            ]
        )
    if topology_source == EXECUTION_TOPOLOGY_SOURCE:
        final = final_integration_work or ExecutionAllocation("none", (), (), ())
        lines.extend(
            [
                "## Final integration work",
                f"Status: {sanitize_historical_text(final.status)}",
                "Deliverables:",
                *(
                    [f"- {sanitize_historical_text(value)}" for value in final.deliverables]
                    or ["- None."]
                ),
                "Acceptance criteria:",
                *(
                    [f"- {sanitize_historical_text(value)}" for value in final.acceptance_criteria]
                    or ["- None."]
                ),
                "Covered scope items: " + ", ".join(final.covered_scope_item_ids),
                "",
            ]
        )
    lines.extend(
        [
            "| Phase | Automation | Child issue | Risk |",
            "| --- | --- | --- | --- |",
        ]
    )
    for index, item in enumerate(created, start=1):
        if item.issue_url:
            child = item.issue_url
        elif item.issue_number is not None:
            child = f"#{item.issue_number}"
        else:
            child = "Created issue URL unavailable from GitHub CLI output."
        human_note = " Human remark and closure required." if item.phase.automation != "agent-pr" else ""
        lines.append(
            f"| {index}. {item.phase.title} | {item.phase.automation} | {child} | {item.phase.rollout_risk}{human_note} |"
        )
    if any(item is not None for item in metadata.dispositions):
        lines.extend(["", "Reviewed execution dispositions:"])
        for index, item in enumerate(created, start=1):
            disposition = getattr(item.phase, "execution_disposition", None)
            lines.append(
                f"- {index}. {sanitize_historical_text(item.phase.title)}: "
                f"{sanitize_historical_text(disposition or 'legacy-ambiguous (routes to child planning)')}"
            )
    lines.extend(
        [
            "",
            "Every phase above has a GitHub child issue; this table is only a summary.",
            "",
            f"<!-- AGENT_PLAN_DECOMPOSITION: {_encode_metadata(metadata)} -->",
            "-- coding-review-agent-loop",
        ]
    )
    return "\n".join(lines)


def retained_parent_excerpt_section(excerpt: str, *, parent_issue: int) -> BoundedSection:
    """Describe the retained-parent excerpt as a shortenable body section.

    Both the renderer that shortens the excerpt and the recovery check that
    reconciles an already-published one build the section here, so the pointer
    notice they look for is the same string (#907).
    """
    return BoundedSection(
        name="retained parent scope excerpt",
        text=excerpt,
        pointer=(
            f"issue #{parent_issue}'s canonical plan comment and its "
            "machine-readable attachments"
        ),
    )


def retained_parent_scope_matches(
    recorded: RetainedParentScope | None,
    expected: RetainedParentScope | None,
    *,
    parent_issue: int,
) -> bool:
    """Compare a published retained-parent scope against a recomputed one.

    A summary published for a large plan carries a shortened excerpt, while the
    scope recomputed from the approved plan on a rerun always carries the full
    text.  Plain equality would then wedge exactly the runs the bounding makes
    publishable, so the excerpt is accepted when it is exactly the recomputed
    text or exactly one of the shortened forms this parent's renderer can
    produce for it; every other field must still match exactly (#907).
    """
    if recorded is None or expected is None:
        return recorded == expected
    if dataclasses.replace(recorded, excerpt="") != dataclasses.replace(expected, excerpt=""):
        return False
    if not expected.excerpt:
        return not recorded.excerpt
    return is_bounded_form(
        recorded.excerpt,
        retained_parent_excerpt_section(expected.excerpt, parent_issue=parent_issue),
    )


def format_decomposition_parent_summary(
    *,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    created: Sequence[CreatedPhaseIssue],
    topology_source: str = "model",
    retained_parent_scope: RetainedParentScope | None = None,
    final_integration_work: ExecutionAllocation | None = None,
    strategy: str | None = None,
    execution_strategy_contract_version: int | None = None,
    recommendation_digest: str | None = None,
    plan_subject: str | None = None,
    inherited_matrix_row_ids: Sequence[str] = (),
) -> str:
    """Render the parent summary, bounding the retained excerpt so it fits.

    The retained-parent excerpt is plan-derived and appears twice: once as
    visible text and once inside the base64 record payload.  Shortening only
    the visible copy therefore cannot make an oversized summary fit, so the
    excerpt is bounded at its source and the summary re-rendered (#907).
    """

    def render(scope: RetainedParentScope | None) -> str:
        return _render_decomposition_parent_summary(
            parent_issue=parent_issue,
            mode=mode,
            plan_hash=plan_hash,
            created=created,
            topology_source=topology_source,
            retained_parent_scope=scope,
            final_integration_work=final_integration_work,
            strategy=strategy,
            execution_strategy_contract_version=execution_strategy_contract_version,
            recommendation_digest=recommendation_digest,
            plan_subject=plan_subject,
            inherited_matrix_row_ids=inherited_matrix_row_ids,
        )

    body = render(retained_parent_scope)
    if len(body) <= MAX_GITHUB_BODY_CHARS:
        return body
    if retained_parent_scope is None or not retained_parent_scope.excerpt:
        raise AgentLoopError(_parent_summary_overflow(parent_issue, body, excerpt_chars=None))
    section = retained_parent_excerpt_section(
        retained_parent_scope.excerpt, parent_issue=parent_issue
    )

    def render_budget(budget: int) -> str:
        return render(
            dataclasses.replace(
                retained_parent_scope, excerpt=shortened_section(section, budget=budget)
            )
        )

    # A retained character does not cost a fixed number of body characters:
    # the record payload is compressed (#909), so its cost depends on how
    # repetitive the excerpt is, and the visible copy costs one character each.
    # Search for the largest budget whose actually rendered body fits instead
    # of assuming a ratio (#907).
    target = MAX_GITHUB_BODY_CHARS - BODY_SAFETY_MARGIN
    shortest = render_budget(0)
    if len(shortest) > target:
        if len(shortest) <= MAX_GITHUB_BODY_CHARS:
            return shortest
        raise AgentLoopError(
            _parent_summary_overflow(parent_issue, shortest, excerpt_chars=len(section.text))
        )
    low, high = 0, len(section.text)
    fitted = shortest
    while low < high:
        middle = (low + high + 1) // 2
        candidate = render_budget(middle)
        if len(candidate) <= target:
            fitted, low = candidate, middle
        else:
            high = middle - 1
    return fitted


def _parent_summary_overflow(parent_issue: int, body: str, *, excerpt_chars: int | None) -> str:
    excess = len(body) - MAX_GITHUB_BODY_CHARS
    if excerpt_chars is None:
        culprit = (
            "it embeds no shortenable retained-parent excerpt, so its fixed contract "
            "text is already too large to publish"
        )
    else:
        culprit = (
            f"even with its retained parent scope excerpt ({excerpt_chars} characters) "
            "reduced to a pointer it does not fit; the surrounding fixed contract text "
            "and the record payload leave no room for it"
        )
    return (
        f"Decomposition parent summary for issue #{parent_issue} exceeds the GitHub body "
        f"limit of {MAX_GITHUB_BODY_CHARS} characters by {excess} characters: {culprit}."
    )


def format_phase_implementation_handoff_comment(
    *,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    phase_index: int,
    created: CreatedPhaseIssue,
    strategy: str | None = None,
    topology_source: str | None = None,
    execution_strategy_contract_version: int | None = None,
    recommendation_digest: str | None = None,
    plan_subject: str | None = None,
    inherited_matrix_row_ids: Sequence[str] = (),
    execution_disposition: str | None = None,
    override_digest: str | None = None,
) -> str:
    if created.issue_number is None:
        raise AgentLoopError(
            "Cannot record decomposed phase implementation handoff because the child issue number is unavailable."
        )
    if execution_disposition is not None and execution_disposition not in {
        EXECUTION_DISPOSITION_DIRECT, EXECUTION_DISPOSITION_PLANNING,
    }:
        raise AgentLoopError(
            f"A phase handoff cannot record execution disposition `{execution_disposition}`."
        )
    if execution_disposition is None and override_digest is not None:
        raise AgentLoopError("A phase handoff override digest requires an explicit disposition.")
    metadata = PhaseImplementationHandoffMetadata(
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        mode=mode,
        phase_index=phase_index,
        phase_title=created.phase.title,
        automation=created.phase.automation,
        child_issue_number=created.issue_number,
        child_issue_url=created.issue_url,
        strategy=strategy,
        topology_source=topology_source,
        execution_strategy_contract_version=execution_strategy_contract_version,
        recommendation_digest=recommendation_digest,
        stage_id=getattr(created.phase, "stage_id", None),
        plan_subject=plan_subject,
        inherited_matrix_row_ids=tuple(inherited_matrix_row_ids),
        execution_disposition=(
            execution_disposition if topology_source == EXECUTION_TOPOLOGY_SOURCE else None
        ),
        override_digest=(
            override_digest if topology_source == EXECUTION_TOPOLOGY_SOURCE else None
        ),
    )
    child = created.issue_url or f"#{created.issue_number}"
    planning = metadata.execution_disposition == EXECUTION_DISPOSITION_PLANNING
    if planning:
        headline = (
            f"Approved plan for issue #{parent_issue} requires child planning for phase "
            f"{phase_index}: {child}."
        )
        resume = (
            "Parent reruns will not automatically re-run this child. The child must produce "
            "and review its own plan before any implementation coder; resume directly with "
            f"`agent-loop issue {created.issue_number} --plan-first --plan-execution-mode auto`."
        )
    else:
        headline = (
            f"Approved plan implementation for issue #{parent_issue} handed off to phase "
            f"{phase_index}: {child}."
        )
        resume = (
            "Parent reruns will not automatically re-run this child implementation. "
            f"Resume directly with `agent-loop issue {created.issue_number}`."
        )
    lines = [
        headline,
        "",
        f"Mode: {mode}",
        f"Phase: {created.phase.title}",
        f"Automation: {created.phase.automation}",
        "",
        resume,
        "",
        f"<!-- AGENT_PLAN_PHASE_IMPLEMENTATION: {_encode_phase_implementation_handoff_metadata(metadata)} -->",
        "-- coding-review-agent-loop",
    ]
    if topology_source == EXECUTION_TOPOLOGY_SOURCE:
        lines.insert(5, f"Canonical strategy: {strategy} (source: {topology_source})")
        lines.insert(6, f"Stable stage ID: {getattr(created.phase, 'stage_id', None)}")
        if metadata.execution_disposition is not None:
            lines.insert(7, f"Execution disposition: {metadata.execution_disposition}")
        if metadata.override_digest is not None:
            lines.insert(8, f"Applied signed override digest: {metadata.override_digest}")
    return "\n".join(lines)


def post_phase_implementation_handoff_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    phase_index: int,
    created: CreatedPhaseIssue,
    strategy: str | None = None,
    topology_source: str | None = None,
    execution_strategy_contract_version: int | None = None,
    recommendation_digest: str | None = None,
    plan_subject: str | None = None,
    inherited_matrix_row_ids: Sequence[str] = (),
    execution_disposition: str | None = None,
    override_digest: str | None = None,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=parent_issue,
        body=TrustedBody.canonical(
            format_phase_implementation_handoff_comment(
                parent_issue=parent_issue,
                mode=mode,
                plan_hash=plan_hash,
                phase_index=phase_index,
                created=created,
                strategy=strategy,
                topology_source=topology_source,
                execution_strategy_contract_version=execution_strategy_contract_version,
                recommendation_digest=recommendation_digest,
                plan_subject=plan_subject,
                inherited_matrix_row_ids=inherited_matrix_row_ids,
                execution_disposition=execution_disposition,
                override_digest=override_digest,
            ),
            expected_tokens=("AGENT_PLAN_PHASE_IMPLEMENTATION",),
        ),
    )


# ---------------------------------------------------------------------------
# Signed child-disposition overrides and the shared reconciliation rule (#808)
# ---------------------------------------------------------------------------

CHILD_DISPOSITION_OVERRIDE_KIND = "child-execution-disposition-override"
CHILD_DISPOSITION_OVERRIDE_SCHEMA_VERSION = 1
_OVERRIDE_RECORD_KEYS = frozenset(
    {"kind", "schema_version", "parent_issue", "plan_hash", "stage_id", "disposition", "rationale"}
)
_FENCED_JSON_RE = re.compile(r"```(?:json)?[ \t]*\n(?P<body>.*?)\n```", re.S | re.I)


@dataclass(frozen=True)
class ChildDispositionOverride:
    """One signed, durable human override of a child's execution disposition."""

    parent_issue: int
    plan_hash: str
    stage_id: str
    disposition: str
    rationale: str
    digest: str
    comment_locator: str

    def record_payload(self) -> dict[str, object]:
        return {
            "kind": CHILD_DISPOSITION_OVERRIDE_KIND,
            "schema_version": CHILD_DISPOSITION_OVERRIDE_SCHEMA_VERSION,
            "parent_issue": self.parent_issue,
            "plan_hash": self.plan_hash,
            "stage_id": self.stage_id,
            "disposition": self.disposition,
            "rationale": self.rationale,
        }


def child_disposition_override_digest(record: dict[str, object]) -> str:
    """The one record digest shared by discovery, the handoff writer, and reconciliation."""
    canonical = json.dumps(
        {key: record[key] for key in sorted(_OVERRIDE_RECORD_KEYS)},
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def format_child_disposition_override_comment(
    *,
    parent_issue: int,
    plan_hash: str,
    stage_id: str,
    disposition: str,
    rationale: str,
) -> str:
    """Render the signed override record format documented for human reviewers."""
    record = {
        "kind": CHILD_DISPOSITION_OVERRIDE_KIND,
        "schema_version": CHILD_DISPOSITION_OVERRIDE_SCHEMA_VERSION,
        "parent_issue": parent_issue,
        "plan_hash": plan_hash,
        "stage_id": stage_id,
        "disposition": disposition,
        "rationale": rationale,
    }
    return (
        "Child execution disposition override:\n\n```json\n"
        + json.dumps(record, indent=2, sort_keys=True)
        + "\n```\n-- Human Reviewer"
    )


def parse_child_disposition_override_records(
    body: str | None, *, comment_locator: str
) -> tuple[tuple[ChildDispositionOverride, ...], tuple[str, ...]]:
    """Return (signed valid records, ignored-record diagnostics) for one comment.

    Only bodies carrying the standalone ``-- Human Reviewer`` signature are
    considered.  Malformed records are ignored (reported) rather than raised
    so an unsigned or half-written comment can never change a route.
    """
    signed = parse_signed_human_requirement_body(body)
    if signed is None:
        return (), ()
    records: list[ChildDispositionOverride] = []
    ignored: list[str] = []
    for match in _FENCED_JSON_RE.finditer(signed):
        try:
            payload = json.loads(match.group("body"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("kind") != CHILD_DISPOSITION_OVERRIDE_KIND:
            continue
        problem = _override_record_problem(payload)
        if problem is not None:
            ignored.append(f"{comment_locator}: malformed override record ignored ({problem})")
            continue
        records.append(
            ChildDispositionOverride(
                parent_issue=int(payload["parent_issue"]),
                plan_hash=str(payload["plan_hash"]),
                stage_id=str(payload["stage_id"]),
                disposition=str(payload["disposition"]),
                rationale=str(payload["rationale"]),
                digest=child_disposition_override_digest(payload),
                comment_locator=comment_locator,
            )
        )
    return tuple(records), tuple(ignored)


def _override_record_problem(payload: dict[str, object]) -> str | None:
    keys = set(payload)
    if keys != _OVERRIDE_RECORD_KEYS:
        missing = sorted(_OVERRIDE_RECORD_KEYS - keys)
        unknown = sorted(keys - _OVERRIDE_RECORD_KEYS)
        return f"missing keys {missing}, unknown keys {unknown}"
    if payload.get("schema_version") != CHILD_DISPOSITION_OVERRIDE_SCHEMA_VERSION:
        return "schema_version must be 1"
    parent = payload.get("parent_issue")
    if not isinstance(parent, int) or isinstance(parent, bool) or parent < 1:
        return "parent_issue must be a positive integer"
    for key in ("plan_hash", "stage_id", "rationale"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"{key} must be a non-empty string"
    if payload.get("disposition") not in {EXECUTION_DISPOSITION_DIRECT, EXECUTION_DISPOSITION_PLANNING}:
        return "disposition must be direct-implementation or requires-child-planning"
    return None


def collect_child_disposition_overrides(
    *,
    parent_comments: Sequence[object],
    child_comments: Sequence[object] = (),
    parent_issue: int,
    plan_hash: str,
    topology_stage_ids: Sequence[str],
    routed_stage_id: str,
    child_stage_id: str | None = None,
    child_issue_number: int | None = None,
    ignored_sink: list[str] | None = None,
) -> tuple[ChildDispositionOverride, ...]:
    """Two-step signed override discovery.

    Step 1 (topology validation) raises a human-decision error for any signed
    record whose parent issue or plan hash does not match the bound topology,
    whose stage is not part of that topology, or, for a record found on a
    child issue, whose stage is not that child's own stage.  Step 2 (stage
    scoping) returns only records for ``routed_stage_id`` with identical
    duplicates collapsed by digest.  Valid records for other stages are
    neither errors nor inputs for the current route.
    """
    found: list[ChildDispositionOverride] = []
    sources = [(f"parent issue #{parent_issue}", parent_comments, None)]
    if child_issue_number is not None:
        sources.append((f"child issue #{child_issue_number}", child_comments, child_stage_id))
    elif child_comments:
        sources.append(("child issue", child_comments, child_stage_id))
    for label, comments, local_stage_id in sources:
        for index, comment in enumerate(comments, start=1):
            body = getattr(comment, "body", None)
            if not isinstance(body, str):
                continue
            locator = f"{label} comment {index}"
            records, ignored = parse_child_disposition_override_records(
                body, comment_locator=locator
            )
            if ignored_sink is not None:
                ignored_sink.extend(ignored)
            for record in records:
                if record.parent_issue != parent_issue or record.plan_hash != plan_hash:
                    raise AgentLoopError(
                        "Human decision required: signed child-disposition override at "
                        f"{record.comment_locator} names parent #{record.parent_issue} / plan "
                        f"{record.plan_hash}, but the bound topology is parent #{parent_issue} / "
                        f"plan {plan_hash}. Remove or correct the record before rerunning."
                    )
                if record.stage_id not in tuple(topology_stage_ids):
                    raise AgentLoopError(
                        "Human decision required: signed child-disposition override at "
                        f"{record.comment_locator} names stage `{record.stage_id}`, which is not a "
                        "stage of the bound topology. Remove or correct the record before rerunning."
                    )
                if local_stage_id is not None and record.stage_id != local_stage_id:
                    raise AgentLoopError(
                        "Human decision required: signed child-disposition override at "
                        f"{record.comment_locator} names stage `{record.stage_id}`, but that child "
                        f"issue is stage `{local_stage_id}`. Post stage overrides for other children "
                        "on the parent issue or on their own child issue."
                    )
                found.append(record)
    scoped: list[ChildDispositionOverride] = []
    seen: set[str] = set()
    for record in found:
        if record.stage_id != routed_stage_id or record.digest in seen:
            continue
        seen.add(record.digest)
        scoped.append(record)
    return tuple(scoped)


# ---------------------------------------------------------------------------
# Signed child-plan supersession and the same-PR rebind audit record (#936)
# ---------------------------------------------------------------------------

CHILD_PLAN_SUPERSESSION_KIND = "child-plan-supersession"
CHILD_PLAN_SUPERSESSION_SCHEMA_VERSION = 1
_SUPERSESSION_RECORD_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "child_issue",
        "parent_issue",
        "stage_id",
        "superseded_plan_hash",
        "rationale",
    }
)
CHILD_PLAN_REBIND_MARKER_RE = re.compile(
    r"<!--\s*AGENT_CHILD_PLAN_REBIND:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", re.I
)
_REBIND_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "child_issue",
        "pr_number",
        "superseded_plan_hash",
        "new_plan_hash",
        "plan_supersession_digest",
        "first_replan_round",
        "approved_round",
    }
)


@dataclass(frozen=True)
class ChildPlanSupersession:
    """One signed human authorization to re-plan an approved child plan."""

    child_issue: int
    parent_issue: int
    stage_id: str
    superseded_plan_hash: str
    rationale: str
    digest: str
    comment_locator: str
    comment_index: int


@dataclass(frozen=True)
class ChildPlanRebindRecord:
    """Audit record posted in the same comment as a same-PR plan replacement."""

    child_issue: int
    pr_number: int
    superseded_plan_hash: str
    new_plan_hash: str
    plan_supersession_digest: str
    first_replan_round: int
    approved_round: int
    comment_index: int = -1


@dataclass(frozen=True)
class AuthorizedReplanLineage:
    """The contiguous digest-bound coder rounds of one authorized re-plan."""

    round_numbers: tuple[int, ...]
    latest_plan_hash: str

    @property
    def first_round(self) -> int:
        return self.round_numbers[0]

    @property
    def latest_round(self) -> int:
        return self.round_numbers[-1]


def child_plan_supersession_digest(record: dict[str, object]) -> str:
    """The one digest shared by discovery, round metadata, and rebind verification."""
    canonical = json.dumps(
        {key: record[key] for key in sorted(_SUPERSESSION_RECORD_KEYS)},
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def format_child_plan_supersession_comment(
    *,
    child_issue: int,
    parent_issue: int,
    stage_id: str,
    superseded_plan_hash: str,
    rationale: str,
) -> str:
    """Render the signed supersession record format documented for human reviewers."""
    record = {
        "kind": CHILD_PLAN_SUPERSESSION_KIND,
        "schema_version": CHILD_PLAN_SUPERSESSION_SCHEMA_VERSION,
        "child_issue": child_issue,
        "parent_issue": parent_issue,
        "stage_id": stage_id,
        "superseded_plan_hash": superseded_plan_hash,
        "rationale": rationale,
    }
    return (
        "Child plan supersession:\n\n```json\n"
        + json.dumps(record, indent=2, sort_keys=True)
        + "\n```\n-- Human Reviewer"
    )


def _supersession_record_problem(payload: dict[str, object]) -> str | None:
    keys = set(payload)
    if keys != _SUPERSESSION_RECORD_KEYS:
        missing = sorted(_SUPERSESSION_RECORD_KEYS - keys)
        unknown = sorted(keys - _SUPERSESSION_RECORD_KEYS)
        return f"missing keys {missing}, unknown keys {unknown}"
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != CHILD_PLAN_SUPERSESSION_SCHEMA_VERSION
    ):
        return "schema_version must be 1"
    for key in ("child_issue", "parent_issue"):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            return f"{key} must be a positive integer"
    for key in ("stage_id", "superseded_plan_hash", "rationale"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"{key} must be a non-empty string"
    return None


def parse_child_plan_supersession_records(
    body: str | None, *, comment_locator: str, comment_index: int = -1
) -> tuple[tuple[ChildPlanSupersession, ...], tuple[str, ...]]:
    """Return (signed valid records, ignored-record diagnostics) for one comment.

    Same contract as the disposition override parser: only a body with the
    standalone human reviewer signature counts, and a malformed record is
    reported and ignored so it can never reopen planning.
    """
    signed = parse_signed_human_requirement_body(body)
    if signed is None:
        return (), ()
    records: list[ChildPlanSupersession] = []
    ignored: list[str] = []
    for match in _FENCED_JSON_RE.finditer(signed):
        try:
            payload = json.loads(match.group("body"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("kind") != CHILD_PLAN_SUPERSESSION_KIND:
            continue
        problem = _supersession_record_problem(payload)
        if problem is not None:
            ignored.append(
                f"{comment_locator}: malformed child-plan supersession record ignored ({problem})"
            )
            continue
        records.append(
            ChildPlanSupersession(
                child_issue=int(payload["child_issue"]),
                parent_issue=int(payload["parent_issue"]),
                stage_id=str(payload["stage_id"]),
                superseded_plan_hash=str(payload["superseded_plan_hash"]),
                rationale=str(payload["rationale"]),
                digest=child_plan_supersession_digest(payload),
                comment_locator=comment_locator,
                comment_index=comment_index,
            )
        )
    return tuple(records), tuple(ignored)


def collect_child_plan_supersessions(
    child_comments: Sequence[object],
    *,
    child_issue: int,
    parent_issue: int,
    stage_id: str,
    ignored_sink: list[str] | None = None,
) -> tuple[ChildPlanSupersession, ...]:
    """Signed supersession discovery on the child issue only.

    A signed record naming another child, parent, or stage fails closed.
    Identical duplicates collapse by digest (the earliest comment is kept).
    Records for different superseded hashes coexist, but two distinct records
    for one superseded hash always fail closed: the orchestrator never
    chooses between them by comment order.
    """
    by_digest: dict[str, ChildPlanSupersession] = {}
    for index, comment in enumerate(child_comments):
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        locator = f"child issue #{child_issue} comment {index + 1}"
        records, ignored = parse_child_plan_supersession_records(
            body, comment_locator=locator, comment_index=index
        )
        if ignored_sink is not None:
            ignored_sink.extend(ignored)
        for record in records:
            if (
                record.child_issue != child_issue
                or record.parent_issue != parent_issue
                or record.stage_id != stage_id
            ):
                raise AgentLoopError(
                    "Human decision required: signed child-plan supersession at "
                    f"{record.comment_locator} names child #{record.child_issue} / parent "
                    f"#{record.parent_issue} / stage `{record.stage_id}`, but this issue is child "
                    f"#{child_issue} / parent #{parent_issue} / stage `{stage_id}`. Remove or "
                    "correct the record before rerunning."
                )
            by_digest.setdefault(record.digest, record)
    by_hash: dict[str, ChildPlanSupersession] = {}
    for record in by_digest.values():
        other = by_hash.get(record.superseded_plan_hash)
        if other is not None:
            raise AgentLoopError(
                "Human decision required: two distinct signed child-plan supersession records "
                f"name superseded plan {record.superseded_plan_hash} "
                f"({other.comment_locator} and {record.comment_locator}). Keep exactly one "
                "record per superseded plan hash before rerunning; records are never chosen "
                "by comment order."
            )
        by_hash[record.superseded_plan_hash] = record
    return tuple(by_digest.values())


def _rebind_record_payload(record: ChildPlanRebindRecord) -> dict[str, object]:
    return {
        "schema_version": 1,
        "child_issue": record.child_issue,
        "pr_number": record.pr_number,
        "superseded_plan_hash": record.superseded_plan_hash,
        "new_plan_hash": record.new_plan_hash,
        "plan_supersession_digest": record.plan_supersession_digest,
        "first_replan_round": record.first_replan_round,
        "approved_round": record.approved_round,
    }


def format_child_plan_rebind_section(
    record: ChildPlanRebindRecord, *, transaction_era: bool = False
) -> str:
    """Visible audit text plus the rebind audit record for the rebind comment.

    A transaction-era rebind (#827) reissues the PR contract in the same
    workflow transaction, so its visible text says so.
    """
    encoded = _encode_json_payload(_rebind_record_payload(record))
    pr_side = (
        "The PR contract is reissued in the same workflow transaction"
        if transaction_era
        else "No PR-side record changed"
    )
    return "\n".join(
        [
            f"Child plan rebind: PR #{record.pr_number} is rebound from approved plan "
            f"{record.superseded_plan_hash} to approved plan {record.new_plan_hash}.",
            f"Signed supersession digest: {record.plan_supersession_digest}",
            f"Re-plan rounds: {record.first_replan_round} through {record.approved_round}.",
            f"{pr_side}; reviewer approvals recorded before this comment "
            "do not count under the new plan.",
            f"<!-- AGENT_CHILD_PLAN_REBIND: {encoded} -->",
        ]
    )


def find_child_plan_rebind_records(
    comments: Sequence[object],
) -> tuple[ChildPlanRebindRecord, ...]:
    """Every decodable rebind audit record, tagged with its comment index."""
    found: list[ChildPlanRebindRecord] = []
    for index, comment in enumerate(comments):
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in CHILD_PLAN_REBIND_MARKER_RE.finditer(body):
            payload = _decode_json_payload(
                match.group("payload"), marker_name="AGENT_CHILD_PLAN_REBIND"
            )
            if set(payload) != _REBIND_RECORD_KEYS or payload.get("schema_version") != 1:
                raise AgentLoopError("Invalid AGENT_CHILD_PLAN_REBIND payload: unexpected keys.")
            for key in ("child_issue", "pr_number", "first_replan_round", "approved_round"):
                value = payload.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise AgentLoopError(
                        f"Invalid AGENT_CHILD_PLAN_REBIND payload: `{key}` must be a positive integer."
                    )
            for key in ("superseded_plan_hash", "new_plan_hash", "plan_supersession_digest"):
                value = payload.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise AgentLoopError(
                        f"Invalid AGENT_CHILD_PLAN_REBIND payload: `{key}` must be a non-empty string."
                    )
            found.append(
                ChildPlanRebindRecord(
                    child_issue=int(payload["child_issue"]),
                    pr_number=int(payload["pr_number"]),
                    superseded_plan_hash=str(payload["superseded_plan_hash"]),
                    new_plan_hash=str(payload["new_plan_hash"]),
                    plan_supersession_digest=str(payload["plan_supersession_digest"]),
                    first_replan_round=int(payload["first_replan_round"]),
                    approved_round=int(payload["approved_round"]),
                    comment_index=index,
                )
            )
    return tuple(found)


def authorized_replan_lineage(
    comments: Sequence[object],
    *,
    superseded_hash: str,
    digest: str,
    supersessions: Sequence[ChildPlanSupersession],
    through_round: int | None = None,
) -> AuthorizedReplanLineage | str:
    """The digest-bound re-plan lineage, or the reason it is not authorized.

    Single authority for resume, rebind eligibility, and rebind verification.
    ``through_round`` bounds the inspected history to one historical re-plan
    (a later re-plan of the replacement plan is a different lineage); when it
    is ``None`` the lineage must reach the latest plan coder round.
    """
    from .round_state import _extract_round_metadata_records

    matching = [
        record
        for record in supersessions
        if record.digest == digest and record.superseded_plan_hash == superseded_hash
    ]
    if len(matching) != 1:
        return (
            f"signed child-plan supersession {digest} for superseded plan {superseded_hash} "
            "is not discoverable on the child issue (exactly one signed record is required)"
        )
    signed = matching[0]
    coder_rounds: dict[int, object] = {}
    for record in _extract_round_metadata_records(comments, flow="plan"):
        if record.metadata.role != "coder":
            continue
        if through_round is not None and record.metadata.round_number > through_round:
            continue
        # The latest record for a round number is the authoritative plan round.
        coder_rounds[record.metadata.round_number] = record
    ordered = [coder_rounds[number] for number in sorted(coder_rounds)]
    bound = [
        record
        for record in ordered
        if record.metadata.plan_supersession_digest == digest
        and record.metadata.plan_supersession_superseded_hash == superseded_hash
    ]
    if not bound:
        return (
            f"no plan round carries signed supersession digest {digest} for superseded plan "
            f"{superseded_hash}"
        )
    first = bound[0]
    first_round = first.metadata.round_number
    if first.index <= signed.comment_index:
        return (
            f"plan round {first_round} carries the supersession digest but was posted before "
            f"the signed record at {signed.comment_locator}"
        )
    # Reviewer-only phase advances consume round numbers without a coder
    # round, so contiguity is judged on the coder-round chain: each bound
    # round must revise exactly the plan subject of the coder round before it.
    first_position = ordered.index(first)
    previous = ordered[first_position - 1] if first_position > 0 else None
    previous_plan = previous.metadata.canonical_plan if previous is not None else None
    if (
        previous is None
        or previous_plan is None
        or approved_plan_hash(previous_plan) != superseded_hash
    ):
        return (
            f"plan round {first_round} is the first digest-bound round, but the plan round "
            f"immediately before it is not superseded plan {superseded_hash}"
        )
    chain_subject = previous.metadata.subject
    round_numbers: list[int] = []
    for record in ordered[first_position:]:
        number = record.metadata.round_number
        if record.metadata.prior_plan_subject != chain_subject:
            return (
                f"the digest-bound re-plan lineage has a gap before plan round {number}: it "
                "does not revise the plan of the preceding plan round"
            )
        chain_subject = record.metadata.subject
        round_numbers.append(number)
        if (
            record.metadata.plan_supersession_digest != digest
            or record.metadata.plan_supersession_superseded_hash != superseded_hash
        ):
            return (
                f"plan round {number} follows the authorized re-plan but is "
                + (
                    "bound to a different supersession record"
                    if record.metadata.plan_supersession_digest is not None
                    else "not bound to the signed supersession digest"
                )
            )
    latest = ordered[-1]
    if through_round is not None and latest.metadata.round_number != through_round:
        return f"no digest-bound plan round {through_round} exists"
    latest_plan = latest.metadata.canonical_plan
    if latest_plan is None:
        return f"plan round {latest.metadata.round_number} has no canonical plan record"
    return AuthorizedReplanLineage(
        round_numbers=tuple(round_numbers),
        latest_plan_hash=approved_plan_hash(latest_plan),
    )


def reconcile_handoff_disposition(
    phase: PlanPhase,
    handoff: PhaseImplementationHandoffMetadata,
    stage_overrides: Sequence[ChildDispositionOverride],
) -> str:
    """The single handoff/phase reconciliation rule shared by every entry path.

    Equal dispositions (legacy absent = direct, phase ``None`` = compatible)
    are accepted.  A differing disposition is accepted only when the handoff's
    recorded override digest resolves to exactly one discoverable signed record
    whose parent issue, plan hash, stage, and disposition all equal the
    handoff's, and no other distinct record exists for that stage.  Every
    other mismatch fails closed with a human-repair request.  Direct readiness
    is never re-judged here and override metadata never supplies evidence.
    """
    handoff_disposition = handoff_effective_disposition(handoff)
    phase_disposition = getattr(phase, "execution_disposition", None)
    # Once dispatch is durable, the handoff is the route authority.  A
    # previously unbound handoff cannot acquire an override after the fact,
    # even when that record happens to request the same disposition.  An
    # override-bound handoff must continue to resolve to exactly its recorded
    # digest with no distinct record for the stage.  Checking this before the
    # equal-disposition fast path prevents comment order or a later edit from
    # silently changing the durable route.
    digest = handoff.override_digest
    if digest is None:
        if stage_overrides:
            record = stage_overrides[0]
            raise AgentLoopError(
                f"Human repair required: phase handoff for stage `{handoff.stage_id}` already "
                f"records disposition `{handoff_disposition}`, but the signed override at "
                f"{record.comment_locator} was not bound by that handoff. An override cannot be "
                "added after dispatch; remove the record or repair the handoff manually."
            )
        if phase_disposition is None or phase_disposition == handoff_disposition:
            return handoff_disposition
        raise AgentLoopError(
            f"Human repair required: phase handoff for stage `{handoff.stage_id}` records "
            f"disposition `{handoff_disposition}`, but the persisted approved phase declares "
            f"`{phase_disposition}` and the handoff carries no override digest. Restore the signed override "
            "record or repair the handoff; the orchestrator never re-resolves this route."
        )
    prefix = (
        f"Human repair required: phase handoff for stage `{handoff.stage_id}` records "
        f"disposition `{handoff_disposition}` with override digest {digest}"
    )
    matches = [record for record in stage_overrides if record.digest == digest]
    if not matches:
        raise AgentLoopError(
            f"{prefix}, but that digest is undiscoverable on the child or parent "
            "issue (the signed record was deleted, edited, or never posted). Restore the exact "
            "record or repair the handoff."
        )
    record = matches[0]
    if (
        record.parent_issue != handoff.parent_issue
        or record.plan_hash != handoff.plan_hash
        or record.stage_id != handoff.stage_id
    ):
        raise AgentLoopError(
            f"{prefix}, but the signed record names a different parent, plan "
            "hash, or stage than the handoff. Repair the handoff or the record."
        )
    if record.disposition != handoff_disposition:
        raise AgentLoopError(
            f"{prefix}, but the signed record declares "
            f"`{record.disposition}`, which differs from the recorded handoff disposition. "
            "Repair the handoff or the record."
        )
    others = sorted({item.digest for item in stage_overrides} - {digest})
    if others:
        raise AgentLoopError(
            f"{prefix}, but the applied override is superseded by a later distinct signed "
            f"record for the same stage ({', '.join(others)}). The orchestrator never chooses "
            "between conflicting records; remove the superseding record or repair the handoff."
        )
    return handoff_disposition


def post_decomposition_parent_summary(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    created: Sequence[CreatedPhaseIssue],
    topology_source: str = "model",
    retained_parent_scope: RetainedParentScope | None = None,
    final_integration_work: ExecutionAllocation | None = None,
    strategy: str | None = None,
    execution_strategy_contract_version: int | None = None,
    recommendation_digest: str | None = None,
    plan_subject: str | None = None,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=parent_issue,
        body=TrustedBody.canonical(
            format_decomposition_parent_summary(
                parent_issue=parent_issue,
                mode=mode,
                plan_hash=plan_hash,
                created=created,
                topology_source=topology_source,
                retained_parent_scope=retained_parent_scope,
                final_integration_work=final_integration_work,
                strategy=strategy,
                execution_strategy_contract_version=execution_strategy_contract_version,
                recommendation_digest=recommendation_digest,
                plan_subject=plan_subject,
            ),
            expected_tokens=("AGENT_PLAN_DECOMPOSITION",),
        ),
    )


def _encode_one_shot_impl_handoff_metadata(
    metadata: OneShotImplementationHandoffMetadata,
) -> str:
    payload: dict[str, object] = {
        "parent_issue": metadata.parent_issue,
        "plan_hash": metadata.plan_hash,
        "plan_subject": metadata.plan_subject,
        "mode": metadata.mode,
        "pr_number": metadata.pr_number,
        "pr_head_sha": metadata.pr_head_sha,
    }
    if metadata.topology_source == EXECUTION_TOPOLOGY_SOURCE:
        payload.update(
            {
                "strategy": metadata.strategy,
                "topology_source": metadata.topology_source,
                "execution_strategy_contract_version": metadata.execution_strategy_contract_version,
                "recommendation_digest": metadata.recommendation_digest,
            }
        )
    return _encode_json_payload(payload)


def _decode_one_shot_impl_handoff_metadata(encoded: str) -> OneShotImplementationHandoffMetadata:
    payload = _decode_json_payload(encoded, marker_name="AGENT_PLAN_ONE_SHOT_IMPL")
    try:
        pr_head_sha = payload.get("pr_head_sha")
        metadata = OneShotImplementationHandoffMetadata(
            parent_issue=int(payload["parent_issue"]),
            plan_hash=str(payload["plan_hash"]),
            plan_subject=str(payload.get("plan_subject") or ""),
            mode=str(payload["mode"]),
            pr_number=int(payload["pr_number"]),
            pr_head_sha=pr_head_sha if isinstance(pr_head_sha, str) else None,
            strategy=(str(payload["strategy"]) if payload.get("strategy") is not None else None),
            topology_source=(
                str(payload["topology_source"])
                if payload.get("topology_source") is not None else None
            ),
            execution_strategy_contract_version=(
                int(payload["execution_strategy_contract_version"])
                if payload.get("execution_strategy_contract_version") is not None else None
            ),
            recommendation_digest=(
                str(payload["recommendation_digest"])
                if payload.get("recommendation_digest") is not None else None
            ),
        )
        fresh_fields = {
            "strategy", "topology_source", "execution_strategy_contract_version",
            "recommendation_digest",
        }
        if metadata.topology_source == EXECUTION_TOPOLOGY_SOURCE:
            if (
                metadata.strategy != "one-shot"
                or metadata.execution_strategy_contract_version != EXECUTION_STRATEGY_CONTRACT_VERSION
                or not metadata.recommendation_digest
            ):
                raise AgentLoopError("Invalid AGENT_PLAN_ONE_SHOT_IMPL fresh identity.")
        elif any(key in payload for key in fresh_fields):
            raise AgentLoopError("Invalid AGENT_PLAN_ONE_SHOT_IMPL source identity.")
        return metadata
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentLoopError("Invalid AGENT_PLAN_ONE_SHOT_IMPL payload.") from exc


def find_existing_one_shot_impl_handoff(
    comments: Sequence[object],
    *,
    parent_issue: int,
    plan_hash: str,
    mode: str,
) -> OneShotImplementationHandoffMetadata | None:
    found: OneShotImplementationHandoffMetadata | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in ONE_SHOT_IMPL_HANDOFF_MARKER_RE.finditer(body):
            metadata = _decode_one_shot_impl_handoff_metadata(match.group("payload"))
            if (
                metadata.parent_issue == parent_issue
                and metadata.plan_hash == plan_hash
                and metadata.mode == mode
            ):
                found = metadata
    return found


def find_latest_one_shot_impl_handoff(
    comments: Sequence[object], *, parent_issue: int, mode: str
) -> OneShotImplementationHandoffMetadata | None:
    """Return the latest valid one-shot handoff regardless of plan hash.

    Plan-first recovery uses this to distinguish an open handoff for an older
    plan from an absent handoff; silently ignoring the former would create a
    duplicate implementation PR.
    """
    found: OneShotImplementationHandoffMetadata | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in ONE_SHOT_IMPL_HANDOFF_MARKER_RE.finditer(body):
            metadata = _decode_one_shot_impl_handoff_metadata(match.group("payload"))
            if metadata.parent_issue == parent_issue and metadata.mode == mode:
                found = metadata
    return found


def find_one_shot_impl_handoffs(
    comments: Sequence[object],
    *,
    parent_issue: int,
    mode: str,
) -> tuple[OneShotImplementationHandoffMetadata, ...]:
    """Return all one-shot handoffs so fresh recovery can detect ambiguity."""
    found: list[OneShotImplementationHandoffMetadata] = []
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in ONE_SHOT_IMPL_HANDOFF_MARKER_RE.finditer(body):
            metadata = _decode_one_shot_impl_handoff_metadata(match.group("payload"))
            if metadata.parent_issue == parent_issue and metadata.mode == mode:
                found.append(metadata)
    return tuple(found)


def format_one_shot_impl_handoff_comment(
    *,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    plan_subject: str,
    pr_number: int,
    pr_head_sha: str | None,
    strategy: str | None = None,
    topology_source: str | None = None,
    execution_strategy_contract_version: int | None = None,
    recommendation_digest: str | None = None,
) -> str:
    metadata = OneShotImplementationHandoffMetadata(
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        plan_subject=plan_subject,
        mode=mode,
        pr_number=pr_number,
        pr_head_sha=pr_head_sha,
        strategy=strategy,
        topology_source=topology_source,
        execution_strategy_contract_version=execution_strategy_contract_version,
        recommendation_digest=recommendation_digest,
    )
    lines = [
        f"Approved plan for issue #{parent_issue} handed off to PR #{pr_number} for one-shot implementation.",
        "",
        f"Mode: {mode}",
        f"Plan hash: {plan_hash}",
        f"Plan subject: {plan_subject}",
        "",
        "Parent reruns will resume the PR review loop for this PR instead of re-implementing.",
        "",
        f"<!-- AGENT_PLAN_ONE_SHOT_IMPL: {_encode_one_shot_impl_handoff_metadata(metadata)} -->",
        "-- coding-review-agent-loop",
    ]
    if topology_source == EXECUTION_TOPOLOGY_SOURCE:
        lines.insert(5, f"Canonical strategy: {strategy} (source: {topology_source})")
        lines.insert(6, f"Recommendation digest: {recommendation_digest}")
    return "\n".join(lines)


def post_one_shot_impl_handoff_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    plan_subject: str,
    pr_number: int,
    pr_head_sha: str | None,
    strategy: str | None = None,
    topology_source: str | None = None,
    execution_strategy_contract_version: int | None = None,
    recommendation_digest: str | None = None,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=parent_issue,
        body=TrustedBody.canonical(
            format_one_shot_impl_handoff_comment(
                parent_issue=parent_issue,
                mode=mode,
                plan_hash=plan_hash,
                plan_subject=plan_subject,
                pr_number=pr_number,
                pr_head_sha=pr_head_sha,
                strategy=strategy,
                topology_source=topology_source,
                execution_strategy_contract_version=execution_strategy_contract_version,
                recommendation_digest=recommendation_digest,
            ),
            expected_tokens=("AGENT_PLAN_ONE_SHOT_IMPL",),
        ),
    )
