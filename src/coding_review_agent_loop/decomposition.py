"""Approved-plan decomposition parsing and publishing helpers."""

from __future__ import annotations

import base64
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
from .protocol_markers import TrustedBody, sanitize_historical_text
from .protocol import (
    ArchitectureImpact,
    ChildStage,
    EXECUTION_AUTOMATION_CLASSES,
    EXECUTION_STRATEGY_CONTRACT_VERSION,
    EXECUTION_TOPOLOGY_SOURCE,
    ExecutionAllocation,
    ExecutionChildStage,
    ExecutionCouplingConstraint,
    ExecutionScopeItem,
    ExecutionStrategyRecommendation,
    parse_architecture_impact,
    parse_execution_recommendation_payload,
    sanitize_architecture_impact,
)
from .round_transport import MAX_GITHUB_BODY_CHARS

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
    }


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
    try:
        if encoded.startswith("v1_"):
            raw = zlib.decompress(
                base64.urlsafe_b64decode(encoded[3:].encode("ascii"))
            )
            payload = json.loads(raw.decode("utf-8"))
        else:
            # Read the original uncompressed representation for checkpoints
            # already posted before the compact payload format was introduced.
            payload = _decode_json_payload(
                encoded, marker_name="AGENT_PLAN_TOPOLOGY_CHECKPOINT"
            )
    except (ValueError, json.JSONDecodeError, zlib.error) as exc:
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.") from exc
    if not isinstance(payload, dict):
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
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


def format_topology_checkpoint(checkpoint: TopologyCheckpoint) -> str:
    raw = json.dumps(
        _checkpoint_payload(checkpoint),
        separators=(",", ":"),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    encoded = "v1_" + base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii")
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
) -> str:
    parent_url = f"https://github.com/{repo}/issues/{parent_issue}"
    if phase.automation == "agent-pr":
        execution = (
            "Run `agent-loop issue <this issue number>` to implement this phase in its own PR. "
            "Keep the PR scoped to this phase."
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
    body = "\n".join(
        [
            f"Child phase issue for parent #{parent_issue}: {parent_url}",
            "",
            "## Approved parent-plan excerpt for this phase",
            sanitize_historical_text(phase.parent_context),
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
        )
    return body


def _fresh_phase_content_matches(
    candidate: FoundIssue,
    *,
    parent_issue: int,
    phase: PlanPhase,
) -> bool:
    """Check the reviewed content around a fresh phase identity marker."""
    if candidate.title != _phase_issue_title(parent_issue, phase.position or 0, phase):
        return False
    body = candidate.body
    if not isinstance(body, str):
        return False
    fragments = [
        f"Child phase issue for parent #{parent_issue}:",
        "## Approved parent-plan excerpt for this phase",
        sanitize_historical_text(phase.parent_context),
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
    ]
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
                        )
                    if not metadata_matches:
                        raise AgentLoopError(
                            f"Invalid decomposition recovery identity metadata for phase {index}."
                        )
                    if fresh and not _fresh_phase_content_matches(
                        candidate,
                        parent_issue=parent_issue,
                        phase=expected_phase,
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


def _encode_json_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_json_payload(encoded: str, *, marker_name: str) -> dict[str, object]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
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
            if decision.parent_issue == parent_issue and decision.plan_hash == plan_hash:
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
    return _encode_json_payload(payload)


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
        )
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
        )
        fresh_fields = {
            "strategy", "topology_source", "execution_strategy_contract_version",
            "recommendation_digest", "stage_id", "plan_subject",
        }
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
    plan_hash: str,
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
                and metadata.plan_hash == plan_hash
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
    summary = find_existing_decomposition(
        comments,
        parent_issue=parent_issue,
        plan_hash=plan_hash,
    )
    if summary is not None and summary.topology_source != EXECUTION_TOPOLOGY_SOURCE:
        raise AgentLoopError(
            "Fresh execution topology conflicts with an existing legacy decomposition summary; "
            "repair the historical mode/source identity before rerunning."
        )
    for mode in ("decompose-only", "implement-by-phase"):
        checkpoint = find_existing_topology_checkpoint(
            comments,
            parent_issue=parent_issue,
            plan_hash=plan_hash,
            mode=mode,
        )
        if checkpoint is not None and checkpoint.topology_source != EXECUTION_TOPOLOGY_SOURCE:
            raise AgentLoopError(
                "Fresh execution topology conflicts with an existing legacy topology checkpoint; "
                "repair the historical mode/source identity before rerunning."
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
) -> str:
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
) -> str:
    if created.issue_number is None:
        raise AgentLoopError(
            "Cannot record decomposed phase implementation handoff because the child issue number is unavailable."
        )
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
    )
    child = created.issue_url or f"#{created.issue_number}"
    lines = [
        f"Approved plan implementation for issue #{parent_issue} handed off to phase {phase_index}: {child}.",
        "",
        f"Mode: {mode}",
        f"Phase: {created.phase.title}",
        f"Automation: {created.phase.automation}",
        "",
        "Parent reruns will not automatically re-run this child implementation. "
        f"Resume directly with `agent-loop issue {created.issue_number}`.",
        "",
        f"<!-- AGENT_PLAN_PHASE_IMPLEMENTATION: {_encode_phase_implementation_handoff_metadata(metadata)} -->",
        "-- coding-review-agent-loop",
    ]
    if topology_source == EXECUTION_TOPOLOGY_SOURCE:
        lines.insert(5, f"Canonical strategy: {strategy} (source: {topology_source})")
        lines.insert(6, f"Stable stage ID: {getattr(created.phase, 'stage_id', None)}")
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
            ),
            expected_tokens=("AGENT_PLAN_PHASE_IMPLEMENTATION",),
        ),
    )


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
