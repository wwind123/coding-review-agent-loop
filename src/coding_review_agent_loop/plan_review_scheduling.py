"""Pure scheduling decisions for staged issue plan review (#904, from #841).

This is the planning counterpart of :mod:`review_scheduling`.  It contains no
GitHub, agent, orchestrator, or persistence calls so every decision can be unit
tested and replayed from the small amount of planning scheduler metadata the
orchestrator persists.

The planning flow has no diff, so the PR module's broad-path rules and exact fix
scopes have no planning counterpart.  Instead the decision is bound to one
canonical *exact-plan candidate key* and to the authenticated cross-cutting
contract identities of the revision that produced it.

At the end of stage 1 nothing in the package imports this module: it is the
reviewed decision interface that the plan-first loop consumes in stage 2.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

from .errors import AgentLoopError
from .review_scheduling import (
    POST_PANEL_PREFIX,
    STRICT_PRE_PANEL_PREFIX,
    ReviewObligation,
    ReviewPolicyCapabilities,
)

# Planning has no ``selective-intermediate`` counterpart: there is no diff to
# scope a selective intermediate round against.
PLAN_REVIEW_POLICIES = frozenset({"all-reviewers", "primary-then-panel"})

# Deliberately the same phase vocabulary as the PR flow so logs, comments, and
# resume records read alike across both flows.
PLAN_SCHEDULER_PHASES = frozenset(
    {
        "full-board",
        "primary",
        "secondary-audit",
        "remediation",
        "final-secondary-sweep",
    }
)

PLAN_FORCE_FULL_SOURCES = frozenset({"operator", "automatic"})

# The generation of plan state that can form a complete candidate key.  A legacy
# unversioned plan has no execution-strategy or risk-matrix identity, so staged
# planning is refused rather than degraded.
PLAN_KEY_GENERATION = 1

OPERATOR_PLAN_FORCE_FULL_REASON = (
    "operator force-full: the complete plan board is authorized by "
    "--plan-review-force-full"
)

PLAN_PRE_PANEL_SAFETY_MESSAGE = (
    "plan pre-panel safety cannot be established: {detail} but no qualified panel "
    "opening exists; no reviewer and no planner turn were invoked. Rerun with "
    "--plan-review-force-full to authorize the complete plan board."
)

PLAN_UNDECODABLE_HISTORY_MESSAGE = (
    "plan pre-panel safety cannot be established: the planning round-metadata record "
    "set could not be extracted ({error}), so panel state, finding ownership, and "
    "exact-plan approvals are unknowable; no reviewer and no planner turn were "
    "invoked. Restore the missing planning round-metadata records (or remove the "
    "incomplete record) and rerun. --plan-review-force-full cannot authorize the "
    "complete plan board over an unreadable record set, because approval, ownership, "
    "and qualification accounting all depend on that history."
)

# The four disjoint classes of degraded planning history.  Exactly one outcome
# each: ``absent``, ``invalid``, and ``contradictory-key`` always fall back and
# continue; only ``transport-failure`` stops the run.
PLAN_HISTORY_INTACT = "intact"
PLAN_HISTORY_ABSENT = "absent"
PLAN_HISTORY_INVALID = "invalid"
PLAN_HISTORY_CONTRADICTORY_KEY = "contradictory-key"
PLAN_HISTORY_TRANSPORT_FAILURE = "transport-failure"

PLAN_HISTORY_CLASSES = frozenset(
    {
        PLAN_HISTORY_INTACT,
        PLAN_HISTORY_ABSENT,
        PLAN_HISTORY_INVALID,
        PLAN_HISTORY_CONTRADICTORY_KEY,
        PLAN_HISTORY_TRANSPORT_FAILURE,
    }
)

# Classes that always continue under a conservative fallback.
PLAN_DEGRADED_CONTINUE_CLASSES = frozenset(
    {PLAN_HISTORY_ABSENT, PLAN_HISTORY_INVALID, PLAN_HISTORY_CONTRADICTORY_KEY}
)

_PLAN_HISTORY_FALLBACK_REASONS = {
    PLAN_HISTORY_ABSENT: (
        "planning scheduler metadata is absent from the round history, so phase, "
        "ownership, and exact-plan approvals cannot be replayed"
    ),
    PLAN_HISTORY_INVALID: (
        "planning scheduler metadata decoded as invalid, so phase, ownership, and "
        "exact-plan approvals cannot be replayed"
    ),
    PLAN_HISTORY_CONTRADICTORY_KEY: (
        "persisted planning candidate-key components contradict the canonical plan, "
        "so no stored approval or panel opening can be trusted"
    ),
}

# Patch fields a narrow planning remediation may touch.  Everything else is a
# cross-cutting contract and latches the complete board.
NARROW_PLAN_PATCH_FIELDS = frozenset(
    {"summary", "plan_steps", "deferred_work", "plan_actions", "external_dependencies"}
)

# Risk-matrix row operations a narrow planning remediation may carry.
NARROW_PLAN_MATRIX_OPERATIONS = frozenset({"matrix_add", "matrix_edit"})


class PlanPrePanelSafetyError(AgentLoopError):
    """Staged planning cannot stay primary-only without guessing.

    Raised instead of silently spending the secondary panel early, or of
    continuing over a record set that cannot be read at all.
    """


def plan_pre_panel_safety_message(detail: str) -> str:
    return PLAN_PRE_PANEL_SAFETY_MESSAGE.format(detail=detail)


def plan_undecodable_history_message(error: object) -> str:
    detail = " ".join(str(error).split())
    if len(detail) > 300:
        detail = detail[:297].rstrip() + "..."
    return PLAN_UNDECODABLE_HISTORY_MESSAGE.format(
        error=detail or "unknown extraction error"
    )


def plan_policy_capabilities(policy: str) -> ReviewPolicyCapabilities:
    """Named planning scheduler capabilities, mirroring the PR vocabulary."""
    if policy == "all-reviewers":
        return ReviewPolicyCapabilities(
            scheduler_enabled=False,
            owner_scoped_reconciliation=False,
            selective_pausing=False,
            phase_aware=False,
            counts_avoided_calls=False,
        )
    if policy == "primary-then-panel":
        return ReviewPolicyCapabilities(
            scheduler_enabled=True,
            owner_scoped_reconciliation=True,
            selective_pausing=True,
            phase_aware=True,
            counts_avoided_calls=True,
            requires_primary=True,
            recovery_latches_force_full=True,
        )
    raise AgentLoopError(f"Unsupported plan review policy: {policy!r}.")


def classify_plan_history(
    scheduler_metadata_status: str | None,
    *,
    key_contradiction: bool = False,
    transport_failure: bool = False,
) -> str:
    """Assign one degraded-history class to authenticated planning history.

    The four classes are disjoint and ordered by authority: a record set that
    cannot be extracted at all outranks any per-record observation, and a
    contradictory key is only observable on a record that decoded cleanly.
    """
    if transport_failure:
        return PLAN_HISTORY_TRANSPORT_FAILURE
    if scheduler_metadata_status is None:
        status = "absent"
    elif not isinstance(scheduler_metadata_status, str):
        raise AgentLoopError("Planning scheduler metadata status must be a string.")
    else:
        status = scheduler_metadata_status
    if status == "absent":
        return PLAN_HISTORY_ABSENT
    if status == "invalid":
        return PLAN_HISTORY_INVALID
    if status != "valid":
        raise AgentLoopError(
            f"Unsupported planning scheduler metadata status: {status!r}."
        )
    if key_contradiction:
        return PLAN_HISTORY_CONTRADICTORY_KEY
    return PLAN_HISTORY_INTACT


def plan_history_fallback_reason(history_class: str) -> str | None:
    """The audit reason for a degraded class, or ``None`` for intact history."""
    if history_class not in PLAN_HISTORY_CLASSES:
        raise AgentLoopError(f"Unsupported planning history class: {history_class!r}.")
    return _PLAN_HISTORY_FALLBACK_REASONS.get(history_class)


def plan_history_continues(history_class: str) -> bool:
    """True when the class is a conservative fallback rather than a stop."""
    if history_class not in PLAN_HISTORY_CLASSES:
        raise AgentLoopError(f"Unsupported planning history class: {history_class!r}.")
    return history_class != PLAN_HISTORY_TRANSPORT_FAILURE


def surfaced_requirement_id_digest(requirement_ids: Sequence[str] | None) -> str:
    """Deterministic digest of the surfaced planning-requirement ID set.

    Sorted and deduplicated so the digest depends on the set in force for the
    round and not on the order the orchestrator happened to collect it in.
    """
    if isinstance(requirement_ids, (str, bytes)):
        raise AgentLoopError(
            "Surfaced planning-requirement IDs must be a sequence of strings."
        )
    values = () if requirement_ids is None else tuple(requirement_ids)
    cleaned: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise AgentLoopError(
                "Surfaced planning-requirement IDs must be non-empty strings."
            )
        cleaned.add(value.strip())
    payload = json.dumps(sorted(cleaned), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PlanCandidateKey:
    """The one canonical exact-plan candidate key.

    Used identically by scheduler records, carried approvals, panel-opening
    evidence, and resume.  ``complete`` is the generation-1 gate: a key missing
    any component, or carrying a non-1 execution-strategy contract version,
    cannot supply an approval or a panel opening.
    """

    subject: str | None
    aggregate_plan_identity: str | None
    execution_strategy_identity: str | None
    risk_test_matrix_identity: str | None
    surfaced_requirement_id_digest: str | None
    execution_strategy_contract_version: int | None = PLAN_KEY_GENERATION

    def __post_init__(self) -> None:
        for name in (
            "subject",
            "aggregate_plan_identity",
            "execution_strategy_identity",
            "risk_test_matrix_identity",
            "surfaced_requirement_id_digest",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise AgentLoopError(
                    f"Planning candidate key component {name!r} must be a non-empty string."
                )
            object.__setattr__(self, name, value.strip())
        version = self.execution_strategy_contract_version
        if version is not None and (
            isinstance(version, bool) or not isinstance(version, int)
        ):
            raise AgentLoopError(
                "Planning candidate key execution_strategy_contract_version must be an integer."
            )

    @property
    def components(self) -> tuple[object, ...]:
        """The ordered tuple that defines key identity."""
        return (
            self.subject,
            self.aggregate_plan_identity,
            self.execution_strategy_identity,
            self.risk_test_matrix_identity,
            self.surfaced_requirement_id_digest,
        )

    @property
    def missing_components(self) -> tuple[str, ...]:
        names = (
            "subject",
            "aggregate_plan_identity",
            "execution_strategy_identity",
            "risk_test_matrix_identity",
            "surfaced_requirement_id_digest",
        )
        return tuple(name for name in names if getattr(self, name) is None)

    @property
    def complete(self) -> bool:
        return (
            not self.missing_components
            and self.execution_strategy_contract_version == PLAN_KEY_GENERATION
        )

    def incompleteness_reason(self) -> str | None:
        missing = self.missing_components
        if missing:
            return (
                "the candidate plan key is missing component(s) "
                f"{', '.join(missing)}"
            )
        if self.execution_strategy_contract_version != PLAN_KEY_GENERATION:
            return (
                "the candidate plan key carries execution-strategy contract version "
                f"{self.execution_strategy_contract_version!r}, not generation "
                f"{PLAN_KEY_GENERATION}"
            )
        return None

    def matches(self, other: "PlanCandidateKey | None") -> bool:
        """Component-for-component equality between two complete keys."""
        if other is None:
            return False
        if not self.complete or not other.complete:
            return False
        return self.components == other.components

    def as_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "aggregate_plan_identity": self.aggregate_plan_identity,
            "execution_strategy_identity": self.execution_strategy_identity,
            "risk_test_matrix_identity": self.risk_test_matrix_identity,
            "surfaced_requirement_id_digest": self.surfaced_requirement_id_digest,
            "execution_strategy_contract_version": (
                self.execution_strategy_contract_version
            ),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "PlanCandidateKey":
        if not isinstance(value, dict):
            raise AgentLoopError("Malformed planning candidate key.")
        version = value.get("execution_strategy_contract_version")
        if version is not None and (
            isinstance(version, bool) or not isinstance(version, int)
        ):
            raise AgentLoopError(
                "Malformed planning candidate key execution-strategy contract version."
            )
        components: dict[str, object] = {}
        for name in (
            "subject",
            "aggregate_plan_identity",
            "execution_strategy_identity",
            "risk_test_matrix_identity",
            "surfaced_requirement_id_digest",
        ):
            raw = value.get(name)
            if raw is not None and not isinstance(raw, str):
                raise AgentLoopError(
                    f"Malformed planning candidate key component {name!r}."
                )
            components[name] = raw
        return cls(
            subject=components["subject"],
            aggregate_plan_identity=components["aggregate_plan_identity"],
            execution_strategy_identity=components["execution_strategy_identity"],
            risk_test_matrix_identity=components["risk_test_matrix_identity"],
            surfaced_requirement_id_digest=components[
                "surfaced_requirement_id_digest"
            ],
            execution_strategy_contract_version=version,
        )


@dataclass(frozen=True)
class PlanCrossCuttingContracts:
    """The authenticated cross-cutting contracts a narrow revision preserves.

    These are identities and digests, never prose: the classifier compares them
    for equality and never attempts to attribute a patch operation to a finding
    or an owner, because the authenticated patch carries no such linkage.
    """

    execution_recommendation_identity: str | None = None
    human_requirement_disposition_digest: str | None = None
    additional_closing_issue_ids: tuple[str, ...] = ()
    architecture_impact_status: str | None = None

    def __post_init__(self) -> None:
        ids = tuple(self.additional_closing_issue_ids)
        if any(not isinstance(item, str) or not item for item in ids):
            raise AgentLoopError(
                "additional_closing_issue_ids entries must be non-empty strings."
            )
        object.__setattr__(self, "additional_closing_issue_ids", tuple(sorted(set(ids))))

    def differences(self, other: "PlanCrossCuttingContracts") -> tuple[str, ...]:
        names = (
            "execution_recommendation_identity",
            "human_requirement_disposition_digest",
            "additional_closing_issue_ids",
            "architecture_impact_status",
        )
        return tuple(
            name for name in names if getattr(self, name) != getattr(other, name)
        )


@dataclass(frozen=True)
class PlanRevisionDescriptor:
    """Authenticated facts about the revision that produced the current key."""

    response_form: str | None = None
    semantic_patch_contract_version: int | None = None
    base_state_identity: str | None = None
    sidecar_bound: bool = True
    operation_fields: tuple[str, ...] = ()
    matrix_operations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("operation_fields", "matrix_operations"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, str) or not item for item in values):
                raise AgentLoopError(
                    f"Planning revision {name} entries must be non-empty strings."
                )
            object.__setattr__(self, name, values)

    @property
    def authenticated_semantic_patch(self) -> bool:
        return (
            self.response_form == "semantic-patch-v1"
            and self.semantic_patch_contract_version == 1
            and self.sidecar_bound
            and isinstance(self.base_state_identity, str)
            and bool(self.base_state_identity)
        )


@dataclass(frozen=True)
class PlanTransitionClassification:
    """``recheck``, ``narrow``, or ``broad`` for one planning transition."""

    kind: str
    reason: str

    def __post_init__(self) -> None:
        if self.kind not in {"recheck", "narrow", "broad"}:
            raise AgentLoopError(
                f"Unsupported plan transition classification: {self.kind!r}."
            )

    @property
    def recheck(self) -> bool:
        return self.kind == "recheck"

    @property
    def narrow(self) -> bool:
        return self.kind == "narrow"

    @property
    def broad(self) -> bool:
        return self.kind == "broad"

    @property
    def owner_scoped(self) -> bool:
        """Narrow and recheck transitions both keep owner-scoped remediation."""
        return self.kind in {"recheck", "narrow"}


@dataclass(frozen=True)
class PlanReviewSchedulingContract:
    """Immutable in-flight planning scheduler contract.

    Carries no PR-only broad-rule or scope-digest state, because the planning
    flow has no diff to classify.
    """

    required_reviewers: tuple[str, ...]
    policy: str = "all-reviewers"
    primary_reviewer: str | None = None

    def __post_init__(self) -> None:
        reviewers = tuple(self.required_reviewers)
        if (
            not reviewers
            or any(not isinstance(item, str) or not item for item in reviewers)
            or len(set(reviewers)) != len(reviewers)
        ):
            raise AgentLoopError(
                "Plan scheduler contract requires a unique reviewer set."
            )
        capabilities = plan_policy_capabilities(self.policy)
        primary = self.primary_reviewer
        if primary is not None and (not isinstance(primary, str) or not primary):
            raise AgentLoopError(
                "Plan scheduler primary reviewer must be a non-empty string."
            )
        if capabilities.requires_primary:
            if len(reviewers) < 2:
                raise AgentLoopError(
                    "primary-then-panel plan review requires one primary and at least "
                    "one secondary reviewer."
                )
            if primary not in reviewers:
                raise AgentLoopError(
                    "primary-then-panel plan review primary reviewer must be a member "
                    "of the reviewer board."
                )
        elif primary is not None:
            raise AgentLoopError(
                "A primary plan reviewer may only be configured with "
                "primary-then-panel plan review."
            )
        object.__setattr__(self, "required_reviewers", reviewers)
        object.__setattr__(self, "primary_reviewer", primary)

    @property
    def secondary_reviewers(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.required_reviewers if name != self.primary_reviewer
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "required_reviewers": list(self.required_reviewers),
            "policy": self.policy,
            "primary_reviewer": self.primary_reviewer,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "PlanReviewSchedulingContract":
        if not isinstance(value, dict):
            raise AgentLoopError("Malformed plan scheduler contract.")
        reviewers = value.get("required_reviewers")
        policy = value.get("policy")
        primary = value.get("primary_reviewer")
        if (
            not isinstance(reviewers, list)
            or any(not isinstance(item, str) or not item for item in reviewers)
            or not isinstance(policy, str)
            or (primary is not None and not isinstance(primary, str))
        ):
            raise AgentLoopError("Incomplete plan scheduler contract.")
        return cls(
            required_reviewers=tuple(reviewers),
            policy=policy,
            primary_reviewer=primary,
        )


@dataclass(frozen=True)
class PlanSchedulerSnapshot:
    """Everything the planning decision depends on, reconstructed by the caller."""

    contract: PlanReviewSchedulingContract
    previous_key: PlanCandidateKey | None = None
    current_key: PlanCandidateKey | None = None
    obligations: tuple[ReviewObligation, ...] = ()
    # Automatic full-board request.  Honored only after a qualified panel
    # opening; before one it degrades to a strict primary-only full-context turn.
    force_full: bool = False
    force_full_source: str | None = None
    # Explicit operator authorization (``--plan-review-force-full`` or a
    # persisted operator-sourced latch).  It always selects the complete board,
    # except over an unreadable record set.
    operator_force_full: bool = False
    # True only when comment-ordered history holds a qualified panel opening.
    panel_evidence: bool = False
    phase: str | None = None
    fallback_reasons: tuple[str, ...] = ()
    degraded_history_class: str = PLAN_HISTORY_INTACT
    # Blocking secondary plan reviews recorded before any qualified opening.
    # They are unqualified artifacts: never approvals, never ownership.
    premature_secondary_reviews: tuple[str, ...] = ()
    transport_error: object | None = None

    def __post_init__(self) -> None:
        source = self.force_full_source
        if source is not None and source not in PLAN_FORCE_FULL_SOURCES:
            raise AgentLoopError(f"Unsupported plan force-full source: {source!r}.")
        if self.degraded_history_class not in PLAN_HISTORY_CLASSES:
            raise AgentLoopError(
                f"Unsupported planning history class: {self.degraded_history_class!r}."
            )
        if self.phase is not None and self.phase not in PLAN_SCHEDULER_PHASES:
            raise AgentLoopError(
                f"Unsupported plan scheduler phase checkpoint: {self.phase!r}."
            )
        object.__setattr__(
            self,
            "premature_secondary_reviews",
            tuple(sorted(set(self.premature_secondary_reviews))),
        )
        object.__setattr__(self, "obligations", tuple(self.obligations))
        object.__setattr__(self, "fallback_reasons", tuple(self.fallback_reasons))

    @property
    def active_obligations(self) -> tuple[ReviewObligation, ...]:
        return tuple(item for item in self.obligations if item.active)

    @property
    def pending_owners(self) -> frozenset[str]:
        return frozenset(
            owner
            for obligation in self.active_obligations
            for owner in obligation.pending_owners
        )


@dataclass(frozen=True)
class PlanSchedulingDecision:
    selected_reviewers: tuple[str, ...]
    paused_reviewers: tuple[tuple[str, str], ...]
    reason: str
    classification: PlanTransitionClassification
    phase: str = "full-board"
    primary_reviewer: str | None = None
    active_owners: tuple[str, ...] = ()
    calls_avoided: int = 0
    final_sweep: bool = False
    degraded_history_class: str = PLAN_HISTORY_INTACT
    # True when the decision raises the durable ``automatic`` post-panel latch.
    latches_force_full: bool = False
    # True when this decision itself records a qualified panel opening.
    records_panel_opening: bool = False


def classify_plan_transition(
    previous_key: PlanCandidateKey | None,
    current_key: PlanCandidateKey | None,
    revision: PlanRevisionDescriptor | None = None,
    *,
    previous_contracts: PlanCrossCuttingContracts | None = None,
    current_contracts: PlanCrossCuttingContracts | None = None,
    ledger_reconstructible: bool = True,
) -> PlanTransitionClassification:
    """Classify one planning transition from authenticated data only.

    ``recheck`` when the candidate key is unchanged, ``narrow`` for an
    authenticated ``semantic-patch-v1`` revision bound to the immediately
    preceding base identity that touches only plan prose, plan steps, deferred
    allocations, and risk-matrix row add/edit operations while every
    cross-cutting contract is unchanged, and ``broad`` for everything else.
    """
    if not ledger_reconstructible:
        return PlanTransitionClassification(
            "broad", "the active plan finding ledger is not reconstructible"
        )
    if current_key is None:
        return PlanTransitionClassification(
            "broad", "no candidate plan key is available for this round"
        )
    incomplete = current_key.incompleteness_reason()
    if incomplete is not None:
        return PlanTransitionClassification("broad", incomplete)
    if previous_key is None:
        return PlanTransitionClassification(
            "broad", "no previous candidate plan key is available for comparison"
        )
    previous_incomplete = previous_key.incompleteness_reason()
    if previous_incomplete is not None:
        return PlanTransitionClassification(
            "broad", f"the previous candidate plan key is unusable: {previous_incomplete}"
        )
    if current_key.matches(previous_key):
        return PlanTransitionClassification(
            "recheck", "the candidate plan key is unchanged"
        )
    if revision is None:
        return PlanTransitionClassification(
            "broad", "the candidate plan key changed with no authenticated revision"
        )
    if not revision.authenticated_semantic_patch:
        return PlanTransitionClassification(
            "broad",
            "the revision is not an authenticated semantic-patch-v1 with a bound sidecar",
        )
    if revision.base_state_identity != previous_key.aggregate_plan_identity:
        return PlanTransitionClassification(
            "broad",
            "the revision patch is not bound to the immediately preceding plan identity",
        )
    if previous_contracts is None or current_contracts is None:
        return PlanTransitionClassification(
            "broad", "the cross-cutting plan contract identities are unavailable"
        )
    changed = previous_contracts.differences(current_contracts)
    if changed:
        return PlanTransitionClassification(
            "broad", f"cross-cutting plan contract(s) changed: {', '.join(changed)}"
        )
    outside = tuple(
        sorted(
            {
                field_name
                for field_name in revision.operation_fields
                if field_name not in NARROW_PLAN_PATCH_FIELDS
            }
        )
    )
    if outside:
        return PlanTransitionClassification(
            "broad",
            f"the revision replaces field(s) outside narrow remediation: {', '.join(outside)}",
        )
    matrix_outside = tuple(
        sorted(
            {
                operation
                for operation in revision.matrix_operations
                if operation not in NARROW_PLAN_MATRIX_OPERATIONS
            }
        )
    )
    if matrix_outside:
        return PlanTransitionClassification(
            "broad",
            "the revision carries risk-matrix operation(s) beyond row add/edit: "
            f"{', '.join(matrix_outside)}",
        )
    return PlanTransitionClassification(
        "narrow",
        "the revision edits plan prose, plan steps, deferred allocations, and "
        "risk-matrix rows while every cross-cutting contract is unchanged",
    )


def _degraded_fallback(snapshot: PlanSchedulerSnapshot) -> tuple[bool, str | None]:
    """Whether degraded history forces a fallback, and its audit reason."""
    history_class = snapshot.degraded_history_class
    if history_class not in PLAN_DEGRADED_CONTINUE_CLASSES:
        return False, None
    return True, plan_history_fallback_reason(history_class)


def select_plan_reviewers(
    snapshot: PlanSchedulerSnapshot,
    classification: PlanTransitionClassification,
    *,
    qualifying_approvals: Sequence[str] = (),
    unavailable_reviewers: Sequence[str] = (),
    phase: str | None = None,
) -> PlanSchedulingDecision:
    """Choose the plan reviewer board for one round.

    Unavailable reviewers stay required: nothing here waives an approval.  A
    transport extraction failure is the only history class that stops the run,
    and the operator override cannot recover it.
    """
    contract = snapshot.contract
    required = contract.required_reviewers
    unavailable = set(unavailable_reviewers)
    available_required = tuple(name for name in required if name not in unavailable)
    approvals = set(qualifying_approvals)
    capabilities = plan_policy_capabilities(contract.policy)
    primary = contract.primary_reviewer
    required_names = set(required)
    history_class = snapshot.degraded_history_class

    if history_class == PLAN_HISTORY_TRANSPORT_FAILURE:
        # Checked before the operator override on purpose: approval, ownership,
        # and qualification accounting all depend on a readable record set.
        raise PlanPrePanelSafetyError(
            plan_undecodable_history_message(
                snapshot.transport_error
                if snapshot.transport_error is not None
                else "the planning round-metadata record set could not be extracted"
            )
        )

    checkpoint_phase = phase if phase is not None else snapshot.phase
    if checkpoint_phase is not None and checkpoint_phase not in PLAN_SCHEDULER_PHASES:
        raise AgentLoopError(
            f"Unsupported plan scheduler phase checkpoint: {checkpoint_phase!r}."
        )

    degraded, degraded_reason = _degraded_fallback(snapshot)
    fallback_reasons = tuple(
        dict.fromkeys(
            tuple(snapshot.fallback_reasons)
            + ((degraded_reason,) if degraded_reason else ())
        )
    )
    fallback_detail = "; ".join(fallback_reasons)
    force_full = snapshot.force_full or degraded

    latches_force_full = False
    records_panel_opening = False
    active_obligations = snapshot.active_obligations
    pending_owner_set = set(snapshot.pending_owners)

    if contract.policy == "all-reviewers":
        selected = available_required
        selected_phase = "full-board"
        reason = "compatibility policy"
    elif snapshot.operator_force_full:
        selected = available_required
        selected_phase = "full-board"
        records_panel_opening = True
        reason = OPERATOR_PLAN_FORCE_FULL_REASON
        superseded = tuple(
            name for name in snapshot.premature_secondary_reviews if name in required_names
        )
        if superseded:
            reason += (
                "; superseded premature secondary plan review(s) from "
                f"{', '.join(superseded)} are excluded from approval and ownership "
                "accounting and supplied only as non-authoritative context"
            )
    else:
        panel_opened = snapshot.panel_evidence
        primary_approved = primary is not None and primary in approvals
        if degraded and not panel_opened:
            # Degraded history cannot establish an exact-plan approval, so it can
            # never open the panel; it degrades to the strict primary-only turn.
            primary_approved = False
        if not panel_opened:
            secondaries = required_names - {primary}
            premature = tuple(
                name for name in snapshot.premature_secondary_reviews if name in secondaries
            )
            if premature:
                raise PlanPrePanelSafetyError(
                    plan_pre_panel_safety_message(
                        "an interrupted round holds a premature blocking plan review "
                        f"from configured secondary reviewer(s) {', '.join(premature)}"
                    )
                )
            secondary_pending = pending_owner_set & secondaries
            if secondary_pending:
                # Secondaries are never plan-finding owners before their first
                # legitimate panel invocation; do not guess ownership and do not
                # spend the panel silently.
                item_ids = sorted(
                    obligation.item_id
                    for obligation in active_obligations
                    if set(obligation.pending_owners) & secondary_pending
                )
                raise PlanPrePanelSafetyError(
                    plan_pre_panel_safety_message(
                        f"plan finding(s) {', '.join(item_ids)} are pending on "
                        "configured secondary reviewer(s) "
                        f"{', '.join(sorted(secondary_pending))}"
                    )
                )
            if not primary_approved:
                selected = (primary,) if primary in available_required else ()
                selected_phase = "primary"
                strict_reasons: list[str] = []
                if force_full or fallback_detail:
                    strict_reasons.append(
                        fallback_detail or "automatic full-board fallback requested"
                    )
                if not classification.owner_scoped and (
                    active_obligations or snapshot.previous_key is not None
                ):
                    strict_reasons.append(classification.reason)
                if strict_reasons:
                    reason = (
                        STRICT_PRE_PANEL_PREFIX
                        + "; ".join(dict.fromkeys(strict_reasons))
                        + "; primary re-invoked with full context"
                    )
                elif active_obligations:
                    # Primary blocking loop: the primary rechecks its own plan
                    # findings on each candidate key until it approves.
                    reason = (
                        "primary phase: primary rechecks its plan findings before the "
                        "secondary panel"
                    )
                else:
                    reason = (
                        "primary phase: an exact-plan primary approval is required "
                        "before the secondary panel"
                    )
            else:
                # An exact-key primary approval opens the panel: every available
                # secondary lacking a qualified exact-plan approval receives its
                # first independent audit against the complete issue context and
                # the byte-identical candidate plan.
                selected = tuple(
                    name
                    for name in available_required
                    if name != primary and name not in approvals
                )
                selected_phase = "secondary-audit"
                records_panel_opening = True
                reason = (
                    "independent secondary audit after an exact-plan primary approval"
                )
                remaining = sorted(
                    obligation.item_id for obligation in active_obligations
                )
                if remaining:
                    reason += (
                        f"; non-reviewer plan obligations remain: {', '.join(remaining)}; "
                        "planner repair follows under post-panel rules"
                    )
        elif force_full:
            selected = available_required
            selected_phase = "full-board"
            latches_force_full = degraded or snapshot.force_full_source == "automatic"
            reason = POST_PANEL_PREFIX + "force-full latch" + (
                f" ({fallback_detail})" if fallback_detail else ""
            )
        elif classification.broad:
            # Any cross-cutting contract change, full-state rewrite, unbindable
            # sidecar, or unreconstructible ledger reactivates the complete
            # board once the panel has opened.
            selected = available_required
            selected_phase = "full-board"
            reason = (
                f"{POST_PANEL_PREFIX}full plan board required after panel evidence: "
                f"{classification.reason}"
            )
        elif active_obligations:
            owners = set(pending_owner_set)
            owners.add(primary)
            selected = tuple(
                name
                for name in available_required
                if name in owners and name not in approvals
            )
            selected_phase = "remediation"
            reason = (
                f"{POST_PANEL_PREFIX}narrow plan remediation: finding owners from the "
                "canonical ledger and the primary must recheck"
            )
        elif not primary_approved:
            selected = (primary,) if primary in available_required else ()
            selected_phase = "remediation"
            reason = (
                "remediation: an exact-plan primary approval is outstanding before the "
                "final secondary sweep"
            )
        else:
            selected = tuple(
                name for name in available_required if name not in approvals
            )
            selected_phase = "final-secondary-sweep"
            reason = "final exact-plan sweep for every required reviewer still missing an approval"

    selected_set = set(selected)
    paused: list[tuple[str, str]] = []
    for name in required:
        if name in selected_set:
            continue
        if name in unavailable:
            pause_reason = "reviewer unavailable; required plan approval remains outstanding"
        elif name in approvals:
            pause_reason = "qualifying exact-plan approval carried; no new turn needed"
        elif contract.policy == "all-reviewers":
            pause_reason = "compatibility scheduling did not select this reviewer"
        elif selected_phase == "primary":
            pause_reason = (
                "primary phase; the secondary panel waits for an exact-plan primary approval"
            )
        elif selected_phase == "secondary-audit":
            pause_reason = (
                "secondary audit has not selected this already qualifying reviewer"
            )
        elif selected_phase == "remediation":
            pause_reason = (
                "remediation did not identify this reviewer as a ledger owner or the "
                "primary; a final exact-plan sweep follows clearance"
            )
        elif selected_phase == "final-secondary-sweep":
            pause_reason = "final exact-plan sweep did not select this reviewer"
        else:
            pause_reason = "reviewer was not selected by the full plan board transition"
        paused.append((name, pause_reason))

    eligible = len(
        tuple(
            name
            for name in required
            if name not in approvals and name not in unavailable
        )
    )
    avoided = (
        max(0, eligible - len(selected)) if capabilities.counts_avoided_calls else 0
    )
    active_owners = tuple(sorted(pending_owner_set))
    return PlanSchedulingDecision(
        selected_reviewers=tuple(selected),
        paused_reviewers=tuple(paused),
        reason=reason,
        classification=classification,
        phase=selected_phase,
        primary_reviewer=primary,
        active_owners=active_owners,
        calls_avoided=avoided,
        final_sweep=selected_phase == "final-secondary-sweep",
        degraded_history_class=history_class,
        latches_force_full=latches_force_full,
        records_panel_opening=records_panel_opening,
    )


def make_plan_contract(
    required_reviewers: Sequence[str],
    policy: str,
    primary_reviewer: str | None = None,
) -> PlanReviewSchedulingContract:
    return PlanReviewSchedulingContract(
        required_reviewers=tuple(required_reviewers),
        policy=policy,
        primary_reviewer=primary_reviewer,
    )
