"""Structural plan-growth signals and the one-shot growth gate (#886).

``--plan-execution-mode auto`` picks one-shot or staged from the planner's
first recommendation.  Reviewers can then drive detail into a one-shot plan
round after round until it no longer fits a comment.  This module measures a
candidate plan deterministically and decides whether a grown one-shot plan
carries the reviewed justification the gate requires.

Every function here is pure: callers supply the canonical plan text length
and the authenticated planner-candidate count, and act on the result.
Reviewer finding counts are never an input.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import AgentLoopError
from .protocol import (
    PLAN_GROWTH_SIGNALS,
    StructuredPlanRevision,
    StructuredPlanState,
)
from .round_transport import MAX_GITHUB_BODY_CHARS

SIGNAL_RENDERED_SIZE = "rendered-size"
SIGNAL_SCOPE_ITEMS = "scope-items"
SIGNAL_MATRIX_ROWS = "matrix-rows"
SIGNAL_REVISION_COUNT = "revision-count"
assert PLAN_GROWTH_SIGNALS == (
    SIGNAL_RENDERED_SIZE, SIGNAL_SCOPE_ITEMS, SIGNAL_MATRIX_ROWS, SIGNAL_REVISION_COUNT
)
STRUCTURAL_SIGNALS = (SIGNAL_RENDERED_SIZE, SIGNAL_SCOPE_ITEMS, SIGNAL_MATRIX_ROWS)

PLAN_GROWTH_GATE_MODES = ("enforce", "off")
# Twice the GitHub body limit.  Canonical plan text carries the execution
# recommendation and risk matrix both rendered and as encoded records, so it
# runs about twice the visible plan: measured approved plans here were ~21k
# characters for a 3-scope-item plan and 50-85k for ordinary 5-6 item plans,
# while plans that grew past one delivery measured 120-195k.  A heuristic for
# "this plan carries too much design detail", not a transport guard: transport
# overflow is handled by the round transport and the compact digest.
DEFAULT_PLAN_GROWTH_MAX_CHARS = MAX_GITHUB_BODY_CHARS * 2
DEFAULT_PLAN_GROWTH_MAX_REVISIONS = 6
DEFAULT_PLAN_GROWTH_MAX_SCOPE_ITEMS = 12
DEFAULT_PLAN_GROWTH_MAX_MATRIX_ROWS = 18

PlanLike = StructuredPlanState | StructuredPlanRevision | Mapping[str, object]


@dataclass(frozen=True)
class PlanGrowthThresholds:
    max_chars: int = DEFAULT_PLAN_GROWTH_MAX_CHARS
    max_revisions: int = DEFAULT_PLAN_GROWTH_MAX_REVISIONS
    max_scope_items: int = DEFAULT_PLAN_GROWTH_MAX_SCOPE_ITEMS
    max_matrix_rows: int = DEFAULT_PLAN_GROWTH_MAX_MATRIX_ROWS

    @classmethod
    def from_config(cls, config: object) -> "PlanGrowthThresholds":
        return cls(
            max_chars=getattr(config, "plan_growth_max_chars", DEFAULT_PLAN_GROWTH_MAX_CHARS),
            max_revisions=getattr(
                config, "plan_growth_max_revisions", DEFAULT_PLAN_GROWTH_MAX_REVISIONS
            ),
            max_scope_items=getattr(
                config, "plan_growth_max_scope_items", DEFAULT_PLAN_GROWTH_MAX_SCOPE_ITEMS
            ),
            max_matrix_rows=getattr(
                config, "plan_growth_max_matrix_rows", DEFAULT_PLAN_GROWTH_MAX_MATRIX_ROWS
            ),
        )


def plan_growth_gate_enforced(config: object) -> bool:
    return getattr(config, "plan_growth_gate", "enforce") == "enforce"


@dataclass(frozen=True)
class PlanGrowthAssessment:
    rendered_chars: int
    # ``None`` when the caller deliberately does not use revision count
    # (the rebind advisory measures structure only).
    revision_count: int | None
    scope_items: int
    matrix_rows: int
    thresholds: PlanGrowthThresholds
    crossed: tuple[str, ...]

    def describe(self) -> str:
        parts = [
            f"canonical plan size {self.rendered_chars} characters "
            f"(threshold {self.thresholds.max_chars}; the GitHub comment limit is "
            f"{MAX_GITHUB_BODY_CHARS})",
            f"scope items {self.scope_items} (threshold {self.thresholds.max_scope_items})",
            f"risk-matrix rows {self.matrix_rows} (threshold {self.thresholds.max_matrix_rows})",
        ]
        if self.revision_count is not None:
            parts.append(
                f"planner revisions {self.revision_count} (threshold "
                f"{self.thresholds.max_revisions}, counted only when the plan is at least "
                f"{self.thresholds.max_chars // 2} characters)"
            )
        return "; ".join(parts)


def _plan_facts(
    plan: PlanLike,
) -> tuple[int | None, Mapping[str, object] | None, Mapping[str, object] | None, Mapping[str, object] | None]:
    """Return (execution contract version, recommendation, matrix, justification)."""
    if isinstance(plan, Mapping):
        version = plan.get("execution_strategy_contract_version")
        recommendation = plan.get("execution_recommendation")
        matrix = plan.get("risk_test_matrix")
        justification = plan.get("one_shot_growth_justification")
        return (
            version if isinstance(version, int) and not isinstance(version, bool) else None,
            recommendation if isinstance(recommendation, Mapping) else None,
            matrix if isinstance(matrix, Mapping) else None,
            justification if isinstance(justification, Mapping) else None,
        )
    recommendation = plan.execution_recommendation
    matrix = plan.risk_test_matrix
    justification = plan.one_shot_growth_justification
    return (
        plan.execution_strategy_contract_version,
        recommendation.to_payload() if recommendation is not None else None,
        matrix.to_payload() if matrix is not None else None,
        justification.to_payload() if justification is not None else None,
    )


def _scope_items(recommendation: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    items = recommendation.get("scope_items") if recommendation is not None else None
    return [item for item in items if isinstance(item, Mapping)] if isinstance(items, list) else []


def plan_strategy(plan: PlanLike) -> str | None:
    """The v1 recommendation strategy, or ``None`` for a legacy unversioned plan."""
    version, recommendation, _matrix, _justification = _plan_facts(plan)
    if version != 1 or recommendation is None:
        return None
    strategy = recommendation.get("strategy")
    return strategy if isinstance(strategy, str) else None


def plan_justification(plan: PlanLike) -> Mapping[str, object] | None:
    return _plan_facts(plan)[3]


def assess_plan_growth(
    plan: PlanLike,
    *,
    rendered_chars: int,
    revision_count: int | None,
    thresholds: PlanGrowthThresholds,
) -> PlanGrowthAssessment:
    """Measure ``plan`` and name the crossed signals.

    ``rendered_chars`` is the length of the candidate's canonical plan text
    (the text ``approved_plan_hash`` hashes).  A signal crosses when its
    measurement reaches its threshold.  Revision count alone never crosses:
    it counts only once the plan is at least half the size threshold.
    """
    _version, recommendation, matrix, _justification = _plan_facts(plan)
    scope_ids = {
        str(item.get("scope_item_id"))
        for item in _scope_items(recommendation)
        if item.get("scope_item_id") is not None
    }
    rows = matrix.get("rows") if matrix is not None else None
    matrix_rows = len(rows) if isinstance(rows, list) else 0
    crossed: list[str] = []
    if rendered_chars >= thresholds.max_chars:
        crossed.append(SIGNAL_RENDERED_SIZE)
    if len(scope_ids) >= thresholds.max_scope_items:
        crossed.append(SIGNAL_SCOPE_ITEMS)
    if matrix_rows >= thresholds.max_matrix_rows:
        crossed.append(SIGNAL_MATRIX_ROWS)
    if (
        revision_count is not None
        and revision_count >= thresholds.max_revisions
        and rendered_chars * 2 >= thresholds.max_chars
    ):
        crossed.append(SIGNAL_REVISION_COUNT)
    return PlanGrowthAssessment(
        rendered_chars=rendered_chars,
        revision_count=revision_count,
        scope_items=len(scope_ids),
        matrix_rows=matrix_rows,
        thresholds=thresholds,
        crossed=tuple(crossed),
    )


def growth_justification_violation(plan: PlanLike, assessment: PlanGrowthAssessment) -> str | None:
    """Why ``plan`` fails the growth gate for its own measurements, or ``None``.

    Legacy unversioned plans are never subject to the gate.  A one-shot plan
    must carry a justification naming exactly the signals it crosses, and
    none when it crosses nothing; a staged plan must carry none.
    """
    strategy = plan_strategy(plan)
    if strategy is None:
        return None
    justification = plan_justification(plan)
    crossed = set(assessment.crossed)
    if strategy != "one-shot":
        if justification is not None:
            return (
                "A staged execution_recommendation must not carry "
                "`one_shot_growth_justification`; remove it (a semantic patch uses "
                "`replace` with value null)."
            )
        return None
    if not crossed:
        if justification is not None:
            return (
                "This one-shot plan crosses no plan-growth threshold "
                f"({assessment.describe()}), so its `one_shot_growth_justification` is stale; "
                "remove it (a semantic patch uses `replace` with value null)."
            )
        return None
    ordered = [signal for signal in PLAN_GROWTH_SIGNALS if signal in crossed]
    if justification is None:
        return (
            "This one-shot plan crosses plan-growth threshold(s) "
            + ", ".join(f"`{signal}`" for signal in ordered)
            + f" ({assessment.describe()}). Either restructure it as a staged "
            "execution_recommendation, keeping scope, stage boundaries, interfaces and "
            "acceptance criteria in this parent plan and moving per-stage design into "
            "requires-child-planning children, or add `one_shot_growth_justification` "
            "naming exactly these signals with a rationale reviewers can evaluate."
        )
    named = justification.get("crossed_signals")
    named_set = {str(item) for item in named} if isinstance(named, list) else set()
    if named_set != crossed:
        missing = [signal for signal in ordered if signal not in named_set]
        extra = sorted(named_set - crossed)
        details = []
        if missing:
            details.append("missing " + ", ".join(f"`{signal}`" for signal in missing))
        if extra:
            details.append("naming signal(s) no longer crossed " + ", ".join(f"`{signal}`" for signal in extra))
        return (
            "`one_shot_growth_justification.crossed_signals` must name exactly the "
            "signals this plan crosses ("
            + ", ".join(f"`{signal}`" for signal in ordered)
            + "): "
            + "; ".join(details)
            + f". Measurements: {assessment.describe()}."
        )
    return None


def check_growth_justification(plan: PlanLike, assessment: PlanGrowthAssessment) -> None:
    """Post-assembly self-check: raise a correctable validation failure."""
    violation = growth_justification_violation(plan, assessment)
    if violation is not None:
        raise AgentLoopError(f"Plan growth gate: {violation}")


def _normalize(text: object) -> str:
    return " ".join(str(text).split())


def check_scope_ledger_preservation(prior: PlanLike | None, candidate: PlanLike) -> None:
    """A one-shot-to-staged conversion must keep the approved scope ledger.

    Every prior scope item keeps its ID, its requirement text and all of its
    acceptance criteria (whitespace-normalized).  New items and added
    criteria are allowed.  Applies to v1 plans only, whatever the growth gate
    mode; legacy unversioned plans are never checked.
    """
    if prior is None or plan_strategy(prior) != "one-shot" or plan_strategy(candidate) != "staged":
        return
    candidate_items = {
        str(item.get("scope_item_id")): item for item in _scope_items(_plan_facts(candidate)[1])
    }
    problems: list[str] = []
    for item in _scope_items(_plan_facts(prior)[1]):
        scope_id = str(item.get("scope_item_id"))
        current = candidate_items.get(scope_id)
        if current is None:
            problems.append(f"scope item `{scope_id}` was dropped or renamed")
            continue
        if _normalize(item.get("requirement", "")) != _normalize(current.get("requirement", "")):
            problems.append(f"scope item `{scope_id}` changed its requirement text")
        prior_criteria = item.get("acceptance_criteria")
        current_criteria = current.get("acceptance_criteria")
        kept = {
            _normalize(value) for value in (current_criteria if isinstance(current_criteria, list) else [])
        }
        lost = [
            value
            for value in (prior_criteria if isinstance(prior_criteria, list) else [])
            if _normalize(value) not in kept
        ]
        if lost:
            problems.append(
                f"scope item `{scope_id}` removed or reworded {len(lost)} acceptance criterion(s)"
            )
    if problems:
        raise AgentLoopError(
            "Converting a one-shot plan to staged must preserve the scope ledger: "
            + "; ".join(problems)
            + ". Keep every prior scope item's ID, requirement text and acceptance criteria "
            "verbatim (adding criteria or new scope items is allowed); make intentional scope "
            "edits in a separate non-converting revision."
        )


def render_growth_notice(assessment: PlanGrowthAssessment) -> str:
    """Orchestrator growth notice for prompts; never a reviewer item."""
    return (
        "Orchestrator plan-growth notice (not a reviewer finding): the current one-shot plan "
        "crosses plan-growth threshold(s) "
        + ", ".join(f"`{signal}`" for signal in assessment.crossed)
        + f" without a justification that covers exactly them. Measurements: "
        f"{assessment.describe()}."
    )
