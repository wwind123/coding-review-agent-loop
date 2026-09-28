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
from .round_transport import _MAX_COMPRESSED, MAX_GITHUB_BODY_CHARS

SIGNAL_RENDERED_SIZE = "rendered-size"
SIGNAL_SCOPE_ITEMS = "scope-items"
SIGNAL_MATRIX_ROWS = "matrix-rows"
SIGNAL_REVISION_COUNT = "revision-count"
assert PLAN_GROWTH_SIGNALS == (
    SIGNAL_RENDERED_SIZE, SIGNAL_SCOPE_ITEMS, SIGNAL_MATRIX_ROWS, SIGNAL_REVISION_COUNT
)
STRUCTURAL_SIGNALS = (SIGNAL_RENDERED_SIZE, SIGNAL_SCOPE_ITEMS, SIGNAL_MATRIX_ROWS)

PLAN_GROWTH_GATE_MODES = ("enforce", "off")
# Size threshold derivation (#1075).  Canonical characters are not published
# characters: round transport spills every field that grows with the plan
# (canonical text, raw response, assembled sidecar, recommendation and matrix
# records) into sidecar comments, and spill stops as soon as the anchor fits,
# so a published anchor's size says little about the plan's.  What cannot
# spill is the anchor's visible plan text plus the metadata floor left once the
# growing fields are references.  When those exceed MAX_GITHUB_BODY_CHARS the
# full plan comment no longer fits, and the orchestrator posts the bounded
# compact digest instead (#948): the round still publishes, but readers see a
# summary rather than the plan.  That digest transition is the anchor point.
# Past it, body size no longer limits publication by canonical size: digest
# text is bounded (9.1-10.5k visible measured on #946 and #1035) and the
# canonical text spills, so the digest overflows only if unspillable metadata
# grows.  The one canonical-size publication limit left is the metadata codec:
# `_attach_round_metadata` encodes the complete metadata before any spill, and
# `encode_mapping` refuses a compressed form above _MAX_COMPRESSED (8,000,000
# bytes).  The metadata carries the plan at most about four times (canonical
# text, raw response and assembled sidecar JSON, plus patch provenance and the
# recommendation and matrix records), so assuming four copies at up to four
# UTF-8 bytes per character and no compression gives a conservative codec
# floor of 500,000 canonical characters, over four times the default
# (PLAN_TRANSPORT_CODEC_CANONICAL_FLOOR_CHARS).  Measured on this repository's
# plan anchors (2026-09-28, `plan_transport_profile`, see docs/local_agent_loop.md):
#
# * visible ratio: #886's approved plan publishes 17,791 visible characters for
#   68,956 canonical characters, 0.258.  That anchor is a full comment and is
#   transport-settled: its recommendation and matrix markers are already spill
#   references and both sections are compacted, so the 17,791 is residual
#   text transport cannot shrink further;
# * metadata floor: the largest residual round metadata of a plan anchor with
#   its canonical text already spilled is 24,995 characters (#871's final
#   candidate; 10.7-17.1k on #894, #943, #1040 and #1043);
# * digest transition: (60,000 - 24,995) / 0.258 = about 135,700 canonical
#   characters.  #871's final candidate published in full with 3,422
#   characters of anchor headroom, so grown plans really do reach it.
#
# The default is that transition less 10%, rounded down to 10,000, so a plan
# that grows past the size signal can still be shown in full.  It rests on one
# visible-ratio sample: plans measured at 120-195k canonical characters (PR
# #1072) still published, so their ratio was lower or they fell back to the
# digest, and prose-heavy plans run a higher ratio and an earlier transition.
# Rerun the profile across more plans before moving it.
PLAN_TRANSPORT_MEASURED_VISIBLE_RATIO = 17_791 / 68_956
PLAN_TRANSPORT_MEASURED_METADATA_FLOOR_CHARS = 24_995
PLAN_TRANSPORT_PROJECTED_DIGEST_TRANSITION_CHARS = int(
    (MAX_GITHUB_BODY_CHARS - PLAN_TRANSPORT_MEASURED_METADATA_FLOOR_CHARS)
    / PLAN_TRANSPORT_MEASURED_VISIBLE_RATIO
)
DEFAULT_PLAN_GROWTH_MAX_CHARS = (
    int(PLAN_TRANSPORT_PROJECTED_DIGEST_TRANSITION_CHARS * 0.9) // 10_000 * 10_000
)
# Four plan copies in the metadata, at up to four UTF-8 bytes per character.
PLAN_TRANSPORT_CODEC_CANONICAL_FLOOR_CHARS = _MAX_COMPRESSED // (4 * 4)
# Revision count combines with size at half the size threshold (60,000).
# #886's approved plan reached 68,956 characters over 7 planner candidates and
# #871's over 8; ordinary plans measured here (#894, #943) approved within 4-5.
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


VERDICT_COMPLIANT = "compliant"
VERDICT_NON_COMPLIANT = "non-compliant"
VERDICT_NOT_APPLICABLE = "not-applicable"
PLAN_GROWTH_VERDICT_STATUSES = (VERDICT_COMPLIANT, VERDICT_NON_COMPLIANT, VERDICT_NOT_APPLICABLE)
_THRESHOLD_KEYS = ("max_chars", "max_revisions", "max_scope_items", "max_matrix_rows")


@dataclass(frozen=True)
class PlanGrowthApprovalVerdict:
    """The growth gate's judgement of a plan when it crossed into implementation (#1074).

    Recorded in the approved-plan handoff record so handoff-backed resume can
    re-validate against what was in force at approval instead of today's
    thresholds.  ``status`` is computed whatever the gate mode, so a plan
    approved with the gate off still records whether it would have passed.
    ``not-applicable`` is the gate's own exemption: a legacy unversioned or
    free-form plan with no authenticated v1 execution recommendation.
    """

    gate: str
    status: str
    crossed_signals: tuple[str, ...]
    thresholds: PlanGrowthThresholds

    def to_payload(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "status": self.status,
            "crossed_signals": list(self.crossed_signals),
            "thresholds": {key: getattr(self.thresholds, key) for key in _THRESHOLD_KEYS},
        }

    @classmethod
    def from_payload(cls, payload: object, *, context: str) -> "PlanGrowthApprovalVerdict":
        if not isinstance(payload, Mapping) or set(payload) != {
            "gate", "status", "crossed_signals", "thresholds"
        }:
            raise AgentLoopError(
                f"{context}: expected exactly gate, status, crossed_signals and thresholds."
            )
        gate = payload["gate"]
        if gate not in PLAN_GROWTH_GATE_MODES:
            raise AgentLoopError(f"{context}: unknown gate mode {gate!r}.")
        status = payload["status"]
        if status not in PLAN_GROWTH_VERDICT_STATUSES:
            raise AgentLoopError(f"{context}: unknown status {status!r}.")
        crossed = payload["crossed_signals"]
        if (
            not isinstance(crossed, list)
            or any(not isinstance(item, str) for item in crossed)
            or crossed != [signal for signal in PLAN_GROWTH_SIGNALS if signal in crossed]
        ):
            raise AgentLoopError(
                f"{context}: `crossed_signals` must list known signals once each, in canonical order."
            )
        if status == VERDICT_NOT_APPLICABLE and crossed:
            raise AgentLoopError(f"{context}: a not-applicable verdict crosses no signal.")
        thresholds = payload["thresholds"]
        if not isinstance(thresholds, Mapping) or set(thresholds) != set(_THRESHOLD_KEYS):
            raise AgentLoopError(
                f"{context}: `thresholds` must carry exactly {', '.join(_THRESHOLD_KEYS)}."
            )
        for key in _THRESHOLD_KEYS:
            value = thresholds[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AgentLoopError(f"{context}: threshold `{key}` must be a positive integer.")
        return cls(
            gate=str(gate),
            status=str(status),
            crossed_signals=tuple(crossed),
            thresholds=PlanGrowthThresholds(**{key: thresholds[key] for key in _THRESHOLD_KEYS}),
        )

    def describe(self) -> str:
        crossed = ", ".join(self.crossed_signals) or "none"
        return f"gate {self.gate}, {self.status} (crossed: {crossed})"


def plan_growth_approval_verdict(
    config: object,
    plan: PlanLike | None,
    assessment: PlanGrowthAssessment | None,
) -> PlanGrowthApprovalVerdict:
    """The verdict to record for ``plan`` under ``config`` at approval time."""
    gate = getattr(config, "plan_growth_gate", "enforce")
    if plan is None or assessment is None or plan_strategy(plan) is None:
        return PlanGrowthApprovalVerdict(
            gate=gate,
            status=VERDICT_NOT_APPLICABLE,
            crossed_signals=(),
            thresholds=PlanGrowthThresholds.from_config(config),
        )
    violation = growth_justification_violation(plan, assessment)
    return PlanGrowthApprovalVerdict(
        gate=gate,
        status=VERDICT_COMPLIANT if violation is None else VERDICT_NON_COMPLIANT,
        crossed_signals=tuple(
            signal for signal in PLAN_GROWTH_SIGNALS if signal in assessment.crossed
        ),
        thresholds=assessment.thresholds,
    )


def handoff_growth_verdict_violation(
    verdict: PlanGrowthApprovalVerdict | None, config: object
) -> str | None:
    """Why a handoff-backed resume must refuse its recorded plan, or ``None``.

    Re-validates against the recorded approval-time verdict, never against
    today's thresholds, so a plan judged compliant when approved stays
    compliant.  A handoff record without a verdict predates it and is
    accepted as legacy by definition: that is an intended compatibility
    allowance, not an omission.  A recorded non-compliant verdict refuses
    while the gate is enforced now; ``--plan-growth-gate off`` keeps the
    explicit operator opt-out it has everywhere else.
    """
    if verdict is None or verdict.status != VERDICT_NON_COMPLIANT:
        return None
    if not plan_growth_gate_enforced(config):
        return None
    crossed = ", ".join(f"`{signal}`" for signal in verdict.crossed_signals) or "none"
    return (
        "the handoff record says the approved plan failed the plan-growth gate when it was "
        f"approved ({verdict.describe()}; crossed signals {crossed}), and the gate is enforced "
        "now. Re-plan so a revision restructures the plan as staged or carries a reviewed "
        "`one_shot_growth_justification`, or rerun with `--plan-growth-gate off` to accept it."
    )


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


def render_growth_measurements(assessment: PlanGrowthAssessment) -> str:
    """Reviewer-facing measurements for any one-shot plan that crosses a signal.

    Shown whether or not the plan is justified, so reviewers can evaluate a
    justification against the crossed signals and thresholds.
    """
    return (
        "Plan-growth measurements (orchestrator, not a reviewer finding): this one-shot plan "
        "crosses plan-growth threshold(s) "
        + ", ".join(f"`{signal}`" for signal in assessment.crossed)
        + f". Measurements: {assessment.describe()}. Evaluate its "
        "`one_shot_growth_justification` against these signals as a semantic claim, or block "
        "with the plan-growth lever."
    )


def render_growth_notice(
    assessment: PlanGrowthAssessment,
    *,
    violation: str | None = None,
    strategy: str | None = "one-shot",
) -> str:
    """Orchestrator growth notice for prompts; never a reviewer item.

    A candidate recovered from history can fail the gate without an uncovered
    crossing: a stale justification on a plan that crosses nothing, or any
    justification on a staged plan.  That notice names the actual violation
    (remove the justification) instead of an empty signal list.
    """
    if strategy != "one-shot" or not assessment.crossed:
        return (
            "Orchestrator plan-growth notice (not a reviewer finding): "
            + (violation or "the plan carries a stale `one_shot_growth_justification`.")
            + " No restructuring is required for this; only the stale justification must go. "
            f"Measurements: {assessment.describe()}."
        )
    return (
        "Orchestrator plan-growth notice (not a reviewer finding): the current one-shot plan "
        "crosses plan-growth threshold(s) "
        + ", ".join(f"`{signal}`" for signal in assessment.crossed)
        + f" without a justification that covers exactly them. Measurements: "
        f"{assessment.describe()}."
    )
