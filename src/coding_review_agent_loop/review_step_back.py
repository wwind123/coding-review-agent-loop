"""Deterministic "step back" bookkeeping for review stalls (#1251).

Pure, history-derived helpers with no I/O.  When a primary reviewer keeps
blocking with a *new* finding every round, patching each finding in turn makes
the artifact grow without converging.  The trigger here is code; the content of
the alternative stays with the planner and the reviewers.

Two concepts are kept deliberately separate:

* the **trigger streak**: consecutive ``new-finding`` primary reviews, which
  opens an episode; and
* the **active episode**: opened by a step-back turn, it governs escalation and
  suppresses a second step-back until an orchestrator-observed event closes it.

Everything is derived from the durable round-metadata records, so a live run and
a resumed run agree at exactly K and M.  A malformed entry marks the history
*degraded*: the trigger and the escalation are suppressed, never the run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .plan_assembly import decode_assembled_plan_sidecar
from .plan_growth import (
    PlanGrowthThresholds,
    assess_plan_growth,
    plan_growth_gate_enforced,
    plan_strategy,
)

if TYPE_CHECKING:
    from .round_state import PostedRoundMetadata, PostedRoundRecord

CLASS_NEW_FINDING = "new-finding"
CLASS_REPEAT_ONLY = "repeat-only"
CLASS_APPROVED = "approved"
CLASS_OTHER = "other"

PHASE_PLAN = "plan"
PHASE_PR = "pr"

# Item and disposition statuses that keep a finding mandatory.  ``future`` and
# follow-up items never count: they do not block the artifact.
_MANDATORY_STATUSES = {
    PHASE_PLAN: frozenset({"blocking", "same-plan"}),
    PHASE_PR: frozenset({"blocking", "same-pr"}),
}

ALTERNATIVE_EXCERPT_LIMIT = 2000


def classify_review(metadata: "PostedRoundMetadata", *, phase: str = PHASE_PLAN) -> str:
    """Classify one reviewer record from its own historical review payload.

    Reads ``new_items`` and the dispositions stored on that round's record,
    never the surviving unresolved ledger: an old blocker kept active plus one
    new ``future`` item is ``repeat-only``.
    """
    if metadata.state == "approved":
        return CLASS_APPROVED
    if metadata.state != "blocking":
        return CLASS_OTHER
    mandatory = _MANDATORY_STATUSES[phase]
    if any(
        item.status in mandatory and item.reviewer == metadata.agent
        for item in metadata.new_items
    ):
        return CLASS_NEW_FINDING
    return CLASS_REPEAT_ONLY


@dataclass(frozen=True)
class StepBackEntry:
    """A reviewer-owned step-back entry recorded on the step-back turn."""

    phase: str
    reviewer: str
    trigger_round: int
    candidate_round: int
    record_index: int


@dataclass(frozen=True)
class PrimaryReview:
    round_number: int
    index: int
    classification: str


@dataclass(frozen=True)
class StepBackState:
    """History-derived step-back state for one tracked reviewer."""

    degraded: bool
    reviews: tuple[PrimaryReview, ...]
    episode: StepBackEntry | None
    escalation_count: int

    def streak_since(self, first_round: int) -> int:
        """Consecutive newest-first ``new-finding`` reviews at or after ``first_round``.

        A repeat-only review, an approval, or the start of an episode ends it.
        """
        floor = first_round if self.episode is None else max(
            first_round, self.episode.candidate_round
        )
        count = 0
        for review in reversed(self.reviews):
            if review.round_number < floor or review.classification != CLASS_NEW_FINDING:
                break
            count += 1
        return count


def derive_plan_step_back_state(
    records: Sequence["PostedRoundRecord"],
    *,
    primary: str,
    panel_opening_index: int | None = None,
    current_issue_digest: str | None = None,
) -> StepBackState:
    """Derive the primary's streak inputs and active episode from plan records.

    Retirement mirrors the primary stall streak: a checkpoint recorded for a
    different issue digest retires its round and everything before it, and the
    operator reset marker retires the rounds *before* the checkpoint's round.
    A panel opening, an invalid scheduler record or a phase-advance record ends
    the primary phase, so nothing at or before it counts.  An approval by the
    primary closes the episode.
    """
    ordered = sorted(records, key=lambda item: item.index)
    degraded = any(record.metadata.step_back_status == "invalid" for record in ordered)
    boundary_index = max(
        (
            record.index
            for record in ordered
            if record.metadata.scheduler_metadata_status == "invalid"
            or (
                record.metadata.role == "summary"
                and record.metadata.phase == "plan-phase-advance"
            )
        ),
        default=-1,
    )
    retired_through = 0
    reset_indices: list[int] = []
    # Indices of the usable primary-phase checkpoints of each round: a counted
    # review must follow one of its own round (as the primary stall streak does).
    checkpoint_indices: dict[int, list[int]] = {}
    for record in ordered:
        metadata = record.metadata
        if (
            metadata.role != "summary"
            or metadata.phase != "scheduler-prelaunch"
            or record.index <= boundary_index
            or metadata.scheduler_metadata_status != "valid"
            or metadata.scheduler_phase != "primary"
        ):
            continue
        checkpoint_indices.setdefault(metadata.round_number, []).append(record.index)
        if metadata.scheduler_stall_reset:
            reset_indices.append(record.index)
            retired_through = max(retired_through, metadata.round_number - 1)
        recorded = metadata.scheduler_issue_digest
        if (
            recorded is not None
            and current_issue_digest is not None
            and recorded != current_issue_digest
        ):
            retired_through = max(retired_through, metadata.round_number)

    latest_review: dict[int, "PostedRoundRecord"] = {}
    for record in ordered:
        metadata = record.metadata
        if metadata.role != "reviewer" or metadata.agent != primary:
            continue
        if record.index <= boundary_index:
            continue
        if panel_opening_index is not None and record.index >= panel_opening_index:
            continue
        # A later record for the same round supersedes an earlier one.
        latest_review[metadata.round_number] = record
    counted: list[PrimaryReview] = []
    for number, record in sorted(latest_review.items()):
        if number <= retired_through:
            continue
        if not any(
            index < record.index for index in checkpoint_indices.get(number, ())
        ):
            # Legacy, full-board or otherwise unqualified history ends the
            # streak: degraded history can only shorten it, never lengthen it.
            counted.clear()
            continue
        counted.append(
            PrimaryReview(
                round_number=number,
                index=record.index,
                classification=classify_review(record.metadata),
            )
        )
    reviews = tuple(counted)

    episode: StepBackEntry | None = None
    for record in ordered:
        metadata = record.metadata
        if (
            metadata.role != "coder"
            or metadata.step_back_status != "valid"
            or record.index <= boundary_index
            or (panel_opening_index is not None and record.index >= panel_opening_index)
        ):
            continue
        for entry in metadata.step_back_entries:
            if entry.get("phase") != PHASE_PLAN or entry.get("reviewer") != primary:
                continue
            if metadata.round_number <= retired_through:
                continue
            episode = StepBackEntry(
                phase=PHASE_PLAN,
                reviewer=primary,
                trigger_round=int(entry["trigger_round"]),
                candidate_round=metadata.round_number,
                record_index=record.index,
            )
    if panel_opening_index is not None:
        # Opening the panel ends the primary phase, and with it the episode.
        episode = None
    if episode is not None and any(index > episode.record_index for index in reset_indices):
        # A reset checkpoint recorded after the step-back turn closes the episode
        # even when it carries the same round number as the step-back candidate.
        episode = None
    escalation_count = 0
    if episode is not None:
        closing = [
            review
            for review in reviews
            if review.classification == CLASS_APPROVED
            and review.round_number >= episode.candidate_round
        ]
        if closing:
            episode = None
        else:
            escalation_count = sum(
                1
                for review in reviews
                if review.round_number >= episode.candidate_round
                and review.classification in {CLASS_NEW_FINDING, CLASS_REPEAT_ONLY}
            )
    return StepBackState(
        degraded=degraded,
        reviews=reviews,
        episode=episode,
        escalation_count=escalation_count,
    )


def plan_growth_crossing_round(
    records: Sequence["PostedRoundRecord"],
    *,
    config: object,
    current_round: int,
) -> int | None:
    """Round of the earliest candidate in the current contiguous crossed run.

    Growth crossings are recomputed from each authenticated planner candidate's
    canonical plan and assembled sidecar, never persisted.  Returns ``None``
    when the newest candidate is not a crossed one-shot plan or the gate is off.
    """
    if not plan_growth_gate_enforced(config):
        return None
    candidates = sorted(
        (
            record.metadata
            for record in records
            if record.metadata.role == "coder"
            and record.metadata.canonical_plan is not None
            and record.metadata.assembled_plan_sidecar is not None
            and record.metadata.round_number <= current_round
        ),
        key=lambda metadata: metadata.round_number,
    )
    # A resumed replay of one round number counts once, the newest record wins.
    by_round = {metadata.round_number: metadata for metadata in candidates}
    thresholds = PlanGrowthThresholds.from_config(config)
    rounds = sorted(by_round)
    crossing: int | None = None
    for position, number in enumerate(rounds, start=1):
        metadata = by_round[number]
        try:
            sidecar = decode_assembled_plan_sidecar(metadata.assembled_plan_sidecar)
        except Exception:  # noqa: BLE001 - an unreadable sidecar ends the run of crossings
            crossing = None
            continue
        crossed = False
        if plan_strategy(sidecar.canonical_json) == "one-shot":
            assessment = assess_plan_growth(
                sidecar.canonical_json,
                rendered_chars=len(metadata.canonical_plan or ""),
                revision_count=position,
                thresholds=thresholds,
            )
            crossed = bool(assessment.crossed)
        if not crossed:
            crossing = None
        elif crossing is None:
            crossing = number
    if not rounds or rounds[-1] != current_round:
        return None
    return crossing


@dataclass(frozen=True)
class PlanStepBackContext:
    """Facts the planner revision prompt renders for a step-back turn."""

    measurements: str
    findings: tuple[str, ...]
    execution_mode: str
    streak: int


def render_plan_step_back_guidance(context: PlanStepBackContext) -> str:
    """The step-back planner instruction, shared by every prompt branch."""
    findings = "\n".join(f"- {line}" for line in context.findings) or "- (none recorded)"
    staging_allowed = context.execution_mode != "implement-one-shot"
    options = [
        "1. A materially simpler alternative design that dissolves the recent "
        "classes of findings (not a patch to the newest one).",
    ]
    if staging_allowed:
        options.append(
            "2. A split into stages. Scope-ledger preservation still applies: keep "
            "every prior scope item's ID, requirement and acceptance criteria verbatim."
        )
    else:
        options.append(
            "2. Under implement-one-shot a split cannot become staged execution in "
            "this run: offer a simpler one-shot design, or state an explicit caveat "
            "recommending that the issue be re-filed as staged work."
        )
    return (
        "STEP-BACK REVISION (orchestrator, not a reviewer finding): this plan crossed "
        "a plan-growth signal and the primary reviewer has blocked "
        f"{context.streak} consecutive rounds, each on a new finding. Patching the "
        "newest finding is not an acceptable response to this turn; it replaces the "
        "usual instruction to address the reviewer items one by one.\n"
        f"{context.measurements}\n"
        "Mandatory findings the primary raised since the crossing:\n"
        f"{findings}\n"
        "Choose exactly one:\n"
        + "\n".join(options)
        + "\nThe `summary` must open with the chosen kind (`simpler design`, "
        "`split` or `re-file caveat`) and state the trade-offs and which prior "
        "findings the alternative dissolves or defers. Prior items still need "
        "dispositions: mark an item the alternative dissolves as resolved with a "
        "note. The approval-time growth gate is unchanged.\n"
    )


def render_plan_step_back_review_notice() -> str:
    """The reviewer notice for the step-back candidate, shared by both branches."""
    return (
        "Step-back notice (orchestrator, not a reviewer finding): this candidate "
        "deliberately changes design direction after repeated blocking rounds. Judge "
        "the alternative on its merits, not as a diff against the prior plan, and say "
        "in your summary whether the direction is acceptable. The plan-growth "
        "approval gate is unchanged.\n"
    )


def render_step_back_human_decision(
    *,
    phase: str,
    measurements: str,
    step_back_round: int,
    blocks: int,
    threshold: int,
    alternative: str | None,
) -> str:
    """Human-decision-required diagnostic after a step-back that did not converge."""
    excerpt = (alternative or "").strip()
    if len(excerpt) > ALTERNATIVE_EXCERPT_LIMIT:
        excerpt = excerpt[: ALTERNATIVE_EXCERPT_LIMIT - 1].rstrip() + "…"
    if phase != PHASE_PLAN:
        raise ValueError(f"unsupported step-back phase: {phase}")
    return (
        "human decision required: the planner stepped back at round "
        f"{step_back_round} (a simplify-or-re-scope revision), and the primary plan "
        f"reviewer has since blocked {blocks} round(s), reaching "
        f"--plan-step-back-escalation-rounds {threshold}, before "
        "--plan-primary-stall-rounds. No reviewer and no planner turn were invoked. "
        f"{measurements} Latest alternative (excerpt of the step-back candidate "
        f"summary): {excerpt or '(not recorded)'} Decide one of: continue patching "
        "(rerun with --plan-step-back-rounds 0, or with --plan-reset-stall-streak, "
        "which also ends the episode); adopt the simpler alternative (narrow the "
        "issue text; an issue edit ends the episode); or split (rerun with "
        "--plan-execution-mode auto, or re-file the issue as staged work)."
    )


def entry_payload_for_plan(*, reviewer: str, trigger_round: int) -> Mapping[str, object]:
    """The durable reviewer-owned plan entry the orchestrator records."""
    return {"phase": PHASE_PLAN, "reviewer": reviewer, "trigger_round": trigger_round}


def plan_step_back_candidate_rounds(records: Sequence["PostedRoundRecord"]) -> frozenset[int]:
    """Planner rounds whose candidate was published by a step-back turn."""
    return frozenset(
        record.metadata.round_number
        for record in records
        if record.metadata.role == "coder"
        and record.metadata.step_back_status == "valid"
        and any(
            entry.get("phase") == PHASE_PLAN for entry in record.metadata.step_back_entries
        )
    )


def mandatory_plan_findings_since(
    records: Sequence["PostedRoundRecord"], *, primary: str, first_round: int, limit: int = 12
) -> tuple[str, ...]:
    """The primary's mandatory findings from ``first_round`` on, as ``id: first line``."""
    mandatory = _MANDATORY_STATUSES[PHASE_PLAN]
    latest: dict[int, "PostedRoundMetadata"] = {}
    for record in sorted(records, key=lambda item: item.index):
        metadata = record.metadata
        if metadata.role == "reviewer" and metadata.agent == primary:
            latest[metadata.round_number] = metadata
    lines: list[str] = []
    for number in sorted(latest):
        if number < first_round or latest[number].state != "blocking":
            continue
        for item in latest[number].new_items:
            if item.status in mandatory and item.reviewer == primary:
                first_line = next((line.strip() for line in item.text.splitlines() if line.strip()), "")
                lines.append(f"[{item.item_id}] (round {number}) {first_line[:240]}")
    return tuple(lines[-limit:])


def step_back_alternative_summary(
    records: Sequence["PostedRoundRecord"], episode: StepBackEntry
) -> str | None:
    """The step-back candidate's structured ``summary``, or ``None`` if unreadable."""
    for record in sorted(records, key=lambda item: item.index, reverse=True):
        metadata = record.metadata
        if metadata.role != "coder" or metadata.round_number != episode.candidate_round:
            continue
        raw = metadata.raw_structured_coder_response
        if not raw:
            return None
        try:
            payload, _end = json.JSONDecoder().raw_decode(raw.lstrip())
        except ValueError:
            return None
        summary = payload.get("summary") if isinstance(payload, dict) else None
        return summary if isinstance(summary, str) else None
    return None
