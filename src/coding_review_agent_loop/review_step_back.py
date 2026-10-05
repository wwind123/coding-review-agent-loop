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

import dataclasses
import json
import re
from collections.abc import Callable, Mapping, Sequence
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
# A blocking publication record whose round has no later reconciliation: its
# new items are unknowable, so it can only end a streak, never start one.
CLASS_BLOCKING_UNRESOLVED = "blocking-unresolved"
BLOCKING_CLASSES = frozenset({CLASS_NEW_FINDING, CLASS_REPEAT_ONLY, CLASS_BLOCKING_UNRESOLVED})

PHASE_PLAN = "plan"
PHASE_PR = "pr"

# Item and disposition statuses that keep a finding mandatory.  ``future`` and
# follow-up items never count: they do not block the artifact.
_MANDATORY_STATUSES = {
    PHASE_PLAN: frozenset({"blocking", "same-plan"}),
    PHASE_PR: frozenset({"blocking", "same-pr"}),
}

ALTERNATIVE_EXCERPT_LIMIT = 2000


_USE_RECORD_ITEMS = object()


def effective_new_items(
    records: Sequence["PostedRoundRecord"], record: "PostedRoundRecord"
) -> "tuple | None":
    """The new items a reviewer record introduced, or ``None`` when unknowable.

    Under ``--review-parallel`` the reviewer comment is posted early as a
    provisional ``publication`` record with no ``new_items``; the minted items land
    on the round's later reconciliation summary.  A publication record therefore
    takes the reviewer-owned items of the latest same-flow, same-round,
    same-subject reconciliation recorded after it, and is unresolvable (``None``)
    when none exists.  Every other record carries its own ``new_items``.
    """
    metadata = record.metadata
    if metadata.phase != "publication":
        return metadata.new_items
    best = None
    for candidate in records:
        meta = candidate.metadata
        if (
            meta.role == "summary"
            and meta.phase == "reconciliation"
            and meta.flow == metadata.flow
            and meta.round_number == metadata.round_number
            and meta.subject == metadata.subject
            and candidate.index > record.index
            and (best is None or candidate.index > best.index)
        ):
            best = candidate
    if best is None:
        return None
    return tuple(item for item in best.metadata.new_items if item.reviewer == metadata.agent)


def classify_review(
    metadata: "PostedRoundMetadata",
    *,
    phase: str = PHASE_PLAN,
    new_items: object = _USE_RECORD_ITEMS,
) -> str:
    """Classify one reviewer record from its historical review payload.

    Reads the review's new items and the dispositions stored on that round's
    record, never the surviving unresolved ledger: an old blocker kept active plus
    one new ``future`` item is ``repeat-only``.  ``new_items`` defaults to the
    record's own; pass :func:`effective_new_items` output for a provisional
    publication record (``None`` means unresolvable).
    """
    if metadata.state == "approved":
        return CLASS_APPROVED
    if metadata.state != "blocking":
        return CLASS_OTHER
    items = metadata.new_items if new_items is _USE_RECORD_ITEMS else new_items
    if items is None:
        return CLASS_BLOCKING_UNRESOLVED
    mandatory = _MANDATORY_STATUSES[phase]
    if any(
        item.status in mandatory and item.reviewer == metadata.agent
        for item in items
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
    # Rounds whose round-start stall stop was deferred for a pending step-back
    # (#1275), from usable primary checkpoints that survive the same retirement
    # rules as the counted reviews.
    deferral_rounds: tuple[int, ...] = ()

    def streak_before(self, first_round: int, before_round: int) -> int:
        """Like :meth:`streak_since`, ignoring reviews of rounds ``>= before_round``."""
        floor = first_round if self.episode is None else max(
            first_round, self.episode.candidate_round
        )
        count = 0
        for review in reversed(self.reviews):
            if review.round_number >= before_round:
                continue
            if review.round_number < floor or review.classification != CLASS_NEW_FINDING:
                break
            count += 1
        return count

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

    Retirement mirrors the primary stall streak: a review whose latest preceding
    same-round checkpoint recorded a different issue digest retires itself and
    everything before it, and the operator reset marker retires the rounds
    *before* the checkpoint's round.
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
    checkpoint_indices: dict[int, list[tuple[int, str | None]]] = {}
    # Every usable primary checkpoint in order, for episode digest closure.
    all_checkpoints: list[tuple[int, str | None]] = []
    deferrals: list[tuple[int, int, str | None]] = []
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
        checkpoint_indices.setdefault(metadata.round_number, []).append(
            (record.index, metadata.scheduler_issue_digest)
        )
        all_checkpoints.append((record.index, metadata.scheduler_issue_digest))
        if metadata.scheduler_stall_reset:
            reset_indices.append(record.index)
            retired_through = max(retired_through, metadata.round_number - 1)
        if metadata.scheduler_step_back_deferral and (
            panel_opening_index is None or record.index < panel_opening_index
        ):
            deferrals.append(
                (record.index, metadata.round_number, metadata.scheduler_issue_digest)
            )

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
        preceding = [
            digest
            for index, digest in checkpoint_indices.get(number, ())
            if index < record.index
        ]
        if not preceding:
            # Legacy, full-board or otherwise unqualified history ends the
            # streak: degraded history can only shorten it, never lengthen it.
            counted.clear()
            continue
        recorded = preceding[-1]
        if (
            recorded is not None
            and current_issue_digest is not None
            and recorded != current_issue_digest
        ):
            # The latest checkpoint before this review (not any same-round
            # checkpoint) names the issue text it judged: an edit retires this
            # review and everything before it, while a fresh re-checkpoint of the
            # same round under the edited issue keeps its own review.
            counted.clear()
            continue
        counted.append(
            PrimaryReview(
                round_number=number,
                index=record.index,
                classification=classify_review(
                    record.metadata, new_items=effective_new_items(ordered, record)
                ),
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
    if episode is not None and current_issue_digest is not None:
        # An issue edit closes the episode.  The turn was produced under the issue
        # text of the latest checkpoint before its record (the trigger round's), so
        # check that one, which covers an edit between publication and the
        # candidate's first checkpoint, and every checkpoint after it.
        before = [
            digest for index, digest in all_checkpoints if index < episode.record_index
        ]
        digests = ([before[-1]] if before else []) + [
            digest for index, digest in all_checkpoints if index > episode.record_index
        ]
        if any(d is not None and d != current_issue_digest for d in digests):
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
                and review.classification
                in {CLASS_NEW_FINDING, CLASS_REPEAT_ONLY, CLASS_BLOCKING_UNRESOLVED}
            )
    # An issue edit retires the latest mismatching deferral and everything before it.
    if current_issue_digest is not None:
        for position in range(len(deferrals) - 1, -1, -1):
            recorded = deferrals[position][2]
            if recorded is not None and recorded != current_issue_digest:
                deferrals = deferrals[position + 1 :]
                break
    deferral_rounds = tuple(
        sorted({number for _index, number, _digest in deferrals if number > retired_through})
    )
    return StepBackState(
        degraded=degraded,
        reviews=reviews,
        episode=episode,
        escalation_count=escalation_count,
        deferral_rounds=deferral_rounds,
    )


DISPOSITION_DEFER_PENDING = "defer-pending"
DISPOSITION_DEFER_EPISODE = "defer-episode"
DISPOSITION_ESCALATED = "escalated"
DISPOSITION_NOT_APPLICABLE = "not-applicable"


@dataclass(frozen=True)
class StallStepBackDisposition:
    """What a reached primary stall limit does about the plan step-back (#1275)."""

    kind: str
    reason: str


def plan_stall_step_back_disposition(
    state: StepBackState | None,
    *,
    round_number: int,
    crossing_round: int | None,
    growth_gate_enforced: bool,
    step_back_rounds: int,
    escalation_rounds: int,
) -> StallStepBackDisposition:
    """Fixed-precedence decision the round-start stall stop consults.

    Eligibility is judged from rounds before ``round_number`` so the answer does
    not change when that round's own primary review is posted.  The stall streak
    is deliberately not an input: an eligible history above the threshold is
    granted too.
    """
    na = DISPOSITION_NOT_APPLICABLE
    if step_back_rounds <= 0:
        return StallStepBackDisposition(na, "step-back disabled (--plan-step-back-rounds 0)")
    if state is None or state.degraded:
        return StallStepBackDisposition(na, "planning step-back history degraded")
    episode = state.episode
    if episode is not None:
        if state.escalation_count >= escalation_rounds:
            return StallStepBackDisposition(
                DISPOSITION_ESCALATED,
                f"step-back episode from round {episode.candidate_round} reached "
                f"{state.escalation_count}/{escalation_rounds} escalation block(s)",
            )
        return StallStepBackDisposition(
            DISPOSITION_DEFER_EPISODE,
            f"step-back episode from round {episode.candidate_round} at "
            f"{state.escalation_count}/{escalation_rounds} escalation block(s)",
        )
    if not growth_gate_enforced:
        return StallStepBackDisposition(na, "plan-growth gate is off")
    if crossing_round is None:
        return StallStepBackDisposition(
            na, "the current candidate crosses no plan-growth signal"
        )
    for deferred in state.deferral_rounds:
        if crossing_round <= deferred < round_number:
            return StallStepBackDisposition(
                na,
                f"a step-back was already deferred at round {deferred} and no step-back "
                "turn followed (an orchestrator-owned revision took precedence)",
            )
    streak = state.streak_before(crossing_round, round_number)
    if streak < step_back_rounds:
        return StallStepBackDisposition(
            na,
            f"only {streak} new-finding primary block(s) since the crossing at round "
            f"{crossing_round}, below --plan-step-back-rounds {step_back_rounds}",
        )
    return StallStepBackDisposition(
        DISPOSITION_DEFER_PENDING,
        f"growth signal crossed at round {crossing_round}; {streak} new-finding "
        "primary block(s) before this round",
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
        f"--plan-step-back-escalation-rounds {threshold} (the stall stop is deferred "
        "while a step-back is pending or in progress). No reviewer and no planner turn were invoked. "
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
    ordered = sorted(records, key=lambda item: item.index)
    latest: dict[int, "PostedRoundRecord"] = {}
    for record in ordered:
        metadata = record.metadata
        if metadata.role == "reviewer" and metadata.agent == primary:
            latest[metadata.round_number] = record
    lines: list[str] = []
    for number in sorted(latest):
        if number < first_round or latest[number].metadata.state != "blocking":
            continue
        for item in effective_new_items(ordered, latest[number]) or ():
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


# ---------------------------------------------------------------------------
# PR fix-loop step-back (#1251, stage 2)
#
# Finding locations are ``path:start[-end]`` references in finding text.  They
# are bound to the head their review was taken on and compared across heads by
# mapping a *anchor* through ``git diff`` hunks.  One membership predicate,
# :func:`in_cluster`, governs clustering, escalation and clearance.
# ---------------------------------------------------------------------------

OUTCOME_SHIFTED = "SHIFTED"
OUTCOME_REWRITTEN = "REWRITTEN"
OUTCOME_UNMAPPABLE = "UNMAPPABLE"

_LOCATION_RE = re.compile(
    r"(?<![\w./:@-])"
    r"((?:[\w.\-]+/)*[\w\-][\w.\-]*)"
    r":(\d+)(?:\s*[-\u2013]\s*(\d+))?(?!\d)"
)
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class FindingLocation:
    """One ``path:start-end`` reference projected from a finding.

    ``item_id`` is always the parent finding's identity; ``sub_item_id`` is set
    when the reference came from a sub-item statement.
    """

    path: str
    start: int
    end: int
    item_id: str
    sub_item_id: str | None = None


def _looks_like_repo_path(path: str) -> bool:
    """Any repository-relative path: extensionless files (``run``, ``Makefile``) are
    valid.  Absolute paths are excluded here and URLs by the pattern's lookbehind."""
    return bool(path) and not path.startswith("/")


def _scan_locations(
    text: str, *, item_id: str, sub_item_id: str | None
) -> list[FindingLocation]:
    found: list[FindingLocation] = []
    for match in _LOCATION_RE.finditer(text or ""):
        if not _looks_like_repo_path(match.group(1)):
            continue
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) else start
        if start < 1:
            continue
        if end < start:
            start, end = end, start
        found.append(FindingLocation(match.group(1), start, end, item_id, sub_item_id))
    return found


def project_finding_locations(item: object) -> tuple[FindingLocation, ...]:
    """Locations in a finding's text, ``fix_scope`` entries and open sub-items.

    References without a line are ignored.  Sub-item locations keep the parent
    item identity, so one finding can contribute several locations.
    """
    item_id = str(getattr(item, "item_id", ""))
    locations = _scan_locations(
        str(getattr(item, "text", "") or ""), item_id=item_id, sub_item_id=None
    )
    for entry in getattr(item, "fix_scope", None) or ():
        locations.extend(_scan_locations(str(entry), item_id=item_id, sub_item_id=None))
    for sub in getattr(item, "sub_items", None) or ():
        if getattr(sub, "status", "open") == "resolved":
            continue
        locations.extend(
            _scan_locations(
                str(getattr(sub, "text", "") or ""),
                item_id=item_id,
                sub_item_id=str(getattr(sub, "sub_item_id", "")),
            )
        )
    seen: set[tuple[str, int, int, str | None]] = set()
    unique: list[FindingLocation] = []
    for location in locations:
        key = (location.path, location.start, location.end, location.sub_item_id)
        if key not in seen:
            seen.add(key)
            unique.append(location)
    return tuple(unique)


@dataclass(frozen=True)
class AnchorMapping:
    """Where an anchor lives on another head, with an explicit outcome."""

    outcome: str
    original_path: str
    path: str
    start: int
    end: int
    reason: str = ""

    @property
    def mappable(self) -> bool:
        return self.outcome != OUTCOME_UNMAPPABLE


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int

    @property
    def old_end(self) -> int:
        return self.old_start + self.old_count - 1


def _name_status_records(text: str) -> list[tuple[str, list[str]]]:
    """``(status, paths)`` records from ``--name-status``, NUL-delimited (``-z``) or
    tab-delimited."""
    records: list[tuple[str, list[str]]] = []
    if "\0" in (text or ""):
        tokens = [token for token in text.split("\0")]
        index = 0
        while index < len(tokens) and tokens[index]:
            status = tokens[index]
            width = 2 if status[:1] in {"R", "C"} else 1
            records.append((status, tokens[index + 1 : index + 1 + width]))
            index += 1 + width
        return records
    for line in (text or "").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0]:
            records.append((parts[0], parts[1:]))
    return records


def parse_name_status(text: str) -> tuple[dict[str, str], frozenset[str]]:
    """``(renames old->new, deleted paths)`` from ``git diff --name-status -M``."""
    renames: dict[str, str] = {}
    deleted: set[str] = set()
    for status, paths in _name_status_records(text):
        if status[:1] == "R" and len(paths) >= 2:
            renames[paths[0]] = paths[1]
        elif status[:1] == "D" and paths:
            deleted.add(paths[0])
    return renames, frozenset(deleted)


def name_status_changed_paths(text: str) -> frozenset[str]:
    """Every path a ``--name-status`` listing names as changed (old and new sides)."""
    return frozenset(
        path
        for status, paths in _name_status_records(text)
        if status[:1] != "C"
        for path in paths
    )


def select_file_diff(text: str, old_path: str, new_path: str) -> str | None:
    """The section of a multi-file diff for exactly ``old_path`` -> ``new_path``.

    A pathspec naming both sides of a rename can also return other file
    identities (a rename chain), so hunks are taken only from the matching
    ``diff --git`` section.  Output with no section headers is one file's hunks;
    ``None`` means sections exist but none belongs to this pair.
    """
    lines = (text or "").splitlines()
    if not any(line.startswith("diff --git ") for line in lines):
        return text or ""
    header = f"diff --git a/{old_path} b/{new_path}"
    selected: list[str] = []
    matched = False
    active = False
    for line in lines:
        if line.startswith("diff --git "):
            active = line == header
            matched = matched or active
        if active:
            selected.append(line)
    return "\n".join(selected) if matched else None


def parse_zero_context_hunks(text: str) -> tuple[Hunk, ...]:
    """Hunks of a ``git diff -U0`` output, in order."""
    hunks: list[Hunk] = []
    for line in (text or "").splitlines():
        match = _HUNK_RE.match(line)
        if match is None:
            continue
        hunks.append(
            Hunk(
                int(match.group(1)),
                1 if match.group(2) is None else int(match.group(2)),
                int(match.group(3)),
                1 if match.group(4) is None else int(match.group(4)),
            )
        )
    return tuple(hunks)


def _unmappable(path: str, start: int, end: int, reason: str, new_path: str | None = None):
    return AnchorMapping(OUTCOME_UNMAPPABLE, path, new_path or path, start, end, reason)


def map_anchor(
    path: str,
    start: int,
    end: int,
    window: int,
    *,
    name_status: str | None,
    diff_text: str | None,
) -> AnchorMapping:
    """Map ``path:start-end`` across a diff, identity-preserving.

    SHIFTED: the range lies in unchanged lines and moves by the cumulative hunk
    offset.  REWRITTEN: changed hunks overlap it, their old side stays within
    ``[start - window, end + window]`` and the new side is not empty; the range
    maps to the replacement span.  UNMAPPABLE: the file was deleted with no
    rename pair, the rewrite spills beyond the window, the new side is empty
    (a deletion, which includes a move elsewhere), or the diff failed.  Ranges
    are never widened to a whole hunk beyond the window.
    """
    if name_status is None:
        return _unmappable(path, start, end, "the diff could not be read")
    renames, deleted = parse_name_status(name_status)
    new_path = renames.get(path, path)
    if new_path == path and path in deleted:
        return _unmappable(path, start, end, "the file was deleted with no rename pair")
    if diff_text is None:
        # Keep a known rename destination so the same-path fallback still works.
        return _unmappable(path, start, end, "the diff could not be read", new_path)
    section = select_file_diff(diff_text, path, new_path)
    if section is None:
        if {path, new_path} & name_status_changed_paths(name_status):
            # The listing says the file changed but its diff section was not found
            # (for example a quoted path): never read that as "unchanged".
            return _unmappable(
                path, start, end, "the changed file's diff section was not found", new_path
            )
        section = ""
    hunks = parse_zero_context_hunks(section)
    offset_before = 0
    overlapping: list[Hunk] = []
    for hunk in hunks:
        if hunk.old_count == 0:
            # Pure insertion after old line ``old_start``.
            if hunk.old_start < start:
                offset_before += hunk.new_count
            elif hunk.old_start < end:
                overlapping.append(hunk)
            continue
        if hunk.old_end < start:
            offset_before += hunk.new_count - hunk.old_count
        elif hunk.old_start <= end:
            overlapping.append(hunk)
    if not overlapping:
        return AnchorMapping(
            OUTCOME_SHIFTED, path, new_path, start + offset_before, end + offset_before
        )
    low = min(h.old_start for h in overlapping)
    high = max(max(h.old_end, h.old_start) for h in overlapping)
    if low < start - window or high > end + window:
        return _unmappable(
            path, start, end, "the anchor was rewritten together with unrelated code", new_path
        )
    if sum(h.new_count for h in overlapping) == 0:
        return _unmappable(
            path, start, end, "the anchored code was deleted or moved elsewhere", new_path
        )
    first, last = overlapping[0], overlapping[-1]
    delta_inside = sum(h.new_count - h.old_count for h in overlapping)
    new_first = first.new_start if first.new_count > 0 else first.new_start + 1
    new_last = (
        last.new_start + last.new_count - 1 if last.new_count > 0 else last.new_start
    )
    mapped_start = start + offset_before if start < first.old_start else new_first
    last_old_end = max(last.old_end, last.old_start)
    mapped_end = end + offset_before + delta_inside if end > last_old_end else new_last
    mapped_start = max(1, mapped_start)
    return AnchorMapping(
        OUTCOME_REWRITTEN, path, new_path, mapped_start, max(mapped_start, mapped_end)
    )


def in_cluster(location: FindingLocation, mapping: AnchorMapping, window: int) -> bool:
    """The single membership predicate for clustering, escalation and clearance.

    A mappable anchor admits locations on the mapped path within the mapped span
    +/- ``window``.  An UNMAPPABLE anchor admits any location on its original or
    rename-destination path (the conservative, logged fallback).
    """
    if mapping.outcome == OUTCOME_UNMAPPABLE:
        return location.path in {mapping.original_path, mapping.path}
    return (
        location.path == mapping.path
        and location.end >= mapping.start - window
        and location.start <= mapping.end + window
    )


@dataclass(frozen=True)
class MappedLocation:
    path: str
    start: int
    end: int
    item_id: str


@dataclass(frozen=True)
class Cluster:
    path: str
    start: int
    end: int
    item_ids: tuple[str, ...]


def find_cluster(
    per_review: Sequence[Sequence[MappedLocation]], window: int
) -> Cluster | None:
    """One connected span on one path with a mapped finding from every review.

    ``per_review`` holds the head-mapped locations of each of the K consecutive
    reviews.  Each path's ranges are split into connected components (a gap of at
    most ``window`` connects neighbours) and a component must contain a finding
    from every review; a distant unrelated finding on the same path does not
    prevent a cluster elsewhere on it.
    """
    if not per_review or any(not locations for locations in per_review):
        return None
    paths = set.intersection(*({loc.path for loc in locations} for locations in per_review))
    for path in sorted(paths):
        ranges = sorted(
            (loc.start, loc.end, loc.item_id, number)
            for number, locations in enumerate(per_review)
            for loc in locations
            if loc.path == path
        )
        components: list[list[tuple[int, int, str, int]]] = []
        span_end = None
        for entry in ranges:
            if span_end is not None and entry[0] - span_end - 1 <= window:
                components[-1].append(entry)
                span_end = max(span_end, entry[1])
            else:
                components.append([entry])
                span_end = entry[1]
        for component in components:
            if {entry[3] for entry in component} == set(range(len(per_review))):
                ids = tuple(dict.fromkeys(entry[2] for entry in component))
                return Cluster(path, component[0][0], max(e[1] for e in component), ids)
    return None


@dataclass(frozen=True)
class PrReviewRecord:
    round_number: int
    index: int
    head: str
    classification: str
    metadata: "PostedRoundMetadata"
    # Resolved via effective_new_items; ``None`` means unresolvable.
    new_items: "tuple | None" = ()


@dataclass(frozen=True)
class PrStepBackEntry:
    """A reviewer-owned PR step-back entry joined with its coder record."""

    reviewer: str
    trigger_round: int
    trigger_head: str
    path: str
    start: int
    end: int
    coder_round: int
    resulting_head: str
    record_index: int


def pr_reviews_for(
    records: Sequence["PostedRoundRecord"], reviewer: str
) -> tuple[PrReviewRecord, ...]:
    """The reviewer's latest PR review record per round, in round order."""
    ordered = sorted(records, key=lambda item: item.index)
    latest: dict[int, "PostedRoundRecord"] = {}
    for record in ordered:
        metadata = record.metadata
        if metadata.flow == "pr" and metadata.role == "reviewer" and metadata.agent == reviewer:
            latest[metadata.round_number] = record
    result = []
    for number, record in sorted(latest.items()):
        items = effective_new_items(ordered, record)
        result.append(
            PrReviewRecord(
                round_number=number,
                index=record.index,
                head=str(record.metadata.subject),
                classification=classify_review(
                    record.metadata, phase=PHASE_PR, new_items=items
                ),
                metadata=record.metadata,
                new_items=items,
            )
        )
    return tuple(result)


def pr_step_back_history(
    records: Sequence["PostedRoundRecord"], reviewer: str
) -> tuple[tuple[PrStepBackEntry, ...], bool]:
    """``(entries for the reviewer in record order, degraded)``."""
    degraded = False
    entries: list[PrStepBackEntry] = []
    for record in sorted(records, key=lambda item: item.index):
        metadata = record.metadata
        if metadata.flow != "pr" or metadata.role != "coder":
            continue
        if metadata.step_back_status == "invalid":
            degraded = True
            continue
        for entry in metadata.step_back_entries:
            if entry.get("phase") != PHASE_PR or entry.get("reviewer") != reviewer:
                continue
            anchor = entry["anchor"]
            entries.append(
                PrStepBackEntry(
                    reviewer=reviewer,
                    trigger_round=int(entry["trigger_round"]),
                    trigger_head=str(entry["trigger_head"]),
                    path=str(anchor["path"]),
                    start=int(anchor["start"]),
                    end=int(anchor["end"]),
                    coder_round=metadata.round_number,
                    resulting_head=str(metadata.subject),
                    record_index=record.index,
                )
            )
    return tuple(entries), degraded


# ``mapper(from_head, to_head, path, start, end)`` maps a range between heads.
AnchorMapper = Callable[[str, str, str, int, int], AnchorMapping]


def map_between_heads(
    mapper: AnchorMapper, from_head: str, to_head: str, path: str, start: int, end: int
) -> AnchorMapping:
    if from_head == to_head:
        return AnchorMapping(OUTCOME_SHIFTED, path, path, start, end)
    return mapper(from_head, to_head, path, start, end)


def _mandatory_owned_new_items(review: PrReviewRecord) -> list:
    mandatory = _MANDATORY_STATUSES[PHASE_PR]
    return [
        item
        for item in review.new_items or ()
        if item.status in mandatory and item.reviewer == review.metadata.agent
    ]


def _remaining_mandatory_items(review: PrReviewRecord) -> list:
    """The reviewer's mandatory items still open after this review's own record."""
    metadata = review.metadata
    mandatory = _MANDATORY_STATUSES[PHASE_PR]
    closed = {
        d.item_id
        for d in metadata.dispositions
        if d.disposition not in mandatory
    }
    dispositions = {d.item_id: d for d in metadata.dispositions}
    remaining = []
    for item in metadata.prior_items:
        if item.reviewer != metadata.agent or getattr(item, "authority", None) is not None:
            continue
        disposition = dispositions.get(item.item_id)
        # The review's own disposition decides the effective status, so a future
        # item promoted to blocking/same-pr counts and a resolved one does not.
        effective = disposition.disposition if disposition is not None else item.status
        if effective not in mandatory:
            continue
        updated = _with_effective_sub_items(item, disposition)
        if (
            disposition is not None
            and disposition.sub_item_dispositions
            and updated.sub_items
            and all(sub.status == "resolved" for sub in updated.sub_items)
        ):
            # A completing entry derives to resolved (the ledger's own semantics).
            continue
        if updated.status != effective:
            updated = dataclasses.replace(updated, status=effective)
        remaining.append(updated)
    seen = {item.item_id for item in remaining}
    remaining.extend(
        item for item in _mandatory_owned_new_items(review) if item.item_id not in seen
    )
    return remaining


def _with_effective_sub_items(item, disposition):
    """The item with this review's own sub-item dispositions applied.

    ``resolved`` closes a sub-item and ``unresolved`` reopens one, so location
    projection sees the post-review state rather than the prior ledger's.
    """
    if disposition is None or not disposition.sub_item_dispositions or not item.sub_items:
        return item
    outcomes = dict(disposition.sub_item_dispositions)
    updated = tuple(
        dataclasses.replace(
            sub,
            status={"resolved": "resolved", "unresolved": "open"}.get(
                outcomes.get(sub.sub_item_id, ""), sub.status
            ),
        )
        for sub in item.sub_items
    )
    return dataclasses.replace(item, sub_items=updated)


def _location_is_member(
    location: FindingLocation,
    source_head: str,
    head: str,
    mapping: AnchorMapping,
    window: int,
    mapper: AnchorMapper,
) -> bool:
    """Membership of a carried location, with anchor and location in ``head``'s coordinates.

    The anchor ``mapping`` is already in ``head``'s coordinates; the location is
    mapped forward from the head its review was taken on.  A location that cannot
    be mapped falls back to same-path membership against the anchor's paths.
    """
    moved = map_between_heads(
        mapper, source_head, head, location.path, location.start, location.end
    )
    if moved.mappable:
        location = dataclasses.replace(
            location, path=moved.path, start=moved.start, end=moved.end
        )
        return in_cluster(location, mapping, window)
    # The path-wide fallback belongs to an UNMAPPABLE anchor only.  A mappable anchor
    # requires the mapped window, so a location that cannot be placed (moved or
    # deleted code, a failed diff) is not declared a member by path alone.
    # Under that policy the location's known rename destination takes part too, so a
    # chain of renames still meets the anchor's original or destination path.
    return not mapping.mappable and bool(
        {location.path, moved.path} & {mapping.original_path, mapping.path}
    )


@dataclass(frozen=True)
class PrEpisodeResult:
    """Outcome of replaying the history after a PR step-back turn."""

    entry: PrStepBackEntry | None
    siblings: tuple[FindingLocation, ...] = ()
    sibling_round: int | None = None
    mapping: AnchorMapping | None = None


def derive_pr_episode(
    records: Sequence["PostedRoundRecord"],
    reviewer: str,
    *,
    window: int,
    mapper: AnchorMapper,
    current_round: int | None = None,
    current_head: str | None = None,
) -> PrEpisodeResult:
    """Replay the reviewer's reviews after its newest step-back turn.

    ``current_round`` is the round being dispatched: only a sibling introduced by
    that round's review of ``current_head`` escalates.  A sibling from an earlier
    round, or from a head the code has since moved past (an operator's pushed
    redesign or documented limitation), already stopped the run once and must not
    stop it again; it stays an open member item, which keeps the episode open until
    it is resolved.

    The episode stays active until the reviewer approves or no unresolved
    mandatory item of theirs has a location that is a member of the cluster
    (the same :func:`in_cluster` predicate escalation uses).  The first review
    that introduces a new mandatory member while the episode is active is a
    sibling and is returned for escalation.
    """
    entries, degraded = pr_step_back_history(records, reviewer)
    if degraded or not entries:
        return PrEpisodeResult(entry=None)
    entry = entries[-1]
    reviews = pr_reviews_for(records, reviewer)
    heads = {review.round_number: review.head for review in reviews}
    cache: dict[tuple[str, str, str, int, int], AnchorMapping] = {}

    def mapping_to(head: str) -> AnchorMapping:
        key = (entry.trigger_head, head, entry.path, entry.start, entry.end)
        if key not in cache:
            cache[key] = map_between_heads(
                mapper, entry.trigger_head, head, entry.path, entry.start, entry.end
            )
        return cache[key]

    last_mapping: AnchorMapping | None = None
    for review in reviews:
        if review.index <= entry.record_index:
            continue
        if review.classification == CLASS_APPROVED:
            return PrEpisodeResult(entry=None)
        if review.classification == CLASS_BLOCKING_UNRESOLVED:
            # Its new items are unknowable without a reconciliation: it can
            # neither escalate nor decide the episode's membership.
            continue
        mapping = mapping_to(review.head)
        last_mapping = mapping
        siblings = tuple(
            location
            for item in _mandatory_owned_new_items(review)
            for location in project_finding_locations(item)
            if in_cluster(location, mapping, window)
        )
        if (
            siblings
            and (current_round is None or review.round_number >= current_round)
            and (current_head is None or review.head == current_head)
        ):
            return PrEpisodeResult(entry, siblings, review.round_number, mapping)
        member_open = False
        for item in _remaining_mandatory_items(review):
            source_head = heads.get(item.source_round, review.head)
            if any(
                _location_is_member(
                    location, source_head, review.head, mapping, window, mapper
                )
                for location in project_finding_locations(item)
            ):
                member_open = True
                break
        if not member_open:
            return PrEpisodeResult(entry=None)
    return PrEpisodeResult(entry=entry, mapping=last_mapping)


@dataclass(frozen=True)
class PrClusterTrigger:
    """A tracked reviewer's K-round cluster that opens a PR step-back."""

    reviewer: str
    trigger_round: int
    trigger_head: str
    cluster: Cluster
    findings: tuple[str, ...]


def find_pr_cluster_trigger(
    records: Sequence["PostedRoundRecord"],
    reviewer: str,
    *,
    k: int,
    window: int,
    current_round: int,
    mapper: AnchorMapper,
    current_head: str | None = None,
) -> PrClusterTrigger | None:
    """The reviewer's cluster after K consecutive new-finding blocks, if any.

    Only reviews recorded after the reviewer's newest step-back turn count, so a
    closed episode re-arms from fresh blocks.  Locations that cannot be mapped to
    the newest head never contribute.
    """
    if k <= 0:
        return None
    entries, degraded = pr_step_back_history(records, reviewer)
    if degraded:
        return None
    floor = entries[-1].record_index if entries else -1
    reviews = [
        review for review in pr_reviews_for(records, reviewer) if review.index > floor
    ]
    if len(reviews) < k:
        return None
    tail = reviews[-k:]
    if tail[-1].round_number != current_round:
        return None
    if any(review.classification != CLASS_NEW_FINDING for review in tail):
        return None
    if any(
        later.round_number != earlier.round_number + 1
        for earlier, later in zip(tail, tail[1:])
    ):
        return None
    # Locations are mapped to the head being dispatched, which an operator may have
    # pushed past the newest review; unmappable ones never contribute.
    head = current_head or tail[-1].head
    per_review: list[list[MappedLocation]] = []
    for review in tail:
        mapped: list[MappedLocation] = []
        for item in _mandatory_owned_new_items(review):
            for location in project_finding_locations(item):
                mapping = map_between_heads(
                    mapper, review.head, head, location.path, location.start, location.end
                )
                if mapping.mappable:
                    mapped.append(
                        MappedLocation(mapping.path, mapping.start, mapping.end, item.item_id)
                    )
        per_review.append(mapped)
    cluster = find_cluster(per_review, window)
    if cluster is None:
        return None
    first_lines: list[str] = []
    wanted = set(cluster.item_ids)
    for review in tail:
        for item in _mandatory_owned_new_items(review):
            if item.item_id in wanted:
                line = next(
                    (part.strip() for part in item.text.splitlines() if part.strip()), ""
                )
                first_lines.append(f"[{item.item_id}] (round {review.round_number}) {line[:240]}")
    return PrClusterTrigger(reviewer, current_round, head, cluster, tuple(first_lines))


def pr_step_back_entry_payload(trigger: PrClusterTrigger) -> Mapping[str, object]:
    """The durable reviewer-owned PR entry the orchestrator records."""
    return {
        "phase": PHASE_PR,
        "reviewer": trigger.reviewer,
        "trigger_round": trigger.trigger_round,
        "trigger_head": trigger.trigger_head,
        "anchor": {
            "path": trigger.cluster.path,
            "start": trigger.cluster.start,
            "end": trigger.cluster.end,
        },
    }


def pr_sweep_entries(
    records: Sequence["PostedRoundRecord"], *, round_number: int, head_sha: str | None
) -> dict[str, PrStepBackEntry]:
    """Reviewers owed a sweep on ``head_sha``: entries of the step-back coder record
    whose resulting head is exactly this head and whose round is this round."""
    if not head_sha:
        return {}
    bound: dict[str, PrStepBackEntry] = {}
    for record in sorted(records, key=lambda item: item.index):
        metadata = record.metadata
        if (
            metadata.flow != "pr"
            or metadata.role != "coder"
            or metadata.step_back_status != "valid"
            or metadata.round_number != round_number
            or metadata.subject != head_sha
        ):
            continue
        for entry in metadata.step_back_entries:
            if entry.get("phase") != PHASE_PR:
                continue
            anchor = entry["anchor"]
            reviewer = str(entry["reviewer"])
            bound[reviewer] = PrStepBackEntry(
                reviewer=reviewer,
                trigger_round=int(entry["trigger_round"]),
                trigger_head=str(entry["trigger_head"]),
                path=str(anchor["path"]),
                start=int(anchor["start"]),
                end=int(anchor["end"]),
                coder_round=metadata.round_number,
                resulting_head=str(metadata.subject),
                record_index=record.index,
            )
    return bound


def step_back_generalization(
    records: Sequence["PostedRoundRecord"], entry: PrStepBackEntry
) -> str | None:
    """The coder's structured ``summary`` on the step-back record, if readable."""
    for record in records:
        metadata = record.metadata
        if record.index != entry.record_index or metadata.role != "coder":
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


def render_pr_step_back_coder_guidance(triggers: Sequence[PrClusterTrigger]) -> str:
    """The generalize-the-class coder instruction for one or more clusters."""
    blocks = []
    for trigger in triggers:
        cluster = trigger.cluster
        findings = "\n".join(f"  - {line}" for line in trigger.findings) or "  - (none recorded)"
        blocks.append(
            f"- {trigger.reviewer}: `{cluster.path}` lines {cluster.start}-{cluster.end} "
            f"(head `{trigger.trigger_head}`). Recent findings:\n{findings}"
        )
    return (
        "STEP-BACK TURN (orchestrator, not a reviewer finding): a reviewer has blocked "
        "several consecutive rounds, each with a new finding in the same part of the code. "
        "Do not patch only the newest finding. Name the common root of these findings, fix "
        "the whole class with one rule, and add a parametrized test that covers every branch "
        "of that rule. This replaces any instruction to keep the change small and "
        "localized. Begin the `summary` of your structured response with `Generalization:` "
        "followed by the rule you applied.\n"
        "Clusters:\n" + "\n".join(blocks) + "\n"
    )


def render_pr_sweep_guidance(entry: PrStepBackEntry) -> str:
    """The reviewer sweep instruction, bound to the step-back head only."""
    return (
        "Sweep (orchestrator, not a reviewer finding; this round only): the coder just "
        f"generalized a class of findings you raised in `{entry.path}` (lines "
        f"{entry.start}-{entry.end} at head `{entry.trigger_head}`, which may have "
        "shifted). Enumerate ALL remaining instances of that pattern in the code under "
        "review, not just the first, giving a `path:line` reference for each. Shape the "
        "output by count: no remaining instance means no finding for the pattern; one "
        "instance is one plain finding; 2-12 instances are one finding whose `sub_items` "
        "list each instance with its `path:line`; more than 12 are several findings, "
        "each a plain finding or holding at most 12 `sub_items`, so the sweep stays "
        "exhaustive. This guidance does not change any other review rule.\n"
    )


def render_pr_step_back_human_decision(
    *,
    reviewer: str,
    entry: PrStepBackEntry,
    mapping: AnchorMapping | None,
    siblings: Sequence[FindingLocation],
    window: int,
    generalization: str | None,
    sibling_round: int | None,
) -> str:
    """Human-decision diagnostic after a step-back that still produced a sibling."""
    excerpt = (generalization or "").strip()
    if len(excerpt) > ALTERNATIVE_EXCERPT_LIMIT:
        excerpt = excerpt[: ALTERNATIVE_EXCERPT_LIMIT - 1].rstrip() + "…"
    if mapping is None:
        mapped = "(not mapped)"
    elif mapping.mappable:
        mapped = (
            f"{mapping.outcome} to `{mapping.path}` lines {mapping.start}-{mapping.end} "
            f"(+/- {window})"
        )
    else:
        mapped = (
            f"{mapping.outcome} ({mapping.reason}); falling back to any finding on "
            f"`{mapping.original_path}` or `{mapping.path}`"
        )
    sibling_text = "; ".join(
        dict.fromkeys(
            f"[{loc.item_id}{'/' + loc.sub_item_id if loc.sub_item_id else ''}] "
            f"`{loc.path}:{loc.start}-{loc.end}`"
            for loc in siblings
        )
    )
    return (
        "human decision required: the coder generalized a class of findings at round "
        f"{entry.coder_round} (a step-back turn after {reviewer} blocked consecutive rounds "
        f"on `{entry.path}` lines {entry.start}-{entry.end}), and {reviewer} still raised a "
        f"new sibling finding in round {sibling_round}: {sibling_text}. Cluster mapping: "
        f"{mapped}. No further coder turn was invoked. Coder generalization (excerpt): "
        f"{excerpt or '(not recorded)'} Decide one of: continue (rerun with "
        "--pr-step-back-rounds 0, which also disables step-back detection), accept a stated "
        "limitation (document it, resolve or defer the sibling, push the commit and rerun), or "
        "redesign the affected code (push the redesign and rerun). After a push, a rerun does "
        "not stop again on this round's sibling; only a new sibling in a later review does."
    )
