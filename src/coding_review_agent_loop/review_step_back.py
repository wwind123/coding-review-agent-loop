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
import difflib
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
from .protocol import PROVENANCE_PRE_EXISTING, PROVENANCE_REVISION_INTRODUCED

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
# Episode anchor display bounds (#1278).  They cap rendering only; membership
# always uses the complete dissolved-ID tuple.
ANCHOR_NOTE_LIMIT = 240
_ANCHOR_NOTED_IDS = 12
_ANCHOR_SHOWN_IDS = 100
_STOP_DISSOLVED_LIMIT = 20


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


@dataclass(frozen=True)
class _PrimaryWalk:
    """The primary's counted reviews and retirement facts from plan records.

    Shared by the plan step-back and the plan sweep (#1505) so their retirement
    rules cannot drift; only the per-review classifier differs.
    """

    ordered: tuple["PostedRoundRecord", ...]
    degraded: bool
    boundary_index: int
    retired_through: int
    reset_indices: tuple[int, ...]
    all_checkpoints: tuple[tuple[int, str | None], ...]
    deferrals: tuple[tuple[int, int, str | None], ...]
    reviews: tuple[PrimaryReview, ...]


def _walk_primary_reviews(
    records: Sequence["PostedRoundRecord"],
    *,
    primary: str,
    panel_opening_index: int | None,
    current_issue_digest: str | None,
    classify: Callable[[Sequence["PostedRoundRecord"], "PostedRoundRecord"], str],
) -> _PrimaryWalk:
    """Walk the primary's latest-per-round reviews under the shared retirement rules.

    Retirement mirrors the primary stall streak: a review whose latest preceding
    same-round checkpoint recorded a different issue digest retires itself and
    everything before it, and the operator reset marker retires the rounds
    *before* the checkpoint's round.  A panel opening, an invalid scheduler
    record or a phase-advance record ends the primary phase, so nothing at or
    before it counts.  ``classify(ordered, record)`` names each counted review.
    """
    ordered = tuple(sorted(records, key=lambda item: item.index))
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
                classification=classify(ordered, record),
            )
        )
    return _PrimaryWalk(
        ordered=ordered,
        degraded=degraded,
        boundary_index=boundary_index,
        retired_through=retired_through,
        reset_indices=tuple(reset_indices),
        all_checkpoints=tuple(all_checkpoints),
        deferrals=tuple(deferrals),
        reviews=tuple(counted),
    )


def derive_plan_step_back_state(
    records: Sequence["PostedRoundRecord"],
    *,
    primary: str,
    panel_opening_index: int | None = None,
    current_issue_digest: str | None = None,
) -> StepBackState:
    """Derive the primary's streak inputs and active episode from plan records.

    Retirement follows :func:`_walk_primary_reviews`.  An approval by the
    primary closes the episode.
    """
    walk = _walk_primary_reviews(
        records,
        primary=primary,
        panel_opening_index=panel_opening_index,
        current_issue_digest=current_issue_digest,
        classify=lambda ordered, record: classify_review(
            record.metadata, new_items=effective_new_items(ordered, record)
        ),
    )
    ordered = walk.ordered
    degraded = walk.degraded
    boundary_index = walk.boundary_index
    retired_through = walk.retired_through
    reset_indices = walk.reset_indices
    all_checkpoints = walk.all_checkpoints
    deferrals = list(walk.deferrals)
    reviews = walk.reviews

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
    # An issue edit retires every deferral recorded at or before the latest primary
    # checkpoint (marked or not) whose issue digest differs from the current one.
    if current_issue_digest is not None:
        mismatched = [
            index
            for index, digest in all_checkpoints
            if digest is not None and digest != current_issue_digest
        ]
        if mismatched:
            latest = max(mismatched)
            deferrals = [entry for entry in deferrals if entry[0] > latest]
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


# ---------------------------------------------------------------------------
# Plan sweep trigger and finding provenance (#1505)
#
# Independent of the plan-growth gate: once the primary has blocked K
# consecutive primary-phase rounds with new mandatory findings, its next review
# is asked for one exhaustive pass.  The counted-review walk and its retirement
# rules are the step-back's (:func:`_walk_primary_reviews`); only the counting
# predicate differs.  It never stops a run and has no episode.
# ---------------------------------------------------------------------------


def eligible_new_mandatory_items(new_items: object, *, reviewer: str) -> tuple:
    """The reviewer-owned mandatory new items that are not repeats (#1505).

    The single eligibility filter shared by the sweep trigger and the history
    export.  ``future`` items and items marked ``repeat_of`` at minting never
    count.
    """
    mandatory = _MANDATORY_STATUSES[PHASE_PLAN]
    return tuple(
        item
        for item in (new_items or ())
        if item.reviewer == reviewer
        and item.status in mandatory
        and getattr(item, "repeat_of", None) is None
    )


def count_sweep_items(new_items: object, *, reviewer: str, first_review: bool) -> int:
    """Trigger-only count of a review's new mandatory findings.

    On a later review, findings the reviewer tagged ``revision_introduced`` are
    excluded; untagged findings count, so omitting the tag cannot suppress the
    trigger.  On the reviewer's first plan review every eligible finding counts,
    whatever its tag.
    """
    eligible = eligible_new_mandatory_items(new_items, reviewer=reviewer)
    if first_review:
        return len(eligible)
    return sum(
        1 for item in eligible
        if getattr(item, "provenance", None) != PROVENANCE_REVISION_INTRODUCED
    )


def classify_sweep_review(
    metadata: "PostedRoundMetadata", new_items: object, *, first_review: bool
) -> str:
    """Classify one primary review for the sweep streak.

    ``new-finding`` needs at least one counted item; any other block is
    ``repeat-only``.  ``new_items`` of ``None`` (an unresolvable publication
    record) is ``blocking-unresolved``.
    """
    if metadata.state == "approved":
        return CLASS_APPROVED
    if metadata.state != "blocking":
        return CLASS_OTHER
    if new_items is None:
        return CLASS_BLOCKING_UNRESOLVED
    if count_sweep_items(new_items, reviewer=metadata.agent, first_review=first_review) > 0:
        return CLASS_NEW_FINDING
    return CLASS_REPEAT_ONLY


def first_plan_review_round(
    records: Sequence["PostedRoundRecord"], *, reviewer: str
) -> int | None:
    """The lowest round of any plan reviewer record by ``reviewer``, if any."""
    rounds = [
        record.metadata.round_number
        for record in records
        if record.metadata.role == "reviewer" and record.metadata.agent == reviewer
    ]
    return min(rounds) if rounds else None


@dataclass(frozen=True)
class PlanSweepState:
    """History-derived sweep streak of the primary plan reviewer (#1505)."""

    degraded: bool
    reviews: tuple[PrimaryReview, ...]

    def _streak(self, round_number: int) -> list[PrimaryReview]:
        streak: list[PrimaryReview] = []
        for review in reversed(self.reviews):
            if review.round_number >= round_number:
                continue
            # The streak must reach back from the round immediately before
            # ``round_number``: an uncounted round anywhere, the newest one
            # included, ends it.
            expected = (streak[-1].round_number if streak else round_number) - 1
            if review.classification != CLASS_NEW_FINDING or review.round_number != expected:
                break
            streak.append(review)
        return streak

    def streak_before(self, round_number: int) -> int:
        """Consecutive newest-first ``new-finding`` reviews in rounds before ``round_number``.

        A repeat-only, approved, unresolvable or other review ends it, and so
        does a round with no counted review.
        """
        return len(self._streak(round_number))

    def streak_first_round(self, round_number: int) -> int | None:
        """The earliest round of the streak counted by :meth:`streak_before`."""
        streak = self._streak(round_number)
        return streak[-1].round_number if streak else None


def derive_plan_sweep_state(
    records: Sequence["PostedRoundRecord"],
    *,
    primary: str,
    panel_opening_index: int | None = None,
    current_issue_digest: str | None = None,
) -> PlanSweepState:
    """Derive the primary's sweep streak with the step-back's retirement rules."""
    first_round = first_plan_review_round(records, reviewer=primary)
    walk = _walk_primary_reviews(
        records,
        primary=primary,
        panel_opening_index=panel_opening_index,
        current_issue_digest=current_issue_digest,
        classify=lambda ordered, record: classify_sweep_review(
            record.metadata,
            effective_new_items(ordered, record),
            first_review=record.metadata.round_number == first_round,
        ),
    )
    return PlanSweepState(degraded=walk.degraded, reviews=walk.reviews)


def plan_sweep_due(
    state: PlanSweepState | None,
    *,
    round_number: int,
    sweep_rounds: int,
) -> bool:
    """Whether round ``round_number``'s primary review gets the sweep request."""
    if sweep_rounds <= 0 or state is None or state.degraded:
        return False
    return state.streak_before(round_number) >= sweep_rounds


PLAN_SWEEP_CLASS_LIMIT = 12


def plan_sweep_finding_classes(
    records: Sequence["PostedRoundRecord"],
    *,
    primary: str,
    first_round: int,
    limit: int = PLAN_SWEEP_CLASS_LIMIT,
) -> tuple[str, ...]:
    """The primary's mandatory finding classes since ``first_round``, repeats skipped.

    The newest ``limit`` entries are kept; an ``and N more`` line names the rest.
    """
    lines = _mandatory_plan_finding_lines(
        records, primary=primary, first_round=first_round, skip_repeats=True
    )
    if len(lines) <= limit:
        return tuple(lines)
    hidden = len(lines) - limit
    return (*lines[-limit:], f"and {hidden} more earlier finding(s) not shown")


def render_plan_sweep_guidance(classes: Sequence[str]) -> str:
    """The primary's sweep request, shared by the full and compact review prompts."""
    listed = "\n".join(f"- {line}" for line in classes) or "- (none recorded)"
    return (
        "PRIMARY SWEEP REQUEST (orchestrator guidance, not a reviewer finding; this "
        "round only): your recent reviews of this plan each blocked on new mandatory "
        "findings. Finding classes you have already raised:\n"
        f"{listed}\n"
        "Make one exhaustive pass over the whole current plan and report every "
        "remaining in-scope defect in this review, rather than across later rounds. "
        "Explicitly check the siblings of the listed classes: other entry points, call "
        "sites, tests, docs and flows of the same kind. The blocking-scope rules, the "
        "same-plan/future classification and the approval rules are unchanged: "
        "out-of-scope findings stay future follow-ups or are omitted, and this sweep "
        "is not a reason to block an otherwise acceptable plan. Give each new "
        "mandatory finding its `provenance` against the previous-plan baseline below "
        "when one is supplied.\n"
    )


BASELINE_STATUS_FIRST_REVIEW = "first-review"
BASELINE_STATUS_AVAILABLE = "available"
BASELINE_STATUS_UNAVAILABLE = "unavailable"

# Rendered baseline forms (#1516).  The four comparison forms are tried in
# this order; ``first-review`` and ``identical`` are informational.
BASELINE_FORM_FIRST_REVIEW = "first-review"
BASELINE_FORM_IDENTICAL = "identical"
BASELINE_FORM_DIFF = "diff"
BASELINE_FORM_COMPACT_DIFF = "compact-diff"
BASELINE_FORM_FULL = "full-baseline"
BASELINE_FORM_UNAVAILABLE = "unavailable"
BASELINE_FORMS = frozenset({
    BASELINE_FORM_FIRST_REVIEW,
    BASELINE_FORM_IDENTICAL,
    BASELINE_FORM_DIFF,
    BASELINE_FORM_COMPACT_DIFF,
    BASELINE_FORM_FULL,
    BASELINE_FORM_UNAVAILABLE,
})
# Hard budget for any baseline payload (``--plan-provenance-baseline-max-chars``;
# 0 disables the baseline).  The diff forms get max(floor, half the current
# plan): a diff larger than that costs nearly as much as the full baseline and
# is harder to read.  Below a 24k plan the floor keeps the #1505 behaviour.
DEFAULT_PROVENANCE_BASELINE_MAX_CHARS = 120000
PROVENANCE_DIFF_MIN_BUDGET = 12000
PROVENANCE_BASELINE_DIFF_LIMIT = PROVENANCE_DIFF_MIN_BUDGET
_COMPACT_CONTEXT_LINE_CHARS = 200
_COMPACT_CONTEXT_UNIT_CHARS = 160
_COMPACT_WORD_PASS_CHARS = 600
_COMPACT_WORD_CONTEXT = 12


@dataclass(frozen=True)
class PlanBaseline:
    """The plan a reviewer last reviewed, resolved from durable records (#1505)."""

    status: str
    round_number: int | None = None
    subject: str | None = None
    text: str | None = None
    reason: str = ""


def reviewer_previous_plan_baseline(
    records: Sequence["PostedRoundRecord"], *, reviewer: str, before_round: int
) -> PlanBaseline:
    """Resolve ``reviewer``'s previous reviewed plan from plan records only.

    The reviewer's latest plan reviewer record before ``before_round`` names
    the plan subject it reviewed; the canonical text is the newest planner
    record of that subject whose canonical plan hashes to it.  No earlier
    review is a first review; an unmatched subject or hash is unavailable.
    """
    from .round_state import _plan_subject

    previous = None
    for record in sorted(records, key=lambda item: item.index):
        metadata = record.metadata
        if (
            metadata.role == "reviewer"
            and metadata.agent == reviewer
            and metadata.round_number < before_round
            and (previous is None or metadata.round_number >= previous.metadata.round_number)
        ):
            previous = record
    if previous is None:
        return PlanBaseline(status=BASELINE_STATUS_FIRST_REVIEW)
    subject = previous.metadata.subject
    round_number = previous.metadata.round_number
    text = None
    for record in sorted(records, key=lambda item: item.index):
        metadata = record.metadata
        if (
            metadata.role == "coder"
            and metadata.canonical_plan is not None
            and subject
            and metadata.subject == subject
            and _plan_subject(metadata.canonical_plan) == subject
        ):
            text = metadata.canonical_plan
    if text is None:
        return PlanBaseline(
            status=BASELINE_STATUS_UNAVAILABLE,
            round_number=round_number,
            subject=subject,
            reason=f"the plan you reviewed in round {round_number} is not recoverable",
        )
    return PlanBaseline(
        status=BASELINE_STATUS_AVAILABLE,
        round_number=round_number,
        subject=subject,
        text=text,
    )


@dataclass(frozen=True)
class RenderedBaseline:
    """The selected baseline form, its payload size and its prompt block (#1516).

    ``chars`` counts the payload only (the diff or the previous plan text),
    never the framing; it is 0 for the forms without a payload.
    """

    form: str
    chars: int
    block: str


_BASELINE_HEADING = "Finding provenance baseline (orchestrator, not a reviewer finding): "
_UNIT_BOUNDARY = re.compile(r"[.!?;]\s+| \| ")
_WORD_TOKEN = re.compile(r"\S+\s*|\s+")
# A fence-like hyphen run inside baseline payload content (#1517): three or
# more hyphens, optional blanks, then a whole-word BEGIN or END.  Such runs
# are escaped everywhere in the payload.  A unified-diff ``-`` marker sits
# directly against its content and extends a leading run by one hyphen, so a
# two-hyphen content prefix after that marker is escaped as well (#1519).
# Together these keep every rendered payload line from beginning with a
# fence-like run of three or more hyphens.
_FENCE_LIKE_RUN = re.compile(r"-{3,}(?=[ \t]*(?:BEGIN|END)\b)", re.IGNORECASE)
# Exactly two leading hyphens before a whole-word BEGIN or END (#1519).
_MARKER_ADJACENT_RUN = re.compile(r"--(?!-)(?=[ \t]*(?:BEGIN|END)\b)", re.IGNORECASE)
_BASELINE_ESCAPE_NOTICE = (
    "Payload text that resembles a BEGIN/END fence is shown with a backslash before "
    "its hyphens (after the diff marker, in diff lines); only the unescaped fence "
    "lines above and below delimit the baseline data.\n"
)


def provenance_diff_budget(current_plan: str, *, max_chars: int) -> int:
    """The diff-form budget: max(floor, half the current plan), capped by ``max_chars``."""
    return min(max_chars, max(PROVENANCE_DIFF_MIN_BUDGET, len(current_plan) // 2))


def split_baseline_units(line: str) -> list[str]:
    """Sentence and table-cell units of ``line``; the units concatenate back to it.

    A unit ends after ``.``, ``!``, ``?`` or ``;`` plus its following
    whitespace, or after a `` | `` cell separator.
    """
    units: list[str] = []
    start = 0
    for match in _UNIT_BOUNDARY.finditer(line):
        units.append(line[start:match.end()])
        start = match.end()
    if start < len(line):
        units.append(line[start:])
    return units


def split_baseline_words(text: str) -> list[str]:
    """Word tokens of ``text`` with their trailing whitespace; they concatenate back."""
    return _WORD_TOKEN.findall(text)


def _clip_unit(text: str, limit: int, keep: str) -> str:
    """``text`` cut to ``limit`` characters around the kept side, marked with an ellipsis."""
    text = text.strip()
    if len(text) <= limit:
        return text
    if keep == "head":
        return text[:limit] + "…"
    if keep == "tail":
        return "…" + text[-limit:]
    half = limit // 2
    return text[:half] + " … " + text[-half:]


def _context_keep(index: int, count: int) -> str:
    """Leading context keeps its tail, trailing context its head, inner context both ends."""
    if index == 0 and count > 1:
        return "tail"
    if index == count - 1 and count > 1:
        return "head"
    return "both"


def _compact_unit_change(old: str, new: str) -> list[str]:
    """``-``/``+`` lines for one changed run of units; long runs get a word pass."""
    if max(len(old), len(new)) <= _COMPACT_WORD_PASS_CHARS:
        lines = []
        if old.strip():
            lines.append(f"- {old.strip()}")
        if new.strip():
            lines.append(f"+ {new.strip()}")
        return lines
    old_words = split_baseline_words(old)
    new_words = split_baseline_words(new)
    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    lines: list[str] = []
    for group in matcher.get_grouped_opcodes(_COMPACT_WORD_CONTEXT):
        i1, i2 = group[0][1], group[-1][2]
        j1, j2 = group[0][3], group[-1][4]
        for prefix, words, start, end in (
            ("-", old_words, i1, i2), ("+", new_words, j1, j2)
        ):
            fragment = "".join(words[start:end]).strip()
            if not fragment:
                continue
            lead = "…" if start > 0 else ""
            tail = "…" if end < len(words) else ""
            lines.append(f"{prefix} {lead}{fragment}{tail}")
    return lines


def _compact_line_change(old: str, new: str) -> list[str]:
    """The changed sentences or cells of one replaced line, with one unit of context."""
    old_units = split_baseline_units(old)
    new_units = split_baseline_units(new)
    matcher = difflib.SequenceMatcher(None, old_units, new_units, autojunk=False)
    lines: list[str] = []
    for group in matcher.get_grouped_opcodes(1):
        for index, (tag, i1, i2, j1, j2) in enumerate(group):
            if tag == "equal":
                keep = _context_keep(index, len(group))
                for unit in old_units[i1:i2]:
                    if unit.strip():
                        lines.append(
                            f"  {_clip_unit(unit, _COMPACT_CONTEXT_UNIT_CHARS, keep)}"
                        )
                continue
            lines.extend(
                _compact_unit_change("".join(old_units[i1:i2]), "".join(new_units[j1:j2]))
            )
    return lines


def compact_plan_diff(previous: str, current: str, *, round_number: int | None) -> str:
    """A deterministic sentence/cell/word-level diff of two plan texts (#1516).

    Lines are matched first; a replaced line pair shows only its changed
    sentences or table cells (a long changed unit only its changed words),
    while wholly added or removed lines are shown in full.  Each hunk keeps at
    most one neighbouring unchanged line of context, clipped.
    """
    old_lines = previous.strip().splitlines()
    new_lines = current.strip().splitlines()
    out = [
        f"--- plan reviewed in round {round_number}",
        "+++ current plan (compact: changed sentences only)",
    ]
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for group in matcher.get_grouped_opcodes(1):
        for index, (tag, i1, i2, j1, j2) in enumerate(group):
            if tag == "equal":
                keep = _context_keep(index, len(group))
                out.extend(
                    f"  {_clip_unit(line, _COMPACT_CONTEXT_LINE_CHARS, keep)}"
                    for line in old_lines[i1:i2]
                )
                continue
            paired = min(i2 - i1, j2 - j1) if tag == "replace" else 0
            for offset in range(paired):
                out.append(f"@@ line {i1 + offset + 1}->{j1 + offset + 1} @@")
                out.extend(
                    _compact_line_change(old_lines[i1 + offset], new_lines[j1 + offset])
                )
            if i1 + paired < i2 or j1 + paired < j2:
                out.append(f"@@ line {i1 + paired + 1}->{j1 + paired + 1} @@")
                out.extend(f"- {line}" for line in old_lines[i1 + paired:i2])
                out.extend(f"+ {line}" for line in new_lines[j1 + paired:j2])
    return "\n".join(out)


def _neutralize_fence_content(text: str) -> str:
    """``text`` with a backslash before every fence-like hyphen run (#1517).

    Text without such a run is returned byte-identical.
    """
    return _FENCE_LIKE_RUN.sub(lambda match: "\\" + match.group(0), text)


def _neutralize_marker_content(marker: str, content: str) -> str:
    """``content`` neutralized for display after diff ``marker`` (#1517, #1519).

    Every fence-like run in ``content`` is escaped.  When the marker ends in a
    hyphen, a two-hyphen content prefix before BEGIN/END would render as a
    three-hyphen run, so it gets one leading backslash too.  The general run
    needs three hyphens and the prefix pattern exactly two, so no run is
    escaped twice.
    """
    escaped = _neutralize_fence_content(content)
    if marker.endswith("-") and _MARKER_ADJACENT_RUN.match(content):
        return "\\" + escaped
    return escaped


def _neutralize_diff_payload(diff: str, *, marker_width: int) -> str:
    """Escape fence-like content in a baseline diff, keeping its markers (#1517).

    The first two lines are the renderer-owned file headers and ``@@`` lines
    are hunk headers; both stay verbatim.  Every other line keeps its first
    ``marker_width`` characters (the diff marker) in place and has only the
    content after them neutralized: every fence-like run of three or more
    hyphens, plus a two-hyphen content prefix directly after a ``-`` marker
    (#1519).  So no payload line after the file headers begins with a
    fence-like run of three or more hyphens.
    """
    lines = diff.split("\n")
    out = lines[:2]
    for line in lines[2:]:
        if line.startswith("@@"):
            out.append(line)
        else:
            marker = line[:marker_width]
            out.append(marker + _neutralize_marker_content(marker, line[marker_width:]))
    return "\n".join(out)


def _escape_notice(raw: str, escaped: str) -> str:
    return _BASELINE_ESCAPE_NOTICE if escaped != raw else ""


def _baseline_tagging_sentence(round_number: int | None, *, shown: str) -> str:
    return (
        f"Tag each new mandatory finding `{PROVENANCE_REVISION_INTRODUCED}` when a change "
        f"{shown} introduced the defect, or `{PROVENANCE_PRE_EXISTING}` when "
        f"it was already present in the plan you reviewed in round {round_number}.\n"
    )


def _unavailable_baseline(reason: str) -> RenderedBaseline:
    return RenderedBaseline(
        form=BASELINE_FORM_UNAVAILABLE,
        chars=0,
        block=(
            _BASELINE_HEADING
            + f"the comparison baseline is unavailable ({reason}); omit `provenance` on "
            "every finding, leaving it untagged, rather than guess.\n"
        ),
    )


def select_provenance_baseline(
    baseline: PlanBaseline | None,
    current_plan: str,
    *,
    max_chars: int = DEFAULT_PROVENANCE_BASELINE_MAX_CHARS,
) -> RenderedBaseline | None:
    """Choose the previous-plan baseline form for one plan review (#1516).

    Tried in order: the unified line diff and the compact diff, each within
    :func:`provenance_diff_budget`, then the full previous plan within
    ``max_chars``.  The unavailable form remains only for an unrecoverable
    plan, ``max_chars == 0``, or a baseline over the hard budget.  Pure and
    deterministic, so a resumed round recomputes the same form.
    """
    if baseline is None:
        return None
    if baseline.status == BASELINE_STATUS_FIRST_REVIEW:
        return RenderedBaseline(
            form=BASELINE_FORM_FIRST_REVIEW,
            chars=0,
            block=(
                _BASELINE_HEADING
                + "this is your first plan review of this issue, so omit `provenance` "
                "on every finding.\n"
            ),
        )
    if baseline.status != BASELINE_STATUS_AVAILABLE or baseline.text is None:
        return _unavailable_baseline(baseline.reason or "no previous plan was recorded")
    round_number = baseline.round_number
    if baseline.text.strip() == current_plan.strip():
        return RenderedBaseline(
            form=BASELINE_FORM_IDENTICAL,
            chars=0,
            block=(
                _BASELINE_HEADING
                + f"the current plan is identical to the plan you reviewed in round "
                f"{round_number}, so any new mandatory finding is "
                f"`{PROVENANCE_PRE_EXISTING}`.\n"
            ),
        )
    if max_chars <= 0:
        return _unavailable_baseline(
            "the provenance baseline is disabled by "
            "--plan-provenance-baseline-max-chars 0"
        )
    diff_budget = provenance_diff_budget(current_plan, max_chars=max_chars)
    unified = "\n".join(
        difflib.unified_diff(
            baseline.text.strip().splitlines(),
            current_plan.strip().splitlines(),
            fromfile=f"plan reviewed in round {round_number}",
            tofile="current plan",
            lineterm="",
            n=3,
        )
    )
    diff = _neutralize_diff_payload(unified, marker_width=1)
    if len(diff) <= diff_budget:
        return RenderedBaseline(
            form=BASELINE_FORM_DIFF,
            chars=len(diff),
            block=(
                _BASELINE_HEADING
                + f"you last reviewed the plan of round {round_number}. Changes since "
                "then, as a unified diff:\n"
                + _escape_notice(unified, diff)
                + "----- BEGIN BASELINE DIFF -----\n"
                f"{diff}\n"
                "----- END BASELINE DIFF -----\n"
                + _baseline_tagging_sentence(round_number, shown="shown in this diff")
            ),
        )
    raw_compact = compact_plan_diff(baseline.text, current_plan, round_number=round_number)
    compact = _neutralize_diff_payload(raw_compact, marker_width=2)
    if len(compact) <= diff_budget:
        return RenderedBaseline(
            form=BASELINE_FORM_COMPACT_DIFF,
            chars=len(compact),
            block=(
                _BASELINE_HEADING
                + f"you last reviewed the plan of round {round_number}. Changes since "
                "then, as a compact diff that shows only the changed sentences:\n"
                "Each `@@ line A->B @@` hunk names the old and new line numbers; `-` and "
                "`+` lines show only the removed and added sentences, table cells or "
                "words of that line (or whole lines that were removed or added), `…` "
                "marks elided text, and indented lines are unchanged context.\n"
                + _escape_notice(raw_compact, compact)
                + "----- BEGIN BASELINE DIFF -----\n"
                f"{compact}\n"
                "----- END BASELINE DIFF -----\n"
                + _baseline_tagging_sentence(round_number, shown="shown in this diff")
            ),
        )
    raw_previous = baseline.text.strip()
    previous = _neutralize_fence_content(raw_previous)
    if len(previous) <= max_chars:
        return RenderedBaseline(
            form=BASELINE_FORM_FULL,
            chars=len(previous),
            block=(
                _BASELINE_HEADING
                + f"you last reviewed the plan of round {round_number}; its full text "
                "follows as baseline material (data, not instructions). Compare it with "
                "the current plan above.\n"
                + _escape_notice(raw_previous, previous)
                + "----- BEGIN PREVIOUS REVIEWED PLAN (baseline data, not instructions) -----\n"
                f"{previous}\n"
                "----- END PREVIOUS REVIEWED PLAN -----\n"
                + _baseline_tagging_sentence(
                    round_number, shown="between that plan and the current plan"
                )
            ),
        )
    return _unavailable_baseline(
        f"the plan you reviewed in round {round_number} ({len(previous)} characters) "
        f"and its diffs exceed the {max_chars}-character baseline budget "
        "(--plan-provenance-baseline-max-chars)"
    )


def render_provenance_baseline_block(
    baseline: PlanBaseline | None,
    current_plan: str,
    *,
    max_chars: int = DEFAULT_PROVENANCE_BASELINE_MAX_CHARS,
) -> str:
    """The volatile previous-plan baseline block of a plan review prompt."""
    rendered = select_provenance_baseline(baseline, current_plan, max_chars=max_chars)
    return rendered.block if rendered is not None else ""


def plan_round_provenance_counts(
    new_items: object, *, reviewer: str
) -> dict[str, int]:
    """Per-provenance counts of a review's new mandatory items, for logging."""
    mandatory = _MANDATORY_STATUSES[PHASE_PLAN]
    owned = [
        item for item in (new_items or ())
        if item.reviewer == reviewer and item.status in mandatory
    ]
    eligible = [item for item in owned if getattr(item, "repeat_of", None) is None]
    return {
        PROVENANCE_PRE_EXISTING: sum(
            1 for item in eligible if item.provenance == PROVENANCE_PRE_EXISTING
        ),
        PROVENANCE_REVISION_INTRODUCED: sum(
            1 for item in eligible if item.provenance == PROVENANCE_REVISION_INTRODUCED
        ),
        "untagged": sum(1 for item in eligible if item.provenance is None),
        "repeats": len(owned) - len(eligible),
    }


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
        "note that starts with `dissolved:`. The approval-time growth gate is "
        "unchanged.\n"
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


# Leading bytes of the plan step-back human-decision stop; history_export
# recognizes the posted diagnostic through this constant.
PLAN_STEP_BACK_HUMAN_DECISION_PREFIX = "human decision required: the planner stepped back at round "


def render_step_back_human_decision(
    *,
    phase: str,
    measurements: str,
    step_back_round: int,
    blocks: int,
    threshold: int,
    alternative: str | None,
    dissolved: Sequence[str] = (),
    reintroduced: Sequence[tuple[str, int, str]] | None = None,
    post_review: bool = False,
) -> str:
    """Human-decision-required diagnostic after a step-back that did not converge.

    ``reintroduced`` is ``None`` for the legacy message.  When provided (the
    episode anchor was derived), the message also reports the dissolved items and
    the planner-declared reintroductions (#1278).
    """
    excerpt = (alternative or "").strip()
    if len(excerpt) > ALTERNATIVE_EXCERPT_LIMIT:
        excerpt = excerpt[: ALTERNATIVE_EXCERPT_LIMIT - 1].rstrip() + "…"
    if phase != PHASE_PLAN:
        raise ValueError(f"unsupported step-back phase: {phase}")
    if post_review and reintroduced is None:
        raise ValueError("post_review requires the episode anchor report")
    timing = "No reviewer and no planner turn were invoked. "
    anchor_report = ""
    if reintroduced is not None:
        if post_review:
            timing = (
                "This round's reviews were posted; no further planner or reviewer "
                "turn will be invoked. "
            )
        anchor_report = _render_anchor_report(dissolved, reintroduced) + " "
    return (
        PLAN_STEP_BACK_HUMAN_DECISION_PREFIX
        + f"{step_back_round} (a simplify-or-re-scope revision), and the primary plan "
        f"reviewer has since blocked {blocks} round(s), reaching "
        f"--plan-step-back-escalation-rounds {threshold} (the stall stop is deferred "
        "while a step-back is pending or in progress). "
        f"{timing}"
        f"{measurements} Latest alternative (excerpt of the step-back candidate "
        f"summary): {excerpt or '(not recorded)'} {anchor_report}Decide one of: continue patching "
        "(rerun with --plan-step-back-rounds 0, or with --plan-reset-stall-streak, "
        "which also ends the episode); adopt the simpler alternative (narrow the "
        "issue text; an issue edit ends the episode); or split (rerun with "
        "--plan-execution-mode auto, or re-file the issue as staged work)."
    )


def _render_anchor_report(
    dissolved: Sequence[str], reintroduced: Sequence[tuple[str, int, str]]
) -> str:
    if dissolved:
        shown = ", ".join(dissolved[:_STOP_DISSOLVED_LIMIT])
        extra = len(dissolved) - _STOP_DISSOLVED_LIMIT
        tail = f", (+{extra} more)" if extra > 0 else ""
        dissolved_text = f"{len(dissolved)} item(s) [{shown}{tail}]"
    else:
        dissolved_text = "(not recorded)"
    if reintroduced:
        entries = "; ".join(
            f"[{item_id}] at round {round_number} ({reason or 'no reason given'})"
            for item_id, round_number, reason in reintroduced
        )
        reintroduced_text = entries
    else:
        reintroduced_text = (
            "none declared (undeclared reintroductions are not detected)"
        )
    return (
        f"Dissolved by the step-back: {dissolved_text}. "
        f"Reintroduced since the step-back (planner-declared): {reintroduced_text}."
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
    return tuple(
        _mandatory_plan_finding_lines(records, primary=primary, first_round=first_round)[-limit:]
    )


def _mandatory_plan_finding_lines(
    records: Sequence["PostedRoundRecord"],
    *,
    primary: str,
    first_round: int,
    skip_repeats: bool = False,
) -> list[str]:
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
            if skip_repeats and getattr(item, "repeat_of", None) is not None:
                continue
            if item.status in mandatory and item.reviewer == primary:
                first_line = next((line.strip() for line in item.text.splitlines() if line.strip()), "")
                lines.append(f"[{item.item_id}] (round {number}) {first_line[:240]}")
    return lines


def _step_back_candidate_payload(
    records: Sequence["PostedRoundRecord"], episode: StepBackEntry
) -> Mapping[str, object] | None:
    """The newest candidate-round coder record's structured payload, if readable."""
    for record in sorted(records, key=lambda item: item.index, reverse=True):
        metadata = record.metadata
        if metadata.role != "coder" or metadata.round_number != episode.candidate_round:
            continue
        return _decode_payload(metadata.raw_structured_coder_response)
    return None


def _decode_payload(raw: object) -> Mapping[str, object] | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        payload, _end = json.JSONDecoder().raw_decode(raw.lstrip())
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def step_back_alternative_summary(
    records: Sequence["PostedRoundRecord"], episode: StepBackEntry
) -> str | None:
    """The step-back candidate's structured ``summary``, or ``None`` if unreadable."""
    payload = _step_back_candidate_payload(records, episode)
    summary = payload.get("summary") if payload is not None else None
    return summary if isinstance(summary, str) else None


@dataclass(frozen=True)
class PlanStepBackAnchor:
    """The active episode's simplified design, derived from history only (#1278)."""

    candidate_round: int
    trade_offs: str | None
    dissolved_ids: tuple[str, ...]
    dissolved_notes: Mapping[str, str]


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def plan_step_back_anchor(
    records: Sequence["PostedRoundRecord"], episode: StepBackEntry
) -> PlanStepBackAnchor:
    """Derive the anchor from the step-back candidate record; never raises."""
    payload = _step_back_candidate_payload(records, episode)
    trade_offs: str | None = None
    entries: list[tuple[str, str]] = []
    if payload is not None:
        summary = payload.get("summary")
        if isinstance(summary, str) and summary.strip():
            trade_offs = _clip(summary, ALTERNATIVE_EXCERPT_LIMIT)
        dispositions = payload.get("prior_plan_item_dispositions")
        if isinstance(dispositions, list):
            for entry in dispositions:
                if not isinstance(entry, dict) or entry.get("disposition") != "resolved":
                    continue
                item_id = entry.get("item_id")
                if not isinstance(item_id, str) or not item_id.strip():
                    continue
                note = entry.get("note")
                entries.append((item_id.strip(), note if isinstance(note, str) else ""))
    prefixed = [e for e in entries if e[1].strip().lower().startswith("dissolved")]
    chosen = prefixed or entries
    ids: list[str] = []
    notes: dict[str, str] = {}
    for item_id, note in chosen:
        if item_id in notes:
            continue
        ids.append(item_id)
        cleaned = re.sub(r"^\s*dissolved\s*:?\s*", "", note, flags=re.IGNORECASE)
        notes[item_id] = _clip(cleaned, ANCHOR_NOTE_LIMIT)
    return PlanStepBackAnchor(
        candidate_round=episode.candidate_round,
        trade_offs=trade_offs,
        dissolved_ids=tuple(ids),
        dissolved_notes=notes,
    )


# A declaration is a whole line (optional whitespace or list/quote/emphasis
# prefix), never a substring of prose such as "No revision reintroduces ...".
_REVERSAL_RE = re.compile(
    r"^[\s>*_\-\u2022]*reintroduces\s+dissolved\s+item\s+[`\[]*([^\s`\]:]+)[`\]]*\s*:\s*(.*)",
    re.IGNORECASE,
)


def declared_dissolution_reversals(
    payload: Mapping[str, object] | None,
) -> tuple[tuple[str, str], ...]:
    """``(item_id, reason)`` pairs declared in a revision payload, deduplicated by ID."""
    if payload is None:
        return ()
    texts: list[str] = []
    summary = payload.get("summary")
    if isinstance(summary, str):
        texts.append(summary)
    dispositions = payload.get("prior_plan_item_dispositions")
    if isinstance(dispositions, list):
        for entry in dispositions:
            if isinstance(entry, dict) and isinstance(entry.get("note"), str):
                texts.append(entry["note"])
    found: dict[str, str] = {}
    for text in texts:
        for line in text.splitlines():
            match = _REVERSAL_RE.match(line)
            if match and match.group(1) not in found:
                found[match.group(1)] = _clip(match.group(2), ANCHOR_NOTE_LIMIT)
    return tuple(found.items())


def reintroduced_dissolved_items(
    records: Sequence["PostedRoundRecord"],
    episode: StepBackEntry,
    anchor: PlanStepBackAnchor,
) -> tuple[tuple[str, int, str], ...]:
    """Dissolved items later revisions declared reintroduced, as ``(id, round, reason)``."""
    dissolved = frozenset(anchor.dissolved_ids)
    newest: dict[int, "PostedRoundRecord"] = {}
    for record in sorted(records, key=lambda item: item.index):
        metadata = record.metadata
        if (
            metadata.role == "coder"
            and metadata.round_number > episode.candidate_round
            and record.index > episode.record_index
        ):
            newest[metadata.round_number] = record
    seen: set[str] = set()
    result: list[tuple[str, int, str]] = []
    for number in sorted(newest):
        payload = _decode_payload(newest[number].metadata.raw_structured_coder_response)
        for item_id, reason in declared_dissolution_reversals(payload):
            if item_id in dissolved and item_id not in seen:
                seen.add(item_id)
                result.append((item_id, number, reason))
    return tuple(result)


def _render_anchor_items(anchor: PlanStepBackAnchor) -> str:
    if not anchor.dissolved_ids:
        return "Dissolved items: (not recorded)\n"
    lines = ["Dissolved items:"]
    shown = anchor.dissolved_ids[:_ANCHOR_SHOWN_IDS]
    for index, item_id in enumerate(shown):
        note = anchor.dissolved_notes.get(item_id, "")
        if index < _ANCHOR_NOTED_IDS and note:
            lines.append(f"- [{item_id}] {note}")
        else:
            lines.append(f"- [{item_id}]")
    hidden = len(anchor.dissolved_ids) - len(shown)
    if hidden > 0:
        lines.append(
            f"({hidden} more dissolved IDs not shown; a declaration naming any "
            "dissolved ID is still recognized)"
        )
    return "\n".join(lines) + "\n"


def _render_anchor_facts(anchor: PlanStepBackAnchor) -> str:
    return (
        f"Step-back round: {anchor.candidate_round}\n"
        + _render_anchor_items(anchor)
        + f"Trade-offs stated by the step-back: {anchor.trade_offs or '(not recorded)'}\n"
    )


def render_plan_step_back_anchor_guidance(anchor: PlanStepBackAnchor) -> str:
    """The planner block for every revision while a step-back episode is active."""
    return (
        "STEP-BACK ANCHOR (orchestrator, not a reviewer finding): the plan is in a "
        "step-back episode and the simplified design is the anchor.\n"
        + _render_anchor_facts(anchor)
        + "Address the blocking findings within the simplified design. Reintroducing "
        "dissolved machinery requires a line `reintroduces dissolved item <ID>: <why "
        "the simpler design cannot meet the requirement otherwise>` in the summary or "
        "in the relevant disposition note. Reintroducing it without that declaration "
        "contradicts the anchor.\n"
    )


def render_plan_step_back_anchor_review_notice(anchor: PlanStepBackAnchor) -> str:
    """The reviewer block while a step-back episode is active (judgement, no filter)."""
    return (
        "Step-back anchor (orchestrator, not a reviewer finding): the plan is in a "
        "step-back episode and the simplified design is the anchor.\n"
        + _render_anchor_facts(anchor)
        + "For each blocking finding, say in the finding text either `requires "
        "reversing dissolved item <ID>` (with the requirement the simplification "
        "cannot meet) or `satisfiable within the simplified design`. This is your "
        "judgement; the orchestrator does not filter or downgrade findings on it.\n"
    )


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
        return _decode_coder_summary(metadata.raw_structured_coder_response)
    return None


# Leading bytes of the two step-back coder choices' ``summary`` (#1628).
PR_STEP_BACK_GENERALIZATION_PREFIX = "Generalization:"
PR_STEP_BACK_APPROACH_ALTERNATIVE_PREFIX = "Approach alternative:"


def render_pr_step_back_coder_guidance(triggers: Sequence[PrClusterTrigger]) -> str:
    """The two-choice step-back coder instruction for one or more clusters."""
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
        "Do not patch only the newest finding. Choose exactly one:\n"
        "(a) Generalize: name the common root of these findings, fix "
        "the whole class with one rule, and add a parametrized test that covers every branch "
        "of that rule. This replaces any instruction to keep the change small and "
        "localized. Begin the `summary` of your structured response with "
        f"`{PR_STEP_BACK_GENERALIZATION_PREFIX}` followed by the rule you applied.\n"
        "(b) Approach alternative: if the approved approach itself generates this class of "
        "findings, do not redesign beyond the approved plan and do not patch the newest "
        "finding. Begin the `summary` with "
        f"`{PR_STEP_BACK_APPROACH_ALTERNATIVE_PREFIX}`, then name the materially smaller "
        "alternative (which may be removing or relaxing an existing check or guard) and say "
        "which of these findings it dissolves. The run then stops for a human decision on "
        "that alternative; no review runs on it first.\n"
        "Clusters:\n" + "\n".join(blocks) + "\n"
    )


@dataclass(frozen=True)
class PrApproachAlternative:
    """A step-back coder turn that proposed replacing the approved approach."""

    coder_round: int
    head: str
    entries: tuple[PrStepBackEntry, ...]
    excerpt: str

    @property
    def reviewers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(entry.reviewer for entry in self.entries))


def _decode_coder_summary(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        payload, _end = json.JSONDecoder().raw_decode(raw.lstrip())
    except ValueError:
        return None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    return summary if isinstance(summary, str) else None


def approach_alternative_excerpt(summary: str | None) -> str | None:
    """The proposed alternative when ``summary`` opens with the prefix, else ``None``."""
    if not isinstance(summary, str):
        return None
    text = summary.lstrip()
    if not text.startswith(PR_STEP_BACK_APPROACH_ALTERNATIVE_PREFIX):
        return None
    return text[len(PR_STEP_BACK_APPROACH_ALTERNATIVE_PREFIX):].strip()


def pr_step_back_approach_alternative(
    records: Sequence["PostedRoundRecord"], *, head_sha: str | None
) -> PrApproachAlternative | None:
    """The newest valid step-back coder record's alternative, bound to ``head_sha``.

    Returns ``None`` (today's behavior, never an approval) unless the newest PR
    coder record carrying a valid step-back entry has a summary opening with
    :data:`PR_STEP_BACK_APPROACH_ALTERNATIVE_PREFIX`, its resulting head is
    exactly ``head_sha``, and no PR review of that head was recorded after it.
    A moved head means a human pushed past the proposal.
    """
    if not head_sha:
        return None
    ordered = sorted(records, key=lambda item: item.index)
    latest: "PostedRoundRecord | None" = None
    for record in ordered:
        metadata = record.metadata
        if (
            metadata.flow == "pr"
            and metadata.role == "coder"
            and metadata.step_back_status == "valid"
            and any(entry.get("phase") == PHASE_PR for entry in metadata.step_back_entries)
        ):
            latest = record
    if latest is None or latest.metadata.subject != head_sha:
        return None
    if any(
        record.index > latest.index
        and record.metadata.flow == "pr"
        and record.metadata.role == "reviewer"
        and record.metadata.subject == head_sha
        for record in ordered
    ):
        return None
    excerpt = approach_alternative_excerpt(
        _decode_coder_summary(latest.metadata.raw_structured_coder_response)
    )
    if excerpt is None:
        return None
    metadata = latest.metadata
    entries = tuple(
        PrStepBackEntry(
            reviewer=str(entry["reviewer"]),
            trigger_round=int(entry["trigger_round"]),
            trigger_head=str(entry["trigger_head"]),
            path=str(entry["anchor"]["path"]),
            start=int(entry["anchor"]["start"]),
            end=int(entry["anchor"]["end"]),
            coder_round=metadata.round_number,
            resulting_head=str(metadata.subject),
            record_index=latest.index,
        )
        for entry in metadata.step_back_entries
        if entry.get("phase") == PHASE_PR
    )
    return PrApproachAlternative(metadata.round_number, head_sha, entries, excerpt)


# Leading bytes of the PR approach-alternative human-decision stop (#1628).
PR_STEP_BACK_APPROACH_ALTERNATIVE_PREFIX_TEXT = (
    "human decision required: the coder proposes replacing the approved approach"
)


def render_pr_step_back_approach_alternative_decision(
    pr_number: int, alternative: PrApproachAlternative
) -> str:
    """Human-decision diagnostic after a step-back turn proposed an approach change."""
    excerpt = alternative.excerpt.strip()
    if len(excerpt) > ALTERNATIVE_EXCERPT_LIMIT:
        excerpt = excerpt[: ALTERNATIVE_EXCERPT_LIMIT - 1].rstrip() + "…"
    clusters = "; ".join(
        f"{entry.reviewer} on `{entry.path}` lines {entry.start}-{entry.end}"
        for entry in alternative.entries
    ) or "(no cluster recorded)"
    return (
        f"PR #{pr_number}: "
        + PR_STEP_BACK_APPROACH_ALTERNATIVE_PREFIX_TEXT
        + f" (step-back turn at round {alternative.coder_round}, resulting head "
        f"`{alternative.head}`, after consecutive blocks from {clusters}). No review or "
        f"coder turn was run on it. Proposed alternative (excerpt): {excerpt or '(empty)'} "
        "Decide one of: rerun with the scope change approved (re-plan or amend the issue, "
        "push the replacement and rerun); accept a stated limitation (document it, resolve "
        "or defer the findings, push the commit and rerun); or continue generalizing (rerun "
        "with --pr-step-back-rounds 0). A pushed head no longer equals the recorded head, so "
        "a rerun after a push does not stop again on this proposal."
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
        "redesign the affected code (push the redesign and rerun). If the approved approach "
        "itself generates this class of findings, approve a scope change (re-plan or amend "
        "the issue), push the replacement and rerun. After a push, a rerun does "
        "not stop again on this round's sibling; only a new sibling in a later review does."
    )


# ---------------------------------------------------------------------------
# PR fix-loop non-convergence (#1372): the same tracked-reviewer items stay
# blocking across K consecutive coder turns that each changed the head, wherever
# the findings are.  Derived only from posted reviewer and coder records.
# ---------------------------------------------------------------------------

NON_CONVERGENCE_TEXT_LIMIT = 240
_EVIDENCE_REQUEST_RE = re.compile(
    r"\b(evidence|runner|real[- ]host|live run|hardware|platform)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class PersistingItem:
    item_id: str
    first_line: str
    attempts: int
    asks_for_evidence: bool = False
    evidence_row_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrNonConvergence:
    """A tracked reviewer's window of K code-changing turns that cleared nothing."""

    reviewer: str
    k: int
    trigger_round: int
    first_round: int
    first_head: str
    current_head: str
    items: tuple[PersistingItem, ...]
    coder_rounds: tuple[int, ...]


def _pr_blocking_set(review: PrReviewRecord) -> dict | None:
    """``{item_id: item}`` the reviewer still holds blocking after this review.

    Only the reviewer's own dispositions apply, so another reviewer's vote never
    extends the tracked reviewer's streak.  Machine and human exact-head evidence
    obligations are already excluded.  Findings tagged with risk-matrix rows count
    like any other; precedence against the evidence-stall stop is decided at the
    dispatch (see ``defers_to_evidence_stall``).  ``None`` when the review is not
    a resolvable block (approved, other, or an unresolvable publication record).
    """
    if review.classification not in BLOCKING_CLASSES or review.new_items is None:
        return None
    if review.classification == CLASS_BLOCKING_UNRESOLVED:
        return None
    metadata = review.metadata
    own = tuple(d for d in metadata.dispositions if d.reviewer == metadata.agent)
    if len(own) != len(metadata.dispositions):
        review = dataclasses.replace(
            review, metadata=dataclasses.replace(metadata, dispositions=own)
        )
    return {item.item_id: item for item in _remaining_mandatory_items(review)}


def _pr_coder_after(
    records: Sequence["PostedRoundRecord"], round_number: int, after_index: int
) -> "PostedRoundRecord | None":
    """The latest PR coder record numbered ``round_number`` posted after ``after_index``."""
    best = None
    for record in records:
        metadata = record.metadata
        if (
            metadata.flow == "pr"
            and metadata.role == "coder"
            and metadata.round_number == round_number
            and record.index > after_index
            and (best is None or record.index > best.index)
        ):
            best = record
    return best


def _pr_code_changing_join(
    records: Sequence["PostedRoundRecord"],
    review: PrReviewRecord,
    next_review: PrReviewRecord,
) -> "PostedRoundRecord | None":
    """The coder record between two consecutive reviews, when it moved the head.

    Production numbering: the loop posts review r and then the coder record
    numbered r+1, whose subject is the head reviewed at round r+1.  A missing
    record, an unchanged head (no-op turn) or a next-review head that differs
    from the coder's subject (external push or rebase) yields ``None``.
    """
    if next_review.round_number != review.round_number + 1:
        return None
    coder = _pr_coder_after(records, review.round_number + 1, review.index)
    if coder is None or coder.index > next_review.index:
        return None
    subject = str(coder.metadata.subject or "")
    if not subject or subject == review.head or next_review.head != subject:
        return None
    return coder


def _open_sub_items(item) -> int:
    return sum(1 for sub in getattr(item, "sub_items", ()) if sub.status != "resolved")


def _still_blocking_without_progress(earlier: dict, later: dict, item_id: str) -> bool:
    """Whether the item stayed blocking from one review to the next with no progress.

    Resolving at least one open sub-item is progress (the sub-item stall gate owns
    sub-item pacing), so it breaks the item's chain.
    """
    if item_id not in earlier or item_id not in later:
        return False
    return _open_sub_items(later[item_id]) >= _open_sub_items(earlier[item_id])


def _first_line(text: str) -> str:
    return next((part.strip() for part in str(text).splitlines() if part.strip()), "")


def find_pr_non_convergence(
    records: Sequence["PostedRoundRecord"],
    reviewer: str,
    *,
    k: int,
    current_round: int,
    current_head: str | None,
) -> PrNonConvergence | None:
    """The reviewer's persisting items after K code-changing coder turns, if any.

    The window is the reviewer's reviews at rounds N-K..N (N = ``current_round``)
    joined by the K coder records numbered N-K+1..N.  Every review must be a
    resolvable block and every join must have moved the head with no external
    push; the persisting items are the IDs blocking in all K+1 reviews whose
    open sub-item count never dropped between two of them.  File spread and line
    windows are ignored by design.  The reported
    first head is the head of the earliest review any persisting item's attempt
    scan reaches, so the measured growth covers every counted attempt.
    """
    if k <= 0 or current_round - k < 1:
        return None
    ordered = sorted(records, key=lambda item: item.index)
    by_round = {review.round_number: review for review in pr_reviews_for(ordered, reviewer)}
    window = [by_round.get(n) for n in range(current_round - k, current_round + 1)]
    if any(review is None for review in window):
        return None
    if current_head and window[-1].head != current_head:
        return None
    blocking_sets = [_pr_blocking_set(review) for review in window]
    if any(found is None for found in blocking_sets):
        return None
    coders = [
        _pr_code_changing_join(ordered, earlier, later)
        for earlier, later in zip(window, window[1:])
    ]
    if any(coder is None for coder in coders):
        return None
    persisting = {
        item_id
        for item_id in blocking_sets[0]
        if all(
            _still_blocking_without_progress(earlier, later, item_id)
            for earlier, later in zip(blocking_sets, blocking_sets[1:])
        )
    }
    if not persisting:
        return None
    items = []
    first = window[0]
    for item_id in sorted(persisting):
        attempts = k
        later, later_set = window[0], blocking_sets[0]
        while True:
            earlier = by_round.get(later.round_number - 1)
            if earlier is None or _pr_code_changing_join(ordered, earlier, later) is None:
                break
            found = _pr_blocking_set(earlier)
            if found is None or not _still_blocking_without_progress(found, later_set, item_id):
                break
            attempts += 1
            later, later_set = earlier, found
        if later.round_number < first.round_number:
            first = later
        item = blocking_sets[-1][item_id]
        text = str(item.text)
        items.append(
            PersistingItem(
                item_id=item_id,
                first_line=_first_line(text),
                attempts=attempts,
                asks_for_evidence=bool(
                    getattr(item, "evidence_row_ids", ()) or _EVIDENCE_REQUEST_RE.search(text)
                ),
                evidence_row_ids=tuple(getattr(item, "evidence_row_ids", ()) or ()),
            )
        )
    return PrNonConvergence(
        reviewer=reviewer,
        k=k,
        trigger_round=current_round,
        first_round=first.round_number,
        first_head=first.head,
        current_head=current_head or window[-1].head,
        items=tuple(items),
        coder_rounds=tuple(coder.metadata.round_number for coder in coders),
    )


def defers_to_evidence_stall(
    result: PrNonConvergence, unsatisfied_rows: frozenset[str] | None
) -> bool:
    """Whether the evidence-stall stop may own this window, so it decides first.

    True only when every persisting item is a finding tagged with rows that are
    all unsatisfied at the dispatch head.  The stall then evaluates the combined
    ledger, the check board and its own window history at this dispatch; when it
    does not stop, the non-convergence stop still fires before any coder turn.
    Any untagged persisting item, a tag naming a satisfied row, or no usable row
    set means the stall can never own the window.
    """
    if not unsatisfied_rows or not result.items:
        return False
    return all(
        item.evidence_row_ids and set(item.evidence_row_ids) <= unsatisfied_rows
        for item in result.items
    )


def pr_non_convergence_may_matter(
    records: Sequence["PostedRoundRecord"], reviewer: str, *, k: int, round_number: int
) -> bool:
    """Snapshot pre-check: could a non-convergence window qualify at this round?

    A live dispatch snapshot predates the newest coder record (and the current
    review), so only the reviewer's reviews N-K..N-1 are checked: present,
    contiguous and all blocking.  Coder joins are verified on the fresh read.
    """
    if k <= 0 or round_number - k < 1:
        return False
    rounds = {
        review.round_number: review
        for review in pr_reviews_for(records, reviewer)
        if review.round_number < round_number
    }
    for number in range(round_number - k, round_number):
        review = rounds.get(number)
        if review is None or review.classification not in BLOCKING_CLASSES:
            return False
    return True


def _clean_decision_text(text: str) -> str:
    text = " ".join(str(text).split()).replace("`", "'")
    if len(text) > NON_CONVERGENCE_TEXT_LIMIT:
        text = text[: NON_CONVERGENCE_TEXT_LIMIT - 1].rstrip() + "…"
    return text


def render_pr_non_convergence_decision(
    pr_number: int,
    result: PrNonConvergence,
    *,
    diff_growth: str | None,
    pr_size: str | None,
) -> str:
    """Human-decision text for a non-converging PR fix loop (#1372)."""
    item_lines = "\n".join(
        f"- [{_clean_decision_text(item.item_id)}] (attempts: {item.attempts}) "
        f"{_clean_decision_text(item.first_line) or '(no text recorded)'}"
        for item in result.items
    )
    evidence = [item.item_id for item in result.items if item.asks_for_evidence]
    evidence_note = (
        "Possibly human-only evidence (these persisting items ask for evidence the "
        "environment may be unable to produce; see option 4): "
        + ", ".join(f"[{_clean_decision_text(item_id)}]" for item_id in evidence)
        + "\n"
        if evidence
        else ""
    )
    return (
        f"PR #{pr_number}: human decision required: {result.reviewer} kept the same "
        f"item(s) blocking across {result.k} consecutive coder turns that each changed the "
        f"head (reviews at rounds {result.first_round}-{result.trigger_round}, current head "
        f"`{result.current_head}`), so the fix loop is not converging. No further coder "
        "turn was invoked.\n"
        f"Persisting items:\n{item_lines}\n"
        f"Diff growth since the first attempt (`{result.first_head[:12]}` to "
        f"`{result.current_head[:12]}`): {diff_growth or 'unavailable'}\n"
        "Total PR size (a scope-growth signal; compare it with the issue's suggested "
        f"direction): {pr_size or 'unavailable'}\n"
        f"{evidence_note}"
        "Decide how to proceed:\n"
        "1. Continue: rerun with `--pr-non-convergence-rounds 0` (or a larger K).\n"
        "2. Accept a stated limitation or narrow the scope with a signed human decision "
        "through the existing signed-requirement path that resolves or defers these items, "
        "then rerun the same command.\n"
        "3. Redesign, or push a change yourself, and rerun: an external push breaks the "
        "chain, so the rerun does not stop again on this window.\n"
        "4. Convert evidence the environment cannot produce to the evidence freeze: have "
        "the reviewer re-file that item as an exact-head evidence request instead of a "
        "blocking finding."
    )
