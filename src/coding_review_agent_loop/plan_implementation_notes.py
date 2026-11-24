"""Non-blocking plan-review implementation notes (#2145).

A plan reviewer may list in-scope findings that need no plan-design change and
that building the change or running its tests would surface anyway (an
exact-inventory test, a pinned fixture set, a snapshot refresh) as
``implementation_notes``.  They never block approval, are never carried items
or planner revision input, and are never filed as issues.

This module owns the notes contract outside the protocol parser (which owns
the per-review bounds) and the round-metadata codec:

* cross-round collection inside one authenticated plan scope
  (:func:`collect_plan_implementation_notes`, :func:`resolve_plan_notes`);
* attaching notes to an :class:`ApprovedPlanContext` at every PR-facing
  construction or recovery site (:func:`attach_plan_implementation_notes`);
* the prompt section, the approval-comment section, and the scope-aware
  notes-only supplement for an identical plan reapproved in a new scope.

Notes are reviewer-authored, untrusted text: every rendering path sanitizes
them like other historical reviewer text, and they never take part in the
plan hash, subject, matrix identity, availability or handoff validation.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from .errors import AgentLoopError
from .protocol import (  # noqa: F401 - per-review bounds are parser-owned
    MAX_IMPLEMENTATION_NOTE_CHARS,
    MAX_IMPLEMENTATION_NOTES_PER_REVIEW,
)
from .protocol_markers import sanitize_historical_text

_logger = logging.getLogger(__name__)

# Collected-section bounds (applied at approval and recovery).
MAX_COLLECTED_NOTES = 20
MAX_COLLECTED_NOTE_CHARS = 6000

PROMPT_SECTION_HEADING = "Implementation notes from plan review"
APPROVAL_SECTION_HEADING = (
    "Implementation notes from plan review (non-blocking; passed to the implementer "
    "and PR reviewers; not filed as issues)"
)
SUPPLEMENT_HEADING_PREFIX = "Implementation notes from plan review (current planning scope"
_SIGNATURE = "-- coding-review-agent-loop"


@dataclass(frozen=True)
class PlanImplementationNote:
    text: str
    reviewers: tuple[str, ...]

    def rendered(self) -> str:
        """One sanitized line: the note text with its merged attribution."""
        text = sanitize_historical_text(" ".join(self.text.split()))
        reviewers = ", ".join(sanitize_historical_text(name) for name in self.reviewers)
        return f"{text} ({reviewers})" if reviewers else text


@dataclass(frozen=True)
class PlanImplementationNotes:
    """Collected notes of one approved plan scope.

    ``unavailable_reason`` is set, with no notes, when the scope could not be
    authenticated; advisory notes then never block or fail the run.
    """

    notes: tuple[PlanImplementationNote, ...] = ()
    omitted: int = 0
    scope_start_round: int = 1
    approving_round: int | None = None
    unavailable_reason: str | None = None
    # Scope-aware approval visibility (#2145 item 3); never part of identity.
    scope_boundary_index: int | None = dataclasses.field(default=None, compare=False)
    announcement_indexes: tuple[int, ...] = dataclasses.field(default=(), compare=False)
    post_boundary_trusted_bodies: tuple[str, ...] = dataclasses.field(default=(), compare=False)

    @property
    def rendered_notes(self) -> tuple[str, ...]:
        return tuple(note.rendered() for note in self.notes)


def normalize_note_key(text: str) -> str:
    """The de-duplication key: whitespace-collapsed, casefolded text."""
    return " ".join(text.split()).casefold()


def settled_plan_reviewer_records(records: Sequence[object]) -> dict[tuple[int, str], object]:
    """One settled plan reviewer record per ``(round, reviewer)``.

    An authoritative record always counts.  A parallel round settles its
    publication checkpoints with one reconciliation summary instead of
    re-posting them, so a publication record counts when that round's summary
    follows it and does not list the reviewer's seat as failed.
    """
    reconciled: dict[int, tuple[int, frozenset[str]]] = {}
    for record in records:
        metadata = getattr(record, "metadata", None)
        if (
            metadata is not None
            and metadata.flow == "plan"
            and metadata.role == "summary"
            and metadata.phase == "reconciliation"
            and metadata.round_number not in reconciled
        ):
            reconciled[metadata.round_number] = (
                record.index, frozenset(metadata.failed_reviewer_seats),
            )
    selected: dict[tuple[int, str], object] = {}
    for record in records:
        metadata = getattr(record, "metadata", None)
        if metadata is None or metadata.flow != "plan" or metadata.role != "reviewer":
            continue
        key = (metadata.round_number, str(metadata.agent))
        if metadata.phase == "authoritative":
            selected[key] = record
            continue
        settlement = reconciled.get(metadata.round_number)
        current = selected.get(key)
        if (
            metadata.phase == "publication"
            and settlement is not None
            and record.index < settlement[0]
            and str(metadata.agent) not in settlement[1]
            and (current is None or current.metadata.phase != "authoritative")
        ):
            selected[key] = record
    return selected


def collect_plan_implementation_notes(
    records: Sequence[object],
    *,
    scope_start_round: int,
    approving_round: int,
) -> PlanImplementationNotes:
    """Collect, de-duplicate and bound the notes of rounds in one plan scope.

    A plan reviewer record in ``[scope_start_round, approving_round]`` counts
    when it is settled: an authoritative record, or a parallel round's
    publication checkpoint followed by that round's reconciliation summary
    (unless the summary lists the reviewer's seat as failed).  One record per
    reviewer and round is read, the authoritative one first, so a checkpoint
    re-published as an authoritative record is never double-counted.  The
    approving round comes first, then earlier rounds descending; reviewers
    keep their posting order.
    """
    selected = {
        key: record
        for key, record in settled_plan_reviewer_records(records).items()
        if scope_start_round <= key[0] <= approving_round
    }
    by_round: dict[int, list[object]] = {}
    for (round_number, _agent), record in selected.items():
        if record.metadata.implementation_notes:
            by_round.setdefault(round_number, []).append(record)
    order: list[str] = []
    merged: dict[str, tuple[str, list[str]]] = {}
    for round_number in sorted(by_round, reverse=True):
        for record in sorted(by_round[round_number], key=lambda item: item.index):
            agent = str(record.metadata.agent)
            for text in record.metadata.implementation_notes:
                key = normalize_note_key(text)
                if not key:
                    continue
                if key not in merged:
                    merged[key] = (text, [])
                    order.append(key)
                if agent not in merged[key][1]:
                    merged[key][1].append(agent)
    kept: list[PlanImplementationNote] = []
    used = 0
    for key in order:
        text, reviewers = merged[key]
        note = PlanImplementationNote(text=text, reviewers=tuple(reviewers))
        size = len(note.rendered())
        if len(kept) >= MAX_COLLECTED_NOTES or used + size > MAX_COLLECTED_NOTE_CHARS:
            break
        kept.append(note)
        used += size
    return PlanImplementationNotes(
        notes=tuple(kept),
        omitted=len(order) - len(kept),
        scope_start_round=scope_start_round,
        approving_round=approving_round,
    )


def _approving_round(records: Sequence[object], *, plan_subject: str) -> int | None:
    rounds = [
        record.metadata.round_number
        for record in settled_plan_reviewer_records(records).values()
        if record.metadata.state == "approved" and record.metadata.subject == plan_subject
    ]
    return max(rounds) if rounds else None


def resolve_plan_notes(
    admitted_comments: Sequence[object],
    trusted_actor: tuple[str, int] | None,
    *,
    issue_number: int,
    plan_subject: str,
    plan_hash: str | None = None,
) -> PlanImplementationNotes:
    """Resolve the approved scope from authenticated history and collect its notes.

    The approving round is the latest plan round with an authoritative
    reviewer approval of ``plan_subject``.  The scope starts at the newest
    authenticated boundary whose start is at or before that round, so a later
    reopen never hides the approved scope's own earlier rounds.  There is no
    approving-round-only fallback: without an authenticated actor the result
    is unavailable.
    """
    from .plan_issue_scope import _author_id, _body, admitted_scope_boundaries
    from .round_state import _extract_round_metadata_records

    if trusted_actor is None:
        return PlanImplementationNotes(
            unavailable_reason="the orchestrator actor could not be authenticated"
        )
    records = _extract_round_metadata_records(admitted_comments, flow="plan")
    approving_round = _approving_round(records, plan_subject=plan_subject)
    if approving_round is None:
        return PlanImplementationNotes()
    boundary = None
    for candidate in admitted_scope_boundaries(
        admitted_comments, trusted_actor, issue_number=issue_number,
    ):
        if candidate.scope_start_round <= approving_round:
            boundary = candidate
    scope_start = boundary.scope_start_round if boundary is not None else 1
    collected = collect_plan_implementation_notes(
        records, scope_start_round=scope_start, approving_round=approving_round,
    )
    announcement_indexes: tuple[int, ...] = ()
    trusted_bodies: tuple[str, ...] = ()
    if boundary is not None:
        trusted_bodies = tuple(
            body.strip()
            for index, comment in enumerate(admitted_comments)
            if index > boundary.comment_index
            and _author_id(comment) == trusted_actor[1]
            and isinstance(body := _body(comment), str)
        )
    if plan_hash is not None:
        from .followups import PLAN_APPROVED_FOLLOWUP_MARKER_RE
        from .pr_round_ingestion import record_body

        announcement_indexes = tuple(
            index
            for index, comment in enumerate(admitted_comments)
            if isinstance(body := record_body(comment), str)
            and any(
                int(match.group("issue")) == issue_number and match.group("plan") == plan_hash
                for match in PLAN_APPROVED_FOLLOWUP_MARKER_RE.finditer(body)
            )
        )
    return dataclasses.replace(
        collected,
        scope_boundary_index=boundary.comment_index if boundary is not None else None,
        announcement_indexes=announcement_indexes,
        post_boundary_trusted_bodies=trusted_bodies,
    )


def _raw_records_carry_notes(comments: Sequence[object]) -> bool:
    """A cheap unauthenticated prescan: does any plan reviewer record carry notes?

    Only decides whether authenticated acquisition is worth a GitHub read; the
    notes themselves always come from the admitted history.
    """
    from .round_state import _decode_round_metadata_records

    try:
        records = _decode_round_metadata_records(tuple(comments), flow="plan")
    except AgentLoopError:
        # An undecodable raw comment (for example a foreign malformed record
        # the admitted reader would mask) must never veto the authenticated
        # read: proceed to admission instead.
        return True
    return any(
        record.metadata.role == "reviewer" and record.metadata.implementation_notes
        for record in records
    )


def _plan_source_matches(comments: Sequence[object], *, plan_subject: str) -> bool:
    from .round_state import _decode_round_metadata_records

    try:
        records = _decode_round_metadata_records(tuple(comments), flow="plan")
    except AgentLoopError:
        return True  # Undecidable raw history: let the authenticated read decide.
    return any(record.metadata.subject == plan_subject for record in records)


def load_plan_implementation_notes(
    runner: object,
    *,
    config: object,
    plan_issue_number: int,
    comments: Sequence[object],
    plan_subject: str,
    plan_hash: str | None = None,
) -> PlanImplementationNotes:
    """Authenticated notes for ``plan_subject`` on one issue; never raises."""
    if not _raw_records_carry_notes(comments):
        return PlanImplementationNotes()
    try:
        from .plan_board_state import admitted_issue_comments

        admitted, trusted_actor = admitted_issue_comments(
            runner, config=config, issue_number=plan_issue_number, comments=comments,
        )
        notes = resolve_plan_notes(
            admitted, trusted_actor, issue_number=plan_issue_number,
            plan_subject=plan_subject, plan_hash=plan_hash,
        )
    except Exception as exc:  # Advisory: a failed read never blocks the run.
        notes = PlanImplementationNotes(unavailable_reason=f"plan history could not be admitted ({exc})")
    if notes.unavailable_reason is not None:
        message = (
            f"Implementation notes for issue #{plan_issue_number} are unavailable: "
            f"{notes.unavailable_reason}"
        )
        _logger.warning("%s", message)
        try:
            from .logging import log

            log(config, message)
        except Exception:  # pragma: no cover - logging must never fail a run
            pass
    return notes


def with_plan_implementation_notes(
    context: object, notes: PlanImplementationNotes, *, origin: str | None = None,
) -> object:
    """Return ``context`` carrying ``notes``; identity fields are untouched."""
    if context is None:
        return context
    return dataclasses.replace(
        context,
        implementation_notes=notes.rendered_notes,
        implementation_notes_omitted=notes.omitted,
        implementation_notes_origin=origin if notes.notes else None,
        implementation_notes_unavailable=notes.unavailable_reason,
        implementation_notes_resolved=True,
    )


def attach_plan_implementation_notes(
    context: object,
    *,
    runner: object,
    config: object,
    sources: Sequence[object | None],
) -> object:
    """Attach the approved scope's notes to a recovered or built plan context.

    ``sources`` are issue contexts (``number`` and ``comments``), the plan
    issue first and an optional staged parent second.  The first source whose
    plan history carries the context's plan subject supplies the notes; a
    parent source labels them with the parent-plan origin.  Contexts that are
    unavailable, or already resolved, are returned unchanged.
    """
    if context is None or not getattr(context, "is_available", False):
        return context
    if getattr(context, "implementation_notes_resolved", False):
        return context
    plan_subject = context.plan_subject
    if plan_subject is None and context.canonical_text:
        from .round_metadata import _plan_subject

        plan_subject = _plan_subject(context.canonical_text)
    if plan_subject is None:
        return context
    candidates = [source for source in sources if source is not None]
    for position, source in enumerate(candidates):
        comments = tuple(getattr(source, "comments", ()) or ())
        number = getattr(source, "number", None)
        if not isinstance(number, int) or not _plan_source_matches(comments, plan_subject=plan_subject):
            continue
        notes = load_plan_implementation_notes(
            runner, config=config, plan_issue_number=number, comments=comments,
            plan_subject=plan_subject, plan_hash=context.plan_hash,
        )
        if (
            notes.approving_round is None
            and notes.unavailable_reason is None
            and position < len(candidates) - 1
        ):
            # The authenticated history holds no approval of this plan here:
            # try the next (parent) source.
            continue
        origin = f"parent plan #{number}" if position > 0 else None
        return with_plan_implementation_notes(context, notes, origin=origin)
    return context


def inherit_parent_plan_implementation_notes(
    child_context: object, parent_context: object, *, parent_issue: int,
) -> object:
    """Copy a staged parent's notes onto a direct-execution child's context."""
    if child_context is None or parent_context is None:
        return child_context
    notes = tuple(getattr(parent_context, "implementation_notes", ()) or ())
    return dataclasses.replace(
        child_context,
        implementation_notes=notes,
        implementation_notes_omitted=getattr(parent_context, "implementation_notes_omitted", 0),
        implementation_notes_origin=f"parent plan #{parent_issue}" if notes else None,
        implementation_notes_unavailable=getattr(parent_context, "implementation_notes_unavailable", None),
        # Only a resolved parent lookup resolves the child; otherwise a later
        # PR-loop parent fallback may still attach the parent's notes.
        implementation_notes_resolved=bool(getattr(parent_context, "implementation_notes_resolved", False)),
    )


def render_prompt_section(context: object) -> str:
    """The bounded prompt section, or ``""`` when there is nothing to show."""
    if context is None:
        return ""
    notes = tuple(getattr(context, "implementation_notes", ()) or ())
    unavailable = getattr(context, "implementation_notes_unavailable", None)
    if not notes and not unavailable:
        return ""
    lines = ["", PROMPT_SECTION_HEADING]
    if not notes:
        lines.append(f"Implementation notes unavailable: {sanitize_historical_text(unavailable)}")
        return "\n".join(lines) + "\n"
    lines.append(
        "Reviewer-authored, non-binding context from the plan review: check each note while "
        "implementing. They are not plan requirements; an unaddressed note is an ordinary PR "
        "review finding, never an automatic block."
    )
    origin = getattr(context, "implementation_notes_origin", None)
    if origin:
        lines.append(
            f"These notes come from the {sanitize_historical_text(origin)}; a note may concern "
            "another stage and applies only where relevant to this stage."
        )
    lines.extend(f"- {sanitize_historical_text(note)}" for note in notes)
    omitted = int(getattr(context, "implementation_notes_omitted", 0) or 0)
    if omitted:
        lines.append(f"- ({omitted} more note(s) omitted; see the plan review comments.)")
    return "\n".join(lines) + "\n"


def render_approval_section(notes: PlanImplementationNotes | None) -> list[str]:
    """Approval-comment lines for the collected notes (empty when none)."""
    if notes is None or (not notes.notes and not notes.unavailable_reason):
        return []
    lines = ["", f"{APPROVAL_SECTION_HEADING}:", ""]
    if not notes.notes:
        lines.append(f"Unavailable: {sanitize_historical_text(notes.unavailable_reason or '')}")
        return lines
    lines.extend(f"- {note.rendered()}" for note in notes.notes)
    if notes.omitted:
        lines.append(f"- ({notes.omitted} more note(s) omitted; see the plan review comments.)")
    return lines


def render_scope_supplement(notes: PlanImplementationNotes) -> str:
    """The deterministic notes-only supplement body for the current scope."""
    lines = [
        f"{SUPPLEMENT_HEADING_PREFIX}, round {notes.approving_round}).",
        "",
        "The approval announcement for this plan text predates the current planning scope. "
        "These notes replace the earlier announcement's notes; they are non-blocking, passed "
        "to the implementer and PR reviewers, and not filed as issues.",
        "",
    ]
    if notes.notes:
        lines.extend(f"- {note.rendered()}" for note in notes.notes)
        if notes.omitted:
            lines.append(f"- ({notes.omitted} more note(s) omitted; see the plan review comments.)")
    else:
        lines.append("- none (the earlier announcement's notes are superseded)")
    lines.extend(["", _SIGNATURE])
    return "\n".join(lines)


def publish_scope_notes_supplement(
    runner: object,
    *,
    config: object,
    issue_number: int,
    notes: PlanImplementationNotes | None,
) -> bool:
    """Post one notes-only supplement when the announcement belongs to an archived scope.

    Runs only after the existing announcement was found.  It posts nothing
    when there is no authenticated current-scope boundary, when an
    announcement for this plan already follows that boundary, or when an
    identical supplement already follows it.  It carries no approval record
    and files no issue.
    """
    if notes is None or notes.scope_boundary_index is None or notes.approving_round is None:
        return False
    boundary = notes.scope_boundary_index
    if not notes.announcement_indexes or max(notes.announcement_indexes) > boundary:
        return False
    body = render_scope_supplement(notes)
    if body.strip() in notes.post_boundary_trusted_bodies:
        return False
    from .github import post_issue_comment

    post_issue_comment(runner, config=config, issue_number=issue_number, body=body)
    return True


def resumed_implementation_notes(
    metadata: object, structured_review: object | None, *, reviewer: str,
) -> tuple[object, ...]:
    """A resumed plan review's notes, restored exactly once.

    The persisted record is authoritative; a publication checkpoint written
    before the record carried notes falls back to re-parsing its canonical
    reviewer response.  The two sources are never merged.
    """
    from .protocol import ApprovedFollowup

    recorded = tuple(getattr(metadata, "implementation_notes", ()) or ())
    if recorded:
        return tuple(ApprovedFollowup(reviewer=reviewer, text=text) for text in recorded)
    if structured_review is None:
        return ()
    return tuple(getattr(structured_review, "implementation_notes", ()) or ())


def resolve_plan_notes_safely(
    admitted_comments: Sequence[object],
    trusted_actor: tuple[str, int] | None,
    *,
    issue_number: int,
    plan_subject: str | None,
    plan_hash: str | None = None,
) -> PlanImplementationNotes:
    """:func:`resolve_plan_notes` for callers that already admitted history; never raises."""
    if plan_subject is None or not _raw_records_carry_notes(admitted_comments):
        return PlanImplementationNotes()
    try:
        return resolve_plan_notes(
            admitted_comments, trusted_actor, issue_number=issue_number,
            plan_subject=plan_subject, plan_hash=plan_hash,
        )
    except Exception as exc:  # Advisory: never fail plan recovery.
        _logger.warning("Implementation notes for issue #%s are unavailable: %s", issue_number, exc)
        return PlanImplementationNotes(unavailable_reason=f"plan history could not be read ({exc})")
