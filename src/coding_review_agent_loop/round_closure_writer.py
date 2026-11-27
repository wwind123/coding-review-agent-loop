"""The round-wide closure decision and the closed-incomplete record writer (#2064).

Stage 2 of #2064 replaces the partial-review-round refusal with one rule,
used by both the PR and the plan loop at round start, before any spool
preflight, scheduler checkpoint or reviewer launch:

    when round N holds any reviewer publication and any configured reviewer
    would need a fresh round-N turn, post one tool-owned closed-incomplete
    record carrying the round's ledger and roll over to round N+1.

A *publication* is a published withheld-round review (resumed or rejected by
resume), a reviewer's own published review that resume rejected, or a spool
record that may be public: a record whose publication began, a legacy record
with unknown publication progress, or a malformed record whose publication
state cannot be excluded.  Such a record is never resumed, so its reviewer
also needs a fresh turn.

A reviewer *needs a fresh turn* when its own published round-N review was
rejected by resume (always, whatever its spool holds), or when it would take
a turn and has neither a settled spooled failure nor a spooled response that
still validates in this run's context.

The published reviews stay on GitHub as history and never approve; their
findings are rebuilt from the records themselves through the same parser
chain and minting rules ordinary resume and settlement use, and enter the
ledger with fresh, never-reused item IDs.  Read failures while deciding are
retryable :class:`RoundStateReadError` stops that post nothing.  No path here
deletes a comment or asks the operator to delete one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

from .errors import AgentLoopError
from .protocol import (
    ParsedPlanReview,
    ParsedReview,
    parse_agent_state,
    parse_plan_review_items,
    parse_plan_state,
    parse_review,
    parse_structured_plan_review,
    parse_structured_pr_review,
    review_freeform_summary_text,
)
from .round_closure import ROUND_CLOSED_INCOMPLETE_PHASE, build_round_closure_payload


class RoundIncompleteSignal(AgentLoopError):
    """Internal signal: round N is incomplete and must close and roll over.

    Raised by former partial-round refusal sites before any reviewer is
    invoked or anything is posted.  The loop's handler posts one closed-
    incomplete record; if the signal ever escapes, the run stops fail-closed
    and the next resume's round-start decision closes the round.
    """

    def __init__(
        self,
        message: str,
        *,
        round_number: int | None = None,
        published: Sequence[str] = (),
        unreplayable: Sequence[tuple[str, str]] = (),
        before_publication: bool = True,
    ) -> None:
        super().__init__(message)
        self.round_number = round_number
        self.published = tuple(published)
        self.unreplayable = tuple(unreplayable)
        self.before_publication = before_publication


class RoundStateReadError(AgentLoopError):
    """A retryable GitHub comment or identity read failure while classifying a round.

    Nothing was posted and no closure record was written; rerun once reads
    succeed.
    """


def round_incomplete_signal(
    *, surface: str, number: int, round_number: int,
    published: Sequence[str], unreplayable: Sequence[tuple[str, str]],
    before_publication: bool = True,
) -> RoundIncompleteSignal:
    target = f"PR #{number}" if surface == "pr" else f"issue #{number}"
    names = ", ".join(name for name, _cause in unreplayable) or "a reviewer"
    return RoundIncompleteSignal(
        f"Review round {round_number} on {target} is incomplete: {names} cannot be replayed "
        "beside the round's published reviews. Nothing was posted for it; the round is "
        f"closed and review continues in round {round_number + 1} (a rerun closes it if "
        "this run stops first).",
        round_number=round_number, published=published, unreplayable=unreplayable,
        before_publication=before_publication,
    )


def excluded_incomplete_round_route(*, evidence_pass: bool, recovery: bool, conflict: bool) -> str | None:
    """The label of a route that keeps its own records instead of closing (#2163).

    The evidence pass, the recovery route and the conflict route launch no
    withheld-round reviewer, so no refusal site runs there; ``None`` means the
    round uses the ordinary closure (a conversion repeat rolls over).
    """
    if evidence_pass:
        return "evidence pass"
    if conflict:
        return "conflict route"
    if recovery:
        return "recovery route"
    return None


def excluded_route_stop(signal: RoundIncompleteSignal, *, route: str, pr_number: int) -> RoundIncompleteSignal:
    """The defensive invariant inside a route that keeps its own records (#2064, #2163).

    The evidence pass, recovery and conflict routes write sequential or coder
    records, not withheld-round publications, so the round is not closed
    there.  The stop names the one working next step: rerun the same command,
    which resumes the route from its own records; the next ordinary round's
    start then closes any incomplete round.  No deletion advice.
    """
    names = ", ".join(name for name, _cause in signal.unreplayable) or "a reviewer"
    peers = ", ".join(signal.published) or "a same-round peer"
    round_number = signal.round_number
    later = (
        f"if round {round_number} still holds an incomplete publication, the next ordinary "
        f"round's start closes it and review continues in round {round_number + 1}"
        if round_number is not None
        else "the next ordinary round's start closes any incomplete round"
    )
    return RoundIncompleteSignal(
        f"Review round {round_number} is incomplete inside its {route}: {names} cannot be "
        f"replayed beside the published review(s) of {peers}. Nothing was posted and no "
        f"reviewer was launched; the {route} writes sequential or coder records, not "
        "withheld-round publications, so it keeps its own records and the round is not "
        f"closed here. Next step: rerun `agent-loop pr {pr_number}` with the same options; "
        f"the {route} resumes from its own records, and {later}.",
        round_number=round_number, published=signal.published,
        unreplayable=signal.unreplayable, before_publication=False,
    )


# The unresolved-boundary detail shared by the refusal sites and the closure
# paths (#1676, #2163), so the wording cannot drift between them.
UNRESOLVED_CONVERSION_BOUNDARY_DETAIL = (
    "this round's evidence-conversion record could not be admitted as a trusted "
    "boundary, so the earlier pass's publications cannot be excluded from the peer "
    "set; inspect the conversion record if reads were healthy"
)


def round_state_read_error(*, surface: str, number: int, round_number: int, detail: str) -> RoundStateReadError:
    target = f"PR #{number}" if surface == "pr" else f"issue #{number}"
    return RoundStateReadError(
        f"Review round {round_number} on {target} could not be classified because a GitHub "
        f"read failed ({detail}). Nothing was posted and no reviewer was launched; rerun once "
        "GitHub comment reads succeed."
    )


# -- decision -----------------------------------------------------------------


@dataclass(frozen=True)
class RoundClosureDecision:
    """Why round N closes: its publications and its unreplayable reviewers."""

    round_number: int
    published: tuple[tuple[str, int | None], ...]
    unreplayable: tuple[tuple[str, str], ...]


def decide_round_closure(
    *,
    round_number: int,
    reviewer_order: Sequence[str],
    publications: Mapping[str, int | None],
    fresh_turns: Mapping[str, str],
    already_closed: bool = False,
    excluded: bool = False,
) -> RoundClosureDecision | None:
    """Pure round-wide rule: closure iff any publication and any fresh turn.

    ``publications`` maps reviewer names with a round-N publication to the
    anchor comment ID (``None`` when unknown); ``fresh_turns`` maps reviewers
    that would need a fresh round-N turn to a short cause.  The rule is
    evaluated over the whole board, never per-reviewer peer sets, so a
    reviewer whose only public review is its own is closed out, not re-run.
    Posts nothing and launches nothing.
    """
    if already_closed or excluded or not publications or not fresh_turns:
        return None
    order = {name: index for index, name in enumerate(reviewer_order)}

    def key(name: str) -> tuple[int, str]:
        return (order.get(name, len(order)), name)

    return RoundClosureDecision(
        round_number=round_number,
        published=tuple((name, publications[name]) for name in sorted(publications, key=key)),
        unreplayable=tuple((name, fresh_turns[name]) for name in sorted(fresh_turns, key=key)),
    )


_POSSIBLY_PUBLIC_STATES = {
    "began": "its same-round publication began and may be partly public",
    "legacy": "its spool record predates the publication protocol and may be partly public",
    "malformed": "its spool record is malformed or unreadable",
}


@dataclass
class RoundClassification:
    publications: dict[str, int | None]
    fresh_turns: dict[str, str]


def classify_round_reviewers(
    runner: object,
    *,
    config: object,
    surface: str,
    number: int,
    round_number: int,
    reviewers: Sequence[object],
    turn_names: Collection[str],
    published_records: Mapping[str, object],
    rejected_records: Mapping[str, object],
    rejection_reasons: Mapping[str, str],
    spool: object,
    validators_for: Callable[[str], dict[str, object]],
    comment_id_for: Callable[[object], int | None],
    resumed_names: Collection[str] = (),
) -> RoundClassification:
    """Collect the round's publications and fresh-turn needs (local reads, no posts).

    ``published_records`` are resumed public withheld-round reviews,
    ``rejected_records`` the reviewers' own published reviews resume rejected,
    and ``turn_names`` the reviewers the round would otherwise invoke.  A
    spool record whose publication began (or may have) is never resumed: it
    is both a publication and a fresh-turn need, so the round closes.  The
    exception is a reviewer in ``resumed_names``: resume admitted its public
    anchor (the last body), so its publication completed and the record is
    stale (the spool preflight discards it).
    """
    from .agents.registry import agent_display_name
    from .review_rounds import _replay_spooled_review

    publications: dict[str, int | None] = {}
    fresh: dict[str, str] = {}
    for name, record in (*published_records.items(), *rejected_records.items()):
        publications[name] = comment_id_for(record)
    for name in rejected_records:
        reason = rejection_reasons.get(name, "")
        fresh[name] = "its own published review was rejected by resume" + (
            f" ({reason})" if reason else ""
        )
    for reviewer in reviewers:
        name = agent_display_name(reviewer)  # type: ignore[arg-type]
        if name in published_records or name in rejected_records or name in resumed_names:
            continue
        state = spool.publication_state(name)  # type: ignore[attr-defined]
        if state in _POSSIBLY_PUBLIC_STATES:
            publications.setdefault(name, None)
            fresh[name] = _POSSIBLY_PUBLIC_STATES[state]
            continue
        if name not in turn_names:
            continue
        fields = spool.load(name)  # type: ignore[attr-defined]
        if fields is None:
            fresh[name] = "no spooled outcome for this round exists on this host"
            continue
        if fields.get("failure") is not None:
            continue
        replayed = _replay_spooled_review(
            runner, config=config, reviewer=reviewer, fields=fields,  # type: ignore[arg-type]
            validators=validators_for(name),
        )
        if replayed is None:
            fresh[name] = "its spooled outcome no longer validates against this run"
    return RoundClassification(publications, fresh)


# -- shared reconstruction (resume and writer) ---------------------------------


def pr_published_review(record: object, *, reviewer: str, parse_requirement_evidence: bool = False,
                        strict: bool = False):
    """Rebuild a posted PR review exactly as ordinary resume does.

    The review output is the canonical reviewer response, else the body; it
    is parsed as a structured review in legacy mode with the legacy-text
    fallback, and every metadata-sourced field comes from the record.
    With ``strict`` (the closure writer), a response with neither a structured
    review nor the legacy state marker raises ``AgentLoopError``.
    Returns ``(parsed_review, review_output)``.
    """
    from .architecture_contract import _resumed_review_architecture
    from .post_merge_obligations import parse_request as parse_post_merge_request

    metadata = record.metadata  # type: ignore[attr-defined]
    canonical = metadata.canonical_reviewer_response
    review_output = canonical or record.body  # type: ignore[attr-defined]
    structured = (
        parse_structured_pr_review(review_output, reviewer=reviewer, architecture_status_mode="legacy",
                                   parse_requirement_evidence=parse_requirement_evidence)
        if canonical is not None else None
    )
    if strict and structured is None:
        # The legacy fallback accepts arbitrary text; the closure writer needs a
        # recognizable review envelope or it would close the round with no findings.
        parse_agent_state(review_output)
    reparsed = structured or parse_review(review_output, reviewer=reviewer)
    impact, degradations = _resumed_review_architecture(
        metadata, structured.architecture_impact if structured is not None else None,
    )
    parsed = ParsedReview(
        state=metadata.state or parse_agent_state(review_output),
        summary=reparsed.summary,
        blocking_items=reparsed.blocking_items,
        followups=reparsed.followups,
        dispositions=metadata.dispositions,
        architecture_impact=impact,
        architecture_impact_degradations=degradations,
        exact_head_evidence_requests=metadata.evidence_requests,
        withdrawn_human_checks=metadata.withdrawn_human_checks,
        requirement_evidence=metadata.requirement_evidence,
        screenshot_requests=metadata.screenshot_requests,
        screenshot_coverage_acceptances=metadata.screenshot_coverage_acceptances,
        post_merge_obligations=tuple(parse_post_merge_request(r) for r in metadata.post_merge_requests),
        sub_item_degradations=metadata.sub_item_degradations,
        ignored_prior_disposition_ids=metadata.ignored_prior_disposition_ids,
    )
    return parsed, review_output


def pr_records_blocking_summary(parsed: object, *, had_prior_items: bool) -> bool:
    """Settlement's rule for minting a blocking review's free-form summary."""
    from .review_rounds import _should_record_new_blocking_item

    if parsed.blocking_items or parsed.followups.same_pr:  # type: ignore[attr-defined]
        return False
    return _should_record_new_blocking_item(
        parsed.summary,  # type: ignore[attr-defined]
        had_prior_items=had_prior_items,
        had_dispositions=bool(parsed.dispositions),  # type: ignore[attr-defined]
        has_active_carried_disposition=any(
            disposition.disposition in {"blocking", "same-pr"}
            for disposition in parsed.dispositions  # type: ignore[attr-defined]
        ),
    )


def mint_pr_review_items(
    parsed: object,
    *,
    reviewer: str,
    round_number: int,
    next_item_number: int,
    has_blocking_summary: bool,
    approved_followups: str,
) -> list[object]:
    """Settlement's outcome-aware minting of one PR review's new items.

    A blocking review mints its blocking items (or its free-form summary),
    its Same-PR follow-ups, then its future follow-ups; an approved review
    mints only its future follow-ups.  Future follow-ups are skipped when
    ``approved_followups`` is ``ignore``.
    """
    from .unresolved_items import _next_unresolved_item

    items: list[object] = []

    def mint(**fields: object) -> None:
        items.append(_next_unresolved_item(
            item_number=next_item_number + len(items), source_round=round_number, **fields,  # type: ignore[arg-type]
        ))

    if parsed.state == "blocking":  # type: ignore[attr-defined]
        if parsed.blocking_items:  # type: ignore[attr-defined]
            for item in parsed.blocking_items:  # type: ignore[attr-defined]
                mint(reviewer=item.reviewer, text=item.text, status="blocking", fix_scope=item.fix_scope,
                     sub_items=item.sub_items, evidence_row_ids=item.evidence_row_ids)
        elif has_blocking_summary:
            mint(reviewer=reviewer, text=parsed.summary, status="blocking")  # type: ignore[attr-defined]
        for followup in parsed.followups.same_pr:  # type: ignore[attr-defined]
            mint(reviewer=followup.reviewer, text=followup.text, status="same-pr",
                 fix_scope=followup.fix_scope, sub_items=followup.sub_items)
    if approved_followups != "ignore":
        for followup in parsed.followups.future:  # type: ignore[attr-defined]
            mint(reviewer=followup.reviewer, text=followup.text, status="future")
    return items


def plan_published_review(record: object, *, reviewer: str, surfaced_requirement_ids: Sequence[str],
                          strict: bool = False):
    """Rebuild a posted plan review exactly as ordinary resume does.

    Structured parse of the canonical response (else the body) with the
    legacy ``parse_plan_review_items`` fallback; state from the metadata.
    Returns ``(parsed_review, review_output)``.
    """
    from .architecture_contract import _resumed_review_architecture
    from .plan_implementation_notes import resumed_implementation_notes

    metadata = record.metadata  # type: ignore[attr-defined]
    review_output = metadata.canonical_reviewer_response or record.body  # type: ignore[attr-defined]
    structured = parse_structured_plan_review(
        review_output, architecture_status_mode="legacy", reviewer=reviewer,
        surfaced_requirement_ids=tuple(surfaced_requirement_ids),
    )
    if strict and structured is None:
        # Arbitrary text parses as an empty legacy review; require the plan envelope.
        parse_plan_state(review_output)
    impact, degradations = _resumed_review_architecture(
        metadata, structured.architecture_impact if structured is not None else None,
    )
    parsed = ParsedPlanReview(
        state=metadata.state or parse_plan_state(review_output),
        summary=(structured.summary if structured is not None else review_freeform_summary_text(review_output)),
        items=(structured.items if structured is not None
               else parse_plan_review_items(review_output, reviewer=reviewer)),
        dispositions=metadata.dispositions,
        architecture_impact=impact,
        architecture_impact_degradations=degradations,
        ignored_prior_disposition_ids=metadata.ignored_prior_disposition_ids,
        # Restored once from the record (#2145); a record written before
        # notes were persisted re-parses its response.
        implementation_notes=resumed_implementation_notes(metadata, structured, reviewer=reviewer),
    )
    return parsed, review_output


def plan_review_item_tags(item: object, prior_items: Sequence[object]) -> dict[str, object]:
    """Minting-time provenance and repeat tags of a plan review finding (#1505).

    ``repeat_of`` is computed from the carried ledger the reviewer was shown,
    never read from agent output.
    """
    from .unresolved_items import carried_item_repeated_by

    return {
        "provenance": getattr(item, "provenance", None),
        "repeat_of": carried_item_repeated_by(item.text, prior_items),  # type: ignore[attr-defined, arg-type]
    }


def mint_plan_review_items(
    parsed: object, *, round_number: int, next_item_number: int, prior_items: Sequence[object]
) -> list[object]:
    """Settlement's minting of one plan review: blocking, same-plan, then future."""
    from .unresolved_items import _next_unresolved_item

    items: list[object] = []
    for status, entries in (
        ("blocking", parsed.items.blocking),  # type: ignore[attr-defined]
        ("same-plan", parsed.items.same_plan),  # type: ignore[attr-defined]
        ("future", parsed.items.future),  # type: ignore[attr-defined]
    ):
        for item in entries:
            items.append(_next_unresolved_item(
                item_number=next_item_number + len(items), reviewer=item.reviewer,
                source_round=round_number, text=item.text, status=status,
                **plan_review_item_tags(item, prior_items),  # type: ignore[arg-type]
            ))
    return items


def impending_opening_source(*, opened: bool, source: str | None, launch_phase: str | None,
                             operator: bool) -> str | None:
    """The panel opening in force for a launch decided before its scheduler checkpoint.

    An already recorded opening keeps its source.  Otherwise the checkpoint about
    to be posted is itself the opening: an operator force-full latch, or a gated
    secondary launch opened by the primary's approval (``panel_evidence``).
    """
    from .round_visibility import GATED_LAUNCH_PHASES, PRIMARY_APPROVAL_OPENING

    if opened:
        return source
    if operator:
        return "operator"
    return PRIMARY_APPROVAL_OPENING if launch_phase in GATED_LAUNCH_PHASES else None


def closure_carried_items(items: Sequence[object], closure_item_ids: Collection[str], *, primary: str | None) -> tuple[str, ...]:
    """Active findings a closed-incomplete record carried forward that are pending on a non-primary owner.

    Like pre-epoch panel items (board change D5a), they are carried obligations:
    a closed review never makes a secondary a finding owner before a qualified
    panel opening, so the next round can open a fresh primary phase and the
    owners recheck them when the panel opens.
    """
    carried: list[str] = []
    for item in items:
        if item.item_id not in closure_item_ids or item.status not in {"blocking", "same-pr"}:  # type: ignore[attr-defined]
            continue
        owners = tuple(item.resolution_owners or (item.reviewer,))  # type: ignore[attr-defined]
        states = dict(item.owner_states or ((owner, "pending") for owner in owners))  # type: ignore[attr-defined]
        if any(owner != primary and states.get(owner) != "cleared" for owner in owners):
            carried.append(item.item_id)  # type: ignore[attr-defined]
    return tuple(carried)


def _item_number(item: object) -> int:
    item_id = str(getattr(item, "item_id", ""))
    suffix = item_id.rsplit("-", 1)[-1]
    return int(suffix) if item_id.startswith("item-") and suffix.isdigit() else 0


# -- writer -------------------------------------------------------------------


def close_incomplete_round(
    runner: object,
    *,
    config: object,
    decision: RoundClosureDecision,
    flow: str,
    subject: str,
    number: int,
    comments_length: int,
    current_resume: object,
    prior_items: Sequence[object],
    published_records: Mapping[str, object],
    next_item_number: int,
    mint_record: Callable[[object, int], list[object]],
    bound_metadata: Callable[..., object],
    post_comment: Callable[[str], None],
    extra_metadata: Mapping[str, object] | None = None,
) -> object:
    """Post one closed-incomplete record for ``decision`` and return the closed resume.

    ``published_records`` holds the round's posted reviewer records in board
    order.  A record that already carries authoritative minted items keeps
    their IDs; a provisional publication checkpoint is rebuilt through
    ``mint_record(record, next_number)`` (the shared resume parser chain and
    settlement minting rules).  Each published exact-head evidence request
    becomes an obligation numbered after the minted items.  Save before emit:
    the ledger is persisted in the posted record before the caller discards
    the round spool.
    """
    from .logging import log
    from .round_metadata import PostedRoundRecord, _attach_round_metadata

    target = f"PR #{number}" if flow == "pr" else f"issue #{number}"
    round_number = decision.round_number
    high_water = max(int(next_item_number) - 1, 0)
    new_items: list[object] = []
    for name, record in published_records.items():
        metadata = record.metadata  # type: ignore[attr-defined]
        if metadata.new_items or metadata.phase != "publication":
            kept = list(metadata.new_items)
        else:
            try:
                kept = mint_record(record, high_water + 1)
            except AgentLoopError as exc:
                raise RoundStateReadError(
                    f"Review round {round_number} on {target} could not be closed: {name}'s "
                    f"published review could not be rebuilt ({exc}). Nothing was posted; rerun "
                    "after inspecting that record."
                ) from exc
        new_items.extend(kept)
        high_water = max([high_water, *(_item_number(item) for item in kept)])
    from .unresolved_items import evidence_obligation_identity

    # A request already restored into the prior ledger (#1922) keeps its item.
    known = {
        identity
        for item in prior_items
        for identity in (getattr(item, "obligation_identity", None),
                         *getattr(item, "alias_obligation_identities", ()))
        if identity
    }
    obligations: list[dict[str, object]] = []
    for name, record in published_records.items():
        metadata = record.metadata  # type: ignore[attr-defined]
        for text in getattr(metadata, "evidence_requests", ()) or ():
            identity = evidence_obligation_identity(str(text), str(metadata.agent))
            if not str(text).strip() or identity in known:
                continue
            known.add(identity)
            high_water += 1
            obligations.append({
                "reviewer": metadata.agent, "source_round": metadata.round_number,
                "head": metadata.subject, "text": str(text), "item_number": high_water,
            })
    payload = build_round_closure_payload(
        flow=flow, round_number=round_number, subject=subject,
        published=decision.published, unreplayable=decision.unreplayable,
        evidence_obligations=obligations, item_number_high_water=high_water,
    )
    if payload is None:
        raise AgentLoopError(
            f"Review round {round_number} on {target}: internal error: the closed-incomplete "
            "record would not decode. Nothing was posted."
        )
    metadata = bound_metadata(
        flow=flow, role="summary", agent="Orchestrator", round_number=round_number,
        subject=subject, prior_items=tuple(prior_items), new_items=tuple(new_items),
        phase=ROUND_CLOSED_INCOMPLETE_PHASE, round_closure=payload,
        **dict(extra_metadata or {}),
    )
    published_names = ", ".join(name for name, _cid in decision.published)
    unreplayable = "; ".join(f"{name}: {cause}" for name, cause in decision.unreplayable)
    body = _attach_round_metadata(
        f"Review round {round_number} closed, incomplete. The published reviews of "
        f"{published_names} stay as history and do not count toward approval; their findings "
        f"carry forward as prior items. Not replayable: {unreplayable}. Round "
        f"{round_number + 1} reviews with the full board. No comment needs to be deleted.",
        metadata,  # type: ignore[arg-type]
    )
    post_comment(body)
    log(config, f"{target}: review round {round_number} closed incomplete "  # type: ignore[arg-type]
        f"(published: {published_names}; unreplayable: {unreplayable}); opening round {round_number + 1}")
    return dataclasses.replace(
        current_resume,  # type: ignore[arg-type]
        reconciliation_record=PostedRoundRecord(index=comments_length, metadata=metadata, body=body),  # type: ignore[arg-type]
        completed_reviews=(),
        reconciled=True,
        current_round_new_items=tuple(new_items),
        next_unresolved_item_number=max(
            int(getattr(current_resume, "next_unresolved_item_number", 1)), high_water + 1,
        ),
    )


def closure_round_limit_note(closed_round: int | None, next_round: int) -> str:
    """Round-limit suffix naming the fresh round a closed-incomplete round requires (#2064)."""
    if closed_round is None or closed_round + 1 != next_round:
        return ""
    return (
        f" Round {closed_round} was closed incomplete; fresh reviews from the full board "
        f"require round {next_round}, beyond --max-rounds. Rerun with a higher --max-rounds "
        f"to continue in round {next_round}."
    )
