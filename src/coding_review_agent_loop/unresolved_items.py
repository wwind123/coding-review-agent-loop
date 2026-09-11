"""Unresolved review item ledger and response validation helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .errors import (
    AgentLoopError,
    IssueImplementationConflictError,
    UnknownPriorItemDispositionError,
)
from .github import PullRequestMergeability
from .prompts import render_coder_human_requirements_prompt_context
from .protocol import (
    ParsedPlanReview,
    ParsedReview,
    ReviewItemDisposition,
    StructuredCoderFollowup,
    StructuredIssueImplementation,
    UnresolvedReviewItem,
    parse_plan_review,
    parse_pr_review,
    validate_human_requirements_acknowledgement,
    validate_structured_coder_followup,
    validate_structured_human_requirements_acknowledgement,
    validate_structured_issue_implementation,
    validate_human_requirement_dispositions,
)

HUMAN_REQUIREMENTS_ACK_ITEM_ID = "item-human-requirements-acknowledgement"
MERGE_CONFLICT_ITEM_ID = "item-merge-conflict"
CODER_DISPUTE_NOTE_PREFIX = "Coder disputes this item"
ALL_RESOLVED_PROSE_RE = re.compile(
    r"^all (?:prior items|listed items|carried-forward items) are resolved\.?$"
    r"|^all prior unresolved items have been resolved\.?$",
    re.I,
)


def _normalize_disposition_section_prose(text: str) -> str:
    return " ".join(text.strip().split())


def _maybe_fill_resolved_dispositions_from_prose(
    parsed: ParsedReview | ParsedPlanReview,
    *,
    reviewer: str,
    unresolved_items: Sequence[UnresolvedReviewItem],
) -> tuple[ReviewItemDisposition, ...]:
    if not unresolved_items or parsed.dispositions:
        return parsed.dispositions
    normalized = _normalize_disposition_section_prose(parsed.raw_dispositions_text)
    if not normalized or not ALL_RESOLVED_PROSE_RE.fullmatch(normalized):
        return parsed.dispositions
    return tuple(
        ReviewItemDisposition(
            item_id=item.item_id,
            reviewer=reviewer,
            disposition="resolved",
        )
        for item in unresolved_items
    )


def _next_unresolved_item(
    *,
    item_number: int,
    reviewer: str,
    source_round: int,
    text: str,
    status: str,
    notes: Sequence[str] = (),
    fix_scope: tuple[str, ...] | None = None,
    resolution_owners: Sequence[str] | None = None,
) -> UnresolvedReviewItem:
    owners = tuple(resolution_owners) if resolution_owners is not None else (reviewer,)
    owner_states = tuple((owner, "pending") for owner in owners)
    return UnresolvedReviewItem(
        item_id=f"item-{item_number}",
        reviewer=reviewer,
        source_round=source_round,
        text=text,
        status=status,
        source_status=status,
        notes=tuple(notes),
        fix_scope=fix_scope,
        resolution_owners=owners,
        owner_states=owner_states,
    )


def _apply_dispute_evidence(
    unresolved_items: Sequence[UnresolvedReviewItem],
    *,
    disputed_items: Sequence[str],
    dispute_evidence: dict[str, str],
) -> list[UnresolvedReviewItem]:
    """Annotate disputed items with coder counter-evidence in their notes field."""
    if not disputed_items:
        return list(unresolved_items)
    disputed_set = set(disputed_items)
    result: list[UnresolvedReviewItem] = []
    for item in unresolved_items:
        if item.item_id not in disputed_set:
            result.append(item)
            continue
        evidence = dispute_evidence.get(item.item_id, "")
        note = f"{CODER_DISPUTE_NOTE_PREFIX}: {evidence}" if evidence else CODER_DISPUTE_NOTE_PREFIX
        if note not in item.notes:
            result.append(
                UnresolvedReviewItem(
                    item_id=item.item_id,
                    reviewer=item.reviewer,
                    source_round=item.source_round,
                    text=item.text,
                    status=item.status,
                    source_status=item.source_status,
                    notes=(*item.notes, note),
                    fix_scope=item.fix_scope,
                    resolution_owners=item.resolution_owners,
                    owner_states=item.owner_states,
                    owner_evidence=item.owner_evidence,
                    owner_dispositions=item.owner_dispositions,
                )
            )
        else:
            result.append(item)
    return result


def _is_disputed_item(item: UnresolvedReviewItem) -> bool:
    """Return True if this item has been disputed by the coder via counter-evidence."""
    return any(note.startswith(CODER_DISPUTE_NOTE_PREFIX) for note in item.notes)


def _raise_if_maintained_disputed_items(
    unresolved_items: Sequence[UnresolvedReviewItem],
    *,
    prior_items: Sequence[UnresolvedReviewItem],
) -> None:
    """Stop after one reconsideration when a disputed blocker remains active."""
    prior_item_ids = {item.item_id for item in prior_items}
    disputed_still_blocking = [
        item
        for item in unresolved_items
        if item.item_id in prior_item_ids
        and item.status in {"blocking", "same-pr"}
        and _is_disputed_item(item)
    ]
    if not disputed_still_blocking:
        return
    item_summaries = "\n".join(
        "\n".join(
            [
                f"- [{item.item_id}] from {item.reviewer}: {item.text[:300]}",
                *[f"  Update/evidence: {note}" for note in item.notes],
            ]
        )
        for item in disputed_still_blocking
    )
    raise AgentLoopError(
        f"Reviewer did not resolve {len(disputed_still_blocking)} disputed item(s) "
        "after seeing coder counter-evidence. Human review required to resolve the "
        f"disagreement.\n\nDisputed items still unresolved:\n{item_summaries}"
    )


def _same_round_prior_disposition_description(
    unknown: Sequence[str],
    current_round_items: Sequence[UnresolvedReviewItem],
) -> str:
    same_round_ids = {item.item_id for item in current_round_items}
    same_round_description = (
        "Same-round findings are informational only and must not be dispositioned "
        "as prior carried items."
    )
    same_round_matches = [item_id for item_id in unknown if item_id in same_round_ids]
    if same_round_matches:
        same_round_description += (
            " Unknown ID(s) matching same-round findings: "
            + ", ".join(same_round_matches)
            + "."
        )
    return same_round_description


def _raise_unknown_prior_item_disposition(
    unknown: Sequence[str],
    *,
    allowed_ids: Sequence[str],
    current_round_items: Sequence[UnresolvedReviewItem],
) -> None:
    raise UnknownPriorItemDispositionError(
        unknown_ids=tuple(sorted(unknown)),
        allowed_ids=tuple(sorted(allowed_ids)),
        same_round_description=_same_round_prior_disposition_description(
            tuple(sorted(unknown)),
            current_round_items,
        ),
    )


def _validate_review_response(
    text: str,
    *,
    reviewer: str,
    unresolved_items: Sequence[UnresolvedReviewItem],
    current_round_items: Sequence[UnresolvedReviewItem] = (),
) -> ParsedReview:
    parsed = parse_pr_review(text, reviewer=reviewer)

    unresolved_by_id = {item.item_id: item for item in unresolved_items}
    dispositions = _maybe_fill_resolved_dispositions_from_prose(
        parsed,
        reviewer=reviewer,
        unresolved_items=unresolved_items,
    )
    if not dispositions and not unresolved_items:
        return parsed
    disposition_ids = [item.item_id for item in dispositions]
    duplicates = sorted({item_id for item_id in disposition_ids if disposition_ids.count(item_id) > 1})
    if duplicates:
        raise AgentLoopError(
            "Review listed prior unresolved items more than once: " + ", ".join(duplicates)
        )
    unknown = sorted(set(disposition_ids) - set(unresolved_by_id))
    if unknown:
        _raise_unknown_prior_item_disposition(
            unknown,
            allowed_ids=tuple(unresolved_by_id),
            current_round_items=current_round_items,
        )
    if unresolved_items:
        missing = sorted(set(unresolved_by_id) - set(disposition_ids))
        if missing:
            raise AgentLoopError(
                "Review did not evaluate all prior unresolved items: " + ", ".join(missing)
            )
    # This is the live-response boundary. Historical comments still use the
    # permissive protocol parser so a resume does not invalidate older reviews.
    for disposition in dispositions:
        if disposition.disposition not in {"blocking", "same-pr"}:
            continue
        note = " ".join((disposition.note or "").strip().split())
        if not note or re.fullmatch(
            r"(?:(?:still|remains?)\s+)?(?:blocking|same[- ]pr|unresolved|not resolved|needs work)[.!]?",
            note,
            flags=re.I,
        ):
            raise AgentLoopError(
                f"Carried item {disposition.item_id} kept {disposition.disposition} requires "
                "an actionable note: explain the remaining defect on the reviewed head, "
                "relevant evidence, and the change or test needed. A status alone or a "
                "review-level summary is not an item-specific explanation."
            )
    return ParsedReview(
        state=parsed.state,
        summary=parsed.summary,
        blocking_items=parsed.blocking_items,
        followups=parsed.followups,
        dispositions=dispositions,
        raw_dispositions_text=parsed.raw_dispositions_text,
    )


def _upsert_human_requirements_ack_item(
    unresolved_items: Sequence[UnresolvedReviewItem],
    *,
    source_round: int,
    text: str,
) -> list[UnresolvedReviewItem]:
    retained = [
        item for item in unresolved_items if item.item_id != HUMAN_REQUIREMENTS_ACK_ITEM_ID
    ]
    retained.append(
        UnresolvedReviewItem(
            item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
            reviewer="Orchestrator",
            source_round=source_round,
            text=text,
            status="blocking",
            source_status="blocking",
            resolution_owners=("Orchestrator",),
            owner_states=(("Orchestrator", "pending"),),
        )
    )
    return retained


def _clear_human_requirements_ack_item(
    unresolved_items: Sequence[UnresolvedReviewItem],
) -> list[UnresolvedReviewItem]:
    return [item for item in unresolved_items if item.item_id != HUMAN_REQUIREMENTS_ACK_ITEM_ID]


def _reconcile_human_requirements_ack_item(
    unresolved_items: Sequence[UnresolvedReviewItem],
    *,
    coder_output: str | None,
    human_requirements,
    source_round: int,
) -> list[UnresolvedReviewItem]:
    if coder_output is None:
        return list(unresolved_items)

    prompt_context = render_coder_human_requirements_prompt_context(human_requirements)
    if not prompt_context.surfaced_requirement_ids and not prompt_context.requires_direct_discussion_ack:
        return _clear_human_requirements_ack_item(unresolved_items)
    try:
        structured_followup = None
        try:
            structured_implementation = validate_structured_issue_implementation(coder_output)
        except IssueImplementationConflictError as exc:
            structured_implementation = exc.payload
        except AgentLoopError:
            structured_implementation = None
        if isinstance(structured_implementation, StructuredIssueImplementation):
            validate_human_requirement_dispositions(
                structured_implementation.human_requirement_dispositions,
                surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
                context="issue_implementation.human_requirement_dispositions",
            )
            validate_structured_human_requirements_acknowledgement(
                structured_implementation.human_requirements.addressed_ids,
                dispositions=structured_implementation.human_requirement_dispositions,
                checked_discussion_directly=structured_implementation.human_requirements.checked_discussion_directly,
                surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
                requires_direct_discussion_ack=prompt_context.requires_direct_discussion_ack,
            )
        else:
            structured_followup = validate_structured_coder_followup(coder_output)
            if structured_followup is None:
                validate_human_requirements_acknowledgement(
                    coder_output,
                    surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
                    requires_direct_discussion_ack=prompt_context.requires_direct_discussion_ack,
                )
            else:
                validate_human_requirement_dispositions(
                    structured_followup.human_requirement_dispositions,
                    surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
                    context="coder_followup.human_requirement_dispositions",
                )
                validate_structured_human_requirements_acknowledgement(
                    structured_followup.human_requirements.addressed_ids,
                    dispositions=structured_followup.human_requirement_dispositions,
                    checked_discussion_directly=structured_followup.human_requirements.checked_discussion_directly,
                    surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
                    requires_direct_discussion_ack=prompt_context.requires_direct_discussion_ack,
                )
    except AgentLoopError as exc:
        return _upsert_human_requirements_ack_item(
            unresolved_items,
            source_round=source_round,
            text=str(exc),
        )
    return _clear_human_requirements_ack_item(unresolved_items)


_CONFIRMED_CONFLICT_HEAD_NOTE_PREFIX = "confirmed-head:"


def _upsert_merge_conflict_item(
    unresolved_items: Sequence[UnresolvedReviewItem],
    *,
    source_round: int,
    text: str,
    confirmed_head_sha: str | None,
) -> list[UnresolvedReviewItem]:
    retained = [item for item in unresolved_items if item.item_id != MERGE_CONFLICT_ITEM_ID]
    notes = (
        (f"{_CONFIRMED_CONFLICT_HEAD_NOTE_PREFIX}{confirmed_head_sha}",)
        if confirmed_head_sha
        else ()
    )
    retained.append(
        UnresolvedReviewItem(
            item_id=MERGE_CONFLICT_ITEM_ID,
            reviewer="Orchestrator",
            source_round=source_round,
            text=text,
            status="blocking",
            source_status="blocking",
            notes=notes,
            resolution_owners=("Orchestrator",),
            owner_states=(("Orchestrator", "pending"),),
        )
    )
    return retained


def _clear_merge_conflict_item(
    unresolved_items: Sequence[UnresolvedReviewItem],
) -> list[UnresolvedReviewItem]:
    return [item for item in unresolved_items if item.item_id != MERGE_CONFLICT_ITEM_ID]


def _confirmed_conflict_head(item: UnresolvedReviewItem) -> str | None:
    for note in item.notes:
        if note.startswith(_CONFIRMED_CONFLICT_HEAD_NOTE_PREFIX):
            return note[len(_CONFIRMED_CONFLICT_HEAD_NOTE_PREFIX):]
    return None


def _reconcile_merge_conflict_item(
    unresolved_items: Sequence[UnresolvedReviewItem],
    *,
    mergeability: PullRequestMergeability,
    source_round: int,
    current_head_sha: str | None,
) -> list[UnresolvedReviewItem]:
    """Keep the synthetic conflict item in sync with the latest GitHub probe.

    A confirmed `conflicted` state (re-)adds the blocking item, recording the
    head it was confirmed against. A positively `mergeable` result clears it.
    `unknown` clears it too -- but only when there was no previously
    confirmed conflict, or the head has since advanced -- so a transient
    GitHub mergeability computation window never creates a spurious blocker
    (#606). If a conflict was already confirmed and `current_head_sha` is
    unchanged, a later `unknown` (a probe hiccup, not new information) must
    not silently drop the blocker: doing so would let reviewers, checks, or
    a merge proceed against a branch GitHub last told us is still conflicted.
    """
    if mergeability.state == "conflicted":
        base = mergeability.base_branch or "the base branch"
        head = mergeability.head_sha or current_head_sha or "the current head"
        text = (
            f"PR branch has a merge conflict with `{base}` (GitHub reports "
            f"mergeable={mergeability.mergeable_raw or 'unknown'}, "
            f"mergeStateStatus={mergeability.merge_state_raw or 'unknown'}) at head `{head}`. "
            "Merge or rebase onto the current base and resolve the conflict before this "
            "PR can be reviewed or merged."
        )
        return _upsert_merge_conflict_item(
            unresolved_items,
            source_round=source_round,
            text=text,
            confirmed_head_sha=mergeability.head_sha or current_head_sha,
        )
    if mergeability.state == "mergeable":
        return _clear_merge_conflict_item(unresolved_items)

    # mergeability.state == "unknown"
    existing = next(
        (item for item in unresolved_items if item.item_id == MERGE_CONFLICT_ITEM_ID), None
    )
    if existing is None:
        return list(unresolved_items)
    if _confirmed_conflict_head(existing) == current_head_sha:
        # Same head the conflict was confirmed against; preserve the blocker
        # until GitHub positively reports `mergeable` or the head advances.
        return list(unresolved_items)
    return _clear_merge_conflict_item(unresolved_items)


def _validate_structured_coder_followup_items(
    parsed: StructuredCoderFollowup,
    *,
    unresolved_items: Sequence[UnresolvedReviewItem],
) -> None:
    excluded_synthetic_ids = {HUMAN_REQUIREMENTS_ACK_ITEM_ID, MERGE_CONFLICT_ITEM_ID}
    allowed_ids = [
        item.item_id for item in unresolved_items if item.item_id not in excluded_synthetic_ids
    ]
    # Future follow-ups are informational carry-forwards, not actionable this round --
    # they are not shown in the same-PR-only follow-up prompt, so the coder cannot
    # always classify them into addressed/remaining/disputed. Allow (but do not
    # require) referencing them when they are shown in the general prompt.
    required_ids = [
        item.item_id
        for item in unresolved_items
        if item.item_id not in excluded_synthetic_ids and item.status != "future"
    ]
    listed_ids = [*parsed.addressed_items, *parsed.remaining_items, *parsed.disputed_items]
    duplicates = sorted({item_id for item_id in listed_ids if listed_ids.count(item_id) > 1})
    if duplicates:
        raise AgentLoopError(
            "Coder follow-up listed unresolved reviewer item IDs more than once: "
            + ", ".join(duplicates)
        )
    unknown = sorted(set(listed_ids) - set(allowed_ids))
    if unknown:
        raise AgentLoopError(
            "Coder follow-up referenced unknown unresolved reviewer item IDs: "
            + ", ".join(unknown)
        )
    missing = sorted(set(required_ids) - set(listed_ids))
    if missing:
        raise AgentLoopError(
            "Coder follow-up did not classify all unresolved reviewer items into "
            "addressed, remaining, or disputed: "
            + ", ".join(missing)
        )


def _validate_coder_followup_response(
    text: str,
    *,
    unresolved_items: Sequence[UnresolvedReviewItem],
    human_requirements,
) -> StructuredCoderFollowup | str:
    prompt_context = render_coder_human_requirements_prompt_context(human_requirements)
    structured_followup = validate_structured_coder_followup(text)
    if structured_followup is not None:
        _validate_structured_coder_followup_items(
            structured_followup,
            unresolved_items=unresolved_items,
        )
        validate_human_requirement_dispositions(
            structured_followup.human_requirement_dispositions,
            surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
            context="coder_followup.human_requirement_dispositions",
        )
        validate_structured_human_requirements_acknowledgement(
            structured_followup.human_requirements.addressed_ids,
            dispositions=structured_followup.human_requirement_dispositions,
            checked_discussion_directly=structured_followup.human_requirements.checked_discussion_directly,
            surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
            requires_direct_discussion_ack=prompt_context.requires_direct_discussion_ack,
        )
        return structured_followup

    raise AgentLoopError("Coder response did not use the required structured format.")


def _apply_unresolved_item_dispositions(
    unresolved_items: Sequence[UnresolvedReviewItem],
    dispositions_by_item: dict[str, list[ReviewItemDisposition]],
    *,
    same_status: str = "same-pr",
    retain_future: bool = True,
    reconciliation_mode: str = "aggregate",
) -> tuple[list[UnresolvedReviewItem], list[UnresolvedReviewItem]]:
    if reconciliation_mode not in {"aggregate", "owner-scoped"}:
        raise AgentLoopError(
            "reconciliation_mode must be `aggregate` or `owner-scoped`."
        )
    next_unresolved: list[UnresolvedReviewItem] = []
    future_items: list[UnresolvedReviewItem] = []
    for item in unresolved_items:
        dispositions = dispositions_by_item.get(item.item_id, [])
        if not dispositions:
            next_unresolved.append(item)
            continue
        # `text` is the item’s canonical claim.  Dispositions may add evidence
        # for that claim, but must never rewrite it: a reviewer that discovers a
        # different concern must file a fresh item rather than silently changing
        # what this stable ID means.
        text = item.text
        notes = list(item.notes)
        outcomes = {disposition.disposition for disposition in dispositions}
        owners = item.resolution_owners or (item.reviewer,)
        owner_states = dict(item.owner_states or ((owner, "pending") for owner in owners))
        owner_evidence = dict(item.owner_evidence)
        owner_dispositions = dict(item.owner_dispositions)
        if reconciliation_mode == "owner-scoped":
            # A reviewer who was not an owner may supply evidence, but only a
            # blocking/same-PR disposition creates a durable new obligation.
            # A non-owner resolved disposition is never a waiver.
            for disposition in dispositions:
                if disposition.disposition in {"blocking", same_status}:
                    if disposition.reviewer not in owners:
                        owners = (*owners, disposition.reviewer)
                    owner_states[disposition.reviewer] = "pending"
                    owner_dispositions[disposition.reviewer] = disposition.disposition
                elif (
                    disposition.disposition in {"resolved", "future"}
                    and disposition.reviewer in owners
                ):
                    # `future` is a valid approving disposition for a carried
                    # item. In owner-scoped mode it clears only this owner's
                    # obligation; other owners must still provide their own
                    # clearing disposition before the item can leave the
                    # active ledger.
                    owner_states[disposition.reviewer] = "cleared"
                    owner_dispositions[disposition.reviewer] = disposition.disposition
                if disposition.note:
                    owner_evidence[disposition.reviewer] = disposition.note
            owners = tuple(dict.fromkeys(owners))
            for owner in owners:
                owner_states.setdefault(owner, "pending")
        for disposition in dispositions:
            if disposition.note:
                note_text = f"{disposition.reviewer}: {disposition.note}"
                if note_text not in notes:
                    notes.append(note_text)
            elif same_status == "same-pr" and disposition.disposition in {"blocking", "same-pr"}:
                note_text = (
                    f"{disposition.reviewer}: saved review kept this item "
                    f"{disposition.disposition} without an item-specific explanation. "
                    "Consult the review summary; do not infer a new claim."
                )
                if note_text not in notes:
                    notes.append(note_text)
        if "blocking" in outcomes:
            next_unresolved.append(
                UnresolvedReviewItem(
                    item_id=item.item_id,
                    reviewer=item.reviewer,
                    source_round=item.source_round,
                    text=text,
                    status="blocking",
                    source_status=item.source_status,
                    notes=tuple(notes),
                    fix_scope=item.fix_scope,
                    resolution_owners=owners,
                    owner_states=tuple((owner, owner_states[owner]) for owner in owners),
                    owner_evidence=tuple((owner, owner_evidence[owner]) for owner in owners if owner in owner_evidence),
                    owner_dispositions=tuple((owner, owner_dispositions[owner]) for owner in owners if owner in owner_dispositions),
                )
            )
            continue
        if same_status in outcomes:
            next_unresolved.append(
                UnresolvedReviewItem(
                    item_id=item.item_id,
                    reviewer=item.reviewer,
                    source_round=item.source_round,
                    text=text,
                    status=same_status,
                    source_status=item.source_status,
                    notes=tuple(notes),
                    fix_scope=item.fix_scope,
                    resolution_owners=owners,
                    owner_states=tuple((owner, owner_states[owner]) for owner in owners),
                    owner_evidence=tuple((owner, owner_evidence[owner]) for owner in owners if owner in owner_evidence),
                    owner_dispositions=tuple((owner, owner_dispositions[owner]) for owner in owners if owner in owner_dispositions),
                )
            )
            continue
        if reconciliation_mode == "owner-scoped":
            pending_owners = [owner for owner in owners if owner_states.get(owner) != "cleared"]
            if pending_owners:
                next_unresolved.append(
                    UnresolvedReviewItem(
                        item_id=item.item_id,
                        reviewer=item.reviewer,
                        source_round=item.source_round,
                        text=text,
                        status=item.status,
                        source_status=item.source_status,
                        notes=tuple(notes),
                        fix_scope=item.fix_scope,
                        resolution_owners=owners,
                        owner_states=tuple((owner, owner_states[owner]) for owner in owners),
                        owner_evidence=tuple((owner, owner_evidence[owner]) for owner in owners if owner in owner_evidence),
                        owner_dispositions=tuple((owner, owner_dispositions[owner]) for owner in owners if owner in owner_dispositions),
                    )
                )
                continue
        if "future" in outcomes or any(
            outcome == "future" for outcome in owner_dispositions.values()
        ):
            future_item = UnresolvedReviewItem(
                item_id=item.item_id,
                reviewer=item.reviewer,
                source_round=item.source_round,
                text=text,
                status="future",
                source_status=item.source_status,
                notes=tuple(notes),
                fix_scope=item.fix_scope,
                resolution_owners=owners,
                owner_states=tuple((owner, owner_states[owner]) for owner in owners),
                owner_evidence=tuple((owner, owner_evidence[owner]) for owner in owners if owner in owner_evidence),
                owner_dispositions=tuple((owner, owner_dispositions[owner]) for owner in owners if owner in owner_dispositions),
            )
            if retain_future:
                next_unresolved.append(future_item)
            else:
                future_items.append(future_item)
    return next_unresolved, future_items


def _collect_prior_compact_summaries(
    prior_items: Sequence[UnresolvedReviewItem],
    remaining_items: Sequence[UnresolvedReviewItem],
    dispositions_by_item: dict[str, list[ReviewItemDisposition]],
) -> tuple[str, ...]:
    """Render append-only summaries for prior items that just left the active ledger."""
    remaining_ids = {item.item_id for item in remaining_items}
    summaries: list[str] = []
    for item in sorted(prior_items, key=lambda prior: prior.item_id):
        if item.item_id in remaining_ids:
            continue
        dispositions = dispositions_by_item.get(item.item_id, [])
        if not dispositions:
            continue
        outcomes = {disposition.disposition for disposition in dispositions}
        label = (
            "future follow-up"
            if "future" in outcomes and not {"blocking", "same-pr", "same-plan"} & outcomes
            else "resolved"
        )
        lines = [
            f"[{item.item_id}] {label}: {item.reviewer} {item.source_status or item.status} item from round {item.source_round}",
            "Original item text:",
            item.text,
        ]
        notes = list(item.notes)
        for disposition in sorted(
            dispositions,
            key=lambda disposition: (disposition.reviewer, disposition.item_id, disposition.disposition, disposition.note or ""),
        ):
            if disposition.note:
                note = f"{disposition.reviewer}: {disposition.disposition}: {disposition.note}"
            else:
                note = f"{disposition.reviewer}: {disposition.disposition}"
            if note not in notes:
                notes.append(note)
        if notes:
            lines.append("Disposition updates:")
            lines.extend(f"- {note}" for note in notes)
        summaries.append("\n".join(lines))
    return tuple(summaries)


def _validate_plan_review_response(
    text: str,
    *,
    reviewer: str,
    unresolved_items: Sequence[UnresolvedReviewItem],
    current_round_items: Sequence[UnresolvedReviewItem] = (),
    surfaced_requirement_ids: Sequence[str] = (),
) -> ParsedPlanReview:
    parsed = parse_plan_review(text, reviewer=reviewer)
    validate_human_requirement_dispositions(
        parsed.human_requirement_dispositions,
        surfaced_requirement_ids=surfaced_requirement_ids,
        context="plan_review.human_requirement_dispositions",
    )

    unresolved_by_id = {item.item_id: item for item in unresolved_items}
    dispositions = _maybe_fill_resolved_dispositions_from_prose(
        parsed,
        reviewer=reviewer,
        unresolved_items=unresolved_items,
    )
    if not dispositions and not unresolved_items:
        return parsed
    disposition_ids = [item.item_id for item in dispositions]
    duplicates = sorted({item_id for item_id in disposition_ids if disposition_ids.count(item_id) > 1})
    if duplicates:
        raise AgentLoopError(
            "Plan review listed prior unresolved plan items more than once: "
            + ", ".join(duplicates)
        )
    unknown = sorted(set(disposition_ids) - set(unresolved_by_id))
    if unknown:
        _raise_unknown_prior_item_disposition(
            unknown,
            allowed_ids=tuple(unresolved_by_id),
            current_round_items=current_round_items,
        )
    if unresolved_items:
        missing = sorted(set(unresolved_by_id) - set(disposition_ids))
        if missing:
            raise AgentLoopError(
                "Plan review did not evaluate all prior unresolved plan items: "
                + ", ".join(missing)
            )
    return ParsedPlanReview(
        state=parsed.state,
        summary=parsed.summary,
        items=parsed.items,
        dispositions=dispositions,
        raw_dispositions_text=parsed.raw_dispositions_text,
    )


def _record_prior_item_disposition(
    prior_dispositions: dict[str, list[ReviewItemDisposition]],
    disposition: ReviewItemDisposition,
    *,
    flow: str,
    round_number: int,
    subject: str,
    reviewer_name: str,
) -> None:
    if disposition.item_id not in prior_dispositions:
        raise AgentLoopError(
            "Resumed "
            f"{flow} round {round_number} reconstructed prior items "
            f"{', '.join(sorted(prior_dispositions)) or '(none)'}, but {reviewer_name} "
            f"dispositioned unknown item `{disposition.item_id}` for subject `{subject}`."
        )
    prior_dispositions[disposition.item_id].append(disposition)


def _format_same_pr_unresolved_items(items: Sequence[UnresolvedReviewItem]) -> str:
    lines: list[str] = []
    for item in items:
        lines.append(
            f"{item.reviewer} same-PR follow-up [{item.item_id}] from round {item.source_round}:"
        )
        lines.append(f"- {item.text}")
        if item.fix_scope:
            lines.append("Reviewer fix scope: " + ", ".join(item.fix_scope))
        if item.resolution_owners:
            owner_states = dict(item.owner_states)
            lines.append(
                "Resolution owners: "
                + ", ".join(
                    f"{owner} ({owner_states.get(owner, 'pending')})"
                    for owner in item.resolution_owners
                )
            )
        if item.notes:
            lines.append("Latest reviewer updates:")
            lines.extend(f"- {note}" for note in item.notes)
        lines.append("")
    return "\n".join(lines).strip()


def apply_item_dispositions(
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions_by_item: dict[str, list],
    *,
    same_status: str,
    retain_future: bool,
) -> tuple[list[UnresolvedReviewItem], list[UnresolvedReviewItem]]:
    """Public wrapper around _apply_unresolved_item_dispositions for skill_runner use."""
    return _apply_unresolved_item_dispositions(
        prior_items,
        dispositions_by_item,
        same_status=same_status,
        retain_future=retain_future,
    )


def _format_unresolved_items_for_coder(items: Sequence[UnresolvedReviewItem]) -> str:
    lines: list[str] = []
    for item in items:
        lines.append(
            f"{item.reviewer} unresolved {item.status} item [{item.item_id}] from round {item.source_round}:"
        )
        lines.append(f"- {item.text}")
        if item.notes:
            lines.append("Latest reviewer updates:")
            lines.extend(f"- {note}" for note in item.notes)
        lines.append("")
    return "\n".join(lines).strip()
