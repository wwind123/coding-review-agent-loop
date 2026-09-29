"""Unresolved review item ledger and response validation helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .errors import (
    AgentLoopError,
    HumanDecisionRequiredError,
    IssueImplementationConflictError,
    UnknownPriorItemDispositionError,
)
from .github import PullRequestMergeability
from .prompts import render_coder_human_requirements_prompt_context
from .protocol import (
    ParseDegradation,
    ParsedPlanReview,
    ParsedReview,
    ReviewItemDisposition,
    ReviewSubItem,
    StructuredCoderFollowup,
    StructuredIssueImplementation,
    UnresolvedReviewItem,
    CI_MACHINE_OBLIGATION_KINDS,
    EVIDENCE_OBLIGATION_KIND,
    MACHINE_AUTHORITY,
    MACHINE_OBLIGATION_KINDS,
    UNKNOWN_MACHINE_AUTHORITY,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
    parse_plan_review,
    parse_pr_review,
    parse_human_requirements_acknowledgement,
    parse_historical_structured_coder_followup,
    parse_historical_structured_issue_implementation,
    validate_human_requirements_acknowledgement,
    validate_structured_coder_followup,
    validate_structured_human_requirements_acknowledgement,
    validate_structured_issue_implementation,
    validate_human_requirement_dispositions,
    _normalized_review_item_text,
)

HUMAN_REQUIREMENTS_ACK_ITEM_ID = "item-human-requirements-acknowledgement"
MERGE_CONFLICT_ITEM_ID = "item-merge-conflict"
CODER_DISPUTE_NOTE_PREFIX = "Coder disputes this item"
MANAGED_CI_OBLIGATION_KIND = "managed-exact-head-ci"
ORDINARY_CI_OBLIGATION_KIND = "github-pr-checks"
MIGRATION_OBLIGATION_KIND = "alembic-migration"
MERGE_CONFLICT_OBLIGATION_KIND = "merge-conflict"
HUMAN_REQUIREMENTS_OBLIGATION_KIND = "human-requirements-acknowledgement"
UNKNOWN_OBLIGATION_KIND = "unknown"
CODER_NON_CLASSIFIABLE_ITEM_IDS = frozenset(
    {HUMAN_REQUIREMENTS_ACK_ITEM_ID, MERGE_CONFLICT_ITEM_ID}
)
ALL_RESOLVED_PROSE_RE = re.compile(
    r"^all (?:prior items|listed items|carried-forward items) are resolved\.?$"
    r"|^all prior unresolved items have been resolved\.?$",
    re.I,
)


def select_coder_followup_items(
    unresolved_items: Sequence[UnresolvedReviewItem],
) -> tuple[UnresolvedReviewItem, ...]:
    """Return exactly the records that belong to coder item classification.

    The acknowledgement and merge-conflict records have dedicated protocol
    paths. Every other record, including machine-owned repair obligations,
    remains in the coder's addressed/remaining/disputed namespace.
    """
    return tuple(
        item
        for item in unresolved_items
        if item.item_id not in CODER_NON_CLASSIFIABLE_ITEM_IDS
        # Human-only exact-head evidence is never coder work (#1068).
        and not _is_evidence_obligation(item)
    )


def coder_followup_is_ci_repair(
    coder_followup_items: Sequence[UnresolvedReviewItem],
) -> bool:
    """Whether a coder round exists only to repair tool-owned CI failures.

    The set must be non-empty: an acknowledgement-only round has no
    classifiable items and must not be reported as a CI repair (#1024).
    """
    return bool(coder_followup_items) and all(
        _machine_obligation_is_ci(item) for item in coder_followup_items
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
    authority: str | None = None,
    obligation_kind: str | None = None,
    lifecycle: str | None = None,
    failed_head_sha: str | None = None,
    candidate_head_sha: str | None = None,
    obligation_identity: str | None = None,
    sub_items: Sequence[str] = (),
) -> UnresolvedReviewItem:
    if obligation_kind is not None and authority is None:
        authority = MACHINE_AUTHORITY
    if authority in {MACHINE_AUTHORITY, UNKNOWN_MACHINE_AUTHORITY}:
        obligation_kind = obligation_kind or UNKNOWN_OBLIGATION_KIND
        lifecycle = lifecycle or "repair_required"
        owners = ()
    else:
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
        authority=authority,
        obligation_kind=obligation_kind,
        lifecycle=lifecycle,
        failed_head_sha=failed_head_sha,
        candidate_head_sha=candidate_head_sha,
        obligation_identity=obligation_identity or (
            f"{obligation_kind}:item-{item_number}" if obligation_kind else None
        ),
        # Reviewer findings only; machine obligations never carry sub-items.
        sub_items=tuple(
            ReviewSubItem(sub_item_id=f"item-{item_number}.s{position}", text=statement)
            for position, statement in enumerate(sub_items, 1)
        ) if authority not in {MACHINE_AUTHORITY, UNKNOWN_MACHINE_AUTHORITY} else (),
    )


def _is_machine_obligation(item: UnresolvedReviewItem) -> bool:
    return item.is_machine_obligation


def _machine_obligation_is_ci(item: UnresolvedReviewItem) -> bool:
    return _is_machine_obligation(item) and item.obligation_kind in CI_MACHINE_OBLIGATION_KINDS


def _is_evidence_obligation(item: UnresolvedReviewItem) -> bool:
    """Whether this record is a human-only exact-head evidence request (#1068)."""
    return _is_machine_obligation(item) and item.obligation_kind == EVIDENCE_OBLIGATION_KIND


def _pending_evidence_obligations(
    unresolved_items: Sequence[UnresolvedReviewItem],
) -> tuple[UnresolvedReviewItem, ...]:
    return tuple(
        item
        for item in unresolved_items
        if _is_evidence_obligation(item)
        and item.status in {"blocking", "same-pr"}
        and item.lifecycle in {"evidence_deferred", "evidence_frozen"}
    )


def _frozen_evidence_obligations(
    unresolved_items: Sequence[UnresolvedReviewItem],
) -> tuple[UnresolvedReviewItem, ...]:
    return tuple(
        item
        for item in _pending_evidence_obligations(unresolved_items)
        if item.lifecycle == "evidence_frozen"
    )


def evidence_obligation_identity(text: str, reviewer: str) -> str:
    """Stable identity of one reviewer's request, independent of item numbering."""
    digest = hashlib.sha256(
        json.dumps(
            [_normalized_review_item_text(text), reviewer], ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()[:24]
    return f"{EVIDENCE_OBLIGATION_KIND}:{digest}"


def _upsert_evidence_obligation(
    unresolved_items: Sequence[UnresolvedReviewItem],
    *,
    item_number: int,
    reviewer: str,
    text: str,
    source_round: int,
    current_head_sha: str | None,
    clearances: Sequence[tuple[str, str]] = (),
) -> tuple[list[UnresolvedReviewItem], bool]:
    """Upsert one reviewer's evidence request by identity.

    Returns the ledger and whether a new item number was consumed.  A request
    whose identity was cleared at ``current_head_sha`` is ignored, so a
    re-emission in the same response that cleared it cannot recreate it; at
    any other head the same request is a new deferred obligation.
    """
    identity = evidence_obligation_identity(text, reviewer)
    if current_head_sha and (identity, current_head_sha) in set(clearances):
        return list(unresolved_items), False
    for index, existing in enumerate(unresolved_items):
        if _is_evidence_obligation(existing) and existing.obligation_identity == identity:
            result = list(unresolved_items)
            if existing.status not in {"blocking", "same-pr"} or existing.lifecycle not in {
                "evidence_deferred", "evidence_frozen"
            }:
                result[index] = replace(
                    existing,
                    status="blocking",
                    source_status="blocking",
                    lifecycle="evidence_deferred",
                    candidate_head_sha=None,
                )
            return result, False
    item = UnresolvedReviewItem(
        item_id=f"item-{item_number}",
        reviewer=reviewer,
        source_round=source_round,
        text=text,
        status="blocking",
        source_status="blocking",
        resolution_owners=(reviewer,),
        owner_states=((reviewer, "pending"),),
        authority=MACHINE_AUTHORITY,
        obligation_kind=EVIDENCE_OBLIGATION_KIND,
        lifecycle="evidence_deferred",
        obligation_identity=identity,
    )
    return [*unresolved_items, item], True


def freeze_evidence_obligations(
    unresolved_items: Sequence[UnresolvedReviewItem], *, head_sha: str
) -> list[UnresolvedReviewItem]:
    """Freeze every pending evidence obligation at the clean head ``head_sha``."""
    return [
        replace(item, lifecycle="evidence_frozen", candidate_head_sha=head_sha)
        if _is_evidence_obligation(item)
        and item.status in {"blocking", "same-pr"}
        and item.lifecycle in {"evidence_deferred", "evidence_frozen"}
        else item
        for item in unresolved_items
    ]


def release_evidence_freeze(
    unresolved_items: Sequence[UnresolvedReviewItem],
) -> list[UnresolvedReviewItem]:
    """Return every frozen evidence obligation to the deferred state."""
    return [
        replace(item, lifecycle="evidence_deferred", candidate_head_sha=None)
        if _is_evidence_obligation(item) and item.lifecycle == "evidence_frozen"
        else item
        for item in unresolved_items
    ]


def _machine_obligation_requires_repair(
    item: UnresolvedReviewItem, *, current_head_sha: str | None
) -> bool:
    if not _is_machine_obligation(item) or item.status not in {"blocking", "same-pr"}:
        return False
    if _is_evidence_obligation(item):
        # No coder round can produce human-only evidence.
        return False
    if item.obligation_kind == UNKNOWN_OBLIGATION_KIND:
        return True
    if item.lifecycle == "repair_required":
        return not bool(
            item.failed_head_sha
            and current_head_sha
            and current_head_sha != item.failed_head_sha
        )
    if item.lifecycle not in {"awaiting_current_head_review", "qualification_ready", "qualifying"}:
        return True
    return bool(
        item.failed_head_sha
        and current_head_sha
        and current_head_sha == item.failed_head_sha
    )


def _machine_obligation_is_revalidation_candidate(
    item: UnresolvedReviewItem, *, current_head_sha: str | None
) -> bool:
    return (
        _is_machine_obligation(item)
        and not _is_evidence_obligation(item)
        and item.obligation_kind in CI_MACHINE_OBLIGATION_KINDS
        and item.lifecycle in {"awaiting_current_head_review", "qualification_ready", "qualifying"}
        and bool(item.candidate_head_sha)
        and item.candidate_head_sha == current_head_sha
        and item.failed_head_sha != current_head_sha
    )


def _machine_obligation_is_pending_non_ci(item: UnresolvedReviewItem) -> bool:
    return _is_machine_obligation(item) and not _machine_obligation_is_ci(item) and item.status in {
        "blocking", "same-pr"
    }


def _upsert_machine_obligation(
    unresolved_items: Sequence[UnresolvedReviewItem],
    *,
    item_number: int,
    kind: str,
    source_round: int,
    text: str,
    failed_head_sha: str | None,
) -> list[UnresolvedReviewItem]:
    """Upsert one stable source-specific obligation instead of accumulating items."""
    if kind == EVIDENCE_OBLIGATION_KIND:
        # Evidence obligations are keyed per request identity, never per kind.
        raise AgentLoopError(
            "Evidence obligations must be upserted by identity with "
            "_upsert_evidence_obligation, not as a kind singleton."
        )
    if kind not in MACHINE_OBLIGATION_KINDS or kind == UNKNOWN_OBLIGATION_KIND:
        kind = UNKNOWN_OBLIGATION_KIND
        failed_head_sha = None
    for index, existing in enumerate(unresolved_items):
        if _is_machine_obligation(existing) and existing.obligation_kind == kind:
            updated = replace(
                existing,
                reviewer=(existing.reviewer if existing.reviewer else "Orchestrator"),
                source_round=source_round,
                text=text,
                status="blocking",
                source_status="blocking",
                notes=existing.notes,
                resolution_owners=(),
                owner_states=(),
                authority=MACHINE_AUTHORITY if kind != UNKNOWN_OBLIGATION_KIND else UNKNOWN_MACHINE_AUTHORITY,
                obligation_kind=kind,
                lifecycle="repair_required",
                failed_head_sha=failed_head_sha,
                candidate_head_sha=None,
                obligation_identity=existing.obligation_identity or f"{kind}:{existing.item_id}",
            )
            result = list(unresolved_items)
            result[index] = updated
            return result
    item = _next_unresolved_item(
        item_number=item_number,
        reviewer=(
            "GitHub managed exact-head CI"
            if kind == MANAGED_CI_OBLIGATION_KIND
            else "GitHub PR checks"
            if kind == ORDINARY_CI_OBLIGATION_KIND
            else "Alembic migration validation"
            if kind == MIGRATION_OBLIGATION_KIND
            else "Orchestrator"
        ),
        source_round=source_round,
        text=text,
        status="blocking",
        authority=(MACHINE_AUTHORITY if kind != UNKNOWN_OBLIGATION_KIND else UNKNOWN_MACHINE_AUTHORITY),
        obligation_kind=kind,
        lifecycle="repair_required",
        failed_head_sha=failed_head_sha,
    )
    return [*unresolved_items, item]


def _advance_machine_obligations_for_head(
    unresolved_items: Sequence[UnresolvedReviewItem],
    *,
    current_head_sha: str | None,
) -> list[UnresolvedReviewItem]:
    """Bind machine revalidation to the first strictly-new candidate head."""
    if not current_head_sha:
        return list(unresolved_items)
    result: list[UnresolvedReviewItem] = []
    for item in unresolved_items:
        if (
            not _is_machine_obligation(item)
            or item.lifecycle == "cleared"
            # Head handling for evidence is owned by the freeze logic.
            or _is_evidence_obligation(item)
        ):
            result.append(item)
            continue
        if item.lifecycle == "repair_required":
            if item.failed_head_sha and item.failed_head_sha != current_head_sha:
                result.append(
                    replace(
                        item,
                        lifecycle="awaiting_current_head_review",
                        candidate_head_sha=current_head_sha,
                        resolution_owners=(),
                        owner_states=(),
                    )
                )
            else:
                result.append(item)
            continue
        if item.failed_head_sha == current_head_sha:
            # A force-push revert/reset-and-repush must never be allowed to
            # requalify the failed head. Return to repair-required instead of
            # constructing an invalid candidate and leaking ValueError out of
            # the orchestration loop.
            result.append(
                replace(item, lifecycle="repair_required", candidate_head_sha=None)
            )
        elif item.candidate_head_sha and item.candidate_head_sha != current_head_sha:
            result.append(
                replace(item, lifecycle="awaiting_current_head_review", candidate_head_sha=current_head_sha)
            )
        else:
            result.append(item)
    return result


def _clear_machine_obligations(
    unresolved_items: Sequence[UnresolvedReviewItem], *, kind: str
) -> list[UnresolvedReviewItem]:
    """Clear all historical/current obligations of exactly one authority kind."""
    return [
        item
        for item in unresolved_items
        if not (_is_machine_obligation(item) and item.obligation_kind == kind)
    ]


def _set_machine_obligation_lifecycle(
    unresolved_items: Sequence[UnresolvedReviewItem],
    *,
    kind: str,
    lifecycle: str,
) -> list[UnresolvedReviewItem]:
    if lifecycle not in {"repair_required", "awaiting_current_head_review", "qualification_ready", "qualifying", "cleared"}:
        raise AgentLoopError(f"Unknown machine-obligation lifecycle: {lifecycle}.")
    if kind == EVIDENCE_OBLIGATION_KIND:
        raise AgentLoopError(
            "Evidence obligation lifecycles are owned by the evidence freeze helpers."
        )
    return [
        replace(item, lifecycle=lifecycle)
        if _is_machine_obligation(item) and item.obligation_kind == kind
        else item
        for item in unresolved_items
    ]


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
                replace(item, notes=(*item.notes, note))
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
    raise HumanDecisionRequiredError(
        f"Reviewer did not resolve {len(disputed_still_blocking)} disputed item(s) "
        "after seeing coder counter-evidence. Human decision required to resolve the "
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
    architecture_status_mode: str,
) -> ParsedReview:
    parsed = parse_pr_review(
        text, reviewer=reviewer, architecture_status_mode=architecture_status_mode
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
    # Sub-item keys must name sub-items of that entry's carried item; anything
    # else (including any key on an item without sub-items or on a machine
    # obligation) is dropped with a degradation instead of rejecting the
    # response (#958).
    sub_item_records: list[ParseDegradation] = list(parsed.sub_item_degradations)
    validated: list[ReviewItemDisposition] = []
    for disposition in dispositions:
        if disposition.sub_item_dispositions:
            carried = unresolved_by_id[disposition.item_id]
            allowed = (
                set()
                if _is_machine_obligation(carried)
                else {sub.sub_item_id for sub in carried.sub_items}
            )
            kept_pairs = tuple(
                pair for pair in disposition.sub_item_dispositions if pair[0] in allowed
            )
            for key, _verdict in disposition.sub_item_dispositions:
                if key not in allowed and len(sub_item_records) < 16:
                    sub_item_records.append(
                        ParseDegradation.build(
                            element_path=f"prior_item_dispositions[{disposition.item_id}].sub_item_dispositions",
                            rule="sub-item-key-is-not-a-sub-item-of-this-item",
                            observed=key,
                            outcome="claim-dropped",
                        )
                    )
            disposition = replace(disposition, sub_item_dispositions=kept_pairs)
        validated.append(disposition)
    dispositions = tuple(validated)
    # This is the live-response boundary. Historical comments still use the
    # permissive protocol parser so a resume does not invalidate older reviews.
    for disposition in dispositions:
        if disposition.disposition not in {"blocking", "same-pr"}:
            continue
        carried = unresolved_by_id[disposition.item_id]
        if carried.sub_items and _entry_completes_sub_items(
            disposition,
            {sub.sub_item_id for sub in carried.sub_items if sub.status == "open"},
            same_status="same-pr",
        ):
            # Resolving every remaining sub-item needs no remaining-defect
            # note; reconciliation derives `resolved` or, if the completion
            # does not take effect, synthesizes the explanation.
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
    # Replace only the dispositions: the assessment, its degradation records
    # and every other parsed field survive a carried-item round (#925).
    return replace(
        parsed, dispositions=dispositions, sub_item_degradations=tuple(sub_item_records)
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
            authority=MACHINE_AUTHORITY,
            obligation_kind=HUMAN_REQUIREMENTS_OBLIGATION_KIND,
            lifecycle="repair_required",
            obligation_identity=HUMAN_REQUIREMENTS_OBLIGATION_KIND,
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
    # Parse the applicable structured response mode first.  Schema, footer,
    # layout, and reviewer-item errors belong to the response validator; they
    # must not be relabeled as a human-requirements obligation here.
    structured_implementation: StructuredIssueImplementation | None = None
    structured_followup: StructuredCoderFollowup | None = None
    try:
        structured_implementation = parse_historical_structured_issue_implementation(coder_output)
    except IssueImplementationConflictError as exc:
        structured_implementation = exc.payload
    except AgentLoopError:
        structured_implementation = None

    if structured_implementation is None:
        try:
            structured_followup = parse_historical_structured_coder_followup(coder_output)
        except AgentLoopError:
            structured_followup = None

    if isinstance(structured_implementation, StructuredIssueImplementation):
        try:
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
        except AgentLoopError as exc:
            return _upsert_human_requirements_ack_item(
                unresolved_items,
                source_round=source_round,
                text=str(exc),
            )
        return _clear_human_requirements_ack_item(unresolved_items)

    if isinstance(structured_followup, StructuredCoderFollowup):
        try:
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

    # A legacy markdown acknowledgement is itself an applicable response mode
    # only when the response attempted to provide its marker or section. A
    # generic free-form/structured response with no parseable acknowledgement
    # must remain a response-format failure, not mint a misleading gate.
    parsed_legacy_ack = parse_human_requirements_acknowledgement(coder_output)
    if (
        parsed_legacy_ack.marker_present
        or parsed_legacy_ack.section_present
        or HUMAN_REQUIREMENTS_ADDRESSED_MARKER in coder_output
    ):
        try:
            validate_human_requirements_acknowledgement(
                coder_output,
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

    # Do not clear a pre-existing acknowledgement record without valid
    # acknowledgement evidence. The original validation failure remains the
    # authoritative diagnostic for this turn.
    return list(unresolved_items)


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
            authority=MACHINE_AUTHORITY,
            obligation_kind=MERGE_CONFLICT_OBLIGATION_KIND,
            lifecycle="repair_required",
            failed_head_sha=confirmed_head_sha,
            obligation_identity=MERGE_CONFLICT_OBLIGATION_KIND,
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
    allowed_ids = [item.item_id for item in select_coder_followup_items(unresolved_items)]
    # The caller supplies the dispatch-specific item set that was rendered in
    # the prompt. Every selected item is therefore required in the coder's
    # addressed/remaining/disputed partition, including retained future work
    # on general and merge-conflict handoffs.
    required_ids = [item.item_id for item in select_coder_followup_items(unresolved_items)]
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
    required_architecture_impact_contract: int = 0,
    delivered_risk_test_matrix=None,
    delivered_risk_test_matrix_identity: str | None = None,
    required_risk_test_matrix_contract: int = 0,
    authoritative_test_observations=None,
    delivered_risk_test_matrix_row_ids=None,
    execution_catalog=None,
    architecture_status_mode: str,
) -> StructuredCoderFollowup | str:
    prompt_context = render_coder_human_requirements_prompt_context(human_requirements)
    structured_followup = validate_structured_coder_followup(
        text,
        required_architecture_impact_contract=required_architecture_impact_contract,
        delivered_risk_test_matrix=delivered_risk_test_matrix,
        delivered_risk_test_matrix_identity=delivered_risk_test_matrix_identity,
        required_risk_test_matrix_contract=required_risk_test_matrix_contract,
        authoritative_test_observations=authoritative_test_observations,
        delivered_risk_test_matrix_row_ids=delivered_risk_test_matrix_row_ids,
        execution_catalog=execution_catalog,
        architecture_status_mode=architecture_status_mode,
        # Advisory sub-item claims may name only sub-items still open on the
        # carried items (#958); invalid claims are dropped, never rejected.
        expected_sub_item_ids={
            item.item_id: tuple(
                sub.sub_item_id for sub in item.sub_items if sub.status == "open"
            )
            for item in unresolved_items
            if item.sub_items
        },
    )
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
    evidence_response_head: str | None = None,
    configured_reviewers: Sequence[str] | None = None,
    evidence_clearances: list[tuple[str, str]] | None = None,
    round_number: int | None = None,
    cleared_items_progress: list["ClearedItemProgress"] | None = None,
    sub_item_degradations: list[str] | None = None,
) -> tuple[list[UnresolvedReviewItem], list[UnresolvedReviewItem]]:
    """Reconcile carried items with this round's reviewer dispositions.

    ``round_number``, ``cleared_items_progress`` and ``sub_item_degradations``
    serve the sub-item contract (#958): the round stamps closures, each item
    that leaves the ledger with sub-items appends a ``ClearedItemProgress``,
    and deferrals or ignored sub-item input append a one-line explanation.

    ``evidence_response_head`` is set only for a re-review at a frozen head
    after new signed human input (#1068).  Only then may the requesting
    reviewer's ``resolved`` disposition clear its own evidence obligation;
    each clearance is appended to ``evidence_clearances`` as
    ``(obligation_identity, head)``.  Every other evidence disposition is a
    note, like any other machine record.
    """
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
        if any(d.sub_item_dispositions for d in dispositions) and _is_machine_obligation(item):
            # Reviewer sub-item input can never touch a machine obligation.
            if sub_item_degradations is not None:
                sub_item_degradations.append(
                    f"{item.item_id}: ignored sub-item dispositions for a machine obligation"
                )
        if _is_evidence_obligation(item):
            notes = list(item.notes)
            for disposition in dispositions:
                if disposition.note:
                    note = f"{disposition.reviewer}: {disposition.note}"
                    if note not in notes:
                        notes.append(note)
            preserved = replace(item, notes=tuple(notes)) if tuple(notes) != item.notes else item
            if (
                evidence_response_head is not None
                and item.lifecycle == "evidence_frozen"
                and item.candidate_head_sha == evidence_response_head
                and _evidence_clearance_authorized(
                    item, dispositions, configured_reviewers=configured_reviewers
                )
            ):
                if evidence_clearances is not None and item.obligation_identity:
                    evidence_clearances.append(
                        (item.obligation_identity, evidence_response_head)
                    )
                continue
            next_unresolved.append(preserved)
            continue
        if _is_machine_obligation(item):
            # Reviewers must still disposition machine records so the durable
            # review contract remains complete, but their prose is evidence
            # only.  In particular, a synthetic CI owner can never be cleared
            # by unanimous reviewer approval.
            notes = list(item.notes)
            for disposition in dispositions:
                if disposition.note:
                    note = f"{disposition.reviewer}: {disposition.note}"
                    if note not in notes:
                        notes.append(note)
            preserved = replace(item, notes=tuple(notes)) if tuple(notes) != item.notes else item
            if preserved.status in {"blocking", "same-pr"}:
                next_unresolved.append(preserved)
            elif preserved.status == "future" and retain_future:
                next_unresolved.append(preserved)
            elif preserved.status == "future":
                future_items.append(preserved)
            continue
        kept_item = _reconcile_item_with_sub_items(
            item,
            dispositions,
            same_status=same_status,
            reconciliation_mode=reconciliation_mode,
            round_number=round_number,
            cleared_items_progress=cleared_items_progress,
            sub_item_degradations=sub_item_degradations,
        )
        if kept_item is None:
            continue
        if kept_item.status == "future" and not retain_future:
            future_items.append(kept_item)
        else:
            next_unresolved.append(kept_item)
    return next_unresolved, future_items


@dataclass(frozen=True)
class ClearedItemProgress:
    """Confirmed record that a carried item with sub-items left the ledger (#958)."""

    item_id: str
    resolved: int
    total: int
    # ``all-sub-items-resolved`` or ``item-level-resolved``.
    cause: str
    round_number: int | None = None
    # Sub-items whose closure was recorded in this round.
    closed_sub_item_ids: tuple[str, ...] = ()


def _entry_completes_sub_items(
    entry: ReviewItemDisposition, open_ids: set[str], *, same_status: str
) -> bool:
    """A ``blocking``/same-status entry that resolves every open sub-item (#958)."""
    if entry.disposition not in {"blocking", same_status} or not open_ids:
        return False
    mapped = dict(entry.sub_item_dispositions)
    if any(verdict == "unresolved" for verdict in mapped.values()):
        return False
    return open_ids <= {key for key, verdict in mapped.items() if verdict == "resolved"}


def _sub_item_count(sub_items: Sequence[ReviewSubItem]) -> tuple[int, int]:
    return sum(1 for sub in sub_items if sub.status == "resolved"), len(sub_items)


def _completion_explanation(
    *,
    item: UnresolvedReviewItem,
    sub_item: ReviewSubItem,
    entry: ReviewItemDisposition,
    dispositions: Sequence[ReviewItemDisposition],
    same_status: str,
    owners: Sequence[str],
    owner_states: dict[str, str],
    reconciliation_mode: str,
    derived: set[int],
    reopeners: dict[str, list[str]],
) -> str:
    """Actionable reason a note-less completing entry did not take effect."""
    prefix = (
        f"completion of {item.item_id} did not take effect: sub-item "
        f'{sub_item.sub_item_id} ("{sub_item.text}") remains open because '
    )
    reopened_by = [name for name in reopeners.get(sub_item.sub_item_id, []) if name != entry.reviewer]
    if reopened_by:
        return prefix + f"{reopened_by[0]} reopened it"
    for index, other in enumerate(dispositions):
        if (
            other is not entry
            and index not in derived
            and other.reviewer != entry.reviewer
            and other.disposition in {"blocking", same_status}
        ):
            quoted = " ".join((other.note or "").split())
            detail = f': "{quoted}"' if quoted else " without an item-specific note"
            return prefix + f"{other.reviewer} kept item-level {other.disposition}{detail}"
    if reconciliation_mode == "owner-scoped":
        for owner in owners:
            if owner != entry.reviewer and owner_states.get(owner) != "cleared":
                return prefix + f"pending owner {owner} has not concurred"
    return prefix + "another reviewer has not concurred"


def _reconcile_item_with_sub_items(
    item: UnresolvedReviewItem,
    dispositions: Sequence[ReviewItemDisposition],
    *,
    same_status: str,
    reconciliation_mode: str,
    round_number: int | None,
    cleared_items_progress: list[ClearedItemProgress] | None,
    sub_item_degradations: list[str] | None,
) -> UnresolvedReviewItem | None:
    """Item-level reconciliation plus reviewer-authoritative sub-item status (#958).

    Today's item-level outcome logic runs unchanged on the effective
    dispositions.  Sub-item status changes only through eligible reviewers'
    ``sub_item_dispositions``; coder claims never reach this function.  An item
    that stays open never ends at K/K sub-items resolved: the completing
    closure is deferred and the item-level outcome is recomputed from the
    literal dispositions so no owner is cleared on its strength.
    """

    def note(text: str) -> None:
        if sub_item_degradations is not None:
            sub_item_degradations.append(f"{item.item_id}: {text}")

    if not item.sub_items:
        if any(d.sub_item_dispositions for d in dispositions):
            note("ignored sub-item dispositions for an item without sub-items")
        return _reconcile_reviewer_item(
            item, dispositions, same_status=same_status, reconciliation_mode=reconciliation_mode
        )

    owners = item.resolution_owners or (item.reviewer,)
    owner_states = dict(item.owner_states or ((owner, "pending") for owner in owners))
    stamp = round_number if round_number is not None else 0

    def eligible(entry: ReviewItemDisposition) -> bool:
        if reconciliation_mode != "owner-scoped":
            return True
        return entry.reviewer in owners or entry.disposition in {"blocking", same_status}

    known = {sub.sub_item_id for sub in item.sub_items}
    resolved_by: dict[str, list[str]] = {}
    reopeners: dict[str, list[str]] = {}
    for entry in dispositions:
        for key, verdict in entry.sub_item_dispositions:
            if key not in known:
                note(f"ignored unknown sub-item key `{key}` from {entry.reviewer}")
            elif not eligible(entry):
                note(
                    f"ignored {entry.reviewer}'s disposition of `{key}`: only owners may change "
                    "sub-item status"
                )
            elif verdict == "unresolved":
                reopeners.setdefault(key, []).append(entry.reviewer)
            else:
                resolved_by.setdefault(key, []).append(entry.reviewer)

    sub_status: dict[str, tuple[str, int | None]] = {}
    closures: list[str] = []
    for sub in item.sub_items:
        sid = sub.sub_item_id
        if reopeners.get(sid):
            sub_status[sid] = ("open", None)
        elif resolved_by.get(sid) and sub.status == "open":
            sub_status[sid] = ("resolved", stamp)
            closures.append(sid)
        else:
            sub_status[sid] = (sub.status, sub.resolved_round)

    def committed() -> tuple[ReviewSubItem, ...]:
        return tuple(
            replace(sub, status=sub_status[sub.sub_item_id][0], resolved_round=sub_status[sub.sub_item_id][1])
            for sub in item.sub_items
        )

    def complete() -> bool:
        return all(state == "resolved" for state, _round in sub_status.values())

    def is_open(result: UnresolvedReviewItem | None) -> bool:
        return result is not None and result.status in {"blocking", same_status}

    open_before = {sub.sub_item_id for sub in item.sub_items if sub.status == "open"}

    completing_indexes = {
        index
        for index, entry in enumerate(dispositions)
        if _entry_completes_sub_items(entry, open_before, same_status=same_status)
        and eligible(entry)
        and (reconciliation_mode != "owner-scoped" or entry.reviewer in owners)
    }
    derive_now = complete()
    derived = set(completing_indexes) if derive_now else set()

    def first_open() -> ReviewSubItem:
        return next(sub for sub in item.sub_items if sub_status[sub.sub_item_id][0] == "open")

    def effective(indexes_derived: set[int]) -> list[ReviewItemDisposition]:
        result: list[ReviewItemDisposition] = []
        for index, entry in enumerate(dispositions):
            if index in indexes_derived:
                result.append(replace(entry, disposition="resolved"))
            elif index in completing_indexes and not (entry.note or "").strip():
                # A note-less completing entry that stays in force gets an
                # actionable explanation instead of the bare status.
                result.append(
                    replace(
                        entry,
                        note=_completion_explanation(
                            item=item,
                            sub_item=first_open(),
                            entry=entry,
                            dispositions=dispositions,
                            same_status=same_status,
                            owners=owners,
                            owner_states=owner_states,
                            reconciliation_mode=reconciliation_mode,
                            derived=indexes_derived,
                            reopeners=reopeners,
                        ),
                    )
                )
            else:
                result.append(entry)
        return result

    # A completing entry that stays literal is explained even when the
    # candidate set is incomplete (reopened by another reviewer).
    result = _reconcile_reviewer_item(
        item,
        effective(derived),
        same_status=same_status,
        reconciliation_mode=reconciliation_mode,
    )
    if not is_open(result):
        # The item leaves the active ledger.
        resolved_count, total = _sub_item_count(committed())
        if complete():
            cause = "all-sub-items-resolved"
        else:
            cause = "item-level-resolved"
        if cleared_items_progress is not None:
            cleared_items_progress.append(
                ClearedItemProgress(
                    item_id=item.item_id,
                    resolved=resolved_count,
                    total=total,
                    cause=cause,
                    round_number=round_number,
                    closed_sub_item_ids=tuple(closures),
                )
            )
        return None if result is None else replace(result, sub_items=committed())

    if complete() and closures:
        # Post-outcome invariant: never persist an open item at K/K.
        deferred = closures[-1]
        sub_status[deferred] = ("open", None)
        closures.remove(deferred)
        note(f"deferred closing {deferred}: the item stays open, so its last sub-item stays open too")
        result = _reconcile_reviewer_item(
            item,
            effective(set()),
            same_status=same_status,
            reconciliation_mode=reconciliation_mode,
        )
    assert result is not None
    return replace(result, sub_items=committed())


def _reconcile_reviewer_item(
    item: UnresolvedReviewItem,
    dispositions: Sequence[ReviewItemDisposition],
    *,
    same_status: str,
    reconciliation_mode: str,
) -> UnresolvedReviewItem | None:
    """Item-level outcome for one non-machine item.

    Returns the item that stays in the ledger (status ``blocking``, the
    same-status value, unchanged while owners are pending, or ``future``), or
    ``None`` when every obligation is cleared.  Sub-item status is never
    touched here; ``item.sub_items`` rides through ``replace`` unchanged.
    """
    # `text` is the item’s canonical claim.  Dispositions may add evidence
    # for that claim, but must never rewrite it: a reviewer that discovers a
    # different concern must file a fresh item rather than silently changing
    # what this stable ID means.
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

    def kept(status: str) -> UnresolvedReviewItem:
        return replace(
            item,
            status=status,
            notes=tuple(notes),
            resolution_owners=owners,
            owner_states=tuple((owner, owner_states[owner]) for owner in owners),
            owner_evidence=tuple((owner, owner_evidence[owner]) for owner in owners if owner in owner_evidence),
            owner_dispositions=tuple((owner, owner_dispositions[owner]) for owner in owners if owner in owner_dispositions),
        )

    if "blocking" in outcomes:
        return kept("blocking")
    if same_status in outcomes:
        return kept(same_status)
    if reconciliation_mode == "owner-scoped":
        pending_owners = [owner for owner in owners if owner_states.get(owner) != "cleared"]
        if pending_owners:
            return kept(item.status)
    if "future" in outcomes or any(
        outcome == "future" for outcome in owner_dispositions.values()
    ):
        return kept("future")
    return None


def _evidence_clearance_authorized(
    item: UnresolvedReviewItem,
    dispositions: Sequence[ReviewItemDisposition],
    *,
    configured_reviewers: Sequence[str] | None,
) -> bool:
    """Only the requester may clear its evidence; a non-owner vote is a note.

    When the requester is no longer on the configured board, every configured
    reviewer must resolve the request, so no single reviewer can waive it.
    """
    owner = item.reviewer
    resolved_by = {
        disposition.reviewer
        for disposition in dispositions
        if disposition.disposition == "resolved"
    }
    if configured_reviewers is not None and owner not in set(configured_reviewers):
        return bool(configured_reviewers) and set(configured_reviewers) <= resolved_by
    return owner in resolved_by


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


# The compact prior ledger rides in every round comment's metadata, so its
# size must not scale with round count (#1003).  The budget is measured as the
# UTF-8 byte length of each JSON-serialized entry, which is exactly what the
# round transport compresses and base64-encodes.  Base64 adds a third and zlib
# cannot expand incompressible input by more than a few bytes, so even a
# worst-case (non-ASCII, high-entropy) ledger encodes to roughly 21,500
# characters, well inside the 60,000-character comment limit.
COMPACT_PRIOR_SUMMARIES_MAX_BYTES = 16_000
COMPACT_PRIOR_DETAILS_OMITTED_SUFFIX = " (details compacted)"
COMPACT_PRIOR_OMITTED_NOTICE_RE = re.compile(
    r"^\[compacted\] (?P<count>\d+) earlier prior item summar(?:y|ies) omitted"
)


def _compact_prior_entry_size(entry: str) -> int:
    return len(json.dumps(entry, ensure_ascii=False).encode("utf-8"))


def _compact_prior_summaries_size(summaries: Sequence[str]) -> int:
    # Serialized UTF-8 bytes plus one JSON list separator between entries.
    return sum(_compact_prior_entry_size(summary) for summary in summaries) + max(
        len(summaries) - 1, 0
    )


def _compact_prior_omitted_notice(count: int) -> str:
    noun = "summary" if count == 1 else "summaries"
    return (
        f"[compacted] {count} earlier prior item {noun} omitted to bound "
        "round metadata; those items were already dispositioned in earlier rounds."
    )


def bound_compact_prior_summaries(
    summaries: Sequence[str],
    *,
    max_bytes: int = COMPACT_PRIOR_SUMMARIES_MAX_BYTES,
) -> tuple[str, ...]:
    """Bound the append-only compact prior ledger independent of round count.

    Oldest entries degrade first: their bodies collapse to the header line
    (item id, disposition label, reviewer, source round), then whole headers
    fold into a single omission notice.  The newest entries stay verbatim.
    The result is idempotent, so re-bounding a persisted ledger is stable.
    """
    entries = list(summaries)
    omitted = 0
    if entries:
        match = COMPACT_PRIOR_OMITTED_NOTICE_RE.match(entries[0])
        if match:
            omitted = int(match.group("count"))
            entries = entries[1:]

    rendered = [_compact_prior_omitted_notice(omitted), *entries] if omitted else entries
    if _compact_prior_summaries_size(rendered) <= max_bytes:
        return tuple(rendered)
    # Fill the budget newest-first.  Once one entry must degrade to its header,
    # every older entry degrades too; once a header no longer fits, every older
    # entry folds into the omission notice.
    notice_reserve = _compact_prior_entry_size(
        _compact_prior_omitted_notice(omitted + len(entries))
    ) + 1
    budget = max_bytes - notice_reserve
    kept: list[str] = []
    used = 0
    headers_only = False
    for position in range(len(entries) - 1, -1, -1):
        entry = entries[position]
        separator = 1 if kept else 0
        entry_size = _compact_prior_entry_size(entry)
        if not headers_only and used + separator + entry_size <= budget:
            kept.append(entry)
            used += separator + entry_size
            continue
        headers_only = True
        header = entry.split("\n", 1)[0]
        if not header.endswith(COMPACT_PRIOR_DETAILS_OMITTED_SUFFIX):
            header += COMPACT_PRIOR_DETAILS_OMITTED_SUFFIX
        header_size = _compact_prior_entry_size(header)
        if used + separator + header_size > budget:
            omitted += position + 1
            break
        kept.append(header)
        used += separator + header_size
    kept.reverse()
    if omitted:
        kept.insert(0, _compact_prior_omitted_notice(omitted))
    return tuple(kept)


def _validate_plan_review_response(
    text: str,
    *,
    reviewer: str,
    unresolved_items: Sequence[UnresolvedReviewItem],
    current_round_items: Sequence[UnresolvedReviewItem] = (),
    surfaced_requirement_ids: Sequence[str] = (),
    architecture_status_mode: str,
) -> ParsedPlanReview:
    parsed = parse_plan_review(
        text, reviewer=reviewer, architecture_status_mode=architecture_status_mode
    )
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
    # Replace only the dispositions: the assessment, its degradation records
    # and every other parsed field survive a carried-item round (#925).
    return replace(parsed, dispositions=dispositions)


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


def format_sub_item_lines(item: UnresolvedReviewItem) -> list[str]:
    """Per-sub-item lines with a ``k/K resolved`` count; empty without sub-items."""
    if not item.sub_items:
        return []
    resolved, total = _sub_item_count(item.sub_items)
    lines = [f"Sub-items ({resolved}/{total} resolved):"]
    for sub in item.sub_items:
        suffix = f" (round {sub.resolved_round})" if sub.resolved_round is not None else ""
        lines.append(f"- [{sub.sub_item_id}] {sub.status}{suffix}: {sub.text}")
    return lines


def _format_same_pr_unresolved_items(items: Sequence[UnresolvedReviewItem]) -> str:
    lines: list[str] = []
    for item in select_coder_followup_items(items):
        lines.append(
            f"{item.reviewer} same-PR follow-up [{item.item_id}] from round {item.source_round}:"
        )
        lines.append(f"- {item.text}")
        lines.extend(format_sub_item_lines(item))
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
    for item in select_coder_followup_items(items):
        lines.append(
            f"{item.reviewer} unresolved {item.status} item [{item.item_id}] from round {item.source_round}:"
        )
        if _is_machine_obligation(item):
            lines.append(
                f"Machine authority: {item.obligation_kind or 'unknown'}; lifecycle="
                f"{item.lifecycle or 'unknown'}; failed head={item.failed_head_sha or '(unknown)'}; "
                f"candidate head={item.candidate_head_sha or '(none)'}."
            )
            lines.append(
                "Reviewer dispositions cannot clear this obligation; fix the failed head and "
                "wait for authoritative source-specific validation."
            )
        lines.append(f"- {item.text}")
        lines.extend(format_sub_item_lines(item))
        if item.notes:
            lines.append("Latest reviewer updates:")
            lines.extend(f"- {note}" for note in item.notes)
        lines.append("")
    return "\n".join(lines).strip()


def _format_human_requirements_ack_for_coder(
    unresolved_items: Sequence[UnresolvedReviewItem],
) -> str:
    """Render the acknowledgement obligation outside reviewer-item fields."""
    item = next(
        (item for item in unresolved_items if item.item_id == HUMAN_REQUIREMENTS_ACK_ITEM_ID),
        None,
    )
    if item is None:
        return ""
    lines = [
        "Human-requirements acknowledgement (dedicated non-classifiable record)",
        f"Internal record ID: `{item.item_id}`. This is not a reviewer item.",
        "Acknowledge or disposition the signed human requirements using only the "
        "`human_requirements` and `human_requirement_dispositions` JSON fields. "
        "Do not put this internal record ID in `addressed_items`, `remaining_items`, "
        "or `disputed_items`.",
        "Stored acknowledgement diagnostic:",
        f"- {item.text}",
    ]
    return "\n".join(lines)


def format_coder_followup_context(
    unresolved_items: Sequence[UnresolvedReviewItem],
) -> str:
    """Render classifiable coder work plus dedicated synthetic obligations."""
    sections = [
        _format_unresolved_items_for_coder(unresolved_items),
        _format_human_requirements_ack_for_coder(unresolved_items),
    ]
    return "\n\n".join(section for section in sections if section)


# --- Sub-item progress signal (#958) ------------------------------------

SUB_ITEM_PROGRESS_CLASSES = ("new", "converging", "stalled", "complete", "cleared")


@dataclass(frozen=True)
class SubItemProgress:
    item_id: str
    resolved: int
    total: int
    # Sub-items closed within the stall window.
    closed_in_window: int
    classification: str
    window: int
    # First round in which the item satisfied the stall rule, else None.
    stalled_since_round: int | None = None


def _closure_rounds(item: UnresolvedReviewItem) -> list[int]:
    return [sub.resolved_round for sub in item.sub_items if sub.resolved_round is not None]


def _classify_sub_item_progress(
    item: UnresolvedReviewItem, *, current_round: int, window: int
) -> SubItemProgress:
    resolved, total = _sub_item_count(item.sub_items)
    closures = _closure_rounds(item)
    if window > 0:
        closed = sum(1 for closed_round in closures if closed_round > current_round - window)
    else:
        # Stall check disabled: report overall closure, never `stalled`.
        closed = len(closures)
    if resolved == total:
        classification = "complete"
    elif closed:
        classification = "converging"
    elif (
        window > 0
        # A finding minted this very round has not been re-reviewed yet.
        and current_round > item.source_round
        and current_round - item.source_round + 1 >= window
    ):
        classification = "stalled"
    else:
        classification = "new"
    stalled_since: int | None = None
    if classification == "stalled":
        # The stall rule first held when ``window`` rounds had passed since
        # the later of the source round and the latest closure.
        anchor = max([item.source_round - 1, *closures]) if closures else item.source_round - 1
        stalled_since = max(anchor + window, item.source_round + 1)
    return SubItemProgress(
        item_id=item.item_id,
        resolved=resolved,
        total=total,
        closed_in_window=closed,
        classification=classification,
        window=window,
        stalled_since_round=stalled_since,
    )


def sub_item_progress(
    items: Sequence[UnresolvedReviewItem],
    cleared_items_progress: Sequence[ClearedItemProgress] = (),
    *,
    current_round: int,
    window: int,
) -> tuple[SubItemProgress, ...]:
    """Classify every carried or just-cleared item that has sub-items.

    ``window`` is ``--sub-item-stall-rounds``; 0 disables the stall check, so
    no item is ever ``stalled``.  A pure function of persisted state: the same
    ledger and cleared records always give the same signal, which is what
    keeps the published notices idempotent across resume.
    """
    progress: list[SubItemProgress] = []
    for item in items:
        if item.sub_items and not item.is_machine_obligation:
            progress.append(
                _classify_sub_item_progress(item, current_round=current_round, window=window)
            )
    for cleared in cleared_items_progress:
        progress.append(
            SubItemProgress(
                item_id=cleared.item_id,
                resolved=cleared.resolved,
                total=cleared.total,
                closed_in_window=len(cleared.closed_sub_item_ids),
                classification="cleared",
                window=window,
            )
        )
    return tuple(progress)


def newly_stalled_items(
    items: Sequence[UnresolvedReviewItem], *, current_round: int, window: int
) -> tuple[SubItemProgress, ...]:
    """Items whose stall rule first held in ``current_round``.

    Derived only from persisted ``resolved_round``/``source_round`` values, so
    an interrupted round recomputes the same notice and later rounds do not
    repeat it.
    """
    if window <= 0:
        return ()
    return tuple(
        entry
        for entry in sub_item_progress(items, (), current_round=current_round, window=window)
        if entry.classification == "stalled" and entry.stalled_since_round == current_round
    )


def render_sub_item_progress_summary(progress: Sequence[SubItemProgress]) -> tuple[str, ...]:
    """One line per carried item with sub-items, for the budget-exit message."""
    lines: list[str] = []
    for entry in progress:
        if entry.classification == "cleared":
            continue
        line = f"{entry.item_id}: {entry.resolved}/{entry.total} sub-items resolved"
        if entry.classification == "complete":
            line += "; all sub-items resolved; awaiting item clearance"
        elif entry.classification == "stalled":
            line += f", none closed in the last {entry.window} rounds (stalled)"
        elif entry.classification == "converging":
            line += (
                f", {entry.closed_in_window} closed in last {entry.window} rounds "
                "(converging); consider raising --max-rounds"
            )
        else:
            line += " (new)"
        lines.append(line)
    return tuple(lines)
