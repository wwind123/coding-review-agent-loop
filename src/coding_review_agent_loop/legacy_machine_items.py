"""Compatibility normalization for historical machine-obligation ledgers."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Sequence

from .protocol import MACHINE_AUTHORITY, UNKNOWN_MACHINE_AUTHORITY, UnresolvedReviewItem

from .unresolved_items import (
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    HUMAN_REQUIREMENTS_OBLIGATION_KIND,
    _upsert_human_requirements_ack_item,
)

if TYPE_CHECKING:
    from .round_state import PostedRoundRecord

_LEGACY_MACHINE_REVIEWERS = {
    "GitHub managed exact-head CI": "managed-exact-head-ci",
    "GitHub PR checks": "github-pr-checks",
    "Alembic migration validation": "alembic-migration",
}


_LEGACY_UNTRUSTED_LINEAGE_NOTE = "Synthetic machine record lacked trusted orchestrator lineage."
_LEGACY_UNBOUND_HEAD_NOTE = "Known machine item could not be bound to a failed head."


def _legacy_machine_kind(item: UnresolvedReviewItem) -> str | None:
    if item.item_id == "item-merge-conflict":
        return "merge-conflict"
    if item.item_id == "item-human-requirements-acknowledgement":
        return "human-requirements-acknowledgement"
    return _LEGACY_MACHINE_REVIEWERS.get(item.reviewer) or (
        "unknown" if item.reviewer == "Orchestrator" else None
    )


def _legacy_machine_promotion(
    item: UnresolvedReviewItem,
    *,
    record: PostedRoundRecord,
    first_record: PostedRoundRecord | None,
) -> UnresolvedReviewItem:
    """Promote only orchestrator-lineage synthetic items from old ledgers.

    A display name in reviewer prose is not authority.  The first durable
    ``new_items`` record (or a coder checkpoint carrying a legacy machine item)
    must be an orchestrator-produced record before a known kind is promoted.
    Ambiguous synthetic records become explicit unknown blockers.
    """
    if item.is_machine_obligation:
        return item
    kind = _legacy_machine_kind(item)
    if kind is None:
        return item
    lineage = first_record or record
    trusted = (
        lineage.metadata.role in {"summary", "coder"}
        and (
            lineage.metadata.agent == "Orchestrator"
            or lineage.metadata.role == "coder"
        )
        and item.status in {"blocking", "same-pr"}
        and lineage.metadata.round_number >= item.source_round
    )
    if not trusted:
        return replace(
            item,
            authority=UNKNOWN_MACHINE_AUTHORITY,
            obligation_kind="unknown",
            lifecycle="repair_required",
            failed_head_sha=None,
            candidate_head_sha=None,
            obligation_identity=f"unknown:{item.item_id}",
            notes=(*item.notes, _LEGACY_UNTRUSTED_LINEAGE_NOTE),
            resolution_owners=(),
            owner_states=(),
        )
    failed_head: str | None = None
    if lineage.metadata.role == "summary" and lineage.metadata.agent == "Orchestrator":
        # An orchestrator summary's subject is the authoritative PR head for
        # the recorded machine event.
        failed_head = lineage.metadata.subject
    elif lineage.metadata.role == "coder":
        # A legacy CI failure was often first carried by the coder checkpoint,
        # after the coder had already pushed a repair head. The scheduler
        # previous/current pair is the producer-owned provenance that lets us
        # recover the failed head; visible CI prose is not authority.
        previous_head = lineage.metadata.scheduler_previous_sha
        current_head = lineage.metadata.scheduler_current_sha
        if (
            previous_head
            and current_head
            and lineage.metadata.subject == current_head
            and previous_head != current_head
        ):
            failed_head = previous_head
    if not failed_head or failed_head == "unknown":
        return replace(
            item,
            authority=UNKNOWN_MACHINE_AUTHORITY,
            obligation_kind="unknown",
            lifecycle="repair_required",
            obligation_identity=f"unknown:{item.item_id}",
            notes=(*item.notes, _LEGACY_UNBOUND_HEAD_NOTE),
            resolution_owners=(),
            owner_states=(),
        )
    return replace(
        item,
        authority=MACHINE_AUTHORITY,
        obligation_kind=kind,
        lifecycle="repair_required",
        failed_head_sha=failed_head,
        candidate_head_sha=None,
        obligation_identity=f"{kind}:{item.item_id}",
        resolution_owners=(),
        owner_states=(),
    )


def _promote_legacy_machine_items(records: Sequence[PostedRoundRecord]) -> tuple[PostedRoundRecord, ...]:
    first_new_item_record: dict[str, PostedRoundRecord] = {}
    for record in records:
        for item in record.metadata.new_items:
            first_new_item_record.setdefault(item.item_id, record)
    promoted: list[PostedRoundRecord] = []
    for record in records:
        metadata = record.metadata
        prior = tuple(
            _legacy_machine_promotion(
                item,
                record=record,
                first_record=first_new_item_record.get(item.item_id),
            )
            for item in metadata.prior_items
        )
        new_items = tuple(
            _legacy_machine_promotion(
                item,
                record=record,
                first_record=first_new_item_record.get(item.item_id, record),
            )
            for item in metadata.new_items
        )
        if prior != metadata.prior_items or new_items != metadata.new_items:
            metadata = replace(metadata, prior_items=prior, new_items=new_items)
        promoted.append(replace(record, metadata=metadata))
    return tuple(promoted)


PLAN_DEMOTED_MACHINE_ITEM_NOTE = (
    "Planning has no machine-obligation clearance path; this orchestrator-authored "
    "plan item was restored to an ordinary reviewer-dispositionable finding (#1005)."
)


_REVIEWER_DISPOSITION_NOTE_RE = re.compile(r"[A-Za-z][A-Za-z0-9 ._-]*: \S")


def _has_legacy_promotion_note_tail(notes: Sequence[str]) -> bool:
    """True when the promotion's diagnostic note is followed only by reviewer notes.

    ``_apply_unresolved_item_dispositions`` appends ``"<reviewer>: <note>"``
    evidence to a machine record each round it is dispositioned, so a promoted
    item carried through approval rounds holds those notes after the diagnostic.
    """
    diagnostics = {_LEGACY_UNTRUSTED_LINEAGE_NOTE, _LEGACY_UNBOUND_HEAD_NOTE}
    positions = [index for index, note in enumerate(notes) if note in diagnostics]
    if len(positions) != 1:
        return False
    return all(
        _REVIEWER_DISPOSITION_NOTE_RE.match(note) for note in notes[positions[0] + 1:]
    )


def _is_legacy_promoted_plan_item(item: UnresolvedReviewItem) -> bool:
    """Recognize exactly what ``_legacy_machine_promotion`` made of a plan item."""
    return (
        item.is_machine_obligation
        and item.reviewer == "Orchestrator"
        and item.item_id not in {"item-merge-conflict", "item-human-requirements-acknowledgement"}
        and item.authority in {MACHINE_AUTHORITY, UNKNOWN_MACHINE_AUTHORITY}
        and item.obligation_kind == "unknown"
        and item.obligation_identity == f"unknown:{item.item_id}"
        and item.lifecycle == "repair_required"
        and item.candidate_head_sha is None
        # Only the authority/head combinations the promotion can emit: machine
        # authority bound to a failed head, or unknown authority with no head
        # plus the diagnostic note the promotion appended for that outcome.
        and (
            (item.authority == MACHINE_AUTHORITY and bool(item.failed_head_sha))
            or (
                item.authority == UNKNOWN_MACHINE_AUTHORITY
                and not item.failed_head_sha
                and _has_legacy_promotion_note_tail(item.notes)
            )
        )
        # The legacy promotion always clears ownership; a machine record that
        # still names owners is not identifiable as that promotion.
        and not item.resolution_owners
        and not item.owner_states
        and item.status in {"blocking", "same-pr"}
        and not any(note.startswith("Invalid persisted machine") for note in item.notes)
    )


def _demote_plan_machine_item(item: UnresolvedReviewItem) -> UnresolvedReviewItem:
    """Return a plan-flow machine record to the ordinary finding representation.

    The planning loop never mints machine obligations and has no path that
    clears one: reviewer dispositions are evidence only for machine records,
    and the human-requirements/merge-conflict/CI clearance paths are PR-only.
    A machine-authority item in a plan ledger (historically the legacy
    promotion of the orchestrator's human-requirements re-injection item)
    would therefore stay blocking forever and force a revision after every
    unanimous approval (#1005).  Restoring the legacy representation lets the
    reviewers disposition it like every other plan finding; the
    human-requirements gate still re-checks acknowledgements each round and
    re-injects a fresh item if they remain missing.

    Only the exact legacy-promotion shape is restored.  A malformed or
    otherwise unrecognized machine record (for example the ``unknown`` blocker
    the decoder mints for an invalid persisted field) stays a machine
    obligation, so the planning loop stops on it with a diagnostic instead of
    letting a reviewer approval clear corrupted state.
    """
    if not _is_legacy_promoted_plan_item(item):
        return item
    notes = item.notes
    if PLAN_DEMOTED_MACHINE_ITEM_NOTE not in notes:
        notes = (*notes, PLAN_DEMOTED_MACHINE_ITEM_NOTE)
    return replace(
        item,
        authority=None,
        obligation_kind=None,
        lifecycle=None,
        failed_head_sha=None,
        candidate_head_sha=None,
        obligation_identity=None,
        notes=notes,
        resolution_owners=(),
        owner_states=(),
    )


def _demote_plan_machine_items(records: Sequence[PostedRoundRecord]) -> tuple[PostedRoundRecord, ...]:
    demoted: list[PostedRoundRecord] = []
    for record in records:
        metadata = record.metadata
        prior = tuple(_demote_plan_machine_item(item) for item in metadata.prior_items)
        new_items = tuple(_demote_plan_machine_item(item) for item in metadata.new_items)
        if prior != metadata.prior_items or new_items != metadata.new_items:
            metadata = replace(metadata, prior_items=prior, new_items=new_items)
            record = replace(record, metadata=metadata)
        demoted.append(record)
    return tuple(demoted)


# The complete historical PR producer template, deliberately distinct from the
# shorter planning message. Names are a nonempty comma-separated display list.
_ACK_TEXT_RE = re.compile(
    r"Reviewer\(s\) [^,\s][^,\n]*(?:, [^,\s][^,\n]*)* approved without "
    r"acknowledging the signed human requirements\. Coder must address the "
    r"human requirements and ensure the reviewer explicitly resolves them "
    r"before approval\."
)


def _ack_candidate_shape(item: UnresolvedReviewItem) -> bool:
    if not (
        re.fullmatch(r"item-[1-9][0-9]*", item.item_id)
        and item.reviewer == "Orchestrator"
        and item.status == "blocking"
        and item.source_round > 0
        and item.source_status in {None, "blocking"}
        and _ACK_TEXT_RE.fullmatch(item.text)
        and not item.fix_scope
        and not item.owner_evidence and not item.owner_dispositions
        and not item.sub_items and not item.evidence_row_ids
        and not item.converted_row_id and not item.provenance and not item.repeat_of
    ):
        return False
    if not item.is_machine_obligation:
        return (
            item.resolution_owners in {(), ("Orchestrator",)}
            and item.owner_states in {(), (("Orchestrator", "pending"),)}
            and all(_REVIEWER_DISPOSITION_NOTE_RE.match(note) for note in item.notes)
        )
    # Reuse the exact persisted promotion-shape guard. Despite its historical
    # name it describes the legacy promotion output, not plan message text.
    return (
        _is_legacy_promoted_plan_item(item)
        and item.failed_head_sha != "unknown"
        and (item.authority != MACHINE_AUTHORITY or not any(
            note in {_LEGACY_UNTRUSTED_LINEAGE_NOTE, _LEGACY_UNBOUND_HEAD_NOTE}
            for note in item.notes
        ))
    ) and all(
        note in {_LEGACY_UNTRUSTED_LINEAGE_NOTE, _LEGACY_UNBOUND_HEAD_NOTE}
        or _REVIEWER_DISPOSITION_NOTE_RE.match(note)
        for note in item.notes
    )


def _canonical_ack(item: UnresolvedReviewItem) -> bool:
    return (
        item.item_id == HUMAN_REQUIREMENTS_ACK_ITEM_ID
        and item.reviewer == "Orchestrator" and item.status == "blocking"
        and item.resolution_owners == ("Orchestrator",)
        and item.owner_states == (("Orchestrator", "pending"),)
        and not item.owner_evidence and not item.owner_dispositions
        and item.authority == MACHINE_AUTHORITY
        and item.obligation_kind == HUMAN_REQUIREMENTS_OBLIGATION_KIND
        and item.obligation_identity == HUMAN_REQUIREMENTS_OBLIGATION_KIND
        and item.lifecycle == "repair_required"
        and item.failed_head_sha is None and item.candidate_head_sha is None
    )


def _normalize_ack_ledger(items, aliases):
    result = []
    for item in items:
        if item.item_id in aliases:
            canonical = _upsert_human_requirements_ack_item(
                (), source_round=item.source_round, text=item.text
            )[0]
            item = replace(canonical, notes=item.notes)
        if _canonical_ack(item):
            existing = next((entry for entry in result if _canonical_ack(entry)), None)
            if existing is not None:
                result[result.index(existing)] = replace(
                    existing, source_round=min(existing.source_round, item.source_round),
                    notes=tuple(dict.fromkeys((*existing.notes, *item.notes))),
                )
                continue
        result.append(item)
    return tuple(result)


def _normalize_ack_dispositions(dispositions, prior_aliases):
    """Rewrite only aliases proven in this record's own input ledger."""
    result = []
    for disposition in dispositions:
        if disposition.item_id in prior_aliases:
            disposition = replace(disposition, item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID)
        if disposition.item_id == HUMAN_REQUIREMENTS_ACK_ITEM_ID and prior_aliases:
            existing = next((entry for entry in result
                             if (entry.item_id, entry.reviewer) == (disposition.item_id, disposition.reviewer)), None)
            if existing is not None:
                # ReviewItemDisposition has one note field. Keep both notes in
                # stable history order and retain the stronger unresolved state.
                rank = {"resolved": 0, "future": 1, "same-pr": 2, "blocking": 3}
                chosen = max((existing, disposition), key=lambda entry: rank.get(entry.disposition, 4))
                result[result.index(existing)] = replace(
                    chosen, note="\n".join(dict.fromkeys(note for note in (existing.note, disposition.note) if note)) or None
                )
                continue
        result.append(disposition)
    return tuple(result)


def recover_pr_acknowledgement_aliases(records: Sequence[PostedRoundRecord]) -> tuple[PostedRoundRecord, ...]:
    """Recover only exact PR aliases with earliest durable producer lineage.

    The entire admitted history matters: a later coder carry cannot launder a
    reviewer origin, and a later reviewer copy cannot revoke a trusted origin.
    All copies must agree on identity and match a legitimate historical shape.
    History is transformed in place in memory, never appended into a newer
    ledger, so a later cleared checkpoint remains cleared on every restart.
    """
    # Authentication-specific carriers are admitted later by their owning
    # reader. They cannot establish stage-1 legacy producer authority here.
    from .round_state import RECOVERY_RECORD_PHASES, record_carries_evidence_authority

    origins = {}
    copies = {}
    for record in records:
        if record.metadata.flow != "pr":
            continue
        for item in (*record.metadata.prior_items, *record.metadata.new_items):
            origins.setdefault(item.item_id, (record, item))
            copies.setdefault(item.item_id, []).append(item)
    aliases = set()
    for item_id, (record, origin) in origins.items():
        metadata = record.metadata
        if not (
            metadata.role in {"coder", "summary"}
            and metadata.phase not in RECOVERY_RECORD_PHASES
            and not record_carries_evidence_authority(metadata)
            and (metadata.role == "coder" or metadata.agent == "Orchestrator")
            and metadata.round_number >= origin.source_round
            and _ack_candidate_shape(origin)
        ):
            continue
        if all(
            _ack_candidate_shape(item)
            and (item.text, item.source_round) == (origin.text, origin.source_round)
            for item in copies[item_id]
        ):
            aliases.add(item_id)
    normalized = []
    for record in records:
        if record.metadata.flow != "pr":
            normalized.append(record)
            continue
        metadata = record.metadata
        high_water = record.original_item_number_high_water
        for item in (*metadata.prior_items, *metadata.new_items):
            match = re.fullmatch(r"item-(\d+)", item.item_id)
            if match:
                high_water = max(high_water, int(match[1]))
        prior_aliases = aliases.intersection(item.item_id for item in metadata.prior_items)
        normalized.append(replace(
            record, original_item_number_high_water=high_water,
            metadata=replace(
                metadata,
                prior_items=_normalize_ack_ledger(metadata.prior_items, aliases),
                new_items=_normalize_ack_ledger(metadata.new_items, aliases),
                dispositions=_normalize_ack_dispositions(metadata.dispositions, prior_aliases),
            ),
        ))
    return tuple(normalized)


def consumed_acknowledgement_disposition(disposition, resumed_record, live_prior_items) -> bool:
    """A resumed review may reference an ack cleared by round-start reconcile.

    Do not generalize unknown-ID suppression: require that exact canonical
    machine target in the completed review's own normalized prior ledger.
    """
    return (
        resumed_record is not None
        and disposition.item_id == HUMAN_REQUIREMENTS_ACK_ITEM_ID
        and not any(item.item_id == disposition.item_id for item in live_prior_items)
        and any(_canonical_ack(item) for item in resumed_record.metadata.prior_items)
    )
