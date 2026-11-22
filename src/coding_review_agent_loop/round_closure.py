"""The closed-incomplete round record and the one closure-boundary predicate (#2064).

A tool-owned Orchestrator summary with phase ``round-closed-incomplete``
closes a PR or plan review round in which some reviews were published but
some reviewer's outcome cannot be replayed.  The published reviews stay on
GitHub as history; every reader treats the round's reviewer records as inert:
they are never resumed, never visible peers, never panel evidence and never
approvals or approval carry.

The record carries the round's ledger.  ``prior_items`` and ``new_items`` use
the ordinary round-metadata fields (the existing item codecs and high-water
readers apply unchanged); the ``round_closure`` payload holds the audit data
and the materialized exact-head evidence obligations:

* ``flow``, ``round`` and ``subject`` bind the payload to its record;
* ``published``: reviewers with a public round-N review and its comment ID;
* ``unreplayable``: reviewers whose outcome could not be replayed, with cause;
* ``evidence_obligations``: published exact-head evidence requests with their
  reviewer, source round, head, text and reserved item number, because the
  inert reviewer records are no longer read by the completed-review loop;
* ``item_number_high_water``: monotonic item-number high-water mark.

The payload decodes strictly; a malformed or mis-bound payload decodes as
``None`` and the record then never closes a round.  Author authentication is
unchanged: only records admitted by ``pr_round_ingestion`` are read at all.

Stage 1 of #2064 adds only these readers; no code path writes the record yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .signed_input_record import is_supersession_boundary

ROUND_CLOSED_INCOMPLETE_PHASE = "round-closed-incomplete"
CLOSURE_FLOWS = frozenset({"pr", "plan"})
_MAX_TEXT = 4000


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _text(value: object, *, limit: int = _MAX_TEXT) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _decode_published(raw: object) -> list[dict[str, object]] | None:
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    entries: list[dict[str, object]] = []
    for entry in raw:
        if not isinstance(entry, Mapping) or set(entry) != {"agent", "comment_id"}:
            return None
        agent, comment_id = entry["agent"], entry["comment_id"]
        if not _text(agent, limit=200) or (comment_id is not None and not _positive_int(comment_id)):
            return None
        entries.append({"agent": agent, "comment_id": comment_id})
    if len({entry["agent"] for entry in entries}) != len(entries):
        return None
    return sorted(entries, key=lambda entry: str(entry["agent"]))


def _decode_unreplayable(raw: object) -> list[dict[str, object]] | None:
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    entries: list[dict[str, object]] = []
    for entry in raw:
        if not isinstance(entry, Mapping) or set(entry) != {"agent", "cause"}:
            return None
        if not _text(entry["agent"], limit=200) or not _text(entry["cause"]):
            return None
        entries.append({"agent": entry["agent"], "cause": entry["cause"]})
    if len({entry["agent"] for entry in entries}) != len(entries):
        return None
    return sorted(entries, key=lambda entry: str(entry["agent"]))


_OBLIGATION_KEYS = frozenset({"reviewer", "source_round", "head", "text", "item_number"})


def _decode_obligations(raw: object, *, round_number: int) -> list[dict[str, object]] | None:
    if not isinstance(raw, (list, tuple)):
        return None
    entries: list[dict[str, object]] = []
    for entry in raw:
        if not isinstance(entry, Mapping) or set(entry) != _OBLIGATION_KEYS:
            return None
        if (
            not _text(entry["reviewer"], limit=200)
            or not _positive_int(entry["source_round"])
            or entry["source_round"] > round_number
            or not _text(entry["head"], limit=200)
            or not _text(entry["text"])
            or not _positive_int(entry["item_number"])
        ):
            return None
        entries.append({key: entry[key] for key in sorted(_OBLIGATION_KEYS)})
    if len({entry["item_number"] for entry in entries}) != len(entries):
        return None
    return sorted(entries, key=lambda entry: int(entry["item_number"]))  # type: ignore[arg-type]


def decode_round_closure_payload(
    raw: object, *, flow: object, round_number: object, subject: object
) -> dict[str, object] | None:
    """Return the canonical payload, or ``None`` when it is malformed or mis-bound."""
    if not isinstance(raw, Mapping):
        return None
    expected = {"flow", "round", "subject", "published", "unreplayable",
                "evidence_obligations", "item_number_high_water"}
    if set(raw) != expected:
        return None
    if flow not in CLOSURE_FLOWS or raw["flow"] != flow:
        return None
    if not _positive_int(round_number) or raw["round"] != round_number:
        return None
    if not isinstance(subject, str) or not subject or raw["subject"] != subject:
        return None
    published = _decode_published(raw["published"])
    unreplayable = _decode_unreplayable(raw["unreplayable"])
    obligations = _decode_obligations(raw["evidence_obligations"], round_number=round_number)  # type: ignore[arg-type]
    if published is None or unreplayable is None or obligations is None:
        return None
    high_water = raw["item_number_high_water"]
    if not isinstance(high_water, int) or isinstance(high_water, bool) or high_water < 0:
        return None
    # Reserved obligation numbers lie at or below the recorded high-water mark.
    if any(int(entry["item_number"]) > high_water for entry in obligations):  # type: ignore[arg-type]
        return None
    return {
        "flow": flow,
        "round": round_number,
        "subject": subject,
        "published": published,
        "unreplayable": unreplayable,
        "evidence_obligations": obligations,
        "item_number_high_water": high_water,
    }


def build_round_closure_payload(
    *,
    flow: str,
    round_number: int,
    subject: str,
    published: Sequence[tuple[str, int | None]],
    unreplayable: Sequence[tuple[str, str]],
    evidence_obligations: Sequence[Mapping[str, object]] = (),
    item_number_high_water: int,
) -> dict[str, object] | None:
    """Encode a payload through the strict decoder; ``None`` when it would not decode."""
    return decode_round_closure_payload(
        {
            "flow": flow,
            "round": round_number,
            "subject": subject,
            "published": [{"agent": a, "comment_id": c} for a, c in published],
            "unreplayable": [{"agent": a, "cause": c} for a, c in unreplayable],
            "evidence_obligations": [dict(entry) for entry in evidence_obligations],
            "item_number_high_water": item_number_high_water,
        },
        flow=flow, round_number=round_number, subject=subject,
    )


def is_round_closed_incomplete(metadata: object) -> bool:
    """A decoded, tool-owned closed-incomplete record (PR or plan flow)."""
    payload = getattr(metadata, "round_closure", None)
    return (
        getattr(metadata, "flow", None) in CLOSURE_FLOWS
        and getattr(metadata, "role", None) == "summary"
        and getattr(metadata, "agent", None) == "Orchestrator"
        and getattr(metadata, "phase", None) == ROUND_CLOSED_INCOMPLETE_PHASE
        and isinstance(payload, Mapping)
        and payload.get("flow") == getattr(metadata, "flow", None)
    )


def is_round_closure_boundary(metadata: object) -> bool:
    """The one closure-boundary predicate: the new record, or a legacy PR one.

    The legacy signed-input supersession record (#2054) keeps closing PR rounds
    for one release window after #2064; no code path writes it any more.
    TODO(#2064): drop the legacy branch after that window.
    """
    return is_round_closed_incomplete(metadata) or is_supersession_boundary(metadata)


def closed_round_reviewer_indexes(records: Sequence[object], *, flow: str) -> frozenset[int]:
    """Indexes of ``flow`` reviewer records closed by a later boundary of their round.

    Every reviewer record with the same round number and subject published
    before the boundary is history only.
    """
    boundaries = [
        (record.index, record.metadata.round_number, record.metadata.subject)  # type: ignore[attr-defined]
        for record in records
        if record.metadata.flow == flow  # type: ignore[attr-defined]
        and is_round_closure_boundary(record.metadata)  # type: ignore[attr-defined]
    ]
    if not boundaries:
        return frozenset()
    return frozenset(
        record.index  # type: ignore[attr-defined]
        for record in records
        if record.metadata.flow == flow  # type: ignore[attr-defined]
        and record.metadata.role == "reviewer"  # type: ignore[attr-defined]
        and any(
            record.index < index  # type: ignore[attr-defined]
            and record.metadata.round_number == round_number  # type: ignore[attr-defined]
            and record.metadata.subject == subject  # type: ignore[attr-defined]
            for index, round_number, subject in boundaries
        )
    )


def closure_item_number_high_water(metadata: object) -> int:
    """The closure payload's item-number high-water mark, or 0."""
    if not is_round_closed_incomplete(metadata):
        return 0
    payload = metadata.round_closure  # type: ignore[attr-defined]
    return max(
        [int(payload["item_number_high_water"])]
        + [int(entry["item_number"]) for entry in payload["evidence_obligations"]]
    )


def closure_evidence_obligations(metadata: object) -> tuple[Mapping[str, object], ...]:
    """The materialized evidence obligations of a closed-incomplete record."""
    if not is_round_closed_incomplete(metadata):
        return ()
    return tuple(metadata.round_closure["evidence_obligations"])  # type: ignore[attr-defined]


def restore_closure_evidence_obligations(
    unresolved_items: Sequence[object],
    metadata: object,
    *,
    next_item_number: int,
    clearances: Sequence[tuple[str, str]] = (),
    pre_clearance_items: Sequence[object] = (),
    known_heads: Sequence[str] = (),
) -> tuple[list[object], int]:
    """Restore a closure record's evidence obligations at their reserved numbers.

    Each obligation is upserted by its stable reviewer/text identity with its
    reserved item number, so a restart from the record alone rebuilds the same
    ledger even when ``new_items`` is empty.  The returned next item number
    never falls below the record's high-water mark (item IDs are never reused).
    """
    from .unresolved_items import _upsert_evidence_obligation

    items = list(unresolved_items)
    for entry in closure_evidence_obligations(metadata):
        items, _consumed = _upsert_evidence_obligation(
            items,  # type: ignore[arg-type]
            item_number=int(entry["item_number"]),  # type: ignore[arg-type]
            reviewer=str(entry["reviewer"]),
            text=str(entry["text"]),
            source_round=int(entry["source_round"]),  # type: ignore[arg-type]
            current_head_sha=str(entry["head"]),
            clearances=clearances,
            known_heads=tuple(dict.fromkeys((*known_heads, str(entry["head"])))),
            pre_clearance_items=pre_clearance_items,  # type: ignore[arg-type]
        )
    return items, max(next_item_number, closure_item_number_high_water(metadata) + 1)
