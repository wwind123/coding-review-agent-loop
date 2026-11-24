"""Legacy reader for the signed-input supersession record payload (#2054).

A tool-owned Orchestrator summary with phase ``signed-input-supersession``
closed an unreconciled PR round whose published reviews were rejected only
because the signed requirement set grew since publication.  Since #2064 no
code path writes this record: the closed-incomplete round record replaced it.
Existing records are still read as "round closed" for one release window.

LEGACY: remove this reader (and the round-metadata field) after one release
window (#2064).

The payload is audit data: the round, head, superseded reviewer records, the
old and new surfaced requirement IDs and the item-number high-water mark.  It
is decoded strictly; a malformed payload decodes as ``None`` and the record
then never closes a round.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

SIGNED_INPUT_SUPERSESSION_PHASE = "signed-input-supersession"
INTEGRITY_ROUTES = frozenset({"carrier", "edit-history"})
_REQUIREMENT_ID_RE = re.compile(r"hr-[0-9a-f]{64}")


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _requirement_ids(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if not all(isinstance(item, str) and _REQUIREMENT_ID_RE.fullmatch(item) for item in value):
        return None
    if len(set(value)) != len(value):
        return None
    return tuple(sorted(value))


def decode_supersession_payload(
    raw: object, *, round_number: object, subject: object
) -> dict[str, object] | None:
    """Return the canonical payload, or ``None`` when it is malformed or mis-bound."""
    if not isinstance(raw, Mapping):
        return None
    if raw.get("round") != round_number or raw.get("head") != subject:
        return None
    if not _positive_int(raw.get("round")) or not isinstance(raw.get("head"), str) or not raw["head"]:
        return None
    previous = _requirement_ids(raw.get("previous_requirement_ids"))
    current = _requirement_ids(raw.get("requirement_ids"))
    # Pure growth only: the previous set is a strict subset of the new one.
    if previous is None or current is None or not set(previous) < set(current):
        return None
    superseded = raw.get("superseded")
    if not isinstance(superseded, (list, tuple)) or not superseded:
        return None
    entries: list[dict[str, object]] = []
    for entry in superseded:
        if not isinstance(entry, Mapping):
            return None
        agent, comment_id, route = entry.get("agent"), entry.get("comment_id"), entry.get("route")
        if not isinstance(agent, str) or not agent or route not in INTEGRITY_ROUTES:
            return None
        if comment_id is not None and not _positive_int(comment_id):
            return None
        entries.append({"agent": agent, "comment_id": comment_id, "route": route})
    if len({entry["agent"] for entry in entries}) != len(entries):
        return None
    high_water = raw.get("item_number_high_water")
    if not isinstance(high_water, int) or isinstance(high_water, bool) or high_water < 0:
        return None
    return {
        "round": raw["round"],
        "head": raw["head"],
        "superseded": entries,
        "previous_requirement_ids": list(previous),
        "requirement_ids": list(current),
        "item_number_high_water": high_water,
    }


def is_supersession_boundary(metadata: object) -> bool:
    """A decoded, tool-owned supersession record that closes its PR round."""
    return (
        getattr(metadata, "flow", None) == "pr"
        and getattr(metadata, "role", None) == "summary"
        and getattr(metadata, "agent", None) == "Orchestrator"
        and getattr(metadata, "phase", None) == SIGNED_INPUT_SUPERSESSION_PHASE
        and getattr(metadata, "signed_input_supersession", None) is not None
    )
