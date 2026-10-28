"""Pure raw obligation readers for managed continuity."""
from collections.abc import Mapping
from .protocol import MACHINE_AUTHORITY, CI_MACHINE_OBLIGATION_KINDS

_MERGE_CONFLICT_OPEN_STATUSES = frozenset({"blocking", "same-pr"})
_MERGE_CONFLICT_LIVE_LIFECYCLES = frozenset({"repair_required", "awaiting_current_head_review"})


def _records_merge_conflict_obligation(payload: Mapping[str, object]) -> bool:
    """Whether this record carries the tool-owned merge-conflict obligation.

    Only the shape the orchestrator writes for a conflict round counts: a
    machine-authority merge-conflict obligation that is still open
    (``blocking``/``same-pr``) and live (``repair_required`` as minted, or
    ``awaiting_current_head_review`` once advanced for the pushed head).  A
    resolved, deferred, cleared or qualification-stage item is not evidence of
    a conflict round (#1057).
    """
    for key in ("prior_items", "new_items"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            # Items are arbitrary JSON from any commenter and are decoded before
            # author filtering, so raw fields are type-checked before they are
            # hashed by a membership test (#1370).
            status = item.get("status")
            lifecycle = item.get("lifecycle")
            if (
                item.get("authority") == MACHINE_AUTHORITY
                and item.get("obligation_kind") == "merge-conflict"
                and isinstance(status, str)
                and status in _MERGE_CONFLICT_OPEN_STATUSES
                and isinstance(lifecycle, str)
                and lifecycle in _MERGE_CONFLICT_LIVE_LIFECYCLES
            ):
                return True
    return False


def _records_ci_repair_transitions(
    payload: Mapping[str, object],
) -> frozenset[tuple[str, str]]:
    """Return the (failed, candidate) heads bound by tool-owned CI obligations.

    Only the post-push shape counts: a machine-authority CI obligation that the
    orchestrator advanced to ``awaiting_current_head_review`` for a candidate
    head distinct from the head that failed.
    """
    transitions: set[tuple[str, str]] = set()
    for key in ("prior_items", "new_items"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            # Items are arbitrary JSON from any commenter and are decoded before
            # author filtering, so raw fields are type-checked before they are
            # hashed by a membership test (#1370).
            obligation_kind = item.get("obligation_kind")
            status = item.get("status")
            failed = item.get("failed_head_sha")
            candidate = item.get("candidate_head_sha")
            if (
                item.get("authority") == MACHINE_AUTHORITY
                and isinstance(obligation_kind, str)
                and obligation_kind in CI_MACHINE_OBLIGATION_KINDS
                and item.get("lifecycle") == "awaiting_current_head_review"
                and isinstance(status, str)
                and status in {"blocking", "same-pr"}
                and isinstance(failed, str)
                and isinstance(candidate, str)
                and failed
                and candidate
                and failed != candidate
            ):
                transitions.add((failed, candidate))
    return frozenset(transitions)
