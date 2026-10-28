"""Bound capture ledger reconciliation, independent of complete code rounds."""
from collections.abc import Mapping

from .capture_acceptance import full_capture_binding, assigned_findings, reviewer_contract
from .review_scope import valid_evidence_scope
from .unresolved_items import _apply_unresolved_item_dispositions


def reconcile_capture_record(items, metadata, *, binding):
    """Apply only a matching assigned owner's dispositions and tagged findings.

    Callers admit the record's actor and effective seat before invoking this
    pure reader. Scope is not a generic ledger-clearance capability.
    """
    scope = metadata.review_scope
    if valid_evidence_scope(scope):
        assignments = {metadata.agent: scope}
        if reviewer_contract(metadata) != scope["contract"]:
            return items
    elif (metadata.role == "summary" and isinstance(scope, Mapping)
          and scope.get("kind") == "capture-reconciliation"
          and isinstance(scope.get("assignments"), Mapping)):
        assignments = scope["assignments"]
    else:
        return items
    assignments = {name: own for name, own in assignments.items()
                   if valid_evidence_scope(own) and binding
                   and all(own[key] == binding[key] for key in ("capture", "head", "matrix"))}
    result = {item.item_id: item for item in items}
    for reviewer, own in assignments.items():
        eligible = assigned_findings(tuple(result.values()), own, reviewer)
        allowed = {item.item_id for item in eligible}
        dispositions = {}
        for disposition in metadata.dispositions:
            if disposition.reviewer == reviewer and disposition.item_id in allowed:
                dispositions.setdefault(disposition.item_id, []).append(disposition)
        kept, future = _apply_unresolved_item_dispositions(eligible, dispositions, retain_future=True)
        for item_id in allowed:
            result.pop(item_id, None)
        result.update((item.item_id, item) for item in (*kept, *future))
        for item in assigned_findings(metadata.new_items, own, reviewer):
            result[item.item_id] = item
    return tuple(result.values())


def capture_finding_ids(records, *, binding):
    """Only findings created by a bound scoped turn are semantic-only work."""
    return frozenset(item.item_id for record in records
                     for item in reconcile_capture_record((), record.metadata, binding=binding))
