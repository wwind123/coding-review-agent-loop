"""Pure assignments and semantic acceptance for admitted capture publications.

Callers supply authenticated history and independently qualified code approvals.
Neither a tool receipt nor an evidence-only verdict grants code authority.
"""
import hashlib
import json
from collections.abc import Mapping

from .review_scope import is_complete_code_review, valid_evidence_scope


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reviewer_contract(metadata):
    """The durable inputs that already govern exact-head code approval."""
    return _digest({key: getattr(metadata, key, None) for key in (
        "agent", "subject", "approved_plan_hash", "approved_plan_subject",
        "surfaced_reviewer_requirement_ids", "architecture_contract_version",
        "architecture_identity", "seat_binding", "configured_model", "configured_effort",
        "provider", "scheduler_contract", "reviewer_board_amendment_digest",
        "pr_board_identity", "reviewer_board_change_digest",
    )})


def _capture_inputs(records, *, head, matrix_identity):
    """Canonical publication, covered converted rows and admitted owners."""
    from .converted_capture_state import evidence_review_boundary

    boundary = evidence_review_boundary(records, head=head, matrix_identity=matrix_identity)
    if boundary < 0:
        return None
    publication = next(r for r in records if r.index == boundary)
    evidence = publication.metadata.risk_test_matrix_evidence
    if not isinstance(evidence, Mapping):
        return None
    matrix_identity = evidence.get("matrix_identity")
    rows = {row["row_id"]: row for row in evidence.get("rows", ())
            if row.get("status") == "verified" and row.get("evidence_citations")}
    owners = {}
    backed = {(item.item_id, item.obligation_identity, item.converted_row_id)
              for item in publication.metadata.prior_items}
    for record in records:
        audit = record.metadata.evidence_conversion
        if record.index >= boundary or audit is None or audit.matrix_identity != matrix_identity:
            continue
        for row in audit.rows:
            if (row.row_id in rows
                and (row.obligation_item_id, row.obligation_identity, row.row_id) in backed):
                if row.row_id in owners and owners[row.row_id] != row.owners:
                    return None
                owners[row.row_id] = row.owners
    if not owners:
        return None
    episode = {"round": publication.metadata.round_number,
               "dispatch_head": publication.metadata.followup_dispatch_head,
               "obligations": sorted((item.item_id, item.obligation_identity)
                                      for item in publication.metadata.prior_items
                                      if item.converted_row_id in owners)}
    capture = _digest({"publication": episode, "head": head, "matrix": matrix_identity,
                       "evidence": {row: rows[row] for row in sorted(owners)}})
    return {"capture": capture, "head": head, "matrix": matrix_identity}, owners


def capture_assignments(records, *, head, matrix_identity, required_reviewers, primary=None,
                        approvals=None):
    """Return bound reviewer scopes, or None when ownership is uncertain.

    Required names and approval contracts come from the effective board, after
    restoration, acquisition, panel, sweep and other code qualification checks.
    """
    inputs = _capture_inputs(records, head=head, matrix_identity=matrix_identity)
    if inputs is None:
        return None
    binding, owners = inputs
    assigned = {}
    for row, requesting in sorted(owners.items()):
        owner = primary if primary in required_reviewers else next(
            (name for name in required_reviewers if name in requesting), None)
        if owner is None or approvals is None or owner not in approvals:
            return None
        assigned.setdefault(owner, []).append(row)
    return {name: {"version": 1, "kind": "capture-evidence", **binding, "rows": row_ids,
                   "contract": reviewer_contract(approvals[name].metadata)}
            for name, row_ids in assigned.items()}


def accepted_capture_reviewers(records, *, assignments, head, boundary, approvals):
    """Only published, bound latest decisions count, never private spool output.

    A complete review after the publication is capture-aware in the current
    protocol: all PR prompt forms include the unabridged capture context.
    Legacy complete reviews retain the conservative all-board fallback.
    """
    latest = {}
    for record in records:
        m = record.metadata
        if m.role != "reviewer" or m.subject != head or record.index <= boundary:
            continue
        expected = assignments.get(m.agent)
        if expected is None:
            continue
        if is_complete_code_review(m):
            latest[m.agent] = bool(
                m.agent in approvals and approvals[m.agent].index == record.index
                and m.state == "approved"
                and m.capture_review_binding == {key: expected[key] for key in ("capture", "head", "matrix")})
        else:
            latest[m.agent] = bool(valid_evidence_scope(m.review_scope) and m.review_scope == expected
                and reviewer_contract(m) == expected["contract"]
                and m.state == "approved" and m.acquisition_outcome == "success")
    return frozenset(name for name, accepted in latest.items() if accepted)


def capture_pending(records, *, head, plan, required_reviewers, primary=None, approvals=None):
    """Shared scheduling and final-qualification semantic acceptance gate."""
    from .converted_capture_state import evidence_review_boundary

    matrix = plan.risk_test_matrix_identity if plan is not None else None
    boundary = evidence_review_boundary(records, head=head, matrix_identity=matrix)
    if boundary < 0:
        return False
    assignments = capture_assignments(records, head=head, matrix_identity=matrix,
        required_reviewers=required_reviewers, primary=primary, approvals=approvals)
    if assignments is None:
        # Ambiguous legacy history requires fresh complete work from the board.
        binding = full_capture_binding(records, head=head, matrix_identity=matrix)
        return any(name not in (approvals or {}) or approvals[name].index <= boundary
                   or not binding or approvals[name].metadata.capture_review_binding != binding
                   for name in required_reviewers)
    return set(assignments) != set(accepted_capture_reviewers(records, assignments=assignments,
        head=head, boundary=boundary, approvals=approvals or {}))


def assigned_findings(items, scope, reviewer):
    """Scope never authorizes machine clearance or another owner's findings."""
    if not valid_evidence_scope(scope):
        return ()
    return tuple(item for item in items if item.reviewer == reviewer and not item.authority
                 and item.evidence_row_ids and set(item.evidence_row_ids) <= set(scope["rows"]))


def evidence_prompt(scope, *, plan=None):
    if scope is None:
        return ""
    requirements = [row for row in (plan.risk_test_matrix_payload or {}).get("rows", ())
                    if row.get("row_id") in scope["rows"]] if plan is not None else []
    return (
        "\nEVIDENCE-ONLY REVIEW: judge semantic coverage only for assigned rows "
        + ", ".join(scope["rows"]) + ". Independent code approvals are already complete. "
        "Your verdict grants capture acceptance only. Do not perform a code sweep, "
        "open a panel, disposition unrelated findings or clear machine obligations. "
        "Report missing transitions as actionable findings tagged with assigned evidence_row_ids. "
        "Keep the existing structured PR-review format and no-execution policy.\n"
        + "Assigned approved row requirements:\n" + json.dumps(requirements, sort_keys=True) + "\n"
    )


def full_capture_binding(records, *, head, matrix_identity):
    """Durable proof that an ordinary prompt carried this capture's claims."""
    inputs = _capture_inputs(records, head=head, matrix_identity=matrix_identity)
    return inputs[0] if inputs is not None else None
