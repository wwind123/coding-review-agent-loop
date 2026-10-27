"""Pure converted-capture eligibility reconstructed from ordered round records."""

from .round_state import EVIDENCE_FREEZE_PHASE


def capture_candidates(records, *, items, head, matrix_identity, diagnostic=None):
    """Only ordered, admitted freeze/coder history can grant an opportunity.

    Every completed coder turn with the source ledger and a dispatch/result
    binding consumes it, even when it reports no evidence or pushes a new head.
    No round arithmetic or commit ordering is evidence of causal linkage.
    """
    def stop(reason):
        if diagnostic is not None:
            diagnostic(reason)
        return ()

    candidates = [i for i in items if i.converted_matrix_identity == matrix_identity
                  and i.converted_row_id and i.lifecycle == "evidence_deferred"]
    if not candidates:
        return ()
    audits = [r.metadata.evidence_conversion for r in records
              if r.metadata.evidence_conversion is not None
              and r.metadata.evidence_conversion.matrix_identity == matrix_identity]
    backed = {(row.obligation_item_id, row.obligation_identity)
              for audit in audits for row in audit.rows}
    candidates = [i for i in candidates if (i.item_id, i.obligation_identity) in backed]
    freezes = [r for r in records if r.metadata.phase == EVIDENCE_FREEZE_PHASE
               and r.metadata.evidence_freeze is not None
               and r.metadata.evidence_freeze.valid
               and r.metadata.evidence_freeze.frozen_head == r.metadata.subject
               and any((i.item_id, i.obligation_identity) in backed
                       for i in r.metadata.prior_items)]
    if not freezes or not candidates:
        return ()
    source = freezes[-1]
    if source.metadata.subject == head:
        return ()
    identities = {(i.item_id, i.obligation_identity) for i in source.metadata.prior_items
                  if (i.item_id, i.obligation_identity) in backed
                  and i.obligation_identity in source.metadata.evidence_freeze.evidence_identities}
    candidates = [i for i in candidates if (i.item_id, i.obligation_identity) in identities]
    after = [r for r in records if r.index > source.index]
    advanced = next((r.metadata.subject for r in after
                     if r.metadata.subject != source.metadata.subject), head)
    for record in after:
        m = record.metadata
        if m.role != "coder":
            continue
        bound = {(i.item_id, i.obligation_identity) for i in m.prior_items}
        # A completed publication consumes the opportunity. Missing causal
        # bindings cannot authorize a retry, including on legacy records.
        if m.followup_dispatch_head == advanced and identities.intersection(bound):
            return stop("A completed coder publication consumed converted recovery capture")
        return stop("Coder history lacks causal capture bindings; retaining the final evidence barrier")
    # Unrecorded second advances are ambiguous, not another capture episode.
    if advanced != head:
        return stop("Unrecorded head advance in capture history; retaining the final evidence barrier")
    return tuple(candidates)


def evidence_review_boundary(records, *, head, matrix_identity=None):
    """Newest converted capture publication whose evidence needs semantic review.

    Runner callers pass admitted history. Pure approval readers may reject
    approvals conservatively on decoded history, but never grant authority.
    Replacement evidence in the same recovery episode needs review too.
    Neither an obligation digest nor a round number establishes freshness.
    """
    from .protocol import EVIDENCE_SOURCE_TOOL_CAPTURED, parse_risk_test_matrix_evidence
    from .errors import AgentLoopError

    boundary = -1
    episodes = set()
    published_evidence = {}
    for position, record in enumerate(records):
        m = record.metadata
        if (m.role != "coder" or m.subject != head or not m.followup_dispatch_head
            or m.risk_test_matrix_capture_classification is None):
            continue
        audits = [r.metadata.evidence_conversion for r in records[:position]
                  if r.metadata.evidence_conversion is not None
                  and (matrix_identity is None
                       or r.metadata.evidence_conversion.matrix_identity == matrix_identity)]
        for audit in audits:
            candidates = capture_candidates(
                records[:position], items=m.prior_items,
                head=m.followup_dispatch_head, matrix_identity=audit.matrix_identity,
            )
            backed = {(row.obligation_item_id, row.obligation_identity, row.row_id)
                      for row in audit.rows}
            bound_rows = {i.converted_row_id for i in m.prior_items
                          if (i.item_id, i.obligation_identity, i.converted_row_id) in backed}
            episode = (audit.matrix_identity, head)
            if not candidates and episode not in episodes:
                continue
            episodes.add(episode)
            if not m.risk_test_matrix_evidence:
                continue
            try:
                evidence = parse_risk_test_matrix_evidence(
                    m.risk_test_matrix_evidence, expected_identity=audit.matrix_identity,
                )
            except AgentLoopError:
                continue
            for row in evidence.rows:
                if (row.row_id not in bound_rows or row.status != "verified"
                    or row.evidence_source != EVIDENCE_SOURCE_TOOL_CAPTURED
                    or not row.evidence_citations):
                    continue
                key = (audit.matrix_identity, row.row_id)
                if published_evidence.get(key) != row:
                    boundary = record.index
                    published_evidence[key] = row
    return boundary


def capture_eligible_records(records, *, head, plan=None):
    """Apply the capture-only freshness rule without consulting scheduler state."""
    boundary = evidence_review_boundary(
        records, head=head,
        matrix_identity=plan.risk_test_matrix_identity if plan is not None else None,
    )
    return tuple(record for record in records if record.index > boundary)


def capture_review_context(records, *, head, plan):
    """Keep captured claims and tool evidence visible even on summary-only resume."""
    import json

    boundary = evidence_review_boundary(
        records, head=head, matrix_identity=plan.risk_test_matrix_identity if plan else None,
    )
    if boundary < 0:
        return ""
    publication = next(record for record in records if record.index == boundary)
    return (
        "\nConverted-row capture awaiting semantic review: the tool verified the "
        "receipts, but reviewers must judge whether the cited tests exercise the "
        "affected rows. Earlier approvals do not approve this capture. Report "
        "coverage gaps as ordinary evidence-tagged findings; do not run commands "
        "or clear machine obligations.\n"
        + json.dumps(publication.metadata.risk_test_matrix_evidence, sort_keys=True)
        + "\n"
    )


def capture_stale_reviewers(records, *, head, boundary):
    """Latest exact-head approvals that have not judged the capture."""
    if boundary < 0:
        return ()
    latest = {}
    for record in records:
        if record.metadata.role == "reviewer" and record.metadata.subject == head:
            latest[record.metadata.agent] = record
    return tuple(name for name, record in latest.items()
                 if record.index <= boundary and record.metadata.state == "approved")


