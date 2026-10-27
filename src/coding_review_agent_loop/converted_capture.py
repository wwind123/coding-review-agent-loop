"""A converted freeze's capture opportunity, reconstructed from existing records."""

from .errors import HumanDecisionRequiredError
from .logging import log
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


def recovery_capture(runner, config, *, pr_number, items, head, plan):
    if plan is None or not plan.risk_test_matrix_identity:
        return ()
    if not any(i.converted_row_id and i.lifecycle == "evidence_deferred" for i in items):
        return ()
    from .pr_loop_support import _authenticated_pr_comments_status
    from .round_state import _extract_round_metadata_records, _recovery_record_admitted

    comments, actor, complete = _authenticated_pr_comments_status(
        runner, config, pr_number=pr_number,
    )
    if not complete:
        log(config, "Converted capture history is unavailable; retaining the final evidence barrier")
        return ()
    records = _extract_round_metadata_records(comments, flow="pr")
    # Author admission is required for the entire causal chain, not just audits.
    if any(not _recovery_record_admitted(r, comments, actor) for r in records):
        log(config, "Converted capture history has uncertain authorship; retaining the final evidence barrier")
        return ()
    return capture_candidates(
        records, items=items, head=head, matrix_identity=plan.risk_test_matrix_identity,
        diagnostic=lambda reason: log(config, reason),
    )


def capture_feedback(items):
    if not items:
        return ""
    rows = ", ".join(sorted(i.converted_row_id for i in items))
    return (
        f"Bounded recovery capture for converted risk-matrix rows: {rows}. "
        "Run the approved capture requirements at this live head through agent-loop run-tests "
        "and report semantic coverage claims and captured evidence. This opportunity also "
        "covers any ordinary code fixes below. Missing or failed capture preserves the "
        "conversion and final human evidence barrier. Do not classify these machine-owned "
        "evidence records as addressed or resolved merely because a test was attempted.\n\n"
    )


def recheck_capture(runner, config, *, pr_number, items, head, plan, expected):
    from .github import get_pr_review_context
    context = get_pr_review_context(runner, config=config, pr_number=pr_number)
    fresh = recovery_capture(runner, config, pr_number=pr_number, items=items, head=head, plan=plan)
    if context.metadata.head_sha != head or tuple(i.item_id for i in fresh) != tuple(i.item_id for i in expected):
        raise HumanDecisionRequiredError("Converted capture inputs changed before dispatch; evidence barriers retained.")
