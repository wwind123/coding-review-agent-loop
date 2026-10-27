"""A converted freeze's capture opportunity, reconstructed from existing records."""

from .errors import HumanDecisionRequiredError
from .logging import log
from .converted_capture_state import (
    capture_candidates,
    evidence_review_boundary,
    capture_eligible_records,
    capture_review_context,
    capture_stale_reviewers,
)
from .pr_loop_support import admitted_capture_history


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
