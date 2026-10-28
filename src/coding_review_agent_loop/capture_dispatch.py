"""PR dispatch adapters for the pure capture acceptance contract."""
from .agents.registry import agent_display_name
from .capture_acceptance import capture_assignments, accepted_capture_reviewers, assigned_findings, reviewer_contract
from .review_scope import is_complete_code_review


def narrow_capture_scopes(records, *, head, plan, required_reviewers, primary,
                          approvals, boundary, items, eligible):
    if not eligible or boundary < 0 or plan is None or set(approvals) != set(required_reviewers):
        return {}
    scopes = capture_assignments(records, head=head, matrix_identity=plan.risk_test_matrix_identity,
        required_reviewers=required_reviewers, primary=primary, approvals=approvals)
    if not scopes:
        return {}
    from .capture_reconciliation import capture_finding_ids
    from .capture_acceptance import full_capture_binding
    semantic_ids = capture_finding_ids(records, binding=full_capture_binding(
        records, head=head, matrix_identity=plan.risk_test_matrix_identity))
    if any(item.status in {"blocking", "same-pr"} and not item.is_inert_ledger_record
           and not item.authority and item.item_id not in semantic_ids for item in items):
        return {}
    accepted = accepted_capture_reviewers(records, assignments=scopes, head=head,
                                          boundary=boundary, approvals=approvals)
    return {name: scope for name, scope in scopes.items() if name not in accepted}


def scoped_review_items(items, scopes, reviewer):
    scope = scopes.get(reviewer)
    return assigned_findings(items, scope, reviewer) if scope is not None else items


def scoped_validator(validate, *, scopes, reviewer, items):
    """Forbid unsolicited dispositions and unassigned/machine evidence requests."""
    scope = scopes.get(reviewer)
    if scope is None:
        return validate
    allowed = {item.item_id for item in assigned_findings(items, scope, reviewer)}
    def checked(text):
        from .errors import AgentLoopError
        parsed = validate(text)
        if any(d.item_id not in allowed for d in parsed.dispositions):
            raise AgentLoopError("Evidence-only review disposition is outside the assigned findings")
        if parsed.exact_head_evidence_requests:
            raise AgentLoopError("Evidence-only review cannot create machine evidence obligations")
        for item in (*parsed.blocking_items, *parsed.followups.same_pr):
            if not item.evidence_row_ids or not set(item.evidence_row_ids) <= set(scope["rows"]):
                raise AgentLoopError("Evidence-only findings must name assigned evidence rows")
        return parsed
    return checked


def qualification_capture_pending(records, *, config, comments, head, plan, requirements):
    from .capture_acceptance import capture_pending
    from .round_state import _latest_pr_approved_reviews_for_head
    from .reviewer_seats import reviewer_seat_binding, SeatAgent, seat_backend
    from .config import resolve_invocation
    acquisition = {}
    for reviewer in config.reviewer:
        invocation = resolve_invocation(config, provider=reviewer, role="reviewer")
        acquisition[agent_display_name(reviewer)] = (
            reviewer.model_chain if isinstance(reviewer, SeatAgent) else invocation.configured_model,
            invocation.resolved_effort, seat_backend(reviewer))
    approvals = _latest_pr_approved_reviews_for_head(comments, head_sha=head,
        configured_reviewers=config.reviewer, approved_plan_context=plan,
        human_requirements=requirements, require_architecture_contract=config.architecture_context_enabled,
        reviewer_acquisition_contract=acquisition if config.pr_review_policy != "all-reviewers" or config.reviewer_seats else None)
    binding = reviewer_seat_binding(config)
    if binding is not None:
        approvals = {name: record for name, record in approvals.items() if record.metadata.seat_binding == binding}
    return capture_pending(records, head=head, plan=plan,
        required_reviewers=tuple(agent_display_name(reviewer) for reviewer in config.reviewer),
        primary=agent_display_name(config.primary_reviewer)
                if config.pr_review_policy == "primary-then-panel" and config.primary_reviewer else None,
        approvals=approvals)


def select_capture_reviewers(decision, scopes, required_reviewers):
    """An evidence pass has no code-sweep or panel-opening authority."""
    from dataclasses import replace
    return replace(decision,
        selected_reviewers=tuple(name for name in required_reviewers if name in scopes),
        paused_reviewers=tuple((name, "complete code approval retained")
                              for name in required_reviewers if name not in scopes),
        final_sweep=False, phase="capture-evidence", reason="bound capture evidence-only acceptance")


def resume_capture_round(records, *, comments, trusted_actor, head):
    """Replay admitted scoped responses through settlement, without code authority."""
    from .round_state import (ResumedReviewRound, _recovery_record_admitted,
                              _max_unresolved_item_number_from_records)
    from .review_scope import scope_present, valid_evidence_scope
    current = [record for record in records if record.metadata.subject == head]
    if not current:
        return None
    latest = current[-1]
    if not scope_present(latest.metadata) or not _recovery_record_admitted(latest, comments, trusted_actor):
        return None
    scope = latest.metadata.review_scope
    if not valid_evidence_scope(scope):
        if not (isinstance(scope, dict) and scope.get("kind") == "capture-reconciliation"
                and isinstance(scope.get("assignments"), dict)
                and scope["assignments"]
                and all(valid_evidence_scope(s) for s in scope["assignments"].values())):
            return None
    coder = next((r for r in reversed(current) if r.metadata.role == "coder"), None)
    # Keep the pre-review ledger intact until the existing settlement path has
    # replayed the public responses. Treating round-end findings as prior work
    # changes scheduling inputs and can redispatch an already published turn.
    items = latest.metadata.prior_items
    completed = [record for record in current
        if record.metadata.round_number == latest.metadata.round_number
        and record.metadata.role == "reviewer"
        and valid_evidence_scope(record.metadata.review_scope)
        and _recovery_record_admitted(record, comments, trusted_actor)]
    reconciled = latest.metadata.role == "summary" and latest.metadata.phase == "reconciliation"
    if reconciled:
        from dataclasses import replace
        completed = [replace(record, metadata=replace(record.metadata, phase="authoritative",
            new_items=assigned_findings(latest.metadata.new_items, record.metadata.review_scope,
                                       record.metadata.agent))) for record in completed]
    return ResumedReviewRound(round_number=latest.metadata.round_number,
        prior_items=tuple(items), completed_reviews=tuple(completed),
        coder_output=(coder.metadata.raw_structured_coder_response or coder.body) if coder else None,
        coder_metadata=coder.metadata if coder else None,
        coder_comment_body=coder.body if coder else None,
        next_unresolved_item_number=_max_unresolved_item_number_from_records(records) + 1,
        reconciled=reconciled, qualification_checkpoint=None)


def capture_review_recovery(records, *, head, plan, required_reviewers, primary,
                            approvals, scopes, resumed, boundary):
    """Recompute carry/resume whenever final scheduling changes capture scope."""
    from .capture_acceptance import capture_pending, full_capture_binding
    pending = not scopes and capture_pending(records, head=head, plan=plan,
        required_reviewers=required_reviewers, primary=primary, approvals=approvals)
    binding = full_capture_binding(records, head=head,
        matrix_identity=plan.risk_test_matrix_identity if plan else None)
    return pending, {name: record for name, record in resumed.items()
        if (name in scopes and record.index > boundary
            and record.metadata.review_scope == scopes[name]
            and reviewer_contract(record.metadata) == scopes[name]["contract"])
        or (name not in scopes and is_complete_code_review(record.metadata)
            and (not pending or (binding is not None and record.index > boundary
                and record.metadata.capture_review_binding == binding)))}


def is_resumable_capture_review(metadata):
    from .review_scope import valid_evidence_scope
    return is_complete_code_review(metadata) or (valid_evidence_scope(metadata.review_scope)
        and metadata.acquisition_outcome == "success")
