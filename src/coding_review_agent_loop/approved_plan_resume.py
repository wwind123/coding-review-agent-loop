"""Read-only approved-plan resume views and refusal diagnostics.

Raw history remains authoritative for lineage and approval qualification. Only
an unused, immediately adjacent neutral advance can select its settled predecessor.
"""
from dataclasses import replace

from .errors import AgentLoopError


def approved_plan_resume_view(comments, resumed, *, config, human_requirements, min_round_number=1):
    from .config import reviewers
    from .panel_evidence import _require_complete_canonical_plan_approval
    from .plan_round_closure import project_plan_history, _plan_round_is_reconciled
    from .round_state import (
        _extract_round_metadata_records, _resume_plan_round, canonical_deferred_history_items,
    )

    if resumed is None:
        return None
    records = tuple(record for record in project_plan_history(
        _extract_round_metadata_records(comments, flow="plan")
    ) if record.metadata.round_number >= min_round_number)
    candidate = resumed
    neutral = None
    if len(records) >= 2:
        advance, predecessor = records[-1], records[-2]
        meta = advance.metadata
        if (
            meta.role == "summary" and meta.phase == "plan-phase-advance"
            and meta.scheduler_contract is None
            and not meta.new_items and not meta.dispositions and not meta.evidence_clearances
            and meta.reviewer_board_amendment_digest is None
            and meta.subject == predecessor.metadata.subject
            and meta.round_number == predecessor.metadata.round_number + 1
            and _plan_round_is_reconciled((predecessor,))
            and (meta.plan_candidate_key is None or predecessor.metadata.plan_candidate_key is None
                 or meta.plan_candidate_key == predecessor.metadata.plan_candidate_key)
        ):
            neutral = advance
            # Prefix selection only: no deletion, rewriting, or global projection.
            candidate = _resume_plan_round(
                comments[:advance.index], configured_reviewers=reviewers(config),
                min_round_number=min_round_number,
            )
    if candidate is None:
        return resumed
    text, plan_round = candidate
    if not plan_round.reconciled:
        return resumed
    try:
        _require_complete_canonical_plan_approval(
            comments, config=config, plan_text=text, plan_round=plan_round,
            human_requirements=human_requirements, error_message="incomplete approval",
        )
    except AgentLoopError:
        return resumed
    # Reuse lifecycle replay rather than unioning historical future labels.
    # Current-round items and dispositions are still applied by the live gate.
    predecessor_records = tuple(
        record for record in records if record.metadata.round_number <= plan_round.round_number
    )
    deferred = canonical_deferred_history_items(
        predecessor_records, reconciliation_mode="aggregate", same_status="same-plan",
    )
    for record in predecessor_records:
        if record.metadata.role == "reviewer":
            invalid = [item.item_id for item in record.metadata.new_items
                       if item.reviewer != record.metadata.agent]
            if invalid:
                raise AgentLoopError(
                    "Cannot reconstruct approved predecessor finding provenance: "
                    + ", ".join(invalid) + ". Repair the named durable records before resuming."
                )
    if neutral is not None:
        from .plan_verification import plan_round_candidate_key
        from .response_validation import _surfaced_reviewer_requirement_ids

        key = plan_round_candidate_key(
            plan_round, plan_text=text, surfaced_ids=_surfaced_reviewer_requirement_ids(
                human_requirements, requirement_scope="planning requirements",
            ),
        )
        if neutral.metadata.plan_candidate_key is not None and (
            key is None or neutral.metadata.plan_candidate_key != key.as_dict()
        ):
            return resumed
        known = {item.item_id: item for item in (
            *plan_round.prior_items, *plan_round.current_round_new_items, *deferred,
        )}
        if any(item.status in {"blocking", "same-plan"} or known.get(item.item_id) != item
               for item in neutral.metadata.prior_items):
            return resumed
    present = {item.item_id for item in (*plan_round.prior_items, *plan_round.current_round_new_items)}
    return text, replace(
        plan_round,
        prior_items=(*plan_round.prior_items, *(item for item in deferred if item.item_id not in present)),
        next_unresolved_item_number=max(plan_round.next_unresolved_item_number, resumed[1].next_unresolved_item_number),
    )


def approval_shortcut_disagreement(*, plan_hash, round_number, missing=(), records=(),
                                  seat_binding=None, restoration_rounds=None,
                                  parent_requirement_ids=None, predicate="live approval", obligations=()):
    details = []
    for name in missing:
        record = next((record for record in reversed(records)
                       if record.metadata.role == "reviewer" and record.metadata.agent == name), None)
        meta = record.metadata if record else None
        details.append(
            f"{name}: status={meta.state if meta else 'missing'}, "
            f"approval round={meta.round_number if meta else 'none'}, "
            f"seat binding={meta.seat_binding if meta else None!r} (required {seat_binding!r}), "
            f"candidate binding={meta.plan_candidate_key if meta else None!r}, "
            f"restoration round={(restoration_rounds or {}).get(name)}, "
            f"parent requirement binding={meta.parent_requirement_ids if meta else None!r} "
            f"(required {parent_requirement_ids!r})"
        )
    return AgentLoopError(
        f"Plan {plan_hash}, resumed round {round_number}: complete-approval classification "
        f"disagrees with {predicate}; missing qualifying reviewer(s): "
        f"{'; '.join(details) or 'none'}; blocking obligation(s): {', '.join(obligations) or 'none'}. "
        "Inspect the named approval bindings and round records, restore qualifying exact-plan "
        "approvals or explicitly reopen planning under the current requirements. "
        "No reviewer-only advance or agent turn was started."
    )
