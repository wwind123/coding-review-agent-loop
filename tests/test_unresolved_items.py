"""Ledger unit tests for human-only exact-head evidence obligations (#1068)."""

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.orchestrator import (
    _partition_unresolved_items,
    _refuse_dispatch_while_evidence_frozen,
    _round_limit_diagnostic,
    _scheduler_obligations,
)
from coding_review_agent_loop.protocol import ReviewItemDisposition
from coding_review_agent_loop.review_scheduling import GitChange, classify_transition
from coding_review_agent_loop.unresolved_items import (
    _advance_machine_obligations_for_head,
    _apply_unresolved_item_dispositions,
    _machine_obligation_is_revalidation_candidate,
    _machine_obligation_requires_repair,
    _next_unresolved_item,
    _pending_evidence_obligations,
    _set_machine_obligation_lifecycle,
    _upsert_evidence_obligation,
    _upsert_machine_obligation,
    coder_followup_is_ci_repair,
    evidence_obligation_identity,
    format_coder_followup_context,
    freeze_evidence_obligations,
    release_evidence_freeze,
    select_coder_followup_items,
)

REQUEST = "Attach authenticated live-CLI suite output for this exact head"
H1 = "1111111111111111111111111111111111111111"
H2 = "2222222222222222222222222222222222222222"


def _code_item(reviewer="Codex", number=1, scope=("src/app.py",)):
    return _next_unresolved_item(
        item_number=number,
        reviewer=reviewer,
        source_round=1,
        text="Fix the off-by-one in the pager.",
        status="blocking",
        fix_scope=scope,
    )


def _evidence(items, *, reviewer="Gemini", number=2, head=H1, clearances=(), text=REQUEST):
    return _upsert_evidence_obligation(
        items,
        item_number=number,
        reviewer=reviewer,
        text=text,
        source_round=1,
        current_head_sha=head,
        clearances=clearances,
    )


def test_evidence_request_is_deduplicated_by_identity_and_stays_deferred():
    ledger, consumed = _evidence([_code_item()])
    assert consumed is True
    # The same reviewer repeating the request (whitespace/case differences
    # included) updates one obligation instead of minting another.
    ledger, consumed_again = _evidence(ledger, number=3, head=H2, text="  attach AUTHENTICATED live-CLI suite output for this exact head. ")
    assert consumed_again is False
    evidence = _pending_evidence_obligations(ledger)
    assert len(evidence) == 1
    assert evidence[0].lifecycle == "evidence_deferred"
    assert evidence[0].obligation_identity == evidence_obligation_identity(REQUEST, "Gemini")
    # A different reviewer asking for the same evidence is its own obligation.
    ledger, consumed_other = _evidence(ledger, reviewer="Codex", number=3)
    assert consumed_other is True
    assert len(_pending_evidence_obligations(ledger)) == 2


def test_evidence_is_final_barrier_not_coder_or_repair_work():
    ledger, _ = _evidence([_code_item()])
    evidence = _pending_evidence_obligations(ledger)[0]

    partitions = _partition_unresolved_items(ledger, current_head_sha=H1)
    assert partitions["evidence_obligations"] == (evidence,)
    assert evidence in partitions["finalization_blockers"]
    assert evidence not in partitions["coder_blockers"]
    assert evidence not in partitions["repair_required_machine_obligations"]
    assert evidence not in partitions["reviewer_blockers"]
    assert _machine_obligation_requires_repair(evidence, current_head_sha=H1) is False
    assert _machine_obligation_is_revalidation_candidate(evidence, current_head_sha=H1) is False

    # Head advance never rewrites the evidence lifecycle; the freeze owns it.
    advanced = _advance_machine_obligations_for_head(ledger, current_head_sha=H2)
    assert _pending_evidence_obligations(advanced) == (evidence,)

    # Coder follow-up classification and its rendered context exclude it.
    assert evidence not in select_coder_followup_items(ledger)
    assert REQUEST not in format_coder_followup_context(ledger)
    assert coder_followup_is_ci_repair(select_coder_followup_items([evidence])) is False

    # The kind-singleton helpers refuse the evidence kind.
    with pytest.raises(AgentLoopError, match="by identity"):
        _upsert_machine_obligation(
            ledger, item_number=9, kind="human-exact-head-evidence",
            source_round=1, text="x", failed_head_sha=None,
        )
    with pytest.raises(AgentLoopError):
        _set_machine_obligation_lifecycle(
            ledger, kind="human-exact-head-evidence", lifecycle="cleared"
        )


def test_deferred_evidence_does_not_force_broad_scheduler_classification():
    """Row evidence-deferred-across-heads: the H1 -> H2 transition is narrow."""
    ledger, _ = _evidence([_code_item()])
    obligations = _scheduler_obligations(ledger, required_reviewers=("Codex", "Gemini"))
    assert [obligation.item_id for obligation in obligations] == ["item-1"]
    assert obligations[0].scope == ("src/app.py",)
    classification = classify_transition(
        H1,
        H2,
        [GitChange("src/app.py")],
        scopes=None,
        obligations=obligations,
    )
    assert classification.kind == "narrow"
    # Without the exclusion the unscoped evidence owner would force broad.
    unscoped = classify_transition(
        H1, H2, [GitChange("src/app.py")], scopes=None,
        obligations=_scheduler_obligations(
            [ledger[0], _code_item(reviewer="Orchestrator", number=5, scope=None)],
            required_reviewers=("Codex", "Gemini"),
        ),
    )
    assert unscoped.kind == "broad"


def test_round_limit_diagnostic_names_code_blocker_first_and_evidence_as_barrier():
    ledger, _ = _evidence([_code_item()])
    diagnostic = _round_limit_diagnostic(
        pr_number=77, round_number=2, items=ledger, current_head_sha=H2
    )
    assert "reviewer-owned findings: Codex (item-1)" in diagnostic
    assert "final barrier" in diagnostic
    evidence_only = _round_limit_diagnostic(
        pr_number=77, round_number=2, items=_pending_evidence_obligations(ledger),
        current_head_sha=H2,
    )
    assert "remaining barrier is human-only exact-head evidence" in evidence_only
    assert "unknown or unreconstructible" not in evidence_only


def test_freeze_and_release_helpers_bind_and_unbind_the_head():
    ledger, _ = _evidence([])
    frozen = freeze_evidence_obligations(ledger, head_sha=H1)
    assert frozen[0].lifecycle == "evidence_frozen"
    assert frozen[0].candidate_head_sha == H1
    released = release_evidence_freeze(frozen)
    assert released[0].lifecycle == "evidence_deferred"
    assert released[0].candidate_head_sha is None


def test_dispatch_guard_refuses_while_frozen_only():
    """Row frozen-refuses-push: a frozen head can never be moved by a dispatch."""
    ledger, _ = _evidence([_code_item()])
    _refuse_dispatch_while_evidence_frozen(ledger, pr_number=77, operation="a coder follow-up")
    frozen = freeze_evidence_obligations(ledger, head_sha=H1)
    for operation in ("a coder follow-up or CI repair", "a merge-conflict resolution"):
        with pytest.raises(AgentLoopError, match=f"Refusing to dispatch {operation}.*frozen at head {H1}"):
            _refuse_dispatch_while_evidence_frozen(frozen, pr_number=77, operation=operation)


def _disposition(item_id, reviewer, disposition, note=None):
    return ReviewItemDisposition(item_id=item_id, reviewer=reviewer, disposition=disposition, note=note)


def test_owner_only_clearance_and_same_head_reemission_is_ignored():
    """Row evidence-owner-and-reemission."""
    ledger, _ = _evidence([], reviewer="Codex", number=1, text="Evidence A")
    ledger, _ = _evidence(ledger, reviewer="Gemini", number=2, text="Evidence B")
    frozen = freeze_evidence_obligations(ledger, head_sha=H1)
    e_a, e_b = frozen
    clearances: list[tuple[str, str]] = []
    dispositions = {
        e_a.item_id: [
            _disposition(e_a.item_id, "Codex", "resolved", "Signed output covers H1."),
            _disposition(e_a.item_id, "Gemini", "resolved", "Looks fine to me."),
        ],
        e_b.item_id: [
            _disposition(e_b.item_id, "Codex", "resolved", "Not my request."),
            _disposition(e_b.item_id, "Gemini", "blocking", "The run output omits the sandbox case."),
        ],
    }
    remaining, _future = _apply_unresolved_item_dispositions(
        frozen,
        dispositions,
        evidence_response_head=H1,
        configured_reviewers=("Codex", "Gemini"),
        evidence_clearances=clearances,
    )
    assert [item.item_id for item in remaining] == [e_b.item_id]
    assert remaining[0].lifecycle == "evidence_frozen"
    # The non-owner's resolution of E_B is a note, never a waiver.
    assert "Codex: Not my request." in remaining[0].notes
    assert clearances == [(e_a.obligation_identity, H1)]

    # Dispositions first, then re-emission: A's identical request at H1 hits
    # the head-scoped clearance entry instead of recreating the obligation.
    after, consumed = _evidence(
        remaining, reviewer="Codex", number=3, head=H1, text="Evidence A", clearances=clearances
    )
    assert consumed is False
    assert [item.item_id for item in after] == [e_b.item_id]
    # At a later head the same request reactivates as deferred.
    later, consumed_later = _evidence(
        after, reviewer="Codex", number=3, head=H2, text="Evidence A", clearances=clearances
    )
    assert consumed_later is True
    reactivated = [item for item in later if item.item_id == "item-3"][0]
    assert reactivated.lifecycle == "evidence_deferred"


def test_owner_cannot_clear_outside_an_evidence_response_pass():
    ledger, _ = _evidence([], reviewer="Codex", number=1)
    frozen = freeze_evidence_obligations(ledger, head_sha=H1)
    dispositions = {"item-1": [_disposition("item-1", "Codex", "resolved", "done")]}
    remaining, _ = _apply_unresolved_item_dispositions(frozen, dispositions)
    assert [item.item_id for item in remaining] == ["item-1"]
    # A response at another head than the frozen one cannot clear either.
    remaining, _ = _apply_unresolved_item_dispositions(
        frozen, dispositions, evidence_response_head=H2, configured_reviewers=("Codex",)
    )
    assert [item.item_id for item in remaining] == ["item-1"]


def test_departed_owner_requires_unanimous_configured_board():
    ledger, _ = _evidence([], reviewer="Claude", number=1)
    frozen = freeze_evidence_obligations(ledger, head_sha=H1)
    partial = {"item-1": [_disposition("item-1", "Codex", "resolved", "ok")]}
    remaining, _ = _apply_unresolved_item_dispositions(
        frozen, partial, evidence_response_head=H1, configured_reviewers=("Codex", "Gemini")
    )
    assert len(remaining) == 1
    unanimous = {
        "item-1": [
            _disposition("item-1", "Codex", "resolved", "ok"),
            _disposition("item-1", "Gemini", "resolved", "ok"),
        ]
    }
    remaining, _ = _apply_unresolved_item_dispositions(
        frozen, unanimous, evidence_response_head=H1, configured_reviewers=("Codex", "Gemini")
    )
    assert remaining == []


# --- evidence_row_ids approved-matrix validation and carry-forward (#1324) ---

def _evidence_review_text(tag):
    import json

    payload = {
        "schema_version": 1, "kind": "pr_review", "state": "blocking", "summary": "s",
        "blocking_items": [{"text": "need citations", "evidence_row_ids": tag}],
        "prior_item_dispositions": [],
    }
    return json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- Codex"


def test_validator_accepts_known_rows_and_rejects_unknown_or_unbacked_tags():
    from coding_review_agent_loop.errors import AgentLoopError
    from coding_review_agent_loop.unresolved_items import _validate_review_response

    kwargs = dict(reviewer="Codex", unresolved_items=(), architecture_status_mode="strict")
    parsed = _validate_review_response(
        _evidence_review_text(["row-a"]), approved_matrix_row_ids=("row-a", "row-b"), **kwargs
    )
    assert parsed.blocking_items[0].evidence_row_ids == ("row-a",)
    with pytest.raises(AgentLoopError, match="unknown row"):
        _validate_review_response(
            _evidence_review_text(["row-z"]), approved_matrix_row_ids=("row-a",), **kwargs
        )
    with pytest.raises(AgentLoopError, match="no applicable risk-test matrix"):
        _validate_review_response(_evidence_review_text(["row-a"]), approved_matrix_row_ids=None, **kwargs)


def test_unresolved_disposition_keeps_the_tag_on_the_same_item():
    from dataclasses import replace as _replace

    from coding_review_agent_loop.protocol import ReviewItemDisposition, UnresolvedReviewItem
    from coding_review_agent_loop.unresolved_items import (
        _apply_unresolved_item_dispositions,
        _next_unresolved_item,
    )

    item = _next_unresolved_item(
        item_number=1, reviewer="Codex", source_round=1, text="need citations",
        status="blocking", evidence_row_ids=("row-a",),
    )
    assert item.evidence_row_ids == ("row-a",)
    machine = _next_unresolved_item(
        item_number=2, reviewer="Orchestrator", source_round=1, text="m", status="blocking",
        obligation_kind="merge-conflict", evidence_row_ids=("row-a",),
        failed_head_sha="a" * 40,
    )
    assert machine.evidence_row_ids == ()
    kept, _future = _apply_unresolved_item_dispositions(
        [item],
        {"item-1": [ReviewItemDisposition(
            item_id="item-1", reviewer="Codex", disposition="blocking", note="still no citation"
        )]},
        round_number=2,
    )
    assert [(entry.item_id, entry.evidence_row_ids) for entry in kept] == [("item-1", ("row-a",))]
