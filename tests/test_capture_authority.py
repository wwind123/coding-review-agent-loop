"""The same fail-closed scope boundary at decoded and raw authority consumers."""
import dataclasses
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.round_state import (
    PostedRoundMetadata, PostedRoundRecord, _encode_round_metadata,
    _decode_round_metadata, _aggregate_record_dispositions,
    _pr_round_is_reconciled, _latest_pr_approved_reviews_for_head, _attach_round_metadata,
)
from coding_review_agent_loop.review_scope import is_complete_code_review, valid_evidence_scope
from coding_review_agent_loop.panel_evidence import _derive_pr_panel_evidence, _pr_record_is_panel_qualified
from coding_review_agent_loop.review_step_back import pr_reviews_for, effective_new_items
from coding_review_agent_loop.managed_ci import _continuity_round_records

SCOPE = {"version": 1, "kind": "capture-evidence", "capture": "capture", "head": "head",
         "matrix": "matrix", "rows": ["row-a"], "contract": "contract"}


def record(index=0, **fields):
    defaults = dict(flow="pr", role="reviewer", agent="Codex", round_number=1,
                    subject="head", state="approved")
    return PostedRoundRecord(index=index, metadata=PostedRoundMetadata(**(defaults | fields)), body="review")


@pytest.mark.parametrize("scope", [None, {}, "unknown", {"kind": "future"}, SCOPE])
def test_scope_roundtrip_never_becomes_complete_review(scope):
    original = record(review_scope=scope, review_scope_present=True).metadata
    decoded = _decode_round_metadata(_encode_round_metadata(original))
    assert decoded.review_scope_present
    assert decoded.review_scope == scope
    assert not is_complete_code_review(decoded)
    assert not is_complete_code_review({"role": "reviewer", "review_scope": scope})
    assert _encode_round_metadata(decoded) == _encode_round_metadata(original)
    ordinary = record().metadata
    assert is_complete_code_review(_decode_round_metadata(_encode_round_metadata(ordinary)))
    assert is_complete_code_review({"role": "reviewer"})


@pytest.mark.parametrize("scope", [None, {}, "unknown", SCOPE])
def test_scoped_primary_cannot_open_panel_or_qualify_sweep(scope):
    scoped = record(review_scope=scope, review_scope_present=True)
    scheduler = record(1, role="summary", agent="Orchestrator", scheduler_metadata_status="valid",
                       scheduler_phase="secondary-audit", scheduler_approved_reviewers=("Codex",))
    evidence = _derive_pr_panel_evidence((scoped, scheduler), primary_reviewer="Codex",
                                         required_reviewers=("Codex", "Gemini"))
    assert not evidence.opened
    assert not _pr_record_is_panel_qualified(scoped, evidence, primary_reviewer="Codex", operator_force_full=False)
    secondary = record(2, agent="Gemini", scheduler_final_sweep=True,
                       review_scope=scope, review_scope_present=True)
    assert not _pr_record_is_panel_qualified(secondary, evidence, primary_reviewer="Codex", operator_force_full=False)
    ordinary = record()
    positive = _derive_pr_panel_evidence((ordinary, scheduler), primary_reviewer="Codex",
                                         required_reviewers=("Codex", "Gemini"))
    assert positive.opened
    assert _pr_record_is_panel_qualified(record(2, agent="Gemini"), positive,
                                        primary_reviewer="Codex", operator_force_full=False)


@pytest.mark.parametrize("state", ["approved", "blocking"])
def test_scoped_rounds_cannot_replace_code_review_or_convergence_history(state):
    complete = record()
    scoped = record(1, round_number=2, state=state, review_scope=SCOPE)
    mixed = record(2, round_number=1, state=state, review_scope=SCOPE)
    assert [r.index for r in pr_reviews_for((complete, scoped, mixed), "Codex")] == [0]
    summary = record(3, role="summary", agent="Orchestrator", round_number=2,
                     phase="reconciliation", review_scope={"kind": "capture-reconciliation"})
    assert not _pr_round_is_reconciled((scoped, summary))
    assert _pr_round_is_reconciled((record(3, role="summary", phase="reconciliation"),))
    comments = [SimpleNamespace(body=_attach_round_metadata(r.body, r.metadata))
                for r in (complete, scoped, mixed)]
    approvals = _latest_pr_approved_reviews_for_head(comments, head_sha="head", configured_reviewers=("codex",))
    assert approvals["Codex"].index == 0


def test_raw_managed_recovery_retains_null_and_unknown_scope():
    for scope in [None, {}, "unknown", SCOPE]:
        m = record(review_scope=scope, review_scope_present=True).metadata
        [raw] = _continuity_round_records([{"body": _attach_round_metadata("review", m)}]).values()
        assert "review_scope" in raw and raw["review_scope"] == scope
        assert not is_complete_code_review(raw)
    [raw] = _continuity_round_records([{"body": _attach_round_metadata("review", record().metadata)}]).values()
    assert is_complete_code_review(raw)


def test_scoped_dispositions_and_summaries_cannot_clear_unrelated_findings():
    from coding_review_agent_loop.protocol import ReviewItemDisposition, UnresolvedReviewItem
    item = UnresolvedReviewItem(item_id="item-1", reviewer="Codex", source_round=1,
                                text="Unrelated code defect", status="blocking", source_status="blocking")
    disposition = ReviewItemDisposition(item_id="item-1", reviewer="Codex", disposition="resolved", note="done")
    scoped = record(review_scope=SCOPE, prior_items=(item,), dispositions=(disposition,))
    summary = record(1, role="summary", review_scope={"kind": "capture-reconciliation"}, dispositions=(disposition,))
    assert _aggregate_record_dispositions((scoped, summary)) == {}
    ordinary = record(2, prior_items=(item,), dispositions=(disposition,))
    assert _aggregate_record_dispositions((scoped, summary, ordinary)) == {"item-1": [disposition]}


def test_scoped_poster_does_not_infer_legacy_board_participation(tmp_path):
    from agent_loop_helpers import make_config
    from coding_review_agent_loop.board_change import derive_legacy_root, identity_from_config
    configured = identity_from_config(make_config(tmp_path, reviewer="codex"))
    scoped = record(agent="Gemini", review_scope=SCOPE)
    root = derive_legacy_root((scoped,), configured=configured, amendments=(), contract_from_metadata=lambda m: None)
    assert root.names == ("Codex",)
    root = derive_legacy_root((record(agent="Gemini"),), configured=configured, amendments=(), contract_from_metadata=lambda m: None)
    assert set(root.names) == {"Codex", "Gemini"}


def test_bound_capture_reconciliation_preserves_other_owners_and_machine_items():
    from coding_review_agent_loop.capture_reconciliation import reconcile_capture_record
    from coding_review_agent_loop.capture_acceptance import reviewer_contract
    from coding_review_agent_loop.protocol import ReviewItemDisposition, UnresolvedReviewItem
    def item(item_id, reviewer, **kwargs):
        return UnresolvedReviewItem(item_id=item_id, reviewer=reviewer, source_round=1,
            text="Evidence coverage", status="blocking", source_status="blocking", **kwargs)
    own = item("item-1", "Codex", evidence_row_ids=("row-a",))
    other = item("item-2", "Gemini", evidence_row_ids=("row-a",))
    machine = item("item-3", "Orchestrator", authority="machine", obligation_kind="human-exact-head-evidence",
                   lifecycle="evidence_deferred", obligation_identity="human-evidence")
    dispositions = tuple(ReviewItemDisposition(item_id=i.item_id, reviewer="Codex", disposition="resolved", note="accepted")
                         for i in (own, other, machine))
    metadata = record(prior_items=(own, other, machine), dispositions=dispositions).metadata
    scope = dict(SCOPE, contract=reviewer_contract(metadata))
    metadata = dataclasses.replace(metadata, review_scope=scope)
    binding = {key: scope[key] for key in ("capture", "head", "matrix")}
    assert reconcile_capture_record((own, other, machine), metadata, binding=binding) == (other, machine)
    assert reconcile_capture_record((own, other, machine), metadata, binding=dict(binding, capture="replacement")) == (own, other, machine)


def test_scoped_records_cannot_establish_recovery_coherence_or_latches():
    from coding_review_agent_loop.round_state import _latest_prior_pr_subject_is_coherent, startup_force_full_latch
    scoped = record(review_scope=SCOPE, scheduler_force_full=True, scheduler_force_full_source="operator")
    assert not _latest_prior_pr_subject_is_coherent((scoped,), head_sha="4ebf8a1679f50e9e49fc965072f996bfe0dc18c6")
    assert startup_force_full_latch((scoped,), requires_primary=False) == (False, False)
    ordinary = record(scheduler_force_full=True, scheduler_force_full_source="operator")
    assert _latest_prior_pr_subject_is_coherent((ordinary,), head_sha="4ebf8a1679f50e9e49fc965072f996bfe0dc18c6")
    assert startup_force_full_latch((ordinary,), requires_primary=False) == (True, False)
