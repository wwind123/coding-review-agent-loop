"""Sub-item contract for conjunctive findings (#958): parse, reconcile, persist."""

import json
from dataclasses import replace

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.orchestrator import (
    _round_limit_diagnostic,
    _sub_item_progress_block,
)
from coding_review_agent_loop.protocol import (
    ReviewItemDisposition,
    ReviewSubItem,
    UnresolvedReviewItem,
)
from coding_review_agent_loop.round_state import (
    _deserialize_disposition,
    _deserialize_unresolved_item,
    _prior_item_ledger_signature,
    _serialize_disposition,
    _serialize_unresolved_item,
)
from coding_review_agent_loop.unresolved_items import (
    ClearedItemProgress,
    _apply_unresolved_item_dispositions,
    _format_same_pr_unresolved_items,
    _format_unresolved_items_for_coder,
    _next_unresolved_item,
    _upsert_machine_obligation,
    _validate_review_response,
    newly_stalled_items,
    render_sub_item_progress_summary,
    sub_item_progress,
)

STATEMENTS = ("wire staged", "wire rebind", "wire managed", "wire approved-plan")


def _item(*, resolved=(), reviewer="Codex", number=1, status="blocking", round_number=1):
    item = _next_unresolved_item(
        item_number=number,
        reviewer=reviewer,
        source_round=round_number,
        text="Four writer paths are unwired.",
        status=status,
        sub_items=STATEMENTS,
    )
    subs = tuple(
        replace(sub, status="resolved", resolved_round=2)
        if sub.sub_item_id in resolved
        else sub
        for sub in item.sub_items
    )
    return replace(item, sub_items=subs)


def _disp(reviewer, disposition, note=None, subs=()):
    return ReviewItemDisposition(
        item_id="item-1",
        reviewer=reviewer,
        disposition=disposition,
        note=note,
        sub_item_dispositions=tuple(subs),
    )


def _reconcile(item, dispositions, *, mode="aggregate", round_number=5):
    cleared: list[ClearedItemProgress] = []
    notes: list[str] = []
    kept, _future = _apply_unresolved_item_dispositions(
        [item],
        {"item-1": list(dispositions)},
        reconciliation_mode=mode,
        round_number=round_number,
        cleared_items_progress=cleared,
        sub_item_degradations=notes,
    )
    return kept, cleared, notes


def _counts(item):
    return sum(1 for sub in item.sub_items if sub.status == "resolved"), len(item.sub_items)


ALL_BUT_LAST = ("item-1.s1", "item-1.s2", "item-1.s3")


def test_minting_assigns_ordered_sub_item_ids():
    item = _item()
    assert [sub.sub_item_id for sub in item.sub_items] == [
        "item-1.s1", "item-1.s2", "item-1.s3", "item-1.s4",
    ]
    assert all(sub.status == "open" and sub.resolved_round is None for sub in item.sub_items)


def test_machine_obligation_cannot_carry_sub_items():
    (machine,) = _upsert_machine_obligation(
        [], item_number=1, kind="managed-exact-head-ci", source_round=1, text="ci", failed_head_sha="a" * 40
    )
    with pytest.raises(ValueError):
        replace(machine, sub_items=(ReviewSubItem("item-1.s1", "x"),))
    kept, cleared, notes = _reconcile(
        machine, [_disp("Codex", "resolved", subs=[("item-1.s1", "resolved")])]
    )
    assert kept and kept[0].status == "blocking" and not cleared
    assert any("machine obligation" in note for note in notes)


def test_owner_resolves_one_sub_item_and_item_stays_open():
    kept, cleared, _ = _reconcile(
        _item(), [_disp("Codex", "blocking", "still wrong", [("item-1.s1", "resolved")])]
    )
    (item,) = kept
    assert _counts(item) == (1, 4)
    assert item.sub_items[0].resolved_round == 5
    assert item.text == "Four writer paths are unwired."
    assert not cleared


def test_literal_resolved_without_map_clears_item_with_true_count():
    kept, cleared, _ = _reconcile(_item(resolved=ALL_BUT_LAST), [_disp("Codex", "resolved")])
    assert kept == []
    (record,) = cleared
    assert (record.resolved, record.total, record.cause) == (3, 4, "item-level-resolved")
    assert record.closed_sub_item_ids == ()


def test_literal_resolved_with_conflicting_reviewer_leaves_sub_items_unchanged():
    item = _item(resolved=ALL_BUT_LAST)
    kept, cleared, _ = _reconcile(
        item,
        [_disp("Codex", "resolved"), _disp("Claude", "blocking", "still broken")],
    )
    assert kept[0].sub_items == item.sub_items and kept[0].status == "blocking"
    assert not cleared


def test_owner_scoped_resolved_shortcut_keeps_item_open_for_pending_owner():
    item = replace(
        _item(resolved=ALL_BUT_LAST),
        resolution_owners=("Codex", "Claude"),
        owner_states=(("Codex", "pending"), ("Claude", "pending")),
    )
    kept, cleared, _ = _reconcile(item, [_disp("Codex", "resolved")], mode="owner-scoped")
    (out,) = kept
    assert dict(out.owner_states) == {"Codex": "cleared", "Claude": "pending"}
    assert out.sub_items == item.sub_items and not cleared


def test_note_less_completing_entry_derives_resolved_and_clears():
    kept, cleared, _ = _reconcile(
        _item(resolved=ALL_BUT_LAST),
        [_disp("Codex", "blocking", subs=[("item-1.s4", "resolved")])],
    )
    assert kept == []
    (record,) = cleared
    assert (record.resolved, record.total) == (4, 4)
    assert record.cause == "all-sub-items-resolved"
    assert record.closed_sub_item_ids == ("item-1.s4",)
    assert record.round_number == 5


def test_completing_closure_deferred_when_another_reviewer_keeps_item_open():
    kept, cleared, notes = _reconcile(
        _item(resolved=ALL_BUT_LAST),
        [
            _disp("Codex", "blocking", subs=[("item-1.s4", "resolved")]),
            _disp("Claude", "blocking", "X"),
        ],
    )
    (out,) = kept
    assert _counts(out) == (3, 4) and not cleared
    assert any("deferred" in note for note in notes)
    explanation = next(note for note in out.notes if note.startswith("Codex:"))
    assert "item-1.s4" in explanation and '"wire approved-plan"' in explanation
    assert 'Claude kept item-level blocking: "X"' in explanation


@pytest.mark.parametrize(
    "other, expected",
    [
        (None, "pending owner Claude has not concurred"),
        (("Gemini", "blocking", "gemini note"), 'Gemini kept item-level blocking: "gemini note"'),
    ],
)
def test_owner_scoped_deferral_keeps_owner_pending_with_evidence(other, expected):
    item = replace(
        _item(resolved=ALL_BUT_LAST),
        resolution_owners=("Codex", "Claude"),
        owner_states=(("Codex", "pending"), ("Claude", "pending")),
    )
    dispositions = [_disp("Codex", "blocking", subs=[("item-1.s4", "resolved")])]
    if other:
        dispositions.append(_disp(*other))
    kept, cleared, _ = _reconcile(item, dispositions, mode="owner-scoped")
    (out,) = kept
    assert _counts(out) == (3, 4) and not cleared
    assert dict(out.owner_states)["Codex"] == "pending"
    assert expected in dict(out.owner_evidence)["Codex"]
    assert expected in " ".join(out.notes)


def test_other_reviewer_reopen_leaves_explained_open_item():
    kept, _cleared, _ = _reconcile(
        _item(resolved=ALL_BUT_LAST),
        [
            _disp("Codex", "blocking", subs=[("item-1.s4", "resolved")]),
            _disp("Claude", "blocking", "regressed", [("item-1.s2", "unresolved")]),
        ],
    )
    (out,) = kept
    assert _counts(out) == (3, 4)
    assert out.sub_items[1].status == "open" and out.sub_items[1].resolved_round is None
    assert out.sub_items[3].status == "resolved"
    assert "Claude reopened it" in " ".join(out.notes)


def test_non_owner_and_unknown_keys_never_change_status():
    item = replace(
        _item(resolved=("item-1.s1",)),
        resolution_owners=("Codex",),
        owner_states=(("Codex", "pending"),),
    )
    kept, _cleared, notes = _reconcile(
        item,
        [
            _disp("Codex", "blocking", "n", [("item-1.s1", "unresolved"), ("item-9.s1", "resolved")]),
            _disp("Gemini", "resolved", subs=[("item-1.s2", "resolved")]),
        ],
        mode="owner-scoped",
    )
    (out,) = kept
    assert out.sub_items[0].status == "open" and out.sub_items[0].resolved_round is None
    assert out.sub_items[1].status == "open"
    assert any("unknown sub-item key" in note for note in notes)
    assert any("only owners" in note for note in notes)


def test_no_item_persists_open_at_full_count_for_any_ordering():
    item = _item(resolved=ALL_BUT_LAST)
    for others in ([], [_disp("Claude", "same-pr", "n")], [_disp("Claude", "blocking", "n")]):
        for mode in ("aggregate", "owner-scoped"):
            kept, _c, _n = _reconcile(
                item,
                [_disp("Codex", "same-pr", subs=[("item-1.s4", "resolved")]), *others],
                mode=mode,
            )
            for out in kept:
                assert _counts(out)[0] < _counts(out)[1]


def test_validation_accepts_note_less_completing_entry_only():
    item = _item(resolved=ALL_BUT_LAST)
    completing = {
        "schema_version": 1, "kind": "pr_review", "state": "blocking", "summary": "s",
        "prior_item_dispositions": [
            {"item_id": "item-1", "disposition": "blocking", "sub_item_dispositions": {"item-1.s4": "resolved"}}
        ],
        "blocking_items": ["new problem"],
    }
    text = json.dumps(completing) + "\n<!-- AGENT_STATE: blocking -->\n-- Codex"
    parsed = _validate_review_response(
        text, reviewer="Codex", unresolved_items=[item], architecture_status_mode="degradable"
    )
    assert parsed.dispositions[0].sub_item_dispositions == (("item-1.s4", "resolved"),)

    leaving_open = json.loads(json.dumps(completing))
    leaving_open["prior_item_dispositions"][0]["sub_item_dispositions"] = {"item-1.s3": "unresolved"}
    with pytest.raises(AgentLoopError, match="actionable note"):
        _validate_review_response(
            json.dumps(leaving_open) + "\n<!-- AGENT_STATE: blocking -->\n-- Codex",
            reviewer="Codex", unresolved_items=[item], architecture_status_mode="degradable",
        )
    partial = _item(resolved=("item-1.s1",))
    with pytest.raises(AgentLoopError, match="actionable note"):
        _validate_review_response(
            text, reviewer="Codex", unresolved_items=[partial], architecture_status_mode="degradable"
        )


def test_validation_drops_unknown_sub_item_keys_with_degradation():
    plain = _next_unresolved_item(
        item_number=1, reviewer="Codex", source_round=1, text="t", status="blocking"
    )
    payload = {
        "schema_version": 1, "kind": "pr_review", "state": "blocking", "summary": "s",
        "prior_item_dispositions": [
            {"item_id": "item-1", "disposition": "blocking", "note": "still broken here",
             "sub_item_dispositions": {"item-1.s1": "resolved"}}
        ],
    }
    parsed = _validate_review_response(
        json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- Codex",
        reviewer="Codex", unresolved_items=[plain], architecture_status_mode="degradable",
    )
    assert parsed.dispositions[0].sub_item_dispositions == ()
    assert parsed.sub_item_degradations


def test_legacy_records_round_trip_byte_identically():
    legacy = _next_unresolved_item(
        item_number=1, reviewer="Codex", source_round=1, text="t", status="blocking"
    )
    payload = _serialize_unresolved_item(legacy)
    assert "sub_items" not in payload
    assert _serialize_unresolved_item(_deserialize_unresolved_item(payload)) == payload
    disposition = ReviewItemDisposition("item-1", "Codex", "resolved")
    encoded = _serialize_disposition(disposition)
    assert "sub_item_dispositions" not in encoded
    assert _deserialize_disposition(encoded) == disposition
    signature = _prior_item_ledger_signature([legacy])[0]
    assert len(signature) == 18


def test_sub_item_state_round_trips_and_extends_signature():
    item = _item(resolved=("item-1.s1", "item-1.s2"))
    restored = _deserialize_unresolved_item(_serialize_unresolved_item(item))
    assert restored == item
    disposition = _disp("Codex", "blocking", "n", [("item-1.s3", "resolved")])
    assert _deserialize_disposition(_serialize_disposition(disposition)) == disposition
    signature = _prior_item_ledger_signature([item])[0]
    assert len(signature) == 19
    assert _prior_item_ledger_signature([_item()]) != _prior_item_ledger_signature([item])


def test_malformed_persisted_sub_items_leave_item_open_without_sub_items():
    payload = _serialize_unresolved_item(_item(resolved=("item-1.s1",)))
    payload["sub_items"] = [{"sub_item_id": "item-1.s1", "text": "x", "status": "bogus"}]
    restored = _deserialize_unresolved_item(payload)
    assert restored.sub_items == () and restored.status == "blocking"
    payload["sub_items"] = "corrupt"
    assert _deserialize_unresolved_item(payload).sub_items == ()


def test_progress_classification_and_stall_window():
    converging = _item(resolved=("item-1.s1",), round_number=1)  # closed in round 2
    (entry,) = sub_item_progress([converging], current_round=4, window=3)
    assert entry.classification == "converging" and entry.closed_in_window == 1
    (entry,) = sub_item_progress([converging], current_round=5, window=3)
    assert entry.classification == "stalled"
    fresh = _item(round_number=1)
    assert sub_item_progress([fresh], current_round=2, window=3)[0].classification == "new"
    assert sub_item_progress([fresh], current_round=3, window=3)[0].classification == "stalled"
    assert sub_item_progress([fresh], current_round=9, window=0)[0].classification == "new"
    complete = _item(resolved=tuple(s.sub_item_id for s in fresh.sub_items))
    assert sub_item_progress([complete], current_round=99, window=3)[0].classification == "complete"
    cleared = ClearedItemProgress("item-2", 4, 4, "all-sub-items-resolved", 5)
    assert sub_item_progress([], [cleared], current_round=5, window=3)[0].classification == "cleared"


def test_newly_stalled_fires_once():
    fresh = _item(round_number=1)
    assert [e.item_id for e in newly_stalled_items([fresh], current_round=3, window=3)] == ["item-1"]
    assert newly_stalled_items([fresh], current_round=4, window=3) == ()
    assert newly_stalled_items([fresh], current_round=2, window=3) == ()
    assert newly_stalled_items([fresh], current_round=3, window=0) == ()
    complete = _item(resolved=tuple(s.sub_item_id for s in fresh.sub_items))
    assert newly_stalled_items([complete], current_round=3, window=3) == ()


def test_progress_summary_and_diagnostic_block_are_byte_identical_without_sub_items():
    plain = _next_unresolved_item(
        item_number=1, reviewer="Codex", source_round=1, text="t", status="blocking"
    )
    assert _sub_item_progress_block([plain], round_number=3, window=3) == ""
    base = _round_limit_diagnostic(pr_number=7, round_number=3, items=[plain], current_head_sha="h")
    assert (
        _round_limit_diagnostic(
            pr_number=7, round_number=3, items=[plain], current_head_sha="h", sub_item_stall_rounds=3
        )
        == base
    )
    converging = _item(resolved=("item-1.s1",), round_number=1)
    message = _round_limit_diagnostic(
        pr_number=7, round_number=3, items=[converging], current_head_sha="h", sub_item_stall_rounds=3
    )
    assert message.startswith(base.split(". The named")[0].split("reviewer-owned")[0])
    assert "Sub-item progress:" in message and "item-1: 1/4 sub-items resolved" in message
    assert "(converging)" in message
    lines = render_sub_item_progress_summary(sub_item_progress([converging], current_round=3, window=3))
    assert lines == (
        "item-1: 1/4 sub-items resolved, 1 closed in last 3 rounds (converging); consider raising --max-rounds",
    )


def test_coder_and_same_pr_formatters_list_sub_items():
    item = _item(resolved=("item-1.s1",))
    for rendered in (
        _format_unresolved_items_for_coder([item]),
        _format_same_pr_unresolved_items([replace(item, status="same-pr")]),
    ):
        assert "Sub-items (1/4 resolved):" in rendered
        assert "[item-1.s1] resolved (round 2): wire staged" in rendered
        assert "[item-1.s2] open: wire rebind" in rendered
    plain = _next_unresolved_item(
        item_number=1, reviewer="Codex", source_round=1, text="t", status="blocking"
    )
    assert "Sub-items" not in _format_unresolved_items_for_coder([plain])


def test_window_of_one_never_stalls_a_finding_minted_this_round():
    fresh = _item(round_number=4)
    assert sub_item_progress([fresh], current_round=4, window=1)[0].classification == "new"
    assert newly_stalled_items([fresh], current_round=4, window=1) == ()
    assert [e.item_id for e in newly_stalled_items([fresh], current_round=5, window=1)] == ["item-1"]
    assert newly_stalled_items([fresh], current_round=6, window=1) == ()


def test_corrupt_persisted_disposition_pairs_are_dropped_not_fatal():
    payload = _serialize_disposition(_disp("Codex", "blocking", "n", [("item-1.s1", "resolved")]))
    payload["sub_item_dispositions"] = [
        ["item-1.s1", []], ["item-1.s2", {"x": 1}], ["item-1.s3", "resolved"], [3, "resolved"], "junk",
    ]
    restored = _deserialize_disposition(payload)
    assert restored.sub_item_dispositions == (("item-1.s3", "resolved"),)


@pytest.mark.parametrize("retain_future", [True, False])
def test_future_reclassification_is_not_published_as_cleared(retain_future):
    item = _item(resolved=ALL_BUT_LAST)
    cleared: list[ClearedItemProgress] = []
    kept, future = _apply_unresolved_item_dispositions(
        [item],
        {"item-1": [_disp("Codex", "future", "later")]},
        retain_future=retain_future,
        round_number=5,
        cleared_items_progress=cleared,
    )
    assert cleared == []
    (out,) = kept if retain_future else future
    assert out.status == "future" and out.sub_items == item.sub_items
    assert sub_item_progress(kept, current_round=99, window=3) == ()
    assert newly_stalled_items(kept, current_round=99, window=3) == ()


def test_deferral_reason_is_persisted_on_the_item_notes_even_with_a_reviewer_note():
    item = _item(resolved=ALL_BUT_LAST)
    kept, _cleared, _notes = _reconcile(
        item,
        [
            _disp("Codex", "blocking", "wrapped up s4", [("item-1.s4", "resolved")]),
            _disp("Claude", "blocking", "X"),
        ],
    )
    (out,) = kept
    assert _counts(out) == (3, 4)
    note = next(n for n in out.notes if n.startswith("Orchestrator: deferred closing item-1.s4"))
    assert '"wire approved-plan"' in note and "item-1" in note
    restored = _deserialize_unresolved_item(_serialize_unresolved_item(out))
    assert note in restored.notes
    # Re-running the same reconciliation never duplicates the note.
    again, _c, _n = _reconcile(out, [_disp("Claude", "blocking", "X")])
    assert sum(1 for n in again[0].notes if n.startswith("Orchestrator: deferred")) <= 1


def test_note_less_completing_non_owner_added_this_round_gets_owner_evidence():
    item = replace(
        _item(resolved=ALL_BUT_LAST),
        resolution_owners=("Codex",),
        owner_states=(("Codex", "pending"),),
    )
    payload = {
        "schema_version": 1, "kind": "pr_review", "state": "blocking", "summary": "s",
        "prior_item_dispositions": [
            {"item_id": "item-1", "disposition": "blocking",
             "sub_item_dispositions": {"item-1.s4": "resolved"}}
        ],
    }
    # Live validation accepts the note-less completing entry ...
    parsed = _validate_review_response(
        json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- Gemini",
        reviewer="Gemini", unresolved_items=[item], architecture_status_mode="degradable",
    )
    (entry,) = parsed.dispositions
    # ... and owner-scoped reconciliation defers it with an explanation.
    kept, cleared, _notes = _reconcile(
        item, [_disp("Codex", "blocking", "still broken here"), entry], mode="owner-scoped"
    )
    (out,) = kept
    assert _counts(out) == (3, 4) and not cleared
    evidence = dict(out.owner_evidence)["Gemini"]
    assert "item-1.s4" in evidence and '"wire approved-plan"' in evidence
    assert 'Codex kept item-level blocking: "still broken here"' in evidence
    assert dict(out.owner_states)["Gemini"] == "pending"
