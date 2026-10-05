"""Unit tests for the bounded finding history and generalization guidance (#1273)."""

from __future__ import annotations

import json

import pytest

from agent_loop_helpers import make_config
from coding_review_agent_loop import finding_history as fh
from coding_review_agent_loop.finding_history import (
    FINDING_HISTORY_MAX_CHARS,
    FINDING_HISTORY_MAX_ROUNDS,
    FindingHistoryLedger,
    detect_generalization,
    log_declared_generalization,
    render_finding_history_block,
    render_generalization_guidance,
)
from coding_review_agent_loop.protocol import ReviewItemDisposition, ReviewSubItem
from coding_review_agent_loop.prompts import (
    build_followup_prompt,
    build_plan_revision_prompt,
    build_same_pr_followup_prompt,
)
from coding_review_agent_loop.review_step_back import PlanStepBackContext
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    PostedRoundRecord,
    UnresolvedReviewItem,
    _canonically_resolved_history_item_ids,
    canonical_history_item_outcomes,
)
from coding_review_agent_loop.unresolved_items import (
    _clear_machine_obligations,
    _upsert_machine_obligation,
)


def _item(item_id, status="blocking", *, reviewer="Codex", round_number=1, text=None, **kw):
    return UnresolvedReviewItem(
        item_id=item_id,
        reviewer=reviewer,
        source_round=round_number,
        text=text or f"finding {item_id} in src/spool.py:{10 + round_number}",
        status=status,
        **kw,
    )


def _ci_item(head, *, identity="ci-1", round_number=2, check="unit-tests", kind="github-pr-checks"):
    return UnresolvedReviewItem(
        item_id="ci-item",
        reviewer="Orchestrator",
        source_round=round_number,
        text=f"Failing checks\nReviewed head: {head}\n- {check}: failure (https://x/y)\nadvisory",
        status="blocking",
        authority="machine",
        obligation_kind=kind,
        lifecycle="repair_required",
        failed_head_sha=head,
        obligation_identity=identity,
    )


def _coder_fix(round_number, summary, notes=None):
    return json.dumps(
        {
            "summary": summary,
            "addressed_items": list(notes or {}),
            "addressed_item_notes": notes or {},
        }
    )


def _ledger(phase="pr"):
    logs: list[str] = []
    return FindingHistoryLedger(phase, log=logs.append), logs


def _view(ledger, round_number):
    return render_finding_history_block(ledger.view(round_number))


# --- cutoff / status ---------------------------------------------------------


def test_prior_fix_and_resolved_finding_visible_but_current_round_finding_excluded():
    ledger, _ = _ledger()
    first = _item("item-1", round_number=1)
    ledger.observe_reconciled([first])
    # Round 2: item-1 cleared, sibling raised in round 2.
    sibling = _item("item-2", round_number=2, text="sibling at src/spool.py:99")
    ledger.observe_reconciled([sibling])
    ledger.add_fix(
        fh.fix_from_payload(
            json.loads(_coder_fix(2, "Fixed undecodable spool", {"item-1": "guarded decode"})),
            published_round=2,
            agent="Claude",
        )
    )
    text = _view(ledger, 2)
    assert "Codex finding item-1 (resolved) src/spool.py:11" in text
    assert "fix by Claude: Fixed undecodable spool" in text
    assert "item-1: guarded decode" in text
    assert "item-2" not in text and "src/spool.py:99" not in text


def test_status_comes_from_live_ledger_open_subitems_and_deferral():
    ledger, _ = _ledger()
    with_subs = _item(
        "item-1",
        sub_items=(ReviewSubItem("a", "x"), ReviewSubItem("b", "y", status="resolved")),
    )
    other = _item("item-2")
    ledger.observe_reconciled([with_subs, other])
    ledger.observe_reconciled([with_subs], [_item("item-2", "future")])
    text = _view(ledger, 5)
    assert "Codex finding item-1 (open, 1 open sub-item(s))" in text
    assert "Codex finding item-2 (deferred)" in text
    # Reintroduction as a mandatory item flips deferred back to open.
    ledger.observe_reconciled([with_subs, other])
    assert "Codex finding item-2 (open)" in _view(ledger, 5)


def test_empty_and_unavailable_states_are_distinct_and_guidance_is_independent():
    ledger, logs = _ledger()
    assert "no earlier-round findings or fixes" in _view(ledger, 1)
    ledger._fail(RuntimeError("boom"))
    text = _view(ledger, 1)
    assert "Earlier-round history is unavailable this turn (RuntimeError: boom)" in text
    assert logs and logs[0].startswith("finding history unavailable:")
    assert "Generalization: this generalizes the fix for" in render_generalization_guidance("pr")
    assert "future follow-up" in render_generalization_guidance("plan")


def test_projection_exception_marks_history_unavailable_not_raise(monkeypatch):
    ledger, logs = _ledger()
    monkeypatch.setattr(fh, "project_finding", lambda item: (_ for _ in ()).throw(RuntimeError("x")))
    ledger.observe_reconciled([_item("item-1")])
    assert ledger.view(2).state == fh.VIEW_UNAVAILABLE
    assert any("finding history unavailable" in line for line in logs)


# --- CI instances ------------------------------------------------------------


def test_ci_instances_keyed_by_identity_and_failed_head():
    ledger, _ = _ledger()
    a, b = "a" * 40, "b" * 40
    ledger.observe_reconciled([_ci_item(a, round_number=2, check="check-x")])
    ledger.observe_reconciled([_ci_item(a, round_number=3, check="check-x")])  # repeat of A
    ledger.observe_reconciled([_ci_item(b, round_number=4, check="check-y")])
    text = _view(ledger, 9)
    assert text.count(f"CI github-pr-checks on {a[:12]}") == 1
    assert f"superseded by the failure on {b[:12]}" in text
    assert f"CI github-pr-checks on {b[:12]} (open)" in text
    assert "- check-x: failure" in text and "- check-y: failure" in text
    assert "Failing checks" not in text
    # A reviewer resolved vote cannot clear it; only absence from the ledger does.
    ledger.observe_reconciled([])
    assert f"CI github-pr-checks on {b[:12]} (resolved)" in _view(ledger, 9)


def test_legacy_ci_snapshot_without_failed_head_uses_fallback_key():
    ledger, _ = _ledger()
    legacy = UnresolvedReviewItem(
        item_id="ci-item", reviewer="Orchestrator", source_round=1, text="legacy\n- x: failure",
        status="blocking", authority="machine", obligation_kind="github-pr-checks",
        lifecycle="cleared",
    )
    ledger.add_finding(legacy)
    ledger.add_finding(legacy)
    assert len(ledger._findings) == 1


# --- bound -------------------------------------------------------------------


def test_history_keeps_newest_rounds_and_reports_omitted():
    ledger, _ = _ledger()
    for n in range(1, 10):
        ledger.add_finding(_item(f"item-{n}", round_number=n))
    text = _view(ledger, 20)
    assert text.count("\nRound ") == FINDING_HISTORY_MAX_ROUNDS
    assert "Round 9:" in text and "Round 4:" in text and "Round 3:" not in text
    assert "(3 earlier round(s) omitted)" in text


def test_history_respects_char_cap_by_dropping_whole_oldest_rounds():
    ledger, _ = _ledger()
    for n in range(1, 7):
        for k in range(6):
            ledger.add_finding(
                _item(f"item-{n}-{k}", round_number=n, text="long " * 100 + f" src/a.py:{n}")
            )
    text = _view(ledger, 20)
    assert len(text) <= FINDING_HISTORY_MAX_CHARS
    assert "earlier round(s) omitted" in text
    for line in text.splitlines():
        assert len(line) <= 400
    assert "Round 6:" in text


def test_single_oversize_round_is_omitted_whole_and_reported():
    ledger, _ = _ledger()
    for k in range(80):
        ledger.add_finding(_item(f"item-{k}", round_number=2, text="y" * 500))
    ledger.add_finding(_item("item-new", round_number=1))
    text = _view(ledger, 9)
    assert len(text) <= FINDING_HISTORY_MAX_CHARS
    assert "(2 earlier round(s) omitted)" in text
    assert "Round 2:" not in text


def test_one_large_fix_response_stays_under_the_cap():
    ledger, _ = _ledger()
    notes = {f"item-{n}": "n" * 400 for n in range(60)}
    ledger.add_fix(
        fh.fix_from_payload(
            json.loads(_coder_fix(2, "big " * 200, notes)), published_round=2, agent="Claude"
        )
    )
    ledger.add_finding(_item("item-old", round_number=1))
    text = _view(ledger, 3)
    assert len(text) <= FINDING_HISTORY_MAX_CHARS
    assert text.count("\n    item-") <= fh.FINDING_HISTORY_MAX_FIX_ITEMS


def test_location_labels_from_findings_and_subitems_are_neutralized():
    from coding_review_agent_loop.protocol_markers import RESERVED_MARKER_REGISTRY

    token = RESERVED_MARKER_REGISTRY[0].token
    ledger, _ = _ledger()
    ledger.add_finding(_item("item-1", text=f"bug in src/{token}.py:12"))
    ledger.add_finding(
        _item(
            "item-2",
            text="no loc",
            sub_items=(ReviewSubItem("a", f"see lib/{token}.py:7"),),
        )
    )
    text = _view(ledger, 5)
    assert token not in text
    assert "src/" in text and "lib/" in text


# --- sources -----------------------------------------------------------------


def _record(index, role, round_number, **kw):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr", role=role, agent=kw.pop("agent", "Codex"),
            round_number=round_number, subject=kw.pop("subject", "head"), **kw,
        ),
        body="",
    )


def test_review_parallel_reconciliation_items_appear_in_history():
    publication = _record(0, "reviewer", 1, phase="publication", state="blocking", new_items=())
    reconciliation = _record(
        1, "summary", 1, agent="Orchestrator", phase="reconciliation",
        new_items=(_item("item-1", round_number=1),),
    )
    ledger, _ = _ledger()
    ledger.seed([publication, reconciliation])
    assert "Codex finding item-1" in _view(ledger, 3)


def test_malformed_fix_payload_is_skipped_while_rest_renders():
    good = _record(
        1, "coder", 2, agent="Claude",
        raw_structured_coder_response=_coder_fix(2, "good fix"),
    )
    bad = _record(2, "coder", 3, agent="Claude", raw_structured_coder_response="{not json")
    review = _record(0, "reviewer", 1, state="blocking", new_items=(_item("item-1"),))
    ledger, logs = _ledger()
    ledger.seed([review, good, bad])
    text = _view(ledger, 4)
    assert "good fix" in text and "item-1" in text
    assert any("skipped an undecodable fix payload" in line for line in logs)
    assert ledger.unavailable_reason is None


def test_planner_fix_projects_resolved_dispositions_with_notes():
    class Parsed:
        summary = "Revised plan"
        prior_plan_item_dispositions = (
            ReviewItemDisposition("item-1", "Codex", "resolved", "covered by row X"),
            ReviewItemDisposition("item-2", "Codex", "blocking", "nope"),
        )

    ledger, _ = _ledger("plan")
    ledger.record_fix(Parsed(), published_round=2, agent="Claude")
    text = _view(ledger, 2)
    assert "item-1: covered by row X" in text and "item-2" not in text


# --- deferred replay ---------------------------------------------------------


def _future_disposition(item_id):
    return ReviewItemDisposition(item_id, "Codex", "future", "later")


def test_replay_reads_deferral_from_second_collection_and_stays_deferred_until_reintroduced():
    item = _item("item-1")
    review = _record(
        0, "reviewer", 2, subject="h1", state="blocking",
        dispositions=(_future_disposition("item-1"),),
    )
    review_carry = _record(0, "reviewer", 1, subject="h0", state="blocking", new_items=(item,))
    coder1 = _record(1, "coder", 3, agent="Claude", subject="h2", prior_items=())
    coder2 = _record(2, "coder", 4, agent="Claude", subject="h3", prior_items=())
    records = [review_carry, review, coder1, coder2]
    outcomes = canonical_history_item_outcomes(
        records, reconciliation_mode="aggregate", same_status="same-pr"
    )
    assert outcomes == {"item-1": "deferred"}
    ledger, _ = _ledger()
    ledger.seed(records, outcomes=outcomes)
    ledger.observe_reconciled([])  # item gone from the live ledger
    assert "Codex finding item-1 (deferred)" in _view(ledger, 6)
    reintroduced = _record(3, "coder", 5, agent="Claude", subject="h4", prior_items=(item,))
    again = canonical_history_item_outcomes(
        [*records, reintroduced], reconciliation_mode="aggregate", same_status="same-pr"
    )
    assert again == {"item-1": "active"}


def test_replay_carries_kept_state_and_resolves_only_when_all_dispositions_resolved():
    item = _item("item-1")
    other = _item("item-2")
    resolving = _record(
        1, "reviewer", 2, subject="h1", state="blocking",
        dispositions=(
            ReviewItemDisposition("item-1", "Codex", "resolved"),
            ReviewItemDisposition("item-2", "Codex", "blocking", "still"),
        ),
    )
    seed = _record(0, "reviewer", 1, subject="h0", state="blocking", new_items=(item, other))
    outcomes = canonical_history_item_outcomes(
        [seed, resolving], reconciliation_mode="aggregate", same_status="same-pr"
    )
    assert outcomes == {"item-1": "resolved", "item-2": "active"}
    # Existing replay is untouched by the new one.
    assert _canonically_resolved_history_item_ids(
        [seed, resolving], reconciliation_mode="aggregate", same_status="same-pr"
    ) == frozenset({"item-1"})


def test_replay_skips_machine_obligations():
    seed = _record(0, "coder", 1, agent="Claude", subject="h0", prior_items=(_ci_item("c" * 40),))
    assert canonical_history_item_outcomes(
        [seed], reconciliation_mode="aggregate", same_status="same-pr"
    ) == {}


def test_replay_return_shape_for_future_disposition_and_already_future_carried():
    from coding_review_agent_loop.unresolved_items import _apply_unresolved_item_dispositions

    kept, future = _apply_unresolved_item_dispositions(
        [_item("item-1"), _item("item-2", "future")],
        {"item-1": [_future_disposition("item-1")], "item-2": [ReviewItemDisposition("item-2", "Codex", "future")]},
        retain_future=False,
    )
    assert kept == []
    assert {i.item_id for i in future} == {"item-1", "item-2"}


# --- detection ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Generalization: this generalizes the fix for [item-1]: all spool errors", True),
        ("Fixed it. Generalization: any malformed record", True),
        ("- generalization: x", True),
        ("This generalizes the fix for item-2 in the same function", True),
        ("Fixed the thing.", False),
        ("The generalization of this approach is discussed elsewhere", False),
        ("", False),
        (None, False),
    ],
)
def test_detect_generalization(text, expected):
    assert (detect_generalization(text) is not None) is expected


def test_detect_generalization_excerpt_is_bounded():
    assert len(detect_generalization("Generalization: " + "z" * 1000)) <= 240


def test_log_declared_generalization_tags_and_silence():
    class Parsed:
        summary = "Generalization: this generalizes the fix for [item-1]: rule"
        addressed_item_notes = {}

    logs: list[str] = []
    log_declared_generalization(Parsed(), log=logs.append, round_number=3, agent="Claude", step_back_directed=False)
    log_declared_generalization(Parsed(), log=logs.append, round_number=3, agent="Claude", step_back_directed=True)
    assert "(proactive)" in logs[0] and "(step-back-directed)" in logs[1]

    class Plain:
        summary = "Nothing special"
        addressed_item_notes = {"item-1": "done"}

    log_declared_generalization(Plain(), log=logs.append, round_number=3, agent="Claude", step_back_directed=False)
    assert len(logs) == 2

    class InNote:
        summary = "x"
        addressed_item_notes = {"item-1": "Generalization: this generalizes the fix for [item-0]: r"}

    log_declared_generalization(InNote(), log=logs.append, round_number=3, agent="Claude", step_back_directed=False)
    assert len(logs) == 3


# --- prompts -----------------------------------------------------------------


def _history_view():
    ledger, _ = _ledger()
    ledger.observe_reconciled([_item("item-1")])
    ledger.observe_reconciled([])
    return ledger.view(3)


def test_coder_prompts_carry_guidance_and_history_before_step_back(tmp_path):
    config = make_config(tmp_path)
    view = _history_view()
    for builder in (build_followup_prompt, build_same_pr_followup_prompt):
        prompt = builder(
            7, 3, "review body", config,
            step_back_context="STEP-BACK TURN: do the thing\n",
            generalization_guidance=True, finding_history=view,
        )
        assert "Proactive generalization" in prompt
        assert "Codex finding item-1 (resolved) src/spool.py:11" in prompt
        assert prompt.index("Proactive generalization") < prompt.index("STEP-BACK TURN")
        assert "small, localized cleanup" not in prompt
        unchanged = builder(7, 3, "review body", config)
        assert "Proactive generalization" not in unchanged


def test_same_pr_prompt_keeps_small_cleanup_framing_without_step_back(tmp_path):
    prompt = build_same_pr_followup_prompt(
        7, 3, "r", make_config(tmp_path), generalization_guidance=True, finding_history=_history_view()
    )
    assert "Proactive generalization" in prompt
    assert prompt.index("Proactive generalization") < prompt.index("small, localized cleanup")


def test_guidance_stays_when_history_unavailable(tmp_path):
    ledger, _ = _ledger()
    ledger._fail(RuntimeError("x"))
    prompt = build_followup_prompt(
        7, 3, "r", make_config(tmp_path), generalization_guidance=True,
        finding_history=ledger.view(3),
    )
    assert "Proactive generalization" in prompt and "unavailable this turn" in prompt


def test_planner_prompts_default_unchanged_and_carry_guidance_in_each_branch(tmp_path):
    config = make_config(tmp_path)
    args = (56, 3, "previous plan", "the review", config)
    view = _history_view()
    for kwargs in ({}, {"compact_context": True}):
        base = build_plan_revision_prompt(*args, **kwargs)
        assert "Proactive generalization" not in base
        prompt = build_plan_revision_prompt(
            *args, generalization_guidance=True, finding_history=view, **kwargs
        )
        assert "Proactive generalization" in prompt
        assert "Codex finding item-1 (resolved)" in prompt
        assert prompt.count("Codex finding item-1 (resolved)") == 1
    compact = build_plan_revision_prompt(
        *args, compact_context=True, generalization_guidance=True, finding_history=view
    )
    from coding_review_agent_loop.prompts import COMPACT_PLANNING_VOLATILE_TAIL_MARKER

    prefix, tail = compact.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER)
    assert "Proactive generalization" in prefix and "item-1" not in prefix
    assert "item-1" in tail and "Proactive generalization" not in tail


def test_owner_scoped_replay_partial_owner_clearance_stays_active_and_state_carries():
    shared = _item("item-1", resolution_owners=("Codex", "Claude"))
    seed = _record(0, "reviewer", 1, subject="h0", state="blocking", new_items=(shared,))
    owner_resolves = _record(
        1, "reviewer", 2, agent="Codex", subject="h1", state="blocking",
        dispositions=(ReviewItemDisposition("item-1", "Codex", "resolved"),),
    )
    other_blocks = _record(
        2, "reviewer", 2, agent="Claude", subject="h1", state="blocking",
        dispositions=(ReviewItemDisposition("item-1", "Claude", "blocking", "no"),),
    )
    outcomes = canonical_history_item_outcomes(
        [seed, owner_resolves, other_blocks],
        reconciliation_mode="owner-scoped", same_status="same-pr",
    )
    assert outcomes == {"item-1": "active"}
    # Second owner resolves in a later group: now resolved (state carried forward).
    second = _record(
        3, "reviewer", 3, agent="Claude", subject="h2", state="blocking",
        dispositions=(ReviewItemDisposition("item-1", "Claude", "resolved"),),
    )
    codex_later = _record(
        4, "reviewer", 3, agent="Codex", subject="h2", state="blocking",
        dispositions=(ReviewItemDisposition("item-1", "Codex", "resolved"),),
    )
    outcomes = canonical_history_item_outcomes(
        [seed, owner_resolves, other_blocks, second, codex_later],
        reconciliation_mode="owner-scoped", same_status="same-pr",
    )
    assert outcomes == {"item-1": "resolved"}


def test_owner_scoped_replay_pending_second_owner_is_not_deferred_or_cleared():
    shared = _item("item-1", resolution_owners=("Codex", "Claude"))
    seed = _record(0, "reviewer", 1, subject="h0", state="blocking", new_items=(shared,))
    only_one = _record(
        1, "reviewer", 2, agent="Codex", subject="h1", state="blocking",
        dispositions=(ReviewItemDisposition("item-1", "Codex", "resolved"),),
    )
    assert canonical_history_item_outcomes(
        [seed, only_one], reconciliation_mode="owner-scoped", same_status="same-pr"
    ) == {"item-1": "active"}


@pytest.mark.parametrize("kind", ["github-pr-checks", "managed-exact-head-ci"])
def test_real_singleton_upsert_yields_distinct_instances_for_both_ci_kinds(kind):
    """A-to-B via the real `_upsert_machine_obligation`, repeat, then clearance."""
    a, b = "a" * 40, "b" * 40

    def text(head, check):
        return f"Failing checks\nReviewed head: {head}\n- {check}: failure\nadvisory"

    ledger, _ = _ledger()
    items = _upsert_machine_obligation(
        [], item_number=1, kind=kind, source_round=2, text=text(a, "check-x"), failed_head_sha=a
    )
    ledger.observe_reconciled(items)
    ledger.observe_reconciled(items)  # repeated snapshot of A
    items = _upsert_machine_obligation(
        items, item_number=1, kind=kind, source_round=4, text=text(b, "check-y"), failed_head_sha=b
    )
    ledger.observe_reconciled(items)
    body = ledger.view(9).body
    assert body.count(f"CI {kind} on") == 2
    assert f"CI {kind} on {a[:12]} (superseded by the failure on {b[:12]})" in body
    assert f"CI {kind} on {b[:12]} (open)" in body
    assert "- check-x: failure" in body and "- check-y: failure" in body
    # First-observed rounds: A is round 2, B is round 4.
    a_at = body.index(f"CI {kind} on {a[:12]}")
    b_at = body.index(f"CI {kind} on {b[:12]} (open)")
    assert body.index("Round 2:") < a_at < body.index("Round 4:") < b_at
    ledger.observe_reconciled(_clear_machine_obligations(items, kind=kind))
    assert f"CI {kind} on {b[:12]} (resolved)" in ledger.view(9).body


def test_owner_scoped_replay_keeps_owner_state_without_a_repeated_vote():
    shared = _item("item-1", resolution_owners=("Codex", "Claude"))
    seed = _record(0, "reviewer", 1, subject="h0", state="blocking", new_items=(shared,))
    codex = _record(
        1, "reviewer", 2, agent="Codex", subject="h1", state="blocking",
        dispositions=(ReviewItemDisposition("item-1", "Codex", "resolved"),),
    )
    claude_blocks = _record(
        2, "reviewer", 2, agent="Claude", subject="h1", state="blocking",
        dispositions=(ReviewItemDisposition("item-1", "Claude", "blocking", "no"),),
    )
    claude_later = _record(
        3, "reviewer", 3, agent="Claude", subject="h2", state="blocking",
        dispositions=(ReviewItemDisposition("item-1", "Claude", "resolved"),),
    )
    # Codex's earlier resolution was carried in the item's owner state: Claude's
    # later resolution alone must now clear it.
    assert canonical_history_item_outcomes(
        [seed, codex, claude_blocks, claude_later],
        reconciliation_mode="owner-scoped", same_status="same-pr",
    ) == {"item-1": "resolved"}


def test_replay_carries_updated_subitem_state_into_a_later_group():
    item = _item(
        "item-1",
        sub_items=(ReviewSubItem("a", "x"), ReviewSubItem("b", "y")),
    )
    seed = _record(0, "reviewer", 1, subject="h0", state="blocking", new_items=(item,))
    partial = _record(
        1, "reviewer", 2, subject="h1", state="blocking",
        dispositions=(
            ReviewItemDisposition(
                "item-1", "Codex", "blocking", "b remains",
                sub_item_dispositions=(("a", "resolved"), ("b", "unresolved")),
            ),
        ),
    )
    assert canonical_history_item_outcomes(
        [seed, partial], reconciliation_mode="aggregate", same_status="same-pr"
    ) == {"item-1": "active"}


def test_semantic_patch_planner_prompt_orders_history_before_step_back(tmp_path):
    config = make_config(tmp_path)
    view = _history_view()
    step_back = PlanStepBackContext(
        measurements="m", findings=("f",), execution_mode="implement-one-shot", streak=2
    )
    kwargs = dict(
        response_form="semantic-patch-v1", base_round_number=1, base_state_identity="abc"
    )
    base = build_plan_revision_prompt(56, 3, "previous plan", "the review", config, **kwargs)
    assert "Proactive generalization" not in base
    prompt = build_plan_revision_prompt(
        56, 3, "previous plan", "the review", config,
        generalization_guidance=True, finding_history=view, step_back_context=step_back,
        **kwargs,
    )
    assert prompt.count("Codex finding item-1 (resolved)") == 1
    assert (
        prompt.index("Proactive generalization")
        < prompt.index("Codex finding item-1")
        < prompt.index("STEP-BACK REVISION")
        < prompt.index("the review")
    )


def test_full_and_compact_planner_prompts_order_guidance_before_step_back(tmp_path):
    config = make_config(tmp_path)
    view = _history_view()
    step_back = PlanStepBackContext(
        measurements="m", findings=("f",), execution_mode="implement-one-shot", streak=2
    )
    for extra in ({}, {"compact_context": True}):
        prompt = build_plan_revision_prompt(
            56, 3, "previous plan", "the review", config,
            generalization_guidance=True, finding_history=view, step_back_context=step_back,
            **extra,
        )
        assert prompt.index("Proactive generalization") < prompt.index("STEP-BACK REVISION")
        assert prompt.index("Codex finding item-1") < prompt.index("STEP-BACK REVISION")


def test_owner_scoped_replay_carries_the_earlier_subitem_closure_into_the_next_group(
    monkeypatch,
):
    """Round 3 never repeats the round-2 closure of `a`, yet its candidate has `a` closed."""
    from coding_review_agent_loop import round_state

    item = _item(
        "item-1",
        resolution_owners=("Codex",),
        sub_items=(ReviewSubItem("a", "x"), ReviewSubItem("b", "y")),
    )
    seed = _record(0, "reviewer", 1, subject="h0", state="blocking", new_items=(item,))
    closes_a = _record(
        1, "reviewer", 2, agent="Codex", subject="h1", state="blocking",
        dispositions=(
            ReviewItemDisposition(
                "item-1", "Codex", "blocking", "b remains",
                sub_item_dispositions=(("a", "resolved"), ("b", "unresolved")),
            ),
        ),
    )
    closes_b = _record(
        2, "reviewer", 3, agent="Codex", subject="h2", state="blocking",
        dispositions=(
            ReviewItemDisposition(
                "item-1", "Codex", "blocking", "closing b",
                sub_item_dispositions=(("b", "resolved"),),
            ),
        ),
    )
    seen = []
    real = round_state._apply_unresolved_item_dispositions

    def spy(candidates, *args, **kwargs):
        seen.append(
            {c.item_id: {s.sub_item_id: s.status for s in c.sub_items} for c in candidates}
        )
        assert kwargs["reconciliation_mode"] == "owner-scoped"
        assert kwargs["retain_future"] is False
        return real(candidates, *args, **kwargs)

    monkeypatch.setattr(round_state, "_apply_unresolved_item_dispositions", spy)
    canonical_history_item_outcomes(
        [seed, closes_a, closes_b], reconciliation_mode="owner-scoped", same_status="same-pr"
    )
    assert seen[0]["item-1"] == {"a": "open", "b": "open"}
    assert seen[1]["item-1"] == {"a": "resolved", "b": "open"}


def test_legacy_ci_snapshots_without_failed_head_seed_once_and_clear():
    """Legacy fallback key (kind, round, text digest): repeats dedupe, clearance resolves."""
    legacy = UnresolvedReviewItem(
        item_id="ci-item", reviewer="Orchestrator", source_round=1,
        text="Legacy\n- old-check: failure", status="blocking", authority="machine",
        obligation_kind="github-pr-checks", lifecycle="qualification_ready",
    )
    again = _record(1, "coder", 2, agent="Claude", subject="h1", prior_items=(legacy,))
    first = _record(0, "coder", 1, agent="Claude", subject="h0", prior_items=(legacy,))
    ledger, _ = _ledger()
    ledger.seed([first, again])
    body = ledger.view(5).body
    assert body.count("CI github-pr-checks") == 1
    assert "- old-check: failure" in body and "(resolved)" in body


def test_future_item_kept_in_the_ledger_and_then_cleared_is_resolved_not_deferred():
    """Full context / planner: a retained future finding that the reconciler clears."""
    ledger, _ = _ledger()
    future = _item("item-1", "future")
    ledger.observe_reconciled([future])
    assert "Codex finding item-1 (deferred)" in _view(ledger, 5)
    ledger.observe_reconciled([future])  # still carried: stays deferred
    assert "Codex finding item-1 (deferred)" in _view(ledger, 5)
    ledger.observe_reconciled([])  # reconciler cleared it
    assert "Codex finding item-1 (resolved)" in _view(ledger, 5)


def test_compact_mode_future_item_stays_deferred_across_absent_snapshots():
    ledger, _ = _ledger()
    ledger.observe_reconciled([], [_item("item-1", "future")])
    ledger.observe_reconciled([])
    ledger.observe_reconciled([])
    assert "Codex finding item-1 (deferred)" in _view(ledger, 5)


def test_replay_lets_a_resolved_disposition_clear_a_carried_future_item():
    item = _item("item-1")
    seed = _record(0, "reviewer", 1, subject="h0", state="blocking", new_items=(item,))
    carried = _record(
        1, "coder", 2, agent="Claude", subject="h1",
        prior_items=(UnresolvedReviewItem(
            item_id="item-1", reviewer="Codex", source_round=1, text=item.text, status="future"
        ),),
    )
    still = canonical_history_item_outcomes(
        [seed, carried], reconciliation_mode="aggregate", same_status="same-pr"
    )
    assert still == {"item-1": "deferred"}
    clears = _record(
        2, "reviewer", 2, subject="h1", state="blocking",
        dispositions=(ReviewItemDisposition("item-1", "Codex", "resolved"),),
    )
    assert canonical_history_item_outcomes(
        [seed, carried, clears], reconciliation_mode="aggregate", same_status="same-pr"
    ) == {"item-1": "resolved"}
