"""Plan-first loop tests: plan step-back turns and episodes (#1543)."""

import json
import re
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.round_state import _decode_round_metadata
from coding_review_agent_loop.protocol import EXECUTION_DISPOSITION_DIRECT
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in
from issue_loop_test_helpers import (
    _STEP_BACK_ANCHOR_PLANNER,
    _STEP_BACK_MARKER,
    _fresh_child_route_fixture,
    _m1103_agent_calls,
    _m1103_blocking_chain,
    _m1103_fresh_base,
    _m1251_justify_patch,
    _m1251_prompts,
    _m1251_run,
    _m1268_narrowing_patch,
    _m1268_staged_raw,
    _m936_patch,
    _plan_round_records,
    _staged_plan_config,
)
from coding_review_agent_loop.plan_assembly import AuthenticatedPlanState
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_review,
    structured_v1_plan_state,
)
import test_child_plan_provenance as _m936_cpp  # noqa: E402
from coding_review_agent_loop.plan_assembly import assemble_plan_revision  # noqa: E402
from coding_review_agent_loop.round_state import PostedRoundRecord  # noqa: E402
from coding_review_agent_loop.round_state import _encode_round_metadata  # noqa: E402
from coding_review_agent_loop.review_step_back import derive_plan_step_back_state  # noqa: E402


# --- Plan step-back turn and early human decision (#1251) ---------------------

_STEP_BACK_NOTICE = "Step-back notice"


def _m1251_coder_record(runner, round_number):
    return next(
        record
        for record in _plan_round_records(runner)
        if record.role == "coder" and record.round_number == round_number
    )


def _m1251_steps_patch(base, *, resolved_item, summary):
    payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": summary,
        "prior_plan_item_dispositions": [
            {"item_id": resolved_item, "disposition": "resolved"}
        ],
        "base_round_number": base.round_number,
        "base_state_identity": base.state_identity,
        "operations": [
            {"op": "replace", "field": "plan_steps", "value": [f"{summary} Implement it."]}
        ],
    }
    next_base = AuthenticatedPlanState.from_plan(
        assemble_plan_revision(base, payload), round_number=base.round_number + 1
    )
    text = (
        json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    return text, next_base


def test_crossed_plan_gets_the_step_back_prompt_at_exactly_the_kth_new_block(tmp_path):
    """`plan-crossed-k-new-blocks`, `plan-prompt-branches`: live trigger boundary."""
    runner, error = _m1251_run(
        tmp_path, last_round=5, max_rounds=5, plan_step_back_escalation_rounds=5
    )

    assert not isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    planner = _m1251_prompts(runner, "claude")
    # fresh plan, then the revisions after rounds 1, 2, 3 and 4.
    assert len(planner) == 5
    assert [_STEP_BACK_MARKER in prompt for prompt in planner] == [
        False, False, False, False, True,
    ]
    assert "Patching the newest finding" in planner[4]
    assert "[item-3] (round 3) Close gap 3." in planner[4]
    assert "[item-4] (round 4) Close gap 4." in planner[4]
    assert "revision-count" in planner[4]
    # The orchestrator records the reviewer-owned entry on the step-back turn.
    assert _m1251_coder_record(runner, 5).step_back_entries == (
        {"phase": "plan", "reviewer": "Codex", "trigger_round": 4},
    )
    assert all(
        record.step_back_entries == ()
        for record in _plan_round_records(runner)
        if not (record.role == "coder" and record.round_number == 5)
    )
    # Only the review of the step-back candidate carries the reviewer notice.
    codex = _m1251_prompts(runner, "codex")
    assert [_STEP_BACK_NOTICE in prompt for prompt in codex] == [
        False, False, False, False, True,
    ]
    # The approval-time growth lever is unchanged by the notice.
    assert "Plan-growth lever" in codex[4]


def test_crossed_plan_gets_the_step_back_prompt_under_review_parallel(tmp_path):
    """`plan-parallel-live-trigger` (#1271): provisional publication + reconciliation."""
    runner, error = _m1251_run(
        tmp_path,
        last_round=5,
        max_rounds=5,
        plan_step_back_escalation_rounds=5,
        review_parallel=True,
    )

    assert not isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    records = _plan_round_records(runner)
    # The primary's reviewer records really take the parallel (provisional) path.
    primary = [r for r in records if r.role == "reviewer" and r.agent == "Codex"]
    assert primary
    assert all(r.phase == "publication" and r.new_items == () for r in primary)
    # The minted items live on the reconciliation summaries.
    assert any(
        r.role == "summary" and r.phase == "reconciliation" and r.new_items for r in records
    )
    planner = _m1251_prompts(runner, "claude")
    assert [_STEP_BACK_MARKER in prompt for prompt in planner] == [
        False, False, False, False, True,
    ]
    assert "[item-3] (round 3) Close gap 3." in planner[4]
    assert "[item-4] (round 4) Close gap 4." in planner[4]
    assert _m1251_coder_record(runner, 5).step_back_entries == (
        {"phase": "plan", "reviewer": "Codex", "trigger_round": 4},
    )
    assert sum(1 for r in records if r.step_back_entries) == 1


def test_uncrossed_plan_keeps_the_ordinary_revision_prompt(tmp_path):
    """`plan-uncrossed-blocks`: K new-finding blocks on a plan that never crosses."""
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, 4, base=base)
    runner = _FakeRunner(claude_outputs=[fresh, *claude], codex_outputs=codex)
    config = _staged_plan_config(tmp_path, max_rounds=4)
    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert not any(_STEP_BACK_MARKER in prompt for prompt in _m1251_prompts(runner, "claude"))
    assert not any(_STEP_BACK_NOTICE in prompt for prompt in _m1251_prompts(runner, "codex"))
    assert all(record.step_back_entries == () for record in _plan_round_records(runner))


def test_repeat_only_block_between_new_findings_keeps_the_ordinary_prompt(tmp_path):
    """`plan-repeat-only-blocks`: a block that only carries item-3 forward is not new."""
    fresh, base0 = _m1103_fresh_base()
    codex1, claude1, base_round2 = _m1103_blocking_chain(1, 1, base=base0)
    codex2, _claude2, _base = _m1103_blocking_chain(2, 2, base=base_round2)
    justify, base_round3 = _m1251_justify_patch(base_round2, resolved_item="item-2")
    codex3, claude3, base_round4 = _m1103_blocking_chain(3, 3, base=base_round3)
    # Round 4 keeps item-3 blocking and raises nothing new.
    repeat_only = structured_plan_review(
        state="blocking",
        summary="Gap 3 still open.",
        prior_plan_item_dispositions=[{"item_id": "item-3", "disposition": "blocking"}],
    )
    patch_text, base_round5 = _m1251_steps_patch(
        base_round4, resolved_item="item-3", summary="Close gap 3 for real."
    )
    round5 = structured_plan_review(
        state="blocking",
        summary="Gap 5 remains.",
        blocking_plan_issues=["Close gap 5."],
        prior_plan_item_dispositions=[{"item_id": "item-3", "disposition": "resolved"}],
    )
    runner = _FakeRunner(
        claude_outputs=[fresh, *claude1, justify, *claude3, patch_text],
        codex_outputs=[*codex1, *codex2, *codex3, repeat_only, round5],
    )
    config = _staged_plan_config(
        tmp_path, max_rounds=5, plan_growth_max_chars=4500, plan_growth_max_revisions=3
    )
    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    # Rounds 3 (new), 4 (repeat-only), 5 (new): the streak never reaches K=2.
    planner = _m1251_prompts(runner, "claude")
    assert len(planner) == 5
    assert not any(_STEP_BACK_MARKER in prompt for prompt in planner)
    assert all(record.step_back_entries == () for record in _plan_round_records(runner))


def test_zero_step_back_rounds_restores_todays_behaviour(tmp_path):
    """`config-disable-and-validation`: 0 disables the trigger and the stop."""
    runner, error = _m1251_run(
        tmp_path, last_round=6, max_rounds=6, plan_step_back_rounds=0,
        plan_step_back_escalation_rounds=1,
    )

    assert not isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert not any(_STEP_BACK_MARKER in prompt for prompt in _m1251_prompts(runner, "claude"))
    assert all(record.step_back_entries == () for record in _plan_round_records(runner))


def test_a_rejected_step_back_stops_for_a_human_decision_before_the_stall_limit(tmp_path):
    """`plan-escalation-m-blocks`: the M-th block stops with no further planner turn."""
    runner, error = _m1251_run(tmp_path, last_round=6, max_rounds=8)

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    message = str(error)
    assert message.startswith("human decision required")
    assert "stepped back at round 5" in message
    assert "blocked 2 round(s)" in message
    assert "--plan-step-back-escalation-rounds 2" in message
    assert "revision-count" in message
    # The step-back candidate's own summary is the latest alternative.
    assert "Close gap 4." in message
    planner = _m1251_prompts(runner, "claude")
    assert sum(_STEP_BACK_MARKER in prompt for prompt in planner) == 1
    # fresh, p1, justification, p3, step-back, p5; then no planner turn at round 6.
    assert _m1103_agent_calls(runner) == [
        "claude", "codex", "claude", "codex", "claude", "codex",
        "claude", "codex", "claude", "codex", "claude", "codex",
    ]
    diagnostics = [
        comment["body"]
        for comment in runner.issue_comments
        if comment["body"].startswith("Plan review scheduling diagnostic (round 6)")
    ]
    assert len(diagnostics) == 1
    # Plain audit text: no round metadata a resume could read as a checkpoint.
    assert "AGENT_LOOP_META" not in diagnostics[0]
    assert "human decision required" in diagnostics[0]


def test_m_equal_one_stops_as_soon_as_the_step_back_candidate_is_rejected(tmp_path):
    runner, error = _m1251_run(
        tmp_path, last_round=5, max_rounds=8, plan_step_back_escalation_rounds=1
    )

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert "blocked 1 round(s)" in str(error)
    assert _m1103_agent_calls(runner)[-2:] == ["claude", "codex"]
    assert len(_m1251_prompts(runner, "claude")) == 5


def test_an_active_step_back_episode_defers_the_stall_stop_until_the_escalation_budget(
    tmp_path,
):
    """`live-episode-defers-stall`: stall limit 5 does not pre-empt M=5 (#1275)."""
    runner, error = _m1251_run(
        tmp_path, last_round=9, max_rounds=12, plan_primary_stall_rounds=5,
        plan_step_back_escalation_rounds=5,
    )

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert "human decision required" in str(error)
    assert "blocked 5 round(s)" in str(error)
    assert "blocked 5 consecutive primary-phase" not in str(error)
    assert len(_m1251_prompts(runner, "codex")) == 9


# --- #1275: the primary stall stop defers to a pending or active step-back ---


def _m1275_history(tmp_path):
    """A crossed history that K=3 left without a step-back, cut before round 5's review.

    Rounds 3 and 4 are new-finding blocks of the crossed plan (two, below K=3),
    and the round-5 candidate is published but not yet reviewed.
    """
    runner, error = _m1251_run(
        tmp_path, last_round=4, max_rounds=8, plan_step_back_rounds=3
    )
    assert not isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert not any(
        record.step_back_entries for record in _plan_round_records(runner)
    )
    return list(runner.issue_comments)


def _m1275_scripts(first_round=5, last_round=9, *, codex_overrides=None):
    """Chained outputs resuming at round ``first_round``'s primary review."""
    _fresh, base0 = _m1103_fresh_base()
    _c, _cl, base_round2 = _m1103_blocking_chain(1, 1, base=base0)
    _justify, base_round3 = _m1251_justify_patch(base_round2, resolved_item="item-2")
    codex, claude, _base = _m1103_blocking_chain(3, last_round, base=base_round3)
    codex = list(codex)
    for number, output in (codex_overrides or {}).items():
        codex[number - 3] = output
    return claude[first_round - 3:], codex[first_round - 3:]


def _m1275_resume(tmp_path, history, *, claude=(), codex=(), **overrides):
    values = {
        "max_rounds": 12,
        "plan_growth_max_chars": 4500,
        "plan_growth_max_revisions": 3,
        "plan_step_back_rounds": 2,
        "plan_step_back_escalation_rounds": 2,
    }
    values.update(overrides)
    rerun = _FakeRunner(
        issue_comments=history, claude_outputs=list(claude), codex_outputs=list(codex)
    )
    config = _staged_plan_config(tmp_path, **values)
    with patch.object(
        plan_first_loop_module, "log", wraps=plan_first_loop_module.log
    ) as logged:
        with pytest.raises(AgentLoopError) as excinfo:
            run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    messages = [str(call.args[-1]) for call in logged.call_args_list]
    return rerun, excinfo.value, messages


def _m1275_deferral_markers(runner):
    return [
        record.round_number
        for record in _plan_round_records(runner)
        if record.scheduler_step_back_deferral
    ]


@pytest.mark.parametrize("stall_rounds", [4, 2])
def test_stalled_step_back_eligible_history_gets_its_step_back_on_resume(
    tmp_path, stall_rounds
):
    """`resume-pending-eligible`, `streak-exceeds-threshold`: at and above the limit."""
    history = _m1275_history(tmp_path)
    claude, codex = _m1275_scripts()
    rerun, error, messages = _m1275_resume(
        tmp_path, history, claude=claude, codex=codex,
        plan_primary_stall_rounds=stall_rounds,
    )

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert "human decision required" in str(error)
    assert "blocked 2 round(s)" in str(error)
    assert any("stall stop deferred: pending plan step-back" in m for m in messages)
    assert any("crossed at round 3; 2 new-finding" in m for m in messages)
    assert _m1275_deferral_markers(rerun) == [5]
    prompts = _m1251_prompts(rerun, "claude")
    # The step-back turn, then one ordinary revision before the M-th block.
    assert [_STEP_BACK_MARKER in prompt for prompt in prompts] == [True, False]
    assert _m1103_agent_calls(rerun) == ["codex", "claude", "codex", "claude", "codex"]
    assert _m1251_coder_record(rerun, 6).step_back_entries == (
        {"phase": "plan", "reviewer": "Codex", "trigger_round": 5},
    )


def test_a_repeat_only_block_in_the_deferred_round_still_gets_the_step_back(tmp_path):
    """`deferred-round-repeat-only`."""
    history = _m1275_history(tmp_path)
    repeat = structured_plan_review(
        state="blocking",
        summary="Gap 4 still open.",
        prior_plan_item_dispositions=[{"item_id": "item-4", "disposition": "blocking"}],
    )
    claude, codex = _m1275_scripts(codex_overrides={5: repeat})
    # The repeat-only review raises no item-5, so the planner resolves item-4.
    step_back_turn = claude[0].replace('"item-5"', '"item-4"')
    rerun, _error, messages = _m1275_resume(
        tmp_path, history, claude=[step_back_turn], codex=codex[:1],
        plan_primary_stall_rounds=4,
    )

    assert any("stall stop deferred: pending plan step-back" in m for m in messages)
    assert any("step-back revision" in m for m in messages)
    assert _STEP_BACK_MARKER in _m1251_prompts(rerun, "claude")[0]
    assert _m1251_coder_record(rerun, 6).step_back_entries == (
        {"phase": "plan", "reviewer": "Codex", "trigger_round": 5},
    )
    assert _m1275_deferral_markers(rerun) == [5]


def test_resume_with_a_rejected_episode_stops_with_the_human_decision(tmp_path):
    """`resume-rejected-episode`: the stall limit is reached too."""
    first, first_error = _m1251_run(
        tmp_path, last_round=6, max_rounds=8, plan_step_back_escalation_rounds=5
    )
    assert not isinstance(first_error, plan_first_loop_module.PlanPrePanelSafetyError)
    rerun, error, _messages = _m1275_resume(
        tmp_path, list(first.issue_comments), plan_primary_stall_rounds=4,
        max_rounds=8,
    )

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert "human decision required" in str(error)
    assert "consecutive primary-phase" not in str(error)
    assert _m1103_agent_calls(rerun) == []


def test_resume_without_a_crossing_keeps_the_stall_stop_and_names_the_reason(tmp_path):
    """`resume-not-eligible`."""
    history = _m1275_history(tmp_path)
    rerun, error, _messages = _m1275_resume(
        tmp_path, history, plan_primary_stall_rounds=4, plan_growth_max_revisions=99,
    )

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert "blocked 4 consecutive primary-phase" in str(error)
    assert (
        "Plan step-back did not apply: the current candidate crosses no plan-growth "
        "signal." in str(error)
    )
    assert _m1103_agent_calls(rerun) == []
    assert _m1275_deferral_markers(rerun) == []


def test_a_spent_deferral_is_not_granted_again(tmp_path):
    """`spent-deferral-suppressed`: a marker at round 4 and no step-back entry."""
    history = _m1275_history(tmp_path)
    real_records = plan_first_loop_module._extract_round_metadata_records

    def mark_round_four(comments, *, flow):
        return tuple(
            PostedRoundRecord(
                index=record.index,
                metadata=(
                    replace(record.metadata, scheduler_step_back_deferral=True)
                    if record.metadata.phase == "scheduler-prelaunch"
                    and record.metadata.round_number == 4
                    and record.metadata.scheduler_phase == "primary"
                    else record.metadata
                ),
                body=record.body,
            )
            for record in real_records(comments, flow=flow)
        )

    with patch.object(
        plan_first_loop_module, "_extract_round_metadata_records",
        side_effect=mark_round_four,
    ):
        rerun, error, _messages = _m1275_resume(
            tmp_path, history, plan_primary_stall_rounds=4, plan_step_back_rounds=2,
        )

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert "blocked 4 consecutive primary-phase" in str(error)
    assert "already deferred at round 4" in str(error)
    assert _m1103_agent_calls(rerun) == []


def test_a_crash_after_the_deferred_review_resumes_into_the_step_back(tmp_path):
    """`crash-after-deferred-review`: no duplicate review, no ordinary revision."""
    history = _m1275_history(tmp_path)
    claude, codex = _m1275_scripts()
    # The first resume dies at the step-back planner turn, after round 5's review.
    crashed, _error, _messages = _m1275_resume(
        tmp_path, history, claude=[], codex=codex[:1], plan_primary_stall_rounds=4,
    )
    assert _m1103_agent_calls(crashed) == ["codex", "claude"]
    assert _m1275_deferral_markers(crashed) == [5]

    resumed, error, _messages = _m1275_resume(
        tmp_path, list(crashed.issue_comments), claude=claude[:2], codex=codex[1:],
        plan_primary_stall_rounds=4,
    )

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert _m1103_agent_calls(resumed)[0] == "claude"
    assert _STEP_BACK_MARKER in _m1251_prompts(resumed, "claude")[0]
    assert _m1251_coder_record(resumed, 6).step_back_entries


@pytest.mark.parametrize(
    ("overrides", "degrade", "reason"),
    [
        ({"plan_step_back_rounds": 0}, False, "step-back disabled"),
        ({}, True, "planning step-back history degraded"),
        ({"plan_growth_gate": "off"}, False, "plan-growth gate is off"),
    ],
)
def test_a_reached_stall_names_why_the_step_back_did_not_apply(
    tmp_path, overrides, degrade, reason
):
    """`disabled-or-degraded` through the stall branch's call site."""
    history = _m1275_history(tmp_path)
    real_records = plan_first_loop_module._extract_round_metadata_records

    def malformed(comments, *, flow):
        return tuple(
            PostedRoundRecord(
                index=record.index,
                metadata=(
                    replace(record.metadata, step_back_status="invalid")
                    if degrade and record.metadata.role == "coder"
                    else record.metadata
                ),
                body=record.body,
            )
            for record in real_records(comments, flow=flow)
        )

    with patch.object(
        plan_first_loop_module, "_extract_round_metadata_records", side_effect=malformed
    ):
        rerun, error, _messages = _m1275_resume(
            tmp_path, history, plan_primary_stall_rounds=4, **overrides
        )

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert "blocked 4 consecutive primary-phase" in str(error)
    assert f"Plan step-back did not apply: {reason}" in str(error)
    assert _m1103_agent_calls(rerun) == []
    assert _m1275_deferral_markers(rerun) == []


def test_resume_after_the_step_back_turn_issues_no_second_step_back_and_keeps_the_count(
    tmp_path,
):
    """`plan-resume-after-step-back`, `plan-episode-new-repeat-new` (resume)."""
    # The first process dies before the round-5 review of the step-back candidate.
    first, first_error = _m1251_run(tmp_path, last_round=4, max_rounds=8)
    assert not isinstance(first_error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert _m1251_coder_record(first, 5).step_back_entries
    history = list(first.issue_comments)

    fresh, base0 = _m1103_fresh_base()
    _c, _cl, base_round2 = _m1103_blocking_chain(1, 1, base=base0)
    _justify, base_round3 = _m1251_justify_patch(base_round2, resolved_item="item-2")
    codex, claude, _base = _m1103_blocking_chain(3, 6, base=base_round3)
    rerun = _FakeRunner(
        issue_comments=history, claude_outputs=[claude[2]], codex_outputs=codex[2:]
    )
    config = _staged_plan_config(
        tmp_path, max_rounds=8, plan_growth_max_chars=4500, plan_growth_max_revisions=3
    )
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError) as excinfo:
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)

    # Round 5's block is the first since the step-back and round 6's the second.
    assert "stepped back at round 5" in str(excinfo.value)
    assert "blocked 2 round(s)" in str(excinfo.value)
    planner = _m1251_prompts(rerun, "claude")
    assert len(planner) == 1 and _STEP_BACK_MARKER not in planner[0]
    assert _m1103_agent_calls(rerun) == ["codex", "claude", "codex"]


def test_step_back_state_is_suppressed_for_a_malformed_entry_without_crashing(tmp_path):
    """`plan-malformed-marker` through the live loop: today's behaviour applies."""
    real_records = plan_first_loop_module._extract_round_metadata_records

    def degrade(comments, *, flow):
        records = real_records(comments, flow=flow)
        return tuple(
            PostedRoundRecord(
                index=record.index,
                metadata=(
                    replace(record.metadata, step_back_status="invalid")
                    if record.metadata.role == "coder"
                    else record.metadata
                ),
                body=record.body,
            )
            for record in records
        )

    with patch.object(
        plan_first_loop_module, "_extract_round_metadata_records", side_effect=degrade
    ), patch.object(
        plan_first_loop_module, "log", wraps=plan_first_loop_module.log
    ) as logged:
        runner, error = _m1251_run(
            tmp_path, last_round=5, max_rounds=5, plan_step_back_escalation_rounds=1
        )

    assert not isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert not any(_STEP_BACK_MARKER in prompt for prompt in _m1251_prompts(runner, "claude"))
    assert any("step-back suppressed" in str(call.args[-1]) for call in logged.call_args_list)


def test_a_primary_approval_of_the_step_back_candidate_ends_the_episode(tmp_path):
    """`plan-step-back-approved`: the panel opens and no stop or second step-back follows."""
    fresh, base0 = _m1103_fresh_base()
    codex1, claude1, base_round2 = _m1103_blocking_chain(1, 1, base=base0)
    codex2, _claude2, _base = _m1103_blocking_chain(2, 2, base=base_round2)
    justify, base_round3 = _m1251_justify_patch(base_round2, resolved_item="item-2")
    codex_rest, claude_rest, _base = _m1103_blocking_chain(3, 4, base=base_round3)
    approve = structured_plan_review(
        state="approved",
        prior_plan_item_dispositions=[{"item_id": "item-4", "disposition": "resolved"}],
    )
    panel = structured_plan_review(state="approved", reviewer="Google Gemini")
    runner = _FakeRunner(
        claude_outputs=[fresh, *claude1, justify, *claude_rest],
        codex_outputs=[*codex1, *codex2, *codex_rest, approve],
        gemini_outputs=[panel],
    )
    config = _staged_plan_config(
        tmp_path,
        max_rounds=8,
        plan_growth_max_chars=4500,
        plan_growth_max_revisions=3,
        plan_step_back_escalation_rounds=1,
    )
    run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    # M=1 would stop at once on a rejected step-back; the approval is not a block.
    planner = _m1251_prompts(runner, "claude")
    assert sum(_STEP_BACK_MARKER in prompt for prompt in planner) == 1
    assert not any(
        comment["body"].startswith("Plan review scheduling diagnostic")
        for comment in runner.issue_comments
    )
    # Every reviewer of the step-back candidate, panel included, sees the notice.
    gemini = _m1251_prompts(runner, "gemini")
    assert len(gemini) == 1 and _STEP_BACK_NOTICE in gemini[0]


def test_the_mth_block_at_max_rounds_still_posts_the_human_decision_diagnostic(tmp_path):
    """The stop is evaluated before the generic max-rounds exit."""
    runner, error = _m1251_run(tmp_path, last_round=6, max_rounds=6)

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert str(error).startswith("human decision required")
    assert "stepped back at round 5" in str(error)
    # No planner turn after the M-th block, and the plain diagnostic was posted.
    assert len(_m1251_prompts(runner, "claude")) == 6
    diagnostics = [
        comment["body"]
        for comment in runner.issue_comments
        if comment["body"].startswith("Plan review scheduling diagnostic (round 6)")
    ]
    assert len(diagnostics) == 1 and "AGENT_LOOP_META" not in diagnostics[0]


def test_round_start_step_back_stop_is_independent_of_the_stall_threshold(tmp_path):
    """`--plan-primary-stall-rounds 0` disables only the stall stop, not the step-back stop."""
    first, first_error = _m1251_run(
        tmp_path, last_round=6, max_rounds=8, plan_primary_stall_rounds=0,
        plan_step_back_escalation_rounds=5,
    )
    assert not isinstance(first_error, plan_first_loop_module.PlanPrePanelSafetyError)
    # Rounds 5 and 6 blocked since the step-back; candidate 7 is published unreviewed.
    history = list(first.issue_comments)
    assert _m1251_coder_record(first, 7)

    rerun = _FakeRunner(issue_comments=history)
    config = _staged_plan_config(
        tmp_path, max_rounds=8, plan_growth_max_chars=4500, plan_growth_max_revisions=3,
        plan_primary_stall_rounds=0, plan_step_back_escalation_rounds=2,
    )
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError) as excinfo:
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)

    assert "blocked 2 round(s)" in str(excinfo.value)
    assert _m1103_agent_calls(rerun) == []


def _m1251_new_repeat_new_script():
    """Scripted outputs: step-back after round 4, then new (5), repeat-only (6), new (7)."""
    fresh, base0 = _m1103_fresh_base()
    codex1, claude1, base_round2 = _m1103_blocking_chain(1, 1, base=base0)
    codex2, _claude2, _base = _m1103_blocking_chain(2, 2, base=base_round2)
    justify, base_round3 = _m1251_justify_patch(base_round2, resolved_item="item-2")
    codex_rest, claude_rest, base_round6 = _m1103_blocking_chain(3, 5, base=base_round3)
    repeat_only = structured_plan_review(
        state="blocking",
        summary="Gap 5 still open.",
        prior_plan_item_dispositions=[{"item_id": "item-5", "disposition": "blocking"}],
    )
    patch6, _base = _m1251_steps_patch(
        base_round6, resolved_item="item-5", summary="Close gap 5 for real."
    )
    round7 = structured_plan_review(
        state="blocking",
        summary="Gap 7 remains.",
        blocking_plan_issues=["Close gap 7."],
        prior_plan_item_dispositions=[{"item_id": "item-5", "disposition": "resolved"}],
    )
    claude = [fresh, *claude1, justify, *claude_rest, patch6]
    codex = [*codex1, *codex2, *codex_rest, repeat_only, round7]
    return claude, codex


def _m1251_new_repeat_new_config(tmp_path):
    return _staged_plan_config(
        tmp_path, max_rounds=8, plan_growth_max_chars=4500, plan_growth_max_revisions=3,
        plan_step_back_escalation_rounds=3,
    )


def test_new_repeat_new_after_a_step_back_counts_all_three_blocks_live(tmp_path):
    """`plan-episode-new-repeat-new` (live): one step-back, stop on the third block."""
    claude, codex = _m1251_new_repeat_new_script()
    runner = _FakeRunner(claude_outputs=claude, codex_outputs=codex)
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError) as excinfo:
        run_issue_loop(
            runner, issue_number=56, config=_m1251_new_repeat_new_config(tmp_path),
            plan_first=True,
        )

    assert "stepped back at round 5" in str(excinfo.value)
    assert "blocked 3 round(s)" in str(excinfo.value)
    planner = _m1251_prompts(runner, "claude")
    # fresh, p1, justification, p3, step-back, p5, p6; none after the third block.
    assert len(planner) == 7
    assert sum(_STEP_BACK_MARKER in prompt for prompt in planner) == 1
    entries = [
        record.round_number
        for record in _plan_round_records(runner)
        if record.role == "coder" and record.step_back_entries
    ]
    assert entries == [5]
    assert len(_m1251_prompts(runner, "codex")) == 7


def test_new_repeat_new_after_a_step_back_counts_all_three_blocks_on_resume(tmp_path):
    """`plan-episode-new-repeat-new` (resume between the blocks)."""
    claude, codex = _m1251_new_repeat_new_script()
    # The first process dies after publishing candidate 6, before its review.
    first = _FakeRunner(claude_outputs=claude[:6], codex_outputs=codex[:5])
    with pytest.raises(AgentLoopError):
        run_issue_loop(
            first, issue_number=56, config=_m1251_new_repeat_new_config(tmp_path),
            plan_first=True,
        )
    assert _m1251_coder_record(first, 6)
    rerun = _FakeRunner(
        issue_comments=list(first.issue_comments), claude_outputs=claude[6:],
        codex_outputs=codex[5:],
    )
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError) as excinfo:
        run_issue_loop(
            rerun, issue_number=56, config=_m1251_new_repeat_new_config(tmp_path),
            plan_first=True,
        )

    assert "blocked 3 round(s)" in str(excinfo.value)
    planner = _m1251_prompts(rerun, "claude")
    assert len(planner) == 1 and _STEP_BACK_MARKER not in planner[0]
    assert _m1103_agent_calls(rerun) == ["codex", "claude", "codex"]


def test_unqualified_legacy_blocks_do_not_count_toward_the_trigger(tmp_path):
    """Reviews without a primary-phase checkpoint end the streak (degraded history)."""
    real_records = plan_first_loop_module._extract_round_metadata_records

    def legacy(comments, *, flow):
        records = real_records(comments, flow=flow)
        return tuple(
            record
            for record in records
            if not (
                record.metadata.role == "summary"
                and record.metadata.phase == "scheduler-prelaunch"
                and record.metadata.round_number <= 3
            )
        )

    with patch.object(
        plan_first_loop_module, "_extract_round_metadata_records", side_effect=legacy
    ):
        runner, _error = _m1251_run(
            tmp_path, last_round=6, max_rounds=6, plan_step_back_escalation_rounds=5
        )

    planner = _m1251_prompts(runner, "claude")
    # Round 3's block is unqualified, so rounds 3+4 do not make K=2; rounds 4+5 do.
    assert [_STEP_BACK_MARKER in prompt for prompt in planner] == [
        False, False, False, False, False, True,
    ]


def _m1251_edit_resume(tmp_path, *, drop_round5_checkpoint):
    """Interrupt in round 5 (candidate 5 is the step-back), edit the issue, resume."""
    first, _error = _m1251_run(tmp_path, last_round=4, max_rounds=8)
    assert _m1251_coder_record(first, 5).step_back_entries
    history = []
    for comment in first.issue_comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment["body"])
        if drop_round5_checkpoint and match is not None:
            metadata = _decode_round_metadata(match.group("payload"))
            if metadata.phase == "scheduler-prelaunch" and metadata.round_number == 5:
                continue
        history.append(comment)

    fresh, base0 = _m1103_fresh_base()
    _c, _cl, base_round2 = _m1103_blocking_chain(1, 1, base=base0)
    _justify, base_round3 = _m1251_justify_patch(base_round2, resolved_item="item-2")
    codex, claude, _base = _m1103_blocking_chain(3, 7, base=base_round3)
    rerun = _FakeRunner(
        issue_comments=history,
        issue_payload={"body": "Narrowed issue body with explicit non-goals."},
        claude_outputs=[claude[2], claude[3]],
        codex_outputs=[codex[2], codex[3], codex[4]],
    )
    # The issue edit keeps the existing plan (#1390 opt-out): these tests pin
    # the in-place step-back episode semantics of an edited issue (#1251).
    config = _staged_plan_config(
        tmp_path, max_rounds=8, plan_growth_max_chars=4500, plan_growth_max_revisions=3,
        plan_step_back_escalation_rounds=1, plan_resume_keep_on_issue_edit=True,
    )
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError) as excinfo:
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    return rerun, str(excinfo.value)


def test_issue_edit_closes_an_episode_published_before_its_first_checkpoint(tmp_path):
    """The edited issue restarts from fresh blocks; the old rejection does not escalate."""
    rerun, message = _m1251_edit_resume(tmp_path, drop_round5_checkpoint=True)

    # M=1 would have stopped at round 5 had the old episode survived the edit.
    assert "stepped back at round 7" in message
    planner = _m1251_prompts(rerun, "claude")
    assert [_STEP_BACK_MARKER in prompt for prompt in planner] == [False, True]
    assert _m1103_agent_calls(rerun) == ["codex", "claude", "codex", "claude", "codex"]


def test_obsolete_same_round_checkpoint_does_not_discard_a_fresh_review(tmp_path):
    """Fresh blocks in rounds 5 and 6 trigger exactly at K=2 after an issue edit."""
    rerun, message = _m1251_edit_resume(tmp_path, drop_round5_checkpoint=False)

    assert "stepped back at round 7" in message
    planner = _m1251_prompts(rerun, "claude")
    # Round 5's block is fresh (1); the step-back lands at round 6's boundary (2).
    assert [_STEP_BACK_MARKER in prompt for prompt in planner] == [False, True]
    checkpoints = [
        record.round_number
        for record in _plan_round_records(rerun)
        if record.phase == "scheduler-prelaunch" and record.round_number == 5
    ]
    # The old-issue checkpoint plus the fresh one published on resume.
    assert len(checkpoints) == 2


_STEP_BACK_ANCHOR_REVIEW = "Step-back anchor (orchestrator"


def test_episode_anchor_reaches_reviewer_and_planner_prompts_after_the_step_back(tmp_path):
    runner, error = _m1251_run(
        tmp_path, last_round=7, max_rounds=9, plan_step_back_escalation_rounds=5
    )

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError) or error
    planner = _m1251_prompts(runner, "claude")
    codex = _m1251_prompts(runner, "codex")
    # Planner: fresh, p1, justification, p3, step-back turn (no anchor), then anchored.
    assert [_STEP_BACK_ANCHOR_PLANNER in prompt for prompt in planner] == [
        False, False, False, False, False, True, True, True,
    ]
    assert all(_STEP_BACK_MARKER not in prompt for prompt in planner[5:])
    assert "within the simplified design" in planner[5]
    # Reviewer: the step-back candidate (round 5) and every later round are anchored.
    anchored = [_STEP_BACK_ANCHOR_REVIEW in prompt for prompt in codex]
    assert anchored[:4] == [False] * 4 and all(anchored[4:]) and len(anchored) > 5
    assert _STEP_BACK_NOTICE in codex[4] and _STEP_BACK_NOTICE not in codex[5]
    assert "does not filter or downgrade findings" in codex[4]
    # No second step-back entry is recorded.
    assert sum(1 for r in _plan_round_records(runner) if r.step_back_entries) == 1


def test_prompts_carry_no_anchor_without_a_step_back_episode(tmp_path):
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, 4, base=base)
    runner = _FakeRunner(claude_outputs=[fresh, *claude], codex_outputs=codex)
    config = _staged_plan_config(tmp_path, max_rounds=4)
    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    for prompt in (*_m1251_prompts(runner, "claude"), *_m1251_prompts(runner, "codex")):
        assert _STEP_BACK_ANCHOR_PLANNER not in prompt
        assert _STEP_BACK_ANCHOR_REVIEW not in prompt


def test_post_review_stop_reports_the_anchor_and_truthful_timing(tmp_path):
    runner, error = _m1251_run(tmp_path, last_round=6, max_rounds=8)

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    message = str(error)
    assert "Dissolved by the step-back:" in message
    assert "none declared (undeclared reintroductions are not detected)" in message
    assert "This round's reviews were posted; no further planner or reviewer turn" in message
    assert "No reviewer and no planner turn were invoked" not in message
    # The stop fires before any further agent launch.
    assert len(_m1251_prompts(runner, "claude")) == 6
    assert len(_m1251_prompts(runner, "codex")) == 6


def test_post_review_stop_lists_a_declared_reintroduction(tmp_path):
    def declare(output):
        if not output.lstrip().startswith("{"):
            return output
        payload, end = json.JSONDecoder().raw_decode(output.lstrip())
        if payload.get("kind") != "plan_revision_patch":
            return output
        marker = "reintroduces dissolved item item-4: needs a synchronous close"
        if marker in payload.get("summary", "") or not payload["summary"].startswith("Close gap 6"):
            return output
        payload["summary"] = f"{payload['summary']}\n{marker}"
        return json.dumps(payload) + output.lstrip()[end:]

    runner, error = _m1251_run(
        tmp_path, last_round=7, max_rounds=9, plan_step_back_escalation_rounds=3,
        claude_transform=declare,
    )

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    message = str(error)
    assert "[item-4] at round 7 (needs a synchronous close)" in message
    assert "none declared" not in message
    records = _plan_round_records(runner)
    assert any(r.phase == "scheduler-prelaunch" and r.round_number == 7 for r in records)
    assert not any(r.round_number > 7 for r in records)
    assert not any(r.role == "coder" and r.round_number >= 8 for r in records)


def _m1278_declaring_transform(output):
    """Make the planner's round-7 revision declare reversing item-4."""
    if not output.lstrip().startswith("{"):
        return output
    payload, end = json.JSONDecoder().raw_decode(output.lstrip())
    if payload.get("kind") != "plan_revision_patch" or not payload["summary"].startswith(
        "Close gap 6"
    ):
        return output
    payload["summary"] = (
        f"{payload['summary']}\nreintroduces dissolved item item-4: needs a synchronous close"
    )
    return json.dumps(payload) + output.lstrip()[end:]


def test_round_start_stop_after_a_post_review_stop_reports_the_declaration(tmp_path):
    """`stop-prelaunch-with-reintroduction`: the resumed round-start stop site."""
    first, error = _m1251_run(
        tmp_path, last_round=7, max_rounds=12, plan_step_back_escalation_rounds=9,
        claude_transform=_m1278_declaring_transform,
    )
    assert not isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    history = list(first.issue_comments)
    before = len(_plan_round_records(first))

    rerun, stop, _messages = _m1275_resume(
        tmp_path, history, plan_step_back_escalation_rounds=2, max_rounds=12,
    )

    assert isinstance(stop, plan_first_loop_module.PlanPrePanelSafetyError)
    message = str(stop)
    assert "[item-4] at round 7 (needs a synchronous close)" in message
    assert "No reviewer and no planner turn were invoked." in message
    assert "This round's reviews were posted" not in message
    # Zero agent launches and no new round record for the stopping round.
    assert _m1103_agent_calls(rerun) == []
    assert len(_plan_round_records(rerun)) == before


def _m1278_episode_history(tmp_path):
    first, error = _m1251_run(
        tmp_path, last_round=7, max_rounds=12, plan_step_back_escalation_rounds=9,
    )
    assert not isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    return list(first.issue_comments)


@pytest.mark.parametrize("review_parallel", [False, True])
def test_reset_round_reviewers_and_planner_carry_no_anchor(tmp_path, review_parallel):
    """`reset-round-reviewer`: the reset closes the episode before reviewers run."""
    history = _m1278_episode_history(tmp_path)
    claude, codex = _m1275_scripts(first_round=8, last_round=9)
    plain, _e, _m = _m1275_resume(
        tmp_path, history, claude=claude[:1], codex=codex[:1],
        plan_step_back_escalation_rounds=9, review_parallel=review_parallel,
    )
    assert _STEP_BACK_ANCHOR_REVIEW in _m1251_prompts(plain, "codex")[0]
    assert _STEP_BACK_ANCHOR_PLANNER in _m1251_prompts(plain, "claude")[0]

    reset, _e, _m = _m1275_resume(
        tmp_path, history, claude=claude[:1], codex=codex[:1],
        plan_step_back_escalation_rounds=9, review_parallel=review_parallel,
        plan_reset_stall_streak=True,
    )
    for agent in ("codex", "claude"):
        for prompt in _m1251_prompts(reset, agent):
            assert _STEP_BACK_ANCHOR_REVIEW not in prompt
            assert _STEP_BACK_ANCHOR_PLANNER not in prompt


def test_parallel_and_sequential_episode_reviewer_prompts_match(tmp_path):
    """`parallel-reviewers`: both launch modes render the same pre-round anchor."""
    history = _m1278_episode_history(tmp_path)
    claude, codex = _m1275_scripts(first_round=8, last_round=9)
    prompts = {}
    for parallel in (False, True):
        rerun, _e, _m = _m1275_resume(
            tmp_path, history, claude=claude[:1], codex=codex[:1],
            plan_step_back_escalation_rounds=9, review_parallel=parallel,
        )
        prompts[parallel] = _m1251_prompts(rerun, "codex")[0]
    block = lambda text: text[text.index(_STEP_BACK_ANCHOR_REVIEW):]  # noqa: E731
    assert block(prompts[False]).split("\n")[:12] == block(prompts[True]).split("\n")[:12]


def test_resumed_episode_prompts_match_the_uninterrupted_run(tmp_path):
    """`resume-replay`: resuming mid-episode rebuilds the uninterrupted anchor blocks."""
    def block(prompt, marker, lines=9):
        return prompt[prompt.index(marker):].split("\n")[:lines]

    uninterrupted, _error = _m1251_run(
        tmp_path, last_round=8, max_rounds=12, plan_step_back_escalation_rounds=9,
    )
    # The uninterrupted run already constructed the same seven-round prefix.
    # Copy it at the round-8 review boundary instead of executing it twice;
    # the resumed runner must still reconstruct its own anchor from history.
    history = []
    for comment in uninterrupted.issue_comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment["body"])
        if match is not None:
            metadata = _decode_round_metadata(match.group("payload"))
            if metadata.role == "reviewer" and metadata.round_number == 8:
                break
        history.append(deepcopy(comment))
    else:
        pytest.fail("uninterrupted history did not reach round 8's review")
    claude, codex = _m1275_scripts(first_round=8, last_round=9)
    resumed, _e, _m = _m1275_resume(
        tmp_path, history, claude=claude[:1], codex=codex[:1],
        plan_step_back_escalation_rounds=9,
    )
    # Round 8's review and the planner revision after it are the resumed run's
    # first prompts; the uninterrupted run reaches them at the same positions.
    assert block(_m1251_prompts(resumed, "codex")[0], _STEP_BACK_ANCHOR_REVIEW) == block(
        _m1251_prompts(uninterrupted, "codex")[7], _STEP_BACK_ANCHOR_REVIEW
    )
    assert block(_m1251_prompts(resumed, "claude")[0], _STEP_BACK_ANCHOR_PLANNER) == block(
        _m1251_prompts(uninterrupted, "claude")[8], _STEP_BACK_ANCHOR_PLANNER
    )


# --- #1278: closure, suppression and resume coverage -------------------------


def _m1278_block(prompt, start, end):
    """The anchor block verbatim, from its opening marker through its closing sentence."""
    begin = prompt.index(start)
    return prompt[begin : prompt.index(end, begin) + len(end)]


_M1278_REVIEW_END = "does not filter or downgrade findings on it."


_M1278_PLANNER_END = "contradicts the anchor."


def test_degraded_history_with_an_existing_episode_suppresses_every_anchor(tmp_path):
    """`degraded-history-suppression`: an established episode, then a malformed entry."""
    history = _m1278_episode_history(tmp_path)
    claude, codex = _m1275_scripts(first_round=8, last_round=9)
    real_records = plan_first_loop_module._extract_round_metadata_records

    def degrade(comments, *, flow):
        return tuple(
            PostedRoundRecord(
                index=record.index,
                metadata=(
                    replace(
                        record.metadata, step_back_status="invalid", step_back_entries=()
                    )
                    if record.metadata.role == "coder"
                    else record.metadata
                ),
                body=record.body,
            )
            for record in real_records(comments, flow=flow)
        )

    with patch.object(
        plan_first_loop_module, "_extract_round_metadata_records", side_effect=degrade
    ):
        rerun, error, messages = _m1275_resume(
            tmp_path, history, claude=claude[:1], codex=codex[:1],
            plan_step_back_escalation_rounds=2,
        )

    for agent in ("codex", "claude"):
        for prompt in _m1251_prompts(rerun, agent):
            assert _STEP_BACK_ANCHOR_REVIEW not in prompt
            assert _STEP_BACK_ANCHOR_PLANNER not in prompt
            assert _STEP_BACK_MARKER not in prompt
    assert any("step-back suppressed" in m for m in messages)
    assert not any("step-back anchor" in m for m in messages)
    # No anchor-bearing stop and no new step-back entry.
    assert "human decision required" not in str(error)
    # Only the pre-existing entry from the history; the rerun records none.
    assert sum(1 for r in _plan_round_records(rerun) if r.step_back_entries) == 1


def test_resume_after_review_publication_rebuilds_the_planner_anchor(tmp_path):
    """`resume-replay`: interrupted after round 8's review posted, before the planner turn."""
    uninterrupted, _error = _m1251_run(
        tmp_path, last_round=8, max_rounds=12, plan_step_back_escalation_rounds=9,
    )
    # Round 9's candidate is the first coder record after round 8's review.
    history = []
    for comment in uninterrupted.issue_comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment["body"])
        if match is not None:
            metadata = _decode_round_metadata(match.group("payload"))
            if metadata.flow == "plan" and metadata.role == "coder" and metadata.round_number >= 9:
                break
        history.append(comment)
    claude, codex = _m1275_scripts(first_round=8, last_round=9)
    resumed, _e, _m = _m1275_resume(
        tmp_path, history, claude=claude[:1], codex=[],
        plan_step_back_escalation_rounds=9,
    )

    # The reviewers' round-8 posts are replayed, so only the planner turn launches.
    assert _m1103_agent_calls(resumed)[0] == "claude"
    planner_resumed = _m1251_prompts(resumed, "claude")[0]
    planner_uninterrupted = _m1251_prompts(uninterrupted, "claude")[8]
    assert _m1278_block(
        planner_resumed, _STEP_BACK_ANCHOR_PLANNER, _M1278_PLANNER_END
    ) == _m1278_block(planner_uninterrupted, _STEP_BACK_ANCHOR_PLANNER, _M1278_PLANNER_END)


def test_complete_anchor_blocks_match_between_sequential_and_parallel(tmp_path):
    """`parallel-reviewers`: full blocks, from the same pre-round snapshot."""
    history = _m1278_episode_history(tmp_path)
    claude, codex = _m1275_scripts(first_round=8, last_round=9)
    blocks = {}
    for parallel in (False, True):
        rerun, _e, _m = _m1275_resume(
            tmp_path, history, claude=claude[:1], codex=codex[:1],
            plan_step_back_escalation_rounds=9, review_parallel=parallel,
        )
        blocks[parallel] = _m1278_block(
            _m1251_prompts(rerun, "codex")[0], _STEP_BACK_ANCHOR_REVIEW, _M1278_REVIEW_END
        )
    assert blocks[False] == blocks[True]
    # The pre-round snapshot excludes round 8's own posts: nothing from round 8
    # (item-8) appears as a dissolved item.
    assert "[item-8]" not in blocks[True]


def test_post_review_stop_keeps_its_checkpoint_and_writes_nothing_later(tmp_path):
    """`stop-post-review-with-reintroduction`: checkpoint present, nothing after it."""
    runner, error = _m1251_run(tmp_path, last_round=6, max_rounds=8)

    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    records = _plan_round_records(runner)
    assert any(
        r.phase == "scheduler-prelaunch" and r.round_number == 6 for r in records
    )
    assert not any(r.round_number > 6 for r in records)
    assert not any(r.role == "coder" and r.round_number >= 7 for r in records)


def test_approved_candidate_panel_prompts_carry_no_anchor(tmp_path):
    """`episode-closed`: after the primary approves, the panel phase has no anchor."""
    fresh, base0 = _m1103_fresh_base()
    codex1, claude1, base_round2 = _m1103_blocking_chain(1, 1, base=base0)
    codex2, _claude2, _base = _m1103_blocking_chain(2, 2, base=base_round2)
    justify, base_round3 = _m1251_justify_patch(base_round2, resolved_item="item-2")
    codex_rest, claude_rest, _base = _m1103_blocking_chain(3, 4, base=base_round3)
    approve = structured_plan_review(
        state="approved",
        prior_plan_item_dispositions=[{"item_id": "item-4", "disposition": "resolved"}],
    )
    panel = structured_plan_review(state="approved", reviewer="Google Gemini")
    runner = _FakeRunner(
        claude_outputs=[fresh, *claude1, justify, *claude_rest],
        codex_outputs=[*codex1, *codex2, *codex_rest, approve],
        gemini_outputs=[panel],
    )
    config = _staged_plan_config(
        tmp_path, max_rounds=8, plan_growth_max_chars=4500, plan_growth_max_revisions=3,
        plan_step_back_escalation_rounds=1,
    )
    run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    gemini = _m1251_prompts(runner, "gemini")
    assert len(gemini) == 1 and _STEP_BACK_ANCHOR_REVIEW not in gemini[0]
    # The primary's own review of the candidate did carry the anchor.
    assert _STEP_BACK_ANCHOR_REVIEW in _m1251_prompts(runner, "codex")[4]


@pytest.mark.parametrize("drop_round5_checkpoint", [True, False])
def test_issue_edit_drops_the_anchor_from_the_first_later_prompts(
    tmp_path, drop_round5_checkpoint
):
    """`episode-closed`: the edit closes the old episode before any later prompt."""
    rerun, _message = _m1251_edit_resume(
        tmp_path, drop_round5_checkpoint=drop_round5_checkpoint
    )

    # The first review and the first revision after the edit still precede the
    # new episode's step-back turn.
    assert _STEP_BACK_ANCHOR_REVIEW not in _m1251_prompts(rerun, "codex")[0]
    assert _STEP_BACK_ANCHOR_PLANNER not in _m1251_prompts(rerun, "claude")[0]


def test_no_episode_run_logs_no_anchor_and_prompts_carry_no_anchor_text(tmp_path):
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, 4, base=base)
    runner = _FakeRunner(claude_outputs=[fresh, *claude], codex_outputs=codex)
    config = _staged_plan_config(tmp_path, max_rounds=4)
    with patch.object(
        plan_first_loop_module, "log", wraps=plan_first_loop_module.log
    ) as logged:
        with pytest.raises(AgentLoopError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert not any("step-back anchor" in str(c.args[-1]) for c in logged.call_args_list)
    for prompt in (*_m1251_prompts(runner, "claude"), *_m1251_prompts(runner, "codex")):
        assert "step-back anchor" not in prompt.lower()


def test_stall_escalated_stop_reports_the_declared_reversal_before_launch(tmp_path):
    """`stop-prelaunch-with-reintroduction`: the DISPOSITION_ESCALATED stall site.

    The round-start step-back stop shares its guards and state with this site and
    normally fires first, so the test suppresses that one derivation to reach the
    stall path with the same escalated history.
    """
    first, error = _m1251_run(
        tmp_path, last_round=7, max_rounds=12, plan_step_back_escalation_rounds=9,
        claude_transform=_m1278_declaring_transform,
    )
    assert not isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    history = list(first.issue_comments)
    before = len(_plan_round_records(first))

    real_derive = plan_first_loop_module.derive_plan_step_back_state
    calls = []

    def derive(*args, **kwargs):
        state = real_derive(*args, **kwargs)
        calls.append(state.escalation_count)
        # Call 1 is the reviewer-anchor snapshot, call 2 the round-start step-back
        # stop: hide the escalation from the latter only.
        return replace(state, escalation_count=0) if len(calls) == 2 else state

    with patch.object(plan_first_loop_module, "derive_plan_step_back_state", derive):
        rerun, stop, messages = _m1275_resume(
            tmp_path, history, plan_step_back_escalation_rounds=2,
            plan_primary_stall_rounds=2, max_rounds=12,
        )

    assert isinstance(stop, plan_first_loop_module.PlanPrePanelSafetyError)
    message = str(stop)
    assert "[item-4] at round 7 (needs a synchronous close)" in message
    assert "No reviewer and no planner turn were invoked." in message
    assert "This round's reviews were posted" not in message
    assert _m1103_agent_calls(rerun) == []
    assert len(_plan_round_records(rerun)) == before


def _m1278_normalized(prompts):
    return [
        re.sub(r"rounds[02]", "roundsN", re.sub(r"/responses/\S+\.md", "/responses/X.md", p))
        for p in prompts
    ]


def test_no_episode_prompts_match_a_run_with_step_back_disabled(tmp_path):
    """`no-episode-prompts`: the anchor machinery leaves no-episode prompt text unchanged."""
    runs = {}
    for rounds in (2, 0):
        fresh, base = _m1103_fresh_base()
        codex, claude, _base = _m1103_blocking_chain(1, 4, base=base)
        runner = _FakeRunner(claude_outputs=[fresh, *claude], codex_outputs=codex)
        workdir = tmp_path / f"rounds{rounds}"
        workdir.mkdir()
        config = _staged_plan_config(workdir, max_rounds=4, plan_step_back_rounds=rounds)
        with pytest.raises(AgentLoopError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
        runs[rounds] = runner
    for agent in ("codex", "claude"):
        left = _m1278_normalized(_m1251_prompts(runs[2], agent))
        right = _m1278_normalized(_m1251_prompts(runs[0], agent))
        assert len(left) == len(right)
        assert left == right


def test_narrowing_revision_inside_an_active_episode_carries_directive_and_anchor(tmp_path):
    """`narrow-revision-in-episode`: the independent planner-anchor path.

    The first resumed round sees a staged recommendation, so ``--plan-narrow-staged``
    owns that round's revision (the step-back gate is excluded); the anchor must
    still follow the narrowing directive, and no new step-back entry is recorded.
    """
    from coding_review_agent_loop.protocol import validate_structured_plan_state

    history = _m1278_episode_history(tmp_path)
    claude, codex = _m1275_scripts(first_round=8, last_round=9)
    real = plan_first_loop_module._current_execution_recommendation
    staged = validate_structured_plan_state(_m1268_staged_raw()).execution_recommendation
    initial_plan = []

    def recommendation(plan, *args, **kwargs):
        # Both ownership validation and narrowing inspect the same candidate.
        # Keep its topology stable across reads until the planner revises it.
        if not initial_plan:
            initial_plan.append(plan)
        return staged if plan == initial_plan[0] else real(plan, *args, **kwargs)

    with patch.object(
        plan_first_loop_module, "_current_execution_recommendation", recommendation
    ):
        rerun, _error, messages = _m1275_resume(
            tmp_path, history, claude=claude[:1], codex=codex[:1],
            plan_step_back_escalation_rounds=9, plan_narrow_staged=True,
            plan_execution_mode="implement-one-shot",
        )

    revision = _m1251_prompts(rerun, "claude")[0]
    directive = "Operator directive (orchestrator, not a reviewer finding)"
    assert directive in revision and _STEP_BACK_ANCHOR_PLANNER in revision
    assert _STEP_BACK_MARKER not in revision
    assert any("step-back anchor" in m for m in messages)
    assert sum(1 for r in _plan_round_records(rerun) if r.step_back_entries) == 1


def _m1278_staged_step_back_run(tmp_path):
    """A durable staged step-back candidate, then a primary block under --plan-narrow-staged."""
    fresh, base0 = _m1103_fresh_base()
    codex1, claude1, base_round2 = _m1103_blocking_chain(1, 1, base=base0)
    codex2, _claude2, _base = _m1103_blocking_chain(2, 2, base=base_round2)
    justify, base_round3 = _m1251_justify_patch(base_round2, resolved_item="item-2")
    codex_34, claude_3, base_round4 = _m1103_blocking_chain(3, 3, base=base_round3)
    codex_4, _claude_4, _base = _m1103_blocking_chain(4, 4, base=base_round4)
    staged_recommendation = json.loads(_m1268_staged_raw().split("\n<!--")[0])[
        "execution_recommendation"
    ]
    # Converting to staged must keep the one-shot plan's scope ledger verbatim.
    one_shot_scope = json.loads(structured_v1_plan_state().split("\n<!--")[0])[
        "execution_recommendation"
    ]["scope_items"]
    staged_recommendation["scope_items"][0] = one_shot_scope[0]
    step_back_payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "split: two stages dissolve the earlier findings.",
        "prior_plan_item_dispositions": [
            {"item_id": "item-4", "disposition": "resolved", "note": "dissolved: staged"}
        ],
        "base_round_number": 4,
        "base_state_identity": base_round4.state_identity,
        "operations": [
            {"op": "replace", "field": "execution_recommendation", "value": staged_recommendation},
            {"op": "replace", "field": "one_shot_growth_justification", "value": None},
        ],
    }
    base_round5 = AuthenticatedPlanState.from_plan(
        assemble_plan_revision(base_round4, step_back_payload), round_number=5
    )
    step_back = (
        json.dumps(step_back_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    codex5 = structured_plan_review(
        state="blocking", summary="Gap 5 remains.", blocking_plan_issues=["Close gap 5."],
        prior_plan_item_dispositions=[{"item_id": "item-4", "disposition": "resolved"}],
    )
    narrowing = _m1268_narrowing_patch(
        base_round5, round_number=5,
        dispositions=[{"item_id": "item-5", "disposition": "resolved"}],
    )
    runner = _FakeRunner(
        claude_outputs=[fresh, *claude1, justify, *claude_3, step_back, narrowing],
        codex_outputs=[*codex1, *codex2, *codex_34, *codex_4, codex5],
    )
    config = _staged_plan_config(
        tmp_path, max_rounds=8, plan_growth_max_chars=4500, plan_growth_max_revisions=3,
        plan_step_back_escalation_rounds=9, plan_narrow_staged=True,
        plan_execution_mode="implement-one-shot",
    )
    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return runner, excinfo.value


def test_staged_step_back_candidate_then_narrowing_carries_directive_and_anchor(tmp_path):
    """`narrow-revision-in-episode`: authentic staged step-back history, no patching.

    Narrowing is the only orchestrator-owned revision reachable inside an active
    episode: the growth-guard, inherited-guard and supersession revisions all
    require ``all_approved`` (the primary approved), which closes the episode, so
    they carry no anchor (see the approval closure tests).
    """
    runner, error = _m1278_staged_step_back_run(tmp_path)

    entries = [r for r in _plan_round_records(runner) if r.step_back_entries]
    assert len(entries) == 1 and entries[0].round_number == 5
    planner = _m1251_prompts(runner, "claude")
    narrowing = planner[5]
    assert "Operator directive (orchestrator, not a reviewer finding)" in narrowing
    assert _STEP_BACK_ANCHOR_PLANNER in narrowing
    assert "[item-4]" in narrowing
    assert _STEP_BACK_MARKER not in narrowing


def test_growth_guard_revision_after_an_in_episode_approval_carries_no_anchor(tmp_path):
    """`narrow-revision-in-episode` (guard branch): the approval closes the episode.

    The episode is active when round 8's review starts (that review carries the
    anchor).  Tightening the growth thresholds on resume makes the justified candidate
    non-compliant, the primary approves it, and the orchestrator-owned growth-guard
    revision follows.  The approval closes the episode: the guard notice is delivered
    with no planner anchor, no anchor log line and no second step-back entry.
    """
    history = _m1278_episode_history(tmp_path)
    approve = structured_plan_review(
        state="approved",
        prior_plan_item_dispositions=[{"item_id": "item-7", "disposition": "resolved"}],
    )
    rerun = _FakeRunner(issue_comments=history, claude_outputs=[], codex_outputs=[approve])
    config = _staged_plan_config(
        tmp_path, max_rounds=12, plan_growth_max_chars=1000, plan_growth_max_revisions=3,
        plan_step_back_rounds=2, plan_step_back_escalation_rounds=9,
    )
    with patch.object(
        plan_first_loop_module, "log", wraps=plan_first_loop_module.log
    ) as logged:
        with pytest.raises(AgentLoopError):
            run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)

    codex = _m1251_prompts(rerun, "codex")
    planner = _m1251_prompts(rerun, "claude")
    assert _STEP_BACK_ANCHOR_REVIEW in codex[0]
    guard = planner[0]
    assert "plan-growth" in guard.lower()
    assert _STEP_BACK_ANCHOR_PLANNER not in guard
    assert _STEP_BACK_MARKER not in guard
    assert not any("step-back anchor" in str(c.args[-1]) for c in logged.call_args_list)
    assert sum(1 for r in _plan_round_records(rerun) if r.step_back_entries) == 1


def _m1278_inject_step_back_entry(comments):
    """Mark the plan's first coder record as a step-back candidate (an episode opens)."""
    out = []
    injected = 0
    for comment in comments:
        body = comment["body"]
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", body)
        if match is not None and not injected:
            metadata = _decode_round_metadata(match.group("payload"))
            if metadata.flow == "plan" and metadata.role == "coder":
                metadata = replace(
                    metadata,
                    step_back_status="valid",
                    step_back_entries=(
                        {"phase": "plan", "reviewer": "Codex", "trigger_round": 1},
                    ),
                )
                payload = _encode_round_metadata(metadata)
                body = body.replace(match.group("payload"), payload)
                comment = {**comment, "body": body}
                injected += 1
        out.append(comment)
    assert injected == 1
    return out


def test_inherited_guard_revision_after_an_episode_closed_by_approval_has_no_anchor(
    tmp_path, monkeypatch
):
    """`narrow-revision-in-episode` (inherited-guard branch): approval closed the episode.

    Runs under primary-then-panel so the step-back machinery is live.  The inherited
    guard only applies to a plan the primary already approved, so the episode opened on
    that plan's candidate was closed by the approval before the round starts.  The
    revision gets its own correction record and no planner anchor, no anchor log line
    and no second step-back entry; the machinery is shown to have run and found no
    episode.
    """
    def config(**overrides):
        return _m936_cpp._plan_config(
            tmp_path,
            reviewer=("codex", "gemini"),
            plan_review_policy="primary-then-panel",
            primary_plan_reviewer="codex",
            **overrides,
        )

    approve_gemini = _m936_cpp.structured_plan_review(
        state="approved", reviewer="Google Gemini"
    )
    weak = _m936_cpp._child_plan_state(_m936_cpp._weak_child_row())
    first = _m936_cpp._ChildPlanningRunner(
        claude_outputs=[weak],
        codex_outputs=[_m936_cpp.structured_plan_review(state="approved")],
        gemini_outputs=[approve_gemini],
    )
    assert issue_loop.run_issue_loop(
        first, issue_number=56, config=config(), plan_first=True
    ) == 0
    history = _m1278_inject_step_back_entry(list(first.issue_comments))

    _m936_cpp._bind_child_planning(monkeypatch)
    good = _m936_patch(weak, _m936_cpp._child_row(), summary="Inherited rows restored.")
    resumed = _m936_cpp._ChildPlanningRunner(
        issue_comments=list(history),
        claude_outputs=[good],
        codex_outputs=[_m936_cpp.structured_plan_review(state="approved")],
        gemini_outputs=[approve_gemini],
    )
    # The injected episode is live until the primary's approval closes it.
    records = []
    for position, comment in enumerate(history):
        marker = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment["body"])
        if marker is not None:
            metadata = _decode_round_metadata(marker.group("payload"))
            if metadata.flow == "plan":
                records.append(PostedRoundRecord(index=position, metadata=metadata, body=""))
    primary = next(r.metadata.agent for r in records if r.metadata.role == "reviewer")
    first_reviewer = min(r.index for r in records if r.metadata.role == "reviewer")
    before_approval = [r for r in records if r.index < first_reviewer]
    assert derive_plan_step_back_state(before_approval, primary=primary).episode is not None
    assert derive_plan_step_back_state(list(records), primary=primary).episode is None

    with patch.object(
        plan_first_loop_module, "log", wraps=plan_first_loop_module.log
    ) as logged:
        assert issue_loop.run_issue_loop(
            resumed, issue_number=56, config=config(), plan_first=True
        ) == 0

    planner = _m936_cpp._agent_prompts(resumed, "claude")
    assert len(planner) == 1
    assert "Orchestrator inherited-matrix check (not a reviewer finding)" in planner[0]
    assert _STEP_BACK_ANCHOR_PLANNER not in planner[0]
    assert not any("step-back anchor" in str(c.args[-1]) for c in logged.call_args_list)
    posted = _plan_round_records(resumed)
    assert sum(1 for r in posted if r.step_back_entries) == 1


def test_staged_direct_child_rerun_after_interrupted_integration_close_finishes_closure(
    tmp_path, monkeypatch, capsys
):
    """Real run_issue_loop entry: a MERGED integration PR with an OPEN staged direct child (#1285)."""
    import coding_review_agent_loop.issue_pr_handoff as handoff_module
    from test_managed_ci_integration_bases import ScriptRunner

    fresh, child, parent = _fresh_child_route_fixture(EXECUTION_DISPOSITION_DIRECT)
    child = replace(child, number=42)

    def forbidden(*_a, **_k):
        raise AssertionError("no agent, memory, dispatch or review may start")

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 42 else parent,
    )
    patch_attr_in(monkeypatch, (discuss_loop, pr_loop, issue_loop), "validate_open_issue", lambda *a, **k: None)
    patch_attr_in(monkeypatch, (discuss_loop, pr_loop, issue_loop), "ensure_agent_workdirs", lambda *a, **k: None)
    patch_attr_in(monkeypatch, (architecture_contract, pr_loop, issue_implementation, issue_loop), "_freeze_prompt_architecture", lambda _r, config: config)
    patch_attr_in(monkeypatch, (execution_policy, child_plan_binding, issue_loop), "_resolve_fresh_child_provenance", lambda **_: fresh)
    patch_attr_in(monkeypatch, (discuss_loop, pr_loop, issue_loop), "prepare_agent_memory", forbidden)
    patch_attr_in(monkeypatch, (plan_first_loop_module, issue_loop), "_dispatch_decomposition_child", forbidden)
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", forbidden)
    monkeypatch.setattr(
        handoff_module, "authenticate_canonical_issue_pr",
        lambda *a, **k: SimpleNamespace(pr_number=7, state="MERGED"),
    )
    runner = ScriptRunner(issue_state="open")
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop", base="refactor/1181")

    assert run_issue_loop(runner, issue_number=42, config=config) == 0
    assert [c for c in runner.commands if c[:3] == ["gh", "issue", "close"]]
    assert not [c for c in runner.commands if "merge" in c and c[:2] == ["gh", "pr"]]
    assert "without replaying the merge" in capsys.readouterr().out
