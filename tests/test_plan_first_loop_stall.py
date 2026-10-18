"""Plan-first loop tests: primary-phase stall stop and streak retirement (#1543)."""

import hashlib
import json
import re
from dataclasses import replace

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.errors import AgentInvocationError
from coding_review_agent_loop.github import HumanReviewRequirement
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.protocol import validate_structured_plan_state
import coding_review_agent_loop.agent_failure as agent_failure
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.integration_close as integration_close
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.panel_evidence as panel_evidence
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.review_rounds as review_rounds
import coding_review_agent_loop.triage as triage
from agent_loop_helpers import patch_attr_in, patch_object_in
from issue_loop_test_helpers import (
    FakeRunner,
    _PlanDiagnosticRunner,
    _invalid_plan_scheduler_comment,
    _m1103_agent_calls,
    _m1103_blocking_chain,
    _m1103_fresh_base,
    _m1103_stalled_runner,
    _m943_decoded,
    _plan_round_records,
    _plan_with_requirement_disposition,
    _staged_plan_config,
)
from coding_review_agent_loop.plan_assembly import AuthenticatedPlanState
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_review,
    structured_v1_plan_state,
)
from coding_review_agent_loop.plan_assembly import assemble_plan_revision  # noqa: E402
import coding_review_agent_loop.validated_agent as validated_agent_module


def test_primary_stall_stop_fires_before_the_next_primary_turn(tmp_path):
    """`stall-stop-threshold`: plain diagnostic, no secondary, no checkpoint."""
    runner, _config, error = _m1103_stalled_runner(tmp_path, threshold=2)

    message = str(error)
    assert "blocked 2 consecutive primary-phase" in message
    assert "--plan-primary-stall-rounds 2" in message
    assert "--plan-review-force-full" in message
    assert "secondary plan panel was not convened" in message
    # Two primary turns, two planner revisions after the fresh plan, and no
    # secondary reviewer ever.
    assert _m1103_agent_calls(runner) == [
        "claude", "codex", "claude", "codex", "claude",
    ]
    records = _plan_round_records(runner)
    prelaunch = [record for record in records if record.phase == "scheduler-prelaunch"]
    assert [record.round_number for record in prelaunch] == [1, 2]
    assert all(record.scheduler_phase == "primary" for record in prelaunch)
    assert not any(record.scheduler_force_full for record in prelaunch)
    diagnostics = [
        comment["body"]
        for comment in runner.issue_comments
        if comment["body"].startswith("Plan review scheduling diagnostic (round 3)")
    ]
    assert len(diagnostics) == 1
    # Plain audit text: no round metadata a resume could read as a checkpoint.
    assert "AGENT_LOOP_META" not in diagnostics[0]


def test_primary_stall_stop_rerun_is_deterministic_and_overridable(tmp_path):
    """`stall-resume-deterministic`: same flags stop again; overrides proceed."""
    runner, config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    history = list(runner.issue_comments)

    rerun = _FakeRunner(issue_comments=list(history))
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="blocked 2"):
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    assert _m1103_agent_calls(rerun) == []
    # The rerun wrote no new checkpoint; only the stopped run's two remain.
    assert [
        record.round_number
        for record in _plan_round_records(rerun)
        if record.phase == "scheduler-prelaunch"
    ] == [1, 2]

    resolved = [{"item_id": "item-2", "disposition": "resolved"}]
    for overrides, expected_phase in (
        ({"plan_primary_stall_rounds": 3}, "primary"),
        ({"plan_primary_stall_rounds": 0}, "primary"),
        ({"plan_review_force_full": True}, "full-board"),
    ):
        proceeding = _FakeRunner(
            issue_comments=list(history),
            codex_outputs=[
                structured_plan_review(
                    state="approved", prior_plan_item_dispositions=resolved
                )
            ],
            gemini_outputs=[
                structured_plan_review(
                    state="approved",
                    reviewer="Google Gemini",
                    prior_plan_item_dispositions=resolved,
                )
            ],
        )
        try:
            run_issue_loop(
                proceeding,
                issue_number=56,
                config=replace(config, **overrides),
                plan_first=True,
            )
        except AgentLoopError:
            pass
        round_three = [
            record
            for record in _plan_round_records(proceeding)
            if record.phase == "scheduler-prelaunch" and record.round_number == 3
        ]
        assert round_three, overrides
        assert round_three[0].scheduler_phase == expected_phase, overrides
        assert "codex" in _m1103_agent_calls(proceeding)
        if expected_phase == "primary":
            # The panel is never auto-convened: round 3 is primary-only.
            assert round_three[0].scheduler_selected_reviewers == ("Codex",)
        else:
            assert round_three[0].scheduler_force_full_source == "operator"


def test_primary_stall_stop_does_not_count_an_interrupted_round(tmp_path):
    """`stall-interrupted-round-not-counted`: the prelaunch checkpoint is not a turn."""
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, 2, base=base)
    runner = _FakeRunner(claude_outputs=[fresh, claude[0]], codex_outputs=[codex[0]])
    config = _staged_plan_config(tmp_path, max_rounds=8, plan_primary_stall_rounds=2)
    real_post = architecture_contract.post_issue_comment

    def interrupt_after_round_two_checkpoint(*args, **kwargs):
        result = real_post(*args, **kwargs)
        metadata = _m943_decoded(kwargs["body"])
        if (
            metadata is not None
            and metadata.phase == "scheduler-prelaunch"
            and metadata.round_number == 2
        ):
            raise KeyboardInterrupt
        return result

    with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment",
        side_effect=interrupt_after_round_two_checkpoint,
    ):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert _m1103_agent_calls(runner) == ["claude", "codex", "claude"]

    runner.codex_outputs = [codex[1]]
    runner.claude_outputs = [claude[1]]
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="blocked 2"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    # The interrupted round's primary turn ran before the stop fired.
    assert _m1103_agent_calls(runner) == ["claude", "codex", "claude", "codex", "claude"]
    assert any(
        comment["body"].startswith("Plan review scheduling diagnostic (round 3)")
        for comment in runner.issue_comments
    )


def test_primary_stall_stop_is_suppressed_across_degraded_history(tmp_path):
    """`stall-degraded-history-boundary`: an invalid record resets the streak."""
    runner, config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    history = list(runner.issue_comments)
    subject = next(
        record.subject
        for record in reversed(_plan_round_records(runner))
        if record.role == "coder"
    )
    # Ordered after every posted comment and before anything the rerun posts.
    from coding_review_agent_loop.round_transport import (
        decode_mapping,
        encode_mapping,
    )

    invalid = _invalid_plan_scheduler_comment(
        subject, created_at=f"2026-05-23T00:00:{len(history):02d}Z"
    )
    payload_text = re.search(r"AGENT_LOOP_META: (\S+) -->", invalid["body"]).group(1)
    payload = dict(decode_mapping(payload_text), round_number=3)
    invalid["body"] = invalid["body"].replace(payload_text, encode_mapping(payload))
    history.append(invalid)

    _fresh, base = _m1103_fresh_base()
    _codex, _claude, base = _m1103_blocking_chain(1, 2, base=base)
    # The degraded round's ledger carries no prior item to disposition.
    codex, claude, _base = _m1103_blocking_chain(3, 4, base=base, resolve_first=False)
    rerun = _FakeRunner(issue_comments=history, codex_outputs=codex, claude_outputs=claude)
    with patch_object_in((agent_failure, integration_close, architecture_contract, validated_agent_module, response_validation, panel_evidence, review_rounds, discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, issue_implementation, plan_first_loop_module, triage, issue_loop), "log",
        wraps=agent_failure.log,
    ) as logged:
        with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="blocked 2"):
            run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    assert any(
        "stall stop suppressed: degraded planning history (invalid)" in str(call.args[-1])
        for call in logged.call_args_list
    )
    # The degraded round ran the conservative primary-only fallback, and the
    # stop needed two further valid blocking rounds after the boundary.
    assert _m1103_agent_calls(rerun) == ["codex", "claude", "codex", "claude"]
    prelaunch = {
        record.round_number: record
        for record in _plan_round_records(rerun)
        if record.phase == "scheduler-prelaunch"
        and record.scheduler_metadata_status == "valid"
    }
    assert prelaunch[3].scheduler_phase == "primary"
    assert prelaunch[3].scheduler_selected_reviewers == ("Codex",)
    assert "strict pre-panel fallback:" in " ".join(prelaunch[3].scheduler_reasons)
    assert prelaunch[4].scheduler_phase == "primary"
    assert 5 not in prelaunch
    assert any(
        comment["body"].startswith("Plan review scheduling diagnostic (round 5)")
        for comment in rerun.issue_comments
    )


def test_primary_approval_at_the_threshold_opens_the_panel(tmp_path):
    """`stall-approval-precedence`: N-1 blocking reviews then an approval."""
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, 2, base=base)
    resolved = [{"item_id": "item-2", "disposition": "resolved"}]
    runner = _FakeRunner(
        claude_outputs=[fresh, *claude],
        codex_outputs=[
            *codex,
            structured_plan_review(state="approved", prior_plan_item_dispositions=resolved),
        ],
        gemini_outputs=[
            structured_plan_review(
                state="approved",
                reviewer="Google Gemini",
                prior_plan_item_dispositions=resolved,
            )
        ],
    )
    config = _staged_plan_config(tmp_path, max_rounds=8, plan_primary_stall_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    prelaunch = [
        record
        for record in _plan_round_records(runner)
        if record.phase == "scheduler-prelaunch"
    ]
    assert [record.scheduler_phase for record in prelaunch] == [
        "primary",
        "primary",
        "primary",
        "secondary-audit",
    ]
    assert prelaunch[3].scheduler_selected_reviewers == ("Gemini",)
    assert prelaunch[2].plan_candidate_key == prelaunch[3].plan_candidate_key
    assert not any(
        comment["body"].startswith("Plan review scheduling diagnostic")
        for comment in runner.issue_comments
    )


def test_growth_gated_primary_approval_resets_the_stall_streak(tmp_path):
    """`stall-growth-gated-approval-resets` through the live loop.

    Rounds 1-2 block and round 3's primary approves under default growth
    thresholds; the run stops before the panel.  The rerun lowers the
    scope-item threshold, so the approved candidate now fails the growth
    gate: round 4 opens no panel and starts with a planner revision.  With a
    threshold of 2, the two earlier blocking reviews would already stop the
    run, so the primary turns in rounds 4 and 5 prove the approval reset the
    streak; the stop fires before round 6.
    """
    fresh, base = _m1103_fresh_base()
    codex, claude, base = _m1103_blocking_chain(1, 2, base=base)
    resolved = [{"item_id": "item-2", "disposition": "resolved"}]
    history_runner = _FakeRunner(
        claude_outputs=[fresh, *claude],
        codex_outputs=[
            *codex,
            structured_plan_review(state="approved", prior_plan_item_dispositions=resolved),
        ],
    )
    with pytest.raises(AgentLoopError):
        run_issue_loop(
            history_runner,
            issue_number=56,
            config=_staged_plan_config(tmp_path, max_rounds=3, plan_primary_stall_rounds=3),
            plan_first=True,
        )
    assert _m1103_agent_calls(history_runner) == [
        "claude", "codex", "claude", "codex", "claude", "codex",
    ]
    history = list(history_runner.issue_comments)
    history_records = _plan_round_records(history_runner)
    assert not any(record.phase == "plan-phase-advance" for record in history_records)

    justify_payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Justify one-shot.",
        "prior_plan_item_dispositions": [],
        "base_round_number": 3,
        "base_state_identity": base.state_identity,
        "operations": [
            {
                "op": "replace",
                "field": "one_shot_growth_justification",
                "value": {
                    "crossed_signals": ["scope-items"],
                    "rationale": "The scope items share one seam and cannot ship separately.",
                },
            }
        ],
    }
    base = AuthenticatedPlanState.from_plan(
        assemble_plan_revision(base, justify_payload), round_number=4
    )
    # Item IDs continue from the history: round 4's finding is item-3.
    codex, claude, _base = _m1103_blocking_chain(
        4, 5, base=base, resolve_first=False, item_offset=1
    )
    # The approved round's ledger still carries item-2 into the next review.
    codex[0] = structured_plan_review(
        state="blocking",
        summary="Gap 4 remains.",
        blocking_plan_issues=["Close gap 4."],
        prior_plan_item_dispositions=resolved,
    )
    rerun = _FakeRunner(
        issue_comments=history,
        claude_outputs=[
            json.dumps(justify_payload)
            + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            *claude,
        ],
        codex_outputs=codex,
    )
    config = _staged_plan_config(
        tmp_path,
        max_rounds=8,
        plan_primary_stall_rounds=2,
        plan_growth_max_scope_items=1,
        # An active step-back episode would defer the stall stop (#1275).
        plan_step_back_rounds=0,
    )
    with patch_object_in((agent_failure, integration_close, architecture_contract, validated_agent_module, response_validation, panel_evidence, review_rounds, discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, issue_implementation, plan_first_loop_module, triage, issue_loop), "log", wraps=agent_failure.log) as logged:
        with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="blocked 2"):
            run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)

    # Round 4 opened with the growth-gated approval: no reviewer and no panel
    # ran on it, only a planner revision carrying the growth notice.
    assert any(
        "the primary approved a candidate that fails the plan-growth gate" in str(call.args[-1])
        for call in logged.call_args_list
    )
    planner_prompts = [cmd[-1] for cmd, _cwd in rerun.commands if cmd[0] == "claude"]
    assert "Orchestrator plan-growth notice" in planner_prompts[0]
    # Two primary turns the pre-approval streak alone would have prevented,
    # and never a secondary.
    assert _m1103_agent_calls(rerun) == ["claude", "codex", "claude", "codex", "claude"]
    new_records = _plan_round_records(rerun)[len(history_records):]
    assert [record.round_number for record in new_records if record.role == "coder"] == [
        4, 5, 6,
    ]
    reviews = [record for record in new_records if record.role == "reviewer"]
    assert [(record.round_number, record.agent, record.state) for record in reviews] == [
        (4, "Codex", "blocking"),
        (5, "Codex", "blocking"),
    ]
    # The growth-gated candidate got no checkpoint of its own: the round-4
    # checkpoint follows the revision, and no panel phase was ever scheduled.
    first_coder = next(i for i, r in enumerate(new_records) if r.role == "coder")
    prelaunch = [
        (index, record)
        for index, record in enumerate(new_records)
        if record.phase == "scheduler-prelaunch"
    ]
    assert [record.round_number for _index, record in prelaunch] == [4, 5]
    assert all(index > first_coder for index, _record in prelaunch)
    assert all(record.scheduler_phase == "primary" for _index, record in prelaunch)
    assert all(
        record.scheduler_selected_reviewers == ("Codex",) for _index, record in prelaunch
    )
    assert not any(record.scheduler_force_full for _index, record in prelaunch)
    assert not any(record.phase == "plan-phase-advance" for record in new_records)
    # The stop fired only after two further blocking primary reviews.
    diagnostics = [
        comment["body"]
        for comment in rerun.issue_comments
        if comment["body"].startswith("Plan review scheduling diagnostic")
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].startswith("Plan review scheduling diagnostic (round 6)")
    assert "AGENT_LOOP_META" not in diagnostics[0]


def test_zero_stall_threshold_runs_to_the_round_budget(tmp_path):
    """0 disables the stop: the run exhausts --max-rounds as before."""
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, 3, base=base)
    runner = _FakeRunner(claude_outputs=[fresh, *claude], codex_outputs=codex)
    config = _staged_plan_config(tmp_path, max_rounds=3, plan_primary_stall_rounds=0)

    with pytest.raises(AgentLoopError, match="still reported blocking plan issues") as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert not isinstance(excinfo.value, plan_first_loop_module.PlanPrePanelSafetyError)
    assert _m1103_agent_calls(runner).count("codex") == 3
    assert "gemini" not in _m1103_agent_calls(runner)


def _m1103_narrative_runner():
    """Primary gate, panel blocker, then a narrative-only recommendation patch."""
    fresh = structured_v1_plan_state()
    base = AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(fresh), round_number=1
    )
    recommendation = json.loads(fresh.split("\n<!--", 1)[0])["execution_recommendation"]
    recommendation = dict(
        recommendation,
        rationale="The reviewed scope is one coherent delivery; reworded.",
        caveats=["A narrative caveat."],
    )
    patch_payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Reword the recommendation rationale.",
        "prior_plan_item_dispositions": [
            {"item_id": "item-1", "disposition": "resolved"}
        ],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {"op": "replace", "field": "execution_recommendation", "value": recommendation}
        ],
    }
    patch_text = (
        json.dumps(patch_payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    return _FakeRunner(
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_plan_review(state="approved", prior_plan_item_dispositions=resolved),
        ],
        gemini_outputs=[
            structured_plan_review(
                state="blocking",
                reviewer="Google Gemini",
                summary="The rationale is unclear.",
                blocking_plan_issues=["Reword the recommendation rationale."],
            ),
            structured_plan_review(
                state="approved",
                reviewer="Google Gemini",
                prior_plan_item_dispositions=resolved,
            ),
        ],
        antigravity_outputs=[
            structured_plan_review(state="approved", reviewer="Google Antigravity"),
            structured_plan_review(
                state="approved",
                reviewer="Google Antigravity",
                prior_plan_item_dispositions=resolved,
            ),
        ],
    )


def _m1103_assert_narrative_remediation(runner):
    prelaunch = {
        record.round_number: record
        for record in _plan_round_records(runner)
        if record.phase == "scheduler-prelaunch"
    }
    remediation = prelaunch[3]
    assert remediation.scheduler_phase == "remediation"
    assert set(remediation.scheduler_selected_reviewers) == {"Gemini", "Codex"}
    assert remediation.scheduler_force_full is False
    assert "narrow plan remediation" in " ".join(remediation.scheduler_reasons)
    # The candidate key still changed, so no approval was carried across it:
    # the final sweep re-covers the secondary lacking an exact-key approval.
    assert remediation.plan_candidate_key != prelaunch[2].plan_candidate_key
    assert prelaunch[4].scheduler_phase == "final-secondary-sweep"
    assert prelaunch[4].scheduler_selected_reviewers == ("Antigravity",)
    assert not any(
        record.scheduler_phase == "full-board" for record in prelaunch.values()
    )


def test_narrative_only_recommendation_patch_stays_owner_scoped_after_the_panel(tmp_path):
    """`postpanel-prose-remediation`: no full board and no automatic latch."""
    runner = _m1103_narrative_runner()
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    _m1103_assert_narrative_remediation(runner)


def test_resume_after_a_narrative_only_patch_reconstructs_the_narrow_classification(tmp_path):
    """`resume-reconstructs-narrow`: the restart classifies as the live run does."""
    runner = _m1103_narrative_runner()
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8
    )
    real_post = architecture_contract.post_issue_comment
    audits = {"count": 0}

    def interrupt_before_remediation_checkpoint(*args, **kwargs):
        if kwargs["body"].startswith("Plan review scheduling audit."):
            audits["count"] += 1
            if audits["count"] == 3:
                raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment",
        side_effect=interrupt_before_remediation_checkpoint,
    ):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    _m1103_assert_narrative_remediation(runner)


# --- Stall streak retirement: issue-text edit and reset flag (#1112) ---------


def _m1112_primary_checkpoints(runner):
    return [
        record
        for record in _plan_round_records(runner)
        if record.phase == "scheduler-prelaunch" and record.scheduler_phase == "primary"
    ]


def _m1112_rerun(history, config, *, issue_payload=None, comments=(), **runner_kwargs):
    runner = _FakeRunner(
        issue_comments=[*history, *comments],
        issue_payload=issue_payload,
        **runner_kwargs,
    )
    error = None
    try:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    except AgentLoopError as exc:
        error = exc
    return runner, error


def _m1112_approvals():
    resolved = [{"item_id": "item-2", "disposition": "resolved"}]
    return {
        "codex_outputs": [
            structured_plan_review(state="approved", prior_plan_item_dispositions=resolved)
        ],
        "gemini_outputs": [
            structured_plan_review(
                state="approved",
                reviewer="Google Gemini",
                prior_plan_item_dispositions=resolved,
            )
        ],
    }


def test_stalled_checkpoints_record_the_issue_digest(tmp_path):
    runner, _config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    digests = {record.scheduler_issue_digest for record in _m1112_primary_checkpoints(runner)}
    assert len(digests) == 1 and re.fullmatch(r"[0-9a-f]{16}", digests.pop())
    assert not any(r.scheduler_stall_reset for r in _plan_round_records(runner))


def test_issue_edit_retires_the_stalled_streak(tmp_path):
    """`edit-retires-streak`, under the keep-the-plan opt-out (#1390)."""
    runner, config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    history = list(runner.issue_comments)
    old_digest = _m1112_primary_checkpoints(runner)[0].scheduler_issue_digest

    rerun, _error = _m1112_rerun(
        history,
        replace(config, plan_resume_keep_on_issue_edit=True),
        issue_payload={"body": "Narrowed issue body with explicit non-goals."},
        **_m1112_approvals(),
    )
    assert "codex" in _m1103_agent_calls(rerun)
    new_checkpoints = [
        record
        for record in _m1112_primary_checkpoints(rerun)
        if record.round_number == 3
    ]
    assert new_checkpoints
    assert new_checkpoints[0].scheduler_issue_digest not in (None, old_digest)
    assert not any(r.scheduler_force_full for r in _plan_round_records(rerun))
    assert not any(r.scheduler_stall_reset for r in _plan_round_records(rerun))


def test_issue_edit_with_blocking_primary_opens_no_panel(tmp_path):
    """`edit-retires-streak`: retirement itself never opens the panel (opt-out, #1390)."""
    runner, config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    config = replace(config, plan_resume_keep_on_issue_edit=True)
    history = list(runner.issue_comments)
    _fresh, base = _m1103_fresh_base()
    _codex, _claude, base = _m1103_blocking_chain(1, 2, base=base)
    codex, claude, _base = _m1103_blocking_chain(3, 4, base=base)

    rerun = _FakeRunner(
        issue_comments=list(history),
        issue_payload={"body": "Narrowed issue body with explicit non-goals."},
        codex_outputs=codex,
        claude_outputs=claude,
    )
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="blocked 2"):
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    # Two new blocking rounds ran under the new text, then the stop re-tripped.
    assert _m1103_agent_calls(rerun) == ["codex", "claude", "codex", "claude"]
    records = _plan_round_records(rerun)
    assert not any(r.scheduler_force_full for r in records)
    assert not any(r.scheduler_stall_reset for r in records)
    assert not any(
        r.scheduler_phase in {"secondary-audit", "remediation", "final-secondary-sweep", "full-board"}
        for r in records
    )
    assert not any(
        r.role == "reviewer" and r.agent != "Codex" for r in records
    )


def test_unchanged_issue_with_new_comment_still_stops(tmp_path):
    """`unchanged-rerun-stops`."""
    runner, config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    history = list(runner.issue_comments)
    before = len(_plan_round_records(runner))

    rerun, error = _m1112_rerun(
        history,
        config,
        comments=[{"author": "operator", "body": "Please narrow the scope."}],
    )
    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert _m1103_agent_calls(rerun) == []
    assert "a comment alone does not" in str(error)
    assert len(_plan_round_records(rerun)) == before


def test_legacy_undigested_rounds_keep_counting_and_name_the_reset_flag(tmp_path):
    """`legacy-undigested`."""
    with patch_object_in((panel_evidence, plan_first_loop_module, issue_loop), "plan_issue_text_digest", lambda _ctx: None):
        runner, config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    history = list(runner.issue_comments)
    assert all(
        record.scheduler_issue_digest is None for record in _m1112_primary_checkpoints(runner)
    )

    rerun, error = _m1112_rerun(
        history, config, issue_payload={"body": "Edited after the stall."}
    )
    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert _m1103_agent_calls(rerun) == []
    assert "predate issue-text tracking" in str(error)
    assert "--plan-reset-stall-streak" in str(error)


def _m1112_crash_at_checkpoint(history, config, *, after_post):
    """A flagged run killed at its first scheduler checkpoint post."""
    runner = _FakeRunner(issue_comments=list(history), **_m1112_approvals())
    real_post = architecture_contract.post_issue_comment

    def crashing_post(*args, **kwargs):
        if "Plan review scheduling audit" in str(kwargs.get("body", "")):
            if after_post:
                real_post(*args, **kwargs)
            raise KeyboardInterrupt("simulated crash at the prelaunch checkpoint")
        return real_post(*args, **kwargs)

    with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment", crashing_post):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return runner


def _m1112_no_panel_or_force_full(runner):
    records = _plan_round_records(runner)
    assert not any(r.scheduler_force_full for r in records)
    assert not any(
        r.scheduler_phase in {"secondary-audit", "full-board"} for r in records
    )
    assert "gemini" not in _m1103_agent_calls(runner)


def test_reset_flag_durably_retires_the_streak(tmp_path):
    """`reset-flag-durable`: the reset checkpoint bounds later flagless reruns."""
    runner, config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    history = list(runner.issue_comments)
    _fresh, base = _m1103_fresh_base()
    _codex, _claude, base = _m1103_blocking_chain(1, 2, base=base)
    codex, claude, _base = _m1103_blocking_chain(3, 4, base=base)

    flagged = _FakeRunner(
        issue_comments=list(history), codex_outputs=codex, claude_outputs=claude
    )
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="blocked 2"):
        run_issue_loop(
            flagged,
            issue_number=56,
            config=replace(config, plan_reset_stall_streak=True),
            plan_first=True,
        )
    # The flagged run gets rounds 3 and 4 (two new blocking rounds), and only
    # the invocation's first primary checkpoint carries the marker.
    assert _m1103_agent_calls(flagged) == ["codex", "claude", "codex", "claude"]
    marked = [r for r in _plan_round_records(flagged) if r.scheduler_stall_reset]
    assert [r.round_number for r in marked] == [3]
    assert marked[0].scheduler_phase == "primary"
    _m1112_no_panel_or_force_full(flagged)

    # A flagless rerun of that history stops again only because N new blocking
    # rounds followed the reset checkpoint, never the two older ones.
    later = _FakeRunner(issue_comments=list(flagged.issue_comments))
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="blocked 2"):
        run_issue_loop(later, issue_number=56, config=config, plan_first=True)
    assert _m1103_agent_calls(later) == []


def test_reset_interrupted_before_checkpoint_stops_again(tmp_path):
    """`reset-interrupted-before-checkpoint`: nothing durable, still stops."""
    runner, config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    history = list(runner.issue_comments)
    crashed = _m1112_crash_at_checkpoint(
        history, replace(config, plan_reset_stall_streak=True), after_post=False
    )
    assert crashed.issue_comments == history
    assert not any(r.scheduler_stall_reset for r in _plan_round_records(crashed))

    rerun, error = _m1112_rerun(crashed.issue_comments, config)
    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert "blocked 2" in str(error)
    assert _m1103_agent_calls(rerun) == []


def test_reset_interrupted_after_checkpoint_resumes_the_primary_review(tmp_path):
    """`reset-interrupted-after-checkpoint`: the posted marker is a boundary."""
    runner, config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    history = list(runner.issue_comments)
    crashed = _m1112_crash_at_checkpoint(
        history, replace(config, plan_reset_stall_streak=True), after_post=True
    )
    marked = [r for r in _plan_round_records(crashed) if r.scheduler_stall_reset]
    assert [r.round_number for r in marked] == [3]
    assert _m1103_agent_calls(crashed) == []

    rerun, error = _m1112_rerun(crashed.issue_comments, config, **_m1112_approvals())
    assert not (
        isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
        and "blocked" in str(error)
    )
    assert "codex" in _m1103_agent_calls(rerun)
    # No second reset marker and no panel or force-full before primary approval.
    assert [
        r.round_number for r in _plan_round_records(rerun) if r.scheduler_stall_reset
    ] == [3]
    records = _plan_round_records(rerun)
    assert not any(r.scheduler_force_full for r in records)
    # The resumed round runs primary-only; the panel opens only after the
    # primary's own approval, never on the reset.
    round_three = [
        r for r in records
        if r.phase == "scheduler-prelaunch" and r.round_number == 3
    ]
    assert round_three
    assert all(r.scheduler_phase == "primary" for r in round_three)


def test_all_reviewers_policy_writes_no_digest_or_reset_fields(tmp_path):
    fresh = structured_v1_plan_state()
    runner = _FakeRunner(
        claude_outputs=[fresh],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    try:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    except AgentLoopError:
        pass
    records = _plan_round_records(runner)
    assert not any(
        r.scheduler_issue_digest is not None or r.scheduler_stall_reset for r in records
    )


def _matrix_plan_candidate_1229(**row_overrides):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    row = {
        "row_id": "row-a", "label": "L", "entry_path_or_mode": "m", "initial_state": "i",
        "event": "e", "expected_outcome": "o",
        "forbidden_side_effects": ["Must not refuse"],
        "proposed_test_level": "unit", "proposed_test_location": "tests/x.py",
        "applicability": "required", "related_scope_item_ids": ["scope-1"],
        "execution_owner": "one-shot",
    }
    row.update(row_overrides)
    payload["risk_test_matrix"] = {
        "applicability": "applicable", "rows": [row], "important_exclusions": [],
    }
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"


def _claude_prompts_1229(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def test_normalized_remaining_defect_is_quoted_and_persisted(tmp_path):
    bad = _matrix_plan_candidate_1229(forbidden_side_effects="Must not refuse", related_scope_item_ids="")
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [bad, bad]
    config = make_config(tmp_path, agent_max_retries=1, execution_strategy_contract_required=True)
    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    exhaustion = error.value.plan_validation_exhaustion
    assert exhaustion is not None
    assert "related_scope_item_ids" in exhaustion.diagnostic
    assert "forbidden_side_effects" not in exhaustion.diagnostic
    assert "one automatic planner replay" in str(error.value)
    assert '"forbidden_side_effects": ["Must not refuse"]' in exhaustion.candidate_text
    independent = _matrix_plan_candidate_1229(
        forbidden_side_effects=["Must not refuse"], related_scope_item_ids=""
    )
    assert exhaustion.candidate_text == independent
    assert exhaustion.candidate_digest == hashlib.sha256(independent.encode()).hexdigest()
    assert len(runner.diagnostic_posts) == 1
    assert len(_claude_prompts_1229(runner)) == 2
    replay_prompt = "\n".join(_claude_prompts_1229(runner)[1])
    assert "Previous response not accepted: risk_test_matrix" in replay_prompt
    assert "related_scope_item_ids" in replay_prompt
    assert "forbidden_side_effects must be a JSON array" not in replay_prompt
    assert error.value.failure_category == "fresh-contract-integrity"


def _issue_level_run_1229(tmp_path, outputs, **config_overrides):
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = list(outputs)
    config = make_config(
        tmp_path, execution_strategy_contract_required=True, quiet=False, **config_overrides
    )
    repair_calls = []
    real_repair = validated_agent_module._run_structured_repair

    def spy(*args, **kwargs):
        repair_calls.append(args)
        return real_repair(*args, **kwargs)

    return runner, config, repair_calls, spy


def _run_issue_level_1229(tmp_path, monkeypatch, outputs, retries=0):
    runner, config, repair_calls, spy = _issue_level_run_1229(
        tmp_path, outputs, agent_max_retries=retries
    )
    posted = []
    real_post = architecture_contract.post_issue_comment

    def capture(_runner, *, config, issue_number, body):
        posted.append(str(body))
        return real_post(_runner, config=config, issue_number=issue_number, body=body)

    patch_attr_in(monkeypatch, (validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair", spy)
    patch_attr_in(monkeypatch, (architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment", capture)
    # Later stages have no scripted output; only the planning turn is under test.
    with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return runner, repair_calls, posted


def _posted_matrix_1229(body):
    import re
    from coding_review_agent_loop.comment_rendering import decode_risk_test_matrix_marker

    match = re.search(r"<!-- AGENT_RISK_TEST_MATRIX: ([A-Za-z0-9+/=_-]+) -->", body)
    assert match, "published plan carries no canonical matrix"
    return decode_risk_test_matrix_marker(match.group(1), bodies=[body])["matrix"]


def test_issue_level_bare_strings_are_accepted_on_first_planner_call(tmp_path, monkeypatch, capsys):
    good = _matrix_plan_candidate_1229(
        forbidden_side_effects="Slip one", related_scope_item_ids="scope-1"
    )
    payload = json.loads(good.split("\n<!--", 1)[0])
    payload["risk_test_matrix"]["important_exclusions"] = "Lone exclusion"
    good = json.dumps(payload) + good[len(good.split("\n<!--", 1)[0]):]
    runner, repair_calls, posted = _run_issue_level_1229(tmp_path, monkeypatch, [good])
    assert len(_claude_prompts_1229(runner)) == 1
    assert repair_calls == []
    assert posted
    matrix = _posted_matrix_1229(posted[0])
    assert matrix["rows"][0]["forbidden_side_effects"] == ["Slip one"]
    assert matrix["rows"][0]["related_scope_item_ids"] == ["scope-1"]
    assert matrix["important_exclusions"] == ["Lone exclusion"]
    err = capsys.readouterr().err
    assert "normalized risk test matrix string field(s)" in err
    for path in ("forbidden_side_effects", "related_scope_item_ids", "important_exclusions"):
        assert path in err


def test_issue_level_unrecoverable_matrix_replays_once_and_is_accepted(tmp_path, monkeypatch, capsys):
    from coding_review_agent_loop.protocol import validate_structured_plan_state

    bad = _matrix_plan_candidate_1229(forbidden_side_effects="")
    with pytest.raises(AgentLoopError) as expected:
        validate_structured_plan_state(
            bad, require_execution_strategy_contract=1, require_risk_test_matrix_contract=1
        )
    good = _matrix_plan_candidate_1229(forbidden_side_effects=["Replay accepted"])
    runner, repair_calls, posted = _run_issue_level_1229(tmp_path, monkeypatch, [bad, good])
    prompts = ["\n".join(c) for c in _claude_prompts_1229(runner)]
    assert len(prompts) == 2
    heading = "Previous response not accepted: risk_test_matrix"
    assert heading not in prompts[0] and heading in prompts[1]
    appended = prompts[1].split(heading, 1)[1]
    assert str(expected.value) in appended
    err = capsys.readouterr().err
    assert err.count("repair backend=none model=fresh-matrix-contract-integrity") == 1
    assert err.count("repair backend=") == 1
    # Only the deterministic guard ran (backend=none); no repair model rewrote the matrix.
    assert len(repair_calls) == 1
    assert posted
    assert _posted_matrix_1229(posted[0])["rows"][0]["forbidden_side_effects"] == ["Replay accepted"]


def test_issue_loop_plan_first_resume_reparses_alias_keyed_requirement_disposition(tmp_path):
    """#1241: a stored review accepted through the `requirement_label` alias resumes."""
    issue_body = "Keep the public API unchanged.\n\n-- Human Reviewer"
    requirement = HumanReviewRequirement(
        source_type="Issue body",
        author=None,
        created_at=None,
        url="https://github.com/OWNER/REPO/issues/56",
        body="Keep the public API unchanged.",
    )
    requirement_id = requirement.requirement_id
    current_plan = _plan_with_requirement_disposition(
        kind="plan_state",
        summary="Keep the public API unchanged.",
        plan_steps=["Keep the public API unchanged."],
        evidence="The plan keeps the public API unchanged.",
    ).replace('"Requirement 1"', f'"{requirement_id}"').replace("- Requirement 1:", f"- Requirement {requirement_id}:")
    subject = _plan_subject(current_plan)
    aliased_review = (
        structured_plan_review(state="approved", summary="Plan covers the requirement.").replace(
            '"human_requirement_dispositions": []',
            json.dumps(
                {
                    "human_requirement_dispositions": [{
                        "requirement_label": requirement_id,
                        "disposition": "addressed",
                        "evidence": "The plan keeps the public API unchanged.",
                    }]
                }
            )[1:-1],
        )
    )
    aliased_review = aliased_review.replace(
        "\n<!-- AGENT_PLAN_STATE: approved -->",
        "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_PLAN_STATE: approved -->",
    )
    assert "requirement_label" in aliased_review
    assert "HUMAN_REQUIREMENTS_RESOLVED" in aliased_review
    coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=1, subject=subject, prior_items=()
        ),
    )
    codex_comment = _attach_round_metadata(
        aliased_review,
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=subject,
            state="approved",
            canonical_reviewer_response=aliased_review,
        ),
    )
    runner = FakeRunner(
        issue_payload={"body": issue_body},
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": codex_comment},
        ],
    )
    config = make_config(tmp_path, reviewer=("codex",), plan_execution_mode="plan-only")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    agent_commands = [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert agent_commands == []
