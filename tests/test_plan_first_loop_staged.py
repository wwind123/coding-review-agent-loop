"""Plan-first loop tests: staged parents and child stages (#1543)."""

import json
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

# A private response root per test: the shared $TMPDIR one races across
# xdist workers (#1653 CI).
pytestmark = pytest.mark.usefixtures("private_response_tempdir")

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
import coding_review_agent_loop.phase_progress as phase_progress_module
from coding_review_agent_loop.phase_progress import StagedTopologyOutcome
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.decomposition import (
    CreatedPhaseIssue,
    PhaseImplementationHandoffMetadata,
    RecordedPhase,
    RetainedParentScope,
)
from coding_review_agent_loop.github import IssueContext
from coding_review_agent_loop.protocol import (
    ExecutionAllocation,
    validate_structured_plan_state,
)
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in
from issue_loop_test_helpers import (
    _STEP_BACK_MARKER,
    _m1103_agent_calls,
    _m1103_blocking_chain,
    _m1103_fresh_base,
    _m1103_stalled_runner,
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
    child_pr_handoff_comment,
    make_config,
    pr_payload_for_state,
    structured_plan_review,
    structured_v1_plan_state,
)
import test_child_plan_provenance as _m936_cpp  # noqa: E402
from coding_review_agent_loop.plan_assembly import assemble_plan_revision  # noqa: E402
from coding_review_agent_loop.round_transport import decode_mapping  # noqa: E402


# --- Staged parent terminal state and human stages (#918) -----------------


def _staged_created(automations):
    return tuple(
        CreatedPhaseIssue(
            phase=RecordedPhase(title=f"Stage {index}", automation=automation),
            issue_url=f"https://github.com/OWNER/REPO/issues/{98 + index}",
            issue_number=98 + index,
        )
        for index, automation in enumerate(automations, start=1)
    )


def _staged_outcome(created, *, retained=None, final=None, plan_hash="staged-plan-hash"):
    return StagedTopologyOutcome(
        created=created,
        stage_ids=tuple(f"stage-{index}" for index in range(1, len(created) + 1)),
        automations=tuple(item.phase.automation for item in created),
        plan_hash=plan_hash,
        mode="implement-by-phase",
        topology_source="model",
        retained_parent_scope=retained,
        final_integration_work=final,
    )


def _staged_handoff(phase_index, child_issue_number, *, plan_hash="staged-plan-hash"):
    return PhaseImplementationHandoffMetadata(
        parent_issue=55,
        plan_hash=plan_hash,
        mode="implement-by-phase",
        phase_index=phase_index,
        phase_title=f"Stage {phase_index}",
        automation="agent-pr",
        child_issue_number=child_issue_number,
        child_issue_url=f"https://github.com/OWNER/REPO/issues/{child_issue_number}",
        stage_id=f"stage-{phase_index}",
    )


def _merged_child_runner(states, *, pr_states=None):
    """FakeRunner presenting per-child issue states and canonical merged PRs."""
    pr_states = pr_states or {}
    issue_payloads = {}
    issue_comments = {}
    pr_payloads = {}
    for number, state in states.items():
        issue_payloads[number] = {"state": state}
        pr_number = pr_states.get(number)
        if pr_number is None:
            issue_comments[number] = []
            continue
        issue_comments[number] = [child_pr_handoff_comment(number, pr_number)]
        pr_payloads[pr_number] = pr_payload_for_state(pr_number, "MERGED")
    return _FakeRunner(
        issue_payloads_by_number=issue_payloads,
        issue_comments_by_number=issue_comments,
        pr_payloads_by_number=pr_payloads,
    )


def _dispatch_staged(runner, tmp_path, outcome, handoffs, monkeypatch, **overrides):
    monkeypatch.setattr(
        phase_progress_module,
        "find_phase_implementation_handoffs_for_parent",
        lambda *_args, **_kwargs: tuple(handoffs),
    )
    parent = IssueContext(
        number=55, repo="OWNER/REPO", title="Parent", body="Parent",
        url="https://github.com/OWNER/REPO/issues/55", comments=(),
    )
    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context",
        lambda _runner, *, config, issue_number: parent,
    )
    kwargs = dict(
        config=make_config(tmp_path), memory=None, usage_context=SimpleNamespace(),
        issue_number=55, current_plan="Approved staged plan", plan_subject="plan-subject",
        outcome=outcome, recommendation=None,
        approved_plan_context=SimpleNamespace(matrix_available=False, risk_test_matrix_payload=None),
        issue_context=parent, mode="implement-by-phase", coder_session_id=None,
    )
    kwargs.update(overrides)
    return plan_first_loop_module._dispatch_current_decomposition_phase(runner, **kwargs)


def test_staged_terminal_report_names_required_parent_obligations(tmp_path, monkeypatch, capsys):
    """Matrix row `terminal-required-obligations`."""
    created = _staged_created(("agent-pr", "agent-pr"))
    outcome = _staged_outcome(
        created,
        retained=RetainedParentScope(
            plan_subject="plan-subject", plan_hash="staged-plan-hash",
            excerpt="Parent keeps the rollout note.", status="required",
            deliverables=("The rollout note.",),
            acceptance_criteria=("The rollout note is published.",),
        ),
        final=ExecutionAllocation(
            status="required", deliverables=("Wire the stages together.",),
            acceptance_criteria=("End-to-end behavior passes.",),
            covered_scope_item_ids=("scope-3",),
        ),
    )
    runner = _merged_child_runner(
        {99: "closed", 100: "closed"}, pr_states={99: 912, 100: 913}
    )
    patch_attr_in(monkeypatch, (plan_first_loop_module, issue_loop), "_dispatch_decomposition_child",
        lambda *_a, **_k: pytest.fail("a delivered topology must not dispatch a child"),
    )

    result = _dispatch_staged(
        runner, tmp_path, outcome,
        (_staged_handoff(1, 99), _staged_handoff(2, 100)), monkeypatch,
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "all 2 staged phases are delivered" in output
    assert "Retained-parent obligations: required; this is operator-owned parent work." in output
    assert "The rollout note." in output
    assert "Final-integration obligations: required" in output
    assert "Wire the stages together." in output
    assert "End-to-end behavior passes." in output
    assert "remains open pending that operator-owned parent work" in output
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_staged_terminal_report_renders_a_human_stage_without_a_pr(tmp_path, monkeypatch, capsys):
    """Matrix row `terminal-human-stage-line`."""
    created = _staged_created(("agent-pr", "human-action"))
    outcome = _staged_outcome(created)
    runner = _merged_child_runner({99: "closed", 100: "closed"}, pr_states={99: 912})

    result = _dispatch_staged(runner, tmp_path, outcome, (_staged_handoff(1, 99),), monkeypatch)

    assert result == 0
    output = capsys.readouterr().out
    assert "stage-1: delivered by child issue #99 with merged PR #912" in output
    assert (
        "stage-2: delivered by human work on child issue #100; "
        "its closure is the operator attestation." in output
    )
    assert "PR #None" not in output
    # No PR evidence is read for the human-owned stage.
    assert not any(
        cmd[:3] == ["gh", "pr", "view"] and "913" in cmd for cmd, _cwd in runner.commands
    )


def test_staged_pending_human_stage_stops_then_advances_when_closed(tmp_path, monkeypatch, capsys):
    """Matrix rows `human-stage-pending-stops` and `closed-human-stage-advances`."""
    created = _staged_created(("agent-pr", "human-action", "agent-pr"))
    outcome = _staged_outcome(created)
    handoffs = (_staged_handoff(1, 99),)
    patch_attr_in(monkeypatch, (plan_first_loop_module, issue_loop), "_dispatch_decomposition_child",
        lambda *_a, **_k: pytest.fail("a human-owned stage must not dispatch an agent"),
    )
    pending = _merged_child_runner({99: "closed", 100: "open"}, pr_states={99: 912})

    assert _dispatch_staged(pending, tmp_path, outcome, handoffs, monkeypatch) == 0
    output = capsys.readouterr().out
    assert "phase 2 (`stage-2`) requires human work (human-action) on child issue #100" in output
    assert not any(
        "AGENT_PLAN_PHASE_IMPLEMENTATION" in comment for comment in pending.comments
    )

    dispatched = []
    patch_attr_in(monkeypatch, (plan_first_loop_module, issue_loop), "_dispatch_decomposition_child",
        lambda *_a, **kwargs: dispatched.append(kwargs) or 0,
    )
    advanced = _merged_child_runner(
        {99: "closed", 100: "closed", 101: "open"}, pr_states={99: 912}
    )

    assert _dispatch_staged(advanced, tmp_path, outcome, handoffs, monkeypatch) == 0
    assert len(dispatched) == 1
    assert dispatched[0]["phase_index"] == 3
    assert dispatched[0]["created"].issue_number == 101


def _m936_historical_approved_plan(tmp_path, *, reviewers_kwargs=None):
    """A weakened child plan approved before the inherited check existed."""
    weak = _m936_cpp._child_plan_state(_m936_cpp._weak_child_row())
    first = _m936_cpp._ChildPlanningRunner(
        claude_outputs=[weak],
        codex_outputs=[_m936_cpp.structured_plan_review(state="approved")],
    )
    assert issue_loop.run_issue_loop(
        first, issue_number=56, config=_m936_cpp._plan_config(tmp_path), plan_first=True
    ) == 0
    return weak, list(first.issue_comments)


def test_m936_approved_inadmissible_plan_is_revised_instead_of_dead_ending(tmp_path, monkeypatch):
    weak, history = _m936_historical_approved_plan(tmp_path)
    _m936_cpp._bind_child_planning(monkeypatch)
    good = _m936_patch(weak, _m936_cpp._child_row(), summary="Inherited rows restored.")
    resumed = _m936_cpp._ChildPlanningRunner(
        issue_comments=list(history),
        claude_outputs=[good],
        codex_outputs=[_m936_cpp.structured_plan_review(state="approved")],
    )
    assert issue_loop.run_issue_loop(
        resumed, issue_number=56, config=_m936_cpp._plan_config(tmp_path), plan_first=True
    ) == 0

    posted = [str(item["body"]) for item in resumed.issue_comments[len(history):]]
    audits = [body for body in posted if "is inadmissible under the inherited-matrix contract" in body]
    assert len(audits) == 1 and "AGENT_LOOP_META" not in audits[0]
    planner_prompts = _m936_cpp._agent_prompts(resumed, "claude")
    assert len(planner_prompts) == 1
    assert "Trusted orchestration correction record" in planner_prompts[0]
    assert "row-owned: forbidden_side_effects dropped" in planner_prompts[0]
    assert "Orchestrator inherited-matrix check (not a reviewer finding)" in planner_prompts[0]
    # No synthetic reviewer item and no signed record or digest is involved.
    revised = [body for body in posted if "Inherited rows restored." in body]
    assert revised
    from coding_review_agent_loop.round_state import _decode_round_metadata
    from coding_review_agent_loop.round_transport import ROUND_RESUME_MARKER_RE

    metadata = _decode_round_metadata(ROUND_RESUME_MARKER_RE.search(revised[0]).group("payload"))
    assert metadata.round_number == 2 and metadata.prior_items == ()
    assert metadata.plan_supersession_digest is None
    # The revised plan was reviewed again by the board and only then approved.
    assert len(_m936_cpp._agent_prompts(resumed, "codex")) == 1
    assert "Inherited coverage check FAILED" not in _m936_cpp._agent_prompts(resumed, "codex")[0]


def test_m936_guard_audit_comment_is_not_duplicated_and_recheck_exhaustion_persists(
    tmp_path, monkeypatch
):
    weak, history = _m936_historical_approved_plan(tmp_path)
    _m936_cpp._bind_child_planning(monkeypatch)
    still_weak = _m936_patch(weak, None, summary=_m936_cpp.WEAK_SUMMARY)
    resumed = _m936_cpp._ChildPlanningRunner(
        issue_comments=list(history), claude_outputs=[still_weak] * 3,
    )
    with pytest.raises(AgentLoopError, match="consecutive plan candidates that weaken"):
        issue_loop.run_issue_loop(
            resumed, issue_number=56, config=_m936_cpp._plan_config(tmp_path), plan_first=True
        )
    assert len(_m936_cpp._agent_prompts(resumed, "claude")) == 3
    assert _m936_cpp.WEAK_SUMMARY not in _m936_cpp._published(resumed)
    assert len(resumed.diagnostic_posts) == 1
    # A rerun finds the audit comment and does not post it again.
    rerun = _m936_cpp._ChildPlanningRunner(
        issue_comments=list(resumed.issue_comments), claude_outputs=[still_weak] * 3,
    )
    with pytest.raises(AgentLoopError):
        issue_loop.run_issue_loop(
            rerun, issue_number=56, config=_m936_cpp._plan_config(tmp_path), plan_first=True
        )
    audits = [
        item for item in rerun.issue_comments
        if "is inadmissible under the inherited-matrix contract" in str(item["body"])
    ]
    assert len(audits) == 1


def test_m936_guard_at_max_rounds_names_the_round_budget(tmp_path, monkeypatch):
    _weak, history = _m936_historical_approved_plan(tmp_path)
    _m936_cpp._bind_child_planning(monkeypatch)
    resumed = _m936_cpp._ChildPlanningRunner(issue_comments=list(history))
    config = make_config(
        tmp_path, max_rounds=1, plan_execution_mode="plan-only",
        execution_strategy_contract_required=True,
    )
    with pytest.raises(AgentLoopError, match="Raise --max-rounds"):
        issue_loop.run_issue_loop(resumed, issue_number=56, config=config, plan_first=True)
    assert _m936_cpp._agent_prompts(resumed, "claude") == []
    assert len(resumed.issue_comments) == len(history)


_M936_STAGED = dict(
    reviewer=("codex", "gemini"), plan_review_policy="primary-then-panel",
    primary_plan_reviewer="codex",
)


def test_m936_guard_full_board_latch_survives_a_restart_after_the_revised_round(
    tmp_path, monkeypatch
):
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    def staged_config():
        return make_config(
            tmp_path, max_rounds=8, plan_execution_mode="plan-only",
            execution_strategy_contract_required=True, **_M936_STAGED,
        )

    gemini_approval = _m936_cpp.structured_plan_review(state="approved", reviewer="Google Gemini")
    weak = _m936_cpp._child_plan_state(_m936_cpp._weak_child_row())
    first = _m936_cpp._ChildPlanningRunner(
        claude_outputs=[weak],
        codex_outputs=[_m936_cpp.structured_plan_review(state="approved")],
        gemini_outputs=[gemini_approval],
    )
    assert issue_loop.run_issue_loop(
        first, issue_number=56, config=staged_config(), plan_first=True
    ) == 0
    _m936_cpp._bind_child_planning(monkeypatch)
    good = _m936_patch(weak, _m936_cpp._child_row(), summary="Inherited rows restored.")
    # The run stops right after the revised round: no reviewer output is scripted.
    interrupted = _m936_cpp._ChildPlanningRunner(
        issue_comments=list(first.issue_comments), claude_outputs=[good],
    )
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        issue_loop.run_issue_loop(
            interrupted, issue_number=56, config=staged_config(), plan_first=True
        )
    comments = [
        _m936_cpp.comment(str(item["body"])) for item in interrupted.issue_comments
    ]
    last_coder = max(
        record.index for record in _extract_round_metadata_records(comments, flow="plan")
        if record.metadata.role == "coder"
    )
    durable = list(interrupted.issue_comments[: last_coder + 1])
    assert child_plan_binding._resumed_inherited_replan_force_full(comments[: last_coder + 1])
    # No digest is involved on this path: the audit comment is the durable key.
    assert _extract_round_metadata_records(
        comments, flow="plan"
    )[-1].metadata.plan_supersession_digest is None

    restarted = _m936_cpp._ChildPlanningRunner(
        issue_comments=durable,
        codex_outputs=[_m936_cpp.structured_plan_review(state="approved")],
        gemini_outputs=[gemini_approval],
    )
    assert issue_loop.run_issue_loop(
        restarted, issue_number=56, config=staged_config(), plan_first=True
    ) == 0
    assert _m936_cpp._agent_prompts(restarted, "claude") == []
    audits = [
        str(item["body"]) for item in restarted.issue_comments[len(durable):]
        if "Plan review scheduling audit" in str(item["body"])
    ]
    assert "Force-full: True (source: automatic)" in audits[0]
    assert "Selected reviewers: Codex, Gemini" in audits[0]
    # An ordinary revision round reconstructs no latch.
    assert not child_plan_binding._resumed_inherited_replan_force_full(
        [_m936_cpp.comment(str(item["body"])) for item in first.issue_comments]
    )


def _m1268_base():
    return AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(_m1268_staged_raw(), require_execution_strategy_contract=1),
        round_number=1,
    )


def _m1268_prompts(runner, agent):
    return [cmd[-1] for cmd, _cwd in runner.commands if cmd[0] == agent]


@pytest.mark.parametrize("policy", ["all-reviewers", "primary-then-panel"])
def test_staged_plan_under_plan_only_stops_on_round_one(tmp_path, policy):
    """`plan-only-staged-fresh`: no reviewer, revision, or checkpoint turn."""
    runner = _FakeRunner(claude_outputs=[_m1268_staged_raw()])
    config = (
        _staged_plan_config(tmp_path, max_rounds=8, plan_primary_stall_rounds=8)
        if policy == "primary-then-panel"
        else make_config(tmp_path, reviewer=("codex", "gemini"))
    )

    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    message = str(excinfo.value)
    assert "staged plan under plan-only" in message
    assert "decompose-only" in message
    assert "--plan-narrow-staged" in message
    assert _m1103_agent_calls(runner) == ["claude"]
    assert not [r for r in _plan_round_records(runner) if r.role == "reviewer"]
    assert not [r for r in _plan_round_records(runner) if r.phase == "scheduler-prelaunch"]


def test_staged_plan_under_implement_one_shot_stops_with_the_phase_remedies(tmp_path):
    """`one-shot-mode-staged`."""
    runner = _FakeRunner(claude_outputs=[_m1268_staged_raw()])
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), plan_execution_mode="implement-one-shot"
    )

    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert "implement-by-phase" in str(excinfo.value)
    assert "auto" in str(excinfo.value)
    assert _m1103_agent_calls(runner) == ["claude"]


@pytest.mark.parametrize("mode", ["decompose-only", "implement-by-phase", "auto"])
@pytest.mark.parametrize("policy", ["all-reviewers", "primary-then-panel"])
def test_staged_plan_under_phase_modes_is_not_stopped(tmp_path, monkeypatch, mode, policy):
    """`phase-modes-staged`: reviews proceed and approval routes to decomposition."""
    runner = _FakeRunner(
        claude_outputs=[_m1268_staged_raw()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
        issue_urls=[f"https://github.com/OWNER/REPO/issues/{100 + i}" for i in range(4)],
    )
    values = {"plan_execution_mode": mode}
    config = (
        _staged_plan_config(tmp_path, **values)
        if policy == "primary-then-panel"
        else make_config(tmp_path, reviewer=("codex", "gemini"), **values)
    )
    dispatched = []
    monkeypatch.setattr(
        plan_first_loop_module,
        "_dispatch_current_decomposition_phase",
        lambda *args, **kwargs: dispatched.append(kwargs["mode"]) or 7,
    )
    result = run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    # Both stages become tracked child issues; by-phase routes dispatch the
    # first phase, decompose-only stops after creating them.
    assert len(runner.issues) == 2
    assert _m1103_agent_calls(runner) == ["claude", "codex", "gemini"]
    if mode == "decompose-only":
        assert (result, dispatched) == (0, [])
    else:
        assert (result, dispatched) == (7, ["implement-by-phase"])


def test_one_shot_plan_under_plan_only_is_unaffected(tmp_path):
    """`plan-only-one-shot`."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0


def test_narrow_staged_blocking_primary_reaches_a_directive_revision_then_stops(tmp_path):
    """`narrow-staged-flag`: blocking primary; the still-staged revision stops."""
    base = _m1268_base()
    still_staged = _m1103_blocking_chain(1, 1, base=base)[1][0]
    runner = _FakeRunner(
        claude_outputs=[_m1268_staged_raw(), still_staged],
        codex_outputs=[
            structured_plan_review(
                state="blocking", summary="Gap.", blocking_plan_issues=["Close the gap."]
            )
        ],
    )
    config = _staged_plan_config(
        tmp_path, max_rounds=8, plan_primary_stall_rounds=1, plan_narrow_staged=True
    )

    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="staged plan under"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert _m1103_agent_calls(runner) == ["claude", "codex", "claude"]
    revision_prompt = _m1268_prompts(runner, "claude")[1]
    assert "Operator directive (orchestrator, not a reviewer finding)" in revision_prompt
    assert "deferred_work" in revision_prompt
    # The directive is not a reviewer item.
    assert "[item-2]" not in revision_prompt


def test_narrow_staged_approving_primary_is_not_approved_or_advanced(tmp_path):
    """`narrow-staged-flag`: an approving primary neither approves nor opens the panel."""
    base = _m1268_base()
    runner = _FakeRunner(
        claude_outputs=[_m1268_staged_raw(), _m1268_narrowing_patch(base)],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_plan_review(state="approved"),
        ],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = _staged_plan_config(tmp_path, max_rounds=8, plan_narrow_staged=True)

    result = run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert result == 0
    calls = _m1103_agent_calls(runner)
    # Round 1: primary approval does not advance to the panel; the planner
    # narrows; the narrowed plan then gets a normal primary-then-panel review.
    assert calls[:3] == ["claude", "codex", "claude"]
    assert "Operator directive (orchestrator" in _m1268_prompts(runner, "claude")[1]
    assert "Operator directive (orchestrator" not in "".join(_m1268_prompts(runner, "codex"))
    prelaunch = [r for r in _plan_round_records(runner) if r.phase == "scheduler-prelaunch"]
    assert prelaunch[0].scheduler_phase == "primary"
    assert not [r for r in _plan_round_records(runner) if r.role == "reviewer" and r.agent == "Gemini" and r.round_number == 1]


def test_narrow_staged_flat_board_does_not_approve_a_staged_candidate(tmp_path):
    """`narrow-staged-flag`: an approving flat board still reaches the revision."""
    base = _m1268_base()
    runner = _FakeRunner(
        claude_outputs=[_m1268_staged_raw(), _m1268_narrowing_patch(base)],
        codex_outputs=[structured_plan_review(state="approved")] * 2,
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")] * 2,
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), plan_narrow_staged=True)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    claude_prompts = _m1268_prompts(runner, "claude")
    assert len(claude_prompts) == 2
    assert "Operator directive (orchestrator" in claude_prompts[1]


def test_narrowed_plan_with_typed_deferred_work_reaches_the_plan_only_stop(tmp_path, capsys):
    """`narrowed-deferred-reviewable`: deferred titles are recorded-only."""
    base = _m1268_base()
    runner = _FakeRunner(
        claude_outputs=[_m1268_staged_raw(), _m1268_narrowing_patch(base)],
        codex_outputs=[structured_plan_review(state="approved")] * 2,
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")] * 2,
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), plan_narrow_staged=True)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    out = capsys.readouterr().out
    assert "Deferred work recorded only" in out
    assert "Stage B follow-up" in out
    assert "--materialize-split-issues" not in out
    # Reviewers see the guard's typed-deferred_work rule.
    assert "typed `deferred_work`" in "".join(_m1268_prompts(runner, "codex"))


def _m1268_rewrite_modes(comments, mode):
    """Rewrite posted history's recorded execution mode (``None`` strips it)."""
    from coding_review_agent_loop.round_transport import encode_mapping

    pattern = re.compile(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->")
    rewritten = []
    for comment in comments:
        def rewrite(match):
            payload = decode_mapping(match.group("payload"))
            for key in ("plan_execution_mode", "scheduler_execution_mode"):
                if key in payload:
                    if mode is None:
                        payload.pop(key)
                    else:
                        payload[key] = mode
            return f"<!-- AGENT_LOOP_META: {encode_mapping(payload)} -->"

        rewritten.append({**comment, "body": pattern.sub(rewrite, comment["body"])})
    return rewritten


def _m1268_issue_urls():
    return [f"https://github.com/OWNER/REPO/issues/{100 + i}" for i in range(4)]


def _m1268_stalled_staged_history(tmp_path, *, record_mode):
    """A staged candidate whose primary blocked up to the stall limit.

    Planned under decompose-only (where a staged candidate is allowed), then
    rewritten to record ``record_mode`` (``None`` models pre-change history).
    """
    codex, claude, _base = _m1103_blocking_chain(1, 2, base=_m1268_base())
    runner = _FakeRunner(claude_outputs=[_m1268_staged_raw(), *claude], codex_outputs=codex)
    config = _staged_plan_config(
        tmp_path, max_rounds=8, plan_primary_stall_rounds=2, plan_execution_mode="decompose-only"
    )
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="blocked 2"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return _m1268_rewrite_modes(list(runner.issue_comments), record_mode), config


def _m1268_resumed_runner(history):
    resolved = [{"item_id": "item-2", "disposition": "resolved"}]
    return _FakeRunner(
        issue_comments=list(history),
        codex_outputs=[
            structured_plan_review(state="approved", prior_plan_item_dispositions=resolved)
        ],
        gemini_outputs=[
            structured_plan_review(
                state="approved", reviewer="Google Gemini", prior_plan_item_dispositions=resolved
            )
        ],
        issue_urls=_m1268_issue_urls(),
    )


def test_round_metadata_records_the_execution_mode(tmp_path):
    runner, _config, _error = _m1103_stalled_runner(tmp_path, threshold=2)
    records = _plan_round_records(runner)
    assert {r.plan_execution_mode for r in records if r.role in {"coder", "reviewer"}} == {
        "plan-only"
    }
    assert {r.scheduler_execution_mode for r in records if r.phase == "scheduler-prelaunch"} == {
        "plan-only"
    }


def test_resume_under_a_changed_mode_retires_the_stall_streak(tmp_path):
    """`resume-mode-changed`: no stall stop, the carried item resolves, children are created."""
    history, config = _m1268_stalled_staged_history(tmp_path, record_mode="plan-only")
    proceeding = _m1268_resumed_runner(history)

    result = run_issue_loop(proceeding, issue_number=56, config=config, plan_first=True)

    assert result == 0
    assert len(proceeding.issues) == 2
    codex_prompts = _m1268_prompts(proceeding, "codex")
    assert "changed the execution mode from plan-only to decompose-only" in codex_prompts[0]
    assert "--plan-execution-mode decompose-only" in codex_prompts[0]
    assert _m1103_agent_calls(proceeding) == ["codex", "gemini"]


def test_resume_without_a_mode_change_still_stalls(tmp_path):
    history, config = _m1268_stalled_staged_history(tmp_path, record_mode="decompose-only")
    rerun = _FakeRunner(issue_comments=list(history))
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="blocked 2"):
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    assert _m1103_agent_calls(rerun) == []


def test_resume_over_legacy_unrecorded_history_names_the_reset_flag(tmp_path):
    """`resume-legacy-stalled`: no silent retirement; the flag recovers."""
    history, config = _m1268_stalled_staged_history(tmp_path, record_mode=None)

    rerun = _FakeRunner(issue_comments=list(history))
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError) as excinfo:
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    assert "--plan-reset-stall-streak" in str(excinfo.value)
    assert _m1103_agent_calls(rerun) == []

    proceeding = _m1268_resumed_runner(history)
    result = run_issue_loop(
        proceeding,
        issue_number=56,
        config=replace(config, plan_reset_stall_streak=True),
        plan_first=True,
    )
    assert result == 0
    assert len(proceeding.issues) == 2
    codex_prompts = _m1268_prompts(proceeding, "codex")
    assert "did not record their execution mode" in codex_prompts[0]


def _m1268_carried_history(tmp_path, *, complete, monkeypatch=None):
    """A staged candidate whose approvals were recorded under decompose-only."""
    runner = _FakeRunner(
        claude_outputs=[_m1268_staged_raw()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
        issue_urls=_m1268_issue_urls(),
    )
    config = _staged_plan_config(
        tmp_path,
        max_rounds=(8 if complete else 1),
        plan_execution_mode="decompose-only",
    )
    if complete:
        monkeypatch.setattr(
            plan_first_loop_module,
            "_decompose_approved_plan",
            lambda *a, **k: (_ for _ in ()).throw(AgentLoopError("approval boundary stub")),
        )
    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return list(runner.issue_comments)


def _m1268_new_records(runner, history):
    seen = len(history)
    return _plan_round_records(runner)[
        len(_plan_round_records(_FakeRunner(issue_comments=list(history)))):
    ] if seen else _plan_round_records(runner)


def test_staged_candidate_with_a_carried_primary_approval_stops_without_new_records(tmp_path):
    """`plan-only-staged-fresh` growth-seam variant: a carried approval does not bypass the stop."""
    history = _m1268_carried_history(tmp_path, complete=False)
    resumed = _FakeRunner(issue_comments=list(history))
    config = _staged_plan_config(tmp_path, max_rounds=8)

    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="staged plan under"):
        run_issue_loop(resumed, issue_number=56, config=config, plan_first=True)

    assert _m1103_agent_calls(resumed) == []
    assert _m1268_new_records(resumed, history) == []
    assert len(resumed.issue_comments) == len(history) + 1  # the plain diagnostic only


def test_narrow_staged_with_a_carried_primary_approval_schedules_no_panel(tmp_path):
    """`narrow-carried-primary-approval`: the extended seam skips the panel."""
    history = _m1268_carried_history(tmp_path, complete=False)
    base = _m1268_base()
    resumed = _FakeRunner(
        issue_comments=list(history),
        claude_outputs=[_m1268_narrowing_patch(base, one_shot=False)],
    )
    config = _staged_plan_config(tmp_path, max_rounds=8, plan_narrow_staged=True)

    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="staged plan under"):
        run_issue_loop(resumed, issue_number=56, config=config, plan_first=True)

    # Only the narrowing revision ran: no reviewer, then the still-staged stop.
    assert _m1103_agent_calls(resumed) == ["claude"]
    assert "Operator directive (orchestrator" in _m1268_prompts(resumed, "claude")[0]
    new = _m1268_new_records(resumed, history)
    assert [r.role for r in new] == ["coder"]
    assert not [r for r in new if r.phase == "scheduler-prelaunch"]


def test_narrow_staged_with_carried_complete_approvals_does_not_approve(tmp_path, monkeypatch):
    """`narrow-staged-flag`: carried complete approvals still reach the revision."""
    history = _m1268_carried_history(tmp_path, complete=True, monkeypatch=monkeypatch)
    base = _m1268_base()
    resumed = _FakeRunner(
        issue_comments=list(history),
        claude_outputs=[_m1268_narrowing_patch(base, one_shot=False)],
    )
    config = _staged_plan_config(tmp_path, max_rounds=8, plan_narrow_staged=True)

    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="staged plan under"):
        run_issue_loop(resumed, issue_number=56, config=config, plan_first=True)

    assert _m1103_agent_calls(resumed) == ["claude"]
    assert "Operator directive (orchestrator" in _m1268_prompts(resumed, "claude")[0]
    assert not [r for r in _m1268_new_records(resumed, history) if r.role == "reviewer"]


def test_narrow_staged_flat_board_resume_over_complete_approvals_does_not_approve(
    tmp_path, monkeypatch
):
    """`narrow-staged-flag`: all-reviewers history holding both approvals of the staged candidate."""
    first = _FakeRunner(
        claude_outputs=[_m1268_staged_raw()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
        issue_urls=_m1268_issue_urls(),
    )
    monkeypatch.setattr(
        plan_first_loop_module,
        "_decompose_approved_plan",
        lambda *a, **k: (_ for _ in ()).throw(AgentLoopError("approval boundary stub")),
    )
    flat = make_config(tmp_path, reviewer=("codex", "gemini"), plan_execution_mode="decompose-only")
    with pytest.raises(AgentLoopError, match="approval boundary stub"):
        run_issue_loop(first, issue_number=56, config=flat, plan_first=True)
    history = list(first.issue_comments)
    approvals = [r for r in _plan_round_records(first) if r.role == "reviewer" and r.state == "approved"]
    assert {r.agent for r in approvals} == {"Codex", "Gemini"}

    base = _m1268_base()
    resumed = _FakeRunner(
        issue_comments=list(history),
        claude_outputs=[_m1268_narrowing_patch(base, one_shot=False)],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), plan_narrow_staged=True)

    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError, match="staged plan under"):
        run_issue_loop(resumed, issue_number=56, config=config, plan_first=True)

    # The flat board re-reviews every round (nothing is carried); the approving
    # board must still not approve the staged candidate: the directive reaches
    # the planner revision, and the still-staged revision stops the next round.
    calls = _m1103_agent_calls(resumed)
    assert calls.count("claude") == 1
    assert "Operator directive (orchestrator" in _m1268_prompts(resumed, "claude")[0]
    assert calls[-1] == "claude"
    assert not resumed.issues


def test_planner_revisions_carry_guidance_and_in_memory_history_without_step_back(tmp_path):
    """#1273 `planner-live-history`: no step-back records, no extra GitHub read."""
    runner, _error = _m1251_run(
        tmp_path, last_round=4, max_rounds=4, plan_step_back_rounds=0,
        plan_step_back_escalation_rounds=1,
    )
    revisions = [
        prompt for prompt in _m1251_prompts(runner, "claude")
        if "Proactive generalization" in prompt
    ]
    assert len(revisions) >= 2
    assert not any(_STEP_BACK_MARKER in prompt for prompt in revisions)
    assert "no earlier-round findings or fixes" in revisions[0] or "[item-1]" in revisions[0]
    later = revisions[-1]
    assert "Earlier-round history for this run" in later
    assert "Codex finding item-1" in later
    assert "fix by " in later


def test_planner_history_failure_is_advisory_and_keeps_the_guidance(tmp_path, monkeypatch):
    """#1273 `history-decode-failure` case A for the plan loop."""
    from coding_review_agent_loop import finding_history

    def boom(item):
        raise RuntimeError("projection exploded")

    monkeypatch.setattr(finding_history, "project_finding", boom)
    runner, error = _m1251_run(
        tmp_path, last_round=3, max_rounds=3, plan_step_back_rounds=0,
        plan_step_back_escalation_rounds=1,
    )
    revisions = [
        prompt for prompt in _m1251_prompts(runner, "claude")
        if "Proactive generalization" in prompt
    ]
    assert revisions
    assert all("history is unavailable this turn" in prompt for prompt in revisions)
    assert "projection exploded" not in str(error)


def _m1273_run_all_reviewers(tmp_path, *, declare, rounds=3, locations=False):
    fresh, base0 = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, rounds, base=base0)
    if locations:
        codex = [
            output.replace(f"Close gap {n}.", f"Close gap {n} in src/spool.py:{10 + n}.")
            for n, output in enumerate(codex, start=1)
        ]
    if declare:
        claude = [
            output.replace(
                '"summary": "Close gap 2."',
                '"summary": "Generalization: this generalizes the fix for [item-1]: every gap"',
            )
            for output in claude
        ]
    runner = _FakeRunner(claude_outputs=[fresh, *claude], codex_outputs=codex)
    config = make_config(
        tmp_path, reviewer=("codex",), max_rounds=rounds, quiet=False,
    )
    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return runner


def test_all_reviewers_planner_history_shows_resolved_finding_and_prior_fix_not_sibling(
    tmp_path,
):
    """`planner-live-history`/`fix-cutoff-sibling` under the compatibility policy."""
    runner = _m1273_run_all_reviewers(tmp_path, declare=False, rounds=4)
    revisions = [
        prompt for prompt in _m1251_prompts(runner, "claude")
        if "Proactive generalization" in prompt
    ]
    assert len(revisions) >= 3
    third = revisions[2]
    history = third.split("Earlier-round history for this run", 1)[1]
    history = history.split("Blocking plan review payload", 1)[0].split("plan review:", 1)[0]
    assert "Codex finding item-1 (resolved)" in history
    assert "fix by " in history and "Close gap 1." in history
    assert "item-3" not in history


def test_planner_declared_generalization_is_logged_with_a_tag_and_silent_otherwise(
    tmp_path, capsys
):
    """`generalization-logged` for the planner (proactive, no step-back)."""
    _m1273_run_all_reviewers(tmp_path, declare=True)
    err = capsys.readouterr().err
    assert err.count("declared a generalization") == 1
    assert "declared a generalization (proactive): Generalization: this generalizes" in err
    _m1273_run_all_reviewers(tmp_path / "quiet", declare=False)
    assert "declared a generalization" not in capsys.readouterr().err


_DECLARATION = "Generalization: this generalizes the fix for [item-1]: every gap"


def test_planner_disposition_note_declaration_is_logged(tmp_path, capsys):
    """`generalization-logged`: a declaration in a disposition note, not the summary."""
    fresh, base0 = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, 3, base=base0)
    marker = '"item_id": "item-2", "disposition": "resolved"'
    assert any(marker in output for output in claude)
    claude = [
        output.replace(marker, marker + f', "note": "{_DECLARATION}"') for output in claude
    ]
    runner = _FakeRunner(claude_outputs=[fresh, *claude], codex_outputs=codex)
    config = make_config(tmp_path, reviewer=("codex",), max_rounds=3, quiet=False)
    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    err = capsys.readouterr().err
    assert err.count("declared a generalization") == 1
    assert "declared a generalization (proactive): Generalization: this generalizes" in err


def test_real_planner_step_back_turn_logs_a_step_back_directed_generalization(
    tmp_path, capsys
):
    runner, _error = _m1251_run(
        tmp_path, last_round=4, max_rounds=5, plan_step_back_escalation_rounds=5,
        quiet=False,
        claude_transform=lambda output: output.replace(
            '"summary": "Close gap 4."', f'"summary": "{_DECLARATION}"'
        ),
    )
    assert any(_STEP_BACK_MARKER in prompt for prompt in _m1251_prompts(runner, "claude"))
    err = capsys.readouterr().err
    assert err.count("declared a generalization") == 1
    assert "declared a generalization (step-back-directed)" in err


def test_all_reviewers_planner_history_cutoff_with_locations(tmp_path):
    runner = _m1273_run_all_reviewers(tmp_path, declare=False, rounds=4, locations=True)
    revisions = [
        prompt for prompt in _m1251_prompts(runner, "claude")
        if "Proactive generalization" in prompt
    ]
    # Prompt for review round 2: round-1 finding resolved with its location, the
    # fix published under round 2 (the first revision) is shown, and the round-2
    # sibling appears only in the review payload.
    second = revisions[1]
    history = second.split("Earlier-round history for this run", 1)[1].split("\n\n", 1)[0]
    assert "Codex finding item-1 (resolved) src/spool.py:11" in history
    assert "fix by " in history and "Close gap 1." in history
    assert "item-2" not in history and "src/spool.py:12" not in history
    assert "src/spool.py:12" in second


def _non_agent_command_count(runner):
    return sum(
        1 for cmd, _cwd in runner.commands if cmd and cmd[0] not in {"codex", "claude", "gemini"}
    )


def test_history_adds_no_github_reads_under_either_planning_policy(tmp_path, monkeypatch):
    from coding_review_agent_loop.finding_history import FindingHistoryLedger

    def counts(label):
        # Fresh directories per call: the second pass must not see the first's state.
        compat = _m1273_run_all_reviewers(tmp_path / label / "c", declare=False, rounds=3)
        staged, _error = _m1251_run(
            tmp_path / label / "s", last_round=3, max_rounds=3, plan_step_back_rounds=0,
            plan_step_back_escalation_rounds=1,
        )
        return _non_agent_command_count(compat), _non_agent_command_count(staged)

    with_history = counts("with-history")
    monkeypatch.setattr(FindingHistoryLedger, "observe_reconciled", lambda *a, **k: None)
    monkeypatch.setattr(FindingHistoryLedger, "record_fix", lambda *a, **k: None)
    monkeypatch.setattr(FindingHistoryLedger, "seed_from_records", lambda *a, **k: None)
    assert counts("without-history") == with_history


def _m1273_patch(n, base, summary, dispositions):
    payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": summary,
        "prior_plan_item_dispositions": dispositions,
        "base_round_number": n,
        "base_state_identity": base.state_identity,
        "operations": [
            {"op": "replace", "field": "plan_steps", "value": [f"Implement scope, revision {n}."]}
        ],
    }
    next_base = AuthenticatedPlanState.from_plan(
        assemble_plan_revision(base, payload), round_number=n + 1
    )
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude", next_base


def test_planner_resume_before_clearance_resolves_a_retained_future_finding(tmp_path):
    """The planner keeps future items in its full ledger; a resumed review clears one.

    Round 2: Codex approves and defers its item-2 while Gemini blocks on item-3,
    so the round still revises with item-2 retained as `future`.  The first
    invocation stops after that revision.  The second invocation's first review
    resolves item-2 while a new finding still forces a revision: that revision's
    history must say resolved, not deferred.
    """
    fresh, base = _m1103_fresh_base()
    rev1, base = _m1273_patch(
        1, base, "Close gap 1.",
        [{"item_id": "item-1", "disposition": "resolved"},
         {"item_id": "item-2", "disposition": "resolved"}],
    )
    rev2, base = _m1273_patch(
        2, base, "Close gap three.", [{"item_id": "item-3", "disposition": "resolved"}]
    )
    rev3, base = _m1273_patch(
        3, base, "Close gap 4.", [{"item_id": "item-4", "disposition": "resolved"}]
    )

    def gemini(**kwargs):
        return structured_plan_review(reviewer="Google Gemini", **kwargs)

    runner = _FakeRunner(
        claude_outputs=[fresh, rev1, rev2],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Close gap 1 in docs/a.md:10.", "Gap two in docs/b.md:20."],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "future", "note": "follow-up"},
                ],
            ),
        ],
        gemini_outputs=[
            gemini(state="approved"),
            gemini(
                state="blocking",
                blocking_plan_issues=["Gap three in docs/c.md:30."],
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=6)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    revisions = [p for p in _m1251_prompts(runner, "claude") if "Proactive generalization" in p]

    def history(prompt):
        return prompt.split("Earlier-round history for this run", 1)[1].split("\n\n", 1)[0]

    assert len(revisions) == 2
    assert "Codex finding item-2 (deferred) docs/b.md:20" in history(revisions[1])

    # Second invocation resumes from the posted records; its first review clears item-2.
    clearing = [
        {"item_id": "item-2", "disposition": "resolved"},
        {"item_id": "item-3", "disposition": "resolved"},
    ]
    runner.codex_outputs.append(
        structured_plan_review(
            state="blocking", blocking_plan_issues=["Close gap 4."],
            prior_plan_item_dispositions=clearing,
        )
    )
    runner.gemini_outputs.append(gemini(state="approved", prior_plan_item_dispositions=clearing))
    runner.claude_outputs.append(rev3)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    revisions = [p for p in _m1251_prompts(runner, "claude") if "Proactive generalization" in p]
    assert len(revisions) == 3
    assert "Codex finding item-2 (resolved) docs/b.md:20" in history(revisions[2])
    assert "(deferred)" not in history(revisions[2])
