"""Ownership correction drives real plan-first orchestration paths."""
import json
from types import SimpleNamespace

import pytest

from agent_loop_helpers import FakeRunner, make_config as _make_config, structured_plan_review, structured_v1_plan_state
from issue_loop_test_helpers import _fresh_staged_plan_for_recovery, _PlanDiagnosticRunner
from coding_review_agent_loop.cli import run_issue_loop
from coding_review_agent_loop.errors import AgentInvocationError, AgentLoopError
from coding_review_agent_loop.protocol import validate_structured_plan_state, risk_test_matrix_identity
from coding_review_agent_loop.comment_rendering import render_canonical_plan_state
from coding_review_agent_loop.decomposition import validate_risk_matrix_ownership, RiskMatrixOwnershipError
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject


def make_config(tmp_path, **kwargs):
    kwargs.setdefault("execution_strategy_contract_required", True)
    return _make_config(tmp_path, **kwargs)


def candidate(owner="operator", staged=False):
    raw = _fresh_staged_plan_for_recovery()[0] if staged else structured_v1_plan_state()
    payload = json.loads(raw.split("\n", 1)[0])
    for stage in payload["execution_recommendation"]["child_stages"]:
        stage["execution_disposition"] = {"disposition": "requires-child-planning", "rationale": "Review the child design before implementation.", "unresolved_design_decisions": ["Choose the child implementation design."]}
    payload["risk_test_matrix"] = {
        "applicability": "applicable", "important_exclusions": [], "rows": [{
            "row_id": "owner-check", "label": "Check owner", "entry_path_or_mode": "plan-first",
            "initial_state": "Candidate exists", "event": "Validate", "expected_outcome": "Correct owner",
            "forbidden_side_effects": ["No invalid dispatch"], "proposed_test_level": "workflow",
            "proposed_test_location": "tests/test_plan_ownership.py", "applicability": "required",
            "related_scope_item_ids": ["scope-1"], "execution_owner": owner,
        }],
    }
    payload["risk_test_matrix_changes"] = [{"operation": "add", "row_ids": ["owner-check"], "rationale": "Cover ownership."}]
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"


def history(raw, config, state="blocking", round_number=1):
    parsed = validate_structured_plan_state(raw)
    canonical = render_canonical_plan_state(parsed, config)
    return {"author": {"login": "bot"}, "createdAt": "2026-09-14T00:00:00Z", "body": _attach_round_metadata(
        canonical + f"\n<!-- AGENT_PLAN_STATE: {state} -->\n-- Anthropic Claude",
        PostedRoundMetadata(flow="plan", role="coder", agent="Anthropic Claude", round_number=round_number,
                            subject=_plan_subject(canonical), canonical_plan=canonical,
                            raw_structured_coder_response=raw, execution_strategy_contract_version=1,
                            risk_test_matrix_contract_version=1,
                            risk_test_matrix_identity=risk_test_matrix_identity(parsed.risk_test_matrix, parsed.risk_test_matrix_changes),
                            risk_test_matrix_boundary_digest=risk_test_matrix_identity(parsed.risk_test_matrix, parsed.risk_test_matrix_changes),
                            risk_test_matrix_payload=parsed.risk_test_matrix.to_payload(),
                            risk_test_matrix_changes_payload=tuple(change.to_payload() for change in parsed.risk_test_matrix_changes),
                            execution_strategy_identity=parsed.execution_recommendation.identity()),
    )}


@pytest.mark.parametrize("staged", [False, True])
def test_fresh_owner_retry_precedes_review(tmp_path, monkeypatch, staged):
    import coding_review_agent_loop.validated_agent as validated
    def no_repair(*args, **kwargs):
        raise AssertionError("Ownership correction must return to the planner")
    monkeypatch.setattr(validated, "_run_structured_repair", no_repair)
    runner = FakeRunner(claude_outputs=[candidate(staged=staged), candidate("one-shot")],
                        codex_outputs=[structured_plan_review(state="approved")])
    config = make_config(tmp_path, plan_execution_mode="plan-only", max_rounds=3)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    agents = [cmd[0] for cmd, _ in runner.commands if cmd[0] in {"claude", "codex"}]
    assert agents[:3] == ["claude", "claude", "codex"]
    assert not any('operator' in body for body in runner.comments)
    retry_prompt = next(cmd[-1] for cmd, _ in runner.commands
                        if cmd[0] == "claude" and "Invalid execution owners: operator" in cmd[-1])
    assert "Allowed execution owners: " + ("stage-1, stage-2" if staged else "one-shot") in retry_prompt


@pytest.mark.parametrize("staged", [False, True])
def test_owner_retry_exhaustion_is_specific_and_persisted(tmp_path, staged):
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [candidate(staged=staged)] * 5
    with pytest.raises(AgentInvocationError, match="fail risk-matrix ownership validation") as error:
        run_issue_loop(runner, issue_number=56, config=make_config(tmp_path), plan_first=True)
    assert "operator" in str(error.value)
    assert "Allowed execution owners" in str(error.value)
    assert error.value.plan_validation_exhaustion is not None
    assert len(runner.diagnostic_posts) == 1
    assert not any(cmd[0] == "codex" for cmd, _ in runner.commands)


@pytest.mark.parametrize("compact,blocking", [(False, False), (True, False), (False, True)])
def test_published_invalid_owner_forces_planner_feedback(tmp_path, monkeypatch, compact, blocking):
    config = make_config(tmp_path, plan_execution_mode="plan-only", max_rounds=3,
                         planning_context_mode="compact" if compact else "full")
    review = structured_plan_review(state="blocking" if blocking else "approved",
                                   blocking_plan_issues=["Unrelated blocker"] if blocking else [])
    from test_child_plan_provenance import _ChildPlanningRunner
    runner = _ChildPlanningRunner(issue_comments=[history(candidate(), config, round_number=2 if compact else 1)], codex_outputs=[review],
                        claude_outputs=[candidate().replace('"kind": "plan_state"', '"kind": "plan_revision", "prior_plan_item_dispositions": []')] * 5)
    if blocking:
        import coding_review_agent_loop.plan_first_loop as loop
        seed_runner = FakeRunner(issue_comments=runner.issue_comments, codex_outputs=[review])
        original = loop._run_validated_agent
        def stop_seed(*args, **kwargs):
            if kwargs["agent"] == config.coder:
                raise RuntimeError("seed blocker published")
            return original(*args, **kwargs)
        with monkeypatch.context() as seed:
            seed.setattr(loop, "plan_ownership_diagnostic", lambda *args: None)
            seed.setattr(loop, "_run_validated_agent", stop_seed)
            with pytest.raises(RuntimeError, match="seed blocker published"):
                run_issue_loop(seed_runner, issue_number=56, config=config, plan_first=True)
        runner.issue_comments = seed_runner.issue_comments
    with pytest.raises(AgentInvocationError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    commands = [cmd for cmd, _ in runner.commands if cmd[0] in {"claude", "codex"}]
    assert all(cmd[0] == "claude" for cmd in commands)
    if blocking:
        assert "Unrelated blocker" in commands[0][-1]
    assert "Invalid execution owners: operator" in commands[0][-1]
    assert "Allowed execution owners: one-shot" in commands[0][-1]
    assert not any("Plan approved" in body for body in runner.comments)


def test_published_invalid_owner_round_limit(tmp_path):
    config = make_config(tmp_path, plan_execution_mode="plan-only", max_rounds=1)
    runner = FakeRunner(issue_comments=[history(candidate(), config)],
                        codex_outputs=[structured_plan_review(state="approved")])
    with pytest.raises(AgentLoopError, match="ownership validation.*planning round budget") as error:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert "--max-rounds" in str(error.value)
    assert not any(cmd[0] == "claude" for cmd, _ in runner.commands)


@pytest.mark.parametrize("strategy,retained,final,owner,valid", [
    ("one-shot", "none", "none", "one-shot", True),
    ("one-shot", "required", "required", "human", False),
    ("staged", "none", "none", "human", True),
    ("staged", "none", "none", "one-shot", False),
    ("staged", "none", "none", "retained-parent", False),
    ("staged", "required", "none", "retained-parent", True),
    ("staged", "none", "required", "final-integration", True),
    ("staged", "none", "none", "final-integration", False),
    ("staged", "none", "none", "operator", False),
])
def test_effective_owner_set(strategy, retained, final, owner, valid):
    matrix = json.loads(candidate(owner).split("\n", 1)[0])["risk_test_matrix"]
    recommendation = SimpleNamespace(strategy=strategy, child_stages=[SimpleNamespace(stage_id="human", automation="human-action")],
                                     retained_parent_work=SimpleNamespace(status=retained), final_integration_work=SimpleNamespace(status=final))
    if valid:
        validate_risk_matrix_ownership(matrix, recommendation)
    else:
        with pytest.raises(RiskMatrixOwnershipError, match="Allowed execution owners") as error:
            validate_risk_matrix_ownership(matrix, recommendation)
        assert owner in str(error.value)
    assert matrix["rows"][0]["execution_owner"] == owner


def test_semantic_revision_checks_assembled_topology(tmp_path):
    from coding_review_agent_loop.plan_assembly import AuthenticatedPlanState
    raw = candidate("stage-1", staged=True)
    base = AuthenticatedPlanState.from_plan(validate_structured_plan_state(raw), round_number=1)
    recommendation = json.loads(candidate("one-shot").split("\n", 1)[0])["execution_recommendation"]
    # Removing the stages leaves an existing row owned by a removed stage.
    patch = {
        "schema_version": 1, "kind": "plan_revision_patch", "semantic_patch_contract_version": 1,
        "state": "blocking", "summary": "Rejected topology", "prior_plan_item_dispositions": [{"item_id": "item-1", "disposition": "resolved"}],
        "base_round_number": 1, "base_state_identity": base.state_identity,
        "operations": [{"op": "replace", "field": "execution_recommendation", "value": recommendation},
                       {"op": "replace", "field": "summary", "value": "Rejected topology"}],
    }
    invalid = json.dumps(patch) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    row = json.loads(raw.split("\n", 1)[0])["risk_test_matrix"]["rows"][0]
    row["execution_owner"] = "one-shot"
    patch["summary"] = "Corrected topology"
    patch["operations"][1]["value"] = "Corrected topology"
    patch["operations"].append({"op": "matrix_edit", "row_id": "owner-check", "row": row, "rationale": "Move ownership to the resulting strategy."})
    good = json.dumps(patch) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    runner = FakeRunner(claude_outputs=[raw, invalid, good], codex_outputs=[
        structured_plan_review(state="blocking", blocking_plan_issues=["Revise topology"]),
        structured_plan_review(state="approved", prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
    ])
    config = make_config(tmp_path, plan_execution_mode="decompose-only", max_rounds=3, plan_growth_gate="off")
    with pytest.raises(AgentLoopError, match="strategy `one-shot` cannot run requested policy `decompose-only`"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    prompts = [cmd[-1] for cmd, _ in runner.commands if cmd[0] == "claude"]
    assert "human-action child stage" in prompts[1]
    assert "exact-head human evidence at the final head" in prompts[1]
    assert "stage-1" in prompts[2] and "Allowed execution owners: one-shot" in prompts[2]
    assert "Rejected topology" not in "\n".join(runner.comments)
    assert "Corrected topology" in "\n".join(runner.comments)
    assert len([cmd for cmd, _ in runner.commands if cmd[0] == "codex"]) == 2


@pytest.mark.parametrize("staged", [False, True])
def test_approved_invalid_owner_keeps_shortcut_and_stops_dispatch(tmp_path, monkeypatch, staged):
    import coding_review_agent_loop.plan_first_loop as loop
    config = make_config(tmp_path, plan_execution_mode="plan-only", max_rounds=3)
    runner = FakeRunner(claude_outputs=[candidate(staged=staged)],
                        codex_outputs=[structured_plan_review(state="approved")])
    # Reproduce the pre-fix approval, which did not check a published candidate.
    with monkeypatch.context() as seed:
        seed.setattr(loop, "validate_candidate_ownership", lambda *args: None)
        seed.setattr(loop, "plan_ownership_diagnostic", lambda *args: None)
        seed.setattr(loop, "phased_delivery_guard_active", lambda *args: False)
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    from dataclasses import replace
    config = replace(config, plan_execution_mode="decompose-only" if staged else "implement-one-shot")
    rerun = FakeRunner(issue_comments=runner.issue_comments)
    def no_new_check(*args):
        raise AssertionError("Approved shortcuts must skip the round-start ownership check")
    monkeypatch.setattr(loop, "plan_ownership_diagnostic", no_new_check)
    with pytest.raises(RiskMatrixOwnershipError, match="new signed human requirement") as error:
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    assert "Allowed execution owners" in str(error.value)
    assert "then rerun" in str(error.value)
    assert not any(cmd[0] in {"claude", "codex"} or cmd[:3] == ["gh", "issue", "create"] for cmd, _ in rerun.commands)


@pytest.mark.parametrize("compact", [False, True])
def test_review_diagnostic_without_inherited_binding(tmp_path, compact):
    from coding_review_agent_loop.prompts import build_plan_review_prompt
    prompt = build_plan_review_prompt(56, 2, "Published plan", make_config(tmp_path), reviewer="codex",
                                     compact_context=compact, ownership_check_failure="operator; Allowed execution owners: one-shot")
    assert "Deterministic ownership check" in prompt
    assert "operator; Allowed execution owners: one-shot" in prompt


@pytest.mark.parametrize("child", [False, True])
def test_ownership_exhaustion_restart_feeds_persisted_diagnostic(tmp_path, monkeypatch, child):
    from test_child_plan_provenance import _ChildPlanningRunner, _bind_child_planning
    from coding_review_agent_loop.decomposition import InheritedMatrixBinding
    if child:
        matrix = json.loads(candidate("api").split("\n", 1)[0])["risk_test_matrix"]
        _bind_child_planning(monkeypatch, InheritedMatrixBinding(parent_issue=55, stage_id="api", parent_matrix=matrix))
    config = make_config(tmp_path, plan_execution_mode="plan-only", max_rounds=3)
    runner = _ChildPlanningRunner(claude_outputs=[candidate()] * 3)
    with pytest.raises(AgentInvocationError, match="fail risk-matrix ownership validation"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert len(runner.diagnostic_posts) == 1
    diagnostic = {"id": 901, "author": {"login": "agent"}, "authorId": 7, "databaseId": 901,
                  "createdAt": "2026-09-17T05:30:00Z", "body": runner.diagnostic_posts[0]}
    import coding_review_agent_loop.github as github
    monkeypatch.setattr(github, "read_rest_issue_comments", lambda *args, **kwargs: (
        github.IssueComment(author="agent", author_id=7, comment_id=901,
                            created_at="2026-09-17T05:30:00Z", body=runner.diagnostic_posts[0]),
    ))
    resumed = _ChildPlanningRunner(issue_comments=[diagnostic], claude_outputs=[candidate("one-shot")],
                                   codex_outputs=[structured_plan_review(state="approved")])
    assert run_issue_loop(resumed, issue_number=56, config=config, plan_first=True) == 0
    prompt = next(cmd[-1] for cmd, _ in resumed.commands if cmd[0] == "claude")
    assert "Invalid execution owners: operator" in prompt
    assert "Allowed execution owners: one-shot" in prompt
    assert len([cmd for cmd, _ in resumed.commands if cmd[0] == "claude"]) == 1


def test_full_revision_uses_resulting_topology(tmp_path):
    config = make_config(tmp_path, plan_execution_mode="decompose-only", max_rounds=3)
    raw = candidate("stage-1", staged=True)
    invalid = json.loads(candidate("stage-1").split("\n", 1)[0])
    invalid.update(kind="plan_revision", summary="Rejected full topology", prior_plan_item_dispositions=[])
    # The matrix is unchanged: the owner's stage was removed from the recommendation.
    invalid["risk_test_matrix_changes"] = []
    good = json.loads(candidate("one-shot").split("\n", 1)[0])
    good.update(kind="plan_revision", summary="Corrected full topology", prior_plan_item_dispositions=[])
    good["risk_test_matrix_changes"] = [{"operation": "change", "row_ids": ["owner-check"], "rationale": "Use resulting topology."}]
    outputs = [json.dumps(p) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude" for p in [invalid, good]]
    # Ownership guard requires correction of this pre-fix published candidate.
    base = candidate("operator", staged=True)
    runner = FakeRunner(issue_comments=[history(base, config)], claude_outputs=outputs,
                        codex_outputs=[structured_plan_review(state="approved")] * 2)
    with pytest.raises(AgentLoopError, match="strategy `one-shot` cannot run requested policy `decompose-only`"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    prompts = [cmd[-1] for cmd, _ in runner.commands if cmd[0] == "claude"]
    assert "Invalid execution owners: stage-1" in prompts[1]
    assert "Allowed execution owners: one-shot" in prompts[1]
    assert "Rejected full topology" not in "\n".join(runner.comments)
    assert "Corrected full topology" in "\n".join(runner.comments)


@pytest.mark.parametrize("policy", ["all-reviewers", "primary-then-panel"])
@pytest.mark.parametrize("reopened", [False, True])
def test_resumed_ownership_correction_precedes_phase_advance(tmp_path, monkeypatch, policy, reopened):
    from dataclasses import replace
    from agent_loop_helpers import patch_attr_in
    from coding_review_agent_loop.github import HumanReviewRequirement
    import coding_review_agent_loop.plan_first_loop as loop
    import coding_review_agent_loop.issue_loop as issue_loop
    import coding_review_agent_loop.discuss_loop as discuss
    import coding_review_agent_loop.execution_policy as execution
    import coding_review_agent_loop.child_plan_binding as child
    import coding_review_agent_loop.pr_loop as pr
    import coding_review_agent_loop.pr_loop_support as support
    import coding_review_agent_loop.github as github
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=4,
                         plan_execution_mode="plan-only", plan_review_policy=policy,
                         primary_plan_reviewer="codex" if policy == "primary-then-panel" else None)
    # Seed a complete pre-fix approved history with the real publication path.
    runner = FakeRunner(claude_outputs=[candidate()], codex_outputs=[structured_plan_review(state="approved")],
                        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")])
    with monkeypatch.context() as seed:
        seed.setattr(loop, "validate_candidate_ownership", lambda *args: None)
        seed.setattr(loop, "plan_ownership_diagnostic", lambda *args: None)
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    if not reopened:
        # Keep only the published candidate: a pre-fix unapproved history.
        runner.issue_comments = [c for c in runner.issue_comments if "### Plan steps" in c["body"]][:1]
    else:
        requirement = HumanReviewRequirement(source_type="Issue comment", author="maintainer",
            created_at="2026-10-31T00:00:00Z", url="https://github.com/OWNER/REPO/issues/56#issuecomment-12",
            body="Request a corrected plan with valid execution owners.")
        original_context = github.get_issue_context
        def with_requirement(*args, **kwargs):
            return replace(original_context(*args, **kwargs), human_requirements=(requirement,))
        patch_attr_in(monkeypatch, (loop, issue_loop, discuss, execution, child, pr, support),
                      "get_issue_context", with_requirement)
    dispositions = ([{"requirement_id": requirement.requirement_id, "disposition": "addressed",
                      "evidence": "Correct the ownership."}] if reopened else [])
    rerun = FakeRunner(issue_comments=runner.issue_comments,
        codex_outputs=[structured_plan_review(state="approved", human_requirements_resolved=reopened,
                                             human_requirement_dispositions=dispositions)],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini",
                                              human_requirements_resolved=reopened, human_requirement_dispositions=dispositions)])
    original_turn = loop._run_validated_agent
    captured = []
    def stop_at_planner(*args, **kwargs):
        if kwargs["agent"] == config.coder:
            captured.append(kwargs["prompt"])
            raise RuntimeError("planner correction reached")
        return original_turn(*args, **kwargs)
    monkeypatch.setattr(loop, "_run_validated_agent", stop_at_planner)
    with pytest.raises(RuntimeError, match="planner correction reached"):
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    assert "Invalid execution owners: operator" in captured[0]
    assert "Allowed execution owners: one-shot" in captured[0]
    assert not any("Plan review phase advance" in body for body in rerun.comments)
    if policy == "primary-then-panel" and not reopened:
        assert not any(cmd[0] == "gemini" for cmd, _ in rerun.commands)


@pytest.mark.parametrize("versioned", [False, True])
def test_valid_and_legacy_approved_resume_reuses_approval(tmp_path, versioned):
    from agent_loop_helpers import structured_plan_state
    config = make_config(tmp_path, execution_strategy_contract_required=versioned, plan_execution_mode="plan-only")
    raw = candidate("one-shot") if versioned else structured_plan_state()
    runner = FakeRunner(claude_outputs=[raw], codex_outputs=[structured_plan_review(state="approved")])
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    resumed = FakeRunner(issue_comments=runner.issue_comments)
    assert run_issue_loop(resumed, issue_number=56, config=config, plan_first=True) == 0
    assert not any(cmd[0] in {"claude", "codex"} for cmd, _ in resumed.commands)


def test_human_action_stage_is_an_effective_owner():
    payload = json.loads(candidate("stage-1", staged=True).split("\n", 1)[0])
    stage = payload["execution_recommendation"]["child_stages"][0]
    stage["automation"] = "human-action"
    stage["execution_disposition"] = {"disposition": "human-owned", "rationale": "A person must use a real phone.", "unresolved_design_decisions": []}
    parsed = validate_structured_plan_state(json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude")
    validate_risk_matrix_ownership(parsed.risk_test_matrix, parsed.execution_recommendation)


def test_ownership_feedback_preserves_simultaneous_inherited_failure(tmp_path, monkeypatch):
    from test_child_plan_provenance import _bind_child_planning
    from coding_review_agent_loop.decomposition import InheritedMatrixBinding
    import coding_review_agent_loop.plan_first_loop as loop
    parent = json.loads(candidate("api").split("\n", 1)[0])["risk_test_matrix"]
    parent["rows"][0]["forbidden_side_effects"].append("Preserve the immutable audit")
    _bind_child_planning(monkeypatch, InheritedMatrixBinding(parent_issue=55, stage_id="api", parent_matrix=parent))
    config = make_config(tmp_path, max_rounds=3, plan_execution_mode="plan-only")
    runner = FakeRunner(issue_comments=[history(candidate(), config)],
                        codex_outputs=[structured_plan_review(state="approved")])
    original = loop._run_validated_agent
    prompts = []
    def stop_at_planner(*args, **kwargs):
        if kwargs["agent"] == config.coder:
            prompts.append(kwargs["prompt"])
            raise RuntimeError("planner correction reached")
        return original(*args, **kwargs)
    monkeypatch.setattr(loop, "_run_validated_agent", stop_at_planner)
    with pytest.raises(RuntimeError, match="planner correction reached"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert "Invalid execution owners: operator" in prompts[0]
    assert "forbidden_side_effects dropped" in prompts[0]
    assert "Preserve the immutable audit" in prompts[0]


@pytest.mark.parametrize("stop", ["round-limit", "correction-exhaustion", "interruption"])
@pytest.mark.parametrize("parallel", [False, True])
def test_invalid_published_plan_remains_correctable_after_stop(tmp_path, monkeypatch, stop, parallel):
    from dataclasses import replace
    from test_child_plan_provenance import _ChildPlanningRunner
    import coding_review_agent_loop.plan_first_loop as loop
    config = make_config(tmp_path, plan_execution_mode="plan-only", max_rounds=1 if stop == "round-limit" else 3,
                         review_parallel=parallel)
    runner = _ChildPlanningRunner(issue_comments=[history(candidate(), config)],
        claude_outputs=[candidate().replace('"kind": "plan_state"', '"kind": "plan_revision", "prior_plan_item_dispositions": []')] * 5,
        codex_outputs=[structured_plan_review(state="approved")])
    with monkeypatch.context() as patch:
        if stop == "interruption":
            original = loop._run_validated_agent
            def interrupt(*args, **kwargs):
                if kwargs["agent"] == config.coder:
                    raise RuntimeError("interrupted correction")
                return original(*args, **kwargs)
            patch.setattr(loop, "_run_validated_agent", interrupt)
        expected = AgentLoopError if stop == "round-limit" else AgentInvocationError if stop == "correction-exhaustion" else RuntimeError
        with pytest.raises(expected):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert not any(cmd[0] == "codex" for cmd, _ in runner.commands)
    assert not any('AGENT_PLAN_STATE: approved' in body for body in runner.comments)
    # Persist every comment, including any correction diagnostic, into a fresh run.
    resumed = FakeRunner(issue_comments=runner.issue_comments)
    prompts = []
    def capture(*args, **kwargs):
        assert kwargs["agent"] == config.coder
        prompts.append(kwargs["prompt"])
        raise RuntimeError("resumed planner correction")
    monkeypatch.setattr(loop, "_run_validated_agent", capture)
    with pytest.raises(RuntimeError, match="resumed planner correction"):
        run_issue_loop(resumed, issue_number=56, config=replace(config, max_rounds=4), plan_first=True)
    assert "Invalid execution owners: operator" in prompts[0]
    assert "Allowed execution owners: one-shot" in prompts[0]


@pytest.mark.parametrize("at_limit", [False, True])
def test_reconciled_primary_invalid_owner_skips_rollover(tmp_path, monkeypatch, at_limit):
    from dataclasses import replace
    import coding_review_agent_loop.plan_first_loop as loop
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=1,
                         review_parallel=True, plan_execution_mode="plan-only",
                         plan_review_policy="primary-then-panel", primary_plan_reviewer="codex")
    runner = FakeRunner(claude_outputs=[candidate()], codex_outputs=[structured_plan_review(state="approved")])
    with monkeypatch.context() as seed:
        seed.setattr(loop, "validate_candidate_ownership", lambda *args: None)
        seed.setattr(loop, "plan_ownership_diagnostic", lambda *args: None)
        with pytest.raises(AgentLoopError, match="reviewer-only plan phase advance"):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert any("reconciliation" in body for body in runner.comments)
    resumed = FakeRunner(issue_comments=runner.issue_comments)
    prompts = []
    def capture(*args, **kwargs):
        assert kwargs["agent"] == config.coder
        prompts.append(kwargs["prompt"])
        raise RuntimeError("planner correction reached")
    monkeypatch.setattr(loop, "_run_validated_agent", capture)
    with pytest.raises(AgentLoopError if at_limit else RuntimeError,
                       match="ownership validation.*planning round budget" if at_limit else "planner correction reached"):
        run_issue_loop(resumed, issue_number=56, config=replace(config, max_rounds=1 if at_limit else 3), plan_first=True)
    if not at_limit:
        assert "Invalid execution owners: operator" in prompts[0]
    assert not any("reviewer-only advance" in body or "Plan review phase advance" in body for body in resumed.comments)
    assert not any(cmd[0] in {"codex", "gemini"} for cmd, _ in resumed.commands)
