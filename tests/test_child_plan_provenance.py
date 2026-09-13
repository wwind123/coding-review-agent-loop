import base64
import json

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_v1_plan_state
from coding_review_agent_loop import orchestrator
from coding_review_agent_loop.comment_rendering import render_execution_recommendation_section
from coding_review_agent_loop.decomposition import (
    CreatedPhaseIssue, PlanPhase, TopologyCheckpoint, approved_plan_hash,
    format_decomposition_parent_summary, format_phase_issue_body,
    format_phase_implementation_handoff_comment,
    normalize_execution_recommendation,
    format_topology_checkpoint, phase_identity,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueComment, IssueContext
from coding_review_agent_loop.issue_pr_handoff import format_issue_pr_handoff_comment
from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata, _plan_subject


def plan_record(plan):
    return _attach_round_metadata(plan, PostedRoundMetadata(
        flow="plan", role="coder", agent="Claude", round_number=1,
        subject=_plan_subject(plan), canonical_plan=plan, raw_structured_coder_response=plan,
    ))


def comment(body):
    return IssueComment(author="bot", created_at="2026-09-13T00:00:00Z", body=body)


def replace_phase_plan_hash(body, plan_hash):
    marker = orchestrator.PHASE_IDENTITY_MARKER_RE.search(body)
    assert marker is not None
    payload = json.loads(
        base64.urlsafe_b64decode(marker.group("payload").encode("ascii")).decode("utf-8")
    )
    payload["plan_hash"] = plan_hash
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")
    return body[:marker.start("payload")] + encoded + body[marker.end("payload"):]


def fresh_staged_plan():
    raw = structured_v1_plan_state()
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["summary"] = "Fresh staged parent plan."
    payload["execution_recommendation"] = {
        "strategy": "staged",
        "rationale": "The two contracts are independently verifiable in order.",
        "staging_feasibility": "safe",
        "scope_items": [
            {
                "scope_item_id": "scope-1",
                "requirement": "Deliver the first contract.",
                "acceptance_criteria": ["The first contract is verified."],
            },
            {
                "scope_item_id": "scope-2",
                "requirement": "Deliver the second contract.",
                "acceptance_criteria": ["The second contract is verified."],
            },
        ],
        "coupling_constraints": [],
        "child_stages": [
            {
                "stage_id": "stage-one",
                "position": 1,
                "title": "First contract",
                "summary": "Deliver the first contract.",
                "deliverables": ["First contract."],
                "non_goals": [],
                "acceptance_criteria": ["The first contract is verified."],
                "depends_on_stage_ids": [],
                "dependency_notes": "No dependencies.",
                "automation": "agent-pr",
                "rollout_risk": "low",
                "compatibility_constraints": [],
                "covered_scope_item_ids": ["scope-1"],
            },
            {
                "stage_id": "stage-two",
                "position": 2,
                "title": "Second contract",
                "summary": "Deliver the second contract.",
                "deliverables": ["Second contract."],
                "non_goals": [],
                "acceptance_criteria": ["The second contract is verified."],
                "depends_on_stage_ids": ["stage-one"],
                "dependency_notes": "After the first contract.",
                "automation": "agent-pr",
                "rollout_risk": "medium",
                "compatibility_constraints": [],
                "covered_scope_item_ids": ["scope-2"],
            },
        ],
        "retained_parent_work": {
            "status": "none", "deliverables": [], "acceptance_criteria": [],
            "covered_scope_item_ids": [],
        },
        "final_integration_work": {
            "status": "none", "deliverables": [], "acceptance_criteria": [],
            "covered_scope_item_ids": [],
        },
        "caveats": [],
    }
    plan = json.dumps(payload) + raw[end:]
    from coding_review_agent_loop.protocol import validate_structured_plan_state

    recommendation = validate_structured_plan_state(plan).execution_recommendation
    assert recommendation is not None
    json_part, footer = plan.split("\n<!-- AGENT_PLAN_STATE:", 1)
    return (
        json_part
        + "\n\n"
        + render_execution_recommendation_section(recommendation)
        + "\n<!-- AGENT_PLAN_STATE:"
        + footer
    )


def fresh_child_contexts(plan, *, stable_stage_id="stage-one"):
    raw_payload, _ = json.JSONDecoder().raw_decode(plan)
    from coding_review_agent_loop.protocol import parse_execution_recommendation_payload

    recommendation = parse_execution_recommendation_payload(
        raw_payload["execution_recommendation"], context="test recommendation"
    )
    assert recommendation is not None
    normalized, retained = normalize_execution_recommendation(
        recommendation, approved_plan=plan, plan_subject=orchestrator._plan_subject(plan)
    )
    phase = normalized.phases[0]
    parent_hash = approved_plan_hash(plan)
    identity = phase_identity(
        parent_issue=55,
        plan_hash=parent_hash,
        topology_source="approved-plan-v1",
        phase_index=1,
        phase=phase,
        stage_id=phase.stage_id,
        execution_strategy_contract_version=1,
    )
    child_body = format_phase_issue_body(
        repo="OWNER/REPO", parent_issue=55, approved_plan=plan, phase=phase,
        created_so_far=(), phase_identity_value=identity,
        topology_source="approved-plan-v1", phase_index=1,
        phase_plan_hash=parent_hash, strategy="staged",
        recommendation_digest=normalized.recommendation_digest,
        execution_strategy_contract_version=1,
    )
    if stable_stage_id != phase.stage_id:
        marker = orchestrator.PHASE_IDENTITY_MARKER_RE.search(child_body)
        assert marker is not None
        marker_payload = json.loads(
            base64.urlsafe_b64decode(marker.group("payload").encode("ascii")).decode("utf-8")
        )
        marker_payload["stage_id"] = stable_stage_id
        encoded = base64.urlsafe_b64encode(
            json.dumps(marker_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).decode("ascii")
        child_body = child_body[:marker.start("payload")] + encoded + child_body[marker.end("payload"):]
    child = IssueContext(
        number=56, repo="OWNER/REPO", title="First contract", body=child_body,
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(comment(format_issue_pr_handoff_comment(
            issue_number=56, pr_number=77,
            pr_url="https://github.com/OWNER/REPO/pull/77",
            pr_head_sha="abc123", flow="approved-plan-implementation", plan_hash=parent_hash,
        )),),
    )
    parent_children = (
        CreatedPhaseIssue(phase=normalized.phases[0], issue_url=child.url, issue_number=56),
        CreatedPhaseIssue(
            phase=normalized.phases[1],
            issue_url="https://github.com/OWNER/REPO/issues/57",
            issue_number=57,
        ),
    )
    summary = format_decomposition_parent_summary(
        parent_issue=55, mode="implement-by-phase", plan_hash=parent_hash,
        created=parent_children, topology_source="approved-plan-v1",
        retained_parent_scope=retained, final_integration_work=normalized.final_integration_work,
        strategy="staged", execution_strategy_contract_version=1,
        recommendation_digest=normalized.recommendation_digest,
        plan_subject=orchestrator._plan_subject(plan),
    )
    handoff = format_phase_implementation_handoff_comment(
        parent_issue=55, mode="implement-by-phase", plan_hash=parent_hash,
        phase_index=1, created=parent_children[0], strategy="staged",
        topology_source="approved-plan-v1", execution_strategy_contract_version=1,
        recommendation_digest=normalized.recommendation_digest,
        plan_subject=orchestrator._plan_subject(plan),
    )
    parent = IssueContext(
        number=55, repo="OWNER/REPO", title="Parent", body="Parent scope.",
        url="https://github.com/OWNER/REPO/issues/55",
        comments=(comment(plan_record(plan)), comment(summary), comment(handoff)),
    )
    return child, parent


@pytest.mark.parametrize("entry", ["issue", "pr"])
@pytest.mark.parametrize(
    "fault", [None, "parent_checkpoint", "phase_identity", "phase_plan_hash", "child_plan"]
)
def test_separately_planned_child_preserves_both_plan_bindings(tmp_path, monkeypatch, entry, fault):
    parent_plan = "Approved parent plan.\n\n## Scope\n- Deliver three sequential stages."
    child_plan = "Approved child plan.\n\n## Scope\n- Implement only the reviewed strategy schema."
    parent_hash = approved_plan_hash(parent_plan)
    child_hash = approved_plan_hash(child_plan)
    assert parent_hash != child_hash
    phase = PlanPhase(
        title="Strategy schema", scope="Implement the contract.", non_goals="No routing.",
        dependency_notes="First stage.", rollout_risk="low", validation="Run focused tests.",
        parent_context="Approved first-stage scope.", automation="agent-pr", depends_on=(),
    )
    identity = phase_identity(parent_issue=55, plan_hash=parent_hash, topology_source="typed",
                              phase_index=1, phase=phase)
    child_body = format_phase_issue_body(
        repo="OWNER/REPO", parent_issue=55, approved_plan=parent_plan, phase=phase,
        created_so_far=(), phase_identity_value="wrong" if fault == "phase_identity" else identity,
        topology_source="typed", phase_index=1, phase_plan_hash=parent_hash,
    )
    if fault == "phase_plan_hash":
        child_body = replace_phase_plan_hash(child_body, "")
    handoff = format_issue_pr_handoff_comment(
        issue_number=56, pr_number=77, pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123", flow="approved-plan-implementation", plan_hash=child_hash,
    )
    child = IssueContext(
        number=56, repo="OWNER/REPO", title="Child", body=child_body,
        url="https://github.com/OWNER/REPO/issues/56",
        comments=tuple([comment(handoff)] if fault == "child_plan" else
                       [comment(plan_record(child_plan)), comment(handoff)]),
    )
    checkpoint = format_topology_checkpoint(TopologyCheckpoint(
        parent_issue=55, plan_hash=parent_hash, mode="decompose-only",
        topology_source="typed", phases=(phase,),
    ))
    parent = IssueContext(
        number=55, repo="OWNER/REPO", title="Parent", body="Parent scope.",
        url="https://github.com/OWNER/REPO/issues/55",
        comments=tuple([comment(plan_record(parent_plan))] +
                       ([] if fault == "parent_checkpoint" else [comment(checkpoint)])),
    )
    monkeypatch.setattr(orchestrator, "get_issue_context",
                        lambda runner, *, config, issue_number: child if issue_number == 56 else parent)
    runner = FakeRunner(
        pr_payload={"number": 77, "body": "Fixes #56", "url": "https://github.com/OWNER/REPO/pull/77"},
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    def run():
        if entry == "issue":
            return orchestrator.run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
        return orchestrator.run_pr_loop(runner, pr_number=77, config=config)

    if fault:
        with pytest.raises(AgentLoopError):
            run()
        assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
        assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)
    else:
        assert run() == 0
        prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
        plan_block = prompt.split("Approved implementation plan context", 1)[1].split(
            "Target child/primary issue context", 1
        )[0]
        assert "Implement only the reviewed strategy schema." in plan_block
        assert "Deliver three sequential stages." not in plan_block


@pytest.mark.parametrize("stable_stage_id, should_fail", [("stage-one", False), ("wrong-stage", True)])
def test_cli_run_pr_loop_validates_fresh_child_phase_identity(
    tmp_path, monkeypatch, stable_stage_id, should_fail
):
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(plan, stable_stage_id=stable_stage_id)
    monkeypatch.setattr(
        orchestrator,
        "get_issue_context",
        lambda runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(
        pr_payload={
            "number": 77,
            "body": "Fixes #56",
            "url": "https://github.com/OWNER/REPO/pull/77",
        },
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    if should_fail:
        with pytest.raises(AgentLoopError, match="stable ID disagrees with its ordinal"):
            orchestrator.run_pr_loop(runner, pr_number=77, config=config)
        assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)
    else:
        assert orchestrator.run_pr_loop(runner, pr_number=77, config=config) == 0
        assert any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)
