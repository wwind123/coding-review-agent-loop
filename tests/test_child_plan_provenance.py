import pytest

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop import orchestrator
from coding_review_agent_loop.decomposition import (
    PlanPhase, TopologyCheckpoint, approved_plan_hash, format_phase_issue_body,
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


@pytest.mark.parametrize("entry", ["issue", "pr"])
@pytest.mark.parametrize("fault", [None, "parent_checkpoint", "phase_identity", "child_plan"])
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
