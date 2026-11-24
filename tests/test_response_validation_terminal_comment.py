"""Response-validation tests: structured no-PR terminal comments (#1543)."""

import json
from coding_review_agent_loop.protocol import validate_structured_issue_implementation
import coding_review_agent_loop.response_validation as response_validation
from issue_loop_test_helpers import FakeRunner
from agent_loop_helpers import make_config, structured_issue_implementation


def test_959_structured_no_pr_terminal_comment_omits_matrix_evidence(tmp_path):
    raw = structured_issue_implementation(pr_number=None, summary="Blocked before a PR.")
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "row-0",
        "execution_refs": ["turn:observation-1"],
        "test_identifiers": ["tests/test_x.py::test_y"],
        "test_locations": ["tests/test_x.py"],
        "workflow_path_claim": "Semantic claim only.",
        "outcome_assertions": ["outcome"],
        "forbidden_effect_assertions": ["forbidden"],
        "caveats": [],
    }]
    parsed = validate_structured_issue_implementation(json.dumps(payload) + raw[end:])
    assert parsed is not None and parsed.pr_number is None
    assert parsed.risk_test_matrix_evidence is None
    runner = FakeRunner()
    config = make_config(tmp_path, coder="claude", reviewer="codex")

    response_validation._post_structured_issue_implementation_terminal_comment(
        runner, config=config, issue_number=56, parsed=parsed, model_used=None
    )

    assert runner.comments
    body = runner.comments[-1]
    assert "Blocked before a PR." in body
    assert "matrix evidence" not in body
    assert "<details>" not in body


def test_no_pr_blocked_requirement_still_posts_terminal_comment_and_stops(tmp_path, monkeypatch):
    """#2120 keeps the no-PR stop: terminal comment, no PR authentication or loop."""
    import pytest

    import coding_review_agent_loop.execution_policy as execution_policy
    import coding_review_agent_loop.issue_implementation as issue_implementation
    import coding_review_agent_loop.issue_loop as issue_loop
    import coding_review_agent_loop.issue_recovery as issue_recovery
    import coding_review_agent_loop.plan_execution as plan_execution
    import coding_review_agent_loop.pr_loop as pr_loop
    import coding_review_agent_loop.validated_agent as validated_agent_module
    from agent_loop_helpers import patch_attr_in
    from coding_review_agent_loop.errors import AgentLoopError
    from coding_review_agent_loop.github import HumanReviewRequirement, IssueContext

    requirement = HumanReviewRequirement(
        source_type="Issue comment",
        author="maintainer",
        created_at="2026-01-01T00:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56#issuecomment-1",
        body="The operator runs the canary on dev after deploy.",
    )
    issue_context = IssueContext(
        number=56, repo="OWNER/REPO", title="Issue", body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56", comments=(),
        human_requirements=(requirement,),
    )
    runner = FakeRunner(claude_outputs=[structured_issue_implementation(
        pr_number=None,
        summary="No PR: the signed requirement is blocked.",
        human_requirement_dispositions=[{
            "requirement_id": requirement.requirement_id,
            "disposition": "blocked",
            "evidence": "Waiting for the operator canary.",
        }],
    )])
    config = make_config(tmp_path)
    gate_calls = []
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, issue_recovery, plan_execution,), "resolve_canonical_pr_for_issue", lambda *args, **kwargs: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation", lambda *args, **kwargs: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *args, **kwargs: None)
    for name, owners in (
        ("validate_open_pr", (response_validation, pr_loop, issue_implementation, issue_loop)),
        ("run_pr_loop", (pr_loop, issue_implementation, plan_execution, issue_loop)),
    ):
        patch_attr_in(monkeypatch, owners, name, lambda *_a, _name=name, **_k: gate_calls.append(_name))

    with pytest.raises(AgentLoopError):
        issue_implementation._implement_approved_issue(
            runner, issue_number=56, approved_plan="Approved implementation plan.",
            config=config, memory=None, issue_context=issue_context, coder_session_id=None,
            usage_context=validated_agent_module._new_usage_context(config),
        )

    assert gate_calls == []
    assert len(runner.comments) == 1
    assert "No pull request was accepted for handoff." in runner.comments[0]
    assert "Waiting for the operator canary." in runner.comments[0]
