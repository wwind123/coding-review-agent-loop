"""Workflow regressions for post-capture limited-authority review."""
import dataclasses
import pytest
import coding_review_agent_loop.pr_loop as loop
from coding_review_agent_loop.capture_acceptance import capture_pending
from coding_review_agent_loop.round_state import _latest_pr_approved_reviews_for_head
from coding_review_agent_loop.pr_loop_support import admitted_capture_history
from test_pr_loop_evidence_conversion import (
    _prepare_verified_recovery, _review, _records, _prompts, _reviews_after,
    authenticated_pr_comments, run_pr_loop,
)


@pytest.mark.parametrize("policy", ["all-reviewers", "selective-intermediate"])
@pytest.mark.parametrize("parallel", [False, True])
def test_capture_dispatch_preserves_code_approval(tmp_path, monkeypatch, policy, parallel):
    runner, config, context = _prepare_verified_recovery(tmp_path, monkeypatch, policy=policy)
    config = dataclasses.replace(config, review_parallel=parallel)
    runner.codex_outputs.append(_review())
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    records = _records(runner)
    capture = next(r for r in reversed(records) if r.metadata.role == "coder")
    [review] = _reviews_after(runner, capture.index)
    assert review.metadata.review_scope is not None
    assert "EVIDENCE-ONLY REVIEW" in _prompts(runner, "codex")[-1]
    assert '"expected_outcome": "outcome row-a"' in _prompts(runner, "codex")[-1]
    comments = authenticated_pr_comments(runner, config, pr_number=77)[0]
    approvals = _latest_pr_approved_reviews_for_head(comments, head_sha="fedcba9",
        configured_reviewers=config.reviewer, approved_plan_context=context)
    assert approvals["Codex"].index < capture.index
    admitted, complete = admitted_capture_history(runner, config, pr_number=77, comments=comments)
    assert complete
    assert not capture_pending(admitted, head="fedcba9", plan=context,
                               required_reviewers=("Codex",), approvals=approvals)
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("parallel", [False, True])
def test_multiple_requesters_receive_only_their_rows(tmp_path, monkeypatch, parallel):
    from agent_loop_helpers import FakeRunner
    from test_pr_loop_evidence_conversion import (
        _config, _plan_context, _implementation_record, _install_derivation, _coder,
        _tagged, _push_external_head,
    )
    from coding_review_agent_loop.errors import HumanDecisionRequiredError
    context = _plan_context("row-a", "row-b")
    limited = {"row-a": "limited", "row-b": "limited"}
    def review(name, **kwargs):
        return _review(reviewer=name, **kwargs)
    runner = FakeRunner(claude_outputs=[_coder(1, 2)],
        codex_outputs=[review("OpenAI Codex", blocking=[_tagged("row-a")]),
                       review("OpenAI Codex", carried=["item-1", "item-2"]),
                       review("OpenAI Codex", kept=["item-3", "item-4"], state="blocking")],
        gemini_outputs=[review("Google Gemini", blocking=[_tagged("row-b")]),
                        review("Google Gemini", carried=["item-1", "item-2"]),
                        review("Google Gemini", kept=["item-3", "item-4"], state="blocking")],
        pr_payload={"comments": [_implementation_record(context, limited)]})
    config = _config(tmp_path, reviewer=("codex", "gemini"), review_parallel=parallel)
    _install_derivation(monkeypatch, context, [limited])
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    _push_external_head(runner, "fedcba9")
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs.append(_coder())
    runner.codex_outputs.extend([review("OpenAI Codex", kept=["item-3", "item-4"], state="blocking"),
                                 review("OpenAI Codex")])
    runner.gemini_outputs.extend([review("Google Gemini", kept=["item-3", "item-4"], state="blocking"),
                                  review("Google Gemini")])
    _install_derivation(monkeypatch, context, [{"row-a": "verified", "row-b": "verified"}])
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    records = _records(runner)
    capture = next(r for r in reversed(records) if r.metadata.role == "coder")
    reviews = _reviews_after(runner, capture.index)
    assert {r.metadata.agent: r.metadata.review_scope["rows"] for r in reviews} == {
        "Codex": ["row-a"], "Gemini": ["row-b"]}
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)
    # The pure assignment receives the effective board from the validated
    # lineage; a removed requester cannot receive work or be substituted.
    from coding_review_agent_loop.capture_acceptance import capture_assignments
    approvals = {r.metadata.agent: r for r in records if r.metadata.role == "reviewer"
                 and r.metadata.subject == "fedcba9" and r.index < capture.index}
    assert capture_assignments(records, head="fedcba9", matrix_identity=context.risk_test_matrix_identity,
        required_reviewers=("Gemini",), approvals=approvals) is None
    restored = capture_assignments(records, head="fedcba9", matrix_identity=context.risk_test_matrix_identity,
        required_reviewers=("Codex", "Gemini"), approvals=approvals)
    assert {name: scope["rows"] for name, scope in restored.items()} == {"Codex": ["row-a"], "Gemini": ["row-b"]}


@pytest.mark.parametrize("parallel", [False, True])
def test_unavailable_scoped_reviewer_keeps_acceptance_pending(tmp_path, monkeypatch, parallel):
    from coding_review_agent_loop.errors import AgentInvocationError
    runner, config, context = _prepare_verified_recovery(tmp_path, monkeypatch)
    config = dataclasses.replace(config, review_parallel=parallel, auto_merge=True)
    # The code-review output is supplied by the fixture. The subsequent scoped
    # reviewer is unavailable; exhaustion exercises the real bounded failure path.
    with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    records = _records(runner)
    capture = next(r for r in reversed(records) if r.metadata.role == "coder")
    assert not _reviews_after(runner, capture.index)
    comments = authenticated_pr_comments(runner, config, pr_number=77)[0]
    approvals = _latest_pr_approved_reviews_for_head(comments, head_sha="fedcba9",
        configured_reviewers=config.reviewer, approved_plan_context=context)
    admitted, _ = admitted_capture_history(runner, config, pr_number=77, comments=comments)
    assert capture_pending(admitted, head="fedcba9", plan=context,
                           required_reviewers=("Codex",), approvals=approvals)
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("replacement", [False, True])
def test_live_qualification_rechecks_identical_and_replacement_evidence(tmp_path, monkeypatch, replacement):
    import json
    from coding_review_agent_loop.round_state import _attach_round_metadata
    from test_pr_loop_evidence_conversion import _comment
    runner, config, context = _prepare_verified_recovery(tmp_path, monkeypatch)
    runner.codex_outputs.append(_review())
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    records = _records(runner)
    capture = next(r for r in reversed(records) if r.metadata.role == "coder")
    evidence = json.loads(json.dumps(capture.metadata.risk_test_matrix_evidence))
    if replacement:
        evidence["rows"][0]["outcome_assertions"] = ["Replacement semantic claim"]
    metadata = dataclasses.replace(capture.metadata, risk_test_matrix_evidence=evidence)
    runner.pr_payload["comments"].append({**_comment(_attach_round_metadata("Repeated capture", metadata)),
                                           "createdAt": "2099-01-01T00:00:00Z"})
    runner.simulate_new_process()
    fresh, _, _, _ = loop._fresh_pr_qualification_snapshot(runner, config=config, pr_number=77,
        issue_context=None, parent_issue_context=None, approved_plan_context=context)
    assert fresh.converted_capture_review_pending == replacement
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("scope", [None, {}, "unknown", {"kind": "unsupported"}])
def test_live_qualification_rejects_malformed_scoped_publication(tmp_path, monkeypatch, scope):
    from coding_review_agent_loop.round_state import _attach_round_metadata
    runner, config, context = _prepare_verified_recovery(tmp_path, monkeypatch)
    runner.codex_outputs.append(_review())
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    records = _records(runner)
    capture = next(r for r in reversed(records) if r.metadata.role == "coder")
    [review] = _reviews_after(runner, capture.index)
    invalid = dataclasses.replace(review.metadata, review_scope=scope, review_scope_present=True)
    runner.pr_payload["comments"][review.index]["body"] = _attach_round_metadata("Malformed scope", invalid)
    runner.simulate_new_process()
    fresh, _, _, _ = loop._fresh_pr_qualification_snapshot(runner, config=config, pr_number=77,
        issue_context=None, parent_issue_context=None, approved_plan_context=context)
    assert fresh.converted_capture_review_pending
    comments = authenticated_pr_comments(runner, config, pr_number=77)[0]
    approvals = _latest_pr_approved_reviews_for_head(comments, head_sha="fedcba9",
        configured_reviewers=config.reviewer, approved_plan_context=context)
    assert approvals["Codex"].index < capture.index
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)
