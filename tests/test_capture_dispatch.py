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
    approvals = _latest_pr_approved_reviews_for_head(comments, head_sha="6946a6f1ac1c65132a6a53600572ceb4af0ee23b",
        configured_reviewers=config.reviewer, approved_plan_context=context)
    assert approvals["Codex"].index < capture.index
    admitted, complete = admitted_capture_history(runner, config, pr_number=77, comments=comments)
    assert complete
    assert not capture_pending(admitted, head="6946a6f1ac1c65132a6a53600572ceb4af0ee23b", plan=context,
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
    _push_external_head(runner, "6946a6f1ac1c65132a6a53600572ceb4af0ee23b")
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
                 and r.metadata.subject == "6946a6f1ac1c65132a6a53600572ceb4af0ee23b" and r.index < capture.index}
    assert capture_assignments(records, head="6946a6f1ac1c65132a6a53600572ceb4af0ee23b", matrix_identity=context.risk_test_matrix_identity,
        required_reviewers=("Gemini",), approvals=approvals) is None
    restored = capture_assignments(records, head="6946a6f1ac1c65132a6a53600572ceb4af0ee23b", matrix_identity=context.risk_test_matrix_identity,
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
    approvals = _latest_pr_approved_reviews_for_head(comments, head_sha="6946a6f1ac1c65132a6a53600572ceb4af0ee23b",
        configured_reviewers=config.reviewer, approved_plan_context=context)
    admitted, _ = admitted_capture_history(runner, config, pr_number=77, comments=comments)
    assert capture_pending(admitted, head="6946a6f1ac1c65132a6a53600572ceb4af0ee23b", plan=context,
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
    approvals = _latest_pr_approved_reviews_for_head(comments, head_sha="6946a6f1ac1c65132a6a53600572ceb4af0ee23b",
        configured_reviewers=config.reviewer, approved_plan_context=context)
    assert approvals["Codex"].index < capture.index
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("seam", ["reviewer", "reconciliation"])
def test_blocking_capture_publication_recovers_without_spool(tmp_path, monkeypatch, parallel, seam):
    from coding_review_agent_loop.review_rounds import _review_round_spool
    from test_pr_loop_evidence_conversion import _tagged
    runner, config, context = _prepare_verified_recovery(tmp_path, monkeypatch, policy="selective-intermediate")
    config = dataclasses.replace(config, review_parallel=parallel, auto_merge=True)
    runner.codex_outputs.append(_review([_tagged("row-a", text="Published semantic defect")]))
    original_post = loop.post_pr_comment
    def post(*args, **kwargs):
        result = original_post(*args, **kwargs)
        latest = _records(runner)[-1]
        if latest.metadata.review_scope is not None and (
                latest.metadata.role == seam or latest.metadata.phase == seam):
            raise KeyboardInterrupt()
        return result
    monkeypatch.setattr(loop, "post_pr_comment", post)
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    publication = next(r for r in reversed(_records(runner)) if r.metadata.role == "reviewer")
    assert publication.metadata.state == "blocking"
    if parallel:
        assert publication.metadata.canonical_reviewer_response
        assert not publication.metadata.new_items
    _review_round_spool(config, surface="pr", number=77,
        round_number=publication.metadata.round_number, subject=publication.metadata.subject).discard()
    monkeypatch.setattr(loop, "post_pr_comment", original_post)
    runner.simulate_new_process()
    original_run = loop._run_validated_agent
    coder_prompts = []
    def run(*args, **kwargs):
        if kwargs.get("role") == "coder":
            coder_prompts.append(kwargs["prompt"])
            raise KeyboardInterrupt()
        if kwargs.get("role") == "reviewer":
            pytest.fail("Published scoped reviewer must not run again")
        return original_run(*args, **kwargs)
    monkeypatch.setattr(loop, "_run_validated_agent", run)
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert len(coder_prompts) == 1
    assert "Published semantic defect" in coder_prompts[0]
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("restored", [False, True])
def test_capture_dispatch_uses_admitted_amendment_lineage(tmp_path, monkeypatch, restored):
    from agent_loop_helpers import FakeRunner
    from pr_loop_test_helpers import _m943_append
    from coding_review_agent_loop.board_amendment import format_reviewer_board_amendment_comment
    from coding_review_agent_loop.errors import HumanDecisionRequiredError
    from test_pr_loop_evidence_conversion import (
        _config, _plan_context, _implementation_record, _install_derivation, _coder,
        _tagged, _push_external_head,
    )
    real_qualify = loop._fresh_pr_qualification_snapshot
    def qualify(*args, **kwargs):
        plan = kwargs.pop("approved_plan_context", None)
        fresh, requirements, _, fresh_config = real_qualify(*args, **kwargs)
        return fresh, requirements, plan, fresh_config
    monkeypatch.setattr(loop, "_fresh_pr_qualification_snapshot", qualify)
    context = _plan_context("row-a", "row-b")
    limited = {"row-a": "limited", "row-b": "limited"}
    runner = FakeRunner(claude_outputs=[_coder(1, 2)],
        codex_outputs=[_review([_tagged("row-a")]),
                       _review(carried=["item-1", "item-2"]),
                       _review(kept=["item-3", "item-4"], state="blocking")],
        gemini_outputs=[_review([_tagged("row-b")], reviewer="Google Gemini"),
                        _review(carried=["item-1", "item-2"], reviewer="Google Gemini"),
                        _review(kept=["item-3", "item-4"], state="blocking", reviewer="Google Gemini")],
        pr_payload={"comments": [_implementation_record(context, limited)]})
    config = _config(tmp_path, reviewer=("codex", "gemini"), pr_review_policy="all-reviewers")
    _install_derivation(monkeypatch, context, [limited])
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    amendment_round = _records(runner)[-1].metadata.round_number
    def amendment(board, removed, added=()):
        return format_reviewer_board_amendment_comment(flow="pr", issue=None, pr_number=77,
            original_required_reviewers=board, policy="all-reviewers", primary_reviewer=None,
            removed_reviewers=removed, restored_reviewers=added, effective_from_round=amendment_round,
            rationale="Backend recovered" if added else "Backend unavailable")
    _m943_append(runner, amendment(("Codex", "Gemini"), ("Gemini",)))
    _push_external_head(runner, "6946a6f1ac1c65132a6a53600572ceb4af0ee23b")
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs.append(_coder())
    runner.codex_outputs.extend([_review(kept=["item-3", "item-4"], state="blocking"), _review()])
    runner.gemini_outputs.extend([
        _review(kept=["item-3", "item-4"], state="blocking", reviewer="Google Gemini"),
        _review(reviewer="Google Gemini")])
    _install_derivation(monkeypatch, context, [{"row-a": "verified", "row-b": "verified"}])
    if restored:
        original_post = loop.post_pr_comment
        def interrupt_capture(*args, **kwargs):
            result = original_post(*args, **kwargs)
            latest = _records(runner)[-1]
            if latest.metadata.role == "coder" and latest.metadata.subject == "6946a6f1ac1c65132a6a53600572ceb4af0ee23b":
                raise KeyboardInterrupt()
            return result
        monkeypatch.setattr(loop, "post_pr_comment", interrupt_capture)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
        monkeypatch.setattr(loop, "post_pr_comment", original_post)
        amendment_round = _records(runner)[-1].metadata.round_number
        _m943_append(runner, amendment(("Codex",), (), ("Gemini",)))
        # Restoration invalidates earlier code approvals. Fresh complete
        # capture-aware reviews first re-establish the restored board contract.
        runner.codex_outputs[:] = [_review()]
        runner.gemini_outputs[:] = [_review(reviewer="Google Gemini")]
    runner.simulate_new_process()
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    records = _records(runner)
    capture = next(r for r in reversed(records) if r.metadata.role == "coder")
    reviews = _reviews_after(runner, capture.index)
    assert {r.metadata.agent for r in reviews} == ({"Codex", "Gemini"} if restored else {"Codex"})
    assert all(r.metadata.review_scope is None and r.metadata.capture_review_binding for r in reviews)
    assert any("Reviewer board amendment applied." in comment for comment in runner.comments)
    effective_config = config if restored else dataclasses.replace(config, reviewer=("codex",))
    fresh, _, _, _ = loop._fresh_pr_qualification_snapshot(runner, config=effective_config,
        pr_number=77, issue_context=None, parent_issue_context=None, approved_plan_context=context)
    assert not fresh.converted_capture_review_pending
    if restored:
        # A replacement capture after fresh restored-seat code approval must
        # dispatch to the original per-row requesters, through the real loop.
        import json
        from coding_review_agent_loop.round_state import _attach_round_metadata
        from test_pr_loop_evidence_conversion import _comment
        evidence = json.loads(json.dumps(capture.metadata.risk_test_matrix_evidence))
        evidence["rows"][0]["outcome_assertions"] = ["Replacement capture after restoration"]
        metadata = dataclasses.replace(capture.metadata, risk_test_matrix_evidence=evidence,
            round_number=records[-1].metadata.round_number + 1,
            **{key: getattr(reviews[-1].metadata, key) for key in (
                "pr_board_root", "pr_board_identity", "pr_board_binding_init",
                "reviewer_board_change_digest", "reviewer_board_amendment_digest")})
        runner.pr_payload["comments"].append({**_comment(_attach_round_metadata("Replacement capture", metadata)),
                                               "createdAt": f"2026-05-23T00:00:{len(runner.pr_payload['comments']):02d}Z"})
        boundary = _records(runner)[-1].index
        runner.codex_outputs[:] = [_review()]
        runner.gemini_outputs[:] = [_review(reviewer="Google Gemini")]
        runner.simulate_new_process()
        assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
        assert {r.metadata.agent: r.metadata.review_scope["rows"] for r in _reviews_after(runner, boundary)} == {
            "Codex": ["row-a"], "Gemini": ["row-b"]}
        fresh, _, _, _ = loop._fresh_pr_qualification_snapshot(runner, config=config,
            pr_number=77, issue_context=None, parent_issue_context=None, approved_plan_context=context)
        assert not fresh.converted_capture_review_pending
    else:
        # Lose the active board's post-capture acceptance and publish a bound
        # approval from the removed requester. It cannot replace the required
        # conservative complete review by the actual effective board.
        from coding_review_agent_loop.capture_acceptance import reviewer_contract
        from coding_review_agent_loop.round_state import _attach_round_metadata
        from coding_review_agent_loop.errors import AgentLoopError
        from test_pr_loop_evidence_conversion import _comment
        metadata = dataclasses.replace(reviews[-1].metadata, agent="Gemini")
        scope = {"version": 1, "kind": "capture-evidence",
                 **metadata.capture_review_binding, "rows": ["row-b"],
                 "contract": reviewer_contract(metadata)}
        metadata = dataclasses.replace(metadata, review_scope=scope, review_scope_present=True)
        for review in reversed(reviews):
            del runner.pr_payload["comments"][review.index]
        runner.pr_payload["comments"].append({**_comment(_attach_round_metadata("Removed requester approval", metadata)),
                                               "createdAt": f"2026-05-23T00:00:{len(runner.pr_payload['comments']):02d}Z"})
        runner.simulate_new_process()
        try:
            fresh, _, _, _ = loop._fresh_pr_qualification_snapshot(runner, config=effective_config,
                pr_number=77, issue_context=None, parent_issue_context=None, approved_plan_context=context)
        except AgentLoopError as exc:
            assert "reviewer" in str(exc).lower() or "board" in str(exc).lower()
        else:
            assert fresh.converted_capture_review_pending
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)
