"""Conservative capture acceptance must schedule a satisfiable review proof."""
import dataclasses

import pytest

import coding_review_agent_loop.pr_loop as loop
from coding_review_agent_loop.capture_acceptance import (
    capture_assignments, capture_pending, full_capture_binding,
)
from coding_review_agent_loop.round_state import _attach_round_metadata
from test_pr_loop_evidence_conversion import (
    _carried_panel_runner, _plan_context, _records, _reviews_after, _push_external_head,
    _coder, _repeat_review, _review, _install_derivation, _comment, _prompts,
    run_pr_loop, HumanDecisionRequiredError,
)


@pytest.mark.parametrize("policy", ["primary-then-panel", "selective-intermediate"])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("fallback", ["conflicting-owners", "missing-approval", "broad-transition"])
def test_conservative_capture_reaches_qualification(tmp_path, monkeypatch, parallel, fallback, policy):
    context = _plan_context("row-a", "row-b")
    runner, config = _carried_panel_runner(tmp_path, monkeypatch, context)
    if policy == "selective-intermediate":
        config = dataclasses.replace(config, pr_review_policy=policy, primary_reviewer=None)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    config = dataclasses.replace(config, review_parallel=parallel)
    _push_external_head(runner, "fedcba9")
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs.append(_coder())
    for name, outputs in [("OpenAI Codex", runner.codex_outputs),
                          ("Google Gemini", runner.gemini_outputs),
                          ("Antigravity", runner.antigravity_outputs)]:
        outputs.extend([_repeat_review(name), _review(reviewer=name)])
    _install_derivation(monkeypatch, context, [{"row-a": "verified", "row-b": "verified"}])
    source = _records(runner)[-1].index
    def captured():
        return any(r.index > source and r.metadata.role == "coder" for r in _records(runner))
    if fallback == "conflicting-owners":
        post = loop.post_pr_comment
        def publish(*args, **kwargs):
            result = post(*args, **kwargs)
            records = _records(runner)
            if records[-1].metadata.role == "coder" and captured():
                conversion = next(r for r in records if r.metadata.evidence_conversion is not None)
                audit = conversion.metadata.evidence_conversion
                conflict = dataclasses.replace(audit, rows=tuple(
                    dataclasses.replace(row, owners=("Gemini",)) for row in audit.rows))
                metadata = dataclasses.replace(conversion.metadata, evidence_conversion=conflict)
                runner.pr_payload["comments"].insert(records[-1].index,
                    _comment(_attach_round_metadata("Conflicting conversion audit", metadata)))
            return result
        monkeypatch.setattr(loop, "post_pr_comment", publish)
    elif fallback == "missing-approval":
        lookup = loop._latest_pr_approved_reviews_for_head
        def approvals(*args, **kwargs):
            result = lookup(*args, **kwargs)
            if captured() and not _reviews_after(runner, next(
                    r.index for r in reversed(_records(runner)) if r.metadata.role == "coder")):
                result.pop("Codex", None)
            return result
        monkeypatch.setattr(loop, "_latest_pr_approved_reviews_for_head", approvals)
    else:
        classify = loop.TransitionClassification
        def classification(kind, *args, **kwargs):
            return classify("broad" if captured() else kind, *args, **kwargs)
        monkeypatch.setattr(loop, "TransitionClassification", classification)
    scopes_seen = []
    narrow = loop.narrow_capture_scopes
    def scopes(*args, **kwargs):
        result = narrow(*args, **kwargs)
        if captured():
            assert capture_pending(args[0], head=kwargs["head"], plan=kwargs["plan"],
                required_reviewers=kwargs["required_reviewers"], primary=kwargs["primary"],
                approvals=kwargs["approvals"])
            scopes_seen.append(result)
        return result
    monkeypatch.setattr(loop, "narrow_capture_scopes", scopes)
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    records = _records(runner)
    capture = next(r for r in reversed(records) if r.metadata.role == "coder")
    assert len([r for r in records if r.index > source and r.metadata.role == "coder"]) == 1
    reviews = _reviews_after(runner, capture.index)
    assert {r.metadata.agent for r in reviews} == {"Codex", "Gemini", "Antigravity"}
    assert len(reviews) == 3  # no repeated qualification/restart rounds
    assert all(r.metadata.review_scope is None and r.metadata.capture_review_binding for r in reviews)
    assert bool(scopes_seen[0]) == (fallback == "broad-transition")
    for command in ("codex", "gemini", "agy"):
        assert "tests/test_row-a.py::test_it" in _prompts(runner, command)[-1]
        assert "EVIDENCE-ONLY REVIEW" not in _prompts(runner, command)[-1]
    if fallback == "conflicting-owners":
        assert capture_assignments(records, head="fedcba9", matrix_identity=context.risk_test_matrix_identity,
            required_reviewers=("Codex", "Gemini", "Antigravity"), primary="Codex",
            approvals={r.metadata.agent: r for r in reviews}) is None
        assert full_capture_binding(records, head="fedcba9", matrix_identity=context.risk_test_matrix_identity)
    runner.simulate_new_process()
    fresh, _, _, _ = loop._fresh_pr_qualification_snapshot(runner, config=config, pr_number=77,
        issue_context=loop.get_issue_context(runner, config=config, issue_number=56),
        parent_issue_context=None, approved_plan_context=context)
    assert not fresh.converted_capture_review_pending
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("fresh", [False, True])
def test_recovery_recomputes_resumed_work_when_scopes_change(monkeypatch, scoped, fresh, bound):
    from types import SimpleNamespace
    import coding_review_agent_loop.capture_acceptance as acceptance
    from coding_review_agent_loop.capture_dispatch import capture_review_recovery
    binding = {"capture": "publication", "head": "head", "matrix": "matrix"}
    monkeypatch.setattr(acceptance, "capture_pending", lambda *args, **kwargs: True)
    monkeypatch.setattr(acceptance, "full_capture_binding", lambda *args, **kwargs: binding)
    record = SimpleNamespace(index=11 if fresh else 9,
        metadata=SimpleNamespace(capture_review_binding=binding if bound else None))
    pending, resumed = capture_review_recovery((), head="head", plan=None,
        required_reviewers=("Codex",), primary=None, approvals={},
        scopes={"Codex": {"rows": ["row-a"]}} if scoped else {},
        resumed={"Codex": record}, boundary=10)
    assert pending == (not scoped)
    assert resumed == ({"Codex": record} if fresh and bound and not scoped else {})
