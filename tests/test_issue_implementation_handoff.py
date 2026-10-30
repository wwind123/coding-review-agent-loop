"""Issue-implementation tests: issue-to-PR handoff and coder receipts (#1543)."""

import json
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.agents.base import AgentResult
from coding_review_agent_loop.comment_rendering import render_risk_test_matrix_section
from coding_review_agent_loop.github import IssueContext
from coding_review_agent_loop.agent_failure import ValidatedAgentResponse
from coding_review_agent_loop.protocol import (
    validate_structured_issue_implementation,
    parse_risk_test_matrix,
    risk_test_matrix_identity,
)
from coding_review_agent_loop.round_state import make_approved_plan_context
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.triage as triage
from agent_loop_helpers import patch_attr_in
from issue_loop_test_helpers import FakeRunner, _metadata_from_public_comment
from agent_loop_helpers import make_config, structured_issue_implementation
import coding_review_agent_loop.validated_agent as validated_agent_module


def _implementation_matrix_context():
    matrix = parse_risk_test_matrix(
        {
            "applicability": "applicable",
            "rows": [{
                "row_id": "implementation-derived-evidence",
                "label": "Implementation evidence reaches the public handoff",
                "entry_path_or_mode": "issue implementation",
                "initial_state": "approved plan and new implementation PR",
                "event": "the authenticated head is reconciled",
                "expected_outcome": "canonical evidence is persisted",
                "forbidden_side_effects": ["Do not lose the created PR."],
                "proposed_test_level": "orchestrator",
                "proposed_test_location": "tests/test_orchestrator_issue.py",
                "applicability": "required",
                "related_scope_item_ids": ["scope-orchestration-integration"],
                "execution_owner": "one-shot",
            }],
            "important_exclusions": ["Planned tests are not evidence."],
        }
    )
    approved_plan = (
        "Approved implementation plan.\n\n"
        + render_risk_test_matrix_section(matrix)
    )
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        approved_plan,
        source_locator="test approved implementation plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    return approved_plan, context


@pytest.mark.parametrize(
    ("checkout_head", "expected_status", "expected_diagnostic"),
    [
        ("6367c48dd193d56ea7b0baad25b19455e529f5ee", "verified", None),
        ("checkout-mismatch", "stale/unverified", "checkout-head-mismatch"),
    ],
)
def test_issue_implementation_keeps_pr_and_persists_derived_evidence_after_head_authentication(
    tmp_path, monkeypatch, checkout_head, expected_status, expected_diagnostic
):
    approved_plan, plan_context = _implementation_matrix_context()
    observation = _workflow_observation(
        execution_ref="coder-turn:observation-1",
        receipt_id="receipt-current-head",
        head="6367c48dd193d56ea7b0baad25b19455e529f5ee",
    )
    implementation_text = _semantic_issue_implementation_text(observation.execution_ref)
    parsed = validate_structured_issue_implementation(
        implementation_text,
        delivered_risk_test_matrix_row_ids=("implementation-derived-evidence",),
        execution_catalog=(observation,),
    )
    assert parsed is not None
    coder_response = ValidatedAgentResponse(
        text=implementation_text,
        session_id=None,
        marker_value=parsed,
        acquisition_test_turn_id="coder-turn",
        acquisition_test_observations=(observation,),
    )
    runner = FakeRunner(
        pr_payload={"body": "Fixes #56", "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee"},
    )
    config = make_config(tmp_path, coder="claude")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(),
    )
    patch_attr_in(monkeypatch, (validated_agent_module, discuss_loop, pr_loop, issue_implementation, plan_first_loop_module, triage, issue_loop), "_run_validated_agent",
        lambda *_args, **_kwargs: (
            setattr(runner, "git_head", "6367c48dd193d56ea7b0baad25b19455e529f5ee-agent-1") or coder_response
        ),
    )
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue",
        lambda *_args, **_kwargs: None,
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation",
        lambda *_args, **_kwargs: None,
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation",
        lambda *_args, **_kwargs: None,
    )
    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "run_pr_loop",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        response_validation,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head=checkout_head,
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )
    monkeypatch.setattr(
        response_validation,
        "reconcile_test_observations",
        lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
    )

    assert (
        issue_implementation._implement_approved_issue(
            runner,
            issue_number=56,
            approved_plan=approved_plan,
            config=config,
            memory=None,
            issue_context=issue_context,
            coder_session_id=None,
            usage_context=validated_agent_module._new_usage_context(config),
            approved_plan_context=plan_context,
        )
        == 0
    )

    raw_comments = [
        item["body"] for item in runner.pr_payload.get("comments", [])
        if isinstance(item, dict) and isinstance(item.get("body"), str)
    ] + [
        item["body"] for item in runner.issue_comments
        if isinstance(item, dict) and isinstance(item.get("body"), str)
    ]
    metadata = next(
        _metadata_from_public_comment(comment)
        for comment in raw_comments
        if "AGENT_LOOP_META: " in comment
        and _metadata_from_public_comment(comment).role == "coder"
    )
    assert metadata.risk_test_matrix_evidence is not None
    assert metadata.risk_test_matrix_evidence["rows"][0]["status"] == expected_status
    # #959: the establishing comment renders the full list and anchors to itself.
    assert metadata.risk_test_matrix_evidence_full_round == metadata.round_number
    assert metadata.risk_test_matrix_evidence_full_round_status == "valid"
    coder_comment = next(
        comment for comment in raw_comments
        if "AGENT_LOOP_META: " in comment
        and _metadata_from_public_comment(comment).role == "coder"
    )
    assert "<summary>Full matrix evidence (1 row)</summary>" in coder_comment
    assert "unchanged since round" not in coder_comment
    if expected_diagnostic is None:
        assert not metadata.risk_test_matrix_diagnostics
    else:
        assert any(
            item["code"] == expected_diagnostic
            for item in metadata.risk_test_matrix_diagnostics
        )
    assert any("AGENT_ISSUE_PR_HANDOFF" in comment for comment in runner.comments)
    assert any(
        "implementation-derived-evidence" in comment for comment in runner.comments
    )


def _semantic_issue_implementation_text(execution_ref: str) -> str:
    raw = structured_issue_implementation(pr_number=77)
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "implementation-derived-evidence",
        "execution_refs": [execution_ref],
        "test_identifiers": ["tests/test_orchestrator_issue.py::test_workflow"],
        "test_locations": ["tests/test_orchestrator_issue.py"],
        "workflow_path_claim": "The issue implementation handoff reached post-head derivation.",
        "outcome_assertions": ["The selected managed observation passed."],
        "forbidden_effect_assertions": ["The PR handoff was not discarded."],
        "caveats": [],
    }]
    return json.dumps(payload) + raw[end:]


def _workflow_observation(*, execution_ref: str, receipt_id: str, head: str):
    return SimpleNamespace(
        execution_ref=execution_ref,
        receipt_id=receipt_id,
        command=("python3", "-m", "pytest", "tests/test_orchestrator_issue.py", "-q"),
        normalized_command="python3 -m pytest tests/test_orchestrator_issue.py -q",
        outcome="passed",
        provenance="parent-observed",
        turn_id="coder-turn",
        attribution={
            "state": "current-head",
            "head": head,
            "tracked_digest": "tree-current",
            "stable": True,
            "untracked_input": False,
            "caveats": [],
        },
        environment_state="not-compared",
        superseded_by=None,
        caveats=(),
        wrapper_bootstrap="verified",
        inner_exec="started",
        suite_start="verified",
    )


@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_diagnostic"),
    [
        ("success", "verified", None),
        ("exhausted", "stale/unverified", "semantic-correction-exhausted"),
        ("head-race", "stale/unverified", "head-changed-during-correction"),
    ],
)
def test_issue_implementation_runs_bounded_post_auth_correction_without_losing_pr(
    tmp_path, monkeypatch, mode, expected_status, expected_diagnostic
):
    approved_plan, plan_context = _implementation_matrix_context()
    wrong = _workflow_observation(
        execution_ref="coder-turn:observation-1",
        receipt_id="receipt-wrong-head",
        head="old-head",
    )
    current = _workflow_observation(
        execution_ref="coder-turn:observation-2",
        receipt_id="receipt-current-head",
        head="6367c48dd193d56ea7b0baad25b19455e529f5ee",
    )
    initial_text = _semantic_issue_implementation_text(wrong.execution_ref)
    corrected_text = _semantic_issue_implementation_text(current.execution_ref)
    initial_parsed = validate_structured_issue_implementation(
        initial_text,
        delivered_risk_test_matrix_row_ids=("implementation-derived-evidence",),
        execution_catalog=(wrong, current),
    )
    assert initial_parsed is not None
    runner = FakeRunner(pr_payload={"body": "Fixes #56", "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee"})
    config = make_config(tmp_path, coder="claude")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(),
    )
    coder_response = ValidatedAgentResponse(
        text=initial_text,
        session_id="coder-session",
        marker_value=initial_parsed,
        acquisition_test_turn_id="coder-turn",
        acquisition_test_observations=(wrong, current),
    )
    run_pr_calls = []
    patch_attr_in(monkeypatch, (validated_agent_module, discuss_loop, pr_loop, issue_implementation, plan_first_loop_module, triage, issue_loop), "_run_validated_agent", lambda *_a, **_k: coder_response
    )
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", lambda *_a, **_k: None
    )
    real_get_pr_review_context = pr_loop_support.get_observed_pr_review_context
    review_context_calls = 0

    def get_pr_review_context_with_optional_race(*args, **kwargs):
        nonlocal review_context_calls
        review_context_calls += 1
        if mode == "head-race" and review_context_calls >= 2:
            runner.pr_payload["headRefOid"] = "a607eb2974b572fbd8ce0410e987804f1e0775a6"
        return real_get_pr_review_context(*args, **kwargs)

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "get_observed_pr_review_context",
        get_pr_review_context_with_optional_race,
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation",
        lambda *_a, **_k: None,
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: None
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "validate_assigned_head_advanced", lambda **_k: None
    )
    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "run_pr_loop",
        lambda *_a, **kwargs: run_pr_calls.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        response_validation,
        "reconcile_test_observations",
        lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
    )
    snapshot_heads = iter(
        ("6367c48dd193d56ea7b0baad25b19455e529f5ee", "6367c48dd193d56ea7b0baad25b19455e529f5ee") if mode != "head-race" else ("6367c48dd193d56ea7b0baad25b19455e529f5ee", "a607eb2974b572fbd8ce0410e987804f1e0775a6")
    )
    monkeypatch.setattr(
        response_validation,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head=next(snapshot_heads),
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )
    patch_attr_in(monkeypatch, (validated_agent_module, response_validation), "run_agent_result",
        lambda *_a, **_k: SimpleNamespace(
            text=("not a structured response" if mode == "exhausted" else corrected_text)
        ),
    )
    result = issue_implementation._implement_approved_issue(
        runner,
        issue_number=56,
        approved_plan=approved_plan,
        config=config,
        memory=None,
        issue_context=issue_context,
        coder_session_id=None,
        usage_context=validated_agent_module._new_usage_context(config),
        approved_plan_context=plan_context,
    )

    assert result == 0
    assert run_pr_calls and run_pr_calls[0]["pr_number"] == 77
    raw_comments = [
        item["body"] for item in runner.pr_payload.get("comments", [])
        if isinstance(item, dict) and isinstance(item.get("body"), str)
    ]
    metadata = _metadata_from_public_comment(
        next(comment for comment in raw_comments if "AGENT_LOOP_META: " in comment)
    )
    evidence_row = metadata.risk_test_matrix_evidence["rows"][0]
    assert evidence_row["status"] == expected_status
    if expected_diagnostic is not None:
        assert any(
            item["code"] == expected_diagnostic
            for item in metadata.risk_test_matrix_diagnostics
        )
    else:
        assert any(
            item["code"] == "wrong-head"
            for item in metadata.risk_test_matrix_diagnostics
        ) is False


def _selector_only_issue_implementation_text(execution_ref: str) -> str:
    """The #1187 shape: a claim with only row_id and execution_refs."""
    raw = structured_issue_implementation(pr_number=77)
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "implementation-derived-evidence",
        "execution_refs": [execution_ref],
    }]
    return json.dumps(payload) + raw[end:]


def _still_incomplete_issue_implementation_text(execution_ref: str) -> str:
    raw = structured_issue_implementation(pr_number=77)
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "implementation-derived-evidence",
        "execution_refs": [execution_ref],
        "test_identifiers": ["tests/test_orchestrator_issue.py::test_workflow"],
        "workflow_path_claim": None,
        "outcome_assertions": [],
    }]
    return json.dumps(payload) + raw[end:]


@pytest.mark.parametrize("mode", ["completes", "still-incomplete"])
def test_issue_implementation_accepts_selector_only_claim_and_corrects_once_after_handoff(
    tmp_path, monkeypatch, mode
):
    """#849: a claim missing all semantic facts must not reject the envelope.

    The PR contract and issue handoff are recorded before the single
    post-authentication correction runs, implementation is not re-run, and
    repair is never invoked.
    """
    approved_plan, plan_context = _implementation_matrix_context()
    current = _workflow_observation(
        execution_ref="coder-turn:observation-1",
        receipt_id="receipt-current-head",
        head="6367c48dd193d56ea7b0baad25b19455e529f5ee",
    )
    initial_text = _selector_only_issue_implementation_text(current.execution_ref)
    corrected_text = (
        _semantic_issue_implementation_text(current.execution_ref)
        if mode == "completes"
        else _still_incomplete_issue_implementation_text(current.execution_ref)
    )
    initial_parsed = validate_structured_issue_implementation(
        initial_text,
        delivered_risk_test_matrix_row_ids=("implementation-derived-evidence",),
        execution_catalog=(current,),
    )
    assert initial_parsed is not None
    assert initial_parsed.risk_test_matrix_claims.claims[0].workflow_path_claim == ""
    runner = FakeRunner(pr_payload={"body": "Fixes #56", "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee"})
    config = make_config(tmp_path, coder="claude")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(),
    )
    coder_response = ValidatedAgentResponse(
        text=initial_text,
        session_id="coder-session",
        marker_value=initial_parsed,
        acquisition_test_turn_id="coder-turn",
        acquisition_test_observations=(current,),
    )
    implementation_calls = []
    recorded = []
    correction_prompts = []
    run_pr_calls = []

    def fake_validated_agent(*_a, **_k):
        implementation_calls.append(1)
        return coder_response

    real_contract_record = issue_implementation.post_trusted_pr_contract_record
    real_handoff = pr_loop.post_issue_pr_handoff_comment

    def recording_contract_record(*args, **kwargs):
        result = real_contract_record(*args, **kwargs)
        recorded.append("pr-contract")
        return result

    def recording_handoff(*args, **kwargs):
        result = real_handoff(*args, **kwargs)
        recorded.append(("issue-handoff", kwargs.get("pr_number")))
        return result

    def fake_correction(*_a, **kwargs):
        assert kwargs.get("label") == "semantic-evidence-correction"
        # Inspect the durable records at call time, not after the flow ends.
        assert "pr-contract" in recorded
        assert ("issue-handoff", 77) in recorded
        correction_prompts.append(kwargs["prompt"])
        return SimpleNamespace(text=corrected_text)

    def forbidden_repair(*_a, **_k):
        raise AssertionError("repair must not run for an incomplete semantic claim")

    patch_attr_in(monkeypatch, (validated_agent_module, discuss_loop, pr_loop, issue_implementation, plan_first_loop_module, triage, issue_loop), "_run_validated_agent", fake_validated_agent)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "post_trusted_pr_contract_record", recording_contract_record)
    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "post_issue_pr_handoff_comment", recording_handoff)
    patch_attr_in(monkeypatch, (validated_agent_module, response_validation), "run_agent_result", fake_correction)
    monkeypatch.setattr(validated_agent_module, "attempt_repair", forbidden_repair)
    monkeypatch.setattr(validated_agent_module, "execute_repair", forbidden_repair)
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", lambda *_a, **_k: None
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation", lambda *_a, **_k: None
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: None
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "validate_assigned_head_advanced", lambda **_k: None
    )
    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "run_pr_loop",
        lambda *_a, **kwargs: run_pr_calls.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        response_validation,
        "reconcile_test_observations",
        lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
    )
    monkeypatch.setattr(
        response_validation,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )

    result = issue_implementation._implement_approved_issue(
        runner,
        issue_number=56,
        approved_plan=approved_plan,
        config=config,
        memory=None,
        issue_context=issue_context,
        coder_session_id=None,
        usage_context=validated_agent_module._new_usage_context(config),
        approved_plan_context=plan_context,
    )

    assert result == 0
    assert implementation_calls == [1]
    assert len(correction_prompts) == 1
    assert "forbidden_effect_assertions" in correction_prompts[0]
    assert run_pr_calls and run_pr_calls[0]["pr_number"] == 77
    raw_comments = [
        item["body"] for item in runner.pr_payload.get("comments", [])
        if isinstance(item, dict) and isinstance(item.get("body"), str)
    ]
    metadata = _metadata_from_public_comment(
        next(comment for comment in raw_comments if "AGENT_LOOP_META: " in comment)
    )
    evidence_row = metadata.risk_test_matrix_evidence["rows"][0]
    codes = [item["code"] for item in metadata.risk_test_matrix_diagnostics]
    if mode == "completes":
        assert evidence_row["status"] == "verified"
        assert "incomplete-semantic-claim" not in codes
    else:
        assert evidence_row["status"] == "stale/unverified"
        assert evidence_row["evidence_citations"] == []
        assert evidence_row["workflow_path_claim"] == ""
        assert evidence_row["outcome_assertions"] == []
        assert "incomplete-semantic-claim" in codes
        assert "semantic-correction-exhausted" in codes


_COMMAND_STRING_REF = (
    "python3 -m pytest tests/test_round_transport.py -q -p no:cacheprovider"
)


def _command_string_issue_implementation_text() -> str:
    """The #859 shape: raw commands in execution_refs, no broker selector."""
    raw = structured_issue_implementation(pr_number=77)
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "implementation-derived-evidence",
        "execution_refs": [_COMMAND_STRING_REF, "python3 -m pytest tests/ -q"],
        "test_identifiers": ["tests/test_orchestrator_issue.py::test_workflow"],
        "test_locations": ["tests/test_orchestrator_issue.py"],
        "workflow_path_claim": "The issue implementation handoff reached post-head derivation.",
        "outcome_assertions": ["The command-string claim was dropped."],
        "forbidden_effect_assertions": ["The PR handoff was not discarded."],
        "caveats": [],
    }]
    return json.dumps(payload) + raw[end:]


@pytest.mark.parametrize("mode", ["corrected", "still-invalid"])
def test_issue_implementation_with_command_string_refs_is_handed_off_and_unverified(
    tmp_path, monkeypatch, mode
):
    """#859: command-string execution_refs must not strand a tested PR.

    The envelope parses with the refs dropped, the PR is authenticated and
    handed off, implementation is not re-run, repair is never invoked, and at
    most one post-authentication correction runs.
    """
    approved_plan, plan_context = _implementation_matrix_context()
    current = _workflow_observation(
        execution_ref="coder-turn:observation-1",
        receipt_id="receipt-current-head",
        head="6367c48dd193d56ea7b0baad25b19455e529f5ee",
    )
    initial_text = _command_string_issue_implementation_text()
    corrected_text = (
        _semantic_issue_implementation_text(current.execution_ref)
        if mode == "corrected"
        else _command_string_issue_implementation_text()
    )
    # Real orchestrator validator with the live current-turn catalog.
    initial_parsed = response_validation._validate_issue_implementation_response(
        initial_text,
        human_requirements=(),
        delivered_risk_test_matrix=plan_context.risk_test_matrix_payload,
        delivered_risk_test_matrix_identity=plan_context.risk_test_matrix_identity,
        require_risk_test_matrix_contract=True,
        authoritative_test_observations=(current,),
        delivered_risk_test_matrix_row_ids=plan_context.risk_test_matrix_expected_row_ids,
        execution_catalog=(current,), architecture_status_mode="legacy",
    )
    assert initial_parsed.pr_number == 77
    initial_claim = initial_parsed.risk_test_matrix_claims.claims[0]
    assert initial_claim.execution_refs == ()
    assert initial_claim.dropped_execution_refs == (
        _COMMAND_STRING_REF, "python3 -m pytest tests/ -q",
    )
    runner = FakeRunner(pr_payload={"body": "Fixes #56", "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee"})
    config = make_config(tmp_path, coder="claude")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(),
    )
    coder_response = ValidatedAgentResponse(
        text=initial_text,
        session_id="coder-session",
        marker_value=initial_parsed,
        acquisition_test_turn_id="coder-turn",
        acquisition_test_observations=(current,),
    )
    implementation_calls = []
    recorded = []
    correction_prompts = []
    run_pr_calls = []

    def fake_validated_agent(*_a, **_k):
        implementation_calls.append(1)
        return coder_response

    real_handoff = pr_loop.post_issue_pr_handoff_comment

    def recording_handoff(*args, **kwargs):
        result = real_handoff(*args, **kwargs)
        recorded.append(("issue-handoff", kwargs.get("pr_number")))
        return result

    def fake_correction(*_a, **kwargs):
        assert kwargs.get("label") == "semantic-evidence-correction"
        assert ("issue-handoff", 77) in recorded
        correction_prompts.append(kwargs["prompt"])
        return SimpleNamespace(text=corrected_text)

    def forbidden_repair(*_a, **_k):
        raise AssertionError("repair must not run for dropped execution_refs")

    patch_attr_in(monkeypatch, (validated_agent_module, discuss_loop, pr_loop, issue_implementation, plan_first_loop_module, triage, issue_loop), "_run_validated_agent", fake_validated_agent)
    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "post_issue_pr_handoff_comment", recording_handoff)
    patch_attr_in(monkeypatch, (validated_agent_module, response_validation), "run_agent_result", fake_correction)
    monkeypatch.setattr(validated_agent_module, "attempt_repair", forbidden_repair)
    monkeypatch.setattr(validated_agent_module, "execute_repair", forbidden_repair)
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", lambda *_a, **_k: None
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation", lambda *_a, **_k: None
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: None
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "validate_assigned_head_advanced", lambda **_k: None
    )
    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "run_pr_loop",
        lambda *_a, **kwargs: run_pr_calls.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        response_validation,
        "reconcile_test_observations",
        lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
    )
    monkeypatch.setattr(
        response_validation,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )

    result = issue_implementation._implement_approved_issue(
        runner,
        issue_number=56,
        approved_plan=approved_plan,
        config=config,
        memory=None,
        issue_context=issue_context,
        coder_session_id=None,
        usage_context=validated_agent_module._new_usage_context(config),
        approved_plan_context=plan_context,
    )

    assert result == 0
    assert implementation_calls == [1]
    assert len(correction_prompts) == 1
    assert "unknown-execution-ref" in correction_prompts[0]
    assert "Command strings and handles outside this catalog are dropped" in correction_prompts[0]
    assert ("issue-handoff", 77) in recorded
    assert any("AGENT_ISSUE_PR_HANDOFF" in comment for comment in runner.comments)
    assert run_pr_calls and run_pr_calls[0]["pr_number"] == 77
    raw_comments = [
        item["body"] for item in runner.pr_payload.get("comments", [])
        if isinstance(item, dict) and isinstance(item.get("body"), str)
    ]
    metadata = _metadata_from_public_comment(
        next(comment for comment in raw_comments if "AGENT_LOOP_META: " in comment)
    )
    evidence_row = metadata.risk_test_matrix_evidence["rows"][0]
    codes = [item["code"] for item in metadata.risk_test_matrix_diagnostics]
    if mode == "corrected":
        assert evidence_row["status"] == "verified"
        assert "unknown-execution-ref" not in codes
    else:
        assert evidence_row["status"] == "stale/unverified"
        assert evidence_row["evidence_citations"] == []
        assert evidence_row["outcome_assertions"] == ["The command-string claim was dropped."]
        assert any(_COMMAND_STRING_REF in caveat for caveat in evidence_row["caveats"])
        assert "unknown-execution-ref" in codes
        assert "semantic-correction-exhausted" in codes


def test_issue_implementation_reasks_unverified_citation_through_real_call_site(
    tmp_path, monkeypatch
):
    """#1240: the real approved-plan hand-off re-asks once and hands off PR #77."""
    approved_plan, plan_context = _implementation_matrix_context()
    unverified = _workflow_observation(
        execution_ref="turn-1:observation-1", receipt_id="receipt-1", head="6367c48dd193d56ea7b0baad25b19455e529f5ee"
    )
    unverified.turn_id = "turn-1"
    unverified.suite_start = "not-started"
    fresh = _workflow_observation(
        execution_ref="turn-2:observation-1", receipt_id="receipt-2", head="6367c48dd193d56ea7b0baad25b19455e529f5ee"
    )
    fresh.turn_id = "turn-2"
    runner = FakeRunner(pr_payload={"body": "Fixes #56", "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee"})
    config = make_config(tmp_path, coder="claude")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(),
    )
    results = [
        AgentResult(
            text=_semantic_issue_implementation_text(unverified.execution_ref),
            returncode=0,
            session_id="coder-session",
            test_turn_id="turn-1",
            test_turn_observations=(unverified,),
        ),
        AgentResult(
            text=_semantic_issue_implementation_text(fresh.execution_ref),
            returncode=0,
            session_id="coder-session",
            test_turn_id="turn-2",
            test_turn_observations=(fresh,),
        ),
    ]
    dispatches = []

    def fake_run_agent_result(*_a, **kwargs):
        dispatches.append(kwargs)
        return results[len(dispatches) - 1]

    def forbid_repair(*_a, **_k):
        pytest.fail("a citation re-ask must not invoke structured repair")

    run_pr_calls = []
    patch_attr_in(monkeypatch, (validated_agent_module, response_validation), "run_agent_result", fake_run_agent_result)
    patch_attr_in(monkeypatch, (validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair", forbid_repair)
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", lambda *_a, **_k: None
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation", lambda *_a, **_k: None
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: None
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "validate_assigned_head_advanced", lambda **_k: None
    )
    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "run_pr_loop",
        lambda *_a, **kwargs: run_pr_calls.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        response_validation, "reconcile_test_observations",
        lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
    )
    monkeypatch.setattr(
        response_validation, "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head="6367c48dd193d56ea7b0baad25b19455e529f5ee", tracked_digest="tree-current", complete=True,
            stable=True, status_clean=True,
        ),
    )
    result = issue_implementation._implement_approved_issue(
        runner,
        issue_number=56,
        approved_plan=approved_plan,
        config=config,
        memory=None,
        issue_context=issue_context,
        coder_session_id=None,
        usage_context=validated_agent_module._new_usage_context(config),
        approved_plan_context=plan_context,
    )

    assert result == 0
    assert len(dispatches) == 2
    assert "Previous response not accepted: test evidence" in dispatches[1]["prompt"]
    assert dispatches[1]["session_id"] == "coder-session"
    assert run_pr_calls and run_pr_calls[0]["pr_number"] == 77
    assert any("AGENT_ISSUE_PR_HANDOFF" in comment for comment in runner.comments)
    assert not any("turn-1:observation-1" in comment for comment in runner.comments)


@pytest.mark.parametrize('one_shot', [False, True])
@pytest.mark.parametrize('outcome', ['desync', 'invalid', 'converged'])
def test_approved_resume_observation_precedes_contract_and_handoffs(tmp_path, monkeypatch, one_shot, outcome):
    from coding_review_agent_loop.pr_head_sync import PrHeadObservationError
    runner = FakeRunner(pr_payload={'body': 'Fixes #56'})
    runner.branch_tip = 'b' * 40
    if outcome == 'invalid': runner.pr_payload['headRefOid'] = 'invalid'
    original = runner.run

    def run(args, **kwargs):
        if args == ['sleep', '2'] and outcome == 'converged':
            runner.pr_payload.update(headRefOid='b' * 40, url='https://example.test/latest', title='Latest context')
        return original(args, **kwargs)

    runner.run = run
    delegated = []
    monkeypatch.setattr(issue_implementation, 'run_pr_loop', lambda *a, **k: delegated.append(k) or 0)
    monkeypatch.setattr(issue_implementation, 'resolve_canonical_pr_for_issue', lambda *a, **k: SimpleNamespace(
        source='legacy-closing-reference', pr_number=77, metadata=None, evidence_summary='authenticated fixture'))
    config = make_config(tmp_path)
    issue = IssueContext(number=56, repo='OWNER/REPO', title='Issue', body='Issue', url='https://github.com/OWNER/REPO/issues/56', comments=())
    def invoke():
        return issue_implementation._implement_approved_issue(runner, issue_number=56, approved_plan='Approved plan.',
            config=config, memory=None, issue_context=issue, coder_session_id=None,
            usage_context=validated_agent_module._new_usage_context(config),
            one_shot_parent_issue=56 if one_shot else None)

    if outcome == 'converged':
        assert invoke() == 0
        assert delegated
        handoffs = [body for body in runner.comments if 'AGENT_ISSUE_PR_HANDOFF' in body or 'AGENT_PLAN_ONE_SHOT_IMPL' in body]
        assert len(handoffs) == (2 if one_shot else 1)
        for body in handoffs:
            if 'AGENT_PLAN_ONE_SHOT_IMPL' in body:
                from coding_review_agent_loop.decomposition import _decode_one_shot_impl_handoff_metadata
                encoded = body.split('AGENT_PLAN_ONE_SHOT_IMPL: ', 1)[1].split(' -->', 1)[0]
                assert _decode_one_shot_impl_handoff_metadata(encoded).pr_head_sha == 'b' * 40
            else:
                assert 'b' * 40 in body and 'https://example.test/latest' in body
                assert '6367c48dd193d56ea7b0baad25b19455e529f5ee' not in body
    else:
        with pytest.raises(PrHeadObservationError):
            invoke()
        assert not delegated
        assert not runner.pr_payload.get('comments')
        assert not any('handoff' in body.lower() for body in runner.comments)
    assert not any(cmd[:1] in (['claude'], ['codex']) for cmd, _ in runner.commands)
