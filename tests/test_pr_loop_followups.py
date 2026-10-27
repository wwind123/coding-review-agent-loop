"""PR-loop tests for followups and pr_loop_support: PR follow-up filing (#1542)."""

import dataclasses
import json
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.comment_rendering import (
    _render_public_coder_followup_comment,
    render_risk_test_matrix_section,
)
from coding_review_agent_loop.followups import (
    MAX_APPROVED_FOLLOWUP_ISSUES,
    reconcile_approved_followups,
)
from coding_review_agent_loop.github import (
    HumanReviewRequirement,
    IssueComment,
    IssueContext,
    PullRequestMetadata,
    PullRequestReviewContext,
)
from coding_review_agent_loop.local_test_evidence import (
    LocalTestObservation,
)
from coding_review_agent_loop.unresolved_items import (
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _decode_round_metadata,
)
from coding_review_agent_loop.agent_failure import ValidatedAgentResponse
from coding_review_agent_loop.protocol import (
    ApprovedFollowup,
    ReviewItemDisposition,
    UnresolvedReviewItem,
    parse_risk_test_matrix,
    risk_test_matrix_identity,
    validate_structured_coder_followup,
)
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    prior_item_dispositions,
    structured_coder_followup,
    structured_pr_review,
)
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.round_state as round_state
import coding_review_agent_loop.triage as triage
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_attr_in
from pr_loop_test_helpers import (
    _followup_matrix_context,
    _followup_observation,
)


def test_m780_09_review_only_recovery_row_survives_coder_handoff_and_pr_reresume(tmp_path):
    matrix = parse_risk_test_matrix({
        "applicability": "applicable",
        "rows": [{
            "row_id": "row-post-review-recovery",
            "label": "Post-review recovery reaches the intended orchestration branch",
            "entry_path_or_mode": "ordinary / review-only",
            "initial_state": "review approval recorded, repaired head pending",
            "event": "repaired head passes",
            "expected_outcome": "Re-enter qualification and complete without a watcher",
            "forbidden_side_effects": ["Do not merge until the repaired head is qualified"],
            "proposed_test_level": "orchestrator",
            "proposed_test_location": "tests/test_orchestrator_pr.py::test_post_review_recovery",
            "applicability": "applicable",
            "related_scope_item_ids": ["scope-recovery"],
            "execution_owner": "one-shot",
        }],
        "important_exclusions": ["Helper-only guard tests do not discharge this row."],
    })
    identity = risk_test_matrix_identity(matrix)
    canonical_plan = "Approved recovery plan.\n\n" + render_risk_test_matrix_section(matrix)
    plan_context = execution_policy.make_approved_plan_context(
        canonical_plan,
        source_locator="test approved recovery plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    coder_output = structured_coder_followup(
        addressed_items=["item-1"],
        summary="The implementation was repaired; the orchestration row remains unverified.",
    )
    runner = FakeRunner(
        claude_outputs=[coder_output],
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                blocking_items=["Exercise the post-review recovery orchestration path."],
            ),
            structured_pr_review(
                prior_item_dispositions=[{
                    "item_id": "item-1",
                    "disposition": "resolved",
                }],
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=config,
        approved_plan_context=plan_context,
    ) == 0

    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    reviewer_prompts = [
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]
    ]
    assert "row-post-review-recovery" in coder_prompt
    assert len(reviewer_prompts) == 2
    assert all("row-post-review-recovery" in prompt for prompt in reviewer_prompts)
    assert any("row-post-review-recovery" in comment for comment in runner.comments)
    coder_metadata_comments = [
        item["body"] for item in runner.pr_payload.get("comments", [])
        if isinstance(item, dict)
        and isinstance(item.get("body"), str)
        and "AGENT_LOOP_META: " in item["body"]
        and round_state._decode_round_metadata(
            item["body"].split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
        ).role == "coder"
    ]
    assert coder_metadata_comments
    coder_metadata = round_state._decode_round_metadata(
        coder_metadata_comments[-1].split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
    )
    assert coder_metadata.risk_test_matrix_evidence is not None
    assert coder_metadata.risk_test_matrix_evidence["rows"][0]["row_id"] == (
        "row-post-review-recovery"
    )
    assert "execution_ref" not in coder_metadata_comments[-1]


def _semantic_coder_followup_text(execution_ref: str) -> str:
    raw = structured_coder_followup(
        addressed_items=["item-1"],
        summary="The follow-up was completed and the selected workflow was exercised.",
    )
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "followup-derived-evidence",
        "execution_refs": [execution_ref],
        "test_identifiers": ["tests/test_orchestrator_pr.py::test_followup_workflow"],
        "test_locations": ["tests/test_orchestrator_pr.py"],
        "workflow_path_claim": "The real PR follow-up caller reached post-head derivation.",
        "outcome_assertions": ["The selected managed observation passed."],
        "forbidden_effect_assertions": ["The PR handoff and review continuation were retained."],
        "caveats": [],
    }]
    return json.dumps(payload) + raw[end:]


@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_diagnostic"),
    [
        ("success", "verified", "head-mismatch"),
        ("exhausted", "stale/unverified", "semantic-correction-exhausted"),
        ("head-race", "stale/unverified", "head-changed-during-correction"),
    ],
)
def test_run_pr_loop_derives_followup_evidence_through_real_coder_caller(
    tmp_path, monkeypatch, mode, expected_status, expected_diagnostic
):
    plan_context = _followup_matrix_context()
    wrong = _followup_observation(
        execution_ref="coder-turn:observation-1",
        receipt_id="receipt-wrong-head",
        head="predecessor-head",
        timestamp="2026-01-01T00:00:00Z",
        tracked_digest="tree-predecessor",
    )
    current = _followup_observation(
        execution_ref="coder-turn:observation-2",
        receipt_id="receipt-current-head",
        head="repaired-head",
        timestamp="2026-01-01T00:00:01Z",
    )
    initial_text = _semantic_coder_followup_text(wrong.execution_ref)
    corrected_text = _semantic_coder_followup_text(current.execution_ref)
    parsed = validate_structured_coder_followup(
        initial_text,
        required_architecture_impact_contract=1,
        delivered_risk_test_matrix=plan_context.risk_test_matrix_payload,
        delivered_risk_test_matrix_identity=plan_context.risk_test_matrix_identity,
        required_risk_test_matrix_contract=1,
        delivered_risk_test_matrix_row_ids=("followup-derived-evidence",),
        execution_catalog=(wrong, current),
    )
    assert parsed is not None
    coder_response = ValidatedAgentResponse(
        text=initial_text,
        session_id="coder-session",
        marker_value=parsed,
        acquisition_test_turn_id="coder-turn",
        acquisition_test_observations=(wrong, current),
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="The follow-up path needs workflow coverage.",
                blocking_items=["Exercise follow-up evidence derivation."],
            ),
            structured_pr_review(
                state="approved",
                summary="The follow-up evidence path is covered.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        pr_payload={"headRefOid": "predecessor-head"},
        git_head="predecessor-head",
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)
    real_validated_agent = validated_agent._run_validated_agent

    def fake_validated_agent(*args, **kwargs):
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = "repaired-head"
            runner.simulate_agent_turn(config, config.claude_dir, head="repaired-head")
            return coder_response
        return real_validated_agent(*args, **kwargs)

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", fake_validated_agent)
    monkeypatch.setattr(
        response_validation,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head=runner.pr_payload["headRefOid"],
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, issue_loop,
    ), "_read_assigned_workdir_head",
        lambda *_args, **_kwargs: runner.git_head,
    )

    real_run_agent_result = validated_agent.run_agent_result
    real_get_pr_review_context = pr_loop_support.get_pr_review_context
    race_pending = False

    def get_pr_context(*args, **kwargs):
        nonlocal race_pending
        if mode == "head-race" and race_pending:
            runner.pr_payload["headRefOid"] = "raced-head"
            try:
                return real_get_pr_review_context(*args, **kwargs)
            finally:
                runner.pr_payload["headRefOid"] = "repaired-head"
                race_pending = False
        return real_get_pr_review_context(*args, **kwargs)

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_pr_review_context", get_pr_context)

    def fake_correction(*_args, **kwargs):
        nonlocal race_pending
        if kwargs.get("label") != "semantic-evidence-correction":
            return real_run_agent_result(*_args, **kwargs)
        if mode == "head-race":
            race_pending = True
        return SimpleNamespace(
            text=("not a structured response" if mode == "exhausted" else corrected_text)
        )

    patch_attr_in(monkeypatch, (
        validated_agent, response_validation,
    ), "run_agent_result", fake_correction)

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=config,
        approved_plan_context=plan_context,
    ) == 0

    coder_comments = [
        item["body"] for item in runner.pr_payload["comments"]
        if isinstance(item, dict)
        and isinstance(item.get("body"), str)
        and "AGENT_LOOP_META: " in item["body"]
        and _decode_round_metadata(
            item["body"].split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
        ).role == "coder"
    ]
    assert coder_comments
    metadata = _decode_round_metadata(
        coder_comments[-1].split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
    )
    assert metadata.risk_test_matrix_evidence is not None
    assert metadata.risk_test_matrix_evidence["rows"][0]["status"] == expected_status, (
        metadata.risk_test_matrix_evidence,
        metadata.risk_test_matrix_diagnostics,
    )
    assert any(
        item["code"] == expected_diagnostic
        for item in metadata.risk_test_matrix_diagnostics
    ) if mode != "success" else not any(
        item["code"] == expected_diagnostic
        for item in metadata.risk_test_matrix_diagnostics
    )
    assert metadata.raw_structured_coder_response is not None
    assert "execution_ref" not in metadata.raw_structured_coder_response
    assert "risk_test_matrix_claims" not in metadata.raw_structured_coder_response
    assert any("follow-up" in comment.lower() for comment in runner.comments)
    assert any("The follow-up evidence path is covered." in comment for comment in runner.comments)


def test_run_pr_loop_accepts_followup_with_command_string_refs_as_unverified(
    tmp_path, monkeypatch
):
    """#859 parity: a follow-up citing command strings is not rejected.

    The advanced head is reconciled, the row derives as unverified with an
    unknown-execution-ref diagnostic, exactly one correction runs, and repair
    is never invoked.
    """
    plan_context = _followup_matrix_context()
    current = _followup_observation(
        execution_ref="coder-turn:observation-1",
        receipt_id="receipt-current-head",
        head="repaired-head",
        timestamp="2026-01-01T00:00:01Z",
    )
    command_ref = "python3 -m pytest tests/test_orchestrator_pr.py -q"
    command_text = _semantic_coder_followup_text(command_ref)
    parsed = validate_structured_coder_followup(
        command_text,
        required_architecture_impact_contract=1,
        delivered_risk_test_matrix=plan_context.risk_test_matrix_payload,
        delivered_risk_test_matrix_identity=plan_context.risk_test_matrix_identity,
        required_risk_test_matrix_contract=1,
        delivered_risk_test_matrix_row_ids=("followup-derived-evidence",),
        execution_catalog=(current,),
    )
    assert parsed is not None
    assert parsed.risk_test_matrix_claims.claims[0].execution_refs == ()
    assert parsed.risk_test_matrix_claims.claims[0].dropped_execution_refs == (command_ref,)
    coder_response = ValidatedAgentResponse(
        text=command_text,
        session_id="coder-session",
        marker_value=parsed,
        acquisition_test_turn_id="coder-turn",
        acquisition_test_observations=(current,),
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="The follow-up path needs workflow coverage.",
                blocking_items=["Exercise follow-up evidence derivation."],
            ),
            structured_pr_review(
                state="approved",
                summary="The follow-up evidence path is covered.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        pr_payload={"headRefOid": "predecessor-head"},
        git_head="predecessor-head",
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)
    real_validated_agent = validated_agent._run_validated_agent

    def fake_validated_agent(*args, **kwargs):
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = "repaired-head"
            runner.simulate_agent_turn(config, config.claude_dir, head="repaired-head")
            return coder_response
        return real_validated_agent(*args, **kwargs)

    def forbidden_repair(*_a, **_k):
        raise AssertionError("repair must not run for dropped execution_refs")

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", fake_validated_agent)
    monkeypatch.setattr(validated_agent, "attempt_repair", forbidden_repair)
    monkeypatch.setattr(validated_agent, "execute_repair", forbidden_repair)
    monkeypatch.setattr(
        response_validation,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head=runner.pr_payload["headRefOid"],
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, issue_loop,
    ), "_read_assigned_workdir_head",
        lambda *_args, **_kwargs: runner.git_head,
    )
    real_run_agent_result = validated_agent.run_agent_result
    correction_prompts = []

    def fake_correction(*_args, **kwargs):
        if kwargs.get("label") != "semantic-evidence-correction":
            return real_run_agent_result(*_args, **kwargs)
        correction_prompts.append(kwargs["prompt"])
        return SimpleNamespace(text=command_text)

    patch_attr_in(monkeypatch, (
        validated_agent, response_validation,
    ), "run_agent_result", fake_correction)

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=config,
        approved_plan_context=plan_context,
    ) == 0

    assert len(correction_prompts) == 1
    assert "unknown-execution-ref" in correction_prompts[0]
    coder_comments = [
        item["body"] for item in runner.pr_payload["comments"]
        if isinstance(item, dict)
        and isinstance(item.get("body"), str)
        and "AGENT_LOOP_META: " in item["body"]
        and _decode_round_metadata(
            item["body"].split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
        ).role == "coder"
    ]
    assert coder_comments
    metadata = _decode_round_metadata(
        coder_comments[-1].split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
    )
    row = metadata.risk_test_matrix_evidence["rows"][0]
    assert row["status"] == "stale/unverified"
    assert row["evidence_citations"] == []
    codes = [item["code"] for item in metadata.risk_test_matrix_diagnostics]
    assert "unknown-execution-ref" in codes
    assert "semantic-correction-exhausted" in codes
    assert any("The follow-up evidence path is covered." in comment for comment in runner.comments)


def test_run_pr_loop_replays_derived_followup_evidence_without_ephemeral_selectors(tmp_path):
    plan_context = _followup_matrix_context()
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Exercise follow-up evidence derivation.",
        status="blocking",
        source_status="blocking",
    )
    raw_coder = _semantic_coder_followup_text("coder-turn:observation-2")
    parsed_coder = validate_structured_coder_followup(raw_coder)
    assert parsed_coder is not None
    evidence = {
        "matrix_identity": plan_context.risk_test_matrix_identity,
        "rows": [{
            "row_id": "followup-derived-evidence",
            "status": "verified",
            "test_identifiers": ["tests/test_orchestrator_pr.py::test_followup_workflow"],
            "test_locations": ["tests/test_orchestrator_pr.py"],
            "workflow_path_claim": "The persisted follow-up reached the authenticated head.",
            "outcome_assertions": ["The selected managed observation passed."],
            "forbidden_effect_assertions": ["The PR handoff was retained."],
            "evidence_citations": [{
                "command": "python3 -m pytest tests/test_orchestrator_pr.py -q",
                "receipt_id": "receipt-current-head",
                "claim": "current-result",
            }],
            "caveats": [],
        }],
    }
    coder_public = _render_public_coder_followup_comment(
        parsed_coder,
        agent="Claude",
        prior_items=(carried_item,),
    ) + (
        "\n\n### Risk-test matrix evidence\n"
        "- followup-derived-evidence: verified (receipt-current-head)"
    )
    coder_comment = _attach_round_metadata(
        coder_public,
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="abc123",
            prior_items=(carried_item,),
            raw_structured_coder_response=raw_coder,
            risk_test_matrix_evidence=evidence,
        ),
    )
    persisted_coder_metadata = _decode_round_metadata(
        coder_comment.split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
    )
    assert persisted_coder_metadata.raw_structured_coder_response is not None
    assert "execution_ref" not in persisted_coder_metadata.raw_structured_coder_response
    assert "risk_test_matrix_claims" not in persisted_coder_metadata.raw_structured_coder_response
    review_raw = structured_pr_review(
        state="blocking",
        summary="The persisted evidence needs one more review.",
        prior_item_dispositions=[
            {"item_id": "item-1", "disposition": "blocking", "note": "Review the handoff."}
        ],
    )
    review_comment = _attach_round_metadata(
        review_raw,
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="abc123",
            prior_items=(carried_item,),
            dispositions=(
                ReviewItemDisposition(
                    "item-1", "OpenAI Codex", "blocking", "Review the handoff."
                ),
            ),
            state="blocking",
        ),
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="The persisted evidence is available to the reviewer.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            )
        ],
        pr_payload={
            "headRefOid": "abc123",
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-06-01T00:00:00Z", "body": coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-06-01T00:01:00Z", "body": review_comment},
            ],
        },
    )

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=make_config(tmp_path, reviewer="codex"),
        approved_plan_context=plan_context,
    ) == 0

    reviewer_prompt = next(
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]
    )
    assert '"risk_test_matrix_evidence"' in reviewer_prompt
    assert "receipt-current-head" in reviewer_prompt
    assert "coder-turn:observation-2" not in reviewer_prompt
    replayed_coder_metadata = _decode_round_metadata(
        runner.pr_payload["comments"][0]["body"].split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
    )
    assert replayed_coder_metadata.risk_test_matrix_evidence == evidence
    assert replayed_coder_metadata.raw_structured_coder_response is not None
    assert "coder-turn:observation-2" not in replayed_coder_metadata.raw_structured_coder_response
    assert "execution_ref" not in replayed_coder_metadata.raw_structured_coder_response
    assert any(
        "The persisted evidence is available to the reviewer." in comment
        for comment in runner.comments
    )


def test_pr_loop_ignores_approved_followups_by_default(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n\n### Non-blocking follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [
        "**Review verdict:** Approved\n\n"
        "LGTM.\n\n### Future follow-ups\n- Add cleanup docs.\n"
        "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex: unknown model (medium)"
    ]

def test_pr_loop_summarizes_approved_followups_from_multiple_reviewers(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Non-blocking follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        claude_outputs=[
            "Claude approves.\n\n### Non-blocking follow-ups\n- Add regression coverage.\n"
            "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        approved_followups="summarize",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.comments) == 3
    summary = runner.comments[-1]
    assert summary.startswith("Approved-review future follow-ups for PR #77:")
    assert "- Add cleanup docs. (Codex)" in summary
    assert "- Add regression coverage. (Claude)" in summary
    assert "future work and did not block merge readiness" in summary
    assert summary.endswith("-- coding-review-agent-loop")

def test_pr_loop_creates_issues_for_approved_followups(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Non-blocking follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        claude_outputs=[
            "Claude approves.\n\n### Non-blocking follow-ups\n- Add regression coverage.\n"
            "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        approved_followups="issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.comments) == 3
    # Every follow-up leads with its stable creation-identity record (#510).
    import re as _re_identity

    identity_line = _re_identity.compile(
        r"<!-- AGENT_FOLLOWUP_CREATION_IDENTITY: [0-9a-f]{64} -->\n"
    )
    for issue in runner.issues:
        assert identity_line.match(issue["body"])
    runner.issues = [
        {**issue, "body": identity_line.sub("", issue["body"], count=1)}
        for issue in runner.issues
    ]
    assert runner.issues == [
        {
            "title": "Follow up future review note: Add cleanup docs.",
            "body": (
                "Future follow-up from approved review on PR #77.\n\n"
                "Source context:\n"
                "- Lookup context: repository=OWNER/REPO; source=pr#77; identity=abc123; related PR(s)=#77\n\n"
                "Reviewer: Codex\n\n"
                "Follow-up:\n"
                "- Add cleanup docs.\n\n"
                "Original reviewer notes:\n"
                "- Codex: Add cleanup docs.\n\n"
                "This was mentioned in an approved review as future work and did not block merge readiness."
            ),
        },
        {
            "title": "Follow up future review note: Add regression coverage.",
            "body": (
                "Future follow-up from approved review on PR #77.\n\n"
                "Source context:\n"
                "- Lookup context: repository=OWNER/REPO; source=pr#77; identity=abc123; related PR(s)=#77\n\n"
                "Reviewer: Claude\n\n"
                "Follow-up:\n"
                "- Add regression coverage.\n\n"
                "Original reviewer notes:\n"
                "- Claude: Add regression coverage.\n\n"
                "This was mentioned in an approved review as future work and did not block merge readiness."
            ),
        },
    ]
    issue_summary = runner.comments[-1]
    assert issue_summary.startswith("Created approved-review future follow-up issues for PR #77:")
    assert "- https://github.com/OWNER/REPO/issues/99" in issue_summary
    assert issue_summary.count("https://github.com/OWNER/REPO/issues/99") == 1
    assert "future work and did not block merge readiness" in issue_summary
    assert issue_summary.endswith("-- coding-review-agent-loop")

def test_pr_loop_deduplicates_approved_followup_issues_across_reviewers(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Future follow-ups\n"
            "- **Remote validation**: Validate explicit workdir git remotes against the target repo.\n"
            "- Add a distinct dry-run smoke test.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        claude_outputs=[
            "Claude approves.\n\n### Future follow-ups\n"
            "- **Remote validation**: Validate explicit workdir git remotes against the target repo.\n"
            "- Document cache cleanup behavior.\n"
            "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
        ],
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/99",
            "https://github.com/OWNER/REPO/issues/100",
            "https://github.com/OWNER/REPO/issues/101",
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        approved_followups="issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert [issue["title"] for issue in runner.issues] == [
        "Follow up future review note: **Remote validation**: Validate explicit workdir git remotes against the target repo.",
        "Follow up future review note: Add a distinct dry-run smoke test.",
        "Follow up future review note: Document cache cleanup behavior.",
    ]
    remote_body = runner.issues[0]["body"]
    assert "Reviewers:\n- Codex\n- Claude" in remote_body
    assert "Original reviewer notes:" in remote_body
    assert "- Codex: **Remote validation**" in remote_body
    assert "- Claude: **Remote validation**" in remote_body
    issue_summary = runner.comments[-1]
    assert "- https://github.com/OWNER/REPO/issues/99" in issue_summary
    assert "- https://github.com/OWNER/REPO/issues/100" in issue_summary
    assert "- https://github.com/OWNER/REPO/issues/101" in issue_summary

def test_reconcile_approved_followups_groups_semantic_duplicates_and_preserves_distinct_items():
    reconciliation = reconcile_approved_followups(
        [
            ApprovedFollowup(
                reviewer="Claude",
                text="Clarify repair-pass ownership across the flowchart and sequence diagram.",
            ),
            ApprovedFollowup(
                reviewer="Gemini",
                text="Document repair pass ownership in the flowchart and sequence diagram so the handoff is clear.",
            ),
            ApprovedFollowup(
                reviewer="Codex",
                text="Add memory freshness checks before planning starts.",
            ),
            ApprovedFollowup(
                reviewer="Claude",
                text="Add sync-before-planning coverage for reviewer workdirs.",
            ),
        ],
        issue_limit=MAX_APPROVED_FOLLOWUP_ISSUES,
    )

    assert len(reconciliation.groups) == 3
    assert reconciliation.deduplicated_count == 1
    assert reconciliation.skipped_by_cap == 0
    grouped_reviewers = [group.reviewers for group in reconciliation.groups]
    assert ("Claude", "Gemini") in grouped_reviewers
    assert any("memory freshness" in group.text for group in reconciliation.groups)
    assert any("sync-before-planning" in group.text for group in reconciliation.groups)

def test_reconcile_approved_followups_selects_more_specific_canonical_wording_and_caps():
    reconciliation = reconcile_approved_followups(
        [
            ApprovedFollowup(reviewer="Claude", text="Clarify repair-pass ownership."),
            ApprovedFollowup(
                reviewer="Gemini",
                text="Clarify repair-pass ownership in `docs/local_agent_loop.md` and the sequence diagram.",
            ),
            ApprovedFollowup(reviewer="Codex", text="Follow up two."),
            ApprovedFollowup(reviewer="Claude", text="Follow up three."),
            ApprovedFollowup(reviewer="Gemini", text="Follow up four."),
        ],
        issue_limit=3,
    )

    assert reconciliation.groups[0].text == (
        "Clarify repair-pass ownership in `docs/local_agent_loop.md` and the sequence diagram."
    )
    assert len(reconciliation.selected_groups) == 3
    assert reconciliation.skipped_by_cap == 1
    assert reconciliation.deduplicated_count == 1

def test_pr_loop_files_earlier_future_followup_not_repeated_in_final_round(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Codex approves with a later cleanup.",
                future_followups=["Add memory freshness checks before planning starts."],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="approved",
                summary="Codex final approval.",
                prior_item_dispositions=[
                    {
                        "item_id": "item-1",
                        "disposition": "future",
                        "note": "Still useful as separate tracking.",
                    },
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Need one current-PR fix.",
                blocking_items=["Fix the current sync regression."],
                reviewer="Anthropic Claude",
            ),
            structured_coder_followup(
                addressed_items=["item-2"],
                remaining_items=["item-1"],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(
                state="approved",
                summary="Claude final approval.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "future", "note": "Still valid."},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="Anthropic Claude",
            ),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        approved_followups="issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == (
        "Follow up future review note: Add memory freshness checks before planning starts."
    )
    assert "Update from Codex: Still useful as separate tracking." in runner.issues[0]["body"]


def test_pr_loop_does_not_allocate_new_items_for_repeated_carried_future_followups(tmp_path):
    future_text = "Rate limit the preferences write endpoint to avoid provider validation abuse."
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Codex approves with a later hardening task.",
                future_followups=[future_text],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="approved",
                summary="Codex final approval.",
                future_followups=[
                    "Add a rate limit to PUT /api/preferences because it triggers provider validation."
                ],
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "future", "note": "Still separate work."},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Need one current-PR fix.",
                blocking_items=["Fix the current sync regression."],
                reviewer="Anthropic Claude",
            ),
            structured_coder_followup(
                addressed_items=["item-2"],
                remaining_items=["item-1"],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(
                state="approved",
                summary="Claude final approval.",
                future_followups=[future_text],
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "future", "note": "Still useful."},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="Anthropic Claude",
            ),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        approved_followups="issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    reviewer_metadata = []
    for comment in runner.pr_payload["comments"]:
        match = re.search(
            r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
            comment["body"],
        )
        if match:
            metadata = _decode_round_metadata(match.group("payload"))
            if metadata.role == "reviewer" and metadata.round_number == 2:
                reviewer_metadata.append(metadata)

    assert len(reviewer_metadata) == 2
    assert all(not metadata.new_items for metadata in reviewer_metadata)
    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == f"Follow up future review note: {future_text}"


def test_pr_loop_does_not_file_resolved_earlier_future_followup(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Codex approves with a later cleanup.",
                future_followups=["Remove stale final-round-only follow-up handling."],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="approved",
                summary="Codex final approval.",
                prior_item_dispositions=[
                    {
                        "item_id": "item-1",
                        "disposition": "resolved",
                        "note": "Fixed in the second commit.",
                    },
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Need one current-PR fix.",
                blocking_items=["Fix the current sync regression."],
                reviewer="Anthropic Claude",
            ),
            structured_coder_followup(
                addressed_items=["item-2"],
                remaining_items=["item-1"],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(
                state="approved",
                summary="Claude final approval.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Fixed."},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="Anthropic Claude",
            ),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        approved_followups="issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.issues == []
    assert not any(comment.startswith("Created approved-review future follow-up issues") for comment in runner.comments)

def test_pr_loop_semantically_deduplicates_followup_issues_and_keeps_provenance(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Codex approves.",
                reviewer="OpenAI Codex",
            )
        ],
        claude_outputs=[
            structured_pr_review(
                state="approved",
                summary="Claude approves.",
                future_followups=[
                    "Clarify repair-pass ownership across the flowchart and sequence diagram."
                ],
                reviewer="Anthropic Claude",
            )
        ],
        gemini_outputs=[
            structured_pr_review(
                state="approved",
                summary="Gemini approves.",
                future_followups=[
                    "Document repair pass ownership in the flowchart and sequence diagram so the handoff is clear."
                ],
                reviewer="Google Gemini",
            )
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude", "gemini"),
        approved_followups="issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.issues) == 1
    body = runner.issues[0]["body"]
    assert "Reviewers:\n- Claude\n- Gemini" in body
    assert "Original reviewer notes:" in body
    assert "- Claude: Clarify repair-pass ownership" in body
    assert "- Gemini: Document repair pass ownership" in body
    assert "Reconciliation: 1 filed, 1 deduplicated, 0 skipped by cap." in runner.comments[-1]

def test_pr_loop_suppresses_followup_issue_summary_when_no_urls_returned(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Future follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        issue_urls=[None],
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.comments) == 1
    assert len(runner.issues) == 1

def test_pr_loop_creates_no_issues_without_approved_followups(tmp_path):
    runner = FakeRunner(
        codex_outputs=["Codex approves.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.comments) == 1
    assert runner.issues == []

def test_pr_loop_logs_created_followup_issue_url(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Future follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, approved_followups="issue", quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    captured = capsys.readouterr()
    assert "Created GitHub issue: https://github.com/OWNER/REPO/issues/99" in captured.err


def test_pr_loop_approval_names_the_signed_requirement_reopen_path(tmp_path, capsys):
    # #1020: the approval line is where an operator learns there is no further
    # move, so it must name the signed-requirement path for new instructions.
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Codex approves.",
                reviewer="OpenAI Codex",
            )
        ],
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    out = capsys.readouterr().out
    assert "PR #77 approved by " in out
    assert "line containing exactly `-- Human Reviewer`" in out
    assert "rerun `agent-loop pr 77`" in out
    assert "Unsigned comments are not read as requirements" in out


@pytest.mark.parametrize("mode", ["summarize", "issue"])
def test_pr_loop_treats_same_pr_followups_as_blocking_without_fix_mode(tmp_path, mode):
    runner = FakeRunner(
        codex_outputs=[
            "Codex found cleanup.\n\n"
            "### Same-PR follow-ups\n"
            "- Rename the helper before merge.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, approved_followups=mode, max_rounds=1)

    with pytest.raises(AgentLoopError, match="still reported blocking"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(runner.comments) == 1
    assert not runner.issues
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

@pytest.mark.parametrize("mode", ["summarize", "issue"])
def test_pr_loop_treats_same_pr_prose_followups_as_blocking_without_fix_mode(tmp_path, mode):
    runner = FakeRunner(
        codex_outputs=[
            "Codex found cleanup.\n\n"
            "### Same-PR follow-ups\n"
            "Rename the helper before merge.\n"
            "Keep the behavior unchanged.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, approved_followups=mode, max_rounds=1)

    with pytest.raises(AgentLoopError, match="still reported blocking"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(runner.comments) == 1
    assert not runner.issues
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

def test_pr_loop_caps_approved_followup_issues(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Non-blocking follow-ups\n"
            "- Follow up one.\n"
            "- Follow up two.\n"
            "- Follow up three.\n"
            "- Follow up four.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert [issue["title"] for issue in runner.issues] == [
        "Follow up future review note: Follow up one.",
        "Follow up future review note: Follow up two.",
        "Follow up future review note: Follow up three.",
    ]
    assert len(runner.comments) == 2
    issue_summary = runner.comments[-1]
    assert "- https://github.com/OWNER/REPO/issues/99" in issue_summary
    assert "Skipped 1 additional item(s) to avoid issue noise" in issue_summary
    assert issue_summary.endswith("-- coding-review-agent-loop")

def test_pr_loop_fix_and_summarize_sends_same_pr_followups_to_coder_then_rereviews(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "### Same-PR follow-ups\n"
            "- Rename the helper before merge.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Add broader integration coverage later.\n"
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Renamed helper.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        issue_payload={"title": "Support issue comments", "body": "Original request."},
        issue_comments=[
            {
                "author": {"login": "commenter"},
                "createdAt": "2026-05-17T10:00:00Z",
                "body": "Clarifying issue comment.",
            }
        ],
    )
    config = make_config(tmp_path, approved_followups="fix-and-summarize")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="commenter",
                created_at="2026-05-17T10:00:00Z",
                body="Clarifying issue comment.",
            ),
        ),
    )

    assert run_pr_loop(runner, pr_number=77, config=config, issue_context=issue_context) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert agent_commands == [["codex", "exec"], ["claude", "--print"], ["codex", "exec"]]
    assert len(runner.comments) == 5
    followup_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "requested same-PR follow-ups" in followup_prompt
    assert "remains blocked pending another review round" in followup_prompt
    assert "Rename the helper before merge." in followup_prompt
    assert "[item-1]" in followup_prompt
    assert "Issue context from GitHub" in followup_prompt
    assert "Title:\nSupport issue comments" in followup_prompt
    assert "Clarifying issue comment." in followup_prompt
    assert "small, localized cleanup for the\ncurrent PR" in followup_prompt
    assert "Keep the change narrowly scoped to the listed items" in followup_prompt
    assert "Do not take on\nlarger redesigns or unrelated future work" in followup_prompt
    assert "Add broader integration coverage later." in runner.comments[-1]


def test_same_pr_followup_repair_uses_only_visible_items_not_retained_future_items(tmp_path):
    """Same-PR dispatch must give repair the same item namespace as its prompt."""
    malformed_coder_response = json.dumps(
        {
            "schema_version": 1,
            "kind": "coder_followup",
            "state": "approved",
            "summary": "The visible follow-up is complete.",
            "addressed_items": ["item-2"],
            "remaining_items": [],
            "human_requirements": {
                "addressed_ids": [],
                "checked_discussion_directly": False,
            },
            "human_requirement_dispositions": [],
            # The source carries its own assessment: repair may fix the
            # envelope but may not supply an omitted one (#925).
            "architecture_impact": json.loads(
                structured_coder_followup().split("\n<!--", 1)[0]
            )["architecture_impact"],
        }
    )
    repaired_coder_response = structured_coder_followup(
        state="approved",
        summary="The visible follow-up is complete.",
        addressed_items=["item-2"],
        remaining_items=[],
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                future_followups=["Document the broader behavior in a later PR."],
            ),
            structured_pr_review(
                state="approved",
                summary="The current-PR behavior is fixed.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "future"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
        ],
        gemini_outputs=[
            structured_pr_review(
                state="blocking",
                same_pr_followups=["Fix the current-PR behavior."],
                reviewer="Google Gemini",
            ),
            structured_pr_review(
                state="approved",
                summary="The current-PR behavior is fixed.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "future"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="Google Gemini",
            ),
        ],
        claude_outputs=[
            malformed_coder_response,
        ],
    )
    config = make_config(
        tmp_path,
        approved_followups="fix-and-summarize",
        agent_max_retries=0,
        reviewer=("codex", "gemini"),
    )
    repair_calls = []

    def fake_attempt_repair(raw, gemini_cmd, *, expected_kind=None, **kwargs):
        repair_calls.append((expected_kind, kwargs.get("unresolved_item_ids")))
        return repaired_coder_response

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_attempt_repair):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert repair_calls == [("coder_followup", ("item-2",))]
    followup_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "Fix the current-PR behavior." in followup_prompt
    assert "Document the broader behavior in a later PR." not in followup_prompt


def test_blocking_same_pr_followup_reaches_coder_even_when_approved_followups_ignored(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "### Same-PR follow-ups\n- Rename the helper before merge.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves.\n" + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Renamed helper.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="ignore")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    followup_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "Rename the helper before merge." in followup_prompt
    assert "does not enable a same-pr fix path" not in followup_prompt

def test_pr_loop_fix_and_issue_uses_final_round_future_followups_after_same_pr_cleanup(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "### Same-PR follow-ups\n"
            "- Tighten the validation message.\n\n"
            "### Future follow-ups\n"
            "- Stale future item from the blocking round.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Add a separate migration dry-run command.\n"
            + prior_item_dispositions("[item-1] resolved", "[item-2] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Tightened message.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "Follow up future review note: Add a separate migration dry-run command."
    assert "Stale future item from the blocking round." not in runner.issues[0]["body"]
    commands = [cmd[:3] for cmd, _cwd in runner.commands]
    assert commands.count(["gh", "issue", "create"]) == 1

def test_pr_loop_fix_and_issue_files_carried_blocking_round_future_followups(tmp_path):
    # Issue #1389: a future follow-up raised on a blocking round is no longer
    # dropped; it is filed at approval like an approved-review future item.
    runner = FakeRunner(
        codex_outputs=[
            "### Same-PR follow-ups\n"
            "- Tighten the validation message.\n\n"
            "### Future follow-ups\n"
            "- Off-scope item from the blocking round.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Add a separate migration dry-run command.\n"
            + prior_item_dispositions("[item-1] resolved", "[item-2] future follow-up: still separate")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Tightened message.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert sorted(issue["title"] for issue in runner.issues) == [
        "Follow up future review note: Add a separate migration dry-run command.",
        "Follow up future review note: Off-scope item from the blocking round.",
    ]
    coder_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert coder_prompts and all("Off-scope item from the blocking round." not in p for p in coder_prompts)

def test_pr_loop_fix_and_issue_uses_only_final_round_future_followups(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex found cleanup.\n\n"
            "### Same-PR follow-ups\n"
            "- Tighten the validation message.\n\n"
            "### Future follow-ups\n"
            "- Stale item fixed by the same-PR pass.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Add a separate migration dry-run command.\n"
            + prior_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Tightened message.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "Follow up future review note: Add a separate migration dry-run command."
    assert "Stale item fixed by the same-PR pass." not in runner.issues[0]["body"]
    assert "- https://github.com/OWNER/REPO/issues/99" in runner.comments[-1]
    commands = [cmd[:3] for cmd, _cwd in runner.commands]
    assert commands.count(["gh", "issue", "create"]) == 1

def test_pr_loop_fix_and_summarize_uses_only_final_round_future_followups(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex found cleanup.\n\n"
            "### Same-PR follow-ups\n"
            "- Add a small assertion before merge.\n\n"
            "### Future follow-ups\n"
            "- Add Codex's larger follow-up later.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Add Codex's final follow-up later.\n"
            + prior_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
                "[item-3] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Claude approves.\n\n"
            "### Future follow-ups\n"
            "- Add Claude's larger follow-up later.\n"
            "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
            "Claude approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Add Claude's final follow-up later.\n"
            + prior_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
                "[item-3] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        gemini_outputs=["Added assertion.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini"],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        approved_followups="fix-and-summarize",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"], ["gemini"])]
    assert agent_commands == [
        ["codex", "exec"],
        ["claude", "--print"],
        ["gemini", "--prompt"],
        ["codex", "exec"],
        ["claude", "--print"],
    ]
    summary = runner.comments[-1]
    assert "- Add Codex's final follow-up later. (Codex)" in summary
    assert "- Add Claude's final follow-up later. (Claude)" in summary
    assert "Add Codex's larger follow-up later." not in summary
    assert "Add Claude's larger follow-up later." not in summary

def test_pr_loop_fix_and_issue_extracts_final_round_bullet_and_prose_future_followups(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex found cleanup.\n\n"
            "### Same-PR follow-ups\n"
            "- Tighten the validation message.\n\n"
            "### Future follow-ups\n"
            "- Stale Codex item fixed by the same-PR pass.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass.\n\n"
            "### Future follow-ups\n"
            "- Refine token estimation for large review prompts.\n"
            + prior_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
                "[item-3] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Claude approves with cleanup.\n\n"
            "### Future follow-ups\n"
            "- Stale Claude item fixed by the same-PR pass.\n"
            "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
            "Claude approves final pass.\n\n"
            "### Future follow-ups\n"
            "The `_parse_gemini_output` helper is dead production code and could be removed\n"
            "in a future cleanup.\n\n"
            "### Same-PR follow-ups\n"
            "No same-PR follow-ups.\n"
            + prior_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
                "[item-3] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        gemini_outputs=["Tightened message.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini"],
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/99",
            "https://github.com/OWNER/REPO/issues/100",
        ],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        approved_followups="fix-and-issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.issues[0]["title"] == (
        "Follow up future review note: Refine token estimation for large review prompts."
    )
    assert runner.issues[1]["title"].startswith(
        "Follow up future review note: The `_parse_gemini_output` helper is dead production code"
    )
    assert "could be removed in a future cleanup." in runner.issues[1]["body"]
    assert "Stale Codex item fixed by the same-PR pass." not in runner.issues[0]["body"]
    assert "Stale Claude item fixed by the same-PR pass." not in runner.issues[1]["body"]
    issue_summary = runner.comments[-1]
    assert "- https://github.com/OWNER/REPO/issues/99" in issue_summary
    assert "- https://github.com/OWNER/REPO/issues/100" in issue_summary
    assert "Stale Codex item fixed by the same-PR pass." not in issue_summary


def test_pr_loop_posts_followup_with_env_prefixed_managed_tests(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Needs a test.",
                blocking_items=["Add a regression test."],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(
                state="approved",
                summary="The regression test resolves the finding.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                reviewer="Anthropic Claude",
            ),
        ],
        codex_outputs=[
            structured_coder_followup(
                summary="Added the managed regression test.",
                addressed_items=["item-1"],
                tests_run=[
                    "DATABASE_URL=postgresql+asyncpg://localhost/example_test "
                    "/outside/bin/agent-loop run-tests --timeout-seconds 120 "
                    "--memory-dir /outside/cache -- "
                    ".venv/bin/python -m pytest tests/test_pat_auth.py -q"
                ],
                reviewer="OpenAI Codex",
            ),
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    followup = next(
        comment
        for comment in runner.comments
        if "Added the managed regression test." in comment
    )
    assert "DATABASE_URL=postgresql+asyncpg://localhost/example_test" in followup
    assert ".venv/bin/python -m pytest tests/test_pat_auth.py -q" in followup
    assert "agent-loop instrumented; whole-command timeout 120s" in followup
    assert "/outside/bin/agent-loop" not in followup
    assert "/outside/cache" not in followup
    assert any(
        "The regression test resolves the finding." in comment
        for comment in runner.comments
    )


def test_pr_loop_records_structured_followup_out_of_checkout_tests_as_context(tmp_path):
    """Issue #991: a clean-base baseline is context, not a rejected hand-off."""
    baseline = (
        "PYTHONPATH=/tmp/scratch-main /usr/bin/python3 -m pytest "
        "/tmp/scratch-main/tests/ -q"
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Needs a test.",
                blocking_items=["Add a regression test."],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(
                state="approved",
                summary="The regression test resolves the finding.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                reviewer="Anthropic Claude",
            ),
        ],
        codex_outputs=[
            structured_coder_followup(
                summary="Added the test.",
                addressed_items=["item-1"],
                tests_run=["python -m pytest tests/test_foo.py -q", baseline],
                reviewer="OpenAI Codex",
            ),
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")
    # The same baseline also ran (and failed) through the managed broker.
    runner._local_test_observations.append(LocalTestObservation(
        command=("python3", "-m", "pytest", "/tmp/scratch-main/tests/", "-q"),
        outcome="failed",
        provenance="parent-observed",
        receipt_id="baseline-broker-failure",
        turn_id="turn-baseline",
        timestamp="2026-09-23T10:00:00+00:00",
        cwd=str(tmp_path),
    ))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup = next(comment for comment in runner.comments if "Added the test." in comment)
    tests_section, _, context_section = followup.partition(
        "### Out-of-checkout context runs (not evidence)"
    )
    assert "python -m pytest tests/test_foo.py -q" in tests_section
    assert "/tmp/scratch-main" not in tests_section
    assert f"- {baseline}" in context_section
    baseline_line = next(
        line for line in followup.splitlines() if "baseline-broker-failure" in line
    )
    assert "out-of-checkout context (not evidence) `failed`" in baseline_line
    assert "authoritative" not in baseline_line


def test_pr_loop_rejects_structured_followup_live_target_tests_before_posting(tmp_path):
    # Regression for #584: a structured `tests_run` entry (origin='structured',
    # orchestrator.py line ~6197) must apply STRICT-COMMAND URL classification
    # -- any URL in the clause is a live-target rejection, matching the same
    # rule freeform `Tests:` prose gets.
    runner = FakeRunner(
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Needs a test.",
                blocking_items=["Add a regression test."],
                reviewer="Anthropic Claude",
            ),
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_coder_followup(
                summary="Added the test.",
                addressed_items=["item-1"],
                tests_run=["pytest tests/test_foo.py https://live.example"],
                reviewer="OpenAI Codex",
            ),
        ] * 2,
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="live remote target"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(runner.comments) == 3
    assert runner.comments[0].startswith("**Review verdict:** Blocking")
    assert not any("Added the test." in comment for comment in runner.comments)


@pytest.mark.parametrize("command", [
    "python -m pytest tests/test_foo.py /tmp/scratch-main/tests/test_foo.py",
    "pytest -c /tmp/scratch-main/pytest.ini tests/test_foo.py",
])
def test_pr_loop_refuses_structured_followup_mixed_outside_tests(tmp_path, command):
    """Issue #991: a run that may also test the checkout is never context."""
    runner = FakeRunner(
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Needs a test.",
                blocking_items=["Add a regression test."],
                reviewer="Anthropic Claude",
            ),
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_coder_followup(
                summary="Added the test.",
                addressed_items=["item-1"],
                tests_run=[command],
                reviewer="OpenAI Codex",
            ),
        ] * 2,
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="as separate commands"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any("Added the test." in comment for comment in runner.comments)
    assert not any("Out-of-checkout context runs" in comment for comment in runner.comments)


# --- Future follow-ups on blocking reviews (#1389) ---------------------------


def _pr_meta_records_1389(runner):
    records = []
    for comment in runner.pr_payload["comments"]:
        match = re.search(
            r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", comment["body"]
        )
        if match:
            records.append(_decode_round_metadata(match.group("payload")))
    return records


def _coder_prompts_1389(runner):
    return [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def _advance_head_after_coder_1389(monkeypatch, runner, *, interrupt=None):
    """Advance the PR head after each coder turn; optionally interrupt a call."""
    original = validated_agent._run_validated_agent
    counter = iter(range(1, 20))
    calls = {"coder": 0, "reviewer": 0}

    class Interrupted(Exception):
        pass

    def run(*args, **kwargs):
        role = kwargs.get("role")
        if role in calls:
            calls[role] += 1
            if interrupt is not None and interrupt == (role, calls[role]):
                raise Interrupted
        response = original(*args, **kwargs)
        if role == "coder":
            runner.pr_payload["headRefOid"] = f"repaired-{next(counter)}"
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)
    return Interrupted


def test_blocking_pr_review_future_followup_is_recorded_and_filed_at_approval(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="One in-scope defect.",
                blocking_items=["Fix the off-by-one in src/a.py:10."],
                future_followups=["Add cross-process locking for the cache directory."],
            ),
            structured_pr_review(
                state="approved",
                summary="Fixed.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    _advance_head_after_coder_1389(monkeypatch, runner)
    config = make_config(
        tmp_path, approved_followups="fix-and-issue", pr_review_context_mode="compact", max_rounds=3
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    round_1 = [
        record for record in _pr_meta_records_1389(runner)
        if record.role == "reviewer" and record.round_number == 1
    ]
    assert [(item.item_id, item.status) for item in round_1[0].new_items] == [
        ("item-1", "blocking"), ("item-2", "future"),
    ]
    assert [issue["title"] for issue in runner.issues] == [
        "Follow up future review note: Add cross-process locking for the cache directory."
    ]
    coder_prompts = _coder_prompts_1389(runner)
    assert len(coder_prompts) == 1
    assert "Fix the off-by-one in src/a.py:10." in coder_prompts[0]
    assert "item [item-2]" not in coder_prompts[0]
    assert "cross-process locking" not in coder_prompts[0]


def test_blocking_pr_review_future_followup_is_not_recorded_under_ignore(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="One in-scope defect.",
                blocking_items=["Fix the off-by-one in src/a.py:10."],
                future_followups=["Add cross-process locking for the cache directory."],
            ),
            structured_pr_review(
                state="approved",
                summary="Fixed.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    _advance_head_after_coder_1389(monkeypatch, runner)
    config = make_config(tmp_path, approved_followups="ignore", max_rounds=3)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert all(
        item.status != "future"
        for record in _pr_meta_records_1389(runner)
        for item in (*record.new_items, *record.prior_items)
    )
    assert runner.issues == []
    assert not any("future follow-ups for PR #77" in comment for comment in runner.comments)
    assert all("item [item-2]" not in prompt for prompt in _coder_prompts_1389(runner))


def test_compact_carried_blocker_downgraded_on_blocking_round_is_filed_once(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Two defects.",
                blocking_items=[
                    "Harden signal handling for the worker pool.",
                    "Fix the parser regression in src/b.py:20.",
                ],
            ),
            structured_pr_review(
                state="blocking",
                summary="Parser still broken; signal handling is off-scope.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "future",
                     "note": "Outside criteria (i)-(iii) for this issue."},
                    {"item_id": "item-2", "disposition": "blocking",
                     "note": "src/b.py:20 still drops the last token; add a regression test."},
                ],
            ),
            structured_pr_review(
                state="approved",
                summary="Parser fixed.",
                prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1", "item-2"]),
            structured_coder_followup(addressed_items=["item-2"]),
        ],
    )
    _advance_head_after_coder_1389(monkeypatch, runner)
    config = make_config(
        tmp_path, approved_followups="fix-and-issue", pr_review_context_mode="compact", max_rounds=4
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert [issue["title"] for issue in runner.issues] == [
        "Follow up future review note: Harden signal handling for the worker pool."
    ]
    coder_prompts = _coder_prompts_1389(runner)
    assert len(coder_prompts) == 2
    assert "unresolved blocking item [item-1]" in coder_prompts[0]
    # Round 2: item-1 left the ledger as future; it is not coder work.
    assert "item [item-1]" not in coder_prompts[1]
    assert "- Harden signal handling for the worker pool.\n" not in coder_prompts[1]
    assert "unresolved blocking item [item-2]" in coder_prompts[1]


def test_resume_after_blocking_round_future_followup_publishes_it_once(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Two defects plus later work.",
                blocking_items=[
                    "Harden signal handling for the worker pool.",
                    "Fix the off-by-one in src/a.py:10.",
                ],
                future_followups=["Add crash recovery for interrupted uploads."],
            ),
            structured_pr_review(
                state="blocking",
                summary="Off-by-one remains; signal handling is off-scope.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "future",
                     "note": "Outside criteria (i)-(iii) for this issue."},
                    {"item_id": "item-2", "disposition": "blocking",
                     "note": "src/a.py:10 still skips the last row; add a regression test."},
                ],
            ),
            structured_pr_review(
                state="approved",
                summary="Fixed.",
                prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1", "item-2"]),
            structured_coder_followup(addressed_items=["item-2"]),
        ],
    )
    # Stop after the round-2 coder advanced the head, before the round-3
    # review is recorded: item-1 has left the compact ledger as future and
    # item-3 is a new future item from a blocking round.
    interrupted = _advance_head_after_coder_1389(monkeypatch, runner, interrupt=("reviewer", 3))
    config = make_config(
        tmp_path, approved_followups="fix-and-issue", pr_review_context_mode="compact", max_rounds=4
    )

    with pytest.raises(interrupted):
        run_pr_loop(runner, pr_number=77, config=config)
    assert runner.issues == []
    assert runner.pr_payload["headRefOid"] == "repaired-2"

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert sorted(issue["title"] for issue in runner.issues) == [
        "Follow up future review note: Add crash recovery for interrupted uploads.",
        "Follow up future review note: Harden signal handling for the worker pool.",
    ]
    commands = [cmd[:3] for cmd, _cwd in runner.commands]
    assert commands.count(["gh", "issue", "create"]) == 2
    coder_prompts = _coder_prompts_1389(runner)
    assert all("item [item-3]" not in prompt for prompt in coder_prompts)
    assert all("- Add crash recovery for interrupted uploads.\n" not in prompt for prompt in coder_prompts)
    assert "item [item-1]" not in coder_prompts[-1]


def test_resume_after_future_item_repromotion_never_publishes_it(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="One defect plus later work.",
                blocking_items=["Fix the off-by-one in src/a.py:10."],
                future_followups=["Validate the config path before use."],
            ),
            structured_pr_review(
                state="blocking",
                summary="The config path is in scope after all.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "blocking",
                     "note": "Criterion (ii): src/c.py:5 now reads an unvalidated path; validate it."},
                ],
            ),
            structured_pr_review(
                state="approved",
                summary="All fixed.",
                prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"]),
            structured_coder_followup(addressed_items=["item-2"]),
        ],
    )
    # Stop right after the re-promoting round-2 review, before its coder turn.
    interrupted = _advance_head_after_coder_1389(monkeypatch, runner, interrupt=("coder", 2))
    config = make_config(
        tmp_path, approved_followups="fix-and-issue", pr_review_context_mode="compact", max_rounds=4
    )

    with pytest.raises(interrupted):
        run_pr_loop(runner, pr_number=77, config=config)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.issues == []
    assert not any("future follow-ups for PR #77" in comment for comment in runner.comments)
    coder_prompts = _coder_prompts_1389(runner)
    assert "Validate the config path before use." not in coder_prompts[0]
    assert "Validate the config path before use." in coder_prompts[-1]


def test_merge_conflict_handoff_omits_retained_future_followup_and_still_files_it(
    tmp_path, monkeypatch
):
    """Retained future work is never coder work on the merge-conflict handoff (#1389)."""
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="One in-scope defect plus later work.",
                blocking_items=["Fix the off-by-one in src/a.py:10."],
                future_followups=["Add cross-process locking for the cache directory."],
            ),
            structured_pr_review(
                state="approved",
                summary="Fixed and rebased.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        # Round 1 is an ordinary fix; round 2 starts with a merge conflict, so the
        # conflict handoff carries item-1 and the retained future item-2.  The
        # coder omits item-2 entirely and the response must validate.
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"]),
            structured_coder_followup(addressed_items=["item-1"]),
        ],
        mergeability_payloads=[
            {"mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN",
             "headRefOid": "abc123", "baseRefName": "main"},
            {"mergeable": "CONFLICTING", "mergeStateStatus": "DIRTY",
             "headRefOid": "abc123", "baseRefName": "main"},
        ],
    )
    _advance_head_after_coder_1389(monkeypatch, runner)
    config = make_config(
        tmp_path, approved_followups="fix-and-issue", pr_review_context_mode="compact", max_rounds=4
    )
    runner.pr_payload.update({"mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN"})

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    coder_prompts = _coder_prompts_1389(runner)
    assert len(coder_prompts) == 2
    conflict_prompt = coder_prompts[1]
    assert "merge conflict with its base branch `main`" in conflict_prompt
    assert "[item-1]" in conflict_prompt
    assert "[item-2]" not in conflict_prompt
    assert "- Add cross-process locking for the cache directory.\n" not in conflict_prompt
    assert [issue["title"] for issue in runner.issues] == [
        "Follow up future review note: Add cross-process locking for the cache directory."
    ]


@pytest.mark.parametrize(
    "dispatch",
    ["general", "same-pr", "merge-conflict"],
)
def test_every_coder_handoff_excludes_retained_future_items(dispatch):
    """One rule for every coder dispatch: future items are optional, never required."""
    from coding_review_agent_loop.unresolved_items import (
        MERGE_CONFLICT_ITEM_ID,
        _validate_structured_coder_followup_items,
        format_coder_followup_context,
        select_coder_followup_items,
    )
    from coding_review_agent_loop.protocol import validate_structured_coder_followup

    status = "same-pr" if dispatch == "same-pr" else "blocking"
    work = UnresolvedReviewItem(
        item_id="item-1", reviewer="Codex", source_round=1, text="Required fix.", status=status,
    )
    future = UnresolvedReviewItem(
        item_id="item-2", reviewer="Codex", source_round=1, text="Later work.", status="future",
    )
    ledger = [work, future]
    if dispatch == "merge-conflict":
        ledger = [item for item in ledger if item.item_id != MERGE_CONFLICT_ITEM_ID]
    selected = select_coder_followup_items(ledger)
    assert [item.item_id for item in selected] == ["item-1"]
    assert "[item-2]" not in format_coder_followup_context(ledger)

    def parsed(addressed):
        return validate_structured_coder_followup(
            structured_coder_followup(addressed_items=addressed),
            architecture_status_mode="degradable",
        )

    _validate_structured_coder_followup_items(parsed(["item-1"]), unresolved_items=ledger)
    _validate_structured_coder_followup_items(
        parsed(["item-1", "item-2"]), unresolved_items=ledger, optional_item_ids=("item-2",)
    )


def _run_1388_acknowledgement_round(tmp_path, monkeypatch, *, recorded_history):
    """Drive a PR loop to an acknowledgement-only coder round (#1388).

    item-1 is resolved in round 2 and item-2 in round 3, so the round-3 coder
    dispatch carries only the acknowledgement obligation.  The coder still
    lists the already-resolved item-1 as addressed.  ``recorded_history``
    controls whether the loop sees the posted round records.
    """
    import coding_review_agent_loop.github as github_module
    import coding_review_agent_loop.validated_agent as validated_agent_module

    ack_item_id = HUMAN_REQUIREMENTS_ACK_ITEM_ID
    first = HumanReviewRequirement(
        source_type="PR comment", author="maintainer", created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="Keep the audit trail.",
    )
    second = HumanReviewRequirement(
        source_type="PR comment", author="maintainer", created_at="2026-05-18T11:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-2",
        body="Also log every audit write.",
    )
    ack_markdown = (
        "Implemented the fix.\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
        "### Human requirements\n- Requirement 1: kept the audit trail.\n"
        "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    note = (
        "src/audit.py still lacks the acknowledged logging requirement; acknowledge "
        "requirement 2 and add tests/test_audit.py coverage for the audit log."
    )
    resolved_claim = structured_coder_followup(
        state="approved",
        summary="Acknowledged the new signed requirement.",
        addressed_items=["item-1"],
        addressed_item_notes={"item-1": "Resolved by the human decision."},
        human_requirement_ids=[first.requirement_id, second.requirement_id],
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Fix.", blocking_items=["Fix the bug."],
                human_requirements_resolved=True,
            ),
            structured_pr_review(
                state="approved", summary="Fixed.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
            structured_pr_review(
                state="blocking", summary="The new requirement is unacknowledged.",
                prior_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved"},
                    {"item_id": ack_item_id, "disposition": "blocking", "note": note},
                ],
            ),
            structured_pr_review(
                state="approved", summary="Acknowledged.", human_requirements_resolved=True,
            ),
        ],
        claude_outputs=[ack_markdown, ack_markdown, resolved_claim],
    )
    metadata = PullRequestMetadata(
        number=77, repo="OWNER/REPO", title="Acknowledgement", head_branch="feature/x",
        base_branch="main", head_sha="abc123", url="https://github.com/OWNER/REPO/pull/77",
    )
    fetches = []
    real_context = github_module.get_pr_review_context

    def review_context(*args, **kwargs):
        fetches.append(True)
        requirements = (first,) if len(fetches) <= 2 else (first, second)
        comments = real_context(*args, **kwargs).comments if recorded_history else ()
        return PullRequestReviewContext(
            metadata=dataclasses.replace(metadata, head_sha=runner.pr_payload["headRefOid"]),
            comments=comments, human_requirements=requirements,
        )

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_pr_review_context", review_context)
    repairs = []
    real_repair = validated_agent_module._run_structured_repair

    def spy_repair(raw, **kwargs):
        repairs.append(raw)
        return real_repair(raw, **kwargs)

    monkeypatch.setattr(validated_agent_module, "_run_structured_repair", spy_repair)
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=4, quiet=False)
    return runner, config, repairs, resolved_claim


def test_1388_coder_listing_a_resolved_item_as_addressed_is_normalized(
    tmp_path, monkeypatch, capsys,
):
    runner, config, repairs, resolved_claim = _run_1388_acknowledgement_round(
        tmp_path, monkeypatch, recorded_history=True
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert resolved_claim not in repairs
    err = capsys.readouterr().err
    assert (
        "Round 3: Claude listed already-resolved item(s) item-1 in addressed_items; "
        "dropped them from the follow-up"
    ) in err
    coder_comment = next(
        comment for comment in runner.comments
        if "Acknowledged the new signed requirement." in comment
    )
    addressed_section = coder_comment.split("### Addressed items", 1)[1].split("###", 1)[0]
    assert "item-1" not in addressed_section
    assert "- None." in addressed_section


def test_1388_addressed_item_without_resolution_history_still_fails_closed(
    tmp_path, monkeypatch, capsys,
):
    runner, config, repairs, resolved_claim = _run_1388_acknowledgement_round(
        tmp_path, monkeypatch, recorded_history=False
    )

    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)

    assert resolved_claim in repairs
    assert "listed already-resolved item(s)" not in capsys.readouterr().err
    assert not any(
        "Acknowledged the new signed requirement." in comment for comment in runner.comments
    )


@pytest.mark.parametrize('mode', ['display', 'partition', 'citation', 'normalized', 'escape'])
def test_scratch_report_followup_workflow(tmp_path, monkeypatch, mode):
    """#1738: drive acquisition, refusal/correction, publication and review."""
    import coding_review_agent_loop.workdir_guard as guard
    root = tmp_path / 'coding-review-agent-loop'
    monkeypatch.setattr(guard, 'scratch_root', lambda: root)
    audit = (
        'python scripts/dependency_audit.py evaluate --policy .github/dependency-audit-policy.json '
        f'--python-report {root}/scratch/1837/pip-audit.json '
        f'--npm-report {root}/scratch/1837/npm-audit.json'
    )
    bad_command = audit
    if mode == 'partition':
        bad_command = 'pytest tests/test_api.py /outside/test_api.py'
    if mode == 'escape':
        bad_command = audit + f' {root}/scratch/1837/pip-audit.json'
    raw = structured_coder_followup(
        summary='Rejected acquisition claim.', addressed_items=['item-1'],
        tests_run=[bad_command], reviewer='OpenAI Codex',
    )
    if mode == 'citation':
        payload, end = json.JSONDecoder().raw_decode(raw)
        payload['test_observations'] = [{
            'command': audit, 'receipt_id': 'scratch-audit', 'claim': 'current-result',
        }]
        raw = json.dumps(payload) + raw[end:]
    if mode == 'normalized':
        raw = raw.replace(audit, 'pytest tests/test_api.py /outside/test_api.py') + '\ntrailing prose'
    corrected = structured_coder_followup(
        summary='Accepted replacement claim.', addressed_items=['item-1'],
        tests_run=[audit], reviewer='OpenAI Codex',
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_pr_review(state='blocking', blocking_items=['Add coverage.'], reviewer='Anthropic Claude'),
            structured_pr_review(state='approved', prior_item_dispositions=[
                {'item_id': 'item-1', 'disposition': 'resolved'}
            ], reviewer='Anthropic Claude'),
        ],
        codex_outputs=[corrected] if mode == 'display' else [raw, corrected],
    )
    config = make_config(tmp_path, coder='codex', reviewer='claude', agent_max_retries=0)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    prompts = [cmd[-1] for cmd, _ in runner.commands if cmd[:1] == ['codex']]
    assert len(prompts) == (1 if mode == 'display' else 2)
    if mode != 'display':
        assert 'Previous response not accepted: test evidence' in prompts[1]
        assert 'Add coverage.' in prompts[1]
    if mode == 'citation':
        assert 'without hiding failing observations' in prompts[1]
    assert not any('Rejected acquisition claim.' in comment for comment in runner.comments)
    published = next(comment for comment in runner.comments if 'Accepted replacement claim.' in comment)
    assert audit in published
    assert 'Out-of-checkout context runs' not in published
    assert 'scratch-audit' not in published


def test_resumed_scratch_report_classification(tmp_path, monkeypatch):
    import coding_review_agent_loop.workdir_guard as guard
    root = tmp_path / 'coding-review-agent-loop'
    monkeypatch.setattr(guard, 'scratch_root', lambda: root)
    audit = f'python scripts/dependency_audit.py evaluate --input={root}/scratch/1837/audit.json'
    baseline = 'python -m pytest /outside/tests/'
    raw = structured_coder_followup(tests_run=[audit, baseline])
    metadata = PostedRoundMetadata(
        flow='pr', role='coder', agent='Codex', round_number=2,
        subject='abc123', raw_structured_coder_response=raw,
    )
    context = pr_loop_support._coder_followup_review_context(
        raw, metadata, head_sha='abc123', assigned_workdir=tmp_path / 'checkout',
    )
    assert json.dumps(audit) in context
    payload, _ = json.JSONDecoder().raw_decode(context[context.index("{"):])
    assert payload['tests_run'] == [audit]
    assert payload['out_of_checkout_context_runs_not_evidence'] == [baseline]
    assert payload['local_test_evidence'] is None
    assert metadata.raw_structured_coder_response == raw
