import base64
import dataclasses
import datetime
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.ci_health import CiInfrastructureStall, StalledCheck
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop, run_pr_loop
from coding_review_agent_loop.comment_rendering import (
    _render_public_coder_followup_comment,
    _render_public_pr_review_comment,
    render_risk_test_matrix_section,
)
from coding_review_agent_loop.errors import HumanDecisionRequiredError, QuotaResetExceededError
from coding_review_agent_loop.followups import (
    MAX_APPROVED_FOLLOWUP_ISSUES,
    PlanApprovedFollowupSource,
    _followup_issue_body,
    _format_approved_followup_summary,
    _format_plan_approval_summary_with_followups,
    _plan_followup_issue_body,
    reconcile_approved_followups,
    reconcile_plan_approved_followups,
)
from coding_review_agent_loop.github import (
    CiWatchOutcome,
    HumanReviewRequirement,
    IssueComment,
    IssueContext,
    PullRequestCheck,
    PullRequestChecks,
    PullRequestMetadata,
    PullRequestMergeability,
    PullRequestReviewContext,
    get_pr_checks,
)
from coding_review_agent_loop.issue_pr_handoff import (
    find_latest_issue_pr_handoff,
    format_issue_pr_handoff_comment,
)
from coding_review_agent_loop.memory import AgentMemoryContext
import coding_review_agent_loop.prompts as prompts_module
from coding_review_agent_loop.migrations import MigrationValidationResult
from coding_review_agent_loop.local_test_evidence import (
    EnvironmentIdentity,
    LocalTestObservation,
    TreeAttribution,
)
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
    ManagedCiOutcome,
    OrdinaryRecoveryCapability,
)
from coding_review_agent_loop.orchestrator import (
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    QualificationCheckpoint,
    PostedRoundMetadata,
    ValidatedAgentResponse,
    _attach_round_metadata,
    _decode_round_metadata,
    _reconcile_human_requirements_ack_item,
    _resume_pr_round,
    _strip_round_metadata,
    _ensure_finalization_ready,
    _latest_pr_architecture_observation,
    _latest_pr_approved_reviews_for_head,
    _preserve_issue_created_managed_suppression,
)
from coding_review_agent_loop.prompts import (
    COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
    build_followup_prompt,
)
import coding_review_agent_loop.test_runtime as runtime
from coding_review_agent_loop.review_scheduling import TransitionClassification
from coding_review_agent_loop.protocol import (
    ApprovedFollowup,
    ReviewItemDisposition,
    UnresolvedReviewItem,
    parse_plan_review,
    parse_pr_review,
    parse_review,
    parse_unresolved_item_dispositions,
    parse_risk_test_matrix,
    risk_test_matrix_identity,
    validate_structured_issue_implementation,
    validate_structured_coder_followup,
)
from agent_loop_helpers import (
    FakeRunner,
    blocking_issues,
    command_index,
    make_config,
    prior_item_dispositions,
    structured_coder_followup,
    structured_issue_implementation,
    structured_plan_review,
    structured_pr_review,
)


@pytest.mark.parametrize("origin", ["issue-created", "source-managed"])
def test_interrupted_tool_created_managed_pr_retains_suppression(origin):
    contract = ManagedCiContract(origin=origin, issue_created_pr=origin == "issue-created")

    assert _preserve_issue_created_managed_suppression(
        contract,
        active_exception=AgentLoopError("malformed coder handoff"),
    )
    assert not _preserve_issue_created_managed_suppression(
        contract,
        active_exception=None,
    )


def test_interrupted_existing_pr_adoption_does_not_claim_durable_suppression():
    contract = ManagedCiContract(adopted_existing_pr=True)

    assert not _preserve_issue_created_managed_suppression(
        contract,
        active_exception=AgentLoopError("review failed"),
    )


def test_latest_pr_architecture_observation_uses_new_checkpoint_once():
    old_identity = {"target_revision": "main", "merge_base_revision": "old"}
    new_identity = {"target_revision": "release", "merge_base_revision": "new"}
    approval = _attach_round_metadata(
        "approved review",
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=1,
            subject="head", state="approved", architecture_identity=old_identity,
            architecture_contract_version=1,
        ),
    )
    checkpoint = _attach_round_metadata(
        "qualification checkpoint",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=2,
            subject="head", phase="qualification-checkpoint",
            architecture_identity=new_identity, architecture_contract_version=1,
        ),
    )
    comments = (SimpleNamespace(body=approval), SimpleNamespace(body=checkpoint))
    assert _latest_pr_architecture_observation(comments, head_sha="head") == new_identity


def test_legacy_same_head_approval_is_not_reusable_with_fresh_architecture_gate():
    approval = _attach_round_metadata(
        "approved review",
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=1,
            subject="head", state="approved",
        ),
    )
    comments = (SimpleNamespace(body=approval),)
    reused = _latest_pr_approved_reviews_for_head(
        comments,
        head_sha="head",
        configured_reviewers=("codex",),
        require_architecture_contract=True,
    )
    assert reused == {}


def _advance_head_after_coder(monkeypatch, runner, head_sha="repaired-head"):
    """Make the fake coder handoff obey the repair-head contract."""
    original = orchestrator._run_validated_agent

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = head_sha
        return response

    monkeypatch.setattr(orchestrator, "_run_validated_agent", run)


def _carried_ci_obligations() -> tuple[UnresolvedReviewItem, ...]:
    return tuple(
        UnresolvedReviewItem(
            item_id=f"item-{index}",
            reviewer=reviewer,
            source_round=1,
            text=f"{kind} failed on the previous head.",
            status="blocking",
            source_status="blocking",
            authority="machine",
            obligation_kind=kind,
            lifecycle="qualification_ready",
            failed_head_sha="old-head",
            candidate_head_sha="abc123",
            obligation_identity=f"{kind}:item-{index}",
        )
        for index, (reviewer, kind) in enumerate(
            (
                ("GitHub managed exact-head CI", "managed-exact-head-ci"),
                ("GitHub PR checks", "github-pr-checks"),
            ),
            start=1,
        )
    )


def _approved_issue_plan_comments(plan: str) -> list[dict[str, object]]:
    subject = orchestrator._plan_subject(plan)
    return [
        {
            "author": {"login": "coding-review-agent-loop"},
            "createdAt": "2026-05-01T00:00:00Z",
            "body": _attach_round_metadata(
                plan,
                PostedRoundMetadata(
                    flow="plan", role="coder", agent="Claude", round_number=1,
                    subject=subject, canonical_plan=plan,
                    raw_structured_coder_response=plan,
                ),
            ),
        },
        {
            "author": {"login": "coding-review-agent-loop"},
            "createdAt": "2026-05-01T00:01:00Z",
            "body": _attach_round_metadata(
                "Approved.",
                PostedRoundMetadata(
                    flow="plan", role="reviewer", agent="Codex",
                    round_number=1, subject=subject, state="approved",
                ),
            ),
        },
    ]


class _FreshScopeCaptured(Exception):
    pass


def test_pr_fresh_authorization_binds_server_recovered_approved_plan(tmp_path, monkeypatch):
    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    runner = FakeRunner(
        issue_comments=_approved_issue_plan_comments(plan),
        pr_payload={
            "headRefName": "agent-loop/managed-56", "headRefOid": "abc123",
            "baseRefName": "main", "body": "Fixes #56",
        },
    )
    captured = {}

    def authorize(*args, **kwargs):
        captured.update(kwargs)
        raise _FreshScopeCaptured

    monkeypatch.setattr(orchestrator, "authorize_fresh_issue_created_resume", authorize)
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",),
    )

    with pytest.raises(_FreshScopeCaptured):
        run_pr_loop(runner, pr_number=77, config=config)

    assert captured["issue_number"] == 56
    assert captured["approved_plan_hash"] == orchestrator.approved_plan_hash(plan)
    assert not any(command[:1] in (["claude"], ["codex"]) for command, _cwd in runner.commands)


def test_pr_ordinary_resume_binds_server_recovered_approved_plan_before_activation(
    tmp_path, monkeypatch,
):
    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    runner = FakeRunner(
        issue_comments=_approved_issue_plan_comments(plan),
        pr_payload={
            "headRefName": "agent-loop/managed-56", "headRefOid": "abc123",
            "baseRefName": "main", "body": "Fixes #56",
        },
    )
    handoff = orchestrator.AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha="abc123", branch="agent-loop/managed-56",
        trusted_actor_login="agent-loop", trusted_actor_id=1,
        protection_mode="voluntary", override_nonce="opening-nonce",
    )
    captured = {}
    monkeypatch.setattr(
        orchestrator, "recover_issue_created_handoff", lambda *_a, **_k: handoff
    )

    def revalidate(*_args, **kwargs):
        captured.update(kwargs)
        raise _FreshScopeCaptured

    monkeypatch.setattr(orchestrator, "revalidate_issue_created_handoff", revalidate)
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",),
    )

    with pytest.raises(_FreshScopeCaptured):
        run_pr_loop(runner, pr_number=77, config=config)

    assert captured["handoff"].approved_plan_hash == orchestrator.approved_plan_hash(plan)
    assert not any(command[:1] in (["claude"], ["codex"]) for command, _cwd in runner.commands)


def test_pr_fresh_authorization_rejects_mismatched_supplied_plan_scope(tmp_path, monkeypatch):
    canonical = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    supplied = orchestrator.make_approved_plan_context(
        "Different plan.\n\n### Plan steps\n1. Change the boundary.", source_locator="test"
    )
    runner = FakeRunner(
        issue_comments=_approved_issue_plan_comments(canonical),
        pr_payload={"headRefName": "agent-loop/managed-56", "body": "Fixes #56"},
    )
    monkeypatch.setattr(
        orchestrator, "authorize_fresh_issue_created_resume",
        lambda *args, **kwargs: (_ for _ in ()).throw(_FreshScopeCaptured()),
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",),
    )

    with pytest.raises(AgentLoopError, match="does not match the canonical issue plan"):
        run_pr_loop(
            runner, pr_number=77, config=config, approved_plan_context=supplied
        )


def test_pr_fresh_authorization_rejects_incomplete_plan_scope(tmp_path, monkeypatch):
    plan = "Unapproved plan."
    subject = orchestrator._plan_subject(plan)
    runner = FakeRunner(
        issue_comments=[{
            "author": {"login": "coding-review-agent-loop"},
            "createdAt": "2026-05-01T00:00:00Z",
            "body": _attach_round_metadata(
                plan,
                PostedRoundMetadata(
                    flow="plan", role="reviewer", agent="OpenAI Codex",
                    round_number=1, subject=subject, state="blocking",
                ),
            ),
        }],
        pr_payload={"headRefName": "agent-loop/managed-56", "body": "Fixes #56"},
    )
    monkeypatch.setattr(
        orchestrator, "authorize_fresh_issue_created_resume",
        lambda *args, **kwargs: (_ for _ in ()).throw(_FreshScopeCaptured()),
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",),
    )

    with pytest.raises(AgentLoopError, match="complete canonical approved plan"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_pr_fresh_authorization_allows_authenticated_planless_issue_scope(tmp_path, monkeypatch):
    runner = FakeRunner(
        pr_payload={"headRefName": "agent-loop/managed-56", "body": "Fixes #56"},
    )
    captured = {}

    def authorize(*args, **kwargs):
        captured.update(kwargs)
        raise _FreshScopeCaptured

    monkeypatch.setattr(orchestrator, "authorize_fresh_issue_created_resume", authorize)
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",),
    )

    with pytest.raises(_FreshScopeCaptured):
        run_pr_loop(runner, pr_number=77, config=config)

    assert captured["approved_plan_hash"] is None


def test_managed_issue_fix_round_publishes_correlated_head_continuity(
    tmp_path, monkeypatch,
):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="A repair is required.",
                blocking_items=["Fix the managed recovery edge case."],
            ),
            structured_pr_review(
                state="approved", summary="The repair is complete.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        pr_payload={
            "headRefName": "agent-loop/managed-56", "headRefOid": "abc123",
            "baseRefName": "main", "body": "Fixes #56",
        },
    )
    handoff = orchestrator.AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha="abc123", branch="agent-loop/managed-56",
        trusted_actor_login="agent-loop", trusted_actor_id=1,
        protection_mode="voluntary", override_nonce="root",
        authorization_kind="creation", authorization_comment_id=17,
    )
    selected = []
    published = []
    monkeypatch.setattr(orchestrator, "revalidate_issue_created_handoff", lambda *_a, **_k: handoff)
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci",
        lambda *_a, **_k: ManagedCiContract(protocol_version=2, issue_created_pr=True),
    )
    monkeypatch.setattr(orchestrator, "revalidate_adopted_managed_ci", lambda *_a, **_k: True)
    monkeypatch.setattr(orchestrator, "managed_label_present", lambda *_a, **_k: True)
    monkeypatch.setattr(
        orchestrator, "find_actor_round_metadata_comment_ids",
        lambda *_a, **kwargs: selected.append(kwargs) or (88, 89),
    )
    def publish(*_args, **kwargs):
        published.append(kwargs)
        raise _FreshScopeCaptured

    monkeypatch.setattr(
        orchestrator, "publish_issue_created_continuity_authorization", publish
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True, reviewer=("codex",), max_rounds=2,
    )

    with pytest.raises(_FreshScopeCaptured):
        run_pr_loop(
            runner, pr_number=77, config=config, managed_ci_handoff=handoff,
            managed_ci_issue_number=56,
        )

    assert len(selected) == len(published) == 1
    assert selected[0]["predecessor_head"] == "abc123"
    assert selected[0]["new_head"] == "abc123-coder-1"
    assert selected[0]["after_comment_id"] == 17
    assert published[0]["round_comment_ids"] == (88, 89)


def test_pr_resume_prompt_uses_current_wrapper_health_without_dropping_command(
    tmp_path, monkeypatch
):
    config = make_config(
        tmp_path,
        test_command=("pytest", "tests/test_protocol.py", "-q"),
    )
    memory = AgentMemoryContext(
        memory_dir=tmp_path / "memory",
        current_commit=None,
        last_analyzed_commit=None,
        changed_files=(),
        repo_summary=None,
        architecture_map=None,
        test_profile=None,
        toolchain=None,
    )
    monkeypatch.setattr(
        prompts_module,
        "preflight_wrapper_candidates",
        lambda **_kwargs: (runtime.LauncherProbeResult(("/opt/agent-loop", "run-tests"), "failed", "import failed"),),
    )
    prompt = build_followup_prompt(768, 2, "Repair the launcher path.", config, memory=memory)
    assert "pytest tests/test_protocol.py -q" in prompt
    assert "no wrapper candidate verified" in prompt


def test_orchestrator_finalization_guard_rejects_uncleared_machine_obligation():
    item = UnresolvedReviewItem(
        item_id="item-30",
        reviewer="GitHub managed exact-head CI",
        source_round=13,
        text="Managed exact-head CI failed.",
        status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="qualification_ready",
        failed_head_sha="oldhead123",
        candidate_head_sha="newhead123",
    )

    with pytest.raises(AgentLoopError, match="No approval or merge was attempted"):
        _ensure_finalization_ready(
            pr_number=777,
            round_number=20,
            items=[item],
            current_head_sha="newhead123",
        )


@pytest.mark.parametrize("status", ["skipped", "neutral"])
def test_ordinary_checks_authority_rejects_non_executed_passing_conclusions(status):
    check = PullRequestCheck(name="test", kind="check_run", status=status)
    checks = PullRequestChecks(
        state="passing",
        required_checks=("test",),
        passing=(check,),
        pending=(),
        failing=(),
        missing_required=(),
        branch_protection_status="configured",
        check_query_status="ok",
    )

    assert not orchestrator._ordinary_checks_snapshot_is_authoritative(checks)


def test_ordinary_checks_authority_requires_a_real_success_conclusion():
    check = PullRequestCheck(name="test", kind="check_run", status="success")
    checks = PullRequestChecks(
        state="passing",
        required_checks=("test",),
        passing=(check,),
        pending=(),
        failing=(),
        missing_required=(),
        branch_protection_status="configured",
        check_query_status="ok",
    )

    assert orchestrator._ordinary_checks_snapshot_is_authoritative(checks)


def test_ordinary_checks_authority_rejects_forbidden_branch_protection():
    checks = _watch_check_board("passing", protection="forbidden")

    assert not orchestrator._ordinary_checks_snapshot_is_authoritative(checks)


def test_ordinary_checks_authority_rejects_absent_partial_or_unavailable_boards():
    success = PullRequestCheck(name="test", kind="check_run", status="success")
    for state, query, missing in (
        ("no_checks", "ok", ()),
        ("unavailable", "unavailable", ()),
        ("passing", "partial", ()),
        ("passing", "ok", ("test",)),
    ):
        checks = PullRequestChecks(
            state=state,
            required_checks=("test",),
            passing=(success,),
            pending=(),
            failing=(),
            missing_required=missing,
            branch_protection_status="configured",
            check_query_status=query,
        )
        assert not orchestrator._ordinary_checks_snapshot_is_authoritative(checks)


def test_machine_authority_clears_only_its_own_obligation_kind():
    managed = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="GitHub managed exact-head CI",
        source_round=1,
        text="Managed CI failed.",
        status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="qualification_ready",
        failed_head_sha="old",
        candidate_head_sha="new",
    )
    ordinary = dataclasses.replace(
        managed,
        item_id="item-2",
        reviewer="GitHub PR checks",
        text="Ordinary checks failed.",
        obligation_kind="github-pr-checks",
        obligation_identity="github-pr-checks:item-2",
    )

    after_managed = orchestrator._clear_machine_obligations(
        [managed, ordinary], kind="managed-exact-head-ci"
    )
    after_ordinary = orchestrator._clear_machine_obligations(
        [managed, ordinary], kind="github-pr-checks"
    )

    assert [item.obligation_kind for item in after_managed] == ["github-pr-checks"]
    assert [item.obligation_kind for item in after_ordinary] == ["managed-exact-head-ci"]


def test_qualification_checkpoint_review_identity_is_fail_closed():
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="GitHub managed exact-head CI",
        source_round=1,
        text="Managed CI failed.",
        status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="qualifying",
        failed_head_sha="old",
        candidate_head_sha="new",
        obligation_identity="managed-exact-head-ci:item-1",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity=item.obligation_identity,
        lifecycle="qualifying",
        failed_head_sha="old",
        candidate_head_sha="new",
        base_branch="main",
        approval_digest="approval",
        requirements_digest="requirements",
        acquisition_digest="acquisition",
        scheduler_digest="scheduler",
        qualification_attempt_id="run/1",
        allowed_rounds=1,
    )

    assert not orchestrator._qualification_checkpoint_review_identity_matches(
        checkpoint,
        configured_reviewers=("codex",),
        current_approvals={},
        unresolved_items=[item],
        expected_plan_digest=None,
        expected_requirements_digest="requirements",
        expected_acquisition_digest="acquisition",
        expected_qualification_attempt_id="run/1",
    )
    complete = dataclasses.replace(
        checkpoint,
        approval_digest=orchestrator._qualification_digest(("Codex",)),
        scheduler_digest=orchestrator._qualification_digest(
            orchestrator._prior_item_ledger_signature([item])
        ),
    )
    assert orchestrator._qualification_checkpoint_review_identity_matches(
        complete,
        configured_reviewers=("codex",),
        current_approvals={"Codex": object()},
        unresolved_items=[item],
        expected_plan_digest=None,
        expected_requirements_digest="requirements",
        expected_acquisition_digest="acquisition",
        expected_qualification_attempt_id="run/1",
    )
    for field in (
        "plan_digest",
        "requirements_digest",
        "acquisition_digest",
        "scheduler_digest",
        "qualification_attempt_id",
    ):
        tampered = dataclasses.replace(complete, **{field: "different"})
        assert not orchestrator._qualification_checkpoint_review_identity_matches(
            tampered,
            configured_reviewers=("codex",),
            current_approvals={"Codex": object()},
            unresolved_items=[item],
            expected_plan_digest=None,
            expected_requirements_digest="requirements",
            expected_acquisition_digest="acquisition",
            expected_qualification_attempt_id="run/1",
        )


def test_machine_checkpoint_revert_returns_to_repair_required_without_value_error():
    item = UnresolvedReviewItem(
        item_id="item-30",
        reviewer="GitHub PR checks",
        source_round=1,
        text="Checks failed.",
        status="blocking",
        authority="machine",
        obligation_kind="github-pr-checks",
        lifecycle="awaiting_current_head_review",
        failed_head_sha="failed-head",
        candidate_head_sha="repair-head",
    )

    advanced = orchestrator._advance_machine_obligations_for_head(
        [item], current_head_sha="failed-head"
    )
    checkpoint = orchestrator._machine_obligation_checkpoint(
        advanced,
        current_head_sha="failed-head",
        base_branch="main",
        allowed_rounds=2,
        watch_failure_extension_used=False,
        watch_head_extension_used=False,
        lifecycle="awaiting_current_head_review",
    )

    assert advanced[0].lifecycle == "repair_required"
    assert advanced[0].candidate_head_sha is None
    assert checkpoint is not None
    assert checkpoint.lifecycle == "repair_required"
    assert checkpoint.candidate_head_sha is None


def test_approval_gated_managed_ci_wait_cannot_block_code_approval():
    item = UnresolvedReviewItem(
        item_id="item-33",
        reviewer="GitHub managed exact-head CI",
        source_round=9,
        text="Managed exact-head CI failed on the previous head.",
        status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="awaiting_current_head_review",
        failed_head_sha="failed-head",
        candidate_head_sha="repair-head",
    )
    parsed = parse_pr_review(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "blocking",
                "summary": "Code is ready; managed exact-head CI has not run.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [
                    {
                        "item_id": "item-33",
                        "disposition": "blocking",
                        "note": "The orchestrator must obtain exact-head qualification.",
                    }
                ],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        reviewer="OpenAI Codex",
    )

    normalized = orchestrator._normalize_approval_gated_managed_ci_review(
        parsed,
        prior_items=[item],
        pr_checks=_watch_check_board("passing"),
        current_head_sha="repair-head",
    )

    assert normalized.state == "approved"
    assert normalized.dispositions[0].disposition == "resolved"


@pytest.mark.parametrize("checks_state", ["pending", "unavailable"])
def test_approval_gated_managed_ci_wait_ignores_nonfailing_intermediate_checks(checks_state):
    item = UnresolvedReviewItem(
        item_id="item-33",
        reviewer="GitHub managed exact-head CI",
        source_round=9,
        text="Managed exact-head CI failed on the previous head.",
        status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="awaiting_current_head_review",
        failed_head_sha="failed-head",
        candidate_head_sha="repair-head",
    )
    parsed = parse_pr_review(
        structured_pr_review(
            state="blocking",
            summary="Managed exact-head qualification has not run yet.",
            prior_item_dispositions=[
                {
                    "item_id": "item-33",
                    "disposition": "blocking",
                    "note": "The orchestrator must obtain exact-head qualification.",
                }
            ],
        ),
        reviewer="OpenAI Codex",
    )

    normalized = orchestrator._normalize_approval_gated_managed_ci_review(
        parsed,
        prior_items=[item],
        pr_checks=_watch_check_board(checks_state),
        current_head_sha="repair-head",
    )

    assert normalized.state == "approved"
    assert normalized.dispositions[0].disposition == "resolved"


def test_approval_gated_managed_ci_wait_requires_both_heads_to_be_known():
    item = UnresolvedReviewItem(
        item_id="item-33",
        reviewer="GitHub managed exact-head CI",
        source_round=9,
        text="Managed exact-head CI failed on the previous head.",
        status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="awaiting_current_head_review",
        failed_head_sha=None,
        candidate_head_sha=None,
    )
    parsed = parse_pr_review(
        structured_pr_review(
            state="blocking",
            summary="Managed exact-head qualification has not run yet.",
            prior_item_dispositions=[
                {
                    "item_id": "item-33",
                    "disposition": "blocking",
                    "note": "The orchestrator must obtain exact-head qualification.",
                }
            ],
        ),
        reviewer="OpenAI Codex",
    )

    normalized = orchestrator._normalize_approval_gated_managed_ci_review(
        parsed,
        prior_items=[item],
        pr_checks=_watch_check_board("passing"),
        current_head_sha=None,
    )

    assert normalized.state == "blocking"


@pytest.mark.parametrize(
    ("mutate_item", "checks_state", "add_code_finding"),
    [
        ({"obligation_kind": "github-pr-checks"}, "passing", False),
        ({"lifecycle": "repair_required", "candidate_head_sha": None}, "passing", False),
        ({"candidate_head_sha": "different-head"}, "passing", False),
        ({}, "failing", False),
        ({}, "passing", True),
    ],
)
def test_managed_ci_wait_normalization_fails_closed(
    mutate_item, checks_state, add_code_finding
):
    item = UnresolvedReviewItem(
        item_id="item-33",
        reviewer="GitHub managed exact-head CI",
        source_round=9,
        text="Managed exact-head CI failed on the previous head.",
        status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="awaiting_current_head_review",
        failed_head_sha="failed-head",
        candidate_head_sha="repair-head",
    )
    item = dataclasses.replace(item, **mutate_item)
    payload = {
        "schema_version": 1,
        "kind": "pr_review",
        "state": "blocking",
        "summary": "Managed exact-head CI has not run.",
        "blocking_items": ([{"text": "A real code defect remains."}] if add_code_finding else []),
        "same_pr_followups": [],
        "future_followups": [],
        "prior_item_dispositions": [
            {
                "item_id": "item-33",
                "disposition": "blocking",
                "note": "The orchestrator must obtain exact-head qualification.",
            }
        ],
    }
    parsed = parse_pr_review(
        json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        reviewer="OpenAI Codex",
    )

    normalized = orchestrator._normalize_approval_gated_managed_ci_review(
        parsed,
        prior_items=[item],
        pr_checks=_watch_check_board(checks_state),
        current_head_sha="repair-head",
    )

    assert normalized.state == "blocking"


@pytest.mark.parametrize("review_parallel", [False, True])
def test_pr_loop_normalizes_managed_ci_wait_before_exact_head_dispatch(
    tmp_path, monkeypatch, review_parallel
):
    item = UnresolvedReviewItem(
        item_id="item-33",
        reviewer="GitHub managed exact-head CI",
        source_round=1,
        text="Managed exact-head CI failed on the previous head.",
        status="blocking",
        source_status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="awaiting_current_head_review",
        failed_head_sha="failed-head",
        candidate_head_sha="abc123",
        obligation_identity="managed-exact-head-ci:item-33",
    )
    coder_output = structured_coder_followup(
        summary="The failed managed-CI head was repaired.",
        addressed_items=[item.item_id],
    )
    coder_comment = _attach_round_metadata(
        coder_output,
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="abc123",
            prior_items=(item,),
            state="blocking",
            raw_structured_coder_response=coder_output,
        ),
    )
    blocking_wait_review = structured_pr_review(
        state="blocking",
        summary="Managed exact-head qualification must run before this can merge.",
        prior_item_dispositions=[
            {
                "item_id": item.item_id,
                "disposition": "blocking",
                "note": "The orchestrator must obtain exact-head qualification.",
            }
        ],
    )
    runner = FakeRunner(
        codex_outputs=[blocking_wait_review],
        pr_payload={
            "headRefOid": "abc123",
            "comments": [
                {"author": {"login": "coding-review-agent-loop"}, "body": coder_comment}
            ],
        },
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        auto_merge=True,
        max_rounds=1,
        review_parallel=review_parallel,
    )
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    dispatches = []
    monkeypatch.setattr(
        orchestrator,
        "dispatch_final_qualification",
        lambda *args, **kwargs: dispatches.append(kwargs),
    )
    monkeypatch.setattr(
        orchestrator,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    monkeypatch.setattr(orchestrator, "merge_pr", lambda *args, **kwargs: None)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert dispatches and dispatches[0]["expected_head_sha"] == "abc123"
    approved_comments = [
        comment for comment in runner.comments if comment.startswith("**Review verdict:** Approved")
    ]
    assert approved_comments
    assert not any("exact-head final sweep is missing reviewer approval" in comment for comment in runner.comments)


def _assert_pending_ci_stop_guidance(text):
    assert "This run cannot confirm the PR is merge-ready yet." in text
    assert "If checks pass, you can merge manually; no rerun is required." in text
    assert (
        "Rerun only if you want agent-loop to re-check or automate the final step."
        in text
    )
    assert "If checks fail, inspect/fix the failure or rerun so the loop can drive a fix." in text
    assert "Rerun after CI completes." not in text
    assert "Rerun once GitHub checks complete" not in text
    assert "because GitHub checks are still pending" not in text
    assert "because GitHub check status is unavailable" not in text


def test_selective_pr_loop_only_rechecks_owner_then_final_missing_reviewers(tmp_path, monkeypatch):
    def review(*, reviewer, state="approved", blocking_items=None, dispositions=None):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": state,
                    "summary": f"{reviewer} review",
                    "blocking_items": blocking_items or [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": dispositions or [],
                }
            )
            + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
        )

    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "worker cleanup is incomplete", "fix_scope": ["src/worker.py"]}
                ],
            ),
            review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[
            review(reviewer="Google Gemini"),
            review(reviewer="Google Gemini"),
        ],
        antigravity_outputs=[
            review(reviewer="Antigravity"),
            review(reviewer="Antigravity"),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
        max_rounds=4,
        pr_review_context_mode="compact",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    reviewer_commands = [
        command for command, _cwd in runner.commands
        if command and command[0] in {"codex", "gemini", "agy"}
    ]
    # Initial full board, one owner-only remediation turn, then the exact-head
    # sweep for the two paused reviewers.
    assert [command[0] for command in reviewer_commands].count("codex") == 2
    assert [command[0] for command in reviewer_commands].count("gemini") == 2
    assert [command[0] for command in reviewer_commands].count("agy") == 2


def test_selective_pr_loop_keeps_multiple_consecutive_narrow_fixes_selective(tmp_path, monkeypatch):
    def review(*, reviewer, state="approved", blocking_items=None, dispositions=None):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": state,
                    "summary": f"{reviewer} review",
                    "blocking_items": blocking_items or [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": dispositions or [],
                }
            )
            + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
        )

    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"]),
            structured_coder_followup(addressed_items=["item-1", "item-2"]),
        ],
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "first cleanup gap", "fix_scope": ["src/worker.py"]}
                ],
            ),
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "second cleanup gap", "fix_scope": ["src/worker.py"]}
                ],
                dispositions=[
                    {"item_id": "item-1", "disposition": "blocking", "note": "still open"}
                ],
            ),
            review(
                reviewer="OpenAI Codex",
                dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
        ],
        gemini_outputs=[
            review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
            ),
            review(reviewer="Google Gemini"),
        ],
        antigravity_outputs=[
            review(reviewer="Antigravity"),
            review(reviewer="Antigravity"),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
        max_rounds=5,
        pr_review_context_mode="compact",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    reviewer_commands = [
        command for command, _cwd in runner.commands
        if command and command[0] in {"codex", "gemini", "agy"}
    ]
    # Both approving reviewers remain paused across both narrow transitions;
    # they return once for the unchanged-head exact approval sweep.
    assert [command[0] for command in reviewer_commands].count("codex") == 3
    assert [command[0] for command in reviewer_commands].count("gemini") == 2
    assert [command[0] for command in reviewer_commands].count("agy") == 2


def test_selective_resume_with_malformed_scheduler_metadata_recovers_after_full_board(tmp_path):
    malformed_checkpoint = _attach_round_metadata(
        "stale scheduler checkpoint",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=1,
            subject="abc123",
            scheduler_contract={"policy": "selective-intermediate"},
        ),
    )
    runner = FakeRunner(
        pr_payload={
            "headRefOid": "abc123",
            "comments": [{"author": {"login": "bot"}, "body": malformed_checkpoint}],
        },
        codex_outputs=[structured_pr_review(summary="Codex reviewed the full board.")],
        gemini_outputs=[structured_pr_review(summary="Gemini reviewed the full board.", reviewer="Google Gemini")],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        pr_review_policy="selective-intermediate",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    reviewer_commands = [
        command for command, _cwd in runner.commands
        if command and command[0] in {"codex", "gemini"}
    ]
    # The invalid checkpoint cannot make the scheduler select a subset.
    assert [command[0] for command in reviewer_commands] == ["codex", "gemini"]
    assert not any(command[:1] == ["claude"] for command, _cwd in runner.commands)


def test_selective_final_sweep_missing_approval_stops_before_migration_or_merge(
    tmp_path, monkeypatch
):
    def review(*, reviewer, state="approved", blocking_items=None, dispositions=None):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": state,
                    "summary": f"{reviewer} review",
                    "blocking_items": blocking_items or [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": dispositions or [],
                }
            )
            + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
        )

    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    migration_calls = []
    monkeypatch.setattr(
        orchestrator,
        "validate_pr_migration_topology",
        lambda *args, **kwargs: migration_calls.append(True) or MigrationValidationResult(ok=True),
    )
    managed_ci_calls = []
    qualification_calls = []
    optional_test_calls = []
    monkeypatch.setattr(
        orchestrator,
        "dispatch_final_qualification",
        lambda *args, **kwargs: managed_ci_calls.append(True),
    )
    monkeypatch.setattr(
        orchestrator,
        "wait_for_final_qualification",
        lambda *args, **kwargs: qualification_calls.append(True),
    )
    monkeypatch.setattr(
        orchestrator,
        "run_optional_tests",
        lambda *args, **kwargs: optional_test_calls.append(True),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}
                ],
            ),
            review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[review(reviewer="Google Gemini")],
        antigravity_outputs=[review(reviewer="Antigravity")],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
        max_rounds=2,
    )

    with pytest.raises(AgentLoopError, match="exact-head final sweep is missing reviewer approval"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert migration_calls == []
    assert managed_ci_calls == []
    assert qualification_calls == []
    assert optional_test_calls == []
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)
    assert sum(command[:1] == ["claude"] for command, _cwd in runner.commands) == 1


def test_selective_final_sweep_blocker_dispatches_coder_remediation(tmp_path, monkeypatch):
    def review(*, reviewer, state="approved", blocking_items=None, dispositions=None):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": state,
                    "summary": f"{reviewer} review",
                    "blocking_items": blocking_items or [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": dispositions or [],
                }
            )
            + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
        )

    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"]),
            structured_coder_followup(addressed_items=["item-2"]),
        ],
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "first cleanup gap", "fix_scope": ["src/worker.py"]}
                ],
            ),
            review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
            review(reviewer="OpenAI Codex"),
        ],
        gemini_outputs=[
            review(reviewer="Google Gemini"),
            review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[
                    {"text": "final-sweep regression", "fix_scope": ["src/worker.py"]}
                ],
            ),
            review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
            ),
        ],
        antigravity_outputs=[
            review(reviewer="Antigravity"),
            review(reviewer="Antigravity"),
            review(reviewer="Antigravity"),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
        max_rounds=4,
    )

    with pytest.raises(AgentLoopError, match="exact-head final sweep is missing reviewer approval"):
        run_pr_loop(runner, pr_number=77, config=config)

    coder_commands = [
        command for command, _cwd in runner.commands if command and command[0] == "claude"
    ]
    assert len(coder_commands) == 2
    assert "final-sweep regression" in coder_commands[1][-1]


def test_selective_owner_unavailability_stops_without_coder_redispatch(tmp_path, monkeypatch):
    def review(*, reviewer, state="approved", blocking_items=None, dispositions=None):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": state,
                    "summary": f"{reviewer} review",
                    "blocking_items": blocking_items or [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": dispositions or [],
                }
            )
            + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
        )

    unavailable = json.dumps(
        {
            "schema_version": 1,
            "kind": "agent_unavailable",
            "retryable": False,
            "category": "environment",
            "summary": "The reviewer cannot access the repository history.",
            "suggested_action": "Repair the reviewer environment before retrying.",
        }
    ) + "\n<!-- AGENT_UNAVAILABLE -->\n-- OpenAI Codex"
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}
                ],
            ),
            unavailable,
        ],
        gemini_outputs=[review(reviewer="Google Gemini")],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        pr_review_policy="selective-intermediate",
        max_rounds=3,
    )

    with pytest.raises(AgentLoopError, match="all remaining resolution owners for item-1 are unavailable"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert sum(command[:1] == ["claude"] for command, _cwd in runner.commands) == 1


def test_selective_resume_uses_a_valid_persisted_scheduler_checkpoint(tmp_path):
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Codex",
        source_round=1,
        text="The worker cleanup still needs verification.",
        status="blocking",
        source_status="blocking",
        fix_scope=("src/worker.py",),
        resolution_owners=("Codex",),
        owner_states=(("Codex", "pending"),),
    )
    contract = orchestrator.make_contract(
        ("Codex", "Gemini", "Antigravity"),
        "selective-intermediate",
        None,
    )
    checkpoint = _attach_round_metadata(
        "Persisted selective reconciliation checkpoint.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=2,
            subject="abc123",
            prior_items=(item,),
            phase="reconciliation",
            scheduler_contract=contract.as_dict(),
            scheduler_previous_sha="abc123",
            scheduler_current_sha="abc123",
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex",),
            scheduler_paused_reviewers=(
                ("Gemini", "approval is historical"),
                ("Antigravity", "approval is historical"),
            ),
            scheduler_reasons=("pending resolution owner",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=3,
        ),
    )

    def review(*, reviewer, dispositions=()):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": "approved",
                    "summary": f"{reviewer} review",
                    "blocking_items": [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": list(dispositions),
                }
            )
            + f"\n<!-- AGENT_STATE: approved -->\n-- {reviewer}"
        )

    runner = FakeRunner(
        pr_payload={
            "headRefOid": "abc123",
            "comments": [{"author": {"login": "bot"}, "body": checkpoint}],
        },
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
        gemini_outputs=[review(reviewer="Google Gemini")],
        antigravity_outputs=[review(reviewer="Antigravity")],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
        max_rounds=3,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    agent_commands = [
        command[0]
        for command, _cwd in runner.commands
        if command and command[0] in {"claude", "codex", "gemini", "agy"}
    ]
    assert agent_commands == ["codex", "gemini", "agy"]
    assert "Persisted selective reconciliation checkpoint." not in "\n".join(
        command[-1] for command, _cwd in runner.commands if command and command[0] == "claude"
    )


def test_selective_same_head_requirement_edit_invalidates_approval_before_merge(
    tmp_path, monkeypatch
):
    requirement_1 = HumanReviewRequirement(
        source_type="PR comment",
        author="maintainer",
        created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="Keep the current audit trail.",
    )
    requirement_2 = HumanReviewRequirement(
        source_type="PR comment",
        author="maintainer",
        created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="The edited instruction changes the required audit trail.",
    )
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Selective requirement edit",
        head_branch="feature/review-context",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/77",
    )
    context_calls = 0

    def changing_context(*args, **kwargs):
        nonlocal context_calls
        context_calls += 1
        return PullRequestReviewContext(
            metadata=metadata,
            comments=(),
            human_requirements=(requirement_1 if context_calls == 1 else requirement_2,),
        )

    monkeypatch.setattr(orchestrator, "get_pr_review_context", changing_context)
    runner = FakeRunner(
        claude_outputs=[],
        codex_outputs=[
            structured_pr_review(
                summary="Codex approves the original requirement.",
                reviewer="OpenAI Codex",
                human_requirements_resolved=True,
            ),
            structured_pr_review(
                summary="Codex approves the edited requirement.",
                reviewer="OpenAI Codex",
                human_requirements_resolved=True,
            ),
        ],
        gemini_outputs=[
            structured_pr_review(
                summary="Gemini approves the original requirement.",
                reviewer="Google Gemini",
                human_requirements_resolved=True,
            ),
            structured_pr_review(
                summary="Gemini approves the edited requirement.",
                reviewer="Google Gemini",
                human_requirements_resolved=True,
            ),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        pr_review_policy="selective-intermediate",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    reviewer_commands = [
        command
        for command, _cwd in runner.commands
        if command and command[0] in {"codex", "gemini"}
    ]
    assert [command[0] for command in reviewer_commands].count("codex") == 2
    assert [command[0] for command in reviewer_commands].count("gemini") == 2
    assert any(requirement_2.body in command[-1] for command in reviewer_commands)
    assert not any(command[:1] == ["claude"] for command, _cwd in runner.commands)


def test_selective_plan_handoff_change_stops_before_migration_or_merge(tmp_path, monkeypatch):
    plan = "Approved plan.\n\n## Scope\n- Preserve the current API."
    plan_context = orchestrator.make_approved_plan_context(plan, source_locator="test")
    handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=plan_context.plan_hash,
    )
    plan_record = _attach_round_metadata(
        plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=plan_context.plan_subject or "plan",
            canonical_plan=plan,
            raw_structured_coder_response=plan,
        ),
    )
    initial_issue = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Original issue.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(author="bot", created_at="2026-05-01T00:00:00Z", body=plan_record),
            IssueComment(author="bot", created_at="2026-05-01T00:01:00Z", body=handoff),
        ),
        human_requirements=(),
    )
    changed_issue = dataclasses.replace(initial_issue, comments=(initial_issue.comments[0],))
    issue_fetches = 0

    def changing_issue(*args, **kwargs):
        nonlocal issue_fetches
        issue_fetches += 1
        return initial_issue if issue_fetches == 1 else changed_issue

    monkeypatch.setattr(orchestrator, "get_issue_context", changing_issue)
    migration_calls = []
    monkeypatch.setattr(
        orchestrator,
        "validate_pr_migration_topology",
        lambda *args, **kwargs: migration_calls.append(True) or MigrationValidationResult(ok=True),
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                summary="The approved plan is implemented.",
                reviewer="OpenAI Codex",
            )
        ]
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        max_rounds=1,
    )

    with pytest.raises(AgentLoopError, match="Approved-plan/handoff identity changed"):
        run_pr_loop(
            runner,
            pr_number=77,
            config=config,
            issue_context=initial_issue,
            approved_plan_context=plan_context,
        )
    assert migration_calls == []
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_selective_valid_plan_handoff_replacement_sweeps_before_migration(
    tmp_path, monkeypatch
):
    old_plan = "Approved plan.\n\n## Scope\n- Preserve the current API."
    new_plan = "Approved replacement plan.\n\n## Scope\n- Preserve the current API and audit trail."
    old_plan_context = orchestrator.make_approved_plan_context(old_plan, source_locator="test")
    new_plan_context = orchestrator.make_approved_plan_context(new_plan, source_locator="test")
    old_handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=old_plan_context.plan_hash,
    )
    old_handoff_metadata = find_latest_issue_pr_handoff(
        [IssueComment(author="bot", created_at="2026-05-01T00:01:00Z", body=old_handoff)],
        issue_number=56,
        repo="OWNER/REPO",
    )
    assert old_handoff_metadata is not None
    new_handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=new_plan_context.plan_hash,
        supersedes_hash=old_handoff_metadata.contract_hash,
    )

    def plan_record(plan_text, timestamp):
        plan_context = orchestrator.make_approved_plan_context(plan_text, source_locator="test")
        return IssueComment(
            author="bot",
            created_at=timestamp,
            body=_attach_round_metadata(
                plan_text,
                PostedRoundMetadata(
                    flow="plan",
                    role="coder",
                    agent="Claude",
                    round_number=1,
                    subject=plan_context.plan_subject or "plan",
                    canonical_plan=plan_text,
                    raw_structured_coder_response=plan_text,
                ),
            ),
        )

    old_plan_record = plan_record(old_plan, "2026-05-01T00:00:00Z")
    new_plan_record = plan_record(new_plan, "2026-05-01T00:02:00Z")
    initial_issue = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Original issue.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            old_plan_record,
            IssueComment(author="bot", created_at="2026-05-01T00:01:00Z", body=old_handoff),
        ),
        human_requirements=(),
    )
    replacement_issue = dataclasses.replace(
        initial_issue,
        comments=(
            old_plan_record,
            IssueComment(author="bot", created_at="2026-05-01T00:01:00Z", body=old_handoff),
            new_plan_record,
            IssueComment(author="bot", created_at="2026-05-01T00:02:00Z", body=new_handoff),
        ),
    )
    issue_fetches = 0

    def changing_issue(*args, **kwargs):
        nonlocal issue_fetches
        issue_fetches += 1
        return initial_issue if issue_fetches == 1 else replacement_issue

    monkeypatch.setattr(orchestrator, "get_issue_context", changing_issue)
    migration_calls = []

    def validate_migration(*args, **kwargs):
        migration_calls.append(True)
        reviewer_commands = [
            command
            for command, _cwd in runner.commands
            if command and command[0] == "codex"
        ]
        assert len(reviewer_commands) == 2
        return MigrationValidationResult(ok=True)

    monkeypatch.setattr(orchestrator, "validate_pr_migration_topology", validate_migration)
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(summary="Approval bound to the original plan."),
            structured_pr_review(summary="Approval bound to the replacement plan."),
        ]
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        max_rounds=2,
    )

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=config,
        issue_context=initial_issue,
        approved_plan_context=old_plan_context,
    ) == 0

    assert migration_calls == [True]
    reviewer_commands = [
        command
        for command, _cwd in runner.commands
        if command and command[0] == "codex"
    ]
    assert any(new_plan in " ".join(command) for command in reviewer_commands)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_selective_scheduler_contract_change_during_managed_ci_blocks_merge(
    tmp_path, monkeypatch
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(summary="Initial exact-head approval.")]
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        auto_merge=True,
        max_rounds=1,
    )
    monkeypatch.setattr(
        orchestrator,
        "activate_managed_ci",
        lambda *args, **kwargs: ManagedCiContract(),
    )
    monkeypatch.setattr(
        orchestrator,
        "dispatch_final_qualification",
        lambda *args, **kwargs: None,
    )
    changed_contract = orchestrator.make_contract(
        ("Codex",), "all-reviewers", None
    )
    changed_audit = _attach_round_metadata(
        "A scheduler contract was changed while managed CI was running.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=2,
            subject="abc123",
            phase="reconciliation",
            scheduler_contract=changed_contract.as_dict(),
            scheduler_previous_sha=None,
            scheduler_current_sha="abc123",
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex",),
            scheduler_paused_reviewers=(),
            scheduler_reasons=("contract changed during managed CI",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=0,
        ),
    )

    def wait_for_managed_ci(*args, **kwargs):
        runner.pr_payload.setdefault("comments", []).append(
            {"author": {"login": "bot"}, "body": changed_audit}
        )
        return ManagedCiOutcome(status="passed", head_sha="abc123")

    monkeypatch.setattr(
        orchestrator,
        "wait_for_final_qualification",
        wait_for_managed_ci,
    )
    monkeypatch.setattr(
        orchestrator,
        "merge_pr",
        lambda *args, **kwargs: pytest.fail("scheduler contract changes must prevent merge"),
    )

    with pytest.raises(AgentLoopError, match="scheduler contract changed during qualification"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def _issue_view_commands(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "issue", "view"]]


def test_direct_pr_resolves_single_linked_issue_context_for_reviewer(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        issue_payload={
            "number": 56,
            "title": "Original acceptance criteria",
            "body": "Keep compatibility.\n\n-- Human Reviewer",
        },
        issue_comments=[
            {"author": {"login": "maintainer"}, "createdAt": "2026-06-01T00:00:00Z", "body": "Later clarification."}
        ],
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(tmp_path, quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(_issue_view_commands(runner)) == 1
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Original acceptance criteria" in prompt
    assert "Keep compatibility." in prompt
    assert "Later clarification." in prompt
    assert "Human requirements" in prompt


def test_pr_review_runs_when_pr_and_issue_text_name_a_reserved_token(tmp_path, capsys):
    """Issue #891: naming a record in PR or issue prose must not abort the run.

    Both surfaces carry the token here, and the run must review normally with
    the token defanged in the reviewer prompt.
    """
    token = "AGENT_PLAN_APPROVED_FOLLOWUPS"
    label = "[protocol PLAN_APPROVED_FOLLOWUPS record]"
    # A `well-formed-only` entry has no bare-name fallback in the scanner, so
    # cover one on the title surfaces too.
    strict_token = "AGENT_ISSUE_PR_HANDOFF"
    strict_label = "[protocol ISSUE_PR_HANDOFF record]"
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        issue_payload={
            "number": 56,
            "title": f"Reserved {strict_token} label",
            "body": f"The planner writes {token} when it approves follow-ups.",
        },
        issue_comments=[
            {
                "author": {"login": "maintainer"},
                "createdAt": "2026-09-19T00:00:00Z",
                "body": f"The {token} name also appears in a comment.",
            }
        ],
        pr_payload={
            "title": f"Rename the {strict_token} label",
            "body": f"Fixes #56\n\nThis PR renames the {token} record label.",
        },
    )
    config = make_config(tmp_path, quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert token not in prompt
    assert strict_token not in prompt
    assert label in prompt
    assert f"- Title: Rename the {strict_label} label" in prompt
    # Surrounding prose survives, so the reviewer still sees the request.
    assert "when it approves follow-ups." in prompt
    assert "name also appears in a comment." in prompt
    # The run says which surfaces named the tokens, titles included, and that
    # they grant no authority.
    logged = capsys.readouterr().err
    assert (
        f"Issue #56 text names reserved protocol marker(s) {strict_token}, {token}"
        in logged
    )
    assert (
        f"Pull request #77 text names reserved protocol marker(s) {strict_token}, {token}"
        in logged
    )
    assert logged.count("They carry no authority.") == 2


def test_plain_pr_recovery_accepts_loop_created_managed_pr_body(tmp_path):
    encoded = base64.urlsafe_b64encode(
        json.dumps(
            {"source_branch": "feature/review-context", "source_sha": "abc123"},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).decode()
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={
            "body": f"<!-- AGENT_MANAGED_PR_SOURCE_V1 {encoded} -->",
            "headRefName": "agent-loop/managed-direct-token",
        },
    )

    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path)) == 0
    assert runner.comments == ["**Review verdict:** Approved\n\nLGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex: unknown model (medium)"]


def test_plain_pr_recovery_accepts_managed_pr_head_after_creation_sha_advanced(tmp_path):
    encoded = base64.urlsafe_b64encode(
        json.dumps(
            {"source_branch": "feature/review-context", "source_sha": "abc123"},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).decode()
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={
            "body": f"<!-- AGENT_MANAGED_PR_SOURCE_V1 {encoded} -->",
            "headRefName": "agent-loop/managed-direct-token",
            "headRefOid": "abc123-coder-1",
        },
    )

    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path)) == 0
    assert runner.comments == ["**Review verdict:** Approved\n\nLGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex: unknown model (medium)"]


def test_in_process_managed_pr_handoff_still_requires_creation_sha(tmp_path):
    encoded = base64.urlsafe_b64encode(
        json.dumps(
            {"source_branch": "feature/review-context", "source_sha": "abc123"},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).decode()
    runner = FakeRunner(
        pr_payload={
            "body": f"<!-- AGENT_MANAGED_PR_SOURCE_V1 {encoded} -->",
            "headRefName": "agent-loop/managed-direct-token",
            "headRefOid": "abc123-coder-1",
        },
    )

    with pytest.raises(AgentLoopError, match="Managed PR head SHA does not match"):
        run_pr_loop(
            runner,
            pr_number=77,
            config=make_config(tmp_path),
            managed_pr_origin=(
                "feature/review-context",
                "abc123",
                "agent-loop/managed-direct-token",
                None,
            ),
        )


def test_approved_followup_public_renderings_sanitize_historical_marker_mentions():
    followup = ApprovedFollowup(
        reviewer="Claude",
        text="Document AGENT_APPROVED_FOLLOWUPS handling.",
    )
    reconciliation = reconcile_approved_followups([followup])

    summary = _format_approved_followup_summary(77, reconciliation)
    issue_body = _followup_issue_body(77, reconciliation.selected_groups[0])

    assert "AGENT_APPROVED_FOLLOWUPS" not in summary
    assert "AGENT_APPROVED_FOLLOWUPS" not in issue_body

    source = PlanApprovedFollowupSource(
        item_id="item-1",
        reviewer="Claude",
        source_round=1,
        text="Document AGENT_PLAN_APPROVED_FOLLOWUPS handling.",
        notes=("AGENT_SPLIT_UNFILED_WARNING was also discussed.",),
    )
    plan_reconciliation = reconcile_plan_approved_followups([source])
    plan_body = _plan_followup_issue_body(
        issue_number=56,
        plan_hash="abc123",
        plan_subject="Plan AGENT_MANAGED_PR_SOURCE_V1 handling.",
        followup=plan_reconciliation.selected_groups[0],
    )
    plan_summary = _format_plan_approval_summary_with_followups(
        56,
        "Plan AGENT_MANAGED_PR_SOURCE_V1 handling.",
        reconciliation=plan_reconciliation,
    )

    for rendered in (plan_body, plan_summary):
        assert "AGENT_PLAN_APPROVED_FOLLOWUPS" not in rendered
        assert "AGENT_SPLIT_UNFILED_WARNING" not in rendered
        assert "AGENT_MANAGED_PR_SOURCE_V1" not in rendered


def test_direct_pr_keeps_linked_issue_context_for_coder_followup(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Needs a small correction.\n\n### Same-PR follow-ups\n"
            "- Correct the implementation.\n\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Corrected it.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        issue_payload={"title": "Linked issue title", "body": "Linked issue body"},
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_prompt = next(
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]
    )
    assert "Linked issue title" in followup_prompt
    assert "Linked issue body" in followup_prompt


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
    plan_context = orchestrator.make_approved_plan_context(
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
        and orchestrator._decode_round_metadata(
            item["body"].split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
        ).role == "coder"
    ]
    assert coder_metadata_comments
    coder_metadata = orchestrator._decode_round_metadata(
        coder_metadata_comments[-1].split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
    )
    assert coder_metadata.risk_test_matrix_evidence is not None
    assert coder_metadata.risk_test_matrix_evidence["rows"][0]["row_id"] == (
        "row-post-review-recovery"
    )
    assert "execution_ref" not in coder_metadata_comments[-1]


def _followup_matrix_context():
    matrix = parse_risk_test_matrix({
        "applicability": "applicable",
        "rows": [{
            "row_id": "followup-derived-evidence",
            "label": "Follow-up evidence reaches the authenticated PR head",
            "entry_path_or_mode": "PR review repair round / coder_followup",
            "initial_state": "blocking review on a predecessor head",
            "event": "the coder follow-up is authenticated",
            "expected_outcome": "canonical evidence is derived for the repaired head",
            "forbidden_side_effects": ["Do not lose the follow-up PR handoff."],
            "proposed_test_level": "orchestrator",
            "proposed_test_location": "tests/test_orchestrator_pr.py",
            "applicability": "required",
            "related_scope_item_ids": ["scope-orchestration-integration"],
            "execution_owner": "one-shot",
        }],
        "important_exclusions": ["Planned tests are not evidence."],
    })
    approved_plan = "Approved follow-up plan.\n\n" + render_risk_test_matrix_section(matrix)
    identity = risk_test_matrix_identity(matrix)
    return orchestrator.make_approved_plan_context(
        approved_plan,
        source_locator="test approved follow-up plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )


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


def _followup_observation(
    *, execution_ref: str, receipt_id: str, head: str, timestamp: str,
    tracked_digest: str = "tree-current",
):
    return LocalTestObservation(
        command=("python3", "-m", "pytest", "tests/test_orchestrator_pr.py", "-q"),
        outcome="passed",
        provenance="parent-observed",
        receipt_id=receipt_id,
        execution_ref=execution_ref,
        turn_id="coder-turn",
        timestamp=timestamp,
        cwd="/tmp/followup-checkout",
        normalized_command="python3 -m pytest tests/test_orchestrator_pr.py -q",
        attribution=TreeAttribution(
            state="current-head",
            head=head,
            tracked_digest=tracked_digest,
            stable=True,
        ),
        environment_state="equivalent",
        environment_identity=EnvironmentIdentity("followup-test", b"followup-test"),
        wrapper_bootstrap="verified",
        inner_exec="started",
        suite_start="verified",
    )


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
    real_validated_agent = orchestrator._run_validated_agent

    def fake_validated_agent(*args, **kwargs):
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = "repaired-head"
            runner.git_head = "repaired-head"
            return coder_response
        return real_validated_agent(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_run_validated_agent", fake_validated_agent)
    monkeypatch.setattr(
        orchestrator,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head=runner.pr_payload["headRefOid"],
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_read_assigned_workdir_head",
        lambda *_args, **_kwargs: runner.git_head,
    )

    real_run_agent_result = orchestrator.run_agent_result
    real_get_pr_review_context = orchestrator.get_pr_review_context
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

    monkeypatch.setattr(orchestrator, "get_pr_review_context", get_pr_context)

    def fake_correction(*_args, **kwargs):
        nonlocal race_pending
        if kwargs.get("label") != "semantic-evidence-correction":
            return real_run_agent_result(*_args, **kwargs)
        if mode == "head-race":
            race_pending = True
        return SimpleNamespace(
            text=("not a structured response" if mode == "exhausted" else corrected_text)
        )

    monkeypatch.setattr(orchestrator, "run_agent_result", fake_correction)

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
    real_validated_agent = orchestrator._run_validated_agent

    def fake_validated_agent(*args, **kwargs):
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = "repaired-head"
            runner.git_head = "repaired-head"
            return coder_response
        return real_validated_agent(*args, **kwargs)

    def forbidden_repair(*_a, **_k):
        raise AssertionError("repair must not run for dropped execution_refs")

    monkeypatch.setattr(orchestrator, "_run_validated_agent", fake_validated_agent)
    monkeypatch.setattr(orchestrator, "attempt_repair", forbidden_repair)
    monkeypatch.setattr(orchestrator, "execute_repair", forbidden_repair)
    monkeypatch.setattr(
        orchestrator,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head=runner.pr_payload["headRefOid"],
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_read_assigned_workdir_head",
        lambda *_args, **_kwargs: runner.git_head,
    )
    real_run_agent_result = orchestrator.run_agent_result
    correction_prompts = []

    def fake_correction(*_args, **kwargs):
        if kwargs.get("label") != "semantic-evidence-correction":
            return real_run_agent_result(*_args, **kwargs)
        correction_prompts.append(kwargs["prompt"])
        return SimpleNamespace(text=command_text)

    monkeypatch.setattr(orchestrator, "run_agent_result", fake_correction)

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


@pytest.mark.parametrize(
    "body, issue_views",
    [
        ("https://github.com/owner/repo/issues/56", 1),
        ("No issue reference.", 0),
        ("Fixes #56; Resolves #57", 0),
        ("Fixes #56; https://github.com/OWNER/REPO/issues/56", 1),
        ("Fixes other/repo#56 https://github.com/other/repo/issues/57", 0),
    ],
)
def test_direct_pr_linked_issue_resolution_handles_urls_absence_and_ambiguity(
    tmp_path, body, issue_views
):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"body": body},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(_issue_view_commands(runner)) == issue_views


def test_issue_mode_context_is_not_replaced_by_pr_link(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        issue_payload={"title": "Fresh issue", "body": "Fresh issue body."},
        pr_payload={"body": "Fixes #99"},
    )
    config = make_config(tmp_path)
    supplied_context = IssueContext(
        number=56, repo="OWNER/REPO", title="Caller issue", body="Caller body", url=None, comments=()
    )

    assert run_pr_loop(runner, pr_number=77, config=config, issue_context=supplied_context) == 0

    assert len(_issue_view_commands(runner)) == 1
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Fresh issue" in prompt
    assert "Caller issue" not in prompt


def test_direct_pr_ignores_unrelated_older_issue_handoff(tmp_path):
    older_handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=66,
        pr_url="https://github.com/OWNER/REPO/pull/66",
        pr_head_sha="older-head",
        flow="issue-implementation",
        plan_hash=None,
    )
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        issue_comments=[
            {
                "author": {"login": "coding-review-agent-loop"},
                "createdAt": "2026-05-01T00:00:00Z",
                "body": older_handoff,
            }
        ],
        pr_payload={"body": "Fixes #56"},
    )

    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path)) == 0

    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert not any("handoff selects PR #66" in comment for comment in runner.comments)


def test_direct_pr_resume_uses_handoff_bound_plan_when_later_plan_exists(tmp_path):
    old_matrix = parse_risk_test_matrix({
        "applicability": "applicable",
        "rows": [{
            "row_id": "row-review-only-recovery",
            "label": "Review-only recovery reaches the intended branch",
            "entry_path_or_mode": "ordinary / review-only",
            "initial_state": "review complete, recovery pending",
            "event": "repaired head passes",
            "expected_outcome": "Complete without requiring a watcher",
            "forbidden_side_effects": ["Do not merge a stale head"],
            "proposed_test_level": "orchestrator",
            "proposed_test_location": "tests/test_orchestrator_pr.py::test_review_only_recovery",
            "applicability": "applicable",
            "related_scope_item_ids": ["scope-review-recovery"],
            "execution_owner": "one-shot",
        }],
        "important_exclusions": ["Unrelated managed-CI combinations."],
    })
    later_matrix = parse_risk_test_matrix({
        **old_matrix.to_payload(),
        "rows": [{
            **old_matrix.rows[0].to_payload(),
            "row_id": "row-unrelated-later-plan",
            "expected_outcome": "Use a watcher before completion",
        }],
    })
    old_identity = risk_test_matrix_identity(old_matrix)
    later_identity = risk_test_matrix_identity(later_matrix)
    old_plan = "Preserve the old API.\n\nApproved old plan.\n\n" + render_risk_test_matrix_section(old_matrix)
    later_plan = "Replace the old API.\n\nUnrelated later plan.\n\n" + render_risk_test_matrix_section(later_matrix)
    old_context = orchestrator.make_approved_plan_context(
        old_plan,
        source_locator="test old approved plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=old_matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=old_identity,
        risk_test_matrix_boundary_digest=old_identity,
    )
    later_context = orchestrator.make_approved_plan_context(
        later_plan,
        source_locator="test later plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=later_matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=later_identity,
        risk_test_matrix_boundary_digest=later_identity,
    )
    old_plan_comment = _attach_round_metadata(
        old_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=old_context.plan_subject or "old-plan",
            canonical_plan=old_plan,
            raw_structured_coder_response=old_plan,
            risk_test_matrix_contract_version=1,
            risk_test_matrix_payload=old_matrix.to_payload(),
            risk_test_matrix_identity=old_identity,
            risk_test_matrix_boundary_digest=old_identity,
        ),
    )
    later_plan_comment = _attach_round_metadata(
        later_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=later_context.plan_subject or "later-plan",
            canonical_plan=later_plan,
            raw_structured_coder_response=later_plan,
            risk_test_matrix_contract_version=1,
            risk_test_matrix_payload=later_matrix.to_payload(),
            risk_test_matrix_identity=later_identity,
            risk_test_matrix_boundary_digest=later_identity,
        ),
    )
    handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=orchestrator.approved_plan_hash(old_plan),
    )
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-01T00:00:00Z", "body": old_plan_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-01T00:01:00Z", "body": handoff},
            {"author": {"login": "bot"}, "createdAt": "2026-05-01T00:02:00Z", "body": later_plan_comment},
        ],
        issue_payload={"number": 56, "title": "Issue", "body": "Original issue."},
        pr_payload={
            "number": 77,
            "url": "https://github.com/OWNER/REPO/pull/77",
            "body": "Fixes #56",
        },
    )

    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path)) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    plan_block = prompt.split("Approved implementation plan context", 1)[1].split(
        "Target child/primary issue context", 1
    )[0]
    assert "Preserve the old API." in plan_block
    assert "Replace the old API." not in plan_block
    assert "row-review-only-recovery" in plan_block
    assert "row-unrelated-later-plan" not in plan_block


def test_pr_loop_runs_tests_and_merge_only_after_codex_approval(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    config = make_config(
        tmp_path,
        auto_merge=True,
        test_command=("pytest", "tests/test_agent_loop.py"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["pytest", "tests/test_agent_loop.py"] in commands
    assert [
        "gh",
        "api",
        "repos/OWNER/REPO/commits/abc123/check-runs",
    ] in commands
    assert [
        "gh",
        "api",
        "repos/OWNER/REPO/commits/abc123/status",
    ] in commands
    assert [
        "gh",
        "api",
        "repos/OWNER/REPO/branches/main/protection/required_status_checks",
    ] in commands
    assert ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge", "--match-head-commit", "abc123"] in commands


def test_ordinary_recovery_readies_and_merges_exact_head(monkeypatch, tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, auto_merge=True)
    capability = OrdinaryRecoveryCapability(
        pr_number=77, repository="OWNER/REPO", base_ref="main", expected_head_sha="abc123",
        released_label_event_id=101, released_at=100,
    )
    validations = []
    merged = []
    monkeypatch.setattr(orchestrator, "refresh_ordinary_recovery_capability", lambda *args, **kwargs: capability)
    monkeypatch.setattr(
        orchestrator,
        "wait_for_ordinary_recovery",
        lambda *args, **kwargs: ManagedCiOutcome(
            status="passed",
            checks=_watch_check_board("passing"),
            head_sha="abc123",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "get_pr_review_context",
        lambda *args, **kwargs: SimpleNamespace(metadata=PullRequestMetadata(
            number=77, repo="OWNER/REPO", title="draft", head_branch="feature",
            base_branch="main", head_sha="abc123", url=None,
        )),
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_ordinary_recovery_capability",
        lambda *args, **kwargs: validations.append(kwargs.get("require_draft", True)) or True,
    )
    monkeypatch.setattr(
        orchestrator,
        "merge_pr",
        lambda *args, **kwargs: merged.append(kwargs["expected_head_sha"]),
    )

    orchestrator._finalize_ordinary_recovery_merge(
        runner, config=config, pr_number=77, capability=capability,
    )

    assert validations == [True, None]
    assert merged == ["abc123"]
    assert ["gh", "pr", "ready", "77", "--repo", "OWNER/REPO"] in [
        command for command, _cwd in runner.commands
    ]


@pytest.mark.parametrize("board", ["absent", "neutral", "skipped", "forbidden"])
def test_ordinary_recovery_does_not_ready_or_merge_without_authoritative_board(
    monkeypatch, tmp_path, board
):
    runner = FakeRunner()
    config = make_config(tmp_path, auto_merge=True)
    capability = OrdinaryRecoveryCapability(
        pr_number=77,
        repository="OWNER/REPO",
        base_ref="main",
        expected_head_sha="abc123",
        released_label_event_id=101,
        released_at=100,
    )
    if board == "absent":
        snapshot = None
    elif board == "forbidden":
        snapshot = _watch_check_board("passing", protection="forbidden")
    else:
        snapshot = _watch_check_board(
            "passing",
            passing=(PullRequestCheck(name="test", kind="check_run", status=board),),
        )
    monkeypatch.setattr(
        orchestrator, "refresh_ordinary_recovery_capability", lambda *args, **kwargs: capability
    )
    monkeypatch.setattr(
        orchestrator,
        "wait_for_ordinary_recovery",
        lambda *args, **kwargs: ManagedCiOutcome(
            status="passed", checks=snapshot, head_sha="abc123"
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "get_pr_review_context",
        lambda *args, **kwargs: SimpleNamespace(metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="draft",
            head_branch="feature",
            base_branch="main",
            head_sha="abc123",
            url=None,
        )),
    )
    monkeypatch.setattr(
        orchestrator,
        "merge_pr",
        lambda *args, **kwargs: pytest.fail("non-authoritative recovery must not merge"),
    )

    ordinary_item = _carried_ci_obligations()[1]
    items = [ordinary_item]
    assert not orchestrator._finalize_ordinary_recovery_checked(
        runner,
        config=config,
        pr_number=77,
        round_number=1,
        items=items,
        current_head_sha="abc123",
        capability=capability,
    )
    assert items == [ordinary_item]
    assert not any(command[:3] == ["gh", "pr", "ready"] for command, _cwd in runner.commands)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_ordinary_recovery_refuses_head_changed_before_ready(monkeypatch, tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, auto_merge=True)
    capability = OrdinaryRecoveryCapability(
        pr_number=77, repository="OWNER/REPO", base_ref="main", expected_head_sha="abc123",
        released_label_event_id=101, released_at=100,
    )
    monkeypatch.setattr(orchestrator, "refresh_ordinary_recovery_capability", lambda *args, **kwargs: capability)
    monkeypatch.setattr(
        orchestrator,
        "wait_for_ordinary_recovery",
        lambda *args, **kwargs: ManagedCiOutcome(status="head_changed", head_sha="new-head"),
    )

    with pytest.raises(AgentLoopError, match="head_changed"):
        orchestrator._finalize_ordinary_recovery_merge(
            runner, config=config, pr_number=77, capability=capability,
        )

    assert not any(command[:3] == ["gh", "pr", "ready"] for command, _cwd in runner.commands)


def test_ordinary_recovery_refuses_provenance_change_after_ready(monkeypatch, tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, auto_merge=True)
    capability = OrdinaryRecoveryCapability(
        pr_number=77, repository="OWNER/REPO", base_ref="main", expected_head_sha="abc123",
        released_label_event_id=101, released_at=100,
    )
    validations = iter([True, False])
    monkeypatch.setattr(orchestrator, "refresh_ordinary_recovery_capability", lambda *args, **kwargs: capability)
    monkeypatch.setattr(
        orchestrator,
        "wait_for_ordinary_recovery",
        lambda *args, **kwargs: ManagedCiOutcome(
            status="passed",
            checks=_watch_check_board("passing"),
            head_sha="abc123",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "get_pr_review_context",
        lambda *args, **kwargs: SimpleNamespace(metadata=PullRequestMetadata(
            number=77, repo="OWNER/REPO", title="draft", head_branch="feature",
            base_branch="main", head_sha="abc123", url=None,
        )),
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_ordinary_recovery_capability",
        lambda *args, **kwargs: next(validations),
    )

    with pytest.raises(AgentLoopError, match="after `gh pr ready`"):
        orchestrator._finalize_ordinary_recovery_merge(
            runner, config=config, pr_number=77, capability=capability,
        )

    assert any(command[:3] == ["gh", "pr", "ready"] for command, _cwd in runner.commands)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_ordinary_recovery_merge_failure_leaves_pr_ready(monkeypatch, tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, auto_merge=True)
    capability = OrdinaryRecoveryCapability(
        pr_number=77, repository="OWNER/REPO", base_ref="main", expected_head_sha="abc123",
        released_label_event_id=101, released_at=100,
    )
    monkeypatch.setattr(orchestrator, "refresh_ordinary_recovery_capability", lambda *args, **kwargs: capability)
    monkeypatch.setattr(
        orchestrator,
        "wait_for_ordinary_recovery",
        lambda *args, **kwargs: ManagedCiOutcome(
            status="passed",
            checks=_watch_check_board("passing"),
            head_sha="abc123",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "get_pr_review_context",
        lambda *args, **kwargs: SimpleNamespace(metadata=PullRequestMetadata(
            number=77, repo="OWNER/REPO", title="draft", head_branch="feature",
            base_branch="main", head_sha="abc123", url=None,
        )),
    )
    monkeypatch.setattr(orchestrator, "validate_ordinary_recovery_capability", lambda *args, **kwargs: True)

    def fail_merge(*args, **kwargs):
        raise AgentLoopError("merge failed")

    monkeypatch.setattr(orchestrator, "merge_pr", fail_merge)

    with pytest.raises(AgentLoopError, match="merge failed"):
        orchestrator._finalize_ordinary_recovery_merge(
            runner, config=config, pr_number=77, capability=capability,
        )

    assert any(command[:3] == ["gh", "pr", "ready"] for command, _cwd in runner.commands)
    assert not any("convert-to-draft" in command for command, _cwd in runner.commands)

def test_pr_loop_does_not_post_gemini_diagnostics_without_agent_state(tmp_path):
    diagnostic = "[ERROR] Invalid stream: The model returned an empty response or malformed tool call."
    runner = FakeRunner(gemini_outputs=[diagnostic, diagnostic, diagnostic])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError, match="No review result was recorded"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert runner.comments == []
    assert not any(diagnostic in comment for comment in runner.comments)
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert sleep_commands == [["sleep", "1"], ["sleep", "1"]]

def test_pr_loop_retries_transient_gemini_diagnostic_and_posts_only_valid_response(tmp_path):
    diagnostic = "[ERROR] Invalid stream: The model returned an empty response or malformed tool call."
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[diagnostic, valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    assert diagnostic not in runner.comments[0]
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert sleep_commands == [["sleep", "1"]]

@pytest.mark.parametrize("terminator", ["", "."])
def test_pr_loop_retries_plain_agent_state_near_miss_once(tmp_path, terminator):
    near_miss = f"LGTM.\nAGENT_STATE: approved{terminator}\n-- Google Gemini"
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[near_miss, valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert sleep_commands == [["sleep", "1"]]

def test_pr_loop_exhausted_transient_retry_reports_attempt_logs(tmp_path):
    diagnostic = "[ERROR] Invalid stream: The model returned an empty response or malformed tool call."
    runner = FakeRunner(gemini_outputs=[(diagnostic, 1), (diagnostic, 1), (diagnostic, 1)])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError) as exc_info:
        run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    assert "No review result was recorded" in message
    assert "Failure category: transient" in message
    assert "Attempt logs:" in message
    assert "gemini.log" in message
    assert runner.comments == []

def test_pr_loop_retries_quota_error(tmp_path):
    quota_output = "Quota exceeded for this project."
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[(quota_output, 1), valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 1

def test_pr_loop_treats_markerless_prose_review_as_unavailable(tmp_path):
    # Issue #871: prose carrying no recoverable review payload has no verdict for
    # repair to recover, so the reviewer turn is retried within the bounded
    # policy and then fails. Repair must never mark up an approval on the
    # reviewer's behalf, so no verdict comment is ever posted.
    output = "I reviewed the PR and it looks fine."
    runner = FakeRunner(gemini_outputs=[output] * 3)
    config = make_config(
        tmp_path, reviewer="gemini", agent_retry_backoff_seconds=0
    )

    with pytest.raises(AgentLoopError, match="review_substance_integrity"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert runner.comments == []

def test_pr_loop_retries_rate_limit_429(tmp_path):
    rate_limit_output = "HTTP 429 Too Many Requests: rate limit exceeded."
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[(rate_limit_output, 1), valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 1

def test_pr_loop_retries_claude_session_limit(tmp_path):
    session_limit_output = "Error: session_limit_exceeded — too many sessions for this project."
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[(session_limit_output, 1), valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 1

def test_pr_loop_retries_gemini_no_capacity(tmp_path):
    no_capacity_output = "No capacity available for model gemini-flash on the server."
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[(no_capacity_output, 1), valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 1

def test_pr_loop_does_not_retry_billing_credit_exhaustion(tmp_path):
    output = "Quota exceeded: billing credits are exhausted."
    runner = FakeRunner(gemini_outputs=[output])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError, match="No review result was recorded"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert runner.comments == []
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)

def test_pr_loop_does_not_retry_auth_failure(tmp_path):
    output = "Unauthorized: invalid api key provided."
    runner = FakeRunner(gemini_outputs=[output])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError, match="No review result was recorded"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert runner.comments == []
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)

def test_pr_loop_failure_log_distinguishes_transient_failure(tmp_path):
    rate_limit_output = "HTTP 429: rate limit exceeded."
    runner = FakeRunner(gemini_outputs=[(rate_limit_output, 1)] * 3)
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError) as exc_info:
        run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    assert "transient" in message
    assert "rerun may succeed" in message

def test_pr_loop_failure_log_identifies_non_retryable(tmp_path):
    billing_output = "Your billing account has no credits remaining."
    runner = FakeRunner(gemini_outputs=[billing_output])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError) as exc_info:
        run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    assert "non-retryable" in message
    assert "credentials or billing" in message

def test_pr_loop_exits_immediately_on_long_reset_rate_limit(tmp_path):
    # "Retry-After: 3600" → 3600 s reset > 300 s threshold → must exit, not retry.
    rate_limit_output = "HTTP 429: rate limit exceeded. Retry-After: 3600"
    runner = FakeRunner(gemini_outputs=[(rate_limit_output, 1)])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(QuotaResetExceededError) as exc_info:
        run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    assert "quota exhausted" in message.lower()
    assert "1h" in message  # 3600 s = 1h
    assert "Rerun when quota resets" in message
    # Must not have slept / retried.
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)

def test_pr_loop_exits_immediately_on_claude_session_limit_reset(tmp_path, monkeypatch):
    class FixedDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            fixed = cls(2026, 6, 3, 5, 33, 48, tzinfo=datetime.timezone.utc)
            if tz is None:
                return fixed.replace(tzinfo=None)
            return fixed.astimezone(tz)

    monkeypatch.setattr(orchestrator.datetime, "datetime", FixedDateTime)
    session_limit_output = json.dumps(
        {
            "type": "result",
            "is_error": True,
            "api_error_status": 429,
            "result": "You've hit your session limit · resets 1:30am (America/Los_Angeles)",
        }
    )
    runner = FakeRunner(claude_outputs=[(session_limit_output, 1)])
    config = make_config(tmp_path, reviewer="claude")

    with pytest.raises(QuotaResetExceededError) as exc_info:
        run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    assert "Claude quota exhausted" in message
    assert "2h 56m" in message
    assert "Rerun when quota resets" in message
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)

def test_pr_loop_retries_on_short_reset_rate_limit(tmp_path):
    # "Retry-After: 60" → 60 s reset ≤ 300 s threshold → retry automatically.
    rate_limit_output = "HTTP 429: rate limit exceeded. Retry-After: 60"
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[(rate_limit_output, 1), valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 1

def test_pr_loop_retries_on_rate_limit_without_reset_time(tmp_path):
    # No parseable reset time → fall back to normal retry behavior.
    rate_limit_output = "HTTP 429: rate limit exceeded."
    valid = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=[(rate_limit_output, 1), valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 1

def test_pr_loop_reinjects_blocking_item_when_human_requirement_marker_missing(tmp_path):
    # Reviewer approves without HUMAN_REQUIREMENTS_RESOLVED → synthetic blocking item,
    # loop hits max_rounds (set to 1) instead of a terminal deadlock.
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "abc123",
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Please use the absolute URL.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
    )
    config = make_config(
        tmp_path,
        auto_merge=True,
        test_command=("pytest", "tests/test_agent_loop.py"),
        approved_followups="summarize",
        max_rounds=1,
    )

    # The old behaviour was a terminal deadlock; now the loop continues and hits max_rounds.
    with pytest.raises(AgentLoopError, match="blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["pytest", "tests/test_agent_loop.py"] not in commands
    assert ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge"] not in commands
    assert not any(comment.startswith("Approved-review future follow-ups") for comment in runner.comments)

def test_pr_loop_recovers_when_second_reviewer_includes_human_requirement_marker(tmp_path):
    # Round 1: reviewer approves without HUMAN_REQUIREMENTS_RESOLVED → blocking item injected.
    # Round 2: coder addresses it; reviewer approves with the marker → success.
    pr_payload = {
        "number": 77,
        "state": "OPEN",
        "url": "https://github.com/OWNER/REPO/pull/77",
        "title": "Improve review prompt context",
        "headRefName": "feature/review-context",
        "baseRefName": "main",
        "headRefOid": "abc123",
        "comments": [
            {
                "author": {"login": "maintainer"},
                "createdAt": "2026-05-18T10:00:00Z",
                "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                "body": "Please use the absolute URL.\n\n-- Human Reviewer",
            }
        ],
        "reviews": [],
    }
    runner = FakeRunner(
        claude_outputs=[
            # Round 2: coder addresses the re-injected blocking item and acknowledges human requirements
            "Addressed human requirements.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: used the absolute URL.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            # Round 1: approves but forgets the marker
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            # Round 2: resolves the synthetic blocking item and acknowledges human requirements
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload=pr_payload,
    )
    config = make_config(tmp_path, max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

def test_pr_loop_allows_approval_with_human_requirement_resolution_marker(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "abc123",
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Please use the absolute URL.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
    )
    config = make_config(tmp_path, test_command=("pytest", "tests/test_agent_loop.py"))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["pytest", "tests/test_agent_loop.py"] in commands

def test_pr_loop_accepts_structured_coder_followup_in_pr_round(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "coder_followup",
                    "state": "blocking",
                    "summary": "Added the requested regression test.",
                    "addressed_items": ["item-1"],
                    "remaining_items": [],
                    "human_requirement_dispositions": [],
                    "addressed_item_notes": {
                        "item-1": "Added the structured coder follow-up regression case."
                    },
                        "human_requirements": {
                            "addressed_ids": [],
                            "checked_discussion_directly": False,
                        },
                        "architecture_impact": {
                            "status": "unchanged",
                            "rationale": "No architectural contract changed.",
                            "affected_components": [],
                            "dependencies": [],
                            "execution_data_flows": [],
                            "persistence": [],
                            "public_contracts": [],
                            "security_boundaries": [],
                            "canonical_document_action": "no-change",
                            "canonical_document_path": None,
                            "canonical_document_rationale": "",
                        },
                        "tests_run": ["pytest tests/test_agent_loop.py -k structured_coder_followup"],
                }
            )
            + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ],
        codex_outputs=[
            "Need one more regression test before merge."
            + blocking_issues("Add the structured coder follow-up regression case.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_comments = [comment for comment in runner.comments if "## Coder follow-up" in comment]
    assert len(followup_comments) == 1
    visible_followup = _strip_round_metadata(followup_comments[0])
    assert "Added the requested regression test." in visible_followup
    assert "### Addressed items\n- item-1: Blocking issue from OpenAI Codex" in visible_followup
    assert "  - Resolution: Added the structured coder follow-up regression case." in visible_followup
    assert "### Remaining items\n- None." in visible_followup
    assert (
        "### Tests run\n- pytest tests/test_agent_loop.py -k structured_coder_followup"
        in visible_followup
    )
    assert '"kind": "coder_followup"' not in visible_followup

def test_pr_loop_rejects_malformed_structured_coder_followup_before_re_review(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "coder_followup",
                    "state": "blocking",
                    "summary": "Tried to handle the feedback.",
                    "addressed_items": ["item-9"],
                    "remaining_items": [],
                    "human_requirement_dispositions": [],
                    "human_requirements": {
                        "addressed_ids": [],
                        "checked_discussion_directly": False,
                    },
                    "architecture_impact": {
                        "status": "unchanged",
                        "rationale": "No architectural contract changed.",
                        "affected_components": [], "dependencies": [],
                        "execution_data_flows": [], "persistence": [],
                        "public_contracts": [], "security_boundaries": [],
                        "canonical_document_action": "no-change",
                        "canonical_document_path": None,
                        "canonical_document_rationale": "",
                    },
                }
            )
            + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ],
        codex_outputs=[
            "Need one more regression test before merge."
            + blocking_issues("Add the structured coder follow-up regression case.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        max_rounds=2,
        agent_max_retries=0,
    )

    with pytest.raises(
        AgentLoopError,
        match="Coder follow-up referenced unknown unresolved reviewer item IDs: item-9",
    ):
        run_pr_loop(runner, pr_number=77, config=config)

def test_reconcile_human_requirements_ack_item_does_not_mint_blocker_for_generic_output():
    human_requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )

    reconciled = _reconcile_human_requirements_ack_item(
        (),
        coder_output="Implemented fix without the extra acknowledgement.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        human_requirements=human_requirements,
        source_round=2,
    )

    assert reconciled == []


def test_reconcile_human_requirements_ack_item_retains_dedicated_failure():
    human_requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )
    invalid_structured = structured_coder_followup(
        addressed_items=[],
        human_requirement_ids=[],
        human_requirement_dispositions=[],
    )

    reconciled = _reconcile_human_requirements_ack_item(
        (),
        coder_output=invalid_structured,
        human_requirements=human_requirements,
        source_round=2,
    )

    assert [item.item_id for item in reconciled] == [HUMAN_REQUIREMENTS_ACK_ITEM_ID]
    assert "missing" in reconciled[0].text.lower()


def test_rejected_structured_response_content_cannot_trigger_provider_advice():
    malformed = structured_coder_followup(
        summary="Authentication, credit, billing, and dirty-tree vocabulary are only content.",
    )

    assert (
        orchestrator._failure_category(
            malformed,
            public_response=True,
            repair_expected_kind="coder_followup",
        )
        == "deterministic"
    )
    error = orchestrator._format_invalid_agent_response_error(
        agent_name="Anthropic Claude",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        reason="schema validation failed; repair invocation failure: timeout",
        result=None,
        log_paths=(),
        category="deterministic",
        classification_text="structured coder_followup response failed trusted validation",
    )
    assert "credentials or billing" not in error

def test_reconcile_human_requirements_ack_item_clears_markdown_ack_blocker():
    human_requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )

    reconciled = _reconcile_human_requirements_ack_item(
        (
            UnresolvedReviewItem(
                item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
                reviewer="Orchestrator",
                source_round=1,
                text="Ack missing.",
                status="blocking",
            ),
        ),
        coder_output=(
            "Implemented follow-up.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            f"- Requirement {human_requirements[0].requirement_id}: updated the URL handling.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ),
        human_requirements=human_requirements,
        source_round=2,
    )

    assert reconciled == []

def test_pr_loop_revalidates_latest_coder_output_against_refreshed_human_requirements(
    tmp_path, monkeypatch
):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented fix with the required acknowledgement.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: updated the URL handling.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Blocking issue.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "abc123",
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Please use the absolute URL.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Improve review prompt context",
        head_branch="feature/review-context",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/77",
    )
    contexts = iter(
        [
            PullRequestReviewContext(
                metadata=metadata,
                comments=(),
                human_requirements=(
                    HumanReviewRequirement(
                        source_type="PR comment",
                        author="maintainer",
                        created_at="2026-05-18T10:00:00Z",
                        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                        body="Please use the absolute URL.",
                    ),
                ),
            ),
            PullRequestReviewContext(
                metadata=metadata,
                comments=(),
                human_requirements=(),
            ),
        ]
    )

    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator.get_pr_review_context",
        lambda *args, **kwargs: next(contexts),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    review_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(review_prompts) == 2
    assert HUMAN_REQUIREMENTS_ACK_ITEM_ID not in review_prompts[1]

def test_pr_loop_routes_migration_validation_failure_through_coder_followup(tmp_path, monkeypatch):
    runner = FakeRunner(
        claude_outputs=["Fixed migration.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "LGTM again."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, test_command=("pytest", "tests/test_agent_loop.py"), max_rounds=2)
    validations = iter(
        [
            MigrationValidationResult(
                ok=False,
                message=(
                    "alembic/versions/e4f5a6b7c8d9_add_pricing.py declares `down_revision = '5d5f0e1a2b3c'`; "
                    "expected current head `402b9e8af79b`."
                ),
            ),
            MigrationValidationResult(ok=True),
        ]
    )

    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator.validate_pr_migration_topology",
        lambda *args, **kwargs: next(validations),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    coder_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(coder_prompts) == 1
    assert "Alembic migration validation unresolved blocking item [item-1]" in coder_prompts[0]
    assert "expected current head `402b9e8af79b`" in coder_prompts[0]

    commands = runner.commands
    pytest_index = command_index(commands, ["pytest", "tests/test_agent_loop.py"])
    first_review_index = [
        index for index, (cmd, _cwd) in enumerate(commands) if cmd[:2] == ["codex", "exec"]
    ][0]
    second_review_index = [
        index for index, (cmd, _cwd) in enumerate(commands) if cmd[:2] == ["codex", "exec"]
    ][1]
    assert first_review_index < pytest_index < second_review_index

def test_pr_loop_routes_failing_github_checks_through_coder_followup(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Still failing upstream."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Investigated CI.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, max_rounds=2, watch_pending_ci=True)
    check_states = iter(
        [
            {
                "check_runs": [
                    {"name": "tests/test_server.py", "status": "completed", "conclusion": "success"},
                    {
                        "name": "tests/test_security.py",
                        "status": "completed",
                        "conclusion": "failure",
                        "html_url": "https://github.com/OWNER/REPO/actions/runs/555",
                    },
                ]
            },
            {
                "check_runs": [
                    {
                        "name": "tests/test_security.py",
                        "status": "completed",
                        "conclusion": "failure",
                        "html_url": "https://github.com/OWNER/REPO/actions/runs/555",
                    }
                ]
            },
            {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]},
            {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]},
        ]
    )

    def advance_checks(*_args, **_kwargs):
        runner.pr_check_runs_payload = next(check_states)
        return original_get_pr_checks(*_args, **_kwargs)

    from coding_review_agent_loop import orchestrator as orchestrator_module

    original_get_pr_checks = orchestrator_module.get_pr_checks
    monkeypatch.setattr(orchestrator_module, "get_pr_checks", advance_checks)
    _advance_head_after_coder(monkeypatch, runner)
    monkeypatch.setattr(
        orchestrator,
        "watch_pr_checks",
        lambda *args, **kwargs: CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha=runner.pr_payload["headRefOid"], attempts_used=1
        ),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert any(
        comment.startswith("GitHub PR checks are failing for PR #77.") for comment in runner.comments
    )
    followup_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"]
        and "GitHub PR checks unresolved blocking item [item-1] from round 1:" in cmd[-1]
    )
    assert "Failing checks: tests/test_security.py (failure)" in followup_prompt
    assert "https://github.com/OWNER/REPO/actions/runs/555" in followup_prompt
    assert "Do not claim global test success unless GitHub PR checks are green." in followup_prompt


def _watch_check_board(
    state,
    *,
    passing=(),
    failing=(),
    pending=(),
    missing_required=(),
    errors=(),
    protection="configured",
):
    if state == "passing" and not passing:
        passing = (PullRequestCheck(name="test", kind="check_run", status="success"),)
    return PullRequestChecks(
        state=state,
        required_checks=("test",),
        passing=tuple(passing),
        pending=tuple(pending),
        failing=tuple(failing),
        missing_required=tuple(missing_required),
        branch_protection_status=protection,
        check_query_status="partial" if errors else "ok",
        check_query_errors=tuple(errors),
    )


@pytest.mark.parametrize("board_state", ["failing", "mixed", "pending", "missing", "unavailable", "stall"])
@pytest.mark.parametrize("auto_merge", [False, True])
def test_blocking_review_handoff_includes_available_ci_without_waiting(
    tmp_path, monkeypatch, board_state, auto_merge
):
    failure = PullRequestCheck(
        name="test", kind="check_run", status="failure",
        url="https://github.com/OWNER/REPO/actions/runs/555",
    )
    pending = PullRequestCheck(name="slow", kind="check_run", status="in_progress")
    board = _watch_check_board(
        "failing" if board_state in {"failing", "mixed", "stall"} else
        "pending" if board_state in {"pending", "missing"} else "unavailable",
        failing=(failure,) if board_state in {"failing", "mixed", "stall"} else (),
        pending=(pending,) if board_state in {"mixed", "pending"} else (),
        missing_required=("test",) if board_state == "missing" else (),
        errors=("API unavailable",) if board_state == "unavailable" else (),
    )
    if board_state == "stall":
        board = orchestrator.dataclasses_replace(board, infrastructure_stalls=(
            StalledCheck(
                name="test", kind="check_run", reason="runner_unavailable",
                check_id=None, run_id="555", url=failure.url, age_seconds=None,
            ),
        ))
    runner = FakeRunner(codex_outputs=[
        "Fix the application bug.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    ])
    snapshots = []

    def snapshot(*args, **kwargs):
        snapshots.append(kwargs["metadata"].head_sha)
        # CI finishes while the reviewer is working, not before the review.
        return _watch_check_board("pending", pending=(pending,)) if len(snapshots) == 1 else board

    monkeypatch.setattr(orchestrator, "get_pr_checks", snapshot)
    monkeypatch.setattr(orchestrator, "watch_pr_checks", lambda *a, **k: pytest.fail("must not wait"))
    original = orchestrator._run_validated_agent
    captured = {}

    class CoderReached(Exception):
        pass

    def capture(*args, **kwargs):
        if kwargs.get("role") == "coder":
            captured.update(kwargs)
            raise CoderReached
        return original(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_run_validated_agent", capture)
    with pytest.raises(CoderReached):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, auto_merge=auto_merge))
    assert snapshots == ["abc123", "abc123"]
    prompt = captured["prompt"]
    assert "Fix the application bug." in prompt
    if board_state in {"failing", "mixed"}:
        assert "Failing checks: test (failure)" in prompt
        assert failure.url in prompt
        assert "Reviewed head: abc123" in prompt
        assert "Do not wait for queued or running CI" in prompt
        assert "item-2" in captured["repair_unresolved_item_ids"]
        assert "Pending checks:" not in prompt
    else:
        assert "Failing checks:" not in prompt
        assert "item-2" not in captured["repair_unresolved_item_ids"]


@pytest.mark.parametrize("auto_merge", [False, True])
def test_round_ci_failure_is_tracked_and_resolved_with_reviewer_findings(tmp_path, monkeypatch, auto_merge):
    runner = FakeRunner(
        codex_outputs=[
            "Fix the application bug.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Both fixes verified."
            + prior_item_dispositions("[item-1] resolved", "[item-2] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Fixed the application and CI.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    failed = PullRequestCheck(name="test", kind="check_run", status="failure", url="https://example.test/555")
    snapshots = iter([
        _watch_check_board("pending"),
        _watch_check_board("failing", failing=(failed,)),
        _watch_check_board("passing"),
        _watch_check_board("passing"),
    ])
    monkeypatch.setattr(orchestrator, "get_pr_checks", lambda *a, **k: next(snapshots))
    _advance_head_after_coder(monkeypatch, runner, "repaired-head")
    monkeypatch.setattr(orchestrator, "watch_pr_checks", lambda *a, **k: CiWatchOutcome(
        status="passed", pr_checks=_watch_check_board("passing"),
        head_sha=runner.pr_payload["headRefOid"], attempts_used=1
    ))
    assert run_pr_loop(
        runner,
        pr_number=77,
        config=make_config(tmp_path, auto_merge=auto_merge, watch_pending_ci=True),
    ) == 0
    second_review = [cmd[-1] for cmd, _ in runner.commands if cmd[:1] == ["codex"]][-1]
    assert "item-2" in second_review
    assert "Failing checks: test (failure)" in second_review
    assert sum(comment.startswith("GitHub PR checks are failing") for comment in runner.comments) == 1


def test_review_only_mode_clears_repaired_ordinary_ci_from_fresh_passing_snapshot(
    tmp_path, monkeypatch
):
    failed = PullRequestCheck(
        name="test", kind="check_run", status="failure", url="https://example.test/555"
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Initial review."),
            structured_pr_review(
                state="approved",
                summary="Repaired head approved.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking", summary="Repaired the failing check.", addressed_items=["item-1"]
            )
        ],
    )

    def checks(*args, **kwargs):
        if kwargs["metadata"].head_sha == "abc123":
            return _watch_check_board("failing", failing=(failed,))
        return _watch_check_board(
            "passing",
            passing=(
                PullRequestCheck(name="test", kind="check_run", status="success"),
                PullRequestCheck(name="docs", kind="check_run", status="skipped"),
            ),
        )

    monkeypatch.setattr(orchestrator, "get_pr_checks", checks)
    _advance_head_after_coder(monkeypatch, runner, "repaired-head")
    monkeypatch.setattr(
        orchestrator,
        "watch_pr_checks",
        lambda *args, **kwargs: pytest.fail("review-only mode must not start the watcher"),
    )

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=make_config(
            tmp_path, auto_merge=False, watch_pending_ci=False, max_rounds=2
        ),
    ) == 0

    assert not any("watching GitHub checks" in comment for comment in runner.comments)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


@pytest.mark.parametrize("auto_merge", [False, True])
def test_watch_mode_success_uses_full_board_without_second_wait(
    tmp_path, monkeypatch, auto_merge
):
    runner = FakeRunner(codex_outputs=[structured_pr_review(state="approved", summary="Approved.")])
    config = make_config(tmp_path, watch_pending_ci=True, auto_merge=auto_merge)
    monkeypatch.setattr(
        orchestrator,
        "watch_pr_checks",
        lambda *args, **kwargs: CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha="abc123", attempts_used=1
        ),
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert any("watching GitHub checks" in comment for comment in runner.comments)
    merge_commands = [
        cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "pr", "merge"]
    ]
    assert bool(merge_commands) is auto_merge


@pytest.mark.parametrize("auto_merge", [False, True])
def test_skipped_only_watcher_pass_does_not_clear_or_merge(
    tmp_path, monkeypatch, auto_merge
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")]
    )
    skipped_only = PullRequestChecks(
        state="passing",
        required_checks=(),
        passing=(PullRequestCheck(name="docs", kind="check_run", status="skipped"),),
        pending=(),
        failing=(),
        missing_required=(),
        branch_protection_status="configured",
        check_query_status="ok",
    )
    monkeypatch.setattr(
        orchestrator,
        "watch_pr_checks",
        lambda *args, **kwargs: CiWatchOutcome(
            status="passed", pr_checks=skipped_only, head_sha="abc123", attempts_used=1
        ),
    )

    if auto_merge:
        with pytest.raises(AgentLoopError, match="non-authoritative"):
            run_pr_loop(
                runner,
                pr_number=77,
                config=make_config(tmp_path, watch_pending_ci=True, auto_merge=True),
            )
    else:
        assert run_pr_loop(
            runner,
            pr_number=77,
            config=make_config(tmp_path, watch_pending_ci=True, auto_merge=False),
        ) == 0

    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


@pytest.mark.parametrize("kind", ["absent", "skipped", "stale", "uncorrelated"])
def test_invalid_watcher_success_cannot_finalize_carried_ci_obligation(
    tmp_path, monkeypatch, kind
):
    ordinary_item = _carried_ci_obligations()[1]
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Approved.",
                prior_item_dispositions=[
                    {"item_id": ordinary_item.item_id, "disposition": "resolved"}
                ],
            )
        ],
        pr_payload={
            "comments": [
                {
                    "author": {"login": "coding-review-agent-loop"},
                    "body": _carried_ci_review_comment((ordinary_item,)),
                }
            ]
        },
    )
    if kind == "absent":
        outcome = CiWatchOutcome(
            status="passed", pr_checks=None, head_sha="abc123", attempts_used=1
        )
        expected = "non-authoritative"
    elif kind == "skipped":
        skipped_only = PullRequestChecks(
            state="passing",
            required_checks=(),
            passing=(PullRequestCheck(name="docs", kind="check_run", status="skipped"),),
            pending=(),
            failing=(),
            missing_required=(),
            branch_protection_status="configured",
            check_query_status="ok",
        )
        outcome = CiWatchOutcome(
            status="passed", pr_checks=skipped_only, head_sha="abc123", attempts_used=1
        )
        expected = "non-authoritative"
    elif kind == "stale":
        outcome = CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha="old-head", attempts_used=1
        )
        expected = "stale head"
    else:
        outcome = CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha=None, attempts_used=1
        )
        expected = "stale head"
    monkeypatch.setattr(orchestrator, "watch_pr_checks", lambda *args, **kwargs: outcome)

    with pytest.raises(AgentLoopError, match=expected):
        run_pr_loop(
            runner,
            pr_number=77,
            config=make_config(tmp_path, watch_pending_ci=True, auto_merge=True),
        )

    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def _carried_ci_review_comment(items):
    dispositions = tuple(
        ReviewItemDisposition(item.item_id, "Codex", "resolved") for item in items
    )
    return _attach_round_metadata(
        structured_pr_review(
            reviewer="OpenAI Codex",
            state="approved",
            summary="The reviewed repair is approved.",
            prior_item_dispositions=[
                {"item_id": item.item_id, "disposition": "resolved"}
                for item in items
            ],
        ),
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="abc123",
            prior_items=items,
            dispositions=dispositions,
            state="approved",
        ),
    )


def test_managed_success_does_not_clear_carried_ordinary_ci_obligation(
    tmp_path, monkeypatch
):
    items = _carried_ci_obligations()
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                reviewer="OpenAI Codex",
                state="approved",
                prior_item_dispositions=[
                    {"item_id": item.item_id, "disposition": "resolved"}
                    for item in items
                ],
            )
        ],
        pr_payload={
            "comments": [
                {
                    "author": {"login": "coding-review-agent-loop"},
                    "body": _carried_ci_review_comment(items),
                }
            ]
        }
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        managed_ci=True,
        auto_merge=True,
        max_rounds=1,
    )
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orchestrator,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    monkeypatch.setattr(
        orchestrator, "merge_pr", lambda *args, **kwargs: pytest.fail("ordinary obligation was bypassed")
    )

    with pytest.raises(AgentLoopError, match="github-pr-checks"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_ordinary_success_does_not_clear_carried_managed_ci_obligation(
    tmp_path, monkeypatch
):
    items = _carried_ci_obligations()
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                reviewer="OpenAI Codex",
                state="approved",
                prior_item_dispositions=[
                    {"item_id": item.item_id, "disposition": "resolved"}
                    for item in items
                ],
            )
        ],
        pr_payload={
            "comments": [
                {
                    "author": {"login": "coding-review-agent-loop"},
                    "body": _carried_ci_review_comment(items),
                }
            ]
        }
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        auto_merge=False,
        watch_pending_ci=False,
        max_rounds=1,
    )
    monkeypatch.setattr(orchestrator, "activate_managed_ci", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orchestrator, "merge_pr", lambda *args, **kwargs: pytest.fail("managed obligation was bypassed")
    )

    with pytest.raises(AgentLoopError, match="managed-exact-head-ci"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_ordinary_watcher_success_does_not_clear_carried_managed_ci_obligation(
    tmp_path, monkeypatch
):
    items = _carried_ci_obligations()
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                reviewer="OpenAI Codex",
                state="approved",
                prior_item_dispositions=[
                    {"item_id": item.item_id, "disposition": "resolved"}
                    for item in items
                ],
            )
        ],
        pr_payload={
            "comments": [
                {
                    "author": {"login": "coding-review-agent-loop"},
                    "body": _carried_ci_review_comment(items),
                }
            ]
        },
    )
    watcher_calls = []
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        auto_merge=True,
        watch_pending_ci=True,
        max_rounds=1,
    )
    monkeypatch.setattr(orchestrator, "activate_managed_ci", lambda *args, **kwargs: None)

    def watch(*args, **kwargs):
        watcher_calls.append(kwargs["metadata"].head_sha)
        return CiWatchOutcome(
            status="passed",
            pr_checks=_watch_check_board("passing"),
            head_sha="abc123",
            attempts_used=1,
        )

    monkeypatch.setattr(orchestrator, "watch_pr_checks", watch)
    monkeypatch.setattr(
        orchestrator,
        "merge_pr",
        lambda *args, **kwargs: pytest.fail("managed obligation was bypassed"),
    )

    with pytest.raises(AgentLoopError, match="managed-exact-head-ci"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert watcher_calls == ["abc123"]


def test_resume_machine_authority_upgrade_forces_full_reviewer_board(
    tmp_path, monkeypatch
):
    legacy = UnresolvedReviewItem(
        item_id="item-30",
        reviewer="GitHub managed exact-head CI",
        source_round=1,
        text="Managed exact-head CI failed.",
        status="blocking",
        source_status="blocking",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity="managed-exact-head-ci:item-30",
        lifecycle="repair_required",
        failed_head_sha="abc123",
        candidate_head_sha=None,
        base_branch="main",
        allowed_rounds=1,
    )
    scheduler_contract = orchestrator.make_contract(
        ("Codex",), "selective-intermediate", None
    )
    summary = _attach_round_metadata(
        "Persisted legacy machine obligation.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=1,
            subject="abc123",
            prior_items=(legacy,),
            state="blocking",
            phase="qualification-checkpoint",
            qualification_checkpoint=checkpoint,
            scheduler_contract=scheduler_contract.as_dict(),
            scheduler_previous_sha="abc123",
            scheduler_current_sha="abc123",
            scheduler_obligation_digest=orchestrator._qualification_digest(
                orchestrator._prior_item_ledger_signature((legacy,))
            ),
            scheduler_selected_reviewers=("Codex",),
            scheduler_paused_reviewers=(),
            scheduler_reasons=("legacy checkpoint",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=0,
        ),
    )
    runner = FakeRunner(
        pr_payload={
            "comments": [
                {"author": {"login": "coding-review-agent-loop"}, "body": summary}
            ]
        }
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        max_rounds=1,
    )
    force_full_values = []
    original_select_reviewers = orchestrator.select_reviewers

    def capture_force_full(snapshot, *args, **kwargs):
        force_full_values.append(snapshot.force_full)
        return original_select_reviewers(snapshot, *args, **kwargs)

    monkeypatch.setattr(orchestrator, "select_reviewers", capture_force_full)

    with pytest.raises(AgentLoopError, match="awaiting a new repair head"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert force_full_values == [True]


def test_resume_reviewer_ledger_digest_mismatch_forces_full_reviewer_board(
    tmp_path, monkeypatch
):
    reviewer_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Codex",
        source_round=1,
        text="The worker still needs cleanup.",
        status="blocking",
        source_status="blocking",
        fix_scope=("src/worker.py",),
    )
    scheduler_contract = orchestrator.make_contract(
        ("Codex",), "selective-intermediate", None
    )
    summary = _attach_round_metadata(
        structured_coder_followup(
            summary="Persisted reviewer ledger with a stale scheduler digest."
        ),
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="abc123",
            prior_items=(reviewer_item,),
            state="blocking",
            scheduler_contract=scheduler_contract.as_dict(),
            scheduler_previous_sha="abc123",
            scheduler_current_sha="abc123",
            scheduler_obligation_digest=orchestrator._qualification_digest(()),
            scheduler_selected_reviewers=("Codex",),
            scheduler_paused_reviewers=(),
            scheduler_reasons=("stale digest",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=0,
        ),
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="The reviewer item remains blocking.",
                prior_item_dispositions=[
                    {
                        "item_id": reviewer_item.item_id,
                        "disposition": "blocking",
                        "note": "The worker cleanup remains incomplete on this head.",
                    }
                ],
            )
        ],
        pr_payload={
            "comments": [
                {"author": {"login": "coding-review-agent-loop"}, "body": summary}
            ]
        }
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        max_rounds=1,
    )
    force_full_values = []
    original_select_reviewers = orchestrator.select_reviewers

    def capture_force_full(snapshot, *args, **kwargs):
        force_full_values.append(snapshot.force_full)
        return original_select_reviewers(snapshot, *args, **kwargs)

    monkeypatch.setattr(orchestrator, "select_reviewers", capture_force_full)

    with pytest.raises(AgentLoopError, match="still reported blocking"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert force_full_values == [True]


def test_post_review_ordinary_recovery_rejects_other_machine_obligation(
    tmp_path, monkeypatch
):
    reviewer_item = UnresolvedReviewItem(
        item_id="item-reviewer",
        reviewer="Codex",
        source_round=1,
        text="The reviewer-owned cleanup is incomplete.",
        status="blocking",
        source_status="blocking",
        fix_scope=("src/worker.py",),
    )
    managed_item = UnresolvedReviewItem(
        item_id="item-managed",
        reviewer="GitHub managed exact-head CI",
        source_round=1,
        text="Managed exact-head CI failed on the previous head.",
        status="blocking",
        source_status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="qualification_ready",
        failed_head_sha="old-head",
        candidate_head_sha="abc123",
        obligation_identity="managed-exact-head-ci:item-managed",
    )
    carried = (reviewer_item, managed_item)
    reviewer_comment = _attach_round_metadata(
        structured_pr_review(
            reviewer="OpenAI Codex",
            state="approved",
            summary="The reviewer-owned item is resolved.",
            prior_item_dispositions=[
                {"item_id": item.item_id, "disposition": "resolved"}
                for item in carried
            ],
        ),
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="abc123",
            prior_items=carried,
            dispositions=tuple(
                ReviewItemDisposition(item.item_id, "Codex", "resolved")
                for item in carried
            ),
            state="approved",
        ),
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                reviewer="OpenAI Codex",
                state="approved",
                summary="The reviewer-owned item is resolved.",
                prior_item_dispositions=[
                    {"item_id": item.item_id, "disposition": "resolved"}
                    for item in carried
                ],
            )
        ],
        pr_payload={
            "comments": [
                {"author": {"login": "coding-review-agent-loop"}, "body": reviewer_comment}
            ]
        },
    )
    capability = OrdinaryRecoveryCapability(
        pr_number=77,
        repository="OWNER/REPO",
        base_ref="main",
        expected_head_sha="abc123",
        released_label_event_id=101,
        released_at=100,
    )
    activation = ManagedCiContract(
        activation_path="ordinary_fallback",
        ordinary_recovery=capability,
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        auto_merge=True,
        max_rounds=2,
    )
    monkeypatch.setattr(orchestrator, "activate_managed_ci", lambda *args, **kwargs: activation)
    monkeypatch.setattr(
        orchestrator,
        "_finalize_ordinary_recovery_merge",
        lambda *args, **kwargs: pytest.fail("ordinary recovery bypassed another machine obligation"),
    )
    guard_calls = []
    original_guard = orchestrator._ensure_finalization_ready

    def capture_guard(*args, **kwargs):
        guard_calls.append(
            {
                "ignored_machine_kinds": kwargs.get("ignored_machine_kinds"),
                "items": tuple(kwargs.get("items", ())),
            }
        )
        return original_guard(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_ensure_finalization_ready", capture_guard)

    with pytest.raises(AgentLoopError, match="managed-exact-head-ci"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)
    assert len(guard_calls) == 1
    assert guard_calls[0]["ignored_machine_kinds"] == frozenset({"github-pr-checks"})
    assert [item.obligation_kind for item in guard_calls[0]["items"]] == [
        "managed-exact-head-ci"
    ]


def test_auto_merge_supported_repo_dispatches_and_merges_exact_approved_head(
    tmp_path, monkeypatch
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")]
    )
    config = make_config(tmp_path, watch_pending_ci=True, auto_merge=True)
    monkeypatch.setattr(
        orchestrator,
        "activate_managed_ci",
        lambda *args, **kwargs: ManagedCiContract(),
    )
    dispatches = []
    monkeypatch.setattr(
        orchestrator,
        "dispatch_final_qualification",
        lambda *args, **kwargs: dispatches.append(kwargs),
    )
    monkeypatch.setattr(
        orchestrator,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    merges = []
    monkeypatch.setattr(
        orchestrator,
        "merge_pr",
        lambda *args, **kwargs: merges.append(kwargs),
    )
    monkeypatch.setattr(
        orchestrator,
        "watch_pr_checks",
        lambda *args, **kwargs: pytest.fail("managed CI must bypass the ordinary watcher"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert dispatches[0]["expected_head_sha"] == "abc123"
    assert merges == [{"expected_head_sha": "abc123"}]


def test_managed_ci_failure_routes_back_to_coder_and_uses_failure_extension(
    tmp_path, monkeypatch
):
    failed_check = PullRequestCheck(
        name="final-ci/exact-head",
        kind="check_run",
        status="failure",
        url="https://github.com/OWNER/REPO/actions/runs/555",
    )
    failed_checks = _watch_check_board("failing", failing=(failed_check,))
    outcomes = iter(
        [
            ManagedCiOutcome(status="failed", checks=failed_checks, head_sha="abc123"),
            ManagedCiOutcome(status="passed", head_sha="abc123-coder-1"),
        ]
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved."),
            structured_pr_review(
                state="approved",
                summary="Approved after managed CI fix.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Fixed managed CI.",
                addressed_items=["item-1"],
            )
        ],
    )
    config = make_config(tmp_path, auto_merge=True, max_rounds=1)
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orchestrator, "wait_for_final_qualification", lambda *args, **kwargs: next(outcomes)
    )
    merges = []
    monkeypatch.setattr(orchestrator, "merge_pr", lambda *args, **kwargs: merges.append(kwargs))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "GitHub managed exact-head CI unresolved blocking item [item-1]" in coder_prompt
    assert "final-ci/exact-head (failure)" in coder_prompt
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 2
    assert merges == [{"expected_head_sha": "abc123-coder-1"}]


def test_managed_qualification_resume_attaches_without_redispatch(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path, managed_ci=True, auto_merge=True, max_rounds=1)
    invocation = orchestrator.resolve_invocation(
        config, provider="codex", role="reviewer"
    )
    acquisition_contract = {
        "Codex": (invocation.configured_model, invocation.resolved_effort, "codex")
    }
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="GitHub managed exact-head CI",
        source_round=1,
        text="Managed CI failed on the previous head.",
        status="blocking",
        source_status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="qualifying",
        failed_head_sha="old-head",
        candidate_head_sha="abc123",
        obligation_identity="managed-exact-head-ci:item-1",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity=item.obligation_identity,
        lifecycle="qualifying",
        failed_head_sha="old-head",
        candidate_head_sha="abc123",
        base_branch="main",
        requirements_digest=orchestrator._qualification_digest(tuple()),
        acquisition_digest=orchestrator._qualification_digest(acquisition_contract),
        approval_digest=orchestrator._qualification_digest(("Codex",)),
        scheduler_digest=orchestrator._qualification_digest(
            orchestrator._prior_item_ledger_signature((item,))
        ),
        qualification_attempt_id="123/1",
        allowed_rounds=1,
    )
    reviewer_comment = _attach_round_metadata(
        "Approved the repaired head.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="abc123",
            prior_items=(dataclasses.replace(item, lifecycle="awaiting_current_head_review"),),
            state="approved",
            architecture_contract_version=1,
        ),
    )
    checkpoint_comment = _attach_round_metadata(
        "Qualification was dispatched.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=1,
            subject="abc123",
            prior_items=(item,),
            state="blocking",
            phase="qualification-checkpoint",
            qualification_checkpoint=checkpoint,
            architecture_contract_version=1,
        ),
    )
    runner = FakeRunner(
        codex_outputs=[],
        pr_payload={
            "comments": [
                {"author": {"login": "coding-review-agent-loop"}, "body": reviewer_comment},
                {"author": {"login": "coding-review-agent-loop"}, "body": checkpoint_comment},
            ]
        },
    )
    contract = ManagedCiContract(
        protocol_version=2, attached_run_id=123, run_attempt=1
    )
    dispatches = []
    merges = []
    monkeypatch.setattr(orchestrator, "activate_managed_ci", lambda *a, **k: contract)
    monkeypatch.setattr(orchestrator, "revalidate_adopted_managed_ci", lambda *a, **k: True)
    monkeypatch.setattr(
        orchestrator,
        "dispatch_final_qualification",
        lambda *a, **k: dispatches.append(k),
    )
    monkeypatch.setattr(
        orchestrator,
        "wait_for_final_qualification",
        lambda *a, **k: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    monkeypatch.setattr(orchestrator, "merge_pr", lambda *a, **k: merges.append(k))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert dispatches == []
    assert merges == [{"expected_head_sha": "abc123"}]
    assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approval_digest", None),
        ("requirements_digest", None),
        ("acquisition_digest", None),
        ("scheduler_digest", None),
        ("qualification_attempt_id", None),
        ("approval_digest", "stale-approval"),
        ("requirements_digest", "stale-requirements"),
        ("acquisition_digest", "stale-acquisition"),
        ("scheduler_digest", "stale-scheduler"),
        ("qualification_attempt_id", "stale-attempt"),
    ],
)
def test_resume_qualification_identity_gap_forces_review_before_qualification(
    tmp_path, monkeypatch, field, value
):
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        managed_ci=False,
        auto_merge=True,
        max_rounds=1,
    )
    invocation = orchestrator.resolve_invocation(
        config, provider="codex", role="reviewer"
    )
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="GitHub PR checks",
        source_round=1,
        text="GitHub PR checks failed on the previous head.",
        status="blocking",
        source_status="blocking",
        authority="machine",
        obligation_kind="github-pr-checks",
        lifecycle="qualification_ready",
        failed_head_sha="old-head",
        candidate_head_sha="abc123",
        obligation_identity="github-pr-checks:item-1",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="github-pr-checks",
        obligation_identity=item.obligation_identity,
        lifecycle="qualifying",
        failed_head_sha="old-head",
        candidate_head_sha="abc123",
        base_branch="main",
        requirements_digest=orchestrator._qualification_digest(tuple()),
        acquisition_digest=orchestrator._qualification_digest(
            {"Codex": (invocation.configured_model, invocation.resolved_effort, "codex")}
        ),
        approval_digest=orchestrator._qualification_digest(("Codex",)),
        scheduler_digest=orchestrator._qualification_digest(
            orchestrator._prior_item_ledger_signature((item,))
        ),
        qualification_attempt_id="github-checks:abc123",
        allowed_rounds=1,
    )
    checkpoint = dataclasses.replace(checkpoint, **{field: value})
    summary = _attach_round_metadata(
        "Persisted qualifying checkpoint without reviewer records.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=1,
            subject="abc123",
            prior_items=(item,),
            state="blocking",
            phase="qualification-checkpoint",
            qualification_checkpoint=checkpoint,
        ),
    )
    runner = FakeRunner(
        pr_payload={
            "comments": [
                {"author": {"login": "coding-review-agent-loop"}, "body": summary}
            ]
        }
    )
    force_full_values = []
    reviewer_calls = []
    original_select_reviewers = orchestrator.select_reviewers
    original_run_validated_agent = orchestrator._run_validated_agent

    def capture_force_full(snapshot, *args, **kwargs):
        force_full_values.append(snapshot.force_full)
        return original_select_reviewers(snapshot, *args, **kwargs)

    def stop_at_reviewer(*args, **kwargs):
        if kwargs.get("role") == "reviewer":
            reviewer_calls.append(kwargs)
            raise AgentLoopError("review board invoked before qualification")
        return original_run_validated_agent(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "select_reviewers", capture_force_full)
    monkeypatch.setattr(orchestrator, "_run_validated_agent", stop_at_reviewer)
    dispatches = []
    monkeypatch.setattr(
        orchestrator,
        "dispatch_final_qualification",
        lambda *args, **kwargs: dispatches.append(kwargs),
    )

    with pytest.raises(AgentLoopError, match="review board invoked"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert reviewer_calls
    assert force_full_values == [True]
    assert dispatches == []
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_managed_manual_success_publishes_result_without_merge_even_with_pending_intermediate_ci(
    tmp_path, monkeypatch, capsys
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
        pr_check_runs_payload={
            "check_runs": [
                {"name": "test", "status": "completed", "conclusion": "success"},
                {"name": "lint", "status": "in_progress"},
            ]
        },
        pr_status_payload={"state": "pending", "statuses": []},
    )
    config = make_config(tmp_path, managed_ci=True, max_rounds=1)
    contract = ManagedCiContract(protocol_version=2, protection_mode="strict")
    published = []

    monkeypatch.setattr(orchestrator, "activate_managed_ci", lambda *args, **kwargs: contract)
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orchestrator,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    monkeypatch.setattr(
        orchestrator,
        "publish_manual_v2_qualification",
        lambda *args, **kwargs: published.append(kwargs["expected_head_sha"]) or "abc123",
    )
    monkeypatch.setattr(
        orchestrator,
        "merge_pr",
        lambda *args, **kwargs: pytest.fail("manual managed-CI success must not merge"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert published == ["abc123"]
    assert "approved and qualified; manual merge required" in capsys.readouterr().out
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_managed_manual_issue_created_label_loss_cancels_qualification(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
    )
    config = make_config(tmp_path, managed_ci=True, max_rounds=1)
    contract = ManagedCiContract(protocol_version=2, issue_created_pr=True)
    monkeypatch.setattr(orchestrator, "activate_managed_ci", lambda *args, **kwargs: contract)

    with pytest.raises(AgentLoopError, match="lost its authenticated"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_managed_manual_adopted_provenance_loss_cancels_qualification(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_adopt_existing_pr=True,
        managed_ci_trusted_actor="agent-loop",
        max_rounds=1,
    )
    contract = ManagedCiContract(
        protocol_version=2,
        base_ref="main",
        adopted_existing_pr=True,
    )
    monkeypatch.setattr(orchestrator, "activate_managed_ci", lambda *args, **kwargs: contract)
    monkeypatch.setattr(orchestrator, "revalidate_adopted_managed_ci", lambda *args, **kwargs: False)

    with pytest.raises(AgentLoopError, match="adoption provenance changed"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_managed_ci_head_change_restarts_review_without_coder(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Old head approved."),
            structured_pr_review(state="approved", summary="New head approved."),
        ]
    )
    config = make_config(tmp_path, auto_merge=True, max_rounds=1)
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *args, **kwargs: None)
    call_count = 0

    def wait_for_managed_ci(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            runner.pr_payload["headRefOid"] = "new-head"
            return ManagedCiOutcome(status="head_changed", head_sha="new-head")
        return ManagedCiOutcome(status="passed", head_sha="new-head")

    monkeypatch.setattr(orchestrator, "wait_for_final_qualification", wait_for_managed_ci)
    merges = []
    monkeypatch.setattr(orchestrator, "merge_pr", lambda *args, **kwargs: merges.append(kwargs))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert call_count == 2
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 2
    assert merges == [{"expected_head_sha": "new-head"}]


def test_managed_ci_merge_conflict_routes_through_reconciliation(tmp_path, monkeypatch):
    conflict = PullRequestMergeability(
        state="conflicted",
        mergeable_raw="CONFLICTING",
        merge_state_raw="DIRTY",
        head_sha="abc123",
        base_branch="main",
    )
    outcomes = iter(
        [
            ManagedCiOutcome(
                status="merge_conflict",
                mergeability=conflict,
                head_sha="abc123",
            ),
            ManagedCiOutcome(status="passed", head_sha="abc123-coder-1"),
        ]
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved before conflict."),
            structured_pr_review(state="approved", summary="Approved after rebase."),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking", summary="Rebased and resolved the conflict."
            )
        ],
    )
    config = make_config(tmp_path, auto_merge=True, max_rounds=2)
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orchestrator, "wait_for_final_qualification", lambda *args, **kwargs: next(outcomes)
    )
    monkeypatch.setattr(orchestrator, "merge_pr", lambda *args, **kwargs: None)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "merge conflict with its base branch `main`" in coder_prompt
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 2


def test_managed_ci_infrastructure_stall_stops_cleanly(tmp_path, monkeypatch):
    stalled_check = StalledCheck(
        name="final-ci/exact-head",
        kind="check_run",
        reason="queued_too_long",
        check_id=99,
        run_id="555",
        url="https://github.com/OWNER/REPO/actions/runs/555",
        age_seconds=1800,
    )
    stall = CiInfrastructureStall(checks=(stalled_check,))
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")]
    )
    config = make_config(tmp_path, auto_merge=True, max_rounds=1)
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orchestrator,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(
            status="infrastructure_stall", head_sha="abc123", stall=stall
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "merge_pr",
        lambda *args, **kwargs: pytest.fail("infrastructure stall must not merge"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert any(
        comment.startswith("External GitHub Actions infrastructure is blocking PR #77.")
        for comment in runner.comments
    )
    assert runner.comments[-1].startswith(
        "PR #77 is blocked on external GitHub Actions infrastructure, not a code defect."
    )


@pytest.mark.parametrize(
    ("pre_review_tests", "test_command"),
    [(False, ("python", "-m", "pytest", "focused")), (True, None)],
)
def test_managed_ci_does_not_publish_readiness_without_configured_pre_review_gate(
    tmp_path, monkeypatch, pre_review_tests, test_command
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")]
    )
    config = make_config(
        tmp_path,
        auto_merge=True,
        pre_review_tests=pre_review_tests,
        test_command=test_command,
    )
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(
        orchestrator,
        "publish_round_readiness",
        lambda *args, **kwargs: pytest.fail("readiness must not be published"),
    )
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orchestrator,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    monkeypatch.setattr(orchestrator, "merge_pr", lambda *args, **kwargs: None)

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=config,
        pre_review_test_pending=True,
    ) == 0


def test_managed_ci_publishes_readiness_after_configured_pre_review_gate(
    tmp_path, monkeypatch
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")]
    )
    config = make_config(
        tmp_path,
        auto_merge=True,
        pre_review_tests=True,
        test_command=("verify-managed-head",),
    )
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    published_heads = []
    monkeypatch.setattr(
        orchestrator,
        "publish_round_readiness",
        lambda *args, **kwargs: published_heads.append(kwargs["head_sha"]),
    )
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orchestrator,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    monkeypatch.setattr(orchestrator, "merge_pr", lambda *args, **kwargs: None)

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=config,
        pre_review_test_pending=True,
    ) == 0

    assert published_heads == ["abc123"]
    assert ["verify-managed-head"] in [cmd for cmd, _cwd in runner.commands]


def test_watch_failure_on_final_round_dispatches_coder_with_check_diagnostic(
    tmp_path, monkeypatch
):
    failure_url = "https://github.com/OWNER/REPO/actions/runs/555"
    failed_check = PullRequestCheck(
        name="test",
        kind="check_run",
        status="failure",
        url=failure_url,
    )
    outcomes = iter(
        [
            CiWatchOutcome(
                status="failed",
                pr_checks=_watch_check_board("failing", failing=(failed_check,)),
                failed_checks=(failed_check,),
                attempts_used=1,
            ),
            CiWatchOutcome(
                status="passed", pr_checks=_watch_check_board("passing"),
                head_sha="repaired-head", attempts_used=1
            ),
        ]
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved."),
            structured_pr_review(
                state="approved",
                summary="Approved after CI fix.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Fixed the CI failure.",
                addressed_items=["item-1"],
            )
        ],
    )
    config = make_config(tmp_path, watch_pending_ci=True, max_rounds=1)
    _advance_head_after_coder(monkeypatch, runner, "repaired-head")
    monkeypatch.setattr(
        orchestrator, "watch_pr_checks", lambda *args, **kwargs: next(outcomes)
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    coder_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(coder_prompts) == 1
    assert "GitHub PR checks unresolved blocking item [item-1] from round 1" in coder_prompts[0]
    assert "Failing checks: test (failure)" in coder_prompts[0]
    assert failure_url in coder_prompts[0]
    assert any(
        comment.startswith("GitHub PR checks are failing for PR #77.")
        and failure_url in comment
        for comment in runner.comments
    )
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 2


def test_watch_head_change_re_reviews_without_coder_and_preserves_budget(
    tmp_path, monkeypatch
):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Old head approved."),
            structured_pr_review(state="approved", summary="New head approved."),
        ]
    )
    config = make_config(
        tmp_path,
        watch_pending_ci=True,
        max_rounds=1,
        ci_timeout_seconds=60,
        ci_poll_interval_seconds=10,
    )
    watch_calls = []

    def watch(*args, **kwargs):
        watch_calls.append((kwargs["deadline"], kwargs["attempts"]))
        if len(watch_calls) == 1:
            runner.pr_payload["headRefOid"] = "new-head"
            return CiWatchOutcome(
                status="head_changed", head_sha="new-head", attempts_used=2
            )
        return CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha="new-head", attempts_used=1
        )

    monkeypatch.setattr(orchestrator, "watch_pr_checks", watch)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 2
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert watch_calls[0][0] == watch_calls[1][0]
    assert [attempts for _deadline, attempts in watch_calls] == [6, 4]


def test_watch_combined_failure_and_head_change_can_use_both_extensions(
    tmp_path, monkeypatch
):
    failed_check = PullRequestCheck(
        name="test", kind="check_run", status="failure", url="https://example.test/run"
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Round one approved."),
            structured_pr_review(
                state="approved",
                summary="Round two approved.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
            structured_pr_review(
                state="approved",
                summary="Round three approved.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Fixed CI.",
                addressed_items=["item-1"],
            )
        ],
    )
    config = make_config(tmp_path, watch_pending_ci=True, max_rounds=1)
    _advance_head_after_coder(monkeypatch, runner, "repaired-head")
    call_count = 0

    def watch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return CiWatchOutcome(
                status="failed",
                pr_checks=_watch_check_board("failing", failing=(failed_check,)),
                failed_checks=(failed_check,),
                attempts_used=1,
            )
        if call_count == 2:
            runner.pr_payload["headRefOid"] = "newer-head"
            return CiWatchOutcome(
                status="head_changed", head_sha="newer-head", attempts_used=1
            )
        return CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha="newer-head", attempts_used=1
        )

    monkeypatch.setattr(orchestrator, "watch_pr_checks", watch)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert call_count == 3
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 3
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]) == 1


def test_watch_combined_head_change_and_failure_can_use_both_extensions(
    tmp_path, monkeypatch
):
    failed_check = PullRequestCheck(
        name="test", kind="check_run", status="failure", url="https://example.test/run"
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Round one approved."),
            structured_pr_review(state="approved", summary="Round two approved."),
            structured_pr_review(
                state="approved",
                summary="Round three approved.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Fixed CI.",
                addressed_items=["item-1"],
            )
        ],
    )
    config = make_config(tmp_path, watch_pending_ci=True, max_rounds=1)
    _advance_head_after_coder(monkeypatch, runner, "repaired-head")
    call_count = 0

    def watch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            runner.pr_payload["headRefOid"] = "newer-head"
            return CiWatchOutcome(
                status="head_changed", head_sha="newer-head", attempts_used=1
            )
        if call_count == 2:
            return CiWatchOutcome(
                status="failed",
                pr_checks=_watch_check_board("failing", failing=(failed_check,)),
                failed_checks=(failed_check,),
                attempts_used=1,
            )
        return CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha="repaired-head", attempts_used=1
        )

    monkeypatch.setattr(orchestrator, "watch_pr_checks", watch)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert call_count == 3
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 3
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]) == 1


@pytest.mark.parametrize(
    ("invocation_argv", "expected_rerun", "expected_note"),
    [
        (
            ("agent-loop", "pr", "77", "--claude-arg", "token value"),
            "agent-loop pr 77 --claude-arg 'token value'",
            "",
        ),
        (
            (),
            "agent-loop pr 77 --watch-pending-ci",
            "deterministic fallback; original invocation unavailable",
        ),
    ],
)
def test_watch_timeout_renders_local_rerun_without_leaking_it_to_comment(
    tmp_path, monkeypatch, capsys, invocation_argv, expected_rerun, expected_note
):
    pending_check = PullRequestCheck(
        name="test", kind="check_run", status="in_progress"
    )
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")]
    )
    config = make_config(
        tmp_path,
        watch_pending_ci=True,
        invocation_argv=invocation_argv,
    )
    monkeypatch.setattr(
        orchestrator,
        "watch_pr_checks",
        lambda *args, **kwargs: CiWatchOutcome(
            status="timeout",
            pr_checks=_watch_check_board(
                "pending",
                pending=(pending_check,),
                errors=("transient check-runs API failure",),
            ),
            attempts_used=1,
        ),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    output = capsys.readouterr().out
    assert expected_rerun in output
    assert expected_note in output
    assert "Pending checks: test (in_progress)" in output
    assert all(expected_rerun not in comment for comment in runner.comments)
    assert all("token value" not in comment for comment in runner.comments)


@pytest.mark.parametrize("auto_merge", [False, True])
def test_watch_budget_exhaustion_stops_before_announcing_new_poll(
    tmp_path, monkeypatch, capsys, auto_merge
):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Old head approved."),
            structured_pr_review(state="approved", summary="New head approved."),
        ]
    )
    config = make_config(
        tmp_path,
        watch_pending_ci=True,
        auto_merge=auto_merge,
        max_rounds=1,
        ci_timeout_seconds=20,
        ci_poll_interval_seconds=10,
    )
    watch_calls = []

    def watch(*args, **kwargs):
        watch_calls.append(kwargs)
        runner.pr_payload["headRefOid"] = "new-head"
        return CiWatchOutcome(status="head_changed", head_sha="new-head", attempts_used=2)

    monkeypatch.setattr(orchestrator, "watch_pr_checks", watch)

    if auto_merge:
        with pytest.raises(AgentLoopError, match="watch budget was exhausted"):
            run_pr_loop(runner, pr_number=77, config=config)
    else:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(watch_calls) == 1
    assert sum("watching GitHub checks" in comment for comment in runner.comments) == 1
    assert sum("CI watch budget was exhausted" in line for line in capsys.readouterr().out.splitlines()) == 1


def test_watch_dry_run_previews_without_poll_sleep_coder_or_merge(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")]
    )
    config = make_config(
        tmp_path,
        watch_pending_ci=True,
        auto_merge=True,
        dry_run=True,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert not any(cmd[:1] == ["sleep"] for cmd in commands)
    assert not any(cmd[:1] == ["claude"] for cmd in commands)
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd in commands)
    assert "dry-run preview did not perform live CI watching" in capsys.readouterr().out


@pytest.mark.parametrize("auto_merge", [False, True])
def test_disabled_watch_mode_preserves_manual_path_and_auto_merge_uses_full_board(
    tmp_path, monkeypatch, auto_merge
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
        pr_check_runs_payload={
            "check_runs": [
                {"name": "test", "status": "in_progress", "conclusion": None}
            ]
        },
    )
    config = make_config(tmp_path, watch_pending_ci=False, auto_merge=auto_merge)
    watch_calls = []

    def watch(*args, **kwargs):
        watch_calls.append((args, kwargs))
        return CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha="abc123", attempts_used=1
        )

    monkeypatch.setattr(orchestrator, "watch_pr_checks", watch)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert bool(watch_calls) is auto_merge
    merge_commands = [
        cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "pr", "merge"]
    ]
    assert bool(merge_commands) is auto_merge
    if not auto_merge:
        assert any("checks are still pending" in comment for comment in runner.comments)


def test_auto_merge_green_full_board_merges_once(
    tmp_path, monkeypatch
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
        pr_check_runs_payload={
            "check_runs": [
                {"name": "lint", "status": "completed", "conclusion": "success"}
            ]
        },
        pr_status_payload={"state": "success", "statuses": []},
        pr_branch_protection_payload={"contexts": []},
    )
    config = make_config(
        tmp_path,
        auto_merge=True,
        watch_pending_ci=False,
    )
    with patch.object(orchestrator, "watch_pr_checks", wraps=orchestrator.watch_pr_checks) as watch_spy:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert watch_spy.call_count == 1
    merge_commands = [
        cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "pr", "merge"]
    ]
    assert merge_commands == [[
        "gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge",
        "--match-head-commit", "abc123",
    ]]


def test_watch_publishes_approved_followups_only_after_terminal_success(
    tmp_path, monkeypatch
):
    failed_check = PullRequestCheck(
        name="test", kind="check_run", status="failure", url="https://example.test/run"
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Approved with follow-up.",
                future_followups=["Document the optional tuning knob."],
            ),
            structured_pr_review(
                state="approved",
                summary="Approved after CI fix.",
                future_followups=["Document the optional tuning knob."],
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "future"},
                    {"item_id": "item-2", "disposition": "resolved"}
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Fixed CI.",
                addressed_items=["item-1", "item-2"],
            )
        ],
    )
    config = make_config(
        tmp_path,
        watch_pending_ci=True,
        max_rounds=1,
        approved_followups="summarize",
    )
    _advance_head_after_coder(monkeypatch, runner, "repaired-head")
    call_count = 0

    def watch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        assert not any(
            comment.startswith("Approved-review future follow-ups")
            for comment in runner.comments
        )
        if call_count == 1:
            return CiWatchOutcome(
                status="failed",
                pr_checks=_watch_check_board("failing", failing=(failed_check,)),
                failed_checks=(failed_check,),
                attempts_used=1,
            )
        return CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha="repaired-head", attempts_used=1
        )

    monkeypatch.setattr(orchestrator, "watch_pr_checks", watch)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    followup_comments = [
        comment
        for comment in runner.comments
        if comment.startswith("Approved-review future follow-ups")
    ]
    assert len(followup_comments) == 1
    assert "Document the optional tuning knob." in followup_comments[0]


def test_pr_loop_refreshes_checks_between_reviewers_and_before_coder(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Codex approves."),
            structured_pr_review(
                state="approved",
                summary="Codex approves after the fix.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[
            structured_pr_review(state="approved", summary="Gemini approves."),
            structured_pr_review(
                state="approved",
                summary="Gemini approves after the fix.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="approved", summary="Coder addressed the failure.", addressed_items=["item-1"]
            )
        ],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), max_rounds=2, watch_pending_ci=True
    )
    _advance_head_after_coder(monkeypatch, runner, "repaired-head")
    failure_url = "https://github.com/OWNER/REPO/actions/runs/555"
    check_states = iter(
        [
            {"check_runs": [{"name": "test", "status": "in_progress", "conclusion": None}]},
            {
                "check_runs": [
                    {
                        "name": "test",
                        "status": "completed",
                        "conclusion": "failure",
                        "html_url": failure_url,
                    }
                ]
            },
            {
                "check_runs": [
                    {
                        "name": "test",
                        "status": "completed",
                        "conclusion": "failure",
                        "html_url": failure_url,
                    }
                ]
            },
            {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]},
            {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]},
            {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]},
        ]
    )

    def next_checks(*args, **kwargs):
        runner.pr_check_runs_payload = next(check_states)
        return original_get_pr_checks(*args, **kwargs)

    from coding_review_agent_loop import orchestrator as orchestrator_module

    original_get_pr_checks = orchestrator_module.get_pr_checks
    monkeypatch.setattr(orchestrator_module, "get_pr_checks", next_checks)
    monkeypatch.setattr(
        orchestrator,
        "watch_pr_checks",
        lambda *args, **kwargs: CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha=runner.pr_payload["headRefOid"], attempts_used=1
        ),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    review_prompts = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] or cmd[:1] == ["gemini"]
    ]
    assert "- Overall state: pending" in review_prompts[0]
    assert "- Overall state: failing" in review_prompts[1]
    assert f"- Failing checks: test (failure) — {failure_url}" in review_prompts[1]
    assert any(
        comment.startswith("GitHub PR checks are failing for PR #77.")
        and failure_url in comment
        for comment in runner.comments
    )
    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert f"Failing checks: test (failure) — {failure_url}" in coder_prompt
    assert "Overall state: pending" not in coder_prompt


def test_pr_loop_refreshes_pending_to_passing_without_ci_coder_round(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Codex approves.")],
        gemini_outputs=[structured_pr_review(state="approved", summary="Gemini approves.")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=1)
    check_states = iter(
        [
            {"check_runs": [{"name": "test", "status": "in_progress", "conclusion": None}]},
            {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]},
            {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]},
        ]
    )

    def next_checks(*args, **kwargs):
        runner.pr_check_runs_payload = next(check_states)
        return original_get_pr_checks(*args, **kwargs)

    from coding_review_agent_loop import orchestrator as orchestrator_module

    original_get_pr_checks = orchestrator_module.get_pr_checks
    monkeypatch.setattr(orchestrator_module, "get_pr_checks", next_checks)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert not any("checks are still pending" in comment for comment in runner.comments)

def test_pr_loop_failing_github_checks_block_approval_even_with_auto_merge(tmp_path):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="LGTM.")],
        pr_check_runs_payload={
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]
        },
    )
    config = make_config(tmp_path, auto_merge=True, max_rounds=1)

    with pytest.raises(AgentLoopError, match="blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert any(
        comment.startswith("GitHub PR checks are failing for PR #77.") for comment in runner.comments
    )
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)

def test_pr_loop_downgrades_pending_ci_only_blocking_review_without_auto_merge(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Review complete.",
                blocking_items=["GitHub check `test` is still pending/in_progress."],
            )
        ],
        pr_check_runs_payload={"check_runs": []},
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    review_comment = runner.comments[0]
    assert review_comment.startswith("**Review verdict:** Approved")
    assert "### Blocking issues" not in review_comment
    stop_comment = runner.comments[-1]
    assert stop_comment.startswith("Reviewers approved PR #77, but GitHub checks are still pending.")
    _assert_pending_ci_stop_guidance(stop_comment)
    assert "Required checks not yet reporting: test" in stop_comment


# Verbatim Sol payloads from PR #757 rounds 1 and 3, previously stripped by
# the pending-CI filter because each real finding mentioned a regression test.
_SUPPRESSED_REVIEWS = json.loads(
    (Path(__file__).parent / "fixtures" / "pending_ci_real_reviews.json").read_text()
)


@pytest.mark.parametrize("payload", _SUPPRESSED_REVIEWS, ids=["round-1", "round-3"])
@pytest.mark.parametrize("parallel", [False, True])
def test_real_findings_survive_publication_and_reconciliation(tmp_path, monkeypatch, payload, parallel):
    # Replay as a fresh round; historical item dispositions need their original
    # ledger, but the summary and every blocking finding remain verbatim.
    response = (json.dumps({**payload, "prior_item_dispositions": []})
                + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex")
    runner = FakeRunner(
        codex_outputs=[response],
        gemini_outputs=[structured_pr_review(state="approved", summary="Review complete.")],
        pr_check_runs_payload={"check_runs": [{"name": "test", "status": "in_progress"}]},
    )
    original = orchestrator._run_validated_agent
    captured = {}

    class CoderReached(Exception):
        pass

    def capture(*args, **kwargs):
        if kwargs.get("role") == "coder":
            captured.update(kwargs)
            raise CoderReached
        return original(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_run_validated_agent", capture)
    monkeypatch.setattr(orchestrator, "watch_pr_checks", lambda *a, **k: pytest.fail("must not wait"))
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=parallel)
    with pytest.raises(CoderReached):
        run_pr_loop(runner, pr_number=77, config=config)
    reviews = [comment for comment in runner.comments if payload["summary"] in comment]
    assert len(reviews) == 1
    assert reviews[0].startswith("**Review verdict:** Blocking")
    for finding in payload["blocking_items"]:
        assert finding in reviews[0]
        assert finding in captured["prompt"]
    assert len(captured["repair_unresolved_item_ids"]) == len(payload["blocking_items"])


@pytest.mark.parametrize("summary", [
    "Authorization is broken. Add a regression test.",
    "CI is pending but the new resume path loses requirements.",
    "The approval reuse implementation is incorrect.",
])
def test_pending_only_items_do_not_override_substantive_summary(summary):
    review = parse_pr_review(
        structured_pr_review(state="blocking", summary=summary,
                             blocking_items=["GitHub check `test` is pending."]),
        reviewer="OpenAI Codex",
    )
    assert not orchestrator._is_pending_ci_only_review(
        review, _watch_check_board("pending", missing_required=("test",))
    )


def test_pr_loop_downgrades_pending_ci_only_blocking_review_with_auto_merge(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Review complete.",
                blocking_items=["GitHub check `lint` is still pending/in_progress."],
            )
        ],
        # "test" (the configured auto-merge check) is already green; "lint" is
        # still running so the overall board is `pending` (driving the
        # downgrade) without leaving the auto-merge wait polling forever.
        pr_check_runs_payload={
            "check_runs": [
                {"name": "test", "status": "completed", "conclusion": "success"},
                {"name": "lint", "status": "in_progress"},
            ]
        },
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path, auto_merge=True)

    with pytest.raises(AgentLoopError, match="full-board CI watch did not pass within"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)
    review_comment = runner.comments[0]
    assert review_comment.startswith("**Review verdict:** Approved")

def test_pr_loop_keeps_blocking_review_when_mixed_with_real_finding(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Missing null check in models.py causing a crash on empty input.",
                blocking_items=[
                    "Missing null check in models.py causing a crash on empty input.",
                    "GitHub check `test` is still pending/in_progress.",
                ],
            ),
            structured_pr_review(
                state="approved",
                summary="Codex final approval.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Added the missing null check.",
                addressed_items=["item-1", "item-2"],
                remaining_items=[],
            ),
        ],
        pr_check_runs_payload={"check_runs": []},
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path, max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    review_comment = next(comment for comment in runner.comments if "Missing null check" in comment)
    assert review_comment.startswith("**Review verdict:** Blocking")
    followup_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "Missing null check in models.py" in followup_prompt
    assert "GitHub check `test` is still pending/in_progress." in followup_prompt

def test_pr_loop_stops_gracefully_when_github_checks_pending_without_auto_merge(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=["Looks good locally.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_check_runs_payload={"check_runs": []},
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert any(
        comment.startswith("GitHub PR checks are still pending for PR #77.")
        for comment in runner.comments
    )
    stop_comment = runner.comments[-1]
    assert stop_comment.startswith("Reviewers approved PR #77, but GitHub checks are still pending.")
    _assert_pending_ci_stop_guidance(stop_comment)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    captured = capsys.readouterr()
    assert (
        "PR #77 was approved by Codex, but GitHub checks are still pending. "
        "This run cannot confirm the PR is merge-ready yet."
        in captured.out
    )
    _assert_pending_ci_stop_guidance(captured.out)

_QUEUED_TOO_LONG_CHECK_RUNS_PAYLOAD = {
    "check_runs": [
        {
            "id": 111,
            "name": "test",
            "status": "queued",
            "conclusion": None,
            "html_url": "https://github.com/OWNER/REPO/actions/runs/31123230205/job/1",
            "created_at": "2020-01-01T00:00:00Z",
            "started_at": None,
            "completed_at": None,
        }
    ]
}

_RUNNER_UNAVAILABLE_CHECK_RUNS_PAYLOAD = {
    "check_runs": [
        {
            "id": 222,
            "name": "test",
            "status": "completed",
            "conclusion": "cancelled",
            "html_url": "https://github.com/OWNER/REPO/actions/runs/31123230206/job/1",
            "created_at": "2020-01-01T00:00:00Z",
            "started_at": None,
            "completed_at": "2020-01-01T00:05:00Z",
        }
    ]
}


def test_pr_loop_infrastructure_stall_canonical_blocking_item_downgrades(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Review complete.",
                blocking_items=[
                    "GitHub check `test` (workflow run 31123230206) was cancelled before "
                    "execution because a hosted runner was unavailable; runner unavailable.",
                ],
            )
        ],
        pr_check_runs_payload=_RUNNER_UNAVAILABLE_CHECK_RUNS_PAYLOAD,
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    infra_comment = next(
        comment for comment in runner.comments
        if comment.startswith("External GitHub Actions infrastructure is blocking PR #77.")
    )
    assert "runner unavailable" in infra_comment.lower()
    stop_comment = runner.comments[-1]
    assert stop_comment.startswith(
        "PR #77 is blocked on external GitHub Actions infrastructure, not a code defect."
    )


def test_pr_loop_infrastructure_stall_queued_too_long_on_approval(tmp_path):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="LGTM.")],
        pr_check_runs_payload=_QUEUED_TOO_LONG_CHECK_RUNS_PAYLOAD,
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert any(
        comment.startswith("External GitHub Actions infrastructure is blocking PR #77.")
        for comment in runner.comments
    )
    assert any("queued" in comment.lower() for comment in runner.comments)


def test_pr_loop_infrastructure_stall_pre_execution_cancel_no_synthetic_item(tmp_path):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="LGTM.")],
        pr_check_runs_payload=_RUNNER_UNAVAILABLE_CHECK_RUNS_PAYLOAD,
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert not any("GitHub PR checks are failing" in comment for comment in runner.comments)


def test_pr_loop_genuine_failure_alongside_stall_not_downgraded(tmp_path):
    mixed_payload = {
        "check_runs": [
            {
                "id": 111,
                "name": "test",
                "status": "completed",
                "conclusion": "failure",
                "html_url": "https://github.com/OWNER/REPO/actions/runs/1/job/1",
            },
            _RUNNER_UNAVAILABLE_CHECK_RUNS_PAYLOAD["check_runs"][0] | {"name": "lint"},
        ]
    }
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Missing null check in models.py causing a crash on empty input.",
                blocking_items=["Missing null check in models.py causing a crash on empty input."],
            ),
        ],
        pr_check_runs_payload=mixed_payload,
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path, max_rounds=1)

    with pytest.raises(AgentLoopError, match="blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_pr_loop_mixed_single_item_stall_plus_defect_not_downgraded(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Review complete.",
                blocking_items=[
                    "GitHub check `test` (workflow run 31123230206) runner unavailable, and "
                    "models.py is missing a null check that crashes on empty input.",
                ],
            ),
        ],
        pr_check_runs_payload=_RUNNER_UNAVAILABLE_CHECK_RUNS_PAYLOAD,
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path, max_rounds=1)

    with pytest.raises(AgentLoopError, match="blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_pr_loop_generic_ci_keyword_item_not_downgraded(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Review complete.",
                blocking_items=["CI infrastructure failed for this PR."],
            ),
        ],
        pr_check_runs_payload=_RUNNER_UNAVAILABLE_CHECK_RUNS_PAYLOAD,
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path, max_rounds=1)

    with pytest.raises(AgentLoopError, match="blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_pr_loop_stall_plus_missing_required_no_infra_stop(tmp_path):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="LGTM.")],
        pr_check_runs_payload=_QUEUED_TOO_LONG_CHECK_RUNS_PAYLOAD,
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test", "lint"]},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not any(
        comment.startswith("External GitHub Actions infrastructure is blocking")
        for comment in runner.comments
    )
    assert any(
        comment.startswith("Reviewers approved PR #77, but GitHub checks are still pending.")
        for comment in runner.comments
    )


def test_pr_loop_stall_plus_partial_query_no_infra_stop(tmp_path):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="LGTM.")],
        pr_check_runs_payload=_QUEUED_TOO_LONG_CHECK_RUNS_PAYLOAD,
        pr_status_payload={"state": "pending", "statuses": []},
        pr_status_returncode=1,
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not any(
        comment.startswith("External GitHub Actions infrastructure is blocking")
        for comment in runner.comments
    )
    assert any(
        comment.startswith("Reviewers approved PR #77, but GitHub check status is unavailable.")
        or comment.startswith("Reviewers approved PR #77, but GitHub checks are still pending.")
        for comment in runner.comments
    )


def test_pr_loop_actively_running_check_unchanged_pending_behavior(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="LGTM.")],
        pr_check_runs_payload={
            "check_runs": [
                {
                    "id": 333,
                    "name": "test",
                    "status": "in_progress",
                    "conclusion": None,
                    "created_at": "2020-01-01T00:00:00Z",
                    "started_at": "2020-01-01T00:00:05Z",
                }
            ]
        },
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not any(
        comment.startswith("External GitHub Actions infrastructure is blocking")
        for comment in runner.comments
    )
    stop_comment = runner.comments[-1]
    assert stop_comment.startswith("Reviewers approved PR #77, but GitHub checks are still pending.")


def test_pr_loop_coder_round_gets_stall_backstop_context(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Missing null check in models.py causing a crash on empty input.",
                blocking_items=["Missing null check in models.py causing a crash on empty input."],
            ),
            structured_pr_review(
                state="approved",
                summary="Codex final approval.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Added the missing null check.",
                addressed_items=["item-1"],
                remaining_items=[],
            ),
        ],
        pr_check_runs_payload={
            "check_runs": [
                _RUNNER_UNAVAILABLE_CHECK_RUNS_PAYLOAD["check_runs"][0],
            ]
        },
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path, max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "External CI infrastructure is currently blocking" in followup_prompt
    assert "31123230206" in followup_prompt
    assert "Do not wait for these checks" in followup_prompt


def test_pr_loop_auto_merge_queued_too_long_stops_without_merge(tmp_path):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="LGTM.")],
        pr_check_runs_payload=_QUEUED_TOO_LONG_CHECK_RUNS_PAYLOAD,
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=1200, ci_poll_interval_seconds=30)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)
    assert any(
        comment.startswith("External GitHub Actions infrastructure is blocking")
        for comment in runner.comments
    )
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) == 0


def test_pr_loop_auto_merge_pre_execution_cancel_stops_without_merge(tmp_path):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="LGTM.")],
        pr_check_runs_payload=_RUNNER_UNAVAILABLE_CHECK_RUNS_PAYLOAD,
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=1200, ci_poll_interval_seconds=30)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)
    assert any(
        comment.startswith("External GitHub Actions infrastructure is blocking")
        for comment in runner.comments
    )


def test_pr_loop_auto_merge_stall_plus_missing_required_keeps_waiting(tmp_path):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="LGTM.")],
        pr_check_runs_payload=_QUEUED_TOO_LONG_CHECK_RUNS_PAYLOAD,
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test", "lint"]},
    )
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=60, ci_poll_interval_seconds=30)

    with pytest.raises(AgentLoopError, match="did not pass within"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


def test_pr_loop_auto_merge_stall_plus_partial_query_keeps_waiting(tmp_path):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="LGTM.")],
        pr_check_runs_payload=_QUEUED_TOO_LONG_CHECK_RUNS_PAYLOAD,
        pr_status_payload={"state": "pending", "statuses": []},
        pr_status_returncode=1,
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=60, ci_poll_interval_seconds=30)

    with pytest.raises(AgentLoopError, match="did not pass within"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


def test_pr_loop_summarizes_approved_followups_before_pending_check_stop(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Looks good locally.\n\n### Future follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        pr_check_runs_payload={"check_runs": []},
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    config = make_config(tmp_path, approved_followups="summarize")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.comments) == 4
    assert runner.comments[1].startswith("Approved-review future follow-ups for PR #77:")
    assert "- Add cleanup docs. (Codex)" in runner.comments[1]
    assert "<!-- AGENT_APPROVED_FOLLOWUPS: pr=77 head=abc123 mode=summarize -->" in runner.comments[1]
    assert runner.comments[2].startswith("GitHub PR checks are still pending for PR #77.")
    assert runner.comments[3].startswith("Reviewers approved PR #77, but GitHub checks are still pending.")
    _assert_pending_ci_stop_guidance(runner.comments[3])

def test_pr_loop_summary_marker_has_single_blank_line_before_footer_marker(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Looks good locally.\n\n### Future follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, approved_followups="summarize")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = runner.comments[-1]
    assert (
        "These were mentioned in approved reviews as future work and did not block merge readiness.\n\n"
        "<!-- AGENT_APPROVED_FOLLOWUPS: pr=77 head=abc123 mode=summarize -->\n"
        "-- coding-review-agent-loop"
    ) in summary

def test_pr_loop_creates_approved_followup_issues_before_unavailable_check_stop(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Looks good locally.\n\n### Future follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        pr_check_runs_payload={"check_runs": []},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="500 Internal Server Error",
        pr_check_runs_returncode=1,
        pr_check_runs_stderr="500 Internal Server Error",
        pr_status_returncode=1,
        pr_status_stderr="500 Internal Server Error",
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "Follow up future review note: Add cleanup docs."
    assert len(runner.comments) == 4
    assert runner.comments[1].startswith("Created approved-review future follow-up issues for PR #77:")
    assert "<!-- AGENT_APPROVED_FOLLOWUPS: pr=77 head=abc123 mode=issue -->" in runner.comments[1]
    assert runner.comments[2].startswith("GitHub PR check status is unavailable for PR #77.")
    assert runner.comments[3].startswith(
        "Reviewers approved PR #77, but GitHub check status is unavailable."
    )
    assert "Wait for GitHub check status to become available." in runner.comments[3]
    _assert_pending_ci_stop_guidance(runner.comments[3])

def test_pr_loop_skips_duplicate_approved_followup_issue_creation_when_marker_exists(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves.\n\n### Future follow-ups\n- Add cleanup docs.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        pr_payload={
            "comments": [
                {
                    "author": {"login": "coding-review-agent-loop"},
                    "createdAt": "2026-05-22T10:00:00Z",
                    "body": (
                        "Created approved-review future follow-up issues for PR #77:\n\n"
                        "- https://github.com/OWNER/REPO/issues/99\n\n"
                        "These were mentioned in approved reviews as future work and did not block merge readiness.\n\n"
                        "<!-- AGENT_APPROVED_FOLLOWUPS: pr=77 head=abc123 mode=issue -->\n"
                        "-- coding-review-agent-loop"
                    ),
                }
            ]
        },
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.issues == []
    assert runner.comments == [
        "**Review verdict:** Approved\n\n"
        "Codex approves.\n\n### Future follow-ups\n- Add cleanup docs.\n"
        "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex: unknown model (medium)"
    ]

def test_pr_loop_allows_repos_without_github_checks_when_branch_protection_404(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_check_runs_payload={"check_runs": []},
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="404 Not Found",
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not any(comment.startswith("GitHub PR checks are") for comment in runner.comments)

def test_pr_loop_allows_repos_without_github_checks_when_branch_protection_403(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_check_runs_payload={"check_runs": []},
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="403 Forbidden",
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not any(comment.startswith("GitHub PR checks are") for comment in runner.comments)

def test_get_pr_checks_returns_no_checks_in_dry_run(tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, dry_run=True)

    pr_checks = get_pr_checks(
        runner,
        config=config,
        metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Improve review prompt context",
            head_branch="feature/review-context",
            base_branch="main",
            head_sha="abc123",
            url="https://github.com/OWNER/REPO/pull/77",
        ),
    )

    assert pr_checks.state == "no_checks"
    assert pr_checks.branch_protection_status == "unavailable"
    assert pr_checks.branch_protection_note == "Dry run mode does not query live GitHub PR checks."

def test_pr_loop_combines_issue_and_pr_signed_human_requirements(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
            ],
        issue_payload={
            "number": 56,
            "title": "Support issue comments",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
            "author": {"login": "issue-author"},
            "createdAt": "2026-05-17T08:00:00Z",
            "url": "https://github.com/OWNER/REPO/issues/56",
        },
        pr_payload={
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Use the absolute URL in the PR path.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
    )
    config = make_config(tmp_path, reviewer="codex")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(
            HumanReviewRequirement(
                source_type="Issue body",
                author="issue-author",
                created_at="2026-05-17T08:00:00Z",
                url="https://github.com/OWNER/REPO/issues/56",
                body="Preserve backward compatibility.",
            ),
        ),
    )

    assert run_pr_loop(runner, pr_number=77, config=config, issue_context=issue_context) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Preserve backward compatibility." in prompt
    assert "Use the absolute URL in the PR path." in prompt
    assert prompt.index("Preserve backward compatibility.") < prompt.index(
        "Use the absolute URL in the PR path."
    )

def test_pr_loop_keeps_blocking_review_when_future_followups_are_misclassified(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            "Still blocked.\n\n"
            "### Same-PR follow-ups\n"
            "- Tighten the reset helper.\n\n"
            "### Future follow-ups\n"
            "- Consider a broader cleanup later.\n\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- Google Gemini",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Fixed review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(
        tmp_path,
        reviewer=("gemini", "codex"),
        approved_followups="fix-and-issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments[0].startswith("**Review verdict:** Blocking\n\nStill blocked.")
    assert "Consider a broader cleanup later." not in runner.comments[0]
    # Only the same-PR item ("Tighten the reset helper.") is tracked -- the
    # blocking summary prose ("Still blocked.") is never itemized when a
    # same_pr_followups entry already represents the actionable concern, so
    # this routes through the lean same-PR-only coder prompt.
    followup_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "Tighten the reset helper." in cmd[-1]
    )
    assert "Address the follow-up items below" in followup_prompt
    assert "Still blocked." in followup_prompt
    assert "Latest reviewer summaries (review-level context):" in followup_prompt

def test_pr_loop_requires_all_reviewers_to_approve(tmp_path):
    runner = FakeRunner(
        codex_outputs=["Codex approves.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        claude_outputs=["Claude approves.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        auto_merge=True,
        test_command=("pytest", "tests/test_agent_loop.py"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert agent_commands == [["codex", "exec"], ["claude", "--print"]]
    assert len(runner.comments) == 3
    commands = [cmd for cmd, _cwd in runner.commands]
    metadata_fetches = [
        cmd
        for cmd in commands
        if cmd[:3] == ["gh", "pr", "view"]
        and "--json" in cmd
        and cmd[cmd.index("--json") + 1]
        == "number,title,headRefName,baseRefName,headRefOid,url,body,comments,reviews"
    ]
    assert len(metadata_fetches) == 1
    assert ["pytest", "tests/test_agent_loop.py"] in commands
    assert ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge", "--match-head-commit", "abc123"] in commands


def test_pr_loop_skips_prior_approval_when_pr_head_is_unchanged(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Codex approves the initial head.",
                reviewer="OpenAI Codex",
            )
        ],
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Claude needs one fix.",
                blocking_items=["Fix the admission cleanup race."],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(
                state="approved",
                summary="Claude accepts the follow-up.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                reviewer="Anthropic Claude",
            ),
        ],
        gemini_outputs=[
            structured_coder_followup(
                addressed_items=["item-1"],
                remaining_items=[],
                reviewer="Google Gemini",
            )
        ],
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    codex_reviews = [
        cmd
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"]
    ]
    assert len(codex_reviews) == 1
    assert "round 1" in codex_reviews[0][-1]


@pytest.mark.parametrize(
    (
        "second_requirement_body",
        "second_requirement_created_at",
        "second_requirement_url",
        "expected_second_prompt_text",
    ),
    [
        (
            "Also preserve the reviewer attribution.",
            "2026-05-18T10:10:00Z",
            "https://github.com/OWNER/REPO/pull/77#issuecomment-2",
            "stable-id",
        ),
        (
            "The edited instruction changes the required audit trail.",
            "2026-05-18T10:00:00Z",
            "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            "The edited instruction changes the required audit trail.",
        ),
    ],
)
def test_pr_loop_rereviews_unchanged_head_when_human_requirement_changes(
    tmp_path,
    monkeypatch,
    second_requirement_body,
    second_requirement_created_at,
    second_requirement_url,
    expected_second_prompt_text,
):
    requirement_1 = HumanReviewRequirement(
        source_type="PR comment",
        author="maintainer",
        created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="Keep the current audit trail.",
    )
    requirement_2 = HumanReviewRequirement(
        source_type="PR comment",
        author="maintainer",
        created_at=second_requirement_created_at,
        url=second_requirement_url,
        body=second_requirement_body,
    )
    if expected_second_prompt_text == "stable-id":
        expected_second_prompt_text = f"Requirement {requirement_2.requirement_id}"
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Codex approves the initial requirement.",
                human_requirements_resolved=True,
            ),
            structured_pr_review(
                state="approved",
                summary="Codex approves after reviewing the new requirement.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": HUMAN_REQUIREMENTS_ACK_ITEM_ID, "disposition": "resolved"},
                ],
                human_requirements_resolved=True,
            ),
        ],
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Claude needs one fix.",
                blocking_items=["Fix the admission cleanup race."],
                reviewer="Anthropic Claude",
                human_requirements_resolved=True,
            ),
            structured_pr_review(
                state="approved",
                summary="Claude accepts the follow-up.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": HUMAN_REQUIREMENTS_ACK_ITEM_ID, "disposition": "resolved"},
                ],
                reviewer="Anthropic Claude",
                human_requirements_resolved=True,
            ),
        ],
        gemini_outputs=[
            structured_coder_followup(
                addressed_items=["item-1"],
                remaining_items=[],
                human_requirement_ids=["Requirement 1"],
                reviewer="Google Gemini",
            )
        ],
        advance_pr_head_on_coder_followup=False,
    )
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Improve review prompt context",
        head_branch="feature/review-context",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/77",
    )
    prior_codex_review = _attach_round_metadata(
        structured_pr_review(
            state="approved",
            summary="Codex approves the initial requirement.",
            human_requirements_resolved=True,
            reviewer="OpenAI Codex",
        ),
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="OpenAI Codex",
            round_number=1,
            subject="abc123",
            state="approved",
            surfaced_reviewer_requirement_ids=("Requirement 1",),
        ),
    )
    context_calls = 0

    def next_context(*args, **kwargs):
        nonlocal context_calls
        context_calls += 1
        return PullRequestReviewContext(
            metadata=metadata,
            comments=(
                IssueComment(
                    author="coding-review-agent-loop",
                    created_at="2026-05-18T11:00:00Z",
                    body=str(prior_codex_review),
                ),
            )
            if context_calls > 1
            else (),
            human_requirements=(
                (requirement_2,)
                if second_requirement_url.endswith("issuecomment-1")
                else (requirement_1, requirement_2)
            )
            if context_calls > 1
            else (requirement_1,),
        )

    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator.get_pr_review_context",
        next_context,
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    codex_reviews = [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(codex_reviews) == 2
    assert expected_second_prompt_text in codex_reviews[1][-1]


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
    assert len(runner.comments) == 4
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

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
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
            + prior_item_dispositions("[item-1] resolved")
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

def test_pr_loop_fix_and_issue_drops_blocking_round_future_followups(tmp_path):
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
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Tightened message.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-issue")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.issues) == 1
    assert "Stale future item from the blocking round." not in runner.issues[0]["body"]

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

def test_pr_loop_reruns_all_reviewers_when_any_reviewer_blocks(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Needs a regression test.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Addressed review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude approves."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Codex approves first pass.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Codex approves second pass."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer=("claude", "codex"))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.comments) == 5
    followup_prompt = next(
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"] and "Address the review below" in cmd[-1]
    )
    assert "Needs a regression test." in followup_prompt
    assert "Codex approves first pass." in followup_prompt
    assert "Latest reviewer summaries (review-level context):" in followup_prompt
    commands = [cmd for cmd, _cwd in runner.commands]
    metadata_fetches = [
        cmd
        for cmd in commands
        if cmd[:3] == ["gh", "pr", "view"]
        and "--json" in cmd
        and cmd[cmd.index("--json") + 1]
        == "number,title,headRefName,baseRefName,headRefOid,url,body,comments,reviews"
    ]
    assert len(metadata_fetches) == 2

def test_pr_loop_rejects_cross_reviewer_approval_without_prior_item_disposition(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Needs a regression test.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude resolves it."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Codex approves first pass.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Codex approves second pass.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        gemini_outputs=["Implemented fix.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini"],
    )
    config = make_config(tmp_path, coder="gemini", reviewer=("claude", "codex"), max_rounds=2)

    with pytest.raises(AgentLoopError, match="did not evaluate all prior unresolved items: item-1"):
        run_pr_loop(runner, pr_number=77, config=config)

    second_codex_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] and "round 2" in cmd[-1]
    ][0]
    assert "Prior unresolved review items from earlier rounds" in second_codex_prompt
    assert "[item-1] blocking from Claude in round 1" in second_codex_prompt

def test_pr_loop_can_downgrade_prior_blocker_to_future_followup_only_in_approved_review(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Addressed review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "Missing docs cleanup.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM now."
            + prior_item_dispositions("[item-1] future follow-up: cleanup can wait")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, approved_followups="summarize", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = runner.comments[-1]
    assert summary.startswith("Approved-review future follow-ups for PR #77:")
    assert "Missing docs cleanup." in summary

def test_pr_loop_persists_downgraded_future_followup_across_later_blocking_rounds(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Still blocked.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "LGTM."
            + prior_item_dispositions("[item-1] future follow-up: cleanup can wait", "[item-2] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Missing docs cleanup.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Implemented fix for Claude.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] future follow-up: cleanup can wait", "[item-2] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        coder="codex",
        approved_followups="summarize",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = runner.comments[-1]
    assert summary.startswith("Approved-review future follow-ups for PR #77:")
    assert "Missing docs cleanup." in summary

def test_pr_loop_finalized_future_followup_summary_preserves_disposition_notes(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Still blocked.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: cleanup can wait until after rollout",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Missing docs cleanup.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Implemented blocker.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: cleanup can wait until after rollout",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        coder="codex",
        approved_followups="summarize",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = runner.comments[-1]
    assert "Missing docs cleanup." in summary
    assert "Update from Codex: cleanup can wait until after rollout" in summary

def test_pr_loop_compact_review_mode_uses_fresh_sessions_and_compact_prior_ledger(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves with future work.\n\n"
            "### Future follow-ups\n"
            "- Document cache cleanup behavior.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Codex approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: still future work",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Claude still blocks.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: still future work",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        gemini_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Implemented blocker and ran focused tests.",
                addressed_items=["item-1", "item-2"],
                remaining_items=[],
                tests_run=["python -m pytest tests/test_agent_loop.py -k compact_pr"],
                reviewer="Google Gemini",
            )
        ],
        pr_payload={"body": "PR body used by compact review mode."},
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        approved_followups="summarize",
        max_rounds=2,
        pr_review_context_mode="compact",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    codex_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(codex_prompts) == 2
    second_codex_prompt = codex_prompts[1]
    assert COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER in second_codex_prompt
    assert "PR body used by compact review mode." in second_codex_prompt
    assert "Implemented blocker and ran focused tests." in second_codex_prompt
    assert "python -m pytest tests/test_agent_loop.py -k compact_pr" in second_codex_prompt
    assert "Document cache cleanup behavior." not in second_codex_prompt
    assert "[item-1] future" not in second_codex_prompt
    assert "Claude still blocks." in second_codex_prompt
    assert not any("--resume" in cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])

    assert runner.comments[-1].startswith("Approved-review future follow-ups for PR #77:")
    assert "Document cache cleanup behavior." in runner.comments[-1]

def test_pr_loop_carries_prior_item_notes_without_creating_duplicate_blocker_items(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Needs regression coverage.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Still blocked."
            + prior_item_dispositions("[item-1] still blocking: include API error path too")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Added coverage.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Expanded coverage.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    second_coder_prompt = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]][1]
    assert "Latest reviewer updates:" in second_coder_prompt
    assert "Codex: include API error path too" in second_coder_prompt
    assert "[item-2]" not in second_coder_prompt

def test_pr_loop_stops_on_incomplete_review_without_coder_followup(tmp_path):
    """An explicit inability to review is an agent failure, not a new blocker."""
    runner = FakeRunner(
        codex_outputs=[
            "Needs regression coverage.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Review incomplete; item-1 requires further inspection of the PR diff "
            "and referenced files before a disposition can be confirmed."
            + prior_item_dispositions(
                "[item-1] resolved: Resolution could not be confirmed; "
                "PR diff and referenced files need review."
            )
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Added regression coverage.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    with pytest.raises(AgentLoopError, match="reviewer-internal error"):
        run_pr_loop(runner, pr_number=77, config=config)

    coder_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(coder_commands) == 1
    assert len(runner.comments) == 2
    assert "Review incomplete" not in "\n".join(runner.comments)


def test_pr_loop_keeps_active_carried_disposition_actionable_without_duplicate_summary_item(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Initial scope concern.",
                blocking_items=["All 14 locale catalogs must be evaluated."],
            ),
            structured_pr_review(
                state="blocking", summary="Unable to verify the translation detail today.",
                prior_item_dispositions=[{
                    "item_id": "item-1", "disposition": "blocking",
                    "note": "The original scope-completeness claim remains unresolved.",
                }],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking", summary="Evaluated every locale.",
                addressed_items=["item-1"], remaining_items=[],
            ),
            structured_coder_followup(
                state="blocking", summary="Need more evidence for the carried claim.",
                addressed_items=[], remaining_items=["item-1"],
            ),
        ],
    )
    config = make_config(tmp_path, max_rounds=2)

    with pytest.raises(AgentLoopError, match="still reported blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=config)

    reviewer_records = [
        _decode_round_metadata(orchestrator.ROUND_RESUME_MARKER_RE.search(comment["body"])["payload"])
        for comment in runner.pr_payload["comments"]
        if orchestrator.ROUND_RESUME_MARKER_RE.search(comment["body"])
    ]
    round_two = next(record for record in reviewer_records if record.agent == "Codex" and record.round_number == 2)
    assert round_two.new_items == ()
    assert round_two.dispositions[0].disposition == "blocking"
    assert not any("agent-unavailable" in comment["body"] for comment in runner.pr_payload["comments"])

def test_pr_loop_quarantines_unavailable_reviewer_while_healthy_reviewer_finishes(tmp_path):
    unavailable = json.dumps(
        {
            "schema_version": 1,
            "kind": "agent_unavailable",
            "retryable": False,
            "category": "environment",
            "summary": "The review checkout cannot access the diff.",
            "suggested_action": "Repair the reviewer sandbox before retrying it.",
        }
    ) + "\n<!-- AGENT_UNAVAILABLE -->\n-- OpenAI Codex"
    runner = FakeRunner(
        codex_outputs=[unavailable],
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="A regression test is required.",
                blocking_items=["Add coverage for the unavailable-reviewer path."],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(
                state="approved",
                summary="The healthy-reviewer finding is resolved.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                reviewer="Anthropic Claude",
            ),
        ],
        gemini_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Added the requested regression coverage.",
                addressed_items=["item-1"],
                reviewer="Google Gemini",
            )
        ],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        max_rounds=3,
    )

    with pytest.raises(AgentLoopError, match="missing required input from Codex"):
        run_pr_loop(runner, pr_number=77, config=config)

    codex_reviews = [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    claude_reviews = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    gemini_coder_turns = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"]]
    assert len(codex_reviews) == 1
    assert len(claude_reviews) == 2
    assert len(gemini_coder_turns) == 1
    assert "**Review status: Incomplete**" in runner.comments[-1]
    assert "Claude" in runner.comments[-1]
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


def test_pr_loop_retries_explicit_retryable_agent_unavailable_response(tmp_path):
    retryable_unavailable = json.dumps(
        {
            "schema_version": 1,
            "kind": "agent_unavailable",
            "retryable": True,
            "category": "provider",
            "summary": "The provider returned a temporary unavailable response.",
            "suggested_action": "Retry the review shortly.",
        }
    ) + "\n<!-- AGENT_UNAVAILABLE -->\n-- OpenAI Codex"
    runner = FakeRunner(
        codex_outputs=[
            retryable_unavailable,
            structured_pr_review(reviewer="OpenAI Codex"),
        ]
    )
    config = make_config(tmp_path, reviewer="codex", agent_max_retries=1)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    codex_reviews = [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(codex_reviews) == 2
    assert any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_pr_loop_posts_human_readable_item_labels_in_new_and_prior_sections(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented the requested PR body change.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Same-PR follow-ups\n"
            "- Require source issue reference in PR body.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", approved_followups="fix-and-summarize", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments[0] == (
        "**Review verdict:** Blocking\n\n"
        "### Same-PR follow-ups\n"
        "- Require source issue reference in PR body.\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- OpenAI Codex: unknown model (medium)"
    )
    assert runner.comments[2] == (
        "**Review verdict:** Approved\n\n"
        "Looks good.\n\n"
        "### Prior unresolved item dispositions\n"
        "- [item-1] RESOLVED\n"
        "  - Original finding: Same-PR follow-up from OpenAI Codex, round 1: Require source issue reference in PR body.\n"
        "<!-- AGENT_STATE: approved -->\n"
        "-- OpenAI Codex: unknown model (medium)"
    )

def test_pr_loop_tracks_blocking_items_text_not_summary_when_they_differ(tmp_path):
    """The tracked unresolved item must use the blocking_items bullet text, not the
    summary prose, so summary is never itemized when blocking_items is populated."""
    runner = FakeRunner(
        claude_outputs=["Implemented fixes.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "Needs one more regression test before merge."
            + blocking_issues("Add the mixed-history resume case to `tests/test_agent_loop.py`.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    second_coder_prompt = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]][0]
    assert "Add the mixed-history resume case to `tests/test_agent_loop.py`." in second_coder_prompt
    assert "Needs one more regression test before merge." in second_coder_prompt
    assert "Needs one more regression test before merge." not in second_coder_prompt.split("Codex unresolved blocking item", 1)[1]
    assert runner.comments[0] == (
        "**Review verdict:** Blocking\n\n"
        "Needs one more regression test before merge.\n\n"
        "### Blocking issues\n"
        "- Add the mixed-history resume case to `tests/test_agent_loop.py`.\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- OpenAI Codex: unknown model (medium)"
    )
    second_review_prompt = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["codex"]][1]
    assert "[item-1]" in second_review_prompt
    assert "Add the mixed-history resume case to `tests/test_agent_loop.py`." in second_review_prompt

def test_pr_loop_same_pr_only_review_does_not_duplicate_summary_as_blocking_item(tmp_path):
    """Regression test for issue #501 / llm-dialectic PR #257: a blocking review whose
    summary prose restates the same concern as its lone same_pr_followups entry must
    produce exactly one tracked item (the same-PR follow-up), not a second,
    summary-derived blocking item."""
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="The PR is docs-only; the new audit doc contains stale path references.",
                same_pr_followups=[
                    "Update docs/SECURITY_FINDINGS.md references to docs/audit/SECURITY_FINDINGS.md."
                ],
            ),
            structured_pr_review(
                state="approved",
                summary="Stale references fixed.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Updated the stale path references.",
                addressed_items=["item-1"],
                remaining_items=[],
            ),
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        max_rounds=2,
        approved_followups="fix-and-summarize",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert (
        "Update docs/SECURITY_FINDINGS.md references to docs/audit/SECURITY_FINDINGS.md."
        in followup_prompt
    )
    assert "The PR is docs-only" in followup_prompt
    assert "The PR is docs-only" not in followup_prompt.split("Codex same-PR follow-up", 1)[1]
    # Routed through the lean same-PR-only prompt, confirming no separate blocking
    # item was created alongside the same-PR item.
    assert "Address the follow-up items below" in followup_prompt

def test_pr_loop_creates_one_tracked_item_per_blocking_items_entry(tmp_path):
    """Each blocking_items entry gets its own tracked item; the summary is never
    itemized when blocking_items is populated."""
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Two separate problems found.",
                blocking_items=[
                    "Fix the null pointer dereference in parser.py.",
                    "Add missing docstring to the public API.",
                ],
            ),
            structured_pr_review(
                state="approved",
                summary="Both issues fixed.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Fixed both issues.",
                addressed_items=["item-1", "item-2"],
                remaining_items=[],
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "[item-1]" in followup_prompt
    assert "[item-2]" in followup_prompt
    assert "Fix the null pointer dereference in parser.py." in followup_prompt
    assert "Add missing docstring to the public API." in followup_prompt
    assert "Two separate problems found." in followup_prompt
    assert "Two separate problems found." not in followup_prompt.split("Codex unresolved blocking item", 1)[1]

def test_pr_loop_falls_back_to_summary_item_when_no_structured_fields_present(tmp_path):
    """When a blocking review has neither blocking_items nor same_pr_followups, the
    summary must still become the tracked item so a genuine blocking verdict is not
    silently dropped from the ledger."""
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="I have concerns about this approach but cannot pinpoint a specific line.",
            ),
            structured_pr_review(
                state="approved",
                summary="Satisfied after discussion.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Clarified the approach.",
                addressed_items=["item-1"],
                remaining_items=[],
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert (
        "I have concerns about this approach but cannot pinpoint a specific line."
        in followup_prompt
    )

def test_resume_pr_round_reparses_orchestrator_rendered_blocking_issues_comment():
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Need one more regression test before merge.",
        status="blocking",
        source_status="blocking",
    )
    rendered_review = _render_public_pr_review_comment(
        parse_review(
            "Need one more regression test before merge."
            + blocking_issues("Exercise the structured-resume path.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            reviewer="OpenAI Codex",
        ),
        reviewer="Codex",
        human_requirements_resolved_flag=False,
        prior_items=(),
        dispositions=(),
    )
    review_comment = _attach_round_metadata(
        rendered_review,
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
            dispositions=(),
            new_items=(),
            state="blocking",
        ),
    )
    coder_comment = _attach_round_metadata(
        "Addressed the review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=review_comment),
        ],
        head_sha="abc123",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    resumed_review = parse_review(resumed.completed_reviews[0].body, reviewer="Codex")
    assert [item.text for item in resumed_review.blocking_items] == [
        "Exercise the structured-resume path."
    ]
    assert resumed_review.summary == "Need one more regression test before merge."

def test_resume_pr_round_prefers_structured_coder_followup_metadata():
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Need one more regression test before merge.",
        status="blocking",
        source_status="blocking",
    )
    raw_structured_followup = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Added the requested regression test.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "human_requirement_dispositions": [],
                "human_requirements": {
                    "addressed_ids": ["Requirement 1"],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    parsed = validate_structured_coder_followup(raw_structured_followup)
    assert parsed is not None
    public_comment = _render_public_coder_followup_comment(parsed, agent="Claude")
    coder_comment = _attach_round_metadata(
        public_comment,
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
            raw_structured_coder_response=raw_structured_followup,
        ),
    )

    resumed = _resume_pr_round(
        [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment)],
        head_sha="abc123",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.coder_output == raw_structured_followup
    resumed_followup = validate_structured_coder_followup(resumed.coder_output)
    assert resumed_followup is not None
    assert resumed_followup.human_requirements.addressed_ids == ("Requirement 1",)
    assert '"kind": "coder_followup"' not in _strip_round_metadata(coder_comment)


def test_resume_pr_round_recovers_summary_only_qualification_checkpoint():
    item = UnresolvedReviewItem(
        item_id="item-30",
        reviewer="GitHub managed exact-head CI",
        source_round=1,
        text="Managed exact-head CI failed.",
        status="blocking",
        source_status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="repair_required",
        failed_head_sha="failed-head",
        candidate_head_sha=None,
        obligation_identity="managed-exact-head-ci:item-30",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity=item.obligation_identity,
        lifecycle="repair_required",
        failed_head_sha="failed-head",
        candidate_head_sha=None,
        base_branch="main",
        watch_failure_extension_used=True,
        allowed_rounds=2,
    )
    summary_comment = _attach_round_metadata(
        "Persisted repair handoff.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=2,
            subject="failed-head",
            prior_items=(item,),
            state="blocking",
            phase="qualification-checkpoint",
            qualification_checkpoint=checkpoint,
        ),
    )

    resumed = _resume_pr_round(
        [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=summary_comment)],
        head_sha="failed-head",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.round_number == 2
    assert resumed.coder_output is None
    assert resumed.completed_reviews == ()
    assert resumed.prior_items == (item,)
    assert resumed.qualification_checkpoint == checkpoint


def test_pr_loop_resume_after_interrupted_coder_keeps_failed_head_and_budget(
    tmp_path, monkeypatch
):
    failed = PullRequestCheck(
        name="test", kind="check_run", status="failure", url="https://example.test/555"
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Initial review."),
            structured_pr_review(
                state="approved",
                summary="Repaired head approved.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking", summary="Repaired the failing check.", addressed_items=["item-1"]
            )
        ],
    )
    config = make_config(tmp_path, auto_merge=False, watch_pending_ci=False, max_rounds=3)

    def checks(*args, **kwargs):
        if kwargs["metadata"].head_sha == "abc123":
            return _watch_check_board("failing", failing=(failed,))
        return _watch_check_board("passing")

    monkeypatch.setattr(orchestrator, "get_pr_checks", checks)
    original = orchestrator._run_validated_agent
    interrupt = True

    class CoderInterrupted(Exception):
        pass

    def interrupt_during_coder(*args, **kwargs):
        nonlocal interrupt
        if kwargs.get("role") == "coder" and interrupt:
            interrupt = False
            raise CoderInterrupted
        return original(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_run_validated_agent", interrupt_during_coder)
    with pytest.raises(CoderInterrupted):
        run_pr_loop(runner, pr_number=77, config=config)

    checkpoint_records = [
        comment for comment in runner.pr_payload["comments"]
        if "qualification checkpoint: machine obligation state persisted before coder handoff"
        in comment["body"]
    ]
    assert len(checkpoint_records) == 1
    assert "AGENT_LOOP_META" in checkpoint_records[0]["body"]

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    review_commands = [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(review_commands) == 2
    assert sum(
        comment.startswith(
            "PR #77 qualification checkpoint: machine obligation state persisted before coder handoff."
        )
        for comment in runner.comments
    ) == 2

def test_resume_pr_round_marks_empty_ledger_incomplete_after_same_subject_prior_new_items():
    prior_new_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Prior same-head item.",
        status="blocking",
    )
    prior_review_comment = _attach_round_metadata(
        "Prior review.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="abc123",
            prior_items=(),
            new_items=(prior_new_item,),
            state="blocking",
        ),
    )
    current_coder_comment = _attach_round_metadata(
        "Current coder output.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(),
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=prior_review_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=current_coder_comment),
        ],
        head_sha="abc123",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.ledger_may_be_incomplete is True

def test_resume_pr_round_does_not_mark_ledger_incomplete_for_cross_subject_prior_new_items():
    prior_new_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Prior other-head item.",
        status="blocking",
    )
    prior_review_comment = _attach_round_metadata(
        "Prior review.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(prior_new_item,),
            state="blocking",
        ),
    )
    current_coder_comment = _attach_round_metadata(
        "Current coder output.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="new-sha",
            prior_items=(),
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=prior_review_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=current_coder_comment),
        ],
        head_sha="new-sha",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.ledger_may_be_incomplete is False

def test_resume_pr_round_recovers_unrecorded_head_advance_reviewer_new_item():
    active_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Fix the regression before merge.",
        status="blocking",
    )
    coder_comment = _attach_round_metadata(
        "Initial PR handoff.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="old-sha",
            prior_items=(),
        ),
    )
    review_comment = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(active_item,),
            state="blocking",
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=review_comment),
        ],
        head_sha="new-sha",
        configured_reviewers=("gemini",),
    )

    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert resumed.ledger_may_be_incomplete is True
    assert resumed.round_number == 1
    assert resumed.completed_reviews == ()
    assert [item.item_id for item in resumed.prior_items] == ["item-2"]
    assert resumed.next_unresolved_item_number == 3


def test_resume_pr_round_recovers_parallel_reconciliation_new_items_after_head_advance():
    """Parallel publication checkpoints omit items; reconciliation owns the ledger."""
    active_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Preserve the first reconciled blocker.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Preserve the second reconciled blocker.",
            status="same-pr",
        ),
    )
    coder_comment = _attach_round_metadata(
        "Initial PR handoff.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=1,
            subject="old-sha", prior_items=(),
        ),
    )
    publication_comment = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=1,
            subject="old-sha", prior_items=(), state="blocking", phase="publication",
        ),
    )
    reconciliation_comment = _attach_round_metadata(
        "Reconciliation.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Review reconciliation", round_number=1,
            subject="old-sha", prior_items=(), new_items=active_items, state="blocking",
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=publication_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:02:00Z", body=reconciliation_comment),
        ],
        head_sha="new-sha",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert [item.item_id for item in resumed.prior_items] == ["item-1", "item-2"]
    assert resumed.next_unresolved_item_number == 3

def test_resume_pr_round_recovers_coder_only_unrecorded_head_advance():
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Still needs a targeted test.",
        status="same-pr",
    )
    future_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Document this later.",
        status="future",
    )
    coder_comment = _attach_round_metadata(
        "Addressed prior feedback.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="old-sha",
            prior_items=(carried_item, future_item),
            compact_prior_summaries=("Older summary.",),
        ),
    )

    resumed = _resume_pr_round(
        [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment)],
        head_sha="new-sha",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert resumed.round_number == 2
    assert [item.item_id for item in resumed.prior_items] == ["item-1"]
    assert resumed.compact_prior_summaries == ("Older summary.",)

def test_resume_pr_round_recovers_reviewer_only_with_aggregated_dispositions():
    prior_blocking = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Fix the flaky test.",
        status="blocking",
    )
    prior_same_pr = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Tighten the docs.",
        status="same-pr",
    )
    future_new_item = UnresolvedReviewItem(
        item_id="item-3",
        reviewer="OpenAI Codex",
        source_round=2,
        text="Follow up in another PR.",
        status="future",
    )
    active_new_item = UnresolvedReviewItem(
        item_id="item-4",
        reviewer="Google Gemini",
        source_round=2,
        text="Add one same-PR assertion.",
        status="same-pr",
    )
    codex_resolution = ReviewItemDisposition(
        item_id="item-1",
        reviewer="OpenAI Codex",
        disposition="resolved",
        note=None,
    )
    gemini_same_pr = ReviewItemDisposition(
        item_id="item-2",
        reviewer="Google Gemini",
        disposition="same-pr",
        note="Still needed before merge.",
    )
    codex_comment = _attach_round_metadata(
        "Codex review.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="old-sha",
            prior_items=(prior_blocking, prior_same_pr),
            dispositions=(codex_resolution,),
            new_items=(future_new_item,),
            state="approved",
        ),
    )
    gemini_comment = _attach_round_metadata(
        "Gemini review.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=2,
            subject="old-sha",
            prior_items=(prior_blocking, prior_same_pr),
            dispositions=(gemini_same_pr,),
            new_items=(active_new_item,),
            state="blocking",
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=codex_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=gemini_comment),
        ],
        head_sha="new-sha",
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert [item.item_id for item in resumed.prior_items] == ["item-2", "item-4"]
    assert resumed.prior_items[0].status == "same-pr"
    assert "Google Gemini: Still needed before merge." in resumed.prior_items[0].notes

def test_resume_pr_round_ignores_unrecorded_head_advance_with_no_active_items():
    future_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Future cleanup.",
        status="future",
    )
    review_comment = _attach_round_metadata(
        "Approved with future follow-up.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(future_item,),
            state="approved",
        ),
    )

    assert (
        _resume_pr_round(
            [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=review_comment)],
            head_sha="new-sha",
            configured_reviewers=("codex",),
        )
        is None
    )

def test_resume_pr_round_fails_early_for_incoherent_unrecorded_head_advance():
    bad_comment = _attach_round_metadata(
        "Bad metadata.\n<!-- AGENT_STATE: blocking -->\n-- Bot",
        PostedRoundMetadata(
            flow="pr",
            role="observer",
            agent="Bot",
            round_number=1,
            subject="old-sha",
        ),
    )

    with pytest.raises(
        AgentLoopError,
        match=(
            "PR head advanced without a recorded coder follow-up.*"
            "Current head: new-sha.*Latest recorded metadata subject: old-sha"
        ),
    ):
        _resume_pr_round(
            [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=bad_comment)],
            head_sha="new-sha",
            configured_reviewers=("codex",),
        )

def test_resume_pr_round_prefers_latest_metadata_ledger_for_same_head_replay():
    stale_item = UnresolvedReviewItem(
        item_id="item-3",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Stale replay item.",
        status="blocking",
        source_status="blocking",
    )
    active_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active replay item.",
        status="blocking",
        source_status="blocking",
    )
    stale_coder_comment = _attach_round_metadata(
        "Stale replay.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(stale_item,),
        ),
    )
    stale_reviewer_comment = _attach_round_metadata(
        "Still blocked."
        + prior_item_dispositions("[item-3] still blocking: stale replay")
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="abc123",
            prior_items=(stale_item,),
            dispositions=(
                parse_unresolved_item_dispositions(
                    prior_item_dispositions("[item-3] still blocking: stale replay"),
                    reviewer="OpenAI Codex",
                )[0],
            ),
            state="blocking",
        ),
    )
    active_coder_comment = _attach_round_metadata(
        "Current replay.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(active_item,),
        ),
    )
    active_reviewer_comment = _attach_round_metadata(
        "Looks good."
        + prior_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=2,
            subject="abc123",
            prior_items=(active_item,),
            dispositions=(
                parse_unresolved_item_dispositions(
                    prior_item_dispositions("[item-1] resolved"),
                    reviewer="Google Gemini",
                )[0],
            ),
            state="approved",
        ),
    )
    previous_head_comment = _attach_round_metadata(
        "Older head.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=4,
            subject="old-head",
            prior_items=(
                UnresolvedReviewItem(
                    item_id="item-9",
                    reviewer="OpenAI Codex",
                    source_round=3,
                    text="Older head item.",
                    status="blocking",
                ),
            ),
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=previous_head_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=stale_coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:02:00Z", body=stale_reviewer_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:03:00Z", body=active_coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:04:00Z", body=active_reviewer_comment),
        ],
        head_sha="abc123",
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    assert [item.item_id for item in resumed.prior_items] == ["item-1"]
    assert resumed.next_unresolved_item_number == 4
    assert [record.metadata.agent for record in resumed.completed_reviews] == ["Gemini"]

def test_pr_loop_resume_hybrid_history_prefers_metadata_ledger_over_legacy_markdown(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Add a regression test before merge.",
        status="blocking",
        source_status="blocking",
    )
    legacy_comment = (
        "Legacy raw markdown review.\n\n"
        "### Blocking issues\n"
        "- Keep the legacy fallback path.\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- OpenAI Codex"
    )
    coder_comment = _attach_round_metadata(
        "Updated the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
        ),
    )
    codex_comment = _attach_round_metadata(
        "Looks good."
        + prior_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
            dispositions=(
                parse_unresolved_item_dispositions(
                    prior_item_dispositions("[item-1] resolved"),
                    reviewer="OpenAI Codex",
                )[0],
            ),
            state="approved",
        ),
    )
    runner = FakeRunner(
        gemini_outputs=[
            "Ship it."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
        ],
        pr_payload={
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": legacy_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:06:00Z", "body": codex_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    gemini_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert "[item-1]" in gemini_prompt
    assert "Add a regression test before merge." in gemini_prompt
    assert "Keep the legacy fallback path." not in gemini_prompt


def test_pr_resume_preserves_persisted_failure_in_next_coder_metadata(tmp_path):
    from coding_review_agent_loop.local_test_evidence import (
        bounded_evidence_for_round,
        decode_bounded_evidence,
    )

    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Preserve the prior local failure.",
        status="blocking",
    )
    prior_evidence = bounded_evidence_for_round({"observations": [{
        "command": ["python", "-m", "pytest"],
        "outcome": "failed",
        "provenance": "parent-observed",
        "receipt_id": "persisted-failure",
        "turn_id": "old-turn",
        "environment": "equivalent",
        "attribution": {"state": "current-head", "head": "abc123", "stable": True, "tracked_digest": "tree-a"},
    }]})
    coder_comment = _attach_round_metadata(
        structured_coder_followup(remaining_items=["item-1"]),
        PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=1,
            subject="abc123", prior_items=(item,), local_test_evidence=prior_evidence,
            raw_structured_coder_response=structured_coder_followup(remaining_items=["item-1"]),
        ),
    )
    review_comment = _attach_round_metadata(
        structured_pr_review(
            state="blocking",
            summary="Failure remains.",
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "blocking", "note": "Fix it."}],
        ),
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=1,
            subject="abc123", prior_items=(item,), state="blocking",
            dispositions=(ReviewItemDisposition("item-1", "OpenAI Codex", "blocking", "Fix it."),),
        ),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"], tests_run=[])],
        codex_outputs=[structured_pr_review(
            state="approved",
            summary="Resolved.",
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        )],
        pr_payload={"comments": [
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:01:00Z", "body": review_comment},
        ]},
    )

    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, reviewer="codex")) == 0
    posted = next(
        comment["body"] for comment in reversed(runner.pr_payload["comments"])
        if "## Coder follow-up" in comment["body"]
    )
    match = re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", posted)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    decoded = decode_bounded_evidence(metadata.local_test_evidence)
    assert decoded is not None
    assert "persisted-failure" in [row.receipt_id for row in decoded.observations]
    preserved = next(row for row in decoded.observations if row.receipt_id == "persisted-failure")
    assert preserved.environment_state == "identity-unknown"
    assert preserved.attribution.state == "stale"

def test_pr_loop_routes_unrecorded_head_advance_through_coder_before_reviewers(tmp_path):
    old_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Preserve the metadata-backed unresolved item on rerun.",
        status="blocking",
    )
    old_coder_comment = _attach_round_metadata(
        "Opened the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="old-sha",
            prior_items=(),
        ),
    )
    old_review_comment = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(old_item,),
            state="blocking",
        ),
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                summary="Addressed the recovered prior item.",
                addressed_items=["item-2"],
                tests_run=["python -m pytest tests/test_agent_loop.py -k unrecorded_head"],
            )
        ],
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Recovered item is resolved.",
                prior_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            )
        ],
        pr_payload={
            "headRefOid": "new-sha",
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": old_coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:01:00Z", "body": old_review_comment},
            ],
        },
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    first_coder = command_index(runner.commands, ["claude"])
    first_reviewer = command_index(runner.commands, ["codex", "exec"])
    assert first_coder < first_reviewer
    reviewer_prompt = runner.commands[first_reviewer][0][-1]
    assert "[item-2]" in reviewer_prompt
    assert "Preserve the metadata-backed unresolved item on rerun." in reviewer_prompt
    posted_coder_comment = next(
        comment["body"]
        for comment in runner.pr_payload["comments"]
        if "## Coder follow-up" in comment["body"]
    )
    match = re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", posted_coder_comment)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.subject == "new-sha"
    assert metadata.round_number == 2
    assert [item.item_id for item in metadata.prior_items] == ["item-2"]

def test_pr_loop_unrecorded_head_advance_prevents_empty_ledger_unknown_item_abort(tmp_path):
    old_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Carry this item instead of starting an empty ledger.",
        status="blocking",
    )
    old_review_comment = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(old_item,),
            state="blocking",
        ),
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                summary="Classified the recovered item.",
                addressed_items=["item-2"],
            )
        ],
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Old item is resolved.",
                prior_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            )
        ],
        pr_payload={
            "headRefOid": "new-sha",
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:01:00Z", "body": old_review_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.claude_outputs == []
    assert runner.codex_outputs == []
    assert not any("unknown item" in comment.lower() for comment in runner.comments)

def test_reconcile_human_requirements_ack_item_accepts_stored_structured_coder_followup():
    human_requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )
    structured_followup = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Implemented the requested URL fix.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "human_requirement_dispositions": [
                    {
                        "requirement_id": human_requirements[0].requirement_id,
                        "disposition": "addressed",
                        "evidence": "The URL fix is implemented.",
                    }
                ],
                "human_requirements": {
                    "addressed_ids": [human_requirements[0].requirement_id],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )

    reconciled = _reconcile_human_requirements_ack_item(
        (
            UnresolvedReviewItem(
                item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
                reviewer="Orchestrator",
                source_round=1,
                text="Ack missing.",
                status="blocking",
            ),
        ),
        coder_output=structured_followup,
        human_requirements=human_requirements,
        source_round=2,
    )

    assert reconciled == []

def test_pr_loop_does_not_expose_same_round_item_ids_to_later_reviewers(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "### Same-PR follow-ups\n"
            "- Require source issue reference in PR body.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Still blocked.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("claude", "codex"),
        approved_followups="fix-and-summarize",
        max_rounds=1,
    )

    with pytest.raises(AgentLoopError, match="still reported blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)

    second_reviewer_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] and "round 1" in cmd[-1]
    ][0]
    assert "Only items listed under `Prior unresolved review items from earlier rounds`" in second_reviewer_prompt
    assert "[item-1]" not in second_reviewer_prompt
    assert "### New tracked unresolved items" not in runner.comments[0]

def test_pr_loop_same_pr_items_remain_blocking_until_explicitly_resolved(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "### Same-PR follow-ups\n"
            "- Rename the helper before merge.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex still wants the rename."
            + prior_item_dispositions("[item-1] same-pr: The caller still uses the old helper name; rename that call site too.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Tried a partial fix.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-summarize", max_rounds=2)

    with pytest.raises(AgentLoopError, match="still reported blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=config)

def test_pr_loop_resumes_with_only_missing_reviewer_for_current_head(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Codex",
        source_round=1,
        text="Add a regression test before merge.",
        status="blocking",
        source_status="blocking",
    )
    coder_comment = _attach_round_metadata(
        "Updated the PR with the requested fix.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
        ),
    )
    codex_comment = _attach_round_metadata(
        "Looks good."
        + prior_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
            dispositions=(
                parse_unresolved_item_dispositions(
                    prior_item_dispositions("[item-1] resolved"),
                    reviewer="OpenAI Codex",
                )[0],
            ),
            state="approved",
        ),
    )
    runner = FakeRunner(
        gemini_outputs=[
            "Ship it."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
        ],
        pr_payload={
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T10:00:00Z", "body": coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T10:05:00Z", "body": codex_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    agent_commands = [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"], ["gemini"])]
    assert agent_commands == ["gemini"]
    gemini_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert "[item-1]" in gemini_prompt
    assert "Add a regression test before merge." in gemini_prompt

def test_pr_loop_resume_raises_agent_loop_error_for_missing_reconstructed_prior_item(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Actual active carried item.",
        status="blocking",
        source_status="blocking",
    )
    coder_comment = _attach_round_metadata(
        "Updated the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
        ),
    )
    invalid_disposition = ReviewItemDisposition(
        item_id="item-1",
        reviewer="OpenAI Codex",
        disposition="resolved",
    )
    codex_comment = _attach_round_metadata(
        "Looks good."
        + prior_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
            dispositions=(invalid_disposition,),
            state="approved",
        ),
    )
    runner = FakeRunner(
        pr_payload={
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T10:00:00Z", "body": coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T10:05:00Z", "body": codex_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer=("codex",))

    with pytest.raises(
        AgentLoopError,
        match=r"Resumed pr round 2 reconstructed prior items item-2, but Codex dispositioned unknown item `item-1`",
    ):
        run_pr_loop(runner, pr_number=77, config=config)

@pytest.mark.parametrize(
    "line",
    [
        "[item-1] same-pr: none",
        "[item-1] still blocking: none",
        "[item-1] future follow-up: none",
    ],
)
def test_pr_loop_rejects_contradictory_disposition_before_extra_coder_round(tmp_path, line):
    runner = FakeRunner(
        codex_outputs=[
            "Needs regression coverage.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good overall."
            + prior_item_dispositions(line)
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Added coverage.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-summarize", max_rounds=3)

    with pytest.raises(AgentLoopError, match="use `resolved` when nothing remains"):
        run_pr_loop(runner, pr_number=77, config=config)

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1

def test_pr_loop_does_not_run_claude_after_final_blocking_round(tmp_path):
    runner = FakeRunner(codex_outputs=["Still blocked.\n<!-- AGENT_STATE: blocking -->"])
    config = make_config(tmp_path, max_rounds=1)

    with pytest.raises(AgentLoopError, match="still reported blocking"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

def test_pr_loop_resolves_pr_base_before_workdir_setup(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"baseRefName": "develop"},
    )
    config = make_config(
        tmp_path,
        base=None,
        reviewer="codex",
        auto_agent_dirs=("codex",),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    pr_context_index = command_index(runner.commands, ["gh", "pr", "view"])
    switch_index = command_index(runner.commands, ["git", "switch", "develop"])
    assert pr_context_index < switch_index
    assert ["git", "pull", "--ff-only", "origin", "develop"] in commands
    assert not any("origin/main" in arg for cmd in commands for arg in cmd)

def test_pr_loop_explicit_base_overrides_pr_base_without_repo_default_query(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"baseRefName": "develop"},
    )
    config = make_config(
        tmp_path,
        base="release",
        reviewer="codex",
        auto_agent_dirs=("codex",),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "switch", "release"] in commands
    assert ["git", "switch", "develop"] not in commands
    assert not any(
        cmd[:3] == ["gh", "repo", "view"] and "defaultBranchRef" in cmd
        for cmd in commands
    )

@pytest.mark.parametrize("pr_base", [None, "", "   "])
def test_pr_loop_falls_back_to_repo_default_when_pr_base_is_missing(tmp_path, pr_base):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"baseRefName": pr_base},
        repo_default_branch="develop",
    )
    config = make_config(
        tmp_path,
        base=None,
        reviewer="codex",
        auto_agent_dirs=("codex",),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    repo_query_index = command_index(runner.commands, ["gh", "repo", "view"])
    switch_index = command_index(runner.commands, ["git", "switch", "develop"])
    assert repo_query_index < switch_index


def test_pr_loop_rejects_non_open_pr_before_running_codex(tmp_path):
    runner = FakeRunner(pr_payload={
        "number": 62,
        "state": "MERGED",
        "url": "https://github.com/OWNER/REPO/pull/62",
    })
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="provide an open PR"):
        run_pr_loop(runner, pr_number=62, config=config)

def test_pr_loop_refreshes_pr_head_without_just_in_time_base_sync(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "fetch", "origin"] in commands
    assert ["git", "fetch", "origin", "+pull/77/head:refs/remotes/origin/pr/77"] in commands
    assert ["git", "switch", "main"] not in commands
    assert ["git", "pull", "--ff-only", "origin", "main"] not in commands


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


def test_pr_loop_rejects_structured_followup_outside_workdir_tests_before_posting(tmp_path):
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
                tests_run=["cd ~/llm-dialectic && python -m pytest"],
                reviewer="OpenAI Codex",
            ),
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(runner.comments) == 1
    assert runner.comments[0].startswith("**Review verdict:** Blocking")
    assert not any("Added the test." in comment for comment in runner.comments)

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
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="live remote target"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(runner.comments) == 1
    assert runner.comments[0].startswith("**Review verdict:** Blocking")
    assert not any("Added the test." in comment for comment in runner.comments)

def test_gemini_review_loop_prefers_public_response_file_over_stdout(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            "Warning: True color (24-bit) support not detected.\n"
            "YOLO mode is enabled. All tool calls will be automatically approved.\n"
            "I will fetch the PR and inspect the diff.\n"
            "Error executing tool run_shell_command: confirmation required.\n"
            "This stdout chatter should not be posted.\n",
        ],
        public_response_outputs=[
            "LGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini",
        ],
    )
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    gemini_call = next(cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert "PUBLIC RESPONSE FILE:" in gemini_call[2]
    assert str(config.gemini_dir / ".git" / "agent-loop" / "responses" / "gemini") in gemini_call[2]
    assert runner.comments == ["**Review verdict:** Approved\n\nLGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"]

def test_claude_review_loop_accepts_valid_response_file_after_nonzero_exit(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            (
                json.dumps(
                    {
                        "result": (
                            "I will inspect the PR diff.\n"
                            "Tool output chatter should not be posted.\n"
                        ),
                        "session_id": "claude-session-1",
                    }
                ),
                1,
            ),
        ],
        public_response_outputs=[
            "LGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, reviewer="claude")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    claude_call = next(cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "PUBLIC RESPONSE FILE:" in claude_call[-1]
    assert "/coding-review-agent-loop/responses/OWNER-REPO/claude/" in claude_call[-1]
    assert runner.comments == ["**Review verdict:** Approved\n\nLGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude: unknown model (medium)"]
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]) == 1
    metadata_match = re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", runner.pr_payload["comments"][0]["body"])
    assert metadata_match is not None
    metadata = _decode_round_metadata(metadata_match.group("payload"))
    assert metadata.acquisition_outcome == "accepted_nonzero_exit"
    assert metadata.acquisition_returncode == 1

def test_claude_review_loop_runs_tests_and_merge_only_after_approval(tmp_path):
    runner = FakeRunner(
        claude_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"],
    )
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer="claude",
        auto_merge=True,
        test_command=("pytest", "tests/test_agent_loop.py"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["pytest", "tests/test_agent_loop.py"] in commands
    assert ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge", "--match-head-commit", "abc123"] in commands

def test_claude_review_loop_does_not_run_codex_after_final_blocking_round(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Still blocked.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude", max_rounds=1)

    with pytest.raises(AgentLoopError, match="still reported blocking"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)

def test_claude_review_loop_rejects_non_open_pr(tmp_path):
    runner = FakeRunner(pr_payload={
        "number": 62,
        "state": "CLOSED",
        "url": "https://github.com/OWNER/REPO/pull/62",
    })
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="provide an open PR"):
        run_pr_loop(runner, pr_number=62, config=config)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)

def test_resume_pr_round_preserves_stored_model_used():
    coder_text = "Implemented the fix.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
    coder_comment = _attach_round_metadata(
        coder_text,
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="sha123",
        ),
    )
    review_text = structured_pr_review(state="approved", summary="LGTM.")
    reviewer_comment = _attach_round_metadata(
        review_text,
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="sha123",
            state="approved",
            model_used="gpt-5.5 (medium)",
        ),
    )
    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=coder_comment),
            IssueComment(author="bot", created_at="2026-05-20T09:01:00Z", body=reviewer_comment),
        ],
        head_sha="sha123",
        configured_reviewers=("codex",),
    )
    assert resumed is not None
    assert resumed.completed_reviews[0].metadata.model_used == "gpt-5.5 (medium)"

def test_run_pr_loop_freeform_coder_followup_includes_model(tmp_path):
    blocking_review_text = structured_pr_review(
        state="blocking",
        summary="Add a regression test.",
    )
    blocking_review_marker = parse_pr_review(blocking_review_text, reviewer="OpenAI Codex")
    coder_followup_text = (
        "Added the regression test.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
    )
    approved_review_text = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    )
    approved_review_marker = parse_pr_review(approved_review_text, reviewer="OpenAI Codex")

    call_count = [0]

    def fake_run_validated_agent(runner, *, agent, **kwargs):
        call_count[0] += 1
        if agent == "codex":
            if call_count[0] == 1:
                return ValidatedAgentResponse(
                    text=blocking_review_text,
                    model_used=None,
                    session_id=None,
                    marker_value=blocking_review_marker,
                )
            return ValidatedAgentResponse(
                text=approved_review_text,
                model_used=None,
                session_id=None,
                marker_value=approved_review_marker,
            )
        return ValidatedAgentResponse(
            text=coder_followup_text,
            model_used="gpt-5.5 (medium)",
            session_id=None,
            marker_value=None,
        )

    runner = FakeRunner()
    config = make_config(tmp_path, coder="claude", reviewer="codex")
    with patch(
        "coding_review_agent_loop.orchestrator._run_validated_agent",
        side_effect=fake_run_validated_agent,
    ):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    # First PR comment is the reviewer blocking; second is the coder followup
    followup_body = runner.pr_payload["comments"][1]["body"]
    stripped = _strip_round_metadata(followup_body)
    assert stripped.endswith("-- Anthropic Claude: gpt-5.5 (medium)")

def test_pr_initial_coder_post_includes_model(tmp_path):
    plan_text = "Plan content.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude"
    plan_review_text = structured_plan_review(summary="Approved.")
    plan_review_marker = parse_plan_review(plan_review_text, reviewer="OpenAI Codex")
    pr_coder_text = structured_issue_implementation(
        summary="Implemented the feature in PR 77.",
        pr_number=77,
        reviewer="Anthropic Claude",
    )
    pr_review_text = structured_pr_review(state="approved", summary="LGTM.")
    pr_review_marker = parse_pr_review(pr_review_text, reviewer="OpenAI Codex")

    call_count = [0]

    def fake_run_validated_agent(runner, *, agent, **kwargs):
        call_count[0] += 1
        if agent == "claude":
            if call_count[0] == 1:
                return ValidatedAgentResponse(
                    text=plan_text, model_used=None, session_id=None, marker_value=None
                )
            # PR host-coder call: advance git head so validate_assigned_head_advanced passes
            before_head = runner.git_head
            runner.git_head = before_head + "-coder"
            return ValidatedAgentResponse(
                text=pr_coder_text,
                model_used="gpt-5.5 (medium)",
                session_id=None,
                marker_value=validate_structured_issue_implementation(pr_coder_text),
            )
        if call_count[0] == 2:
            return ValidatedAgentResponse(
                text=plan_review_text,
                model_used=None,
                session_id=None,
                marker_value=plan_review_marker,
            )
        return ValidatedAgentResponse(
            text=pr_review_text,
            model_used=None,
            session_id=None,
            marker_value=pr_review_marker,
        )

    runner = FakeRunner(pr_payload={"body": "Fixes #56"})
    config = make_config(tmp_path, coder="claude", reviewer="codex")
    with patch(
        "coding_review_agent_loop.orchestrator._run_validated_agent",
        side_effect=fake_run_validated_agent,
    ):
        assert (
            run_issue_loop(
                runner,
                issue_number=56,
                config=config,
                plan_first=True,
                implement_after_approval=True,
            )
            == 0
        )

    # First PR comment is from the host-coder initial post
    pr_initial_body = runner.pr_payload["comments"][0]["body"]
    stripped = _strip_round_metadata(pr_initial_body)
    assert stripped.endswith("-- Anthropic Claude: gpt-5.5 (medium)")


def test_pr_loop_dispute_resolved_when_reviewer_reconsiders(tmp_path):
    """Coder disputes a blocking item; reviewer sees evidence and approves."""
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                addressed_items=[],
                remaining_items=[],
                disputed_items=["item-1"],
                dispute_evidence={"item-1": "Official docs confirm $1.50/1M tokens is correct."},
                summary="Disputing item-1: reviewer pricing claim is factually incorrect.",
            ),
        ],
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Pricing constant is wrong.",
                blocking_items=["The gemini-3.5-flash pricing constant is wrong ($0.30 not $1.50)."],
                prior_item_dispositions=[],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="approved",
                summary="Coder provided valid pricing evidence; approved.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Coder provided valid pricing evidence."},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)

    assert run_pr_loop(runner, pr_number=55, config=config) == 0

    followup_comments = [c for c in runner.comments if "## Coder follow-up" in c]
    assert len(followup_comments) == 1
    followup_body = _strip_round_metadata(followup_comments[0])
    assert "### Disputed items" in followup_body
    assert "item-1" in followup_body
    assert "Official docs confirm $1.50/1M tokens is correct." in followup_body


def test_pr_loop_releases_adopted_suppression_when_reviewer_rejects_dispute(
    tmp_path, monkeypatch
):
    """Human-decision escalation releases an invocation-owned adoption label."""
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                addressed_items=[],
                remaining_items=[],
                disputed_items=["item-1"],
                dispute_evidence={"item-1": "Official docs confirm $1.50/1M tokens is correct."},
                summary="Disputing item-1: reviewer pricing claim is factually incorrect.",
            ),
        ],
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Pricing constant is wrong.",
                blocking_items=["The gemini-3.5-flash pricing constant is wrong ($0.30 not $1.50)."],
                prior_item_dispositions=[],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="blocking",
                summary="Pricing still incorrect despite coder evidence.",
                blocking_items=["Pricing is still incorrect."],
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "blocking", "note": "I checked and the pricing is still wrong."},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "activate_managed_ci",
        lambda *_args, **_kwargs: ManagedCiContract(
            adopted_existing_pr=True,
            invocation_applied_label=True,
        ),
    )
    posted_comment_calls = []
    original_run = runner.run

    def capture_posted_comment(args, *, cwd, **kwargs):
        command = [str(arg) for arg in args]
        if command[:4] == ["gh", "pr", "comment", "55"]:
            body_path = Path(command[command.index("--body-file") + 1])
            posted_comment_calls.append((command, body_path.read_text(encoding="utf-8")))
        return original_run(args, cwd=cwd, **kwargs)

    monkeypatch.setattr(runner, "run", capture_posted_comment)
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)

    with pytest.raises(
        HumanDecisionRequiredError,
        match="Reviewer did not resolve 1 disputed item",
    ) as excinfo:
        run_pr_loop(runner, pr_number=55, config=config)

    assert "Update/evidence: Codex: I checked and the pricing is still wrong." in str(excinfo.value)
    release_command = [
        "gh",
        "api",
        "--method",
        "DELETE",
        "repos/OWNER/REPO/issues/55/labels/agent-loop-managed",
    ]
    commands = [command for command, _cwd in runner.commands]
    release_index = commands.index(release_command)
    human_decision_comment_calls = [
        (command, body)
        for command, body in posted_comment_calls
        if "## Human decision required" in body
    ]
    assert len(human_decision_comment_calls) == 1
    comment_command, human_decision_comment = human_decision_comment_calls[0]
    assert comment_command[:4] == ["gh", "pr", "comment", "55"]
    assert "--body-file" in comment_command
    assert "Reviewer did not resolve 1 disputed item(s)" in human_decision_comment
    assert "Update/evidence: Codex: I checked and the pricing is still wrong." in human_decision_comment
    assert "-- Human Reviewer" in human_decision_comment
    comment_index = commands.index(comment_command)
    assert comment_index < release_index


def test_pr_loop_escalates_when_reviewer_downgrades_disputed_item_to_same_pr(tmp_path):
    """Coder disputes a blocking item; reviewer downgrades to same-pr instead of resolving → escalate."""
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                addressed_items=[],
                remaining_items=[],
                disputed_items=["item-1"],
                dispute_evidence={"item-1": "Official docs confirm $1.50/1M tokens is correct."},
                summary="Disputing item-1: reviewer pricing claim is factually incorrect.",
            ),
        ],
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Pricing constant is wrong.",
                blocking_items=["The gemini-3.5-flash pricing constant is wrong ($0.30 not $1.50)."],
                prior_item_dispositions=[],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="blocking",
                summary="Ok I'll accept the coder's pricing evidence but still want a same-pr fix.",
                same_pr_followups=["Please add a comment citing the pricing source."],
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "same-pr", "note": "Downgraded from blocking but still needs attention."},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)

    with pytest.raises(
        AgentLoopError,
        match="Reviewer did not resolve 1 disputed item",
    ):
        run_pr_loop(runner, pr_number=55, config=config)


def test_pr_loop_dispute_note_is_visible_to_reviewer_in_next_round(tmp_path):
    """After coder disputes, the dispute evidence note appears in prior items for reviewer."""
    from coding_review_agent_loop.unresolved_items import CODER_DISPUTE_NOTE_PREFIX

    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                addressed_items=[],
                remaining_items=[],
                disputed_items=["item-1"],
                dispute_evidence={"item-1": "Evidence: price is $1.50 not $0.30."},
                summary="Disputing item-1.",
            ),
        ],
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Price is wrong.",
                blocking_items=["Price is wrong."],
                prior_item_dispositions=[],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="approved",
                summary="Approved after reviewing coder evidence.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Accepted coder evidence."},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)
    run_pr_loop(runner, pr_number=55, config=config)

    # The prior_items stored in the coder's PR comment should include dispute notes
    # runner.pr_payload["comments"] retains the raw body with AGENT_LOOP_META intact
    coder_followup_raw = next(
        (c["body"] for c in runner.pr_payload["comments"] if "## Coder follow-up" in c["body"]),
        None,
    )
    assert coder_followup_raw is not None, "Expected a coder follow-up comment in PR"
    meta_match = re.search(
        r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
        coder_followup_raw,
    )
    assert meta_match is not None, "Expected AGENT_LOOP_META in coder followup comment"
    metadata = _decode_round_metadata(meta_match.group("payload"))
    assert metadata is not None
    disputed_item = next(
        (item for item in metadata.prior_items if item.item_id == "item-1"), None
    )
    assert disputed_item is not None
    assert any(CODER_DISPUTE_NOTE_PREFIX in note for note in disputed_item.notes)


# ---------------------------------------------------------------------------
# primary-then-panel staged PR review policy (#810)
# ---------------------------------------------------------------------------


def _staged_review(*, reviewer, state="approved", blocking_items=None, dispositions=None, resolved=False):
    return (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": state,
                "summary": f"{reviewer} review",
                "blocking_items": blocking_items or [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": dispositions or [],
            }
        )
        + ("\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->" if resolved else "")
        + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
    )


def _staged_config(tmp_path, **overrides):
    values = dict(
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="primary-then-panel",
        primary_reviewer="codex",
        max_rounds=6,
    )
    values.update(overrides)
    return make_config(tmp_path, **values)


def _agent_sequence(runner):
    return [
        command[0]
        for command, _cwd in runner.commands
        if command and command[0] in {"claude", "codex", "gemini", "agy"}
    ]


def _audit_phases(runner):
    phases = []
    for comment in runner.comments:
        if comment.startswith("PR review scheduling audit:"):
            match = re.search(r"phase: ([a-z-]+); head: ([^;]+);", comment)
            assert match is not None, comment
            phases.append((match.group(1), match.group(2)))
    return phases


def _posted_scheduler_metadata(runner):
    return [
        record.metadata
        for record in orchestrator._extract_round_metadata_records(
            [
                SimpleNamespace(body=comment["body"])
                for comment in runner.pr_payload.get("comments", [])
            ],
            flow="pr",
        )
        if record.metadata.scheduler_metadata_status != "absent"
    ]


def test_staged_primary_blocking_loop_gates_panel_then_independent_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    # Primary blocks (head abc123), coder fixes, primary re-checks alone
    # (still the primary phase), then both secondaries audit the new head.
    assert _agent_sequence(runner) == ["codex", "claude", "codex", "gemini", "agy"]
    phases = _audit_phases(runner)
    assert [phase for phase, _head in phases] == ["primary", "primary", "secondary-audit"]
    assert phases[0][1] == "abc123"
    assert phases[1][1] == phases[2][1] == "abc123-coder-1"
    assert not any("final-secondary-sweep" in comment for comment in runner.comments)
    assert not any("remediation" in phase for phase, _head in phases)
    # The secondary prompts carry the independent-audit instruction and the
    # phase audit record rather than a primary-finding validation task.
    for name in ("gemini", "agy"):
        prompt = next(command[-1] for command, _cwd in runner.commands if command[:1] == [name])
        assert "Selected phase: secondary-audit" in prompt
        assert "Inspect the complete base-to-head diff independently" in prompt
        assert "do not merely validate, repeat, or\ntriage findings attributed to the primary" in prompt
    metadata = _posted_scheduler_metadata(runner)
    assert {item.scheduler_primary_reviewer for item in metadata} == {"Codex"}
    assert all(item.scheduler_force_full is False for item in metadata)
    assert metadata[-1].scheduler_calls_avoided >= 2


def test_staged_secondary_scoped_remediation_rechecks_owner_and_primary_then_sweeps_others(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    migration_calls = []
    monkeypatch.setattr(
        orchestrator,
        "validate_pr_migration_topology",
        lambda *args, **kwargs: migration_calls.append(True) or MigrationValidationResult(ok=True),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity"),
        ],
    )
    config = _staged_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sequence = _agent_sequence(runner)
    # primary -> panel (gemini, agy) -> coder -> remediation (codex + gemini)
    # -> final exact-head sweep (agy) -> migration gate.
    assert sequence[:3] == ["codex", "gemini", "agy"]
    assert sequence[3] == "claude"
    assert sorted(sequence[4:6]) == ["codex", "gemini"]
    assert sequence[6:] == ["agy"]
    assert [phase for phase, _head in _audit_phases(runner)] == [
        "primary", "secondary-audit", "remediation", "final-secondary-sweep",
    ]
    remediation_audit = next(c for c in runner.comments if "phase: remediation" in c)
    assert "active owners: Gemini" in remediation_audit
    assert "paused Antigravity" in remediation_audit
    sweep_audit = next(c for c in runner.comments if "phase: final-secondary-sweep" in c)
    assert "selected Antigravity" in sweep_audit
    # Antigravity's old-head approval never counted as final; the migration
    # gate ran only after the sweep completed.
    assert len(migration_calls) == 1
    sweep_index = runner.comments.index(sweep_audit)
    assert not any(
        "Alembic" in comment for comment in runner.comments[:sweep_index]
    )


def test_staged_owner_and_primary_clearance_does_not_transition_directly_to_ci(tmp_path, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    migration_calls = []
    monkeypatch.setattr(
        orchestrator,
        "validate_pr_migration_topology",
        lambda *args, **kwargs: migration_calls.append(True) or MigrationValidationResult(ok=True),
    )
    monkeypatch.setattr(
        orchestrator, "merge_pr", lambda *args, **kwargs: pytest.fail("merged without the sweep")
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    # Rounds: 1 primary, 2 panel, 3 remediation; the sweep would be round 4.
    config = _staged_config(tmp_path, max_rounds=3, auto_merge=True)

    with pytest.raises(AgentLoopError, match="exact-head final sweep is missing reviewer approval from Antigravity"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert migration_calls == []
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)
    assert _agent_sequence(runner).count("agy") == 1


def test_staged_broad_remediation_reactivates_complete_board(tmp_path, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("broad", "diff path 'src/other.py' is outside obligation scopes"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
    )
    config = _staged_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sequence = _agent_sequence(runner)
    assert sequence[:4] == ["codex", "gemini", "agy", "claude"]
    # Every reviewer, including the non-owner Antigravity, reviews the broad head.
    assert sorted(sequence[4:]) == ["agy", "codex", "gemini"]
    assert [phase for phase, _head in _audit_phases(runner)] == [
        "primary", "secondary-audit", "full-board",
    ]
    full_board_audit = next(c for c in runner.comments if "phase: full-board" in c)
    assert "paused none" in full_board_audit
    assert "outside obligation scopes" in full_board_audit


def test_staged_ambiguous_ownership_reactivates_complete_board(tmp_path, monkeypatch):
    from coding_review_agent_loop.review_scheduling import GitChange, classify_transition

    observed_scopes = []

    def classify_from_ledger(runner, *, checkout, previous_sha, current_sha, scopes, broad_rules, obligations):
        observed_scopes.append(tuple(scopes))
        return classify_transition(
            previous_sha, current_sha, [GitChange("src/worker.py")],
            scopes=scopes, broad_rules=broad_rules, obligations=obligations,
        )

    monkeypatch.setattr(orchestrator, "_observe_pr_transition", classify_from_ledger)
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            # No fix_scope: ownership scope is absent, so remediation is unsafe.
            _staged_review(reviewer="Google Gemini", state="blocking", blocking_items=["panel regression"]),
            _staged_review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
    )
    config = _staged_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sequence = _agent_sequence(runner)
    assert sorted(sequence[4:]) == ["agy", "codex", "gemini"]
    assert [phase for phase, _head in _audit_phases(runner)][-1] == "full-board"
    # The finding carried no reviewer scope, so the classifier had nothing to
    # match the change against and the orchestrator did not guess one.
    assert observed_scopes == [()]
    assert any("an obligation has no valid exact fix scope" in comment for comment in runner.comments)


def test_staged_force_full_selects_complete_board_from_the_primary_phase(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path, pr_review_force_full=True, max_rounds=1)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert _agent_sequence(runner) == ["codex", "gemini", "agy"]
    assert _audit_phases(runner) == [("full-board", "abc123")]
    assert "force-full: True" in runner.comments[0]
    assert all(item.scheduler_force_full is True for item in _posted_scheduler_metadata(runner))


def test_staged_legacy_scheduler_payload_before_panel_is_strict_primary_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    contract = orchestrator.make_contract(("Codex", "Gemini", "Antigravity"), "primary-then-panel", None, "Codex")
    # A pre-phase-aware record: complete legacy mandatory keys, no phase authority.
    legacy_checkpoint = _attach_round_metadata(
        "Legacy scheduler checkpoint.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=1,
            subject="abc123",
            scheduler_contract=contract.as_dict(),
            scheduler_previous_sha=None,
            scheduler_current_sha="abc123",
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex",),
            scheduler_paused_reviewers=(("Gemini", "primary phase"), ("Antigravity", "primary phase")),
            scheduler_reasons=("primary phase",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=2,
        ),
    )
    decoded = orchestrator._extract_round_metadata_records(
        [SimpleNamespace(body=legacy_checkpoint)], flow="pr"
    )[0].metadata
    assert decoded.scheduler_metadata_status == "valid"
    assert decoded.scheduler_phase is None
    runner = FakeRunner(
        pr_payload={"comments": [{"author": {"login": "bot"}, "body": legacy_checkpoint}]},
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "gap", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            _staged_review(reviewer="Google Gemini"),
            _staged_review(reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
    )
    config = _staged_config(tmp_path, quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sequence = _agent_sequence(runner)
    # #840 (row prepanel-resume-incomplete-metadata): phase-less legacy data
    # before any panel evidence re-invokes only the primary with full context
    # and never latches; the panel starts only after exact-head approval.
    assert sequence == ["codex", "claude", "codex", "gemini", "agy"]
    assert [phase for phase, _head in _audit_phases(runner)] == ["primary", "primary", "secondary-audit"]
    audits = [c for c in runner.comments if c.startswith("PR review scheduling audit:")]
    assert "strict pre-panel fallback: scheduler metadata recovery: scheduler metadata lacks phase authority" in audits[0]
    assert "primary re-invoked with full context" in audits[1]
    assert "post-panel fallback" not in "".join(runner.comments)
    posted = _posted_scheduler_metadata(runner)
    assert posted[0].scheduler_force_full is False  # the seeded legacy record
    assert all(item.scheduler_force_full is False for item in posted[1:])
    assert all(item.scheduler_force_full_source is None for item in posted)
    assert "durable force-full latch" not in capsys.readouterr().err


def test_staged_legacy_selective_payload_is_contract_drift(tmp_path):
    contract = orchestrator.make_contract(("Codex", "Gemini"), "selective-intermediate", None)
    checkpoint = _attach_round_metadata(
        "Selective checkpoint.",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=1, subject="abc123",
            scheduler_contract=contract.as_dict(), scheduler_previous_sha=None,
            scheduler_current_sha="abc123", scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex", "Gemini"), scheduler_reasons=("full board",),
            scheduler_final_sweep=False, scheduler_force_full=False, scheduler_calls_avoided=0,
        ),
    )
    runner = FakeRunner(pr_payload={"comments": [{"author": {"login": "bot"}, "body": checkpoint}]})
    config = _staged_config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_sequence(runner) == []


def test_staged_primary_drift_is_contract_drift(tmp_path):
    contract = orchestrator.make_contract(("Codex", "Gemini"), "primary-then-panel", None, "Gemini")
    checkpoint = _attach_round_metadata(
        "Staged checkpoint.",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=1, subject="abc123",
            scheduler_contract=contract.as_dict(), scheduler_previous_sha=None,
            scheduler_current_sha="abc123", scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Gemini",), scheduler_paused_reviewers=(("Codex", "primary phase"),),
            scheduler_reasons=("primary phase",), scheduler_final_sweep=False,
            scheduler_force_full=False, scheduler_calls_avoided=1, scheduler_phase="primary",
            scheduler_primary_reviewer="Gemini",
        ),
    )
    runner = FakeRunner(pr_payload={"comments": [{"author": {"login": "bot"}, "body": checkpoint}]})
    config = _staged_config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
        run_pr_loop(runner, pr_number=77, config=config)


def _unavailable(reviewer):
    return json.dumps(
        {
            "schema_version": 1,
            "kind": "agent_unavailable",
            "retryable": False,
            "category": "provider",
            "summary": "The reviewer provider is unavailable.",
            "suggested_action": "Retry later.",
        }
    ) + f"\n<!-- AGENT_UNAVAILABLE -->\n-- {reviewer}"


def test_staged_primary_failure_is_not_approval_and_resume_reuses_no_work(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_unavailable("OpenAI Codex"), _staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path)

    with pytest.raises(AgentLoopError, match="missing required input from Codex"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_sequence(runner) == ["codex"]
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)

    # Resume from the durable comments: the primary is dispatched again, and
    # only after its approval does the panel run.
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _agent_sequence(runner) == ["codex", "codex", "gemini", "agy"]
    assert [phase for phase, _head in _audit_phases(runner)] == ["primary", "primary", "secondary-audit"]


def test_staged_secondary_failure_keeps_healthy_result_and_resume_dispatches_only_outstanding(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_unavailable("Antigravity"), _staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path, auto_merge=True)
    merges = []

    with pytest.raises(AgentLoopError, match="missing required input from Antigravity"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_sequence(runner) == ["codex", "gemini", "agy"]
    assert any("Healthy reviewers approved" in comment or "Google Gemini review" in comment for comment in runner.comments)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)

    import coding_review_agent_loop.orchestrator as module

    original_merge = module.merge_pr
    module.merge_pr = lambda *args, **kwargs: merges.append(kwargs)
    try:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    finally:
        module.merge_pr = original_merge
    # Completed exact-head work (Codex, Gemini) is reused once; only the
    # outstanding secondary is dispatched.
    assert _agent_sequence(runner) == ["codex", "gemini", "agy", "agy"]
    assert merges == [{"expected_head_sha": "abc123"}]


def test_staged_resume_after_remediation_checkpoint_dispatches_final_sweep_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    # Stop right after the remediation round settles (round 3 of 3).
    first_config = _staged_config(tmp_path, max_rounds=3)
    with pytest.raises(AgentLoopError, match="missing reviewer approval from Antigravity"):
        run_pr_loop(runner, pr_number=77, config=first_config)
    assert _agent_sequence(runner).count("agy") == 1

    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity"))
    resumed_config = _staged_config(tmp_path, max_rounds=6)
    assert run_pr_loop(runner, pr_number=77, config=resumed_config) == 0
    sequence = _agent_sequence(runner)
    # Codex and Gemini exact-head approvals are reused; only Antigravity sweeps.
    assert sequence.count("codex") == 2
    assert sequence.count("gemini") == 2
    assert sequence.count("agy") == 2
    assert sequence[-1] == "agy"
    assert [phase for phase, _head in _audit_phases(runner)][-1] == "final-secondary-sweep"


def test_staged_external_head_change_invalidates_primary_approval_before_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("broad", "external change"),
    )
    runner = FakeRunner(
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex"),
        ],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    original = orchestrator._run_validated_agent
    calls = []

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "reviewer" and kwargs.get("agent") == "codex":
            calls.append(True)
            if len(calls) == 1:
                runner.pr_payload["headRefOid"] = "external-head"
        return response

    monkeypatch.setattr(orchestrator, "_run_validated_agent", run)
    config = _staged_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sequence = _agent_sequence(runner)
    # The stale primary approval on abc123 never opens the panel: the primary
    # must approve the external head first, then the secondaries audit it.
    assert sequence == ["codex", "codex", "gemini", "agy"]
    phases = _audit_phases(runner)
    assert [phase for phase, _head in phases][-1] == "secondary-audit"
    assert phases[-1][1] == "external-head"
    assert all(head == "external-head" for _phase, head in phases[1:])


def test_staged_head_change_during_final_sweep_invalidates_partial_sweep_evidence(tmp_path, monkeypatch):
    from coding_review_agent_loop.review_scheduling import GitChange, classify_transition

    def classify_from_ledger(runner, *, checkout, previous_sha, current_sha, scopes, broad_rules, obligations):
        # The scoped remediation touches only the owned path; the external
        # push after the sweep has no obligation scope to match against.
        return classify_transition(
            previous_sha, current_sha, [GitChange("src/worker.py")],
            scopes=scopes, broad_rules=broad_rules, obligations=obligations,
        )

    monkeypatch.setattr(orchestrator, "_observe_pr_transition", classify_from_ledger)
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
            _staged_review(reviewer="OpenAI Codex"),
        ],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
            _staged_review(reviewer="Google Gemini"),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity"),
        ],
    )
    original = orchestrator._run_validated_agent
    agy_calls = []

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "reviewer" and kwargs.get("agent") == "antigravity":
            agy_calls.append(True)
            if len(agy_calls) == 2:
                # An external actor pushes while the final sweep is settling.
                runner.pr_payload["headRefOid"] = "pushed-during-sweep"
        return response

    monkeypatch.setattr(orchestrator, "_run_validated_agent", run)
    config = _staged_config(tmp_path, max_rounds=8)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    phases = _audit_phases(runner)
    # The sweep on the old head settles, but the pushed head has no obligation
    # scope to classify against, so every reviewer (including the primary and
    # the secondaries that had just approved) re-reviews the complete diff.
    # No partial sweep approval carries across the mutation.
    assert phases == [
        ("primary", "abc123"),
        ("secondary-audit", "abc123"),
        ("remediation", "abc123-coder-1"),
        ("final-secondary-sweep", "abc123-coder-1"),
        ("full-board", "pushed-during-sweep"),
    ]
    sequence = _agent_sequence(runner)
    assert sequence == [
        "codex", "gemini", "agy", "claude", "codex", "gemini", "agy", "codex", "gemini", "agy",
    ]
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands[: -1])


def test_staged_exact_head_panel_gate_stops_before_migration_and_merge(tmp_path, monkeypatch):
    migration_calls = []
    monkeypatch.setattr(
        orchestrator,
        "validate_pr_migration_topology",
        lambda *args, **kwargs: migration_calls.append(True) or MigrationValidationResult(ok=True),
    )
    monkeypatch.setattr(
        orchestrator, "merge_pr", lambda *args, **kwargs: pytest.fail("merged on primary approval alone")
    )
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path, max_rounds=1, auto_merge=True)

    with pytest.raises(AgentLoopError, match="exact-head final sweep is missing reviewer approval from Gemini, Antigravity"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert migration_calls == []
    assert _agent_sequence(runner) == ["codex"]
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_staged_unacknowledged_human_requirement_remains_blocking_after_panel_settlement(
    tmp_path, monkeypatch
):
    requirement = HumanReviewRequirement(
        source_type="PR comment",
        author="maintainer",
        created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="Keep the audit trail intact.",
    )
    original_context = orchestrator.get_pr_review_context

    def context_with_requirement(runner, *args, **kwargs):
        context = original_context(runner, *args, **kwargs)
        return dataclasses.replace(context, human_requirements=(requirement,))

    monkeypatch.setattr(orchestrator, "get_pr_review_context", context_with_requirement)
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex", resolved=True)],
        # Gemini approves without acknowledging the signed requirement.
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity", resolved=True)],
    )
    config = _staged_config(tmp_path, max_rounds=2, auto_merge=True)
    monkeypatch.setattr(
        orchestrator, "merge_pr", lambda *args, **kwargs: pytest.fail("merged with an unacknowledged requirement")
    )

    with pytest.raises(AgentLoopError, match=r"blocking issues after round 2.*Orchestrator \(item-1\)"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert _agent_sequence(runner)[:3] == ["codex", "gemini", "agy"]
    assert any("phase: secondary-audit" in comment for comment in runner.comments)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_staged_ci_failure_remains_blocking_after_unanimous_panel_approval(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
        pr_check_runs_payload={
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]
        },
    )
    config = _staged_config(tmp_path, max_rounds=2, auto_merge=True)
    monkeypatch.setattr(
        orchestrator, "merge_pr", lambda *args, **kwargs: pytest.fail("merged with CI failure")
    )

    with pytest.raises(AgentLoopError, match="blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert _agent_sequence(runner) == ["codex", "gemini", "agy"]
    assert any("GitHub PR checks are failing for PR #77." in comment for comment in runner.comments)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_staged_policy_does_not_change_existing_policy_selection(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=1)
    assert config.pr_review_policy == "all-reviewers"
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _agent_sequence(runner) == ["codex", "gemini"]
    assert not any(comment.startswith("PR review scheduling audit:") for comment in runner.comments)
    assert _posted_scheduler_metadata(runner) == []


# ---------------------------------------------------------------------------
# #840: strict primary-only pre-panel phase for primary-then-panel
# ---------------------------------------------------------------------------


def _rewrite_pr_metadata(runner, transform):
    """Rewrite persisted round-metadata payloads in place (legacy simulation).

    ``transform`` receives the decoded payload mapping and returns a new
    mapping, or ``None`` to leave the comment unchanged.
    """
    from coding_review_agent_loop.round_transport import (
        ROUND_RESUME_MARKER_RE,
        decode_mapping,
        encode_mapping,
    )

    for comment in runner.pr_payload.get("comments", []):
        body = comment["body"]
        matches = list(ROUND_RESUME_MARKER_RE.finditer(body))
        if not matches:
            continue
        match = matches[-1]
        payload = transform(dict(decode_mapping(match.group("payload"))))
        if payload is None:
            continue
        comment["body"] = (
            body[: match.start("payload")] + encode_mapping(payload) + body[match.end("payload"):]
        )


def _legacyize_operator_latches(payload):
    """Turn operator-attributed latches into pre-#840 unattributed latches."""
    if payload.get("scheduler_force_full_source") is None:
        return None
    payload = dict(payload)
    del payload["scheduler_force_full_source"]
    return payload


def _scheduling_audits(runner):
    return [c for c in runner.comments if c.startswith("PR review scheduling audit:")]


def _scheduling_diagnostics(runner):
    return [c for c in runner.comments if c.startswith("PR review scheduling diagnostic")]


def _comment_index(runner, predicate, *, start=0):
    return next(index for index, comment in enumerate(runner.comments) if index >= start and predicate(comment))


@pytest.mark.parametrize(
    "fix_scope, changed_path, expected_reason",
    [
        (["pyproject.toml"], "pyproject.toml", "matches a broad rule"),
        (["src/worker.py"], "src/other.py", "is outside obligation scopes"),
    ],
)
def test_840_broad_or_out_of_scope_change_before_primary_approval_is_primary_only(
    tmp_path, monkeypatch, fix_scope, changed_path, expected_reason
):
    """Row prepanel-broad-change / prepanel-ambiguous-scope (fresh run)."""
    from coding_review_agent_loop.review_scheduling import GitChange, classify_transition

    def classify(runner, *, checkout, previous_sha, current_sha, scopes, broad_rules, obligations):
        return classify_transition(
            previous_sha, current_sha, [GitChange(changed_path)],
            scopes=scopes, broad_rules=broad_rules, obligations=obligations,
        )

    monkeypatch.setattr(orchestrator, "_observe_pr_transition", classify)
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "worker cleanup gap", "fix_scope": fix_scope}],
            ),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    # No secondary is invoked before the primary approves the exact head.
    assert _agent_sequence(runner) == ["codex", "claude", "codex", "gemini", "agy"]
    assert _audit_phases(runner) == [
        ("primary", "abc123"),
        ("primary", "abc123-coder-1"),
        ("secondary-audit", "abc123-coder-1"),
    ]
    strict_audit = _scheduling_audits(runner)[1]
    assert "reason: strict pre-panel fallback: " in strict_audit
    assert expected_reason in strict_audit
    assert "primary re-invoked with full context" in strict_audit
    assert "force-full: False (source: none)" in strict_audit
    codex_prompts = [command[-1] for command, _cwd in runner.commands if command[:1] == ["codex"]]
    assert "Scheduling reason: strict pre-panel fallback" in codex_prompts[1]
    assert "Inspect the complete base-to-head diff independently" in codex_prompts[1]
    metadata = _posted_scheduler_metadata(runner)
    assert all(item.scheduler_force_full is False for item in metadata)
    assert all(item.scheduler_force_full_source is None for item in metadata)
    assert not any("post-panel fallback" in comment for comment in runner.comments)


def test_840_panel_starts_after_primary_approval_and_post_panel_broad_change_latches_automatic(
    tmp_path, monkeypatch
):
    """Row panel-start-and-final-barrier: narrow, then broad, then narrow fixes after the panel opens."""
    transitions = iter(
        [
            TransitionClassification("narrow", "scoped fix"),
            TransitionClassification("broad", "diff path 'pyproject.toml' matches a broad rule"),
            TransitionClassification("narrow", "scoped fix"),
        ]
    )
    monkeypatch.setattr(orchestrator, "_observe_pr_transition", lambda *args, **kwargs: next(transitions))

    def resolved(*item_ids):
        return [{"item_id": item_id, "disposition": "resolved"} for item_id in item_ids]

    def finding(text):
        return [{"text": text, "fix_scope": ["src/worker.py"]}]

    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"]),
            structured_coder_followup(addressed_items=["item-2"]),
            structured_coder_followup(addressed_items=["item-3"]),
        ],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=resolved("item-1")),
            _staged_review(reviewer="OpenAI Codex", dispositions=resolved("item-2")),
            _staged_review(reviewer="OpenAI Codex", dispositions=resolved("item-3")),
        ],
        gemini_outputs=[
            _staged_review(reviewer="Google Gemini", state="blocking", blocking_items=finding("panel regression")),
            _staged_review(
                reviewer="Google Gemini", state="blocking",
                dispositions=resolved("item-1"), blocking_items=finding("second regression"),
            ),
            _staged_review(
                reviewer="Google Gemini", state="blocking",
                dispositions=resolved("item-2"), blocking_items=finding("third regression"),
            ),
            _staged_review(reviewer="Google Gemini", dispositions=resolved("item-3")),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity", dispositions=resolved("item-2")),
            _staged_review(reviewer="Antigravity", dispositions=resolved("item-3")),
        ],
    )
    merges = []
    monkeypatch.setattr(orchestrator, "merge_pr", lambda *args, **kwargs: merges.append(kwargs))
    config = _staged_config(tmp_path, max_rounds=10, auto_merge=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    phases = [phase for phase, _head in _audit_phases(runner)]
    # primary -> panel -> owner+primary remediation (narrow) -> full board
    # (broad) -> full board again on the later narrow head (monotonic latch).
    assert phases == ["primary", "secondary-audit", "remediation", "full-board", "full-board"]
    audits = _scheduling_audits(runner)
    assert "post-panel fallback: narrow remediation" in audits[2]
    assert "force-full: False (source: none)" in audits[2]
    assert "reason: post-panel fallback: full board required: diff path 'pyproject.toml'" in audits[3]
    assert "force-full: True (source: automatic)" in audits[3]
    assert "reason: post-panel fallback: force-full latch" in audits[4]
    assert "force-full: True (source: automatic)" in audits[4]
    assert "paused none" in audits[4]
    assert not any("strict pre-panel fallback" in audit for audit in audits[1:])
    sequence = _agent_sequence(runner)
    assert sequence[:3] == ["codex", "gemini", "agy"]
    assert sorted(sequence[-3:]) == ["agy", "codex", "gemini"]
    # The durable latch is persisted on every record from the broad head's
    # prelaunch on, including the coder checkpoint for the following narrow
    # head.  (A coder checkpoint carries the next round number but the state
    # of the round that produced it.)
    metadata = _posted_scheduler_metadata(runner)
    first_latched = next(
        index for index, item in enumerate(metadata)
        if item.phase == "scheduler-prelaunch" and item.scheduler_phase == "full-board"
    )
    assert all(
        item.scheduler_force_full is True and item.scheduler_force_full_source == "automatic"
        for item in metadata[first_latched:]
    )
    assert all(item.scheduler_force_full is False for item in metadata[:first_latched])
    assert any(item.role == "coder" for item in metadata[first_latched:])
    # Completion requires the exact final head approved by every reviewer.
    final_head = _audit_phases(runner)[-1][1]
    final_approvals = {
        record.metadata.agent
        for record in orchestrator._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
        )
        if record.metadata.role == "reviewer"
        and record.metadata.subject == final_head
        and record.metadata.state == "approved"
    }
    assert final_approvals == {"Codex", "Gemini", "Antigravity"}
    assert merges == [{"expected_head_sha": final_head}]


def test_840_post_panel_automatic_fallback_latches_with_automatic_source(tmp_path, monkeypatch):
    """An automatic recovery reason after a qualified opening keeps the durable latch."""
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
    )
    # The architecture identity observed for the coder's head is stale, which
    # is an automatic full-board reason raised after the qualified opening.
    stale_observations = []

    def observation(comments, *, head_sha):
        if head_sha == "abc123-coder-1" and not stale_observations:
            stale_observations.append(head_sha)
            return {"stale": "identity"}
        return None

    monkeypatch.setattr(orchestrator, "_latest_pr_architecture_observation", observation)
    config = _staged_config(tmp_path, max_rounds=6)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    phases = [phase for phase, _head in _audit_phases(runner)]
    assert phases[:2] == ["primary", "secondary-audit"]
    assert phases[2] == "full-board"
    full_board = _scheduling_audits(runner)[2]
    assert "post-panel fallback: force-full latch" in full_board
    assert "force-full: True (source: automatic)" in full_board
    latched = [
        item for item in _posted_scheduler_metadata(runner) if item.scheduler_force_full
    ]
    assert latched and all(item.scheduler_force_full_source == "automatic" for item in latched)


def test_840_resumed_primary_phase_with_incomplete_metadata_is_primary_only(tmp_path, monkeypatch):
    """Row prepanel-resume-incomplete-metadata: absent, invalid, phase-less, contradictory."""
    variants = {
        "absent": lambda payload: {
            key: value for key, value in payload.items() if not key.startswith("scheduler_")
        },
        "invalid": lambda payload: (
            {**payload, "scheduler_obligation_digest": "not-a-digest"}
            if "scheduler_contract" in payload else None
        ),
        "phase-less": lambda payload: (
            {
                key: value
                for key, value in payload.items()
                if key not in {"scheduler_phase", "scheduler_primary_reviewer"}
            }
            if "scheduler_contract" in payload else None
        ),
        "contradictory": lambda payload: (
            {**payload, "scheduler_current_sha": "some-other-head"}
            if "scheduler_contract" in payload else None
        ),
    }
    for name, transform in variants.items():
        variant_path = tmp_path / name
        variant_path.mkdir()
        runner = FakeRunner(
            codex_outputs=[
                _staged_review(
                    reviewer="OpenAI Codex",
                    state="blocking",
                    blocking_items=[{"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}],
                ),
            ],
        )
        # The first invocation stops after the primary's blocking review.
        with pytest.raises(AgentLoopError):
            run_pr_loop(runner, pr_number=77, config=_staged_config(variant_path, max_rounds=1))
        assert _agent_sequence(runner) == ["codex"], name
        _rewrite_pr_metadata(runner, transform)
        runner.claude_outputs.append(structured_coder_followup(addressed_items=["item-1"]))
        runner.codex_outputs.append(
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}])
        )
        runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini"))
        runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity"))
        comments_before = len(runner.comments)

        if name in {"invalid", "contradictory"}:
            # The pre-existing qualification guard still refuses malformed or
            # contradictory history, but only after the strictly primary-only
            # phase: no secondary was spent on the uncertainty.
            with pytest.raises(AgentLoopError, match="scheduler (head )?metadata was observed during qualification"):
                run_pr_loop(runner, pr_number=77, config=_staged_config(variant_path))
            assert set(_agent_sequence(runner)) == {"codex", "claude"}, name
            new_audits = [
                c for c in runner.comments[comments_before:] if c.startswith("PR review scheduling audit:")
            ]
            assert new_audits and all("phase: primary" in audit for audit in new_audits)
            assert "strict pre-panel fallback: scheduler metadata recovery" in new_audits[0]
            assert not any(item.scheduler_force_full for item in _posted_scheduler_metadata(runner))
            continue

        assert run_pr_loop(runner, pr_number=77, config=_staged_config(variant_path)) == 0, name

        sequence = _agent_sequence(runner)[1:]
        first_secondary = min(
            index for index, agent in enumerate(sequence) if agent in {"gemini", "agy"}
        )
        # Only the primary (and coder) run until the primary approves the exact head.
        assert set(sequence[:first_secondary]) <= {"codex", "claude"}, (name, sequence)
        assert sequence[first_secondary - 1] == "codex", (name, sequence)
        new_audits = [
            c for c in runner.comments[comments_before:] if c.startswith("PR review scheduling audit:")
        ]
        assert all("phase: primary" in audit for audit in new_audits[:-1]), name
        assert "phase: secondary-audit" in new_audits[-1], name
        posted = _posted_scheduler_metadata(runner)
        assert not any(item.scheduler_force_full for item in posted), name
        assert not any("post-panel fallback" in c for c in runner.comments), name


def _seed_premature_full_board_round(tmp_path, *, gemini_output, parallel=False):
    """Simulate a pre-#840 run: an automatic pre-approval full board on abc123.

    The operator flag produces the full board; the operator attribution is then
    stripped so the history matches the unattributed latch pre-#840 runs wrote.
    Antigravity becomes unavailable, so the round is interrupted.
    """
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[gemini_output],
        antigravity_outputs=[_unavailable("Antigravity")],
    )
    config = _staged_config(tmp_path, pr_review_force_full=True, review_parallel=parallel)
    # An approving round stops on the unavailable reviewer; a blocking round
    # stops when the (unscripted) coder follow-up cannot run.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert sorted(_agent_sequence(runner)[:3]) == ["agy", "codex", "gemini"]
    _rewrite_pr_metadata(runner, _legacyize_operator_latches)
    assert not any(item.scheduler_force_full_source for item in _posted_scheduler_metadata(runner))
    return runner


@pytest.mark.parametrize("parallel", [False, True])
def test_840_premature_secondary_approvals_are_not_resumed_carried_or_counted(tmp_path, parallel):
    """Row premature-secondary-approval-not-carried (same-round and historical)."""
    runner = _seed_premature_full_board_round(
        tmp_path, gemini_output=_staged_review(reviewer="Google Gemini"), parallel=parallel
    )
    seeded = len(runner.comments)
    if parallel:
        # Parallel reviewers publish completion-order records.
        records = orchestrator._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
        )
        assert any(
            record.metadata.agent == "Gemini" and record.metadata.phase == "publication"
            for record in records
        )
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini"))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity"))
    merges = []
    import coding_review_agent_loop.orchestrator as module

    original_merge = module.merge_pr
    module.merge_pr = lambda *args, **kwargs: merges.append(kwargs)
    try:
        assert run_pr_loop(
            runner, pr_number=77, config=_staged_config(tmp_path, auto_merge=True, review_parallel=parallel)
        ) == 0
    finally:
        module.merge_pr = original_merge

    new_agents = _agent_sequence(runner)[3:]
    # The primary's own same-round approval is resumed; every secondary gets a
    # fresh post-opening turn, including Gemini whose premature approval exists.
    assert sorted(new_agents) == ["agy", "gemini"]
    new_comments = runner.comments[seeded:]
    audits = [c for c in new_comments if c.startswith("PR review scheduling audit:")]
    assert len(audits) == 1
    assert "phase: secondary-audit" in audits[0]
    assert "selected Gemini, Antigravity" in audits[0]
    opening = _comment_index(runner, lambda c: c.startswith("PR review scheduling audit:"), start=seeded)
    gemini_review = _comment_index(runner, lambda c: "Google Gemini review" in c, start=seeded)
    assert gemini_review > opening
    posted = _posted_scheduler_metadata(runner)
    opening_record = next(item for item in posted if item.phase == "scheduler-prelaunch" and item.scheduler_phase == "secondary-audit")
    # The premature Gemini approval never entered the opening's approval set.
    assert opening_record.scheduler_approved_reviewers == ("Codex",)
    assert merges == [{"expected_head_sha": "abc123"}]


def test_840_premature_historical_secondary_approvals_are_not_carried(tmp_path):
    """Premature approvals from an earlier round on the same head are not carried."""
    runner = FakeRunner(
        codex_outputs=[_unavailable("OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    with pytest.raises(AgentLoopError, match="Codex"):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True))
    _rewrite_pr_metadata(runner, _legacyize_operator_latches)
    seeded = len(runner.comments)
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex"))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini"))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity"))

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    # Round 1 resumes: premature same-round secondary approvals are dropped and
    # the primary reviews alone.  Round 2 (same head): the round-1 secondary
    # approvals are older-round exact-head approvals, yet neither is carried.
    assert _agent_sequence(runner)[3:] == ["codex", "gemini", "agy"]
    audits = [c for c in runner.comments[seeded:] if c.startswith("PR review scheduling audit:")]
    assert [re.search(r"phase: ([a-z-]+);", audit).group(1) for audit in audits] == [
        "primary", "secondary-audit",
    ]
    assert "unqualified pre-approval panel history was ignored" in audits[0]
    assert "rerun with --pr-review-force-full to restore the complete board" in audits[0]
    assert "selected Gemini, Antigravity" in audits[1]
    assert not any("skipping Gemini; it approved unchanged" in c for c in runner.comments)


def test_840_blocking_premature_secondary_completion_stops_with_diagnostic(tmp_path):
    """Rows premature-panel-history-resume / prepanel-unsafe-stop (resume path)."""
    runner = _seed_premature_full_board_round(
        tmp_path,
        gemini_output=_staged_review(
            reviewer="Google Gemini",
            state="blocking",
            blocking_items=[{"text": "premature cache race", "fix_scope": ["src/worker.py"]}],
        ),
    )
    seeded_agents = len(_agent_sequence(runner))
    seeded_comments = len(runner.pr_payload["comments"])

    with pytest.raises(orchestrator.PrePanelSafetyError, match="--pr-review-force-full"):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))

    # No reviewer or coder ran, nothing was latched, and the only new comment
    # is the plain diagnostic (no scheduler metadata).
    assert len(_agent_sequence(runner)) == seeded_agents
    new_comments = runner.pr_payload["comments"][seeded_comments:]
    assert len(new_comments) == 1
    assert new_comments[0]["body"].startswith("PR review scheduling diagnostic (round 1): pre-panel safety")
    assert "AGENT_LOOP_META" not in new_comments[0]["body"]
    assert "Gemini (round 1, head abc123, state blocking" in new_comments[0]["body"]


def test_840_operator_override_supersedes_blocking_premature_completion(tmp_path):
    """Row premature-blocking-completion-operator-recovery (k3 with the flag)."""
    runner = _seed_premature_full_board_round(
        tmp_path,
        gemini_output=_staged_review(
            reviewer="Google Gemini",
            state="blocking",
            blocking_items=[{"text": "premature cache race", "fix_scope": ["src/worker.py"]}],
        ),
    )
    seeded_agents = len(_agent_sequence(runner))
    seeded = len(runner.comments)
    premature_item_ids = {
        item.item_id
        for record in orchestrator._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
        )
        if record.metadata.agent == "Gemini"
        for item in record.metadata.new_items
    }
    assert premature_item_ids == {"item-1"}
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex"))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini"))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity"))

    assert run_pr_loop(
        runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True)
    ) == 0

    # Every configured reviewer is freshly invoked after the operator opening;
    # none is resumed or carried.
    assert sorted(_agent_sequence(runner)[seeded_agents:]) == ["agy", "codex", "gemini"]
    opening = _comment_index(runner, lambda c: c.startswith("PR review scheduling audit:"), start=seeded)
    opening_text = runner.comments[opening]
    assert "phase: full-board" in opening_text
    assert "force-full: True (source: operator)" in opening_text
    assert "Superseded premature panel reviews" in opening_text
    assert "Gemini (round 1, head abc123, state blocking; items: item-1)" in opening_text
    for marker in ("OpenAI Codex review", "Google Gemini review", "Antigravity review"):
        assert _comment_index(runner, lambda c, marker=marker: marker in c, start=seeded) > opening
    gemini_prompt = [command[-1] for command, _cwd in runner.commands if command[:1] == ["gemini"]][-1]
    assert "Superseded pre-panel review context (non-authoritative; context only):" in gemini_prompt
    assert "- Earlier claim: premature cache race" in gemini_prompt
    codex_prompt = [command[-1] for command, _cwd in runner.commands if command[:1] == ["codex"]][-1]
    assert "Superseded pre-panel review context" not in codex_prompt
    # The premature item never entered the ledger or ownership accounting.
    later_records = [
        record
        for record in orchestrator._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
        )
        if record.index >= len(runner.pr_payload["comments"]) - (len(runner.comments) - seeded)
    ]
    for record in later_records:
        assert not ({item.item_id for item in record.metadata.prior_items} & premature_item_ids)
        assert not ({item.item_id for item in record.metadata.new_items} & premature_item_ids)
    operator_records = [item for item in _posted_scheduler_metadata(runner) if item.scheduler_force_full_source == "operator"]
    assert operator_records


@pytest.mark.parametrize("operator", [False, True])
def test_840_stale_identity_premature_blocking_completion_is_not_silently_dropped(tmp_path, operator):
    """Row premature-blocking-completion-operator-recovery with a stale resume identity.

    A signed requirement surfaced after the premature blocking review makes
    that record ineligible for resume.  It must still reach the diagnostic
    (without the flag) or the audited, context-preserving supersession (with
    it) instead of being dropped before panel classification.
    """
    runner = _seed_premature_full_board_round(
        tmp_path,
        gemini_output=_staged_review(
            reviewer="Google Gemini",
            state="blocking",
            blocking_items=[{"text": "premature cache race", "fix_scope": ["src/worker.py"]}],
        ),
    )
    runner.pr_payload["comments"].append(
        {
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-18T10:00:00Z",
            "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-840",
            "body": "Keep the worker cleanup idempotent.\n\n-- Human Reviewer",
        }
    )
    premature = next(
        record
        for record in orchestrator._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
        )
        if record.metadata.agent == "Gemini"
    )
    from coding_review_agent_loop.github import _parse_pr_human_requirements

    requirements = _parse_pr_human_requirements(runner.pr_payload)
    assert requirements
    assert not orchestrator._resumed_pr_reviewer_matches_requirements(premature, requirements)
    seeded_agents = len(_agent_sequence(runner))
    seeded = len(runner.comments)

    if not operator:
        with pytest.raises(orchestrator.PrePanelSafetyError, match="--pr-review-force-full"):
            run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
        assert len(_agent_sequence(runner)) == seeded_agents
        new_comments = runner.comments[seeded:]
        assert len(new_comments) == 1
        assert new_comments[0].startswith("PR review scheduling diagnostic (round 1): pre-panel safety")
        assert "Gemini (round 1, head abc123, state blocking; items: item-1)" in new_comments[0]
        return

    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", resolved=True))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini", resolved=True))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", resolved=True))

    assert run_pr_loop(
        runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True)
    ) == 0

    assert sorted(_agent_sequence(runner)[seeded_agents:]) == ["agy", "codex", "gemini"]
    opening = _comment_index(runner, lambda c: c.startswith("PR review scheduling audit:"), start=seeded)
    opening_text = runner.comments[opening]
    assert "force-full: True (source: operator)" in opening_text
    assert "Superseded premature panel reviews" in opening_text
    assert "Gemini (round 1, head abc123, state blocking; items: item-1)" in opening_text
    assert _comment_index(runner, lambda c: "Google Gemini review" in c, start=seeded) > opening
    gemini_prompt = [command[-1] for command, _cwd in runner.commands if command[:1] == ["gemini"]][-1]
    assert "Superseded pre-panel review context (non-authoritative; context only):" in gemini_prompt
    assert "- Earlier claim: premature cache race" in gemini_prompt


def test_840_operator_force_full_is_durable_across_resume(tmp_path):
    """Row operator-force-full: the latch survives a resume that omits the flag."""
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_unavailable("Antigravity"), _staged_review(reviewer="Antigravity")],
    )
    with pytest.raises(AgentLoopError, match="Antigravity"):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True))
    assert _agent_sequence(runner) == ["codex", "gemini", "agy"]
    first_audit = _scheduling_audits(runner)[0]
    assert "reason: operator force-full" in first_audit
    assert "force-full: True (source: operator)" in first_audit

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    # Completed post-opening work is reused; the operator latch is not
    # downgraded to a primary-only round on resume.
    assert _agent_sequence(runner) == ["codex", "gemini", "agy", "agy"]
    audits = _scheduling_audits(runner)
    assert "phase: full-board" in audits[-1]
    assert "force-full: True (source: operator)" in audits[-1]
    assert all(
        item.scheduler_force_full is True and item.scheduler_force_full_source == "operator"
        for item in _posted_scheduler_metadata(runner)
    )


def test_840_secondary_owned_ledger_item_before_panel_stops_unless_operator(tmp_path, monkeypatch):
    """Row premature-panel-history-resume: a premature secondary finding reached the ledger."""
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "premature cache race", "fix_scope": ["src/worker.py"]}],
            ),
        ],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    # Round 1 is a pre-#840 style full board; the coder then fixes the item and
    # the run stops when round 2's first reviewer has no scripted output.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True))
    assert _agent_sequence(runner)[:4] == ["codex", "gemini", "agy", "claude"]
    _rewrite_pr_metadata(runner, _legacyize_operator_latches)
    seeded_agents = len(_agent_sequence(runner))

    with pytest.raises(orchestrator.PrePanelSafetyError) as raised:
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
    message = str(raised.value)
    assert "item-1" in message and "Gemini" in message
    assert "--pr-review-force-full" in message
    assert len(_agent_sequence(runner)) == seeded_agents
    assert _scheduling_diagnostics(runner)

    # The operator override authorizes the complete board instead.  (The fake
    # runner would otherwise treat the replayed coder JSON in reviewer prompts
    # as a new coder push.)
    runner.advance_pr_head_on_coder_followup = False
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", dispositions=resolved))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini", dispositions=resolved))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", dispositions=resolved))
    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True)) == 0
    assert sorted(_agent_sequence(runner)[seeded_agents:]) == ["agy", "codex", "gemini"]
    assert "force-full: True (source: operator)" in _scheduling_audits(runner)[-1]


def test_840_premature_approved_secondary_history_resumes_as_strict_primary_turn(tmp_path, monkeypatch):
    """Row premature-panel-history-resume variant: premature secondaries approved."""
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}],
            ),
        ],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True))
    assert _agent_sequence(runner)[:4] == ["codex", "gemini", "agy", "claude"]
    _rewrite_pr_metadata(runner, _legacyize_operator_latches)
    seeded_agents = len(_agent_sequence(runner))
    seeded = len(runner.comments)
    runner.codex_outputs.append(
        _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}])
    )
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini"))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity"))

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    assert _agent_sequence(runner)[seeded_agents:] == ["codex", "gemini", "agy"]
    audits = [c for c in runner.comments[seeded:] if c.startswith("PR review scheduling audit:")]
    assert "phase: primary" in audits[0]
    assert "strict pre-panel fallback" in audits[0]
    assert "unqualified pre-approval panel history was ignored" in audits[0]
    assert "phase: secondary-audit" in audits[1]
    assert not any("post-panel fallback" in c for c in runner.comments[seeded:])
    new_metadata = _posted_scheduler_metadata(runner)[-4:]
    assert not any(item.scheduler_force_full for item in new_metadata)


def test_840_machine_obligation_before_panel_is_primary_only_then_panel_opens(tmp_path, monkeypatch):
    """Rows prepanel-machine-obligation and prepanel-primary-approved-with-machine-obligation."""
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
        pr_check_runs_payload={
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]
        },
    )
    monkeypatch.setattr(
        orchestrator, "merge_pr", lambda *args, **kwargs: pytest.fail("merged with CI failure")
    )
    config = _staged_config(tmp_path, max_rounds=2, auto_merge=True)

    with pytest.raises(AgentLoopError, match="blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=config)

    # The primary approved the exact head, so the panel opened on it even
    # though the CI obligation remains; no diagnostic was raised.
    assert _agent_sequence(runner) == ["codex", "gemini", "agy"]
    assert not _scheduling_diagnostics(runner)
    audits = _scheduling_audits(runner)
    assert "phase: secondary-audit" in audits[1]

    # A resume derives qualified panel evidence from that prelaunch record.
    records = orchestrator._extract_round_metadata_records(
        [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
    )
    evidence = orchestrator._derive_pr_panel_evidence(
        records, primary_reviewer="Codex", required_reviewers=("Codex", "Gemini", "Antigravity")
    )
    assert evidence.opened and evidence.opening_source == "primary-approval"


def test_840_ci_obligation_before_primary_approval_does_not_invoke_panel(tmp_path, monkeypatch):
    """Row prepanel-machine-obligation: CI item active while the primary lacks approval."""
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}],
            ),
        ],
        pr_check_runs_payload={
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, max_rounds=3))
    # Round 2 only re-invokes the primary; no diagnostic, no secondary.
    assert "gemini" not in _agent_sequence(runner)
    assert "agy" not in _agent_sequence(runner)
    assert not _scheduling_diagnostics(runner)
    assert all("phase: primary" in audit for audit in _scheduling_audits(runner))
    assert not any(item.scheduler_force_full for item in _posted_scheduler_metadata(runner))


def test_840_architecture_identity_change_before_panel_is_primary_only(tmp_path, monkeypatch):
    """Row prepanel-checkpoint-or-architecture-invalidation."""
    stale_observations = []

    def observation(comments, *, head_sha):
        if not stale_observations:
            stale_observations.append(head_sha)
            return {"stale": "identity"}
        return None

    monkeypatch.setattr(orchestrator, "_latest_pr_architecture_observation", observation)
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    assert _agent_sequence(runner) == ["codex", "gemini", "agy"]
    audits = _scheduling_audits(runner)
    assert "phase: primary" in audits[0]
    assert "strict pre-panel fallback: architecture identity changed" in audits[0]
    assert "phase: secondary-audit" in audits[1]
    assert not any(item.scheduler_force_full for item in _posted_scheduler_metadata(runner))


def test_840_legacy_latch_after_qualified_opening_is_honored(tmp_path, monkeypatch):
    """Row legacy-latch-compatibility: a post-opening unattributed latch keeps the full board."""
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
        ],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    # primary -> panel -> coder; the remediation round then runs out of output.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
    assert _agent_sequence(runner)[:4] == ["codex", "gemini", "agy", "claude"]

    def latch_after_opening(payload):
        # Pretend the coder checkpoint (after the qualified opening) latched
        # the board under pre-#840 code, without an attribution source.
        if payload.get("role") != "coder" or "scheduler_contract" not in payload:
            return None
        return {**payload, "scheduler_force_full": True}

    _rewrite_pr_metadata(runner, latch_after_opening)
    seeded_agents = len(_agent_sequence(runner))
    runner.advance_pr_head_on_coder_followup = False
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", dispositions=resolved))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini", dispositions=resolved))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", dispositions=resolved))

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    assert sorted(_agent_sequence(runner)[seeded_agents:]) == ["agy", "codex", "gemini"]
    audit = _scheduling_audits(runner)[-1]
    assert "phase: full-board" in audit
    assert "post-panel fallback: force-full latch" in audit
    assert "force-full: True (source: automatic)" in audit


@pytest.mark.parametrize("operator", [False, True])
def test_840_round_boundary_undecodable_history_stops_with_diagnostic(tmp_path, monkeypatch, operator):
    """Row prepanel-unsafe-stop: panel state is unknowable when history cannot be decoded."""
    state = {"broken": False}
    original_extract = orchestrator._extract_round_metadata_records
    original_mergeability = orchestrator.get_pr_mergeability

    def extract(comments, *, flow):
        if state["broken"]:
            raise AgentLoopError("Incomplete round metadata: sidecars are unavailable")
        return original_extract(comments, flow=flow)

    def mergeability(*args, **kwargs):
        # History becomes undecodable after startup, at the round boundary.
        state["broken"] = True
        return original_mergeability(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_extract_round_metadata_records", extract)
    monkeypatch.setattr(orchestrator, "get_pr_mergeability", mergeability)
    monkeypatch.setattr(orchestrator, "_round_ledger_may_be_incomplete", lambda **kwargs: False)
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path, pr_review_force_full=operator, max_rounds=1)

    # Resume, approval, ledger, and qualification accounting all depend on the
    # history, so neither flag state may spend reviewers over an undecodable
    # ledger: both stop through the same no-reviewer diagnostic.
    with pytest.raises(orchestrator.PrePanelSafetyError, match="could not be decoded"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_sequence(runner) == []
    diagnostics = _scheduling_diagnostics(runner)
    assert len(diagnostics) == 1
    assert diagnostics[0].startswith("PR review scheduling diagnostic (round 1): ")
    assert "Restore the missing round-metadata records or sidecars" in diagnostics[0]
    assert "--pr-review-force-full cannot authorize" in diagnostics[0]
    assert not _scheduling_audits(runner)


@pytest.mark.parametrize("operator", [False, True])
def test_840_startup_undecodable_history_stops_with_diagnostic(tmp_path, operator):
    """Row prepanel-unsafe-stop at startup: a real missing sidecar reference."""
    from coding_review_agent_loop.round_transport import (
        ROUND_RESUME_MARKER_RE,
        decode_mapping,
        encode_mapping,
    )

    contract = orchestrator.make_contract(("Codex", "Gemini", "Antigravity"), "primary-then-panel", None, "Codex")
    checkpoint = _attach_round_metadata(
        "Staged checkpoint.",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=1, subject="abc123",
            scheduler_contract=contract.as_dict(), scheduler_previous_sha=None,
            scheduler_current_sha="abc123", scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex",),
            scheduler_paused_reviewers=(("Gemini", "primary phase"), ("Antigravity", "primary phase")),
            scheduler_reasons=("primary phase",), scheduler_final_sweep=False,
            scheduler_force_full=False, scheduler_calls_avoided=2, scheduler_phase="primary",
            scheduler_primary_reviewer="Codex",
        ),
    )
    match = ROUND_RESUME_MARKER_RE.search(checkpoint)
    payload = decode_mapping(match.group("payload"))
    # Reference a spilled field whose sidecar comment does not exist.
    payload["canonical_reviewer_response"] = {
        "$round_transport_spill": "missing-anchor",
        "parts": 1,
        "sha256": "0" * 64,
        "spill": "0" * 64,
    }
    broken = checkpoint[: match.start("payload")] + encode_mapping(payload) + checkpoint[match.end("payload"):]
    with pytest.raises(AgentLoopError, match="sidecars are unavailable"):
        orchestrator._extract_round_metadata_records([SimpleNamespace(body=broken)], flow="pr")
    runner = FakeRunner(
        pr_payload={"comments": [{"author": {"login": "bot"}, "body": broken}]},
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )

    with pytest.raises(orchestrator.PrePanelSafetyError, match="could not be decoded") as raised:
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=operator))

    assert "sidecars are unavailable" in str(raised.value)
    assert _agent_sequence(runner) == []
    diagnostics = _scheduling_diagnostics(runner)
    assert len(diagnostics) == 1
    assert diagnostics[0].startswith("PR review scheduling diagnostic (startup): pre-panel safety")
    assert "--pr-review-force-full cannot authorize" in diagnostics[0]
    # The diagnostic carries no round metadata that could be mistaken for a checkpoint.
    assert "AGENT_LOOP_META" not in runner.pr_payload["comments"][-1]["body"]
    assert not _scheduling_audits(runner)


def test_840_startup_undecodable_history_keeps_legacy_error_for_other_policies(tmp_path):
    """Row other-policies-unchanged: non-staged policies still raise the decode error."""
    from coding_review_agent_loop.round_transport import (
        ROUND_RESUME_MARKER_RE,
        decode_mapping,
        encode_mapping,
    )

    body = _attach_round_metadata(
        "Reviewer record.",
        PostedRoundMetadata(flow="pr", role="reviewer", agent="Codex", round_number=1, subject="abc123", state="approved"),
    )
    match = ROUND_RESUME_MARKER_RE.search(body)
    payload = decode_mapping(match.group("payload"))
    payload["canonical_reviewer_response"] = {
        "$round_transport_spill": "missing-anchor", "parts": 1, "sha256": "0" * 64, "spill": "0" * 64,
    }
    broken = body[: match.start("payload")] + encode_mapping(payload) + body[match.end("payload"):]
    runner = FakeRunner(pr_payload={"comments": [{"author": {"login": "bot"}, "body": broken}]})
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), pr_review_policy="selective-intermediate"
    )
    with pytest.raises(AgentLoopError, match="sidecars are unavailable") as raised:
        run_pr_loop(runner, pr_number=77, config=config)
    assert not isinstance(raised.value, orchestrator.PrePanelSafetyError)
    assert not _scheduling_diagnostics(runner)


# --- Visible sidecar labels on the PR posting seam (#842) --------------------


def _oversized_pr_round_body(role: str):
    import os

    from coding_review_agent_loop.protocol_markers import TrustedBody
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata

    text = _attach_round_metadata(
        "Visible PR round response",
        PostedRoundMetadata(
            flow="pr",
            role=role,
            agent="claude",
            round_number=2,
            subject="sidecar-label-pr-seam",
            canonical_reviewer_response=base64.urlsafe_b64encode(os.urandom(70_000)).decode("ascii"),
        ),
    )
    return TrustedBody.canonical(text, expected_tokens=("AGENT_LOOP_META",))


def _post_oversized_pr_round(monkeypatch, role: str) -> list[str]:
    import coding_review_agent_loop.github as github_module

    monkeypatch.setattr(github_module, "active_workdir", lambda config: None)
    posted: list[str] = []

    class _Runner:
        def run(self, args, *, cwd, input_text=None, check=True, env=None):
            posted.append(Path(args[args.index("--body-file") + 1]).read_text())
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    github_module.post_pr_comment(
        _Runner(),
        config=SimpleNamespace(quiet=True, dry_run=False, gh_cmd="gh", repo="owner/repo"),
        pr_number=857,
        body=_oversized_pr_round_body(role),
    )
    return posted


def test_pr_reviewer_round_sidecars_use_review_wording_before_anchor(monkeypatch):
    from coding_review_agent_loop.round_transport import (
        ROUND_RESUME_MARKER_RE,
        ROUND_TRANSPORT_SIDECAR_RE,
        is_round_transport_sidecar,
    )

    posted = _post_oversized_pr_round(monkeypatch, "reviewer")

    sidecars, anchor = posted[:-1], posted[-1]
    assert sidecars
    for position, body in enumerate(sidecars, start=1):
        assert body.startswith(f"Agent-loop review attachment {position}/{len(sidecars)} ")
        assert body.endswith(ROUND_TRANSPORT_SIDECAR_RE.search(body).group(0))
        assert "see the following review comment." in body
        payload = json.loads(
            base64.urlsafe_b64decode(ROUND_TRANSPORT_SIDECAR_RE.search(body).group("payload"))
        )
        assert "kind" not in payload
    assert not is_round_transport_sidecar(anchor)
    assert ROUND_RESUME_MARKER_RE.search(anchor)


def test_pr_coder_response_sidecars_use_neutral_wording(monkeypatch):
    posted = _post_oversized_pr_round(monkeypatch, "coder")

    sidecars = posted[:-1]
    assert sidecars
    for body in sidecars:
        assert body.startswith("Agent-loop attachment ")
        assert "review attachment" not in body and "plan attachment" not in body


# ---------------------------------------------------------------------------
# #862: resolved-history proof for incomplete-ledger disposition strips
# ---------------------------------------------------------------------------

from coding_review_agent_loop.protocol import ReviewItemDisposition as _Disp862
from coding_review_agent_loop.protocol import UnresolvedReviewItem as _Item862
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata as _Meta862,
    PostedRoundRecord as _Record862,
    _canonically_resolved_history_item_ids,
)


def _item862(item_id="item-1", reviewer="Codex", owners=(), status="blocking", **extra):
    return _Item862(
        item_id=item_id,
        reviewer=reviewer,
        source_round=1,
        text="worker cleanup gap",
        status=status,
        resolution_owners=owners,
        **extra,
    )


def _record862(
    index,
    *,
    subject,
    round_number,
    role="reviewer",
    agent="Codex",
    prior_items=(),
    dispositions=(),
    new_items=(),
    flow="pr",
):
    return _Record862(
        index=index,
        metadata=_Meta862(
            flow=flow,
            role=role,
            agent=agent,
            round_number=round_number,
            subject=subject,
            prior_items=tuple(prior_items),
            dispositions=tuple(dispositions),
            new_items=tuple(new_items),
        ),
        body="",
    )


def _raise_then_resolve_records(*, dispositions, prior=None):
    item = prior or _item862()
    return [
        _record862(0, subject="h1", round_number=1, new_items=[item]),
        _record862(
            1, subject="h2", round_number=2, prior_items=[item], dispositions=dispositions
        ),
    ]


@pytest.mark.parametrize("mode", ["aggregate", "owner-scoped"])
def test_resolved_history_includes_single_owner_raise_then_resolve(mode):
    records = _raise_then_resolve_records(
        dispositions=[_Disp862("item-1", "Codex", "resolved")]
    )
    assert _canonically_resolved_history_item_ids(
        records, reconciliation_mode=mode, same_status="same-pr"
    ) == frozenset({"item-1"})


@pytest.mark.parametrize("mode", ["aggregate", "owner-scoped"])
@pytest.mark.parametrize(
    "order",
    [("blocking", "resolved"), ("resolved", "blocking")],
)
def test_resolved_history_excludes_same_round_conflict_in_both_orders(mode, order):
    item = _item862()
    records = [
        _record862(0, subject="h1", round_number=1, new_items=[item]),
        _record862(
            1, subject="h2", round_number=2, agent="Codex", prior_items=[item],
            dispositions=[_Disp862("item-1", "Codex", order[0], note="n")],
        ),
        _record862(
            2, subject="h2", round_number=2, agent="Gemini", prior_items=[item],
            dispositions=[_Disp862("item-1", "Gemini", order[1], note="n")],
        ),
    ]
    assert "item-1" not in _canonically_resolved_history_item_ids(
        records, reconciliation_mode=mode, same_status="same-pr"
    )


def test_resolved_history_excludes_owner_scoped_non_owner_only_resolution():
    records = _raise_then_resolve_records(
        prior=_item862(owners=("Codex", "Gemini")),
        dispositions=[_Disp862("item-1", "Antigravity", "resolved")],
    )
    assert _canonically_resolved_history_item_ids(
        records, reconciliation_mode="owner-scoped", same_status="same-pr"
    ) == frozenset()


def test_resolved_history_excludes_owner_scoped_partial_multi_owner_clearance():
    records = _raise_then_resolve_records(
        prior=_item862(owners=("Codex", "Gemini")),
        dispositions=[_Disp862("item-1", "Codex", "resolved")],
    )
    assert _canonically_resolved_history_item_ids(
        records, reconciliation_mode="owner-scoped", same_status="same-pr"
    ) == frozenset()


def test_resolved_history_excludes_item_reintroduced_after_resolution():
    records = _raise_then_resolve_records(
        dispositions=[_Disp862("item-1", "Codex", "resolved")]
    )
    records.append(
        _record862(2, subject="h3", round_number=3, agent="Gemini", new_items=[_item862()])
    )
    assert _canonically_resolved_history_item_ids(
        records, reconciliation_mode="aggregate", same_status="same-pr"
    ) == frozenset()


def test_resolved_history_excludes_item_recarried_after_resolution():
    records = _raise_then_resolve_records(
        dispositions=[_Disp862("item-1", "Codex", "resolved")]
    )
    records.append(
        _record862(2, subject="h3", round_number=3, prior_items=[_item862()])
    )
    assert _canonically_resolved_history_item_ids(
        records, reconciliation_mode="aggregate", same_status="same-pr"
    ) == frozenset()


def test_resolved_history_excludes_future_machine_other_flow_and_carried_items():
    future = _raise_then_resolve_records(
        prior=_item862("item-2"),
        dispositions=[_Disp862("item-2", "Codex", "future")],
    )
    assert _canonically_resolved_history_item_ids(
        future, reconciliation_mode="aggregate", same_status="same-pr"
    ) == frozenset()

    machine = _item862(
        "item-3",
        reviewer="GitHub managed exact-head CI",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="repair_required",
        failed_head_sha="h1",
    )
    machine_records = _raise_then_resolve_records(
        prior=machine,
        dispositions=[_Disp862("item-3", "Codex", "resolved")],
    )
    assert _canonically_resolved_history_item_ids(
        machine_records, reconciliation_mode="owner-scoped", same_status="same-pr"
    ) == frozenset()

    resolved = _raise_then_resolve_records(
        dispositions=[_Disp862("item-1", "Codex", "resolved")]
    )
    # Other-flow comments never decode into this flow's records.
    plan_records = orchestrator._extract_round_metadata_records(
        [
            SimpleNamespace(
                body=orchestrator._attach_round_metadata(
                    "plan review",
                    dataclasses.replace(record.metadata, flow="plan"),
                )
            )
            for record in resolved
        ],
        flow="pr",
    )
    assert plan_records == ()
    assert _canonically_resolved_history_item_ids(
        resolved,
        reconciliation_mode="aggregate",
        same_status="same-pr",
        current_carried_ids=("item-1",),
    ) == frozenset()


def test_staged_secondary_audit_strips_resolved_primary_disposition_under_incomplete_ledger(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    ledger_flags = []
    real_ledger_check = orchestrator._round_ledger_may_be_incomplete

    def ledger_spy(**kwargs):
        result = real_ledger_check(**kwargs)
        ledger_flags.append(result)
        return result

    monkeypatch.setattr(orchestrator, "_round_ledger_may_be_incomplete", ledger_spy)
    logged = []
    real_log = orchestrator.log
    monkeypatch.setattr(
        orchestrator,
        "log",
        lambda config, message, *a, **k: (logged.append(message), real_log(config, message, *a, **k))[1],
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        # The secondary audit repeats the primary's already-cleared item.
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    repair_mock.assert_not_called()
    assert _agent_sequence(runner) == ["codex", "claude", "codex", "gemini", "agy"]
    assert [phase for phase, _head in _audit_phases(runner)] == [
        "primary", "primary", "secondary-audit",
    ]
    # The secondary-audit round is exactly the incomplete-ledger state.
    assert ledger_flags[-1] is True
    assert any(
        "removed canonically resolved historical prior-item disposition ID(s) item-1 "
        "despite incomplete ledger" in message
        for message in logged
    )
    assert not any(
        "deterministically removed unknown prior-item" in message for message in logged
    )
    # The stripped response keeps its approval and posts no disposition.
    gemini_record = next(
        record
        for record in orchestrator._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload.get("comments", [])],
            flow="pr",
        )
        if record.metadata.agent == "Gemini" and record.metadata.role == "reviewer"
    )
    assert gemini_record.metadata.state == "approved"
    assert gemini_record.metadata.dispositions == ()
    assert gemini_record.metadata.new_items == ()


# --- Issue #871: a narration-only PR reviewer is unavailable, not blocking ---

_NARRATION_ONLY_PR_REVIEW = (
    "I have launched the test command in the background and will wait for it "
    "to complete.\nterminating 1 background task(s) on exit"
)


def _fabricated_pr_review():
    return structured_pr_review(
        state="blocking",
        summary="PR review incomplete: the test command was terminated.",
        blocking_items=["PR review incomplete: the test command was terminated."],
        reviewer="OpenAI Codex",
    )


def test_pr_loop_records_narration_only_reviewer_as_unavailable(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_NARRATION_ONLY_PR_REVIEW] * 4,
        claude_outputs=[
            structured_pr_review(
                state="approved",
                summary="The diff is correct.",
                reviewer="Anthropic Claude",
            )
        ],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        max_rounds=1,
        agent_max_retries=0,
        agent_retry_backoff_seconds=0,
    )

    with patch(
        "coding_review_agent_loop.orchestrator.attempt_repair",
        lambda raw, gemini_cmd, **kwargs: _fabricated_pr_review(),
    ):
        with pytest.raises(AgentLoopError, match="missing required input from Codex"):
            run_pr_loop(runner, pr_number=77, config=config)

    assert "**Review status: Incomplete**" in runner.comments[-1]
    assert not any("PR review incomplete" in body for body in runner.comments)
    # No coder follow-up is started from a synthesized finding, and nothing merges.
    assert not any(cmd[:1] == ["gemini"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


def test_pr_loop_single_narration_only_reviewer_stops_fatally(tmp_path):
    runner = FakeRunner(codex_outputs=[_NARRATION_ONLY_PR_REVIEW] * 4)
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        max_rounds=1,
        agent_max_retries=0,
        agent_retry_backoff_seconds=0,
    )

    with patch(
        "coding_review_agent_loop.orchestrator.attempt_repair",
        lambda raw, gemini_cmd, **kwargs: _fabricated_pr_review(),
    ):
        with pytest.raises(AgentLoopError, match="review_substance_integrity"):
            run_pr_loop(runner, pr_number=77, config=config)

    assert not any("PR review incomplete" in body for body in runner.comments)
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


def _plan_comment(plan: str) -> IssueComment:
    return IssueComment(
        author="coding-review-agent-loop",
        created_at="2026-05-01T00:00:00Z",
        body=_attach_round_metadata(
            plan,
            PostedRoundMetadata(
                flow="plan", role="coder", agent="Claude", round_number=1,
                subject=orchestrator._plan_subject(plan), canonical_plan=plan,
                raw_structured_coder_response=plan,
            ),
        ),
    )


def _handoff_comment(plan_hash: str) -> IssueComment:
    return IssueComment(
        author="coding-review-agent-loop",
        created_at="2026-05-01T00:02:00Z",
        body=format_issue_pr_handoff_comment(
            issue_number=56, pr_number=77,
            pr_url="https://github.com/OWNER/REPO/pull/77",
            pr_head_sha="abc123", flow="approved-plan-implementation",
            plan_hash=plan_hash,
        ),
    )


def _issue(number: int, comments: tuple[IssueComment, ...]) -> IssueContext:
    return IssueContext(
        number=number, repo="OWNER/REPO", title=f"Issue {number}",
        body="Scope.", url=f"https://github.com/OWNER/REPO/issues/{number}",
        comments=comments,
    )


def _managed_resume_runner() -> FakeRunner:
    return FakeRunner(
        pr_payload={
            "headRefName": "agent-loop/managed-56", "headRefOid": "abc123",
            "baseRefName": "main", "body": "Fixes #56",
        },
    )


def _ordinary_managed_config(tmp_path, **overrides):
    return make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",), **overrides,
    )


def _issue_created_handoff() -> orchestrator.AuthenticatedIssueCreatedHandoff:
    return orchestrator.AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha="abc123", branch="agent-loop/managed-56",
        trusted_actor_login="agent-loop", trusted_actor_id=1,
        protection_mode="voluntary", override_nonce="opening-nonce",
    )


def test_pr_ordinary_resume_without_parent_context_still_fails_closed(tmp_path, monkeypatch):
    """Plain PR-mode resume has no parent identity and keeps the hard stop."""
    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    child = _issue(56, (_handoff_comment(orchestrator.approved_plan_hash(plan)),))
    monkeypatch.setattr(orchestrator, "validate_open_issue", lambda *_a, **_k: None)
    monkeypatch.setattr(
        orchestrator, "get_issue_context", lambda *_a, **_k: child
    )
    monkeypatch.setattr(
        orchestrator, "recover_issue_created_handoff",
        lambda *_a, **_k: _issue_created_handoff(),
    )
    monkeypatch.setattr(
        orchestrator, "revalidate_issue_created_handoff",
        lambda *_a, **_k: (_ for _ in ()).throw(_FreshScopeCaptured()),
    )
    runner = _managed_resume_runner()

    with pytest.raises(
        AgentLoopError,
        match="Managed-CI ordinary resume could not recover the canonical approved plan.",
    ):
        run_pr_loop(runner, pr_number=77, config=_ordinary_managed_config(tmp_path))

    assert not any(command[:1] in (["claude"], ["codex"]) for command, _cwd in runner.commands)


def test_pr_ordinary_resume_ignores_parent_when_child_plan_recovers(tmp_path, monkeypatch):
    """A recoverable child plan wins and the parent comments are never read."""
    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    decoy = "Decoy plan.\n\n### Plan steps\n1. Do something else entirely."
    plan_hash = orchestrator.approved_plan_hash(plan)
    child = _issue(56, (_plan_comment(plan), _handoff_comment(plan_hash)))
    parent = _issue(55, (_plan_comment(decoy),))
    read_comments: list[int] = []

    real_recover = orchestrator.recover_approved_plan_context

    def recording_recover(comments, **kwargs):
        read_comments.append(len(comments))
        return real_recover(comments, **kwargs)

    monkeypatch.setattr(orchestrator, "recover_approved_plan_context", recording_recover)
    monkeypatch.setattr(orchestrator, "validate_open_issue", lambda *_a, **_k: None)
    monkeypatch.setattr(
        orchestrator, "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    monkeypatch.setattr(
        orchestrator, "recover_issue_created_handoff",
        lambda *_a, **_k: _issue_created_handoff(),
    )
    captured: dict[str, object] = {}

    def revalidate(*_args, **kwargs):
        captured.update(kwargs)
        raise _FreshScopeCaptured

    monkeypatch.setattr(orchestrator, "revalidate_issue_created_handoff", revalidate)
    runner = _managed_resume_runner()

    with pytest.raises(_FreshScopeCaptured):
        run_pr_loop(
            runner, pr_number=77, config=_ordinary_managed_config(tmp_path),
            parent_issue_context=parent,
        )

    assert captured["handoff"].approved_plan_hash == plan_hash
    # Only the child comments were consulted by the managed-CI recovery.
    assert read_comments == [len(child.comments)]


def test_pr_fresh_authorization_recovers_parent_held_plan(tmp_path, monkeypatch):
    """A staged child under --managed-ci-fresh recovers the parent-held plan."""
    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    plan_hash = orchestrator.approved_plan_hash(plan)
    child = _issue(56, (_handoff_comment(plan_hash),))
    parent = _issue(55, (_plan_comment(plan),))
    monkeypatch.setattr(orchestrator, "validate_open_issue", lambda *_a, **_k: None)
    monkeypatch.setattr(
        orchestrator, "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    captured: dict[str, object] = {}

    def authorize(*_args, **kwargs):
        captured.update(kwargs)
        raise _FreshScopeCaptured

    monkeypatch.setattr(orchestrator, "authorize_fresh_issue_created_resume", authorize)
    config = _ordinary_managed_config(
        tmp_path, managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
    )
    runner = _managed_resume_runner()

    with pytest.raises(_FreshScopeCaptured):
        run_pr_loop(
            runner, pr_number=77, config=config, parent_issue_context=parent,
        )

    assert captured["approved_plan_hash"] == plan_hash
    assert not any(command[:1] in (["claude"], ["codex"]) for command, _cwd in runner.commands)


def test_pr_fresh_authorization_without_parent_context_still_fails_closed(tmp_path, monkeypatch):
    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    child = _issue(56, (_handoff_comment(orchestrator.approved_plan_hash(plan)),))
    monkeypatch.setattr(orchestrator, "validate_open_issue", lambda *_a, **_k: None)
    monkeypatch.setattr(orchestrator, "get_issue_context", lambda *_a, **_k: child)
    monkeypatch.setattr(
        orchestrator, "authorize_fresh_issue_created_resume",
        lambda *_a, **_k: (_ for _ in ()).throw(_FreshScopeCaptured()),
    )
    config = _ordinary_managed_config(
        tmp_path, managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
    )

    with pytest.raises(
        AgentLoopError,
        match="could not recover the canonical approved plan for the explicit issue scope",
    ):
        run_pr_loop(_managed_resume_runner(), pr_number=77, config=config)


# --- #959: delta-rendered coder matrix evidence ----------------------------

from coding_review_agent_loop.round_transport import (  # noqa: E402
    decode_mapping,
    encode_mapping,
)

_EVIDENCE_HEADING_959 = "### Risk-based mode and transition test matrix evidence"


def _coder_records_959(runner):
    records = []
    for item in runner.pr_payload.get("comments", []):
        body = item.get("body") if isinstance(item, dict) else None
        if not isinstance(body, str) or "AGENT_LOOP_META: " not in body:
            continue
        metadata = orchestrator._decode_round_metadata(
            body.split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
        )
        if metadata.role == "coder":
            records.append((body, metadata))
    return records


def _blocking_reviews_959(count):
    reviews = [
        structured_pr_review(state="blocking", blocking_items=["Exercise the follow-up row."])
    ]
    for _ in range(count - 1):
        reviews.append(structured_pr_review(
            state="blocking",
            prior_item_dispositions=[{
                "item_id": "item-1", "disposition": "blocking", "note": "Still open.",
            }],
        ))
    reviews.append(structured_pr_review(
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    ))
    return reviews


def test_959_in_process_coder_rounds_render_full_then_deltas_with_stable_anchor(tmp_path):
    plan_context = _followup_matrix_context()
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"], summary=f"Round {n}.")
            for n in (1, 2, 3)
        ],
        codex_outputs=_blocking_reviews_959(3),
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=4)
    spy_calls = []
    real_resolve = orchestrator.resolve_matrix_evidence_render

    def spy(current, previous, current_round):
        decision = real_resolve(current, previous, current_round)
        spy_calls.append((previous, current_round, decision))
        return decision

    with patch.object(orchestrator, "resolve_matrix_evidence_render", spy):
        assert run_pr_loop(
            runner, pr_number=77, config=config, approved_plan_context=plan_context
        ) == 0

    records = _coder_records_959(runner)
    assert len(records) == 3
    (first_body, first), (second_body, second), (third_body, third) = records
    # The establishing round renders every row and anchors to its own record.
    assert first.risk_test_matrix_evidence_full_round == first.round_number
    assert first.risk_test_matrix_evidence_full_round_status == "valid"
    assert "<summary>Full matrix evidence (1 row)</summary>" in first_body
    assert "**followup-derived-evidence**" in first_body
    assert [call[1] for call in spy_calls] == [
        first.round_number, second.round_number, third.round_number
    ]
    # The in-process record reused as previous metadata is valid by construction.
    assert spy_calls[1][0].risk_test_matrix_evidence_full_round_status == "valid"
    for body, record, previous in ((second_body, second, first), (third_body, third, second)):
        assert record.risk_test_matrix_evidence_full_round == first.round_number
        assert record.risk_test_matrix_evidence is not None
        assert len(record.risk_test_matrix_evidence["rows"]) == 1
        assert "**followup-derived-evidence**" not in body
        assert (
            f"1 row unchanged since round {previous.round_number}; "
            f"full matrix in round {first.round_number}."
        ) in body
        assert "<details>" in body and body.index(_EVIDENCE_HEADING_959) < body.index("<details>")


def _seeded_resume_runner_959(plan_context, *, anchor_mutation=None, full_round=None):
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Exercise follow-up evidence derivation.",
        status="blocking",
        source_status="blocking",
    )
    raw_coder = structured_coder_followup(addressed_items=["item-1"], summary="Seeded.")
    parsed_coder = validate_structured_coder_followup(raw_coder)
    evidence = {
        "matrix_identity": plan_context.risk_test_matrix_identity,
        "rows": [{
            "row_id": "followup-derived-evidence",
            "status": "missing",
            "test_identifiers": [],
            "test_locations": [],
            "workflow_path_claim": "Seeded legacy evidence.",
            "outcome_assertions": [],
            "forbidden_effect_assertions": [],
            "evidence_citations": [],
            "caveats": [],
        }],
    }
    extra = {}
    if full_round is not None:
        extra["risk_test_matrix_evidence_full_round"] = full_round
    coder_comment = _attach_round_metadata(
        _render_public_coder_followup_comment(
            parsed_coder, agent="Claude", prior_items=(carried_item,)
        ),
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="abc123",
            prior_items=(carried_item,),
            raw_structured_coder_response=raw_coder,
            risk_test_matrix_evidence=evidence,
            **extra,
        ),
    )
    if anchor_mutation is not None:
        encoded = coder_comment.split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
        payload = decode_mapping(encoded)
        payload["risk_test_matrix_evidence_full_round"] = anchor_mutation
        coder_comment = coder_comment.replace(encoded, encode_mapping(payload))
    review_comment = _attach_round_metadata(
        structured_pr_review(
            state="blocking",
            prior_item_dispositions=[
                {"item_id": "item-1", "disposition": "blocking", "note": "Still open."}
            ],
        ),
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="abc123",
            prior_items=(carried_item,),
            dispositions=(
                ReviewItemDisposition("item-1", "OpenAI Codex", "blocking", "Still open."),
            ),
            state="blocking",
        ),
    )
    return FakeRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"], summary="Resumed follow-up.")
        ],
        # Resume re-reviews first, so every review carries the item-1 disposition.
        codex_outputs=_blocking_reviews_959(2)[1:],
        pr_payload={
            "headRefOid": "abc123",
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-06-01T00:00:00Z", "body": coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-06-01T00:01:00Z", "body": review_comment},
            ],
        },
    )


def test_959_resume_from_legacy_coder_record_anchors_to_its_round(tmp_path):
    plan_context = _followup_matrix_context()
    runner = _seeded_resume_runner_959(plan_context)
    seeded = _coder_records_959(runner)[0][1]
    assert seeded.risk_test_matrix_evidence_full_round_status == "absent"

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3),
        approved_plan_context=plan_context,
    ) == 0

    records = _coder_records_959(runner)
    assert len(records) >= 2
    body, posted = records[-1]
    assert posted.round_number > seeded.round_number
    assert posted.risk_test_matrix_evidence_full_round == seeded.round_number
    assert posted.risk_test_matrix_evidence_full_round_status == "valid"
    assert f"full matrix in round {seeded.round_number}." in body
    assert f"unchanged since round {seeded.round_number}" in body
    assert posted.risk_test_matrix_evidence is not None
    assert len(posted.risk_test_matrix_evidence["rows"]) == 1


@pytest.mark.parametrize(
    "seed", [{"anchor_mutation": "bogus"}, {"anchor_mutation": 0}, {"full_round": 5}],
    ids=["invalid-string", "invalid-zero", "future"],
)
def test_959_unusable_prior_anchor_posts_full_comment_anchored_to_own_record(tmp_path, seed):
    plan_context = _followup_matrix_context()
    runner = _seeded_resume_runner_959(plan_context, **seed)
    seeded = _coder_records_959(runner)[0][1]
    if "anchor_mutation" in seed:
        assert seeded.risk_test_matrix_evidence_full_round_status == "invalid"
    else:
        assert seeded.risk_test_matrix_evidence_full_round > seeded.round_number

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3),
        approved_plan_context=plan_context,
    ) == 0

    body, posted = _coder_records_959(runner)[-1]
    assert posted.round_number > seeded.round_number
    assert posted.risk_test_matrix_evidence_full_round == posted.round_number
    assert "<summary>Full matrix evidence (1 row)</summary>" in body
    assert "**followup-derived-evidence**" in body
    assert "unchanged since round" not in body


def test_959_followup_without_evidence_posts_no_section_then_full_on_evidence(tmp_path):
    plan_context = _followup_matrix_context()
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"], summary=f"Round {n}.")
            for n in (1, 2)
        ],
        codex_outputs=_blocking_reviews_959(2),
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)
    real_derive = orchestrator._derive_authenticated_risk_evidence_for_coder
    derive_calls = []

    def derive(followup, **kwargs):
        derive_calls.append(followup)
        derived, result = real_derive(followup, **kwargs)
        if len(derive_calls) == 1:
            # First follow-up: no canonical evidence after derivation.
            return dataclasses.replace(derived, risk_test_matrix_evidence=None), result
        return derived, result

    resolve_calls = []
    real_resolve = orchestrator.resolve_matrix_evidence_render

    def resolve(current, previous, current_round):
        assert current is not None
        resolve_calls.append(current_round)
        return real_resolve(current, previous, current_round)

    with patch.object(orchestrator, "_derive_authenticated_risk_evidence_for_coder", derive), \
            patch.object(orchestrator, "resolve_matrix_evidence_render", resolve):
        assert run_pr_loop(
            runner, pr_number=77, config=config, approved_plan_context=plan_context
        ) == 0

    records = _coder_records_959(runner)
    assert len(records) == 2
    (first_body, first), (second_body, second) = records
    assert first.risk_test_matrix_evidence is None
    assert first.risk_test_matrix_evidence_full_round is None
    assert first.risk_test_matrix_evidence_full_round_status == "absent"
    assert _EVIDENCE_HEADING_959 not in first_body
    assert "<details>" not in first_body.split("AGENT_LOOP_META", 1)[0]
    assert resolve_calls == [second.round_number]
    assert second.risk_test_matrix_evidence is not None
    assert second.risk_test_matrix_evidence_full_round == second.round_number
    assert "<summary>Full matrix evidence (1 row)</summary>" in second_body
    assert "unchanged since round" not in second_body


# --- Signed reviewer-board amendment (#943) -------------------------------

from coding_review_agent_loop.board_amendment import (  # noqa: E402
    format_reviewer_board_amendment_comment as _m943_amendment_comment,
)


def _m943_partial_pr_round(tmp_path, **payload):
    """Codex and Gemini review; Antigravity's backend fails before it posts."""
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[],
        **payload,
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
    posted = _posted_scheduler_metadata(runner)
    assert posted and all(
        tuple(item.scheduler_contract["required_reviewers"]) == ("Codex", "Gemini", "Antigravity")
        for item in posted
    )
    return runner


def _m943_amendment_from_error(message):
    start = message.index("Reviewer board amendment:")
    template = message[start:]
    return template.replace(
        "<why the removed reviewer cannot be reached>", "Antigravity quota exhausted."
    )


def _m943_append(runner, body, login="operator"):
    comments = runner.pr_payload.setdefault("comments", [])
    comments.append(
        {
            "author": {"login": login},
            "createdAt": f"2026-05-23T00:00:{len(comments):02d}Z",
            "body": body,
        }
    )


@pytest.mark.parametrize("linked_issue", [False, True], ids=["standalone", "issue-mode"])
def test_pr_board_amendment_resumes_and_qualifies_standalone_pr(tmp_path, monkeypatch, linked_issue):
    """Row pr-qualification, standalone (issue_context=None) and with a linked issue."""
    payload = (
        {
            "issue_payload": {"number": 56, "title": "Linked issue", "body": "Scope."},
            "pr_payload": {"body": "Fixes #56"},
        }
        if linked_issue
        else {}
    )
    runner = _m943_partial_pr_round(tmp_path, **payload)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), auto_merge=True)
    calls_before = _agent_sequence(runner)

    # No record: the drift error prints a filled PR amendment template.
    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume") as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    assert _agent_sequence(runner) == calls_before
    template = _m943_amendment_from_error(str(excinfo.value))
    assert '"flow": "pr"' in template and '"pr_number": 77' in template
    assert '"issue": null' in template

    _m943_append(runner, template)
    monkeypatch.setattr(
        orchestrator, "merge_pr", lambda *args, **kwargs: None
    )
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0

    # The qualification gate re-reads the fresh PR comments, including the
    # pre-amendment three-reviewer records, and accepts the amended board.
    amended_contract = orchestrator.make_contract(
        ("Codex", "Gemini"), "primary-then-panel", None, "Codex"
    )
    orchestrator._fresh_pr_qualification_snapshot(
        runner,
        config=reduced,
        pr_number=77,
        issue_context=None,
        parent_issue_context=None,
        scheduler_contract=amended_contract,
    )
    # The original board is no longer an acceptable configured contract.
    with pytest.raises(AgentLoopError, match="no qualification or merge is permitted"):
        orchestrator._fresh_pr_qualification_snapshot(
            runner,
            config=reduced,
            pr_number=77,
            issue_context=None,
            parent_issue_context=None,
            scheduler_contract=orchestrator.make_contract(
                ("Codex", "Gemini", "Antigravity"), "primary-then-panel", None, "Codex"
            ),
        )
    assert "agy" not in _agent_sequence(runner)[len(calls_before):]
    assert any("Reviewer board amendment applied." in comment for comment in runner.comments)
    posted = _posted_scheduler_metadata(runner)
    # Gemini's round-2 review was reused and Codex's approval carried, yet the
    # activation round still persists a fresh digest-bound scheduler decision.
    amended = [item for item in posted if item.reviewer_board_amendment_digest is not None]
    assert amended
    assert amended[0].phase == "scheduler-prelaunch"
    assert amended[0].round_number == 2
    for item in posted:
        board = tuple(item.scheduler_contract["required_reviewers"])
        if item.reviewer_board_amendment_digest is not None:
            assert board == ("Codex", "Gemini")
        else:
            assert board == ("Codex", "Gemini", "Antigravity")
    completion_notes = [
        comment for comment in runner.comments
        if comment.startswith("Review completed on a reduced reviewer board.")
    ]
    assert len(completion_notes) == 1
    assert "required board now Codex, Gemini" in completion_notes[0]


def test_pr_board_amendment_qualification_refuses_a_stale_contract(tmp_path, monkeypatch):
    """Rows pr-qualification and digest-binding: the gate re-reads fresh PR comments."""
    runner = _m943_partial_pr_round(tmp_path)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), auto_merge=True)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _m943_append(runner, _m943_amendment_from_error(str(excinfo.value)))
    stale_contract = orchestrator.make_contract(
        ("Codex", "Gemini", "Antigravity"), "primary-then-panel", None, "Codex"
    )
    stale = _attach_round_metadata(
        "A stale-board scheduler record posted while managed CI was running.",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=9, subject="abc123",
            phase="reconciliation", scheduler_contract=stale_contract.as_dict(),
            scheduler_previous_sha=None, scheduler_current_sha="abc123",
            scheduler_obligation_digest="0" * 16, scheduler_selected_reviewers=("Codex",),
            scheduler_reasons=("stale",), scheduler_final_sweep=False,
            scheduler_force_full=False, scheduler_calls_avoided=0,
            scheduler_phase="primary", scheduler_primary_reviewer="Codex",
        ),
    )
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *args, **kwargs: None)

    def wait_and_inject(*args, **kwargs):
        _m943_append(runner, stale, login="bot")
        return ManagedCiOutcome(status="passed", head_sha="abc123")

    monkeypatch.setattr(orchestrator, "wait_for_final_qualification", wait_and_inject)
    monkeypatch.setattr(
        orchestrator, "merge_pr", lambda *args, **kwargs: pytest.fail("stale contract must block merge")
    )
    with pytest.raises(AgentLoopError, match="no qualification or merge is permitted"):
        run_pr_loop(runner, pr_number=77, config=reduced)


def test_pr_board_amendment_on_the_owning_issue_fails_closed(tmp_path):
    """Row wrong-surface: a pr amendment belongs on the PR, not the issue."""
    record = _m943_amendment_comment(
        flow="pr", issue=None, pr_number=77,
        original_required_reviewers=("Codex", "Gemini", "Antigravity"),
        policy="primary-then-panel", primary_reviewer="Codex",
        removed_reviewers=("Antigravity",), effective_from_round=1,
        rationale="Antigravity quota exhausted.",
    )
    runner = FakeRunner(
        issue_payload={"number": 56, "title": "Linked issue", "body": "Scope."},
        issue_comments=[
            {"author": {"login": "operator"}, "createdAt": "2026-06-01T00:00:00Z", "body": record}
        ],
        pr_payload={"body": "Fixes #56"},
    )
    with pytest.raises(AgentLoopError, match="Post this record on PR #77"):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, reviewer=("codex", "gemini")))
    assert _agent_sequence(runner) == []


@pytest.mark.parametrize("auto_merge", [True, False], ids=["auto-merge", "manual-qualification"])
def test_pr_board_amendment_managed_completion_names_the_reduced_board(
    tmp_path, monkeypatch, capsys, auto_merge
):
    """Managed-CI completion paths repeat the reduced-board note durably."""
    runner = _m943_partial_pr_round(tmp_path)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), auto_merge=auto_merge)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _m943_append(runner, _m943_amendment_from_error(str(excinfo.value)))
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *args, **kwargs: None)
    waits = []
    monkeypatch.setattr(
        orchestrator,
        "wait_for_final_qualification",
        lambda *args, **kwargs: waits.append(True) or ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    merges = []
    monkeypatch.setattr(orchestrator, "prepare_v2_merge", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orchestrator, "_merge_with_exact_head_proof", lambda *args, **kwargs: merges.append(kwargs)
    )
    monkeypatch.setattr(
        orchestrator, "publish_manual_v2_qualification", lambda *args, **kwargs: "abc123"
    )
    capsys.readouterr()
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0

    assert waits, "the managed qualification path was not exercised"
    assert bool(merges) is auto_merge
    out = capsys.readouterr().out
    assert "Reviewer board amended from round 2" in out
    assert "required board now Codex, Gemini" in out
    notes = [
        comment for comment in runner.comments
        if comment.startswith("Review completed on a reduced reviewer board.")
    ]
    assert len(notes) == 1


def test_pr_board_amendment_qualification_refuses_a_digest_on_a_contract_neutral_record(
    tmp_path, monkeypatch
):
    """Row digest-binding at the gate: a neutral record carrying a digest blocks merge."""
    from coding_review_agent_loop.board_amendment import collect_reviewer_board_amendments

    runner = _m943_partial_pr_round(tmp_path)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), auto_merge=True)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    template = _m943_amendment_from_error(str(excinfo.value))
    _m943_append(runner, template)
    (amendment,) = collect_reviewer_board_amendments(
        [SimpleNamespace(body=template)], flow="pr", pr_number=77
    )
    neutral = _attach_round_metadata(
        _staged_review(reviewer="OpenAI Codex"),
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=9, subject="abc123",
            reviewer_board_amendment_digest=amendment.digest,
        ),
    )
    assert orchestrator._extract_round_metadata_records(
        [SimpleNamespace(body=neutral)], flow="pr"
    )[0].metadata.scheduler_metadata_status == "absent"
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *args, **kwargs: None)
    injected = []

    def wait_and_inject(*args, **kwargs):
        _m943_append(runner, neutral, login="bot")
        injected.append(True)
        return ManagedCiOutcome(status="passed", head_sha="abc123")

    monkeypatch.setattr(orchestrator, "wait_for_final_qualification", wait_and_inject)
    monkeypatch.setattr(
        orchestrator, "merge_pr", lambda *args, **kwargs: pytest.fail("a neutral digest must block merge")
    )
    monkeypatch.setattr(
        orchestrator, "_merge_with_exact_head_proof",
        lambda *args, **kwargs: pytest.fail("a neutral digest must block merge"),
    )
    with pytest.raises(AgentLoopError, match="(?i)no qualification or merge is permitted") as gate:
        run_pr_loop(runner, pr_number=77, config=reduced)
    assert injected
    assert "contract-neutral" in str(gate.value)
