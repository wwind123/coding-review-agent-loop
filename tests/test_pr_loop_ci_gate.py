"""PR-loop tests for checks and ci_health: the PR-loop CI gate (#1542)."""

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from coding_review_agent_loop.ci_health import StalledCheck
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.github import (
    CiWatchOutcome,
    HumanReviewRequirement,
    PullRequestCheck,
    PullRequestChecks,
    PullRequestMetadata,
    PullRequestMergeability,
    PullRequestReviewContext,
    get_pr_checks,
)
from coding_review_agent_loop.migrations import MigrationValidationResult
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
    ManagedCiOutcome,
    OrdinaryRecoveryCapability,
)
from coding_review_agent_loop.unresolved_items import (
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
)
from coding_review_agent_loop.round_state import (
    QualificationCheckpoint,
    PostedRoundMetadata,
    _attach_round_metadata,
)
from coding_review_agent_loop.checks import _pending_ci_stop_guidance, _pending_ci_stop_message
from coding_review_agent_loop.pr_loop_support import (
    _ensure_finalization_ready,
)
from coding_review_agent_loop.protocol import (
    ReviewItemDisposition,
    UnresolvedReviewItem,
    parse_pr_review,
)
from agent_loop_helpers import (
    FakeRunner,
    command_index,
    make_config,
    prior_item_dispositions,
    structured_coder_followup,
    structured_pr_review,
)
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.review_rounds as review_rounds
import coding_review_agent_loop.round_state as round_state
import coding_review_agent_loop.triage as triage
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_attr_in
from pr_loop_test_helpers import (
    _carried_ci_obligations,
    _carried_ci_review_comment,
    _watch_check_board,
)


def _advance_head_after_coder(monkeypatch, runner, head_sha="15cab520f7c90f2ab1738eb929c70e65374e7bb8"):
    """Make the fake coder handoff obey the repair-head contract."""
    original = validated_agent._run_validated_agent

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "coder":
            runner.pr_payload["headRefOid"] = head_sha
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)


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

    assert not pr_loop_support._ordinary_checks_snapshot_is_authoritative(checks)


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

    assert pr_loop_support._ordinary_checks_snapshot_is_authoritative(checks)


def test_ordinary_checks_authority_rejects_forbidden_branch_protection():
    checks = _watch_check_board("passing", protection="forbidden")

    assert not pr_loop_support._ordinary_checks_snapshot_is_authoritative(checks)


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
        assert not pr_loop_support._ordinary_checks_snapshot_is_authoritative(checks)


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

    after_managed = pr_loop._clear_machine_obligations(
        [managed, ordinary], kind="managed-exact-head-ci"
    )
    after_ordinary = pr_loop._clear_machine_obligations(
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

    assert not pr_loop_support._qualification_checkpoint_review_identity_matches(
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
        approval_digest=pr_loop_support._qualification_digest(("Codex",)),
        scheduler_digest=pr_loop_support._qualification_digest(
            pr_loop_support._prior_item_ledger_signature([item])
        ),
    )
    assert pr_loop_support._qualification_checkpoint_review_identity_matches(
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
        assert not pr_loop_support._qualification_checkpoint_review_identity_matches(
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
        failed_head_sha="0109780857a5f763d3dacdb5f43c546d6776b3ba",
        candidate_head_sha="repair-head",
    )

    advanced = pr_loop._advance_machine_obligations_for_head(
        [item], current_head_sha="0109780857a5f763d3dacdb5f43c546d6776b3ba"
    )
    checkpoint = pr_loop_support._machine_obligation_checkpoint(
        advanced,
        current_head_sha="0109780857a5f763d3dacdb5f43c546d6776b3ba",
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
        failed_head_sha="0109780857a5f763d3dacdb5f43c546d6776b3ba",
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

    normalized = review_rounds._normalize_approval_gated_managed_ci_review(
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
        failed_head_sha="0109780857a5f763d3dacdb5f43c546d6776b3ba",
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

    normalized = review_rounds._normalize_approval_gated_managed_ci_review(
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

    normalized = review_rounds._normalize_approval_gated_managed_ci_review(
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
        failed_head_sha="0109780857a5f763d3dacdb5f43c546d6776b3ba",
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

    normalized = review_rounds._normalize_approval_gated_managed_ci_review(
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
        failed_head_sha="0109780857a5f763d3dacdb5f43c546d6776b3ba",
        candidate_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
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
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
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
            "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
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
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    dispatches = []
    monkeypatch.setattr(
        pr_loop,
        "dispatch_final_qualification",
        lambda *args, **kwargs: dispatches.append(kwargs),
    )
    monkeypatch.setattr(
        pr_loop,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"),
    )
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: None)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert dispatches and dispatches[0]["expected_head_sha"] == "6367c48dd193d56ea7b0baad25b19455e529f5ee"
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
        "coding_review_agent_loop.pr_loop.validate_pr_migration_topology",
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


    original_get_pr_checks = pr_loop.get_pr_checks
    monkeypatch.setattr(pr_loop, "get_pr_checks", advance_checks)
    _advance_head_after_coder(monkeypatch, runner)
    monkeypatch.setattr(
        pr_loop,
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
        board = architecture_contract.dataclasses_replace(board, infrastructure_stalls=(
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

    monkeypatch.setattr(pr_loop, "get_pr_checks", snapshot)
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: pytest.fail("must not wait"))
    original = validated_agent._run_validated_agent
    captured = {}

    class CoderReached(Exception):
        pass

    def capture(*args, **kwargs):
        if kwargs.get("role") == "coder":
            captured.update(kwargs)
            raise CoderReached
        return original(*args, **kwargs)

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", capture)
    with pytest.raises(CoderReached):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, auto_merge=auto_merge))
    assert snapshots == ["6367c48dd193d56ea7b0baad25b19455e529f5ee", "6367c48dd193d56ea7b0baad25b19455e529f5ee"]
    prompt = captured["prompt"]
    assert "Fix the application bug." in prompt
    if board_state in {"failing", "mixed"}:
        assert "Failing checks: test (failure)" in prompt
        assert failure.url in prompt
        assert "Reviewed head: 6367c48dd193d56ea7b0baad25b19455e529f5ee" in prompt
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
    monkeypatch.setattr(pr_loop, "get_pr_checks", lambda *a, **k: next(snapshots))
    _advance_head_after_coder(monkeypatch, runner, "15cab520f7c90f2ab1738eb929c70e65374e7bb8")
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: CiWatchOutcome(
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
        if kwargs["metadata"].head_sha == "6367c48dd193d56ea7b0baad25b19455e529f5ee":
            return _watch_check_board("failing", failing=(failed,))
        return _watch_check_board(
            "passing",
            passing=(
                PullRequestCheck(name="test", kind="check_run", status="success"),
                PullRequestCheck(name="docs", kind="check_run", status="skipped"),
            ),
        )

    monkeypatch.setattr(pr_loop, "get_pr_checks", checks)
    _advance_head_after_coder(monkeypatch, runner, "15cab520f7c90f2ab1738eb929c70e65374e7bb8")
    monkeypatch.setattr(
        pr_loop,
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


def test_review_only_mode_reports_passing_but_uncleared_ci_at_the_gate(
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
        if kwargs["metadata"].head_sha == "6367c48dd193d56ea7b0baad25b19455e529f5ee":
            return _watch_check_board("failing", failing=(failed,))
        return _watch_check_board(
            "passing",
            passing=(PullRequestCheck(name="docs", kind="check_run", status="skipped"),),
        )

    monkeypatch.setattr(pr_loop, "get_pr_checks", checks)
    _advance_head_after_coder(monkeypatch, runner, "15cab520f7c90f2ab1738eb929c70e65374e7bb8")
    monkeypatch.setattr(
        pr_loop,
        "watch_pr_checks",
        lambda *args, **kwargs: pytest.fail("review-only mode must not start the watcher"),
    )

    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(
            runner,
            pr_number=77,
            config=make_config(
                tmp_path, auto_merge=False, watch_pending_ci=False, max_rounds=2
            ),
        )

    message = str(excinfo.value)
    assert message.startswith("PR #77 cannot finalize: PR #77 ")
    assert "awaiting authoritative qualification after round" in message
    assert "github-pr-checks (item-1)" in message
    legacy_prefix, _, _detail = message.partition("\nBlocking obligations:\n")
    assert legacy_prefix == (
        "PR #77 cannot finalize: PR #77 has a unanimously reviewed correction "
        "awaiting authoritative qualification after round 2: github-pr-checks "
        "(item-1). No approval or merge was attempted."
    )
    assert "Blocking obligations:" in message
    assert "observed in this run: GitHub checks read passing at 15cab520f7c90f2ab1738eb929c70e65374e7bb8" in message
    assert "only skipped or neutral" in message
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)
    assert not any("observed in this run" in comment for comment in runner.comments)
    metadata_bodies = [
        body
        for body in [
            *runner.comments,
            *(
                item["body"]
                for item in runner.pr_payload.get("comments", [])
                if isinstance(item, dict) and isinstance(item.get("body"), str)
            ),
        ]
        if "AGENT_LOOP_META: " in body
    ]
    assert metadata_bodies
    for body in metadata_bodies:
        decoded = round_state._decode_round_metadata(
            body.split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
        )
        rendered = repr(decoded)
        assert "observed in this run" not in rendered
        assert "read passing at" not in rendered
        assert "not authoritative" not in rendered


@pytest.mark.parametrize("auto_merge", [False, True])
def test_watch_mode_success_uses_full_board_without_second_wait(
    tmp_path, monkeypatch, auto_merge
):
    runner = FakeRunner(codex_outputs=[structured_pr_review(state="approved", summary="Approved.")])
    config = make_config(tmp_path, watch_pending_ci=True, auto_merge=auto_merge)
    monkeypatch.setattr(
        pr_loop,
        "watch_pr_checks",
        lambda *args, **kwargs: CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", attempts_used=1
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
        pr_loop,
        "watch_pr_checks",
        lambda *args, **kwargs: CiWatchOutcome(
            status="passed", pr_checks=skipped_only, head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", attempts_used=1
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
            status="passed", pr_checks=None, head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", attempts_used=1
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
            status="passed", pr_checks=skipped_only, head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", attempts_used=1
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
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *args, **kwargs: outcome)

    with pytest.raises(AgentLoopError, match=expected):
        run_pr_loop(
            runner,
            pr_number=77,
            config=make_config(tmp_path, watch_pending_ci=True, auto_merge=True),
        )

    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


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
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pr_loop,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"),
    )
    monkeypatch.setattr(
        pr_loop_support, "merge_pr", lambda *args, **kwargs: pytest.fail("ordinary obligation was bypassed")
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
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pr_loop_support, "merge_pr", lambda *args, **kwargs: pytest.fail("managed obligation was bypassed")
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
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *args, **kwargs: None)

    def watch(*args, **kwargs):
        watcher_calls.append(kwargs["metadata"].head_sha)
        return CiWatchOutcome(
            status="passed",
            pr_checks=_watch_check_board("passing"),
            head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            attempts_used=1,
        )

    monkeypatch.setattr(pr_loop, "watch_pr_checks", watch)
    monkeypatch.setattr(
        pr_loop_support,
        "merge_pr",
        lambda *args, **kwargs: pytest.fail("managed obligation was bypassed"),
    )

    with pytest.raises(AgentLoopError, match="managed-exact-head-ci"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert watcher_calls == ["6367c48dd193d56ea7b0baad25b19455e529f5ee"]


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
        failed_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        candidate_head_sha=None,
        base_branch="main",
        allowed_rounds=1,
    )
    scheduler_contract = pr_loop.make_contract(
        ("Codex",), "selective-intermediate", None
    )
    summary = _attach_round_metadata(
        "Persisted legacy machine obligation.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=1,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(legacy,),
            state="blocking",
            phase="qualification-checkpoint",
            qualification_checkpoint=checkpoint,
            scheduler_contract=scheduler_contract.as_dict(),
            scheduler_previous_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            scheduler_current_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            scheduler_obligation_digest=pr_loop_support._qualification_digest(
                pr_loop_support._prior_item_ledger_signature((legacy,))
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
    original_select_reviewers = pr_loop.select_reviewers

    def capture_force_full(snapshot, *args, **kwargs):
        force_full_values.append(snapshot.force_full)
        return original_select_reviewers(snapshot, *args, **kwargs)

    monkeypatch.setattr(pr_loop, "select_reviewers", capture_force_full)

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
    scheduler_contract = pr_loop.make_contract(
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
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(reviewer_item,),
            state="blocking",
            scheduler_contract=scheduler_contract.as_dict(),
            scheduler_previous_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            scheduler_current_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            scheduler_obligation_digest=pr_loop_support._qualification_digest(()),
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
    original_select_reviewers = pr_loop.select_reviewers

    def capture_force_full(snapshot, *args, **kwargs):
        force_full_values.append(snapshot.force_full)
        return original_select_reviewers(snapshot, *args, **kwargs)

    monkeypatch.setattr(pr_loop, "select_reviewers", capture_force_full)

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
        candidate_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
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
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
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
        expected_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
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
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *args, **kwargs: activation)
    monkeypatch.setattr(
        pr_loop_support,
        "_finalize_ordinary_recovery_merge",
        lambda *args, **kwargs: pytest.fail("ordinary recovery bypassed another machine obligation"),
    )
    guard_calls = []
    original_guard = pr_loop_support._ensure_finalization_ready

    def capture_guard(*args, **kwargs):
        guard_calls.append(
            {
                "ignored_machine_kinds": kwargs.get("ignored_machine_kinds"),
                "items": tuple(kwargs.get("items", ())),
            }
        )
        return original_guard(*args, **kwargs)

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop,
    ), "_ensure_finalization_ready", capture_guard)

    with pytest.raises(AgentLoopError, match="managed-exact-head-ci"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)
    assert len(guard_calls) == 1
    assert guard_calls[0]["ignored_machine_kinds"] == frozenset({"github-pr-checks"})
    assert [item.obligation_kind for item in guard_calls[0]["items"]] == [
        "managed-exact-head-ci"
    ]


def _spy_coder_followup_dispatch(monkeypatch):
    """Record the item set each real coder dispatch labels, then log it (#1024)."""
    dispatched = []
    real = pr_loop_support._log_coder_followup_dispatch

    def spy(config, round_number, coder_name, coder_followup_items):
        dispatched.append((round_number, tuple(coder_followup_items)))
        return real(config, round_number, coder_name, coder_followup_items)

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_log_coder_followup_dispatch", spy)
    return dispatched


def _round_label_lines(err):
    return [
        line.split("] ", 1)[-1] for line in err.splitlines()
        if "addressing reviewer feedback" in line or "repairing failed CI" in line
    ]


def test_run_pr_loop_labels_reviewer_and_acknowledgement_rounds_as_reviewer_feedback(
    tmp_path, monkeypatch, capsys,
):
    """Round 1 is reviewer-driven; round 2 is the acknowledgement re-injection."""
    requirement = HumanReviewRequirement(
        source_type="PR comment", author="maintainer", created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="Keep the audit trail.",
    )
    ack_markdown = (
        "Implemented the fix.\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
        "### Human requirements\n- Requirement 1: kept the audit trail.\n"
        "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Fix.", blocking_items=["Fix the bug."],
                human_requirements_resolved=True,
            ),
            # Approves without the acknowledgement, so the run re-injects it.
            structured_pr_review(
                state="approved", summary="Fixed.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
            structured_pr_review(
                state="approved", summary="Acknowledged.", human_requirements_resolved=True,
            ),
        ],
        claude_outputs=[ack_markdown, ack_markdown],
    )
    runner.pr_payload["headRefName"] = "feature/x"
    metadata = PullRequestMetadata(
        number=77, repo="OWNER/REPO", title="Acknowledgement", head_branch="feature/x",
        base_branch="main", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", url="https://github.com/OWNER/REPO/pull/77",
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_observed_pr_review_context",
        lambda *_a, **_k: PullRequestReviewContext(
            metadata=dataclasses.replace(metadata, head_sha=runner.pr_payload["headRefOid"]),
            comments=(), human_requirements=(requirement,),
        ),
    )
    dispatched = _spy_coder_followup_dispatch(monkeypatch)
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3, quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert [round_number for round_number, _items in dispatched] == [1, 2]
    assert [item.item_id for item in dispatched[0][1]] == ["item-1"]
    assert not dispatched[0][1][0].is_machine_obligation
    # Dedicated acknowledgement obligations are excluded from coder findings;
    # the coder acknowledges the surfaced requirement through its response.
    assert dispatched[1][1] == ()
    err = capsys.readouterr().err
    assert "re-injecting as blocking item" in err
    assert _round_label_lines(err) == [
        "Round 1: Claude addressing reviewer feedback",
        "Round 2: Claude addressing reviewer feedback",
    ]


class _AcknowledgementDispatchReached(Exception):
    """Stop right after the acknowledgement-only coder dispatch is labelled."""


def test_run_pr_loop_labels_an_acknowledgement_only_round_as_reviewer_feedback(
    tmp_path, monkeypatch, capsys,
):
    """The dedicated acknowledgement obligation alone selects an empty coder set.

    A signed requirement that appears after the coder's last acknowledgement
    mints the acknowledgement obligation; once the reviewer resolves every
    other item and keeps only that obligation blocking, the next coder
    dispatch carries it alone and must not be labelled a CI repair (#1024).
    """
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
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Fix.", blocking_items=["Fix the bug."],
                human_requirements_resolved=True,
            ),
            # Approves without the acknowledgement: the run re-injects it as item-2.
            structured_pr_review(
                state="approved", summary="Fixed.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
            # Resolves item-2 and keeps only the acknowledgement obligation open.
            structured_pr_review(
                state="blocking", summary="The new requirement is unacknowledged.",
                prior_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved"},
                    {"item_id": ack_item_id, "disposition": "blocking", "note": note},
                ],
            ),
        ],
        claude_outputs=[ack_markdown, ack_markdown],
    )
    runner.pr_payload["headRefName"] = "feature/x"
    metadata = PullRequestMetadata(
        number=77, repo="OWNER/REPO", title="Acknowledgement", head_branch="feature/x",
        base_branch="main", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", url="https://github.com/OWNER/REPO/pull/77",
    )
    fetches = []

    def review_context(*_args, **_kwargs):
        fetches.append(True)
        # The second signed requirement arrives after the first coder turn.
        requirements = (first,) if len(fetches) <= 2 else (first, second)
        return PullRequestReviewContext(
            metadata=dataclasses.replace(metadata, head_sha=runner.pr_payload["headRefOid"]),
            comments=(), human_requirements=requirements,
        )

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_observed_pr_review_context", review_context)
    selections = []
    real_select = pr_loop.select_coder_followup_items

    def select(items):
        selected = real_select(items)
        selections.append(([item.item_id for item in items], selected))
        return selected

    monkeypatch.setattr(pr_loop, "select_coder_followup_items", select)
    dispatched = []
    real_log = pr_loop_support._log_coder_followup_dispatch

    def log_dispatch(config, round_number, coder_name, coder_followup_items):
        dispatched.append((round_number, tuple(coder_followup_items)))
        real_log(config, round_number, coder_name, coder_followup_items)
        if round_number == 3:
            raise _AcknowledgementDispatchReached

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop,
    ), "_log_coder_followup_dispatch", log_dispatch)
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=4, quiet=False)

    with pytest.raises(_AcknowledgementDispatchReached):
        run_pr_loop(runner, pr_number=77, config=config)

    # Round 3's real dispatch carried only the acknowledgement obligation,
    # which selects an empty classifiable follow-up set.
    assert selections[-1] == ([ack_item_id], ())
    assert dispatched[-1] == (3, ())
    labels = _round_label_lines(capsys.readouterr().err)
    assert labels[-1] == "Round 3: Claude addressing reviewer feedback"
    assert not any("repairing failed CI" in label for label in labels)


def test_run_pr_loop_labels_a_mixed_reviewer_and_ci_round_as_reviewer_feedback(
    tmp_path, monkeypatch, capsys,
):
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
    monkeypatch.setattr(pr_loop, "get_pr_checks", lambda *a, **k: next(snapshots))
    _advance_head_after_coder(monkeypatch, runner, "15cab520f7c90f2ab1738eb929c70e65374e7bb8")
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: CiWatchOutcome(
        status="passed", pr_checks=_watch_check_board("passing"),
        head_sha=runner.pr_payload["headRefOid"], attempts_used=1
    ))
    dispatched = _spy_coder_followup_dispatch(monkeypatch)

    assert run_pr_loop(
        runner, pr_number=77,
        config=make_config(tmp_path, watch_pending_ci=True, quiet=False),
    ) == 0

    ((round_number, items),) = dispatched
    assert round_number == 1
    kinds = sorted(item.obligation_kind or "reviewer" for item in items)
    assert kinds == ["github-pr-checks", "reviewer"]
    assert _round_label_lines(capsys.readouterr().err) == [
        "Round 1: Claude addressing reviewer feedback"
    ]


def test_ci_repair_log_label_requires_a_nonempty_ci_only_followup_set(tmp_path, capsys):
    config = make_config(tmp_path, quiet=False)
    ci_item = UnresolvedReviewItem(
        item_id="item-1", reviewer="GitHub managed exact-head CI", source_round=1,
        text="Managed CI failed.", status="blocking", authority="machine",
        obligation_kind="managed-exact-head-ci", lifecycle="repair_required",
        failed_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
    )
    reviewer_item = UnresolvedReviewItem(
        item_id="item-2", reviewer="Codex", source_round=1, text="Fix it.", status="blocking",
    )
    ack_item = UnresolvedReviewItem(
        item_id=pr_loop.HUMAN_REQUIREMENTS_ACK_ITEM_ID, reviewer="Orchestrator",
        source_round=1, text="Acknowledge.", status="blocking", authority="machine",
        obligation_kind="human-requirements-acknowledgement", lifecycle="repair_required",
    )
    cases = {
        "ci-only": (ci_item,),
        "acknowledgement-only": (ack_item,),
        "mixed": (ci_item, reviewer_item),
        "reviewer-only": (reviewer_item,),
    }
    labels = {}
    for name, items in cases.items():
        pr_loop_support._log_coder_followup_dispatch(
            config, 4, "Claude", pr_loop.select_coder_followup_items(items)
        )
        labels[name] = capsys.readouterr().err

    assert "Round 4: Claude repairing failed CI" in labels["ci-only"]
    for name in ("acknowledgement-only", "mixed", "reviewer-only"):
        assert "Round 4: Claude addressing reviewer feedback" in labels[name]
        assert "repairing failed CI" not in labels[name]


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
                head_sha="15cab520f7c90f2ab1738eb929c70e65374e7bb8", attempts_used=1
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
    _advance_head_after_coder(monkeypatch, runner, "15cab520f7c90f2ab1738eb929c70e65374e7bb8")
    monkeypatch.setattr(
        pr_loop, "watch_pr_checks", lambda *args, **kwargs: next(outcomes)
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
            runner.pr_payload["headRefOid"] = "4ebf8a1679f50e9e49fc965072f996bfe0dc18c6"
            return CiWatchOutcome(
                status="head_changed", head_sha="4ebf8a1679f50e9e49fc965072f996bfe0dc18c6", attempts_used=2
            )
        return CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha="4ebf8a1679f50e9e49fc965072f996bfe0dc18c6", attempts_used=1
        )

    monkeypatch.setattr(pr_loop, "watch_pr_checks", watch)

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
    _advance_head_after_coder(monkeypatch, runner, "15cab520f7c90f2ab1738eb929c70e65374e7bb8")
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
            runner.pr_payload["headRefOid"] = "8db9b1737b10a778f93e4549c993743580a87b05"
            return CiWatchOutcome(
                status="head_changed", head_sha="8db9b1737b10a778f93e4549c993743580a87b05", attempts_used=1
            )
        return CiWatchOutcome(
            status="passed", pr_checks=_watch_check_board("passing"),
            head_sha="8db9b1737b10a778f93e4549c993743580a87b05", attempts_used=1
        )

    monkeypatch.setattr(pr_loop, "watch_pr_checks", watch)

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
    _advance_head_after_coder(monkeypatch, runner, "15cab520f7c90f2ab1738eb929c70e65374e7bb8")
    call_count = 0

    def watch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            runner.pr_payload["headRefOid"] = "8db9b1737b10a778f93e4549c993743580a87b05"
            return CiWatchOutcome(
                status="head_changed", head_sha="8db9b1737b10a778f93e4549c993743580a87b05", attempts_used=1
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
            head_sha="15cab520f7c90f2ab1738eb929c70e65374e7bb8", attempts_used=1
        )

    monkeypatch.setattr(pr_loop, "watch_pr_checks", watch)

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
        pr_loop,
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
        runner.pr_payload["headRefOid"] = "4ebf8a1679f50e9e49fc965072f996bfe0dc18c6"
        return CiWatchOutcome(status="head_changed", head_sha="4ebf8a1679f50e9e49fc965072f996bfe0dc18c6", attempts_used=2)

    monkeypatch.setattr(pr_loop, "watch_pr_checks", watch)

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
            head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", attempts_used=1
        )

    monkeypatch.setattr(pr_loop, "watch_pr_checks", watch)

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
    with patch.object(pr_loop, "watch_pr_checks", wraps=pr_loop.watch_pr_checks) as watch_spy:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert watch_spy.call_count == 1
    merge_commands = [
        cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "pr", "merge"]
    ]
    assert merge_commands == [[
        "gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge",
        "--match-head-commit", "6367c48dd193d56ea7b0baad25b19455e529f5ee",
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
    _advance_head_after_coder(monkeypatch, runner, "15cab520f7c90f2ab1738eb929c70e65374e7bb8")
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
            head_sha="15cab520f7c90f2ab1738eb929c70e65374e7bb8", attempts_used=1
        )

    monkeypatch.setattr(pr_loop, "watch_pr_checks", watch)

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
    _advance_head_after_coder(monkeypatch, runner, "15cab520f7c90f2ab1738eb929c70e65374e7bb8")
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


    original_get_pr_checks = pr_loop.get_pr_checks
    monkeypatch.setattr(pr_loop, "get_pr_checks", next_checks)
    monkeypatch.setattr(
        pr_loop,
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


    original_get_pr_checks = pr_loop.get_pr_checks
    monkeypatch.setattr(pr_loop, "get_pr_checks", next_checks)

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
    original = validated_agent._run_validated_agent
    captured = {}

    class CoderReached(Exception):
        pass

    def capture(*args, **kwargs):
        if kwargs.get("role") == "coder":
            captured.update(kwargs)
            raise CoderReached
        return original(*args, **kwargs)

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", capture)
    monkeypatch.setattr(pr_loop, "watch_pr_checks", lambda *a, **k: pytest.fail("must not wait"))
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
    assert not review_rounds._is_pending_ci_only_review(
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
    # #1477: the watch stays opt-in, so the stop names the flag and its bound.
    hint = (
        "Rerun with --watch-pending-ci to watch CI (bounded by --ci-timeout-seconds) and "
        "re-run plumbing-only failures automatically."
    )
    assert hint in stop_comment
    assert hint in captured.out


@pytest.mark.parametrize("auto_merge", [False, True])
def test_pending_ci_watch_hint_is_not_emitted_when_the_watch_was_active(
    tmp_path, monkeypatch, capsys, auto_merge
):
    runner = FakeRunner(
        codex_outputs=["Looks good locally.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_check_runs_payload={"check_runs": []},
        pr_status_payload={"state": "pending", "statuses": []},
        pr_branch_protection_payload={"contexts": ["test"]},
    )
    monkeypatch.setattr(
        pr_loop,
        "watch_pr_checks",
        lambda *args, **kwargs: CiWatchOutcome(
            status="timeout", pr_checks=_watch_check_board("pending"), head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            attempts_used=1,
        ),
    )
    config = make_config(tmp_path, watch_pending_ci=True, auto_merge=auto_merge)

    if auto_merge:
        with pytest.raises(AgentLoopError):
            run_pr_loop(runner, pr_number=77, config=config)
    else:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    captured = capsys.readouterr()
    assert "Rerun with --watch-pending-ci" not in captured.out
    assert not any("Rerun with --watch-pending-ci" in comment for comment in runner.comments)


def test_pending_ci_stop_guidance_hint_is_opt_in():
    assert "--watch-pending-ci" not in _pending_ci_stop_guidance("pending")
    assert "--watch-pending-ci" in _pending_ci_stop_guidance("pending", watch_hint=True)
    assert "--ci-timeout-seconds" in _pending_ci_stop_message(77, "unavailable", [], watch_hint=True)

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


def _forbidden_protection_runner(merge_state):
    return FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="LGTM.")],
        pr_payload={"mergeable": "MERGEABLE", "mergeStateStatus": merge_state},
        pr_check_runs_payload={
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]
        },
        pr_status_payload={"state": "success", "statuses": []},
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="gh: Resource not accessible by integration (HTTP 403)",
    )


def test_pr_loop_auto_merge_unreadable_protection_with_clean_merge_state_merges(tmp_path):
    # #1055: the classic protection endpoint returns 403, but GitHub reports
    # CLEAN for the green current head, so the head-pinned merge proceeds.
    runner = _forbidden_protection_runner("CLEAN")
    config = make_config(tmp_path, auto_merge=True, ci_timeout_seconds=1200, ci_poll_interval_seconds=30)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    merges = [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "pr", "merge"]]
    assert len(merges) == 1
    assert merges[0][-2:] == ["--match-head-commit", "6367c48dd193d56ea7b0baad25b19455e529f5ee"]


@pytest.mark.parametrize("merge_state", ["BLOCKED", "UNSTABLE", "UNKNOWN"])
def test_pr_loop_auto_merge_unreadable_protection_without_clean_stops_fast(tmp_path, merge_state):
    runner = _forbidden_protection_runner(merge_state)
    config = make_config(
        tmp_path, auto_merge=True, ci_timeout_seconds=1200, ci_poll_interval_seconds=30,
        ci_startup_timeout_seconds=60,
    )

    with pytest.raises(AgentLoopError, match="merge state is not CLEAN; no merge attempted"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)
    stop_comments = [c for c in runner.comments if "merge readiness cannot be confirmed" in c]
    assert len(stop_comments) == 1
    assert "every observed GitHub check passed" in stop_comments[0]
    assert "HTTP 403" in stop_comments[0]
    assert f"GitHub merge state for the current head: {merge_state}" in stop_comments[0]
    assert not any("GitHub checks are still pending" in c for c in runner.comments)
    # Bounded by the startup window, not the 1200s watch timeout.
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert len(sleep_commands) < 1200 // 30 - 1


def test_ordinary_checks_authority_accepts_forbidden_protection_with_clean_head():
    checks = _watch_check_board("passing", protection="forbidden")
    clean = PullRequestMergeability("mergeable", "MERGEABLE", "CLEAN", "6367c48dd193d56ea7b0baad25b19455e529f5ee", "main")

    assert pr_loop_support._ordinary_checks_snapshot_is_authoritative(
        checks, clean, head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"
    )
    assert not pr_loop_support._ordinary_checks_snapshot_is_authoritative(
        checks, clean, head_sha="other"
    )
    for merge_state in ("BLOCKED", "UNSTABLE", "BEHIND", "DRAFT", None):
        assert not pr_loop_support._ordinary_checks_snapshot_is_authoritative(
            checks,
            PullRequestMergeability("mergeable", "MERGEABLE", merge_state, "6367c48dd193d56ea7b0baad25b19455e529f5ee", "main"),
            head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        )
    # Deferral (the recovery finalizer re-checks CLEAN after readiness) only
    # relaxes an unreadable protection; the board conditions still apply.
    draft = PullRequestMergeability("mergeable", "MERGEABLE", "DRAFT", "6367c48dd193d56ea7b0baad25b19455e529f5ee", "main")
    assert pr_loop_support._ordinary_checks_snapshot_is_authoritative(
        checks, draft, head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", defer_unreadable_protection=True,
    )
    assert not pr_loop_support._ordinary_checks_snapshot_is_authoritative(
        _watch_check_board("passing", protection="unavailable"),
        draft,
        head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        defer_unreadable_protection=True,
    )
    assert not pr_loop_support._ordinary_checks_snapshot_is_authoritative(
        _watch_check_board("pending", pending=(PullRequestCheck("x", "check_run", "queued"),),
                           protection="forbidden"),
        draft,
        head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        defer_unreadable_protection=True,
    )


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
    assert "<!-- AGENT_APPROVED_FOLLOWUPS: pr=77 head=6367c48dd193d56ea7b0baad25b19455e529f5ee mode=summarize -->" in runner.comments[1]
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
        "<!-- AGENT_APPROVED_FOLLOWUPS: pr=77 head=6367c48dd193d56ea7b0baad25b19455e529f5ee mode=summarize -->\n"
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
    assert "<!-- AGENT_APPROVED_FOLLOWUPS: pr=77 head=6367c48dd193d56ea7b0baad25b19455e529f5ee mode=issue -->" in runner.comments[1]
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
                        "<!-- AGENT_APPROVED_FOLLOWUPS: pr=77 head=6367c48dd193d56ea7b0baad25b19455e529f5ee mode=issue -->\n"
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
            head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            url="https://github.com/OWNER/REPO/pull/77",
        ),
    )

    assert pr_checks.state == "no_checks"
    assert pr_checks.branch_protection_status == "unavailable"
    assert pr_checks.branch_protection_note == "Dry run mode does not query live GitHub PR checks."


def test_gate_details_separate_excluded_dispatch_from_failing_pr_checks(tmp_path):
    from coding_review_agent_loop.checks import _pr_check_details
    from coding_review_agent_loop.github import get_pr_checks
    from test_ci_health import (
        SUITE_DISPATCH, _StubGhRunner, _cr, _dispatch, _history, _listing, _metadata,
    )

    neg = _cr(2, "validate", "failure", suite=SUITE_DISPATCH, run=7000)
    ok = _cr(1, "ci", "success", run=5000)
    board = get_pr_checks(
        _StubGhRunner(
            check_runs_payload=_listing(neg, ok),
            dispatch_payload=_dispatch(),
            history_stdout=_history(neg, ok),
        ),
        config=make_config(tmp_path),
        metadata=_metadata(),
    )
    assert board.state == "passing"
    details = _pr_check_details(board)
    assert any(d.startswith("Excluded ad hoc workflow_dispatch checks") for d in details)
    assert not any(d.startswith("Failing checks") for d in details)

    failing = get_pr_checks(
        _StubGhRunner(check_runs_payload=_listing(_cr(1, "ci", "failure", run=5000))),
        config=make_config(tmp_path),
        metadata=_metadata(),
    )
    assert failing.state == "failing"
    assert any(d.startswith("Failing checks") for d in _pr_check_details(failing))


def _scoped_boards_for_loop(tmp_path):
    from coding_review_agent_loop.github import get_pr_checks
    from test_ci_health import (
        SUITE_DISPATCH, _StubGhRunner, _cr, _dispatch, _history, _listing, _metadata,
    )

    config = make_config(tmp_path)
    neg = _cr(2, "validate", "failure", suite=SUITE_DISPATCH, run=7000)
    ok = _cr(1, "ci", "success", run=5000)
    bad = _cr(1, "ci", "failure", run=5000)
    excluded_only = get_pr_checks(
        _StubGhRunner(
            check_runs_payload=_listing(neg, ok),
            dispatch_payload=_dispatch(),
            history_stdout=_history(neg, ok),
        ),
        config=config,
        metadata=_metadata(),
    )
    failing_pr = get_pr_checks(
        _StubGhRunner(check_runs_payload=_listing(bad)), config=config, metadata=_metadata()
    )
    return excluded_only, failing_pr


def test_loop_excluded_failing_dispatch_creates_no_checks_blocking_item(tmp_path, monkeypatch):
    excluded_only, _failing = _scoped_boards_for_loop(tmp_path)
    assert excluded_only.state == "passing" and excluded_only.excluded
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        claude_outputs=[],
    )
    monkeypatch.setattr(pr_loop, "get_pr_checks", lambda *a, **k: excluded_only)
    monkeypatch.setattr(
        pr_loop,
        "watch_pr_checks",
        lambda *a, **k: CiWatchOutcome(
            status="passed", pr_checks=excluded_only, head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", attempts_used=1
        ),
    )

    assert run_pr_loop(
        runner, pr_number=77, config=make_config(tmp_path, watch_pending_ci=True)
    ) == 0

    assert not any(c.startswith("GitHub PR checks are") for c in runner.comments)
    assert not [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def test_loop_failing_pull_request_check_creates_checks_blocking_item(tmp_path, monkeypatch):
    excluded_only, failing_pr = _scoped_boards_for_loop(tmp_path)
    assert failing_pr.state == "failing" and not failing_pr.excluded
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Fixed."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Fixed CI.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    _advance_head_after_coder(monkeypatch, runner)
    monkeypatch.setattr(
        pr_loop,
        "get_pr_checks",
        lambda *a, **k: failing_pr
        if runner.pr_payload["headRefOid"] == "6367c48dd193d56ea7b0baad25b19455e529f5ee"
        else excluded_only,
    )
    monkeypatch.setattr(
        pr_loop,
        "watch_pr_checks",
        lambda *a, **k: CiWatchOutcome(
            status="passed", pr_checks=excluded_only,
            head_sha=runner.pr_payload["headRefOid"], attempts_used=1,
        ),
    )

    assert run_pr_loop(
        runner, pr_number=77, config=make_config(tmp_path, max_rounds=2, watch_pending_ci=True)
    ) == 0

    assert any(c.startswith("GitHub PR checks are failing for PR #77.") for c in runner.comments)
    assert [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
