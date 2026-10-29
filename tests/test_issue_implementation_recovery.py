"""Issue-implementation tests: completion recovery and managed issue resume (#1543)."""

import json
import re
from dataclasses import replace

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.decomposition import (
    approved_plan_hash,
    format_one_shot_impl_handoff_comment,
)
from coding_review_agent_loop.github import (
    HumanReviewRequirement,
    IssueContext,
    get_issue_context,
)
from coding_review_agent_loop.managed_ci import (
    ManagedCiCreationIntent,
    UNPROTECTED_OVERRIDE_TRAILER,
    parse_issue_created_authorization_comment,
)
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.prompts import build_completion_recovery_prompt
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.triage as triage
from agent_loop_helpers import patch_attr_in
from issue_loop_test_helpers import (
    FakeRunner,
    _IssueRecoveryWorkflowRunner,
    _managed_issue_handoff,
    _provenance_pages,
)
from coding_review_agent_loop.runner import CommandResult
from agent_loop_helpers import (
    make_config,
    prior_item_dispositions,
    structured_pr_review,
    structured_issue_implementation,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


def test_approved_plan_completion_recovery_carries_parent_and_child_requirements(
    tmp_path, monkeypatch
):
    parent_requirement = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/55",
        body="Preserve the parent audit trail.",
    )
    child_requirement = HumanReviewRequirement(
        source_type="Issue comment",
        author="maintainer",
        created_at="2026-05-17T08:10:00Z",
        url="https://github.com/OWNER/REPO/issues/56#issuecomment-2",
        body="Preserve the child API.",
    )
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child issue",
        body="Child issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(child_requirement,),
    )
    parent_context = IssueContext(
        number=55,
        repo="OWNER/REPO",
        title="Parent issue",
        body="Parent issue body",
        url="https://github.com/OWNER/REPO/issues/55",
        comments=(),
        human_requirements=(parent_requirement,),
    )
    captured = {}

    class _StopAfterCapture(Exception):
        pass

    def capture_policy(*args, **kwargs):
        captured["policy"] = kwargs["completion_recovery"]
        raise _StopAfterCapture

    patch_attr_in(monkeypatch, (validated_agent_module, discuss_loop, pr_loop, issue_implementation, plan_first_loop_module, triage, issue_loop), "_run_validated_agent", capture_policy)
    config = make_config(tmp_path)

    with pytest.raises(_StopAfterCapture):
        issue_implementation._implement_approved_issue(
            FakeRunner(),
            issue_number=56,
            approved_plan="Approved plan.\n\n### Scope\n- Preserve both APIs.",
            config=config,
            memory=None,
            issue_context=issue_context,
            parent_issue_context=parent_context,
            coder_session_id=None,
            usage_context=validated_agent_module._new_usage_context(config),
        )

    policy = captured["policy"]
    assert policy.human_requirements == (parent_requirement, child_requirement)
    recovery_prompt = build_completion_recovery_prompt(
        config,
        approved_plan_context=policy.approved_plan_context,
        human_requirements=policy.human_requirements,
    )
    assert "Preserve the parent audit trail." in recovery_prompt
    assert "Preserve the child API." in recovery_prompt


def test_codex_issue_loop_creates_pr_then_claude_approves(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    command_names = [cmd[:2] for cmd, _cwd in runner.commands]
    assert ["codex", "exec"] in command_names
    assert ["claude", "--print"] in command_names
    assert len(runner.comments) == 3
    assert "<!-- AGENT_ISSUE_PR_HANDOFF:" in runner.comments[0]
    assert runner.comments[1].startswith("## Issue implementation")
    assert runner.comments[2].startswith("**Review verdict:** Approved\n\nLooks good.")


def test_codex_issue_loop_alternates_until_claude_approval(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Implemented fix.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Addressed Claude's review.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Missing test.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    assert len(runner.comments) == 6
    assert runner.comments[-1].startswith("**Review verdict:** Approved\n\nLGTM.")


def test_codex_issue_loop_requires_codex_to_report_pr_number(tmp_path):
    runner = FakeRunner(
        codex_outputs=["Did some work.\n<!-- AGENT_STATE: blocking -->"],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="valid PR"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("terminal_marker, expected_state", [
    ("<!-- AGENT_STATE: blocking -->", "blocking"),
    ("<!-- AGENT_CLARIFY -->", "clarification"),
])
def test_issue_loop_stops_before_pr_lookup_for_invalid_pr_terminal_result(
    tmp_path, terminal_marker, expected_state
):
    runner = FakeRunner(codex_outputs=[f"Cannot proceed.\n<!-- AGENT_PR: 0 -->\n{terminal_marker}"])
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match=f"implementation is {expected_state}"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    # A genuine coder-declared no-PR terminal is posted to the issue like a
    # PR-success implementation result already is (#588).
    if expected_state == "blocking":
        assert runner.comments[0].startswith("## Issue implementation")
        assert "Cannot proceed." in runner.comments[0]
        assert "No pull request was accepted for handoff." in runner.comments[0]
    else:
        assert runner.comments == [
            f"Cannot proceed.\n<!-- AGENT_PR: 0 -->\n{terminal_marker}\n-- OpenAI Codex: unknown model (medium)"
        ]


def test_issue_loop_invalid_pr_without_terminal_state_is_protocol_error(tmp_path):
    runner = FakeRunner(codex_outputs=["Cannot proceed.\n<!-- AGENT_PR: malformed -->"])
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(
        AgentLoopError,
        match="required structured `issue_implementation` format",
    ):
        run_issue_loop(runner, issue_number=56, config=config)

    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)


def test_issue_loop_rejects_outside_workdir_tests_before_posting_pr_comment(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: python -m pytest https://live.example\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="live remote target"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert runner.comments == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_issue_loop_outside_workdir_after_reported_pr_mentions_confirmed_resume(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: python -m pytest https://live.example\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError) as exc_info:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(exc_info.value)
    assert "live remote target" in message
    assert "PR #77 was confirmed open" in message
    assert "handoff/reviewer comments were not posted" in message
    assert "agent-loop pr 77" in message
    assert runner.comments == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_managed_issue_invalid_post_pr_report_persists_authorization_before_rejection(
    tmp_path, monkeypatch,
):
    runner = _IssueRecoveryWorkflowRunner(
        labeled=True,
        codex_outputs=[
            structured_issue_implementation(
                pr_number=77,
                tests_run=["python3 -m pytest"],
                test_observations=[{
                    "command": "cd /outside && python -m pytest",
                    "receipt_id": "outside-receipt",
                    "claim": "current-result",
                }],
                reviewer="OpenAI Codex",
            )
        ],
        claude_outputs=[
            structured_pr_review(
                state="approved",
                summary="Reviewed the post-report recovery head.",
                reviewer="Anthropic Claude",
            )
        ],
    )
    config = make_config(
        tmp_path, coder="codex", reviewer="claude", managed_ci=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56", trusted_actor="agent-loop",
        protection_mode="voluntary", audit_nonce="opening-nonce",
    )
    handoff = _managed_issue_handoff(nonce="opening-nonce")
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "authenticate_issue_created_handoff", lambda *_a, **_k: handoff
    )

    with pytest.raises(AgentLoopError, match="authorization checkpoint.*persisted"):
        run_issue_loop(runner, issue_number=56, config=config)

    records = [
        parsed
        for comment in runner.authorization_comments
        if (parsed := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    assert len(records) == 1 and records[0].kind == "creation"
    assert runner.comments == []

    # A later ordinary managed issue invocation discovers and enters the real
    # recovery/activation seam for the same head instead of invoking the
    # implementation coder again.
    runner.open_prs_payload = [{"number": 77, "body": "Fixes #56"}]
    runner.pr_commit_pages = _provenance_pages(
        "Implement issue.\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=direct"
    )
    coder_calls = sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    )
    _stop_issue_resume_after_reviewer(monkeypatch)
    with pytest.raises(_RealManagedReviewReached):
        run_issue_loop(runner, issue_number=56, config=config)

    assert sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    ) == coder_calls
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    reviewer_command = next(
        command for command, _cwd in runner.commands if command[:1] == ["claude"]
    )
    assert "abc123" in " ".join(reviewer_command)
    assert any(
        "Reviewed the post-report recovery head." in comment
        for comment in runner.comments
    )
    assert not any(
        marker in comment
        for comment in runner.comments
        for marker in (
            "AGENT_ISSUE_PR_HANDOFF",
            "AGENT_TEST_OBSERVATION", "AGENT_MANAGED_CI_READINESS",
        )
    )


def test_approved_plan_invalid_post_pr_observation_keeps_authorization_resumable(
    tmp_path, monkeypatch,
):
    # Exercise the post-authentication defence independently of acquisition.
    monkeypatch.setattr(issue_implementation, "_validate_followup_workdir_report", lambda parsed, **_k: parsed)
    runner = _IssueRecoveryWorkflowRunner(
        labeled=True,
        codex_outputs=[
            structured_issue_implementation(
                pr_number=77,
                tests_run=["python3 -m pytest"],
                test_observations=[{
                    "command": "cd /outside && python -m pytest",
                    "receipt_id": "outside-receipt",
                    "claim": "current-result",
                }],
                reviewer="OpenAI Codex",
            )
        ],
    )
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer="claude",
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    issue_context = get_issue_context(runner, config=config, issue_number=56)
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56",
        trusted_actor="agent-loop",
        protection_mode="voluntary",
        audit_nonce="opening-nonce",
    )
    handoff = _managed_issue_handoff(nonce="opening-nonce")
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "authenticate_issue_created_handoff", lambda *_a, **_k: handoff
    )

    with pytest.raises(AgentLoopError) as exc_info:
        issue_implementation._implement_approved_issue(
            runner,
            issue_number=56,
            approved_plan="Plan:\n- Preserve managed recovery.",
            config=config,
            memory=None,
            issue_context=issue_context,
            coder_session_id=None,
            usage_context=validated_agent_module._new_usage_context(config),
        )

    message = str(exc_info.value)
    assert "structured test-observation report was invalid" in message
    assert "PR #77 was confirmed open" in message
    assert "authorization checkpoint" in message
    assert "agent-loop pr 77" in message
    records = [
        parsed
        for comment in runner.authorization_comments
        if (parsed := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    assert len(records) == 1 and records[0].kind == "creation"
    assert runner.comments == []
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0


class _MalformedAuthorizationResponseRunner(_IssueRecoveryWorkflowRunner):
    """Return a successful but unverifiable response for the auth comment POST."""

    def _run_locked(self, args, *, cwd, check, input_text=None):
        command = list(args)
        endpoint = next(
            (part for part in command if part.startswith("repos/")), ""
        )
        body = self._form_value(command, "body") or ""
        if (
            endpoint == "repos/OWNER/REPO/issues/77/comments"
            and "POST" in command
            and "ISSUE_AUTHORIZATION" in body
        ):
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(recorded, cwd_path, "{}", "", 0)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


class _RealManagedActivationReached(Exception):
    pass


def _stop_issue_resume_after_real_activation(monkeypatch):
    patch_attr_in(monkeypatch, (architecture_contract, pr_loop, issue_implementation, issue_loop), "_freeze_prompt_architecture",
        lambda _runner, config, **_kwargs: config,
    )
    real_activate_managed_ci = pr_loop.activate_managed_ci

    def activate_then_stop(*args, **kwargs):
        result = real_activate_managed_ci(*args, **kwargs)
        raise _RealManagedActivationReached(result)

    monkeypatch.setattr(
        pr_loop,
        "activate_managed_ci",
        activate_then_stop,
    )


class _RealManagedReviewReached(Exception):
    pass


def _stop_issue_resume_after_reviewer(monkeypatch):
    patch_attr_in(monkeypatch, (architecture_contract, pr_loop, issue_implementation, issue_loop), "_freeze_prompt_architecture",
        lambda _runner, config, **_kwargs: config,
    )
    real_post_pr_comment = pr_loop_support.post_pr_comment

    def post_and_stop(*args, **kwargs):
        result = real_post_pr_comment(*args, **kwargs)
        if str(kwargs.get("body") or "").startswith("**Review verdict:**"):
            raise _RealManagedReviewReached
        return result

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop, issue_loop), "post_pr_comment", post_and_stop)


def test_managed_issue_resume_reviews_same_head_after_post_pr_report_rejection(
    tmp_path, monkeypatch,
):
    runner = _IssueRecoveryWorkflowRunner(
        labeled=True,
        codex_outputs=[
            "Fixed issue.\nTests: python -m pytest https://live.example\n"
            "<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex"
        ],
        claude_outputs=[
            structured_pr_review(state="approved", summary="Reviewed the resumed exact head.")
        ],
    )
    config = make_config(
        tmp_path, coder="codex", reviewer="claude", managed_ci=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        pre_review_tests=False,
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56", trusted_actor="agent-loop",
        protection_mode="voluntary", audit_nonce="opening-nonce",
    )
    handoff = _managed_issue_handoff(nonce="opening-nonce")
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "authenticate_issue_created_handoff", lambda *_a, **_k: handoff
    )

    with pytest.raises(AgentLoopError, match="authorization checkpoint.*persisted"):
        run_issue_loop(runner, issue_number=56, config=config)

    runner.open_prs_payload = [{"number": 77, "body": "Fixes #56"}]
    runner.pr_commit_pages = _provenance_pages(
        "Implement issue.\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=direct"
    )
    coder_calls = sum(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)
    _stop_issue_resume_after_reviewer(monkeypatch)

    with pytest.raises(_RealManagedReviewReached):
        run_issue_loop(runner, issue_number=56, config=config)

    assert sum(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands) == coder_calls
    assert sum(command[:1] == ["claude"] for command, _cwd in runner.commands) == 1
    reviewer_command = next(command for command, _cwd in runner.commands if command[:1] == ["claude"])
    assert "abc123" in " ".join(reviewer_command)
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert any("Reviewed the resumed exact head." in comment for comment in runner.comments)
    assert not any(
        marker in comment
        for comment in runner.comments
        for marker in (
            "AGENT_ISSUE_PR_HANDOFF", "AGENT_TEST_OBSERVATION",
            "AGENT_MANAGED_CI_READINESS",
        )
    )


def test_invalid_post_pr_report_then_issue_resume_runs_real_activation_without_reimplementation(
    tmp_path, monkeypatch,
):
    runner = _IssueRecoveryWorkflowRunner(
        labeled=True,
        codex_outputs=[
            "Fixed issue.\nTests: python -m pytest https://live.example\n"
            "<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex"
        ],
        claude_outputs=[
            structured_pr_review(state="approved", summary="Reviewed the durable recovery head.")
        ],
    )
    config = make_config(
        tmp_path, coder="codex", reviewer="claude", managed_ci=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56", trusted_actor="agent-loop",
        protection_mode="voluntary", audit_nonce="opening-nonce",
    )
    handoff = _managed_issue_handoff(nonce="opening-nonce")
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "authenticate_issue_created_handoff", lambda *_a, **_k: handoff
    )

    with pytest.raises(AgentLoopError, match="authorization checkpoint.*persisted"):
        run_issue_loop(runner, issue_number=56, config=config)

    records = [
        parsed
        for comment in runner.authorization_comments
        if (parsed := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    assert len(records) == 1 and records[0].kind == "creation"
    assert runner.comments == []
    coder_calls = sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    )

    runner.open_prs_payload = [{"number": 77, "body": "Fixes #56"}]
    runner.pr_commit_pages = _provenance_pages(
        "Implement issue.\n\nAgent-Issue-Provenance: v1 "
        "repo=owner/repo issue=56 flow=direct"
    )
    _stop_issue_resume_after_reviewer(monkeypatch)
    recovery_start = len(runner.commands)
    with pytest.raises(_RealManagedReviewReached):
        run_issue_loop(runner, issue_number=56, config=config)

    recovery_commands = runner.commands[recovery_start:]
    assert any("repos/OWNER/REPO/pulls/77" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("repos/OWNER/REPO/issues/77/comments?" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("actions/variables/AGENT_LOOP_MANAGED_ACTOR" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("contents/.github/workflows/ci.yml" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("repos/OWNER/REPO/commits/main" in " ".join(command) for command, _cwd in recovery_commands)
    assert sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    ) == coder_calls
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    reviewer_command = next(command for command, _cwd in runner.commands if command[:1] == ["claude"])
    assert "abc123" in " ".join(reviewer_command)
    assert any("Reviewed the durable recovery head." in comment for comment in runner.comments)
    assert not any(
        marker in comment
        for comment in runner.comments
        for marker in (
            "AGENT_ISSUE_PR_HANDOFF",
            "AGENT_TEST_OBSERVATION", "AGENT_MANAGED_CI_READINESS",
        )
    )


def test_pre_pr_number_response_rejection_then_fresh_issue_recovery_uses_real_activation(
    tmp_path, monkeypatch,
):
    valid = structured_issue_implementation(
        pr_number=77,
        tests_run=["python3 -m pytest tests/test_managed_ci.py -q"],
    )
    payload, end = json.JSONDecoder().raw_decode(valid)
    payload.pop("architecture_impact")
    rejected = json.dumps(payload) + valid[end:]
    runner = _IssueRecoveryWorkflowRunner(
        labeled=False,
        codex_outputs=[rejected, rejected],
        claude_outputs=[
            structured_pr_review(state="approved", summary="Reviewed after explicit recovery.")
        ],
    )
    ordinary_config = make_config(
        tmp_path, coder="codex", reviewer="claude", managed_ci=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        agent_max_retries=0,
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56", trusted_actor="agent-loop",
        protection_mode="voluntary", audit_nonce="opening-nonce",
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    patch_attr_in(monkeypatch, (validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair",
        lambda *_a, **_k: (None, None, ()),
    )

    with pytest.raises(AgentLoopError, match="architecture_impact"):
        run_issue_loop(runner, issue_number=56, config=ordinary_config)

    assert runner.authorization_comments == []
    assert runner.comments == []
    coder_calls = sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    )

    runner.open_prs_payload = [{"number": 77, "body": "Fixes #56"}]
    runner.pr_commit_pages = _provenance_pages(
        "Implement issue.\n\nAgent-Issue-Provenance: v1 "
        "repo=owner/repo issue=56 flow=direct"
    )
    fresh_config = replace(
        ordinary_config,
        managed_ci_fresh_authorization=True,
        invocation_argv=(
            "agent-loop", "issue", "56", "--managed-ci", "--managed-ci-fresh",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    _stop_issue_resume_after_reviewer(monkeypatch)
    recovery_start = len(runner.commands)
    with pytest.raises(_RealManagedReviewReached):
        run_issue_loop(runner, issue_number=56, config=fresh_config)

    recovery_commands = runner.commands[recovery_start:]
    assert any("repos/OWNER/REPO/pulls/77" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("repos/OWNER/REPO/issues/77/comments?" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("repos/OWNER/REPO/issues/56/timeline?" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("actions/variables/AGENT_LOOP_MANAGED_ACTOR" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("contents/.github/workflows/ci.yml" in " ".join(command) for command, _cwd in recovery_commands)
    records = [
        parsed
        for comment in runner.authorization_comments
        if (parsed := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    assert len(records) == 1 and records[0].kind == "fresh"
    assert sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    ) == coder_calls
    assert runner.labels_posted is True
    assert runner.dispatch_count == 0
    reviewer_command = next(command for command, _cwd in runner.commands if command[:1] == ["claude"])
    assert "abc123" in " ".join(reviewer_command)
    assert any("Reviewed after explicit recovery." in comment for comment in runner.comments)
    assert not any(
        marker in comment
        for comment in runner.comments
        for marker in (
            "AGENT_ISSUE_PR_HANDOFF", "AGENT_TEST_OBSERVATION",
            "AGENT_MANAGED_CI_READINESS",
        )
    )


def test_strict_pre_pr_number_rejection_directs_ordinary_discovery_without_reimplementation(
    tmp_path, monkeypatch,
):
    valid = structured_issue_implementation(
        pr_number=77,
        tests_run=["python3 -m pytest tests/test_managed_ci.py -q"],
    )
    payload, end = json.JSONDecoder().raw_decode(valid)
    payload.pop("architecture_impact")
    rejected = json.dumps(payload) + valid[end:]
    runner = _IssueRecoveryWorkflowRunner(
        labeled=True,
        codex_outputs=[rejected],
        claude_outputs=[
            structured_pr_review(
                state="approved", summary="Reviewed the strict recovered PR."
            )
        ],
        pr_branch_protection_payload={"contexts": ["final-ci/exact-head"]},
    )
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer="claude",
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        agent_max_retries=0,
        invocation_argv=(
            "agent-loop", "issue", "56", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
        ),
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56",
        trusted_actor="agent-loop",
        protection_mode="strict",
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    patch_attr_in(monkeypatch, (validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair",
        lambda *_a, **_k: (None, None, ()),
    )

    with pytest.raises(AgentLoopError, match="ordinary managed-CI") as exc_info:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(exc_info.value)
    assert "--managed-ci-fresh" not in message
    assert "unprotected fresh-authorization path is unavailable" in message
    coder_calls = sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    )

    # The PR was created before the structured response was rejected. Ordinary
    # issue discovery can now reach that same strict draft without invoking the
    # implementation coder again.
    runner.open_prs_payload = [{"number": 77, "body": "Fixes #56"}]
    runner.pr_payload["body"] = "Fixes #56"
    runner.rest_pr["body"] = "Fixes #56"
    runner.pr_commit_pages = _provenance_pages(
        "Implement issue.\n\nAgent-Issue-Provenance: v1 "
        "repo=owner/repo issue=56 flow=direct"
    )
    _stop_issue_resume_after_reviewer(monkeypatch)
    with pytest.raises(_RealManagedReviewReached):
        run_issue_loop(runner, issue_number=56, config=config)

    assert sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    ) == coder_calls


@pytest.mark.parametrize(
    ("protection_mode", "unreadable_waiver", "fresh_offered"),
    [
        ("unreadable", True, True),
        ("unreadable", False, False),
        ("voluntary", False, True),
        ("plan_limited", False, True),
    ],
)
def test_pre_pr_number_rejection_guidance_follows_the_state_specific_waiver(
    tmp_path, monkeypatch, protection_mode, unreadable_waiver, fresh_offered,
):
    valid = structured_issue_implementation(
        pr_number=77,
        tests_run=["python3 -m pytest tests/test_managed_ci.py -q"],
    )
    payload, end = json.JSONDecoder().raw_decode(valid)
    payload.pop("architecture_impact")
    rejected = json.dumps(payload) + valid[end:]
    runner = _IssueRecoveryWorkflowRunner(labeled=True, codex_outputs=[rejected])
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer="claude",
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        allow_unreadable_protection=unreadable_waiver,
        agent_max_retries=0,
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56",
        trusted_actor="agent-loop",
        protection_mode=protection_mode,
    )
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    patch_attr_in(monkeypatch, (validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair",
        lambda *_a, **_k: (None, None, ()),
    )

    with pytest.raises(AgentLoopError, match="rejected before a PR number was accepted") as exc_info:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(exc_info.value)
    assert "do not rerun implementation" in message
    if fresh_offered:
        assert "--managed-ci-fresh" in message
        if protection_mode == "unreadable":
            assert "--allow-unprotected-managed-ci --allow-unreadable-protection" in message
        else:
            assert "(including the unprotected waiver);" in message
    else:
        assert "--managed-ci-fresh" not in message
        assert "Fresh authorization is unavailable" in message
        assert "--allow-unreadable-protection" in message


class _CloudSessionIssueRunner(_IssueRecoveryWorkflowRunner):
    """Refuse the Actions variable and classic protection reads with gh 403s (#1040)."""

    _REFUSED = {
        "/actions/variables/AGENT_LOOP_MANAGED_ACTOR": (
            "gh: Access to this GitHub Actions path is not permitted through this proxy (HTTP 403)\n"
        ),
        "/branches/main/protection/required_status_checks": (
            "gh: Resource not accessible by integration (HTTP 403)\n"
        ),
    }

    def _run_locked(self, args, *, cwd, check, input_text=None):
        endpoint = next((part for part in args if part.startswith("repos/")), "")
        for suffix, stderr in self._REFUSED.items():
            if endpoint.endswith(suffix) and "--method" not in args:
                recorded, cwd_path = self._record_command(args, cwd)
                return CommandResult(recorded, cwd_path, "", stderr, 1)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


class _CloudSessionCreatingCoderRunner(_CloudSessionIssueRunner):
    """The fake coder creates the reserved managed PR its prompt describes."""

    def __init__(self, **kwargs):
        super().__init__(labeled=False, **kwargs)
        # No PR exists until the coder creates one.
        self.pr_payload["body"] = "Fixes #56"
        self.rest_pr["body"] = "Fixes #56"
        self.created_pr_body = None

    def _run_with_log_locked(self, args, *, cwd, log_path, check, input_text=None):
        self._create_reserved_pr(args, input_text)
        return super()._run_with_log_locked(
            args, cwd=cwd, log_path=log_path, check=check, input_text=input_text,
        )

    def _run_locked(self, args, *, cwd, check, input_text=None):
        self._create_reserved_pr(args, input_text)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)

    def _create_reserved_pr(self, args, input_text):
        if [str(part) for part in args[:2]] == ["codex", "exec"] and self.created_pr_body is None:
            prompt = " ".join(str(part) for part in args) + (input_text or "")
            branch = re.search(r"reserved branch `([^`]+)`", prompt)
            nonce = re.search(r"nonce=([A-Za-z0-9_-]+)", prompt)
            assert branch is not None and branch.group(1) == "agent-loop/managed-56"
            assert "gh pr create --draft --label agent-loop-managed" in prompt
            assert nonce is not None
            # Mirrors `gh pr create --draft --label agent-loop-managed --body-file`.
            body = f"Fixes #56\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce={nonce.group(1)}"
            self.created_pr_body = body
            self.pr_payload["body"] = body
            self.rest_pr["body"] = body
            self.rest_pr["draft"] = True
            self.rest_pr["labels"] = [{"name": "agent-loop-managed"}]


@pytest.mark.parametrize("unreadable_waiver", [True, False])
def test_cloud_session_issue_run_creates_reserved_managed_pr_only_with_both_waivers(
    tmp_path, monkeypatch, capsys, unreadable_waiver,
):
    # The coder creates the reserved PR and reports it. Its legacy report is
    # rejected only after the orchestrator has accepted the PR number,
    # authenticated the opening tuple, and persisted the creation authorization.
    runner = _CloudSessionCreatingCoderRunner(
        codex_outputs=[
            "Fixed issue.\nTests: python -m pytest https://live.example\n"
            "<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex"
        ],
    )
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer="claude",
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        allow_unreadable_protection=unreadable_waiver,
    )

    with pytest.raises(AgentLoopError) as exc_info:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(exc_info.value)
    commands = [command for command, _cwd in runner.commands]
    coder_calls = [command for command in commands if command[:2] == ["codex", "exec"]]
    # The orchestrator itself never applies a label or dispatches: the coder
    # creates the PR born draft and labeled.
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    records = [
        parsed
        for comment in runner.authorization_comments
        if (parsed := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    if not unreadable_waiver:
        assert "--allow-unreadable-protection" in message
        assert "no PR was created" in message
        assert coder_calls == []
        assert runner.created_pr_body is None
        assert records == []
        assert runner.rest_pr["labels"] == []
        assert not any("--method" in command for command in commands)
        return

    assert runner.created_pr_body is not None, message
    assert re.search("authorization checkpoint.*persisted", message), (message, runner.created_pr_body)
    assert len(coder_calls) == 1
    assert runner.rest_pr["labels"] == [{"name": "agent-loop-managed"}]
    body_nonce = runner.created_pr_body.rsplit("nonce=", 1)[1]
    assert len(records) == 1
    record = records[0]
    assert (record.kind, record.pr_number, record.issue_number) == ("creation", 77, 56)
    assert (record.protection, record.waiver) == ("unreadable", "allow-unreadable-protection")
    assert record.nonce == body_nonce
    assert record.actor_login == "agent-loop"
    assert "--allow-unreadable-protection is also active" in capsys.readouterr().out


def test_managed_issue_legacy_recovery_rejects_unexpected_closing_reference(
    tmp_path,
):
    body = "Fixes #56\nCloses #999"
    runner = _IssueRecoveryWorkflowRunner(
        labeled=False,
        open_prs_payload=[{"number": 77, "body": body}],
        pr_payload={"body": body},
    )
    config = make_config(
        tmp_path, coder="codex", reviewer="claude", managed_ci=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
    )

    with pytest.raises(AgentLoopError, match="outside the expected contract"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert runner.comments == []
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert not any(
        command[:1] in (["claude"], ["codex"])
        for command, _cwd in runner.commands
    )


def test_managed_issue_authorization_publication_failure_prints_fresh_recovery(
    tmp_path, monkeypatch,
):
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "issue", "56", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    monkeypatch.setattr(
        issue_implementation, "publish_issue_created_authorization",
        lambda *_a, **_k: (_ for _ in ()).throw(AgentLoopError("comment returned no ID")),
    )
    with pytest.raises(AgentLoopError) as exc_info:
        issue_implementation._publish_issue_authorization_with_recovery(
            FakeRunner(), config=config,
            handoff=_managed_issue_handoff(nonce="nonce"),
            metadata=architecture_contract.PullRequestMetadata(
                number=77, repo="OWNER/REPO", title="PR",
                head_branch="agent-loop/managed-56", base_branch="main",
                head_sha="abc123", url="https://github.test/pull/77",
            ),
            issue_number=56,
        )
    message = str(exc_info.value)
    assert "publication was interrupted" in message
    assert "--managed-ci-fresh" in message
    assert "agent-loop issue 56" in message


def test_managed_issue_publication_malformed_response_stops_before_handoff_or_review(
    tmp_path, monkeypatch,
):
    runner = _MalformedAuthorizationResponseRunner(
        labeled=True,
        codex_outputs=[
            "Implemented.\nTests: python3 -m pytest tests/test_managed_ci.py\n"
            "<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(
        tmp_path, coder="codex", reviewer="claude", managed_ci=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56", trusted_actor="agent-loop",
        protection_mode="voluntary", audit_nonce="opening-nonce",
    )
    handoff = _managed_issue_handoff(nonce="opening-nonce")
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: intent)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "authenticate_issue_created_handoff", lambda *_a, **_k: handoff)

    with pytest.raises(AgentLoopError, match="publication was interrupted"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert any(
        endpoint in " ".join(command)
        for command, _cwd in runner.commands
        for endpoint in ("repos/OWNER/REPO/issues/77/comments",)
        if "POST" in command and "ISSUE_AUTHORIZATION" in " ".join(command)
    )
    assert runner.authorization_comments == []
    assert runner.comments == []
    assert not any(command[:1] == ["claude"] for command, _cwd in runner.commands)


def test_issue_fresh_recovery_discovers_pre_handoff_pr_without_reimplementing(
    tmp_path, monkeypatch,
):
    runner = _IssueRecoveryWorkflowRunner(
        labeled=False,
        open_prs_payload=[{"number": 77, "body": "Fixes #56"}],
        pr_commit_pages=_provenance_pages(
            "Implement issue.\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=direct"
        ),
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Reviewed the fresh-recovery exact head.",
                reviewer="OpenAI Codex",
            )
        ],
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_fresh_authorization=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "issue", "56", "--managed-ci", "--managed-ci-fresh",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    _stop_issue_resume_after_reviewer(monkeypatch)

    with pytest.raises(_RealManagedReviewReached):
        run_issue_loop(runner, issue_number=56, config=config)

    records = [
        parsed
        for comment in runner.authorization_comments
        if (parsed := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    assert len(records) == 1 and records[0].kind == "fresh"
    assert runner.labels_posted is True
    assert runner.dispatch_count == 0
    assert sum(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands) == 1
    assert any(
        "Reviewed the fresh-recovery exact head." in comment
        for comment in runner.comments
    )


def test_issue_loop_outside_workdir_after_reported_pr_hedges_unconfirmed_pr(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: cd /outside && python -m pytest\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        pr_payload={"state": "CLOSED"},
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError) as exc_info:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(exc_info.value)
    assert "PR #77 is CLOSED" in message
    assert "agent-loop pr 77" not in message
    assert runner.comments == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_issue_loop_accepts_absolute_interpreter_test_command_through_response_path(tmp_path):
    # Regression for #584: an absolute system-interpreter path in program
    # position (the only path outside the assigned checkout) must not be
    # misclassified as an external test location when reported through the
    # freeform `Tests:` response path (origin='response', orchestrator.py
    # line ~3286/6223), matching the PR #484 follow-up reproducer.
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: `/usr/bin/python3 -m pytest tests/test_durable_jobs.py -q` "
            "- 12 passed, run from the assigned checkout.\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0
    assert len(runner.comments) >= 1


def test_issue_loop_records_out_of_checkout_baseline_as_context(tmp_path):
    """Issue #991: an honestly reported clean-base baseline must not reject the hand-off."""
    baseline = (
        "PYTHONPATH=/tmp/scratch-main-1176 timeout 900 /usr/bin/python3 -m pytest "
        "/tmp/scratch-main-1176/tests/ -q"
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_issue_implementation(
                pr_number=77,
                tests_run=["python3 -m pytest tests/test_api.py -q", baseline],
                reviewer="OpenAI Codex",
            )
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")
    # The same baseline also ran (and failed) through the managed broker.
    from coding_review_agent_loop.local_test_evidence import LocalTestObservation

    runner._local_test_observations.append(LocalTestObservation(
        command=("python3", "-m", "pytest", "/tmp/scratch-main-1176/tests/", "-q"),
        outcome="failed",
        provenance="parent-observed",
        receipt_id="baseline-broker-failure",
        turn_id="turn-baseline",
        timestamp="2026-09-23T10:00:00+00:00",
        cwd=str(tmp_path),
    ))

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    handoff = next(
        comment for comment in runner.comments if comment.startswith("## Issue implementation")
    )
    tests_section, _, context_section = handoff.partition(
        "### Out-of-checkout context runs (not evidence)"
    )
    assert "python3 -m pytest tests/test_api.py -q" in tests_section
    assert "/tmp/scratch-main-1176" not in tests_section
    assert f"- {baseline}" in context_section
    baseline_line = next(
        line for line in handoff.splitlines() if "baseline-broker-failure" in line
    )
    assert "out-of-checkout context (not evidence) `failed`" in baseline_line
    assert "authoritative" not in baseline_line


def test_issue_loop_live_target_after_reported_pr_mentions_confirmed_resume(tmp_path):
    # Same post-PR guidance wrapper as the outside-workdir case above, but
    # triggered by a live-remote-target rejection instead of a path
    # rejection, proving the URL pass also reaches
    # _validate_response_tests_with_post_pr_context.
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: ran curl https://live.example/health to verify.\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError) as exc_info:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(exc_info.value)
    assert "live remote target" in message
    assert "PR #77 was confirmed open" in message
    assert "handoff/reviewer comments were not posted" in message
    assert "agent-loop pr 77" in message
    assert runner.comments == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_issue_loop_rejects_reported_pr_when_assigned_head_unchanged(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: python -m pytest passed.\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        advance_git_head_on_pr=False,
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="HEAD did not advance"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert runner.comments == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_gemini_issue_loop_creates_pr_then_codex_approves(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            "Fixed issue.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        ],
        codex_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="gemini", reviewer="codex")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["gemini"], ["codex"])]
    assert agent_commands == [["gemini", "--prompt"], ["codex", "exec"]]
    assert len(runner.comments) == 3
    assert "<!-- AGENT_ISSUE_PR_HANDOFF:" in runner.comments[0]
    assert runner.comments[1].startswith("## Issue implementation")
    assert runner.comments[2].startswith("**Review verdict:** Approved\n\nLooks good.")


def test_gemini_issue_loop_resumes_session_for_followup(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            json.dumps({
                "response": "Fixed issue.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
                "session_id": "gemini-session-1",
            }),
            # Plain-text output intentionally clears the tracked session; a third
            # Gemini turn would start without --resume.
            "Addressed review.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        ],
        codex_outputs=[
            "Needs a regression test.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer="codex",
        gemini_args=("--output-format", "json"),
    )

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    gemini_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"]]
    assert len(gemini_calls) == 2
    assert "--resume" not in gemini_calls[0]
    assert gemini_calls[1][-2:] == ["--resume", "gemini-session-1"]


def test_issue_loop_plan_first_one_shot_rerun_resumes_pr_loop(tmp_path):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    handoff = format_one_shot_impl_handoff_comment(
        parent_issue=56,
        mode="implement-one-shot",
        plan_hash=approved_plan_hash(plan),
        plan_subject=_plan_subject(plan),
        pr_number=77,
        pr_head_sha="abc123",
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": handoff},
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 0
    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_issue_scratch_input_report_stays_in_checkout(tmp_path, monkeypatch):
    import coding_review_agent_loop.workdir_guard as guard
    root = tmp_path / 'coding-review-agent-loop'
    monkeypatch.setattr(guard, 'scratch_root', lambda: root)
    audit = (
        'python scripts/dependency_audit.py evaluate --policy .github/dependency-audit-policy.json '
        f'--python-report {root}/scratch/1837/pip-audit.json '
        f'--npm-report {root}/scratch/1837/npm-audit.json'
    )
    runner = FakeRunner(
        codex_outputs=[structured_issue_implementation(tests_run=[audit], reviewer='OpenAI Codex')],
        claude_outputs=[structured_pr_review(state='approved', reviewer='Anthropic Claude')],
    )
    config = make_config(tmp_path, coder='codex', reviewer='claude')
    assert run_issue_loop(runner, issue_number=56, config=config) == 0
    assert any(audit in comment for comment in runner.comments)
    assert not any('Out-of-checkout context runs' in comment for comment in runner.comments)


@pytest.mark.parametrize("managed", [False, True])
def test_initial_report_correction_exhaustion_precedes_authentication(tmp_path, monkeypatch, managed):
    from coverage_reask_helpers import CoverageHarness, response_text
    from test_issue_implementation_coverage_reask import _report_refusal, _claude_commands
    refused = _report_refusal(response_text())
    h = CoverageHarness(tmp_path, monkeypatch, real_agent=True, applicable=False,
                        managed_ci=managed, claude_outputs=[refused, refused])
    if managed:
        intent = ManagedCiCreationIntent(branch="agent-loop/managed-56", trusted_actor="agent-loop",
                                        protection_mode="voluntary", audit_nonce="nonce")
        monkeypatch.setattr(issue_implementation, "preflight_managed_ci_creation", lambda *_a, **_k: intent)
    authenticated = []
    monkeypatch.setattr(issue_implementation, "authenticate_issue_created_handoff", lambda *_a, **_k: authenticated.append(1))
    with pytest.raises(AgentLoopError) as error:
        h.run()
    assert "semantic-evidence-rejection" in str(error.value)
    assert "confirmed open" not in str(error.value)
    assert "agent-loop pr 77" not in str(error.value)
    assert len(_claude_commands(h)) == 2
    assert not authenticated
    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in h.runner.commands)
    assert not h.all_comments()
    assert not h.run_pr_calls
