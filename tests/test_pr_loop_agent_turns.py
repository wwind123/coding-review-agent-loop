"""PR-loop tests for validated_agent, agent_failure and repair: PR-loop agent turns (#1542)."""

import dataclasses
import datetime
import json
import os
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop, run_pr_loop
from coding_review_agent_loop.errors import QuotaResetExceededError
from coding_review_agent_loop.github import (
    HumanReviewRequirement,
    IssueComment,
    PullRequestMetadata,
    PullRequestReviewContext,
)
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _decode_round_metadata,
    _resume_pr_round,
    _strip_round_metadata,
)
from coding_review_agent_loop.agent_failure import ValidatedAgentResponse
from coding_review_agent_loop.protocol import (
    parse_plan_review,
    parse_pr_review,
    validate_structured_issue_implementation,
)
from agent_loop_helpers import (
    FakeRunner,
    command_index,
    make_config,
    structured_issue_implementation,
    structured_plan_review,
    structured_pr_review,
)
import coding_review_agent_loop.agent_failure as agent_failure
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.triage as triage
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_attr_in, patch_object_in
import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from agent_loop_helpers import FakeRunner as _FakeRunner


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

    monkeypatch.setattr(agent_failure.datetime, "datetime", FixedDateTime)
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
    assert ["git", "reset", "--hard", runner.git_head] in commands
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
    with patch_object_in((
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent",
        side_effect=fake_run_validated_agent,
    ):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    # First PR comment is the reviewer blocking; second is the coder followup
    followup_body = runner.pr_payload["comments"][2]["body"]
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
            runner.simulate_agent_turn(config, config.claude_dir, head=before_head + "-coder")
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
    with patch_object_in((
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent",
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
        "coding_review_agent_loop.validated_agent.attempt_repair",
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
        "coding_review_agent_loop.validated_agent.attempt_repair",
        lambda raw, gemini_cmd, **kwargs: _fabricated_pr_review(),
    ):
        with pytest.raises(AgentLoopError, match="review_substance_integrity"):
            run_pr_loop(runner, pr_number=77, config=config)

    assert not any("PR review incomplete" in body for body in runner.comments)
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


# --- #925 round 5: the PR acknowledgement repair through the real flow -------

_ACK_UNCORROBORATED = {"status": "modified", "rationale": "Something changed."}


def _ack_with_impact(rendered, impact):
    split = rendered.index("}\n") + 1
    payload = json.loads(rendered[:split])
    if impact is None:
        payload.pop("architecture_impact", None)
    else:
        payload["architecture_impact"] = impact
    return json.dumps(payload) + rendered[split:]


def _ack_pr_run(tmp_path, monkeypatch, *, repaired_status=None):
    requirement = HumanReviewRequirement(
        source_type="PR comment",
        author="maintainer",
        created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="Keep the current audit trail.",
    )
    metadata = PullRequestMetadata(
        number=77, repo="OWNER/REPO", title="Acknowledgement repair",
        head_branch="feature/review-context", base_branch="main",
        head_sha="abc123", url="https://github.com/OWNER/REPO/pull/77",
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_pr_review_context",
        lambda *args, **kwargs: PullRequestReviewContext(
            metadata=metadata, comments=(), human_requirements=(requirement,)
        ),
    )
    # The reviewer approves with an uncorroborated near miss and no
    # acknowledgement; acceptance removes the assessment and records it.
    unacknowledged = _ack_with_impact(
        structured_pr_review(summary="Codex approves.", reviewer="OpenAI Codex"),
        _ACK_UNCORROBORATED,
    )
    repaired = _ack_with_impact(
        structured_pr_review(
            summary="Codex approves.", reviewer="OpenAI Codex",
            human_requirements_resolved=True,
        ),
        None if repaired_status is None
        else {"status": repaired_status, "rationale": "No architectural change."},
    )
    runner = FakeRunner(codex_outputs=[unacknowledged] * 3)
    repair_calls = []

    def fake_repair(raw, cmd, **kwargs):
        repair_calls.append(raw)
        return repaired

    config = make_config(
        tmp_path, reviewer="codex", max_rounds=1,
        agent_max_retries=0, agent_retry_backoff_seconds=0,
    )
    repair_results = []
    real_structured_repair = validated_agent._run_structured_repair

    def spy_repair(*args, **kwargs):
        result = real_structured_repair(*args, **kwargs)
        repair_results.append(result)
        return result

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_repair), patch_object_in((
        validated_agent, pr_loop, plan_first_loop,
    ), "_run_structured_repair", spy_repair
    ):
        try:
            outcome = run_pr_loop(runner, pr_number=77, config=config)
        except AgentLoopError as exc:
            outcome = exc
    _ack_pr_run.repair_results = repair_results
    records = []
    for comment in runner.pr_payload["comments"]:
        match = re.search(
            r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", comment["body"]
        )
        if match:
            decoded = _decode_round_metadata(match.group("payload"))
            if decoded.role == "reviewer":
                records.append(decoded)
    return outcome, records, repair_calls


def test_pr_acknowledgement_repair_keeps_the_accepted_assessment_and_record(
    tmp_path, monkeypatch
):
    outcome, records, repair_calls = _ack_pr_run(tmp_path, monkeypatch)
    assert repair_calls, "the acknowledgement repair must run through the flow"
    assert '"modified"' not in repair_calls[0]
    # Exactly one reviewer publication, made from the accepted carrier before
    # the gate; the successful repair posts nothing new.
    assert len(records) == 1
    assert records[0].architecture_impact is None
    assert [r.outcome for r in records[0].architecture_impact_degradations] == [
        "degraded-to-undetermined"
    ]
    assert outcome == 0, outcome


@pytest.mark.parametrize("status", ["unchanged", "none"])
def test_pr_acknowledgement_repair_cannot_add_an_assessment(tmp_path, monkeypatch, status):
    outcome, records, repair_calls = _ack_pr_run(tmp_path, monkeypatch, repaired_status=status)
    assert repair_calls
    refusal = {"unchanged": "must not introduce one", "none": "must be `changed` or `unchanged`"}
    assert _ack_pr_run.repair_results
    for _text, validated, attempts in _ack_pr_run.repair_results:
        assert validated is None
        assert refusal[status] in attempts[-1].diagnostic
    # The refused repair leaves the approval unacknowledged: nothing merges.
    assert outcome != 0
    assert all(record.architecture_impact is None for record in records)


def _ack_pr_resume_run(tmp_path, monkeypatch, *, acknowledged, parallel):
    """Interrupt right after Codex's review is posted, then resume the round."""
    requirement = HumanReviewRequirement(
        source_type="PR comment",
        author="maintainer",
        created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="Keep the current audit trail.",
    )
    real_context = pr_loop_support.get_pr_review_context
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_pr_review_context",
        lambda *args, **kwargs: dataclasses.replace(
            real_context(*args, **kwargs), human_requirements=(requirement,)
        ),
    )
    review = _ack_with_impact(
        structured_pr_review(
            summary="Codex approves.", reviewer="OpenAI Codex",
            human_requirements_resolved=acknowledged,
        ),
        _ACK_UNCORROBORATED,
    )
    repaired = structured_pr_review(
        summary="Codex approves.", reviewer="OpenAI Codex", human_requirements_resolved=True
    )
    runner = FakeRunner(codex_outputs=[review] * 3)
    config = make_config(
        tmp_path, reviewer="codex", max_rounds=1, review_parallel=parallel,
        agent_max_retries=0, agent_retry_backoff_seconds=0,
    )

    def codex_records():
        found = []
        for comment in runner.pr_payload["comments"]:
            match = re.search(
                r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", comment["body"]
            )
            if match:
                decoded = _decode_round_metadata(match.group("payload"))
                if decoded.role == "reviewer" and decoded.agent == "Codex":
                    found.append(decoded)
        return found

    real_post = pr_loop_support.post_pr_comment
    interrupted = []

    def interrupt_after_review(*args, **kwargs):
        result = real_post(*args, **kwargs)
        if not interrupted and codex_records():
            interrupted.append(True)
            raise KeyboardInterrupt
        return result

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", lambda raw, cmd, **kw: repaired), \
            patch_object_in((
                pr_loop_support, pr_loop, issue_loop,
            ), "post_pr_comment", side_effect=interrupt_after_review):
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)
    codex_before = sum(1 for cmd, _cwd in runner.commands if cmd[:1] == ["codex"])

    resumed, carriers, repair_calls = [], [], []
    real_resumed = architecture_contract._resumed_review_architecture
    real_pin = architecture_contract._pin_acknowledgement_repair

    def spy_resumed(*args, **kwargs):
        result = real_resumed(*args, **kwargs)
        resumed.append(result)
        return result

    def spy_pin(accepted, repaired_value, **kwargs):
        carriers.append(accepted)
        return real_pin(accepted, repaired_value, **kwargs)

    def fake_repair(raw, cmd, **kwargs):
        repair_calls.append(raw)
        return repaired

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_repair), patch_object_in((
        architecture_contract, pr_loop, plan_first_loop,
    ), "_resumed_review_architecture", spy_resumed
    ), patch_object_in((
        architecture_contract, pr_loop, plan_first_loop,
    ), "_pin_acknowledgement_repair", spy_pin):
        try:
            outcome = run_pr_loop(runner, pr_number=77, config=config)
        except AgentLoopError as exc:
            outcome = exc
    codex_after = sum(1 for cmd, _cwd in runner.commands if cmd[:1] == ["codex"])
    return SimpleNamespace(
        outcome=outcome, resumed=resumed, carriers=carriers, repair_calls=repair_calls,
        records=codex_records(), reinvoked=codex_after - codex_before,
    )


@pytest.mark.parametrize("parallel", [False, True], ids=["authoritative", "publication"])
def test_resumed_acknowledged_pr_review_is_rebuilt_from_round_metadata(
    tmp_path, monkeypatch, parallel
):
    run = _ack_pr_resume_run(tmp_path, monkeypatch, acknowledged=True, parallel=parallel)
    # The posted review resumes without re-invoking the reviewer, and its
    # carrier's assessment and records come from round metadata, whatever the
    # record's phase: a null assessment plus the original degradation record.
    assert run.reinvoked == 0
    ((impact, records),) = run.resumed
    assert impact is None
    assert [r.outcome for r in records] == ["degraded-to-undetermined"]
    assert run.repair_calls == []
    for record in run.records:
        assert record.architecture_impact is None
        assert [r.outcome for r in record.architecture_impact_degradations] == [
            "degraded-to-undetermined"
        ]


@pytest.mark.parametrize("parallel", [False, True], ids=["authoritative", "publication"])
def test_resumed_unacknowledged_pr_review_is_reinvoked_not_repaired_from_a_stale_source(
    tmp_path, monkeypatch, parallel
):
    # In the PR flow a same-head review resumes only when its record already
    # carries the acknowledgement (_resumed_pr_reviewer_matches_requirements),
    # so an unacknowledged posted review is re-run instead of resumed. The
    # acknowledgement repair then sees the fresh accepted carrier, which keeps
    # the null assessment and its record.
    run = _ack_pr_resume_run(tmp_path, monkeypatch, acknowledged=False, parallel=parallel)
    assert run.reinvoked == 1
    assert run.resumed == []
    assert run.carriers
    for carrier in run.carriers:
        assert carrier.architecture_impact is None
        assert [r.outcome for r in carrier.architecture_impact_degradations] == [
            "degraded-to-undetermined"
        ]


@pytest.mark.parametrize("state_report", [None, "measured"])
def test_m1067_run_pr_loop_prints_fallback_state_report(tmp_path, monkeypatch, capsys, state_report):
    # #1067 review item-8: an ordinary fallback with no recovery capability
    # stops before agents and prints the measured report when one exists.
    report = (
        "Before this run PR #77 was ready/unlabeled; it is now draft/labeled. Do not ready "
        "or unlabel the PR."
    )
    activation = ManagedCiContract(
        activation_path="ordinary_fallback",
        ordinary_recovery=None,
        state_report=report if state_report else None,
    )
    runner = FakeRunner()
    config = make_config(tmp_path, auto_merge=True)
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *args, **kwargs: activation)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    out = capsys.readouterr().out
    if state_report:
        assert f"PR #77 was not merged because managed recovery provenance or an unlabeled CI route is unavailable. {report}" in out
        assert "remains draft and unmerged" not in out
    else:
        assert "PR #77 remains draft and unmerged because managed recovery provenance" in out
    commands = [command for command, _cwd in runner.commands]
    assert not any(command[:3] == ["gh", "pr", "ready"] for command in commands)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command in commands)
    assert not any("DELETE" in command or "POST" in command for command in commands)
    assert not any(command[:1] in (["codex"], ["claude"], ["gemini"]) for command in commands)


def test_owning_pr_loop_writes_run_window_on_normal_exit(tmp_path):
    """Reservation telemetry (#1107): the real run_pr_loop, normal return."""
    from coding_review_agent_loop.worker_telemetry import load_records, telemetry_log_path

    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        issue_payload={"number": 56, "title": "T", "body": "B\n\n-- Human Reviewer"},
        issue_comments=[],
        pr_payload={"body": "Fixes #56"},
    )
    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, quiet=False)) == 0
    records = list(load_records(telemetry_log_path()))
    assert [r["record"] for r in records] == ["run-start", "run-end"]
    assert records[0]["pr_number"] == 77 and records[0]["run_id"] == records[1]["run_id"]
    assert runner.telemetry_attribution is None


def test_owning_pr_loop_writes_run_end_on_exception(tmp_path, monkeypatch):
    from coding_review_agent_loop.worker_telemetry import load_records, telemetry_log_path

    runner = _FakeRunner()

    def boom(*_a, **_k):
        raise AgentLoopError("pr stop")

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "get_pr_review_context", boom)
    with pytest.raises(AgentLoopError, match="pr stop"):
        pr_loop.run_pr_loop(runner, pr_number=77, config=make_config(tmp_path))
    records = list(load_records(telemetry_log_path()))
    assert [r["record"] for r in records] == ["run-start", "run-end"]
    assert records[0]["pr_number"] == 77 and records[0]["run_id"] == records[1]["run_id"]
    assert runner.telemetry_attribution is None


# --- Per-turn response directory cleanup (#1560) ----------------------------


@pytest.fixture
def turn_dirs(tmp_path, monkeypatch):
    """Record every per-turn response directory, under a private TMPDIR."""
    import tempfile as tempfile_module

    from coding_review_agent_loop import agent_permissions

    private = tmp_path / "private-tmp"
    private.mkdir()
    monkeypatch.setattr(tempfile_module, "tempdir", str(private))
    allocated = []
    real_allocate = agent_permissions.allocate_turn_directory

    def allocate(parent):
        path = real_allocate(parent)
        allocated.append(path)
        return path

    monkeypatch.setattr(agent_permissions, "allocate_turn_directory", allocate)
    return allocated


def _run_validated(runner, config, validate):
    from coding_review_agent_loop.validated_agent import _run_validated_agent

    return _run_validated_agent(
        runner, agent="codex", config=config, prompt="Review.",
        marker_description="VALID", validate=validate, role="reviewer",
    )


def _validate_exact(expected):
    def validate(text):
        if text != expected:
            raise AgentLoopError("invalid response")
        return text

    return validate


def test_validated_turn_removes_its_turn_directory_after_success(tmp_path, turn_dirs):
    config = make_config(tmp_path, agent_max_retries=0)
    seen = []
    runner = FakeRunner(codex_outputs=[{"public_response": "VALID", "stdout": "", "returncode": 0}])

    def validate(text):
        # The file is still readable while the turn validates it.
        seen.append([path.exists() for path in turn_dirs])
        return _validate_exact("VALID")(text)

    response = _run_validated(runner, config, validate)
    assert response.text == "VALID"
    assert seen and all(all(flags) for flags in seen)
    assert turn_dirs and all(not path.exists() for path in turn_dirs)
    # Only the per-turn directory is removed; its parent stays.
    assert all(path.parent.is_dir() for path in turn_dirs)


def test_validated_turn_removes_its_turn_directories_after_failure(tmp_path, turn_dirs, monkeypatch):
    import coding_review_agent_loop.agent_failure as agent_failure_module
    from coding_review_agent_loop.errors import AgentInvocationError

    diagnosed = []
    real_diagnostic = agent_failure_module._public_response_file_diagnostic

    def diagnostic(*, result, **kwargs):
        path = result.response_file_path if result is not None else None
        diagnosed.append(path is not None and path.parent.is_dir())
        return real_diagnostic(result=result, **kwargs)

    monkeypatch.setattr(agent_failure_module, "_public_response_file_diagnostic", diagnostic)
    config = make_config(tmp_path, agent_max_retries=1, agent_retry_backoff_seconds=(0, 0))
    runner = FakeRunner(codex_outputs=[
        {"public_response": "nonsense one", "stdout": "", "returncode": 0},
        {"public_response": "nonsense two", "stdout": "", "returncode": 0},
    ])
    with pytest.raises(AgentInvocationError) as info:
        _run_validated(runner, config, _validate_exact("VALID"))
    # Diagnostics read the response file before cleanup.
    assert "Reason: invalid response" in str(info.value)
    # Failure diagnostics ran while the turn directory still existed.
    assert diagnosed and all(diagnosed)
    assert len(turn_dirs) >= 1
    assert all(not path.exists() for path in turn_dirs)


def test_turn_directory_cleanup_failure_is_logged_without_changing_outcome(tmp_path, turn_dirs, monkeypatch):
    import coding_review_agent_loop.agents.base as base_module
    import coding_review_agent_loop.validated_agent as validated_agent_module

    config = make_config(tmp_path, agent_max_retries=0)
    messages = []
    monkeypatch.setattr(validated_agent_module, "log", lambda _config, message: messages.append(message))

    def failing_rmtree(path, *args, **kwargs):
        raise OSError("device busy")

    monkeypatch.setattr(base_module.shutil, "rmtree", failing_rmtree)
    runner = FakeRunner(codex_outputs=[{"public_response": "VALID", "stdout": "", "returncode": 0}])
    response = _run_validated(runner, config, _validate_exact("VALID"))
    assert response.text == "VALID"
    assert any("Could not remove per-turn response directory" in message for message in messages)
    assert all(path.exists() for path in turn_dirs)


def test_completion_recovery_terminal_write_precedes_turn_cleanup(tmp_path, turn_dirs, monkeypatch):
    from coding_review_agent_loop.errors import AgentInvocationError
    import coding_review_agent_loop.validated_agent as validated_agent_module

    written = []
    real_post = validated_agent_module._post_completion_recovery_terminal_comment

    def post(runner, *, recovery_result, terminal_text, **kwargs):
        real_post(runner, recovery_result=recovery_result, terminal_text=terminal_text, **kwargs)
        path = recovery_result.response_file_path
        written.append((path, path.read_text(encoding="utf-8") if path is not None else None))

    monkeypatch.setattr(validated_agent_module, "_post_completion_recovery_terminal_comment", post)
    runner = FakeRunner(claude_outputs=[
        json.dumps({"result": "I'll wait for the background test run to finish.", "session_id": "sess-1"}),
        ("", 1),
    ])
    config = make_config(tmp_path)
    with pytest.raises(AgentInvocationError) as info:
        run_issue_loop(runner, issue_number=56, config=config)
    terminal = info.value.terminal_public_response
    assert terminal is not None
    assert written and written[0][1] == terminal
    assert written[0][0].parent in turn_dirs
    assert all(not path.exists() for path in turn_dirs)


# Generalization (#1560 review item-3): every backend invocation goes through
# run_agent_result, which scopes its per-turn directory.  Inside a validated
# turn the outer scope defers cleanup past diagnostics; a direct invocation
# (post-authentication evidence correction, semantic dedupe) is cleaned up as
# soon as the backend has captured its response text.


@pytest.mark.parametrize("outcome", ["success", "nonzero-exit", "backend-raises"])
def test_direct_backend_invocation_removes_its_turn_directory(tmp_path, turn_dirs, monkeypatch, outcome):
    from coding_review_agent_loop.agents.codex import BACKEND as CODEX_BACKEND
    from coding_review_agent_loop.agents.registry import run_agent_result

    config = make_config(tmp_path, agent_max_retries=0)
    returncode = 0 if outcome == "success" else 1
    runner = FakeRunner(codex_outputs=[{"public_response": "DIRECT", "stdout": "", "returncode": returncode}])
    if outcome == "backend-raises":
        real_run = CODEX_BACKEND.run

        def raising_run(*args, **kwargs):
            real_run(*args, **kwargs)
            raise RuntimeError("provider crashed after allocating its response file")

        monkeypatch.setattr(CODEX_BACKEND, "run", raising_run)
        with pytest.raises(RuntimeError, match="provider crashed"):
            run_agent_result(runner, agent="codex", config=config, prompt="Correct.", role="coder")
    else:
        result = run_agent_result(runner, agent="codex", config=config, prompt="Correct.", role="coder")
        # The backend captured the text before the directory was removed.
        assert result.text == "DIRECT"
    assert turn_dirs and all(not path.exists() for path in turn_dirs)
    assert all(path.parent.is_dir() for path in turn_dirs)


def test_semantic_dedupe_transport_removes_its_turn_directory(tmp_path, turn_dirs):
    from coding_review_agent_loop.semantic_dedupe import default_semantic_transport

    config = make_config(tmp_path, semantic_followup_backend="codex", semantic_followup_model="m")
    runner = FakeRunner(codex_outputs=[{"public_response": '{"matches": []}', "stdout": "", "returncode": 0}])
    result = default_semantic_transport("Compare.", runner, config, 30.0)
    assert result.text == '{"matches": []}'
    assert turn_dirs and all(not path.exists() for path in turn_dirs)


def test_evidence_correction_turn_directory_is_removed_after_ingestion(tmp_path, turn_dirs):
    """The post-authentication correction coder runs outside any validated turn."""
    import coding_review_agent_loop.response_validation as response_validation_module
    from coding_review_agent_loop.agents.registry import run_agent_result

    # response_validation calls the shared, scoped entry point.
    assert response_validation_module.run_agent_result is run_agent_result
    config = make_config(tmp_path, agent_max_retries=0)
    runner = FakeRunner(codex_outputs=[{"public_response": "CORRECTED", "stdout": "", "returncode": 0}])
    result = response_validation_module.run_agent_result(
        runner, agent="codex", config=config, prompt="Correct.", role="coder",
        label="semantic-evidence-correction",
    )
    assert result.text == "CORRECTED"
    assert turn_dirs and all(not path.exists() for path in turn_dirs)


def test_direct_completion_recovery_writes_terminal_text_before_cleanup(tmp_path, turn_dirs, monkeypatch):
    """Called outside a validated turn, recovery still writes before its own scope cleans up."""
    import coding_review_agent_loop.validated_agent as validated_agent_module

    written = []
    real_post = validated_agent_module._post_completion_recovery_terminal_comment

    def post(runner, *, recovery_result, terminal_text, **kwargs):
        real_post(runner, recovery_result=recovery_result, terminal_text=terminal_text, **kwargs)
        written.append(recovery_result.response_file_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(validated_agent_module, "_post_completion_recovery_terminal_comment", post)
    runner = FakeRunner(claude_outputs=[("", 1)])
    config = make_config(tmp_path, coder="claude")
    outcome = validated_agent_module._attempt_claude_completion_recovery(
        runner, config=config,
        completion_recovery=validated_agent_module.CompletionRecoveryPolicy(issue_number=56),
        session_id="sess-1", validate=lambda text: text,
        usage_context=validated_agent_module._new_usage_context(config),
        run_id="run-1", role=None, label=None, timeout_seconds=None,
    )
    assert outcome.terminal_public_response is not None
    assert written == [outcome.terminal_public_response]
    assert turn_dirs and all(not path.exists() for path in turn_dirs)


# Generalization (#1560 review item-4): every recursive deletion of a directory
# an agent ran in goes through agent_permissions.remove_checkout_tree, which
# releases a hidden Gemini response root first.  The semantic-dedupe isolated
# workdir is one such directory.


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root bypasses directory read permission")
@pytest.mark.parametrize("backend", ["gemini", "codex"])
@pytest.mark.parametrize("outcome", ["success", "nonzero-exit", "backend-raises"])
def test_semantic_dedupe_transport_removes_its_whole_isolated_workdir(
    tmp_path, turn_dirs, monkeypatch, backend, outcome
):
    import coding_review_agent_loop.semantic_dedupe as semantic_dedupe_module
    from coding_review_agent_loop.agents.registry import get_backend

    isolated_dirs = []
    real_isolated = semantic_dedupe_module._isolated_provider_config

    def isolated(*args, **kwargs):
        config, directory = real_isolated(*args, **kwargs)
        isolated_dirs.append(directory)
        return config, directory

    monkeypatch.setattr(semantic_dedupe_module, "_isolated_provider_config", isolated)
    returncode = 0 if outcome == "success" else 1
    output = '{"matches": []}'
    runner = FakeRunner(
        gemini_outputs=[(output, returncode)],
        codex_outputs=[{"public_response": output, "stdout": "", "returncode": returncode}],
    )
    hidden_while_running = []
    backend_impl = get_backend(backend)
    real_run = backend_impl.run

    def run(*args, **kwargs):
        result = real_run(*args, **kwargs)
        # The response root inside the isolated workdir is non-listable.
        component = result.response_file_path.parent.parent
        hidden_while_running.append(
            (str(component).startswith(str(kwargs.get("config", args[1]).gemini_dir)), oct(os.lstat(component).st_mode & 0o777))
        )
        if outcome == "backend-raises":
            raise RuntimeError("provider crashed")
        return result

    monkeypatch.setattr(backend_impl, "run", run)
    config = make_config(tmp_path, semantic_followup_backend=backend, semantic_followup_model="m")
    if outcome == "backend-raises":
        with pytest.raises(RuntimeError, match="provider crashed"):
            semantic_dedupe_module.default_semantic_transport("Compare.", runner, config, 30.0)
    else:
        semantic_dedupe_module.default_semantic_transport("Compare.", runner, config, 30.0)
    if backend == "gemini":
        # Gemini's response root lives inside the isolated workdir and is hidden.
        assert hidden_while_running == [(True, oct(0o300))]
    assert hidden_while_running
    assert isolated_dirs and all(not os.path.lexists(path) for path in isolated_dirs)
    assert turn_dirs and all(not path.exists() for path in turn_dirs)
