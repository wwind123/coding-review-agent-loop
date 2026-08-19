import datetime
import json
import re
from unittest.mock import patch

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.ci_health import CiInfrastructureStall, StalledCheck
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop, run_pr_loop
from coding_review_agent_loop.comment_rendering import (
    _render_public_coder_followup_comment,
    _render_public_pr_review_comment,
)
from coding_review_agent_loop.errors import QuotaResetExceededError
from coding_review_agent_loop.followups import MAX_APPROVED_FOLLOWUP_ISSUES, reconcile_approved_followups
from coding_review_agent_loop.github import (
    CiWaitOutcome,
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
from coding_review_agent_loop.migrations import MigrationValidationResult
from coding_review_agent_loop.managed_ci import ManagedCiContract, ManagedCiOutcome
from coding_review_agent_loop.orchestrator import (
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    PostedRoundMetadata,
    ValidatedAgentResponse,
    _attach_round_metadata,
    _decode_round_metadata,
    _reconcile_human_requirements_ack_item,
    _resume_pr_round,
    _strip_round_metadata,
)
from coding_review_agent_loop.prompts import (
    COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
)
from coding_review_agent_loop.protocol import (
    ApprovedFollowup,
    ReviewItemDisposition,
    UnresolvedReviewItem,
    parse_plan_review,
    parse_pr_review,
    parse_review,
    parse_unresolved_item_dispositions,
    validate_structured_coder_followup,
)
from agent_loop_helpers import (
    FakeRunner,
    blocking_issues,
    command_index,
    make_config,
    prior_item_dispositions,
    structured_coder_followup,
    structured_plan_review,
    structured_pr_review,
)


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
        pr_payload={"body": "Fixes #99"},
    )
    config = make_config(tmp_path)
    supplied_context = IssueContext(
        number=56, repo="OWNER/REPO", title="Caller issue", body="Caller body", url=None, comments=()
    )

    assert run_pr_loop(runner, pr_number=77, config=config, issue_context=supplied_context) == 0

    assert not _issue_view_commands(runner)
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Caller issue" in prompt


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

def test_pr_loop_does_not_retry_normal_missing_marker_response(tmp_path):
    output = "I reviewed the PR and it looks fine."
    runner = FakeRunner(gemini_outputs=[output])
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError, match="AGENT_STATE"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert runner.comments == []
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)

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

def test_reconcile_human_requirements_ack_item_surfaces_markdown_ack_blocker():
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

    assert [item.item_id for item in reconciled] == [HUMAN_REQUIREMENTS_ACK_ITEM_ID]
    assert "missing required signed human requirements marker" in reconciled[0].text

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
            "- Requirement 1: updated the URL handling.\n"
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
    config = make_config(tmp_path, max_rounds=2)
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
    failing=(),
    pending=(),
    missing_required=(),
    errors=(),
):
    return PullRequestChecks(
        state=state,
        required_checks=("test",),
        passing=(),
        pending=tuple(pending),
        failing=tuple(failing),
        missing_required=tuple(missing_required),
        branch_protection_status="configured",
        check_query_status="partial" if errors else "ok",
        check_query_errors=tuple(errors),
    )


@pytest.mark.parametrize("auto_merge", [False, True])
def test_watch_mode_success_skips_legacy_wait_for_ci(
    tmp_path, monkeypatch, auto_merge
):
    runner = FakeRunner(codex_outputs=[structured_pr_review(state="approved", summary="Approved.")])
    config = make_config(tmp_path, watch_pending_ci=True, auto_merge=auto_merge)
    monkeypatch.setattr(
        orchestrator,
        "watch_pr_checks",
        lambda *args, **kwargs: CiWatchOutcome(status="passed", attempts_used=1),
    )
    monkeypatch.setattr(
        orchestrator,
        "wait_for_ci",
        lambda *args, **kwargs: pytest.fail("legacy CI waiter must not run in watch mode"),
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert any("watching GitHub checks" in comment for comment in runner.comments)
    merge_commands = [
        cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "pr", "merge"]
    ]
    assert bool(merge_commands) is auto_merge


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
            CiWatchOutcome(status="passed", attempts_used=1),
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
        return CiWatchOutcome(status="passed", head_sha="new-head", attempts_used=1)

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
            structured_pr_review(state="approved", summary="Round three approved."),
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
        return CiWatchOutcome(status="passed", head_sha="newer-head", attempts_used=1)

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
        return CiWatchOutcome(status="passed", head_sha="newer-head", attempts_used=1)

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
def test_disabled_watch_mode_preserves_legacy_pending_and_auto_merge_paths(
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
    monkeypatch.setattr(
        orchestrator,
        "watch_pr_checks",
        lambda *args, **kwargs: pytest.fail("disabled watch mode must not invoke watcher"),
    )
    wait_calls = []

    def legacy_wait(*args, **kwargs):
        wait_calls.append((args, kwargs))
        return CiWaitOutcome(status="passed")

    monkeypatch.setattr(orchestrator, "wait_for_ci", legacy_wait)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert bool(wait_calls) is auto_merge
    merge_commands = [
        cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "pr", "merge"]
    ]
    assert bool(merge_commands) is auto_merge
    if not auto_merge:
        assert any("checks are still pending" in comment for comment in runner.comments)


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
                addressed_items=["item-2"],
            )
        ],
    )
    config = make_config(
        tmp_path,
        watch_pending_ci=True,
        max_rounds=1,
        approved_followups="summarize",
    )
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
        return CiWatchOutcome(status="passed", attempts_used=1)

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
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=2)
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
                summary="Model-default consistency gaps are fixed.",
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

def test_pr_loop_downgrades_pending_ci_only_blocking_review_with_auto_merge(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Model-default consistency gaps are fixed.",
                blocking_items=["GitHub check `test` is still pending/in_progress."],
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

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)
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
        "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
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
    assert "Still blocked." not in followup_prompt

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
    assert len(runner.comments) == 2
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


def test_pr_loop_rereviews_unchanged_head_when_new_human_requirement_is_surfaced(
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
        created_at="2026-05-18T10:10:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-2",
        body="Also preserve the reviewer attribution.",
    )
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
    contexts = iter(
        [
            PullRequestReviewContext(
                metadata=metadata,
                comments=(),
                human_requirements=(requirement_1,),
            ),
            PullRequestReviewContext(
                metadata=metadata,
                comments=(),
                human_requirements=(requirement_1, requirement_2),
            ),
        ]
    )
    monkeypatch.setattr(
        "coding_review_agent_loop.orchestrator.get_pr_review_context",
        lambda *args, **kwargs: next(contexts),
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
    assert "Requirement 2" in codex_reviews[1][-1]


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
        "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
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
    assert "Codex approves first pass." not in followup_prompt
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
        "-- OpenAI Codex"
    )
    assert runner.comments[2] == (
        "**Review verdict:** Approved\n\n"
        "Looks good.\n\n"
        "### Prior unresolved item dispositions\n"
        "- [item-1] Same-PR follow-up from OpenAI Codex, round 1: Require source issue reference in PR body. -> resolved\n"
        "<!-- AGENT_STATE: approved -->\n"
        "-- OpenAI Codex"
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
    assert "Needs one more regression test before merge." not in second_coder_prompt
    assert runner.comments[0] == (
        "**Review verdict:** Blocking\n\n"
        "Needs one more regression test before merge.\n\n"
        "### Blocking issues\n"
        "- Add the mixed-history resume case to `tests/test_agent_loop.py`.\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- OpenAI Codex"
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
    assert "The PR is docs-only" not in followup_prompt
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
    assert "Two separate problems found." not in followup_prompt

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
                        "requirement_id": "Requirement 1",
                        "disposition": "addressed",
                        "evidence": "The URL fix is implemented.",
                    }
                ],
                "human_requirements": {
                    "addressed_ids": ["Requirement 1"],
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
            + prior_item_dispositions("[item-1] same-pr")
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
    assert runner.comments == ["**Review verdict:** Approved\n\nLGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"]
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
    pr_coder_text = (
        "Implemented the feature.\n"
        "<!-- AGENT_PR: 77 -->\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- Anthropic Claude"
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
                marker_value=77,
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


def test_pr_loop_escalates_to_human_when_reviewer_rejects_dispute(tmp_path):
    """Coder disputes a blocking item; reviewer still blocks after seeing evidence → escalate."""
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
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)

    with pytest.raises(
        AgentLoopError,
        match="Reviewer did not resolve 1 disputed item",
    ) as excinfo:
        run_pr_loop(runner, pr_number=55, config=config)

    assert "Update/evidence: Codex: I checked and the pricing is still wrong." in str(excinfo.value)


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
