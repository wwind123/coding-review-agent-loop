import base64
import datetime
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from coding_review_agent_loop.agents.base import with_public_response_file_instruction
from coding_review_agent_loop.agents.gemini import PUBLIC_RESPONSE_MARKER
from coding_review_agent_loop.cli import (
    AgentLoopError,
    Runner,
    build_parser,
    config_from_args,
    ensure_log_dir_ignored,
    is_clarification_request,
    run_issue_loop,
    run_pr_loop,
    run_task_loop,
)
from coding_review_agent_loop.comment_rendering import normalize_freeform_signature
from coding_review_agent_loop.config import (
    default_agent_memory_dir,
    default_agent_workdir,
    default_cache_root,
    resolve_base_branch,
)
from coding_review_agent_loop.errors import QuotaResetExceededError
from coding_review_agent_loop.github import IssueComment
from coding_review_agent_loop.orchestrator import (
    PostedRoundMetadata,
    ValidatedAgentResponse,
    _attach_round_metadata,
    _decode_round_metadata,
    _encode_round_metadata,
    _failure_category,
    _format_reset_duration,
    _is_transient_agent_output,
    _is_transient_public_response,
    _parse_rate_limit_reset_seconds,
    _plan_subject,
    _resume_plan_round,
    _run_validated_agent,
    _strip_round_metadata,
    _validate_coder_followup_response,
    _validate_plan_review_response,
    _validate_review_response,
    render_canonical_plan_revision,
    render_public_agent_comment,
)
from coding_review_agent_loop.protocol import (
    UnresolvedReviewItem,
    parse_plan_item_dispositions,
    parse_plan_review,
    parse_pr_review,
    validate_structured_coder_followup,
    validate_structured_plan_revision,
)
from agent_loop_helpers import (
    FakeRunner,
    command_index,
    make_config,
    prior_item_dispositions,
    prior_plan_item_dispositions,
    structured_coder_followup,
    structured_plan_review,
    structured_plan_revision,
    structured_pr_review,
)


def test_pre_review_tests_can_be_disabled(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\nTests: pytest passed.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(
        tmp_path,
        test_command=("pytest", "tests/test_agent_loop.py"),
        pre_review_tests=False,
    )

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    first_test = commands.index(["pytest", "tests/test_agent_loop.py"])
    first_review = command_index(runner.commands, ["codex", "exec"])
    assert first_review < first_test
    assert commands.count(["pytest", "tests/test_agent_loop.py"]) == 1


def test_ensure_log_dir_ignored_does_not_overwrite_existing_file(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    gitignore = log_dir / ".gitignore"
    gitignore.write_text("custom\n", encoding="utf-8")

    ensure_log_dir_ignored(log_dir)

    assert gitignore.read_text(encoding="utf-8") == "custom\n"


@pytest.mark.parametrize(
    "text",
    [
        "orchestrator.py lines 577-581: it currently falls back to parse_plan_state(text)",
        "orchestrator.py:577-581: it currently falls back to parse_plan_state(text)",
        "A bare 500 in diagnostic prose without HTTP context.",
    ],
)
def test_source_line_references_with_5xx_numbers_are_not_transient(text):
    assert not _is_transient_agent_output(text)
    assert _failure_category(text) == "deterministic"


@pytest.mark.parametrize(
    "text",
    [
        "Internal Server Error",
        "Bad Gateway",
        "Service Unavailable",
        "Gateway Timeout",
    ],
)
def test_explicit_server_error_phrases_remain_transient(text):
    assert _is_transient_agent_output(text)
    assert _failure_category(text) == "transient"


@pytest.mark.parametrize(
    "text",
    [
        "The authoritative PR diff shows no regressions.",
        "Authoritative source confirms the change.",
        "The author of this commit fixed the bug.",
        "This is an authoritative reference.",
    ],
)
def test_auth_prefix_words_are_not_non_retryable(text):
    """Words starting with 'auth' that are not auth-failure keywords must not match."""
    from coding_review_agent_loop.transient import NON_RETRYABLE_AGENT_OUTPUT_RE

    assert not NON_RETRYABLE_AGENT_OUTPUT_RE.search(text), (
        f"NON_RETRYABLE_AGENT_OUTPUT_RE unexpectedly matched: {text!r}"
    )
    assert _is_transient_agent_output(text) is False or True  # no crash; classification is unrelated
    assert _failure_category(text) != "non-retryable"


@pytest.mark.parametrize(
    "text",
    [
        "authentication failed",
        "Authorization error",
        "auth failed",
        "unauthorized",
        "forbidden",
        "Invalid API Key",
        "billing issue",
        "credit limit exceeded",
        "dirty checkout",
    ],
)
def test_genuine_auth_and_billing_terms_remain_non_retryable(text):
    """Real auth/billing/dirty-checkout diagnostics must still be non-retryable."""
    from coding_review_agent_loop.transient import NON_RETRYABLE_AGENT_OUTPUT_RE

    assert NON_RETRYABLE_AGENT_OUTPUT_RE.search(text), (
        f"NON_RETRYABLE_AGENT_OUTPUT_RE did not match: {text!r}"
    )
    assert _failure_category(text) == "non-retryable"


def test_plan_review_does_not_post_diagnostics_without_plan_state(tmp_path):
    diagnostic = "[ERROR] Invalid stream: The model returned an empty response or malformed tool call."
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Update the CLI.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        gemini_outputs=[diagnostic, diagnostic, diagnostic],
    )
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError, match="AGENT_PLAN_STATE"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert len(runner.comments) == 1
    assert runner.comments[0].startswith("Plan:")
    assert not any(diagnostic in comment for comment in runner.comments)


def test_plan_loop_retries_plain_agent_plan_state_near_miss_once(tmp_path):
    near_miss = "Plan looks sound.\nAGENT_PLAN_STATE: approved.\n-- Google Gemini"
    valid = "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Update the CLI.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        gemini_outputs=[near_miss, valid],
    )
    config = make_config(tmp_path, reviewer="gemini")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert near_miss not in runner.comments
    assert any(comment == f"**Review verdict:** Approved\n\n{valid}" for comment in runner.comments)
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert sleep_commands == [["sleep", "1"]]


def test_gemini_public_response_file_is_inside_git_dir(tmp_path):
    valid = "LGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=["stdout should be ignored"], public_response_outputs=[valid])
    config = make_config(tmp_path, reviewer="gemini")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    gemini_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"]]
    assert len(gemini_commands) == 1
    prompt = "\n".join(gemini_commands[0])
    expected_prefix = str(config.gemini_dir / ".git" / "agent-loop" / "responses" / "gemini")
    assert expected_prefix in prompt
    assert "/tmp/coding-review-agent-loop/responses/" not in prompt


def test_gemini_public_response_file_resolves_worktree_git_dir(tmp_path):
    valid = "LGTM from response file.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(gemini_outputs=["stdout should be ignored"], public_response_outputs=[valid])
    config = make_config(tmp_path, reviewer="gemini")
    git_dir = tmp_path / "main-repo" / ".git" / "worktrees" / "gemini"
    git_dir.mkdir(parents=True)
    (config.gemini_dir / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments == [f"**Review verdict:** Approved\n\n{valid}"]
    gemini_call = next(cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert str(git_dir / "agent-loop" / "responses" / "gemini") in gemini_call[2]
    assert str(config.gemini_dir / ".git" / "agent-loop") not in gemini_call[2]


def test_gemini_pre_marker_429_does_not_suppress_structured_review_repair(tmp_path):
    malformed_public_review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Found one issue.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\nProse between JSON and footer should be repaired.\n"
        "<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    raw_stdout = (
        "Attempt 1 failed with status 429. Retrying with backoff... "
        "No capacity available for model gemini-3-flash-preview on the server.\n"
        f"{PUBLIC_RESPONSE_MARKER}\n"
        f"{malformed_public_review}"
    )
    repaired_review = structured_pr_review(
        state="approved",
        summary="Review passed after repair.",
        reviewer="Google Gemini",
    )
    runner = FakeRunner(gemini_outputs=[{"stdout": raw_stdout}])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)
    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append(raw)
        assert expected_kind == "pr_review"
        return repaired_review

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert captured_repairs == [malformed_public_review]
    assert "429" not in captured_repairs[0]
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)
    assert any("Review passed after repair." in comment for comment in runner.comments)


def test_gemini_response_file_repair_ignores_raw_stdout_transient_diagnostics(tmp_path):
    malformed_public_review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Found one issue.",
                "blocking_items": ["Approved reviews cannot have blocking items."],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    repaired_review = structured_pr_review(
        state="approved",
        summary="Response file review passed after repair.",
        reviewer="Google Gemini",
    )
    runner = FakeRunner(
        gemini_outputs=[
            {"stdout": "Attempt 1 failed with status 429. No capacity available, then recovered."}
        ],
        public_response_outputs=[{"text": malformed_public_review}],
    )
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)
    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append(raw)
        return repaired_review

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert captured_repairs == [malformed_public_review]
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)
    assert any("Response file review passed after repair." in comment for comment in runner.comments)


def test_diagnostic_shaped_public_response_remains_transient(tmp_path):
    public_response = (
        f"{PUBLIC_RESPONSE_MARKER}\n"
        "HTTP 429 Too Many Requests: rate limit exceeded.\n"
        "<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    runner = FakeRunner(gemini_outputs=[{"stdout": public_response}])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        with pytest.raises(AgentLoopError) as exc_info:
            run_pr_loop(runner, pr_number=77, config=config)

    repair_mock.assert_not_called()
    assert "Failure category: transient" in str(exc_info.value)


def test_public_response_error_payload_remains_transient():
    assert _is_transient_public_response(
        json.dumps(
            {
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota exceeded. Retry-After: 60",
                }
            }
        )
    )


def test_public_response_structured_json_after_known_artifact_is_not_transient():
    text = (
        f"{PUBLIC_RESPONSE_MARKER}\n"
        + structured_pr_review(
            summary="Wrong structured kind discusses 429, quota, capacity, and transient behavior.",
            reviewer="Google Gemini",
        )
    )

    assert not _is_transient_public_response(text, repair_expected_kind="coder_followup")


def test_structured_plan_review_transient_terms_with_trailing_prose_normalizes(tmp_path):
    malformed_review = (
        structured_plan_review(
            state="approved",
            summary=(
                "The plan discusses 429, quota, resource exhausted, timeout, capacity, "
                "and transient retry handling as domain text."
            ),
            reviewer="Google Gemini",
        )
        + "\nTrailing prose after the signature should be repaired."
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the plan.",
            marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=lambda text: _validate_plan_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
            ),
            use_repair=True,
            repair_expected_kind="plan_review",
        )

    assert response.text == structured_plan_review(
        state="approved",
        summary=(
            "The plan discusses 429, quota, resource exhausted, timeout, capacity, "
            "and transient retry handling as domain text."
        ),
        reviewer="Google Gemini",
    )
    repair_mock.assert_not_called()
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_structured_pr_review_transient_terms_duplicate_footer_normalizes(tmp_path):
    malformed_review = (
        structured_pr_review(
            state="approved",
            summary=(
                "The review covers capacity, timeout, 429, quota, resource-exhausted, "
                "and transient classifier behavior."
            ),
            reviewer="Google Gemini",
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )
    runner = FakeRunner(gemini_outputs=[malformed_review])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="gemini",
            config=config,
            prompt="Review the PR.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_review_response(
                text,
                reviewer="Google Gemini",
                unresolved_items=(),
            ),
            use_repair=True,
            repair_expected_kind="pr_review",
        )

    assert response.text == structured_pr_review(
        state="approved",
        summary=(
            "The review covers capacity, timeout, 429, quota, resource-exhausted, "
            "and transient classifier behavior."
        ),
        reviewer="Google Gemini",
    )
    repair_mock.assert_not_called()
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_structured_coder_followup_transient_terms_before_footer_runs_repair(tmp_path):
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Add timeout regression coverage.",
            status="blocking",
        ),
    )
    malformed_followup = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "approved",
                "summary": "Updated timeout and capacity handling without treating prose as transient.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n## Changes made\nMentioned timeout and capacity in prose before the footer.\n"
        "<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"
    )
    repaired_followup = structured_coder_followup(
        state="approved",
        summary="Updated timeout and capacity handling.",
        addressed_items=["item-1"],
        remaining_items=[],
        reviewer="Anthropic Claude",
    )
    runner = FakeRunner(claude_outputs=[malformed_followup])
    config = make_config(tmp_path, coder="claude", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired_followup) as repair_mock:
        response = _run_validated_agent(
            runner,
            agent="claude",
            config=config,
            prompt="Address review feedback.",
            marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=lambda text: _validate_coder_followup_response(
                text,
                unresolved_items=unresolved_items,
                human_requirements=(),
            ),
            use_repair=True,
            repair_expected_kind="coder_followup",
        )

    assert response.text == repaired_followup
    repair_mock.assert_called_once_with(
        malformed_followup,
        config.gemini_cmd,
        expected_kind="coder_followup",
    )
    assert not any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_run_validated_agent_recovers_coder_followup_from_message_text_when_response_file_markdown(tmp_path):
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-8",
            reviewer="Orchestrator",
            source_round=4,
            text="Acknowledge signed human requirements.",
            status="blocking",
        ),
    )
    valid_followup = structured_coder_followup(
        state="approved",
        summary="Acknowledged the signed human requirements.",
        addressed_items=["item-8"],
        remaining_items=[],
        reviewer="OpenAI Codex",
    )
    markdown_response_file = (
        "### Human requirements\n\n"
        "Acknowledged.\n\n"
        "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
        "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )
    runner = FakeRunner(
        codex_outputs=[{"public_response": valid_followup, "stdout": "diagnostic output"}],
        public_response_outputs=[{"text": markdown_response_file}],
    )
    config = make_config(tmp_path, coder="codex", agent_max_retries=0)

    response = _run_validated_agent(
        runner,
        agent="codex",
        config=config,
        prompt="Address feedback.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=lambda text: _validate_coder_followup_response(
            text,
            unresolved_items=unresolved_items,
            human_requirements=(),
        ),
        repair_expected_kind="coder_followup",
    )

    assert response.text == valid_followup
    assert response.marker_value.addressed_items == ("item-8",)


def test_run_validated_agent_recovers_fenced_coder_followup_from_raw_stdout(tmp_path):
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Fix the bug.",
            status="blocking",
        ),
    )
    valid_followup = structured_coder_followup(
        state="approved",
        addressed_items=["item-1"],
        remaining_items=[],
        reviewer="OpenAI Codex",
    )
    json_part, footer = valid_followup.split("\n<!-- AGENT_STATE:", 1)
    fenced_stdout = f"tool diagnostic\n```json\n{json_part}\n```\n<!-- AGENT_STATE:{footer}"
    runner = FakeRunner(
        codex_outputs=[{"public_response": "legacy markdown", "stdout": fenced_stdout}],
        public_response_outputs=[{"text": "### Update\nFixed it.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"}],
    )
    config = make_config(tmp_path, coder="codex", agent_max_retries=0)

    response = _run_validated_agent(
        runner,
        agent="codex",
        config=config,
        prompt="Address feedback.",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=lambda text: _validate_coder_followup_response(
            text,
            unresolved_items=unresolved_items,
            human_requirements=(),
        ),
        repair_expected_kind="coder_followup",
    )

    assert response.text == valid_followup


# ---------------------------------------------------------------------------
# Issue #271: coder_followup path through attempt_envelope_normalization
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Issue #275: strip_unknown_prior_item_dispositions with tightly-packed input
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Issue #274: combined envelope+disposition strip via _run_validated_agent
# ---------------------------------------------------------------------------


def test_gemini_duplicate_trailing_agent_state_marker_normalizes_without_repair(tmp_path):
    malformed_public_review = (
        structured_pr_review(
            state="approved",
            summary="Found one issue.",
            reviewer="Google Gemini",
        )
        + "\n\n<!-- AGENT_STATE: approved -->"
    )
    raw_stdout = f"{PUBLIC_RESPONSE_MARKER}\n{malformed_public_review}"
    runner = FakeRunner(gemini_outputs=[{"stdout": raw_stdout}])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    repair_mock.assert_not_called()
    assert any("Found one issue." in comment for comment in runner.comments)
    assert all(comment.count("<!-- AGENT_STATE: approved -->") == 1 for comment in runner.comments)


def test_gemini_pre_marker_429_malformed_public_response_fails_deterministically(tmp_path):
    malformed_public_review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Found one issue.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\nExtra prose before the footer.\n"
        "<!-- AGENT_STATE: approved -->\n-- Google Gemini"
    )
    raw_stdout = (
        "Attempt 1 failed with status 429. No capacity available for model gemini.\n"
        f"{PUBLIC_RESPONSE_MARKER}\n"
        f"{malformed_public_review}"
    )
    runner = FakeRunner(gemini_outputs=[{"stdout": raw_stdout}])
    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value="still invalid"):
        with pytest.raises(AgentLoopError) as exc_info:
            run_pr_loop(runner, pr_number=77, config=config)

    message = str(exc_info.value)
    assert "Failure category: deterministic" in message
    assert "Failure category: transient" not in message


@pytest.mark.parametrize("text,expected_secs", [
    ("Retry-After: 3600", 3600),
    ("retry after 1800", 1800),
    ("retryDelay: '7200s'", 7200),
    ("try again in 2h 30m", 9000),
    ("try again in 45m", 2700),
    ("resets in 1h", 3600),
    ("reset in 5m", 300),
])
def test_parse_rate_limit_reset_seconds(text, expected_secs):
    assert _parse_rate_limit_reset_seconds(text) == expected_secs


def test_parse_rate_limit_reset_seconds_claude_absolute_time():
    now = datetime.datetime(2026, 6, 3, 5, 33, 48, tzinfo=datetime.timezone.utc)
    text = "You've hit your session limit · resets 1:30am (America/Los_Angeles)"

    assert _parse_rate_limit_reset_seconds(text, now_utc=now) == 10572


@pytest.mark.parametrize("text", [
    "HTTP 429: rate limit exceeded.",
    "Too many requests.",
    "quota exceeded",
])
def test_parse_rate_limit_reset_seconds_returns_none_when_unparseable(text):
    assert _parse_rate_limit_reset_seconds(text) is None


@pytest.mark.parametrize("seconds,expected", [
    (3600, "1h"),
    (7200, "2h"),
    (9000, "2h 30m"),
    (300, "5m"),
    (45, "45s"),
    (3660, "1h 1m"),
])
def test_format_reset_duration(seconds, expected):
    assert _format_reset_duration(seconds) == expected


def test_quota_reset_exceeded_error_exit_code():
    assert QuotaResetExceededError.EXIT_CODE == 3


def test_agent_memory_default_parent_ignores_generated_contents(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    gitignore = tmp_path / "claude" / ".agent-loop" / ".gitignore"
    assert gitignore.read_text(encoding="utf-8") == "*\n!.gitignore\n"


def test_agent_memory_does_not_ignore_custom_parent_directory(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    memory_dir = tmp_path / "custom-memory"
    config = make_config(tmp_path, agent_memory_dir=memory_dir)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not (tmp_path / ".gitignore").exists()


def test_agent_memory_detects_changed_files_since_previous_commit(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_head="def456",
        changed_files=["src/coding_review_agent_loop/prompts.py", "tests/test_agent_loop.py"],
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "last-analyzed-commit").write_text("abc123\n", encoding="utf-8")
    config = make_config(tmp_path, agent_memory_dir=memory_dir)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    diff_commands = [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["git", "diff", "--name-only"]]
    assert ["git", "diff", "--name-only", "abc123..def456"] in diff_commands
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "src/coding_review_agent_loop/prompts.py" in prompt
    assert "tests/test_agent_loop.py" in prompt
    assert (memory_dir / "last-analyzed-commit").read_text(encoding="utf-8") == "def456\n"


def test_agent_memory_logs_when_changed_file_diff_falls_back(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_head="def456",
        diff_returncode=128,
        diff_stderr="fatal: bad revision 'abc123..def456'",
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "last-analyzed-commit").write_text("abc123\n", encoding="utf-8")
    config = make_config(tmp_path, agent_memory_dir=memory_dir, quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    captured = capsys.readouterr()
    assert "Could not diff agent memory baseline abc123..def456" in captured.err
    assert "treating all tracked files as changed" in captured.err
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "README.md" in prompt


def test_test_profile_records_provided_test_command(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    memory_dir = tmp_path / "memory"
    config = make_config(
        tmp_path,
        agent_memory_dir=memory_dir,
        test_command=("python", "-m", "pytest", "-q"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    profile = (memory_dir / "test-profile.md").read_text(encoding="utf-8")
    assert "`python -m pytest -q`" in profile
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "prefer verified test commands from the execution profile" in prompt


def test_agent_memory_can_be_disabled(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    memory_dir = tmp_path / "memory"
    config = make_config(tmp_path, agent_memory=False, agent_memory_dir=memory_dir)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not memory_dir.exists()
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Agent memory context:" not in prompt


def test_resume_plan_round_marks_empty_ledger_incomplete_after_same_subject_prior_new_items():
    plan = "Plan text."
    subject = _plan_subject(plan)
    prior_new_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Prior same-plan item.",
        status="blocking",
    )
    prior_review_comment = _attach_round_metadata(
        "Prior plan review.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=subject,
            prior_items=(),
            new_items=(prior_new_item,),
            state="blocking",
        ),
    )
    current_coder_comment = _attach_round_metadata(
        plan + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(),
            canonical_plan=plan,
        ),
    )

    resumed = _resume_plan_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=prior_review_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=current_coder_comment),
        ],
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed[1].ledger_may_be_incomplete is True


def test_shared_workdir_requires_explicit_override(tmp_path):
    runner = FakeRunner()
    shared = tmp_path / "repo"
    shared.mkdir()
    config = make_config(tmp_path, claude_dir=shared, codex_dir=shared)

    with pytest.raises(AgentLoopError, match="same directory"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_gemini_shared_workdir_requires_explicit_override(tmp_path):
    runner = FakeRunner()
    shared = tmp_path / "repo"
    shared.mkdir()
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        codex_dir=shared,
        gemini_dir=shared,
    )

    with pytest.raises(AgentLoopError, match="same directory"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_missing_agent_workdirs_are_created(tmp_path):
    runner = FakeRunner(
        claude_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"],
    )
    claude_dir = tmp_path / "missing" / "claude"
    codex_dir = tmp_path / "missing" / "codex"
    config = make_config(
        tmp_path,
        claude_dir=claude_dir,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="claude",
        create_dirs=False,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert claude_dir.is_dir()
    assert codex_dir.is_dir()


def test_missing_gemini_workdir_is_created_when_configured(tmp_path):
    runner = FakeRunner(
        gemini_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"],
    )
    gemini_dir = tmp_path / "missing" / "gemini"
    config = make_config(
        tmp_path,
        reviewer="gemini",
        gemini_dir=gemini_dir,
        create_dirs=False,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert gemini_dir.is_dir()


def test_non_codex_loop_uses_active_workdir_for_github_and_tests(tmp_path):
    runner = FakeRunner(
        gemini_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"],
    )
    codex_dir = tmp_path / "inactive" / "codex"
    config = make_config(
        tmp_path,
        claude_dir=tmp_path / "missing" / "claude",
        codex_dir=codex_dir,
        gemini_dir=tmp_path / "missing" / "gemini",
        coder="claude",
        reviewer="gemini",
        test_command=("pytest", "tests/test_agent_loop.py"),
        create_dirs=False,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not codex_dir.exists()
    github_or_test_cwds = [
        cwd
        for cmd, cwd in runner.commands
        if cmd[:1] == ["gh"] or cmd == ["pytest", "tests/test_agent_loop.py"]
    ]
    assert github_or_test_cwds
    bootstrap_pr_queries = [
        cwd
        for cmd, cwd in runner.commands
        if cmd[:3] == ["gh", "pr", "view"]
        and "number,title,headRefName,baseRefName,headRefOid,url,body,comments,reviews" in cmd
        and cwd != config.claude_dir
    ]
    assert bootstrap_pr_queries == [Path.cwd()]
    assert set(github_or_test_cwds) == {Path.cwd(), config.claude_dir}


def test_omitted_agent_dirs_default_to_repo_scoped_temp_checkouts(monkeypatch, tmp_path):
    parser = build_parser()
    cache_home = tmp_path / "cache"
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
    ])

    config = config_from_args(args, FakeRunner())

    assert config.codex_dir == default_agent_workdir("OWNER/REPO", "codex").resolve()
    assert config.claude_dir == default_agent_workdir("OWNER/REPO", "claude").resolve()
    assert config.gemini_dir == default_agent_workdir("OWNER/REPO", "gemini").resolve()
    assert config.antigravity_dir == default_agent_workdir("OWNER/REPO", "antigravity").resolve()
    assert set(config.auto_agent_dirs) == {"claude", "codex", "gemini", "antigravity"}
    assert config.agent_memory_dir == (
        cache_home / "coding-review-agent-loop" / "repos" / "OWNER-REPO" / "memory"
    ).resolve()


@pytest.mark.parametrize(
    ("coder", "reviewer", "missing_command", "override_flag"),
    [
        ("claude", "codex", "missing-claude", "--claude-cmd"),
        ("claude", "gemini", "missing-gemini", "--gemini-cmd"),
    ],
)
def test_config_preflight_rejects_missing_agent_before_repo_detection(
    monkeypatch,
    coder,
    reviewer,
    missing_command,
    override_flag,
):
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--coder",
        coder,
        "--reviewer",
        reviewer,
        f"--{coder}-cmd",
        missing_command if override_flag == f"--{coder}-cmd" else coder,
        f"--{reviewer}-cmd",
        missing_command if override_flag == f"--{reviewer}-cmd" else reviewer,
    ])
    detection_calls = []
    monkeypatch.setattr(
        "coding_review_agent_loop.config.detect_repo",
        lambda *call_args: detection_calls.append(call_args),
    )
    monkeypatch.setattr(
        "coding_review_agent_loop.config.shutil.which",
        lambda command: None if command == missing_command else f"/bin/{command}",
    )

    with pytest.raises(
        AgentLoopError,
        match=rf"{missing_command} CLI not found on PATH.*{override_flag}",
    ):
        config_from_args(args, Runner())

    assert detection_calls == []


def test_config_preflight_checks_only_unique_configured_agents(monkeypatch, tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "codex",
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    checked = []

    def fake_which(command):
        checked.append(command)
        return f"/bin/{command}"

    monkeypatch.setattr("coding_review_agent_loop.config.shutil.which", fake_which)

    config = config_from_args(args, Runner())

    assert config.coder == "codex"
    assert checked == ["codex", "agy"]


def test_config_preflight_accepts_custom_absolute_command(tmp_path):
    command = tmp_path / "custom-codex"
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    command.chmod(0o755)
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "codex",
        "--codex-cmd",
        str(command),
    ])

    config = config_from_args(args, Runner())

    assert config.codex_cmd == str(command)


def test_config_preflight_skips_dry_run_command_preview(monkeypatch):
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--dry-run",
        "--claude-cmd",
        "missing-claude",
        "--codex-cmd",
        "missing-codex",
    ])
    monkeypatch.setattr(
        "coding_review_agent_loop.config.shutil.which",
        lambda command: pytest.fail(f"unexpected preflight for {command}"),
    )

    config = config_from_args(args, Runner(dry_run=True))

    assert config.dry_run is True


def test_preflight_absolute_path_valid(tmp_path):
    """Absolute path to an existing executable passes preflight and is stored."""
    from coding_review_agent_loop.config import preflight_agent_commands

    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO",
        "--claude-cmd", sys.executable,
        "--codex-cmd", "codex",
    ])
    args.coder = "claude"
    runner = Runner()
    preflight_agent_commands(args, runner, ())
    assert runner._resolved_commands[sys.executable] == sys.executable


def test_preflight_absolute_path_not_found_gives_path_error(tmp_path):
    """Nonexistent absolute path gives 'not found or not executable', not 'not found on PATH'."""
    from coding_review_agent_loop.config import preflight_agent_commands

    nonexistent = str(tmp_path / "no-such-binary")
    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO",
        "--claude-cmd", nonexistent,
    ])
    args.coder = "claude"
    with pytest.raises(AgentLoopError, match="not found or not executable"):
        preflight_agent_commands(args, Runner(), ())


def test_preflight_absolute_path_dangling_symlink_gives_path_error(tmp_path):
    """Dangling absolute-path symlink gives 'not found or not executable', not 'not found on PATH'."""
    from coding_review_agent_loop.config import preflight_agent_commands

    dangling = tmp_path / "dangling-claude"
    dangling.symlink_to(tmp_path / "missing-target")
    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO",
        "--claude-cmd", str(dangling),
    ])
    args.coder = "claude"
    with pytest.raises(AgentLoopError, match="not found or not executable"):
        preflight_agent_commands(args, Runner(), ())


def test_preflight_bare_name_not_found_gives_path_message(monkeypatch):
    """Bare name not on PATH still gives 'not found on PATH' message."""
    from coding_review_agent_loop.config import preflight_agent_commands

    monkeypatch.setattr("coding_review_agent_loop.config.shutil.which", lambda cmd: None)
    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO",
        "--claude-cmd", "missing-bare-name",
    ])
    args.coder = "claude"
    with pytest.raises(AgentLoopError, match="not found on PATH"):
        preflight_agent_commands(args, Runner(), ())


def test_omitted_cli_base_is_preserved_for_runtime_resolution(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    assert config_from_args(args, FakeRunner()).base is None


def test_pre_review_tests_cli_defaults_on_and_can_be_disabled(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    config = config_from_args(args, FakeRunner())
    assert config.pre_review_tests is True

    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--no-pre-review-tests",
    ])
    config = config_from_args(args, FakeRunner())
    assert config.pre_review_tests is False


@pytest.mark.parametrize("repo", ["OWNER", "OWNER/", "/REPO", "OWNER/REPO/EXTRA"])
def test_default_agent_workdir_rejects_invalid_repo_formats(repo):
    with pytest.raises(AgentLoopError, match="OWNER/REPO"):
        default_agent_workdir(repo, "codex")


def test_default_agent_memory_dir_uses_xdg_cache_and_repo_scope(monkeypatch, tmp_path):
    cache_home = tmp_path / "xdg-cache"
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))

    assert default_agent_memory_dir("OWNER/REPO") == (
        cache_home / "coding-review-agent-loop" / "repos" / "OWNER-REPO" / "memory"
    )


def test_default_cache_root_uses_posix_home_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    assert default_cache_root() == tmp_path / ".cache" / "coding-review-agent-loop"


@pytest.mark.parametrize(
    ("platform", "home_parts"),
    [
        ("darwin", ("Library", "Caches", "coding-review-agent-loop")),
        ("win32", ("AppData", "Local", "coding-review-agent-loop", "Cache")),
    ],
)
def test_default_cache_root_uses_platform_home_fallbacks(
    monkeypatch,
    tmp_path,
    platform,
    home_parts,
):
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", platform)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    assert default_cache_root() == tmp_path.joinpath(*home_parts)


def test_default_cache_root_uses_windows_local_app_data(monkeypatch, tmp_path):
    local_app_data = tmp_path / "local-app-data"
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    assert default_cache_root() == local_app_data / "coding-review-agent-loop" / "Cache"


@pytest.mark.parametrize("repo", ["OWNER", "OWNER/", "/REPO", "OWNER/REPO/EXTRA"])
def test_default_agent_memory_dir_rejects_invalid_repo_formats(repo):
    with pytest.raises(AgentLoopError, match="OWNER/REPO"):
        default_agent_memory_dir(repo)


@pytest.mark.parametrize("mode", ["ignore", "summarize", "issue", "fix-and-summarize", "fix-and-issue"])
def test_approved_followups_cli_mode_is_configurable(tmp_path, mode):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--approved-followups",
        mode,
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.approved_followups == mode


@pytest.mark.parametrize(
    "mode",
    ["plan-only", "decompose-only", "implement-one-shot", "implement-by-phase"],
)
def test_plan_execution_mode_cli_is_configurable(tmp_path, mode):
    parser = build_parser()
    args = parser.parse_args([
        "issue",
        "56",
        "--repo",
        "OWNER/REPO",
        "--plan-first",
        "--plan-execution-mode",
        mode,
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.plan_execution_mode == mode


def test_explicit_agent_dirs_are_preserved_when_others_default(tmp_path):
    parser = build_parser()
    codex_dir = tmp_path / "codex"
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
        "--codex-dir",
        str(codex_dir),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.codex_dir == codex_dir
    assert config.claude_dir == default_agent_workdir("OWNER/REPO", "claude").resolve()
    assert set(config.auto_agent_dirs) == {"claude", "gemini", "antigravity"}


def test_relative_log_dir_defaults_under_active_coder_workdir(tmp_path):
    parser = build_parser()
    claude_dir = tmp_path / "claude"
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "claude",
        "--reviewer",
        "gemini",
        "--claude-dir",
        str(claude_dir),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.log_dir == claude_dir / ".agent-loop-logs"


def test_agent_memory_flags_configure_memory_dir_and_refresh(tmp_path):
    parser = build_parser()
    codex_dir = tmp_path / "codex"
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
        "--codex-dir",
        str(codex_dir),
        "--no-agent-memory",
        "--refresh-agent-memory",
        "--refresh-test-profile",
        "--agent-memory-dir",
        "custom-memory",
    ])

    config = config_from_args(args, FakeRunner())

    assert config.agent_memory is False
    assert config.refresh_agent_memory is True
    assert config.refresh_test_profile is True
    assert config.agent_memory_dir == codex_dir / "custom-memory"


def test_agent_memory_explicit_absolute_dir_is_resolved(tmp_path):
    parser = build_parser()
    memory_dir = tmp_path / "memory-parent" / ".." / "agent-memory"
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--agent-memory-dir",
        str(memory_dir),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.agent_memory_dir == memory_dir.resolve()


def test_agent_memory_default_ignores_active_coder_workdir(tmp_path, monkeypatch):
    parser = build_parser()
    cache_home = tmp_path / "cache"
    codex_dir = tmp_path / "codex"
    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
        "--codex-dir",
        str(codex_dir),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.agent_memory_dir == (
        cache_home / "coding-review-agent-loop" / "repos" / "OWNER-REPO" / "memory"
    ).resolve()
    assert codex_dir not in config.agent_memory_dir.parents


def test_auto_created_agent_dir_is_cloned_before_use(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "tmp-root" / "owner-repo" / "codex" / "repo"
    config = make_config(
        tmp_path,
        claude_dir=tmp_path / "explicit-claude",
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert ["gh", "repo", "clone", "OWNER/REPO", str(codex_dir)] in [
        cmd for cmd, _cwd in runner.commands
    ]
    assert codex_dir.is_dir()


def test_clean_existing_auto_agent_dir_is_synced(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "fetch", "origin"] in commands
    assert ["git", "switch", "main"] in commands
    assert ["git", "pull", "--ff-only", "origin", "main"] in commands


@pytest.mark.parametrize("mode", ["issue", "task"])
def test_issue_and_task_loops_use_repo_default_when_base_is_omitted(tmp_path, mode):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"baseRefName": "develop"},
        repo_default_branch="develop",
    )
    config = make_config(
        tmp_path,
        base=None,
        reviewer="codex",
        auto_agent_dirs=("claude", "codex"),
    )

    if mode == "issue":
        assert run_issue_loop(runner, issue_number=56, config=config) == 0
    else:
        assert run_task_loop(runner, task_text="Add /healthz endpoint.", config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "switch", "develop"] in commands
    assert not any("origin/main" in arg for cmd in commands for arg in cmd)


def test_unresolved_base_metadata_produces_targeted_override_error(tmp_path):
    runner = FakeRunner(
        pr_payload={"baseRefName": None},
        repo_default_branch=None,
        repo_default_branch_returncode=1,
    )
    config = make_config(tmp_path, base=None, reviewer="codex")

    with pytest.raises(
        AgentLoopError,
        match=r"Unable to resolve a base branch for OWNER/REPO.*--base <branch>",
    ):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["git", "switch"] for cmd, _cwd in runner.commands)


def test_dry_run_base_resolution_defaults_to_main_without_github_query(tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, base=None, dry_run=True)

    resolved = resolve_base_branch(config, runner)

    assert resolved.base == "main"
    assert not any(cmd[:1] == ["gh"] for cmd, _cwd in runner.commands)


def test_reviewer_checkout_is_refreshed_to_pr_head_before_review(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    review_index = command_index(runner.commands, ["codex", "exec"])
    fetch_index = command_index(runner.commands, ["git", "fetch", "origin"], start=0)
    pr_fetch_index = command_index(
        runner.commands,
        ["git", "fetch", "origin", "+pull/77/head:refs/remotes/origin/pr/77"],
    )
    checkout_index = command_index(
        runner.commands,
        ["git", "checkout", "--detach", "refs/remotes/origin/pr/77"],
    )
    head_index = command_index(runner.commands, ["git", "rev-parse", "HEAD"], start=checkout_index)

    assert commands[pr_fetch_index] == ["git", "fetch", "origin", "+pull/77/head:refs/remotes/origin/pr/77"]
    assert fetch_index < pr_fetch_index < checkout_index < head_index < review_index


def test_reviewer_checkout_refreshes_each_round_before_review(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Fixed.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "Please fix it.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    pr_fetches = [
        index
        for index, cmd in enumerate(commands)
        if cmd == ["git", "fetch", "origin", "+pull/77/head:refs/remotes/origin/pr/77"]
    ]
    review_indices = [index for index, cmd in enumerate(commands) if cmd[:2] == ["codex", "exec"]]

    assert len(pr_fetches) == 3
    assert len(review_indices) == 2
    assert pr_fetches[0] < review_indices[0]
    assert pr_fetches[1] < review_indices[1]


def test_dirty_default_reviewer_checkout_is_cleaned_before_review(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_status=" M stale.py\n",
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="claude",
        reviewer="codex",
        auto_agent_dirs=("claude", "codex"),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True) == 0

    reset_index = command_index(runner.commands, ["git", "reset", "--hard"])
    clean_index = command_index(runner.commands, ["git", "clean", "-fd"])
    review_index = command_index(runner.commands, ["codex", "exec"])

    assert reset_index < clean_index < review_index


def test_dirty_explicit_reviewer_checkout_fails_before_review_invocation(tmp_path):
    runner = FakeRunner(
        codex_outputs=["This should not run.\n<!-- AGENT_STATE: approved -->"],
        git_status=" M stale.py\n",
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", agent_memory=False)

    with pytest.raises(AgentLoopError, match="--codex-dir is dirty"):
        run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_dirty_existing_auto_agent_dir_is_cleaned_before_sync(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_status=" M file.py\n",
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
        quiet=False,
    )
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["git", "reset", "--hard"] in commands
    assert ["git", "clean", "-fd"] in commands
    assert ["git", "pull", "--ff-only", "origin", "main"] in commands
    captured = capsys.readouterr()
    assert f"Cleaning dirty default codex workdir: {codex_dir}" in captured.err


def test_dirty_explicit_agent_dir_fails_clearly(tmp_path):
    runner = FakeRunner(git_status=" M file.py\n")
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="codex",
        create_dirs=False,
    )
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="--codex-dir is dirty"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("loop_name", ["issue", "task"])
def test_dirty_explicit_coder_dir_fails_before_issue_or_task_coder_invocation(tmp_path, loop_name):
    runner = FakeRunner(
        git_status=" M file.py\n",
        codex_outputs=[
            "Implemented.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="codex",
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="--codex-dir is dirty"):
        if loop_name == "issue":
            run_issue_loop(runner, issue_number=56, config=config)
        else:
            run_task_loop(runner, task_text="Add /healthz endpoint.", config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_explicit_agent_dir_must_match_requested_repo(tmp_path):
    runner = FakeRunner(git_remote="git@github.com:OTHER/REPO.git")
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        coder="codex",
        reviewer="codex",
        create_dirs=False,
    )
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not 'OWNER/REPO'"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_existing_auto_agent_dir_must_be_git_checkout(tmp_path):
    runner = FakeRunner(git_inside=False)
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not a git checkout"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_stale_default_workdir_only_logs_is_recreated(tmp_path, capsys):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / ".agent-loop-logs").mkdir()

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
        quiet=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    clone_cmds = [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "repo", "clone"]]
    assert any(cmd[4] == str(codex_dir) for cmd in clone_cmds), "Expected fresh clone of stale workdir"

    captured = capsys.readouterr()
    assert "Stale default codex workdir detected" in captured.err
    assert "recreating" in captured.err


def test_stale_default_workdir_with_unknown_files_still_fails(tmp_path):
    runner = FakeRunner(git_inside=False)
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / ".agent-loop-logs").mkdir()
    (codex_dir / "some-user-file.py").write_text("# user work", encoding="utf-8")

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not a git checkout"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:3] == ["gh", "repo", "clone"] for cmd, _cwd in runner.commands)


def test_stale_default_workdir_empty_is_recreated(tmp_path, capsys):
    """An empty workdir (no .git, no files) is treated as stale and re-cloned."""
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()  # exists but empty

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
        quiet=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    clone_cmds = [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "repo", "clone"]]
    assert any(cmd[4] == str(codex_dir) for cmd in clone_cmds), "Expected fresh clone of empty stale workdir"

    captured = capsys.readouterr()
    assert "Stale default codex workdir detected" in captured.err
    assert "recreating" in captured.err


def test_stale_default_workdir_git_only_is_recreated(tmp_path, capsys):
    """A workdir with only a .git dir (no working tree) is treated as stale and re-cloned."""
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / ".git").mkdir()  # .git present, but no source files

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
        quiet=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    clone_cmds = [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "repo", "clone"]]
    assert any(cmd[4] == str(codex_dir) for cmd in clone_cmds), "Expected fresh clone of git-only stale workdir"

    captured = capsys.readouterr()
    assert "Stale default codex workdir detected" in captured.err
    assert "recreating" in captured.err


def test_explicit_dir_not_git_checkout_is_not_recreated(tmp_path):
    runner = FakeRunner(git_inside=False)
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / ".agent-loop-logs").mkdir()

    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not a git checkout"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:3] == ["gh", "repo", "clone"] for cmd, _cwd in runner.commands)


def test_existing_auto_agent_dir_must_match_requested_repo(tmp_path):
    runner = FakeRunner(git_remote="git@github.com:OTHER/REPO.git")
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = make_config(
        tmp_path,
        codex_dir=codex_dir,
        reviewer="codex",
        auto_agent_dirs=("codex",),
        create_dirs=False,
    )
    config.claude_dir.mkdir(parents=True)
    config.gemini_dir.mkdir(parents=True)

    with pytest.raises(AgentLoopError, match="not 'OWNER/REPO'"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_agent_workdir_existing_file_fails_clearly(tmp_path):
    runner = FakeRunner()
    claude_path = tmp_path / "claude-file"
    claude_path.write_text("not a dir", encoding="utf-8")
    config = make_config(tmp_path, claude_dir=claude_path, create_dirs=False)

    with pytest.raises(AgentLoopError, match="not a directory"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_gemini_workdir_existing_file_fails_clearly(tmp_path):
    runner = FakeRunner()
    gemini_path = tmp_path / "gemini-file"
    gemini_path.write_text("not a dir", encoding="utf-8")
    config = make_config(
        tmp_path,
        reviewer="gemini",
        gemini_dir=gemini_path,
        create_dirs=False,
    )

    with pytest.raises(AgentLoopError, match="not a directory"):
        run_pr_loop(runner, pr_number=77, config=config)


def test_config_allows_same_coder_and_reviewer(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "codex",
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == "codex"
    assert config.reviewer == ("codex",)


def test_config_allows_coder_in_multiple_reviewers(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "codex",
        "--reviewer",
        "claude",
        "--reviewer",
        "codex",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == "codex"
    assert config.reviewer == ("claude", "codex")


def test_config_accepts_gemini_as_coder_and_reviewer(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        "gemini",
        "--reviewer",
        "claude",
        "--reviewer",
        "gemini",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == "gemini"
    assert config.reviewer == ("claude", "gemini")
    assert config.gemini_dir == tmp_path / "gemini"


@pytest.mark.parametrize(
    ("coder", "reviewer"),
    [
        ("agy", "codex"),
        ("codex", "agy"),
        ("antigravity", "codex"),
    ],
)
def test_config_normalizes_antigravity_agent_names(tmp_path, coder, reviewer):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--coder",
        coder,
        "--reviewer",
        reviewer,
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    config = config_from_args(args, FakeRunner())

    assert config.coder == ("antigravity" if coder == "agy" else coder)
    assert config.reviewer == (
        "antigravity" if reviewer == "agy" else reviewer,
    )


def test_config_rejects_duplicate_reviewers(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--reviewer",
        "codex",
        "--reviewer",
        "codex",
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    with pytest.raises(AgentLoopError, match="same agent more than once"):
        config_from_args(args, FakeRunner())


def test_config_rejects_alias_and_canonical_duplicate_reviewers(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--reviewer",
        "agy",
        "--reviewer",
        "antigravity",
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    with pytest.raises(AgentLoopError, match="same agent more than once"):
        config_from_args(args, FakeRunner())


@pytest.mark.parametrize("max_rounds", ["0", "-1"])
def test_config_rejects_non_positive_max_rounds(tmp_path, max_rounds):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--max-rounds",
        max_rounds,
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])

    with pytest.raises(AgentLoopError, match="--max-rounds must be greater than zero"):
        config_from_args(args, FakeRunner())


def test_config_defaults_do_not_bypass_agent_permissions(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    config = config_from_args(args, FakeRunner())

    assert config.claude_args == ()
    assert config.codex_args == ()
    assert config.gemini_args == ()


def test_config_can_opt_into_dangerous_agent_permissions(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--dangerous-agent-permissions",
    ])
    config = config_from_args(args, FakeRunner())

    assert config.claude_args == ("--dangerously-skip-permissions",)
    assert config.codex_args == ("--dangerously-bypass-approvals-and-sandbox",)
    assert config.gemini_args == ("--yolo", "--skip-trust")


def test_explicit_agent_args_replace_dangerous_profile(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
        "--dangerous-agent-permissions",
        "--claude-arg=--permission-mode",
        "--claude-arg=acceptEdits",
        "--codex-arg=--sandbox",
        "--codex-arg=workspace-write",
        "--gemini-arg=--approval-mode",
        "--gemini-arg=auto_edit",
    ])
    config = config_from_args(args, FakeRunner())

    assert config.claude_args == ("--permission-mode", "acceptEdits")
    assert config.codex_args == ("--sandbox", "workspace-write")
    assert config.gemini_args == ("--approval-mode", "auto_edit")


def test_resume_plan_round_prefers_latest_metadata_ledger_for_same_plan_replay():
    current_plan = "Revised plan.\n- Add the active-ledger replay test.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    subject = _plan_subject(current_plan)
    stale_item = UnresolvedReviewItem(
        item_id="item-3",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Stale plan replay item.",
        status="same-plan",
        source_status="same-plan",
    )
    active_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active plan replay item.",
        status="same-plan",
        source_status="same-plan",
    )
    stale_coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(stale_item,),
            canonical_plan=current_plan,
        ),
    )
    stale_reviewer_comment = _attach_round_metadata(
        "Still needs work."
        + prior_plan_item_dispositions("[item-3] same-plan: stale replay")
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject=subject,
            prior_items=(stale_item,),
            dispositions=(
                parse_plan_item_dispositions(
                    prior_plan_item_dispositions("[item-3] same-plan: stale replay"),
                    reviewer="OpenAI Codex",
                )[0],
            ),
            state="blocking",
        ),
    )
    active_coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(active_item,),
            canonical_plan=current_plan,
        ),
    )
    active_reviewer_comment = _attach_round_metadata(
        "Plan looks sound."
        + prior_plan_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Gemini",
            round_number=2,
            subject=subject,
            prior_items=(active_item,),
            dispositions=(
                parse_plan_item_dispositions(
                    prior_plan_item_dispositions("[item-1] resolved"),
                    reviewer="Google Gemini",
                )[0],
            ),
            state="approved",
        ),
    )

    resumed = _resume_plan_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=stale_coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=stale_reviewer_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:02:00Z", body=active_coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:03:00Z", body=active_reviewer_comment),
        ],
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    current_plan_text, resumed_state = resumed
    assert current_plan_text == current_plan
    assert [item.item_id for item in resumed_state.prior_items] == ["item-1"]
    assert resumed_state.next_unresolved_item_number == 4
    assert [record.metadata.agent for record in resumed_state.completed_reviews] == ["Gemini"]


def test_resume_plan_round_prefers_canonical_plan_metadata():
    public_body = (
        "Revised plan summary.\n\n### Plan steps\n1. Public body copy.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    canonical_plan = (
        "Revised plan summary.\n\n### Prior plan review item dispositions\n- None.\n\n"
        "### Plan steps\n1. Canonical copy."
    )
    coder_comment = _attach_round_metadata(
        public_body,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(canonical_plan),
            prior_items=(),
            canonical_plan=canonical_plan,
        ),
    )

    resumed = _resume_plan_round(
        [IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=coder_comment)],
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    current_plan, state = resumed
    assert current_plan == canonical_plan
    assert state.coder_output == canonical_plan


def test_resume_plan_round_prefers_structured_plan_revision_metadata_for_coder_output():
    public_body = (
        "## Revised plan\n\nRevised plan summary.\n\n### Plan steps\n1. Public body copy.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    raw_structured_revision = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revised plan summary.",
                "prior_plan_item_dispositions": [],
                "plan_steps": ["Canonical copy."],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    parsed = validate_structured_plan_revision(raw_structured_revision)
    assert parsed is not None
    canonical_plan = render_canonical_plan_revision(parsed, ())
    coder_comment = _attach_round_metadata(
        public_body,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(canonical_plan),
            prior_items=(),
            canonical_plan=canonical_plan,
            raw_structured_coder_response=raw_structured_revision,
        ),
    )

    resumed = _resume_plan_round(
        [IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=coder_comment)],
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    current_plan, state = resumed
    assert current_plan == canonical_plan
    assert state.coder_output == raw_structured_revision
    assert validate_structured_plan_revision(state.coder_output) is not None
    assert '"kind": "plan_revision"' not in _strip_round_metadata(coder_comment)


def test_resume_plan_round_falls_back_to_raw_body_for_markdown_plan():
    plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    coder_comment = _attach_round_metadata(
        plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(plan),
            prior_items=(),
        ),
    )

    resumed = _resume_plan_round(
        [IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=coder_comment)],
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    current_plan, state = resumed
    assert current_plan == plan
    assert state.coder_output == plan


def test_plan_subject_ignores_trailing_whitespace_added_by_metadata_round_trip():
    plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"

    attached = _attach_round_metadata(
        f"{plan}\n",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(f"{plan}\n"),
            prior_items=(),
        ),
    )

    assert _plan_subject(f"{plan}\n") == _plan_subject(_strip_round_metadata(attached))


def test_round_metadata_round_trip_preserves_canonical_plan():
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Claude",
        round_number=2,
        subject="abc",
        canonical_plan="Summary\n\n### Plan steps\n1. Canonical step.",
    )

    assert _decode_round_metadata(_encode_round_metadata(metadata)).canonical_plan == metadata.canonical_plan


def test_round_metadata_round_trip_preserves_compact_prior_summaries():
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Claude",
        round_number=2,
        subject="abc",
        compact_prior_summaries=("[item-1] resolved: full prior text",),
    )

    decoded = _decode_round_metadata(_encode_round_metadata(metadata))

    assert decoded.compact_prior_summaries == metadata.compact_prior_summaries


def test_decode_old_round_metadata_defaults_compact_prior_summaries_to_empty():
    payload = {
        "flow": "plan",
        "role": "coder",
        "agent": "Claude",
        "round_number": 2,
        "subject": "abc",
        "prior_items": [],
        "dispositions": [],
        "new_items": [],
        "state": None,
        "canonical_plan": None,
        "raw_structured_coder_response": None,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")

    assert _decode_round_metadata(encoded).compact_prior_summaries == ()


def test_resume_plan_round_restores_compact_prior_summaries_across_subject_change():
    old_plan = "Old plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    new_plan = "New plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_subject = _plan_subject(old_plan)
    new_subject = _plan_subject(new_plan)
    old_summary = "[item-1] resolved: old-subject resolved summary"
    old_coder_comment = _attach_round_metadata(
        old_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=old_subject,
            compact_prior_summaries=(old_summary,),
        ),
    )
    new_coder_comment = _attach_round_metadata(
        new_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=3,
            subject=new_subject,
            compact_prior_summaries=(old_summary,),
        ),
    )

    resumed = _resume_plan_round(
        [
            IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=old_coder_comment),
            IssueComment(author="bot", created_at="2026-05-20T09:10:00Z", body=new_coder_comment),
        ],
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    _current_plan, state = resumed
    assert state.compact_prior_summaries == (old_summary,)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"flow": "plan"},
        {
            "flow": "plan",
            "role": "coder",
            "agent": "Claude",
            "round_number": "not-an-int",
            "subject": "abc",
        },
    ],
)
def test_decode_round_metadata_rejects_missing_or_invalid_required_fields(payload):
    encoded = json.dumps(payload).encode("utf-8")

    with pytest.raises(AgentLoopError, match="Invalid AGENT_LOOP_META payload"):
        _decode_round_metadata(encoded=base64.urlsafe_b64encode(encoded).decode("ascii"))


def test_is_clarification_request_detects_marker():
    assert is_clarification_request("need more info\n<!-- AGENT_CLARIFY -->")
    assert is_clarification_request("<!-- agent_clarify -->")
    assert not is_clarification_request("done\n<!-- AGENT_STATE: blocking -->")


def test_is_clarification_request_state_marker_after_clarify_takes_precedence():
    # AGENT_PLAN_STATE after inline AGENT_CLARIFY example: issue #216 / #278 shape.
    # Inline (non-standalone) AGENT_CLARIFY never triggers, regardless of state markers.
    plan_with_embedded_clarify = (
        "Here is my plan.\n\n"
        "If I needed clarification I would emit <!-- AGENT_CLARIFY --> as a marker.\n\n"
        "But I have enough information, so here is the full plan:\n\n"
        "1. Do step one\n2. Do step two\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert not is_clarification_request(plan_with_embedded_clarify)

    # Inline AGENT_CLARIFY without any state marker: still not clarification.
    inline_only = (
        "The protocol supports <!-- AGENT_CLARIFY --> for clarification requests.\n\n"
        "Here is my fix."
    )
    assert not is_clarification_request(inline_only)

    # AGENT_STATE after inline AGENT_CLARIFY example: PR/coder blocking response.
    pr_response_with_embedded_clarify = (
        "The protocol supports <!-- AGENT_CLARIFY --> for clarification requests.\n\n"
        "Here is my fix.\n\n"
        "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert not is_clarification_request(pr_response_with_embedded_clarify)

    # AGENT_PR after inline AGENT_CLARIFY example: coder PR-creation response.
    pr_created_with_embedded_clarify = (
        "Use <!-- AGENT_CLARIFY --> if you need more info.\n\n"
        "Implemented the fix.\n\n"
        "<!-- AGENT_PR: 42 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert not is_clarification_request(pr_created_with_embedded_clarify)

    # PR URL after inline AGENT_CLARIFY: treated as final state marker.
    pr_url_with_embedded_clarify = (
        "Use <!-- AGENT_CLARIFY --> for questions.\n\n"
        "See https://github.com/OWNER/REPO/pull/99 for the PR."
    )
    assert not is_clarification_request(pr_url_with_embedded_clarify)

    # Real clarification request: standalone AGENT_CLARIFY is the final marker.
    real_clarify = "Which endpoint should I use?\n<!-- AGENT_CLARIFY -->\n-- Anthropic Claude"
    assert is_clarification_request(real_clarify)

    # Standalone AGENT_CLARIFY on its own line, after a state marker in prose.
    clarify_after_state = (
        "Round 1 ended with <!-- AGENT_STATE: blocking -->, but I still need more info.\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(clarify_after_state)

    # Standalone AGENT_CLARIFY on its own line, appearing after AGENT_PLAN_STATE in prose.
    plan_state_in_prose_clarify_last = (
        "The previous round used <!-- AGENT_PLAN_STATE: blocking --> to signal issues,\n"
        "but now I have a question:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(plan_state_in_prose_clarify_last)


def test_is_clarification_request_standalone_marker_positional_semantics():
    # Standalone AGENT_PLAN_STATE footer appearing BEFORE a standalone AGENT_CLARIFY
    # does NOT suppress it — AGENT_CLARIFY is the final marker and wins.
    plan_footer_then_clarify_appendix = (
        "Plan content.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n"
        "-- Anthropic Claude\n\n"
        "Appendix:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(plan_footer_then_clarify_appendix)

    # Standalone AGENT_STATE appearing BEFORE AGENT_CLARIFY also does not suppress.
    state_then_clarify = (
        "<!-- AGENT_STATE: blocking -->\n\n"
        "Note:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(state_then_clarify)

    # Standalone AGENT_STATE appearing AFTER AGENT_CLARIFY does suppress it.
    clarify_then_state = (
        "<!-- AGENT_CLARIFY -->\n"
        "<!-- AGENT_STATE: blocking -->"
    )
    assert not is_clarification_request(clarify_then_state)

    # Inline (non-standalone) AGENT_STATE in prose does NOT suppress AGENT_CLARIFY —
    # it may be a quoted reference to a previous round's state.
    inline_state_then_clarify = (
        "Round 1 ended with <!-- AGENT_STATE: blocking -->, but I still need more info.\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(inline_state_then_clarify)

    # Inline AGENT_PLAN_STATE in prose also does not suppress.
    inline_plan_state_then_clarify = (
        "The previous round used <!-- AGENT_PLAN_STATE: blocking --> to signal issues,\n"
        "but now I have a question:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(inline_plan_state_then_clarify)


def test_is_clarification_request_pr_marker_takes_precedence():
    # AGENT_PR: N standalone marker appearing AFTER AGENT_CLARIFY suppresses it.
    pr_after_clarify = (
        "<!-- AGENT_CLARIFY -->\n"
        "Actually I have enough info.\n"
        "<!-- AGENT_PR: 55 -->"
    )
    assert not is_clarification_request(pr_after_clarify)

    # AGENT_PR: N standalone marker appearing BEFORE AGENT_CLARIFY does NOT suppress —
    # AGENT_CLARIFY is the final marker and wins.
    pr_before_clarify = (
        "<!-- AGENT_PR: 55 -->\n"
        "<!-- AGENT_STATE: blocking -->\n\n"
        "Note:\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(pr_before_clarify)


def test_is_clarification_request_ignores_fenced_code_block_examples():
    # AGENT_CLARIFY on its own line inside a backtick fence: not clarification.
    fenced_no_state = (
        "Here's how the marker looks:\n\n"
        "```\n"
        "<!-- AGENT_CLARIFY -->\n"
        "```\n\n"
        "That's all."
    )
    assert not is_clarification_request(fenced_no_state)

    # Fenced example with AGENT_PLAN_STATE after the block: still not clarification.
    fenced_with_plan_state = (
        "Protocol example:\n\n"
        "```\n"
        "<!-- AGENT_CLARIFY -->\n"
        "```\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert not is_clarification_request(fenced_with_plan_state)

    # Fenced example where the code block appears AFTER a state marker: not clarification.
    state_then_fenced = (
        "<!-- AGENT_PLAN_STATE: blocking -->\n\n"
        "Appendix:\n\n"
        "```\n"
        "<!-- AGENT_CLARIFY -->\n"
        "```"
    )
    assert not is_clarification_request(state_then_fenced)

    # Tilde fence also excluded.
    tilde_fenced = (
        "~~~\n"
        "<!-- AGENT_CLARIFY -->\n"
        "~~~"
    )
    assert not is_clarification_request(tilde_fenced)

    # Real standalone AGENT_CLARIFY outside a fence: still detected.
    outside_fence = (
        "```\n"
        "some code\n"
        "```\n\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(outside_fence)

    # AGENT_CLARIFY both inside and outside a fence: outside occurrence is active.
    inside_and_outside = (
        "```\n"
        "<!-- AGENT_CLARIFY -->\n"
        "```\n\n"
        "<!-- AGENT_CLARIFY -->"
    )
    assert is_clarification_request(inside_and_outside)


def test_is_clarification_request_requires_clarify_at_end():
    # Non-blank, non-signature content after AGENT_CLARIFY means it's embedded.
    embedded_with_trailing = (
        "<!-- AGENT_CLARIFY -->\n\n"
        "Some trailing prose that isn't a signature.\n"
        "<!-- AGENT_STATE: blocking -->"
    )
    # AGENT_STATE suppresses it via the presence-based check above.
    assert not is_clarification_request(embedded_with_trailing)

    # Standalone AGENT_CLARIFY with only blank lines after it: valid.
    clarify_then_blank = "<!-- AGENT_CLARIFY -->\n\n"
    assert is_clarification_request(clarify_then_blank)

    # Standalone AGENT_CLARIFY with only a signature after it: valid.
    clarify_then_sig = "<!-- AGENT_CLARIFY -->\n-- Anthropic Claude\n"
    assert is_clarification_request(clarify_then_sig)

    # Standalone AGENT_CLARIFY with real prose content after it (no state marker):
    # should NOT be treated as an active clarification.
    clarify_then_prose = (
        "<!-- AGENT_CLARIFY -->\n\n"
        "Continuing thoughts about the plan.\n"
    )
    assert not is_clarification_request(clarify_then_prose)

    # AGENT_CLARIFY in plan body with plan footer after: suppressed by state marker
    # (presence-based check catches it before trailing-content check).
    in_plan_body = (
        "Here are my questions:\n\n"
        "<!-- AGENT_CLARIFY -->\n\n"
        "More explanation here.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n"
    )
    assert not is_clarification_request(in_plan_body)

    # Multiple AGENT_CLARIFY; last one has only signature trailing: valid.
    multi_clarify = (
        "First question set:\n<!-- AGENT_CLARIFY -->\n\nOther text.\n\n"
        "<!-- AGENT_CLARIFY -->\n-- Anthropic Claude\n"
    )
    assert is_clarification_request(multi_clarify)

    # Multiple AGENT_CLARIFY; last one has prose trailing: not valid.
    multi_clarify_bad = (
        "<!-- AGENT_CLARIFY -->\n\n"
        "<!-- AGENT_CLARIFY -->\n\n"
        "But wait, there's more content.\n"
    )
    assert not is_clarification_request(multi_clarify_bad)


# ---------------------------------------------------------------------------
# Reverse flow: Codex creates PR, Claude reviews
# ---------------------------------------------------------------------------


def test_public_response_file_instruction_mentions_plan_revision_human_ack_exception(tmp_path):
    prompt = with_public_response_file_instruction(
        "Review the PR.",
        tmp_path / "response.md",
    )

    assert "For structured plan revisions only" in prompt
    assert "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->" in prompt
    assert "`### Human requirements` section after the JSON object" in prompt
    assert "before the\n`AGENT_PLAN_STATE` footer" in prompt


# ---------------------------------------------------------------------------
# Repair pass tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# New tests for issue #246: repair approved reviews with active prior dispositions
# ---------------------------------------------------------------------------


# --- repair.py prompt content tests ---


# --- _reviewer_human_requirements_instruction tests ---


# --- _surfaced_reviewer_requirement_ids tests ---


# --- PR loop repair-first tests ---


# --- Plan loop repair-first tests ---


# --- Protocol regression tests ---


# ---------------------------------------------------------------------------
# Round 2 follow-up tests: same-pr/same-plan followup recording and FORMAT fix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests for issue #273: deterministic recovery of same-round prior-item dispositions
# ---------------------------------------------------------------------------


# --- Unit tests for strip_unknown_prior_item_dispositions ---


# --- Integration tests via _run_validated_agent ---


# ---------------------------------------------------------------------------
# Antigravity (agy) backend + Gemini retirement guidance (#215)
# ---------------------------------------------------------------------------


def test_runner_pty_reports_tty_and_strips_ansi(tmp_path):
    """The real PTY path: the child sees a TTY and ANSI codes are stripped."""
    import sys
    from coding_review_agent_loop.runner import Runner, strip_ansi

    assert strip_ansi("\x1b[31mred\x1b[0m\r\ndone") == "red\ndone"

    program = (
        "import sys\n"
        "sys.stdout.write('istty=%s\\n' % sys.stdout.isatty())\n"
        "sys.stdout.write('\\x1b[32mGREEN\\x1b[0m\\n')\n"
    )
    log_path = tmp_path / "logs" / "pty.log"
    result = Runner().run_with_log(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        log_path=log_path,
        label="PtyProbe",
        progress_interval_seconds=999,
        check=True,
        use_pty=True,
    )
    assert "istty=True" in result.stdout
    assert "GREEN" in result.stdout
    assert "\x1b[" not in result.stdout  # ANSI stripped from captured output
    assert result.returncode == 0


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_retries_dangling_symlink_spawn_and_recovers(
    monkeypatch,
    tmp_path,
    use_pty,
):
    import coding_review_agent_loop.runner as runner_module

    command_name = "bare-agent"
    missing_target = tmp_path / "updating-agent-target"
    command = tmp_path / command_name
    command.symlink_to(missing_target)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    runner = Runner()
    runner.remember_agent_command(command_name, str(command), "--codex-cmd")
    original_popen = runner_module.subprocess.Popen
    popen_calls = []
    sleep_calls = []

    def flaky_popen(*args, **kwargs):
        popen_calls.append(args[0])
        if len(popen_calls) == 1:
            raise FileNotFoundError(command_name)
        return original_popen(*args, **kwargs)

    def restore_command(delay):
        sleep_calls.append(delay)
        command.unlink()
        command.symlink_to(sys.executable)

    monkeypatch.setattr(runner_module.subprocess, "Popen", flaky_popen)
    monkeypatch.setattr(runner_module.time, "sleep", restore_command)

    result = runner.run_with_log(
        [command_name, "-c", "print('recovered')"],
        cwd=tmp_path,
        log_path=tmp_path / "logs" / f"retry-{use_pty}.log",
        label="Retry probe",
        progress_interval_seconds=999,
        use_pty=use_pty,
    )

    assert result.returncode == 0
    assert "recovered" in result.stdout
    assert len(popen_calls) == 2
    assert sleep_calls[0] == 2
    assert all(delay == 1 for delay in sleep_calls[1:])


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_dangling_symlink_spawn_retry_is_bounded(
    monkeypatch,
    tmp_path,
    use_pty,
):
    import coding_review_agent_loop.runner as runner_module

    command_name = "bare-agent"
    command = tmp_path / command_name
    command.symlink_to(tmp_path / "missing-target")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    runner = Runner()
    runner.remember_agent_command(command_name, str(command), "--codex-cmd")
    popen_calls = []
    sleep_calls = []

    def missing_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise FileNotFoundError(command_name)

    monkeypatch.setattr(runner_module.subprocess, "Popen", missing_popen)
    monkeypatch.setattr(
        runner_module.time,
        "sleep",
        lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(
        AgentLoopError,
        match=r"CLI not found on PATH.*--codex-cmd",
    ):
        runner.run_with_log(
            [command_name, "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / f"bounded-{use_pty}.log",
            label="Bounded retry probe",
            progress_interval_seconds=999,
            use_pty=use_pty,
        )

    assert len(popen_calls) == 3
    assert sleep_calls == [2, 2]


def test_runner_missing_command_without_dangling_evidence_does_not_retry(
    monkeypatch,
    tmp_path,
):
    import coding_review_agent_loop.runner as runner_module

    popen_calls = []
    sleep_calls = []

    def missing_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise FileNotFoundError("missing-agent")

    monkeypatch.setattr(runner_module.shutil, "which", lambda command: None)
    monkeypatch.setattr(runner_module.subprocess, "Popen", missing_popen)
    monkeypatch.setattr(
        runner_module.time,
        "sleep",
        lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(AgentLoopError, match="missing-agent CLI not found on PATH"):
        Runner().run_with_log(
            ["missing-agent", "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / "missing.log",
            label="Missing probe",
            progress_interval_seconds=999,
        )

    assert len(popen_calls) == 1
    assert sleep_calls == []


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_absolute_path_spawn_does_not_retry(monkeypatch, tmp_path, use_pty):
    """Absolute-path FileNotFoundError raises immediately (no retry, no sleep)."""
    import coding_review_agent_loop.runner as runner_module

    abs_cmd = str(tmp_path / "no-such-binary")
    popen_calls = []
    sleep_calls = []

    def missing_popen(*args, **kwargs):
        popen_calls.append(args[0])
        raise FileNotFoundError(abs_cmd)

    monkeypatch.setattr(runner_module.subprocess, "Popen", missing_popen)
    monkeypatch.setattr(
        runner_module.time, "sleep", lambda delay: sleep_calls.append(delay),
    )

    with pytest.raises(AgentLoopError, match="not found or not executable"):
        Runner().run_with_log(
            [abs_cmd, "--version"],
            cwd=tmp_path,
            log_path=tmp_path / "logs" / f"abs-no-retry-{use_pty}.log",
            label="Absolute-path no-retry probe",
            progress_interval_seconds=999,
            use_pty=use_pty,
        )

    assert len(popen_calls) == 1
    assert sleep_calls == []


# ---------------------------------------------------------------------------
# Dynamic model-specific signatures (#332)
# ---------------------------------------------------------------------------


def test_agent_signature_generic_without_config():
    from coding_review_agent_loop.agents.registry import agent_signature
    assert agent_signature("codex") == "OpenAI Codex"
    assert agent_signature("antigravity") == "Google Antigravity"


def test_agent_signature_uses_configured_model(tmp_path):
    from coding_review_agent_loop.agents.registry import agent_signature
    config = make_config(tmp_path, codex_model="gpt-5.2-codex", codex_reasoning_effort="high")
    assert agent_signature("codex", config) == "OpenAI Codex: gpt-5.2-codex (high)"
    # antigravity model is always declared (effort already embedded).
    assert agent_signature("antigravity", config) == "Google Antigravity: Gemini 3.5 Flash (High)"
    # gemini with no declared model falls back to the generic signature.
    assert agent_signature("gemini", make_config(tmp_path)) == "Google Gemini"


def test_agent_signature_model_used_overrides_config(tmp_path):
    from coding_review_agent_loop.agents.registry import agent_signature
    config = make_config(tmp_path, antigravity_model="Gemini 3.1 Pro (High)")
    # #333 fallback: the model that actually ran wins over the configured one.
    assert (
        agent_signature("antigravity", config, "Gemini 3.5 Flash (High)")
        == "Google Antigravity: Gemini 3.5 Flash (High)"
    )


def test_config_rejects_model_arg_conflicts(tmp_path):
    for kwargs in (
        {"codex_model": "gpt-5", "codex_args": ("--model", "other")},
        {"codex_reasoning_effort": "high", "codex_args": ("-c", 'model_reasoning_effort="low"')},
        {"gemini_model": "g", "gemini_args": ("--model", "other")},
        {"claude_model": "c", "claude_args": ("--model", "other")},
        {"antigravity_args": ("--model", "x")},
    ):
        with pytest.raises(AgentLoopError, match="conflicts with"):
            make_config(tmp_path, **kwargs)


def test_config_rejects_codex_effort_without_model(tmp_path):
    # Rollout model detection is best-effort, so effort alone cannot be labeled
    # reliably and requires an explicit --codex-model.
    with pytest.raises(AgentLoopError, match="requires --codex-model"):
        make_config(tmp_path, codex_reasoning_effort="high")
    # With a model it's accepted.
    config = make_config(tmp_path, codex_model="gpt-5", codex_reasoning_effort="high")
    assert config.codex_reasoning_effort == "high"


def test_config_allows_declared_model_without_conflict(tmp_path):
    config = make_config(tmp_path, codex_model="gpt-5", gemini_model="g", claude_model="c")
    assert config.codex_model == "gpt-5"
    assert config.gemini_model == "g"
    assert config.claude_model == "c"


def test_codex_backend_passes_model_and_effort(tmp_path):
    from coding_review_agent_loop.agents.codex import CodexBackend
    runner = FakeRunner(codex_outputs=[("STATE: approved\n\nok", 0)])
    config = make_config(tmp_path, codex_model="gpt-5.2-codex", codex_reasoning_effort="high")
    result = CodexBackend().run(runner, config, "Review", run_id="r")
    cmd = runner.commands[-1][0]
    assert cmd[cmd.index("--model") + 1] == "gpt-5.2-codex"
    assert 'model_reasoning_effort="high"' in cmd
    assert result.model_used == "gpt-5.2-codex (high)"


def test_gemini_backend_passes_model_and_sets_model_used(tmp_path):
    import coding_review_agent_loop.agents.gemini as gm
    runner = FakeRunner(gemini_outputs=[("STATE: approved\n\nok", 0)])
    config = make_config(tmp_path, gemini_model="gemini-3.5-flash")
    result = gm.BACKEND.run(runner, config, "Review", run_id="r")
    cmd = runner.commands[-1][0]
    assert cmd[cmd.index("--model") + 1] == "gemini-3.5-flash"
    assert result.model_used == "gemini-3.5-flash"


def test_claude_backend_passes_model_when_declared(tmp_path):
    from coding_review_agent_loop.agents.claude import ClaudeBackend
    runner = FakeRunner(claude_outputs=[("STATE: approved\n\nok", 0)])
    config = make_config(tmp_path, claude_model="opus")
    ClaudeBackend().run(runner, config, "Review", run_id="r")
    cmd = runner.commands[-1][0]
    assert cmd[cmd.index("--model") + 1] == "opus"


def test_public_reviewer_name_config_aware_no_leakage(tmp_path):
    from coding_review_agent_loop.comment_rendering import _public_reviewer_name
    config = make_config(tmp_path, codex_model="gpt-5", antigravity_model="Gemini 3.1 Pro (High)")
    assert _public_reviewer_name("Codex", config) == "OpenAI Codex: gpt-5"
    assert _public_reviewer_name("Antigravity", config) == "Google Antigravity: Gemini 3.1 Pro (High)"
    # No declared model → generic; unknown display name → passthrough.
    assert _public_reviewer_name("Claude", config) == "Anthropic Claude"
    assert _public_reviewer_name("Codex") == "OpenAI Codex"
    assert _public_reviewer_name("Somebody") == "Somebody"


def test_render_public_agent_comment_stamps_model_for_every_kind():
    model = "Gemini 3.1 Pro (High)"

    pr_review = parse_pr_review(
        structured_pr_review(state="approved", reviewer="Google Antigravity"),
        reviewer="Google Antigravity",
    )
    plan_review = parse_plan_review(
        structured_plan_review(state="approved", reviewer="Google Antigravity"),
        reviewer="Google Antigravity",
    )
    coder_followup = validate_structured_coder_followup(
        structured_coder_followup(state="approved", reviewer="Google Antigravity")
    )
    plan_revision = validate_structured_plan_revision(
        structured_plan_revision(reviewer="Google Antigravity")
    )
    assert coder_followup is not None
    assert plan_revision is not None

    rendered = [
        render_public_agent_comment(
            kind="pr_review",
            parsed=pr_review,
            agent="Antigravity",
            dispositions=pr_review.dispositions,
            model_used=model,
        ),
        render_public_agent_comment(
            kind="plan_review",
            parsed=plan_review,
            agent="Antigravity",
            dispositions=plan_review.dispositions,
            model_used=model,
        ),
        render_public_agent_comment(
            kind="coder_followup",
            parsed=coder_followup,
            agent="antigravity",
            model_used=model,
        ),
        render_public_agent_comment(
            kind="plan_revision",
            parsed=plan_revision,
            agent="antigravity",
            raw_text=structured_plan_revision(reviewer="Google Antigravity"),
            model_used=model,
        ),
    ]

    assert all(comment.endswith(f"-- Google Antigravity: {model}") for comment in rendered)


# ---------------------------------------------------------------------------
# Antigravity prompt — turn-end requirement (#385)
# ---------------------------------------------------------------------------


def test_base_response_file_instruction_includes_must_write_before_turn_ends(tmp_path):
    from coding_review_agent_loop.agents.base import with_public_response_file_instruction
    composed = with_public_response_file_instruction("BASE PROMPT", tmp_path / "response.md")
    assert "before your turn ends" in composed


# ── Tests: issue #400 – toolPermission: "strict" injection for reviewer ────────


def test_reviewer_and_coder_call_sites_pass_correct_role(tmp_path):
    """_run_validated_agent propagates role= to run_agent_result correctly."""
    from unittest.mock import patch
    from coding_review_agent_loop.agents.base import AgentResult

    config = make_config(tmp_path, reviewer="gemini", agent_max_retries=0)
    captured_roles: list = []

    def mock_run(runner, *, agent, config, prompt, session_id=None, run_id=None, role=None):
        captured_roles.append(role)
        return AgentResult(text="ok")

    with patch("coding_review_agent_loop.orchestrator.run_agent_result", mock_run):
        _run_validated_agent(
            FakeRunner(),
            agent="gemini",
            config=config,
            prompt="Review the plan.",
            marker_description="test",
            validate=lambda text: text,
            role="reviewer",
        )

    assert captured_roles == ["reviewer"]
    captured_roles.clear()

    with patch("coding_review_agent_loop.orchestrator.run_agent_result", mock_run):
        _run_validated_agent(
            FakeRunner(),
            agent="gemini",
            config=config,
            prompt="Implement.",
            marker_description="test",
            validate=lambda text: text,
        )

    assert captured_roles == [None]


def test_run_agent_result_passes_role_to_backend(tmp_path, monkeypatch):
    """run_agent_result threads role= through to the backend's run() method."""
    from coding_review_agent_loop.agents.registry import run_agent_result
    from coding_review_agent_loop.agents.base import AgentResult
    from coding_review_agent_loop.agents import registry as reg_mod

    captured: dict = {}

    class TrackingBackend:
        name = "gemini"
        display_name = "Gemini"
        signature = "Google Gemini"

        def workdir(self, config):
            return tmp_path

        def default_args(self, *, dangerous):
            return ()

        def run(self, runner, config, prompt, session_id=None, run_id=None, role=None):
            captured["role"] = role
            return AgentResult(text="ok")

    monkeypatch.setitem(reg_mod.BACKENDS, "gemini", TrackingBackend())
    config = make_config(tmp_path, reviewer="gemini")
    run_agent_result(FakeRunner(), agent="gemini", config=config, prompt="Test", role="reviewer")
    assert captured["role"] == "reviewer"


# ---------------------------------------------------------------------------
# Shared PR review guidance unit and integration tests (#413, #417)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PostedRoundMetadata.model_used and normalize_freeform_signature (#416)
# ---------------------------------------------------------------------------


def test_posted_round_metadata_model_used_round_trip():
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Claude",
        round_number=1,
        subject="abc",
        model_used="gpt-5.5 (medium)",
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.model_used == "gpt-5.5 (medium)"


def test_posted_round_metadata_model_used_none_round_trip():
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Claude",
        round_number=1,
        subject="abc",
        model_used=None,
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.model_used is None


def test_posted_round_metadata_model_used_backward_compat():
    payload = {
        "flow": "plan",
        "role": "coder",
        "agent": "Claude",
        "round_number": 1,
        "subject": "abc",
        "prior_items": [],
        "dispositions": [],
        "new_items": [],
        "state": None,
        "canonical_plan": None,
        "raw_structured_coder_response": None,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")
    decoded = _decode_round_metadata(encoded)
    assert decoded.model_used is None


def test_resume_plan_round_preserves_stored_model_used():
    plan = "Plan content.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude"
    coder_comment = _attach_round_metadata(
        plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=_plan_subject(plan),
        ),
    )
    review_text = structured_plan_review(state="approved", summary="LGTM.")
    reviewer_comment = _attach_round_metadata(
        review_text,
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=_plan_subject(plan),
            state="approved",
            model_used="gpt-5.5 (medium)",
        ),
    )
    resumed = _resume_plan_round(
        [
            IssueComment(author="bot", created_at="2026-05-20T09:00:00Z", body=coder_comment),
            IssueComment(author="bot", created_at="2026-05-20T09:01:00Z", body=reviewer_comment),
        ],
        configured_reviewers=("codex",),
    )
    assert resumed is not None
    _current_plan, state = resumed
    assert state.completed_reviews[0].metadata.model_used == "gpt-5.5 (medium)"


def test_normalize_freeform_signature_replaces_existing(tmp_path):
    config = make_config(tmp_path)
    result = normalize_freeform_signature(
        "Plan text.\n-- OpenAI Codex",
        agent="codex",
        config=config,
        model_used="gpt-5.5 (medium)",
    )
    assert result.endswith("-- OpenAI Codex: gpt-5.5 (medium)")
    assert "Plan text." in result


def test_normalize_freeform_signature_appends_when_absent(tmp_path):
    config = make_config(tmp_path)
    result = normalize_freeform_signature(
        "Plan text without a signature.",
        agent="codex",
        config=config,
        model_used="gpt-5.5 (medium)",
    )
    assert result.endswith("-- OpenAI Codex: gpt-5.5 (medium)")
    assert "Plan text without a signature." in result


def test_normalize_freeform_signature_skips_html_comments(tmp_path):
    config = make_config(tmp_path)
    text = "Plan text.\n-- OpenAI Codex\n<!-- AGENT_PLAN_STATE: approved -->"
    result = normalize_freeform_signature(
        text,
        agent="codex",
        config=config,
        model_used="gpt-5.5 (medium)",
    )
    assert "-- OpenAI Codex: gpt-5.5 (medium)" in result
    assert "<!-- AGENT_PLAN_STATE: approved -->" in result
    assert result.endswith("<!-- AGENT_PLAN_STATE: approved -->")


def test_normalize_freeform_signature_no_duplicate_when_already_canonical(tmp_path):
    config = make_config(tmp_path)
    canonical = "Plan text.\n-- OpenAI Codex: gpt-5.5 (medium)"
    result = normalize_freeform_signature(
        canonical,
        agent="codex",
        config=config,
        model_used="gpt-5.5 (medium)",
    )
    assert result == canonical


def test_run_plan_loop_freeform_initial_plan_includes_model(tmp_path):
    plan_text = "My initial plan.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude"
    reviewer_text = structured_plan_review(summary="LGTM.")
    reviewer_marker = parse_plan_review(reviewer_text, reviewer="OpenAI Codex")

    def fake_run_validated_agent(runner, *, agent, **kwargs):
        if agent == "claude":
            return ValidatedAgentResponse(
                text=plan_text,
                model_used="gpt-5.5 (medium)",
                session_id=None,
                marker_value=None,
            )
        return ValidatedAgentResponse(
            text=reviewer_text,
            model_used=None,
            session_id=None,
            marker_value=reviewer_marker,
        )

    runner = FakeRunner()
    config = make_config(tmp_path, coder="claude", reviewer="codex")
    with patch(
        "coding_review_agent_loop.orchestrator._run_validated_agent",
        side_effect=fake_run_validated_agent,
    ):
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    posted_body = runner.issue_comments[0]["body"]
    stripped = _strip_round_metadata(posted_body)
    assert stripped.endswith("-- Anthropic Claude: gpt-5.5 (medium)")


def test_run_plan_loop_freeform_revision_includes_model(tmp_path):
    initial_plan = "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    blocking_review_text = structured_plan_review(
        state="blocking",
        summary="Missing test strategy.",
        blocking_plan_issues=["Missing test strategy."],
    )
    blocking_review_marker = parse_plan_review(blocking_review_text, reviewer="OpenAI Codex")
    revised_plan = "Revised plan.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude"
    approved_review_text = structured_plan_review(
        state="approved",
        summary="LGTM.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    )
    approved_review_marker = parse_plan_review(approved_review_text, reviewer="OpenAI Codex")

    call_count = [0]

    def fake_run_validated_agent(runner, *, agent, **kwargs):
        call_count[0] += 1
        if agent == "claude":
            if call_count[0] == 1:
                return ValidatedAgentResponse(
                    text=initial_plan, model_used=None, session_id=None, marker_value=None
                )
            return ValidatedAgentResponse(
                text=revised_plan,
                model_used="gpt-5.5 (medium)",
                session_id=None,
                marker_value=None,
            )
        if call_count[0] == 2:
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

    runner = FakeRunner()
    config = make_config(tmp_path, coder="claude", reviewer="codex")
    with patch(
        "coding_review_agent_loop.orchestrator._run_validated_agent",
        side_effect=fake_run_validated_agent,
    ):
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    # Second issue_comments entry is the plan revision (index 1: reviewer, index 2: revision)
    revision_body = runner.issue_comments[2]["body"]
    stripped = _strip_round_metadata(revision_body)
    assert stripped.endswith("-- Anthropic Claude: gpt-5.5 (medium)")


