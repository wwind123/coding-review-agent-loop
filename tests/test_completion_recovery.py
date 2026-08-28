"""Tests for the bounded Claude completion-recovery pass (#588).

Covers the low-level ``_attempt_claude_completion_recovery`` helper directly
(all terminal outcomes: success, agent-declared unavailable, transport
failure, still-invalid text, and response-file attempt-isolation) plus
end-to-end wiring through ``run_issue_loop``'s direct implementation path
(exactly-one-resume bounded retry, GitHub posting, and negative controls for
call sites that never opt in).
"""

import json

import pytest

from coding_review_agent_loop.cli import run_issue_loop
from coding_review_agent_loop.errors import AgentInvocationError, AgentLoopError
from coding_review_agent_loop.orchestrator import (
    CompletionRecoveryPolicy,
    ValidatedAgentResponse,
    _attempt_claude_completion_recovery,
    _new_usage_context,
)
from coding_review_agent_loop.protocol import (
    AgentUnavailable,
    StructuredIssueImplementation,
    parse_agent_unavailable,
    validate_structured_issue_implementation,
)

from agent_loop_helpers import FakeRunner, make_config, structured_issue_implementation


def _validate_structured_implementation_result(text):
    parsed = validate_structured_issue_implementation(text)
    if parsed is None:
        raise AgentLoopError(
            "Issue implementation response must use the required structured `issue_implementation` format."
        )
    return parsed


def _claude_resume_commands(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def test_recovery_success_returns_validated_response_and_resumes_session(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_issue_implementation(pr_number=99),
        ],
    )
    config = make_config(tmp_path, coder="claude")

    outcome = _attempt_claude_completion_recovery(
        runner,
        config=config,
        completion_recovery=CompletionRecoveryPolicy(issue_number=56),
        session_id="sess-1",
        validate=_validate_structured_implementation_result,
        usage_context=_new_usage_context(config),
        run_id="run-1",
        role=None,
        label=None,
        timeout_seconds=None,
    )

    assert outcome.validated is not None
    assert isinstance(outcome.validated, ValidatedAgentResponse)
    assert isinstance(outcome.validated.marker_value, StructuredIssueImplementation)
    assert outcome.validated.marker_value.pr_number == 99
    assert outcome.terminal_public_response is None
    claude_commands = _claude_resume_commands(runner)
    assert len(claude_commands) == 1
    assert "--resume" in claude_commands[0]
    assert claude_commands[0][claude_commands[0].index("--resume") + 1] == "sess-1"
    # Success is never posted by the recovery helper itself; the caller
    # (_implement_approved_issue / run_issue_loop) posts PR-success comments
    # through the existing post_pr_comment/post_issue_pr_handoff_comment path.
    assert runner.comments == []


def test_recovery_success_no_pr_blocking_result_is_not_posted_by_helper(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_issue_implementation(
                pr_number=None,
                summary="Cannot proceed safely.",
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude")

    outcome = _attempt_claude_completion_recovery(
        runner,
        config=config,
        completion_recovery=CompletionRecoveryPolicy(issue_number=56),
        session_id="sess-1",
        validate=validate_structured_issue_implementation,
        usage_context=_new_usage_context(config),
        run_id="run-1",
        role=None,
        label=None,
        timeout_seconds=None,
    )

    assert outcome.validated is not None
    assert isinstance(outcome.validated.marker_value, StructuredIssueImplementation)
    assert outcome.validated.marker_value.pr_number is None
    # Posting a genuine no-PR terminal to the issue is the caller's job
    # (_post_no_pr_implementation_terminal_comment), not this helper's.
    assert runner.comments == []


@pytest.mark.parametrize("retryable", [True, False])
def test_recovery_agent_declared_unavailable_is_terminal_regardless_of_retryable(
    tmp_path, retryable
):
    unavailable_text = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "agent_unavailable",
                "retryable": retryable,
                "category": "environment",
                "summary": "Sandbox lost network access mid-resume.",
                "suggested_action": "Retry once connectivity is restored.",
            }
        )
        + "\n<!-- AGENT_UNAVAILABLE -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(claude_outputs=[unavailable_text])
    config = make_config(tmp_path, coder="claude")
    validate_calls = []

    def _validate(text):
        validate_calls.append(text)
        return _validate_structured_implementation_result(text)

    outcome = _attempt_claude_completion_recovery(
        runner,
        config=config,
        completion_recovery=CompletionRecoveryPolicy(issue_number=56),
        session_id="sess-1",
        validate=_validate,
        usage_context=_new_usage_context(config),
        run_id="run-1",
        role=None,
        label=None,
        timeout_seconds=None,
    )

    assert outcome.validated is None
    assert outcome.failure_category == "agent-unavailable"
    # Never sent through the ordinary implementation validator/PR checks: the
    # bounded one-recovery-attempt policy overrides the agent's own retryable
    # preference, so there is no second --resume and no PR validation either
    # way.
    assert validate_calls == []
    assert outcome.terminal_public_response == unavailable_text
    assert runner.comments == [unavailable_text]
    assert outcome.result.response_file_path.read_text(encoding="utf-8") == unavailable_text
    assert len(_claude_resume_commands(runner)) == 1


def test_recovery_transport_failure_nonzero_exit_synthesizes_unavailable(tmp_path):
    runner = FakeRunner(claude_outputs=[("agent crashed", 1)])
    config = make_config(tmp_path, coder="claude")

    outcome = _attempt_claude_completion_recovery(
        runner,
        config=config,
        completion_recovery=CompletionRecoveryPolicy(issue_number=56),
        session_id="sess-1",
        validate=_validate_structured_implementation_result,
        usage_context=_new_usage_context(config),
        run_id="run-1",
        role=None,
        label=None,
        timeout_seconds=None,
    )

    assert outcome.validated is None
    assert outcome.failure_category == "agent-unavailable"
    parsed = parse_agent_unavailable(outcome.terminal_public_response)
    assert parsed is not None
    assert parsed.retryable is False
    assert parsed.category == "tooling"
    assert runner.comments == [outcome.terminal_public_response]
    assert outcome.result.response_file_path.read_text(encoding="utf-8") == outcome.terminal_public_response
    assert len(_claude_resume_commands(runner)) == 1


@pytest.mark.parametrize("returncode", [1, None])
def test_recovery_accepts_valid_response_file_after_failed_exit(tmp_path, returncode):
    valid = structured_issue_implementation(pr_number=99)
    runner = FakeRunner(
        claude_outputs=[("Error: timeout waiting for response", returncode)],
        public_response_outputs=[valid],
    )
    usage = _new_usage_context(make_config(tmp_path, coder="claude"))
    config = make_config(tmp_path, coder="claude")
    outcome = _attempt_claude_completion_recovery(
        runner, config=config, completion_recovery=CompletionRecoveryPolicy(issue_number=56),
        session_id="sess-1", validate=validate_structured_issue_implementation,
        usage_context=usage, run_id="run-1", role=None, label=None, timeout_seconds=30.0,
    )
    assert outcome.validated is not None
    assert outcome.validated.text == valid
    assert outcome.validated.acquisition_returncode == returncode
    assert outcome.validated.acquisition_outcome == (
        "accepted_timeout" if returncode is None else "accepted_nonzero_exit"
    )
    assert outcome.terminal_public_response is None
    assert runner.comments == []
    assert usage.records[0].validation_status == "validated"


def test_recovery_transport_failure_empty_output_synthesizes_unavailable(tmp_path):
    runner = FakeRunner(claude_outputs=[("", 0)])
    config = make_config(tmp_path, coder="claude")

    outcome = _attempt_claude_completion_recovery(
        runner,
        config=config,
        completion_recovery=CompletionRecoveryPolicy(issue_number=56),
        session_id="sess-1",
        validate=_validate_structured_implementation_result,
        usage_context=_new_usage_context(config),
        run_id="run-1",
        role=None,
        label=None,
        timeout_seconds=None,
    )

    assert outcome.validated is None
    assert parse_agent_unavailable(outcome.terminal_public_response) is not None
    assert len(_claude_resume_commands(runner)) == 1


def test_recovery_transport_failure_timeout_synthesizes_unavailable(tmp_path):
    runner = FakeRunner(claude_outputs=[("", None)])
    config = make_config(tmp_path, coder="claude")

    outcome = _attempt_claude_completion_recovery(
        runner,
        config=config,
        completion_recovery=CompletionRecoveryPolicy(issue_number=56),
        session_id="sess-1",
        validate=_validate_structured_implementation_result,
        usage_context=_new_usage_context(config),
        run_id="run-1",
        role=None,
        label=None,
        timeout_seconds=30.0,
    )

    assert outcome.validated is None
    parsed = parse_agent_unavailable(outcome.terminal_public_response)
    assert parsed is not None
    assert parsed.category == "environment"
    assert "30" in outcome.error
    assert len(_claude_resume_commands(runner)) == 1


def test_recovery_repeated_background_wait_is_deterministic_not_unavailable(tmp_path):
    # Two incomplete textual responses in a row (the original attempt and the
    # resume attempt): the resume attempt still has no valid PR/blocking/
    # clarify marker.
    runner = FakeRunner(
        claude_outputs=["I'll wait for the background test run to finish, one more time."],
    )
    config = make_config(tmp_path, coder="claude")

    outcome = _attempt_claude_completion_recovery(
        runner,
        config=config,
        completion_recovery=CompletionRecoveryPolicy(issue_number=56),
        session_id="sess-1",
        validate=_validate_structured_implementation_result,
        usage_context=_new_usage_context(config),
        run_id="run-1",
        role=None,
        label=None,
        timeout_seconds=None,
    )

    assert outcome.validated is None
    assert outcome.failure_category == "deterministic"
    assert outcome.terminal_public_response is None
    assert "again deferred to background work" in outcome.error
    assert runner.comments == []
    assert len(_claude_resume_commands(runner)) == 1


def test_recovery_response_file_is_attempt_local_even_when_original_file_still_has_content(
    tmp_path,
):
    # The original (failed) attempt's public response file stays on disk with
    # real content; the resume attempt writes nothing to its own (distinct)
    # response file. Validation, the terminal payload, and salvage must all
    # come from the recovery attempt's own state, never the original's.
    from coding_review_agent_loop.agents.registry import run_agent_result

    runner = FakeRunner(
        # Only the original attempt's file gets written; the resume
        # attempt's own (distinct, freshly minted) response file is never
        # written by the fake agent, simulating a resume that produced no
        # response file at all.
        public_response_outputs=["I'll wait for the background test run to finish."],
        claude_outputs=[
            "I'll wait for the background test run to finish.",
            "I'll wait for the background test run to finish, again.",
        ],
    )
    config = make_config(tmp_path, coder="claude")

    original_result = run_agent_result(
        runner,
        agent="claude",
        config=config,
        prompt="Implement issue 56.",
        session_id=None,
        run_id="run-1",
    )
    assert original_result.response_file_text == "I'll wait for the background test run to finish."

    outcome = _attempt_claude_completion_recovery(
        runner,
        config=config,
        completion_recovery=CompletionRecoveryPolicy(issue_number=56),
        session_id="sess-1",
        validate=_validate_structured_implementation_result,
        usage_context=_new_usage_context(config),
        run_id="run-1",
        role=None,
        label=None,
        timeout_seconds=None,
    )

    # Attempt-local by construction: a distinct (freshly minted) response
    # file, and no fallback to the original attempt's file text even though
    # that file is still on disk with real content.
    assert outcome.result.response_file_path != original_result.response_file_path
    assert outcome.result.response_file_text is None
    assert outcome.result.text == "I'll wait for the background test run to finish, again."
    assert original_result.response_file_path.read_text(encoding="utf-8") == (
        "I'll wait for the background test run to finish."
    )
    assert outcome.validated is None
    assert "required structured" in outcome.error


def test_issue_loop_recovers_after_one_bounded_resume_then_succeeds(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "result": "I'll wait for the background test run to finish.",
                    "session_id": "sess-1",
                }
            ),
            structured_issue_implementation(pr_number=77, summary="Created PR."),
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "LGTM."
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    claude_commands = _claude_resume_commands(runner)
    resume_commands = [cmd for cmd in claude_commands if "--resume" in cmd]
    assert len(resume_commands) == 1
    assert resume_commands[0][resume_commands[0].index("--resume") + 1] == "sess-1"
    # No spurious AGENT_UNAVAILABLE comment: the recovered PR flow posts
    # normally through the existing PR-review comment path.
    assert not any(
        '"kind": "agent_unavailable"' in comment for comment in runner.comments
    )


def test_issue_loop_recovers_from_claude_waiting_on_background_wording(tmp_path):
    """Match the exact #531 completion text, not only ``waiting for`` variants."""
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "result": (
                        "Waiting on the background test run and the exit-monitor; "
                        "I'll continue once results arrive."
                    ),
                    "session_id": "sess-waiting-on",
                }
            ),
            structured_issue_implementation(pr_number=88, summary="Created PR."),
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    resume_commands = [
        cmd for cmd in _claude_resume_commands(runner) if "--resume" in cmd
    ]
    assert len(resume_commands) == 1
    assert resume_commands[0][resume_commands[0].index("--resume") + 1] == "sess-waiting-on"


def test_issue_loop_repeated_background_wait_reports_deterministic_failure(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "result": "I'll wait for the background test run to finish.",
                    "session_id": "sess-repeat-wait",
                }
            ),
            "I'll wait for the background test run to finish again.",
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentInvocationError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    assert excinfo.value.failure_category == "deterministic"
    assert excinfo.value.terminal_public_response is None
    assert "again deferred to background work" in str(excinfo.value)
    assert len(_claude_resume_commands(runner)) == 2
    assert runner.comments == []


def test_issue_loop_exhausts_recovery_and_raises_with_terminal_public_response(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "result": "I'll wait for the background test run to finish.",
                    "session_id": "sess-1",
                }
            ),
            ("", 1),
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentInvocationError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    assert excinfo.value.terminal_public_response is not None
    parsed = parse_agent_unavailable(excinfo.value.terminal_public_response)
    assert parsed is not None
    assert parsed.retryable is False

    claude_commands = _claude_resume_commands(runner)
    resume_commands = [cmd for cmd in claude_commands if "--resume" in cmd]
    assert len(resume_commands) == 1
    # Bounded: no ordinary retries stacked on top of the single resume call.
    assert len(claude_commands) == 2
    assert runner.comments == [excinfo.value.terminal_public_response]


def test_issue_loop_does_not_resume_for_unrelated_failure_text(tmp_path):
    # Missing marker, but nothing about backgrounded/deferred completion work:
    # this must fall back to today's existing failure/retry behavior, never a
    # resume attempt.
    config = make_config(tmp_path)
    plain_failure = "I do not have enough information to proceed."
    runner = FakeRunner(claude_outputs=[plain_failure] * (config.agent_max_retries + 1))

    with pytest.raises(AgentInvocationError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    assert excinfo.value.terminal_public_response is None
    claude_commands = _claude_resume_commands(runner)
    assert not any("--resume" in cmd for cmd in claude_commands)
    assert runner.comments == []


def test_coder_followup_pr_loop_never_resumes_even_with_backgrounding_language(tmp_path):
    # The coder follow-up / PR loop call site does not pass completion_recovery
    # (#588 scopes the bounded resume to the two direct implementation call
    # sites only), so a backgrounded-sounding follow-up failure must exhaust
    # ordinary retries/repair instead of triggering a --resume.
    config = make_config(tmp_path)
    backgrounded_followup_failure = "I'll wait for the background test run to finish."
    runner = FakeRunner(
        claude_outputs=[
            structured_issue_implementation(pr_number=77, summary="Created PR."),
        ]
        + [backgrounded_followup_failure] * (config.agent_max_retries + 1),
        codex_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )

    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config)

    claude_commands = _claude_resume_commands(runner)
    assert not any("--resume" in cmd for cmd in claude_commands)
