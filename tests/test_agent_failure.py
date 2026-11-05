"""Agent-failure tests: failure suggestions and validated responses (#1543)."""

import pytest

import coding_review_agent_loop.agent_failure as agent_failure
from issue_loop_test_helpers import FakeRunner, _TRANSIENT_REPAIR_SUGGESTION
from coding_review_agent_loop.salvage import SalvageContext
from agent_loop_helpers import make_config


def test_repair_provider_failure_suggestion_matches_transient_reason_only():
    reason = (
        "plan_review is invalid; repair invocation failure: "
        "antigravity/Gemini 3.8 Flash (Medium): transient_provider_error "
        "(model-access validation errors)"
    )
    assert (
        agent_failure._failure_suggestion("repair-provider-failure", reason, "Codex")
        == _TRANSIENT_REPAIR_SUGGESTION
    )
    assert agent_failure._failure_suggestion(
        "repair-provider-failure",
        "repair invocation failure: antigravity/model: nonzero_exit (boom)",
        "Codex",
    ) == ""


def test_task_and_pr_followup_salvage_scopes_post_no_github_comment(tmp_path):
    config = make_config(tmp_path)
    runner = FakeRunner(
        post_agent_git_status=" M file.txt\n",
        post_agent_git_diff=(
            "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n"
        ),
    )
    runner._mark_agent_command_seen()

    for scope in (
        agent_failure.TASK_IMPLEMENTATION_SALVAGE_SCOPE,
        agent_failure.PR_FOLLOWUP_SALVAGE_SCOPE,
    ):
        diagnostic = agent_failure._capture_failed_run_salvage_diagnostic(
            runner=runner,
            config=config,
            agent_name="claude",
            salvage_context=SalvageContext(
                repo=config.repo,
                issue_number=56,
                scope=scope,
                agent="claude",
            ),
            operation_description="task implementation",
            failure_category="transient",
            failure_reason="agent failed",
            classification_text="",
            marker_description="<!-- AGENT_PR: <number> -->",
            result=None,
        )
        assert diagnostic.artifacts is not None
        assert "No GitHub salvage comment was posted." in diagnostic.line

    assert runner.comments == []
    assert runner.issue_comments == []



@pytest.mark.parametrize("scope", ["issue-implementation", "approved-plan-implementation", "task-implementation", "pr-followup", None])
@pytest.mark.parametrize("legacy, reported", [
    (None, True),
    ("## Blocking\n\nCapture timed out.\n\nI opened PR #77.", True),
    ("Capture timed out.\n\nI opened https://github.com/OWNER/REPO/pull/77.", True),
    ("Capture timed out. PR #77.\n\nI also opened PR #78.", False),
    ("Capture timed out.\n\n> Earlier PR #78\n\n```text\nPR #79\n```\n\nI opened PR #77.", True),
    ("Capture timed out.\n\n~~~text\nPR #79\n~~~\n\nI opened PR #77.", True),
])
def test_failed_run_retains_timeout_and_advisory_pr(tmp_path, scope, legacy, reported):
    from coding_review_agent_loop.agents.base import AgentResult
    from agent_loop_helpers import structured_issue_implementation
    config = make_config(tmp_path)
    result = AgentResult(text=structured_issue_implementation(
        pr_number=None, summary="Capture timed out; reported PR https://github.com/OWNER/REPO/pull/77."))
    if legacy is not None:
        result = AgentResult(text=legacy + "\n<!-- AGENT_STATE: blocking -->")
    diagnostics = agent_failure._failed_run_diagnostics(
        runner=FakeRunner(), config=config, agent_name="Codex",
        salvage_context=(SalvageContext(repo=config.repo, issue_number=56, scope=scope, agent="codex")
                         if scope is not None else None),
        operation_description="issue implementation", failure_category="timeout",
        failure_reason="watchdog expired", classification_text="", marker_description="implementation result",
        result=result)
    message = agent_failure._format_invalid_agent_response_error(
        agent_name="Codex", marker_description="implementation result", reason="watchdog expired",
        result=result, log_paths=[], category="timeout", config=config) + diagnostics.response_line
    assert "Failure category: timeout" in message
    assert "watchdog expired" in message
    expected = scope in {"issue-implementation", "approved-plan-implementation"}
    assert message.count("Coder terminal context") == int(expected)
    assert ("Reported PR (unverified): https://github.com/OWNER/REPO/pull/77" in message) == (expected and reported)
    if expected:
        assert "Capture timed out" in message


@pytest.mark.parametrize("kind", ["pr_review", "plan_review", "coder_followup"])
def test_other_structured_roles_never_become_legacy_coder_context(kind):
    import json
    from coding_review_agent_loop.terminal_diagnostics import terminal_response_diagnostic
    text = json.dumps({"kind": kind, "summary": "Blocked on PR #77"})
    text += "\n<!-- AGENT_STATE: blocking -->\n-- Reviewer"
    assert terminal_response_diagnostic(text, repo="OWNER/REPO") == ""


def test_success_state_alone_does_not_establish_terminal_diagnostic():
    from coding_review_agent_loop.terminal_diagnostics import terminal_response_diagnostic
    from agent_loop_helpers import structured_issue_implementation
    assert terminal_response_diagnostic(
        structured_issue_implementation(pr_number=77, summary="Implemented the plan."),
        repo="OWNER/REPO") == ""


def test_terminal_diagnostic_does_not_scan_quoted_history():
    from coding_review_agent_loop.terminal_diagnostics import terminal_response_diagnostic
    text = ("Cannot proceed.\n\n> Previously opened PR https://github.com/OWNER/REPO/pull/77\n"
            "<!-- AGENT_STATE: blocking -->")
    message = terminal_response_diagnostic(text, repo="OWNER/REPO")
    assert "Cannot proceed" in message
    assert "Reported PR" not in message


@pytest.mark.parametrize("parsed", [False, True])
@pytest.mark.parametrize("number,current,reported", [(None, "", False), (None, "PR #77", True), (77, "", True), (77, "PR #77", True)])
@pytest.mark.parametrize("history", ["> Earlier PR #79", "```text\nPR #79\n```", "~~~text\nPR #79\n~~~", "<!-- Earlier PR #79 -->"])
def test_structured_terminal_filters_history_preserving_typed_identity(parsed, number, current, reported, history):
    from types import SimpleNamespace
    from agent_loop_helpers import structured_issue_implementation
    from coding_review_agent_loop.terminal_diagnostics import terminal_response_diagnostic
    summary = f"Blocked: capture timed out. {current}\n\n{history}"
    message = terminal_response_diagnostic(
        structured_issue_implementation(pr_number=number, summary=summary), repo="OWNER/REPO",
        parsed=SimpleNamespace(summary=summary, pr_number=number) if parsed else None,
        terminal=True)
    assert "capture timed out" in message
    assert ("Reported PR (unverified): https://github.com/OWNER/REPO/pull/77" in message) == reported
    assert "Reported PR (unverified): https://github.com/OWNER/REPO/pull/79" not in message
