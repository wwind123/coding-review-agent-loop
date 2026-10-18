"""plan_first_loop tests: plan review and revision turn handling and plan-comment posting (#1544)."""

import pytest

from coding_review_agent_loop.cli import (
    AgentLoopError,
    run_issue_loop,
)
from coding_review_agent_loop.agent_failure import (
    ValidatedAgentResponse,
)
from coding_review_agent_loop.discuss_loop import ROUND_RESUME_MARKER_RE
from coding_review_agent_loop.round_state import _decode_round_metadata, _strip_round_metadata
from coding_review_agent_loop.protocol import (
    parse_plan_review,
)
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_plan_review,
    structured_plan_state,
)
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.triage as triage
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_object_in


def test_plan_review_does_not_post_diagnostics_without_plan_state(tmp_path):
    diagnostic = "[ERROR] Invalid stream: The model returned an empty response or malformed tool call."
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Update the CLI.", plan_steps=["Update the CLI."]),
        ],
        gemini_outputs=[diagnostic, diagnostic, diagnostic],
    )
    config = make_config(tmp_path, reviewer="gemini")

    with pytest.raises(AgentLoopError, match="AGENT_PLAN_STATE"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert len(runner.comments) == 1
    assert runner.comments[0].startswith("## Plan")
    assert not any(diagnostic in comment for comment in runner.comments)


def test_plan_loop_retries_plain_agent_plan_state_near_miss_once(tmp_path):
    near_miss = "Plan looks sound.\nAGENT_PLAN_STATE: approved.\n-- Google Gemini"
    valid = "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini"
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Update the CLI.", plan_steps=["Update the CLI."]),
        ],
        gemini_outputs=[near_miss, valid],
    )
    config = make_config(tmp_path, reviewer="gemini")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert near_miss not in runner.comments
    assert any(comment == f"**Review verdict:** Approved\n\n{valid}" for comment in runner.comments)
    sleep_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]
    assert sleep_commands == [["sleep", "1"]]


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
    with patch_object_in((validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage, issue_loop), "_run_validated_agent",
        side_effect=fake_run_validated_agent,
    ):
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    posted_body = runner.issue_comments[0]["body"]
    stripped = _strip_round_metadata(posted_body)
    assert stripped.endswith("-- Anthropic Claude: gpt-5.5 (medium)")
    match = ROUND_RESUME_MARKER_RE.search(posted_body)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.canonical_plan == plan_text


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
    with patch_object_in((validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage, issue_loop), "_run_validated_agent",
        side_effect=fake_run_validated_agent,
    ):
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    # Second issue_comments entry is the plan revision (index 1: reviewer, index 2: revision)
    revision_body = runner.issue_comments[2]["body"]
    stripped = _strip_round_metadata(revision_body)
    assert stripped.endswith("-- Anthropic Claude: gpt-5.5 (medium)")
    match = ROUND_RESUME_MARKER_RE.search(revision_body)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.canonical_plan == revised_plan
