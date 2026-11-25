"""Plan-first loop tests: plan validation (#1543)."""

import hashlib
import json

import pytest

from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.errors import AgentInvocationError
from issue_loop_test_helpers import _PlanDiagnosticRunner, _TRANSIENT_REPAIR_SUGGESTION
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_state,
    structured_v1_plan_state,
)


def test_exhausted_plan_validation_persists_from_the_planning_orchestration_path(
    tmp_path, monkeypatch
):
    invalid_payload = json.loads(structured_plan_state().split("\n", 1)[0])
    invalid_payload.pop("architecture_impact")
    invalid_candidate = (
        json.dumps(invalid_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [invalid_candidate]
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert error.value.plan_validation_exhaustion is not None
    assert len(runner.diagnostic_posts) == 1
    assert "not persisted" not in str(error.value)


# --- Issue #871: a reviewer turn with no review is never turned into one ---

_NARRATION_ONLY_PLAN_REVIEW = (
    "I have launched the test command for tests/test_test_runtime.py in the "
    "background and will wait for it to complete.\n"
    "root agent idle; waiting up to 5s for 1 background task(s)\n"
    "terminating 1 background task(s) on exit"
)


def test_narration_only_plan_reviewer_is_unavailable_and_posts_no_verdict(tmp_path):
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[_NARRATION_ONLY_PLAN_REVIEW] * 4,
    )

    config = make_config(tmp_path, agent_max_retries=1, agent_retry_backoff_seconds=0)
    with pytest.raises(AgentLoopError, match="no review substance") as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert "agent-unavailable" in str(excinfo.value)
    # No fabricated verdict is posted and no ledger item is numbered.
    assert not any("Plan review incomplete" in body for body in runner.comments)
    assert not any("[item-1]" in body for body in runner.comments)
    # The reviewer is retried within the configured policy before the run stops.
    codex_turns = [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(codex_turns) >= 2


def test_empty_plan_reviewer_response_is_refused(tmp_path):
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=["", "", "", ""],
    )

    config = make_config(tmp_path, agent_max_retries=0, agent_retry_backoff_seconds=0)
    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert not any("Plan review incomplete" in body for body in runner.comments)


def test_refused_plan_reviewer_leaves_nothing_for_a_resume_to_replay(tmp_path):
    """A refused reviewer must not leave a checkpoint a rerun could replay."""
    config = make_config(tmp_path, agent_max_retries=0, agent_retry_backoff_seconds=0)

    def run_once():
        runner = _FakeRunner(
            claude_outputs=[structured_v1_plan_state()],
            codex_outputs=[_NARRATION_ONLY_PLAN_REVIEW] * 4,
        )
        with pytest.raises(AgentLoopError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
        return runner

    first = run_once()
    second = run_once()
    for runner in (first, second):
        assert not any("Plan review incomplete" in body for body in runner.comments)
        assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
