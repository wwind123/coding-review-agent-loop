"""issue_implementation tests: approved implementation role configuration (#1544)."""

import pytest

from agent_loop_helpers import (
    FakeRunner,
    make_config,
)


@pytest.mark.parametrize(
    ("provider", "effort_option", "agent_effort_option"),
    [
        ("codex", "implementation_codex_reasoning_effort", "codex_reasoning_effort"),
        ("claude", "implementation_claude_effort", "claude_effort"),
    ],
)
def test_approved_implementation_role_override_reaches_real_backend_turn(
    tmp_path, provider, effort_option, agent_effort_option
):
    from coding_review_agent_loop.agents.registry import run_agent_result
    from coding_review_agent_loop.issue_implementation import _approved_implementation_config

    config = make_config(
        tmp_path,
        coder=provider,
        implementation_coder=provider,
        **{agent_effort_option: "medium"},
        **{effort_option: "high"},
    )
    implementation_config, reuse_session = _approved_implementation_config(config)
    assert not reuse_session

    runner = FakeRunner(**{f"{provider}_outputs": [("ok", 0), ("ok", 0)]})
    implementation_result = run_agent_result(
        runner,
        agent=provider,
        config=implementation_config,
        prompt="Implement the approved plan.",
        role="coder",
    )
    reviewer_result = run_agent_result(
        runner,
        agent=provider,
        config=implementation_config,
        prompt="Review the implementation.",
        role="reviewer",
    )

    assert (implementation_result.configured_effort, implementation_result.effort_source) == (
        "high",
        "role_override",
    )
    assert (reviewer_result.configured_effort, reviewer_result.effort_source) == (
        "medium",
        "agent_wide",
    )
    commands = [command for command, _cwd in runner.commands if command[0] == provider]
    if provider == "codex":
        assert 'model_reasoning_effort="high"' in commands[0]
        assert 'model_reasoning_effort="medium"' in commands[1]
    else:
        assert commands[0][commands[0].index("--effort") + 1] == "high"
        assert commands[1][commands[1].index("--effort") + 1] == "medium"
