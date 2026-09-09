"""Independent reviewer selection must survive the implementation handoff."""

from dataclasses import replace
import shlex

import pytest

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop.agents.registry import agent_signature, run_agent_result
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import config_from_args, resolve_invocation
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.managed_ci import _render_recovery_command
from coding_review_agent_loop.orchestrator import (
    _approved_implementation_config,
    _build_unsupported_model_diagnostic,
)
from coding_review_agent_loop.prompts import (
    build_discuss_review_prompt,
    build_plan_review_prompt,
    build_review_prompt,
)


@pytest.mark.parametrize("command", ["issue", "pr", "discuss"])
def test_reviewer_options_reach_config(tmp_path, command):
    args = build_parser().parse_args([
        command, "123", "--repo", "OWNER/REPO",
        "--coder", "codex", "--reviewer", "claude", "--reviewer", "codex",
        "--codex-dir", str(tmp_path / "codex"),
        "--claude-dir", str(tmp_path / "claude"),
        "--subprocess-log-dir", str(tmp_path / "logs"),
        "--reviewer-codex-model", "gpt-5.6-sol",
        "--reviewer-codex-reasoning-effort", "medium",
        "--reviewer-claude-model", "claude-review-model",
        "--reviewer-claude-effort", "high",
    ])
    config = config_from_args(args, FakeRunner())
    assert config.reviewer_codex_model == "gpt-5.6-sol"
    assert config.reviewer_codex_reasoning_effort == "medium"
    assert config.reviewer_claude_model == "claude-review-model"
    assert config.reviewer_claude_effort == "high"


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("handoff", [False, True])
def test_actual_backend_uses_independent_model_effort_and_identity(tmp_path, provider, handoff):
    effort = "codex_reasoning_effort" if provider == "codex" else "claude_effort"
    config = make_config(
        tmp_path, coder=provider, reviewer=provider,
        **{f"{provider}_model": "planner-model" if handoff else "implementation-model",
           effort: "medium" if handoff else "xhigh",
           f"reviewer_{provider}_model": "review-model",
           f"reviewer_{effort}": "high"},
    )
    if handoff:
        config = replace(config, implementation_coder=provider,
                         implementation_coder_model="implementation-model",
                         **{f"implementation_{effort}": "xhigh"})
        config, reuse = _approved_implementation_config(config)
        assert not reuse

    runner = FakeRunner(**{f"{provider}_outputs": [("ok", 0)] * 3})
    for role, model, level in [
        ("coder", "implementation-model", "xhigh"),
        ("reviewer", "review-model", "high"),
        ("coder", "implementation-model", "xhigh"),
    ]:
        result = run_agent_result(runner, agent=provider, config=config, prompt="Test", role=role)
        command = [cmd for cmd, _cwd in runner.commands if cmd[0] == provider][-1]
        assert command[command.index("--model") + 1] == model
        if provider == "codex":
            assert f'model_reasoning_effort="{level}"' in command
        else:
            assert command[command.index("--effort") + 1] == level
        assert result.configured_model == model
        assert result.configured_effort == level
        assert result.role == role
        assert result.model_used == f"{model} ({level})"
        assert agent_signature(provider, config, role=role).endswith(result.model_used)
        if role == "reviewer":
            assert result.effort_source == "role_override"


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_model_and_effort_overrides_are_independent(tmp_path, provider):
    effort = "codex_reasoning_effort" if provider == "codex" else "claude_effort"
    base = make_config(tmp_path, **{f"{provider}_model": "base", effort: "high"})
    model_only = replace(base, **{f"reviewer_{provider}_model": "review"})
    effort_only = replace(base, **{f"reviewer_{effort}": "low"})
    for config, model, level, source in [
        (base, "base", "high", "agent_wide"),
        (model_only, "review", "high", "agent_wide"),
        (effort_only, "base", "low", "role_override"),
        (replace(model_only, **{effort: ""}), "review", "medium", "tool_default"),
    ]:
        invocation = resolve_invocation(config, provider=provider, role="reviewer")
        assert (invocation.configured_model, invocation.resolved_effort, invocation.effort_source) == (
            model, level, source,
        )
    for role in (None, "coder", "analyzer", "repair"):
        invocation = resolve_invocation(model_only, provider=provider, role=role)
        assert invocation.configured_model == "base"
        assert resolve_invocation(effort_only, provider=provider, role=role).resolved_effort == "high"


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("compact", [False, True])
def test_review_prompts_use_reviewer_signature(tmp_path, provider, compact):
    effort = "codex_reasoning_effort" if provider == "codex" else "claude_effort"
    config = make_config(tmp_path, coder=provider, reviewer=provider,
                         **{f"{provider}_model": "coder-model", effort: "xhigh",
                            f"reviewer_{provider}_model": "review-model",
                            f"reviewer_{effort}": "medium"})
    signature = "-- " + agent_signature(provider, config, role="reviewer")
    assert signature.endswith("review-model (medium)")
    prompts = [
        build_plan_review_prompt(1, 1, "Plan", config, reviewer=provider, compact_context=compact),
        build_review_prompt(2, 1, config, reviewer=provider, compact_context=compact),
        build_discuss_review_prompt(1, config, reviewer=provider),
    ]
    for prompt in prompts:
        assert signature in prompt
        assert "-- " + agent_signature(provider, config, role="coder") not in prompt


@pytest.mark.parametrize("option", ["reviewer_codex_reasoning_effort", "reviewer_claude_effort"])
def test_invalid_reviewer_effort_is_rejected(tmp_path, option):
    with pytest.raises(AgentLoopError, match=option.replace("_", "-")):
        make_config(tmp_path, **{option: "invalid"})


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_reviewer_model_rejects_raw_model_argument_and_blank_value(tmp_path, provider):
    with pytest.raises(AgentLoopError, match=f"reviewer-{provider}-model"):
        make_config(tmp_path, **{f"reviewer_{provider}_model": "review",
                                 f"{provider}_args": ("--model", "other")})
    with pytest.raises(AgentLoopError, match="must not be blank"):
        make_config(tmp_path, **{f"reviewer_{provider}_model": "   "})


def test_observed_model_still_wins_signature(tmp_path):
    config = make_config(tmp_path, reviewer_codex_model="requested")
    assert agent_signature("codex", config, model_used="observed (high)", role="reviewer").endswith(
        "observed (high)"
    )


@pytest.mark.parametrize("target", ["issue", "pr"])
def test_recovery_commands_preserve_reviewer_overrides(tmp_path, target):
    config = make_config(tmp_path, invocation_argv=(
        "agent-loop", "issue", "--reviewer-codex-model", "gpt-5.6-sol",
        "--reviewer-codex-reasoning-effort", "medium",
        "--reviewer-claude-model=claude-review-model", "--reviewer-claude-effort", "high",
        "123", "--repo", "OWNER/REPO",
    ))
    command = _render_recovery_command(config, target=target, identifier=456, managed_ci=False)
    args = build_parser().parse_args(shlex.split(command)[1:])
    assert args.command == target
    assert getattr(args, f"{target}_number") == 456
    assert args.reviewer_codex_model == "gpt-5.6-sol"
    assert args.reviewer_codex_reasoning_effort == "medium"
    assert args.reviewer_claude_model == "claude-review-model"
    assert args.reviewer_claude_effort == "high"


def test_unsupported_reviewer_model_diagnostic_names_the_reviewer_option(tmp_path):
    config = make_config(tmp_path, codex_model="coder-model", reviewer_codex_model="gpt-5.5-pro")
    diagnostic = _build_unsupported_model_diagnostic(
        agent="codex", agent_name="Codex", config=config, role="reviewer", result=None,
        classification_text="The gpt-5.5-pro model is not supported when using Codex with a ChatGPT account.",
    )
    assert diagnostic.requested_model == "gpt-5.5-pro"
    assert diagnostic.fallback_flag == "--reviewer-codex-model"
    assert diagnostic.fallback_value == "gpt-5.5"
