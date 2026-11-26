"""CLI and config surface of the single no-progress detector (#2097, #2098)."""

from __future__ import annotations

import dataclasses

import pytest

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import (
    DEFAULT_NO_PROGRESS_ROUNDS,
    config_from_args,
)
from coding_review_agent_loop.errors import AgentLoopError

# Retired by the single no-progress detector (#2097, #2098) and removed from
# the parser in #2162: passing one is a startup error naming the replacement.
PLAN_RETIRED_FLAGS = (
    "--plan-step-back-rounds",
    "--plan-step-back-escalation-rounds",
    "--plan-primary-stall-rounds",
    "--plan-sweep-rounds",
)
RETIRED_FLAGS = (
    "--pr-step-back-rounds",
    "--pr-step-back-line-window",
    "--pr-non-convergence-rounds",
    "--sub-item-stall-rounds",
    *PLAN_RETIRED_FLAGS,
)
RETIRED_DESTS = {flag.lstrip("-").replace("-", "_") for flag in RETIRED_FLAGS}


def _argv(tmp_path, *extra):
    return [
        "pr", "77", "--repo", "OWNER/REPO",
        "--reviewer", "codex", "--codex-dir", str(tmp_path / "codex"),
        "--claude-dir", str(tmp_path / "claude"),
        "--dangerous-agent-permissions",
        *extra,
    ]


def _config(tmp_path, *extra):
    args = build_parser().parse_args(_argv(tmp_path, *extra))
    return args, config_from_args(args, FakeRunner())


def test_no_progress_rounds_defaults_to_three(tmp_path, capsys):
    args, config = _config(tmp_path)
    assert args.no_progress_rounds is None
    assert config.no_progress_rounds == DEFAULT_NO_PROGRESS_ROUNDS == 3
    assert "deprecated" not in capsys.readouterr().err


def test_no_progress_rounds_parses_a_custom_window(tmp_path):
    _args, config = _config(tmp_path, "--no-progress-rounds", "5")
    assert config.no_progress_rounds == 5
    _args, config = _config(tmp_path, "--no-progress-rounds", "1")
    assert config.no_progress_rounds == 1


@pytest.mark.parametrize("value", ["0", "-1"])
def test_no_progress_rounds_below_one_is_rejected(tmp_path, value):
    """The detector cannot be turned off; --max-rounds stays the outer bound."""
    args = build_parser().parse_args(_argv(tmp_path, "--no-progress-rounds", value))
    with pytest.raises(AgentLoopError, match="--no-progress-rounds must be an integer of at least 1"):
        config_from_args(args, FakeRunner())


@pytest.mark.parametrize("invalid", [0, -2, True, 1.5, "3"])
def test_direct_config_rejects_an_invalid_window(tmp_path, invalid):
    with pytest.raises(AgentLoopError, match="--no-progress-rounds must be"):
        make_config(tmp_path, no_progress_rounds=invalid)


@pytest.mark.parametrize("flag", RETIRED_FLAGS)
@pytest.mark.parametrize("value", ["0", "3", "17"])
def test_each_retired_flag_is_a_startup_error_naming_the_window(tmp_path, capsys, flag, value):
    """#2162: the one-release no-op window is over; nothing is silently ignored."""
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(_argv(tmp_path, flag, value))
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert f"{flag} was removed (#2162)" in err
    assert "--no-progress-rounds" in err


def test_retired_flags_never_reach_the_config():
    from coding_review_agent_loop.config import AgentLoopConfig

    names = {field.name for field in dataclasses.fields(AgentLoopConfig)}
    assert len(RETIRED_DESTS) == len(RETIRED_FLAGS)
    assert not RETIRED_DESTS & names


def test_retired_flags_are_hidden_from_help(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["pr", "--help"])
    out = "".join(capsys.readouterr().out.split())  # argparse wraps at hyphens
    assert "--no-progress-roundsKStopforahumandecision(exit4)" in out
    for flag in RETIRED_FLAGS:
        assert flag not in out


def _issue_argv(tmp_path, *extra):
    return [
        "issue", "56", "--repo", "OWNER/REPO", "--plan-first",
        "--reviewer", "codex", "--reviewer", "gemini",
        "--codex-dir", str(tmp_path / "codex"), "--gemini-dir", str(tmp_path / "gemini"),
        "--claude-dir", str(tmp_path / "claude"),
        "--dangerous-agent-permissions",
        *extra,
    ]


@pytest.mark.parametrize("policy", ["all-reviewers", "primary-then-panel"])
@pytest.mark.parametrize("flag", PLAN_RETIRED_FLAGS)
def test_each_retired_plan_flag_is_rejected_under_either_plan_policy(tmp_path, capsys, policy, flag):
    policy_args = ["--plan-review-policy", policy]
    if policy == "primary-then-panel":
        policy_args += ["--primary-plan-reviewer", "codex"]
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(_issue_argv(tmp_path, *policy_args, flag, "1"))
    assert raised.value.code == 2
    assert "--no-progress-rounds" in capsys.readouterr().err


def test_retired_plan_fields_never_reach_a_child_or_verification_config(tmp_path):
    from coding_review_agent_loop import plan_execution

    dests = {flag.lstrip("-").replace("-", "_") for flag in PLAN_RETIRED_FLAGS}
    assert len(dests) == 4
    assert not dests & set(plan_execution._PLAN_BOARD_CONFIG_FIELDS)
    with pytest.raises(TypeError):
        make_config(tmp_path, plan_primary_stall_rounds=3)


def test_the_reset_flag_help_names_the_no_progress_window(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["issue", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "restart the plan no-progress window" in out
