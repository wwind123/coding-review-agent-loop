"""CLI and config surface of the single PR no-progress detector (#2097)."""

from __future__ import annotations

import dataclasses

import pytest

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import (
    DEFAULT_NO_PROGRESS_ROUNDS,
    DEPRECATED_CONVERGENCE_FLAG_DESTS,
    config_from_args,
    deprecated_convergence_flags,
)
from coding_review_agent_loop.errors import AgentLoopError

DEPRECATED_FLAGS = (
    "--pr-step-back-rounds",
    "--pr-step-back-line-window",
    "--pr-non-convergence-rounds",
    "--sub-item-stall-rounds",
)


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


def _comparable(config):
    """Every numeric or boolean threshold: none may depend on a retired flag.

    Per-run paths and tokens differ between any two invocations, so they are
    left out; the retired flags never had a path or string value.
    """
    return {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(config)
        if isinstance(getattr(config, field.name), (bool, int))
    }


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


@pytest.mark.parametrize("flag", DEPRECATED_FLAGS)
@pytest.mark.parametrize("value", ["0", "3", "17"])
def test_each_retired_flag_parses_is_ignored_and_logs_one_notice(tmp_path, capsys, flag, value):
    """`flags-deprecated`: any integer, including 0, parses and changes nothing."""
    _args, baseline = _config(tmp_path)
    capsys.readouterr()
    args, config = _config(tmp_path, flag, value)
    err = capsys.readouterr().err
    notices = [line for line in err.splitlines() if "deprecated flag(s) ignored" in line]
    assert len(notices) == 1
    assert flag in notices[0] and "--no-progress-rounds" in notices[0]
    assert deprecated_convergence_flags(args) == (flag,)
    assert _comparable(config) == _comparable(baseline)
    assert config.no_progress_rounds == DEFAULT_NO_PROGRESS_ROUNDS


def test_every_retired_flag_together_logs_one_notice_naming_each(tmp_path, capsys):
    extra = [part for flag in DEPRECATED_FLAGS for part in (flag, "0")]
    args, config = _config(tmp_path, *extra)
    err = capsys.readouterr().err
    (notice,) = [line for line in err.splitlines() if "deprecated flag(s) ignored" in line]
    for flag in DEPRECATED_FLAGS:
        assert flag in notice
    assert deprecated_convergence_flags(args) == DEPRECATED_FLAGS
    assert config.no_progress_rounds == DEFAULT_NO_PROGRESS_ROUNDS


def test_retired_flags_never_reach_the_config():
    from coding_review_agent_loop.config import AgentLoopConfig

    names = {field.name for field in dataclasses.fields(AgentLoopConfig)}
    assert tuple(flag for _dest, flag in DEPRECATED_CONVERGENCE_FLAG_DESTS) == DEPRECATED_FLAGS
    for dest, _flag in DEPRECATED_CONVERGENCE_FLAG_DESTS:
        assert dest not in names


def test_retired_flags_are_marked_deprecated_in_help(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["pr", "--help"])
    out = "".join(capsys.readouterr().out.split())  # argparse wraps at hyphens
    assert "--no-progress-roundsKStopforahumandecision(exit4)" in out
    for flag in DEPRECATED_FLAGS:
        assert f"{flag}NDeprecatedandignoredforonerelease;see--no-progress-rounds." in out
