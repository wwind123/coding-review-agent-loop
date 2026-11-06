"""Config and CLI surface of the PR non-convergence stop (#1372)."""

from __future__ import annotations

import pytest

from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import config_from_args
from coding_review_agent_loop.errors import AgentLoopError

from agent_loop_helpers import FakeRunner, make_config


def _base(tmp_path):
    return [
        "pr", "77", "--repo", "OWNER/REPO", "--reviewer", "codex",
        "--codex-dir", str(tmp_path / "codex"), "--dangerous-agent-permissions",
    ]


def test_non_convergence_rounds_flag_defaults_validates_and_parses(tmp_path):
    """`nc-flag-independence` (config half)."""
    assert make_config(tmp_path).pr_non_convergence_rounds == 3
    assert make_config(tmp_path, pr_non_convergence_rounds=0).pr_non_convergence_rounds == 0
    for invalid in (-1, True, False, 1.5, "2"):
        with pytest.raises(
            AgentLoopError, match=r"--pr-non-convergence-rounds must be a non-negative integer \(0 disables it\)\."
        ):
            make_config(tmp_path, pr_non_convergence_rounds=invalid)
    parser = build_parser()
    base = _base(tmp_path)
    assert parser.parse_args(base).pr_non_convergence_rounds is None
    assert config_from_args(parser.parse_args(base), FakeRunner()).pr_non_convergence_rounds == 3
    parsed = parser.parse_args([*base, "--pr-non-convergence-rounds", "5"])
    assert config_from_args(parsed, FakeRunner()).pr_non_convergence_rounds == 5
    with pytest.raises(AgentLoopError, match="--pr-non-convergence-rounds must be"):
        config_from_args(
            parser.parse_args([*base, "--pr-non-convergence-rounds", "-1"]), FakeRunner()
        )


def test_the_two_pr_gates_have_independent_flags(tmp_path):
    parser = build_parser()
    base = _base(tmp_path)
    step_back_off = config_from_args(
        parser.parse_args([*base, "--pr-step-back-rounds", "0"]), FakeRunner()
    )
    assert step_back_off.pr_step_back_rounds == 0
    assert step_back_off.pr_non_convergence_rounds == 3
    non_convergence_off = config_from_args(
        parser.parse_args([*base, "--pr-non-convergence-rounds", "0"]), FakeRunner()
    )
    assert non_convergence_off.pr_non_convergence_rounds == 0
    assert non_convergence_off.pr_step_back_rounds == 3


def test_managed_ci_recovery_parses_the_flag_value():
    from coding_review_agent_loop.managed_ci_recovery_commands import (
        _RECOVERY_VALUE_OPTIONS,
    )

    assert "--pr-non-convergence-rounds" in _RECOVERY_VALUE_OPTIONS
