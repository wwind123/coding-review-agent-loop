"""Finding-provenance baseline budget flag (#1516): validation and copy boundaries."""

from __future__ import annotations

import re
import shutil

import pytest

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.review_step_back import DEFAULT_PROVENANCE_BASELINE_MAX_CHARS
from agent_loop_helpers import FakeRunner as _FakeRunner

FLAG = "--plan-provenance-baseline-max-chars"


def _staged_values(**overrides):
    values = dict(
        reviewer=("codex", "gemini"),
        plan_review_policy="primary-then-panel",
        primary_plan_reviewer="codex",
    )
    values.update(overrides)
    return values


def test_default_budget_is_120000_and_valid_under_every_policy(tmp_path):
    assert DEFAULT_PROVENANCE_BASELINE_MAX_CHARS == 120000
    assert make_config(tmp_path, reviewer=("codex",)).plan_provenance_baseline_max_chars == 120000
    staged = make_config(tmp_path, **_staged_values(plan_provenance_baseline_max_chars=0))
    assert staged.plan_provenance_baseline_max_chars == 0


@pytest.mark.parametrize(
    ("value", "staged", "message"),
    [
        (-1, True, f"{FLAG} must be a non-negative integer"),
        (True, True, f"{FLAG} must be a non-negative integer"),
        ("1000", True, f"{FLAG} must be a non-negative integer"),
        (1.5, True, f"{FLAG} must be a non-negative integer"),
        (50000, False, f"{FLAG} requires --plan-review-policy primary-then-panel"),
        (0, False, f"{FLAG} requires --plan-review-policy primary-then-panel"),
    ],
)
def test_invalid_budget_is_rejected_at_config_time(tmp_path, value, staged, message):
    """`flag-validation-inheritance`: rejected before any agent turn can start."""
    values = (
        _staged_values(plan_provenance_baseline_max_chars=value)
        if staged
        else {"reviewer": ("codex", "gemini"), "plan_provenance_baseline_max_chars": value}
    )
    with pytest.raises(AgentLoopError, match=re.escape(message)):
        make_config(tmp_path, **values)


def test_budget_flag_reaches_the_config_from_the_cli(tmp_path):
    from coding_review_agent_loop.cli import build_parser
    from coding_review_agent_loop.config import config_from_args

    base_argv = [
        "issue", "56", "--repo", "OWNER/REPO", "--plan-first",
        "--reviewer", "codex", "--reviewer", "gemini",
        "--codex-dir", str(tmp_path / "codex"), "--gemini-dir", str(tmp_path / "gemini"),
        "--dangerous-agent-permissions",
        "--plan-review-policy", "primary-then-panel", "--primary-plan-reviewer", "codex",
    ]
    parser = build_parser()
    defaults = config_from_args(parser.parse_args(base_argv), FakeRunner())
    assert defaults.plan_provenance_baseline_max_chars == DEFAULT_PROVENANCE_BASELINE_MAX_CHARS
    config = config_from_args(parser.parse_args([*base_argv, FLAG, "60000"]), FakeRunner())
    assert config.plan_provenance_baseline_max_chars == 60000
    with pytest.raises(AgentLoopError, match=f"{FLAG} must be"):
        config_from_args(parser.parse_args([*base_argv, FLAG, "-5"]), FakeRunner())


def test_budget_flag_is_a_recovery_value_option():
    from coding_review_agent_loop.managed_ci import _RECOVERY_VALUE_OPTIONS

    assert FLAG in _RECOVERY_VALUE_OPTIONS


def _approving_context(policy: str):
    from coding_review_agent_loop.plan_verification import PlanVerificationContext
    from coding_review_agent_loop.reviewer_floor import BoardSeat

    seats = (
        BoardSeat(seat_id="Codex", backend="codex"),
        BoardSeat(seat_id="Gemini", backend="gemini"),
    )
    return PlanVerificationContext(
        issue_number=56,
        original_reviewers=("Codex", "Gemini"),
        effective_reviewers=("Codex", "Gemini"),
        original_seats=seats,
        effective_seats=seats,
        policy=policy,
        primary_reviewer="Codex" if policy == "primary-then-panel" else None,
        seat_binding=None,
        signed_amendment_count=0,
    )


def test_child_planning_inherits_the_budget_only_when_staged(tmp_path):
    """`flag-validation-inheritance`: staged passes through, otherwise the default."""
    from coding_review_agent_loop.semantic_dedupe import _isolated_provider_config

    config = make_config(tmp_path, **_staged_values(plan_provenance_baseline_max_chars=50000))
    staged = _approving_context("primary-then-panel").config_for(config)
    assert staged.plan_provenance_baseline_max_chars == 50000
    full_board = _approving_context("all-reviewers").config_for(config)
    assert full_board.plan_provenance_baseline_max_chars == DEFAULT_PROVENANCE_BASELINE_MAX_CHARS
    isolated, isolated_dir = _isolated_provider_config(config, "claude", "")
    try:
        assert isolated.plan_provenance_baseline_max_chars == DEFAULT_PROVENANCE_BASELINE_MAX_CHARS
    finally:
        shutil.rmtree(isolated_dir, ignore_errors=True)


def test_approved_all_reviewers_board_switch_resets_and_restores_the_budget(tmp_path):
    """`flag-validation-inheritance`: the approved-plan board copy boundary (#1377)."""
    from coding_review_agent_loop.plan_first_loop import (
        _PLAN_BOARD_CONFIG_FIELDS,
        _approving_plan_board_config,
        _restore_invocation_board,
    )

    assert "plan_provenance_baseline_max_chars" in _PLAN_BOARD_CONFIG_FIELDS
    invocation = make_config(
        tmp_path, **_staged_values(plan_provenance_baseline_max_chars=50000)
    )
    # Switching to a completely approved all-reviewers plan board raises no
    # config validation error: the reset budget travels with the policy.
    approving = _approving_plan_board_config(invocation, _approving_context("all-reviewers"))
    assert approving.plan_review_policy == "all-reviewers"
    assert approving.plan_provenance_baseline_max_chars == DEFAULT_PROVENANCE_BASELINE_MAX_CHARS
    restored = _restore_invocation_board(approving, invocation)
    assert restored.plan_review_policy == "primary-then-panel"
    assert restored.plan_provenance_baseline_max_chars == 50000


def test_narrow_staged_flag_is_validated(tmp_path):
    """`narrow-flag-validation`."""
    with pytest.raises(AgentLoopError, match="--plan-narrow-staged"):
        make_config(tmp_path, plan_narrow_staged=True, plan_execution_mode="decompose-only")
    make_config(tmp_path, plan_narrow_staged=True, plan_execution_mode="plan-only")
    make_config(tmp_path, plan_narrow_staged=True, plan_execution_mode="implement-one-shot")
    assert main_exit_for(["issue", "56", "--repo", "OWNER/REPO", "--plan-narrow-staged"]) == 1


def main_exit_for(argv):
    from coding_review_agent_loop.cli import main

    return main(argv)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"plan_sweep_rounds": 1}, "--plan-sweep-rounds requires --plan-review-policy primary-then-panel"),
        ({"plan_sweep_reasoning_effort": "high"}, "--plan-sweep-reasoning-effort requires"),
        ({"plan_sweep_rounds": -1}, "--plan-sweep-rounds must be a non-negative integer"),
        ({"plan_sweep_rounds": True}, "--plan-sweep-rounds must be a non-negative integer"),
        ({"plan_sweep_reasoning_effort": "turbo"}, "--plan-sweep-reasoning-effort value 'turbo'"),
    ],
)
def test_sweep_flags_are_validated_at_config_time(tmp_path, overrides, message):
    """`disabled-and-policy`, `sweep-effort-turn` (config unit)."""
    values = {"reviewer": ("codex", "gemini")}
    if "requires" not in message:
        values.update(plan_review_policy="primary-then-panel", primary_plan_reviewer="codex")
    values.update(overrides)
    with pytest.raises(AgentLoopError, match=re.escape(message)):
        make_config(tmp_path, **values)


def test_sweep_flags_reach_the_config_from_the_cli(tmp_path):
    from coding_review_agent_loop.cli import build_parser
    from coding_review_agent_loop.config import config_from_args

    base_argv = [
        "issue", "56", "--repo", "OWNER/REPO", "--plan-first",
        "--reviewer", "codex", "--reviewer", "gemini",
        "--codex-dir", str(tmp_path / "codex"), "--gemini-dir", str(tmp_path / "gemini"),
        "--dangerous-agent-permissions",
        "--plan-review-policy", "primary-then-panel", "--primary-plan-reviewer", "codex",
    ]
    parser = build_parser()
    defaults = config_from_args(parser.parse_args(base_argv), _FakeRunner())
    assert (defaults.plan_sweep_rounds, defaults.plan_sweep_reasoning_effort) == (2, "")
    config = config_from_args(
        parser.parse_args(
            [*base_argv, "--plan-sweep-rounds", "1", "--plan-sweep-reasoning-effort", "high"]
        ),
        _FakeRunner(),
    )
    assert (config.plan_sweep_rounds, config.plan_sweep_reasoning_effort) == (1, "high")
    with pytest.raises(AgentLoopError, match="--plan-sweep-rounds must be"):
        config_from_args(parser.parse_args([*base_argv, "--plan-sweep-rounds", "-1"]), _FakeRunner())
