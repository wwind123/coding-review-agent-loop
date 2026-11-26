"""Flags removed in #2162 after their one-release no-op window (#2065-#2067)."""

import re
from pathlib import Path

import pytest

from agent_loop_helpers import FakeRunner
from coding_review_agent_loop.cli import (
    RETIRED_COMMON_FLAGS,
    RETIRED_CONTAINMENT_FLAGS,
    RETIRED_FLAGS,
    RETIRED_PR_FLAGS,
    build_parser,
)
from coding_review_agent_loop.config import config_from_args


_STALL_FLAGS = (
    "--sub-item-stall-rounds",
    "--plan-primary-stall-rounds",
    "--plan-step-back-rounds",
    "--plan-sweep-rounds",
    "--plan-step-back-escalation-rounds",
    "--pr-step-back-rounds",
    "--pr-step-back-line-window",
    "--pr-non-convergence-rounds",
)
_REPAIR_FLAGS = (
    "--repair-backend",
    "--repair-model",
    "--repair-reasoning-effort",
    "--repair-timeout-seconds",
    "--containment-repair-memory-high",
    "--containment-repair-memory-max",
    "--containment-repair-memory-swap-max",
    "--containment-repair-tasks-max",
)

# The subcommand prefix each formerly accepting parser needs to reach option parsing.
_COMMON_PARSERS = {
    "issue": ["issue", "1"],
    "pr": ["pr", "1"],
    "managed-pr": ["managed-pr", "--head", "feature", "--title", "T"],
    "task": ["task", "do it"],
    "discuss": ["discuss", "1"],
}
_CONTAINMENT_ONLY_PARSERS = {
    "run-tests": ["run-tests"],
    "containment-preflight": ["containment-preflight"],
}


def _replacement_phrase(flag):
    if flag in _STALL_FLAGS:
        return "--no-progress-rounds"
    if flag in _REPAIR_FLAGS:
        return "no repair pass exists"
    assert flag == "--review-unrecorded-head"
    return "resumes as a review of the current head"


def _cases():
    for flag in RETIRED_COMMON_FLAGS + RETIRED_CONTAINMENT_FLAGS:
        for name, prefix in _COMMON_PARSERS.items():
            yield pytest.param(flag, prefix, id=f"{name}{flag}")
    for flag in RETIRED_CONTAINMENT_FLAGS:
        for name, prefix in _CONTAINMENT_ONLY_PARSERS.items():
            yield pytest.param(flag, prefix, id=f"{name}{flag}")
    for flag in RETIRED_PR_FLAGS:
        yield pytest.param(flag, _COMMON_PARSERS["pr"], id=f"pr{flag}")


def test_retired_table_covers_exactly_the_removed_flags():
    assert set(RETIRED_FLAGS) == {*_STALL_FLAGS, *_REPAIR_FLAGS, "--review-unrecorded-head"}
    assert set(RETIRED_COMMON_FLAGS + RETIRED_CONTAINMENT_FLAGS + RETIRED_PR_FLAGS) == set(RETIRED_FLAGS)


@pytest.mark.parametrize("form", ["bare", "valued", "equals"])
@pytest.mark.parametrize("flag,prefix", list(_cases()))
def test_retired_flag_is_rejected_with_its_replacement(capsys, flag, prefix, form):
    """`retired-flag-rejected`: exit 2, naming the flag and its replacement."""
    extra = {"bare": [flag], "valued": [flag, "3"], "equals": [f"{flag}=3"]}[form]
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args([*prefix, *extra])
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert f"{flag} was removed (#2162)" in err
    assert _replacement_phrase(flag) in err


@pytest.mark.parametrize("flag", sorted(RETIRED_FLAGS))
def test_retired_flag_prefix_is_not_matched_onto_a_live_option(capsys, flag):
    """A unique prefix of a removed name hits the retired entry, never a live option."""
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["pr", "1", flag[:-1]])
    assert raised.value.code == 2
    assert f"{flag} was removed (#2162)" in capsys.readouterr().err


def test_plan_sweep_rounds_is_not_abbreviated_to_the_live_effort_option(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["issue", "1", "--plan-sweep-rounds", "high"])
    assert "--plan-sweep-rounds was removed" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["issue", "pr", "managed-pr", "task", "discuss", "run-tests", "containment-preflight"])
def test_retired_flags_are_absent_from_help(capsys, command):
    with pytest.raises(SystemExit):
        build_parser().parse_args([command, "--help"])
    help_text = capsys.readouterr().out
    for flag in RETIRED_FLAGS:
        assert flag not in help_text
    assert "Deprecated no-op" not in help_text
    assert "ignored for one release" not in help_text


def test_live_neighbours_still_parse_into_config(tmp_path):
    """`live-neighbours-parse`: similar surviving options keep working."""
    args = build_parser().parse_args([
        "issue", "1", "--repo", "OWNER/REPO", "--plan-first",
        "--reviewer", "codex", "--reviewer", "gemini",
        "--codex-dir", str(tmp_path / "codex"), "--gemini-dir", str(tmp_path / "gemini"),
        "--claude-dir", str(tmp_path / "claude"),
        "--subprocess-log-dir", str(tmp_path / "logs"),
        "--dangerous-agent-permissions",
        "--plan-review-policy", "primary-then-panel", "--primary-plan-reviewer", "codex",
        "--plan-sweep-reasoning-effort", "high",
        "--pr-evidence-stall-rounds", "4",
        "--no-progress-rounds", "5",
        "--containment-coder-memory-high", "900M",
    ])
    assert "retired_flag_sink" not in vars(args)
    config = config_from_args(args, FakeRunner())
    assert config.plan_sweep_reasoning_effort == "high"
    assert config.pr_evidence_stall_rounds == 4
    assert config.no_progress_rounds == 5
    assert dict(config.containment_policy.roles)["coder"].memory_high == 900_000_000


def test_live_containment_role_options_still_parse_on_run_tests():
    args = build_parser().parse_args([
        "run-tests", "--containment-reviewer-tasks-max", "321",
        "--containment-coder-memory-max", "200M", "--", "pytest",
    ])
    assert args.containment_reviewer_tasks_max == "321"
    assert args.containment_coder_memory_max == "200M"


_SRC = Path(__file__).resolve().parents[1] / "src" / "coding_review_agent_loop"
_DEAD_IDENTIFIERS = re.compile(
    r"repair_backend|repair_model|repair_timeout_seconds|reask_on_|review_unrecorded_head"
)


def test_no_dead_repair_or_reask_identifier_remains_in_src():
    """`src-static-guard`: no file is exempt."""
    offenders = {
        path.relative_to(_SRC).as_posix(): sorted(set(_DEAD_IDENTIFIERS.findall(text)))
        for path in sorted(_SRC.rglob("*.py"))
        if _DEAD_IDENTIFIERS.search(text := path.read_text(encoding="utf-8"))
    }
    assert offenders == {}


def test_repair_role_and_entry_point_are_gone():
    from coding_review_agent_loop import checkout_verification, coder_ci_guard, config, containment
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    assert "repair" not in containment.CONTAINED_ROLES
    assert not hasattr(containment, "DEPRECATED_NOOP_ROLES")
    assert not hasattr(AntigravityBackend, "run_repair")
    assert not hasattr(config, "DEFAULT_REPAIR_MODELS")
    assert "repair" not in checkout_verification.TOOL_ISOLATED_ROLES
    assert "repair" not in coder_ci_guard.GUARD_ROLES
