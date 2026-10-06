"""Package-script resolution: worker policy, admissibility and prompts (issue #1294)."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

import coding_review_agent_loop.test_runtime as runtime
import coding_review_agent_loop.workdir_guard as workdir_guard
from agent_loop_helpers import make_config
from coding_review_agent_loop.local_test_evidence import (
    OUT_OF_CHECKOUT_CONTEXT_CAVEAT,
    LocalTestObservation,
    mark_out_of_checkout_context,
    observation_from_mapping,
    redact_observation,
)
from coding_review_agent_loop.response_validation import _admissible_evidence_observations
from coding_review_agent_loop.test_workers import WorkerBudget


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in ("AGENT_LOOP_INVOCATION_ID", "NODE_OPTIONS", "PW_TEST_REPORTER"):
        monkeypatch.delenv(name, raising=False)


def _package(root: Path, scripts: dict[str, str]) -> None:
    (root / "package.json").write_text(json.dumps({"scripts": scripts}), encoding="utf-8")


def _fake_pytest(root: Path) -> Path:
    bindir = root / "fakebin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "pytest"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    return bindir


def _budget(mode="clamp", workers=2):
    return WorkerBudget(workers, "inherited", mode, "cpu", {}, False)


# --- worker policy -----------------------------------------------------------


def test_resolved_pytest_body_gets_direct_pytest_clamp_policy(tmp_path, monkeypatch):
    from coding_review_agent_loop import runner as runner_module

    bindir = _fake_pytest(tmp_path)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    _package(tmp_path, {"test:py": "pytest -p no:_agent_loop_worker_cap tests/test_x.py"})
    result = runner_module.run_foreground_test(
        ["npm", "run", "test:py"], cwd=tmp_path, timeout_seconds=60, echo_output=False,
        worker_budget=_budget("clamp"), worker_lock_root=tmp_path / "locks",
    )
    assert "npm" not in result.args
    assert "no:_agent_loop_worker_cap" not in result.args
    assert result.args[0].endswith("pytest")
    assert result.suite_start == "verified"


def test_resolved_pytest_body_is_refused_in_refuse_mode(tmp_path, monkeypatch):
    from coding_review_agent_loop import runner as runner_module

    bindir = _fake_pytest(tmp_path)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    _package(tmp_path, {"test:py": "pytest -p no:_agent_loop_worker_cap tests/test_x.py"})
    result = runner_module.run_foreground_test(
        ["npm", "run", "test:py"], cwd=tmp_path, timeout_seconds=60, echo_output=False,
        worker_budget=_budget("refuse"), worker_lock_root=tmp_path / "locks",
    )
    assert result.outcome == "worker-budget-refused"


def test_non_package_clamped_pytest_keeps_post_policy_argv(tmp_path, monkeypatch):
    from coding_review_agent_loop import runner as runner_module

    bindir = _fake_pytest(tmp_path)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    result = runner_module.run_foreground_test(
        ["pytest", "-p", "no:_agent_loop_worker_cap", "tests/test_x.py"], cwd=tmp_path,
        timeout_seconds=60, echo_output=False,
        worker_budget=_budget("clamp"), worker_lock_root=tmp_path / "locks",
    )
    assert result.args[0] == "pytest"
    assert "no:_agent_loop_worker_cap" not in result.args


# --- admissibility -----------------------------------------------------------


def _live(command, executed=(), cwd="/tmp/work"):
    return LocalTestObservation(
        command=tuple(command), outcome="failed", provenance="parent-observed",
        cwd=cwd, executed_command=tuple(executed),
        wrapper_bootstrap="verified", inner_exec="started", suite_start="verified",
    )


def test_package_manager_command_needs_an_executed_argv(tmp_path):
    cmd = ["npm", "run", "test:x"]
    assert not workdir_guard.command_is_admissible_evidence(cmd, assigned_workdir=tmp_path)
    assert not workdir_guard.command_is_admissible_evidence(
        ["env", "A=1", *cmd], assigned_workdir=tmp_path
    )
    inside = [str(tmp_path / "node_modules/.bin/playwright"), "test", "--project=a"]
    assert workdir_guard.command_is_admissible_evidence(
        cmd, assigned_workdir=tmp_path, executed_argv=inside
    )
    outside = ["node", "--test", "/outside/tests/test_x.js"]
    assert not workdir_guard.command_is_admissible_evidence(
        cmd, assigned_workdir=tmp_path, executed_argv=outside
    )
    mixed = ["node", "--test", "/outside/tests/test_x.js", "tests/test_y.js"]
    assert not workdir_guard.command_is_admissible_evidence(
        cmd, assigned_workdir=tmp_path, executed_argv=mixed
    )
    assert workdir_guard.command_is_admissible_evidence(["pytest", "tests/"], assigned_workdir=tmp_path)


def test_outside_only_body_is_context_but_mixed_failure_is_not(tmp_path):
    cmd = ("npm", "run", "test:x")
    outside = _live(cmd, ("pytest", "/outside/tests/test_x.py"))
    mixed = _live(cmd, ("pytest", "/outside/tests/test_x.py", "tests/test_y.py"))
    inside = _live(cmd, ("pytest", "tests/test_y.py"))
    marked = mark_out_of_checkout_context([outside, mixed, inside], assigned_workdir=tmp_path)
    assert [OUT_OF_CHECKOUT_CONTEXT_CAVEAT in row.caveats for row in marked] == [True, False, False]
    kept = _admissible_evidence_observations(
        [outside, mixed, inside], assigned_workdir=tmp_path, selectable=True
    )
    assert kept == (inside,)


def test_restored_package_manager_row_without_executed_command_is_not_selectable(tmp_path):
    live = _live(("npm", "run", "test:x"), ("node", "--test", "tests/test_y.js"))
    restored = observation_from_mapping(live.public_projection())
    assert restored.executed_command == ()
    assert _admissible_evidence_observations(
        [restored], assigned_workdir=tmp_path, selectable=True
    ) == ()
    assert "executed_command" not in live.public_projection()


# --- verbatim projection signal ----------------------------------------------


def test_command_verbatim_tracks_redaction_and_survives_redact_observation():
    clean = LocalTestObservation(
        command=("node", "tests/test_x.js"), outcome="passed", provenance="parent-observed", cwd="/tmp/w",
    )
    assert clean.public_projection()["command_verbatim"] is True
    assert redact_observation(clean).public_projection()["command_verbatim"] is True
    for command in (
        ("node", "tests/my\ttest.js"),
        ("node", "tests/test_x.js", "--token=abc"),
        ("node", "/home/u/tests/test_x.js"),
    ):
        dirty = replace(clean, command=command)
        assert dirty.public_projection()["command_verbatim"] is False
        redacted = redact_observation(dirty)
        # Re-projecting the already-clean tokens must not launder the flag.
        assert redacted.public_projection()["command_verbatim"] is False
    assert observation_from_mapping({"command": ["node", "t.js"]}).command_verbatim is False


# --- prompts -----------------------------------------------------------------


def test_coder_prompt_states_node_test_and_npm_run_rules(tmp_path):
    from coding_review_agent_loop.prompts import (
        build_issue_implementation_prompt,
        build_followup_prompt,
    )

    config = make_config(tmp_path)
    initial = " ".join(build_issue_implementation_prompt(56, "1. Fix it.", config).split())
    followup = " ".join(build_followup_prompt(77, 2, "Fix the bug.", config).split())
    for prompt in (initial, followup):
        assert "`node --test <file>`" in prompt
        assert "`npm run <name>`" in prompt



def test_recommendation_cohorts_follow_the_resolved_package_script(tmp_path):
    from coding_review_agent_loop.prompts import _expected_worker_cohorts

    config = make_config(tmp_path, test_workers=3, test_worker_enforcement="clamp")
    _package(config.claude_dir, {"test:py": "pytest tests/", "test:j": "jest"})
    (config.claude_dir / "pyproject.toml").write_text(
        '[project.optional-dependencies]\ndev = ["pytest-xdist"]\n', encoding="utf-8",
    )
    assert _expected_worker_cohorts(config, ["pytest", "tests/"]) == ["3", "serial"]
    assert _expected_worker_cohorts(config, ["npm", "run", "test:py"]) == ["3", "serial"]
    # A discarded body falls back to today's cohorts for the original command.
    assert _expected_worker_cohorts(config, ["npm", "run", "test:j"]) == ["unknown"]
