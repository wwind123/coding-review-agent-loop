import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_pr_review
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import DEFAULT_REASONING_EFFORT, config_from_args
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.protocol import parse_structured_pr_review
from coding_review_agent_loop.repair import execute_repair
from coding_review_agent_loop.runner import CommandResult
from coding_review_agent_loop.usage import RunUsageContext


class RepairRunner:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def run_with_log(self, args, **kwargs):
        self.calls.append((args, kwargs))
        response = next(self.outputs)
        if isinstance(response, Exception):
            raise response
        text, returncode = response
        if args[1] == "exec":
            Path(args[args.index("--output-last-message") + 1]).write_text(text)
            stdout = json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 100, "output_tokens": 50, "cached_input_tokens": 0,
            }})
        else:
            stdout = json.dumps({"result": text, "usage": {"input_tokens": 100, "output_tokens": 50}})
        kwargs["log_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["log_path"].write_text(stdout)
        return CommandResult(args, kwargs["cwd"], stdout, "", returncode)


@pytest.mark.parametrize("backend", ["codex", "claude"])
@pytest.mark.parametrize("effort", ["", "high"])
def test_cli_repair_isolated_and_independent(tmp_path, backend, effort):
    valid = structured_pr_review(summary="Keep the original finding.")
    runner = RepairRunner([(valid, 0)])
    config = make_config(
        tmp_path, repair_backend=backend, repair_models=("repair-model",),
        repair_reasoning_effort=effort, codex_model="coder-model", claude_model="coder-model",
        codex_reasoning_effort="xhigh", claude_effort="max",
        codex_args=("--dangerously-bypass-approvals-and-sandbox",),
        claude_args=("--dangerously-skip-permissions",),
    )
    usage = RunUsageContext("repair", tmp_path / "usage.json")
    with patch("coding_review_agent_loop.repair.AntigravityBackend.discover_models") as discovery:
        repaired, _, attempts = execute_repair(
            valid, runner=runner, config=config, run_id="repair", usage_context=usage,
            validate=lambda text: parse_structured_pr_review(text, reviewer="OpenAI Codex"),
            expected_kind="pr_review",
        )
    discovery.assert_not_called()
    assert repaired == valid
    assert attempts[0].outcome == "succeeded"
    args, kwargs = runner.calls[0]
    assert args[args.index("--model") + 1] == "repair-model"
    assert not any("dangerous" in arg for arg in args)
    assert "coder-model" not in args
    assert valid in kwargs["input_text"]
    assert valid not in " ".join(args)
    assert kwargs["cwd"] not in (config.codex_dir, config.claude_dir)
    assert not kwargs["cwd"].exists()
    assert kwargs["containment_role"] == "repair"
    assert kwargs["timeout_seconds"] == 120
    assert not kwargs["check"]
    resolved = effort or DEFAULT_REASONING_EFFORT
    if backend == "codex":
        assert "--ignore-user-config" in args and "--ephemeral" in args
        assert args[args.index("--sandbox") + 1] == "read-only"
        assert 'web_search="disabled"' in args
        assert "features.shell_tool=false" in args
        assert f'model_reasoning_effort="{resolved}"' in args
        assert args[-1] == "-"
    else:
        assert "--safe-mode" in args and "--strict-mcp-config" in args
        assert args[args.index("--tools") + 1] == ""
        assert args[args.index("--effort") + 1] == resolved
        assert kwargs["env"]["CLAUDE_CODE_EFFORT_LEVEL"] == resolved
    record = usage.records[0]
    assert record.role == record.turn_role == "repair"
    assert record.configured_model == "repair-model"
    assert record.configured_effort == resolved
    assert record.raw_backend_usage["input_tokens"] == 100
    assert record.validation_status == "validated"


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_cli_repair_preservation_and_model_fallback(tmp_path, backend):
    raw = json.dumps({"kind": "plan_review", "blocking_plan_issues": [
        {"title": "Test the claim", "detail": "Wire the getter; add a two-round test; no mutation on 503."},
    ]})
    lossy = json.dumps({"kind": "plan_review", "blocking_plan_issues": ["Test the claim"]})
    faithful = json.dumps({"kind": "plan_review", "blocking_plan_issues": [
        "Test the claim: Wire the getter; add a two-round test; no mutation on 503.",
    ]})
    runner = RepairRunner([(lossy, 0), (faithful, 0)])
    repaired, _, attempts = execute_repair(
        raw, runner=runner, config=make_config(tmp_path, repair_backend=backend,
                                              repair_models=("first", "second")),
        run_id="repair", usage_context=None, validate=json.loads, expected_kind="plan_review",
    )
    assert repaired == faithful
    assert [a.outcome for a in attempts] == ["invalid_output", "succeeded"]
    assert "content preservation failed" in attempts[0].diagnostic
    assert runner.calls[0][1]["cwd"] != runner.calls[1][1]["cwd"]
    assert [args[args.index("--model") + 1] for args, _ in runner.calls] == ["first", "second"]


@pytest.mark.parametrize("backend", ["codex", "claude"])
@pytest.mark.parametrize("response,outcome", [
    ((structured_pr_review(), 1), "nonzero_exit"),
    ((structured_pr_review(), None), "timeout"),
    (("", 0), "empty_output"),
    (("not json", 0), "invalid_output"),
    (subprocess.TimeoutExpired("cli", 120), "timeout"),
    (OSError("missing executable"), "spawn_error"),
])
def test_cli_repair_failures_fail_closed(tmp_path, backend, response, outcome):
    repaired, _, attempts = execute_repair(
        "malformed", runner=RepairRunner([response]),
        config=make_config(tmp_path, repair_backend=backend, repair_models=("repair-model",)),
        run_id="repair", usage_context=None, validate=json.loads, expected_kind="pr_review",
    )
    assert repaired is None
    assert attempts[0].outcome == outcome


# --- #2066: --repair-* and --containment-repair-* are deprecated no-ops ---------


def _issue_args(tmp_path, *extra):
    return build_parser().parse_args([
        "issue", "1", "--repo", "OWNER/REPO",
        "--codex-dir", str(tmp_path / "codex"), "--claude-dir", str(tmp_path / "claude"),
        "--subprocess-log-dir", str(tmp_path / "logs"),
        *extra,
    ])


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_repair_backend_without_model_is_a_noop_with_a_deprecation_notice(tmp_path, capsys, backend):
    config = config_from_args(_issue_args(tmp_path, "--repair-backend", backend), FakeRunner())
    notice = capsys.readouterr().err
    assert "deprecated no-op flag(s) ignored: --repair-backend" in notice
    assert notice.count("deprecated no-op") == 1
    # The no-op value never reaches the config: no repair backend exists.
    assert config.repair_backend == "antigravity"


@pytest.mark.parametrize("backend,effort", [("codex", "max"), ("claude", "minimal"), ("antigravity", "high")])
def test_former_repair_validation_never_fails_a_run(tmp_path, capsys, backend, effort):
    config_from_args(_issue_args(
        tmp_path, "--repair-backend", backend, "--repair-model", " ",
        "--repair-reasoning-effort", effort, "--repair-timeout-seconds", "-5",
    ), FakeRunner())
    notice = capsys.readouterr().err
    for flag in ("--repair-backend", "--repair-model", "--repair-reasoning-effort", "--repair-timeout-seconds"):
        assert flag in notice
    # Library-built configs never validate the retired fields either.
    make_config(tmp_path, repair_backend=backend, repair_reasoning_effort=effort, repair_timeout_seconds=0)


def test_no_notice_without_repair_flags(tmp_path, capsys):
    config_from_args(_issue_args(tmp_path), FakeRunner())
    assert "deprecated" not in capsys.readouterr().err


@pytest.mark.parametrize("mode,number", [("pr", "1"), ("issue", "2"), ("discuss", "3")])
def test_parser_still_accepts_deprecated_repair_settings(mode, number):
    args = build_parser().parse_args([
        mode, number, "--repair-backend", "codex", "--repair-model", "first",
        "--repair-model", "second", "--repair-reasoning-effort", "high",
        "--repair-timeout-seconds", "180",
        "--containment-repair-memory-max", "not-a-size",
    ])
    assert args.repair_backend == "codex"
    assert args.repair_model == ["first", "second"]


def test_repair_backend_never_requires_the_agy_cli(tmp_path, monkeypatch):
    import shutil

    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda command: None if command == "agy" else real_which(command) or f"/usr/bin/{command}")
    runner = FakeRunner()
    args = _issue_args(tmp_path, "--repair-backend", "antigravity")
    args.dry_run = False
    config_from_args(args, runner)


@pytest.mark.parametrize("repair_values", [
    {"containment_repair_memory_max": "not-a-size"},
    {"containment_repair_memory_high": "900M", "containment_repair_memory_max": "100M"},
    {"containment_repair_tasks_max": "-3"},
])
def test_invalid_containment_repair_overrides_are_discarded(tmp_path, repair_values):
    from coding_review_agent_loop.containment import policy_from_values

    values = {
        "containment_coder_memory_max": "200M",
        "containment_reviewer_tasks_max": "321",
        **repair_values,
    }
    policy = policy_from_values(values)
    roles = dict(policy.roles)
    assert roles["coder"].memory_max == 200_000_000
    assert roles["reviewer"].tasks_max == 321
    default_roles = dict(policy_from_values({}).roles)
    assert roles["repair"] == default_roles["repair"]
    # The config and the runner build the same policy, so neither rejects it.
    config = make_config(tmp_path, **values)
    assert dict(config.containment_policy.roles)["coder"].memory_max == 200_000_000


def test_invalid_containment_repair_flag_reaches_neither_config_nor_runner(tmp_path, capsys):
    from types import SimpleNamespace

    args = _issue_args(
        tmp_path, "--containment-repair-memory-max", "not-a-size",
        "--containment-repair-memory-high", "1T",
        "--containment-coder-memory-max", "300M",
    )
    config = config_from_args(args, FakeRunner())
    assert "--containment-repair-memory-max" in capsys.readouterr().err
    assert dict(config.containment_policy.roles)["coder"].memory_max == 300_000_000
    # Runner setup parses the same values and never rejects them either.
    runner = FakeRunner()
    runner.configure_from_config(SimpleNamespace(**vars(args)))
    assert dict(runner.containment_policy.roles)["coder"].memory_max == 300_000_000


def test_default_antigravity_chain_uses_current_flash_generation(tmp_path):
    from coding_review_agent_loop.config import DEFAULT_ANTIGRAVITY_MODELS

    assert DEFAULT_ANTIGRAVITY_MODELS == (
        "Gemini 3.8 Flash (High)",
        "Claude Opus 5.5 (Medium)",
    )
    config = config_from_args(_issue_args(tmp_path), FakeRunner())
    assert config.antigravity_models == DEFAULT_ANTIGRAVITY_MODELS


def test_repair_flag_help_says_deprecated_noop(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["issue", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--repair-backend REPAIR_BACKEND Deprecated no-op." in help_text
    assert "--containment-repair-memory-max CONTAINMENT_REPAIR_MEMORY_MAX Deprecated no-op" in help_text
