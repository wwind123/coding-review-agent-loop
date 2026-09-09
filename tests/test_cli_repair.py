import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_pr_review
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import config_from_args
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
    resolved = effort or "medium"
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


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_cli_repair_requires_explicit_model(tmp_path, backend):
    with pytest.raises(AgentLoopError, match="explicit --repair-model"):
        make_config(tmp_path, repair_backend=backend)


@pytest.mark.parametrize("backend,effort", [("codex", "max"), ("claude", "minimal"), ("antigravity", "high")])
def test_cli_repair_invalid_effort(tmp_path, backend, effort):
    with pytest.raises(AgentLoopError, match="--repair-reasoning-effort"):
        make_config(tmp_path, repair_backend=backend, repair_models=("repair-model",), repair_reasoning_effort=effort)


@pytest.mark.parametrize("mode,number", [("pr", "1"), ("issue", "2")])
def test_parser_accepts_cli_repair_settings(mode, number):
    args = build_parser().parse_args([
        mode, number, "--repair-backend", "codex", "--repair-model", "first",
        "--repair-model", "second", "--repair-reasoning-effort", "high",
    ])
    assert args.repair_backend == "codex"
    assert args.repair_model == ["first", "second"]
    assert args.repair_reasoning_effort == "high"


@pytest.mark.parametrize("backend", ["codex", "claude"])
@pytest.mark.parametrize("mode", ["issue", "pr", "discuss"])
def test_repair_options_reach_config(tmp_path, backend, mode):
    args = build_parser().parse_args([
        mode, "1", "--repo", "OWNER/REPO",
        "--codex-dir", str(tmp_path / "codex"), "--claude-dir", str(tmp_path / "claude"),
        "--subprocess-log-dir", str(tmp_path / "logs"),
        "--repair-backend", backend, "--repair-model", "first", "--repair-model", "second",
        "--repair-reasoning-effort", "high", "--repair-timeout-seconds", "180",
    ])
    config = config_from_args(args, FakeRunner())
    assert config.repair_backend == backend
    assert config.repair_models == ("first", "second")
    assert config.repair_reasoning_effort == "high"
    assert config.repair_timeout_seconds == 180
