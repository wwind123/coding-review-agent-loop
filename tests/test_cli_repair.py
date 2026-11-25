import importlib
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_pr_review
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import DEFAULT_REASONING_EFFORT, config_from_args
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.protocol import parse_structured_pr_review
from coding_review_agent_loop.runner import CommandResult
from coding_review_agent_loop.usage import RunUsageContext


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


_SRC = Path(__file__).resolve().parents[1] / "src" / "coding_review_agent_loop"
_DELETED_MODULES = ("repair", "repair_preservation", "shape_check_audit", "agents.format_repair")
# The deprecated no-op flag handling (#2066): parsing, the no-op notice and
# managed-CI recovery command forwarding.  Nothing else may name the backend.
_NOOP_FLAG_OWNERS = {"cli.py", "config.py", "managed_ci_recovery_commands.py"}


def test_repair_backend_is_named_only_by_the_noop_flag_handling():
    """`no-repair-backend-static`: grep repair_backend over src (#2092)."""
    matches = sorted(
        path.relative_to(_SRC).as_posix()
        for path in _SRC.rglob("*.py")
        if re.search(r"repair[_-]backend", path.read_text(encoding="utf-8"))
    )
    assert matches and set(matches) <= _NOOP_FLAG_OWNERS, matches


@pytest.mark.parametrize("module", _DELETED_MODULES)
def test_deleted_repair_modules_are_neither_present_nor_imported(module):
    """`no-repair-backend-static`: no lingering reference to a deleted module."""
    assert not (_SRC / (module.replace(".", "/") + ".py")).exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"coding_review_agent_loop.{module}")
    leaf = module.rsplit(".", 1)[-1]
    import_re = re.compile(
        rf"^\s*(?:from\s+[.\w]*\b{leaf}\b\s+import|import\s+[.\w]*\b{leaf}\b"
        rf"|from\s+[.\w]+\s+import\s+(?:\(\s*)?(?:[\w\s,]*,\s*)?{leaf}\b)",
        re.MULTILINE,
    )
    offenders = [
        path.relative_to(_SRC).as_posix()
        for path in _SRC.rglob("*.py")
        if import_re.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_no_shape_check_annotation_remains_in_src():
    """The shape-check audit registry and its inline annotations are gone (#2092)."""
    assert [
        path.relative_to(_SRC).as_posix()
        for path in _SRC.rglob("*.py")
        if "# shape-check:" in path.read_text(encoding="utf-8")
    ] == []
