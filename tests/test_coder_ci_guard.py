"""Coder-turn CI-dispatch guard (#1366)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from coding_review_agent_loop import coder_ci_guard
from coding_review_agent_loop.coder_ci_guard import (
    EXIT_NO_TARGET,
    EXIT_REFUSED,
    GUARD_ACTIVE,
    GUARD_DIR_ENV,
    GUARD_ENV,
    REAL_GH_ENV,
    TARGET_ENV,
    classify_ci_trigger,
    coder_ci_guard_environment,
)
from coding_review_agent_loop.config import AgentLoopConfig

from agent_loop_helpers import make_config

SRC = Path(__file__).resolve().parents[1] / "src"
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX launcher")


# Every pflag spelling of a dispatch/rerun POST (#1366 item-1/item-2): one
# table drives both the classifier and the real launcher, so a new spelling
# is covered at both layers at once.
_RERUN = "repos/o/r/actions/runs/42/rerun"
_DISPATCH = "repos/o/r/actions/workflows/ci.yml/dispatches"
API_TRIGGER_SPELLINGS = [
    ("api", _RERUN, "-X", "POST"),
    ("api", _RERUN, "-XPOST"),
    ("api", _RERUN, "-X=POST"),
    ("api", _RERUN, "-X=post"),
    ("api", _RERUN, "--method", "POST"),
    ("api", _RERUN, "--method=POST"),
    ("api", _RERUN, "-iX", "POST"),
    ("api", _RERUN, "-iXPOST"),
    ("api", _RERUN, "-iX=POST"),
    ("api", _DISPATCH, "-f", "ref=main"),
    ("api", _DISPATCH, "-fref=main"),
    ("api", _DISPATCH, "-f=ref=main"),
    ("api", _DISPATCH, "-if", "ref=branch"),
    ("api", _DISPATCH, "-iFref=branch"),
    ("api", _DISPATCH, "--field=ref=main"),
    ("api", _DISPATCH, "--raw-field", "ref=main"),
    ("api", _DISPATCH, "--input=body.json"),
    ("api", "-H", "Accept: x", _DISPATCH, "-f", "ref=main"),
    ("-R", "api", "api", _RERUN, "-X", "POST"),
    ("api", "-X", "POST", "--", _RERUN),
]
API_NON_TRIGGER_SPELLINGS = [
    ("api", _RERUN),
    ("api", _RERUN, "-i"),
    ("api", _RERUN, "-X=GET"),
    ("api", _RERUN, "-iX", "GET"),
    ("api", _DISPATCH, "-X", "GET", "-f", "ref=main"),
    ("api", _DISPATCH, "--method=GET", "-fref=main"),
    ("api", "repos/o/r/actions/runs", "-X=POST"),
]


@pytest.mark.parametrize("argv", API_TRIGGER_SPELLINGS)
def test_every_pflag_spelling_of_an_api_trigger_is_classified(argv):
    assert classify_ci_trigger(argv) is not None


@pytest.mark.parametrize("argv", API_NON_TRIGGER_SPELLINGS)
def test_get_spellings_of_trigger_endpoints_are_not_triggers(argv):
    assert classify_ci_trigger(argv) is None


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["workflow", "run", "ci.yml"], "gh workflow run"),
        (["workflow", "run", "ci.yml", "--ref", "agent-loop/managed-1", "-f", "a="], "gh workflow run"),
        (["workflow", "-R", "o/r", "run", "ci.yml"], "gh workflow run"),
        (["run", "rerun", "1"], "gh run rerun"),
        (["run", "rerun", "1", "--failed"], "gh run rerun"),
        (["api", "-X", "POST", "repos/o/r/actions/workflows/ci.yml/dispatches", "-f", "ref=main"], "gh api"),
        (["api", "repos/o/r/actions/workflows/ci.yml/dispatches", "-f", "ref=main"], "gh api"),
        (["api", "--method=POST", "/repos/o/r/actions/workflows/1/dispatches"], "gh api"),
        (["api", "-XPOST", "repos/{owner}/{repo}/actions/runs/9/rerun"], "gh api"),
        (["api", "--input", "body.json", "repos/o/r/actions/workflows/ci.yml/dispatches"], "gh api"),
        (["api", "-X", "post", "repos/o/r/actions/runs/9/rerun-failed-jobs"], "gh api"),
        (["api", "-Fref=main", "repos/o/r/actions/jobs/9/rerun"], "gh api"),
        (["api", "-X", "POST", "https://api.github.com/repos/o/r/actions/runs/9/rerun"], "gh api"),
    ],
)
def test_ci_trigger_forms_are_classified(argv, expected):
    trigger = classify_ci_trigger(argv)
    assert trigger is not None and trigger.startswith(expected)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["pr", "checks", "5"],
        ["run", "view", "1", "--json", "status,conclusion"],
        ["run", "list", "--workflow", "ci.yml"],
        ["workflow", "view", "ci.yml"],
        ["workflow", "list"],
        ["api", "repos/o/r/actions/runs"],
        ["api", "repos/o/r/actions/workflows/ci.yml/dispatches"],
        ["api", "-X", "GET", "repos/o/r/actions/runs/9/rerun"],
        ["api", "repos/o/r/actions/runs?event=workflow_dispatch", "-q", ".total_count"],
        ["api", "-X", "POST", "repos/o/r/issues/5/comments", "-f", "body=x"],
        ["pr", "create", "--draft", "--label", "agent-loop-managed", "--body-file", "b.md"],
        ["pr", "view", "5", "--json", "number"],
    ],
)
def test_read_only_and_unrelated_forms_are_not_triggers(argv):
    assert classify_ci_trigger(argv) is None


# ---------------------------------------------------------------- launcher ---


def _fake_gh(directory: Path, *, exit_code: int = 7, name: str = "gh") -> tuple[Path, Path]:
    """A fake real gh that records argv and PATH, prints, and exits non-zero."""
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / "calls.jsonl"
    script = directory / name
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"open({str(record)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'path': os.environ.get('PATH', '')}}) + '\\n')\n"
        "print('fake-gh-out ' + ' '.join(sys.argv[1:]))\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script, record


def _calls(record: Path) -> list[dict]:
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]


def _guarded(tmp_path: Path, *, gh_cmd: str = "gh", ambient: dict | None = None, path: str = "") -> dict:
    config = replace(make_config(tmp_path), gh_cmd=gh_cmd, coder_ci_dispatch_guard=True)
    values = {"PATH": path, **(ambient or {})}
    env = coder_ci_guard_environment(config, "claude", "coder", ambient=values, root=tmp_path / "guard-root")
    assert env, "guard environment should be installed"
    return {**os.environ, **values, **env}


def _run_gh(env: dict, *argv: str) -> subprocess.CompletedProcess[str]:
    launcher = Path(env[GUARD_DIR_ENV]) / "gh"
    return subprocess.run([str(launcher), *argv], env=env, capture_output=True, text=True, timeout=60)


def test_guard_refuses_trigger_without_reaching_real_gh(tmp_path):
    fake, record = _fake_gh(tmp_path / "bin")
    env = _guarded(tmp_path, path=str(fake.parent))
    result = _run_gh(env, "workflow", "run", "ci.yml", "--ref", "agent-loop/managed-1")
    assert result.returncode == EXIT_REFUSED
    assert "refused `gh workflow run`" in result.stderr
    assert "orchestrator" in result.stderr and "operator" in result.stderr
    assert _calls(record) == []


@pytest.mark.parametrize(
    "argv",
    [
        ("api", "-X", "POST", "repos/o/r/actions/workflows/ci.yml/dispatches", "-f", "ref=x"),
        ("api", "repos/o/r/actions/workflows/ci.yml/dispatches", "-f", "ref=x"),
        ("run", "rerun", "42"),
        ("api", "-X", "POST", "repos/o/r/actions/runs/42/rerun-failed-jobs"),
        *API_TRIGGER_SPELLINGS,
    ],
)
def test_guard_refuses_api_and_rerun_forms(tmp_path, argv):
    fake, record = _fake_gh(tmp_path / "bin")
    env = _guarded(tmp_path, path=str(fake.parent))
    result = _run_gh(env, *argv)
    assert result.returncode == EXIT_REFUSED
    assert _calls(record) == []


@pytest.mark.parametrize(
    "argv",
    [
        ("pr", "checks", "5"),
        ("run", "view", "1", "--json", "status,conclusion,startedAt"),
        ("api", "repos/o/r/actions/runs?event=workflow_dispatch"),
    ],
)
def test_read_only_gh_passes_through_with_identical_argv(tmp_path, argv):
    fake, record = _fake_gh(tmp_path / "bin", exit_code=7)
    env = _guarded(tmp_path, path=str(fake.parent))
    result = _run_gh(env, *argv)
    assert result.returncode == 7
    assert result.stdout.strip() == "fake-gh-out " + " ".join(argv)
    calls = _calls(record)
    assert [call["argv"] for call in calls] == [list(argv)]
    guard_dir = os.path.realpath(env[GUARD_DIR_ENV])
    assert guard_dir not in [os.path.realpath(p) for p in calls[0]["path"].split(os.pathsep) if p]


def test_guard_is_inert_without_the_flag(tmp_path):
    fake, record = _fake_gh(tmp_path / "bin", exit_code=0)
    env = _guarded(tmp_path, path=str(fake.parent))
    env.pop(GUARD_ENV)
    result = _run_gh(env, "workflow", "run", "ci.yml")
    assert result.returncode == 0
    assert [call["argv"] for call in _calls(record)] == [["workflow", "run", "ci.yml"]]


def test_guard_fails_closed_when_target_missing_or_self_pointing(tmp_path):
    fake, record = _fake_gh(tmp_path / "bin")
    env = _guarded(tmp_path, path=str(fake.parent))
    missing = {**env, TARGET_ENV: str(tmp_path / "nope" / "gh")}
    result = _run_gh(missing, "pr", "checks", "5")
    assert result.returncode == EXIT_NO_TARGET
    selfish = {**env, TARGET_ENV: str(Path(env[GUARD_DIR_ENV]) / "gh")}
    result = _run_gh(selfish, "pr", "checks", "5")
    assert result.returncode == EXIT_NO_TARGET
    assert "back at the guard" in result.stderr
    assert _calls(record) == []


# ------------------------------------------------------ shim composition ---


def _shim_entry(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    entry = directory / "agent-loop-gh"
    entry.write_text(
        f"#!{sys.executable} -I\n"
        f"import sys; sys.path.insert(0, {str(SRC)!r})\n"
        "from coding_review_agent_loop.gh_rest_shim import main\n"
        "sys.exit(main())\n",
        encoding="utf-8",
    )
    entry.chmod(0o755)
    return entry


def _fake_real_gh_for_shim(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / "calls.jsonl"
    pr = {
        "number": 5, "title": "T", "body": "", "state": "open", "html_url": "https://x/pull/5",
        "user": {"login": "a"}, "draft": True, "head": {"sha": "abc", "ref": "b"}, "base": {"ref": "main"},
    }
    script = directory / "gh"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"open({str(record)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['api', 'repos/acme/widget/pulls/5']:\n"
        f"    print(json.dumps({pr!r}))\n"
        "    sys.exit(0)\n"
        "print('real ' + ' '.join(args))\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script, record


@pytest.mark.parametrize("ambient_real", [False, True])
def test_guard_composes_with_rest_shim(tmp_path, ambient_real):
    shim = _shim_entry(tmp_path / "shim-bin")
    real, record = _fake_real_gh_for_shim(tmp_path / "real-bin")
    ambient = {"AGENT_LOOP_GH_TRANSPORT": "rest", "XDG_CACHE_HOME": str(tmp_path / "cache")}
    if ambient_real:
        ambient[REAL_GH_ENV] = str(real)
    env = _guarded(
        tmp_path,
        gh_cmd="agent-loop-gh",
        ambient=ambient,
        path=os.pathsep.join([str(shim.parent), str(real.parent)]),
    )
    assert env[TARGET_ENV] == str(shim)
    # The shim's real-gh resolution is preserved or derived pre-guard, never the guard.
    assert env[REAL_GH_ENV] == str(real)
    assert os.path.realpath(env[REAL_GH_ENV]) != os.path.realpath(Path(env[GUARD_DIR_ENV]) / "gh")

    viewed = _run_gh(env, "run", "view", "1")
    assert viewed.returncode == 0, viewed.stderr
    assert viewed.stdout.strip() == "real run view 1"

    emulated = _run_gh(env, "pr", "view", "5", "--repo", "acme/widget", "--json", "number,headRefOid")
    assert emulated.returncode == 0, emulated.stderr
    assert json.loads(emulated.stdout) == {"number": 5, "headRefOid": "abc"}

    refused = _run_gh(env, "workflow", "run", "ci.yml")
    assert refused.returncode == EXIT_REFUSED
    calls = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
    assert ["run", "view", "1"] in calls
    assert not any(call[:2] == ["workflow", "run"] for call in calls)


# ------------------------------------------------------------ role scope ---


@pytest.mark.parametrize("role", ["coder", "repair"])
def test_guard_environment_present_for_guarded_coder_and_repair(tmp_path, role):
    fake, _ = _fake_gh(tmp_path / "bin")
    config = replace(make_config(tmp_path), coder_ci_dispatch_guard=True)
    env = coder_ci_guard_environment(
        config, "codex", role, ambient={"PATH": str(fake.parent)}, root=tmp_path / "root"
    )
    assert env[GUARD_ENV] == GUARD_ACTIVE
    assert env["PATH"].split(os.pathsep)[0] == env[GUARD_DIR_ENV]
    assert env[TARGET_ENV] == str(fake)


@pytest.mark.parametrize("role", ["reviewer", "planner", "step_back", None])
def test_guard_environment_absent_for_other_roles(tmp_path, role):
    config = replace(make_config(tmp_path), coder_ci_dispatch_guard=True)
    assert coder_ci_guard_environment(config, "claude", role, root=tmp_path / "root") == {}


@pytest.mark.parametrize("role", ["coder", "repair"])
def test_guard_environment_absent_when_unmanaged(tmp_path, role):
    config = replace(make_config(tmp_path), managed_ci=True, auto_merge=True)
    # Never derived from managed_ci / auto-merge: only the invocation flag counts.
    assert config.coder_ci_dispatch_guard is False
    assert coder_ci_guard_environment(config, "claude", role, root=tmp_path / "root") == {}


def test_guard_environment_degrades_to_prompt_only_without_gh(tmp_path):
    config = replace(make_config(tmp_path), gh_cmd="definitely-not-gh-xyz", coder_ci_dispatch_guard=True)
    assert coder_ci_guard_environment(config, "claude", "coder", ambient={"PATH": ""}, root=tmp_path / "r") == {}


def test_coder_ci_dispatch_guard_defaults_off():
    assert AgentLoopConfig.__dataclass_fields__["coder_ci_dispatch_guard"].default is False


def test_every_coder_launch_site_uses_the_shared_helper():
    """Static enumeration: each coder/repair launch path installs the guard."""
    root = SRC / "coding_review_agent_loop"
    sites = {
        "agents/claude.py": 'coder_ci_guard_environment(config, "claude", role)',
        "agents/codex.py": 'coder_ci_guard_environment(config, "codex", role)',
        "agents/gemini.py": 'coder_ci_guard_environment(config, "gemini", role)',
        "agents/antigravity.py": 'coder_ci_guard_environment(config, "antigravity", role)',
        "agents/format_repair.py": 'coder_ci_guard_environment(config, backend, "repair")',
        "repair.py": 'coder_ci_guard_environment(config, "gemini", "repair")',
    }
    for relative, needle in sites.items():
        assert needle in (root / relative).read_text(encoding="utf-8"), relative
    # The test-hook-only Gemini subprocess in attempt_repair is not a
    # production launch; flag it here if it ever stops being hook-only.
    repair_text = (root / "repair.py").read_text(encoding="utf-8")
    assert repair_text.count("subprocess.run(") == 2


def test_module_exports_are_stable():
    assert coder_ci_guard.GUARD_ROLES == frozenset({"coder", "repair"})


# ------------------------------------------- agent-loop executable pin (#1632) ---


def test_guard_composes_with_agent_loop_executable_pin(tmp_path, monkeypatch):
    import shutil
    import tempfile

    from coding_review_agent_loop import agent_executable

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(agent_executable, "_LAUNCHER_CACHE", {})
    fake, record = _fake_gh(tmp_path / "bin", exit_code=7)
    stale = tmp_path / "bin" / "agent-loop"
    stale.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
    stale.chmod(0o755)
    guarded = _guarded(tmp_path, path=str(fake.parent))
    # The runner applies the pin after merging the backend (guard) env.
    launch_env = {**guarded, **agent_executable.agent_executable_environment(guarded)}
    entries = launch_env["PATH"].split(os.pathsep)
    launcher = launch_env[agent_executable.EXECUTABLE_ENV]
    guard_dir = launch_env[GUARD_DIR_ENV]
    assert entries[:2] == [str(Path(launcher).parent), guard_dir]
    assert shutil.which("gh", path=launch_env["PATH"]) == str(Path(guard_dir) / "gh")
    assert shutil.which("agent-loop", path=launch_env["PATH"]) == launcher
    # The guard still refuses triggers and strips only its own directory.
    assert _run_gh(launch_env, "workflow", "run", "ci.yml").returncode == EXIT_REFUSED
    result = _run_gh(launch_env, "pr", "checks", "5")
    assert result.returncode == 7
    seen = _calls(record)[0]["path"].split(os.pathsep)
    assert os.path.realpath(guard_dir) not in [os.path.realpath(p) for p in seen if p]
    assert str(Path(launcher).parent) in seen
    assert str(fake.parent) in seen
