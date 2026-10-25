"""The agent-loop executable pin for agent subprocesses (#1632)."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import coding_review_agent_loop
from coding_review_agent_loop import agent_executable as ae
from coding_review_agent_loop.runner import Runner

PACKAGE_DIR = Path(coding_review_agent_loop.__file__).resolve().parent


@pytest.fixture(autouse=True)
def _private_launcher_root(tmp_path, monkeypatch):
    """Install launchers under this test's tmp dir, never the shared scratch root."""
    tmp = tmp_path / "tmpdir"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    monkeypatch.setattr(ae, "_LAUNCHER_CACHE", {})
    monkeypatch.setattr(ae, "_WARNED", set())
    return tmp


def _stale_agent_loop(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stale = directory / "agent-loop"
    stale.write_text("#!/bin/sh\necho STALE\nexit 9\n", encoding="utf-8")
    stale.chmod(0o755)
    return stale


def _fail_install(monkeypatch):
    def refuse(_root=None):
        raise OSError("scratch root is not writable")

    monkeypatch.setattr(ae, "install_agent_loop_launcher", refuse)


def test_launcher_is_content_addressed_isolated_and_idempotent(_private_launcher_root):
    launcher = ae.install_agent_loop_launcher()
    text = launcher.read_text(encoding="utf-8")
    assert text == ae.launcher_text()
    assert text.startswith("#!/bin/sh\nexec ")
    assert f"{shlex.quote(sys.executable)} -I -c " in text
    assert repr(str(PACKAGE_DIR.parent)) in text
    assert launcher.name == "agent-loop"
    assert launcher.parent.parent == _private_launcher_root / "coding-review-agent-loop" / "agent-loop-exe"
    assert oct(launcher.stat().st_mode & 0o777) == oct(0o700)
    # The launcher directory holds only the launcher, never python or pytest.
    assert sorted(os.listdir(launcher.parent)) == ["agent-loop"]
    before = launcher.stat().st_mtime_ns
    assert ae.install_agent_loop_launcher() == launcher
    assert launcher.stat().st_mtime_ns == before


def test_launcher_ignores_hostile_pythonpath_and_cwd(tmp_path):
    shadow = tmp_path / "checkout"
    package = shadow / "coding_review_agent_loop"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("print('SHADOW INIT')\n", encoding="utf-8")
    (package / "cli.py").write_text(
        "import sys\n\ndef main():\n    print('SHADOW PACKAGE')\n    return 7\n", encoding="utf-8"
    )
    launcher = ae.install_agent_loop_launcher()
    env = {**os.environ, "PYTHONPATH": str(shadow), "PYTHONSAFEPATH": ""}
    result = subprocess.run(
        [str(launcher), "run-tests", "--preflight"],
        cwd=shadow, env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stderr
    assert "agent-loop preflight: verified" in result.stdout
    # Neither the PYTHONPATH entry nor the cwd package was imported.
    assert "SHADOW" not in result.stdout + result.stderr


def test_environment_prepends_only_the_launcher_dir_without_stacking(tmp_path):
    stale = _stale_agent_loop(tmp_path / "stale-bin")
    ambient = {"PATH": os.pathsep.join([str(stale.parent), "/usr/bin", "/bin"])}
    env = ae.agent_executable_environment(ambient)
    launcher = Path(env[ae.EXECUTABLE_ENV])
    entries = env["PATH"].split(os.pathsep)
    assert entries[0] == str(launcher.parent)
    assert entries[1:] == ambient["PATH"].split(os.pathsep)
    assert shutil.which("agent-loop", path=env["PATH"]) == str(launcher)
    again = ae.agent_executable_environment({**ambient, **env})
    assert again == env
    assert again["PATH"].split(os.pathsep).count(str(launcher.parent)) == 1


@pytest.mark.parametrize(
    "inherited",
    [":/usr/bin", "/usr/bin:", "/usr/bin::/bin", "", ":", "/usr/bin"],
    ids=["leading-empty", "trailing-empty", "interior-empty", "empty", "only-empty", "plain"],
)
def test_environment_preserves_every_inherited_entry_including_empty_ones(inherited):
    env = ae.agent_executable_environment({"PATH": inherited})
    directory = str(Path(env[ae.EXECUTABLE_ENV]).parent)
    assert env["PATH"] == directory + os.pathsep + inherited
    # Idempotent: a second application removes only the launcher dir copy.
    assert ae.agent_executable_environment({"PATH": env["PATH"]}) == env
    stacked = os.pathsep.join([directory, inherited, directory])
    assert ae.agent_executable_environment({"PATH": stacked})["PATH"] == env["PATH"]


def test_environment_without_inherited_path_is_only_the_launcher_dir():
    env = ae.agent_executable_environment({})
    assert env["PATH"] == str(Path(env[ae.EXECUTABLE_ENV]).parent)


def test_pinned_agent_environment_merges_overrides_then_pins():
    env = ae.pinned_agent_environment(
        {"PATH": "/guard:/usr/bin", "MARKER": "x"}, ambient={"PATH": "/ignored", "KEEP": "y"}
    )
    directory = str(Path(env[ae.EXECUTABLE_ENV]).parent)
    assert env["PATH"] == os.pathsep.join([directory, "/guard", "/usr/bin"])
    assert env["MARKER"] == "x" and env["KEEP"] == "y"


def test_install_failure_is_fail_soft_and_warned(tmp_path, monkeypatch, capsys):
    _fail_install(monkeypatch)
    ambient = {"PATH": "/usr/bin"}
    assert ae.agent_executable_environment(ambient) == {}
    assert ae.agent_executable_environment(ambient) == {}
    err = capsys.readouterr().err
    assert err.count("agent-loop executable pin not installed") == 1
    assert "scratch root is not writable" in err


def _runner_env_probe(tmp_path: Path, *, use_pty: bool, env: dict) -> dict[str, str]:
    program = (
        "import os, shutil\n"
        "print('WHICH=' + str(shutil.which('agent-loop')))\n"
        "print('EXE=' + os.environ.get('AGENT_LOOP_EXECUTABLE', ''))\n"
        "print('PATH=' + os.environ.get('PATH', ''))\n"
    )
    result = Runner().run_with_log(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        log_path=tmp_path / "logs" / ("pty.log" if use_pty else "pipe.log"),
        label="Claude",
        progress_interval_seconds=999,
        env=env,
        use_pty=use_pty,
    )
    values = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key in {"WHICH", "EXE", "PATH"}:
            values[key] = value
    return values


@pytest.mark.parametrize("use_pty", [False, True])
def test_runner_agent_launch_resolves_running_install_over_stale_path(tmp_path, use_pty):
    stale = _stale_agent_loop(tmp_path / "stale-bin")
    inherited = os.pathsep.join([str(stale.parent), os.environ.get("PATH", "")])
    values = _runner_env_probe(tmp_path, use_pty=use_pty, env={"PATH": inherited})
    launcher = values["EXE"]
    assert launcher and Path(launcher).name == "agent-loop"
    assert values["WHICH"] == launcher
    entries = values["PATH"].split(os.pathsep)
    assert entries[0] == str(Path(launcher).parent)
    assert entries[1] == str(stale.parent)
    assert entries.count(str(Path(launcher).parent)) == 1


def test_runner_agent_launch_keeps_inherited_path_when_install_fails(tmp_path, monkeypatch, capsys):
    _fail_install(monkeypatch)
    stale = _stale_agent_loop(tmp_path / "stale-bin")
    inherited = os.pathsep.join([str(stale.parent), os.environ.get("PATH", "")])
    values = _runner_env_probe(tmp_path, use_pty=False, env={"PATH": inherited})
    assert values["PATH"] == inherited
    assert values["EXE"] == ""
    assert values["WHICH"] == str(stale)
    assert "agent-loop executable pin not installed" in capsys.readouterr().err


def test_preflight_is_silent_when_pinned(tmp_path):
    assert ae.preflight_messages({"PATH": str(tmp_path / "empty")}) == []


def test_preflight_notes_a_different_inherited_agent_loop(tmp_path):
    stale = _stale_agent_loop(tmp_path / "stale-bin")
    lines = ae.preflight_messages({"PATH": str(stale.parent)})
    assert len(lines) == 1
    assert not lines[0].startswith("Warning")
    assert str(stale) in lines[0] and "pinned to the running install" in lines[0]
    assert str(PACKAGE_DIR) in lines[0]


def test_preflight_warns_when_agents_resolve_elsewhere(tmp_path):
    stale = _stale_agent_loop(tmp_path / "stale-bin")

    def which(name, path=None):
        return str(stale)

    lines = ae.preflight_messages({"PATH": ""}, which=which)
    warning = [line for line in lines if line.startswith("Warning")]
    assert len(warning) == 1
    launcher = ae.agent_loop_launcher()
    assert str(stale) in warning[0] and str(launcher) in warning[0] and str(PACKAGE_DIR) in warning[0]


def test_preflight_warns_when_launcher_is_unavailable(tmp_path, monkeypatch):
    _fail_install(monkeypatch)
    stale = _stale_agent_loop(tmp_path / "stale-bin")
    lines = ae.preflight_messages({"PATH": str(stale.parent)})
    assert len(lines) == 1
    assert lines[0].startswith("Warning: agents will resolve agent-loop to " + str(stale))
    assert "launcher unavailable" in lines[0] and str(PACKAGE_DIR) in lines[0]


def test_preflight_never_raises(tmp_path):
    def which(name, path=None):
        raise RuntimeError("boom")

    lines = ae.preflight_messages({"PATH": ""}, which=which)
    assert lines == ["Warning: agent-loop executable preflight failed (RuntimeError)"]


def test_log_preflight_logs_and_never_blocks(tmp_path, monkeypatch):
    _fail_install(monkeypatch)
    logged = []
    monkeypatch.setattr("coding_review_agent_loop.logging.log", lambda _config, line: logged.append(line))
    ae.log_preflight(object())
    assert logged and logged[0].startswith("Warning: agents will resolve agent-loop")


def test_cli_startup_runs_the_executable_preflight():
    source = (PACKAGE_DIR / "cli.py").read_text(encoding="utf-8")
    capture = source.index("tool_provenance.capture_process_provenance(config)")
    assert "agent_executable.log_preflight(config)" in source[capture:capture + 300]
