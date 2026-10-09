"""Configured test interpreter, bounded interpreter probes and render-scoped checks (#1343)."""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path

import pytest

from _proc_probe import wait_until_gone

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop import lifecycle_probe as lp
from coding_review_agent_loop import prompts
from coding_review_agent_loop import test_runtime as runtime
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import config_from_args
from coding_review_agent_loop.errors import AgentLoopError

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX shell fixtures")


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _pid_alive(pid: int) -> bool:
    return not wait_until_gone(pid, timeout=5.0)


# ------------------------------------------------------------ probe-bounded


@posix_only
@pytest.mark.parametrize("code", [0, 3, 4, 7])
def test_exit_code_probe_returns_the_exit_code(tmp_path, code):
    fake = _script(tmp_path / "python", f"#!/bin/sh\nexit {code}\n")
    assert runtime.run_exit_code_probe([str(fake)], env=None, cwd=tmp_path, timeout_seconds=5) == code


@posix_only
def test_exit_code_probe_times_out_and_kills_the_tree(tmp_path):
    pid_file = tmp_path / "child.pid"
    fake = _script(tmp_path / "python", f"#!/bin/sh\nsleep 60 &\necho $! > {pid_file}\nsleep 60\n")
    started = time.monotonic()
    assert runtime.run_exit_code_probe([str(fake)], env=None, cwd=tmp_path, timeout_seconds=1) is None
    assert time.monotonic() - started < 10
    assert not _pid_alive(int(pid_file.read_text().strip()))


@posix_only
@pytest.mark.parametrize("code", [0, 4])
def test_exit_code_probe_kills_a_descendant_after_a_normal_exit(tmp_path, code):
    # The direct child exits at once but leaves a long-lived descendant that
    # holds the inherited descriptors; the probe returns promptly and leaves
    # nothing live.
    pid_file = tmp_path / "child.pid"
    fake = _script(
        tmp_path / "python",
        f"#!/bin/sh\nsleep 60 &\necho $! > {pid_file}\nexit {code}\n",
    )
    started = time.monotonic()
    assert runtime.run_exit_code_probe([str(fake)], env=None, cwd=tmp_path, timeout_seconds=10) == code
    assert time.monotonic() - started < 8
    assert not _pid_alive(int(pid_file.read_text().strip()))


@posix_only
def test_exit_code_probe_discards_a_flood_of_output(tmp_path):
    fake = _script(
        tmp_path / "python",
        "#!/bin/sh\nhead -c 8000000 /dev/zero\nhead -c 8000000 /dev/zero 1>&2\nexit 0\n",
    )
    assert runtime.run_exit_code_probe([str(fake)], env=None, cwd=tmp_path, timeout_seconds=20) == 0


def test_exit_code_probe_missing_executable_is_unknown_and_never_raises(tmp_path):
    missing = tmp_path / "no-such-python"
    assert runtime.run_exit_code_probe([str(missing)], env=None, cwd=tmp_path, timeout_seconds=5) is None
    assert runtime.run_exit_code_probe([], env=None, cwd=tmp_path, timeout_seconds=5) is None


@posix_only
def test_exit_code_probe_never_invokes_a_shell(tmp_path):
    marker = tmp_path / "marker"
    fake = _script(tmp_path / "python", "#!/bin/sh\nexit 0\n")
    # A shell would execute the substitution and create the marker.
    assert runtime.run_exit_code_probe(
        [str(fake), f"$(touch {marker})"], env=None, cwd=tmp_path, timeout_seconds=5,
    ) == 0
    assert not marker.exists()


@posix_only
def test_run_check_migration_keeps_the_contract_and_cleans_descendants(tmp_path):
    pid_file = tmp_path / "child.pid"
    fake = _script(tmp_path / "python", f"#!/bin/sh\nsleep 60 &\necho $! > {pid_file}\nexit 3\n")
    assert lp._run_check((str(fake),), "pass", dict(os.environ), tmp_path, 5.0) == 3
    assert not _pid_alive(int(pid_file.read_text().strip()))


def test_pytest_absent_classification_is_unchanged(tmp_path):
    # The real independent import check through the migrated probe: a
    # checkout-local pytest that is not importable is still "pytest-absent".
    (tmp_path / "pytest.py").write_text("raise ImportError('no pytest here')\n", encoding="utf-8")
    argv = [sys.executable, "-m", "pytest", "-q"]
    eligibility = lp.launch_eligibility_for(argv, dict(os.environ), tmp_path)
    assert eligibility is not None and eligibility.shape == "module"
    # find_spec finds the shadow, so this is "unavailable", exactly as before;
    # with no pytest at all the result is "pytest-absent".
    assert lp.run_independent_import_check(eligibility, argv, dict(os.environ), tmp_path) == "unavailable"
    empty = tmp_path / "empty-path"
    empty.mkdir()
    isolated = {**os.environ, "PYTHONNOUSERSITE": "1"}
    isolated.pop("PYTHONPATH", None)
    code = lp._run_check(
        (sys.executable, "-S"), lp._IMPORT_CHECK_CODE, isolated, empty, lp.IMPORT_CHECK_TIMEOUT_SECONDS,
    )
    assert code == 3


# ------------------------------------------------------- xdist import check


def test_xdist_check_really_imports_each_module(tmp_path):
    importable = tmp_path / "importable"
    (importable / "xdist").mkdir(parents=True)
    (importable / "xdist" / "__init__.py").write_text("", encoding="utf-8")
    (importable / "xdist" / "plugin.py").write_text("", encoding="utf-8")
    assert lp.xdist_import_check((sys.executable,), dict(os.environ), importable) == lp.XDIST_IMPORTABLE

    broken = tmp_path / "broken"
    (broken / "xdist").mkdir(parents=True)
    (broken / "xdist" / "__init__.py").write_text("", encoding="utf-8")
    # Discoverable but unimportable: find_spec succeeds, the import raises.
    (broken / "xdist" / "plugin.py").write_text("raise ImportError('broken plugin')\n", encoding="utf-8")
    assert lp.xdist_import_check((sys.executable,), dict(os.environ), broken) == lp.XDIST_MISSING

    no_pytest = tmp_path / "no-pytest"
    no_pytest.mkdir()
    (no_pytest / "pytest.py").write_text("raise ImportError('broken pytest')\n", encoding="utf-8")
    assert lp.xdist_import_check((sys.executable,), dict(os.environ), no_pytest) == lp.XDIST_PYTEST_MISSING


@posix_only
@pytest.mark.parametrize(("code", "expected"), [(0, 0), (3, 3), (4, 4), (1, None), (2, None)])
def test_xdist_check_maps_other_exit_codes_to_unknown(tmp_path, code, expected):
    fake = _script(tmp_path / "python", f"#!/bin/sh\nexit {code}\n")
    assert lp.xdist_import_check((str(fake),), dict(os.environ), tmp_path) == expected


# ------------------------------------------------- invalid-test-python-config


def _parse(tmp_path, *extra):
    parser = build_parser()
    args = parser.parse_args(
        [
            "pr", "123", "--repo", "OWNER/REPO", "--coder", "claude", "--reviewer", "codex",
            "--claude-dir", str(tmp_path / "claude"), "--codex-dir", str(tmp_path / "codex"),
            "--dangerous-agent-permissions", *extra,
        ]
    )
    return args


@posix_only
def test_valid_absolute_interpreter_is_accepted_from_the_cli(tmp_path):
    fake = _script(tmp_path / "venv" / "bin" / "python", "#!/bin/sh\nexit 0\n")
    args = _parse(tmp_path, "--test-python", str(fake))
    assert args.test_python == str(fake)
    assert make_config(tmp_path, test_python=str(fake)).test_python == str(fake)


@pytest.mark.parametrize("kind", ["relative", "missing", "directory", "not-executable", "blank"])
def test_invalid_test_python_is_rejected_naming_the_option(tmp_path, kind):
    if kind == "relative":
        value = "venv/bin/python"
    elif kind == "missing":
        value = str(tmp_path / "missing" / "python")
    elif kind == "directory":
        (tmp_path / "dir").mkdir()
        value = str(tmp_path / "dir")
    elif kind == "not-executable":
        path = tmp_path / "python"
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o644)
        value = str(path)
    else:
        value = "  "
    with pytest.raises(AgentLoopError, match="--test-python"):
        make_config(tmp_path, test_python=value)


def test_relative_value_never_falls_back_to_path_discovery(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent))
    with pytest.raises(AgentLoopError, match="absolute"):
        make_config(tmp_path, test_python=Path(sys.executable).name)


def test_config_from_args_carries_test_python(tmp_path):
    fake = tmp_path / "venv" / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.symlink_to(sys.executable)
    args = _parse(tmp_path, "--test-python", str(fake))
    config = config_from_args(args, FakeRunner())
    assert config.test_python == str(fake)


# --------------------------------------------- interpreter-identity-and-cache


def test_lexical_symlink_identity_is_kept_everywhere(tmp_path, monkeypatch):
    real = _script(tmp_path / "real" / "python3.12", "#!/bin/sh\nexit 0\n")
    first = tmp_path / "venv-a" / "bin" / "python"
    second = tmp_path / "venv-b" / "bin" / "python"
    for link in (first, second):
        link.parent.mkdir(parents=True)
        link.symlink_to(real)
    config = make_config(tmp_path, test_python=str(first))
    other = make_config(tmp_path, test_python=str(second))
    # Stored lexically: both share one realpath but keep their own paths.
    assert config.test_python == str(first) and other.test_python == str(second)
    assert os.path.realpath(config.test_python) == os.path.realpath(other.test_python)
    # Exported lexically for every backend.
    for provider in ("claude", "codex", "gemini", "antigravity"):
        assert runtime.agent_shell_environment(config, provider, "coder", ambient={})[
            runtime.ENV_TEST_PYTHON
        ] == str(first)
    # Invoked and granted lexically.
    monkeypatch.setattr(runtime, "verified_wrapper_prefix", lambda **_kwargs: ("/w/python", "-m", "w"))
    runtime.reset_coder_test_invocations()
    try:
        invocation = runtime.resolve_coder_test_invocation(config, cwd=tmp_path)
        assert invocation.endswith(f"-- {first} -m pytest")
        assert str(real) not in invocation
    finally:
        runtime.reset_coder_test_invocations()
    seen = []
    monkeypatch.setattr(lp, "_run_check", lambda invocation, *_a, **_k: seen.append(tuple(invocation)) or 0)
    prompts.configured_interpreter_guidance(config)
    assert seen == [(str(first),)]


@posix_only
def test_each_render_reruns_the_check_without_a_cache(tmp_path):
    state = tmp_path / "state"
    state.write_text("4", encoding="utf-8")
    fake = _script(tmp_path / "venv" / "bin" / "python", f"#!/bin/sh\nexit $(cat {state})\n")
    mtime = fake.stat().st_mtime_ns
    config = make_config(tmp_path, test_python=str(fake))
    first = prompts.parallel_test_worker_guidance(config)
    assert "could not import pytest-xdist" in first
    # Packages change; the executable itself does not.
    state.write_text("0", encoding="utf-8")
    assert fake.stat().st_mtime_ns == mtime
    second = prompts.parallel_test_worker_guidance(config)
    assert "found pytest and pytest-xdist importable" in second
    assert "could not import" not in second
