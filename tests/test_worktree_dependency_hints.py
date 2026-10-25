"""Advisory --worktree-link/--worktree-setup remedy text (#1651).

Every remedy is presentation only: the tests compare outcomes, classification,
recorded results and exit codes with and without the hint context.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import coding_review_agent_loop.cli as cli_module
from coding_review_agent_loop import test_runtime as runtime
from coding_review_agent_loop.checks import _raise_for_gate_result
from coding_review_agent_loop.cli import main
from coding_review_agent_loop.config import default_run_worktree
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.local_test_evidence import (
    PRE_COLLECTION_LAUNCH_FAILURE_PREFIX,
    BrokerRunResult,
)
from coding_review_agent_loop.runner import run_foreground_test
from coding_review_agent_loop.workdir_guard import (
    partition_reported_tests_by_workdir,
    validate_test_commands_within_workdir,
    validate_test_observation_citations_within_workdir,
)
from coding_review_agent_loop.worktree_hints import (
    WORKTREE_DEPENDENCY_HINT_ENV,
    WORKTREE_DEPENDENCY_HINT_VALUE,
    WORKTREE_DEPENDENCY_REMEDY,
    dependency_hint_environment,
    dependency_remedy,
    with_remedy,
)

from agent_loop_helpers import make_config

TOKEN = "tok1651"


@pytest.fixture(autouse=True)
def _isolated_pytest_environment(monkeypatch):
    for name in (
        "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_XDIST_AUTO_NUM_WORKERS", "PYTEST_XDIST_WORKER",
        "PYTEST_XDIST_WORKER_COUNT", "PYTEST_XDIST_TESTRUNUID", "PYTEST_CURRENT_TEST",
        WORKTREE_DEPENDENCY_HINT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def _config(tmp_path, monkeypatch, *, shape):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    overrides = {}
    if shape != "explicit-dir":
        overrides = {
            "claude_dir": default_run_worktree("OWNER/REPO", "claude", TOKEN),
            "run_token": TOKEN,
            "default_checkout_stores": (("claude", tmp_path / "store"),),
            "worktree_links": (".venv",) if shape == "linked" else (),
        }
    return make_config(tmp_path, **overrides)


# --- hint context from parent-owned config -----------------------------------


@pytest.mark.parametrize("shape", ["unlinked", "linked", "explicit-dir"])
def test_hint_context_comes_from_parent_config(tmp_path, monkeypatch, shape):
    config = _config(tmp_path, monkeypatch, shape=shape)
    expected = shape == "unlinked"
    assert (dependency_remedy(config, config.claude_dir) == WORKTREE_DEPENDENCY_REMEDY) is expected
    assert bool(dependency_hint_environment(config, config.claude_dir)) is expected
    # A path that is not this run's worktree never gets the remedy.
    assert dependency_remedy(config, tmp_path / "elsewhere") is None


def test_remedy_names_both_flags_and_is_appended_once():
    assert "--worktree-link" in WORKTREE_DEPENDENCY_REMEDY
    assert "--worktree-setup" in WORKTREE_DEPENDENCY_REMEDY
    once = with_remedy("refused.", WORKTREE_DEPENDENCY_REMEDY)
    assert once.startswith("refused.")
    assert with_remedy(once, WORKTREE_DEPENDENCY_REMEDY) == once
    assert with_remedy("refused.", None) == "refused."


# --- run-tests client: fallback launch failures --------------------------------


def _cli_fixtures(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli_module, "record_launcher_health", lambda *a, **k: calls.append(("health", k.get("state"))))
    monkeypatch.setattr(cli_module, "_record_run_tests_result", lambda *a, **k: calls.append(("result", k.get("outcome"))))
    monkeypatch.chdir(tmp_path)
    return calls


def _run_both_ways(monkeypatch, capsys, argv):
    observed = {}
    for hinted in (False, True):
        if hinted:
            monkeypatch.setenv(WORKTREE_DEPENDENCY_HINT_ENV, WORKTREE_DEPENDENCY_HINT_VALUE)
        else:
            monkeypatch.delenv(WORKTREE_DEPENDENCY_HINT_ENV, raising=False)
        code = main(argv)
        observed[hinted] = (code, capsys.readouterr().err)
    return observed


def test_fallback_missing_interpreter_shows_remedy_only_with_hint(tmp_path, monkeypatch, capsys):
    calls = _cli_fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: None)
    missing = str(tmp_path / ".venv" / "bin" / "python")
    observed = _run_both_ways(monkeypatch, capsys, [
        "run-tests", "--timeout-seconds", "60", "--memory-dir", str(tmp_path / "memory"), "--",
        missing, "-m", "pytest", "-q",
    ])
    (plain_code, plain_err), (hint_code, hint_err) = observed[False], observed[True]
    assert plain_code == hint_code != 0
    assert WORKTREE_DEPENDENCY_REMEDY not in plain_err
    assert WORKTREE_DEPENDENCY_REMEDY in hint_err
    # Launcher health is recorded identically in both runs (the hint is not read).
    assert calls[: len(calls) // 2] == calls[len(calls) // 2:]
    # The same launch, run directly, is the unchanged OSError launch-failed result.
    result = run_foreground_test(
        [missing, "-m", "pytest"], cwd=tmp_path, timeout_seconds=60, echo_output=False,
        classify_pre_collection=True,
    )
    assert (result.outcome, result.inner_exec, result.pre_collection_launch_failure) == (
        "launch-failed", "failed", None,
    )


def _no_pytest_python(tmp_path):
    venv = tmp_path / "venv"
    done = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], capture_output=True, timeout=120,
    )
    if done.returncode != 0:
        pytest.skip("cannot create a venv without pytest")
    python = venv / "bin" / "python"
    if subprocess.run([str(python), "-c", "import pytest"], capture_output=True).returncode == 0:
        pytest.skip("venv unexpectedly provides pytest")
    return str(python)


def test_fallback_missing_pytest_shows_remedy_only_with_hint(tmp_path, monkeypatch, capsys):
    python = _no_pytest_python(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    (project / "test_x.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    calls = _cli_fixtures(project, monkeypatch)
    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: None)
    observed = _run_both_ways(monkeypatch, capsys, [
        "run-tests", "--timeout-seconds", "120", "--memory-dir", str(tmp_path / "memory"), "--",
        python, "-m", "pytest", "test_x.py",
    ])
    (plain_code, plain_err), (hint_code, hint_err) = observed[False], observed[True]
    assert plain_code == hint_code == 1
    reason = runtime.PRE_COLLECTION_NO_PYTEST_REASON
    assert f"pre-collection launch failure ({reason})" in plain_err
    assert f"pre-collection launch failure ({reason})" in hint_err
    assert WORKTREE_DEPENDENCY_REMEDY not in plain_err
    assert WORKTREE_DEPENDENCY_REMEDY in hint_err
    # The remedy follows the existing pre-collection diagnostic.
    assert hint_err.index("pre-collection launch failure") < hint_err.index(WORKTREE_DEPENDENCY_REMEDY)
    # A classified run records nothing either way.
    assert calls == []
    for hinted in (False, True):
        env = {WORKTREE_DEPENDENCY_HINT_ENV: WORKTREE_DEPENDENCY_HINT_VALUE} if hinted else {}
        result = run_foreground_test(
            [python, "-m", "pytest", "test_x.py"], cwd=project, timeout_seconds=120,
            echo_output=False, classify_pre_collection=True,
            env={**os.environ, **env},
        )
        assert (result.outcome, result.returncode, result.pre_collection_launch_failure) == (
            "failed", 1, reason,
        )


def test_generic_module_not_found_is_still_not_pre_collection(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()
    (project / "test_x.py").write_text("import not_a_real_module_1651\n", encoding="utf-8")
    monkeypatch.setenv(WORKTREE_DEPENDENCY_HINT_ENV, WORKTREE_DEPENDENCY_HINT_VALUE)
    result = run_foreground_test(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_x.py"], cwd=project,
        timeout_seconds=120, echo_output=False, classify_pre_collection=True,
    )
    assert "ModuleNotFoundError" in result.output_tail
    assert result.pre_collection_launch_failure is None
    calls = _cli_fixtures(project, monkeypatch)
    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: None)
    code = main([
        "run-tests", "--timeout-seconds", "120", "--memory-dir", str(tmp_path / "memory"), "--",
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_x.py",
    ])
    err = capsys.readouterr().err
    assert code != 0
    # An ordinary collection error is suite output, not a launch failure.
    assert WORKTREE_DEPENDENCY_REMEDY not in err
    assert ("result", "failed") in calls


# --- run-tests client: broker frames --------------------------------------------


FRAMES = {
    "pre-collection": BrokerRunResult(
        receipt_id="r1", execution_ref="e1", outcome="launch-failed", returncode=1, elapsed_seconds=1.0,
        wrapper_bootstrap="verified", inner_exec="started", suite_start="not-started",
        diagnostic=f"{PRE_COLLECTION_LAUNCH_FAILURE_PREFIX}: {runtime.PRE_COLLECTION_NO_PYTEST_REASON}",
    ),
    "launch-failed": BrokerRunResult(
        receipt_id="r2", execution_ref=None, outcome="launch-failed", returncode=None, elapsed_seconds=0.1,
        wrapper_bootstrap="verified", inner_exec="failed", suite_start="not-started",
        diagnostic="[Errno 2] No such file or directory: '.venv/bin/python'",
    ),
    "ordinary": BrokerRunResult(
        receipt_id="r3", execution_ref="e3", outcome="failed", returncode=1, elapsed_seconds=1.0,
        wrapper_bootstrap="verified", inner_exec="started", suite_start="unknown",
    ),
}


@pytest.mark.parametrize("frame", list(FRAMES))
def test_broker_frames_show_remedy_only_with_hint_and_unchanged_outcome(tmp_path, monkeypatch, capsys, frame):
    calls = _cli_fixtures(tmp_path, monkeypatch)

    class FakeBroker:
        def run(self, *args, **kwargs):
            return FRAMES[frame]

    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: FakeBroker())
    observed = _run_both_ways(monkeypatch, capsys, [
        "run-tests", "--timeout-seconds", "5", "--memory-dir", str(tmp_path / "memory"), "--",
        ".venv/bin/python", "-m", "pytest",
    ])
    (plain_code, plain_err), (hint_code, hint_err) = observed[False], observed[True]
    assert plain_code == hint_code
    assert calls[: len(calls) // 2] == calls[len(calls) // 2:]
    assert WORKTREE_DEPENDENCY_REMEDY not in plain_err
    assert (WORKTREE_DEPENDENCY_REMEDY in hint_err) is (frame != "ordinary")
    assert hint_err.replace(f"agent-loop: {WORKTREE_DEPENDENCY_REMEDY}\n", "") == plain_err
    if frame == "pre-collection":
        assert calls == []
        assert hint_err.index("pre-collection launch failure") < hint_err.index(WORKTREE_DEPENDENCY_REMEDY)


# --- parent-side presentation: test gate and workdir guard -------------------


class _GateResult:
    outcome = "launch-failed"
    inner_exec = "failed"
    diagnostic = "[Errno 2] No such file or directory: '.venv/bin/python'"
    output_tail = ""
    args = (".venv/bin/python", "-m", "pytest")
    overlap_rejected = False

    def __init__(self, cwd):
        self.cwd = cwd


@pytest.mark.parametrize("shape", ["unlinked", "linked", "explicit-dir"])
def test_coder_test_gate_launch_failure_names_remedy_only_for_unlinked_run_worktree(tmp_path, monkeypatch, shape):
    config = _config(tmp_path, monkeypatch, shape=shape)
    with pytest.raises(AgentLoopError) as caught:
        _raise_for_gate_result(_GateResult(config.claude_dir), config)
    message = str(caught.value)
    assert message.startswith("Local test command could not start; this is a launcher/bootstrap failure")
    assert (WORKTREE_DEPENDENCY_REMEDY in message) is (shape == "unlinked")


def test_workdir_guard_program_refusal_names_remedy_and_keeps_refusing(tmp_path):
    assigned = tmp_path / "checkout"
    assigned.mkdir()
    command = "/elsewhere/bin/run-suite tests/"
    with pytest.raises(AgentLoopError) as plain:
        validate_test_commands_within_workdir([command], assigned_workdir=assigned)
    with pytest.raises(AgentLoopError) as hinted:
        validate_test_commands_within_workdir(
            [command], assigned_workdir=assigned, dependency_hint=WORKTREE_DEPENDENCY_REMEDY,
        )
    assert WORKTREE_DEPENDENCY_REMEDY not in str(plain.value)
    assert str(hinted.value) == f"{plain.value} {WORKTREE_DEPENDENCY_REMEDY}"
    assert str(plain.value).startswith("Coder reported tests from outside the assigned checkout")
    # Receipt citations go through the same guard.
    citation = type("Citation", (), {"command": command})()
    with pytest.raises(AgentLoopError, match="--worktree-setup"):
        validate_test_observation_citations_within_workdir(
            [citation], assigned_workdir=assigned, dependency_hint=WORKTREE_DEPENDENCY_REMEDY,
        )
    # The partitioned display report refuses the same command with the remedy.
    with pytest.raises(AgentLoopError) as partitioned:
        partition_reported_tests_by_workdir(
            [command], assigned_workdir=assigned, dependency_hint=WORKTREE_DEPENDENCY_REMEDY,
        )
    assert str(partitioned.value).endswith(WORKTREE_DEPENDENCY_REMEDY)


def test_workdir_guard_non_program_refusal_gets_no_remedy(tmp_path):
    assigned = tmp_path / "checkout"
    assigned.mkdir()
    command = "python -m pytest /elsewhere/tests"
    with pytest.raises(AgentLoopError) as caught:
        validate_test_commands_within_workdir(
            [command], assigned_workdir=assigned, dependency_hint=WORKTREE_DEPENDENCY_REMEDY,
        )
    assert WORKTREE_DEPENDENCY_REMEDY not in str(caught.value)
    # An outside-only baseline still degrades to context, never a refusal.
    partition = partition_reported_tests_by_workdir(
        [command], assigned_workdir=assigned, dependency_hint=WORKTREE_DEPENDENCY_REMEDY,
    )
    assert partition.out_of_checkout == (command,)
    # In-checkout toolchain interpreters are still accepted.
    validate_test_commands_within_workdir(
        ["/usr/bin/python3 -m pytest tests/"], assigned_workdir=assigned,
        dependency_hint=WORKTREE_DEPENDENCY_REMEDY,
    )


def test_hint_text_contains_no_reserved_protocol_marker():
    from coding_review_agent_loop.protocol_markers import EXCLUDED_PROTOCOL_LITERALS

    assert WORKTREE_DEPENDENCY_HINT_ENV in EXCLUDED_PROTOCOL_LITERALS
    assert "<!--" not in WORKTREE_DEPENDENCY_REMEDY


def test_unrelated_path_shapes_never_raise(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, shape="unlinked")
    assert dependency_remedy(config, None) is None
    assert dependency_hint_environment(config, Path("relative/path")) == {}
