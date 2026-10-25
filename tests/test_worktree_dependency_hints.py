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
    assert dependency_hint_environment(config, config.claude_dir) == {
        WORKTREE_DEPENDENCY_HINT_ENV: WORKTREE_DEPENDENCY_HINT_VALUE if expected else ""
    }
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
    assert dependency_hint_environment(config, Path("relative/path")) == {WORKTREE_DEPENDENCY_HINT_ENV: ""}


# --- round 2 (#1651 review) --------------------------------------------------


def _git_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "t@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "T"], check=True)
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
    return root


def _broker_result(kind):
    from types import SimpleNamespace

    if kind == "missing-interpreter":
        return SimpleNamespace(
            outcome="launch-failed", returncode=None, elapsed_seconds=0.01, output_tail="",
            wrapper_bootstrap="verified", inner_exec="failed", suite_start="not-started",
            diagnostic="[Errno 2] No such file or directory: '.venv/bin/python'",
        )
    if kind == "missing-pytest":
        return SimpleNamespace(
            outcome="failed", returncode=1, elapsed_seconds=0.01,
            output_tail="/x/python: No module named pytest", wrapper_bootstrap="verified",
            inner_exec="started", suite_start="unknown",
            pre_collection_launch_failure=runtime.PRE_COLLECTION_NO_PYTEST_REASON,
        )
    return SimpleNamespace(
        outcome=kind, returncode=0 if kind == "passed" else 1, elapsed_seconds=0.01,
        output_tail="", wrapper_bootstrap="verified", inner_exec="started", suite_start="verified",
    )


def _broker_environment(server, **extra):
    return {
        **server.environment,
        "AGENT_LOOP_INVOCATION_ID": server.turn_id,
        "PATH": os.environ.get("PATH", ""),
        **extra,
    }


# [item-1] The advisory selector never enters evidence identity.


def test_hint_variable_never_splits_broker_identity_or_supersession(tmp_path):
    from coding_review_agent_loop.local_test_evidence import (
        ENVIRONMENT_EXCLUSIONS,
        EnvironmentIdentityRegistry,
        TestBrokerClient,
        TestBrokerServer,
        canonical_environment_bytes,
        reconcile_test_observations,
    )

    assert WORKTREE_DEPENDENCY_HINT_ENV in ENVIRONMENT_EXCLUSIONS
    base = {"PATH": "/usr/bin"}
    for value in (WORKTREE_DEPENDENCY_HINT_VALUE, "", "spoofed"):
        assert canonical_environment_bytes({**base, WORKTREE_DEPENDENCY_HINT_ENV: value}) == (
            canonical_environment_bytes(base)
        )

    repo = _git_repo(tmp_path / "repo")
    outcomes = ["failed", "passed", "passed", "passed"]
    registry = EnvironmentIdentityRegistry()
    server = TestBrokerServer(
        root=repo, turn_id="turn-1651",
        execute=lambda argv, cwd, timeout, environment, stream: _broker_result(outcomes.pop(0)),
        environment_registry=registry,
    ).start()
    argv = [sys.executable, "-m", "pytest", "tracked.txt", "-q"]
    try:
        for extra in (
            {WORKTREE_DEPENDENCY_HINT_ENV: WORKTREE_DEPENDENCY_HINT_VALUE},  # failure, hint present
            {},  # pass, hint stripped
            {WORKTREE_DEPENDENCY_HINT_ENV: "spoofed"},  # pass, hint changed
            {WORKTREE_DEPENDENCY_HINT_ENV: "spoofed", "TEST_INPUT": "other"},  # control
        ):
            TestBrokerClient(_broker_environment(server, **extra)).run(argv, timeout_seconds=5, cwd=repo)
        rows = list(server.journal)
    finally:
        server.stop()
    evidence = reconcile_test_observations(rows, registry=registry)
    failure, stripped, changed, control = evidence.observations
    assert stripped.environment_state == "equivalent"
    assert changed.environment_state == "equivalent"
    # The comparison is live: an ordinary variable still splits identity.
    assert control.environment_state == "different"
    assert failure.superseded_by == stripped.receipt_id
    assert evidence.authoritative_failures == ()


# [item-5] Parent-side broker presentation selected from parent config.


@pytest.mark.parametrize("kind", ["missing-interpreter", "missing-pytest", "failed"])
@pytest.mark.parametrize("shape", ["unlinked", "linked", "explicit-dir"])
def test_parent_broker_attaches_config_selected_advisory_without_changing_evidence(
    tmp_path, monkeypatch, kind, shape
):
    from coding_review_agent_loop.local_test_evidence import TestBrokerClient, TestBrokerServer
    from coding_review_agent_loop.runner import _broker_dependency_remedy

    config = _config(tmp_path, monkeypatch, shape=shape)
    # The Runner derives the broker remedy from the backend override env, which
    # the parent builds from its own config.
    remedy = _broker_dependency_remedy(dependency_hint_environment(config, config.claude_dir))
    assert (remedy == WORKTREE_DEPENDENCY_REMEDY) is (shape == "unlinked")
    # The broker socket lives under the real TMPDIR (AF_UNIX path length).
    monkeypatch.setattr(tempfile, "tempdir", None)
    repo = _git_repo(tmp_path / "repo")
    argv = [sys.executable, "-m", "pytest", "tracked.txt"]

    def run(dependency_remedy):
        server = TestBrokerServer(
            root=repo, turn_id=f"turn-{dependency_remedy is not None}",
            execute=lambda *_args: _broker_result(kind), dependency_remedy=dependency_remedy,
        ).start()
        try:
            # The client's own selector is stripped: the advisory must come from the parent.
            result = TestBrokerClient(_broker_environment(server)).run(argv, timeout_seconds=5, cwd=repo)
            journal = [(row.outcome, row.caveats, row.diagnostic) for row in server.journal]
        finally:
            server.stop()
        return result, journal

    hinted, hinted_journal = run(remedy)
    plain, plain_journal = run(None)
    expected = shape == "unlinked" and kind != "failed"
    assert (hinted.advisory == WORKTREE_DEPENDENCY_REMEDY) is expected
    assert plain.advisory == ""
    for field in ("outcome", "returncode", "inner_exec", "suite_start", "diagnostic", "output_tail"):
        assert getattr(hinted, field) == getattr(plain, field)
    assert hinted_journal == plain_journal
    assert all(WORKTREE_DEPENDENCY_REMEDY not in repr(row) for row in hinted_journal)
    assert WORKTREE_DEPENDENCY_REMEDY not in hinted.diagnostic


def test_client_prints_parent_advisory_even_with_its_selector_stripped(tmp_path, monkeypatch, capsys):
    from dataclasses import replace

    calls = _cli_fixtures(tmp_path, monkeypatch)
    frame = replace(FRAMES["launch-failed"], advisory=WORKTREE_DEPENDENCY_REMEDY)

    class FakeBroker:
        def run(self, *args, **kwargs):
            return frame

    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: FakeBroker())
    monkeypatch.delenv(WORKTREE_DEPENDENCY_HINT_ENV, raising=False)
    code = main([
        "run-tests", "--timeout-seconds", "5", "--memory-dir", str(tmp_path / "memory"), "--",
        ".venv/bin/python", "-m", "pytest",
    ])
    err = capsys.readouterr().err
    assert err.count(WORKTREE_DEPENDENCY_REMEDY) == 1
    # With both the frame advisory and the selector, the line is printed once.
    monkeypatch.setenv(WORKTREE_DEPENDENCY_HINT_ENV, WORKTREE_DEPENDENCY_HINT_VALUE)
    assert main([
        "run-tests", "--timeout-seconds", "5", "--memory-dir", str(tmp_path / "memory"), "--",
        ".venv/bin/python", "-m", "pytest",
    ]) == code
    assert capsys.readouterr().err.count(WORKTREE_DEPENDENCY_REMEDY) == 1
    assert calls[: len(calls) // 2] == calls[len(calls) // 2:]


# [item-3] An inherited selector is neutralized when the condition is false.


@pytest.mark.parametrize("shape", ["unlinked", "linked", "explicit-dir"])
def test_inherited_selector_is_neutralized_in_the_effective_child_environment(tmp_path, monkeypatch, shape):
    from coding_review_agent_loop.containment import default_policy
    from coding_review_agent_loop.runner import Runner
    from coding_review_agent_loop.worktree_hints import dependency_hint_requested

    config = _config(tmp_path, monkeypatch, shape=shape)
    monkeypatch.setenv(WORKTREE_DEPENDENCY_HINT_ENV, WORKTREE_DEPENDENCY_HINT_VALUE)
    child_env = dependency_hint_environment(config, config.claude_dir)
    monkeypatch.setattr(tempfile, "tempdir", None)
    runner = Runner(containment_policy=default_policy(mode="off", cache_dir=tmp_path / ".runtime"))
    workdir = tmp_path / "child-cwd"
    workdir.mkdir()
    result = runner.run_with_log(
        [sys.executable, "-c", f"import os; print(repr(os.environ.get({WORKTREE_DEPENDENCY_HINT_ENV!r})))"],
        cwd=workdir, log_path=tmp_path / "child.log", label="probe", progress_interval_seconds=1,
        env=child_env,
    )
    seen = eval(result.stdout.strip().splitlines()[-1])  # noqa: S307 - repr of a str or None
    assert dependency_hint_requested({WORKTREE_DEPENDENCY_HINT_ENV: seen}) is (shape == "unlinked")


# [item-4] Free-form response guards carry the remedy too.


def test_freeform_response_guard_names_remedy_and_keeps_refusing(tmp_path):
    from coding_review_agent_loop.workdir_guard import validate_response_tests_within_workdir

    assigned = tmp_path / "checkout"
    assigned.mkdir()
    text = "Fixed it.\nTests: `/elsewhere/bin/run-suite tests/`\n"
    with pytest.raises(AgentLoopError) as plain:
        validate_response_tests_within_workdir(text, assigned_workdir=assigned)
    with pytest.raises(AgentLoopError) as hinted:
        validate_response_tests_within_workdir(
            text, assigned_workdir=assigned, dependency_hint=WORKTREE_DEPENDENCY_REMEDY,
        )
    assert WORKTREE_DEPENDENCY_REMEDY not in str(plain.value)
    assert str(hinted.value) == f"{plain.value} {WORKTREE_DEPENDENCY_REMEDY}"


@pytest.mark.parametrize("shape", ["unlinked", "linked"])
def test_task_post_pr_freeform_report_names_remedy_from_config(tmp_path, monkeypatch, shape):
    from coding_review_agent_loop.response_validation import _validate_response_tests_with_post_pr_context

    from agent_loop_helpers import FakeRunner

    config = _config(tmp_path, monkeypatch, shape=shape)
    with pytest.raises(AgentLoopError) as caught:
        _validate_response_tests_with_post_pr_context(
            "Done.\nTests: `/elsewhere/bin/run-suite tests/`\n",
            runner=FakeRunner(), config=config, pr_number=77,
        )
    message = str(caught.value)
    assert "Coder reported tests from outside the assigned checkout" in message
    assert (WORKTREE_DEPENDENCY_REMEDY in message) is (shape == "unlinked")


# [item-2] Pushed-branch recovery on terminal no-PR stops.


def _poisoned_after_push_runner(config, *, poison=True, **kwargs):
    from coding_review_agent_loop import checkout_verification as cv

    from agent_loop_helpers import FakeRunner

    class PoisoningRunner(FakeRunner):
        def run_with_log(self, args, *a, **kw):
            result = super().run_with_log(args, *a, **kw)
            if poison and list(args)[:1] == [config.claude_cmd]:
                # What the backend's post-turn baseline refresh does after the
                # coder pushed its branch and replaced a link.
                cv.poison_checkout(config.claude_dir, "baseline capture after agent turn failed: link replaced")
            return result

    return PoisoningRunner(**kwargs)


@pytest.mark.parametrize("poison, pushed", [(True, True), (True, False), (False, True)])
def test_direct_issue_terminal_no_pr_stop_prints_recovery_only_when_poisoned_after_push(
    tmp_path, monkeypatch, poison, pushed
):
    from coding_review_agent_loop import checkout_verification as cv
    from coding_review_agent_loop import issue_loop, worktree_hints
    from coding_review_agent_loop.issue_loop import run_issue_loop

    config = _config(tmp_path, monkeypatch, shape="unlinked")
    # The fake runner cannot build a real shared store; the per-run worktree
    # condition still comes from the config.
    monkeypatch.setattr(issue_loop, "ensure_agent_workdirs", lambda *_a, **_k: None)
    monkeypatch.setattr(issue_loop, "sync_coder_base_before_implementation", lambda *_a, **_k: None)
    branch = "agent-loop/managed-56"
    monkeypatch.setattr(worktree_hints, "_pushed_branch", lambda *_a: branch if pushed else None)
    runner = _poisoned_after_push_runner(
        config, poison=poison,
        claude_outputs=["Created something.\n<!-- AGENT_STATE: blocking -->"],
    )
    try:
        with pytest.raises(AgentLoopError) as caught:
            run_issue_loop(runner, issue_number=56, config=config)
        message = str(caught.value)
        assert type(caught.value) is AgentLoopError
        assert "valid PR" in message
        recovered = f"agent-loop managed-pr --head {branch} --expected-closing-issue 56" in message
        assert recovered is (poison and pushed)
        assert ("worktree remove --force" in message) is (poison and pushed)
        if poison:
            assert cv.has_poisoned_entry(config.claude_dir)
    finally:
        cv.reset_checkout_baselines()


def test_approved_plan_terminal_no_pr_stop_prints_recovery_after_push(tmp_path, monkeypatch):
    from coding_review_agent_loop import checkout_verification as cv
    from coding_review_agent_loop import worktree_hints

    from coverage_reask_helpers import CoverageHarness, response_text

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    run_dir = default_run_worktree("OWNER/REPO", "claude", TOKEN)
    harness = CoverageHarness(
        tmp_path, monkeypatch, claude_dir=run_dir, run_token=TOKEN,
        default_checkout_stores=(("claude", tmp_path / "store"),),
    )
    branch = "agent-loop/managed-56"
    monkeypatch.setattr(worktree_hints, "_pushed_branch", lambda *_a: branch)

    def coder_turn(h):
        cv.poison_checkout(h.repo, "baseline capture after agent turn failed: link replaced")
        return h.response(response_text(pr_number=None))

    harness.script = [coder_turn]
    try:
        with pytest.raises(AgentLoopError) as caught:
            harness.run()
        message = str(caught.value)
        assert type(caught.value) is AgentLoopError
        assert "valid PR" in message
        assert f"agent-loop managed-pr --head {branch} --expected-closing-issue 56" in message
        assert f"worktree remove --force {run_dir}" in message
        assert harness.run_pr_calls == []
        assert run_dir.is_dir()
    finally:
        cv.reset_checkout_baselines()
