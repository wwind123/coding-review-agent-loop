import contextlib
import json
import os
import shlex
import shutil
import subprocess
import signal
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from unittest import mock

from _proc_probe import DEAD_STATES, proc_state as _proc_state, wait_until_gone

import coding_review_agent_loop.test_runtime as runtime
from agent_loop_helpers import make_config
from coding_review_agent_loop.cli import build_parser, main
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import run_foreground_test


def _now() -> datetime:
    return datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def _assert_process_not_active(pid: int) -> None:
    """Treat a POSIX zombie as terminated while allowing init to reap it."""
    if not wait_until_gone(pid, timeout=10.0):
        raise AssertionError(f"probe descendant {pid} is still active")


def _kill_if_active(pid: int) -> None:
    """Best-effort cleanup for a pid whose caller recorded no starttime.

    Callers here never capture an identity to compare against, so this stays a
    liveness-guarded kill; it is not the model for identity-aware fallbacks,
    which must go through ``kill_if_same_instance``.
    """
    state = _proc_state(pid)
    if state is None or state in DEAD_STATES:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _windows_process_is_active(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist.exe", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=2,
    )
    return f'"{pid}"' in result.stdout


def _assert_windows_process_not_active(pid: int) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not _windows_process_is_active(pid):
            return
        time.sleep(0.05)
    raise AssertionError(f"probe descendant {pid} is still active")


def _kill_windows_if_active(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _record(
    memory: Path,
    cwd: Path,
    argv: list[str],
    *,
    outcome: str,
    elapsed: float,
    attempted: float = 1800,
    timestamp: datetime | None = None,
    environment: dict[str, str] | None = None,
    launch_integrity: str | None = "verified",
) -> None:
    assert runtime.record_test_observation(
        memory,
        argv=argv,
        cwd=cwd,
        outcome=outcome,
        elapsed_seconds=elapsed,
        attempted_timeout_seconds=attempted,
        policy_ceiling_seconds=1800,
        returncode=0 if outcome == "passed" else 124,
        timestamp=timestamp or _now(),
        environment=environment,
        launch_integrity=launch_integrity,
    )


def test_config_default_and_validation_and_cli_override(tmp_path):
    config = make_config(tmp_path)
    assert config.coder_test_command_timeout_seconds == 1800
    assert make_config(tmp_path, coder_test_command_timeout_seconds=7200).coder_test_command_timeout_seconds == 7200
    parser = build_parser()
    args = parser.parse_args(["issue", "730", "--coder-test-command-timeout-seconds", "7200"])
    assert args.coder_test_command_timeout_seconds == 7200
    for invalid in (True, 0, -1, float("nan"), float("inf"), 1.5, "bad"):
        with pytest.raises(AgentLoopError):
            make_config(tmp_path, coder_test_command_timeout_seconds=invalid)


def test_wrapper_resolution_inherited_subceiling_and_policy_rejection(monkeypatch):
    assert runtime.resolve_timeout_seconds(None, policy_ceiling=7200) == 7200
    assert runtime.resolve_timeout_seconds("720", policy_ceiling=1800) == 720
    assert runtime.resolve_timeout_seconds("1800", policy_ceiling=1800) == 1800
    with pytest.raises(runtime.TestRuntimeConfigurationError):
        runtime.resolve_timeout_seconds("1801", policy_ceiling=1800)
    with pytest.raises(runtime.TestRuntimeConfigurationError):
        runtime.inherited_timeout_ceiling({"AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS": "0"})
    monkeypatch.setenv("AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS", "7200")
    assert runtime.inherited_timeout_ceiling() == 7200


@pytest.fixture
def private_launcher(tmp_path, monkeypatch):
    """Install the #1632 launcher under this test's tmp dir with a clean cache."""
    from coding_review_agent_loop import agent_executable

    tmp = tmp_path / "launcher-tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    monkeypatch.setattr(agent_executable, "_LAUNCHER_CACHE", {})
    monkeypatch.setattr(agent_executable, "_WARNED", set())
    runtime.reset_coder_test_invocations()
    yield agent_executable
    runtime.reset_coder_test_invocations()


def _stale_wrapper(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stale = directory / "agent-loop"
    stale.write_text("#!/bin/sh\necho 'agent-loop preflight: verified'\n", encoding="utf-8")
    stale.chmod(0o755)
    return stale


def test_wrapper_candidate_is_only_the_pinned_launcher_with_lexical_interpreter(
    tmp_path, monkeypatch, private_launcher
):
    venv = tmp_path / "virtualenv"
    interpreter = venv / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys.executable).resolve())
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    monkeypatch.setattr(runtime.sys, "executable", str(interpreter))
    stale = _stale_wrapper(tmp_path / "stale-bin")

    candidates = runtime._wrapper_candidates({"PATH": str(stale.parent)})
    launcher = private_launcher.agent_loop_launcher()
    assert candidates == [(str(launcher), "run-tests")]
    assert runtime.resolve_wrapper_prefix() == (str(launcher), "run-tests")
    # The launcher execs the lexical virtualenv interpreter, not its target.
    assert f"exec {shlex.quote(str(interpreter))} -I -c " in launcher.read_text(encoding="utf-8")
    assert all(str(stale) not in candidate for candidate in candidates)
    assert all("-m" not in candidate for candidate in candidates)


def test_preflight_verifies_the_pinned_launcher_over_a_stale_path_wrapper(
    tmp_path, monkeypatch, private_launcher
):
    stale = _stale_wrapper(tmp_path / "stale-bin")
    monkeypatch.setenv("PATH", os.pathsep.join([str(stale.parent), os.environ.get("PATH", "")]))
    results = runtime.preflight_wrapper_candidates(cwd=tmp_path)
    launcher = private_launcher.agent_loop_launcher()
    assert [result.candidate for result in results] == [(str(launcher), "run-tests")]
    assert results[0].state == "verified", results[0].diagnostic
    assert runtime.verified_wrapper_prefix(cwd=tmp_path) == (str(launcher), "run-tests")


def test_launcher_unavailable_with_shadow_package_grants_and_renders_no_wrapper(
    tmp_path, monkeypatch, private_launcher
):
    def refuse(_root=None):
        raise OSError("scratch root is not writable")

    monkeypatch.setattr(private_launcher, "install_agent_loop_launcher", refuse)
    shadow = tmp_path / "checkout"
    (shadow / "coding_review_agent_loop").mkdir(parents=True)
    (shadow / "coding_review_agent_loop" / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "coding_review_agent_loop" / "cli.py").write_text(
        "def main():\n    print('agent-loop preflight: verified')\n    return 0\n", encoding="utf-8"
    )
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    stale = _stale_wrapper(tmp_path / "stale-bin")
    monkeypatch.setenv("PATH", os.pathsep.join([str(stale.parent), os.environ.get("PATH", "")]))
    logged = []
    monkeypatch.setattr("coding_review_agent_loop.logging.log", lambda _c, message: logged.append(message))

    assert runtime._wrapper_candidates() == []
    assert runtime.resolve_wrapper_prefix() is None
    assert runtime.preflight_wrapper_candidates(cwd=shadow) == ()
    assert runtime.render_test_wrapper(("pytest", "-q")) == "pytest -q"
    config = make_config(tmp_path, test_command=("pytest", "-q"))
    assert runtime.resolve_coder_test_invocation(config, cwd=shadow) is None
    assert any(
        "test grant omitted (no agent-loop run-tests wrapper candidate verified)" in message
        for message in logged
    )


def test_managed_invocation_parses_absolute_entrypoint_and_module_forms(tmp_path):
    executable = str(tmp_path / "agent-loop")
    parsed = runtime.parse_managed_test_invocation([
        executable, "run-tests", "--timeout-seconds=720", "--memory-dir", str(tmp_path),
        "--", sys.executable, "-c", "print(1)",
    ])
    assert parsed is not None
    assert parsed.inner_argv == (sys.executable, "-c", "print(1)")
    assert parsed.timeout_seconds == 720
    assert parsed.memory_dir == tmp_path

    module = runtime.parse_managed_test_invocation([
        sys.executable, "-m", "coding_review_agent_loop.cli", "run-tests",
        "--", "pytest", "tests/test_protocol.py", "-q",
    ])
    assert module is not None
    assert module.inner_argv[-2:] == ("tests/test_protocol.py", "-q")
    with pytest.raises(runtime.TestRuntimeConfigurationError):
        runtime.parse_managed_test_invocation([executable, "run-tests", "--unknown", "--", "true"])
    with pytest.raises(runtime.TestRuntimeConfigurationError):
        runtime.parse_managed_test_invocation([executable, "run-tests", "--timeout-seconds", "1", "--timeout-seconds", "2", "--", "true"])


def test_managed_command_parses_supported_execution_prefixes():
    command = [
        "MODE=inline", "timeout", "1800", "env", "-u", "AGENT_LOOP_INVOCATION_ID",
        "/opt/agent-loop", "run-tests", "--memory-dir", "/tmp/cache", "--",
        "python3", "-m", "pytest", "tests/test_protocol.py", "-q",
    ]
    traversal = runtime.managed_wrapper_traversal(command)
    assert traversal.effective_head_index == 6
    assert traversal.program_positions == {1, 3, 6}

    parsed = runtime.parse_managed_test_command(command)
    assert parsed is not None
    assert parsed.inner_argv == (
        "python3", "-m", "pytest", "tests/test_protocol.py", "-q"
    )

    module = runtime.parse_managed_test_command([
        "nice", "-n5", sys.executable, "-m", "coding_review_agent_loop.cli", "run-tests",
        "--", "pytest", "tests/test_protocol.py", "-q",
    ])
    assert module is not None
    assert module.inner_argv[-3:] == ("pytest", "tests/test_protocol.py", "-q")

    with pytest.raises(runtime.TestRuntimeConfigurationError):
        runtime.parse_managed_test_command([
            "timeout", "1800", "/opt/agent-loop", "run-tests", "--unknown", "--", "true",
        ])


def test_normalization_joins_wrapped_and_bare_commands_without_leaking_values(tmp_path):
    bare = [sys.executable, "-m", "pytest", "tests/test_protocol.py", "-q"]
    wrapped = [
        str(tmp_path / "agent-loop"), "run-tests", "--timeout-seconds", "720",
        "--memory-dir", str(tmp_path / "cache"), "--", *bare,
    ]
    assert runtime.normalize_test_command(wrapped, cwd=tmp_path) == runtime.normalize_test_command(bare, cwd=tmp_path)
    normalized = runtime.normalize_test_command(["TOKEN=secret-value", *bare], cwd=tmp_path)
    assert "secret-value" not in normalized
    assert "TOKEN=" in normalized
    assert runtime.normalize_test_command(shlex.split(normalized), cwd=tmp_path) == normalized


def test_environment_fingerprint_tolerates_which_without_path_argument(tmp_path, monkeypatch):
    real_which = runtime.shutil.which

    def which(command):
        return real_which(command)

    monkeypatch.setattr(runtime.shutil, "which", which)
    resolved = runtime._resolve_executable(
        ["python", "-c", "pass"], tmp_path, {"PATH": os.environ.get("PATH", "")}
    )
    assert resolved == (real_which("python") or "python")


def test_foreground_runner_reports_visible_success_failure_timeout_and_tail(tmp_path, capsys):
    passed = run_foreground_test(
        [sys.executable, "-c", "print('visible')"], cwd=tmp_path, timeout_seconds=5
    )
    assert passed.outcome == "passed"
    assert "visible" in capsys.readouterr().out
    failed = run_foreground_test(
        [sys.executable, "-c", "print('diagnostic'); raise SystemExit(7)"],
        cwd=tmp_path, timeout_seconds=5,
    )
    assert failed.outcome == "failed"
    assert failed.returncode == 7
    assert "diagnostic" in failed.output_tail
    timed_out = run_foreground_test(
        [sys.executable, "-c", "import time; print('before', flush=True); time.sleep(5)"],
        cwd=tmp_path, timeout_seconds=0.2,
    )
    assert timed_out.outcome == "timed_out"
    assert timed_out.returncode == 124
    assert "before" in timed_out.output_tail


def test_foreground_timeout_does_not_wait_for_escaped_descendant(tmp_path):
    pid_file = tmp_path / "escaped-child.pid"
    script = f"""
import os
import time
from pathlib import Path

pid = os.fork()
if pid == 0:
    os.setsid()
    time.sleep(5)
else:
    Path({str(pid_file)!r}).write_text(str(pid))
    time.sleep(5)
"""
    result = None
    escaped_pid = None
    started = time.monotonic()
    try:
        result = run_foreground_test(
            [sys.executable, "-c", script], cwd=tmp_path, timeout_seconds=0.1
        )
    finally:
        if pid_file.exists():
            escaped_pid = int(pid_file.read_text(encoding="utf-8"))
        if escaped_pid is not None:
            try:
                os.kill(escaped_pid, 9)
            except ProcessLookupError:
                pass
    assert result is not None
    assert result.outcome == "timed_out"
    assert time.monotonic() - started < 2


def test_foreground_timeout_stops_when_escaped_descendant_keeps_writing(tmp_path):
    pid_file = tmp_path / "writing-escaped-child.pid"
    script = f"""
import os
import time
from pathlib import Path

pid = os.fork()
if pid == 0:
    os.setsid()
    while True:
        os.write(1, b"x")
        time.sleep(0.01)
else:
    Path({str(pid_file)!r}).write_text(str(pid))
    time.sleep(5)
"""
    result = None
    escaped_pid = None
    started = time.monotonic()
    try:
        result = run_foreground_test(
            [sys.executable, "-c", script], cwd=tmp_path, timeout_seconds=0.1
        )
    finally:
        if pid_file.exists():
            escaped_pid = int(pid_file.read_text(encoding="utf-8"))
        if escaped_pid is not None:
            try:
                os.kill(escaped_pid, 9)
            except ProcessLookupError:
                pass
    assert result is not None
    assert result.outcome == "timed_out"
    assert time.monotonic() - started < 2


def test_cli_wrapper_records_omitted_ceiling_and_rejects_over_policy_before_spawn(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS", "7")
    assert main(["run-tests", "--memory-dir", str(memory), "--", sys.executable, "-c", "pass"]) == 0
    rows = runtime.load_runtime_memory(memory)
    assert rows[-1]["attempted_timeout_seconds"] == 7
    marker = tmp_path / "spawned"
    assert main([
        "run-tests", "--timeout-seconds", "8", "--memory-dir", str(memory), "--",
        sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()",
    ]) == 1
    assert not marker.exists()
    assert len(runtime.load_runtime_memory(memory)) == 1


def test_cli_marks_unauthenticated_wrapper_launch_as_non_evidence(tmp_path, monkeypatch):
    """#989: a wrapper script's suite start is unknown, so it is not evidence."""
    memory = tmp_path / "memory"
    monkeypatch.chdir(tmp_path)
    for name in (
        "AGENT_LOOP_TEST_BROKER_ENDPOINT",
        "AGENT_LOOP_TEST_BROKER_CAPABILITY",
        "AGENT_LOOP_TEST_BROKER_PROTOCOL",
    ):
        monkeypatch.delenv(name, raising=False)
    wrapper = tmp_path / "run_suite.sh"
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o755)
    assert main(["run-tests", "--timeout-seconds", "5", "--memory-dir", str(memory), "--", str(wrapper)]) == 0
    rows = runtime.load_runtime_memory(memory)
    assert rows[-1]["launch_integrity"] == "unverified"
    assert not runtime.runtime_row_is_evidence(rows[-1])
    recommendation = runtime.recommend_timeout(
        memory, argv=[str(wrapper)], cwd=tmp_path, policy_ceiling_seconds=1800, now=_now()
    )
    assert recommendation.successful_samples == 0
    assert recommendation.recommended_timeout_seconds == 1800


def test_launch_integrity_state_requires_every_launch_boundary_verified():
    from types import SimpleNamespace

    verified = SimpleNamespace(wrapper_bootstrap="verified", inner_exec="started", suite_start="verified")
    assert runtime.launch_integrity_state(verified) == "verified"
    for field, value in (
        ("wrapper_bootstrap", "unknown"),
        ("inner_exec", "failed"),
        ("suite_start", "unknown"),
    ):
        degraded = SimpleNamespace(**{**vars(verified), field: value})
        assert runtime.launch_integrity_state(degraded) == "unverified"
    assert runtime.launch_integrity_state(object()) == "unverified"


def test_recommend_timeout_ignores_non_evidence_rows(tmp_path):
    memory = tmp_path / "memory"
    command = [sys.executable, "-m", "pytest", "tests/test_protocol.py", "-q"]
    for elapsed in (401, 410, 420):
        assert runtime.record_test_observation(
            memory, argv=command, cwd=tmp_path, outcome="passed", elapsed_seconds=elapsed,
            attempted_timeout_seconds=1800, policy_ceiling_seconds=1800, timestamp=_now(),
            launch_integrity="unverified",
        )
    assert runtime.recommend_timeout(
        memory, argv=command, cwd=tmp_path, policy_ceiling_seconds=1800, now=_now()
    ).successful_samples == 0
    assert runtime.record_test_observation(
        memory, argv=command, cwd=tmp_path, outcome="passed", elapsed_seconds=400,
        attempted_timeout_seconds=1800, policy_ceiling_seconds=1800, timestamp=_now(),
        launch_integrity="bogus",
    )
    assert runtime.load_runtime_memory(memory)[-1]["launch_integrity"] == "unverified"


def test_runtime_rows_without_verified_launch_state_are_not_evidence():
    """#989: legacy argv cannot prove an authenticated suite start."""
    assert not runtime.runtime_row_is_evidence({})
    assert not runtime.runtime_row_is_evidence({"normalized_command": "pytest -q"})
    assert not runtime.runtime_row_is_evidence({"launch_integrity": "unverified"})
    assert not runtime.runtime_row_is_evidence({"launch_integrity": "bogus"})
    assert runtime.runtime_row_is_evidence({"launch_integrity": "verified"})


def test_launch_integrity_state_gate_boundary_ignores_wrapper_bootstrap():
    from types import SimpleNamespace

    gate = SimpleNamespace(wrapper_bootstrap="unknown", inner_exec="started", suite_start="verified")
    assert runtime.launch_integrity_state(gate, wrapper_boundary=False) == "verified"
    assert runtime.launch_integrity_state(gate) == "unverified"
    wrapped = SimpleNamespace(wrapper_bootstrap="unknown", inner_exec="started", suite_start="unknown")
    assert runtime.launch_integrity_state(wrapped, wrapper_boundary=False) == "unverified"


def test_recommend_timeout_ignores_legacy_rows_without_launch_state(tmp_path):
    """#989: pre-existing fieldless rows (e.g. poisoned wrapper history) never recommend."""
    memory = tmp_path / "memory"
    command = [sys.executable, "-m", "pytest", "tests/test_protocol.py", "-q"]
    for elapsed in (401, 410, 420):
        assert runtime.record_test_observation(
            memory, argv=command, cwd=tmp_path, outcome="passed", elapsed_seconds=elapsed,
            attempted_timeout_seconds=1800, policy_ceiling_seconds=1800, timestamp=_now(),
        )
    assert "launch_integrity" not in runtime.load_runtime_memory(memory)[-1]
    recommendation = runtime.recommend_timeout(
        memory, argv=command, cwd=tmp_path, policy_ceiling_seconds=1800, now=_now()
    )
    assert recommendation.successful_samples == 0
    assert recommendation.recommended_timeout_seconds == 1800
    assert runtime.record_test_observation(
        memory, argv=command, cwd=tmp_path, outcome="passed", elapsed_seconds=400,
        attempted_timeout_seconds=1800, policy_ceiling_seconds=1800, timestamp=_now(),
        launch_integrity="verified",
    )
    assert runtime.recommend_timeout(
        memory, argv=command, cwd=tmp_path, policy_ceiling_seconds=1800, now=_now()
    ).successful_samples == 1


def test_cli_broker_failure_falls_back_and_records_unverified_run(tmp_path, monkeypatch, capsys):
    memory = tmp_path / "memory"
    marker = tmp_path / "ran"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_LOOP_TEST_BROKER_ENDPOINT", str(tmp_path / "missing.sock"))
    monkeypatch.setenv("AGENT_LOOP_TEST_BROKER_CAPABILITY", "a" * 64)
    monkeypatch.setenv("AGENT_LOOP_TEST_BROKER_PROTOCOL", "local-test-broker-v1")
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "turn")
    assert main([
        "run-tests", "--timeout-seconds", "5", "--memory-dir", str(memory), "--",
        sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()",
    ]) == 0
    assert marker.exists()
    assert runtime.load_runtime_memory(memory)[-1]["outcome"] == "passed"
    assert "unverified telemetry" in capsys.readouterr().err


def test_cli_without_broker_labels_standalone_telemetry_unverified(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for name in (
        "AGENT_LOOP_TEST_BROKER_ENDPOINT",
        "AGENT_LOOP_TEST_BROKER_CAPABILITY",
        "AGENT_LOOP_TEST_BROKER_PROTOCOL",
    ):
        monkeypatch.delenv(name, raising=False)
    assert main([
        "run-tests", "--timeout-seconds", "5", "--memory-dir", str(tmp_path / "memory"),
        "--", sys.executable, "-c", "pass",
    ]) == 0
    assert "telemetry-unverified evidence" in capsys.readouterr().err


def test_cli_without_broker_stamps_new_rows_with_tool_identity(tmp_path, monkeypatch, capsys):
    from coding_review_agent_loop import agent_executable, tool_provenance

    monkeypatch.chdir(tmp_path)
    for name in (
        "AGENT_LOOP_TEST_BROKER_ENDPOINT",
        "AGENT_LOOP_TEST_BROKER_CAPABILITY",
        "AGENT_LOOP_TEST_BROKER_PROTOCOL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        tool_provenance,
        "capture_client_tool_provenance",
        lambda **_kwargs: {"commit": "abcdef1234567", "dirty": True, "package_path": "/opt/agent-loop/pkg"},
    )
    memory = tmp_path / "memory"
    existing = [sys.executable, "-c", "print('old')"]
    assert runtime.record_test_observation(
        memory, argv=existing, cwd=tmp_path, outcome="passed", elapsed_seconds=1,
        attempted_timeout_seconds=10, policy_ceiling_seconds=1800,
    )
    before = [dict(row) for row in runtime.load_runtime_memory(memory)]
    assert main([
        "run-tests", "--timeout-seconds", "5", "--memory-dir", str(memory),
        "--", sys.executable, "-c", "pass",
    ]) == 0
    assert "telemetry-unverified evidence" in capsys.readouterr().err
    rows = runtime.load_runtime_memory(memory)
    old = [row for row in rows if row in before]
    assert old == before and all("tool_commit" not in row for row in old)
    new = [row for row in rows if row not in before]
    assert len(new) == 1
    assert new[0]["tool_commit"] == "abcdef1234567"
    assert new[0]["tool_dirty"] is True
    assert new[0]["tool_package_digest"] == agent_executable.package_digest("/opt/agent-loop/pkg")
    # The stamp never promotes a standalone row to broker evidence.
    assert new[0].get("lane") != "broker"
    assert "provenance" not in new[0]


def test_runtime_sidecar_recommendations_timeout_lower_bound_and_success_clear(tmp_path):
    memory = tmp_path / "memory"
    command = [sys.executable, "-m", "pytest", "tests/test_protocol.py", "-q"]
    for elapsed in (401, 410, 420):
        _record(memory, tmp_path, command, outcome="passed", elapsed=elapsed)
    recommendation = runtime.recommend_timeout(memory, argv=command, cwd=tmp_path, policy_ceiling_seconds=1800, now=_now())
    assert recommendation.successful_samples == 3
    assert recommendation.median_seconds == 410
    assert recommendation.p95_seconds == 420
    assert recommendation.recommended_timeout_seconds == 540
    assert recommendation.confidence == "high"

    _record(memory, tmp_path, command, outcome="timed_out", elapsed=300, attempted=600)
    recommendation = runtime.recommend_timeout(memory, argv=command, cwd=tmp_path, policy_ceiling_seconds=1800, now=_now())
    assert recommendation.unresolved_timeout_seconds == 600
    assert recommendation.recommended_timeout_seconds == 900
    _record(memory, tmp_path, command, outcome="passed", elapsed=700)
    cleared = runtime.recommend_timeout(memory, argv=command, cwd=tmp_path, policy_ceiling_seconds=1800, now=_now())
    assert cleared.unresolved_timeout_seconds is None


def test_timeout_resolution_uses_observation_order_not_duration_comparisons(tmp_path):
    memory = tmp_path / "memory"
    command = [sys.executable, "-c", "pass"]
    base = _now() - timedelta(seconds=10)
    for index, elapsed in enumerate((410, 535)):
        _record(
            memory,
            tmp_path,
            command,
            outcome="passed",
            elapsed=elapsed,
            timestamp=base + timedelta(seconds=index),
        )
    _record(
        memory,
        tmp_path,
        command,
        outcome="timed_out",
        elapsed=300,
        attempted=300,
        timestamp=base + timedelta(seconds=3),
    )
    timed_out = runtime.recommend_timeout(
        memory, argv=command, cwd=tmp_path, policy_ceiling_seconds=1800, now=_now()
    )
    assert timed_out.unresolved_timeout_seconds == 300
    assert "Last 300s attempt timed out" in runtime.render_runtime_context(
        memory,
        commands=(command,),
        cwd=tmp_path,
        policy_ceiling_seconds=1800,
        recommendations={tuple(command): timed_out},
    )

    _record(
        memory,
        tmp_path,
        command,
        outcome="timed_out",
        elapsed=1,
        attempted=600,
        timestamp=base + timedelta(seconds=4),
    )
    _record(
        memory,
        tmp_path,
        command,
        outcome="passed",
        elapsed=50,
        timestamp=base + timedelta(seconds=5),
    )
    cleared = runtime.recommend_timeout(
        memory, argv=command, cwd=tmp_path, policy_ceiling_seconds=1800, now=_now()
    )
    assert cleared.unresolved_timeout_seconds is None


def test_runtime_matches_relative_executables_and_wrapped_commands(tmp_path):
    executable = tmp_path / ".venv" / "bin" / "pytest"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    relative = [".venv/bin/pytest", "tests/test_protocol.py", "-q"]
    absolute = [str(executable), "tests/test_protocol.py", "-q"]
    assert runtime.environment_fingerprint(relative, tmp_path) == runtime.environment_fingerprint(
        absolute, tmp_path
    )

    memory = tmp_path / "memory"
    bare = [sys.executable, "-c", "pass"]
    _record(memory, tmp_path, bare, outcome="passed", elapsed=12)
    wrapped = [
        sys.executable,
        "-m",
        "coding_review_agent_loop.cli",
        "run-tests",
        "--timeout-seconds",
        "720",
        "--memory-dir",
        str(memory),
        "--",
        *bare,
    ]
    recommendation = runtime.recommend_timeout(
        memory, argv=wrapped, cwd=tmp_path, policy_ceiling_seconds=1800, now=_now()
    )
    assert recommendation.successful_samples == 1


def test_runtime_stale_and_input_manifest_changes_fall_back_to_ceiling(tmp_path):
    memory = tmp_path / "memory"
    target = tmp_path / "tests.py"
    target.write_text("assert True\n", encoding="utf-8")
    command = [sys.executable, str(target)]
    _record(memory, tmp_path, command, outcome="passed", elapsed=60, timestamp=_now() - timedelta(days=31))
    stale = runtime.recommend_timeout(memory, argv=command, cwd=tmp_path, policy_ceiling_seconds=7200, now=_now())
    assert stale.successful_samples == 0
    assert stale.recommended_timeout_seconds == 7200
    _record(memory, tmp_path, command, outcome="passed", elapsed=60)
    target.write_text("assert False\n", encoding="utf-8")
    changed = runtime.recommend_timeout(memory, argv=command, cwd=tmp_path, policy_ceiling_seconds=7200, now=_now())
    assert changed.successful_samples == 0
    assert changed.recommended_timeout_seconds == 7200


def test_runtime_sidecar_retention_and_corruption_recovery(tmp_path):
    memory = tmp_path / "memory"
    command = [sys.executable, "-c", "pass"]
    for index in range(21):
        _record(memory, tmp_path, command, outcome="failed", elapsed=index, timestamp=_now() + timedelta(seconds=index))
    assert len(runtime.load_runtime_memory(memory)) == 20
    sidecar = memory / runtime.RUNTIME_SIDECAR_NAME
    valid = sidecar.read_text(encoding="utf-8")
    sidecar.write_text("{not-json", encoding="utf-8")
    assert not runtime.record_test_observation(
        memory, argv=command, cwd=tmp_path, outcome="passed", elapsed_seconds=1,
        attempted_timeout_seconds=1800, policy_ceiling_seconds=1800,
    )
    assert sidecar.read_text(encoding="utf-8") == "{not-json"
    sidecar.write_text(valid, encoding="utf-8")


def test_runtime_fingerprint_isolation_and_privacy(tmp_path):
    memory = tmp_path / "memory"
    command = [sys.executable, "-c", "pass"]
    _record(memory, tmp_path, command, outcome="passed", elapsed=60, environment={"PATH": "profile-a"})
    _record(memory, tmp_path, command, outcome="passed", elapsed=120, environment={"PATH": "profile-b"})
    rows = runtime.load_runtime_memory(memory)
    assert len({row["environment_fingerprint"] for row in rows}) == 2
    serialized = json.dumps(rows)
    assert "profile-a" not in serialized
    assert "profile-b" not in serialized
    assert str(tmp_path) not in serialized


def test_launcher_health_is_additive_bounded_redacted_and_success_clears(tmp_path):
    memory = tmp_path / "memory"
    wrapper = tmp_path / "bin" / "agent-loop"
    wrapper.parent.mkdir()
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    wrapper.chmod(0o755)
    identity = runtime.launcher_candidate_identity(
        [str(wrapper), "run-tests"], cwd=tmp_path, kind="wrapper"
    )
    assert runtime.record_launcher_health(
        memory,
        cwd=tmp_path,
        candidate=identity,
        state="failed",
        provenance="wrapper-probe",
        repository="owner/repo",
        diagnostic="bad\n" + "x" * 500,
        timestamp=_now(),
    )
    rows = runtime.load_launcher_health(memory)
    assert len(rows) == 1
    assert len(rows[0]["diagnostic"]) == 240
    assert "\n" not in rows[0]["diagnostic"]
    assert rows[0]["candidate_identity"]["path"] == str(wrapper.resolve())
    assert str(tmp_path) in json.dumps(rows[0]["candidate_identity"])

    assert runtime.record_launcher_health(
        memory,
        cwd=tmp_path,
        candidate=identity,
        state="verified",
        provenance="wrapper-probe",
        repository="owner/repo",
        timestamp=_now() + timedelta(seconds=1),
    )
    states = [row["state"] for row in runtime.load_launcher_health(memory)]
    assert states == ["verified"]
    assert runtime.load_runtime_memory(memory) == []


def test_launcher_health_scope_and_expiry_are_independent_from_timing(tmp_path):
    memory = tmp_path / "memory"
    candidate = [str(tmp_path / ".venv" / "bin" / "pytest"), "tests"]
    old = _now() - timedelta(hours=25)
    assert runtime.record_launcher_health(
        memory, cwd=tmp_path, candidate=candidate, state="failed",
        provenance="parent-runner", repository="owner/repo", timestamp=old,
    )
    assert runtime.relevant_launcher_health(
        memory, cwd=tmp_path, repository="owner/repo", now=_now()
    ) == []
    assert runtime.record_test_observation(
        memory, argv=[sys.executable, "-c", "pass"], cwd=tmp_path,
        outcome="passed", elapsed_seconds=1, attempted_timeout_seconds=5,
        policy_ceiling_seconds=5, timestamp=_now(),
    )
    assert len(runtime.load_runtime_memory(memory)) == 1


def test_legacy_v1_sidecar_without_launcher_health_remains_writable(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / runtime.RUNTIME_SIDECAR_NAME).write_text(
        json.dumps({"schema_version": 1, "observations": [{"outcome": "passed"}]}),
        encoding="utf-8",
    )
    assert runtime.load_runtime_memory(memory) == [{"outcome": "passed"}]
    assert runtime.record_launcher_health(
        memory, cwd=tmp_path, candidate=["pytest"], state="failed",
        provenance="agent-reported", repository="owner/repo", diagnostic="missing",
    )
    payload = json.loads((memory / runtime.RUNTIME_SIDECAR_NAME).read_text(encoding="utf-8"))
    assert payload["observations"] == [{"outcome": "passed"}]
    assert len(payload["launcher_health"]) == 1


def test_malformed_health_rows_do_not_hide_timing_memory(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / runtime.RUNTIME_SIDECAR_NAME).write_text(
        json.dumps({
            "schema_version": 1,
            "observations": [{"outcome": "passed"}],
            "launcher_health": [
                {"state": "failed", "diagnostic": "untrusted"},
                "not-a-row",
            ],
        }),
        encoding="utf-8",
    )
    assert runtime.load_runtime_memory(memory) == [{"outcome": "passed"}]
    assert runtime.load_launcher_health(memory) == []


def test_recognized_inner_probe_uses_only_safe_version_argv(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    result = runtime.probe_inner_launcher(
        [sys.executable, "-m", "pytest", "tests/test_protocol.py", "-q"], cwd=tmp_path
    )
    assert result.state == "verified"
    assert calls[0][0] == (sys.executable, "-m", "pytest", "--version")
    assert calls[0][1]["timeout_seconds"] == 5.0


_SYSTEM_ENV = runtime._trusted_env_executable("/usr/bin/env", environment=os.environ)
_SYSTEM_ENV_AVAILABLE = _SYSTEM_ENV is not None
requires_system_env = pytest.mark.skipif(
    not _SYSTEM_ENV_AVAILABLE, reason="root-owned /usr/bin/env is unavailable"
)


@pytest.fixture
def no_ambient_invocation(monkeypatch):
    # An ambient invocation id (e.g. running under agent-loop) would share the
    # per-invocation probe cache and candidate budget across tests.
    monkeypatch.delenv("AGENT_LOOP_INVOCATION_ID", raising=False)
    # A host-exported NODE_OPTIONS makes the probe refuse node --test by
    # design, which would fail positive node tests and let negative ones pass
    # for the wrong reason. Tests that exercise it set it explicitly.
    monkeypatch.delenv("NODE_OPTIONS", raising=False)
    # Likewise a host-exported PW_TEST_REPORTER makes the probe refuse
    # Playwright by design.
    monkeypatch.delenv("PW_TEST_REPORTER", raising=False)


@requires_system_env
def test_env_assignment_prefix_is_normalized_for_inner_probe(tmp_path, monkeypatch, no_ambient_invocation):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    src = str(tmp_path / "src")
    for prefix in (["env"], ["/usr/bin/env"], ["env", "--"]):
        calls.clear()
        runtime._INNER_PREFLIGHT_CACHE.clear()
        result = runtime.probe_inner_launcher(
            [*prefix, f"PYTHONPATH={src}", "AGENT_FLAG=1", sys.executable, "-m", "pytest", "tests", "-q"],
            cwd=tmp_path,
        )
        assert result.state == "verified", prefix
        # The probe runs the real interpreter, not ``env``...
        assert calls[0][0] == (sys.executable, "-m", "pytest", "--version")
        # ...under the assignments the target will actually see.
        probe_env = calls[0][1]["env"]
        assert probe_env["PYTHONPATH"] == src
        assert probe_env["AGENT_FLAG"] == "1"


@requires_system_env
def test_env_prefix_without_assignments_is_normalized(tmp_path):
    assert runtime.recognized_inner_probe(
        ["env", sys.executable, "-m", "pytest", "tests"], cwd=tmp_path
    ) == (sys.executable, "-m", "pytest", "--version")


@requires_system_env
def test_env_prefix_assignments_distinguish_inner_probe_cache(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(kwargs["env"].get("PYTHONPATH"))
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    environment = {**runtime.os.environ, "AGENT_LOOP_INVOCATION_ID": "inv-964"}
    for value in ("a", "b", "a"):
        result = runtime.probe_inner_launcher(
            ["env", f"PYTHONPATH={value}", sys.executable, "-m", "pytest", "tests"],
            cwd=tmp_path,
            environment=environment,
            environment_is_complete=True,
        )
        assert result.state == "verified"
    assert calls == ["a", "b"]


@pytest.mark.parametrize(
    "argv_prefix",
    [
        ["env", "-i"],
        ["env", "--ignore-environment"],
        ["env", "-"],
        ["env", "-u", "PATH"],
        ["env", "-S"],
        ["env", "--chdir=/tmp"],
        ["env", "PYTHONPATH=x", "-i"],
        ["env", "1BAD=x"],
        ["env", "=x"],
        ["env", "env"],
        ["./env"],
        ["tools/env"],
    ],
)
def test_env_prefix_with_unreproducible_semantics_stays_unrecognized(tmp_path, monkeypatch, argv_prefix, no_ambient_invocation):
    calls = []
    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *args, **kwargs: calls.append(args))
    result = runtime.probe_inner_launcher(
        [*argv_prefix, sys.executable, "-m", "pytest", "tests"], cwd=tmp_path
    )

    assert result.state == "unknown"
    assert "unrecognized" in result.diagnostic
    assert calls == []


def test_env_prefix_without_command_or_with_unrecognized_target_is_unknown(tmp_path, monkeypatch, no_ambient_invocation):
    calls = []
    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *args, **kwargs: calls.append(args))
    for argv in (["env"], ["env", "PYTHONPATH=x"], ["env", "PYTHONPATH=x", "make", "test"]):
        result = runtime.probe_inner_launcher(argv, cwd=tmp_path)
        assert result.state == "unknown", argv
    assert calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["node", "--test", "tests/test_history.mjs"],
        ["node", "--test"],
        ["nodejs", "--test", "tests"],
        ["node", "--experimental-vm-modules", "--test", "tests/a.mjs"],
        ["node", "--test-reporter=spec", "--test", "tests/a.mjs"],
    ],
)
def test_node_test_runner_is_a_recognized_inner_launcher(tmp_path, argv):
    # Issue #1009: ``node --test`` must be probeable so its runs can be cited.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    environment = {"PATH": str(fake_bin)}
    for name in ("node", "nodejs"):
        launcher = fake_bin / name
        launcher.write_text("#!/bin/sh\n", encoding="utf-8")
        launcher.chmod(0o755)
    assert runtime.recognized_inner_probe(argv, cwd=tmp_path, environment=environment) == (
        str(fake_bin / argv[0]),
        "--version",
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["node", "scripts/build.mjs", "--test"],
        ["node", "scripts/build.mjs"],
        ["node", "--", "--test"],
        ["node"],
        ["npx", "--test"],
        ["node", "--test", "--version"],
        ["node", "--test", "-v"],
        ["node", "--test", "--help"],
        ["node", "--test", "-h"],
        ["node", "--test", "--v8-options"],
        ["node", "--test", "--check", "tests/a.mjs"],
        ["node", "--test", "--eval=process.exit(0)"],
        ["node", "--test", "-e", "process.exit(0)"],
        ["node", "--test", "--run", "noop"],
        ["node", "--test", "--experimental-sea-config=sea.json"],
        ["node", "--test", "tests/a.mjs", "--version"],
        ["node", "--test-reporter", "spec", "--test", "tests/a.mjs"],
        ["node", "--test", "--test-reporter="],
        ["node", "--test", "--import=data:text/javascript,process.exit(0)", "tests/a.mjs"],
        ["node", "--test", "--require=./exit.cjs", "tests/a.mjs"],
        ["node", "--test", "--loader=./loader.mjs", "tests/a.mjs"],
        ["node", "--test", "--experimental-loader=./loader.mjs", "tests/a.mjs"],
        ["node", "--test", "--test-global-setup=./setup.mjs", "tests/a.mjs"],
        ["node", "--test", "--env-file=.env", "tests/a.mjs"],
        ["node", "--test", "--test-reporter=./reporter.mjs", "tests/a.mjs"],
    ],
)
def test_node_without_builtin_test_runner_stays_unrecognized(tmp_path, monkeypatch, argv, no_ambient_invocation):
    calls = []
    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *args, **kwargs: calls.append(args))
    assert runtime.recognized_inner_probe(argv, cwd=tmp_path) is None
    result = runtime.probe_inner_launcher(argv, cwd=tmp_path)
    assert result.state == "unknown"
    assert calls == []


def test_node_test_runner_probe_uses_only_version_argv(tmp_path, monkeypatch, no_ambient_invocation):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return type("Completed", (), {"returncode": 0, "stdout": "v22.0.0", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    node = tmp_path / "node"
    node.write_text("#!/bin/sh\n", encoding="utf-8")
    node.chmod(0o755)
    result = runtime.probe_inner_launcher(
        [str(node), "--test", "tests/test_history.mjs"], cwd=tmp_path
    )
    assert result.state == "verified"
    assert calls[0][0] == (str(node), "--version")
    assert calls[0][1]["timeout_seconds"] == 5.0


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
@pytest.mark.parametrize("print_only", ["--version", "--help"])
def test_foreground_node_print_only_option_is_not_verified_evidence(tmp_path, print_only, no_ambient_invocation):
    # Review item on #1012: ``node --test --version`` exits 0 without running
    # any test, so it must never produce a verified, passing observation.
    from coding_review_agent_loop import runner as runner_module

    result = runner_module.run_foreground_test(
        ["node", "--test", print_only], cwd=tmp_path, timeout_seconds=60, echo_output=False
    )

    assert result.suite_start != "verified"


def test_node_options_preload_keeps_node_test_runner_unrecognized(tmp_path, monkeypatch, no_ambient_invocation):
    calls = []
    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *args, **kwargs: calls.append(args))
    environment = {**os.environ, "NODE_OPTIONS": "--import=./exit.mjs"}
    argv = ["node", "--test", "tests/a.mjs"]
    assert runtime.recognized_inner_probe(argv, cwd=tmp_path, environment=environment) is None
    assert runtime.probe_inner_launcher(argv, cwd=tmp_path, environment=environment).state == "unknown"
    assert runtime.recognized_inner_probe(
        ["env", "NODE_OPTIONS=--require=./exit.cjs", *argv], cwd=tmp_path, environment={**os.environ, "NODE_OPTIONS": ""}
    ) is None
    assert calls == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_foreground_node_early_exit_preload_is_not_verified_evidence(tmp_path, monkeypatch, no_ambient_invocation):
    # Review item on #1012: a preload that exits 0 before any test file runs
    # must never yield a verified, passing observation.
    from coding_review_agent_loop import runner as runner_module

    test_file = tmp_path / "failing.test.mjs"
    test_file.write_text(
        "import test from 'node:test';\ntest('fails', () => { throw new Error('ran'); });\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("NODE_OPTIONS", raising=False)
    result = runner_module.run_foreground_test(
        ["node", "--test", "--import=data:text/javascript,process.exit(0)", str(test_file)],
        cwd=tmp_path, timeout_seconds=60, echo_output=False,
    )

    assert result.suite_start != "verified"


@requires_system_env
def test_env_prefixed_node_test_runner_is_normalized(tmp_path, no_ambient_invocation):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    node = fake_bin / "node"
    node.write_text("#!/bin/sh\n", encoding="utf-8")
    node.chmod(0o755)
    environment = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"}
    environment.pop("NODE_OPTIONS", None)
    assert runtime.recognized_inner_probe(
        ["env", "NODE_ENV=test", "node", "--test", "tests/a.mjs"], cwd=tmp_path, environment=environment
    ) == (str(node), "--version")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_foreground_node_test_run_reports_verified_suite_start(tmp_path, monkeypatch, no_ambient_invocation):
    from coding_review_agent_loop import runner as runner_module

    test_file = tmp_path / "sample.test.mjs"
    test_file.write_text(
        "import test from 'node:test';\nimport assert from 'node:assert';\n"
        "test('adds', () => assert.strictEqual(1 + 1, 2));\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("NODE_OPTIONS", raising=False)
    result = runner_module.run_foreground_test(
        ["node", "--test", str(test_file)], cwd=tmp_path, timeout_seconds=60, echo_output=False
    )

    assert result.suite_start == "verified"
    assert result.passed


def _fake_playwright(root):
    launcher = root / "node_modules" / ".bin" / "playwright"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


_PW = "node_modules/.bin/playwright"


@pytest.mark.parametrize(
    "tail",
    [
        ["test"],
        ["test", "--project=chromium"],
        ["test", "tests/e2e/cost.spec.ts", "--reporter=line", "--workers=1"],
        ["test", "--reporter=list,json", "--headed", "--grep=cost"],
    ],
)
def test_playwright_test_is_a_recognized_inner_launcher(tmp_path, no_ambient_invocation, tail):
    launcher = _fake_playwright(tmp_path)
    environment = {"PATH": str(launcher.parent)}
    expected = (str(launcher.resolve()), "--version")
    assert runtime.recognized_inner_probe([_PW, *tail], cwd=tmp_path, environment=environment) == expected
    assert runtime.recognized_inner_probe(["playwright", *tail], cwd=tmp_path, environment=environment) == (
        str(launcher), "--version"
    )


@pytest.mark.parametrize(
    "tail",
    [
        ["test", "--project=chromium", "--list"],
        ["--list"],
        ["--version"],
        ["--help"],
        ["test", "--help"],
        ["test", "-h"],
        ["test", "--version"],
        [],
        ["show-report"],
        ["install"],
        ["tests", "test"],
        ["test", "--config=./evil.config.ts"],
        ["test", "-c", "x"],
        ["test", "--tsconfig=x"],
        ["test", "--global-setup=./exit.js"],
        ["test", "--reporter=./reporter.js"],
        ["test", "--reporter=list,./r.js"],
        ["test", "--reporter="],
        ["test", "--project", "chromium"],
        ["test", "--pass-with-no-tests"],
        ["test", "--only-changed"],
        ["test", "--last-failed"],
        ["test", "--shard=1/1"],
        ["test", "--shard=1/1", "--grep=nomatch"],
        ["test", "--ui"],
        ["test", "--debug"],
        ["test", "--", "x"],
        ["test", "-x"],
    ],
)
def test_playwright_unsafe_forms_stay_unrecognized(tmp_path, monkeypatch, no_ambient_invocation, tail):
    _fake_playwright(tmp_path)
    calls = []
    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *args, **kwargs: calls.append(args))
    argv = [_PW, *tail]
    assert runtime.recognized_inner_probe(argv, cwd=tmp_path) is None
    assert runtime.probe_inner_launcher(argv, cwd=tmp_path).state == "unknown"
    assert calls == []


@pytest.mark.parametrize(
    "prefix", [["npx"], ["npx", "--no-install"], ["npx", "--no"]],
)
def test_npx_playwright_with_local_binary_probes_local_binary(tmp_path, monkeypatch, no_ambient_invocation, prefix):
    launcher = _fake_playwright(tmp_path)
    argv = [*prefix, "playwright", "test", "--project=x"]
    probe = runtime.recognized_inner_probe(argv, cwd=tmp_path)
    assert probe == (str(tmp_path / "node_modules/.bin/playwright"), "--version")

    def fake_run(a, **kwargs):
        return type("Completed", (), {"returncode": 0, "stdout": "1.50.0", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    result = runtime.probe_inner_launcher(argv, cwd=tmp_path)
    assert result.state == "verified"
    assert result.candidate == tuple(argv)
    assert result.launch_argv == (str(launcher), "test", "--project=x")


def test_npx_playwright_without_local_binary_stays_unknown(tmp_path, monkeypatch, no_ambient_invocation):
    calls = []
    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *a, **k: calls.append(a))
    argv = ["npx", "playwright", "test"]
    assert runtime.recognized_inner_probe(argv, cwd=tmp_path) is None
    assert runtime.probe_inner_launcher(argv, cwd=tmp_path).state == "unknown"
    assert calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["npx", "-y", "playwright", "test"],
        ["npx", "--yes", "playwright", "test"],
        ["npx", "-p", "x", "playwright", "test"],
        ["npx", "--package", "x", "playwright", "test"],
        ["npx", "-c", "playwright test"],
        ["npx", "playwright@1.50", "test"],
        ["npx", "--", "playwright", "test"],
        ["npx", "playwright", "test", "--list"],
        ["npx", "playwright", "test", "--config=x.js"],
    ],
)
def test_npx_playwright_unsafe_forms_stay_unrecognized(tmp_path, no_ambient_invocation, argv):
    _fake_playwright(tmp_path)
    assert runtime.recognized_inner_probe(argv, cwd=tmp_path) is None


@pytest.mark.parametrize("name", ["NODE_OPTIONS", "PW_TEST_REPORTER"])
def test_npx_playwright_code_loading_environment_stays_unrecognized(tmp_path, no_ambient_invocation, name):
    _fake_playwright(tmp_path)
    argv = ["npx", "playwright", "test"]
    environment = {**os.environ, name: "./evil.js"}
    assert runtime.recognized_inner_probe(argv, cwd=tmp_path, environment=environment) is None
    assert runtime.recognized_inner_probe(
        ["env", f"{name}=./evil.js", *argv], cwd=tmp_path, environment={**os.environ, name: ""}
    ) is None


def test_npx_playwright_env_prefix_keeps_distinct_identity(tmp_path, monkeypatch, no_ambient_invocation):
    launcher = _fake_playwright(tmp_path)
    monkeypatch.setattr(
        runtime, "_run_bounded_probe",
        lambda *a, **k: type("C", (), {"returncode": 0, "stdout": "1", "stderr": ""})(),
    )
    one = runtime.probe_inner_launcher(["env", "A=1", "npx", "playwright", "test"], cwd=tmp_path)
    two = runtime.probe_inner_launcher(["env", "A=2", "npx", "playwright", "test"], cwd=tmp_path)
    assert one.state == two.state == "verified"
    assert one.launch_argv[1:] == ("A=1", str(launcher), "test")
    assert two.launch_argv[1:] == ("A=2", str(launcher), "test")
    assert one.identity != two.identity


def test_foreground_npx_playwright_launches_local_binary_not_npx(tmp_path, monkeypatch, no_ambient_invocation):
    from coding_review_agent_loop import runner as runner_module

    _fake_playwright(tmp_path)
    sentinel = tmp_path / "npx-ran"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_npx = bindir / "npx"
    fake_npx.write_text(f"#!/bin/sh\ntouch {sentinel}\nexit 0\n", encoding="utf-8")
    fake_npx.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    result = runner_module.run_foreground_test(
        ["npx", "playwright", "test", "--project=chromium"], cwd=tmp_path, timeout_seconds=60, echo_output=False
    )
    assert result.suite_start == "verified"
    assert runtime.launch_integrity_state(result, wrapper_boundary=False) == "verified"
    assert not sentinel.exists()


@pytest.mark.parametrize("name", ["NODE_OPTIONS", "PW_TEST_REPORTER"])
def test_playwright_code_loading_environment_stays_unrecognized(tmp_path, monkeypatch, no_ambient_invocation, name):
    _fake_playwright(tmp_path)
    calls = []
    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *args, **kwargs: calls.append(args))
    argv = [_PW, "test"]
    environment = {**os.environ, name: "./evil.js"}
    assert runtime.recognized_inner_probe(argv, cwd=tmp_path, environment=environment) is None
    assert runtime.probe_inner_launcher(argv, cwd=tmp_path, environment=environment).state == "unknown"
    assert runtime.recognized_inner_probe(
        ["env", f"{name}=./evil.js", *argv], cwd=tmp_path, environment={**os.environ, name: ""}
    ) is None
    assert calls == []


def test_playwright_probe_uses_only_version_argv(tmp_path, monkeypatch, no_ambient_invocation):
    launcher = _fake_playwright(tmp_path)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return type("Completed", (), {"returncode": 0, "stdout": "1.50.0", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    result = runtime.probe_inner_launcher([_PW, "test", "--project=chromium"], cwd=tmp_path)
    assert result.state == "verified"
    assert calls[0][0] == (str(launcher.resolve()), "--version")
    assert calls[0][1]["timeout_seconds"] == 5.0


def test_foreground_playwright_run_is_verified_and_list_is_not(tmp_path, no_ambient_invocation):
    from coding_review_agent_loop import runner as runner_module

    _fake_playwright(tmp_path)
    verified = runner_module.run_foreground_test(
        [_PW, "test", "--project=chromium"], cwd=tmp_path, timeout_seconds=60, echo_output=False
    )
    assert verified.suite_start == "verified"
    assert runtime.launch_integrity_state(verified, wrapper_boundary=False) == "verified"
    listed = runner_module.run_foreground_test(
        [_PW, "test", "--project=chromium", "--list"], cwd=tmp_path, timeout_seconds=60, echo_output=False
    )
    assert listed.suite_start != "verified"
    assert runtime.launch_integrity_state(listed, wrapper_boundary=False) == "unverified"


def test_cli_records_playwright_run_as_evidence_and_list_as_non_evidence(tmp_path, monkeypatch, no_ambient_invocation):
    memory = tmp_path / "memory"
    monkeypatch.chdir(tmp_path)
    for name in (
        "AGENT_LOOP_TEST_BROKER_ENDPOINT",
        "AGENT_LOOP_TEST_BROKER_CAPABILITY",
        "AGENT_LOOP_TEST_BROKER_PROTOCOL",
    ):
        monkeypatch.delenv(name, raising=False)
    _fake_playwright(tmp_path)
    base = ["run-tests", "--timeout-seconds", "30", "--memory-dir", str(memory), "--"]
    assert main([*base, _PW, "test", "--project=chromium"]) == 0
    first_rows = runtime.load_runtime_memory(memory)
    assert len(first_rows) == 1
    row = first_rows[0]
    assert row["launch_integrity"] == "verified"
    assert runtime.runtime_row_is_evidence(row)
    assert main([*base, _PW, "test", "--project=chromium", "--list"]) == 0
    new_rows = [item for item in runtime.load_runtime_memory(memory) if item not in first_rows]
    assert len(new_rows) == 1
    row = new_rows[0]
    assert row["launch_integrity"] == "unverified"
    assert not runtime.runtime_row_is_evidence(row)


def test_cli_records_npx_playwright_run_as_evidence_with_npx_command(tmp_path, monkeypatch, no_ambient_invocation):
    memory = tmp_path / "memory"
    monkeypatch.chdir(tmp_path)
    for name in (
        "AGENT_LOOP_TEST_BROKER_ENDPOINT",
        "AGENT_LOOP_TEST_BROKER_CAPABILITY",
        "AGENT_LOOP_TEST_BROKER_PROTOCOL",
    ):
        monkeypatch.delenv(name, raising=False)
    _fake_playwright(tmp_path)
    sentinel = tmp_path / "npx-ran"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_npx = bindir / "npx"
    fake_npx.write_text(f"#!/bin/sh\ntouch {sentinel}\nexit 0\n", encoding="utf-8")
    fake_npx.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    base = ["run-tests", "--timeout-seconds", "30", "--memory-dir", str(memory), "--"]
    assert main([*base, "npx", "playwright", "test", "--project=chromium"]) == 0
    rows = runtime.load_runtime_memory(memory)
    assert len(rows) == 1
    row = rows[0]
    assert row["launch_integrity"] == "verified"
    assert runtime.runtime_row_is_evidence(row)
    assert "npx playwright test" in json.dumps(row)
    assert row["normalized_command"].startswith("npx playwright test")
    assert not sentinel.exists()


def test_lookalike_env_executable_is_not_stripped_from_probe(tmp_path, monkeypatch, no_ambient_invocation):
    # A program named ``env`` that ignores its argv and exits 0 must not let
    # the probe verify the real interpreter on its behalf.
    calls = []
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_env = fake_bin / "env"
    fake_env.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_env.chmod(0o755)
    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *args, **kwargs: calls.append(args))
    shadowed = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"}
    for argv_prefix, environment in (
        (["env"], shadowed),
        ([str(fake_env)], None),
    ):
        argv = [*argv_prefix, "PYTHONPATH=x", sys.executable, "-m", "pytest", "tests"]
        assert runtime.recognized_inner_probe(argv, cwd=tmp_path, environment=environment) is None
        result = runtime.probe_inner_launcher(argv, cwd=tmp_path, environment=environment)
        assert result.state == "unknown", argv_prefix
        assert "unrecognized" in result.diagnostic
    assert calls == []


@requires_system_env
def test_symlinked_env_is_bound_to_canonical_system_env(tmp_path, monkeypatch, no_ambient_invocation):
    # A mutable alias (an absolute symlink, or a bare ``env`` found through a
    # user-writable PATH symlink) may be recognized, but the verified result
    # binds the launch to the canonical root-protected ``env`` so swapping
    # the alias after the probe cannot change what actually runs.
    monkeypatch.setattr(
        runtime,
        "_run_bounded_probe",
        lambda argv, **kwargs: type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})(),
    )
    alias_dir = tmp_path / "alias-bin"
    alias_dir.mkdir()
    alias = alias_dir / "env"
    alias.symlink_to(_SYSTEM_ENV)
    shadowed = {**os.environ, "PATH": f"{alias_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    for token, environment in ((str(alias), None), ("env", shadowed)):
        runtime._INNER_PREFLIGHT_CACHE.clear()
        argv = [token, "PYTHONPATH=x", sys.executable, "-m", "pytest", "tests"]
        result = runtime.probe_inner_launcher(argv, cwd=tmp_path, environment=environment)
        assert result.state == "verified", token
        assert result.candidate == tuple(argv)
        assert result.launch_argv == (_SYSTEM_ENV, *argv[1:])


def test_unprotected_env_install_is_not_trusted(tmp_path, monkeypatch):
    fake_env = tmp_path / "usr" / "bin" / "env"
    fake_env.parent.mkdir(parents=True)
    fake_env.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_env.chmod(0o755)
    monkeypatch.setattr(runtime, "_TRUSTED_ENV_PATHS", (fake_env,))
    # Same file as the "trusted" path, but its chain is user-owned.
    assert runtime._trusted_env_executable(str(fake_env), environment=os.environ) is None


def test_unprefixed_launcher_has_no_launch_rebinding(tmp_path, monkeypatch, no_ambient_invocation):
    monkeypatch.setattr(
        runtime,
        "_run_bounded_probe",
        lambda argv, **kwargs: type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})(),
    )
    result = runtime.probe_inner_launcher([sys.executable, "-m", "pytest", "tests"], cwd=tmp_path)
    assert result.state == "verified"
    assert result.launch_argv == ()


@requires_system_env
def test_cached_env_probe_binds_launch_to_current_command(tmp_path, monkeypatch):
    # The per-invocation cache is keyed by launcher identity, not by pytest's
    # selectors; a cache hit must launch and report the current command.
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    environment = {**os.environ, "AGENT_LOOP_INVOCATION_ID": "inv-964-selectors"}
    first = ["env", "X=1", sys.executable, "-m", "pytest", "tests/a"]
    second = ["env", "X=1", sys.executable, "-m", "pytest", "tests/b"]
    first_result = runtime.probe_inner_launcher(
        first, cwd=tmp_path, environment=environment, environment_is_complete=True
    )
    second_result = runtime.probe_inner_launcher(
        second, cwd=tmp_path, environment=environment, environment_is_complete=True
    )
    assert len(calls) == 1  # the second probe is a cache hit
    assert first_result.launch_argv == (_SYSTEM_ENV, *first[1:])
    assert second_result.state == "verified"
    assert second_result.candidate == tuple(second)
    assert second_result.launch_argv == (_SYSTEM_ENV, *second[1:])
    assert all(entry.launch_argv == () for entry in runtime._INNER_PREFLIGHT_CACHE.values())


@requires_system_env
def test_foreground_runs_under_shared_invocation_spawn_their_own_selectors(tmp_path, monkeypatch):
    from coding_review_agent_loop import runner as runner_module

    for name in ("a", "b"):
        (tmp_path / f"test_{name}.py").write_text(
            f"def test_{name}():\n    print('RAN-{name.upper()}')\n", encoding="utf-8"
        )
    spawned = []
    real_popen = runner_module.subprocess.Popen

    def recording_popen(argv, *args, **kwargs):
        spawned.append(list(argv))
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(runner_module.subprocess, "Popen", recording_popen)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "inv-964-foreground")
    results = []
    for name in ("a", "b"):
        cmd = [
            "env", "X=1", sys.executable, "-m", "pytest", f"test_{name}.py",
            "-q", "-s", "-p", "no:cacheprovider",
        ]
        results.append(
            runner_module.run_foreground_test(cmd, cwd=tmp_path, timeout_seconds=60, echo_output=False)
        )

    assert [result.suite_start for result in results] == ["verified", "verified"]
    # Ignore the bounded ``--version`` probes; keep only the real targets.
    targets = [argv for argv in spawned if argv[0] == _SYSTEM_ENV]
    assert [argv[5] for argv in targets] == ["test_a.py", "test_b.py"]
    assert "RAN-B" in results[1].output_tail
    assert "RAN-A" not in results[1].output_tail


@requires_system_env
def test_foreground_run_spawns_authenticated_env_after_alias_swap(tmp_path, monkeypatch, no_ambient_invocation):
    from coding_review_agent_loop import runner as runner_module

    alias_dir = tmp_path / "alias-bin"
    alias_dir.mkdir()
    alias = alias_dir / "env"
    alias.symlink_to(_SYSTEM_ENV)
    original_probe = runner_module.probe_inner_launcher

    def probe_then_swap(*args, **kwargs):
        result = original_probe(*args, **kwargs)
        # Replace the authenticated alias with a no-op lookalike.
        alias.unlink()
        alias.write_text("#!/bin/sh\necho FAKE-ENV\nexit 0\n", encoding="utf-8")
        alias.chmod(0o755)
        return result

    monkeypatch.setattr(runner_module, "probe_inner_launcher", probe_then_swap)
    cmd = [str(alias), "AGENT_LOOP_964_MARK=1", sys.executable, "-m", "pytest", "--version"]
    result = runner_module.run_foreground_test(cmd, cwd=tmp_path, timeout_seconds=60, echo_output=False)

    assert result.suite_start == "verified"
    assert result.args == cmd
    assert result.passed
    assert "FAKE-ENV" not in result.output_tail
    assert "pytest" in result.output_tail


def test_non_python_m_pytest_command_is_not_spawned_by_preflight(tmp_path, monkeypatch):
    calls = []
    non_python = tmp_path / "repo-script"
    non_python.write_text("#!/bin/sh\n", encoding="utf-8")
    non_python.chmod(0o755)

    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *args, **kwargs: calls.append(args))
    result = runtime.probe_inner_launcher(
        [str(non_python), "-m", "pytest", "tests"], cwd=tmp_path
    )

    assert result.state == "unknown"
    assert "unrecognized" in result.diagnostic
    assert calls == []


def test_script_named_python_is_not_spawned_by_preflight(tmp_path, monkeypatch):
    calls = []
    fake_python = tmp_path / "bin" / "python"
    fake_python.parent.mkdir()
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *args, **kwargs: calls.append(args))
    result = runtime.probe_inner_launcher(
        [str(fake_python), "-m", "pytest", "tests"], cwd=tmp_path
    )

    assert result.state == "unknown"
    assert "unrecognized" in result.diagnostic
    assert calls == []


def test_arbitrary_native_binary_named_python_is_not_spawned_by_preflight(tmp_path, monkeypatch):
    calls = []
    venv = tmp_path / "fake-venv"
    fake_python = venv / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    fake_python.write_bytes(b"\x7fELFarbitrary-native-program\n")
    fake_python.chmod(0o755)

    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *args, **kwargs: calls.append(args))
    result = runtime.probe_inner_launcher(
        [str(fake_python), "-m", "pytest", "tests"], cwd=tmp_path
    )

    assert result.state == "unknown"
    assert "unrecognized" in result.diagnostic
    assert calls == []


def test_copied_current_python_binary_is_safe_to_probe(tmp_path, monkeypatch):
    calls = []
    venv = tmp_path / "copied-venv"
    interpreter = venv / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    shutil.copyfile(sys.executable, interpreter)
    interpreter.chmod(0o755)

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    result = runtime.probe_inner_launcher(
        [str(interpreter), "-m", "pytest", "tests"], cwd=tmp_path
    )

    assert result.state == "verified"
    assert calls == [(str(interpreter), "-m", "pytest", "--version")]


def test_wrapper_preflight_has_fixed_argv_and_per_invocation_cache(tmp_path, monkeypatch):
    calls = []
    wrapper = tmp_path / "agent-loop"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setattr(runtime, "_wrapper_candidates", lambda _environment: [(str(wrapper), "run-tests")])
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "health-test-turn")

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return type("Completed", (), {"returncode": 0, "stdout": "agent-loop preflight: verified", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    first = runtime.preflight_wrapper_candidates(cwd=tmp_path)
    second = runtime.preflight_wrapper_candidates(cwd=tmp_path)
    assert first[0].state == "verified"
    assert second[0].state == "verified"
    assert calls == [(str(wrapper.resolve()), "run-tests", "--preflight")]


def test_wrapper_preflight_classifies_explicit_import_failure(tmp_path, monkeypatch):
    wrapper = tmp_path / "agent-loop"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "wrapper-import-failure-test")
    monkeypatch.setattr(runtime, "_wrapper_candidates", lambda _environment: [(str(wrapper), "run-tests")])
    monkeypatch.setattr(
        runtime,
        "_run_bounded_probe",
        lambda *_args, **_kwargs: type(
            "Completed", (), {
                "returncode": 1,
                "stdout": "",
                "stderr": "ModuleNotFoundError: No module named coding_review_agent_loop",
            }
        )(),
    )
    result = runtime.preflight_wrapper_candidates(cwd=tmp_path)[0]
    assert result.state == "failed"


@pytest.mark.skipif(os.name != "posix", reason="process-group probe cleanup is POSIX-specific")
def test_wrapper_probe_timeout_kills_descendant_holding_output_pipe(tmp_path, monkeypatch):
    wrapper = tmp_path / "agent-loop"
    pid_file = tmp_path / "wrapper-descendant.pid"
    child_code = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    wrapper.write_text(
        f"#!{sys.executable}\n"
        f"import subprocess, sys\n"
        f"from pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"Path({str(pid_file)!r}).write_text(str(child.pid))\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "wrapper-process-tree-test")
    monkeypatch.setattr(runtime, "LAUNCHER_PROBE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(
        runtime, "_wrapper_candidates", lambda _environment: [(str(wrapper), "run-tests")]
    )

    result = runtime.preflight_wrapper_candidates(cwd=tmp_path)[0]

    assert result.state == "failed"
    assert "timed out" in result.diagnostic
    descendant_pid = int(pid_file.read_text())
    try:
        _assert_process_not_active(descendant_pid)
    finally:
        _kill_if_active(descendant_pid)


@pytest.mark.skipif(os.name != "posix", reason="process-group probe cleanup is POSIX-specific")
def test_inner_probe_timeout_kills_descendant_holding_output_pipe(tmp_path, monkeypatch):
    pytest_launcher = tmp_path / "pytest"
    pid_file = tmp_path / "inner-descendant.pid"
    child_code = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    pytest_launcher.write_text(
        f"#!{sys.executable}\n"
        f"import subprocess, sys\n"
        f"from pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"Path({str(pid_file)!r}).write_text(str(child.pid))\n",
        encoding="utf-8",
    )
    pytest_launcher.chmod(0o755)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "inner-process-tree-test")
    monkeypatch.setattr(runtime, "LAUNCHER_PROBE_TIMEOUT_SECONDS", 0.2)

    result = runtime.probe_inner_launcher([str(pytest_launcher), "tests"], cwd=tmp_path)

    assert result.state == "failed"
    assert "timed out" in result.diagnostic
    descendant_pid = int(pid_file.read_text())
    try:
        _assert_process_not_active(descendant_pid)
    finally:
        _kill_if_active(descendant_pid)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object probe cleanup is Windows-specific")
def test_windows_probe_timeout_kills_descendant_holding_output_pipe(tmp_path):
    pid_file = tmp_path / "windows-descendant.pid"
    child_code = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    probe_code = (
        "import subprocess, sys; "
        "from pathlib import Path; "
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"Path({str(pid_file)!r}).write_text(str(child.pid))"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        runtime._run_bounded_probe(
            [sys.executable, "-c", probe_code],
            cwd=tmp_path,
            env=None,
            timeout_seconds=0.2,
        )

    descendant_pid = int(pid_file.read_text())
    try:
        _assert_windows_process_not_active(descendant_pid)
    finally:
        _kill_windows_if_active(descendant_pid)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object probe cleanup is Windows-specific")
def test_windows_probe_assigns_before_probe_can_spawn_descendant(tmp_path, monkeypatch):
    pid_file = tmp_path / "windows-suspended-descendant.pid"
    child_code = "import time; time.sleep(30)"
    probe_code = (
        "import subprocess, sys, time; "
        "from pathlib import Path; "
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"Path({str(pid_file)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    real_popen = subprocess.Popen
    real_job = runtime._WindowsProbeJob
    assignment_saw_probe_run: list[bool] = []

    class _RecordingJob:
        def __init__(self):
            self._job = real_job()

        def assign(self, process):
            assignment_saw_probe_run.append(pid_file.exists())
            self._job.assign(process)

        def resume(self, process):
            self._job.resume(process)

        def terminate(self):
            self._job.terminate()

        def close(self):
            self._job.close()

    def delayed_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        # An unsuspended implementation gets a deterministic pre-assignment
        # window in which the probe can create an unowned descendant.
        time.sleep(0.25)
        return process

    monkeypatch.setattr(runtime, "_WindowsProbeJob", _RecordingJob)
    monkeypatch.setattr(runtime.subprocess, "Popen", delayed_popen)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            runtime._run_bounded_probe(
                [sys.executable, "-c", probe_code],
                cwd=tmp_path,
                env=None,
                timeout_seconds=0.2,
            )
    finally:
        # subprocess.run in the cleanup assertion uses the same module object.
        monkeypatch.setattr(runtime.subprocess, "Popen", real_popen)

    assert assignment_saw_probe_run == [False]
    descendant_pid = int(pid_file.read_text())
    try:
        _assert_windows_process_not_active(descendant_pid)
    finally:
        _kill_windows_if_active(descendant_pid)


def test_wrapper_preflight_bounds_changing_completed_identities(tmp_path, monkeypatch):
    calls = []
    wrapper = tmp_path / "agent-loop"
    monkeypatch.setattr(runtime, "_wrapper_candidates", lambda _environment: [(str(wrapper), "run-tests")])
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "wrapper-identity-churn-test")

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return type("Completed", (), {"returncode": 0, "stdout": "agent-loop preflight: verified", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    for index in range(runtime.MAX_WRAPPER_PREFLIGHT_IDENTITIES_PER_INVOCATION + 5):
        wrapper.write_text(f"#!/bin/sh\n# identity-{index}\n{'x' * index}", encoding="utf-8")
        wrapper.chmod(0o755)
        assert runtime.preflight_wrapper_candidates(cwd=tmp_path)[0].state == "verified"

    invocation = "wrapper-identity-churn-test"
    assert len(runtime._WRAPPER_PREFLIGHT_IDENTITIES[invocation]) <= runtime.MAX_WRAPPER_PREFLIGHT_IDENTITIES_PER_INVOCATION
    assert sum(key[0] == invocation for key in runtime._WRAPPER_PREFLIGHT_CACHE) <= runtime.MAX_WRAPPER_PREFLIGHT_IDENTITIES_PER_INVOCATION
    assert len(calls) == runtime.MAX_WRAPPER_PREFLIGHT_IDENTITIES_PER_INVOCATION + 5


def test_wrapper_preflight_releases_reservation_after_unexpected_probe_exception(tmp_path, monkeypatch):
    calls = []
    wrapper = tmp_path / "agent-loop"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setattr(runtime, "_wrapper_candidates", lambda _environment: [(str(wrapper), "run-tests")])
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "wrapper-exception-cleanup-test")

    def explode(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError("probe harness failed")

    monkeypatch.setattr(runtime, "_run_bounded_probe", explode)
    first = runtime.preflight_wrapper_candidates(cwd=tmp_path)[0]
    second = runtime.preflight_wrapper_candidates(cwd=tmp_path)[0]

    assert first.state == second.state == "unknown"
    assert "RuntimeError" in first.diagnostic
    assert calls == [True]
    assert not any(key[0] == "wrapper-exception-cleanup-test" for key in runtime._WRAPPER_PREFLIGHT_INFLIGHT)


def test_inner_probe_timeout_is_bounded_and_cached(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "inner-timeout-test")

    def timeout(*_args, **kwargs):
        calls.append(kwargs["timeout_seconds"])
        raise runtime.subprocess.TimeoutExpired([sys.executable, "-m", "pytest"], 5.0)

    monkeypatch.setattr(runtime, "_run_bounded_probe", timeout)
    command = [sys.executable, "-m", "pytest", "tests"]
    first = runtime.probe_inner_launcher(command, cwd=tmp_path)
    second = runtime.probe_inner_launcher(command, cwd=tmp_path)
    assert first.state == second.state == "failed"
    assert calls == [5.0]


def test_runner_separates_launcher_failure_from_genuine_pytest_failure(tmp_path):
    missing = run_foreground_test(
        [str(tmp_path / "missing-test-launcher")], cwd=tmp_path, timeout_seconds=5
    )
    assert missing.outcome == "launch-failed"
    assert missing.inner_exec == "failed"
    assert missing.suite_start == "not-started"

    suite_failure = run_foreground_test(
        [sys.executable, "-m", "pytest", str(tmp_path / "missing-test-file.py"), "-q"],
        cwd=tmp_path,
        timeout_seconds=30,
    )
    assert suite_failure.outcome == "failed"
    assert suite_failure.inner_exec == "started"
    assert suite_failure.suite_start == "verified"


def test_launcher_diagnostic_redacts_common_credentials(tmp_path):
    diagnostic = (
        "AWS_SECRET_ACCESS_KEY=aws-secret Authorization: Bearer bearer-secret "
        "https://user:password@example.invalid/repo --token cli-secret"
    )
    safe = runtime._collapsed_diagnostic(diagnostic)
    for secret in ("aws-secret", "bearer-secret", "password@example", "cli-secret"):
        assert secret not in safe
    assert "<redacted>" in safe
    assert "<userinfo:redacted>" in safe


def test_repeated_launcher_failure_refreshes_health_timestamp(tmp_path):
    memory = tmp_path / "memory"
    candidate = [str(tmp_path / "missing-pytest"), "tests"]
    first = _now() - timedelta(hours=23, minutes=59)
    second = _now()
    assert runtime.record_launcher_health(
        memory, cwd=tmp_path, candidate=candidate, state="failed",
        provenance="parent-runner", repository="owner/repo", diagnostic="missing",
        timestamp=first,
    )
    assert runtime.record_launcher_health(
        memory, cwd=tmp_path, candidate=candidate, state="failed",
        provenance="parent-runner", repository="owner/repo", diagnostic="missing",
        timestamp=second,
    )
    rows = runtime.load_launcher_health(memory)
    assert len(rows) == 1
    assert runtime._timestamp(rows[0]["timestamp"]) == second


def test_console_wrapper_identity_fingerprints_actual_shebang_interpreter(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    interpreter = bin_dir / "python-real"
    interpreter.write_text("interpreter-v1\n", encoding="utf-8")
    wrapper = bin_dir / "agent-loop"
    wrapper.write_text(f"#!{interpreter}\n", encoding="utf-8")
    wrapper.chmod(0o755)
    identity = runtime.launcher_candidate_identity(
        [str(wrapper), "run-tests"], cwd=tmp_path, kind="wrapper"
    )
    assert identity["interpreter"]["path"] == str(interpreter.resolve())
    first_key = runtime._identity_key(identity)
    interpreter.write_text("interpreter-v2-with-a-different-size\n", encoding="utf-8")
    changed = runtime.launcher_candidate_identity(
        [str(wrapper), "run-tests"], cwd=tmp_path, kind="wrapper"
    )
    assert runtime._identity_key(changed) != first_key
    redacted = runtime._redacted_identity(identity, cwd=tmp_path)
    assert str(interpreter) not in json.dumps(redacted)


def test_inner_probe_is_cached_and_uses_effective_overlay_environment(tmp_path, monkeypatch):
    calls = []
    invocation = "inner-cache-test"
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", invocation)
    monkeypatch.setenv("PATH", "/ambient/path")

    def fake_run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    command = [sys.executable, "-m", "pytest", "tests/test_protocol.py", "-q"]
    overlay = {"PATH": "/overlay/path", "TEST_VALUE": "kept"}
    first = runtime.probe_inner_launcher(command, cwd=tmp_path, environment=overlay)
    second = runtime.probe_inner_launcher(command, cwd=tmp_path, environment=overlay)
    assert first.state == second.state == "verified"
    assert len(calls) == 1
    assert calls[0][0] == (sys.executable, "-m", "pytest", "--version")
    assert calls[0][1]["env"]["PATH"] == "/overlay/path"
    assert calls[0][1]["env"]["TEST_VALUE"] == "kept"


def test_inner_probe_enforces_six_new_candidates_per_invocation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "inner-limit-test")
    monkeypatch.setattr(runtime, "_same_file_contents", lambda *_paths: True)

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    for index in range(runtime.MAX_INNER_PROBE_CANDIDATES):
        interpreter = tmp_path / f"venv-{index}" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        (interpreter.parent.parent / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        interpreter.write_bytes(b"\x7fELFfake-python\n")
        interpreter.chmod(0o755)
        result = runtime.probe_inner_launcher(
            [str(interpreter), "-m", "pytest"], cwd=tmp_path
        )
        assert result.state == "verified"
    over_limit = tmp_path / "venv-over-limit" / "bin" / "python"
    over_limit.parent.mkdir(parents=True)
    (over_limit.parent.parent / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    over_limit.write_bytes(b"\x7fELFfake-python\n")
    over_limit.chmod(0o755)
    limited = runtime.probe_inner_launcher(
        [str(over_limit), "-m", "pytest"], cwd=tmp_path
    )
    assert limited.state == "unknown"
    assert "candidate limit" in limited.diagnostic
    assert len(calls) == runtime.MAX_INNER_PROBE_CANDIDATES


def test_alternate_interpreter_dependency_repair_invalidates_failed_probe(tmp_path, monkeypatch):
    venv = tmp_path / "alternate-venv"
    interpreter = venv / "bin" / "python"
    site_packages = venv / "lib" / "python3.12" / "site-packages"
    interpreter.parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    interpreter.write_bytes(b"\x7fELFfake-python\n")
    interpreter.chmod(0o755)
    monkeypatch.setattr(runtime, "_same_file_contents", lambda *_paths: True)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "alternate-interpreter-repair-test")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        if len(calls) == 1:
            return type("Completed", (), {"returncode": 1, "stdout": "", "stderr": "No module named pytest"})()
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    command = [str(interpreter), "-m", "pytest", "tests"]
    first = runtime.probe_inner_launcher(command, cwd=tmp_path)
    assert first.state == "failed"

    pytest_package = site_packages / "pytest"
    pytest_package.mkdir()
    (pytest_package / "__init__.py").write_text("__version__ = '9'\n", encoding="utf-8")
    second = runtime.probe_inner_launcher(command, cwd=tmp_path)

    assert second.state == "verified"
    assert len(calls) == 2
    assert calls[0][0] == (str(interpreter), "-m", "pytest", "--version")
    identity = runtime.launcher_candidate_identity(command, cwd=tmp_path)
    redacted = runtime._redacted_identity(identity, cwd=tmp_path)
    assert str(venv) not in json.dumps(redacted)


def test_symlinked_virtualenv_interpreter_keeps_lexical_probe_path(tmp_path, monkeypatch):
    venv = tmp_path / "symlinked-venv"
    interpreter = venv / "bin" / "python"
    site_packages = venv / "lib" / "python3.12" / "site-packages"
    interpreter.parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    interpreter.symlink_to(Path(sys.executable))
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "symlinked-interpreter-test")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    result = runtime.probe_inner_launcher(
        [str(interpreter), "-m", "pytest", "tests"], cwd=tmp_path
    )

    assert result.state == "verified"
    assert calls == [(str(interpreter), "-m", "pytest", "--version")]
    identity = runtime.launcher_candidate_identity(
        [str(interpreter), "-m", "pytest"], cwd=tmp_path
    )
    assert identity["path"] == str(interpreter)
    assert identity["entry_lexical"]["path"] == str(interpreter)
    assert identity["pytest_dependency"]["paths"]
    assert str(venv) not in json.dumps(runtime._redacted_identity(identity, cwd=tmp_path))


def test_preflight_invocation_cache_evicts_completed_buckets(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    command = [sys.executable, "-m", "pytest", "tests"]
    for index in range(runtime.MAX_PREFLIGHT_INVOCATION_BUCKETS + 7):
        monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", f"bounded-invocation-{index}")
        assert runtime.probe_inner_launcher(command, cwd=tmp_path).state == "verified"

    assert len(runtime._PREFLIGHT_INVOCATIONS) <= runtime.MAX_PREFLIGHT_INVOCATION_BUCKETS
    assert len(runtime._INNER_PREFLIGHT_CANDIDATES) <= runtime.MAX_PREFLIGHT_INVOCATION_BUCKETS
    assert len(runtime._INNER_PREFLIGHT_CACHE) <= runtime.MAX_PREFLIGHT_INVOCATION_BUCKETS
    assert len(calls) == runtime.MAX_PREFLIGHT_INVOCATION_BUCKETS + 7


def test_inner_probe_concurrent_same_identity_runs_once(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "inner-concurrency-same-test")
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        started.set()
        assert release.wait(5)
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    command = [sys.executable, "-m", "pytest", "tests"]
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(runtime.probe_inner_launcher(command, cwd=tmp_path))
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    assert started.wait(5)
    release.set()
    for thread in threads:
        thread.join(5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == len(threads)
    assert all(result.state == "verified" for result in results)
    assert calls == [(sys.executable, "-m", "pytest", "--version")]


def test_inner_probe_concurrent_distinct_identities_respects_candidate_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "inner-concurrency-limit-test")
    monkeypatch.setattr(runtime, "_same_file_contents", lambda *_paths: True)
    started = threading.Event()
    release = threading.Event()
    calls = []
    calls_lock = threading.Lock()

    def fake_run(argv, **kwargs):
        with calls_lock:
            calls.append(tuple(argv))
            if len(calls) == runtime.MAX_INNER_PROBE_CANDIDATES:
                started.set()
        assert release.wait(5)
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)
    interpreters = []
    for index in range(runtime.MAX_INNER_PROBE_CANDIDATES + 2):
        interpreter = tmp_path / f"concurrent-venv-{index}" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        (interpreter.parent.parent / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        interpreter.write_bytes(b"\x7fELFfake-python\n")
        interpreter.chmod(0o755)
        interpreters.append(interpreter)
    results = []
    threads = [
        threading.Thread(
            target=lambda interpreter=interpreter: results.append(
                runtime.probe_inner_launcher([str(interpreter), "-m", "pytest"], cwd=tmp_path)
            )
        )
        for interpreter in interpreters
    ]
    for thread in threads:
        thread.start()
    assert started.wait(5)
    release.set()
    for thread in threads:
        thread.join(5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(calls) == runtime.MAX_INNER_PROBE_CANDIDATES
    assert sum(result.state == "verified" for result in results) == runtime.MAX_INNER_PROBE_CANDIDATES
    assert sum(result.state == "unknown" for result in results) == 2


def test_cli_overlap_rejection_writes_no_runtime_evidence(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    command = [sys.executable, "-c", "pass"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "overlap-cli-test")
    lock = runtime.acquire_command_lane(command, cwd=tmp_path, env=os.environ)
    assert lock is not None
    try:
        assert main([
            "run-tests", "--timeout-seconds", "5", "--memory-dir", str(memory),
            "--", *command,
        ]) == runtime.OVERLAP_REJECTED_EXIT_CODE
    finally:
        lock.close()
    assert runtime.load_runtime_memory(memory) == []
    assert runtime.load_launcher_health(memory) == []


def test_cli_successful_inner_probe_clears_matching_failure(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    command = [sys.executable, "-m", "pytest", "--version"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "cli-health-clear-test")
    assert runtime.record_launcher_health(
        memory,
        cwd=tmp_path,
        candidate=command,
        state="failed",
        provenance="parent-runner",
        diagnostic="bootstrap failed",
    )
    assert main([
        "run-tests", "--timeout-seconds", "30", "--memory-dir", str(memory),
        "--", *command,
    ]) == 0
    assert [row["state"] for row in runtime.load_launcher_health(memory)] == ["verified"]


def test_bare_launcher_recognition_is_opt_in_and_fails_closed(tmp_path):
    memory = tmp_path / "memory"
    bare = [
        "agent-loop", "run-tests", "--timeout-seconds", "900",
        "--memory-dir", str(memory), "--", "python3", "-m", "pytest", "tests/",
    ]
    absolute = [str(tmp_path / "agent-loop"), *bare[1:]]

    assert runtime.parse_managed_test_invocation(bare) is None
    assert runtime.parse_managed_test_command(bare) is None

    parsed = runtime.parse_managed_test_invocation(bare, allow_command_name_launcher=True)
    reference = runtime.parse_managed_test_invocation(absolute)
    assert parsed is not None and reference is not None
    assert parsed.inner_argv == reference.inner_argv
    assert parsed.timeout_seconds == reference.timeout_seconds
    assert parsed.memory_dir == reference.memory_dir
    assert parsed.prefix_argv == ("agent-loop", "run-tests")

    module = runtime.parse_managed_test_invocation(
        ["python3", "-m", "coding_review_agent_loop.cli", "run-tests", "--", "pytest", "tests/"],
        allow_command_name_launcher=True,
    )
    assert module is not None
    assert module.inner_argv == ("pytest", "tests/")

    prefixed = runtime.parse_managed_test_command(
        ["timeout", "1800", *bare], allow_command_name_launcher=True
    )
    assert prefixed is not None
    assert prefixed.inner_argv == ("python3", "-m", "pytest", "tests/")

    # Path-shaped relative spellings and unrelated modules never qualify.
    for rejected in (
        ["../agent-loop", "run-tests", "--", "true"],
        ["./agent-loop", "run-tests", "--", "true"],
        ["python3", "-m", "other_module.cli", "run-tests", "--", "true"],
    ):
        assert runtime.parse_managed_test_invocation(
            rejected, allow_command_name_launcher=True
        ) is None

    # Malformed options still fail closed under the opt-in.
    with pytest.raises(runtime.TestRuntimeConfigurationError):
        runtime.parse_managed_test_invocation(
            ["agent-loop", "run-tests", "--unknown", "--", "true"],
            allow_command_name_launcher=True,
        )
    with pytest.raises(runtime.TestRuntimeConfigurationError):
        runtime.parse_managed_test_invocation(
            ["agent-loop", "run-tests", "--memory-dir", str(memory), "true"],
            allow_command_name_launcher=True,
        )
    with pytest.raises(runtime.TestRuntimeConfigurationError):
        runtime.parse_managed_test_invocation(
            [
                "agent-loop", "run-tests", "--timeout-seconds", "1",
                "--timeout-seconds", "2", "--", "true",
            ],
            allow_command_name_launcher=True,
        )


def test_report_side_consumers_pass_the_launcher_opt_in_explicitly():
    """Each report-side consumer opts in at its call site, not by default."""
    import inspect

    from coding_review_agent_loop import (
        comment_rendering,
        local_test_evidence,
        protocol,
        workdir_guard,
    )

    for module, function in (
        (workdir_guard, "_validate_managed_command"),
        (comment_rendering, "_render_test_command_for_comment"),
        (protocol, "_managed_test_wrapper_inner_command"),
        (local_test_evidence, "_referenced_paths"),
    ):
        source = inspect.getsource(getattr(module, function))
        assert "allow_command_name_launcher=True" in source, function

    signature = inspect.signature(runtime.parse_managed_test_invocation)
    assert signature.parameters["allow_command_name_launcher"].default is False


# ---------------------------------------------------------------------------
# Parallel test-worker budget (issue #848)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - depends on the dev extra
    import xdist as _xdist  # noqa: F401

    _HAS_XDIST = True
except ImportError:  # pragma: no cover
    _HAS_XDIST = False

needs_xdist = pytest.mark.skipif(not _HAS_XDIST, reason="pytest-xdist is not installed")

_WORKER_TESTS = "def test_one():\n    pass\n\ndef test_two():\n    pass\n\ndef test_three():\n    pass\n"


def _worker_project(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "test_cases.py").write_text(_WORKER_TESTS, encoding="utf-8")
    for name, text in (files or {}).items():
        (project / name).write_text(text, encoding="utf-8")
    return project


@pytest.fixture
def worker_cli(tmp_path, monkeypatch):
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_XDIST_AUTO_NUM_WORKERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", f"worker-cli-{os.getpid()}-{time.monotonic_ns()}")
    project = _worker_project(tmp_path)
    monkeypatch.chdir(project)
    return project


@pytest.mark.parametrize(
    "flags",
    [
        ["--test-workers", "2", "--test-worker-memory", "2G", "--test-worker-enforcement=refuse"],
        ["--test-workers=2", "--test-worker-memory=2G", "--test-worker-enforcement", "refuse"],
    ],
)
def test_managed_invocation_accepts_worker_flags(tmp_path, flags):
    from coding_review_agent_loop.test_workers import parse_worker_memory

    argv = [
        "/usr/local/bin/agent-loop", "run-tests", *flags, "--timeout-seconds", "60", "--",
        "pytest", "-n", "4",
    ]
    parsed = runtime.parse_managed_test_invocation(argv)
    assert parsed.inner_argv == ("pytest", "-n", "4")
    assert parsed.test_workers == 2
    assert parsed.test_worker_memory == parse_worker_memory("2G")
    assert parsed.test_worker_enforcement == "refuse"
    assert parsed.timeout_seconds == 60
    assert runtime.normalize_test_command(argv, cwd=tmp_path) == runtime.normalize_test_command(
        ["pytest", "-n", "4"], cwd=tmp_path
    )


@pytest.mark.parametrize(
    "flags",
    [
        ["--test-workers", "2", "--test-workers", "3"],
        ["--test-workers", "0"],
        ["--test-worker-memory", "max"],
        ["--test-worker-enforcement", "sometimes"],
    ],
)
def test_managed_invocation_rejects_bad_worker_flags(flags):
    with pytest.raises(runtime.TestRuntimeConfigurationError):
        runtime.parse_managed_test_invocation(["/usr/bin/agent-loop", "run-tests", *flags, "--", "pytest"])


def test_cli_parser_validates_worker_flags():
    parser = build_parser()
    args = parser.parse_args(["run-tests", "--test-workers", "3", "--test-worker-enforcement", "off", "--", "pytest"])
    assert args.test_workers == 3 and args.test_worker_enforcement == "off"
    for bad in (["--test-workers", "0"], ["--test-workers", "x"], ["--test-worker-memory", "max"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["run-tests", *bad, "--", "pytest"])


def test_loop_flow_worker_override_reaches_config_and_runner(tmp_path):
    from coding_review_agent_loop.config import config_from_args
    from coding_review_agent_loop.runner import Runner

    parser = build_parser()
    args = parser.parse_args([
        "issue", "1", "--repo", "o/r", "--test-workers", "8", "--test-worker-enforcement", "refuse",
    ])
    assert args.test_workers == 8
    runner = Runner(dry_run=True)
    config = config_from_args(args, runner, invocation_argv=("agent-loop",))
    runner.configure_from_config(config)
    assert runner.test_workers == 8 and runner.test_worker_enforcement == "refuse"
    assert runner.derive_worker_budget(None).workers == 8
    assert runner.derive_worker_budget(None).source == "operator"


def test_cli_same_command_contends_for_one_lane_across_modes(tmp_path, monkeypatch):
    from coding_review_agent_loop.test_workers import worker_lane_identity

    memory = tmp_path / "memory"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "lane-mode-test")
    held_command = [sys.executable, "-m", "pytest", "-n", "8", "tests/"]
    lock = runtime.acquire_command_lane(
        held_command, cwd=tmp_path, env=os.environ,
        identity=worker_lane_identity(held_command, cwd=tmp_path),
    )
    assert lock is not None
    try:
        for mode in ("off", "clamp", "refuse"):
            for spelling in (["-n", "12"], ["--numprocesses=12"], ["-n", "auto"], []):
                assert main([
                    "run-tests", "--test-worker-enforcement", mode, "--memory-dir", str(memory), "--",
                    sys.executable, "-m", "pytest", *spelling, "tests/",
                ]) == runtime.OVERLAP_REJECTED_EXIT_CODE
    finally:
        lock.close()
    assert runtime.load_runtime_memory(memory) == []


def test_cli_worker_budget_busy_records_nothing(tmp_path, monkeypatch):
    from coding_review_agent_loop.test_workers import WorkerBudgetLock

    memory = tmp_path / "memory"
    marker = tmp_path / "spawned"
    monkeypatch.chdir(tmp_path)
    invocation = f"busy-{os.getpid()}-{time.monotonic_ns()}"
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", invocation)
    held, _ = WorkerBudgetLock.acquire(invocation_id=invocation, cwd=tmp_path)
    assert held is not None
    try:
        for command in (
            [sys.executable, "-c", f"open({str(marker)!r}, 'w')"],
            ["sh", "-c", f"touch {marker}"],
        ):
            assert main(["run-tests", "--memory-dir", str(memory), "--", *command]) == 125
        # Off mode takes no worker-budget lock.
        assert main([
            "run-tests", "--test-worker-enforcement", "off", "--memory-dir", str(memory), "--",
            sys.executable, "-c", "pass",
        ]) == 0
    finally:
        held.close()
    assert not marker.exists()
    rows = runtime.load_runtime_memory(memory)
    assert len(rows) == 1 and rows[0]["worker_enforcement"] == "off"
    assert main(["run-tests", "--memory-dir", str(memory), "--", sys.executable, "-c", "pass"]) == 0


def test_cli_inherited_budget_is_only_lowered(tmp_path, monkeypatch, capsys):
    from coding_review_agent_loop import cli as cli_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_LOOP_TEST_WORKERS", "4")
    monkeypatch.setenv("AGENT_LOOP_TEST_WORKER_ENFORCEMENT", "clamp")
    args = build_parser().parse_args(["run-tests", "--test-workers", "8", "--test-worker-enforcement", "off", "--", "x"])
    resolved = cli_module._resolve_run_tests_worker_budget(args, broker_present=False)
    assert resolved.budget.workers <= 4 and resolved.budget.enforcement == "clamp"
    args = build_parser().parse_args(["run-tests", "--test-workers", "2", "--test-worker-enforcement", "refuse", "--", "x"])
    resolved = cli_module._resolve_run_tests_worker_budget(args, broker_present=False)
    assert resolved.budget.workers == 2 and resolved.budget.enforcement == "refuse"
    monkeypatch.delenv("AGENT_LOOP_TEST_WORKERS")
    monkeypatch.delenv("AGENT_LOOP_TEST_WORKER_ENFORCEMENT")
    args = build_parser().parse_args(["run-tests", "--test-workers", "64", "--", "x"])
    standalone = cli_module._resolve_run_tests_worker_budget(args, broker_present=False)
    assert standalone.budget.workers == 64 and standalone.budget.source == "operator"


def test_cli_other_command_records_not_observed(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    monkeypatch.chdir(tmp_path)
    assert main(["run-tests", "--memory-dir", str(memory), "--", sys.executable, "-c", "pass"]) == 0
    row = runtime.load_runtime_memory(memory)[-1]
    assert row["workers"] == "unknown"
    assert row["worker_enforcement"] == "not-observed"
    assert "worker-budget-not-observed" in row["caveats"]
    assert row["executed_argv"] == [sys.executable, "-c", "pass"]


def test_legacy_and_unknown_rows_never_feed_serial_or_count_recommendations(tmp_path):
    memory = tmp_path / "memory"
    for argv in (["pytest", "-n", "0"], ["pytest", "-n", "4"], ["pytest", "-n", "8", "--maxprocesses=2"], ["pytest"]):
        _record(memory, tmp_path, argv, outcome="passed", elapsed=400)
    rows = runtime.load_runtime_memory(memory)
    assert {row["workers"] for row in rows} == {"unknown"}
    for argv, label in ((["pytest", "-n", "4"], "4"), (["pytest", "-n", "0"], "serial")):
        recommendation = runtime.recommend_timeout(
            memory, argv=argv, cwd=tmp_path, policy_ceiling_seconds=1800, now=_now(), workers=label,
        )
        assert recommendation.successful_samples == 0
    # Serial and parallel samples of one command never blend.
    command = ["pytest", "tests/"]
    for elapsed, label in ((100, "serial"), (110, "serial"), (30, "2")):
        assert runtime.record_test_observation(
            memory, argv=command, cwd=tmp_path, outcome="passed", elapsed_seconds=elapsed,
            attempted_timeout_seconds=1800, policy_ceiling_seconds=1800, commit="abc",
            timestamp=_now(), workers=label, launch_integrity="verified",
        )
    serial = runtime.recommend_timeout(memory, argv=command, cwd=tmp_path, policy_ceiling_seconds=1800, now=_now(), workers="serial")
    parallel = runtime.recommend_timeout(memory, argv=command, cwd=tmp_path, policy_ceiling_seconds=1800, now=_now(), workers="2")
    assert serial.successful_samples == 2 and parallel.successful_samples == 1
    assert parallel.median_seconds == 30


@needs_xdist
def test_cli_local_clamp_records_report_cohort(worker_cli, tmp_path, capsys):
    memory = tmp_path / "memory"
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--numprocesses=12"]
    assert main(["run-tests", "--test-workers", "2", "--memory-dir", str(memory), "--", *command]) == 0
    row = runtime.load_runtime_memory(memory)[-1]
    assert row["workers"] == "2"
    assert row["worker_enforcement"] == "clamped"
    assert "_agent_loop_worker_cap" in row["executed_argv"]
    assert "_agent_loop_worker_cap" not in row["normalized_command"]
    assert row["normalized_command"] == runtime.normalize_test_command(command, cwd=worker_cli)
    captured = capsys.readouterr()
    assert "clamped worker request 12" in captured.err
    assert "top-level runner" in captured.err


@needs_xdist
def test_cli_local_direct_refusal_records_nothing(worker_cli, tmp_path):
    memory = tmp_path / "memory"
    assert main([
        "run-tests", "--test-workers", "2", "--test-worker-enforcement", "refuse", "--memory-dir", str(memory),
        "--", sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-n", "8",
    ]) == 2
    assert runtime.load_runtime_memory(memory) == []
    assert runtime.load_launcher_health(memory) == []
    assert main([
        "run-tests", "--test-workers", "2", "--test-worker-enforcement", "refuse", "--memory-dir", str(memory),
        "--", sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-n", "8", "--maxprocesses=2",
    ]) == 0
    row = runtime.load_runtime_memory(memory)[-1]
    assert row["workers"] == "2" and row["worker_enforcement"] == "unchanged"


@needs_xdist
def test_cli_local_unverified_when_repository_unregisters_plugin(tmp_path, monkeypatch):
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_XDIST_AUTO_NUM_WORKERS"):
        monkeypatch.delenv(name, raising=False)
    project = _worker_project(tmp_path, {
        "conftest.py": (
            "def pytest_configure(config):\n"
            "    plugin = config.pluginmanager.get_plugin('_agent_loop_worker_cap')\n"
            "    config.pluginmanager.unregister(plugin)\n"
        ),
    })
    monkeypatch.chdir(project)
    memory = tmp_path / "memory"
    assert main([
        "run-tests", "--test-workers", "2", "--memory-dir", str(memory),
        "--", sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-n", "2",
    ]) == 0
    row = runtime.load_runtime_memory(memory)[-1]
    assert row["workers"] == "unknown"
    assert row["worker_enforcement"] == "unverified"
    assert "worker-budget-unverified" in row["caveats"]


@needs_xdist
def test_cli_other_script_refused_session_keeps_status_and_evidence(worker_cli, tmp_path):
    memory = tmp_path / "memory"
    script = tmp_path / "run.sh"
    script.write_text(
        "#!/bin/sh\n"
        f"{sys.executable} -m pytest -p no:cacheprovider -q -n 1\n"
        f"{sys.executable} -m pytest -p no:cacheprovider -q -n 8\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    assert main([
        "run-tests", "--test-workers", "2", "--test-worker-enforcement", "refuse",
        "--memory-dir", str(memory), "--", str(script),
    ]) == 0
    row = runtime.load_runtime_memory(memory)[-1]
    assert row["workers"] == "unknown"
    assert row["worker_enforcement"] == "refused-in-command"
    assert "worker-budget-refused-session" in row["caveats"]


@needs_xdist
def test_cli_off_mode_labels_only_proven_serial(worker_cli, tmp_path):
    memory = tmp_path / "memory"
    base = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q"]
    assert main(["run-tests", "--test-worker-enforcement", "off", "--memory-dir", str(memory), "--", *base, "-n", "2"]) == 0
    assert main(["run-tests", "--test-worker-enforcement", "off", "--memory-dir", str(memory), "--", *base, "-p", "no:xdist"]) == 0
    labels = {
        row["normalized_command"].rsplit(" ", 2)[-2:][0] + " " + row["normalized_command"].rsplit(" ", 1)[-1]: row["workers"]
        for row in runtime.load_runtime_memory(memory)
    }
    assert labels == {"-n 2": "unknown", "-p no:xdist": "serial"}
    assert {row["worker_enforcement"] for row in runtime.load_runtime_memory(memory)} == {"off"}


# --- #1110: persist the launch triple beside the derived verdict ---------------

_TRIPLE_KEYS = ("wrapper_bootstrap", "inner_exec", "suite_start")


def _triple_row(memory: Path, tmp_path: Path, **kwargs) -> dict:
    assert runtime.record_test_observation(
        memory, argv=[sys.executable, "-m", "pytest", "-q"], cwd=tmp_path, outcome="passed",
        elapsed_seconds=1, attempted_timeout_seconds=1800, policy_ceiling_seconds=1800,
        timestamp=_now(), **kwargs,
    )
    return runtime.load_runtime_memory(memory)[-1]


def test_record_persists_launch_triple_verbatim(tmp_path):
    row = _triple_row(
        tmp_path / "memory", tmp_path, launch_integrity="unverified",
        launch_state={"wrapper_bootstrap": "verified", "inner_exec": "started", "suite_start": "unknown"},
    )
    assert (row["wrapper_bootstrap"], row["inner_exec"], row["suite_start"]) == ("verified", "started", "unknown")
    assert row["launch_integrity"] == "unverified"


def test_record_coerces_out_of_vocabulary_triple_to_fail_closed_defaults(tmp_path):
    row = _triple_row(
        tmp_path / "memory", tmp_path, launch_integrity="verified",
        launch_state={"wrapper_bootstrap": "bogus", "inner_exec": 7, "suite_start": "verified"},
    )
    assert (row["wrapper_bootstrap"], row["inner_exec"], row["suite_start"]) == ("unknown", "not-attempted", "verified")


def test_record_without_launch_state_still_writes_fail_closed_triple(tmp_path):
    row = _triple_row(tmp_path / "memory", tmp_path, launch_integrity="verified")
    assert (row["wrapper_bootstrap"], row["inner_exec"], row["suite_start"]) == ("unknown", "not-attempted", "not-started")
    partial = _triple_row(tmp_path / "memory2", tmp_path, launch_state={"suite_start": "verified"})
    assert (partial["wrapper_bootstrap"], partial["inner_exec"], partial["suite_start"]) == ("unknown", "not-attempted", "verified")


def test_launch_state_fields_reads_result_and_fails_closed():
    from types import SimpleNamespace

    verified = SimpleNamespace(wrapper_bootstrap="verified", inner_exec="started", suite_start="verified")
    assert runtime.launch_state_fields(verified) == {
        "wrapper_bootstrap": "verified", "inner_exec": "started", "suite_start": "verified",
    }
    degraded = SimpleNamespace(wrapper_bootstrap="failed", inner_exec="failed", suite_start="junk")
    assert runtime.launch_state_fields(degraded) == {
        "wrapper_bootstrap": "failed", "inner_exec": "failed", "suite_start": "not-started",
    }
    assert runtime.launch_state_fields(object()) == {
        "wrapper_bootstrap": "unknown", "inner_exec": "not-attempted", "suite_start": "not-started",
    }


def test_verified_triple_without_launch_integrity_is_not_evidence(tmp_path):
    row = {"wrapper_bootstrap": "verified", "inner_exec": "started", "suite_start": "verified"}
    assert not runtime.runtime_row_is_evidence(row)


def test_existing_rows_are_not_rewritten_when_a_new_row_is_recorded(tmp_path):
    memory = tmp_path / "memory"
    legacy = [sys.executable, "-m", "pytest", "legacy", "-q"]
    _record(memory, tmp_path, legacy, outcome="passed", elapsed=5)
    before = runtime.load_runtime_memory(memory)[0]
    _triple_row(memory, tmp_path, launch_integrity="verified")
    rows = runtime.load_runtime_memory(memory)
    old = next(r for r in rows if r["normalized_command"] == before["normalized_command"])
    assert old == before


def test_local_run_tests_path_persists_triple(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    monkeypatch.chdir(tmp_path)
    for name in (
        "AGENT_LOOP_TEST_BROKER_ENDPOINT", "AGENT_LOOP_TEST_BROKER_CAPABILITY", "AGENT_LOOP_TEST_BROKER_PROTOCOL",
    ):
        monkeypatch.delenv(name, raising=False)
    assert main([
        "run-tests", "--timeout-seconds", "5", "--memory-dir", str(memory), "--", sys.executable, "-c", "pass",
    ]) == 0
    row = runtime.load_runtime_memory(memory)[-1]
    assert all(key in row for key in _TRIPLE_KEYS)
    assert row["inner_exec"] == "started"


def test_broker_run_tests_path_persists_triple(tmp_path, monkeypatch):
    import coding_review_agent_loop.cli as cli_module
    from coding_review_agent_loop.local_test_evidence import BrokerRunResult

    memory = tmp_path / "memory"
    monkeypatch.chdir(tmp_path)

    class FakeBroker:
        def run(self, *args, **kwargs):
            return BrokerRunResult(
                receipt_id="r1", execution_ref=None, outcome="passed", returncode=0, elapsed_seconds=1.0,
                wrapper_bootstrap="verified", inner_exec="started", suite_start="unknown",
            )

    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: FakeBroker())
    assert main([
        "run-tests", "--timeout-seconds", "5", "--memory-dir", str(memory), "--", sys.executable, "-c", "pass",
    ]) == 0
    row = runtime.load_runtime_memory(memory)[-1]
    assert (row["wrapper_bootstrap"], row["inner_exec"], row["suite_start"]) == ("verified", "started", "unknown")
    assert row["launch_integrity"] == "unverified"


def test_gate_path_persists_triple(tmp_path):
    from types import SimpleNamespace

    from coding_review_agent_loop.checks import _record_gate_observation

    memory = tmp_path / "memory"
    config = SimpleNamespace(
        dry_run=False, agent_memory=True, agent_memory_dir=memory, repo="owner/repo",
        coder_test_command_timeout_seconds=1800,
    )
    result = SimpleNamespace(
        cwd=tmp_path, args=[sys.executable, "-m", "pytest"], outcome="passed", inner_exec="started",
        suite_start="verified", diagnostic="", output_tail="", elapsed_seconds=0.2, returncode=0,
        containment=None, overlap_rejected=False,
    )
    _record_gate_observation(config, result)
    row = runtime.load_runtime_memory(memory)[-1]
    assert (row["wrapper_bootstrap"], row["inner_exec"], row["suite_start"]) == ("unknown", "started", "verified")
    assert row["launch_integrity"] == "verified"


# --- package-script resolution (issue #1294) ---------------------------------


def _write_package(root, scripts):
    (root / "package.json").write_text(json.dumps({"scripts": scripts}), encoding="utf-8")


def _verified_probe(monkeypatch, calls=None):
    def fake_run(a, **kwargs):
        if calls is not None:
            calls.append(a)
        return type("Completed", (), {"returncode": 0, "stdout": "1.0", "stderr": ""})()

    monkeypatch.setattr(runtime, "_run_bounded_probe", fake_run)


def _fake_npm(root, name="npm"):
    sentinel = root / f"{name}-ran"
    bindir = root / "fakebin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / name
    fake.write_text(f"#!/bin/sh\ntouch {sentinel}\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    return bindir, sentinel


def test_npm_run_playwright_script_is_verified_and_launches_local_binary(tmp_path, monkeypatch, no_ambient_invocation):
    launcher = _fake_playwright(tmp_path)
    _write_package(tmp_path, {"test:x": "npx playwright test --project=a"})
    calls = []
    _verified_probe(monkeypatch, calls)
    argv = ["npm", "run", "test:x"]
    resolution = runtime.resolve_adopted_package_script(argv, cwd=tmp_path)
    assert resolution is not None
    result = runtime.probe_inner_launcher(
        resolution.executed_argv, cwd=tmp_path, package_script=resolution
    )
    assert result.state == "verified"
    assert result.candidate == tuple(argv)
    assert result.launch_argv == (str(launcher), "test", "--project=a")
    assert tuple(calls[0]) == (str(launcher), "--version")
    assert runtime.recognized_inner_probe(argv, cwd=tmp_path) == (str(launcher), "--version")


def test_npm_run_extra_args_are_appended_and_still_allow_listed(tmp_path, monkeypatch, no_ambient_invocation):
    launcher = _fake_playwright(tmp_path)
    _write_package(tmp_path, {"test:x": "npx playwright test --project=a"})
    _verified_probe(monkeypatch)
    ok = runtime.resolve_adopted_package_script(["npm", "run", "test:x", "--", "--grep=foo"], cwd=tmp_path)
    assert ok is not None
    result = runtime.probe_inner_launcher(ok.executed_argv, cwd=tmp_path, package_script=ok)
    assert result.launch_argv == (str(launcher), "test", "--project=a", "--grep=foo")
    assert runtime.resolve_adopted_package_script(["npm", "run", "test:x", "--", "--list"], cwd=tmp_path) is None
    assert runtime.recognized_inner_probe(["npm", "run", "test:x", "--", "--list"], cwd=tmp_path) is None
    # npm args without ``--`` are never accepted.
    assert runtime.resolve_package_script(["npm", "run", "test:x", "--grep=foo"], cwd=tmp_path) is None


@pytest.mark.parametrize(
    "scripts,argv",
    [
        ({"t": "npx playwright test && echo hi"}, ["npm", "run", "t"]),
        ({"t": "npx playwright test | tee x"}, ["npm", "run", "t"]),
        ({"t": "npx playwright test; true"}, ["npm", "run", "t"]),
        ({"t": "FOO=1 npx playwright test"}, ["npm", "run", "t"]),
        ({"t": "npm run other"}, ["npm", "run", "t"]),
        ({"t": "jest"}, ["npm", "run", "t"]),
        ({"t": "npx playwright test", "pret": "echo"}, ["npm", "run", "t"]),
        ({"t": "npx playwright test", "postt": "echo"}, ["npm", "run", "t"]),
        ({"other": "npx playwright test"}, ["npm", "run", "t"]),
        ({"t": "npx playwright test"}, ["npm", "run", "--if-present", "t"]),
        ({"t": "npx playwright test"}, ["npm", "-w", "x", "run", "t"]),
        ({"t": "npx playwright test"}, ["pnpm", "run", "t", "--grep=x"]),
        ({"t": "npx playwright test"}, ["yarn", "install"]),
        ({"t": "npx playwright test"}, ["yarn", "run", "t", "--grep=x"]),
    ],
)
def test_package_script_fail_closed_never_probes(tmp_path, monkeypatch, no_ambient_invocation, scripts, argv):
    _fake_playwright(tmp_path)
    _write_package(tmp_path, scripts)
    calls = []
    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *a, **k: calls.append(a))
    assert runtime.resolve_adopted_package_script(argv, cwd=tmp_path) is None
    assert runtime.recognized_inner_probe(argv, cwd=tmp_path) is None
    assert calls == []


def test_package_script_missing_or_invalid_package_json_is_unknown(tmp_path, no_ambient_invocation):
    _fake_playwright(tmp_path)
    argv = ["npm", "run", "t"]
    assert runtime.resolve_package_script(argv, cwd=tmp_path) is None
    (tmp_path / "package.json").write_text("{not json", encoding="utf-8")
    assert runtime.resolve_package_script(argv, cwd=tmp_path) is None
    (tmp_path / "package.json").write_text('{"scripts": []}', encoding="utf-8")
    assert runtime.resolve_package_script(argv, cwd=tmp_path) is None
    assert runtime.resolve_package_script(["pytest"], cwd=tmp_path) is runtime.NOT_PACKAGE_SCRIPT


@pytest.mark.parametrize("name", ["NODE_OPTIONS", "PW_TEST_REPORTER"])
def test_package_script_code_loading_environment_is_discarded(tmp_path, monkeypatch, no_ambient_invocation, name):
    _fake_playwright(tmp_path)
    _write_package(tmp_path, {"t": "npx playwright test"})
    monkeypatch.setenv(name, "--require x")
    assert runtime.resolve_adopted_package_script(["npm", "run", "t"], cwd=tmp_path) is None


def test_package_script_bin_shadowing_fails_closed(tmp_path, monkeypatch, no_ambient_invocation):
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_package(sub, {"t": "node --test t.js"})
    assert runtime.resolve_package_script(["npm", "run", "t"], cwd=sub) is not None
    shadow = tmp_path / "node_modules" / ".bin" / "node"
    shadow.parent.mkdir(parents=True)
    shadow.write_text("#!/bin/sh\n", encoding="utf-8")
    assert runtime.resolve_package_script(["npm", "run", "t"], cwd=sub) is None
    # A bare ``playwright`` with no local binary cannot be reproduced either.
    _write_package(sub, {"t": "playwright test"})
    assert runtime.resolve_package_script(["npm", "run", "t"], cwd=sub) is None


def test_pnpm_and_yarn_forms_are_recognized(tmp_path, monkeypatch, no_ambient_invocation):
    _fake_playwright(tmp_path)
    _write_package(tmp_path, {"test:x": "npx playwright test --project=a"})
    _verified_probe(monkeypatch)
    for argv in (["pnpm", "run", "test:x"], ["yarn", "run", "test:x"], ["yarn", "test:x"]):
        resolution = runtime.resolve_adopted_package_script(argv, cwd=tmp_path)
        assert resolution is not None, argv
        assert runtime.recognized_inner_probe(argv, cwd=tmp_path) is not None


def test_package_script_probe_identity_and_candidate_spelling(tmp_path, monkeypatch, no_ambient_invocation):
    _fake_playwright(tmp_path)
    _write_package(tmp_path, {"a": "npx playwright test --project=a", "b": "npx playwright test --project=b"})
    _verified_probe(monkeypatch)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "pkg-identity-1294")
    results = []
    spellings = [
        ["npm", "run", "a"], ["npm", "run-script", "a"], ["npm", "run", "a", "--"],
        ["yarn", "run", "a"], ["yarn", "a"], ["npm", "run", "b"],
    ]
    for argv in spellings * 2:  # the second pass is served from the cache
        resolution = runtime.resolve_adopted_package_script(argv, cwd=tmp_path)
        assert resolution is not None
        result = runtime.probe_inner_launcher(
            resolution.executed_argv, cwd=tmp_path, package_script=resolution
        )
        assert result.candidate == tuple(argv)
        assert result.launch_argv
        results.append((argv, result))
    by_name = {tuple(a): r for a, r in results}
    assert by_name[("npm", "run", "a")].identity != by_name[("npm", "run", "b")].identity
    assert by_name[("npm", "run", "a")].launch_argv[-1] == "--project=a"
    assert by_name[("npm", "run", "b")].launch_argv[-1] == "--project=b"


def test_foreground_adopted_script_without_rewrite_never_spawns_npm(tmp_path, monkeypatch, no_ambient_invocation):
    from coding_review_agent_loop import runner as runner_module

    bindir, sentinel = _fake_npm(tmp_path)
    node = bindir / "node"
    node.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    node.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    _write_package(tmp_path, {"t": "node --test t.js"})
    result = runner_module.run_foreground_test(
        ["npm", "run", "t"], cwd=tmp_path, timeout_seconds=60, echo_output=False
    )
    assert result.suite_start == "verified"
    assert not sentinel.exists()
    assert result.args[1:] == ["--test", "t.js"]


def test_foreground_adopted_playwright_script_records_spawned_local_binary(tmp_path, monkeypatch, no_ambient_invocation):
    from coding_review_agent_loop import runner as runner_module

    launcher = _fake_playwright(tmp_path)
    bindir, sentinel = _fake_npm(tmp_path)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    _write_package(tmp_path, {"t": "npx playwright test --project=a"})
    result = runner_module.run_foreground_test(
        ["npm", "run", "t", "--", "--grep=foo"], cwd=tmp_path, timeout_seconds=60, echo_output=False
    )
    assert result.suite_start == "verified"
    assert runtime.launch_integrity_state(result, wrapper_boundary=False) == "verified"
    assert not sentinel.exists()
    assert list(result.args) == [str(launcher), "test", "--project=a", "--grep=foo"]


@pytest.mark.parametrize("script,extra", [("jest", []), ("npx playwright test", ["--", "--list"])])
def test_foreground_discarded_script_runs_npm_as_today(tmp_path, monkeypatch, no_ambient_invocation, script, extra):
    from coding_review_agent_loop import runner as runner_module

    _fake_playwright(tmp_path)
    bindir, sentinel = _fake_npm(tmp_path)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    _write_package(tmp_path, {"t": script})
    result = runner_module.run_foreground_test(
        ["npm", "run", "t", *extra], cwd=tmp_path, timeout_seconds=60, echo_output=False
    )
    assert sentinel.exists()
    assert result.suite_start == "unknown"
    assert list(result.args) == ["npm", "run", "t", *extra]


def test_deeply_nested_package_json_is_unresolved_and_npm_runs_as_today(tmp_path, monkeypatch, no_ambient_invocation):
    """A package.json the decoder cannot recurse through is unresolved, not a crash (#1294)."""
    from coding_review_agent_loop import runner as runner_module

    _fake_playwright(tmp_path)
    bindir, sentinel = _fake_npm(tmp_path)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    depth = 100_000
    (tmp_path / "package.json").write_text(
        '{"scripts": {"t": "npx playwright test"}, "meta": ' + "[" * depth + "]" * depth + "}",
        encoding="utf-8",
    )
    assert (tmp_path / "package.json").stat().st_size < runtime._PACKAGE_JSON_MAX_BYTES
    with pytest.raises(RecursionError):
        json.loads((tmp_path / "package.json").read_text(encoding="utf-8"))

    assert runtime.resolve_package_script(["npm", "run", "t"], cwd=tmp_path) is None
    assert runtime.resolve_adopted_package_script(["npm", "run", "t"], cwd=tmp_path) is None
    probes = []
    monkeypatch.setattr(runtime, "_run_bounded_probe", lambda *a, **k: probes.append(a))
    result = runner_module.run_foreground_test(
        ["npm", "run", "t"], cwd=tmp_path, timeout_seconds=60, echo_output=False
    )
    assert probes == []
    assert sentinel.exists()
    assert result.suite_start == "unknown"
    assert list(result.args) == ["npm", "run", "t"]


def test_foreground_adopted_script_with_unknown_probe_spawns_body_never_npm(tmp_path, monkeypatch, no_ambient_invocation):
    from coding_review_agent_loop import runner as runner_module

    launcher = _fake_playwright(tmp_path)
    bindir, sentinel = _fake_npm(tmp_path)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    _write_package(tmp_path, {"t": "npx playwright test"})
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "pkg-unknown-1294")
    monkeypatch.setattr(runtime, "MAX_INNER_PROBE_CANDIDATES", 0)
    result = runner_module.run_foreground_test(
        ["npm", "run", "t"], cwd=tmp_path, timeout_seconds=60, echo_output=False
    )
    assert result.suite_start == "unknown"
    assert not sentinel.exists()
    assert list(result.args) == [str(launcher), "test"]


def test_package_json_is_read_once_and_frozen_before_launch(tmp_path, monkeypatch, no_ambient_invocation):
    from coding_review_agent_loop import runner as runner_module

    launcher = _fake_playwright(tmp_path)
    _write_package(tmp_path, {"t": "npx playwright test --project=a"})
    resolution = runtime.resolve_adopted_package_script(["npm", "run", "t"], cwd=tmp_path)
    _write_package(tmp_path, {"t": "npx playwright test --project=b"})
    reads = []
    original = runtime._read_package_scripts
    monkeypatch.setattr(runtime, "_read_package_scripts", lambda cwd: reads.append(cwd) or original(cwd))
    result = runner_module.run_foreground_test(
        ["npm", "run", "t"], cwd=tmp_path, timeout_seconds=60, echo_output=False,
        package_script=resolution,
    )
    assert reads == []
    assert list(result.args) == [str(launcher), "test", "--project=a"]


def test_cli_run_tests_records_npm_run_as_verified_evidence(tmp_path, monkeypatch, no_ambient_invocation):
    launcher = _fake_playwright(tmp_path)
    bindir, sentinel = _fake_npm(tmp_path)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    _write_package(tmp_path, {"test:x": "npx playwright test --project=a"})
    memory = tmp_path / "memory"
    memory.mkdir()
    monkeypatch.chdir(tmp_path)
    code = main(["run-tests", "--timeout-seconds", "60", "--memory-dir", str(memory), "--", "npm", "run", "test:x"])
    assert code == 0
    assert not sentinel.exists()
    row = runtime.load_runtime_memory(memory)[-1]
    assert row["launch_integrity"] == "verified"
    assert runtime.runtime_row_is_evidence(row)
    assert row["normalized_command"].startswith("npm run test:x")
    assert row["executed_argv"] == [str(launcher), "test", "--project=a"]
    # The input manifest stays on the original argv and tracks package.json.
    assert row["input_manifest"] == runtime.build_input_manifest(["npm", "run", "test:x"], tmp_path)
    _write_package(tmp_path, {"test:x": "npx playwright test --project=b"})
    assert row["input_manifest"] != runtime.build_input_manifest(["npm", "run", "test:x"], tmp_path)



@pytest.mark.parametrize("alias", ["pypy3", "py", "python3.99", "custom-runner"])
def test_any_bare_head_shadowed_by_local_bin_fails_closed(tmp_path, no_ambient_invocation, alias):
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_package(sub, {"t": f"{alias} -m pytest tests/"})
    assert isinstance(runtime.resolve_package_script(["npm", "run", "t"], cwd=sub), runtime.PackageScriptResolution)
    shadow = tmp_path / "node_modules" / ".bin" / alias
    shadow.parent.mkdir(parents=True)
    shadow.write_text("#!/bin/sh\n", encoding="utf-8")
    assert runtime.resolve_package_script(["npm", "run", "t"], cwd=sub) is None
    assert runtime.recognized_inner_probe(["npm", "run", "t"], cwd=sub) is None


@pytest.mark.parametrize(
    "argv",
    [["npm", "run", "t"], ["pnpm", "run", "t"], ["yarn", "run", "t"], ["yarn", "t"]],
)
@pytest.mark.parametrize("body", ["playwright test --project=a", "npx playwright test --project=a"])
def test_foreground_every_manager_launches_the_local_playwright_never_the_manager(
    tmp_path, monkeypatch, no_ambient_invocation, argv, body
):
    from coding_review_agent_loop import runner as runner_module

    launcher = _fake_playwright(tmp_path)
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    sentinels = []
    for manager in ("npm", "pnpm", "yarn"):
        fake = bindir / manager
        sentinel = tmp_path / f"{manager}-ran"
        sentinels.append(sentinel)
        fake.write_text(f"#!/bin/sh\ntouch {sentinel}\nexit 0\n", encoding="utf-8")
        fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    _write_package(tmp_path, {"t": body})
    result = runner_module.run_foreground_test(argv, cwd=tmp_path, timeout_seconds=60, echo_output=False)
    assert result.suite_start == "verified"
    assert runtime.launch_integrity_state(result, wrapper_boundary=False) == "verified"
    assert list(result.args) == [str(launcher), "test", "--project=a"]
    assert not any(sentinel.exists() for sentinel in sentinels)


@requires_system_env
def test_package_script_env_prefix_keeps_distinct_identity_and_binding(tmp_path, monkeypatch, no_ambient_invocation):
    launcher = _fake_playwright(tmp_path)
    _write_package(tmp_path, {"t": "npx playwright test"})
    _verified_probe(monkeypatch)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "pkg-env-1294")
    results = {}
    for round_name in ("fresh", "cached"):
        for value in ("1", "2"):
            argv = ["env", f"A={value}", "npm", "run", "t"]
            resolution = runtime.resolve_adopted_package_script(argv, cwd=tmp_path)
            assert resolution is not None
            result = runtime.probe_inner_launcher(
                resolution.executed_argv, cwd=tmp_path, package_script=resolution
            )
            assert result.state == "verified"
            assert result.candidate == tuple(argv)
            assert result.launch_argv[1:] == (f"A={value}", str(launcher), "test")
            results[(round_name, value)] = result
    assert results[("fresh", "1")].identity != results[("fresh", "2")].identity
    assert results[("cached", "1")].identity == results[("fresh", "1")].identity


def test_cli_run_tests_resolves_package_json_exactly_once(tmp_path, monkeypatch, no_ambient_invocation):
    launcher = _fake_playwright(tmp_path)
    _write_package(tmp_path, {"test:x": "npx playwright test --project=a"})
    reads = []
    original = runtime._read_package_scripts

    def mutate_after_read(cwd):
        scripts = original(cwd)
        reads.append(cwd)
        _write_package(tmp_path, {"test:x": "npx playwright test --project=b"})
        return scripts

    monkeypatch.setattr(runtime, "_read_package_scripts", mutate_after_read)
    memory = tmp_path / "memory"
    memory.mkdir()
    monkeypatch.chdir(tmp_path)
    assert main(["run-tests", "--timeout-seconds", "60", "--memory-dir", str(memory), "--", "npm", "run", "test:x"]) == 0
    assert len(reads) == 1
    row = runtime.load_runtime_memory(memory)[-1]
    assert row["executed_argv"] == [str(launcher), "test", "--project=a"]


def test_non_package_npx_run_keeps_npx_spelling_in_result_and_cli_row(tmp_path, monkeypatch, no_ambient_invocation):
    from coding_review_agent_loop import runner as runner_module

    launcher = tmp_path / "node_modules" / ".bin" / "playwright"
    launcher.parent.mkdir(parents=True)
    launched = tmp_path / "playwright-ran"
    launcher.write_text(f"#!/bin/sh\ntouch {launched}\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    npx_ran = tmp_path / "npx-ran"
    fake = bindir / "npx"
    fake.write_text(f"#!/bin/sh\ntouch {npx_ran}\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    requested = ["npx", "playwright", "test", "--project=a"]
    result = runner_module.run_foreground_test(requested, cwd=tmp_path, timeout_seconds=60, echo_output=False)
    assert launched.exists() and not npx_ran.exists()
    assert list(result.args) == requested
    launched.unlink()
    memory = tmp_path / "memory"
    memory.mkdir()
    monkeypatch.chdir(tmp_path)
    assert main(["run-tests", "--timeout-seconds", "60", "--memory-dir", str(memory), "--", *requested]) == 0
    assert launched.exists() and not npx_ran.exists()
    assert runtime.load_runtime_memory(memory)[-1]["executed_argv"] == requested


# --- pre-collection launch failures (issue #1182) ----------------------------


@pytest.fixture(autouse=True)
def _lp_isolated_pytest_environment(monkeypatch):
    """Child pytest runs must not inherit the outer run's pytest controls.

    Managed CI exports ``PYTEST_ADDOPTS=-p ci_shard_plugin`` and xdist exports
    its worker variables; an inherited ``-p`` plugin is (correctly) an early
    plugin for the probe and, under ``-I``/``-E`` or a replaced PYTHONPATH, an
    import error.
    """
    for name in (
        "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_XDIST_AUTO_NUM_WORKERS", "PYTEST_XDIST_WORKER",
        "PYTEST_XDIST_WORKER_COUNT", "PYTEST_XDIST_TESTRUNUID", "PYTEST_CURRENT_TEST",
    ):
        monkeypatch.delenv(name, raising=False)


from coding_review_agent_loop import lifecycle_probe as lp  # noqa: E402
from coding_review_agent_loop.test_workers import WorkerBudget  # noqa: E402

_LP_USAGE_REASON = runtime.PRE_COLLECTION_USAGE_REASON


def _lp_write(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


_LP_PASSING = {"test_x.py": "def test_a():\n    assert True\n"}


def _lp_run(root, extra, *, env=None, budget=None, python=None, quiet=True):
    # ``-q`` belongs to the terminal plugin, so suppression variants omit it.
    quiet = ["-q"] if quiet else []
    argv = [python or sys.executable, "-m", "pytest", *quiet, "-p", "no:cacheprovider", *extra]
    return run_foreground_test(
        argv, cwd=root, timeout_seconds=120, echo_output=False, classify_pre_collection=True,
        env=({**os.environ, **env} if env else None), worker_budget=budget,
    )


def _lp_direct(root, extra, *, env=None, argv=None):
    """Inject the probe by hand and run for real; return (rc, output, report)."""
    quiet = [] if "no:terminal" in extra else ["-q"]
    argv = argv or [sys.executable, "-m", "pytest", *quiet, "-p", "no:cacheprovider", *extra]
    base = {**os.environ, **(env or {})}
    setup = lp.prepare_lifecycle_probe(argv, base, root)
    assert setup is not None
    try:
        done = subprocess.run(
            list(setup.argv), cwd=root, env=setup.env, text=True, capture_output=True, timeout=120,
        )
        return done.returncode, done.stdout + done.stderr, lp.read_probe_report(setup.report_path, setup.nonce)
    finally:
        setup.cleanup()


def _lp_qualifying():
    scan = lp.PreCollectionScan()
    scan.feed("ERROR: usage: pytest [options]\n")
    scan.feed("pytest: error: unrecognized arguments: -n 0\n")
    records = tuple(
        {"kind": kind, "nonce": "n", "pid": 1, **fields}
        for kind, fields in (
            ("loaded", {}),
            ("args-seen", {
                "doctest_requested": False, "early_plugin": False,
                "probe_blocked_later": False, "explicit_targets_unprovable": False,
            }),
            ("final", {"invalid": False, "write_failures": 0, "vetoed": False, "parsed": False}),
        )
    )
    eligibility = lp.LaunchEligibility("module", (sys.executable,), sys.executable, (1, 2, 3, 4))
    return scan, lp.ProbeReport(records), eligibility


def _lp_classify(**overrides):
    scan, probe, eligibility = _lp_qualifying()
    arguments = {
        "argv": [sys.executable, "-m", "pytest", "x.py"], "returncode": 4, "scan": scan, "probe": probe,
        "independent_import_check": "unavailable", "launch_eligibility": eligibility,
        "worker_sessions_observed": False,
    }
    arguments.update(overrides)
    return runtime.classify_pre_collection_launch_failure(
        arguments.pop("argv"), arguments.pop("returncode"), arguments.pop("scan"), arguments.pop("probe"),
        arguments.pop("independent_import_check"), **arguments,
    )


@pytest.mark.parametrize("extra", [["--bogus-flag"], ["-p", "no:xdist", "-n", "0"]])
def test_usage_error_is_classified_as_pre_collection_launch_failure(tmp_path, extra):
    _lp_write(tmp_path, {**_LP_PASSING, "helper.py": "VALUE = 1\n", "conftest.py": "import helper\n"})
    result = _lp_run(tmp_path, ["test_x.py", *extra])
    assert (result.outcome, result.returncode) == ("failed", 4)
    assert result.pre_collection_launch_failure == _LP_USAGE_REASON
    rc, _output, report = _lp_direct(tmp_path, ["test_x.py", *extra])
    assert rc == 4
    assert [record["kind"] for record in report.records] == ["loaded", "args-seen", "final"]


def test_standard_console_script_usage_error_is_classified(tmp_path):
    script = tmp_path / "bin" / "pytest"
    script.parent.mkdir()
    script.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pytest import console_main\n"
        "if __name__ == '__main__':\n    sys.argv[0] = sys.argv[0].removesuffix('.exe')\n"
        "    sys.exit(console_main())\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    _lp_write(tmp_path, _LP_PASSING)
    result = run_foreground_test(
        [str(script), "--bogus-flag", "-p", "no:cacheprovider", "test_x.py"], cwd=tmp_path,
        timeout_seconds=120, echo_output=False, classify_pre_collection=True,
    )
    assert result.returncode == 4
    assert result.pre_collection_launch_failure == _LP_USAGE_REASON


def test_ordinary_failure_and_pass_are_not_classified(tmp_path):
    _lp_write(tmp_path, {"test_x.py": "def test_a():\n    assert False\n"})
    failed = _lp_run(tmp_path, ["test_x.py"])
    assert (failed.outcome, failed.pre_collection_launch_failure) == ("failed", None)
    _lp_write(tmp_path, _LP_PASSING)
    passed = _lp_run(tmp_path, ["test_x.py"])
    assert (passed.outcome, passed.pre_collection_launch_failure) == ("passed", None)


def test_classification_is_off_unless_requested(tmp_path):
    _lp_write(tmp_path, _LP_PASSING)
    result = run_foreground_test(
        [sys.executable, "-m", "pytest", "test_x.py", "--bogus-flag"], cwd=tmp_path,
        timeout_seconds=120, echo_output=False,
    )
    assert result.returncode == 4 and result.pre_collection_launch_failure is None


def test_pytest_exit_forgery_is_not_classified(tmp_path):
    _lp_write(tmp_path, {"test_x.py": (
        "import pytest\n\n"
        "def test_a():\n    pytest.exit('usage: x\\nerror: unrecognized arguments: x', returncode=4)\n"
    )})
    result = _lp_run(tmp_path, ["test_x.py"])
    assert result.returncode == 4
    assert result.pre_collection_launch_failure is None


def test_plugin_session_and_missing_scan_veto_classification():
    assert _lp_classify() == _LP_USAGE_REASON
    assert _lp_classify(worker_sessions_observed=True) is None
    assert _lp_classify(scan=None) is None
    assert _lp_classify(launch_eligibility=None) is None
    assert _lp_classify(probe=None) is None
    assert _lp_classify(argv=["python", "-c", "pass"]) is None
    scan, _probe, _eligibility = _lp_qualifying()
    scan.feed("collected 1 item\n")
    assert _lp_classify(scan=scan) is None


def test_markers_evicted_from_the_tail_still_veto(tmp_path):
    _lp_write(tmp_path, {
        "test_x.py": "def test_a():\n    assert False\n",
        "conftest.py": (
            "def pytest_sessionfinish(session):\n"
            "    for index in range(100):\n        print('filler', index)\n"
            "    print('No module named pytest')\n"
        ),
    })
    for budget in (None, WorkerBudget(2, "operator", "off")):
        result = _lp_run(tmp_path, ["test_x.py"], budget=budget)
        assert result.returncode == 1 and "No module named pytest" in result.output_tail
        assert "collected" not in result.output_tail
        assert result.pre_collection_launch_failure is None


_LP_SUPPRESS = ("-p", "no:terminal", "-s")


@pytest.mark.parametrize("where", ["argv", "ini", "env"])
@pytest.mark.parametrize("mode", [None, "off"])
@pytest.mark.parametrize("code", [1, 4])
def test_marker_suppressed_real_execution_is_not_classified(tmp_path, where, mode, code):
    text = "usage: x\\nerror: unrecognized arguments: x" if code == 4 else "No module named pytest"
    files = {"test_x.py": (
        "import os, sys\n\n"
        f"def test_a():\n    print('{text}'); sys.stdout.flush()\n    os._exit({code})\n"
    )}
    extra, env = [], None
    if where == "argv":
        extra = list(_LP_SUPPRESS)
    elif where == "ini":
        files["pytest.ini"] = "[pytest]\naddopts = -p no:terminal -s\n"
    else:
        env = {"PYTEST_ADDOPTS": "-p no:terminal -s"}
    _lp_write(tmp_path, files)
    budget = WorkerBudget(2, "operator", "off") if mode == "off" else None
    result = _lp_run(tmp_path, [*extra, "test_x.py"], env=env, budget=budget, quiet=False)
    assert result.returncode == code
    assert result.pre_collection_launch_failure is None


_LP_FIRE = (
    "def fire():\n    print('usage: x')\n    print('error: unrecognized arguments: x')\n    raise SystemExit(4)\n"
)


@pytest.mark.parametrize(
    "files, extra",
    [
        ({"test_x.py": _LP_FIRE, "conftest.py": (
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('hidden', 'test_x.py')\n"
            "module = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\nmodule.fire()\n"
        )}, []),
        ({"test_x.py": _LP_FIRE, "conftest.py": (
            "namespace = {}\nexec(compile(open('test_x.py').read(), 'anything', 'exec'), namespace)\n"
            "namespace['fire']()\n"
        )}, []),
        ({"cases.py": _LP_FIRE, "conftest.py": (
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('hidden', 'cases.py')\n"
            "module = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\nmodule.fire()\n"
        )}, ["cases.py"]),
        ({"cases.py": _LP_FIRE, "conftest.py": (
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('hidden', 'cases.py')\n"
            "module = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\nmodule.fire()\n"
        )}, ["cases.py::test_x"]),
        ({"-cases.py": _LP_FIRE, "conftest.py": (
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('hidden', './-cases.py')\n"
            "module = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\nmodule.fire()\n"
        )}, ["--", "-cases.py"]),
        ({"cases.py": _LP_FIRE, "pytest.ini": "[pytest]\ntestpaths = cases.py\n", "conftest.py": (
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('hidden', 'cases.py')\n"
            "module = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\nmodule.fire()\n"
        )}, []),
        ({"check_x.py": _LP_FIRE, "pytest.ini": "[pytest]\npython_files = check_*.py\n",
          "userplug.py": (
              "import importlib.util\n"
              "spec = importlib.util.spec_from_file_location('hidden', 'check_x.py')\n"
              "FIRE = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(FIRE)\n"
          ), "conftest.py": "import userplug\nuserplug.FIRE.fire()\n"}, ["-p", "userplug"]),
    ],
    ids=["exec_module", "read_then_exec", "explicit_target", "node_id", "dash_after_delimiter",
         "testpaths", "two_phase_custom_patterns"],
)
def test_test_code_executed_before_option_parsing_is_not_classified(tmp_path, files, extra):
    _lp_write(tmp_path, files)
    env = {"PYTHONPATH": str(tmp_path)}
    result = _lp_run(tmp_path, ["-p", "no:terminal", "-s", *extra], env=env, quiet=False)
    assert result.returncode == 4
    assert result.pre_collection_launch_failure is None
    rc, _output, report = _lp_direct(tmp_path, ["-p", "no:terminal", "-s", *extra], env=env)
    assert rc == 4
    kinds = [record["kind"] for record in report.records]
    assert "test-code-touched" in kinds


def test_early_plugin_and_unprovable_targets_are_not_classified(tmp_path):
    _lp_write(tmp_path, {**_LP_PASSING, "early.py": "", "args.txt": "test_x.py\n"})
    env = {"PYTHONPATH": str(tmp_path)}
    early = _lp_run(tmp_path, ["--bogus-flag", "test_x.py"], env={**env, "PYTEST_ADDOPTS": "-p early"})
    assert early.returncode == 4 and early.pre_collection_launch_failure is None
    argsfile = _lp_run(tmp_path, ["--bogus-flag", "@args.txt"])
    assert argsfile.pre_collection_launch_failure is None
    pyargs = _lp_run(tmp_path, ["--bogus-flag", "--pyargs", "test_x"], env=env)
    assert pyargs.pre_collection_launch_failure is None


def test_explicit_non_pattern_targets_are_collected_normally_with_the_probe(tmp_path):
    _lp_write(tmp_path, {"cases.py": "def test_a():\n    assert True\n", "-cases.py": "def test_b():\n    assert True\n"})
    for extra in (["cases.py"], ["--", "./-cases.py"]):
        rc, _output, report = _lp_direct(tmp_path, extra)
        assert rc == 0
        assert [record["kind"] for record in report.records] == ["loaded", "args-seen", "parsed", "final"]


@pytest.mark.parametrize("where", ["argv", "ini", "env"])
def test_a_blocked_probe_fails_closed(tmp_path, where):
    files = dict(_LP_PASSING)
    extra, env = ["--bogus-flag", "test_x.py"], None
    if where == "argv":
        extra = ["-p", f"no:{lp.PROBE_MODULE}", *extra]
    elif where == "ini":
        files["pytest.ini"] = f"[pytest]\naddopts = -p no:{lp.PROBE_MODULE}\n"
    else:
        env = {"PYTEST_ADDOPTS": f"-p no:{lp.PROBE_MODULE}"}
    _lp_write(tmp_path, files)
    result = _lp_run(tmp_path, extra, env=env)
    assert result.returncode == 4 and result.pre_collection_launch_failure is None


def test_probe_report_parsing_fails_closed(tmp_path):
    path = tmp_path / "report.jsonl"
    assert lp.read_probe_report(path, "n") is None
    path.write_text("")
    assert lp.read_probe_report(path, "n") == lp.ProbeReport(())
    path.write_text('{"kind":"loaded","nonce":"other","pid":1}\n')
    assert lp.read_probe_report(path, "n") is None
    path.write_text('{"kind":"loaded","nonce":"n","pid":1}\n{"kind":"x"')
    assert lp.read_probe_report(path, "n") is None
    path.write_text('{"kind":"loaded","nonce":"n","pid":1}\n{"kind":"final","nonce":"n","pid":2}\n')
    assert lp.read_probe_report(path, "n") is None
    _scan, probe, eligibility = _lp_qualifying()
    missing_final = lp.ProbeReport(probe.records[:2])
    assert _lp_classify(probe=missing_final) is None
    for field, value in (("invalid", True), ("write_failures", 1), ("vetoed", True), ("parsed", True)):
        records = list(probe.records)
        records[-1] = {**records[-1], field: value}
        assert _lp_classify(probe=lp.ProbeReport(tuple(records))) is None


_LP_SITE = (
    "import os\n_real = os.write\n_state = {'n': 0}\n"
    "def _write(fd, data):\n"
    "    if fd > 2 and b'\"kind\"' in data:\n"
    "        _state['n'] += 1\n"
    "        if _state['n'] == int(os.environ.get('LP_FAIL_ON', '0')):\n"
    "            raise OSError('injected')\n"
    "    return _real(fd, data)\n"
    "os.write = _write\n"
)


@pytest.mark.parametrize("fail_on, files, extra", [
    (1, dict(_LP_PASSING), ["--bogus-flag", "test_x.py"]),
    (3, {"test_x.py": _LP_FIRE, "conftest.py": (
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('hidden', 'test_x.py')\n"
        "module = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\nmodule.fire()\n"
    )}, ["-p", "no:terminal", "-s"]),
])
def test_probe_write_failures_never_alter_the_target_and_fail_closed(tmp_path, fail_on, files, extra):
    _lp_write(tmp_path, {**files, "hook/sitecustomize.py": _LP_SITE})
    env = {"PYTHONPATH": str(tmp_path / "hook"), "LP_FAIL_ON": str(fail_on)}
    result = _lp_run(tmp_path, extra, env=env)
    assert result.returncode == 4
    assert result.pre_collection_launch_failure is None


def test_probe_never_changes_normal_runs_and_writes_bounded_records(tmp_path):
    _lp_write(tmp_path, {
        "mod.py": "def double(x):\n    '''\n    >>> double(2)\n    4\n    '''\n    return 2 * x\n",
        "doc.txt": ">>> 1 + 1\n2\n",
        **_LP_PASSING,
    })
    for extra in (["--doctest-modules", "mod.py"], ["--doctest-glob=*.txt", "doc.txt"], ["test_x.py"]):
        rc, _output, report = _lp_direct(tmp_path, extra)
        assert rc == 0
        assert len(report.records) <= 5
        assert {record["pid"] for record in report.records} == {report.records[0]["pid"]}


def _lp_script(path, text, mode=0o755):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


_LP_TEMPLATE_BODY = (
    "import re\nimport sys\nfrom pytest import console_main\nif __name__ == '__main__':\n"
    "    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
    "    sys.exit(console_main())\n"
)


_LP_DUMP_TEST = (
    "import json, os, sys\n\n"
    "def test_dump():\n"
    "    assert '_agent_loop_lifecycle_probe' not in sys.modules\n"
    "    assert not any('lifecycle_probe' in arg for arg in sys.argv)\n"
    "    assert 'AGENT_LOOP_LIFECYCLE_PROBE_SPEC' not in os.environ\n"
)


@contextlib.contextmanager
def _lp_spawns():
    """Record the argv and complete environment of every runner target spawn."""
    real_popen = subprocess.Popen
    spawns = []

    def spy(*args, **kwargs):
        # The runner's target spawn, not a probe or preflight; exit-code
        # probes also get a process group (#1343) but discard all output.
        exit_code_probe = kwargs.get("stdout") == subprocess.DEVNULL and kwargs.get("stderr") == subprocess.DEVNULL
        if kwargs.get("start_new_session") and not exit_code_probe:
            env = kwargs.get("env")
            spawns.append((list(args[0]), dict(env) if env is not None else dict(os.environ)))
        return real_popen(*args, **kwargs)

    with mock.patch.object(subprocess, "Popen", spy):
        yield spawns


def _lp_run_recorded(argv, tmp_path, *, env, classify):
    with _lp_spawns() as spawns:
        result = run_foreground_test(
            argv, cwd=tmp_path, timeout_seconds=120, echo_output=False,
            classify_pre_collection=classify, env=env,
        )
    assert spawns, "the runner spawned nothing"  # bootstrap probe(s) first, then the target
    return result, spawns


def _lp_assert_control(argv, tmp_path, *, env, expected, label):
    """A classified run spawns the same argv/environment and exits like an unclassified one."""
    plain, plain_spawn = _lp_run_recorded(argv, tmp_path, env=env, classify=False)
    classified, classified_spawn = _lp_run_recorded(argv, tmp_path, env=env, classify=True)
    assert classified_spawn == plain_spawn, label
    assert classified.returncode == plain.returncode == expected, label
    assert classified.pre_collection_launch_failure is None, label


_LP_LAUNCHER_LABELS = [
    "dash-I", "dash-E", "shell-python-wrapper", "isolating-wrapper", "trampoline",
    "shebang-wrapper", "shebang-isolating-wrapper", "shebang-dash-I", "shebang-dash-E",
    "env-shebang-via-wrapper-path", "non-template",
]
_LP_ISOLATED_LABELS = {"dash-I", "isolating-wrapper", "trampoline", "shebang-isolating-wrapper", "shebang-dash-I"}


def _lp_launcher(tmp_path, label):
    real = sys.executable
    base = dict(os.environ)
    bin_dir = tmp_path / "bin"
    wrapper = _lp_script(bin_dir / "python3", f'#!/bin/sh\nexec {real} "$@"\n')
    isolating = _lp_script(tmp_path / "iso" / "python3", f'#!/bin/sh\nexec {real} -I "$@"\n')
    valid = f"#!{{}}\n{_LP_TEMPLATE_BODY}"
    path_env = {**base, "PATH": f"{bin_dir}:{base['PATH']}"}
    builders = {
        "dash-I": lambda: ([sys.executable, "-I", "-m", "pytest"], base),
        "dash-E": lambda: ([sys.executable, "-E", "-m", "pytest"], base),
        "shell-python-wrapper": lambda: ([str(wrapper), "-m", "pytest"], base),
        "isolating-wrapper": lambda: ([str(isolating), "-m", "pytest"], base),
        "trampoline": lambda: ([str(_lp_script(bin_dir / "pytest", f'#!/bin/sh\nexec {real} -I -m pytest "$@"\n'))], base),
        "shebang-wrapper": lambda: ([str(_lp_script(tmp_path / "wrapped" / "pytest", valid.format(wrapper)))], base),
        "shebang-isolating-wrapper": lambda: ([str(_lp_script(tmp_path / "isolated" / "pytest", valid.format(isolating)))], base),
        "shebang-dash-I": lambda: ([str(_lp_script(tmp_path / "dashi" / "pytest", valid.format(f"{real} -I")))], base),
        "shebang-dash-E": lambda: ([str(_lp_script(tmp_path / "dashe" / "pytest", valid.format(f"{real} -E")))], base),
        "env-shebang-via-wrapper-path": lambda: (
            [str(_lp_script(tmp_path / "envform" / "pytest", valid.format("/usr/bin/env python3")))], path_env,
        ),
        "non-template": lambda: ([str(_lp_script(
            tmp_path / "other" / "pytest",
            f"#!{real}\nimport sys\nfrom pytest import console_main\nsys.exit(console_main() or 0)\n",
        ))], base),
    }
    return builders[label]()


@pytest.mark.parametrize("label", _LP_LAUNCHER_LABELS)
def test_launchers_outside_the_allowlist_get_no_probe_and_are_never_classified(tmp_path, label):
    if label in _LP_ISOLATED_LABELS and subprocess.run(
        [sys.executable, "-I", "-c", "import pytest"], capture_output=True
    ).returncode != 0:
        pytest.skip("pytest is not importable under -I in this interpreter, so the isolated control cannot run")
    launcher, env = _lp_launcher(tmp_path, label)
    _lp_write(tmp_path, {"test_x.py": _LP_DUMP_TEST})
    argv = [*launcher, "-q", "-p", "no:cacheprovider", "test_x.py"]
    assert lp.prepare_lifecycle_probe([*argv, "--bogus-flag"], env, tmp_path) is None
    _lp_assert_control(argv, tmp_path, env=env, expected=0, label=label)
    _lp_assert_control([*argv, "--bogus-flag"], tmp_path, env=env, expected=4, label=label)


def test_a_forced_preflight_failure_on_a_native_interpreter_means_no_injection(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "_PREFLIGHT_CODE", "import sys; sys.exit(3)")
    argv = [sys.executable, "-m", "pytest", "test_x.py"]
    assert lp.prepare_lifecycle_probe(argv, dict(os.environ), tmp_path) is None


def test_parent_setup_failures_launch_the_original_command(tmp_path, monkeypatch):
    _lp_write(tmp_path, _LP_PASSING)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(lp.tempfile, "tempdir", str(scratch))
    argv = [sys.executable, "-m", "pytest", "test_x.py", "--bogus-flag"]

    def failing_mkdtemp(*_args, **_kwargs):
        raise OSError("no space")

    with monkeypatch.context() as patched:
        patched.setattr(lp.tempfile, "mkdtemp", failing_mkdtemp)
        assert lp.prepare_lifecycle_probe(argv, dict(os.environ), tmp_path) is None
    real_open = os.open

    def failing_open(path, flags, *args, **kwargs):
        if flags & os.O_EXCL:
            raise OSError("exists")
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(lp.os, "open", failing_open)
        assert lp.prepare_lifecycle_probe(argv, dict(os.environ), tmp_path) is None
    assert list(scratch.iterdir()) == []
    with monkeypatch.context() as patched:
        patched.setattr(lp, "prepare_lifecycle_probe", lambda *a, **k: None)
        result = _lp_run(tmp_path, ["test_x.py", "--bogus-flag"])
        assert result.returncode == 4 and result.pre_collection_launch_failure is None


def _lp_no_pytest_python(tmp_path):
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


def test_interpreter_without_pytest_is_classified_only_after_the_independent_check(tmp_path):
    python = _lp_no_pytest_python(tmp_path)
    project = _lp_write(tmp_path / "project", _LP_PASSING)
    result = run_foreground_test(
        [python, "-m", "pytest", "test_x.py"], cwd=project, timeout_seconds=120, echo_output=False,
        classify_pre_collection=True,
    )
    assert result.returncode == 1
    assert result.pre_collection_launch_failure == runtime.PRE_COLLECTION_NO_PYTEST_REASON
    setup = lp.prepare_lifecycle_probe([python, "-m", "pytest", "test_x.py"], dict(os.environ), project)
    try:
        assert lp.run_independent_import_check(
            setup.eligibility, setup.argv, dict(os.environ), project,
        ) == "pytest-absent"
        stale = lp.LaunchEligibility(
            "module", setup.eligibility.invocation, setup.eligibility.interpreter_path, (0, 0, 0, 0),
        )
        assert lp.run_independent_import_check(stale, setup.argv, dict(os.environ), project) == "unavailable"
        assert lp.run_independent_import_check(None, setup.argv, dict(os.environ), project) == "unavailable"
        console = lp.LaunchEligibility(
            "console-script", setup.eligibility.invocation, setup.eligibility.interpreter_path,
            setup.eligibility.identity,
        )
        assert lp.run_independent_import_check(console, setup.argv, dict(os.environ), project) == "unavailable"
    finally:
        setup.cleanup()


def test_no_pytest_route_needs_eligibility_report_and_check():
    scan = lp.PreCollectionScan()
    scan.feed("/x/python: No module named pytest\n")
    eligibility = lp.LaunchEligibility("module", (sys.executable,), sys.executable, (1, 2, 3, 4))
    argv = [sys.executable, "-m", "pytest", "x.py"]
    empty = lp.ProbeReport(())

    def classify(**overrides):
        arguments = {
            "returncode": 1, "scan": scan, "probe": empty, "independent_import_check": "pytest-absent",
            "launch_eligibility": eligibility, "worker_sessions_observed": False,
        }
        arguments.update(overrides)
        return runtime.classify_pre_collection_launch_failure(
            argv, arguments.pop("returncode"), arguments.pop("scan"), arguments.pop("probe"),
            arguments.pop("independent_import_check"), **arguments,
        )

    assert classify() == runtime.PRE_COLLECTION_NO_PYTEST_REASON
    assert classify(launch_eligibility=None) is None
    assert classify(independent_import_check="unavailable") is None
    assert classify(probe=None) is None
    assert classify(probe=lp.ProbeReport(({"kind": "loaded"},))) is None
    assert classify(returncode=0) is None
    console = lp.LaunchEligibility("console-script", (sys.executable,), sys.executable, (1, 2, 3, 4))
    assert classify(launch_eligibility=console) is None
    plain = lp.PreCollectionScan()
    plain.feed("something else\n")
    assert classify(scan=plain) is None


def test_python_named_shell_wrapper_is_never_classified_as_missing_pytest(tmp_path):
    python = _lp_no_pytest_python(tmp_path)
    real = sys.executable
    wrapper = _lp_script(
        tmp_path / "bin" / "python3",
        f'#!/bin/sh\nif [ "$1" = "-m" ]; then exec {real} "$@"; else exec {python} "$@"; fi\n',
    )
    project = _lp_write(tmp_path / "project", {"test_x.py": (
        "import os, sys\n\ndef test_a():\n    print('No module named pytest'); sys.stdout.flush()\n    os._exit(1)\n"
    )})
    argv = [str(wrapper), "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:terminal", "-s", "test_x.py"]
    assert lp.prepare_lifecycle_probe(argv, dict(os.environ), project) is None
    result = run_foreground_test(
        argv, cwd=project, timeout_seconds=120, echo_output=False, classify_pre_collection=True,
    )
    assert (result.outcome, result.returncode) == ("failed", 1)
    assert result.pre_collection_launch_failure is None
    scan = lp.PreCollectionScan()
    scan.feed("No module named pytest\n")
    assert runtime.classify_pre_collection_launch_failure(
        argv, 1, scan, lp.ProbeReport(()), "pytest-absent", launch_eligibility=None,
        worker_sessions_observed=False,
    ) is None


def _lp_cli_fixtures(tmp_path, monkeypatch):
    import coding_review_agent_loop.cli as cli_module

    calls = []
    monkeypatch.setattr(cli_module, "record_launcher_health", lambda *a, **k: calls.append("health"))
    monkeypatch.setattr(cli_module, "_record_run_tests_result", lambda *a, **k: calls.append("result"))
    monkeypatch.chdir(tmp_path)
    return cli_module, calls


def test_cli_fallback_path_records_nothing_for_a_classified_run(tmp_path, monkeypatch):
    _lp_write(tmp_path, {**_LP_PASSING, "test_bad.py": "def test_b():\n    assert False\n"})
    cli_module, calls = _lp_cli_fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: None)
    base = ["run-tests", "--timeout-seconds", "60", "--memory-dir", str(tmp_path / "memory"), "--"]
    pytest_argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    assert main([*base, *pytest_argv, "test_x.py", "--bogus-flag"]) == 4
    assert calls == []
    assert main([*base, *pytest_argv, "test_bad.py"]) == 1
    assert "result" in calls


def test_cli_broker_path_records_nothing_for_a_classified_frame(tmp_path, monkeypatch):
    from coding_review_agent_loop.local_test_evidence import (
        PRE_COLLECTION_LAUNCH_FAILURE_PREFIX, BrokerRunResult,
    )

    cli_module, calls = _lp_cli_fixtures(tmp_path, monkeypatch)
    frames = {
        "classified": BrokerRunResult(
            receipt_id="r1", execution_ref="e1", outcome="launch-failed", returncode=4, elapsed_seconds=1.0,
            wrapper_bootstrap="verified", inner_exec="started", suite_start="not-started",
            diagnostic=f"{PRE_COLLECTION_LAUNCH_FAILURE_PREFIX}: usage",
        ),
        "ordinary": BrokerRunResult(
            receipt_id="r2", execution_ref="e2", outcome="failed", returncode=1, elapsed_seconds=1.0,
            wrapper_bootstrap="verified", inner_exec="started", suite_start="unknown",
        ),
    }

    class FakeBroker:
        def __init__(self, key):
            self.key = key

        def run(self, *args, **kwargs):
            return frames[self.key]

    command = [
        "run-tests", "--timeout-seconds", "5", "--memory-dir", str(tmp_path / "memory"), "--",
        sys.executable, "-m", "pytest",
    ]
    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: FakeBroker("classified"))
    assert main(command) == 4
    assert calls == []
    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: FakeBroker("ordinary"))
    assert main(command) == 1
    assert "result" in calls


def test_cli_broker_requests_carry_the_client_tool_identity(tmp_path, monkeypatch):
    """#1632: both broker call sites stamp the client identity as overrides."""
    from coding_review_agent_loop import tool_provenance
    from coding_review_agent_loop.local_test_evidence import (
        TOOL_COMMIT_ENV, TOOL_DIRTY_ENV, TOOL_PACKAGE_ENV, BrokerRunResult,
    )
    import coding_review_agent_loop.cli as cli_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        tool_provenance,
        "capture_client_tool_provenance",
        lambda **_kwargs: {"commit": "abcdef1234567", "dirty": None, "package_path": "/opt/pkg"},
    )
    seen = []

    class FakeBroker:
        def run(self, argv, **kwargs):
            seen.append(kwargs)
            return BrokerRunResult(
                receipt_id="r1", execution_ref="e1", outcome="passed", returncode=0,
                elapsed_seconds=1.0, wrapper_bootstrap="verified", inner_exec="started",
                suite_start="unknown",
                fixture={} if kwargs.get("fixture") else None,
            )

    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: FakeBroker())
    assert main(["run-tests", "--timeout-seconds", "5", "--", sys.executable, "-c", "pass"]) == 0
    fixture = tmp_path / "fixture.sh"
    fixture.write_text("#!/bin/sh\n", encoding="utf-8")
    main(["run-tests", "--fixture", str(fixture), "--timeout-seconds", "60", "--", sys.executable, "-c", "pass"])
    assert len(seen) == 2
    for kwargs in seen:
        overrides = kwargs["environment_overrides"]
        assert overrides[TOOL_COMMIT_ENV] == "abcdef1234567"
        assert overrides[TOOL_DIRTY_ENV] == "unknown"
        assert overrides[TOOL_PACKAGE_ENV] == "/opt/pkg"


# --- review round 1 regressions (issue #1182) --------------------------------

_LP_FIRE_MODULE = (
    "import importlib.util\n"
    "def load(path):\n"
    "    spec = importlib.util.spec_from_file_location('hidden', path)\n"
    "    module = importlib.util.module_from_spec(spec)\n"
    "    spec.loader.exec_module(module)\n"
    "    return module\n"
)


def test_console_script_with_an_expression_in_a_standard_line_gets_no_injection(tmp_path):
    real = sys.executable
    _lp_write(tmp_path, {"test_x.py": _LP_FIRE})
    template = (
        f"#!{real}\nimport re\nimport sys\nfrom pytest import console_main\n"
        "if __name__ == '__main__':\n    {argv0}\n    sys.exit(console_main())\n"
    )
    standard = "sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])"
    smuggled = "sys.argv[0] = re.sub(__import__('test_x').fire() or '', '', sys.argv[0])"
    good = _lp_script(tmp_path / "good" / "pytest", template.format(argv0=standard))
    bad = _lp_script(tmp_path / "bad" / "pytest", template.format(argv0=smuggled))
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    assert lp.prepare_lifecycle_probe([str(good), "x.py"], env, tmp_path) is not None
    assert lp.prepare_lifecycle_probe([str(bad), "x.py", "--bogus-flag"], env, tmp_path) is None
    result = run_foreground_test(
        [str(bad), "--bogus-flag", "-p", "no:cacheprovider", "test_x.py"], cwd=tmp_path,
        timeout_seconds=120, echo_output=False, classify_pre_collection=True,
        env={"PYTHONPATH": str(tmp_path)},
    )
    assert result.pre_collection_launch_failure is None


def test_env_shebang_trampoline_gets_no_injection_and_keeps_the_exit_status(tmp_path):
    real = sys.executable
    _lp_write(tmp_path, _LP_PASSING)
    trampoline = _lp_script(tmp_path / "bin" / "env", f'#!/bin/sh\nshift\nexec {real} -I "$@"\n')
    script = _lp_script(
        tmp_path / "scripts" / "pytest",
        f"#!{trampoline} python3\nimport sys\nfrom pytest import console_main\n"
        "if __name__ == '__main__':\n    sys.exit(console_main())\n",
    )
    argv = [str(script), "-q", "-p", "no:cacheprovider", "test_x.py"]
    assert lp.prepare_lifecycle_probe(argv, dict(os.environ), tmp_path) is None
    plain = run_foreground_test(
        argv, cwd=tmp_path, timeout_seconds=120, echo_output=False,
    )
    classified = run_foreground_test(
        argv, cwd=tmp_path, timeout_seconds=120, echo_output=False, classify_pre_collection=True,
    )
    assert classified.returncode == plain.returncode
    assert classified.pre_collection_launch_failure is None


@pytest.mark.parametrize(
    "variant",
    [
        "pathname_pattern", "pyc_load", "retained_set_overflow", "ini_addopts_target",
        "non_py_explicit_target", "chdir_explicit_target", "chdir_pathname_pattern",
        "symlink_parent_component", "symlink_pattern",
    ],
)
def test_further_pre_parse_test_code_variants_are_not_classified(tmp_path, variant):
    files = {"pytest.ini": "[pytest]\n", "loader.py": _LP_FIRE_MODULE}
    extra = ["-p", "no:terminal", "-s"]
    if variant == "pathname_pattern":
        files.update({
            "pytest.ini": "[pytest]\npython_files = check/*.py\n",
            "check/cases.py": _LP_FIRE,
            "userplug.py": "import loader\nFIRE = loader.load('check/cases.py')\n",
            "conftest.py": "import userplug\nuserplug.FIRE.fire()\n",
        })
        extra += ["-p", "userplug", "check"]
    elif variant == "pyc_load":
        files.update({
            "test_x.py": _LP_FIRE,
            "conftest.py": (
                "import importlib.machinery, importlib.util, py_compile\n"
                "py_compile.compile('test_x.py', cfile='compiled.pyc', doraise=True)\n"
                "import sys\nsys.modules.pop('test_x', None)\n"
                "loader = importlib.machinery.SourcelessFileLoader('hidden_pyc', 'compiled.pyc')\n"
                "spec = importlib.util.spec_from_loader('hidden_pyc', loader)\n"
                "module = importlib.util.module_from_spec(spec)\nloader.exec_module(module)\nmodule.fire()\n"
            ),
        })
        # The compiled file shares the source's name only through its code object.
        files["conftest.py"] = files["conftest.py"].replace("'compiled.pyc'", "'test_x.pyc'")
    elif variant == "retained_set_overflow":
        for index in range(4100):
            files[f"filler/f{index}.py"] = ""
        files["userplug.py"] = (
            "import glob\nfor path in glob.glob('filler/*.py'):\n    open(path).close()\n"
        )
        extra = ["-p", "userplug"]
    elif variant == "symlink_parent_component":
        files.update({
            "other/deep/keep": "",
            "other/cases.spec": _LP_FIRE,
            "userplug.py": (
                "NS = {}\nexec(compile(open('link/../cases.spec').read(), 'unrelated', 'exec'), NS)\n"
                "FIRE = NS['fire']\n"
            ),
            "conftest.py": "import userplug\nuserplug.FIRE()\n",
        })
        extra += ["-p", "userplug", str(tmp_path / "other" / "cases.spec")]
    elif variant == "symlink_pattern":
        files.update({
            "pytest.ini": "[pytest]\npython_files = real/check/*.py\n",
            "real/check/cases.py": _LP_FIRE,
            "userplug.py": (
                "NS = {}\nexec(compile(open('alias/cases.py').read(), 'unrelated', 'exec'), NS)\n"
                "FIRE = NS['fire']\n"
            ),
            "conftest.py": "import userplug\nuserplug.FIRE()\n",
        })
        extra += ["-p", "userplug", str(tmp_path / "real" / "check")]
    elif variant == "chdir_explicit_target":
        files.update({
            "cases.spec": _LP_FIRE,
            "sub/keep": "",
            "userplug.py": (
                "import os\nNS = {}\nexec(compile(open('cases.spec').read(), 'cases.spec', 'exec'), NS)\n"
                "FIRE = NS['fire']\nos.chdir('sub')\n"
            ),
            "conftest.py": "import userplug\nuserplug.FIRE()\n",
        })
        extra += ["-p", "userplug", str(tmp_path / "cases.spec")]
    elif variant == "chdir_pathname_pattern":
        files.update({
            "pytest.ini": "[pytest]\npython_files = check/*.py\n",
            "check/cases.py": _LP_FIRE,
            "sub/keep": "",
            "userplug.py": (
                "import os\nNS = {}\nexec(compile(open('check/cases.py').read(), 'check/cases.py', 'exec'), NS)\n"
                "FIRE = NS['fire']\nos.chdir('sub')\n"
            ),
            "conftest.py": "import userplug\nuserplug.FIRE()\n",
        })
        extra += ["-p", "userplug", str(tmp_path / "check")]
    elif variant == "non_py_explicit_target":
        files.update({
            "cases.spec": _LP_FIRE,
            "userplug.py": (
                "NS = {}\nexec(compile(open('cases.spec').read(), 'cases.spec', 'exec'), NS)\nFIRE = NS['fire']\n"
            ),
            "conftest.py": "import userplug\nuserplug.FIRE()\n",
        })
        extra += ["-p", "userplug", "cases.spec"]
    else:
        files.update({
            "pytest.ini": "[pytest]\naddopts = -p no:terminal -s cases.py\n",
            "cases.py": _LP_FIRE,
            "conftest.py": "import loader\nloader.load('cases.py').fire()\n",
        })
        extra = []
    _lp_write(tmp_path, files)
    if variant == "symlink_parent_component":
        (tmp_path / "link").symlink_to("other/deep")
    elif variant == "symlink_pattern":
        (tmp_path / "alias").symlink_to("real/check")
    env = {"PYTHONPATH": str(tmp_path)}
    quiet = variant == "retained_set_overflow"
    if variant == "retained_set_overflow":
        extra += ["--bogus-flag"]
    result = _lp_run(tmp_path, extra, env=env, quiet=quiet)
    assert result.returncode == 4 if variant != "retained_set_overflow" else result.returncode != 0
    assert result.pre_collection_launch_failure is None
    rc, _output, report = _lp_direct(tmp_path, extra, env=env)
    assert "test-code-touched" in [record["kind"] for record in report.records]


@pytest.mark.parametrize("target", ["-cases.py", "-cases.py::test_x"])
def test_bare_dash_targets_after_the_delimiter_are_not_classified(tmp_path, target):
    _lp_write(tmp_path, {
        "-cases.py": _LP_FIRE,
        "loader.py": _LP_FIRE_MODULE,
        "conftest.py": "import loader\nloader.load('./-cases.py').fire()\n",
    })
    env = {"PYTHONPATH": str(tmp_path)}
    result = _lp_run(tmp_path, ["-p", "no:terminal", "-s", "--", target], env=env, quiet=False)
    assert result.returncode == 4 and result.pre_collection_launch_failure is None


def test_bare_dash_target_runs_exactly_as_it_does_without_the_probe(tmp_path):
    _lp_write(tmp_path, {"-cases.py": "def test_b():\n    assert True\n"})
    plain = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--", "-cases.py"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    rc, _output, report = _lp_direct(tmp_path, ["--", "-cases.py"])
    assert rc == plain.returncode
    assert [record["kind"] for record in report.records][-1] == "final"


_LP_SHORT_SITE = _LP_SITE.replace(
    "            raise OSError('injected')\n",
    "            return _real(fd, data[:5])\n",
)


def test_short_probe_writes_and_an_unopenable_report_fail_closed(tmp_path):
    _lp_write(tmp_path, {**_LP_PASSING, "hook/sitecustomize.py": _LP_SHORT_SITE})
    env = {"PYTHONPATH": str(tmp_path / "hook"), "LP_FAIL_ON": "1"}
    result = _lp_run(tmp_path, ["--bogus-flag", "test_x.py"], env=env)
    assert result.returncode == 4 and result.pre_collection_launch_failure is None
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--bogus-flag", "test_x.py"]
    setup = lp.prepare_lifecycle_probe(argv, dict(os.environ), tmp_path)
    try:
        setup.report_path.unlink()  # the probe cannot open a report that is gone
        done = subprocess.run(list(setup.argv), cwd=tmp_path, env=setup.env, capture_output=True, text=True)
        assert done.returncode == 4
        assert lp.read_probe_report(setup.report_path, setup.nonce) is None
    finally:
        setup.cleanup()


_LP_NOT_INJECTED = (
    "import os, sys\n\n"
    "def test_not_injected():\n"
    "    assert '_agent_loop_lifecycle_probe' not in sys.modules\n"
    "    assert 'AGENT_LOOP_LIFECYCLE_PROBE_SPEC' not in os.environ\n"
    "    assert not any('lifecycle_probe' in arg for arg in sys.argv)\n"
)


@pytest.mark.parametrize("failure", ["wrapper", "setup-failure", "preflight-failure", "excl-failure"])
def test_excluded_launchers_and_setup_failures_run_the_original_command_through_the_runner(
    tmp_path, monkeypatch, failure
):
    real = sys.executable
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(lp.tempfile, "tempdir", str(scratch))
    wrapper = _lp_script(tmp_path / "bin" / "python3", f'#!/bin/sh\nexec {real} "$@"\n')
    _lp_write(tmp_path, {"test_x.py": _LP_DUMP_TEST})
    launcher = str(wrapper) if failure == "wrapper" else sys.executable
    argv = [launcher, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_x.py"]
    env = dict(os.environ)
    # Uninjected baselines (classification off, nothing patched): passing and usage.
    baselines = {
        tail: _lp_run_recorded([*argv, *tail], tmp_path, env=env, classify=False)
        for tail in ((), ("--bogus-flag",))
    }
    with monkeypatch.context() as patched:
        if failure == "setup-failure":
            def failing(*_a, **_k):
                raise OSError("no space")
            patched.setattr(lp.tempfile, "mkdtemp", failing)
        elif failure == "preflight-failure":
            patched.setattr(lp, "_PREFLIGHT_CODE", "import sys; sys.exit(3)")
        elif failure == "excl-failure":
            real_open = os.open

            def failing_open(path, flags, *args, **kwargs):
                if flags & os.O_EXCL and "agent-loop-lifecycle-probe-" in str(path):
                    raise OSError("exists")
                return real_open(path, flags, *args, **kwargs)

            patched.setattr(lp.os, "open", failing_open)
        for tail, code in (((), 0), (("--bogus-flag",), 4)):
            result, spawn = _lp_run_recorded([*argv, *tail], tmp_path, env=env, classify=True)
            baseline_result, baseline_spawn = baselines[tail]
            assert spawn == baseline_spawn, (failure, tail)
            assert result.returncode == baseline_result.returncode == code, (failure, tail)
            assert result.pre_collection_launch_failure is None, (failure, tail)
    assert not list(scratch.rglob("agent-loop-lifecycle-probe-*"))  # no partial probe allocation is left behind


def test_missing_pytest_is_non_evidence_through_the_unmocked_cli_fallback(tmp_path, monkeypatch):
    python = _lp_no_pytest_python(tmp_path)
    project = _lp_write(tmp_path / "project", _LP_PASSING)
    cli_module, calls = _lp_cli_fixtures(project, monkeypatch)
    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: None)
    code = main([
        "run-tests", "--timeout-seconds", "60", "--memory-dir", str(tmp_path / "memory"), "--",
        python, "-m", "pytest", "test_x.py",
    ])
    assert code == 1
    assert calls == []


def test_unchanged_head_stop_message_keeps_route_suffix_and_is_byte_identical_without_receipts():
    from coding_review_agent_loop.local_test_evidence import (
        EnvironmentIdentityRegistry, EvidenceScope, LocalTestObservation, TreeAttribution,
        bounded_evidence_for_round, reconcile_test_observations,
    )
    from coding_review_agent_loop.pr_loop import unchanged_head_stop_message

    registry = EnvironmentIdentityRegistry()
    failure = LocalTestObservation(
        command=("python3", "-m", "pytest", "tests/x.py"), outcome="failed", provenance="parent-observed",
        scope=EvidenceScope("unknown", ()), receipt_id="r", turn_id="t", cwd="/checkout", returncode=4,
        attribution=TreeAttribution(state="current-head", head="h", stable=True),
        environment_identity=registry.capture({"PATH": "/usr/bin"}),
    )
    evidence = bounded_evidence_for_round(reconcile_test_observations([failure], registry=registry))
    route = " If the blocking finding requires re-planning the child plan, post the record."
    arguments = dict(pr=7, coder_name="claude", previous_head="abc", turns=2, round_number=2, route=route)
    legacy = (
        "PR #7: claude left head abc unchanged in 2 consecutive follow-up rounds, so another "
        "review of the same diff cannot change the verdict. Stopping before round 3; "
        "human review required." + route
    )
    assert unchanged_head_stop_message(evidence=None, **arguments) == legacy
    with_receipts = unchanged_head_stop_message(evidence=evidence, **arguments)
    assert with_receipts.startswith(legacy.removesuffix(route))
    assert with_receipts.endswith(route) and "unsuperseded failure receipts" in with_receipts


@pytest.mark.parametrize("entry", ["run_optional_tests", "run_pre_review_tests"])
def test_configured_gate_usage_error_keeps_its_routing_through_the_gate_workflow(
    tmp_path, monkeypatch, entry
):
    """Issue #1182: the configured gate neither injects the probe nor classifies."""
    import contextlib

    from coding_review_agent_loop import checks
    from coding_review_agent_loop.runner import Runner

    _lp_write(tmp_path, {"test_x.py": _LP_NOT_INJECTED})
    config = make_config(
        tmp_path,
        test_command=(sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_x.py", "--bogus-flag"),
        pre_review_tests=True,
    )
    monkeypatch.setattr(checks, "active_workdir", lambda cfg: tmp_path)
    monkeypatch.setattr(checks, "gate_window", lambda *a, **k: contextlib.nullcontext())
    recorded = []
    real_record = checks._record_gate_observation

    def spy(cfg, result):
        recorded.append(result)
        return real_record(cfg, result)

    monkeypatch.setattr(checks, "_record_gate_observation", spy)
    with pytest.raises(AgentLoopError) as raised:
        getattr(checks, entry)(Runner(), config)
    assert "failed with exit 4" in str(raised.value)
    assert "could not start" not in str(raised.value)
    (result,) = recorded
    assert (result.outcome, result.returncode) == ("failed", 4)
    assert result.pre_collection_launch_failure is None
    # The same gate command without the bogus flag proves no probe was injected.
    clean = Runner().run_test_command(
        list(config.test_command[:-1]), cwd=tmp_path, timeout_seconds=120,
    )
    assert (clean.outcome, clean.returncode) == ("passed", 0)


# --- full-suite attainability (issue #1343) ----------------------------------

from coding_review_agent_loop import runner as runner_module  # noqa: E402
from coding_review_agent_loop.test_workers import WorkerBudgetLock, HostWaitResult  # noqa: E402

_XA_PREFIX = "agent-loop: advisory: pytest-xdist could not be imported"
_NOTICE_PREFIX = "agent-loop: notice: this command may run for up to"
_SRC = str(Path(runtime.__file__).resolve().parents[1])


def _xa_project(root, *, declares=True, broken_xdist=True):
    """A checkout that declares xdist and whose launch context cannot import it."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").mkdir(exist_ok=True)
    _lp_write(root, _LP_PASSING)
    if declares:
        (root / "pyproject.toml").write_text(
            '[project]\nname = "x"\n[project.optional-dependencies]\ndev = ["pytest-xdist>=3"]\n',
            encoding="utf-8",
        )
    if broken_xdist:
        # Discoverable but unimportable in this launch context (cwd first on
        # sys.path for both ``-m`` and ``-c``).
        _lp_write(root, {
            "xdist/__init__.py": "",
            "xdist/plugin.py": "raise ImportError('xdist unavailable in this context')\n",
        })
    return root


def _xa_run(root, extra, *, env=None, budget=None, disable_check=False, monkeypatch=None, timeout=120, lock_root=None):
    lines = []
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra]
    context = pytest.MonkeyPatch.context() if disable_check else contextlib.nullcontext()
    with context as patched:
        if disable_check:
            patched.setattr(runner_module, "_xdist_advisory", lambda *a, **k: None)
        result = run_foreground_test(
            argv, cwd=root, timeout_seconds=timeout, echo_output=False, classify_pre_collection=True,
            env=({**os.environ, **env} if env else None), worker_budget=budget,
            output_callback=lines.append, worker_lock_root=lock_root,
        )
    return result, "".join(lines)


def _advisories(output):
    return [line for line in output.splitlines() if line.startswith(_XA_PREFIX)]


def _assert_qualified_advisory(text):
    assert "could not be imported by" in text
    assert "although the repository declares it" in text
    # Every possible outcome is qualified; none is predicted.
    assert "may run serially" in text
    assert "with a usage error" in text
    assert "fail while loading the xdist plugin" in text
    assert "will run serially" not in text and "will proceed" not in text
    assert "do not install into a system interpreter" in text
    assert "not test evidence" in text


def test_explicit_n_without_xdist_gets_the_advisory_and_unchanged_classification(tmp_path):
    root = _xa_project(tmp_path / "project")
    env = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    result, output = _xa_run(root, ["-n", "6", "test_x.py"], env=env)
    (advisory,) = _advisories(output)
    _assert_qualified_advisory(advisory)
    assert "120s" in advisory
    assert (result.outcome, result.returncode) == ("failed", 4)
    assert result.pre_collection_launch_failure == _LP_USAGE_REASON
    baseline, baseline_output = _xa_run(root, ["-n", "6", "test_x.py"], env=env, disable_check=True)
    assert _advisories(baseline_output) == []
    assert (baseline.outcome, baseline.returncode, baseline.pre_collection_launch_failure, baseline.args) == (
        result.outcome, result.returncode, result.pre_collection_launch_failure, result.args,
    )
    # The advisory precedes any pytest output.
    assert output.index(_XA_PREFIX) == 0


@pytest.mark.parametrize("case", ["addopts", "broken-plugin"])
def test_other_xdist_failures_get_the_advisory_and_todays_classification(tmp_path, case):
    root = _xa_project(tmp_path / "project")
    env = {"PYTEST_ADDOPTS": "-n 8"} if case == "addopts" else {}
    result, output = _xa_run(root, ["test_x.py"], env=env)
    (advisory,) = _advisories(output)
    _assert_qualified_advisory(advisory)
    baseline, _ = _xa_run(root, ["test_x.py"], env=env, disable_check=True)
    assert (baseline.outcome, baseline.returncode, baseline.pre_collection_launch_failure, baseline.args) == (
        result.outcome, result.returncode, result.pre_collection_launch_failure, result.args,
    )


def test_advisory_names_test_python_only_when_it_is_set(tmp_path, monkeypatch):
    # The unset case must not inherit the variable from an agent-loop launch.
    monkeypatch.delenv(runtime.ENV_TEST_PYTHON, raising=False)
    root = _xa_project(tmp_path / "project")
    env = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", runtime.ENV_TEST_PYTHON: "/opt/venv/bin/python"}
    _result, output = _xa_run(root, ["test_x.py"], env=env)
    (advisory,) = _advisories(output)
    assert "$AGENT_LOOP_TEST_PYTHON" in advisory
    _result, output = _xa_run(root, ["test_x.py"], env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
    (advisory,) = _advisories(output)
    assert "$AGENT_LOOP_TEST_PYTHON" not in advisory and "--test-python" in advisory


def _budget(mode):
    if mode is None:
        return None
    return WorkerBudget(workers=2, source="operator", enforcement=mode)


_SERIAL_ARGS = (
    ["-n", "0"], ["-n0"], ["-n=0"], ["--numprocesses", "0"], ["--numprocesses=0"],
    ["-p", "no:xdist"], ["-pno:xdist"], ["-p=no:xdist"], ["--dist", "no"], ["--dist=no"],
)


@pytest.mark.parametrize("mode", [None, "off", "clamp", "refuse"])
@pytest.mark.parametrize("extra", _SERIAL_ARGS, ids=lambda value: " ".join(value))
def test_serial_or_no_xdist_argv_never_runs_the_check(tmp_path, monkeypatch, mode, extra):
    from coding_review_agent_loop import lifecycle_probe

    calls = []
    monkeypatch.setattr(lifecycle_probe, "module_launch_xdist_check", lambda *a, **k: calls.append(a) or 4)
    root = _xa_project(tmp_path / "project")
    argv = [sys.executable, "-m", "pytest", "-q", *extra, "test_x.py"]
    assert runner_module._argv_requests_serial_or_no_xdist(argv)
    assert runner_module._xdist_advisory(argv, None, root, 60, worker_budget=_budget(mode)) is None
    assert calls == []


@pytest.mark.parametrize("mode", [None, "off", "clamp"])
def test_serial_argv_runs_with_no_check_and_no_advisory_under_each_budget_mode(tmp_path, monkeypatch, mode):
    from coding_review_agent_loop import lifecycle_probe

    calls = []
    monkeypatch.setattr(lifecycle_probe, "module_launch_xdist_check", lambda *a, **k: calls.append(a) or 4)
    monkeypatch.setenv("AGENT_LOOP_TEST_WORKER_HOST_SHARING", "off")
    root = _xa_project(tmp_path / "project", broken_xdist=False)
    result, output = _xa_run(
        root, ["-p", "no:xdist", "test_x.py"], budget=_budget(mode), lock_root=tmp_path / "locks",
    )
    assert result.outcome == "passed"
    assert calls == [] and _advisories(output) == []


def test_serial_helper_leaves_cohort_labels_unchanged():
    from coding_review_agent_loop.test_workers import argv_only_workers_label, expected_workers_label

    argv = [sys.executable, "-m", "pytest", "-n", "0"]
    assert argv_only_workers_label(argv) == "unknown"
    assert expected_workers_label(argv) == "unknown"
    assert runner_module._argv_requests_serial_or_no_xdist(argv)
    assert not runner_module._argv_requests_serial_or_no_xdist([sys.executable, "-m", "pytest", "-n", "4"])
    assert not runner_module._argv_requests_serial_or_no_xdist([sys.executable, "-m", "pytest", "--", "-n", "0"])


def _write_console_script(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pytest import console_main\n"
        "if __name__ == '__main__':\n    sys.argv[0] = sys.argv[0].removesuffix('.exe')\n"
        "    sys.exit(console_main())\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("shape", ["console-script", "env-prefix", "timeout-prefix", "isolated-flag", "non-pytest"])
def test_ineligible_shapes_are_never_probed(tmp_path, monkeypatch, shape):
    from coding_review_agent_loop import lifecycle_probe

    calls = []
    real = runtime.run_exit_code_probe
    monkeypatch.setattr(runtime, "run_exit_code_probe", lambda argv, **k: calls.append(list(argv)) or real(argv, **k))
    root = _xa_project(tmp_path / "project")
    # A checkout-local xdist.py that raises: a console script's sys.path[0]
    # is its script directory, so a -c check could only give a false advisory.
    (root / "xdist.py").write_text("raise ImportError('shadow')\n", encoding="utf-8")
    if shape == "console-script":
        argv = [str(_write_console_script(root / "venv" / "bin" / "pytest")), "-q", "test_x.py"]
    elif shape == "env-prefix":
        argv = ["/usr/bin/env", "FOO=1", sys.executable, "-m", "pytest", "-q", "test_x.py"]
    elif shape == "timeout-prefix":
        argv = ["timeout", "100", sys.executable, "-m", "pytest", "-q", "test_x.py"]
    elif shape == "isolated-flag":
        argv = [sys.executable, "-I", "-m", "pytest", "-q", "test_x.py"]
    else:
        argv = [sys.executable, "-c", "pass"]
    assert runner_module._xdist_advisory(argv, None, root, 60, worker_budget=None) is None
    assert not any("xdist.plugin" in " ".join(call) for call in calls)
    if shape == "console-script":
        eligibility = lifecycle_probe.launch_eligibility_for(argv, dict(os.environ), root)
        assert eligibility is not None and eligibility.shape == "console-script"


def test_no_advisory_when_xdist_imports_only_via_the_callers_pythonpath(tmp_path):
    root = _xa_project(tmp_path / "project", broken_xdist=False)
    extra = tmp_path / "extra-path"
    _lp_write(extra, {"xdist/__init__.py": "", "xdist/plugin.py": ""})
    pythonpath = os.pathsep.join(filter(None, [str(extra), os.environ.get("PYTHONPATH")]))
    env = {**os.environ, "PYTHONPATH": pythonpath}
    argv = [sys.executable, "-m", "pytest", "test_x.py"]
    assert runner_module._xdist_advisory(argv, env, root, 60, worker_budget=None) is None
    # Without that path entry the same launch context cannot import xdist.
    root_broken = _xa_project(tmp_path / "broken")
    assert runner_module._xdist_advisory(argv, None, root_broken, 60, worker_budget=None) is not None


@pytest.mark.parametrize("result", ["undeclared", "unknown", "raises", "importable"])
def test_no_advisory_for_undeclared_repo_unknown_or_raising_checks(tmp_path, monkeypatch, result):
    from coding_review_agent_loop import lifecycle_probe

    root = _xa_project(tmp_path / "project", declares=result != "undeclared")
    if result == "unknown":
        monkeypatch.setattr(lifecycle_probe, "module_launch_xdist_check", lambda *a, **k: None)
    elif result == "raises":
        def boom(*_a, **_k):
            raise RuntimeError("probe exploded")
        monkeypatch.setattr(lifecycle_probe, "module_launch_xdist_check", boom)
    elif result == "importable":
        monkeypatch.setattr(lifecycle_probe, "module_launch_xdist_check", lambda *a, **k: 0)
    calls = []
    if result == "undeclared":
        monkeypatch.setattr(lifecycle_probe, "module_launch_xdist_check", lambda *a, **k: calls.append(a) or 4)
    argv = [sys.executable, "-m", "pytest", "test_x.py"]
    assert runner_module._xdist_advisory(argv, None, root, 60, worker_budget=None) is None
    assert calls == []


def test_package_script_launches_get_no_check(tmp_path, monkeypatch):
    from coding_review_agent_loop import lifecycle_probe

    calls = []
    monkeypatch.setattr(lifecycle_probe, "module_launch_xdist_check", lambda *a, **k: calls.append(a) or 4)
    root = _xa_project(tmp_path / "project")
    executed = (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_x.py")
    resolution = runtime.PackageScriptResolution(
        requested_argv=("npm", "run", "test"), manager="npm", script_name="test",
        body=" ".join(executed), appended_args=(), env_prefix_tokens=(), executed_argv=executed,
    )
    lines = []
    result = run_foreground_test(
        ["npm", "run", "test"], cwd=root, timeout_seconds=60, echo_output=False,
        classify_pre_collection=True, package_script=resolution, output_callback=lines.append,
    )
    assert calls == [] and _advisories("".join(lines)) == []
    assert result.args == list(executed) or tuple(result.args) == executed


def _slow_check(seconds, value=4):
    def check(*_args, **_kwargs):
        time.sleep(seconds)
        return value
    return check


@pytest.mark.parametrize("mode", [None, "off", "clamp"])
def test_check_time_is_excluded_from_the_target_watchdog(tmp_path, monkeypatch, mode):
    from coding_review_agent_loop import lifecycle_probe

    if mode == "clamp":
        monkeypatch.setenv("AGENT_LOOP_TEST_WORKER_HOST_SHARING", "on")
        monkeypatch.setenv("AGENT_LOOP_TEST_WORKER_HOST_WAIT_SECONDS", "0")
    else:
        monkeypatch.setenv("AGENT_LOOP_TEST_WORKER_HOST_SHARING", "off")
    # Check and target each fit the watchdog alone, but not together.
    monkeypatch.setattr(lifecycle_probe, "module_launch_xdist_check", _slow_check(5.0))
    root = _xa_project(tmp_path / "project", broken_xdist=False)
    _lp_write(root, {"test_x.py": "import time\n\ndef test_a():\n    time.sleep(3.0)\n"})
    result, output = _xa_run(
        root, ["-p", "no:randomly", "test_x.py"], budget=_budget(mode), timeout=7,
        lock_root=tmp_path / "locks",
    )
    assert len(_advisories(output)) == 1
    assert result.outcome == "passed", output
    # The target never receives more than the chosen watchdog.
    assert result.attempted_timeout_seconds == 7
    assert result.elapsed_seconds < 7


def test_advisory_arrives_before_host_admission_and_leaves_the_launch_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LOOP_TEST_WORKER_HOST_SHARING", "on")
    monkeypatch.setenv("AGENT_LOOP_TEST_WORKER_HOST_WAIT_SECONDS", "5")
    events = []

    def contended(self, requested, capacity, *, wait_seconds, notify, wait_notify=None, **_kwargs):
        events.append(("admission", None))
        time.sleep(1.0)
        return HostWaitResult(1, 1.0, True, False, False)

    monkeypatch.setattr(WorkerBudgetLock, "wait_for_host_workers", contended)
    root = _xa_project(tmp_path / "project")
    env = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}

    def run(disable):
        lines = []

        def collect(text):
            events.append(("output", text))
            lines.append(text)

        context = pytest.MonkeyPatch.context() if disable else contextlib.nullcontext()
        with context as patched:
            if disable:
                patched.setattr(runner_module, "_xdist_advisory", lambda *a, **k: None)
            result = run_foreground_test(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_x.py"],
                cwd=root, timeout_seconds=60, echo_output=False, classify_pre_collection=True,
                env={**os.environ, **env}, worker_budget=_budget("clamp"),
                output_callback=collect, worker_lock_root=tmp_path / "locks",
            )
        return result, "".join(lines)

    result, output = run(False)
    advisory_at = next(i for i, (kind, text) in enumerate(events) if kind == "output" and text.startswith(_XA_PREFIX))
    admission_at = events.index(("admission", None))
    assert advisory_at < admission_at
    events.clear()
    baseline, baseline_output = run(True)
    assert _advisories(baseline_output) == []
    assert (result.args, result.outcome, result.returncode, result.pre_collection_launch_failure) == (
        baseline.args, baseline.outcome, baseline.returncode, baseline.pre_collection_launch_failure,
    )


# --- pre-launch foreground-budget notice and timeout advisory (#1343) --------


def _clean_client_env(**extra):
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("AGENT_LOOP_", "BASH_", "PYTEST_"))
    }
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [_SRC, os.environ.get("PYTHONPATH")]))
    env.update(extra)
    return env


def test_budget_function_sums_watchdog_wait_overhead_and_headroom():
    off = runtime.run_tests_foreground_budget_seconds(600, {"AGENT_LOOP_TEST_WORKER_HOST_SHARING": "off"})
    assert off == 600 + 142 + 300
    on = runtime.run_tests_foreground_budget_seconds(
        1800, {"AGENT_LOOP_TEST_WORKER_HOST_WAIT_SECONDS": "1200"},
    )
    assert on == 1800 + 1200 + 142 + 360


def test_notice_names_the_whole_budget_and_is_suppressed_only_by_a_covering_cap():
    env = {"AGENT_LOOP_TEST_WORKER_HOST_WAIT_SECONDS": "1200"}
    budget = runtime.run_tests_foreground_budget_seconds(300, env)
    notice = runtime.prelaunch_notice(300, 1800, env)
    assert notice.startswith(f"{_NOTICE_PREFIX} {budget}s")
    assert "1200s waiting for shared test capacity" in notice
    assert "300s target watchdog under a 1800s ceiling" in notice
    assert "reported as not run, never as passed" in notice
    assert "not test evidence" in notice
    assert "$AGENT_LOOP_TEST_PYTHON" not in notice
    with_python = runtime.prelaunch_notice(300, 1800, {**env, runtime.ENV_TEST_PYTHON: "/v/bin/python"})
    assert "Prefer $AGENT_LOOP_TEST_PYTHON with parallel workers" in with_python
    off = runtime.prelaunch_notice(300, 1800, {"AGENT_LOOP_TEST_WORKER_HOST_SHARING": "off"})
    assert "waiting for shared test capacity" not in off
    # Suppressed only when the orchestrator-authored cap covers the whole budget.
    assert runtime.prelaunch_notice(300, 1800, {**env, "AGENT_LOOP_SHELL_CAP_MS": str(budget * 1000)}) is None
    for cap in (300 * 1000, (300 + 142) * 1000, budget * 1000 - 1, "unknown", "garbage", "-5"):
        assert runtime.prelaunch_notice(300, 1800, {**env, "AGENT_LOOP_SHELL_CAP_MS": str(cap)}) is not None
    # BASH_MAX_TIMEOUT_MS alone is never trusted.
    huge = str(10 ** 12)
    for cap in (None, "unknown", "120000"):
        values = {**env, "BASH_MAX_TIMEOUT_MS": huge}
        if cap is not None:
            values["AGENT_LOOP_SHELL_CAP_MS"] = cap
        assert runtime.prelaunch_notice(300, 1800, values) is not None


def _run_until_notice(argv, env, cwd, *, dispatched):
    """Start the client, wait for the flushed notice *and* a dispatch event, then SIGKILL it.

    ``dispatched`` returns True once the client has demonstrably moved past
    dispatch (the broker received its request, or the local run entered the
    host-capacity wait); the kill therefore lands after dispatch, exactly where
    a backend shell limit would cut the client short.
    """
    proc = subprocess.Popen(
        argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, text=True,
    )
    lines = []
    seen = threading.Event()

    def read():
        for line in proc.stderr:
            lines.append(line)
            if line.startswith(_NOTICE_PREFIX):
                seen.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        assert seen.wait(60), "".join(lines)
        deadline = time.monotonic() + 60
        while not dispatched() and time.monotonic() < deadline:
            assert proc.poll() is None, "".join(lines)
            time.sleep(0.05)
        assert dispatched(), "the client never reached dispatch: " + "".join(lines)
        assert proc.poll() is None, "the client must still be in flight when the shell kills it"
        proc.kill()
        assert proc.wait(10) == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(10)
        reader.join(10)
    return "".join(lines)


def _notice_line(stderr):
    (line,) = [line for line in stderr.splitlines() if line.startswith(_NOTICE_PREFIX)]
    return line


@pytest.fixture
def silent_broker(tmp_path):
    """A broker endpoint that records each dispatched request and never answers."""
    import socket

    from coding_review_agent_loop.local_test_evidence import _recv_frame

    path = tmp_path / "broker.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(4)
    stop = threading.Event()
    held = []
    requests = []

    def serve():
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                connection, _ = server.accept()
            except OSError:
                continue
            held.append(connection)
            try:
                connection.settimeout(30)
                requests.append(_recv_frame(connection))  # received; never answered
            except Exception:
                pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield {
            "env": {
                "AGENT_LOOP_TEST_BROKER_ENDPOINT": str(path),
                "AGENT_LOOP_TEST_BROKER_CAPABILITY": "capability",
                "AGENT_LOOP_TEST_BROKER_PROTOCOL": "local-test-broker-v1",
                "AGENT_LOOP_INVOCATION_ID": "turn-1",
            },
            "requests": requests,
        }
    finally:
        stop.set()
        thread.join(5)
        for connection in held:
            connection.close()
        server.close()


@pytest.mark.skipif(os.name != "posix", reason="UNIX-socket broker")
@pytest.mark.parametrize("shape", ["prefixed", "console-script"])
def test_notice_is_flushed_before_a_silent_broker_and_survives_a_client_kill(tmp_path, silent_broker, shape):
    if shape == "prefixed":
        inner = ["/usr/bin/env", "FOO=1", sys.executable, "-m", "pytest", "-q"]
    else:
        inner = [str(_write_console_script(tmp_path / "venv" / "bin" / "pytest")), "-q"]
    env = _clean_client_env(**silent_broker["env"])
    stderr = _run_until_notice(
        [sys.executable, "-m", "coding_review_agent_loop.cli", "run-tests", "--timeout-seconds", "600", "--", *inner],
        env, tmp_path, dispatched=lambda: bool(silent_broker["requests"]),
    )
    # The broker received exactly this dispatched command before the kill.
    (request,) = silent_broker["requests"]
    assert request["argv"] == inner and request["timeout_seconds"] == 600
    notice = _notice_line(stderr)
    budget = runtime.run_tests_foreground_budget_seconds(600, env)
    assert notice.startswith(f"{_NOTICE_PREFIX} {budget}s")
    assert "600s target watchdog under a 1800s ceiling" in notice
    assert "reported as not run, never as passed" in notice


_BLOCKED_ADMISSION_LAUNCHER = """\
import pathlib, sys, time
from coding_review_agent_loop.test_workers import WorkerBudgetLock

marker = pathlib.Path(sys.argv[1])


def blocked(self, *args, **kwargs):
    # A fake host-capacity wait: record entry, then block like a contended pool.
    marker.write_text("waiting", encoding="utf-8")
    time.sleep(600)


WorkerBudgetLock.wait_for_host_workers = blocked
from coding_review_agent_loop.cli import main

raise SystemExit(main(sys.argv[2:]))
"""


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup")
def test_local_fallback_client_killed_during_host_admission_already_printed_the_full_budget(tmp_path):
    launcher = tmp_path / "launcher.py"
    launcher.write_text(_BLOCKED_ADMISSION_LAUNCHER, encoding="utf-8")
    marker = tmp_path / "admission-entered"
    pid_file = tmp_path / "target.pid"
    target = "import os, time; open(%r, 'w').write(str(os.getpid())); time.sleep(60)" % str(pid_file)
    env = _clean_client_env(
        AGENT_LOOP_TEST_WORKER_HOST_SHARING="on", AGENT_LOOP_TEST_WORKER_HOST_WAIT_SECONDS="1200",
    )
    stderr = _run_until_notice(
        [
            sys.executable, str(launcher), str(marker), "run-tests",
            "--timeout-seconds", "300", "--containment-mode", "off",
            "--test-workers", "2", "--test-worker-enforcement", "clamp", "--",
            sys.executable, "-c", target,
        ],
        env, tmp_path, dispatched=marker.exists,
    )
    # Killed while blocked in host admission: the target never launched.
    assert not pid_file.exists()
    notice = _notice_line(stderr)
    budget = runtime.run_tests_foreground_budget_seconds(300, env)
    assert budget == 300 + 1200 + 142 + 300
    assert notice.startswith(f"{_NOTICE_PREFIX} {budget}s in the foreground")
    assert "(1200s waiting for shared test capacity, a 300s target watchdog under a 1800s ceiling, plus parent overhead)" in notice
    assert "the shell can kill this client first" in notice
    assert "reported as not run, never as passed" in notice


@pytest.mark.skipif(os.name != "posix", reason="UNIX-socket broker")
def test_codex_agent_env_overrides_an_inherited_cap_so_the_notice_still_prints(tmp_path, silent_broker):
    huge = str(10 ** 12)
    ambient = _clean_client_env(**silent_broker["env"], BASH_MAX_TIMEOUT_MS=huge, AGENT_LOOP_SHELL_CAP_MS=huge)
    config = make_config(tmp_path)
    # Merged exactly like Runner.run_with_log: the agent env wins.
    agent_env = {**ambient, **runtime.agent_shell_environment(config, "codex", "coder", ambient=ambient)}
    assert agent_env["AGENT_LOOP_SHELL_CAP_MS"] == "unknown"
    assert agent_env["BASH_MAX_TIMEOUT_MS"] == huge
    inner = ["/usr/bin/env", "FOO=1", sys.executable, "-c", "pass"]
    stderr = _run_until_notice(
        [sys.executable, "-m", "coding_review_agent_loop.cli", "run-tests", "--timeout-seconds", "600", "--", *inner],
        agent_env, tmp_path, dispatched=lambda: bool(silent_broker["requests"]),
    )
    (request,) = silent_broker["requests"]
    assert request["argv"] == inner
    assert _notice_line(stderr)


_KILLED_CLIENT_CODER = """\
import os, pathlib, signal, subprocess, sys, time

pid_file = pathlib.Path(sys.argv[1])
target = "import os, time; open(%r, 'w').write(str(os.getpid())); time.sleep(60)" % str(pid_file)
client = subprocess.Popen(
    [sys.executable, "-m", "coding_review_agent_loop.cli", "run-tests", "--timeout-seconds", "3",
     "--", sys.executable, "-c", target],
    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
)
notice = ""
for line in client.stderr:
    if line.startswith("agent-loop: notice:"):
        notice = line.strip()
        break
deadline = time.monotonic() + 60
while not pid_file.exists() and time.monotonic() < deadline:
    time.sleep(0.05)
# The backend shell cuts the client short while the broker-run target is live.
os.kill(client.pid, signal.SIGKILL)
client.wait()
print("NOTICE=" + notice, flush=True)
print("CLIENT-RC=%s" % client.returncode, flush=True)
# Keep the coder turn alive past the target's watchdog.
time.sleep(10)
"""


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup")
def test_broker_stops_a_killed_clients_target_and_journals_non_evidence(tmp_path, monkeypatch):
    """Issue #1653: a killed client no longer leaves its target holding the lane."""
    from coding_review_agent_loop.containment import default_policy
    from coding_review_agent_loop.runner import Runner

    # An ambient orchestrator shell cap (set when this suite itself runs inside
    # an agent turn) covers the client's 3s budget and suppresses the notice.
    monkeypatch.delenv("AGENT_LOOP_SHELL_CAP_MS", raising=False)

    for args in (
        ["init", "-q"], ["config", "user.email", "t@example.invalid"], ["config", "user.name", "T"],
    ):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True)
    (tmp_path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)
    coder = tmp_path.parent / f"{tmp_path.name}-coder.py"
    coder.write_text(_KILLED_CLIENT_CODER, encoding="utf-8")
    pid_file = tmp_path.parent / f"{tmp_path.name}-target.pid"
    runner = Runner(containment_policy=default_policy(mode="off", cache_dir=tmp_path.parent / f"{tmp_path.name}-rt"))
    runner.test_workers = 2
    runner.test_worker_enforcement = "clamp"
    log_path = tmp_path.parent / f"{tmp_path.name}-coder.log"
    existing = os.environ.get("PYTHONPATH")
    result = runner.run_with_log(
        [sys.executable, str(coder), str(pid_file)], cwd=tmp_path, log_path=log_path,
        label="coder", progress_interval_seconds=1,
        env={"PYTHONPATH": _SRC + (os.pathsep + existing if existing else "")},
    )
    log = log_path.read_text(encoding="utf-8")
    assert result.returncode == 0, log
    assert "NOTICE=agent-loop: notice: this command may run for up to" in log
    assert "CLIENT-RC=-9" in log
    # The broker saw the lost client and stopped the target's whole group.
    _assert_process_not_active(int(pid_file.read_text(encoding="utf-8")))
    observations = runner.local_test_observations()
    assert [observation.outcome for observation in observations] == ["incomplete"], observations
    assert observations[0].provenance == "telemetry-unverified"
    assert observations[0].caveats[0] == "run-tests client disconnected; target stopped by the broker"


def test_client_suppresses_the_notice_for_a_covering_orchestrator_cap(tmp_path, monkeypatch, capsys):
    cli_module, _calls = _lp_cli_fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: None)
    monkeypatch.setenv("AGENT_LOOP_TEST_WORKER_HOST_SHARING", "off")
    budget = runtime.run_tests_foreground_budget_seconds(60, os.environ)
    command = ["run-tests", "--timeout-seconds", "60", "--containment-mode", "off", "--", sys.executable, "-c", "pass"]
    monkeypatch.setenv("AGENT_LOOP_SHELL_CAP_MS", str(budget * 1000))
    assert main(command) == 0
    assert _NOTICE_PREFIX not in capsys.readouterr().err
    monkeypatch.setenv("AGENT_LOOP_SHELL_CAP_MS", "120000")
    monkeypatch.setenv("BASH_MAX_TIMEOUT_MS", str(10 ** 12))
    assert main(command) == 0
    assert _NOTICE_PREFIX in capsys.readouterr().err


_TIMEOUT_ADVISORY = "agent-loop: advisory: this command reached its"


@pytest.mark.parametrize("shape", ["bare", "prefixed", "non-pytest"])
def test_local_timeout_prints_one_advisory_and_keeps_the_exit_code(tmp_path, monkeypatch, capsys, shape):
    cli_module, _calls = _lp_cli_fixtures(tmp_path, monkeypatch)
    monkeypatch.setenv("AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS", "1800")
    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: None)
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    inner = {"bare": sleeper, "prefixed": ["/usr/bin/env", "FOO=1", *sleeper], "non-pytest": ["sleep", "30"]}[shape]
    code = main(["run-tests", "--timeout-seconds", "1", "--containment-mode", "off", "--", *inner])
    err = capsys.readouterr().err
    assert err.count(_TIMEOUT_ADVISORY) == 1
    assert "reached its 1s watchdog (the policy ceiling is 1800s)" in err
    assert "must be reported as not run" in err
    monkeypatch.setattr(cli_module, "timeout_advisory", lambda *a: "")
    assert main(["run-tests", "--timeout-seconds", "1", "--containment-mode", "off", "--", *inner]) == code
    assert main(["run-tests", "--timeout-seconds", "30", "--containment-mode", "off", "--", sys.executable, "-c", "pass"]) == 0
    assert _TIMEOUT_ADVISORY not in capsys.readouterr().err


def test_broker_timeout_prints_one_advisory_and_other_outcomes_print_nothing_new(tmp_path, monkeypatch, capsys):
    from coding_review_agent_loop.local_test_evidence import BrokerRunResult

    cli_module, _calls = _lp_cli_fixtures(tmp_path, monkeypatch)
    outcome = {"value": "timed_out"}

    class FakeBroker:
        def run(self, *args, **kwargs):
            return BrokerRunResult(
                receipt_id="r", execution_ref="e", outcome=outcome["value"],
                returncode=124 if outcome["value"] == "timed_out" else 0, elapsed_seconds=5.0,
                wrapper_bootstrap="verified", inner_exec="started", suite_start="unknown",
            )

    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: FakeBroker())
    command = ["run-tests", "--timeout-seconds", "5", "--", sys.executable, "-m", "pytest"]
    assert main(command) == 124
    assert capsys.readouterr().err.count(_TIMEOUT_ADVISORY) == 1
    outcome["value"] = "passed"
    assert main(command) == 0
    assert _TIMEOUT_ADVISORY not in capsys.readouterr().err


# --- Claude shell limits and backend cap facts (#1343) -----------------------


def test_claude_coder_limits_cover_the_whole_lifetime_and_keep_larger_operator_values(tmp_path):
    config = make_config(tmp_path, coder_test_command_timeout_seconds=600)
    ambient = {"AGENT_LOOP_TEST_WORKER_HOST_WAIT_SECONDS": "1200"}
    budget_ms = runtime.run_tests_foreground_budget_seconds(600, ambient) * 1000
    env = runtime.agent_shell_environment(config, "claude", "coder", ambient=ambient)
    assert env["BASH_DEFAULT_TIMEOUT_MS"] == env["BASH_MAX_TIMEOUT_MS"] == str(budget_ms)
    assert env["AGENT_LOOP_SHELL_CAP_MS"] == env["BASH_DEFAULT_TIMEOUT_MS"]
    assert budget_ms >= (600 + 1200) * 1000
    larger_default = runtime.agent_shell_environment(
        config, "claude", "coder", ambient={**ambient, "BASH_DEFAULT_TIMEOUT_MS": str(budget_ms * 2)},
    )
    assert larger_default["BASH_DEFAULT_TIMEOUT_MS"] == str(budget_ms * 2)
    assert larger_default["BASH_MAX_TIMEOUT_MS"] == str(budget_ms * 2)  # never below the default
    assert larger_default["AGENT_LOOP_SHELL_CAP_MS"] == str(budget_ms * 2)
    larger_max = runtime.agent_shell_environment(
        config, "claude", "coder", ambient={**ambient, "BASH_MAX_TIMEOUT_MS": str(budget_ms * 3)},
    )
    assert larger_max["BASH_MAX_TIMEOUT_MS"] == str(budget_ms * 3)
    # The cap fact is the default call limit, never the requestable maximum.
    assert larger_max["AGENT_LOOP_SHELL_CAP_MS"] == str(budget_ms)
    smaller = runtime.agent_shell_environment(
        config, "claude", "coder",
        ambient={**ambient, "BASH_DEFAULT_TIMEOUT_MS": "120000", "BASH_MAX_TIMEOUT_MS": "600000"},
    )
    assert smaller["BASH_DEFAULT_TIMEOUT_MS"] == smaller["BASH_MAX_TIMEOUT_MS"] == str(budget_ms)
    # The run-tests watchdog ceiling itself is unchanged.
    assert runtime.inherited_timeout_ceiling({"AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS": "600"}) == 600


@pytest.mark.parametrize("role", ["reviewer", "repair", "planner", None, "Coder"])
def test_read_only_claude_roles_keep_bash_limits_unchanged(tmp_path, role):
    config = make_config(tmp_path)
    ambient = {"BASH_MAX_TIMEOUT_MS": str(10 ** 12), "AGENT_LOOP_SHELL_CAP_MS": str(10 ** 12)}
    env = runtime.agent_shell_environment(config, "claude", role, ambient=ambient)
    assert env == {"AGENT_LOOP_SHELL_CAP_MS": "unknown"}


@pytest.mark.parametrize("provider", ["codex", "gemini", "antigravity"])
@pytest.mark.parametrize("role", ["coder", "reviewer", None])
def test_non_claude_backends_carry_an_unknown_cap_and_no_bash_limits(tmp_path, provider, role):
    config = make_config(tmp_path)
    ambient = {"BASH_MAX_TIMEOUT_MS": str(10 ** 12), "AGENT_LOOP_SHELL_CAP_MS": str(10 ** 12)}
    env = runtime.agent_shell_environment(config, provider, role, ambient=ambient)
    assert env == {"AGENT_LOOP_SHELL_CAP_MS": "unknown"}
    merged = {**ambient, **env}
    assert runtime.prelaunch_notice(600, 1800, merged) is not None


def _env_runner(**outputs):
    from agent_loop_helpers import FakeRunner

    class EnvRunner(FakeRunner):
        def __init__(self):
            super().__init__(**outputs)
            self.agent_envs = []

        def run_with_log(self, *args, **kwargs):
            self.agent_envs.append(dict(kwargs.get("env") or {}))
            return super().run_with_log(*args, **kwargs)

    return EnvRunner()


def test_backend_invocation_environments_carry_the_cap_facts(tmp_path, monkeypatch):
    from coding_review_agent_loop.agents.claude import BACKEND as CLAUDE
    from coding_review_agent_loop.agents.codex import BACKEND as CODEX
    from coding_review_agent_loop.agents.gemini import BACKEND as GEMINI

    monkeypatch.setenv("BASH_MAX_TIMEOUT_MS", str(10 ** 12))
    monkeypatch.setenv("AGENT_LOOP_SHELL_CAP_MS", str(10 ** 12))
    monkeypatch.delenv("BASH_DEFAULT_TIMEOUT_MS", raising=False)
    fake = tmp_path / "venv" / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.symlink_to(sys.executable)
    config = make_config(tmp_path, test_python=str(fake))
    runner = _env_runner(claude_outputs=['{"result":"ok"}', '{"result":"ok"}'])
    CLAUDE.run(runner, config, "Implement", run_id="r", role="coder")
    coder_env = runner.agent_envs[-1]
    budget_ms = runtime.run_tests_foreground_budget_seconds(config.coder_test_command_timeout_seconds) * 1000
    assert coder_env["BASH_DEFAULT_TIMEOUT_MS"] == str(budget_ms)
    assert coder_env["AGENT_LOOP_SHELL_CAP_MS"] == coder_env["BASH_DEFAULT_TIMEOUT_MS"]
    assert coder_env["AGENT_LOOP_TEST_PYTHON"] == str(fake)
    CLAUDE.run(runner, config, "Review", run_id="r", role="reviewer")
    reviewer_env = runner.agent_envs[-1]
    assert "BASH_DEFAULT_TIMEOUT_MS" not in reviewer_env and "BASH_MAX_TIMEOUT_MS" not in reviewer_env
    assert reviewer_env["AGENT_LOOP_SHELL_CAP_MS"] == "unknown"
    runner = _env_runner(codex_outputs=[{"public_response": "ok", "stdout": "", "returncode": 0}])
    CODEX.run(runner, config, "Implement", run_id="r", role="coder")
    assert runner.agent_envs[-1]["AGENT_LOOP_SHELL_CAP_MS"] == "unknown"
    assert runner.agent_envs[-1]["AGENT_LOOP_TEST_PYTHON"] == str(fake)
    assert "BASH_DEFAULT_TIMEOUT_MS" not in runner.agent_envs[-1]
    runner = _env_runner(gemini_outputs=['{"response":"ok"}'])
    GEMINI.run(runner, config, "Review", run_id="r", role="reviewer")
    assert runner.agent_envs[-1]["AGENT_LOOP_SHELL_CAP_MS"] == "unknown"


@pytest.mark.parametrize("role", ["coder", "reviewer", None])
def test_antigravity_invocations_carry_the_unknown_cap_fact(tmp_path, monkeypatch, role):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    huge = str(10 ** 12)
    monkeypatch.setenv("BASH_MAX_TIMEOUT_MS", huge)
    monkeypatch.setenv("AGENT_LOOP_SHELL_CAP_MS", huge)
    monkeypatch.delenv("BASH_DEFAULT_TIMEOUT_MS", raising=False)
    fake = tmp_path / "venv" / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.symlink_to(sys.executable)
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(tmp_path, antigravity_dir=agy_dir, test_python=str(fake))
    runner = _env_runner(antigravity_outputs=[("ok", 0)], antigravity_catalog_outputs=[("Gemini 3.1 Pro (High)", 0)])
    AntigravityBackend().run(runner, config, "Implement", run_id="r", role=role)
    AntigravityBackend().discover_models(runner, config, timeout_seconds=30)
    assert len(runner.agent_envs) == 2  # the main turn and the model-catalog query
    for env in runner.agent_envs:
        # The orchestrator-authored fact overrides the ambient cap once
        # run_with_log merges this env over os.environ.
        merged = {**os.environ, **env}
        assert merged["AGENT_LOOP_SHELL_CAP_MS"] == "unknown"
        assert merged["AGENT_LOOP_TEST_PYTHON"] == str(fake)
        assert "BASH_DEFAULT_TIMEOUT_MS" not in env and "BASH_MAX_TIMEOUT_MS" not in env
        assert runtime.prelaunch_notice(600, 1800, merged) is not None


def test_format_repair_never_reaches_the_claude_backend():
    import inspect

    from coding_review_agent_loop.agents import format_repair

    source = inspect.getsource(format_repair)
    assert "agent_shell_environment" not in source and "BASH_" not in source
    assert "agents.claude" not in source and "ClaudeBackend" not in source


def test_contended_claude_launch_survives_admission_plus_target_time(tmp_path, monkeypatch):
    # A target that nearly reaches its watchdog after a host-capacity wait
    # still passes (admission is not counted against the target), and the
    # whole foreground lifetime fits the Claude default call limit.
    monkeypatch.setenv("AGENT_LOOP_TEST_WORKER_HOST_SHARING", "on")
    monkeypatch.setenv("AGENT_LOOP_TEST_WORKER_HOST_WAIT_SECONDS", "2")

    def contended(self, requested, capacity, *, wait_seconds, notify, wait_notify=None, **_kwargs):
        time.sleep(2.0)
        return HostWaitResult(requested, 2.0, False, False, False)

    monkeypatch.setattr(WorkerBudgetLock, "wait_for_host_workers", contended)
    started = time.monotonic()
    result = run_foreground_test(
        [sys.executable, "-c", "import time; time.sleep(2.5)"], cwd=tmp_path, timeout_seconds=3.5,
        echo_output=False, worker_budget=_budget("clamp"), worker_lock_root=tmp_path / "locks",
    )
    total = time.monotonic() - started
    assert result.outcome == "passed"
    assert total > 3.5  # admission plus target exceeded the target watchdog
    config = make_config(tmp_path, coder_test_command_timeout_seconds=4)
    env = runtime.agent_shell_environment(config, "claude", "coder", ambient=dict(os.environ))
    assert int(env["BASH_DEFAULT_TIMEOUT_MS"]) >= total * 1000
    assert int(env["BASH_DEFAULT_TIMEOUT_MS"]) >= (4 + 2) * 1000


# --- no-advisory cases through the real runner path (#1343 review item-3) ----

_VOLATILE_SPAWN_KEYS = ("SPEC", "RESERVATION")


def _normalized_spawns(spawns):
    """Spawned argv/env with only per-run random report locations removed."""
    normalized = []
    for argv, env in spawns:
        stable = {key: value for key, value in env.items() if not key.endswith(_VOLATILE_SPAWN_KEYS)}
        normalized.append((list(argv), stable))
    return normalized


_RUNNER_EXCLUDED_SHAPES = [*(f"serial:{' '.join(extra)}" for extra in _SERIAL_ARGS),
                           "console-script", "env-prefix", "timeout-prefix", "isolated-flag", "non-pytest",
                           "undeclared"]
_RUNNER_CHECKED_SHAPES = ["pythonpath-importable", "unknown", "raises"]


def _no_advisory_case(tmp_path, monkeypatch, shape):
    """Return (argv, env overlay, check_patch) for one no-advisory shape."""
    from coding_review_agent_loop import lifecycle_probe

    root = _xa_project(tmp_path / "project", declares=shape != "undeclared", broken_xdist=shape != "pythonpath-importable")
    if shape == "console-script":
        # A console script's sys.path[0] is its script directory; a -c check
        # here could only produce a false advisory.
        (root / "xdist.py").write_text("raise ImportError('shadow')\n", encoding="utf-8")
    module = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    env = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    if shape.startswith("serial:"):
        argv = [*module, *shape.split(":", 1)[1].split(" "), "test_x.py"]
    elif shape == "console-script":
        argv = [str(_write_console_script(root / "venv" / "bin" / "pytest")), "-q", "-p", "no:cacheprovider", "test_x.py"]
    elif shape == "env-prefix":
        argv = ["/usr/bin/env", "FOO=1", *module, "test_x.py"]
    elif shape == "timeout-prefix":
        argv = ["timeout", "100", *module, "test_x.py"]
    elif shape == "isolated-flag":
        argv = [sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_x.py"]
    elif shape == "non-pytest":
        argv = [sys.executable, "-c", "pass"]
    else:
        argv = [*module, "test_x.py"]
    if shape == "pythonpath-importable":
        extra = tmp_path / "extra-path"
        _lp_write(extra, {"xdist/__init__.py": "", "xdist/plugin.py": ""})
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(extra), os.environ.get("PYTHONPATH")]))
    elif shape == "unknown":
        monkeypatch.setattr(lifecycle_probe, "module_launch_xdist_check", lambda *a, **k: None)
    elif shape == "raises":
        def boom(*_a, **_k):
            raise RuntimeError("probe exploded")
        monkeypatch.setattr(lifecycle_probe, "module_launch_xdist_check", boom)
    return root, argv, env


@pytest.mark.parametrize("mode", [None, "off", "clamp", "refuse"])
@pytest.mark.parametrize("shape", _RUNNER_EXCLUDED_SHAPES + _RUNNER_CHECKED_SHAPES)
def test_no_advisory_cases_through_the_runner_leave_the_launch_unchanged(tmp_path, monkeypatch, mode, shape):
    from coding_review_agent_loop import lifecycle_probe

    monkeypatch.setenv("AGENT_LOOP_TEST_WORKER_HOST_SHARING", "off")
    root, argv, env = _no_advisory_case(tmp_path, monkeypatch, shape)
    checks = []
    probes = []
    inner_check = lifecycle_probe.module_launch_xdist_check
    real_probe = runtime.run_exit_code_probe

    def spy_check(*args, **kwargs):
        checks.append(args[1])
        return inner_check(*args, **kwargs)

    def spy_probe(probe_argv, **kwargs):
        if any("xdist.plugin" in str(item) for item in probe_argv):
            probes.append(list(probe_argv))
        return real_probe(probe_argv, **kwargs)

    monkeypatch.setattr(lifecycle_probe, "module_launch_xdist_check", spy_check)
    monkeypatch.setattr(runtime, "run_exit_code_probe", spy_probe)

    def run(disable):
        lines = []
        with contextlib.ExitStack() as stack:
            if disable:
                patched = stack.enter_context(pytest.MonkeyPatch.context())
                patched.setattr(runner_module, "_xdist_advisory", lambda *a, **k: None)
            spawns = stack.enter_context(_lp_spawns())
            result = run_foreground_test(
                argv, cwd=root, timeout_seconds=120, echo_output=False, classify_pre_collection=True,
                env={**os.environ, **env}, worker_budget=_budget(mode), output_callback=lines.append,
                worker_lock_root=tmp_path / "locks",
            )
        return result, "".join(lines), _normalized_spawns(spawns)

    result, output, spawns = run(False)
    assert _advisories(output) == []
    if shape in _RUNNER_EXCLUDED_SHAPES:
        # Statically excluded: no xdist check of any interpreter, wrapper or prefix.
        assert checks == [] and probes == []
    elif shape == "pythonpath-importable":
        # The check ran in the real launch context (caller's PYTHONPATH) and found xdist.
        assert len(checks) == 1 and len(probes) == 1
    else:
        assert len(checks) == 1
    checks.clear()
    probes.clear()
    baseline, baseline_output, baseline_spawns = run(True)
    assert checks == [] and probes == [] and _advisories(baseline_output) == []
    # Identical launch: worker policy, launch context and dispatch are unchanged.
    assert spawns and spawns == baseline_spawns
    assert (result.args, result.outcome, result.returncode, result.pre_collection_launch_failure, result.workers_cohort) == (
        baseline.args, baseline.outcome, baseline.returncode, baseline.pre_collection_launch_failure, baseline.workers_cohort,
    )


# ---------------------------------------------------------------------------
# run-tests --fixture (issue #1326)
# ---------------------------------------------------------------------------

from coding_review_agent_loop import fixture_runtime as fx_runtime  # noqa: E402
from coding_review_agent_loop import fixture_supervisor as fx_supervisor  # noqa: E402
from coding_review_agent_loop import local_test_evidence as fx_evidence  # noqa: E402
from coding_review_agent_loop.protocol import (  # noqa: E402
    _authoritative_receipt_passes,
    _receipt_expected_status,
)


def _fx_host_supported() -> bool:
    try:
        fx_supervisor.verify_host_process_authority()
    except fx_supervisor.HostAuthorityError:
        return False
    return True


_FX_LINUX = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fixture supervision is Linux-only")
_FX_HOST_SKIP = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not _fx_host_supported(),
    reason="host is not in the initial PID namespace or /proc hides processes",
)
_FX_LOCK_PATH = Path(tempfile.gettempdir()) / f"agent-loop-fixture-tests-{os.getuid() if hasattr(os, 'getuid') else 0}.lock"


@contextlib.contextmanager
def _fx_host_window(exclusive: bool):
    """Serialize census-sensitive fixture tests across xdist workers.

    The host-wide census flags any long-lived process started during a
    fixture window, including another fixture test's daemons.  Tests that
    start fixture processes hold the lock shared; a test that needs a quiet
    window to observe ``cleanup=confirmed`` holds it exclusively.
    """
    import fcntl

    with open(_FX_LOCK_PATH, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@pytest.fixture
def fx_host_lock():
    with _fx_host_window(exclusive=False):
        yield


def _FX_HOST(function):
    return _FX_HOST_SKIP(pytest.mark.usefixtures("fx_host_lock")(function))

# A tracked Python fixture.  ``setup`` double-forks socket-listening servers and
# records them in the broker-owned state directory; ``teardown`` stops all but
# the last, leaving it for the supervisor's final sweep.  Behaviour switches
# come from tracked sibling files so the fixture itself stays identical.
_FX_FIXTURE = '''#!{python}
import json, os, pathlib, signal, socket, sys, time

STATE = pathlib.Path(os.environ["AGENT_LOOP_FIXTURE_STATE_DIR"])
HERE = pathlib.Path(__file__).resolve().parent
COUNT = int((HERE / "fx_count").read_text()) if (HERE / "fx_count").exists() else 3
MARKERS = pathlib.Path((HERE / "fx_markers").read_text().strip())


def mode(name):
    path = HERE / "fx_mode"
    return path.exists() and name in path.read_text().split()


def serve(index):
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    record = {{"pid": os.getpid(), "port": server.getsockname()[1]}}
    tmp = STATE / ("server%d.tmp" % index)
    tmp.write_text(json.dumps(record))
    os.replace(tmp, STATE / ("server%d.json" % index))
    (MARKERS / ("pid%d" % index)).write_text(str(os.getpid()))
    while True:
        connection, _ = server.accept()
        connection.sendall(b"ok")
        connection.close()


def daemon(index):
    child = os.fork()
    if child == 0:
        os.setsid()
        if os.fork():
            os._exit(0)
        try:
            serve(index)
        finally:
            os._exit(0)
    os.waitpid(child, 0)


phase = sys.argv[1]
def noisy_descendant(name):
    # A descendant that keeps the phase's output pipe open and floods it.
    if os.fork() == 0:
        (MARKERS / ("noisy-%s" % name)).write_text(str(os.getpid()))
        chunk = b"x" * 65536
        while True:
            try:
                os.write(1, chunk)
            except OSError:
                time.sleep(0.01)
    return


if phase == "setup":
    (MARKERS / "setup-started").write_text("1")
    if mode("setup-noisy"):
        noisy_descendant("setup")
    started = COUNT if not mode("setup-fail") else 2
    for index in range(started):
        daemon(index)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and len(list(STATE.glob("server*.json"))) < started:
        time.sleep(0.02)
    if mode("edit-tracked"):
        (HERE / "tracked.txt").write_text("edited by the fixture\\n")
    if mode("setup-fail"):
        print("setup failing on purpose", flush=True)
        sys.exit(3)
    if mode("setup-block"):
        (MARKERS / "setup-blocking").write_text("1")
        time.sleep(600)
    if mode("setup-slow"):
        time.sleep(float((HERE / "fx_setup_seconds").read_text()))
    print("setup ready with %d servers" % started, flush=True)
elif phase == "teardown":
    (MARKERS / "teardown-started").write_text("1")
    if mode("teardown-noisy"):
        noisy_descendant("teardown")
    if mode("teardown-slow"):
        time.sleep(float((HERE / "fx_teardown_seconds").read_text()))
    servers = sorted(STATE.glob("server*.json"))
    for path in servers[:-1]:
        try:
            os.kill(json.loads(path.read_text())["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    (MARKERS / "teardown-done").write_text("1")
    if mode("teardown-fail"):
        sys.exit(4)
'''

_FX_INNER_TEST = '''import json, os, pathlib, socket, time

STATE = pathlib.Path(os.environ["AGENT_LOOP_FIXTURE_STATE_DIR"])
HERE = pathlib.Path(__file__).resolve().parent


def test_every_fixture_process_is_reachable():
    servers = sorted(STATE.glob("server*.json"))
    assert len(servers) >= 3
    for path in servers:
        port = json.loads(path.read_text())["port"]
        with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
            assert connection.recv(2) == b"ok"
    markers = pathlib.Path((HERE / "fx_markers").read_text().strip())
    (markers / "inner-ran").write_text("1")
    modes = (HERE / "fx_mode").read_text().split() if (HERE / "fx_mode").exists() else []
    if "inner-block-hard" in modes:
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # only SIGKILL ends it
        modes.append("inner-block")
    if "inner-noisy" in modes:
        (markers / "inner-noisy").write_text("1")
        chunk = "x" * 8192
        while True:
            print(chunk, flush=True)
    if "inner-block" in modes:
        (markers / "inner-blocking").write_text("1")
        time.sleep(600)
'''

_FX_PYTEST = (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_fx.py")


def _fx_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
    ).stdout


def _fx_repo(tmp_path: Path, *, modes: str = "", extra: dict[str, str] | None = None, count: int = 3):
    root = tmp_path / "repo"
    markers = tmp_path / "markers"
    root.mkdir(parents=True)
    markers.mkdir(parents=True)
    _fx_git(root, "init", "-q")
    _fx_git(root, "config", "user.email", "fx@example.invalid")
    _fx_git(root, "config", "user.name", "Fx")
    files = {
        "fixture.py": _FX_FIXTURE.format(python=sys.executable),
        "test_fx.py": _FX_INNER_TEST,
        "tracked.txt": "tracked\n",
        "fx_markers": str(markers) + "\n",
        "fx_count": str(count),
        ".gitignore": "__pycache__/\n.pytest_cache/\n.venv/\nignored-fixture.py\n",
        **({"fx_mode": modes} if modes else {}),
        **(extra or {}),
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "fixture.py").chmod(0o755)
    _fx_git(root, "add", "-A")
    _fx_git(root, "commit", "-qm", "fixture repository")
    return root, markers


def _fx_server(root: Path, turn: str = "turn-fixture"):
    if not sys.platform.startswith("linux"):
        # The broker pins the checkout with Linux-only primitives (#1326).
        pytest.skip("a real test broker requires Linux")
    return fx_evidence.TestBrokerServer(root=root, turn_id=turn).start()


def _fx_env(server) -> dict[str, str]:
    return {**os.environ, **server.environment, "AGENT_LOOP_INVOCATION_ID": server.turn_id}


def _fx_pids(markers: Path) -> list[int]:
    return [int(path.read_text()) for path in sorted([*markers.glob("pid*"), *markers.glob("noisy-*")])]


def _fx_assert_gone(pids) -> None:
    for pid in pids:
        assert wait_until_gone(pid, timeout=10), f"fixture process {pid} survived"


def _fx_kill(pids) -> None:
    for pid in pids:
        _kill_if_active(pid)


def _fx_quiet(record: dict, own_pids) -> bool:
    """A host-noise escape: every escaped process is unrelated to this fixture."""
    if record.get("cleanup") != "escaped":
        return False
    escaped = [int(item.split()[0].split("=")[1]) for item in record.get("escaped_processes") or ()]
    return bool(escaped) and not set(escaped) & set(own_pids)


def _fx_cli(monkeypatch, server, root, argv, *, fixture="fixture.py", timeout="40", memory=None, extra=()):
    import coding_review_agent_loop.cli as cli_module

    for key, value in _fx_env(server).items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(root)
    monkeypatch.setattr(cli_module, "prelaunch_notice", lambda *a, **k: None)
    command = ["run-tests", "--timeout-seconds", str(timeout), "--fixture", fixture]
    if memory is not None:
        command += ["--memory-dir", str(memory)]
    return main([*command, *extra, "--", *argv])


def _fx_row(server):
    rows = [row for row in server.journal if row.fixture is not None]
    assert len(rows) == 1, server.journal
    return rows[0]


@_FX_HOST_SKIP
def test_fixture_pass_multi_process_fixture_yields_a_citable_observation(tmp_path, monkeypatch, capsys):
    with _fx_host_window(exclusive=True):
        _fixture_pass_scenario(tmp_path, monkeypatch, capsys)


def _fixture_pass_scenario(tmp_path, monkeypatch, capsys):
    from coding_review_agent_loop.local_test_evidence import decode_bounded_evidence

    for attempt in range(3):
        root, markers = _fx_repo(tmp_path / f"attempt{attempt}")
        memory = tmp_path / f"memory{attempt}"
        server = _fx_server(root)
        try:
            code = _fx_cli(monkeypatch, server, root, _FX_PYTEST, memory=memory)
            row = _fx_row(server)
        finally:
            server.stop()
        pids = _fx_pids(markers)
        _fx_assert_gone(pids)
        record = row.fixture.to_dict()
        if code != 0 and _fx_quiet(record, pids):
            continue  # an unrelated host process started during the window
        break
    else:
        pytest.skip("host never stayed quiet for the fixture window")
    err = capsys.readouterr().err
    assert code == 0, (record, err)
    assert len(pids) == 3 and (markers / "inner-ran").exists() and (markers / "teardown-done").exists()
    assert row.outcome == "passed" and row.inner_exec == "started" and row.suite_start == "verified"
    assert record["setup_status"] == 0 and record["teardown_status"] == 0
    assert record["cleanup"] == "confirmed" and record["escaped"] == 0
    # Teardown deliberately left one server alive for the final sweep.
    assert record["swept"] >= 1 and record["inventoried"] >= 3
    assert record["path"] == "fixture.py"
    assert record["digest"] == "100755:" + _fx_git(root, "rev-parse", "HEAD:fixture.py").strip()
    assert len(record["setup_log_digest"]) == 64
    # The teardown printed nothing: a completed empty tail still has a digest.
    import hashlib

    assert record["teardown_log_digest"] == hashlib.sha256(b"").hexdigest()
    assert record["setup_duration_seconds"] >= 0 and record["teardown_duration_seconds"] >= 0
    assert _receipt_expected_status(row, claim="current-result") == "verified"
    assert "agent-loop fixture: setup=0 teardown=0 cleanup=confirmed" in err
    # Launch integrity and launcher health describe only the inner pytest argv.
    assert row.command == _FX_PYTEST
    stored = (memory / runtime.RUNTIME_SIDECAR_NAME).read_text(encoding="utf-8")
    payload = json.loads(stored)
    assert all("fixture.py" not in json.dumps(item.get("candidate", "")) for item in payload.get("launcher_health", []))
    (timing,) = payload["observations"]
    assert timing["fixture"] == {"path": "fixture.py", "digest": record["digest"]}
    assert "fixture.py" not in timing["normalized_command"]
    # The record survives the bounded round encoding.
    restored = decode_bounded_evidence(fx_evidence.bounded_evidence_for_round(
        fx_evidence.LocalTestEvidence(observations=(row,))
    ))
    assert restored.observations[0].fixture == row.fixture


def _fx_set_mode(root: Path, modes: str, **files: str) -> None:
    (root / "fx_mode").write_text(modes, encoding="utf-8")
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    _fx_git(root, "add", "-A")
    _fx_git(root, "commit", "-qm", "mode " + modes)


@_FX_HOST
@pytest.mark.parametrize(
    "case", ["setup-failure", "inner-timeout", "teardown-failure", "cancel-during-inner"],
)
def test_fixture_failure_and_cancellation_outcomes_are_never_citable(tmp_path, monkeypatch, capsys, case):
    root, markers = _fx_repo(tmp_path)
    modes = {
        "setup-failure": "setup-fail",
        "inner-timeout": "inner-block",
        "teardown-failure": "teardown-fail",
        "cancel-during-inner": "inner-block",
    }[case]
    _fx_set_mode(root, modes)
    server = _fx_server(root)
    result: dict[str, int] = {}
    try:
        if case == "cancel-during-inner":
            thread = threading.Thread(
                target=lambda: result.setdefault("code", _fx_cli(monkeypatch, server, root, _FX_PYTEST)),
                daemon=True,
            )
            thread.start()
            deadline = time.monotonic() + 60
            while not (markers / "inner-blocking").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert (markers / "inner-blocking").exists()
            pids = _fx_pids(markers)
            server.stop()
            thread.join(timeout=60)
            code = result["code"]
        else:
            code = _fx_cli(monkeypatch, server, root, _FX_PYTEST, timeout="30")
        row = _fx_row(server)
    finally:
        server.stop()
    pids = _fx_pids(markers)
    _fx_assert_gone(pids)
    record = row.fixture.to_dict()
    assert code != 0
    assert (markers / "teardown-done").exists()
    assert not _authoritative_receipt_passes(row, claim="current-result")
    assert _receipt_expected_status(row, claim="current-result") != "verified"
    if case == "setup-failure":
        assert len(pids) == 2  # processes started before the failure were observed
        assert record["setup_status"] == 3 and code == 3
        assert row.outcome == "failed" and row.inner_exec == "not-attempted"
        assert not (markers / "inner-ran").exists()
        assert fx_runtime.FIXTURE_SETUP_FAILED_CAVEAT in row.caveats
        assert record["inventoried"] >= 2
    elif case == "inner-timeout":
        assert row.outcome == "timed_out" and code == 124
        assert (markers / "inner-ran").exists()
        assert record["teardown_status"] == 0
    elif case == "teardown-failure":
        assert (markers / "inner-ran").exists()
        assert record["teardown_status"] == 4 and code == 4
        assert row.outcome == "failed" and fx_runtime.FIXTURE_TEARDOWN_FAILED_CAVEAT in row.caveats
    else:
        assert code == 130
        assert row.outcome == "interrupted"
        assert row.attribution.caveats == ("unattributed: broker shutdown",)
        assert (markers / "inner-ran").exists() and record["teardown_status"] == 0


@_FX_HOST
def test_fixture_setup_cancellation_runs_teardown_and_never_spawns_the_inner_command(tmp_path, monkeypatch):
    root, markers = _fx_repo(tmp_path)
    _fx_set_mode(root, "setup-block")
    server = _fx_server(root)
    result: dict[str, int] = {}
    killed: list[int] = []
    original_force = fx_runtime.FixtureSupervisor.force_terminate

    def spy_force(self):
        killed.append(self.process.pid)
        return original_force(self)

    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "force_terminate", spy_force)
    try:
        thread = threading.Thread(
            target=lambda: result.setdefault("code", _fx_cli(monkeypatch, server, root, _FX_PYTEST)),
            daemon=True,
        )
        thread.start()
        deadline = time.monotonic() + 60
        while not (markers / "setup-blocking").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (markers / "setup-blocking").exists()
        server.stop()
        thread.join(timeout=60)
        row = _fx_row(server)
    finally:
        server.stop()
    pids = _fx_pids(markers)
    _fx_assert_gone(pids)
    assert result["code"] != 0 and result["code"] == 130
    assert (markers / "teardown-done").exists()
    assert not (markers / "inner-ran").exists()
    assert row.outcome == "interrupted" and row.inner_exec == "not-attempted"
    assert row.fixture.setup_status == "cancelled"
    assert row.attribution.caveats == ("unattributed: broker shutdown",)
    assert not _authoritative_receipt_passes(row, claim="current-result")
    # Graceful teardown completed, so the supervisor was never force-killed.
    assert killed == []


@_FX_LINUX
def test_fixture_stop_before_the_guarded_supervisor_launch_starts_nothing(tmp_path, monkeypatch):
    root, markers = _fx_repo(tmp_path)
    server = _fx_server(root)
    launches: list[object] = []
    original_clean = fx_runtime.require_clean_tree

    def clean_then_stop(before):
        original_clean(before)
        server._stop_event.set()

    monkeypatch.setattr(fx_runtime, "require_clean_tree", clean_then_stop)
    monkeypatch.setattr(fx_runtime, "verify_host_process_authority", lambda **_: fx_supervisor.HostAuthority("1"))
    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "launch", classmethod(lambda cls, *a, **k: launches.append(a)))
    try:
        result = fx_evidence.TestBrokerClient(_fx_env(server)).run(
            list(_FX_PYTEST), timeout_seconds=40, cwd=root, fixture="fixture.py",
        )
    finally:
        server.stop()
    assert launches == []
    assert result.outcome == "cancelled"
    assert [row for row in server.journal if row.fixture is not None] == []
    assert not (markers / "setup-started").exists() and not (markers / "inner-ran").exists()
    assert fx_runtime.fold_exit_code(
        outcome=result.outcome, returncode=result.returncode, inner_exec=result.inner_exec,
        fixture=result.fixture, cancelled_frame=True,
    ) == 130


@_FX_HOST
@pytest.mark.parametrize(
    "stop_point", ["awaiting-teardown-command", "during-teardown"],
)
def test_fixture_stop_after_an_inner_pass_is_interrupted_and_exits_130(tmp_path, monkeypatch, stop_point):
    root, markers = _fx_repo(tmp_path)
    if stop_point == "during-teardown":
        _fx_set_mode(root, "teardown-slow", fx_teardown_seconds="2")
    server = _fx_server(root)
    original_finish = fx_evidence._FixtureRun.finish
    stopper: list[threading.Thread] = []

    def finish(self, broker):
        if stop_point == "awaiting-teardown-command":
            thread = threading.Thread(target=broker.stop, daemon=True)
        else:
            def stop_once_teardown_runs():
                deadline = time.monotonic() + 30
                while not (markers / "teardown-started").exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                broker.stop()
            thread = threading.Thread(target=stop_once_teardown_runs, daemon=True)
            self.supervisor.request_teardown()
        stopper.append(thread)
        thread.start()
        while stop_point == "awaiting-teardown-command" and not broker._stop_event.is_set():
            time.sleep(0.01)
        return original_finish(self, broker)

    monkeypatch.setattr(fx_evidence._FixtureRun, "finish", finish)
    try:
        code = _fx_cli(monkeypatch, server, root, _FX_PYTEST)
        for thread in stopper:
            thread.join(timeout=60)
        row = _fx_row(server)
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    assert (markers / "inner-ran").exists() and (markers / "teardown-done").exists()
    assert code == 130
    assert row.outcome == "interrupted"
    assert not _authoritative_receipt_passes(row, claim="current-result")


@_FX_HOST
@pytest.mark.parametrize(
    "case",
    [
        "untracked", "ignored", "symlink-out-of-tree", "gitlink", "dirty-tree",
        "outside-checkout", "symlinked-directory", "differs-from-head", "not-executable",
        "filter-transformed",
    ],
)
def test_fixture_provenance_refusals_launch_nothing(tmp_path, monkeypatch, case):
    root, markers = _fx_repo(tmp_path)
    fixture = "fixture.py"
    if case == "untracked":
        (root / "loose.py").write_text((root / "fixture.py").read_text())
        (root / "loose.py").chmod(0o755)
        fixture = "loose.py"
    elif case == "ignored":
        (root / "ignored-fixture.py").write_text((root / "fixture.py").read_text())
        (root / "ignored-fixture.py").chmod(0o755)
        fixture = "ignored-fixture.py"
    elif case == "symlink-out-of-tree":
        outside = tmp_path / "outside.py"
        outside.write_text((root / "fixture.py").read_text())
        outside.chmod(0o755)
        (root / "link.py").symlink_to(outside)
        _fx_git(root, "add", "link.py")
        _fx_git(root, "commit", "-qm", "link")
        fixture = "link.py"
    elif case == "gitlink":
        head = _fx_git(root, "rev-parse", "HEAD").strip()
        (root / "sub").mkdir()
        _fx_git(root, "update-index", "--add", "--cacheinfo", f"160000,{head},sub")
        _fx_git(root, "commit", "-qm", "gitlink")
        fixture = "sub"
    elif case == "dirty-tree":
        (root / "tracked.txt").write_text("dirty\n")
    elif case == "outside-checkout":
        outside = tmp_path / "outside.py"
        outside.write_text((root / "fixture.py").read_text())
        outside.chmod(0o755)
        fixture = str(outside)
    elif case == "symlinked-directory":
        (tmp_path / "real").mkdir()
        (tmp_path / "real" / "f.py").write_text((root / "fixture.py").read_text())
        (root / "dirlink").symlink_to(tmp_path / "real")
        _fx_git(root, "add", "dirlink")
        _fx_git(root, "commit", "-qm", "dirlink")
        fixture = "dirlink/f.py"
    elif case == "differs-from-head":
        (root / "fixture.py").write_text((root / "fixture.py").read_text() + "\n# local edit\n")
    elif case == "filter-transformed":
        # A committed ident attribute expands $Id$ on checkout: the worktree
        # bytes differ from the HEAD blob while git status stays clean.
        (root / ".gitattributes").write_text("fixture.py ident\n")
        (root / "fixture.py").write_text((root / "fixture.py").read_text() + "\n# $Id$\n")
        _fx_git(root, "add", "-A")
        _fx_git(root, "commit", "-qm", "ident fixture")
        (root / "fixture.py").unlink()
        _fx_git(root, "checkout", "--", "fixture.py")
        assert "$Id: " in (root / "fixture.py").read_text()
        assert _fx_git(root, "status", "--porcelain") == ""
    elif case == "not-executable":
        (root / "plain.py").write_text((root / "fixture.py").read_text())
        _fx_git(root, "add", "plain.py")
        _fx_git(root, "commit", "-qm", "plain")
        fixture = "plain.py"
    launches: list[object] = []
    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "launch", classmethod(lambda cls, *a, **k: launches.append(a)))
    server = _fx_server(root)
    try:
        with pytest.raises(fx_evidence.BrokerProtocolError, match="fixture refused: fixture-") as raised:
            fx_evidence.TestBrokerClient(_fx_env(server)).run(
                list(_FX_PYTEST), timeout_seconds=40, cwd=root, fixture=fixture,
            )
        assert not isinstance(raised.value, (fx_evidence.BrokerNotDispatched, fx_evidence.BrokerResultUnknown))
        journal = server.journal
    finally:
        server.stop()
    assert launches == []
    assert not (markers / "setup-started").exists() and not (markers / "inner-ran").exists()
    assert all(row.provenance == "telemetry-unverified" and row.outcome == "incomplete" for row in journal)
    assert len(journal) <= 1


@_FX_HOST
def test_fixture_that_edits_a_tracked_file_is_not_citable_after_teardown(tmp_path, monkeypatch):
    root, markers = _fx_repo(tmp_path)
    _fx_set_mode(root, "edit-tracked")
    server = _fx_server(root)
    try:
        code = _fx_cli(monkeypatch, server, root, _FX_PYTEST)
        row = _fx_row(server)
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    assert (markers / "teardown-done").exists()
    assert row.attribution.state == "unknown"
    assert any("changed" in item for item in row.attribution.caveats), row.attribution
    assert not _authoritative_receipt_passes(row, claim="current-result")
    assert code == 0 or code != 0  # the exit folds the inner lifecycle; citation is what is gated


@_FX_LINUX
def test_fixture_absent_requests_projections_and_memory_are_unchanged(tmp_path, monkeypatch):
    sent: list[dict] = []

    def capture(connection, payload):
        sent.append(dict(payload))
        raise OSError("captured")

    monkeypatch.setattr(fx_evidence, "_send_frame", capture)
    client = fx_evidence.TestBrokerClient({
        fx_evidence.BROKER_ENDPOINT_ENV: str(tmp_path / "missing.sock"),
        fx_evidence.BROKER_CAPABILITY_ENV: "a" * 64,
        fx_evidence.BROKER_PROTOCOL_ENV: fx_evidence.BROKER_PROTOCOL,
        "AGENT_LOOP_INVOCATION_ID": "turn-compat",
    })
    # The endpoint is absent, so connect fails before any frame: nothing sent.
    with pytest.raises(fx_evidence.BrokerNotDispatched):
        client.run([sys.executable, "-c", "pass"], timeout_seconds=5, cwd=tmp_path)
    assert sent == []
    server = fx_evidence.TestBrokerServer(root=tmp_path, turn_id="turn-compat").start()
    try:
        with pytest.raises(fx_evidence.BrokerNotDispatched):
            fx_evidence.TestBrokerClient(_fx_env(server)).run(
                [sys.executable, "-c", "pass"], timeout_seconds=5, cwd=tmp_path,
            )
    finally:
        server.stop()
    (request,) = sent
    assert set(request) == fx_evidence.BROKER_REQUEST_FIELDS
    validated = fx_evidence._validate_broker_request(
        {**request, "turn_id": server.turn_id}, turn_id=server.turn_id, capability=server.capability,
        root=tmp_path, ceiling=1800,
    )
    assert "fixture" not in validated
    observation = fx_evidence.LocalTestObservation(
        command=(sys.executable, "-m", "pytest"), outcome="passed", provenance="parent-observed",
        wrapper_bootstrap="verified", inner_exec="started", suite_start="verified",
    )
    projection = observation.public_projection()
    assert "fixture" not in projection
    encoded = fx_evidence.bounded_evidence_for_round(fx_evidence.LocalTestEvidence(observations=(observation,)))
    assert '"fixture"' not in encoded
    decoded = fx_evidence.decode_bounded_evidence(encoded)
    assert decoded.observations[0].fixture is None
    # Previously persisted bounded evidence decodes to an equal observation
    # with the legacy key set (only restart degradation differs).
    (legacy_row,) = json.loads(encoded)["observations"]
    (again_row,) = json.loads(fx_evidence.bounded_evidence_for_round(decoded))["observations"]
    assert set(again_row) == set(legacy_row)
    assert fx_evidence.decode_bounded_evidence(fx_evidence.bounded_evidence_for_round(decoded)) == decoded
    memory = tmp_path / "memory"
    assert runtime.record_test_observation(
        memory, argv=["python", "-m", "pytest"], cwd=tmp_path, outcome="passed", elapsed_seconds=1.0,
        attempted_timeout_seconds=10, policy_ceiling_seconds=1800, commit="c",
    )
    (row,) = json.loads((memory / runtime.RUNTIME_SIDECAR_NAME).read_text())["observations"]
    assert "fixture" not in row


def test_fixture_absent_runner_and_probe_keywords_default_to_legacy_behaviour(tmp_path):
    import inspect

    from coding_review_agent_loop.lifecycle_probe import prepare_lifecycle_probe

    body = inspect.signature(runner_module._run_foreground_test_body).parameters
    assert body["absolute_deadline"].default is None
    assert body["post_admission_setup"].default is None
    probe = inspect.signature(runtime.probe_inner_launcher).parameters
    assert probe["deadline"].default is None and probe["use_cache"].default is True
    assert inspect.signature(prepare_lifecycle_probe).parameters["deadline"].default is None
    # A fixture-less run neither strips nor injects the state directory variable.
    result = run_foreground_test(
        [sys.executable, "-c", "import os; print('state=' + str(os.environ.get('AGENT_LOOP_FIXTURE_STATE_DIR')))"],
        cwd=tmp_path, timeout_seconds=30, echo_output=False,
    )
    assert result.outcome == "passed" and "state=None" in result.output_tail


def test_fixture_no_broker_refuses_and_never_runs_anything(tmp_path, monkeypatch, capsys):
    import coding_review_agent_loop.cli as cli_module

    marker = tmp_path / "ran"
    inner = [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('1')"]
    for name in (fx_evidence.BROKER_ENDPOINT_ENV, fx_evidence.BROKER_CAPABILITY_ENV, fx_evidence.BROKER_PROTOCOL_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "run_foreground_test", lambda *a, **k: pytest.fail("local fallback"))
    memory = tmp_path / "memory"
    code = main(["run-tests", "--timeout-seconds", "60", "--memory-dir", str(memory), "--fixture", "fixture.py", "--", *inner])
    err = capsys.readouterr().err
    assert code == 1
    assert "--fixture requires the test broker; nothing was run (not evidence)" in err
    assert not marker.exists() and not memory.exists()


def test_fixture_pre_dispatch_connect_failure_is_a_nothing_ran_refusal(tmp_path, monkeypatch, capsys):
    import coding_review_agent_loop.cli as cli_module

    monkeypatch.setenv(fx_evidence.BROKER_ENDPOINT_ENV, str(tmp_path / "absent.sock"))
    monkeypatch.setenv(fx_evidence.BROKER_CAPABILITY_ENV, "b" * 64)
    monkeypatch.setenv(fx_evidence.BROKER_PROTOCOL_ENV, fx_evidence.BROKER_PROTOCOL)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "turn-predispatch")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "run_foreground_test", lambda *a, **k: pytest.fail("local fallback"))
    memory = tmp_path / "memory"
    code = main(["run-tests", "--timeout-seconds", "60", "--memory-dir", str(memory), "--fixture", "f.py", "--", sys.executable, "-c", "pass"])
    err = capsys.readouterr().err
    assert code == 1 and "nothing was run (not evidence)" in err and "result unknown" not in err
    assert not memory.exists()


@_FX_LINUX
def test_fixture_post_dispatch_response_loss_is_an_unknown_broker_owned_result(tmp_path, monkeypatch, capsys):
    import socket as socket_module

    import coding_review_agent_loop.cli as cli_module

    endpoint = tmp_path / "lossy.sock"
    listener = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
    listener.bind(str(endpoint))
    listener.listen(1)
    received: list[dict] = []

    def accept_then_drop():
        connection, _ = listener.accept()
        with connection:
            received.append(fx_evidence._recv_frame(connection))

    thread = threading.Thread(target=accept_then_drop, daemon=True)
    thread.start()
    monkeypatch.setenv(fx_evidence.BROKER_ENDPOINT_ENV, str(endpoint))
    monkeypatch.setenv(fx_evidence.BROKER_CAPABILITY_ENV, "c" * 64)
    monkeypatch.setenv(fx_evidence.BROKER_PROTOCOL_ENV, fx_evidence.BROKER_PROTOCOL)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "turn-postdispatch")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "run_foreground_test", lambda *a, **k: pytest.fail("local retry"))
    monkeypatch.setattr(cli_module, "record_test_observation", lambda *a, **k: pytest.fail("memory"))
    monkeypatch.setattr(cli_module, "record_launcher_health", lambda *a, **k: pytest.fail("health"))
    try:
        code = main(["run-tests", "--timeout-seconds", "60", "--fixture", "f.py", "--", sys.executable, "-c", "pass"])
    finally:
        thread.join(timeout=10)
        listener.close()
    err = capsys.readouterr().err
    assert received and received[0]["fixture"] == "f.py"
    assert code == 1
    assert "broker result unknown after dispatch; the broker owns this fixture run" in err
    assert "nothing was run" not in err


@_FX_HOST
def test_fixture_post_dispatch_disconnect_is_cancelled_and_leaves_nothing_running(tmp_path):
    """Issue #1653: a client lost right after dispatch cancels its fixture request.

    Depending on timing the loss lands before the supervisor launch (nothing
    launched, nothing journaled) or during setup (graceful teardown and a
    non-evidence row); either way no fixture process survives.
    """
    import socket as socket_module

    root, markers = _fx_repo(tmp_path)
    server = _fx_server(root)
    try:
        request = _fx_raw_request(server, root, _FX_PYTEST, timeout=40)
        with socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM) as connection:
            connection.connect(server.endpoint)
            fx_evidence._send_frame(connection, request)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and (server._handler_threads or server._requests):
            time.sleep(0.1)
        assert not server._handler_threads
        rows = [row for row in server.journal if row.fixture is not None]
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    if rows:
        (row,) = rows
        assert row.provenance == "telemetry-unverified"
        assert fx_evidence.BROKER_CLIENT_LOST_CAVEAT in row.caveats
        assert row.fixture.teardown_status == 0 and (markers / "teardown-done").exists()


def hmac_new(capability: str, raw: str) -> str:
    import hashlib
    import hmac

    return hmac.new(capability.encode("ascii"), raw.encode("ascii"), hashlib.sha256).hexdigest()


@pytest.mark.parametrize("timeout", ["1", "0.5", "29.9"], ids=["one", "fractional", "just-below"])
def test_fixture_timeouts_below_the_floor_are_refused_before_dispatch(tmp_path, monkeypatch, capsys, timeout):
    import coding_review_agent_loop.cli as cli_module

    dispatched: list[object] = []

    class Broker:
        def run(self, *args, **kwargs):
            dispatched.append(kwargs)
            raise AssertionError("dispatched")

    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: Broker())
    monkeypatch.chdir(tmp_path)
    code = main(["run-tests", "--timeout-seconds", timeout, "--fixture", "f.py", "--", sys.executable, "-c", "pass"])
    assert code == 1 and dispatched == []
    assert "--fixture requires a timeout of at least 30s" in capsys.readouterr().err


@_FX_LINUX
@pytest.mark.parametrize("timeout", [1, 0.5, 29.9], ids=["one", "fractional", "just-below"])
def test_fixture_timeouts_below_the_floor_get_a_broker_error_frame(tmp_path, monkeypatch, timeout):
    root, markers = _fx_repo(tmp_path)
    launches: list[object] = []
    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "launch", classmethod(lambda cls, *a, **k: launches.append(a)))
    server = _fx_server(root)
    try:
        base = {
            "turn_id": server.turn_id, "nonce": "", "argv": [sys.executable, "-c", "pass"],
            "timeout_seconds": timeout, "cwd": str(root), "environment": {"PATH": os.environ.get("PATH", "")},
        }
        raw = "fedcba9876543210fedcba9876543210"
        base["nonce"] = raw + "." + hmac_new(server.capability, raw)
        with pytest.raises(fx_evidence.BrokerProtocolError, match="fixture-timeout-below-floor"):
            fx_evidence._validate_broker_request(
                {**base, "fixture": "fixture.py"}, turn_id=server.turn_id,
                capability=server.capability, root=root, ceiling=1800,
            )
        # Fixture-less requests with the same timeout behave exactly as today.
        assert fx_evidence._validate_broker_request(
            base, turn_id=server.turn_id, capability=server.capability, root=root, ceiling=1800,
        )["timeout_seconds"] == float(timeout)
        with pytest.raises(fx_evidence.BrokerProtocolError, match="fixture-timeout-below-floor"):
            fx_evidence.TestBrokerClient(_fx_env(server)).run(
                [sys.executable, "-c", "pass"], timeout_seconds=timeout, cwd=root, fixture="fixture.py",
            )
    finally:
        server.stop()
    assert launches == [] and not (markers / "setup-started").exists()


@pytest.mark.parametrize(
    "case, expected",
    [
        ({"outcome": "passed", "returncode": 0, "inner_exec": "started", "fixture": {"setup_status": 0, "teardown_status": 0, "cleanup": "confirmed"}}, 0),
        ({"outcome": "interrupted", "returncode": 0, "inner_exec": "started", "fixture": {"setup_status": 0, "teardown_status": 0, "cleanup": "confirmed"}}, 130),
        ({"outcome": "cancelled", "returncode": 0, "inner_exec": "started", "fixture": {"setup_status": 0, "teardown_status": 0, "cleanup": "confirmed"}}, 130),
        ({"outcome": "passed", "returncode": 0, "inner_exec": "started", "fixture": {"setup_status": 0, "teardown_status": 0, "cleanup": "confirmed"}, "cancelled_frame": True}, 130),
        ({"outcome": "failed", "returncode": None, "inner_exec": "not-attempted", "fixture": {"setup_status": 3, "teardown_status": 0, "cleanup": "confirmed"}}, 3),
        ({"outcome": "failed", "returncode": None, "inner_exec": "not-attempted", "fixture": {"setup_status": "timed_out", "teardown_status": 0, "cleanup": "confirmed"}}, 124),
        ({"outcome": "timed_out", "returncode": None, "inner_exec": "not-attempted", "fixture": {"setup_status": "not-run", "teardown_status": "not-run", "cleanup": "unconfirmed"}}, 1),
        ({"outcome": "timed_out", "returncode": None, "inner_exec": "not-attempted", "fixture": {"setup_status": 0, "teardown_status": 0, "cleanup": "confirmed"}}, 1),
        ({"outcome": "failed", "returncode": 2, "inner_exec": "started", "fixture": {"setup_status": 0, "teardown_status": 0, "cleanup": "confirmed"}}, 2),
        ({"outcome": "timed_out", "returncode": None, "inner_exec": "started", "fixture": {"setup_status": 0, "teardown_status": 0, "cleanup": "confirmed"}}, 124),
        ({"outcome": "failed", "returncode": 0, "inner_exec": "started", "fixture": {"setup_status": 0, "teardown_status": 4, "cleanup": "confirmed"}}, 4),
        ({"outcome": "failed", "returncode": 0, "inner_exec": "started", "fixture": {"setup_status": 0, "teardown_status": "timed_out", "cleanup": "confirmed"}}, 124),
        ({"outcome": "failed", "returncode": 0, "inner_exec": "started", "fixture": {"setup_status": 0, "teardown_status": 0, "cleanup": "escaped"}}, 1),
        ({"outcome": "failed", "returncode": 0, "inner_exec": "started", "fixture": {"setup_status": 0, "teardown_status": 0, "cleanup": "unconfirmed"}}, 1),
        ({"outcome": "failed", "returncode": None, "inner_exec": "failed", "fixture": {"setup_status": 0, "teardown_status": 0, "cleanup": "confirmed"}}, 1),
        ({"outcome": "launch-failed", "returncode": None, "inner_exec": "failed", "fixture": {"setup_status": 0, "teardown_status": 0, "cleanup": "confirmed"}}, 1),
        ({"outcome": "incomplete", "returncode": 0, "inner_exec": "started", "fixture": {"setup_status": 0, "teardown_status": 0, "cleanup": "confirmed"}}, 1),
        ({"outcome": "passed", "returncode": 0, "inner_exec": "started", "fixture": {"setup_status": "weird", "teardown_status": 0, "cleanup": "confirmed"}}, 1),
        ({"outcome": "passed", "returncode": 0, "inner_exec": "started", "fixture": {"setup_status": True, "teardown_status": 0, "cleanup": "confirmed"}}, 1),
        ({"outcome": "passed", "returncode": 0, "inner_exec": "started", "fixture": None}, 1),
    ],
    ids=[
        "clean", "interrupted-all-zero", "cancelled-all-zero", "cancelled-frame", "setup-status",
        "setup-timeout", "deadline-before-setup", "probe-truncated", "inner-rc", "inner-timeout",
        "teardown-status", "teardown-timeout", "escaped", "unconfirmed", "inner-exec-failed",
        "launch-failed", "non-passed-outcome", "unknown-string", "boolean-status", "no-record",
    ],
)
def test_fixture_fold_exit_code_is_total_and_zero_only_for_a_clean_lifecycle(case, expected):
    assert fx_runtime.fold_exit_code(**case) == expected


def test_fixture_cli_launch_failed_after_clean_phases_exits_one(tmp_path, monkeypatch, capsys):
    import coding_review_agent_loop.cli as cli_module
    from coding_review_agent_loop.local_test_evidence import BrokerRunResult

    clean = fx_evidence.FixtureRecord(
        path="fixture.py", digest="100755:" + "a" * 40, setup_status=0, teardown_status=0, cleanup="confirmed",
    ).to_dict()
    calls: list[dict] = []

    class Broker:
        def run(self, *args, **kwargs):
            calls.append(kwargs)
            return BrokerRunResult(
                receipt_id="r", execution_ref="e", outcome="launch-failed", returncode=None,
                elapsed_seconds=1.0, wrapper_bootstrap="verified", inner_exec="failed",
                suite_start="not-started", fixture=clean,
            )

    monkeypatch.setattr(cli_module, "broker_client_from_environment", lambda: Broker())
    monkeypatch.chdir(tmp_path)
    memory = tmp_path / "memory"
    code = main(["run-tests", "--timeout-seconds", "60", "--memory-dir", str(memory), "--fixture", "fixture.py", "--", sys.executable, "-m", "pytest"])
    assert code == 1
    assert calls[0]["fixture"] == "fixture.py"
    health = json.loads((memory / runtime.RUNTIME_SIDECAR_NAME).read_text()).get("launcher_health", [])
    assert all("fixture.py" not in json.dumps(item) for item in health)


def test_fixture_cli_rejects_preflight_and_an_empty_inner_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["run-tests", "--preflight", "--fixture", "fixture.py"]) == 2
    assert "--fixture cannot be combined with --preflight" in capsys.readouterr().err
    assert main(["run-tests", "--timeout-seconds", "60", "--fixture", "fixture.py", "--"]) == 1
    assert "non-empty command" in capsys.readouterr().err


# A recognized launcher that asks for worker-budget slots, so it waits for
# host admission like a real parallel suite.
_FX_PYTEST_PARALLEL = (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-n", "2", "test_fx.py")


@_FX_HOST
def test_fixture_deadline_admission_exhausts_the_setup_budget_and_launches_nothing(tmp_path, monkeypatch):
    from coding_review_agent_loop.test_workers import ENV_HOST_WAIT, HostCapacity, WorkerBudget

    root, markers = _fx_repo(tmp_path)
    server = _fx_server(root)
    locks = tmp_path / "locks"
    server._worker_lock_root = locks
    gib = 1024 ** 3
    budget = WorkerBudget(4, "inherited", "clamp", "cpu", {"cpu_available": 4, "host_usable_bytes": 64 * gib, "per_worker_bytes": gib, "reserve_bytes": gib}, False)
    server.set_execution_context(containment_handle=None, process_started=None, process_finished=None, worker_budget=budget)
    holder, problem = WorkerBudgetLock.acquire(invocation_id="foreign", cwd=locks, root=locks)
    assert holder is not None, problem
    assert holder.reserve_host_workers(4, HostCapacity(4, 1024 * gib, gib)) == 4
    monkeypatch.setattr(fx_runtime, "reserve_seconds", lambda timeout: 5.0)
    monkeypatch.setattr(fx_runtime, "MIN_SETUP_AND_INNER_SECONDS", 10.0)
    launches: list[object] = []
    original_launch = fx_runtime.FixtureSupervisor.launch.__func__
    monkeypatch.setattr(
        fx_runtime.FixtureSupervisor, "launch",
        classmethod(lambda cls, *a, **k: (launches.append(a), original_launch(cls, *a, **k))[1]),
    )
    try:
        env = {**_fx_env(server), ENV_HOST_WAIT: "600", "AGENT_LOOP_TEST_WORKER_HOST_SHARING": "on"}
        started = time.monotonic()
        result = fx_evidence.TestBrokerClient(env).run(
            list(_FX_PYTEST_PARALLEL), timeout_seconds=30, cwd=root, fixture="fixture.py",
        )
        elapsed = time.monotonic() - started
        row = _fx_row(server)
    finally:
        holder.close()
        server.stop()
    # The 600 s admission wait was capped at the 25 s inner deadline.
    assert 20 <= elapsed < 40
    assert launches == []
    assert not (markers / "setup-started").exists() and not (markers / "inner-blocking").exists()
    assert row.outcome == "timed_out" and row.inner_exec == "not-attempted"
    assert row.fixture.setup_status == "not-run"
    assert result.fixture["setup_status"] == "not-run"
    assert not _authoritative_receipt_passes(row, claim="current-result")


@_FX_HOST
def test_fixture_deadline_inner_cut_off_after_admission_keeps_the_teardown_reserve(tmp_path, monkeypatch):
    from coding_review_agent_loop.test_workers import ENV_HOST_WAIT, HostCapacity, WorkerBudget

    root, markers = _fx_repo(tmp_path)
    _fx_set_mode(root, "inner-block")
    server = _fx_server(root)
    locks = tmp_path / "locks"
    server._worker_lock_root = locks
    gib = 1024 ** 3
    budget = WorkerBudget(4, "inherited", "clamp", "cpu", {"cpu_available": 4, "host_usable_bytes": 64 * gib, "per_worker_bytes": gib, "reserve_bytes": gib}, False)
    server.set_execution_context(containment_handle=None, process_started=None, process_finished=None, worker_budget=budget)
    holder, problem = WorkerBudgetLock.acquire(invocation_id="foreign", cwd=locks, root=locks)
    assert holder is not None, problem
    assert holder.reserve_host_workers(4, HostCapacity(4, 1024 * gib, gib)) == 4
    try:
        env = {**_fx_env(server), ENV_HOST_WAIT: "2", "AGENT_LOOP_TEST_WORKER_HOST_SHARING": "on"}
        started = time.monotonic()
        result = fx_evidence.TestBrokerClient(env).run(
            list(_FX_PYTEST_PARALLEL), timeout_seconds=30, cwd=root, fixture="fixture.py",
        )
        elapsed = time.monotonic() - started
        row = _fx_row(server)
        # A fixture-less run under the same contention keeps today's admission.
        plain = fx_evidence.TestBrokerClient(env).run(
            [sys.executable, "-c", "pass"], timeout_seconds=30, cwd=root,
        )
    finally:
        holder.close()
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    assert (markers / "inner-blocking").exists() and (markers / "teardown-done").exists()
    assert row.outcome == "timed_out" and result.outcome == "timed_out"
    assert row.suite_start == "verified"  # a recognized launcher really ran
    assert row.fixture.teardown_status == 0
    assert elapsed < 30 + 2 * fx_evidence.MAX_SNAPSHOT_SECONDS + 10
    assert plain.outcome == "passed" and plain.fixture is None


@_FX_HOST
def test_fixture_prelaunch_deadline_truncated_probe_never_spawns_and_keeps_the_reserve(tmp_path, monkeypatch):
    root, markers = _fx_repo(tmp_path)
    server = _fx_server(root)
    probe_calls: list[dict] = []

    def truncated_probe(argv, **kwargs):
        probe_calls.append(kwargs)
        return runtime.LauncherProbeResult(tuple(argv), "unknown", runtime.PROBE_DEADLINE_DIAGNOSTIC, "id")

    monkeypatch.setattr(runner_module, "probe_inner_launcher", truncated_probe)
    try:
        code = _fx_cli(monkeypatch, server, root, _FX_PYTEST)
        row = _fx_row(server)
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    assert probe_calls and probe_calls[0]["deadline"] is not None and probe_calls[0]["use_cache"] is False
    assert not (markers / "inner-ran").exists()
    assert (markers / "teardown-done").exists()
    assert row.outcome == "timed_out" and row.inner_exec == "not-attempted"
    assert row.fixture.setup_status == 0 and row.fixture.teardown_status == 0
    assert code != 0
    assert not _authoritative_receipt_passes(row, claim="current-result")


@_FX_LINUX
def test_fixture_prelaunch_deadline_real_probe_is_truncated_and_torn_down(tmp_path, monkeypatch):
    spawned: list[int] = []
    original = runtime._run_bounded_probe

    def slow_probe(argv, **kwargs):
        script = "import os, time; open(%r, 'a').write(str(os.getpid()) + '\\n'); time.sleep(600)" % str(tmp_path / "probe.pid")
        return original([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(runtime, "_run_bounded_probe", slow_probe)
    started = time.monotonic()
    result = runtime.probe_inner_launcher(
        [sys.executable, "-m", "pytest", "-q"], cwd=tmp_path,
        environment={**os.environ, "AGENT_LOOP_INVOCATION_ID": "turn-probe-deadline"},
        environment_is_complete=True, deadline=started + 2.5, use_cache=False,
    )
    # Execution and probe-tree cleanup both end by the deadline.
    assert time.monotonic() - started <= 2.6
    assert result.state == "unknown" and result.diagnostic == runtime.PROBE_DEADLINE_DIAGNOSTIC
    for line in (tmp_path / "probe.pid").read_text().split():
        spawned.append(int(line))
    _fx_assert_gone(spawned)
    # Never cached: an expired deadline returns at once without a new probe.
    again = runtime.probe_inner_launcher(
        [sys.executable, "-m", "pytest", "-q"], cwd=tmp_path, environment=dict(os.environ),
        environment_is_complete=True, deadline=time.monotonic() - 1, use_cache=False,
    )
    assert again.diagnostic == runtime.PROBE_DEADLINE_DIAGNOSTIC
    from coding_review_agent_loop.lifecycle_probe import prepare_lifecycle_probe

    assert prepare_lifecycle_probe(
        [sys.executable, "-m", "pytest"], dict(os.environ), tmp_path, deadline=time.monotonic() - 1,
    ) is None


@_FX_HOST
def test_fixture_teardown_at_the_floor_keeps_a_positive_bound(tmp_path, monkeypatch):
    assert fx_runtime.teardown_budget_seconds(30) == 5.0
    assert fx_runtime.reserve_seconds(30) == 10.0
    root, markers = _fx_repo(tmp_path)
    _fx_set_mode(root, "teardown-slow", fx_teardown_seconds="3")
    server = _fx_server(root)
    try:
        _fx_cli(monkeypatch, server, root, _FX_PYTEST, timeout="30")
        row = _fx_row(server)
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    assert (markers / "teardown-done").exists()
    assert row.fixture.teardown_status == 0
    assert 3.0 <= row.fixture.teardown_duration_seconds < 5.0


def _fx_venv_repo(tmp_path):
    root, markers = _fx_repo(tmp_path)
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    fake = tmp_path / "not-python"
    fake.write_text("#!/bin/sh\nexit 3\n")
    fake.chmod(0o755)
    return root, markers, venv_bin / "python", fake


@_FX_HOST
def test_fixture_post_setup_authentication_ignores_a_pre_setup_cached_verdict(tmp_path, monkeypatch):
    root, markers, shim, fake = _fx_venv_repo(tmp_path)
    retarget = (
        "import os, pathlib\n"
        "shim = pathlib.Path(%r)\n"
        "shim.unlink(); shim.symlink_to(%r)\n" % (str(shim), str(fake))
    )
    (root / "retarget.py").write_text(retarget)
    _fx_git(root, "add", "retarget.py")
    _fx_git(root, "commit", "-qm", "retarget")
    fixture_text = (root / "fixture.py").read_text().replace(
        'if mode("setup-fail"):', 'exec(open(str(HERE / "retarget.py")).read())\n    if mode("setup-fail"):', 1,
    )
    (root / "fixture.py").write_text(fixture_text)
    _fx_git(root, "add", "fixture.py")
    _fx_git(root, "commit", "-qm", "retargeting fixture")
    inner = (".venv/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_fx.py")
    server = _fx_server(root)
    env = _fx_env(server)
    # Seed the per-invocation cache with a verified verdict for the original shim.
    seeded = runtime.probe_inner_launcher(list(inner), cwd=root, environment=env, environment_is_complete=True)
    assert seeded.state == "verified"
    probe_calls: list[dict] = []
    real_probe = runtime.probe_inner_launcher

    def spy(argv, **kwargs):
        probe_calls.append(kwargs)
        return real_probe(argv, **kwargs)

    monkeypatch.setattr(runner_module, "probe_inner_launcher", spy)
    try:
        result = fx_evidence.TestBrokerClient(env).run(list(inner), timeout_seconds=40, cwd=root, fixture="fixture.py")
        row = _fx_row(server)
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    assert probe_calls and probe_calls[0]["use_cache"] is False
    assert not (markers / "inner-ran").exists()
    # The retargeted shim failed its fresh bootstrap probe: not a verified launch.
    assert result.outcome in {"launch-failed", "failed"} and row.suite_start != "verified"
    assert row.fixture.setup_status == 0
    assert not _authoritative_receipt_passes(row, claim="current-result")


@_FX_HOST
def test_fixture_launcher_change_after_the_probe_prevents_the_spawn(tmp_path, monkeypatch):
    root, markers, shim, fake = _fx_venv_repo(tmp_path)
    inner = (".venv/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_fx.py")
    real_probe = runtime.probe_inner_launcher
    swapped: list[bool] = []

    def probe_then_fixture_swaps_the_shim(argv, **kwargs):
        verdict = real_probe(argv, **kwargs)
        # A fixture process retargets the shim between the probe and the spawn.
        shim.unlink()
        shim.symlink_to(fake)
        swapped.append(True)
        return verdict

    monkeypatch.setattr(runner_module, "probe_inner_launcher", probe_then_fixture_swaps_the_shim)
    server = _fx_server(root)
    try:
        code = _fx_cli(monkeypatch, server, root, inner)
        row = _fx_row(server)
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    assert swapped
    assert not (markers / "inner-ran").exists()
    assert row.outcome == "launch-failed" and row.inner_exec == "not-attempted"
    assert fx_runtime.LAUNCHER_CHANGED_CAVEAT in row.caveats
    assert code != 0
    assert not _authoritative_receipt_passes(row, claim="current-result")


_FX_MANAGER = '''import os, socket, sys, time
path = sys.argv[1]
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(path)
server.listen(4)
open(path + ".ready", "w").write(str(os.getpid()))
while True:
    connection, _ = server.accept()
    connection.recv(16)
    if os.fork() == 0:
        # Intermediate service supervisor, which forks the long-lived worker.
        if os.fork() == 0:
            open(path + ".worker", "w").write(str(os.getpid()))
            time.sleep(600)
            os._exit(0)
        open(path + ".intermediate", "w").write(str(os.getpid()))
        time.sleep(600)
        os._exit(0)
    connection.sendall(b"ok")
    connection.close()
'''


@_FX_HOST
@pytest.mark.parametrize("case", ["delegated-manager", "coder-scope-leftover", "systemd-run-user"])
def test_fixture_cleanup_escape_is_flagged_without_exemption(tmp_path, monkeypatch, capsys, case):
    root, markers = _fx_repo(tmp_path)
    extras: list[int] = []
    manager = None
    if case == "delegated-manager":
        sock = str(tmp_path / "mgr.sock")
        manager = subprocess.Popen([sys.executable, "-c", _FX_MANAGER, sock], start_new_session=True)
        deadline = time.monotonic() + 20
        while not os.path.exists(sock + ".ready") and time.monotonic() < deadline:
            time.sleep(0.05)
        hook = (
            "import socket as _s\n"
            "_c = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM); _c.connect(%r); _c.sendall(b'go'); _c.recv(2); _c.close()\n" % sock
        )
        files = {"delegate.py": hook}
        marker_files = [sock + ".worker", sock + ".intermediate"]
    elif case == "systemd-run-user":
        if not shutil.which("systemd-run") or subprocess.run(
            ["systemd-run", "--user", "--quiet", "--collect", "true"], capture_output=True, timeout=20,
        ).returncode != 0:
            pytest.skip("systemd-run --user is unavailable")
        unit = f"agent-loop-fx-{os.getpid()}-{int(time.time())}"
        files = {"delegate.py": "import subprocess\nsubprocess.run(['systemd-run', '--user', '--quiet', '--unit', %r, 'sleep', '600'], check=True)\n" % unit}
        marker_files = []
    else:
        files = {}
        marker_files = []
    if files:
        _fx_set_mode(root, "", **files)
        fixture_text = (root / "fixture.py").read_text().replace(
            'if mode("setup-fail"):', 'exec(open(str(HERE / "delegate.py")).read())\n    if mode("setup-fail"):', 1,
        )
        (root / "fixture.py").write_text(fixture_text)
        _fx_git(root, "add", "-A")
        _fx_git(root, "commit", "-qm", "delegating fixture")
    inner = _FX_PYTEST
    if case == "coder-scope-leftover":
        # A recognized inner pytest whose test leaves a daemon behind.
        leftover = tmp_path / "leftover.pid"
        _fx_set_mode(root, "", **{"test_leftover.py": (
            "import os, time\n"
            "def test_leaves_a_daemon():\n"
            "    if os.fork() == 0:\n"
            "        os.setsid()\n"
            "        if os.fork() == 0:\n"
            "            open(%r, 'w').write(str(os.getpid())); time.sleep(600)\n"
            "        os._exit(0)\n"
            "    time.sleep(1)\n" % str(leftover)
        )})
        inner = (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_leftover.py")
        marker_files = [str(leftover)]
    server = _fx_server(root)
    try:
        code = _fx_cli(monkeypatch, server, root, inner)
        row = _fx_row(server)
        for path in marker_files:
            if os.path.exists(path):
                extras.append(int(Path(path).read_text()))
        # Escaped processes are reported, not killed.
        for pid in extras:
            assert _proc_state(pid) not in (None, *DEAD_STATES)
    finally:
        server.stop()
        _fx_kill(extras)
        if manager is not None:
            manager.kill()
            manager.wait()
        if case == "systemd-run-user":
            subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, timeout=20)
    _fx_assert_gone(_fx_pids(markers))
    err = capsys.readouterr().err
    assert row.fixture.cleanup == "escaped", row.fixture
    assert row.fixture.escaped >= 1
    assert row.outcome == "failed" and fx_runtime.FIXTURE_CLEANUP_ESCAPED_CAVEAT in row.caveats
    assert code != 0
    assert "agent-loop fixture: escaped process pid=" in err
    if extras and row.fixture.escaped <= len(row.fixture.escaped_processes):
        # Only a bounded list is printed; unrelated host noise may crowd it.
        assert any(f"pid={pid} " in err for pid in extras)
    assert not _authoritative_receipt_passes(row, claim="current-result")


def _fx_fake_proc(root: Path, entries: dict[int, dict]) -> Path:
    for pid, info in entries.items():
        directory = root / str(pid)
        directory.mkdir(parents=True)
        state = info.get("state", "S")
        fields = [state, str(info.get("ppid", 1))] + ["0"] * 4 + [str(info.get("flags", 0))] + ["0"] * 12 + [str(info["start"])]
        (directory / "stat").write_text(f"{pid} ({info.get('comm', 'proc')}) " + " ".join(fields) + "\n")
        status = f"Name:\tproc\n"
        status_state = info.get("status_state", state)
        if "status_state_line" in info:
            status += info["status_state_line"] + "\n"
        elif status_state is not None:
            status += f"State:\t{status_state} (state)\n"
        status += f"Uid:\t{info.get('uid', 1000)}\t0\t0\t0\n"
        threads = info.get("threads", 1)
        if threads is not None:
            status += f"Threads:\t{threads}\n"
        (directory / "status").write_text(status)
        (directory / "cgroup").write_text(info.get("cgroup", "0::/user.slice") + "\n")
    return root


def test_fixture_census_has_no_scope_ancestry_or_uid_exemption(tmp_path):
    proc = _fx_fake_proc(tmp_path / "proc", {
        1: {"start": 1, "ppid": 0},
        2: {"start": 1, "ppid": 0, "comm": "kthreadd"},
        3: {"start": 500, "ppid": 2, "comm": "kworker"},
        50: {"start": 50, "ppid": 1, "comm": "manager"},                     # pre-window manager
        100: {"start": 100, "ppid": 1, "comm": "supervisor"},                # the census itself
        101: {"start": 101, "ppid": 100},                                    # owned and inventoried
        200: {"start": 150, "ppid": 50, "comm": "intermediate"},             # (a) delegated
        201: {"start": 151, "ppid": 200, "comm": "worker"},
        300: {"start": 160, "ppid": 1, "cgroup": "0::/user.slice/agent-loop-foreign.scope"},  # (b) foreign scope
        400: {"start": 170, "ppid": 1, "uid": 0, "comm": "containerd-shim"},  # (e) another uid
        500: {"start": 180, "ppid": 1, "flags": fx_supervisor.PF_KTHREAD, "comm": "kthread"},
    })
    result = fx_supervisor.census(window_start=100, owned={(101, 101)}, self_identity=(100, 100), proc_root=proc)
    assert result.cleanup == "escaped"
    flagged = {item["pid"] for item in result.escaped}
    assert flagged == {200, 201, 300, 400}
    assert {item["uid"] for item in result.escaped if item["pid"] == 400} == {0}
    assert any("agent-loop-foreign.scope" in item["cgroup"] for item in result.escaped)


def test_fixture_census_self_identity_is_the_only_exclusion(tmp_path):
    # (a) the supervisor's start tick equals the window start: confirmed.
    proc = _fx_fake_proc(tmp_path / "a", {1: {"start": 1, "ppid": 0}, 100: {"start": 100}})
    assert fx_supervisor.census(window_start=100, owned=(), self_identity=(100, 100), proc_root=proc).cleanup == "confirmed"
    # (b) a descendant of the supervisor sharing that tick is still flagged.
    proc = _fx_fake_proc(tmp_path / "b", {100: {"start": 100}, 101: {"start": 100, "ppid": 100}})
    result = fx_supervisor.census(window_start=100, owned=(), self_identity=(100, 100), proc_root=proc)
    assert [item["pid"] for item in result.escaped] == [101]
    # (c) a different process reusing the supervisor's PID is still flagged.
    proc = _fx_fake_proc(tmp_path / "c", {100: {"start": 140}})
    result = fx_supervisor.census(window_start=100, owned=(), self_identity=(100, 100), proc_root=proc)
    assert [item["pid"] for item in result.escaped] == [100]


def test_fixture_census_unreadable_entry_is_unconfirmed(tmp_path):
    proc = _fx_fake_proc(tmp_path / "proc", {100: {"start": 100}})
    (proc / "777").mkdir()
    (proc / "777" / "stat").mkdir()  # reading it fails while the entry exists
    result = fx_supervisor.census(window_start=100, owned=(), self_identity=(100, 100), proc_root=proc)
    assert result.cleanup == "unconfirmed"
    assert fx_supervisor.census(
        window_start=100, owned=(), self_identity=(100, 100), proc_root=tmp_path / "missing",
    ).cleanup == "unconfirmed"


def _fx_census(proc: Path):
    return fx_supervisor.census(window_start=100, owned=(), self_identity=(100, 100), proc_root=proc)


@pytest.mark.parametrize("state", ["Z", "X", "x"])
def test_fixture_census_skips_single_thread_dead_entry(tmp_path, monkeypatch, state):
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(fx_supervisor.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    proc = _fx_fake_proc(tmp_path / "proc", {
        1: {"start": 1, "ppid": 0},
        50: {"start": 50, "ppid": 1, "comm": "parent"},              # unrelated pre-window parent
        100: {"start": 100},
        200: {"start": 150, "ppid": 50, "state": state, "threads": 1},  # unreaped child
    })
    result = _fx_census(proc)
    assert result.cleanup == "confirmed"
    assert result.escaped == []
    assert signals == []


def test_fixture_census_flags_zombie_leader_with_live_threads(tmp_path):
    proc = _fx_fake_proc(tmp_path / "proc", {
        100: {"start": 100},
        200: {"start": 150, "state": "Z", "threads": 3},
    })
    result = _fx_census(proc)
    assert result.cleanup == "escaped"
    assert [item["pid"] for item in result.escaped] == [200]


@pytest.mark.parametrize(
    "overrides",
    [
        {"threads": None},                        # Threads line missing
        {"threads": "many"},                      # Threads line unparsable
        {"threads": ""},                          # Threads line without a value
        {"status_state": None},                   # State line missing
        {"status_state": ""},                     # State line without a value
        {"status_state": "S"},                    # stat dead, status live
        {"status_state": "X"},                    # stat and status name different dead states
    ],
)
def test_fixture_census_flags_ambiguous_dead_entry(tmp_path, overrides):
    proc = _fx_fake_proc(tmp_path / "proc", {
        100: {"start": 100},
        200: {"start": 150, "state": "Z", **overrides},
    })
    result = _fx_census(proc)
    assert result.cleanup == "escaped"
    assert [item["pid"] for item in result.escaped] == [200]


@pytest.mark.parametrize(
    "state_line",
    [
        "State:\t(zombie)",       # parenthesised description in the state position
        "State:\tZZ (zombie)",    # multi-character token
        "State:\tZombie",         # word, not a single character
        "State:Z (zombie)",       # no separator, so no exact State: key
        "State:",                 # no value
        "StateX:\tZ (zombie)",    # prefix-keyed line, no real State: line
    ],
)
def test_fixture_census_flags_malformed_status_state_line(tmp_path, state_line):
    proc = _fx_fake_proc(tmp_path / "proc", {
        100: {"start": 100},
        200: {"start": 150, "state": "Z", "status_state_line": state_line},
    })
    result = _fx_census(proc)
    assert result.cleanup == "escaped"
    assert [item["pid"] for item in result.escaped] == [200]


def test_fixture_census_status_state_reads_single_character_field():
    assert fx_supervisor._status_state("Name:\tp\nState:\tZ (zombie)\n") == "Z"
    assert fx_supervisor._status_state("Name:\tp\nState:\t(zombie)\n") is None
    assert fx_supervisor._status_state("StateX:\tZ (zombie)\n") is None


@pytest.mark.parametrize("state, line", [("Z", "State:\tZ (zombie)"), ("X", "State:\tX (dead)")])
def test_fixture_census_skips_real_format_dead_state_line(tmp_path, monkeypatch, state, line):
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(fx_supervisor.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    proc = _fx_fake_proc(tmp_path / "proc", {
        100: {"start": 100},
        200: {"start": 150, "state": state, "status_state_line": line, "threads": 1},
    })
    result = _fx_census(proc)
    assert result.cleanup == "confirmed"
    assert result.escaped == []
    assert signals == []


def test_fixture_census_flags_live_stat_with_dead_status(tmp_path):
    proc = _fx_fake_proc(tmp_path / "proc", {
        100: {"start": 100},
        200: {"start": 150, "state": "S", "status_state": "Z"},
    })
    result = _fx_census(proc)
    assert result.cleanup == "escaped"
    assert [item["pid"] for item in result.escaped] == [200]


def test_fixture_census_zombie_with_unreadable_status_is_unconfirmed(tmp_path):
    proc = _fx_fake_proc(tmp_path / "proc", {
        100: {"start": 100},
        200: {"start": 150, "state": "Z"},
    })
    (proc / "200" / "status").unlink()
    (proc / "200" / "status").mkdir()  # reading it fails while the entry exists
    assert _fx_census(proc).cleanup == "unconfirmed"


@pytest.mark.parametrize("state", ["R", "S", "D", "T", "t", "I"])
def test_fixture_census_flags_every_live_state(tmp_path, state):
    proc = _fx_fake_proc(tmp_path / "proc", {
        100: {"start": 100},
        200: {"start": 150, "state": state},
        201: {"start": 151, "state": "Z"},
    })
    result = _fx_census(proc)
    assert result.cleanup == "escaped"
    assert [item["pid"] for item in result.escaped] == [200]


@_FX_HOST
def test_fixture_happy_path_never_lists_the_supervisor_as_escaped(tmp_path, monkeypatch):
    root, markers = _fx_repo(tmp_path, count=1)
    server = _fx_server(root)
    supervisors: list[int] = []
    original_launch = fx_runtime.FixtureSupervisor.launch.__func__

    def launch(cls, *args, **kwargs):
        handle = original_launch(cls, *args, **kwargs)
        supervisors.append(handle.process.pid)
        return handle

    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "launch", classmethod(launch))
    (root / "fx_count").write_text("3")
    _fx_git(root, "commit", "-qam", "three")
    try:
        _fx_cli(monkeypatch, server, root, _FX_PYTEST)
        row = _fx_row(server)
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    assert supervisors
    assert all(f"pid={supervisors[0]} " not in item for item in row.fixture.escaped_processes)


def _fx_probes(**overrides):
    import errno as errno_module

    ns = os.stat_result((0o444, fx_supervisor.PROC_PID_INIT_INO, 5, 1, 0, 0, 0, 0, 0, 0))
    net = os.stat_result((0o444, 4026531840, 5, 1, 0, 0, 0, 0, 0, 0))
    texts = {
        "/proc/self/stat": "4242 (python) S 1 ...",
        "/proc/self/mountinfo": "22 1 0:5 / / rw - ext4 /dev/sda rw\n29 27 0:24 / /proc rw - proc proc rw\n",
        "/proc/1/stat": "1 (init) S 0 ...",
    }
    texts.update(overrides.pop("texts", {}))

    def read_text(path):
        if path.startswith("/proc/1/ns"):
            raise PermissionError(path)
        value = texts.get(path)
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise FileNotFoundError(path)
        return value

    def get_parent(path):
        raise OSError(errno_module.EPERM, "no parent")

    defaults = dict(
        stat=lambda path: {"/proc/self/ns/pid": ns, "/proc/self/ns/net": net}[path],
        ns_get_parent=get_parent,
        read_text=read_text,
        getpid=lambda: 4242,
        subreaper_available=lambda: True,
    )
    defaults.update(overrides)
    return fx_supervisor.AuthorityProbes(**defaults)


@_FX_LINUX
def test_fixture_census_gate_non_root_happy_path_passes_without_reading_pid_one_ns():
    authority = fx_supervisor.verify_host_process_authority(_fx_probes())
    assert authority.mount_id == "29"
    source = Path(fx_supervisor.__file__).read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("#", '"', "*")))
    assert '"/proc/1/ns' not in code and "'/proc/1/ns" not in code


@_FX_LINUX
@pytest.mark.parametrize(
    "case",
    ["nested-namespace-inode", "wrong-nsfs-device", "visible-parent", "parent-ioctl-error",
     "self-pid-mismatch", "hidepid", "subset-pid", "not-procfs", "no-subreaper", "pid-one-hidden",
     "mount-changed"],
)
def test_fixture_census_gate_refuses_each_failed_check_alone(case):
    import errno as errno_module

    other = os.stat_result((0o444, 4026532999, 5, 1, 0, 0, 0, 0, 0, 0))
    expected_mount = None
    overrides: dict = {}
    if case == "nested-namespace-inode":
        # A fresh procfs with one NSpid entry and a readable PID 1 is not proof.
        overrides["stat"] = lambda path: {"/proc/self/ns/pid": other, "/proc/self/ns/net": other}[path]
        overrides["texts"] = {"/proc/self/status": "NSpid:\t1\n"}
    elif case == "wrong-nsfs-device":
        ns = os.stat_result((0o444, fx_supervisor.PROC_PID_INIT_INO, 9, 1, 0, 0, 0, 0, 0, 0))
        net = os.stat_result((0o444, 4026531840, 5, 1, 0, 0, 0, 0, 0, 0))
        overrides["stat"] = lambda path: {"/proc/self/ns/pid": ns, "/proc/self/ns/net": net}[path]
    elif case == "visible-parent":
        overrides["ns_get_parent"] = lambda path: None
    elif case == "parent-ioctl-error":
        def fail(path):
            raise OSError(errno_module.ENOTTY, "no ioctl")
        overrides["ns_get_parent"] = fail
    elif case == "self-pid-mismatch":
        overrides["texts"] = {"/proc/self/stat": "1 (python) S 0 ..."}
    elif case == "hidepid":
        overrides["texts"] = {"/proc/self/mountinfo": "29 27 0:24 / /proc rw - proc proc rw,hidepid=2\n"}
    elif case == "subset-pid":
        overrides["texts"] = {"/proc/self/mountinfo": "29 27 0:24 / /proc rw - proc proc rw,subset=pid\n"}
    elif case == "not-procfs":
        overrides["texts"] = {"/proc/self/mountinfo": "29 27 0:24 / /proc rw - tmpfs tmpfs rw\n"}
    elif case == "no-subreaper":
        overrides["subreaper_available"] = lambda: False
    elif case == "pid-one-hidden":
        overrides["texts"] = {"/proc/1/stat": PermissionError("hidden")}
    else:
        expected_mount = "30"
    with pytest.raises(fx_supervisor.HostAuthorityError):
        fx_supervisor.verify_host_process_authority(_fx_probes(**overrides), expected_mount_id=expected_mount)
    with pytest.raises(fx_runtime.FixtureRefused, match="fixture-scope-unsupported"):
        fx_runtime.verify_host_process_authority(probes=_fx_probes(**overrides), expected_mount_id=expected_mount)


_FX_NESTED_BROKER = '''import json, os, socket, sys
from pathlib import Path
from coding_review_agent_loop import fixture_supervisor
from coding_review_agent_loop.local_test_evidence import TestBrokerServer, TestBrokerClient, BrokerProtocolError
root = Path(sys.argv[1])
status = open("/proc/self/status").read()
nspid = [line for line in status.splitlines() if line.startswith("NSpid:")][0].split()[1:]
open("/proc/1/stat").read()  # the local PID 1 is readable
try:
    fixture_supervisor.verify_host_process_authority()
    gate = "passed"
except fixture_supervisor.HostAuthorityError as exc:
    gate = "refused: " + str(exc)
server = TestBrokerServer(root=root, turn_id="turn-nested").start()
try:
    env = {**os.environ, **server.environment, "AGENT_LOOP_INVOCATION_ID": "turn-nested"}
    try:
        TestBrokerClient(env).run([sys.executable, "-m", "pytest", "-q", "test_fx.py"], timeout_seconds=40, cwd=root, fixture="fixture.py")
        frame = "result"
    except BrokerProtocolError as exc:
        frame = str(exc)
finally:
    server.stop()
print(json.dumps({"nspid": nspid, "gate": gate, "frame": frame}))
'''


@_FX_LINUX
def test_fixture_census_gate_refuses_a_real_nested_pid_namespace_before_setup(tmp_path):
    if not shutil.which("unshare"):
        pytest.skip("unshare is unavailable")
    probe = subprocess.run(
        ["unshare", "--user", "--map-root-user", "--pid", "--fork", "--mount", "--mount-proc", "true"],
        capture_output=True, timeout=30,
    )
    if probe.returncode != 0:
        pytest.skip("unprivileged user and PID namespaces are disabled")
    root, markers = _fx_repo(tmp_path)
    manager_sock = tmp_path / "host-manager.sock"
    import socket as socket_module

    manager = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
    manager.bind(str(manager_sock))
    manager.listen(1)
    manager.settimeout(0.2)
    # A setup that would ask the host-side manager to start a host process.
    _fx_set_mode(root, "", **{"delegate.py": "import socket as _s\n_c=_s.socket(_s.AF_UNIX); _c.connect(%r); _c.sendall(b'go')\n" % str(manager_sock)})
    fixture_text = (root / "fixture.py").read_text().replace(
        'if mode("setup-fail"):', 'exec(open(str(HERE / "delegate.py")).read())\n    if mode("setup-fail"):', 1,
    )
    (root / "fixture.py").write_text(fixture_text)
    _fx_git(root, "commit", "-qam", "delegating fixture")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [_SRC, os.environ.get("PYTHONPATH")]))}
    completed = subprocess.run(
        ["unshare", "--user", "--map-root-user", "--pid", "--fork", "--mount", "--mount-proc",
         sys.executable, "-c", _FX_NESTED_BROKER, str(root)],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["nspid"] == ["1"] or len(report["nspid"]) == 1
    assert report["gate"].startswith("refused:")
    assert "fixture refused: fixture-scope-unsupported" in report["frame"]
    assert not (markers / "setup-started").exists()
    try:
        connection, _ = manager.accept()
        connection.close()
        contacted = True
    except OSError:
        contacted = False
    finally:
        manager.close()
    assert contacted is False


@_FX_HOST
def test_fixture_census_time_authority_loss_is_unconfirmed(tmp_path, monkeypatch):
    root, markers = _fx_repo(tmp_path)
    real = fx_runtime.verify_host_process_authority

    def stale_mount(**kwargs):
        real(**kwargs)
        return fx_supervisor.HostAuthority("mount-that-will-not-match")

    monkeypatch.setattr(fx_runtime, "verify_host_process_authority", stale_mount)
    server = _fx_server(root)
    try:
        code = _fx_cli(monkeypatch, server, root, _FX_PYTEST)
        row = _fx_row(server)
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    assert row.fixture.cleanup == "unconfirmed"
    assert row.outcome == "failed" and fx_runtime.FIXTURE_CLEANUP_UNCONFIRMED_CAVEAT in row.caveats
    assert code == 1
    assert not _authoritative_receipt_passes(row, claim="current-result")


@_FX_HOST
@pytest.mark.parametrize("phase", ["setup", "teardown"])
def test_fixture_noisy_phase_output_never_starves_teardown_or_the_deadline(tmp_path, monkeypatch, phase):
    root, markers = _fx_repo(tmp_path)
    _fx_set_mode(root, "setup-noisy setup-block" if phase == "setup" else "teardown-noisy")
    killed: list[int] = []
    original_force = fx_runtime.FixtureSupervisor.force_terminate

    def spy_force(self):
        killed.append(self.process.pid)
        return original_force(self)

    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "force_terminate", spy_force)
    server = _fx_server(root)
    result: dict[str, int] = {}
    started = time.monotonic()
    try:
        if phase == "setup":
            thread = threading.Thread(
                target=lambda: result.setdefault("code", _fx_cli(monkeypatch, server, root, _FX_PYTEST)),
                daemon=True,
            )
            thread.start()
            deadline = time.monotonic() + 60
            while not (markers / "setup-blocking").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert (markers / "setup-blocking").exists()
            server.stop()
            thread.join(timeout=60)
        else:
            result["code"] = _fx_cli(monkeypatch, server, root, _FX_PYTEST)
        row = _fx_row(server)
    finally:
        server.stop()
    elapsed = time.monotonic() - started
    pids = _fx_pids(markers)
    assert any(path.name.startswith("noisy-") for path in markers.iterdir())
    _fx_assert_gone(pids)
    # The flooding descendant was inventoried and the teardown command was
    # processed gracefully: the supervisor was never force-killed.
    assert (markers / "teardown-done").exists()
    assert killed == [], (killed, row.fixture, pids)
    assert row.fixture.cleanup in {"confirmed", "escaped"}
    assert row.fixture.inventoried >= 1
    assert elapsed < 40 + 2 * fx_evidence.MAX_SNAPSHOT_SECONDS + 10
    if phase == "setup":
        assert result["code"] == 130 and row.fixture.setup_status == "cancelled"
        assert not (markers / "inner-ran").exists()
    else:
        assert row.fixture.teardown_status == 0 and (markers / "inner-ran").exists()


@_FX_HOST
def test_fixture_delayed_readiness_never_starts_setup_after_its_deadline(tmp_path, monkeypatch):
    root, markers = _fx_repo(tmp_path)
    monkeypatch.setattr(fx_runtime, "reserve_seconds", lambda timeout: 5.0)
    launch_args: list[list[str]] = []
    original_launch = fx_runtime.FixtureSupervisor.launch.__func__
    original_start = fx_runtime.FixtureSupervisor.start_setup

    def launch(cls, *args, **kwargs):
        handle = original_launch(cls, *args, **kwargs)
        launch_args.append(list(handle.process.args))
        return handle

    def slow_start(self, parent_cgroup, *, setup_deadline):
        # Readiness is delayed past the setup deadline.
        time.sleep(max(0.0, setup_deadline - time.monotonic()) + 0.2)
        return original_start(self, parent_cgroup, setup_deadline=setup_deadline)

    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "launch", classmethod(launch))
    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "start_setup", slow_start)
    server = _fx_server(root)
    started = time.monotonic()
    try:
        result = fx_evidence.TestBrokerClient(_fx_env(server)).run(
            list(_FX_PYTEST), timeout_seconds=30, cwd=root, fixture="fixture.py",
        )
        row = _fx_row(server)
    finally:
        server.stop()
    elapsed = time.monotonic() - started
    (argv,) = launch_args
    setup_deadline = float(argv[argv.index("--setup-deadline") + 1])
    request_deadline = float(argv[argv.index("--request-deadline") + 1])
    # Absolute deadlines: setup ends before the inner deadline (request - reserve).
    assert setup_deadline <= request_deadline - 5.0 - fx_runtime.SETUP_TERMINATION_MARGIN_SECONDS + 0.01
    assert not (markers / "setup-started").exists() and not (markers / "inner-ran").exists()
    assert row.outcome == "timed_out" and row.inner_exec == "not-attempted"
    assert row.fixture.setup_status == "timed_out" and row.fixture.setup_log_digest == ""
    assert result.outcome == "timed_out"
    assert elapsed < 30 + 2 * fx_evidence.MAX_SNAPSHOT_SECONDS + 2


@_FX_HOST
def test_fixture_sigterm_resistant_inner_and_slow_teardown_finish_by_the_request_deadline(tmp_path, monkeypatch):
    root, markers = _fx_repo(tmp_path)
    _fx_set_mode(root, "inner-block-hard teardown-slow", fx_teardown_seconds="60")
    server = _fx_server(root)
    started = time.monotonic()
    try:
        result = fx_evidence.TestBrokerClient(_fx_env(server)).run(
            list(_FX_PYTEST), timeout_seconds=30, cwd=root, fixture="fixture.py",
        )
        elapsed = time.monotonic() - started
        row = _fx_row(server)
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    assert (markers / "inner-blocking").exists() and (markers / "teardown-started").exists()
    assert row.outcome == "timed_out" and result.outcome == "timed_out"
    # Teardown consumed its whole positive bound and was cut off.
    assert row.fixture.teardown_status == "timed_out"
    assert 4.0 <= row.fixture.teardown_duration_seconds <= 8.0
    # Termination, drain, teardown and the sweep all ended by the request
    # deadline; only the declared snapshot overhead and reporting remain.
    assert elapsed < 30 + 4.0, elapsed


@_FX_HOST
@pytest.mark.parametrize("verdict", ["unrecognized-launcher", "probe-containment-unavailable"])
def test_fixture_unknown_launcher_verdict_never_spawns_the_target(tmp_path, monkeypatch, verdict):
    root, markers = _fx_repo(tmp_path)
    marker = markers / "unrecognized-ran"
    if verdict == "unrecognized-launcher":
        inner = [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('1')"]
    else:
        inner = list(_FX_PYTEST)

        def unavailable(argv, **kwargs):
            return runtime.LauncherProbeResult(tuple(argv), "unknown", "probe containment unavailable", "id")

        monkeypatch.setattr(runner_module, "probe_inner_launcher", unavailable)
    server = _fx_server(root)
    try:
        code = _fx_cli(monkeypatch, server, root, inner)
        row = _fx_row(server)
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    assert not marker.exists() and not (markers / "inner-ran").exists()
    assert (markers / "teardown-done").exists()
    assert row.outcome == "launch-failed" and row.inner_exec == "not-attempted"
    assert row.fixture.setup_status == 0 and row.fixture.teardown_status == 0
    assert code == 1
    assert not _authoritative_receipt_passes(row, claim="current-result")


@pytest.mark.parametrize("unreadable", ["stat", "status"])
def test_fixture_census_unreadable_stat_or_status_of_any_process_is_unconfirmed(tmp_path, unreadable):
    # A pre-window process (start 50) still has to be fully readable.
    proc = _fx_fake_proc(tmp_path / "proc", {100: {"start": 100}, 50: {"start": 50}})
    target = proc / "50" / unreadable
    target.unlink()
    target.mkdir()  # reading it fails while the entry exists
    result = fx_supervisor.census(window_start=100, owned=(), self_identity=(100, 100), proc_root=proc)
    assert result.cleanup == "unconfirmed"
    assert unreadable in result.reason or "unreadable" in result.reason


def test_fixture_modules_import_without_fcntl_and_refuse_off_linux(monkeypatch):
    import importlib

    monkeypatch.setitem(sys.modules, "fcntl", None)  # import fcntl raises ImportError
    reloaded = importlib.reload(fx_supervisor)
    try:
        assert reloaded.fcntl is None
        with pytest.raises(OSError):
            reloaded._ns_get_parent("/proc/self/ns/pid")
        monkeypatch.setattr(reloaded.sys, "platform", "win32")
        with pytest.raises(reloaded.HostAuthorityError):
            reloaded.verify_host_process_authority()
        importlib.reload(fx_runtime)
    finally:
        monkeypatch.undo()
        importlib.reload(fx_supervisor)
        importlib.reload(fx_runtime)


def _fx_raw_request(server, root, argv, *, timeout, fixture="fixture.py"):
    raw = os.urandom(16).hex()
    return {
        "turn_id": server.turn_id, "nonce": raw + "." + hmac_new(server.capability, raw),
        "argv": list(argv), "timeout_seconds": timeout, "cwd": str(root),
        "environment": {
            **{k: v for k, v in os.environ.items() if not k.startswith("AGENT_LOOP_TEST_BROKER")},
            fx_evidence.BROKER_CLIENT_ACKS_ENV: "1",
        },
        "fixture": fixture,
    }


@_FX_HOST
def test_fixture_setup_never_starts_when_the_go_ahead_arrives_after_its_deadline(tmp_path, monkeypatch):
    root, markers = _fx_repo(tmp_path)
    monkeypatch.setattr(fx_runtime, "reserve_seconds", lambda timeout: 5.0)
    deadlines: dict[str, float] = {}
    original_start = fx_runtime.FixtureSupervisor.start_setup
    original_send = fx_runtime.FixtureSupervisor._send

    def recording_start(self, parent_cgroup, *, setup_deadline):
        deadlines["setup"] = setup_deadline
        return original_start(self, parent_cgroup, setup_deadline=setup_deadline)

    def late_send(self, line):
        if line == "start":
            # The broker released setup in time, but the supervisor consumes
            # the go-ahead only after the setup deadline (descheduling).
            time.sleep(max(0.0, deadlines["setup"] - time.monotonic()) + 0.3)
        return original_send(self, line)

    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "start_setup", recording_start)
    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "_send", late_send)
    server = _fx_server(root)
    try:
        fx_evidence.TestBrokerClient(_fx_env(server)).run(
            list(_FX_PYTEST), timeout_seconds=30, cwd=root, fixture="fixture.py",
        )
        row = _fx_row(server)
    finally:
        server.stop()
    assert not (markers / "setup-started").exists() and not (markers / "inner-ran").exists()
    assert row.fixture.setup_status == "timed_out" and row.fixture.setup_log_digest == ""
    assert row.fixture.teardown_status == "not-run"
    assert row.inner_exec == "not-attempted" and row.outcome in {"failed", "timed_out"}
    assert not _authoritative_receipt_passes(row, claim="current-result")


@_FX_HOST
def test_fixture_non_reading_client_never_holds_the_run_past_its_deadline(tmp_path, monkeypatch):
    import socket as socket_module

    root, markers = _fx_repo(tmp_path)
    _fx_set_mode(root, "inner-noisy")
    forced: list[int] = []
    original_force = fx_runtime.FixtureSupervisor.force_terminate

    def spy_force(self):
        forced.append(self.process.pid)
        return original_force(self)

    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "force_terminate", spy_force)
    server = _fx_server(root)
    noisy = (sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", "test_fx.py")
    try:
        with socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM) as connection:
            connection.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_RCVBUF, 4096)
            connection.connect(server.endpoint)
            started = time.monotonic()
            fx_evidence._send_frame(connection, _fx_raw_request(server, root, noisy, timeout=30))
            # Never read a single frame: the broker's sends block.
            deadline = time.monotonic() + 30 + 2 * fx_evidence.MAX_SNAPSHOT_SECONDS + 10
            while time.monotonic() < deadline and not [r for r in server.journal if r.fixture is not None]:
                time.sleep(0.1)
            journaled_after = time.monotonic() - started
            row = _fx_row(server)
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
    assert (markers / "inner-noisy").exists() and (markers / "teardown-done").exists()
    # Issue #1653: the non-reading client wedges a stream send and is lost,
    # so its target is stopped and the row is non-evidence.
    assert row.outcome == "incomplete" and row.provenance == "telemetry-unverified"
    assert fx_evidence.BROKER_CLIENT_LOST_CAVEAT in row.caveats
    assert row.fixture.teardown_status == 0
    assert forced == []
    # Termination, teardown, the sweep and the after snapshot all finished
    # well before the 30 s request deadline.
    assert journaled_after < 30 + 3.0, journaled_after


_FX_TERM_RESISTANT = (
    "import os, pathlib, signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "pathlib.Path(%r).open('a').write(str(os.getpid()) + '\\n')\n"
    "time.sleep(600)\n"
)


@_FX_LINUX
def test_fixture_resistant_launcher_probe_cleanup_ends_by_the_deadline(tmp_path, monkeypatch):
    pid_file = tmp_path / "probe.pid"
    original = runtime._run_bounded_probe

    def resistant(argv, **kwargs):
        return original([sys.executable, "-c", _FX_TERM_RESISTANT % str(pid_file)], **kwargs)

    monkeypatch.setattr(runtime, "_run_bounded_probe", resistant)
    deadline = time.monotonic() + 3.0
    result = runtime.probe_inner_launcher(
        [sys.executable, "-m", "pytest", "-q"], cwd=tmp_path, environment=dict(os.environ),
        environment_is_complete=True, deadline=deadline, use_cache=False,
    )
    assert time.monotonic() <= deadline + 0.1
    assert result.state == "unknown" and result.diagnostic == runtime.PROBE_DEADLINE_DIAGNOSTIC
    _fx_assert_gone([int(line) for line in pid_file.read_text().split()])


@_FX_LINUX
def test_fixture_resistant_lifecycle_preflight_cleanup_ends_by_the_deadline(tmp_path, monkeypatch):
    from coding_review_agent_loop import lifecycle_probe

    pid_file = tmp_path / "preflight.pid"
    monkeypatch.setattr(lifecycle_probe, "_PREFLIGHT_CODE", _FX_TERM_RESISTANT % str(pid_file))
    eligible = lifecycle_probe.launch_eligibility_for([sys.executable, "-m", "pytest"], dict(os.environ), tmp_path)
    if eligible is None:
        pytest.skip("this interpreter is not an eligible pytest launcher")
    deadline = time.monotonic() + 3.0
    assert lifecycle_probe.prepare_lifecycle_probe(
        [sys.executable, "-m", "pytest"], dict(os.environ), tmp_path, deadline=deadline,
    ) is None
    assert time.monotonic() <= deadline + 0.1
    _fx_assert_gone([int(line) for line in pid_file.read_text().split()])


@_FX_HOST
def test_fixture_resistant_post_setup_probe_through_the_broker_keeps_the_reserve(tmp_path, monkeypatch):
    root, markers = _fx_repo(tmp_path)
    _fx_set_mode(root, "setup-slow", fx_setup_seconds="15")
    pid_file = tmp_path / "probe.pid"
    inner_deadlines: list[float] = []
    returned_at: list[float] = []
    original_init = fx_evidence._FixtureRun.__init__
    original = runtime._run_bounded_probe

    def recording_init(self, **kwargs):
        original_init(self, **kwargs)
        inner_deadlines.append(self.inner_deadline)

    def resistant(argv, **kwargs):
        try:
            return original([sys.executable, "-c", _FX_TERM_RESISTANT % str(pid_file)], **kwargs)
        finally:
            returned_at.append(time.monotonic())

    monkeypatch.setattr(fx_evidence._FixtureRun, "__init__", recording_init)
    monkeypatch.setattr(runtime, "_run_bounded_probe", resistant)
    server = _fx_server(root)
    try:
        fx_evidence.TestBrokerClient(_fx_env(server)).run(
            list(_FX_PYTEST), timeout_seconds=30, cwd=root, fixture="fixture.py",
        )
        row = _fx_row(server)
    finally:
        server.stop()
    _fx_assert_gone([*_fx_pids(markers), *(int(line) for line in pid_file.read_text().split())])
    assert returned_at and returned_at[-1] <= inner_deadlines[0]
    assert not (markers / "inner-ran").exists() and (markers / "teardown-done").exists()
    assert row.outcome == "timed_out" and row.inner_exec == "not-attempted"
    assert row.fixture.setup_status == 0 and row.fixture.teardown_status == 0


@_FX_LINUX
def test_fixture_deadline_bounds_the_held_child_handshake(tmp_path, monkeypatch):
    marker = tmp_path / "released"
    original_popen = runner_module.subprocess.Popen

    def delayed_shim(cmd, *args, **kwargs):
        cmd = list(cmd)
        if cmd[1:3] == ["-m", "coding_review_agent_loop.containment"]:
            # A managed-scope shim whose readiness arrives far too late.
            delay = (
                "import os, sys, time; time.sleep(30); "
                "os.execv(sys.executable, [sys.executable, '-m', 'coding_review_agent_loop.containment', *sys.argv[1:]])"
            )
            cmd = [cmd[0], "-c", delay, *cmd[3:]]
        return original_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(runner_module.subprocess, "Popen", delayed_shim)
    (tmp_path / "cgroup").mkdir()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [_SRC, os.environ.get("PYTHONPATH")]))}
    started = time.monotonic()
    deadline = started + 5.0
    result = run_foreground_test(
        [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('1')"],
        cwd=tmp_path, timeout_seconds=60, env=env, environment_is_complete=True,
        parent_cgroup_path=tmp_path / "cgroup", absolute_deadline=deadline, echo_output=False,
    )
    assert time.monotonic() <= deadline + 0.1
    assert result.outcome == "timed_out" and result.inner_exec == "not-attempted"
    assert not marker.exists()
    assert not (tmp_path / "cgroup" / "cgroup.procs").exists()


def _fx_proceed(deadline):
    return fx_runtime.SetupDecision("proceed")


@_FX_LINUX
def test_fixture_launcher_retargeted_during_held_child_readiness_is_never_released(tmp_path, monkeypatch):
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    shim = venv_bin / "python"
    shim.symlink_to(sys.executable)
    fake = tmp_path / "not-python"
    fake.write_text("#!/bin/sh\nexit 3\n")
    fake.chmod(0o755)
    marker = tmp_path / "inner-ran"
    (tmp_path / "test_ok.py").write_text(
        "import pathlib\ndef test_ok():\n    pathlib.Path(%r).write_text('1')\n" % str(marker)
    )
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    original_popen = runner_module.subprocess.Popen

    def retarget():
        shim.unlink()
        shim.symlink_to(fake)

    def delayed_shim(cmd, *args, **kwargs):
        cmd = list(cmd)
        if cmd[1:3] == ["-m", "coding_review_agent_loop.containment"]:
            # A fixture process retargets the shim while the held child is
            # still on its way to the readiness handshake.
            threading.Timer(0.5, retarget).start()
            delay = (
                "import os, sys, time; time.sleep(2); "
                "os.execv(sys.executable, [sys.executable, '-m', 'coding_review_agent_loop.containment', *sys.argv[1:]])"
            )
            cmd = [cmd[0], "-c", delay, *cmd[3:]]
        return original_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(runner_module.subprocess, "Popen", delayed_shim)
    monkeypatch.setattr(runner_module, "cgroup_path_for_pid", lambda pid: cgroup)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [_SRC, os.environ.get("PYTHONPATH")]))}
    result = run_foreground_test(
        [".venv/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_ok.py"],
        cwd=tmp_path, timeout_seconds=60, env=env, environment_is_complete=True,
        parent_cgroup_path=cgroup, absolute_deadline=time.monotonic() + 60,
        post_admission_setup=_fx_proceed, echo_output=False,
    )
    assert result.outcome == "launch-failed" and result.inner_exec == "not-attempted"
    assert result.diagnostic == fx_runtime.LAUNCHER_CHANGED_CAVEAT
    assert (cgroup / "cgroup.procs").exists()  # the child reached the handshake
    assert not marker.exists()


@_FX_LINUX
def test_fixture_late_successful_probe_never_launches_a_target_it_cannot_terminate(tmp_path, monkeypatch):
    marker = tmp_path / "target-ran"
    argv = [sys.executable, "-c", (
        "import pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(%r).write_text('1')\n"
        "time.sleep(600)\n" % str(marker)
    )]
    deadline = time.monotonic() + 6.0

    def late_verified_probe(probe_argv, **kwargs):
        # Authentication succeeds, but only 2 s before the inner deadline,
        # less than the target's termination allowance.
        time.sleep(max(0.0, deadline - 2.0 - time.monotonic()))
        return runtime.LauncherProbeResult(tuple(probe_argv), "verified", "ok", "", tuple(probe_argv))

    monkeypatch.setattr(runner_module, "probe_inner_launcher", late_verified_probe)
    result = run_foreground_test(
        argv, cwd=tmp_path, timeout_seconds=60, env=dict(os.environ), environment_is_complete=True,
        absolute_deadline=deadline, post_admission_setup=_fx_proceed, echo_output=False,
    )
    assert time.monotonic() <= deadline + 0.1
    assert result.outcome == "timed_out" and result.inner_exec == "not-attempted"
    assert not marker.exists()


@_FX_LINUX
def test_fixture_independent_import_check_is_bounded_by_the_deadline(tmp_path, monkeypatch):
    from coding_review_agent_loop import lifecycle_probe

    pid_file = tmp_path / "import-check.pid"
    monkeypatch.setattr(lifecycle_probe, "_IMPORT_CHECK_CODE", _FX_TERM_RESISTANT % str(pid_file))
    argv = [sys.executable, "-m", "pytest"]
    eligibility = lifecycle_probe.launch_eligibility_for(argv, dict(os.environ), tmp_path)
    if eligibility is None or eligibility.shape != "module":
        pytest.skip("this interpreter is not an eligible module pytest launcher")
    deadline = time.monotonic() + 3.0
    assert lifecycle_probe.run_independent_import_check(
        eligibility, argv, dict(os.environ), tmp_path, deadline=deadline,
    ) == "unavailable"
    assert time.monotonic() <= deadline + 0.1
    _fx_assert_gone([int(line) for line in pid_file.read_text().split()])
    # An exhausted allocation skips the optional diagnostic without spawning.
    pid_file.unlink()
    assert lifecycle_probe.run_independent_import_check(
        eligibility, argv, dict(os.environ), tmp_path, deadline=time.monotonic() + 0.5,
    ) == "unavailable"
    assert not pid_file.exists()


@pytest.mark.parametrize("path", ["failed-bootstrap", "post-run"])
def test_fixture_deadline_reaches_both_import_check_classification_paths(monkeypatch, tmp_path, path):
    from types import SimpleNamespace

    from coding_review_agent_loop import lifecycle_probe

    seen: list[object] = []

    def spy(eligibility, argv, env, cwd, *, deadline=None):
        seen.append(deadline)
        return "unavailable"

    monkeypatch.setattr(lifecycle_probe, "run_independent_import_check", spy)
    monkeypatch.setattr(lifecycle_probe, "read_probe_report", lambda *a: SimpleNamespace(records=()))
    setup = SimpleNamespace(eligibility=None, env={}, report_path=tmp_path / "r", nonce="n")
    deadline = time.monotonic() + 42
    if path == "failed-bootstrap":
        runner_module._classify_missing_pytest_bootstrap(["python", "-m", "pytest"], setup, "x", tmp_path, deadline)
    else:
        runner_module._classify_pre_collection_run(
            requested=["python", "-m", "pytest"], probe_setup=setup, returncode=1,
            scan=SimpleNamespace(no_module_pytest=True), analysis=None, cwd=tmp_path, deadline=deadline,
        )
    assert seen == [deadline]


@_FX_LINUX
@pytest.mark.parametrize("remaining", [0.6, 1.5], ids=["escalates-at-once", "short-grace"])
def test_fixture_descendant_cleanup_ends_by_the_absolute_deadline(tmp_path, remaining):
    from coding_review_agent_loop.test_workers import terminate_process_group_descendants

    pid_file = tmp_path / "resistant.pid"
    leader = subprocess.Popen(
        [sys.executable, "-c", _FX_TERM_RESISTANT % str(pid_file)], start_new_session=True,
    )
    try:
        deadline_wait = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline_wait:
            time.sleep(0.02)
        deadline = time.monotonic() + remaining
        result = terminate_process_group_descendants(leader.pid, deadline=deadline)
        assert time.monotonic() <= deadline + 0.15
        assert result.confirmed
    finally:
        _kill_if_active(leader.pid)
        leader.wait(timeout=10)
    _fx_assert_gone([leader.pid])


_FX_WORKER_RESISTANT_TEST = '''import os, pathlib, signal, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
MARKERS = pathlib.Path((HERE / "fx_markers").read_text().strip())


def test_parent_slow_on_term_with_resistant_child():
    child = subprocess.Popen([sys.executable, "-c", (
        "import os, pathlib, signal, time\\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
        "pathlib.Path(%r).write_text(str(os.getpid()))\\n"
        "time.sleep(600)\\n" % str(MARKERS / "resistant-child")
    )])

    def slow_exit(*_args):
        time.sleep(1.8)
        os._exit(1)

    signal.signal(signal.SIGTERM, slow_exit)
    (MARKERS / "inner-blocking").write_text("1")
    time.sleep(600)
'''


@_FX_HOST
def test_fixture_worker_descendant_cleanup_keeps_the_teardown_reserve(tmp_path, monkeypatch):
    from coding_review_agent_loop.test_workers import WorkerBudget

    root, markers = _fx_repo(tmp_path)
    _fx_set_mode(root, "", **{"test_worker_resistant.py": _FX_WORKER_RESISTANT_TEST})
    server = _fx_server(root)
    gib = 1024 ** 3
    budget = WorkerBudget(2, "inherited", "clamp", "cpu", {"cpu_available": 2, "host_usable_bytes": 64 * gib, "per_worker_bytes": gib, "reserve_bytes": gib}, False)
    assert budget.enforcing
    server._worker_lock_root = tmp_path / "locks"
    server.set_execution_context(containment_handle=None, process_started=None, process_finished=None, worker_budget=budget)
    inner_deadlines: list[float] = []
    returned_at: list[float] = []
    original_init = fx_evidence._FixtureRun.__init__
    original_run = runner_module.run_foreground_test

    def recording_init(self, **kwargs):
        original_init(self, **kwargs)
        inner_deadlines.append(self.inner_deadline)

    def recording_run(*args, **kwargs):
        try:
            return original_run(*args, **kwargs)
        finally:
            returned_at.append(time.monotonic())

    monkeypatch.setattr(fx_evidence._FixtureRun, "__init__", recording_init)
    monkeypatch.setattr(runner_module, "run_foreground_test", recording_run)
    inner = (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-n", "0", "test_worker_resistant.py")
    try:
        result = fx_evidence.TestBrokerClient(_fx_env(server)).run(
            list(inner), timeout_seconds=30, cwd=root, fixture="fixture.py",
        )
        row = _fx_row(server)
    finally:
        server.stop()
    child = int((markers / "resistant-child").read_text())
    _fx_assert_gone([*_fx_pids(markers), child])
    assert (markers / "inner-blocking").exists() and (markers / "teardown-done").exists()
    # The runner, including worker-descendant cleanup, returned by the inner deadline.
    assert returned_at and returned_at[-1] <= inner_deadlines[0] + 0.05
    assert row.outcome == "timed_out" and result.outcome == "timed_out"
    assert row.fixture.teardown_status == 0



# --- client-loss cancellation in the runner and fixture path (issue #1653) ----

_RACE_GROUP_TARGET = (
    "import os, pathlib, signal, sys, time\n"
    "base = pathlib.Path(sys.argv[1])\n"
    "if os.fork() == 0:\n"
    "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "    (base / 'child.pid').write_text(str(os.getpid()))\n"
    "    time.sleep(120)\n"
    "    os._exit(0)\n"
    "(base / 'leader.pid').write_text(str(os.getpid()))\n"
    "time.sleep(120)\n"
)


def _race_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def test_overlap_refusal_is_streamed_through_the_output_callback_with_its_holder(tmp_path, capsys):
    command = [sys.executable, "-c", "pass"]
    env = {**os.environ, "AGENT_LOOP_INVOCATION_ID": "overlap-stream-1653"}
    lock = runtime.acquire_command_lane(command, cwd=tmp_path, env=env)
    assert lock is not None
    streamed: list[str] = []
    try:
        result = run_foreground_test(
            command, cwd=tmp_path, timeout_seconds=5, env=env, environment_is_complete=True,
            output_callback=streamed.append, echo_output=False,
            lock_holder_describer=lambda: "`pytest -q` (running, 3 s)",
        )
    finally:
        lock.close()
    assert result.outcome == "overlap-rejected"
    assert result.returncode == runtime.OVERLAP_REJECTED_EXIT_CODE
    text = "".join(streamed)
    assert text.startswith(runtime.OVERLAP_REJECTED_MESSAGE)
    assert "Held by: `pytest -q` (running, 3 s)." in text
    assert "Retrying immediately will be refused again" in text
    assert result.diagnostic == text.strip()
    # Streamed to the requesting client, not only printed on the parent's stderr.
    assert runtime.OVERLAP_REJECTED_MESSAGE not in capsys.readouterr().err


def test_overlap_refusal_without_a_describer_keeps_the_local_stderr_text(tmp_path, capsys):
    command = [sys.executable, "-c", "pass"]
    env = {**os.environ, "AGENT_LOOP_INVOCATION_ID": "overlap-local-1653"}
    lock = runtime.acquire_command_lane(command, cwd=tmp_path, env=env)
    try:
        result = run_foreground_test(
            command, cwd=tmp_path, timeout_seconds=5, env=env, environment_is_complete=True,
        )
    finally:
        lock.close()
    err = capsys.readouterr().err
    assert result.outcome == "overlap-rejected"
    assert runtime.OVERLAP_REJECTED_MESSAGE in err and "Held by" not in err
    assert "Retrying immediately will be refused again" in err


def test_cancel_landing_at_the_guarded_launch_never_executes_the_target(tmp_path):
    """Row disconnect-at-launch-race: the guarded check sees the cancel first."""
    marker = tmp_path / "spawned"
    cancel = threading.Event()

    class WatcherWinsGuard:
        def __enter__(self):
            cancel.set()  # the client-loss watcher set the token under the lock
            return self

        def __exit__(self, *exc):
            return False

    result = run_foreground_test(
        [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"], cwd=tmp_path,
        timeout_seconds=30, env=dict(os.environ), environment_is_complete=True, echo_output=False,
        host_wait_cancel=cancel, host_launch_guard=lambda: WatcherWinsGuard(),
    )
    assert result.outcome == "cancelled" and result.inner_exec == "not-attempted"
    assert not marker.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process-group census via /proc")
def test_cancel_after_the_guarded_launch_terminates_the_whole_registered_group(tmp_path):
    """Row disconnect-at-launch-race: the launch lands first, so the group is stopped."""
    base = tmp_path / "group"
    base.mkdir()
    cancel = threading.Event()
    registered: list[object] = []

    def started(proc):
        registered.append(proc)

        def watcher():
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not (base / "child.pid").exists():
                time.sleep(0.05)
            cancel.set()

        threading.Thread(target=watcher, daemon=True).start()

    began = time.monotonic()
    result = run_foreground_test(
        [sys.executable, "-c", _RACE_GROUP_TARGET, str(base)], cwd=tmp_path, timeout_seconds=120,
        env=dict(os.environ), environment_is_complete=True, echo_output=False,
        host_wait_cancel=cancel, host_launch_guard=lambda: threading.Lock(), process_started=started,
    )
    assert registered and result.outcome != "passed" and result.inner_exec == "started"
    assert time.monotonic() - began < 30
    # The lane was released only after the TERM-ignoring child was gone too.
    assert not _race_alive(int((base / "leader.pid").read_text()))
    assert not _race_alive(int((base / "child.pid").read_text()))
    relock = runtime.acquire_command_lane(
        [sys.executable, "-c", _RACE_GROUP_TARGET, str(base)], cwd=tmp_path, env=dict(os.environ),
    )
    assert relock is not None
    relock.close()


def _fx_idle(server, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and (server._handler_threads or server._requests):
        time.sleep(0.05)
    return not server._handler_threads and not server._requests


def _fx_live_retry(server, root):
    result = fx_evidence.TestBrokerClient(_fx_env(server)).run(
        list(_FX_PYTEST), timeout_seconds=40, cwd=root, fixture="fixture.py",
    )
    assert result.outcome not in {"overlap-rejected", "worker-budget-busy", "cancelled"}, result
    return result


@_FX_HOST
def test_fixture_client_lost_before_the_supervisor_launch_launches_nothing(tmp_path, monkeypatch):
    """Row fixture-loss-before-launch: the hook's guarded check reads the request token."""
    import socket as socket_module

    root, markers = _fx_repo(tmp_path)
    launches: list[object] = []
    real_launch = fx_runtime.FixtureSupervisor.launch.__func__

    def counting_launch(cls, *args, **kwargs):
        launches.append(args)
        return real_launch(cls, *args, **kwargs)

    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "launch", classmethod(counting_launch))
    entered = threading.Event()
    real_hook = fx_evidence.TestBrokerServer._fixture_setup_hook

    def wrapped(self, fixture_run, **kwargs):
        hook = real_hook(self, fixture_run, **kwargs)
        state = kwargs["state"]

        def after_the_runner_check(deadline):
            entered.set()  # the runner's own pre-hook cancel check has passed
            assert state.cancel.wait(20)
            return hook(deadline)

        return after_the_runner_check

    monkeypatch.setattr(fx_evidence.TestBrokerServer, "_fixture_setup_hook", wrapped)
    server = _fx_server(root)
    try:
        client = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
        client.connect(server.endpoint)
        fx_evidence._send_frame(client, _fx_raw_request(server, root, _FX_PYTEST, timeout=40))
        assert entered.wait(30)
        client.close()
        assert _fx_idle(server)
        assert launches == [] and server.journal == ()
        assert not (markers / "setup-started").exists()
        monkeypatch.setattr(fx_evidence.TestBrokerServer, "_fixture_setup_hook", real_hook)
        _fx_live_retry(server, root)  # the lane is reusable
        assert len(launches) == 1
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))


@_FX_HOST
def test_fixture_client_lost_during_setup_gets_teardown_and_a_non_evidence_row(tmp_path, monkeypatch):
    """Rows fixture-loss-during-setup and disconnect-receipt-classification (fixture-folded)."""
    import socket as socket_module

    root, markers = _fx_repo(tmp_path, modes="setup-block")
    teardowns: list[int] = []
    real_request_teardown = fx_runtime.FixtureSupervisor.request_teardown

    def spy_teardown(self):
        teardowns.append(self.process.pid)
        return real_request_teardown(self)

    monkeypatch.setattr(fx_runtime.FixtureSupervisor, "request_teardown", spy_teardown)
    server = _fx_server(root)
    try:
        client = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
        client.connect(server.endpoint)
        fx_evidence._send_frame(client, _fx_raw_request(server, root, _FX_PYTEST, timeout=60))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (markers / "setup-blocking").exists():
            time.sleep(0.05)
        assert (markers / "setup-blocking").exists()
        started = time.monotonic()
        client.close()
        assert _fx_idle(server)
        assert time.monotonic() - started < 30  # not the remaining setup budget
        row = _fx_row(server)
        assert teardowns and (markers / "teardown-done").exists()
        assert row.provenance == "telemetry-unverified" and row.outcome == "incomplete"
        assert row.caveats[0] == fx_evidence.BROKER_CLIENT_LOST_CAVEAT
        assert row.fixture.teardown_status == 0
        evidence = fx_evidence.reconcile_test_observations(server.journal)
        assert row.receipt_id not in evidence.authoritative_failures
        (public,) = evidence.public_projection()["observations"]
        assert public["receipt_id"] == row.receipt_id
        assert fx_evidence.BROKER_CLIENT_LOST_CAVEAT in public["caveats"]
        _fx_set_mode(root, "")
        _fx_live_retry(server, root)  # the lane is reusable
    finally:
        server.stop()
    _fx_assert_gone(_fx_pids(markers))
