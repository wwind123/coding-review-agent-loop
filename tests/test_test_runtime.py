import json
import os
import shlex
import shutil
import subprocess
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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


def test_wrapper_module_fallback_preserves_lexical_virtualenv_interpreter(tmp_path, monkeypatch):
    venv = tmp_path / "virtualenv"
    interpreter = venv / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys.executable).resolve())
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    monkeypatch.setattr(runtime.sys, "executable", str(interpreter))
    monkeypatch.setattr(runtime.shutil, "which", lambda _name, path=None: None)

    assert runtime._wrapper_candidates({"PATH": ""}) == [
        (str(interpreter), "-m", "coding_review_agent_loop.cli", "run-tests")
    ]


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
