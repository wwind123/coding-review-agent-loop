import json
import os
import shlex
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import coding_review_agent_loop.test_runtime as runtime
from agent_loop_helpers import make_config
from coding_review_agent_loop.cli import build_parser, main
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import run_foreground_test


def _now() -> datetime:
    return datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


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
        memory, commands=(command,), cwd=tmp_path, policy_ceiling_seconds=1800
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

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    result = runtime.probe_inner_launcher(
        [sys.executable, "-m", "pytest", "tests/test_protocol.py", "-q"], cwd=tmp_path
    )
    assert result.state == "verified"
    assert calls[0][0] == (sys.executable, "-m", "pytest", "--version")
    assert calls[0][1]["timeout"] == 5.0


def test_non_python_m_pytest_command_is_not_spawned_by_preflight(tmp_path, monkeypatch):
    calls = []
    non_python = tmp_path / "repo-script"
    non_python.write_text("#!/bin/sh\n", encoding="utf-8")
    non_python.chmod(0o755)

    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    result = runtime.probe_inner_launcher(
        [str(non_python), "-m", "pytest", "tests"], cwd=tmp_path
    )

    assert result.state == "unknown"
    assert "unrecognized" in result.diagnostic
    assert calls == []


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

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
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
        runtime.subprocess,
        "run",
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


def test_inner_probe_timeout_is_bounded_and_cached(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "inner-timeout-test")

    def timeout(*_args, **kwargs):
        calls.append(kwargs["timeout"])
        raise runtime.subprocess.TimeoutExpired([sys.executable, "-m", "pytest"], 5.0)

    monkeypatch.setattr(runtime.subprocess, "run", timeout)
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

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
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

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    for index in range(runtime.MAX_INNER_PROBE_CANDIDATES):
        interpreter = tmp_path / f"venv-{index}" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("python\n", encoding="utf-8")
        interpreter.chmod(0o755)
        result = runtime.probe_inner_launcher(
            [str(interpreter), "-m", "pytest"], cwd=tmp_path
        )
        assert result.state == "verified"
    over_limit = tmp_path / "venv-over-limit" / "bin" / "python"
    over_limit.parent.mkdir(parents=True)
    over_limit.write_text("python\n", encoding="utf-8")
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
    interpreter.write_text("python\n", encoding="utf-8")
    interpreter.chmod(0o755)
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "alternate-interpreter-repair-test")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        if len(calls) == 1:
            return type("Completed", (), {"returncode": 1, "stdout": "", "stderr": "No module named pytest"})()
        return type("Completed", (), {"returncode": 0, "stdout": "pytest 9", "stderr": ""})()

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    command = [str(interpreter), "-m", "pytest", "tests"]
    first = runtime.probe_inner_launcher(command, cwd=tmp_path)
    assert first.state == "failed"

    pytest_package = site_packages / "pytest"
    pytest_package.mkdir()
    (pytest_package / "__init__.py").write_text("__version__ = '9'\n", encoding="utf-8")
    second = runtime.probe_inner_launcher(command, cwd=tmp_path)

    assert second.state == "verified"
    assert len(calls) == 2
    assert calls[0][0] == (str(interpreter.resolve()), "-m", "pytest", "--version")
    identity = runtime.launcher_candidate_identity(command, cwd=tmp_path)
    redacted = runtime._redacted_identity(identity, cwd=tmp_path)
    assert str(venv) not in json.dumps(redacted)


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

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
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

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    interpreters = []
    for index in range(runtime.MAX_INNER_PROBE_CANDIDATES + 2):
        interpreter = tmp_path / f"concurrent-venv-{index}" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("python\n", encoding="utf-8")
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
