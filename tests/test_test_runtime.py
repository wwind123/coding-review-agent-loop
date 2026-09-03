import json
import sys
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
