"""Focused policy, shim, fallback, and lane tests for issue #731."""

import json
import sys

import pytest

from coding_review_agent_loop.containment import (
    ContainmentPolicy,
    ResourceLimits,
    _counter_capabilities,
    _shim,
    default_policy,
    preflight_containment,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import run_foreground_test
from coding_review_agent_loop.test_runtime import (
    OVERLAP_REJECTED_EXIT_CODE,
    acquire_command_lane,
)


def test_limits_parse_percentages_and_reject_contradictions(monkeypatch):
    monkeypatch.setattr(
        "coding_review_agent_loop.containment.host_memory_bytes", lambda: 1_000_000
    )
    policy = default_policy(
        mode="off", memory_high="40%", memory_max="50%", memory_swap_max="0", tasks_max=32
    )
    assert policy.aggregate.memory_high == 400_000
    assert policy.aggregate.memory_max == 500_000
    assert policy.role_limits("coder").memory_max <= 500_000
    with pytest.raises(AgentLoopError, match="MemoryHigh cannot exceed"):
        ContainmentPolicy(
            mode="off", aggregate=ResourceLimits(10, 5, 0, 10), roles=()
        )


def test_optional_counter_absence_is_capability_metadata(tmp_path):
    (tmp_path / "memory.events").write_text("oom_kill 0\n", encoding="ascii")
    (tmp_path / "pids.events").write_text("max 0\n", encoding="ascii")
    supported, unavailable, required_missing = _counter_capabilities(tmp_path)
    assert "memory.events" in supported
    assert "memory.peak" in unavailable
    assert required_missing == ()


def test_shim_report_is_authoritative_for_target_exit_one_and_203(tmp_path):
    for code in (1, 203):
        report = tmp_path / f"report-{code}.json"
        assert _shim(
            [
                "--shim", "--report", str(report), "--",
                sys.executable, "-c", f"raise SystemExit({code})",
            ]
        ) == code
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["state"] == "target-exited"
        assert payload["returncode"] == code


def test_same_lane_duplicate_is_rejected_before_spawn(tmp_path, capsys):
    env = {"AGENT_LOOP_INVOCATION_ID": "same-turn"}
    command = [sys.executable, "-c", "raise SystemExit(9)"]
    first = acquire_command_lane(command, cwd=tmp_path, env=env)
    assert first is not None
    try:
        result = run_foreground_test(
            command,
            cwd=tmp_path,
            timeout_seconds=2,
            env=env,
        )
    finally:
        first.close()
    assert result.outcome == "overlap-rejected"
    assert result.returncode == OVERLAP_REJECTED_EXIT_CODE
    assert "already running" in capsys.readouterr().err


def test_nested_test_wrapper_inherits_parent_scope(monkeypatch, tmp_path):
    policy = default_policy(mode="auto", cache_dir=tmp_path / "runtime")
    inherited = tmp_path / "parent-cgroup"
    inherited.mkdir()
    monkeypatch.setattr(
        "coding_review_agent_loop.runner.cgroup_path_for_pid",
        lambda _pid: inherited,
    )
    monkeypatch.setattr(
        "coding_review_agent_loop.runner.InvocationHandle.prepare",
        lambda *_args, **_kwargs: pytest.fail("nested wrapper must not create a sibling scope"),
    )
    result = run_foreground_test(
        [sys.executable, "-c", "print('inherited')"],
        cwd=tmp_path,
        timeout_seconds=2,
        env={"AGENT_LOOP_INVOCATION_ID": "parent-turn"},
        containment_policy=policy,
    )
    assert result.passed
    assert result.containment is not None
    assert result.containment.backend == "systemd-cgroup-v2-inherited"


def test_required_preflight_reports_unsupported_counter_tree(tmp_path):
    (tmp_path / "cgroup.controllers").write_text("memory pids\n", encoding="ascii")
    (tmp_path / "memory.events").write_text("oom_kill 0\n", encoding="ascii")
    (tmp_path / "pids.events").write_text("max 0\n", encoding="ascii")
    policy = default_policy(mode="required", cache_dir=tmp_path)
    manifest = preflight_containment(policy, cgroup_root=tmp_path, probe=False)
    assert manifest.ready is True
    assert "memory.peak" in manifest.unavailable_counters


@pytest.mark.skipif(sys.platform != "linux", reason="systemd containment is Linux-only")
def test_managed_foreground_scope_has_cleanup_evidence(tmp_path):
    policy = default_policy(mode="auto", cache_dir=tmp_path / "runtime")
    manifest = preflight_containment(policy)
    if not manifest.memory_ceiling_claimed:
        pytest.skip(manifest.reason or "systemd cgroup preflight unavailable")
    result = run_foreground_test(
        [sys.executable, "-c", "print('contained')"],
        cwd=tmp_path,
        timeout_seconds=5,
        containment_policy=policy,
    )
    assert result.passed
    assert result.containment is not None
    assert result.containment.backend == "systemd-cgroup-v2"
    assert result.containment.cleanup_confirmed
