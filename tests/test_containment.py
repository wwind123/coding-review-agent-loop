"""Focused policy, shim, fallback, and lane tests for issue #731."""

import errno
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.containment as containment_module
from coding_review_agent_loop.checks import _raise_for_gate_result
from coding_review_agent_loop.containment import (
    AggregateLease,
    CapabilityManifest,
    ContainmentPolicy,
    ContainmentEvidence,
    InvocationHandle,
    ResourceLimits,
    _counter_capabilities,
    evidence_for,
    _shim,
    default_policy,
    parse_limit,
    preflight_containment,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import Runner, run_foreground_test
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


def test_default_memory_high_tracks_explicit_memory_max(monkeypatch):
    monkeypatch.setattr(
        "coding_review_agent_loop.containment.host_memory_bytes", lambda: 1_000_000
    )
    policy = default_policy(mode="off", memory_max=100_000)
    assert policy.aggregate.memory_max == 100_000
    assert policy.aggregate.memory_high == 87_000


def test_limit_parser_rejects_invalid_iec_suffix():
    with pytest.raises(AgentLoopError, match="must be bytes"):
        parse_limit("5ib", name="MemoryMax")


def test_restrict_clamps_memory_high_to_stricter_memory_max():
    combined = ResourceLimits(800, 1000, None, None).restrict(
        ResourceLimits(None, 500, None, None)
    )
    assert combined == ResourceLimits(500, 500, None, None)


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


def test_auto_preflight_reports_unsupported_counter_tree(tmp_path):
    (tmp_path / "cgroup.controllers").write_text("memory pids\n", encoding="ascii")
    (tmp_path / "memory.events").write_text("oom_kill 0\n", encoding="ascii")
    (tmp_path / "pids.events").write_text("max 0\n", encoding="ascii")
    policy = default_policy(mode="auto", cache_dir=tmp_path)
    manifest = preflight_containment(policy, cgroup_root=tmp_path, probe=False)
    assert manifest.ready is True
    assert "memory.peak" in manifest.unavailable_counters


def test_required_preflight_fails_closed_on_unsupported_host(monkeypatch, tmp_path):
    policy = default_policy(mode="required", cache_dir=tmp_path)
    monkeypatch.setattr(containment_module.platform, "system", lambda: "Darwin")
    with pytest.raises(AgentLoopError, match="required containment unavailable"):
        preflight_containment(policy, cgroup_root=tmp_path, probe=False)


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


def test_resource_counters_alone_do_not_fail_a_valid_invocation(tmp_path):
    policy = default_policy(mode="off", cache_dir=tmp_path)
    handle = InvocationHandle(
        policy=policy,
        role="coder",
        invocation_id="counter-only",
        unit_name="agent-loop-counter-only.scope",
        report_path=tmp_path / "report.json",
        launcher_argv=(),
        child_limits=policy.role_limits("coder"),
        aggregate_limits=policy.aggregate,
    )
    evidence = evidence_for(
        handle,
        {},
        {"memory.events": {"oom_kill": 1}, "pids.events": {"max": 1}},
        cleanup_confirmed=True,
    )
    assert evidence.resource_exhausted is False


def test_systemd_oom_result_classifies_resource_exhaustion(monkeypatch, tmp_path):
    policy = default_policy(mode="auto", cache_dir=tmp_path)
    handle = InvocationHandle(
        policy=policy,
        role="coder",
        invocation_id="oom-result",
        unit_name="agent-loop-oom-result.scope",
        report_path=tmp_path / "report.json",
        launcher_argv=(),
        child_limits=policy.role_limits("coder"),
        aggregate_limits=policy.aggregate,
        backend="systemd-cgroup-v2",
        capabilities=CapabilityManifest("systemd-cgroup-v2", True),
    )
    monkeypatch.setattr(containment_module.shutil, "which", lambda _name: "/bin/systemctl")
    monkeypatch.setattr(
        containment_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="oom-kill\n", stderr=""),
    )
    assert handle.refresh_resource_status() == "oom-kill"
    evidence = evidence_for(handle, {}, {}, cleanup_confirmed=True)
    assert evidence.resource_exhausted is True
    assert evidence.termination_cause == "oom"
    assert evidence.applicable_limit == "MemoryMax/MemorySwapMax"


def test_passing_test_gate_is_not_overridden_by_resource_evidence(tmp_path):
    evidence = ContainmentEvidence(
        backend="systemd-cgroup-v2",
        termination_cause="oom",
        cleanup_confirmed=True,
    )
    result = SimpleNamespace(outcome="passed", containment=evidence)
    _raise_for_gate_result(result, SimpleNamespace(coder_test_command_timeout_seconds=5))


def test_confirm_empty_does_not_accept_cgroup_read_error(monkeypatch, tmp_path):
    policy = default_policy(mode="auto", cache_dir=tmp_path)
    handle = InvocationHandle(
        policy=policy,
        role="coder",
        invocation_id="read-error",
        unit_name="agent-loop-read-error.scope",
        report_path=tmp_path / "report.json",
        launcher_argv=(),
        child_limits=policy.role_limits("coder"),
        aggregate_limits=policy.aggregate,
        backend="systemd-cgroup-v2",
        capabilities=CapabilityManifest("systemd-cgroup-v2", True),
    )
    handle.cgroup_path = tmp_path / "scope"
    monkeypatch.setattr(handle, "_unit_inactive", lambda: True)
    monkeypatch.setattr(handle, "_cgroup_members", lambda: None)
    assert handle.confirm_empty(timeout=0) is False


def test_invocation_prepare_uses_supplied_manifest(monkeypatch, tmp_path):
    policy = default_policy(mode="auto", cache_dir=tmp_path)
    manifest = CapabilityManifest("process-group", False, reason="test manifest")
    monkeypatch.setattr(
        containment_module,
        "preflight_containment",
        lambda *_args, **_kwargs: pytest.fail("preflight should use the cached manifest"),
    )
    handle = InvocationHandle.prepare(
        policy,
        role="coder",
        target_argv=[sys.executable, "-c", "pass"],
        manifest=manifest,
    )
    try:
        assert handle.backend == "process-group"
        assert handle.capabilities == manifest
    finally:
        handle.close()


def test_contained_target_exec_error_uses_preflight_retry_guidance(monkeypatch, tmp_path):
    runner = Runner()
    command = "provider-cli"
    runner.remember_agent_command(command, "/tmp/provider-cli", "--provider-cmd")
    monkeypatch.setattr("coding_review_agent_loop.runner.shutil.which", lambda _name: None)
    retryable, detail = runner.target_exec_retry_decision(command, errno.ENOENT)
    assert retryable is True
    assert "disappeared after successful preflight" in detail


def test_aggregate_leases_reapply_strictest_live_ceiling(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        containment_module,
        "_apply_slice_properties",
        lambda _policy, limits: calls.append(limits) or True,
    )
    first_policy = default_policy(
        mode="auto", cache_dir=tmp_path, memory_high=600, memory_max=800, tasks_max=100
    )
    second_policy = default_policy(
        mode="auto", cache_dir=tmp_path, memory_high=300, memory_max=400, tasks_max=50
    )
    first = AggregateLease.acquire(first_policy)
    second = AggregateLease.acquire(second_policy)
    try:
        assert second.limits.memory_max == 400
        assert second.limits.memory_high == 300
        second.close()
        assert calls[-1].memory_max == 800
        assert calls[-1].memory_high == 600
    finally:
        first.close()


def test_aggregate_property_update_is_inside_lease_lock(monkeypatch, tmp_path):
    lock_held = False
    applied_while_locked = []
    real_flock = containment_module.fcntl.flock

    def observe_lock(fd, operation):
        nonlocal lock_held
        if operation == containment_module.fcntl.LOCK_EX:
            lock_held = True
        return real_flock(fd, operation)

    def apply(_policy, _limits):
        applied_while_locked.append(lock_held)
        return True

    monkeypatch.setattr(containment_module.fcntl, "flock", observe_lock)
    monkeypatch.setattr(containment_module, "_apply_slice_properties", apply)
    policy = default_policy(mode="auto", cache_dir=tmp_path, memory_max=800)
    lease = AggregateLease.acquire(policy)
    lease.close()
    assert applied_while_locked == [True, True]


def test_timeout_terminates_descendant_process_group(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
    )
    result = run_foreground_test(
        [sys.executable, "-c", code, str(pid_file)],
        cwd=tmp_path,
        timeout_seconds=0.2,
        containment_policy=default_policy(mode="off", cache_dir=tmp_path / "runtime"),
    )
    assert result.outcome == "timed_out"
    child_pid = int(pid_file.read_text(encoding="ascii"))
    time.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@pytest.mark.parametrize("use_pty", [False, True])
def test_contained_spawn_uses_runner_retry_and_cleans_on_exception(monkeypatch, tmp_path, use_pty):
    class StubHandle:
        managed = True
        launcher_argv = ("systemd-run",)
        invocation_id = "exceptional-run"
        target_exec_errno = None

        def __init__(self):
            self.terminated = False
            self.confirmed = False
            self.closed = False

        def terminate(self):
            self.terminated = True

        def confirm_empty(self, **_kwargs):
            self.confirmed = True
            return True

        def close(self):
            self.closed = True

    runner = Runner()
    handle = StubHandle()
    calls = []

    def prepare(*_args, **_kwargs):
        runner._active_handles[handle.invocation_id] = handle
        return handle

    def spawn_with_retry(cmd, _spawn):
        calls.append(cmd)
        raise RuntimeError("synthetic spawn failure")

    monkeypatch.setattr(runner, "_prepare_containment", prepare)
    monkeypatch.setattr(runner, "_spawn_with_retry", spawn_with_retry)
    with pytest.raises(RuntimeError, match="synthetic spawn failure"):
        runner.run_with_log(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            log_path=tmp_path / "run.log",
            label="agent",
            progress_interval_seconds=1,
            check=False,
            use_pty=use_pty,
        )
    assert calls == [[sys.executable, "-c", "pass"]]
    assert handle.terminated and handle.confirmed and handle.closed
    assert runner._active_handles == {}
