"""Focused policy, shim, fallback, and lane tests for issue #731."""

import errno
import json
import os
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest
from _pytest.outcomes import Failed

from _proc_probe import (
    PUBLISH_SNIPPET,
    kill_if_same_instance,
    read_pid_record,
    readiness_gate,
    wait_group_gone,
    wait_until_gone,
)

import coding_review_agent_loop.cli as cli_module
import coding_review_agent_loop.containment as containment_module
import coding_review_agent_loop.runner as runner_module
from coding_review_agent_loop.checks import _raise_for_gate_result, _record_gate_observation
from coding_review_agent_loop.containment import (
    AggregateLease,
    CapabilityManifest,
    ContainmentPolicy,
    ContainmentEvidence,
    InvocationHandle,
    ResourceLimits,
    _counter_capabilities,
    build_scope_argv,
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
    load_launcher_health,
    load_runtime_memory,
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


def test_nested_test_wrapper_uses_ambient_turn_when_env_is_omitted(monkeypatch, tmp_path):
    policy = default_policy(mode="auto", cache_dir=tmp_path / "runtime")
    inherited = tmp_path / "parent-cgroup"
    inherited.mkdir()
    monkeypatch.setenv("AGENT_LOOP_INVOCATION_ID", "parent-turn")
    monkeypatch.setattr(runner_module, "cgroup_path_for_pid", lambda _pid: inherited)
    monkeypatch.setattr(
        runner_module.InvocationHandle,
        "prepare",
        lambda *_args, **_kwargs: pytest.fail("nested wrapper must not create a sibling scope"),
    )

    result = run_foreground_test(
        [sys.executable, "-c", "print('inherited')"],
        cwd=tmp_path,
        timeout_seconds=2,
        env=None,
        containment_policy=policy,
    )

    assert result.passed
    assert result.containment is not None
    assert result.containment.backend == "systemd-cgroup-v2-inherited"


def test_broker_held_exec_attaches_and_verifies_before_release(monkeypatch, tmp_path):
    parent = tmp_path / "coder-cgroup"
    parent.mkdir()
    (parent / "cgroup.procs").write_text("", encoding="ascii")
    monkeypatch.setattr(runner_module, "cgroup_path_for_pid", lambda _pid: parent)

    result = run_foreground_test(
        [sys.executable, "-c", "print('attached')"],
        cwd=tmp_path,
        timeout_seconds=2,
        parent_cgroup_path=parent,
    )

    assert result.passed
    assert (parent / "cgroup.procs").read_text(encoding="ascii").isdigit()


def test_broker_held_exec_verification_mismatch_terminates_child(monkeypatch, tmp_path):
    parent = tmp_path / "coder-cgroup"
    other = tmp_path / "other-cgroup"
    parent.mkdir()
    other.mkdir()
    (parent / "cgroup.procs").write_text("", encoding="ascii")
    monkeypatch.setattr(runner_module, "cgroup_path_for_pid", lambda _pid: other)

    with pytest.raises(AgentLoopError, match="could not be verified"):
        run_foreground_test(
            [sys.executable, "-c", "print('must not run')"],
            cwd=tmp_path,
            timeout_seconds=2,
            parent_cgroup_path=parent,
        )


def test_broker_held_exec_handshake_timeout_terminates_child(monkeypatch, tmp_path):
    parent = tmp_path / "coder-cgroup"
    parent.mkdir()
    (parent / "cgroup.procs").write_text("", encoding="ascii")
    monkeypatch.setattr(runner_module.select, "select", lambda *_args, **_kwargs: ([], [], []))

    with pytest.raises(AgentLoopError, match="did not reach the containment handshake"):
        run_foreground_test(
            [sys.executable, "-c", "print('must not run')"],
            cwd=tmp_path,
            timeout_seconds=2,
            parent_cgroup_path=parent,
        )


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


def test_containment_preflight_cli_renders_portable_fallback(monkeypatch, capsys):
    manifest = CapabilityManifest(
        "process-group", False, reason="test host has no delegated user manager"
    )
    monkeypatch.setattr(cli_module, "preflight_containment", lambda _policy: manifest)

    assert cli_module.main(["containment-preflight", "--containment-mode", "auto"]) == 0
    output = capsys.readouterr().out
    assert "Containment preflight" in output
    assert "backend: process-group" in output
    assert "portable process-group fallback" in output
    assert "test host has no delegated user manager" in output


@pytest.mark.skipif(sys.platform != "linux", reason="systemd containment is Linux-only")
def test_managed_foreground_scope_has_cleanup_evidence(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_LOOP_INVOCATION_ID", raising=False)
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


def test_resource_result_is_reset_after_evidence_is_recorded(monkeypatch, tmp_path):
    policy = default_policy(mode="auto", cache_dir=tmp_path)
    handle = InvocationHandle(
        policy=policy,
        role="coder",
        invocation_id="oom-reset",
        unit_name="agent-loop-oom-reset.scope",
        report_path=tmp_path / "report.json",
        launcher_argv=(),
        child_limits=policy.role_limits("coder"),
        aggregate_limits=policy.aggregate,
        backend="systemd-cgroup-v2",
        capabilities=CapabilityManifest("systemd-cgroup-v2", True),
    )
    commands = []
    monkeypatch.setattr(containment_module.shutil, "which", lambda _name: "/bin/systemctl")

    def fake_run(argv, **_kwargs):
        commands.append(tuple(argv))
        if "show" in argv:
            return SimpleNamespace(returncode=0, stdout="oom-kill\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(containment_module.subprocess, "run", fake_run)
    assert handle.refresh_resource_status() == "oom-kill"
    evidence = evidence_for(handle, {}, {}, cleanup_confirmed=True)
    handle.close()
    handle.close()

    assert evidence.resource_exhausted is True
    assert commands == [
        (
            "systemctl", "--user", "show", "agent-loop-oom-reset.scope",
            "-p", "Result", "--value",
        ),
        ("systemctl", "--user", "reset-failed", "agent-loop-oom-reset.scope"),
    ]


def test_passing_test_gate_is_not_overridden_by_resource_evidence(tmp_path):
    evidence = ContainmentEvidence(
        backend="systemd-cgroup-v2",
        termination_cause="oom",
        cleanup_confirmed=True,
    )
    result = SimpleNamespace(outcome="passed", containment=evidence)
    _raise_for_gate_result(result, SimpleNamespace(coder_test_command_timeout_seconds=5))


def test_failed_test_gate_reports_resource_limit_and_diagnostics(tmp_path):
    evidence = ContainmentEvidence(
        backend="systemd-cgroup-v2",
        termination_cause="oom",
        applicable_limit="MemoryMax/MemorySwapMax",
        cleanup_confirmed=True,
        diagnostics=("systemd scope result=oom-kill",),
    )
    result = SimpleNamespace(
        args=["pytest", "tests/test_app.py"],
        outcome="failed",
        containment=evidence,
        output_tail="partial test output",
    )
    with pytest.raises(AgentLoopError, match="resource-exhausted") as exc_info:
        _raise_for_gate_result(
            result, SimpleNamespace(coder_test_command_timeout_seconds=5)
        )
    assert "MemoryMax/MemorySwapMax" in str(exc_info.value)
    assert "systemd scope result=oom-kill" in str(exc_info.value)


def test_gate_launch_failure_is_health_only_and_suite_failure_clears_health(tmp_path):
    memory = tmp_path / "memory"
    config = SimpleNamespace(
        dry_run=False,
        agent_memory=True,
        agent_memory_dir=memory,
        repo="owner/repo",
        coder_test_command_timeout_seconds=5,
    )
    missing = SimpleNamespace(
        cwd=tmp_path,
        args=[sys.executable, "-m", "pytest"],
        outcome="launch-failed",
        inner_exec="failed",
        suite_start="not-started",
        diagnostic="missing executable",
        output_tail="",
        elapsed_seconds=0.0,
        returncode=None,
        containment=None,
        overlap_rejected=False,
    )
    _record_gate_observation(config, missing)
    assert load_runtime_memory(memory) == []
    assert load_launcher_health(memory)[0]["state"] == "failed"

    suite_failure = SimpleNamespace(
        cwd=tmp_path,
        args=[sys.executable, "-m", "pytest"],
        outcome="failed",
        inner_exec="started",
        suite_start="verified",
        diagnostic="",
        output_tail="assertion failed",
        elapsed_seconds=0.2,
        returncode=1,
        containment=None,
        overlap_rejected=False,
    )
    _record_gate_observation(config, suite_failure)
    assert load_runtime_memory(memory)[0]["outcome"] == "failed"
    assert [row["state"] for row in load_launcher_health(memory)] == ["verified"]


def test_gate_overlap_is_neither_health_nor_timing(tmp_path):
    memory = tmp_path / "memory"
    config = SimpleNamespace(
        dry_run=False,
        agent_memory=True,
        agent_memory_dir=memory,
        repo="owner/repo",
    )
    overlap = SimpleNamespace(
        cwd=tmp_path,
        args=[sys.executable, "-c", "pass"],
        outcome="overlap-rejected",
        inner_exec="not-attempted",
        suite_start="not-started",
        diagnostic="overlap",
        output_tail="",
        elapsed_seconds=0.0,
        returncode=125,
        containment=None,
        overlap_rejected=True,
    )
    _record_gate_observation(config, overlap)
    assert load_runtime_memory(memory) == []
    assert load_launcher_health(memory) == []


def test_inner_probe_failure_precedes_containment_resource_allocation(monkeypatch, tmp_path):
    from coding_review_agent_loop.test_runtime import LauncherProbeResult

    monkeypatch.setattr(
        runner_module,
        "probe_inner_launcher",
        lambda *_args, **_kwargs: LauncherProbeResult(("pytest",), "failed", "bootstrap failed"),
    )
    monkeypatch.setattr(runner_module.os, "pipe", lambda: pytest.fail("pipes allocated before probe"))
    result = run_foreground_test(
        ["pytest", "tests"],
        cwd=tmp_path,
        timeout_seconds=5,
        parent_cgroup_path=tmp_path,
    )
    assert result.outcome == "launch-failed"
    assert result.inner_exec == "failed"
    assert result.suite_start == "not-started"


def test_popen_failure_closes_all_broker_handshake_descriptors(monkeypatch, tmp_path):
    descriptors = []
    real_pipe = runner_module.os.pipe

    def tracking_pipe():
        pair = real_pipe()
        descriptors.extend(pair)
        return pair

    monkeypatch.setattr(runner_module.os, "pipe", tracking_pipe)
    monkeypatch.setattr(
        runner_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic spawn failure")),
    )
    result = run_foreground_test(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        timeout_seconds=5,
        parent_cgroup_path=tmp_path,
    )
    assert result.outcome == "launch-failed"
    assert result.inner_exec == "failed"
    assert len(descriptors) == 4
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


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


def test_fake_systemd_memory_limit_terminates_descendant_tree(monkeypatch, tmp_path):
    """A managed-limit termination kills all descendants and is typed as OOM."""
    monkeypatch.delenv("AGENT_LOOP_INVOCATION_ID", raising=False)
    policy = default_policy(
        mode="auto", cache_dir=tmp_path / "runtime", memory_max=8 * 1024 * 1024,
        memory_high=6 * 1024 * 1024, memory_swap_max=0, tasks_max=32,
    )
    target = [sys.executable, "-c", "pass"]
    scope_argv = build_scope_argv(
        policy, target, report_path=tmp_path / "report.json",
        unit_name="agent-loop-limit.scope", limits=policy.role_limits("coder"),
    )
    assert f"MemoryMax={policy.role_limits('coder').memory_max}" in scope_argv
    assert "MemorySwapMax=0" in scope_argv
    assert "TasksMax=32" in scope_argv

    pid_file = tmp_path / "descendants.pid"
    child_code = "payload=bytearray(4 * 1024 * 1024); import time; time.sleep(30)"
    parent_code = (
        PUBLISH_SNIPPET
        + "import subprocess, sys, time\n"
        f"child_code = {child_code!r}\n"
        "children = [subprocess.Popen([sys.executable, '-c', child_code]) for _ in range(2)]\n"
        "_publish(sys.argv[1], [child.pid for child in children])\n"
        "time.sleep(30)\n"
    )
    command = [sys.executable, "-c", parent_code, str(pid_file)]
    events = []

    class FakeManagedScope:
        backend = "systemd-cgroup-v2"
        managed = True
        unit_name = "agent-loop-limit.scope"
        launcher_argv = tuple(command)
        cgroup_path = None
        capabilities = CapabilityManifest("systemd-cgroup-v2", True)
        aggregate_limits = policy.aggregate
        child_limits = policy.role_limits("coder")
        target_exec_errno = None
        diagnostics = ()
        termination_cause = None
        applicable_limit = None
        cleanup_confirmed = False

        def refresh_report(self):
            return None

        def refresh_resource_status(self):
            return None

        def terminate(self, *, signal_name="TERM"):
            events.append(f"terminate:{signal_name}")
            if signal_name == "TERM":
                self.termination_cause = "oom"
                self.applicable_limit = "MemoryMax"

        def confirm_empty(self, **_kwargs):
            events.append("confirm-empty")
            self.cleanup_confirmed = True
            return True

        def close(self):
            events.append("close")

    scope = FakeManagedScope()
    monkeypatch.setattr(
        "coding_review_agent_loop.runner.InvocationHandle.prepare",
        lambda *_args, **_kwargs: scope,
    )
    gate = readiness_gate(pid_file)
    result = run_foreground_test(
        command,
        cwd=tmp_path,
        timeout_seconds=1.0,
        containment_policy=policy,
        containment_role="coder",
        process_started=gate,
    )

    records: list[tuple[int, str]] = []
    try:
        # The gate held the watchdog until the record existed, so a missing
        # record here is a startup failure rather than a timing race.
        gate.assert_ready()
        assert pid_file.exists(), (
            "the parent did not publish its descendant record within the gate cap"
        )
        records = read_pid_record(pid_file)
        assert result.outcome == "timed_out"
        assert result.containment is not None
        assert result.containment.resource_exhausted is True
        assert result.containment.applicable_limit == "MemoryMax"
        assert events == ["terminate:TERM", "confirm-empty", "close"]

        for child_pid, child_start in records:
            if not wait_until_gone(child_pid, start_time=child_start):
                pytest.fail(f"descendant {child_pid} survived managed limit termination")
    finally:
        for child_pid, child_start in records:
            kill_if_same_instance(child_pid, child_start)


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
    applied_while_locked = []
    real_flock = containment_module.fcntl.flock
    real_open = containment_module.Path.open

    class ObservedLockFile:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            value = self.wrapped.__enter__()
            nonlocal_lock_held[0] = True
            return value

        def __exit__(self, *args):
            nonlocal_lock_held[0] = False
            return self.wrapped.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    nonlocal_lock_held = [False]

    def observe_open(path, *args, **kwargs):
        opened = real_open(path, *args, **kwargs)
        return ObservedLockFile(opened) if path.name == "leases.lock" else opened

    def observe_lock(fd, operation):
        if operation == containment_module.fcntl.LOCK_EX:
            nonlocal_lock_held[0] = True
        return real_flock(fd, operation)

    def apply(_policy, _limits):
        applied_while_locked.append(nonlocal_lock_held[0])
        return True

    monkeypatch.setattr(containment_module.fcntl, "flock", observe_lock)
    monkeypatch.setattr(containment_module.Path, "open", observe_open)
    monkeypatch.setattr(containment_module, "_apply_slice_properties", apply)
    policy = default_policy(mode="auto", cache_dir=tmp_path, memory_max=800)
    lease = AggregateLease.acquire(policy)
    lease.close()
    assert applied_while_locked == [True, True]
    assert nonlocal_lock_held == [False]


def test_concurrent_aggregate_leases_keep_the_stricter_live_ceiling(monkeypatch, tmp_path):
    applied = []
    monkeypatch.setattr(
        containment_module,
        "_apply_slice_properties",
        lambda _policy, limits: applied.append(limits) or True,
    )
    first_policy = default_policy(
        mode="auto", cache_dir=tmp_path, memory_high=600, memory_max=800, tasks_max=100
    )
    second_policy = default_policy(
        mode="auto", cache_dir=tmp_path, memory_high=300, memory_max=400, tasks_max=50
    )
    first_ready = threading.Event()
    second_ready = threading.Event()
    release = threading.Event()
    acquired = []
    errors = []

    def worker(policy, is_first):
        lease = None
        try:
            lease = AggregateLease.acquire(policy)
            acquired.append(lease.limits)
            (first_ready if is_first else second_ready).set()
            assert release.wait(5)
        except BaseException as exc:  # pragma: no cover - assertion surfaced below
            errors.append(exc)
            first_ready.set()
            second_ready.set()
        finally:
            if lease is not None:
                lease.close()

    first = threading.Thread(target=worker, args=(first_policy, True))
    second = threading.Thread(target=worker, args=(second_policy, False))
    first.start()
    assert first_ready.wait(5)
    second.start()
    assert second_ready.wait(5)
    release.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert sorted(limit.memory_max for limit in acquired) == [400, 800]
    assert any(limit.memory_max == 400 and limit.tasks_max == 50 for limit in applied)


def test_interrupt_waits_for_scope_empty_before_refusing_replacement(monkeypatch, tmp_path):
    events = []

    class ActiveScope:
        managed = True

        def terminate(self, *, signal_name="TERM"):
            events.append(f"terminate:{signal_name}")

        def confirm_empty(self, **_kwargs):
            events.append("confirm-empty")
            return events.count("confirm-empty") > 1

    policy = default_policy(mode="auto", cache_dir=tmp_path)
    runner = Runner(containment_policy=policy)
    runner._active_handles["active"] = ActiveScope()
    runner.terminate_active_processes()
    assert events == ["terminate:TERM", "confirm-empty", "terminate:KILL", "confirm-empty"]

    monkeypatch.setattr(
        runner,
        "preflight_containment",
        lambda: CapabilityManifest("process-group", False, reason="test fallback"),
    )
    with pytest.raises(AgentLoopError, match="shutting down"):
        runner._prepare_containment([sys.executable, "-c", "pass"], role="coder", env=None)


def test_timeout_terminates_descendant_process_group(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    code = (
        PUBLISH_SNIPPET
        + "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "_publish(sys.argv[1], [child.pid])\n"
        "time.sleep(30)\n"
    )
    gate = readiness_gate(pid_file)
    result = run_foreground_test(
        [sys.executable, "-c", code, str(pid_file)],
        cwd=tmp_path,
        timeout_seconds=0.5,
        containment_policy=default_policy(mode="off", cache_dir=tmp_path / "runtime"),
        process_started=gate,
    )
    records: list[tuple[int, str]] = []
    try:
        gate.assert_ready()
        assert pid_file.exists(), "the descendant pid was not recorded before cleanup"
        records = read_pid_record(pid_file)
        assert result.outcome == "timed_out"
        for child_pid, child_start in records:
            if not wait_until_gone(child_pid, start_time=child_start):
                pytest.fail(f"descendant {child_pid} survived process-group timeout")
    finally:
        for child_pid, child_start in records:
            kill_if_same_instance(child_pid, child_start)


def test_readiness_gate_holds_watchdog_until_descendant_record_published(tmp_path):
    """Publication after the watchdog deadline still precedes termination."""
    pid_file = tmp_path / "late.pid"
    code = (
        PUBLISH_SNIPPET
        + "import os, sys, time\n"
        "time.sleep(2.0)\n"
        "_publish(sys.argv[1], [os.getpid()])\n"
        "time.sleep(60)\n"
    )
    gate = readiness_gate(pid_file)
    result = run_foreground_test(
        [sys.executable, "-c", code, str(pid_file)],
        cwd=tmp_path,
        timeout_seconds=0.3,
        containment_policy=default_policy(mode="off", cache_dir=tmp_path / "runtime"),
        process_started=gate,
    )
    try:
        gate.assert_ready()
        assert gate.release_reason == "published"
        # An unheld watchdog would kill the parent at 0.3s, long before it
        # publishes at 2.0s, so a surviving record proves the gate held it.
        assert pid_file.exists()
        records = read_pid_record(pid_file)
        assert [pid for pid, _ in records] == [gate.pid]
        assert gate.waited_seconds >= 2.0
        assert result.elapsed_seconds >= 2.0
        assert result.outcome == "timed_out"
        assert wait_until_gone(gate.pid, start_time=gate.start_time)
    finally:
        kill_if_same_instance(gate.pid, gate.start_time)


def test_readiness_gate_cap_expiry_reports_startup_failure_and_cleans_up(tmp_path):
    """A parent that never publishes yields a bounded, controlled readiness failure."""
    pid_file = tmp_path / "never.pid"
    observed = tmp_path / "observed.pid"
    code = (
        PUBLISH_SNIPPET
        + "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "_publish(sys.argv[2], [child.pid])\n"
        "time.sleep(30)\n"
        "_publish(sys.argv[1], [child.pid])\n"
    )
    gate = readiness_gate(pid_file, cap=0.5)
    started = time.monotonic()
    result = run_foreground_test(
        [sys.executable, "-c", code, str(pid_file), str(observed)],
        cwd=tmp_path,
        timeout_seconds=0.2,
        containment_policy=default_policy(mode="off", cache_dir=tmp_path / "runtime"),
        process_started=gate,
    )
    elapsed = time.monotonic() - started
    descendants: list[tuple[int, str]] = []
    try:
        assert gate.ready is False
        assert gate.release_reason == "cap-expired"
        assert str(pid_file) in gate.failure_reason
        assert "0.5s readiness cap" in gate.failure_reason
        assert gate.cleanup_invocations == 1
        assert elapsed < 20.0
        assert result.outcome != "passed"
        with pytest.raises(Failed, match="startup/readiness failed"):
            gate.assert_ready()

        # The parent claim is unconditional; the descendant claim holds only
        # when the parent reached its separate observation record in time.
        assert wait_until_gone(gate.pid, start_time=gate.start_time)
        if observed.exists():
            descendants = read_pid_record(observed)
            for child_pid, child_start in descendants:
                if not wait_until_gone(child_pid, start_time=child_start):
                    pytest.fail(f"descendant {child_pid} survived readiness cleanup")
        # Supplementary only: an orphaned zombie can hold the group id until
        # init reaps it, so this bounded poll never stands alone as proof.
        if not wait_group_gone(gate.pid, timeout=10.0):
            pytest.fail(f"process group {gate.pid} lingered after readiness cleanup")
    finally:
        kill_if_same_instance(gate.pid, gate.start_time)
        for child_pid, child_start in descendants:
            kill_if_same_instance(child_pid, child_start)


def test_readiness_gate_cap_expiry_kills_a_sigterm_ignoring_descendant(tmp_path):
    """Cap-expiry cleanup escalates to SIGKILL when a descendant ignores TERM."""
    pid_file = tmp_path / "never.pid"
    observed = tmp_path / "observed.pid"
    # The descendant only ignores SIGTERM once its handler is installed, so the
    # parent waits for its readiness marker before publishing the record the
    # test reads; otherwise SIGTERM would win the startup race and the SIGKILL
    # escalation would never be exercised.
    child_code = (
        "import signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').close(); "
        "time.sleep(60)"
    )
    ready = tmp_path / "descendant.ready"
    code = (
        PUBLISH_SNIPPET
        + "import pathlib, subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}, sys.argv[3]])\n"
        "while not pathlib.Path(sys.argv[3]).exists():\n"
        "    time.sleep(0.01)\n"
        "_publish(sys.argv[2], [child.pid])\n"
        "time.sleep(30)\n"
        "_publish(sys.argv[1], [child.pid])\n"
    )
    gate = readiness_gate(pid_file, cap=3.0)
    started = time.monotonic()
    result = run_foreground_test(
        [sys.executable, "-c", code, str(pid_file), str(observed), str(ready)],
        cwd=tmp_path,
        timeout_seconds=0.2,
        containment_policy=default_policy(mode="off", cache_dir=tmp_path / "runtime"),
        process_started=gate,
    )
    elapsed = time.monotonic() - started
    descendants: list[tuple[int, str]] = []
    try:
        assert gate.ready is False
        assert gate.release_reason == "cap-expired"
        assert gate.cleanup_invocations == 1
        assert elapsed < 30.0
        assert result.outcome != "passed"
        assert wait_until_gone(gate.pid, start_time=gate.start_time)
        # The parent obeys SIGTERM, so only group-driven escalation can reach a
        # descendant that ignores it.  The claim stays conditional on the
        # separately published record, which the cap does not guarantee.
        if observed.exists():
            descendants = read_pid_record(observed)
            for child_pid, child_start in descendants:
                if not wait_until_gone(child_pid, start_time=child_start):
                    pytest.fail(
                        f"SIGTERM-ignoring descendant {child_pid} survived readiness cleanup"
                    )
    finally:
        kill_if_same_instance(gate.pid, gate.start_time)
        for child_pid, child_start in descendants:
            kill_if_same_instance(child_pid, child_start)


def test_production_proc_helpers_treat_process_lookup_error_as_gone(monkeypatch):
    """Production /proc inspection never raises when the entry vanishes."""
    real_read_text = Path.read_text

    def read_text(self, *args, **kwargs):
        if str(self).startswith("/proc/"):
            raise ProcessLookupError(errno.ESRCH, "No such process")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(containment_module.Path, "read_text", read_text)
    pid = os.getpid()
    assert containment_module._start_identity(pid) == "dead"
    # A vanished /proc entry falls through to os.kill, which still sees self.
    assert containment_module._pid_alive(pid) is True
    assert containment_module.cgroup_path_for_pid(pid) is None


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
