"""Unit tests for the containment-aware test-worker budget (issue #848)."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from coding_review_agent_loop import containment
from coding_review_agent_loop.containment import ResourceLimits
from coding_review_agent_loop.test_runtime import normalize_test_command
from coding_review_agent_loop.test_workers import (
    CAVEAT_EXCEEDED,
    CAVEAT_MIXED,
    CAVEAT_NESTED,
    CAVEAT_NOT_OBSERVED,
    CAVEAT_PARTIAL,
    CAVEAT_REFUSED_SESSION,
    CAVEAT_REMOTE,
    CAVEAT_UNVERIFIED,
    ENV_TEST_WORKER_ENFORCEMENT,
    ENV_TEST_WORKERS,
    ENV_WORKER_CAP_NESTED,
    ENV_WORKER_CAP_SPEC,
    ENV_XDIST_AUTO,
    MANAGED_BACKEND,
    PLUGIN_MODULE,
    AncestryLimits,
    WorkerBudget,
    WorkerBudgetError,
    WorkerBudgetLock,
    analyze_worker_report,
    apply_worker_budget,
    argv_only_workers_label,
    classify_command,
    client_refusal,
    derive_worker_budget,
    detect_parallel_support,
    expected_workers_label,
    parse_worker_count,
    parse_worker_memory,
    plugin_directory,
    probe_cpu_count,
    read_ancestry_limits,
    render_worker_guidance,
    resolve_worker_budget,
    row_workers_label,
    worker_budget_lock_root,
    worker_lane_identity,
)

GIB = 1024 ** 3
NO_ANCESTRY = AncestryLimits()


def _cpu(count: int):
    return lambda: (count, "affinity")


def _mem(value):
    return lambda: value


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def test_post_admission_budget_uses_lease_restricted_handle_limits():
    configured = ResourceLimits(12 * GIB, None, 0, None)
    lease_restricted = ResourceLimits(6 * GIB, None, 0, None)
    child = ResourceLimits(5 * GIB, None, 0, None)
    budget = derive_worker_budget(
        backend=MANAGED_BACKEND,
        child_limits=child.restrict(lease_restricted),
        aggregate_limits=lease_restricted,
        cpu_reader=_cpu(8),
        memory_reader=_mem(64 * GIB),
        ancestry=NO_ANCESTRY,
    )
    # min(child 5G, aggregate 6G, host) - 1G reserve = 4 workers of 1G.
    assert budget.workers == 4
    assert budget.limiting_factor == "child-limit"
    assert budget.enforced_ceiling is True
    configured_budget = derive_worker_budget(
        backend=MANAGED_BACKEND,
        child_limits=None,
        aggregate_limits=configured,
        cpu_reader=_cpu(8),
        memory_reader=_mem(64 * GIB),
        ancestry=NO_ANCESTRY,
    )
    assert configured_budget.workers == 8  # cpu-bound; the lease is what lowers it
    assert budget.workers <= 8


def test_operator_scope_ancestry_limit_sizes_budget():
    ancestry = AncestryLimits(memory=(("user.slice:memory.high", 9 * GIB),))
    budget = derive_worker_budget(
        backend="process-group", cpu_reader=_cpu(8), memory_reader=_mem(31 * GIB), ancestry=ancestry,
    )
    # floor((9G - 1G reserve) / 1G) = 8 = the CPU term.
    assert budget.workers == 8
    assert budget.enforced_ceiling is True
    tighter = derive_worker_budget(
        backend="process-group", cpu_reader=_cpu(8),
        memory_reader=_mem(31 * GIB), ancestry=AncestryLimits(memory=(("x:memory.max", 4 * GIB),)),
    )
    assert tighter.workers == 3
    assert tighter.limiting_factor == "ancestry-limit"


def test_unmanaged_backend_ignores_policy_limits():
    small = ResourceLimits(2 * GIB, None, 0, None)
    budget = derive_worker_budget(
        backend="process-group",
        child_limits=small,
        aggregate_limits=small,
        cpu_reader=_cpu(4),
        memory_reader=_mem(64 * GIB),
        ancestry=NO_ANCESTRY,
    )
    assert budget.workers == 4
    assert budget.enforced_ceiling is False
    assert "no agent-loop memory ceiling enforced" in budget.describe()


def test_memory_unknown_fallback():
    budget = derive_worker_budget(
        backend="process-group", cpu_reader=_cpu(8), memory_reader=_mem(None), ancestry=NO_ANCESTRY,
    )
    assert budget.workers == 4
    assert budget.limiting_factor == "memory-unknown"
    assert budget.inputs["host_memory"] == "unreadable"
    single = derive_worker_budget(
        backend="process-group", cpu_reader=_cpu(1), memory_reader=_mem(None), ancestry=NO_ANCESTRY,
    )
    assert single.workers == 1


def test_limit_above_host_memory_uses_host_candidate():
    # 16G host with 10% headroom -> 14.4G usable; managed child 24G.
    budget = derive_worker_budget(
        backend=MANAGED_BACKEND,
        child_limits=ResourceLimits(24 * GIB, None, 0, None),
        os_headroom_percent=10,
        cpu_reader=_cpu(64),
        memory_reader=_mem(16 * GIB),
        ancestry=AncestryLimits(memory=(("x:memory.max", 32 * GIB),)),
    )
    assert budget.limiting_factor == "host-memory"
    assert budget.workers == int((int(16 * GIB * 0.9) - GIB) // GIB)
    assert budget.enforced_ceiling is True


def test_cpu_quota_and_affinity_never_exceeded():
    budget = derive_worker_budget(
        backend="process-group",
        cpu_reader=_cpu(8),
        memory_reader=_mem(256 * GIB),
        ancestry=AncestryLimits(cpu_quotas=(3,)),
    )
    assert budget.workers == 3
    assert budget.limiting_factor == "cpu-quota"


def test_per_worker_memory_estimate_lowers_budget():
    budget = derive_worker_budget(
        backend="process-group",
        per_worker_bytes=parse_worker_memory("3GiB"),
        cpu_reader=_cpu(16),
        memory_reader=_mem(16 * GIB),
        os_headroom_percent=25,
        ancestry=NO_ANCESTRY,
    )
    assert budget.workers == (12 * GIB - GIB) // (3 * GIB)


def test_ancestry_reader_touches_only_allowed_files(tmp_path):
    root = tmp_path / "cg"
    leaf = root / "user.slice" / "app.scope"
    leaf.mkdir(parents=True)
    (root / "user.slice" / "memory.high").write_text("9663676416\n")
    (root / "user.slice" / "memory.max").write_text("max\n")
    (leaf / "cpu.max").write_text("200000 100000\n")
    (leaf / "memory.current").write_text("1\n")
    proc = tmp_path / "self-cgroup"
    proc.write_text("0::/user.slice/app.scope\n")
    seen: list[Path] = []

    def reader(path: Path):
        seen.append(path)
        try:
            return path.read_text()
        except OSError:
            return None

    limits = read_ancestry_limits(cgroup_reader=reader, proc_cgroup=proc, cgroup_root=root)
    assert limits.cpu_quotas == (2,)
    assert [value for _label, value in limits.memory] == [9663676416]
    assert seen[0] == proc
    assert {path.name for path in seen[1:]} <= {"cpu.max", "memory.high", "memory.max"}
    for path in seen[1:]:
        assert root in path.parents or path.parent == root


def test_cpu_probe_fallbacks(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert probe_cpu_count() == (6, "cpu-count")
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert probe_cpu_count() == (1, "fallback-1")
    monkeypatch.setattr(os, "cpu_count", lambda: 0)
    assert probe_cpu_count() == (1, "fallback-1")

    def raising(_pid):
        raise OSError("nope")

    monkeypatch.setattr(os, "sched_getaffinity", raising, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    assert probe_cpu_count() == (3, "cpu-count")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(), raising=False)
    assert probe_cpu_count() == (3, "cpu-count")
    budget = derive_worker_budget(backend="process-group", memory_reader=_mem(None), ancestry=NO_ANCESTRY)
    assert budget.inputs["cpu_source"] == "cpu-count"


def test_containment_off_macos_like_host_never_raises(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(containment, "probe_host_memory_bytes", lambda: None)
    budget = derive_worker_budget(backend="process-group", cgroup_reader=lambda _p: None)
    assert budget.workers >= 1
    assert budget.limiting_factor == "memory-unknown"


# ---------------------------------------------------------------------------
# Precedence and validation
# ---------------------------------------------------------------------------


def _derived(workers: int = 4) -> WorkerBudget:
    return WorkerBudget(workers, "derived", "clamp", "cpu", {}, False)


@pytest.mark.parametrize("value", [8, 2])
def test_loop_and_standalone_override_replaces_derivation(value):
    resolved = resolve_worker_budget(_derived(4), operator_workers=value, env={}, has_parent=False)
    assert resolved.budget.workers == value
    assert resolved.budget.source == "operator"
    assert "operator-supplied" in resolved.budget.describe()


def test_child_may_only_lower_or_tighten():
    env = {ENV_TEST_WORKERS: "4", ENV_TEST_WORKER_ENFORCEMENT: "clamp"}
    assert resolve_worker_budget(_derived(16), operator_workers=8, env=env).budget.workers == 4
    assert resolve_worker_budget(_derived(16), operator_workers=2, env=env).budget.workers == 2
    assert resolve_worker_budget(_derived(16), operator_enforcement="off", env=env).budget.enforcement == "clamp"
    assert resolve_worker_budget(_derived(16), operator_enforcement="refuse", env=env).budget.enforcement == "refuse"
    # Local derivation (ancestry) may lower the inherited value too.
    assert resolve_worker_budget(_derived(3), env=env).budget.workers == 3


def test_unparseable_inherited_value_fails_closed():
    resolved = resolve_worker_budget(_derived(8), env={ENV_TEST_WORKERS: "lots"})
    assert resolved.budget.workers == 1
    assert resolved.diagnostics


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "x", True])
def test_worker_count_validation(value):
    with pytest.raises(WorkerBudgetError):
        parse_worker_count(value)


@pytest.mark.parametrize("value", ["0", "max", "infinity", "-1G"])
def test_worker_memory_validation(value):
    with pytest.raises(WorkerBudgetError):
        parse_worker_memory(value)


# ---------------------------------------------------------------------------
# Classification and injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["pytest", "-n", "4"], "direct-pytest"),
        (["python3", "-m", "pytest", "tests"], "direct-pytest"),
        (["/usr/bin/python3", "-B", "-m", "pytest"], "direct-pytest"),
        (["env", "A=1", "timeout", "60", "pytest"], "direct-pytest"),
        (["nice", "-n", "5", "py.test"], "direct-pytest"),
        (["make", "test"], "other"),
        (["./run-tests.sh"], "other"),
        (["tox", "-e", "py"], "other"),
        (["npm", "test"], "other"),
        (["go", "test", "./..."], "other"),
        (["agent-loop", "run-tests", "--", "pytest"], "other"),
    ],
)
def test_command_classification(argv, expected):
    assert classify_command(argv).command_class == expected


def _budget(workers=3, mode="clamp"):
    return WorkerBudget(workers, "inherited", mode, "cpu", {}, True)


def test_direct_pytest_injection_clamp(tmp_path):
    decision = apply_worker_budget(
        ["python3", "-m", "pytest", "--numprocesses=12", "tests"],
        {"PYTHONPATH": "src", "PYTEST_XDIST_AUTO_NUM_WORKERS": "8", ENV_WORKER_CAP_NESTED: "/x"},
        tmp_path, _budget(4),
    )
    try:
        assert decision.command_class == "direct-pytest"
        assert decision.argv == ("python3", "-m", "pytest", "-p", PLUGIN_MODULE, "--numprocesses=12", "tests")
        env = decision.env
        assert env["PYTHONPATH"].split(os.pathsep) == [str(plugin_directory()), "src"]
        assert env["PYTEST_PLUGINS"] == PLUGIN_MODULE
        assert env[ENV_XDIST_AUTO] == "4"
        assert env[ENV_TEST_WORKERS] == "4" and env[ENV_TEST_WORKER_ENFORCEMENT] == "clamp"
        assert ENV_WORKER_CAP_NESTED not in env
        spec = json.loads(env[ENV_WORKER_CAP_SPEC])
        assert spec == {"version": 1, "budget": 4, "mode": "clamp", "report": str(decision.report_path)}
    finally:
        decision.cleanup()


def test_refuse_mode_keeps_caller_auto_cap_and_never_refuses_worker_values(tmp_path):
    for argv in (["pytest", "-n", "8"], ["pytest", "-n", "8", "--maxprocesses=2"], ["pytest", "-n", "auto"]):
        decision = apply_worker_budget(argv, {ENV_XDIST_AUTO: "6"}, tmp_path, _budget(2, "refuse"))
        try:
            assert decision.refused is None
            assert decision.env[ENV_XDIST_AUTO] == "6"
            assert decision.argv[:3] == ("pytest", "-p", PLUGIN_MODULE)
        finally:
            decision.cleanup()
    decision = apply_worker_budget(["pytest", "-n", "8"], {}, tmp_path, _budget(2, "refuse"))
    assert ENV_XDIST_AUTO not in decision.env
    decision.cleanup()


def test_plugin_disable_token_stripped_in_clamp_refused_in_refuse(tmp_path):
    argv = ["pytest", "-n", "8", "-p", f"no:{PLUGIN_MODULE}"]
    clamp = apply_worker_budget(argv, {}, tmp_path, _budget(2))
    assert f"no:{PLUGIN_MODULE}" not in clamp.argv and clamp.notices
    clamp.cleanup()
    refuse = apply_worker_budget(argv, {}, tmp_path, _budget(2, "refuse"))
    assert refuse.refused
    assert client_refusal(argv, _budget(2, "refuse"))
    assert client_refusal(argv, _budget(2, "clamp")) is None
    assert client_refusal(["pytest", "-n", "8"], _budget(2, "refuse")) is None


@pytest.mark.parametrize("argv", [["make", "test"], ["./script.sh"], ["tox", "-e", "py"], ["npm", "test"], ["go", "test", "./..."]])
def test_other_commands_keep_argv_and_get_identical_env(tmp_path, argv):
    decision = apply_worker_budget(argv, {"PATH": "/usr/bin"}, tmp_path, _budget(2))
    try:
        assert decision.argv == tuple(argv)
        assert decision.command_class == "other"
        assert decision.env["PYTEST_PLUGINS"] == PLUGIN_MODULE
        assert ENV_WORKER_CAP_SPEC in decision.env
    finally:
        decision.cleanup()


def test_off_mode_injects_nothing(tmp_path):
    env = {"PYTEST_ADDOPTS": "-n 8", ENV_XDIST_AUTO: "6"}
    decision = apply_worker_budget(["pytest", "-n", "8"], env, tmp_path, _budget(2, "off"))
    assert decision.argv == ("pytest", "-n", "8")
    assert decision.report_path is None
    assert "PYTEST_PLUGINS" not in decision.env
    assert decision.env["PYTEST_ADDOPTS"] == "-n 8" and decision.env[ENV_XDIST_AUTO] == "6"
    assert decision.env[ENV_TEST_WORKERS] == "2" and decision.env[ENV_TEST_WORKER_ENFORCEMENT] == "off"


def test_inline_env_segments_cannot_disable_plugin(tmp_path):
    budget = _budget(3)
    forged = apply_worker_budget(
        ["env", f"{ENV_WORKER_CAP_SPEC}=forged", f"{ENV_TEST_WORKERS}=16",
         f"{ENV_TEST_WORKER_ENFORCEMENT}=off", "pytest", "-n", "12"],
        {}, tmp_path, budget,
    )
    head = forged.argv.index("pytest")
    assigned = dict(token.split("=", 1) for token in forged.argv[:head] if "=" in token)
    assert assigned[ENV_TEST_WORKERS] == "3"
    assert assigned[ENV_TEST_WORKER_ENFORCEMENT] == "clamp"
    assert json.loads(assigned[ENV_WORKER_CAP_SPEC])["budget"] == 3
    forged.cleanup()

    auto = apply_worker_budget(["env", f"{ENV_XDIST_AUTO}=99", "pytest", "-n", "auto"], {}, tmp_path, budget)
    assert f"{ENV_XDIST_AUTO}=3" in auto.argv
    auto.cleanup()
    auto_refuse = apply_worker_budget(
        ["env", f"{ENV_XDIST_AUTO}=99", "pytest", "-n", "auto"], {}, tmp_path, _budget(3, "refuse")
    )
    assert f"{ENV_XDIST_AUTO}=99" in auto_refuse.argv
    auto_refuse.cleanup()

    unset = apply_worker_budget(["env", "-u", "PYTHONPATH", "pytest", "-n", "8"], {}, tmp_path, budget)
    index = unset.argv.index("pytest")
    assert unset.argv[3].startswith("PYTHONPATH=") and index > 3
    unset.cleanup()

    cleared = apply_worker_budget(["env", "-i", "PATH=/usr/bin", "pytest", "-n", "8"], {}, tmp_path, budget)
    names = {token.split("=", 1)[0] for token in cleared.argv if "=" in token}
    assert {"PYTHONPATH", "PYTEST_PLUGINS", ENV_WORKER_CAP_SPEC, ENV_XDIST_AUTO} <= names
    cleared.cleanup()

    plugins = apply_worker_budget(["env", "PYTEST_PLUGINS=", "pytest", "-n", "8"], {}, tmp_path, budget)
    assert f"PYTEST_PLUGINS={PLUGIN_MODULE}" in plugins.argv
    plugins.cleanup()

    nested = apply_worker_budget(["env", f"{ENV_WORKER_CAP_NESTED}=/r", "pytest"], {}, tmp_path, budget)
    assert not any(token.startswith(ENV_WORKER_CAP_NESTED) for token in nested.argv)
    nested.cleanup()


# ---------------------------------------------------------------------------
# Lane identity
# ---------------------------------------------------------------------------


def test_lane_identity_is_mode_and_budget_independent(tmp_path):
    spellings = [
        ["pytest", "-n", "8"],
        ["pytest", "-n", "12"],
        ["pytest", "--numprocesses=12"],
        ["pytest", "-n", "auto"],
        ["pytest", "-n", "1"],
        ["pytest", "-n", "8", "--dist", "load"],
        ["env", f"{ENV_WORKER_CAP_SPEC}=x", "pytest", "-n", "12"],
    ]
    identities = {worker_lane_identity(argv, cwd=tmp_path) for argv in spellings}
    assert len(identities) == 1
    for argv in (["pytest", "tests/"], ["make", "test"], ["python3", "-m", "pytest", "-q"]):
        assert worker_lane_identity(argv, cwd=tmp_path) == normalize_test_command(argv, cwd=tmp_path)
    # Two effective decisions with different report paths never change it.
    one = apply_worker_budget(["pytest", "-n", "8"], {}, tmp_path, _budget(2))
    two = apply_worker_budget(["pytest", "-n", "8"], {}, tmp_path, _budget(3, "refuse"))
    assert one.report_path != two.report_path
    assert worker_lane_identity(["pytest", "-n", "8"], cwd=tmp_path) not in {
        normalize_test_command(one.argv, cwd=tmp_path), normalize_test_command(two.argv, cwd=tmp_path)
    }
    one.cleanup()
    two.cleanup()


# ---------------------------------------------------------------------------
# Worker-budget lock
# ---------------------------------------------------------------------------


def test_worker_budget_lock_is_per_invocation(tmp_path):
    root = tmp_path / "locks"
    first, problem = WorkerBudgetLock.acquire(invocation_id="inv", cwd=tmp_path / "a", root=root)
    assert first is not None and problem is None
    second, problem = WorkerBudgetLock.acquire(invocation_id="inv", cwd=tmp_path / "b", root=root)
    assert second is None and problem is None
    other, _ = WorkerBudgetLock.acquire(invocation_id="other", cwd=tmp_path, root=root)
    assert other is not None
    first.close()
    again, _ = WorkerBudgetLock.acquire(invocation_id="inv", cwd=tmp_path, root=root)
    assert again is not None
    again.close()
    other.close()
    standalone_a, _ = WorkerBudgetLock.acquire(invocation_id=None, cwd=tmp_path / "a", root=root)
    standalone_b, _ = WorkerBudgetLock.acquire(invocation_id=None, cwd=tmp_path / "b", root=root)
    assert standalone_a is not None and standalone_b is not None
    standalone_a.close()
    standalone_b.close()


def test_worker_budget_lock_rejects_unsafe_root(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    os.chmod(root, 0o777)
    lock, problem = WorkerBudgetLock.acquire(invocation_id="inv", cwd=tmp_path, root=root)
    assert lock is None and "writable" in problem


def test_worker_budget_lock_root_reads_no_environment(monkeypatch):
    class Recording(dict):
        reads: list[str] = []

        def get(self, key, default=None):
            self.reads.append(key)
            return super().get(key, default)

        def __getitem__(self, key):
            self.reads.append(key)
            return super().__getitem__(key)

    recording = Recording(os.environ)
    monkeypatch.setattr(os, "environ", recording)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/tmp/elsewhere")
    Recording.reads.clear()
    root = worker_budget_lock_root()
    assert "XDG_RUNTIME_DIR" not in Recording.reads
    info = root.stat()
    assert info.st_uid == os.getuid()
    assert not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def test_worker_budget_lock_root_falls_back_to_tmp(monkeypatch):
    import coding_review_agent_loop.test_workers as module

    real_path = module.Path

    def fake_path(value, *rest):
        text = str(value)
        if text.startswith("/run/user/"):
            return real_path("/nonexistent-run-user") / text.rsplit("/", 1)[-1]
        return real_path(value, *rest)

    monkeypatch.setattr(module, "Path", fake_path)
    root = worker_budget_lock_root()
    assert str(root).startswith(f"/tmp/coding-review-agent-loop-{os.getuid()}/")


# ---------------------------------------------------------------------------
# Report analysis and cohorts
# ---------------------------------------------------------------------------


def _report(tmp_path: Path, *rows: dict) -> Path:
    path = tmp_path / "report.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "not json\n", encoding="utf-8")
    return path


def _confirmed(session="s1", effective=2, remote=0, action="unchanged", planned=None):
    return {"kind": "confirmed", "session": session, "effective": effective, "remote": remote,
            "action": action, "planned": planned, "requested": 2}


def test_direct_single_confirmed_session_gives_numeric_cohort(tmp_path):
    analysis = analyze_worker_report(_report(tmp_path, _confirmed(effective=2)), command_class="direct-pytest", mode="clamp")
    assert analysis.workers_cohort == "2" and analysis.enforcement == "unchanged"
    serial = analyze_worker_report(_report(tmp_path, _confirmed(effective=0)), command_class="direct-pytest", mode="clamp")
    assert serial.workers_cohort == "serial"


def test_direct_single_refused_session_is_configuration_refusal(tmp_path):
    report = _report(tmp_path, {"kind": "refused", "session": "s1", "stage": "cmdline", "requested": 8, "budget": 2})
    analysis = analyze_worker_report(report, command_class="direct-pytest", mode="refuse")
    assert analysis.direct_refusal and analysis.enforcement == "refused"


@pytest.mark.parametrize(
    "rows",
    [
        [{"kind": "refused", "session": "s1"}, {"kind": "refused", "session": "s2"}],
        [{"kind": "refused", "session": "s1"}, _confirmed("s1")],
        [{"kind": "refused", "session": "s1"}, {"kind": "nested", "where": "child"}],
    ],
)
def test_direct_refusal_without_proof_keeps_evidence(tmp_path, rows):
    analysis = analyze_worker_report(_report(tmp_path, *rows), command_class="direct-pytest", mode="refuse")
    assert not analysis.direct_refusal
    assert analysis.enforcement == "refused-in-command"
    assert CAVEAT_REFUSED_SESSION in analysis.caveats
    assert analysis.workers_cohort == "unknown"


def test_other_command_is_always_unknown(tmp_path):
    analysis = analyze_worker_report(_report(tmp_path, _confirmed(effective=2)), command_class="other", mode="clamp")
    assert analysis.workers_cohort == "unknown"
    assert CAVEAT_PARTIAL in analysis.caveats
    refused = analyze_worker_report(
        _report(tmp_path, _confirmed("s1"), {"kind": "refused", "session": "s2"}), command_class="other", mode="refuse"
    )
    assert refused.enforcement == "refused-in-command" and not refused.direct_refusal


def test_missing_report_unverified_or_not_observed(tmp_path):
    direct = analyze_worker_report(tmp_path / "missing", command_class="direct-pytest", mode="clamp")
    assert direct.enforcement == "unverified" and CAVEAT_UNVERIFIED in direct.caveats and direct.notices
    other = analyze_worker_report(tmp_path / "missing", command_class="other", mode="clamp")
    assert other.enforcement == "not-observed" and CAVEAT_NOT_OBSERVED in other.caveats and not other.notices


def test_caveats_for_mixed_remote_exceeded_nested_and_decision_only(tmp_path):
    mixed = analyze_worker_report(
        _report(tmp_path, _confirmed("a", 1), _confirmed("b", 2)), command_class="other", mode="clamp"
    )
    assert CAVEAT_MIXED in mixed.caveats and mixed.workers_cohort == "unknown"
    remote = analyze_worker_report(_report(tmp_path, _confirmed(remote=1)), command_class="direct-pytest", mode="clamp")
    assert CAVEAT_REMOTE in remote.caveats and remote.workers_cohort == "unknown"
    exceeded = analyze_worker_report(
        _report(tmp_path, _confirmed(effective=3, action="exceeded", planned=2)), command_class="direct-pytest", mode="clamp"
    )
    assert exceeded.enforcement == "exceeded" and CAVEAT_EXCEEDED in exceeded.caveats
    assert exceeded.workers_cohort == "unknown"
    nested = analyze_worker_report(
        _report(tmp_path, _confirmed(), {"kind": "nested", "where": "in-process"}), command_class="direct-pytest", mode="clamp"
    )
    assert CAVEAT_NESTED in nested.caveats and nested.workers_cohort == "unknown"
    assert nested.enforcement == "unchanged"
    decision_only = analyze_worker_report(
        _report(tmp_path, {"kind": "decision", "session": "s1", "planned": 2}), command_class="direct-pytest", mode="clamp"
    )
    assert decision_only.enforcement == "unverified" and CAVEAT_UNVERIFIED in decision_only.caveats


@pytest.mark.parametrize(
    "argv, label",
    [
        (["pytest", "-p", "no:xdist"], "serial"),
        (["python", "-m", "pytest", "-p", "no:xdist", "tests"], "serial"),
        (["pytest", "-p", "no:xdist", "-p", "xdist"], "unknown"),
        (["pytest", "-n", "0"], "unknown"),
        (["pytest", "-n", "4"], "unknown"),
        (["pytest", "-n", "8", "--maxprocesses=2"], "unknown"),
        (["pytest"], "unknown"),
        (["pytest", "-n", "auto"], "unknown"),
        (["pytest", "--tx", "3*popen"], "unknown"),
        (["make", "test"], "unknown"),
    ],
)
def test_argv_only_labels(argv, label):
    assert argv_only_workers_label(argv) == label
    assert row_workers_label({"normalized_command": " ".join(argv)}) == label


def test_row_label_prefers_recorded_workers():
    assert row_workers_label({"workers": "2", "normalized_command": "pytest"}) == "2"


def test_expected_labels_for_lookup():
    assert expected_workers_label(["pytest", "-n", "8"], budget=2, mode="clamp") == "2"
    assert expected_workers_label(["pytest"], budget=2, mode="clamp") == "serial"
    assert expected_workers_label(["pytest", "-n", "auto"], budget=2, mode="clamp") == "unknown"
    assert expected_workers_label(["make"], budget=2, mode="clamp") == "unknown"
    assert expected_workers_label(["pytest", "-n", "4"], budget=2, mode="off") == "unknown"


# ---------------------------------------------------------------------------
# Detection and prompt text
# ---------------------------------------------------------------------------


def test_parallel_detection_is_bounded_and_text_only(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["pytest-xdist>=3"]\n')
    assert detect_parallel_support(tmp_path)
    readme_only = tmp_path / "b"
    readme_only.mkdir()
    (readme_only / "README.md").write_text("Run:\n\n```bash\npytest -n 6\n```\n")
    assert detect_parallel_support(readme_only)
    prose = tmp_path / "c"
    prose.mkdir()
    (prose / "README.md").write_text("We could use pytest-xdist and pytest -n 6 someday.\n")
    assert not detect_parallel_support(prose)
    empty = tmp_path / "d"
    empty.mkdir()
    assert not detect_parallel_support(empty)
    workflow = tmp_path / "e" / ".github" / "workflows"
    workflow.mkdir(parents=True)
    (workflow / "ci.yml").write_text("steps:\n  - run: python -m pytest -n auto\n")
    assert detect_parallel_support(tmp_path / "e")


@pytest.mark.parametrize(
    "mode, sentence",
    [("clamp", "lowered to the budget"), ("refuse", "refused and the run does not execute"), ("off", "advisory")],
)
def test_guidance_sentence_matches_mode(mode, sentence):
    budget = WorkerBudget(3, "derived", mode, "cpu", {}, False)
    text = render_worker_guidance(budget, parallel_supported=True)
    assert sentence in text and "$AGENT_LOOP_TEST_WORKERS" in text
    assert "keep focused" in text
    if mode != "clamp":
        assert "lowered" not in text
    fallback = render_worker_guidance(budget, parallel_supported=False)
    assert "do not add parallel" in fallback


# ---------------------------------------------------------------------------
# Plugin module helpers and packaging
# ---------------------------------------------------------------------------


def _load_plugin(monkeypatch):
    monkeypatch.delenv(ENV_WORKER_CAP_SPEC, raising=False)
    monkeypatch.delenv(ENV_WORKER_CAP_NESTED, raising=False)
    path = plugin_directory() / f"{PLUGIN_MODULE}.py"
    spec = importlib.util.spec_from_file_location("_agent_loop_worker_cap_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plugin_spec_counting_table(monkeypatch):
    module = _load_plugin(monkeypatch)
    specs = ["popen", "execmodel=main_thread_only//popen", "socket=127.0.0.1:1", "ssh=host", "vagrant_ssh=box", "weird=x"]
    assert module.count_specs(specs) == (6, 4)
    assert module.count_specs(["socket=a", "socket=b", "socket=c"]) == (3, 3)


def test_plugin_uses_only_new_style_wrappers_and_is_stdlib_only(monkeypatch):
    source = (plugin_directory() / f"{PLUGIN_MODULE}.py").read_text(encoding="utf-8")
    assert "hookwrapper=True" not in source
    assert "coding_review_agent_loop" not in source.replace("``coding_review_agent_loop``", "")
    module = _load_plugin(monkeypatch)
    assert module._ARMED is None


def test_plugin_defines_no_hooks_on_old_pluggy(monkeypatch):
    import pluggy

    monkeypatch.setattr(pluggy, "__version__", "1.0.0")
    module = _load_plugin(monkeypatch)
    assert not hasattr(module, "pytest_cmdline_main")
    assert not hasattr(module, "pytest_xdist_setupnodes")


def test_plugin_ships_as_package_data():
    import tomllib

    directory = plugin_directory()
    assert (directory / f"{PLUGIN_MODULE}.py").is_file()
    assert not (directory / "__init__.py").exists()
    package_root = Path(__file__).resolve().parents[1] / "src" / "coding_review_agent_loop"
    assert directory.resolve().parent == package_root.resolve()
    pyproject = tomllib.loads((package_root.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert "src/coding_review_agent_loop" in wheel["packages"]
    assert not wheel.get("exclude")
    assert any("pytest-xdist" in item for item in pyproject["project"]["optional-dependencies"]["dev"])


# ---------------------------------------------------------------------------
# Host memory probe (containment)
# ---------------------------------------------------------------------------


def test_probe_host_memory_none_keeps_host_memory_fallback(monkeypatch):
    real_path = containment.Path

    class Unreadable(type(real_path())):
        def read_text(self, *args, **kwargs):
            raise OSError("denied")

    monkeypatch.setattr(containment, "Path", Unreadable)

    def broken(_name):
        raise OSError("no sysconf")

    monkeypatch.setattr(containment.os, "sysconf", broken)
    assert containment.probe_host_memory_bytes() is None
    assert containment.host_memory_bytes() == GIB
    budget = derive_worker_budget(backend="process-group", cpu_reader=_cpu(4), ancestry=NO_ANCESTRY)
    assert budget.limiting_factor == "memory-unknown"


def test_default_policy_unchanged_for_readable_host():
    total = containment.host_memory_bytes()
    assert total == (containment.probe_host_memory_bytes() or GIB)
    policy = containment.default_policy()
    assert policy.aggregate == containment.default_policy().aggregate


# ---------------------------------------------------------------------------
# Runner integration (post-admission derivation, env export, run_foreground_test)
# ---------------------------------------------------------------------------


class _ManagedHandle:
    managed = True
    backend = MANAGED_BACKEND

    def __init__(self, child, aggregate):
        self.child_limits = child
        self.aggregate_limits = aggregate


def test_runner_derives_from_admitted_handle(monkeypatch):
    from coding_review_agent_loop import test_workers
    from coding_review_agent_loop.runner import Runner

    monkeypatch.setattr(test_workers, "probe_cpu_count", lambda: (8, "affinity"))
    monkeypatch.setattr(containment, "probe_host_memory_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(test_workers, "read_ancestry_limits", lambda **_kw: NO_ANCESTRY)
    runner = Runner(containment_policy=containment.default_policy(mode="off"))
    handle = _ManagedHandle(ResourceLimits(3 * GIB, None, 0, None), ResourceLimits(6 * GIB, None, 0, None))
    env: dict[str, str] = {}
    budget = runner._apply_worker_budget_env("coder", handle, env)
    assert budget.workers == 2
    assert env == {ENV_TEST_WORKERS: "2", ENV_TEST_WORKER_ENFORCEMENT: "clamp"}
    assert runner._apply_worker_budget_env("reviewer", handle, {}) is None
    # Process-group fallback ignores all role limits.
    unmanaged = runner.derive_worker_budget(None)
    assert unmanaged.workers == 8
    runner.test_workers = 5
    assert runner.derive_worker_budget(handle).workers == 5


def test_runner_exports_budget_to_coder_launch(tmp_path):
    from coding_review_agent_loop.runner import Runner

    runner = Runner(containment_policy=containment.default_policy(mode="off", cache_dir=tmp_path / ".rt"))
    runner.test_workers = 3
    runner.test_worker_enforcement = "refuse"
    script = (
        "import os, sys; print(os.environ.get('AGENT_LOOP_TEST_WORKERS'), "
        "os.environ.get('AGENT_LOOP_TEST_WORKER_ENFORCEMENT'))"
    )
    subprocess_env = {k: v for k, v in os.environ.items()}
    result = runner.run_with_log(
        [sys.executable, "-c", script], cwd=tmp_path, log_path=tmp_path / "coder.log",
        label="coder", progress_interval_seconds=1, env=subprocess_env,
    )
    assert result.returncode == 0
    assert "3 refuse" in (tmp_path / "coder.log").read_text()


def test_run_foreground_test_without_budget_takes_no_lock_and_injects_nothing(tmp_path, monkeypatch):
    from coding_review_agent_loop import runner as runner_module

    calls: list[object] = []
    monkeypatch.setattr(
        runner_module.WorkerBudgetLock, "acquire",
        classmethod(lambda cls, **kw: calls.append(kw) or (None, None)),
    )
    env = dict(os.environ)
    env.update({ENV_TEST_WORKERS: "1", ENV_TEST_WORKER_ENFORCEMENT: "refuse", "AGENT_LOOP_INVOCATION_ID": "outer"})
    result = runner_module.run_foreground_test(
        [sys.executable, "-c", "import os; print(os.environ.get('PYTEST_PLUGINS'))"],
        cwd=tmp_path, timeout_seconds=30, env=env, environment_is_complete=True, echo_output=False,
    )
    assert result.returncode == 0
    assert calls == []
    assert "None" in result.output_tail
    assert result.workers_cohort is None


def _enforced(workers=2, mode="clamp"):
    return WorkerBudget(workers, "inherited", mode, "cpu", {}, False)


def test_run_foreground_test_busy_when_invocation_lock_held(tmp_path):
    from coding_review_agent_loop.runner import run_foreground_test

    root = tmp_path / "locks"
    held, _ = WorkerBudgetLock.acquire(invocation_id="inv-1", cwd=tmp_path, root=root)
    env = {**os.environ, "AGENT_LOOP_INVOCATION_ID": "inv-1"}
    marker = tmp_path / "spawned"
    try:
        result = run_foreground_test(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x')"],
            cwd=tmp_path, timeout_seconds=30, env=env, environment_is_complete=True,
            echo_output=False, worker_budget=_enforced(), worker_lock_root=root,
        )
        assert result.outcome == "worker-budget-busy"
        assert result.returncode == 125
        assert not marker.exists()
        # Off mode takes no worker-budget lock and runs.
        off = run_foreground_test(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x')"],
            cwd=tmp_path, timeout_seconds=30, env=env, environment_is_complete=True,
            echo_output=False, worker_budget=_enforced(mode="off"), worker_lock_root=root,
        )
        assert off.returncode == 0 and marker.exists()
        # Another invocation is independent.
        other = run_foreground_test(
            [sys.executable, "-c", "pass"], cwd=tmp_path, timeout_seconds=30,
            env={**os.environ, "AGENT_LOOP_INVOCATION_ID": "inv-2"}, environment_is_complete=True,
            echo_output=False, worker_budget=_enforced(), worker_lock_root=root,
        )
        assert other.returncode == 0
    finally:
        held.close()
    after = run_foreground_test(
        [sys.executable, "-c", "pass"], cwd=tmp_path, timeout_seconds=30, env=env,
        environment_is_complete=True, echo_output=False, worker_budget=_enforced(), worker_lock_root=root,
    )
    assert after.returncode == 0
    # The lock is released after a failure and a timeout too.
    failed = run_foreground_test(
        [sys.executable, "-c", "raise SystemExit(3)"], cwd=tmp_path, timeout_seconds=30, env=env,
        environment_is_complete=True, echo_output=False, worker_budget=_enforced(), worker_lock_root=root,
    )
    assert failed.returncode == 3
    timed = run_foreground_test(
        [sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path, timeout_seconds=0.5, env=env,
        environment_is_complete=True, echo_output=False, worker_budget=_enforced(), worker_lock_root=root,
    )
    assert timed.outcome == "timed_out"
    probe, _ = WorkerBudgetLock.acquire(invocation_id="inv-1", cwd=tmp_path, root=root)
    assert probe is not None
    probe.close()


def test_run_foreground_test_refuses_plugin_disable_before_spawn(tmp_path):
    from coding_review_agent_loop.runner import run_foreground_test

    result = run_foreground_test(
        [sys.executable, "-m", "pytest", "-p", f"no:{PLUGIN_MODULE}"], cwd=tmp_path, timeout_seconds=30,
        env=dict(os.environ), environment_is_complete=True, echo_output=False,
        worker_budget=_enforced(mode="refuse"), worker_lock_root=tmp_path / "locks",
    )
    assert result.outcome == "worker-budget-refused"
    assert result.returncode == 2
    assert result.suite_start == "not-started"


def test_lingering_descendants_terminated_before_release(tmp_path):
    from coding_review_agent_loop.runner import run_foreground_test

    pid_file = tmp_path / "child.pid"
    script = (
        "import subprocess, sys; "
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))"
    )
    result = run_foreground_test(
        [sys.executable, "-c", script], cwd=tmp_path, timeout_seconds=30, env=dict(os.environ),
        environment_is_complete=True, echo_output=False, worker_budget=_enforced(),
        worker_lock_root=tmp_path / "locks",
    )
    assert result.returncode == 0
    assert "worker-budget-descendants-terminated" in result.worker_caveats
    child = int(pid_file.read_text())
    import time

    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        with open(f"/proc/{child}/stat") as stream:
            if stream.read().split(")")[-1].split()[0] == "Z":
                break
        time.sleep(0.05)
    else:  # pragma: no cover - failure path
        pytest.fail("background descendant survived the worker-budget release")


def test_setsid_descendant_is_documented_exclusion_and_off_mode_terminates_nothing(tmp_path):
    from coding_review_agent_loop.runner import run_foreground_test

    pid_file = tmp_path / "child.pid"
    script = (
        "import subprocess, sys; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'], start_new_session=True); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))"
    )
    for mode in ("clamp", "off"):
        result = run_foreground_test(
            [sys.executable, "-c", script], cwd=tmp_path, timeout_seconds=30, env=dict(os.environ),
            environment_is_complete=True, echo_output=False, worker_budget=_enforced(mode=mode),
            worker_lock_root=tmp_path / "locks",
        )
        assert result.descendants_terminated == 0
        child = int(pid_file.read_text())
        os.kill(child, 0)  # still alive: outside the process group
        os.kill(child, 9)
