"""Issue-loop tests: run-window telemetry (#1543)."""

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in
from issue_loop_test_helpers import FakeRunner
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_review,
    structured_plan_state,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


def test_run_windows_are_written_once_per_owning_run_and_survive_exceptions(tmp_path, monkeypatch):
    """Reservation telemetry (#1107): run-start/run-end bracket an owning run."""
    from coding_review_agent_loop.worker_telemetry import load_records, telemetry_log_path

    runner = _FakeRunner()
    seen = {}

    def boom(*_args, **_kwargs):
        seen["attribution"] = dict(runner.telemetry_attribution)
        raise AgentLoopError("stop")

    patch_attr_in(monkeypatch, (discuss_loop, pr_loop, issue_loop), "resolve_base_branch", boom)
    with pytest.raises(AgentLoopError, match="stop"):
        run_issue_loop(runner, issue_number=56, config=make_config(tmp_path))
    records = list(load_records(telemetry_log_path()))
    assert [r["record"] for r in records] == ["run-start", "run-end"]
    assert {r["run_id"] for r in records} == {seen["attribution"]["run_id"]}
    assert records[0]["issue_number"] == 56
    assert runner.telemetry_attribution is None


def test_issue_to_pr_workflow_carries_outer_run_and_pr_number_through_a_real_attempt(tmp_path, monkeypatch):
    """Reservation telemetry (#1107): issue -> nested PR phase, one run window."""
    import sys

    from coding_review_agent_loop.runner import run_foreground_test
    from coding_review_agent_loop.worker_telemetry import (
        ReservationTelemetry, load_records, run_duty_cycles, telemetry_log_path,
    )

    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Implement it."),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[structured_plan_review(state="approved", summary="Plan approved.")],
        pr_payload={"body": "Fixes #56", "headRefName": "agent-loop/x", "headRefOid": "abc123"},
    )
    config = make_config(tmp_path)
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: None)
    seen = {}

    def nested_pr_phase(inner_runner, *, pr_number, config, usage_context, **_kwargs):
        # The real run_pr_loop attributes its phase exactly like this and then
        # its coder turn's broker runs the test command with the runner's
        # attribution snapshot.
        token = validated_agent_module._begin_run_telemetry(
            inner_runner, config, usage_context, False, pr_number=pr_number
        )
        try:
            attribution = {**inner_runner.telemetry_attribution, "attribution_source": "runner", "lane": "broker"}
            run_foreground_test(
                [sys.executable, "-c", "pass"], cwd=tmp_path, timeout_seconds=30, echo_output=False,
                reservation_telemetry=ReservationTelemetry(telemetry_log_path(), attribution),
            )
        finally:
            validated_agent_module._end_run_telemetry(inner_runner, token)
        seen["after_nested"] = dict(inner_runner.telemetry_attribution)
        return 0

    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "run_pr_loop", nested_pr_phase)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True) == 0

    records = list(load_records(telemetry_log_path()))
    kinds = [r["record"] for r in records]
    assert kinds == ["run-start", "attempt", "run-end"]
    attempt = records[1]
    assert (attempt["issue_number"], attempt["pr_number"]) == (56, 77)
    assert attempt["run_id"] == records[0]["run_id"] == records[2]["run_id"]
    assert seen["after_nested"]["pr_number"] is None and seen["after_nested"]["issue_number"] == 56
    assert runner.telemetry_attribution is None
    assert run_duty_cycles(records)[attempt["run_id"]]["attempts"] == 1


@pytest.mark.parametrize("loop", ["task", "discuss"])
def test_other_owning_loops_write_run_end_on_exception(tmp_path, monkeypatch, loop):
    from coding_review_agent_loop.worker_telemetry import load_records, telemetry_log_path

    runner = _FakeRunner()
    config = make_config(tmp_path)

    def boom(*_a, **_k):
        raise AgentLoopError("stop")

    patch_attr_in(monkeypatch, (discuss_loop, pr_loop, issue_loop), "resolve_base_branch", boom)
    with pytest.raises(AgentLoopError, match="stop"):
        if loop == "task":
            issue_loop.run_task_loop(runner, task_text="do it", config=config)
        else:
            discuss_loop.run_discuss_loop(runner, issue_number=56, config=config)
    assert [r["record"] for r in load_records(telemetry_log_path())] == ["run-start", "run-end"]
    assert runner.telemetry_attribution is None


def test_real_nested_pr_loop_runs_broker_attempt_with_outer_run_and_both_numbers(tmp_path, monkeypatch):
    """Reservation telemetry (#1107): the real run_pr_loop under a real issue loop."""
    import os
    import sys

    from coding_review_agent_loop.local_test_evidence import TestBrokerClient as BrokerClient
    from coding_review_agent_loop.runner import Runner
    from coding_review_agent_loop.test_workers import WorkerBudget
    from coding_review_agent_loop.worker_telemetry import load_records, telemetry_log_path

    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Implement it."),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[structured_plan_review(state="approved", summary="Plan approved.")],
        pr_payload={"body": "Fixes #56", "headRefName": "agent-loop/x", "headRefOid": "abc123"},
    )
    config = make_config(tmp_path)
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: None)
    seen = {}
    real_pr_loop = pr_loop.run_pr_loop

    real_pr_context = pr_loop_support.get_pr_review_context

    def observed_pr_loop(*args, **kwargs):
        seen["in_pr_loop"] = True
        try:
            return real_pr_loop(*args, **kwargs)
        finally:
            seen["after_nested"] = dict(runner.telemetry_attribution)

    def first_pr_read(*a, **k):
        if not seen.get("in_pr_loop"):
            return real_pr_context(*a, **k)
        # Inside the real run_pr_loop: run a coder-turn broker test request
        # through the real Runner broker setup and attribution snapshot.
        seen["during"] = dict(runner.telemetry_attribution)
        real = Runner()
        real.telemetry_attribution = dict(runner.telemetry_attribution)
        broker, _turn = real._start_test_broker(cwd=tmp_path, role="coder", env=None)
        assert broker is not None
        broker._worker_lock_root = tmp_path / "locks"
        broker.set_execution_context(
            containment_handle=None, process_started=None, process_finished=None,
            worker_budget=WorkerBudget(2, "derived", "clamp", "cpu", {}, False),
        )
        try:
            client = BrokerClient({**os.environ, **broker.environment, "AGENT_LOOP_INVOCATION_ID": broker.turn_id})
            assert client.run([sys.executable, "-c", "pass"], timeout_seconds=10, cwd=tmp_path).outcome == "passed"
        finally:
            broker.stop()
        raise AgentLoopError("stop after broker attempt")

    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "run_pr_loop", observed_pr_loop)
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "get_pr_review_context", first_pr_read)
    with pytest.raises(AgentLoopError, match="stop after broker attempt"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True)

    records = list(load_records(telemetry_log_path()))
    assert [r["record"] for r in records].count("run-start") == 1
    assert [r["record"] for r in records].count("run-end") == 1
    (attempt,) = [r for r in records if r["record"] == "attempt"]
    run_id = next(r for r in records if r["record"] == "run-start")["run_id"]
    assert attempt["run_id"] == run_id and attempt["attribution_source"] == "runner"
    assert (attempt["issue_number"], attempt["pr_number"], attempt["lane"]) == (56, 77, "broker")
    assert seen["during"]["pr_number"] == 77
    assert seen["after_nested"]["pr_number"] is None and seen["after_nested"]["issue_number"] == 56
    assert runner.telemetry_attribution is None


def test_real_issue_to_pr_coder_turn_runs_through_its_own_broker(tmp_path, monkeypatch):
    """#1107: the PR phase's coder followup turn is a real launch with a real broker."""
    import subprocess
    import sys

    from agent_loop_helpers import structured_coder_followup, structured_pr_review
    from coding_review_agent_loop.runner import Runner
    from coding_review_agent_loop.worker_telemetry import load_records, telemetry_log_path

    checkout = tmp_path / "coder-checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.email", "t@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "T"], check=True)
    (checkout / "f.txt").write_text("x\n")
    subprocess.run(["git", "-C", str(checkout), "add", "f.txt"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "init"], check=True)
    src = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src")
    child = (
        "import sys; from coding_review_agent_loop.local_test_evidence import broker_client_from_environment; "
        "print('outcome=' + broker_client_from_environment().run("
        "[sys.executable, '-c', 'pass'], timeout_seconds=30).outcome)"
    )
    real_launches = []

    # Keep worker-budget locks off the host's runtime directories, which
    # differ between developer machines and CI runners.
    from coding_review_agent_loop import test_workers as _test_workers

    # Likewise pin the runtime directory the command lane and broker use.
    runtime_dir = tmp_path / "xdg-runtime"
    runtime_dir.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))

    lock_dir = tmp_path / "worker-locks"
    real_lock_root = _test_workers._ensure_private_directory
    monkeypatch.setattr(
        _test_workers, "worker_budget_lock_root", lambda: real_lock_root(lock_dir)
    )

    class HybridRunner(FakeRunner):
        """Scripted agents, but the PR-phase coder turn is launched by the real
        ``Runner.run_with_log`` on this very object, so its attribution and
        broker come from the state the orchestrator loops set on it."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            from coding_review_agent_loop.containment import default_policy

            # Containment off: the host's systemd/cgroup setup must not decide
            # whether the broker can start the test command.
            Runner.__init__(self, containment_policy=default_policy(mode="off", cache_dir=tmp_path / ".runtime"))

        def run_with_log(self, args, *, cwd, log_path, label, progress_interval_seconds, **kwargs):
            attribution = self.telemetry_attribution
            if list(args[:1]) == ["claude"] and attribution and attribution.get("pr_number"):
                launched = Runner.run_with_log(
                    self, [sys.executable, "-c", child], cwd=checkout,
                    log_path=tmp_path / "real-coder.log", label="coder",
                    progress_interval_seconds=1,
                    env={"PYTHONPATH": src, "AGENT_LOOP_TEST_WORKER_HOST_SHARING": "off"},
                )
                real_launches.append((launched.returncode, (tmp_path / "real-coder.log").read_text()))
            return super().run_with_log(
                args, cwd=cwd, log_path=log_path, label=label,
                progress_interval_seconds=progress_interval_seconds, **kwargs,
            )

    runner = HybridRunner(
        claude_outputs=[
            structured_plan_state(summary="Implement it."),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            structured_coder_followup(addressed_items=["item-1"]),
        ],
        codex_outputs=[
            structured_plan_review(state="approved", summary="Plan approved."),
            structured_pr_review(state="blocking", summary="Fix it.", blocking_items=["Fix the edge case."]),
            structured_pr_review(
                state="approved", summary="Fixed.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        pr_payload={"body": "Fixes #56", "headRefName": "agent-loop/x", "headRefOid": "abc123"},
    )
    config = make_config(tmp_path, max_rounds=3)
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: None)
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *_a, **_k: None)
    seen = {}
    real_pr_loop = pr_loop.run_pr_loop

    def observed(*args, **kwargs):
        try:
            return real_pr_loop(*args, **kwargs)
        finally:
            seen["after_nested"] = dict(runner.telemetry_attribution)

    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "run_pr_loop", observed)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True) == 0

    assert real_launches and real_launches[0][0] == 0 and "outcome=passed" in real_launches[0][1]
    records = list(load_records(telemetry_log_path()))
    kinds = [r["record"] for r in records]
    assert kinds.count("run-start") == 1 and kinds.count("run-end") == 1 and kinds[-1] == "run-end"
    (attempt,) = [r for r in records if r["record"] == "attempt"]
    run_id = next(r for r in records if r["record"] == "run-start")["run_id"]
    assert attempt["run_id"] == run_id
    assert (attempt["issue_number"], attempt["pr_number"], attempt["lane"]) == (56, 77, "broker")
    assert seen["after_nested"]["pr_number"] is None
    assert runner.telemetry_attribution is None
