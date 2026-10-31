"""Opt-in waiting for a parsed usage-limit reset (#1601)."""
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_loop_helpers import FakeRunner, make_config, patch_attr_in, patch_object_in
from coding_review_agent_loop import (
    cli,
    discuss_loop,
    quota_wait,
    response_validation,
    review_rounds,
    validated_agent,
    workdir_claims,
)
from coding_review_agent_loop.agents.base import AgentResult
from coding_review_agent_loop.errors import QuotaResetExceededError
from test_semantic_evidence_repair_routing import (
    _forbid_repair,
    _implementation_with_claim,
    _observation,
    _sequenced_validator,
)

import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop

SIX_HOURS = 6 * 3600


def _claude_limit(reset="resets in 2h"):
    return AgentResult(
        text=(
            '{"type":"result","is_error":true,"api_error_status":429,'
            f'"result":"Claude AI usage limit reached (usage_limit_reached) · {reset}"}}'
        ),
        returncode=1,
    )


OK = AgentResult(text="fine", returncode=0, session_id="s-ok")


@pytest.fixture
def slept(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(quota_wait, "_sleeper", calls.append)
    return calls


@pytest.fixture
def logs(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(validated_agent, "log", lambda _c, m: lines.append(m))
    monkeypatch.setattr(quota_wait, "log", lambda _c, m: lines.append(m))
    return lines


def _turn(tmp_path, results, *, wait=SIX_HOURS, retries=0, runner=None, session_id="s0", **kwargs):
    runner = runner or FakeRunner()
    config = make_config(tmp_path, agent_max_retries=retries, wait_for_quota_reset_seconds=wait)
    with patch.object(validated_agent, "run_agent_result", side_effect=list(results)) as invoke:
        try:
            response = validated_agent._run_validated_agent(
                runner, agent="claude", config=config, prompt="PROMPT",
                marker_description="none", validate=lambda text: text,
                session_id=session_id, **kwargs,
            )
            error = None
        except QuotaResetExceededError as exc:
            response, error = None, exc
    return response, error, invoke, runner


def _runner_sleeps(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]


# --- off-default / wait-within / wait-beyond / no-parsed-reset ----------------

def test_off_default_raises_immediately_without_budget(tmp_path, slept, monkeypatch):
    monkeypatch.setattr(
        quota_wait, "quota_wait_budget_for_current_run",
        lambda _c: pytest.fail("budget consulted with the option off"),
    )
    _r, error, invoke, runner = _turn(tmp_path, [_claude_limit("resets in 5h")], wait=None)
    assert isinstance(error, QuotaResetExceededError)
    assert "quota exhausted. Reset in 5h" in str(error)
    assert "Rerun when quota resets" in str(error)
    assert invoke.call_count == 1
    assert slept == [] and _runner_sleeps(runner) == []
    assert QuotaResetExceededError.EXIT_CODE == 3


def test_wait_within_duration_replays_same_invocation(tmp_path, slept, logs):
    response, error, invoke, runner = _turn(tmp_path, [_claude_limit(), OK])
    assert error is None and response.text == "fine"
    assert invoke.call_count == 2
    first, second = (call.kwargs for call in invoke.call_args_list)
    assert first["prompt"] == second["prompt"] == "PROMPT"
    assert first["session_id"] == second["session_id"] == "s0"
    assert first["timeout_seconds"] == second["timeout_seconds"]
    assert first["attempt_suffix"] == "attempt1"  # no ordinary retry consumed
    assert second["attempt_suffix"] == "quota-wait-1"
    # One wait of reset + margin, delivered in bounded slices.
    assert sum(slept) == 7200 + quota_wait.QUOTA_WAIT_MARGIN_SECONDS
    assert max(slept) <= quota_wait.QUOTA_WAIT_SLICE_SECONDS
    assert _runner_sleeps(runner) == []  # no ordinary retry backoff either
    waiting = [line for line in logs if "quota exhausted; waiting until" in line]
    assert len(waiting) == 1
    assert "(2h 1m) for reset (wait 1 of 2)" in waiting[0]
    assert any("quota reset wait finished" in line for line in logs)


def test_wait_beyond_duration_raises_without_sleeping(tmp_path, slept):
    _r, error, invoke, runner = _turn(tmp_path, [_claude_limit("resets in 5h")], wait=3600)
    assert isinstance(error, QuotaResetExceededError)
    assert "Reset in 5h" in str(error)
    assert invoke.call_count == 1
    assert slept == [] and _runner_sleeps(runner) == []


def test_unparsed_reset_keeps_ordinary_retry_and_never_waits(tmp_path, slept):
    unparsed = AgentResult(text="HTTP 429: rate limit exceeded.", returncode=1)
    response, error, invoke, runner = _turn(tmp_path, [unparsed, OK], retries=1)
    assert error is None and response.text == "fine"
    assert slept == []
    assert len(_runner_sleeps(runner)) == 1
    assert invoke.call_args_list[1].kwargs["attempt_suffix"] == "attempt2"


# --- quota-replay-frozen -------------------------------------------------------

def test_quota_on_evidence_reask_replays_the_reask_prompt_and_session(tmp_path, slept, monkeypatch):
    patch_attr_in(
        monkeypatch, (validated_agent, pr_loop, plan_first_loop), "_run_structured_repair", _forbid_repair
    )
    results = [
        AgentResult(text=_implementation_with_claim(), returncode=0, session_id="s-first"),
        _claude_limit(),
        AgentResult(text=_implementation_with_claim(), returncode=0),
    ]
    catalogs = [[_observation(suite_start="not-started")], [_observation()], [_observation()]]
    config = make_config(tmp_path, agent_max_retries=0, wait_for_quota_reset_seconds=SIX_HOURS)
    with patch_object_in(
        (validated_agent, response_validation), "run_agent_result", side_effect=results
    ) as invoke:
        response = validated_agent._run_validated_agent(
            FakeRunner(), agent="claude", config=config, prompt="Implement the issue.",
            marker_description="structured issue_implementation result",
            validate=_sequenced_validator(catalogs, invoke), role="coder", use_repair=True,
            repair_expected_kind="issue_implementation", session_id="caller",
            reask_on_evidence_rejection=True,
        )
    assert response.marker_value is not None
    assert invoke.call_count == 3
    _original, reask, replay = (call.kwargs for call in invoke.call_args_list)
    assert "Previous response not accepted: test evidence" in reask["prompt"]
    assert reask["session_id"] == "s-first"
    assert replay["prompt"] == reask["prompt"]
    assert replay["session_id"] == reask["session_id"]
    assert reask["attempt_suffix"] == "evidence-reask"
    assert replay["attempt_suffix"] == "quota-wait-1"
    assert sum(slept) == 7260


# --- count-bound / duration-bound / shared-budget ------------------------------

def test_third_limit_after_two_waits_raises_without_sleeping(tmp_path, slept):
    _r, error, invoke, _runner = _turn(
        tmp_path, [_claude_limit("resets in 1h")] * 3 + [OK], wait=SIX_HOURS
    )
    assert isinstance(error, QuotaResetExceededError)
    assert invoke.call_count == 3
    assert sum(slept) == 2 * (3600 + quota_wait.QUOTA_WAIT_MARGIN_SECONDS)


def test_run_budget_with_two_waits_used_refuses_another(tmp_path, slept):
    config = make_config(tmp_path, wait_for_quota_reset_seconds=SIX_HOURS)
    with workdir_claims.workdir_claim_scope():
        budget = quota_wait.quota_wait_budget_for_current_run(config)
        assert budget.try_reserve(60) and budget.try_reserve(60)
        slept.clear()
        _r, error, invoke, _runner = _turn(tmp_path, [_claude_limit("resets in 10m"), OK])
    assert isinstance(error, QuotaResetExceededError)
    assert invoke.call_count == 1 and slept == []


def test_cumulative_duration_bound():
    budget = quota_wait.QuotaWaitBudget(3 * 3600)
    budget.waits_used, budget.seconds_waited = 1, 9000
    assert budget.try_reserve(3600) is None
    assert (budget.waits_used, budget.seconds_waited) == (1, 9000)
    reservation = budget.try_reserve(1800 - quota_wait.QUOTA_WAIT_MARGIN_SECONDS)
    assert reservation.sleep_seconds == 1800 and reservation.wait_number == 2


def test_cumulative_duration_bound_through_a_turn(tmp_path, slept):
    config = make_config(tmp_path, wait_for_quota_reset_seconds=3 * 3600)
    with workdir_claims.workdir_claim_scope():
        budget = quota_wait.quota_wait_budget_for_current_run(config)
        budget.waits_used, budget.seconds_waited = 1, 9000
        _r, error, invoke, _runner = _turn(
            tmp_path, [_claude_limit("resets in 1h"), OK], wait=3 * 3600
        )
    assert isinstance(error, QuotaResetExceededError)
    assert invoke.call_count == 1 and slept == []


def test_reviewer_threads_share_one_serialized_run_budget(tmp_path):
    config = make_config(tmp_path, wait_for_quota_reset_seconds=3 * 3600)
    barrier = threading.Barrier(2)
    seen = []

    def run_turn(reviewer):
        budget = quota_wait.quota_wait_budget_for_current_run(config)
        barrier.wait(timeout=60)
        seen.append((budget, budget.try_reserve(2 * 3600)))
        return review_rounds._ReviewerTurnResult(reviewer_name=str(reviewer))

    with workdir_claims.workdir_claim_scope():
        outer = quota_wait.quota_wait_budget_for_current_run(config)
        review_rounds._launch_reviewer_turns(
            FakeRunner(), ["codex", "gemini"], thread_name_prefix="t", run_turn=run_turn
        )
        assert seen[0][0] is seen[1][0] is outer
        assert sum(reservation is not None for _b, reservation in seen) == 1
        assert outer.waits_used == 1 and outer.seconds_waited <= outer.limit_seconds
    assert quota_wait._run_budgets == {}
    with workdir_claims.workdir_claim_scope():
        assert quota_wait.quota_wait_budget_for_current_run(config).waits_used == 0


def test_no_owner_budget_is_local():
    config = type("C", (), {"wait_for_quota_reset_seconds": 60})()
    assert quota_wait.quota_wait_budget_for_current_run(config) is not (
        quota_wait.quota_wait_budget_for_current_run(config)
    )


def test_slice_timeout_observes_signal_cancellation(tmp_path, monkeypatch, logs):
    slices = []

    def timed_slice(seconds):
        assert quota_wait.active_wait_count() == 1
        assert not quota_wait._cancelled()
        slices.append(seconds)
        # Model a signal becoming pending after the wait's interrupt window:
        # the timed wait returns normally, without raising KeyboardInterrupt.
        monkeypatch.setattr(quota_wait, "_signal_cancel_requested", True)

    monkeypatch.setattr(quota_wait, "_sleeper", timed_slice)
    response, error, invoke, runner = _turn(tmp_path, [_claude_limit(), OK])

    assert response is None
    assert isinstance(error, QuotaResetExceededError)
    assert "quota reset wait interrupted" in str(error)
    assert len(slices) == 1
    assert 0 < slices[0] <= 1
    assert invoke.call_count == 1  # cancellation prevents replay
    assert quota_wait.active_wait_count() == 0
    assert not any("quota reset wait finished" in line for line in logs)
    assert runner.comments == []


# --- signals -------------------------------------------------------------------

def _signal_when_waiting(signum, *, count=1, executor_ready=None):
    def fire():
        if executor_ready is not None:
            assert executor_ready.wait(20)
        deadline = time.monotonic() + 20
        while quota_wait.active_wait_count() < count and time.monotonic() < deadline:
            time.sleep(0.01)
        # Direct the signal at the main thread, as a terminal or supervisor
        # signal effectively is.  A process-directed os.kill from this thread
        # may be delivered to any other live thread in the test process, and
        # then the main thread can stay blocked until its timed slice ends.
        signal.pthread_kill(threading.main_thread().ident, signum)

    thread = threading.Thread(target=fire, daemon=True)
    thread.start()
    return thread


PR_ARGS = ["pr", "77", "--repo", "OWNER/REPO", "--wait-for-quota-reset", "6h"]


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_signal_during_main_thread_wait_is_a_clean_quota_stop(tmp_path, monkeypatch, capsys, signum):
    previous = signal.getsignal(signal.SIGTERM)
    runner = FakeRunner()
    config = make_config(tmp_path, agent_max_retries=0, wait_for_quota_reset_seconds=SIX_HOURS)

    def fake_pr_loop(*_a, **_k):
        with patch.object(validated_agent, "run_agent_result", side_effect=[_claude_limit(), OK]):
            validated_agent._run_validated_agent(
                runner, agent="claude", config=config, prompt="P",
                marker_description="none", validate=lambda t: t,
            )
        pytest.fail("the wait was not interrupted")

    monkeypatch.setattr(cli, "run_pr_loop", fake_pr_loop)
    started = time.monotonic()
    firing = _signal_when_waiting(signum)
    assert cli.main(PR_ARGS) == QuotaResetExceededError.EXIT_CODE
    firing.join(5)
    assert time.monotonic() - started < 20
    assert "Claude quota reset wait interrupted; rerun to resume" in capsys.readouterr().err
    assert signal.getsignal(signal.SIGTERM) == previous
    assert runner.comments == []  # the wait writes nothing to GitHub
    assert not quota_wait.wait_was_interrupted()  # cleared for the next command
    assert quota_wait.active_wait_count() == 0


def _record_terminations(runner):
    calls = []
    original = runner.terminate_active_processes

    def spy():
        calls.append(True)
        return original()

    runner.terminate_active_processes = spy
    return calls


def _waiting_worker(config, name):
    reservation = quota_wait.QuotaWaitReservation(7200, 1, 2)
    quota_wait.wait_for_reset(config, name, reservation)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_signal_cancels_a_waiting_reviewer_worker(tmp_path, monkeypatch, capsys, signum):
    # Interrupt the waiting executor, after submit has registered its threads.
    # ThreadPoolExecutor.start itself has an interruptible startup window;
    # signalling there tests startup interruption rather than waiter cleanup.
    executor_ready = threading.Event()
    real_as_completed = review_rounds.as_completed

    def waiting_as_completed(futures):
        executor_ready.set()
        return real_as_completed(futures)

    monkeypatch.setattr(review_rounds, "as_completed", waiting_as_completed)
    previous = signal.getsignal(signal.SIGTERM)
    runner = FakeRunner()
    terminations = _record_terminations(runner)
    config = make_config(tmp_path, wait_for_quota_reset_seconds=SIX_HOURS)
    outcomes = []

    def run_turn(reviewer):
        try:
            _waiting_worker(config, "Codex")
        except QuotaResetExceededError as exc:
            outcomes.append(exc)
            return review_rounds._ReviewerTurnResult(reviewer_name="Codex", error=exc)
        return review_rounds._ReviewerTurnResult(reviewer_name="Codex")

    def fake_pr_loop(*_a, **_k):
        review_rounds._launch_reviewer_turns(runner, ["codex"], thread_name_prefix="t", run_turn=run_turn)
        pytest.fail("the interrupt did not reach the main thread")

    monkeypatch.setattr(cli, "run_pr_loop", fake_pr_loop)
    started = time.monotonic()
    firing = _signal_when_waiting(signum, executor_ready=executor_ready)
    assert cli.main(PR_ARGS) == QuotaResetExceededError.EXIT_CODE
    firing.join(5)
    assert time.monotonic() - started < 20
    assert terminations == [True]
    assert len(outcomes) == 1 and "wait interrupted" in str(outcomes[0])
    assert "quota reset wait interrupted" in capsys.readouterr().err
    assert signal.getsignal(signal.SIGTERM) == previous
    assert quota_wait.active_wait_count() == 0


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_signal_cancels_waiting_discuss_debaters(tmp_path, monkeypatch, capsys, signum):
    runner = FakeRunner()
    terminations = _record_terminations(runner)
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), discuss_parallel=True,
        wait_for_quota_reset_seconds=SIX_HOURS,
    )
    outcomes = []

    def debater_turn(_runner, *, reviewer_name, **_kwargs):
        try:
            _waiting_worker(config, reviewer_name)
        except QuotaResetExceededError as exc:
            outcomes.append(exc)
            raise
        pytest.fail("the debater wait was not cancelled")

    monkeypatch.setattr(discuss_loop, "_run_discuss_debater_turn", debater_turn)

    def fake_discuss(*_a, **_k):
        discuss_loop.run_discuss_loop(runner, issue_number=56, config=config)
        pytest.fail("the interrupt did not reach the main thread")

    monkeypatch.setattr(cli, "run_discuss_loop", fake_discuss)
    started = time.monotonic()
    firing = _signal_when_waiting(signum, count=2)
    assert cli.main(
        ["discuss", "56", "--repo", "OWNER/REPO", "--wait-for-quota-reset", "6h"]
    ) == QuotaResetExceededError.EXIT_CODE
    firing.join(5)
    assert time.monotonic() - started < 20
    assert terminations == [True]
    # The second debater records its cancellation only once its thread has
    # unwound, which can land just after main() returns; wait briefly for it.
    deadline = time.monotonic() + 5
    while len(outcomes) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(outcomes) == 2
    assert quota_wait.active_wait_count() == 0
    assert "quota reset wait interrupted" in capsys.readouterr().err


def test_sigterm_without_active_wait_keeps_default_termination(tmp_path):
    script = (
        "import os, signal, time\n"
        "from coding_review_agent_loop import quota_wait\n"
        "assert quota_wait.install_sigterm_handler() == signal.SIG_DFL\n"
        "os.kill(os.getpid(), signal.SIGTERM)\n"
        "time.sleep(10)\n"
        "raise SystemExit(0)\n"
    )
    src = Path(quota_wait.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(src)}
    completed = subprocess.run(
        [sys.executable, "-c", script], env=env, timeout=60, capture_output=True, text=True,
    )
    assert completed.returncode == -signal.SIGTERM, completed.stderr


def test_sigterm_without_active_wait_chains_to_a_python_handler():
    received = []
    original = signal.signal(signal.SIGTERM, lambda signum, _frame: received.append(signum))
    try:
        previous = quota_wait.install_sigterm_handler()
        os.kill(os.getpid(), signal.SIGTERM)
        deadline = time.monotonic() + 5
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received == [signal.SIGTERM]
        assert not quota_wait.wait_was_interrupted()
        quota_wait.restore_sigterm_handler(previous)
        assert signal.getsignal(signal.SIGTERM) is previous
    finally:
        signal.signal(signal.SIGTERM, original)


def test_ordinary_keyboard_interrupt_still_propagates(monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(cli, "run_pr_loop", lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        cli.main(PR_ARGS)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_option_off_installs_no_sigterm_handler(monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    seen = []
    monkeypatch.setattr(cli, "run_pr_loop", lambda *_a, **_k: seen.append(signal.getsignal(signal.SIGTERM)) or 0)
    assert cli.main(["pr", "77", "--repo", "OWNER/REPO"]) == 0
    assert seen == [previous]
    monkeypatch.setattr(cli, "run_pr_loop", lambda *_a, **_k: seen.append(signal.getsignal(signal.SIGTERM)) or 0)
    assert cli.main(PR_ARGS) == 0
    assert seen[-1] != previous
    assert signal.getsignal(signal.SIGTERM) == previous


# --- repeated in-process main() calls (#1607) -----------------------------------

def test_repeated_main_after_cancelled_wait_waits_normally(tmp_path, monkeypatch, capsys):
    runner = FakeRunner()
    config = make_config(tmp_path, agent_max_retries=0, wait_for_quota_reset_seconds=SIX_HOURS)

    def interrupted_pr_loop(*_a, **_k):
        with patch.object(validated_agent, "run_agent_result", side_effect=[_claude_limit(), OK]):
            validated_agent._run_validated_agent(
                runner, agent="claude", config=config, prompt="P",
                marker_description="none", validate=lambda t: t,
            )
        pytest.fail("the wait was not interrupted")

    monkeypatch.setattr(cli, "run_pr_loop", interrupted_pr_loop)
    firing = _signal_when_waiting(signal.SIGINT)
    assert cli.main(PR_ARGS) == QuotaResetExceededError.EXIT_CODE
    firing.join(5)
    assert "quota reset wait interrupted" in capsys.readouterr().err

    slept: list[float] = []
    monkeypatch.setattr(quota_wait, "_sleeper", slept.append)
    responses = []

    def waiting_pr_loop(*_a, **_k):
        with patch.object(validated_agent, "run_agent_result", side_effect=[_claude_limit(), OK]) as invoke:
            responses.append(validated_agent._run_validated_agent(
                runner, agent="claude", config=config, prompt="P",
                marker_description="none", validate=lambda t: t,
            ))
        assert invoke.call_count == 2
        return 0

    monkeypatch.setattr(cli, "run_pr_loop", waiting_pr_loop)
    assert cli.main(PR_ARGS) == 0
    assert [r.text for r in responses] == ["fine"]
    assert sum(slept) == 7200 + quota_wait.QUOTA_WAIT_MARGIN_SECONDS
    assert "quota reset wait interrupted" not in capsys.readouterr().err


def test_option_off_keyboard_interrupt_clears_cancel_latch(tmp_path, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    seen = []

    def interrupted_pr_loop(*_a, **_k):
        seen.append(signal.getsignal(signal.SIGTERM))
        quota_wait.cancel_active_waits()  # what an executor does on Ctrl-C
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_pr_loop", interrupted_pr_loop)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["pr", "77", "--repo", "OWNER/REPO"])
    assert seen == [previous]
    assert signal.getsignal(signal.SIGTERM) == previous
    # Checked in the test body, before the autouse fixture resets state.
    assert not quota_wait._cancel_event.is_set()
    assert not quota_wait.wait_was_interrupted()

    slept: list[float] = []
    monkeypatch.setattr(quota_wait, "_sleeper", slept.append)
    reservation = quota_wait.QuotaWaitReservation(7200, 1, 2)
    quota_wait.wait_for_reset(make_config(tmp_path), "Claude", reservation)
    assert sum(slept) == reservation.sleep_seconds


def test_main_chains_sigterm_to_host_python_handler(monkeypatch):
    received = []

    def host_handler(signum, _frame):
        received.append(signum)

    original = signal.signal(signal.SIGTERM, host_handler)
    try:
        installed = []

        def pr_loop(*_a, **_k):
            installed.append(signal.getsignal(signal.SIGTERM))
            os.kill(os.getpid(), signal.SIGTERM)
            deadline = time.monotonic() + 5
            while not received and time.monotonic() < deadline:
                time.sleep(0.01)
            return 0

        monkeypatch.setattr(cli, "run_pr_loop", pr_loop)
        assert cli.main(PR_ARGS) == 0
        assert installed and installed[0] is not host_handler  # main's handler chained
        assert received == [signal.SIGTERM]
        assert not quota_wait.wait_was_interrupted()
        assert signal.getsignal(signal.SIGTERM) is host_handler
    finally:
        signal.signal(signal.SIGTERM, original)


# --- signals at wait-lifecycle boundaries (subprocess: a deadlock times out) ----

_BOUNDARY_SCRIPT = r"""
import os, signal, sys, time
from coding_review_agent_loop import quota_wait
from coding_review_agent_loop.errors import QuotaResetExceededError

boundary, signum = sys.argv[1], int(sys.argv[2])
signal.signal(signal.SIGINT, signal.default_int_handler)
quota_wait.install_sigterm_handler()
fired = []

def fire():
    if not fired:
        fired.append(True)
        os.kill(os.getpid(), signum)
        time.sleep(0)  # the handler runs at the next bytecode boundary

class HookSet(set):
    def add(self, item):
        super().add(item)
        if boundary == "registered":
            fire()

    def discard(self, item):
        if boundary == "unregister" and item in self:
            fire()  # still registered when the signal lands
        super().discard(item)

quota_wait._active_waits = HookSet()
logs = []

def log(_config, message):
    logs.append(message)
    if (boundary, len(logs)) in {("wait-log", 1), ("resume-log", 2)}:
        fire()

quota_wait.log = log
quota_wait._sleeper = lambda seconds: None
if boundary == "lock-held":
    # The handler must not need the registry/latch lock held by the thread it interrupts.
    quota_wait._active_waits.add(0)
    try:
        with quota_wait._waiters_lock:
            fire()
            time.sleep(20)
    except KeyboardInterrupt:
        sys.exit(130 if quota_wait.wait_was_interrupted() else 1)
    sys.exit(2)
try:
    quota_wait.wait_for_reset(None, "Claude", quota_wait.QuotaWaitReservation(60, 1, 2))
except QuotaResetExceededError as exc:
    clean = "Claude quota reset wait interrupted" in str(exc) and not quota_wait._active_waits
    sys.exit(3 if clean and quota_wait.wait_was_interrupted() else 4)
sys.exit(0)
"""


def _run_boundary(boundary, signum):
    src = Path(quota_wait.__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-c", _BOUNDARY_SCRIPT, boundary, str(int(signum))],
        env={**os.environ, "PYTHONPATH": str(src)}, timeout=60, capture_output=True, text=True,
    )


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
@pytest.mark.parametrize("boundary", ["registered", "wait-log", "resume-log", "unregister"])
def test_signal_anywhere_in_the_registered_wait_is_a_clean_quota_stop(boundary, signum):
    completed = _run_boundary(boundary, signum)
    assert completed.returncode == 3, completed.stderr


def test_sigterm_handler_never_blocks_on_a_lock_the_main_thread_holds():
    completed = _run_boundary("lock-held", signal.SIGTERM)
    assert completed.returncode == 130, completed.stderr


# --- attempt ceiling renewed by each wait ----------------------------------------

@pytest.mark.parametrize("backend", ["gemini", "antigravity"])
def test_waits_renew_attempts_with_replacement_replay_and_renewed_fallbacks(tmp_path, monkeypatch, backend):
    from coding_review_agent_loop.agents import antigravity as agy
    from test_validated_agent_recovery import _observed_replacement_result

    now = [1000.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(agy, "_now", lambda: now[0])
    monkeypatch.setattr(quota_wait, "_sleeper", sleep)
    quota = "Error: quota exceeded, try again in 10m"

    def limit(model):
        return AgentResult(text=quota, raw_output=quota, returncode=1, model_used=model)

    replacement = _observed_replacement_result(
        tmp_path, command="agy" if backend == "antigravity" else "gemini",
        reason="executable changed during invocation", model_used="ModelA",
    )
    if backend == "antigravity":
        # Replacement replay; A, B quota -> wait 1; A, B quota -> wait 2;
        # A quota renews the fallback to B, which succeeds (7 dispatches).
        results = [replacement, *(limit(m) for m in "ABABA"), AgentResult(text="done", returncode=0, model_used="B")]
        extra = {"antigravity_models": ("ModelA", "ModelB")}
    else:
        # Replacement replay; quota -> wait 1; quota -> wait 2; success.
        results = [replacement, limit("g"), limit("g"), AgentResult(text="done", returncode=0)]
        extra = {}
    runner = FakeRunner()
    runner.wait_for_executable_stability = lambda command, *, deadline=None: True
    config = make_config(tmp_path, agent_max_retries=0, wait_for_quota_reset_seconds=SIX_HOURS, **extra)
    with workdir_claims.workdir_claim_scope(), patch_object_in(
        (validated_agent, response_validation), "run_agent_result", side_effect=results
    ) as invoke:
        response = validated_agent._run_validated_agent(
            runner, agent=backend, config=config, prompt="P", marker_description="none",
            validate=lambda text: text,
        )
    assert response.text == "done"
    assert invoke.call_count == len(results)
    assert sum(slept) == 2 * (600 + quota_wait.QUOTA_WAIT_MARGIN_SECONDS)
    assert _runner_sleeps(runner) == []  # no ordinary retry was spent


# --- cancel latch lifetime (#1609) --------------------------------------------

RESERVATION = quota_wait.QuotaWaitReservation(90, 1, 2)


def _wait(tmp_path):
    quota_wait.wait_for_reset(make_config(tmp_path), "Claude", RESERVATION)


def test_stale_latch_is_cleared_for_a_new_programmatic_run(tmp_path, slept, logs):
    quota_wait.cancel_active_waits()  # e.g. an ordinary Ctrl-C with no wait active
    assert quota_wait._cancel_event.is_set()
    assert workdir_claims.live_run_count() == 0
    with workdir_claims.workdir_claim_scope():
        _wait(tmp_path)
    assert sum(slept) == RESERVATION.sleep_seconds
    assert not quota_wait.wait_was_interrupted()


def test_about_to_start_wait_in_an_interrupted_run_is_still_cancelled(tmp_path, slept, logs):
    with workdir_claims.workdir_claim_scope():
        quota_wait.cancel_active_waits()
        with pytest.raises(QuotaResetExceededError, match="interrupted"):
            _wait(tmp_path)
    assert slept == []


def test_nested_scope_joins_without_clearing_the_latch(tmp_path, slept, logs):
    with workdir_claims.workdir_claim_scope():
        quota_wait.cancel_active_waits()
        with workdir_claims.workdir_claim_scope():
            assert workdir_claims.live_run_count() == 1
            with pytest.raises(QuotaResetExceededError):
                _wait(tmp_path)
    assert slept == []


def test_new_run_does_not_clear_while_a_wait_is_active():
    quota_wait._active_waits.add(-1)
    try:
        quota_wait.cancel_active_waits()
        with workdir_claims.workdir_claim_scope():
            assert quota_wait._cancel_event.is_set()
            assert quota_wait.wait_was_interrupted()
    finally:
        quota_wait._active_waits.discard(-1)


def _in_thread(target):
    errors: list[BaseException] = []

    def run():
        try:
            target()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the test
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, errors


def test_independent_run_cannot_clear_a_live_interrupted_runs_latch(tmp_path, slept, logs):
    cancelled = threading.Event()
    proceed = threading.Event()
    outcome: list[str] = []

    def run_a():
        with workdir_claims.workdir_claim_scope("issue", 1):
            quota_wait.cancel_active_waits()  # executor cancel before the worker registers
            cancelled.set()
            assert proceed.wait(10)
            try:
                _wait(tmp_path)
            except QuotaResetExceededError:
                outcome.append("stopped")

    def run_b():
        with workdir_claims.workdir_claim_scope("pr", 2):
            assert workdir_claims.live_run_count() == 2

    thread_a, errors_a = _in_thread(run_a)
    assert cancelled.wait(10)
    thread_b, errors_b = _in_thread(run_b)
    thread_b.join(10)
    assert not thread_b.is_alive() and errors_b == []
    proceed.set()
    thread_a.join(10)
    assert not thread_a.is_alive() and errors_a == []
    assert outcome == ["stopped"]
    assert slept == []
    # After every run ended, the next run starts with a clear latch.
    with workdir_claims.workdir_claim_scope():
        assert not quota_wait._cancel_event.is_set()
        _wait(tmp_path)
    assert sum(slept) == RESERVATION.sleep_seconds


def test_concurrent_run_is_admitted_only_after_the_first_runs_reset(tmp_path, slept, logs):
    in_hook = threading.Event()
    release_hook = threading.Event()
    b_entered = threading.Event()
    outcome: list[str] = []

    def pausing_hook():
        if threading.current_thread().name == "run-a":
            in_hook.set()
            assert release_hook.wait(10)

    # Run the pausing hook before the quota reset, so A's reset is still pending.
    hooks = dict(workdir_claims._first_run_start_hooks)
    workdir_claims._first_run_start_hooks.clear()
    workdir_claims._first_run_start_hooks["test-pause"] = pausing_hook
    workdir_claims._first_run_start_hooks.update(hooks)

    a_done = threading.Event()

    def run_a():
        with workdir_claims.workdir_claim_scope("issue", 1):
            assert a_done.wait(10)

    def run_b():
        with workdir_claims.workdir_claim_scope("pr", 2):
            b_entered.set()
            quota_wait.cancel_active_waits()
            try:
                _wait(tmp_path)
            except QuotaResetExceededError:
                outcome.append("stopped")

    thread_a = threading.Thread(target=run_a, name="run-a", daemon=True)
    thread_a.start()
    assert in_hook.wait(10)
    thread_b, errors_b = _in_thread(run_b)
    assert not b_entered.wait(0.3)  # B blocks at scope entry
    release_hook.set()
    thread_b.join(10)
    assert not thread_b.is_alive() and errors_b == []
    assert outcome == ["stopped"]
    assert slept == []
    a_done.set()
    thread_a.join(10)
    assert not thread_a.is_alive()
