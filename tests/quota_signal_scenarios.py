"""Explicit-interpreter signal scenarios; never collected by pytest.

The parent owns the process deadline. These functions retain the original
assertions, with fake-runner fixtures recreated explicitly in main().
"""
import contextlib
import faulthandler
import inspect
import io
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop import cli, discuss_loop, quota_wait, review_rounds, validated_agent
from coding_review_agent_loop.errors import QuotaResetExceededError
from test_quota_reset_wait import _claude_limit, OK, SIX_HOURS, PR_ARGS

def _record_event(message):
    # One pipe write keeps concurrent worker records on separate lines.
    os.write(sys.stdout.fileno(), (message + "\n").encode())


@contextlib.contextmanager
def _signal_when_waiting(signum, *, count=1, executor_ready=None, readiness_timeout=10):
    stop = threading.Event()
    errors = []
    def deliver():
        deadline = time.monotonic() + readiness_timeout
        while not stop.is_set():
            ready = executor_ready is None or executor_ready.is_set()
            if ready and quota_wait.active_wait_count() >= count:
                _record_event('signal delivery: registered=' + str(count))
                signal.pthread_kill(threading.main_thread().ident, signum)
                return
            if time.monotonic() >= deadline:
                errors.append(AssertionError('signal readiness was not acknowledged'))
                return  # Never fire after readiness expires.
            stop.wait(0.001)
    def fire():
        try:
            deliver()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=fire, name='quota-signal-sender')
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(2)
        assert not thread.is_alive(), 'sender did not finish'
        if errors:
            raise errors[0]


def _waiting_worker(config, name):
    reservation = quota_wait.QuotaWaitReservation(7200, 1, 2)
    try:
        quota_wait.wait_for_reset(config, name, reservation)
    finally:
        _record_event('worker exit: ' + name)


def _record_terminations(runner):
    calls = []
    original = runner.terminate_active_processes

    def spy():
        _record_event('main cancellation / terminate processes')
        calls.append(True)
        return original()

    runner.terminate_active_processes = spy
    return calls


def scenario_signal_during_main_thread_wait_is_a_clean_quota_stop(tmp_path, monkeypatch, capsys, signum):
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
    with _signal_when_waiting(signum):
        assert cli.main(PR_ARGS) == QuotaResetExceededError.EXIT_CODE
    assert time.monotonic() - started < 20
    assert "Claude quota reset wait interrupted; rerun to resume" in capsys.readouterr().err
    assert signal.getsignal(signal.SIGTERM) == previous
    assert runner.comments == []  # the wait writes nothing to GitHub
    assert not quota_wait.wait_was_interrupted()  # cleared for the next command
    assert quota_wait.active_wait_count() == 0


def scenario_signal_cancels_a_waiting_reviewer_worker(tmp_path, monkeypatch, capsys, signum):
    # Interrupt the waiting executor, after submit has registered its threads.
    # ThreadPoolExecutor.start itself has an interruptible startup window;
    # signalling there tests startup interruption rather than waiter cleanup.
    executor_ready = threading.Event()
    real_wait = review_rounds.wait

    def waiting_wait(*args, **kwargs):
        _record_event("reviewer wait entry")
        executor_ready.set()
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(review_rounds, "wait", waiting_wait)
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
    with _signal_when_waiting(signum, executor_ready=executor_ready):
        assert cli.main(PR_ARGS) == QuotaResetExceededError.EXIT_CODE
    assert time.monotonic() - started < 20
    assert terminations == [True]
    assert len(outcomes) == 1 and "wait interrupted" in str(outcomes[0])
    assert "quota reset wait interrupted" in capsys.readouterr().err
    assert signal.getsignal(signal.SIGTERM) == previous
    assert quota_wait.active_wait_count() == 0


def scenario_signal_cancels_waiting_discuss_debaters(tmp_path, monkeypatch, capsys, signum):
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
    monkeypatch.setattr(discuss_loop, '_run_discuss_analyzer',
                        lambda *a, **k: pytest.fail('analyzer ran after cancellation'))

    def fake_discuss(*_a, **_k):
        discuss_loop.run_discuss_loop(runner, issue_number=56, config=config)
        pytest.fail("the interrupt did not reach the main thread")

    monkeypatch.setattr(cli, "run_discuss_loop", fake_discuss)
    started = time.monotonic()
    executor_ready = threading.Event()
    real_wait = discuss_loop.wait
    def waiting_wait(*args, **kwargs):
        _record_event('discuss wait entry')
        executor_ready.set()
        return real_wait(*args, **kwargs)
    monkeypatch.setattr(discuss_loop, 'wait', waiting_wait)
    with _signal_when_waiting(signum, count=2, executor_ready=executor_ready):
        assert cli.main(
            ['discuss', '56', '--repo', 'OWNER/REPO', '--wait-for-quota-reset', '6h']
        ) == QuotaResetExceededError.EXIT_CODE
    assert time.monotonic() - started < 20
    assert terminations == [True]
    assert len(outcomes) == 2
    assert quota_wait.active_wait_count() == 0
    assert "quota reset wait interrupted" in capsys.readouterr().err


def scenario_sigterm_without_active_wait_chains_to_a_python_handler():
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


def scenario_ordinary_keyboard_interrupt_still_propagates(monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(cli, "run_pr_loop", lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        cli.main(PR_ARGS)
    assert signal.getsignal(signal.SIGTERM) == previous


def scenario_option_off_installs_no_sigterm_handler(monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    seen = []
    monkeypatch.setattr(cli, "run_pr_loop", lambda *_a, **_k: seen.append(signal.getsignal(signal.SIGTERM)) or 0)
    assert cli.main(["pr", "77", "--repo", "OWNER/REPO"]) == 0
    assert seen == [previous]
    monkeypatch.setattr(cli, "run_pr_loop", lambda *_a, **_k: seen.append(signal.getsignal(signal.SIGTERM)) or 0)
    assert cli.main(PR_ARGS) == 0
    assert seen[-1] != previous
    assert signal.getsignal(signal.SIGTERM) == previous


def scenario_repeated_main_after_cancelled_wait_waits_normally(tmp_path, monkeypatch, capsys):
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
    with _signal_when_waiting(signal.SIGINT):
        assert cli.main(PR_ARGS) == QuotaResetExceededError.EXIT_CODE
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


def scenario_option_off_keyboard_interrupt_clears_cancel_latch(tmp_path, monkeypatch):
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


def scenario_main_chains_sigterm_to_host_python_handler(monkeypatch):
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


def scenario_missing_readiness(tmp_path, monkeypatch, capsys, signum):
    delivered = []
    monkeypatch.setattr(signal, 'pthread_kill', lambda *args: delivered.append(args))
    with pytest.raises(AssertionError, match='readiness'):
        with _signal_when_waiting(signum, readiness_timeout=0.1):
            time.sleep(0.2)
    assert delivered == []


def scenario_stuck_child(tmp_path, monkeypatch, capsys, signum):
    print('intentional stuck child', flush=True)
    threading.Event().wait()


def scenario_condition_reacquire(tmp_path, monkeypatch, capsys, signum):
    # Inject one real signal exactly before lock restoration, rather than
    # pretending a timing-only probe reproduces the CI scheduling window.
    condition = threading.Condition(threading.Lock())
    quota_wait._active_waits.add(-1)
    if signum == signal.SIGTERM:
        quota_wait.install_sigterm_handler()
    elif signum == signal.SIGALRM:
        def alarm(_signum, _frame):
            raise TimeoutError('controlled alarm')
        signal.signal(signal.SIGALRM, alarm)
    restore = condition._acquire_restore
    events = []
    def interrupted_restore(state):
        events.append('reacquire-entry')
        signal.raise_signal(signum)
        restore(state)
    condition._acquire_restore = interrupted_restore
    try:
        with condition:
            condition.wait(0.001)
    except RuntimeError as exc:
        assert str(exc) == 'release unlocked lock'
        assert isinstance(exc.__context__, (KeyboardInterrupt, TimeoutError))
        traceback.print_exc(file=sys.stdout)
    else:
        raise AssertionError('expected interrupted reacquisition')
    assert events == ['reacquire-entry']
    quota_wait._active_waits.clear()
    # The non-interrupted control restores normally.
    condition._acquire_restore = restore
    with condition:
        assert condition.wait(0.001) is False


def scenario_sender_failure(tmp_path, monkeypatch, capsys, signum):
    quota_wait._active_waits.add(-1)
    def broken_sender(*args):
        raise OSError('controlled sender failure')
    monkeypatch.setattr(signal, 'pthread_kill', broken_sender)
    try:
        with pytest.raises(OSError, match='controlled sender failure'):
            with _signal_when_waiting(signum):
                time.sleep(0.05)
    finally:
        quota_wait._active_waits.clear()


def scenario_pending_alarm_during_sigterm_reacquire(tmp_path, monkeypatch, capsys, signum):
    condition = threading.Condition(threading.Lock())
    quota_wait._active_waits.add(-1)
    quota_wait.install_sigterm_handler()
    def alarm(*args):
        raise TimeoutError('controlled pending alarm')
    signal.signal(signal.SIGALRM, alarm)
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    signal.raise_signal(signal.SIGALRM)
    assert signal.SIGALRM in signal.sigpending()
    def interrupted_restore(state):
        try:
            signal.raise_signal(signal.SIGTERM)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    condition._acquire_restore = interrupted_restore
    try:
        with pytest.raises(RuntimeError, match='release unlocked lock') as caught:
            with condition:
                condition.wait(0.001)
        assert isinstance(caught.value.__context__, TimeoutError)
        assert isinstance(caught.value.__context__.__context__, KeyboardInterrupt)
        traceback.print_exception(caught.value, file=sys.stdout)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        quota_wait._active_waits.clear()


class _Capture:
    def __init__(self):
        self.stderr = io.StringIO()
    def readouterr(self):
        text = self.stderr.getvalue()
        self.stderr.seek(0)
        self.stderr.truncate()
        return SimpleNamespace(err=text, out='')


def main():
    # No pytest recursion: invoke one scenario directly and explicitly install
    # the same fake-runner isolation that the parent autouse fixtures provide.
    import conftest
    name, directory, number = sys.argv[1:]
    tmp_path = Path(directory)
    tmp_path.mkdir(parents=True, exist_ok=True)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    class Factory:
        def mktemp(self, prefix):
            return Path(tempfile.mkdtemp(prefix=prefix, dir=tmp_path))
    factory = Factory()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
    signal.signal(signal.SIGINT, signal.default_int_handler)
    capture = _Capture()
    with contextlib.ExitStack() as stack:
        monkeypatch = stack.enter_context(pytest.MonkeyPatch.context())
        fixture_args = {'monkeypatch': monkeypatch, 'tmp_path_factory': factory}
        fixtures = (
            '_no_real_repair', '_isolated_agy_log_dir', '_no_real_github_backoff',
            '_reset_checkout_baselines', '_fake_runner_private_transport',
            '_stub_tool_provenance', '_agent_commands_available',
            '_isolate_worker_budget_environment', '_isolate_workdir_claims',
            '_reset_quota_wait_state', '_isolate_store_locks',
            '_round_publication_needs_an_actor_surface',
        )
        for name_fixture in fixtures:
            fixture = getattr(conftest, name_fixture).__wrapped__
            kwargs = {key: fixture_args[key] for key in inspect.signature(fixture).parameters}
            if inspect.isgeneratorfunction(fixture):
                stack.enter_context(contextlib.contextmanager(fixture)(**kwargs))
            else:
                fixture(**kwargs)
        scenario = globals()['scenario_' + name]
        args = {'tmp_path': tmp_path, 'monkeypatch': monkeypatch,
                'capsys': capture, 'signum': int(number)}
        try:
            with contextlib.redirect_stderr(capture.stderr):
                scenario(**{key: args[key] for key in inspect.signature(scenario).parameters})
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            print(capture.stderr.getvalue(), end='')
            assert not any(t.name.startswith('quota-signal-sender') for t in threading.enumerate())
    print('scenario completed: ' + name)


if __name__ == '__main__':
    main()
