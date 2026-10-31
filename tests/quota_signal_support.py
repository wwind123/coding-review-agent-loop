"""Bound real signals to disposable interpreters, never pytest's signal state."""
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = Path(__file__).with_name('quota_signal_scenarios.py')


def child_environment():
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(('PYTEST_', 'PYTHON', 'AGENT_LOOP_', 'CI_SHARD_'))
        and key not in {'SHARDS', 'SHARD_INDEX', 'DURATIONS_FILE', 'TEST_COMMAND'}
    }
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT / 'src'), str(ROOT / 'tests')))
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    return env


class ChildDeadlineExceeded(AssertionError):
    def __init__(self, command, timeout, pid, output):
        self.pid = pid
        super().__init__(f'child deadline exceeded ({timeout}s): {command}\n{output}')


def run_child(command, *, timeout, env=None, cwd=None):
    """Kill descendants on all exits, with bounded output collection and reap."""
    if sys.platform != 'linux' or not hasattr(signal, 'pthread_kill'):
        pytest.skip('requires Linux POSIX signals and private process groups')
    process = subprocess.Popen(
        command, cwd=cwd or ROOT, env=env or child_environment(),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # The scenario registers SIGUSR1 solely for stack diagnostics. Nested
        # pytest controllers do not, so only request stacks from that helper.
        if str(SCENARIOS) in command:
            try:
                os.kill(process.pid, signal.SIGUSR1)
                process.communicate(timeout=0.2)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = process.communicate(timeout=5)
    if timed_out:
        raise ChildDeadlineExceeded(command, timeout, process.pid, output)
    return subprocess.CompletedProcess(command, process.returncode, output, output)


def run_signal_scenario(name, tmp_path, *, signum=signal.SIGINT, timeout=30):
    result = run_child(
        [sys.executable, str(SCENARIOS), name, str(tmp_path), str(int(signum))],
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout
    return result.stdout
