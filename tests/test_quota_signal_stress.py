"""Opt-in matched whole-file stress evidence for #1814.

Run with QUOTA_SIGNAL_STRESS=1 and QUOTA_SIGNAL_STRESS_WORKERS=0 or 2,
once per configuration so each gets its own whole-command watchdog. Ordinary
CI collects but skips the expensive repetition. The iteration command and environment match the external baseline harness.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from quota_signal_stress import run_supervised


@pytest.mark.skipif(os.environ.get('QUOTA_SIGNAL_STRESS') != '1', reason='opt-in 50-iteration stress')
@pytest.mark.timeout(1600)
@pytest.mark.parametrize('workers', [0, 2], ids=['serial', 'xdist'])
def test_matched_whole_file_stress(workers):
    selected = os.environ.get('QUOTA_SIGNAL_STRESS_WORKERS')
    assert selected in {'0', '2'}, 'select one stress configuration per bounded invocation'
    if workers != int(selected):
        pytest.skip('the other configuration runs in a separate bounded invocation')
    harness = Path(__file__).with_name('quota_signal_stress.py')
    scratch = Path('/tmp/coding-review-agent-loop/scratch')
    command = [sys.executable, str(harness), '--iterations', '50', '--workers', str(workers),
               '--per-run-timeout', '120', '--label', 'fixed']
    # Output remains visible in the managed foreground run (-s). Every child
    # has its own process-group deadline inside the harness, below this cap.
    returncode = run_supervised(command, timeout=1550)
    baseline = json.loads((scratch / f'baseline-{workers}' / 'summary.json').read_text())
    fixed = json.loads((scratch / f'fixed-{workers}' / 'summary.json').read_text())
    assert baseline['attempts'] == fixed['attempts'] == 50
    assert baseline['command'] == fixed['command']
    assert baseline['python'] == fixed['python']
    assert baseline['plugins'] == fixed['plugins']
    assert returncode == 0
    assert fixed['passes'] == 50 and fixed['failures'] == fixed['timeouts'] == 0


@pytest.mark.parametrize('stop', ['outer-timeout', 'interrupt', 'harness-interrupt', 'iteration-timeout', 'normal'])
def test_stress_supervisor_reaps_active_iteration_and_stops_descendants(tmp_path, monkeypatch, stop):
    import signal
    import time

    if sys.platform != 'linux':
        pytest.skip('requires Linux process groups and /proc exit observation')
    # A real iteration creates a descendant. A rendezvous ensures that the
    # cancellation under test occurs while both are alive, not during startup.
    pid_file = tmp_path / 'pids.json'
    release = tmp_path / 'release'
    iteration = tmp_path / 'iteration.py'
    iteration.write_text(
        'import json, os, subprocess, sys, time\n'
        'from pathlib import Path\n'
        'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n'
        f'Path({str(pid_file)!r}).write_text(json.dumps([os.getpid(), child.pid]))\n'
        f'while not Path({str(release)!r}).exists(): time.sleep(0.01)\n'
    )
    driver = tmp_path / 'driver.py'
    driver.write_text(
        'import os, sys\nfrom pathlib import Path\n'
        f'sys.path.insert(0, {str(Path(__file__).parent)!r})\n'
        'from quota_signal_stress import run_iterations\n'
        f'sys.exit(run_iterations([sys.executable, {str(iteration)!r}], '
        f'root=Path({str(tmp_path)!r}), iterations=1, '
        f'per_run_timeout={2 if stop == "iteration-timeout" else 60}, '
        'label="probe", workers=0, env=os.environ.copy()))\n'
    )
    original_wait = subprocess.Popen.wait
    supervisor_pid = None

    def wait_after_ready(process, timeout=None):
        nonlocal supervisor_pid
        # Intercept just the supervisor's initial wait, preserving real waits
        # for cleanup and subprocess creation internals.
        if supervisor_pid is None:
            supervisor_pid = process.pid
            deadline = time.monotonic() + 5
            while not pid_file.exists():
                assert time.monotonic() < deadline, 'iteration did not start'
                time.sleep(0.01)
            if stop == 'interrupt':
                raise KeyboardInterrupt
            if stop == 'outer-timeout':
                return original_wait(process, timeout=0.05)
            if stop == 'harness-interrupt':
                os.kill(process.pid, signal.SIGINT)
            if stop == 'normal':
                release.touch()
        return original_wait(process, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, 'wait', wait_after_ready)
    expected = {'outer-timeout': subprocess.TimeoutExpired, 'interrupt': KeyboardInterrupt}
    try:
        if stop in expected:
            with pytest.raises(expected[stop]):
                run_supervised([sys.executable, str(driver)], timeout=10)
        else:
            assert run_supervised([sys.executable, str(driver)], timeout=10) == (
                0 if stop == 'normal' else 1
            )
        pids = json.loads(pid_file.read_text())
        assert not Path(f'/proc/{supervisor_pid}').exists(), 'harness was not reaped'
        assert not Path(f'/proc/{pids[0]}').exists(), 'iteration was not reaped'
        # A killed grandchild may remain a zombie until host init reaps it;
        # it must not remain running or hold open descriptors.
        for pid in pids:
            stat = Path(f'/proc/{pid}/stat')
            assert not stat.exists() or stat.read_text().split(') ')[1].startswith('Z ')
        summary = json.loads((tmp_path / 'summary.json').read_text())
        assert summary['attempts'] == 1
        assert summary['cancelled'] == (stop in expected or stop == 'harness-interrupt')
        assert summary['timeouts'] == (stop == 'iteration-timeout')
    finally:
        # Keep a failing regression bounded as well.
        if pid_file.exists():
            for pid in json.loads(pid_file.read_text()):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
