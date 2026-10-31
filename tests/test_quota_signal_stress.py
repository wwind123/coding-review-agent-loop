"""Opt-in matched whole-file stress evidence for #1814.

Run with QUOTA_SIGNAL_STRESS=1 and QUOTA_SIGNAL_STRESS_WORKERS=0 or 2,
once per configuration so each gets its own whole-command watchdog. Ordinary
CI collects but skips the expensive repetition. The harness is byte-identical to the external baseline harness.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


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
    assert harness.read_bytes() == (scratch / 'quota_signal_stress.py').read_bytes()
    command = [sys.executable, str(harness), '--iterations', '50', '--workers', str(workers),
               '--per-run-timeout', '120', '--label', 'fixed']
    # Output remains visible in the managed foreground run (-s). Every child
    # has its own process-group deadline inside the harness, below this cap.
    result = subprocess.run(command, stdin=subprocess.DEVNULL, timeout=1550)
    baseline = json.loads((scratch / f'baseline-{workers}' / 'summary.json').read_text())
    fixed = json.loads((scratch / f'fixed-{workers}' / 'summary.json').read_text())
    assert baseline['attempts'] == fixed['attempts'] == 50
    assert baseline['command'] == fixed['command']
    assert baseline['python'] == fixed['python']
    assert baseline['plugins'] == fixed['plugins']
    assert result.returncode == 0
    assert fixed['passes'] == 50 and fixed['failures'] == fixed['timeouts'] == 0
