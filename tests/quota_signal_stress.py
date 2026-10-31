"""Bounded whole-file repetitions, with cooperative outer-supervisor cleanup."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run_supervised(command, *, timeout):
    """Let the harness kill/reap its private iteration group before returning.

    Unlike subprocess.run's timeout path, termination goes through the harness's
    cancellation handler and finally block. This also applies to caller interrupts.
    """
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    try:
        return process.wait(timeout=timeout)
    finally:
        if process.poll() is None:
            process.terminate()
        # The harness polls cancellation every 0.1s and bounds iteration reap at
        # 10s. Keep this grace below the stress test's own pytest timeout.
        process.wait(timeout=15)


def run_iterations(command, *, root, iterations, per_run_timeout, label, workers, env):
    cancelled = False

    def cancel(_signum, _frame):
        nonlocal cancelled
        # Do not raise between Popen and its cleanup guard. A pending stop is
        # observed as soon as the process handle is safely owned.
        cancelled = True

    previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    summary = {
        'command': command, 'python': sys.version,
        'plugins': {name: importlib.metadata.version(name)
                    for name in ('pytest', 'pytest-xdist', 'pytest-timeout')},
        'attempts': 0, 'passes': 0, 'failures': 0, 'timeouts': 0,
    }
    start = time.monotonic()
    try:
        for i in range(iterations):
            if cancelled:
                break
            with (root / f'{i + 1:03}.log').open('w') as log:
                proc = subprocess.Popen(
                    command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                deadline = time.monotonic() + per_run_timeout
                try:
                    while True:
                        if cancelled:
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            summary['timeouts'] += 1
                            break
                        try:
                            proc.wait(timeout=min(0.1, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    code = proc.wait(timeout=10)
            summary['attempts'] += 1
            summary['passes' if code == 0 and not cancelled else 'failures'] += 1
            print(f'{label} workers={workers} iteration={i + 1} exit={code}', flush=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary['elapsed_seconds'] = time.monotonic() - start
    summary['cancelled'] = cancelled
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return int(cancelled or bool(summary['failures']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--per-run-timeout', type=int, default=120)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    root = Path('/tmp/coding-review-agent-loop/scratch') / f'{args.label}-{args.workers}'
    root.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PYTEST_', 'CI_SHARD_'))
           and k not in {'SHARDS', 'SHARD_INDEX', 'DURATIONS_FILE', 'TEST_COMMAND'}}
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               '-p', 'no:randomly', '-n', str(args.workers), 'tests/test_quota_reset_wait.py']
    return run_iterations(command, root=root, iterations=args.iterations,
                          per_run_timeout=args.per_run_timeout, label=args.label,
                          workers=args.workers, env=env)


if __name__ == '__main__':
    sys.exit(main())
