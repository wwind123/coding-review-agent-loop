"""Exercise the repository timeout policy in isolated real pytest processes."""

import json
import os
import selectors
import time
from pathlib import Path
import signal
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "pyproject.toml"


def test_repository_timeout_policy():
    options = tomllib.loads(CONFIG.read_text())["tool"]["pytest"]["ini_options"]
    assert options["faulthandler_timeout"] == 300
    assert options["faulthandler_timeout"] < options["timeout"] < 30 * 60
    assert options["timeout"] == 600
    assert options["timeout_method"] == "signal"
    assert options.get("timeout_func_only", False) is False


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="signal timeout requires SIGALRM")
@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_named_timeout_and_fast_control(tmp_path, workers, phase):
    suite = tmp_path / "test_timeout_probe.py"
    suite.write_text(
        "import time\nimport pytest\n"
        "@pytest.fixture\ndef slow_fixture():\n"
        f"    {'time.sleep(10)' if phase == 'setup' else 'pass'}\n"
        "    yield\n"
        f"    {'time.sleep(10)' if phase == 'teardown' else 'pass'}\n"
        "def test_timeout_probe(slow_fixture):\n"
        f"    {'time.sleep(10)' if phase == 'call' else 'pass'}\n"
        "def test_fast_control():\n    assert True\n"
    )
    # Do not inherit the outer session's plugin injection, sharding, manifests,
    # worker identity, or environment-based timeout overrides. Explicitly load
    # only the two required plugins; child output stays under tmp_path.
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "PYTHON", "AGENT_LOOP_", "CI_SHARD_"))
        and key not in {"SHARDS", "SHARD_INDEX", "DURATIONS_FILE", "TEST_COMMAND"}
    }
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    command = [
        sys.executable, "-m", "pytest", "-c", str(CONFIG),
        "--rootdir", str(tmp_path), "--confcutdir", str(tmp_path),
        "-p", "pytest_timeout", "-p", "xdist.plugin", "-n", str(workers),
        "-p", "no:cacheprovider", "--timeout=0.5", "-q", str(suite),
    ]
    process = subprocess.Popen(
        command, cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=60)
    finally:
        # Also bound xdist descendants if the child controller gets stuck or
        # exits early; terminate the private process group before reaping it.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=10)
    assert process.returncode == 1, output
    assert "test_timeout_probe" in output
    assert "Timeout (>0.5s) from pytest-timeout" in output
    assert ("FAILED" if phase == "call" else "ERROR") in output
    assert ("2 passed" if phase == "teardown" else "1 passed") in output
    assert "crashed" not in output.lower()
    assert "replacing crashed worker" not in output.lower()
    assert not list(tmp_path.rglob("manifest*.json"))


@pytest.mark.skipif(sys.platform != 'linux', reason='requires Linux POSIX signal scenarios')
@pytest.mark.parametrize('workers', [0, 2], ids=['serial', 'xdist'])
def test_quota_signal_scenarios_preserve_timeout_continuity(tmp_path, workers):
    from quota_signal_support import child_environment, run_child
    suite = tmp_path / 'test_signal_continuity.py'
    suite.write_text('''import os, signal, time
import pytest
from quota_signal_support import run_signal_scenario

class TestContinuity:
    def test_01_successful_signals(self, tmp_path):
        before = signal.getsignal(signal.SIGALRM)
        timer = signal.getitimer(signal.ITIMER_REAL)
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        run_signal_scenario('signal_cancels_a_waiting_reviewer_worker', tmp_path / 'reviewer', signum=signal.SIGINT)
        run_signal_scenario('signal_cancels_waiting_discuss_debaters', tmp_path / 'discuss', signum=signal.SIGTERM)
        assert signal.getsignal(signal.SIGALRM) is before
        assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == mask
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 0 < remaining <= timer[0] and interval == timer[1]
        (tmp_path.parent / 'worker-pid').write_text(str(os.getpid()))

    def test_02_stuck_signal_child(self, tmp_path):
        before = signal.getsignal(signal.SIGALRM)
        timer = signal.getitimer(signal.ITIMER_REAL)
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        try:
            run_signal_scenario('stuck_child', tmp_path, timeout=5)
        finally:
            assert signal.getsignal(signal.SIGALRM) is before
            assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == mask
            remaining, interval = signal.getitimer(signal.ITIMER_REAL)
            assert 0 < remaining <= timer[0] and interval == timer[1]

    def test_03_fast_control(self, tmp_path):
        assert (tmp_path.parent / 'worker-pid').read_text() == str(os.getpid())
        (tmp_path.parent / 'fast-control').write_text('ran')

    @pytest.mark.timeout(0.5)
    def test_04_named_timeout(self, tmp_path):
        assert (tmp_path.parent / 'fast-control').read_text() == 'ran'
        assert (tmp_path.parent / 'worker-pid').read_text() == str(os.getpid())
        time.sleep(10)
''')
    env = child_environment()
    command = [
        sys.executable, '-m', 'pytest', '-c', str(CONFIG), '--rootdir', str(tmp_path),
        '--confcutdir', str(tmp_path), '-p', 'pytest_timeout', '-p', 'xdist.plugin',
        # The entire class is one continuity sequence. Use structural class
        # affinity rather than marker-based grouping: splitting this sequence
        # across workers makes the control/timeout checks meaningless.
        '-n', str(workers), '--dist=loadscope', '-p', 'no:cacheprovider',
        '--timeout=20', '-q', str(suite),
    ]
    result = run_child(command, env=env, cwd=tmp_path, timeout=60)
    assert result.returncode == 1, result.stdout
    assert 'test_02_stuck_signal_child' in result.stdout
    assert 'child deadline exceeded (5s)' in result.stdout
    assert 'test_04_named_timeout' in result.stdout
    assert 'Timeout (>0.5s) from pytest-timeout' in result.stdout
    assert '2 failed, 2 passed' in result.stdout
    assert 'crashed' not in result.stdout.lower()
    assert 'replacing' not in result.stdout.lower()
    assert not list(tmp_path.rglob('manifest*.json'))


# The built-in plugin and this milestone channel independently duplicate stderr
# while global capture is suspended. Never change the plugin's own dump path.
DIAGNOSTIC_CONFTEST = '''import json, os, signal, time
import pytest

live_fd = None

def pytest_configure(config):
    global live_fd
    assert config.getoption("capture") == "fd"
    live_fd = os.dup(2)

def pytest_unconfigure(config):
    global live_fd
    if live_fd is not None:
        os.close(live_fd)
        live_fd = None

def record(kind, **fields):
    os.write(live_fd, ("PROBE " + json.dumps(dict(kind=kind, **fields)) + "\\n").encode())

def stalled_DIAGNOSTIC_FRAME():
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    assert remaining > 0 and interval == 0
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    ready = time.monotonic()
    record("ready", remaining=remaining, at=ready)
    expired = False
    while True:
        now = time.monotonic()
        if not expired and now > ready + remaining + 0.1:
            timer = signal.getitimer(signal.ITIMER_REAL)
            assert timer == (0.0, 0.0)
            record("expired", at=now, timer=timer)
            expired = True
        time.sleep(0.01)
'''


def diagnostic_command(tmp_path, workers, *, signal_timeout, diagnostic_timeout):
    """Use repository policy, explicit plugins, and unchanged default capture."""
    return [
        sys.executable, "-m", "pytest", "-c", str(CONFIG),
        "--rootdir", str(tmp_path), "--confcutdir", str(tmp_path),
        "-p", "pytest_timeout", "-p", "xdist.plugin", "-n", str(workers),
        "--dist=loadscope", "-p", "no:cacheprovider", "--timeout-method=signal",
        f"--timeout={signal_timeout}", "-o", f"faulthandler_timeout={diagnostic_timeout}",
        "-q", str(tmp_path / "test_diagnostic_probe.py"),
    ]


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux process groups and SIGALRM")
@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_independent_diagnostic_after_ignored_signal(tmp_path, workers, phase):
    from quota_signal_support import child_environment

    frame = f"stalled_{phase}_DIAGNOSTIC_FRAME"
    (tmp_path / "conftest.py").write_text(DIAGNOSTIC_CONFTEST.replace("stalled_DIAGNOSTIC_FRAME", frame))
    (tmp_path / "test_diagnostic_probe.py").write_text(
        f"from conftest import {frame}\nimport pytest\n"
        "@pytest.fixture\ndef probe():\n"
        f"    {frame + '()' if phase == 'setup' else 'pass'}\n"
        "    yield\n"
        f"    {frame + '()' if phase == 'teardown' else 'pass'}\n"
        "def test_diagnostic_probe(probe):\n"
        f"    {frame + '()' if phase == 'call' else 'pass'}\n"
    )
    command = diagnostic_command(tmp_path, workers, signal_timeout=1, diagnostic_timeout=3)
    process = subprocess.Popen(
        command, cwd=tmp_path, env=child_environment(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
    )
    output = b""
    try:
        # Read live bytes rather than waiting for a summary from a deliberate hang.
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                for key, _ in selector.select(timeout=0.2):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    assert chunk, output.decode(errors="replace")
                    output += chunk
                if b"Timeout (0:00:03)!" in output and frame.encode() in output:
                    break
                assert process.poll() is None, output.decode(errors="replace")
        text = output.decode(errors="replace")
        assert process.poll() is None, text
        lines = [json.loads(line.split("PROBE ", 1)[1]) for line in text.splitlines() if "PROBE " in line]
        assert [line["kind"] for line in lines] == ["ready", "expired"], text
        ready, expired = lines
        assert ready["remaining"] > 0
        assert expired["at"] > ready["at"] + ready["remaining"]
        assert expired["timer"] == [0.0, 0.0]
        header = text.index("Timeout (0:00:03)!")
        assert text.index('"kind": "ready"') < text.index('"kind": "expired"') < header
        assert frame in text[header:], text
        assert "from pytest-timeout" not in text
        assert "short test summary" not in text
        assert " passed" not in text and " failed" not in text
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=10)
        assert process.poll() is not None


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux process groups")
@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
@pytest.mark.parametrize("control", ["fast", "next-test"])
def test_diagnostic_success_and_timer_cancellation(tmp_path, workers, control):
    from quota_signal_support import child_environment, run_child

    (tmp_path / "conftest.py").write_text(DIAGNOSTIC_CONFTEST)
    # Class affinity keeps both tests on the same worker. Each completes below
    # its own two-second timer, but the second crosses the first timer's deadline.
    delay = 1.2 if control == "next-test" else 0.01
    (tmp_path / "test_diagnostic_probe.py").write_text(
        "import time\nclass TestControls:\n"
        f"    def test_01_completes(self):\n        time.sleep({delay})\n"
        f"    def test_02_crosses_previous_deadline(self):\n        time.sleep({delay})\n"
    )
    command = diagnostic_command(tmp_path, workers, signal_timeout=10, diagnostic_timeout=2)
    result = run_child(command, env=child_environment(), cwd=tmp_path, timeout=45)
    assert result.returncode == 0, result.stdout
    assert "2 passed" in result.stdout
    assert "Timeout (" not in result.stdout
    assert "from pytest-timeout" not in result.stdout
