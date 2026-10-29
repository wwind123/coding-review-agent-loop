"""Exercise the repository timeout policy in isolated real pytest processes."""

import os
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
