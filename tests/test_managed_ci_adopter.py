"""Executable adopter harness for the reusable workflow's shard entry point (#1210).

A temporary adopter project that has none of this repository's ``tests``
helpers runs the same ``ci/managed/run_shard.py`` the reusable workflows call,
so plugin loading, manifest production and trusted verification are proven
without GitHub Actions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CALLEE = ROOT / "ci" / "managed"
RUN_SHARD = CALLEE / "run_shard.py"
VERIFY = CALLEE / "ci_shard_verify.py"
RUN_ID = "9001"
PYTEST_COMMAND = f'"{sys.executable}" -m pytest -q -p no:cacheprovider'

sys.path.insert(0, str(CALLEE))
import run_shard  # noqa: E402

sys.path.remove(str(CALLEE))

TESTS = """
def test_one(): assert 1
def test_two(): assert 2
def test_three(): assert 3
def test_four(): assert 4
def test_five(): assert 5
def test_six(): assert 6
"""


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CI_SHARD", "PYTEST_ADDOPTS", "PYTHONPATH"))}
    env["GITHUB_RUN_ID"] = RUN_ID
    env["GITHUB_RUN_ATTEMPT"] = "1"
    env.update(extra)
    return env


def _git(project, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=project, check=True, capture_output=True,
    )


@pytest.fixture
def adopter(tmp_path):
    """A project that is not this repository: no conftest, no shard helpers."""
    project = tmp_path / "adopter"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_app.py").write_text(TESTS)
    _git(project, "init", "-q")
    _git(project, "add", "-A")
    _git(project, "commit", "-q", "-m", "adopter")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert not list(project.rglob("_ci_shard*")) and not list(project.rglob("ci_shard_verify*"))
    return project, head


def _leg(project, shards, index, command=PYTEST_COMMAND, manifest=None, durations=""):
    manifest = manifest or f"shard-manifest/{index}/manifest.json"
    return subprocess.run(
        [sys.executable, str(RUN_SHARD), "--shards", str(shards), "--index", str(index),
         "--manifest", manifest, "--durations", durations, "--test-command", command],
        cwd=project, env=_env(), capture_output=True, text=True,
    )


def _verify(project, head, count, result="success", directory="shard-manifest"):
    return subprocess.run(
        [sys.executable, str(VERIFY), directory, "--count", str(count), "--head", head,
         "--run-id", RUN_ID, "--result", result],
        cwd=project, capture_output=True, text=True,
    )


def test_single_shard_runs_the_command_unchanged_and_writes_no_manifest(adopter):
    project, _ = adopter
    done = _leg(project, 1, 1)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "6 passed" in done.stdout
    assert not (project / "shard-manifest").exists()


def test_single_shard_failure_is_the_command_failure(adopter):
    project, _ = adopter
    (project / "tests" / "test_app.py").write_text("def test_bad(): assert 0\n")
    assert _leg(project, 1, 1).returncode == 1


@pytest.mark.parametrize("durations", ["", "d.json"])
def test_three_shards_load_the_callee_plugin_and_verify_exactly_once(adopter, durations):
    project, head = adopter
    if durations:
        (project / durations).write_text(json.dumps({"tests/test_app.py::test_one": 9.0}))
    ran = 0
    for index in (1, 2, 3):
        done = _leg(project, 3, index, durations=durations)
        assert done.returncode == 0, done.stdout + done.stderr
        ran += int(done.stdout.split(" passed")[0].split()[-1])
    assert ran == 6  # every test ran in exactly one leg
    verified = _verify(project, head, 3)
    assert verified.returncode == 0, verified.stderr
    manifests = [json.loads(p.read_text()) for p in sorted((project / "shard-manifest").glob("*/manifest.json"))]
    assert [m["shard_index"] for m in manifests] == [1, 2, 3]
    assert sorted(i for m in manifests for i in m["selected_ids"]) == sorted(
        f"tests/test_app.py::test_{n}" for n in ("one", "two", "three", "four", "five", "six")
    )


def test_a_failing_leg_yields_failure_not_success(adopter):
    project, head = adopter
    (project / "tests" / "test_app.py").write_text(TESTS + "\ndef test_zz_bad(): assert 0\n")
    codes = {}
    for index in (1, 2, 3):
        done = _leg(project, 3, index)
        codes[index] = done.returncode
        if done.returncode != 0:
            # Only passing legs publish a manifest (the workflow uploads on success()).
            (project / "shard-manifest" / str(index) / "manifest.json").unlink()
    assert sorted(codes.values()).count(0) == 2 and max(codes.values()) == 1
    assert _verify(project, head, 3).returncode == 1
    assert _verify(project, head, 3, result="failure").returncode == 1


def test_a_missing_manifest_fails_the_leg_and_the_aggregate(adopter):
    project, head = adopter
    # Exits 0 but never loads the plugin, so no manifest can exist.
    done = _leg(project, 3, 1, command=f'"{sys.executable}" -m pytest --version')
    assert done.returncode == 1 and "wrote no manifest" in done.stderr
    assert _verify(project, head, 3).returncode == 1


def test_a_stale_manifest_cannot_satisfy_a_leg(adopter):
    project, _ = adopter
    manifest = project / "shard-manifest" / "1" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    done = _leg(project, 3, 1, command=f'"{sys.executable}" -m pytest --version')
    assert done.returncode == 1


def test_sharding_requires_a_pytest_command(adopter):
    project, _ = adopter
    done = _leg(project, 3, 1, command="make test")
    assert done.returncode == 2 and "requires a pytest test command" in done.stderr
    assert not (project / "shard-manifest").exists()


@pytest.mark.parametrize("shards,index", [(0, 1), (3, 0), (3, 4)])
def test_out_of_range_shards_are_rejected(adopter, shards, index):
    project, _ = adopter
    assert _leg(project, shards, index).returncode == 2


def test_environment_composition_is_what_the_workflow_relies_on():
    base = {"PATH": "/bin", "PYTHONPATH": "/x", "PYTEST_ADDOPTS": "-q", "CI_SHARD_INDEX": "9"}
    env = run_shard.compose(shards=3, index=2, manifest="m.json", durations="d.json", base_env=base)
    assert env["PYTHONPATH"].split(os.pathsep) == [str(CALLEE), "/x"]
    assert env["PYTEST_ADDOPTS"] == "-q -p ci_shard_plugin"
    assert (env["CI_SHARD_INDEX"], env["CI_SHARD_COUNT"]) == ("2", "3")
    assert env["CI_SHARD_MANIFEST"] == "m.json" and env["CI_SHARD_DURATIONS"] == "d.json"
    single = run_shard.compose(shards=1, index=1, manifest="m.json", durations="d.json", base_env=base)
    assert single == {"PATH": "/bin", "PYTHONPATH": "/x", "PYTEST_ADDOPTS": "-q"}
    no_durations = run_shard.compose(shards=2, index=1, manifest="m", durations="", base_env={})
    assert "CI_SHARD_DURATIONS" not in no_durations
