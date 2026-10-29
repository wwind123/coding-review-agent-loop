"""CI test sharding: partitioner, plugin, manifest verifier, workflow harness (#1235)."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import importlib.util

import yaml

from fixtures.managed_ci import publisher

pytest_plugins = ["pytester"]

TESTS_DIR = Path(__file__).parent
ROOT = TESTS_DIR.parent
CALLEE_DIR = ROOT / "ci" / "managed"
# The plugin, runner and verifier are owned by the reusable workflow
# (ci/managed), not by this repository's tests. Managed CI puts the *base*
# branch's ci/managed first on PYTHONPATH and loads its ci_shard_plugin into the
# outer session, so importing them by name would test the base copy instead of
# the one this checkout changes. Load this checkout's copies by path instead.
if str(CALLEE_DIR) not in sys.path:
    sys.path.append(str(CALLEE_DIR))


def _load_checkout_callee(name):
    spec = importlib.util.spec_from_file_location(f"checkout_{name}", CALLEE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


_ci_shard = _load_checkout_callee("ci_shard_plugin")
run_shard = _load_checkout_callee("run_shard")
ci_shard_verify = _load_checkout_callee("ci_shard_verify")
VERIFIER = CALLEE_DIR / "ci_shard_verify.py"
# The copy the outer session injected with ``-p ci_shard_plugin`` (None in an
# ordinary local run); only its stash key finds the recorded real collection.
_ACTIVE_CI_SHARD = sys.modules.get("ci_shard_plugin")


def _use_checkout_plugin(monkeypatch):
    # In-process pytester sessions resolve ``-p ci_shard_plugin`` through
    # sys.modules; point that name at this checkout's copy. Subprocess sessions
    # get it from shard_env's PYTHONPATH instead.
    monkeypatch.setitem(sys.modules, "ci_shard_plugin", _ci_shard)


@pytest.fixture(autouse=True)
def _nested_sessions_use_checkout_plugin(monkeypatch):
    _use_checkout_plugin(monkeypatch)


def _recorded_full_collection(request):
    if _ACTIVE_CI_SHARD is None:
        return None
    return request.config.stash.get(_ACTIVE_CI_SHARD._FULL_KEY, None)
ORDINARY_WORKFLOW = ROOT / ".github" / "workflows" / "managed-ci-ordinary.yml"
MANAGED_WORKFLOW = ROOT / ".github" / "workflows" / "managed-ci.yml"
HEAD = "b" * 40
RUN_ID = "4242"


# --------------------------------------------------------------------------
# Partition function
# --------------------------------------------------------------------------


def test_partition_is_deterministic_across_orderings_and_covers_exactly_once():
    ids = [f"t.py::test_{i}" for i in range(40)]
    durations = {i: float(n % 7) for n, i in enumerate(ids)}
    first = _ci_shard.partition(ids, durations, 3)
    assert first == _ci_shard.partition(list(reversed(ids)), durations, 3)
    flat = [i for group in first for i in group]
    assert sorted(flat) == sorted(ids) and len(flat) == len(set(flat))
    _ci_shard.check_exactly_once(first, ids)


def test_partition_balances_a_skewed_fixture():
    durations = {"big::a": 100.0, "big::b": 100.0, "big::c": 100.0}
    durations.update({f"small::{i}": 1.0 for i in range(30)})
    groups = _ci_shard.partition(list(durations), durations, 3)
    loads = [sum(durations[i] for i in group) for group in groups]
    assert max(loads) - min(loads) <= 1.0


def test_unknown_ids_get_mean_estimate_and_stale_ids_are_ignored():
    durations = {"a": 10.0, "b": 10.0, "stale::gone": 999.0}
    groups = _ci_shard.partition(["a", "b", "new1", "new2"], durations, 2)
    assert sorted(i for g in groups for i in g) == ["a", "b", "new1", "new2"]
    assert all(len(g) == 2 for g in groups)
    assert _ci_shard.partition(["x", "y"], {}, 2) == [["x"], ["y"]]


def test_partition_count_one_keeps_everything():
    assert _ci_shard.partition(["b", "a"], {}, 1) == [["a", "b"]]


def test_exactly_once_guard_rejects_overlap_and_gaps():
    with pytest.raises(AssertionError):
        _ci_shard.check_exactly_once([["a"], ["a", "b"]], ["a", "b"])
    with pytest.raises(AssertionError):
        _ci_shard.check_exactly_once([["a"], []], ["a", "b"])


@pytest.mark.parametrize("module", ["ci_shard_plugin", "run_shard", "ci_shard_verify"])
def test_callee_modules_under_test_are_this_checkouts_copies(module):
    # Managed CI runs this file under the base branch's callee copies; the tests
    # must still exercise the copies this checkout changes.
    loaded = {"ci_shard_plugin": _ci_shard, "run_shard": run_shard, "ci_shard_verify": ci_shard_verify}[module]
    assert Path(loaded.__file__).resolve() == (CALLEE_DIR / f"{module}.py").resolve()


def test_in_process_sessions_load_this_checkouts_plugin_over_a_preloaded_copy(pytester, monkeypatch, tmp_path):
    # Simulate CI: another ci_shard_plugin is already imported under the plugin's name.
    decoy = type(sys)("ci_shard_plugin")
    decoy.__file__ = "/base-branch/ci/managed/ci_shard_plugin.py"
    monkeypatch.setitem(sys.modules, "ci_shard_plugin", decoy)
    _use_checkout_plugin(monkeypatch)
    record = tmp_path / "plugin-file.txt"
    monkeypatch.setenv("PLUGIN_FILE_RECORD", str(record))
    pytester.makeconftest("""
import os
from pathlib import Path
def pytest_configure(config):
    plugin = config.pluginmanager.get_plugin("ci_shard_plugin")
    Path(os.environ["PLUGIN_FILE_RECORD"]).write_text(str(Path(plugin.__file__).resolve()))
""")
    pytester.makepyfile(test_suite="def test_a(): pass")
    result = pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
    result.assert_outcomes(passed=1)
    assert record.read_text() == str((CALLEE_DIR / "ci_shard_plugin.py").resolve())


def test_real_collection_is_covered_exactly_once(request):
    # The plugin is injected by the reusable workflow (``-p ci_shard_plugin``);
    # an ordinary local run never loads it, so there is no recorded collection.
    full = _recorded_full_collection(request)
    if not full:
        pytest.skip("collection not recorded in this process")
    assert full
    groups = _ci_shard.partition(full, _ci_shard._load_durations(TESTS_DIR / ".test_durations"), 3)
    _ci_shard.check_exactly_once(groups, full)


# --------------------------------------------------------------------------
# Plugin (pytester)
# --------------------------------------------------------------------------

# pytester points HOME at a temporary directory, which hides a user-site
# install of pytest from child interpreters; hand them the directories this
# process imported pytest and xdist from.
_INSTALL_DIRS = os.pathsep.join(
    sorted({str(Path(m.__file__).resolve().parent.parent) for m in (pytest, yaml)})
)

SUITE = """
import time
def test_a(): time.sleep(0.01)
def test_b(): pass
def test_c(): pass
def test_d(): pass
def test_e(): pass
"""


@pytest.fixture
def shard_env(monkeypatch):
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(CALLEE_DIR), _INSTALL_DIRS, os.environ.get("PYTHONPATH", "")])
    )

    def apply(**values):
        for key, value in values.items():
            monkeypatch.setenv(key, str(value))

    return apply


def test_plugin_is_inert_without_shard_variables(pytester):
    pytester.makepyfile(test_suite=SUITE)
    result = pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
    result.assert_outcomes(passed=5)
    assert not list(pytester.path.glob("**/*.json"))


@pytest.mark.parametrize(
    "env",
    [
        {"CI_SHARD_INDEX": "1"},
        {"CI_SHARD_COUNT": "3"},
        {"CI_SHARD_INDEX": "x", "CI_SHARD_COUNT": "3"},
        {"CI_SHARD_INDEX": "4", "CI_SHARD_COUNT": "3"},
        {"CI_SHARD_INDEX": "0", "CI_SHARD_COUNT": "3"},
        {"CI_SHARD_INDEX": "1", "CI_SHARD_COUNT": "0"},
    ],
)
def test_malformed_shard_configuration_fails_closed(pytester, shard_env, env):
    shard_env(**env)
    pytester.makepyfile(test_suite=SUITE)
    result = pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*CI_SHARD*"])


def test_shards_select_disjoint_cover_and_write_manifest(pytester, shard_env, tmp_path):
    pytester.makepyfile(test_suite=SUITE)
    selected = []
    for index in (1, 2, 3):
        manifest = tmp_path / f"m{index}" / "manifest.json"
        shard_env(CI_SHARD_INDEX=index, CI_SHARD_COUNT=3, CI_SHARD_MANIFEST=manifest,
                  GITHUB_RUN_ID=RUN_ID, GITHUB_RUN_ATTEMPT=2)
        pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
        data = json.loads(manifest.read_text())
        assert data["schema_version"] == 1
        assert (data["shard_index"], data["shard_count"]) == (index, 3)
        assert data["full_collection_size"] == 5
        assert data["run_id"] == RUN_ID and data["run_attempt"] == 2
        assert data["full_collection_digest"] == ci_shard_verify.collection_digest(
            [f"test_suite.py::test_{c}" for c in "abcde"]
        )
        selected.extend(data["selected_ids"])
    assert sorted(selected) == [f"test_suite.py::test_{c}" for c in "abcde"]


def test_xdist_manifest_is_written_once_and_matches_shard(pytester, shard_env, tmp_path):
    pytest.importorskip("xdist")
    pytester.makepyfile(test_suite=SUITE)
    manifest = tmp_path / "x" / "manifest.json"
    shard_env(CI_SHARD_INDEX=2, CI_SHARD_COUNT=2, CI_SHARD_MANIFEST=manifest)
    result = pytester.runpytest_subprocess("-p", "ci_shard_plugin", "-n", "2", "-q")
    assert result.ret == 0
    data = json.loads(manifest.read_text())
    assert data["shard_index"] == 2
    assert result.parseoutcomes()["passed"] == len(data["selected_ids"])
    assert not list((tmp_path / "x").glob("*.tmp*"))


def test_duration_refresh_is_written_by_plain_session_and_controller(pytester, shard_env, tmp_path):
    pytester.makepyfile(test_suite="""
import time
import pytest

@pytest.fixture
def slow_setup():
    time.sleep(0.05)
    yield
    time.sleep(0.05)

def test_a(slow_setup): time.sleep(0.05)
def test_b(): pass
def test_c(): pass
def test_d(): pass
""")
    plain = tmp_path / "plain.json"
    shard_env(CI_SHARD_STORE_DURATIONS=plain)
    pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
    data = json.loads(plain.read_text())
    assert set(data) == {f"test_suite.py::test_{c}" for c in "abcd"}
    assert data["test_suite.py::test_a"] >= 0.15  # setup + call + teardown

    pytest.importorskip("xdist")
    dist = tmp_path / "dist.json"
    shard_env(CI_SHARD_STORE_DURATIONS=dist)
    result = pytester.runpytest_subprocess("-p", "ci_shard_plugin", "-n", "2", "-q")
    assert result.ret == 0
    data = json.loads(dist.read_text())
    assert set(data) == {f"test_suite.py::test_{c}" for c in "abcd"}
    assert data["test_suite.py::test_a"] >= 0.15



# --------------------------------------------------------------------------
# Durations store mode (#1660)
# --------------------------------------------------------------------------

STORE_SUITE = """
def test_a(): pass
def test_b(): pass
def test_c(): pass
"""
STORE_IDS = {f"test_suite.py::test_{c}" for c in "abc"}


def _seed_store(path):
    seeded = {
        "test_suite.py::test_a": 123.456,  # an existing test's sentinel timing
        "stale::gone": 9.0,  # a test that no longer exists
        "odd::non_numeric": "not-a-number",
    }
    path.write_text(json.dumps(seeded, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return seeded


@pytest.mark.parametrize("workers", [None, "2"])
def test_add_missing_adds_only_absent_ids_and_keeps_existing_entries(pytester, shard_env, tmp_path, workers):
    if workers:
        pytest.importorskip("xdist")
    pytester.makepyfile(test_suite=STORE_SUITE)
    store = tmp_path / "store.json"
    seeded = _seed_store(store)
    shard_env(CI_SHARD_STORE_DURATIONS=store, CI_SHARD_STORE_DURATIONS_MODE="add-missing")
    if workers:
        result = pytester.runpytest_subprocess("-p", "ci_shard_plugin", "-n", workers, "-q")
    else:
        result = pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
    assert result.ret == 0
    data = json.loads(store.read_text())
    assert set(data) == set(seeded) | STORE_IDS
    for key, value in seeded.items():
        assert data[key] == value
    for key in STORE_IDS - set(seeded):
        assert isinstance(data[key], float)
    assert not list(tmp_path.glob("*.tmp*"))


def test_add_missing_leaves_a_complete_store_byte_identical(pytester, shard_env, tmp_path):
    pytester.makepyfile(test_suite=STORE_SUITE)
    store = tmp_path / "store.json"
    # Deliberately non-canonical formatting: a no-op must not even reformat it.
    store.write_text('{"test_suite.py::test_c": 3, "test_suite.py::test_a": 1.5,\n"test_suite.py::test_b": 2}')
    before = store.read_bytes()
    shard_env(CI_SHARD_STORE_DURATIONS=store, CI_SHARD_STORE_DURATIONS_MODE="add-missing")
    result = pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
    assert result.ret == 0
    assert store.read_bytes() == before


def test_add_missing_creates_a_missing_store(pytester, shard_env, tmp_path):
    pytester.makepyfile(test_suite=STORE_SUITE)
    store = tmp_path / "new" / "store.json"
    shard_env(CI_SHARD_STORE_DURATIONS=store, CI_SHARD_STORE_DURATIONS_MODE="add-missing")
    result = pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
    assert result.ret == 0
    assert set(json.loads(store.read_text())) == STORE_IDS


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '"text"'])
def test_add_missing_fails_closed_on_a_malformed_store(pytester, shard_env, tmp_path, content):
    pytester.makepyfile(test_suite=STORE_SUITE)
    store = tmp_path / "store.json"
    store.write_text(content)
    before = store.read_bytes()
    shard_env(CI_SHARD_STORE_DURATIONS=store, CI_SHARD_STORE_DURATIONS_MODE="add-missing")
    result = pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*CI_SHARD_STORE_DURATIONS_MODE=add-missing*"])
    assert "passed" not in result.stdout.str()
    assert store.read_bytes() == before


def test_invalid_store_mode_is_a_usage_error(pytester, shard_env, tmp_path):
    pytester.makepyfile(test_suite=STORE_SUITE)
    store = tmp_path / "store.json"
    seeded = _seed_store(store)
    shard_env(CI_SHARD_STORE_DURATIONS=store, CI_SHARD_STORE_DURATIONS_MODE="merge")
    result = pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*CI_SHARD_STORE_DURATIONS_MODE*'merge'*"])
    assert json.loads(store.read_text()) == seeded


@pytest.mark.parametrize("mode", [None, "", "replace"])
def test_replace_mode_still_rewrites_the_store_with_this_session_only(pytester, shard_env, tmp_path, mode):
    pytester.makepyfile(test_suite=STORE_SUITE)
    store = tmp_path / "store.json"
    _seed_store(store)
    shard_env(CI_SHARD_STORE_DURATIONS=store)
    if mode is not None:
        shard_env(CI_SHARD_STORE_DURATIONS_MODE=mode)
    result = pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
    assert result.ret == 0
    data = json.loads(store.read_text())
    assert set(data) == STORE_IDS
    assert data["test_suite.py::test_a"] != 123.456


@pytest.mark.parametrize(
    "raw,expected", [(None, "replace"), ("", "replace"), ("replace", "replace"), ("add-missing", "add-missing")]
)
def test_parse_env_store_mode_values(raw, expected):
    environ = {} if raw is None else {"CI_SHARD_STORE_DURATIONS_MODE": raw}
    assert _ci_shard.parse_env(environ).store_mode == expected


@pytest.mark.parametrize("raw", ["Add-Missing", "append", " replace"])
def test_parse_env_rejects_unknown_store_modes(raw):
    with pytest.raises(pytest.UsageError, match="CI_SHARD_STORE_DURATIONS_MODE"):
        _ci_shard.parse_env({"CI_SHARD_STORE_DURATIONS_MODE": raw})


def test_store_mode_variable_is_declared_and_stripped_from_shard_legs(monkeypatch):
    assert "CI_SHARD_STORE_DURATIONS_MODE" in _ci_shard.SHARD_ENV_VARS
    assert run_shard.SHARD_ENV_VARS == _ci_shard.SHARD_ENV_VARS
    base = {"PATH": "/bin", "CI_SHARD_STORE_DURATIONS": "x.json", "CI_SHARD_STORE_DURATIONS_MODE": "add-missing"}
    for shards in (1, 3):
        env = run_shard.compose(shards=shards, index=1, manifest="m.json", durations="", base_env=base)
        assert "CI_SHARD_STORE_DURATIONS_MODE" not in env
        assert "CI_SHARD_STORE_DURATIONS" not in env
    # The nested-session test below proves the autouse scrub with the variable set.

OUTER_CONFTEST = """
pytest_plugins = ["pytester"]
"""

OUTER_TEST = """
import json
import os
from pathlib import Path


def test_nested_sessions(pytester, monkeypatch):
    manifest = Path(os.environ["OUTER_MANIFEST"])
    durations = Path(os.environ["OUTER_DURATIONS"])
    before = manifest.read_bytes()
    seeded = durations.read_bytes()
    assert "CI_SHARD_STORE_DURATIONS_MODE" not in os.environ
    pytester.makepyfile(test_inner=\"\"\"
def test_1(): pass
def test_2(): pass
def test_3(): pass
\"\"\")
    # No opt-in: the scrubbed environment keeps the nested session inert.
    result = pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
    result.assert_outcomes(passed=3)
    assert manifest.read_bytes() == before
    assert durations.read_bytes() == seeded
    # Opt-in with private outputs only.
    private = Path(os.environ["PRIVATE_DIR"])
    monkeypatch.setenv("CI_SHARD_INDEX", "1")
    monkeypatch.setenv("CI_SHARD_COUNT", "2")
    monkeypatch.setenv("CI_SHARD_MANIFEST", str(private / "manifest.json"))
    monkeypatch.setenv("CI_SHARD_STORE_DURATIONS", str(private / "durations.json"))
    pytester.runpytest_inprocess("-p", "ci_shard_plugin", "-q")
    assert (private / "manifest.json").exists()
    assert (private / "durations.json").exists()
    assert manifest.read_bytes() == before
    assert durations.read_bytes() == seeded


def test_filler_a(): pass
def test_filler_b(): pass
"""


@pytest.mark.parametrize("workers", [None, "2"])
def test_nested_sessions_cannot_clobber_outer_sharded_outputs(pytester, shard_env, tmp_path, workers):
    if workers:
        pytest.importorskip("xdist")
    pytester.makeconftest(OUTER_CONFTEST)
    pytester.makepyfile(test_outer=OUTER_TEST)
    outer_manifest = tmp_path / "outer" / "manifest.json"
    outer_durations = tmp_path / "outer" / "durations.json"
    private = tmp_path / "private"
    private.mkdir()
    outer_durations.parent.mkdir(exist_ok=True)
    outer_durations.write_text('{"seed::only": 1.0}\n')
    # Count 1 keeps every outer test (including the nested-launching one) in this shard.
    shard_env(
        CI_SHARD_INDEX=1, CI_SHARD_COUNT=1, CI_SHARD_MANIFEST=outer_manifest,
        CI_SHARD_STORE_DURATIONS=outer_durations, CI_SHARD_STORE_DURATIONS_MODE="replace", OUTER_MANIFEST=outer_manifest, OUTER_DURATIONS=outer_durations,
        PRIVATE_DIR=private,
    )
    args = ["-p", "ci_shard_plugin", "-q"] + (["-n", workers] if workers else [])
    result = pytester.runpytest_subprocess(*args)
    assert result.ret == 0, result.stdout.str()
    manifest = json.loads(outer_manifest.read_text())
    assert manifest["selected_ids"] == sorted(manifest["selected_ids"])
    assert all(i.startswith("test_outer.py::") for i in manifest["selected_ids"])
    assert manifest["full_collection_size"] == 3
    durations = json.loads(outer_durations.read_text())
    assert set(durations) == set(manifest["selected_ids"])
    assert not any("test_inner" in key for key in durations)
    assert json.loads((private / "manifest.json").read_text())["full_collection_size"] == 3


def test_ownership_comes_from_session_state_not_inherited_env(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw7")

    class Config:
        class pluginmanager:
            @staticmethod
            def get_plugin(_name):
                return None

    assert _ci_shard._owns_manifest(Config) is True
    assert _ci_shard._owns_durations(Config) is True

    class Worker(Config):
        workerinput = {"workerid": "gw1"}

    assert _ci_shard._owns_manifest(Worker) is False
    assert _ci_shard._owns_durations(Worker) is False
    Worker.workerinput = {"workerid": "gw0"}
    assert _ci_shard._owns_manifest(Worker) is True


# --------------------------------------------------------------------------
# Manifest verifier
# --------------------------------------------------------------------------

FULL = ["a", "b", "c"]


def _manifest(index, selected, *, full=FULL, count=3, head=HEAD, run_id=RUN_ID, attempt=1, **over):
    data = {
        "schema_version": 1,
        "shard_index": index,
        "shard_count": count,
        "full_collection_size": len(full),
        "full_collection_digest": ci_shard_verify.collection_digest(full),
        "selected_ids": selected,
        "head_sha": head,
        "run_id": run_id,
        "run_attempt": attempt,
    }
    data.update(over)
    return data


def _write(directory: Path, manifests):
    directory.mkdir(parents=True, exist_ok=True)
    for n, data in enumerate(manifests):
        sub = directory / f"shard-manifest-{n}"
        sub.mkdir(exist_ok=True)
        (sub / "manifest.json").write_text(json.dumps(data))
    return directory


def _good():
    return [_manifest(1, ["a"]), _manifest(2, ["b"]), _manifest(3, ["c"])]


def _verify(tmp_path, manifests, result="success", count=3):
    directory = _write(tmp_path / "m", manifests)
    return ci_shard_verify.verify(directory, count, HEAD, RUN_ID, result)


def test_verifier_accepts_valid_manifests(tmp_path):
    assert _verify(tmp_path, _good()) == []


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", "", "timed_out"])
def test_verifier_rejects_non_success_results(tmp_path, result):
    assert _verify(tmp_path, _good(), result=result)


def test_verifier_rejects_missing_index_with_rerun_diagnostic(tmp_path):
    errors = _verify(tmp_path, _good()[:2])
    assert any("re-run all jobs" in e for e in errors)


def test_verifier_rejects_digest_mismatch(tmp_path):
    bad = _good()
    bad[1]["full_collection_digest"] = "0" * 64
    assert _verify(tmp_path, bad)


def test_verifier_rejects_overlap_and_short_union(tmp_path):
    assert _verify(tmp_path, [_manifest(1, ["a"]), _manifest(2, ["a"]), _manifest(3, ["c"])])
    assert _verify(tmp_path / "s", [_manifest(1, ["a"]), _manifest(2, ["b"]), _manifest(3, [])])


def test_verifier_rejects_same_size_substituted_union(tmp_path):
    assert _verify(tmp_path, [_manifest(1, ["a"]), _manifest(2, ["b"]), _manifest(3, ["z"])])


def test_verifier_rejects_duplicate_ids_within_manifest(tmp_path):
    assert _verify(tmp_path, [_manifest(1, ["a", "a"]), _manifest(2, ["b"]), _manifest(3, ["c"])])


def test_verifier_rejects_inconsistent_count_and_size(tmp_path):
    bad = _good()
    bad[0]["shard_count"] = 2
    assert _verify(tmp_path, bad)
    other = _good()
    other[2] = _manifest(3, ["c"], full=["a", "b", "c", "d"])
    assert _verify(tmp_path / "o", other)


def test_verifier_rejects_foreign_head_or_run(tmp_path):
    assert _verify(tmp_path, [_manifest(1, ["a"], head="c" * 40)] + _good()[1:])
    assert _verify(tmp_path / "r", [_manifest(1, ["a"], run_id="1")] + _good()[1:])


def test_verifier_repaired_single_shard_uses_highest_attempt(tmp_path):
    manifests = [
        _manifest(1, ["a"], attempt=1),
        _manifest(2, ["b"], attempt=2),
        _manifest(3, ["c"], attempt=1),
    ]
    assert _verify(tmp_path, manifests) == []


def test_verifier_ignores_stale_lower_attempt_with_different_content(tmp_path):
    manifests = _good() + [_manifest(2, ["zzz"], attempt=0)]
    assert _verify(tmp_path, manifests) == []
    stale_wins = _good()[:1] + [_manifest(2, ["zzz"], attempt=2)] + _good()[1:]
    assert _verify(tmp_path / "w", stale_wins)


def test_verifier_cli_exit_codes(tmp_path):
    directory = _write(tmp_path / "m", _good())
    args = [sys.executable, str(VERIFIER), str(directory), "--count", "3", "--head", HEAD, "--run-id", RUN_ID]
    assert subprocess.run(args + ["--result", "success"]).returncode == 0
    assert subprocess.run(args + ["--result", "failure"], capture_output=True).returncode == 1
    assert subprocess.run(args, capture_output=True).returncode == 1


def test_verifier_module_is_stdlib_only():
    tree = __import__("ast").parse(VERIFIER.read_text())
    imported = set()
    for node in __import__("ast").walk(tree):
        if isinstance(node, __import__("ast").Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, __import__("ast").ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}


# --------------------------------------------------------------------------
# Connected workflow harness
# --------------------------------------------------------------------------

WORKFLOW_SHA = "a" * 40


def _jobs(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]


def _job_block(workflow, job_id):
    path = ORDINARY_WORKFLOW if workflow == "ordinary" else MANAGED_WORKFLOW
    return _jobs(path)[job_id]


def _env_block(job):
    return {k: str(v) for k, v in (job.get("env") or {}).items()}


def _steps(job):
    return [
        {
            "run": step.get("run"),
            "env": {k: str(v) for k, v in (step.get("env") or {}).items()},
            "name": step.get("name"),
            "if": step.get("if"),
        }
        for step in job["steps"]
    ]


def _if_expression(job):
    return " ".join(str(job["if"]).split())


def _eval_condition(expr, results):
    """Bounded evaluator: always(), ==/!= on needs.*.result, and &&."""
    for term in (t.strip() for t in expr.split("&&")):
        if term == "always()":
            continue
        match = re.fullmatch(r"needs\.([A-Za-z0-9_-]+)\.result (==|!=) '([a-z_]+)'", term)
        assert match, f"unsupported term {term!r}"
        actual = results[match.group(1)]
        if (actual == match.group(3)) != (match.group(2) == "=="):
            return False
    return True


def _expand(template, context):
    return re.sub(r"\$\{\{\s*([^}]+?)\s*\}\}", lambda m: str(context[m.group(1)]), template)


def _run_aggregate(
    tmp_path, job, results, manifests, *, verifier="real", workflow_sha=WORKFLOW_SHA, shards=3
):
    """Evaluate an aggregate the way Actions would, with per-step environments."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "verifier.log"
    spy = tmp_path / "spy.sh"
    body = "exit 0" if verifier == "stub" else f'exec "{sys.executable}" "{VERIFIER}" "$@"'
    spy.write_text(f'#!/bin/bash\necho called >> "{log}"\n{body}\n')
    spy.chmod(spy.stat().st_mode | stat.S_IXUSR)
    directory = _write(tmp_path / "manifests", manifests)
    context = {f"needs.{k}.result": v for k, v in results.items()}
    context.update({
        "github.sha": HEAD,
        "github.run_id": RUN_ID,
        "needs.validate-managed.outputs.target_sha": HEAD,
        "inputs.shards": shards,
        "inputs.python_version": "3.12",
        "job.workflow_repository": "OWNER/REPO" if workflow_sha is not None else "",
        "job.workflow_sha": workflow_sha or "",
    })
    outcome = {"ran": False, "gate": None, "verifier_calls": 0, "result": "skipped"}
    if not _eval_condition(_if_expression(job), results):
        return outcome
    outcome["ran"] = True
    job_env = {k: _expand(v, context) for k, v in _env_block(job).items()}
    base_env = {"PATH": os.environ["PATH"]}
    for step in _steps(job):
        run = step["run"]
        if run is None:
            continue
        if step["if"] == "inputs.shards > 1" and shards <= 1:
            continue
        env = {**base_env, **job_env, **{k: _expand(v, context) for k, v in step["env"].items()}}
        command = _expand(run, context)
        if "ci_shard_verify.py" in command:
            command = command.replace(
                "python .managed-ci/ci/managed/ci_shard_verify.py shard-manifests", f'"{spy}" "{directory}"'
            )
        done = subprocess.run(["bash", "-eo", "pipefail", "-c", command], env=env, capture_output=True, text=True)
        if 'SHARD_RESULT" = success' in run:
            outcome["gate"] = done.returncode
        if done.returncode != 0:
            outcome["result"] = "failure"
            break
    else:
        outcome["result"] = "success"
    outcome["verifier_calls"] = len(log.read_text().splitlines()) if log.exists() else 0
    return outcome


NON_SUCCESS = ["failure", "cancelled", "skipped"]


def _publisher_state(validation_result, aggregate_result):
    request = publisher.build_status_payload(
        target_sha=HEAD, validation_result=validation_result, test_result=aggregate_result,
        nonce="n" * 32, run_id=RUN_ID, run_attempt="1", server_url="https://github.com",
        repository="OWNER/REPO",
    )
    return None if request is None else request["payload"]["state"]


def test_workflow_conditions_are_the_expected_literals():
    assert _if_expression(_job_block("ordinary", "test")) == (
        "always() && needs.test-shard.result != 'skipped'"
    )
    assert _if_expression(_job_block("managed", "exact-head")) == (
        "always() && needs.validate-managed.result == 'success'"
    )


def test_ordinary_aggregate_all_success_runs_real_verifier(tmp_path):
    block = _job_block("ordinary", "test")
    out = _run_aggregate(tmp_path, block, {"test-shard": "success"}, _good())
    assert out == {"ran": True, "gate": 0, "verifier_calls": 1, "result": "success"}


@pytest.mark.parametrize("result", ["failure", "cancelled"])
def test_ordinary_aggregate_fails_at_gate_without_verifier(tmp_path, result):
    block = _job_block("ordinary", "test")
    for verifier in ("real", "stub"):
        out = _run_aggregate(tmp_path / verifier, block, {"test-shard": result}, _good(), verifier=verifier)
        assert out["result"] == "failure" and out["gate"] != 0 and out["verifier_calls"] == 0


def test_ordinary_aggregate_is_skipped_when_routing_suppresses_shards(tmp_path):
    block = _job_block("ordinary", "test")
    out = _run_aggregate(tmp_path, block, {"test-shard": "skipped"}, _good(), verifier="stub")
    assert out["result"] == "skipped" and out["gate"] is None and out["verifier_calls"] == 0


@pytest.mark.parametrize(
    "defect",
    ["missing", "substituted", "overlap"],
)
def test_ordinary_aggregate_defective_manifests_fail_via_verifier(tmp_path, defect):
    block = _job_block("ordinary", "test")
    manifests = {
        "missing": _good()[:2],
        "substituted": [_manifest(1, ["a"]), _manifest(2, ["b"]), _manifest(3, ["z"])],
        "overlap": [_manifest(1, ["a", "b"]), _manifest(2, ["b"]), _manifest(3, ["c"])],
    }[defect]
    out = _run_aggregate(tmp_path, block, {"test-shard": "success"}, manifests)
    assert out["gate"] == 0 and out["verifier_calls"] == 1 and out["result"] == "failure"


def test_managed_aggregate_success_publishes_success(tmp_path):
    block = _job_block("managed", "exact-head")
    out = _run_aggregate(
        tmp_path, block, {"validate-managed": "success", "exact-head-shard": "success"}, _good()
    )
    assert out["result"] == "success"
    assert _publisher_state("success", out["result"]) == "success"


@pytest.mark.parametrize("result", NON_SUCCESS)
@pytest.mark.parametrize("verifier", ["real", "stub"])
def test_managed_non_success_shard_never_publishes_success(tmp_path, result, verifier):
    block = _job_block("managed", "exact-head")
    out = _run_aggregate(
        tmp_path, block, {"validate-managed": "success", "exact-head-shard": result}, _good(),
        verifier=verifier,
    )
    assert out["ran"] and out["gate"] != 0 and out["verifier_calls"] == 0
    assert out["result"] == "failure"
    assert _publisher_state("success", out["result"]) == "failure"


@pytest.mark.parametrize("defect", ["missing", "substituted"])
def test_managed_defective_manifests_publish_failure(tmp_path, defect):
    block = _job_block("managed", "exact-head")
    manifests = (
        _good()[:2] if defect == "missing"
        else [_manifest(1, ["a"]), _manifest(2, ["b"]), _manifest(3, ["z"])]
    )
    out = _run_aggregate(
        tmp_path, block, {"validate-managed": "success", "exact-head-shard": "success"}, manifests
    )
    assert out["gate"] == 0 and out["verifier_calls"] == 1 and out["result"] == "failure"
    assert _publisher_state("success", out["result"]) == "failure"


def test_managed_validation_failure_skips_aggregate_and_writes_no_status(tmp_path):
    block = _job_block("managed", "exact-head")
    out = _run_aggregate(
        tmp_path, block, {"validate-managed": "failure", "exact-head-shard": "skipped"}, _good()
    )
    assert out["result"] == "skipped" and not out["ran"]
    assert _publisher_state("failure", out["result"]) is None


@pytest.mark.parametrize("workflow,job,needs", [("ordinary", "test", "test-shard"), ("managed", "exact-head", "exact-head-shard")])
@pytest.mark.parametrize("workflow_sha", [None, "", "not-a-sha", "A" * 40])
def test_aggregate_fails_closed_without_a_resolved_callee_revision(tmp_path, workflow, job, needs, workflow_sha):
    """An empty or malformed job.workflow_sha never runs, skips or fakes the verifier."""
    block = _job_block(workflow, job)
    results = {needs: "success", "validate-managed": "success"}
    out = _run_aggregate(tmp_path, block, results, _good(), workflow_sha=workflow_sha)
    assert out["ran"] and out["gate"] == 0
    assert out["result"] == "failure" and out["verifier_calls"] == 0


@pytest.mark.parametrize("workflow,job,needs", [("ordinary", "test", "test-shard"), ("managed", "exact-head", "exact-head-shard")])
def test_single_shard_aggregate_keeps_the_literal_gate_and_needs_no_manifests(tmp_path, workflow, job, needs):
    block = _job_block(workflow, job)
    ok = _run_aggregate(tmp_path / "ok", block, {needs: "success", "validate-managed": "success"}, [], shards=1)
    assert ok["result"] == "success" and ok["verifier_calls"] == 0
    bad = _run_aggregate(tmp_path / "bad", block, {needs: "failure", "validate-managed": "success"}, [], shards=1)
    assert bad["result"] == "failure" and bad["gate"] != 0


@pytest.mark.parametrize("workflow,job,needs", [("ordinary", "test", "test-shard"), ("managed", "exact-head", "exact-head-shard")])
def test_step_scoped_shard_result_is_detected_by_the_harness(tmp_path, workflow, job, needs):
    """A SHARD_RESULT defined only in the gate step leaves the verifier empty."""
    import copy

    block = _job_block(workflow, job)
    assert block["env"]["SHARD_RESULT"] == "${{ needs." + needs + ".result }}"
    broken = copy.deepcopy(block)
    del broken["env"]["SHARD_RESULT"]
    broken["steps"][0]["env"] = {"SHARD_RESULT": "${{ needs." + needs + ".result }}"}
    results = {needs: "success", "validate-managed": "success"}
    assert _run_aggregate(tmp_path / "ok", block, results, _good())["result"] == "success"
    out = _run_aggregate(tmp_path / "bad", broken, results, _good())
    assert out["gate"] == 0 and out["verifier_calls"] == 1 and out["result"] == "failure"


def test_unsharded_workflow_command_is_still_detected_as_parallel():
    from coding_review_agent_loop.test_workers import detect_parallel_support

    assert detect_parallel_support(ROOT) is True


def test_committed_durations_cover_most_of_the_real_collection(request):
    durations = json.loads((TESTS_DIR / ".test_durations").read_text())
    full = _recorded_full_collection(request) or []
    if len(full) < 1000:
        pytest.skip("only meaningful when the whole suite is collected")
    known = sum(1 for i in full if i in durations)
    assert known / len(full) > 0.9, (
        "add the missing tests/.test_durations entries with CI_SHARD_STORE_DURATIONS_MODE=add-missing "
        "(see README); a full refresh belongs in a dedicated maintenance PR"
    )


def test_callee_files_depend_only_on_the_standard_library_and_pytest():
    """The injected plugin must not depend on this repository's tests helpers."""
    import ast

    for name, allowed in (
        ("ci_shard_plugin.py", {"pytest"}),
        ("ci_shard_verify.py", set()),
        ("run_shard.py", set()),
    ):
        tree = ast.parse((CALLEE_DIR / name).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        assert imported <= set(sys.stdlib_module_names) | {"__future__"} | allowed, (name, imported)


@pytest.mark.parametrize("workflow,job", [(ORDINARY_WORKFLOW, "test-shard"), (MANAGED_WORKFLOW, "exact-head-shard")], ids=["ordinary", "managed"])
@pytest.mark.parametrize("shards", [None, 3], ids=["default-single", "multi"])
@pytest.mark.parametrize("budget", [None, 60], ids=["default-budget", "consumer-budget"])
def test_reusable_shard_budget_routing(workflow, job, shards, budget):
    document = yaml.safe_load(workflow.read_text())
    # PyYAML follows YAML 1.1, which treats the unquoted on key as True.
    inputs = document.get("on", document.get(True))["workflow_call"]["inputs"]
    declaration = inputs["shard_timeout_minutes"]
    assert declaration["type"] == "number"
    assert declaration["required"] is False
    assert type(declaration["default"]) is int and declaration["default"] == 30
    assert inputs["shards"]["default"] == 1
    runner = _jobs(workflow)[job]
    assert runner["timeout-minutes"] == "${{ inputs.shard_timeout_minutes }}"
    assert runner["strategy"]["fail-fast"] is False
    selected_shards = int(inputs["shards"]["default"]) if shards is None else shards
    selected_budget = int(declaration["default"]) if budget is None else budget
    # Bounded expression resolution, shared by every matrix leg, including 1.
    for _ in range(selected_shards):
        assert int(_expand(runner["timeout-minutes"], {"inputs.shard_timeout_minutes": selected_budget})) == (30 if budget is None else 60)


def test_repository_callers_use_reusable_budget_defaults():
    jobs = _jobs(ROOT / ".github" / "workflows" / "ci.yml")
    for caller, workflow in [("ci", ORDINARY_WORKFLOW), ("managed", MANAGED_WORKFLOW)]:
        job = jobs[caller]
        assert job["uses"] == f"./.github/workflows/{workflow.name}"
        assert "shard_timeout_minutes" not in job["with"]
        assert "timeout-minutes" not in job
        assert job["with"]["shards"] == 3


def test_actual_publisher_scheduling_and_validated_bindings():
    job = _job_block("managed", "publish-exact-head")
    assert job["needs"] == ["validate-managed", "exact-head"]
    condition = _if_expression(job)
    validation_condition = _if_expression(_job_block("managed", "validate-managed"))
    assert condition == "always() && " + validation_condition
    assert job["permissions"] == {"statuses": "write"}
    env = job["steps"][0]["env"]
    assert env["TEST_RESULT"] == "${{ needs.exact-head.result }}"
    assert env["VALIDATION_RESULT"] == "${{ needs.validate-managed.result }}"
    assert env["TARGET_SHA"] == "${{ needs.validate-managed.outputs.target_sha }}"
    assert env["NONCE"] == "${{ needs.validate-managed.outputs.managed_nonce }}"
