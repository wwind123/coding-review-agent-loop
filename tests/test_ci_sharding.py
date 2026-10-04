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

import _ci_shard
import ci_shard_verify
from fixtures.managed_ci import publisher

pytest_plugins = ["pytester"]

TESTS_DIR = Path(__file__).parent
ROOT = TESTS_DIR.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
VERIFIER = TESTS_DIR / "ci_shard_verify.py"
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


def test_real_collection_is_covered_exactly_once(request):
    full = request.config.stash[_ci_shard._FULL_KEY]
    if not full:
        pytest.skip("collection not recorded in this process")
    assert full
    groups = _ci_shard.partition(full, _ci_shard._load_durations(), 3)
    _ci_shard.check_exactly_once(groups, full)


# --------------------------------------------------------------------------
# Plugin (pytester)
# --------------------------------------------------------------------------

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
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(TESTS_DIR), os.environ.get("PYTHONPATH", "")]))

    def apply(**values):
        for key, value in values.items():
            monkeypatch.setenv(key, str(value))

    return apply


def test_plugin_is_inert_without_shard_variables(pytester):
    pytester.makepyfile(test_suite=SUITE)
    result = pytester.runpytest_inprocess("-p", "_ci_shard", "-q")
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
    result = pytester.runpytest_inprocess("-p", "_ci_shard", "-q")
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*CI_SHARD*"])


def test_shards_select_disjoint_cover_and_write_manifest(pytester, shard_env, tmp_path):
    pytester.makepyfile(test_suite=SUITE)
    selected = []
    for index in (1, 2, 3):
        manifest = tmp_path / f"m{index}" / "manifest.json"
        shard_env(CI_SHARD_INDEX=index, CI_SHARD_COUNT=3, CI_SHARD_MANIFEST=manifest,
                  GITHUB_RUN_ID=RUN_ID, GITHUB_RUN_ATTEMPT=2)
        pytester.runpytest_inprocess("-p", "_ci_shard", "-q")
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
    result = pytester.runpytest_subprocess("-p", "_ci_shard", "-n", "2", "-q")
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
    pytester.runpytest_inprocess("-p", "_ci_shard", "-q")
    data = json.loads(plain.read_text())
    assert set(data) == {f"test_suite.py::test_{c}" for c in "abcd"}
    assert data["test_suite.py::test_a"] >= 0.15  # setup + call + teardown

    pytest.importorskip("xdist")
    dist = tmp_path / "dist.json"
    shard_env(CI_SHARD_STORE_DURATIONS=dist)
    result = pytester.runpytest_subprocess("-p", "_ci_shard", "-n", "2", "-q")
    assert result.ret == 0
    data = json.loads(dist.read_text())
    assert set(data) == {f"test_suite.py::test_{c}" for c in "abcd"}
    assert data["test_suite.py::test_a"] >= 0.15


OUTER_CONFTEST = """
import pytest
from _ci_shard import SHARD_ENV_VARS

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _scrub(monkeypatch):
    for name in SHARD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
"""

OUTER_TEST = """
import json
import os
from pathlib import Path


def test_nested_sessions(pytester, monkeypatch):
    manifest = Path(os.environ["OUTER_MANIFEST"])
    before = manifest.read_bytes()
    pytester.makepyfile(test_inner=\"\"\"
def test_1(): pass
def test_2(): pass
def test_3(): pass
\"\"\")
    # No opt-in: the scrubbed environment keeps the nested session inert.
    result = pytester.runpytest_inprocess("-p", "_ci_shard", "-q")
    result.assert_outcomes(passed=3)
    # Opt-in with private outputs only.
    private = Path(os.environ["PRIVATE_DIR"])
    monkeypatch.setenv("CI_SHARD_INDEX", "1")
    monkeypatch.setenv("CI_SHARD_COUNT", "2")
    monkeypatch.setenv("CI_SHARD_MANIFEST", str(private / "manifest.json"))
    monkeypatch.setenv("CI_SHARD_STORE_DURATIONS", str(private / "durations.json"))
    pytester.runpytest_inprocess("-p", "_ci_shard", "-q")
    assert (private / "manifest.json").exists()
    assert manifest.read_bytes() == before


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
    # Count 1 keeps every outer test (including the nested-launching one) in this shard.
    shard_env(
        CI_SHARD_INDEX=1, CI_SHARD_COUNT=1, CI_SHARD_MANIFEST=outer_manifest,
        CI_SHARD_STORE_DURATIONS=outer_durations, OUTER_MANIFEST=outer_manifest,
        PRIVATE_DIR=private,
    )
    args = ["-p", "_ci_shard", "-q"] + (["-n", workers] if workers else [])
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


def _workflow_text():
    return WORKFLOW.read_text(encoding="utf-8")


def _job_block(text, job_id):
    match = re.search(
        rf"^  {re.escape(job_id)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)", text, re.S | re.M
    )
    assert match, job_id
    return match.group(1)


def _env_block(block):
    match = re.search(r"^    env:\n((?:      .+\n)+)", block, re.M)
    if not match:
        return {}
    return dict(re.findall(r"^      ([A-Z_]+): (.+)$", match.group(1), re.M))


def _steps(block):
    body = block.split("    steps:\n", 1)[1]
    steps = []
    for chunk in re.split(r"^      - ", body, flags=re.M)[1:]:
        run = re.search(r"^\s*(?:- )?run: (.+)$", "      - " + chunk, re.M)
        env_match = re.search(r"^        env:\n((?:          .+\n)+)", chunk, re.M)
        env = dict(re.findall(r"^          ([A-Z_]+): (.+)$", env_match.group(1), re.M)) if env_match else {}
        steps.append({"run": run.group(1) if run else None, "env": env,
                      "text": chunk, "name": (re.match(r"name: (.+)", chunk) or [None, None])[1]})
    return steps


def _if_expression(block):
    return re.search(r"^    if: (.+)$", block, re.M).group(1)


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


def _run_aggregate(tmp_path, block, results, manifests, *, verifier="real", validate_ok=True):
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
    })
    outcome = {"ran": False, "gate": None, "verifier_calls": 0, "result": "skipped"}
    if not _eval_condition(_if_expression(block), results):
        return outcome
    outcome["ran"] = True
    job_env = {k: _expand(v, context) for k, v in _env_block(block).items()}
    base_env = {"PATH": os.environ["PATH"]}
    for step in _steps(block):
        run = step["run"]
        if run is None:
            continue
        env = {**base_env, **job_env, **{k: _expand(v, context) for k, v in step["env"].items()}}
        command = _expand(run, context)
        if "ci_shard_verify.py" in command:
            command = command.replace("python tests/ci_shard_verify.py shard-manifests", f'"{spy}" "{directory}"')
        done = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True)
        if "SHARD_RESULT\" = success" in run:
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
    text = _workflow_text()
    assert _if_expression(_job_block(text, "test")) == "always() && needs.test-shard.result != 'skipped'"
    assert _if_expression(_job_block(text, "exact-head")) == (
        "always() && needs.validate-managed.result == 'success'"
    )


def test_ordinary_aggregate_all_success_runs_real_verifier(tmp_path):
    block = _job_block(_workflow_text(), "test")
    out = _run_aggregate(tmp_path, block, {"test-shard": "success"}, _good())
    assert out == {"ran": True, "gate": 0, "verifier_calls": 1, "result": "success"}


@pytest.mark.parametrize("result", ["failure", "cancelled"])
def test_ordinary_aggregate_fails_at_gate_without_verifier(tmp_path, result):
    block = _job_block(_workflow_text(), "test")
    for verifier in ("real", "stub"):
        out = _run_aggregate(tmp_path / verifier, block, {"test-shard": result}, _good(), verifier=verifier)
        assert out["result"] == "failure" and out["gate"] != 0 and out["verifier_calls"] == 0


def test_ordinary_aggregate_is_skipped_when_routing_suppresses_shards(tmp_path):
    block = _job_block(_workflow_text(), "test")
    out = _run_aggregate(tmp_path, block, {"test-shard": "skipped"}, _good(), verifier="stub")
    assert out["result"] == "skipped" and out["gate"] is None and out["verifier_calls"] == 0


@pytest.mark.parametrize(
    "defect",
    ["missing", "substituted", "overlap"],
)
def test_ordinary_aggregate_defective_manifests_fail_via_verifier(tmp_path, defect):
    block = _job_block(_workflow_text(), "test")
    manifests = {
        "missing": _good()[:2],
        "substituted": [_manifest(1, ["a"]), _manifest(2, ["b"]), _manifest(3, ["z"])],
        "overlap": [_manifest(1, ["a", "b"]), _manifest(2, ["b"]), _manifest(3, ["c"])],
    }[defect]
    out = _run_aggregate(tmp_path, block, {"test-shard": "success"}, manifests)
    assert out["gate"] == 0 and out["verifier_calls"] == 1 and out["result"] == "failure"


def test_managed_aggregate_success_publishes_success(tmp_path):
    block = _job_block(_workflow_text(), "exact-head")
    out = _run_aggregate(
        tmp_path, block, {"validate-managed": "success", "exact-head-shard": "success"}, _good()
    )
    assert out["result"] == "success"
    assert _publisher_state("success", out["result"]) == "success"


@pytest.mark.parametrize("result", NON_SUCCESS)
@pytest.mark.parametrize("verifier", ["real", "stub"])
def test_managed_non_success_shard_never_publishes_success(tmp_path, result, verifier):
    block = _job_block(_workflow_text(), "exact-head")
    out = _run_aggregate(
        tmp_path, block, {"validate-managed": "success", "exact-head-shard": result}, _good(),
        verifier=verifier,
    )
    assert out["ran"] and out["gate"] != 0 and out["verifier_calls"] == 0
    assert out["result"] == "failure"
    assert _publisher_state("success", out["result"]) == "failure"


@pytest.mark.parametrize("defect", ["missing", "substituted"])
def test_managed_defective_manifests_publish_failure(tmp_path, defect):
    block = _job_block(_workflow_text(), "exact-head")
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
    block = _job_block(_workflow_text(), "exact-head")
    out = _run_aggregate(
        tmp_path, block, {"validate-managed": "failure", "exact-head-shard": "skipped"}, _good()
    )
    assert out["result"] == "skipped" and not out["ran"]
    assert _publisher_state("failure", out["result"]) is None


@pytest.mark.parametrize("job,needs", [("test", "test-shard"), ("exact-head", "exact-head-shard")])
def test_step_scoped_shard_result_is_detected_by_the_harness(tmp_path, job, needs):
    """A SHARD_RESULT defined only in the gate step leaves the verifier empty."""
    block = _job_block(_workflow_text(), job)
    line = f"      SHARD_RESULT: ${{{{ needs.{needs}.result }}}}\n"
    assert line in block
    broken = block.replace("    env:\n" + line, "")
    broken = broken.replace(
        "        run: test \"$SHARD_RESULT\" = success",
        "        env:\n          SHARD_RESULT: ${{ needs." + needs + ".result }}\n"
        "        run: test \"$SHARD_RESULT\" = success",
    )
    results = {needs: "success", "validate-managed": "success"}
    assert _run_aggregate(tmp_path / "ok", block, results, _good())["result"] == "success"
    out = _run_aggregate(tmp_path / "bad", broken, results, _good())
    assert out["gate"] == 0 and out["verifier_calls"] == 1 and out["result"] == "failure"


def test_unsharded_workflow_command_is_still_detected_as_parallel():
    from coding_review_agent_loop.test_workers import detect_parallel_support

    assert detect_parallel_support(ROOT) is True


def test_committed_durations_cover_most_of_the_real_collection(request):
    durations = json.loads((TESTS_DIR / ".test_durations").read_text())
    full = request.config.stash[_ci_shard._FULL_KEY]
    if len(full) < 1000:
        pytest.skip("only meaningful when the whole suite is collected")
    known = sum(1 for i in full if i in durations)
    assert known / len(full) > 0.9, "refresh tests/.test_durations (see README)"
