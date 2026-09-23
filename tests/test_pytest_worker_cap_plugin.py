"""Real pytest + pytest-xdist tests for the injected worker-cap plugin (issue #848).

Each case runs an inner pytest in a subprocess with the wrapper-private spec
and reads the plugin's JSON-lines report.  The module skips cleanly when
pytest-xdist is not installed in the running interpreter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("xdist")

from coding_review_agent_loop.test_workers import (  # noqa: E402
    ENV_WORKER_CAP_NESTED,
    ENV_WORKER_CAP_SPEC,
    PLUGIN_MODULE,
    analyze_worker_report,
    plugin_directory,
)

TESTS = textwrap.dedent(
    """
    import os

    def _mark(name):
        path = os.environ.get("RAN_MARKER")
        if path:
            with open(path, "a") as stream:
                stream.write(name + "\\n")

    def test_one():
        _mark("one")

    def test_two():
        _mark("two")

    def test_three():
        _mark("three")

    def test_four():
        _mark("four")
    """
)

_SCRUB = (
    "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_XDIST_AUTO_NUM_WORKERS",
    ENV_WORKER_CAP_SPEC, ENV_WORKER_CAP_NESTED, "PYTEST_XDIST_WORKER",
)


def _project(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "test_cases.py").write_text(TESTS, encoding="utf-8")
    for name, content in (files or {}).items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
    return project


def _run(
    project: Path,
    args: list[str],
    *,
    budget: int = 3,
    mode: str = "clamp",
    env: dict[str, str] | None = None,
    inject_token: bool = True,
    report: Path | None = None,
    command: list[str] | None = None,
    timeout: float = 120,
):
    report = report or (project.parent / "report.jsonl")
    marker = project.parent / "ran.txt"
    for path in (report, marker):
        if path.exists():
            path.unlink()
    child_env = {key: value for key, value in os.environ.items() if key not in _SCRUB}
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(plugin_directory()), str(Path(__file__).resolve().parents[1] / "src")]
    )
    child_env["PYTEST_PLUGINS"] = PLUGIN_MODULE
    child_env["PYTEST_XDIST_AUTO_NUM_WORKERS"] = "8"
    child_env["RAN_MARKER"] = str(marker)
    child_env[ENV_WORKER_CAP_SPEC] = json.dumps({"version": 1, "budget": budget, "mode": mode, "report": str(report)})
    child_env.update(env or {})
    for key, value in list(child_env.items()):
        if value is None:
            del child_env[key]
    argv = command or [
        sys.executable, "-m", "pytest", *(["-p", PLUGIN_MODULE] if inject_token else []),
        "-p", "no:cacheprovider", "-q", *args,
    ]
    proc = subprocess.run(
        argv, cwd=project, env=child_env, capture_output=True, text=True, timeout=timeout,
    )
    rows = []
    if report.exists():
        rows = [json.loads(line) for line in report.read_text().splitlines() if line.strip()]
    ran = marker.read_text().split() if marker.exists() else []
    return proc, rows, ran


def _only(rows, kind):
    return [row for row in rows if row.get("kind") == kind]


def _confirmed(rows):
    (row,) = _only(rows, "confirmed")
    return row


# ---------------------------------------------------------------------------
# Option sources (clamp and refuse)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "files, args, env",
    [
        ({}, ["-n", "12"], {}),
        ({}, ["--numprocesses=12"], {}),
        ({}, ["-n", "auto"], {}),
        ({}, [], {"PYTEST_ADDOPTS": "-n 12"}),
        ({}, ["-o", "addopts=-n 9"], {}),
        ({"pytest.ini": "[pytest]\naddopts = -n 12\n"}, [], {}),
        ({"pyproject.toml": "[tool.pytest.ini_options]\naddopts = '-n 12'\n"}, [], {}),
        ({"alt.ini": "[pytest]\naddopts = -n 12\n"}, ["-calt.ini"], {}),
        ({"alt.ini": "[pytest]\naddopts = -n 12\n"}, ["--config-file=alt.ini"], {}),
    ],
)
def test_clamp_lowers_every_option_source(tmp_path, files, args, env):
    project = _project(tmp_path, files)
    proc, rows, ran = _run(project, args, budget=3, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    confirmed = _confirmed(rows)
    assert confirmed["effective"] == 3
    assert confirmed["action"] == "clamped"
    assert sorted(ran) == ["four", "one", "three", "two"]
    analysis = analyze_worker_report(project.parent / "report.jsonl", command_class="direct-pytest", mode="clamp")
    assert analysis.workers_cohort == "3"


def test_nested_ini_selected_by_rootdir_rules(tmp_path):
    project = _project(tmp_path, {"tests/nested/pytest.ini": "[pytest]\naddopts = -n 12\n"})
    (project / "tests" / "nested" / "test_nested.py").write_text(TESTS, encoding="utf-8")
    proc, rows, _ran = _run(project, ["tests/nested"], budget=3)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _confirmed(rows)["effective"] == 3


@pytest.mark.parametrize(
    "files, args, env, expected",
    [
        ({"pytest.ini": "[pytest]\naddopts = -n 12\n"}, [], {"PYTEST_ADDOPTS": "-n 0"}, 0),
        ({"pytest.ini": "[pytest]\naddopts = -n 12\n"}, ["-o", "addopts=-n 2"], {}, 2),
        ({}, ["-n", "2"], {}, 2),
        ({}, [], {}, 0),
    ],
)
def test_serial_and_in_budget_values_are_never_raised(tmp_path, files, args, env, expected):
    project = _project(tmp_path, files)
    proc, rows, _ran = _run(project, args, budget=3, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    confirmed = _confirmed(rows)
    assert confirmed["effective"] == expected
    assert confirmed["action"] == "unchanged"


@pytest.mark.parametrize(
    "files, args, env, stage",
    [
        ({}, ["-n", "8"], {}, "cmdline"),
        ({"pytest.ini": "[pytest]\naddopts = -n 8\n"}, [], {}, "cmdline"),
        ({}, [], {"PYTEST_ADDOPTS": "-n 8"}, "cmdline"),
        ({}, ["-o", "addopts=-n 8"], {}, "cmdline"),
        ({}, ["-n", "auto"], {}, "auto"),
        ({}, ["--dist=load", "--tx", "8*popen"], {}, "setupnodes"),
        ({}, ["-n", "8", "--maxprocesses=4"], {}, "cmdline"),
        ({}, ["-n", "auto", "--maxprocesses=4"], {}, "auto"),
    ],
)
def test_refuse_writes_one_refused_line_before_any_test(tmp_path, files, args, env, stage):
    project = _project(tmp_path, files)
    proc, rows, ran = _run(project, args, budget=2, mode="refuse", env=env)
    assert proc.returncode == 4, proc.stdout + proc.stderr
    assert ran == []
    refused = _only(rows, "refused")
    assert len(refused) == 1 and refused[0]["stage"] == stage
    assert not _only(rows, "confirmed")
    assert "agent-loop worker budget refused" in proc.stdout + proc.stderr
    analysis = analyze_worker_report(project.parent / "report.jsonl", command_class="direct-pytest", mode="refuse")
    assert analysis.direct_refusal


@pytest.mark.parametrize(
    "files, args",
    [
        ({}, ["-n", "8", "--maxprocesses=2"]),
        ({}, ["-n", "auto", "--maxprocesses=2"]),
        ({"pytest.ini": "[pytest]\naddopts = -n 8\n"}, ["--maxprocesses=2"]),
    ],
)
def test_refuse_honours_maxprocesses(tmp_path, files, args):
    project = _project(tmp_path, files)
    proc, rows, ran = _run(project, args, budget=2, mode="refuse")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    confirmed = _confirmed(rows)
    assert confirmed["effective"] == 2 and confirmed["action"] == "unchanged"
    assert len(ran) == 4


# ---------------------------------------------------------------------------
# Auto-resolution hook, hook ordering and node creation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hook_value, expected", [(16, 3), (1, 1)])
def test_repository_auto_hook_is_capped_through_new_style_wrapper(tmp_path, hook_value, expected):
    project = _project(tmp_path, {"conftest.py": f"def pytest_xdist_auto_num_workers(config):\n    return {hook_value}\n"})
    proc, rows, _ran = _run(project, ["-n", "auto"], budget=3)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    confirmed = _confirmed(rows)
    assert confirmed["auto_raw"] == hook_value
    assert confirmed["effective"] == expected
    refuse_proc, refuse_rows, ran = _run(project, ["-n", "auto"], budget=3, mode="refuse")
    if hook_value > 3:
        assert refuse_proc.returncode == 4 and ran == []
        assert _only(refuse_rows, "refused")[0]["stage"] == "auto"
    else:
        assert refuse_proc.returncode == 0


@pytest.mark.parametrize("inject_token", [True, False])
@pytest.mark.parametrize("source", ["argv", "addopts"])
def test_budget_one_runs_serially_in_every_registration_order(tmp_path, inject_token, source):
    project = _project(tmp_path)
    args = ["-n", "4"] if source == "argv" else []
    env = {"PYTEST_ADDOPTS": "-n 4"} if source == "addopts" else {}
    proc, rows, ran = _run(project, args, budget=1, env=env, inject_token=inject_token)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    confirmed = _confirmed(rows)
    assert confirmed["effective"] == 0 and confirmed["action"] == "clamped"
    assert not _only(rows, "decision")
    assert len(ran) == 4


def test_session_start_and_inner_setupnodes_changes_are_trimmed(tmp_path):
    conftest = """
    import execnet, pytest

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionstart(session):
        pass

    def pytest_xdist_setupnodes(config, specs):
        specs.append(execnet.XSpec("popen"))
        specs.append(execnet.XSpec("popen"))
    """
    project = _project(tmp_path, {"conftest.py": conftest})
    proc, rows, ran = _run(project, ["-n", "2"], budget=2)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    decision = _only(rows, "decision")[0]
    confirmed = _confirmed(rows)
    assert decision["planned"] == 2
    assert confirmed["effective"] == 2 and confirmed["action"] == "clamped"
    assert rows.index(decision) < rows.index(confirmed)
    assert len(ran) == 4


def test_late_tryfirst_wrapper_is_reported_exceeded(tmp_path):
    conftest = """
    import execnet, pytest

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_xdist_setupnodes(config, specs):
        result = yield
        specs.append(execnet.XSpec("popen"))
        return result
    """
    project = _project(tmp_path, {"conftest.py": conftest})
    proc, rows, _ran = _run(project, ["-n", "4"], budget=2)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _only(rows, "decision")[0]["planned"] == 2
    confirmed = _confirmed(rows)
    assert confirmed["effective"] == 3 and confirmed["action"] == "exceeded"
    assert "worker budget WARNING" in proc.stdout + proc.stderr
    analysis = analyze_worker_report(project.parent / "report.jsonl", command_class="direct-pytest", mode="clamp")
    assert analysis.enforcement == "exceeded" and analysis.workers_cohort == "unknown"


def test_tx_without_dist_creates_no_gateway(tmp_path):
    project = _project(tmp_path)
    proc, rows, _ran = _run(project, ["--tx", "3*popen"], budget=2)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _confirmed(rows)["effective"] == 0


def test_tx_list_is_trimmed_in_order(tmp_path):
    project = _project(tmp_path)
    proc, rows, _ran = _run(project, ["--dist=load", "--tx", "5*popen"], budget=2)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _confirmed(rows)["effective"] == 2


# ---------------------------------------------------------------------------
# Bypass, reports, workers and sessions
# ---------------------------------------------------------------------------


def test_injected_token_overrides_config_and_addopts_disable(tmp_path):
    project = _project(tmp_path, {"pytest.ini": f"[pytest]\naddopts = -p no:{PLUGIN_MODULE} -n 12\n"})
    proc, rows, _ran = _run(project, [], budget=3, env={"PYTEST_ADDOPTS": f"-p no:{PLUGIN_MODULE}"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _confirmed(rows)["effective"] == 3


@pytest.mark.parametrize("body, expected", [("pass", 0), ("assert False", 1)])
def test_unwritable_report_never_changes_exit_status(tmp_path, body, expected):
    project = _project(tmp_path, {"test_extra.py": f"def test_extra():\n    {body}\n"})
    report = tmp_path / "missing-dir" / "report.jsonl"
    proc, rows, _ran = _run(project, ["-n", "2"], budget=2, report=report)
    assert proc.returncode == expected
    assert rows == []
    assert "agent-loop worker cap: report not written" in proc.stderr
    analysis = analyze_worker_report(report, command_class="direct-pytest", mode="clamp")
    assert analysis.enforcement == "unverified" and analysis.workers_cohort == "unknown"


def test_worker_processes_write_nothing_and_executed_marker_once(tmp_path):
    project = _project(tmp_path)
    proc, rows, _ran = _run(project, ["-n", "2"], budget=2)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    sessions = {row["session"] for row in rows if "session" in row}
    assert len(sessions) == 1
    assert {row["pid"] for row in rows} == {rows[0]["pid"]}
    assert len(_only(rows, "executed")) == 1
    assert not _only(rows, "nested")


def test_plugin_inert_without_xdist(tmp_path):
    project = _project(tmp_path)
    proc, rows, _ran = _run(project, ["-p", "no:xdist"], budget=2)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    confirmed = _confirmed(rows)
    assert confirmed["effective"] == 0 and confirmed["action"] == "unchanged"


def test_sequential_pytest_main_sessions_each_claim_and_report(tmp_path):
    project = _project(tmp_path)
    runner = project.parent / "runner.py"
    runner.write_text(
        textwrap.dedent(
            """
            import sys, pytest
            first = pytest.main(["-p", "_agent_loop_worker_cap", "-p", "no:cacheprovider", "-q", "-n", sys.argv[1]])
            second = pytest.main(["-p", "_agent_loop_worker_cap", "-p", "no:cacheprovider", "-q", "-n", sys.argv[2]])
            sys.exit(int(first) or int(second))
            """
        ),
        encoding="utf-8",
    )
    proc, rows, _ran = _run(project, [], budget=2, command=[sys.executable, str(runner), "1", "16"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    confirmed = _only(rows, "confirmed")
    assert len({row["session"] for row in confirmed}) == 2
    assert [row["effective"] for row in confirmed] == [1, 2]
    analysis = analyze_worker_report(project.parent / "report.jsonl", command_class="other", mode="clamp")
    assert analysis.workers_cohort == "unknown"
    assert "worker-budget-mixed" in analysis.caveats and "worker-budget-partial-observation" in analysis.caveats

    refused_first, rows, _ran = _run(
        project, [], budget=2, mode="refuse", command=[sys.executable, str(runner), "16", "1"]
    )
    assert refused_first.returncode == 4
    assert len(_only(rows, "refused")) == 1 and len(_only(rows, "confirmed")) == 1
    analysis = analyze_worker_report(project.parent / "report.jsonl", command_class="other", mode="refuse")
    assert analysis.enforcement == "refused-in-command" and not analysis.direct_refusal


def test_nested_sessions_are_not_enforced_and_write_nested_markers(tmp_path):
    nested_test = """
    import os, subprocess, sys
    import pytest

    def test_nested_in_process(tmp_path):
        (tmp_path / "test_inner.py").write_text("def test_inner():\\n    pass\\n")
        assert pytest.main(["-p", "no:cacheprovider", "-q", str(tmp_path / "test_inner.py")]) == 0

    def test_nested_child(tmp_path):
        (tmp_path / "test_inner.py").write_text("def test_inner():\\n    pass\\n")
        env = dict(os.environ)
        assert "AGENT_LOOP_WORKER_CAP_SPEC" not in env
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", str(tmp_path / "test_inner.py")],
            env=env,
        )
        assert proc.returncode == 0
    """
    project = _project(tmp_path, {"test_nested.py": nested_test})
    proc, rows, _ran = _run(project, ["test_nested.py"], budget=2)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    nested = _only(rows, "nested")
    assert sorted(row["where"] for row in nested) == ["child", "in-process"]
    assert len(_only(rows, "confirmed")) == 1
    analysis = analyze_worker_report(project.parent / "report.jsonl", command_class="direct-pytest", mode="clamp")
    assert analysis.workers_cohort == "unknown"
    assert "worker-budget-nested-session" in analysis.caveats


def test_filtered_child_pytest_is_unobservable_top_level_only(tmp_path):
    child_test = """
    import os, subprocess, sys

    def test_child(tmp_path):
        (tmp_path / "test_inner.py").write_text("def test_inner():\\n    pass\\n")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-n", "2", str(tmp_path / "test_inner.py")],
            env=env,
        )
        assert proc.returncode == 0
    """
    project = _project(tmp_path, {"test_child.py": child_test})
    proc, rows, _ran = _run(project, ["-n", "2", "test_child.py"], budget=2)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not _only(rows, "nested")
    analysis = analyze_worker_report(project.parent / "report.jsonl", command_class="direct-pytest", mode="clamp")
    assert analysis.workers_cohort == "2" and analysis.enforcement == "unchanged"


def test_script_wrapper_other_command_is_capped_through_env(tmp_path):
    project = _project(tmp_path)
    script = project.parent / "run.sh"
    script.write_text(f"#!/bin/sh\nexec {sys.executable} -m pytest -p no:cacheprovider -q -n 8\n", encoding="utf-8")
    script.chmod(0o755)
    proc, rows, _ran = _run(project, [], budget=2, command=[str(script)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _confirmed(rows)["effective"] == 2
    analysis = analyze_worker_report(project.parent / "report.jsonl", command_class="other", mode="clamp")
    assert analysis.workers_cohort == "unknown" and analysis.enforcement == "clamped"
