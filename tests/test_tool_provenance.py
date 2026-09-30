"""Tool provenance capture, rendering and run-level recording (#1111)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from coding_review_agent_loop import agent_permissions as ap
from coding_review_agent_loop import cli as cli_module
from coding_review_agent_loop import tool_provenance as tp
from coding_review_agent_loop.cli import main
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.inspect_tool import ExecResult
from coding_review_agent_loop.orchestrator import (
    _new_usage_context,
    _persist_usage_summary,
    _run_validated_agent,
)
from coding_review_agent_loop.usage import RunUsageContext

from agent_loop_helpers import FakeRunner, make_config

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git is required")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        [GIT, "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real repo tracking a nested package layout, used as the tool checkout."""
    root = tmp_path / "toolrepo"
    package = root / "src" / "coding_review_agent_loop"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "tool_provenance.py").write_text("")
    _git(root.parent, "init", str(root))
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "init")
    monkeypatch.setattr(tp, "__file__", str(package / "tool_provenance.py"))
    return root


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    return make_config(tmp_path)


def _capture(config, **kwargs):
    return tp.capture_tool_provenance(config, which=lambda _n: GIT, **kwargs)


def test_clean_nested_checkout(repo, config):
    result = _capture(config)
    assert result["error"] is None
    assert result["commit"] == _git(repo, "rev-parse", "HEAD")
    assert result["dirty"] is False
    assert "(clean)" in tp.format_tool_provenance_line(result)


def test_dirty_checkout_reports_paths(repo, config):
    (repo / "src" / "coding_review_agent_loop" / "__init__.py").write_text("x")
    (repo / "new.txt").write_text("y")
    result = _capture(config)
    assert result["dirty"] is True
    assert any("new.txt" in line for line in result["dirty_paths_sample"])
    assert "DIRTY: 2 changed paths" in tp.format_tool_provenance_line(result)
    assert (repo / "new.txt").exists()


def test_non_repository(tmp_path, monkeypatch, config):
    package = tmp_path / "plain" / "pkg"
    package.mkdir(parents=True)
    monkeypatch.setattr(tp, "__file__", str(package / "tool_provenance.py"))
    result = _capture(config)
    assert result["commit"] is None
    assert result["error"] in {"not-a-git-checkout", "config-gate-refused"}


def test_ancestor_repository_is_not_attributed(tmp_path, monkeypatch, config):
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "a.txt").write_text("a")
    _git(outer.parent, "init", str(outer))
    _git(outer, "add", "a.txt")
    _git(outer, "commit", "-m", "x")
    package = outer / "venv" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "tool_provenance.py").write_text("")
    monkeypatch.setattr(tp, "__file__", str(package / "tool_provenance.py"))
    result = _capture(config)
    assert result["commit"] is None
    assert result["error"] == "package-not-in-repository"


def test_hostile_config_is_refused_without_execution(repo, config, tmp_path):
    marker = tmp_path / "marker"
    script = tmp_path / "evil.sh"
    script.write_text(f"#!/bin/sh\ntouch {marker}\n")
    script.chmod(0o755)
    _git(repo, "config", "core.fsmonitor", str(script))
    _git(repo, "config", "filter.x.clean", str(script))
    result = _capture(config)
    assert result["error"] == "config-gate-refused"
    assert result["commit"] is None
    assert not marker.exists()


def test_untrusted_executable_in_checkout_is_never_run(repo, config, tmp_path):
    marker = tmp_path / "marker"
    inside = Path(config.claude_dir)
    inside.mkdir(exist_ok=True)
    fake = inside / "git"
    fake.write_text(f"#!/bin/sh\ntouch {marker}\n")
    fake.chmod(0o755)
    link = tmp_path / "gitlink"
    link.symlink_to(fake)
    calls = []

    def recorder(*args):
        calls.append(args)
        return ExecResult(0)

    for candidate in (str(fake), str(link)):
        result = tp.capture_tool_provenance(config, which=lambda _n, c=candidate: c, executor=recorder)
        assert result["error"] == "git-untrusted-location"
        assert result["commit"] is None
        assert "refused" in tp.format_tool_provenance_line(result)
    assert calls == []
    assert not marker.exists()


def test_executable_location_refusal_agrees_with_provenance_check(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked-tmp"
    link.symlink_to(real)
    monkeypatch.setattr(tempfile, "tempdir", str(link))
    config = make_config(tmp_path)
    resolved_root = ap._response_component_paths(config)[0]
    inside = resolved_root / "bin" / "git"
    inside.parent.mkdir(parents=True)
    inside.write_text("")
    assert not ap._is_within(str(inside), str(ap.response_root(config)))
    assert ap.executable_location_refusal(str(inside), config)
    with pytest.raises(AgentLoopError):
        ap._check_provenance_outside(
            _FakeProvenance(str(inside)), [], response_root_path=str(resolved_root)
        )
    outside = tmp_path / "usr" / "git"
    outside.parent.mkdir()
    outside.write_text("")
    assert ap.executable_location_refusal(str(outside), config) is None


class _FakeProvenance:
    def __init__(self, git):
        self.git = git


@pytest.fixture(autouse=True)
def _fake_provenance_paths(monkeypatch):
    monkeypatch.setattr(
        ap, "_provenance_paths", lambda p: [("git", p.git)] if hasattr(p, "git") else []
    )


def test_oserror_and_timeout_never_raise(repo, config):
    def boom(*_a):
        raise OSError("nope")

    def slow(*_a):
        raise subprocess.TimeoutExpired("git", 1)

    assert _capture(config, executor=boom)["error"] == "git-unavailable"
    assert _capture(config, executor=slow)["error"] == "git-timeout"
    assert tp.capture_tool_provenance(config, which=lambda _n: None)["error"] == "git-unavailable"


def test_deadline_bounds_whole_capture(repo, config, monkeypatch):
    import time

    seen = []
    real_run = subprocess.run

    def slow_run(argv, **kwargs):
        seen.append(kwargs["timeout"])
        time.sleep(kwargs["timeout"])
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(tp.subprocess, "run", slow_run)
    start = time.monotonic()
    result = _capture(config, deadline_seconds=0.6)
    assert time.monotonic() - start < 1.5
    assert result["error"] == "git-timeout"
    assert result["commit"] is None
    assert seen and max(seen) <= 0.6
    monkeypatch.setattr(tp.subprocess, "run", real_run)


def _fail_on(repo_config, sub, *, timeout=False):
    real = tp.inspect_tool._subprocess_executor

    def executor(argv, env, cwd, capture):
        if sub in argv:
            if timeout:
                raise subprocess.TimeoutExpired(argv, 1)
            return ExecResult(1, b"", b"boom")
        return real(argv, env, cwd, capture)

    return executor


def test_partial_results(repo, config):
    head = _git(repo, "rev-parse", "HEAD")
    failed = _capture(config, executor=_fail_on(config, "status"))
    assert (failed["commit"], failed["dirty"], failed["error"]) == (head, None, "git-status-failed")
    timed = _capture(config, executor=_fail_on(config, "status", timeout=True))
    assert (timed["commit"], timed["dirty"], timed["error"]) == (head, None, "git-timeout")
    member = _capture(config, executor=_fail_on(config, "ls-files", timeout=True))
    assert member["commit"] is None and member["error"] == "git-timeout"


def test_format_variants():
    sha = "a" * 40
    root = "/r"
    base = {"commit": sha, "checkout_root": root, "error": None}
    clean = tp.format_tool_provenance_line({**base, "dirty": False})
    dirty = tp.format_tool_provenance_line({**base, "dirty": True, "dirty_paths_sample": ["a"]})
    unknown = tp.format_tool_provenance_line({**base, "dirty": None, "error": "git-timeout"})
    none = tp.format_tool_provenance_line({"commit": None, "error": "not-a-git-checkout"})
    assert "(clean)" in clean and "DIRTY" in dirty
    assert sha in unknown and "(clean)" not in unknown and "cleanliness unknown: git-timeout" in unknown
    assert none == "agent-loop tool commit unknown (not-a-git-checkout)"
    assert tp.format_tool_commit_suffix({**base, "dirty": False}) == "tool_commit=aaaaaaa clean"
    assert tp.format_tool_commit_suffix({**base, "dirty": True}) == "tool_commit=aaaaaaa dirty"
    assert tp.format_tool_commit_suffix({**base, "dirty": None}) == "tool_commit=aaaaaaa dirty=unknown"
    assert tp.format_tool_commit_suffix({"commit": None}) == "tool_commit=unknown"
    assert tp.format_tool_commit_suffix(None) == "tool_commit=unknown"


def test_summary_payload_and_dispatch_idempotent(tmp_path, config):
    context = _new_usage_context(config)
    assert context.tool_provenance["commit"]
    context.note_agent_dispatch()
    first = context.first_agent_dispatch_at
    context.note_agent_dispatch()
    assert context.first_agent_dispatch_at == first
    payload = context.summary_payload()
    assert payload["tool_provenance"]["commit"] == context.tool_provenance["commit"]
    assert set(payload["timing"]) == {
        "process_started_at", "run_started_at", "first_agent_dispatch_at", "startup_gap_seconds",
    }
    assert payload["timing"]["startup_gap_seconds"] is not None
    assert "totals" in payload and "calls" in payload


def test_usage_summary_line_suffix(tmp_path, capsys):
    config = make_config(tmp_path, quiet=False)
    context = RunUsageContext(
        run_id="r", summary_path=tmp_path / "s.json",
        tool_provenance={"commit": "b" * 40, "dirty": None},
    )
    _persist_usage_summary(config, context)
    assert "tool_commit=bbbbbbb dirty=unknown" in capsys.readouterr().err
    assert json.loads((tmp_path / "s.json").read_text())["timing"]["run_started_at"] is None
    context.tool_provenance = None
    _persist_usage_summary(config, context)
    assert "tool_commit=unknown" in capsys.readouterr().err


@pytest.mark.parametrize("dry_run", [False, True])
def test_dispatch_stamp_precedes_completion(tmp_path, dry_run):
    import time

    config = make_config(tmp_path, dry_run=dry_run)
    context = _new_usage_context(config)
    observed = {}

    class Delayed(FakeRunner):
        pass

    from coding_review_agent_loop import orchestrator

    real = orchestrator.run_agent_result

    def delayed(*args, **kwargs):
        observed["at_call"] = context.first_agent_dispatch_at
        time.sleep(0.05)
        return real(*args, **kwargs)

    runner = FakeRunner(codex_outputs=["OK"], claude_outputs=["OK"])
    orchestrator.run_agent_result = delayed
    try:
        _run_validated_agent(
            runner, agent="codex", config=config, prompt="p", marker_description="OK",
            validate=lambda t: t, usage_context=context,
        )
    finally:
        orchestrator.run_agent_result = real
    assert observed["at_call"] is not None
    assert context.first_agent_dispatch_at == observed["at_call"]


def test_validated_agent_without_context(tmp_path):
    config = make_config(tmp_path)
    runner = FakeRunner(codex_outputs=["OK"])
    response = _run_validated_agent(
        runner, agent="codex", config=config, prompt="p", marker_description="OK",
        validate=lambda t: t,
    )
    assert response.text


def _cli_setup(monkeypatch, tmp_path, calls):
    monkeypatch.setattr(tp, "_PROCESS_PROVENANCE", None)
    monkeypatch.setattr(
        tp, "capture_tool_provenance",
        lambda config, **_k: calls.append(1) or {
            "commit": "c" * 40, "dirty": False, "checkout_root": "/r", "error": None,
        },
    )


def test_cli_logs_line_before_claim_conflict(tmp_path, monkeypatch, capsys):
    calls = []
    _cli_setup(monkeypatch, tmp_path, calls)
    config = make_config(tmp_path, quiet=False)
    monkeypatch.setattr(cli_module, "config_from_args", lambda *a, **k: config)

    def conflict(**_k):
        raise AgentLoopError("claim conflict")

    monkeypatch.setattr(cli_module, "workdir_claim_scope", conflict)
    assert main(["pr", "77", "--repo", "OWNER/REPO"]) == 1
    err = capsys.readouterr().err
    assert "agent-loop tool commit " + "c" * 40 + " (clean)" in err
    assert err.index("agent-loop tool commit") < err.index("claim conflict")
    assert calls == [1]


def test_cli_quiet_suppresses_line(tmp_path, monkeypatch, capsys):
    calls = []
    _cli_setup(monkeypatch, tmp_path, calls)
    config = make_config(tmp_path, quiet=True)
    monkeypatch.setattr(cli_module, "config_from_args", lambda *a, **k: config)
    monkeypatch.setattr(
        cli_module, "workdir_claim_scope",
        lambda **_k: (_ for _ in ()).throw(AgentLoopError("claim conflict")),
    )
    assert main(["pr", "77", "--repo", "OWNER/REPO", "--quiet"]) == 1
    assert "agent-loop tool commit" not in capsys.readouterr().err
    assert calls == [1]
    assert tp._PROCESS_PROVENANCE["commit"] == "c" * 40


def test_process_capture_is_once_and_shared_by_nested_contexts(tmp_path, monkeypatch, capsys):
    calls = []
    _cli_setup(monkeypatch, tmp_path, calls)
    config = make_config(tmp_path, quiet=False)
    first = _new_usage_context(config)
    second = _new_usage_context(config)
    assert first.tool_provenance is second.tool_provenance
    assert calls == [1]
    assert capsys.readouterr().err.count("agent-loop tool commit") == 1


# ---- owning-run integration through the real finally path ----

_TASK_OUTPUTS = dict(
    claude_outputs=[
        "Implemented.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
    ],
    codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    pr_payload={"baseRefName": "develop", "body": "Fixes #56"},
    repo_default_branch="develop",
)


def _summaries(config):
    return sorted(Path(config.log_dir).glob("*-usage-summary.json"))


@pytest.mark.parametrize("quiet", [False, True])
def test_owning_task_run_writes_summary_with_provenance(tmp_path, capsys, quiet):
    from coding_review_agent_loop.orchestrator import run_task_loop

    config = make_config(
        tmp_path, base=None, reviewer="codex", auto_agent_dirs=("claude", "codex"), quiet=quiet,
    )
    assert run_task_loop(FakeRunner(**_TASK_OUTPUTS), task_text="Add /healthz.", config=config) == 0
    (path,) = _summaries(config)
    payload = json.loads(path.read_text())
    assert payload["tool_provenance"]["commit"]
    assert payload["tool_provenance"]["dirty"] is False
    timing = payload["timing"]
    assert timing["run_started_at"] and timing["first_agent_dispatch_at"]
    assert timing["first_agent_dispatch_at"] >= timing["run_started_at"]
    assert timing["startup_gap_seconds"] is not None
    err = capsys.readouterr().err
    assert ("tool_commit=" in err) is (not quiet)


def test_failing_owning_run_still_writes_provenance_summary(tmp_path):
    from coding_review_agent_loop.orchestrator import run_task_loop

    config = make_config(tmp_path)
    with pytest.raises(AgentLoopError, match="Task text is empty"):
        run_task_loop(FakeRunner(), task_text="  ", config=config)
    (path,) = _summaries(config)
    payload = json.loads(path.read_text())
    assert payload["tool_provenance"]["commit"]
    assert payload["timing"]["run_started_at"]
    assert payload["timing"]["first_agent_dispatch_at"] is None
    assert payload["timing"]["startup_gap_seconds"] is None


def test_nested_run_shares_context_without_recapture(tmp_path, monkeypatch, capsys):
    from coding_review_agent_loop.orchestrator import run_pr_loop

    calls = []
    _cli_setup(monkeypatch, tmp_path, calls)
    config = make_config(tmp_path, quiet=False, reviewer="codex", auto_agent_dirs=("codex",),
                         create_dirs=False)
    Path(config.claude_dir).mkdir(parents=True)
    Path(config.gemini_dir).mkdir(parents=True)
    Path(config.codex_dir).mkdir(parents=True, exist_ok=True)
    outer = _new_usage_context(config)
    before = dict(outer.tool_provenance)
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"baseRefName": "main", "body": "Fixes #1"},
    )
    assert run_pr_loop(runner, pr_number=77, config=config, usage_context=outer) == 0
    assert calls == [1]
    assert outer.tool_provenance == before
    assert _summaries(config) == []  # the nested run does not persist the shared context
    assert outer.first_agent_dispatch_at is not None
    assert capsys.readouterr().err.count("agent-loop tool commit") == 1
    _persist_usage_summary(config, outer)
    (path,) = _summaries(config)
    assert json.loads(path.read_text())["tool_provenance"]["commit"] == "c" * 40


def test_relative_path_git_is_pinned_to_one_absolute_executable(repo, config, tmp_path, monkeypatch):
    """A relative PATH hit is validated and executed as the same absolute path."""
    seen = []

    def recorder(argv, env, cwd, capture):
        seen.append(argv[0])
        return ExecResult(1, b"", b"")

    monkeypatch.chdir(tmp_path)
    result = tp.capture_tool_provenance(config, which=lambda _n: "bin/git", executor=recorder)
    expected = os.path.abspath("bin/git")
    assert seen and set(seen) == {expected}
    assert result["error"] in {"not-a-git-checkout", "config-gate-refused"}


def test_relative_path_git_inside_checkout_is_refused(repo, config, tmp_path, monkeypatch):
    inside = Path(config.claude_dir)
    (inside / "bin").mkdir(parents=True, exist_ok=True)
    (inside / "bin" / "git").write_text("#!/bin/sh\n")
    (inside / "bin" / "git").chmod(0o755)
    monkeypatch.chdir(inside)
    calls = []
    result = tp.capture_tool_provenance(
        config, which=lambda _n: "bin/git", executor=lambda *a: calls.append(a) or ExecResult(0)
    )
    assert result["error"] == "git-untrusted-location"
    assert calls == []
