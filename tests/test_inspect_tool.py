"""Tests for the hardened read-only ``agent-loop inspect`` runner (#1035)."""

from __future__ import annotations

import ast
import io
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from coding_review_agent_loop import inspect_tool
from coding_review_agent_loop.inspect_tool import (
    ENV_ALLOWLIST,
    EXIT_REJECTED,
    FORCED_ENV,
    ExecResult,
    InspectRejected,
    build_subprocess_env,
    run_inspect,
    validate_format,
)

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git is required")


def test_replaced_inspection_helper_is_rejected_before_python_launch(tmp_path, monkeypatch):
    package = tmp_path / "coding_review_agent_loop"
    package.mkdir()
    (package / "inspect_tool.py").write_text("# installed controller\n")
    marker = tmp_path / "launched"
    (package / "inspect_git_subprocess.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    monkeypatch.setattr(inspect_tool, "__file__", str(package / "inspect_tool.py"))
    with pytest.raises(InspectRejected, match="helper identity changed"):
        inspect_tool._subprocess_executor((GIT, "status"), {}, str(tmp_path), True)
    assert not marker.exists()


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class Recorder:
    """Records every subprocess inspect would launch; the gate sees clean config."""

    def __init__(self, gate_stdout: bytes = b"", gate_returncode: int = 0):
        self.calls: list[tuple[list[str], dict[str, str], str, bool]] = []
        self.gate_stdout = gate_stdout
        self.gate_returncode = gate_returncode

    def __call__(self, argv, env, cwd, capture):
        self.calls.append((list(argv), dict(env), cwd, capture))
        if "config" in argv and "--list" in argv:
            return ExecResult(self.gate_returncode, self.gate_stdout, b"")
        return ExecResult(0, b"out\n", b"")


@pytest.fixture
def pinned(tmp_path):
    bindir = tmp_path / "trusted-bin"
    gh = _executable(bindir / "gh", "#!/bin/sh\necho gh-stub \"$@\"\n")
    return {"git": GIT, "gh": str(gh)}


HOSTILE_ENV = {
    "GIT_TRACE": "/tmp/trace",
    "GIT_TRACE2": "/tmp/trace2",
    "GIT_TRACE2_EVENT": "/tmp/event",
    "GIT_TRACE2_PERF": "/tmp/perf",
    "GIT_TRACE_PACKET": "/tmp/packet",
    "GIT_TRACE_SETUP": "/tmp/setup",
    "GIT_TRACE_CURL": "/tmp/curl",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "core.pager",
    "GIT_CONFIG_VALUE_0": "touch /tmp/pwned",
    "GIT_DIR": "/elsewhere",
    "GIT_WORK_TREE": "/elsewhere",
    "GIT_INDEX_FILE": "/elsewhere/index",
    "GIT_SSH_COMMAND": "touch /tmp/pwned",
    "GH_DEBUG": "api",
    "PYTHONPATH": "/checkout",
    "UNKNOWN_VARIABLE": "x",
    "PATH": "/checkout/bin:/usr/bin",
    "HOME": "/home/user",
    "GH_TOKEN": "token",
    "HTTPS_PROXY": "http://proxy:1",
    "https_proxy": "http://proxy:1",
    "SSL_CERT_FILE": "/etc/ssl/ca.pem",
    "LC_CTYPE": "C.UTF-8",
}


def test_build_subprocess_env_is_closed_allowlist():
    env = build_subprocess_env(HOSTILE_ENV, ["/usr/bin", "/opt/gh/bin", "/usr/bin"])
    allowed = set(ENV_ALLOWLIST) | set(FORCED_ENV) | {"PATH"}
    assert all(key in allowed or key.startswith("LC_") for key in env)
    assert env["PATH"] == os.pathsep.join(["/usr/bin", "/opt/gh/bin"])
    assert not any(key.startswith("GIT_") and key not in FORCED_ENV for key in env)
    assert "GH_DEBUG" not in env and "UNKNOWN_VARIABLE" not in env and "PYTHONPATH" not in env
    for key in ("HOME", "GH_TOKEN", "HTTPS_PROXY", "https_proxy", "SSL_CERT_FILE", "LC_CTYPE"):
        assert env[key] == HOSTILE_ENV[key]
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_NO_LAZY_FETCH"] == "1"
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert env["GIT_PAGER"] == env["PAGER"] == env["GH_PAGER"] == "cat"


@pytest.mark.parametrize(
    "args",
    [
        ["git", "diff", "--stat"],
        ["git", "diff", "--name-only", "main...HEAD"],
        ["git", "diff", "-U3", "--", "src/x.py"],
        ["git", "diff", "--unified=5", "--numstat"],
        ["git", "log", "-n", "5", "--oneline"],
        ["git", "log", "--max-count=2", "--format=format:%H %an %s%n"],
        ["git", "log", "--pretty=fuller"],
        ["git", "show", "HEAD", "--stat", "-p"],
        ["git", "status", "--porcelain", "--branch"],
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        ["git", "ls-files"],
        ["gh", "issue", "view", "12", "--comments", "--repo", "o/r"],
        ["gh", "pr", "view", "3", "--json", "title,body"],
        ["gh", "pr", "diff", "3", "--name-only"],
        ["gh", "pr", "checks", "3"],
    ],
)
def test_allowed_vectors_run_hardened_argv_with_closed_env(tmp_path, pinned, args):
    recorder = Recorder()
    code = run_inspect(
        [f"--git={pinned['git']}", f"--gh={pinned['gh']}", *args],
        environ=HOSTILE_ENV,
        cwd=str(tmp_path),
        executor=recorder,
        stdout=io.BytesIO(),
        stderr=io.StringIO(),
    )
    assert code == 0
    gate, command = recorder.calls
    # The gate runs first, with the pinned git and empty trace2 targets.
    assert gate[0][0] == pinned["git"]
    assert gate[0][-6:] == ["config", "--list", "--null", "--show-origin", "--show-scope", "--includes"]
    for target in ("trace2.eventTarget=", "trace2.perfTarget=", "trace2.normalTarget="):
        assert target in gate[0]
    expected_env = build_subprocess_env(HOSTILE_ENV, [os.path.dirname(pinned["git"]), os.path.dirname(pinned["gh"])])
    assert gate[1] == expected_env and command[1] == expected_env
    if args[0] == "git":
        argv = command[0]
        assert argv[0] == pinned["git"]
        assert argv[1] == "--no-pager"
        for item in inspect_tool.FORCED_GIT_CONFIG:
            index = argv.index(item)
            assert argv[index - 1] == "-c"
        sub_index = argv.index(args[1])
        if args[1] in {"diff", "log", "show"}:
            assert argv[sub_index + 1 : sub_index + 3] == ["--no-ext-diff", "--no-textconv"]
    else:
        assert command[0][0] == pinned["gh"]
        assert command[0][1:] == args[1:]


@pytest.mark.parametrize(
    "args",
    [
        ["git", "grep", "x"],
        ["git", "diff", "--open-files-in-pager=touch x"],
        ["git", "diff", "-Otouch"],
        ["git", "diff", "--output=f"],
        ["git", "diff", "--out=f"],
        ["git", "diff", "--ext-diff"],
        ["git", "diff", "--textconv"],
        ["git", "diff", "--no-index", "a", "b"],
        ["git", "log", "--show-signature"],
        ["git", "log", "--format=%G?"],
        ["git", "log", "--pretty=format:%GS"],
        ["git", "log", "--format=format:%G?"],
        ["git", "log", "--format=tformat:%GK"],
        ["git", "log", "--format=%(describe)"],
        ["git", "log", "--format=format:%(describe)"],
        ["git", "log", "--format=format:%Cred%H"],
        ["git", "-c", "k=v", "diff"],
        ["git", "diff", "--sta"],
        ["git", "diff", "--unknown-option"],
        ["git", "log", "-n", "x"],
        ["git", "fetch"],
        ["git"],
        ["gh", "pr", "view", "3", "--web"],
        ["gh", "api", "repos/o/r"],
        ["gh", "pr", "view", "3", "--template", "{{.}}"],
        ["gh", "pr", "view", "3", "--jq", ".title"],
        ["gh", "pr", "merge", "3"],
        ["gh", "pr", "view", "3", "--repo", "o/r;touch x"],
        ["sh", "-c", "id"],
    ],
)
def test_rejected_vectors_never_execute(tmp_path, pinned, args):
    recorder = Recorder()
    err = io.StringIO()
    code = run_inspect(
        [f"--git={pinned['git']}", f"--gh={pinned['gh']}", *args],
        environ=HOSTILE_ENV,
        cwd=str(tmp_path),
        executor=recorder,
        stdout=io.BytesIO(),
        stderr=err,
    )
    assert code == EXIT_REJECTED
    assert recorder.calls == []
    assert "rejected" in err.getvalue()


@pytest.mark.parametrize(
    "prefix",
    [
        [],
        ["--git=relative/git"],
        ["--gh={gh}", "--git={git}"],
        ["--git={git}", "--git={git}"],
        ["--git={git}", "--gh={gh}", "--gh={gh}"],
        ["--git={missing}"],
        ["--git={notexec}"],
    ],
)
def test_pinned_option_positions_and_paths(tmp_path, pinned, prefix):
    notexec = tmp_path / "notexec"
    notexec.write_text("x", encoding="utf-8")
    rendered = [
        item.format(git=pinned["git"], gh=pinned["gh"], missing=tmp_path / "nope", notexec=notexec)
        for item in prefix
    ]
    recorder = Recorder()
    code = run_inspect(
        [*rendered, "git", "status"],
        environ={},
        cwd=str(tmp_path),
        executor=recorder,
        stdout=io.BytesIO(),
        stderr=io.StringIO(),
    )
    assert code == EXIT_REJECTED
    assert recorder.calls == []


def test_pinned_option_after_subcommand_is_rejected(tmp_path, pinned):
    recorder = Recorder()
    code = run_inspect(
        [f"--git={pinned['git']}", "git", "status", f"--git={tmp_path}/git"],
        environ={}, cwd=str(tmp_path), executor=recorder,
        stdout=io.BytesIO(), stderr=io.StringIO(),
    )
    assert code == EXIT_REJECTED and recorder.calls == []


def test_gh_subcommand_requires_pinned_gh(tmp_path, pinned):
    recorder = Recorder()
    err = io.StringIO()
    code = run_inspect(
        [f"--git={pinned['git']}", "gh", "pr", "view", "3"],
        environ={}, cwd=str(tmp_path), executor=recorder,
        stdout=io.BytesIO(), stderr=err,
    )
    assert code == EXIT_REJECTED and recorder.calls == []
    assert "gh inspection is unavailable" in err.getvalue()


@pytest.mark.parametrize("value", ["oneline", "fuller", "format:%H %h %an <%ae> %s%n%b", "tformat:%%%d%D"])
def test_validate_format_accepts_closed_set(value):
    validate_format(value)


@pytest.mark.parametrize("value", ["%H", "format:%G?", "format:%GG", "format:%x00", "format:%C(red)", "raw"])
def test_validate_format_rejects_others(value):
    with pytest.raises(InspectRejected):
        validate_format(value)


def test_gate_rejects_non_allowlisted_key_naming_origin(tmp_path, pinned):
    listing = (
        b"local\x00file:.git/config\x00core.bare\nfalse\x00"
        b"local\x00file:/x/inc.cfg\x00filter.x.clean\ntouch m\x00"
    )
    recorder = Recorder(gate_stdout=listing)
    err = io.StringIO()
    code = run_inspect(
        [f"--git={pinned['git']}", "git", "status"],
        environ={}, cwd=str(tmp_path), executor=recorder,
        stdout=io.BytesIO(), stderr=err,
    )
    assert code == EXIT_REJECTED
    assert len(recorder.calls) == 1  # only the gate ran
    assert "filter.x.clean" in err.getvalue() and "/x/inc.cfg" in err.getvalue()


def test_gate_failure_fails_closed(tmp_path, pinned):
    recorder = Recorder(gate_returncode=128)
    code = run_inspect(
        [f"--git={pinned['git']}", "git", "status"],
        environ={}, cwd=str(tmp_path), executor=recorder,
        stdout=io.BytesIO(), stderr=io.StringIO(),
    )
    assert code == EXIT_REJECTED and len(recorder.calls) == 1


def test_gate_runs_before_any_gh_subprocess(tmp_path, pinned):
    recorder = Recorder(gate_stdout=b"local\x00file:.git/config\x00alias.st\nstatus\x00")
    code = run_inspect(
        [f"--git={pinned['git']}", f"--gh={pinned['gh']}", "gh", "pr", "view", "3"],
        environ={}, cwd=str(tmp_path), executor=recorder,
        stdout=io.BytesIO(), stderr=io.StringIO(),
    )
    assert code == EXIT_REJECTED
    assert [call[0][0] for call in recorder.calls] == [pinned["git"]]


def test_core_bare_true_is_rejected():
    with pytest.raises(InspectRejected):
        inspect_tool.check_repository_config([("local", "file:.git/config", "core.bare", "true")])


def test_inspect_tool_imports_only_standard_library():
    source = Path(inspect_tool.__file__).read_text(encoding="utf-8")
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "inspect_tool must not import package modules"
            modules.add((node.module or "").split(".")[0])
    assert modules <= set(sys.stdlib_module_names)


# ------------------------------------------------------------ real git


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [GIT, *args], cwd=repo, check=True, capture_output=True,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(repo.parent),
             "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"},
    )


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "remote.origin.url", "https://github.com/o/r.git")
    _git(repo, "config", "branch.feature.with.dots.merge", "refs/heads/feature.with.dots")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "first")
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "second")
    return repo


def _marker_script(tmp_path: Path) -> tuple[Path, Path]:
    marker = tmp_path / "MARKER"
    script = _executable(tmp_path / "hostile" / "run.sh", f"#!/bin/sh\ntouch {marker}\ncat\n")
    return script, marker


def _real(repo: Path, args: list[str], *, environ=None, gh: str | None = None):
    out, err = io.BytesIO(), io.StringIO()
    prefix = [f"--git={GIT}"] + ([f"--gh={gh}"] if gh else [])
    code = run_inspect(
        [*prefix, *args],
        environ=environ if environ is not None else {"HOME": str(repo.parent)},
        cwd=str(repo), stdout=out, stderr=err,
    )
    return code, out.getvalue().decode(), err.getvalue()


def test_real_git_typical_clone_config_is_accepted(repo):
    code, out, err = _real(repo, ["git", "log", "-n", "1", "--format=format:%s"])
    assert code == 0, err
    assert out.strip() == "second"
    code, out, _ = _real(repo, ["git", "diff", "HEAD~1...HEAD", "--stat"])
    assert code == 0 and "a.txt" in out


def test_real_git_planted_signature_program_never_runs(repo, tmp_path):
    script, marker = _marker_script(tmp_path)
    _git(repo, "config", "log.showSignature", "true")
    _git(repo, "config", "gpg.program", str(script))
    for args in (["git", "log", "-n", "1"], ["git", "show", "HEAD"]):
        code, _out, err = _real(repo, args)
        assert code == EXIT_REJECTED
        assert "gpg.program" in err or "log.showsignature" in err
    assert not marker.exists()


def test_real_git_planted_clean_filter_never_runs(repo, tmp_path):
    script, marker = _marker_script(tmp_path)
    _git(repo, "config", "filter.x.clean", str(script))
    (repo / ".gitattributes").write_text("* filter=x\n", encoding="utf-8")
    (repo / "a.txt").write_text("modified\n", encoding="utf-8")
    for args in (["git", "diff"], ["git", "status"]):
        code, _out, err = _real(repo, args)
        assert code == EXIT_REJECTED
        assert "filter.x.clean" in err
    assert not marker.exists()


def _plant_submodule_clean_filter(repo: Path, script: Path) -> None:
    """Coder-style setup: a populated gitlink whose own config runs ``script``."""
    sub = repo / "sub"
    sub.mkdir()
    _git(sub, "init", "-q")
    _git(sub, "config", "user.name", "Test")
    _git(sub, "config", "user.email", "t@example.com")
    (sub / "b.txt").write_text("one\n", encoding="utf-8")
    _git(sub, "add", "b.txt")
    _git(sub, "commit", "-q", "-m", "sub")
    _git(repo, "add", "sub")
    (repo / ".gitmodules").write_text(
        '[submodule "sub"]\n\tpath = sub\n\turl = ./sub\n\tignore = none\n', encoding="utf-8"
    )
    _git(repo, "add", ".gitmodules")
    _git(repo, "commit", "-q", "-m", "add submodule")
    # Planted only in the submodule's config, which the top-level gate never scans.
    (sub / ".gitattributes").write_text("* filter=x\n", encoding="utf-8")
    _git(sub, "config", "filter.x.clean", str(script))
    (sub / "b.txt").write_text("two\n", encoding="utf-8")


def test_real_git_submodule_clean_filter_never_runs(repo, tmp_path):
    script, marker = _marker_script(tmp_path)
    _plant_submodule_clean_filter(repo, script)
    assert not marker.exists()
    for args in (
        ["git", "status"],
        ["git", "status", "--porcelain"],
        ["git", "diff"],
        ["git", "diff", "--stat"],
        ["git", "log", "-n", "1", "-p"],
        ["git", "show", "HEAD"],
    ):
        code, _out, err = _real(repo, args)
        assert code == 0, (args, err)
        assert not marker.exists(), args


def test_submodule_recursion_is_forced_off_in_git_argv():
    for sub in ("diff", "log", "show", "status"):
        argv = inspect_tool.git_argv("/usr/bin/git", sub, [])
        assert "--ignore-submodules=all" in argv
        assert argv.index("--ignore-submodules=all") > argv.index(sub)
    for item in ("diff.ignoreSubmodules=all", "submodule.recurse=false"):
        assert item in inspect_tool.FORCED_GIT_CONFIG
    assert inspect_tool.FORCED_GIT_CONFIG[-3:] == (
        "trace2.eventTarget=", "trace2.perfTarget=", "trace2.normalTarget=",
    )


def test_real_git_included_filter_is_rejected_naming_included_origin(repo, tmp_path):
    script, marker = _marker_script(tmp_path)
    included = tmp_path / "included.cfg"
    included.write_text(f'[filter "x"]\n\tclean = {script}\n', encoding="utf-8")
    _git(repo, "config", "include.path", str(included))
    (repo / ".gitattributes").write_text("* filter=x\n", encoding="utf-8")
    code, _out, err = _real(repo, ["git", "status"])
    assert code == EXIT_REJECTED
    assert "include.path" in err or str(included) in err
    assert not marker.exists()


def test_real_git_include_if_is_rejected(repo, tmp_path):
    included = tmp_path / "cond.cfg"
    included.write_text("[alias]\n\tst = status\n", encoding="utf-8")
    _git(repo, "config", f"includeIf.gitdir:{repo}/.path", str(included))
    code, _out, err = _real(repo, ["git", "status"])
    assert code == EXIT_REJECTED
    assert "includeif" in err.lower()


@pytest.mark.parametrize(
    "key,value",
    [
        ("alias.diff", "!touch MARKER"),
        ("core.hooksPath", "hooks"),
        ("core.worktree", "/tmp"),
        ("diff.x.command", "touch MARKER"),
        ("remote.origin.promisor", "true"),
        ("credential.helper", "!touch MARKER"),
        ("trace2.eventTarget", "TRACE"),
    ],
)
def test_real_git_other_planted_keys_are_rejected(repo, key, value):
    _git(repo, "config", key, value)
    code, _out, err = _real(repo, ["git", "status"])
    assert code == EXIT_REJECTED
    assert key.lower() in err.lower()
    assert not (repo / "MARKER").exists()
    assert not (repo / "TRACE").exists()


def test_real_git_global_only_filter_is_ignored(repo, tmp_path):
    script, marker = _marker_script(tmp_path)
    home = tmp_path / "home"
    (home / ".config" / "git").mkdir(parents=True)
    (home / ".gitconfig").write_text(f'[filter "x"]\n\tclean = {script}\n', encoding="utf-8")
    (home / ".config" / "git" / "config").write_text(
        f'[filter "x"]\n\tclean = {script}\n', encoding="utf-8"
    )
    (repo / ".gitattributes").write_text("* filter=x\n", encoding="utf-8")
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    code, _out, err = _real(
        repo, ["git", "status", "--short"],
        environ={"HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config")},
    )
    assert code == 0, err
    assert not marker.exists()


@pytest.mark.parametrize(
    "variable", ["GIT_TRACE", "GIT_TRACE2_EVENT", "GIT_TRACE2_PERF", "GIT_TRACE_PACKET"]
)
@pytest.mark.parametrize("inside", [True, False])
def test_real_git_inherited_trace_variables_write_nothing(repo, tmp_path, variable, inside):
    target = (repo / "trace-out") if inside else (tmp_path / "trace-out")
    environ = {"HOME": str(tmp_path), variable: str(target)}
    for args in (["git", "status"], ["git", "log", "-n", "1"]):
        code, _out, err = _real(repo, args, environ=environ)
        assert code == 0, err
    _git(repo, "config", "alias.x", "status")
    code, _out, _err = _real(repo, ["git", "status"], environ=environ)
    assert code == EXIT_REJECTED
    assert not target.exists()


def test_real_process_checkout_local_git_and_gh_never_run(repo, tmp_path):
    marker = tmp_path / "PLANTED"
    checkout_bin = repo / "bin"
    for name in ("git", "gh"):
        _executable(checkout_bin / name, f"#!/bin/sh\ntouch {marker}\n")
    # A trusted gh stub that itself looks up `git` on its PATH.
    trusted_gh = _executable(
        tmp_path / "trusted" / "gh", "#!/bin/sh\ngit rev-parse HEAD >/dev/null\necho gh-ok\n"
    )
    environ = {"HOME": str(tmp_path), "PATH": f"{checkout_bin}:{os.environ.get('PATH', '')}"}
    for args in (["git", "status"], ["git", "log", "-n", "1"], ["gh", "pr", "checks", "1"]):
        code, out, err = _real(repo, args, environ=environ, gh=str(trusted_gh))
        assert code == 0, err
    assert "gh-ok" in out
    assert not marker.exists()


def test_module_entry_point_loads_only_inspect_tool(repo, tmp_path):
    """`python -I -m coding_review_agent_loop.cli inspect` must not import the rest."""
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    # An isolated interpreter ignores pytest's PYTHONPATH; bootstrap the
    # checkout source path while retaining -I for the inspected entry point.
    bootstrap = (f"import sys, runpy; sys.path.insert(0, {source_root!r}); "
                 "runpy.run_module('coding_review_agent_loop.cli', run_name='__main__')")
    completed = subprocess.run(
        [sys.executable, "-I", "-X", "importtime", "-c", bootstrap,
         "inspect", f"--git={GIT}", "git", "rev-parse", "HEAD"],
        cwd=repo, capture_output=True, text=True, timeout=60,
        env={"HOME": str(tmp_path), "PATH": os.environ.get("PATH", "")},
    )
    assert completed.returncode == 0, completed.stderr
    imported = {
        line.rsplit("|", 1)[-1].strip()
        for line in completed.stderr.splitlines()
        if line.startswith("import time:")
    }
    ours = {name for name in imported if name.startswith("coding_review_agent_loop")}
    assert ours <= {"coding_review_agent_loop", "coding_review_agent_loop.inspect_tool"}
