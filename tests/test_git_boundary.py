"""Real Git regressions for agent-loop-owned operations in writable checkouts."""

from __future__ import annotations

import os
import base64
import subprocess
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from coding_review_agent_loop import secure_git
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.git_transport import default_origin, import_ref, trusted_url
from coding_review_agent_loop.local_test_evidence import SnapshotCancelled, _run_git
from coding_review_agent_loop.runner import Runner


def git(path: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(("git", "-C", str(path), *args), capture_output=True,
                            text=True, check=False)
    if check and result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


@pytest.fixture
def repositories(tmp_path: Path, monkeypatch):
    for key in ("GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0",
                "GIT_TRACE", "GIT_DIR", "GIT_WORK_TREE", "GIT_SSH_COMMAND"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    origin, seed, checkout = (tmp_path / name for name in ("origin.git", "seed", "checkout"))
    subprocess.run(("git", "init", "-q", "--bare", "-b", "main", str(origin)), check=True)
    subprocess.run(("git", "init", "-q", "-b", "main", str(seed)), check=True)
    (seed / "a.txt").write_text("one\n")
    git(seed, "add", "a.txt")
    git(seed, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "one")
    git(seed, "push", "-q", str(origin), "main")
    subprocess.run(("git", "clone", "-q", str(origin), str(checkout)), check=True)
    return origin, seed, checkout


def test_planted_hook_fsmonitor_and_environment_do_not_execute(repositories, monkeypatch, tmp_path):
    origin, _, checkout = repositories
    marker = tmp_path / "planted"
    trace = tmp_path / "trace"
    hook = checkout / ".git" / "hooks" / "post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    git(checkout, "config", "core.fsmonitor", f"touch {marker}")
    sha = git(checkout, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", f"touch {marker}")
    monkeypatch.setenv("GIT_TRACE", str(trace))
    monkeypatch.setenv("GIT_DIR", str(origin))
    before = dict(os.environ)
    runner = Runner()
    assert runner.run(("git", "status", "--porcelain"), cwd=checkout).returncode == 0
    assert runner.run(("git", "checkout", "--detach", sha), cwd=checkout).returncode == 0
    assert runner.run_binary(("git", "rev-parse", "HEAD"), cwd=checkout).stdout.strip() == sha.encode()
    assert not marker.exists() and not trace.exists()
    assert dict(os.environ) == before


@pytest.mark.parametrize("operation", ["status", "reset", "clean"])
def test_local_git_binds_worktree_despite_repo_config(repositories, tmp_path, operation):
    _, _, checkout = repositories
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("outside\n")
    (outside / "untracked.txt").write_text("outside\n")
    git(checkout, "config", "core.worktree", str(outside))
    (checkout / "a.txt").write_text("dirty\n")
    (checkout / "untracked.txt").write_text("checkout\n")
    args = {"status": ("status", "--porcelain"), "reset": ("reset", "--hard"),
            "clean": ("clean", "-fd")}[operation]
    result = Runner().run(("git", *args), cwd=checkout)
    if operation == "status":
        assert "a.txt" in result.stdout and "untracked.txt" in result.stdout
    elif operation == "reset":
        assert (checkout / "a.txt").read_text() == "one\n"
    else:
        assert not (checkout / "untracked.txt").exists()
    assert (outside / "a.txt").read_text() == "outside\n"
    assert (outside / "untracked.txt").read_text() == "outside\n"


def test_nested_git_probe_uses_checkout_root_despite_repo_config(repositories, tmp_path):
    _, _, checkout = repositories
    outside = tmp_path / "outside"
    outside.mkdir()
    git(checkout, "config", "core.worktree", str(outside))
    nested = checkout / "package" / "module"
    nested.mkdir(parents=True)
    (checkout / "a.txt").write_text("dirty\n")
    result = Runner().run(("git", "status", "--porcelain"), cwd=nested)
    assert "a.txt" in result.stdout


def test_private_pack_import_and_ref_cas(repositories):
    origin, _, checkout = repositories
    runner = Runner()
    sha = import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main",
                     repo="OWNER/REPO", runner=runner, local_origin=origin)
    assert sha == git(checkout, "rev-parse", "HEAD")
    assert runner.run(("git", "cat-file", "-t", sha), cwd=checkout).stdout.strip() == "commit"
    assert git(checkout, "rev-parse", "refs/remotes/origin/main") == sha


def test_plain_pr_head_mismatch_keeps_checkout_head(repositories, tmp_path):
    from agent_loop_helpers import make_config
    from coding_review_agent_loop.config import sync_checkout_to_pr

    origin, seed, checkout = repositories
    previous = git(checkout, "rev-parse", "HEAD")
    (seed / "a.txt").write_text("moved\n")
    git(seed, "add", "a.txt")
    git(seed, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "moved")
    git(seed, "push", "-q", str(origin), "HEAD:refs/pull/7/head")
    config = make_config(tmp_path, claude_dir=checkout, trusted_local_origin=origin)
    with pytest.raises(AgentLoopError, match="advertises head SHA"):
        sync_checkout_to_pr(
            config, Runner(), path=checkout, label="Default claude workdir",
            default_owned=True, pr_number=7, pr_metadata=SimpleNamespace(head_sha=previous),
        )
    assert git(checkout, "rev-parse", "HEAD") == previous
    assert (checkout / "a.txt").read_text() == "one\n"


def test_private_import_rejects_tracking_ref_race(repositories):
    origin, _, checkout = repositories
    tracking = "refs/remotes/origin/main"

    class MoveRef(Runner):
        def run(self, args, *, cwd, **kwargs):
            if tuple(args[:3]) == ("git", "update-ref", tracking):
                git(checkout, "update-ref", "-d", tracking)
            return super().run(args, cwd=cwd, **kwargs)

    with pytest.raises(AgentLoopError, match="Command failed"):
        import_ref(checkout, "refs/heads/main", tracking,
                   repo="OWNER/REPO", runner=MoveRef(), local_origin=origin)
    assert git(checkout, "show-ref", "--verify", tracking, check=False) == ""


def test_wrong_origin_is_rejected_before_private_fetch(repositories, monkeypatch, tmp_path):
    origin, _, checkout = repositories
    marker = tmp_path / "transport-started"
    git(checkout, "config", "remote.origin.url", "https://evil.example/OWNER/REPO.git")
    from coding_review_agent_loop import git_transport

    monkeypatch.setattr(git_transport, "_private_git", lambda *a, **k: marker.touch())
    with pytest.raises(AgentLoopError, match="operator-trusted endpoint"):
        import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main",
                   repo="OWNER/REPO", runner=Runner(), local_origin=origin)
    assert not marker.exists()
    assert trusted_url("OWNER/REPO", "git@github.com:OWNER/REPO.git") == "git@github.com:OWNER/REPO.git"
    assert trusted_url("OWNER/REPO", "https://github.com/OWNER/REPO.git") == "https://github.com/OWNER/REPO.git"


@pytest.mark.parametrize("observed", [
    "https://github.com/OWNER/REPO.git",
    "git@github.com:OWNER/REPO.git",
    "ssh://git@github.com/OWNER/REPO.git",
    "https://evil.example/OWNER/REPO.git",
])
def test_existing_local_checkout_rejects_every_other_origin_before_fetch(
    repositories, monkeypatch, tmp_path, observed,
):
    origin, _, checkout = repositories
    marker = tmp_path / "transport-started"
    git(checkout, "config", "remote.origin.url", observed)
    from coding_review_agent_loop import git_transport

    monkeypatch.setattr(git_transport, "_private_git", lambda *a, **k: marker.touch())
    with pytest.raises(AgentLoopError, match="operator-trusted endpoint"):
        import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main",
                   repo="OWNER/REPO", runner=Runner(), local_origin=origin)
    assert not marker.exists()


@pytest.mark.parametrize("protocol,expected", [
    ("https", "https://github.com/OWNER/REPO.git"),
    ("ssh", "git@github.com:OWNER/REPO.git"),
])
def test_fresh_checkout_origin_uses_operator_protocol(protocol, expected):
    assert default_origin("OWNER/REPO", protocol=protocol) == expected


def test_windows_ssh_command_quotes_pinned_path(monkeypatch):
    from coding_review_agent_loop import git_transport

    monkeypatch.setattr(git_transport.sys, "platform", "win32")
    monkeypatch.setattr(git_transport.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(git_transport, "_git_path", lambda: r"C:\Program Files\Git\bin\git.exe")
    monkeypatch.setattr(git_transport, "_trusted_executable", lambda path: path)
    command = git_transport._transport_env("git@github.com:OWNER/REPO.git", None)["GIT_SSH_COMMAND"]
    assert command.startswith('"C:\\Program Files\\Git\\usr\\bin\\ssh.exe" -F ')
    assert "ProxyCommand=none" in command


def test_fresh_default_checkout_initializes_pinned_ssh_origin(tmp_path, monkeypatch):
    from agent_loop_helpers import make_config
    from coding_review_agent_loop import config as config_module

    checkout = tmp_path / "fresh"
    config = make_config(tmp_path, create_dirs=False, claude_dir=checkout,
                         trusted_origin_protocol="ssh")
    monkeypatch.setattr(config_module, "_sync_base_branch", lambda *_args, **_kwargs: None)
    config_module.ensure_temp_checkout(checkout, agent="claude", config=config, runner=Runner())
    assert git(checkout, "config", "--local", "--get", "remote.origin.url") == (
        "git@github.com:OWNER/REPO.git"
    )


def test_private_https_retry_scopes_token_without_exposing_it(repositories, monkeypatch):
    from coding_review_agent_loop import git_transport

    _, _, checkout = repositories
    git(checkout, "config", "remote.origin.url", "https://github.com/OWNER/REPO.git")
    secret = "private-token-example"
    calls = []

    def fake_private(args, *, cwd, env, **kwargs):
        calls.append((args, dict(env)))
        if args and args[0] == "fetch" and "GIT_CONFIG_COUNT" not in env:
            raise AgentLoopError("public fetch failed")
        if args and args[0] == "fetch":
            raise AgentLoopError("authenticated fetch failed")
        return b""

    monkeypatch.setattr(git_transport, "_private_git", fake_private)
    monkeypatch.setattr(git_transport, "_token_from_gh", lambda _gh, _dest, _host: secret)
    with pytest.raises(AgentLoopError, match="authenticated fetch failed") as raised:
        import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main",
                   repo="OWNER/REPO", runner=Runner(), gh_cmd="gh")
    assert secret not in str(raised.value)
    assert all(secret not in " ".join(args) for args, _ in calls)
    authorized = [env for args, env in calls if args and args[0] == "fetch" and "GIT_CONFIG_COUNT" in env]
    assert len(authorized) == 1
    assert authorized[0]["GIT_CONFIG_KEY_0"] == "http.https://github.com/OWNER/REPO.git.extraheader"
    assert base64.b64decode(authorized[0]["GIT_CONFIG_VALUE_0"].split()[-1]) == (
        f"x-access-token:{secret}".encode()
    )


def test_token_cli_cannot_be_planted_in_checkout(repositories, tmp_path, monkeypatch):
    from coding_review_agent_loop.git_transport import _token_from_gh

    _, _, checkout = repositories
    marker = tmp_path / "planted-gh-ran"
    planted = checkout / "gh"
    planted.write_text(f"#!/bin/sh\ntouch {marker}\necho token\n")
    planted.chmod(0o755)
    with pytest.raises(AgentLoopError, match="agent-inaccessible installation"):
        _token_from_gh(str(planted), checkout, "github.com")
    assert not marker.exists()
    trusted = tmp_path / "trusted-gh"
    argv = tmp_path / "trusted-argv"
    trusted.write_text(f"#!/bin/sh\nprintf '%s ' \"$@\" > {argv}\necho trusted-token\n")
    trusted.chmod(0o755)
    with pytest.raises(AgentLoopError, match="agent-inaccessible installation"):
        _token_from_gh(str(trusted), checkout, "github.com")
    assert not argv.exists()


def test_token_cli_rejects_sibling_checkout_on_path(repositories, tmp_path, monkeypatch):
    from coding_review_agent_loop.git_transport import _token_from_gh

    _, _, checkout = repositories
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    marker = tmp_path / "sibling-gh-ran"
    planted = sibling / "gh"
    planted.write_text(f"#!/bin/sh\ntouch {marker}\necho planted-token\n")
    planted.chmod(0o755)
    monkeypatch.setenv("PATH", f"{sibling}:{os.environ.get('PATH', '')}")
    with pytest.raises(AgentLoopError, match="agent-inaccessible installation"):
        _token_from_gh(str(planted), checkout, "github.com")
    assert not marker.exists()


@pytest.mark.parametrize("target", ["git", "gh"])
def test_same_user_private_installation_is_not_trusted(repositories, tmp_path, monkeypatch, target):
    """A 0700 sibling can still be replaced by an unrestricted same-user agent."""
    from coding_review_agent_loop.git_transport import _token_from_gh

    _, _, checkout = repositories
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    planted = private / target
    marker = tmp_path / "executed"
    planted.write_text(f"#!/bin/sh\ntouch {marker}\necho token\n")
    planted.chmod(0o700)
    if target == "gh":
        with pytest.raises(AgentLoopError, match="agent-inaccessible installation"):
            _token_from_gh(str(planted), checkout, "github.com")
    else:
        monkeypatch.setattr(secure_git, "_PINNED_GIT", None)
        monkeypatch.setattr(secure_git.sys, "platform", "darwin")
        real_isfile = secure_git.os.path.isfile
        monkeypatch.setattr(secure_git.os.path, "isfile", lambda name: name == "/opt/homebrew/bin/git" or real_isfile(name))
        real_realpath = secure_git.os.path.realpath
        monkeypatch.setattr(secure_git.os.path, "realpath", lambda name: str(planted) if name == "/opt/homebrew/bin/git" else real_realpath(name))
        with pytest.raises(AgentLoopError, match="trusted installation"):
            secure_git._git_path()
    assert not marker.exists()


@pytest.mark.parametrize("helper", ["git_exec_guard.c", "git_exec_guard_macos.c", "git_windows_launcher.py"])
def test_modified_guard_helper_is_rejected_before_launch(tmp_path, monkeypatch, helper):
    package = tmp_path / "package"
    package.mkdir(mode=0o700)
    module = package / "secure_git.py"
    module.write_text("# simulated installed module\n")
    planted = package / helper
    planted.write_text("# planted helper\n")
    monkeypatch.setattr(secure_git, "__file__", str(module))
    with pytest.raises(AgentLoopError, match="helper identity changed"):
        secure_git._trusted_helper(helper)


@pytest.mark.skipif(os.name != "posix" or not hasattr(os, "memfd_create"),
                    reason="Linux anonymous guard output is unavailable")
def test_guard_compiler_output_has_no_replaceable_path(monkeypatch):
    import fcntl

    monkeypatch.setattr(secure_git, "_GUARD_FD", None)
    original_run = secure_git.subprocess.run
    output_paths = []

    def checked_run(args, *positional, **kwargs):
        if "-o" in args:
            output = args[args.index("-o") + 1]
            output_paths.append(output)
            assert output.startswith("/proc/self/fd/")
            assert kwargs["pass_fds"] == (int(output.rsplit("/", 1)[1]),)
        return original_run(args, *positional, **kwargs)

    monkeypatch.setattr(secure_git.subprocess, "run", checked_run)
    fd = secure_git._guard_fd()
    try:
        assert output_paths
        seals = fcntl.fcntl(fd, fcntl.F_GET_SEALS)
        assert seals & fcntl.F_SEAL_WRITE
    finally:
        os.close(fd)
        monkeypatch.setattr(secure_git, "_GUARD_FD", None)


def test_filter_changed_at_launch_cannot_execute(repositories, monkeypatch, tmp_path):
    _, seed, checkout = repositories
    marker = tmp_path / "filter-planted"
    (seed / ".gitattributes").write_text("a.txt filter=planted\n")
    git(seed, "add", ".gitattributes")
    git(seed, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "attributes")
    git(checkout, "-c", "protocol.file.allow=always", "fetch", "-q", str(seed), "main")
    sha = git(checkout, "rev-parse", "FETCH_HEAD")
    runner = Runner()
    runner.run(("git", "checkout", "--detach", sha), cwd=checkout)
    (checkout / "a.txt").unlink()
    original = secure_git.local_command

    def mutate(args, *, environ=None, checkout=None):
        if args and args[0] == "reset":
            git(checkout, "config", "filter.planted.smudge", f"touch {marker}")
            git(checkout, "config", "filter.planted.required", "true")
        return original(args, environ=environ, checkout=checkout)

    monkeypatch.setattr(secure_git, "local_command", mutate)
    result = runner.run(("git", "reset", "--hard", sha), cwd=checkout, check=False)
    assert result.returncode != 0
    assert not marker.exists()


def test_named_diff_trace_and_submodule_controls(repositories, monkeypatch, tmp_path):
    _, _, checkout = repositories
    marker = tmp_path / "diff-planted"
    trace = tmp_path / "trace-planted"
    (checkout / ".gitattributes").write_text("a.txt diff=planted\n")
    git(checkout, "add", ".gitattributes")
    git(checkout, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "attributes")
    git(checkout, "config", "diff.planted.command", f"touch {marker}")
    git(checkout, "config", "diff.planted.textconv", f"touch {marker}")
    (checkout / "a.txt").write_text("two\n")
    monkeypatch.setenv("GIT_TRACE", str(trace))
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", f"touch {marker}")
    runner = Runner()
    diff = runner.run(("git", "diff", "--", "a.txt"), cwd=checkout)
    status = runner.run(("git", "status", "--porcelain"), cwd=checkout)
    assert "two" in diff.stdout and "a.txt" in status.stdout
    assert not marker.exists() and not trace.exists()


def test_pr_anchor_mapping_disables_named_diff_after_global_git_options(repositories, tmp_path):
    from coding_review_agent_loop.pr_loop_support import _git_anchor_mapper

    _, _, checkout = repositories
    before = git(checkout, "rev-parse", "HEAD")
    (checkout / ".gitattributes").write_text("a.txt diff=planted\n")
    (checkout / "a.txt").write_text("two\n")
    git(checkout, "add", ".gitattributes", "a.txt")
    git(checkout, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "two")
    after = git(checkout, "rev-parse", "HEAD")
    marker = tmp_path / "anchor-diff-planted"
    git(checkout, "config", "diff.planted.command", f"touch {marker}")
    mapper = _git_anchor_mapper(Runner(), SimpleNamespace(quiet=True), checkout=checkout, window=3)
    result = mapper(before, after, "a.txt", 1, 1)
    assert result.mappable
    assert not marker.exists()


def test_populated_submodule_config_is_not_traversed(repositories, tmp_path):
    _, _, checkout = repositories
    sub = tmp_path / "sub-origin"
    subprocess.run(("git", "init", "-q", "-b", "main", str(sub)), check=True)
    (sub / "sub.txt").write_text("one\n")
    git(sub, "add", "sub.txt")
    git(sub, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "sub")
    git(checkout, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "sub")
    git(checkout, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qam", "submodule")
    marker = tmp_path / "submodule-fsmonitor"
    git(checkout / "sub", "config", "core.fsmonitor", f"touch {marker}")
    runner = Runner()
    assert runner.run(("git", "status", "--porcelain"), cwd=checkout).stdout == ""
    assert runner.run(("git", "diff", "HEAD^", "HEAD"), cwd=checkout).returncode == 0
    assert not marker.exists()


def test_binary_and_cancellable_snapshot_share_guard(repositories):
    _, _, checkout = repositories
    data = _run_git(checkout, ("rev-parse", "HEAD"))
    assert len(data.strip()) == 40
    cancel = Event()
    cancel.set()
    with pytest.raises(SnapshotCancelled):
        _run_git(checkout, ("rev-parse", "HEAD"), cancel=cancel)


def test_unsupported_backend_refuses_first_checkout_probe(repositories, monkeypatch):
    _, _, checkout = repositories
    monkeypatch.setattr(secure_git, "_GUARD_FD", None)
    monkeypatch.setattr(secure_git.sys, "platform", "unsupported-test")
    with pytest.raises(AgentLoopError, match="Unsupported Git confinement"):
        Runner().run(("git", "status", "--porcelain"), cwd=checkout)
