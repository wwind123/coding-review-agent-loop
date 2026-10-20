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


def _commit(repo: Path, message: str, files: dict[str, str | None]) -> str:
    for name, text in files.items():
        if text is None:
            git(repo, "rm", "-q", name)
        else:
            (repo / name).write_text(text)
            git(repo, "add", name)
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


def _packs(checkout: Path) -> set[Path]:
    return set((checkout / ".git" / "objects" / "pack").glob("*.pack"))


def _pack_objects(pack: Path) -> set[str]:
    listed = git(pack.parent, "verify-pack", "-v", str(pack.with_suffix(".idx")))
    return {line.split()[0] for line in listed.splitlines()
            if line and line.split()[1:2] and line.split()[1] in {"commit", "tree", "blob", "tag"}}


def _spy_streams(monkeypatch, fail_thin: bool = False, fail_full: bool = False,
                 failure: str = "return") -> list:
    from coding_review_agent_loop import git_transport

    real = git_transport._stream_pack
    calls = []

    def spy(private, env, destination, revs, *, thin):
        calls.append((thin, revs))
        if (fail_thin and thin) or (fail_full and not thin):
            if failure == "timeout":
                raise subprocess.TimeoutExpired(("index-pack",), 90)
            if failure == "oserror":
                raise BrokenPipeError("pack stream closed")
            return False
        return real(private, env, destination, revs, thin=thin)

    monkeypatch.setattr(git_transport, "_stream_pack", spy)
    return calls


def _import_main(checkout: Path, origin: Path, runner: Runner | None = None) -> str:
    return import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main",
                      repo="OWNER/REPO", runner=runner or Runner(), local_origin=origin)


def _closure(checkout: Path, sha: str) -> bool:
    """Walk sha's full closure, ignoring shallow, graft and replace state."""
    absent = str(checkout.parent / "absent-history-file")
    env = {**os.environ, "GIT_SHALLOW_FILE": absent, "GIT_GRAFT_FILE": absent,
           "GIT_NO_REPLACE_OBJECTS": "1"}
    return subprocess.run(("git", "-C", str(checkout), "rev-list", "--objects",
                           "--missing=error", "--quiet", sha),
                          capture_output=True, env=env, check=False).returncode == 0


def test_unchanged_complete_ref_reimport_writes_no_pack(repositories, monkeypatch):
    from coding_review_agent_loop import git_transport

    origin, _, checkout = repositories
    first = _import_main(checkout, origin)
    calls = _spy_streams(monkeypatch)
    checked = []
    real_closure = git_transport._closure_complete
    monkeypatch.setattr(git_transport, "_closure_complete",
                        lambda *a: checked.append(a[1]) or real_closure(*a))
    before = _packs(checkout)
    second = _import_main(checkout, origin)
    assert second == first == git(checkout, "rev-parse", "refs/remotes/origin/main")
    assert _packs(checkout) == before
    assert calls == [] and checked == [first]


def test_new_upstream_commit_imports_thin_incremental_pack(repositories, monkeypatch):
    origin, seed, checkout = repositories
    lines = [f"line {index} of a delta-friendly blob with some padding text\n"
             for index in range(3000)]
    _commit(seed, "big", {"big.txt": "".join(lines)})
    git(seed, "push", "-q", str(origin), "main")
    previous = _import_main(checkout, origin)
    lines[1500] = "changed line\n"
    new = _commit(seed, "change", {"big.txt": "".join(lines)})
    git(seed, "push", "-q", str(origin), "main")
    calls = _spy_streams(monkeypatch)
    before = _packs(checkout)
    assert _import_main(checkout, origin) == new
    assert git(checkout, "rev-parse", "refs/remotes/origin/main") == new
    assert [thin for thin, _ in calls] == [True]
    assert f"^{previous}" in calls[0][1]
    added = _packs(checkout) - before
    assert len(added) == 1
    contents = _pack_objects(added.pop())
    new_objects = {line.split()[0] for line in
                   git(checkout, "rev-list", "--objects", new, "--not", previous).splitlines()}
    full_history = git(checkout, "rev-list", "--objects", new).splitlines()
    assert new_objects <= contents
    assert len(contents) < len(full_history)
    assert _closure(checkout, new)


def _loose_history_with_lost_blob(origin: Path, checkout: Path) -> tuple[str, str]:
    """Build A0 -> A in the destination with loose objects and drop A0's only blob."""
    _commit(checkout, "A0", {"only.txt": "only in A0\n"})
    blob = git(checkout, "rev-parse", "HEAD:only.txt")
    tip = _commit(checkout, "A", {"only.txt": None, "b.txt": "b\n"})
    git(checkout, "push", "-q", "origin", "main")
    assert git(checkout, "rev-parse", "refs/remotes/origin/main") == tip
    (checkout / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    assert not _closure(checkout, tip)
    return tip, blob


def test_missing_object_below_excluded_tip_is_repaired_by_full_export(
        repositories, monkeypatch, tmp_path):
    origin, _, checkout = repositories
    tip, blob = _loose_history_with_lost_blob(origin, checkout)
    other = tmp_path / "other"
    subprocess.run(("git", "clone", "-q", str(origin), str(other)), check=True)
    descendant = _commit(other, "B", {"c.txt": "c\n"})
    git(other, "push", "-q", "origin", "main")
    calls = _spy_streams(monkeypatch)
    assert _import_main(checkout, origin) == descendant
    assert [thin for thin, _ in calls] == [True, False]
    assert calls[0][1] == (descendant, f"^{tip}")
    assert git(checkout, "cat-file", "-t", blob) == "blob"
    assert _closure(checkout, descendant)
    assert git(checkout, "rev-parse", "refs/remotes/origin/main") == descendant


def test_missing_object_below_unchanged_tip_is_repaired(repositories, monkeypatch):
    origin, _, checkout = repositories
    tip, blob = _loose_history_with_lost_blob(origin, checkout)
    calls = _spy_streams(monkeypatch)
    assert _import_main(checkout, origin) == tip
    assert calls == [(False, (tip,))]
    assert git(checkout, "cat-file", "-t", blob) == "blob"
    assert _closure(checkout, tip)
    assert git(checkout, "rev-parse", "refs/remotes/origin/main") == tip


def _mask_history(checkout: Path, tip: str, mask: str) -> None:
    """Make tip look like a root commit through destination-side history state."""
    if mask == "shallow":
        (checkout / ".git" / "shallow").write_text(f"{tip}\n")
    elif mask == "graft":
        (checkout / ".git" / "info").mkdir(exist_ok=True)
        (checkout / ".git" / "info" / "grafts").write_text(f"{tip}\n")
    else:
        git(checkout, "-c", "user.name=Test", "-c", "user.email=test@example.com",
            "replace", "--graft", tip)


@pytest.mark.parametrize("advance", [False, True], ids=["unchanged-tip", "advancing-tip"])
@pytest.mark.parametrize("mask", ["shallow", "graft", "replace"])
def test_closure_gate_ignores_destination_history_substitution(
        repositories, monkeypatch, tmp_path, mask, advance):
    origin, _, checkout = repositories
    tip, blob = _loose_history_with_lost_blob(origin, checkout)
    _mask_history(checkout, tip, mask)
    # The destination's own walk now stops at tip and misses the lost blob.
    assert subprocess.run(("git", "-C", str(checkout), "rev-list", "--objects",
                           "--missing=error", "--quiet", tip),
                          capture_output=True, check=False).returncode == 0
    expected = tip
    if advance:
        other = tmp_path / "other"
        subprocess.run(("git", "clone", "-q", str(origin), str(other)), check=True)
        expected = _commit(other, "B", {"c.txt": "c\n"})
        git(other, "push", "-q", "origin", "main")
    calls = _spy_streams(monkeypatch)
    assert _import_main(checkout, origin) == expected
    assert [thin for thin, _ in calls] == ([True, False] if advance else [False])
    assert calls[-1][1] == (expected,)
    assert git(checkout, "cat-file", "-t", blob) == "blob"
    assert _closure(checkout, expected)
    assert git(checkout, "rev-parse", "refs/remotes/origin/main") == expected


def test_local_only_and_non_commit_tips_are_never_negative_revs(repositories, monkeypatch):
    origin, seed, checkout = repositories
    base = git(checkout, "rev-parse", "HEAD")
    git(checkout, "checkout", "-q", "-b", "local")
    local = _commit(checkout, "local only", {"local.txt": "local\n"})
    git(checkout, "-c", "user.name=Test", "-c", "user.email=test@example.com",
        "tag", "-a", "-m", "annotated", "annotated", base)
    tag = git(checkout, "rev-parse", "refs/tags/annotated")
    tree = git(checkout, "rev-parse", "HEAD^{tree}")
    git(checkout, "update-ref", "refs/misc/tree", tree)
    new = _commit(seed, "upstream", {"a.txt": "two\n"})
    git(seed, "push", "-q", str(origin), "main")
    calls = _spy_streams(monkeypatch)
    assert _import_main(checkout, origin) == new
    assert calls == [(True, (new, f"^{base}"))]
    assert not {f"^{local}", f"^{tag}", f"^{tree}"} & set(calls[0][1])
    assert _closure(checkout, new)


def test_incremental_failure_falls_back_once_to_full_export(repositories, monkeypatch):
    origin, seed, checkout = repositories
    new = _commit(seed, "upstream", {"a.txt": "two\n"})
    git(seed, "push", "-q", str(origin), "main")
    calls = _spy_streams(monkeypatch, fail_thin=True)
    assert _import_main(checkout, origin) == new
    assert [thin for thin, _ in calls] == [True, False]
    assert calls[1][1] == (new,)
    assert _closure(checkout, new)
    assert git(checkout, "rev-parse", "refs/remotes/origin/main") == new


@pytest.mark.parametrize("failure", ["return", "timeout", "oserror"])
def test_failed_full_export_raises_fixed_error_without_moving_ref(
        repositories, monkeypatch, failure):
    origin, seed, checkout = repositories
    previous = git(checkout, "rev-parse", "refs/remotes/origin/main")
    _commit(seed, "upstream", {"a.txt": "two\n"})
    git(seed, "push", "-q", str(origin), "main")
    calls = _spy_streams(monkeypatch, fail_thin=True, fail_full=True, failure=failure)
    with pytest.raises(AgentLoopError) as caught:
        _import_main(checkout, origin)
    assert str(caught.value) == "Verified object-pack import failed."
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert [thin for thin, _ in calls] == [True, False]
    assert git(checkout, "rev-parse", "refs/remotes/origin/main") == previous


def test_private_import_rejects_tracking_ref_race_on_pack_path(repositories, monkeypatch):
    origin, seed, checkout = repositories
    tracking = "refs/remotes/origin/main"
    _commit(seed, "upstream", {"a.txt": "two\n"})
    git(seed, "push", "-q", str(origin), "main")
    calls = _spy_streams(monkeypatch)

    class MoveRef(Runner):
        def run(self, args, *, cwd, **kwargs):
            if tuple(args[:3]) == ("git", "update-ref", tracking):
                git(checkout, "update-ref", "-d", tracking)
            return super().run(args, cwd=cwd, **kwargs)

    with pytest.raises(AgentLoopError, match="Command failed"):
        _import_main(checkout, origin, MoveRef())
    assert [thin for thin, _ in calls] == [True]
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
        monkeypatch.setattr(secure_git.os.path, "isfile", lambda name: name == "/opt/homebrew/bin/git")
        real_realpath = secure_git.os.path.realpath
        monkeypatch.setattr(secure_git.os.path, "realpath", lambda name: str(planted) if name == "/opt/homebrew/bin/git" else real_realpath(name))
        with pytest.raises(AgentLoopError, match="trusted installation"):
            secure_git._git_path()
    assert not marker.exists()


@pytest.mark.parametrize("target", ["git", "gh"])
@pytest.mark.parametrize("later_trusted", [True, False])
def test_untrusted_earlier_candidate_falls_through_to_trusted_later_one(
    repositories, tmp_path, monkeypatch, target, later_trusted
):
    """A user-owned Homebrew prefix must not hide a root-owned later install."""
    from coding_review_agent_loop import git_transport

    _, _, checkout = repositories
    # Each list's earlier entry is untrusted and a later one is root-owned.
    untrusted, trusted = (("/opt/homebrew/bin/git", "/usr/local/bin/git") if target == "git"
                          else ("/usr/bin/gh", "/usr/local/bin/gh"))
    present = {untrusted, trusted}
    trusted_target = str(tmp_path / ("trusted-" + target))
    calls: list[str] = []

    def fake_trusted(path):
        calls.append(path)
        if path == trusted and later_trusted:
            return trusted_target
        raise AgentLoopError("untrusted installation")

    monkeypatch.setattr(secure_git.sys, "platform", "darwin")
    monkeypatch.setattr(secure_git.os.path, "isfile", lambda name: name in present)
    monkeypatch.setattr(secure_git, "_trusted_executable", fake_trusted)
    if target == "git":
        monkeypatch.setattr(secure_git, "_PINNED_GIT", None)
        monkeypatch.setattr(secure_git, "_PINNED_GIT_HASH", None)
        monkeypatch.setattr(secure_git.Path, "read_bytes", lambda self: b"git")
        if later_trusted:
            assert secure_git._git_path() == trusted_target
        else:
            with pytest.raises(AgentLoopError, match="trusted installation"):
                secure_git._git_path()
    else:
        monkeypatch.setattr(git_transport.sys, "platform", "darwin")
        if later_trusted:
            def fake_run(argv, **kwargs):
                assert argv[0] == trusted_target
                return subprocess.CompletedProcess(argv, 0, b"token\n", b"")
            monkeypatch.setattr(git_transport.subprocess, "run", fake_run)
            assert git_transport._token_from_gh("gh", checkout, "github.com") == "token"
        else:
            with pytest.raises(AgentLoopError, match="agent-inaccessible installation"):
                git_transport._token_from_gh("gh", checkout, "github.com")
    assert calls[:2] == [untrusted, trusted]


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


@pytest.mark.parametrize("program", ["git", "/usr/bin/git", r"C:\Program Files\Git\bin\git.exe", "GIT.EXE"])
@pytest.mark.parametrize("path", ["text", "binary"])
def test_runner_routes_every_git_spelling_through_local_boundary(tmp_path, monkeypatch, program, path):
    """A Git launch by absolute path or git.exe must not bypass the boundary."""
    from coding_review_agent_loop import runner as runner_module
    from coding_review_agent_loop.runner import Runner, run_binary_capture

    routed: list[tuple[str, ...]] = []

    def fake_local_command(args, *, environ=None, checkout=None):
        routed.append(tuple(args))
        return ["true"], dict(environ or {}), ()

    monkeypatch.setattr(secure_git, "local_command", fake_local_command)
    if path == "text":
        Runner().run([program, "status", "--porcelain"], cwd=tmp_path)
    else:
        run_binary_capture([program, "status", "--porcelain"], cwd=tmp_path)
    assert routed == [("status", "--porcelain")]
    assert runner_module._is_git_launch(["gitk"]) is False


# --- SSH identity selection from ~/.ssh/config (#1380) -------------------------

import re as _re
import shlex as _shlex
import threading as _threading

_SSH_PREFIX = [
    "/usr/bin/ssh", "-F", os.devnull, "-o", "ProxyCommand=none", "-o", "PermitLocalCommand=no",
    "-o", "ClearAllForwardings=yes", "-o", "StrictHostKeyChecking=yes",
]
_SAFE = _re.compile(r"[A-Za-z0-9._/+,:@-]+")


@pytest.fixture
def ssh_home(tmp_path, monkeypatch):
    """A trusted 0700 HOME with ~/.ssh; skip where tmp ancestry is not operator-trusted."""
    from coding_review_agent_loop import git_transport

    if os.name != "posix" or not Path("/usr/bin/ssh").exists():
        pytest.skip("POSIX OpenSSH client is unavailable")
    try:
        git_transport._trusted_executable("/usr/bin/ssh")
    except AgentLoopError:
        pytest.skip("/usr/bin/ssh is not in a root-owned installation")
    tmp_path.chmod(0o700)
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    home.chmod(0o700)
    (home / ".ssh").chmod(0o700)
    probe = home / ".ssh" / "probe"
    probe.write_text("")
    probe.chmod(0o600)
    try:
        git_transport._trusted_operator_file(probe, forbidden_roots=())
    except git_transport._UntrustedPath:
        pytest.skip("tmp_path ancestry is not operator-owned and private")
    probe.unlink()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.setattr(git_transport, "_EMITTED_WARNINGS", set())
    return home


def _write_ssh_config(home: Path, text: str) -> Path:
    path = home / ".ssh" / "config"
    path.write_text(text)
    path.chmod(0o600)
    return path


def _key(directory: Path, name: str = "k") -> Path:
    if not directory.exists():
        directory.mkdir(parents=True)
        directory.chmod(0o700)
    path = directory / name
    path.write_text("placeholder key material is never read\n")
    path.chmod(0o600)
    return path


def _ssh_argv(url: str = "git@github.com:OWNER/REPO.git", **kwargs) -> list[str]:
    from coding_review_agent_loop import git_transport

    return _shlex.split(git_transport._transport_env(url, None, **kwargs)["GIT_SSH_COMMAND"])


def _option_values(argv: list[str], flag: str) -> list[str]:
    return [argv[index + 1] for index, token in enumerate(argv[:-1]) if token == flag]


def _no_process(monkeypatch):
    from coding_review_agent_loop import git_transport

    def refuse(*_args, **_kwargs):
        raise AssertionError("SSH option resolution must not run a process (no ssh -G)")

    monkeypatch.setattr(git_transport.subprocess, "run", refuse)
    monkeypatch.setattr(git_transport.subprocess, "Popen", refuse)


def test_config_identity_file_is_carried_after_hardening_prefix(ssh_home, monkeypatch):
    key = _key(ssh_home / ".ssh")
    _write_ssh_config(ssh_home, "Host github.com\n  IdentityFile ~/.ssh/k\n")
    _no_process(monkeypatch)
    argv = _ssh_argv()
    assert argv[:len(_SSH_PREFIX)] == _SSH_PREFIX
    assert argv[len(_SSH_PREFIX):] == ["-i", str(key)]


def test_dangerous_ssh_directives_are_ignored_and_include_is_not_opened(ssh_home, tmp_path, monkeypatch):
    from coding_review_agent_loop import git_transport

    key = _key(ssh_home / ".ssh")
    _key(ssh_home / ".ssh", "evil")
    marker = tmp_path / "marker"
    included = ssh_home / ".ssh" / "other"
    included.write_text("Host *\n  IdentityFile ~/.ssh/evil\n")
    included.chmod(0o600)
    _write_ssh_config(ssh_home, (
        "Include other\n"
        "Host github.com\n"
        f"  ProxyCommand touch {marker}\n"
        "  ProxyJump jump.example.com\n"
        f"  LocalCommand touch {marker}\n"
        f"  KnownHostsCommand touch {marker}\n"
        "  User operator\n"
        "  IdentityFile ~/.ssh/k\n"
        f'Match exec "touch {marker}"\n'
        "  IdentityFile ~/.ssh/evil\n"
        "  HostName evil.example.com\n"
    ))
    opened: list[str] = []
    real_open = os.open

    def recording_open(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(git_transport.os, "open", recording_open)
    _no_process(monkeypatch)
    argv = _ssh_argv()
    assert argv[len(_SSH_PREFIX):] == ["-i", str(key)]
    assert opened == [str(ssh_home / ".ssh" / "config")]
    assert not marker.exists()
    assert "evil" not in " ".join(argv) and "ProxyJump" not in " ".join(argv)


def test_ssh_host_pattern_semantics_first_value_and_negation():
    from coding_review_agent_loop.git_transport import _ssh_host_options

    text = (
        "IdentityFile ~/.ssh/global\n"
        "Host *.example.com !bad.example.com\n"
        "  HostName wild.example.net\n"
        "  IdentityFile ~/.ssh/wild\n"
        "Host other.com\n"
        "  IdentityFile ~/.ssh/other\n"
        "  Port 1111\n"
        "Host GITHUB.COM gitlab.com\n"
        "  HostName=first.example.net\n"
        '  IdentityFile "~/.ssh/k"  # trailing comment\n'
        "Host *\n"
        "  HostName second.example.net\n"
        "  IdentityFile ~/.ssh/k2\n"
        "  Port 2200\n"
        "Host g?thub.com\n"
        "  IdentitiesOnly yes\n"
    )
    github = _ssh_host_options(text, "github.com")
    assert github["hostname"] == ("first.example.net",)
    assert github["port"] == ("2200",)
    assert github["identitiesonly"] == ("yes",)
    assert github["identityfile"] == [("~/.ssh/global",), ("~/.ssh/k",), ("~/.ssh/k2",)]
    good = _ssh_host_options(text, "good.example.com")
    assert good["hostname"] == ("wild.example.net",)
    bad = _ssh_host_options(text, "bad.example.com")
    assert bad["hostname"] == ("second.example.net",)
    assert ("~/.ssh/wild",) not in bad["identityfile"]
    assert all(("~/.ssh/other",) != entry for entry in github["identityfile"])


@pytest.mark.parametrize("line,filename", [
    ('IdentityFile "~/.ssh/a b"', "a b"),
    ("IdentityFile '~/.ssh/q'", "q"),
    ("IdentityFile ~/.ssh/a\\b", "a\\b"),
    ("IdentityFile ~/.ssh/$(touch x)", "$(touch x)"),
    ("IdentityFile ~/.ssh/`x`", "`x`"),
    ("IdentityFile ~/.ssh/a=b", "a=b"),
    ("IdentityFile ~/.ssh/a#b", "a#b"),
    ("IdentityFile ~/.ssh/%h_%r", "%h_%r"),
    ("IdentityFile ${HOME}/.ssh/k", "k"),
    ("IdentityFile -oProxyCommand=x", "k"),
    ("IdentityFile ~operator/.ssh/k", "k"),
    ("IdentityFile .ssh/k", "k"),
])
def test_unsafe_identity_values_are_refused_not_escaped(ssh_home, monkeypatch, capsys, line, filename):
    _key(ssh_home / ".ssh", filename)
    _key(ssh_home / ".ssh")
    _write_ssh_config(ssh_home, f"Host github.com\n  {line}\n")
    argv = _ssh_argv()
    assert argv[len(_SSH_PREFIX):] == []
    assert "IdentityFile for trusted Git transport" in capsys.readouterr().err


@pytest.mark.parametrize("line,key", [
    ("Port 99999", "Port"), ("Port 22x", "Port"), ("Port -1", "Port"),
    ("HostName -oProxyCommand=x", "HostName"), ("HostName %h.example.com", "HostName"),
    ("HostName a;b", "HostName"), ("IdentitiesOnly maybe", "IdentitiesOnly"),
    ("UserKnownHostsFile ~/.ssh/a ~/.ssh/b", "UserKnownHostsFile"),
    ("UserKnownHostsFile ~/.ssh/%k", "UserKnownHostsFile"),
    ('HostName "unterminated', "HostName"),
])
def test_unsafe_scalar_values_are_skipped_with_fixed_warning(ssh_home, capsys, line, key):
    _write_ssh_config(ssh_home, f"Host github.com\n  {line}\n")
    argv = _ssh_argv()
    assert argv[len(_SSH_PREFIX):] == []
    err = capsys.readouterr().err
    assert f"{key} for trusted Git transport" in err
    assert line.split(None, 1)[1] not in err


@pytest.mark.parametrize("home_name", ["h%h", "h${X}", "h space"])
def test_unsafe_home_or_realpath_refuses_clean_raw_values(tmp_path, monkeypatch, capsys, ssh_home, home_name):
    home = tmp_path / home_name
    (home / ".ssh").mkdir(parents=True)
    home.chmod(0o700)
    (home / ".ssh").chmod(0o700)
    _key(home / ".ssh")
    _write_ssh_config(home, "Host github.com\n  IdentityFile ~/.ssh/k\n"
                            "  UserKnownHostsFile ~/.ssh/known_hosts\n")
    monkeypatch.setenv("HOME", str(home))
    argv = _ssh_argv()
    assert argv[len(_SSH_PREFIX):] == []
    assert home_name not in " ".join(argv)
    assert "IdentityFile for trusted Git transport" in capsys.readouterr().err


def test_symlinked_identity_with_unsafe_realpath_is_refused(ssh_home, capsys):
    real = _key(ssh_home / ".ssh", "real%k")
    (ssh_home / ".ssh" / "k").symlink_to(real)
    _write_ssh_config(ssh_home, "Host github.com\n  IdentityFile ~/.ssh/k\n")
    assert _ssh_argv()[len(_SSH_PREFIX):] == []


def test_emitted_config_values_are_allowlisted(ssh_home):
    key = _key(ssh_home / ".ssh", "github_key")
    _write_ssh_config(ssh_home, (
        "Host github.com\n  HostName ssh.github.com\n  Port 443\n  IdentitiesOnly YES\n"
        "  UserKnownHostsFile ~/.ssh/known_hosts_github\n  IdentityFile ~/.ssh/github_key\n"
    ))
    argv = _ssh_argv()
    assert argv[len(_SSH_PREFIX):] == [
        "-o", "HostName=ssh.github.com", "-o", "Port=443", "-o", "IdentitiesOnly=yes",
        "-o", f"UserKnownHostsFile={ssh_home / '.ssh' / 'known_hosts_github'}", "-i", str(key),
    ]
    for token in argv[len(_SSH_PREFIX):]:
        if token not in ("-o", "-i"):
            assert _SAFE.fullmatch(token.split("=", 1)[-1])


def test_config_user_is_ignored_and_port_only_without_url_port(ssh_home):
    _write_ssh_config(ssh_home, "Host github.com\n  User operator\n  Port 2200\n")
    scp = _ssh_argv("git@github.com:OWNER/REPO.git")
    assert scp[len(_SSH_PREFIX):] == ["-o", "Port=2200"]
    explicit = _ssh_argv("ssh://git@github.com:2222/OWNER/REPO.git")
    assert explicit[len(_SSH_PREFIX):] == []
    for argv in (scp, explicit):
        assert not any(token.lower().startswith("user=") for token in argv)
        assert "-l" not in argv


def test_explicit_identity_overrides_config_identity_and_identities_only(ssh_home):
    config_key = _key(ssh_home / ".ssh", "a")
    explicit = _key(ssh_home / "keys", "c")
    (ssh_home / "keys").chmod(0o700)
    _write_ssh_config(ssh_home, "Host github.com\n  IdentityFile ~/.ssh/a\n  IdentitiesOnly no\n"
                                "  HostName ssh.github.com\n")
    argv = _ssh_argv(ssh_identity=explicit)
    assert _option_values(argv, "-i") == [str(explicit)]
    assert str(config_key) not in argv
    assert [value for value in _option_values(argv, "-o") if value.startswith("IdentitiesOnly")] == [
        "IdentitiesOnly=yes"
    ]
    assert "HostName=ssh.github.com" in argv


def test_git_ssh_identity_cli_beats_environment(tmp_path, monkeypatch):
    from coding_review_agent_loop.cli import build_parser
    from coding_review_agent_loop.config import resolve_git_ssh_identity

    monkeypatch.setenv("AGENT_LOOP_GIT_SSH_IDENTITY", "/env/key")
    args = build_parser().parse_args(["issue", "1", "--git-ssh-identity", "/cli/key"])
    assert args.git_ssh_identity == Path("/cli/key")
    assert resolve_git_ssh_identity(args, local_origin=None) == Path("/cli/key")
    assert resolve_git_ssh_identity(SimpleNamespace(git_ssh_identity=None), local_origin=None) == Path("/env/key")
    monkeypatch.setenv("AGENT_LOOP_GIT_SSH_IDENTITY", "")
    assert resolve_git_ssh_identity(SimpleNamespace(git_ssh_identity=None), local_origin=None) is None


@pytest.mark.parametrize("value,local_origin,message", [
    ("relative/key", None, "absolute path"),
    ("/keys/%h", None, "plain path"),
    ("/keys/${HOME}", None, "plain path"),
    ("/keys/k", Path("/srv/origin"), "cannot be combined with --trusted-local-origin"),
])
def test_git_ssh_identity_option_validation(monkeypatch, value, local_origin, message):
    from coding_review_agent_loop.config import resolve_git_ssh_identity

    monkeypatch.delenv("AGENT_LOOP_GIT_SSH_IDENTITY", raising=False)
    with pytest.raises(AgentLoopError, match=_re.escape(message)):
        resolve_git_ssh_identity(SimpleNamespace(git_ssh_identity=Path(value)), local_origin=local_origin)
    monkeypatch.setenv("AGENT_LOOP_GIT_SSH_IDENTITY", value)
    with pytest.raises(AgentLoopError, match=_re.escape(message)):
        resolve_git_ssh_identity(SimpleNamespace(git_ssh_identity=None), local_origin=local_origin)


def _foreign_lstat(monkeypatch, target: Path):
    """Report ``target`` (not its referent) as owned by another uid."""
    real = Path.lstat

    def fake(self):
        info = real(self)
        if str(self) == str(target):
            values = list(info[:10])
            values[4] = os.getuid() + 4242
            return os.stat_result(values)
        return info

    monkeypatch.setattr(Path, "lstat", fake)


@pytest.mark.parametrize("case", [
    "group-writable", "world-writable", "writable-ancestor", "foreign-owned", "directory",
    "fifo", "sibling-checkout", "seat-workdir", "store", "prior-run-dir",
    "symlink-to-group-writable", "symlink-under-writable-ancestor", "symlink-into-checkout",
    "foreign-owned-symlink",
])
def test_untrusted_identity_is_refused(ssh_home, tmp_path, monkeypatch, capsys, case):
    from coding_review_agent_loop import git_transport

    roots = {name: tmp_path / name for name in ("sibling-checkout", "seat-workdir", "store", "prior-run-dir")}
    for root in roots.values():
        root.mkdir(mode=0o700)
        root.chmod(0o700)
    keys = ssh_home / "keys"
    keys.mkdir()
    keys.chmod(0o700)
    if case in roots:
        identity = _key(roots[case] / "nested")
    elif case == "directory":
        identity = keys / "dir"
        identity.mkdir()
    elif case == "fifo":
        identity = keys / "fifo"
        os.mkfifo(identity, 0o600)
    elif case == "writable-ancestor":
        identity = _key(keys / "shared")
        (keys / "shared").chmod(0o770)
    elif case.startswith("symlink") or case == "foreign-owned-symlink":
        target_dir = {"symlink-under-writable-ancestor": keys / "shared",
                      "symlink-into-checkout": roots["sibling-checkout"]}.get(case, keys)
        target = _key(target_dir, "target")
        if case == "symlink-to-group-writable":
            target.chmod(0o620)
        if case == "symlink-under-writable-ancestor":
            target_dir.chmod(0o770)
        identity = keys / "link"
        identity.symlink_to(target)
        if case == "foreign-owned-symlink":
            _foreign_lstat(monkeypatch, identity)
    else:
        identity = _key(keys)
        if case == "group-writable":
            identity.chmod(0o620)
        elif case == "world-writable":
            identity.chmod(0o602)
        elif case == "foreign-owned":
            _foreign_lstat(monkeypatch, identity)
    forbidden = tuple(roots.values())
    with pytest.raises(AgentLoopError) as raised:
        _ssh_argv(ssh_identity=identity, forbidden_roots=forbidden)
    assert str(raised.value) == git_transport._SSH_IDENTITY_REFUSED
    assert str(identity) not in str(raised.value)
    _write_ssh_config(ssh_home, f"Host github.com\n  IdentityFile {identity}\n")
    argv = _ssh_argv(forbidden_roots=forbidden)
    assert "-i" not in argv
    assert "IdentityFile for trusted Git transport" in capsys.readouterr().err


def test_trusted_symlinks_are_accepted_and_realpaths_emitted(tmp_path, monkeypatch, ssh_home):
    real_ssh = tmp_path / "real-ssh"
    real_ssh.mkdir(mode=0o700)
    key = _key(real_ssh, "github_key")
    home = tmp_path / "linked-home"
    home.mkdir(mode=0o700)
    (home / ".ssh").symlink_to(real_ssh)
    (home / "id_link").symlink_to(key)
    assert (home / "id_link").lstat().st_mode & 0o777 == 0o777 or os.uname().sysname != "Linux"
    _write_ssh_config(home, "Host github.com\n  IdentityFile ~/id_link\n")
    monkeypatch.setenv("HOME", str(home))
    argv = _ssh_argv()
    assert argv[len(_SSH_PREFIX):] == ["-i", str(key)]


def test_trusted_operator_file_return_shapes(ssh_home):
    from coding_review_agent_loop.git_transport import _trusted_operator_file

    key = _key(ssh_home / ".ssh")
    info = key.stat()
    existing = _trusted_operator_file(key, forbidden_roots=())
    assert existing.path == key.resolve() and existing.identity == (info.st_dev, info.st_ino)
    missing = _trusted_operator_file(ssh_home / ".ssh" / "known_hosts_new", forbidden_roots=(),
                                     allow_missing=True)
    assert missing.path == ssh_home.resolve() / ".ssh" / "known_hosts_new"
    assert missing.identity is None


def test_config_swapped_between_walk_and_open_is_ignored(ssh_home, monkeypatch, capsys):
    from coding_review_agent_loop import git_transport

    _key(ssh_home / ".ssh")
    config = _write_ssh_config(ssh_home, "Host github.com\n  IdentityFile ~/.ssh/k\n")
    monkeypatch.setattr(git_transport, "_trusted_operator_file",
                        lambda *_a, **_k: git_transport.TrustedFile(config.resolve(), (0, 0)))
    assert git_transport._read_ssh_config(ssh_home, ()) is None
    assert git_transport._SSH_CONFIG_IGNORED in capsys.readouterr().err


@pytest.mark.parametrize("case", [
    "config-group-writable", "ssh-dir-world-writable", "home-group-writable", "foreign-owned",
    "home-inside-run-root", "fifo", "oversized",
])
def test_untrusted_ssh_config_is_ignored_entirely(ssh_home, tmp_path, monkeypatch, capsys, case):
    from coding_review_agent_loop import git_transport

    home = ssh_home
    roots: tuple[Path, ...] = ()
    if case == "home-inside-run-root":
        roots = (home,)
    _key(home / ".ssh")
    config = home / ".ssh" / "config"
    text = ("Host github.com\n  HostName evil.example.com\n  IdentityFile ~/.ssh/k\n"
            "  UserKnownHostsFile ~/.ssh/kh\n")
    if case == "fifo":
        os.mkfifo(config, 0o600)
    else:
        _write_ssh_config(home, text + ("#" * (256 * 1024) if case == "oversized" else ""))
    if case == "config-group-writable":
        config.chmod(0o620)
    elif case == "ssh-dir-world-writable":
        (home / ".ssh").chmod(0o703)
    elif case == "home-group-writable":
        home.chmod(0o770)
    elif case == "foreign-owned":
        _foreign_lstat(monkeypatch, config)
    result: dict[str, list[str]] = {}
    worker = _threading.Thread(target=lambda: result.update(argv=_ssh_argv(forbidden_roots=roots)),
                               daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "reading a non-regular config must not block"
    assert result["argv"][len(_SSH_PREFIX):] == []
    assert capsys.readouterr().err.count(git_transport._SSH_CONFIG_IGNORED) == 1


@pytest.mark.parametrize("case", ["checkout", "store", "writable-ancestor", "two-paths", "missing"])
def test_known_hosts_file_passes_shared_trust_walk(ssh_home, tmp_path, capsys, case):
    checkout, store = tmp_path / "checkout", tmp_path / "store"
    for root in (checkout, store):
        root.mkdir()
        root.chmod(0o700)
    shared = ssh_home / "shared"
    shared.mkdir()
    shared.chmod(0o770)
    value = {
        "checkout": str(checkout / "known_hosts"),
        "store": str(store / "known_hosts"),
        "writable-ancestor": str(shared / "known_hosts"),
        "two-paths": "~/.ssh/a ~/.ssh/b",
        "missing": "~/.ssh/known_hosts_missing",
    }[case]
    for path in (checkout / "known_hosts", store / "known_hosts", shared / "known_hosts"):
        path.write_text("")
        path.chmod(0o600)
    _write_ssh_config(ssh_home, f"Host github.com\n  UserKnownHostsFile {value}\n")
    argv = _ssh_argv(forbidden_roots=(checkout, store))
    if case == "missing":
        assert argv[len(_SSH_PREFIX):] == [
            "-o", f"UserKnownHostsFile={ssh_home.resolve() / '.ssh' / 'known_hosts_missing'}"
        ]
        assert not (ssh_home / ".ssh" / "known_hosts_missing").exists()
    else:
        assert argv[len(_SSH_PREFIX):] == []
        assert "UserKnownHostsFile for trusted Git transport" in capsys.readouterr().err


def test_windows_ssh_command_is_unchanged_and_override_refused(tmp_path, monkeypatch):
    from coding_review_agent_loop import git_transport

    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "config").write_text("Host github.com\n  IdentityFile ~/.ssh/k\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(git_transport.sys, "platform", "win32")
    monkeypatch.setattr(git_transport, "_git_path", lambda: r"C:\Program Files\Git\bin\git.exe")
    monkeypatch.setattr(git_transport, "_trusted_executable", lambda path: path)
    real_isfile = os.path.isfile
    monkeypatch.setattr(git_transport.os.path, "isfile",
                        lambda path: path.endswith("ssh.exe") or real_isfile(path))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Windows must not read ~/.ssh/config or call os.getuid")

    monkeypatch.setattr(git_transport, "_read_ssh_config", forbidden)
    monkeypatch.setattr(git_transport.os, "getuid", forbidden, raising=False)
    command = git_transport._transport_env("git@github.com:OWNER/REPO.git", None)["GIT_SSH_COMMAND"]
    assert command == (
        f'"C:\\Program Files\\Git\\usr\\bin\\ssh.exe" -F {os.devnull} -o ProxyCommand=none '
        "-o PermitLocalCommand=no -o ClearAllForwardings=yes"
    )
    with pytest.raises(AgentLoopError, match="not supported on Windows"):
        git_transport._transport_env("git@github.com:OWNER/REPO.git", None,
                                     ssh_identity=Path("C:/keys/k"))


def test_import_ref_passes_identity_and_destination_forbidden_root(repositories, monkeypatch):
    from coding_review_agent_loop import git_transport

    _, _, checkout = repositories
    git(checkout, "config", "remote.origin.url", "git@github.com:OWNER/REPO.git")
    seen = {}

    def recording_env(url, gh_cmd, **kwargs):
        seen.update(kwargs)
        raise AgentLoopError("stop before transport")

    monkeypatch.setattr(git_transport, "_transport_env", recording_env)
    store = Path("/srv/agent-store")
    with pytest.raises(AgentLoopError, match="stop before transport"):
        import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main", repo="OWNER/REPO",
                   runner=Runner(), ssh_identity=Path("/keys/k"), forbidden_roots=(store,))
    assert seen == {"ssh_identity": Path("/keys/k"), "forbidden_roots": (store, checkout)}


_FOOTER = ("fatal: Could not read from remote repository.\n\n"
           "Please make sure you have the correct access rights\nand the repository exists.\n")
_SECRET = "ghp_fixtureSecretValue123"


@pytest.mark.parametrize("stderr,label", [
    (f"ssh: connect to host h port 22: Connection refused {_SECRET}\n" + _FOOTER, "network or timeout"),
    (f"ssh: connect to host h port 22: Connection timed out {_SECRET}\n" + _FOOTER, "network or timeout"),
    (f"ssh: Could not resolve hostname h: {_SECRET}\n" + _FOOTER, "network or timeout"),
    (f"ERROR: Repository not found. {_SECRET}\n" + _FOOTER, "remote not found"),
    (f"Host key verification failed. {_SECRET}\n" + _FOOTER, "host-key failure"),
    (f"git@github.com: Permission denied (publickey). {_SECRET}\n" + _FOOTER, "authentication failure"),
    (_FOOTER + _SECRET, "authentication failure"),
    (f"fatal: something new {_SECRET}\n", "unclassified"),
])
@pytest.mark.parametrize("ssh", [True, False])
def test_transport_failure_class_is_fixed_and_nonsecret(monkeypatch, capsys, stderr, label, ssh):
    from coding_review_agent_loop import git_transport

    monkeypatch.setattr(git_transport, "_git_path", lambda: "/usr/bin/git")
    monkeypatch.setattr(
        git_transport.subprocess, "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 128, b"", stderr.encode()),
    )
    with pytest.raises(AgentLoopError) as raised:
        git_transport._private_git(("fetch", f"https://x-access-token:{_SECRET}@example/r"),
                                   cwd=Path("/"), env={}, ssh=ssh)
    message = str(raised.value)
    assert message.startswith(f"Trusted Git transport failed with exit 128 ({label}).")
    # #1383: the same fixed class is structured data on a TransportFailure.
    assert isinstance(raised.value, git_transport.TransportFailure)
    assert raised.value.label == label
    assert _SECRET not in repr(vars(raised.value))
    assert _SECRET not in message and "fetch" not in message and "Connection" not in message
    assert ("--git-ssh-identity" in message) == (ssh and label == "authentication failure")
    assert ("known_hosts" in message) == (ssh and label == "host-key failure")
    captured = capsys.readouterr()
    assert _SECRET not in captured.out + captured.err


def test_transport_timeout_reports_network_class(monkeypatch):
    from coding_review_agent_loop import git_transport

    def timeout(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, 1, stderr=_SECRET.encode())

    monkeypatch.setattr(git_transport, "_git_path", lambda: "/usr/bin/git")
    monkeypatch.setattr(git_transport.subprocess, "run", timeout)
    with pytest.raises(AgentLoopError, match=_re.escape("timed out (network or timeout)")) as raised:
        git_transport._private_git(("fetch",), cwd=Path("/"), env={}, ssh=True)
    assert _SECRET not in str(raised.value)
    assert isinstance(raised.value, git_transport.TransportFailure)
    assert raised.value.label == "network or timeout"
    assert str(raised.value) == "Trusted Git transport timed out (network or timeout)."


def test_https_auth_failure_still_retries_with_gh_token(repositories, monkeypatch):
    from coding_review_agent_loop import git_transport

    _, _, checkout = repositories
    git(checkout, "config", "remote.origin.url", "https://github.com/OWNER/REPO.git")
    fetches = []
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if command[0] == git_transport._git_path() and "fetch" in command:
            fetches.append("GIT_CONFIG_COUNT" in kwargs["env"])
            return subprocess.CompletedProcess(command, 128, b"", b"returned error: 401\n")
        return real_run(command, **kwargs)

    monkeypatch.setattr(git_transport.subprocess, "run", fake_run)
    monkeypatch.setattr(git_transport, "_token_from_gh", lambda *_args: "token")
    with pytest.raises(AgentLoopError, match=_re.escape("(authentication failure).")) as raised:
        import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main",
                   repo="OWNER/REPO", runner=Runner(), gh_cmd="gh")
    assert fetches == [False, True]
    assert "--git-ssh-identity" not in str(raised.value)


@pytest.mark.parametrize("defect", [
    None, "writable-intermediate-dir", "foreign-intermediate-link", "selection-in-forbidden-root",
])
@pytest.mark.parametrize("selection", [
    "identity-override", "config-identity", "ssh-config", "known-hosts-missing",
])
def test_every_symlink_hop_and_selection_path_is_trust_checked(
        ssh_home, tmp_path, monkeypatch, capsys, selection, defect):
    """An intermediate link, the directory it lives in, and every selection path
    (not only the final referent) must be operator-controlled and outside agent roots."""
    from coding_review_agent_loop import git_transport

    checkout = tmp_path / "agent-checkout"
    checkout.mkdir()
    checkout.chmod(0o700)
    real = ssh_home / "real"
    real.mkdir()
    real.chmod(0o700)
    hopdir = ssh_home / ("shared" if defect == "writable-intermediate-dir" else "hops")
    hopdir.mkdir()
    hopdir.chmod(0o770 if defect == "writable-intermediate-dir" else 0o700)
    plain_key = _key(ssh_home / ".ssh", "plain")
    if selection == "known-hosts-missing":
        target = real
    elif selection == "ssh-config":
        target = _write_ssh_config(ssh_home, "Host github.com\n  IdentityFile ~/.ssh/plain\n")
        target = target.rename(real / "config")
    else:
        target = _key(real, "key")
    hop = hopdir / "hop"
    hop.symlink_to(target)
    if defect == "foreign-intermediate-link":
        _foreign_lstat(monkeypatch, hop)
    base = checkout if defect == "selection-in-forbidden-root" else ssh_home
    roots = (checkout,)
    trusted = defect is None

    if selection == "ssh-config":
        home = ssh_home
        if defect == "selection-in-forbidden-root":
            # HOME inside an agent checkout whose .ssh points outward to trusted files.
            home = checkout / "home"
            home.mkdir(mode=0o700)
            (home / ".ssh").symlink_to(real)
            monkeypatch.setenv("HOME", str(home))
        else:
            (ssh_home / ".ssh" / "config").symlink_to(hop)
        argv = _ssh_argv(forbidden_roots=roots)
        if trusted:
            assert argv[len(_SSH_PREFIX):] == ["-i", str(plain_key)]
        else:
            assert argv[len(_SSH_PREFIX):] == []
            assert git_transport._SSH_CONFIG_IGNORED in capsys.readouterr().err
        return

    entry = base / "entry"
    entry.symlink_to(hop)
    if selection == "identity-override":
        if trusted:
            assert _option_values(_ssh_argv(ssh_identity=entry, forbidden_roots=roots), "-i") == [str(target)]
        else:
            with pytest.raises(AgentLoopError) as raised:
                _ssh_argv(ssh_identity=entry, forbidden_roots=roots)
            assert str(raised.value) == git_transport._SSH_IDENTITY_REFUSED
        return
    if selection == "config-identity":
        _write_ssh_config(ssh_home, f"Host github.com\n  IdentityFile {entry}\n")
        argv = _ssh_argv(forbidden_roots=roots)
        assert argv[len(_SSH_PREFIX):] == (["-i", str(target)] if trusted else [])
    else:
        _write_ssh_config(ssh_home, f"Host github.com\n  UserKnownHostsFile {entry}/known_hosts\n")
        argv = _ssh_argv(forbidden_roots=roots)
        assert argv[len(_SSH_PREFIX):] == (
            ["-o", f"UserKnownHostsFile={target / 'known_hosts'}"] if trusted else []
        )
    if not trusted:
        assert "for trusted Git transport" in capsys.readouterr().err


def test_symlink_loop_is_refused_without_hanging(ssh_home):
    from coding_review_agent_loop.git_transport import _UntrustedPath, _trusted_operator_file

    loop_a, loop_b = ssh_home / "a", ssh_home / "b"
    loop_a.symlink_to(loop_b)
    loop_b.symlink_to(loop_a)
    with pytest.raises(_UntrustedPath):
        _trusted_operator_file(loop_a, forbidden_roots=())


@pytest.mark.parametrize("value", ["/", "~/rootlink"])
@pytest.mark.parametrize("selection", ["config-identity", "known-hosts", "identity-override"])
def test_filesystem_root_is_refused_as_non_regular(ssh_home, capsys, selection, value):
    """`/` (directly or through a link) traverses no file component; it must be
    refused with the fixed warning or error, never crash."""
    from coding_review_agent_loop import git_transport

    (ssh_home / "rootlink").symlink_to("/")
    if selection == "identity-override":
        override = Path(str(ssh_home) + value[1:]) if value.startswith("~") else Path(value)
        with pytest.raises(AgentLoopError) as raised:
            _ssh_argv(ssh_identity=override)
        assert str(raised.value) == git_transport._SSH_IDENTITY_REFUSED
        return
    key = "IdentityFile" if selection == "config-identity" else "UserKnownHostsFile"
    _write_ssh_config(ssh_home, f"Host github.com\n  {key} {value}\n")
    argv = _ssh_argv()
    assert argv[len(_SSH_PREFIX):] == []
    assert f"{key} for trusted Git transport" in capsys.readouterr().err
