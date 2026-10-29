"""Per-run worktrees over a shared store (#1162): the four operator scenarios."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_review_agent_loop import run_worktrees, workdir_claims
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import (
    config_from_args,
    default_agent_workdir,
    default_run_worktree_root,
    ensure_agent_workdirs,
    sync_checkout_to_pr,
    sync_coder_base_before_implementation,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import CommandResult, Runner
from coding_review_agent_loop.workdir_claims import (
    claim_agent_workdirs,
    probe_free_claim,
    workdir_claim_scope,
)

CHILD = r"""
import json, os, sys, tempfile
from pathlib import Path
claim, stores, tmp, xdg, bare, gh, coder, reviewer = sys.argv[1:9]
tempfile.tempdir = tmp
os.environ["XDG_CACHE_HOME"] = xdg
os.environ["FAKE_GH_BARE"] = bare
from coding_review_agent_loop import run_worktrees as rw, workdir_claims as w
w._set_claim_root_for_tests(Path(claim))
rw._set_store_lock_root_for_tests(Path(stores))
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import config_from_args, ensure_agent_workdirs
from coding_review_agent_loop.runner import Runner
args = build_parser().parse_args([
    "task", "x", "--repo", "OWNER/REPO", "--coder", coder, "--reviewer", reviewer,
    "--base", "main", "--gh-cmd", gh, "--trusted-local-origin", bare, "--claude-cmd", "/bin/true",
    "--codex-cmd", "/bin/true", "--antigravity-cmd", "/bin/true", "--quiet",
    *sys.argv[9:],
])
runner = Runner()
config = config_from_args(args, runner)
with w.workdir_claim_scope(command="task"):
    w.claim_agent_workdirs(config)
    ensure_agent_workdirs(config, runner)
    print(json.dumps({"claude": str(config.claude_dir), "codex": str(config.codex_dir)}), flush=True)
    sys.stdin.read()
"""

STORE_HOLDER = r"""
import sys
from pathlib import Path
from coding_review_agent_loop import workdir_claims as w
w._set_claim_root_for_tests(Path(sys.argv[1]))
with w.workdir_claim_scope(command="task"):
    w.acquire_workdir_claim(Path(sys.argv[2]), agent="codex", repo="OWNER/REPO")
    print("held", flush=True)
    sys.stdin.read()
"""


SRC = str(Path(run_worktrees.__file__).resolve().parent.parent)


def child_env():
    return {**os.environ, "PYTHONPATH": SRC}


def git(path, *args, check=True):
    result = subprocess.run(
        ["git", "-C", str(path), *args], text=True, capture_output=True, check=False
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {args} failed: {result.stderr}")
    return result.stdout.strip()


class Env:
    """A local bare origin, a fake ``gh repo clone``, and isolated roots."""

    def __init__(self, tmp_path: Path, monkeypatch):
        self.tmp = tmp_path / "tmp"
        self.tmp.mkdir()
        self.xdg = tmp_path / "xdg"
        monkeypatch.setattr(tempfile, "tempdir", str(self.tmp))
        monkeypatch.setenv("XDG_CACHE_HOME", str(self.xdg))
        for key, value in (
            ("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.com"),
            ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@example.com"),
            ("GIT_CONFIG_GLOBAL", os.devnull), ("GIT_CONFIG_SYSTEM", os.devnull),
        ):
            monkeypatch.setenv(key, value)
        self.bare = tmp_path / "origins" / "OWNER" / "REPO.git"
        self.bare.parent.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.bare)], check=True)
        self.seed = tmp_path / "seed"
        subprocess.run(["git", "clone", "-q", str(self.bare), str(self.seed)], check=True)
        git(self.seed, "checkout", "-q", "-B", "main")
        self.commit("a.txt", "one")
        monkeypatch.setenv("FAKE_GH_BARE", str(self.bare))
        self.gh = tmp_path / "fake-gh"
        self.gh.write_text('#!/bin/sh\nexec git clone -q "$FAKE_GH_BARE" "$4"\n')
        self.gh.chmod(0o755)

    def commit(self, name: str, text: str) -> str:
        (self.seed / name).write_text(text)
        git(self.seed, "add", name)
        git(self.seed, "commit", "-q", "-m", text)
        git(self.seed, "push", "-q", "origin", "HEAD:refs/heads/main")
        return git(self.seed, "rev-parse", "HEAD")

    def set_pr_head(self, number: int, sha: str) -> None:
        git(self.bare, "update-ref", f"refs/pull/{number}/head", sha)

    def config(self, *, coder="claude", reviewer="codex", extra=()):
        args = build_parser().parse_args([
            "task", "x", "--repo", "OWNER/REPO", "--coder", coder, "--reviewer", reviewer,
            "--base", "main", "--gh-cmd", str(self.gh), "--trusted-local-origin", str(self.bare), "--claude-cmd", "/bin/true",
            "--codex-cmd", "/bin/true", "--antigravity-cmd", "/bin/true", "--quiet", *extra,
        ])
        return config_from_args(args, Runner())

    def seed_store(self, agent: str, *, venv: bool = True) -> Path:
        """Clone the store the way prepare does, then add an untracked ``.venv``."""
        store = default_agent_workdir("OWNER/REPO", agent)
        store.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(self.gh), "repo", "clone", "OWNER/REPO", str(store)], check=True)
        if venv:
            marker = store / ".venv" / "bin" / "python"
            marker.parent.mkdir(parents=True)
            marker.write_text("#!/bin/sh\n")
        return store

    def seed_stores(self, *, venv: bool = True) -> None:
        for agent in ("claude", "codex"):
            self.seed_store(agent, venv=venv)

    def child(self, *, coder="claude", reviewer="codex", extra=()):
        proc = subprocess.Popen(
            [
                sys.executable, "-c", CHILD,
                str(workdir_claims.claim_root()), str(run_worktrees.store_lock_root()),
                str(self.tmp), str(self.xdg), str(self.bare), str(self.gh), coder, reviewer,
                *extra,
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=child_env(),
        )
        line = proc.stdout.readline()
        assert line, "child exited before preparing its worktrees"
        return proc, json.loads(line)


def stop(proc, *, kill=False):
    if kill:
        proc.send_signal(signal.SIGKILL)
    else:
        proc.stdin.close()
    proc.wait(timeout=60)


class RecordingRunner(Runner):
    def __init__(self, hook=None):
        super().__init__()
        self.calls: list[tuple[str, ...]] = []
        self.hook = hook

    def run(self, args, *, cwd, **kwargs):
        args = tuple(str(a) for a in args)
        self.calls.append(args)
        if self.hook is not None:
            self.hook(args, Path(cwd))
        return super().run(args, cwd=cwd, **kwargs)


class FailingGitRunner(Runner):
    """Fails the named git subcommands (e.g. ``worktree remove``) with exit 1."""

    def __init__(self, *failing: tuple[str, str]):
        super().__init__()
        self.failing = failing

    def run(self, args, *, cwd, **kwargs):
        args = tuple(str(a) for a in args)
        if args[:1] == ("git",) and args[1:3] in self.failing:
            return CommandResult(list(args), cwd, "", "forced failure", 1)
        return super().run(args, cwd=cwd, **kwargs)


def prepare(config, runner):
    with workdir_claim_scope(command="task"):
        claim_agent_workdirs(config)
        ensure_agent_workdirs(config, runner)


def test_concurrent_default_runs_use_distinct_worktrees(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    old_sha = git(env.seed, "rev-parse", "HEAD")
    proc_a, paths_a = env.child()
    try:
        a_path = Path(paths_a["claude"])
        a_head = git(a_path, "rev-parse", "HEAD")
        config_b = env.config()
        runner = Runner()
        with workdir_claim_scope(command="task"):
            claim_agent_workdirs(config_b)
            ensure_agent_workdirs(config_b, runner)
            b_path = config_b.claude_dir
            store = default_agent_workdir("OWNER/REPO", "claude").resolve()

            # Both proceed in distinct worktrees; A's live worktree survived B's prune.
            assert b_path != a_path and a_path.is_dir() and b_path.is_dir()
            (b_path / "only-b.txt").write_text("b")
            assert not (a_path / "only-b.txt").exists()
            assert git(a_path, "status", "--porcelain") == ""
            assert git(a_path, "rev-parse", "HEAD") == a_head == old_sha

            # Base checked out elsewhere: the local-base fast-forward is skipped.
            git(a_path, "switch", "-q", "main")
            new_sha = env.commit("a.txt", "two")
            (b_path / "only-b.txt").unlink()
            sync_coder_base_before_implementation(config_b, runner)
            assert git(b_path, "rev-parse", "HEAD") == new_sha
            assert git(b_path, "symbolic-ref", "-q", "HEAD", check=False) == ""
            assert git(store, "rev-parse", "refs/heads/main") == old_sha
            assert git(a_path, "rev-parse", "HEAD") == old_sha
            assert git(a_path, "symbolic-ref", "HEAD") == "refs/heads/main"
            git(a_path, "switch", "-q", "--detach")

            # PR pin interleaving: another run advances the shared PR ref mid-sync.
            pinned = env.commit("b.txt", "pr-one")
            env.set_pr_head(7, pinned)
            advanced = env.commit("b.txt", "pr-two")

            def advance(args, cwd):
                if args[:3] == ("git", "checkout", "--detach") and cwd == b_path:
                    env.set_pr_head(7, advanced)
                    # The store lock must be free here (a nested take would raise).
                    other = run_worktrees.pin_pr_head(store, 7, config=config_b, runner=Runner())
                    assert other == advanced

            env.set_pr_head(7, pinned)
            pausing = RecordingRunner(hook=advance)
            sync_checkout_to_pr(
                config_b, pausing, path=b_path, label="Default claude workdir",
                default_owned=True, pr_number=7, pr_metadata=SimpleNamespace(head_sha=pinned),
            )
            assert git(b_path, "rev-parse", "HEAD") == pinned
            # Without interleaving, an advertised head that differs from the pin still errors.
            before = git(b_path, "rev-parse", "HEAD")
            with pytest.raises(AgentLoopError, match="advertises head SHA"):
                sync_checkout_to_pr(
                    config_b, Runner(), path=b_path, label="Default claude workdir",
                    default_owned=True, pr_number=7, pr_metadata=SimpleNamespace(head_sha=pinned),
                )
            assert git(b_path, "rev-parse", "HEAD") == before
        assert a_path.is_dir()
    finally:
        stop(proc_a)
    assert not a_path.exists()


def test_linked_store_sync_ignores_planted_execution(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    config = env.config()
    with workdir_claim_scope(command="task"):
        claim_agent_workdirs(config)
        ensure_agent_workdirs(config, Runner())
        store = default_agent_workdir("OWNER/REPO", "claude").resolve()
        marker = tmp_path / "shared-store-marker"
        hook = store / ".git" / "hooks" / "post-checkout"
        hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
        hook.chmod(0o755)
        git(store, "config", "core.fsmonitor", f"touch {marker}")
        sha = env.commit("a.txt", "new base")
        sync_coder_base_before_implementation(config, Runner())
        assert git(config.claude_dir, "rev-parse", "HEAD") == sha
        env.set_pr_head(7, sha)
        assert run_worktrees.pin_pr_head(store, 7, config=config, runner=Runner()) == sha
        assert not marker.exists()


def test_failed_fresh_materialization_keeps_pending_record_only(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    config = env.config()
    store = default_agent_workdir("OWNER/REPO", "claude").resolve()
    sha = run_worktrees.prepare_store(store, config=config, runner=Runner())
    path = config.claude_dir

    class FailReset(RecordingRunner):
        def run(self, args, *, cwd, **kwargs):
            if tuple(args[:2]) == ("git", "reset"):
                raise AgentLoopError("forced reset failure")
            return super().run(args, cwd=cwd, **kwargs)

    with run_worktrees.store_lock(store):
        with pytest.raises(AgentLoopError, match="forced reset failure"):
            run_worktrees.add_run_worktree(store, path, sha, config=config, runner=FailReset())
        record = run_worktrees._read_record(store, path.name)
        assert record is not None and record["state"] == "pending"
        assert not path.exists()


@pytest.mark.parametrize("failure", ["admin_link", "reset", "registration", "ready_record"])
def test_fresh_worktree_failure_cleans_recorded_identities(tmp_path, monkeypatch, failure):
    env = Env(tmp_path, monkeypatch)
    config = env.config()
    store = default_agent_workdir("OWNER/REPO", "claude").resolve()
    sha = run_worktrees.prepare_store(store, config=config, runner=Runner())
    path = config.claude_dir
    admin = store / ".git" / "worktrees" / path.name

    if failure == "admin_link":
        monkeypatch.setattr(run_worktrees, "_admin_dir_of", lambda *_: None)
    elif failure == "registration":
        from coding_review_agent_loop import agent_permissions
        monkeypatch.setattr(agent_permissions, "register_checkout",
                            lambda *_: (_ for _ in ()).throw(AgentLoopError("registration failed")))
    elif failure == "ready_record":
        real_write = run_worktrees._write_reservation

        def fail_ready(record, value):
            if value["state"] == "ready":
                raise AgentLoopError("ready record failed")
            return real_write(record, value)

        monkeypatch.setattr(run_worktrees, "_write_reservation", fail_ready)

    class FailReset(RecordingRunner):
        def run(self, args, *, cwd, **kwargs):
            if failure == "reset" and tuple(args[:2]) == ("git", "reset"):
                raise AgentLoopError("reset failed")
            return super().run(args, cwd=cwd, **kwargs)

    with run_worktrees.store_lock(store):
        with pytest.raises(AgentLoopError):
            run_worktrees.add_run_worktree(store, path, sha, config=config, runner=FailReset())
        assert run_worktrees._read_record(store, path.name)["state"] == "pending"
        assert not path.exists() and not admin.exists()


@pytest.mark.parametrize("failure", ["write", "fsync"])
def test_partial_gitfile_failure_cleans_recorded_link(tmp_path, monkeypatch, failure):
    env = Env(tmp_path, monkeypatch)
    config = env.config()
    store = default_agent_workdir("OWNER/REPO", "claude").resolve()
    sha = run_worktrees.prepare_store(store, config=config, runner=Runner())
    path = config.claude_dir
    admin = store / ".git" / "worktrees" / path.name
    original_open = run_worktrees.os.open
    original_fdopen = run_worktrees.os.fdopen
    original_fsync = run_worktrees.os.fsync
    target_fd = None

    def tracked_open(target, *args, **kwargs):
        nonlocal target_fd
        fd = original_open(target, *args, **kwargs)
        if Path(target) == path / ".git":
            target_fd = fd
        return fd

    class BrokenWrite:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, _value):
            raise OSError("injected gitfile write failure")

        def __getattr__(self, name):
            return getattr(self.handle, name)

    def tracked_fdopen(fd, *args, **kwargs):
        handle = original_fdopen(fd, *args, **kwargs)
        return BrokenWrite(handle) if failure == "write" and fd == target_fd else handle

    def tracked_fsync(fd):
        if failure == "fsync" and fd == target_fd:
            raise OSError("injected gitfile fsync failure")
        return original_fsync(fd)

    monkeypatch.setattr(run_worktrees.os, "open", tracked_open)
    monkeypatch.setattr(run_worktrees.os, "fdopen", tracked_fdopen)
    monkeypatch.setattr(run_worktrees.os, "fsync", tracked_fsync)
    with run_worktrees.store_lock(store):
        with pytest.raises(OSError, match="injected gitfile"):
            run_worktrees.add_run_worktree(store, path, sha, config=config, runner=Runner())
        assert run_worktrees._read_record(store, path.name)["state"] == "pending"
        assert not path.exists() and not admin.exists()


def test_fresh_worktree_cleanup_failure_is_reported(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    config = env.config()
    store = default_agent_workdir("OWNER/REPO", "claude").resolve()
    sha = run_worktrees.prepare_store(store, config=config, runner=Runner())
    path = config.claude_dir
    real_rmtree = run_worktrees.shutil.rmtree

    def refuse_root(target, *args, **kwargs):
        if Path(target) == path:
            raise OSError("cleanup refused")
        return real_rmtree(target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(run_worktrees, "_admin_dir_of", lambda *_: None)
        patch.setattr(run_worktrees.shutil, "rmtree", refuse_root)
        with run_worktrees.store_lock(store):
            with pytest.raises(AgentLoopError, match="cleanup failed.*cleanup refused"):
                run_worktrees.add_run_worktree(store, path, sha, config=config, runner=Runner())
            assert run_worktrees._read_record(store, path.name)["state"] == "pending"


def test_run_end_removes_worktree(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    store = default_agent_workdir("OWNER/REPO", "claude").resolve()
    claim_held_during_removal: list[bool] = []

    def watch(args, cwd):
        if args[1:3] == ("worktree", "remove"):
            with probe_free_claim(Path(args[-1])) as free:
                claim_held_during_removal.append(not free)

    config = env.config()
    runner = RecordingRunner(hook=watch)
    with workdir_claim_scope(command="task"):
        claim_agent_workdirs(config)
        ensure_agent_workdirs(config, runner)
        path = config.claude_dir
        assert path.is_dir()
        config.log_dir.mkdir(parents=True)
        (config.log_dir / "salvage.patch").write_text("patch")
    assert not path.exists()
    assert str(path) not in git(store, "worktree", "list", "--porcelain")
    assert not run_worktrees._record_path(store, path.name).exists()
    assert (config.log_dir / "salvage.patch").read_text() == "patch"
    assert claim_held_during_removal and all(claim_held_during_removal)
    with probe_free_claim(path) as free:
        assert free

    # A store whose worktree list cannot be read is never recreated under live worktrees.
    live = env.config()
    with workdir_claim_scope(command="task"):
        claim_agent_workdirs(live)
        ensure_agent_workdirs(live, Runner())
        for entry in store.iterdir():
            if entry.name != ".git":
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        with pytest.raises(AgentLoopError, match="leaving the store untouched"):
            run_worktrees.prepare_store(
                store, config=live, runner=FailingGitRunner(("worktree", "list"))
            )
        assert (store / ".git").is_dir() and live.claude_dir.is_dir()
        # With a readable list, the live worktree also protects the store from recreation.
        run_worktrees.prepare_store(store, config=live, runner=Runner())
        assert (store / ".git").is_dir() and live.claude_dir.is_dir()

    # Failed unregistration keeps the owner record until git really forgets the worktree.
    stuck = env.config()
    with workdir_claim_scope(command="task"):
        claim_agent_workdirs(stuck)
        ensure_agent_workdirs(stuck, FailingGitRunner(("worktree", "remove")))
        stuck_path = stuck.claude_dir
        git(store, "worktree", "lock", str(stuck_path))
    assert not stuck_path.exists()
    stuck_record = run_worktrees._record_path(store, stuck_path.name)
    assert stuck_record.exists()  # registration survived, so the evidence is retained
    git(store, "worktree", "unlock", str(stuck_path), check=False)
    after = env.config()
    prepare(after, Runner())  # startup prune completes the cleanup and drops the record
    assert not stuck_record.exists()

    # A stale store with no .git (only tool artifacts) is re-cloned, not aborted.
    shutil.rmtree(store)
    (store / ".agent-loop-logs").mkdir(parents=True)
    prepare(env.config(), Runner())
    assert (store / ".git").is_dir()

    # A failing run removes its worktree too and its own exception propagates.
    failing = env.config()
    with pytest.raises(RuntimeError, match="boom"):
        with workdir_claim_scope(command="task"):
            claim_agent_workdirs(failing)
            ensure_agent_workdirs(failing, Runner())
            failed_path = failing.claude_dir
            assert failed_path.is_dir()
            raise RuntimeError("boom")
    assert not failed_path.exists()
    with probe_free_claim(failed_path) as free:
        assert free


def test_killed_run_worktree_pruned_at_next_startup(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    store = default_agent_workdir("OWNER/REPO", "claude").resolve()

    killed, killed_paths = env.child()
    killed_path = Path(killed_paths["claude"])
    (killed_path / "a.txt").write_text("dirty tracked change")
    (killed_path / "untracked.txt").write_text("dirty")
    stop(killed, kill=True)

    replaced, replaced_paths = env.child()
    replaced_path = Path(replaced_paths["claude"])
    stop(replaced, kill=True)
    moved = replaced_path.with_name(replaced_path.name + ".moved")
    replaced_path.rename(moved)
    replaced_path.mkdir()
    shutil.copy(moved / ".git", replaced_path / ".git")
    (replaced_path / "marker.txt").write_text("replacement")

    runs_root = default_run_worktree_root("OWNER/REPO", "claude").resolve()
    notes = runs_root / "notes"
    notes.mkdir()
    (notes / "keep.txt").write_text("keep")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    link = runs_root / "20200101-000000-000000-aaaaaaaaaaaa"
    link.symlink_to(outside)

    config = env.config()
    prepare(config, Runner())  # no dirty-checkout or claim refusal, no blocking

    assert not killed_path.exists()
    assert not run_worktrees._record_path(store, killed_path.name).exists()
    assert str(killed_path) not in git(store, "worktree", "list", "--porcelain")
    # Unverifiable entries are left alone.
    assert (replaced_path / "marker.txt").read_text() == "replacement"
    assert moved.is_dir()
    assert (notes / "keep.txt").read_text() == "keep"
    assert link.is_symlink() and (outside / "keep.txt").read_text() == "keep"


def test_explicit_claude_dir_unchanged(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    explicit = tmp_path / "explicit"
    subprocess.run(["git", "clone", "-q", str(env.bare), str(explicit)], check=True)
    config = env.config(reviewer="claude", extra=("--claude-dir", str(explicit)))
    assert config.default_checkout_stores == () or "claude" not in dict(config.default_checkout_stores)
    runner = RecordingRunner()
    prepare(config, runner)
    assert not any(call[:2] == ("git", "worktree") for call in runner.calls)
    runs_root = default_run_worktree_root("OWNER/REPO", "claude")
    assert not runs_root.exists()
    assert git(explicit, "symbolic-ref", "HEAD") == "refs/heads/main"

    # A claimed store (explicit-dir or older run) is only fetched, never modified.
    store = default_agent_workdir("OWNER/REPO", "codex").resolve()
    subprocess.run(["git", "clone", "-q", str(env.bare), str(store)], check=True)
    git(store, "switch", "-q", "-c", "holder-branch")
    (store / "a.txt").write_text("holder dirt")
    (store / "staged.txt").write_text("staged")
    git(store, "add", "staged.txt")
    before = (
        git(store, "symbolic-ref", "HEAD"), git(store, "rev-parse", "HEAD"),
        git(store, "status", "--porcelain"), git(store, "rev-parse", "refs/heads/main"),
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", STORE_HOLDER, str(workdir_claims.claim_root()), str(store)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=child_env(),
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        env.commit("a.txt", "newer origin")
        codex_config = env.config(coder="codex", reviewer="codex")
        prepare(codex_config, Runner())
        after = (
            git(store, "symbolic-ref", "HEAD"), git(store, "rev-parse", "HEAD"),
            git(store, "status", "--porcelain"), git(store, "rev-parse", "refs/heads/main"),
        )
        assert after == before
    finally:
        stop(holder)


# --- --worktree-link (#1176) ------------------------------------------------


def commit_gitignore(env, text):
    env.commit(".gitignore", text)


def prepare_inline(config):
    runner = Runner()
    claim_agent_workdirs(config)
    ensure_agent_workdirs(config, runner)
    return runner


def redirected_root(path):
    """Rename ``path`` aside and put a symlink to it at ``path``; return an undo callable."""
    aside = path.with_name(path.name + "-aside")
    path.rename(aside)
    path.symlink_to(aside)

    def undo():
        path.unlink()
        aside.rename(path)

    return undo


def test_worktree_link_created_and_ignored_link_is_integrity_checked(tmp_path, monkeypatch):
    from coding_review_agent_loop.local_test_evidence import (
        attribute_current_head,
        stable_tracked_tree_snapshot,
    )

    env = Env(tmp_path, monkeypatch)
    commit_gitignore(env, "/.venv\n")
    env.seed_stores()
    config = env.config(extra=("--worktree-link", ".venv"))
    with workdir_claim_scope(command="task"):
        prepare_inline(config)
        root = config.claude_dir
        link = root / ".venv"
        store_venv = default_agent_workdir("OWNER/REPO", "claude") / ".venv"
        assert link.is_symlink()
        assert os.path.realpath(link) == os.path.realpath(store_venv)
        snap = stable_tracked_tree_snapshot(root)
        assert snap.complete and ".venv" not in snap.untracked_paths
        target = os.readlink(link)

        other = tmp_path / "other-venv"
        other.mkdir()
        link.unlink()
        link.symlink_to(other)
        retargeted = stable_tracked_tree_snapshot(root)
        assert retargeted.complete is False
        assert attribute_current_head(retargeted, retargeted, current_head=snap.head).state == "unknown"
        link.unlink()
        removed = stable_tracked_tree_snapshot(root)
        assert removed.complete is False
        assert attribute_current_head(removed, removed, current_head=snap.head).state == "unknown"
        link.symlink_to(target)

        undo = redirected_root(root)
        try:
            redirected = stable_tracked_tree_snapshot(root)
            assert redirected.complete is False
            assert attribute_current_head(redirected, redirected, current_head=snap.head).state == "unknown"
        finally:
            undo()
        assert stable_tracked_tree_snapshot(root).complete is True


def test_worktree_link_survives_sync_clean_and_verifies(tmp_path, monkeypatch):
    from coding_review_agent_loop import checkout_verification as cv
    from coding_review_agent_loop import local_test_evidence as lte

    env = Env(tmp_path, monkeypatch)
    commit_gitignore(env, "!/.venv\n")
    env.seed_stores()
    config = env.config(extra=("--worktree-link", ".venv"))
    with workdir_claim_scope(command="task"):
        runner = prepare_inline(config)
        root = config.claude_dir
        target = os.readlink(root / ".venv")
        cv.verify_checkout(config, runner, path=root, agent="claude", purpose="agent turn")

        (root / "dirt.txt").write_text("x")
        (root / "a.txt").write_text("changed")
        cv.record_checkout_baseline(config, runner, root, source="agent turn")

        sync_coder_base_before_implementation(config, Runner())
        assert not (root / "dirt.txt").exists()
        assert (root / "a.txt").read_text() == "one"
        assert os.readlink(root / ".venv") == target
        cv.verify_checkout(config, runner, path=root, agent="claude", purpose="agent turn")

        argv = (".venv/bin/python", "-m", "pytest")
        snap = lte.stable_tracked_tree_snapshot(root, argv=argv)
        assert snap.complete and snap.stable
        assert ".venv" not in snap.untracked_paths
        assert ".venv" not in snap.referenced_untracked_paths
        assert lte.attribute_current_head(snap, snap, current_head=snap.head).state == "current-head"

        undo = redirected_root(root)
        try:
            redirected = lte.stable_tracked_tree_snapshot(root)
            assert redirected.complete is False
            assert lte.attribute_current_head(redirected, redirected, current_head=snap.head).state == "unknown"
        finally:
            undo()


def test_worktree_link_off_by_default(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    env.seed_stores()
    config = env.config()
    with workdir_claim_scope(command="task"):
        prepare_inline(config)
        assert config.claude_dir.is_dir()
        assert not os.path.lexists(config.claude_dir / ".venv")


def test_worktree_link_missing_source_fails_before_worktree(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    env.seed_stores(venv=False)
    config = env.config(extra=("--worktree-link", "missing-env"))
    with workdir_claim_scope(command="task"):
        claim_agent_workdirs(config)
        with pytest.raises(AgentLoopError, match="missing-env") as caught:
            ensure_agent_workdirs(config, Runner())
        runs_root = default_run_worktree_root("OWNER/REPO", "claude")
        assert not runs_root.exists() or not list(runs_root.iterdir())
    # The message names the shared clone and the remedy (#1650).
    message = str(caught.value)
    stores = {
        os.path.realpath(default_agent_workdir("OWNER/REPO", agent)) for agent in ("claude", "codex")
    }
    assert any(store in message for store in stores)
    assert "--worktree-setup CMD" in message
    assert "--worktree-setup-pr-local" in message
    assert "Refusing worktree creation" in message
    assert "succeeded but did not create" not in message


def _unignored_warnings(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(run_worktrees, "log", lambda _config, message: messages.append(message))
    return messages


@pytest.mark.parametrize("rule, warned", [("node_modules/\n", True), ("/node_modules\n", False)])
def test_worktree_link_not_ignored_as_symlink_warns_without_writing_ignore_state(
    tmp_path, monkeypatch, rule, warned
):
    env = Env(tmp_path, monkeypatch)
    commit_gitignore(env, rule)
    for agent in ("claude", "codex"):
        store = env.seed_store(agent, venv=False)
        (store / "node_modules").mkdir()
        (store / "node_modules" / "ready").touch()
    exclude = default_agent_workdir("OWNER/REPO", "claude") / ".git" / "info" / "exclude"
    before = exclude.read_bytes() if exclude.exists() else None
    messages = _unignored_warnings(monkeypatch)
    config = env.config(extra=("--worktree-link", "node_modules"))
    with workdir_claim_scope(command="task"):
        prepare_inline(config)
        assert (config.claude_dir / "node_modules").is_symlink()
    warnings = [m for m in messages if "is not matched by any ignore rule" in m]
    if warned:
        assert warnings and all("'/node_modules'" in m and "info/exclude" in m for m in warnings)
        assert "without a trailing slash" in warnings[0]
    else:
        assert warnings == []
    assert (exclude.read_bytes() if exclude.exists() else None) == before


def test_worktree_link_ignore_probe_failure_is_only_a_hint(tmp_path, monkeypatch):
    messages = _unignored_warnings(monkeypatch)

    class Broken:
        def run(self, args, *, cwd, check=True):
            raise AgentLoopError("probe failed")

    class Odd:
        def run(self, args, *, cwd, check=True):
            return SimpleNamespace(returncode=128, stdout="", stderr="fatal")

    for runner in (Broken(), Odd()):
        assert run_worktrees.warn_unignored_links(
            tmp_path, tmp_path, ("node_modules",), config=None, runner=runner
        ) == []
    assert messages == []


def test_worktree_links_are_per_run_and_removal_keeps_store_target(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    env.seed_stores()
    store_marker = default_agent_workdir("OWNER/REPO", "claude") / ".venv" / "bin" / "python"
    proc_a, paths_a = env.child(extra=("--worktree-link", ".venv"))
    try:
        a_link = Path(paths_a["claude"]) / ".venv"
        config_b = env.config(extra=("--worktree-link", ".venv"))
        with workdir_claim_scope(command="task"):
            prepare_inline(config_b)
            b_link = config_b.claude_dir / ".venv"
            assert a_link.is_symlink() and b_link.is_symlink()
            assert a_link != b_link
            assert os.lstat(a_link).st_ino != os.lstat(b_link).st_ino
            assert os.path.realpath(a_link) == os.path.realpath(b_link)
            a_root = Path(paths_a["claude"])
            stop(proc_a)
            assert not a_root.exists()
            assert store_marker.is_file()
            assert (b_link / "bin" / "python").is_file()
    finally:
        if proc_a.poll() is None:
            stop(proc_a)


def _record_imports(monkeypatch):
    from coding_review_agent_loop import git_transport

    calls = []
    real_import = git_transport.import_ref

    def recording_import(destination, source_ref, tracking_ref, **kwargs):
        calls.append((source_ref, kwargs["ssh_identity"], kwargs["forbidden_roots"]))
        return real_import(destination, source_ref, tracking_ref, **kwargs)

    monkeypatch.setattr(git_transport, "import_ref", recording_import)
    return calls


def test_transport_forbidden_roots_cover_every_agent_location(tmp_path, monkeypatch):
    """#1380: SSH config, identities and known-hosts are refused in any agent-controlled root."""
    from coding_review_agent_loop.config import default_run_worktree

    env = Env(tmp_path, monkeypatch)
    explicit = tmp_path / "explicit-codex"
    seat_dir = tmp_path / "seat-checkout"
    config = env.config(extra=(
        "--codex-dir", str(explicit),
        "--reviewer-seat", "audit=gemini", "--seat-model", "audit=gemini-test",
        "--seat-dir", f"audit={seat_dir}",
        "--reviewer-seat", "second=claude", "--seat-model", "second=claude-test",
    ))
    roots = set(config.transport_forbidden_roots)
    expected = {
        config.claude_dir, explicit.resolve(), config.gemini_dir, config.antigravity_dir,
        seat_dir.resolve(),
        default_run_worktree("OWNER/REPO", "second", config.run_token).resolve(),
        env.bare.resolve(),
    }
    for agent in ("claude", "gemini", "antigravity", "second"):
        expected.add(default_agent_workdir("OWNER/REPO", agent).resolve())
        expected.add(default_run_worktree_root("OWNER/REPO", agent).resolve())
    assert expected <= roots
    assert default_agent_workdir("OWNER/REPO", "codex").resolve() not in roots
    assert len(roots) == len(config.transport_forbidden_roots)


def test_spool_overlap_set_covers_defaulted_seat_workdirs(tmp_path, monkeypatch):
    """#1566: a seat without --seat-dir contributes its default per-run worktree."""
    from dataclasses import replace

    from coding_review_agent_loop import review_rounds
    from coding_review_agent_loop.config import default_run_worktree

    env = Env(tmp_path, monkeypatch)
    seat_dir = tmp_path / "seat-checkout"
    config = env.config(extra=(
        "--reviewer-seat", "audit=gemini", "--seat-model", "audit=gemini-test",
        "--seat-dir", f"audit={seat_dir}",
        "--reviewer-seat", "second=claude", "--seat-model", "second=claude-test",
    ))
    defaulted = default_run_worktree("OWNER/REPO", "second", config.run_token).resolve()
    entries = review_rounds._spool_overlap_set(config)
    assert ("reviewer seat second workdir", os.path.abspath(str(defaulted))) in entries
    assert ("reviewer seat audit workdir", os.path.abspath(str(seat_dir.resolve()))) in entries

    misplaced = replace(config, agent_memory_dir=defaulted / "cache" / "memory")
    with pytest.raises(AgentLoopError, match="overlaps") as info:
        review_rounds.validate_review_spool_placement(misplaced)
    assert "reviewer seat second workdir" in str(info.value)


def test_every_import_call_site_forwards_identity_and_shared_roots(tmp_path, monkeypatch):
    """Both run_worktrees.py and all three config.py import_ref sites forward both values."""
    from dataclasses import replace

    from coding_review_agent_loop.config import sync_coder_base_before_implementation

    env = Env(tmp_path, monkeypatch)
    explicit = tmp_path / "explicit-claude"
    subprocess.run(["git", "clone", "-q", str(env.bare), str(explicit)], check=True)
    identity = Path("/keys/operator_key")
    store_config = replace(env.config(), git_ssh_identity=identity)
    explicit_config = replace(env.config(extra=("--claude-dir", str(explicit))),
                              git_ssh_identity=identity)
    sha = env.commit("b.txt", "two")
    env.set_pr_head(7, sha)
    calls = _record_imports(monkeypatch)

    # run_worktrees.py: _fetch_and_pin and pin_pr_head.
    store = default_agent_workdir("OWNER/REPO", "claude").resolve()
    assert run_worktrees.prepare_store(store, config=store_config, runner=Runner()) == sha
    assert run_worktrees.pin_pr_head(store, 7, config=store_config, runner=Runner()) == sha
    assert [ref for ref, *_ in calls] == ["refs/heads/main", "refs/heads/main", "refs/pull/7/head"]
    assert all(call[1:] == (identity, store_config.transport_forbidden_roots) for call in calls)

    # config.py: _sync_base_branch and both sync_checkout_to_pr imports.
    calls.clear()
    sync_coder_base_before_implementation(explicit_config, Runner())
    sync_checkout_to_pr(
        explicit_config, Runner(), path=explicit, label="--claude-dir", default_owned=False,
        pr_number=7, pr_metadata=SimpleNamespace(head_sha=sha),
    )
    assert [ref for ref, *_ in calls] == ["refs/heads/main", "refs/heads/main", "refs/pull/7/head"]
    assert explicit.resolve() in explicit_config.transport_forbidden_roots
    assert all(call[1:] == (identity, explicit_config.transport_forbidden_roots) for call in calls)

    # Test doubles without the new fields keep working with empty defaults.
    calls.clear()
    doubles = SimpleNamespace(repo="OWNER/REPO", gh_cmd=str(env.gh), base="main",
                              trusted_local_origin=env.bare)
    run_worktrees._fetch_and_pin(store, "main", config=doubles, runner=Runner(), free=False)
    assert calls == [("refs/heads/main", None, ())]


def _gemini_turn_in(config, checkout):
    """Allocate a Gemini-style response turn in ``checkout``'s git directory."""
    from coding_review_agent_loop.agents.base import public_response_path
    from coding_review_agent_loop.agents.gemini import _gemini_public_response_root

    root = _gemini_public_response_root(checkout)
    path = public_response_path(config, "gemini", root=root)
    path.write_text("gemini response", encoding="utf-8")
    legacy = root / "gemini" / "0123456789abcdef.md"
    legacy.write_text("legacy", encoding="utf-8")
    return root


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory read permission")
def test_hidden_gemini_response_root_does_not_block_worktree_removal_or_prune(tmp_path, monkeypatch):
    """A non-listable response root in a worktree admin dir is released before removal (#1560)."""
    env = Env(tmp_path, monkeypatch)
    store = default_agent_workdir("OWNER/REPO", "claude").resolve()

    config = env.config()
    with workdir_claim_scope(command="task"):
        claim_agent_workdirs(config)
        ensure_agent_workdirs(config, Runner())
        path = config.claude_dir
        root = _gemini_turn_in(config, path)
        admin = root.parent.parent
        assert admin.parent == store / ".git" / "worktrees"
        for hidden in (root, root / "gemini"):
            with pytest.raises(PermissionError):
                os.listdir(hidden)
    # Run end removed the worktree and git's admin directory despite the 0o300 tree.
    assert not path.exists()
    assert not admin.exists()
    assert str(path) not in git(store, "worktree", "list", "--porcelain")

    # A worktree that vanished without cleanup is pruned at the next startup.
    orphan_config = env.config()
    with workdir_claim_scope(command="task"):
        claim_agent_workdirs(orphan_config)
        ensure_agent_workdirs(orphan_config, Runner())
        orphan = orphan_config.claude_dir
        orphan_root = _gemini_turn_in(orphan_config, orphan)
        orphan_admin = orphan_root.parent.parent
        shutil.rmtree(orphan)
    assert orphan_admin.exists()
    prepare(env.config(), Runner())
    assert not orphan_admin.exists()


# --- stage-2 diagnostics (#1651) ---------------------------------------------


def test_replaced_worktree_link_refusal_names_store_source_and_remedy(tmp_path, monkeypatch):
    from coding_review_agent_loop import checkout_verification as cv
    from coding_review_agent_loop.errors import CheckoutVerificationError

    env = Env(tmp_path, monkeypatch)
    commit_gitignore(env, "/.venv\n")
    env.seed_stores()
    config = env.config(extra=("--worktree-link", ".venv"))
    with workdir_claim_scope(command="task"):
        runner = prepare_inline(config)
        root = config.claude_dir
        store = default_agent_workdir("OWNER/REPO", "claude")
        cv.verify_checkout(config, runner, path=root, agent="claude", purpose="agent turn")
        other = tmp_path / "private-venv"
        other.mkdir()
        (root / ".venv").unlink()
        (root / ".venv").symlink_to(other)
        with pytest.raises(CheckoutVerificationError) as direct:
            cv.check_worktree_links(config, runner, root)
        cv.record_checkout_baseline(config, runner, root, source="agent turn")
        with pytest.raises(CheckoutVerificationError) as poisoned:
            cv.verify_checkout(config, runner, path=root, agent="claude", purpose="agent turn")
    for message in (str(direct.value), str(poisoned.value)):
        assert "worktree link .venv was removed, retargeted or reached through a symlink" in message
        assert str(store / ".venv") in message
        assert "--worktree-setup-pr-local" in message
        assert "--<agent>-dir checkout" in message
    assert "can no longer be trusted" in str(poisoned.value)


def _push_branch_from(root: Path, branch: str) -> None:
    git(root, "checkout", "-q", "-b", branch)
    (root / "work.txt").write_text("work")
    git(root, "add", "work.txt")
    git(root, "commit", "-q", "-m", "work")


def test_poisoned_after_push_stop_prints_resume_and_cleanup(tmp_path, monkeypatch):
    from coding_review_agent_loop import checkout_verification as cv
    from coding_review_agent_loop import issue_loop
    from coding_review_agent_loop.errors import CheckoutVerificationError
    from coding_review_agent_loop.worktree_hints import add_pushed_branch_recovery

    env = Env(tmp_path, monkeypatch)
    env.seed_stores()
    config = env.config()
    branch = "agent-loop/managed-1651"
    with workdir_claim_scope(command="task"):
        runner = prepare_inline(config)
        root = config.claude_dir
        store = default_agent_workdir("OWNER/REPO", "claude")
        _push_branch_from(root, branch)
        cv.poison_checkout(root, "baseline capture after agent turn failed: link replaced")
        with pytest.raises(CheckoutVerificationError) as caught:
            cv.verify_checkout(config, runner, path=root, agent="claude", purpose="agent turn")
        # Not pushed yet: no recovery text, the refusal is unchanged.
        original = str(caught.value)
        assert add_pushed_branch_recovery(caught.value, runner=runner, config=config, issue_number=1651) is False
        assert str(caught.value) == original
        git(root, "push", "-q", "origin", branch)
        recovered = CheckoutVerificationError(original)
        assert add_pushed_branch_recovery(recovered, runner=runner, config=config, issue_number=1651) is True
        message = str(recovered)
        assert message.startswith(original)
        assert f"agent-loop managed-pr --head {branch} --expected-closing-issue 1651" in message
        assert f"git -C {store} worktree remove --force {root}" in message
        # Applied once only; a terminal no-PR stop (plain AgentLoopError) gets it too.
        assert add_pushed_branch_recovery(recovered, runner=runner, config=config, issue_number=1651) is False
        assert str(recovered) == message
        no_pr = AgentLoopError("Coder did not create a valid PR; implementation is blocking.")
        assert add_pushed_branch_recovery(no_pr, runner=runner, config=config, issue_number=1651) is True
        assert type(no_pr) is AgentLoopError and f"--head {branch}" in str(no_pr)

        # The issue-run stop carries the same text, and keeps refusing.
        def refuse(*_args, **_kwargs):
            raise caught.value

        monkeypatch.setattr(issue_loop, "_normalize_requested_execution_policy", refuse)
        with pytest.raises(CheckoutVerificationError) as stopped:
            issue_loop.run_issue_loop(runner, issue_number=1651, config=config)
        assert f"--head {branch} --expected-closing-issue 1651" in str(stopped.value)
        assert "worktree remove --force" in str(stopped.value)
        # The worktree is never removed automatically.
        assert root.is_dir() and (root / "work.txt").exists()
        assert cv.has_poisoned_entry(root)


def _pre_1670_recovery(*, issue_number: int, branch: str, store: Path, worktree: Path) -> str:
    return (
        f"Recovery after a pushed branch: branch {branch} was already pushed, so the work "
        f"is not lost. Resume with `agent-loop managed-pr --head {branch} --expected-closing-issue "
        f"{issue_number}` plus your usual --managed-ci/--auto-merge and "
        "--managed-ci-trusted-actor options (or `agent-loop pr <PR> --expected-closing-issue "
        f"{issue_number}` when a PR for that branch already exists). The untrusted run "
        "worktree is left in place for inspection; agent-loop does not remove it. Remove it "
        f"manually with `git -C {store} worktree remove --force {worktree}`."
    )


@pytest.mark.parametrize("state", ["ahead", "equal", "behind", "count-fails"])
def test_pushed_branch_recovery_compares_local_tip_with_tracking_ref(tmp_path, monkeypatch, state):
    """Local-only commits after the push are named instead of claiming the work is not lost (#1670)."""
    from coding_review_agent_loop import checkout_verification as cv
    from coding_review_agent_loop import worktree_hints
    from coding_review_agent_loop.errors import CheckoutVerificationError

    env = Env(tmp_path, monkeypatch)
    env.seed_stores()
    config = env.config()
    branch = "agent-loop/managed-1670"
    with workdir_claim_scope(command="task"):
        runner = prepare_inline(config)
        root = config.claude_dir
        store = default_agent_workdir("OWNER/REPO", "claude")
        _push_branch_from(root, branch)
        git(root, "push", "-q", "origin", branch)
        if state == "ahead":
            for name in ("later-1.txt", "later-2.txt"):
                (root / name).write_text(name)
                git(root, "add", name)
                git(root, "commit", "-q", "-m", name)
        elif state == "behind":
            pushed = git(root, "rev-parse", "HEAD")
            (root / "later.txt").write_text("later")
            git(root, "add", "later.txt")
            git(root, "commit", "-q", "-m", "later")
            git(root, "push", "-q", "origin", branch)
            git(root, "reset", "-q", "--hard", pushed)
        elif state == "count-fails":
            real_probe = cv._probe

            def failing_probe(config_, runner_, path, args):
                if args[:1] == ("rev-list",):
                    raise cv._Incomplete("probe `git rev-list` exited 128")
                return real_probe(config_, runner_, path, args)

            monkeypatch.setattr(cv, "_probe", failing_probe)
        head_before = git(root, "rev-parse", "HEAD")
        branches_before = git(store, "branch", "--list")
        cv.poison_checkout(root, "baseline capture after agent turn failed: link replaced")
        error = CheckoutVerificationError("changed outside agent-loop")
        assert worktree_hints.add_pushed_branch_recovery(
            error, runner=runner, config=config, issue_number=1670
        ) is True
        assert type(error) is CheckoutVerificationError
        text = str(error).split("\n", 1)[1]
        # Read-only: no Git write and no worktree removal.
        assert git(root, "rev-parse", "HEAD") == head_before
        assert git(store, "branch", "--list") == branches_before
        assert root.is_dir()
    if state in ("equal", "behind"):
        assert text == _pre_1670_recovery(issue_number=1670, branch=branch, store=store, worktree=root)
        return
    assert "is not lost" not in text
    assert f"agent-loop managed-pr --head {branch} --expected-closing-issue 1670" in text
    assert f"git -C {store} worktree remove --force {root}" in text
    assert (
        "discards uncommitted changes in it but keeps the local branch "
        f"{branch} and its commits in the shared repository"
    ) in text
    assert "git push" not in text  # never advise pushing from the untrusted worktree
    if state == "ahead":
        assert f"the local branch has 2 commits that are not on origin/{branch}" in text
        assert "resuming from the pushed branch omits them" in text
    else:
        assert f"could not be compared with origin/{branch}" in text
        assert (
            f"`git -C {store} --no-pager log refs/remotes/origin/{branch}..refs/heads/{branch}`"
        ) in text
        # The store-rooted check resolves both refs in the shared repository.
        advised = subprocess.run(
            ["git", "-C", str(store), "--no-pager", "log",
             f"refs/remotes/origin/{branch}..refs/heads/{branch}"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
        assert advised.returncode == 0, advised.stderr
    # No command is ever advised inside the untrusted worktree except removal from the store.
    assert f"git -C {root}" not in text


def test_unpoisoned_checkout_gets_no_pushed_branch_recovery(tmp_path, monkeypatch):
    from coding_review_agent_loop.errors import CheckoutVerificationError
    from coding_review_agent_loop.worktree_hints import add_pushed_branch_recovery

    env = Env(tmp_path, monkeypatch)
    env.seed_stores()
    config = env.config()
    with workdir_claim_scope(command="task"):
        runner = prepare_inline(config)
        _push_branch_from(config.claude_dir, "feature-x")
        git(config.claude_dir, "push", "-q", "origin", "feature-x")
        error = CheckoutVerificationError("changed outside agent-loop")
        assert add_pushed_branch_recovery(error, runner=runner, config=config, issue_number=7) is False
        assert str(error) == "changed outside agent-loop"


@pytest.mark.parametrize('replace_link', [False, True])
def test_shared_source_damage_poison_and_zero_backend_launches(tmp_path, monkeypatch, replace_link):
    from coding_review_agent_loop import checkout_verification as cv, worktree_link_integrity as health
    from coding_review_agent_loop.agents import registry
    from coding_review_agent_loop.errors import CheckoutVerificationError
    env = Env(tmp_path, monkeypatch)
    env.commit('.gitignore', '/.venv\n')
    env.seed_stores()
    config = env.config(extra=('--worktree-link', '.venv'))
    with workdir_claim_scope(command='task'):
        claim_agent_workdirs(config)
        ensure_agent_workdirs(config, Runner())
        root = config.claude_dir
        store = default_agent_workdir('OWNER/REPO', 'claude')
        source = store / '.venv'
        if replace_link:
            (root / '.venv').unlink()
            (root / '.venv').mkdir()
            # An already poisoned checkout still inspects the original source.
            cv.poison_checkout(root, 'original link violation')
        shutil.rmtree(source / 'bin')
        cv.record_checkout_baseline(config, Runner(), root, source='coder')
        assert cv.has_poisoned_entry(root)
        assert health.inspect(store, ('.venv',))[1]
        other = default_agent_workdir('OWNER/REPO', 'codex')
        assert health.inspect(other, ('.venv',))[1] is None
        launches = []
        backend = registry.get_backend('claude')
        monkeypatch.setattr(type(backend), 'run', lambda *a, **k: launches.append(1))
        for attempt in range(2):
            with pytest.raises(CheckoutVerificationError) as caught:
                registry.run_agent_result(Runner(), agent='claude', config=config,
                                          prompt='x', role='coder', attempt_suffix=str(attempt))
            message = str(caught.value)
            assert str(store.absolute()) in message and str(source.absolute()) in message
            assert '--worktree-setup CMD' in message and '--worktree-setup-pr-local' in message
        assert launches == []


def test_replaced_link_alone_keeps_shared_health_usable(tmp_path, monkeypatch):
    from coding_review_agent_loop import checkout_verification as cv, worktree_link_integrity as health
    env = Env(tmp_path, monkeypatch)
    env.commit('.gitignore', '/.venv\n')
    env.seed_stores()
    config = env.config(extra=('--worktree-link', '.venv'))
    with workdir_claim_scope(command='task'):
        claim_agent_workdirs(config)
        ensure_agent_workdirs(config, Runner())
        root = config.claude_dir
        (root / '.venv').unlink()
        (root / '.venv').mkdir()
        cv.record_checkout_baseline(config, Runner(), root, source='coder')
        assert cv.has_poisoned_entry(root)
        store = default_agent_workdir('OWNER/REPO', 'claude')
        assert health.inspect(store, ('.venv',))[1] is None
