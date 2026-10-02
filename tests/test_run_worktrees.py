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
    "--base", "main", "--gh-cmd", gh, "--claude-cmd", "/bin/true",
    "--codex-cmd", "/bin/true", "--antigravity-cmd", "/bin/true", "--quiet",
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
            "--base", "main", "--gh-cmd", str(self.gh), "--claude-cmd", "/bin/true",
            "--codex-cmd", "/bin/true", "--antigravity-cmd", "/bin/true", "--quiet", *extra,
        ])
        return config_from_args(args, Runner())

    def child(self, *, coder="claude", reviewer="codex"):
        proc = subprocess.Popen(
            [
                sys.executable, "-c", CHILD,
                str(workdir_claims.claim_root()), str(run_worktrees.store_lock_root()),
                str(self.tmp), str(self.xdg), str(self.bare), str(self.gh), coder, reviewer,
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
            with pytest.raises(AgentLoopError, match="advertises head SHA"):
                sync_checkout_to_pr(
                    config_b, Runner(), path=b_path, label="Default claude workdir",
                    default_owned=True, pr_number=7, pr_metadata=SimpleNamespace(head_sha=pinned),
                )
        assert a_path.is_dir()
    finally:
        stop(proc_a)
    assert not a_path.exists()


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
