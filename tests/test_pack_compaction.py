"""Bounded pack compaction for managed checkouts (#1578)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_review_agent_loop import run_worktrees, secure_git
from coding_review_agent_loop.config import (
    default_agent_workdir,
    sync_checkout_to_pr,
    sync_coder_base_before_implementation,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import Runner
from test_run_worktrees import Env

THRESHOLD = 3
CONFIG = SimpleNamespace(quiet=True)


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    for key, value in (
        ("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.com"),
        ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@example.com"),
        ("GIT_CONFIG_GLOBAL", os.devnull), ("GIT_CONFIG_SYSTEM", os.devnull),
    ):
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(run_worktrees, "PACK_COMPACTION_THRESHOLD", THRESHOLD)


@pytest.fixture
def logs(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(run_worktrees, "_compaction_log", lambda config, message: messages.append(message))
    return messages


def git(path: Path, *args: str, input_text: str | None = None) -> str:
    result = subprocess.run(["git", "-C", str(path), *args], input=input_text, text=True,
                            capture_output=True, check=False)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


def pack_dir(repo: Path) -> Path:
    return repo / ".git" / "objects" / "pack"


def write_pack(repo: Path, *revs: str, extra: tuple[str, ...] = ()) -> str:
    """Pack the full history of ``revs`` the way every production import does."""
    digest = git(repo, "pack-objects", "--revs", *extra, str(pack_dir(repo) / "pack"),
                 input_text="".join(f"{rev}\n" for rev in revs))
    git(repo, "prune-packed")
    return digest


def non_kept_packs(repo: Path) -> list[str]:
    return sorted(
        name for name in os.listdir(pack_dir(repo))
        if name.endswith(".pack") and not (pack_dir(repo) / name).with_suffix(".keep").exists()
    )


def listing(repo: Path) -> dict[str, bytes]:
    return {name: (pack_dir(repo) / name).read_bytes()
            for name in sorted(os.listdir(pack_dir(repo)))}


def unreferenced_commit(repo: Path, name: str) -> str:
    """A commit with its own blob and tree that no ref, reflog or index references."""
    blob = git(repo, "hash-object", "-w", "--stdin", input_text=f"{name}\n")
    tree = git(repo, "mktree", input_text=f"100644 blob {blob}\t{name}.txt\n")
    return git(repo, "commit-tree", tree, "-p", "HEAD", "-m", name)


class Repo:
    """A checkout with one pack per commit and a packed, now-unreferenced PR head."""

    def __init__(self, tmp_path: Path, *, packs: int = THRESHOLD + 2, bitmap: bool = False):
        self.path = tmp_path / "checkout"
        subprocess.run(["git", "init", "-q", "-b", "main", str(self.path)], check=True)
        self.commits: list[str] = []
        for index in range(packs):
            (self.path / "a.txt").write_text(f"{index}\n")
            git(self.path, "add", "a.txt")
            git(self.path, "commit", "-q", "-m", f"c{index}")
            self.commits.append(git(self.path, "rev-parse", "HEAD"))
            write_pack(self.path, "HEAD")
        self.pr_head = unreferenced_commit(self.path, "pr")
        git(self.path, "update-ref", "refs/remotes/origin/pr/7", self.pr_head)
        write_pack(self.path, "refs/remotes/origin/pr/7", "^HEAD")
        git(self.path, "update-ref", "-d", "refs/remotes/origin/pr/7")
        git(self.path, "reflog", "expire", "--expire=now", "--all")
        if bitmap:
            write_pack(self.path, extra=("--all", "--write-bitmap-index"))
            assert any(name.endswith(".bitmap") for name in os.listdir(pack_dir(self.path)))

    def objects(self) -> set[str]:
        ids: set[str] = set()
        for name in os.listdir(pack_dir(self.path)):
            if name.endswith(".idx"):
                with open(pack_dir(self.path) / name, "rb") as handle:
                    raw = subprocess.run(["git", "show-index"], stdin=handle,
                                         capture_output=True, check=True).stdout.decode()
                ids |= {line.split()[1] for line in raw.splitlines()}
        return ids

    def assert_all_present(self, ids) -> None:
        for sha in ids:
            git(self.path, "cat-file", "-e", sha)

    def compact(self, runner=None) -> None:
        with run_worktrees.store_lock(self.path):
            run_worktrees.compact_packs_if_needed(self.path, config=CONFIG, runner=runner or Runner())


def _local_command_spy(monkeypatch) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []
    real = run_worktrees.local_command

    def spy(args, **kwargs):
        calls.append(tuple(args))
        return real(args, **kwargs)

    monkeypatch.setattr(run_worktrees, "local_command", spy)
    return calls


# --- above the threshold -----------------------------------------------------


@pytest.mark.parametrize("bitmap", [False, True], ids=["plain", "bitmap"])
def test_over_threshold_compacts_to_one_pack_and_keeps_every_object(tmp_path, monkeypatch, logs, bitmap):
    repo = Repo(tmp_path, bitmap=bitmap)
    before = repo.objects()
    assert len(non_kept_packs(repo.path)) > THRESHOLD
    calls = _local_command_spy(monkeypatch)

    repo.compact()

    assert logs and not any("skipped" in message for message in logs), logs
    assert len(non_kept_packs(repo.path)) == 1
    repo.assert_all_present(before)
    repo.assert_all_present([*repo.commits, repo.pr_head])
    git(repo.path, "fsck", "--connectivity-only")
    # Only builtin, exec-free commands ran through the confined boundary.
    assert {call[0] for call in calls} == {"show-index", "pack-objects", "index-pack"}
    pack_objects = next(call for call in calls if call[0] == "pack-objects")
    assert "--no-use-bitmap-index" in pack_objects and "--keep-unreachable" in pack_objects
    assert "--honor-pack-keep" not in pack_objects
    assert not any(name.startswith((".agent-loop-compact-", "tmp_")) for name in os.listdir(pack_dir(repo.path)))


def test_prepare_and_pin_compact_a_real_shared_store(tmp_path, monkeypatch, logs):
    env = Env(tmp_path, monkeypatch)
    config = env.config()
    store = env.seed_store("claude").resolve()
    shas = [run_worktrees.prepare_store(store, config=config, runner=Runner())]
    # A PR head off main, fetched by SHA and then no longer referenced locally.
    pr_head = unreferenced_commit(env.seed, "pr")
    git(env.seed, "push", "-q", "origin", f"{pr_head}:refs/pull/7/head")
    assert run_worktrees.pin_pr_head(store, 7, config=config, runner=Runner()) == pr_head
    git(store, "update-ref", "-d", "refs/remotes/origin/pr/7")
    git(store, "reflog", "expire", "--expire=now", "--all")
    logs.clear()
    # Each new base commit makes the next import write a distinct pack.
    for index in range(THRESHOLD + 1):
        shas.append(env.commit("a.txt", f"base{index}"))
        assert run_worktrees.prepare_store(store, config=config, runner=Runner()) == shas[-1]
    assert any(message.startswith("Compacted") for message in logs), logs
    assert len(non_kept_packs(store)) <= THRESHOLD
    for sha in [*shas, pr_head]:
        git(store, "cat-file", "-e", sha)
    git(store, "fsck", "--connectivity-only")


# --- at or below the threshold -------------------------------------------------


def test_at_threshold_does_nothing(tmp_path, monkeypatch, logs):
    repo = Repo(tmp_path, packs=THRESHOLD - 1)
    assert len(non_kept_packs(repo.path)) == THRESHOLD
    before = listing(repo.path)
    calls = _local_command_spy(monkeypatch)

    repo.compact()

    # Every compaction subprocess goes through local_command: none was built.
    assert calls == [] and logs == []
    assert listing(repo.path) == before


def test_dry_run_returns_immediately(tmp_path, monkeypatch, logs):
    repo = Repo(tmp_path)
    before = listing(repo.path)
    calls = _local_command_spy(monkeypatch)
    repo.compact(runner=Runner(dry_run=True))
    assert calls == [] and logs == []
    assert listing(repo.path) == before


def test_lock_not_held_raises_before_any_git_call(tmp_path, monkeypatch, logs):
    repo = Repo(tmp_path)
    calls = _local_command_spy(monkeypatch)
    runner_calls = []

    class Recording(Runner):
        def run(self, args, **kwargs):
            runner_calls.append(args)
            return super().run(args, **kwargs)

    with pytest.raises(AgentLoopError, match="store lock"):
        run_worktrees.compact_packs_if_needed(repo.path, config=CONFIG, runner=Recording())
    assert calls == [] and runner_calls == [] and logs == []


# --- failures ------------------------------------------------------------------


def _failing_subprocess(monkeypatch, *, command: str, mode: str):
    real = subprocess.run

    def fake(argv, **kwargs):
        if command in argv:
            if command == "pack-objects":
                kwargs["stdout"].write(b"PACK\x00\x00\x00\x02partial")
                kwargs["stdout"].flush()
            if mode == "timeout":
                raise subprocess.TimeoutExpired(argv, 1)
            return subprocess.CompletedProcess(argv, 7, b"", b"secret stderr")
        return real(argv, **kwargs)

    monkeypatch.setattr(run_worktrees.subprocess, "run", fake)


@pytest.mark.parametrize(("command", "mode", "reason"), [
    ("pack-objects", "exit", "pack-objects failed (exit 7)"),
    ("pack-objects", "timeout", "pack-objects timed out"),
    ("index-pack", "exit", "index-pack failed (exit 7)"),
])
def test_command_failure_is_logged_and_keeps_old_packs(tmp_path, monkeypatch, logs, command, mode, reason):
    repo = Repo(tmp_path)
    before = listing(repo.path)
    _failing_subprocess(monkeypatch, command=command, mode=mode)

    repo.compact()

    assert logs == [f"Pack compaction for {repo.path} skipped: {reason}"]
    assert listing(repo.path) == before


def test_verification_mismatch_is_logged_and_keeps_old_packs(tmp_path, monkeypatch, logs):
    repo = Repo(tmp_path)
    before = listing(repo.path)
    real = run_worktrees._pack_index_ids

    def lossy(idx, **kwargs):
        ids = real(idx, **kwargs)
        if idx.name == "new.idx":
            ids.discard(repo.pr_head)
        return ids

    monkeypatch.setattr(run_worktrees, "_pack_index_ids", lossy)

    repo.compact()

    assert logs == [f"Pack compaction for {repo.path} skipped: verification mismatch"]
    assert listing(repo.path) == before


def test_fetch_path_returns_its_sha_when_compaction_fails(tmp_path, monkeypatch, logs):
    env = Env(tmp_path, monkeypatch)
    config = env.config()
    store = default_agent_workdir("OWNER/REPO", "claude").resolve()
    sha = run_worktrees.prepare_store(store, config=config, runner=Runner())
    env.set_pr_head(7, sha)

    def broken(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(run_worktrees, "_compact", broken)
    assert run_worktrees.prepare_store(store, config=config, runner=Runner()) == sha
    assert run_worktrees.pin_pr_head(store, 7, config=config, runner=Runner()) == sha
    assert logs == [f"Pack compaction for {store} skipped: unexpected OSError"] * 2


# --- kept and concurrent packs ---------------------------------------------------


def test_kept_pack_is_untouched_and_post_snapshot_pack_survives(tmp_path, monkeypatch, logs):
    repo = Repo(tmp_path)
    # A kept pack sharing history with the snapshot packs.
    kept = write_pack(repo.path, repo.commits[1])
    (pack_dir(repo.path) / f"pack-{kept}.keep").write_text("")
    kept_files = {suffix: (pack_dir(repo.path) / f"pack-{kept}{suffix}").read_bytes()
                  for suffix in (".pack", ".idx", ".keep")}
    before = repo.objects()
    late_holder: dict[str, str] = {}
    real = run_worktrees._pack_index_ids

    def inject(idx, **kwargs):
        if idx.name == "new.idx" and not late_holder:
            late = unreferenced_commit(repo.path, "late")
            late_holder["sha"] = late
            late_holder["pack"] = write_pack(repo.path, late, "^HEAD")
        return real(idx, **kwargs)

    monkeypatch.setattr(run_worktrees, "_pack_index_ids", inject)

    repo.compact()

    assert not any("skipped" in message for message in logs), logs
    remaining = non_kept_packs(repo.path)
    assert len(remaining) == 2 and f"pack-{late_holder['pack']}.pack" in remaining
    for suffix, data in kept_files.items():
        assert (pack_dir(repo.path) / f"pack-{kept}{suffix}").read_bytes() == data
    repo.assert_all_present([*before, late_holder["sha"]])


# --- call-site wiring --------------------------------------------------------------


def test_every_managed_fetch_site_compacts_under_its_lock(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    explicit = tmp_path / "explicit-claude"
    subprocess.run(["git", "clone", "-q", str(env.bare), str(explicit)], check=True)
    store_config = env.config()
    explicit_config = env.config(extra=("--claude-dir", str(explicit)))
    sha = env.commit("b.txt", "two")
    env.set_pr_head(7, sha)
    seen: list[tuple[Path, bool]] = []

    def stub(checkout, *, config, runner):
        seen.append((Path(checkout), run_worktrees.store_key(checkout) in run_worktrees._held_keys()))

    monkeypatch.setattr(run_worktrees, "compact_packs_if_needed", stub)
    store = default_agent_workdir("OWNER/REPO", "claude").resolve()
    assert run_worktrees.prepare_store(store, config=store_config, runner=Runner()) == sha
    assert run_worktrees.pin_pr_head(store, 7, config=store_config, runner=Runner()) == sha
    sync_coder_base_before_implementation(explicit_config, Runner())
    sync_checkout_to_pr(
        explicit_config, Runner(), path=explicit, label="--claude-dir", default_owned=False,
        pr_number=7, pr_metadata=SimpleNamespace(head_sha=sha),
    )
    assert git(explicit, "rev-parse", "HEAD") == sha
    assert [(path.resolve(), held) for path, held in seen] == [
        (store, True), (store, True), (explicit.resolve(), True), (explicit.resolve(), True),
    ]


# --- hardening -------------------------------------------------------------------


def test_hardening_config_is_unchanged():
    assert secure_git._CONFIG == (
        "core.hooksPath=/dev/null", "core.fsmonitor=false", "core.pager=cat",
        "core.sshCommand=ssh", "diff.external=", "diff.trustExitCode=false",
        "log.showSignature=false", "gpg.program=false", "gpg.ssh.program=false",
        "gpg.x509.program=false", "protocol.allow=never", "protocol.ext.allow=never",
        "submodule.recurse=false", "status.submoduleSummary=false",
        "diff.ignoreSubmodules=all", "gc.auto=0", "credential.helper=",
        "trace2.eventTarget=", "trace2.perfTarget=", "trace2.normalTarget=",
    )
