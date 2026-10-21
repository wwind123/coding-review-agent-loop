"""Bounded pack compaction for managed checkouts (#1578)."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
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
        # Newer Git (2.55 on CI) runs auto maintenance after porcelain commands
        # such as commit, which repacks fixture packs before the test starts.
        # Only the fixture's unconfined git sees this; confined calls rebuild
        # their environment from an allowlist.
        ("GIT_CONFIG_COUNT", "2"),
        ("GIT_CONFIG_KEY_0", "gc.auto"), ("GIT_CONFIG_VALUE_0", "0"),
        ("GIT_CONFIG_KEY_1", "maintenance.auto"), ("GIT_CONFIG_VALUE_1", "false"),
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

    def __init__(self, tmp_path: Path, *, packs: int = THRESHOLD + 2, bitmap: bool = False,
                 name: str = "checkout"):
        self.path = tmp_path / name
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
        # One pack per commit plus the PR pack (the bitmapped pack has the same
        # content, so the same name, as the last commit's pack): fail setup
        # loudly if anything merged them behind the fixture's back.
        assert len(non_kept_packs(self.path)) == packs + 1

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


def marker(checkout: Path) -> Path:
    return run_worktrees._backoff_marker_path(checkout)


def read_marker(checkout: Path) -> dict:
    return json.loads(marker(checkout).read_text())


def plant_marker(checkout: Path, payload) -> Path:
    path = marker(checkout)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


@pytest.fixture
def clock(monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(run_worktrees, "_now", lambda: now[0])
    return now


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
    assert not marker(repo.path).exists()


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
    assert not marker(repo.path).exists()


def test_guards_leave_an_existing_marker_untouched(tmp_path, monkeypatch, logs, clock):
    repo = Repo(tmp_path)
    run_worktrees._write_backoff(repo.path, 2, clock[0] - 10)
    planted = marker(repo.path).read_bytes()
    with pytest.raises(AgentLoopError, match="store lock"):
        run_worktrees.compact_packs_if_needed(repo.path, config=CONFIG, runner=Runner())
    repo.compact(runner=Runner(dry_run=True))
    assert logs == [] and marker(repo.path).read_bytes() == planted


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
    ("index-pack", "exit", "index-pack failed (exit 7)"),
])
def test_command_failure_is_logged_and_keeps_old_packs(tmp_path, monkeypatch, logs, command, mode, reason):
    repo = Repo(tmp_path)
    before = listing(repo.path)
    _failing_subprocess(monkeypatch, command=command, mode=mode)

    repo.compact()

    assert logs == [f"Pack compaction for {repo.path} skipped: {reason}"]
    assert listing(repo.path) == before
    # Only a timeout starts a backoff (#1580).
    assert not marker(repo.path).exists()


# --- timeout backoff (#1580) ------------------------------------------------------


def _hint(checkout: Path) -> str:
    return (f"to compact manually run `git -C {shlex.quote(str(checkout))} repack -a -d --keep-unreachable` "
            "while no agent-loop run uses this store")


@pytest.mark.parametrize("command", ["pack-objects", "index-pack", "show-index"])
def test_timeout_records_backoff_and_logs_the_manual_repack_hint(tmp_path, monkeypatch, logs, clock, command):
    repo = Repo(tmp_path)
    before = listing(repo.path)
    _failing_subprocess(monkeypatch, command=command, mode="timeout")

    repo.compact()

    assert logs == [
        f"Pack compaction for {repo.path} skipped: {command} timed out; backing off 1 h "
        f"(consecutive timeouts: 1); {_hint(repo.path)}"
    ]
    assert "secret stderr" not in logs[0]
    assert listing(repo.path) == before
    assert read_marker(repo.path) == {"version": 1, "failures": 1, "next_attempt_at": clock[0] + 3600}
    info = marker(repo.path).lstat()
    assert (info.st_mode & 0o777) == 0o600 and info.st_uid == os.getuid()
    assert [name for name in os.listdir(marker(repo.path).parent) if name.endswith(".tmp")] == []


@pytest.mark.parametrize("name", ["my store", "it's $(touch pwned); `id` & x"],
                         ids=["space", "metacharacters"])
def test_manual_repack_hint_shell_quotes_the_checkout_in_both_logs(tmp_path, monkeypatch, logs, clock, name):
    repo = Repo(tmp_path, name=name)
    _failing_subprocess(monkeypatch, command="pack-objects", mode="timeout")
    repo.compact()
    repo.compact()

    assert len(logs) == 2 and "backing off" in logs[0] and "backoff after" in logs[1]
    for message in logs:
        command = message.split("to compact manually run `", 1)[1].rsplit("` while no agent-loop run", 1)[0]
        assert shlex.split(command) == ["git", "-C", str(repo.path), "repack", "-a", "-d", "--keep-unreachable"]


def test_skip_while_backoff_is_active_runs_no_pack_command(tmp_path, monkeypatch, logs, clock):
    repo = Repo(tmp_path)
    run_worktrees._write_backoff(repo.path, 1, clock[0])
    planted = marker(repo.path).read_bytes()
    before = listing(repo.path)
    clock[0] += 3599
    calls = _local_command_spy(monkeypatch)

    repo.compact()

    assert calls == []
    assert logs == [
        f"Pack compaction for {repo.path} skipped: backoff after 1 consecutive timeout(s); "
        f"next attempt after 2027-01-15T09:00:00Z; {_hint(repo.path)}"
    ]
    assert listing(repo.path) == before
    assert marker(repo.path).read_bytes() == planted


def test_consecutive_timeouts_double_up_to_the_cap(tmp_path, monkeypatch, logs, clock):
    repo = Repo(tmp_path)
    _failing_subprocess(monkeypatch, command="pack-objects", mode="timeout")
    delays = []
    for expected in range(1, 9):
        repo.compact()
        state = read_marker(repo.path)
        assert state["failures"] == expected
        delays.append(state["next_attempt_at"] - clock[0])
        # Exactly at the deadline the window is over and the next attempt runs.
        clock[0] = state["next_attempt_at"]
    assert delays == [3600, 7200, 14400, 28800, 57600, 86400, 86400, 86400]
    assert all("backing off" in message for message in logs) and len(logs) == 8
    assert "backing off 24 h (consecutive timeouts: 8)" in logs[-1]


def test_backoff_delay_saturates_before_exponentiation():
    cap = run_worktrees.PACK_COMPACTION_BACKOFF_MAX_SECONDS
    assert run_worktrees._backoff_delay(1) == 3600
    assert run_worktrees._backoff_delay(10**11) == cap
    assert run_worktrees.PACK_COMPACTION_BACKOFF_BASE_SECONDS * 2 ** run_worktrees._BACKOFF_SATURATION_EXPONENT > cap


@pytest.mark.parametrize(("failures", "expected"), [
    (run_worktrees.PACK_COMPACTION_BACKOFF_MAX_FAILURES, run_worktrees.PACK_COMPACTION_BACKOFF_MAX_FAILURES),
    (10**11, 1),
])
def test_large_failure_counts_stay_bounded(tmp_path, monkeypatch, logs, clock, failures, expected):
    repo = Repo(tmp_path)
    plant_marker(repo.path, {"version": 1, "failures": failures, "next_attempt_at": clock[0] - 1})
    _failing_subprocess(monkeypatch, command="pack-objects", mode="timeout")

    repo.compact()

    state = read_marker(repo.path)
    assert state["failures"] == expected
    assert state["next_attempt_at"] - clock[0] == run_worktrees._backoff_delay(expected) <= 86400


def test_successful_compaction_clears_the_marker(tmp_path, monkeypatch, logs, clock):
    repo = Repo(tmp_path)
    run_worktrees._write_backoff(repo.path, 3, clock[0])
    clock[0] += 14400

    repo.compact()

    assert len(non_kept_packs(repo.path)) == 1
    assert not marker(repo.path).exists()
    assert logs and logs[-1].startswith("Compacted"), logs


def _refuse_marker_unlink(monkeypatch, checkout: Path) -> Path:
    # Only the marker refuses; pack sidecar deletion keeps working.
    path = marker(checkout)
    real_unlink = Path.unlink

    def refusing(self, *args, **kwargs):
        if self == path:
            raise PermissionError("denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refusing)
    return path


def test_successful_compaction_logs_success_when_the_marker_cannot_be_cleared(tmp_path, monkeypatch, logs, clock):
    repo = Repo(tmp_path)
    run_worktrees._write_backoff(repo.path, 3, clock[0])
    clock[0] += 14400
    path = _refuse_marker_unlink(monkeypatch, repo.path)

    repo.compact()

    assert len(non_kept_packs(repo.path)) == 1
    assert logs and logs[-1].startswith("Compacted"), logs
    assert not any("skipped" in line or "unexpected" in line for line in logs), logs
    assert path.exists()


def test_at_or_below_threshold_clears_the_marker_without_compacting(tmp_path, monkeypatch, logs, clock):
    # A manual repack brings the store back under the threshold.
    repo = Repo(tmp_path, packs=THRESHOLD - 1)
    run_worktrees._write_backoff(repo.path, 4, clock[0])
    before = listing(repo.path)
    calls = _local_command_spy(monkeypatch)

    repo.compact()

    assert calls == [] and logs == []
    assert listing(repo.path) == before
    assert not marker(repo.path).exists()


def test_at_or_below_threshold_tolerates_an_unclearable_marker(tmp_path, monkeypatch, logs, clock):
    repo = Repo(tmp_path, packs=THRESHOLD - 1)
    run_worktrees._write_backoff(repo.path, 4, clock[0])
    path = _refuse_marker_unlink(monkeypatch, repo.path)
    before = listing(repo.path)
    calls = _local_command_spy(monkeypatch)

    repo.compact()

    assert calls == [] and logs == []
    assert listing(repo.path) == before
    assert path.exists()


def _in_window(now: float) -> dict:
    return {"version": 1, "failures": 1, "next_attempt_at": now + 60}


@pytest.mark.parametrize("payload", [
    "{",
    "\xff not json",
    json.dumps(_in_window(1_800_000_000.0)) + " " * 4096,
    "[]",
    {"version": 2, "failures": 1, "next_attempt_at": 1_800_000_060.0},
    {"failures": 1, "next_attempt_at": 1_800_000_060.0},
    {"version": 1, "failures": True, "next_attempt_at": 1_800_000_060.0},
    {"version": 1, "failures": 1, "next_attempt_at": True},
    {"version": 1, "failures": "1", "next_attempt_at": 1_800_000_060.0},
    {"version": 1, "failures": 1, "next_attempt_at": "1800000060"},
    {"version": 1, "failures": 0, "next_attempt_at": 1_800_000_060.0},
    {"version": 1, "failures": run_worktrees.PACK_COMPACTION_BACKOFF_MAX_FAILURES + 1,
     "next_attempt_at": 1_800_000_060.0},
    {"version": 1, "failures": 1, "next_attempt_at": float("inf")},
    {"version": 1, "failures": 1, "next_attempt_at": float("nan")},
    # Ten years ahead: clock skew or forgery, never a valid write.
    {"version": 1, "failures": 1, "next_attempt_at": 1_800_000_000.0 + 10 * 365 * 86400},
    # An integer deadline too large for a float raises OverflowError on conversion.
    '{"version": 1, "failures": 1, "next_attempt_at": 1' + "0" * 400 + "}",
    # Deep nesting under the byte cap raises RecursionError inside json.loads.
    "[" * 1500 + "]" * 1500,
    '{"version": 1, "failures": 1, "next_attempt_at": ' + "[" * 1500 + "]" * 1500 + "}",
], ids=["truncated", "non-utf8", "oversized", "list", "version", "no-version", "bool-failures",
        "bool-deadline", "string-failures", "string-deadline", "zero", "too-many", "inf", "nan", "far-future",
        "overflowing-deadline", "deep-list", "deep-deadline"])
def test_invalid_marker_fails_open(tmp_path, monkeypatch, logs, clock, payload):
    repo = Repo(tmp_path)
    plant_marker(repo.path, payload)

    repo.compact()

    assert len(non_kept_packs(repo.path)) == 1
    assert logs and logs[-1].startswith("Compacted"), logs
    assert not marker(repo.path).exists()


def test_symlinked_marker_is_not_followed(tmp_path, monkeypatch, logs, clock):
    repo = Repo(tmp_path)
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps(_in_window(clock[0])))
    marker(repo.path).symlink_to(target)

    repo.compact()

    assert logs and logs[-1].startswith("Compacted"), logs
    assert target.read_text() == json.dumps(_in_window(clock[0]))


def _compact_in_thread(repo: Repo) -> None:
    worker = threading.Thread(target=repo.compact, daemon=True)
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "compaction blocked on the marker"


def test_writerless_fifo_marker_does_not_block_and_compaction_proceeds(tmp_path, monkeypatch, logs, clock):
    repo = Repo(tmp_path)
    os.mkfifo(marker(repo.path))

    _compact_in_thread(repo)

    assert logs and logs[-1].startswith("Compacted"), logs
    assert len(non_kept_packs(repo.path)) == 1


def test_writerless_fifo_marker_is_replaced_by_a_regular_marker_on_timeout(tmp_path, monkeypatch, logs, clock):
    repo = Repo(tmp_path)
    os.mkfifo(marker(repo.path))
    _failing_subprocess(monkeypatch, command="pack-objects", mode="timeout")

    _compact_in_thread(repo)

    assert "backing off 1 h (consecutive timeouts: 1)" in logs[-1]
    assert os.path.isfile(marker(repo.path)) and read_marker(repo.path)["failures"] == 1


def test_once_valid_marker_never_outlives_its_stored_deadline(tmp_path, monkeypatch, logs, clock):
    repo = Repo(tmp_path)
    plant_marker(repo.path, {"version": 1, "failures": 6, "next_attempt_at": clock[0] + 86400})

    repo.compact()
    assert "backoff after 6 consecutive timeout(s)" in logs[-1]

    clock[0] += 86400 + 1
    repo.compact()

    assert logs[-1].startswith("Compacted"), logs
    assert not marker(repo.path).exists()


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
    assert not marker(repo.path).exists()


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


def _timing_out(*args, **kwargs):
    raise run_worktrees._CompactionTimeout("pack-objects timed out")


def test_fetch_path_returns_its_sha_when_the_backoff_marker_cannot_be_written(tmp_path, monkeypatch, logs):
    env = Env(tmp_path, monkeypatch)
    config = env.config()
    store = default_agent_workdir("OWNER/REPO", "claude").resolve()
    sha = run_worktrees.prepare_store(store, config=config, runner=Runner())
    env.set_pr_head(7, sha)

    def unwritable(*args, **kwargs):
        raise OSError("read-only lock root")

    monkeypatch.setattr(run_worktrees, "_compact", _timing_out)
    monkeypatch.setattr(run_worktrees, "_write_backoff", unwritable)
    assert run_worktrees.prepare_store(store, config=config, runner=Runner()) == sha
    assert run_worktrees.pin_pr_head(store, 7, config=config, runner=Runner()) == sha
    assert logs == [
        f"Pack compaction for {store} skipped: pack-objects timed out; backoff marker not written; "
        f"{_hint(store)}"
    ] * 2
    assert not marker(store).exists()


def test_fetch_path_returns_its_sha_after_a_timeout_with_a_saturated_marker(tmp_path, monkeypatch, logs, clock):
    env = Env(tmp_path, monkeypatch)
    config = env.config()
    store = default_agent_workdir("OWNER/REPO", "claude").resolve()
    sha = run_worktrees.prepare_store(store, config=config, runner=Runner())
    plant_marker(store, {"version": 1, "failures": 10**11, "next_attempt_at": clock[0] - 1})
    monkeypatch.setattr(run_worktrees, "_compact", _timing_out)

    assert run_worktrees.prepare_store(store, config=config, runner=Runner()) == sha
    assert read_marker(store) == {"version": 1, "failures": 1, "next_attempt_at": clock[0] + 3600}
    run_worktrees._write_backoff(store, run_worktrees.PACK_COMPACTION_BACKOFF_MAX_FAILURES, clock[0] - 90000)
    assert run_worktrees.prepare_store(store, config=config, runner=Runner()) == sha
    state = read_marker(store)
    assert state["failures"] == run_worktrees.PACK_COMPACTION_BACKOFF_MAX_FAILURES
    assert state["next_attempt_at"] == clock[0] + 86400


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


# --- multi-pack-index (#1579) ---------------------------------------------------


def _midx_names(packdir: Path) -> list[str]:
    return sorted(name for name in os.listdir(packdir) if name.startswith("multi-pack-index"))


def write_midx(repo: Path) -> bool:
    """A real git-written MIDX, with a bitmap when this git supports it; True if bitmapped."""
    result = subprocess.run(["git", "-C", str(repo), "multi-pack-index", "write", "--bitmap"],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        git(repo, "multi-pack-index", "write")
    assert (pack_dir(repo) / "multi-pack-index").is_file()
    return result.returncode == 0


def plant_chain(packdir: Path, *, unexpected: bool = False) -> Path:
    chain = packdir / "multi-pack-index.d"
    chain.mkdir()
    layer = "multi-pack-index-" + "a" * 40
    (chain / "multi-pack-index-chain").write_text("a" * 40 + "\n")
    (chain / f"{layer}.midx").write_bytes(b"MIDX")
    (chain / f"{layer}.bitmap").write_bytes(b"BITM")
    if unexpected:
        (chain / "operator-notes.txt").write_text("keep me\n")
    return chain


def _pack_files(repo: Path) -> dict[str, bytes]:
    return {name: (pack_dir(repo) / name).read_bytes()
            for name in sorted(os.listdir(pack_dir(repo))) if name.endswith((".pack", ".idx"))}


def _midx_aware_spy(monkeypatch, packdir: Path) -> list[tuple[str, list[str]]]:
    """Record each confined command with the MIDX artifacts present when it was built."""
    calls: list[tuple[str, list[str]]] = []
    real = run_worktrees.local_command

    def spy(args, **kwargs):
        calls.append((tuple(args)[0], _midx_names(packdir)))
        return real(args, **kwargs)

    monkeypatch.setattr(run_worktrees, "local_command", spy)
    return calls


def test_real_midx_with_bitmap_is_removed_before_any_pack_command(tmp_path, monkeypatch, logs):
    repo = Repo(tmp_path)
    bitmapped = write_midx(repo.path)
    packdir = pack_dir(repo.path)
    assert _midx_names(packdir)
    assert not bitmapped or any(name.endswith(".bitmap") for name in _midx_names(packdir))
    before = repo.objects()
    calls = _midx_aware_spy(monkeypatch, packdir)

    repo.compact()

    assert calls and calls[0][0] == "show-index"
    assert all(present == [] for _command, present in calls), calls
    assert {command for command, _present in calls} == {"show-index", "pack-objects", "index-pack"}
    assert _midx_names(packdir) == []
    assert len(non_kept_packs(repo.path)) == 1
    git(repo.path, "fsck", "--no-dangling")
    git(repo.path, "multi-pack-index", "verify")
    repo.assert_all_present(before)
    repo.assert_all_present([*repo.commits, repo.pr_head])
    assert logs[0] == f"Removed stale multi-pack-index for {repo.path} before compaction"
    assert logs[-1].startswith("Compacted") and not any("skipped" in message for message in logs), logs


def test_no_midx_keeps_the_command_sequence_and_logs(tmp_path, monkeypatch, logs):
    repo = Repo(tmp_path)
    calls = _local_command_spy(monkeypatch)

    repo.compact()

    assert [call[0] for call in calls] == ["show-index"] * (THRESHOLD + 3) + ["pack-objects", "index-pack", "show-index"]
    assert len(logs) == 1 and logs[0].startswith("Compacted"), logs


def test_planted_symlink_chain_and_unrelated_names(tmp_path, monkeypatch, logs):
    repo = Repo(tmp_path)
    packdir = pack_dir(repo.path)
    outside = tmp_path / "outside-midx"
    outside.write_bytes(b"not yours")
    (packdir / "multi-pack-index").symlink_to(outside)
    sidecar_target = tmp_path / "outside-bitmap"
    sidecar_target.write_bytes(b"bitmap target")
    (packdir / f"multi-pack-index-{'b' * 40}.bitmap").symlink_to(sidecar_target)
    (packdir / f"multi-pack-index-{'c' * 64}.rev").write_bytes(b"RIDX")
    plant_chain(packdir)
    unrelated = {
        "multi-pack-index-foo.txt": b"foo",
        "multi-pack-index.lock": b"lock",
        f"multi-pack-index-{'d' * 40}.midx": b"top-level midx layer is not a sidecar",
    }
    for name, data in unrelated.items():
        (packdir / name).write_bytes(data)
    before = repo.objects()

    repo.compact()

    assert _midx_names(packdir) == sorted(unrelated)
    for name, data in unrelated.items():
        assert (packdir / name).read_bytes() == data
    assert outside.read_bytes() == b"not yours"
    assert sidecar_target.read_bytes() == b"bitmap target"
    assert len(non_kept_packs(repo.path)) == 1
    repo.assert_all_present(before)
    assert logs[-1].startswith("Compacted"), logs


def test_symlinked_chain_directory_is_unlinked_not_followed(tmp_path, monkeypatch, logs):
    repo = Repo(tmp_path)
    packdir = pack_dir(repo.path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "multi-pack-index-chain").write_text("x\n")
    (packdir / "multi-pack-index.d").symlink_to(elsewhere)

    repo.compact()

    assert not os.path.lexists(packdir / "multi-pack-index.d")
    assert (elsewhere / "multi-pack-index-chain").read_text() == "x\n"
    assert logs[-1].startswith("Compacted"), logs


def _assert_removal_failed_cleanly(repo: Repo, before: dict[str, bytes], objects, calls, logs) -> None:
    assert _pack_files(repo.path) == before
    assert calls == []
    repo.assert_all_present(objects)
    assert logs == [f"Pack compaction for {repo.path} skipped: multi-pack-index could not be removed; old packs kept"]
    assert not marker(repo.path).exists()
    assert not any(name.startswith(".agent-loop-compact-") for name in os.listdir(pack_dir(repo.path)))


def test_unexpected_chain_entry_stops_before_any_pack_work(tmp_path, monkeypatch, logs):
    repo = Repo(tmp_path)
    chain = plant_chain(pack_dir(repo.path), unexpected=True)
    before, objects = _pack_files(repo.path), repo.objects()
    calls = _local_command_spy(monkeypatch)

    repo.compact()

    _assert_removal_failed_cleanly(repo, before, objects, calls, logs)
    assert (chain / "operator-notes.txt").read_text() == "keep me\n"


def test_unlink_failure_stops_before_any_pack_work(tmp_path, monkeypatch, logs):
    repo = Repo(tmp_path)
    write_midx(repo.path)
    midx = pack_dir(repo.path) / "multi-pack-index"
    before, objects = _pack_files(repo.path), repo.objects()
    real_unlink = os.unlink

    def refusing(path, *args, **kwargs):
        if Path(path) == midx:
            raise PermissionError("denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(run_worktrees.os, "unlink", refusing)
    calls = _local_command_spy(monkeypatch)

    repo.compact()

    _assert_removal_failed_cleanly(repo, before, objects, calls, logs)
    assert midx.is_file()
    assert "denied" not in logs[0]


def test_fetch_path_returns_its_sha_when_midx_removal_fails(tmp_path, monkeypatch, logs):
    env = Env(tmp_path, monkeypatch)
    config = env.config()
    store = env.seed_store("claude").resolve()
    shas = [run_worktrees.prepare_store(store, config=config, runner=Runner())]
    packdir = Path(git(store, "rev-parse", "--absolute-git-dir")) / "objects" / "pack"
    plant_chain(packdir, unexpected=True)
    calls = _local_command_spy(monkeypatch)
    logs.clear()
    for index in range(THRESHOLD + 2):
        shas.append(env.commit("a.txt", f"base{index}"))
        assert run_worktrees.prepare_store(store, config=config, runner=Runner()) == shas[-1]
    reason = f"Pack compaction for {store} skipped: multi-pack-index could not be removed; old packs kept"
    assert reason in logs and not any(message.startswith("Compacted") for message in logs), logs
    assert not any(call[0] in {"show-index", "pack-objects", "index-pack"} for call in calls), calls
    assert len(non_kept_packs(store)) > THRESHOLD
    assert not marker(store).exists()
    for sha in shas:
        git(store, "cat-file", "-e", sha)


def test_midx_is_untouched_at_or_below_the_threshold(tmp_path, monkeypatch, logs):
    repo = Repo(tmp_path, packs=THRESHOLD - 1)
    write_midx(repo.path)
    before = listing(repo.path)

    repo.compact()

    assert logs == []
    assert listing(repo.path) == before


def test_midx_is_untouched_while_a_backoff_is_active(tmp_path, monkeypatch, logs, clock):
    repo = Repo(tmp_path)
    write_midx(repo.path)
    run_worktrees._write_backoff(repo.path, 1, clock[0])
    before = listing(repo.path)
    calls = _local_command_spy(monkeypatch)

    repo.compact()

    assert calls == []
    assert len(logs) == 1 and "backoff after 1 consecutive timeout(s)" in logs[0], logs
    assert listing(repo.path) == before
