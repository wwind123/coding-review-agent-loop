"""Operator --worktree-setup keyed on lockfile digests (#1650, parent #1630).

Every scenario uses real temporary Git repositories, a local bare origin and
the production ``Runner`` (whose Git calls go through the confined boundary).
The setup command appends its working directory to a counter file outside the
repository, so each test can count how often setup ran in which clone.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from coding_review_agent_loop import run_worktrees, worktree_setup, workdir_claims
from coding_review_agent_loop.cli import build_parser
from coding_review_agent_loop.config import (
    config_from_args,
    default_agent_workdir,
    default_run_worktree_root,
    ensure_agent_workdirs,
    sync_coder_base_before_implementation,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import Runner
from coding_review_agent_loop.workdir_claims import claim_agent_workdirs, workdir_claim_scope

from test_run_worktrees import STORE_HOLDER, Env, RecordingRunner, child_env, git, stop


# --- helpers ---------------------------------------------------------------


def setup_command(counter: Path, tail: str = "mkdir -p .venv && touch .venv/ready") -> str:
    # Keep a tail's background operator from also backgrounding the counter
    # write: process-group cleanup may kill that job before it records the run.
    return f'echo "$PWD" >> {shlex.quote(str(counter))} && {{ {tail}\n}}'


def runs_in(counter: Path, store: Path) -> int:
    if not counter.exists():
        return 0
    real = os.path.realpath(store)
    return sum(1 for line in counter.read_text().splitlines() if os.path.realpath(line) == real)


def claude_store() -> Path:
    return default_agent_workdir("OWNER/REPO", "claude").resolve()


def setup_config(env: Env, command: str, *, keys=("requirements.txt",), links=(".venv",), extra=()):
    argv = ["--worktree-setup", command]
    for key in keys:
        argv += ["--worktree-setup-key", key]
    for link in links:
        argv += ["--worktree-link", link]
    return env.config(extra=(*argv, *extra))


def prepare_full(config, runner=None):
    runner = runner or Runner()
    claim_agent_workdirs(config)
    ensure_agent_workdirs(config, runner)
    return runner


def seeded(tmp_path, monkeypatch, *, gitignore="/.venv\n"):
    env = Env(tmp_path, monkeypatch)
    env.commit(".gitignore", gitignore)
    env.commit("requirements.txt", "pytest==1\n")
    env.commit("package.json", '{"name": "x"}\n')
    return env


# --- setup-fresh-clone --------------------------------------------------------


def test_fresh_clone_runs_setup_once_in_store_and_records_outside_clone(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, setup_command(counter))
    with workdir_claim_scope(command="task"):
        prepare_full(config)
        store = claude_store()
        assert runs_in(counter, store) == 1
        # Never in the run worktree.
        assert runs_in(counter, config.claude_dir) == 0
        record_path = worktree_setup.setup_record_path(store)
        assert record_path.is_file()
        assert os.path.commonpath((str(record_path), str(store))) != str(store)
        assert oct(record_path.stat().st_mode & 0o777) == "0o600"
        record = worktree_setup.read_setup_record(store)
        assert record["base_sha"] == git(store, "rev-parse", "HEAD") == git(env.seed, "rev-parse", "HEAD")
        assert record["store_identity"] == run_worktrees._identity(store)
        digest, _ = worktree_setup.setup_digest(
            store, record["base_sha"], command=config.worktree_setup,
            keys=config.worktree_setup_keys, runner=Runner(),
        )
        assert record["digest"] == digest
        link = config.claude_dir / ".venv"
        assert link.is_symlink() and os.path.realpath(link) == os.path.realpath(store / ".venv")


# --- setup-digest-unchanged / setup-key-or-command-changed ----------------------


def test_unchanged_digest_skips_setup(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, setup_command(counter))
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    # A non-key commit leaves the digest unchanged.
    env.commit("b.txt", "unrelated")
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert runs_in(counter, store) == 1
    # A second full run with a fresh config also skips.
    with workdir_claim_scope(command="task"):
        prepare_full(setup_config(env, setup_command(counter)))
    assert runs_in(counter, store) == 1


def test_key_or_command_change_reruns_setup_from_pinned_blobs(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, setup_command(counter))
    store = claude_store()
    first = run_worktrees.prepare_store(store, config=config, runner=Runner())
    old_digest = worktree_setup.read_setup_record(store)["digest"]

    # The digest comes from Git objects at the SHA, never working-tree bytes.
    (store / "requirements.txt").write_text("edited locally\n")
    assert worktree_setup.setup_digest(
        store, first, command=config.worktree_setup, keys=config.worktree_setup_keys, runner=Runner()
    )[0] == old_digest
    git(store, "checkout", "--", "requirements.txt")

    second = env.commit("requirements.txt", "pytest==2\n")
    assert run_worktrees.prepare_store(store, config=config, runner=Runner()) == second
    assert runs_in(counter, store) == 2
    record = worktree_setup.read_setup_record(store)
    assert record["base_sha"] == second and record["digest"] != old_digest
    assert git(store, "rev-parse", "HEAD") == second
    assert (store / "requirements.txt").read_text() == "pytest==2\n"

    changed = dataclasses.replace(config, worktree_setup=setup_command(counter, "mkdir -p .venv && touch .venv/ready && true"))
    run_worktrees.prepare_store(store, config=changed, runner=Runner())
    assert runs_in(counter, store) == 3
    assert worktree_setup.read_setup_record(store)["command_sha256"] == worktree_setup.command_hash(
        changed.worktree_setup
    )


def test_keyless_setup_runs_once_per_clone_lifetime(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, setup_command(counter), keys=(), links=())
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    env.commit("requirements.txt", "changed\n")
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert runs_in(counter, store) == 1


# --- setup-failure / setup-failed-rerun-invalidates -----------------------------


def test_setup_failure_stops_with_output_tail_and_no_record_or_links(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    failing = setup_command(counter, "for i in $(seq 1 100); do echo line$i; done; exit 3")
    config = setup_config(env, failing)
    with workdir_claim_scope(command="task"):
        claim_agent_workdirs(config)
        with pytest.raises(AgentLoopError) as caught:
            ensure_agent_workdirs(config, Runner())
    message = str(caught.value)
    clones = [str(default_agent_workdir("OWNER/REPO", agent).resolve()) for agent in ("claude", "codex")]
    assert "exited 3" in message and any(f"shared clone {clone};" in message for clone in clones)
    assert "line100" in message and "line61" in message and "line60\n" not in message
    for clone in clones:
        assert worktree_setup.read_setup_record(Path(clone)) is None
        assert not worktree_setup.setup_record_path(Path(clone)).exists()
    runs_root = default_run_worktree_root("OWNER/REPO", "claude")
    assert not runs_root.exists() or not list(runs_root.iterdir())


def test_setup_timeout_kills_and_records_nothing(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = dataclasses.replace(
        setup_config(env, setup_command(counter, "echo started && sleep 30")),
        worktree_setup_timeout_seconds=1,
    )
    store = claude_store()
    with pytest.raises(AgentLoopError, match="timed out after 1s") as caught:
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert "--worktree-setup-timeout" in str(caught.value)
    assert "started" in str(caught.value) and str(store) in str(caught.value)
    assert worktree_setup.read_setup_record(store) is None


@pytest.mark.parametrize("mode", ["nonzero", "timeout"])
def test_failed_different_setup_invalidates_prior_success(tmp_path, monkeypatch, mode):
    env = seeded(tmp_path, monkeypatch)
    counter_a, counter_b = tmp_path / "counter-a", tmp_path / "counter-b"
    config_a = setup_config(env, setup_command(counter_a))
    store = claude_store()
    run_worktrees.prepare_store(store, config=config_a, runner=Runner())
    assert worktree_setup.read_setup_record(store) is not None

    damage = "echo damaged > .venv/state; " + ("exit 1" if mode == "nonzero" else "sleep 30")
    config_b = dataclasses.replace(
        config_a, worktree_setup=setup_command(counter_b, damage), worktree_setup_timeout_seconds=2,
    )
    with pytest.raises(AgentLoopError):
        run_worktrees.prepare_store(store, config=config_b, runner=Runner())
    assert runs_in(counter_b, store) == 1
    assert (store / ".venv" / "state").read_text() == "damaged\n"
    assert worktree_setup.read_setup_record(store) is None

    run_worktrees.prepare_store(store, config=config_a, runner=Runner())
    assert runs_in(counter_a, store) == 2
    assert worktree_setup.read_setup_record(store) is not None


def test_invalidation_failure_never_launches_setup(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter_a, counter_b = tmp_path / "counter-a", tmp_path / "counter-b"
    config_a = setup_config(env, setup_command(counter_a))
    store = claude_store()
    run_worktrees.prepare_store(store, config=config_a, runner=Runner())
    record_path = worktree_setup.setup_record_path(store)
    real_unlink = os.unlink

    def refuse(path, *args, **kwargs):
        if Path(path) == record_path:
            raise PermissionError("denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(worktree_setup.os, "unlink", refuse)
    config_b = dataclasses.replace(config_a, worktree_setup=setup_command(counter_b))
    with pytest.raises(AgentLoopError, match="Could not invalidate"):
        run_worktrees.prepare_store(store, config=config_b, runner=Runner())
    assert not counter_b.exists()


# --- setup-clean-old-commit / setup-dirty-key-refused ----------------------------


class SubmoduleEnv:
    """A seed whose keys cover every Git entry type, and a store lagging the base."""

    def __init__(self, tmp_path: Path, monkeypatch):
        self.env = env = seeded(tmp_path, monkeypatch)
        self.sub = tmp_path / "sub"
        subprocess.run(["git", "init", "-q", "-b", "main", str(self.sub)], check=True)
        (self.sub / "s.txt").write_text("sub\n")
        git(self.sub, "add", "s.txt")
        git(self.sub, "commit", "-q", "-m", "sub")
        self.sub_sha = git(self.sub, "rev-parse", "HEAD")
        (env.seed / "deps").mkdir()
        (env.seed / "deps" / "x.txt").write_text("x\n")
        os.symlink("a.txt", env.seed / "link-key")
        git(env.seed, "add", "deps", "link-key")
        for name in ("mods/pop", "mods/uninit"):
            git(env.seed, "update-index", "--add", "--cacheinfo", f"160000,{self.sub_sha},{name}")
        git(env.seed, "commit", "-q", "-m", "entry types")
        git(env.seed, "push", "-q", "origin", "HEAD:refs/heads/main")
        self.old = git(env.seed, "rev-parse", "HEAD")
        self.store = env.seed_store("claude", venv=False).resolve()
        # Populate one submodule at its gitlink; leave the other uninitialized.
        shutil.rmtree(self.store / "mods" / "pop")
        subprocess.run(["git", "clone", "-q", str(self.sub), str(self.store / "mods" / "pop")], check=True)
        assert (self.store / "mods" / "uninit").is_dir()
        assert not any((self.store / "mods" / "uninit").iterdir())
        self.counter = tmp_path / "counter"
        self.keys = ("requirements.txt", "link-key", "deps", "mods/pop", "mods/uninit", "absent.lock")

    def config(self, command=None):
        return setup_config(self.env, command or setup_command(self.counter), keys=self.keys)


def test_clean_store_behind_pin_is_materialized_and_verified_by_entry_type(tmp_path, monkeypatch):
    s = SubmoduleEnv(tmp_path, monkeypatch)
    new = s.env.commit("requirements.txt", "pytest==3\n")
    assert git(s.store, "rev-parse", "HEAD") == s.old
    runner = RecordingRunner()
    assert run_worktrees.prepare_store(s.store, config=s.config(), runner=runner) == new
    assert runs_in(s.counter, s.store) == 1
    assert git(s.store, "rev-parse", "HEAD") == new
    assert (s.store / "requirements.txt").read_text() == "pytest==3\n"
    # Parent probes never override the confined --ignore-submodules=all, and
    # nothing is ever passed to hash-object.
    assert not any("hash-object" in call for call in runner.calls)
    assert not any(arg.startswith("--ignore-submodules") for call in runner.calls for arg in call)
    _, entries = worktree_setup.setup_digest(
        s.store, new, command=s.config().worktree_setup, keys=s.keys, runner=Runner()
    )
    assert entries["deps"][1] == "tree" and entries["link-key"][0] == "120000"
    assert entries["mods/pop"][:2] == ("160000", "commit") and entries["absent.lock"] is None


def test_drifted_populated_submodule_is_refused(tmp_path, monkeypatch):
    s = SubmoduleEnv(tmp_path, monkeypatch)
    pop = s.store / "mods" / "pop"
    (pop / "s.txt").write_text("drift\n")
    git(pop, "commit", "-q", "-am", "drift")
    drifted = git(pop, "rev-parse", "HEAD")
    with pytest.raises(AgentLoopError, match="mods/pop") as caught:
        run_worktrees.prepare_store(s.store, config=s.config(), runner=Runner())
    assert str(s.store) in str(caught.value)
    assert git(pop, "rev-parse", "HEAD") == drifted
    assert runs_in(s.counter, s.store) == 0


def test_untracked_file_shadowing_absent_key_is_refused(tmp_path, monkeypatch):
    s = SubmoduleEnv(tmp_path, monkeypatch)
    (s.store / "absent.lock").write_text("planted\n")
    with pytest.raises(AgentLoopError, match="absent.lock") as caught:
        run_worktrees.prepare_store(s.store, config=s.config(), runner=Runner())
    assert "may have been created by the --worktree-setup command itself" in str(caught.value)
    assert runs_in(s.counter, s.store) == 0
    assert (s.store / "absent.lock").read_text() == "planted\n"


def _dirty_tree_key(store: Path):
    (store / "deps" / "x.txt").write_text("changed\n")
    return "deps/x.txt"


def _dirty_symlink_key(store: Path):
    (store / "link-key").unlink()
    os.symlink("other.txt", store / "link-key")
    return "link-key"


def _dirty_key(store: Path):
    (store / "requirements.txt").write_text("local\n")
    return "requirements.txt"


def _dirty_non_key(store: Path):
    (store / "package.json").write_text('{"scripts": {"postinstall": "evil"}}\n')
    return "package.json"


def _staged_non_key(store: Path):
    (store / "package.json").write_text('{"staged": true}\n')
    git(store, "add", "package.json")
    return "package.json"


def _deleted_non_key(store: Path):
    (store / "a.txt").unlink()
    return "a.txt"


def _dirty_submodule_file(store: Path):
    (store / "mods" / "pop" / "s.txt").write_text("dirty\n")
    return "mods/pop"


def _drifted_submodule_head(store: Path):
    pop = store / "mods" / "pop"
    (pop / "s.txt").write_text("moved\n")
    git(pop, "commit", "-q", "-am", "moved")
    return "mods/pop"


@pytest.mark.parametrize("head", ["at-sha", "behind"])
@pytest.mark.parametrize("dirty", [
    _dirty_key, _dirty_non_key, _staged_non_key, _deleted_non_key, _dirty_tree_key,
    _dirty_symlink_key, _dirty_submodule_file, _drifted_submodule_head,
])
def test_tracked_modification_stops_required_setup_before_invalidation(tmp_path, monkeypatch, head, dirty):
    s = SubmoduleEnv(tmp_path, monkeypatch)
    config = s.config()
    run_worktrees.prepare_store(s.store, config=config, runner=Runner())
    assert runs_in(s.counter, s.store) == 1
    record_bytes = worktree_setup.setup_record_path(s.store).read_bytes()
    counter_b = tmp_path / "counter-b"
    if head == "at-sha":
        # Same pin, a different command: setup is required with HEAD == SHA.
        config = dataclasses.replace(config, worktree_setup=setup_command(counter_b))
        expected_head = git(s.store, "rev-parse", "HEAD")
    else:
        expected_head = git(s.store, "rev-parse", "HEAD")
        s.env.commit("requirements.txt", "pytest==9\n")
    path = dirty(s.store)
    sub_head = git(s.store / "mods" / "pop", "rev-parse", "HEAD")
    snapshot = {
        p: (s.store / p).read_bytes() for p in ("requirements.txt", "package.json", "deps/x.txt")
    }
    with pytest.raises(AgentLoopError, match="refused") as caught:
        run_worktrees.prepare_store(s.store, config=config, runner=Runner())
    assert str(s.store) in str(caught.value) and path in str(caught.value)
    assert worktree_setup.setup_record_path(s.store).read_bytes() == record_bytes
    assert runs_in(s.counter, s.store) == 1 and not counter_b.exists()
    assert git(s.store, "rev-parse", "HEAD") == expected_head
    assert git(s.store / "mods" / "pop", "rev-parse", "HEAD") == sub_head
    for p, data in snapshot.items():
        assert (s.store / p).read_bytes() == data


# --- setup-concurrent -----------------------------------------------------------


def test_concurrent_runs_on_one_store_set_up_once(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, setup_command(counter, "sleep 1 && mkdir -p .venv && touch .venv/ready"))
    store = claude_store()
    run_worktrees.prepare_store(  # clone once so both threads contend only on setup
        store, config=dataclasses.replace(config, worktree_setup=None, worktree_links=()), runner=Runner()
    )
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker():
        try:
            barrier.wait()
            run_worktrees.prepare_store(store, config=config, runner=Runner())
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not errors
    assert runs_in(counter, store) == 1


# --- setup-base-advance-resync -----------------------------------------------------


def test_base_advance_reruns_setup_before_checkout_uses_new_pin(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, setup_command(counter))
    store = claude_store()
    with workdir_claim_scope(command="task"):
        prepare_full(config)
        sha_a = git(env.seed, "rev-parse", "HEAD")
        assert worktree_setup.read_setup_record(store)["base_sha"] == sha_a
        assert runs_in(counter, store) == 1
        sha_b = env.commit("requirements.txt", "pytest==2\n")
        seen: list[tuple[int, str]] = []

        def hook(args, cwd):
            if args[:3] == ("git", "checkout", "--detach") and cwd == config.claude_dir:
                record = worktree_setup.read_setup_record(store)
                seen.append((runs_in(counter, store), record["base_sha"]))

        sync_coder_base_before_implementation(config, RecordingRunner(hook=hook))
        assert seen == [(2, sha_b)]
        assert git(config.claude_dir, "rev-parse", "HEAD") == sha_b
        digest, _ = worktree_setup.setup_digest(
            store, sha_b, command=config.worktree_setup, keys=config.worktree_setup_keys, runner=Runner()
        )
        assert worktree_setup.read_setup_record(store)["digest"] == digest


# --- setup-held-store ----------------------------------------------------------


def test_held_store_refuses_required_setup_and_keeps_record(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter_a, counter_b = tmp_path / "counter-a", tmp_path / "counter-b"
    config = setup_config(env, setup_command(counter_a))
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    record_bytes = worktree_setup.setup_record_path(store).read_bytes()
    head = git(store, "rev-parse", "HEAD")
    env.commit("requirements.txt", "pytest==5\n")
    holder = subprocess.Popen(
        [sys.executable, "-c", STORE_HOLDER, str(workdir_claims.claim_root()), str(store)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=child_env(),
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        changed = dataclasses.replace(config, worktree_setup=setup_command(counter_b))
        with pytest.raises(AgentLoopError, match="needs --worktree-setup to run.*holds its claim"):
            run_worktrees.prepare_store(store, config=changed, runner=Runner())
        assert not counter_b.exists()
        assert worktree_setup.setup_record_path(store).read_bytes() == record_bytes
        assert git(store, "rev-parse", "HEAD") == head
    finally:
        stop(holder)


# --- setup-recloned-store -------------------------------------------------------------


def test_stale_store_recreation_deletes_record_before_reclone(tmp_path, monkeypatch):
    from coding_review_agent_loop import agent_permissions
    from coding_review_agent_loop import config as config_module

    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, setup_command(counter))
    store = claude_store()
    # Identity is forced to match, as a reused inode would.
    monkeypatch.setattr(worktree_setup, "_store_identity", lambda _store: [1, 2])
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert runs_in(counter, store) == 1

    record_seen: list[bool] = []
    real_remove = agent_permissions.remove_checkout_tree

    def remove(path):
        record_seen.append(worktree_setup.setup_record_path(store).exists())
        return real_remove(path)

    monkeypatch.setattr(agent_permissions, "remove_checkout_tree", remove)
    # Genuinely artifact-only: no working-tree content, no index and an unborn HEAD.
    for entry in store.iterdir():
        if entry.name != ".git":
            shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()
    (store / ".git" / "index").unlink()
    git(store, "symbolic-ref", "HEAD", "refs/heads/unborn")
    assert config_module._is_stale_default_workdir(store)
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert record_seen == [False]
    assert runs_in(counter, store) == 2
    assert worktree_setup.read_setup_record(store) is not None


def test_missing_store_clone_deletes_leftover_record(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, setup_command(counter))
    store = claude_store()
    monkeypatch.setattr(worktree_setup, "_store_identity", lambda _store: [1, 2])
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    shutil.rmtree(store)  # for example a reboot wiping /tmp but not the lock root
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert runs_in(counter, store) == 2


@pytest.mark.parametrize("damage", ["deleted", "dangling"])
def test_unusable_link_source_forces_setup_despite_matching_record(tmp_path, monkeypatch, damage):
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    recreate = '{ if [ -L .venv ]; then mkdir -p "$(readlink .venv)" && touch .venv/ready; else mkdir -p .venv && touch .venv/ready; fi; }'
    config = setup_config(env, setup_command(counter, recreate))
    with workdir_claim_scope(command="task"):
        prepare_full(config)
    store = claude_store()
    assert runs_in(counter, store) == 1
    shutil.rmtree(store / ".venv")
    if damage == "dangling":
        os.symlink(tmp_path / "vanished-venv", store / ".venv")
    assert not run_worktrees.link_source_usable(store, ".venv")
    with workdir_claim_scope(command="task"):
        config_2 = setup_config(env, setup_command(counter, recreate))
        prepare_full(config_2)
        assert runs_in(counter, store) == 2
        assert run_worktrees.link_source_usable(store, ".venv")
        assert os.path.exists(config_2.claude_dir / ".venv")


def test_succeeded_but_did_not_create_only_when_setup_ran(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, setup_command(counter, "true"), links=("missing-env",))
    store = claude_store()
    with pytest.raises(AgentLoopError, match="succeeded but did not create"):
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert worktree_setup.read_setup_record(store) is None
    with pytest.raises(AgentLoopError) as caught:
        run_worktrees.check_link_sources(store, ("missing-env",))
    assert "succeeded but did not create" not in str(caught.value)


# --- detection-hint -------------------------------------------------------------


@pytest.mark.parametrize("flags", [(), ("--worktree-link", ".venv"), ("--worktree-setup", "true")])
def test_detection_hint_only_without_setup_or_link_flags(tmp_path, monkeypatch, flags):
    env = seeded(tmp_path, monkeypatch)
    env.commit("package-lock.json", "{}\n")
    env.seed_stores()
    hints: list[str] = []
    monkeypatch.setattr(worktree_setup, "log", lambda _config, message: hints.append(message))
    config = env.config(extra=flags)
    with workdir_claim_scope(command="task"):
        prepare_full(config)
    hint_lines = [line for line in hints if line.startswith("Hint:")]
    if flags:
        assert hint_lines == []
    else:
        # One line per run even though two agents' stores were prepared.
        assert len(hint_lines) == 1
        assert "package-lock.json" in hint_lines[0] and "requirements.txt" in hint_lines[0]
        assert "--worktree-setup" in hint_lines[0] and "--worktree-link" in hint_lines[0]
        assert worktree_setup.read_setup_record(claude_store()) is None


def test_dependency_manifests_reads_pinned_tree_not_working_tree(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    store = env.seed_store("claude", venv=False)
    sha = git(store, "rev-parse", "HEAD")
    (store / "requirements.txt").write_text("untracked\n")
    assert worktree_setup.dependency_manifests(store, sha, runner=Runner()) == []
    env.commit("requirements-dev.txt", "x\n")
    env.commit("pyproject.toml", "x\n")
    git(store, "fetch", "-q", "origin")
    new = git(store, "rev-parse", "origin/main")
    assert worktree_setup.dependency_manifests(store, new, runner=Runner()) == [
        "pyproject.toml", "requirements-dev.txt",
    ]


# --- CLI, config and recovery ------------------------------------------------------


def test_cli_parses_setup_flags_and_dedupes_keys(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    config = env.config(extra=(
        "--worktree-setup", "npm ci", "--worktree-setup-key", "package-lock.json",
        "--worktree-setup-key", "package-lock.json", "--worktree-setup-key", "requirements.txt",
    ))
    assert config.worktree_setup == "npm ci"
    assert config.worktree_setup_keys == ("package-lock.json", "requirements.txt")


@pytest.mark.parametrize("extra, match", [
    (("--worktree-setup-key", "package-lock.json"), "requires --worktree-setup"),
    (("--worktree-setup", "true", "--worktree-setup-key", "/abs"), "Invalid --worktree-setup-key"),
    (("--worktree-setup", "true", "--worktree-setup-key", "a/../b"), "Invalid --worktree-setup-key"),
    (("--worktree-setup", "true", "--worktree-setup-key", "a//b"), "Invalid --worktree-setup-key"),
    (("--worktree-setup", "  "), "non-empty"),
])
def test_cli_rejects_invalid_setup_flags(tmp_path, monkeypatch, extra, match):
    env = Env(tmp_path, monkeypatch)
    with pytest.raises(AgentLoopError, match=match):
        env.config(extra=extra)


def test_setup_ignored_without_default_per_run_worktree(tmp_path, monkeypatch, capsys):
    env = Env(tmp_path, monkeypatch)
    a, b = tmp_path / "a", tmp_path / "b"
    for path in (a, b):
        subprocess.run(["git", "clone", "-q", str(env.bare), str(path)], check=True)
    args = build_parser().parse_args([
        "task", "x", "--repo", "OWNER/REPO", "--coder", "claude", "--reviewer", "codex",
        "--base", "main", "--gh-cmd", str(env.gh), "--trusted-local-origin", str(env.bare),
        "--claude-cmd", "/bin/true", "--codex-cmd", "/bin/true", "--antigravity-cmd", "/bin/true",
        "--quiet", "--claude-dir", str(a), "--codex-dir", str(b), "--worktree-setup", "true",
    ])
    config = config_from_args(args, Runner())
    assert config.worktree_setup is None and config.worktree_setup_keys == ()
    assert "--worktree-setup is ignored" in capsys.readouterr().err


def test_recovery_commands_preserve_setup_flags(tmp_path):
    from agent_loop_helpers import make_config

    from coding_review_agent_loop.managed_ci import _RECOVERY_VALUE_OPTIONS, _render_recovery_command

    assert {"--worktree-setup", "--worktree-setup-key"} <= _RECOVERY_VALUE_OPTIONS
    config = make_config(tmp_path, invocation_argv=(
        "agent-loop", "issue", "123", "--repo", "OWNER/REPO",
        "--worktree-setup", "npm ci && python3 -m venv .venv",
        "--worktree-setup-key", "package-lock.json", "--worktree-link", "node_modules",
    ))
    command = _render_recovery_command(config, target="pr", identifier=456, managed_ci=False)
    args = build_parser().parse_args(shlex.split(command)[1:])
    assert args.worktree_setup == "npm ci && python3 -m venv .venv"
    assert args.worktree_setup_key == ["package-lock.json"]
    assert args.worktree_link == ["node_modules"]


def test_dry_run_only_logs_setup(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    messages: list[str] = []
    monkeypatch.setattr(worktree_setup, "log", lambda _config, message: messages.append(message))
    config = setup_config(env, "touch should-not-exist")
    store = claude_store()
    with run_worktrees.store_lock(store):
        assert worktree_setup.ensure_store_setup(
            store, "0" * 40, free=True, config=config, runner=Runner(dry_run=True)
        ) is False
    assert messages == [f"Would run worktree setup in {store}"]
    assert not (store / "should-not-exist").exists()


# --- review round 1: tracked deletions, index flags, bounded tails ----------------


@pytest.mark.parametrize("deletion", ["unstaged", "staged"])
def test_tracked_deletions_are_refused_not_recloned_as_stale_store(tmp_path, monkeypatch, deletion):
    """A real repository whose tracked files were all deleted is not an artifact-only store."""
    from coding_review_agent_loop import config as config_module

    env = seeded(tmp_path, monkeypatch)
    counter_a, counter_b = tmp_path / "counter-a", tmp_path / "counter-b"
    config = setup_config(env, setup_command(counter_a, "true"), links=())
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    record_bytes = worktree_setup.setup_record_path(store).read_bytes()
    git_marker = store / ".git" / "agent-loop-test-marker"
    git_marker.write_text("kept\n")
    if deletion == "staged":
        git(store, "rm", "-rq", ".")
        assert git(store, "ls-files") == ""  # an empty index: ls-files alone looks untracked
    for entry in store.iterdir():
        if entry.name != ".git":
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    staged_before = git(store, "diff", "--cached", "--name-status")
    assert config_module._is_stale_default_workdir(store)
    changed = dataclasses.replace(config, worktree_setup=setup_command(counter_b, "true"))
    with pytest.raises(AgentLoopError, match="refused") as caught:
        run_worktrees.prepare_store(store, config=changed, runner=Runner())
    assert "requirements.txt" in str(caught.value) and str(store) in str(caught.value)
    assert worktree_setup.setup_record_path(store).read_bytes() == record_bytes
    assert git_marker.read_text() == "kept\n"  # the clone was not re-created
    assert not (store / "requirements.txt").exists()  # nothing was restored
    assert git(store, "diff", "--cached", "--name-status") == staged_before  # staged state kept
    assert not counter_b.exists()


def test_kept_stale_store_logs_restore_remedy(tmp_path, monkeypatch):
    """Keeping a store whose tracked files were deleted names the restore command (#1655)."""
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, setup_command(counter, "true"), links=())
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    git_marker = store / ".git" / "agent-loop-test-marker"
    git_marker.write_text("kept\n")
    for entry in store.iterdir():
        if entry.name != ".git":
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    messages: list[str] = []
    monkeypatch.setattr(run_worktrees, "log", lambda _config, message: messages.append(message))
    try:
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    except AgentLoopError:
        pass  # the setup gate may still refuse; only the keep-path log matters here
    kept = [message for message in messages if "has tracked content deleted" in message]
    assert len(kept) == 1
    assert shlex.join(["git", "-C", str(store), "checkout", "--", "."]) in kept[0]
    assert "Per-run worktrees are unaffected" in kept[0]
    assert git_marker.read_text() == "kept\n"  # the store was kept, not re-cloned
    assert not (store / "requirements.txt").exists()  # nothing was restored automatically


@pytest.mark.parametrize("state, tracked", [
    ("unborn-empty-index", False),
    ("index-entries", True),
    ("head-tree-only", True),
    ("broken-head", True),
])
def test_store_tracks_paths_fails_closed(tmp_path, monkeypatch, state, tracked):
    env = seeded(tmp_path, monkeypatch)
    store = env.seed_store("claude", venv=False).resolve()
    if state == "unborn-empty-index":
        (store / ".git" / "index").unlink()
        git(store, "symbolic-ref", "HEAD", "refs/heads/unborn")
    elif state == "head-tree-only":
        git(store, "rm", "-rq", "--cached", ".")
    elif state == "broken-head":
        (store / ".git" / "index").unlink()
        (store / ".git" / "HEAD").write_text("0" * 40 + "\n")
    assert run_worktrees._store_tracks_paths(store, Runner()) is tracked


def _assume_unchanged(store: Path) -> str:
    (store / "package.json").write_text('{"scripts": {"preinstall": "hidden"}}\n')
    git(store, "update-index", "--assume-unchanged", "package.json")
    return "package.json"


def _skip_worktree(store: Path) -> str:
    (store / "setup.py").write_text("raise SystemExit('hidden')\n")
    git(store, "update-index", "--skip-worktree", "setup.py")
    return "setup.py"


def _submodule_skip_worktree(store: Path) -> str:
    pop = store / "mods" / "pop"
    (pop / "s.txt").write_text("hidden\n")
    git(pop, "update-index", "--skip-worktree", "s.txt")
    return "mods/pop/s.txt"


@pytest.mark.parametrize("hide", [_assume_unchanged, _skip_worktree, _submodule_skip_worktree])
def test_index_flags_hiding_tracked_edits_refuse_setup(tmp_path, monkeypatch, hide):
    s = SubmoduleEnv(tmp_path, monkeypatch)
    s.env.commit("setup.py", "print('base')\n")
    config = s.config()
    run_worktrees.prepare_store(s.store, config=config, runner=Runner())
    record_bytes = worktree_setup.setup_record_path(s.store).read_bytes()
    path = hide(s.store)
    # Status alone would call this tree clean.
    assert worktree_setup._porcelain_paths(
        subprocess.run(["git", "-C", str(s.store), "status", "--porcelain=v1", "-z",
                        "--untracked-files=no", "--ignore-submodules=all"],
                       capture_output=True, check=True).stdout
    ) == []
    hidden = (s.store / path).read_bytes()
    counter_b = tmp_path / "counter-b"
    changed = dataclasses.replace(config, worktree_setup=setup_command(counter_b))
    with pytest.raises(AgentLoopError, match="assume-unchanged or skip-worktree") as caught:
        run_worktrees.prepare_store(s.store, config=changed, runner=Runner())
    assert path in str(caught.value)
    assert worktree_setup.setup_record_path(s.store).read_bytes() == record_bytes
    assert (s.store / path).read_bytes() == hidden
    assert not counter_b.exists()


def test_read_output_tail_reads_a_bounded_suffix(tmp_path):
    big = tmp_path / "big.log"
    big.write_bytes(b"x" * (10 * 1024 * 1024) + b"\nLAST\n")
    reads: list[int] = []
    with open(big, "rb") as handle:
        real_read = handle.read

        class Spy:
            def seek(self, *args):
                return handle.seek(*args)

            def tell(self):
                return handle.tell()

            def read(self, size=-1):
                reads.append(size)
                return real_read(size)

        data = worktree_setup.read_output_tail(Spy())
    assert data.endswith(b"LAST\n") and len(data) <= 2 * worktree_setup.OUTPUT_TAIL_BYTES
    assert reads and all(0 < size <= 2 * worktree_setup.OUTPUT_TAIL_BYTES for size in reads)


@pytest.mark.parametrize("mode", ["nonzero", "timeout"])
def test_large_failure_output_is_tail_bounded(tmp_path, monkeypatch, mode):
    seen: list[int] = []
    real_tail = worktree_setup.output_tail

    def spy(data):
        seen.append(len(data))
        return real_tail(data)

    monkeypatch.setattr(worktree_setup, "output_tail", spy)
    noise = "head -c 5000000 /dev/zero | tr '\\0' 'x'; echo; echo FINAL-LINE"
    tail = "exit 4" if mode == "nonzero" else "sleep 30"
    with pytest.raises(AgentLoopError) as caught:
        worktree_setup.run_setup_command(tmp_path, f"{noise}; {tail}", timeout_seconds=3)
    message = str(caught.value)
    assert "FINAL-LINE" in message
    assert ("exited 4" in message) if mode == "nonzero" else ("timed out" in message)
    assert seen and max(seen) <= 2 * worktree_setup.OUTPUT_TAIL_BYTES
    assert len(message) < 4 * worktree_setup.OUTPUT_TAIL_BYTES


@pytest.mark.parametrize("command", ["true", "exit 5", "sleep 30"])
def test_process_group_is_signalled_only_while_the_leader_is_unreaped(tmp_path, monkeypatch, command):
    calls: list[int] = []
    real_killpg = os.killpg

    def recording_killpg(pgid, sig):
        # The leader (pid == pgid) must still exist, alive or as a zombie, so the
        # group ID cannot have been reused by an unrelated process.
        os.waitid(os.P_PID, pgid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        calls.append(pgid)
        return real_killpg(pgid, sig)

    monkeypatch.setattr(worktree_setup.os, "killpg", recording_killpg)
    try:
        worktree_setup.run_setup_command(tmp_path, command, timeout_seconds=1)
    except AgentLoopError:
        pass
    assert calls


@pytest.mark.parametrize("exit_command", ["exit 1", "exit 0", "sleep 30"])
def test_surviving_setup_child_cannot_write_after_the_command_ends(tmp_path, exit_command):
    """A background writer left by setup is killed before run_setup_command returns or raises."""
    state = tmp_path / "state"
    command = f"(sleep 1; echo damaged > {state}) & {exit_command}"
    started = time.monotonic()
    try:
        worktree_setup.run_setup_command(tmp_path, command, timeout_seconds=0.5 if "sleep" in exit_command else 10)
    except AgentLoopError:
        assert exit_command != "exit 0"
    else:
        assert exit_command == "exit 0"
    assert time.monotonic() - started < 1
    time.sleep(1.5)
    assert not state.exists()


def test_failed_setup_child_cannot_damage_the_environment_after_a_retry(tmp_path, monkeypatch):
    """A failed attempt's background writer cannot mutate the store after a retry records success."""
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    store = claude_store()
    failing = setup_config(env, setup_command(counter, "(sleep 1; echo damaged > .venv-state) & exit 1"))
    with pytest.raises(AgentLoopError, match="exited 1"):
        run_worktrees.prepare_store(store, config=failing, runner=Runner())
    assert worktree_setup.read_setup_record(store) is None
    retry = setup_config(env, setup_command(counter))
    run_worktrees.prepare_store(store, config=retry, runner=Runner())
    assert worktree_setup.read_setup_record(store) is not None
    assert runs_in(counter, store) == 2
    time.sleep(1.5)
    assert not (store / ".venv-state").exists()


def test_setup_refuses_before_launch_without_unreaped_wait(tmp_path, monkeypatch):
    """Without os.waitid the group cannot be killed safely, so setup never launches."""
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    store = claude_store()
    config = setup_config(env, setup_command(counter, "(sleep 1; echo damaged > .venv-state) & exit 1"))
    monkeypatch.delattr(os, "waitid")
    with pytest.raises(AgentLoopError, match="os.waitid is unavailable"):
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert runs_in(counter, store) == 0
    assert worktree_setup.read_setup_record(store) is None
    time.sleep(1.5)
    assert not (store / ".venv-state").exists()


# --- #1649: setup-created key files and --worktree-setup-timeout -----------------------


LOCK = "package-lock.json"
HINT = "may have been created by the --worktree-setup command itself"


def lock_command(counter: Path, content: str = "gen-1", tail: str = "") -> str:
    """Write the untracked lockfile (only while the repository tracks none), like npm install."""
    write = f"if ! git ls-files --error-unmatch {LOCK} >/dev/null 2>&1; then printf {content} > {LOCK}; fi"
    return setup_command(counter, f"{write}; mkdir -p .venv && touch .venv/ready{tail}")


def lock_config(env: Env, command: str):
    return setup_config(env, command, keys=("requirements.txt", LOCK))


def sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def ledger(store: Path) -> dict[str, str]:
    return worktree_setup.read_generated_ledger(store, ("requirements.txt", LOCK))


def first_lock_setup(tmp_path, monkeypatch):
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = lock_config(env, lock_command(counter))
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert runs_in(counter, store) == 1
    assert (store / LOCK).read_text() == "gen-1"
    return env, counter, config, store


def test_setup_created_absent_key_does_not_block_a_later_setup(tmp_path, monkeypatch):
    """gen-key-second-setup."""
    env, counter, config, store = first_lock_setup(tmp_path, monkeypatch)
    assert ledger(store) == {LOCK: sha256(b"gen-1")}
    ledger_path = worktree_setup.generated_ledger_path(store)
    assert ledger_path.parent == worktree_setup.setup_record_path(store).parent
    assert oct(ledger_path.stat().st_mode & 0o777) == "0o600"
    # The success record format is unchanged.
    assert "generated_keys" not in worktree_setup.read_setup_record(store)

    counter_b = tmp_path / "counter-b"
    changed = dataclasses.replace(config, worktree_setup=lock_command(counter_b, "gen-2"))
    run_worktrees.prepare_store(store, config=changed, runner=Runner())
    assert runs_in(counter_b, store) == 1
    assert ledger(store) == {LOCK: sha256(b"gen-2")}
    assert worktree_setup.read_setup_record(store)["command_sha256"] == worktree_setup.command_hash(
        changed.worktree_setup
    )


def test_unrecorded_key_is_not_exempted_by_the_ledger(tmp_path, monkeypatch):
    """gen-key-second-setup: only keys named in the ledger are exempt."""
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, lock_command(counter), keys=("requirements.txt", LOCK, "yarn.lock"))
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    (store / "yarn.lock").write_text("operator\n")
    changed = dataclasses.replace(config, worktree_setup=lock_command(tmp_path / "counter-b"))
    with pytest.raises(AgentLoopError, match="yarn.lock") as caught:
        run_worktrees.prepare_store(store, config=changed, runner=Runner())
    assert HINT in str(caught.value) and LOCK + "," not in str(caught.value)
    assert not (tmp_path / "counter-b").exists()


def _edit_bytes(store: Path) -> None:
    (store / LOCK).write_text("edited")


def _symlink_same_bytes(store: Path) -> None:
    target = store.parent / "lock-target"
    target.write_text("gen-1")
    (store / LOCK).unlink()
    os.symlink(target, store / LOCK)


def _directory(store: Path) -> None:
    (store / LOCK).unlink()
    (store / LOCK).mkdir()


@pytest.mark.parametrize("tamper", [_edit_bytes, _symlink_same_bytes, _directory])
def test_tampered_generated_key_is_refused_with_hint(tmp_path, monkeypatch, tamper):
    """gen-key-tampered."""
    env, counter, config, store = first_lock_setup(tmp_path, monkeypatch)
    tamper(store)
    record_bytes = worktree_setup.setup_record_path(store).read_bytes()
    ledger_bytes = worktree_setup.generated_ledger_path(store).read_bytes()
    counter_b = tmp_path / "counter-b"
    changed = dataclasses.replace(config, worktree_setup=lock_command(counter_b))
    with pytest.raises(AgentLoopError, match="absent at the base commit") as caught:
        run_worktrees.prepare_store(store, config=changed, runner=Runner())
    assert LOCK in str(caught.value) and HINT in str(caught.value)
    assert "npm install writes package-lock.json" in str(caught.value)
    assert worktree_setup.setup_record_path(store).read_bytes() == record_bytes
    assert worktree_setup.generated_ledger_path(store).read_bytes() == ledger_bytes
    assert not counter_b.exists()


@pytest.mark.parametrize("before_first", [True, False])
def test_operator_placed_absent_key_still_stops_with_hint(tmp_path, monkeypatch, before_first):
    """foreign-untracked-key."""
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = lock_config(env, setup_command(counter))  # never writes the lockfile
    store = claude_store()
    if before_first:
        run_worktrees.prepare_store(store, config=dataclasses.replace(config, worktree_setup=None, worktree_links=()), runner=Runner())
    else:
        run_worktrees.prepare_store(store, config=config, runner=Runner())
        assert ledger(store) == {}
        config = dataclasses.replace(config, worktree_setup=setup_command(counter, "mkdir -p .venv && touch .venv/ready && true"))
    runs = runs_in(counter, store)
    (store / LOCK).write_text("planted")
    with pytest.raises(AgentLoopError, match="absent at the base commit") as caught:
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert LOCK in str(caught.value) and HINT in str(caught.value)
    assert runs_in(counter, store) == runs
    assert (store / LOCK).read_text() == "planted"


def _ledger_malformed(store, path):
    path.write_text("{not json")


def _ledger_wrong_version(store, path):
    data = json_load(path)
    data["version"] = 2
    path.write_text(json_dump(data))


def _ledger_foreign_identity(store, path):
    data = json_load(path)
    data["store_identity"] = [1, 2]
    path.write_text(json_dump(data))


def _ledger_unconfigured_key(store, path):
    data = json_load(path)
    data["generated_keys"]["other.lock"] = sha256(b"x")
    path.write_text(json_dump(data))


def _ledger_bad_hash(store, path):
    data = json_load(path)
    data["generated_keys"][LOCK] = data["generated_keys"][LOCK].upper()
    path.write_text(json_dump(data))


def _ledger_not_a_map(store, path):
    data = json_load(path)
    data["generated_keys"] = [LOCK]
    path.write_text(json_dump(data))


def _ledger_symlink(store, path):
    target = path.with_name("elsewhere.json")
    target.write_bytes(path.read_bytes())
    path.unlink()
    os.symlink(target, path)


def _ledger_wrong_owner(store, path):
    real = os.getuid()
    worktree_setup_os = worktree_setup.os
    return lambda monkeypatch: monkeypatch.setattr(worktree_setup_os, "getuid", lambda: real + 1)


def json_load(path):
    import json

    return json.loads(path.read_text())


def json_dump(data):
    import json

    return json.dumps(data)


@pytest.mark.parametrize("corrupt", [
    _ledger_malformed, _ledger_wrong_version, _ledger_foreign_identity, _ledger_unconfigured_key,
    _ledger_bad_hash, _ledger_not_a_map, _ledger_symlink, _ledger_wrong_owner,
], ids=lambda f: f.__name__.strip("_"))
def test_untrusted_ledger_exempts_nothing(tmp_path, monkeypatch, corrupt):
    """record-untrusted-ledger."""
    env, counter, config, store = first_lock_setup(tmp_path, monkeypatch)
    path = worktree_setup.generated_ledger_path(store)
    patch = corrupt(store, path)
    if patch is not None:
        patch(monkeypatch)
    assert ledger(store) == {}
    if patch is not None:
        return  # every later file check also sees the foreign owner
    counter_b = tmp_path / "counter-b"
    changed = dataclasses.replace(config, worktree_setup=lock_command(counter_b))
    with pytest.raises(AgentLoopError, match=HINT):
        run_worktrees.prepare_store(store, config=changed, runner=Runner())
    assert not counter_b.exists()


def test_store_recreation_deletes_record_and_ledger(tmp_path, monkeypatch):
    """record-untrusted-ledger: a re-created store never inherits a ledger."""
    env, counter, config, store = first_lock_setup(tmp_path, monkeypatch)
    assert ledger(store) == {LOCK: sha256(b"gen-1")}
    seen: list[tuple[bool, bool]] = []
    real = worktree_setup.run_setup_command

    def recording(*args, **kwargs):
        seen.append((
            worktree_setup.setup_record_path(store).exists(),
            worktree_setup.generated_ledger_path(store).exists(),
        ))
        return real(*args, **kwargs)

    monkeypatch.setattr(worktree_setup, "run_setup_command", recording)
    shutil.rmtree(store)
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert seen == [(False, False)]
    worktree_setup.invalidate_store_setup_state(store)
    assert not worktree_setup.setup_record_path(store).exists()
    assert not worktree_setup.generated_ledger_path(store).exists()


@pytest.mark.parametrize("mode", ["nonzero", "timeout"])
@pytest.mark.parametrize("rewrites", [False, True])
def test_ledger_survives_a_failed_later_setup(tmp_path, monkeypatch, mode, rewrites):
    """ledger-survives-failed-setup."""
    env, counter, config, store = first_lock_setup(tmp_path, monkeypatch)
    ledger_bytes = worktree_setup.generated_ledger_path(store).read_bytes()
    failure = "exit 1" if mode == "nonzero" else "sleep 30"
    damage = f"printf rewritten > {LOCK}; " if rewrites else ""
    failing = dataclasses.replace(
        config, worktree_setup=setup_command(tmp_path / "counter-b", f"{damage}{failure}"),
        worktree_setup_timeout_seconds=2,
    )
    with pytest.raises(AgentLoopError):
        run_worktrees.prepare_store(store, config=failing, runner=Runner())
    assert worktree_setup.read_setup_record(store) is None
    assert worktree_setup.generated_ledger_path(store).read_bytes() == ledger_bytes
    counter_c = tmp_path / "counter-c"
    fixed = dataclasses.replace(config, worktree_setup=lock_command(counter_c))
    if rewrites:
        with pytest.raises(AgentLoopError, match=HINT):
            run_worktrees.prepare_store(store, config=fixed, runner=Runner())
        assert not counter_c.exists()
        return
    run_worktrees.prepare_store(store, config=fixed, runner=Runner())
    assert runs_in(counter_c, store) == 1
    assert ledger(store) == {LOCK: sha256(b"gen-1")}
    assert worktree_setup.record_matches(
        worktree_setup.read_setup_record(store), store=store,
        digest=worktree_setup.setup_digest(
            store, git(store, "rev-parse", "HEAD"), command=fixed.worktree_setup,
            keys=fixed.worktree_setup_keys, runner=Runner(),
        )[0],
        command=fixed.worktree_setup,
    )


def test_failed_setup_that_created_a_key_is_not_exempted(tmp_path, monkeypatch):
    """failed-setup-created-key."""
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = lock_config(env, setup_command(counter, f"printf partial > {LOCK}; exit 1"))
    store = claude_store()
    with pytest.raises(AgentLoopError, match="exited 1"):
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert worktree_setup.read_setup_record(store) is None
    assert ledger(store) == {}
    with pytest.raises(AgentLoopError, match=HINT):
        run_worktrees.prepare_store(store, config=lock_config(env, lock_command(counter)), runner=Runner())
    assert runs_in(counter, store) == 1
    assert (store / LOCK).read_text() == "partial"


def test_record_without_a_ledger_still_matches(tmp_path, monkeypatch):
    """legacy-record: a pre-#1649 record (no ledger file) forces no rerun."""
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, setup_command(counter))
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    worktree_setup.generated_ledger_path(store).unlink()
    record = worktree_setup.read_setup_record(store)
    assert set(record) == {"version", "digest", "command_sha256", "base_sha", "store_identity", "completed_at", "health_generation"}
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert runs_in(counter, store) == 1


def _exclude(store: Path, path: str) -> None:
    with (store / ".git" / "info" / "exclude").open("a") as handle:
        handle.write(f"/{path}\n")


@pytest.mark.parametrize("ignored", [False, True])
def test_matching_generated_key_is_removed_when_the_base_starts_tracking_it(tmp_path, monkeypatch, ignored):
    """gen-key-becomes-tracked: content-verified removal, then a successful checkout."""
    env, counter, config, store = first_lock_setup(tmp_path, monkeypatch)
    if ignored:
        _exclude(store, LOCK)
    new = env.commit(LOCK, "tracked\n")
    runner = RecordingRunner()
    assert run_worktrees.prepare_store(store, config=config, runner=runner) == new
    assert git(store, "rev-parse", "HEAD") == new
    assert (store / LOCK).read_text() == "tracked\n"
    assert runs_in(counter, store) == 2
    assert ledger(store) == {}
    checkouts = [call for call in runner.calls if call[:2] == ("git", "checkout") and new in call]
    assert checkouts and all("--no-overwrite-ignore" in call for call in checkouts)
    assert all("--force" not in call and "-f" not in call for call in checkouts)


def _changed_generated(store: Path) -> None:
    (store / LOCK).write_text("edited")


def _unrecorded(store: Path) -> None:
    worktree_setup.generated_ledger_path(store).unlink()


@pytest.mark.parametrize("ignored", [False, True])
@pytest.mark.parametrize("state", [_changed_generated, _unrecorded, _symlink_same_bytes])
def test_unverified_file_at_newly_tracked_key_is_refused_before_invalidation(tmp_path, monkeypatch, ignored, state):
    """gen-key-becomes-tracked: refusal before the record is touched or setup runs."""
    env, counter, config, store = first_lock_setup(tmp_path, monkeypatch)
    if ignored:
        _exclude(store, LOCK)
    state(store)
    before = os.readlink(store / LOCK) if (store / LOCK).is_symlink() else (store / LOCK).read_bytes()
    record_bytes = worktree_setup.setup_record_path(store).read_bytes()
    head = git(store, "rev-parse", "HEAD")
    env.commit(LOCK, "tracked\n")
    with pytest.raises(AgentLoopError, match="that the base commit tracks") as caught:
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert LOCK in str(caught.value) and HINT in str(caught.value)
    after = os.readlink(store / LOCK) if (store / LOCK).is_symlink() else (store / LOCK).read_bytes()
    assert after == before
    assert runs_in(counter, store) == 1
    assert worktree_setup.setup_record_path(store).read_bytes() == record_bytes
    assert git(store, "rev-parse", "HEAD") == head


def test_operator_ignored_file_at_newly_tracked_key_is_refused(tmp_path, monkeypatch):
    """gen-key-becomes-tracked: an operator file never recorded, ignored, is never overwritten."""
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = lock_config(env, setup_command(counter))
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    _exclude(store, LOCK)
    (store / LOCK).write_text("operator")
    env.commit(LOCK, "tracked\n")
    with pytest.raises(AgentLoopError, match="that the base commit tracks"):
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert (store / LOCK).read_text() == "operator"
    assert runs_in(counter, store) == 1


def test_removal_reverifies_bytes_immediately_before_unlink(tmp_path, monkeypatch):
    """gen-key-becomes-tracked: a file changed after classification is refused, not removed."""
    env, counter, config, store = first_lock_setup(tmp_path, monkeypatch)
    env.commit(LOCK, "tracked\n")
    real = worktree_setup.invalidate_setup_record

    def invalidate_then_edit(path):
        real(path)
        (store / LOCK).write_text("raced")

    monkeypatch.setattr(worktree_setup, "invalidate_setup_record", invalidate_then_edit)
    with pytest.raises(AgentLoopError, match=HINT):
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert (store / LOCK).read_text() == "raced"
    assert runs_in(counter, store) == 1


def test_ignored_non_key_file_at_newly_tracked_path_is_never_overwritten(tmp_path, monkeypatch):
    """ignored-nonkey-overwrite."""
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = setup_config(env, setup_command(counter))
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    _exclude(store, "generated.txt")
    (store / "generated.txt").write_text("operator\n")
    env.commit("generated.txt", "tracked\n")
    env.commit("requirements.txt", "pytest==7\n")
    with pytest.raises(AgentLoopError, match="could not check out"):
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert (store / "generated.txt").read_text() == "operator\n"
    assert worktree_setup.read_setup_record(store) is None
    assert runs_in(counter, store) == 1


def test_materialize_never_overwrites_ignored_files():
    """ignored-nonkey-overwrite: the store checkout always passes --no-overwrite-ignore."""
    calls: list[tuple[str, ...]] = []

    class Fake:
        def run(self, args, *, cwd, check):
            calls.append(tuple(args))
            if args[:2] == ("git", "rev-parse"):
                stdout = "b" * 40 if len(calls) > 2 else "a" * 40
                return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    worktree_setup._materialize(Path("/nonexistent"), "b" * 40, runner=Fake())
    assert ("git", "checkout", "-q", "--detach", "--no-overwrite-ignore", "b" * 40) in calls


def test_key_caused_checkout_refusal_carries_the_hint():
    class Fake:
        def run(self, args, *, cwd, check):
            if args[:2] == ("git", "rev-parse"):
                return subprocess.CompletedProcess(args, 0, stdout="a" * 40, stderr="")
            return subprocess.CompletedProcess(
                args, 1, stdout="", stderr=f"error: The following untracked working tree files would be overwritten by checkout:\n\t{LOCK}",
            )

    with pytest.raises(AgentLoopError, match=HINT):
        worktree_setup._materialize(Path("/nonexistent"), "b" * 40, runner=Fake(), keys=(LOCK,))
    with pytest.raises(AgentLoopError) as caught:
        worktree_setup._materialize(Path("/nonexistent"), "b" * 40, runner=Fake(), keys=("other.lock",))
    assert HINT not in str(caught.value)


def test_cli_worktree_setup_timeout_default_and_value(tmp_path, monkeypatch):
    """timeout-cli."""
    env = Env(tmp_path, monkeypatch)
    config = env.config(extra=("--worktree-setup", "true", "--coder-test-command-timeout-seconds", "5"))
    assert config.worktree_setup_timeout_seconds == 3600 == worktree_setup.DEFAULT_WORKTREE_SETUP_TIMEOUT_SECONDS
    assert config.coder_test_command_timeout_seconds == 5
    assert worktree_setup.setup_timeout_seconds(config) == 3600
    config = env.config(extra=("--worktree-setup", "true", "--worktree-setup-timeout", "7200"))
    assert config.worktree_setup_timeout_seconds == 7200
    assert worktree_setup.setup_timeout_seconds(config) == 7200


@pytest.mark.parametrize("extra, match", [
    (("--worktree-setup", "true", "--worktree-setup-timeout", "0"), "--worktree-setup-timeout must be a positive integer"),
    (("--worktree-setup", "true", "--worktree-setup-timeout", "-5"), "--worktree-setup-timeout must be a positive integer"),
    (("--worktree-setup", "true", "--worktree-setup-timeout", "1.5"), "--worktree-setup-timeout must be a positive integer"),
    (("--worktree-setup", "true", "--worktree-setup-timeout", "abc"), "--worktree-setup-timeout must be a positive integer"),
    (("--worktree-setup-timeout", "60"), "--worktree-setup-timeout requires --worktree-setup"),
])
def test_cli_rejects_invalid_worktree_setup_timeout(tmp_path, monkeypatch, extra, match):
    """timeout-cli."""
    env = Env(tmp_path, monkeypatch)
    with pytest.raises(AgentLoopError, match=match):
        env.config(extra=extra)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "60"])
def test_config_rejects_invalid_worktree_setup_timeout_field(tmp_path, monkeypatch, value):
    env = Env(tmp_path, monkeypatch)
    config = env.config(extra=("--worktree-setup", "true"))
    with pytest.raises(AgentLoopError, match="--worktree-setup-timeout must be a positive integer"):
        dataclasses.replace(config, worktree_setup_timeout_seconds=value)


def test_shared_clone_setup_timeout_names_the_flag(tmp_path, monkeypatch):
    """timeout-exceeded: killed, no record, and the message names --worktree-setup-timeout."""
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / "counter"
    config = dataclasses.replace(
        setup_config(env, setup_command(counter, "mkdir -p .venv && touch .venv/ready; sleep 30 & sleep 30")),
        worktree_setup_timeout_seconds=1, coder_test_command_timeout_seconds=600,
    )
    store = claude_store()
    started = time.monotonic()
    with pytest.raises(AgentLoopError) as caught:
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert time.monotonic() - started < 25
    message = str(caught.value)
    assert "timed out after 1s" in message
    assert "--worktree-setup-timeout SECONDS (currently 1s)" in message
    assert worktree_setup.read_setup_record(store) is None


def test_recovery_commands_preserve_worktree_setup_timeout(tmp_path):
    """timeout-recovery."""
    from agent_loop_helpers import make_config

    from coding_review_agent_loop.managed_ci import _RECOVERY_VALUE_OPTIONS, _render_recovery_command

    assert "--worktree-setup-timeout" in _RECOVERY_VALUE_OPTIONS
    config = make_config(tmp_path, invocation_argv=(
        "agent-loop", "issue", "123", "--repo", "OWNER/REPO",
        "--worktree-setup", "npm ci", "--worktree-setup-timeout", "7200",
        "--worktree-setup-key", "package-lock.json", "--worktree-link", "node_modules",
    ))
    for target, identifier in (("pr", 456), ("issue", 123)):
        command = _render_recovery_command(config, target=target, identifier=identifier, managed_ci=False)
        args = build_parser().parse_args(shlex.split(command)[1:])
        assert args.worktree_setup_timeout == "7200"
        assert args.worktree_setup == "npm ci"


def test_tracked_tree_probe_failure_names_the_checkout(tmp_path, monkeypatch):
    """The dirty-tree probe names the shared clone by default and the run worktree from PR-local setup (#1673)."""
    from coding_review_agent_loop import local_test_evidence, worktree_pr_env

    def fail(*_args, **_kwargs):
        raise AgentLoopError("probe exploded")

    monkeypatch.setattr(local_test_evidence, "_run_git", fail)
    with pytest.raises(AgentLoopError) as shared:
        worktree_setup.store_tracked_tree_dirty_paths(tmp_path)
    assert str(shared.value).startswith(
        f"Could not verify the tracked tree of the shared clone {tmp_path} before --worktree-setup: "
        "probe exploded"
    )
    run_path = tmp_path / "run-worktree"
    with pytest.raises(AgentLoopError) as run:
        worktree_pr_env._uncommitted_inputs(tmp_path, {}, runner=Runner(), path=run_path)
    message = str(run.value)
    assert message.startswith(f"Could not verify the tracked tree of the run worktree {run_path} ")
    assert "shared clone" not in message


@pytest.mark.parametrize('with_setup', [False, True])
@pytest.mark.parametrize('links', [('node_modules',), ('.venv', 'node_modules')])
def test_empty_source_rebuild_or_actionable_refusal(tmp_path, monkeypatch, with_setup, links):
    from coding_review_agent_loop import worktree_link_integrity as health
    env = seeded(tmp_path, monkeypatch, gitignore='/.venv\n/node_modules\n')
    store = env.seed_store('claude', venv='.venv' in links)
    (store / 'node_modules').mkdir()
    command = setup_command(tmp_path / 'counter', 'mkdir -p node_modules && touch node_modules/package')
    config = setup_config(env, command, links=links)
    if not with_setup:
        config = dataclasses.replace(config, worktree_setup=None)
        with pytest.raises(AgentLoopError) as caught:
            run_worktrees.prepare_store(store, config=config, runner=Runner())
        message = str(caught.value)
        for expected in (str(store.absolute()), str((store / 'node_modules').absolute()),
                         'node_modules', 'empty', '--worktree-setup CMD', "--worktree-setup 'npm ci'",
                         '--worktree-setup-pr-local', 'exits nonzero'):
            assert expected in message
        assert not config.claude_dir.exists()
    else:
        run_worktrees.prepare_store(store, config=config, runner=Runner())
        assert health.inspect(store, links)[1] is None
        assert worktree_setup.read_setup_record(store)


@pytest.mark.parametrize('failure', ['exit', 'empty', 'interrupt', 'publication'])
def test_stale_recovery_failure_retains_history_and_ledger(tmp_path, monkeypatch, failure):
    from coding_review_agent_loop import worktree_link_integrity as health
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / 'counter'
    config = setup_config(env, setup_command(counter))
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    ledger = worktree_setup.generated_ledger_path(store).read_bytes()
    with run_worktrees.store_lock(store):
        health.mark_stale(store, ('.venv',), 'turn damage')
    if failure == 'exit':
        monkeypatch.setattr(worktree_setup, 'run_setup_command', lambda *a, **k: (_ for _ in ()).throw(AgentLoopError('setup exited 1')))
    elif failure == 'interrupt':
        monkeypatch.setattr(worktree_setup, 'run_setup_command', lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    elif failure == 'empty':
        def empty(*a, **k):
            shutil.rmtree(store / '.venv')
            (store / '.venv').mkdir()
        monkeypatch.setattr(worktree_setup, 'run_setup_command', empty)
    else:
        monkeypatch.setattr(health, 'publish', lambda *a, **k: (_ for _ in ()).throw(OSError('publication failed')))
    with pytest.raises((AgentLoopError, KeyboardInterrupt)):
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert worktree_setup.read_setup_record(store) is None
    assert worktree_setup.generated_ledger_path(store).read_bytes() == ledger
    assert health.inspect(store, ('.venv',))[1]


@pytest.mark.parametrize('failure', ['timeout', 'dirty', 'held', 'success-record'])
def test_stale_recovery_guards_and_success_publication(tmp_path, monkeypatch, failure):
    from contextlib import contextmanager
    from coding_review_agent_loop import worktree_link_integrity as health
    env = seeded(tmp_path, monkeypatch)
    counter = tmp_path / 'counter'
    config = setup_config(env, setup_command(counter))
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    ledger = worktree_setup.generated_ledger_path(store).read_bytes()
    with run_worktrees.store_lock(store):
        health.mark_stale(store, ('.venv',), 'damaged by another run')
    if failure == 'timeout':
        config = dataclasses.replace(config, worktree_setup=setup_command(counter, 'sleep 30'),
                                     worktree_setup_timeout_seconds=1)
    elif failure == 'dirty':
        (store / 'a.txt').write_text('operator changes')
    elif failure == 'held':
        @contextmanager
        def held(_store):
            yield False
        monkeypatch.setattr(run_worktrees, 'probe_free_claim', held)
    else:
        def refuse(*a, **k):
            raise OSError('success record publication failed')
        monkeypatch.setattr(worktree_setup, '_write_setup_record', refuse)
    with pytest.raises((AgentLoopError, OSError)):
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert worktree_setup.read_setup_record(store) is None
    assert worktree_setup.generated_ledger_path(store).read_bytes() == ledger
    assert health.inspect(store, ('.venv',))[1]
    if failure in ('dirty', 'held'):
        assert runs_in(counter, store) == 1


def test_health_without_matching_setup_success_refuses_after_restart(tmp_path, monkeypatch):
    from coding_review_agent_loop import checkout_verification as cv, worktree_link_integrity as health
    env = seeded(tmp_path, monkeypatch)
    config = setup_config(env, setup_command(tmp_path / 'counter'))
    store = claude_store()
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    data = worktree_setup._read_private_json(health.health_path(store))
    assert data['setup_backed']
    assert worktree_setup.read_setup_record(store)['health_generation'] == data['generation']
    worktree_setup.setup_record_path(store).unlink()  # crash between the two publications
    cv.reset_checkout_baselines()
    without_setup = dataclasses.replace(config, worktree_setup=None)
    with pytest.raises(AgentLoopError, match='setup success is missing'):
        run_worktrees.prepare_store(store, config=without_setup, runner=Runner())
    run_worktrees.prepare_store(store, config=config, runner=Runner())
    assert health.inspect(store, ('.venv',))[1] is None


@pytest.mark.parametrize('with_setup', [False, True])
@pytest.mark.parametrize('failure', ['empty', 'root-open', 'root-count', 'nested-count', 'root-close'])
@pytest.mark.parametrize('links', [('node_modules',), ('.venv', 'node_modules')])
def test_initial_capture_failure_keeps_source_attribution(tmp_path, monkeypatch, with_setup, failure, links):
    from contextlib import contextmanager
    from coding_review_agent_loop import worktree_link_integrity as health
    env = seeded(tmp_path, monkeypatch, gitignore='/.venv\n/node_modules\n')
    store = env.seed_store('claude', venv='.venv' in links)
    source = store / 'node_modules'
    source.mkdir()
    if failure != 'empty':
        (source / 'package').mkdir()
        (source / 'package' / 'ready').touch()
    original_root = health._root
    if failure in ('root-open', 'root-close'):
        @contextmanager
        def fail_root(root_store, link):
            if link == 'node_modules' and failure == 'root-open':
                raise OSError('root-open failure')
            with original_root(root_store, link) as root:
                yield root
            if link == 'node_modules' and failure == 'root-close':
                raise OSError('root-close failure')
        monkeypatch.setattr(health, '_root', fail_root)
    elif failure in ('root-count', 'nested-count'):
        original_entries = health._entries
        def fail_entries(fd, path, **kwargs):
            # Identify the source by its anchored descriptor, including optional probes.
            if (os.fstat(fd).st_ino == source.stat().st_ino and
                    path == ('' if failure == 'root-count' else 'package')):
                raise OSError(f'{failure} failure')
            return original_entries(fd, path, **kwargs)
        monkeypatch.setattr(health, '_entries', fail_entries)
    config = setup_config(env, 'true', links=links)
    if not with_setup:
        config = dataclasses.replace(config, worktree_setup=None)
    with pytest.raises(AgentLoopError) as caught:
        run_worktrees.prepare_store(store, config=config, runner=Runner())
    message = str(caught.value)
    assert f"--worktree-link 'node_modules' source {str(source.absolute())!r}" in message
    for expected in (str(store.absolute()), '--worktree-setup CMD', "--worktree-setup 'npm ci'",
                     '--worktree-setup-pr-local', 'empty' if failure == 'empty' else f'{failure} failure'):
        assert expected in message
    assert not config.claude_dir.exists()
    assert worktree_setup.read_setup_record(store) is None
    if with_setup:
        data, reason, link = health.inspect(store, links)
        assert data['state'] == 'pending'
        assert data['stale_link'] == link == 'node_modules'
        assert reason in message
        # A subsequent recovery attempt must not replace the known failing source.
        with run_worktrees.store_lock(store):
            health.begin_recovery(store, links, 'retry required')
        assert health.inspect(store, links)[2] == 'node_modules'


@pytest.mark.parametrize('subcommand', ['issue', 'pr', 'task'])
def test_cli_help_describes_link_health_recovery(subcommand, capsys):
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args([subcommand, '--help'])
    assert caught.value.code == 0
    text = ' '.join(capsys.readouterr().out.split())
    assert 'reruns only' not in text
    assert 'Directory sources must be nonempty' in text
    assert 'damaged, stale or unknown source health' in text
    assert 'legacy setup records without health' in text
    assert 'Test dependencies in per-run worktrees' in text
