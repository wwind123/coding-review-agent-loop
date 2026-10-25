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


def setup_command(counter: Path, tail: str = "mkdir -p .venv") -> str:
    return f'echo "$PWD" >> {shlex.quote(str(counter))} && {tail}'


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

    changed = dataclasses.replace(config, worktree_setup=setup_command(counter, "mkdir -p .venv && true"))
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
        coder_test_command_timeout_seconds=1,
    )
    store = claude_store()
    with pytest.raises(AgentLoopError, match="timed out after 1s") as caught:
        run_worktrees.prepare_store(store, config=config, runner=Runner())
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
        config_a, worktree_setup=setup_command(counter_b, damage), coder_test_command_timeout_seconds=2,
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
    with pytest.raises(AgentLoopError, match="absent.lock"):
        run_worktrees.prepare_store(s.store, config=s.config(), runner=Runner())
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
    config = setup_config(env, setup_command(counter, "sleep 1 && mkdir -p .venv"))
    store = claude_store()
    run_worktrees.prepare_store(  # clone once so both threads contend only on setup
        store, config=dataclasses.replace(config, worktree_setup=None), runner=Runner()
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
    recreate = '{ if [ -L .venv ]; then mkdir -p "$(readlink .venv)"; else mkdir -p .venv; fi; }'
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
