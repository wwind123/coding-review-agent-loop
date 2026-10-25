"""PR-local dependency environments behind --worktree-setup-pr-local (#1652, parent #1630).

Every scenario uses real temporary Git repositories, a local bare origin, the
production ``Runner`` and the production preparation flow (``ensure_agent_workdirs``
followed by the initial reconcile).  The setup command appends its working
directory to a counter file outside every repository and creates
``./node_modules``, so each test can count where setup ran.  Markers in the
committed ``package-lock.json`` select failure modes of the same command.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_review_agent_loop import checkout_verification as cv
from coding_review_agent_loop import worktree_pr_env
from coding_review_agent_loop.agents import registry
from coding_review_agent_loop.agents.base import AgentResult
from coding_review_agent_loop.config import (
    default_agent_workdir,
    ensure_agent_workdirs,
    sync_checkout_to_pr,
    sync_coder_base_before_implementation,
)
from coding_review_agent_loop.errors import AgentLoopError, CheckoutVerificationError
from coding_review_agent_loop.local_test_evidence import capture_tracked_tree_snapshot
from coding_review_agent_loop.runner import Runner
from coding_review_agent_loop.workdir_claims import claim_agent_workdirs, workdir_claim_scope

from test_run_worktrees import Env, RecordingRunner, git

LOCK = "package-lock.json"
SANDBOXED = (
    "--agent-permissions", "sandboxed", "--repair-backend", "claude", "--repair-model", "m",
    "--no-semantic-followup-dedupe",
)


# --- helpers ---------------------------------------------------------------


def setup_command(counter: Path) -> str:
    return (
        f'echo "$PWD" >> {shlex.quote(str(counter))}; '
        f"if grep -q SLOW {LOCK}; then sleep 30; fi; "
        f"if grep -q FAIL {LOCK}; then echo boom-tail; exit 3; fi; "
        'case "$PWD" in */runs/*) '
        f'if grep -q LINKSTORE {LOCK}; then ln -s "$PR_ENV_STORE_SOURCE" node_modules; exit 0; fi; '
        f'if grep -q LINKOUT {LOCK}; then ln -s "$PR_ENV_OUTSIDE" node_modules; exit 0; fi; '
        f"if grep -q FILEENV {LOCK}; then cp {LOCK} node_modules; exit 0; fi;; "
        "esac; "
        f"mkdir -p node_modules && cp {LOCK} node_modules/lock-copy"
    )


def runs_in(counter: Path, path: Path) -> int:
    if not counter.exists():
        return 0
    real = os.path.realpath(path)
    return sum(1 for line in counter.read_text().splitlines() if os.path.realpath(line) == real)


class PrEnv:
    def __init__(self, tmp_path: Path, monkeypatch, *, gitignore="/node_modules\n", pre=None):
        self.tmp_path = tmp_path
        self.env = env = Env(tmp_path, monkeypatch)
        if pre is not None:
            pre(env)
        env.commit(".gitignore", gitignore)
        env.commit(LOCK, "{}\n")
        self.counter = tmp_path / "counter"
        self.outside = tmp_path / "outside-env"
        self.outside.mkdir()
        (self.outside / "keep").write_text("outside\n")
        monkeypatch.setenv("PR_ENV_OUTSIDE", str(self.outside))
        monkeypatch.setenv("PR_ENV_STORE_SOURCE", str(self.store("claude") / "node_modules"))
        self.messages: list[str] = []
        monkeypatch.setattr(worktree_pr_env, "log", lambda _config, message: self.messages.append(message))

    def store(self, agent: str) -> Path:
        return default_agent_workdir("OWNER/REPO", agent).resolve()

    def config(self, *, flag=True, sandboxed=False, keys=(LOCK,), extra=()):
        argv = ["--worktree-setup", setup_command(self.counter), "--worktree-link", "node_modules"]
        for key in keys:
            argv += ["--worktree-setup-key", key]
        if flag:
            argv.append("--worktree-setup-pr-local")
        if sandboxed:
            argv += list(SANDBOXED)
        return self.env.config(extra=(*argv, *extra))

    def prepare(self, config, runner=None):
        runner = runner or Runner()
        claim_agent_workdirs(config)
        ensure_agent_workdirs(config, runner)
        return runner

    def push_pr_head(self, number: int, files: dict[str, str]) -> str:
        seed = self.env.seed
        git(seed, "checkout", "-q", "-B", f"pr-{number}", "main")
        for name, text in files.items():
            (seed / name).write_text(text)
            git(seed, "add", name)
        git(seed, "commit", "-q", "-m", f"pr {number}")
        sha = git(seed, "rev-parse", "HEAD")
        git(seed, "push", "-q", "origin", f"HEAD:refs/pull/{number}/head")
        git(seed, "checkout", "-q", "main")
        return sha


def commit_in(root: Path, name: str, text: str) -> str:
    (root / name).write_text(text)
    git(root, "add", name)
    git(root, "commit", "-q", "-m", f"change {name}")
    return git(root, "rev-parse", "HEAD")


def end_turn(config, root: Path) -> None:
    """What ``run_agent_result``'s finally block does after a coder turn."""
    cv.record_checkout_baseline(config, Runner(), root, source="claude coder turn")


def source_state(store: Path) -> tuple[tuple[int, int], dict[str, bytes]]:
    source = store / "node_modules"
    info = os.lstat(source)
    files = {
        os.path.relpath(os.path.join(base, name), source): Path(base, name).read_bytes()
        for base, _dirs, names in os.walk(source) for name in names
    }
    return (info.st_dev, info.st_ino), files


def private_identity(root: Path) -> tuple[int, int]:
    info = os.lstat(root / "node_modules")
    return info.st_dev, info.st_ino


def poison_reason(config, root: Path) -> str:
    with pytest.raises(CheckoutVerificationError) as caught:
        cv.verify_checkout(config, Runner(), path=root, agent="claude", purpose="agent turn")
    return str(caught.value)


class FakeBackend:
    display_name = "Fake"
    signature = "Fake"

    def __init__(self, agent: str, hook=None):
        self.agent = agent
        self.hook = hook
        self.calls = 0

    def workdir(self, config):
        return getattr(config, f"{self.agent}_dir")

    def run(self, runner, config, prompt, **kwargs):
        self.calls += 1
        if self.hook is not None:
            self.hook(self.workdir(config))
        return AgentResult(text="ok")


def run_turn(monkeypatch, config, agent: str, role: str, hook=None) -> FakeBackend:
    backend = FakeBackend(agent, hook)
    monkeypatch.setitem(registry.BACKENDS, agent, backend)
    runner = Runner()
    runner.configure_from_config = None
    registry.run_agent_result(runner, agent=agent, config=config, prompt="p", role=role)
    return backend


# --- flag-off-unchanged --------------------------------------------------------


def test_flag_off_key_change_with_replaced_link_is_refused_as_today(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config(flag=False)
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        assert (root / "node_modules").is_symlink()
        assert cv.environment_record(root) is None
        commit_in(root, LOCK, '{"v": 2}\n')
        (root / "node_modules").unlink()
        (root / "node_modules").mkdir()
        end_turn(config, root)
        reason = poison_reason(config, root)
    assert "can no longer be trusted" in reason
    assert "worktree link node_modules was removed, retargeted or reached through a symlink" in reason
    assert "--worktree-setup-pr-local" in reason
    assert "--<agent>-dir checkout" in reason
    assert "not yet available" not in reason
    assert runs_in(s.counter, root) == 0
    assert not s.messages


# --- in-turn-install-isolated ----------------------------------------------------


def test_writable_worktree_is_private_before_its_first_turn(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config()
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        store = s.store("claude")
        before = source_state(store)
        seen = {}

        def coder_turn(workdir: Path) -> None:
            # No symlink into the store exists when the turn starts.
            seen["symlink"] = (workdir / "node_modules").is_symlink()
            commit_in(workdir, LOCK, '{"v": 2}\n')
            (workdir / "node_modules" / "installed-in-turn").write_text("x\n")

        backend = run_turn(monkeypatch, config, "claude", "coder", coder_turn)
        assert backend.calls == 1 and seen == {"symlink": False}
        assert source_state(store) == before
        assert not (store / "node_modules" / "installed-in-turn").exists()
        # The committed key change was rebuilt privately after the turn.
        registration = cv.link_registration(root)
        assert registration.env_generation == 2 and not registration.links
        assert (root / "node_modules" / "lock-copy").read_text() == '{"v": 2}\n'
        assert not cv.has_poisoned_entry(root)
    assert runs_in(s.counter, root) == 2


def test_backstop_refuses_writable_turn_while_links_remain(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config()
    # Force the initial privatization to defer, so the writable worktree keeps its links.
    monkeypatch.setattr(worktree_pr_env, "_uncommitted_inputs", lambda *a, **k: ["a.txt"])
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        assert (root / "node_modules").is_symlink()
        backend = FakeBackend("claude")
        monkeypatch.setitem(registry.BACKENDS, "claude", backend)
        runner = Runner()
        runner.configure_from_config = None
        with pytest.raises(CheckoutVerificationError, match="still link into the shared clone"):
            registry.run_agent_result(runner, agent="claude", config=config, prompt="p", role="coder")
        assert backend.calls == 0


def test_sandboxed_read_only_reviewer_keeps_links_and_coder_role_is_refused(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config(sandboxed=True)
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        coder_root, reviewer_root = config.claude_dir, config.codex_dir
        assert not (coder_root / "node_modules").is_symlink()
        assert (reviewer_root / "node_modules").is_symlink()
        assert cv.link_registration(reviewer_root).writable is False
        backend = run_turn(monkeypatch, config, "codex", "reviewer")
        assert backend.calls == 1
        assert (reviewer_root / "node_modules").is_symlink()
        # A later coder-role turn in the read-only-registered worktree is refused.
        backend = FakeBackend("codex")
        monkeypatch.setitem(registry.BACKENDS, "codex", backend)
        runner = Runner()
        runner.configure_from_config = None
        with pytest.raises(CheckoutVerificationError, match="still link into the shared clone"):
            registry.run_agent_result(runner, agent="codex", config=config, prompt="p", role="coder")
        assert backend.calls == 0
    assert runs_in(s.counter, reviewer_root) == 0
    assert runs_in(s.counter, coder_root) == 1


# --- no-key-change-refused ---------------------------------------------------------


@pytest.mark.parametrize("replacement", ["delete", "directory", "file", "store-symlink"])
def test_private_environment_replaced_without_key_change_is_refused(tmp_path, monkeypatch, replacement):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config()
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        private = root / "node_modules"
        shutil.rmtree(private)
        if replacement == "directory":
            private.mkdir()
        elif replacement == "file":
            private.write_text("coder-made\n")
        elif replacement == "store-symlink":
            private.symlink_to(s.store("claude") / "node_modules")
        commit_in(root, "a.txt", "non-key change\n")
        end_turn(config, root)
        reason = poison_reason(config, root)
    assert "private dependency environment node_modules was removed, replaced" in reason
    assert "no --worktree-setup-key file changed at HEAD" in reason
    assert runs_in(s.counter, root) == 1


# --- key-change-intact-link / pr-mode-head-change ----------------------------------------


def test_read_only_worktree_privatizes_on_key_changing_pr_sync(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config(sandboxed=True)
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.codex_dir
        store = s.store("codex")
        before = source_state(store)
        unchanged = s.push_pr_head(6, {"a.txt": "pr six\n"})
        sync_checkout_to_pr(
            config, Runner(), path=root, label="Default codex workdir", default_owned=True,
            pr_number=6, pr_metadata=SimpleNamespace(head_sha=unchanged),
        )
        assert (root / "node_modules").is_symlink() and runs_in(s.counter, root) == 0
        head = s.push_pr_head(7, {LOCK: '{"v": 7}\n'})
        sync_checkout_to_pr(
            config, Runner(), path=root, label="Default codex workdir", default_owned=True,
            pr_number=7, pr_metadata=SimpleNamespace(head_sha=head),
        )
        private = root / "node_modules"
        assert private.is_dir() and not private.is_symlink()
        assert (private / "lock-copy").read_text() == '{"v": 7}\n'
        assert runs_in(s.counter, root) == 1
        assert source_state(store) == before
        registration = cv.link_registration(root)
        assert set(registration.private) == {b"node_modules"} and not registration.links
        # The baseline was recorded after the rebuild, so it is not poisoned and verifies.
        assert not cv.has_poisoned_entry(root)
        cv.verify_checkout(config, Runner(), path=root, agent="codex", purpose="review turn")
    assert any(
        message.startswith("PR-local dependency environment: rebuilt node_modules privately in "
                           f"{root} for changed setup keys {LOCK} at {head}")
        and f"the shared clone {store} was not modified" in message
        for message in s.messages
    )


def test_re_preparation_keeps_registration_and_refuses_deleted_private_path(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config()
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        first = cv.link_registration(root)
        identity = private_identity(root)
        marker = (root / "node_modules" / "lock-copy").read_bytes()
        # run_pr_loop without workdirs_ready re-enters preparation on the same path.
        ensure_agent_workdirs(config, Runner())
        again = cv.link_registration(root)
        assert (again.env_nonce, again.env_generation, again.private) == (
            first.env_nonce, first.env_generation, first.private,
        )
        assert private_identity(root) == identity
        assert (root / "node_modules" / "lock-copy").read_bytes() == marker
        shutil.rmtree(root / "node_modules")
        with pytest.raises(CheckoutVerificationError, match="node_modules"):
            ensure_agent_workdirs(config, Runner())
        assert not os.path.lexists(root / "node_modules")
        assert cv.link_registration(root).env_nonce == first.env_nonce
    assert runs_in(s.counter, root) == 1


# --- coder-replacement-rebuilt / rebuild-on-second-change ---------------------------------


@pytest.mark.parametrize("replacement", ["directory", "file", "store-symlink"])
def test_coder_replacement_with_key_change_is_removed_and_rebuilt(tmp_path, monkeypatch, replacement):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config()
    removed_trees: list[str] = []
    real_rmtree = shutil.rmtree

    def spy_rmtree(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None:
            removed_trees.append(os.fsdecode(path))
        return real_rmtree(path, *args, **kwargs)

    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        store = s.store("claude")
        before = source_state(store)
        private = root / "node_modules"
        shutil.rmtree(private)
        if replacement == "directory":
            private.mkdir()
            (private / "coder-made").write_text("x\n")
        elif replacement == "file":
            private.write_text("coder-made\n")
        else:
            private.symlink_to(store / "node_modules")
        commit_in(root, LOCK, '{"v": 2}\n')
        with monkeypatch.context() as patched:
            patched.setattr(worktree_pr_env.shutil, "rmtree", spy_rmtree)
            end_turn(config, root)
        assert not cv.has_poisoned_entry(root)
        assert private.is_dir() and not private.is_symlink()
        assert not (private / "coder-made").exists()
        assert (private / "lock-copy").read_text() == '{"v": 2}\n'
        assert cv.link_registration(root).env_generation == 2
    assert removed_trees == (["node_modules"] if replacement == "directory" else [])
    assert source_state(store) == before
    assert runs_in(s.counter, root) == 2


def test_rebuild_on_second_key_change_and_regular_file_environment(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config()
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        commit_in(root, LOCK, '{"v": "FILEENV"}\n')
        end_turn(config, root)
        assert (root / "node_modules").is_file()
        assert cv.link_registration(root).env_generation == 2
        # An unchanged digest runs nothing.
        commit_in(root, "a.txt", "non-key\n")
        end_turn(config, root)
        assert runs_in(s.counter, root) == 2
        # A second key change unlinks the private regular file and rebuilds.
        commit_in(root, LOCK, '{"v": 3}\n')
        end_turn(config, root)
        assert (root / "node_modules").is_dir() and not (root / "node_modules").is_symlink()
        assert (root / "node_modules" / "lock-copy").read_text() == '{"v": 3}\n'
        registration = cv.link_registration(root)
        assert registration.env_generation == 3 and not registration.links
        assert not cv.has_poisoned_entry(root)
        # Returning the keys to the base digest keeps a private rebuild; it never re-links.
        commit_in(root, LOCK, "{}\n")
        end_turn(config, root)
        assert not (root / "node_modules").is_symlink()
        assert cv.link_registration(root).env_generation == 4
    assert runs_in(s.counter, root) == 4


# --- uncommitted-key ---------------------------------------------------------------------


def _uncommitted_key(root):
    (root / LOCK).write_text('{"v": "uncommitted"}\n')
    return LOCK


def _modified_non_key(root):
    (root / "a.txt").write_text("edited\n")
    return "a.txt"


def _assume_unchanged_key(root):
    git(root, "update-index", "--assume-unchanged", LOCK)
    (root / LOCK).write_text('{"v": "hidden"}\n')
    return LOCK


def _skip_worktree_key(root):
    git(root, "update-index", "--skip-worktree", LOCK)
    (root / LOCK).write_text('{"v": "hidden"}\n')
    return LOCK


def _ignored_copy_of_deleted_key(root):
    git(root, "rm", "-q", "yarn.lock")
    git(root, "commit", "-q", "-m", "drop yarn.lock")
    (root / "yarn.lock").write_text("planted\n")
    return "yarn.lock"


def _drifted_submodule_key(root):
    pop = root / "mods" / "pop"
    if pop.exists():
        pop.rmdir()
    subprocess.run(["git", "clone", "-q", os.environ["PR_ENV_SUBMODULE"], str(pop)], check=True)
    (pop / "s.txt").write_text("drift\n")
    git(pop, "commit", "-q", "-am", "drift")
    return "mods/pop"


def _seed_extras(tmp_path: Path, monkeypatch):
    def pre(env):
        (env.seed / "yarn.lock").write_text("yarn\n")
        git(env.seed, "add", "yarn.lock")
        sub = tmp_path / "sub"
        subprocess.run(["git", "init", "-q", "-b", "main", str(sub)], check=True)
        (sub / "s.txt").write_text("sub\n")
        git(sub, "add", "s.txt")
        git(sub, "commit", "-q", "-m", "sub")
        monkeypatch.setenv("PR_ENV_SUBMODULE", str(sub))
        git(env.seed, "update-index", "--add", "--cacheinfo", f"160000,{git(sub, 'rev-parse', 'HEAD')},mods/pop")
        git(env.seed, "commit", "-q", "-m", "extras")
        git(env.seed, "push", "-q", "origin", "HEAD:refs/heads/main")

    return pre


UNCOMMITTED = [
    _uncommitted_key, _modified_non_key, _assume_unchanged_key, _skip_worktree_key,
    _ignored_copy_of_deleted_key, _drifted_submodule_key,
]
EXTRA_KEYS = (LOCK, "yarn.lock", "mods/pop")


@pytest.mark.parametrize("hide", UNCOMMITTED, ids=lambda f: f.__name__.strip("_"))
def test_uncommitted_inputs_defer_rebuild_after_a_coder_turn(tmp_path, monkeypatch, hide):
    s = PrEnv(tmp_path, monkeypatch, gitignore="/node_modules\n/yarn.lock\n", pre=_seed_extras(tmp_path, monkeypatch))
    config = s.config(keys=EXTRA_KEYS)
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        identity = private_identity(root)
        commit_in(root, LOCK, '{"v": 2}\n')
        offending = hide(root)
        end_turn(config, root)
        assert not cv.has_poisoned_entry(root)
        assert private_identity(root) == identity
        assert (root / "node_modules" / "lock-copy").read_text() == "{}\n"
        assert cv.link_registration(root).env_generation == 1
        assert offending in " ".join(cv.link_registration(root).deferred)
        # Replacing the deferred environment is refused, naming the offending paths.
        shutil.rmtree(root / "node_modules")
        end_turn(config, root)
        reason = poison_reason(config, root)
    assert "deferred the PR-local rebuild" in reason and offending in reason
    assert runs_in(s.counter, root) == 1
    deferrals = [message for message in s.messages if "deferred the rebuild" in message]
    assert deferrals
    # #1649: only an absent-key shadow carries the setup-created hint.
    hinted = any("created by the --worktree-setup command itself" in message for message in deferrals)
    assert hinted == (hide is _ignored_copy_of_deleted_key)


def test_pr_local_shadow_deferral_keeps_file_and_names_setup_command(tmp_path, monkeypatch):
    """#1649 pr-local-shadow-hint: no exemption in PR-local worktrees, only the hint."""
    s = PrEnv(tmp_path, monkeypatch, gitignore="/node_modules\n/yarn.lock\n", pre=_seed_extras(tmp_path, monkeypatch))
    config = s.config(keys=EXTRA_KEYS)
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        git(root, "rm", "-q", "yarn.lock")
        commit_in(root, LOCK, '{"v": 3}\n')
        # As if a setup had produced it: still deferred, never exempted or removed.
        (root / "yarn.lock").write_text("generated by setup\n")
        end_turn(config, root)
        assert "yarn.lock" in " ".join(cv.link_registration(root).deferred)
        assert (root / "yarn.lock").read_text() == "generated by setup\n"
    assert runs_in(s.counter, root) == 1
    message = next(message for message in s.messages if "deferred the rebuild" in message)
    assert "yarn.lock (untracked or ignored, absent at HEAD)" in message
    assert "may have been created by the --worktree-setup command itself" in message
    assert "nothing was removed" in message


def test_both_setup_sites_use_worktree_setup_timeout(tmp_path, monkeypatch):
    """#1649 timeout-exceeded: the PR-local rebuild receives --worktree-setup-timeout."""
    from coding_review_agent_loop import worktree_setup

    s = PrEnv(tmp_path, monkeypatch)
    config = dataclasses.replace(
        s.config(), worktree_setup_timeout_seconds=1234, coder_test_command_timeout_seconds=5
    )
    seen: list[tuple[str, float]] = []
    real = worktree_setup.run_setup_command

    def recording(store, command, *, timeout_seconds, **kwargs):
        seen.append((kwargs.get("where") or "shared", timeout_seconds))
        return real(store, command, timeout_seconds=timeout_seconds, **kwargs)

    monkeypatch.setattr(worktree_setup, "run_setup_command", recording)
    with workdir_claim_scope(command="task"):
        s.prepare(config)
    assert any(where.startswith("the run worktree") for where, _ in seen)
    assert any(where == "shared" for where, _ in seen)
    assert {timeout for _, timeout in seen} == {1234}


def test_drifted_submodule_defers_read_only_privatization_after_sync(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch, gitignore="/node_modules\n/yarn.lock\n", pre=_seed_extras(tmp_path, monkeypatch))
    config = s.config(sandboxed=True, keys=EXTRA_KEYS)
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.codex_dir
        # A drifted populated submodule survives the sync's reset and clean.
        _drifted_submodule_key(root)
        cv.record_checkout_baseline(config, Runner(), root, source="test setup")
        head = s.push_pr_head(8, {LOCK: '{"v": 8}\n'})
        sync_checkout_to_pr(
            config, Runner(), path=root, label="Default codex workdir", default_owned=True,
            pr_number=8, pr_metadata=SimpleNamespace(head_sha=head),
        )
        assert git(root, "rev-parse", "HEAD") == head
        assert (root / "node_modules").is_symlink()
        assert "mods/pop" in " ".join(cv.link_registration(root).deferred)
        assert not cv.has_poisoned_entry(root)
    assert runs_in(s.counter, root) == 0


# --- setup-failure ----------------------------------------------------------------------


@pytest.mark.parametrize("marker", ["FAIL", "SLOW"])
def test_setup_failure_after_a_turn_poisons_with_output_tail(tmp_path, monkeypatch, marker):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config()
    if marker == "SLOW":
        config = dataclasses.replace(config, worktree_setup_timeout_seconds=1, coder_test_command_timeout_seconds=600)
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        before = source_state(s.store("claude"))
        commit_in(root, LOCK, f'{{"v": "{marker}"}}\n')
        end_turn(config, root)  # never raises out of the finally block
        reason = poison_reason(config, root)
        assert source_state(s.store("claude")) == before
    assert "can no longer be trusted" in reason
    if marker == "FAIL":
        assert "exited 3 in the run worktree" in reason and "boom-tail" in reason
    else:
        assert "timed out after 1s in the run worktree" in reason
        assert "--worktree-setup-timeout" in reason


def test_setup_failure_at_sync_raises_and_at_startup_creates_no_entry(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config(sandboxed=True)
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.codex_dir
        head = s.push_pr_head(9, {LOCK: '{"v": "FAIL"}\n'})
        with pytest.raises(CheckoutVerificationError, match="boom-tail"):
            sync_checkout_to_pr(
                config, Runner(), path=root, label="Default codex workdir", default_owned=True,
                pr_number=9, pr_metadata=SimpleNamespace(head_sha=head),
            )
        assert cv.has_poisoned_entry(root)
    # Startup: the initial privatization of a writable worktree fails.
    (tmp_path / "second").mkdir()
    s2 = PrEnv(tmp_path / "second", monkeypatch)
    failing = 'case "$PWD" in */runs/*) echo boom-tail; exit 3;; esac; ' + setup_command(s2.counter)
    config2 = dataclasses.replace(s2.config(), worktree_setup=failing)
    with workdir_claim_scope(command="task"):
        with pytest.raises(CheckoutVerificationError, match="boom-tail"):
            s2.prepare(config2)
        assert not cv.has_entry(config2.claude_dir)


# --- private-path-integrity ---------------------------------------------------------------


@pytest.mark.parametrize("marker", ["LINKSTORE", "LINKOUT"])
def test_setup_that_leaves_a_symlink_is_refused(tmp_path, monkeypatch, marker):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config()
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        commit_in(root, LOCK, f'{{"v": "{marker}"}}\n')
        end_turn(config, root)
        reason = poison_reason(config, root)
        assert cv.link_registration(root).env_generation == 1
    assert "as a symlink" in reason
    assert (s.outside / "keep").read_text() == "outside\n"


@pytest.mark.parametrize("change", ["stage", "commit"])
def test_tracked_private_environment_is_refused_by_capture_and_evidence(tmp_path, monkeypatch, change):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config()
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        assert capture_tracked_tree_snapshot(root).complete is True
        git(root, "add", "-f", "node_modules/lock-copy")
        if change == "commit":
            git(root, "commit", "-q", "-m", "track the environment")
        snapshot = capture_tracked_tree_snapshot(root)
        assert snapshot.complete is False
        assert "tracked or staged" in " ".join(snapshot.caveats)
        end_turn(config, root)
        reason = poison_reason(config, root)
    assert "private dependency environment node_modules is staged in the index" in reason


def test_replaced_private_environment_is_refused_by_the_evidence_snapshot(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config()
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        intact = capture_tracked_tree_snapshot(root)
        assert intact.complete and not any(p.startswith("node_modules") for p in intact.untracked_paths)
        shutil.rmtree(root / "node_modules")
        (root / "node_modules").symlink_to(s.store("claude") / "node_modules")
        snapshot = capture_tracked_tree_snapshot(root)
    assert snapshot.complete is False
    assert "registered private environment was removed or replaced" in " ".join(snapshot.caveats)


# --- root-redirect --------------------------------------------------------------------------


def test_redirected_root_is_poisoned_before_any_removal_or_setup(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config()
    foreign = tmp_path / "foreign"
    subprocess.run(["git", "clone", "-q", str(s.env.bare), str(foreign)], check=True)
    commit_in(foreign, LOCK, '{"v": "foreign"}\n')
    (foreign / "node_modules").mkdir()
    (foreign / "node_modules" / "foreign-dep").write_text("keep\n")
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        commit_in(root, LOCK, '{"v": 2}\n')
        root.rename(root.with_name(root.name + "-aside"))
        root.symlink_to(foreign)
        try:
            end_turn(config, root)
            reason = poison_reason(config, root)
        finally:
            root.unlink()
            root.with_name(root.name + "-aside").rename(root)
    assert "no longer the directory agent-loop prepared" in reason
    assert (foreign / "node_modules" / "foreign-dep").read_text() == "keep\n"
    assert runs_in(s.counter, foreign) == 0
    assert runs_in(s.counter, root) == 1


# --- clean-dispatch-private ----------------------------------------------------------------


def test_sync_cleaning_keeps_an_unignored_private_environment(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch, gitignore="# nothing ignored\n")
    config = s.config()
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        assert not cv.registered_links(root) and cv.private_paths(root)
        identity = private_identity(root)
        (root / "dirt.txt").write_text("x\n")
        (root / "a.txt").write_text("changed\n")
        end_turn(config, root)
        runner = RecordingRunner()
        sync_coder_base_before_implementation(config, runner)
        assert not (root / "dirt.txt").exists()
        assert (root / "a.txt").read_text() == "one"
        assert private_identity(root) == identity
        assert (root / "node_modules" / "lock-copy").read_text() == "{}\n"
        assert ("git", "clean", "-fd", "-e", "/node_modules") in runner.calls
        assert ("git", "clean", "-fd") not in runner.calls
        cv.verify_checkout(config, Runner(), path=root, agent="claude", purpose="agent turn")


# --- explicit-dir-noop --------------------------------------------------------------------


def test_reconcile_is_a_noop_for_explicit_checkouts_and_dry_run(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config()
    explicit = tmp_path / "explicit"
    subprocess.run(["git", "clone", "-q", str(s.env.bare), str(explicit)], check=True)
    assert worktree_pr_env.reconcile_pr_local_environment(config, Runner(), explicit) is None
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        commit_in(root, LOCK, '{"v": 2}\n')
        assert worktree_pr_env.reconcile_pr_local_environment(config, Runner(dry_run=True), root) is None
        assert cv.link_registration(root).env_generation == 1
    assert runs_in(s.counter, explicit) == 0 and runs_in(s.counter, root) == 1


def test_flag_off_registration_never_reconciles_or_records_an_environment(tmp_path, monkeypatch):
    s = PrEnv(tmp_path, monkeypatch)
    config = s.config(flag=False)
    with workdir_claim_scope(command="task"):
        s.prepare(config)
        root = config.claude_dir
        on = dataclasses.replace(config, worktree_setup_pr_local=True)
        # A flag-off registration stays inert even for a flag-on config.
        assert worktree_pr_env.reconcile_pr_local_environment(on, Runner(), root) is None
        assert cv.environment_record(root) is None
        assert (root / "node_modules").is_symlink()
