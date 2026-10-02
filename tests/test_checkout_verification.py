"""Per-turn checkout verification (#1130), exercised against real git repositories."""

import fcntl
import os
import subprocess
from pathlib import Path

import pytest

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop import checkout_verification as cv
from coding_review_agent_loop.agents import registry
from coding_review_agent_loop.agents.antigravity import (
    GEMINI_MD_NAME,
    SafeGeminiAccessError,
    _git_lock_path,
    build_gemini_injection,
    safe_read_gemini_md,
    single_shot_session_instruction,
)
from coding_review_agent_loop.agents.base import AgentResult
from coding_review_agent_loop.errors import AgentLoopError, CheckoutVerificationError
from coding_review_agent_loop.runner import Runner

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
}


def git(cwd, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, capture_output=True, env=GIT_ENV
    )


class GitRunner(FakeRunner):
    """Scripted gh/agent behaviour, but checkout probes hit the real git."""

    def run_binary(self, args, **kwargs):
        return Runner.run_binary(self, args, **kwargs)


class StubBackend:
    name = "codex"
    display_name = "Codex"
    signature = "OpenAI Codex"

    def __init__(self, workdir, on_run=None):
        self._workdir = workdir
        self.on_run = on_run
        self.calls = 0

    def workdir(self, config):
        return self._workdir

    def run(self, runner, config, prompt, **kwargs):
        self.calls += 1
        if self.on_run:
            self.on_run()
        return AgentResult(text="ok")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "a.py").write_text("a = 1\n")
    (root / "b.py").write_text("b = 1\n")
    (root / "dir").mkdir()
    (root / "dir" / "x.txt").write_text("x\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture
def harness(tmp_path, repo, monkeypatch):
    config = make_config(tmp_path, codex_dir=repo)
    runner = GitRunner()
    backend = StubBackend(repo)
    monkeypatch.setitem(registry.BACKENDS, "codex", backend)
    cv.establish_initial_baseline(config, runner, repo)

    class H:
        pass

    h = H()
    h.config, h.runner, h.backend, h.repo = config, runner, backend, repo

    def turn(role="coder"):
        return registry.run_agent_result(
            runner, agent="codex", config=config, prompt="p", role=role
        )

    h.turn = turn
    return h


def refused(h, *fragments):
    before = h.backend.calls
    with pytest.raises(CheckoutVerificationError) as info:
        h.turn()
    assert h.backend.calls == before
    for fragment in fragments:
        assert fragment in str(info.value)
    return str(info.value)


# ----- required tests -----------------------------------------------------


def test_wrong_branch_fails_closed_naming_it(harness):
    git(harness.repo, "switch", "-q", "-c", "other")
    message = refused(harness, "expected main", "observed other", "coder turn", "codex")
    assert str(harness.repo) in message
    assert git(harness.repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == b"other"


def test_moved_head_and_detached_head_fail_closed(harness):
    (harness.repo / "a.py").write_text("a = 2\n")
    git(harness.repo, "commit", "-q", "-am", "foreign")
    refused(harness, "expected main at", "observed main at")
    git(harness.repo, "checkout", "-q", "--detach")
    refused(harness, "observed HEAD")


def test_unexpected_modified_file_fails_closed_without_discarding_it(harness):
    (harness.repo / "a.py").write_text("a = 99\n")
    refused(harness, "a.py (added)")
    assert (harness.repo / "a.py").read_text() == "a = 99\n"


def test_unexpected_untracked_file_fails_closed_without_deleting_it(harness):
    (harness.repo / "stray.txt").write_text("hi")
    refused(harness, "stray.txt (added)")
    assert (harness.repo / "stray.txt").exists()


def test_tool_generated_gemini_md_proceeds(harness):
    body = single_shot_session_instruction("main")
    (harness.repo / GEMINI_MD_NAME).write_bytes(build_gemini_injection(body, orig="absent"))
    harness.turn()
    assert harness.backend.calls == 1
    assert not (harness.repo / GEMINI_MD_NAME).exists()


def test_clean_turn_proceeds_and_updates_baseline(harness):
    def write():
        (harness.repo / "made.txt").write_text("by the turn")

    harness.backend.on_run = write
    harness.turn()
    assert harness.backend.calls == 1
    harness.backend.on_run = None
    harness.turn()  # the turn's own file is the new baseline, not foreign dirt
    assert harness.backend.calls == 2


# ----- content identity ---------------------------------------------------


def dirty_baseline(h):
    (h.repo / "a.py").write_text("a = 2\n")
    (h.repo / "gen").mkdir()
    (h.repo / "gen" / "one.txt").write_text("1")
    (h.repo / b"odd-\xff.txt".decode("utf-8", "surrogateescape")).write_text("odd")
    cv.record_checkout_baseline(h.config, h.runner, h.repo, source="test")


def test_same_status_content_change_is_refused(harness):
    dirty_baseline(harness)
    harness.turn()
    (harness.repo / "a.py").write_text("a = 3\n")
    refused(harness, "a.py (content changed)")


def test_foreign_revert_of_baseline_entry_is_refused(harness):
    dirty_baseline(harness)
    git(harness.repo, "checkout", "--", "a.py")
    refused(harness, "a.py (removed)")


def test_new_file_in_untracked_directory_is_refused(harness):
    dirty_baseline(harness)
    (harness.repo / "gen" / "two.txt").write_text("2")
    refused(harness, "gen/two.txt (added)")


def test_non_utf8_filename_keeps_identity(harness):
    dirty_baseline(harness)
    harness.turn()
    name = os.fsdecode(b"odd-\xff.txt")
    (harness.repo / name).write_text("changed")
    message = refused(harness, "(content changed)")
    assert name in message


def test_symlinked_ancestor_is_not_traversed(harness, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.txt").write_text("secret")
    (harness.repo / "dir" / "x.txt").write_text("edited")
    cv.record_checkout_baseline(harness.config, harness.runner, harness.repo, source="test")
    (harness.repo / "dir" / "x.txt").unlink()
    (harness.repo / "dir").rmdir()
    (harness.repo / "dir").symlink_to(outside)
    refused(harness, "ancestor-not-directory")
    assert (outside / "x.txt").read_text() == "secret"


def test_fifo_is_never_opened(harness):
    (harness.repo / "b.py").unlink()
    os.mkfifo(harness.repo / "b.py")
    message = refused(harness, "unsupported fifo")
    assert "b.py" in message
    assert (harness.repo / "b.py").exists()


def test_probe_failure_fails_closed(harness, monkeypatch):
    def broken(self, args, **kwargs):
        raise AgentLoopError("boom")

    monkeypatch.setattr(GitRunner, "run_binary", broken)
    refused(harness, "Cannot verify", "boom")


def test_empty_status_is_clean(harness):
    harness.turn()
    assert harness.backend.calls == 1


def test_retry_after_turn_with_staged_rename_proceeds(harness):
    def edit():
        git(harness.repo, "mv", "b.py", "c.py")
        (harness.repo / "c.py").write_text("c = 1\n")
        (harness.repo / "new.txt").write_text("n")

    harness.backend.on_run = edit
    harness.turn()
    harness.backend.on_run = None
    harness.turn()
    assert harness.backend.calls == 2
    (harness.repo / "c.py").write_text("c = 2\n")
    refused_calls = harness.backend.calls
    with pytest.raises(CheckoutVerificationError):
        harness.turn()
    assert harness.backend.calls == refused_calls


# ----- poisoning, caps and budgets ---------------------------------------


def test_failed_post_turn_capture_poisons_the_checkout(harness, monkeypatch):
    real = cv.capture_fingerprint
    state = {"fail": False}

    def flaky(config, runner, path):
        if state["fail"]:
            raise cv._Incomplete("simulated probe failure")
        return real(config, runner, path)

    monkeypatch.setattr(cv, "capture_fingerprint", flaky)

    def switch():
        state["fail"] = True

    harness.backend.on_run = switch
    harness.turn()
    state["fail"] = False
    harness.backend.on_run = None
    refused(harness, "can no longer be trusted", "simulated probe failure")


def test_large_legitimate_tree_does_not_poison(harness, monkeypatch):
    monkeypatch.setattr(cv, "MAX_HASH_FILE_BYTES", 10)

    def build():
        (harness.repo / "tree").mkdir()
        for index in range(30):
            (harness.repo / "tree" / f"f{index}.txt").write_text("x")
        (harness.repo / "big.bin").write_bytes(b"y" * 100)

    harness.backend.on_run = build
    harness.turn()
    harness.backend.on_run = None
    harness.turn()
    assert harness.backend.calls == 2
    (harness.repo / "big.bin").write_bytes(b"z" * 100)
    with pytest.raises(CheckoutVerificationError, match="big.bin"):
        harness.turn()


def test_path_cap_message_is_actionable(harness, monkeypatch):
    monkeypatch.setattr(cv, "MAX_DIRTY_PATHS", 3)
    for index in range(5):
        (harness.repo / f"n{index}.txt").write_text("x")
    message = refused(harness, "3-path limit", "5 dirty paths")
    assert "n" in message


# ----- exemptions ---------------------------------------------------------


def test_isolated_roles_and_dry_run_skip_verification(harness):
    git(harness.repo, "switch", "-q", "-c", "other")
    registry.run_agent_result(
        harness.runner, agent="codex", config=harness.config, prompt="p", role="semantic-dedupe"
    )
    assert harness.backend.calls == 1
    harness.runner.dry_run = True
    harness.turn()
    assert harness.backend.calls == 2
    harness.runner.dry_run = False
    with pytest.raises(CheckoutVerificationError):
        registry.run_agent_result(
            harness.runner, agent="codex", config=harness.config, prompt="p", role="reviewer"
        )


def test_sandboxed_probes_use_the_hardened_bytes_runner(harness, monkeypatch):
    from coding_review_agent_loop import agent_permissions

    seen = []
    real = Runner.run_binary

    def fake_hardened(config):
        def run(args, workdir):
            seen.append(args)
            result = real(harness.runner, ("git", *args), cwd=workdir, check=False)
            return result

        return run

    monkeypatch.setattr(agent_permissions, "hardened_git_probe_runner_bytes", fake_hardened)
    monkeypatch.setattr(agent_permissions, "is_sandboxed", lambda config: True)
    monkeypatch.setattr(
        GitRunner, "run_binary", lambda self, *a, **k: pytest.fail("bare git ran in sandboxed mode")
    )
    name = os.fsdecode(b"odd-\xfe.txt")
    (harness.repo / name).write_text("one")
    cv.record_checkout_baseline(harness.config, harness.runner, harness.repo, source="test")
    harness.turn()
    (harness.repo / name).write_text("two")
    refused(harness, "(content changed)")
    assert any(args[0] == "status" for args in seen)


# ----- syncs and test gates ----------------------------------------------


@pytest.fixture
def synced(tmp_path):
    from coding_review_agent_loop import config as config_module

    origin = tmp_path / "OWNER" / "REPO"
    origin.mkdir(parents=True)
    git(origin, "init", "-q", "-b", "main")
    (origin / "a.py").write_text("a = 1\n")
    git(origin, "add", "-A")
    git(origin, "commit", "-q", "-m", "init")
    checkout = tmp_path / "work"
    git(tmp_path, "clone", "-q", str(origin), str(checkout))
    cfg = make_config(tmp_path, codex_dir=checkout, create_dirs=False)
    runner = Runner()
    return config_module, cfg, runner, checkout


def test_sync_refuses_before_destroying_foreign_files(synced):
    config_module, cfg, runner, checkout = synced
    config_module._sync_base_branch(
        checkout, label="Default codex workdir", default_owned=True, config=cfg, runner=runner
    )
    config_module._sync_base_branch(
        checkout, label="Default codex workdir", default_owned=True, config=cfg, runner=runner
    )
    (checkout / "a.py").write_text("foreign edit\n")
    (checkout / "foreign.txt").write_text("keep me")
    with pytest.raises(CheckoutVerificationError):
        config_module._sync_base_branch(
            checkout, label="Default codex workdir", default_owned=True, config=cfg, runner=runner
        )
    assert (checkout / "a.py").read_text() == "foreign edit\n"
    assert (checkout / "foreign.txt").exists()


def test_first_preparation_keeps_startup_cleaning_and_sets_baseline(synced):
    config_module, cfg, runner, checkout = synced
    (checkout / "dirt.txt").write_text("x")
    config_module._sync_base_branch(
        checkout, label="Default codex workdir", default_owned=True, config=cfg, runner=runner
    )
    assert not (checkout / "dirt.txt").exists()
    assert cv.has_entry(checkout)


def test_explicit_non_git_dir_is_refused_at_startup(tmp_path):
    from coding_review_agent_loop import config as config_module

    empty = tmp_path / "empty"
    empty.mkdir()
    cfg = make_config(tmp_path, codex_dir=empty, create_dirs=False)
    with pytest.raises(AgentLoopError, match="--codex-dir is not a git checkout"):
        config_module.validate_explicit_workdir(empty, "--codex-dir", cfg, Runner())
    assert not cv.has_entry(empty)


def test_disabled_gate_leaves_ledger_alone(harness):
    from coding_review_agent_loop import checks

    git(harness.repo, "switch", "-q", "-c", "other")
    checks.run_optional_tests(harness.runner, harness.config)  # no test command: no-op
    assert cv.has_entry(harness.repo)


def gate_config(h):
    from dataclasses import replace

    return replace(h.config, test_command=("true",), claude_dir=h.repo)


def test_gate_verifies_first_and_adopts_new_artifacts(harness, monkeypatch):
    from coding_review_agent_loop import checks

    config = gate_config(harness)
    monkeypatch.setattr(checks, "active_workdir", lambda cfg: harness.repo)
    monkeypatch.setattr(checks, "_record_gate_observation", lambda *a: None)
    monkeypatch.setattr(checks, "_raise_for_gate_result", lambda *a: None)
    ran = []

    def fake_gate(args, *, cwd, timeout_seconds, env=None):
        ran.append(1)
        (harness.repo / "artifact.log").write_text("new")

    harness.runner.run_test_command = fake_gate
    checks.run_optional_tests(harness.runner, config)
    harness.turn()  # the new artifact was adopted
    (harness.repo / "foreign.txt").write_text("x")
    with pytest.raises(CheckoutVerificationError):
        checks.run_optional_tests(harness.runner, config)
    assert ran == [1]


def test_gate_changing_a_preexisting_path_is_refused(harness, monkeypatch):
    from coding_review_agent_loop import checks

    (harness.repo / "a.py").write_text("a = 2\n")
    cv.record_checkout_baseline(harness.config, harness.runner, harness.repo, source="test")
    config = gate_config(harness)
    monkeypatch.setattr(checks, "active_workdir", lambda cfg: harness.repo)
    monkeypatch.setattr(checks, "_record_gate_observation", lambda *a: None)
    monkeypatch.setattr(checks, "_raise_for_gate_result", lambda *a: None)
    harness.runner.run_test_command = lambda args, **kw: (harness.repo / "a.py").write_text("a = 3\n")
    with pytest.raises(CheckoutVerificationError):
        checks.run_optional_tests(harness.runner, config)


# ----- GEMINI.md ----------------------------------------------------------


def inject(repo, body="# injected\n", orig="absent", rest=b""):
    (repo / GEMINI_MD_NAME).write_bytes(build_gemini_injection(body, orig=orig) + rest)


def test_remainder_is_preserved_and_judged_normally(harness):
    inject(harness.repo, rest=b"operator rules\n")
    refused(harness, "GEMINI.md (added)")
    assert (harness.repo / GEMINI_MD_NAME).read_bytes() == b"operator rules\n"


def test_unverifiable_header_is_not_removed(harness):
    bad = build_gemini_injection("# injected\n", orig="absent").replace(b"injected", b"tampered")
    (harness.repo / GEMINI_MD_NAME).write_bytes(bad)
    refused(harness, "GEMINI.md (added)")
    assert (harness.repo / GEMINI_MD_NAME).read_bytes() == bad


def test_tracked_leftover_from_other_base_and_oversized_prompt_is_restored(harness):
    (harness.repo / GEMINI_MD_NAME).write_text("rules\n")
    git(harness.repo, "add", GEMINI_MD_NAME)
    git(harness.repo, "commit", "-q", "-m", "gemini")
    cv.record_checkout_baseline(harness.config, harness.runner, harness.repo, source="test")
    body = single_shot_session_instruction("release") + "# Agent Loop Task\n\n" + "x" * 5000
    inject(harness.repo, body=body, orig="present", rest=b"rules\n")
    harness.turn()
    assert (harness.repo / GEMINI_MD_NAME).read_text() == "rules\n"


def test_originally_empty_tracked_file_stays_present_with_mode(harness):
    (harness.repo / GEMINI_MD_NAME).write_bytes(b"")
    os.chmod(harness.repo / GEMINI_MD_NAME, 0o600)
    git(harness.repo, "add", GEMINI_MD_NAME)
    git(harness.repo, "commit", "-q", "-m", "empty gemini")
    cv.record_checkout_baseline(harness.config, harness.runner, harness.repo, source="test")
    inject(harness.repo, orig="present")
    os.chmod(harness.repo / GEMINI_MD_NAME, 0o600)
    harness.turn()
    path = harness.repo / GEMINI_MD_NAME
    assert path.exists() and path.read_bytes() == b""
    assert (path.stat().st_mode & 0o777) == 0o600
    assert git(harness.repo, "status", "--porcelain").stdout == b""


def test_legacy_prefix_recovered_only_on_exact_match(harness):
    legacy = single_shot_session_instruction("main")
    (harness.repo / GEMINI_MD_NAME).write_text(legacy)
    harness.turn()
    assert not (harness.repo / GEMINI_MD_NAME).exists()
    (harness.repo / GEMINI_MD_NAME).write_text(legacy.replace("Single-Shot", "Edited"))
    refused(harness, "GEMINI.md (added)")


def test_recovery_refuses_while_the_gemini_lock_is_held(harness):
    inject(harness.repo)
    lock = _git_lock_path(harness.repo).open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        refused(harness, "concurrent holder")
        assert (harness.repo / GEMINI_MD_NAME).exists()
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    harness.turn()
    assert not (harness.repo / GEMINI_MD_NAME).exists()


def test_symlinked_gemini_md_is_never_followed(harness, tmp_path):
    outside = tmp_path / "outside.md"
    payload = build_gemini_injection("# injected\n", orig="absent")
    outside.write_bytes(payload)
    (harness.repo / GEMINI_MD_NAME).symlink_to(outside)
    refused(harness, "GEMINI.md (added)")
    assert outside.read_bytes() == payload
    assert (harness.repo / GEMINI_MD_NAME).is_symlink()


def test_clean_tracked_symlink_refuses_antigravity_before_injection(tmp_path, repo):
    outside = tmp_path / "outside.md"
    outside.write_text("external\n")
    (repo / GEMINI_MD_NAME).symlink_to(outside)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "link")
    runner = GitRunner(antigravity_outputs=[("never", 0)])
    config = make_config(tmp_path, antigravity_dir=repo, antigravity_cmd="agy")
    cv.establish_initial_baseline(config, runner, repo)
    with pytest.raises(CheckoutVerificationError, match="symlink"):
        registry.run_agent_result(
            runner, agent="antigravity", config=config, prompt="p", role="reviewer"
        )
    assert outside.read_text() == "external\n"
    assert not any("agy" in cmd[0][0] for cmd in runner.commands if cmd[0])


def test_swap_between_lstat_and_open_refuses_and_preserves_the_replacement(
    harness, monkeypatch
):
    from coding_review_agent_loop.agents import antigravity

    inject(harness.repo)
    real_open = os.open

    def swapping_open(path, flags, *args, **kwargs):
        if path == GEMINI_MD_NAME and flags & os.O_NONBLOCK:
            os.unlink(harness.repo / GEMINI_MD_NAME)
            os.mkfifo(harness.repo / GEMINI_MD_NAME)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(antigravity.os, "open", swapping_open)
    refused(harness, "swapped")
    assert (harness.repo / GEMINI_MD_NAME).exists()
    assert not (harness.repo / GEMINI_MD_NAME).is_file()


def test_safe_read_rejects_non_regular_entries(tmp_path):
    (tmp_path / GEMINI_MD_NAME).mkdir()
    with pytest.raises(SafeGeminiAccessError):
        safe_read_gemini_md(tmp_path)


# ----- explicit directory startup ----------------------------------------


def test_explicit_dir_startup_accepts_leftover_injection_but_not_other_dirt(tmp_path, repo):
    from coding_review_agent_loop import config as config_module

    git(repo, "remote", "add", "origin", "https://github.com/OWNER/REPO.git")
    cfg = make_config(tmp_path, antigravity_dir=repo, create_dirs=False)
    runner = Runner()
    # Written under a different base than this run uses.
    body = single_shot_session_instruction("release") + "# Agent Loop Task\n\n" + "y" * 4000
    (repo / GEMINI_MD_NAME).write_bytes(build_gemini_injection(body, orig="absent"))
    config_module.validate_explicit_workdir(repo, "--antigravity-dir", cfg, runner)
    assert not (repo / GEMINI_MD_NAME).exists()
    assert cv.has_entry(repo)

    cv.reset_checkout_baselines()
    (repo / GEMINI_MD_NAME).write_text("operator authored rules\n")
    with pytest.raises(AgentLoopError, match="is dirty"):
        config_module.validate_explicit_workdir(repo, "--antigravity-dir", cfg, runner)
    assert (repo / GEMINI_MD_NAME).read_text() == "operator authored rules\n"

    (repo / GEMINI_MD_NAME).unlink()
    (repo / "a.py").write_text("dirty\n")
    with pytest.raises(AgentLoopError, match="is dirty"):
        config_module.validate_explicit_workdir(repo, "--antigravity-dir", cfg, runner)
    assert (repo / "a.py").read_text() == "dirty\n"


def test_missing_explicit_dir_is_refused_not_silently_accepted(tmp_path):
    from coding_review_agent_loop import config as config_module

    missing = tmp_path / "nope"
    cfg = make_config(tmp_path, codex_dir=missing, create_dirs=False)
    config_module.ensure_workdir(missing, "--codex-dir")
    with pytest.raises(AgentLoopError, match="is not a git checkout"):
        config_module.validate_explicit_workdir(missing, "--codex-dir", cfg, Runner())


# ----- review round 1 regressions ------------------------------------------


def test_stale_recreation_never_launders_a_prepared_checkout(synced):
    config_module, cfg, runner, checkout = synced
    config_module.ensure_temp_checkout(checkout, agent="codex", config=cfg, runner=runner)
    for tracked in list(checkout.iterdir()):
        if tracked.name != ".git":
            tracked.unlink()  # a foreign writer emptied the tree: now "stale"
    with pytest.raises(CheckoutVerificationError):
        config_module.ensure_temp_checkout(checkout, agent="codex", config=cfg, runner=runner)
    assert (checkout / ".git").is_dir() and not (checkout / "a.py").exists()
    cv.poison_checkout(checkout, "capture after the turn failed: boom")
    with pytest.raises(CheckoutVerificationError, match="boom"):
        config_module.ensure_temp_checkout(checkout, agent="codex", config=cfg, runner=runner)
    assert (checkout / ".git").is_dir()


def test_legacy_oversized_injection_is_reported_not_stripped(harness):
    legacy = single_shot_session_instruction("main") + "# Agent Loop Task\n\nbig prompt\n\n---\n\n"
    target = harness.repo / GEMINI_MD_NAME
    target.write_text(legacy)
    refused(harness, "GEMINI.md (added)")
    assert target.read_text() == legacy
    git(harness.repo, "add", GEMINI_MD_NAME)
    git(harness.repo, "commit", "-q", "-m", "gemini")
    cv.record_checkout_baseline(harness.config, harness.runner, harness.repo, source="test")
    target.write_text(legacy + "committed rules\n")
    refused(harness, "GEMINI.md (added)")
    assert target.read_text() == legacy + "committed rules\n"


def test_filesystem_errors_during_capture_fail_closed_and_poison(harness, monkeypatch):
    (harness.repo / "link").symlink_to("a.py")
    cv.record_checkout_baseline(harness.config, harness.runner, harness.repo, source="test")

    def vanished(*args, **kwargs):
        raise FileNotFoundError("link vanished between lstat and readlink")

    monkeypatch.setattr(cv.os, "readlink", vanished)
    refused(harness, "could not be read while fingerprinting")
    # post-turn: the error must poison, not leak or leave the stale baseline
    harness.backend.on_run = None
    cv.record_checkout_baseline(harness.config, harness.runner, harness.repo, source="test")
    monkeypatch.undo()
    refused(harness, "can no longer be trusted", "link vanished")


def test_hashing_io_error_is_a_verification_error(harness, monkeypatch):
    (harness.repo / "a.py").write_text("a = 2\n")
    real_fdopen = cv.os.fdopen

    def failing(*args, **kwargs):
        raise OSError(5, "input/output error")

    monkeypatch.setattr(cv.os, "fdopen", failing)
    refused(harness, "could not be read while fingerprinting")
    monkeypatch.setattr(cv.os, "fdopen", real_fdopen)


def test_fake_runner_binary_and_text_probes_agree(tmp_path):
    runner = FakeRunner(git_status=" M src/a.py\n?? new.txt\n", git_head="h-1")
    text = runner.run(["git", "status", "--porcelain"], cwd=tmp_path).stdout
    binary = runner.run_binary(
        ["git", "status", "--porcelain", "-z", "--untracked-files=all", "--no-renames"], cwd=tmp_path
    ).stdout
    assert binary.split(b"\0")[:-1] == [line.encode() for line in text.splitlines()]
    assert (tmp_path / "src" / "a.py").is_file() and (tmp_path / "new.txt").is_file()
    head = runner.run(["git", "rev-parse", "HEAD"], cwd=tmp_path).stdout.strip()
    assert runner.run_binary(["git", "rev-parse", "HEAD"], cwd=tmp_path).stdout.strip() == head.encode()


def test_fake_runner_keeps_per_checkout_state(tmp_path):
    runner = FakeRunner(git_head="h-1", codex_outputs=["ok"])
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    assert runner.run_binary(["git", "rev-parse", "HEAD"], cwd=two).stdout == b"h-1\n"
    runner.run_with_log(
        ["codex", "exec", "-"], cwd=one, log_path=tmp_path / "log", label="x",
        progress_interval_seconds=30, check=False,
    )
    runner.git_head = "h-2"  # a coder commit in ONE must not move TWO
    for form in ("binary", "text"):
        probe = (
            (lambda cwd: runner.run_binary(["git", "rev-parse", "HEAD"], cwd=cwd).stdout.decode())
            if form == "binary"
            else (lambda cwd: runner.run(["git", "rev-parse", "HEAD"], cwd=cwd).stdout)
        )
        assert probe(one) == "h-2\n", form
        assert probe(two) == "h-1\n", form


def test_fake_runner_probes_agree_after_agent_turn_and_detached_checkout(tmp_path):
    runner = FakeRunner(
        git_head="h-1", git_status="", post_agent_git_status=" M src/a.py\n",
        codex_outputs=["ok"],
    )
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()

    def both(cwd):
        text = runner.run(["git", "status", "--porcelain"], cwd=cwd).stdout
        binary = runner.run_binary(["git", "status", "--porcelain", "-z"], cwd=cwd).stdout
        return text, binary

    assert both(one) == ("", b"") and both(two) == ("", b"")
    runner.run_with_log(
        ["codex", "exec", "-"], cwd=one, log_path=tmp_path / "log", label="x",
        progress_interval_seconds=30, check=False,
    )
    text, binary = both(one)
    assert text == " M src/a.py\n" and binary == b" M src/a.py\0"
    assert both(two) == ("", b"")  # the turn happened in ONE only
    assert runner.run_binary(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=one).stdout == b"main\n"
    runner.run(["git", "checkout", "--detach", "refs/remotes/origin/pr/7"], cwd=one)
    assert runner.run_binary(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=one).stdout == b"HEAD\n"
    assert runner.run_binary(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=two).stdout == b"main\n"


def test_fake_runner_scripted_binary_probe_failures_and_staged_state(tmp_path):
    runner = FakeRunner()
    runner.binary_probe_exceptions.append(AgentLoopError("scripted probe failure"))
    with pytest.raises(AgentLoopError, match="scripted probe failure"):
        runner.run_binary(["git", "rev-parse", "HEAD"], cwd=tmp_path)
    runner.binary_probe_results.append({"stdout": b"", "returncode": 128})
    assert runner.run_binary(["git", "rev-parse", "HEAD"], cwd=tmp_path).returncode == 128
    runner.checkout_staged[tmp_path] = b":100644 100644 a b M\0a.py\0"
    assert runner.run_binary(["git", "diff", "--cached", "--raw", "-z"], cwd=tmp_path).stdout.startswith(b":100644")


def test_fake_runner_foreign_deletion_is_exposed_not_recreated(tmp_path):
    runner = FakeRunner(git_status=" M src/a.py\n")
    probe = ["git", "status", "--porcelain", "-z"]
    runner.run_binary(probe, cwd=tmp_path)
    assert (tmp_path / "src" / "a.py").is_file()
    (tmp_path / "src" / "a.py").unlink()  # a foreign writer deletes it
    runner.run_binary(probe, cwd=tmp_path)
    assert not (tmp_path / "src" / "a.py").exists()
    config = make_config(tmp_path)
    with pytest.raises(cv._Incomplete, match="missing from the worktree"):
        cv.capture_fingerprint(config, runner, tmp_path)


@pytest.fixture
def pr_synced(synced):
    config_module, cfg, runner, checkout = synced
    origin = checkout.parent / "OWNER" / "REPO"
    git(origin, "switch", "-q", "-c", "feature")
    (origin / "feature.txt").write_text("f\n")
    git(origin, "add", "-A")
    git(origin, "commit", "-q", "-m", "feature")
    sha = git(origin, "rev-parse", "HEAD").stdout.decode().strip()
    git(origin, "update-ref", "refs/pull/7/head", sha)
    git(origin, "switch", "-q", "main")
    from coding_review_agent_loop.github import PullRequestMetadata

    meta = PullRequestMetadata(7, "OWNER/REPO", "t", "feature", "main", sha, None)
    return config_module, cfg, runner, checkout, meta


def sync_pr(pr_synced):
    config_module, cfg, runner, checkout, meta = pr_synced
    config_module.sync_checkout_to_pr(
        cfg, runner, path=checkout, label="Default codex workdir", default_owned=True,
        pr_number=7, pr_metadata=meta,
    )


def test_pr_sync_verifies_first_then_refreshes_the_baseline(pr_synced):
    config_module, cfg, runner, checkout, meta = pr_synced
    config_module._sync_base_branch(
        checkout, label="Default codex workdir", default_owned=True, config=cfg, runner=runner
    )
    (checkout / "foreign.txt").write_text("keep me")
    with pytest.raises(CheckoutVerificationError, match="foreign.txt"):
        sync_pr(pr_synced)
    assert (checkout / "foreign.txt").exists()
    assert git(checkout, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == b"main"
    (checkout / "foreign.txt").unlink()
    sync_pr(pr_synced)
    assert (checkout / "feature.txt").exists()
    # the refreshed baseline matches the detached PR head: verification passes
    cv.verify_checkout(cfg, runner, path=checkout, agent="codex", purpose="reviewer turn")


def test_poisoned_checkout_refuses_clean_wrong_branch_and_syncs(pr_synced, monkeypatch):
    config_module, cfg, runner, checkout, meta = pr_synced
    config_module._sync_base_branch(
        checkout, label="Default codex workdir", default_owned=True, config=cfg, runner=runner
    )
    real = cv.capture_fingerprint
    state = {"fail": False}

    def flaky(config, run, path):
        if state["fail"]:
            raise cv._Incomplete("capture failed after the turn")
        return real(config, run, path)

    monkeypatch.setattr(cv, "capture_fingerprint", flaky)
    state["fail"] = True
    cv.record_checkout_baseline(cfg, runner, checkout, source="codex reviewer turn")
    state["fail"] = False
    git(checkout, "switch", "-q", "-c", "clean-wrong")  # clean, but the wrong branch
    with pytest.raises(CheckoutVerificationError, match="capture failed after the turn"):
        cv.verify_checkout(cfg, runner, path=checkout, agent="codex", purpose="reviewer turn")
    with pytest.raises(CheckoutVerificationError, match="capture failed after the turn"):
        sync_pr(pr_synced)
    with pytest.raises(CheckoutVerificationError, match="capture failed after the turn"):
        config_module._sync_base_branch(
            checkout, label="Default codex workdir", default_owned=True, config=cfg, runner=runner
        )
    assert git(checkout, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == b"clean-wrong"


def test_path_cap_exceeded_after_a_turn_poisons_with_the_cap_message(harness, monkeypatch):
    monkeypatch.setattr(cv, "MAX_DIRTY_PATHS", 3)

    def flood():
        for index in range(5):
            (harness.repo / f"gen{index}.txt").write_text("x")

    harness.backend.on_run = flood
    harness.turn()
    harness.backend.on_run = None
    monkeypatch.setattr(cv, "MAX_DIRTY_PATHS", 100_000)
    for index in range(5):
        (harness.repo / f"gen{index}.txt").unlink()
    refused(harness, "can no longer be trusted", "3-path limit", "5 dirty paths")


def test_symlinked_ancestor_target_is_never_opened(harness, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.txt").write_text("secret")
    (harness.repo / "dir" / "x.txt").write_text("edited")
    cv.record_checkout_baseline(harness.config, harness.runner, harness.repo, source="test")
    (harness.repo / "dir" / "x.txt").unlink()
    (harness.repo / "dir").rmdir()
    (harness.repo / "dir").symlink_to(outside)
    opened = []
    real_open = os.open

    def spy(path, flags, *args, **kwargs):
        opened.append((path, kwargs.get("dir_fd")))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(cv.os, "open", spy)
    refused(harness, "ancestor-not-directory")
    assert all(os.fsdecode(path) != "x.txt" for path, _ in opened)
    assert all(not str(path).startswith(str(outside)) for path, _ in opened)


def test_every_attempt_is_verified_including_after_a_failed_attempt(harness):
    def nonzero():
        (harness.repo / "partial.txt").write_text("own edit")
        raise AgentLoopError("agent exited nonzero")

    harness.backend.on_run = nonzero
    with pytest.raises(AgentLoopError):
        harness.turn()
    harness.backend.on_run = None
    harness.turn()  # retry: the failed attempt's own edit is the baseline
    (harness.repo / "foreign.txt").write_text("x")
    refused(harness, "foreign.txt (added)")


def test_pre_review_test_gate_verifies_and_adopts(harness, monkeypatch):
    from coding_review_agent_loop import checks

    config = gate_config(harness)
    monkeypatch.setattr(checks, "active_workdir", lambda cfg: harness.repo)
    monkeypatch.setattr(checks, "_record_gate_observation", lambda *a: None)
    monkeypatch.setattr(checks, "_raise_for_gate_result", lambda *a: None)
    harness.runner.run_test_command = lambda args, **kw: (harness.repo / "report.xml").write_text("r")
    checks.run_pre_review_tests(harness.runner, config)
    harness.turn()
    (harness.repo / "foreign.txt").write_text("x")
    ran = []
    harness.runner.run_test_command = lambda args, **kw: ran.append(1)
    with pytest.raises(CheckoutVerificationError):
        checks.run_pre_review_tests(harness.runner, config)
    assert ran == []


def test_antigravity_normal_cleanup_keeps_an_originally_empty_tracked_file(tmp_path, repo):
    (repo / GEMINI_MD_NAME).write_bytes(b"")
    os.chmod(repo / GEMINI_MD_NAME, 0o600)
    git(repo, "add", GEMINI_MD_NAME)
    git(repo, "commit", "-q", "-m", "empty gemini")
    runner = GitRunner(antigravity_outputs=[("done", 0)])
    config = make_config(tmp_path, antigravity_dir=repo, antigravity_cmd="agy")
    cv.establish_initial_baseline(config, runner, repo)
    registry.run_agent_result(runner, agent="antigravity", config=config, prompt="p", role="reviewer")
    assert (repo / GEMINI_MD_NAME).exists() and (repo / GEMINI_MD_NAME).read_bytes() == b""
    assert ((repo / GEMINI_MD_NAME).stat().st_mode & 0o777) == 0o600
    assert git(repo, "status", "--porcelain").stdout == b""


# ----- round 2: real attempt loop, missing-directory recreation, GEMINI.md I/O --


import shutil  # noqa: E402

from coding_review_agent_loop import orchestrator  # noqa: E402
from coding_review_agent_loop.runner import CommandResult  # noqa: E402


class _LoopStub(StubBackend):
    """Backend whose scripted results drive the orchestrator attempt loop."""

    def __init__(self, workdir, script, name):
        super().__init__(workdir)
        self.name = name
        self.script = list(script)

    def run(self, runner, config, prompt, **kwargs):
        self.calls += 1
        step = self.script.pop(0)
        if step.get("edit"):
            step["edit"]()
        return AgentResult(
            text=step.get("text", ""),
            returncode=step.get("returncode", 0),
            self_update_reason=step.get("self_update_reason"),
            command_result=(
                CommandResult([self.name], self._workdir, "", "", step.get("returncode", 0))
                if step.get("self_update_reason") else None
            ),
        )


def _loop(tmp_path, repo, monkeypatch, *, agent, script, **config_kwargs):
    runner = GitRunner()
    runner.wait_for_executable_stability = lambda *a, **k: True
    config_kwargs.setdefault("agent_max_retries", 2)
    config = make_config(
        tmp_path, codex_dir=repo, antigravity_dir=repo, coder=agent, **config_kwargs,
    )
    backend = _LoopStub(repo, script, agent)
    monkeypatch.setitem(registry.BACKENDS, agent, backend)
    cv.establish_initial_baseline(config, runner, repo)

    def run_it():
        return orchestrator._run_validated_agent(
            runner, agent=agent, config=config, prompt="p",
            marker_description="VALID",
            validate=lambda text: text if text == "VALID" else (_ for _ in ()).throw(AgentLoopError("invalid")),
            role="coder",
        )

    return runner, config, backend, run_it


TRANSIENT = "Error: 503 Service Unavailable, please try again later"


def _own_edit(repo):
    def edit():
        git(repo, "mv", "b.py", "c.py")
        (repo / "c.py").write_text("c = 1\n")
        (repo / "new.txt").write_text("own")

    return edit


def _foreign_between_attempts(runner, repo, monkeypatch, hook_owner, hook_name):
    original = getattr(hook_owner, hook_name)
    state = {"done": False}

    def hooked(*args, **kwargs):
        if not state["done"]:
            state["done"] = True
            (repo / "foreign.txt").write_text("another writer")
        return original(*args, **kwargs)

    monkeypatch.setattr(hook_owner, hook_name, hooked)


def test_orchestrator_retry_keeps_own_staged_rename_edits(tmp_path, repo, monkeypatch):
    runner, config, backend, run_it = _loop(
        tmp_path, repo, monkeypatch, agent="codex",
        script=[{"edit": _own_edit(repo), "text": TRANSIENT, "returncode": 1}, {"text": "VALID"}],
    )
    assert run_it().text == "VALID"
    assert backend.calls == 2  # the retry proceeded over its own staged rename


def test_orchestrator_retry_refuses_foreign_change_before_the_next_spawn(tmp_path, repo, monkeypatch):
    runner, config, backend, run_it = _loop(
        tmp_path, repo, monkeypatch, agent="codex",
        script=[{"edit": _own_edit(repo), "text": TRANSIENT, "returncode": 1}, {"text": "VALID"}],
    )
    _foreign_between_attempts(runner, repo, monkeypatch, runner, "run")
    with pytest.raises(CheckoutVerificationError, match="foreign.txt"):
        run_it()
    assert backend.calls == 1 and runner.comments == []
    assert (repo / "foreign.txt").exists() and (repo / "c.py").exists()


def test_orchestrator_replacement_replay_refuses_foreign_change(tmp_path, repo, monkeypatch):
    runner, config, backend, run_it = _loop(
        tmp_path, repo, monkeypatch, agent="codex",
        script=[
            {"edit": _own_edit(repo), "text": "", "returncode": 1, "self_update_reason": "exe replaced"},
            {"text": "VALID"},
        ],
    )
    original = runner.wait_for_executable_stability

    def foreign_then_stable(*args, **kwargs):
        (repo / "foreign.txt").write_text("another writer")
        return original(*args, **kwargs)

    runner.wait_for_executable_stability = foreign_then_stable
    with pytest.raises(CheckoutVerificationError, match="foreign.txt"):
        run_it()
    assert backend.calls == 1  # the replay never spawned


def test_orchestrator_replacement_replay_proceeds_over_own_edits(tmp_path, repo, monkeypatch):
    runner, config, backend, run_it = _loop(
        tmp_path, repo, monkeypatch, agent="codex",
        script=[
            {"edit": _own_edit(repo), "text": "", "returncode": 1, "self_update_reason": "exe replaced"},
            {"text": "VALID"},
        ],
    )
    assert run_it().text == "VALID" and backend.calls == 2


def test_orchestrator_antigravity_fallback_refuses_foreign_change(tmp_path, repo, monkeypatch):
    capacity = {"text": "quota exceeded please try again", "returncode": 1}
    runner, config, backend, run_it = _loop(
        tmp_path, repo, monkeypatch, agent="antigravity",
        script=[dict(capacity, edit=_own_edit(repo)), {"text": "VALID"}],
        antigravity_models=("model-one", "model-two"), agent_max_retries=0,
    )
    _foreign_between_attempts(
        runner, repo, monkeypatch, orchestrator, "classify_antigravity_capacity"
    )
    with pytest.raises(CheckoutVerificationError, match="foreign.txt"):
        run_it()
    assert backend.calls == 1


def test_orchestrator_antigravity_fallback_proceeds_over_own_edits(tmp_path, repo, monkeypatch):
    capacity = {"text": "quota exceeded please try again", "returncode": 1}
    runner, config, backend, run_it = _loop(
        tmp_path, repo, monkeypatch, agent="antigravity",
        script=[dict(capacity, edit=_own_edit(repo)), {"text": "VALID"}],
        antigravity_models=("model-one", "model-two"), agent_max_retries=0,
    )
    assert run_it().text == "VALID" and backend.calls == 2


@pytest.fixture
def prepared_default(synced):
    config_module, cfg, runner, checkout = synced
    config_module.ensure_temp_checkout(checkout, agent="codex", config=cfg, runner=runner)
    return config_module, cfg, checkout


class _CloneSpy(Runner):
    def __init__(self):
        super().__init__()
        self.clones = []

    def run(self, args, **kwargs):
        if tuple(args[:3]) == ("gh", "repo", "clone"):
            self.clones.append(tuple(args))
            raise AssertionError("must not clone over a prepared checkout")
        return super().run(args, **kwargs)


@pytest.mark.parametrize("poisoned", [False, True])
def test_vanished_prepared_checkout_is_never_recreated(prepared_default, monkeypatch, poisoned):
    from coding_review_agent_loop import agent_permissions

    config_module, cfg, checkout = prepared_default
    if poisoned:
        cv.poison_checkout(checkout, "capture after the turn failed: boom")
    entry_before = cv._LEDGER[checkout.resolve()]
    shutil.rmtree(checkout)
    registered = []
    monkeypatch.setattr(agent_permissions, "register_checkout", lambda *a, **k: registered.append(a))
    spy = _CloneSpy()
    with pytest.raises(CheckoutVerificationError):
        config_module.ensure_temp_checkout(checkout, agent="codex", config=cfg, runner=spy)
    assert spy.clones == [] and registered == [] and not checkout.exists()
    assert cv._LEDGER[checkout.resolve()] is entry_before


def test_recovery_read_io_error_is_a_terminal_refusal(harness, monkeypatch):
    from coding_review_agent_loop.agents import antigravity

    (harness.repo / GEMINI_MD_NAME).write_bytes(build_gemini_injection("# i\n", orig="absent"))
    before = (harness.repo / GEMINI_MD_NAME).read_bytes()

    def eio(*args, **kwargs):
        raise OSError(5, "input/output error")

    monkeypatch.setattr(antigravity.os, "fdopen", eio)
    refused(harness, "cannot read", "input/output error")
    monkeypatch.undo()
    assert (harness.repo / GEMINI_MD_NAME).read_bytes() == before


def test_recovery_lock_or_root_io_errors_are_terminal_refusals(harness, monkeypatch):
    from coding_review_agent_loop.agents import antigravity

    (harness.repo / GEMINI_MD_NAME).write_bytes(build_gemini_injection("# i\n", orig="absent"))

    def broken_lock(path):
        raise OSError(13, "permission denied")

    monkeypatch.setattr(antigravity, "_git_lock_path", broken_lock)
    refused(harness, "cannot inspect", "permission denied")


def test_cleanup_io_failure_does_not_mask_the_result_and_stays_fail_closed(tmp_path, repo, monkeypatch):
    from coding_review_agent_loop.agents import antigravity

    runner = GitRunner(antigravity_outputs=[("done", 0), ("done", 0)])
    config = make_config(tmp_path, antigravity_dir=repo, antigravity_cmd="agy")
    cv.establish_initial_baseline(config, runner, repo)

    def failing_strip(root):
        raise OSError(5, "input/output error")

    monkeypatch.setattr(antigravity, "strip_gemini_injection", failing_strip)
    result = registry.run_agent_result(
        runner, agent="antigravity", config=config, prompt="p", role="reviewer"
    )
    assert result.text  # the completed backend result survives the cleanup failure
    assert (repo / GEMINI_MD_NAME).exists()  # the leftover injection is still there
    monkeypatch.undo()
    with pytest.raises(CheckoutVerificationError):  # and the next turn refuses, fail-closed
        registry.run_agent_result(runner, agent="antigravity", config=config, prompt="p", role="reviewer")
