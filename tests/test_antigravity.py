from agent_loop_helpers import *  # noqa: F403
from coding_review_agent_loop.config import default_run_worktree_root


def test_antigravity_backend_command_and_prefers_response_file(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(
        antigravity_outputs=[("stdout fallback text", 0)],
        public_response_outputs=["response file text"],
    )
    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_cmd="agy",
        antigravity_model="Gemini 3.1 Pro (High)",
        antigravity_args=("--dangerously-skip-permissions",),
    )
    result = AntigravityBackend().run(runner, config, "Review this PR.", run_id="run-1")
    cmd = runner.commands[-1][0]
    assert cmd[0] == "agy"
    assert cmd[cmd.index("--model") + 1] == "Gemini 3.1 Pro (High)"
    assert cmd[cmd.index("--print-timeout") + 1] == "600s"
    assert "--dangerously-skip-permissions" in cmd
    # The prompt is the value of --print and must be the last argument (agy's
    # --print/--prompt consumes the next token, not a trailing positional).
    assert cmd[-2] == "--print"
    assert "Review this PR." in cmd[-1]
    assert result.text == "response file text"
    assert result.response_file_path is not None
    assert result.response_file_path.read_text(encoding="utf-8").strip() == "response file text"
    assert result.session_id is None  # agy --print exposes no conversation id

def test_antigravity_backend_stdout_fallback(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(antigravity_outputs=[("plain stdout review", 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    result = AntigravityBackend().run(runner, config, "Review this PR.", run_id="run-1")
    assert result.text == "plain stdout review"
    assert result.text_source == "stdout"


def test_antigravity_backend_uses_configured_print_timeout(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(antigravity_outputs=[("ok", 0)])
    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_print_timeout_seconds=900,
    )

    AntigravityBackend().run(runner, config, "Review this PR.", run_id="run-1")

    cmd = runner.commands[-1][0]
    assert cmd[cmd.index("--print-timeout") + 1] == "900s"


def test_antigravity_backend_places_large_prompt_in_injected_gemini_md(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            captured.append((cwd / "GEMINI.md").read_text(encoding="utf-8"))
            return super().run_with_log(args, cwd=cwd, **kwargs)

    prompt = "x" * 150_000
    runner = CapturingRunner(antigravity_outputs=[("ok", 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)

    AntigravityBackend().run(runner, config, prompt, run_id="r1")

    cmd = runner.commands[-1][0]
    assert cmd[-2] == "--print"
    assert cmd[-1] == "Follow the complete task included in the Agent Loop Task section of GEMINI.md."
    assert all(prompt not in arg for arg in cmd)
    assert captured and prompt in captured[0]
    assert not (agy_dir / "GEMINI.md").exists()

def test_antigravity_attempt_state_retries_then_falls_back(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityAttemptState
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_models=("ModelA", "ModelB", "ModelC"),
        antigravity_quota_signatures=("quota",)
    )
    state = AntigravityAttemptState.from_config(config, retries=2)
    models = []
    for expected in ("retry", "retry", "fallback", "fallback", "stop"):
        models.append(state.singleton_config(config).antigravity_models)
        assert state.next_after_failure(retryable=True, provider_capacity=True) == expected
    assert models == [("ModelA",), ("ModelA",), ("ModelA",), ("ModelB",), ("ModelC",)]


def test_legacy_cli_pr_fallback_counts_one_exact_head_approval(tmp_path):
    from coding_review_agent_loop.cli import build_parser, config_from_args, run_pr_loop

    claude_dir = tmp_path / "claude"
    agy_dir = tmp_path / "antigravity"
    claude_dir.mkdir()
    agy_dir.mkdir()
    args = build_parser().parse_args([
        "pr", "77", "--repo", "OWNER/REPO", "--reviewer", "agy",
        "--antigravity-models", "ModelA", "ModelB",
        "--claude-dir", str(claude_dir), "--antigravity-dir", str(agy_dir),
        "--agent-max-retries", "0", "--no-pre-review-tests",
    ])
    runner = FakeRunner(antigravity_outputs=[
        ("quota exceeded", 1),
        (structured_pr_review(state="approved", summary="Approved.", reviewer="Google Antigravity"), 0),
    ])
    config = config_from_args(args, runner)
    assert config.reviewer == ("antigravity",)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    models = [cmd[cmd.index("--model") + 1] for cmd, _ in runner.commands
              if cmd and cmd[0] == "agy" and "--model" in cmd]
    assert models == ["ModelA", "ModelB"]
    reviews = [comment for comment in runner.comments if "**Review verdict:**" in comment]
    assert len(reviews) == 1
    assert "Approved" in reviews[0]
    assert "abc123" == runner.pr_payload["headRefOid"]


def test_two_named_antigravity_seats_approve_one_pr_head(tmp_path):
    from coding_review_agent_loop.cli import build_parser, config_from_args, run_pr_loop
    from coding_review_agent_loop.agents.registry import agent_signature

    claude_dir = tmp_path / "claude"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    for directory in (claude_dir, first_dir, second_dir):
        directory.mkdir()
    args = build_parser().parse_args([
        "pr", "77", "--repo", "OWNER/REPO", "--review-parallel",
        "--reviewer-seat", "flash=agy", "--seat-model", "flash=Gemini 3.8 Flash (High)",
        "--seat-dir", f"flash={first_dir}",
        "--reviewer-seat", "opus=agy", "--seat-model", "opus=Claude Opus 5.5 (Medium)",
        "--seat-dir", f"opus={second_dir}",
        "--claude-dir", str(claude_dir), "--agent-max-retries", "0",
        "--no-pre-review-tests",
    ])
    runner = FakeRunner(antigravity_outputs=[
        (structured_pr_review(reviewer="flash (Google Antigravity: Gemini 3.8 Flash (High))"), 0),
        (structured_pr_review(reviewer="opus (Google Antigravity: Claude Opus 5.5 (Medium))"), 0),
    ])
    config = config_from_args(args, runner)
    assert tuple(agent_signature(seat) for seat in config.reviewer) == (
        "flash (Google Antigravity: Gemini 3.8 Flash (High))",
        "opus (Google Antigravity: Claude Opus 5.5 (Medium))",
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    models = [cmd[cmd.index("--model") + 1] for cmd, _ in runner.commands
              if cmd and cmd[0] == "agy" and "--model" in cmd]
    assert sorted(models) == sorted(["Gemini 3.8 Flash (High)", "Claude Opus 5.5 (Medium)"])
    reviews = [comment for comment in runner.comments if "**Review verdict:**" in comment]
    assert len(reviews) == 2
    assert any("flash" in comment for comment in reviews)
    assert any("opus" in comment for comment in reviews)
    import json
    first_usage = json.loads(next(config.log_dir.glob("*-usage-summary.json")).read_text())
    assert sorted((call["agent"], call["backend"]) for call in first_usage["calls"]) == [
        ("flash", "antigravity"), ("opus", "antigravity")
    ]
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len([cmd for cmd, _ in runner.commands if cmd and cmd[0] == "agy" and "--model" in cmd]) == 2
    from dataclasses import replace
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent
    new_flash = SeatAgent(
        ReviewerSeat("flash", "antigravity", ("Gemini 3.7 Flash (High)",)), first_dir
    )
    changed_config = replace(
        config,
        reviewer=(new_flash, config.reviewer[1]),
        reviewer_seats=(new_flash, config.reviewer[1]),
    )
    runner.antigravity_outputs.extend([
        (structured_pr_review(reviewer="flash (Google Antigravity: Gemini 3.7 Flash (High))"), 0),
        (structured_pr_review(reviewer="opus (Google Antigravity: Claude Opus 5.5 (Medium))"), 0),
    ])
    assert run_pr_loop(runner, pr_number=77, config=changed_config) == 0
    new_models = [cmd[cmd.index("--model") + 1] for cmd, _ in runner.commands
                  if cmd and cmd[0] == "agy" and "--model" in cmd]
    assert len(new_models) == 4
    assert new_models.count("Gemini 3.7 Flash (High)") == 1
    assert run_pr_loop(runner, pr_number=77, config=changed_config) == 0
    assert len([cmd for cmd, _ in runner.commands if cmd and cmd[0] == "agy" and "--model" in cmd]) == 4

def test_antigravity_backend_stops_on_other_errors(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(
        antigravity_outputs=[
            ("some regular error", 1),
            ("ok fallback answered", 0)
        ]
    )
    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_models=("ModelA", "ModelB"),
        antigravity_quota_signatures=("quota",)
    )
    result = AntigravityBackend().run(runner, config, "Review", run_id="r1")
    
    assert runner.commands[-1][0][runner.commands[-1][0].index("--model") + 1] == "ModelA"
    assert result.returncode == 1
    assert result.model_used == "ModelA"

def test_antigravity_backend_ignores_stale_partial_response_file(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    from coding_review_agent_loop.agents.base import public_response_path
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(antigravity_outputs=[("success", 0)], public_response_outputs=["successful response"])
    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_models=("ModelA",),
        antigravity_quota_signatures=("quota",)
    )
    stale_path = public_response_path(config, "antigravity")
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("stale partial response", encoding="utf-8")
    result = AntigravityBackend().run(runner, config, "Review", run_id="r1")
    
    assert result.text == "successful response"
    assert result.model_used == "ModelA"
    assert result.text != "stale partial response"

def test_antigravity_backend_writes_gemini_md_single_shot_instruction(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            gemini_md = cwd / "GEMINI.md"
            captured.append(gemini_md.read_text(encoding="utf-8") if gemini_md.exists() else "")
            return super().run_with_log(args, cwd=cwd, **kwargs)

    runner = CapturingRunner(antigravity_outputs=[("ok", 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(runner, config, "Review this PR.", run_id="r1")

    assert captured, "run_with_log was not called"
    assert "Do NOT spawn background execution tasks" in captured[0]
    assert "DO NOT run tests, builds, compilation" in captured[0]
    assert "strict allow-listed read-only commands" in captured[0]
    assert "DO NOT run tests, shell commands, or compile code" not in captured[0]
    # prefix is stripped → file deleted (no remaining content after it)
    assert not (agy_dir / "GEMINI.md").exists()
    # Lock file must not appear in the worktree root (it lives in .git/ only)
    assert not (agy_dir / "GEMINI.md.lock").exists()

def test_antigravity_backend_preserves_existing_gemini_md_during_and_after_run(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    original_content = "# My project rules\nUse tabs.\n"
    (agy_dir / "GEMINI.md").write_text(original_content, encoding="utf-8")
    captured: list[str] = []

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            gemini_md = cwd / "GEMINI.md"
            captured.append(gemini_md.read_text(encoding="utf-8") if gemini_md.exists() else "")
            return super().run_with_log(args, cwd=cwd, **kwargs)

    runner = CapturingRunner(antigravity_outputs=[("ok", 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(runner, config, "Review this PR.", run_id="r1")

    assert captured, "run_with_log was not called"
    # Instruction was prepended before the original content during the run
    assert "Do NOT spawn background execution tasks" in captured[0]
    assert "My project rules" in captured[0]
    assert captured[0].index("Do NOT spawn") < captured[0].index("My project rules")
    # Prefix stripped after the run → original content remains
    after = (agy_dir / "GEMINI.md").read_text(encoding="utf-8")
    assert after == original_content
    assert "Do NOT spawn" not in after

def test_antigravity_backend_preserves_agent_edits_to_gemini_md(tmp_path):
    """Agent (coder role) edits the content after our prefix — preserved after run."""
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    original_content = "# Original rules\n"
    (agy_dir / "GEMINI.md").write_text(original_content, encoding="utf-8")

    class EditingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            # Simulate agent appending to the file after the injected prefix
            gemini_md = cwd / "GEMINI.md"
            current = gemini_md.read_text(encoding="utf-8")
            gemini_md.write_text(current + "# New agent-added rules\n", encoding="utf-8")
            return super().run_with_log(args, cwd=cwd, **kwargs)

    runner = EditingRunner(antigravity_outputs=[("ok", 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(runner, config, "Review this PR.", run_id="r1")

    after = (agy_dir / "GEMINI.md").read_text(encoding="utf-8")
    # Original content and agent's new content both present; our prefix stripped
    assert "Original rules" in after
    assert "New agent-added rules" in after
    assert "Do NOT spawn" not in after

def test_antigravity_backend_cleans_up_gemini_md_on_exception(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)

    class RaisingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            raise RuntimeError("subprocess failed")

    # Sub-test A: no pre-existing GEMINI.md — prefix stripped → file deleted
    runner = RaisingRunner()
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    with pytest.raises(RuntimeError):
        AntigravityBackend().run(runner, config, "Review this PR.", run_id="r1")
    assert not (agy_dir / "GEMINI.md").exists()

    # Sub-test B: pre-existing GEMINI.md — prefix stripped, original content restored
    original_content = "# Existing rules\n"
    (agy_dir / "GEMINI.md").write_text(original_content, encoding="utf-8")
    runner2 = RaisingRunner()
    with pytest.raises(RuntimeError):
        AntigravityBackend().run(runner2, config, "Review this PR.", run_id="r2")
    after = (agy_dir / "GEMINI.md").read_text(encoding="utf-8")
    assert after == original_content
    assert "Do NOT spawn" not in after


# Bound for every main-thread rendezvous wait and join in the flock
# serialization tests (#1473, following #1471). Event.wait and Thread.join
# return as soon as the peer arrives, so a green run pays nothing; the bound
# only caps how long a real serialization regression takes to fail.
_LOCK_PEER_WAIT_TIMEOUT_SECONDS = 60.0
# Bound for the lock holder's own self-release wait (#1473). It must exceed the
# main-thread peer waits that run before the release (at most two sequential
# waits), so a slow contender fails as a named rendezvous timeout instead of a
# holder that released the lock by itself and let the LOCK_NB probe succeed.
_LOCK_HOLDER_RELEASE_TIMEOUT_SECONDS = 4 * _LOCK_PEER_WAIT_TIMEOUT_SECONDS

_LOCK_SERIALIZATION_TEST_NAMES = (
    "test_antigravity_backend_gemini_md_lock_serializes_concurrent_access",
    "test_antigravity_settings_lock_serializes_reviewer_vs_reviewer",
    "test_antigravity_settings_lock_serializes_reviewer_vs_coder_both_orders",
    "test_antigravity_settings_lock_restoration_precedes_unlock_on_exception",
)


def _release_and_join(events, threads):
    """Release every holder event, then join every started worker with a bound.

    Called from a ``finally`` block so a failing assertion cannot leave a thread
    holding a flock or still restoring settings while monkeypatch teardown runs.
    It never raises, so the original test failure is reported unmasked.
    """
    for event in events:
        try:
            event.set()
        except Exception:
            pass
    for thread in threads:
        try:
            thread.join(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS)
        except Exception:
            pass


def test_antigravity_backend_gemini_md_lock_serializes_concurrent_access(tmp_path):
    """flock on GEMINI.md.lock prevents a second run from starting until the first
    completes its inject→run→strip sequence."""
    import fcntl
    import threading
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(tmp_path, antigravity_dir=agy_dir)

    order: list[str] = []
    # Lock lives in .git/ to avoid polluting the worktree
    lock_path = agy_dir / ".git" / "GEMINI.md.lock"
    (agy_dir / ".git").mkdir(parents=True, exist_ok=True)

    # Thread A: pre-acquires the exclusive lock, records "A-holds", sleeps briefly,
    # records "A-releases", then releases the lock. This simulates another process
    # holding the lock while running agy.
    lock_acquired = threading.Event()
    lock_released = threading.Event()

    def hold_lock():
        lf = lock_path.open("a+")
        fcntl.flock(lf, fcntl.LOCK_EX)
        order.append("A-holds")
        lock_acquired.set()
        lock_released.wait(timeout=_LOCK_HOLDER_RELEASE_TIMEOUT_SECONDS)
        order.append("A-releases")
        fcntl.flock(lf, fcntl.LOCK_UN)
        lf.close()

    # Thread B (main): tries to run AntigravityBackend — should block on LOCK_EX
    # until Thread A releases.
    run_started = threading.Event()

    class RecordingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            order.append("B-run")
            return super().run_with_log(args, cwd=cwd, **kwargs)

    def run_backend():
        AntigravityBackend().run(
            RecordingRunner(antigravity_outputs=[("ok", 0)]),
            config,
            "Review.",
            run_id="r1",
        )
        order.append("B-done")

    started: list[threading.Thread] = []
    try:
        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        started.append(t)
        assert lock_acquired.wait(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS), (
            "holder never acquired GEMINI.md lock"
        )

        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        started.append(backend_thread)

        # Give the backend thread a moment to block on the lock, then release it.
        import time
        time.sleep(0.05)
        lock_released.set()
        t.join(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS)
        backend_thread.join(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS)

        assert not t.is_alive(), "Lock holder deadlocked"
        assert not backend_thread.is_alive(), "Backend thread deadlocked"
        # A-holds must precede B-run and A-releases must precede B-run
        assert "A-holds" in order
        assert "A-releases" in order
        assert "B-run" in order
        assert order.index("A-holds") < order.index("B-run")
        assert order.index("A-releases") < order.index("B-run")
    finally:
        _release_and_join([lock_released], started)

def test_antigravity_module_imports_without_fcntl():
    """Antigravity module must import cleanly even when fcntl is unavailable (Windows)."""
    import sys

    import coding_review_agent_loop.agents as agents_pkg

    # Keep the original module objects: other modules hold references to
    # their functions, and a later test that monkeypatches the module (for
    # example its ``_now`` clock) must patch the object production code uses.
    saved = {k: v for k, v in sys.modules.items() if "antigravity" in k}
    saved_attr = agents_pkg.__dict__.get("antigravity")
    for m in saved:
        del sys.modules[m]

    # Simulate a platform without fcntl by hiding it
    original = sys.modules.pop("fcntl", None)
    sys.modules["fcntl"] = None  # type: ignore[assignment]
    try:
        import coding_review_agent_loop.agents.antigravity as mod
        assert hasattr(mod, "AntigravityBackend")
    finally:
        if original is not None:
            sys.modules["fcntl"] = original
        else:
            sys.modules.pop("fcntl", None)
        # Restore the original modules so later tests see the same objects.
        for k in [k for k in sys.modules if "antigravity" in k]:
            del sys.modules[k]
        sys.modules.update(saved)
        if saved_attr is not None:
            agents_pkg.antigravity = saved_attr

def test_antigravity_backend_git_lock_path_follows_linked_worktree(tmp_path):
    """_git_lock_path resolves a file-form .git marker (linked worktree) to the real
    git dir instead of trying to mkdir the .git file."""
    from coding_review_agent_loop.agents.antigravity import _git_lock_path

    agy_dir = tmp_path / "worktree"
    agy_dir.mkdir()
    real_git_dir = tmp_path / "repo.git" / "worktrees" / "wt"
    real_git_dir.mkdir(parents=True)

    # Simulate the .git file that git worktree add creates
    (agy_dir / ".git").write_text(
        f"gitdir: {real_git_dir}\n", encoding="utf-8"
    )

    lock = _git_lock_path(agy_dir)
    assert lock.parent == real_git_dir
    assert lock.name == "GEMINI.md.lock"
    # Must not attempt to mkdir over the .git file
    assert (agy_dir / ".git").is_file()

def test_antigravity_backend_strips_public_response_marker(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    from coding_review_agent_loop.protocol import PUBLIC_RESPONSE_MARKER
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    stdout = (
        "I will inspect the diff and run the tests.\n"
        f"{PUBLIC_RESPONSE_MARKER}\n"
        "STATE: approved\n\nLooks good to me."
    )
    runner = FakeRunner(antigravity_outputs=[(stdout, 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    result = AntigravityBackend().run(runner, config, "Review this PR.", run_id="run-1")
    assert result.text == "STATE: approved\n\nLooks good to me."
    assert result.text_source == "stdout_marker"
    assert "I will inspect" not in result.text

def test_antigravity_backend_resume_uses_conversation(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(antigravity_outputs=[("ok", 0)])
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(runner, config, "x", session_id="conv-7", run_id="r")
    cmd = runner.commands[-1][0]
    assert cmd[cmd.index("--conversation") + 1] == "conv-7"

def test_antigravity_registry():
    from coding_review_agent_loop.agents.registry import (
        agent_display_name,
        agent_signature,
        get_backend,
    )
    assert agent_display_name("antigravity") == "Antigravity"
    assert agent_signature("antigravity") == "Google Antigravity"
    assert get_backend("antigravity").name == "antigravity"

def test_config_from_args_antigravity_defaults(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr", "123", "--repo", "OWNER/REPO",
        "--coder", "antigravity", "--reviewer", "codex",
        "--codex-dir", str(tmp_path / "codex"),
        "--dangerous-agent-permissions",
    ])
    config = config_from_args(args, FakeRunner())
    assert config.coder == "antigravity"
    assert config.antigravity_cmd == "agy"
    assert config.antigravity_model is None
    assert config.antigravity_models == (
        "Gemini 3.8 Flash (High)",
        "Gemini 3.7 Flash (High)",
        "Gemini 3.6 Flash (High)",
        "Gemini 3.1 Pro (High)",
        "Claude Opus 5.5 (Medium)",
    )
    assert config.antigravity_print_timeout_seconds == 600
    assert config.antigravity_quota_signatures == (
        "quota", "rate limit", "too many requests", "resource exhausted",
        "RESOURCE_EXHAUSTED", "429", "high traffic", "try again in a minute",
        "overload", "no capacity", "temporarily at capacity",
    )
    assert config.antigravity_args == ("--dangerously-skip-permissions",)
    assert config.antigravity_dir.parent == default_run_worktree_root("OWNER/REPO", "antigravity").resolve()
    assert dict(config.default_checkout_stores)["antigravity"] == default_agent_workdir("OWNER/REPO", "antigravity").resolve()
    # The per-run worktree is removed at run end, so the log dir is durable and outside it.
    assert config.antigravity_dir not in config.log_dir.parents
    assert config.run_token in config.log_dir.parts

def test_antigravity_quota_signatures_default_single_source(tmp_path):
    """The quota-signatures default comes from one constant — no drift across the
    dataclass field, the CLI flag, and config_from_args (#348, #350)."""
    from coding_review_agent_loop.config import DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES as DEFAULT
    # dataclass field default
    assert make_config(tmp_path).antigravity_quota_signatures == DEFAULT
    # CLI flag default and config_from_args both derive from the constant
    parser = build_parser()
    args = parser.parse_args(["pr", "123", "--repo", "OWNER/REPO", "--codex-dir", str(tmp_path / "codex")])
    assert tuple(args.antigravity_quota_signatures) == DEFAULT
    assert config_from_args(args, FakeRunner()).antigravity_quota_signatures == DEFAULT

def test_antigravity_models_default_chain_from_constant(tmp_path):
    """The default model fallback chain resolves from the named constant when
    neither a legacy model nor an explicit chain is given."""
    from coding_review_agent_loop.config import DEFAULT_ANTIGRAVITY_MODELS
    assert make_config(tmp_path).antigravity_models == DEFAULT_ANTIGRAVITY_MODELS
    parser = build_parser()
    args = parser.parse_args(["pr", "123", "--repo", "OWNER/REPO", "--codex-dir", str(tmp_path / "codex")])
    assert config_from_args(args, FakeRunner()).antigravity_models == DEFAULT_ANTIGRAVITY_MODELS


def test_antigravity_print_timeout_cli_override_and_validation(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "pr", "123", "--repo", "OWNER/REPO", "--codex-dir", str(tmp_path / "codex"),
        "--antigravity-print-timeout-seconds", "900",
    ])
    assert config_from_args(args, FakeRunner()).antigravity_print_timeout_seconds == 900

    with pytest.raises(AgentLoopError, match="--antigravity-print-timeout-seconds"):
        make_config(tmp_path, antigravity_print_timeout_seconds=0)

def test_cli_rejects_both_antigravity_model_flags():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "pr", "123", "--repo", "OWNER/REPO", "--antigravity-model", "Gemini", "--antigravity-models", "Gemini", "Claude"
        ])

def test_config_rejects_blank_antigravity_models(tmp_path):
    with pytest.raises(AgentLoopError, match="cannot be empty or contain blank entries"):
        make_config(tmp_path, antigravity_models=("",))

def test_config_rejects_both_model_flags(tmp_path):
    with pytest.raises(AgentLoopError, match="Cannot specify both antigravity_model and a custom antigravity_models chain"):
        make_config(tmp_path, antigravity_model="Gemini", antigravity_models=("Gemini", "Claude"))

def test_config_rejects_both_model_flags_even_if_default(tmp_path):
    with pytest.raises(AgentLoopError, match="Cannot specify both antigravity_model and a custom antigravity_models chain"):
        make_config(tmp_path, antigravity_model="Gemini", antigravity_models=("Gemini 3.1 Pro (High)", "Gemini 3.5 Flash (High)"))

def test_distinct_workdir_validation_covers_antigravity(tmp_path):
    from coding_review_agent_loop.config import ensure_distinct_workdirs
    shared = tmp_path / "shared"
    config = make_config(
        tmp_path,
        coder="antigravity",
        reviewer="codex",
        allow_shared_dir=False,
        antigravity_dir=shared,
        codex_dir=shared,
    )
    with pytest.raises(AgentLoopError, match="same directory"):
        ensure_distinct_workdirs(config)

def test_antigravity_prompt_includes_terminal_response_instruction():
    from coding_review_agent_loop.agents.antigravity import _with_public_response_marker_instruction
    composed = _with_public_response_marker_instruction("BASE PROMPT")
    assert "end your turn immediately" in composed
    assert "do not defer to a background task result" in composed

def test_antigravity_prompt_excludes_old_wait_instruction():
    from coding_review_agent_loop.agents.antigravity import _with_public_response_marker_instruction
    composed = _with_public_response_marker_instruction("BASE PROMPT")
    assert "Do not print the marker until you are done with all internal reasoning" not in composed

def test_antigravity_backend_injects_strict_mode_for_reviewer(tmp_path, monkeypatch):
    """Reviewer run injects toolPermission:strict and expected allow-list while running."""
    import json as _json
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import (
        AntigravityBackend,
        _REVIEWER_SETTINGS_INJECTION,
    )

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    original = _json.dumps({"existingKey": "existingValue"})
    settings_file.write_text(original, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    captured_settings: list[dict] = []

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            captured_settings.append(
                _json.loads(settings_file.read_text(encoding="utf-8"))
            )
            return super().run_with_log(args, cwd=cwd, **kwargs)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(
        CapturingRunner(antigravity_outputs=[("ok", 0)]),
        config, "Review PR.", role="reviewer",
    )

    assert len(captured_settings) == 1
    settings = captured_settings[0]
    assert settings["toolPermission"] == "strict"
    assert settings["permissions"] == _REVIEWER_SETTINGS_INJECTION["permissions"]
    assert settings["existingKey"] == "existingValue"
    allowed = settings["permissions"]["allow"]
    assert "command(git)" not in allowed
    assert {
        "command(git diff)",
        "command(git show)",
        "command(git status)",
        "command(git log)",
        "command(rg)",
        "command(sed)",
        "command(cat)",
        "command(head)",
    } <= set(allowed)
    assert not any(
        command in allowed
        for command in (
            "command(pytest)",
            "command(npm)",
            "command(go)",
            "command(make)",
            "command(git checkout)",
            "command(git reset)",
            "command(git clean)",
            "command(git commit)",
        )
    )

def test_antigravity_backend_restores_original_settings_after_reviewer_run(tmp_path, monkeypatch):
    """Settings file is verbatim-restored after a successful reviewer run."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    original_text = '{"key": "val", "nested": {"a": 1}}'
    settings_file.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(
        FakeRunner(antigravity_outputs=[("ok", 0)]),
        config, "Review.", role="reviewer",
    )

    assert settings_file.read_text(encoding="utf-8") == original_text

def test_antigravity_backend_restores_settings_on_run_exception(tmp_path, monkeypatch):
    """Settings file is verbatim-restored even when the runner raises."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    original_text = '{"keep": "me"}'
    settings_file.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    class RaisingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            raise RuntimeError("agy crashed")

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    with pytest.raises(RuntimeError, match="agy crashed"):
        AntigravityBackend().run(RaisingRunner(), config, "Review.", role="reviewer")

    assert settings_file.read_text(encoding="utf-8") == original_text

def test_antigravity_backend_restores_settings_on_injection_write_failure(tmp_path, monkeypatch):
    """If the injection write_text partially truncates then raises, original bytes are restored."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    original_text = '{"preserve": "exactly"}'
    settings_file.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    write_calls = [0]
    real_write_text = Path.write_text

    def patched_write_text(self, *args, **kwargs):
        if self == settings_file:
            write_calls[0] += 1
            if write_calls[0] == 1:
                real_write_text(self, "PARTIAL")
                raise OSError("simulated disk full")
        real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", patched_write_text)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    with pytest.raises(OSError, match="simulated disk full"):
        AntigravityBackend().run(
            FakeRunner(antigravity_outputs=[("ok", 0)]), config, "Review.", role="reviewer",
        )

    assert settings_file.read_text(encoding="utf-8") == original_text

def test_antigravity_backend_does_not_touch_settings_for_non_reviewer(tmp_path, monkeypatch):
    """Non-reviewer run holds the settings lock but does not read or modify settings.json."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    original_text = '{"untouched": true}'
    settings_file.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    captured_during: list[str] = []

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            captured_during.append(settings_file.read_text(encoding="utf-8"))
            return super().run_with_log(args, cwd=cwd, **kwargs)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(
        CapturingRunner(antigravity_outputs=[("ok", 0)]),
        config, "Implement.", role=None,
    )

    assert captured_during[0] == original_text
    assert settings_file.read_text(encoding="utf-8") == original_text

def test_antigravity_backend_fails_fast_on_malformed_settings_json(tmp_path, monkeypatch):
    """AgentLoopError raised before any run if settings.json is not valid JSON."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    from coding_review_agent_loop.errors import AgentLoopError

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("not json at all {{{", encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    with pytest.raises(AgentLoopError, match="settings.json malformed"):
        AntigravityBackend().run(
            FakeRunner(antigravity_outputs=[("ok", 0)]), config, "Review.", role="reviewer",
        )

@pytest.mark.parametrize("invalid", ["[]", "null", "42"])
def test_antigravity_backend_fails_fast_on_non_object_settings_json(
    tmp_path, monkeypatch, invalid
):
    """AgentLoopError raised if settings.json root is not a JSON object."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    from coding_review_agent_loop.errors import AgentLoopError

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(invalid, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    with pytest.raises(AgentLoopError, match="not a JSON object"):
        AntigravityBackend().run(
            FakeRunner(antigravity_outputs=[("ok", 0)]), config, "Review.", role="reviewer",
        )

def test_antigravity_backend_strips_dangerously_skip_permissions_for_reviewer(
    tmp_path, monkeypatch
):
    """--dangerously-skip-permissions is removed from args when role=reviewer."""
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    captured_args: list[list[str]] = []

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            captured_args.append(list(args))
            return super().run_with_log(args, cwd=cwd, **kwargs)

    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_args=("--dangerously-skip-permissions",),
    )
    AntigravityBackend().run(
        CapturingRunner(antigravity_outputs=[("ok", 0)]),
        config, "Review.", role="reviewer",
    )
    assert "--dangerously-skip-permissions" not in captured_args[-1]

    # Non-reviewer run keeps the flag
    captured_args.clear()
    AntigravityBackend().run(
        CapturingRunner(antigravity_outputs=[("ok", 0)]),
        config, "Implement.", role=None,
    )
    assert "--dangerously-skip-permissions" in captured_args[-1]


@pytest.mark.parametrize("role", [None, "reviewer"])
def test_antigravity_after_snapshot_runs_after_gemini_cleanup_under_lock(
    tmp_path, monkeypatch, role
):
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    from coding_review_agent_loop.runner import CommandResult, ExecutableIdentity, ExecutionObservation

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)
    if role == "reviewer":
        settings_file.write_text('{"existing": true}', encoding="utf-8")

    identity = ExecutableIdentity(
        "agy", "agy-target", (1, 1, 90_000_000_000), (1, 2, 90_000_000_000)
    )
    changed = ExecutableIdentity(
        "agy", "agy-new-target", (1, 3, 100_000_000_000), (1, 4, 100_000_000_000)
    )
    observation = ExecutionObservation(100, 100, 101, 1, identity, changed, False)
    snapshots_seen: list[tuple[bool, str]] = []

    class ObservedRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            raw = super().run_with_log(args, cwd=cwd, **kwargs)
            return CommandResult(
                raw.args, raw.cwd, raw.stdout, raw.stderr, raw.returncode, observation,
                raw.capture_diagnostics,
            )

        def run(self, args, *, cwd, **kwargs):
            if tuple(args) == ("git", "status", "--porcelain"):
                path = Path(cwd) / "GEMINI.md"
                snapshots_seen.append((path.exists(), path.read_text(encoding="utf-8") if path.exists() else ""))
            return super().run(args, cwd=cwd, **kwargs)

    output = "agy v0.8.0\nModel: Gemini 3.1 Pro (High)\nError: Cannot find module launcher.js\n"
    runner = ObservedRunner(
        antigravity_outputs=[(output, 1)],
        git_probe_results=[
            {"stdout": "abc123\n", "returncode": 0},
            {"stdout": "", "returncode": 0},
            {"stdout": "abc123\n", "returncode": 0},
            {"stdout": "", "returncode": 0},
        ],
    )
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    result = AntigravityBackend().run(runner, config, "Review.", role=role)

    assert result.self_update_reason == "Antigravity executable changed during invocation"
    assert result.self_update_replay_refusal_kind is None
    assert len(snapshots_seen) == 2
    assert snapshots_seen[1][0] is False
    assert "Agent Loop Single-Shot Session" not in snapshots_seen[1][1]


def test_antigravity_repair_skips_checkout_probes_and_replacement_metadata(tmp_path, monkeypatch):
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)
    config = make_config(tmp_path, antigravity_dir=tmp_path / "agy")
    runner = FakeRunner(antigravity_outputs=[("MODULE_NOT_FOUND", 1)])

    result = AntigravityBackend().run_repair(
        runner, config, "Repair this response.", model="ModelA"
    )

    assert runner.git_probe_calls == []
    assert result.self_update_reason is None
    assert result.self_update_replay_refusal_kind is None
    assert result.self_update_replay_refusal_detail is None

def test_antigravity_backend_lock_order_settings_outer_gemini_inner(tmp_path, monkeypatch):
    """Settings lock (outer) is acquired before GEMINI.md lock (inner);
    settings are restored before the settings lock is released."""
    import fcntl as fcntl_mod
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    (agy_dir / ".git").mkdir(parents=True, exist_ok=True)
    settings_file = tmp_path / "settings.json"
    original_text = '{"v": 1}'
    settings_file.write_text(original_text, encoding="utf-8")
    settings_lock_path = str(settings_file.with_suffix(".json.lock"))
    gemini_lock_path = str(agy_dir / ".git" / "GEMINI.md.lock")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    real_flock = fcntl_mod.flock
    operations: list[tuple[str, int]] = []
    settings_content_at_unlock: list[str | None] = [None]

    def tracking_flock(fd, operation):
        name = str(getattr(fd, "name", ""))
        if settings_lock_path in name:
            label = "settings"
        elif gemini_lock_path in name:
            label = "gemini"
        else:
            label = "other"
        if label == "settings" and operation == fcntl_mod.LOCK_UN:
            settings_content_at_unlock[0] = settings_file.read_text(encoding="utf-8")
        operations.append((label, operation))
        real_flock(fd, operation)

    monkeypatch.setattr(fcntl_mod, "flock", tracking_flock)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    AntigravityBackend().run(
        FakeRunner(antigravity_outputs=[("ok", 0)]),
        config, "Review.", role="reviewer",
    )

    relevant = [(label, op) for label, op in operations if label in ("settings", "gemini")]
    assert relevant == [
        ("settings", fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB),
        ("gemini", fcntl_mod.LOCK_EX),
        ("gemini", fcntl_mod.LOCK_UN),
        ("settings", fcntl_mod.LOCK_UN),
    ]
    assert settings_content_at_unlock[0] == original_text

def test_antigravity_settings_lock_serializes_reviewer_vs_reviewer(tmp_path, monkeypatch):
    """Two concurrent reviewer runs are serialized: runner B starts only after A completes."""
    import threading
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"k": "v"}', encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    config = make_config(tmp_path, antigravity_dir=agy_dir)
    order: list[str] = []
    a_in_runner = threading.Event()
    a_release = threading.Event()
    b_in_runner = threading.Event()

    class RunnerA(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            order.append("a_in_runner")
            a_in_runner.set()
            a_release.wait(timeout=_LOCK_HOLDER_RELEASE_TIMEOUT_SECONDS)
            return super().run_with_log(args, cwd=cwd, **kwargs)

    class RunnerB(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            order.append("b_in_runner")
            b_in_runner.set()
            return super().run_with_log(args, cwd=cwd, **kwargs)

    def run_a():
        AntigravityBackend().run(RunnerA(antigravity_outputs=[("ok", 0)]), config, "PromptA", role="reviewer")

    def run_b():
        AntigravityBackend().run(RunnerB(antigravity_outputs=[("ok", 0)]), config, "PromptB", role="reviewer")

    started: list[threading.Thread] = []
    try:
        ta = threading.Thread(target=run_a, daemon=True)
        ta.start()
        started.append(ta)
        assert a_in_runner.wait(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS), (
            "Thread A never entered runner"
        )

        tb = threading.Thread(target=run_b, daemon=True)
        tb.start()
        started.append(tb)

        a_release.set()
        assert b_in_runner.wait(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS), (
            "Thread B never entered runner"
        )

        ta.join(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS)
        tb.join(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS)

        assert not ta.is_alive(), "Thread A deadlocked"
        assert not tb.is_alive(), "Thread B deadlocked"
        assert order == ["a_in_runner", "b_in_runner"]
        assert settings_file.read_text(encoding="utf-8") == '{"k": "v"}'
    finally:
        _release_and_join([a_release], started)

def test_antigravity_settings_lock_serializes_reviewer_vs_coder_both_orders(
    tmp_path, monkeypatch
):
    """Reviewer-coder and coder-reviewer serialization; LOCK_NB on the contender backend's
    settings-lock acquisition confirms the holder is actively holding the lock."""
    import fcntl as fcntl_mod
    import threading
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    settings_lock_path_str = str(settings_file.with_suffix(".json.lock"))
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    real_flock = fcntl_mod.flock
    _is_contender: threading.local = threading.local()
    _probe_done: threading.local = threading.local()

    for holder_role, contender_role in [("reviewer", None), (None, "reviewer")]:
        settings_file.write_text('{"initial": true}', encoding="utf-8")

        holder_in_runner = threading.Event()
        release_holder = threading.Event()
        contender_attempting_lock = threading.Event()
        contender_probe_done = threading.Event()
        contender_in_runner = threading.Event()
        nb_probe_results: list = []

        def instrumented_flock(fd, operation,
                               _slp=settings_lock_path_str,
                               _cat=contender_attempting_lock,
                               _cpd=contender_probe_done,
                               _nbr=nb_probe_results):
            if (
                getattr(_is_contender, "value", False)
                and _slp in str(getattr(fd, "name", ""))
                and operation == fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB
                and not getattr(_probe_done, "done", False)
            ):
                _probe_done.done = True
                _cat.set()
                try:
                    real_flock(fd, fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB)
                    _nbr.append(None)
                except BlockingIOError as exc:
                    _nbr.append(exc)
                    _cpd.set()
                    raise
                _cpd.set()
                return
            real_flock(fd, operation)

        monkeypatch.setattr(fcntl_mod, "flock", instrumented_flock)

        config = make_config(tmp_path, antigravity_dir=agy_dir)

        class HolderRunner(FakeRunner):
            def run_with_log(self, args, *, cwd, **kwargs):
                holder_in_runner.set()
                release_holder.wait(timeout=_LOCK_HOLDER_RELEASE_TIMEOUT_SECONDS)
                return super().run_with_log(args, cwd=cwd, **kwargs)

        class ContenderRunner(FakeRunner):
            def run_with_log(self, args, *, cwd, **kwargs):
                contender_in_runner.set()
                return super().run_with_log(args, cwd=cwd, **kwargs)

        def run_holder(role=holder_role):
            AntigravityBackend().run(
                HolderRunner(antigravity_outputs=[("ok", 0)]),
                config, "Holder", role=role,
            )

        def run_contender(role=contender_role):
            _is_contender.value = True
            _probe_done.done = False
            AntigravityBackend().run(
                ContenderRunner(antigravity_outputs=[("ok", 0)]),
                config, "Contender", role=role,
            )

        roles = f"holder_role={holder_role!r}, contender_role={contender_role!r}"
        started: list[threading.Thread] = []
        try:
            th = threading.Thread(target=run_holder, daemon=True)
            th.start()
            started.append(th)
            assert holder_in_runner.wait(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS), (
                f"holder never entered runner ({roles})"
            )

            tc = threading.Thread(target=run_contender, daemon=True)
            tc.start()
            started.append(tc)

            assert contender_attempting_lock.wait(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS), (
                f"contender never attempted the settings lock ({roles})"
            )
            assert contender_probe_done.wait(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS), (
                f"contender LOCK_NB probe never completed ({roles})"
            )
            release_holder.set()
            assert contender_in_runner.wait(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS), (
                f"contender never entered runner ({roles})"
            )

            th.join(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS)
            tc.join(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS)

            assert not th.is_alive(), f"Holder deadlocked (holder_role={holder_role!r})"
            assert not tc.is_alive(), f"Contender deadlocked (contender_role={contender_role!r})"
            assert contender_in_runner.is_set()
            assert len(nb_probe_results) == 1
            assert isinstance(nb_probe_results[0], BlockingIOError), (
                f"LOCK_NB probe should have failed with BlockingIOError "
                f"but got {nb_probe_results[0]!r} "
                f"(holder_role={holder_role!r}, contender_role={contender_role!r})"
            )
        finally:
            _release_and_join([release_holder], started)

def test_antigravity_settings_lock_restoration_precedes_unlock_on_exception(
    tmp_path, monkeypatch
):
    """Settings are restored before the settings lock is released, even when runner raises.

    Thread A: reviewer run; runner signals `injected_event` (settings injected, lock held),
    waits for `contention_confirmed_event`, then raises RuntimeError.
    Thread B: raw-lock contender; LOCK_NB proves A holds the lock, then blocks on LOCK_EX.
    After A's exception path restores settings and releases the lock, B unblocks and
    reads the settings file — which must already be restored to the original content.
    """
    import fcntl as fcntl_mod
    import threading
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir(parents=True)
    settings_file = tmp_path / "settings.json"
    settings_lock_path = settings_file.with_suffix(".json.lock")
    original_text = '{"preserve": "this"}'
    settings_file.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)

    injected_event = threading.Event()
    contention_confirmed_event = threading.Event()
    thread_a_exc: list[BaseException | None] = [None]
    thread_b_result: list[str | None] = [None]
    thread_b_nb_error: list = [None]

    class InjectingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            injected_event.set()
            contention_confirmed_event.wait(timeout=_LOCK_HOLDER_RELEASE_TIMEOUT_SECONDS)
            raise RuntimeError("simulated reviewer failure")

    def run_a():
        try:
            AntigravityBackend().run(
                InjectingRunner(),
                make_config(tmp_path, antigravity_dir=agy_dir),
                "Review.", role="reviewer",
            )
        except RuntimeError as exc:
            thread_a_exc[0] = exc

    def run_b():
        lock_f = settings_lock_path.open("a+")
        try:
            try:
                fcntl_mod.flock(lock_f, fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB)
                thread_b_nb_error[0] = None
            except BlockingIOError as exc:
                thread_b_nb_error[0] = exc
            contention_confirmed_event.set()
            fcntl_mod.flock(lock_f, fcntl_mod.LOCK_EX)
            thread_b_result[0] = settings_file.read_text(encoding="utf-8")
        finally:
            fcntl_mod.flock(lock_f, fcntl_mod.LOCK_UN)
            lock_f.close()

    started: list[threading.Thread] = []
    try:
        ta = threading.Thread(target=run_a, daemon=True)
        ta.start()
        started.append(ta)
        assert injected_event.wait(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS), (
            "Thread A never injected settings"
        )

        tb = threading.Thread(target=run_b, daemon=True)
        tb.start()
        started.append(tb)

        ta.join(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS)
        tb.join(timeout=_LOCK_PEER_WAIT_TIMEOUT_SECONDS)

        assert not ta.is_alive(), "Thread A deadlocked"
        assert not tb.is_alive(), "Thread B deadlocked"
        assert isinstance(thread_a_exc[0], RuntimeError)
        assert isinstance(thread_b_nb_error[0], BlockingIOError), (
            f"LOCK_NB should have raised BlockingIOError; got {thread_b_nb_error[0]!r}"
        )
        assert thread_b_result[0] == original_text, (
            f"Settings must be restored before the lock is released; got {thread_b_result[0]!r}"
        )
    finally:
        _release_and_join([contention_confirmed_event], started)


def test_antigravity_lock_serialization_tests_use_named_wait_bounds():
    """Static guard (#1473): the flock serialization tests keep named, bounded
    waits and a release-then-join cleanup in ``finally``."""
    import ast
    import inspect
    import textwrap

    assert _LOCK_HOLDER_RELEASE_TIMEOUT_SECONDS > 2 * _LOCK_PEER_WAIT_TIMEOUT_SECONDS

    for name in _LOCK_SERIALIZATION_TEST_NAMES:
        tree = ast.parse(textwrap.dedent(inspect.getsource(globals()[name])))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("wait", "join")
        ]
        assert calls, name
        for call in calls:
            timeout_args = list(call.args) + [
                kw.value for kw in call.keywords if kw.arg == "timeout"
            ]
            assert timeout_args, f"{name}: unbounded .{call.func.attr}() at line {call.lineno}"
            for arg in timeout_args:
                assert not isinstance(arg, ast.Constant), (
                    f"{name}: literal timeout in .{call.func.attr}() at line {call.lineno}"
                )
        finally_calls_cleanup = any(
            isinstance(node, ast.Try)
            and any(
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "_release_and_join"
                for stmt in node.finalbody
                for sub in ast.walk(stmt)
            )
            for node in ast.walk(tree)
        )
        assert finally_calls_cleanup, f"{name}: missing finally with _release_and_join"

def test_gemini_retirement_signal():
    from coding_review_agent_loop.agents.gemini import _gemini_retirement_signal
    assert _gemini_retirement_signal("Error: quota exceeded")
    assert _gemini_retirement_signal("PERMISSION_DENIED for this account")
    assert _gemini_retirement_signal("request was unauthenticated")
    assert not _gemini_retirement_signal("SyntaxError: invalid token")
    assert not _gemini_retirement_signal("connection reset by peer")

def test_gemini_date_advisory_fires_only_in_window(tmp_path, capsys, monkeypatch):
    import coding_review_agent_loop.agents.gemini as gm
    from datetime import date

    class _NearCutoff(date):
        @classmethod
        def today(cls):
            return date(2026, 6, 18)

    class _BeforeWindow(date):
        @classmethod
        def today(cls):
            return date(2026, 1, 1)

    config = make_config(tmp_path, quiet=False)

    monkeypatch.setattr(gm, "date", _NearCutoff)
    gm.BACKEND.run(FakeRunner(gemini_outputs=[("ok", 0)]), config, "Review", run_id="r1")
    assert "2026-06-18" in capsys.readouterr().err  # advisory fires near cutoff

    monkeypatch.setattr(gm, "date", _BeforeWindow)
    gm.BACKEND.run(FakeRunner(gemini_outputs=[("ok", 0)]), config, "Review", run_id="r2")
    assert "Antigravity" not in capsys.readouterr().err  # no advisory long before cutoff

def test_gemini_failure_appends_migration_guidance(tmp_path, capsys, monkeypatch):
    import coding_review_agent_loop.agents.gemini as gm
    from datetime import date

    class _BeforeWindow(date):
        @classmethod
        def today(cls):
            return date(2026, 1, 1)  # advisory off, so this isolates the failure path

    monkeypatch.setattr(gm, "date", _BeforeWindow)
    config = make_config(tmp_path, quiet=False)
    runner = FakeRunner(gemini_outputs=[{"stdout": "Error: quota exceeded", "returncode": 1}])
    result = gm.BACKEND.run(runner, config, "Review", run_id="r")
    err = capsys.readouterr().err
    assert "Antigravity" in err and "2026-06-18" in err
    # The guidance must travel with the *returned* result (raw_output and text),
    # since run_external classifies/persists failures from those, not stderr (#215).
    assert "2026-06-18" in result.raw_output
    assert "antigravity" in result.raw_output
    assert result.raw_output.startswith("Error: quota exceeded")  # original error preserved
    assert "2026-06-18" in result.text

def test_gemini_success_does_not_append_migration_guidance(tmp_path, monkeypatch):
    import coding_review_agent_loop.agents.gemini as gm
    from datetime import date

    class _BeforeWindow(date):
        @classmethod
        def today(cls):
            return date(2026, 1, 1)

    monkeypatch.setattr(gm, "date", _BeforeWindow)
    config = make_config(tmp_path)
    runner = FakeRunner(gemini_outputs=[("STATE: approved\n\nLGTM", 0)])
    result = gm.BACKEND.run(runner, config, "Review", run_id="r")
    assert "2026-06-18" not in result.raw_output  # no guidance on a clean success
    assert "2026-06-18" not in result.text


# --- Shared agy settings-lock wait (#1400) ---------------------------------


class _SettingsLockHarness:
    """Fake monotonic clock plus a flock interposer simulating another holder.

    The settings lock reports busy while ``now < release_at``.  ``events`` are
    ``(time, callback)`` pairs fired as the fake clock passes them.
    """

    def __init__(self, tmp_path, monkeypatch, *, release_at=0.0):
        import fcntl
        import importlib

        # Another test reloads the adapter after collection; patch the active module.
        self.mod = importlib.import_module("coding_review_agent_loop.agents.antigravity")
        self.settings = tmp_path / "settings.json"
        self.original_settings = '{"k": "v"}'
        self.settings.write_text(self.original_settings, encoding="utf-8")
        self.lock_path = self.settings.with_suffix(".json.lock")
        self.holder_path = self.lock_path.with_name(self.lock_path.name + ".holder")
        self.start = 1000.0
        self.now = self.start
        self.release_at = self.start + release_at
        self.events: list[tuple[float, object]] = []
        self.blocked_polls = 0
        self.acquired_at: float | None = None
        monkeypatch.setattr(self.mod, "_antigravity_settings_path", lambda: self.settings)
        monkeypatch.setattr(self.mod, "_settings_lock_monotonic", lambda: self.now)
        monkeypatch.setattr(self.mod, "_settings_lock_sleep", self.sleep)
        real_flock = fcntl.flock
        lock_name = str(self.lock_path)

        def interposed(handle, operation):
            if (
                getattr(handle, "name", None) == lock_name
                and operation == fcntl.LOCK_EX | fcntl.LOCK_NB
            ):
                if self.now < self.release_at:
                    self.blocked_polls += 1
                    raise BlockingIOError
                self.acquired_at = self.now
            return real_flock(handle, operation)

        monkeypatch.setattr(fcntl, "flock", interposed)

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds
        for when, callback in list(self.events):
            if self.now >= self.start + when:
                self.events.remove((when, callback))
                callback()

    def write_holder(self, *, token="tok-a", pid=4242, run_id="run-other",
                     seat="flash", role="reviewer", turn_bound_seconds=600):
        import json

        self.holder_path.write_text(json.dumps({
            "token": token, "pid": pid, "run_id": run_id, "seat": seat,
            "role": role, "turn_bound_seconds": turn_bound_seconds,
        }), encoding="utf-8")

    def config(self, tmp_path, **overrides):
        agy_dir = tmp_path / "agy"
        agy_dir.mkdir(parents=True, exist_ok=True)
        return make_config(tmp_path, antigravity_dir=agy_dir, quiet=False, **overrides)


def _agy_commands(runner):
    return [command for command, _ in runner.commands if command[:1] == ["agy"]]


def test_settings_lock_budget_helpers_cover_reviewer_repair_and_explicit_timeouts(
    tmp_path, monkeypatch
):
    import importlib

    agy_mod = importlib.import_module("coding_review_agent_loop.agents.antigravity")

    reviewer = make_config(tmp_path, antigravity_print_timeout_seconds=600, repair_timeout_seconds=120)
    repair = make_config(tmp_path, antigravity_print_timeout_seconds=60, repair_timeout_seconds=900)
    tiny = make_config(tmp_path, antigravity_print_timeout_seconds=1, repair_timeout_seconds=1)
    assert agy_mod.settings_lock_turn_bound_seconds(reviewer) == 600
    assert agy_mod.settings_lock_turn_bound_seconds(repair) == 900
    # An explicit coder/reviewer turn timeout above the print timeout wins.
    assert agy_mod.settings_lock_turn_bound_seconds(reviewer, 1500) == 1500
    assert agy_mod.settings_lock_turn_bound_seconds(reviewer, None) == 600
    assert agy_mod.settings_lock_wait_budget_seconds(600) == 600 + agy_mod.SETTINGS_LOCK_WAIT_MARGIN_SECONDS
    assert agy_mod.settings_lock_wait_budget_seconds(1500) == 1680
    assert agy_mod.settings_lock_wait_budget_seconds(
        agy_mod.settings_lock_turn_bound_seconds(tiny)
    ) == 181
    # Never below the floor, even with a smaller margin.
    assert agy_mod.SETTINGS_LOCK_WAIT_FLOOR_SECONDS == 120
    monkeypatch.setattr(agy_mod, "SETTINGS_LOCK_WAIT_MARGIN_SECONDS", 0)
    assert agy_mod.settings_lock_wait_budget_seconds(
        agy_mod.settings_lock_turn_bound_seconds(tiny)
    ) == 120
    assert agy_mod.settings_lock_wait_budget_seconds(600) == 600
    # The legacy constant remains importable as an alias of the floor.
    assert agy_mod.SETTINGS_LOCK_WAIT_SECONDS == agy_mod.SETTINGS_LOCK_WAIT_FLOOR_SECONDS


def test_settings_lock_waiter_outlasts_a_300s_holder_within_its_turn_bound(
    tmp_path, monkeypatch, capsys
):
    harness = _SettingsLockHarness(tmp_path, monkeypatch, release_at=300)
    harness.write_holder(turn_bound_seconds=600)
    seen_settings: list[str] = []

    class Runner(FakeRunner):
        def run_with_log(self, args, **kwargs):
            seen_settings.append(harness.settings.read_text(encoding="utf-8"))
            assert harness.acquired_at is not None  # agy never starts before the lock
            return super().run_with_log(args, **kwargs)

    runner = Runner(antigravity_outputs=[("the review", 0)])
    config = harness.config(tmp_path, antigravity_print_timeout_seconds=600)
    result = harness.mod.AntigravityBackend().run(runner, config, "Review.", role="reviewer")

    assert result.text == "the review"
    assert len(_agy_commands(runner)) == 1
    assert harness.acquired_at == pytest.approx(harness.start + 300, abs=0.5)
    assert '"k": "v"' in seen_settings[0] and seen_settings[0] != harness.original_settings
    assert harness.settings.read_text(encoding="utf-8") == harness.original_settings
    err = capsys.readouterr().err
    assert "waiting for shared agy settings lock (holder pid 4242, run run-other" in err
    assert "acquired shared agy settings lock after 300." in err


def test_settings_lock_waiter_honors_a_longer_advertised_holder_bound(
    tmp_path, monkeypatch, capsys
):
    harness = _SettingsLockHarness(tmp_path, monkeypatch, release_at=2000)
    harness.write_holder(turn_bound_seconds=2400)
    runner = FakeRunner(antigravity_outputs=[("ok", 0)])
    config = harness.config(tmp_path, antigravity_print_timeout_seconds=600)
    result = harness.mod.AntigravityBackend().run(runner, config, "Review.", role="reviewer")
    assert result.text == "ok"
    assert harness.acquired_at == pytest.approx(harness.start + 2000, abs=0.5)
    assert "turn bound 2400s; allowing 2580s" in capsys.readouterr().err


@pytest.mark.parametrize(
    "record",
    [None, "{not json", '{"token": "t", "turn_bound_seconds": -5}',
     '{"token": "t", "turn_bound_seconds": "Infinity"}',
     '{"token": "t", "turn_bound_seconds": 1e9}',
     '{"token": "t", "turn_bound_seconds": true}',
     # A huge JSON integer must not overflow float() and abort the wait.
     '{"token": "t", "turn_bound_seconds": 1' + "0" * 400 + '}'],
)
def test_settings_lock_waiter_falls_back_to_own_budget_for_unknown_or_untrusted_holder(
    tmp_path, monkeypatch, capsys, record
):
    from coding_review_agent_loop.errors import AgentSettingsLockWaitTimeout

    # Own budget: 600 + 180 = 780s.  The holder never releases in time.
    harness = _SettingsLockHarness(tmp_path, monkeypatch, release_at=5000)
    if record is not None:
        harness.holder_path.write_text(record, encoding="utf-8")
    runner = FakeRunner(antigravity_outputs=[("ok", 0)])
    config = harness.config(tmp_path, antigravity_print_timeout_seconds=600)
    with pytest.raises(AgentSettingsLockWaitTimeout) as caught:
        harness.mod.AntigravityBackend().run(runner, config, "Review.", role="reviewer")
    assert caught.value.waited_seconds == pytest.approx(780, abs=0.5)
    assert caught.value.holder_turn_bound_seconds is None
    err = capsys.readouterr().err
    if record is None or record == "{not json":
        assert "unknown holder; allowing 780s" in err
    else:
        assert "turn bound unknown; allowing 780s" in err
    assert _agy_commands(runner) == []


def test_settings_lock_window_restarts_per_holder_and_overall_cap_fires(
    tmp_path, monkeypatch, capsys
):
    from coding_review_agent_loop.errors import AgentSettingsLockWaitTimeout

    # Each holder stays inside its own 780s allowance; the queue does not.
    harness = _SettingsLockHarness(tmp_path, monkeypatch, release_at=1400)
    harness.write_holder(token="tok-a", pid=1, run_id="run-a")
    harness.events.append((700, lambda: harness.write_holder(token="tok-b", pid=2, run_id="run-b")))
    runner = FakeRunner(antigravity_outputs=[("ok", 0)])
    config = harness.config(tmp_path, antigravity_print_timeout_seconds=600)
    assert harness.mod.AntigravityBackend().run(runner, config, "Review.", role="reviewer").text == "ok"
    assert harness.acquired_at == pytest.approx(harness.start + 1400, abs=0.5)
    err = capsys.readouterr().err
    assert "passed to a new holder (holder pid 2, run run-b" in err

    # A queue of holders that each stay inside 780s still hits 3 x 780s.
    harness = _SettingsLockHarness(tmp_path, monkeypatch, release_at=10_000)
    harness.write_holder(token="tok-0", pid=10)
    for index in range(1, 6):
        harness.events.append(
            (700 * index, lambda index=index: harness.write_holder(token=f"tok-{index}", pid=10 + index))
        )
    runner = FakeRunner(antigravity_outputs=[("ok", 0)])
    with pytest.raises(AgentSettingsLockWaitTimeout, match="overall contention cap") as caught:
        harness.mod.AntigravityBackend().run(runner, config, "Review.", role="reviewer")
    assert caught.value.waited_seconds == pytest.approx(3 * 780, abs=0.5)
    assert _agy_commands(runner) == []


def test_settings_lock_overall_cap_is_never_shorter_than_a_long_holder_allowance(
    tmp_path, monkeypatch
):
    # A short first holder, then a 2400s holder: the cap follows the largest
    # allowance seen (3 x 2580s), so the long holder is waited out.
    harness = _SettingsLockHarness(tmp_path, monkeypatch, release_at=100 + 2500)
    harness.write_holder(token="short", turn_bound_seconds=60)
    harness.events.append((100, lambda: harness.write_holder(token="long", turn_bound_seconds=2400)))
    runner = FakeRunner(antigravity_outputs=[("ok", 0)])
    config = harness.config(tmp_path, antigravity_print_timeout_seconds=600)
    assert harness.mod.AntigravityBackend().run(runner, config, "Review.", role="reviewer").text == "ok"
    assert harness.acquired_at == pytest.approx(harness.start + 2600, abs=0.5)


def test_settings_lock_timeout_is_typed_transient_and_leaves_holder_state_untouched(
    tmp_path, monkeypatch
):
    import os
    from coding_review_agent_loop.errors import AgentInvocationError, AgentSettingsLockWaitTimeout

    harness = _SettingsLockHarness(tmp_path, monkeypatch, release_at=10_000)
    harness.write_holder(token="active", pid=777, run_id="run-active", seat="opus",
                         role="coder", turn_bound_seconds=600)
    holder_before = harness.holder_path.read_text(encoding="utf-8")
    opened: list[object] = []
    real_open = type(harness.lock_path).open

    def tracking_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        if str(path) == str(harness.lock_path):
            opened.append(handle)
        return handle

    monkeypatch.setattr(type(harness.lock_path), "open", tracking_open)
    runner = FakeRunner(antigravity_outputs=[("ok", 0)])
    config = harness.config(tmp_path, antigravity_print_timeout_seconds=600,
                            active_reviewer_seat_id="flash")
    with pytest.raises(AgentSettingsLockWaitTimeout) as caught:
        harness.mod.AntigravityBackend().run(runner, config, "Review.", role="reviewer")
    error = caught.value
    assert isinstance(error, AgentInvocationError)
    assert error.failure_category == "transient"
    assert (error.holder_pid, error.holder_run_id, error.holder_seat, error.holder_role) == (
        777, "run-active", "opus", "coder"
    )
    assert error.holder_turn_bound_seconds == 600
    message = str(error)
    assert message.startswith("flash waited 780s for the shared agy settings lock")
    assert "holder pid 777, run run-active" in message
    assert "settings lock" in message and "outage" not in message
    # No agy command, settings byte-identical, the active holder's record kept,
    # the lock descriptor closed and no stray temp records.
    assert _agy_commands(runner) == []
    assert harness.settings.read_text(encoding="utf-8") == harness.original_settings
    assert harness.holder_path.read_text(encoding="utf-8") == holder_before
    assert opened and all(handle.closed for handle in opened)
    assert not [name for name in os.listdir(tmp_path) if name.endswith(".tmp")]


def test_settings_lock_holder_record_lifecycle(tmp_path, monkeypatch):
    import fcntl
    import json
    import os

    harness = _SettingsLockHarness(tmp_path, monkeypatch)
    order: list[str] = []
    real_flock = fcntl.flock

    def ordered_flock(handle, operation):
        if getattr(handle, "name", None) == str(harness.lock_path) and operation == fcntl.LOCK_UN:
            order.append(
                "unlock:"
                + ("record" if harness.holder_path.exists() else "no-record")
                + ":" + ("restored" if harness.settings.read_text(encoding="utf-8") == harness.original_settings else "injected")
            )
        return real_flock(handle, operation)

    monkeypatch.setattr(fcntl, "flock", ordered_flock)
    published: list[dict] = []

    class Runner(FakeRunner):
        def run_with_log(self, args, **kwargs):
            published.append(json.loads(harness.holder_path.read_text(encoding="utf-8")))
            return super().run_with_log(args, **kwargs)

    config = harness.config(tmp_path, active_reviewer_seat_id="flash")
    harness.mod.AntigravityBackend().run(
        Runner(antigravity_outputs=[("ok", 0)]), config, "Review.", run_id="run-me", role="reviewer"
    )
    assert published[0]["pid"] == os.getpid()
    assert published[0]["run_id"] == "run-me"
    assert published[0]["seat"] == "flash"
    assert published[0]["role"] == "reviewer"
    assert published[0]["turn_bound_seconds"] == 600
    assert published[0]["token"]
    # Removed by its owner after the settings restore and before unlock.
    assert order == ["unlock:no-record:restored"]
    assert not harness.holder_path.exists()

    # The runner raising still removes the owner's record before unlock.
    order.clear()

    class Raising(FakeRunner):
        def run_with_log(self, args, **kwargs):
            raise RuntimeError("agy crashed")

    with pytest.raises(RuntimeError, match="agy crashed"):
        harness.mod.AntigravityBackend().run(Raising(), config, "Review.", role="reviewer")
    assert order == ["unlock:no-record:restored"]

    # A record that no longer carries this invocation's token is left alone.
    order.clear()

    class Replacing(FakeRunner):
        def run_with_log(self, args, **kwargs):
            harness.write_holder(token="someone-else")
            return super().run_with_log(args, **kwargs)

    harness.mod.AntigravityBackend().run(
        Replacing(antigravity_outputs=[("ok", 0)]), config, "Review.", role="reviewer"
    )
    assert json.loads(harness.holder_path.read_text(encoding="utf-8"))["token"] == "someone-else"
    harness.holder_path.unlink()


def test_settings_lock_holder_record_io_failures_are_only_logged(tmp_path, monkeypatch, capsys):
    harness = _SettingsLockHarness(tmp_path, monkeypatch)

    def failing_publish(path, holder):
        raise OSError("disk full")

    monkeypatch.setattr(harness.mod, "_publish_settings_lock_holder", failing_publish)
    harness.write_holder(token="foreign")  # stale record owned by someone else
    config = harness.config(tmp_path)
    result = harness.mod.AntigravityBackend().run(
        FakeRunner(antigravity_outputs=[("ok", 0)]), config, "Review.", role="reviewer"
    )
    assert result.text == "ok"
    assert "could not publish the shared agy settings lock holder record: disk full" in capsys.readouterr().err
    # A failed publish never removes a record it does not own.
    assert harness.holder_path.exists()
    assert harness.settings.read_text(encoding="utf-8") == harness.original_settings
    harness.holder_path.unlink()

    monkeypatch.undo()
    harness = _SettingsLockHarness(tmp_path, monkeypatch)

    def failing_remove(path, token):
        raise OSError("read-only")

    monkeypatch.setattr(harness.mod, "_remove_own_settings_lock_holder", failing_remove)
    result = harness.mod.AntigravityBackend().run(
        FakeRunner(antigravity_outputs=[("ok", 0)]), config, "Review.", role="reviewer"
    )
    assert result.text == "ok"
    assert "could not remove its shared agy settings lock holder record: read-only" in capsys.readouterr().err
    assert harness.settings.read_text(encoding="utf-8") == harness.original_settings


def test_settings_lock_crashed_holder_record_does_not_delay_the_waiter(tmp_path, monkeypatch):
    import json

    # The holder died: its record remains but the kernel released the flock.
    harness = _SettingsLockHarness(tmp_path, monkeypatch, release_at=0)
    harness.write_holder(token="dead", pid=99999, turn_bound_seconds=2400)
    seen: list[dict] = []

    class Runner(FakeRunner):
        def run_with_log(self, args, **kwargs):
            seen.append(json.loads(harness.holder_path.read_text(encoding="utf-8")))
            return super().run_with_log(args, **kwargs)

    config = harness.config(tmp_path)
    assert harness.mod.AntigravityBackend().run(
        Runner(antigravity_outputs=[("ok", 0)]), config, "Review.", role="reviewer"
    ).text == "ok"
    assert harness.blocked_polls == 0
    assert harness.now == harness.start
    assert seen[0]["token"] != "dead"  # overwritten by the new holder
    assert not harness.holder_path.exists()


def test_settings_lock_owner_cleanup_tolerates_an_oversized_integer_bound(tmp_path, monkeypatch):
    harness = _SettingsLockHarness(tmp_path, monkeypatch)

    class Overwriting(FakeRunner):
        def run_with_log(self, args, **kwargs):
            harness.holder_path.write_text(
                '{"token": "x", "turn_bound_seconds": 1' + "0" * 400 + "}", encoding="utf-8"
            )
            return super().run_with_log(args, **kwargs)

    result = harness.mod.AntigravityBackend().run(
        Overwriting(antigravity_outputs=[("ok", 0)]), harness.config(tmp_path), "Review.", role="reviewer"
    )
    assert result.text == "ok"
    assert harness.mod.read_settings_lock_holder(harness.holder_path).turn_bound_seconds is None
    assert harness.holder_path.exists()  # foreign token: left alone


def _semantic_dedupe_setup(tmp_path, monkeypatch):
    import json as _json
    from coding_review_agent_loop.agents import antigravity as agy_mod

    workdir = tmp_path / "isolated"
    workdir.mkdir()
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(_json.dumps({"existingKey": "existingValue"}), encoding="utf-8")
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)
    config = make_config(
        tmp_path,
        antigravity_dir=workdir,
        antigravity_models=("Gemini Flash",),
        antigravity_args=("--dangerously-skip-permissions",),
    )
    return workdir, settings_file, config


def test_antigravity_semantic_dedupe_runs_isolated_classification_turn(tmp_path, monkeypatch):
    """`agy-dedupe-success` (#1472): classification wrapper, strict settings, stdout JSON."""
    import json as _json
    from coding_review_agent_loop.agents.antigravity import (
        AntigravityBackend,
        _REPAIR_SETTINGS_INJECTION,
        _SEMANTIC_DEDUPE_GEMINI_MD,
    )
    from coding_review_agent_loop.protocol import PUBLIC_RESPONSE_MARKER
    from coding_review_agent_loop.semantic_dedupe import parse_semantic_match

    workdir, settings_file, config = _semantic_dedupe_setup(tmp_path, monkeypatch)
    original_settings = settings_file.read_text(encoding="utf-8")
    seen: dict[str, object] = {}

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            seen["settings"] = _json.loads(settings_file.read_text(encoding="utf-8"))
            seen["gemini_md"] = (cwd / "GEMINI.md").read_text(encoding="utf-8")
            seen["timeout"] = kwargs.get("timeout_seconds")
            seen["sibling_lock"] = (cwd.parent / f".{cwd.name}.GEMINI.md.lock").exists()
            return super().run_with_log(args, cwd=cwd, **kwargs)

    answer = '{"duplicate_of": 7, "confidence": "high", "reason": "same work"}'
    runner = CapturingRunner(
        antigravity_outputs=[(f"thinking...\n{PUBLIC_RESPONSE_MARKER}\n{answer}\n", 0)]
    )
    result = AntigravityBackend().run(
        runner, config, "Classify. Return strict JSON only.",
        role="semantic-dedupe", label="semantic-followup-dedupe", timeout_seconds=120.0,
    )

    cmd = runner.commands[-1][0]
    prompt = cmd[cmd.index("--print") + 1]
    assert "--dangerously-skip-permissions" not in cmd
    assert PUBLIC_RESPONSE_MARKER in prompt
    assert "PUBLIC RESPONSE FILE" not in prompt
    assert "AGENT_STATE" not in prompt
    assert "AGENT_PR" not in prompt
    assert "footer" not in prompt.lower()
    assert "signature" not in prompt.lower()
    assert "Run any verification steps" not in prompt
    assert seen["settings"]["toolPermission"] == "strict"
    assert seen["settings"]["permissions"] == _REPAIR_SETTINGS_INJECTION["permissions"]
    assert seen["settings"]["existingKey"] == "existingValue"
    assert _SEMANTIC_DEDUPE_GEMINI_MD in seen["gemini_md"]
    assert "Agent Loop Single-Shot Session" not in seen["gemini_md"]
    assert seen["timeout"] == 120.0
    assert seen["sibling_lock"] is True
    # Settings restored, sibling lock removed, no git metadata or probes.
    assert settings_file.read_text(encoding="utf-8") == original_settings
    assert not (workdir.parent / f".{workdir.name}.GEMINI.md.lock").exists()
    assert not (workdir / ".git").exists()
    assert not (workdir / "GEMINI.md").exists()
    assert runner.git_probe_calls == []
    assert result.text_source == "stdout_marker"
    match = parse_semantic_match(result.text.strip(), allowed_ids={7, 8})
    assert match.duplicate_of == 7


def test_antigravity_semantic_dedupe_ignores_response_files(tmp_path, monkeypatch):
    """`agy-dedupe-stale-response-file` (#1472): only stdout is the answer."""
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    from coding_review_agent_loop.agents.base import public_response_path
    from coding_review_agent_loop.protocol import PUBLIC_RESPONSE_MARKER

    _workdir, _settings_file, config = _semantic_dedupe_setup(tmp_path, monkeypatch)
    stale_path = public_response_path(config, "antigravity")
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("stale review response", encoding="utf-8")
    answer = '{"duplicate_of": null, "confidence": "low", "reason": "distinct"}'
    runner = FakeRunner(
        antigravity_outputs=[(f"{PUBLIC_RESPONSE_MARKER}\n{answer}", 0)],
        public_response_outputs=["a file written during the turn"],
    )

    result = AntigravityBackend().run(runner, config, "Classify.", role="semantic-dedupe")

    assert result.text == answer
    assert result.text_source == "stdout_marker"
    assert result.response_file_text is None
    assert "stale" not in result.text and "file written" not in result.text


def test_antigravity_repair_and_reviewer_paths_keep_their_behaviour(tmp_path, monkeypatch):
    """`agy-other-roles-unchanged` (#1472): only semantic-dedupe gets the classification path."""
    import json as _json
    from coding_review_agent_loop.agents import antigravity as agy_mod
    from coding_review_agent_loop.agents.antigravity import (
        AntigravityBackend,
        _REPAIR_GEMINI_MD,
        _REVIEWER_SETTINGS_INJECTION,
        _SEMANTIC_DEDUPE_GEMINI_MD,
    )

    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(agy_mod, "_antigravity_settings_path", lambda: settings_file)
    seen: list[tuple[str, dict | None]] = []

    class CapturingRunner(FakeRunner):
        def run_with_log(self, args, *, cwd, **kwargs):
            settings = (
                _json.loads(settings_file.read_text(encoding="utf-8"))
                if settings_file.exists()
                else None
            )
            seen.append(((cwd / "GEMINI.md").read_text(encoding="utf-8"), settings))
            return super().run_with_log(args, cwd=cwd, **kwargs)

    agy_dir = tmp_path / "agy"
    agy_dir.mkdir()
    config = make_config(tmp_path, antigravity_dir=agy_dir)
    runner = CapturingRunner(antigravity_outputs=[("ok", 0), ("ok", 0), ("ok", 0)])
    AntigravityBackend().run(runner, config, "Review PR.", role="reviewer")
    reviewer_prompt = runner.commands[-1][0][-1]
    AntigravityBackend().run(runner, config, "Implement.", role="coder")
    coder_prompt = runner.commands[-1][0][-1]
    AntigravityBackend().run_repair(runner, config, "Repair this.", model="ModelA")
    repair_prompt = runner.commands[-1][0][-1]

    (reviewer_md, reviewer_settings), (coder_md, coder_settings), (repair_md, repair_settings) = seen
    assert "Agent Loop Single-Shot Session" in reviewer_md
    assert reviewer_settings["permissions"] == _REVIEWER_SETTINGS_INJECTION["permissions"]
    assert "Agent Loop Single-Shot Session" in coder_md
    assert coder_settings is None
    assert _REPAIR_GEMINI_MD in repair_md
    assert repair_settings["permissions"] == {"allow": []}
    for md in (reviewer_md, coder_md, repair_md):
        assert _SEMANTIC_DEDUPE_GEMINI_MD not in md
    for prompt in (reviewer_prompt, coder_prompt, repair_prompt):
        assert "PUBLIC RESPONSE FILE" in prompt
    assert "Run any verification steps" in coder_prompt
    # Reviewer and coder still take the git-metadata lock and snapshots.
    assert runner.git_probe_calls


# --- Role-aware advertised settings-lock bound (#1508) ----------------------


def test_settings_lock_advertised_turn_bound_narrows_only_semantic_dedupe(tmp_path):
    """`dedupe-without-timeout` / `other-roles-unchanged` (#1508)."""
    import importlib

    agy_mod = importlib.import_module("coding_review_agent_loop.agents.antigravity")
    advertised = agy_mod.settings_lock_advertised_turn_bound_seconds
    config = make_config(tmp_path, antigravity_print_timeout_seconds=600, repair_timeout_seconds=120)
    repair_heavy = make_config(tmp_path, antigravity_print_timeout_seconds=60, repair_timeout_seconds=900)

    assert advertised(config, "semantic-dedupe", 120) == 120
    assert advertised(config, "semantic-dedupe", 120.0) == 120
    assert advertised(config, "semantic-dedupe", 900) == 900
    assert advertised(config, "semantic-dedupe", 5) == 5
    # Without a positive finite timeout the dedupe role never under-advertises.
    for missing in (None, 0, -1):
        assert advertised(config, "semantic-dedupe", missing) == 600
    assert advertised(repair_heavy, "semantic-dedupe", None) == 900
    # Every other role keeps exactly the role-independent bound.
    for role in ("reviewer", "coder", "repair", None):
        for timeout in (None, 120, 1500):
            assert advertised(config, role, timeout) == agy_mod.settings_lock_turn_bound_seconds(
                config, timeout
            )
            assert advertised(repair_heavy, role, timeout) == agy_mod.settings_lock_turn_bound_seconds(
                repair_heavy, timeout
            )
    assert advertised(config, "reviewer", 120) == 600
    assert advertised(config, "coder", 120) == 600
    assert advertised(repair_heavy, "repair", None) == 900


def _record_holder_publications(harness, monkeypatch):
    import fcntl
    import json

    order: list[str] = []
    published: list[dict] = []
    real_flock = fcntl.flock

    def ordered_flock(handle, operation):
        if getattr(handle, "name", None) == str(harness.lock_path) and operation == fcntl.LOCK_UN:
            order.append("unlock:" + ("record" if harness.holder_path.exists() else "no-record"))
        return real_flock(handle, operation)

    monkeypatch.setattr(fcntl, "flock", ordered_flock)

    class Runner(FakeRunner):
        def run_with_log(self, args, **kwargs):
            published.append(json.loads(harness.holder_path.read_text(encoding="utf-8")))
            return super().run_with_log(args, **kwargs)

    return Runner, published, order


def test_semantic_dedupe_holder_publishes_its_runner_timeout(tmp_path, monkeypatch):
    """`dedupe-holder-publishes-runner-timeout` (#1508)."""
    from coding_review_agent_loop.protocol import PUBLIC_RESPONSE_MARKER

    harness = _SettingsLockHarness(tmp_path, monkeypatch)
    Runner, published, order = _record_holder_publications(harness, monkeypatch)
    config = harness.config(
        tmp_path, antigravity_print_timeout_seconds=600, active_reviewer_seat_id="flash"
    )
    answer = '{"duplicate_of": null, "confidence": "low", "reason": "distinct"}'
    result = harness.mod.AntigravityBackend().run(
        Runner(antigravity_outputs=[(f"{PUBLIC_RESPONSE_MARKER}\n{answer}", 0)]),
        config, "Classify.", run_id="run-dedupe",
        role="semantic-dedupe", label="semantic-followup-dedupe", timeout_seconds=120.0,
    )

    assert result.text == answer
    assert len(published) == 1
    assert published[0]["role"] == "semantic-dedupe"
    assert published[0]["turn_bound_seconds"] == 120
    assert published[0]["run_id"] == "run-dedupe"
    assert set(published[0]) == {"token", "pid", "run_id", "seat", "role", "turn_bound_seconds"}
    # Removed by its owner before unlock, exactly as for other roles.
    assert order == ["unlock:no-record"]
    assert not harness.holder_path.exists()
    assert harness.settings.read_text(encoding="utf-8") == harness.original_settings


@pytest.mark.parametrize(
    ("role", "timeout_seconds", "expected"),
    [("reviewer", None, 600), ("reviewer", 1500.0, 1500), ("coder", 120.0, 600),
     ("coder", None, 600), ("repair", None, 900)],
)
def test_non_dedupe_holders_publish_the_role_independent_bound(
    tmp_path, monkeypatch, role, timeout_seconds, expected
):
    """`other-roles-unchanged` (#1508): only semantic-dedupe narrows its bound."""
    harness = _SettingsLockHarness(tmp_path, monkeypatch)
    Runner, published, order = _record_holder_publications(harness, monkeypatch)
    repair_timeout = 900 if role == "repair" else 120
    config = harness.config(
        tmp_path, antigravity_print_timeout_seconds=600, repair_timeout_seconds=repair_timeout
    )
    kwargs = {"timeout_seconds": timeout_seconds} if timeout_seconds is not None else {}
    harness.mod.AntigravityBackend().run(
        Runner(antigravity_outputs=[("ok", 0)]), config, "Work.", role=role, **kwargs
    )
    assert published[0]["role"] == role
    assert published[0]["turn_bound_seconds"] == expected
    assert published[0]["turn_bound_seconds"] == harness.mod.settings_lock_turn_bound_seconds(
        config, timeout_seconds
    )
    assert order == ["unlock:no-record"]


def test_settings_lock_waiter_behind_a_dedupe_holder_allows_300s(tmp_path, monkeypatch, capsys):
    """`waiter-behind-dedupe-holder` (#1508): 120s bound + 180s margin, not 780s."""
    from coding_review_agent_loop.errors import AgentSettingsLockWaitTimeout

    harness = _SettingsLockHarness(tmp_path, monkeypatch, release_at=5000)
    harness.write_holder(role="semantic-dedupe", turn_bound_seconds=120)
    runner = FakeRunner(antigravity_outputs=[("ok", 0)])
    config = harness.config(tmp_path, antigravity_print_timeout_seconds=600)
    with pytest.raises(AgentSettingsLockWaitTimeout) as caught:
        harness.mod.AntigravityBackend().run(runner, config, "Review.", role="reviewer")
    assert caught.value.waited_seconds == pytest.approx(300, abs=0.5)
    assert caught.value.holder_role == "semantic-dedupe"
    assert caught.value.holder_turn_bound_seconds == 120
    assert "turn bound 120s; allowing 300s" in capsys.readouterr().err
    assert _agy_commands(runner) == []
    assert harness.settings.read_text(encoding="utf-8") == harness.original_settings


@pytest.mark.parametrize("record", [None, "{not json", '{"token": "t", "role": "semantic-dedupe"}'])
def test_semantic_dedupe_waiter_keeps_role_independent_unknown_holder_allowance(
    tmp_path, monkeypatch, capsys, record
):
    """`dedupe-waiter-unknown-holder` (#1508): own fallback is not the dedupe timeout."""
    from coding_review_agent_loop.errors import AgentSettingsLockWaitTimeout

    harness = _SettingsLockHarness(tmp_path, monkeypatch, release_at=5000)
    if record is not None:
        harness.holder_path.write_text(record, encoding="utf-8")
    runner = FakeRunner(antigravity_outputs=[("ok", 0)])
    config = harness.config(tmp_path, antigravity_print_timeout_seconds=600)
    with pytest.raises(AgentSettingsLockWaitTimeout) as caught:
        harness.mod.AntigravityBackend().run(
            runner, config, "Classify.", role="semantic-dedupe", timeout_seconds=120.0
        )
    assert caught.value.waited_seconds == pytest.approx(780, abs=0.5)
    assert caught.value.holder_turn_bound_seconds is None
    assert "allowing 780s" in capsys.readouterr().err
    assert _agy_commands(runner) == []


# --- silent quota failure: agy CLI log lookup (#1604) -------------------------

import os as _os
import time as _time

import importlib as _importlib

from coding_review_agent_loop.runner import CommandResult as _CommandResult


class _ActiveAgy:
    """The live adapter module: another test re-imports it after collection."""

    def __getattr__(self, name):
        return getattr(_importlib.import_module("coding_review_agent_loop.agents.antigravity"), name)


_agy = _ActiveAgy()


def _active_agy_module():
    return _importlib.import_module("coding_review_agent_loop.agents.antigravity")

_AGY_429_LOG = AGY_LOG_STARTUP_LINES + AGY_LOG_429_LINE + "\n"


def _cmd_result(stdout, returncode=0):
    return _CommandResult(["agy"], Path("."), stdout, "", returncode)


@pytest.mark.parametrize(
    "stdout,returncode,response_file,expected",
    [
        ("", 0, None, True),
        (AGY_SILENT_WARNING + "\n", 0, None, True),
        ("\x1b[33mWarning: Run ended with no output and no recorded error\x1b[0m\r\n", 1, None, True),
        (AGY_SILENT_WARNING + "\nerror: Your AI credits balance is too low\n", 0, None, False),
        ("some review text\n", 0, None, False),
        ("", None, None, False),
        ("", 0, "response file text", False),
    ],
    ids=["empty", "warning", "ansi-warning", "warning-plus-text", "other-text", "timeout", "response-file"],
)
def test_silent_antigravity_attempt_detector(stdout, returncode, response_file, expected):
    assert _agy.is_silent_antigravity_attempt(_cmd_result(stdout, returncode), response_file) is expected


def test_agy_log_inventory_missing_and_unreadable(tmp_path):
    assert _agy.snapshot_antigravity_logs(tmp_path / "missing") == frozenset()
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert _agy.snapshot_antigravity_logs(blocker / "log") is None
    (tmp_path / "log").mkdir()
    (tmp_path / "log" / "cli-20261022_020350.log").write_text("", encoding="utf-8")
    (tmp_path / "log" / "other.txt").write_text("", encoding="utf-8")
    assert _agy.snapshot_antigravity_logs(tmp_path / "log") == frozenset({"cli-20261022_020350.log"})


def _write_log(log_dir, wall_time, mtime, text=_AGY_429_LOG):
    path = log_dir / agy_log_name(wall_time)
    path.write_text(text, encoding="utf-8")
    _os.utime(path, (mtime, mtime))
    return path


def test_agy_log_locator_matches_single_new_in_window_log(tmp_path):
    spawn = _time.time()
    path = _write_log(tmp_path, spawn, spawn + 0.5)
    lookup = _agy.find_antigravity_attempt_log(tmp_path, frozenset(), spawn, spawn + 1)
    assert lookup.log_path == path


def test_agy_log_locator_rejects_pre_existing_log_inside_both_windows(tmp_path):
    spawn = _time.time()
    prior = _write_log(tmp_path, spawn - 1, spawn - 0.5)
    inventory = _agy.snapshot_antigravity_logs(tmp_path)
    assert prior.name in inventory
    lookup = _agy.find_antigravity_attempt_log(tmp_path, inventory, spawn, spawn + 1)
    assert lookup.log_path is None
    assert "no new agy log" in lookup.reason


@pytest.mark.parametrize(
    "case",
    ["name-outside-window", "stale-mtime", "symlink", "ambiguous", "no-directory", "no-inventory"],
)
def test_agy_log_locator_refuses_unusable_candidates(tmp_path, case):
    spawn = _time.time()
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    inventory = frozenset()
    if case == "name-outside-window":
        _write_log(log_dir, spawn + 60, spawn + 0.5)
    elif case == "stale-mtime":
        _write_log(log_dir, spawn, spawn - 30)
    elif case == "symlink":
        target = tmp_path / "elsewhere.log"
        target.write_text(_AGY_429_LOG, encoding="utf-8")
        (log_dir / agy_log_name(spawn)).symlink_to(target)
    elif case == "ambiguous":
        _write_log(log_dir, spawn, spawn + 0.5)
        _write_log(log_dir, spawn + 1, spawn + 0.5)
    elif case == "no-directory":
        log_dir.rmdir()
    else:
        _write_log(log_dir, spawn, spawn + 0.5)
        inventory = None
    lookup = _agy.find_antigravity_attempt_log(log_dir, inventory, spawn, spawn + 1)
    assert lookup.log_path is None
    assert lookup.reason
    if case == "ambiguous":
        assert "ambiguous" in lookup.reason
    if case == "no-inventory":
        assert lookup.reason == "no pre-spawn inventory"


def test_agy_log_reader_finds_only_executor_429_lines(tmp_path):
    path = tmp_path / "cli.log"
    path.write_text(_AGY_429_LOG, encoding="utf-8")
    assert _agy.antigravity_log_quota_line(path) == AGY_LOG_429_LINE
    path.write_text(
        AGY_LOG_STARTUP_LINES
        + "E1022 02:03:52.215232 1 x.go:1] agent executor error: generating and executing: "
        "INTERNAL (code 500)\n"
        + "I1022 02:03:52.215232 1 x.go:1] agent executor error: RESOURCE_EXHAUSTED (code 429)\n"
        + "E1022 02:03:52.215232 1 x.go:1] fetch failed: RESOURCE_EXHAUSTED (code 429)\n",
        encoding="utf-8",
    )
    assert _agy.antigravity_log_quota_line(path) is None
    path.write_text(AGY_LOG_STARTUP_LINES, encoding="utf-8")
    assert _agy.antigravity_log_quota_line(path) is None
    link = tmp_path / "link.log"
    path.write_text(_AGY_429_LOG, encoding="utf-8")
    link.symlink_to(path)
    assert _agy.antigravity_log_quota_line(link) is None
    assert _agy.antigravity_log_quota_line(tmp_path / "missing.log") is None


def test_agy_log_reader_is_bounded_to_the_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(_active_agy_module(), "AGY_LOG_READ_LIMIT_BYTES", 4096)
    path = tmp_path / "cli.log"
    path.write_text(AGY_LOG_429_LINE + "\n" + ("I1022 filler line\n" * 2000), encoding="utf-8")
    assert _agy.antigravity_log_quota_line(path) is None
    with path.open("a", encoding="utf-8") as handle:
        handle.write(AGY_LOG_429_LINE + "\n")
    assert _agy.antigravity_log_quota_line(path) == AGY_LOG_429_LINE


def _backend_config(tmp_path):
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    return make_config(tmp_path, antigravity_dir=agy_dir)


def test_backend_attaches_evidence_for_silent_attempt_with_new_429_log(tmp_path, _isolated_agy_log_dir):
    AntigravityBackend = _agy.AntigravityBackend

    runner = AgyLogRunner(
        antigravity_outputs=[(AGY_SILENT_WARNING, 0)],
        log_dir=_isolated_agy_log_dir, agy_logs=[_AGY_429_LOG],
    )
    result = AntigravityBackend().run(runner, _backend_config(tmp_path), "Review.", run_id="r1")
    written = runner.written_agy_logs[0]
    assert result.antigravity_log_path == written
    assert result.antigravity_log_quota_evidence.log_path == written
    assert result.antigravity_log_quota_evidence.line == AGY_LOG_429_LINE


def test_backend_records_log_path_without_evidence_when_no_429(tmp_path, _isolated_agy_log_dir):
    AntigravityBackend = _agy.AntigravityBackend

    runner = AgyLogRunner(
        antigravity_outputs=[("", 0)],
        log_dir=_isolated_agy_log_dir, agy_logs=[AGY_LOG_STARTUP_LINES],
    )
    result = AntigravityBackend().run(runner, _backend_config(tmp_path), "Review.", run_id="r1")
    assert result.antigravity_log_path == runner.written_agy_logs[0]
    assert result.antigravity_log_quota_evidence is None


@pytest.mark.parametrize(
    "output,response_file",
    [(("plain review text", 0), None), ((AGY_SILENT_WARNING, None), None), (("", 0), "response file text")],
    ids=["other-text", "timeout", "response-file"],
)
def test_backend_skips_lookup_for_non_silent_attempts(
    tmp_path, _isolated_agy_log_dir, monkeypatch, output, response_file
):
    AntigravityBackend = _agy.AntigravityBackend

    calls = []
    monkeypatch.setattr(
        _agy, "find_antigravity_attempt_log", lambda *a, **k: calls.append(a) or None
    )
    runner = AgyLogRunner(
        antigravity_outputs=[output],
        public_response_outputs=[response_file] if response_file else [],
        log_dir=_isolated_agy_log_dir, agy_logs=[_AGY_429_LOG],
    )
    result = AntigravityBackend().run(runner, _backend_config(tmp_path), "Review.", run_id="r1")
    assert calls == []
    assert result.antigravity_log_path is None
    assert result.antigravity_log_quota_evidence is None


def test_backend_takes_log_inventory_under_settings_lock_before_spawn(tmp_path, _isolated_agy_log_dir, monkeypatch):
    import fcntl
    AntigravityBackend = _agy.AntigravityBackend

    settings_file = tmp_path / "agy-settings" / "settings.json"
    monkeypatch.setattr(_active_agy_module(), "_antigravity_settings_path", lambda: settings_file)
    lock_path = settings_file.with_suffix(".json.lock")
    events = []
    real_snapshot = _agy.snapshot_antigravity_logs

    def snapshot(log_dir):
        with lock_path.open("a+") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                events.append("snapshot-locked")
            else:
                fcntl.flock(probe, fcntl.LOCK_UN)
                events.append("snapshot-unlocked")
        return real_snapshot(log_dir)

    monkeypatch.setattr(_active_agy_module(), "snapshot_antigravity_logs", snapshot)

    class OrderRunner(AgyLogRunner):
        def run_with_log(self, args, **kwargs):
            if list(args[:1]) == ["agy"]:
                events.append("spawn")
            return super().run_with_log(args, **kwargs)

    runner = OrderRunner(
        antigravity_outputs=[(AGY_SILENT_WARNING, 0)],
        log_dir=_isolated_agy_log_dir, agy_logs=[_AGY_429_LOG],
    )
    result = AntigravityBackend().run(runner, _backend_config(tmp_path), "Review.", run_id="r1")
    assert events == ["snapshot-locked", "spawn"]
    assert result.antigravity_log_quota_evidence is not None
