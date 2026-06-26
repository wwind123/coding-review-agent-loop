from agent_loop_helpers import *  # noqa: F403


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
    assert "--dangerously-skip-permissions" in cmd
    # The prompt is the value of --print and must be the last argument (agy's
    # --print/--prompt consumes the next token, not a trailing positional).
    assert cmd[-2] == "--print"
    assert "Review this PR." in cmd[-1]
    assert result.text == "response file text"
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

def test_antigravity_backend_fallback_chain_on_quota_signal(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(
        antigravity_outputs=[
            ("quota exceeded please try again", 1),
            ("quota exceeded again", 1),
            ("ok fallback answered", 0)
        ]
    )
    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_models=("ModelA", "ModelB", "ModelC"),
        antigravity_quota_signatures=("quota",)
    )
    result = AntigravityBackend().run(runner, config, "Review", run_id="r1")
    
    assert runner.commands[-3][0][runner.commands[-3][0].index("--model") + 1] == "ModelA"
    assert runner.commands[-2][0][runner.commands[-2][0].index("--model") + 1] == "ModelB"
    assert runner.commands[-1][0][runner.commands[-1][0].index("--model") + 1] == "ModelC"
    
    assert result.text == "ok fallback answered"
    assert result.model_used == "ModelC"

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

def test_antigravity_backend_ignores_partial_response_file_on_fallback(tmp_path):
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    agy_dir = tmp_path / "antigravity"
    agy_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner(
        antigravity_outputs=[
            ("quota error", 1),
            ("success", 0)
        ],
        public_response_outputs=[
            "partial failed response",
            "successful response"
        ]
    )
    config = make_config(
        tmp_path,
        antigravity_dir=agy_dir,
        antigravity_models=("ModelA", "ModelB"),
        antigravity_quota_signatures=("quota",)
    )
    result = AntigravityBackend().run(runner, config, "Review", run_id="r1")
    
    assert result.text == "successful response"
    assert result.model_used == "ModelB"

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
        lock_released.wait()
        order.append("A-releases")
        fcntl.flock(lf, fcntl.LOCK_UN)
        lf.close()

    t = threading.Thread(target=hold_lock, daemon=True)
    t.start()
    lock_acquired.wait()

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

    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()

    # Give the backend thread a moment to block on the lock, then release it.
    import time
    time.sleep(0.05)
    lock_released.set()
    t.join(timeout=5)
    backend_thread.join(timeout=10)

    # A-holds must precede B-run and A-releases must precede B-run
    assert "A-holds" in order
    assert "A-releases" in order
    assert "B-run" in order
    assert order.index("A-holds") < order.index("B-run")
    assert order.index("A-releases") < order.index("B-run")

def test_antigravity_module_imports_without_fcntl():
    """Antigravity module must import cleanly even when fcntl is unavailable (Windows)."""
    import importlib
    import sys

    # Remove any cached import of the module under test
    mods_to_remove = [k for k in sys.modules if "antigravity" in k]
    for m in mods_to_remove:
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
        # Re-remove so later tests get a clean import
        for k in list(sys.modules):
            if "antigravity" in k:
                del sys.modules[k]
        real_mod = importlib.import_module("coding_review_agent_loop.agents.antigravity")
        import coding_review_agent_loop.agents.registry as registry_mod
        import coding_review_agent_loop.repair as repair_mod

        registry_mod.BACKENDS["antigravity"] = real_mod.BACKEND
        repair_mod.AntigravityBackend = real_mod.AntigravityBackend

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
    assert config.antigravity_models == ("Gemini 3.5 Flash (High)", "Gemini 3.1 Pro (High)")
    assert config.antigravity_quota_signatures == ("quota", "rate limit", "resource exhausted", "RESOURCE_EXHAUSTED", "429")
    assert config.antigravity_args == ("--dangerously-skip-permissions",)
    assert config.antigravity_dir == default_agent_workdir("OWNER/REPO", "antigravity").resolve()
    # antigravity is the coder -> primary/log dir lives under its checkout.
    assert str(config.log_dir).startswith(str(config.antigravity_dir))

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
        ("settings", fcntl_mod.LOCK_EX),
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
            a_release.wait(timeout=10)
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

    ta = threading.Thread(target=run_a, daemon=True)
    ta.start()
    a_in_runner.wait(timeout=10)

    tb = threading.Thread(target=run_b, daemon=True)
    tb.start()

    a_release.set()
    b_in_runner.wait(timeout=10)

    ta.join(timeout=10)
    tb.join(timeout=10)

    assert not ta.is_alive(), "Thread A deadlocked"
    assert not tb.is_alive(), "Thread B deadlocked"
    assert order == ["a_in_runner", "b_in_runner"]
    assert settings_file.read_text(encoding="utf-8") == '{"k": "v"}'

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
                and operation == fcntl_mod.LOCK_EX
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
                real_flock(fd, fcntl_mod.LOCK_EX)
                return
            real_flock(fd, operation)

        monkeypatch.setattr(fcntl_mod, "flock", instrumented_flock)

        config = make_config(tmp_path, antigravity_dir=agy_dir)

        class HolderRunner(FakeRunner):
            def run_with_log(self, args, *, cwd, **kwargs):
                holder_in_runner.set()
                release_holder.wait(timeout=10)
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

        th = threading.Thread(target=run_holder, daemon=True)
        th.start()
        holder_in_runner.wait(timeout=10)

        tc = threading.Thread(target=run_contender, daemon=True)
        tc.start()

        contender_attempting_lock.wait(timeout=10)
        contender_probe_done.wait(timeout=10)
        release_holder.set()
        contender_in_runner.wait(timeout=10)

        th.join(timeout=10)
        tc.join(timeout=10)

        assert not th.is_alive(), f"Holder deadlocked (holder_role={holder_role!r})"
        assert not tc.is_alive(), f"Contender deadlocked (contender_role={contender_role!r})"
        assert contender_in_runner.is_set()
        assert len(nb_probe_results) == 1
        assert isinstance(nb_probe_results[0], BlockingIOError), (
            f"LOCK_NB probe should have failed with BlockingIOError "
            f"but got {nb_probe_results[0]!r} "
            f"(holder_role={holder_role!r}, contender_role={contender_role!r})"
        )

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
            contention_confirmed_event.wait(timeout=10)
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

    ta = threading.Thread(target=run_a, daemon=True)
    ta.start()
    injected_event.wait(timeout=10)

    tb = threading.Thread(target=run_b, daemon=True)
    tb.start()

    ta.join(timeout=10)
    tb.join(timeout=10)

    assert not ta.is_alive(), "Thread A deadlocked"
    assert not tb.is_alive(), "Thread B deadlocked"
    assert isinstance(thread_a_exc[0], RuntimeError)
    assert isinstance(thread_b_nb_error[0], BlockingIOError), (
        f"LOCK_NB should have raised BlockingIOError; got {thread_b_nb_error[0]!r}"
    )
    assert thread_b_result[0] == original_text, (
        f"Settings must be restored before the lock is released; got {thread_b_result[0]!r}"
    )

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
