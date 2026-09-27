"""Tests for ``--agent-permissions sandboxed`` role grants and boundaries (#1035)."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import typing
from pathlib import Path

import pytest

import coding_review_agent_loop
from coding_review_agent_loop import agent_permissions as ap
from coding_review_agent_loop import test_runtime
from coding_review_agent_loop.agents.claude import BACKEND as CLAUDE_BACKEND
from coding_review_agent_loop.agents.codex import BACKEND as CODEX_BACKEND
from coding_review_agent_loop.agents.registry import run_agent_result
from coding_review_agent_loop.config import AgentLoopConfig
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.prompts import _coder_workdir_guidance
from coding_review_agent_loop.protocol import parse_agent_unavailable

from agent_loop_helpers import FakeRunner, make_config

PACKAGE_DIR = str(Path(coding_review_agent_loop.__file__).resolve().parent)
WRAPPER = ("/opt/tools/venv/bin/python", "-m", "coding_review_agent_loop.cli", "run-tests")


def _executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    ap.reset_sandbox_state()
    trusted = tmp_path / "trusted" / "bin"
    tools = {"git": _executable(trusted / "git"), "gh": _executable(trusted / "gh")}
    monkeypatch.setattr(test_runtime, "verified_wrapper_prefix", lambda **_kwargs: WRAPPER)

    def establish(config, *, which=None, locator=None):
        ap.establish_response_root_boundary(config)
        return ap.establish_inspect_provenance(
            config,
            which=which or (lambda name: str(tools[name]) if name in tools else None),
            package_locator=locator or (lambda _interpreter: PACKAGE_DIR),
        )

    yield {"tmp": tmp, "tools": tools, "establish": establish, "trusted": trusted}
    ap.reset_sandbox_state()


def sandboxed_config(tmp_path, **overrides):
    values = dict(
        agent_permissions="sandboxed",
        reviewer=("codex", "claude"),
        repair_backend="claude",
        repair_models=("repair-model",),
        semantic_followup_backend="claude",
        antigravity_dir=tmp_path / "antigravity",
    )
    values.update(overrides)
    return make_config(tmp_path, **values)


def _rule_section(argv, option):
    index = argv.index(option)
    rules = []
    for item in argv[index + 1 :]:
        if item.startswith("--"):
            break
        rules.append(item)
    return rules


def _allowed_tools(argv):
    return _rule_section(argv, "--allowedTools")


def _disallowed_tools(argv):
    return _rule_section(argv, "--disallowedTools")


# ----------------------------------------------------------- grant builder


def test_role_class_fails_closed():
    assert ap.permission_class_for_role("coder") == "coder"
    for role in (None, "reviewer", "summary", "repair", "Coder", "coder ", "planner", "?"):
        assert ap.permission_class_for_role(role) == "read-only"


def test_claude_reviewer_gets_restricted_read_only_grant(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    provenance = sandbox["establish"](config)
    root = ap.sandboxed_response_root(config)
    argv = ap.role_permission_args(config, "claude", "reviewer")
    assert "--restricted" in argv
    assert argv[argv.index("--tools") + 1] == "Read,Grep,Glob,Write,Edit,Bash"
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert argv[argv.index("--permission-prompts") + 1] == "none"
    assert "--strict-mcp-config" in argv
    assert json.loads(argv[argv.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert argv[argv.index("--add-dir") + 1] == str(root)
    rules = _allowed_tools(list(argv))
    assert rules[:2] == [f"Write(/{root}/**)", f"Edit(/{root}/**)"]
    bash_rules = [rule for rule in rules if rule.startswith("Bash(")]
    expected_prefix = (
        f"{os.path.abspath(sys.executable)} -I -m coding_review_agent_loop.cli inspect "
        f"--git={sandbox['tools']['git']} --gh={sandbox['tools']['gh']}"
    )
    assert provenance.prefix_text == expected_prefix
    assert bash_rules == [f"Bash({expected_prefix} *)"]
    assert "acceptEdits" not in argv
    assert not any("git *" == rule[5:-1] or "grep" in rule for rule in bash_rules)
    assert "run-tests" not in " ".join(argv)
    # The prompt names the same inspect forms.
    guidance = _coder_workdir_guidance(config, implementation=False, agent="claude")
    assert f"`{expected_prefix} git diff main...HEAD`" in guidance


@pytest.mark.parametrize("role", [None, "summary", "analyzer", "semantic-dedupe", "unknown-role"])
def test_role_less_and_unknown_roles_get_read_only_grant(tmp_path, sandbox, role):
    config = sandboxed_config(tmp_path)
    sandbox["establish"](config)
    argv = ap.role_permission_args(config, "claude", role)
    assert "--restricted" in argv and "acceptEdits" not in argv
    assert "Bash(git *)" not in _allowed_tools(list(argv))
    assert "Bash(git *)" in _disallowed_tools(list(argv))
    assert ap.role_permission_args(config, "codex", role)[:2] == ("--sandbox", "read-only")


def test_claude_coder_grant_has_exact_test_rule_matching_prompt(tmp_path, sandbox):
    config = sandboxed_config(tmp_path, test_command=("pytest", "-q"))
    sandbox["establish"](config)
    root = ap.sandboxed_response_root(config)
    argv = list(ap.role_permission_args(config, "claude", "coder"))
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    assert argv[argv.index("--setting-sources") + 1] == "user"
    assert argv[argv.index("--add-dir") + 1] == str(root)
    rules = _allowed_tools(argv)
    assert "Bash(git *)" in rules
    for sub in ("gh pr create", "gh pr view", "gh pr edit", "gh issue view", "gh pr checks"):
        assert f"Bash({sub} *)" in rules
    invocation = test_runtime.resolve_coder_test_invocation(config)
    assert invocation == test_runtime.render_test_wrapper(
        ("pytest", "-q"), memory_dir=config.agent_memory_dir, prefix=WRAPPER
    )
    test_rules = [rule for rule in rules if "run-tests" in rule]
    assert test_rules == [f"Bash({invocation})"]
    assert "*" not in test_rules[0]
    assert "Bash(*)" not in rules
    assert "--dangerously-skip-permissions" not in argv and "--restricted" not in argv
    # The coder prompt line is the same string.
    guidance = _coder_workdir_guidance(config)
    assert f"`{invocation}`" in guidance


def test_checkout_settings_cannot_widen_grants(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    settings = config.claude_dir / ".claude"
    settings.mkdir(parents=True)
    (settings / "settings.json").write_text('{"permissions":{"allow":["Bash(*)"]}}', encoding="utf-8")
    (settings / "settings.local.json").write_text('{"permissions":{"allow":["Bash(*)"]}}', encoding="utf-8")
    sandbox["establish"](config)
    reviewer = ap.role_permission_args(config, "claude", "reviewer")
    coder = list(ap.role_permission_args(config, "claude", "coder"))
    assert "--restricted" in reviewer
    assert coder[coder.index("--setting-sources") + 1] == "user"
    assert "project" not in " ".join(coder) and "local" not in coder


def test_default_and_dangerous_modes_add_no_role_args(tmp_path, sandbox):
    for mode in ("default", "dangerous"):
        config = make_config(tmp_path, agent_permissions=mode)
        assert ap.role_permission_args(config, "claude", "reviewer") == ()
        assert ap.role_permission_args(config, "codex", "coder") == ()


def test_codex_coder_role_is_refused_at_runtime(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    sandbox["establish"](config)
    with pytest.raises(AgentLoopError, match="read-only"):
        ap.role_permission_args(config, "codex", "coder")


# ------------------------------------------------------ coder test resolver


def test_resolver_prefers_test_command_then_profile(tmp_path, sandbox):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "test-profile.md").write_text(
        "# Test Profile\n\nVerified test commands:\n- `python -m pytest tests/ -q`\n\nSuggested:\n",
        encoding="utf-8",
    )
    config = sandboxed_config(tmp_path, agent_memory_dir=memory_dir, test_command=("make", "test"))
    first = test_runtime.resolve_coder_test_invocation(config)
    assert first.endswith("-- make test")
    ap.reset_sandbox_state()
    config = sandboxed_config(tmp_path, agent_memory_dir=memory_dir)
    second = test_runtime.resolve_coder_test_invocation(config)
    assert second.endswith("-- python -m pytest tests/ -q")
    assert second.startswith(" ".join(WRAPPER))
    # Memoized: the same string for the permission builder and the prompt.
    assert test_runtime.resolve_coder_test_invocation(config) is second


def test_resolver_without_command_or_wrapper_omits_rule_and_warns(tmp_path, sandbox, monkeypatch):
    logged = []
    monkeypatch.setattr("coding_review_agent_loop.logging.log", lambda _c, message: logged.append(message))
    config = sandboxed_config(tmp_path, agent_memory=False)
    sandbox["establish"](config)
    assert test_runtime.resolve_coder_test_invocation(config) is None
    assert any("test grant omitted" in message for message in logged)
    rules = _allowed_tools(list(ap.role_permission_args(config, "claude", "coder")))
    assert not any("run-tests" in rule for rule in rules)
    assert "Bash(*)" not in rules
    assert "No test command was resolved" in _coder_workdir_guidance(config)

    ap.reset_sandbox_state()
    monkeypatch.setattr(test_runtime, "verified_wrapper_prefix", lambda **_kwargs: None)
    config = sandboxed_config(tmp_path, test_command=("pytest",))
    assert test_runtime.resolve_coder_test_invocation(config) is None


def test_resolver_uses_verified_prefix_not_resolved_interpreter(tmp_path, sandbox, monkeypatch):
    seen = {}

    def verified(**kwargs):
        seen.update(kwargs)
        return WRAPPER

    monkeypatch.setattr(test_runtime, "verified_wrapper_prefix", verified)
    monkeypatch.setattr(
        test_runtime, "resolve_wrapper_prefix", lambda: pytest.fail("must not resolve the venv away")
    )
    config = sandboxed_config(tmp_path, test_command=("pytest",))
    invocation = test_runtime.resolve_coder_test_invocation(config)
    assert invocation.startswith("/opt/tools/venv/bin/python -m coding_review_agent_loop.cli run-tests")
    assert seen["cwd"] == config.claude_dir.resolve()


# ------------------------------------------------------ response-root boundary


def test_response_root_is_created_private_and_granted(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    state = ap.establish_response_root_boundary(config)
    assert state.resolved_root == Path(os.path.realpath(sandbox["tmp"])) / (
        "coding-review-agent-loop/responses/OWNER-REPO"
    )
    path = ap.prepare_response_file(config, "claude")
    assert path.parent == state.resolved_root / "claude"
    assert path.read_text() == ""
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_tmpdir_inside_checkout_fails_fast(tmp_path, monkeypatch):
    config = sandboxed_config(tmp_path)
    inside = config.claude_dir / "tmp"
    inside.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(inside))
    ap.reset_sandbox_state()
    with pytest.raises(AgentLoopError, match="overlaps agent checkout"):
        ap.establish_response_root_boundary(config)


def test_symlinked_component_into_checkout_fails_fast(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    target = config.claude_dir / "responses"
    target.mkdir()
    (sandbox["tmp"] / "coding-review-agent-loop").mkdir()
    (sandbox["tmp"] / "coding-review-agent-loop" / "responses").symlink_to(target)
    with pytest.raises(AgentLoopError, match="not a real directory"):
        ap.establish_response_root_boundary(config)


def test_checkout_inside_response_root_fails_fast(tmp_path, sandbox):
    root = sandbox["tmp"] / "coding-review-agent-loop" / "responses" / "OWNER-REPO"
    config = sandboxed_config(tmp_path, claude_dir=root / "claude" / "checkout")
    with pytest.raises(AgentLoopError, match="overlaps agent checkout"):
        ap.establish_response_root_boundary(config)


def test_foreign_owned_component_fails_fast(tmp_path, sandbox, monkeypatch):
    config = sandboxed_config(tmp_path)
    monkeypatch.setattr(os, "getuid", lambda: 424242)
    with pytest.raises(AgentLoopError, match="owned by uid"):
        ap.establish_response_root_boundary(config)


def test_component_swapped_for_symlink_blocks_next_spawn(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    state = ap.establish_response_root_boundary(config)
    provider_dir = state.resolved_root / "codex"
    shutil.rmtree(provider_dir)
    provider_dir.symlink_to(config.claude_dir)
    with pytest.raises(ap.SandboxBoundaryError):
        ap.prepare_response_file(config, "codex")


def test_checkout_replaced_between_turns_blocks_spawn(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    ap.establish_response_root_boundary(config)
    config.claude_dir.rename(tmp_path / "claude-old")
    config.claude_dir.mkdir()
    with pytest.raises(ap.SandboxBoundaryError, match="changed since it was recorded"):
        ap.prepare_response_file(config, "claude")


def test_checkout_redirected_into_response_root_blocks_spawn(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    state = ap.establish_response_root_boundary(config)
    shutil.rmtree(config.claude_dir)
    config.claude_dir.symlink_to(state.resolved_root / "claude")
    with pytest.raises(ap.SandboxBoundaryError):
        ap.prepare_response_file(config, "claude")


def test_checkout_moved_so_stored_path_resolves_inside_root_blocks_spawn(tmp_path, sandbox):
    parent = tmp_path / "parent"
    parent.mkdir()
    config = sandboxed_config(tmp_path, claude_dir=parent / "checkout")
    state = ap.establish_response_root_boundary(config)
    moved = tmp_path / "parent-moved"
    parent.rename(moved)
    parent.symlink_to(state.resolved_root)
    with pytest.raises(ap.SandboxBoundaryError):
        ap.prepare_response_file(config, "claude")


def test_lazily_created_checkout_and_register_checkout(tmp_path, sandbox):
    config = sandboxed_config(tmp_path, claude_dir=tmp_path / "later" / "claude", create_dirs=False)
    state = ap.establish_response_root_boundary(config)
    record = state.checkouts[str(config.claude_dir)]
    assert record.exists is False
    assert record.resolved == os.path.join(os.path.realpath(tmp_path), "later", "claude")
    config.claude_dir.mkdir(parents=True)
    ap.prepare_response_file(config, "claude")  # adopted after the overlap check
    config.claude_dir.rename(tmp_path / "later" / "claude-old")
    config.claude_dir.mkdir()
    with pytest.raises(ap.SandboxBoundaryError):
        ap.prepare_response_file(config, "claude")
    ap.register_checkout(config, config.claude_dir)  # intentional re-creation
    ap.prepare_response_file(config, "claude")


def test_preexisting_symlink_at_file_path_fails_closed(tmp_path, sandbox, monkeypatch):
    config = sandboxed_config(tmp_path)
    state = ap.establish_response_root_boundary(config)
    victim = config.claude_dir / "victim.md"
    fixed = "0" * 32
    (state.resolved_root / "claude" / f"{fixed}.md").symlink_to(victim)
    monkeypatch.setattr(ap.uuid, "uuid4", lambda: type("U", (), {"hex": fixed})())
    with pytest.raises(ap.SandboxBoundaryError, match="exclusively create"):
        ap.prepare_response_file(config, "claude")
    assert not victim.exists()


def test_system_symlink_base_is_accepted(tmp_path, monkeypatch):
    real = tmp_path / "private-tmp"
    real.mkdir()
    link = tmp_path / "tmp-link"
    link.symlink_to(real)
    monkeypatch.setattr(tempfile, "tempdir", str(link))
    ap.reset_sandbox_state()
    config = sandboxed_config(tmp_path)
    state = ap.establish_response_root_boundary(config)
    assert str(state.resolved_root).startswith(str(real))
    ap.prepare_response_file(config, "codex")
    ap.reset_sandbox_state()


def test_tmpdir_inside_repository_checkout_fails_fast(tmp_path, monkeypatch):
    """The orchestrator's repository checkout bounds the response root too."""
    repository = tmp_path / "orchestrator-repo"
    (repository / ".git").mkdir(parents=True)
    inside = repository / "tmp"
    inside.mkdir()
    monkeypatch.chdir(repository)
    monkeypatch.setattr(tempfile, "tempdir", str(inside))
    ap.reset_sandbox_state()
    config = sandboxed_config(tmp_path)  # every provider uses a separate clone
    with pytest.raises(AgentLoopError, match="overlaps repository checkout"):
        ap.establish_response_root_boundary(config)
    ap.reset_sandbox_state()


def test_repository_checkout_is_reverified_but_may_hold_the_install(tmp_path, sandbox, monkeypatch):
    repository = tmp_path / "orchestrator-repo"
    (repository / ".git").mkdir(parents=True)
    monkeypatch.chdir(repository)
    config = sandboxed_config(tmp_path)
    state = ap.establish_response_root_boundary(config)
    assert state.checkouts[str(repository)].agent is False
    # A development install inside the orchestrator's own clone is allowed.
    sandbox["establish"](config, locator=lambda _i: str(repository / "src" / "pkg"))
    repository.rename(tmp_path / "moved-repo")
    repository.symlink_to(state.resolved_root)
    with pytest.raises(ap.SandboxBoundaryError):
        ap.prepare_response_file(config, "claude")


@pytest.mark.parametrize("mode", [0o777, 0o770, 0o722])
def test_group_or_world_writable_component_is_rejected(tmp_path, sandbox, mode):
    config = sandboxed_config(tmp_path)
    existing = sandbox["tmp"] / "coding-review-agent-loop"
    existing.mkdir()
    existing.chmod(mode)
    with pytest.raises(AgentLoopError, match="group- or world-writable"):
        ap.establish_response_root_boundary(config)


def test_component_mode_widened_between_turns_blocks_spawn(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    state = ap.establish_response_root_boundary(config)
    (state.resolved_root / "claude").chmod(0o777)
    with pytest.raises(ap.SandboxBoundaryError, match="group- or world-writable"):
        ap.prepare_response_file(config, "claude")


def test_absent_checkout_ancestor_redirect_blocks_spawn(tmp_path, sandbox):
    parent = tmp_path / "parent"
    parent.mkdir()
    config = sandboxed_config(tmp_path, claude_dir=parent / "later" / "claude", create_dirs=False)
    state = ap.establish_response_root_boundary(config)
    assert state.checkouts[str(config.claude_dir)].exists is False
    # The checkout is still absent, but its ancestor now points elsewhere.
    parent.rename(tmp_path / "parent-old")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    parent.symlink_to(elsewhere)
    with pytest.raises(ap.SandboxBoundaryError, match="not yet created"):
        ap.prepare_response_file(config, "claude")


def test_mixed_provider_implementation_coder_resolves_in_its_own_checkout(tmp_path, sandbox, monkeypatch):
    seen = []

    def verified(**kwargs):
        seen.append(kwargs["cwd"])
        return WRAPPER if kwargs["cwd"] == config.claude_dir.resolve() else None

    monkeypatch.setattr(test_runtime, "verified_wrapper_prefix", verified)
    config = sandboxed_config(
        tmp_path, coder="codex", implementation_coder="claude",
        plan_execution_mode="implement-one-shot", test_command=("pytest",),
    )
    sandbox["establish"](config)
    ap.validate_sandboxed_flow(config, command="issue", plan_first=True)
    rules = _allowed_tools(list(ap.role_permission_args(config, "claude", "coder")))
    test_rules = [rule for rule in rules if "run-tests" in rule]
    assert len(test_rules) == 1
    assert seen == [config.claude_dir.resolve()]
    # The switched implementation config's prompt shows the same string.
    switched = dataclasses.replace(config, coder="claude")
    assert test_rules[0][5:-1] in _coder_workdir_guidance(switched)


def test_ephemeral_checkout_records_are_dropped(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    ap.establish_response_root_boundary(config)
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    ephemeral = dataclasses.replace(config, claude_dir=isolated, codex_dir=isolated)
    ap.prepare_response_file(ephemeral, "claude")
    shutil.rmtree(isolated)
    ap.prepare_response_file(config, "claude")


# ------------------------------------------------------ inspect provenance


def test_provenance_inside_workdir_fails_fast(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    with pytest.raises(AgentLoopError, match="Install agent-loop outside"):
        sandbox["establish"](config, locator=lambda _i: str(config.claude_dir / "pkg"))


def test_first_git_on_path_inside_checkout_fails_fast(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    planted = _executable(config.codex_dir / "bin" / "git")
    with pytest.raises(AgentLoopError, match="Remove checkout directories from PATH"):
        sandbox["establish"](config, which=lambda name: str(planted) if name == "git" else None)


def test_pinned_symlink_target_inside_checkout_fails_fast(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    real = _executable(config.claude_dir / "tools" / "git")
    link = sandbox["trusted"] / "git-link"
    link.symlink_to(real)
    with pytest.raises(AgentLoopError, match="inside agent checkout"):
        sandbox["establish"](config, which=lambda name: str(link) if name == "git" else None)


def test_checkout_dir_after_system_dir_pins_system_binaries(tmp_path, sandbox, monkeypatch):
    config = sandboxed_config(tmp_path)
    checkout_bin = config.claude_dir / "bin"
    _executable(checkout_bin / "git")
    monkeypatch.setenv("PATH", f"{sandbox['trusted']}{os.pathsep}{checkout_bin}")
    ap.establish_response_root_boundary(config)
    provenance = ap.establish_inspect_provenance(config, package_locator=lambda _i: PACKAGE_DIR)
    assert provenance.git.path == str(sandbox["tools"]["git"])


def test_no_gh_omits_gh_option_and_prompt_says_unavailable(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    provenance = sandbox["establish"](
        config, which=lambda name: str(sandbox["tools"]["git"]) if name == "git" else None
    )
    assert provenance.gh is None
    assert not any(item.startswith("--gh=") for item in provenance.prefix)
    assert "gh inspection is unavailable" in ap.inspect_forms(config)


def test_missing_git_fails_fast(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    with pytest.raises(AgentLoopError, match="requires git"):
        sandbox["establish"](config, which=lambda _name: None)


def test_unsafe_prefix_character_fails_fast(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    odd = _executable(tmp_path / "odd dir" / "git")
    with pytest.raises(AgentLoopError, match="shell-safe"):
        sandbox["establish"](config, which=lambda name: str(odd) if name == "git" else None)


@pytest.mark.parametrize("change", ["contents", "symlink", "directory", "package"])
def test_changed_provenance_blocks_next_read_only_claude_spawn(tmp_path, sandbox, monkeypatch, change):
    config = sandboxed_config(tmp_path)
    trusted = sandbox["trusted"]
    real_git = _executable(tmp_path / "real" / "git")
    link = trusted / "gitlink"
    link.symlink_to(real_git)
    fake_package = tmp_path / "installed" / "coding_review_agent_loop"
    fake_package.mkdir(parents=True)
    (fake_package / "cli.py").write_text("x = 1\n", encoding="utf-8")
    sandbox["establish"](
        config,
        which=lambda name: str(link) if name == "git" else None,
        locator=lambda _i: str(fake_package),
    )
    if change == "contents":
        real_git.write_text("#!/bin/sh\necho changed\n", encoding="utf-8")
    elif change == "symlink":
        other = _executable(tmp_path / "other" / "git")
        link.unlink()
        link.symlink_to(other)
    elif change == "directory":
        trusted.rename(tmp_path / "trusted-old")
        trusted.mkdir()
        (trusted / "gitlink").symlink_to(real_git)
    else:
        (fake_package / "cli.py").write_text("x = 2\n", encoding="utf-8")
    runner = FakeRunner(claude_outputs=[json.dumps({"result": "should not run"})])
    result = CLAUDE_BACKEND.run(runner, config, "Review.", run_id="r", role="reviewer")
    assert runner.commands == []
    unavailable = parse_agent_unavailable(result.text)
    assert unavailable is not None and unavailable.category == "environment"
    expected = "cli.py" if change == "package" else "gitlink"
    assert expected in unavailable.summary
    # A coder turn does not use the inspect provenance pin.
    if change == "package":
        coder_runner = FakeRunner(claude_outputs=[json.dumps({"result": "ok"})])
        CLAUDE_BACKEND.run(coder_runner, config, "Implement.", run_id="r", role="coder")
        assert coder_runner.commands


def test_checkout_local_shadow_package_does_not_change_isolated_resolution(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    shadow = config.claude_dir / "coding_review_agent_loop"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise SystemExit('shadow')\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-I", "-c",
         "import importlib.util;print(list(importlib.util.find_spec('coding_review_agent_loop').submodule_search_locations)[0])"],
        cwd=config.claude_dir, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0
    assert os.path.realpath(completed.stdout.strip()) == PACKAGE_DIR
    sandbox["establish"](config)
    ap.verify_inspect_provenance(config)


def test_real_package_locator_uses_isolated_interpreter(tmp_path, sandbox):
    assert ap._locate_package(sys.executable) == PACKAGE_DIR


# ------------------------------------------------------ backend wiring


def test_claude_reviewer_backend_argv_and_spawn_order(tmp_path, sandbox, monkeypatch):
    config = sandboxed_config(tmp_path)
    sandbox["establish"](config)
    events = []
    original = ap.prepare_response_file

    def spy(cfg, provider):
        events.append(("prepare", provider))
        return original(cfg, provider)

    monkeypatch.setattr(ap, "prepare_response_file", spy)
    runner = FakeRunner(claude_outputs=[json.dumps({"result": "ok"})])
    original_run = runner.run_with_log

    def run_with_log(*args, **kwargs):
        events.append(("spawn", "claude"))
        return original_run(*args, **kwargs)

    runner.run_with_log = run_with_log
    result = CLAUDE_BACKEND.run(runner, config, "Review.", run_id="r", role="reviewer")
    assert events == [("prepare", "claude"), ("spawn", "claude")]
    argv = runner.argv_commands[-1][0]
    assert "--restricted" in argv
    assert result.response_file_path.parent == ap.sandboxed_response_root(config) / "claude"


def test_codex_reviewer_backend_writes_response_via_last_message(tmp_path, sandbox, monkeypatch):
    config = sandboxed_config(tmp_path)
    sandbox["establish"](config)
    events = []
    original = ap.prepare_response_file
    monkeypatch.setattr(
        ap, "prepare_response_file", lambda cfg, provider: events.append(provider) or original(cfg, provider)
    )
    runner = FakeRunner(codex_outputs=[{"public_response": "REVIEW TEXT", "stdout": "{}", "returncode": 0}])
    result = CODEX_BACKEND.run(runner, config, "Review.", run_id="r", role="reviewer")
    argv = runner.argv_commands[-1][0]
    assert events == ["codex"]
    target = argv[argv.index("--output-last-message") + 1]
    assert Path(target).parent == ap.sandboxed_response_root(config) / "codex"
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert 'approval_policy="never"' in argv
    for forbidden in ("workspace-write", "--add-dir", "--dangerously-bypass-approvals-and-sandbox"):
        assert forbidden not in argv
    assert not any("network_access" in item for item in argv)
    assert result.response_file_text == "REVIEW TEXT"
    assert result.text_source == "response_file"
    assert Path(target).exists()  # the validated file is kept, not unlinked
    assert "Your final message is written verbatim" in runner.last_input_text
    assert "PUBLIC RESPONSE FILE" not in runner.last_input_text


def test_codex_dry_run_sandboxed_argv(tmp_path, sandbox):
    config = sandboxed_config(tmp_path, dry_run=True)
    sandbox["establish"](config)
    runner = FakeRunner(codex_outputs=[("ok", 0)])
    CODEX_BACKEND.run(runner, config, "Plan.", run_id="r", role=None)
    argv = runner.argv_commands[-1][0]
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert Path(argv[argv.index("--output-last-message") + 1]).parent.name == "codex"


def test_empty_precreated_file_is_no_response(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    sandbox["establish"](config)
    runner = FakeRunner(codex_outputs=[{"public_response": "", "stdout": "", "returncode": None}])
    result = CODEX_BACKEND.run(runner, config, "Review.", run_id="r", role="reviewer")
    assert result.response_file_text is None


def test_run_agent_result_refuses_unsupported_agent(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    runner = FakeRunner(gemini_outputs=[("x", 0)])
    with pytest.raises(AgentLoopError, match="refuses to run 'gemini'"):
        run_agent_result(runner, agent="gemini", config=config, prompt="p", role="reviewer")
    assert runner.commands == []


# ------------------------------------------------------ selections


@pytest.mark.parametrize(
    "overrides,fragment",
    [
        ({"coder": "gemini"}, "--coder"),
        ({"implementation_coder": "antigravity"}, "--implementation-coder"),
        ({"reviewer": ("claude", "gemini")}, "--reviewer"),
        ({"discuss_analyzer": "gemini"}, "--discuss-analyzer"),
        ({"repair_backend": "antigravity", "repair_models": ("m",)}, "--repair-backend claude|codex --repair-model MODEL"),
        ({"semantic_followup_backend": "gemini"}, "--no-semantic-followup-dedupe"),
    ],
)
def test_unsupported_selection_is_rejected(tmp_path, overrides, fragment):
    with pytest.raises(AgentLoopError, match="supports only Claude and Codex") as info:
        sandboxed_config(tmp_path, **overrides)
    assert fragment in str(info.value)


@pytest.mark.parametrize("field_name", ["primary_reviewer", "primary_plan_reviewer"])
def test_primary_reviewer_selections_are_rejected(field_name):
    from types import SimpleNamespace

    values = dict(
        coder="claude", implementation_coder=None, reviewer=("claude",),
        primary_reviewer=None, primary_plan_reviewer=None, discuss_analyzer=None,
        repair_backend="claude", semantic_followup_dedupe=False, semantic_followup_backend="gemini",
    )
    values[field_name] = "gemini"
    with pytest.raises(AgentLoopError, match=field_name):
        ap.validate_sandboxed_selections(SimpleNamespace(**values))


def test_default_semantic_backend_allowed_when_dedupe_disabled(tmp_path):
    config = sandboxed_config(tmp_path, semantic_followup_backend="gemini", semantic_followup_dedupe=False)
    assert config.agent_permissions == "sandboxed"


def test_configured_agent_selections_covers_every_agent_typed_field():
    hints = typing.get_type_hints(AgentLoopConfig)
    agent_fields = {
        field.name
        for field in dataclasses.fields(AgentLoopConfig)
        if "AgentName" in repr(hints[field.name]) or "Literal['claude'" in repr(hints[field.name])
    }
    # auto_agent_dirs records which directories were defaulted; it selects no agent.
    agent_fields -= {"auto_agent_dirs"}
    agent_fields |= {"repair_backend"}  # a str-typed agent selection
    covered = {
        "coder", "implementation_coder", "reviewer", "primary_reviewer",
        "primary_plan_reviewer", "discuss_analyzer", "repair_backend", "semantic_followup_backend",
    }
    assert agent_fields == covered
    source = Path(ap.__file__).read_text(encoding="utf-8")
    for name in covered:
        assert f'"{name}"' in source


@pytest.mark.parametrize(
    "command,plan_first,mode,coder,implementation_coder,rejected",
    [
        ("issue", False, "plan-only", "codex", None, True),
        ("issue", True, "auto", "codex", None, True),
        ("issue", True, "implement-one-shot", "claude", "codex", True),
        ("issue", True, "implement-by-phase", "codex", None, True),
        ("pr", False, "plan-only", "codex", None, True),
        ("managed-pr", False, "plan-only", "codex", None, True),
        ("task", False, "plan-only", "codex", None, True),
        ("issue", True, "plan-only", "codex", None, False),
        ("issue", True, "decompose-only", "codex", None, False),
        ("issue", True, "implement-one-shot", "codex", "claude", False),
        ("discuss", False, "plan-only", "codex", None, False),
    ],
)
def test_codex_committing_coder_rejected_per_flow(
    tmp_path, command, plan_first, mode, coder, implementation_coder, rejected
):
    config = sandboxed_config(
        tmp_path, coder=coder, implementation_coder=implementation_coder, plan_execution_mode=mode
    )
    if rejected:
        with pytest.raises(AgentLoopError, match="read-only"):
            ap.validate_sandboxed_flow(config, command=command, plan_first=plan_first)
    else:
        ap.validate_sandboxed_flow(config, command=command, plan_first=plan_first)


def test_prompts_unchanged_outside_sandbox(tmp_path):
    config = make_config(tmp_path)
    assert "Sandboxed permissions" not in _coder_workdir_guidance(config)
    assert "inspect" not in _coder_workdir_guidance(config, implementation=False, agent="claude")


def test_read_only_roles_deny_bare_git_and_gh_because_allow_lists_cannot(tmp_path, sandbox):
    """Claude auto-approves some read-only commands whatever --allowedTools says.

    ``git status`` succeeded for a reviewer in the live suite at 85988b4, which
    would run a planted ``.git/config`` in the shared checkout, so the grant has
    to deny the programs outright rather than merely omit them.
    """
    config = sandboxed_config(tmp_path)
    sandbox["establish"](config)
    argv = list(ap.role_permission_args(config, "claude", "reviewer"))
    denied = _disallowed_tools(argv)
    for program in ("git", "gh"):
        assert f"Bash({program})" in denied
        assert f"Bash({program} *)" in denied
        assert f"Bash({program}:*)" in denied
    # The deny section must not swallow the inspect grant that replaces them.
    prefix = ap.inspect_prefix(config)
    assert f"Bash({prefix} *)" in _allowed_tools(argv)
    assert not any(rule.startswith(f"Bash({prefix}") for rule in denied)
    # Deny is declared before allow so neither variadic option absorbs the other.
    assert argv.index("--disallowedTools") < argv.index("--allowedTools")


def test_coder_keeps_git_and_gh(tmp_path, sandbox):
    """The deny rules are read-only-role scoped; a coder still commits and pushes."""
    config = sandboxed_config(tmp_path)
    sandbox["establish"](config)
    argv = list(ap.role_permission_args(config, "claude", "coder"))
    assert "Bash(git *)" in _allowed_tools(argv)
    assert "--disallowedTools" not in argv


@pytest.mark.parametrize("role", [None, "reviewer", "coder", "planner"])
def test_sandboxed_env_neutralizes_inherited_git_tracing_and_config_injection(
    tmp_path, sandbox, role
):
    """GIT_TRACE2_EVENT reached the CLI itself and wrote trace2.json at 85988b4.

    ``inspect`` has an environment allowlist; the CLI that calls it does not, so
    the variables are overridden on the agent process for every role.
    """
    config = sandboxed_config(tmp_path)
    sandbox["establish"](config)
    env = ap.role_permission_env(config, "claude", role)
    for name in ap.GIT_TRACE_VARIABLES:
        assert env[name] == "0"
    assert "GIT_TRACE2_EVENT" in ap.GIT_TRACE_VARIABLES
    assert env["GIT_CONFIG_COUNT"] == "0"
    assert env["GIT_EXTERNAL_DIFF"] == ""


def test_unsandboxed_env_is_untouched(tmp_path):
    config = sandboxed_config(tmp_path, agent_permissions="default")
    assert ap.role_permission_env(config, "claude", "reviewer") == {}
    assert ap.role_permission_env(config, "codex", "coder") == {}


def test_establish_neutralizes_git_env_in_agent_loop_own_process(tmp_path, sandbox, monkeypatch):
    """agent-loop runs git in-process; scrubbing only the child was not enough.

    At 85988b4 an inherited GIT_TRACE2_EVENT survived into agent-loop's own
    ``git rev-parse HEAD`` and wrote trace2.json into the checkout; the trace
    named ``python`` as its parent process, not the agent CLI.
    """
    trace = tmp_path / "trace2.json"
    monkeypatch.setenv("GIT_TRACE2_EVENT", str(trace))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", "/bin/false")
    config = sandboxed_config(tmp_path)
    # The full startup path, not the fixture's provenance-only helper: the
    # neutralization belongs to establishing the run, before any git call.
    ap.establish_sandboxed_run(config, command="pr")
    assert os.environ["GIT_TRACE2_EVENT"] == "0"
    assert os.environ["GIT_CONFIG_COUNT"] == "0"
    assert os.environ["GIT_EXTERNAL_DIFF"] == ""


def test_default_mode_leaves_the_process_environment_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_TRACE2_EVENT", "/tmp/keep-me.json")
    config = sandboxed_config(tmp_path, agent_permissions="default")
    ap.establish_sandboxed_run(config, command="pr")
    assert os.environ["GIT_TRACE2_EVENT"] == "/tmp/keep-me.json"


# ------------------------------------------- orchestrator's own git probes


def _real_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(repo.parent),
             "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"},
    )


def _plant_fsmonitor_checkout(checkout: Path, marker: Path) -> None:
    _real_git(checkout, "init", "-q")
    _real_git(checkout, "config", "user.name", "Test")
    _real_git(checkout, "config", "user.email", "t@example.com")
    (checkout / "a.txt").write_text("one\n", encoding="utf-8")
    _real_git(checkout, "add", "a.txt")
    _real_git(checkout, "commit", "-q", "-m", "first")
    hook = _executable(checkout.parent / "hostile" / "fsmonitor.sh", f"#!/bin/sh\ntouch {marker}\n")
    # What a coder turn can plant with its own `Bash(git *)` grant.
    _real_git(checkout, "config", "core.fsmonitor", str(hook))


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_planted_fsmonitor_runs_under_bare_git_status(tmp_path):
    # Sanity check for the test below: the plant is live for an unhardened probe.
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    marker = tmp_path / "MARKER"
    _plant_fsmonitor_checkout(checkout, marker)
    subprocess.run(["git", "status", "--porcelain"], cwd=checkout, capture_output=True, check=False)
    assert marker.exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_sandboxed_claude_workdir_snapshot_never_runs_planted_git_config(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    marker = tmp_path / "MARKER"
    _plant_fsmonitor_checkout(config.claude_dir, marker)
    real_git = shutil.which("git")
    sandbox["establish"](
        config, which=lambda name: real_git if name == "git" else str(sandbox["tools"][name])
    )
    runner = FakeRunner(claude_outputs=[json.dumps({"result": "ok"})])
    CLAUDE_BACKEND.run(runner, config, "Review.", run_id="r", role="reviewer")
    # The pre-spawn snapshot went through the config gate, not the runner's bare git.
    assert not [cmd for cmd, *_ in runner.commands if cmd and cmd[0] == "git"]
    assert not marker.exists()
    with pytest.raises(AgentLoopError, match="core.fsmonitor"):
        ap.hardened_git_probe_runner(config)(("status", "--porcelain"), config.claude_dir)
    assert not marker.exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_hardened_probe_reads_clean_checkout(tmp_path, sandbox):
    config = sandboxed_config(tmp_path)
    checkout = config.claude_dir
    _real_git(checkout, "init", "-q")
    _real_git(checkout, "config", "user.name", "Test")
    _real_git(checkout, "config", "user.email", "t@example.com")
    (checkout / "a.txt").write_text("one\n", encoding="utf-8")
    _real_git(checkout, "add", "a.txt")
    _real_git(checkout, "commit", "-q", "-m", "first")
    real_git = shutil.which("git")
    sandbox["establish"](
        config, which=lambda name: real_git if name == "git" else str(sandbox["tools"][name])
    )
    probe = ap.hardened_git_probe_runner(config)
    head = probe(("rev-parse", "HEAD"), checkout)
    assert head.returncode == 0 and len(head.stdout.strip()) == 40
    (checkout / "a.txt").write_text("two\n", encoding="utf-8")
    status = probe(("status", "--porcelain"), checkout)
    assert status.returncode == 0 and "a.txt" in status.stdout
