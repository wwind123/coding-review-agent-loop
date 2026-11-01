"""Seat identity, unpaid catalog admission and unchanged reviewer boundaries."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop import review_images_capabilities as capabilities
from coding_review_agent_loop.agents.registry import run_agent_result
from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent
from coding_review_agent_loop.runner import CommandResult
from test_review_images_transport import inputs, claude_events, encode


def setup_config(tmp_path, monkeypatch, provider="claude", *, named=False, mode="default"):
    model = "claude-sonnet-4-6" if provider == "claude" else "gpt-6.1-sol"
    executable = tmp_path / "executable"
    executable.write_text("fixture executable")
    executable.chmod(0o700)
    secondary = "codex" if provider == "claude" else "claude"
    config = make_config(tmp_path, reviewer=(provider, secondary), primary_reviewer=provider,
                         pr_review_policy="primary-then-panel", repair_backend="codex", repair_models=("gpt-6.1-sol",),
                         semantic_followup_backend="codex",
                         agent_permissions=mode,
                         **{f"{provider}_cmd": str(executable), f"{provider}_model": "agent-wide-conflict",
                            f"reviewer_{provider}_model": model,
                            "containment_coder_memory_max": "4G", "containment_reviewer_memory_max": "2G"})
    if named:
        agent = SeatAgent(ReviewerSeat("visual", provider, (model,), "high"), getattr(config, f"{provider}_dir"))
        config = replace(config, reviewer=(agent, secondary), primary_reviewer=agent)
    else:
        agent = provider
    version = "2.1.316 (Claude Code)" if provider == "claude" else "codex-cli 0.166.0"
    checksum = capabilities.EXECUTABLE_DIGESTS["claude-stream-v1" if provider == "claude" else "codex-native-v1"]
    monkeypatch.setattr(capabilities, "executable_identity", lambda _: ((1, 2, 3, 4), checksum))
    calls = []
    def probe(args, **kwargs):
        calls.append((args, kwargs))
        assert args == [str(executable), "--version"]
        return SimpleNamespace(stdout=(version + "\n").encode(), stderr=b"")
    monkeypatch.setattr(capabilities.subprocess, "run", probe)
    return config, agent, calls


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("mode", ["default", "sandboxed"])
def test_exact_seat_model_effort_permissions_and_reviewer_containment(tmp_path, monkeypatch, provider, named, mode):
    from coding_review_agent_loop import agent_permissions, checkout_verification, worktree_pr_env
    from coding_review_agent_loop import review_images_adapter as adapter
    from coding_review_agent_loop.review_images_transport import acquire_images
    config, agent, probes = setup_config(tmp_path, monkeypatch, provider, named=named, mode=mode)
    capability = capabilities.preflight_images(config)
    snapshot, unused = inputs(provider)
    unused.close()
    images = acquire_images(snapshot, capability, head_sha=snapshot.head_sha,
                            config_digest=snapshot.config_digest, snapshot_digest=snapshot.digest)
    snapshot.close()
    expected = capability.invocation_view
    seen = []
    monkeypatch.setattr(checkout_verification, "verify_checkout", lambda *a, **k: None)
    monkeypatch.setattr(checkout_verification, "record_checkout_baseline", lambda *a, **k: None)
    monkeypatch.setattr(worktree_pr_env, "refuse_linked_turn", lambda *a, **k: None)
    response = tmp_path / "response.md"
    response.touch()
    monkeypatch.setattr(adapter, "public_response_path", lambda *a: response)
    monkeypatch.setattr(adapter, "prepare_sandboxed_spawn", lambda c, p, r: response)
    def permission_args(c, p, r, *, response_dir):
        seen.append((p, r, response_dir))
        return ("--narrow-role-grant",)
    monkeypatch.setattr(adapter, "role_permission_args", permission_args)
    monkeypatch.setattr(adapter, "role_permission_env", lambda *a: {})
    monkeypatch.setattr(adapter, "executable_identity", capabilities.executable_identity)
    class CaptureRunner(FakeRunner):
        def configure_from_config(self, c):
            self.captured_config = c
        def set_containment_role(self, r):
            self.captured_role = r
        def run_with_log(self, args, **kwargs):
            self.actual = args, kwargs
            assert self.captured_role == kwargs["containment_role"] == "reviewer"
            assert self.captured_config.containment_reviewer_memory_max == "2G"
            assert kwargs["cwd"] == getattr(expected, f"{provider}_dir")
            assert args[args.index("--model") + 1] == capability.model
            assert capability.effort == ("high" if named else "medium")
            if provider == "claude":
                assert args[args.index("--effort") + 1] == capability.effort
                message = json.loads(kwargs["input_text"])
                assert len(message["message"]["content"]) == 1 + len(images.entries)
                raw = encode(claude_events(images))
            else:
                assert args[-2:] == ["--", "-"]
                assert [arg for arg in args if arg.startswith("--image=")] == [f"--image={path}" for path in images.paths()]
                assert f'model_reasoning_effort="{capability.effort}"' in args
                raw = "clean text without evidence"
            assert "base64" not in kwargs["stdin_log_summary"]
            return CommandResult(args, kwargs["cwd"], raw, "", 0)
    runner = CaptureRunner()
    result = run_agent_result(runner, agent=agent, config=config, prompt="visible defects",
                              role="reviewer", timeout_seconds=30, image_inputs=images)
    assert images._closed
    assert bool(result.image_delivery_proof) is (provider == "claude")
    assert seen == ([(provider, "reviewer", tmp_path)] if mode == "sandboxed" else [])
    assert all(call[0][-1] == "--version" for call in probes)


@pytest.mark.parametrize("fault", ["primary", "model", "alias", "fallback", "args", "binary", "version", "provider"])
def test_unverified_contracts_refuse_without_model_call(tmp_path, monkeypatch, fault):
    config, agent, calls = setup_config(tmp_path, monkeypatch)
    if fault == "primary": config = replace(config, primary_reviewer=None, pr_review_policy="all-reviewers")
    elif fault == "model": config = replace(config, reviewer_claude_model=None, claude_model=None)
    elif fault == "alias": config = replace(config, reviewer_claude_model="sonnet")
    elif fault == "fallback":
        seat = SeatAgent(ReviewerSeat("visual", "claude", ("claude-sonnet-4-6", "unknown")), config.claude_dir)
        config = replace(config, reviewer=(seat, "codex"), primary_reviewer=seat)
    elif fault == "args": config = replace(config, claude_args=("--verbose",))
    elif fault == "binary": monkeypatch.setattr(capabilities, "executable_identity", lambda _: ((1, 2, 3, 4), "not-admitted"))
    elif fault == "version": monkeypatch.setattr(capabilities.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=b"new-release", stderr=b""))
    elif fault == "provider": config = replace(config, reviewer=("gemini", "codex"), primary_reviewer="gemini")
    with pytest.raises(ValueError):
        capabilities.preflight_images(config)
    assert all(call[0][-1] == "--version" for call in calls)


def test_capability_drift_closed_input_wrong_seat_and_role_refuse(tmp_path, monkeypatch):
    from coding_review_agent_loop.review_images_transport import acquire_images
    config, agent, _ = setup_config(tmp_path, monkeypatch)
    cap = capabilities.preflight_images(config)
    snapshot, unused = inputs()
    unused.close()
    images = acquire_images(snapshot, cap, head_sha=snapshot.head_sha,
                            config_digest=snapshot.config_digest, snapshot_digest=snapshot.digest)
    try:
        for role, session in (("coder", None), ("reviewer", "resume")):
            with pytest.raises(ValueError):
                capabilities.require_image_invocation(images, config, agent, role=role, session_id=session)
        with pytest.raises(ValueError):
            capabilities.require_image_invocation(images, replace(config, reviewer_claude_effort="high"), agent,
                                                  role="reviewer", session_id=None)
        with pytest.raises(ValueError):
            capabilities.require_image_invocation(images, config, "absent", role="reviewer", session_id=None)
        images.close()
        with pytest.raises(ValueError):
            capabilities.require_image_invocation(images, config, agent, role="reviewer", session_id=None)
    finally:
        images.close(); snapshot.close()


@pytest.mark.parametrize("raw_kind", ["malformed", "partial", "replayed-transient"])
def test_adapter_failure_never_exposes_replay_or_images(tmp_path, monkeypatch, raw_kind):
    from coding_review_agent_loop import review_images_adapter as adapter
    from coding_review_agent_loop.review_images_transport import acquire_images
    config, agent, _ = setup_config(tmp_path, monkeypatch)
    cap = capabilities.preflight_images(config)
    snapshot, unused = inputs()
    unused.close()
    images = acquire_images(snapshot, cap, head_sha=snapshot.head_sha,
                            config_digest=snapshot.config_digest, snapshot_digest=snapshot.digest)
    snapshot.close()
    events = claude_events(images)
    if raw_kind == "partial": events[1]["message"]["content"].pop()
    raw = encode(events)
    if raw_kind == "malformed": raw += "\nmalformed base64 overload retry"
    elif raw_kind == "replayed-transient": events.pop(2); raw = encode(events)
    response = tmp_path / "response.md"
    monkeypatch.setattr(adapter, "public_response_path", lambda *a: response)
    monkeypatch.setattr(adapter, "role_permission_env", lambda *a: {})
    monkeypatch.setattr(adapter, "executable_identity", capabilities.executable_identity)
    class Scripted:
        def run_with_log(self, args, **kwargs):
            return CommandResult(args, kwargs["cwd"], raw, "provider overload from replay", 1)
    result = adapter.run_image_adapter(Scripted(), cap.invocation_view, "schema example", images,
                                       provider="claude", role="reviewer", session_id=None,
                                       run_id=None, label=None, timeout_seconds=30, attempt_suffix=None)
    assert result.image_delivery_proof is None and images._closed
    assert result.raw_output == "" and result.message_text is None
    assert result.command_result is None and result.response_file_text is None
    assert "base64" not in repr(result) and "overload" not in repr(result)


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_text_only_model_mismatch_warns_and_retains_completed_turn(tmp_path, monkeypatch, provider):
    from coding_review_agent_loop.agents import registry
    from coding_review_agent_loop.agents.base import AgentResult

    config, agent, probes = setup_config(tmp_path, monkeypatch, provider)
    messages = []
    calls = []

    def acquire(*args, **kwargs):
        calls.append(kwargs)
        assert "image_inputs" not in kwargs
        return AgentResult(text="completed text review", observed_model="different-model")

    backend = SimpleNamespace(workdir=lambda c: getattr(c, f"{provider}_dir"), run=acquire)
    monkeypatch.setattr(registry, "get_backend", lambda _: backend)
    monkeypatch.setattr(registry, "log", lambda c, message: messages.append(message))
    runner = FakeRunner()
    runner.dry_run = True  # The backend is a stub; no checkout or CLI is launched.
    result = run_agent_result(runner, agent=agent, config=config, prompt="text review", role="reviewer")

    assert len(calls) == 1 and not probes
    assert result.text == "completed text review" and result.returncode == 0
    assert result.observed_model == "different-model" and result.image_delivery_proof is None
    assert any("observed model" in message and "retaining the completed turn" in message
               for message in messages)
