"""Backend tests extracted from test_agent_loop.py (lines 1302–2015).

Tests for Claude, Gemini, and Codex backend output parsing and normalization.
"""
import json
from pathlib import Path

import pytest

from coding_review_agent_loop.agents.claude import (
    BACKEND as CLAUDE_BACKEND,
    classify_self_update_interruption,
    _normalize_claude_usage,
    _parse_claude_output,
)
from coding_review_agent_loop.runner import CommandResult, ExecutableIdentity, ExecutionObservation
from coding_review_agent_loop.agents.codex import (
    BACKEND as CODEX_BACKEND,
    classify_executable_replacement_interruption,
    _extract_codex_usage,
    _normalize_codex_usage,
)
from coding_review_agent_loop.agents.gemini import (
    BACKEND as GEMINI_BACKEND,
    PUBLIC_RESPONSE_MARKER,
    _OVERSIZED_PROMPT_DIRECTIVE,
    _normalize_gemini_usage,
    _parse_gemini_payload,
    classify_gemini_executable_replacement_interruption,
)
from coding_review_agent_loop.agents.antigravity import (
    classify_antigravity_executable_replacement_interruption,
)

from agent_loop_helpers import (
    FakeRunner,
    make_config,
)


def _write_codex_rollout(
    codex_home: Path,
    thread_id: str,
    records: list[object],
    *,
    name_prefix: str = "rollout-2026-06-18T12-00-00",
) -> Path:
    rollout_path = (
        codex_home
        / "sessions"
        / "2026"
        / "06"
        / "18"
        / f"{name_prefix}-{thread_id}.jsonl"
    )
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_path.write_text(
        "\n".join(
            record if isinstance(record, str) else json.dumps(record)
            for record in records
        ),
        encoding="utf-8",
    )
    return rollout_path

def test_parse_claude_output_extracts_text_and_session_id():
    raw = json.dumps({"result": "Hello.", "session_id": "abc123"})
    text, sid, usage, raw_usage, model = _parse_claude_output(raw)
    assert text == "Hello."
    assert sid == "abc123"
    assert usage is None
    assert raw_usage is None
    assert model is None


def test_parse_claude_output_falls_back_on_plain_text():
    raw = "plain response"
    text, sid, usage, raw_usage, model = _parse_claude_output(raw)
    assert text == "plain response"
    assert sid is None
    assert usage is None
    assert raw_usage is None
    assert model is None


def test_claude_self_update_classification_requires_bounded_evidence():
    identity = ExecutableIdentity("/bin/claude", "/bin/claude", (1, 1, 1), (1, 1, 1))
    changed = ExecutableIdentity("/bin/claude", "/bin/claude", (1, 2, 2_000_000_000), (1, 2, 2_000_000_000))
    observation = ExecutionObservation(2, 10, 11, 1, identity, changed, False)
    failed = CommandResult(["claude"], Path.cwd(), "", "", 1, observation)
    assert classify_self_update_interruption(
        failed, command="claude", response_file_text=None, session_id=None
    ).reason == "Claude executable changed during invocation"
    ordinary = CommandResult(["claude"], Path.cwd(), "ordinary failure", "", 1,
                             ExecutionObservation(2, 10, 11, 1, identity, identity, False))
    assert classify_self_update_interruption(
        ordinary, command="claude", response_file_text=None, session_id=None
    ) is None
    diagnostic = CommandResult(["claude"], Path.cwd(), "Loading...\nInstalling Claude Code v2.1.226", "", 1,
                               ExecutionObservation(2, 10, 11, 1, identity, identity, False))
    assert classify_self_update_interruption(
        diagnostic, command="claude", response_file_text=None, session_id=None
    ).reason == "Claude Code updater diagnostic"


def test_claude_self_update_workdir_gate_reports_changed_status_without_replacing_evidence():
    from coding_review_agent_loop.workdir_guard import capture_workdir_snapshot

    identity = ExecutableIdentity("/bin/claude", "/bin/claude", (1, 1, 1), (1, 1, 1))
    observation = ExecutionObservation(2, 10, 11, 1, identity, identity, False)
    result = CommandResult(
        ["claude"], Path.cwd(), "Loading...\nInstalling Claude Code v2.1.226", "", 1, observation
    )
    before_runner = FakeRunner(
        git_probe_results=[
            {"stdout": "abc123\n", "returncode": 0},
            {"stdout": "", "returncode": 0},
        ]
    )
    after_runner = FakeRunner(
        git_probe_results=[
            {"stdout": "abc123\n", "returncode": 0},
            {"stdout": " M changed.py\n", "returncode": 0},
        ]
    )
    before = capture_workdir_snapshot(before_runner, Path.cwd())
    after = capture_workdir_snapshot(after_runner, Path.cwd())

    evidence = classify_self_update_interruption(
        result,
        command="claude",
        response_file_text=None,
        session_id=None,
        before_snapshot=before,
        after_snapshot=after,
    )

    assert evidence is not None
    assert evidence.reason == "Claude Code updater diagnostic"
    assert evidence.replay_refusal_kind == "changed-status"
    assert "dirty workdir" in evidence.replay_refusal_detail


def test_claude_backend_probes_before_every_call_but_after_only_for_candidates(tmp_path):
    from coding_review_agent_loop.agents.claude import ClaudeBackend

    identity = ExecutableIdentity("claude", "claude", (1, 1, 1), (1, 1, 1))
    observation = ExecutionObservation(2, 10, 11, 1, identity, identity, False)

    class ObservedRunner(FakeRunner):
        def run_with_log(self, *args, **kwargs):
            result = super().run_with_log(*args, **kwargs)
            return CommandResult(
                result.args,
                result.cwd,
                result.stdout,
                result.stderr,
                result.returncode,
                observation,
            )

    candidate_runner = ObservedRunner(
        claude_outputs=[("fatal: auto-update in progress", 1)],
        git_probe_results=[
            {"stdout": "abc123\n", "returncode": 0},
            {"stdout": "", "returncode": 0},
            {"stdout": "abc123\n", "returncode": 0},
            {"stdout": " M changed.py\n", "returncode": 0},
        ],
    )
    candidate = ClaudeBackend().run(candidate_runner, make_config(tmp_path), "Review")
    assert candidate.self_update_reason == "Claude Code updater diagnostic"
    assert candidate.self_update_replay_refusal_kind == "changed-status"
    assert candidate_runner.git_probe_calls == [
        ("git", "rev-parse", "HEAD"),
        ("git", "status", "--porcelain"),
        ("git", "rev-parse", "HEAD"),
        ("git", "status", "--porcelain"),
    ]

    ordinary_runner = ObservedRunner(
        claude_outputs=[("ordinary failure", 1)],
        git_probe_results=[
            {"stdout": "abc123\n", "returncode": 0},
            {"stdout": "", "returncode": 0},
        ],
    )
    ordinary = ClaudeBackend().run(ordinary_runner, make_config(tmp_path), "Review")
    assert ordinary.self_update_reason is None
    assert ordinary_runner.git_probe_calls == [
        ("git", "rev-parse", "HEAD"),
        ("git", "status", "--porcelain"),
    ]


def _codex_result(
    *,
    stdout="",
    returncode=1,
    before=None,
    after=None,
    interrupted=False,
    capture_diagnostics=(),
    elapsed_seconds=1,
):
    before = before or ExecutableIdentity(
        "/tmp/codex", "/tmp/codex-target", (1, 1, 100_000_000_000), (1, 2, 100_000_000_000)
    )
    after = after or ExecutableIdentity(
        "/tmp/codex", "/tmp/codex-target", (1, 3, 101_000_000_000), (1, 4, 101_000_000_000)
    )
    observation = ExecutionObservation(
        100,
        100,
        100 + elapsed_seconds,
        elapsed_seconds,
        before,
        after,
        interrupted,
    )
    return CommandResult(
        ["codex", "exec", "--json"],
        Path.cwd(),
        stdout,
        "",
        returncode,
        observation,
        capture_diagnostics,
    )


def _replacement_result(*, command, stdout="", returncode=1, changed=True, **kwargs):
    before = ExecutableIdentity(
        command, f"{command}-target", (1, 1, 90_000_000_000), (1, 2, 90_000_000_000)
    )
    after = (
        ExecutableIdentity(
            command, f"{command}-new-target", (1, 3, 100_000_000_000),
            (1, 4, 100_000_000_000),
        )
        if changed else before
    )
    observation = ExecutionObservation(100, 100, 101, 1, before, after, kwargs.pop("interrupted", False))
    return CommandResult(
        [command], Path.cwd(), stdout, "", returncode, observation,
        kwargs.pop("capture_diagnostics", ()),
    )


def _snapshot(head="abc123", status=""):
    from coding_review_agent_loop.workdir_guard import GitProbeResult, WorkdirSnapshot

    return WorkdirSnapshot(
        GitProbeResult(("git", "rev-parse", "HEAD"), "available", head),
        GitProbeResult(("git", "status", "--porcelain"), "available", status),
    )


def test_gemini_replacement_accepts_realistic_node_loader_failure_and_snapshot_gate():
    output = (
        "Gemini CLI v1.2.3\n"
        "Using model: gemini-3.5-flash\n"
        "node:internal/modules/cjs/loader:1228\n"
        "throw err;\n"
        "Error: Cannot find module '/opt/gemini/launcher.js'\n"
        "code: 'MODULE_NOT_FOUND'\n"
    )
    result = _replacement_result(command="gemini", stdout=output)

    evidence = classify_gemini_executable_replacement_interruption(
        result, command="gemini", response_file_text=None,
    )
    assert evidence is not None
    assert evidence.reason == "Gemini executable changed during invocation"

    accepted = classify_gemini_executable_replacement_interruption(
        result,
        command="gemini",
        response_file_text=None,
        before_snapshot=_snapshot(),
        after_snapshot=_snapshot(),
    )
    assert accepted is not None
    assert accepted.replay_refusal_kind is None

    refused = classify_gemini_executable_replacement_interruption(
        result,
        command="gemini",
        response_file_text=None,
        before_snapshot=_snapshot(),
        after_snapshot=_snapshot(status=" M changed.py\n"),
    )
    assert refused is not None
    assert refused.replay_refusal_kind == "changed-status"


@pytest.mark.parametrize(
    "output",
    [
        "I will inspect the repository before responding.",
        f"{PUBLIC_RESPONSE_MARKER}\nSTATE: approved",
        json.dumps({"response": "STATE: approved", "session_id": "s"}),
    ],
)
def test_gemini_replacement_rejects_progress_public_payload_and_artifacts(output):
    result = _replacement_result(command="gemini", stdout=output)
    assert classify_gemini_executable_replacement_interruption(
        result, command="gemini", response_file_text=None,
    ) is None
    assert classify_gemini_executable_replacement_interruption(
        result, command="gemini", response_file_text="malformed but present",
    ) is None


@pytest.mark.parametrize("returncode", [0, 1])
def test_antigravity_replacement_accepts_startup_chrome_and_loader_failure(returncode):
    output = (
        "Antigravity CLI v0.8.0\n"
        "Model: Gemini 3.1 Pro (High)\n"
        "Error: Cannot find module '/opt/agy/launcher.js'\n"
        "code: 'MODULE_NOT_FOUND'\n"
    )
    result = _replacement_result(command="agy", stdout=output, returncode=returncode)
    evidence = classify_antigravity_executable_replacement_interruption(
        result,
        command="agy",
        response_file_text=None,
        before_snapshot=_snapshot(),
        after_snapshot=_snapshot(),
    )
    assert evidence is not None
    assert evidence.reason == "Antigravity executable changed during invocation"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interrupted": True},
        {"capture_diagnostics": ("capture_read_failed:OSError: injected",)},
        {"returncode": None},
    ],
)
def test_antigravity_replacement_rejects_unhealthy_capture(kwargs):
    result = _replacement_result(command="agy", stdout="", **kwargs)
    assert classify_antigravity_executable_replacement_interruption(
        result, command="agy", response_file_text=None,
    ) is None


@pytest.mark.parametrize("returncode", [0, 1])
def test_codex_executable_replacement_classifies_outputless_exit(returncode):
    result = _codex_result(returncode=returncode)

    assert classify_executable_replacement_interruption(
        result,
        command="codex",
        response_file_text=None,
        last_message_artifact=None,
    ) == "Codex executable changed during invocation"


def test_codex_executable_replacement_has_no_elapsed_cap():
    result = _codex_result(elapsed_seconds=31)

    assert classify_executable_replacement_interruption(
        result,
        command="codex",
        response_file_text=None,
        last_message_artifact=None,
    ) == "Codex executable changed during invocation"


def test_codex_executable_replacement_accepts_one_setup_event_only():
    result = _codex_result(
        stdout=json.dumps({"type": "thread.started", "thread_id": "thread"})
    )

    assert classify_executable_replacement_interruption(
        result,
        command="codex",
        response_file_text=None,
        last_message_artifact=None,
    ) == "Codex executable changed during invocation"


@pytest.mark.parametrize(
    "stdout",
    [
        json.dumps({"type": "item.started"}),
        json.dumps({"type": "turn.completed"}),
        "\n".join([json.dumps({"type": "thread.started"}), json.dumps({"type": "item.started"})]),
    ],
)
def test_codex_executable_replacement_rejects_work_progress_events(stdout):
    result = _codex_result(stdout=stdout)

    assert classify_executable_replacement_interruption(
        result,
        command="codex",
        response_file_text=None,
        last_message_artifact=None,
    ) is None


def test_codex_executable_replacement_bare_command_observes_path_entry_change():
    before = ExecutableIdentity(
        "/one/codex", "/one/target", (1, 1, 100_000_000_000), (1, 2, 100_000_000_000)
    )
    after = ExecutableIdentity(
        "/two/codex", "/two/target", (1, 3, 101_000_000_000), (1, 4, 101_000_000_000)
    )
    result = _codex_result(before=before, after=after)

    assert classify_executable_replacement_interruption(
        result,
        command="codex",
        response_file_text=None,
        last_message_artifact=None,
    ) == "Codex executable changed during invocation"


def test_codex_executable_replacement_absolute_override_requires_exact_identity_change():
    command = "/configured/codex"
    unchanged = ExecutableIdentity(
        command, command, (1, 1, 100_000_000_000), (1, 2, 100_000_000_000)
    )
    result = _codex_result(before=unchanged, after=unchanged)
    assert classify_executable_replacement_interruption(
        result,
        command=command,
        response_file_text=None,
        last_message_artifact=None,
    ) is None

    replaced = ExecutableIdentity(
        command, "/configured/new-target", (1, 3, 101_000_000_000), (1, 4, 101_000_000_000)
    )
    result = _codex_result(before=unchanged, after=replaced)
    assert classify_executable_replacement_interruption(
        result,
        command=command,
        response_file_text=None,
        last_message_artifact=None,
    ) == "Codex executable changed during invocation"


@pytest.mark.parametrize(
    ("response_file_text", "last_message_artifact"),
    [("valid public response", None), (None, "malformed but present")],
)
def test_codex_artifacts_suppress_replacement_classification(
    response_file_text, last_message_artifact
):
    result = _codex_result()
    assert classify_executable_replacement_interruption(
        result,
        command="codex",
        response_file_text=response_file_text,
        last_message_artifact=last_message_artifact,
    ) is None


@pytest.mark.parametrize(
    ("interrupted", "capture_diagnostics"),
    [(True, ()), (False, ("capture_read_failed:OSError: injected",))],
)
def test_codex_replacement_excludes_interruptions_and_capture_loss(
    interrupted, capture_diagnostics
):
    result = _codex_result(
        interrupted=interrupted, capture_diagnostics=capture_diagnostics
    )
    assert classify_executable_replacement_interruption(
        result,
        command="codex",
        response_file_text=None,
        last_message_artifact=None,
    ) is None


def test_parse_claude_output_falls_back_on_non_string_result():
    raw = json.dumps({"result": 42, "session_id": "abc"})
    text, sid, usage, raw_usage, model = _parse_claude_output(raw)
    assert text == raw  # non-string result → fall back to raw
    assert sid == "abc"
    assert usage is None
    assert raw_usage is None
    assert model is None


def test_parse_claude_output_extracts_model_from_model_usage():
    raw = json.dumps({"result": "Hi.", "modelUsage": {"claude-sonnet-4-6": {"outputTokens": 5}}})
    text, sid, usage, raw_usage, model = _parse_claude_output(raw)
    assert text == "Hi."
    assert model == "claude-sonnet-4-6"


def test_parse_claude_output_extracts_primary_model_from_model_usage():
    raw = json.dumps({
        "result": "Reviewed.",
        "modelUsage": {
            "claude-haiku-4-5-20251001": {"inputTokens": 4750, "outputTokens": 20},
            "claude-sonnet-4-6": {
                "inputTokens": 18,
                "outputTokens": 8263,
                "cacheReadInputTokens": 715975,
            },
        },
    })
    text, sid, usage, raw_usage, model = _parse_claude_output(raw)
    assert text == "Reviewed."
    assert model == "claude-sonnet-4-6"


def test_parse_claude_output_prefers_top_level_model_over_model_usage():
    raw = json.dumps({
        "result": "Reviewed.",
        "model": "claude-opus-4-1",
        "modelUsage": {
            "claude-haiku-4-5-20251001": {"outputTokens": 20},
            "claude-sonnet-4-6": {"outputTokens": 8263},
        },
    })
    text, sid, usage, raw_usage, model = _parse_claude_output(raw)
    assert text == "Reviewed."
    assert model == "claude-opus-4-1"


def test_parse_gemini_output_extracts_json_response():
    raw = json.dumps({
        "response": "Reviewed.\n<!-- AGENT_STATE: approved -->",
        "session_id": "gemini-session-1",
    })
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text == "Reviewed.\n<!-- AGENT_STATE: approved -->"
    assert sid == "gemini-session-1"
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_falls_back_on_plain_text():
    text, sid, usage, raw_usage, source = _parse_gemini_payload("plain response")
    assert text == "plain response"
    assert sid is None
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_falls_back_on_non_string_response():
    raw = json.dumps({"response": 42, "session_id": "gemini-session-1"})
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text == raw
    assert sid == "gemini-session-1"
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_prefers_public_response_marker():
    raw = f"""Warning: True color (24-bit) support not detected.
YOLO mode is enabled. All tool calls will be automatically approved.
I will inspect the PR before giving the final answer.
Error executing tool read_file: Path not in workspace.
{PUBLIC_RESPONSE_MARKER}
## Review

No blocking findings.

<!-- AGENT_STATE: approved -->

-- Google Gemini
"""
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text.startswith("## Review")
    assert "True color" not in text
    assert "YOLO mode" not in text
    assert "I will inspect" not in text
    assert "Error executing tool" not in text
    assert "<!-- AGENT_STATE: approved -->" in text
    assert sid is None
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_uses_last_public_response_marker():
    raw = f"""Gemini may mention {PUBLIC_RESPONSE_MARKER} while planning.
{PUBLIC_RESPONSE_MARKER}
intermediate draft
{PUBLIC_RESPONSE_MARKER}
Final answer.
<!-- AGENT_STATE: approved -->
"""
    text, _sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text == "Final answer.\n<!-- AGENT_STATE: approved -->\n"
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_json_response_strips_public_response_marker():
    raw = json.dumps({
        "response": f"diagnostic\n{PUBLIC_RESPONSE_MARKER}\nReviewed.\n<!-- AGENT_STATE: approved -->",
        "session_id": "gemini-session-1",
    })
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text == "Reviewed.\n<!-- AGENT_STATE: approved -->"
    assert sid == "gemini-session-1"
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_strips_cli_preamble_before_final_response():
    raw = """Warning: True color (24-bit) support not detected.
YOLO mode is enabled. All tool calls will be automatically approved.
Attempt 1 failed with status 429. Retrying with backoff... _GaxiosError: [{
  "error": {
    "code": 429,
    "message": "No capacity available for model gemini-3-flash-preview on the server"
  }
}]
I am now ready to provide my final response.

---

## Code Review

Looks good.

<!-- AGENT_STATE: approved -->

-- Google Gemini
"""
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text.startswith("## Code Review")
    assert "_GaxiosError" not in text
    assert "YOLO mode" not in text
    assert "<!-- AGENT_STATE: approved -->" in text
    assert sid is None
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_strips_cli_preamble_before_plan_state_marker():
    raw = """Warning: True color (24-bit) support not detected.
YOLO mode is enabled.
I will now review the plan.

---

## Plan Review

Looks like a solid approach.

<!-- AGENT_PLAN_STATE: approved -->

-- Google Gemini
"""
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text.startswith("## Plan Review")
    assert "YOLO mode" not in text
    assert "<!-- AGENT_PLAN_STATE: approved -->" in text
    assert sid is None


def test_parse_gemini_output_preserves_markdown_rules_after_preamble():
    raw = """Warning: True color (24-bit) support not detected.
YOLO mode is enabled.

---

## Summary

Reviewed the change.

---

## Details

Still looks good.

<!-- AGENT_STATE: approved -->
"""
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text.startswith("## Summary")
    assert "YOLO mode" not in text
    assert "## Details" in text
    assert "\n---\n\n## Details" in text
    assert sid is None
    assert usage is None
    assert raw_usage is None


def test_parse_gemini_output_strips_preamble_before_clarification_marker():
    raw = """Warning: True color (24-bit) support not detected.
I need to ask a question.

---

    Which endpoint should I update?
<!-- AGENT_CLARIFY -->
"""
    text, sid, usage, raw_usage, source = _parse_gemini_payload(raw)
    assert text.startswith("    Which endpoint")
    assert "True color" not in text
    assert "<!-- AGENT_CLARIFY -->" in text
    assert sid is None
    assert usage is None
    assert raw_usage is None


def test_normalize_claude_usage_keeps_zero_cached_tokens_exact():
    usage = _normalize_claude_usage(
        {
            "input_tokens": 12,
            "cached_input_tokens": 0,
            "output_tokens": 8,
            "total_tokens": 20,
        }
    )

    assert usage is not None
    assert usage.mode == "exact"
    assert usage.cached_input_tokens == 0


def test_normalize_codex_usage_keeps_zero_reasoning_tokens():
    usage = _normalize_codex_usage(
        {
            "input_tokens": 12,
            "cached_input_tokens": 0,
            "output_tokens": 8,
            "reasoning_tokens": 0,
            "total_tokens": 20,
        }
    )

    assert usage is not None
    assert usage.mode == "exact"
    assert usage.reasoning_tokens == 0


def test_normalize_gemini_usage_keeps_zero_token_values_exact():
    usage = _normalize_gemini_usage(
        {
            "inputTokenCount": 0,
            "cachedInputTokenCount": 0,
            "outputTokenCount": 4,
            "totalTokenCount": 4,
        }
    )

    assert usage is not None
    assert usage.mode == "exact"
    assert usage.input_tokens == 0
    assert usage.cached_input_tokens == 0


def test_extract_codex_usage_reads_turn_completed_jsonl():
    usage, raw_usage = _extract_codex_usage(
        "\n".join(
            [
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 120,
                            "cached_input_tokens": 30,
                            "output_tokens": 45,
                            "reasoning_tokens": 11,
                            "total_tokens": 206,
                        },
                    }
                ),
            ]
        )
    )

    assert usage is not None
    assert usage.mode == "exact"
    assert usage.input_tokens == 120
    assert usage.cached_input_tokens == 30
    assert usage.output_tokens == 45
    assert usage.reasoning_tokens == 11
    assert usage.total_tokens == 206
    assert raw_usage == {
        "input_tokens": 120,
        "cached_input_tokens": 30,
        "output_tokens": 45,
        "reasoning_tokens": 11,
        "total_tokens": 206,
    }


def test_claude_backend_prefers_response_file_over_message_text(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "result": "stdout message text",
                    "session_id": "claude-session-1",
                }
            )
        ],
        public_response_outputs=["response file text"],
    )
    config = make_config(tmp_path)

    result = CLAUDE_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert runner.last_input_text is None
    assert any("Review this PR." in arg for arg in runner.commands[-1][0])
    assert result.response_file_text == "response file text"
    assert result.response_file_path is not None
    assert result.response_file_path.read_text(encoding="utf-8").strip() == "response file text"
    assert result.message_text == "stdout message text"
    assert result.text == "response file text"
    assert result.session_id == "claude-session-1"


def test_claude_backend_sends_large_prompt_on_stdin(tmp_path):
    runner = FakeRunner(claude_outputs=[json.dumps({"result": "ok"})])
    config = make_config(tmp_path)
    prompt = "x" * 150_000

    CLAUDE_BACKEND.run(runner, config, prompt, run_id="run-1")

    assert runner.last_input_text is not None
    assert prompt in runner.last_input_text
    assert all(prompt not in arg for arg in runner.commands[-1][0])


def test_gemini_backend_prefers_response_file_over_message_text(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            json.dumps(
                {
                    "response": f"diagnostic\n{PUBLIC_RESPONSE_MARKER}\nstdout message text",
                    "session_id": "gemini-session-1",
                }
            )
        ],
        public_response_outputs=["response file text"],
    )
    config = make_config(tmp_path)

    result = GEMINI_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.response_file_text == "response file text"
    assert result.response_file_path is not None
    assert result.response_file_path.read_text(encoding="utf-8").strip() == "response file text"
    assert result.message_text == "stdout message text"
    assert result.text == "response file text"
    assert result.session_id == "gemini-session-1"


def test_gemini_backend_sends_oversized_decorated_prompt_on_stdin(tmp_path, monkeypatch):
    import coding_review_agent_loop.agents.gemini as gemini_module

    monkeypatch.setattr(gemini_module, "STDIN_PROMPT_THRESHOLD_BYTES", 100)
    raw_prompt = "review Ω " * 30
    runner = FakeRunner(
        gemini_outputs=[json.dumps({"response": "stdout message"})],
        public_response_outputs=["response file text"],
    )
    config = make_config(
        tmp_path,
        gemini_args=("--yolo", "--skip-trust", "--output-format", "json"),
        gemini_model="gemini-test",
    )

    result = GEMINI_BACKEND.run(runner, config, raw_prompt, session_id="resume-1", run_id="run-1")

    cmd = runner.commands[-1][0]
    assert cmd[:2] == ["gemini", "--prompt"]
    assert cmd[2] == _OVERSIZED_PROMPT_DIRECTIVE
    assert all(raw_prompt not in arg for arg in cmd)
    assert runner.last_input_text is not None and raw_prompt in runner.last_input_text
    assert all(flag in cmd for flag in ("--yolo", "--skip-trust", "--output-format", "--model", "--resume"))
    assert "gemini-test" in cmd and "resume-1" in cmd
    assert result.text == "response file text"


def test_gemini_backend_measures_fully_decorated_prompt_in_utf8_bytes(tmp_path, monkeypatch):
    import coding_review_agent_loop.agents.gemini as gemini_module

    raw_prompt = "é" * 20
    config = make_config(tmp_path)
    response_path = gemini_module.public_response_path(
        config, "gemini", root=gemini_module._gemini_public_response_root(config.gemini_dir)
    )
    rendered = gemini_module._with_public_response_marker_instruction(
        gemini_module.with_public_response_file_instruction(raw_prompt, response_path)
    )
    # The raw multibyte text makes the UTF-8 payload 20 bytes larger than its
    # character count, so a character-based threshold check would take argv.
    monkeypatch.setattr(gemini_module, "STDIN_PROMPT_THRESHOLD_BYTES", len(rendered))
    runner = FakeRunner(gemini_outputs=["ok"])

    GEMINI_BACKEND.run(runner, config, raw_prompt, run_id="run-1")

    cmd = runner.commands[-1][0]
    assert cmd[2] == _OVERSIZED_PROMPT_DIRECTIVE
    assert runner.last_input_text is not None and raw_prompt in runner.last_input_text


def test_gemini_backend_uses_prompt_argument_at_exact_threshold(tmp_path, monkeypatch):
    import coding_review_agent_loop.agents.gemini as gemini_module

    config = make_config(tmp_path)
    runner = FakeRunner(gemini_outputs=["ok"])
    # Determine the decorated byte size so equality exercises the small path.
    response_path = gemini_module.public_response_path(
        config, "gemini", root=gemini_module._gemini_public_response_root(config.gemini_dir)
    )
    rendered = gemini_module._with_public_response_marker_instruction(
        gemini_module.with_public_response_file_instruction("short", response_path)
    )
    monkeypatch.setattr(gemini_module, "STDIN_PROMPT_THRESHOLD_BYTES", len(rendered.encode("utf-8")))

    GEMINI_BACKEND.run(runner, config, "short", run_id="run-1")

    assert runner.last_input_text is None
    assert runner.commands[-1][0][2] != _OVERSIZED_PROMPT_DIRECTIVE
    assert len(runner.commands[-1][0][2].encode("utf-8")) == len(rendered.encode("utf-8"))


def test_codex_backend_prefers_response_file_over_last_message_and_stdout(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": "\n".join(
                    [
                        "noisy stdout chatter",
                        json.dumps(
                            {
                                "type": "turn.completed",
                                "usage": {
                                    "input_tokens": 12,
                                    "cached_input_tokens": 3,
                                    "output_tokens": 4,
                                    "reasoning_tokens": 1,
                                    "total_tokens": 20,
                                },
                            }
                        ),
                    ]
                ),
            }
        ],
        public_response_outputs=["response file text"],
    )
    config = make_config(tmp_path)

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.response_file_text == "response file text"
    assert result.response_file_path is not None
    assert result.response_file_path.read_text(encoding="utf-8").strip() == "response file text"
    assert result.message_text == "last message text"
    assert result.text == "response file text"
    assert result.usage is not None
    assert result.usage.total_tokens == 20


def test_codex_backend_sends_large_prompt_on_stdin(tmp_path):
    runner = FakeRunner(
        codex_outputs=[{"public_response": "last message text"}],
        public_response_outputs=["response file text"],
    )
    config = make_config(tmp_path)
    prompt = "x" * 150_000

    result = CODEX_BACKEND.run(runner, config, prompt, run_id="run-1")

    assert runner.last_input_text is not None
    assert prompt in runner.last_input_text
    assert all(prompt not in arg for arg in runner.commands[-1][0])
    assert result.response_file_text == "response file text"


def test_codex_backend_prefers_last_message_over_stdout_without_response_file(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": "raw stdout fallback",
            }
        ]
    )
    config = make_config(tmp_path)

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.response_file_text is None
    assert result.message_text == "last message text"
    assert result.text == "last message text"


def test_codex_backend_uses_stdout_when_files_are_absent_or_empty(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "",
                "stdout": "raw stdout fallback",
            }
        ]
    )
    config = make_config(tmp_path)

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.response_file_text is None
    assert result.message_text == "raw stdout fallback"
    assert result.text == "raw stdout fallback"


def test_codex_backend_dry_run_sets_message_text_without_response_file(tmp_path):
    runner = FakeRunner(codex_outputs=[{"stdout": "dry run stdout"}])
    config = make_config(tmp_path, dry_run=True)

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.response_file_text is None
    assert result.message_text == "dry run stdout"
    assert result.text == "dry run stdout"


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"payload": {"model": "gpt-5.5"}}, "gpt-5.5"),
        (
            {"turn": {"model": "gpt-5.5", "model_reasoning_effort": "medium"}},
            "gpt-5.5 (medium)",
        ),
    ],
)
def test_codex_backend_detects_model_from_rollout(tmp_path, monkeypatch, record, expected):
    thread_id = "019ed9d8-1111-7222-8333-444444444444"
    codex_home = tmp_path / "codex-home"
    _write_codex_rollout(codex_home, thread_id, [record])
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": json.dumps({"type": "thread.started", "thread_id": thread_id}),
            }
        ]
    )
    config = make_config(tmp_path)

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.model_used == expected


def test_codex_backend_parses_current_turn_context_rollout_schema(tmp_path, monkeypatch):
    thread_id = "019ed9d8-1111-7222-8333-444444444444"
    codex_home = tmp_path / "codex-home"
    _write_codex_rollout(
        codex_home,
        thread_id,
        [
            {
                "timestamp": "2026-06-18T12:00:00.000Z",
                "type": "turn_context",
                "payload": {
                    "model": "gpt-5.5",
                    "effort": "high",
                },
            }
        ],
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": json.dumps({"type": "thread.started", "thread_id": thread_id}),
            }
        ]
    )

    result = CODEX_BACKEND.run(
        runner,
        make_config(tmp_path),
        "Review this PR.",
        run_id="run-1",
    )

    assert result.model_used == "gpt-5.5 (high)"


def test_codex_backend_accepts_mixed_case_uuid_for_rollout_lookup(tmp_path, monkeypatch):
    thread_id = "019ED9d8-1111-7222-8333-444444444444"
    codex_home = tmp_path / "codex-home"
    _write_codex_rollout(
        codex_home,
        thread_id,
        [{"payload": {"model": "gpt-5.5", "effort": "medium"}}],
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": json.dumps({"type": "thread.started", "thread_id": thread_id}),
            }
        ]
    )

    result = CODEX_BACKEND.run(
        runner,
        make_config(tmp_path),
        "Review this PR.",
        run_id="run-1",
    )

    assert result.model_used == "gpt-5.5 (medium)"


@pytest.mark.parametrize(
    ("thread_id", "declared_model", "expected"),
    [
        ("*", None, None),
        ("deadbeef-dead-beef-dead-beef", "gpt-5.4", "gpt-5.4"),
    ],
)
def test_codex_backend_rejects_non_uuid_thread_id_before_rollout_lookup(
    tmp_path,
    monkeypatch,
    thread_id,
    declared_model,
    expected,
):
    codex_home = tmp_path / "codex-home"
    _write_codex_rollout(
        codex_home,
        thread_id,
        [{"payload": {"model": "wrong-model"}}],
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": json.dumps({"type": "thread.started", "thread_id": thread_id}),
            }
        ]
    )

    result = CODEX_BACKEND.run(
        runner,
        make_config(tmp_path, codex_model=declared_model),
        "Review this PR.",
        run_id="run-1",
    )

    assert result.model_used == expected


def test_codex_backend_declared_model_takes_precedence_over_rollout(tmp_path, monkeypatch):
    thread_id = "019ed9d8-1111-7222-8333-444444444444"
    codex_home = tmp_path / "codex-home"
    _write_codex_rollout(
        codex_home,
        thread_id,
        [{"payload": {"model": "gpt-5.5", "effort": "high"}}],
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": json.dumps({"type": "thread.started", "thread_id": thread_id}),
            }
        ]
    )
    config = make_config(
        tmp_path,
        codex_model="gpt-5.4",
        codex_reasoning_effort="medium",
    )

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.model_used == "gpt-5.4 (medium)"


@pytest.mark.parametrize(
    ("stdout", "records"),
    [
        ("not json", [{"payload": {"model": "gpt-5.5"}}]),
        (
            json.dumps(
                {
                    "type": "thread.started",
                    "thread_id": "019ed9d8-1111-7222-8333-444444444444",
                }
            ),
            None,
        ),
        (
            json.dumps(
                {
                    "type": "thread.started",
                    "thread_id": "019ed9d8-1111-7222-8333-444444444444",
                }
            ),
            ["not json", {"payload": {"model": ""}}, {"payload": {"model": 55}}],
        ),
    ],
)
def test_codex_backend_missing_or_invalid_rollout_model_falls_back_to_none(
    tmp_path,
    monkeypatch,
    stdout,
    records,
):
    thread_id = "019ed9d8-1111-7222-8333-444444444444"
    codex_home = tmp_path / "codex-home"
    if records is not None:
        _write_codex_rollout(codex_home, thread_id, records)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[{"public_response": "last message text", "stdout": stdout}]
    )
    config = make_config(tmp_path)

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.model_used is None


def test_codex_backend_invalid_rollout_falls_back_to_declared_model(tmp_path, monkeypatch):
    thread_id = "019ed9d8-1111-7222-8333-444444444444"
    codex_home = tmp_path / "codex-home"
    _write_codex_rollout(codex_home, thread_id, ["not json"])
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    runner = FakeRunner(
        codex_outputs=[
            {
                "public_response": "last message text",
                "stdout": json.dumps({"type": "thread.started", "thread_id": thread_id}),
            }
        ]
    )
    config = make_config(tmp_path, codex_model="gpt-5.4")

    result = CODEX_BACKEND.run(runner, config, "Review this PR.", run_id="run-1")

    assert result.model_used == "gpt-5.4"
