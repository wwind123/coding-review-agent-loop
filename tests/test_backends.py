"""Backend tests extracted from test_agent_loop.py (lines 1302–2015).

Tests for Claude, Gemini, and Codex backend output parsing and normalization.
"""
import json
from pathlib import Path

import pytest

from coding_review_agent_loop.agents.claude import (
    BACKEND as CLAUDE_BACKEND,
    _normalize_claude_usage,
    _parse_claude_output,
)
from coding_review_agent_loop.agents.codex import (
    BACKEND as CODEX_BACKEND,
    _extract_codex_usage,
    _normalize_codex_usage,
)
from coding_review_agent_loop.agents.gemini import (
    BACKEND as GEMINI_BACKEND,
    PUBLIC_RESPONSE_MARKER,
    _normalize_gemini_usage,
    _parse_gemini_payload,
)

from agent_loop_helpers import (
    FakeRunner,
    make_config,
)
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _no_real_repair():
    """Prevent attempt_repair from calling the real Gemini CLI in all tests."""
    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _agent_commands_available(monkeypatch):
    """Keep config tests independent of agent CLIs installed on the test host."""
    import coding_review_agent_loop.config as config_module

    real_which = config_module.shutil.which

    def which(command):
        resolved = real_which(command)
        if resolved is not None:
            return resolved
        if command in {"claude", "codex", "gemini", "agy"}:
            return f"/mock/bin/{command}"
        return None

    monkeypatch.setattr(config_module.shutil, "which", which)


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

    assert result.response_file_text == "response file text"
    assert result.message_text == "stdout message text"
    assert result.text == "response file text"
    assert result.session_id == "claude-session-1"


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
    assert result.message_text == "stdout message text"
    assert result.text == "response file text"
    assert result.session_id == "gemini-session-1"


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
    assert result.message_text == "last message text"
    assert result.text == "response file text"
    assert result.usage is not None
    assert result.usage.total_tokens == 20


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


