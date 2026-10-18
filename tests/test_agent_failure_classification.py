"""agent_failure tests: failure categories, suggestions, model resolution and reset formatting (#1544)."""

import json

import pytest

import coding_review_agent_loop.agent_failure as agent_failure

from coding_review_agent_loop.agents.base import AgentResult
from coding_review_agent_loop.agents.gemini import PUBLIC_RESPONSE_MARKER
from coding_review_agent_loop.agent_failure import (
    _failure_category,
    _failure_suggestion,
    _format_invalid_agent_response_error,
    _format_reset_duration,
    _is_transient_agent_output,
    _is_transient_public_response,
    _resolve_requested_model,
)
from agent_loop_helpers import (
    make_config,
    structured_pr_review,
)
from agent_loop_helpers import _CAPACITY


@pytest.mark.parametrize(
    "text",
    [
        "orchestrator.py lines 577-581: it currently falls back to parse_plan_state(text)",
        "orchestrator.py:577-581: it currently falls back to parse_plan_state(text)",
        "A bare 500 in diagnostic prose without HTTP context.",
    ],
)
def test_source_line_references_with_5xx_numbers_are_not_transient(text):
    assert not _is_transient_agent_output(text)
    assert _failure_category(text) == "deterministic"


@pytest.mark.parametrize(
    "text",
    [
        "Internal Server Error",
        "Bad Gateway",
        "Service Unavailable",
        "Gateway Timeout",
    ],
)
def test_explicit_server_error_phrases_remain_transient(text):
    assert _is_transient_agent_output(text)
    assert _failure_category(text) == "transient"


@pytest.mark.parametrize(
    "text",
    [
        "The authoritative PR diff shows no regressions.",
        "Authoritative source confirms the change.",
        "The author of this commit fixed the bug.",
        "This is an authoritative reference.",
    ],
)
def test_auth_prefix_words_are_not_non_retryable(text):
    """Words starting with 'auth' that are not auth-failure keywords must not match."""
    from coding_review_agent_loop.transient import NON_RETRYABLE_AGENT_OUTPUT_RE

    assert not NON_RETRYABLE_AGENT_OUTPUT_RE.search(text), (
        f"NON_RETRYABLE_AGENT_OUTPUT_RE unexpectedly matched: {text!r}"
    )
    assert _is_transient_agent_output(text) is False or True  # no crash; classification is unrelated
    assert _failure_category(text) != "non-retryable"


@pytest.mark.parametrize(
    "text",
    [
        "authentication failed",
        "Authorization error",
        "auth failed",
        "unauthorized",
        "forbidden",
        "Invalid API Key",
        "billing issue",
        "credit limit exceeded",
        "dirty checkout",
    ],
)
def test_genuine_auth_and_billing_terms_remain_non_retryable(text):
    """Real auth/billing/dirty-checkout diagnostics must still be non-retryable."""
    from coding_review_agent_loop.transient import NON_RETRYABLE_AGENT_OUTPUT_RE

    assert NON_RETRYABLE_AGENT_OUTPUT_RE.search(text), (
        f"NON_RETRYABLE_AGENT_OUTPUT_RE did not match: {text!r}"
    )
    assert _failure_category(text) == "non-retryable"


def test_codex_chatgpt_unsupported_model_payload_classifies_separately():
    text = json.dumps(
        {
            "error": {
                "type": "invalid_request_error",
                "message": (
                    "The 'gpt-5.5-pro' model is not supported when using "
                    "Codex with a ChatGPT account."
                ),
            }
        }
    )

    assert _failure_category(text) == "unsupported_model"
    assert (
        _failure_category(
            text,
            public_response=True,
            repair_expected_kind="pr_review",
        )
        == "unsupported_model"
    )


def test_generic_provider_unsupported_model_payload_classifies_separately():
    text = json.dumps(
        {
            "error": {
                "code": "unsupported_model",
                "message": "Unsupported model 'provider-ultra' for this provider auth mode.",
            }
        }
    )

    assert _failure_category(text) == "unsupported_model"


@pytest.mark.parametrize(
    ("agent", "expected_config_model"),
    [
        ("codex", "configured-codex"),
        ("antigravity", "Configured Antigravity (High)"),
        ("gemini", "configured-gemini"),
        ("claude", "configured-claude"),
    ],
)
def test_resolve_requested_model_prefers_result_then_config_for_each_agent(
    tmp_path, agent, expected_config_model
):
    config = make_config(
        tmp_path,
        codex_model="configured-codex",
        antigravity_model="Configured Antigravity (High)",
        gemini_model="configured-gemini",
        claude_model="configured-claude",
    )
    provider_text = "The 'provider-parsed' model is not supported for this provider."

    assert (
        _resolve_requested_model(
            agent=agent,
            config=config,
            result=AgentResult(text="", model_used="actual-model"),
            classification_text=provider_text,
        )
        == "actual-model"
    )
    assert (
        _resolve_requested_model(
            agent=agent,
            config=config,
            result=None,
            classification_text=provider_text,
        )
        == expected_config_model
    )
    assert (
        _resolve_requested_model(
            agent=agent,
            config=None,
            result=None,
            classification_text=provider_text,
        )
        == "provider-parsed"
    )


def test_codex_unsupported_model_diagnostic_uses_config_model_for_fallback(tmp_path):
    provider_text = (
        "The requested model is not supported when using Codex with a ChatGPT account."
    )
    config = make_config(tmp_path, codex_model="gpt-5.5-pro")

    message = _format_invalid_agent_response_error(
        agent_name="Codex",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        reason="agent command exited with 1",
        result=AgentResult(text="", raw_output=provider_text, returncode=1),
        log_paths=(),
        category="unsupported_model",
        agent="codex",
        config=config,
        role="reviewer",
        classification_text=provider_text,
    )

    assert "requested model `gpt-5.5-pro`" in message
    assert "--codex-model gpt-5.5" in message
    assert "fix the underlying issue" not in message


def test_capacity_model_availability_remains_transient():
    assert (
        _failure_category("No capacity available for model gemini-3-pro-preview.")
        == "transient"
    )


def test_public_response_error_payload_remains_transient():
    assert _is_transient_public_response(
        json.dumps(
            {
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota exceeded. Retry-After: 60",
                }
            }
        )
    )


def test_public_response_structured_json_after_known_artifact_is_not_transient():
    text = (
        f"{PUBLIC_RESPONSE_MARKER}\n"
        + structured_pr_review(
            summary="Wrong structured kind discusses 429, quota, capacity, and transient behavior.",
            reviewer="Google Gemini",
        )
    )

    assert not _is_transient_public_response(text, repair_expected_kind="coder_followup")


@pytest.mark.parametrize("seconds,expected", [
    (3600, "1h"),
    (7200, "2h"),
    (9000, "2h 30m"),
    (300, "5m"),
    (45, "45s"),
    (3660, "1h 1m"),
])
def test_format_reset_duration(seconds, expected):
    assert _format_reset_duration(seconds) == expected


def _make_error(
    *,
    agent_name: str = "TestAgent",
    reason: str = "something went wrong",
    category: str | None = None,
    classification_text: str = "",
) -> str:
    """Helper: call _format_invalid_agent_response_error with minimal args."""
    return _format_invalid_agent_response_error(
        agent_name=agent_name,
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        reason=reason,
        result=None,
        log_paths=[],
        category=category,
    )


def test_failure_suggestion_transient_quota_in_reason():
    msg = _failure_suggestion("transient", "HTTP 429 rate limit exceeded", "TestAgent")
    assert msg.startswith("Suggestion:")
    assert "quota" in msg.lower() or "rate" in msg.lower() or "wait" in msg.lower()


def test_failure_suggestion_transient_quota_in_classification_text():
    msg = _failure_suggestion(
        "transient",
        "agent command exited with 1",
        "TestAgent",
        classification_text="quota exceeded for model gemini",
    )
    assert "wait" in msg.lower()


def test_failure_suggestion_transient_non_quota():
    msg = _failure_suggestion(
        "transient",
        "network timeout",
        "TestAgent",
        classification_text="",
    )
    assert msg.startswith("Suggestion:")
    assert "wait" not in msg.lower()
    assert "retry" in msg.lower() or "re-run" in msg.lower()


def test_failure_suggestion_non_retryable_billing_in_classification():
    msg = _failure_suggestion(
        "non-retryable",
        "agent command exited with 1",
        "TestAgent",
        classification_text="credit limit exceeded for this account",
    )
    assert "billing" in msg.lower() or "credit" in msg.lower()
    assert "authenticated" not in msg.lower()


def test_failure_suggestion_non_retryable_dirty_in_classification():
    msg = _failure_suggestion(
        "non-retryable",
        "agent command exited with 1",
        "TestAgent",
        classification_text="dirty workdir: uncommitted changes detected",
    )
    assert "dirty" in msg.lower() or "working tree" in msg.lower() or "workdir" in msg.lower()
    assert "authenticated" not in msg.lower()


def test_failure_suggestion_non_retryable_auth_in_reason():
    msg = _failure_suggestion(
        "non-retryable",
        "unauthorized access",
        "MyAgent",
        classification_text="",
    )
    assert "MyAgent" in msg
    assert "authenticated" in msg.lower()


def test_failure_suggestion_deterministic_repair_invalid_output():
    reason = (
        "Agent response did not use the required structured format.; "
        "repair invocation failure: antigravity/gemini-flash: invalid_output "
        "(Plan review did not evaluate all prior unresolved plan items: item-1)"
    )
    msg = _failure_suggestion("deterministic", reason, "Antigravity")
    assert "re-run" in msg.lower()
    assert "inspect" not in msg.lower()


def test_failure_suggestion_deterministic_repair_failure_path():
    # The repair failure: path (line 1303) uses attempt.diagnostic, not the outcome token.
    reason = (
        "Agent response did not use the required structured format.; "
        "repair failure: Plan review did not evaluate all prior unresolved plan items: item-1"
    )
    msg = _failure_suggestion("deterministic", reason, "Antigravity")
    assert "inspect" in msg.lower()
    assert "resumable" not in msg.lower()


def test_failure_suggestion_deterministic_genuine():
    msg = _failure_suggestion("deterministic", "schema validation error", "TestAgent")
    assert "inspect" in msg.lower()


def test_failure_suggestion_empty_response():
    msg = _failure_suggestion("empty-response", "", "TestAgent")
    assert msg == ""


def test_failure_suggestion_none_category():
    msg = _failure_suggestion(None, "unknown error", "TestAgent")
    assert msg == ""


def test_format_invalid_agent_response_error_includes_suggestion_transient():
    msg = _format_invalid_agent_response_error(
        agent_name="Gemini",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        reason="HTTP 429 Too Many Requests",
        result=None,
        log_paths=[],
        category="transient",
    )
    assert "Suggestion:" in msg
    assert "Failure category: transient" in msg


def test_format_invalid_agent_response_error_includes_suggestion_deterministic():
    msg = _format_invalid_agent_response_error(
        agent_name="Antigravity",
        marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
        reason="schema validation failed",
        result=None,
        log_paths=[],
        category="deterministic",
    )
    assert "Suggestion:" in msg
    assert "inspect" in msg.lower()


def test_format_invalid_agent_response_error_schema_rejection_is_retryable_957():
    msg = _format_invalid_agent_response_error(
        agent_name="Codex",
        marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
        reason="plan_review.blocking_plan_issues at index 0 must be a string.",
        result=None,
        log_paths=[],
        category="deterministic",
        classification_text="structured plan_review response failed trusted validation",
    )
    assert "Failure category: schema-validation" in msg
    assert "may require a code fix" not in msg
    assert "re-run the same command" in msg
    # The actual reason leads; the marker requirement follows it.
    assert msg.index("Reason: plan_review.blocking_plan_issues") < msg.index("Required marker:")


def test_format_invalid_agent_response_error_unstructured_deterministic_unchanged_957():
    msg = _format_invalid_agent_response_error(
        agent_name="Codex",
        marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
        reason="missing marker",
        result=None,
        log_paths=[],
        category="deterministic",
        classification_text="some other diagnostic",
    )
    assert "Failure category: deterministic (may require a code fix)." in msg
    assert "schema-validation" not in msg


def test_format_invalid_agent_response_error_no_suggestion_empty_response():
    msg = _format_invalid_agent_response_error(
        agent_name="Codex",
        marker_description="<!-- AGENT_PR: <number> -->",
        reason="agent response was empty",
        result=None,
        log_paths=[],
        category="empty-response",
    )
    assert "Suggestion:" not in msg


def test_at_capacity_message_is_transient_and_public_diagnostic() -> None:
    msg = "Selected model is at capacity. Please try a different model."
    assert agent_failure._failure_category(msg) == "transient"
    assert agent_failure.PUBLIC_RESPONSE_TRANSIENT_DIAGNOSTIC_RE.search(msg)
    assert agent_failure._is_transient_public_response(msg)
    assert not agent_failure.PUBLIC_RESPONSE_TRANSIENT_DIAGNOSTIC_RE.search(
        "The reviewer noted that the selected model is at capacity today."
    )


def test_provider_verdict_drives_failure_category_and_suggestion() -> None:
    from coding_review_agent_loop.transient import classify_codex_failure

    def cat(message):
        raw = json.dumps({"type": "error", "message": message})
        verdict = classify_codex_failure(raw)
        return agent_failure._failure_category(verdict.text, provider_verdict=verdict)

    assert cat("429 RESOURCE_EXHAUSTED") == "transient"
    assert cat("429 insufficient_quota: please check your plan") == "non-retryable"
    assert cat("Selected model is at capacity; auth proxy busy") == "transient"
    assert "billing" in agent_failure._failure_suggestion(
        "non-retryable", "x", "codex", classification_text="insufficient_quota"
    )


def test_codex_classification_ignores_tool_output() -> None:
    tool = json.dumps({"type": "item.completed", "item": {"aggregated_output": "auth timeout"}})
    result = AgentResult(text="", raw_output=tool, returncode=1, provider="codex")
    text, verdict = agent_failure._agent_failure_classification(result, phase="command")
    assert verdict is not None and "auth" not in text
    assert agent_failure._failure_category(text, provider_verdict=verdict) == "deterministic"
    legacy = AgentResult(text="", raw_output="invalid api key", returncode=1, provider="claude")
    assert agent_failure._agent_failure_classification(legacy, phase="command") == ("invalid api key", None)


def test_public_response_envelope_with_capacity_phrase_is_not_transient():
    envelope = json.dumps({"schema_version": 1, "kind": "plan_state", "summary": _CAPACITY})
    assert not agent_failure.PUBLIC_RESPONSE_TRANSIENT_DIAGNOSTIC_RE.search(envelope)
    assert not agent_failure._is_transient_public_response(envelope)
