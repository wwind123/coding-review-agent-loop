"""Unit tests for the lightweight transient-output classifier."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coding_review_agent_loop import transient


def test_transient_signals_are_retryable() -> None:
    assert transient.is_transient_agent_output("Error: 429 Too Many Requests")
    assert transient.is_transient_agent_output("the model is overloaded")
    assert transient.is_transient_agent_output("connection timed out")
    assert transient.is_transient_agent_output("503 Service Unavailable")
    assert transient.is_transient_agent_output("resource exhausted")


def test_non_retryable_and_clean_output_are_not_transient() -> None:
    assert not transient.is_transient_agent_output("invalid api key")
    assert not transient.is_transient_agent_output("billing problem on your account")
    assert not transient.is_transient_agent_output("the plan looks good to me")
    assert not transient.is_transient_agent_output("")


def test_non_retryable_overrides_transient_signal() -> None:
    # A non-retryable signal (auth/billing) wins even when a transient term is present.
    assert not transient.is_transient_agent_output("got 429 but the api key is unauthorized")


def test_antigravity_capacity_requires_framed_provider_failure() -> None:
    signatures = ("high traffic", "try again in a minute", "429", "overload", "no capacity")
    for text in (
        "Error: Our servers are experiencing high traffic right now, please try again in a minute.",
        "Fatal: 429 Too Many Requests; temporarily at capacity",
        '{"error": {"code": "RESOURCE_EXHAUSTED", "message": "overload: no capacity"}}',
    ):
        assert transient.classify_antigravity_capacity(
            text, returncode=1, empty_response=False, signatures=signatures
        ).is_capacity
    assert transient.classify_antigravity_capacity(
        "Reading the diff to draft the review...\n"
        "Error: Our servers are experiencing high traffic right now, please try again in a minute.",
        returncode=1,
        empty_response=False,
        signatures=signatures,
    ).is_capacity
    assert not transient.classify_antigravity_capacity(
        "The review quotes: Error: Our servers are experiencing high traffic right now, please try again in a minute.",
        returncode=1,
        empty_response=False,
        signatures=signatures,
    ).is_capacity
    assert not transient.classify_antigravity_capacity(
        "The review quotes the provider failure below:\n"
        "> Error: Our servers are experiencing high traffic right now, please try again in a minute.",
        returncode=1,
        empty_response=False,
        signatures=signatures,
    ).is_capacity
    assert not transient.classify_antigravity_capacity(
        "Error: high traffic but invalid API key", returncode=1, empty_response=False, signatures=signatures
    ).is_capacity
    assert transient.classify_antigravity_capacity(
        "quota exceeded please try again", returncode=1, empty_response=False, signatures=("quota",)
    ).is_capacity


def test_orchestrator_alias_preserves_identity() -> None:
    # orchestrator keeps the old private name as an alias to the moved implementation.
    assert agent_failure._is_transient_agent_output is transient.is_transient_agent_output


def test_backgrounded_completion_phrases_match() -> None:
    assert transient.looks_like_backgrounded_completion(
        "I'll wait for the background test run to finish."
    )
    assert transient.looks_like_backgrounded_completion(
        "Waiting for background tests to complete."
    )
    assert transient.looks_like_backgrounded_completion(
        "Waiting on the background test run and the exit-monitor; I'll continue once results arrive."
    )
    assert transient.looks_like_backgrounded_completion(
        "I started the full suite in the background; will get notified when it finishes."
    )
    assert transient.looks_like_backgrounded_completion(
        "You will be notified once the background build finishes."
    )
    assert transient.looks_like_backgrounded_completion(
        "Let me wait for the tests in the background to complete before finishing this."
    )
    assert transient.looks_like_backgrounded_completion(
        "Running the test suite in the background now."
    )


def test_unrelated_failure_text_does_not_match() -> None:
    assert not transient.looks_like_backgrounded_completion(
        "I do not have enough information to proceed."
    )
    assert not transient.looks_like_backgrounded_completion(
        "Please wait while I review the diff."
    )
    assert not transient.looks_like_backgrounded_completion("The tests passed and the PR is ready.")
    assert not transient.looks_like_backgrounded_completion("")


def test_backgrounded_completion_phrase_matches_regardless_of_embedded_markers() -> None:
    # The phrase heuristic is purely textual (#588): eligibility for a
    # completion-recovery attempt is gated separately, by the caller's own
    # terminal-result validator having already rejected the response -- not
    # by whether some other (possibly invalid-for-that-validator) marker is
    # present. An embedded AGENT_STATE: approved or a quoted AGENT_PLAN_STATE
    # marker must never exempt matching text from this detector.
    assert transient.looks_like_backgrounded_completion(
        "I'll wait for the background test run to finish.\n<!-- AGENT_STATE: approved -->"
    )
    assert transient.looks_like_backgrounded_completion(
        "As discussed in the prior round (<!-- AGENT_PLAN_STATE: blocking -->), "
        "I'll wait for the background build to finish before continuing."
    )


# --- #1236: provider-frame quota exhaustion --------------------------------

import subprocess

import pytest

from coding_review_agent_loop import agent_failure, reset_parsing
from coding_review_agent_loop.config import DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES as _SIGS

_LIVE = (
    "error: RESOURCE_EXHAUSTED (code 429): Resource has been exhausted (e.g. check quota).\n"
    'AGY_ERROR: {"status":"RESOURCE_EXHAUSTED","error_code":429,"retryable":true}'
)


def _classify(text: str, returncode: int = 1):
    capacity = transient.classify_antigravity_capacity(
        text, returncode=returncode, empty_response=False, signatures=_SIGS
    )
    return capacity, transient.classify_antigravity_quota_exhaustion(capacity)


def test_live_agy_sample_is_cooldown_exhaustion() -> None:
    capacity, quota = _classify(_LIVE)
    assert capacity.is_capacity and "AGY_ERROR" in capacity.frame
    assert quota is not None and quota.reset_source == "cooldown" and quota.reset_seconds is None


@pytest.mark.parametrize(
    "text",
    [
        "Review quota accounting try again in 4h\nError: high traffic",
        'quoted "quota ... try again in 4h"\nother\nstuff\nmore\nError: high traffic',
        "Error: capacity exhausted; high traffic",
        "Error: quota exceeded, try again in 2m",
        "Error: 429 too many requests",
        "429",
        "Error: high traffic, try again in a minute",
    ],
)
def test_non_exhaustion_frames_mark_nothing(text: str) -> None:
    _capacity, quota = _classify(text)
    assert quota is None


def test_auth_failure_stays_non_retryable() -> None:
    capacity, quota = _classify("Error: unauthorized, quota check failed")
    assert not capacity.is_capacity and quota is None


def test_indented_multiline_frame_parses_reset_and_blank_line_ends_frame() -> None:
    text = "Error: quota exceeded\n    a\n    b\n    try again in 4h"
    capacity, quota = _classify(text)
    assert capacity.is_capacity and quota.reset_seconds == 14400 and quota.reset_source == "parsed"
    _c, quota = _classify("Error: quota exceeded\n\nunrelated try again in 4h")
    assert quota is not None and quota.reset_seconds is None


def test_pretty_json_frame_uses_quoted_retry_delay() -> None:
    text = (
        '{\n  "error": {\n    "code": 429,\n    "status": "RESOURCE_EXHAUSTED",\n'
        '    "details": [{\n      "retryDelay": "14400s"\n    }]\n  }\n}'
    )
    capacity, quota = _classify(text)
    assert capacity.is_capacity and quota.reset_seconds == 14400 and quota.reset_source == "parsed"


def test_head_only_frame_in_long_output_is_ignored() -> None:
    head = "Error: quota exceeded, try again in 4h\n"
    text = head + ("filler line of transcript text\n" * 200) + "final answer incomplete"
    capacity, quota = _classify(text)
    assert not capacity.is_capacity and quota is None


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        ('"retryDelay": "14400s"', 14400),
        ('"retryDelay": "14400.5s"', 14400),
        ("retryDelay: '7200s'", 7200),
        ('{\n  "details": {\n    "retryDelay": "3600s"\n  }\n}', 3600),
        ("a\n  b\n  try again in 4h", 14400),
        ("Error: quota exceeded", None),
    ],
)
def test_antigravity_frame_reset_seconds(frame: str, expected) -> None:
    got = transient._antigravity_frame_reset_seconds(frame)
    assert got == expected


def test_previously_qualifying_capacity_inputs_still_qualify() -> None:
    for text in (
        "Error: failed\nquota exceeded",
        "quota exceeded please try again",
        '{"error": {"code": 429, "message": "quota"}}',
        "reviewing\nError: high traffic",
    ):
        assert _classify(text)[0].is_capacity, text


def test_reset_parsing_reexports_are_identical() -> None:
    for name in (
        "_parse_rate_limit_reset_seconds", "_parse_absolute_reset_seconds",
        "_RETRY_AFTER_SECONDS_RE", "_TRY_AGAIN_IN_RE", "_RESET_IN_RE",
        "_ABSOLUTE_RESET_TIME_RE", "_ISO_TIMESTAMP_RE",
    ):
        assert getattr(agent_failure, name) is getattr(reset_parsing, name)
    root = str(Path(__file__).parent.parent / "src")
    for order in (
        "import coding_review_agent_loop.transient, coding_review_agent_loop.agent_failure",
        "import coding_review_agent_loop.agent_failure, coding_review_agent_loop.transient",
    ):
        proc = subprocess.run(
            [sys.executable, "-c", order], env={"PYTHONPATH": root}, capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr


def test_sparse_newline_head_header_has_no_frame() -> None:
    text = "Error: quota exceeded, try again in 4h\n" + "x" * 3000
    capacity, quota = _classify(text)
    assert capacity.is_capacity and capacity.frame == ""
    assert quota is None


# ---------------------------------------------------------------------------
# Codex provider error channel (#1269)
# ---------------------------------------------------------------------------

import json as _json


def _ev(**kw) -> str:
    return _json.dumps(kw)


_CAP = "Selected model is at capacity. Please try a different model."
_TOOL = _ev(
    type="item.completed",
    item={"type": "command_execution", "aggregated_output": "loadHistoryList(owner = auth.captureOwner()) billing"},
)


def _stream(*lines: str) -> str:
    return "\n".join(lines)


def test_at_capacity_is_transient_text() -> None:
    assert transient.TRANSIENT_AGENT_OUTPUT_RE.search(_CAP)


def test_issue_capacity_pair_is_transient_despite_tool_output() -> None:
    raw = _stream(
        _ev(type="thread.started", thread_id="t"),
        _TOOL,
        _ev(type="error", message=_CAP),
        _ev(type="turn.failed", error={"message": _CAP}),
    )
    verdict = transient.classify_codex_failure(raw)
    assert verdict is not None
    assert verdict.category == "transient" and verdict.source == "structured"
    assert "auth" not in verdict.text and "billing" not in verdict.text


@pytest.mark.parametrize("message", ["401 Unauthorized", "invalid api key"])
def test_structured_auth_is_non_retryable(message) -> None:
    verdict = transient.classify_codex_failure(_ev(type="error", message=message))
    assert verdict.category == "non-retryable" and not verdict.billing


def test_structured_auth_beats_stderr_capacity() -> None:
    raw = _stream(_ev(type="error", message="401 Unauthorized"), "warning: model at capacity, retrying")
    assert transient.classify_codex_failure(raw).category == "non-retryable"


def test_structured_429_beats_stderr_billing() -> None:
    raw = _stream(
        _ev(type="error", message="429 Too Many Requests: rate limit reached"),
        "billing notice",
    )
    verdict = transient.classify_codex_failure(raw)
    assert verdict.category == "transient" and verdict.source == "structured"


def test_insufficient_quota_is_billing_non_retryable() -> None:
    raw = _ev(
        type="turn.failed",
        error={"message": "429 insufficient_quota: You exceeded your current quota, please check your plan."},
    )
    verdict = transient.classify_codex_failure(raw)
    assert (verdict.category, verdict.billing) == ("non-retryable", True)


def test_code_only_and_numeric_status() -> None:
    q = transient.classify_codex_failure(
        _ev(type="turn.failed", error={"code": "insufficient_quota", "message": "429 quota exceeded"})
    )
    assert (q.category, q.billing) == ("non-retryable", True)
    cap = transient.classify_codex_failure(
        _ev(type="turn.failed", error={"code": "model_capacity_exhausted", "message": "request failed"})
    )
    assert cap.category == "transient"
    for status in (503, "503"):
        v = transient.classify_codex_failure(
            _ev(type="turn.failed", error={"message": "request failed", "status": status})
        )
        assert v.category == "transient"
    v = transient.classify_codex_failure(_ev(type="turn.failed", error={"message": "request failed", "status": 401}))
    assert v.category == "non-retryable"
    v = transient.classify_codex_failure(
        _ev(type="turn.failed", error={"status": 429, "code": "insufficient_quota"})
    )
    assert (v.category, v.billing) == ("non-retryable", True)


def test_stderr_only_error_uses_legacy_order_and_ignores_items() -> None:
    raw = _stream(
        _ev(type="item.completed", item={"text": "capacity prose"}),
        "ERROR: 401 Unauthorized",
    )
    verdict = transient.classify_codex_failure(raw)
    assert verdict.source == "stderr" and verdict.text == "ERROR: 401 Unauthorized"
    assert verdict.category == "non-retryable"


_TRUNC = '{"type":"item.completed","item":{"aggregated_output":"auth.captureOwner() invalid api key timeout'


def test_truncated_item_event_is_dropped() -> None:
    verdict = transient.classify_codex_failure(_stream(_ev(type="thread.started", thread_id="t"), _TRUNC))
    assert (verdict.source, verdict.category) == ("none", "deterministic")


def test_malformed_only_capture_is_event_shaped_but_plain_text_is_not() -> None:
    verdict = transient.classify_codex_failure(_TRUNC)
    assert verdict is not None and verdict.source == "none" and verdict.category == "deterministic"
    assert transient.extract_codex_error_channel("error: unexpected argument --foo") is None


def test_no_error_events_is_deterministic() -> None:
    verdict = transient.classify_codex_failure(_stream(_TOOL, _ev(type="turn.started")))
    assert (verdict.source, verdict.category) == ("none", "deterministic")


def test_oversized_structured_errors_keep_authority() -> None:
    big = _ev(type="turn.failed", error={"message": "x" * 30000, "status": 503})
    v = transient.classify_codex_failure(_stream(big, "ERROR: 401 Unauthorized"))
    assert (v.source, v.category) == ("structured", "transient")
    many = [_ev(type="error", message=f"warning {i}: something odd " + "y" * 900) for i in range(40)]
    many.append(_ev(type="error", message="429 insufficient_quota: check your plan"))
    v = transient.classify_codex_failure(_stream(*many))
    assert (v.category, v.billing) == ("non-retryable", True)


def test_stderr_verdict_text_keeps_decisive_middle() -> None:
    stderr = "timeout " + "n" * 7000 + " 401 Unauthorized " + "n" * 7000 + " timeout"
    v = transient.classify_codex_failure(_stream(_ev(type="thread.started", thread_id="t"), stderr))
    assert (v.source, v.category) == ("stderr", "non-retryable")
    assert "401 Unauthorized" in v.text


# ---------------------------------------------------------------------------
# Authoritative Codex structured error (#1280)
# ---------------------------------------------------------------------------

_MCP_AUTH = "MCP server failed to start: authentication required"


def test_auth_warning_then_capacity_terminal_is_transient() -> None:
    raw = _stream(
        _ev(type="error", message=_MCP_AUTH),
        _ev(type="turn.failed", error={"message": _CAP}),
    )
    verdict = transient.classify_codex_failure(raw)
    assert (verdict.source, verdict.category, verdict.billing) == ("structured", "transient", False)
    assert verdict.text.startswith(f"message: {_CAP}")
    assert _MCP_AUTH in verdict.text.splitlines()[1]


def test_billing_warning_then_503_terminal_is_transient_without_billing() -> None:
    raw = _stream(
        _ev(type="error", message="billing notice: credit balance low"),
        _ev(type="turn.failed", error={"message": "request failed", "status": 503}),
    )
    verdict = transient.classify_codex_failure(raw)
    assert (verdict.category, verdict.billing) == ("transient", False)
    assert verdict.text.startswith("message: request failed")


def test_transient_warnings_then_auth_terminal_is_non_retryable() -> None:
    raw = _stream(
        _ev(type="error", message="Reconnecting... 1/5 (429 rate limit)"),
        _ev(type="error", message="Reconnecting... 2/5 (stream disconnected)"),
        _ev(type="turn.failed", error={"message": "401 Unauthorized"}),
    )
    verdict = transient.classify_codex_failure(raw)
    assert (verdict.category, verdict.billing) == ("non-retryable", False)
    assert verdict.text.startswith("message: 401 Unauthorized")


def test_transient_warnings_then_generic_terminal_is_deterministic() -> None:
    raw = _stream(
        _ev(type="error", message="Reconnecting... 1/5 (429 rate limit)"),
        _ev(type="turn.failed", error={"message": "request failed"}),
    )
    assert transient.classify_codex_failure(raw).category == "deterministic"


def test_last_of_several_turn_failed_decides() -> None:
    raw = _stream(
        _ev(type="turn.failed", error={"message": "401 Unauthorized"}),
        _ev(type="turn.failed", error={"message": _CAP}),
    )
    channel = transient.extract_codex_error_channel(raw)
    assert channel.authoritative_index == 1
    assert transient.classify_codex_failure(raw).category == "transient"
    reverse = _stream(
        _ev(type="turn.failed", error={"message": _CAP}),
        _ev(type="turn.failed", error={"message": "401 Unauthorized"}),
    )
    assert transient.classify_codex_failure(reverse).category == "non-retryable"


def test_deduplicated_terminal_keeps_authority() -> None:
    raw = _stream(
        _ev(type="error", message=_CAP),
        _ev(type="error", message="401 Unauthorized"),
        _ev(type="turn.failed", error={"type": "error", "message": _CAP}),
    )
    channel = transient.extract_codex_error_channel(raw)
    assert len(channel.structured_errors) == 2
    assert channel.authoritative_index == 0
    verdict = transient.classify_codex_failure(raw)
    assert verdict.category == "transient" and not verdict.billing
    assert verdict.text.startswith(channel.structured_errors[0].rendered)


def test_non_equal_capacity_pair_indexes_turn_failed_entry() -> None:
    raw = _stream(
        _ev(type="error", message=_CAP),
        _ev(type="turn.failed", error={"message": _CAP}),
    )
    channel = transient.extract_codex_error_channel(raw)
    assert len(channel.structured_errors) == 2
    assert channel.authoritative_index == 1
    assert transient.classify_codex_failure(raw).category == "transient"


def test_last_error_occurrence_decides_without_turn_failed() -> None:
    raw = _stream(
        _ev(type="error", message=_CAP),
        _ev(type="error", message="401 Unauthorized"),
        _ev(type="error", message=_CAP),
    )
    channel = transient.extract_codex_error_channel(raw)
    assert len(channel.structured_errors) == 2
    assert channel.authoritative_index == 0
    verdict = transient.classify_codex_failure(raw)
    assert (verdict.category, verdict.billing) == ("transient", False)
    assert verdict.text.startswith(channel.structured_errors[0].rendered)


def test_auth_warning_then_final_capacity_error_without_turn_failed() -> None:
    raw = _stream(_ev(type="error", message=_MCP_AUTH), _ev(type="error", message=_CAP))
    channel = transient.extract_codex_error_channel(raw)
    assert channel.authoritative_index == 1
    assert transient.classify_codex_failure(raw).category == "transient"


def test_authoritative_index_is_none_without_structured_errors() -> None:
    channel = transient.extract_codex_error_channel(_stream(_TOOL, _ev(type="turn.started")))
    assert channel.structured_errors == () and channel.authoritative_index is None


def test_agent_failure_uses_authoritative_codex_error() -> None:
    from coding_review_agent_loop import agent_failure
    from coding_review_agent_loop.agents.base import AgentResult

    raw = _stream(
        _ev(type="thread.started", thread_id="t"),
        _ev(type="error", message=_MCP_AUTH),
        _ev(type="turn.failed", error={"message": _CAP}),
    )
    result = AgentResult(text="", raw_output=raw, returncode=1, provider="codex")
    text, verdict = agent_failure._agent_failure_classification(result, phase="command")
    assert verdict is not None and verdict.category == "transient"
    assert agent_failure._failure_category(text, provider_verdict=verdict) == "transient"


# --- Antigravity credits-balance exhaustion (#1598) ---------------------------

from coding_review_agent_loop.config import DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES  # noqa: E402

_CREDITS_LINE = "error: Your AI credits balance is too low to continue."
_CREDITS_AGY_ERROR = (
    'AGY_ERROR: {"short_error":"RESOURCE_EXHAUSTED (code 429): Resource has been exhausted '
    '(e.g. check quota).","status":"RESOURCE_EXHAUSTED","error_code":429,'
    '"code_kind":"http","retryable":true}'
)
CREDITS_ISSUE_OUTPUT = f"{_CREDITS_LINE}\n{_CREDITS_AGY_ERROR}"
_RETRYABLE_429 = (
    'AGY_ERROR: {"status":"RESOURCE_EXHAUSTED","error_code":429,'
    '"message":"Resource has been exhausted","retryable":true}'
)


def _agy(text: str, *, returncode: int = 1) -> transient.AntigravityCapacityClassification:
    return transient.classify_antigravity_capacity(
        text,
        returncode=returncode,
        empty_response=False,
        signatures=DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES,
    )


def test_antigravity_credits_issue_output_is_capacity_with_cooldown() -> None:
    capacity = _agy(CREDITS_ISSUE_OUTPUT)
    assert capacity.is_capacity and capacity.reason == "credits"
    assert not capacity.auth_failure
    assert "credits balance" in capacity.frame and "RESOURCE_EXHAUSTED" in capacity.frame
    quota = transient.classify_antigravity_quota_exhaustion(capacity)
    assert quota is not None
    assert (quota.reset_seconds, quota.reset_source) == (None, "cooldown")


def test_antigravity_credits_with_429_text_frame_but_no_json_is_capacity() -> None:
    capacity = _agy(
        f"{_CREDITS_LINE}\nAGY_ERROR: {{RESOURCE_EXHAUSTED (code 429) truncated"
    )
    assert capacity.is_capacity and capacity.reason == "credits"


def test_antigravity_auth_beats_credits_exemption() -> None:
    for text in (
        f"{_CREDITS_LINE}\n"
        'AGY_ERROR: {"status":"UNAUTHENTICATED","error_code":401,"retryable":true}',
        f"{CREDITS_ISSUE_OUTPUT}\ninvalid api key",
        "error: invalid api key",
    ):
        capacity = _agy(text)
        assert not capacity.is_capacity and capacity.auth_failure, text
        assert transient.classify_antigravity_quota_exhaustion(capacity) is None


def test_antigravity_terminal_numeric_auth_frames_without_auth_words() -> None:
    for text in (
        'AGY_ERROR: {"status":"X","error_code":401,"message":"nope"}',
        'AGY_ERROR: {"status":"X","error_code":403,"message":"nope"}',
        'AGY_ERROR: {"error":{"code":"403","message":"nope"}}',
        "Error: request failed (code 401), timed out after 429 retries",
        "Error: HTTP 403 while contacting provider",
    ):
        capacity = _agy(text)
        assert capacity.auth_failure and not capacity.is_capacity, text


def test_antigravity_nonterminal_status_403_json_does_not_veto_capacity() -> None:
    text = "\n".join(
        [
            '{"message":"Resource not accessible","status":"403"}',
            "Reading the diff to draft the review...",
            "Checking tests next.",
            "Still drafting.",
            "Almost there.",
            "Error: RESOURCE_EXHAUSTED",
            _RETRYABLE_429,
        ]
    )
    capacity = _agy(text)
    assert capacity.is_capacity
    assert not capacity.auth_failure
    assert capacity.reason == "capacity"


def test_antigravity_dirty_checkout_beats_credits_exemption() -> None:
    capacity = _agy(f"Error: dirty checkout\n{CREDITS_ISSUE_OUTPUT}")
    assert not capacity.is_capacity
    assert not capacity.auth_failure
    assert transient.classify_antigravity_quota_exhaustion(capacity) is None


def test_antigravity_credits_without_retryable_frame_is_not_capacity() -> None:
    for text in (
        _CREDITS_LINE,
        "error: billing problem on your account\n"
        'AGY_ERROR: {"status":"FAILED_PRECONDITION","error_code":400,"retryable":false}',
        "error: billing problem on your account",
    ):
        capacity = _agy(text)
        assert not capacity.is_capacity and not capacity.auth_failure, text


def test_antigravity_transcript_only_credits_wording_gets_no_exemption() -> None:
    text = "\n".join(
        [
            "The issue quotes: Your AI credits balance is too low to continue.",
            "Reviewing the classifier next.",
            "Error: high traffic",
            _RETRYABLE_429,
        ]
    )
    capacity = _agy(text)
    assert capacity.reason != "credits"
    assert not capacity.is_capacity
    assert transient.classify_antigravity_quota_exhaustion(capacity) is None


def test_antigravity_credits_correlated_with_capacity_only_wording_is_rejected() -> None:
    capacity = _agy(f"error: credits balance too low; high traffic\n{_RETRYABLE_429}")
    assert not capacity.is_capacity


def test_antigravity_non_retryable_partition_matches_public_regex() -> None:
    samples = (
        "auth failed", "authentication required", "authorization denied", "unauthorized",
        "forbidden", "invalid api key", "AI credits balance", "credit", "billing issue",
        "dirty checkout", "dirty workdir", "dirty working tree",
        "the plan looks good", "429 rate limit", "author wrote", "authored by",
    )
    partition = (
        transient._ANTIGRAVITY_AUTH_RE,
        transient._ANTIGRAVITY_DIRTY_CHECKOUT_RE,
        transient._ANTIGRAVITY_BILLING_RE,
    )
    for sample in samples:
        assert bool(transient.NON_RETRYABLE_AGENT_OUTPUT_RE.search(sample)) == any(
            pattern.search(sample) for pattern in partition
        ), sample


def test_non_antigravity_credit_and_billing_classification_unchanged() -> None:
    assert not transient.is_transient_agent_output(CREDITS_ISSUE_OUTPUT)
    assert not transient.is_transient_agent_output("billing problem: 429 rate limit")
    assert not transient.is_transient_agent_output("your credit card was declined; timed out")
    # No Codex event stream: no structured verdict, so the text classifier decides.
    assert transient.classify_codex_failure(CREDITS_ISSUE_OUTPUT) is None
    from coding_review_agent_loop.agent_failure import _failure_category

    assert _failure_category(CREDITS_ISSUE_OUTPUT) == "non-retryable"
    billing = transient.classify_codex_failure(
        _ev(type="turn.failed", error={"message": "billing hard limit reached"})
    )
    assert (billing.category, billing.billing) == ("non-retryable", True)
