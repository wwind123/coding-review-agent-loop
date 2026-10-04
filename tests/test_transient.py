"""Unit tests for the lightweight transient-output classifier."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coding_review_agent_loop import orchestrator, transient


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
    assert orchestrator._is_transient_agent_output is transient.is_transient_agent_output


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
