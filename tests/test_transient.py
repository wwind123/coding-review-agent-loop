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


def test_orchestrator_alias_preserves_identity() -> None:
    # orchestrator keeps the old private name as an alias to the moved implementation.
    assert orchestrator._is_transient_agent_output is transient.is_transient_agent_output
