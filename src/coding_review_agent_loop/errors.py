"""Shared exceptions for the agent loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .containment import ContainmentEvidence


class AgentLoopError(RuntimeError):
    """Raised for expected orchestration failures."""


class UnknownPriorItemDispositionError(AgentLoopError):
    """Raised when an agent dispositions a non-carried prior item ID."""

    def __init__(
        self,
        *,
        unknown_ids: tuple[str, ...],
        allowed_ids: tuple[str, ...],
        same_round_description: str,
    ) -> None:
        self.unknown_ids = tuple(unknown_ids)
        self.allowed_ids = tuple(allowed_ids)
        self.same_round_description = same_round_description
        message = (
            f"Unknown prior-item disposition ID(s) {sorted(unknown_ids)!r}; "
            f"allowed carried prior IDs: {sorted(allowed_ids) or '(none)'}; "
            f"{same_round_description}"
        )
        super().__init__(message)


class IssueImplementationConflictError(AgentLoopError):
    """A parsed implementation result cannot be handed off as reported.

    A coder may discover a signed human requirement is blocked after opening a
    PR.  The payload is still valuable operator evidence, but a positive PR
    identity combined with a blocked requirement is not an accepted
    implementation result.  Retain the parsed payload so the orchestrator can
    publish that terminal conflict without retrying or entering PR gates.
    """

    def __init__(self, payload: object) -> None:
        self.payload = payload
        super().__init__(
            "issue_implementation cannot report a positive PR while a signed "
            "human requirement is blocked."
        )


class AgentInvocationError(AgentLoopError):
    """Raised when an agent invocation fails after retries/repair.

    Carries the failure category (`transient`, `non-retryable`,
    `unsupported_model`, `deterministic`, `timeout`, ...) so callers such as
    the discuss debater failure policy can surface it in summaries and metadata
    without re-parsing the message.
    """

    def __init__(
        self,
        message: str,
        *,
        failure_category: str | None = None,
        terminal_public_response: str | None = None,
        containment: "ContainmentEvidence | None" = None,
    ) -> None:
        super().__init__(message)
        self.failure_category = failure_category
        # Protocol-valid text (agent-declared AGENT_UNAVAILABLE text verbatim,
        # or an orchestrator-synthesized rendering) already persisted/posted by
        # a bounded completion-recovery attempt (#588). None for every other
        # failure path, which is unchanged.
        self.terminal_public_response = terminal_public_response
        self.containment = containment


class QuotaResetExceededError(AgentLoopError):
    """Raised when a rate-limit reset time exceeds the auto-retry threshold.

    Exit code 3 distinguishes "quota exhausted, retry later" from
    "something is broken, fix it first" (exit code 1).
    """

    EXIT_CODE = 3
