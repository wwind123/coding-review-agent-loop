"""Shared exceptions for the agent loop."""

from __future__ import annotations


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


class AgentInvocationError(AgentLoopError):
    """Raised when an agent invocation fails after retries/repair.

    Carries the failure category (`transient`, `non-retryable`,
    `unsupported_model`, `deterministic`, `timeout`, ...) so callers such as
    the discuss debater failure policy can surface it in summaries and metadata
    without re-parsing the message.
    """

    def __init__(self, message: str, *, failure_category: str | None = None) -> None:
        super().__init__(message)
        self.failure_category = failure_category


class QuotaResetExceededError(AgentLoopError):
    """Raised when a rate-limit reset time exceeds the auto-retry threshold.

    Exit code 3 distinguishes "quota exhausted, retry later" from
    "something is broken, fix it first" (exit code 1).
    """

    EXIT_CODE = 3
