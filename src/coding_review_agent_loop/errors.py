"""Shared exceptions for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .containment import ContainmentEvidence


class AgentLoopError(RuntimeError):
    """Raised for expected orchestration failures."""


class HumanDecisionRequiredError(AgentLoopError):
    """The run reached a deliberate operator decision boundary."""

    EXIT_CODE = 4


class WorkflowTransactionError(AgentLoopError):
    """A cross-surface workflow transaction is missing, partial, or contradictory (#827).

    The message always names the transaction(s), the expected record set, each
    offending record, and the one supported recovery action, so an operator can
    act without reading protocol payloads.
    """

    def __init__(
        self,
        summary: str,
        *,
        transaction_ids: tuple[str, ...] = (),
        successor_kind: str | None = None,
        expected_record_set: tuple[str, ...] = (),
        problems: tuple[str, ...] = (),
        recovery_action: str = "rerun to finish the transaction",
        code: str | None = None,
    ) -> None:
        self.summary = summary
        self.transaction_ids = tuple(transaction_ids)
        self.successor_kind = successor_kind
        self.expected_record_set = tuple(expected_record_set)
        self.problems = tuple(problems)
        self.recovery_action = recovery_action
        self.code = code
        parts = [summary.rstrip(".") + "."]
        if code:
            parts.append(f"Code: {code}.")
        parts.append(
            "Transaction(s): " + (", ".join(self.transaction_ids) or "(none)") + "."
        )
        if successor_kind:
            parts.append(f"Successor kind: {successor_kind}.")
        parts.append(
            "Expected record set: " + (", ".join(self.expected_record_set) or "(none)") + "."
        )
        if self.problems:
            parts.append("Records: " + "; ".join(self.problems) + ".")
        parts.append(f"Recovery: {recovery_action}.")
        super().__init__(" ".join(parts))


class FreshContractIntegrityError(AgentLoopError):
    """A fresh plan cannot be format-repaired without recoverable v1 data."""


class ReviewSubstanceIntegrityError(AgentLoopError):
    """A reviewer turn carries no recoverable review payload to repair.

    Repair is lossless format recovery.  When a reviewer response is narration,
    diagnostics, or a bare protocol state footer, there is no verdict or finding
    for a repair model to recover, and synthesizing one would post a fabricated
    review on the reviewer's behalf.  Refuse instead, fail-closed.
    """


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


class SemanticPatchPayloadRejection(AgentLoopError):
    """A semantic-patch rejection that names content inside the patch payload.

    Repair preserves a ``plan_revision_patch`` payload exactly and may change
    only its envelope/footer, so no repair output can satisfy this rejection.
    It is routed to the planner's bounded replan instead (#979).
    """


class SemanticPatchUnknownPriorItemDispositionError(
    UnknownPriorItemDispositionError, SemanticPatchPayloadRejection
):
    """An unknown prior-item disposition inside a semantic patch payload."""


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


@dataclass(frozen=True)
class DeterministicPlanValidationExhaustion:
    """The final rejected structured planning candidate after retry exhaustion.

    The candidate is retained only in memory so the orchestrator can compute
    provenance.  Durable recovery stores the digest and bounded validator
    diagnostic, never this raw candidate.
    """

    candidate_kind: str
    candidate_text: str
    diagnostic: str
    candidate_digest: str


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
        plan_validation_exhaustion: DeterministicPlanValidationExhaustion | None = None,
        bounded_replan_rejection: DeterministicPlanValidationExhaustion | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_category = failure_category
        # Protocol-valid text (agent-declared AGENT_UNAVAILABLE text verbatim,
        # or an orchestrator-synthesized rendering) already persisted/posted by
        # a bounded completion-recovery attempt (#588). None for every other
        # failure path, which is unchanged.
        self.terminal_public_response = terminal_public_response
        self.containment = containment
        self.plan_validation_exhaustion = plan_validation_exhaustion
        # A semantic-patch payload rejection that repair could never satisfy
        # (#979). The planner's bounded replan consumes it as a diagnostic.
        self.bounded_replan_rejection = bounded_replan_rejection


class QuotaResetExceededError(AgentLoopError):
    """Raised when a rate-limit reset time exceeds the auto-retry threshold.

    Exit code 3 distinguishes "quota exhausted, retry later" from
    "something is broken, fix it first" (exit code 1).
    """

    EXIT_CODE = 3
