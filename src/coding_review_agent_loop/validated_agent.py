"""Run telemetry, same-agent re-asks, completion recovery and the validated agent turn.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1192); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import datetime
import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace as dataclasses_replace
from pathlib import Path

from .agents.base import AgentName, AgentResult, response_turn_scope
from .agents.antigravity import AntigravityAttemptState
from .agents.antigravity_structured import antigravity_structured_kind_for_turn
from .agents.registry import agent_display_name, agent_signature, run_agent_result
from .config import AgentLoopConfig
from .errors import (
    CITATION_REASKABLE_REASONS,
    TESTS_RUN_WORKDIR_REASON,
    AgentInvocationError,
    AgentLoopError,
    AgentSettingsLockWaitTimeout,
    DeterministicPlanValidationExhaustion,
    PreservedUnsatisfiedResponse,
    NonRepairableEvidenceRejection,
    QuotaResetExceededError,
    RejectedCandidate,
    SemanticPatchPayloadRejection,
    MissingJudgementFieldError,
    MissingPriorItemDispositionError,
    TestObservationsShapeError,
    UnknownPriorItemDispositionError,
)
from .github import (
    IssueContext,
    HumanReviewRequirement,
    post_issue_comment,
    reset_host_footer_log_latch,
)
from . import tool_provenance
from .logging import log, new_run_id, run_usage_summary_path
from .managed_ci_protection import (
    waivable_protection_states,
    waiver_flags_for_protection,
)
from .prompts import build_completion_recovery_prompt
from .protocol_risk import (
    TEST_OBSERVATIONS_NOT_ARRAY_AFTER_REPAIR_RULE,
)
from .protocol import (
    AgentUnavailable,
    parse_agent_unavailable,
)
from .protocol_common import (
    ParseDegradation,
)
from .response_normalization import (
    attempt_envelope_normalization,
    attempt_risk_test_matrix_string_list_normalization,
    attempt_semantic_patch_disposition_normalization,
    recover_payload,
    review_substance_missing,
    strip_unknown_prior_item_dispositions,
    unknown_dispositions_are_resolved_history,
)
from .response_reask import (
    MAX_SAME_AGENT_REASKS,
    build_reask_prompt,
    sanitize_validator_message,
)
from .runner import Runner
from .salvage import SalvageContext
from .transient import (
    antigravity_log_quota_capacity,
    classify_antigravity_capacity,
    classify_antigravity_quota_exhaustion,
    looks_like_backgrounded_completion,
)
from .usage import RunUsageContext, UsageMetadata, estimate_usage
from .worker_telemetry import append_record, run_record, telemetry_log_path
from .workdirs import active_workdir
from .workdir_guard import GitProbeResult, read_workdir_head
from .comment_rendering import render_agent_unavailable_comment
from . import quota_wait
from .round_metadata import (
    ApprovedPlanContext,
)
from .agent_failure import (
    STRUCTURED_PUBLIC_RESPONSE_KINDS,
    LONG_RESET_THRESHOLD_SECONDS,
    _PROVIDER_DEFINITIVE_FAILURE_CATEGORIES,
    _QUOTA_RATE_LIMIT_RE,
    ValidatedAgentResponse,
    _response_identity_fields,
    _AgentUnavailableResponse,
    _parse_rate_limit_reset_seconds,
    _format_reset_duration,
    _format_reset_at_utc,
    _is_transient_agent_output,
    _recognized_structured_public_response_kind,
    _is_transient_public_response,
    _is_retryable_marker_near_miss,
    _SEMANTIC_EVIDENCE_REJECTION_CLASSIFICATION,
    _JUDGEMENT_FIELD_CLASSIFICATION,
    _failure_category,
    _response_file_structured_status,
    _neutralize_untrusted_markers,
    _recover_valid_structured_candidate,
    _HumanRequirementsRecoveryContext,
    _plan_revision_missing_human_acknowledgement,
    _recover_plan_revision_human_requirements_acknowledgement,
    _retry_delay,
    _executable_replacement_failure_detail,
    _format_invalid_agent_response_error,
    _operation_description_from_context,
    _failed_run_diagnostics,
    _agent_failure_classification,
    _agent_failure_classification_text,
)
from .architecture_contract import (
    _unwrap_architecture_result,
    _architecture_contract_diagnostic,
    _attach_architecture_degradations,
    _AcceptedCandidate,
    _AcceptedTextCanonicalizationError,
    _with_validation_context,
    _accept_candidate,
    _accepted_validated_response,
    _architecture_contract_retry_prompt,
    _architecture_document_path_diagnostic,
    _architecture_document_path_rejection,
    _architecture_document_path_retry_prompt,
    _reported_canonical_document_path,
    _evidence_rejection_reask_prompt,
    _prior_disposition_omission_reask_prompt,
    _missing_judgement_field_reask_prompt,
    _ArchitectureImpactContractUnsatisfied,
)


def _new_usage_context(config: AgentLoopConfig) -> RunUsageContext:
    # Invariant: only the run entries that own a usage context call this
    # (run_issue_loop, run_task_loop, run_pr_loop, run_discuss_loop), so each
    # owning invocation starts a fresh host-footer log latch, while a nested
    # run that received a usage context shares the outer latch (#1043).
    reset_host_footer_log_latch()
    run_id = new_run_id()
    return RunUsageContext(
        run_id=run_id,
        summary_path=run_usage_summary_path(config, run_id),
        tool_provenance=tool_provenance.process_provenance(config),
        run_started_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )


def _begin_run_telemetry(
    runner: Runner,
    config: AgentLoopConfig,
    usage_context: RunUsageContext,
    owned: bool,
    *,
    issue_number: int | None = None,
    pr_number: int | None = None,
) -> tuple[bool, dict | None, dict | None]:
    """Attribute reservation telemetry to this run (#1107); never raises.

    An owning loop opens the run window.  A nested loop that received a usage
    context only adds its PR number to the outer attribution.  Returns a token
    for ``_end_run_telemetry``.
    """
    previous = getattr(runner, "telemetry_attribution", None)
    try:
        if owned:
            attribution = {
                "repo": config.repo,
                "run_id": usage_context.run_id,
                "issue_number": issue_number,
                "pr_number": pr_number,
            }
            runner.telemetry_attribution = attribution
            append_record(telemetry_log_path(), run_record("run-start", attribution))
            return (True, previous, attribution)
        if previous is not None and pr_number is not None:
            runner.telemetry_attribution = {**previous, "pr_number": pr_number}
    except Exception:
        pass
    return (False, previous, None)


def _end_run_telemetry(runner: Runner, token: tuple[bool, dict | None, dict | None]) -> None:
    owned, previous, attribution = token
    try:
        if owned and attribution is not None:
            append_record(telemetry_log_path(), run_record("run-end", attribution))
        runner.telemetry_attribution = previous
    except Exception:
        pass


def _resolve_usage_metadata(
    *,
    config: AgentLoopConfig,
    prompt: str,
    result: AgentResult,
) -> UsageMetadata | None:
    if result.usage is not None:
        return result.usage.with_io_sizes(prompt=prompt, response=result.text)
    if config.dry_run:
        return None
    return estimate_usage(prompt, result.text)


def _persist_usage_summary(config: AgentLoopConfig, usage_context: RunUsageContext) -> None:
    usage_context.write_summary()
    totals = usage_context.totals()
    if config.review_images_paths or any(record.visual_kind is not None for record in usage_context.records):
        log(config, usage_context.visual_summary_line())
    log(
        config,
        "Usage summary written to "
        f"{usage_context.summary_path} "
        f"(calls={totals.call_count}, exact={totals.exact_calls}, "
        f"partial={totals.partial_calls}, estimated={totals.estimated_calls}, "
        f"{tool_provenance.format_tool_commit_suffix(usage_context.tool_provenance)})",
    )


# A NonRepairableEvidenceRejection whose old observations are no longer
# citable: the coder re-ask may rerun the covering test, and its dispatch
# becomes the active evidence acquisition (#1240, #2066).  A refused
# ``tests_run`` entry is a resend-only correction.
_FRESH_EVIDENCE_REASONS = CITATION_REASKABLE_REASONS - {TESTS_RUN_WORKDIR_REASON}
_CODER_EXPECTED_KINDS = frozenset({"coder_followup", "issue_implementation"})
REASK_TERMINAL_PREFIX = "response rejected after {count} same-agent re-ask{plural}"


@dataclass(frozen=True)
class _PendingReask:
    """One scheduled same-agent re-ask (#2066)."""

    number: int
    prompt: str
    session_id: str | None
    fresh_evidence: bool
    # The coder checkout head read before the re-ask; None for other roles.
    head_before: str | None
    # The model the rejected response came from, when it differs from the
    # configured one, so the re-ask goes to that same model.
    producer_model: str | None = None


_MODEL_FIELDS_BY_BACKEND = {
    "claude": ("claude_model", "reviewer_claude_model"),
    "codex": ("codex_model", "reviewer_codex_model"),
    "gemini": ("gemini_model",),
}
_ARGS_FIELD_BY_BACKEND = {"claude": "claude_args", "codex": "codex_args", "gemini": "gemini_args"}


def _without_model_args(args: tuple[str, ...]) -> tuple[str, ...]:
    """Drop ``--model VALUE`` and ``--model=VALUE`` from passthrough arguments."""
    kept: list[str] = []
    skip_value = False
    for arg in args:
        if skip_value:
            skip_value = False
            continue
        if arg == "--model":
            skip_value = True
            continue
        if arg.startswith("--model="):
            continue
        kept.append(arg)
    return tuple(kept)


def _pin_producer_model(
    agent: AgentName, config: AgentLoopConfig, backend: str, model: str | None
) -> tuple[AgentName, AgentLoopConfig]:
    """Route a re-ask to the model that produced the rejected response (#2066).

    Antigravity keeps its chain position across re-asks, so it needs no pin.
    A named seat carries its own model chain, so the seat is re-issued with
    the producer model first; a plain provider gets its model fields pinned.
    A model passed through the provider's agent arguments (``--model VALUE``
    or ``--model=VALUE``) is replaced by the pin, keeping every unrelated
    argument, so the pin is never silently abandoned.
    """
    from .reviewer_seats import SeatAgent

    if not model or backend not in _MODEL_FIELDS_BY_BACKEND:
        return agent, config
    if isinstance(agent, SeatAgent):
        pinned = str.__new__(SeatAgent, str(agent))
        pinned.backend = agent.backend
        pinned.model_chain = (model, *(m for m in agent.model_chain if m != model))
        pinned.effort = agent.effort
        pinned.workdir = agent.workdir
        return pinned, config
    args_field = _ARGS_FIELD_BY_BACKEND[backend]
    updates: dict[str, object] = {field: model for field in _MODEL_FIELDS_BY_BACKEND[backend]}
    updates[args_field] = _without_model_args(getattr(config, args_field))
    return agent, dataclasses_replace(config, **updates)


def _read_reask_head(runner: Runner, config: AgentLoopConfig) -> GitProbeResult:
    """Read the assigned checkout head for the coder re-ask guard."""
    return read_workdir_head(runner, active_workdir(config), tolerate_exceptions=True)


def _reask_hint_prompt(
    prompt: str,
    exc: AgentLoopError,
    *,
    contract_unsatisfied: bool,
    document_path_rejection: tuple[str, frozenset[str]] | None,
) -> str:
    """The original prompt plus the typed hint section for ``exc``, if it has one.

    Missing carried-item IDs, a missing judgement field with its allowed
    values, the evidence rejection reason and the architecture contract field
    keep their bounded, marker-free sections; the generic re-ask appendix is
    added by :func:`build_reask_prompt`.
    """
    if contract_unsatisfied:
        if document_path_rejection is not None:
            return _architecture_document_path_retry_prompt(prompt, *document_path_rejection)
        return _architecture_contract_retry_prompt(prompt, str(exc))
    if isinstance(exc, MissingJudgementFieldError):
        return _missing_judgement_field_reask_prompt(
            prompt, exc.field_path, exc.allowed_values, exc.observed_preview
        )
    if isinstance(exc, MissingPriorItemDispositionError):
        return _prior_disposition_omission_reask_prompt(prompt, tuple(exc.missing_ids))
    if (
        isinstance(exc, NonRepairableEvidenceRejection)
        and exc.reason in CITATION_REASKABLE_REASONS
    ):
        return _evidence_rejection_reask_prompt(prompt, str(exc), reason=exc.reason)
    return prompt


def _evidence_reask_log_purpose(reason: str) -> str:
    """What the coder is re-asked to do, by rejection reason (#2052)."""
    if reason == TESTS_RUN_WORKDIR_REASON:
        return "correct its tests_run report"
    if reason in _FRESH_EVIDENCE_REASONS:
        return "cite a verified passing observation"
    return "resend its response"


def _semantic_evidence_last_error(rejection: NonRepairableEvidenceRejection) -> str:
    return f"{rejection} (semantic evidence rejection)"


@dataclass(frozen=True)
class CompletionRecoveryPolicy:
    """Explicit opt-in for the bounded same-session completion-recovery pass (#588).

    Passed only by the direct issue-implementation call sites that validate
    structured `issue_implementation` results; every other
    `_run_validated_agent` caller (planning, plan/PR review, discuss, task,
    and the coder follow-up/PR loop) leaves this `None` and is therefore
    ineligible by construction -- eligibility is never inferred from the
    agent name or response text alone.
    """

    issue_number: int
    issue_context: IssueContext | None = None
    approved_plan_context: ApprovedPlanContext | None = None
    parent_issue_context: IssueContext | None = None
    human_requirements: tuple[HumanReviewRequirement, ...] | None = None


@dataclass(frozen=True)
class _CompletionRecoveryOutcome:
    validated: ValidatedAgentResponse | None
    result: AgentResult
    error: str
    classification_text: str
    failure_category: str
    # Protocol-valid text already persisted to the recovery attempt's own
    # response file and posted to the GitHub issue; set only when validated
    # is None.
    terminal_public_response: str | None
    # The resumed response's only defect is the required architecture-impact
    # contract: the caller takes the ordinary deterministic retry (#925).
    contract_unsatisfied: bool = False
    # An otherwise valid response whose accepted text could not be
    # canonicalized; never reported as an unsatisfied contract.
    canonicalization_failed: bool = False


def _post_completion_recovery_terminal_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    completion_recovery: CompletionRecoveryPolicy,
    recovery_result: AgentResult,
    terminal_text: str,
) -> None:
    if recovery_result.response_file_path is not None:
        recovery_result.response_file_path.write_text(terminal_text, encoding="utf-8")
    post_issue_comment(
        runner,
        config=config,
        issue_number=completion_recovery.issue_number,
        body=terminal_text,
    )


def _synthesized_completion_recovery_unavailable(
    *, config: AgentLoopConfig, recovery_result: AgentResult, category: str, summary: str
) -> tuple[AgentUnavailable, str]:
    unavailable = AgentUnavailable(
        schema_version=1,
        kind="agent_unavailable",
        retryable=False,
        category=category,
        summary=summary,
        suggested_action=(
            "Inspect the completion-recovery log and salvage artifacts, "
            "then retry the implementation manually."
        ),
    )
    signature = agent_signature("claude", config, model_used=recovery_result.model_used)
    return unavailable, render_agent_unavailable_comment(unavailable, signature=signature)


def _attempt_claude_completion_recovery(*args, **kwargs):
    """Run completion recovery inside a response-turn scope.

    The recovery writes its terminal text to the recovery attempt's response
    file after the backend returns, so its per-turn directory must outlive
    the backend invocation: inside a validated turn this defers to that
    turn's scope, and a direct call gets its own (#1560).
    """
    config = kwargs.get("config")

    def report(message: str) -> None:
        if config is not None:
            log(config, message)

    with response_turn_scope(on_error=report, reuse_enclosing=True):
        return _attempt_claude_completion_recovery_in_scope(*args, **kwargs)


def _attempt_claude_completion_recovery_in_scope(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    completion_recovery: CompletionRecoveryPolicy,
    session_id: str,
    validate: Callable[[str], object],
    usage_context: RunUsageContext | None,
    run_id: str | None,
    role: str | None,
    label: str | None,
    timeout_seconds: float | None,
    acquisition_result: AgentResult | None = None,
    accept: Callable[[str, object], _AcceptedCandidate] | None = None,
) -> _CompletionRecoveryOutcome:
    """One bounded ``claude --resume`` completion-recovery pass (#588).

    Entirely self-contained and attempt-local: every variable here is scoped
    to this one recovery call, never the caller's original (failed) attempt.
    The caller's original ``AgentResult`` is not even passed in, so isolation
    is structural rather than merely asserted. Four outcomes:

    - a valid PR/blocking/clarify per ``validate()`` -> success, returned as
      a normal ``ValidatedAgentResponse`` subject to ordinary PR validation;
      this includes a current invocation response-file artifact that validates
      after a timeout or nonzero exit, with its acquisition diagnostic retained;
    - the resume turn itself declares a protocol-valid ``AGENT_UNAVAILABLE``
      -> terminal, posted/persisted verbatim, never passed to ``validate()``,
      and always terminal regardless of its own ``retryable`` flag (the
      bounded one-recovery-attempt policy overrides the agent's preference);
    - transport failures, or text that still fails ``validate()`` -> terminal
      with a synthesized non-retryable ``AGENT_UNAVAILABLE`` rendered,
      persisted, and posted; except a second background-wait-only response,
      which is a deterministic protocol failure rather than an operational
      unavailability.

    In every terminal case there is exactly one ``--resume`` call total.
    An unsatisfied required architecture-impact contract is not terminal: it
    returns a deterministic ``contract_unsatisfied`` outcome for the caller's
    ordinary retry, with no comment and no synthesized unavailability.
    """
    if accept is None:
        def accept(text: str, marker_value: object) -> _AcceptedCandidate:
            return _AcceptedCandidate(text, marker_value)

    def _contract_unsatisfied(exc: AgentLoopError, candidate: str) -> _CompletionRecoveryOutcome:
        return _CompletionRecoveryOutcome(
            validated=None, result=recovery_result, error=str(exc),
            classification_text=candidate, failure_category="deterministic",
            terminal_public_response=None, contract_unsatisfied=True,
        )

    def _canonicalization_failed(exc: AgentLoopError, candidate: str) -> _CompletionRecoveryOutcome:
        return _CompletionRecoveryOutcome(
            validated=None, result=recovery_result, error=str(exc),
            classification_text=candidate, failure_category="deterministic",
            terminal_public_response=None, canonicalization_failed=True,
        )

    recovery_prompt = build_completion_recovery_prompt(
        config,
        issue_context=completion_recovery.issue_context,
        approved_plan_context=completion_recovery.approved_plan_context,
        parent_issue_context=completion_recovery.parent_issue_context,
        human_requirements=completion_recovery.human_requirements,
    )
    recovery_result = run_agent_result(
        runner,
        agent="claude",
        config=config,
        prompt=recovery_prompt,
        session_id=session_id,
        run_id=run_id,
        role=role,
        label=label,
        timeout_seconds=timeout_seconds,
    )
    recovery_usage_record = None
    if usage_context is not None:
        recovery_usage = _resolve_usage_metadata(
            config=config, prompt=recovery_prompt, result=recovery_result
        )
        if recovery_usage is not None:
            recovery_usage_record = usage_context.add_record(
                agent="claude",
                session_id=recovery_result.session_id,
                returncode=recovery_result.returncode,
                usage=recovery_usage,
                raw_backend_usage=recovery_result.raw_usage,
                role="completion-recovery",
                turn_role=role,
                model=recovery_result.model_used,
                configured_model=recovery_result.configured_model,
                configured_effort=recovery_result.configured_effort,
                effort_source=recovery_result.effort_source,
                observed_model=recovery_result.observed_model,
                observed_effort=recovery_result.observed_effort,
                observation_provenance=recovery_result.observation_provenance,
            )
    else:
        recovery_usage = None

    def _terminal(category: str, summary: str, *, error: str) -> _CompletionRecoveryOutcome:
        _unavailable, rendered = _synthesized_completion_recovery_unavailable(
            config=config, recovery_result=recovery_result, category=category, summary=summary
        )
        _post_completion_recovery_terminal_comment(
            runner,
            config=config,
            completion_recovery=completion_recovery,
            recovery_result=recovery_result,
            terminal_text=rendered,
        )
        return _CompletionRecoveryOutcome(
            validated=None,
            result=recovery_result,
            error=error,
            classification_text=recovery_result.raw_output or recovery_result.text,
            failure_category="agent-unavailable",
            terminal_public_response=rendered,
        )

    # A response-file artifact is authoritative for this invocation even when
    # the CLI reports a failed exit.  Do not consult stdout here: it may only
    # contain diagnostics.  Invalid artifacts intentionally fall through to
    # the existing terminal transport handling below.
    recovery_artifact = recovery_result.response_file_text
    if recovery_artifact:
        try:
            unavailable = parse_agent_unavailable(recovery_artifact)
        except AgentLoopError:
            unavailable = None
        if unavailable is not None:
            _post_completion_recovery_terminal_comment(
                runner, config=config, completion_recovery=completion_recovery,
                recovery_result=recovery_result, terminal_text=recovery_artifact,
            )
            return _CompletionRecoveryOutcome(
                validated=None, result=recovery_result,
                error=("agent explicitly reported it cannot continue after completion "
                       f"recovery ({unavailable.category}): {unavailable.summary}"),
                classification_text=recovery_artifact, failure_category="agent-unavailable",
                terminal_public_response=recovery_artifact,
            )
        try:
            marker_value = validate(recovery_artifact)
            accepted_artifact = accept(recovery_artifact, marker_value)
        except _ArchitectureImpactContractUnsatisfied as exc:
            # The artifact is authoritative for this invocation; do not fall
            # through to the transport checks.
            return _contract_unsatisfied(exc, recovery_artifact)
        except _AcceptedTextCanonicalizationError as exc:
            return _canonicalization_failed(exc, recovery_artifact)
        except AgentLoopError:
            pass
        else:
            accepted_outcome = (
                "accepted_timeout" if recovery_result.returncode is None
                else "accepted_nonzero_exit" if recovery_result.returncode != 0 else "success"
            )
            if recovery_usage_record is not None:
                recovery_usage_record.validation_status = "validated"
                recovery_usage_record.outcome = accepted_outcome
            if accepted_outcome != "success":
                log(config, "claude completion-recovery accepted a valid response-file artifact "
                    f"despite returncode={recovery_result.returncode!r}")
            return _CompletionRecoveryOutcome(
                validated=_accepted_validated_response(
                    accepted_artifact, session_id=recovery_result.session_id,
                    usage=recovery_usage,
                    model_used=recovery_result.model_used,
                    **_response_identity_fields(
                        recovery_result, acquisition_result=acquisition_result
                    ),
                    acquisition_outcome=accepted_outcome,
                    acquisition_returncode=recovery_result.returncode,
                ),
                result=recovery_result, error="", classification_text="", failure_category="",
                terminal_public_response=None,
            )
    if recovery_result.returncode is None:
        limit = f" after {timeout_seconds:g}s" if timeout_seconds is not None else ""
        return _terminal(
            "environment",
            "The bounded claude --resume completion-recovery pass timed out.",
            error=f"completion-recovery resume timed out{limit}",
        )
    if recovery_result.returncode != 0:
        return _terminal(
            "tooling",
            "The bounded claude --resume completion-recovery pass exited with a non-zero status.",
            error=f"completion-recovery resume exited with {recovery_result.returncode}",
        )
    recovery_text = recovery_result.text
    if not recovery_text.strip():
        return _terminal(
            "tooling",
            "The bounded claude --resume completion-recovery pass produced no output.",
            error="completion-recovery resume produced no output",
        )

    try:
        unavailable = parse_agent_unavailable(recovery_text)
    except AgentLoopError:
        unavailable = None
    if unavailable is not None:
        # Agent-declared: post/persist verbatim, never validate()'d, and
        # always terminal regardless of unavailable.retryable.
        _post_completion_recovery_terminal_comment(
            runner,
            config=config,
            completion_recovery=completion_recovery,
            recovery_result=recovery_result,
            terminal_text=recovery_text,
        )
        return _CompletionRecoveryOutcome(
            validated=None,
            result=recovery_result,
            error=(
                "agent explicitly reported it cannot continue after completion "
                f"recovery ({unavailable.category}): {unavailable.summary}"
            ),
            classification_text=recovery_text,
            failure_category="agent-unavailable",
            terminal_public_response=recovery_text,
        )

    try:
        marker_value = validate(recovery_text)
        accepted_text = accept(recovery_text, marker_value)
    except _ArchitectureImpactContractUnsatisfied as exc:
        return _contract_unsatisfied(exc, recovery_text)
    except _AcceptedTextCanonicalizationError as exc:
        return _canonicalization_failed(exc, recovery_text)
    except AgentLoopError as exc:
        if looks_like_backgrounded_completion(recovery_text):
            # A completed CLI turn that again says it is waiting for
            # background work is not an environment/provider outage. Do not
            # overwrite the real diagnostic with AGENT_UNAVAILABLE or post a
            # misleading operational-failure comment (#593).
            return _CompletionRecoveryOutcome(
                validated=None,
                result=recovery_result,
                error=(
                    "completion-recovery resume again deferred to background work "
                    f"without a terminal response: {exc}"
                ),
                classification_text=recovery_text,
                failure_category="deterministic",
                terminal_public_response=None,
            )
        _unavailable, rendered = _synthesized_completion_recovery_unavailable(
            config=config,
            recovery_result=recovery_result,
            category="tooling",
            summary=(
                "The bounded claude --resume completion-recovery pass did not "
                f"produce a valid terminal response: {exc}"
            ),
        )
        _post_completion_recovery_terminal_comment(
            runner,
            config=config,
            completion_recovery=completion_recovery,
            recovery_result=recovery_result,
            terminal_text=rendered,
        )
        return _CompletionRecoveryOutcome(
            validated=None,
            result=recovery_result,
            error=str(exc),
            classification_text=recovery_text,
            failure_category="agent-unavailable",
            terminal_public_response=rendered,
        )

    return _CompletionRecoveryOutcome(
        validated=_accepted_validated_response(
            accepted_text,
            session_id=recovery_result.session_id,
            usage=recovery_usage,
            model_used=recovery_result.model_used,
            **_response_identity_fields(
                recovery_result, acquisition_result=acquisition_result
            ),
        ),
        result=recovery_result,
        error="",
        classification_text="",
        failure_category="",
        terminal_public_response=None,
    )


def _refresh_plan_validation_capture(
    exhaustion: DeterministicPlanValidationExhaustion | None,
    *,
    eligible: bool,
    text: str,
    diagnostic: str,
) -> DeterministicPlanValidationExhaustion | None:
    """Re-point an eligible capture at a normalized candidate; never create one."""
    if not eligible or exhaustion is None:
        return exhaustion
    return DeterministicPlanValidationExhaustion(
        candidate_kind=exhaustion.candidate_kind,
        candidate_text=text,
        diagnostic=diagnostic,
        candidate_digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _semantic_patch_payload_rejection(
    exc: AgentLoopError,
    *,
    text: str,
    normalized: str | None,
    payload_validator: Callable[[dict], object] | None,
) -> tuple[str, str] | None:
    """Return ``(candidate, diagnostic)`` when no envelope repair can succeed.

    Repair preservation pins a ``plan_revision_patch`` payload byte-for-byte,
    so the defects repair may fix are exactly the envelope/footer ones.  The
    recovered payload is therefore checked on its own, independent of which
    error the full validator reported first: an envelope defect can mask a
    payload defect that no repair could clear.  Any payload rejection -- a
    strict patch-schema failure or a ledger/disposition check -- is
    unsatisfiable by repair and goes to the bounded replan instead (#979).
    That includes a recovered JSON object with a missing or wrong ``kind``,
    which the repair integrity gate would refuse anyway.  Only text with no
    recoverable JSON object stays on the existing semantic-patch integrity
    path.
    """
    if isinstance(exc, SemanticPatchPayloadRejection):
        return text, str(exc)
    if payload_validator is None:
        return None
    for candidate in (text, normalized):
        if candidate is None:
            continue
        payload = recover_payload(candidate)
        if not isinstance(payload, dict):
            continue
        try:
            payload_validator(payload)
        except AgentLoopError as payload_exc:
            return candidate, str(payload_exc)
        return None
    return None


def _history_strip_reason(ledger_incomplete: bool) -> str:
    """Log phrasing for why a history-proven strip was allowed."""
    return (
        "despite incomplete ledger"
        if ledger_incomplete
        else "under the lossless semantic-patch history proof"
    )


def _history_strip_allowed(
    text: str,
    exc: UnknownPriorItemDispositionError,
    *,
    resolved_history_item_ids: Sequence[str] | None,
    expected_kind: str | None,
) -> bool:
    """Whether an incomplete-ledger strip removes only no-op historical entries (#862)."""
    if not resolved_history_item_ids:
        return False
    return unknown_dispositions_are_resolved_history(
        text,
        unknown_ids=exc.unknown_ids,
        resolved_history_ids=resolved_history_item_ids,
        expected_kind=expected_kind,
    )


@dataclass(frozen=True)
class _QuotaReplay:
    """The exact invocation a quota wait replays (#1601)."""

    prompt: str
    session_id: str | None
    timeout_seconds: int | None
    invocation_kwargs: dict[str, object]
    # The same-agent re-ask this dispatch carried, so a replay keeps its
    # evidence scope and head guard (#2066).
    reask: _PendingReask | None = None


class _ProviderEnvelopeFailure(Exception):
    """Internal control flow: a non-SUCCESS agy envelope whose fallback failed validation (#2029).

    Deliberately not an AgentLoopError, so the format-failure handling below
    never sees it.
    """

    def __init__(self, validation_error: AgentLoopError) -> None:
        self.validation_error = validation_error
        super().__init__(str(validation_error))


def _run_validated_agent(*args, **kwargs):
    """Run one validated agent turn, then remove its per-turn response directories.

    Every backend invocation inside the turn (the first attempt, same-agent
    re-asks, retries and completion recovery) allocates its own private
    directory; they are all removed once the turn has finished with them,
    whatever its outcome (#1560).  A cleanup failure is only logged.
    """
    config = kwargs.get("config")

    def report(message: str) -> None:
        if config is not None:
            log(config, message)

    with response_turn_scope(on_error=report):
        if kwargs.get("image_inputs") is not None:
            from .review_images_validation import validate_image_turn
            return validate_image_turn(*args, run_agent=run_agent_result, **kwargs)
        from .review_images_coverage_validation import validate_coverage_turn, guard_ordinary_validation
        if kwargs.get("coverage_scope") is not None:
            return validate_coverage_turn(*args, run_agent=run_agent_result, **kwargs)
        guard_ordinary_validation(args[0] if args else kwargs["runner"], kwargs)
        return _run_validated_agent_in_scope(*args, **kwargs)


def _run_validated_agent_in_scope(
    runner: Runner,
    *,
    agent: AgentName,
    config: AgentLoopConfig,
    prompt: str,
    marker_description: str,
    validate: Callable[[str], object],
    session_id: str | None = None,
    usage_context: RunUsageContext | None = None,
    repair_expected_kind: str | None = None,
    repair_surfaced_requirement_ids: Sequence[str] | None = None,
    repair_reviewer_requirement_ids: Sequence[str] | None = None,
    repair_requires_direct_discussion_ack: bool = False,
    require_execution_strategy_contract: bool = False,
    require_risk_test_matrix_contract: bool = False,
    reject_unsolicited_risk_test_matrix_contract: bool = False,
    ledger_incomplete: bool = False,
    repair_resolved_history_item_ids: Sequence[str] | None = None,
    role: str | None = None,
    label: str | None = None,
    timeout_seconds: float | None = None,
    salvage_context: SalvageContext | None = None,
    operation_description: str | None = None,
    completion_recovery: CompletionRecoveryPolicy | None = None,
    managed_ci_recovery_protection: str | None = None,
    plan_validation_failure_handler: Callable[
        [DeterministicPlanValidationExhaustion, AgentInvocationError], None
    ] | None = None,
    require_architecture_impact_contract: bool = False,
    semantic_patch_payload_validator: Callable[[dict], object] | None = None,
    degrade_architecture_impact: bool = False,
    strict_revalidate: Callable[[str], object] | None = None,
    test_observations_fallback_validate: Callable[[str], object] | None = None,
    test_observations_fallback_strict_revalidate: Callable[[str], object] | None = None,
    reviewer_judgement_turn: bool = False,
    architecture_document_paths: Callable[[], frozenset[str] | None] | None = None,
) -> ValidatedAgentResponse:
    # Agent responses are current untrusted visible text.  Keep this guard in
    # the validation seam so every artifact recovery and repair path receives
    # the same provenance check before it can be accepted.
    response_validator = validate
    if degrade_architecture_impact and strict_revalidate is None:
        raise AgentLoopError(
            "Internal error: an invocation that enables architecture-impact degradation "
            "must supply its strict re-parse."
        )
    if (test_observations_fallback_validate is None) != (
        test_observations_fallback_strict_revalidate is None
    ):
        raise AgentLoopError(
            "Internal error: a test_observations fallback validator must be supplied "
            "together with its strict re-parse."
        )
    validation_acquisition: AgentResult | None = None
    # The last candidate text parsed, with the acquisition it was parsed
    # under; carried on the final rejection for post-push degradation (#2086).
    last_candidate: tuple[str, AgentResult | None] | None = None
    # The last response refused for an unsatisfied architecture-impact
    # contract.  It is retained and surfaced, never silently discarded.
    preserved_unsatisfied: PreservedUnsatisfiedResponse | None = None
    # The (rejected path, set members) of the latest refusal when it was an
    # out-of-set canonical_document_path (#1454); it selects the re-ask text.
    document_path_rejection: tuple[str, frozenset[str]] | None = None

    def contract_refusal(
        result: object, text: str, records: tuple[ParseDegradation, ...] = ()
    ) -> str | None:
        """Store an unsatisfied result as the preserved candidate; return its diagnostic."""
        nonlocal preserved_unsatisfied, document_path_rejection
        diagnostic = (
            _architecture_contract_diagnostic(result)
            if require_architecture_impact_contract else None
        )
        rejection: tuple[str, frozenset[str]] | None = None
        if (
            diagnostic is None
            and architecture_document_paths is not None
            and _reported_canonical_document_path(result) is not None
        ):
            # A semantic membership refusal, never a format-repair target:
            # repair pins schema-valid paths, so only the coder can fix it.
            members = architecture_document_paths()
            rejected = _architecture_document_path_rejection(result, members)
            if rejected is not None and members is not None:
                rejection = (rejected, members)
                diagnostic = _architecture_document_path_diagnostic(rejected, members)
        if diagnostic is None:
            return None
        document_path_rejection = rejection
        carrier = _unwrap_architecture_result(result)
        preserved_unsatisfied = PreservedUnsatisfiedResponse(
            text=text,
            diagnostic=diagnostic,
            architecture_impact_degradations=tuple(
                getattr(carrier, "architecture_impact_degradations", ()) or records
            ),
        )
        log(
            config,
            f"{agent_display_name(agent)}: retained a response refused for an unsatisfied "
            f"architecture_impact contract ({len(text)} chars; "
            f"{len(preserved_unsatisfied.architecture_impact_degradations)} degradation record(s))",
        )
        return diagnostic

    def validate(text: str) -> object:
        # The refusing validator: parse, then refuse an unsatisfied contract
        # uniformly for every call site that requires one (#925).  Every
        # accepting branch treats an AgentLoopError here as not accepted, so a
        # candidate whose accepted text cannot be canonicalized is refused on
        # the same path, with its own diagnostic.
        result = parse_response(text)
        refusal = contract_refusal(result, text)
        if refusal is not None:
            raise _ArchitectureImpactContractUnsatisfied(refusal)
        accept_candidate(text, result)
        return result

    def parse_response(text: str) -> object:
        nonlocal last_candidate
        last_candidate = (text, validation_acquisition)
        return _with_validation_context(
            response_validator, runner=runner, acquisition=validation_acquisition
        )(text)

    def accept_candidate(
        text: str, marker_value: object, acquisition: AgentResult | None = None
    ) -> _AcceptedCandidate:
        # The acquisition is captured when the candidate is accepted, never
        # read later from the mutable closure, so a repair turn cannot change
        # the catalog the strict re-parse sees.
        candidate = _accept_candidate(
            text,
            marker_value,
            strict_revalidate=strict_revalidate,
            runner=runner,
            acquisition=acquisition if acquisition is not None else validation_acquisition,
            fallback_revalidate=response_validator,
        )
        return candidate

    def accept_test_observations_fallback(
        text: str, *, source_records: tuple[ParseDegradation, ...] = ()
    ) -> _AcceptedCandidate | None:
        """Accept the original response with today's #927 drop (#1626).

        Only after the same-agent re-asks of a non-array test_observations
        produced no accepted candidate (#2066).  The fallback validator differs from the primary
        one only in its ``degrade-after-repair`` mode, so every other defect
        still rejects; it runs the same contract refusal and is accepted
        through the same boundary with the matching strict re-parse.
        """
        if test_observations_fallback_validate is None:
            return None
        try:
            fallback_result = _with_validation_context(
                test_observations_fallback_validate,
                runner=runner,
                acquisition=validation_acquisition,
            )(text)
            # A repaired candidate carries the source's near-miss record,
            # as the successful repair path does (merge, never replace).
            fallback_result = _attach_architecture_degradations(fallback_result, source_records)
            if contract_refusal(fallback_result, text) is not None:
                return None
            if not any(
                record.rule == TEST_OBSERVATIONS_NOT_ARRAY_AFTER_REPAIR_RULE
                for record in getattr(
                    _unwrap_architecture_result(fallback_result),
                    "test_observation_degradations",
                    (),
                )
            ):
                # The fallback exists only to drop a non-array field; any
                # other acceptance belongs to the repair-first path.
                return None
            return _accept_candidate(
                text,
                fallback_result,
                strict_revalidate=test_observations_fallback_strict_revalidate,
                runner=runner,
                acquisition=validation_acquisition,
            )
        except NonRepairableEvidenceRejection:
            raise
        except AgentLoopError as fallback_exc:
            log(
                config,
                f"{agent_display_name(agent)}: test_observations fallback not accepted "
                f"({fallback_exc})",
            )
            return None

    agent_name = agent_display_name(agent)
    operation_description = operation_description or _operation_description_from_context(
        salvage_context=salvage_context,
        repair_expected_kind=repair_expected_kind,
        role=role,
        label=label,
        marker_description=marker_description,
    )
    log_paths: list[object] = []
    from .reviewer_seats import SeatAgent, seat_backend
    backend_agent = seat_backend(agent)
    if isinstance(agent, SeatAgent):
        seat_overrides = {"active_reviewer_seat_id": str(agent)}
        if backend_agent == "antigravity":
            seat_overrides.update(
                antigravity_models=agent.model_chain,
                antigravity_model=None,
            )
        config = dataclasses_replace(config, **seat_overrides)
    antigravity_attempts = (
        AntigravityAttemptState.from_config(config, config.agent_max_retries)
        if backend_agent == "antigravity"
        else None
    )
    # agy schema-constrained output applies only to an eligible turn (#2029);
    # it rides on the per-attempt config so other backends stay untouched.
    antigravity_structured_kind = (
        antigravity_structured_kind_for_turn(
            repair_expected_kind,
            surfaced_requirement_ids=repair_surfaced_requirement_ids,
            reviewer_requirement_ids=repair_reviewer_requirement_ids,
            requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
        )
        if antigravity_attempts is not None
        else None
    )
    # Each fallback retains an initial attempt; retry allowance is shared. The
    # provider-specific replacement replay gets one explicit extra slot.
    max_attempts = (
        len(config.antigravity_models) + config.agent_max_retries + 2
        if antigravity_attempts is not None
        else config.agent_max_retries + 2
    )
    # The shared same-agent re-ask budget (#2066); it never consumes
    # agent_max_retries or Antigravity fallback state.
    max_attempts += MAX_SAME_AGENT_REASKS
    if antigravity_structured_kind is not None:
        # One dedicated structured-off replay after agy rejects agent-loop's
        # request schema (#2039); never consumes agent_max_retries.
        max_attempts += 1
    # Run-scoped wait budget, fetched lazily so an unset option never creates one.
    quota_wait_budget: quota_wait.QuotaWaitBudget | None = None
    # Only ever created from an invocation dispatched in this turn that then
    # failed quota; None at turn entry (#1601).
    pending_quota_replay: _QuotaReplay | None = None
    last_dispatch: _QuotaReplay | None = None
    quota_replays_used = 0
    # Quota-exhausted dispatches since the last wait.  Each wait renews them
    # by extending the attempt ceiling, so a replay and any renewed Antigravity
    # fallback never consume the ordinary allowance (#1601).
    quota_dispatches_since_wait = 0
    last_error =f"{agent_name} produced no output."
    last_result: AgentResult | None = None
    # Every agy CLI log located for a silent Antigravity attempt in this turn
    # (#1604); each terminal error names them so operators can find them.
    agy_log_paths: list[Path] = []

    def _agy_log_suffix() -> str:
        return "".join(f" agy log: {path}." for path in agy_log_paths)
    last_classification_text = ""
    last_failure_category = "empty-response"
    # Set only by an exhausted completion-recovery attempt (#588): the
    # protocol-valid text already persisted to the recovery attempt's own
    # response file and posted to the GitHub issue, attached to the final
    # AgentInvocationError so callers/tests can assert on it without
    # re-parsing the message.
    terminal_public_response: str | None = None
    plan_validation_exhaustion: DeterministicPlanValidationExhaustion | None = None
    bounded_replan_rejection: DeterministicPlanValidationExhaustion | None = None
    # This latch is reset per invocation attempt and set only at the exact
    # structured-validator capture point. It prevents a stale typed candidate
    # from reaching the persistence callback after an ineligible path.
    plan_validation_capture_eligible = False
    completion_recovery_attempted = False
    # Keep the detection guard separate from the replay marker: a failed
    # stability check must not make the next ordinary retry look like a replay.
    executable_replacement_considered = False
    executable_replacement_replay_pending = False
    executable_replacement_provider: AgentName | None = None
    executable_replacement_reason: str | None = None
    ordinary_retries_used = 0
    self_update_deadline: float | None = None
    self_update_stability_error: str | None = None
    # Refusals are diagnostic-only context. Keep only the latest one, without
    # treating it as accepted executable-replacement evidence.
    latest_replay_refusal_detail: str | None = None
    next_timeout_seconds = timeout_seconds
    settings_lock_retry_used = False
    # One structured-off replay per turn after a request-schema rejection
    # (#2039); the replayed dispatch is set only by that branch.
    structured_request_fallback_used = False
    pending_structured_off_replay: _QuotaReplay | None = None
    # The single same-agent re-ask rule (#2066): one shared budget for every
    # deterministic validation failure, replacing the repair pass and the
    # per-defect re-ask slots.
    is_coder_turn = role == "coder" or repair_expected_kind in _CODER_EXPECTED_KINDS
    reasks_used = 0
    pending_reask: _PendingReask | None = None
    reask_budget_exhausted = False
    reask_skip_detail: str | None = None
    # The model that produced the turn's responses once it diverged from the
    # configuration: kept for every later re-ask of the turn, so a pinned
    # re-ask whose result now reports the pinned model as configured is not
    # sent back to the configured model (#2066).
    turn_producer_model: str | None = None
    executable_replacement_policies: dict[AgentName, tuple[str, str, str, bool]] = {
        "claude": (
            config.claude_cmd,
            "Claude self-update",
            "self-update-attempt2",
            True,
        ),
        "codex": (
            config.codex_cmd,
            "Codex executable replacement",
            "executable-replacement-attempt2",
            False,
        ),
        "gemini": (
            config.gemini_cmd,
            "Gemini executable replacement",
            "executable-replacement-attempt2",
            False,
        ),
        "antigravity": (
            config.antigravity_cmd,
            "Antigravity executable replacement",
            "executable-replacement-attempt2",
            False,
        ),
    }

    def _quota_reset_error(reset_secs: int, classification_text: str) -> QuotaResetExceededError:
        duration_str = _format_reset_duration(reset_secs)
        at_str = _format_reset_at_utc(reset_secs)
        message = (
            f"{agent_name} quota exhausted. Reset in {duration_str} (at {at_str}). "
            "Rerun when quota resets, or switch to a different API key / model."
        )
        replacement_detail = _executable_replacement_failure_detail(
            provider=executable_replacement_provider,
            reason=executable_replacement_reason,
            stability_error=self_update_stability_error,
        )
        diagnostic_classification_text = classification_text
        if replacement_detail:
            message += f" {replacement_detail}"
            diagnostic_classification_text = (
                f"{classification_text}\n{replacement_detail}"
            ).strip()
        if latest_replay_refusal_detail:
            message += f" {latest_replay_refusal_detail}"
            diagnostic_classification_text = (
                f"{diagnostic_classification_text}\n{latest_replay_refusal_detail}"
            ).strip()
        diagnostics = _failed_run_diagnostics(
            runner=runner,
            config=config,
            agent_name=agent_name,
            salvage_context=salvage_context,
            operation_description=operation_description,
            failure_category=last_failure_category,
            failure_reason=message,
            classification_text=diagnostic_classification_text,
            marker_description=marker_description,
            result=last_result,
        )
        message += _agy_log_suffix()
        message += diagnostics.format_for_error()
        return QuotaResetExceededError(message)

    def _reserve_quota_wait(reset_secs: int) -> quota_wait.QuotaWaitReservation | None:
        nonlocal quota_wait_budget
        if config.wait_for_quota_reset_seconds is None:
            return None
        if quota_wait_budget is None:
            quota_wait_budget = quota_wait.quota_wait_budget_for_current_run(config)
        return quota_wait_budget.try_reserve(reset_secs)

    def _wait_for_quota_reset(reservation: quota_wait.QuotaWaitReservation) -> None:
        """Wait, then renew the attempts the wait reopens (bounded by the 2-wait budget)."""
        nonlocal max_attempts, quota_dispatches_since_wait
        quota_wait.wait_for_reset(config, agent_name, reservation)
        max_attempts += quota_dispatches_since_wait
        quota_dispatches_since_wait = 0

    def _antigravity_quota_wait() -> quota_wait.QuotaWaitReservation | None:
        """Reserve a wait only for a verified all-groups parsed long reset within DURATION."""
        state = antigravity_attempts
        limit = config.wait_for_quota_reset_seconds
        if limit is None or not state.stop_is_verified_long_reset():
            return None
        latest = state.latest_reset_seconds()
        earliest = state.earliest_reset_seconds()
        if latest is None or earliest is None or latest > limit:
            return None
        return _reserve_quota_wait(earliest)

    def _antigravity_stop_error(
        classification_text: str, *, shared_limit: bool
    ) -> AgentInvocationError:
        """Early stop: every reachable quota group is exhausted (#1236)."""
        state = antigravity_attempts
        if state.stop_is_verified_long_reset():
            return _quota_reset_error(state.earliest_reset_seconds() or 1, classification_text)
        # Transient, like the ordinary chain-exhausted failure it replaces: multi-reviewer
        # rounds then mark Antigravity unavailable instead of aborting the round.
        return AgentInvocationError(
            state.unavailable_message(shared_limit) + _agy_log_suffix(),
            failure_category="transient",
        )

    def schedule_reask(
        *,
        message: str,
        result: AgentResult,
        hint_prompt: str | None = None,
        fresh_evidence: bool = False,
    ) -> bool:
        """Schedule one same-agent re-ask; False when none may run (#2066).

        A coder re-ask needs a readable checkout head as its baseline; without
        one the turn fails as before.  The re-ask resumes the response's own
        session when there is one and otherwise resends the full prompt with
        a bounded, sanitized copy of the rejected response (#2128).
        """
        nonlocal reasks_used, pending_reask, reask_budget_exhausted, reask_skip_detail
        nonlocal turn_producer_model
        if reasks_used >= MAX_SAME_AGENT_REASKS:
            reask_budget_exhausted = True
            return False
        head_before: str | None = None
        if is_coder_turn:
            probe = _read_reask_head(runner, config)
            if not probe.available or not probe.value:
                reask_skip_detail = (
                    "coder re-ask skipped: the checkout head could not be read "
                    f"({probe.kind})"
                )
                log(config, f"{agent_name}: {reask_skip_detail}")
                return False
            head_before = probe.value
        reasks_used += 1
        observed_producer = result.observed_model or turn_producer_model
        if observed_producer and (
            turn_producer_model is not None
            or observed_producer != result.configured_model
        ):
            turn_producer_model = observed_producer
        reask_session = result.session_id or session_id
        pending_reask = _PendingReask(
            number=reasks_used,
            prompt=build_reask_prompt(
                hint_prompt if hint_prompt is not None else prompt,
                message,
                reasks_used,
                coder=is_coder_turn,
                fresh_evidence=fresh_evidence,
                config=config,
                # A session-less re-ask is a fresh turn: quote the rejected
                # response back so the agent can keep its judgements (#2128).
                prior_response=result.text if reask_session is None else None,
            ),
            session_id=reask_session,
            fresh_evidence=fresh_evidence,
            head_before=head_before,
            producer_model=turn_producer_model,
        )
        log(
            config,
            f"{agent_name}: response rejected ("
            f"{sanitize_validator_message(message, config=config, limit=300)}); "
            f"re-asking the same agent ({reasks_used}/{MAX_SAME_AGENT_REASKS}"
            + ("; fresh test evidence allowed" if fresh_evidence else "")
            + ("; resuming session)" if reask_session else "; fresh turn)"),
        )
        return True

    attempt = 0
    # A while loop: a quota wait may extend max_attempts mid-turn.
    while attempt < max_attempts:
        attempt += 1
        replacement_stability_failed = False
        if (
            antigravity_attempts is not None
            and antigravity_attempts.ensure_eligible_before_attempt() == "all-exhausted"
        ):
            log(config, f"{agent_name}: every Antigravity quota group is cooling down; not invoking")
            reservation = _antigravity_quota_wait()
            if reservation is None:
                raise _antigravity_stop_error(last_classification_text, shared_limit=False)
            # Pre-dispatch: nothing failed in this attempt, so wait, re-check
            # and build the intended invocation normally -- never a replay,
            # and pending corrective context is left untouched (#1601).
            _wait_for_quota_reset(reservation)
            if antigravity_attempts.ensure_eligible_before_attempt() == "all-exhausted":
                raise _antigravity_stop_error(last_classification_text, shared_limit=False)
        attempt_started_at = time.monotonic()
        attempt_config = (
            dataclasses_replace(
                antigravity_attempts.singleton_config(config),
                antigravity_structured_kind=antigravity_structured_kind,
            )
            if antigravity_attempts is not None
            else config
        )
        invocation_kwargs: dict[str, object] = {}
        is_executable_replacement_replay = executable_replacement_replay_pending
        is_quota_replay = pending_quota_replay is not None
        if pending_quota_replay is not None:
            # A quota-wait replay takes precedence over every other builder:
            # the exact effective prompt, session, timeout and settings of the
            # invocation that failed quota (#1601).
            quota_replays_used += 1
            attempt_prompt = pending_quota_replay.prompt
            attempt_session_id = pending_quota_replay.session_id
            next_timeout_seconds = pending_quota_replay.timeout_seconds
            invocation_kwargs = dict(pending_quota_replay.invocation_kwargs)
            if backend_agent == "claude":
                invocation_kwargs["attempt_suffix"] = f"quota-wait-{quota_replays_used}"
            dispatch_reask = pending_quota_replay.reask
            pending_quota_replay = None
        elif pending_structured_off_replay is not None:
            # The exact dispatch agy rejected for its schema, re-sent without
            # schema flags (#2039): not a corrective re-prompt.
            attempt_prompt = pending_structured_off_replay.prompt
            attempt_session_id = pending_structured_off_replay.session_id
            next_timeout_seconds = pending_structured_off_replay.timeout_seconds
            invocation_kwargs = dict(pending_structured_off_replay.invocation_kwargs)
            dispatch_reask = pending_structured_off_replay.reask
            pending_structured_off_replay = None
        else:
            if backend_agent == "claude":
                invocation_kwargs["attempt_suffix"] = (
                    "self-update-attempt2"
                    if is_executable_replacement_replay
                    else f"attempt{ordinary_retries_used + 1}"
                )
            elif is_executable_replacement_replay:
                invocation_kwargs["attempt_suffix"] = executable_replacement_policies[backend_agent][2]
            attempt_prompt = prompt
            attempt_session_id = session_id
            dispatch_reask = pending_reask
            pending_reask = None
            if dispatch_reask is not None:
                # The same agent, the model that produced the rejected response
                # (the Antigravity chain is not advanced) and its own session.
                attempt_prompt = dispatch_reask.prompt
                attempt_session_id = dispatch_reask.session_id
                if backend_agent == "claude":
                    invocation_kwargs["attempt_suffix"] = f"reask{dispatch_reask.number}"
        last_dispatch = _QuotaReplay(
            attempt_prompt,
            attempt_session_id,
            next_timeout_seconds,
            dict(invocation_kwargs),
            reask=dispatch_reask,
        )
        if usage_context is not None:
            usage_context.note_agent_dispatch()
        dispatch_agent, dispatch_config = agent, attempt_config
        if dispatch_reask is not None and dispatch_reask.producer_model:
            dispatch_agent, dispatch_config = _pin_producer_model(
                agent, attempt_config, backend_agent, dispatch_reask.producer_model
            )
        while True:
            try:
                result = run_agent_result(
                    runner,
                    agent=dispatch_agent,
                    config=dispatch_config,
                    prompt=attempt_prompt,
                    session_id=attempt_session_id,
                    run_id=usage_context.run_id if usage_context is not None else None,
                    role=role,
                    label=label,
                    timeout_seconds=next_timeout_seconds,
                    **invocation_kwargs,
                )
                break
            except AgentSettingsLockWaitTimeout as exc:
                # Host-wide lock contention, not a model or turn failure: one
                # dedicated in-place re-wait that consumes no agent_max_retries,
                # fallback or quota state, and runs no repair (#1400). A second
                # expiry propagates as a transient AgentInvocationError.
                if settings_lock_retry_used:
                    raise
                settings_lock_retry_used = True
                log(
                    config,
                    f"{agent_name}: shared agy settings lock wait expired ({exc}); "
                    "retrying the lock wait once",
                )
        # The active evidence acquisition (#2066): the latest dispatch whose
        # instruction allowed test runs -- a primary or ordinary-retry
        # dispatch, or a fresh-evidence re-ask.  A resend-only re-ask inherits
        # it unchanged, so its own receipts are never admitted and catalogs
        # are never merged.  The snapshot stays fixed through every callback.
        if dispatch_reask is None or dispatch_reask.fresh_evidence:
            validation_acquisition = result
        audit_result = result
        if result.text_source == "structured_output":
            # The payload recovered from the agy envelope won (#2029, #2049):
            # the rendered payload is this
            # invocation's authoritative artifact. A response file the model
            # also wrote is audit-only, so artifact salvage, response-file
            # status and structured-candidate recovery see the rendered
            # payload and never override or reclassify it. ``last_result``
            # below keeps the original file for diagnostics.
            result = dataclasses_replace(result, response_file_text=result.text)
        # The bounded deadline belongs only to the interrupted invocation and
        # its dedicated replay. A later ordinary retry has its normal budget.
        if is_executable_replacement_replay or is_quota_replay:
            executable_replacement_replay_pending = False
            next_timeout_seconds = timeout_seconds
        last_result = audit_result
        # A typed planning-validation candidate is valid only for the final
        # deterministic failure of this invocation. A later timeout, provider,
        # marker-safety, or containment failure must clear it.
        plan_validation_exhaustion = None
        plan_validation_capture_eligible = False
        bounded_replan_rejection = None
        if result.log_path is not None:
            log_paths.append(result.log_path)
        if (
            result.antigravity_log_path is not None
            and result.antigravity_log_path not in agy_log_paths
        ):
            agy_log_paths.append(result.antigravity_log_path)
        text = _neutralize_untrusted_markers(result.text, config=config, agent_name=agent_name)
        usage = _resolve_usage_metadata(config=config, prompt=attempt_prompt, result=result)
        usage_record = None
        if usage_context is not None and usage is not None:
            usage_record = usage_context.add_record(
                agent=agent,
                backend=backend_agent if isinstance(agent, SeatAgent) else None,
                session_id=result.session_id,
                returncode=result.returncode,
                usage=usage,
                raw_backend_usage=result.raw_usage,
                turn_role=role,
                model=result.model_used,
                configured_model=result.configured_model,
                configured_effort=result.configured_effort,
                effort_source=result.effort_source,
                observed_model=result.observed_model,
                observed_effort=result.observed_effort,
                observation_provenance=result.observation_provenance,
                containment=(result.containment.to_dict() if result.containment is not None else None),
            )
            if result.antigravity_log_path is not None:
                usage_record.agy_log_path = str(result.antigravity_log_path)

        if dispatch_reask is not None and dispatch_reask.head_before is not None:
            # Coder head guard (#2066): a re-ask may change only the response
            # text.  A moved (or unreadable) head is never accepted and gets
            # no further re-ask; pr_loop's post-push coder-record path then
            # handles the moved head (#2086).
            head_after = _read_reask_head(runner, config)
            if not head_after.available or head_after.value != dispatch_reask.head_before:
                after_label = head_after.value if head_after.available else f"unavailable ({head_after.kind})"
                last_error = (
                    f"coder re-ask {dispatch_reask.number} moved the checkout head "
                    f"({dispatch_reask.head_before} -> {after_label}); the re-asked "
                    "response was not accepted"
                )
                last_classification_text = ""
                last_failure_category = "deterministic"
                if usage_record is not None:
                    usage_record.validation_status = "invalid"
                log(config, f"{agent_name}: {last_error}")
                break

        # A backend always loads the uniquely assigned public response file.
        # On a timeout/nonzero exit, that artifact can still be a complete
        # response; stdout is diagnostics only and must never be salvaged.
        artifact = result.response_file_text
        artifact_unavailable = None
        # An authoritative artifact whose only defect is the architecture
        # contract is a deterministic field refusal, never a transport failure.
        artifact_contract_refusal: _ArchitectureImpactContractUnsatisfied | None = None
        # An artifact that parses but whose accepted text cannot be
        # canonicalized keeps its own deterministic diagnostic; it is never
        # reclassified as a timeout or command failure.
        artifact_canonicalization_failure: _AcceptedTextCanonicalizationError | None = None
        # Preserve the normal zero-exit path below, including its marker
        # recovery diagnostic. Failed exits alone may be salvaged from the
        # per-invocation response-file artifact.
        if artifact and result.returncode != 0:
            # Salvage must not depend on the exit code.  The zero-exit path
            # already defangs reserved names before validation, so defang the
            # artifact the same way instead of discarding a complete answer
            # whose prose merely names a record (#891).
            artifact = (
                text
                if artifact == result.text
                else _neutralize_untrusted_markers(
                    artifact, config=config, agent_name=agent_name
                )
            )
            try:
                artifact_unavailable = parse_agent_unavailable(artifact)
            except AgentLoopError:
                artifact_unavailable = None
            if artifact_unavailable is None:
                try:
                    artifact_marker_value = validate(artifact)
                except _ArchitectureImpactContractUnsatisfied as exc:
                    artifact_contract_refusal = exc
                except _AcceptedTextCanonicalizationError as exc:
                    artifact_canonicalization_failure = exc
                except AgentLoopError:
                    pass
                else:
                    acquisition_outcome = (
                        "accepted_timeout" if result.returncode is None
                        else "accepted_nonzero_exit" if result.returncode != 0 else "success"
                    )
                    if usage_record is not None:
                        usage_record.validation_status = "validated"
                        usage_record.outcome = acquisition_outcome
                    if acquisition_outcome != "success":
                        log(
                            config,
                            f"{agent_name}: accepted valid response-file artifact despite "
                            f"returncode={result.returncode!r}",
                        )
                    return _accepted_validated_response(
                        accept_candidate(artifact, artifact_marker_value),
                        session_id=result.session_id,
                        usage=usage,
                        model_used=result.model_used,
                        **_response_identity_fields(result, acquisition_result=validation_acquisition),
                        acquisition_outcome=acquisition_outcome,
                        acquisition_returncode=result.returncode,
                    )
            else:
                # Let the existing agent-unavailable policy handle the valid
                # envelope, even when the command itself failed.
                text = artifact

        containment = result.containment
        target_exec_retryable = False
        target_exec_command = ""
        if (
            containment is not None
            and result.returncode is not None
            and containment.resource_exhausted
            # Validate a successful response before consulting cumulative
            # cgroup counters. A recovered descendant can increment those
            # counters while the invocation still returns a complete answer.
            and not (result.returncode == 0 and text.strip())
        ):
            # Resource evidence is typed by the runner and takes precedence
            # over provider/transient regexes.  A killed coder is not required
            # to narrate its own OOM; response-file salvage above remains valid.
            last_error = (
                "agent invocation terminated as resource-exhausted"
                f" (limit={containment.applicable_limit or 'cgroup resource limit'}; "
                f"backend={containment.backend})"
            )
            last_classification_text = ""
            last_failure_category = "resource-exhausted"
            break
        if (
            containment is not None
            and containment.termination_cause == "target-exec-error"
        ):
            decision = getattr(runner, "target_exec_retry_decision", None)
            if decision is not None:
                target_argv = (
                    result.command_result.args
                    if result.command_result is not None
                    else ()
                )
                target_exec_command = target_argv[0] if target_argv else ""
                target_exec_retryable, detail = decision(
                    target_exec_command,
                    containment.target_exec_errno,
                )
            else:
                detail = "agent target could not be executed; target exec error"
            if target_exec_retryable:
                last_error = detail
                last_classification_text = detail
                last_failure_category = "transient"
            else:
                last_error = (
                    "agent target could not be executed; "
                    + ("; ".join(containment.diagnostics) or detail)
                )
                last_classification_text = ""
                last_failure_category = "deterministic"
                break
        if (
            containment is not None
            and containment.managed_cleanup_unconfirmed
        ):
            last_error = (
                "agent invocation cleanup-failed: managed scope emptiness was not "
                "confirmed; retry is blocked until the invocation tree is gone"
            )
            last_classification_text = ""
            last_failure_category = "containment-indeterminate"
            break

        if result.self_update_replay_refusal_kind is not None:
            refusal_detail = result.self_update_replay_refusal_detail or (
                f"{agent_name} replay refused ({result.self_update_replay_refusal_kind})"
            )
            latest_replay_refusal_detail = refusal_detail
            refusal_outcome = (
                "self_update_replay_refused_changed_workdir"
                if result.self_update_replay_refusal_kind in {"changed-head", "changed-status"}
                else "self_update_replay_refused_unavailable_workdir"
            )
            if usage_record is not None:
                usage_record.outcome = refusal_outcome
                usage_record.log_path = str(result.log_path) if result.log_path else None
            log(
                config,
                f"{agent_name} attempt replay refusal: {refusal_detail}; "
                "continuing with provider-derived retry classification",
            )

        # A configured provider executable can be replaced after spawning. Each
        # backend supplies its own evidence gate, while this branch owns the
        # bounded stability wait, one replay, and retry accounting. A workdir
        # refusal deliberately remains diagnostic-only and cannot enter it.
        if (
            backend_agent in executable_replacement_policies
            and result.self_update_reason is not None
            and result.self_update_replay_refusal_kind is None
            and not executable_replacement_considered
            and result.command_result is not None
        ):
            executable_replacement_considered = True
            executable_replacement_provider = agent
            executable_replacement_reason = result.self_update_reason
            observation = result.command_result.observation
            command, provider_label, _suffix, uses_remaining_deadline = (
                executable_replacement_policies[backend_agent]
            )
            if uses_remaining_deadline:
                self_update_deadline = (
                    observation.spawn_monotonic + timeout_seconds
                    if timeout_seconds is not None and observation is not None
                    else None
                )
                stable = runner.wait_for_executable_stability(
                    config.claude_cmd, deadline=self_update_deadline
                )
                remaining = (
                    self_update_deadline - time.monotonic()
                    if self_update_deadline is not None else None
                )
                replay_timeout = remaining
                deadline_exhausted = remaining is not None and remaining <= 0
            else:
                # Codex, Gemini, and Antigravity use a bounded six-second
                # stability observation. Their replay is fresh and receives the
                # complete configured timeout, if any.
                stable = runner.wait_for_executable_stability(
                    command, deadline=None
                )
                replay_timeout = timeout_seconds
                deadline_exhausted = False
            if not stable or deadline_exhausted:
                replacement_stability_failed = True
                deadline_label = (
                    "within the invocation deadline"
                    if uses_remaining_deadline
                    else "within the bounded stability window"
                )
                self_update_stability_error = (
                    f"likely {provider_label} interruption ({result.self_update_reason}); "
                    f"executable did not stabilize {deadline_label}"
                )
                if usage_record is not None:
                    usage_record.outcome = (
                        "self_update_interruption"
                        if uses_remaining_deadline
                        else "executable_replacement_interruption"
                    )
                    usage_record.log_path = str(result.log_path) if result.log_path else None
                # Preserve the ordinary retry allowance: a failed stability
                # observation does not make provider errors final.
            else:
                executable_replacement_replay_pending = True
                # A replayed re-ask stays a re-ask: same prompt, evidence
                # scope and head baseline, no extra budget.
                pending_reask = dispatch_reask
                if usage_record is not None:
                    usage_record.outcome = (
                        "self_update_interruption"
                        if uses_remaining_deadline
                        else "executable_replacement_interruption"
                    )
                    usage_record.log_path = str(result.log_path) if result.log_path else None
                next_timeout_seconds = replay_timeout
                log(
                    config,
                    f"{agent_name}: {result.self_update_reason}; replaying once after executable stability",
                )
                continue
        if (
            backend_agent == "antigravity"
            and antigravity_attempts is not None
            and antigravity_structured_kind is not None
            and not structured_request_fallback_used
            and result.antigravity_structured_request_rejection is not None
            and result.antigravity_log_quota_evidence is None
            and result.returncode is not None
            and artifact_unavailable is None
            and not target_exec_retryable
            and last_dispatch is not None
        ):
            # agy refused agent-loop's own request schema before the model ran
            # (#2039). Replay the same dispatch once with structured output off
            # for the rest of this turn: no retry budget, no format-failure
            # count, no model-chain advance and no repair.
            structured_request_fallback_used = True
            antigravity_structured_kind = None
            pending_structured_off_replay = last_dispatch
            last_error = "Antigravity rejected agent-loop's structured-output request"
            last_classification_text = result.antigravity_structured_request_rejection
            log(
                config,
                f"WARNING: {agent_name}: Antigravity rejected agent-loop's structured-output "
                f"request ({result.antigravity_structured_request_rejection[:600]}); retrying "
                "this turn once with structured output off",
            )
            continue
        should_retry = False
        # Set only when the reviewer's turn carried no recoverable structured
        # payload and repair was refused for it (#2029).
        format_failure = False
        # Codex structured verdicts read the reset from the authoritative error
        # only (#1281); None falls back to the whole classification text.
        reset_evidence_text: str | None = None
        provider_capacity = False
        capacity = None
        if artifact_contract_refusal is not None:
            # The artifact stays authoritative for this invocation: skip the
            # timeout and nonzero-exit transport branches, make no repair
            # invocation, and retry with the field-naming re-prompt.
            last_error = str(artifact_contract_refusal)
            classification_text = "response-file artifact failed the architecture_impact contract"
            last_classification_text = artifact
            last_failure_category = "deterministic"
            if usage_record is not None:
                usage_record.validation_status = "invalid"
            if schedule_reask(
                message=last_error,
                result=result,
                hint_prompt=_reask_hint_prompt(
                    prompt,
                    artifact_contract_refusal,
                    contract_unsatisfied=True,
                    document_path_rejection=document_path_rejection,
                ),
            ):
                continue
            break
        elif artifact_canonicalization_failure is not None:
            # The authoritative artifact was otherwise valid but its accepted
            # text failed the strict canonical comparison.  Report that
            # diagnostic as a deterministic failure, never as transport, and
            # never as an unsatisfied architecture contract.
            last_error = str(artifact_canonicalization_failure)
            last_classification_text = artifact
            last_failure_category = "deterministic"
            if usage_record is not None:
                usage_record.validation_status = "invalid"
            log(
                config,
                f"{agent_name}: response-file artifact not accepted "
                f"({artifact_canonicalization_failure})"[:600],
            )
            break
        elif result.returncode is None and artifact_unavailable is None:
            # Timed out (returncode=None from Runner.run_with_log). Detected
            # before transient classification: a kill deadline is not a
            # provider hiccup, so retrying or repairing would only waste the
            # same wall-clock budget again (#475).
            limit = f" after {timeout_seconds:g}s" if timeout_seconds is not None else ""
            last_error = f"agent command timed out{limit}"
            last_classification_text = ""
            last_failure_category = "timeout"
            break
        elif (
            backend_agent == "antigravity"
            and antigravity_attempts is not None
            and result.antigravity_log_quota_evidence is not None
            and artifact_unavailable is None
            and not target_exec_retryable
            and not replacement_stability_failed
            and not (result.command_result and result.command_result.capture_diagnostics)
        ):
            # agy printed nothing, but its own CLI log for this attempt shows
            # the executor's RESOURCE_EXHAUSTED 429 (#1604). Route it into the
            # quota path below: no format validation and no repair pass.
            evidence = result.antigravity_log_quota_evidence
            capacity = antigravity_log_quota_capacity(evidence.log_path, evidence.line)
            provider_capacity = True
            should_retry = True
            last_failure_category = "transient"
            last_error = (
                f"Antigravity printed no output; agy log {evidence.log_path.name} shows "
                "RESOURCE_EXHAUSTED (code 429)"
            )
            classification_text = evidence.line
            last_classification_text = evidence.line
            log(
                config,
                f"Antigravity printed no output; agy log {evidence.log_path.name} shows "
                "RESOURCE_EXHAUSTED (code 429); treating as quota exhaustion",
            )
        elif result.returncode != 0 and artifact_unavailable is None:
            last_error = f"agent command exited with {result.returncode}"
            classification_text, provider_verdict = _agent_failure_classification(
                result, phase="command"
            )
            if target_exec_retryable:
                target_argv = (
                    result.command_result.args
                    if result.command_result is not None
                    else ()
                )
                last_error = str(
                    getattr(runner, "target_exec_retry_decision")(
                        target_argv[0] if target_argv else "",
                        containment.target_exec_errno,
                    )[1]
                )
                classification_text = f"{last_error}\n{classification_text}".strip()
            if result.command_result is not None and result.command_result.capture_diagnostics:
                classification_text += "\nsubprocess capture unavailable; retryable tooling failure"
            last_classification_text = classification_text
            structured_verdict = (
                provider_verdict
                if provider_verdict is not None and provider_verdict.source == "structured"
                else None
            )
            if structured_verdict is not None:
                reset_evidence_text = structured_verdict.authoritative_text
            should_retry = (
                target_exec_retryable
                or replacement_stability_failed
                or bool(result.command_result and result.command_result.capture_diagnostics)
                or (
                    structured_verdict.category == "transient"
                    if structured_verdict is not None
                    else _is_transient_agent_output(classification_text)
                )
            )
            last_failure_category = _failure_category(
                classification_text, provider_verdict=structured_verdict
            )
            if target_exec_retryable:
                # The typed runner decision outranks textual classification:
                # a preflighted CLI mid self-update is transient (#1226).
                last_failure_category = "transient"
            capacity = classify_antigravity_capacity(
                classification_text,
                returncode=result.returncode,
                empty_response=False,
                signatures=config.antigravity_quota_signatures,
            ) if backend_agent == "antigravity" else None
            provider_capacity = bool(capacity and capacity.is_capacity)
            if provider_capacity:
                should_retry = True
                last_failure_category = "transient"
            elif (
                capacity is not None
                and capacity.auth_failure
                and not target_exec_retryable
                and not replacement_stability_failed
                and not (result.command_result and result.command_result.capture_diagnostics)
            ):
                # A typed Antigravity auth verdict beats generic transient wording,
                # but yields to typed runner retry decisions (#1226, #1598).
                should_retry = False
                last_failure_category = "non-retryable"
        elif (
            backend_agent == "antigravity"
            and antigravity_structured_kind is not None
            and result.antigravity_structured_ambiguity is not None
        ):
            # The agy envelope's response held several candidate JSON objects,
            # an object of another kind, or a contradicting footer (#2049).
            # That is a pure format failure: classify from the tool-authored
            # diagnostic only (never raw output or rejected payload prose, so
            # no quota/auth/capacity path), skip repair, and let the #2029
            # per-model format accounting decide retry or format-fallback.
            diagnostic = result.antigravity_structured_ambiguity
            last_error = diagnostic
            classification_text = diagnostic
            last_classification_text = diagnostic
            last_failure_category = "agent-unavailable"
            should_retry = True
            format_failure = True
            if usage_record is not None:
                usage_record.validation_status = "invalid"
            log(config, f"{agent_name}: {diagnostic}; no repair (format failure)")
        elif not text.strip():
            last_error = "agent response was empty"
            classification_text, provider_verdict = _agent_failure_classification(
                result, phase="empty"
            )
            last_classification_text = classification_text
            structured_verdict = (
                provider_verdict
                if provider_verdict is not None and provider_verdict.source == "structured"
                else None
            )
            if structured_verdict is not None:
                reset_evidence_text = structured_verdict.authoritative_text
            should_retry = replacement_stability_failed or (
                structured_verdict.category == "transient"
                if structured_verdict is not None
                else _is_transient_agent_output(classification_text)
            )
            last_failure_category = _failure_category(
                classification_text, provider_verdict=structured_verdict
            )
            capacity = classify_antigravity_capacity(
                classification_text,
                returncode=result.returncode,
                empty_response=True,
                signatures=config.antigravity_quota_signatures,
            ) if backend_agent == "antigravity" else None
            provider_capacity = bool(capacity and capacity.is_capacity)
            if provider_capacity:
                should_retry = True
                last_failure_category = "transient"
            elif (
                capacity is not None
                and capacity.auth_failure
                and not replacement_stability_failed
            ):
                should_retry = False
                last_failure_category = "non-retryable"
        else:
            response_file_pre_status = (
                _response_file_structured_status(result.response_file_text)
                if result.response_file_text
                else None
            )
            recovery_contract_retry = False
            try:
                unavailable = parse_agent_unavailable(text)
                if unavailable is not None:
                    raise _AgentUnavailableResponse(unavailable)
                try:
                    marker_value = validate(text)
                except AgentLoopError as validation_exc:
                    if result.provider_error_text and not isinstance(
                        validation_exc, _ArchitectureImpactContractUnsatisfied
                    ):
                        # agy itself reported a failed turn and the fallback
                        # content did not validate (#2029): an operational
                        # provider failure, never a model format failure.
                        raise _ProviderEnvelopeFailure(validation_exc) from validation_exc
                    raise
                if response_file_pre_status == "leading-public-response-marker-recovered":
                    log(
                        config,
                        f"{agent_name}: response file contained stdout filtering marker and "
                        "validated after stripping it",
                    )
            except _ProviderEnvelopeFailure as exc:
                # Classify like a failed command: the typed provider error is
                # the terminal frame, so auth/billing stay non-retryable and
                # quota/capacity reach the quota path. No repair and no
                # format-failure transition for an operational failure.
                last_error = (
                    f"Antigravity reported a failed turn ({result.provider_error_text}); "
                    f"fallback response rejected: {exc.validation_error}"
                )[:2000]
                classification_text, _provider_verdict = _agent_failure_classification(
                    result, phase="command"
                )
                last_classification_text = classification_text
                should_retry = _is_transient_agent_output(classification_text)
                last_failure_category = _failure_category(classification_text)
                # A non-SUCCESS envelope is a failed invocation even at exit 0.
                capacity = classify_antigravity_capacity(
                    classification_text,
                    returncode=result.returncode or 1,
                    empty_response=False,
                    signatures=config.antigravity_quota_signatures,
                )
                provider_capacity = bool(capacity and capacity.is_capacity)
                if provider_capacity:
                    should_retry = True
                    last_failure_category = "transient"
                elif capacity.auth_failure:
                    should_retry = False
                    last_failure_category = "non-retryable"
                if usage_record is not None:
                    usage_record.validation_status = "invalid"
                log(
                    config,
                    f"{agent_name}: agy reported a failed turn and the fallback response did "
                    f"not validate ({last_failure_category})",
                )
            except _AgentUnavailableResponse as exc:
                unavailable = exc.unavailable
                last_error = (
                    f"agent explicitly reported it cannot continue ({unavailable.category}): "
                    f"{unavailable.summary}. Suggested action: {unavailable.suggested_action}"
                )
                classification_text = text
                last_classification_text = classification_text
                last_failure_category = "agent-unavailable"
                should_retry = unavailable.retryable
                log(
                    config,
                    f"{agent_name}: explicitly reported agent-unavailable "
                    f"({unavailable.category}, retryable={'yes' if unavailable.retryable else 'no'})",
                )
            except AgentLoopError as exc:
                last_error = str(exc)
                marker_safety_failure = "Current untrusted GitHub text contains reserved protocol marker(s):" in str(exc)
                structured_kind = _recognized_structured_public_response_kind(result.text)
                contract_unsatisfied = isinstance(exc, _ArchitectureImpactContractUnsatisfied)
                if contract_unsatisfied and structured_kind is None:
                    # A parsed-but-unsatisfied response is a deterministic
                    # field-scope defect, whatever its envelope kind.
                    classification_text = "structured response failed the architecture_impact contract"
                    public_text_is_transient = False
                    last_failure_category = "deterministic"
                elif structured_kind is not None:
                    # Keep validation context authoritative. The structured
                    # payload's prose is untrusted content and must not be
                    # treated as evidence of provider auth, billing, credit,
                    # timeout, or dirty-worktree failure.
                    classification_text = (
                        f"structured {structured_kind} response failed trusted validation"
                    )
                    public_text_is_transient = False
                    last_failure_category = "deterministic"
                else:
                    classification_text = _agent_failure_classification_text(result, phase="validation")
                    public_text_is_transient = _is_transient_public_response(
                        classification_text,
                        repair_expected_kind=repair_expected_kind,
                    )
                    last_failure_category = _failure_category(
                        classification_text,
                        public_response=True,
                        repair_expected_kind=repair_expected_kind,
                    )
                last_classification_text = classification_text
                if result.command_result is not None and result.command_result.capture_diagnostics:
                    last_failure_category = "transient"
                    public_text_is_transient = True
                if (
                    last_failure_category == "deterministic"
                    and not marker_safety_failure
                    and repair_expected_kind in {"plan_state", "plan_revision"}
                    and structured_kind in {"plan_state", "plan_revision"}
                    and structured_kind == repair_expected_kind
                ):
                    plan_validation_capture_eligible = True
                    plan_validation_exhaustion = DeterministicPlanValidationExhaustion(
                        candidate_kind=repair_expected_kind,
                        candidate_text=text,
                        diagnostic=str(exc),
                        candidate_digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    )
                if (
                    completion_recovery is not None
                    and not completion_recovery_attempted
                    and backend_agent == "claude"
                    and result.session_id
                    and looks_like_backgrounded_completion(classification_text)
                ):
                    # At most one same-session resume, ever, for this call
                    # (#588): mark it attempted before invoking so a failure
                    # cannot loop back into this branch on a later attempt.
                    completion_recovery_attempted = True
                    recovery_outcome = _attempt_claude_completion_recovery(
                        runner,
                        config=config,
                        completion_recovery=completion_recovery,
                        session_id=result.session_id,
                        validate=validate,
                        usage_context=usage_context,
                        run_id=usage_context.run_id if usage_context is not None else None,
                        role=role,
                        label=label,
                        timeout_seconds=timeout_seconds,
                        acquisition_result=validation_acquisition,
                        accept=(
                            lambda text, marker, acquisition=validation_acquisition:
                            accept_candidate(text, marker, acquisition)
                        ),
                    )
                    if recovery_outcome.validated is not None:
                        return recovery_outcome.validated
                    last_result = recovery_outcome.result
                    last_error = recovery_outcome.error
                    last_classification_text = recovery_outcome.classification_text
                    last_failure_category = recovery_outcome.failure_category
                    terminal_public_response = recovery_outcome.terminal_public_response
                    if recovery_outcome.contract_unsatisfied:
                        recovery_contract_retry = True
                        # No terminal comment: the unmet contract goes back to
                        # the same agent through the shared re-ask budget.
                        # The resume stays attempted, so there is never a
                        # second one.  The recovered response is the one
                        # rejected, so its producer model and session drive
                        # the re-ask; the resumed session is the fallback.
                        recovery_producer = recovery_outcome.result
                        if not recovery_producer.session_id:
                            recovery_producer = dataclasses_replace(
                                recovery_producer, session_id=result.session_id
                            )
                        if schedule_reask(
                            message=recovery_outcome.error,
                            result=recovery_producer,
                            hint_prompt=_architecture_contract_retry_prompt(
                                prompt, recovery_outcome.error
                            ),
                        ):
                            if usage_record is not None:
                                usage_record.validation_status = "invalid"
                            continue
                        should_retry = False
                    else:
                        should_retry = False
                        break
                if not recovery_contract_retry:
                    response_failure_is_unsupported = last_failure_category == "unsupported_model"
                    # Marker near-misses keep a first-attempt ordinary nudge
                    # only when no same-agent re-ask can be scheduled.
                    should_retry = replacement_stability_failed or public_text_is_transient or (
                        not response_failure_is_unsupported
                        and attempt == 1
                        and _is_retryable_marker_near_miss(classification_text)
                    )
                    if (
                        result.raw_output
                        and result.raw_output != classification_text
                        and _is_transient_agent_output(result.raw_output)
                        and not public_text_is_transient
                    ):
                        log(
                            config,
                            f"{agent_name}: transient diagnostics were present outside the public response",
                        )
                    response_file_status = None
                    if result.response_file_text:
                        response_file_status = response_file_pre_status
                        if response_file_status == "leading-public-response-marker-not-recoverable":
                            log(
                                config,
                                f"{agent_name}: response file contained stdout filtering marker but "
                                "the remainder was not recoverable",
                            )
                        elif response_file_status in {"markdown-or-prose", "fenced-or-markdown"}:
                            log(
                                config,
                                f"{agent_name}: public response file was not structured "
                                f"({response_file_status})",
                            )
                    response_file_not_structured = response_file_status in {
                        "leading-public-response-marker-not-recoverable",
                        "markdown-or-prose",
                        "fenced-or-markdown",
                    }
                    if (
                        result.response_file_text
                        and response_file_not_structured
                        and not response_failure_is_unsupported
                    ):
                        recovered = _recover_valid_structured_candidate(
                            result,
                            validate=validate,
                            expected_kind=repair_expected_kind,
                            config=config,
                            agent_name=agent_name,
                        )
                        if recovered is not None:
                            recovered_text, marker_value = recovered
                            if usage_record is not None:
                                usage_record.validation_status = "validated"
                            return _accepted_validated_response(
                                accept_candidate(recovered_text, marker_value),
                                session_id=result.session_id,
                                usage=usage,
                                model_used=result.model_used,
                                **_response_identity_fields(result, acquisition_result=validation_acquisition),
                            )
                    human_ack_context = _HumanRequirementsRecoveryContext(
                        surfaced_requirement_ids=tuple(repair_surfaced_requirement_ids or ()),
                        requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                    )

                    def recover_human_ack(candidate: str) -> ValidatedAgentResponse | None:
                        """Deterministic plan_revision acknowledgement recovery for a candidate."""
                        if not (
                            repair_expected_kind == "plan_revision"
                            and result.response_file_text
                            and _plan_revision_missing_human_acknowledgement(
                                candidate, context=human_ack_context
                            )
                        ):
                            return None
                        recovered = _recover_plan_revision_human_requirements_acknowledgement(
                            result,
                            **({} if candidate is result.text else {"text": candidate}),
                            validate=validate,
                            context=human_ack_context,
                            config=config,
                            agent_name=agent_name,
                        )
                        if recovered is None:
                            return None
                        recovered_text, recovered_marker = recovered
                        if usage_record is not None:
                            usage_record.validation_status = "validated"
                        return _accepted_validated_response(
                            accept_candidate(recovered_text, recovered_marker),
                            session_id=result.session_id,
                            usage=usage,
                            model_used=result.model_used,
                            **_response_identity_fields(result, acquisition_result=validation_acquisition),
                        )

                    if (
                        not response_failure_is_unsupported
                        and not isinstance(exc, UnknownPriorItemDispositionError)
                    ):
                        recovered_ack = recover_human_ack(result.text)
                        if recovered_ack is not None:
                            return recovered_ack
                    # Purely mechanical normalization runs for every turn, with
                    # no model call and independent of the retired repair flag
                    # (#2066); every rewritten candidate is re-validated.
                    normalization_allowed = (
                        not public_text_is_transient and not response_failure_is_unsupported
                    )
                    if (
                        normalization_allowed
                        and repair_expected_kind in {"plan_state", "plan_revision"}
                    ):
                        matrix_normalized = attempt_risk_test_matrix_string_list_normalization(
                            text, expected_kind=repair_expected_kind
                        )
                        if matrix_normalized is not None:
                            matrix_text, matrix_paths = matrix_normalized
                            matrix_notice = (
                                f"{agent_name}: normalized risk test matrix string field(s) "
                                f"to one-element list(s): {', '.join(matrix_paths)}"
                            )
                            try:
                                marker_value = validate(matrix_text)
                            except AgentLoopError as norm_exc:
                                log(config, matrix_notice)
                                # The normalized candidate is now the state every
                                # later check, prompt and capture must describe.
                                text = matrix_text
                                exc = norm_exc
                                last_error = str(norm_exc)
                                plan_validation_exhaustion = _refresh_plan_validation_capture(
                                    plan_validation_exhaustion,
                                    eligible=plan_validation_capture_eligible,
                                    text=matrix_text,
                                    diagnostic=str(norm_exc),
                                )
                            else:
                                log(config, matrix_notice)
                                if usage_record is not None:
                                    usage_record.validation_status = "validated"
                                return _accepted_validated_response(
                                    accept_candidate(matrix_text, marker_value),
                                    session_id=result.session_id,
                                    usage=usage,
                                    model_used=result.model_used,
                                    **_response_identity_fields(result, acquisition_result=validation_acquisition),
                                )
                    if normalization_allowed and repair_expected_kind == "plan_revision_patch":
                        disposition_normalized = attempt_semantic_patch_disposition_normalization(text)
                        if disposition_normalized is not None:
                            try:
                                marker_value = validate(disposition_normalized)
                            except AgentLoopError:
                                pass
                            else:
                                log(
                                    config,
                                    f"{agent_name}: deterministic semantic-patch disposition "
                                    "normalization recovered malformed response",
                                )
                                if usage_record is not None:
                                    usage_record.validation_status = "validated"
                                return _accepted_validated_response(
                                    accept_candidate(disposition_normalized, marker_value),
                                    session_id=result.session_id,
                                    usage=usage,
                                    model_used=result.model_used,
                                    **_response_identity_fields(result, acquisition_result=validation_acquisition),
                                )
                    # A semantic patch must always prove that every removed ID
                    # is canonically resolved history (#872).  An incomplete
                    # ledger demands the same proof for every kind (#862);
                    # without it the response is re-asked, never stripped.
                    strip_requires_history_proof = (
                        ledger_incomplete or repair_expected_kind == "plan_revision_patch"
                    )
                    normalized: str | None = None
                    # An authority rejection exposed only once the envelope is
                    # normalized is as unfixable by normalization as one on the
                    # raw text (#990).
                    normalized_evidence_rejection: NonRepairableEvidenceRejection | None = None
                    # The rejection of the envelope-normalized candidate, when
                    # normalization ran but did not make it acceptable: that
                    # candidate and its own defect become the turn's state, so
                    # the re-ask names the remaining defect and the
                    # post-re-ask degradation sees it (#2066).
                    normalized_rejection: AgentLoopError | None = None
                    if (
                        normalization_allowed
                        and repair_expected_kind in STRUCTURED_PUBLIC_RESPONSE_KINDS
                        and not (
                            isinstance(exc, UnknownPriorItemDispositionError)
                            and ledger_incomplete
                        )
                    ):
                        normalized = attempt_envelope_normalization(
                            text,
                            expected_kind=repair_expected_kind,
                        )
                        if normalized is not None:
                            try:
                                marker_value = validate(normalized)
                            except UnknownPriorItemDispositionError as norm_exc:
                                normalized_rejection = norm_exc
                                # Combined fix (issue #274): envelope normalization removed the
                                # trailing defect but the normalized candidate still has unknown
                                # prior dispositions. Try stripping them from the normalized text
                                # so both defects are resolved in one deterministic pass.
                                normalized_history_strip = (
                                    strip_requires_history_proof
                                    and _history_strip_allowed(
                                        normalized,
                                        norm_exc,
                                        resolved_history_item_ids=repair_resolved_history_item_ids,
                                        expected_kind=repair_expected_kind,
                                    )
                                )
                                if (
                                    not isinstance(exc, UnknownPriorItemDispositionError)
                                    and (
                                        not strip_requires_history_proof
                                        or normalized_history_strip
                                    )
                                    and repair_expected_kind in {"pr_review", "plan_review", "plan_revision", "plan_revision_patch"}
                                ):
                                    stripped_from_normalized = strip_unknown_prior_item_dispositions(
                                        normalized,
                                        allowed_ids=frozenset(norm_exc.allowed_ids),
                                        expected_kind=repair_expected_kind,
                                    )
                                    if stripped_from_normalized is not None:
                                        try:
                                            marker_value = validate(stripped_from_normalized)
                                        except AgentLoopError:
                                            recovered_ack = recover_human_ack(stripped_from_normalized)
                                            if recovered_ack is not None:
                                                return recovered_ack
                                        else:
                                            removed = ", ".join(sorted(norm_exc.unknown_ids))
                                            allowed_str = ", ".join(sorted(norm_exc.allowed_ids)) or "(none)"
                                            if normalized_history_strip:
                                                log(
                                                    config,
                                                    f"{agent_name}: combined envelope normalization and "
                                                    f"deterministic strip removed canonically resolved "
                                                    f"historical prior-item disposition ID(s) {removed} "
                                                    f"{_history_strip_reason(ledger_incomplete)}; "
                                                    f"allowed carried prior IDs: {allowed_str}",
                                                )
                                            else:
                                                log(
                                                    config,
                                                    f"{agent_name}: combined envelope normalization and "
                                                    f"deterministic strip recovered malformed response; "
                                                    f"removed prior-item ID(s) {removed}; "
                                                    f"allowed carried prior IDs: {allowed_str}",
                                                )
                                            if usage_record is not None:
                                                usage_record.validation_status = "validated"
                                            return _accepted_validated_response(
                                                accept_candidate(stripped_from_normalized, marker_value),
                                                session_id=result.session_id,
                                                usage=usage,
                                                model_used=result.model_used,
                                                **_response_identity_fields(result, acquisition_result=validation_acquisition),
                                            )
                            except NonRepairableEvidenceRejection as norm_exc:
                                normalized_evidence_rejection = norm_exc
                                normalized_rejection = norm_exc
                            except AgentLoopError as norm_exc:
                                normalized_rejection = norm_exc
                            else:
                                log(
                                    config,
                                    f"{agent_name}: envelope normalization recovered malformed response",
                                )
                                if usage_record is not None:
                                    usage_record.validation_status = "validated"
                                return _accepted_validated_response(
                                    accept_candidate(normalized, marker_value),
                                    session_id=result.session_id,
                                    usage=usage,
                                    model_used=result.model_used,
                                    **_response_identity_fields(result, acquisition_result=validation_acquisition),
                                )
                    if normalized_rejection is not None and normalized is not None:
                        text = normalized
                        exc = normalized_rejection
                        last_error = str(normalized_rejection)
                        plan_validation_exhaustion = _refresh_plan_validation_capture(
                            plan_validation_exhaustion,
                            eligible=plan_validation_capture_eligible,
                            text=normalized,
                            diagnostic=str(normalized_rejection),
                        )
                    history_strip = (
                        strip_requires_history_proof
                        and isinstance(exc, UnknownPriorItemDispositionError)
                        and _history_strip_allowed(
                            text,
                            exc,
                            resolved_history_item_ids=repair_resolved_history_item_ids,
                            expected_kind=repair_expected_kind,
                        )
                    )
                    if (
                        normalization_allowed
                        and isinstance(exc, UnknownPriorItemDispositionError)
                        and (not strip_requires_history_proof or history_strip)
                        and repair_expected_kind in {"pr_review", "plan_review", "plan_revision", "plan_revision_patch"}
                    ):
                        stripped_text = strip_unknown_prior_item_dispositions(
                            text,
                            allowed_ids=frozenset(exc.allowed_ids),
                            expected_kind=repair_expected_kind,
                        )
                        if stripped_text is not None:
                            try:
                                marker_value = validate(stripped_text)
                            except AgentLoopError:
                                recovered_ack = recover_human_ack(stripped_text)
                                if recovered_ack is not None:
                                    return recovered_ack
                            else:
                                removed = ", ".join(sorted(exc.unknown_ids))
                                allowed_str = ", ".join(sorted(exc.allowed_ids)) or "(none)"
                                if history_strip:
                                    log(
                                        config,
                                        f"{agent_name}: removed canonically resolved historical "
                                        f"prior-item disposition ID(s) {removed} "
                                        f"{_history_strip_reason(ledger_incomplete)}; "
                                        f"allowed carried prior IDs: {allowed_str}",
                                    )
                                else:
                                    log(
                                        config,
                                        f"{agent_name}: deterministically removed unknown prior-item "
                                        f"disposition ID(s) {removed}; allowed carried prior IDs: {allowed_str}",
                                    )
                                if usage_record is not None:
                                    usage_record.validation_status = "validated"
                                return _accepted_validated_response(
                                    accept_candidate(stripped_text, marker_value),
                                    session_id=result.session_id,
                                    usage=usage,
                                    model_used=result.model_used,
                                    **_response_identity_fields(result, acquisition_result=validation_acquisition),
                                )
                    payload_rejection = (
                        _semantic_patch_payload_rejection(
                            exc,
                            text=text,
                            normalized=normalized,
                            payload_validator=semantic_patch_payload_validator,
                        )
                        if (
                            repair_expected_kind == "plan_revision_patch"
                            and not public_text_is_transient
                            and not response_failure_is_unsupported
                        )
                        else None
                    )
                    evidence_rejection = (
                        exc
                        if isinstance(exc, NonRepairableEvidenceRejection)
                        else normalized_evidence_rejection
                    )
                    if payload_rejection is not None:
                        # A semantic patch payload rejection goes to the
                        # bounded replan, which re-asks the planner with the
                        # full plan context (#979).
                        candidate_text, rejection_diagnostic = payload_rejection
                        log(
                            config,
                            f"{agent_name}: semantic patch payload rejected ({rejection_diagnostic}); "
                            "routing to bounded replan",
                        )
                        bounded_replan_rejection = DeterministicPlanValidationExhaustion(
                            candidate_kind="plan_revision",
                            candidate_text=candidate_text,
                            diagnostic=rejection_diagnostic,
                            candidate_digest=hashlib.sha256(
                                candidate_text.encode("utf-8")
                            ).hexdigest(),
                        )
                        should_retry = False
                        last_failure_category = "deterministic"
                    else:
                        # A plain-prose response that names an auth, billing
                        # or similar provider stop is a provider failure, not a
                        # response defect: it keeps its stop and spends no
                        # re-ask.  An attempted structured response is re-asked
                        # whatever words its prose contains.
                        provider_stop_prose = (
                            last_failure_category in _PROVIDER_DEFINITIVE_FAILURE_CATEGORIES
                            and recover_payload(text) is None
                            and not text.lstrip().startswith(("{", "`"))
                        )
                        if (
                            not public_text_is_transient
                            and not response_failure_is_unsupported
                            and not provider_stop_prose
                        ):
                            # The single same-agent re-ask rule (#2066): the
                            # validator's message goes back to the agent that
                            # produced the response, at most twice per turn.
                            reask_cause = evidence_rejection if evidence_rejection is not None else exc
                            fresh_evidence = (
                                is_coder_turn
                                and isinstance(reask_cause, NonRepairableEvidenceRejection)
                                and reask_cause.reason in _FRESH_EVIDENCE_REASONS
                            )
                            if schedule_reask(
                                message=str(reask_cause),
                                result=result,
                                hint_prompt=_reask_hint_prompt(
                                    prompt,
                                    reask_cause,
                                    contract_unsatisfied=contract_unsatisfied,
                                    document_path_rejection=document_path_rejection,
                                ),
                                fresh_evidence=fresh_evidence,
                            ):
                                if usage_record is not None:
                                    usage_record.validation_status = "invalid"
                                continue
                            # No re-ask is left (or the coder head could not be
                            # read): the turn fails as before, without an
                            # ordinary retry of the same deterministic defect.
                            should_retry = False
                        if evidence_rejection is not None:
                            log(
                                config,
                                f"{agent_name}: semantic evidence rejected ({evidence_rejection}); "
                                "no same-agent re-ask left",
                            )
                            last_error = _semantic_evidence_last_error(evidence_rejection)
                            last_classification_text = _SEMANTIC_EVIDENCE_REJECTION_CLASSIFICATION
                            should_retry = False
                            last_failure_category = "deterministic"
                        elif (
                            (repair_expected_kind in {"plan_review", "pr_review"} or reviewer_judgement_turn)
                            and isinstance(exc, MissingJudgementFieldError)
                        ):
                            # Only the reviewer can supply its decision; the
                            # terminal error names the field and its values.
                            last_error = (
                                f"{exc} (the reviewer did not supply {exc.field_path} "
                                f"(allowed: {' | '.join(exc.allowed_values)}))"
                            )
                            last_classification_text = _JUDGEMENT_FIELD_CLASSIFICATION
                            last_failure_category = "deterministic"
                        elif (
                            last_failure_category == "deterministic"
                            and not marker_safety_failure
                            and (
                                missing_substance := review_substance_missing(
                                    text, expected_kind=repair_expected_kind
                                )
                            ) is not None
                        ):
                            # A reviewer turn that still carries no review
                            # substance after the re-asks (narration, a tool
                            # cut short) is a reviewer availability failure,
                            # never a verdict: keep the ordinary retry and
                            # Antigravity format-fallback policy (#2029).
                            plan_validation_exhaustion = None
                            plan_validation_capture_eligible = False
                            should_retry = True
                            format_failure = True
                            last_failure_category = "agent-unavailable"
                            last_error = f"{last_error} (no review substance: {missing_substance})"
                            last_classification_text = (
                                "reviewer response carried no recoverable review substance "
                                f"({missing_substance})"
                            )
                        elif (
                            isinstance(exc, TestObservationsShapeError)
                            and exc.fallback_degradable
                            and not public_text_is_transient
                            and not response_failure_is_unsupported
                            # A head-guard refusal is not exhaustion: the
                            # degradation is earned only by spent re-asks.
                            and reask_budget_exhausted
                            and reask_skip_detail is None
                        ):
                            # Re-ask first, degrade after the re-asks (#1626,
                            # #2066): accept the latest response with today's
                            # recorded drop of the non-array field.
                            try:
                                fallback = accept_test_observations_fallback(
                                    normalized if normalized is not None else text
                                )
                            except NonRepairableEvidenceRejection as fallback_exc:
                                fallback = None
                                last_error = _semantic_evidence_last_error(fallback_exc)
                                last_classification_text = _SEMANTIC_EVIDENCE_REJECTION_CLASSIFICATION
                                should_retry = False
                                last_failure_category = "deterministic"
                            if fallback is not None:
                                log(
                                    config,
                                    f"{agent_name}: same-agent re-asks exhausted; degraded "
                                    "test_observations",
                                )
                                if usage_record is not None:
                                    usage_record.validation_status = "validated"
                                return _accepted_validated_response(
                                    fallback,
                                    session_id=result.session_id,
                                    usage=usage,
                                    model_used=result.model_used,
                                    **_response_identity_fields(result, acquisition_result=validation_acquisition),
                                )
            else:
                if usage_record is not None:
                    usage_record.validation_status = "validated"
                return _accepted_validated_response(
                    accept_candidate(text, marker_value),
                    session_id=result.session_id,
                    usage=usage,
                    model_used=result.model_used,
                    **_response_identity_fields(result, acquisition_result=validation_acquisition),
                )

        if should_retry:
            quota = None
            if antigravity_attempts is not None:
                if capacity is not None:
                    attempt_seconds = time.monotonic() - attempt_started_at
                    if capacity.is_capacity:
                        frame_lines = capacity.frame.strip().splitlines()
                        log(
                            config,
                            f"Antigravity {result.model_used} capacity failure after "
                            f"{attempt_seconds:.1f}s: {frame_lines[0] if frame_lines else 'no frame'}",
                        )
                    quota = classify_antigravity_quota_exhaustion(
                        capacity, threshold=LONG_RESET_THRESHOLD_SECONDS
                    )
                    if quota is not None:
                        quota_dispatches_since_wait += 1
                        previous_model = antigravity_attempts.models[antigravity_attempts.model_index]
                        previous_group = antigravity_attempts.groups[antigravity_attempts.model_index]
                        quota_transition = antigravity_attempts.next_after_quota_exhaustion(
                            quota, attempt_seconds
                        )
                        if quota_transition == "fallback":
                            next_model = antigravity_attempts.models[antigravity_attempts.model_index]
                            if capacity.reason == "credits":
                                log(
                                    config,
                                    f'Antigravity model "{previous_model}" out of credits; '
                                    f'falling back to "{next_model}"',
                                )
                            log(
                                config,
                                f"{agent_name}: quota exhausted for {previous_model}; skipping quota "
                                f"group {previous_group} to "
                                f"{antigravity_attempts.models[antigravity_attempts.model_index]} "
                                "without delay",
                            )
                            continue
                        reservation = _antigravity_quota_wait()
                        if reservation is None:
                            raise _antigravity_stop_error(
                                classification_text, shared_limit=quota_transition == "shared-limit"
                            )
                        # Post-failure: replay the invocation that failed quota on
                        # the group that becomes eligible first (#1601).
                        _wait_for_quota_reset(reservation)
                        pending_quota_replay = last_dispatch
                        antigravity_attempts.exhaustion_jumps = 0
                        continue
            elif _QUOTA_RATE_LIMIT_RE.search(
                reset_text := (
                    reset_evidence_text if reset_evidence_text is not None else classification_text
                )
            ):
                reset_secs = _parse_rate_limit_reset_seconds(reset_text)
                if reset_secs is not None and reset_secs > LONG_RESET_THRESHOLD_SECONDS:
                    reservation = _reserve_quota_wait(reset_secs)
                    if reservation is None:
                        raise _quota_reset_error(reset_secs, classification_text)
                    # Wait outside run_agent_result (no subprocess or settings
                    # lock held), then replay the same invocation (#1601).
                    quota_dispatches_since_wait += 1
                    _wait_for_quota_reset(reservation)
                    pending_quota_replay = last_dispatch
                    continue
            format_failed_model = (
                antigravity_attempts.models[antigravity_attempts.model_index]
                if antigravity_attempts is not None
                else None
            )
            transition = (
                antigravity_attempts.next_after_failure(
                    retryable=should_retry,
                    provider_capacity=provider_capacity,
                    format_failure=format_failure,
                )
                if antigravity_attempts is not None
                else ("retry" if ordinary_retries_used < config.agent_max_retries else "stop")
            )
            if transition == "retry":
                if antigravity_attempts is None:
                    ordinary_retries_used += 1
                delay = _retry_delay(config, attempt)
                category = last_failure_category
                retry_attempt = (
                    ordinary_retries_used + 1
                    if antigravity_attempts is None
                    else attempt + 1
                )
                retry_budget = (
                    config.agent_max_retries + 1
                    if antigravity_attempts is None
                    else max_attempts
                )
                log(
                    config,
                    f"{agent_name}: {category} failure ({last_error}); "
                    f"retrying in {delay}s (attempt {retry_attempt}/{retry_budget})",
                )
                stability_wait = getattr(runner, "wait_for_executable_stability", None)
                if target_exec_retryable and target_exec_command and stability_wait is not None:
                    # Give a self-updating CLI a bounded chance to settle.  The
                    # result is deliberately ignored: an unstable binary still
                    # gets the ordinary bounded relaunch (#1226).
                    stability_wait(target_exec_command, deadline=None)
                runner.run(("sleep", str(delay)), cwd=active_workdir(config))
                continue
            if transition == "format-fallback":
                failures = antigravity_attempts.format_failures.get(format_failed_model, 0)
                log(
                    config,
                    f"{agent_name}: {format_failed_model} produced no recoverable structured "
                    f"payload {failures} time{'s' if failures != 1 else ''}; falling back to "
                    f"{antigravity_attempts.models[antigravity_attempts.model_index]}",
                )
                continue
            if transition == "fallback":
                log(
                    config,
                    f"{agent_name}: provider capacity exhausted for "
                    f"{result.model_used}; trying {antigravity_attempts.models[antigravity_attempts.model_index]} "
                    "without additional delay",
                )
                continue
        break

    if executable_replacement_provider is not None:
        replacement_detail = _executable_replacement_failure_detail(
            provider=executable_replacement_provider,
            reason=executable_replacement_reason,
            stability_error=self_update_stability_error,
        )
        context_details = [replacement_detail]
        if latest_replay_refusal_detail:
            context_details.append(latest_replay_refusal_detail)
        last_error = f"{'; '.join(context_details)}; final failure: {last_error}"
        classification_parts = [last_classification_text, replacement_detail]
        if latest_replay_refusal_detail:
            classification_parts.append(latest_replay_refusal_detail)
        last_classification_text = "\n".join(
            part for part in classification_parts if part
        ).strip()
        last_failure_category = (
            "self-update-interruption"
            if executable_replacement_provider == "claude"
            else "executable-replacement"
        )
    elif latest_replay_refusal_detail:
        # Refusal context is added only after all retry decisions. In
        # particular, the provider-derived category remains untouched.
        last_error = f"{latest_replay_refusal_detail}; final failure: {last_error}"
        last_classification_text = "\n".join(
            part for part in (last_classification_text, latest_replay_refusal_detail) if part
        ).strip()
    if reask_budget_exhausted:
        last_error = (
            REASK_TERMINAL_PREFIX.format(
                count=reasks_used, plural="" if reasks_used == 1 else "s"
            )
            + f"; last validator message: {last_error}"
        )
    elif reask_skip_detail is not None:
        last_error = f"{last_error}; {reask_skip_detail}"
    diagnostics = _failed_run_diagnostics(
        runner=runner,
        config=config,
        agent_name=agent_name,
        salvage_context=salvage_context,
        operation_description=operation_description,
        failure_category=last_failure_category,
        failure_reason=last_error,
        classification_text=last_classification_text,
        marker_description=marker_description,
        result=last_result,
    )
    message = _format_invalid_agent_response_error(
        agent_name=agent_name,
        marker_description=marker_description,
        reason=last_error,
        result=last_result,
        log_paths=log_paths,
        category=last_failure_category,
        agent=agent,
        config=config,
        role=role,
        classification_text=last_classification_text,
    )
    if repair_expected_kind == "issue_implementation" and config.managed_ci:
        message += " The implementation response was rejected before a PR number was accepted."
        if managed_ci_recovery_protection in waivable_protection_states(config):
            message += (
                " If the coder opened a PR before that rejection, discover and resume that same PR "
                "with explicit --managed-ci-fresh authorization (including the unprotected waiver"
                + (
                    ": --allow-unprotected-managed-ci --allow-unreadable-protection"
                    if managed_ci_recovery_protection == "unreadable"
                    else ""
                )
                + "); do not rerun implementation to recreate it."
            )
        elif managed_ci_recovery_protection == "unreadable":
            message += (
                " If the coder opened a PR before that rejection, use the ordinary managed-CI "
                "issue/PR discovery and resume path for that same PR. Fresh authorization is "
                "unavailable because unreadable branch protection also requires "
                f"{waiver_flags_for_protection('unreadable')}; do not rerun implementation "
                "to recreate the PR."
            )
        else:
            message += (
                " If the coder opened a PR before that rejection, use the ordinary managed-CI "
                "issue/PR discovery and resume path for that same PR. The exceptional "
                "unprotected fresh-authorization path is unavailable here; do not rerun "
                "implementation to recreate the PR."
            )
    message += _agy_log_suffix()
    message += diagnostics.format_for_error()
    if (
        last_failure_category not in {"deterministic", "fresh-contract-integrity"}
        or not plan_validation_capture_eligible
    ):
        plan_validation_exhaustion = None
    invocation_error = AgentInvocationError(
        message,
        failure_category=last_failure_category,
        terminal_public_response=terminal_public_response,
        containment=last_result.containment if last_result is not None else None,
        plan_validation_exhaustion=plan_validation_exhaustion,
        preserved_unsatisfied_response=preserved_unsatisfied,
        bounded_replan_rejection=bounded_replan_rejection,
        rejected_candidate=(
            RejectedCandidate(
                text=last_candidate[0],
                test_turn_id=getattr(last_candidate[1], "test_turn_id", None),
                test_turn_observations=tuple(getattr(last_candidate[1], "test_turn_observations", ()) or ()),
                session_id=getattr(last_candidate[1], "session_id", None),
                model_used=getattr(last_candidate[1], "model_used", None),
            )
            if last_candidate is not None and last_candidate[1] is not None
            else None
        ),
    )
    if (
        plan_validation_failure_handler is not None
        and plan_validation_exhaustion is not None
    ):
        plan_validation_failure_handler(plan_validation_exhaustion, invocation_error)
    raise invocation_error
