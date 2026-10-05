"""Run telemetry, structured repair, completion recovery and the validated agent turn.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1192); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import datetime
import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .agents.base import AgentName, AgentResult
from .agents.antigravity import AntigravityAttemptState
from .agents.registry import agent_display_name, agent_signature, run_agent_result
from .config import AgentLoopConfig
from .errors import (
    CITATION_REASKABLE_REASONS,
    AgentInvocationError,
    AgentLoopError,
    DeterministicPlanValidationExhaustion,
    FreshContractIntegrityError,
    PreservedUnsatisfiedResponse,
    NonRepairableEvidenceRejection,
    QuotaResetExceededError,
    ReviewSubstanceIntegrityError,
    SemanticPatchPayloadRejection,
    MissingJudgementFieldError,
    MissingPriorItemDispositionError,
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
from .managed_ci import waivable_protection_states, waiver_flags_for_protection
from .prompts import build_completion_recovery_prompt
from .protocol import AgentUnavailable, parse_agent_unavailable, ParseDegradation
from .repair import (
    CandidateDecision,
    RepairAttemptResult,
    attempt_envelope_normalization,
    attempt_risk_test_matrix_string_list_normalization,
    attempt_semantic_patch_disposition_normalization,
    attempt_repair,
    execute_repair,
    strip_unknown_prior_item_dispositions,
    unknown_dispositions_are_resolved_history,
    require_recoverable_fresh_execution_contract,
    require_recoverable_fresh_risk_test_matrix_contract,
    require_recoverable_review_substance,
)
from .repair_preservation import (
    normalize_architecture_impact_near_miss,
    recover_payload,
    require_recoverable_semantic_patch,
    require_repair_architecture_impact_absent,
    validate_repair_preservation,
)
from .runner import Runner
from .salvage import SalvageContext
from .transient import (
    classify_antigravity_capacity,
    classify_antigravity_quota_exhaustion,
    looks_like_backgrounded_completion,
)
from .usage import RunUsageContext, UsageMetadata, estimate_usage
from .worker_telemetry import append_record, run_record, telemetry_log_path
from .workdirs import active_workdir
from .comment_rendering import render_agent_unavailable_comment
from .round_state import ApprovedPlanContext
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
    _carrier_payload,
    _unwrap_architecture_result,
    _architecture_contract_diagnostic,
    _attach_architecture_degradations,
    _AcceptedCandidate,
    _AcceptedTextCanonicalizationError,
    _with_validation_context,
    _accept_candidate,
    _accepted_validated_response,
    _architecture_contract_retry_prompt,
    _fresh_matrix_contract_retry_prompt,
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
    log(
        config,
        "Usage summary written to "
        f"{usage_context.summary_path} "
        f"(calls={totals.call_count}, exact={totals.exact_calls}, "
        f"partial={totals.partial_calls}, estimated={totals.estimated_calls}, "
        f"{tool_provenance.format_tool_commit_suffix(usage_context.tool_provenance)})",
    )


_ORIGINAL_ATTEMPT_REPAIR = attempt_repair


def _run_structured_repair(
    raw: str,
    *,
    runner: Runner,
    config: AgentLoopConfig,
    usage_context: RunUsageContext | None,
    validate: Callable[[str], object],
    repair_kwargs: dict[str, object],
    require_architecture_impact_contract: bool = False,
    degrade_architecture_impact: bool = False,
    forbid_architecture_impact: bool = False,
    parse_response: Callable[[str], object] | None = None,
    contract_refusal: Callable[[object, str, tuple[ParseDegradation, ...]], str | None] | None = None,
) -> tuple[str | None, object | None, list[RepairAttemptResult]]:
    """Run configured repair, retaining compatibility with patched legacy test hooks.

    Every new parameter is an optional control keyword, never a repair-prompt
    keyword, and defaults to today's behavior.  For an invocation that enabled
    degradation, a near miss is normalized in the raw payload before the
    repair prompt is built; its record travels out of band.  Each candidate is
    parsed with the non-refusing ``parse_response``, checked by preservation
    (including the absence pin), given the merged records, and only then
    offered to ``contract_refusal`` -- all before it can be accepted (#925).
    """
    if parse_response is None:
        parse_response = validate
    if repair_kwargs.get("expected_kind") == "plan_revision_patch":
        try:
            require_recoverable_semantic_patch(raw)
        except AgentLoopError as exc:
            return None, None, [
                RepairAttemptResult(
                    backend="none",
                    model="semantic-patch-integrity",
                    prompt="",
                    output=raw,
                    returncode=None,
                    outcome="semantic_patch_integrity",
                    diagnostic=str(exc),
                    log_path=None,
                    fallback_planned=False,
                )
            ]
    review_expected_kind = repair_kwargs.get("expected_kind")
    if review_expected_kind in {"plan_review", "pr_review"}:
        try:
            require_recoverable_review_substance(raw, expected_kind=review_expected_kind)
        except ReviewSubstanceIntegrityError as exc:
            # No configuration may bypass this: a reviewer turn with no review
            # substance has no verdict for repair to recover.
            return None, None, [
                RepairAttemptResult(
                    backend="none",
                    model="review-substance-integrity",
                    prompt="",
                    output=raw,
                    returncode=None,
                    outcome="review_substance_integrity",
                    diagnostic=str(exc),
                    log_path=None,
                    fallback_planned=False,
                )
            ]
    if repair_kwargs.get("require_execution_strategy_contract"):
        expected_kind = repair_kwargs.get("expected_kind")
        if isinstance(expected_kind, str):
            try:
                require_recoverable_fresh_execution_contract(raw, expected_kind=expected_kind)
            except FreshContractIntegrityError as exc:
                return None, None, [
                    RepairAttemptResult(
                        backend="none",
                        model="fresh-contract-integrity",
                        prompt="",
                        output=raw,
                        returncode=None,
                        outcome="fresh_contract_integrity",
                        diagnostic=str(exc),
                        log_path=None,
                        fallback_planned=False,
                        integrity_contract="execution_recommendation",
                    )
                ]
    if repair_kwargs.get("require_risk_test_matrix_contract"):
        expected_kind = repair_kwargs.get("expected_kind")
        if isinstance(expected_kind, str):
            matrix_normalized = attempt_risk_test_matrix_string_list_normalization(
                raw, expected_kind=expected_kind
            )
            if matrix_normalized is not None:
                raw = matrix_normalized[0]
                log(
                    config,
                    "repair guard: normalized risk test matrix string field(s) "
                    f"to one-element list(s): {', '.join(matrix_normalized[1])}",
                )
            try:
                require_recoverable_fresh_risk_test_matrix_contract(
                    raw, expected_kind=expected_kind
                )
            except FreshContractIntegrityError as exc:
                return None, None, [
                    RepairAttemptResult(
                        backend="none",
                        model="fresh-matrix-contract-integrity",
                        prompt="",
                        output=raw,
                        returncode=None,
                        outcome="fresh_contract_integrity",
                        diagnostic=str(exc),
                        log_path=None,
                        fallback_planned=False,
                        integrity_contract="risk_test_matrix",
                    )
                ]
    degradation_records: tuple[ParseDegradation, ...] = ()
    forbid = forbid_architecture_impact
    if degrade_architecture_impact:
        near_miss = normalize_architecture_impact_near_miss(
            raw,
            required_contract=require_architecture_impact_contract,
            expected_kind=(
                repair_kwargs.get("expected_kind")
                if isinstance(repair_kwargs.get("expected_kind"), str) else None
            ),
        )
        raw = near_miss.raw
        degradation_records = (near_miss.record,) if near_miss.record is not None else ()
        forbid = forbid or near_miss.forbid_architecture_impact

    def candidate_refusal(output: str, parsed: object) -> CandidateDecision:
        # Merge, never replace: source records first, then any record the
        # candidate's own degradable parse produced.
        record_bearing = _attach_architecture_degradations(parsed, degradation_records)
        payload = _carrier_payload(record_bearing)
        records = tuple(payload.architecture_impact_degradations) if payload is not None else ()
        refusal = (
            contract_refusal(record_bearing, output, records)
            if contract_refusal is not None else None
        )
        return CandidateDecision(parsed=record_bearing, refusal=refusal)

    if attempt_repair is not _ORIGINAL_ATTEMPT_REPAIR:
        try:
            repaired = attempt_repair(raw, config.gemini_cmd, **repair_kwargs)
        except TypeError as exc:
            # Keep older test/integration hooks callable while the reviewer-ID
            # context is rolled out. The real repair API accepts this keyword.
            if not any(
                name in str(exc)
                for name in (
                    "reviewer_requirement_ids",
                    "require_execution_strategy_contract",
                    "require_risk_test_matrix_contract",
                    "reject_unsolicited_risk_test_matrix_contract",
                )
            ):
                raise
            legacy_kwargs = dict(repair_kwargs)
            legacy_kwargs.pop("reviewer_requirement_ids", None)
            legacy_kwargs.pop("require_execution_strategy_contract", None)
            legacy_kwargs.pop("require_risk_test_matrix_contract", None)
            legacy_kwargs.pop("reject_unsolicited_risk_test_matrix_contract", None)
            repaired = attempt_repair(raw, config.gemini_cmd, **legacy_kwargs)
        if repaired is None:
            return None, None, []
        try:
            parsed = parse_response(repaired)
            if repair_kwargs.get("expected_kind") in {
                "plan_revision_patch", "plan_review", "pr_review",
            }:
                validate_repair_preservation(
                    raw,
                    repaired,
                    allowed_prior_item_ids=repair_kwargs.get("allowed_prior_item_ids"),
                    forbid_architecture_impact=forbid,
                )
            elif forbid:
                require_repair_architecture_impact_absent(repaired)
        except AgentLoopError as exc:
            return repaired, None, [
                RepairAttemptResult(
                    backend="gemini",
                    model="legacy-test-hook",
                    prompt="",
                    output=repaired,
                    returncode=0,
                    outcome="invalid_output",
                    diagnostic=str(exc),
                    log_path=None,
                    fallback_planned=False,
                )
            ]
        decision = candidate_refusal(repaired, parsed)
        if decision.refusal is None:
            return repaired, decision.parsed, []
        payload = _carrier_payload(decision.parsed)
        return repaired, None, [
            RepairAttemptResult(
                backend="gemini",
                model="legacy-test-hook",
                prompt="",
                output=repaired,
                returncode=0,
                outcome="architecture_contract_unsatisfied",
                diagnostic=decision.refusal,
                log_path=None,
                fallback_planned=False,
                validation_result=decision.parsed,
                architecture_impact_degradations=(
                    tuple(payload.architecture_impact_degradations) if payload is not None else ()
                ),
            )
        ]
    return execute_repair(
        raw,
        runner=runner,
        config=config,
        run_id=usage_context.run_id if usage_context is not None else None,
        usage_context=usage_context,
        validate=parse_response,
        forbid_architecture_impact=forbid,
        candidate_refusal=candidate_refusal,
        **repair_kwargs,
    )


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


def _attempt_claude_completion_recovery(
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


def _refused_contract_attempt(
    attempts: Sequence[RepairAttemptResult],
) -> RepairAttemptResult | None:
    """The last repair candidate refused for an unsatisfied contract, if any."""
    for attempt in reversed(attempts):
        if attempt.outcome == "architecture_contract_unsatisfied":
            return attempt
    return None


def _log_repair_attempts(config: AgentLoopConfig, prefix: str, attempts: Sequence[RepairAttemptResult]) -> None:
    for attempt in attempts:
        diagnostic = attempt.diagnostic or "(none)"
        log(
            config,
            f"{prefix}: repair backend={attempt.backend} model={attempt.model} "
            f"outcome={attempt.outcome} returncode="
            f"{attempt.returncode if attempt.returncode is not None else 'none'}; "
            f"diagnostic={diagnostic}; log={attempt.log_path or '(none)'}; "
            f"fallback_planned={'yes' if attempt.fallback_planned else 'no'}",
        )


def _capture_terminal_plan_repair_rejection(
    attempt: RepairAttemptResult,
    *,
    repair_expected_kind: str | None,
    validate: Callable[[str], object],
    contract_diagnostic: Callable[[object], str | None] | None = None,
) -> DeterministicPlanValidationExhaustion | None:
    """Capture a deterministically rejected final planning repair candidate.

    Repair diagnostics can also describe preservation checks or backend output,
    so re-run the authoritative validator and retain only its exact diagnostic.
    Shape and marker-safety checks keep unrelated or unsafe repair output out of
    the durable planning-diagnostic channel.
    """
    candidate = attempt.output
    if (
        attempt.outcome != "invalid_output"
        or repair_expected_kind not in {"plan_state", "plan_revision"}
        or _recognized_structured_public_response_kind(candidate) != repair_expected_kind
    ):
        return None
    try:
        parsed = validate(candidate)
    except AgentLoopError as exc:
        if "Current untrusted GitHub text contains reserved protocol marker(s):" in str(exc):
            return None
        return DeterministicPlanValidationExhaustion(
            candidate_kind=repair_expected_kind,
            candidate_text=candidate,
            diagnostic=str(exc),
            candidate_digest=hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        )
    # ``validate`` is the non-refusing parse on required-contract sites; the
    # pure diagnostic check never mutates the retained unsatisfied candidate.
    diagnostic = contract_diagnostic(parsed) if contract_diagnostic is not None else None
    if diagnostic is not None:
        return DeterministicPlanValidationExhaustion(
            candidate_kind=repair_expected_kind,
            candidate_text=candidate,
            diagnostic=diagnostic,
            candidate_digest=hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        )
    return None


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


def _run_validated_agent(
    runner: Runner,
    *,
    agent: AgentName,
    config: AgentLoopConfig,
    prompt: str,
    marker_description: str,
    validate: Callable[[str], object],
    session_id: str | None = None,
    usage_context: RunUsageContext | None = None,
    use_repair: bool = False,
    repair_expected_kind: str | None = None,
    repair_unresolved_item_ids: Sequence[str] | None = None,
    repair_surfaced_requirement_ids: Sequence[str] | None = None,
    repair_reviewer_requirement_ids: Sequence[str] | None = None,
    repair_requires_direct_discussion_ack: bool = False,
    require_execution_strategy_contract: bool = False,
    require_risk_test_matrix_contract: bool = False,
    reject_unsolicited_risk_test_matrix_contract: bool = False,
    repair_allowed_prior_item_ids: Sequence[str] | None = None,
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
    reask_on_prior_disposition_omission: bool = False,
    reask_on_missing_judgement_field: bool = False,
    reask_on_evidence_rejection: bool = False,
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
    validation_acquisition: AgentResult | None = None
    # The last response refused for an unsatisfied architecture-impact
    # contract.  It is retained and surfaced, never silently discarded.
    preserved_unsatisfied: PreservedUnsatisfiedResponse | None = None

    def contract_refusal(
        result: object, text: str, records: tuple[ParseDegradation, ...] = ()
    ) -> str | None:
        """Store an unsatisfied result as the preserved candidate; return its diagnostic."""
        nonlocal preserved_unsatisfied
        if not require_architecture_impact_contract:
            return None
        diagnostic = _architecture_contract_diagnostic(result)
        if diagnostic is None:
            return None
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
        return _with_validation_context(
            response_validator, runner=runner, acquisition=validation_acquisition
        )(text)

    def accept_candidate(
        text: str, marker_value: object, acquisition: AgentResult | None = None
    ) -> _AcceptedCandidate:
        # The acquisition is captured when the candidate is accepted, never
        # read later from the mutable closure, so a repair turn cannot change
        # the catalog the strict re-parse sees.
        return _accept_candidate(
            text,
            marker_value,
            strict_revalidate=strict_revalidate,
            runner=runner,
            acquisition=acquisition if acquisition is not None else validation_acquisition,
        )

    agent_name = agent_display_name(agent)
    operation_description = operation_description or _operation_description_from_context(
        salvage_context=salvage_context,
        repair_expected_kind=repair_expected_kind,
        role=role,
        label=label,
        marker_description=marker_description,
    )
    log_paths: list[object] = []
    antigravity_attempts = (
        AntigravityAttemptState.from_config(config, config.agent_max_retries)
        if agent == "antigravity"
        else None
    )
    # Each fallback retains an initial attempt; retry allowance is shared. The
    # provider-specific replacement replay gets one explicit extra slot.
    max_attempts = (
        len(config.antigravity_models) + config.agent_max_retries + 2
        if antigravity_attempts is not None
        else config.agent_max_retries + 2
    )
    if reask_on_prior_disposition_omission or reask_on_missing_judgement_field:
        # One dedicated slot shared by both re-ask kinds (#1167, #1185); that never consumes agent_max_retries.
        max_attempts += 1
    if reask_on_evidence_rejection:
        # One dedicated slot that never consumes agent_max_retries (#1240).
        max_attempts += 1
    if require_risk_test_matrix_contract:
        # One dedicated planner replay for a fresh matrix integrity refusal;
        # never consumes agent_max_retries or Antigravity fallback state.
        max_attempts += 1
    last_error =f"{agent_name} produced no output."
    last_result: AgentResult | None = None
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
    marker_safety_repair_attempted = False
    # A field-naming diagnostic for the next attempt's prompt, set only by an
    # unsatisfied architecture-impact contract and consumed by one attempt.
    pending_contract_reprompt: str | None = None
    omission_reask_used = False
    matrix_integrity_replay_used = False
    terminal_integrity_contract: str | None = None
    pending_matrix_integrity_reprompt: str | None = None
    pending_omission_reask_ids: tuple[str, ...] | None = None
    pending_judgement_reask: tuple[str, tuple[str, ...], str | None] | None = None
    evidence_reask_used = False
    pending_evidence_reask_detail: str | None = None
    pending_evidence_reask_session: str | None = None
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
        message += diagnostics.format_for_error()
        return QuotaResetExceededError(message)

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
            state.unavailable_message(shared_limit), failure_category="transient"
        )

    for attempt in range(1, max_attempts + 1):
        replacement_stability_failed = False
        if (
            antigravity_attempts is not None
            and antigravity_attempts.ensure_eligible_before_attempt() == "all-exhausted"
        ):
            log(config, f"{agent_name}: every Antigravity quota group is cooling down; not invoking")
            raise _antigravity_stop_error(last_classification_text, shared_limit=False)
        attempt_started_at = time.monotonic()
        attempt_config = (
            antigravity_attempts.singleton_config(config)
            if antigravity_attempts is not None
            else config
        )
        invocation_kwargs: dict[str, object] = {}
        is_executable_replacement_replay = executable_replacement_replay_pending
        if agent == "claude":
            invocation_kwargs["attempt_suffix"] = (
                "self-update-attempt2"
                if is_executable_replacement_replay
                else f"attempt{ordinary_retries_used + 1}"
            )
        elif is_executable_replacement_replay:
            invocation_kwargs["attempt_suffix"] = executable_replacement_policies[agent][2]
        if pending_matrix_integrity_reprompt is not None:
            attempt_prompt = _fresh_matrix_contract_retry_prompt(
                prompt, pending_matrix_integrity_reprompt
            )
            if agent == "claude":
                invocation_kwargs["attempt_suffix"] = "matrix-integrity-replay"
            pending_matrix_integrity_reprompt = None
        elif pending_judgement_reask is not None:
            # Same frozen prompt and session as the original turn (#1185).
            attempt_prompt = _missing_judgement_field_reask_prompt(
                prompt, *pending_judgement_reask
            )
            if agent == "claude":
                invocation_kwargs["attempt_suffix"] = "judgement-reask"
            pending_judgement_reask = None
        elif pending_omission_reask_ids is not None:
            # Reuse the frozen prompt and session: publishing or rebuilding the
            # prompt between the two turns would reopen #1156.
            attempt_prompt = _prior_disposition_omission_reask_prompt(
                prompt, pending_omission_reask_ids
            )
            if agent == "claude":
                invocation_kwargs["attempt_suffix"] = "omission-reask"
            pending_omission_reask_ids = None
        else:
            attempt_prompt = (
                _architecture_contract_retry_prompt(prompt, pending_contract_reprompt)
                if pending_contract_reprompt is not None
                else prompt
            )
        pending_contract_reprompt = None
        attempt_session_id = session_id
        if pending_evidence_reask_detail is not None:
            # Same frozen prompt plus the sanitized rejection; resume the
            # rejected acquisition's own session when one exists (#1240).
            attempt_prompt = _evidence_rejection_reask_prompt(
                prompt, pending_evidence_reask_detail
            )
            attempt_session_id = pending_evidence_reask_session
            if agent == "claude":
                invocation_kwargs["attempt_suffix"] = "evidence-reask"
            pending_evidence_reask_detail = None
            pending_evidence_reask_session = None
        if usage_context is not None:
            usage_context.note_agent_dispatch()
        result = run_agent_result(
            runner,
            agent=agent,
            config=attempt_config,
            prompt=attempt_prompt,
            session_id=attempt_session_id,
            run_id=usage_context.run_id if usage_context is not None else None,
            role=role,
            label=label,
            timeout_seconds=next_timeout_seconds,
            **invocation_kwargs,
        )
        # Keep this acquisition result fixed through every validation and
        # repair callback for the current response. Repair itself may replace
        # runner.latest_test_turn_id, but it cannot replace this snapshot.
        validation_acquisition = result
        # The bounded deadline belongs only to the interrupted invocation and
        # its dedicated replay. A later ordinary retry has its normal budget.
        if is_executable_replacement_replay:
            executable_replacement_replay_pending = False
            next_timeout_seconds = timeout_seconds
        last_result = result
        # A typed planning-validation candidate is valid only for the final
        # deterministic failure of this invocation. A later timeout, provider,
        # marker-safety, or containment failure must clear it.
        plan_validation_exhaustion = None
        plan_validation_capture_eligible = False
        bounded_replan_rejection = None
        if result.log_path is not None:
            log_paths.append(result.log_path)
        text = _neutralize_untrusted_markers(result.text, config=config, agent_name=agent_name)
        usage = _resolve_usage_metadata(config=config, prompt=attempt_prompt, result=result)
        usage_record = None
        if usage_context is not None and usage is not None:
            usage_record = usage_context.add_record(
                agent=agent,
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
                        **_response_identity_fields(result),
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
            and containment.backend == "systemd-cgroup-v2"
            and not containment.cleanup_confirmed
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
            agent in executable_replacement_policies
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
                executable_replacement_policies[agent]
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
        should_retry = False
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
            should_retry = True
            pending_contract_reprompt = last_error
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
            ) if agent == "antigravity" else None
            provider_capacity = bool(capacity and capacity.is_capacity)
            if provider_capacity:
                should_retry = True
                last_failure_category = "transient"
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
            ) if agent == "antigravity" else None
            provider_capacity = bool(capacity and capacity.is_capacity)
            if provider_capacity:
                should_retry = True
                last_failure_category = "transient"
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
                marker_value = validate(text)
                if response_file_pre_status == "leading-public-response-marker-recovered":
                    log(
                        config,
                        f"{agent_name}: response file contained stdout filtering marker and "
                        "validated after stripping it",
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
                    reask_on_missing_judgement_field
                    and not omission_reask_used
                    and isinstance(exc, MissingJudgementFieldError)
                    and not public_text_is_transient
                    and not marker_safety_failure
                    and last_failure_category != "unsupported_model"
                    and not contract_unsatisfied
                ):
                    # A missing reviewer judgement cannot be supplied by repair:
                    # re-ask the same reviewer once through the shared slot (#1185).
                    omission_reask_used = True
                    pending_judgement_reask = (
                        exc.field_path, exc.allowed_values, exc.observed_preview
                    )
                    if usage_record is not None:
                        usage_record.validation_status = "invalid"
                    log(
                        config,
                        f"{agent_name}: review gave no valid {exc.field_path} (allowed: "
                        f"{' | '.join(exc.allowed_values)}); re-asking the same reviewer "
                        "once (no repair)",
                    )
                    continue
                if (
                    reask_on_prior_disposition_omission
                    and not omission_reask_used
                    and isinstance(exc, MissingPriorItemDispositionError)
                    and not public_text_is_transient
                    and not marker_safety_failure
                    and last_failure_category != "unsupported_model"
                ):
                    # A missing carried-item disposition is a missing judgement
                    # that only the reviewer can supply: re-ask the same reviewer
                    # once, outside the retry budget and the Antigravity model
                    # fallback, and never through repair (#1167).
                    omission_reask_used = True
                    pending_omission_reask_ids = tuple(exc.missing_ids)
                    if usage_record is not None:
                        usage_record.validation_status = "invalid"
                    log(
                        config,
                        f"{agent_name}: review omitted disposition(s) for carried item(s) "
                        f"{', '.join(pending_omission_reask_ids)}; re-asking the same reviewer "
                        "once (no repair)",
                    )
                    continue
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
                    and agent == "claude"
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
                        # No repair, no terminal comment: one ordinary retry
                        # with the field-naming re-prompt.  The resume stays
                        # attempted, so there is never a second one.
                        should_retry = True
                        pending_contract_reprompt = recovery_outcome.error
                    else:
                        should_retry = False
                        break
                if not recovery_contract_retry:
                    response_failure_is_unsupported = last_failure_category == "unsupported_model"
                    # Marker near-misses are a separate first-attempt nudge for common footer typos;
                    # structured JSON protocol drift still remains repairable when retries are exhausted.
                    should_retry = replacement_stability_failed or public_text_is_transient or (
                        not response_failure_is_unsupported
                        and attempt == 1
                        and _is_retryable_marker_near_miss(classification_text)
                    )
                    if contract_unsatisfied:
                        # A contract-only failure skips the futile repair pass and
                        # takes one ordinary retry with a field-naming re-prompt,
                        # within the existing budget and attempt bound.
                        should_retry = True
                        pending_contract_reprompt = str(exc)
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
                                **_response_identity_fields(result),
                            )
                    if (
                        not response_failure_is_unsupported
                        and repair_expected_kind == "plan_revision"
                        and result.response_file_text
                        and not isinstance(exc, UnknownPriorItemDispositionError)
                        and _plan_revision_missing_human_acknowledgement(
                            result.text,
                            context=_HumanRequirementsRecoveryContext(
                                surfaced_requirement_ids=tuple(
                                    repair_surfaced_requirement_ids or ()
                                ),
                                requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                            ),
                        )
                    ):
                        recovered = _recover_plan_revision_human_requirements_acknowledgement(
                            result,
                            validate=validate,
                            context=_HumanRequirementsRecoveryContext(
                                surfaced_requirement_ids=tuple(
                                    repair_surfaced_requirement_ids or ()
                                ),
                                requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                            ),
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
                                **_response_identity_fields(result),
                            )
                    if (
                        use_repair
                        and not public_text_is_transient
                        and not response_failure_is_unsupported
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
                                    **_response_identity_fields(result),
                                )
                    if (
                        use_repair
                        and not public_text_is_transient
                        and not response_failure_is_unsupported
                        and repair_expected_kind == "plan_revision_patch"
                    ):
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
                                    **_response_identity_fields(result),
                                )
                    # A semantic patch can only ever be recovered by the deterministic
                    # strip (repair must preserve it byte-for-byte), so it always has to
                    # prove that every removed ID is canonically resolved history (#872).
                    # An incomplete ledger demands the same proof for every kind (#862).
                    strip_requires_history_proof = (
                        ledger_incomplete or repair_expected_kind == "plan_revision_patch"
                    )
                    normalized: str | None = None
                    # An authority rejection exposed only once the envelope is
                    # normalized is as unrepairable as one on the raw text (#990).
                    normalized_evidence_rejection: NonRepairableEvidenceRejection | None = None
                    if (
                        use_repair
                        and not public_text_is_transient
                        and not response_failure_is_unsupported
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
                                # Combined fix (issue #274): envelope normalization removed the
                                # trailing defect but the normalized candidate still has unknown
                                # prior dispositions. Try stripping them from the normalized text
                                # so both defects are resolved in one deterministic pass.
                                # Only apply when the original error was structural; when it was
                                # already UnknownPriorItemDispositionError, block 2 handles it.
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
                                            if (
                                                repair_expected_kind == "plan_revision"
                                                and result.response_file_text
                                                and _plan_revision_missing_human_acknowledgement(
                                                    stripped_from_normalized,
                                                    context=_HumanRequirementsRecoveryContext(
                                                        surfaced_requirement_ids=tuple(
                                                            repair_surfaced_requirement_ids or ()
                                                        ),
                                                        requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                                                    ),
                                                )
                                            ):
                                                recovered = _recover_plan_revision_human_requirements_acknowledgement(
                                                    result,
                                                    text=stripped_from_normalized,
                                                    validate=validate,
                                                    context=_HumanRequirementsRecoveryContext(
                                                        surfaced_requirement_ids=tuple(
                                                            repair_surfaced_requirement_ids or ()
                                                        ),
                                                        requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                                                    ),
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
                                                        **_response_identity_fields(result),
                                                    )
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
                                                **_response_identity_fields(result),
                                            )
                            except NonRepairableEvidenceRejection as norm_exc:
                                normalized_evidence_rejection = norm_exc
                            except AgentLoopError:
                                pass
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
                                    **_response_identity_fields(result),
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
                        use_repair
                        and not public_text_is_transient
                        and not response_failure_is_unsupported
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
                                if (
                                    repair_expected_kind == "plan_revision"
                                    and result.response_file_text
                                    and _plan_revision_missing_human_acknowledgement(
                                        stripped_text,
                                        context=_HumanRequirementsRecoveryContext(
                                            surfaced_requirement_ids=tuple(
                                                repair_surfaced_requirement_ids or ()
                                            ),
                                            requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                                        ),
                                    )
                                ):
                                    recovered = _recover_plan_revision_human_requirements_acknowledgement(
                                        result,
                                        text=stripped_text,
                                        validate=validate,
                                        context=_HumanRequirementsRecoveryContext(
                                            surfaced_requirement_ids=tuple(
                                                repair_surfaced_requirement_ids or ()
                                            ),
                                            requires_direct_discussion_ack=repair_requires_direct_discussion_ack,
                                        ),
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
                                            **_response_identity_fields(result),
                                        )
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
                                    **_response_identity_fields(result),
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
                    if payload_rejection is not None:
                        # Repair may change only a semantic patch's envelope, and
                        # this rejection names its payload: no repair output can
                        # satisfy both. Hand it to the bounded replan (#979).
                        candidate_text, rejection_diagnostic = payload_rejection
                        log(
                            config,
                            f"{agent_name}: semantic patch payload rejected ({rejection_diagnostic}); "
                            "repair may change only the envelope, routing to bounded replan",
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
                    elif (
                        evidence_rejection := (
                            exc
                            if isinstance(exc, NonRepairableEvidenceRejection)
                            else normalized_evidence_rejection
                        )
                    ) is not None:
                        # Selecting a real broker handle that is not an
                        # authoritative passing observation is an authority
                        # decision. Repair may only reshape the envelope around
                        # the coder's claims, so it cannot satisfy this; running
                        # it (and its fallback chain) only burns its timeout and
                        # then misreports the stop as that timeout (#990).
                        if (
                            reask_on_evidence_rejection
                            and not evidence_reask_used
                            and evidence_rejection.reason in CITATION_REASKABLE_REASONS
                            and not public_text_is_transient
                            and not marker_safety_failure
                            and not response_failure_is_unsupported
                        ):
                            # A non-citable selected observation is missing
                            # judgement only the coder can supply (rerun and
                            # re-cite): re-ask once, outside the retry budget
                            # and fallback, never via repair (#1240).
                            evidence_reask_used = True
                            pending_evidence_reask_detail = str(evidence_rejection)
                            pending_evidence_reask_session = result.session_id or session_id
                            if usage_record is not None:
                                usage_record.validation_status = "invalid"
                            log(
                                config,
                                f"{agent_name}: semantic evidence rejected ({evidence_rejection}); "
                                "re-asking the coder once to cite a verified passing observation "
                                "(no repair)"
                                + (
                                    "; resuming session"
                                    if pending_evidence_reask_session
                                    else "; fresh turn"
                                ),
                            )
                            continue
                        log(
                            config,
                            f"{agent_name}: semantic evidence rejected ({evidence_rejection}); "
                            "not repairable by reformatting, skipping repair pass",
                        )
                        last_error = (
                            f"{evidence_rejection} (semantic evidence rejection; "
                            "repair skipped because reformatting cannot change it)"
                        )
                        last_classification_text = _SEMANTIC_EVIDENCE_REJECTION_CLASSIFICATION
                        should_retry = False
                        last_failure_category = "deterministic"
                    elif (
                        repair_expected_kind in {"plan_review", "pr_review"}
                        and isinstance(exc, MissingJudgementFieldError)
                    ):
                        # Reformatting cannot supply the reviewer's decision (#1185).
                        allowed = " | ".join(exc.allowed_values)
                        log(
                            config,
                            f"{agent_name}: reviewer gave no valid {exc.field_path} "
                            f"(allowed: {allowed}); not repairable by reformatting, "
                            "skipping repair pass",
                        )
                        last_error = (
                            f"{exc} (repair skipped because the reviewer did not supply "
                            f"{exc.field_path} (allowed: {allowed}); a repair pass cannot "
                            "supply the reviewer's decision)"
                        )
                        last_failure_category = "deterministic"
                    elif (
                        use_repair
                        and not public_text_is_transient
                        and not response_failure_is_unsupported
                        and (not marker_safety_failure or not marker_safety_repair_attempted)
                        and not (
                            isinstance(exc, UnknownPriorItemDispositionError)
                            and ledger_incomplete
                        )
                        # Repair may not supply an assessment the response lacks,
                        # so a response whose only defect is the unsatisfied
                        # contract has nothing left for a repair model to fix.
                        and not contract_unsatisfied
                    ):
                        log(config, f"{agent_name}: schema validation failed ({exc}); attempting repair pass")
                        repair_kwargs: dict[str, object] = {"expected_kind": repair_expected_kind}
                        if repair_unresolved_item_ids is not None:
                            repair_kwargs["unresolved_item_ids"] = tuple(repair_unresolved_item_ids)
                        if repair_expected_kind in {"issue_implementation", "plan_state", "plan_revision"}:
                            repair_kwargs["surfaced_requirement_ids"] = tuple(
                                repair_surfaced_requirement_ids or ()
                            )
                            repair_kwargs["requires_direct_discussion_ack"] = (
                                repair_requires_direct_discussion_ack
                            )
                            if (
                                require_execution_strategy_contract
                                and repair_expected_kind in {"plan_state", "plan_revision"}
                            ):
                                repair_kwargs["require_execution_strategy_contract"] = True
                            if (
                                require_risk_test_matrix_contract
                                and repair_expected_kind in {"plan_state", "plan_revision"}
                            ):
                                repair_kwargs["require_risk_test_matrix_contract"] = True
                            if (
                                reject_unsolicited_risk_test_matrix_contract
                                and repair_expected_kind == "plan_revision"
                            ):
                                repair_kwargs["reject_unsolicited_risk_test_matrix_contract"] = True
                        elif (
                            repair_expected_kind == "coder_followup"
                            and (
                                repair_surfaced_requirement_ids is not None
                                or repair_requires_direct_discussion_ack
                            )
                        ):
                            repair_kwargs["surfaced_requirement_ids"] = tuple(repair_surfaced_requirement_ids or ())
                            repair_kwargs["requires_direct_discussion_ack"] = repair_requires_direct_discussion_ack
                        elif (
                            repair_expected_kind in {"plan_review", "pr_review"}
                            and repair_reviewer_requirement_ids is not None
                        ):
                            repair_kwargs["reviewer_requirement_ids"] = tuple(
                                repair_reviewer_requirement_ids
                            )
                        if isinstance(exc, UnknownPriorItemDispositionError):
                            repair_kwargs["allowed_prior_item_ids"] = exc.allowed_ids
                            repair_kwargs["unknown_prior_item_ids"] = exc.unknown_ids
                            repair_kwargs["same_round_context"] = exc.same_round_description
                        elif repair_allowed_prior_item_ids is not None:
                            repair_kwargs["allowed_prior_item_ids"] = tuple(repair_allowed_prior_item_ids)
                        original_validation_error = str(exc)
                        if marker_safety_failure:
                            marker_safety_repair_attempted = True
                        repaired, repaired_marker, repair_attempts = _run_structured_repair(
                            normalized if normalized is not None else text,
                            runner=runner,
                            config=config,
                            usage_context=usage_context,
                            validate=validate,
                            repair_kwargs=repair_kwargs,
                            require_architecture_impact_contract=require_architecture_impact_contract,
                            degrade_architecture_impact=degrade_architecture_impact,
                            parse_response=parse_response,
                            contract_refusal=(
                                contract_refusal if require_architecture_impact_contract else None
                            ),
                        )
                        _log_repair_attempts(config, agent_name, repair_attempts)
                        terminal_repair = repair_attempts[-1] if repair_attempts else None
                        # A refused contract candidate outranks every later
                        # non-accepting failure in the same chain: it is a complete
                        # response whose only defect is the contract.
                        refused_contract = (
                            _refused_contract_attempt(repair_attempts)
                            if repaired_marker is None else None
                        )
                        if refused_contract is not None:
                            # Deterministic, never a repair-provider failure; the
                            # planning capture is built from the refused candidate
                            # itself so the retained candidate and records are kept.
                            last_failure_category = "deterministic"
                            last_classification_text = (
                                f"structured {repair_expected_kind} repair failed the "
                                "architecture_impact contract"
                            )
                            should_retry = True
                            pending_contract_reprompt = refused_contract.diagnostic
                            if repair_expected_kind in {"plan_state", "plan_revision"}:
                                plan_validation_exhaustion = DeterministicPlanValidationExhaustion(
                                    candidate_kind=repair_expected_kind,
                                    candidate_text=refused_contract.output,
                                    diagnostic=refused_contract.diagnostic,
                                    candidate_digest=hashlib.sha256(
                                        refused_contract.output.encode("utf-8")
                                    ).hexdigest(),
                                )
                                plan_validation_capture_eligible = True
                        elif (
                            terminal_repair is not None
                            and terminal_repair.outcome == "fresh_contract_integrity"
                        ):
                            # A fresh planning response with no mechanically
                            # recoverable contract must not be repaired by
                            # synthesis. Execution-recommendation integrity keeps
                            # its established planner replay, while a risk-matrix
                            # integrity failure is deterministic and fail-fast.
                            # The original structured validator rejection remains
                            # authoritative for terminal diagnostic persistence.
                            should_retry = terminal_repair.integrity_contract == "execution_recommendation"
                            if (
                                terminal_repair.integrity_contract == "risk_test_matrix"
                                and not matrix_integrity_replay_used
                            ):
                                matrix_integrity_replay_used = True
                                pending_matrix_integrity_reprompt = original_validation_error
                            terminal_integrity_contract = terminal_repair.integrity_contract
                            last_failure_category = "fresh-contract-integrity"
                            last_classification_text = (
                                "fresh planning execution recommendation requires a new planner turn"
                                if terminal_repair.integrity_contract == "execution_recommendation"
                                else "fresh planning risk-test-matrix contract is not mechanically recoverable"
                            )
                        elif (
                            terminal_repair is not None
                            and terminal_repair.outcome == "review_substance_integrity"
                        ):
                            # The reviewer's own turn produced no review, usually
                            # because its tooling cut the turn short. That is a
                            # reviewer availability failure, not a verdict: retry
                            # within the configured policy and never let repair
                            # synthesize a blocking item on the reviewer's behalf.
                            plan_validation_exhaustion = None
                            plan_validation_capture_eligible = False
                            if (
                                last_failure_category
                                not in _PROVIDER_DEFINITIVE_FAILURE_CATEGORIES
                            ):
                                # A provider/credential diagnostic the reviewer's own
                                # output already named stays authoritative: refusing
                                # its repair says nothing about availability, and a
                                # rerun cannot fix an auth or billing failure.
                                should_retry = True
                                last_failure_category = "agent-unavailable"
                                last_classification_text = (
                                    "reviewer response carried no recoverable review substance; "
                                    "repair refused"
                                )
                        elif (
                            terminal_repair is not None
                            and terminal_repair.outcome == "semantic_patch_integrity"
                        ):
                            # A malformed semantic payload has no authenticated
                            # decision set for an envelope-only repair to retain.
                            # Give the planner a fresh attempt; never let a repair
                            # model synthesize operations, rationales, or bindings.
                            should_retry = True
                            last_failure_category = "semantic-patch-integrity"
                            last_classification_text = (
                                "semantic patch is not mechanically recoverable; planner retry required"
                            )
                        elif terminal_repair is not None:
                            repaired_exhaustion = _capture_terminal_plan_repair_rejection(
                                terminal_repair,
                                repair_expected_kind=repair_expected_kind,
                                validate=(
                                    parse_response if require_architecture_impact_contract else validate
                                ),
                                contract_diagnostic=(
                                    _architecture_contract_diagnostic
                                    if require_architecture_impact_contract else None
                                ),
                            )
                            if repaired_exhaustion is not None:
                                # The repair candidate, rather than the source
                                # candidate, is the final deterministic rejection.
                                plan_validation_exhaustion = repaired_exhaustion
                                plan_validation_capture_eligible = True
                                last_failure_category = "deterministic"
                                last_classification_text = (
                                    f"structured {repair_expected_kind} repair failed trusted validation"
                                )
                            else:
                                # A terminal repair transport/provider failure (or
                                # non-matching/unsafe output) cannot persist stale
                                # source-candidate provenance.
                                plan_validation_exhaustion = None
                                plan_validation_capture_eligible = False
                                if terminal_repair.outcome == "timeout":
                                    last_failure_category = "timeout"
                                elif terminal_repair.outcome != "invalid_output":
                                    # Includes a terminal transient_provider_error
                                    # (agy model-access failure), even after
                                    # earlier invalid_output attempts: it is a
                                    # resumable provider failure, not a
                                    # deterministic plan-validation rejection.
                                    last_failure_category = "repair-provider-failure"
                        if repaired is not None:
                            if repaired_marker is None:
                                repair_detail = (
                                    repair_attempts[-1].diagnostic
                                    if repair_attempts
                                    else "repair output failed validation"
                                )
                                last_error = (
                                    f"{original_validation_error}; repair failure: {repair_detail}"
                                )
                                log(
                                    config,
                                    f"{agent_name}: repair pass produced invalid output ({repair_detail})",
                                )
                            else:
                                marker_value = repaired_marker
                                try:
                                    # The repaired candidate was parsed under
                                    # the primary attempt's acquisition, never
                                    # the repair turn's.
                                    accepted_repair = accept_candidate(
                                        repaired, marker_value, validation_acquisition
                                    )
                                except _AcceptedTextCanonicalizationError as canon_exc:
                                    accepted_repair = None
                                    last_error = f"{original_validation_error}; repair failure: {canon_exc}"
                                    last_failure_category = "deterministic"
                                    log(config, f"{agent_name}: repaired response not accepted ({canon_exc})")
                            if repaired_marker is not None and accepted_repair is not None:
                                if isinstance(exc, UnknownPriorItemDispositionError):
                                    removed = ", ".join(sorted(exc.unknown_ids))
                                    allowed = ", ".join(sorted(exc.allowed_ids)) or "(none)"
                                    log(
                                        config,
                                        f"{agent_name}: repair pass removed unknown prior-item "
                                        f"disposition ID(s) {removed}; allowed carried prior IDs: {allowed}",
                                    )
                                else:
                                    log(config, f"{agent_name}: repair pass recovered malformed response")
                                if usage_record is not None:
                                    usage_record.validation_status = "validated"
                                return _accepted_validated_response(
                                    accepted_repair,
                                    session_id=result.session_id,
                                    usage=usage,
                                    model_used=result.model_used,
                                    **_response_identity_fields(result),
                                )
                        elif repair_attempts:
                            details = "; ".join(
                                f"{attempt.backend}/{attempt.model}: {attempt.outcome}"
                                + (f" ({attempt.diagnostic})" if attempt.diagnostic else "")
                                for attempt in repair_attempts
                            )
                            last_error = f"{original_validation_error}; repair invocation failure: {details}"
            else:
                if usage_record is not None:
                    usage_record.validation_status = "validated"
                return _accepted_validated_response(
                    accept_candidate(text, marker_value),
                    session_id=result.session_id,
                    usage=usage,
                    model_used=result.model_used,
                    **_response_identity_fields(result),
                )

        if pending_matrix_integrity_reprompt is not None:
            if usage_record is not None:
                usage_record.validation_status = "invalid"
            log(
                config,
                f"{agent_name}: retrying planner turn once (fresh risk-test-matrix "
                "contract; outside retry budget)",
            )
            continue
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
                        previous_model = antigravity_attempts.models[antigravity_attempts.model_index]
                        previous_group = antigravity_attempts.groups[antigravity_attempts.model_index]
                        quota_transition = antigravity_attempts.next_after_quota_exhaustion(
                            quota, attempt_seconds
                        )
                        if quota_transition == "fallback":
                            log(
                                config,
                                f"{agent_name}: quota exhausted for {previous_model}; skipping quota "
                                f"group {previous_group} to "
                                f"{antigravity_attempts.models[antigravity_attempts.model_index]} "
                                "without delay",
                            )
                            continue
                        raise _antigravity_stop_error(
                            classification_text, shared_limit=quota_transition == "shared-limit"
                        )
            elif _QUOTA_RATE_LIMIT_RE.search(classification_text):
                reset_secs = _parse_rate_limit_reset_seconds(classification_text)
                if reset_secs is not None and reset_secs > LONG_RESET_THRESHOLD_SECONDS:
                    raise _quota_reset_error(reset_secs, classification_text)
            transition = (
                antigravity_attempts.next_after_failure(
                    retryable=should_retry, provider_capacity=provider_capacity
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
    if matrix_integrity_replay_used and last_failure_category == "fresh-contract-integrity":
        if terminal_integrity_contract == "risk_test_matrix":
            replay_note = (
                "one automatic planner replay was already attempted "
                "and also failed the fresh risk-test-matrix contract"
            )
        else:
            replay_note = (
                "one automatic planner replay for the risk-test-matrix contract was "
                f"already attempted; the final failure is the {terminal_integrity_contract} contract"
            )
        last_error = f"{last_error}; {replay_note}"
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
    )
    if (
        plan_validation_failure_handler is not None
        and plan_validation_exhaustion is not None
    ):
        plan_validation_failure_handler(plan_validation_exhaustion, invocation_error)
    raise invocation_error
