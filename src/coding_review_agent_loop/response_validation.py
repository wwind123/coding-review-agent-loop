"""Coder/plan response validators and human-requirement checks.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1193); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace as dataclasses_replace
from pathlib import Path
from typing import TypeVar
from .agents.registry import run_agent_result
from .config import AgentLoopConfig
from .errors import (
    CheckoutVerificationError,
    AgentLoopError,
    IssueImplementationConflictError,
    SemanticPatchPayloadRejection,
    SemanticPatchUnknownPriorItemDispositionError,
    UnknownPriorItemDispositionError,
)
from .github import (
    IssueContext,
    PullRequestReviewContext,
    HumanReviewRequirement,
    deduplicate_human_requirements,
    post_issue_comment,
    validate_open_pr,
)
from .logging import log
from .prompts import render_coder_human_requirements_prompt_context
from .protocol import (
    ApprovedFollowups,
    ExecutionStrategyRecommendation,
    PlanReviewItems,
    ReviewItemDisposition,
    StructuredCoderFollowup,
    StructuredIssueImplementation,
    DerivedRiskEvidenceResult,
    PostAuthClaimDiagnostic,
    SemanticRiskCoverageClaims,
    DEGRADED_ROW_CLAIM_DIAGNOSTIC,
    UNAPPROVED_ROW_CLAIM_DIAGNOSTIC,
    StructuredPlanState,
    StructuredPlanRevision,
    PlanRevisionPatch,
    parse_plan_revision_patch,
    StructuredTaskResult,
    UnresolvedReviewItem,
    human_requirements_resolved,
    is_clarification_request,
    parse_agent_state,
    parse_pr_number,
    validate_human_requirements_acknowledgement,
    validate_human_requirement_dispositions,
    validate_structured_coder_followup,
    validate_structured_human_requirements_acknowledgement,
    validate_structured_issue_implementation,
    validate_structured_plan_state,
    validate_structured_plan_revision,
    validate_structured_plan_revision_patch,
    validate_structured_task_result,
    derive_risk_test_matrix_evidence,
    semantic_risk_claim_schema_text,
)
from .runner import Runner
from .local_test_evidence import reconcile_test_observations, stable_tracked_tree_snapshot
from .workdirs import active_workdir
from .workdir_guard import (
    validate_response_tests_within_workdir,
    command_is_admissible_evidence,
    command_targets_outside_workdir,
    partition_reported_tests_by_workdir,
    validate_test_observation_citations_within_workdir,
)
from .comment_rendering import normalize_freeform_signature, render_public_agent_comment
from .followups import _approved_followup_from_unresolved_item, _dedupe_approved_followups
from .round_state import ApprovedPlanContext, PostedRoundRecord, RequirementsContext
from .agent_failure import ValidatedAgentResponse
from .architecture_contract import (
    _VALIDATION_TEST_TURN_CONTEXT,
    _TerminalNoPrImplementation,
    _TerminalIssueImplementationConflict,
    architecture_impact_contract_unsatisfied,
)


@dataclass(frozen=True)
class ResolvedExecution:
    """The single post-approval routing decision.

    ``requested_policy`` records the caller's intent, while ``action`` is the
    canonical downstream action.  In particular, ``auto`` is never itself a
    topology or dispatch mode after this boundary.
    """

    requested_policy: str
    action: str
    strategy: str | None
    recommendation: ExecutionStrategyRecommendation | None

    @property
    def is_automatic(self) -> bool:
        return self.requested_policy == "auto"


def _validate_issue_implementation_contract(
    parsed: StructuredIssueImplementation,
    *,
    human_requirements,
) -> None:
    prompt_context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="implementation requirements",
        full_omission_fallback="Fetch the issue discussion directly before implementing.",
    )
    validate_human_requirement_dispositions(
        parsed.human_requirement_dispositions,
        surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
        context="issue_implementation.human_requirement_dispositions",
    )
    validate_structured_human_requirements_acknowledgement(
        parsed.human_requirements.addressed_ids,
        dispositions=parsed.human_requirement_dispositions,
        checked_discussion_directly=parsed.human_requirements.checked_discussion_directly,
        surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
        requires_direct_discussion_ack=prompt_context.requires_direct_discussion_ack,
    )


def _validate_issue_implementation_response(
    text: str,
    *,
    human_requirements,
    require_architecture_impact: bool = False,
    delivered_risk_test_matrix: object = None,
    delivered_risk_test_matrix_identity: str | None = None,
    require_risk_test_matrix_contract: bool = False,
    authoritative_test_observations=None,
    delivered_risk_test_matrix_row_ids=None,
    execution_catalog=None,
    architecture_status_mode: str,
) -> StructuredIssueImplementation | _TerminalNoPrImplementation | _TerminalIssueImplementationConflict:
    """Validate an implementation result and isolate the terminal conflict path."""
    if is_clarification_request(text):
        return _TerminalNoPrImplementation("clarification")
    try:
        parsed = validate_structured_issue_implementation(
            text,
            required_architecture_impact_contract=(1 if require_architecture_impact else 0),
            delivered_risk_test_matrix=delivered_risk_test_matrix,
            delivered_risk_test_matrix_identity=delivered_risk_test_matrix_identity,
            required_risk_test_matrix_contract=(1 if require_risk_test_matrix_contract else 0),
            authoritative_test_observations=authoritative_test_observations,
            delivered_risk_test_matrix_row_ids=delivered_risk_test_matrix_row_ids,
            execution_catalog=execution_catalog,
            architecture_status_mode=architecture_status_mode,
        )
    except IssueImplementationConflictError as exc:
        parsed = exc.payload
        if not isinstance(parsed, StructuredIssueImplementation):
            raise AgentLoopError("Issue implementation conflict did not retain a typed payload.") from exc
        # A conflict is terminal only after the normal acknowledgement and
        # exact-ledger contract has been proven valid for this issue.
        _validate_issue_implementation_contract(parsed, human_requirements=human_requirements)
        return _TerminalIssueImplementationConflict(parsed)
    if parsed is None:
        raise AgentLoopError(
            "Issue implementation response must use the required structured `issue_implementation` format."
        )
    _validate_issue_implementation_contract(parsed, human_requirements=human_requirements)
    return parsed


def _current_test_turn_observations(runner: Runner) -> tuple[object, ...]:
    """Return the closed invocation-local catalog, never the cumulative journal."""
    validation_context = _VALIDATION_TEST_TURN_CONTEXT.get()
    if validation_context is not None and validation_context[0] is runner:
        return validation_context[2]
    turn_id = getattr(runner, "latest_test_turn_id", None)
    if not isinstance(turn_id, str) or not turn_id:
        return ()
    current = getattr(runner, "current_test_turn_observations", None)
    if callable(current):
        observations = tuple(current())
    else:
        observations = tuple(runner.local_test_observations())

    def observation_turn_id(observation: object) -> object:
        if isinstance(observation, Mapping):
            return observation.get("turn_id")
        return getattr(observation, "turn_id", None)

    return tuple(
        observation for observation in observations
        if observation_turn_id(observation) == turn_id
    )


def _admissible_evidence_observations(
    observations: Sequence[object],
    *,
    assigned_workdir: Path,
    selectable: bool = True,
) -> tuple[object, ...]:
    """Drop broker runs that targeted paths outside the assigned checkout.

    The broker confines ``cwd`` but not test operands, so a run of a clean
    base-branch baseline can carry a valid selector. It stays visible as
    context, but can never back a risk-matrix citation (#991).

    ``selectable=True`` filters the execution catalog: anything the guard
    cannot prove in-checkout is dropped.  ``selectable=False`` filters the
    failure journal: only runs proven to target another location are dropped,
    so an unvalidatable failure still degrades the rows it would affect.
    """

    def keep(command: str | tuple[str, ...]) -> bool:
        if selectable:
            return command_is_admissible_evidence(command, assigned_workdir=assigned_workdir)
        return not command_targets_outside_workdir(command, assigned_workdir=assigned_workdir)

    def argv(observation: object) -> object:
        if isinstance(observation, Mapping):
            return observation.get("argv", observation.get("command", ()))
        return getattr(observation, "command", ())

    kept: list[object] = []
    for observation in observations:
        command = argv(observation)
        if isinstance(command, str):
            admissible = keep(command)
        elif isinstance(command, Sequence):
            admissible = keep(tuple(str(item) for item in command))
        else:
            admissible = not selectable
        if admissible:
            kept.append(observation)
    return tuple(kept)


def _safe_execution_handle_catalog(observations: Sequence[object]) -> tuple[dict[str, object], ...]:
    """Project correction context without exposing receipt authority."""
    catalog: list[dict[str, object]] = []
    for observation in observations:
        execution_ref = getattr(observation, "execution_ref", None)
        if not isinstance(execution_ref, str) or not execution_ref:
            continue
        attribution = getattr(observation, "attribution", None)
        if isinstance(attribution, Mapping):
            attribution_state = attribution.get("state")
        else:
            attribution_state = getattr(attribution, "state", None)
        caveats = getattr(observation, "caveats", ())
        catalog.append({
            "execution_ref": execution_ref,
            "outcome": getattr(observation, "outcome", "unknown"),
            "provenance": getattr(observation, "provenance", "unknown"),
            "attribution_state": attribution_state or "unknown",
            "caveats": [str(item) for item in tuple(caveats)[:4]],
        })
    return tuple(catalog)


def _post_auth_correction_prompt(
    parsed: StructuredIssueImplementation | StructuredCoderFollowup,
    *,
    matrix_row_ids: Sequence[str],
    diagnostics: Sequence[PostAuthClaimDiagnostic],
    execution_catalog: Sequence[Mapping[str, object]],
) -> str:
    return (
        "The orchestrator authenticated the PR, then found a bounded semantic "
        "coverage defect. This is a claims-only correction in the same coder "
        "session; do not change code, PR identity, reviewer-item classifications, "
        "human-requirement fields, summary, or any other coder-owned fact. Return "
        "the same structured response kind and correct only "
        "`risk_test_matrix_claims`. Do not emit canonical evidence, matrix "
        "identities, canonical rows, receipt IDs, citations, mappings, statuses, "
        "or envelope bookkeeping. Use only approved row IDs and handles in the "
        "closed catalog below; a missing claim is allowed when no admissible "
        "execution exists. Command strings and handles outside this catalog "
        "are dropped and leave the row unverified, so omit a claim that has no "
        "admissible handle instead of inventing one. "
        + semantic_risk_claim_schema_text()
        + " If you cannot truthfully state a fact, leave it empty rather than "
        "inventing it; the row then stays unverified.\n\n"
        f"Response kind: {parsed.kind}\n"
        f"Approved enforceable row IDs: {json.dumps(list(matrix_row_ids))}\n"
        f"Post-authentication diagnostics: {json.dumps([item.to_payload() for item in diagnostics])}\n"
        f"Closed execution-handle catalog: {json.dumps(list(execution_catalog))}\n"
    )


def _parse_fresh_correction_claims(
    text: str,
    original: StructuredIssueImplementation | StructuredCoderFollowup,
    *,
    row_ids: Sequence[str],
    execution_catalog: Sequence[object],
) -> StructuredIssueImplementation | StructuredCoderFollowup | None:
    if isinstance(original, StructuredIssueImplementation):
        candidate = validate_structured_issue_implementation(
            text,
            architecture_status_mode="legacy",
            delivered_risk_test_matrix_row_ids=row_ids,
            execution_catalog=execution_catalog,
        )
    else:
        candidate = validate_structured_coder_followup(
            text,
            architecture_status_mode="legacy",
            delivered_risk_test_matrix_row_ids=row_ids,
            execution_catalog=execution_catalog,
        )
    if candidate is None or candidate.kind != original.kind:
        return None
    # The correction continuation is not a second coder handoff. Preserve all
    # coder-owned facts from the authenticated response and accept only its
    # newly validated semantic claim set.  Claims the original response lost
    # to an unapproved row (#920) or any other row-ID defect (#926) stay on
    # the audit record even when the correction omits them.
    claims = candidate.risk_test_matrix_claims
    original_claims = original.risk_test_matrix_claims
    original_dropped = original_claims.dropped_row_ids if original_claims is not None else ()
    original_degradations = original_claims.degradations if original_claims is not None else ()
    original_exact_ids = (
        original_claims.unapproved_claim_row_ids if original_claims is not None else ()
    )
    if original_dropped or original_degradations:
        claims = claims or SemanticRiskCoverageClaims()
        # Each drop is its own audit record, so records are concatenated, not
        # deduplicated; the correction's records are relabeled so a drop at
        # the same index in both responses keeps a distinct position.
        relabeled = {
            record: dataclasses_replace(
                record, element_path=f"correction.{record.element_path}"
            )
            for record in claims.degradations
        }
        claims = dataclasses_replace(
            claims,
            dropped_row_ids=tuple(dict.fromkeys((*original_dropped, *claims.dropped_row_ids))),
            degradations=(
                *original_degradations,
                *(relabeled[record] for record in claims.degradations),
            ),
            unapproved_claim_row_ids=(
                *original_exact_ids,
                *(
                    (relabeled.get(record, record), row_id)
                    for record, row_id in claims.unapproved_claim_row_ids
                ),
            ),
        )
    return dataclasses_replace(
        original,
        risk_test_matrix_claims=claims,
        risk_test_matrix_evidence=None,
        risk_test_matrix_diagnostics=(),
    )


_NON_ACTIONABLE_RISK_DIAGNOSTICS = frozenset({
    "missing-claim",
    "unsuperseded-journal-failure",
    UNAPPROVED_ROW_CLAIM_DIAGNOSTIC,
    DEGRADED_ROW_CLAIM_DIAGNOSTIC,
})


def _derive_authenticated_risk_evidence_for_coder(
    parsed: StructuredIssueImplementation | StructuredCoderFollowup,
    *,
    approved_plan_context: ApprovedPlanContext | None,
    runner: Runner,
    assigned_workdir: Path,
    head_sha: str | None,
    predecessor_head: str | None = None,
    config: AgentLoopConfig | None = None,
    session_id: str | None = None,
    reauthenticate_head: Callable[[], str | None] | None = None,
    invocation_id: str | None = None,
    assigned_worktree_head: str | None = None,
    _closed_execution_catalog: Sequence[object] | None = None,
    _journal_observations: Sequence[object] | None = None,
    _correction_attempted: bool = False,
) -> tuple[StructuredIssueImplementation | StructuredCoderFollowup, DerivedRiskEvidenceResult | None]:
    """Attach canonical evidence only after the PR head is authenticated.

    The broker catalog and the evidence journal have different trust roles.
    The journal can retain bounded history for aggregate failure handling, but
    semantic selectors must resolve only through the closed catalog belonging
    to the response being acquired.  The private correction arguments make
    that boundary stable across the one permitted post-authentication repair
    continuation, whose own broker observations must not silently become
    selectors for the original response.
    """
    if approved_plan_context is None or not approved_plan_context.matrix_available:
        return parsed, None
    matrix_payload = approved_plan_context.risk_test_matrix_payload
    identity = approved_plan_context.risk_test_matrix_identity
    if matrix_payload is None or identity is None:
        return parsed, None
    bound_invocation_id = invocation_id or runner.latest_test_turn_id
    closed_catalog = (
        tuple(_closed_execution_catalog)
        if _closed_execution_catalog is not None
        else _current_test_turn_observations(runner)
    )
    journal_observations = (
        tuple(_journal_observations)
        if _journal_observations is not None
        else tuple(
            observation
            for observation in runner.local_test_observations()
            if bound_invocation_id is None
            or (
                observation.get("turn_id")
                if isinstance(observation, Mapping)
                else getattr(observation, "turn_id", None)
            ) == bound_invocation_id
        )
    )
    closed_catalog = _admissible_evidence_observations(
        closed_catalog, assigned_workdir=assigned_workdir
    )
    journal_observations = _admissible_evidence_observations(
        journal_observations, assigned_workdir=assigned_workdir, selectable=False
    )
    try:
        snapshot = stable_tracked_tree_snapshot(assigned_workdir)
    except Exception:
        snapshot = None
    reconciled = reconcile_test_observations(
        journal_observations,
        current_head=head_sha,
        current_snapshot=snapshot,
        cwd=assigned_workdir,
    )
    reconciled_catalog = reconcile_test_observations(
        closed_catalog,
        current_head=head_sha,
        current_snapshot=snapshot,
        cwd=assigned_workdir,
    )
    # Authentication is conjunctive: the remote PR metadata and the assigned
    # checkout must identify the same clean, stable tree.  In particular, a
    # checkout that is merely newer than the pre-turn snapshot is not enough.
    authenticated_checkout_head = snapshot.head if snapshot is not None else None
    authenticated_tree_clean = bool(
        snapshot is not None
        and snapshot.complete
        and snapshot.stable is True
        and snapshot.status_clean is True
        and authenticated_checkout_head is not None
        and authenticated_checkout_head == head_sha
        and (
            assigned_worktree_head is None
            or assigned_worktree_head == authenticated_checkout_head
        )
    )
    result = derive_risk_test_matrix_evidence(
        matrix=matrix_payload,
        claims=parsed.risk_test_matrix_claims,
        observations=reconciled.observations,
        execution_catalog=reconciled_catalog.observations,
        invocation_id=bound_invocation_id,
        current_head=head_sha,
        current_tree_digest=(snapshot.tracked_digest if snapshot is not None else None),
        authenticated_checkout_head=authenticated_checkout_head,
        authenticated_tree_clean=authenticated_tree_clean,
        predecessor_head=predecessor_head,
        expected_identity=identity,
        execution_owner=approved_plan_context.risk_test_matrix_execution_owner,
    )
    # A dropped unapproved-row claim (#920) is an audit record, not something
    # a correction may relabel onto another row, so it never triggers or
    # fails the bounded correction.
    actionable = tuple(
        diagnostic for diagnostic in result.diagnostics
        if diagnostic.code not in _NON_ACTIONABLE_RISK_DIAGNOSTICS
    )
    if (
        actionable
        and not _correction_attempted
        and config is not None
        and session_id
        and reauthenticate_head is not None
    ):
        # Keep the original response's closed catalog even though the
        # correction invocation itself gets a fresh broker turn.  A repair
        # response may correct selectors, but it cannot mint new execution
        # authority or turn an unrelated correction observation into evidence.
        catalog = closed_catalog
        prompt = _post_auth_correction_prompt(
            parsed,
            matrix_row_ids=approved_plan_context.risk_test_matrix_expected_row_ids,
            diagnostics=actionable,
            execution_catalog=_safe_execution_handle_catalog(catalog),
        )
        try:
            correction_result = run_agent_result(
                runner,
                agent=config.coder,
                config=config,
                prompt=prompt,
                session_id=session_id,
                role="coder",
                label="semantic-evidence-correction",
                timeout_seconds=config.coder_test_command_timeout_seconds,
            )
            corrected = _parse_fresh_correction_claims(
                correction_result.text,
                parsed,
                row_ids=approved_plan_context.risk_test_matrix_expected_row_ids,
                execution_catalog=catalog,
            )
        except CheckoutVerificationError:
            # A corrupted assigned checkout is a fail-closed run failure, never
            # a recoverable "correction unavailable" (#1130).
            raise
        except Exception as exc:
            corrected = None
            correction_error = f"semantic correction unavailable: {type(exc).__name__}"
        else:
            correction_error = None
        if corrected is not None:
            corrected_head = reauthenticate_head()
            if corrected_head == head_sha:
                corrected_parsed, corrected_result = _derive_authenticated_risk_evidence_for_coder(
                    corrected,
                    approved_plan_context=approved_plan_context,
                    runner=runner,
                    assigned_workdir=assigned_workdir,
                    head_sha=head_sha,
                    predecessor_head=predecessor_head,
                    invocation_id=bound_invocation_id,
                    _closed_execution_catalog=closed_catalog,
                    _journal_observations=journal_observations,
                    _correction_attempted=True,
                )
                if corrected_result is not None and any(
                    diagnostic.code not in _NON_ACTIONABLE_RISK_DIAGNOSTICS
                    for diagnostic in corrected_result.diagnostics
                ):
                    exhausted = PostAuthClaimDiagnostic(
                        actionable[0].row_id if actionable else "risk-test-matrix",
                        "semantic-correction-exhausted",
                        "The bounded semantic correction did not produce fully admissible post-authentication claims.",
                    )
                    corrected_result = dataclasses_replace(
                        corrected_result,
                        diagnostics=tuple((*corrected_result.diagnostics, exhausted)),
                    )
                    corrected_parsed = dataclasses_replace(
                        corrected_parsed,
                        risk_test_matrix_evidence=corrected_result.evidence,
                        risk_test_matrix_diagnostics=corrected_result.diagnostics,
                    )
                return corrected_parsed, corrected_result
            if corrected_head is not None:
                raced_parsed, raced_result = _derive_authenticated_risk_evidence_for_coder(
                    corrected,
                    approved_plan_context=approved_plan_context,
                    runner=runner,
                    assigned_workdir=assigned_workdir,
                    head_sha=corrected_head,
                    predecessor_head=predecessor_head,
                    invocation_id=bound_invocation_id,
                    _closed_execution_catalog=closed_catalog,
                    _journal_observations=journal_observations,
                    _correction_attempted=True,
                )
                if raced_result is not None:
                    # A remote-head race terminates correction.  Even if the
                    # assigned checkout happened to move with the remote
                    # between the two reads, claims acquired before the race
                    # cannot be promoted to verified evidence for that new
                    # head in this handoff.
                    raced_rows = tuple(
                        dataclasses_replace(
                            row,
                            status=("stale/unverified" if row.status == "verified" else row.status),
                            evidence_citations=(),
                            caveats=tuple((*row.caveats, "head changed during bounded semantic correction")),
                        )
                        for row in raced_result.evidence.rows
                    )
                    raced_evidence = dataclasses_replace(
                        raced_result.evidence,
                        rows=raced_rows,
                    )
                    raced_result = dataclasses_replace(
                        raced_result,
                        evidence=raced_evidence,
                        diagnostics=tuple((*raced_result.diagnostics, PostAuthClaimDiagnostic(
                            actionable[0].row_id if actionable else "risk-test-matrix",
                            "head-changed-during-correction",
                            "The authenticated PR head changed during semantic correction; evidence was rebuilt for the new head.",
                        ))),
                    )
                    raced_parsed = dataclasses_replace(
                        raced_parsed,
                        risk_test_matrix_evidence=raced_result.evidence,
                        risk_test_matrix_diagnostics=raced_result.diagnostics,
                    )
                return raced_parsed, raced_result
            correction_error = "semantic correction was discarded because the authenticated PR head changed"
        diagnostic = PostAuthClaimDiagnostic(
            actionable[0].row_id if actionable else "risk-test-matrix",
            "semantic-correction-exhausted",
            correction_error or "The bounded semantic correction did not produce a valid claim set.",
        )
        result = dataclasses_replace(
            result,
            diagnostics=tuple((*result.diagnostics, diagnostic)),
        )
    return dataclasses_replace(
        parsed,
        risk_test_matrix_evidence=result.evidence,
        risk_test_matrix_diagnostics=result.diagnostics,
    ), result


def _post_no_pr_implementation_terminal_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    coder_response: ValidatedAgentResponse,
) -> None:
    """Post a genuine coder-declared no-PR blocking/clarify result to GitHub (#588).

    Matches how a PR-success implementation result is already posted
    (post_pr_comment / post_issue_pr_handoff_comment): the coder's own text
    is the actionable public record, whether or not a PR was created.
    """
    post_issue_comment(
        runner,
        config=config,
        issue_number=issue_number,
        body=normalize_freeform_signature(
            coder_response.text,
            agent=config.coder,
            config=config,
            model_used=coder_response.model_used,
        ),
    )


def _post_structured_issue_implementation_terminal_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    parsed: StructuredIssueImplementation,
    model_used: str | None,
) -> None:
    """Publish a typed no-PR or rejected-conflict implementation result."""
    post_issue_comment(
        runner,
        config=config,
        issue_number=issue_number,
        body=render_public_agent_comment(
            kind="issue_implementation",
            parsed=parsed,
            agent=config.coder,
            config=config,
            model_used=model_used,
        ),
    )


def _require_pr_number(text: str) -> int:
    pr_number = parse_pr_number(text)
    if pr_number is None:
        raise AgentLoopError("Agent response did not include a valid positive PR marker or PR URL.")
    return pr_number


def _require_pr_number_or_clarification(text: str) -> int | str:
    pr_number = parse_pr_number(text)
    if pr_number is not None:
        return pr_number
    if is_clarification_request(text):
        return "clarification"
    raise AgentLoopError(
        "Agent response did not include a PR marker, PR URL, or clarification marker."
    )


def _require_task_implementation_result(
    text: str,
    *,
    required_architecture_impact_contract: int = 0,
    architecture_status_mode: str,
) -> int | str | StructuredTaskResult | _TerminalNoPrImplementation:
    """Validate the fresh structured task envelope, with legacy recovery opt-in."""
    structured = validate_structured_task_result(
        text,
        required_architecture_impact_contract=required_architecture_impact_contract,
        architecture_status_mode=architecture_status_mode,
    )
    if structured is not None:
        if structured.outcome == "opened_pr":
            return structured
        if structured.outcome == "clarification":
            return structured
        if architecture_impact_contract_unsatisfied(structured):
            # A blocking wrapper must never hide an unsatisfied contract;
            # return the parsed result so the seam refuses it.
            return structured
        # Keep the satisfied payload inside the terminal wrapper so its
        # assessment and degradation records still reach round metadata.
        return _TerminalNoPrImplementation("blocking", parsed=structured)
    if required_architecture_impact_contract == 1:
        raise AgentLoopError(
            "Fresh task implementation responses must use the structured "
            "task_result contract with architecture_impact."
        )
    pr_number = parse_pr_number(text)
    if pr_number is not None:
        return pr_number
    if is_clarification_request(text):
        return "clarification"
    try:
        state = parse_agent_state(text)
    except AgentLoopError:
        state = None
    if state == "blocking":
        return _TerminalNoPrImplementation("blocking")
    raise AgentLoopError(
        "Agent response did not include a PR marker, PR URL, clarification marker, "
        "or a terminal blocking marker."
    )


def _require_plan_state_or_clarification(
    text: str, *, required_architecture_impact_contract: int = 0,
    require_execution_strategy_contract: int = 0,
    require_risk_test_matrix_contract: int = 0,
    architecture_status_mode: str,
) -> StructuredPlanState | str:
    if is_clarification_request(text):
        return "clarification"
    structured_plan = validate_structured_plan_state(
        text,
        required_architecture_impact_contract=required_architecture_impact_contract,
        require_execution_strategy_contract=require_execution_strategy_contract,
        require_risk_test_matrix_contract=require_risk_test_matrix_contract,
        # Fresh planner turns must declare every child's execution
        # disposition (#808); recovery parsing elsewhere tolerates absence.
        require_child_dispositions=require_execution_strategy_contract == 1,
        architecture_status_mode=architecture_status_mode,
    )
    if structured_plan is None:
        raise AgentLoopError(
            "Initial planning response must include a structured `plan_state` JSON object."
        )
    return structured_plan
def _validate_response_with_human_requirements(
    text: str,
    *,
    marker_validator: Callable[[str], object],
    human_requirements,
    requirement_scope: str,
    full_omission_fallback: str,
) -> object:
    marker_value = marker_validator(text)
    prompt_context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope=requirement_scope,
        full_omission_fallback=full_omission_fallback,
    )
    validate_human_requirements_acknowledgement(
        text,
        surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
        requires_direct_discussion_ack=prompt_context.requires_direct_discussion_ack,
    )
    if hasattr(marker_value, "human_requirement_dispositions"):
        validate_human_requirement_dispositions(
            marker_value.human_requirement_dispositions,
            surfaced_requirement_ids=prompt_context.surfaced_requirement_ids,
            context=f"{getattr(marker_value, 'kind', 'structured')}.human_requirement_dispositions",
        )
    return marker_value


def _current_plan_has_complete_human_requirement_dispositions(
    coder_output: str | None,
    *,
    surfaced_requirement_ids: Sequence[str],
    inherited_dispositions: Sequence[object] | None = None,
) -> bool:
    """Return whether the current coder plan has the required attestation.

    The raw structured coder response is retained in round metadata specifically
    so resume can apply this same gate to a canonical markdown revision.
    """
    if coder_output is None:
        return False
    try:
        try:
            parsed = validate_structured_plan_state(coder_output, architecture_status_mode="legacy")
        except AgentLoopError:
            try:
                parsed = validate_structured_plan_revision(coder_output, architecture_status_mode="legacy")
            except AgentLoopError:
                patch = validate_structured_plan_revision_patch(coder_output)
                if patch is None:
                    return False
                dispositions = _effective_plan_revision_patch_dispositions(
                    patch,
                    inherited_dispositions,
                )
                if dispositions is None:
                    return False
                validate_human_requirement_dispositions(
                    dispositions,
                    surfaced_requirement_ids=surfaced_requirement_ids,
                    context="plan_revision_patch.human_requirement_dispositions",
                )
                return True
        if parsed is None:
            return False
        validate_human_requirement_dispositions(
            parsed.human_requirement_dispositions,
            surfaced_requirement_ids=surfaced_requirement_ids,
            context=f"{parsed.kind}.human_requirement_dispositions",
        )
    except AgentLoopError:
        return False
    return True


def _effective_plan_revision_patch_dispositions(
    patch: PlanRevisionPatch,
    inherited_dispositions: Sequence[object] | None,
) -> object | None:
    """Return the dispositions that semantic assembly will publish.

    Omission means preserve the authenticated base. Treating omission as an
    empty list would reject an unrelated edit and encourage a payload-identical
    no-op replacement of a field the patch does not own.
    """
    for operation in patch.operations:
        if operation.op == "replace" and operation.field == "human_requirement_dispositions":
            return operation.value
    return inherited_dispositions


def _merge_human_requirements(
    issue_context: IssueContext | None,
    pr_context: PullRequestReviewContext,
    *,
    parent_issue_context: IssueContext | None = None,
    primary_issue_context: IssueContext | None = None,
) -> tuple[HumanReviewRequirement, ...]:
    """Merge only signed requirements from explicitly labeled sources."""
    target = primary_issue_context or issue_context
    combined: list[HumanReviewRequirement] = []
    if parent_issue_context is not None:
        combined.extend(parent_issue_context.human_requirements)
    if target is not None:
        combined.extend(target.human_requirements)
    combined.extend(pr_context.human_requirements)
    return deduplicate_human_requirements(combined)


def _build_requirements_context(
    *,
    target_issue_context: IssueContext | None,
    pr_context: PullRequestReviewContext,
    parent_issue_context: IssueContext | None = None,
) -> RequirementsContext:
    effective = _merge_human_requirements(
        target_issue_context,
        pr_context,
        parent_issue_context=parent_issue_context,
    )
    return RequirementsContext(
        target_child=target_issue_context,
        primary_issue=target_issue_context,
        authoritative_parent=parent_issue_context,
        pr_sources=tuple(pr_context.human_requirements),
        effective_requirements=tuple(effective),
    )


def _describe_requirement_set_change(
    previous_ids: set[str], refreshed_ids: set[str]
) -> str:
    """Name the added and withdrawn signed requirement IDs for a stop message."""
    added = sorted(refreshed_ids - previous_ids)
    withdrawn = sorted(previous_ids - refreshed_ids)
    parts: list[str] = []
    if added:
        parts.append(f"added {', '.join(added)}")
    if withdrawn:
        parts.append(f"withdrawn {', '.join(withdrawn)}")
    return "; ".join(parts) if parts else "the surfaced requirement set changed"


def _surfaced_reviewer_requirement_ids(
    human_requirements: Sequence,
    *,
    requirement_scope: str,
) -> tuple[str, ...]:
    return render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope=requirement_scope,
    ).surfaced_requirement_ids


def _reviewer_requirement_identity_ids(
    human_requirements: Sequence[HumanReviewRequirement],
) -> tuple[str, ...]:
    """Return the stable IDs used by prompts, responses, and coverage metadata."""
    return tuple(requirement.requirement_id for requirement in human_requirements)


def _reviewer_requirement_coverage_matches(
    human_requirements: Sequence[HumanReviewRequirement],
    persisted_ids: Sequence[str],
) -> bool:
    """Return whether persisted reviewer coverage applies to these requirements.

    Metadata written before digest-backed identities were introduced contains
    positional labels. It is intentionally treated as legacy and never carries
    approval across a non-empty signed-requirement set; the reviewer must run
    again once current identities are available.
    """
    expected_ids = set(_reviewer_requirement_identity_ids(human_requirements))
    if not expected_ids:
        return True
    persisted = {str(item) for item in persisted_ids}
    if any(not re.fullmatch(r"hr-[0-9a-f]{64}", item) for item in persisted):
        return False
    return expected_ids.issubset(persisted)


def _resumed_pr_reviewer_matches_requirements(
    record: PostedRoundRecord,
    human_requirements: Sequence[HumanReviewRequirement],
    approved_plan_context: ApprovedPlanContext | None = None,
    reviewer_acquisition_contract: Mapping[str, tuple[object, ...]] | None = None,
) -> bool:
    """Return whether a same-head resumed review still covers current requirements.

    A reviewer comment can be posted before the process is interrupted and then
    resumed after signed requirements are added or edited.  Apply the same
    explicit legacy policy as carried approvals: empty requirements remain
    resumable, while a non-empty current set requires both the resolution marker
    and digest-backed identities from the persisted metadata.
    """
    plan_matches = approved_plan_context is None or (
        record.metadata.approved_plan_hash == approved_plan_context.plan_hash
        and record.metadata.approved_plan_subject == approved_plan_context.plan_subject
    )
    acquisition_matches = True
    if reviewer_acquisition_contract is not None:
        expected = reviewer_acquisition_contract.get(record.metadata.agent)
        acquisition_matches = (
            expected is not None
            and len(expected) >= 2
            and record.metadata.acquisition_outcome == "success"
            and record.metadata.configured_model == expected[0]
            and record.metadata.configured_effort == expected[1]
            and (len(expected) < 3 or record.metadata.provider == expected[2])
        )
    return plan_matches and acquisition_matches and (
        not human_requirements
        or (
            human_requirements_resolved(record.body)
            and _reviewer_requirement_coverage_matches(
                human_requirements,
                record.metadata.surfaced_reviewer_requirement_ids,
            )
        )
    )


def _validate_plan_revision_response(
    text: str,
    *,
    unresolved_items: Sequence[UnresolvedReviewItem] = (),
    require_architecture_impact: bool = False,
    require_execution_strategy_contract: bool = False,
    require_risk_test_matrix_contract: bool = False,
    reject_unsolicited_risk_test_matrix_contract: bool = False,
    architecture_status_mode: str,
) -> StructuredPlanRevision | str:
    parsed = validate_structured_plan_revision(
        text,
        architecture_status_mode=architecture_status_mode,
        required_architecture_impact_contract=(1 if require_architecture_impact else 0),
        require_execution_strategy_contract=(1 if require_execution_strategy_contract else 0),
        require_risk_test_matrix_contract=(1 if require_risk_test_matrix_contract else 0),
        reject_unsolicited_risk_test_matrix_contract=reject_unsolicited_risk_test_matrix_contract,
        require_child_dispositions=bool(require_execution_strategy_contract),
    )
    if parsed is not None:
        allowed_ids = {item.item_id for item in unresolved_items}
        unknown = {
            disposition.item_id
            for disposition in parsed.prior_plan_item_dispositions
        } - allowed_ids
        if unknown:
            raise UnknownPriorItemDispositionError(
                unknown_ids=tuple(sorted(unknown)),
                allowed_ids=tuple(sorted(allowed_ids)),
                same_round_description=(
                    "Same-round findings are informational only and must not be "
                    "dispositioned as prior carried items."
                ),
            )
        return parsed
    raise AgentLoopError("Plan revision did not use the required structured format.")


def _validate_plan_revision_patch_response(
    text: str,
    *,
    unresolved_items: Sequence[UnresolvedReviewItem] = (),
    human_requirements=(),
    inherited_human_requirement_dispositions: Sequence[object] | None = None,
) -> object:
    """Validate a semantic patch and its carried review-item ledger."""
    parsed = validate_structured_plan_revision_patch(text)
    if parsed is None:
        raise AgentLoopError("Semantic plan revision did not use the required structured patch format.")
    _check_plan_revision_patch_ledger(parsed, unresolved_items=unresolved_items)
    requirements_context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    validate_human_requirements_acknowledgement(
        text,
        surfaced_requirement_ids=requirements_context.surfaced_requirement_ids,
        requires_direct_discussion_ack=requirements_context.requires_direct_discussion_ack,
    )
    _check_plan_revision_patch_human_requirement_dispositions(
        parsed,
        surfaced_requirement_ids=requirements_context.surfaced_requirement_ids,
        inherited_human_requirement_dispositions=inherited_human_requirement_dispositions,
    )
    return parsed


def _validate_plan_revision_patch_payload(
    payload: dict,
    *,
    unresolved_items: Sequence[UnresolvedReviewItem] = (),
    human_requirements=(),
    inherited_human_requirement_dispositions: Sequence[object] | None = None,
) -> object:
    """Run every semantic-patch check that depends only on the JSON payload.

    Repair preservation pins this payload exactly, so any failure here is
    unsatisfiable by an envelope-only repair, whatever envelope error the full
    validator happened to report first (#979).  Every failure is raised as a
    ``SemanticPatchPayloadRejection``.
    """
    if payload.get("kind") != "plan_revision_patch":
        # Repair never runs on a non-patch payload (the integrity gate refuses
        # it), so a missing or wrong kind is a replan diagnostic too.
        raise SemanticPatchPayloadRejection(
            "Structured response kind mismatch: expected `plan_revision_patch`."
        )
    try:
        parsed = parse_plan_revision_patch(payload)
    except AgentLoopError as exc:
        raise SemanticPatchPayloadRejection(str(exc)) from exc
    _check_plan_revision_patch_ledger(parsed, unresolved_items=unresolved_items)
    requirements_context = render_coder_human_requirements_prompt_context(
        human_requirements,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    _check_plan_revision_patch_human_requirement_dispositions(
        parsed,
        surfaced_requirement_ids=requirements_context.surfaced_requirement_ids,
        inherited_human_requirement_dispositions=inherited_human_requirement_dispositions,
    )
    return parsed


def _check_plan_revision_patch_ledger(
    parsed: PlanRevisionPatch,
    *,
    unresolved_items: Sequence[UnresolvedReviewItem],
) -> None:
    allowed_ids = {item.item_id for item in unresolved_items}
    unknown = {item.item_id for item in parsed.prior_plan_item_dispositions} - allowed_ids
    if unknown:
        # Payload-level: repair must preserve the patch payload exactly, so
        # only the deterministic strip or a bounded replan can fix it (#979).
        raise SemanticPatchUnknownPriorItemDispositionError(
            unknown_ids=tuple(sorted(unknown)),
            allowed_ids=tuple(sorted(allowed_ids)),
            same_round_description=(
                "Same-round findings are informational only and must not be dispositioned "
                "as prior carried items."
            ),
        )


def _check_plan_revision_patch_human_requirement_dispositions(
    parsed: PlanRevisionPatch,
    *,
    surfaced_requirement_ids: Sequence[str],
    inherited_human_requirement_dispositions: Sequence[object] | None,
) -> None:
    dispositions = _effective_plan_revision_patch_dispositions(
        parsed,
        inherited_human_requirement_dispositions,
    )
    if dispositions is None:
        dispositions = ()
    try:
        validate_human_requirement_dispositions(
            dispositions,
            surfaced_requirement_ids=surfaced_requirement_ids,
            context="plan_revision_patch.human_requirement_dispositions",
        )
    except SemanticPatchPayloadRejection:
        raise
    except AgentLoopError as exc:
        raise SemanticPatchPayloadRejection(str(exc)) from exc


def _drop_repeated_carried_future_followups(
    followups: ApprovedFollowups,
    *,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
) -> ApprovedFollowups:
    """Keep carried future work in its original ledger item.

    A reviewer records a carried item's status through `prior_item_dispositions`.
    Repeating that same concern in `future_followups` used to allocate another
    item ID, even though final follow-up publishing later grouped the two.
    """
    future_ids = {
        disposition.item_id
        for disposition in dispositions
        if disposition.disposition == "future"
    }
    carried = [
        _approved_followup_from_unresolved_item(item)
        for item in prior_items
        if item.item_id in future_ids
    ]
    if not carried or not followups.future:
        return followups

    carried_group_count = len(_dedupe_approved_followups(carried))
    retained = tuple(
        followup
        for followup in followups.future
        if len(_dedupe_approved_followups([*carried, followup])) > carried_group_count
    )
    if retained == followups.future:
        return followups
    return ApprovedFollowups(same_pr=followups.same_pr, future=retained)


def _drop_repeated_carried_plan_future_followups(
    items: PlanReviewItems,
    *,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
) -> PlanReviewItems:
    """Keep repeated carried plan follow-ups in their original ledger items."""
    followups = _drop_repeated_carried_future_followups(
        ApprovedFollowups(same_pr=items.same_plan, future=items.future),
        prior_items=prior_items,
        dispositions=dispositions,
    )
    if followups.future == items.future:
        return items
    return PlanReviewItems(
        blocking=items.blocking,
        same_plan=items.same_plan,
        future=followups.future,
    )


def _validate_tests_with_post_pr_context(
    validate_tests: Callable[[], None],
    *,
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
    report_description: str,
) -> None:
    try:
        validate_tests()
    except AgentLoopError as exc:
        try:
            validate_open_pr(runner, config=config, pr_number=pr_number)
        except Exception as pr_exc:
            raise AgentLoopError(
                f"{exc}\n\n"
                f"The coder reported PR #{pr_number}, but the orchestrator could not confirm it is open. "
                f"The handoff/reviewer comments were not posted because the {report_description} was invalid. "
                "Inspect the PR state on GitHub before deciding whether to resume the existing PR or rerun "
                "implementation."
            ) from pr_exc
        raise AgentLoopError(
            f"{exc}\n\n"
            f"PR #{pr_number} was confirmed open, but the handoff/reviewer comments were not posted because "
            f"the {report_description} was invalid. The managed-CI authorization checkpoint, when required, "
            "was persisted before this report was rejected. Correct the PR/comment if needed, then continue safely with "
            f"`agent-loop pr {pr_number}` instead of rerunning implementation and creating a duplicate PR."
        ) from exc


def _validate_response_tests_with_post_pr_context(
    text: str,
    *,
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
) -> None:
    _validate_tests_with_post_pr_context(
        lambda: validate_response_tests_within_workdir(
            text, assigned_workdir=active_workdir(config)
        ),
        runner=runner,
        config=config,
        pr_number=pr_number,
        report_description="test report",
    )


_StructuredTestReport = TypeVar(
    "_StructuredTestReport", StructuredIssueImplementation, StructuredCoderFollowup
)


def _degrade_out_of_checkout_tests(
    parsed: _StructuredTestReport, *, config: AgentLoopConfig
) -> _StructuredTestReport:
    """Move reported out-of-checkout runs to non-evidence context (#991).

    A baseline run on a clean base-branch copy is honest context, not a reason
    to reject the whole hand-off. It is removed from ``tests_run`` so it never
    becomes a self-reported evidence row, and rendered separately. Live remote
    targets and other unvalidatable reports still raise.
    """
    partition = partition_reported_tests_by_workdir(
        parsed.tests_run, assigned_workdir=active_workdir(config)
    )
    if not partition.out_of_checkout:
        return parsed
    log(
        config,
        f"Recorded {len(partition.out_of_checkout)} reported test run(s) outside the "
        "assigned checkout as non-evidence context",
    )
    return dataclasses_replace(
        parsed,
        tests_run=partition.in_checkout,
        out_of_checkout_tests_run=partition.out_of_checkout,
    )


def _validate_structured_response_tests_with_post_pr_context(
    parsed: _StructuredTestReport,
    *,
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
) -> _StructuredTestReport:
    """Validate structured test commands with the same confirmed-PR diagnostic."""
    result: list[_StructuredTestReport] = []
    _validate_tests_with_post_pr_context(
        lambda: result.append(_degrade_out_of_checkout_tests(parsed, config=config)),
        runner=runner,
        config=config,
        pr_number=pr_number,
        report_description="structured test report",
    )
    return result[0]


def _validate_structured_response_observations_with_post_pr_context(
    test_observations: Sequence[object] | None,
    *,
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
) -> None:
    """Validate structured receipt citations with the same confirmed-PR diagnostic."""
    _validate_tests_with_post_pr_context(
        lambda: validate_test_observation_citations_within_workdir(
            test_observations,
            assigned_workdir=active_workdir(config),
        ),
        runner=runner,
        config=config,
        pr_number=pr_number,
        report_description="structured test-observation report",
    )
