"""Architecture-impact contract helpers and accepted-candidate carriers.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1191); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace as dataclasses_replace

from .agents.base import AgentResult
from .architecture_context import architecture_material, freeze_architecture_context
from .config import AgentLoopConfig
from .decomposition import PlanDecomposition
from .errors import AgentInvocationError, AgentLoopError
from .github import PullRequestMetadata, post_issue_comment
from .logging import log
from .protocol import (
    EVIDENCE_REASK_MAX_CHARS,
    render_evidence_rejection_detail,
    ParsedPlanReview,
    ParsedReview,
    StructuredCoderFollowup,
    StructuredIssueImplementation,
    StructuredPlanState,
    StructuredPlanRevision,
    StructuredTaskResult,
    sanitize_architecture_impact,
    ARCHITECTURE_IMPACT_DECLARED_STATUSES,
    ARCHITECTURE_IMPACT_UNDETERMINED,
    ArchitectureImpact,
    ParseDegradation,
)
from .repair_preservation import (
    canonicalize_architecture_near_miss_text,
    recover_payload,
)
from .runner import Runner
from .workdirs import active_workdir
from .comment_rendering import (
    render_decomposition_degradation_comment,
    render_refused_decomposition_comment,
)
from .round_state import _extract_round_metadata_records
from .protocol_markers import TrustedBody, sanitize_historical_text
from .agent_failure import ValidatedAgentResponse


# Validation may invoke format repair, whose runner intentionally starts a new
# broker turn. Keep the acquisition snapshot in a context-local seam so the
# existing validators continue to receive the coder turn while repair runs.
_VALIDATION_TEST_TURN_CONTEXT: ContextVar[
    tuple[Runner, str | None, tuple[object, ...]] | None
] = ContextVar("validation_test_turn_context", default=None)


def _freeze_prompt_architecture(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    target_revision: str | None = None,
    candidate_revision: str | None = None,
    pr_pair: bool = False,
) -> AgentLoopConfig:
    """Attach one acquisition-local architecture snapshot to prompt config.

    Architecture acquisition is deliberately capability-based: test doubles
    and wrappers may participate when they implement the normal Runner
    command surface, while unavailable Git objects simply produce no prompt
    material and preserve the legacy prompt path.
    """
    if not config.architecture_context_enabled:
        return config
    checkout = active_workdir(config)
    if not checkout.is_dir():
        return config
    context = freeze_architecture_context(
        runner,
        checkout=checkout,
        repository=config.repo,
        path=config.architecture_path,
        read_size=config.architecture_read_size,
        # Keep the normalized live target ref in the identity.  The pair's
        # merge-base is the immutable commit used for document reads; a moving
        # target tip must not masquerade as a retarget when the ref and
        # merge-base remain unchanged.
        target_revision=target_revision,
        candidate_revision=candidate_revision,
    )
    return dataclasses_replace(config, architecture_context=context)


def _architecture_metadata_fields(
    config: AgentLoopConfig, *, result: object | None = None
) -> dict[str, object]:
    """Persist the complete acquisition identity, including unavailable states.

    Writers pass the whole parsed ``result`` rather than a bare assessment so
    an assessment can never be persisted without its degradation records.
    """
    context = config.architecture_context
    identity = context.identity() if hasattr(context, "identity") else None
    impact, degradations = _architecture_result_fields(result)
    if impact is not None and getattr(impact, "status", None) not in ARCHITECTURE_IMPACT_DECLARED_STATUSES:
        # Durable metadata never holds the parser-only degraded status, which
        # strict rehydration would reject; the record explains the absence.
        impact = None
    return {
        "architecture_identity": identity,
        "architecture_impact": sanitize_architecture_impact(impact),
        "architecture_impact_degradations": degradations,
        # This records the response-contract generation, not document
        # availability. A fresh turn must remain distinguishable from a
        # legacy record even when architecture acquisition is opted out or
        # unavailable.
        "architecture_contract_version": 1,
    }


def _test_observation_degradation_fields(result: object | None) -> dict[str, object]:
    """Persist dropped-citation records beside the coder's raw response (#927).

    Only coder follow-ups and issue implementations carry them; every other
    result contributes nothing, so its metadata encoding is unchanged.
    """
    result = _unwrap_architecture_result(result)
    if isinstance(result, (StructuredCoderFollowup, StructuredIssueImplementation)):
        fields: dict[str, object] = {
            "test_observation_degradations": tuple(result.test_observation_degradations)
        }
        if isinstance(result, StructuredCoderFollowup):
            # Advisory sub-item claims (#958), emitted only when present.
            if result.addressed_sub_items:
                fields["addressed_sub_items"] = tuple(result.addressed_sub_items)
            if result.sub_item_claim_degradations:
                fields["sub_item_claim_degradations"] = tuple(result.sub_item_claim_degradations)
        return fields
    return {}


def _latest_pr_approval_architecture_identity(
    comments: Sequence[object], *, head_sha: str | None = None
) -> dict | None:
    """Read the architecture identity bound to the latest approved review.

    Coder and checkpoint records are observations, not approvals. Resume and
    qualification therefore bind invalidation to an actual approved reviewer
    record instead of whichever metadata comment happened to be posted last.
    """
    try:
        records = _extract_round_metadata_records(comments, flow="pr")
    except AgentLoopError:
        return None
    for record in reversed(records):
        metadata = record.metadata
        if metadata.role != "reviewer" or metadata.state != "approved":
            continue
        if head_sha is not None and metadata.subject != head_sha:
            continue
        if isinstance(metadata.architecture_identity, dict):
            return metadata.architecture_identity
    return None


def _latest_pr_architecture_observation(
    comments: Sequence[object], *, head_sha: str | None = None
) -> dict | None:
    """Return the newest durable PR architecture observation.

    Qualification checkpoints are written before a rerun. Reading the newest
    observation makes the invalidation transition idempotent: once the new
    identity is persisted, an identical gate does not schedule another review.
    """
    try:
        records = _extract_round_metadata_records(comments, flow="pr")
    except AgentLoopError:
        return None
    for record in reversed(records):
        metadata = record.metadata
        if head_sha is not None and metadata.subject != head_sha:
            continue
        if isinstance(metadata.architecture_identity, dict):
            return metadata.architecture_identity
    return None


def _revalidate_pr_architecture_identity(
    runner: Runner, *, config: AgentLoopConfig, metadata: PullRequestMetadata,
    stored_identity: dict | None = None,
) -> tuple[object | None, bool]:
    """Reacquire the live PR pair at every qualification/merge snapshot."""
    stored = config.architecture_context
    if not config.architecture_context_enabled:
        return None, False
    fresh_config = _freeze_prompt_architecture(
        runner,
        config,
        target_revision=metadata.base_branch,
        candidate_revision=metadata.head_sha,
        pr_pair=True,
    )
    fresh = fresh_config.architecture_context
    previous_identity = stored_identity
    if previous_identity is None and hasattr(stored, "identity"):
        previous_identity = stored.identity()
    fresh_identity = fresh.identity() if hasattr(fresh, "identity") else None
    changed = bool(fresh_identity is not None and fresh_identity != previous_identity)
    if changed and isinstance(previous_identity, dict) and isinstance(fresh_identity, dict):
        # An unavailable/missing document has no architecture content whose
        # candidate revision can affect a prompt.  Still observe ref, merge
        # base, and availability transitions, but let the ordinary PR-head
        # scheduler handle a contentless coder push.
        def _availability(identity: dict) -> object:
            if "availability" in identity:
                return identity.get("availability")
            return (
                identity.get("base", {}).get("availability"),
                identity.get("candidate", {}).get("availability"),
            )

        if not architecture_material(stored) and not architecture_material(fresh):
            comparable_previous = dict(previous_identity)
            comparable_fresh = dict(fresh_identity)
            comparable_previous.pop("candidate_revision", None)
            comparable_fresh.pop("candidate_revision", None)
            for key in ("base", "candidate"):
                if isinstance(comparable_previous.get(key), dict):
                    comparable_previous[key] = dict(comparable_previous[key])
                    comparable_previous[key].pop("revision", None)
                if isinstance(comparable_fresh.get(key), dict):
                    comparable_fresh[key] = dict(comparable_fresh[key])
                    comparable_fresh[key].pop("revision", None)
            changed = (
                comparable_previous != comparable_fresh
                or _availability(previous_identity) != _availability(fresh_identity)
            )
    # A changed observation is a normal exact-once scheduling transition. The
    # caller marks the fresh context and persists it with the next checkpoint;
    # it must not abort a review/fix/re-review cycle.
    return fresh, changed


@dataclass(frozen=True)
class _TerminalNoPrImplementation:
    state: str
    # A satisfied blocking task result keeps its structured payload (#925);
    # clarification and legacy markers carry none.
    parsed: StructuredTaskResult | None = None


@dataclass(frozen=True)
class _TerminalIssueImplementationConflict:
    """A valid implementation payload rejected from handoff by semantics."""

    parsed: StructuredIssueImplementation


# Every parsed result type that carries an architecture-impact contract.  The
# unsatisfied check enumerates these explicitly rather than defaulting through
# ``getattr``, so a new result wrapper cannot silently bypass the contract.
_ARCHITECTURE_CONTRACT_CARRIERS = (
    StructuredCoderFollowup,
    StructuredIssueImplementation,
    StructuredTaskResult,
    StructuredPlanRevision,
    StructuredPlanState,
    PlanDecomposition,
)
# Review carriers hold records but never a required contract.
_ARCHITECTURE_RECORD_CARRIERS = (*_ARCHITECTURE_CONTRACT_CARRIERS, ParsedReview, ParsedPlanReview)


_TERMINAL_PAYLOAD_WRAPPERS = (_TerminalNoPrImplementation, _TerminalIssueImplementationConflict)


def _carrier_payload(result: object) -> object | None:
    """Return the structured payload a result carries, looking inside wrappers.

    Both terminal wrappers expose ``parsed``; a carrier is its own payload;
    clarification, legacy PR numbers and payload-less wrappers carry none.
    """
    if isinstance(result, _TERMINAL_PAYLOAD_WRAPPERS):
        return result.parsed
    if isinstance(result, _ARCHITECTURE_RECORD_CARRIERS):
        return result
    return None


def _with_carrier_payload(result: object, payload: object) -> object:
    """Rebuild ``result`` around ``payload`` without ever dropping a wrapper."""
    if isinstance(result, _TERMINAL_PAYLOAD_WRAPPERS):
        return dataclasses_replace(result, parsed=payload)
    return payload


def _is_contract_free_result(result: object) -> bool:
    """Results that carry no structured payload: clarification, legacy PR, no-PR."""
    if isinstance(result, _TerminalNoPrImplementation):
        return result.parsed is None
    return isinstance(result, (str, int))


def _unwrap_architecture_result(result: object) -> object:
    if isinstance(result, _TERMINAL_PAYLOAD_WRAPPERS) and result.parsed is not None:
        return result.parsed
    return result


def _architecture_result_fields(
    result: object | None,
) -> tuple[object | None, tuple[ParseDegradation, ...]]:
    """Return the assessment and its degradation records for a metadata writer.

    The input list is enumerated: ``None``, clarification strings, legacy PR
    numbers and payload-less wrappers yield nothing; wrappers holding a
    payload are unwrapped; carriers yield their own fields; any other type
    raises rather than defaulting.
    """
    result = _unwrap_architecture_result(result)
    if result is None or _is_contract_free_result(result):
        return None, ()
    if isinstance(result, _ARCHITECTURE_RECORD_CARRIERS):
        return result.architecture_impact, tuple(result.architecture_impact_degradations)
    # An assembled semantic-patch plan is a StructuredPlanRevision, already
    # enumerated above; no duck-typed fallback may silently drop records.
    raise AgentLoopError(
        f"Internal error: {type(result).__name__} is not an enumerated architecture-impact "
        "result type for round metadata."
    )


def architecture_impact_contract_unsatisfied(result: object) -> bool:
    """Whether a validated result fails a required architecture-impact contract.

    Fails closed: an unknown result type raises rather than defaulting to
    satisfied.
    """
    result = _unwrap_architecture_result(result)
    if _is_contract_free_result(result):
        return False
    if isinstance(result, _ARCHITECTURE_CONTRACT_CARRIERS):
        contract = result.architecture_impact_contract
        return contract.required and not contract.satisfied
    raise AgentLoopError(
        f"Internal error: {type(result).__name__} is not an enumerated architecture-impact "
        "contract result type."
    )


def _architecture_contract_diagnostic(result: object) -> str | None:
    """Pure field-naming diagnostic for an unsatisfied contract, else None."""
    if not architecture_impact_contract_unsatisfied(result):
        return None
    carrier = _unwrap_architecture_result(result)
    kind = "plan_decomposition" if isinstance(carrier, PlanDecomposition) else carrier.kind
    parts = [
        f"{kind} must include architecture_impact for this fresh contract turn.",
        "architecture_impact.status must be `changed` or `unchanged`; an omitted or "
        "undetermined assessment does not satisfy the contract.",
    ]
    for record in carrier.architecture_impact_degradations:
        parts.append(
            f"Degraded element {record.element_path}: rule {record.rule}; observed "
            f"'{record.observed_preview}'; outcome {record.outcome}."
        )
    return " ".join(parts)


def _attach_architecture_degradations(result: object, records: Sequence[ParseDegradation]) -> object:
    """Merge out-of-band records onto the payload a result carries.

    Existing parser-derived records are kept; exact duplicates are dropped.
    """
    if not records:
        return result
    payload = _carrier_payload(result)
    if payload is None:
        return result
    merged = _merge_degradation_records(records, payload.architecture_impact_degradations)
    return _with_carrier_payload(
        result, dataclasses_replace(payload, architecture_impact_degradations=merged)
    )


def _merge_degradation_records(
    first: Sequence[ParseDegradation], second: Sequence[ParseDegradation]
) -> tuple[ParseDegradation, ...]:
    """Order-preserving union: ``first`` leads, exact duplicates dropped."""
    merged: list[ParseDegradation] = []
    for record in (*first, *second):
        if record not in merged:
            merged.append(record)
    return tuple(merged)


def _architecture_mode_validators(
    factory: Callable[[str], Callable[[str], object]],
) -> dict[str, object]:
    """Build an opt-in invocation's degradable validate and strict re-parse.

    Both callables come from one factory, so they are identical -- same
    helper, wrapper, parser and invocation context -- except for the mode.
    """
    return {
        "validate": factory("degradable"),
        "strict_revalidate": factory("strict"),
        "degrade_architecture_impact": True,
    }


def _canonical_comparison_projection(result: object) -> object:
    """Project a result for the degradable-versus-strict equality check.

    Records are cleared, an `undetermined` assessment maps to absence, and raw
    source-text echo fields are ignored; every other field is compared.
    """
    payload = _carrier_payload(result)
    if payload is None:
        return result
    changes: dict[str, object] = {"architecture_impact_degradations": ()}
    impact = payload.architecture_impact
    if impact is not None and impact.status == ARCHITECTURE_IMPACT_UNDETERMINED:
        changes["architecture_impact"] = None
    if hasattr(payload, "raw_dispositions_text"):
        changes["raw_dispositions_text"] = ""
    projected = dataclasses_replace(payload, **changes)
    return (type(result).__name__, getattr(result, "state", None), projected)


@dataclass(frozen=True)
class _AcceptedCandidate:
    text: str
    marker_value: object


class _AcceptedTextCanonicalizationError(AgentLoopError):
    """An otherwise valid candidate whose accepted text could not be canonicalized.

    Never an unsatisfied architecture contract: it names the canonicalization
    failure and is handled as a not-accepted candidate.
    """


def _with_validation_context(
    fn: Callable[[str], object], *, runner: Runner, acquisition: AgentResult | None
) -> Callable[[str], object]:
    """Wrap a parse in the TrustedBody guard and the acquisition's test turn."""

    def run(text: str) -> object:
        TrustedBody.current_untrusted_visible(text)
        token = None
        if acquisition is not None and acquisition.test_turn_id is not None:
            token = _VALIDATION_TEST_TURN_CONTEXT.set(
                (runner, acquisition.test_turn_id, acquisition.test_turn_observations)
            )
        try:
            return fn(text)
        finally:
            if token is not None:
                _VALIDATION_TEST_TURN_CONTEXT.reset(token)

    return run


def _accept_candidate(
    text: str,
    marker_value: object,
    *,
    strict_revalidate: Callable[[str], object] | None,
    runner: Runner,
    acquisition: AgentResult | None,
) -> _AcceptedCandidate:
    """The single acceptance boundary for a validated candidate (#925).

    A record-free candidate passes through byte-identical.  Otherwise the text
    is rewritten to its wire-valid form, re-parsed strictly by the
    invocation's own validate chain under the candidate's own acquisition,
    checked against the degradable result under the comparison projection,
    and the records are reattached inside any terminal wrapper.
    """
    payload = _carrier_payload(marker_value)
    records = tuple(payload.architecture_impact_degradations) if payload is not None else ()
    if not records:
        return _AcceptedCandidate(text, marker_value)
    if strict_revalidate is None:
        raise _AcceptedTextCanonicalizationError(
            "Accepted-text canonicalization failed: no strict re-parse was supplied."
        )
    rewritten = canonicalize_architecture_near_miss_text(text)
    try:
        reparsed = _with_validation_context(
            strict_revalidate, runner=runner, acquisition=acquisition
        )(rewritten)
    except AgentLoopError as exc:
        raise _AcceptedTextCanonicalizationError(
            f"Accepted-text canonicalization failed: the rewritten text did not re-parse strictly ({exc})."
        ) from exc
    if _canonical_comparison_projection(reparsed) != _canonical_comparison_projection(marker_value):
        raise _AcceptedTextCanonicalizationError(
            "Accepted-text canonicalization failed: the strict re-parse differs from the accepted result."
        )
    reparsed_payload = _carrier_payload(reparsed)
    if reparsed_payload is None:
        raise _AcceptedTextCanonicalizationError(
            "Accepted-text canonicalization failed: the strict re-parse carries no structured payload."
        )
    return _AcceptedCandidate(
        rewritten,
        _with_carrier_payload(
            reparsed,
            dataclasses_replace(reparsed_payload, architecture_impact_degradations=records),
        ),
    )


def _accepted_validated_response(
    candidate: _AcceptedCandidate, **fields: object
) -> ValidatedAgentResponse:
    """The only constructor of an accepted response; text comes from the boundary."""
    return ValidatedAgentResponse(
        text=candidate.text, marker_value=candidate.marker_value, **fields
    )


_ARCHITECTURE_CONTRACT_RETRY_SECTION_CHARS = 1200


def _architecture_contract_retry_prompt(prompt: str, diagnostic: str) -> str:
    """Append one bounded, marker-free section naming the unmet contract."""
    detail = " ".join(sanitize_historical_text(diagnostic).replace("<", "(").replace(">", ")").split())
    if len(detail) > _ARCHITECTURE_CONTRACT_RETRY_SECTION_CHARS:
        detail = detail[:_ARCHITECTURE_CONTRACT_RETRY_SECTION_CHARS] + "..."
    return (
        f"{prompt}\n\n## Previous response not accepted: architecture_impact\n\n"
        "The previous response was not accepted because its required `architecture_impact` "
        "assessment was absent or not determinable. Include `architecture_impact` with "
        "`status` set to exactly `changed` or `unchanged` and a non-empty rationale.\n"
        f"Diagnostic: {detail}\n"
    )


def _fresh_matrix_contract_retry_prompt(prompt: str, diagnostic: str) -> str:
    """Append one bounded, marker-free section quoting the matrix validation error."""
    detail = " ".join(sanitize_historical_text(diagnostic).replace("<", "(").replace(">", ")").split())
    if len(detail) > _ARCHITECTURE_CONTRACT_RETRY_SECTION_CHARS:
        detail = detail[:_ARCHITECTURE_CONTRACT_RETRY_SECTION_CHARS] + "..."
    return (
        f"{prompt}\n\n## Previous response not accepted: risk_test_matrix\n\n"
        "The risk test matrix in the previous response failed strict validation. Re-emit "
        "the complete response with every matrix list field as a JSON array of strings.\n"
        f"Validation error: {detail}\n"
    )


_OMISSION_REASK_MAX_IDS = 50


def _prior_disposition_omission_reask_prompt(prompt: str, missing_ids: Sequence[str]) -> str:
    """Append one bounded, marker-free section naming omitted carried item IDs."""
    ids = [
        " ".join(sanitize_historical_text(item_id).replace("<", "(").replace(">", ")").split())
        for item_id in missing_ids
    ]
    listed = ", ".join(ids[:_OMISSION_REASK_MAX_IDS])
    if len(ids) > _OMISSION_REASK_MAX_IDS:
        listed += ", ..."
    return (
        f"{prompt}\n\n## Previous response not accepted: prior item dispositions\n\n"
        "Your previous review omitted a `prior_item_dispositions` entry for these carried "
        f"item IDs: {listed}. Evaluate each of them against the current PR, then re-emit "
        "your complete review with every carried item dispositioned exactly once.\n"
    )


_JUDGEMENT_REASK_MAX_CHARS = 200


def _missing_judgement_field_reask_prompt(
    prompt: str,
    field_path: str,
    allowed_values: Sequence[str],
    observed_preview: str | None = None,
) -> str:
    """Append one bounded, marker-free section naming a missing judgement field (#1185)."""

    def _clean(text: str) -> str:
        cleaned = " ".join(sanitize_historical_text(text).replace("<", "(").replace(">", ")").split())
        if len(cleaned) > _JUDGEMENT_REASK_MAX_CHARS:
            cleaned = cleaned[: _JUDGEMENT_REASK_MAX_CHARS - 3] + "..."
        return cleaned

    allowed = " | ".join(_clean(value) for value in allowed_values)
    observed = (
        f" Your previous value was: {_clean(observed_preview)}." if observed_preview else ""
    )
    return (
        f"{prompt}\n\n## Previous response not accepted: required judgement field\n\n"
        f"Your previous review did not supply a valid `{_clean(field_path)}` "
        f"(allowed: {allowed}).{observed} Decide the value from your own assessment, "
        "then re-emit your complete review.\n"
    )


_EVIDENCE_REASK_MAX_CHARS = EVIDENCE_REASK_MAX_CHARS


def _evidence_rejection_reask_prompt(prompt: str, detail: str) -> str:
    """Append one bounded, marker-free section quoting a rejected test citation."""
    quoted = render_evidence_rejection_detail(detail)
    return (
        f"{prompt}\n\n## Previous response not accepted: test evidence\n\n"
        f"The test observation you selected is not citable: {quoted}\n\n"
        "Observations from the previous turn are no longer selectable. Rerun the test "
        "command that covers this change in this turn through the test wrapper, then cite "
        "the new verified, passing observation. If you cannot obtain one, return a "
        "blocking result instead.\n"
    )


_COVERAGE_REASK_MAX_ROWS = 24


def _risk_coverage_reask_prompt(prompt: str, assessment: object) -> str:
    """Append one bounded, marker-free section naming the deficient coverage rows (#1290)."""
    levels = {entry.row_id: entry for entry in getattr(assessment, "entries", ())}
    lines = []
    for deficiency in tuple(getattr(assessment, "deficiencies", ()))[:_COVERAGE_REASK_MAX_ROWS * 3]:
        entry = levels.get(deficiency.row_id)
        required = (entry.required_level if entry is not None else None) or "unclassified"
        detail = " ".join(sanitize_historical_text(deficiency.detail).replace("<", "(").replace(">", ")").split())
        lines.append(
            f"- {sanitize_historical_text(deficiency.row_id)}: {deficiency.code} "
            f"(required level: {required}){': ' + detail if detail else ''}"
        )
    return (
        f"{prompt}\n\n## Previous response not ready for review: risk-matrix coverage map\n\n"
        "The orchestrator checked your coverage map against the PR head and found these rows "
        "incomplete:\n" + "\n".join(lines) + "\n\n"
        "You may add and commit tests and push them to the same PR. Rerun them through "
        "`agent-loop run-tests` in this turn (selectors from the previous turn are not "
        "selectable) and resubmit your full structured result for the SAME PR. If a row "
        "genuinely cannot be tested as specified, declare it in `risk_test_matrix_coverage_gaps` "
        "with a reason and the evidence-backed correction. This is the only coverage re-ask.\n"
    )


def _surface_decomposition_degradations(
    runner: Runner, *, config: AgentLoopConfig, issue_number: int, decomposition: object
) -> None:
    """Make an accepted decomposition's degradation records operator-visible.

    PlanDecomposition has no round metadata, and its topology checkpoint must
    stay byte-identical, so records surface through a bounded log line and one
    plain, marker-free parent-issue comment instead.
    """
    records = tuple(getattr(decomposition, "architecture_impact_degradations", ()) or ())
    body = render_decomposition_degradation_comment(records)
    if body is None:
        return
    summary = "; ".join(
        f"{record.element_path} {record.outcome} (observed '{record.observed_preview}')"
        for record in records[:4]
    )
    log(config, f"Plan decomposition for issue #{issue_number} parse degradations: {summary}"[:600])
    post_issue_comment(runner, config=config, issue_number=issue_number, body=body)


def _architecture_impact_from_metadata(value: object) -> ArchitectureImpact | None:
    """Rebuild an accepted assessment from its durable round-metadata shape.

    Metadata stores ``sanitize_architecture_impact`` of the accepted carrier's
    assessment.  It is used only when its keys are exactly the dataclass
    fields and its status is declared; anything else yields None, so an
    assessment is never fabricated.
    """
    if not isinstance(value, Mapping):
        return None
    names = {field.name for field in dataclasses.fields(ArchitectureImpact)}
    if set(value) != names or value.get("status") not in ARCHITECTURE_IMPACT_DECLARED_STATUSES:
        return None
    kwargs: dict[str, object] = {}
    for name in names:
        item = value[name]
        kwargs[name] = tuple(item) if isinstance(item, list) else item
    try:
        return ArchitectureImpact(**kwargs)
    except TypeError:
        return None


def _resumed_review_architecture(
    metadata: object, legacy_reparse: ArchitectureImpact | None
) -> tuple[ArchitectureImpact | None, tuple[ParseDegradation, ...]]:
    """The accepted assessment and records of a resumed review record.

    A record written with the architecture contract takes both from its round
    metadata, whatever its phase, so a rendered-prose post and a publication
    post rebuild the same carrier.  Only a record older than architecture
    metadata falls back to the legacy re-parse of its text, with no records.
    """
    if getattr(metadata, "architecture_contract_version", None) is None:
        return legacy_reparse, ()
    impact = _architecture_impact_from_metadata(getattr(metadata, "architecture_impact", None))
    return impact, tuple(getattr(metadata, "architecture_impact_degradations", ()) or ())


def _acknowledgement_repair_forbids_assessment(review_output: str) -> bool:
    """An acknowledgement repair may not add an assessment the source lacks.

    It exists only to add the acknowledgement, so absence is pinned whether
    the review omitted the assessment or a degraded one was removed.
    """
    payload = recover_payload(review_output)
    return isinstance(payload, dict) and "architecture_impact" not in payload


def _pin_acknowledgement_repair(
    accepted: object | None,
    repaired: object | None,
    *,
    config: AgentLoopConfig,
    reviewer_name: str,
) -> object | None:
    """Keep the accepted assessment and records through an acknowledgement repair.

    A repair whose assessment differs, in its durable shape, from the accepted
    one is a failed repair; otherwise the accepted records are reattached.
    """
    if repaired is None or accepted is None:
        return repaired
    if sanitize_architecture_impact(repaired.architecture_impact) != sanitize_architecture_impact(
        accepted.architecture_impact
    ):
        log(
            config,
            f"{reviewer_name}: acknowledgement repair changed the accepted architecture_impact; "
            "treating the repair as failed",
        )
        return None
    return dataclasses_replace(
        repaired, architecture_impact_degradations=accepted.architecture_impact_degradations
    )


def _surface_refused_decomposition(
    runner: Runner, *, config: AgentLoopConfig, issue_number: int, error: AgentInvocationError
) -> None:
    """Post one degradation comment for a decomposition refused by its contract.

    Runs before the exhaustion error propagates.  Nothing was checkpointed or
    created; a failure to post is logged and never masks the original error.
    """
    preserved = getattr(error, "preserved_unsatisfied_response", None)
    if preserved is None:
        return
    records = tuple(preserved.architecture_impact_degradations)
    summary = "; ".join(
        f"{record.element_path} {record.outcome}" for record in records[:4]
    ) or "architecture_impact omitted"
    log(config, f"Plan decomposition for issue #{issue_number} refused: {summary}"[:600])
    try:
        post_issue_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=render_refused_decomposition_comment(records, diagnostic=preserved.diagnostic),
        )
    except Exception as post_exc:  # noqa: BLE001 - never mask the exhaustion error
        log(config, f"Could not post the decomposition degradation comment: {post_exc}"[:600])


class _ArchitectureImpactContractUnsatisfied(AgentLoopError):
    """Private attempt bookkeeping confined to ``_run_validated_agent``.

    Validators never raise this; the seam raises it so every normalization,
    stripping, recovery and repair branch treats an unsatisfied result as
    not accepted.
    """
