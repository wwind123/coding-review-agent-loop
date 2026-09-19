"""Round metadata persistence and resume helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .agents.base import AgentName
from .agents.registry import agent_display_name, agent_signature
from .errors import AgentLoopError
from .round_transport import (
    MAX_PLAN_VALIDATION_DIAGNOSTIC_CHARS,
    PLAN_VALIDATION_DIAGNOSTIC_MARKER_RE,
    ROUND_RESUME_MARKER_RE,
    decode_mapping,
    encode_mapping,
    hydrate_mapping,
)
from .comment_rendering import (
    EXECUTION_RECOMMENDATION_MARKER_RE,
    decode_execution_recommendation_marker,
    RISK_TEST_MATRIX_MARKER_RE,
    decode_risk_test_matrix_marker,
)
from .local_test_evidence import canonicalize_bounded_evidence
from .protocol_markers import (
    ISSUE_COMMENT_SURFACE,
    TrustedBody,
    sanitize_historical_text,
    scan_reserved_markers,
)
from .workdir_guard import validate_checkout_inspected_evidence
from .protocol import (
    HTML_COMMENT_RE,
    SIGNATURE_RE,
    ParsedDiscussAgenda,
    ParsedDiscussAnswer,
    ParsedDiscussFinalSynthesis,
    ParsedDiscussRoundSynthesis,
    ParsedDiscussResponse,
    ParsedDiscussReview,
    ReviewItemDisposition,
    UnresolvedReviewItem,
    MACHINE_AUTHORITY,
    MACHINE_OBLIGATION_FIELDS,
    MACHINE_LIFECYCLE_STATES,
    MACHINE_OBLIGATION_KINDS,
    UNKNOWN_MACHINE_AUTHORITY,
    failed_discuss_review_placeholder,
    failed_discuss_answer_placeholder,
    parse_structured_discuss_agenda,
    parse_canonical_discuss_final_synthesis,
    parse_canonical_discuss_round_synthesis,
    parse_structured_discuss_review,
    parse_structured_discuss_answer,
    parse_legacy_structured_discuss_answer,
    RiskTestMatrix,
    RiskTestMatrixChange,
    parse_risk_test_matrix,
    parse_risk_test_matrix_changes,
    risk_test_matrix_identity,
    sanitize_risk_test_matrix,
    parse_plan_revision_patch,
)
from .plan_assembly import decode_assembled_plan_sidecar, rendered_plan_identity
from .review_scheduling import ReviewSchedulingContract, SCHEDULER_PHASES
from .unresolved_items import _apply_unresolved_item_dispositions


@dataclass(frozen=True)
class PostedRoundMetadata:
    flow: str
    role: str
    agent: str
    round_number: int
    subject: str
    # For plan coder rounds, bind the published candidate to the exact plan
    # subject that was supplied as its revision input.  ``None`` is the
    # intentional value for a fresh plan.  Older metadata omits this field
    # and therefore cannot supersede a newer authenticated diagnostic for a
    # revision context.
    prior_plan_subject: str | None = None
    prior_items: tuple[UnresolvedReviewItem, ...] = ()
    dispositions: tuple[ReviewItemDisposition, ...] = ()
    new_items: tuple[UnresolvedReviewItem, ...] = ()
    state: str | None = None
    canonical_plan: str | None = None
    raw_structured_coder_response: str | None = None
    # Identity of the approved plan surfaced to a PR reviewer.  These fields
    # are deliberately separate from ``subject`` (the PR head for PR reviews)
    # so same-head approvals cannot cross a plan handoff.
    approved_plan_hash: str | None = None
    approved_plan_subject: str | None = None
    compact_prior_summaries: tuple[str, ...] = ()
    usage: dict | None = None
    model_used: str | None = None
    provider: str | None = None
    configured_model: str | None = None
    configured_effort: str | None = None
    effort_source: str | None = None
    observed_model: str | None = None
    observed_effort: str | None = None
    observation_provenance: str | None = None
    # How the response was acquired.  Legacy metadata represents ordinary
    # zero-exit success.
    acquisition_outcome: str = "success"
    acquisition_returncode: int | None = None
    consensus_kind: str | None = None
    is_final: bool = False
    agenda: tuple[str, ...] = ()
    # Raw structured discuss-agenda response from the optional analyzer (#467).
    # None on final summaries, plain-mode rounds, and legacy comments.
    analyzer_response: str | None = None
    # Raw structured final-only analyzer response (#529).  Kept separate from
    # the prior-round agenda so resumes/audit cannot treat history as current.
    final_analyzer_response: str | None = None
    # Discuss research policy in effect when the comment was posted (#477).
    # None on non-discuss flows and legacy comments.
    research_mode: str | None = None
    # Debaters that failed or timed out in a partial round (#475), as
    # (display name, failure category) pairs on the round summary. Resume
    # treats these debaters' missing comments as accounted for.
    failed_debaters: tuple[tuple[str, str], ...] = ()
    # Merged discuss split proposals recorded on a final summary comment
    # (#476), so a resumed run can materialize them into child issues without
    # having to reconstruct them from debater comment metadata. Empty on
    # non-final, non-split, and legacy comments.
    split_proposals: tuple[str, ...] = ()
    # Missing metadata decodes as triage for legacy transcripts.
    result_mode: str = "triage"
    # Canonical bounded #535 evidence artifact on final summaries.  Keeping it
    # here makes resume idempotent without feeding evidence into analyzer agenda.
    evidence_reconciliation: dict | None = None
    # Requirement IDs shown to a PR reviewer when this comment was posted.
    # Persisting this lets a later round distinguish an approval that covered
    # the current signed requirements from one made before new requirements
    # were surfaced for the same immutable PR head.
    surfaced_reviewer_requirement_ids: tuple[str, ...] = ()
    # Parallel reviewers publish a durable provisional checkpoint before the
    # configured-order settlement barrier.  Legacy records are authoritative.
    phase: str = "authoritative"
    canonical_reviewer_response: str | None = None
    # Answer-mode synthesis snapshots are canonical bounded JSON. These keys
    # are omitted from serialized triage/review metadata for byte stability.
    round_synthesis: str | None = None
    final_synthesis: str | None = None
    raw_synthesis_response: str | None = None
    synthesis_provenance: dict | None = None
    # Canonical bounded local-test evidence. Raw environments and identity
    # bytes are never persisted in this field.
    local_test_evidence: str | None = None
    # Orchestrator-derived matrix evidence and bounded post-auth diagnostics.
    # Execution handles are deliberately absent from this durable payload.
    risk_test_matrix_evidence: dict | None = None
    risk_test_matrix_diagnostics: tuple[dict, ...] = ()
    # Optional selective-intermediate scheduler audit fields. They are omitted
    # from legacy encodings unless a scheduler checkpoint actually wrote them.
    scheduler_contract: dict | None = None
    scheduler_previous_sha: str | None = None
    scheduler_current_sha: str | None = None
    scheduler_obligation_digest: str | None = None
    scheduler_selected_reviewers: tuple[str, ...] = ()
    scheduler_paused_reviewers: tuple[tuple[str, str], ...] = ()
    scheduler_reasons: tuple[str, ...] = ()
    scheduler_final_sweep: bool | None = None
    scheduler_force_full: bool | None = None
    scheduler_calls_avoided: int | None = None
    # Phase-aware scheduler authority is optional so pre-staged-policy records
    # remain valid legacy metadata and cannot silently authorize a reduced set.
    scheduler_phase: str | None = None
    scheduler_primary_reviewer: str | None = None
    scheduler_approved_reviewers: tuple[str, ...] = ()
    scheduler_active_owners: tuple[str, ...] = ()
    scheduler_scope_digest: str | None = None
    # This is an in-memory decode-quality signal, deliberately not serialized.
    # ``absent`` is the legacy-compatible state; ``invalid`` means scheduler
    # fields were present but could not be reconstructed safely.
    scheduler_metadata_status: str = "absent"
    # Durable checkpoint for a source-specific exact-head qualification.  It
    # is intentionally independent of the scheduler optimization so a restart
    # can attach to an existing attempt without minting another one.
    qualification_checkpoint: "QualificationCheckpoint | None" = None
    architecture_identity: dict | None = None
    architecture_impact: dict | None = None
    architecture_contract_version: int | None = None
    # Planning generation discriminator.  Absent is intentionally legacy
    # undecided; generation 1 is required to resume a fresh recommendation.
    execution_strategy_contract_version: int | None = None
    execution_strategy_identity: dict | None = None
    # Generation-1 risk matrix semantic carrier. These fields are intentionally
    # independent of rendered canonical plan prose and are matrix-channel
    # diagnostics, not reviewer findings.
    risk_test_matrix_contract_version: int | None = None
    risk_test_matrix_payload: dict | None = None
    risk_test_matrix_changes_payload: tuple[dict, ...] = ()
    risk_test_matrix_identity: str | None = None
    risk_test_matrix_boundary_digest: str | None = None
    risk_test_matrix_diagnostic: str | None = None
    # Semantic planning provenance is prospective and optional so historical
    # round records retain their exact legacy encoding.  When populated, the
    # canonical sidecar is the authenticated full-state source for restart;
    # raw patches remain provenance and are never treated as canonical state.
    response_form: str | None = None
    base_round_number: int | None = None
    base_state_identity: str | None = None
    aggregate_plan_identity: str | None = None
    raw_patch_provenance: dict | None = None
    assembled_plan_sidecar: dict | None = None

    def __post_init__(self) -> None:
        if self.scheduler_metadata_status not in {"absent", "valid", "invalid"}:
            raise ValueError("invalid scheduler metadata status")
        if self.execution_strategy_contract_version not in (None, 1):
            raise ValueError("invalid execution strategy contract version")
        if self.response_form is not None and (
            not isinstance(self.response_form, str)
            or self.response_form not in {
            "semantic-patch-v1", "legacy-full-state", "fresh-plan-state"
            }
        ):
            raise ValueError("invalid planning response form")
        if self.base_round_number is not None and (
            isinstance(self.base_round_number, bool)
            or not isinstance(self.base_round_number, int)
            or self.base_round_number < 0
        ):
            raise ValueError("invalid semantic base round number")
        for identity_name, identity in (
            ("base_state_identity", self.base_state_identity),
            ("aggregate_plan_identity", self.aggregate_plan_identity),
        ):
            if identity is not None and (
                not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity)
            ):
                raise ValueError(f"invalid {identity_name}")
        if self.raw_patch_provenance is not None and not isinstance(self.raw_patch_provenance, dict):
            raise ValueError("invalid raw semantic patch provenance")
        if self.assembled_plan_sidecar is not None and not isinstance(self.assembled_plan_sidecar, dict):
            raise ValueError("invalid assembled plan sidecar")
        if self.assembled_plan_sidecar is not None:
            from .plan_assembly import decode_assembled_plan_sidecar

            try:
                sidecar = decode_assembled_plan_sidecar(self.assembled_plan_sidecar)
            except AgentLoopError as exc:
                raise ValueError("invalid assembled plan sidecar") from exc
            if self.response_form is not None and sidecar.response_form != self.response_form:
                raise ValueError("semantic response form does not match assembled sidecar")
            if (
                self.aggregate_plan_identity is not None
                and sidecar.aggregate_identity != self.aggregate_plan_identity
            ):
                raise ValueError("semantic aggregate identity does not match assembled sidecar")
        _validate_semantic_round_metadata(self)
        if (
            self.execution_strategy_identity is not None
            and not isinstance(self.execution_strategy_identity, dict)
        ):
            raise ValueError("invalid execution strategy identity")
        # Programmatically-created scheduler checkpoints (including tests and
        # callers that have not round-tripped through the transport) are valid
        # when they carry scheduler fields. Decoded malformed payloads pass an
        # explicit ``invalid`` status and remain distinguishable.
        if self.scheduler_metadata_status == "absent" and any(
            value not in (None, (), [])
            for value in (
                self.scheduler_contract,
                self.scheduler_previous_sha,
                self.scheduler_current_sha,
                self.scheduler_obligation_digest,
                self.scheduler_selected_reviewers,
                self.scheduler_paused_reviewers,
                self.scheduler_reasons,
                self.scheduler_final_sweep,
                self.scheduler_force_full,
                self.scheduler_calls_avoided,
                self.scheduler_phase,
                self.scheduler_primary_reviewer,
                self.scheduler_approved_reviewers,
                self.scheduler_active_owners,
                self.scheduler_scope_digest,
            )
        ):
            object.__setattr__(self, "scheduler_metadata_status", "valid")

    @property
    def aggregate_identity(self) -> str | None:
        """Compatibility alias for the semantic assembled-plan identity."""
        return self.aggregate_plan_identity

    @property
    def raw_patch(self) -> dict | None:
        """Compatibility alias for raw semantic patch provenance."""
        return self.raw_patch_provenance


_SEMANTIC_METADATA_FIELDS = frozenset(
    {
        "response_form",
        "base_round_number",
        "base_state_identity",
        "aggregate_plan_identity",
        "raw_patch_provenance",
        "assembled_plan_sidecar",
    }
)


def _validate_semantic_round_metadata(metadata: PostedRoundMetadata) -> None:
    """Validate the all-or-nothing semantic authority group.

    Historical metadata has none of these fields and remains compatible. Once
    any field is present, the record must identify one complete response form;
    otherwise a restart could mistake a damaged semantic record for legacy
    absence and silently fall back to weaker provenance.
    """
    values = {
        "response_form": metadata.response_form,
        "base_round_number": metadata.base_round_number,
        "base_state_identity": metadata.base_state_identity,
        "aggregate_plan_identity": metadata.aggregate_plan_identity,
        "raw_patch_provenance": metadata.raw_patch_provenance,
        "assembled_plan_sidecar": metadata.assembled_plan_sidecar,
    }
    if all(value is None for value in values.values()):
        return
    if metadata.response_form is None:
        raise ValueError("semantic planning metadata is missing response form")
    if metadata.response_form == "semantic-patch-v1" and any(
        value is None for value in values.values()
    ):
        raise ValueError("semantic patch metadata is incomplete")
    if metadata.response_form in {"legacy-full-state", "fresh-plan-state"}:
        if metadata.aggregate_plan_identity is None or metadata.assembled_plan_sidecar is None:
            raise ValueError("full-state semantic metadata is incomplete")
        if metadata.base_round_number is not None or metadata.base_state_identity is not None:
            raise ValueError("full-state semantic metadata must not carry a patch base")
        if metadata.raw_patch_provenance is not None:
            raise ValueError("full-state semantic metadata must not carry patch provenance")

    from .plan_assembly import decode_assembled_plan_sidecar

    assert metadata.response_form is not None
    assert metadata.aggregate_plan_identity is not None
    assert metadata.assembled_plan_sidecar is not None
    try:
        sidecar = decode_assembled_plan_sidecar(metadata.assembled_plan_sidecar)
    except AgentLoopError as exc:
        raise ValueError("invalid assembled plan sidecar") from exc
    if sidecar.round_number != metadata.round_number:
        raise ValueError("assembled plan sidecar round does not match metadata round")
    if sidecar.response_form != metadata.response_form:
        raise ValueError("semantic response form does not match assembled sidecar")
    if sidecar.aggregate_identity != metadata.aggregate_plan_identity:
        raise ValueError("semantic aggregate identity does not match assembled sidecar")
    if sidecar.raw_patch != metadata.raw_patch_provenance:
        raise ValueError("raw semantic patch does not match assembled sidecar")

    expected_kind = (
        "plan_state" if metadata.response_form == "fresh-plan-state" else "plan_revision"
    )
    if sidecar.canonical_json.get("kind") != expected_kind:
        raise ValueError(
            f"semantic response form {metadata.response_form} does not match canonical plan kind"
        )

    if metadata.response_form == "semantic-patch-v1":
        assert metadata.raw_patch_provenance is not None
        assert metadata.base_round_number is not None
        assert metadata.base_state_identity is not None
        try:
            patch = parse_plan_revision_patch(metadata.raw_patch_provenance)
        except AgentLoopError as exc:
            raise ValueError("invalid semantic patch provenance") from exc
        if patch.base_round_number != metadata.base_round_number:
            raise ValueError("semantic base round does not match raw patch provenance")
        if patch.base_state_identity != metadata.base_state_identity:
            raise ValueError("semantic base identity does not match raw patch provenance")
        if sidecar.raw_patch is None:
            raise ValueError("semantic patch sidecar is missing raw patch provenance")
    else:
        # Full-state publication seeds an authenticated canonical base but is
        # not a patch against an earlier base.
        if sidecar.raw_patch is not None:
            raise ValueError("full-state semantic metadata must not carry patch provenance")


@dataclass(frozen=True)
class ApprovedPlanContext:
    """Lossless, PR-bound context for the approved implementation plan.

    This is deliberately separate from :class:`IssueContext`: issue comments
    are a bounded discussion history, while this record is an immutable
    contract selected by the handoff's expected plan hash.
    """

    canonical_text: str | None = None
    plan_hash: str | None = None
    plan_subject: str | None = None
    source_locator: str | None = None
    scope: tuple[str, ...] = ()
    deferred_work: tuple[str, ...] = ()
    availability: Literal[
        "available", "unavailable", "mismatched", "omitted", "not-planned"
    ] = "unavailable"
    diagnostic: str | None = None
    # ``mismatched`` is used both when no record has the expected hash and when
    # a record with that hash is internally conflicting (or has the wrong
    # subject).  Parent fallback is safe only in the former case.
    has_matching_candidate: bool = False
    # Optional atomic semantic matrix channel. ``canonical_text`` is allowed
    # to be absent when this channel is complete and verified.
    risk_test_matrix_availability: Literal["available", "unavailable", "omitted", "not-planned"] = "not-planned"
    risk_test_matrix_diagnostic: str | None = None
    risk_test_matrix_contract_version: int | None = None
    risk_test_matrix_identity: str | None = None
    risk_test_matrix_payload: dict | None = None
    risk_test_matrix_changes_payload: tuple[dict, ...] = ()
    risk_test_matrix_source_locator: str | None = None
    risk_test_matrix_boundary_digest: str | None = None
    # Staged children receive the complete authenticated semantics for audit,
    # but only these owner-matching rows are enforceable in their turn.
    risk_test_matrix_enforceable_row_ids: tuple[str, ...] | None = None
    risk_test_matrix_pending_row_ids: tuple[str, ...] = ()
    risk_test_matrix_execution_owner: str | None = None

    @property
    def raw_canonical_text(self) -> str | None:
        return self.canonical_text

    @property
    def raw_text(self) -> str | None:
        return self.canonical_text

    @property
    def subject(self) -> str | None:
        return self.plan_subject

    @property
    def identity(self) -> str | None:
        return self.plan_hash

    @property
    def approved_plan_hash(self) -> str | None:
        return self.plan_hash

    @property
    def availability_state(self) -> str:
        return self.availability

    @property
    def is_available(self) -> bool:
        return self.availability == "available" and bool(self.canonical_text or self.matrix_available)

    @property
    def matrix_available(self) -> bool:
        return self.risk_test_matrix_availability == "available"

    @property
    def risk_test_matrix(self) -> dict | None:
        return self.risk_test_matrix_payload

    @property
    def risk_test_matrix_expected_row_ids(self) -> tuple[str, ...] | None:
        if not self.matrix_available or self.risk_test_matrix_payload is None:
            return None
        if self.risk_test_matrix_enforceable_row_ids is not None:
            return self.risk_test_matrix_enforceable_row_ids
        return tuple(
            str(row["row_id"])
            for row in self.risk_test_matrix_payload.get("rows", [])
            if isinstance(row, dict)
            and row.get("applicability") in {"applicable", "required"}
        )


def scope_approved_plan_matrix(
    context: ApprovedPlanContext,
    *,
    execution_owner: str,
    valid_stage_ids: Sequence[str] = (),
) -> ApprovedPlanContext:
    """Bind a staged implementation to its owned matrix rows.

    The full authenticated payload remains available for read-only pending
    context. Evidence validation uses only the owner-matching applicable IDs.
    """
    if not context.matrix_available or context.risk_test_matrix_payload is None:
        return context
    rows = context.risk_test_matrix_payload.get("rows", [])
    if not isinstance(rows, list):
        return context
    # A not-applicable matrix carries no enforceable owner obligations.  Do
    # not validate a downstream phase identifier in this case: legacy
    # decomposition adapters may expose a positional placeholder even when a
    # fresh recommendation has no matrix rows to assign.
    if not any(
        isinstance(row, dict)
        and row.get("applicability") in {"applicable", "required"}
        for row in rows
    ):
        return context
    if execution_owner not in {"one-shot", "retained-parent", "final-integration"} and valid_stage_ids:
        if execution_owner not in set(valid_stage_ids):
            raise AgentLoopError(
                f"Risk matrix execution owner `{execution_owner}` is not an approved stage ID."
            )
    owned = tuple(
        str(row["row_id"])
        for row in rows
        if isinstance(row, dict)
        and row.get("execution_owner") == execution_owner
        and row.get("applicability") in {"applicable", "required"}
    )
    pending = tuple(
        str(row["row_id"])
        for row in rows
        if isinstance(row, dict)
        and row.get("row_id") not in owned
        and row.get("applicability") in {"applicable", "required"}
    )
    return replace(
        context,
        risk_test_matrix_enforceable_row_ids=owned,
        risk_test_matrix_pending_row_ids=pending,
        risk_test_matrix_execution_owner=execution_owner,
    )


@dataclass(frozen=True)
class RequirementsContext:
    """Labeled issue sources and the effective stable signed requirements."""

    target_child: object | None = None
    primary_issue: object | None = None
    authoritative_parent: object | None = None
    pr_sources: tuple[object, ...] = ()
    effective_requirements: tuple[object, ...] = ()

    @property
    def child_context(self) -> object | None:
        return self.target_child

    @property
    def parent_context(self) -> object | None:
        return self.authoritative_parent

    @property
    def human_requirements(self) -> tuple[object, ...]:
        return self.effective_requirements


@dataclass(frozen=True)
class PostedRoundRecord:
    index: int
    metadata: PostedRoundMetadata
    body: str


@dataclass(frozen=True)
class ResumedRoundSelection:
    anchor_record: PostedRoundRecord
    current_round_records: tuple[PostedRoundRecord, ...]


@dataclass(frozen=True)
class ResumedReviewRound:
    round_number: int
    prior_items: tuple[UnresolvedReviewItem, ...]
    coder_output: str | None
    completed_reviews: tuple[PostedRoundRecord, ...]
    next_unresolved_item_number: int
    ledger_may_be_incomplete: bool = False
    compact_prior_summaries: tuple[str, ...] = ()
    unrecorded_head_advance: bool = False
    reconciled: bool = False
    coder_metadata: PostedRoundMetadata | None = None
    local_test_evidence: str | None = None
    qualification_checkpoint: QualificationCheckpoint | None = None
    plan_validation_diagnostic: "PlanValidationDiagnosticTransport | None" = None
    # Parallel publication checkpoints intentionally omit provisional item
    # numbers. A settled reconciliation record carries the authoritative
    # items; preserve them on resume instead of minting duplicate IDs.
    current_round_new_items: tuple[UnresolvedReviewItem, ...] = ()


PLAN_VALIDATION_DIAGNOSTIC_SUFFIX = "[diagnostic truncated]"
_PLAN_VALIDATION_SENSITIVE_RE = re.compile(
    r"(?ix)"
    r"(?:ghp_[A-Za-z0-9_\-]+|github_pat_[A-Za-z0-9_\-]+|sk-[A-Za-z0-9_\-]+|"
    r"xox[baprs]-[A-Za-z0-9-]+|"
    r"(?:authorization|password|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*\S+)"
)


def sanitize_plan_validation_diagnostic(value: object) -> str:
    """Return the bounded, marker-safe diagnostic shared by storage and prompts."""
    text = value if isinstance(value, str) else str(value)
    text = sanitize_historical_text(text)
    text = _PLAN_VALIDATION_SENSITIVE_RE.sub("[redacted]", text)
    text = "".join(
        character
        for character in text
        if character in "\n\r\t" or ord(character) >= 32
    ).strip()
    if not text:
        text = "deterministic plan validation failed"
    if len(text) <= MAX_PLAN_VALIDATION_DIAGNOSTIC_CHARS:
        return text
    available = MAX_PLAN_VALIDATION_DIAGNOSTIC_CHARS - len(PLAN_VALIDATION_DIAGNOSTIC_SUFFIX)
    return text[:available].rstrip() + PLAN_VALIDATION_DIAGNOSTIC_SUFFIX


@dataclass(frozen=True)
class PlanValidationDiagnosticPayload:
    """Immutable values known before the diagnostic comment is posted."""

    repository: str
    issue_number: int
    planning_generation: int
    target_coder_round: int
    prior_plan_subject: str | None
    candidate_kind: Literal["plan_state", "plan_revision"]
    architecture_contract_version: int | None
    execution_strategy_contract_version: int | None
    risk_test_matrix_contract_version: int | None
    expected_producer_login: str
    expected_producer_id: int
    failure_attempt: int
    candidate_digest: str
    category: Literal["deterministic"]
    diagnostic: str

    def __post_init__(self) -> None:
        if not self.repository.strip() or self.issue_number < 1:
            raise ValueError("diagnostic payload requires a repository and issue")
        if self.planning_generation < 1 or self.target_coder_round < 1:
            raise ValueError("diagnostic payload requires positive planning coordinates")
        if self.prior_plan_subject is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.prior_plan_subject
        ):
            raise ValueError("invalid prior plan subject")
        if self.candidate_kind not in {"plan_state", "plan_revision"}:
            raise ValueError("invalid planning candidate kind")
        for version in (
            self.architecture_contract_version,
            self.execution_strategy_contract_version,
            self.risk_test_matrix_contract_version,
        ):
            if version is not None and version != 1:
                raise ValueError("unsupported planning contract version")
        if not self.expected_producer_login.strip() or self.expected_producer_id < 1:
            raise ValueError("diagnostic payload requires an authenticated producer")
        if self.failure_attempt < 1:
            raise ValueError("diagnostic payload requires a positive attempt")
        if not re.fullmatch(r"[0-9a-f]{64}", self.candidate_digest):
            raise ValueError("invalid candidate digest")
        if self.category != "deterministic":
            raise ValueError("only deterministic diagnostics are durable")
        bounded = sanitize_plan_validation_diagnostic(self.diagnostic)
        if bounded != self.diagnostic:
            object.__setattr__(self, "diagnostic", bounded)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "plan_validation_diagnostic",
            "repository": self.repository,
            "issue_number": self.issue_number,
            "planning_generation": self.planning_generation,
            "target_coder_round": self.target_coder_round,
            "prior_plan_subject": self.prior_plan_subject,
            "candidate_kind": self.candidate_kind,
            "architecture_contract_version": self.architecture_contract_version,
            "execution_strategy_contract_version": self.execution_strategy_contract_version,
            "risk_test_matrix_contract_version": self.risk_test_matrix_contract_version,
            "expected_producer_login": self.expected_producer_login,
            "expected_producer_id": self.expected_producer_id,
            "failure_attempt": self.failure_attempt,
            "candidate_digest": self.candidate_digest,
            "category": self.category,
            "diagnostic": self.diagnostic,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PlanValidationDiagnosticPayload":
        expected_keys = {
            "schema_version", "kind", "repository", "issue_number",
            "planning_generation", "target_coder_round", "prior_plan_subject",
            "candidate_kind", "architecture_contract_version",
            "execution_strategy_contract_version", "risk_test_matrix_contract_version",
            "expected_producer_login", "expected_producer_id", "failure_attempt",
            "candidate_digest", "category", "diagnostic",
        }
        if (
            set(value) != expected_keys
            or not isinstance(value.get("schema_version"), int)
            or isinstance(value.get("schema_version"), bool)
            or value.get("schema_version") != 1
            or value.get("kind") != "plan_validation_diagnostic"
        ):
            raise AgentLoopError("Invalid plan-validation diagnostic payload shape.")
        try:
            def strict_int(name: str) -> int:
                item = value[name]
                if not isinstance(item, int) or isinstance(item, bool):
                    raise ValueError(f"{name} must be an integer")
                return item

            def optional_int(name: str) -> int | None:
                item = value[name]
                if item is None:
                    return None
                return strict_int(name)

            def strict_string(name: str) -> str:
                item = value[name]
                if not isinstance(item, str):
                    raise ValueError(f"{name} must be a string")
                return item

            diagnostic = strict_string("diagnostic")
            if sanitize_plan_validation_diagnostic(diagnostic) != diagnostic:
                raise ValueError("diagnostic must already be canonical and bounded")

            return cls(
                repository=strict_string("repository"),
                issue_number=strict_int("issue_number"),
                planning_generation=strict_int("planning_generation"),
                target_coder_round=strict_int("target_coder_round"),
                prior_plan_subject=(
                    strict_string("prior_plan_subject")
                    if value["prior_plan_subject"] is not None else None
                ),
                candidate_kind=strict_string("candidate_kind"),  # type: ignore[arg-type]
                architecture_contract_version=optional_int("architecture_contract_version"),
                execution_strategy_contract_version=optional_int("execution_strategy_contract_version"),
                risk_test_matrix_contract_version=optional_int("risk_test_matrix_contract_version"),
                expected_producer_login=strict_string("expected_producer_login"),
                expected_producer_id=strict_int("expected_producer_id"),
                failure_attempt=strict_int("failure_attempt"),
                candidate_digest=strict_string("candidate_digest"),
                category=strict_string("category"),  # type: ignore[arg-type]
                diagnostic=diagnostic,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise AgentLoopError("Invalid plan-validation diagnostic payload.") from exc

    def matches_context(
        self,
        *,
        repository: str,
        issue_number: int,
        planning_generation: int,
        target_coder_round: int,
        prior_plan_subject: str | None,
        candidate_kind: str,
        architecture_contract_version: int | None,
        execution_strategy_contract_version: int | None,
        risk_test_matrix_contract_version: int | None,
    ) -> bool:
        return (
            self.repository == repository
            and self.issue_number == issue_number
            and self.planning_generation == planning_generation
            and self.target_coder_round == target_coder_round
            and self.prior_plan_subject == prior_plan_subject
            and self.candidate_kind == candidate_kind
            and self.architecture_contract_version == architecture_contract_version
            and self.execution_strategy_contract_version == execution_strategy_contract_version
            and self.risk_test_matrix_contract_version == risk_test_matrix_contract_version
        )


@dataclass(frozen=True)
class PlanValidationDiagnosticTransport:
    """Authenticated live transport values wrapped around one immutable payload."""

    payload: PlanValidationDiagnosticPayload
    server_comment_id: int
    authoritative_created_at: str
    exact_live_body: str
    live_producer_login: str
    live_producer_id: int

    @property
    def diagnostic(self) -> str:
        return self.payload.diagnostic

    @property
    def failure_attempt(self) -> int:
        return self.payload.failure_attempt

    @property
    def candidate_digest_prefix(self) -> str:
        return self.payload.candidate_digest[:16]


def encode_plan_validation_diagnostic_body(
    payload: PlanValidationDiagnosticPayload,
) -> TrustedBody:
    """Encode only the immutable pre-POST diagnostic payload."""
    encoded = encode_mapping(payload.as_dict())
    return TrustedBody.canonical(
        f"<!-- AGENT_PLAN_VALIDATION_DIAGNOSTIC: {encoded} -->",
        surface=ISSUE_COMMENT_SURFACE,
        expected_tokens=("AGENT_PLAN_VALIDATION_DIAGNOSTIC",),
    )


def decode_plan_validation_diagnostic_body(
    body: str,
) -> PlanValidationDiagnosticPayload:
    matches = tuple(PLAN_VALIDATION_DIAGNOSTIC_MARKER_RE.finditer(body))
    if len(matches) != 1 or body.strip() != matches[0].group(0):
        raise AgentLoopError("Plan-validation diagnostic is not an exact canonical record.")
    encoded = matches[0].group("payload")
    try:
        mapping = decode_mapping(encoded)
        if encode_mapping(mapping) != encoded:
            raise ValueError("noncanonical mapping")
        TrustedBody.canonical(
            body,
            surface=ISSUE_COMMENT_SURFACE,
            expected_tokens=("AGENT_PLAN_VALIDATION_DIAGNOSTIC",),
        )
        return PlanValidationDiagnosticPayload.from_mapping(mapping)
    except (AgentLoopError, TypeError, ValueError, KeyError) as exc:
        if isinstance(exc, AgentLoopError) and str(exc).startswith("Invalid plan-validation"):
            raise
        raise AgentLoopError("Invalid plan-validation diagnostic record.") from exc


def has_plan_validation_diagnostic_marker(comments: Sequence[object]) -> bool:
    return any(
        isinstance(getattr(comment, "body", None), str)
        and PLAN_VALIDATION_DIAGNOSTIC_MARKER_RE.search(getattr(comment, "body", ""))
        for comment in comments
    )


def _comment_identity(comment: object) -> tuple[int | None, str | None, int | None, str | None]:
    comment_id = getattr(comment, "comment_id", None)
    if comment_id is None:
        comment_id = getattr(comment, "id", None)
    author_login = getattr(comment, "author", None)
    author_id = getattr(comment, "author_id", None)
    created_at = getattr(comment, "created_at", None)
    return comment_id, author_login, author_id, created_at


def _authenticated_canonical_plan_success_exists(
    comments: Sequence[object],
    *,
    expected_author_login: str,
    expected_author_id: int,
    target_coder_round: int,
    prior_plan_subject: str | None,
    candidate_kind: str,
    architecture_contract_version: int | None,
    execution_strategy_contract_version: int | None,
    risk_test_matrix_contract_version: int | None,
) -> bool:
    bodies = tuple(
        body for comment in comments if isinstance((body := getattr(comment, "body", None)), str)
    )
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str) or not ROUND_RESUME_MARKER_RE.search(body):
            continue
        comment_id, author_login, author_id, created_at = _comment_identity(comment)
        if (
            not isinstance(comment_id, int)
            or comment_id < 1
            or not isinstance(created_at, str)
            or not created_at
            or author_login != expected_author_login
            or author_id != expected_author_id
        ):
            continue
        try:
            match = list(ROUND_RESUME_MARKER_RE.finditer(body))[-1]
            payload, missing = hydrate_mapping(decode_mapping(match.group("payload")), bodies)
            if missing:
                continue
            metadata = _decode_round_metadata_mapping(payload)
        except (AgentLoopError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if (
            metadata.flow == "plan"
            and metadata.role == "coder"
            and metadata.round_number == target_coder_round
            and metadata.canonical_plan
            and _plan_subject(metadata.canonical_plan) == metadata.subject
            and metadata.prior_plan_subject == prior_plan_subject
            and metadata.architecture_contract_version == architecture_contract_version
            and metadata.execution_strategy_contract_version == execution_strategy_contract_version
            and metadata.risk_test_matrix_contract_version == risk_test_matrix_contract_version
            and (
                (candidate_kind == "plan_state" and target_coder_round == 1)
                or (candidate_kind == "plan_revision" and target_coder_round > 1)
            )
        ):
            return True
    return False


def recover_plan_validation_diagnostic(
    comments: Sequence[object],
    *,
    repository: str,
    issue_number: int,
    expected_author_login: str,
    expected_author_id: int,
    planning_generation: int,
    target_coder_round: int,
    prior_plan_subject: str | None,
    candidate_kind: str,
    architecture_contract_version: int | None,
    execution_strategy_contract_version: int | None,
    risk_test_matrix_contract_version: int | None,
) -> PlanValidationDiagnosticTransport | None:
    """Recover one authenticated current diagnostic using payload attempt order."""
    if not expected_author_login or expected_author_id < 1:
        raise AgentLoopError("Plan-validation diagnostic recovery requires an authenticated actor.")
    candidates: list[PlanValidationDiagnosticTransport] = []
    by_id: dict[int, str] = {}
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str) or not PLAN_VALIDATION_DIAGNOSTIC_MARKER_RE.search(body):
            continue
        comment_id, author_login, author_id, created_at = _comment_identity(comment)
        # Old or shape-only comments are not trusted recovery records.
        if (
            not isinstance(comment_id, int)
            or comment_id < 1
            or not isinstance(created_at, str)
            or not created_at
            or author_login != expected_author_login
            or author_id != expected_author_id
        ):
            continue
        try:
            payload = decode_plan_validation_diagnostic_body(body)
        except AgentLoopError:
            # Historical comments are untrusted input. A malformed record is
            # ineligible rather than fatal; only authenticated, canonical
            # records may participate in context selection and conflicts.
            continue
        previous_body = by_id.get(comment_id)
        if previous_body is not None and previous_body != body:
            raise AgentLoopError(
                "Conflicting live bodies share one plan-validation diagnostic comment identity."
            )
        by_id[comment_id] = body
        if payload.expected_producer_login != expected_author_login or payload.expected_producer_id != expected_author_id:
            continue
        if not payload.matches_context(
            repository=repository,
            issue_number=issue_number,
            planning_generation=planning_generation,
            target_coder_round=target_coder_round,
            prior_plan_subject=prior_plan_subject,
            candidate_kind=candidate_kind,
            architecture_contract_version=architecture_contract_version,
            execution_strategy_contract_version=execution_strategy_contract_version,
            risk_test_matrix_contract_version=risk_test_matrix_contract_version,
        ):
            continue
        candidates.append(
            PlanValidationDiagnosticTransport(
                payload=payload,
                server_comment_id=comment_id,
                authoritative_created_at=created_at,
                exact_live_body=body,
                live_producer_login=author_login,
                live_producer_id=author_id,
            )
        )
    if not candidates:
        return None
    if _authenticated_canonical_plan_success_exists(
        comments,
        expected_author_login=expected_author_login,
        expected_author_id=expected_author_id,
        target_coder_round=target_coder_round,
        prior_plan_subject=prior_plan_subject,
        candidate_kind=candidate_kind,
        architecture_contract_version=architecture_contract_version,
        execution_strategy_contract_version=execution_strategy_contract_version,
        risk_test_matrix_contract_version=risk_test_matrix_contract_version,
    ):
        return None
    highest_attempt = max(item.payload.failure_attempt for item in candidates)
    highest = [item for item in candidates if item.payload.failure_attempt == highest_attempt]
    distinct_payloads = {json.dumps(item.payload.as_dict(), sort_keys=True) for item in highest}
    if len(distinct_payloads) > 1:
        raise AgentLoopError(
            "Conflicting authenticated plan-validation diagnostics claim the highest failure attempt."
        )
    return highest[0]


@dataclass(frozen=True)
class QualificationCheckpoint:
    """Bounded durable state for a machine-obligation qualification attempt."""

    obligation_kind: str
    obligation_identity: str
    lifecycle: str
    failed_head_sha: str | None
    candidate_head_sha: str | None
    base_branch: str | None = None
    approval_digest: str | None = None
    plan_digest: str | None = None
    requirements_digest: str | None = None
    acquisition_digest: str | None = None
    scheduler_digest: str | None = None
    qualification_attempt_id: str | None = None
    watch_failure_extension_used: bool = False
    watch_head_extension_used: bool = False
    allowed_rounds: int = 0
    valid: bool = True

    def __post_init__(self) -> None:
        if not self.valid:
            return
        if self.obligation_kind not in MACHINE_OBLIGATION_KINDS - {"unknown"}:
            raise ValueError("qualification checkpoints require a known machine kind")
        if self.lifecycle not in MACHINE_LIFECYCLE_STATES - {"cleared"}:
            raise ValueError("qualification checkpoints require an active lifecycle")
        if self.lifecycle == "repair_required" and self.obligation_kind in {
            "managed-exact-head-ci", "github-pr-checks"
        } and not self.failed_head_sha:
            raise ValueError("CI qualification checkpoints need a failed head")
        if self.lifecycle in {
            "awaiting_current_head_review", "qualification_ready", "qualifying"
        } and not self.candidate_head_sha:
            raise ValueError("qualification-ready checkpoints need a candidate head")
        if self.candidate_head_sha and self.candidate_head_sha == self.failed_head_sha:
            raise ValueError("qualification checkpoints cannot target the failed head")

    def as_dict(self) -> dict[str, object]:
        return {
            "obligation_kind": self.obligation_kind,
            "obligation_identity": self.obligation_identity,
            "lifecycle": self.lifecycle,
            "failed_head_sha": self.failed_head_sha,
            "candidate_head_sha": self.candidate_head_sha,
            "base_branch": self.base_branch,
            "approval_digest": self.approval_digest,
            "plan_digest": self.plan_digest,
            "requirements_digest": self.requirements_digest,
            "acquisition_digest": self.acquisition_digest,
            "scheduler_digest": self.scheduler_digest,
            "qualification_attempt_id": self.qualification_attempt_id,
            "watch_failure_extension_used": self.watch_failure_extension_used,
            "watch_head_extension_used": self.watch_head_extension_used,
            "allowed_rounds": self.allowed_rounds,
            "valid": self.valid,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "QualificationCheckpoint":
        if not isinstance(value, dict):
            return cls.invalid("checkpoint is not an object")
        required = {
            "obligation_kind", "obligation_identity", "lifecycle",
            "failed_head_sha", "candidate_head_sha", "base_branch",
            "approval_digest", "plan_digest", "requirements_digest",
            "acquisition_digest", "scheduler_digest", "qualification_attempt_id",
            "watch_failure_extension_used", "watch_head_extension_used",
            "allowed_rounds",
        }
        if not required.issubset(value):
            return cls.invalid("checkpoint is partial")
        strings = (
            "obligation_kind", "obligation_identity", "lifecycle", "failed_head_sha",
            "candidate_head_sha", "base_branch", "approval_digest", "plan_digest",
            "requirements_digest", "acquisition_digest", "scheduler_digest",
            "qualification_attempt_id",
        )
        if any(value[key] is not None and not isinstance(value[key], str) for key in strings):
            return cls.invalid("checkpoint contains a non-string identity")
        if value["obligation_kind"] not in MACHINE_OBLIGATION_KINDS - {"unknown"}:
            return cls.invalid("checkpoint has an unknown obligation kind")
        if value["lifecycle"] not in MACHINE_LIFECYCLE_STATES - {"cleared"}:
            return cls.invalid("checkpoint has an unknown lifecycle")
        if value["lifecycle"] in {"qualification_ready", "qualifying"}:
            required_identities = (
                "approval_digest",
                "requirements_digest",
                "acquisition_digest",
                "scheduler_digest",
            )
            if any(
                not isinstance(value[key], str) or not value[key].strip()
                for key in required_identities
            ):
                return cls.invalid(
                    "qualification checkpoint is missing a current contract identity"
                )
            if value["lifecycle"] == "qualifying" and (
                not isinstance(value["qualification_attempt_id"], str)
                or not value["qualification_attempt_id"].strip()
            ):
                return cls.invalid(
                    "qualifying checkpoint is missing its qualification attempt identity"
                )
        if (
            not isinstance(value["watch_failure_extension_used"], bool)
            or not isinstance(value["watch_head_extension_used"], bool)
            or isinstance(value["allowed_rounds"], bool)
            or not isinstance(value["allowed_rounds"], int)
            or value["allowed_rounds"] < 0
        ):
            return cls.invalid("checkpoint has invalid budget fields")
        if "valid" in value and value["valid"] is not True:
            return cls.invalid("checkpoint is not marked valid")
        failed = value["failed_head_sha"]
        candidate = value["candidate_head_sha"]
        if candidate is not None and candidate == failed:
            return cls.invalid("checkpoint requalifies the failed head")
        try:
            return cls(
                obligation_kind=value["obligation_kind"],
                obligation_identity=value["obligation_identity"],
                lifecycle=value["lifecycle"],
                failed_head_sha=failed,
                candidate_head_sha=candidate,
                base_branch=value["base_branch"],
                approval_digest=value["approval_digest"],
                plan_digest=value["plan_digest"],
                requirements_digest=value["requirements_digest"],
                acquisition_digest=value["acquisition_digest"],
                scheduler_digest=value["scheduler_digest"],
                qualification_attempt_id=value["qualification_attempt_id"],
                watch_failure_extension_used=value["watch_failure_extension_used"],
                watch_head_extension_used=value["watch_head_extension_used"],
                allowed_rounds=value["allowed_rounds"],
                valid=True,
            )
        except (TypeError, ValueError):
            # A well-typed but contradictory persisted checkpoint must be
            # treated exactly like any other malformed checkpoint. Recovery
            # can then force a fresh board instead of making the whole ledger
            # unresumable or accidentally treating it as ready.
            return cls.invalid("checkpoint contains contradictory lifecycle fields")

    @classmethod
    def invalid(cls, reason: str) -> "QualificationCheckpoint":
        return cls(
            obligation_kind="unknown",
            obligation_identity="invalid-checkpoint",
            lifecycle="repair_required",
            failed_head_sha=None,
            candidate_head_sha=None,
            allowed_rounds=0,
            valid=False,
        )


def _serialize_unresolved_item(item: UnresolvedReviewItem) -> dict[str, object]:
    payload = {
        "item_id": item.item_id,
        "reviewer": item.reviewer,
        "source_round": item.source_round,
        "text": item.text,
        "status": item.status,
        "source_status": item.source_status,
        "notes": list(item.notes),
        **({"fix_scope": list(item.fix_scope)} if item.fix_scope is not None else {}),
        **({"resolution_owners": list(item.resolution_owners)} if item.resolution_owners else {}),
        **({"owner_states": [list(pair) for pair in item.owner_states]} if item.owner_states else {}),
        **({"owner_evidence": [list(pair) for pair in item.owner_evidence]} if item.owner_evidence else {}),
        **({"owner_dispositions": [list(pair) for pair in item.owner_dispositions]} if item.owner_dispositions else {}),
    }
    machine_fields = {
        "authority": item.authority,
        "obligation_kind": item.obligation_kind,
        "lifecycle": item.lifecycle,
        "failed_head_sha": item.failed_head_sha,
        "candidate_head_sha": item.candidate_head_sha,
        "obligation_identity": item.obligation_identity,
    }
    payload.update({key: value for key, value in machine_fields.items() if value is not None})
    return payload


def _deserialize_unresolved_item(payload: object) -> UnresolvedReviewItem:
    if not isinstance(payload, dict):
        raise AgentLoopError("Invalid round metadata unresolved-item payload.")
    raw_notes = payload.get("notes") or []
    notes = tuple(str(note) for note in raw_notes) if isinstance(raw_notes, list) else ()
    raw_scope = payload.get("fix_scope")
    fix_scope = tuple(str(path) for path in raw_scope) if isinstance(raw_scope, list) else None
    raw_owners = payload.get("resolution_owners") or []
    owners = tuple(str(owner) for owner in raw_owners) if isinstance(raw_owners, list) else ()
    raw_states = payload.get("owner_states") or []
    states = tuple(
        (str(pair[0]), str(pair[1]))
        for pair in raw_states
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ) if isinstance(raw_states, list) else ()
    raw_evidence = payload.get("owner_evidence") or []
    evidence = tuple(
        (str(pair[0]), str(pair[1]))
        for pair in raw_evidence
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ) if isinstance(raw_evidence, list) else ()
    raw_dispositions = payload.get("owner_dispositions") or []
    owner_dispositions = tuple(
        (str(pair[0]), str(pair[1]))
        for pair in raw_dispositions
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ) if isinstance(raw_dispositions, list) else ()
    core = dict(
        item_id=str(payload["item_id"]),
        reviewer=str(payload["reviewer"]),
        source_round=int(payload["source_round"]),
        text=str(payload["text"]),
        status=str(payload["status"]),
        source_status=str(payload["source_status"]) if payload.get("source_status") is not None else None,
        notes=notes,
        fix_scope=fix_scope,
        resolution_owners=owners,
        owner_states=states,
        owner_evidence=evidence,
        owner_dispositions=owner_dispositions,
    )
    machine_keys = MACHINE_OBLIGATION_FIELDS
    if machine_keys & payload.keys():
        for key in machine_keys:
            if key in payload and payload[key] is not None and not isinstance(payload[key], str):
                # Invalid persisted authority is itself a blocker.  Do not
                # reject the entire ledger and do not let a malformed field
                # fall back to an ordinary reviewer finding.
                return UnresolvedReviewItem(
                    **{**core, "notes": (*notes, f"Invalid persisted machine field: {key}")},
                    authority=UNKNOWN_MACHINE_AUTHORITY,
                    obligation_kind="unknown",
                    lifecycle="repair_required",
                    obligation_identity="invalid-machine-record",
                    # ``core`` already carries the original notes; append the
                    # diagnostic through a copied mapping to avoid duplicate
                    # keyword arguments.
                )
        machine = dict(
            authority=payload.get("authority"),
            obligation_kind=payload.get("obligation_kind"),
            lifecycle=payload.get("lifecycle"),
            failed_head_sha=payload.get("failed_head_sha"),
            candidate_head_sha=payload.get("candidate_head_sha"),
            obligation_identity=payload.get("obligation_identity"),
        )
        try:
            item = UnresolvedReviewItem(**core, **machine)
            # ``None`` is a valid default for the in-memory dataclass, but a
            # serialized record that contains machine keys and no usable
            # machine value is still a machine-shaped record.  Preserve it as
            # an explicit unknown blocker instead of reviving a reviewer
            # finding that dispositions can clear.
            if not item.is_machine_obligation:
                raise ValueError("machine record contains no valid machine fields")
            return item
        except ValueError as exc:
            return UnresolvedReviewItem(
                **{**core, "notes": (*notes, f"Invalid persisted machine record: {exc}")},
                authority=UNKNOWN_MACHINE_AUTHORITY,
                obligation_kind="unknown",
                lifecycle="repair_required",
                obligation_identity="invalid-machine-record",
            )
    return UnresolvedReviewItem(**core)


_SCHEDULER_METADATA_KEYS = frozenset(
    {
        "scheduler_contract",
        "scheduler_previous_sha",
        "scheduler_current_sha",
        "scheduler_obligation_digest",
        "scheduler_selected_reviewers",
        "scheduler_paused_reviewers",
        "scheduler_reasons",
        "scheduler_final_sweep",
        "scheduler_force_full",
        "scheduler_calls_avoided",
    }
)
_SCHEDULER_AUXILIARY_KEYS = frozenset(
    {
        "scheduler_phase",
        "scheduler_primary_reviewer",
        "scheduler_approved_reviewers",
        "scheduler_active_owners",
        "scheduler_scope_digest",
    }
)


def _decode_scheduler_fields(payload: Mapping[str, object]) -> dict[str, object]:
    """Decode scheduler metadata, dropping it conservatively when unsafe.

    Scheduler records are an optimization over the required full-board review.
    A malformed or partial record therefore becomes legacy metadata instead of
    being allowed to select a smaller reviewer set during resume.
    """
    scheduler_keys = _SCHEDULER_METADATA_KEYS | _SCHEDULER_AUXILIARY_KEYS
    if not (scheduler_keys & payload.keys()):
        return {"scheduler_metadata_status": "absent"}
    required = _SCHEDULER_METADATA_KEYS
    if not required.issubset(payload.keys()):
        return {"scheduler_metadata_status": "invalid"}
    try:
        contract = ReviewSchedulingContract.from_mapping(payload["scheduler_contract"])
        previous = payload["scheduler_previous_sha"]
        current = payload["scheduler_current_sha"]
        digest = payload["scheduler_obligation_digest"]
        if previous is not None and (not isinstance(previous, str) or not previous):
            raise ValueError("invalid previous scheduler SHA")
        if not isinstance(current, str) or not current:
            raise ValueError("invalid current scheduler SHA")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{16}", digest):
            raise ValueError("invalid scheduler obligation digest")
        selected = payload["scheduler_selected_reviewers"]
        paused = payload["scheduler_paused_reviewers"]
        reasons = payload["scheduler_reasons"]
        if not isinstance(selected, list) or any(not isinstance(name, str) for name in selected):
            raise ValueError("invalid selected reviewer list")
        if len(set(selected)) != len(selected) or not set(selected).issubset(contract.required_reviewers):
            raise ValueError("contradictory selected reviewer list")
        if not isinstance(paused, list):
            raise ValueError("invalid paused reviewer list")
        paused_pairs: list[tuple[str, str]] = []
        for pair in paused:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not isinstance(pair[1], str)
                or not pair[0]
                or not pair[1]
            ):
                raise ValueError("invalid paused reviewer entry")
            paused_pairs.append((pair[0], pair[1]))
        paused_names = [name for name, _reason in paused_pairs]
        if (
            len(set(paused_names)) != len(paused_names)
            or not set(paused_names).issubset(contract.required_reviewers)
            or set(selected) & set(paused_names)
            or set(selected) | set(paused_names) != set(contract.required_reviewers)
        ):
            raise ValueError("contradictory reviewer scheduling state")
        if not isinstance(reasons, list) or any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("invalid scheduler reasons")
        final_sweep = payload["scheduler_final_sweep"]
        force_full = payload["scheduler_force_full"]
        calls_avoided = payload["scheduler_calls_avoided"]
        if not isinstance(final_sweep, bool) or not isinstance(force_full, bool):
            raise ValueError("invalid scheduler boolean")
        if isinstance(calls_avoided, bool) or not isinstance(calls_avoided, int) or calls_avoided < 0:
            raise ValueError("invalid avoided-call count")
        phase = payload.get("scheduler_phase")
        if phase is not None and (not isinstance(phase, str) or phase not in SCHEDULER_PHASES):
            raise ValueError("invalid scheduler phase")
        primary = payload.get("scheduler_primary_reviewer")
        if primary is not None and (
            not isinstance(primary, str) or primary != contract.primary_reviewer
        ):
            raise ValueError("contradictory scheduler primary reviewer")
        approved = payload.get("scheduler_approved_reviewers", [])
        if (
            not isinstance(approved, list)
            or any(not isinstance(name, str) for name in approved)
            or len(set(approved)) != len(approved)
            or not set(approved).issubset(contract.required_reviewers)
        ):
            raise ValueError("invalid scheduler approval set")
        owners = payload.get("scheduler_active_owners", [])
        if (
            not isinstance(owners, list)
            or any(not isinstance(name, str) or not name for name in owners)
            or len(set(owners)) != len(owners)
        ):
            raise ValueError("invalid scheduler owner set")
        scope_digest = payload.get("scheduler_scope_digest")
        if scope_digest is not None and (
            not isinstance(scope_digest, str) or not re.fullmatch(r"[0-9a-f]{16}", scope_digest)
        ):
            raise ValueError("invalid scheduler scope digest")
    except (AgentLoopError, TypeError, ValueError, KeyError):
        return {"scheduler_metadata_status": "invalid"}
    return {
        "scheduler_contract": contract.as_dict(),
        "scheduler_previous_sha": previous,
        "scheduler_current_sha": current,
        "scheduler_obligation_digest": digest,
        "scheduler_selected_reviewers": tuple(selected),
        "scheduler_paused_reviewers": tuple(paused_pairs),
        "scheduler_reasons": tuple(reasons),
        "scheduler_final_sweep": final_sweep,
        "scheduler_force_full": force_full,
        "scheduler_calls_avoided": calls_avoided,
        "scheduler_phase": phase,
        "scheduler_primary_reviewer": primary,
        "scheduler_approved_reviewers": tuple(approved),
        "scheduler_active_owners": tuple(owners),
        "scheduler_scope_digest": scope_digest,
        "scheduler_metadata_status": "valid",
    }


def _serialize_disposition(disposition: ReviewItemDisposition) -> dict[str, object]:
    return {
        "item_id": disposition.item_id,
        "reviewer": disposition.reviewer,
        "disposition": disposition.disposition,
        "note": disposition.note,
    }


def _deserialize_disposition(payload: object) -> ReviewItemDisposition:
    if not isinstance(payload, dict):
        raise AgentLoopError("Invalid round metadata disposition payload.")
    return ReviewItemDisposition(
        item_id=str(payload["item_id"]),
        reviewer=str(payload["reviewer"]),
        disposition=str(payload["disposition"]),
        note=str(payload["note"]) if payload.get("note") is not None else None,
    )


def _matrix_json(value: object) -> str | None:
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _decode_matrix_json(value: object, *, context: str) -> object | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or len(value.encode("utf-8")) > 200_000:
        raise ValueError(f"{context} is not a bounded JSON payload")
    return json.loads(value)


def _risk_test_matrix_metadata_present(metadata: PostedRoundMetadata) -> bool:
    """Return whether metadata carries a matrix record rather than defaults.

    ``risk_test_matrix_changes_payload`` defaults to an empty tuple for
    backwards-compatible in-memory construction.  That default is not a
    durable matrix discriminator and must not make a legacy record appear
    matrix-bearing during recovery.
    """
    return any(
        value is not None
        for value in (
            metadata.risk_test_matrix_contract_version,
            metadata.risk_test_matrix_payload,
            metadata.risk_test_matrix_identity,
            metadata.risk_test_matrix_boundary_digest,
        )
    ) or bool(metadata.risk_test_matrix_changes_payload)


def _matrix_metadata_fields(
    payload: Mapping[str, object], *, canonical_text: str | None = None
) -> dict[str, object]:
    """Decode and independently authenticate the structured matrix carrier.

    Matrix corruption is deliberately converted into a closed matrix channel;
    callers can still recover an otherwise hash/subject-valid approved plan.
    """
    raw_version = payload.get("risk_test_matrix_contract_version")
    raw_matrix = payload.get("risk_test_matrix_payload")
    raw_changes = payload.get("risk_test_matrix_changes_payload")
    raw_identity = payload.get("risk_test_matrix_identity")
    raw_boundary = payload.get("risk_test_matrix_boundary_digest")
    if all(value is None for value in (raw_version, raw_matrix, raw_changes, raw_identity, raw_boundary)):
        return {}
    try:
        version = int(raw_version)
        if version != 1:
            raise ValueError("unsupported matrix contract")
        matrix_value = _decode_matrix_json(raw_matrix, context="risk_test_matrix_payload")
        changes_value = _decode_matrix_json(raw_changes, context="risk_test_matrix_changes_payload")
        matrix = parse_risk_test_matrix(matrix_value, context="stored risk_test_matrix_payload")
        changes = parse_risk_test_matrix_changes(
            changes_value if changes_value is not None else [],
            context="stored risk_test_matrix_changes_payload",
        )
        identity = risk_test_matrix_identity(matrix, changes, contract_version=version)
        if not isinstance(raw_identity, str) or raw_identity != identity:
            raise ValueError("matrix identity mismatch")
        if raw_boundary != identity:
            raise ValueError("matrix payload-bound section boundary mismatch")
        if canonical_text and not _canonical_matrix_boundary_matches(
            canonical_text, identity
        ):
            raise ValueError("canonical plan risk matrix section boundary mismatch")
        return {
            "risk_test_matrix_contract_version": version,
            "risk_test_matrix_payload": matrix.to_payload(),
            "risk_test_matrix_changes_payload": tuple(change.to_payload() for change in changes),
            "risk_test_matrix_identity": identity,
            "risk_test_matrix_boundary_digest": raw_boundary,
            "risk_test_matrix_diagnostic": None,
        }
    except (AgentLoopError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "risk_test_matrix_contract_version": 1 if raw_version == 1 else None,
            "risk_test_matrix_payload": None,
            "risk_test_matrix_changes_payload": (),
            "risk_test_matrix_identity": None,
            "risk_test_matrix_boundary_digest": None,
            "risk_test_matrix_diagnostic": f"Matrix unavailable: {exc}",
        }


def _canonical_matrix_boundary_matches(text: str, identity: str) -> bool:
    """Authenticate the stored renderer boundary without re-rendering prose."""
    from .round_transport import risk_test_matrix_section_boundary

    if risk_test_matrix_section_boundary(identity) not in text:
        return False
    marker = RISK_TEST_MATRIX_MARKER_RE.search(text)
    if marker is None:
        return False
    try:
        return decode_risk_test_matrix_marker(marker.group("payload"))["identity"] == identity
    except (AgentLoopError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _sanitize_durable_coder_response(raw: str | None) -> str | None:
    """Remove invocation-local semantic selectors before durable persistence.

    ``risk_test_matrix_claims`` is a fresh-turn acquisition contract. Its
    execution handles are only meaningful in the closed broker catalog for
    that invocation and must never be restored as if they were receipt
    authority. The derived evidence and diagnostics are persisted in their
    dedicated metadata fields, so dropping the ephemeral claim set from the
    historical raw response preserves useful resume context without making a
    later invocation selector-capable.
    """
    if not raw:
        return raw
    normalized = raw.lstrip()
    try:
        payload, end = json.JSONDecoder().raw_decode(normalized)
    except (TypeError, json.JSONDecodeError):
        return raw
    if not isinstance(payload, dict) or payload.get("kind") not in {
        "issue_implementation",
        "coder_followup",
    }:
        return raw
    if "risk_test_matrix_claims" not in payload:
        return raw
    payload = dict(payload)
    payload.pop("risk_test_matrix_claims", None)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + normalized[end:]


def _encode_round_metadata(metadata: PostedRoundMetadata) -> str:
    payload = {
        "flow": metadata.flow,
        "role": metadata.role,
        "agent": metadata.agent,
        "round_number": metadata.round_number,
        "subject": metadata.subject,
        "prior_plan_subject": metadata.prior_plan_subject,
        "prior_items": [_serialize_unresolved_item(item) for item in metadata.prior_items],
        "dispositions": [_serialize_disposition(item) for item in metadata.dispositions],
        "new_items": [_serialize_unresolved_item(item) for item in metadata.new_items],
        "state": metadata.state,
        "canonical_plan": metadata.canonical_plan,
        "raw_structured_coder_response": _sanitize_durable_coder_response(
            metadata.raw_structured_coder_response
        ),
        "approved_plan_hash": metadata.approved_plan_hash,
        "approved_plan_subject": metadata.approved_plan_subject,
        "architecture_identity": metadata.architecture_identity,
        "architecture_impact": metadata.architecture_impact,
        "architecture_contract_version": metadata.architecture_contract_version,
        "execution_strategy_contract_version": metadata.execution_strategy_contract_version,
        "execution_strategy_identity": metadata.execution_strategy_identity,
        "compact_prior_summaries": list(metadata.compact_prior_summaries),
        "usage": metadata.usage,
        "model_used": metadata.model_used,
        "provider": metadata.provider,
        "configured_model": metadata.configured_model,
        "configured_effort": metadata.configured_effort,
        "effort_source": metadata.effort_source,
        "observed_model": metadata.observed_model,
        "observed_effort": metadata.observed_effort,
        "observation_provenance": metadata.observation_provenance,
        "acquisition_outcome": metadata.acquisition_outcome,
        "acquisition_returncode": metadata.acquisition_returncode,
        "consensus_kind": metadata.consensus_kind,
        "is_final": metadata.is_final,
        "agenda": list(metadata.agenda),
        "analyzer_response": metadata.analyzer_response,
        "final_analyzer_response": metadata.final_analyzer_response,
        "research_mode": metadata.research_mode,
        "failed_debaters": [list(pair) for pair in metadata.failed_debaters],
        "split_proposals": list(metadata.split_proposals),
        "result_mode": metadata.result_mode,
        "evidence_reconciliation": metadata.evidence_reconciliation,
        "surfaced_reviewer_requirement_ids": list(metadata.surfaced_reviewer_requirement_ids),
        "phase": metadata.phase,
        "canonical_reviewer_response": metadata.canonical_reviewer_response,
        "local_test_evidence": (
            canonicalize_bounded_evidence(metadata.local_test_evidence)
            if metadata.local_test_evidence is not None else None
        ),
        "risk_test_matrix_evidence": metadata.risk_test_matrix_evidence,
        "risk_test_matrix_diagnostics": [
            dict(item) for item in metadata.risk_test_matrix_diagnostics
        ],
    }
    semantic_values = {
        "response_form": metadata.response_form,
        "base_round_number": metadata.base_round_number,
        "base_state_identity": metadata.base_state_identity,
        "aggregate_plan_identity": metadata.aggregate_plan_identity,
        "raw_patch_provenance": metadata.raw_patch_provenance,
        "assembled_plan_sidecar": metadata.assembled_plan_sidecar,
    }
    if any(value not in (None, (), []) for value in semantic_values.values()):
        payload.update(
            {
                key: value
                for key, value in semantic_values.items()
                if value not in (None, (), [])
            }
        )
    if metadata.result_mode == "answer":
        payload.update(
            {
                "round_synthesis": metadata.round_synthesis,
                "final_synthesis": metadata.final_synthesis,
                "raw_synthesis_response": metadata.raw_synthesis_response,
                "synthesis_provenance": metadata.synthesis_provenance,
            }
        )
    scheduler_values = {
        "scheduler_contract": metadata.scheduler_contract,
        "scheduler_previous_sha": metadata.scheduler_previous_sha,
        "scheduler_current_sha": metadata.scheduler_current_sha,
        "scheduler_obligation_digest": metadata.scheduler_obligation_digest,
        "scheduler_selected_reviewers": list(metadata.scheduler_selected_reviewers),
        "scheduler_paused_reviewers": [list(pair) for pair in metadata.scheduler_paused_reviewers],
        "scheduler_reasons": list(metadata.scheduler_reasons),
        "scheduler_final_sweep": metadata.scheduler_final_sweep,
        "scheduler_force_full": metadata.scheduler_force_full,
        "scheduler_calls_avoided": metadata.scheduler_calls_avoided,
        "scheduler_phase": metadata.scheduler_phase,
        "scheduler_primary_reviewer": metadata.scheduler_primary_reviewer,
        "scheduler_approved_reviewers": list(metadata.scheduler_approved_reviewers),
        "scheduler_active_owners": list(metadata.scheduler_active_owners),
        "scheduler_scope_digest": metadata.scheduler_scope_digest,
    }
    if any(value not in (None, (), []) for value in scheduler_values.values()):
        # Phase-aware fields are optional: omit empty ones so records written
        # by the existing policies keep the legacy mandatory key shape.
        payload.update(
            {
                key: value
                for key, value in scheduler_values.items()
                if key not in _SCHEDULER_AUXILIARY_KEYS or value not in (None, [])
            }
        )
    matrix_present = _risk_test_matrix_metadata_present(metadata)
    matrix_values = {
        "risk_test_matrix_contract_version": metadata.risk_test_matrix_contract_version,
        "risk_test_matrix_payload": _matrix_json(metadata.risk_test_matrix_payload),
        "risk_test_matrix_changes_payload": (
            _matrix_json(list(metadata.risk_test_matrix_changes_payload))
            if matrix_present else None
        ),
        "risk_test_matrix_identity": metadata.risk_test_matrix_identity,
        "risk_test_matrix_boundary_digest": metadata.risk_test_matrix_boundary_digest,
    }
    if any(value not in (None, (), []) for value in matrix_values.values()):
        payload.update(matrix_values)
    if metadata.qualification_checkpoint is not None:
        payload["qualification_checkpoint"] = (
            metadata.qualification_checkpoint.as_dict()
            if isinstance(metadata.qualification_checkpoint, QualificationCheckpoint)
            else metadata.qualification_checkpoint
        )
    return encode_mapping(payload)


def _decode_round_metadata_mapping(payload: Mapping[str, object]) -> PostedRoundMetadata:
    try:
        semantic_keys_present = _SEMANTIC_METADATA_FIELDS.intersection(payload)
        if semantic_keys_present:
            # Do not coerce malformed new authority to None: None is reserved
            # for true legacy absence (or the explicitly absent patch-base
            # fields of a full-state seed).
            if "response_form" not in payload:
                raise ValueError("semantic metadata is missing response_form")
            for key in semantic_keys_present:
                if payload[key] is None:
                    raise ValueError(f"semantic metadata field {key} may not be null")
            response_form = payload.get("response_form")
            if response_form is not None and not isinstance(response_form, str):
                raise ValueError("semantic response_form must be a string")
            base_round = payload.get("base_round_number")
            if base_round is not None and (
                isinstance(base_round, bool) or not isinstance(base_round, int)
            ):
                raise ValueError("semantic base_round_number must be an integer")
            for key in ("base_state_identity", "aggregate_plan_identity"):
                identity = payload.get(key)
                if identity is not None and not isinstance(identity, str):
                    raise ValueError(f"semantic {key} must be a string")
            raw_patch = payload.get("raw_patch_provenance")
            if raw_patch is not None and not isinstance(raw_patch, dict):
                raise ValueError("semantic raw_patch_provenance must be an object")
            sidecar = payload.get("assembled_plan_sidecar")
            if sidecar is not None and not isinstance(sidecar, dict):
                raise ValueError("semantic assembled_plan_sidecar must be an object")
        return PostedRoundMetadata(
            flow=str(payload["flow"]),
            role=str(payload["role"]),
            agent=str(payload["agent"]),
            round_number=int(payload["round_number"]),
            subject=str(payload["subject"]),
            prior_plan_subject=(
                str(payload["prior_plan_subject"])
                if payload.get("prior_plan_subject") is not None
                else None
            ),
            prior_items=tuple(_deserialize_unresolved_item(item) for item in payload.get("prior_items", [])),
            dispositions=tuple(_deserialize_disposition(item) for item in payload.get("dispositions", [])),
            new_items=tuple(_deserialize_unresolved_item(item) for item in payload.get("new_items", [])),
            state=str(payload["state"]) if payload.get("state") is not None else None,
            canonical_plan=(
                str(payload["canonical_plan"])
                if payload.get("canonical_plan") is not None
                else None
            ),
            raw_structured_coder_response=(
                str(payload["raw_structured_coder_response"])
                if payload.get("raw_structured_coder_response") is not None
                else None
            ),
            response_form=(
                payload["response_form"]
                if "response_form" in payload else None
            ),
            base_round_number=(
                payload["base_round_number"]
                if "base_round_number" in payload else None
            ),
            base_state_identity=(
                payload["base_state_identity"]
                if "base_state_identity" in payload else None
            ),
            aggregate_plan_identity=(
                payload["aggregate_plan_identity"]
                if "aggregate_plan_identity" in payload else None
            ),
            raw_patch_provenance=(
                payload["raw_patch_provenance"]
                if "raw_patch_provenance" in payload else None
            ),
            assembled_plan_sidecar=(
                payload["assembled_plan_sidecar"]
                if "assembled_plan_sidecar" in payload else None
            ),
            approved_plan_hash=(
                str(payload["approved_plan_hash"])
                if payload.get("approved_plan_hash") is not None
                else None
            ),
            approved_plan_subject=(
                str(payload["approved_plan_subject"])
                if payload.get("approved_plan_subject") is not None
                else None
            ),
            architecture_identity=(
                payload.get("architecture_identity")
                if isinstance(payload.get("architecture_identity"), dict) else None
            ),
            architecture_impact=(
                payload.get("architecture_impact")
                if isinstance(payload.get("architecture_impact"), dict) else None
            ),
            architecture_contract_version=(
                int(payload["architecture_contract_version"])
                if payload.get("architecture_contract_version") is not None else None
            ),
            execution_strategy_contract_version=(
                int(payload["execution_strategy_contract_version"])
                if payload.get("execution_strategy_contract_version") is not None else None
            ),
            execution_strategy_identity=(
                payload.get("execution_strategy_identity")
                if isinstance(payload.get("execution_strategy_identity"), dict) else None
            ),
            **_matrix_metadata_fields(
                payload,
                canonical_text=(
                    str(payload["canonical_plan"])
                    if payload.get("canonical_plan") is not None
                    else None
                ),
            ),
            compact_prior_summaries=tuple(
                str(summary) for summary in payload.get("compact_prior_summaries", [])
            ),
            usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
            model_used=str(payload["model_used"]) if payload.get("model_used") is not None else None,
            provider=str(payload["provider"]) if payload.get("provider") is not None else None,
            configured_model=(
                str(payload["configured_model"])
                if payload.get("configured_model") is not None else None
            ),
            configured_effort=(
                str(payload["configured_effort"])
                if payload.get("configured_effort") is not None else None
            ),
            effort_source=(
                str(payload["effort_source"])
                if payload.get("effort_source") is not None else None
            ),
            observed_model=(
                str(payload["observed_model"])
                if payload.get("observed_model") is not None else None
            ),
            observed_effort=(
                str(payload["observed_effort"])
                if payload.get("observed_effort") is not None else None
            ),
            observation_provenance=(
                str(payload["observation_provenance"])
                if payload.get("observation_provenance") is not None else None
            ),
            acquisition_outcome=str(payload.get("acquisition_outcome", "success")),
            acquisition_returncode=(
                int(payload["acquisition_returncode"])
                if payload.get("acquisition_returncode") is not None else None
            ),
            consensus_kind=str(payload["consensus_kind"]) if payload.get("consensus_kind") is not None else None,
            is_final=bool(payload.get("is_final", False)),
            agenda=tuple(str(item) for item in payload.get("agenda", [])),
            analyzer_response=(
                str(payload["analyzer_response"])
                if payload.get("analyzer_response") is not None
                else None
            ),
            final_analyzer_response=(
                str(payload["final_analyzer_response"])
                if payload.get("final_analyzer_response") is not None
                else None
            ),
            research_mode=(
                str(payload["research_mode"])
                if payload.get("research_mode") is not None
                else None
            ),
            failed_debaters=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in payload.get("failed_debaters", [])
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            ),
            split_proposals=tuple(str(item) for item in payload.get("split_proposals", [])),
            result_mode=str(payload.get("result_mode", "triage")),
            evidence_reconciliation=(
                payload.get("evidence_reconciliation")
                if isinstance(payload.get("evidence_reconciliation"), dict)
                else None
            ),
            surfaced_reviewer_requirement_ids=tuple(
                str(item) for item in payload.get("surfaced_reviewer_requirement_ids", [])
            ),
            phase=str(payload.get("phase", "authoritative")),
            canonical_reviewer_response=(
                str(payload["canonical_reviewer_response"])
                if payload.get("canonical_reviewer_response") is not None else None
            ),
            local_test_evidence=(
                canonicalize_bounded_evidence(payload.get("local_test_evidence"))
            ),
            risk_test_matrix_evidence=(
                payload.get("risk_test_matrix_evidence")
                if isinstance(payload.get("risk_test_matrix_evidence"), dict)
                else None
            ),
            risk_test_matrix_diagnostics=tuple(
                dict(item)
                for item in payload.get("risk_test_matrix_diagnostics", [])
                if isinstance(item, dict)
            ),
            qualification_checkpoint=(
                QualificationCheckpoint.from_mapping(payload["qualification_checkpoint"])
                if payload.get("qualification_checkpoint") is not None
                else None
            ),
            round_synthesis=(
                str(payload["round_synthesis"])
                if payload.get("round_synthesis") is not None else None
            ),
            final_synthesis=(
                str(payload["final_synthesis"])
                if payload.get("final_synthesis") is not None else None
            ),
            raw_synthesis_response=(
                str(payload["raw_synthesis_response"])
                if payload.get("raw_synthesis_response") is not None else None
            ),
            synthesis_provenance=(
                payload.get("synthesis_provenance")
                if isinstance(payload.get("synthesis_provenance"), dict) else None
            ),
            **_decode_scheduler_fields(payload),
        )
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise AgentLoopError(f"Invalid AGENT_LOOP_META payload: {exc}") from exc


def _decode_round_metadata(encoded: str) -> PostedRoundMetadata:
    return _decode_round_metadata_mapping(decode_mapping(encoded))


def _attach_round_metadata(body: str, metadata: PostedRoundMetadata) -> str:
    marker = f"<!-- AGENT_LOOP_META: {_encode_round_metadata(metadata)} -->"
    lines = body.splitlines()
    index = len(lines)
    while index > 0 and not lines[index - 1].strip():
        index -= 1
    metadata_start = index
    while metadata_start > 0:
        candidate = lines[metadata_start - 1]
        if not candidate.strip() or HTML_COMMENT_RE.match(candidate) or SIGNATURE_RE.match(candidate):
            metadata_start -= 1
            continue
        break
    prefix = "\n".join(lines[:metadata_start]).rstrip("\n")
    suffix = "\n".join(lines[metadata_start:]).lstrip("\n")
    if not prefix:
        rendered = "\n".join(part for part in (marker, suffix) if part)
    elif not suffix:
        rendered = "\n".join((prefix, marker))
    else:
        rendered = "\n".join((prefix, marker, suffix))
    expected = tuple(item.definition.token for item in scan_reserved_markers(rendered))
    if "AGENT_LOOP_META" not in expected:
        expected = (*expected, "AGENT_LOOP_META")
    return TrustedBody.canonical(rendered, expected_tokens=expected)


def _strip_round_metadata(body: str) -> str:
    cleaned = re.sub(
        r"\n?\s*<!--\s*AGENT_LOOP_META:\s*[A-Za-z0-9+/=_-]+\s*-->\s*\n?",
        "\n",
        body,
        flags=re.I,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


_LEGACY_MACHINE_REVIEWERS = {
    "GitHub managed exact-head CI": "managed-exact-head-ci",
    "GitHub PR checks": "github-pr-checks",
    "Alembic migration validation": "alembic-migration",
}


def _legacy_machine_kind(item: UnresolvedReviewItem) -> str | None:
    if item.item_id == "item-merge-conflict":
        return "merge-conflict"
    if item.item_id == "item-human-requirements-acknowledgement":
        return "human-requirements-acknowledgement"
    return _LEGACY_MACHINE_REVIEWERS.get(item.reviewer) or (
        "unknown" if item.reviewer == "Orchestrator" else None
    )


def _legacy_machine_promotion(
    item: UnresolvedReviewItem,
    *,
    record: PostedRoundRecord,
    first_record: PostedRoundRecord | None,
) -> UnresolvedReviewItem:
    """Promote only orchestrator-lineage synthetic items from old ledgers.

    A display name in reviewer prose is not authority.  The first durable
    ``new_items`` record (or a coder checkpoint carrying a legacy machine item)
    must be an orchestrator-produced record before a known kind is promoted.
    Ambiguous synthetic records become explicit unknown blockers.
    """
    if item.is_machine_obligation:
        return item
    kind = _legacy_machine_kind(item)
    if kind is None:
        return item
    lineage = first_record or record
    trusted = (
        lineage.metadata.role in {"summary", "coder"}
        and (
            lineage.metadata.agent == "Orchestrator"
            or lineage.metadata.role == "coder"
        )
        and item.status in {"blocking", "same-pr"}
        and lineage.metadata.round_number >= item.source_round
    )
    if not trusted:
        return replace(
            item,
            authority=UNKNOWN_MACHINE_AUTHORITY,
            obligation_kind="unknown",
            lifecycle="repair_required",
            failed_head_sha=None,
            candidate_head_sha=None,
            obligation_identity=f"unknown:{item.item_id}",
            notes=(*item.notes, "Synthetic machine record lacked trusted orchestrator lineage."),
            resolution_owners=(),
            owner_states=(),
        )
    failed_head: str | None = None
    if lineage.metadata.role == "summary" and lineage.metadata.agent == "Orchestrator":
        # An orchestrator summary's subject is the authoritative PR head for
        # the recorded machine event.
        failed_head = lineage.metadata.subject
    elif lineage.metadata.role == "coder":
        # A legacy CI failure was often first carried by the coder checkpoint,
        # after the coder had already pushed a repair head. The scheduler
        # previous/current pair is the producer-owned provenance that lets us
        # recover the failed head; visible CI prose is not authority.
        previous_head = lineage.metadata.scheduler_previous_sha
        current_head = lineage.metadata.scheduler_current_sha
        if (
            previous_head
            and current_head
            and lineage.metadata.subject == current_head
            and previous_head != current_head
        ):
            failed_head = previous_head
    if not failed_head or failed_head == "unknown":
        return replace(
            item,
            authority=UNKNOWN_MACHINE_AUTHORITY,
            obligation_kind="unknown",
            lifecycle="repair_required",
            obligation_identity=f"unknown:{item.item_id}",
            notes=(*item.notes, "Known machine item could not be bound to a failed head."),
            resolution_owners=(),
            owner_states=(),
        )
    return replace(
        item,
        authority=MACHINE_AUTHORITY,
        obligation_kind=kind,
        lifecycle="repair_required",
        failed_head_sha=failed_head,
        candidate_head_sha=None,
        obligation_identity=f"{kind}:{item.item_id}",
        resolution_owners=(),
        owner_states=(),
    )


def _promote_legacy_machine_items(records: Sequence[PostedRoundRecord]) -> tuple[PostedRoundRecord, ...]:
    first_new_item_record: dict[str, PostedRoundRecord] = {}
    for record in records:
        for item in record.metadata.new_items:
            first_new_item_record.setdefault(item.item_id, record)
    promoted: list[PostedRoundRecord] = []
    for record in records:
        metadata = record.metadata
        prior = tuple(
            _legacy_machine_promotion(
                item,
                record=record,
                first_record=first_new_item_record.get(item.item_id),
            )
            for item in metadata.prior_items
        )
        new_items = tuple(
            _legacy_machine_promotion(
                item,
                record=record,
                first_record=first_new_item_record.get(item.item_id, record),
            )
            for item in metadata.new_items
        )
        if prior != metadata.prior_items or new_items != metadata.new_items:
            metadata = replace(metadata, prior_items=prior, new_items=new_items)
        promoted.append(replace(record, metadata=metadata))
    return tuple(promoted)


def _extract_round_metadata_records(comments: Sequence[object], *, flow: str) -> tuple[PostedRoundRecord, ...]:
    records: list[PostedRoundRecord] = []
    bodies = tuple(body for comment in comments if isinstance((body := getattr(comment, "body", None)), str))
    for index, comment in enumerate(comments):
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        matches = list(ROUND_RESUME_MARKER_RE.finditer(body))
        if not matches:
            continue
        payload, missing = hydrate_mapping(decode_mapping(matches[-1].group("payload")), bodies)
        # Synthesis is advisory. A missing synthesis sidecar must not make a
        # durable vote or legacy agenda unrecoverable; those fields simply
        # decode as None. Existing PR/review spill fields remain strict.
        discuss_optional_missing = (
            {
                "round_synthesis", "final_synthesis", "raw_synthesis_response",
                "analyzer_response", "final_analyzer_response",
            }
            if flow == "discuss"
            else set()
        )
        required_missing = missing - discuss_optional_missing
        if required_missing:
            raise AgentLoopError(
                "Incomplete round metadata: "
                f"{', '.join(sorted(required_missing))} sidecars are unavailable; "
                "restore sidecars or remove the incomplete anchor and rerun."
            )
        metadata = _decode_round_metadata_mapping(payload)
        if metadata.flow != flow:
            continue
        records.append(
            PostedRoundRecord(
                index=index,
                metadata=metadata,
                body=_strip_round_metadata(body),
            )
        )
    return _promote_legacy_machine_items(tuple(records))


def _latest_pr_approved_reviews_for_head(
    comments: Sequence[object],
    *,
    head_sha: str | None,
    configured_reviewers: Sequence[AgentName],
    approved_plan_context: ApprovedPlanContext | None = None,
    human_requirements: Sequence[object] = (),
    reviewer_acquisition_contract: Mapping[str, tuple[object, ...]] | None = None,
    require_architecture_contract: bool = False,
) -> dict[str, PostedRoundRecord]:
    """Return each reviewer's latest approval for the current immutable PR head."""
    if not head_sha:
        return {}
    configured_names = {agent_display_name(agent) for agent in configured_reviewers}
    latest_by_reviewer: dict[str, PostedRoundRecord] = {}
    for record in reversed(_extract_round_metadata_records(comments, flow="pr")):
        metadata = record.metadata
        if (
            metadata.subject != head_sha
            or metadata.role != "reviewer"
            or metadata.agent not in configured_names
            or metadata.agent in latest_by_reviewer
        ):
            continue
        # A fresh architecture-aware resume must be explicitly bound to the
        # current response contract.  Legacy records remain decodable, but
        # their missing discriminator cannot satisfy a newly prompted turn.
        if require_architecture_contract and metadata.architecture_contract_version != 1:
            continue
        latest_by_reviewer[metadata.agent] = record

    def acquisition_contract_matches(record: PostedRoundRecord) -> bool:
        if reviewer_acquisition_contract is None:
            return True
        expected = reviewer_acquisition_contract.get(record.metadata.agent)
        if expected is None or len(expected) < 2:
            return False
        if record.metadata.acquisition_outcome != "success":
            return False
        return (
            record.metadata.configured_model == expected[0]
            and record.metadata.configured_effort == expected[1]
            and (len(expected) < 3 or record.metadata.provider == expected[2])
        )

    return {
        reviewer: record
        for reviewer, record in latest_by_reviewer.items()
        if record.metadata.state == "approved"
        and acquisition_contract_matches(record)
        and (
            approved_plan_context is None
            or (
                record.metadata.approved_plan_hash == approved_plan_context.plan_hash
                and record.metadata.approved_plan_subject == approved_plan_context.plan_subject
            )
        )
        and (
            not human_requirements
            or (
                "HUMAN_REQUIREMENTS_RESOLVED" in record.body
                and {
                    str(getattr(item, "requirement_id", ""))
                    for item in human_requirements
                }.issubset(set(record.metadata.surfaced_reviewer_requirement_ids))
            )
        )
    }


def _prior_item_ledger_signature(items: Sequence[UnresolvedReviewItem]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            item.item_id,
            item.reviewer,
            item.source_round,
            item.text,
            item.status,
            item.source_status,
            item.notes,
            item.fix_scope,
            item.resolution_owners,
            item.owner_states,
            item.owner_evidence,
            item.owner_dispositions,
            item.authority,
            item.obligation_kind,
            item.lifecycle,
            item.failed_head_sha,
            item.candidate_head_sha,
            item.obligation_identity,
        )
        for item in items
    )


def _select_current_round_records(
    records: Sequence[PostedRoundRecord],
    *,
    subject: str,
) -> ResumedRoundSelection | None:
    subject_records = [record for record in records if record.metadata.subject == subject]
    if not subject_records:
        return None
    anchor_record = subject_records[-1]
    anchor_metadata = anchor_record.metadata
    prior_items_signature = _prior_item_ledger_signature(anchor_metadata.prior_items)
    current_round_records = tuple(
        record
        for record in subject_records
        if record.metadata.round_number == anchor_metadata.round_number
        and _prior_item_ledger_signature(record.metadata.prior_items) == prior_items_signature
    )
    latest_coder_record = next(
        (
            record
            for record in reversed(current_round_records)
            if record.metadata.role == "coder"
        ),
        None,
    )
    if latest_coder_record is not None:
        current_round_records = tuple(
            record for record in current_round_records if record.index >= latest_coder_record.index
        )
    return ResumedRoundSelection(
        anchor_record=anchor_record,
        current_round_records=current_round_records,
    )


def _max_unresolved_item_number_from_records(records: Sequence[PostedRoundRecord]) -> int:
    max_number = 0
    for record in records:
        for item in (*record.metadata.prior_items, *record.metadata.new_items):
            match = re.fullmatch(r"item-(\d+)", item.item_id)
            if match:
                max_number = max(max_number, int(match.group(1)))
    return max_number


def _active_pr_items(items: Sequence[UnresolvedReviewItem]) -> list[UnresolvedReviewItem]:
    return [item for item in items if item.status in {"blocking", "same-pr"}]


def _latest_qualification_checkpoint(
    records: Sequence[PostedRoundRecord], *, head_sha: str
) -> QualificationCheckpoint | None:
    """Return the newest checkpoint bound to the live head.

    An invalid decoded checkpoint is intentionally returned as a value with
    ``valid=False``.  Callers can then force full reconstruction instead of
    treating malformed persistence as an absent, safe-to-merge state.
    """
    for record in reversed(records):
        checkpoint = record.metadata.qualification_checkpoint
        if checkpoint is not None and record.metadata.subject == head_sha:
            return checkpoint
    return None


def _latest_qualification_checkpoint_record(
    records: Sequence[PostedRoundRecord], *, head_sha: str
) -> PostedRoundRecord | None:
    """Return the metadata record that anchors the latest qualification checkpoint.

    Qualification checkpoints are deliberately posted as summary records before
    a CI dispatch or coder handoff.  A process interruption can therefore leave
    a current-head transcript with no coder/reviewer record for the next round.
    Keeping the record, rather than only its decoded value, lets PR recovery use
    that summary as a durable round anchor.
    """
    for record in reversed(records):
        if (
            record.metadata.subject == head_sha
            and record.metadata.qualification_checkpoint is not None
        ):
            return record
    return None


def _append_active_pr_new_items(
    active_items: list[UnresolvedReviewItem],
    records: Sequence[PostedRoundRecord],
) -> None:
    seen_item_ids = {item.item_id for item in active_items}
    for record in records:
        for item in record.metadata.new_items:
            if item.status not in {"blocking", "same-pr"} or item.item_id in seen_item_ids:
                continue
            active_items.append(item)
            seen_item_ids.add(item.item_id)


def _aggregate_record_dispositions(
    records: Sequence[PostedRoundRecord],
) -> dict[str, list[ReviewItemDisposition]]:
    dispositions_by_item: dict[str, list[ReviewItemDisposition]] = {}
    for record in records:
        for disposition in record.metadata.dispositions:
            dispositions_by_item.setdefault(disposition.item_id, []).append(disposition)
    return dispositions_by_item


def _recover_unrecorded_pr_head_advance(
    records: Sequence[PostedRoundRecord],
    *,
    head_sha: str,
    reconciliation_mode: str = "aggregate",
) -> ResumedReviewRound | None:
    prior_records = [record for record in records if record.metadata.subject != head_sha]
    if not prior_records:
        return None
    latest_prior_subject = prior_records[-1].metadata.subject
    selection = _select_current_round_records(records, subject=latest_prior_subject)
    if selection is None:
        return None

    current_round_records = selection.current_round_records
    anchor_metadata = selection.anchor_record.metadata
    latest_coder_record = next(
        (
            record
            for record in reversed(current_round_records)
            if record.metadata.role == "coder"
        ),
        None,
    )
    reviewer_records_after_coder = tuple(
        record
        for record in current_round_records
        if record.metadata.role == "reviewer"
        and (latest_coder_record is None or record.index > latest_coder_record.index)
    )
    new_item_records_after_coder = tuple(
        record
        for record in current_round_records
        if record.metadata.role in {"reviewer", "summary"}
        and (latest_coder_record is None or record.index > latest_coder_record.index)
    )
    all_reviewer_records = tuple(
        record for record in current_round_records if record.metadata.role == "reviewer"
    )
    all_new_item_records = tuple(
        record
        for record in current_round_records
        if record.metadata.role in {"reviewer", "summary"}
    )

    if latest_coder_record is not None and not reviewer_records_after_coder:
        round_number = latest_coder_record.metadata.round_number
        recovered_items = _active_pr_items(latest_coder_record.metadata.prior_items)
        coder_output = latest_coder_record.metadata.raw_structured_coder_response or latest_coder_record.body
        compact_prior_summaries = latest_coder_record.metadata.compact_prior_summaries
    elif latest_coder_record is not None:
        round_number = latest_coder_record.metadata.round_number
        recovered_items, _future_items = _apply_unresolved_item_dispositions(
            latest_coder_record.metadata.prior_items,
            _aggregate_record_dispositions(reviewer_records_after_coder),
            retain_future=False,
            reconciliation_mode=reconciliation_mode,
        )
        recovered_items = _active_pr_items(recovered_items)
        _append_active_pr_new_items(recovered_items, new_item_records_after_coder)
        coder_output = latest_coder_record.metadata.raw_structured_coder_response or latest_coder_record.body
        compact_prior_summaries = latest_coder_record.metadata.compact_prior_summaries
    elif all_reviewer_records:
        round_number = anchor_metadata.round_number
        recovered_items, _future_items = _apply_unresolved_item_dispositions(
            anchor_metadata.prior_items,
            _aggregate_record_dispositions(all_reviewer_records),
            retain_future=False,
            reconciliation_mode=reconciliation_mode,
        )
        recovered_items = _active_pr_items(recovered_items)
        _append_active_pr_new_items(recovered_items, all_new_item_records)
        coder_output = None
        compact_prior_summaries = ()
    else:
        return None

    if not recovered_items:
        return None

    return ResumedReviewRound(
        round_number=round_number,
        prior_items=tuple(recovered_items),
        coder_output=coder_output,
        completed_reviews=(),
        next_unresolved_item_number=_max_unresolved_item_number_from_records(records) + 1,
        ledger_may_be_incomplete=True,
        compact_prior_summaries=compact_prior_summaries,
        unrecorded_head_advance=True,
        coder_metadata=latest_coder_record.metadata if latest_coder_record else None,
        local_test_evidence=(
            latest_coder_record.metadata.local_test_evidence
            if latest_coder_record is not None
            else None
        ),
        qualification_checkpoint=_latest_qualification_checkpoint(records, head_sha=head_sha),
    )


def _latest_prior_pr_subject_is_coherent(records: Sequence[PostedRoundRecord], *, head_sha: str) -> bool:
    prior_records = [record for record in records if record.metadata.subject != head_sha]
    if not prior_records:
        return True
    latest_prior_subject = prior_records[-1].metadata.subject
    selection = _select_current_round_records(records, subject=latest_prior_subject)
    if selection is None:
        return False
    return any(
        record.metadata.role in {"coder", "reviewer"}
        for record in selection.current_round_records
    )


def _plan_subject(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _approved_plan_hash(text: str) -> str:
    # Keep this local to avoid coupling the transport layer to decomposition.
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def _plan_declarations(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract bounded scope/deferred declarations without rewriting plan text."""
    sections: dict[str, list[str]] = {"scope": [], "deferred": []}
    active: str | None = None
    for line in text.splitlines():
        heading = re.match(r"^\s{0,3}#{2,6}\s+(.+?)\s*$", line)
        if heading:
            title = heading.group(1).strip().casefold()
            if any(token in title for token in ("deferred", "future", "out of scope", "non-goal")):
                active = "deferred"
            elif any(token in title for token in ("scope", "in scope", "current work")):
                active = "scope"
            else:
                active = None
            continue
        if active is not None and line.strip():
            sections[active].append(line.strip())
    return tuple(sections["scope"]), tuple(sections["deferred"])


def _approved_matrix_channel(
    *,
    contract_version: int | None,
    payload: object,
    changes_payload: object,
    identity: str | None,
    source_locator: str | None,
    diagnostic: str | None = None,
    boundary_digest: str | None = None,
    canonical_text: str | None = None,
) -> dict[str, object]:
    if contract_version is None and payload is None and changes_payload in (None, (), []):
        return {
            "risk_test_matrix_availability": "not-planned",
            "risk_test_matrix_diagnostic": None,
            "risk_test_matrix_contract_version": None,
            "risk_test_matrix_identity": None,
            "risk_test_matrix_payload": None,
            "risk_test_matrix_changes_payload": (),
            "risk_test_matrix_source_locator": source_locator,
            "risk_test_matrix_boundary_digest": boundary_digest,
        }
    if diagnostic:
        return {
            "risk_test_matrix_availability": "unavailable",
            "risk_test_matrix_diagnostic": diagnostic,
            "risk_test_matrix_contract_version": contract_version,
            "risk_test_matrix_identity": None,
            "risk_test_matrix_payload": None,
            "risk_test_matrix_changes_payload": (),
            "risk_test_matrix_source_locator": source_locator,
            "risk_test_matrix_boundary_digest": boundary_digest,
        }
    try:
        if contract_version != 1:
            raise ValueError("unsupported risk matrix contract version")
        matrix = parse_risk_test_matrix(payload, context="approved risk_test_matrix")
        changes = parse_risk_test_matrix_changes(
            changes_payload if changes_payload is not None else (),
            context="approved risk_test_matrix_changes",
        )
        actual_identity = risk_test_matrix_identity(matrix, changes, contract_version=contract_version)
        if not identity or identity != actual_identity:
            raise ValueError("risk matrix identity mismatch")
        if boundary_digest != actual_identity:
            raise ValueError("risk matrix payload-bound section boundary mismatch")
        if canonical_text and not _canonical_matrix_boundary_matches(
            canonical_text, actual_identity
        ):
            raise ValueError("canonical plan risk matrix section boundary mismatch")
        return {
            "risk_test_matrix_availability": "available",
            "risk_test_matrix_diagnostic": None,
            "risk_test_matrix_contract_version": contract_version,
            "risk_test_matrix_identity": actual_identity,
            "risk_test_matrix_payload": matrix.to_payload(),
            "risk_test_matrix_changes_payload": tuple(change.to_payload() for change in changes),
            "risk_test_matrix_source_locator": source_locator,
            "risk_test_matrix_boundary_digest": boundary_digest,
        }
    except (AgentLoopError, TypeError, ValueError) as exc:
        return {
            "risk_test_matrix_availability": "unavailable",
            "risk_test_matrix_diagnostic": f"Matrix unavailable: {exc}",
            "risk_test_matrix_contract_version": contract_version,
            "risk_test_matrix_identity": None,
            "risk_test_matrix_payload": None,
            "risk_test_matrix_changes_payload": (),
            "risk_test_matrix_source_locator": source_locator,
            "risk_test_matrix_boundary_digest": boundary_digest,
        }


def make_approved_plan_context(
    text: str | None,
    *,
    source_locator: str | None = None,
    expected_hash: str | None = None,
    expected_subject: str | None = None,
    risk_test_matrix_contract_version: int | None = None,
    risk_test_matrix_payload: object = None,
    risk_test_matrix_changes_payload: object = None,
    risk_test_matrix_identity: str | None = None,
    risk_test_matrix_source_locator: str | None = None,
    risk_test_matrix_diagnostic: str | None = None,
    risk_test_matrix_boundary_digest: str | None = None,
) -> ApprovedPlanContext:
    """Build and validate a plan context from raw canonical text.

    Hash and subject validation happen before any historical-text sanitizing or
    prompt truncation.  Callers can therefore safely pass the returned context
    through compact and full prompt builders.
    """
    matrix_fields = _approved_matrix_channel(
        contract_version=risk_test_matrix_contract_version,
        payload=risk_test_matrix_payload,
        changes_payload=risk_test_matrix_changes_payload,
        identity=risk_test_matrix_identity,
        source_locator=risk_test_matrix_source_locator or source_locator,
        diagnostic=risk_test_matrix_diagnostic,
        boundary_digest=risk_test_matrix_boundary_digest,
        canonical_text=text,
    )
    if (
        matrix_fields["risk_test_matrix_availability"] == "not-planned"
        and text
        and not risk_test_matrix_contract_version
    ):
        marker = RISK_TEST_MATRIX_MARKER_RE.search(text)
        if marker is not None:
            try:
                marker_payload = decode_risk_test_matrix_marker(marker.group("payload"))
                matrix_fields = _approved_matrix_channel(
                    contract_version=int(marker_payload["contract_version"]),
                    # Keep the marker's validated JSON-compatible payloads
                    # intact.  The channel parser accepts raw mappings; a
                    # parsed RiskTestMatrixChange is intentionally not a
                    # mapping and would make marker-only recovery fail for
                    # the normal non-empty draft audit case.
                    payload=marker_payload["matrix"],
                    changes_payload=marker_payload["changes"],
                    identity=str(marker_payload["identity"]),
                    source_locator=risk_test_matrix_source_locator or source_locator,
                    boundary_digest=str(marker_payload["identity"]),
                    canonical_text=text,
                )
            except (AgentLoopError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                matrix_fields = _approved_matrix_channel(
                    contract_version=1,
                    payload=None,
                    changes_payload=None,
                    identity=None,
                    source_locator=risk_test_matrix_source_locator or source_locator,
                    diagnostic=f"Matrix unavailable: {exc}",
                )
    if not text or not text.strip():
        if matrix_fields["risk_test_matrix_availability"] == "available":
            return ApprovedPlanContext(
                plan_hash=expected_hash,
                plan_subject=expected_subject,
                source_locator=source_locator,
                availability="available",
                has_matching_candidate=True,
                diagnostic="Canonical approved plan prose omitted; authenticated structured matrix remains available.",
                **matrix_fields,
            )
        return ApprovedPlanContext(
            plan_hash=expected_hash,
            plan_subject=expected_subject,
            source_locator=source_locator,
            availability="unavailable",
            diagnostic="The approved plan text is unavailable; recover the canonical plan record before reviewing.",
            **matrix_fields,
        )
    raw = text.strip()
    actual_hash = _approved_plan_hash(raw)
    actual_subject = _plan_subject(raw)
    if expected_hash is not None and actual_hash != expected_hash:
        return ApprovedPlanContext(
            canonical_text=raw,
            plan_hash=actual_hash,
            plan_subject=actual_subject,
            source_locator=source_locator,
            availability="mismatched",
            has_matching_candidate=False,
            diagnostic=(
                f"Recovered plan hash {actual_hash} does not match handoff hash {expected_hash}."
            ),
            **matrix_fields,
        )
    if expected_subject is not None and actual_subject != expected_subject:
        return ApprovedPlanContext(
            canonical_text=raw,
            plan_hash=actual_hash,
            plan_subject=actual_subject,
            source_locator=source_locator,
            availability="mismatched",
            has_matching_candidate=True,
            diagnostic=(
                f"Recovered plan subject {actual_subject} does not match handoff subject {expected_subject}."
            ),
            **matrix_fields,
        )
    scope, deferred = _plan_declarations(raw)
    return ApprovedPlanContext(
        canonical_text=raw,
        plan_hash=actual_hash,
        plan_subject=actual_subject,
        source_locator=source_locator,
        scope=scope,
        deferred_work=deferred,
        availability="available",
        has_matching_candidate=True,
        **matrix_fields,
    )


def _legacy_freeform_plan_candidates(record: PostedRoundRecord) -> tuple[str, ...]:
    """Return raw-text candidates for pre-canonical free-form plan comments.

    Older plan comments stored only the rendered body.  The renderer may have
    replaced a trailing provider-only signature with a model-qualified one, so
    migration recovery considers the visible body, the body without that
    generated signature, and the provider signatures that could have been
    replaced.  The caller still validates the short hash and full subject
    against the durable handoff before accepting any candidate.
    """
    visible = record.body.strip()
    if not visible:
        return ()
    candidates: list[str] = [visible]
    lines = visible.splitlines()
    tail = len(lines)
    while tail > 0 and (
        not lines[tail - 1].strip() or HTML_COMMENT_RE.match(lines[tail - 1])
    ):
        tail -= 1
    if tail <= 0 or not SIGNATURE_RE.match(lines[tail - 1]):
        return tuple(dict.fromkeys(candidates))

    signature_index = tail - 1
    suffix = lines[tail:]
    without_signature = "\n".join([*lines[:signature_index], *suffix]).strip()
    if without_signature:
        candidates.append(without_signature)

    for agent in ("claude", "codex", "gemini", "antigravity"):
        if record.metadata.agent != agent_display_name(agent):
            continue
        signatures = {
            agent_signature(agent),
            agent_signature(agent, model_used=record.metadata.model_used),
        }
        if record.metadata.configured_model:
            configured_label = record.metadata.configured_model
            if record.metadata.configured_effort:
                configured_label = f"{configured_label} ({record.metadata.configured_effort})"
            signatures.add(agent_signature(agent, model_used=configured_label))
        for signature in signatures:
            reconstructed = "\n".join(
                [*lines[:signature_index], f"-- {signature}", *suffix]
            ).strip()
            if reconstructed:
                candidates.append(reconstructed)
        break
    spaced_candidates: list[str] = []
    for candidate in candidates:
        spaced_candidates.append(candidate)
        candidate_lines = candidate.splitlines()
        with_comment_spacing: list[str] = []
        for line in candidate_lines:
            if (
                HTML_COMMENT_RE.match(line)
                and with_comment_spacing
                and with_comment_spacing[-1].strip()
            ):
                with_comment_spacing.append("")
            with_comment_spacing.append(line)
        spaced = "\n".join(with_comment_spacing).strip()
        if spaced:
            spaced_candidates.append(spaced)
    return tuple(dict.fromkeys(spaced_candidates))


def recover_approved_plan_context(
    comments: Sequence[object],
    *,
    expected_hash: str,
    expected_subject: str | None = None,
    expected_matrix_identity: str | None = None,
) -> ApprovedPlanContext:
    """Recover exactly the plan selected by a durable implementation handoff."""
    records = _extract_round_metadata_records(comments, flow="plan")
    candidates: list[tuple[int, str]] = []
    observed_hashes: set[str] = set()
    observed_subjects: set[str] = set()
    subject_rejected = False
    for record in records:
        raw = record.metadata.canonical_plan
        if raw is None:
            # Legacy freeform plan records have no canonical sidecar. Their
            # visible body is only a safe fallback when its raw hash and plan
            # subject match. normalize_freeform_signature may have replaced a
            # provider-only trailing signature, so try the body with that
            # generated signature removed/reconstructed as well.
            raw = record.metadata.raw_structured_coder_response
        raw_candidates = (raw,) if raw is not None else _legacy_freeform_plan_candidates(record)
        # A canonical sidecar is already authoritative; only the migration
        # fallback needs the legacy metadata subject as a second identity
        # check. Existing canonical records may use arbitrary subjects from
        # older metadata versions and must remain recoverable by handoff hash.
        required_subject = (
            expected_subject
            if raw is not None
            else expected_subject or record.metadata.subject
        )
        for candidate in raw_candidates:
            if candidate is None or not candidate.strip():
                continue
            candidate = candidate.strip()
            actual_hash = _approved_plan_hash(candidate)
            observed_hashes.add(actual_hash)
            actual_subject = _plan_subject(candidate)
            observed_subjects.add(actual_subject)
            if actual_hash != expected_hash:
                continue
            if required_subject and actual_subject != required_subject:
                subject_rejected = True
                continue
            candidates.append((record.index, candidate))
    if not candidates:
        available = ", ".join(sorted(observed_hashes)) or "none"
        subject_detail = ""
        if subject_rejected:
            subject_detail = (
                " The handoff hash was observed, but no candidate also matched the "
                f"expected plan subject; recovered subjects: "
                f"{', '.join(sorted(observed_subjects)) or 'none'}."
            )
        return ApprovedPlanContext(
            plan_hash=expected_hash,
            plan_subject=expected_subject,
            availability="mismatched" if observed_hashes else "unavailable",
            has_matching_candidate=False,
            diagnostic=(
                f"No canonical approved plan matches handoff hash {expected_hash}; "
                f"recovered plan hashes: {available}.{subject_detail}"
            ),
        )
    unique_texts = {raw for _index, raw in candidates}
    if len(unique_texts) != 1:
        return ApprovedPlanContext(
            plan_hash=expected_hash,
            plan_subject=expected_subject,
            availability="mismatched",
            has_matching_candidate=True,
            diagnostic=(
                f"Multiple divergent canonical plan records match handoff hash {expected_hash}."
            ),
        )
    matching_indices = {index for index, _raw in candidates}
    matching_records = [record for record in records if record.index in matching_indices]
    matrix_presence = [
        _risk_test_matrix_metadata_present(record.metadata)
        for record in matching_records
    ]
    matching_matrix_identities = {
        record.metadata.risk_test_matrix_identity
        for record, present in zip(matching_records, matrix_presence)
        if present
    }
    matching_matrix_identity_missing = any(
        present and record.metadata.risk_test_matrix_identity is None
        for record, present in zip(matching_records, matrix_presence)
    )
    matrix_conflict = bool(matching_matrix_identities) and (
        len(matching_matrix_identities) > 1
        or not all(matrix_presence)
        or matching_matrix_identity_missing
    )
    index, raw = candidates[-1]
    matching_record = next((record for record in reversed(records) if record.index == index), None)
    matrix_fields: dict[str, object] = {}
    if matching_record is not None:
        matrix_fields = {
            "risk_test_matrix_contract_version": matching_record.metadata.risk_test_matrix_contract_version,
            "risk_test_matrix_payload": matching_record.metadata.risk_test_matrix_payload,
            "risk_test_matrix_changes_payload": matching_record.metadata.risk_test_matrix_changes_payload,
            "risk_test_matrix_identity": matching_record.metadata.risk_test_matrix_identity,
            "risk_test_matrix_boundary_digest": matching_record.metadata.risk_test_matrix_boundary_digest,
            "risk_test_matrix_source_locator": f"issue comment index {index}",
            "risk_test_matrix_diagnostic": matching_record.metadata.risk_test_matrix_diagnostic,
        }
        if matrix_conflict:
            # Canonical plan identity is still unambiguous. Close only the
            # semantic matrix channel when matching records disagree; do not
            # turn a matrix-only conflict into a whole-plan mismatch.
            matrix_fields.update(
                {
                    "risk_test_matrix_diagnostic": (
                        f"Multiple divergent risk matrix records match approved plan {expected_hash}."
                    ),
                    "risk_test_matrix_payload": None,
                    "risk_test_matrix_changes_payload": (),
                    "risk_test_matrix_identity": None,
                    "risk_test_matrix_boundary_digest": None,
                }
            )
        if expected_matrix_identity is not None and matching_record.metadata.risk_test_matrix_identity != expected_matrix_identity:
            matrix_fields["risk_test_matrix_diagnostic"] = (
                f"Recovered matrix identity does not match handoff identity {expected_matrix_identity}."
            )
            matrix_fields["risk_test_matrix_payload"] = None
            matrix_fields["risk_test_matrix_changes_payload"] = ()
            matrix_fields["risk_test_matrix_identity"] = None
    return make_approved_plan_context(
        raw,
        source_locator=f"issue comment index {index}",
        expected_hash=expected_hash,
        expected_subject=expected_subject,
        **matrix_fields,
    )


def _resume_pr_round(
    comments: Sequence[object],
    *,
    head_sha: str | None,
    configured_reviewers: Sequence[AgentName],
    reconciliation_mode: str = "aggregate",
) -> ResumedReviewRound | None:
    if not head_sha:
        return None
    records = _extract_round_metadata_records(comments, flow="pr")
    if not records:
        return None
    selection = _select_current_round_records(records, subject=head_sha)
    if selection is None:
        recovered = _recover_unrecorded_pr_head_advance(
            records, head_sha=head_sha, reconciliation_mode=reconciliation_mode
        )
        if recovered is not None:
            return recovered
        if _latest_prior_pr_subject_is_coherent(records, head_sha=head_sha):
            return None
        latest_prior_subject = records[-1].metadata.subject
        raise AgentLoopError(
            "PR head advanced without a recorded coder follow-up and the metadata-backed "
            "handoff could not be recovered safely. "
            f"Current head: {head_sha}. Latest recorded metadata subject: {latest_prior_subject}. "
            "Rerun after posting a valid structured coder follow-up for the current head "
            "or repair the metadata-backed handoff."
        )
    current_round_records = selection.current_round_records
    anchor_metadata = selection.anchor_record.metadata
    checkpoint_record = _latest_qualification_checkpoint_record(
        records, head_sha=head_sha
    )
    latest_coder_record = next(
        (
            record
            for record in reversed(current_round_records)
            if record.metadata.role == "coder"
        ),
        None,
    )
    reviewer_records: dict[str, PostedRoundRecord] = {}
    configured_reviewer_names = {agent_display_name(agent) for agent in configured_reviewers}
    for record in current_round_records:
        metadata = record.metadata
        if metadata.role != "reviewer" or metadata.agent not in configured_reviewer_names:
            continue
        reviewer_records[metadata.agent] = record
    if latest_coder_record is None and not reviewer_records:
        if checkpoint_record is None:
            return None
        # A qualification checkpoint is a deliberate handoff boundary.  It is
        # valid even when it is the only record for the current round: recovery
        # must retain its ledger, round number, and budget instead of treating
        # the summary-only record as an incomplete review transcript.
        anchor_metadata = checkpoint_record.metadata
        current_round_records = (checkpoint_record,)
    prior_items = anchor_metadata.prior_items
    round_number = anchor_metadata.round_number
    ledger_may_be_incomplete = (
        len(anchor_metadata.prior_items) == 0
        and any(
            record.metadata.new_items
            for record in records
            if record.metadata.subject == anchor_metadata.subject
            and record.metadata.round_number < anchor_metadata.round_number
        )
    )
    return ResumedReviewRound(
        round_number=round_number,
        prior_items=prior_items,
        coder_output=(
            latest_coder_record.metadata.raw_structured_coder_response
            or latest_coder_record.body
            if latest_coder_record is not None
            else None
        ),
        coder_metadata=latest_coder_record.metadata if latest_coder_record else None,
        completed_reviews=tuple(reviewer_records[agent_display_name(agent)] for agent in configured_reviewers if agent_display_name(agent) in reviewer_records),
        next_unresolved_item_number=_max_unresolved_item_number_from_records(
            [record for record in records if record.metadata.subject == head_sha]
        )
        + 1,
        ledger_may_be_incomplete=ledger_may_be_incomplete,
        compact_prior_summaries=(
            latest_coder_record.metadata.compact_prior_summaries
            if latest_coder_record is not None
            else ()
        ),
        reconciled=any(record.metadata.role == "summary" for record in current_round_records),
        local_test_evidence=(
            latest_coder_record.metadata.local_test_evidence
            if latest_coder_record is not None
            else None
        ),
        qualification_checkpoint=_latest_qualification_checkpoint(records, head_sha=head_sha),
    )


def _resume_plan_round(
    comments: Sequence[object],
    *,
    configured_reviewers: Sequence[AgentName],
) -> tuple[str, ResumedReviewRound] | None:
    records = _extract_round_metadata_records(comments, flow="plan")
    if not records:
        return None
    latest_coder_record = next((record for record in reversed(records) if record.metadata.role == "coder"), None)
    if latest_coder_record is None:
        return None
    selection = _select_current_round_records(records, subject=latest_coder_record.metadata.subject)
    if selection is None:
        return None
    current_round_records = selection.current_round_records
    anchor_metadata = selection.anchor_record.metadata
    current_plan = latest_coder_record.metadata.canonical_plan or latest_coder_record.body
    coder_output = latest_coder_record.metadata.raw_structured_coder_response or current_plan
    metadata_version = latest_coder_record.metadata.execution_strategy_contract_version
    matrix_metadata_version = latest_coder_record.metadata.risk_test_matrix_contract_version
    semantic_response_form = latest_coder_record.metadata.response_form
    semantic_sidecar_authoritative = semantic_response_form == "semantic-patch-v1"
    if semantic_response_form in {"fresh-plan-state", "legacy-full-state"}:
        # Fresh and legacy full-state publications seed the next semantic
        # revision.  Their authenticated sidecar must bind to the exact
        # canonical Markdown persisted with the round; a subject check alone
        # is insufficient because both the Markdown and subject can be
        # replaced while leaving the sidecar untouched.
        metadata = latest_coder_record.metadata
        if metadata.assembled_plan_sidecar is None or metadata.canonical_plan is None:
            raise AgentLoopError(
                "Authenticated full-state planning metadata is incomplete: "
                "authenticated assembled state and canonical Markdown are both required "
                "for restart."
            )
        try:
            sidecar = decode_assembled_plan_sidecar(metadata.assembled_plan_sidecar)
            if sidecar.response_form != semantic_response_form:
                raise AgentLoopError("full-state sidecar response form mismatch")
            if sidecar.round_number != metadata.round_number:
                raise AgentLoopError("full-state sidecar round mismatch")
            if metadata.aggregate_plan_identity != sidecar.aggregate_identity:
                raise AgentLoopError("full-state sidecar aggregate identity mismatch")
            if (
                sidecar.rendered_plan_identity is None
                or sidecar.rendered_plan_identity
                != rendered_plan_identity(metadata.canonical_plan)
            ):
                raise AgentLoopError(
                    "full-state sidecar rendered-plan identity does not match canonical Markdown"
                )
        except (AgentLoopError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise AgentLoopError(
                "Authenticated full-state planning state is missing or contradictory; "
                f"refusing to resume from an unbound canonical Markdown: {exc}"
            ) from exc
    if semantic_sidecar_authoritative:
        # Semantic rounds resume from the authenticated assembled sidecar. The
        # raw model patch is provenance only and is never reparsed into the
        # canonical plan used by prompts or reviewers.
        metadata = latest_coder_record.metadata
        if metadata.assembled_plan_sidecar is None or metadata.canonical_plan is None:
            raise AgentLoopError(
                "Semantic planning metadata is incomplete: authenticated assembled state "
                "and canonical Markdown are both required for restart."
            )
        try:
            sidecar = decode_assembled_plan_sidecar(metadata.assembled_plan_sidecar)
            if sidecar.response_form != "semantic-patch-v1":
                raise AgentLoopError("semantic sidecar response form mismatch")
            if sidecar.round_number != metadata.round_number:
                raise AgentLoopError("semantic sidecar round mismatch")
            if metadata.aggregate_plan_identity != sidecar.aggregate_identity:
                raise AgentLoopError("semantic sidecar aggregate identity mismatch")
            patch = parse_plan_revision_patch(metadata.raw_patch_provenance or {})
            if patch.base_round_number != metadata.base_round_number:
                raise AgentLoopError("semantic patch base round mismatch")
            if patch.base_state_identity != metadata.base_state_identity:
                raise AgentLoopError("semantic patch base identity mismatch")
            if sidecar.raw_patch != patch.to_payload():
                raise AgentLoopError("semantic sidecar patch provenance mismatch")
            if (
                sidecar.rendered_plan_identity is None
                or sidecar.rendered_plan_identity
                != rendered_plan_identity(metadata.canonical_plan)
            ):
                raise AgentLoopError(
                    "semantic sidecar rendered-plan identity does not match canonical Markdown"
                )
        except (AgentLoopError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise AgentLoopError(
                "Authenticated semantic planning state is missing or contradictory; "
                f"refusing to prompt from raw patch text: {exc}"
            ) from exc
        if _plan_subject(metadata.canonical_plan) != metadata.subject:
            raise AgentLoopError(
                "Semantic planning metadata subject does not match canonical Markdown."
            )
    if metadata_version == 1 and not semantic_sidecar_authoritative:
        # A generation-1 plan is identified by its canonical rendered text.
        # Never resume a record whose subject was computed from a different
        # representation (for example, raw host JSON versus rendered plan
        # markdown); doing so would make the next skill invocation fork the
        # same plan into a new round.
        if latest_coder_record.metadata.canonical_plan is None:
            raise AgentLoopError(
                "Generation-1 planning metadata has no canonical plan text; "
                "repair the handoff or start a new plan round."
            )
        if _plan_subject(current_plan) != latest_coder_record.metadata.subject:
            raise AgentLoopError(
                "Generation-1 planning metadata subject does not match its canonical plan; "
                "repair the handoff or start a new plan round."
            )
    if semantic_response_form != "semantic-patch-v1":
        all_bodies = tuple(
            body for comment in comments if isinstance((body := getattr(comment, "body", None)), str)
        )
        fresh_artifact = False
        fresh_matrix_artifact = False
        try:
            raw_payload, _ = json.JSONDecoder().raw_decode(coder_output.lstrip())
            fresh_artifact = (
                isinstance(raw_payload, dict)
                and (
                    "execution_strategy_contract_version" in raw_payload
                    or "execution_recommendation" in raw_payload
                )
            ) or "AGENT_EXECUTION_RECOMMENDATION" in latest_coder_record.body
            fresh_matrix_artifact = (
                isinstance(raw_payload, dict)
                and (
                    "risk_test_matrix_contract_version" in raw_payload
                    or "risk_test_matrix" in raw_payload
                )
            ) or RISK_TEST_MATRIX_MARKER_RE.search(latest_coder_record.body) is not None
        except (AttributeError, json.JSONDecodeError):
            fresh_artifact = "AGENT_EXECUTION_RECOMMENDATION" in latest_coder_record.body
            fresh_matrix_artifact = RISK_TEST_MATRIX_MARKER_RE.search(latest_coder_record.body) is not None
        if fresh_artifact and metadata_version != 1:
            raise AgentLoopError(
                "Fresh generation-1 planning data is present but its round metadata is "
                "missing or not generation 1; restart the planning handoff instead of "
                "downgrading it to legacy-undecided."
            )
        if fresh_matrix_artifact and matrix_metadata_version != 1:
            raise AgentLoopError(
                "Fresh generation-1 risk matrix data is present but its round metadata is "
                "missing or not generation 1; restart the planning handoff instead of "
                "downgrading it to legacy-undecided."
            )
    if metadata_version == 1 and semantic_response_form != "semantic-patch-v1":
        # A generation-1 round is never downgraded to legacy on resume.  The
        # raw response is the provenance source; the visible canonical plan is
        # intentionally markdown and cannot substitute for it.
        from .protocol import (
            validate_structured_plan_revision,
            validate_structured_plan_state,
        )

        try:
            raw_payload, _ = json.JSONDecoder().raw_decode(coder_output.lstrip())
            response_kind = raw_payload.get("kind") if isinstance(raw_payload, dict) else None
            if response_kind == "plan_revision":
                parsed_revision = validate_structured_plan_revision(
                    coder_output,
                    require_execution_strategy_contract=1,
                    require_risk_test_matrix_contract=(1 if matrix_metadata_version == 1 else 0),
                )
                if parsed_revision is None:
                    raise AgentLoopError(
                        "Generation-1 planning metadata has no recoverable structured response."
                    )
                parsed_recommendation = parsed_revision.execution_recommendation
            elif response_kind == "plan_state":
                parsed = validate_structured_plan_state(
                    coder_output,
                    require_execution_strategy_contract=1,
                    require_risk_test_matrix_contract=(1 if matrix_metadata_version == 1 else 0),
                )
                if parsed is None:
                    raise AgentLoopError(
                        "Generation-1 planning metadata has no recoverable structured response."
                    )
                parsed_recommendation = parsed.execution_recommendation
            else:
                raise AgentLoopError(
                    "Generation-1 planning metadata has an unknown structured response kind."
                )
            if parsed_recommendation is None:
                raise AgentLoopError(
                    "Generation-1 planning metadata has no execution recommendation."
                )

            # The canonical plan is the public, hashed rendering.  Recover the
            # recommendation sidecar from it (including transport sidecars) and
            # compare it with both the raw response and durable metadata.  This
            # prevents a restart from accepting a changed recommendation that
            # retained only the old short topology summary.
            marker_match = EXECUTION_RECOMMENDATION_MARKER_RE.search(current_plan)
            if marker_match is None:
                marker_match = EXECUTION_RECOMMENDATION_MARKER_RE.search(
                    latest_coder_record.body
                )
            if marker_match is None:
                raise AgentLoopError(
                    "Generation-1 planning metadata has no canonical execution recommendation sidecar."
                )
            from .protocol import parse_execution_recommendation_payload

            sidecar_payload = decode_execution_recommendation_marker(
                marker_match.group("payload"), bodies=all_bodies
            )
            sidecar_recommendation = parse_execution_recommendation_payload(
                sidecar_payload, context="canonical execution_recommendation"
            )
            if (
                not isinstance(latest_coder_record.metadata.execution_strategy_identity, dict)
                or latest_coder_record.metadata.execution_strategy_identity
                != parsed_recommendation.identity()
                or sidecar_recommendation.identity() != parsed_recommendation.identity()
            ):
                raise AgentLoopError(
                    "Generation-1 planning metadata has a missing or mismatched complete "
                    "strategy/topology identity across raw response, canonical sidecar, and metadata."
                )
            if matrix_metadata_version == 1:
                parsed_matrix = (
                    parsed_revision.risk_test_matrix
                    if response_kind == "plan_revision"
                    else parsed.risk_test_matrix
                )
                parsed_changes = (
                    parsed_revision.risk_test_matrix_changes
                    if response_kind == "plan_revision"
                    else parsed.risk_test_matrix_changes
                )
                if parsed_matrix is None or not latest_coder_record.metadata.risk_test_matrix_identity:
                    raise AgentLoopError("Generation-1 planning metadata has no complete risk matrix identity.")
                if risk_test_matrix_identity(parsed_matrix, parsed_changes) != latest_coder_record.metadata.risk_test_matrix_identity:
                    raise AgentLoopError("Generation-1 planning metadata has a mismatched risk matrix identity.")
                matrix_marker_match = RISK_TEST_MATRIX_MARKER_RE.search(current_plan)
                if matrix_marker_match is None:
                    raise AgentLoopError("Generation-1 planning metadata has no canonical risk matrix sidecar.")
                matrix_marker = decode_risk_test_matrix_marker(matrix_marker_match.group("payload"))
                if matrix_marker["identity"] != latest_coder_record.metadata.risk_test_matrix_identity:
                    raise AgentLoopError("Generation-1 canonical risk matrix sidecar does not match metadata.")
        except (AgentLoopError, AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AgentLoopError(
                "Generation-1 planning round metadata is missing or has a malformed "
                "execution strategy contract; repair the handoff or start a new plan round."
            ) from exc
    ledger_may_be_incomplete = (
        len(anchor_metadata.prior_items) == 0
        and any(
            record.metadata.new_items
            for record in records
            if record.metadata.subject == anchor_metadata.subject
            and record.metadata.round_number < anchor_metadata.round_number
        )
    )
    reviewer_records: dict[str, PostedRoundRecord] = {}
    configured_reviewer_names = {agent_display_name(agent) for agent in configured_reviewers}
    for record in current_round_records:
        metadata = record.metadata
        if metadata.role != "reviewer" or metadata.agent not in configured_reviewer_names:
            continue
        reviewer_records[metadata.agent] = record
    settled_new_items: list[UnresolvedReviewItem] = []
    settled_item_ids: set[str] = set()
    for record in current_round_records:
        for item in record.metadata.new_items:
            if item.item_id in settled_item_ids:
                continue
            settled_item_ids.add(item.item_id)
            settled_new_items.append(item)
    return (
        current_plan,
        ResumedReviewRound(
            round_number=anchor_metadata.round_number,
            prior_items=anchor_metadata.prior_items,
            coder_output=coder_output,
            completed_reviews=tuple(reviewer_records[agent_display_name(agent)] for agent in configured_reviewers if agent_display_name(agent) in reviewer_records),
            next_unresolved_item_number=_max_unresolved_item_number_from_records(
                [record for record in records if record.metadata.subject == anchor_metadata.subject]
            )
            + 1,
            ledger_may_be_incomplete=ledger_may_be_incomplete,
            compact_prior_summaries=latest_coder_record.metadata.compact_prior_summaries,
            reconciled=any(record.metadata.role == "summary" for record in current_round_records),
            coder_metadata=latest_coder_record.metadata,
            local_test_evidence=latest_coder_record.metadata.local_test_evidence,
            current_round_new_items=tuple(settled_new_items),
        ),
    )


@dataclass(frozen=True)
class ResumedDiscussState:
    done: bool
    round_history: tuple[tuple[ParsedDiscussResponse, ...], ...]
    next_round_number: int
    prior_round_agenda: tuple[str, ...] = ()
    prior_analyzer_agenda: ParsedDiscussAgenda | None = None
    prior_round_synthesis: ParsedDiscussRoundSynthesis | None = None
    final_synthesis: ParsedDiscussFinalSynthesis | None = None
    synthesis_provenance: dict | None = None
    in_progress_votes: dict[str, ParsedDiscussResponse] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.in_progress_votes is None:
            object.__setattr__(self, "in_progress_votes", {})


def _decode_analyzer_agenda(raw: str | None) -> ParsedDiscussAgenda | None:
    """Reparse a persisted analyzer agenda; corrupt/legacy payloads resume in plain mode."""
    if not raw:
        return None
    try:
        return parse_structured_discuss_agenda(raw)
    except AgentLoopError:
        return None


def _decode_round_synthesis(raw: str | None) -> ParsedDiscussRoundSynthesis | None:
    """Decode optional canonical synthesis without poisoning a valid resume."""
    if not raw or len(raw.encode("utf-8")) > 16_000:
        return None
    try:
        return parse_canonical_discuss_round_synthesis(raw)
    except (AgentLoopError, TypeError, ValueError):
        return None


def _decode_final_synthesis(raw: str | None) -> ParsedDiscussFinalSynthesis | None:
    """Decode an optional canonical final synthesis without poisoning resume."""
    if not raw or len(raw.encode("utf-8")) > 16_000:
        return None
    try:
        return parse_canonical_discuss_final_synthesis(raw)
    except (AgentLoopError, TypeError, ValueError):
        return None


def _decode_discuss_vote(
    record: PostedRoundRecord, *, round_number: int, reviewer_workdirs: Mapping[str, Path]
) -> ParsedDiscussResponse:
    text = record.metadata.raw_structured_coder_response or record.body
    if record.metadata.result_mode == "answer":
        try:
            vote = parse_structured_discuss_answer(
                text, reviewer=record.metadata.agent, round_number=round_number
            )
        except AgentLoopError:
            # Only persisted round metadata gets the explicitly opt-in legacy
            # decoder. Its exact-key validation still rejects malformed/mixed
            # payloads rather than hiding a corrupt transcript.
            vote = parse_legacy_structured_discuss_answer(
                text, reviewer=record.metadata.agent, round_number=round_number
            )
    else:
        vote = parse_structured_discuss_review(text, reviewer=record.metadata.agent, round_number=round_number)
    if vote is None:
        raise AgentLoopError(
            f"Could not decode discuss response metadata for {record.metadata.agent} "
            f"(round {round_number}); the posted comment's structured payload is missing "
            "or invalid."
        )
    workdir = reviewer_workdirs.get(record.metadata.agent)
    if workdir is None:
        raise AgentLoopError(
            f"Discuss round metadata references reviewer {record.metadata.agent!r} "
            "with no known assigned checkout (it is not in the currently configured "
            "reviewers); a persisted vote from it cannot be checkout-validated."
        )
    validate_checkout_inspected_evidence(vote.evidence_claims, assigned_workdir=workdir)
    return vote


def _resume_discuss_round(
    comments: Sequence[object],
    *,
    subject: str,
    configured_reviewers: Sequence[AgentName],
    reviewer_workdirs: Mapping[str, Path],
    result_mode: str = "triage",
) -> ResumedDiscussState | None:
    records = _extract_round_metadata_records(comments, flow="discuss")
    subject_records = [record for record in records if record.metadata.subject == subject]
    if not subject_records:
        return None
    stored_modes = {record.metadata.result_mode for record in subject_records}
    if stored_modes != {result_mode}:
        raise AgentLoopError(
            "Discuss transcript result mode conflicts with the requested mode; "
            "use the same --discuss-result-mode used to create the transcript."
        )
    configured_reviewer_names = [agent_display_name(agent) for agent in configured_reviewers]
    rounds: dict[int, list[PostedRoundRecord]] = {}
    for record in subject_records:
        rounds.setdefault(record.metadata.round_number, []).append(record)

    round_history: list[tuple[ParsedDiscussResponse, ...]] = []
    prior_round_agenda: tuple[str, ...] = ()
    prior_analyzer_agenda: ParsedDiscussAgenda | None = None
    prior_round_synthesis: ParsedDiscussRoundSynthesis | None = None
    synthesis_provenance: dict | None = None
    next_round_number = 1
    in_progress_votes: dict[str, ParsedDiscussResponse] = {}
    for round_number in sorted(rounds):
        round_records = rounds[round_number]
        summary_record = next(
            (record for record in round_records if record.metadata.role == "summary"), None
        )
        debater_records: dict[str, PostedRoundRecord] = {}
        for record in round_records:
            if record.metadata.role == "debater":
                debater_records[record.metadata.agent] = record
        if summary_record is not None:
            # Debaters recorded as failed in a partial round (#475) post no
            # comment; their summary metadata accounts for the gap.
            failed_by_name = dict(summary_record.metadata.failed_debaters)
            missing = [
                name
                for name in configured_reviewer_names
                if name not in debater_records and name not in failed_by_name
            ]
            if missing:
                raise AgentLoopError(
                    "Discuss round metadata is inconsistent: round "
                    f"{round_number} has a summary comment but is missing debater "
                    f"comments for: {', '.join(missing)}."
                )
            votes = tuple(
                _decode_discuss_vote(
                    debater_records[name], round_number=round_number, reviewer_workdirs=reviewer_workdirs
                )
                if name in debater_records
                else (
                    failed_discuss_answer_placeholder(name, failed_by_name[name])
                    if result_mode == "answer"
                    else failed_discuss_review_placeholder(name, failed_by_name[name])
                )
                for name in configured_reviewer_names
            )
            round_history.append(votes)
            prior_round_agenda = summary_record.metadata.agenda
            prior_analyzer_agenda = _decode_analyzer_agenda(summary_record.metadata.analyzer_response)
            prior_round_synthesis = _decode_round_synthesis(summary_record.metadata.round_synthesis)
            final_synthesis = _decode_final_synthesis(summary_record.metadata.final_synthesis)
            synthesis_provenance = summary_record.metadata.synthesis_provenance
            if summary_record.metadata.is_final:
                return ResumedDiscussState(
                    done=True,
                    round_history=tuple(round_history),
                    next_round_number=round_number,
                    prior_round_agenda=prior_round_agenda,
                    prior_analyzer_agenda=prior_analyzer_agenda,
                    prior_round_synthesis=prior_round_synthesis,
                    final_synthesis=final_synthesis,
                    synthesis_provenance=synthesis_provenance,
                )
            next_round_number = round_number + 1
        else:
            next_round_number = round_number
            for name, record in debater_records.items():
                in_progress_votes[name] = _decode_discuss_vote(
                    record, round_number=round_number, reviewer_workdirs=reviewer_workdirs
                )
            break
    return ResumedDiscussState(
        done=False,
        round_history=tuple(round_history),
        next_round_number=next_round_number,
        prior_round_agenda=prior_round_agenda,
        prior_analyzer_agenda=prior_analyzer_agenda,
        prior_round_synthesis=prior_round_synthesis,
        final_synthesis=None,
        synthesis_provenance=synthesis_provenance,
        in_progress_votes=in_progress_votes,
    )
