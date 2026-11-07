"""Round metadata value types, codecs and diagnostic transport."""

from __future__ import annotations

import hashlib
import json
import logging
import contextvars
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from .review_scope import encode_review_scope, decode_review_scope
from .errors import AgentLoopError, IssueImplementationConflictError
from .round_state_scheduler import (
    _decode_execution_mode,
    _decode_scheduler_fields,
    _SCHEDULER_AUXILIARY_KEYS,
)
from .round_transport import (
    MAX_PLAN_VALIDATION_DIAGNOSTIC_CHARS,
    PLAN_VALIDATION_DIAGNOSTIC_MARKER_RE,
    ROUND_RESUME_MARKER_RE,
    decode_mapping,
    encode_mapping,
    hydrate_mapping,
)
from .comment_rendering import (
    RISK_TEST_MATRIX_MARKER_RE,
    decode_risk_test_matrix_marker,
    render_parse_degradations_section,
    render_test_observation_degradations_section,
)
from .local_test_evidence import canonicalize_bounded_evidence
from .protocol_markers import (
    ISSUE_COMMENT_SURFACE,
    TrustedBody,
    protocol_record_label,
    sanitize_historical_text,
    scan_reserved_markers,
    strip_known_host_footer,
)
from .protocol_risk import (
    SignedEvidenceReference,
    parse_risk_test_matrix,
    parse_risk_test_matrix_changes,
    risk_test_matrix_identity,
)
from .protocol import (
    ReviewItemDisposition,
    ReviewSubItem,
    SUB_ITEM_STATUSES,
    UnresolvedReviewItem,
    EVIDENCE_OBLIGATION_KIND,
    MACHINE_OBLIGATION_FIELDS,
    UNKNOWN_MACHINE_AUTHORITY,
    PLAN_REVIEW_PROVENANCE_VALUES,
    parse_plan_revision_patch,
    StructuredCoderFollowup,
    StructuredIssueImplementation,
    parse_historical_structured_coder_followup,
    parse_historical_structured_issue_implementation,
)
from .protocol_common import (
    HTML_COMMENT_RE,
    SIGNATURE_RE,
    ParseDegradation,
    parse_degradation_payload,
)
from .plan_assembly import decode_assembled_plan_sidecar
from .review_scheduling import FORCE_FULL_SOURCES

from .evidence_stall import stall_snapshot_is_consistent
from .evidence_conversion import EvidenceConversionAudit, decode_capture_classification


# Preserve the historical diagnostic logger while changing code ownership.
logger = logging.getLogger("coding_review_agent_loop.round_state")


_PR_BOARD_FIELDS = ("pr_board_root", "pr_board_identity", "pr_board_binding_init")
_PLAN_BOARD_FIELDS = ("plan_board_root", "plan_board_identity", "plan_board_binding_init")
_BOARD_FIELDS = (*_PR_BOARD_FIELDS, *_PLAN_BOARD_FIELDS)


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
    seat_binding: dict | None = None
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
    # Authoritative staged-parent signed requirement IDs current when a child
    # plan reviewer record was written (#1377).  Requirement-binding evidence
    # only; ``None`` on records written without a staged parent and on every
    # historical record, and omitted from the encoding when ``None``.
    parent_requirement_ids: tuple[str, ...] | None = None
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
    # Who raised ``scheduler_force_full`` (#840): ``operator`` for the explicit
    # override, ``automatic`` for a post-panel recovery latch.  Null when the
    # latch is off; legacy latched records without it remain valid metadata.
    scheduler_force_full_source: str | None = None
    # Exact-plan candidate key (#905, from #841).  Written by staged planning
    # scheduler records and by every plan reviewer record so a later round can
    # compare a stored approval component-for-component.  It is deliberately
    # not a ``scheduler_*`` field: a reviewer record that carries it is not a
    # scheduler checkpoint and must keep ``scheduler_metadata_status`` absent.
    plan_candidate_key: dict | None = None
    # Planning-only scheduler auxiliary: the key the previous planning round
    # was scheduled against.  ``None`` on the first staged planning round.
    scheduler_plan_previous_key: dict | None = None
    # Planning-only, primary-phase checkpoint fields (#1112).  The digest is
    # the normalized issue title/body the run loaded; the reset marker is the
    # operator's durable ``--plan-reset-stall-streak`` streak boundary.  Both
    # are omitted from the serialized record when unset.
    scheduler_issue_digest: str | None = None
    scheduler_stall_reset: bool = False
    # Planning-only, primary-phase marker (#1275): this round's round-start stall
    # stop was deferred for a pending plan step-back.  Omitted when False.
    scheduler_step_back_deferral: bool = False
    # The ``--plan-execution-mode`` a planning round ran under (#1268).  Written
    # on plan-flow planner and reviewer records, and on planning scheduler
    # checkpoints as ``scheduler_execution_mode``; both are omitted when None
    # so legacy records stay byte-stable, and an unknown value decodes to None.
    plan_execution_mode: str | None = None
    scheduler_execution_mode: str | None = None
    # The normalized issue title/body digest a plan-flow planner (coder) record
    # was planned against (#1390).  Deliberately not a ``scheduler_*`` field,
    # so a planner record is never mistaken for a scheduler checkpoint; omitted
    # when None so legacy records stay byte-stable.
    plan_issue_digest: str | None = None
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
    # Parser-derived degradation records for this round's assessment (#924).
    # Only the orchestrator writes metadata, so agents cannot author these.
    architecture_impact_degradations: tuple[ParseDegradation, ...] = ()
    # Records of malformed follow-up citations dropped by the degradable
    # citation parser (#927); restored onto the coder carrier on resume.
    test_observation_degradations: tuple[ParseDegradation, ...] = ()
    # Validated advisory coder sub-item claims and the records of dropped ones
    # (#958).  Restored verbatim on resume rather than re-validated against a
    # ledger that may have advanced.
    addressed_sub_items: tuple[str, ...] = ()
    sub_item_claim_degradations: tuple[ParseDegradation, ...] = ()
    # Reviewer-side sub-item parse degradations (#958), restored on resume.
    sub_item_degradations: tuple[ParseDegradation, ...] = ()
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
    # Causal binding of a signed child-plan re-plan (#936).  Both are absent
    # on every historical record and omitted from the encoding when absent.
    plan_supersession_digest: str | None = None
    plan_supersession_superseded_hash: str | None = None
    # Signed reviewer-board amendment binding (#943).  Written only on
    # contract-bearing scheduler records posted under an amended contract;
    # absent on every historical record and omitted from the encoding.
    reviewer_board_amendment_digest: str | None = None
    # PR board lineage (#1376).  ``pr_board_root`` is the immutable root board
    # identity of the run and ``pr_board_identity`` the identity of the
    # lineage element governing the record; ``reviewer_board_change_digest``
    # binds a record to the board-change link that governs it, and
    # ``pr_board_binding_init`` records, once, the configured binding of seats
    # whose legacy binding was unknown.  All are absent on historical records
    # and omitted from the encoding when absent.
    pr_board_root: dict | None = None
    pr_board_identity: dict | None = None
    pr_board_binding_init: dict | None = None
    reviewer_board_change_digest: str | None = None
    # Plan board lineage (#1377): the same fields for issue plan-review runs.
    # They are flow-scoped: ``plan_board_*`` only on plan-flow records and
    # ``pr_board_*`` only on PR-flow records.
    plan_board_root: dict | None = None
    plan_board_identity: dict | None = None
    plan_board_binding_init: dict | None = None
    # Coder record round that last rendered the full visible matrix-evidence
    # row list (#959).  Writers set it iff they persist matrix evidence; it is
    # omitted from the encoding when None so legacy records stay byte-stable.
    risk_test_matrix_evidence_full_round: int | None = None
    # In-memory decode-quality signal for the anchor, never serialized:
    # ``absent`` is a legacy (pre-#959) record, ``invalid`` a present but
    # malformed value.  Constructing with an anchor promotes it to ``valid``.
    risk_test_matrix_evidence_full_round_status: str = "absent"
    # PR head a coder follow-up was dispatched against (#1034).  Written only
    # on PR coder follow-up records and omitted from the encoding when None,
    # so historical records decode unchanged and re-encode byte-identically.
    followup_dispatch_head: str | None = None
    # Human-only exact-head evidence (#1068).  ``evidence_requests`` is the
    # reviewer's parsed request list, written on the reviewer record itself so
    # a deferred request is durable from the moment that record is posted.
    # The freeze/release payloads and head-scoped clearances ride only on the
    # orchestrator's single freeze or release comment.  All are omitted from
    # the encoding when empty so historical records stay byte-stable.
    post_merge_authenticated: bool = False
    post_merge_registry: tuple[dict, ...] = ()
    post_merge_diagnostics: tuple[str, ...] = ()
    post_merge_publication: dict | None = None
    post_merge_requests: tuple[dict, ...] = ()
    screenshot_requests: tuple[dict, ...] = ()
    screenshot_coverage_acceptances: tuple[dict, ...] = ()
    screenshot_coverage: object = None
    screenshot_receipt_refs: tuple[int, ...] = ()
    evidence_requests: tuple[str, ...] = ()
    requirement_evidence: tuple[dict, ...] = ()
    evidence_freeze: "EvidenceFreezeRecord | None" = None
    evidence_release: "EvidenceReleaseRecord | None" = None
    evidence_clearances: tuple[tuple[str, str], ...] = ()
    # Reviewer-owned step-back episode entries (#1251), written by the
    # orchestrator on the planner (or coder) record of a step-back turn and
    # never taken from agent output.  Absent is legacy/no step-back.  The status
    # is an in-memory decode-quality signal, never serialized: a malformed
    # value decodes as ``invalid`` with no entries instead of failing the
    # record, so the step-back trigger can be suppressed for that history.
    step_back_entries: tuple[Mapping[str, object], ...] = ()
    step_back_status: str = "absent"
    # Orchestrator-written plan sweep marker (#1505) on the primary's plan
    # reviewer record of a round whose prompt carried the sweep request; never
    # taken from agent output.  Omitted when False, and a malformed stored value
    # decodes as False.
    plan_sweep: bool = False
    # Orchestrator-written finding-provenance baseline form and payload size
    # (#1516) on a plan reviewer record: the form the deterministic selector
    # rendered into that reviewer's prompt.  Encoded as one optional
    # ``provenance_baseline`` object only when set; a malformed stored value
    # decodes as absent.
    provenance_baseline_form: str | None = None
    provenance_baseline_chars: int | None = None
    # Coder-recovery and head-review-recovery records (#1292).  All are
    # optional and omitted from the encoding when unset, so every historical
    # record re-encodes byte-identically.  A malformed scalar decodes as
    # absent (never raises); ``_recovery_record_problems`` then reports it.
    dispatch_round: int | None = None
    dispatch_head: str | None = None
    dispatch_attempt: int | None = None
    recovery_dispatch: bool | None = None
    carried_rejection_reasons: tuple[str, ...] = ()
    rejected_coder_followup_reason: str | None = None
    rejected_coder_followup_from_head: str | None = None
    head_review_recovery_source: str | None = None
    recovery_round_budget: "RecoveryRoundBudget | None" = None
    # Orchestrator-derived evidence-only stall snapshot (#1324), written only on
    # PR coder follow-up records.  It classifies review round ``round_number - 1``
    # and is omitted from the encoding when None.  An invalid stored value
    # decodes as ``{"invalid": True}`` so history treats it as non-qualifying.
    evidence_stall: Mapping[str, object] | None = None
    # Evidence conversion (#1325).  Every field below is orchestrator-written
    # authority, admitted on resume only from the authenticated actor's own
    # comments (see ``admit_evidence_authority_records``), and omitted from
    # the encoding when empty so historical records decode unchanged.
    # Per-attempt capture classification on PR coder records; a malformed or
    # mis-bound value decodes as None (a non-qualifying attempt).
    risk_test_matrix_capture_classification: Mapping[str, object] | None = None
    # The conversion audit on an ``evidence-conversion`` summary record; an
    # invalid audit decodes as None and is treated as unconverted.
    evidence_conversion: "EvidenceConversionAudit | None" = None
    # The review snapshot of the conversion round, which posts no coder
    # follow-up record; ``review_round`` equals the record's own round.
    conversion_stall_snapshot: Mapping[str, object] | None = None
    # The round whose same-head repeat must re-run the complete board fresh.
    conversion_review_round: int | None = None
    # Cumulative signed-evidence references carried on every terminal freeze
    # or release record and on conversion records; malformed entries drop.
    signed_evidence_references: tuple[SignedEvidenceReference, ...] = ()
    review_scope: object = None
    review_scope_present: bool = False
    visual_lifecycle: object = None
    visual_lifecycle_present: bool = False
    capture_review_binding: object = None

    def __post_init__(self) -> None:
        if self.scheduler_metadata_status not in {"absent", "valid", "invalid"}:
            raise ValueError("invalid scheduler metadata status")
        if self.step_back_status not in {"absent", "valid", "invalid"}:
            raise ValueError("invalid step-back status")
        if self.step_back_entries:
            if self.step_back_status == "invalid":
                raise ValueError("an invalid step-back status cannot carry entries")
            for entry in self.step_back_entries:
                if not step_back_entry_is_valid(entry):
                    raise ValueError("invalid step-back entry")
            if self.step_back_status == "absent":
                object.__setattr__(self, "step_back_status", "valid")
        if self.risk_test_matrix_evidence_full_round_status not in {"absent", "valid", "invalid"}:
            raise ValueError("invalid matrix evidence full-round status")
        if self.risk_test_matrix_evidence_full_round is not None:
            if self.risk_test_matrix_evidence_full_round_status == "invalid":
                raise ValueError("a matrix evidence full-round anchor cannot be invalid")
            if self.risk_test_matrix_evidence_full_round_status == "absent":
                object.__setattr__(
                    self, "risk_test_matrix_evidence_full_round_status", "valid"
                )
        if (self.plan_supersession_digest is None) != (
            self.plan_supersession_superseded_hash is None
        ):
            raise ValueError("plan supersession digest and superseded hash must be set together")
        if self.plan_supersession_digest is not None and (
            not isinstance(self.plan_supersession_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.plan_supersession_digest)
            or not isinstance(self.plan_supersession_superseded_hash, str)
            or not self.plan_supersession_superseded_hash.strip()
        ):
            raise ValueError("invalid plan supersession binding")
        if self.reviewer_board_amendment_digest is not None and (
            not isinstance(self.reviewer_board_amendment_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.reviewer_board_amendment_digest)
        ):
            raise ValueError("invalid reviewer board amendment digest")
        if self.reviewer_board_change_digest is not None and (
            not isinstance(self.reviewer_board_change_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.reviewer_board_change_digest)
        ):
            raise ValueError("invalid reviewer board change digest")
        for board_field in _BOARD_FIELDS:
            if getattr(self, board_field) is not None and not isinstance(
                getattr(self, board_field), dict
            ):
                raise ValueError(f"invalid {board_field}")
        if (self.pr_board_root is None) != (self.pr_board_identity is None):
            raise ValueError("pr_board_root and pr_board_identity must be set together")
        if (self.plan_board_root is None) != (self.plan_board_identity is None):
            raise ValueError("plan_board_root and plan_board_identity must be set together")
        # Board identity fields are flow-scoped (#1377).
        if self.flow != "pr" and any(
            getattr(self, name) is not None for name in _PR_BOARD_FIELDS
        ):
            raise ValueError("pr_board_* fields are only valid on PR-flow records")
        if self.flow != "plan" and any(
            getattr(self, name) is not None for name in _PLAN_BOARD_FIELDS
        ):
            raise ValueError("plan_board_* fields are only valid on plan-flow records")
        if self.followup_dispatch_head is not None and not _is_followup_dispatch_head(
            self.followup_dispatch_head
        ):
            raise ValueError("invalid followup_dispatch_head: expected a hex Git commit SHA")
        if self.scheduler_force_full_source is not None and (
            self.scheduler_force_full_source not in FORCE_FULL_SOURCES
            or self.scheduler_force_full is not True
        ):
            raise ValueError("scheduler force-full source requires an active force-full latch")
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
                self.scheduler_force_full_source,
                self.scheduler_plan_previous_key,
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
    original_item_number_high_water: int = 0


@dataclass(frozen=True)
class ResumedRoundSelection:
    anchor_record: PostedRoundRecord
    current_round_records: tuple[PostedRoundRecord, ...]


def rebuild_resumed_coder_carrier(
    text: str | None, metadata: "PostedRoundMetadata | None"
) -> StructuredCoderFollowup | StructuredIssueImplementation | None:
    """Rebuild a resumed round's coder carrier from its durable raw response.

    Dropped-citation records are restored from the round metadata written
    when the response was accepted (#927), so a restart never loses them.
    The degradable citation parser is deterministic, so re-parsing the
    stored ``raw_structured_coder_response`` yields the same records.
    """
    if not text:
        return None
    carrier: StructuredCoderFollowup | StructuredIssueImplementation | None
    try:
        carrier = parse_historical_structured_issue_implementation(text)
    except IssueImplementationConflictError as exc:
        carrier = exc.payload if isinstance(exc.payload, StructuredIssueImplementation) else None
    except AgentLoopError:
        carrier = None
    if carrier is None:
        try:
            carrier = parse_historical_structured_coder_followup(text)
        except AgentLoopError:
            return None
    if carrier is None or metadata is None:
        return carrier
    carrier = replace(
        carrier, test_observation_degradations=tuple(metadata.test_observation_degradations)
    )
    if isinstance(carrier, StructuredCoderFollowup) and (
        metadata.addressed_sub_items or metadata.sub_item_claim_degradations
    ):
        carrier = replace(
            carrier,
            addressed_sub_items=tuple(metadata.addressed_sub_items),
            sub_item_claim_degradations=tuple(metadata.sub_item_claim_degradations),
        )
    return carrier


@dataclass(frozen=True)
class ResumedReviewRound:
    round_number: int
    prior_items: tuple[UnresolvedReviewItem, ...]
    coder_output: str | None
    completed_reviews: tuple[PostedRoundRecord, ...]
    next_unresolved_item_number: int
    ledger_may_be_incomplete: bool = False
    # Public coder comment of the anchor round; only its orchestrator-authored
    # coverage-map section is consumed (#1290), never the coder's own prose.
    coder_comment_body: str | None = None
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
    # The latest evidence freeze or release record that anchors this resume
    # (#1068), and the frozen head when an external push broke a freeze.
    evidence_boundary: PostedRoundMetadata | None = None
    broken_evidence_freeze_head: str | None = None
    # Head-scoped evidence clearances carried by the latest terminal record.
    evidence_clearances: tuple[tuple[str, str], ...] = ()
    # Coder-recovery context (#1292): why the previous coder follow-up was not
    # accepted, the head it was dispatched on, and its consecutive attempt.
    rejected_coder_followup_reason: str | None = None
    rejected_coder_followup_from_head: str | None = None
    dispatch_attempt: int | None = None
    # Bounded round budget carried by a recovery record.
    recovery_round_budget: "RecoveryRoundBudget | None" = None
    # ``legacy-checkpoint`` or ``operator`` when the resume is an ordinary
    # review round of the current head; ``post_required`` is True while the
    # handoff record for the current head has not been written yet.
    head_review_recovery: str | None = None
    head_review_recovery_post_required: bool = False
    # Rejection reasons carried into the next coder-dispatch record (bounded).
    carried_rejection_reasons: tuple[str, ...] = ()
    # The anchor round's own reconciliation-phase summary (#1553).  Set only
    # by the ordinary transcript resume, never by a qualification-checkpoint,
    # evidence-boundary, conversion or recovery anchor.  It closes the round:
    # a later decision that needs fresh reviewer turns opens the next round,
    # with the ledger settled from this record.
    reconciliation_record: PostedRoundRecord | None = None


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


def _plan_validation_diagnostic_label(
    payload: PlanValidationDiagnosticPayload,
) -> str:
    """Render the deterministic label for one diagnostic payload.

    Both identity fields used here live in the payload, so the label of an
    already-posted record can be regenerated exactly during decoding.
    """
    return protocol_record_label(
        "plan_validation_diagnostic",
        issue_number=payload.issue_number,
        attempt=payload.failure_attempt,
    )


def encode_plan_validation_diagnostic_body(
    payload: PlanValidationDiagnosticPayload,
) -> TrustedBody:
    """Encode only the immutable pre-POST diagnostic payload."""
    encoded = encode_mapping(payload.as_dict())
    return TrustedBody.canonical(
        f"{_plan_validation_diagnostic_label(payload)}\n\n"
        f"<!-- AGENT_PLAN_VALIDATION_DIAGNOSTIC: {encoded} -->",
        surface=ISSUE_COMMENT_SURFACE,
        expected_tokens=("AGENT_PLAN_VALIDATION_DIAGNOSTIC",),
    )


def decode_plan_validation_diagnostic_body(
    body: str,
) -> PlanValidationDiagnosticPayload:
    matches = tuple(PLAN_VALIDATION_DIAGNOSTIC_MARKER_RE.finditer(body))
    if len(matches) != 1:
        raise AgentLoopError("Plan-validation diagnostic is not an exact canonical record.")
    marker = matches[0].group(0)
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
        payload = PlanValidationDiagnosticPayload.from_mapping(mapping)
    except (AgentLoopError, TypeError, ValueError, KeyError) as exc:
        if isinstance(exc, AgentLoopError) and str(exc).startswith("Invalid plan-validation"):
            raise
        raise AgentLoopError("Invalid plan-validation diagnostic record.") from exc
    # Exactly two canonical carrier forms are accepted: the historical bare
    # marker, and the marker introduced by its own regenerated label.  Any other
    # surrounding text is still rejected.
    # The comparison is against the unmodified body: surrounding whitespace
    # would be a third carrier form, and only these two are canonical.
    labeled = f"{_plan_validation_diagnostic_label(payload)}\n\n{marker}"
    if body not in {marker, labeled}:
        raise AgentLoopError("Plan-validation diagnostic is not an exact canonical record.")
    return payload


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
                # A fresh plan opens round 1 or, after an edited-issue
                # rescope, the scope start round (#1390).
                (candidate_kind == "plan_state" and target_coder_round >= 1)
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
    on_host_footer: Callable[[str], None] | None = None,
) -> PlanValidationDiagnosticTransport | None:
    """Recover one authenticated current diagnostic using payload attempt order."""
    if not expected_author_login or expected_author_id < 1:
        raise AgentLoopError("Plan-validation diagnostic recovery requires an authenticated actor.")
    candidates: list[PlanValidationDiagnosticTransport] = []
    by_id: dict[int, str] = {}
    for comment in comments:
        raw_body = getattr(comment, "body", None)
        if not isinstance(raw_body, str):
            continue
        # Comment-ingestion boundary (#1043): remove the known host footer
        # exactly once here.  The decoder never strips, so a doubled or variant
        # suffix stays ineligible.
        body, footered = strip_known_host_footer(raw_body)
        # Report the observation at the boundary, before any marker, identity,
        # or decoder exit, like the other ingestion boundaries.
        if footered and on_host_footer is not None:
            on_host_footer("plan-validation diagnostic recovery")
        if not PLAN_VALIDATION_DIAGNOSTIC_MARKER_RE.search(body):
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


EVIDENCE_FREEZE_PHASE = "evidence-freeze"
EVIDENCE_RESPONSE_PHASE = "evidence-response"
EVIDENCE_RELEASE_PHASE = "evidence-release"
EVIDENCE_TERMINAL_PHASES = frozenset({EVIDENCE_FREEZE_PHASE, EVIDENCE_RELEASE_PHASE})
# The orchestrator's single conversion record (#1325): it carries the
# post-conversion ledger and the audit, and is written before any dispatch.
EVIDENCE_CONVERSION_PHASE = "evidence-conversion"
EVIDENCE_RELEASE_REASONS = frozenset(
    {"evidence-cleared", "findings", "machine-gate", "refresh-clean", "screenshot-coverage", "post-merge-scheduled"}
)


def _decode_evidence_budget(value: Mapping[str, object]) -> tuple[int | None, bool | None, bool | None, bool]:
    """Decode the round-budget triple; a missing or mistyped field is invalid."""
    allowed = value.get("allowed_rounds")
    failure = value.get("watch_failure_extension_used")
    head = value.get("watch_head_extension_used")
    valid = (
        isinstance(allowed, int)
        and not isinstance(allowed, bool)
        and allowed >= 1
        and isinstance(failure, bool)
        and isinstance(head, bool)
    )
    if not valid:
        return None, None, None, False
    return allowed, failure, head, True


def _decode_string_list(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        return None
    return tuple(value)


@dataclass(frozen=True)
class EvidenceFreezeRecord:
    """Durable payload of the single freeze comment (#1068).

    ``valid`` covers the non-budget fields; a record whose only defect is its
    budget keeps the freeze in effect with ``budget_valid=False`` so recovery
    can fall back to the conservative defaults and re-persist a corrected
    still-frozen record instead of treating the frozen ledger as absent.
    """

    frozen_head: str
    evidence_identities: tuple[str, ...]
    signed_requirement_ids_at_freeze: tuple[str, ...]
    allowed_rounds: int | None
    watch_failure_extension_used: bool | None
    watch_head_extension_used: bool | None
    valid: bool = True
    budget_valid: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "frozen_head": self.frozen_head,
            "evidence_identities": list(self.evidence_identities),
            "signed_requirement_ids_at_freeze": list(self.signed_requirement_ids_at_freeze),
            "allowed_rounds": self.allowed_rounds,
            "watch_failure_extension_used": self.watch_failure_extension_used,
            "watch_head_extension_used": self.watch_head_extension_used,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "EvidenceFreezeRecord":
        if not isinstance(value, Mapping):
            return cls.invalid()
        head = value.get("frozen_head")
        identities = _decode_string_list(value.get("evidence_identities"))
        signed = _decode_string_list(value.get("signed_requirement_ids_at_freeze"))
        if not isinstance(head, str) or not head or not identities or signed is None:
            return cls.invalid()
        allowed, failure, head_ext, budget_valid = _decode_evidence_budget(value)
        return cls(
            frozen_head=head,
            evidence_identities=identities,
            signed_requirement_ids_at_freeze=signed,
            allowed_rounds=allowed,
            watch_failure_extension_used=failure,
            watch_head_extension_used=head_ext,
            valid=True,
            budget_valid=budget_valid,
        )

    @classmethod
    def invalid(cls) -> "EvidenceFreezeRecord":
        return cls(
            frozen_head="",
            evidence_identities=(),
            signed_requirement_ids_at_freeze=(),
            allowed_rounds=None,
            watch_failure_extension_used=None,
            watch_head_extension_used=None,
            valid=False,
            budget_valid=False,
        )


@dataclass(frozen=True)
class EvidenceReleaseRecord:
    """Durable payload of a terminal release comment (#1068).

    ``signed_requirement_ids_surfaced`` is the signed-requirement ID set the
    reviewers of the releasing pass saw.  ``None`` means the baseline is
    missing or invalid; revalidation then treats it as empty, which forces a
    same-head refresh pass whenever any signed input exists.
    """

    released_head: str
    reason: str
    signed_requirement_ids_surfaced: tuple[str, ...] | None
    allowed_rounds: int | None
    watch_failure_extension_used: bool | None
    watch_head_extension_used: bool | None
    valid: bool = True
    budget_valid: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "released_head": self.released_head,
            "reason": self.reason,
            "signed_requirement_ids_surfaced": (
                list(self.signed_requirement_ids_surfaced)
                if self.signed_requirement_ids_surfaced is not None
                else None
            ),
            "allowed_rounds": self.allowed_rounds,
            "watch_failure_extension_used": self.watch_failure_extension_used,
            "watch_head_extension_used": self.watch_head_extension_used,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "EvidenceReleaseRecord":
        if not isinstance(value, Mapping):
            return cls.invalid()
        head = value.get("released_head")
        reason = value.get("reason")
        if not isinstance(head, str) or not head or reason not in EVIDENCE_RELEASE_REASONS:
            return cls.invalid()
        allowed, failure, head_ext, budget_valid = _decode_evidence_budget(value)
        signed = _decode_string_list(value.get("signed_requirement_ids_surfaced"))
        return cls(
            released_head=head,
            reason=str(reason),
            signed_requirement_ids_surfaced=signed,
            allowed_rounds=allowed,
            watch_failure_extension_used=failure,
            watch_head_extension_used=head_ext,
            valid=True,
            budget_valid=budget_valid,
        )

    @classmethod
    def invalid(cls) -> "EvidenceReleaseRecord":
        return cls(
            released_head="",
            reason="machine-gate",
            signed_requirement_ids_surfaced=None,
            allowed_rounds=None,
            watch_failure_extension_used=None,
            watch_head_extension_used=None,
            valid=False,
            budget_valid=False,
        )


from .qualification_checkpoint import QualificationCheckpoint


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
    if item.alias_item_ids or item.alias_obligation_identities:
        payload["alias_item_ids"] = list(item.alias_item_ids)
        payload["alias_obligation_identities"] = list(item.alias_obligation_identities)
    if item.sub_items:
        payload["sub_items"] = [_serialize_sub_item(sub) for sub in item.sub_items]
    if item.evidence_row_ids:
        payload["evidence_row_ids"] = list(item.evidence_row_ids)
    if item.converted_row_ids:
        payload["converted_row_ids"] = list(item.converted_row_ids)
    if item.converted_row_id is not None:
        payload["converted_row_id"] = item.converted_row_id
        payload["converted_matrix_identity"] = item.converted_matrix_identity
    # Plan review telemetry (#1505): omitted when unset so legacy items
    # re-encode byte-identically.
    if item.provenance is not None:
        payload["provenance"] = item.provenance
    if item.screenshot_request is not None:
        payload["screenshot_request"] = item.screenshot_request
    if item.origin_scope is not None:
        payload["origin_scope"] = item.origin_scope
    if item.repeat_of is not None:
        payload["repeat_of"] = item.repeat_of
    return payload


def _deserialize_converted_row_ids(payload: dict, evidence_row_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Decode the converted subset of a finding's tag (#1325); malformed is none.

    Dropping it restores the full residual demand, so corrupt history never
    hides a row from the coder or the stall detector.
    """
    raw = payload.get("converted_row_ids")
    if raw is None:
        return ()
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(entry, str) or entry not in evidence_row_ids for entry in raw)
        or len(set(raw)) != len(raw)
    ):
        logger.warning("Ignoring malformed converted_row_ids on a persisted review item")
        return ()
    return tuple(raw)


def _deserialize_converted_row(payload: dict) -> dict[str, str]:
    """Decode a row-converted obligation's row binding (#1325).

    The pair must reproduce the obligation identity; otherwise the binding is
    dropped and the item stays an ordinary evidence obligation that every
    configured reviewer must clear.
    """
    row_id = payload.get("converted_row_id")
    identity = payload.get("converted_matrix_identity")
    if row_id is None and identity is None:
        return {}
    from .unresolved_items import converted_evidence_obligation_identity

    if (
        isinstance(row_id, str)
        and row_id.strip()
        and isinstance(identity, str)
        and re.fullmatch(r"[0-9a-f]{64}", identity)
        and payload.get("obligation_kind") == EVIDENCE_OBLIGATION_KIND
        and payload.get("obligation_identity")
        == converted_evidence_obligation_identity(row_id, identity)
    ):
        return {"converted_row_id": row_id, "converted_matrix_identity": identity}
    logger.warning("Ignoring a malformed converted-row binding on a persisted evidence obligation")
    return {}


def _deserialize_evidence_row_ids(payload: dict) -> tuple[str, ...]:
    """Decode a persisted evidence tag; a malformed value yields none (#1324).

    Dropping the tag degrades the item to an ordinary finding, so corrupt
    history can never hide a finding from the stall detector.
    """
    raw = payload.get("evidence_row_ids")
    if raw is None:
        return ()
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(entry, str) or not entry.strip() for entry in raw)
        or len(set(raw)) != len(raw)
    ):
        logger.warning("Ignoring malformed evidence_row_ids on a persisted review item")
        return ()
    return tuple(raw)


def _deserialize_item_provenance(payload: dict) -> str | None:
    """Decode a persisted finding provenance tag; malformed is untagged (#1505)."""
    raw = payload.get("provenance")
    if raw is None:
        return None
    if not isinstance(raw, str) or raw not in PLAN_REVIEW_PROVENANCE_VALUES:
        logger.warning("Ignoring malformed provenance on a persisted review item")
        return None
    return raw


def _deserialize_repeat_of(payload: dict) -> str | None:
    """Decode a persisted repeat marker; malformed is absent (#1505)."""
    raw = payload.get("repeat_of")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        logger.warning("Ignoring malformed repeat_of on a persisted review item")
        return None
    return raw


def _serialize_sub_item(sub: ReviewSubItem) -> dict[str, object]:
    return {
        "sub_item_id": sub.sub_item_id,
        "text": sub.text,
        "status": sub.status,
        "resolved_round": sub.resolved_round,
    }


def _deserialize_sub_items(payload: dict) -> tuple[ReviewSubItem, ...]:
    """Decode persisted sub-items; a malformed payload yields none (#958).

    Dropping the sub-items loses only progress detail: the carried item stays
    open, so a corrupted record can never clear or weaken it.
    """
    raw = payload.get("sub_items")
    if not raw or not isinstance(raw, list):
        return ()
    decoded: list[ReviewSubItem] = []
    try:
        for entry in raw:
            if not isinstance(entry, dict):
                return ()
            status = entry["status"]
            resolved_round = entry.get("resolved_round")
            if (
                status not in SUB_ITEM_STATUSES
                or not isinstance(entry["sub_item_id"], str)
                or not isinstance(entry["text"], str)
                or (resolved_round is not None and (
                    isinstance(resolved_round, bool) or not isinstance(resolved_round, int)
                ))
                or (status == "open" and resolved_round is not None)
            ):
                return ()
            decoded.append(
                ReviewSubItem(
                    sub_item_id=entry["sub_item_id"],
                    text=entry["text"],
                    status=status,
                    resolved_round=resolved_round,
                )
            )
    except (KeyError, TypeError):
        return ()
    if len({sub.sub_item_id for sub in decoded}) != len(decoded):
        return ()
    return tuple(decoded)


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
            **_deserialize_evidence_aliases(payload),
        )
        try:
            item = UnresolvedReviewItem(**core, **machine, **_deserialize_converted_row(payload),
                screenshot_request=payload.get("screenshot_request"))
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
    evidence_row_ids = _deserialize_evidence_row_ids(payload)
    converted_row_ids = _deserialize_converted_row_ids(payload, evidence_row_ids)
    if core["status"] == "converted" and (
        not converted_row_ids or set(converted_row_ids) != set(evidence_row_ids)
    ):
        # An inconsistent converted finding is restored as an open blocker.
        core = {**core, "status": "blocking"}
    return UnresolvedReviewItem(
        **core,
        sub_items=_deserialize_sub_items(payload),
        evidence_row_ids=evidence_row_ids,
        converted_row_ids=converted_row_ids,
        provenance=_deserialize_item_provenance(payload),
        repeat_of=_deserialize_repeat_of(payload),
        origin_scope=payload.get("origin_scope"),
        screenshot_request=payload.get("screenshot_request"),
    )


def _deserialize_evidence_aliases(payload):
    aliases = {}
    for key, pattern in (("alias_item_ids", r"item-\d+"),
                         ("alias_obligation_identities", r"human-exact-head-evidence:[0-9a-f]{24}")):
        value = payload.get(key, [])
        if not isinstance(value, list) or any(not isinstance(v, str) or not re.fullmatch(pattern, v) for v in value):
            raise AgentLoopError("Invalid persisted evidence aliases")
        aliases[key] = tuple(dict.fromkeys(value))
    return aliases


def _serialize_disposition(disposition: ReviewItemDisposition) -> dict[str, object]:
    payload: dict[str, object] = {
        "item_id": disposition.item_id,
        "reviewer": disposition.reviewer,
        "disposition": disposition.disposition,
        "note": disposition.note,
    }
    if disposition.evidence_requirement_ids:
        payload["evidence_requirement_ids"] = list(disposition.evidence_requirement_ids)
    if disposition.sub_item_dispositions:
        payload["sub_item_dispositions"] = [
            list(pair) for pair in disposition.sub_item_dispositions
        ]
    return payload


def _deserialize_disposition(payload: object) -> ReviewItemDisposition:
    if not isinstance(payload, dict):
        raise AgentLoopError("Invalid round metadata disposition payload.")
    from .native_evidence import citation_ids
    return ReviewItemDisposition(
        evidence_requirement_ids=citation_ids(payload.get("evidence_requirement_ids", [])),
        item_id=str(payload["item_id"]),
        reviewer=str(payload["reviewer"]),
        disposition=str(payload["disposition"]),
        note=str(payload["note"]) if payload.get("note") is not None else None,
        sub_item_dispositions=_deserialize_sub_item_dispositions(payload),
    )


def _deserialize_sub_item_dispositions(payload: dict) -> tuple[tuple[str, str], ...]:
    raw = payload.get("sub_item_dispositions")
    if not raw or not isinstance(raw, list):
        return ()
    return tuple(
        (pair[0], pair[1])
        for pair in raw
        if isinstance(pair, (list, tuple))
        and len(pair) == 2
        and isinstance(pair[0], str)
        and isinstance(pair[1], str)
        and pair[1] in {"resolved", "unresolved"}
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
    if (
        "risk_test_matrix_claims" not in payload
        and "risk_test_matrix_coverage_gaps" not in payload
    ):
        return raw
    payload = dict(payload)
    payload.pop("risk_test_matrix_claims", None)
    # Coverage-map gap declarations are a fresh-turn acquisition contract too
    # and are never persisted as authority (#1290).
    payload.pop("risk_test_matrix_coverage_gaps", None)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + normalized[end:]


CODER_DISPATCH_PHASE = "coder-dispatch"
CODER_FOLLOWUP_REJECTED_PHASE = "coder-followup-rejected"
HEAD_REVIEW_RECOVERY_PHASE = "head-review-recovery"
RECOVERY_RECORD_PHASES = frozenset(
    {CODER_DISPATCH_PHASE, CODER_FOLLOWUP_REJECTED_PHASE, HEAD_REVIEW_RECOVERY_PHASE}
)
HEAD_REVIEW_RECOVERY_SOURCES = frozenset({"operator", "legacy-checkpoint"})
MAX_REJECTED_DISPATCH_ATTEMPTS = 2
MAX_RECOVERY_REASON_CHARS = 2000
NO_REASON_TEXT = "rejected without a reason"
UNRECORDED_DISPATCH_REASON = (
    "the previous coder follow-up response was not recorded (rejected or interrupted)"
)
REVIEW_UNRECORDED_HEAD_HINT = (
    "Rerun with `--review-unrecorded-head` to run an ordinary review round of the "
    "current head from the recorded items."
)


def sanitize_recovery_reason(value: object) -> str:
    """Bound and redact a rejection reason before it is persisted or prompted."""
    text = value if isinstance(value, str) else str(value)
    text = sanitize_historical_text(text)
    # A reason is quoted into a public comment and a prompt: it must never carry
    # an HTML-comment protocol marker of its own.
    text = HTML_COMMENT_RE.sub("[comment removed]", text)
    text = _PLAN_VALIDATION_SENSITIVE_RE.sub("[redacted]", text)
    text = "".join(
        character for character in text if character in "\n\r\t" or ord(character) >= 32
    ).strip()
    if not text:
        return NO_REASON_TEXT
    if len(text) > MAX_RECOVERY_REASON_CHARS:
        text = text[: MAX_RECOVERY_REASON_CHARS - 3].rstrip() + "..."
    return text


@dataclass(frozen=True)
class RecoveryRoundBudget:
    """Bounded round budget carried by a recovery record."""

    allowed_rounds: int
    watch_failure_extension_used: bool
    watch_head_extension_used: bool
    valid: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed_rounds": self.allowed_rounds,
            "watch_failure_extension_used": self.watch_failure_extension_used,
            "watch_head_extension_used": self.watch_head_extension_used,
        }

    @classmethod
    def invalid(cls) -> "RecoveryRoundBudget":
        return cls(0, False, False, valid=False)

    @classmethod
    def from_mapping(cls, value: object) -> "RecoveryRoundBudget":
        if not isinstance(value, Mapping):
            return cls.invalid()
        allowed = value.get("allowed_rounds")
        failure = value.get("watch_failure_extension_used")
        head = value.get("watch_head_extension_used")
        if (
            isinstance(allowed, bool)
            or not isinstance(allowed, int)
            or allowed < 1
            or not isinstance(failure, bool)
            or not isinstance(head, bool)
        ):
            return cls.invalid()
        return cls(allowed, failure, head)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: "QualificationCheckpoint | None"
    ) -> "RecoveryRoundBudget | None":
        if checkpoint is None:
            return None
        if not checkpoint.valid:
            return cls.invalid()
        return cls(
            checkpoint.allowed_rounds,
            checkpoint.watch_failure_extension_used,
            checkpoint.watch_head_extension_used,
        )


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _recovery_record_problems(metadata: PostedRoundMetadata) -> tuple[str, ...]:
    """Names of the fields that make a recovery record structurally invalid.

    Rules are per phase; no head rule is shared across phases.  An empty tuple
    means valid.  Callers must run this before reading any recovery field.
    """
    phase = metadata.phase
    problems: list[str] = []
    budget = metadata.recovery_round_budget
    budget_ok = budget is not None and budget.valid
    if phase in {CODER_DISPATCH_PHASE, CODER_FOLLOWUP_REJECTED_PHASE}:
        if not _is_positive_int(metadata.dispatch_round):
            problems.append("dispatch_round")
        if not metadata.dispatch_head:
            problems.append("dispatch_head")
        attempt = metadata.dispatch_attempt
        if not _is_positive_int(attempt) or attempt > MAX_REJECTED_DISPATCH_ATTEMPTS:
            problems.append("dispatch_attempt")
        if not isinstance(metadata.recovery_dispatch, bool) or (
            _is_positive_int(attempt)
            and metadata.recovery_dispatch != (attempt > 1)
        ):
            problems.append("recovery_dispatch")
        if not budget_ok:
            problems.append("recovery_round_budget")
        if phase == CODER_DISPATCH_PHASE:
            if not metadata.dispatch_head or metadata.subject != metadata.dispatch_head:
                problems.append("subject")
            if not _is_positive_int(metadata.dispatch_round) or (
                metadata.round_number != metadata.dispatch_round + 1
            ):
                problems.append("round_number")
        else:
            if not metadata.subject or metadata.subject == "unknown":
                problems.append("subject")
            if (
                not metadata.rejected_coder_followup_from_head
                or metadata.rejected_coder_followup_from_head != metadata.dispatch_head
            ):
                problems.append("rejected_coder_followup_from_head")
            if not metadata.rejected_coder_followup_reason:
                problems.append("rejected_coder_followup_reason")
            if not _is_positive_int(metadata.dispatch_round) or (
                metadata.round_number != metadata.dispatch_round
            ):
                problems.append("round_number")
    elif phase == HEAD_REVIEW_RECOVERY_PHASE:
        if metadata.head_review_recovery_source not in HEAD_REVIEW_RECOVERY_SOURCES:
            problems.append("head_review_recovery_source")
        if not budget_ok:
            problems.append("recovery_round_budget")
        if not _is_positive_int(metadata.round_number):
            problems.append("round_number")
    return tuple(problems)


def recovery_record_problems(metadata: PostedRoundMetadata) -> tuple[str, ...]:
    """Public view of the per-phase recovery-record validation (#1367)."""
    return _recovery_record_problems(metadata)


def decode_round_metadata_mapping(payload: Mapping[str, object]) -> PostedRoundMetadata:
    """Decode one hydrated round-metadata payload with the canonical decoder."""
    return _decode_round_metadata_mapping(payload)


def _decode_recovery_fields(payload: Mapping[str, object]) -> dict[str, object]:
    """Decode the optional #1292 recovery fields; a malformed one is absent."""
    fields: dict[str, object] = {}
    for key in ("dispatch_round", "dispatch_attempt"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            fields[key] = value
    for key in (
        "dispatch_head",
        "rejected_coder_followup_reason",
        "rejected_coder_followup_from_head",
        "head_review_recovery_source",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value:
            fields[key] = value
    if isinstance(payload.get("recovery_dispatch"), bool):
        fields["recovery_dispatch"] = payload["recovery_dispatch"]
    reasons = payload.get("carried_rejection_reasons")
    if isinstance(reasons, list) and all(isinstance(item, str) for item in reasons):
        fields["carried_rejection_reasons"] = tuple(
            sanitize_recovery_reason(item)
            for item in reasons[:MAX_REJECTED_DISPATCH_ATTEMPTS]
        )
    if "recovery_round_budget" in payload:
        fields["recovery_round_budget"] = RecoveryRoundBudget.from_mapping(
            payload["recovery_round_budget"]
        )
    return fields


def _encode_recovery_fields(metadata: PostedRoundMetadata) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key in (
        "dispatch_round",
        "dispatch_head",
        "dispatch_attempt",
        "recovery_dispatch",
        "rejected_coder_followup_reason",
        "rejected_coder_followup_from_head",
        "head_review_recovery_source",
    ):
        value = getattr(metadata, key)
        if value is not None:
            payload[key] = value
    if metadata.carried_rejection_reasons:
        payload["carried_rejection_reasons"] = list(metadata.carried_rejection_reasons)
    if metadata.recovery_round_budget is not None:
        payload["recovery_round_budget"] = (
            metadata.recovery_round_budget.as_dict()
            if metadata.recovery_round_budget.valid
            else {"allowed_rounds": 0}
        )
    return payload


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
        **(
            {"parent_requirement_ids": list(metadata.parent_requirement_ids)}
            if metadata.parent_requirement_ids is not None
            else {}
        ),
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
    if metadata.seat_binding is not None:
        payload["seat_binding"] = metadata.seat_binding
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
        "scheduler_force_full_source": metadata.scheduler_force_full_source,
        "scheduler_plan_previous_key": metadata.scheduler_plan_previous_key,
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
    if metadata.scheduler_issue_digest is not None:
        payload["scheduler_issue_digest"] = metadata.scheduler_issue_digest
    if metadata.scheduler_stall_reset:
        payload["scheduler_stall_reset"] = True
    if metadata.scheduler_step_back_deferral:
        payload["scheduler_step_back_deferral"] = True
    if metadata.scheduler_execution_mode is not None:
        payload["scheduler_execution_mode"] = metadata.scheduler_execution_mode
    if metadata.plan_execution_mode is not None:
        payload["plan_execution_mode"] = metadata.plan_execution_mode
    if metadata.plan_issue_digest is not None:
        payload["plan_issue_digest"] = metadata.plan_issue_digest
    if metadata.plan_supersession_digest is not None:
        payload["plan_supersession_digest"] = metadata.plan_supersession_digest
        payload["plan_supersession_superseded_hash"] = (
            metadata.plan_supersession_superseded_hash
        )
    if metadata.reviewer_board_amendment_digest is not None:
        payload["reviewer_board_amendment_digest"] = metadata.reviewer_board_amendment_digest
    if metadata.reviewer_board_change_digest is not None:
        payload["reviewer_board_change_digest"] = metadata.reviewer_board_change_digest
    for board_field in _BOARD_FIELDS:
        if getattr(metadata, board_field) is not None:
            payload[board_field] = getattr(metadata, board_field)
    if metadata.plan_candidate_key is not None:
        payload["plan_candidate_key"] = metadata.plan_candidate_key
    if metadata.followup_dispatch_head is not None:
        payload["followup_dispatch_head"] = metadata.followup_dispatch_head
    if metadata.evidence_stall is not None:
        payload["evidence_stall"] = dict(metadata.evidence_stall)
    if metadata.step_back_entries:
        payload["step_back_entries"] = [dict(entry) for entry in metadata.step_back_entries]
    if metadata.plan_sweep:
        payload["plan_sweep"] = True
    if metadata.provenance_baseline_form is not None:
        payload["provenance_baseline"] = {
            "form": metadata.provenance_baseline_form,
            "chars": metadata.provenance_baseline_chars or 0,
        }
    if metadata.risk_test_matrix_evidence_full_round is not None:
        payload["risk_test_matrix_evidence_full_round"] = (
            metadata.risk_test_matrix_evidence_full_round
        )
    matrix_present =_risk_test_matrix_metadata_present(metadata)
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
    if metadata.architecture_impact_degradations:
        # Optional: omitted when empty so undegraded rounds keep their shape.
        payload["architecture_impact_degradations"] = [
            record.to_payload() for record in metadata.architecture_impact_degradations
        ]
    if metadata.test_observation_degradations:
        # Optional, like the field above, so rounds without citation records
        # keep their exact historical encoding.
        payload["test_observation_degradations"] = [
            record.to_payload() for record in metadata.test_observation_degradations
        ]
    if metadata.addressed_sub_items:
        payload["addressed_sub_items"] = list(metadata.addressed_sub_items)
    if metadata.sub_item_claim_degradations:
        payload["sub_item_claim_degradations"] = [
            record.to_payload() for record in metadata.sub_item_claim_degradations
        ]
    if metadata.sub_item_degradations:
        payload["sub_item_degradations"] = [
            record.to_payload() for record in metadata.sub_item_degradations
        ]
    if metadata.qualification_checkpoint is not None:
        payload["qualification_checkpoint"] = (
            metadata.qualification_checkpoint.as_dict()
            if isinstance(metadata.qualification_checkpoint, QualificationCheckpoint)
            else metadata.qualification_checkpoint
        )
    from .review_images_coverage_history import encode_fields
    payload.update(encode_fields(metadata))
    from .post_merge_state import encode_fields as encode_post_merge
    payload.update(encode_post_merge(metadata))
    if metadata.requirement_evidence:
        payload["requirement_evidence"] = list(metadata.requirement_evidence)
    if metadata.evidence_requests:
        payload["evidence_requests"] = list(metadata.evidence_requests)
    if metadata.evidence_freeze is not None:
        payload["evidence_freeze"] = metadata.evidence_freeze.as_dict()
    if metadata.evidence_release is not None:
        payload["evidence_release"] = metadata.evidence_release.as_dict()
    if metadata.evidence_clearances:
        payload["evidence_clearances"] = [list(pair) for pair in metadata.evidence_clearances]
    if metadata.risk_test_matrix_capture_classification is not None:
        payload["risk_test_matrix_capture_classification"] = dict(
            metadata.risk_test_matrix_capture_classification
        )
    if metadata.evidence_conversion is not None:
        payload["evidence_conversion"] = metadata.evidence_conversion.to_payload()
    if metadata.conversion_stall_snapshot is not None:
        payload["conversion_stall_snapshot"] = dict(metadata.conversion_stall_snapshot)
    if metadata.conversion_review_round is not None:
        payload["conversion_review_round"] = metadata.conversion_review_round
    if metadata.signed_evidence_references:
        payload["signed_evidence_references"] = [
            reference.to_payload() for reference in metadata.signed_evidence_references
        ]
    payload.update(encode_review_scope(metadata))
    payload.update(_encode_recovery_fields(metadata))
    return encode_mapping(payload)


def _decode_parse_degradations(value: object) -> tuple[ParseDegradation, ...]:
    """Rehydrate records strictly; historical or malformed entries yield none."""
    if not isinstance(value, list):
        return ()
    records: list[ParseDegradation] = []
    for item in value[:32]:
        try:
            records.append(parse_degradation_payload(item))
        except AgentLoopError:
            continue
    return tuple(records)


# A Git commit SHA (SHA-1 or SHA-256, abbreviated or full) in lowercase hex;
# placeholders such as ``unknown`` never qualify as a dispatch head (#1034).
_FOLLOWUP_DISPATCH_HEAD_RE = re.compile(r"[0-9a-f]{4,64}")


EVIDENCE_STALL_INVALID: Mapping[str, object] = {"invalid": True}


def _decode_evidence_stall(payload: Mapping[str, object]) -> Mapping[str, object] | None:
    """Decode the stall snapshot, never raising on a malformed value (#1324).

    The snapshot's ``review_round`` must equal the enclosing record round
    minus one and its content must be internally consistent; anything else
    decodes as invalid and counts as non-qualifying.
    """
    if "evidence_stall" not in payload:
        return None
    raw = payload["evidence_stall"]
    try:
        round_number = payload.get("round_number")
        if (
            not isinstance(raw, dict)
            or raw.get("version") != 1
            or isinstance(raw.get("review_round"), bool)
            or not isinstance(raw.get("review_round"), int)
            or not isinstance(round_number, int)
            or raw["review_round"] != round_number - 1
            or not isinstance(raw.get("review_head"), str)
            or not isinstance(raw.get("matrix_identity"), str)
            or not isinstance(raw.get("qualifies"), bool)
            or not isinstance(raw.get("unsatisfied_row_ids"), list)
            or not all(isinstance(v, str) for v in raw["unsatisfied_row_ids"])
            or not isinstance(raw.get("reasons"), list)
            or not all(isinstance(v, str) for v in raw["reasons"])
        ):
            return EVIDENCE_STALL_INVALID
    except (TypeError, KeyError):
        return EVIDENCE_STALL_INVALID
    # Contradictory or open-set content (qualifies with reasons, an unknown
    # reason code, duplicate rows) is corrupt history, never a qualification.
    if not stall_snapshot_is_consistent(raw):
        return EVIDENCE_STALL_INVALID
    return {
        "version": 1,
        "review_round": raw["review_round"],
        "review_head": raw["review_head"],
        "matrix_identity": raw["matrix_identity"],
        "qualifies": raw["qualifies"],
        "unsatisfied_row_ids": sorted(raw["unsatisfied_row_ids"]),
        "reasons": sorted(raw["reasons"]),
        # The attempt pairing (#1325); absent on legacy snapshots, which then
        # mean "attempt unknown" and never count toward conversion.
        **(
            {"bound_attempt_round": raw["bound_attempt_round"]}
            if "bound_attempt_round" in raw
            else {}
        ),
    }


def _decode_conversion_fields(payload: Mapping[str, object]) -> dict[str, object]:
    """Decode the #1325 fields; an ill-typed value is absent, never fatal."""
    try:
        return _decode_conversion_fields_strict(payload)
    except (TypeError, ValueError, KeyError, AttributeError):
        logger.warning("Ignoring ill-typed evidence-conversion fields on a round record")
        return {
            key: (() if key == "signed_evidence_references" else None)
            for key in (
                "risk_test_matrix_capture_classification", "evidence_conversion",
                "conversion_stall_snapshot", "conversion_review_round",
                "signed_evidence_references",
            )
            if key in payload
        }


def _decode_conversion_fields_strict(payload: Mapping[str, object]) -> dict[str, object]:
    """Decode the #1325 conversion fields; malformed values never raise.

    A malformed classification, audit, or snapshot decodes as absent (a
    non-qualifying attempt or an unconverted row); a malformed reference is
    dropped.  None of them can ever grant authority by being malformed.
    """
    fields: dict[str, object] = {}
    if "risk_test_matrix_capture_classification" in payload:
        evidence = payload.get("risk_test_matrix_evidence")
        identity = evidence.get("matrix_identity") if isinstance(evidence, Mapping) else None
        subject = payload.get("subject")
        fields["risk_test_matrix_capture_classification"] = decode_capture_classification(
            payload["risk_test_matrix_capture_classification"],
            subject=subject if isinstance(subject, str) else None,
            evidence_identity=identity if isinstance(identity, str) else None,
        )
        if fields["risk_test_matrix_capture_classification"] is None:
            logger.warning("Ignoring a malformed or mis-bound capture classification")
    if "evidence_conversion" in payload:
        audit = EvidenceConversionAudit.from_payload(payload["evidence_conversion"])
        if audit is None or audit.head != payload.get("subject"):
            logger.warning("Ignoring a malformed evidence conversion audit record")
            audit = None
        fields["evidence_conversion"] = audit
    if "conversion_stall_snapshot" in payload:
        raw = payload["conversion_stall_snapshot"]
        decoded = _decode_evidence_stall(
            {"evidence_stall": raw, "round_number": (
                payload["round_number"] + 1
                if isinstance(payload.get("round_number"), int)
                and not isinstance(payload.get("round_number"), bool)
                else None
            )}
        )
        fields["conversion_stall_snapshot"] = decoded
    if "conversion_review_round" in payload:
        value = payload["conversion_review_round"]
        fields["conversion_review_round"] = (
            value
            if isinstance(value, int)
            and not isinstance(value, bool)
            and value == payload.get("round_number")
            else None
        )
    if "signed_evidence_references" in payload:
        raw_references = payload["signed_evidence_references"]
        references: list[SignedEvidenceReference] = []
        if isinstance(raw_references, list):
            for entry in raw_references:
                reference = SignedEvidenceReference.from_payload(entry)
                if reference is None:
                    logger.warning("Ignoring a malformed signed-evidence reference")
                    continue
                if reference.reference_id not in {ref.reference_id for ref in references}:
                    references.append(reference)
        fields["signed_evidence_references"] = tuple(references)
    return fields


def _is_followup_dispatch_head(value: object) -> bool:
    return isinstance(value, str) and _FOLLOWUP_DISPATCH_HEAD_RE.fullmatch(value) is not None


def _decode_followup_dispatch_head(payload: Mapping[str, object]) -> str | None:
    # Absent is historical; a present malformed value is rejected rather than
    # decoded as absence, so it can never silently drop the head signal.
    if "followup_dispatch_head" not in payload:
        return None
    value = payload["followup_dispatch_head"]
    if not _is_followup_dispatch_head(value):
        raise ValueError("followup_dispatch_head must be a hex Git commit SHA")
    return str(value)


STEP_BACK_PHASES = frozenset({"plan", "pr"})


def step_back_entry_is_valid(entry: object) -> bool:
    """Whether ``entry`` is a well-formed reviewer-owned step-back entry (#1251)."""
    if not isinstance(entry, Mapping):
        return False
    allowed = {"phase", "reviewer", "trigger_round", "trigger_head", "anchor"}
    if not set(entry) <= allowed:
        return False
    phase = entry.get("phase")
    reviewer = entry.get("reviewer")
    trigger_round = entry.get("trigger_round")
    if not isinstance(phase, str) or phase not in STEP_BACK_PHASES:
        return False
    if not isinstance(reviewer, str) or not reviewer.strip():
        return False
    if isinstance(trigger_round, bool) or not isinstance(trigger_round, int) or trigger_round < 1:
        return False
    if phase == "plan":
        return "trigger_head" not in entry and "anchor" not in entry
    if not _is_followup_dispatch_head(entry.get("trigger_head")):
        return False
    anchor = entry.get("anchor")
    if not isinstance(anchor, Mapping) or set(anchor) != {"path", "start", "end"}:
        return False
    start, end = anchor.get("start"), anchor.get("end")
    return (
        isinstance(anchor.get("path"), str)
        and bool(str(anchor["path"]).strip())
        and all(isinstance(value, int) and not isinstance(value, bool) for value in (start, end))
        and 1 <= start <= end
    )


def _decode_step_back_fields(payload: Mapping[str, object]) -> dict[str, object]:
    """Decode the optional step-back entries; malformed degrades, never raises."""
    if "step_back_entries" not in payload:
        return {}
    value = payload["step_back_entries"]
    if (
        not isinstance(value, list)
        or not value
        or not all(step_back_entry_is_valid(entry) for entry in value)
    ):
        return {"step_back_status": "invalid"}
    return {
        "step_back_entries": tuple(dict(entry) for entry in value),
        "step_back_status": "valid",
    }


def _decode_plan_sweep(payload: Mapping[str, object]) -> bool:
    """Decode the optional plan sweep marker (#1505); malformed is absent, never raises."""
    if "plan_sweep" not in payload:
        return False
    if (
        payload["plan_sweep"] is True
        and payload.get("flow") == "plan"
        and payload.get("role") == "reviewer"
    ):
        return True
    logger.warning("Ignoring a malformed plan_sweep marker on a persisted round record")
    return False


def _decode_provenance_baseline(payload: Mapping[str, object]) -> dict[str, object]:
    """Decode the optional provenance baseline record (#1516); malformed is absent."""
    from .review_step_back import BASELINE_FORMS

    if "provenance_baseline" not in payload:
        return {}
    value = payload["provenance_baseline"]
    if (
        isinstance(value, Mapping)
        and set(value) == {"form", "chars"}
        and isinstance(value.get("form"), str)
        and value["form"] in BASELINE_FORMS
        and isinstance(value.get("chars"), int)
        and not isinstance(value["chars"], bool)
        and value["chars"] >= 0
        and payload.get("flow") == "plan"
        and payload.get("role") == "reviewer"
    ):
        return {
            "provenance_baseline_form": value["form"],
            "provenance_baseline_chars": value["chars"],
        }
    logger.warning("Ignoring a malformed provenance_baseline field on a persisted round record")
    return {}


def _decode_evidence_fields(payload: Mapping[str, object]) -> dict[str, object]:
    """Decode the optional #1068 evidence fields; absent is legacy.

    A present request list or clearance list that is malformed rejects the
    record rather than decoding as absence, so a persisted request can never be
    silently dropped.  Freeze/release payloads never raise: a malformed payload
    decodes as an invalid record that recovery turns into a blocker.
    """
    fields: dict[str, object] = {}
    from .review_images_coverage_history import decode_fields
    fields.update(decode_fields(payload))
    from .requirement_evidence import decode_associations
    fields["requirement_evidence"] = decode_associations(payload)
    if "evidence_requests" in payload:
        requests = _decode_string_list(payload["evidence_requests"])
        if requests is None:
            raise ValueError("evidence_requests must be a list of non-empty strings")
        fields["evidence_requests"] = requests
    if "evidence_freeze" in payload:
        fields["evidence_freeze"] = EvidenceFreezeRecord.from_mapping(payload["evidence_freeze"])
    if "evidence_release" in payload:
        fields["evidence_release"] = EvidenceReleaseRecord.from_mapping(payload["evidence_release"])
    if "evidence_clearances" in payload:
        raw = payload["evidence_clearances"]
        if not isinstance(raw, list) or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(value, str) and value for value in pair)
            for pair in raw
        ):
            raise ValueError("evidence_clearances must be a list of [identity, head] pairs")
        fields["evidence_clearances"] = tuple((pair[0], pair[1]) for pair in raw)
    return fields


def _decode_plan_supersession_field(payload: Mapping[str, object], key: str) -> str | None:
    # A present-but-malformed binding must not decode as legacy absence.
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _decode_parent_requirement_ids(payload: Mapping[str, object]) -> tuple[str, ...] | None:
    # A present-but-malformed evidence list must not decode as legacy absence.
    if "parent_requirement_ids" not in payload:
        return None
    value = payload["parent_requirement_ids"]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("parent_requirement_ids must be a list of strings")
    return tuple(value)


def _decode_board_object(payload: Mapping[str, object], key: str) -> dict | None:
    # A present-but-malformed board field must not decode as legacy absence.
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _decode_matrix_evidence_full_round(payload: Mapping[str, object]) -> dict[str, object]:
    # Missing is legacy absence; a present malformed value never raises and
    # stays distinguishable from absence so it cannot claim the legacy rule.
    if "risk_test_matrix_evidence_full_round" not in payload:
        return {"risk_test_matrix_evidence_full_round_status": "absent"}
    value = payload["risk_test_matrix_evidence_full_round"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return {"risk_test_matrix_evidence_full_round_status": "invalid"}
    return {
        "risk_test_matrix_evidence_full_round": value,
        "risk_test_matrix_evidence_full_round_status": "valid",
    }


def _decode_round_metadata_mapping(payload: Mapping[str, object]) -> PostedRoundMetadata:
    try:
        if "seat_binding" in payload and not isinstance(payload["seat_binding"], dict):
            raise ValueError("seat_binding must be an object")
        for key in ("prior_items", "risk_test_matrix_evidence"):
            # A spill reference must be hydrated first; never read it as
            # review items or canonical matrix evidence (#953).
            value = payload.get(key)
            if isinstance(value, Mapping) and "$round_transport_spill" in value:
                raise ValueError(f"{key} is an unhydrated transport reference")
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
        from .post_merge_state import decode_fields as decode_post_merge
        return PostedRoundMetadata(
            **decode_post_merge(payload),
            flow=str(payload["flow"]),
            role=str(payload["role"]),
            agent=str(payload["agent"]),
            seat_binding=(
                payload["seat_binding"]
                if isinstance(payload.get("seat_binding"), dict)
                else None
            ),
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
            architecture_impact_degradations=_decode_parse_degradations(
                payload.get("architecture_impact_degradations")
            ),
            test_observation_degradations=_decode_parse_degradations(
                payload.get("test_observation_degradations")
            ),
            addressed_sub_items=(
                tuple(
                    entry for entry in payload["addressed_sub_items"] if isinstance(entry, str)
                )
                if isinstance(payload.get("addressed_sub_items"), list) else ()
            ),
            sub_item_claim_degradations=_decode_parse_degradations(
                payload.get("sub_item_claim_degradations")
            ),
            sub_item_degradations=_decode_parse_degradations(
                payload.get("sub_item_degradations")
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
            parent_requirement_ids=_decode_parent_requirement_ids(payload),
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
            plan_candidate_key=(
                payload["plan_candidate_key"]
                if isinstance(payload.get("plan_candidate_key"), dict)
                else None
            ),
            plan_supersession_digest=_decode_plan_supersession_field(
                payload, "plan_supersession_digest"
            ),
            plan_supersession_superseded_hash=_decode_plan_supersession_field(
                payload, "plan_supersession_superseded_hash"
            ),
            reviewer_board_amendment_digest=_decode_plan_supersession_field(
                payload, "reviewer_board_amendment_digest"
            ),
            reviewer_board_change_digest=_decode_plan_supersession_field(
                payload, "reviewer_board_change_digest"
            ),
            pr_board_root=_decode_board_object(payload, "pr_board_root"),
            pr_board_identity=_decode_board_object(payload, "pr_board_identity"),
            pr_board_binding_init=_decode_board_object(payload, "pr_board_binding_init"),
            plan_board_root=_decode_board_object(payload, "plan_board_root"),
            plan_board_identity=_decode_board_object(payload, "plan_board_identity"),
            plan_board_binding_init=_decode_board_object(payload, "plan_board_binding_init"),
            followup_dispatch_head=_decode_followup_dispatch_head(payload),
            evidence_stall=_decode_evidence_stall(payload),
            **_decode_conversion_fields(payload),
            **decode_review_scope(payload),
            plan_execution_mode=_decode_execution_mode(payload.get("plan_execution_mode")),
            plan_issue_digest=_decode_plan_issue_digest(payload),
            **_decode_recovery_fields(payload),
            **_decode_step_back_fields(payload),
            plan_sweep=_decode_plan_sweep(payload),
            **_decode_provenance_baseline(payload),
            **_decode_evidence_fields(payload),
            **_decode_matrix_evidence_full_round(payload),
            **_decode_scheduler_fields(payload),
        )
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise AgentLoopError(f"Invalid AGENT_LOOP_META payload: {exc}") from exc


def _decode_plan_issue_digest(payload: Mapping[str, object]) -> str | None:
    """Tolerant decode (#1390): only a 16-hex digest on a plan-flow record survives."""
    value = payload.get("plan_issue_digest")
    if (
        payload.get("flow") == "plan"
        and isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{16}", value)
    ):
        return value
    return None


def prior_plan_execution_mode(records: Sequence[PostedRoundRecord]) -> str | None:
    """The mode on the newest plan-flow record that recorded one (#1268)."""
    for record in reversed(tuple(records)):
        metadata = record.metadata
        if metadata.flow != "plan":
            continue
        mode = metadata.plan_execution_mode or metadata.scheduler_execution_mode
        if mode is not None:
            return mode
    return None


def _decode_round_metadata(encoded: str) -> PostedRoundMetadata:
    return _decode_round_metadata_mapping(decode_mapping(encoded))


def followup_head_unchanged_sha(metadata: PostedRoundMetadata | None) -> str | None:
    """Return the PR head a coder follow-up left unchanged, else ``None`` (#1034).

    Only a persisted dispatch head equal to the result subject qualifies.
    """
    if metadata is None or metadata.role != "coder":
        return None
    dispatch_head = metadata.followup_dispatch_head
    subject = metadata.subject
    if not dispatch_head or not subject or subject == "unknown":
        return None
    return subject if subject == dispatch_head else None


# The PR board stamp of the running PR loop (#1376).  Every PR round record is
# encoded here, so stamping at this one point covers every writer.
_PR_BOARD_STAMP: "contextvars.ContextVar[object | None]" = contextvars.ContextVar(
    "pr_board_stamp", default=None
)


def set_pr_board_stamp(stamp: object | None) -> "contextvars.Token":
    return _PR_BOARD_STAMP.set(stamp)


def reset_pr_board_stamp(token: "contextvars.Token") -> None:
    _PR_BOARD_STAMP.reset(token)


# The plan board stamp of the running plan loop (#1377), applied only to
# plan-flow records at the same single encoder.
_PLAN_BOARD_STAMP: "contextvars.ContextVar[object | None]" = contextvars.ContextVar(
    "plan_board_stamp", default=None
)


def set_plan_board_stamp(stamp: object | None) -> "contextvars.Token":
    return _PLAN_BOARD_STAMP.set(stamp)


def reset_plan_board_stamp(token: "contextvars.Token") -> None:
    _PLAN_BOARD_STAMP.reset(token)


def _attach_round_metadata(body: str, metadata: PostedRoundMetadata) -> str:
    stamp = _PR_BOARD_STAMP.get()
    if stamp is not None and metadata.flow == "pr":
        metadata = stamp.apply(metadata)  # type: ignore[attr-defined]
    plan_stamp = _PLAN_BOARD_STAMP.get()
    if plan_stamp is not None and metadata.flow == "plan":
        metadata = plan_stamp.apply(metadata)  # type: ignore[attr-defined]
    marker = f"<!-- AGENT_LOOP_META: {_encode_round_metadata(metadata)} -->"
    degradations = render_parse_degradations_section(metadata.architecture_impact_degradations)
    if degradations is not None and degradations not in str(body):
        # Surface degraded elements in the round summary so a degraded round
        # never reads as fully assessed.
        body = _insert_before_trailing_markers(str(body), degradations)
    citation_degradations = render_test_observation_degradations_section(
        metadata.test_observation_degradations
    )
    if citation_degradations is not None and citation_degradations not in str(body):
        body = _insert_before_trailing_markers(str(body), citation_degradations)
    sub_item_records = render_parse_degradations_section(
        metadata.sub_item_degradations, heading="### Sub-item parse degradations"
    )
    if sub_item_records is not None and sub_item_records not in str(body):
        body = _insert_before_trailing_markers(str(body), sub_item_records)
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


def _insert_before_trailing_markers(body: str, section: str) -> str:
    lines = body.splitlines()
    index = len(lines)
    while index > 0:
        candidate = lines[index - 1]
        if not candidate.strip() or HTML_COMMENT_RE.match(candidate) or SIGNATURE_RE.match(candidate):
            index -= 1
            continue
        break
    prefix = "\n".join(lines[:index]).rstrip("\n")
    suffix = "\n".join(lines[index:]).lstrip("\n")
    return "\n\n".join(part for part in (prefix, section, suffix) if part)


def _strip_round_metadata(body: str) -> str:
    cleaned = re.sub(
        r"\n?\s*<!--\s*AGENT_LOOP_META:\s*[A-Za-z0-9+/=_-]+\s*-->\s*\n?",
        "\n",
        body,
        flags=re.I,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()



def _plan_subject(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
