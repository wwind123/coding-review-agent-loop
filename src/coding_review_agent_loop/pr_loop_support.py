"""PR-loop helpers: exact-head merge proof, finalization, evidence freeze,
qualification snapshot and transition observation.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1199); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

from .managed_ci_result import VerifiedManagedResult

import hashlib
import json
import re
import unicodedata
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import (
    dataclass,
    replace as dataclasses_replace,
)
from pathlib import Path
from typing import Literal, NoReturn
from .local_test_evidence import tool_identity_clause, unsuperseded_receipts_sentence
from .agents.base import AgentName
from .agents.registry import agent_display_name
from .reviewer_seats import reviewer_seat_binding
from .config import (
    AgentLoopConfig,
    reviewers,
)
from .board_amendment import (
    collect_reviewer_board_amendments,
    reject_misplaced_pr_amendments,
)
from .decomposition import approved_plan_hash
from .errors import (
    AgentLoopError,
    HumanDecisionRequiredError,
    IssueImplementationConflictError,
)
from .github_retry import describe_gh_failure
from .github import (
    _classify_check_status,
    IssueContext,
    PullRequestMetadata,
    PullRequestCheck,
    PullRequestChecks,
    PullRequestMergeability,
    PullRequestReviewContext,
    reconciled_pr_ready,
    board_protection_is_reliable,
    protection_awaits_readiness,
    get_pr_head_sha,
    get_issue_context,
    get_pr_mergeability,
    parse_linked_issue_numbers,
    get_pr_review_context,
    merge_pr,
    merge_pr_comment_transport_identity,
    post_pr_comment,
    resolve_authenticated_github_actor,
    strip_bot_login_suffix,
)
from .issue_pr_handoff import find_latest_issue_pr_handoff
from .logging import log
from . import evidence_stall as _evidence_stall
from . import evidence_conversion as _evidence_conversion
from .evidence_conversion import (
    ConversionAuditRow,
    EvidenceConversionAudit,
)
from .managed_ci import (
    AuthenticatedIssueCreatedHandoff,
    FINAL_CONTEXT,
    ManagedCiContract,
    ManagedCiOutcome,
    OrdinaryRecoveryCapability,
    refresh_ordinary_recovery_capability,
    require_recorded_base,
    render_managed_ci_resume_command,
    validate_ordinary_recovery_capability,
    verify_managed_pr_plan_binding,
    wait_for_ordinary_recovery,
)
from .protocol import (
    ReviewItemDisposition,
    CITATION_DROPPED_OUTCOME,
    TEST_OBSERVATIONS_NOT_ARRAY_AFTER_REPAIR_RULE,
    TEST_OBSERVATIONS_NOT_ARRAY_RULES,
    ApprovedFollowup,
    ParsedReview,
    StructuredCoderFollowup,
    StructuredIssueImplementation,
    UnresolvedReviewItem,
    CI_MACHINE_OBLIGATION_KINDS,
    EVIDENCE_SOURCE_OPERATOR_SIGNED,
    SignedEvidenceReference,
    apply_signed_evidence_references,
    parse_risk_test_matrix_evidence,
    MACHINE_OBLIGATION_KINDS,
    parse_historical_structured_coder_followup,
    parse_historical_structured_issue_implementation,
)
from .runner import Runner
from . import review_step_back as _step_back
from .usage import RunUsageContext
from .workdirs import active_workdir
from .workdir_guard import (
    read_workdir_head,
    partition_reported_tests_by_workdir,
)
from .checks import (
    _ci_plumbing_stop_message,
    final_context_state_line,
    _pending_ci_stop_message,
    _pr_check_details,
    _unreadable_protection_stop_message,
)
from .comment_rendering import (
    render_sub_item_progress_comment,
    sub_item_progress_digest,
    sub_item_progress_record_keys,
)
from .followups import (
    _publish_approved_followups,
    FollowupSourceContext,
)
from .round_state import (
    ApprovedPlanContext,
    EVIDENCE_FREEZE_PHASE,
    EVIDENCE_CONVERSION_PHASE,
    admit_evidence_authority_records,
    record_carries_evidence_authority,
    EVIDENCE_RELEASE_PHASE,
    EvidenceFreezeRecord,
    EvidenceReleaseRecord,
    QualificationCheckpoint,
    PostedRoundMetadata,
    followup_head_unchanged_sha,
    rebuild_resumed_coder_carrier,
    PostedRoundRecord,
    _attach_round_metadata,
    _extract_round_metadata_records,
    _is_followup_dispatch_head,
    _prior_item_ledger_signature,
    _resume_plan_round,
    scope_approved_plan_matrix,
    recover_approved_plan_context,
    _resume_pr_round,
    CODER_DISPATCH_PHASE,
    CODER_FOLLOWUP_REJECTED_PHASE,
    HEAD_REVIEW_RECOVERY_PHASE,
    RecoveryRoundBudget,
    ResumedReviewRound,
    _recovery_record_problems,
    pr_resume_needs_author_admission,
    unauthenticated_recovery_record_indexes,
    sanitize_recovery_reason,
)
from .protocol_markers import (
    TrustedBody,
    sanitize_historical_text,
)
from .review_scheduling import (
    GitChange,
    ReviewObligation,
    ReviewSchedulingContract,
    TransitionClassification,
    classify_transition,
)
from .unresolved_items import (
    ClearedItemProgress,
    newly_stalled_items,
    render_sub_item_progress_summary,
    sub_item_progress,
    CODER_DISPUTE_NOTE_PREFIX,
    _is_machine_obligation,
    _machine_obligation_is_revalidation_candidate,
    _machine_obligation_requires_repair,
    coder_followup_is_ci_repair,
    _frozen_evidence_obligations,
    _is_evidence_obligation,
    _pending_evidence_obligations,
    freeze_evidence_obligations,
    _is_row_converted_obligation,
    _upsert_converted_evidence_obligation,
    apply_residual_row_conversion,
    converted_evidence_obligation_identity,
    converted_row_owners,
    release_evidence_freeze,
)
from .architecture_contract import (
    _architecture_metadata_fields,
    _latest_pr_approval_architecture_identity,
    _latest_pr_architecture_observation,
    _revalidate_pr_architecture_identity,
)
from .response_validation import _build_requirements_context
from .panel_evidence import (
    _board_amendment_route_clause,
    _stale_amendment_repost_clause,
    _require_complete_canonical_plan_approval,
)
from .child_plan_binding import (
    _PlanningChildBinding,
    verify_child_plan_rebind,
    _reject_pending_child_plan_supersession,
)


@dataclass(frozen=True)
class ExactHeadCiProof:
    """Non-empty current-head CI evidence required by automated merge paths."""

    head_sha: str
    source: str
    # Qualification is bound to the base it was granted for (#1285); empty for
    # sources that predate the recorded-base guard.
    base_ref: str | None = None
    repository: str | None = None
    managed_result: VerifiedManagedResult | None = None


def _render_ci_rerun_command(config: AgentLoopConfig, *, pr_number: int) -> str:
    """Render local CI recovery guidance through the shared token builder."""
    return render_managed_ci_resume_command(
        config,
        pr_number=pr_number,
        managed_ci=config.managed_ci,
        preserve_managed_options=(
            config.managed_ci
            or config.managed_ci_trusted_actor is not None
            or config.allow_unprotected_managed_ci
            or config.allow_unreadable_protection
            or config.managed_ci_adopt_existing_pr
        ),
        include_context=False,
    )


def _print_unprotected_managed_ci_warning(protection_mode: str) -> None:
    message = (
        "WARNING: --allow-unprotected-managed-ci is active for this invocation. GitHub cannot "
        "prevent a manual merge, other automation, a compromised credential, or an agent-loop "
        "defect from bypassing the voluntary final-ci/exact-head gate."
    )
    if protection_mode == "unreadable":
        message += (
            " --allow-unreadable-protection is also active: classic branch protection could not "
            "be read by this token, so the exact-head gate is treated as voluntary for this "
            "invocation."
        )
    print(message)


def _ready_failure_suffix(result) -> str:
    detail = describe_gh_failure(result)
    return f"\n{detail}" if detail else ""


def _merge_with_exact_head_proof(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    proof: ExactHeadCiProof,
) -> None:
    """Make the live-head read the final remote operation before merging."""
    # Fetch a fresh, minimal head as the final normal remote read. If GitHub
    # serves an inconsistent GraphQL projection while it is converging, fetch
    # the full live PR tuple and require that authoritative view to agree with
    # the proof too; neither cached review metadata nor an old check board is
    # accepted.
    if proof.managed_result is not None:
        try:
            proof.managed_result.require_binding(repository=config.repo, pr_number=pr_number,
                                                head_sha=proof.head_sha, base_ref=proof.base_ref)
        except ValueError as exc:
            raise AgentLoopError(str(exc)) from exc
    live_head = get_pr_head_sha(runner, config, pr_number)
    if live_head != proof.head_sha:
        live_head = get_pr_review_context(
            runner, config=config, pr_number=pr_number
        ).metadata.head_sha
    if live_head != proof.head_sha:
        raise AgentLoopError(
            f"PR #{pr_number} head changed after {proof.source} CI proof; no merge attempted."
        )
    require_recorded_base(
        runner, config=config, pr_number=pr_number, base_ref=proof.base_ref,
        repository=proof.repository, head_sha=proof.head_sha, action="merge",
    )
    merge_pr(runner, config, pr_number, expected_head_sha=proof.head_sha)


def _read_assigned_workdir_head(runner: Runner, config: AgentLoopConfig) -> str | None:
    # The PR handoff guard intentionally keeps strict exception behavior: a
    # runner/tooling exception must not be silently converted into evidence
    # that the coder advanced the assigned checkout. Ordinary Git failures
    # and blank HEAD output remain an unavailable (None) observation.
    probe = read_workdir_head(runner, active_workdir(config))
    return probe.value if probe.available else None


# Consecutive PR follow-up coder turns allowed to leave the head unchanged
# before the loop stops instead of re-reviewing an identical diff (#985).
MAX_UNCHANGED_HEAD_CODER_TURNS = 2


class _UnchangedHeadTracker:
    """Count consecutive coder follow-ups that left one PR head unchanged (#985).

    The count belongs to a single head: a follow-up on any other head,
    including one advanced externally between rounds, starts a fresh count,
    and a follow-up that moves the head clears it.

    Each tracked head also has one evidence-progress grant (#1626): one extra
    review round when the follow-up at the limit made cited, authoritative
    evidence progress.  The grant resets whenever the tracked head changes,
    so the extra rounds are bounded at one per head.  Like the count, it is
    in-memory only.
    """

    def __init__(self) -> None:
        self.head_sha: str | None = None
        self.count = 0
        self.evidence_progress_granted = False

    def observe(self, reviewed_head: str | None, head_after_followup: str | None) -> int:
        previous_head = self.head_sha
        if not reviewed_head or head_after_followup != reviewed_head:
            self.head_sha, self.count = None, 0
        elif reviewed_head == self.head_sha:
            self.count += 1
        else:
            self.head_sha, self.count = reviewed_head, 1
        if self.head_sha != previous_head:
            self.evidence_progress_granted = False
        return self.count

    def grant_evidence_progress(self) -> bool:
        """Spend this head's one evidence-progress grant; False once it is used."""
        if self.head_sha is None or self.evidence_progress_granted:
            return False
        self.evidence_progress_granted = True
        return True


@dataclass(frozen=True)
class FollowupCitationDiagnosis:
    """What happened to one accepted follow-up's test_observations (#1626).

    ``uncited_receipts`` counts the distinct receipts in the coder
    acquisition's frozen catalog that no surviving citation cites; it never
    reads the bounded round evidence or the runner's mutable latest turn.
    """

    non_array_drop: bool
    repair_attempted: bool
    uncited_receipts: int


def followup_citation_diagnosis(
    parsed: object, coder_turn_receipt_ids: Iterable[str]
) -> FollowupCitationDiagnosis | None:
    """Diagnose an accepted follow-up's citations against its frozen catalog."""
    if not isinstance(parsed, StructuredCoderFollowup):
        return None
    non_array = [
        record
        for record in parsed.test_observation_degradations
        if record.outcome == CITATION_DROPPED_OUTCOME
        and record.rule in TEST_OBSERVATIONS_NOT_ARRAY_RULES
        and record.element_path == "coder_followup.test_observations"
    ]
    cited = {citation.receipt_id for citation in parsed.test_observations}
    return FollowupCitationDiagnosis(
        non_array_drop=bool(non_array),
        repair_attempted=any(
            record.rule == TEST_OBSERVATIONS_NOT_ARRAY_AFTER_REPAIR_RULE for record in non_array
        ),
        uncited_receipts=len(frozenset(coder_turn_receipt_ids) - cited),
    )


def cited_evidence_progress(
    parsed: object,
    superseding_receipts: Iterable[str],
) -> frozenset[str]:
    """Superseding current-turn receipts that a surviving citation also cites (#1626).

    Live reconciliation supersedes a failure from any compatible pass, cited
    or not; the grant also requires that the pass was presented to
    reviewers, so a failed repair (citations dropped) never earns it.
    """
    if not isinstance(parsed, StructuredCoderFollowup):
        return frozenset()
    cited = {citation.receipt_id for citation in parsed.test_observations}
    return frozenset(superseding_receipts) & cited


def _coder_followup_head_log(
    round_number: int,
    coder_name: str,
    dispatch_head: str | None,
    observed_head: str | None,
    unchanged_count: int,
) -> str:
    """Describe a PR coder follow-up by its observed head change (#1034).

    The refetch cannot prove who moved the head, so an advance is reported
    actor-neutrally, and an unknown head makes no push or unchanged claim.
    """
    if not dispatch_head or not observed_head:
        return (
            f"Round {round_number}: {coder_name} follow-up complete; "
            "PR head change could not be determined"
        )
    if dispatch_head != observed_head:
        return (
            f"Round {round_number}: PR head advanced {dispatch_head[:12]}..{observed_head[:12]} "
            f"during {coder_name} follow-up; re-reviewing"
        )
    return (
        f"Round {round_number}: {coder_name} follow-up left PR head {observed_head[:12]} "
        f"unchanged ({unchanged_count}/{MAX_UNCHANGED_HEAD_CODER_TURNS}); re-reviewing"
    )


def _coder_followup_review_context(
    text: str | None,
    metadata: PostedRoundMetadata | None,
    *,
    head_sha: str | None,
    assigned_workdir: Path | None = None,
    coverage_map: str | None = None,
    signed_evidence_references: Sequence[SignedEvidenceReference] = (),
) -> str:
    if not text or metadata is None:
        return ""
    if not head_sha or metadata.subject != head_sha:
        return (
            "Latest coder explanation omitted: its recorded head does not match "
            "the current PR head. Do not treat earlier fix claims as current evidence.\n"
        )
    summary = _extract_structured_coder_summary(text)
    tests = _extract_structured_coder_tests_run(text)
    out_of_checkout_tests: tuple[str, ...] = ()
    if tests and assigned_workdir is not None:
        # The persisted response keeps the coder's raw list; reapply the same
        # classification the public comment used so the reviewer never sees
        # an out-of-checkout baseline as an ordinary test run (#991).
        try:
            partition = partition_reported_tests_by_workdir(
                tests, assigned_workdir=assigned_workdir
            )
        except AgentLoopError:
            tests, out_of_checkout_tests = (), tuple(tests)
        else:
            tests, out_of_checkout_tests = partition.in_checkout, partition.out_of_checkout
    # Dropped-citation records are restored from the round metadata (#927).
    parsed = rebuild_resumed_coder_carrier(text, metadata)
    payload: dict[str, object] = {
        "summary": summary,
        "tests_run": tests,
        "local_test_evidence": metadata.local_test_evidence,
    }
    if out_of_checkout_tests:
        payload["out_of_checkout_context_runs_not_evidence"] = out_of_checkout_tests
    # Reviewers receive the orchestrator-derived authority carried by the
    # same round metadata as the public comment. Do not reconstruct it from
    # the coder's fresh response or from the cumulative local journal.
    if metadata.risk_test_matrix_evidence is not None:
        payload["risk_test_matrix_evidence"] = _overlay_signed_evidence_payload(
            metadata.risk_test_matrix_evidence, signed_evidence_references, head_sha=head_sha
        )
    if metadata.risk_test_matrix_diagnostics:
        payload["risk_test_matrix_diagnostics"] = [
            dict(item) for item in metadata.risk_test_matrix_diagnostics
        ]
    if coverage_map:
        # Orchestrator-authored completeness map (#1290); display-only context.
        payload["risk_matrix_coverage_map"] = coverage_map
    if isinstance(parsed, StructuredCoderFollowup):
        payload.update(
            addressed_items=parsed.addressed_items,
            addressed_item_notes=parsed.addressed_item_notes,
            remaining_items=parsed.remaining_items,
            remaining_item_notes=parsed.remaining_item_notes,
            disputed_items=parsed.disputed_items,
            dispute_evidence=parsed.dispute_evidence,
            test_observations=[
                {
                    "command": item.command,
                    "receipt_id": item.receipt_id,
                    "claim": item.claim,
                }
                for item in parsed.test_observations
            ],
        )
    if (
        isinstance(parsed, (StructuredCoderFollowup, StructuredIssueImplementation))
        and parsed.test_observation_degradations
    ):
        # A dropped citation supports nothing; reviewers see that it was dropped.
        payload["test_observation_degradations"] = [
            record.to_payload() for record in parsed.test_observation_degradations
        ]
    if isinstance(parsed, StructuredCoderFollowup):
        if parsed.addressed_sub_items:
            # Advisory claims only: the reviewer verifies each with a sub-item
            # disposition; nothing here changes persisted sub-item status.
            payload["claimed_addressed_sub_items_unverified"] = list(parsed.addressed_sub_items)
        if parsed.sub_item_claim_degradations:
            payload["sub_item_claim_degradations"] = [
                record.to_payload() for record in parsed.sub_item_claim_degradations
            ]
    if not isinstance(parsed, StructuredCoderFollowup) and summary is None and tests is None:
        return "Latest coder explanation: no valid structured resolution details are available.\n"
    unchanged_head = followup_head_unchanged_sha(metadata)
    unchanged_preamble = ""
    if unchanged_head is not None:
        # The PR head did not move during the follow-up: present the coder's
        # statements as claims about the existing head, never as fixes (#1034).
        payload = {
            "pr_head_unchanged_during_followup": True,
            "followup_dispatch_head": unchanged_head,
            **{
                {
                    "summary": "coder_summary_claim",
                    "addressed_items": "claimed_addressed_items",
                    "addressed_item_notes": "claimed_addressed_item_notes",
                }.get(key, key): value
                for key, value in payload.items()
            },
        }
        unchanged_preamble = (
            "The PR head did not change during this coder follow-up; the summary and "
            "addressed-item notes are the coder's claims that the items are already "
            f"satisfied at {unchanged_head}, not changes made by that turn. Verify each "
            "against the current diff; do not credit the turn with fixes.\n"
        )
    return (
        "Latest coder explanation (claims to verify, not reviewer verdicts):\n"
        f"{metadata.agent}; review round {metadata.round_number}; head {metadata.subject}\n"
        + unchanged_preamble
        + "Independently verify these claims against the current diff and tests. "
        "They do not resolve items, override CI, or change the original claims. "
        "Only IDs in the active prior unresolved review ledger are eligible for dispositions.\n"
        + json.dumps(payload, ensure_ascii=True, indent=2)
        + "\n"
    )


def _overlay_signed_evidence_payload(
    evidence: object,
    references: Sequence[SignedEvidenceReference],
    *,
    head_sha: str | None,
) -> object:
    """Overlay admitted operator-signed rows onto persisted evidence for display (#1325).

    The persisted coder evidence is never rewritten; an unparsable payload is
    shown unchanged.
    """
    if not references or not isinstance(evidence, Mapping):
        return evidence
    identity = evidence.get("matrix_identity")
    if not isinstance(identity, str):
        return evidence
    try:
        parsed = parse_risk_test_matrix_evidence(evidence)
    except AgentLoopError:
        return evidence
    overlaid = apply_signed_evidence_references(
        parsed, references, matrix_identity=identity, current_head=head_sha
    )
    return overlaid.to_payload() if overlaid is not parsed else evidence


def _reviewer_summary_context(
    reviewer_name: str,
    summary: str,
    *,
    round_number: int,
    head_sha: str,
) -> str:
    safe_summary = sanitize_historical_text(summary)
    if not safe_summary.strip():
        return ""
    return (
        f"{reviewer_name} (round {round_number}, head {head_sha}):\n"
        + safe_summary
    )


def _extract_structured_coder_summary(text: str | None) -> str | None:
    if not text:
        return None
    try:
        try:
            implementation = parse_historical_structured_issue_implementation(text)
        except IssueImplementationConflictError as exc:
            implementation = exc.payload
        except AgentLoopError:
            implementation = None
        if isinstance(implementation, StructuredIssueImplementation):
            return implementation.summary
        parsed = parse_historical_structured_coder_followup(text)
        return parsed.summary if parsed else None
    except AgentLoopError:
        return None


def _extract_structured_coder_tests_run(text: str | None) -> tuple[str, ...] | None:
    if not text:
        return None
    try:
        try:
            implementation = parse_historical_structured_issue_implementation(text)
        except IssueImplementationConflictError as exc:
            implementation = exc.payload
        except AgentLoopError:
            implementation = None
        if isinstance(implementation, StructuredIssueImplementation):
            return implementation.tests_run
        parsed = parse_historical_structured_coder_followup(text)
        return parsed.tests_run if parsed else None
    except AgentLoopError:
        return None


def _finalize_ordinary_recovery_merge(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    capability: OrdinaryRecoveryCapability,
) -> bool:
    """Qualify, ready, and merge only the draft released by this invocation."""
    refreshed = refresh_ordinary_recovery_capability(
        runner, config=config, capability=capability,
    )
    if refreshed is None:
        raise AgentLoopError(
            f"PR #{pr_number} ordinary recovery provenance changed before finalization; no merge attempted."
        )
    capability = refreshed
    outcome = wait_for_ordinary_recovery(
        runner, config=config, capability=capability,
        metadata=get_pr_review_context(runner, config=config, pr_number=pr_number).metadata,
    )
    if outcome.status != "passed":
        if outcome.status == "not_started":
            command = render_managed_ci_resume_command(
                config, pr_number=pr_number, managed_ci=False,
            )
            log(
                config,
                f"PR #{pr_number}: ordinary recovery CI did not start within the bounded startup window; "
                "leaving the PR draft and unmerged",
            )
            print(
                f"PR #{pr_number} remains draft and unmerged because ordinary recovery CI did not "
                f"materialize for the current head. Resume with `{command}`."
            )
            return False
        if outcome.status == "protection_unreadable":
            merge_state = (
                outcome.mergeability.merge_state_raw
                if outcome.mergeability is not None
                else None
            )
            log(
                config,
                f"PR #{pr_number}: ordinary recovery board is green but branch protection is "
                f"unreadable and the merge state is {merge_state or 'unavailable'}; "
                "leaving the PR draft and unmerged",
            )
            print(
                f"PR #{pr_number} remains draft and unmerged: ordinary recovery CI passed for "
                f"{capability.expected_head_sha}, but the current GitHub token cannot read branch "
                "protection (HTTP 403) and GitHub reports merge state "
                f"{merge_state or 'unavailable'} (DRAFT or CLEAN for the same head is required). "
                "Grant the token administration read access or resolve the merge state, then "
                "rerun agent-loop."
            )
            raise AgentLoopError(
                f"PR #{pr_number} ordinary recovery could not confirm merge readiness: branch "
                "protection is unreadable and GitHub's merge state is neither DRAFT nor CLEAN "
                "for the exact head; the draft was left unmerged."
            )
        raise AgentLoopError(
            f"PR #{pr_number} ordinary recovery did not qualify the exact head "
            f"({outcome.status}); the draft was left unmerged."
        )
    clean_required_after_ready = outcome.checks is not None and protection_awaits_readiness(
        outcome.checks, outcome.mergeability, head_sha=outcome.head_sha,
    )
    if not _ordinary_checks_snapshot_is_authoritative(
        outcome.checks,
        outcome.mergeability,
        head_sha=outcome.head_sha,
        defer_unreadable_protection=clean_required_after_ready,
    ):
        details = (
            _pr_check_details(outcome.checks)
            if outcome.checks is not None
            else ["No authoritative current-head check snapshot was available."]
        )
        log(
            config,
            f"PR #{pr_number}: ordinary recovery reported aggregate passing status "
            "without an authoritative success-only current-head check snapshot; "
            "leaving the PR draft and unmerged",
        )
        print(
            f"PR #{pr_number} remains draft and unmerged because ordinary recovery "
            "did not produce an authoritative success-only current-head check board "
            f"({'; '.join(details)})."
        )
        return False
    if not validate_ordinary_recovery_capability(runner, config=config, capability=capability):
        raise AgentLoopError(
            f"PR #{pr_number} ordinary recovery provenance changed before readiness; no merge attempted."
        )
    ready = reconciled_pr_ready(
        runner, config=config, pr_number=pr_number,
        expected_head_sha=capability.expected_head_sha,
    )
    if ready.returncode != 0:
        raise AgentLoopError(
            f"Unable to mark recovered PR #{pr_number} ready for review."
            f"{_ready_failure_suffix(ready)}"
        )
    if not validate_ordinary_recovery_capability(
        runner, config=config, capability=capability, require_draft=None,
    ):
        raise AgentLoopError(
            f"PR #{pr_number} head or provenance changed after `gh pr ready`; "
            "the PR remains ready and was not merged."
        )
    if clean_required_after_ready:
        assert outcome.checks is not None
        _require_clean_merge_state_after_ready(
            runner,
            config=config,
            pr_number=pr_number,
            checks=outcome.checks,
            head_sha=capability.expected_head_sha,
        )
    try:
        _merge_with_exact_head_proof(
            runner,
            config=config,
            pr_number=pr_number,
            proof=ExactHeadCiProof(
                head_sha=capability.expected_head_sha,
                source="ordinary recovery",
            ),
        )
    except Exception:
        # Do not convert a successfully readied PR back into a draft. A safe
        # rerun can now inspect the ready exact head and retry the merge gate.
        log(config, f"PR #{pr_number}: merge failed after ordinary recovery readiness; PR remains ready")
        raise
    return True


def _require_clean_merge_state_after_ready(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    checks: PullRequestChecks,
    head_sha: str,
) -> None:
    """Require GitHub's CLEAN merge state after a draft recovery is readied.

    Under unreadable (403) classic protection, the draft could only report
    ``DRAFT``.  Once ready, GitHub recomputes the merge state from the real
    protection; poll it for the bounded startup window and refuse to merge
    unless it is ``CLEAN`` for the same exact head.
    """
    attempts = max(
        1,
        (config.ci_startup_timeout_seconds + config.ci_poll_interval_seconds - 1)
        // config.ci_poll_interval_seconds,
    )
    mergeability: PullRequestMergeability | None = None
    for attempt in range(attempts):
        mergeability = get_pr_mergeability(runner, config=config, pr_number=pr_number)
        if board_protection_is_reliable(checks, mergeability, head_sha=head_sha):
            return
        if mergeability.state == "conflicted" or (
            mergeability.head_sha is not None and mergeability.head_sha != head_sha
        ):
            break
        if attempt < attempts - 1:
            runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
    merge_state = mergeability.merge_state_raw if mergeability is not None else None
    log(
        config,
        f"PR #{pr_number}: branch protection is unreadable and the readied PR's merge state "
        f"is {merge_state or 'unavailable'}, not CLEAN; PR remains ready and unmerged",
    )
    print(
        f"PR #{pr_number} was marked ready after ordinary recovery CI passed, but the current "
        "GitHub token cannot read branch protection (HTTP 403) and GitHub reports merge state "
        f"{merge_state or 'unavailable'} (CLEAN is required) for {head_sha}. Satisfy the remaining "
        "protection rules or grant the token administration read access, then merge manually "
        f"with `--match-head-commit {head_sha}` or rerun agent-loop."
    )
    raise AgentLoopError(
        f"PR #{pr_number} branch protection is unreadable and GitHub's merge state is not CLEAN "
        "after readiness; the PR remains ready and was not merged."
    )


def _stop_on_terminal_without_status(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    outcome: ManagedCiOutcome,
) -> int:
    conclusion = outcome.workflow_conclusion or "unknown"
    attempt_text = (
        f"run `{outcome.run_id}` attempt `{outcome.run_attempt}`"
        if outcome.run_id is not None
        else "the correlated managed-CI attempt"
    )
    body = (
        f"PR #{pr_number} managed exact-head CI stopped because {attempt_text} "
        f"reached terminal workflow state `{conclusion}` without publishing a "
        f"correlated `{FINAL_CONTEXT}` status. No terminal status was synthesized "
        "and no merge was attempted.\n\n"
        "The round is resumable: for the unchanged head, rerun the command after "
        "a legitimate GitHub rerun creates a higher attempt, or rerun it to dispatch "
        "a fresh eligible same-nonce run. If the head was corrected, restart exact-head "
        "review so a new ledger is created/used."
    )
    post_pr_comment(runner, config=config, pr_number=pr_number, body=body)
    log(
        config,
        f"Round {round_number}: managed CI reached terminal state without "
        "publishing the correlated exact-head status; no merge attempted",
    )
    print(
        f"PR #{pr_number} managed exact-head CI reached terminal workflow state "
        f"`{conclusion}` without publishing its correlated status. No merge was "
        "attempted; rerun after a legitimate GitHub rerun or fresh same-nonce "
        "dispatch (or restart review if the head changed)."
    )
    return 0


def _stop_on_ci_plumbing_failure(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    reason: str,
    run_urls: Sequence[str],
    plumbing: Sequence[str],
    final_line: str | None,
    recovery: str | None = None,
    watch_hint: bool = False,
) -> int:
    """Stop on a plumbing-only CI failure without a coder obligation (#1477).

    Exit semantics match the infrastructure-stall stop: a clean, resumable
    stop with no coder round and no merge.
    """
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=_ci_plumbing_stop_message(
            pr_number,
            reason=reason,
            run_urls=run_urls,
            plumbing=plumbing,
            final_line=final_line,
            recovery=recovery,
            watch_hint=watch_hint,
        ),
    )
    log(
        config,
        f"Round {round_number}: PR #{pr_number} CI failed only in plumbing jobs ({reason}); "
        "stopping on infrastructure without a coder round or merge",
    )
    print(
        f"PR #{pr_number} CI stopped on infrastructure: {reason} "
        f"Run: {', '.join(run_urls) or 'unavailable'}."
        + (f" {final_line}." if final_line else "")
        + " No coder round was started and no merge was attempted."
        + (f" {recovery}" if recovery else "")
    )
    return 0


def _pr_followup_source_context(
    *,
    config: AgentLoopConfig,
    pr_number: int,
    pr_metadata: PullRequestMetadata,
    issue_context: IssueContext | None,
) -> FollowupSourceContext:
    linked = parse_linked_issue_numbers(pr_metadata.body, repo=config.repo)
    parent_numbers = (issue_context.number,) if issue_context is not None else ()
    related = tuple(number for number in linked if number not in parent_numbers)
    if issue_context is None and len(linked) > 1:
        log(
            config,
            f"PR #{pr_number} has multiple linked issue references; preserving them as related context instead of inventing a parent",
        )
    if issue_context is None and not linked:
        log(config, f"Unable to resolve a parent issue for PR #{pr_number} follow-up lookup; using PR and topic context")
    return FollowupSourceContext(
        repo=config.repo,
        source_kind="pr",
        source_number=pr_number,
        source_identity=pr_metadata.head_sha,
        parent_issue_numbers=parent_numbers,
        related_issue_numbers=related,
        related_pr_numbers=(pr_number,),
    )


def _stop_after_ci_watch_timeout(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    head_sha: str | None,
    pr_comments: Sequence[object],
    followups: list[ApprovedFollowup],
    details: list[str],
    reason: Literal["budget_exhausted", "timeout", "non_authoritative", "protection_unreadable"],
    source_context: FollowupSourceContext,
    usage_context: RunUsageContext | None = None,
    checks_board: PullRequestChecks | None = None,
    stuck_checks: Sequence[PullRequestCheck] = (),
) -> int:
    """Publish resumable guidance for a watch that cannot continue or finish.

    Every watch-ending stop names the observed final-ci/exact-head state here
    (#1477), so no caller can drop known final-state evidence.
    """
    final_line = final_context_state_line(checks_board, stuck=stuck_checks)
    if final_line and final_line not in details:
        details = [*details, final_line]
    # Printed and raised reports carry the same state as the comment.
    final_suffix = f" {final_line}." if final_line else ""
    _publish_approved_followups(
        runner,
        config=config,
        pr_number=pr_number,
        head_sha=head_sha,
        pr_comments=pr_comments,
        followups=followups,
        source_context=source_context,
        usage_context=usage_context,
    )
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=(
            _unreadable_protection_stop_message(pr_number, details)
            if reason == "protection_unreadable"
            else _pending_ci_stop_message(pr_number, "pending", details)
        ),
    )
    rerun = _render_ci_rerun_command(config, pr_number=pr_number)
    note = (
        ""
        if config.invocation_argv
        else " (deterministic fallback; original invocation unavailable)"
    )
    if reason == "budget_exhausted":
        log(
            config,
            f"Round {round_number}: PR #{pr_number} shared CI watch budget was "
            "exhausted before a fresh poll; no merge attempted",
        )
        print(
            f"PR #{pr_number} CI watch budget was exhausted by earlier rounds; "
            f"no fresh poll was performed and no merge was attempted.{final_suffix} "
            f"Rerun: {rerun}{note}"
        )
        if config.auto_merge:
            raise AgentLoopError(
                f"PR #{pr_number} full-board CI watch budget was exhausted before "
                "a fresh poll; no merge attempted."
                + final_suffix
            )
    elif reason == "timeout":
        print(
            f"PR #{pr_number} CI watch timed out: {'; '.join(details)}. "
            f"Rerun: {rerun}{note}"
        )
        if config.auto_merge:
            raise AgentLoopError(
                f"PR #{pr_number} full-board CI watch did not pass within "
                f"{config.ci_timeout_seconds}s; no merge attempted."
                + final_suffix
            )
    elif reason == "protection_unreadable":
        log(
            config,
            f"Round {round_number}: PR #{pr_number} branch protection is unreadable and "
            "GitHub did not report a CLEAN merge state for the green board; no merge attempted",
        )
        print(
            f"PR #{pr_number} CI watch stopped: every observed check passed, but the current "
            "GitHub token cannot read branch protection (HTTP 403) and GitHub did not report "
            f"a CLEAN merge state for the current head: {'; '.join(details)}. "
            "Grant the token administration read access, or satisfy the remaining protection "
            f"rules (for example required reviews), then rerun: {rerun}{note}"
        )
        if config.auto_merge:
            raise AgentLoopError(
                f"PR #{pr_number} branch protection is unreadable and GitHub's merge state "
                "is not CLEAN; no merge attempted."
                + final_suffix
            )
    else:
        log(
            config,
            f"Round {round_number}: ordinary CI watcher returned a non-authoritative "
            "passing-looking board; no merge attempted",
        )
        print(
            f"PR #{pr_number} CI watch did not produce an authoritative success for the "
            f"current head: {'; '.join(details)}. No merge was attempted; rerun: {rerun}{note}"
        )
        if config.auto_merge:
            raise AgentLoopError(
                f"PR #{pr_number} full-board CI watch returned a non-authoritative "
                "passing-looking result; no merge attempted."
                + final_suffix
            )
    return 0


def _scheduler_obligations(
    items: Sequence[UnresolvedReviewItem],
    *,
    required_reviewers: Sequence[str] = (),
    active_statuses: frozenset[str] = frozenset({"blocking", "same-pr"}),
) -> tuple[ReviewObligation, ...]:
    """Build scheduler obligations from the canonical finding ledger.

    ``active_statuses`` defaults to the PR statuses.  The planning flow retains
    its follow-up findings as ``same-plan`` (#905, from #841), so it passes that
    status instead; otherwise a post-panel narrow revision for a ``same-plan``
    finding would lose its durable owner and the scheduler would fall through to
    a final sweep instead of invoking that owner plus the primary.
    """
    obligations: list[ReviewObligation] = []
    required = set(required_reviewers)
    for item in items:
        if item.status not in active_statuses:
            continue
        if _is_evidence_obligation(item):
            # No coder round can resolve human-only evidence, and its
            # unscoped owner would otherwise make every later code transition
            # broad (#1068).  Evidence passes run the full board by design.
            continue
        owners = item.resolution_owners or (item.reviewer,)
        raw_states = item.owner_states
        states = dict(raw_states or ((owner, "pending") for owner in owners))
        reconstructible = (
            bool(owners)
            and len(set(owners)) == len(owners)
            and all(isinstance(owner, str) and owner for owner in owners)
            and (not raw_states or (len(raw_states) == len(owners) and set(states) == set(owners)))
            and all(state in {"pending", "cleared"} for state in states.values())
            and item.reviewer in required
            and not any(note.startswith(CODER_DISPUTE_NOTE_PREFIX) for note in item.notes)
        )
        # A synthetic or orchestrator-owned item has no reviewer-authored
        # objective scope, so it must conservatively force a full board.
        scope = item.fix_scope if reconstructible else None
        if item.reviewer == "Orchestrator" or item.reviewer not in required:
            scope = None
        obligations.append(
            ReviewObligation(
                item_id=item.item_id,
                status=item.status,
                scope=scope,
                resolution_owners=tuple(owners),
                pending_owners=tuple(owner for owner in owners if states.get(owner) != "cleared"),
            )
        )
    return tuple(obligations)


def _partition_unresolved_items(
    items: Sequence[UnresolvedReviewItem],
    *,
    current_head_sha: str | None,
) -> dict[str, tuple[UnresolvedReviewItem, ...]]:
    """Separate human review work from machine qualification work.

    ``must_fix_items`` historically mixed these categories and made a
    synthetic CI reviewer look like a human owner.  The partitions are kept
    explicit so a reviewed repair head may qualify while every machine gate
    remains pending and visible until its own authority clears it.
    """
    reviewer_blockers: list[UnresolvedReviewItem] = []
    repair_required: list[UnresolvedReviewItem] = []
    revalidation_candidates: list[UnresolvedReviewItem] = []
    finalization_blockers: list[UnresolvedReviewItem] = []
    evidence_obligations: list[UnresolvedReviewItem] = []
    for item in items:
        if item.status not in {"blocking", "same-pr"}:
            continue
        finalization_blockers.append(item)
        if _is_evidence_obligation(item):
            # Human-only exact-head evidence is a final barrier (#1068): it
            # blocks finalization but is never coder or repair work.
            evidence_obligations.append(item)
        elif not _is_machine_obligation(item):
            reviewer_blockers.append(item)
        elif _machine_obligation_requires_repair(item, current_head_sha=current_head_sha):
            repair_required.append(item)
        elif _machine_obligation_is_revalidation_candidate(
            item, current_head_sha=current_head_sha
        ):
            revalidation_candidates.append(item)
        else:
            repair_required.append(item)
    coder_blockers = [*reviewer_blockers, *repair_required]
    return {
        "reviewer_blockers": tuple(reviewer_blockers),
        "repair_required_machine_obligations": tuple(repair_required),
        "revalidation_candidates": tuple(revalidation_candidates),
        "finalization_blockers": tuple(finalization_blockers),
        "coder_blockers": tuple(coder_blockers),
        "evidence_obligations": tuple(evidence_obligations),
    }


def _log_coder_followup_dispatch(
    config: AgentLoopConfig,
    round_number: int,
    coder_name: str,
    coder_followup_items: Sequence[UnresolvedReviewItem],
) -> None:
    """Announce a coder round by what actually routed it (#1024)."""
    if coder_followup_is_ci_repair(coder_followup_items):
        log(config, f"Round {round_number}: {coder_name} repairing failed CI")
    else:
        log(config, f"Round {round_number}: {coder_name} addressing reviewer feedback")


def _machine_obligation_checkpoint(
    items: Sequence[UnresolvedReviewItem],
    *,
    current_head_sha: str | None,
    base_branch: str | None,
    allowed_rounds: int,
    watch_failure_extension_used: bool,
    watch_head_extension_used: bool,
    lifecycle: str | None = None,
    qualification_attempt_id: str | None = None,
    approval_digest: str | None = None,
    plan_digest: str | None = None,
    requirements_digest: str | None = None,
    acquisition_digest: str | None = None,
    scheduler_digest: str | None = None,
    previous_checkpoint: QualificationCheckpoint | None = None,
    managed_contract: ManagedCiContract | None = None,
) -> QualificationCheckpoint | None:
    """Create a bounded checkpoint from the active source-specific obligation."""
    candidates = [
        item for item in items
        if _is_machine_obligation(item)
        and item.status in {"blocking", "same-pr"}
        and item.obligation_kind in CI_MACHINE_OBLIGATION_KINDS
    ]
    item = next(
        (
            candidate for candidate in candidates
            if _machine_obligation_is_revalidation_candidate(
                candidate, current_head_sha=current_head_sha
            )
        ),
        None,
    ) or (candidates[0] if candidates else None)
    if item is None:
        return None
    effective_lifecycle = lifecycle or item.lifecycle or "repair_required"
    # Head advancement is deliberately fail-closed.  A force-push revert to a
    # previously failed head leaves the obligation in ``repair_required`` with
    # no candidate.  Several callers request the normal awaiting-review state
    # after a head change, but that request is invalid for this transition and
    # must not leak QualificationCheckpoint's ValueError out of the loop.
    if (
        not item.candidate_head_sha
        or item.candidate_head_sha == item.failed_head_sha
        or item.lifecycle == "repair_required"
    ):
        effective_lifecycle = "repair_required"
    from .managed_ci_reconciliation import capture_failed_run
    reference = None
    if (previous_checkpoint is not None and previous_checkpoint.valid
        and previous_checkpoint.obligation_identity == (item.obligation_identity or item.item_id)
        and previous_checkpoint.failed_head_sha == item.failed_head_sha
        and previous_checkpoint.base_branch == base_branch):
        reference = previous_checkpoint.failed_managed_run
    if reference is not None and previous_checkpoint is not None and current_head_sha == previous_checkpoint.failed_head_sha:
        approval_digest = approval_digest or previous_checkpoint.approval_digest
        plan_digest = plan_digest or previous_checkpoint.plan_digest
        requirements_digest = requirements_digest or previous_checkpoint.requirements_digest
        acquisition_digest = acquisition_digest or previous_checkpoint.acquisition_digest
        scheduler_digest = scheduler_digest or _qualification_digest(_prior_item_ledger_signature(items))
    checkpoint = QualificationCheckpoint(
        obligation_kind=item.obligation_kind or "unknown",
        obligation_identity=item.obligation_identity or item.item_id,
        lifecycle=effective_lifecycle,
        failed_head_sha=item.failed_head_sha,
        candidate_head_sha=item.candidate_head_sha,
        base_branch=base_branch,
        approval_digest=approval_digest,
        plan_digest=plan_digest,
        requirements_digest=requirements_digest,
        acquisition_digest=acquisition_digest,
        scheduler_digest=scheduler_digest,
        qualification_attempt_id=qualification_attempt_id,
        watch_failure_extension_used=watch_failure_extension_used,
        watch_head_extension_used=watch_head_extension_used,
        allowed_rounds=allowed_rounds,
        failed_managed_run=reference,
    )
    if reference is None:
        reference = capture_failed_run(managed_contract, checkpoint, item.source_round)
    return dataclasses_replace(checkpoint, failed_managed_run=reference)


def _qualification_digest(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:16]


def _managed_checkpoint_needs_fresh_attempt(
    checkpoint: QualificationCheckpoint | None,
    contract: ManagedCiContract | None,
    head_sha: str | None,
) -> bool:
    """True when a fresh process must re-dispatch a ``qualifying`` managed attempt (#1177).

    Real activation mints a new intent generation that never adopts an
    earlier generation's attempt, so a persisted ``qualifying`` managed
    checkpoint cannot name an attempt this invocation owns.  Such a
    checkpoint may resume as ``qualification_ready`` only when the contract
    is a managed v2 contract that owns no attempt at all (no attachment,
    nonce or issued dispatch) and is not bound to another head.  Every other
    combination keeps the strict attempt-ID comparison authoritative.
    """
    return bool(
        checkpoint is not None
        and checkpoint.valid
        and checkpoint.lifecycle == "qualifying"
        and checkpoint.obligation_kind == "managed-exact-head-ci"
        and head_sha
        and checkpoint.candidate_head_sha == head_sha
        and contract is not None
        and contract.protocol_version == 2
        and contract.activation_path == "managed"
        and (contract.expected_head_sha is None or contract.expected_head_sha == head_sha)
        and contract.attached_run_id is None
        and contract.run_attempt is None
        and contract.nonce is None
        and not contract.dispatch_issued
    )


def _qualification_checkpoint_review_identity_matches(
    checkpoint: QualificationCheckpoint,
    *,
    configured_reviewers: Sequence[AgentName],
    current_approvals: Mapping[str, object],
    unresolved_items: Sequence[UnresolvedReviewItem],
    expected_plan_digest: str | None,
    expected_requirements_digest: str,
    expected_acquisition_digest: str,
    expected_qualification_attempt_id: str | None = None,
    historical_result: bool = False,
) -> bool:
    """Check the durable proof needed to skip a resumed reviewer board.

    Repair handoffs do not claim that the candidate has been reviewed.  Once a
    checkpoint says that the candidate is ready for, or is already in, final
    qualification, however, it may suppress reviewer execution only when the
    persisted identities still describe the live transcript.  In particular,
    a summary-only checkpoint or a checkpoint with a changed ledger must force
    the normal full-board recovery path.
    """
    if not historical_result and checkpoint.lifecycle not in {"qualification_ready", "qualifying"}:
        return True
    expected_reviewers = {
        agent_display_name(reviewer) for reviewer in configured_reviewers
    }
    observed_reviewers = set(current_approvals)
    if observed_reviewers != expected_reviewers:
        return False
    if (
        not checkpoint.approval_digest
        or not checkpoint.requirements_digest
        or not checkpoint.acquisition_digest
        or not checkpoint.scheduler_digest
    ):
        return False
    current_approval_digest = _qualification_digest(tuple(sorted(observed_reviewers)))
    current_scheduler_digest = _qualification_digest(
        _prior_item_ledger_signature(unresolved_items)
    )
    return (
        checkpoint.approval_digest == current_approval_digest
        and checkpoint.plan_digest == expected_plan_digest
        and checkpoint.requirements_digest == expected_requirements_digest
        and checkpoint.acquisition_digest == expected_acquisition_digest
        and checkpoint.scheduler_digest == current_scheduler_digest
        and (
            checkpoint.lifecycle != "qualifying"
            or (
                expected_qualification_attempt_id is not None
                and checkpoint.qualification_attempt_id == expected_qualification_attempt_id
            )
        )
    )


def approved_pr_reopen_hint(pr_number: int) -> str:
    """Name the operator path for new instructions on an approved PR (#1020).

    An ordinary PR comment is not a requirement, so an approved head exits
    without dispatching an agent. A signed human requirement invalidates the
    carried approvals and re-invokes reviewers against it at the same head.
    """
    return (
        f" To add instructions this PR must still satisfy, post a PR comment that ends "
        f"with a line containing exactly `-- Human Reviewer`, then rerun `agent-loop pr "
        f"{pr_number}`. Unsigned comments are not read as requirements. Only a human may "
        "sign; an agent relaying an operator decision must disclose the relay in the body."
    )


def _sub_item_progress_already_posted(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    digest: str,
) -> bool:
    """Whether a trusted orchestrator comment already carries this round's record.

    Only comments written by the authenticated actor count; the same record in
    anyone else's comment is ordinary text and grants nothing (#958).
    """
    try:
        actor_login, actor_id = resolve_authenticated_github_actor(runner, config=config)
    except AgentLoopError:
        return False
    comments = get_pr_review_context(runner, config=config, pr_number=pr_number).comments
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        author_id = getattr(comment, "author_id", None)
        if author_id is not None:
            if author_id != actor_id:
                continue
        elif getattr(comment, "author", None) != actor_login:
            continue
        if (pr_number, round_number, digest) in sub_item_progress_record_keys(body):
            return True
    return False


def _publish_sub_item_progress(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    items: Sequence[UnresolvedReviewItem],
    cleared: Sequence[ClearedItemProgress],
    notes: Sequence[str] = (),
) -> None:
    """Log the sub-item signal and post the once-per-round confirmed outcome (#958).

    Advisory only: nothing here changes the round budget.  The comment is a
    pure function of the reconciled ledger, so a resumed round recomputes the
    same digest and the trusted-record check keeps the post exactly-once.
    """
    for note in notes:
        log(config, f"Round {round_number}: sub-item note: {note}")
    window = config.sub_item_stall_rounds
    for entry in sub_item_progress(items, (), current_round=round_number, window=window):
        if entry.classification == "stalled":
            log(
                config,
                f"Round {round_number}: WARNING: {entry.item_id} has "
                f"{entry.resolved}/{entry.total} sub-items resolved and none closed in the "
                f"last {entry.window} rounds (stalled; advisory, the round budget is unchanged).",
            )
    stalled = newly_stalled_items(items, current_round=round_number, window=window)
    if not cleared and not stalled:
        return
    digest = sub_item_progress_digest(cleared, stalled)
    if _sub_item_progress_already_posted(
        runner, config=config, pr_number=pr_number, round_number=round_number, digest=digest
    ):
        return
    body = render_sub_item_progress_comment(
        pr_number=pr_number, round_number=round_number, cleared=cleared, stalled=stalled
    )
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=TrustedBody.canonical(body, expected_tokens=("AGENT_SUB_ITEM_PROGRESS",)),
    )


def _sub_item_progress_block(
    items: Sequence[UnresolvedReviewItem], *, round_number: int, window: int
) -> str:
    """Per-item converging/stalled summary for a budget-exit message (#958).

    Empty when no carried item has sub-items, so every diagnostic stays
    byte-identical for findings without them.
    """
    lines = render_sub_item_progress_summary(
        sub_item_progress(items, (), current_round=round_number, window=window)
    )
    if not lines:
        return ""
    return "\nSub-item progress:\n" + "\n".join(f"- {line}" for line in lines)


def _round_limit_diagnostic(
    *,
    pr_number: int,
    round_number: int,
    items: Sequence[UnresolvedReviewItem],
    current_head_sha: str | None,
    sub_item_stall_rounds: int | None = None,
) -> str:
    """Describe the blocker plus, when items have sub-items, their progress (#958)."""
    message = _round_limit_blocker_message(
        pr_number=pr_number,
        round_number=round_number,
        items=items,
        current_head_sha=current_head_sha,
    )
    if sub_item_stall_rounds is None:
        return message
    return message + _sub_item_progress_block(
        items, round_number=round_number, window=sub_item_stall_rounds
    )


def _round_limit_blocker_message(
    *,
    pr_number: int,
    round_number: int,
    items: Sequence[UnresolvedReviewItem],
    current_head_sha: str | None,
) -> str:
    """Describe the actual terminal/resumable blocker, not its display owner."""
    partitions = _partition_unresolved_items(items, current_head_sha=current_head_sha)
    reviewer_blockers = partitions["reviewer_blockers"]
    repair = partitions["repair_required_machine_obligations"]
    candidates = partitions["revalidation_candidates"]
    evidence = partitions["evidence_obligations"]
    # Code and machine blockers are named first; deferred human-only evidence
    # is described as the remaining final barrier, never as the blocker.
    evidence_suffix = _evidence_barrier_note(evidence)
    if reviewer_blockers:
        names = ", ".join(
            f"{item.reviewer} ({item.item_id})" for item in reviewer_blockers
        )
        return (
            f"PR #{pr_number} still reported blocking issues after round {round_number}: "
            f"reviewer-owned findings: {names}. "
            "The named reviewer/owner must provide actionable resolution evidence."
            + evidence_suffix
        )
    if repair:
        details = ", ".join(
            f"{item.obligation_kind or 'unknown'} ({item.item_id})"
            for item in repair
        )
        return (
            f"PR #{pr_number} has blocking issues after round {round_number}: machine obligation(s) "
            "awaiting a new repair head: "
            f"{details}. Reviewer approval cannot clear them; push a strictly "
            "different corrected head."
            + evidence_suffix
        )
    if candidates:
        details = ", ".join(
            f"{item.obligation_kind or 'unknown'} ({item.item_id})"
            for item in candidates
        )
        return (
            f"PR #{pr_number} has a unanimously reviewed correction awaiting authoritative "
            f"qualification after round {round_number}: {details}."
            + evidence_suffix
        )
    if evidence and len(evidence) == len(
        [item for item in items if item.status in {"blocking", "same-pr"}]
    ):
        return (
            f"PR #{pr_number} has no open code or machine findings after round {round_number}; "
            "the remaining barrier is human-only exact-head evidence: "
            + ", ".join(f"{item.reviewer} ({item.item_id})" for item in evidence)
            + ". It is requested only at the clean head through an evidence freeze."
        )
    active = [item for item in items if item.status in {"blocking", "same-pr"}]
    if active:
        return (
            f"PR #{pr_number} has an unknown or unreconstructible persisted obligation after "
            f"round {round_number}; no qualification or merge is permitted."
        )
    return f"Reached the review budget after round {round_number} for PR #{pr_number}; human review required."


def _single_line_diagnostic(value: object) -> str:
    """Escape every control character so a value renders on exactly one line."""
    out: list[str] = []
    for ch in str(value):
        if unicodedata.category(ch)[0] == "C" or unicodedata.category(ch) in {"Zl", "Zp"}:
            code = ord(ch)
            out.append(f"\\x{code:02x}" if code <= 0xFF else f"\\u{code:04x}")
        else:
            out.append(ch)
    return "".join(out)


def _finalization_obligation_predicate(
    item: UnresolvedReviewItem, *, current_head_sha: str | None
) -> str:
    """Describe, from ledger fields only, what keeps one obligation unsatisfied.

    Never claims success that was not observed: no wording here says a source
    passed or succeeded, and ``failed at`` appears only where a failed head is
    recorded.
    """
    kind = item.obligation_kind
    lifecycle = item.lifecycle
    cur = current_head_sha or "none"
    candidate = item.candidate_head_sha
    failed = item.failed_head_sha
    if kind not in MACHINE_OBLIGATION_KINDS or kind == "unknown":
        return "unknown or unreconstructible obligation; no qualification or merge is permitted"
    if _is_evidence_obligation(item):
        frozen = f" at {candidate}" if candidate else ""
        return f"human-only exact-head evidence pending{frozen}"
    if kind == "human-requirements-acknowledgement":
        return (
            "signed human requirements have not been validly acknowledged; "
            "a response acknowledging them is required"
        )
    if kind == "merge-conflict":
        head = (
            f"merge conflict with the base branch confirmed at {failed}"
            if failed
            else "merge conflict with the base branch reported; head not confirmed"
        )
        return f"{head}; resolve the conflict on a new head"
    if kind == "alembic-migration":
        head = (
            f"migration validation failed at {failed}"
            if failed
            else "migration validation has no recorded clearance"
        )
        return (
            f"{head}; it is re-probed each round and clears only when "
            f"validation of the current head {cur} succeeds"
        )
    if kind in CI_MACHINE_OBLIGATION_KINDS:
        if lifecycle == "repair_required":
            if failed and failed == current_head_sha:
                return (
                    f"authoritative source failed at {failed}, which is the current head; "
                    "a strictly different head is required"
                )
            if failed:
                return (
                    f"authoritative source failed at {failed}; current head {cur} "
                    "has not been bound as a revalidation candidate"
                )
            return (
                "authoritative source reported a failure at an unrecorded head; "
                "a corrected head is required"
            )
        if lifecycle in {"awaiting_current_head_review", "qualification_ready", "qualifying"}:
            if not candidate:
                return f"lifecycle {lifecycle} has no recorded candidate head; failing closed"
            if failed and failed == current_head_sha:
                return (
                    f"lifecycle {lifecycle}, but the current head {cur} is the recorded "
                    "failed head; a strictly different head is required"
                )
            if candidate != current_head_sha:
                return (
                    f"lifecycle {lifecycle} is bound to candidate head {candidate}, "
                    f"not the current head {cur}; the current head must be reviewed "
                    "before qualification"
                )
            if lifecycle == "awaiting_current_head_review":
                return f"awaiting unanimous reviewer approval at candidate head {candidate}"
            if lifecycle == "qualification_ready":
                return f"approved at {candidate}; authoritative qualification not yet dispatched"
            return (
                f"qualification in progress at {candidate}; "
                "no authoritative success recorded for this source in this run"
            )
    return f"{kind} in lifecycle {lifecycle or 'none'}; no clearance recorded; failing closed"


def _finalization_obligation_detail(
    items: Sequence[UnresolvedReviewItem],
    *,
    current_head_sha: str | None,
    observations: Mapping[str, str] | None,
) -> str:
    """Render one escaped line per machine-obligation blocker (#1119)."""
    lines = _finalization_obligation_lines(
        items, current_head_sha=current_head_sha, observations=observations
    )
    if not lines:
        return ""
    return "\nBlocking obligations:\n" + "\n".join(f"- {line}" for line in lines)


def _finalization_obligation_lines(
    items: Sequence[UnresolvedReviewItem],
    *,
    current_head_sha: str | None,
    observations: Mapping[str, str] | None,
) -> list[str]:
    partitions = _partition_unresolved_items(items, current_head_sha=current_head_sha)
    ordered = (
        *partitions["repair_required_machine_obligations"],
        *partitions["revalidation_candidates"],
        *partitions["evidence_obligations"],
    )
    esc = _single_line_diagnostic
    lines: list[str] = []
    for item in ordered:
        if item.item_id in (observations or {}):
            predicate = f"observed in this run: {esc(observations[item.item_id])}"
        else:
            predicate = esc(
                _finalization_obligation_predicate(item, current_head_sha=current_head_sha)
            )
        lines.append(
            f"{esc(item.obligation_kind or 'unknown')} ({esc(item.item_id)}): "
            f"lifecycle={esc(item.lifecycle or 'none')}, "
            f"candidate_head={esc(item.candidate_head_sha or 'none')}, "
            f"failed_head={esc(item.failed_head_sha or 'none')}; {predicate}"
        )
    return lines


def _ensure_finalization_ready(
    *,
    pr_number: int,
    round_number: int,
    items: Sequence[UnresolvedReviewItem],
    current_head_sha: str | None,
    ignored_machine_kinds: frozenset[str] = frozenset(),
    sub_item_stall_rounds: int | None = None,
    observations: Mapping[str, str] | None = None,
    config: AgentLoopConfig | None = None,
) -> None:
    """Fail closed unless every non-ignored obligation is actually cleared.

    A coder-blocker-free partition is sufficient to start source-specific
    qualification, but it is never sufficient to approve or merge.  The
    authoritative success path may explicitly ignore the one source it is
    about to validate (ordinary recovery); all other ledger obligations must
    be gone before a finalization side effect.
    """
    blockers = tuple(
        item
        for item in _partition_unresolved_items(
            items, current_head_sha=current_head_sha
        )["finalization_blockers"]
        if not (
            _is_machine_obligation(item)
            and item.obligation_kind in ignored_machine_kinds
        )
    )
    if blockers:
        diagnostic = _round_limit_diagnostic(
            pr_number=pr_number,
            round_number=round_number,
            items=blockers,
            current_head_sha=current_head_sha,
            sub_item_stall_rounds=sub_item_stall_rounds,
        )
        detail = _finalization_obligation_detail(
            blockers, current_head_sha=current_head_sha, observations=observations
        )
        if config is not None:
            # Operator diagnostics only: logged, never posted (#1119).
            for item in blockers:
                if _is_machine_obligation(item):
                    continue
                log(
                    config,
                    f"{_single_line_diagnostic(item.reviewer)} "
                    f"({_single_line_diagnostic(item.item_id)}): reviewer-owned finding",
                )
            for line in _finalization_obligation_lines(
                blockers, current_head_sha=current_head_sha, observations=observations
            ):
                log(config, f"PR #{pr_number} blocking obligation {line}")
        raise AgentLoopError(
            f"PR #{pr_number} cannot finalize: {diagnostic} No approval or merge was attempted."
            + detail
        )


class _RepeatableRoundSequence:
    """Round numbers for the PR loop; the current round may run again (#1068).

    Evidence-response and same-head refresh passes re-run the review board
    inside the current round, so they never advance ``round_number`` and the
    pre-round budget guard cannot reject them.  Each repeat is triggered only
    by a signed requirement ID no reviewer has seen, and a hard cap backs that
    bound so a misbehaving source cannot spin the loop.
    """

    MAX_REPEATS_PER_ROUND = 8

    def __init__(self, start: int, stop: int) -> None:
        self._next = start
        self._stop = stop
        self._current: int | None = None
        self._repeat = False
        self._repeats = 0

    def __iter__(self) -> "_RepeatableRoundSequence":
        return self

    def __next__(self) -> int:
        if self._repeat and self._current is not None:
            self._repeat = False
            return self._current
        if self._next >= self._stop:
            raise StopIteration
        self._current = self._next
        self._next += 1
        self._repeats = 0
        return self._current

    def repeat_current(self) -> None:
        self._repeats += 1
        if self._repeats > self.MAX_REPEATS_PER_ROUND:
            raise HumanDecisionRequiredError(
                f"Round {self._current} was re-run {self.MAX_REPEATS_PER_ROUND} times for new "
                "signed human input without settling; human review required."
            )
        self._repeat = True


def _evidence_barrier_note(items: Sequence[UnresolvedReviewItem]) -> str:
    """Name pending human-only evidence as the remaining final barrier."""
    evidence = _pending_evidence_obligations(items)
    if not evidence:
        return ""
    return (
        " Human-only exact-head evidence remains the final barrier ("
        + ", ".join(f"{item.item_id} from {item.reviewer}" for item in evidence)
        + "); it is requested only once every code finding and machine gate is clean."
    )


def _append_evidence_barrier_note(body: str, items: Sequence[UnresolvedReviewItem]) -> str:
    """Add the evidence note to a clean-stop comment, before its signature."""
    note = _evidence_barrier_note(items).strip()
    if not note:
        return body
    text = str(body)
    if "\n-- " in text:
        prefix, signature = text.rsplit("\n-- ", 1)
        return f"{prefix.rstrip()}\n\n{note}\n\n-- {signature}"
    return f"{text.rstrip()}\n\n{note}"


def _evidence_requester(item: UnresolvedReviewItem) -> str:
    if item.converted_row_id is not None:
        owners = ", ".join(item.resolution_owners) or item.reviewer
        return f"converted row `{item.converted_row_id}`, owned by {owners}"
    return f"requested by {item.reviewer}"


def _evidence_item_lines(items: Sequence[UnresolvedReviewItem]) -> list[str]:
    return [
        f"- [{item.item_id}] {_evidence_requester(item)}: {item.text}"
        for item in items
    ]


def _evidence_freeze_diagnostic(
    *, pr_number: int, head_sha: str | None, items: Sequence[UnresolvedReviewItem]
) -> str:
    evidence = _pending_evidence_obligations(items)
    return (
        f"PR #{pr_number} is frozen at head {head_sha or 'unknown'} awaiting human-only "
        "exact-head evidence: "
        + "; ".join(
            f"{item.item_id} for converted row {item.converted_row_id}"
            if item.converted_row_id is not None
            else f"{item.item_id} from {item.reviewer}"
            for item in evidence
        )
        + ". No coder, CI qualification, or merge was started. Supply the evidence for "
        "exactly this head, or withdraw the request, in a PR comment whose last line is "
        f"exactly `-- Human Reviewer`, then rerun `agent-loop pr {pr_number}`. Pushing a new "
        "commit breaks the freeze and the new head needs full review and fresh evidence."
    )


def _render_evidence_freeze_notice(
    *,
    pr_number: int,
    head_sha: str,
    items: Sequence[UnresolvedReviewItem],
    still_frozen: bool,
) -> str:
    evidence = _pending_evidence_obligations(items)
    heading = (
        "## Exact-head evidence freeze (still in effect)"
        if still_frozen
        else "## Exact-head evidence freeze"
    )
    return "\n".join(
        [
            heading,
            "",
            f"Every reviewer reports no code findings at head `{head_sha}` and every machine "
            "gate is clean. The remaining barrier is evidence that an agent session cannot "
            "produce:",
            "",
            *_evidence_item_lines(evidence),
            "",
            f"Evidence requested at `{head_sha}`. No further code changes will be accepted "
            "until the authenticated live evidence for this head is supplied or the request "
            "is withdrawn.",
            "",
            "To respond, post a PR comment that supplies the evidence for exactly this head "
            "(or withdraws the request) and ends with a line containing exactly "
            f"`-- Human Reviewer`, then rerun `agent-loop pr {pr_number}`. The requesting "
            "reviewer re-reviews this same head. Pushing a new commit breaks the freeze; the "
            "new head then needs full review and fresh evidence.",
            "",
            "-- coding-review-agent-loop",
        ]
    )


_EVIDENCE_RELEASE_MESSAGES = {
    "machine-gate": (
        "a machine gate failed at the frozen head ({detail}). The requested evidence returns "
        "to deferred; the failure is repaired first and the evidence is requested again at "
        "the next clean head."
    ),
    "findings": (
        "a reviewer raised new findings at the frozen head. The requested evidence returns "
        "to deferred and is requested again at the next clean head; evidence for this head "
        "does not carry forward."
    ),
    "evidence-cleared": (
        "the requesting reviewer(s) accepted the supplied evidence or its withdrawal. The PR "
        "proceeds to final validation at the same head."
    ),
    "refresh-clean": (
        "a same-head refresh for new signed input found no findings. The PR proceeds to "
        "final validation at the same head."
    ),
}


def _publish_evidence_freeze(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    head_sha: str,
    items: Sequence[UnresolvedReviewItem],
    signed_requirement_ids: Sequence[str],
    allowed_rounds: int,
    watch_failure_extension_used: bool,
    watch_head_extension_used: bool,
    clearances: Sequence[tuple[str, str]] = (),
    still_frozen: bool = False,
    signed_evidence_references: Sequence[SignedEvidenceReference] = (),
) -> list[UnresolvedReviewItem]:
    """Publish and persist a freeze with exactly one comment (#1068).

    The frozen ledger exists only in memory until this single write; the
    comment body is the human notice and its round metadata is the only
    persistence.  A failed write therefore leaves no freeze behind.
    """
    frozen = freeze_evidence_obligations(items, head_sha=head_sha)
    identities = tuple(
        item.obligation_identity or item.item_id for item in _frozen_evidence_obligations(frozen)
    )
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=_attach_round_metadata(
            _render_evidence_freeze_notice(
                pr_number=pr_number,
                head_sha=head_sha,
                items=frozen,
                still_frozen=still_frozen,
            ),
            PostedRoundMetadata(
                flow="pr",
                role="summary",
                agent="Orchestrator",
                seat_binding=reviewer_seat_binding(config),
                round_number=round_number,
                subject=head_sha,
                prior_items=tuple(frozen),
                state="blocking",
                phase=EVIDENCE_FREEZE_PHASE,
                evidence_freeze=EvidenceFreezeRecord(
                    frozen_head=head_sha,
                    evidence_identities=identities,
                    signed_requirement_ids_at_freeze=tuple(sorted(set(signed_requirement_ids))),
                    allowed_rounds=allowed_rounds,
                    watch_failure_extension_used=watch_failure_extension_used,
                    watch_head_extension_used=watch_head_extension_used,
                ),
                evidence_clearances=tuple(clearances),
                signed_evidence_references=_cumulative_references(signed_evidence_references),
                **_architecture_metadata_fields(config),
            ),
        ),
    )
    return frozen


def _publish_evidence_release(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    head_sha: str,
    items: Sequence[UnresolvedReviewItem],
    reason: str,
    surfaced_requirement_ids: Sequence[str],
    allowed_rounds: int,
    watch_failure_extension_used: bool,
    watch_head_extension_used: bool,
    clearances: Sequence[tuple[str, str]] = (),
    detail: str = "",
    signed_evidence_references: Sequence[SignedEvidenceReference] = (),
) -> list[UnresolvedReviewItem]:
    """Post the single terminal release comment carrying the released ledger.

    The cumulative signed-evidence references (#1325) ride on every terminal
    record, so neither a partial clearance nor a release for findings loses
    a reference minted in this pass.
    """
    released = release_evidence_freeze(items)
    message = _EVIDENCE_RELEASE_MESSAGES[reason].format(detail=detail or "see the checks comment")
    heading = {
        "evidence-cleared": "## Exact-head evidence accepted",
        "refresh-clean": "## Same-head refresh complete",
    }.get(reason, "## Exact-head evidence freeze released")
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=_attach_round_metadata(
            f"{heading}\n\nAt head `{head_sha}`: {message}"
            + _operator_signed_rows_note(signed_evidence_references, head_sha=head_sha)
            + _evidence_barrier_note(released)
            + "\n\n-- coding-review-agent-loop",
            PostedRoundMetadata(
                flow="pr",
                role="summary",
                agent="Orchestrator",
                seat_binding=reviewer_seat_binding(config),
                round_number=round_number,
                subject=head_sha,
                prior_items=tuple(released),
                state="blocking",
                phase=EVIDENCE_RELEASE_PHASE,
                evidence_release=EvidenceReleaseRecord(
                    released_head=head_sha,
                    reason=reason,
                    signed_requirement_ids_surfaced=tuple(sorted(set(surfaced_requirement_ids))),
                    allowed_rounds=allowed_rounds,
                    watch_failure_extension_used=watch_failure_extension_used,
                    watch_head_extension_used=watch_head_extension_used,
                ),
                evidence_clearances=tuple(clearances),
                signed_evidence_references=_cumulative_references(signed_evidence_references),
                **_architecture_metadata_fields(config),
            ),
        ),
    )
    return released


def _cumulative_references(
    references: Sequence[SignedEvidenceReference],
) -> tuple[SignedEvidenceReference, ...]:
    """Deduplicated references in first-seen order (#1325)."""
    seen: dict[str, SignedEvidenceReference] = {}
    for reference in references:
        seen.setdefault(reference.reference_id, reference)
    return tuple(seen.values())


def _operator_signed_rows_note(
    references: Sequence[SignedEvidenceReference], *, head_sha: str
) -> str:
    """Public labels for rows verified by operator-signed evidence at this head."""
    rows = sorted(
        {reference.row_id for reference in references if reference.head == head_sha}
    )
    if not rows:
        return ""
    return "\n\nRisk-matrix rows verified by operator-signed evidence:\n" + "\n".join(
        f"- `{row_id}`: verified ({EVIDENCE_SOURCE_OPERATOR_SIGNED} at `{head_sha}`)"
        for row_id in rows
    )


@dataclass(frozen=True)
class _EvidenceGateOutcome:
    """What the finalization-point evidence gate decided (#1068)."""

    action: str  # "proceed" | "head_changed" | "refresh"
    context: PullRequestReviewContext | None = None


def _evidence_freeze_gate(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    head_sha: str | None,
    items: Sequence[UnresolvedReviewItem],
    surfaced_requirement_ids: Sequence[str] | None,
    collect_requirement_ids: Callable[[PullRequestReviewContext], Sequence[str]],
    allowed_rounds: int,
    watch_failure_extension_used: bool,
    watch_head_extension_used: bool,
    clearances: Sequence[tuple[str, str]] = (),
    signed_evidence_references: Sequence[SignedEvidenceReference] = (),
    approved_plan_context: ApprovedPlanContext | None = None,
    in_process_audits: Sequence[EvidenceConversionAudit] = (),
) -> _EvidenceGateOutcome:
    """Final revalidation and, when evidence is pending, the single freeze write.

    Runs immediately before ``_ensure_finalization_ready`` and any merge, after
    every machine gate.  The two reads are the last before the write:

    * a live head other than ``head_sha`` returns ``head_changed`` without
      posting anything about evidence;
    * a live signed requirement ID outside the set surfaced to this round's
      reviewers returns ``refresh`` so the board re-runs at the same head
      before any freeze or approval (a missing baseline counts as empty).

    With both matching and evidence pending, the freeze is published and the
    run stops at the human decision boundary.  With nothing pending the
    caller proceeds to finalization.
    """
    fresh = get_pr_review_context(runner, config=config, pr_number=pr_number)
    if not head_sha or fresh.metadata.head_sha != head_sha:
        return _EvidenceGateOutcome("head_changed", fresh)
    live_ids = tuple(collect_requirement_ids(fresh))
    if set(live_ids) - set(surfaced_requirement_ids or ()):
        return _EvidenceGateOutcome("refresh", fresh)
    if any(_is_row_converted_obligation(item) for item in items):
        # Defense in depth (#1325): a stale converted tombstone that survived
        # into finalization reopens here and freezes instead of proceeding.
        items = _reconcile_converted_evidence(
            runner,
            config,
            pr_number=pr_number,
            ledger=items,
            head_sha=head_sha,
            approved_plan_context=approved_plan_context,
            comments=fresh.comments,
            in_process_references=signed_evidence_references,
            in_process_audits=in_process_audits,
        )
    if not _pending_evidence_obligations(items):
        return _EvidenceGateOutcome("proceed", fresh)
    still_frozen = bool(_frozen_evidence_obligations(items))
    frozen = _publish_evidence_freeze(
        runner,
        config=config,
        pr_number=pr_number,
        round_number=round_number,
        head_sha=head_sha,
        items=items,
        signed_requirement_ids=live_ids,
        allowed_rounds=allowed_rounds,
        watch_failure_extension_used=watch_failure_extension_used,
        watch_head_extension_used=watch_head_extension_used,
        clearances=clearances,
        still_frozen=still_frozen,
        signed_evidence_references=signed_evidence_references,
    )
    raise HumanDecisionRequiredError(
        _evidence_freeze_diagnostic(pr_number=pr_number, head_sha=head_sha, items=frozen)
    )


def _evidence_gate_needed(items: Sequence[UnresolvedReviewItem]) -> bool:
    """Pending evidence, or a converted row whose tombstone must be rebound (#1325)."""
    return bool(_pending_evidence_obligations(items)) or any(
        _is_row_converted_obligation(item) for item in items
    )


def _refuse_dispatch_while_evidence_frozen(
    items: Sequence[UnresolvedReviewItem], *, pr_number: int, operation: str
) -> None:
    """Choke point before any head-changing dispatch (#1068).

    Every legitimate path releases a freeze before dispatching, so this only
    catches bugs: a frozen head must never move under a human evidence run.
    """
    frozen = _frozen_evidence_obligations(items)
    if not frozen:
        return
    raise AgentLoopError(
        f"Refusing to dispatch {operation} while exact-head evidence is frozen. "
        + _evidence_freeze_diagnostic(
            pr_number=pr_number, head_sha=frozen[0].candidate_head_sha, items=frozen
        )
    )


def _is_evidence_only_blocking_review(
    parsed: ParsedReview, prior_items: Sequence[UnresolvedReviewItem]
) -> bool:
    """A blocking review whose only open items are evidence requests.

    Missing human evidence is not a code blocker, so such a review approves
    the code while the evidence obligations stay in the ledger.
    """
    if parsed.state != "blocking" or parsed.blocking_items or parsed.followups.same_pr:
        return False
    evidence_ids = {item.item_id for item in prior_items if _is_evidence_obligation(item)}
    active = [
        disposition
        for disposition in parsed.dispositions
        if disposition.disposition in {"blocking", "same-pr"}
    ]
    if any(disposition.item_id not in evidence_ids for disposition in active):
        return False
    return bool(active) or bool(parsed.exact_head_evidence_requests)


def _evidence_review_context(
    items: Sequence[UnresolvedReviewItem], *, response_head: str | None
) -> str:
    """Reviewer context for carried evidence obligations and response passes."""
    evidence = _pending_evidence_obligations(items)
    if not evidence:
        return ""
    lines = ["", "Human-only exact-head evidence obligations (not code findings):"]
    for item in evidence:
        state = (
            f"frozen at head `{item.candidate_head_sha}`"
            if item.lifecycle == "evidence_frozen"
            else "deferred until every code finding and machine gate is clean"
        )
        lines.append(f"- [{item.item_id}] {_evidence_requester(item)}; {state}: {item.text}")
    lines.append(
        "These are machine-owned records. Disposition each one: `blocking` with a short "
        "note keeps a request that is still needed; `resolved` is only a note unless the "
        "response rule below applies. A review whose only open items are kept evidence "
        "requests uses `state: blocking` with empty `blocking_items` and is treated as "
        "approving the code. Never list missing human evidence in `blocking_items`."
    )
    if response_head:
        lines.append(
            f"This is an evidence-response re-review at frozen head `{response_head}` after "
            "new signed human input. If you requested an item, dispose it `resolved` only "
            "when the signed input supplies adequate evidence for exactly this head or "
            "withdraws the request; otherwise keep it `blocking` with a note. Other "
            "reviewers' dispositions on your item are notes only. Do not re-emit a request "
            "you just resolved. Report any code defect the evidence reveals in "
            "`blocking_items`."
        )
    if any(item.converted_row_id is not None for item in evidence):
        lines.append(
            "A converted-row obligation stands for an approved risk-matrix row whose evidence "
            "could not be tool-captured. Every listed owner still on the board must dispose it "
            "`resolved` at the frozen head for the row to be verified as operator-signed. Do "
            "not re-tag a converted row in `evidence_row_ids`; a new finding naming it is an "
            "ordinary code finding."
        )
    return "\n".join(lines) + "\n"


def _finalize_ordinary_recovery_checked(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    items: Sequence[UnresolvedReviewItem],
    current_head_sha: str | None,
    capability: OrdinaryRecoveryCapability,
) -> bool:
    """Run ordinary recovery only after every unrelated obligation is clear."""
    _ensure_finalization_ready(
        pr_number=pr_number,
        round_number=round_number,
        items=items,
        current_head_sha=current_head_sha,
        ignored_machine_kinds=frozenset({"github-pr-checks"}),
        config=config,
    )
    return _finalize_ordinary_recovery_merge(
        runner,
        config=config,
        pr_number=pr_number,
        capability=capability,
    )


def _mergeability_for_unreadable_protection(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    checks: PullRequestChecks | None,
) -> PullRequestMergeability | None:
    """Fetch GitHub's merge state only when classic protection returned 403."""
    if checks is None or checks.branch_protection_status != "forbidden":
        return None
    return get_pr_mergeability(runner, config=config, pr_number=pr_number)


def _ordinary_snapshot_nonauthority_reason(
    checks: PullRequestChecks | None,
    mergeability: PullRequestMergeability | None = None,
    *,
    head_sha: str | None = None,
    defer_unreadable_protection: bool = False,
) -> str:
    """Name the first unmet conjunct of ``_ordinary_checks_snapshot_is_authoritative``.

    Empty exactly when that predicate is True. Edit both functions together.
    """
    if checks is None:
        return "the check board is unavailable"
    if checks.state != "passing":
        return f"the aggregate state is {checks.state}"
    if checks.check_query_status != "ok":
        return f"the check query status is {checks.check_query_status}"
    if not (
        board_protection_is_reliable(checks, mergeability, head_sha=head_sha)
        or (defer_unreadable_protection and checks.branch_protection_status == "forbidden")
    ):
        merge_state = mergeability.merge_state_raw if mergeability is not None else None
        return (
            f"branch protection is not reliable for the head (protection status "
            f"{checks.branch_protection_status}, merge state {merge_state or 'unknown'})"
        )
    if checks.pending:
        return "checks are still pending: " + ", ".join(c.name for c in checks.pending)
    if checks.missing_required:
        return "required checks are missing: " + ", ".join(checks.missing_required)
    successful = [c for c in checks.passing if c.status.strip().lower() == "success"]
    if not successful:
        return "no check has a real success conclusion (only skipped or neutral)"
    required = set(checks.required_checks)
    success_names = {c.name for c in successful if c.name in required}
    if success_names != required:
        return "required checks without a success conclusion: " + ", ".join(
            sorted(required - success_names)
        )
    for check in checks.passing:
        if check.name in required and check.status.strip().lower() != "success":
            return (
                f"required check {check.name} has a non-success {check.kind} "
                f"observation ({check.status})"
            )
    return ""


def _ordinary_checks_snapshot_is_authoritative(
    checks: PullRequestChecks | None,
    mergeability: PullRequestMergeability | None = None,
    *,
    head_sha: str | None = None,
    defer_unreadable_protection: bool = False,
) -> bool:
    """Return whether a fresh ordinary-check snapshot can clear its ledger.

    ``get_pr_checks`` is queried against the current PR head immediately before
    this decision. Requiring both API surfaces and a known branch-protection
    result keeps a passing-looking partial, absent, or unavailable snapshot
    from becoming a final gate. An unreadable (403) classic protection is
    accepted only with GitHub's ``CLEAN`` merge state for ``head_sha``, unless
    ``defer_unreadable_protection`` says the caller checks ``CLEAN`` itself
    after readiness and before merging (a draft reports ``DRAFT``).

    Keep ``_ordinary_snapshot_nonauthority_reason`` in step with this predicate.
    """
    # ``PullRequestChecks.state`` deliberately treats neutral and skipped
    # conclusions as passing for ordinary status reporting.  That aggregate is
    # useful for display, but it is not evidence that a final gate actually
    # ran.  A clearing snapshot must contain a real success conclusion; when
    # branch protection names required checks, each of those checks must also
    # have that conclusion.
    successful_checks = tuple(
        check for check in (checks.passing if checks is not None else ())
        if check.status.strip().lower() == "success"
    )
    required_names = set(checks.required_checks) if checks is not None else set()
    required_success_names = {
        check.name for check in successful_checks if check.name in required_names
    }
    required_observations = (
        check for check in (checks.passing if checks is not None else ())
        if check.name in required_names
    )
    return bool(
        checks is not None
        and checks.state == "passing"
        and checks.check_query_status == "ok"
        and (
            board_protection_is_reliable(checks, mergeability, head_sha=head_sha)
            or (
                defer_unreadable_protection
                and checks.branch_protection_status == "forbidden"
            )
        )
        and not checks.pending
        and not checks.missing_required
        and successful_checks
        and required_success_names == required_names
        and all(
            check.status.strip().lower() == "success"
            for check in required_observations
        )
    )


def _managed_success_supersedes_ordinary_checks(
    items: Sequence[UnresolvedReviewItem],
    *,
    outcome: ManagedCiOutcome,
    current_head_sha: str | None,
    mergeability: PullRequestMergeability | None,
) -> tuple[str, str]:
    """Decide whether a correlated managed success retires carried ordinary checks (#1117).

    Returns ``(verdict, predicate)`` with verdict ``not_applicable``, ``cleared``
    or ``unqualified``. Only a ``github-pr-checks`` revalidation candidate at the
    qualified head is considered; the judgement is made from the full counted
    exact-head board and never from obligation text.
    """
    candidates = [
        item
        for item in items
        if item.obligation_kind == "github-pr-checks"
        and _machine_obligation_is_revalidation_candidate(item, current_head_sha=current_head_sha)
    ]
    if not candidates:
        return "not_applicable", ""
    checks = outcome.checks
    if checks is None or not current_head_sha or outcome.head_sha != current_head_sha:
        return "unqualified", "the qualified exact-head check board is unavailable or not for the current head"
    if checks.check_query_status != "ok" or checks.check_query_errors:
        errors = "; ".join(checks.check_query_errors) or checks.check_query_status
        return "unqualified", f"the exact-head check board had query errors ({errors})"
    if not checks.listing_complete:
        return "unqualified", "the exact-head check listing is incomplete (total_count mismatch or unparsed entries)"
    final = [check for check in (*checks.passing, *checks.pending, *checks.failing) if check.name == FINAL_CONTEXT]
    if not any(check.status.strip().lower() == "success" for check in final):
        return "unqualified", f"`{FINAL_CONTEXT}` has no success observation"
    if checks.failing or checks.pending or checks.missing_required:
        names = [
            check.name for check in (*checks.failing, *checks.pending)
        ] + list(checks.missing_required)
        return "unqualified", f"checks are failing, pending or missing at the qualified head: {', '.join(names)}"
    required = set(checks.required_checks)
    for check in checks.shadowed:
        state = _classify_check_status(check.status)
        if state in {"failing", "pending"}:
            return "unqualified", f"a same-name observation of `{check.name}` is {check.status.lower()}"
        if check.name in required and check.status.strip().lower() != "success":
            return "unqualified", (
                f"required check `{check.name}` has a same-name observation that is {check.status.lower()}"
            )
    if not _ordinary_checks_snapshot_is_authoritative(checks, mergeability, head_sha=current_head_sha):
        if protection_awaits_readiness(checks, mergeability, head_sha=current_head_sha):
            return "unqualified", (
                f"branch protection is unreadable (HTTP 403) and GitHub reports DRAFT for draft PR at "
                f"{current_head_sha}, so required contexts cannot be verified before readiness; grant the "
                "token read access to branch protection (administration: read), then resume. Do not mark "
                "the PR ready manually: a ready PR that still carries the managed label is a mixed "
                "lifecycle that managed resume refuses"
            )
        if checks.branch_protection_status == "forbidden":
            state = mergeability.merge_state_raw if mergeability is not None else None
            return "unqualified", (
                "branch protection is unreadable (HTTP 403) and GitHub's merge state is "
                f"{state or 'unavailable'} rather than CLEAN for {current_head_sha}"
            )
        for observed in checks.passing:
            if observed.name in required and observed.status.strip().lower() != "success":
                return "unqualified", (
                    f"required check `{observed.name}` is {observed.status.lower()} rather than a "
                    "real success"
                )
        for name in sorted(required):
            if not any(
                observed.name == name and observed.status.strip().lower() == "success"
                for observed in checks.passing
            ):
                return "unqualified", f"required check `{name}` has no real success at the qualified head"
        return "unqualified", "the exact-head board is not an authoritative success-only snapshot"
    return "cleared", ""


def _persist_qualification_checkpoint(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    head_sha: str | None,
    unresolved_items: Sequence[UnresolvedReviewItem],
    checkpoint: QualificationCheckpoint | None,
    message: str,
) -> None:
    """Write a resumable round-metadata checkpoint before risky continuation."""
    if checkpoint is None:
        return
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=_attach_round_metadata(
            message,
            PostedRoundMetadata(
                flow="pr",
                role="summary",
                agent="Orchestrator",
                seat_binding=reviewer_seat_binding(config),
                round_number=round_number,
                subject=str(head_sha or "unknown"),
                prior_items=tuple(unresolved_items),
                state="blocking",
                phase="qualification-checkpoint",
                qualification_checkpoint=checkpoint,
                **_architecture_metadata_fields(config),
            ),
        ),
    )


def _resume_pr_round_admitted(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    comments: Sequence[object],
    head_sha: str | None,
    configured_reviewers: Sequence[AgentName],
    reconciliation_mode: str = "aggregate",
) -> ResumedReviewRound | None:
    """Resume a PR round, admitting recovery records only from the authenticated actor.

    REST identities and the actor are fetched only when a recovery record, a
    head advance, or ``--review-unrecorded-head`` makes admission necessary; an
    incomplete read stops here, before any agent call.
    """
    trusted_actor: tuple[str, int] | None = None
    if not config.dry_run and pr_resume_needs_author_admission(
        comments, head_sha, config.review_unrecorded_head
    ):
        comments = merge_pr_comment_transport_identity(
            runner, config=config, pr_number=pr_number, comments=tuple(comments)  # type: ignore[arg-type]
        )
        missing = unauthenticated_recovery_record_indexes(
            comments, head_sha, config.review_unrecorded_head
        )
        if missing:
            raise AgentLoopError(
                "PR recovery records cannot be authenticated: the REST comment read did not "
                f"cover recovery record(s) at comment index {', '.join(map(str, missing))}; "
                "rerun once the complete comment history is readable."
            )
        trusted_actor = resolve_authenticated_github_actor(runner, config=config)
    return _resume_pr_round(
        comments,
        head_sha=head_sha,
        configured_reviewers=configured_reviewers,
        reconciliation_mode=reconciliation_mode,
        trusted_actor=trusted_actor,
        review_unrecorded_head=config.review_unrecorded_head,
    )


def _cached_trusted_actor(runner: Runner) -> tuple[str, int] | None:
    cached = getattr(runner, "_agent_loop_authenticated_actor", None)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached
    return None


def _post_recovery_record(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    message: str,
    metadata: PostedRoundMetadata,
) -> None:
    from .reviewer_seats import reviewer_seat_binding

    binding = reviewer_seat_binding(config)
    if binding is not None:
        metadata = dataclasses_replace(metadata, seat_binding=binding)
    # Writer-side assertion: a record this process posts must pass its own
    # per-phase structural validation, or recovery would refuse it later.
    problems = _recovery_record_problems(metadata)
    if problems:
        raise AgentLoopError(
            f"Refusing to post a malformed {metadata.phase} record: {', '.join(problems)}."
        )
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=_attach_round_metadata(message, metadata),
    )


def _persist_coder_dispatch(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    dispatch_round: int,
    dispatch_head: str | None,
    ledger: Sequence[UnresolvedReviewItem],
    budget: RecoveryRoundBudget,
    attempt: int,
    carried_reasons: Sequence[str] = (),
) -> None:
    """Record the slot, ledger, budget and attempt before a coder is dispatched."""
    if not dispatch_head or dispatch_head == "unknown":
        raise AgentLoopError(
            f"PR #{pr_number}: the PR head is unknown, so the coder dispatch cannot be recorded; "
            "no coder was dispatched."
        )
    _post_recovery_record(
        runner,
        config=config,
        pr_number=pr_number,
        message=(
            f"PR #{pr_number} coder dispatch record: round {dispatch_round}, attempt {attempt}, "
            f"head `{dispatch_head}`. Written before the coder is invoked so an interrupted "
            "or rejected turn can be resumed."
        ),
        metadata=PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=dispatch_round + 1,
            subject=dispatch_head,
            prior_items=tuple(ledger),
            state="blocking",
            phase=CODER_DISPATCH_PHASE,
            dispatch_round=dispatch_round,
            dispatch_head=dispatch_head,
            dispatch_attempt=attempt,
            recovery_dispatch=attempt > 1,
            carried_rejection_reasons=tuple(carried_reasons),
            recovery_round_budget=budget,
            **_architecture_metadata_fields(config),
        ),
    )


def _persist_rejected_coder_followup(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    dispatch_round: int,
    dispatch_head: str,
    observed_head: str,
    ledger: Sequence[UnresolvedReviewItem],
    budget: RecoveryRoundBudget,
    attempt: int,
    reason: object,
    carried_reasons: Sequence[str] = (),
) -> None:
    """Record that a coder follow-up was rejected, keyed to the head observed afterwards."""
    bounded = sanitize_recovery_reason(reason)
    _post_recovery_record(
        runner,
        config=config,
        pr_number=pr_number,
        message=(
            f"PR #{pr_number} rejected coder follow-up record: the response for round "
            f"{dispatch_round} (attempt {attempt}, dispatched on `{dispatch_head}`, head now "
            f"`{observed_head}`) was not accepted: {bounded}"
        ),
        metadata=PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=dispatch_round,
            subject=observed_head,
            prior_items=tuple(ledger),
            state="blocking",
            phase=CODER_FOLLOWUP_REJECTED_PHASE,
            dispatch_round=dispatch_round,
            dispatch_head=dispatch_head,
            dispatch_attempt=attempt,
            recovery_dispatch=attempt > 1,
            carried_rejection_reasons=tuple(carried_reasons),
            rejected_coder_followup_reason=bounded,
            rejected_coder_followup_from_head=dispatch_head,
            recovery_round_budget=budget,
            **_architecture_metadata_fields(config),
        ),
    )


def _record_coder_followup_rejection(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    error: BaseException,
    dispatch_round: int,
    pre_turn_head: str,
    ledger: Sequence[UnresolvedReviewItem],
    budget: RecoveryRoundBudget,
    attempt: int,
    recovery_dispatch: bool,
    carried_reasons: Sequence[str] = (),
) -> None:
    """Persist a rejected coder follow-up when it left a recoverable state (#1292).

    A rejection after a push strands the new head with no coder or reviewer
    record; a rejected recovery dispatch must keep its attempt count even with
    an unchanged head.  An ordinary first-attempt rejection on an unchanged head
    writes nothing.  Never raises: the caller re-raises the original error.
    """
    try:
        observed_head = get_pr_head_sha(runner, config, pr_number)
        observed_head = (observed_head or "").strip()
        if not observed_head or (observed_head == pre_turn_head and not recovery_dispatch):
            return
        _persist_rejected_coder_followup(
            runner,
            config=config,
            pr_number=pr_number,
            dispatch_round=dispatch_round,
            dispatch_head=pre_turn_head,
            observed_head=observed_head,
            ledger=ledger,
            budget=budget,
            attempt=attempt,
            reason=str(error) or "rejected without a reason",
            carried_reasons=carried_reasons,
        )
    except Exception as secondary:  # noqa: BLE001 - never mask the rejection
        log(
            config,
            f"PR #{pr_number}: could not record the rejected coder follow-up "
            f"({secondary}); the coder-dispatch record still carries the attempt",
        )


def _persist_head_review_recovery(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    round_number: int,
    head_sha: str | None,
    ledger: Sequence[UnresolvedReviewItem],
    budget: RecoveryRoundBudget,
    source: str,
) -> None:
    """Write the handoff that makes an ordinary head-review recovery resumable."""
    _post_recovery_record(
        runner,
        config=config,
        pr_number=pr_number,
        message=(
            f"PR #{pr_number} head review recovery record ({source}): the current head "
            f"`{head_sha}` gets an ordinary review round {round_number} with the recorded "
            "active items carried. No coder record was written."
        ),
        metadata=PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=round_number,
            subject=str(head_sha or "unknown"),
            prior_items=tuple(ledger),
            state="blocking",
            phase=HEAD_REVIEW_RECOVERY_PHASE,
            head_review_recovery_source=source,
            recovery_round_budget=budget,
            **_architecture_metadata_fields(config),
        ),
    )


def _visibility_snapshot(
    *,
    fresh_records: Sequence[PostedRoundRecord] | None,
    base_records: Sequence[PostedRoundRecord],
    base_length: int,
    checkpoint: PostedRoundMetadata | None,
    own_conversion: PostedRoundMetadata | None = None,
) -> tuple[tuple[PostedRoundRecord, ...], bool]:
    """History as visible once this invocation's own scheduler checkpoint is posted (#1156).

    Prefers the refreshed history when it holds the checkpoint; otherwise the
    pre-post history plus a synthetic record for it, never the stale evidence
    alone.  Returns the records and whether the checkpoint is present.

    ``own_conversion`` is a conversion record this invocation itself posted
    and the base history lacks (#1664).  When the refreshed history is
    unreadable it is added as a synthetic record before the checkpoint, so
    the fallback keeps the conversion's visibility boundary.  Only a record
    this process wrote may be passed; nothing read from history is ever
    synthesized.
    """
    if checkpoint is None:
        return tuple(base_records), False
    base_top = max((record.index for record in base_records), default=-1)
    if fresh_records is None:
        history: tuple[PostedRoundRecord, ...] = tuple(base_records)
    else:
        history = tuple(fresh_records)
        if any(
            record.index > base_top
            and record.metadata.flow == checkpoint.flow
            and record.metadata.role == "summary"
            and record.metadata.phase == "scheduler-prelaunch"
            and record.metadata.round_number == checkpoint.round_number
            and record.metadata.subject == checkpoint.subject
            for record in history
        ):
            return history, True
    # The refreshed history (when readable) keeps every publication it
    # exposed; only the checkpoint this invocation just posted is added.
    top = max((record.index for record in history), default=-1)
    next_index = max(base_length, base_top + 1, top + 1)
    added: list[PostedRoundRecord] = []
    if own_conversion is not None and fresh_records is None:
        added.append(PostedRoundRecord(index=next_index, metadata=own_conversion, body=""))
        next_index += 1
    added.append(PostedRoundRecord(index=next_index, metadata=checkpoint, body=""))
    return (*history, *added), True


def _latest_pr_reviewer_records(
    records: Sequence[PostedRoundRecord],
    configured_reviewers: Sequence[AgentName],
) -> dict[str, PostedRoundRecord]:
    """Return the latest persisted review for every configured reviewer.

    This intentionally searches all candidate subjects.  The exact-head
    approval helper cannot be used here because a stale-head record is the
    evidence needed to decide whether a returning reviewer needs a fresh
    context and session.
    """
    configured_names = {agent_display_name(reviewer) for reviewer in configured_reviewers}
    latest: dict[str, PostedRoundRecord] = {}
    for record in reversed(records):
        if record.metadata.role != "reviewer" or record.metadata.agent not in configured_names:
            continue
        latest.setdefault(record.metadata.agent, record)
    return latest


def _reviewer_needs_fresh_context(
    reviewer: AgentName,
    *,
    selective_policy: bool,
    current_head_sha: str,
    current_round: int,
    latest_reviewer_records: Mapping[str, PostedRoundRecord],
) -> bool:
    """Identify a reviewer that must receive a new full-context turn.

    A reviewer with no persisted review, a review of an older head, or a
    review that missed an intervening round cannot safely reuse a compact
    session.  In particular, the lookup must not be limited to exact-head
    approvals: those records intentionally exclude the stale record we need
    to inspect.
    """
    if not selective_policy:
        return False
    record = latest_reviewer_records.get(agent_display_name(reviewer))
    if record is None:
        return True
    return (
        record.metadata.subject != current_head_sha
        or record.metadata.round_number < current_round - 1
    )


def _reviewer_diff_summary(
    runner: Runner,
    *,
    checkout: Path,
    last_reviewed_sha: str | None,
    current_head_sha: str,
) -> str:
    """Render a bounded, orchestrator-observed diff summary for a returner."""
    if not last_reviewed_sha:
        return "No prior reviewer SHA was recorded; inspect the complete base-to-head diff."
    if last_reviewed_sha == current_head_sha:
        return "The last reviewed SHA is the current head; inspect the complete base-to-head diff for any missed context."
    try:
        result = runner.run(
            ["git", "diff", "--name-status", "--find-renames", "--find-copies", last_reviewed_sha, current_head_sha],
            cwd=checkout,
            check=False,
        )
        if result.returncode != 0:
            return (
                f"The diff from {last_reviewed_sha} to {current_head_sha} was unavailable; "
                "inspect the complete base-to-head diff independently."
            )
        paths = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        if not paths:
            return (
                f"The diff from {last_reviewed_sha} to {current_head_sha} had no name-status output; "
                "inspect the complete base-to-head diff independently."
            )
        displayed = paths[:32]
        suffix = "; ..." if len(paths) > len(displayed) else ""
        return (
            f"Observed diff from {last_reviewed_sha} to {current_head_sha}: "
            + "; ".join(displayed)
            + suffix
            + ". Inspect the complete base-to-head diff independently."
        )
    except (OSError, AttributeError, TypeError):
        return (
            f"The diff from {last_reviewed_sha} to {current_head_sha} could not be observed; "
            "inspect the complete base-to-head diff independently."
        )


def _reviewer_history_is_reconstructible(
    runner: Runner,
    *,
    checkout: Path,
    record: PostedRoundRecord | None,
    current_head_sha: str,
) -> bool:
    """Check that a returning reviewer's own history can be reconstructed.

    A reviewer may legitimately miss several narrow coder rounds. Their older
    reviewed SHA is still sufficient when Git can observe that SHA as an
    ancestor and can produce the complete span diff. This check is intentionally
    independent of the scheduler's previous transition SHA: trusting that SHA
    here would turn normal selective pauses into alternating full-board rounds.
    """
    if record is None:
        return False
    last_sha = record.metadata.subject
    if not last_sha or last_sha == current_head_sha:
        return bool(last_sha)
    try:
        ancestry = runner.run(
            ["git", "merge-base", "--is-ancestor", last_sha, current_head_sha],
            cwd=checkout,
            check=False,
        )
        if ancestry.returncode != 0:
            return False
        diff = runner.run(
            [
                "git", "diff", "--name-status", "--find-renames", "--find-copies",
                last_sha, current_head_sha,
            ],
            cwd=checkout,
            check=False,
        )
        return diff.returncode == 0
    except (OSError, AttributeError, TypeError):
        return False


def _returning_reviewer_context(
    runner: Runner,
    *,
    reviewer: AgentName,
    checkout: Path,
    current_head_sha: str,
    current_round: int,
    latest_reviewer_records: Mapping[str, PostedRoundRecord],
) -> str:
    record = latest_reviewer_records.get(agent_display_name(reviewer))
    last_sha = record.metadata.subject if record is not None else None
    if record is None:
        history = "No prior review record is available for this reviewer."
    else:
        missed_rounds = max(0, current_round - record.metadata.round_number - 1)
        history = (
            f"Last reviewed SHA: {record.metadata.subject}; "
            f"last attended round: {record.metadata.round_number}."
            + (
                f" Missed {missed_rounds} intervening review round(s); reconstruct context from the current history."
                if missed_rounds else ""
            )
        )
    return (
        "\nReturning reviewer handoff context (orchestrator-derived):\n"
        f"{history}\n"
        + _reviewer_diff_summary(
            runner,
            checkout=checkout,
            last_reviewed_sha=last_sha,
            current_head_sha=current_head_sha,
        )
        + "\nYour review must inspect the complete current base-to-head diff; the coder's summary is not a substitute.\n"
    )


def _all_pending_resolution_owners_unavailable(
    item: UnresolvedReviewItem,
    unavailable_names: set[str],
) -> bool:
    """Return true only when a non-empty pending owner set is unavailable."""
    owners = item.resolution_owners or (item.reviewer,)
    states = dict(item.owner_states or ((owner, "pending") for owner in owners))
    pending = [owner for owner in owners if states.get(owner) != "cleared"]
    return bool(pending) and all(owner in unavailable_names for owner in pending)


def _observe_pr_transition(
    runner: Runner,
    *,
    checkout: Path,
    previous_sha: str | None,
    current_sha: str | None,
    scopes: Sequence[str],
    broad_rules: Sequence[str],
    obligations: Sequence[ReviewObligation] = (),
) -> TransitionClassification:
    """Collect repository-observed ancestry and exact diff facts."""
    if not previous_sha or not current_sha or previous_sha == current_sha:
        return TransitionClassification("broad", "missing or unchanged review SHA")
    try:
        ancestry = runner.run(
            ["git", "merge-base", "--is-ancestor", previous_sha, current_sha],
            cwd=checkout,
            check=False,
        )
        if ancestry.returncode != 0:
            return TransitionClassification("broad", "candidate history is not an available ancestor")
        names = runner.run(
            [
                "git", "diff", "--name-status", "-z", "--find-renames", "--find-copies",
                previous_sha, current_sha,
            ],
            cwd=checkout,
            check=False,
        )
        if names.returncode != 0:
            return TransitionClassification("broad", "complete exact Git diff was unavailable")
        raw = names.stdout
        if not raw:
            return TransitionClassification("broad", "transition contained no observable diff")
        tokens = raw.split("\0")
        changes: list[GitChange] = []
        index = 0
        while index < len(tokens):
            status = tokens[index]
            index += 1
            if not status:
                continue
            code = status[:1].upper()
            if code in {"R", "C"}:
                # NUL-formatted rename/copy entries carry old and new paths.
                if index + 1 >= len(tokens):
                    return TransitionClassification("broad", "diff contained an incomplete rename/copy record")
                old_path, new_path = tokens[index], tokens[index + 1]
                index += 2
                changes.append(GitChange(old_path, status=status))
                changes.append(GitChange(new_path, status=status))
            else:
                if index >= len(tokens):
                    return TransitionClassification("broad", "diff contained an incomplete name-status record")
                changes.append(GitChange(tokens[index], status=status))
                index += 1
        # A binary check is separate from name-status because a binary file can
        # otherwise look exactly like an ordinary modification.
        numstat = runner.run(
            ["git", "diff", "--numstat", previous_sha, current_sha],
            cwd=checkout,
            check=False,
        )
        if numstat.returncode != 0:
            return TransitionClassification("broad", "binary/text diff classification was unavailable")
        binary_paths = {
            line.rsplit("\t", 1)[-1]
            for line in numstat.stdout.splitlines()
            if line.startswith("-\t-\t")
        }
        mode = runner.run(
            ["git", "diff", "--summary", previous_sha, current_sha],
            cwd=checkout,
            check=False,
        )
        if mode.returncode != 0:
            return TransitionClassification("broad", "diff mode-change classification was unavailable")
        added_paths = {
            change.path
            for change in changes
            if change.status[:1].upper() == "A"
        }
        unsafe_summary = False
        for summary_line in mode.stdout.splitlines():
            lowered = summary_line.lower()
            create_marker = "create mode 100644 "
            create_index = lowered.find(create_marker)
            if create_index >= 0:
                # Git reports the mode of an ordinary newly-added text file in
                # the summary.  The name-status record is authoritative for
                # distinguishing that safe A entry from an executable,
                # symlink, or other mode/type change.
                created_path = summary_line[create_index + len(create_marker):].strip()
                if created_path in added_paths:
                    continue
            if any(
                token in lowered
                for token in ("mode change", "create mode", "delete mode", "submodule", "rename", "copy")
            ):
                unsafe_summary = True
                break
        if binary_paths or unsafe_summary:
            return TransitionClassification("broad", "diff contained binary or mode changes")
        return classify_transition(
            previous_sha,
            current_sha,
            changes,
            scopes=scopes,
            broad_rules=broad_rules,
            obligations=obligations,
        )
    except (OSError, AttributeError, TypeError):
        return TransitionClassification("broad", "repository history or diff observation failed")


def _scheduler_contract_from_metadata(
    metadata: PostedRoundMetadata,
) -> ReviewSchedulingContract | None:
    if metadata.scheduler_contract is None:
        return None
    from .review_scheduling import ReviewSchedulingContract as _Contract

    return _Contract.from_mapping(metadata.scheduler_contract)


def _pr_contract_drift_error(
    persisted: ReviewSchedulingContract,
    detail: str,
    *,
    pr_number: int,
    configured: ReviewSchedulingContract | None,
    start_round_number: Callable[[], int | None],
    during: str = "resume",
    amendments_recognized: bool = False,
) -> AgentLoopError:
    """The fail-closed PR contract-drift error, with the amendment route (#943)."""
    return AgentLoopError(
        f"PR review scheduler contract changed during {during}; "
        + ("no qualification or merge is permitted; " if during == "qualification" else "")
        + "required reviewers, policy, and broad-path rules must remain immutable. PR "
        f"#{pr_number} carries a scheduler contract for policy {persisted.policy} with "
        f"primary {persisted.primary_reviewer or '(none)'} and reviewer board "
        f"{', '.join(persisted.required_reviewers)} ({detail})."
        + (
            ""
            if during == "qualification"
            else _stale_amendment_repost_clause(detail, start_round_number)
            + _board_amendment_route_clause(
                flow="pr",
                issue_number=None,
                pr_number=pr_number,
                persisted=persisted,
                configured=configured,
                start_round_number=start_round_number,
                amendments_recognized=amendments_recognized,
            )
        )
    )


REDUCED_BOARD_COMPLETION_HEADING = "Review completed on a reduced reviewer board."


def _post_reduced_board_completion_note(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    digest: str,
    note: str,
) -> None:
    """Post one plain completion note per amendment digest (no round metadata)."""
    try:
        comments = get_pr_review_context(runner, config=config, pr_number=pr_number).comments
    except AgentLoopError:
        comments = ()
    if any(
        REDUCED_BOARD_COMPLETION_HEADING in (getattr(comment, "body", "") or "")
        and digest in (getattr(comment, "body", "") or "")
        for comment in comments
    ):
        return
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=(
            f"{REDUCED_BOARD_COMPLETION_HEADING}\n\n- {note}\n"
            f"- Signed amendment digest: `{digest}`\n\n-- Orchestrator"
        ),
    )


def _pr_amendment_start_round(
    pr_context: PullRequestReviewContext,
    configured_reviewers: Sequence[AgentName],
    scheduler_capabilities: object,
    *,
    runner: Runner | None = None,
    config: AgentLoopConfig | None = None,
) -> int:
    """The round a PR resume would re-enter; only used to fill an error template."""
    try:
        resumed = _resume_pr_round(
            pr_context.comments,
            head_sha=pr_context.metadata.head_sha,
            configured_reviewers=configured_reviewers,
            reconciliation_mode=(
                "owner-scoped"
                if getattr(scheduler_capabilities, "owner_scoped_reconciliation", False)
                else "aggregate"
            ),
            trusted_actor=_cached_trusted_actor(runner) if runner is not None else None,
            review_unrecorded_head=bool(config is not None and config.review_unrecorded_head),
        )
    except AgentLoopError:
        # Only filling an error template; the real resume reports the refusal.
        return 1
    from .board_change import board_reentry_round

    return board_reentry_round(
        pr_context.comments, resumed.round_number if resumed is not None else None,
        signed_amendment=True,
    )


def _is_completed_full_board_scheduler_record(
    record: PostedRoundRecord,
    *,
    scheduler_contract: ReviewSchedulingContract,
) -> bool:
    """Return whether ``record`` is a completed conservative recovery point.

    Invalid scheduler metadata must force the next scheduling decision to the
    full board.  Once that decision has completed, however, the invalid record
    is historical state and must not permanently veto qualification on every
    resume.  The reconciliation checkpoint is the durable completion marker;
    a prelaunch checkpoint or an individual reviewer record is not sufficient.
    """
    metadata = record.metadata
    if (
        metadata.scheduler_metadata_status != "valid"
        or metadata.role != "summary"
        or metadata.phase != "reconciliation"
        or metadata.scheduler_current_sha != metadata.subject
        or set(metadata.scheduler_selected_reviewers)
        != set(scheduler_contract.required_reviewers)
        or metadata.scheduler_paused_reviewers
    ):
        return False
    try:
        return _scheduler_contract_from_metadata(metadata) == scheduler_contract
    except AgentLoopError:
        return False


def _managed_binding_retired_plan_hashes(
    handoff: AuthenticatedIssueCreatedHandoff | None,
) -> frozenset[str]:
    """Plans a verified signed rebind retired for this managed handoff (#993)."""
    return handoff.retired_plan_hashes if handoff is not None else frozenset()


def _managed_binding_protection_mode(
    handoff: AuthenticatedIssueCreatedHandoff | None,
) -> str | None:
    """Return ``strict`` only for a handoff that publishes no PR-side record.

    Authorization records are skipped exactly when the authenticated handoff
    carries no override nonce; any other handoff must bind through them.
    """
    if handoff is not None and handoff.protection_mode == "strict" and handoff.override_nonce is None:
        return "strict"
    return None


_PR_AMENDMENT_PLAN_BOARD_HINT = (
    " Rerun with the original reviewer board configured; the signed PR amendment still "
    "removes the reviewer from PR review."
)


def _pr_amendment_plan_board_hint(
    pr_comments: Sequence[object],
    *,
    pr_number: int,
    supplied_reviewers: Sequence[AgentName],
) -> str:
    """Rerun hint when a PR-only amendment explains a reduced supplied board (#1133)."""
    try:
        amendments = collect_reviewer_board_amendments(
            pr_comments, flow="pr", pr_number=pr_number
        )
    except AgentLoopError:
        return ""
    supplied = {agent_display_name(reviewer) for reviewer in supplied_reviewers}
    for amendment in amendments:
        original = set(amendment.original_required_reviewers)
        if any(name not in supplied for name in amendment.removed_reviewers) and supplied <= original:
            return _PR_AMENDMENT_PLAN_BOARD_HINT
    return ""


def _verify_strict_managed_plan_binding(
    *,
    config: AgentLoopConfig,
    pr_number: int,
    issue_context: IssueContext,
    metadata: PullRequestMetadata,
    expected_plan_hash: str,
    pr_comments: Sequence[object] = (),
    runner: Runner | None = None,
) -> None:
    """Bind a strict-protection managed PR to the issue's canonical plan.

    A strictly protected base never publishes a PR-side authorization record,
    and managed recovery never synthesizes the issue-side handoff.  The durable
    binding is therefore the one the managed resume itself used: the reserved
    managed branch for this issue and the issue's canonical, completely
    approved plan, whose hash must still be the plan the reviewers were bound
    to.  GitHub's exact-head protection independently gates the merge.
    """

    def fail(reason: str) -> AgentLoopError:
        return AgentLoopError(
            "Approved-plan/handoff identity changed or disappeared during PR qualification; "
            f"the strict managed-CI binding does not tie PR #{pr_number} to approved plan "
            f"{expected_plan_hash} ({reason}). Stale approvals cannot be used for this head."
            + _pr_amendment_plan_board_hint(
                pr_comments, pr_number=pr_number, supplied_reviewers=reviewers(config)
            )
        )

    if metadata.head_branch != f"agent-loop/managed-{issue_context.number}" or not metadata.head_sha:
        raise fail(f"the PR is not the reserved managed branch for issue #{issue_context.number}")
    # The approving plan board comes from the issue's own history, never the
    # PR board (#1373).  Without plan records there is nothing to derive.
    from .plan_board_state import admitted_issue_comments
    from .plan_verification import plan_verification_inputs

    try:
        # The plan's own lineage, admitted on the same fresh fetch (#1377).
        issue_comments, trusted_actor = admitted_issue_comments(
            runner, config=config, issue_number=issue_context.number,
            comments=issue_context.comments,
        )
        plan_config, plan_comments = plan_verification_inputs(
            config, issue_comments, issue_number=issue_context.number,
            trusted_actor=trusted_actor,
        )
    except AgentLoopError as exc:
        raise fail(f"the approving plan board cannot be verified: {exc}") from exc
    resumed_plan = _resume_plan_round(
        plan_comments,
        configured_reviewers=reviewers(plan_config),
    )
    if resumed_plan is None:
        raise fail("the issue carries no canonical approved plan")
    plan_text, plan_round = resumed_plan
    _require_complete_canonical_plan_approval(
        plan_comments,
        config=plan_config,
        plan_text=plan_text,
        plan_round=plan_round,
        human_requirements=issue_context.human_requirements,
        error_message=str(fail("the canonical plan is not completely approved")),
    )
    from .plan_verification import require_recoverable_plan_approval

    try:
        require_recoverable_plan_approval(
            runner, config=config, issue_context=issue_context, comments=issue_comments,
            trusted_actor=trusted_actor, error_message="the canonical plan is not completely approved",
        )
    except AgentLoopError as exc:
        raise fail(str(exc)) from exc
    if approved_plan_hash(plan_text) != expected_plan_hash:
        raise fail("the issue's canonical approved plan changed")


def _fresh_pr_qualification_snapshot(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    issue_context: IssueContext | None,
    parent_issue_context: IssueContext | None,
    approved_plan_context: ApprovedPlanContext | None = None,
    scheduler_contract: ReviewSchedulingContract | None = None,
    allow_plan_handoff_change: bool = False,
    planning_child_binding: _PlanningChildBinding | None = None,
    managed_protection_mode: str | None = None,
    managed_retired_plan_hashes: frozenset[str] = frozenset(),
    plan_binding_reviewers: Sequence[AgentName] | None = None,
    board_check: Callable[[Sequence[object]], None] | None = None,
) -> tuple[PullRequestReviewContext, tuple[str, ...], ApprovedPlanContext | None, AgentLoopConfig]:
    """Refetch the PR-side qualification inputs immediately before a gate.

    ``plan_binding_reviewers`` is the operator-supplied board the issue plan is
    re-verified against; it differs from ``config.reviewer`` when a signed PR
    amendment reduced the effective PR board (#1133).
    """
    staged_owner = (
        approved_plan_context.risk_test_matrix_execution_owner
        if approved_plan_context is not None
        else None
    )
    context = get_pr_review_context(runner, config=config, pr_number=pr_number)
    approved_identity = _latest_pr_approval_architecture_identity(
        context.comments, head_sha=context.metadata.head_sha
    )
    stored_identity = _latest_pr_architecture_observation(
        context.comments, head_sha=context.metadata.head_sha
    )
    if stored_identity is None:
        # A reviewer approval is the durable resume authority when no
        # qualification checkpoint has recorded a newer observation. Never
        # invent an identity for legacy records.
        stored_identity = approved_identity
    if stored_identity is None:
        stored_identity = _latest_pr_architecture_observation(context.comments)
    fresh_architecture, architecture_changed = _revalidate_pr_architecture_identity(
        runner, config=config, metadata=context.metadata, stored_identity=stored_identity
    )
    if fresh_architecture is not None:
        # AgentLoopConfig is frozen and hashable.  Rebind a replacement config
        # instead of mutating the captured instance during a qualification gate.
        config = dataclasses_replace(config, architecture_context=fresh_architecture)
    if architecture_changed and fresh_architecture is not None:
        from .architecture_contract import _pr_architecture_observation_record, _architecture_identity_differences
        source = _pr_architecture_observation_record(context.comments, head_sha=context.metadata.head_sha)
        source = source or _pr_architecture_observation_record(context.comments)
        provenance = (
            f"{source.metadata.role} round {source.metadata.round_number} subject {source.metadata.subject}"
            if source is not None else "in-memory acquisition"
        )
        differences = _architecture_identity_differences(stored_identity, fresh_architecture.identity())
        log(config, f"Architecture identity changed from {provenance}; differing fields: {', '.join(differences)}")
        context = dataclasses_replace(context, architecture_identity_changed=True)
    if scheduler_contract is not None:
        # Scheduler metadata is an optimization over the immutable required
        # reviewer contract. A fresh qualification read must never silently
        # accept malformed or conflicting audit state, including a record
        # posted while managed CI was running.
        fresh_scheduler_records = _extract_round_metadata_records(
            context.comments, flow="pr"
        )
        superseded_invalid_indexes: set[int] = set()
        for record_position, record in enumerate(fresh_scheduler_records):
            status = record.metadata.scheduler_metadata_status
            if status == "invalid":
                if any(
                    candidate.index > record.index
                    and _is_completed_full_board_scheduler_record(
                        candidate,
                        scheduler_contract=scheduler_contract,
                    )
                    for candidate in fresh_scheduler_records[record_position + 1 :]
                ):
                    # Resume recovery deliberately selected the full board.
                    # A later completed reconciliation checkpoint supersedes
                    # this historical malformed optimization record.
                    superseded_invalid_indexes.add(record.index)
                    continue
                raise AgentLoopError(
                    "Malformed or contradictory PR review scheduler metadata was observed "
                    "during qualification; no qualification or merge is permitted."
                )
            if status != "valid":
                continue
            try:
                _scheduler_contract_from_metadata(record.metadata)
            except AgentLoopError as exc:
                raise AgentLoopError(
                    "Malformed PR review scheduler contract was observed during qualification; "
                    "no qualification or merge is permitted."
                ) from exc
            if record.metadata.scheduler_current_sha != record.metadata.subject:
                raise AgentLoopError(
                    "Contradictory PR review scheduler head metadata was observed during "
                    "qualification; no qualification or merge is permitted."
                )
    # Signed amendments and board-change links are re-read from this same
    # fresh PR comment fetch (#943, #1376), so the gate never relies on a stale
    # or separately refreshed source.  The unified lineage is re-resolved for
    # every policy and the floor is rechecked.
    if board_check is None:
        from .board_change import identity_from_config
        from .pr_board_state import qualification_board_check

        board_check = qualification_board_check(
            runner, config=config, pr_number=pr_number, expected=identity_from_config(config),
        )
    board_check(context.comments)
    fresh_issue = issue_context
    fresh_parent = parent_issue_context
    fresh_approved_plan_context = approved_plan_context
    if issue_context is not None:
        fresh_issue = get_issue_context(runner, config=config, issue_number=issue_context.number)
        if scheduler_contract is not None:
            try:
                reject_misplaced_pr_amendments(
                    fresh_issue.comments, issue_number=fresh_issue.number
                )
            except AgentLoopError as exc:
                raise AgentLoopError(
                    f"{exc} No qualification or merge is permitted."
                ) from exc
    if parent_issue_context is not None:
        fresh_parent = get_issue_context(
            runner, config=config, issue_number=parent_issue_context.number
        )
    if approved_plan_context is not None:
        if fresh_issue is None or not approved_plan_context.plan_hash:
            raise AgentLoopError(
                "Approved-plan/handoff identity is missing during PR qualification; "
                "stale approvals cannot be used for this head."
            )
        fresh_handoff = find_latest_issue_pr_handoff(
            fresh_issue.comments,
            issue_number=fresh_issue.number,
            repo=config.repo,
        )
        if fresh_handoff is None and config.managed_ci:
            # Managed-CI recovery deliberately does not synthesize the
            # issue-side handoff (#966).  A voluntary or plan-limited base binds
            # through the trusted PR-side authorization chain; a strictly
            # protected base publishes no such record, so it is bound the way
            # its resume was: reserved branch plus the issue's canonical plan.
            if managed_protection_mode == "strict":
                _verify_strict_managed_plan_binding(
                    config=(
                        config
                        if plan_binding_reviewers is None
                        else dataclasses_replace(
                            config, reviewer=tuple(plan_binding_reviewers)
                        )
                    ),
                    pr_number=pr_number,
                    issue_context=fresh_issue,
                    metadata=context.metadata,
                    expected_plan_hash=approved_plan_context.plan_hash,
                    pr_comments=context.comments,
                    runner=runner,
                )
            else:
                verify_managed_pr_plan_binding(
                    runner,
                    config=config,
                    pr_number=pr_number,
                    issue_number=fresh_issue.number,
                    live_head=context.metadata.head_sha,
                    approved_plan_hash=approved_plan_context.plan_hash,
                    retired_plan_hashes=managed_retired_plan_hashes,
                )
        elif (
            fresh_handoff is None
            or fresh_handoff.pr_number != pr_number
            or fresh_handoff.flow != "approved-plan-implementation"
        ):
            raise AgentLoopError(
                "Approved-plan/handoff identity changed or disappeared during PR qualification; "
                "stale approvals cannot be used for this head."
            )
        if fresh_handoff is not None and fresh_handoff.plan_hash != approved_plan_context.plan_hash:
            if not allow_plan_handoff_change:
                raise AgentLoopError(
                    "Approved-plan/handoff identity changed or disappeared during PR qualification; "
                    "stale approvals cannot be used for this head."
                )
            replacement_plan: ApprovedPlanContext | None = None
            plan_candidates = [fresh_issue.comments]
            if fresh_parent is not None:
                plan_candidates.append(fresh_parent.comments)
            for comments in plan_candidates:
                candidate = recover_approved_plan_context(
                    comments,
                    expected_hash=fresh_handoff.plan_hash,
                )
                if candidate.is_available:
                    replacement_plan = candidate
                    break
            if replacement_plan is None:
                raise AgentLoopError(
                    "Approved-plan/handoff identity changed during PR qualification, but the "
                    "replacement approved plan could not be recovered; stale approvals cannot "
                    "be used for this head."
                )
            if planning_child_binding is not None:
                # Mid-run adoption (#936): a planning child's replacement
                # plan passes the same rebind verifier and admissibility rule
                # as the entry paths before it can reach a final sweep, merge,
                # or managed-CI gate.  Non-child PRs keep today's behavior.
                verified_replacement = verify_child_plan_rebind(
                    fresh_issue.comments,
                    repo=config.repo,
                    parent_plan_context=planning_child_binding.parent_plan_context,
                    child_issue=planning_child_binding.child_issue,
                    parent_issue=planning_child_binding.parent_issue,
                    stage_id=planning_child_binding.stage_id,
                    pr_number=pr_number,
                )
                if (
                    verified_replacement is None
                    or verified_replacement.plan_hash != fresh_handoff.plan_hash
                ):
                    raise AgentLoopError(
                        f"Human repair required: the approved-plan handoff for child issue "
                        f"#{planning_child_binding.child_issue} changed to plan "
                        f"{fresh_handoff.plan_hash} while PR #{pr_number} was under review, but "
                        "it is not a verified same-PR plan replacement with a rebind audit "
                        "record; no final sweep, merge, or managed-CI gate ran."
                    )
            fresh_approved_plan_context = replacement_plan
        # The handoff hash alone is not enough: recover the canonical plan
        # again from the freshly fetched issue/parent comments and require the
        # same hash and subject that the reviewers were bound to.
        plan_candidates = [fresh_issue.comments]
        if fresh_parent is not None:
            plan_candidates.append(fresh_parent.comments)
        recovered_plan = next(
            (
                candidate
                for comments in plan_candidates
                if (
                    candidate := recover_approved_plan_context(
                        comments,
                        expected_hash=fresh_approved_plan_context.plan_hash,
                        expected_subject=(
                            fresh_approved_plan_context.plan_subject
                            if fresh_approved_plan_context.plan_hash == approved_plan_context.plan_hash
                            else None
                        ),
                    )
                ).is_available
            ),
            None,
        )
        if (
            recovered_plan is None
            or recovered_plan.plan_hash != fresh_approved_plan_context.plan_hash
            or (
                fresh_approved_plan_context.plan_hash == approved_plan_context.plan_hash
                and recovered_plan.plan_subject != approved_plan_context.plan_subject
            )
        ):
            raise AgentLoopError(
                "Approved plan identity changed or disappeared during PR qualification; "
                "stale approvals cannot be used for this head."
            )
        if planning_child_binding is not None:
            # A signed supersession posted while reviewers ran must not be
            # bypassed by qualifying the plan it authorizes replacing (#985).
            _reject_pending_child_plan_supersession(
                fresh_issue.comments,
                binding=planning_child_binding,
                plan_hash=fresh_approved_plan_context.plan_hash,
                pr_number=pr_number,
                stopped="no final sweep, merge, or managed-CI gate ran",
            )
    if (
        staged_owner
        and fresh_approved_plan_context is not None
        and fresh_approved_plan_context.matrix_available
    ):
        fresh_approved_plan_context = scope_approved_plan_matrix(
            fresh_approved_plan_context,
            execution_owner=staged_owner,
        )
    requirements = _build_requirements_context(
        target_issue_context=fresh_issue,
        pr_context=context,
        parent_issue_context=fresh_parent,
    )
    result = (
        requirement.requirement_id
        for requirement in requirements.effective_requirements
    )
    requirement_ids = tuple(result)
    from .converted_capture_state import evidence_review_boundary
    records, complete = admitted_capture_history(
        runner, config, pr_number=pr_number, comments=context.comments,
    )
    boundary = evidence_review_boundary(
        records, head=context.metadata.head_sha,
        matrix_identity=(fresh_approved_plan_context.risk_test_matrix_identity
                         if fresh_approved_plan_context is not None else None),
    )
    if not complete and evidence_review_boundary(
        _extract_round_metadata_records(context.comments, flow="pr"), head=context.metadata.head_sha,
    ) >= 0:
        raise AgentLoopError("Converted capture history is incomplete; semantic approval cannot be qualified.")
    if boundary >= 0:
        names = {agent_display_name(agent) for agent in config.reviewer}
        latest = {}
        for record in records:
            if record.metadata.role == "reviewer" and record.metadata.subject == context.metadata.head_sha:
                latest[record.metadata.agent] = record
        if not complete or any(name not in latest or latest[name].index <= boundary
                               or latest[name].metadata.state != "approved" for name in names):
            context = dataclasses_replace(context, converted_capture_review_pending=True)
    return context, requirement_ids, fresh_approved_plan_context, config


def _preserve_issue_created_managed_suppression(
    contract: ManagedCiContract | None,
    *,
    active_exception: BaseException | None,
) -> bool:
    """Keep tool-created PRs suppressed when orchestration is interrupted."""

    return bool(
        active_exception is not None
        and contract is not None
        and (
            contract.issue_created_pr
            or contract.origin in {"issue-created", "source-managed"}
        )
    )


def _recover_managed_ci_approved_plan(
    child_comments: Sequence[object],
    *,
    expected_hash: str,
    parent_comments: Sequence[object] | None = None,
) -> ApprovedPlanContext:
    """Recover the canonical approved plan for a managed-CI resume.

    A staged decomposition child carries only its issue-to-PR handoff record;
    the approved plan round lives on the authoritative parent issue named by
    the child's authenticated fresh-phase identity.  The fallback predicate is
    exactly what the recovery model reports: the child recovery is unavailable
    *and* ``has_matching_candidate`` is False, i.e. no candidate survived.
    Only then are the in-process parent comments consulted, as the ordinary
    PR-mode recovery path does.

    That predicate is deliberately broader than "the child carries no record
    with ``expected_hash``".  No ``expected_subject`` is passed here, but
    ``recover_approved_plan_context`` still derives a required subject for a
    legacy free-form record from its own metadata, so such a record can carry
    the handoff hash, be rejected on that derived subject, and still leave
    ``has_matching_candidate`` False.  The fallback is permitted in that case:
    the subject-rejected record is never adopted, and the canonical handoff
    plan hash remains the sole binding on whatever the parent yields.

    Only divergent accepted candidates — several records matching the hash
    that disagree on the plan text — set ``has_matching_candidate`` True, and
    those keep failing closed: the parent is never consulted.  The parent
    identity is only ever the in-process context supplied by the staged-child
    dispatch or, in PR mode, the parent authenticated from the child's own
    fresh-phase identity (``_derive_managed_ci_parent_context``, #1662);
    nothing is inferred from PR body text or from the handoff record's own
    fields.
    """

    candidate = recover_approved_plan_context(
        child_comments,
        expected_hash=expected_hash,
    )
    if candidate.is_available or candidate.has_matching_candidate:
        return candidate
    if parent_comments is None:
        return candidate
    parent_candidate = recover_approved_plan_context(
        parent_comments,
        expected_hash=expected_hash,
    )
    if parent_candidate.is_available:
        return parent_candidate
    return candidate


# ---------------------------------------------------------------------------
# PR fix-loop step-back (#1251, stage 2): runner-facing glue around the pure
# helpers in ``review_step_back``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrStepBackDecision:
    """Clusters that open a step-back turn at this coder dispatch."""

    triggers: tuple[_step_back.PrClusterTrigger, ...] = ()
    # A non-convergence stop held back so the evidence-stall stop can decide
    # first at this dispatch (#1372); raised before any coder invocation.
    deferred_stop: str | None = None

    def coder_guidance(self) -> str:
        if not self.triggers:
            return ""
        return _step_back.render_pr_step_back_coder_guidance(self.triggers)

    def entry_payloads(self) -> tuple[Mapping[str, object], ...]:
        return tuple(_step_back.pr_step_back_entry_payload(t) for t in self.triggers)


def _pr_step_back_tracked_reviewers(
    config: AgentLoopConfig, configured_reviewers: Sequence[AgentName]
) -> tuple[str, ...]:
    """The reviewers tracked independently, in configured-reviewer order."""
    if config.pr_review_policy == "primary-then-panel" and config.primary_reviewer is not None:
        return (agent_display_name(config.primary_reviewer),)
    return tuple(agent_display_name(reviewer) for reviewer in configured_reviewers)


def _git_anchor_mapper(
    runner: Runner, config: AgentLoopConfig, *, checkout: Path, window: int
) -> _step_back.AnchorMapper:
    """Map anchors between heads through ``git diff`` hunks.

    Rename pairs come from ``--name-status -M`` with no pathspec so both sides
    are visible; hunks then come from ``-U0`` restricted to the old and new
    paths.  A failed diff is UNMAPPABLE, never an error.
    """
    name_status_cache: dict[tuple[str, str], str | None] = {}
    results: dict[tuple[str, str, str, int, int], _step_back.AnchorMapping] = {}

    def run_git(args: Sequence[str]) -> str | None:
        try:
            result = runner.run(["git", *args], cwd=checkout, check=False)
        except (OSError, AttributeError, TypeError, AgentLoopError):
            return None
        return result.stdout if result.returncode == 0 else None

    def mapper(
        from_head: str, to_head: str, path: str, start: int, end: int
    ) -> _step_back.AnchorMapping:
        key = (from_head, to_head, path, start, end)
        if key in results:
            return results[key]
        pair = (from_head, to_head)
        if pair not in name_status_cache:
            name_status_cache[pair] = run_git(
                ["-c", "core.quotePath=false", "diff", "--name-status", "-z", "-M", from_head, to_head]
            )
        name_status = name_status_cache[pair]
        diff_text: str | None = None
        if name_status is not None:
            renames, _deleted = _step_back.parse_name_status(name_status)
            new_path = renames.get(path, path)
            pathspec = list(dict.fromkeys((path, new_path)))
            diff_text = run_git(
                ["-c", "core.quotePath=false", "diff", "-M", "-U0", from_head, to_head, "--", *pathspec]
            )
        mapping = _step_back.map_anchor(
            path, start, end, window, name_status=name_status, diff_text=diff_text
        )
        if not mapping.mappable:
            log(
                config,
                f"PR step-back: anchor {path}:{start}-{end} is UNMAPPABLE between "
                f"{from_head[:12]} and {to_head[:12]} ({mapping.reason}); membership falls "
                "back to any finding on the same path",
            )
        results[key] = mapping
        return mapping

    return mapper


def _pr_step_back_records_may_matter(
    records: Sequence[PostedRoundRecord],
    tracked: Sequence[str],
    *,
    k: int,
    round_number: int,
    non_convergence_k: int = 0,
) -> bool:
    """Cheap pre-check on the round-start snapshot: could this dispatch step back?

    The snapshot lacks at most the current round's review, so a trigger needs the
    previous K-1 reviews to already be consecutive new findings, and an episode
    needs a recorded entry.  A non-convergence window needs the previous
    ``non_convergence_k`` reviews to be contiguous blocks; its coder joins are
    checked only on the fresh read, because a live snapshot predates the newest
    coder record.  Anything else skips the fresh read entirely.
    """
    for reviewer in tracked:
        if _step_back.pr_non_convergence_may_matter(
            records, reviewer, k=non_convergence_k, round_number=round_number
        ):
            return True
        if k <= 0:
            continue
        entries, degraded = _step_back.pr_step_back_history(records, reviewer)
        if degraded or entries:
            return True
        if k <= 1:
            return True
        # The snapshot may or may not already hold the current round's review (a
        # resume restores it), so the preceding K-1 reviews are taken before it.
        recent = [
            review
            for review in _step_back.pr_reviews_for(records, reviewer)
            if review.round_number < round_number
        ][-(k - 1):]
        if (
            len(recent) == k - 1
            and recent[-1].round_number == round_number - 1
            and all(r.classification == _step_back.CLASS_NEW_FINDING for r in recent)
        ):
            return True
    return False


_SHORTSTAT_RE = re.compile(r"^\s*\d+ files? changed")


def _pr_non_convergence_measurements(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    first_head: str,
    current_head: str,
    base_branch: str | None,
) -> tuple[str | None, str | None]:
    """``(diff growth since first_head, total PR size)``; ``None`` when unavailable.

    Each Git call is wrapped like the anchor mapper's: a non-zero exit,
    unparseable output or a runner error makes that measurement unavailable and
    never raises.
    """
    checkout = active_workdir(config)

    def run_git(args: Sequence[str]) -> str | None:
        try:
            result = runner.run(["git", *args], cwd=checkout, check=False)
        except (OSError, AttributeError, TypeError, AgentLoopError):
            return None
        if getattr(result, "returncode", 1) != 0 or not isinstance(result.stdout, str):
            return None
        return result.stdout

    def shortstat(from_rev: str) -> str | None:
        output = run_git(["diff", "--shortstat", from_rev, current_head])
        if output is None:
            return None
        text = output.strip()
        if not text:
            return "no change"
        return text if _SHORTSTAT_RE.match(text) else None

    growth = shortstat(first_head)
    size = None
    if base_branch:
        for base_ref in (f"origin/{base_branch}", base_branch):
            merge_base = run_git(["merge-base", base_ref, current_head])
            merge_base = (merge_base or "").strip()
            if merge_base and _is_followup_dispatch_head(merge_base):
                size = shortstat(merge_base)
                break
    return growth, size


def _pr_step_back_decision(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    round_number: int,
    snapshot_comments: Sequence[object],
    tracked: Sequence[str],
    head_sha: str | None,
    base_branch: str | None = None,
    approved_plan_context: ApprovedPlanContext | None = None,
) -> PrStepBackDecision:
    """Escalate on a clustered sibling or non-convergence, else name the clusters.

    Evaluated at the coder dispatch from freshly read history so a live run and
    a resumed run agree.  Raises ``HumanDecisionRequiredError`` (exit 4) before
    any coder invocation for either stop: when a reviewer's step-back still
    produced a clustered sibling (#1408), and when a tracked reviewer kept the
    same items blocking across ``--pr-non-convergence-rounds`` code-changing
    coder turns (#1372).  Each gate keeps its own K, so disabling one never
    disables the other.
    """
    k = config.pr_step_back_rounds
    non_convergence_k = config.pr_non_convergence_rounds
    window = config.pr_step_back_line_window
    if (k <= 0 and non_convergence_k <= 0) or not tracked or not head_sha:
        return PrStepBackDecision()
    try:
        snapshot = _extract_round_metadata_records(snapshot_comments, flow="pr")
    except AgentLoopError:
        log(config, "PR step-back suppressed: round history could not be decoded")
        return PrStepBackDecision()
    if not _pr_step_back_records_may_matter(
        snapshot, tracked, k=k, round_number=round_number,
        non_convergence_k=non_convergence_k,
    ):
        return PrStepBackDecision()
    try:
        fresh = _extract_round_metadata_records(
            get_pr_review_context(runner, config=config, pr_number=pr_number).comments,
            flow="pr",
        )
    except AgentLoopError as exc:
        log(config, f"PR step-back suppressed: fresh round history is unavailable ({exc})")
        return PrStepBackDecision()
    mapper = _git_anchor_mapper(runner, config, checkout=active_workdir(config), window=window)
    # (1) A clustered sibling after a step-back escalates first.
    trigger_candidates: list[str] = []
    for reviewer in tracked if k > 0 else ():
        _entries, degraded = _step_back.pr_step_back_history(fresh, reviewer)
        if degraded:
            log(config, f"PR step-back suppressed for {reviewer}: malformed step-back history")
            continue
        episode = _step_back.derive_pr_episode(
            fresh, reviewer, window=window, mapper=mapper,
            current_round=round_number, current_head=head_sha,
        )
        if episode.siblings and episode.entry is not None:
            mapping = episode.mapping
            if mapping is not None and not mapping.mappable:
                log(
                    config,
                    f"PR step-back: escalating {reviewer} on the unmappable-anchor same-path "
                    "fallback",
                )
            raise HumanDecisionRequiredError(
                f"PR #{pr_number}: "
                + _step_back.render_pr_step_back_human_decision(
                    reviewer=reviewer,
                    entry=episode.entry,
                    mapping=mapping,
                    siblings=episode.siblings,
                    window=window,
                    generalization=_step_back.step_back_generalization(fresh, episode.entry),
                    sibling_round=episode.sibling_round,
                )
            )
        if episode.entry is None:
            trigger_candidates.append(reviewer)
    # (2) Non-convergence wins over a fresh clustered step-back turn.  When every
    # persisting item is evidence the stall could own, the stop is deferred so the
    # evidence-stall stop decides first at this dispatch.
    unsatisfied_rows = (
        _non_convergence_evidence_owned_rows(fresh, head_sha, approved_plan_context)
        if non_convergence_k > 0 and config.pr_evidence_stall_rounds > 0
        else None
    )
    deferred: str | None = None
    for reviewer in tracked if non_convergence_k > 0 else ():
        stalled = _step_back.find_pr_non_convergence(
            fresh, reviewer, k=non_convergence_k, current_round=round_number,
            current_head=head_sha,
        )
        if stalled is None:
            continue
        if deferred is not None and _step_back.defers_to_evidence_stall(
            stalled, unsatisfied_rows
        ):
            continue
        log(
            config,
            f"Round {round_number}: {reviewer} kept "
            f"{', '.join(item.item_id for item in stalled.items)} blocking across "
            f"{non_convergence_k} code-changing coder turns; stopping for a human decision",
        )
        diff_growth, pr_size = _pr_non_convergence_measurements(
            runner, config, first_head=stalled.first_head, current_head=stalled.current_head,
            base_branch=base_branch,
        )
        message = _step_back.render_pr_non_convergence_decision(
            pr_number, stalled, diff_growth=diff_growth, pr_size=pr_size
        )
        if _step_back.defers_to_evidence_stall(stalled, unsatisfied_rows):
            log(
                config,
                f"Round {round_number}: {reviewer}'s persisting items are all evidence the "
                "evidence-stall stop may own; it decides first at this dispatch",
            )
            deferred = message
            continue
        raise HumanDecisionRequiredError(message)
    if deferred is not None:
        return PrStepBackDecision(deferred_stop=deferred)
    # (3) The clustered trigger opens a step-back turn.
    triggers: list[_step_back.PrClusterTrigger] = []
    for reviewer in trigger_candidates:
        trigger = _step_back.find_pr_cluster_trigger(
            fresh, reviewer, k=k, window=window, current_round=round_number, mapper=mapper,
            current_head=head_sha,
        )
        if trigger is not None and not _is_followup_dispatch_head(trigger.trigger_head):
            log(
                config,
                f"PR step-back suppressed for {reviewer}: head {trigger.trigger_head!r} is not "
                "a Git commit SHA, so it cannot be recorded",
            )
        elif trigger is not None:
            triggers.append(trigger)
            log(
                config,
                f"Round {round_number}: {reviewer} blocked {k} consecutive rounds with new "
                f"findings clustered in {trigger.cluster.path}:{trigger.cluster.start}-"
                f"{trigger.cluster.end}; the coder follow-up is a step-back turn",
            )
    return PrStepBackDecision(tuple(triggers))


def _pr_step_back_sweep_contexts(
    config: AgentLoopConfig,
    comments: Sequence[object],
    *,
    round_number: int,
    head_sha: str | None,
    carried: Mapping[str, "_step_back.PrStepBackEntry"] | None = None,
) -> dict[str, str]:
    """Sweep prompt text per entry reviewer, for the step-back head's review only."""
    if config.pr_step_back_rounds <= 0 or not head_sha or round_number < 2:
        return {}
    try:
        records = _extract_round_metadata_records(comments, flow="pr")
    except AgentLoopError:
        records = ()
    bound = _step_back.pr_sweep_entries(records, round_number=round_number, head_sha=head_sha)
    for reviewer, entry in (carried or {}).items():
        if entry.coder_round == round_number and entry.resulting_head == head_sha:
            bound.setdefault(reviewer, entry)
    return {
        reviewer: _step_back.render_pr_sweep_guidance(entry) for reviewer, entry in bound.items()
    }


def _pr_step_back_approach_alternative_stop(
    config: AgentLoopConfig,
    *,
    pr_number: int,
    records: Sequence[PostedRoundRecord],
    head_sha: str | None,
) -> None:
    """Stop for a human decision when a step-back turn proposed an approach change.

    Shared by the live call (the just-posted coder record) and the round-start
    resume call (durable history), so both agree (#1628).  Disabled with
    ``--pr-step-back-rounds 0``; anything unreadable keeps today's behavior.
    """
    if config.pr_step_back_rounds <= 0:
        return
    alternative = _step_back.pr_step_back_approach_alternative(records, head_sha=head_sha)
    if alternative is None:
        return
    log(
        config,
        f"PR step-back: the coder proposed an approach alternative at round "
        f"{alternative.coder_round} (head {alternative.head}); stopping for a human decision",
    )
    raise HumanDecisionRequiredError(
        _step_back.render_pr_step_back_approach_alternative_decision(pr_number, alternative)
    )


def _pr_step_back_carried_entries(
    entries: Sequence[Mapping[str, object]], *, coder_round: int, head_sha: str | None
) -> dict[str, "_step_back.PrStepBackEntry"]:
    """The entries just recorded, kept in memory for the very next review.

    The next round's comment snapshot is read before the coder record is posted,
    so a live run carries the entries across the iteration; a resumed run
    rebuilds the same bindings from the durable record.
    """
    if not head_sha:
        return {}
    carried: dict[str, _step_back.PrStepBackEntry] = {}
    for entry in entries:
        anchor = entry["anchor"]
        reviewer = str(entry["reviewer"])
        carried[reviewer] = _step_back.PrStepBackEntry(
            reviewer=reviewer,
            trigger_round=int(entry["trigger_round"]),
            trigger_head=str(entry["trigger_head"]),
            path=str(anchor["path"]),
            start=int(anchor["start"]),
            end=int(anchor["end"]),
            coder_round=coder_round,
            resulting_head=str(head_sha),
            record_index=-1,
        )
    return carried


_EVIDENCE_STALL_EXPLANATION_LIMIT = 240


def _evidence_stall_check_board(pr_checks: PullRequestChecks | None) -> _evidence_stall.CheckBoardSummary | None:
    """Summarize the dispatch-path check snapshot; ``None`` when none was fetched."""
    if pr_checks is None:
        return None
    return _evidence_stall.CheckBoardSummary(
        failing=tuple(sorted(f"{check.kind}:{check.name}" for check in pr_checks.failing)),
        infrastructure_stalls=tuple(
            sorted(f"{check.kind}:{check.name}" for check in pr_checks.infrastructure_stalls)
        ),
    )


def render_evidence_stall_decision(
    pr_number: int,
    head_sha: str,
    matrix_identity: str,
    rows: Mapping[str, str],
    k: int,
    evidence: object = None,
) -> str:
    """Human-decision text for an evidence-only stall (#1324).

    Names the PR, the exact head, the approved matrix identity prefix, K, and
    each unsatisfied row with its orchestrator-derived explanation.
    ``evidence`` (the bound coder record's bounded local test evidence) adds
    the run-tests tool identity clause when a receipt was flagged (#1632).
    """
    def clean(text: str) -> str:
        text = " ".join(str(text).split()).replace("`", "'")
        return text[:_EVIDENCE_STALL_EXPLANATION_LIMIT]

    row_lines = "\n".join(
        f"- `{clean(row_id)}`: {clean(explanation)}" for row_id, explanation in sorted(rows.items())
    )
    return (
        f"PR #{pr_number} is stalled on evidence only at head `{head_sha}` "
        f"(approved matrix `{matrix_identity[:12]}`): for {k} consecutive reviews every open "
        "blocker has asked only for admissible citation evidence, no code finding, failing "
        "check or other machine obligation was open, and the unsatisfied risk-matrix rows did "
        "not change. Unsatisfied rows and why their capture failed:\n"
        f"{row_lines}\n"
        f"{_tool_identity_line(evidence)}"
        "Decide how to proceed:\n"
        "1. Continue so the coder retries capture: rerun with `--pr-evidence-stall-rounds 0` "
        "(or a larger K).\n"
        "2. Convert the named rows to the operator evidence freeze: rerun with "
        f"`--pr-convert-evidence-rows {','.join(clean(row_id) for row_id in sorted(rows))}` "
        "(only rows listed above are accepted); the rows are then answered by signed "
        "operator evidence at the final exact-head freeze.\n"
        "3. Override with a signed human decision through the existing signed-requirement "
        "path, then rerun the same command."
    )


def _tool_identity_line(evidence: object) -> str:
    from .local_test_evidence import tool_identity_clause

    clause = tool_identity_clause(evidence).strip() if evidence else ""
    return f"{clause}\n" if clause else ""


def _head_bound_coder_record(
    records: Sequence[PostedRoundRecord], head_sha: str
) -> PostedRoundRecord | None:
    """The newest PR coder record whose subject is ``head_sha``."""
    return next(
        (
            record
            for record in reversed(records)
            if record.metadata.flow == "pr"
            and record.metadata.role == "coder"
            and record.metadata.subject == head_sha
        ),
        None,
    )


def _head_bound_unsatisfied_rows(
    records: Sequence[PostedRoundRecord],
    head_sha: str,
    approved_plan_context: ApprovedPlanContext,
    *,
    excluded_row_ids: Iterable[str] = (),
    signed_references: Sequence[SignedEvidenceReference] = (),
) -> "_evidence_stall.UnsatisfiedRows | None":
    """Unsatisfied rows from the newest coder record bound to ``head_sha``.

    ``None`` when no coder record produced that head.  Rows already converted
    to the operator evidence freeze are excluded and admitted signed
    references for ``head_sha`` overlay the evidence (#1325).
    """
    bound = _head_bound_coder_record(records, head_sha)
    if bound is None:
        return None
    return _evidence_stall.unsatisfied_rows(
        bound.metadata.risk_test_matrix_evidence,
        bound.metadata.risk_test_matrix_diagnostics,
        approved_plan_context.risk_test_matrix_expected_row_ids or (),
        approved_plan_context.risk_test_matrix_identity,
        excluded_row_ids=excluded_row_ids,
        signed_references=signed_references,
        head_sha=head_sha,
    )


def _non_convergence_evidence_owned_rows(
    records: Sequence[PostedRoundRecord],
    head_sha: str,
    approved_plan_context: ApprovedPlanContext | None,
) -> frozenset[str] | None:
    """Unsatisfied rows at the dispatch head, read as the evidence stall reads them.

    ``None`` for absent or mismatched evidence, or no applicable matrix: the
    evidence stall can then never qualify, so non-convergence never defers to it.
    """
    if (
        approved_plan_context is None
        or not approved_plan_context.matrix_available
        or not approved_plan_context.risk_test_matrix_identity
    ):
        return None
    unsatisfied = _head_bound_unsatisfied_rows(records, head_sha, approved_plan_context)
    if unsatisfied is None or unsatisfied.reason is not None or not unsatisfied.rows:
        return None
    return frozenset(unsatisfied.rows)


def authenticated_pr_comments(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    comments: Sequence[object] | None = None,
    resume: bool = False,
) -> tuple[tuple[object, ...], tuple[str, int] | None]:
    """Hydrated comments and the actor; see :func:`_authenticated_pr_comments_status`."""
    hydrated, actor, _complete = _authenticated_pr_comments_status(
        runner, config, pr_number=pr_number, comments=comments, resume=resume
    )
    return hydrated, actor


def _authenticated_pr_comments_status(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    comments: Sequence[object] | None = None,
    resume: bool = False,
) -> tuple[tuple[object, ...], tuple[str, int] | None, bool]:
    """PR comments with REST identities whenever evidence authority is present (#1325).

    The one accessor every consumer of conversion or signed-evidence authority
    uses instead of the raw GraphQL projection.  When any record carries such
    authority, identities are merged from the REST history and the
    authenticated actor is resolved (and cached on the runner).  On
    ``resume`` an incomplete REST read raises before any agent runs; in a
    mid-run reread an uncovered record simply stays unadmitted.  In dry-run
    nothing is fetched and history grants no authority.

    The third value says whether every authority-carrying record was read
    with a covered identity and a resolved actor.  ``False`` means history is
    *unavailable*, not rejected: callers must keep any established evidence
    barrier rather than treat the missing authority as absent (#1325).
    """
    if comments is None:
        comments = get_pr_review_context(runner, config=config, pr_number=pr_number).comments
    comments = tuple(comments)
    try:
        records = _extract_round_metadata_records(comments, flow="pr")
    except AgentLoopError:
        if resume:
            raise
        return comments, None, False
    authority_indexes = [
        record.index for record in records if record_carries_evidence_authority(record.metadata)
    ]
    if not authority_indexes:
        return comments, _cached_trusted_actor(runner), True
    if config.dry_run:
        return comments, None, False
    try:
        merged = tuple(
            merge_pr_comment_transport_identity(
                runner, config=config, pr_number=pr_number, comments=comments  # type: ignore[arg-type]
            )
        )
        actor = resolve_authenticated_github_actor(runner, config=config)
    except AgentLoopError as exc:
        if resume:
            raise
        log(config, f"Evidence-conversion authority not admitted: identity read failed ({exc})")
        return comments, None, False
    missing = [
        index for index in authority_indexes
        if index < len(merged) and getattr(merged[index], "author_id", None) is None
    ]
    if missing and resume:
        raise AgentLoopError(
            "PR evidence-conversion records cannot be authenticated: the REST comment read did "
            f"not cover record(s) at comment index {', '.join(map(str, missing))}; rerun once "
            "the complete comment history is readable. No agent was run."
        )
    if missing:
        log(
            config,
            "Evidence-conversion records at comment index "
            + ", ".join(map(str, missing))
            + " were not covered by the REST identity read and are not admitted",
        )
    return merged, actor, not missing


def admitted_capture_history(runner, config, *, pr_number, comments):
    """Admit the entire capture chain, including freezes without references.

    Conversion admission strips authority fields on foreign records but keeps
    ordinary content. A capture additionally needs authenticated coder and
    freeze publications, so do not reconstruct it from that ordinary content.
    """
    from .round_state import _extract_round_metadata_records, _recovery_record_admitted

    records = _extract_round_metadata_records(comments, flow="pr")
    if not any(record.metadata.evidence_conversion is not None for record in records):
        return records, True
    hydrated, actor, complete = _authenticated_pr_comments_status(
        runner, config, pr_number=pr_number, comments=comments,
    )
    records = _extract_round_metadata_records(hydrated, flow="pr")
    return tuple(record for record in records
                 if record.metadata.role == "reviewer"
                 or _recovery_record_admitted(record, hydrated, actor)), complete


def admitted_pr_records(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    comments: Sequence[object] | None = None,
    resume: bool = False,
) -> tuple[PostedRoundRecord, ...]:
    """PR round records with #1325 authority admitted only from the actor."""
    records, _complete = admitted_pr_records_status(
        runner, config, pr_number=pr_number, comments=comments, resume=resume
    )
    return records


def admitted_pr_records_status(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    comments: Sequence[object] | None = None,
    resume: bool = False,
) -> tuple[tuple[PostedRoundRecord, ...], bool]:
    """Admitted records plus whether the authority history was fully available."""
    hydrated, actor, complete = _authenticated_pr_comments_status(
        runner, config, pr_number=pr_number, comments=comments, resume=resume
    )
    return admit_evidence_authority_records(
        _extract_round_metadata_records(hydrated, flow="pr"), hydrated, actor
    ), complete


_EVIDENCE_AUTHORITY_PHASES = frozenset(
    {EVIDENCE_FREEZE_PHASE, EVIDENCE_RELEASE_PHASE, EVIDENCE_CONVERSION_PHASE}
)


def evidence_authority_from_records(
    records: Sequence[PostedRoundRecord],
) -> tuple[tuple[SignedEvidenceReference, ...], tuple[EvidenceConversionAudit, ...]]:
    """Union of admitted references and the admitted audits (#1325).

    Only orchestrator summary records of the freeze, release and conversion
    phases carry either; callers pass records already through
    :func:`admit_evidence_authority_records`.
    """
    references: dict[str, SignedEvidenceReference] = {}
    audits: list[EvidenceConversionAudit] = []
    for record in records:
        metadata = record.metadata
        if metadata.role != "summary" or metadata.phase not in _EVIDENCE_AUTHORITY_PHASES:
            continue
        for reference in metadata.signed_evidence_references:
            references.setdefault(reference.reference_id, reference)
        if metadata.phase == EVIDENCE_CONVERSION_PHASE and metadata.evidence_conversion is not None:
            audits.append(metadata.evidence_conversion)
    return tuple(references.values()), tuple(audits)


def _tool_verified_rows_at_head(
    records: Sequence[PostedRoundRecord],
    head_sha: str,
    approved_plan_context: ApprovedPlanContext,
) -> frozenset[str]:
    """Rows tool-verified by the canonical evidence of the coder record at ``head_sha``."""
    bound = _head_bound_coder_record(records, head_sha)
    if bound is None or bound.metadata.risk_test_matrix_evidence is None:
        return frozenset()
    # A converted capture's canonical evidence is authority only alongside
    # its admitted, head-bound derivation. Legacy or foreign publications keep
    # the converted barrier rather than clearing it without a review boundary.
    if (any(item.converted_row_id for item in bound.metadata.prior_items)
        and bound.metadata.risk_test_matrix_capture_classification is None):
        return frozenset()
    try:
        evidence = parse_risk_test_matrix_evidence(
            bound.metadata.risk_test_matrix_evidence,
            expected_identity=approved_plan_context.risk_test_matrix_identity,
            expected_row_ids=tuple(approved_plan_context.risk_test_matrix_expected_row_ids or ()),
        )
    except AgentLoopError:
        return frozenset()
    return frozenset(row.row_id for row in evidence.rows if row.status == "verified")


def _matrix_applies(approved_plan_context: ApprovedPlanContext | None) -> bool:
    return bool(
        approved_plan_context is not None
        and approved_plan_context.matrix_available
        and approved_plan_context.risk_test_matrix_identity
    )


def _reconcile_converted_evidence(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    ledger: Sequence[UnresolvedReviewItem],
    head_sha: str | None,
    approved_plan_context: ApprovedPlanContext | None,
    comments: Sequence[object] | None = None,
    in_process_references: Sequence[SignedEvidenceReference] = (),
    records: Sequence[PostedRoundRecord] | None = None,
    history_complete: bool = True,
    in_process_audits: Sequence[EvidenceConversionAudit] = (),
) -> list[UnresolvedReviewItem]:
    """Live-head reconciliation of row-converted obligations (#1325).

    Runs at round start after the authoritative ledger assignment and again
    inside the finalization gate.  Without any converted obligation or
    conversion record this is a no-op that performs no extra read.

    Conversions this process posted (``in_process_audits``) are trusted
    directly.  When the authority history is unavailable (a failed or
    uncovered identity read, ``history_complete`` false) nothing is dropped
    as unbacked: an established evidence barrier is kept, never removed for
    want of a reread.  Tombstones still need an admitted reference, so they
    reopen rather than satisfy, which is the fail-closed direction.
    """
    if not head_sha or not _matrix_applies(approved_plan_context):
        return list(ledger)
    assert approved_plan_context is not None
    if records is None:
        try:
            raw_records = _extract_round_metadata_records(
                comments if comments is not None else get_pr_review_context(
                    runner, config=config, pr_number=pr_number
                ).comments,
                flow="pr",
            )
        except AgentLoopError:
            raw_records = ()
        if not any(_is_row_converted_obligation(item) for item in ledger) and not any(
            record.metadata.phase == EVIDENCE_CONVERSION_PHASE for record in raw_records
        ):
            return list(ledger)
        try:
            records, history_complete = admitted_pr_records_status(
                runner, config, pr_number=pr_number, comments=comments
            )
        except AgentLoopError as exc:
            log(config, f"Converted-evidence reconciliation used no history ({exc})")
            records, history_complete = (), False
    references, audits = evidence_authority_from_records(records)
    if not history_complete:
        log(
            config,
            "Converted-evidence history is unavailable; keeping every converted obligation "
            "already in the ledger",
        )
    return _evidence_conversion.reconcile_converted_obligations(
        ledger,
        live_head=head_sha,
        matrix_identity=approved_plan_context.risk_test_matrix_identity,
        admitted_audits=(*audits, *in_process_audits),
        drop_unbacked=history_complete,
        admitted_references=_cumulative_references((*references, *in_process_references)),
        tool_verified_rows_at_live_head=_tool_verified_rows_at_head(
            records, head_sha, approved_plan_context
        ),
    )


@dataclass(frozen=True)
class EvidenceConversionOutcome:
    """The ledger after a persisted conversion (#1325)."""

    ledger: tuple[UnresolvedReviewItem, ...]
    next_item_number: int
    audit: EvidenceConversionAudit


@dataclass(frozen=True)
class EvidenceStallOutcome:
    """The dispatch-path stall classification and any automatic conversion."""

    snapshot: dict[str, object] | None
    conversion: EvidenceConversionOutcome | None = None


def _render_evidence_conversion_notice(
    *,
    pr_number: int,
    head_sha: str,
    audit: EvidenceConversionAudit,
    explanations: Mapping[str, str],
) -> str:
    source = (
        "after repeated coder attempts that could only produce capture-limited evidence"
        if audit.source == "auto"
        else "at the operator's request (`--pr-convert-evidence-rows`)"
    )
    lines = [
        "## Risk-matrix evidence converted to the operator evidence freeze",
        "",
        f"At head `{head_sha}`, {source}, these approved risk-matrix rows now need "
        "human-only exact-head evidence instead of tool-captured citations:",
        "",
    ]
    for row in audit.rows:
        detail = " ".join(str(explanations.get(row.row_id, "")).split()).replace("`", "'")[:240]
        lines.append(
            f"- `{row.row_id}` -> {row.obligation_item_id}, owned by {', '.join(row.owners)}"
            + (f"; capture status: {detail}" if detail else "")
        )
    lines.extend(
        [
            "",
            "The approved matrix is unchanged; only the source of these rows' evidence changed. "
            "The evidence is requested once every code finding and machine gate is clean, at "
            "the final barrier; a signed `-- Human Reviewer` answer for exactly the frozen head, "
            "cleared by the owning reviewer(s), verifies each row as operator-signed.",
            "",
            "-- coding-review-agent-loop",
        ]
    )
    return "\n".join(lines)


def _apply_evidence_conversion(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    round_number: int,
    head_sha: str,
    matrix_identity: str,
    ledger: Sequence[UnresolvedReviewItem],
    rows: Mapping[str, Sequence[int]],
    explanations: Mapping[str, str],
    source: str,
    next_item_number: int,
    configured_reviewers: Sequence[str],
    stall_snapshot: Mapping[str, object] | None,
    signed_evidence_references: Sequence[SignedEvidenceReference] = (),
) -> EvidenceConversionOutcome:
    """Mint one obligation per row and persist the conversion in one write.

    The converted ledger exists only in memory until the single summary
    record is posted, so a failed write leaves no conversion behind.
    """
    result = list(ledger)
    owners_by_row = {
        row_id: converted_row_owners(ledger, row_id, configured_reviewers=configured_reviewers)
        for row_id in rows
    }
    obligation_ids: dict[str, str] = {}
    for row_id in sorted(rows):
        result, obligation, consumed = _upsert_converted_evidence_obligation(
            result,
            item_number=next_item_number,
            row_id=row_id,
            matrix_identity=matrix_identity,
            head_sha=head_sha,
            owners=owners_by_row[row_id],
            source_round=round_number,
            explanation=explanations.get(row_id, ""),
        )
        if consumed:
            next_item_number += 1
        obligation_ids[row_id] = obligation.item_id
    result, converted_by_row = apply_residual_row_conversion(result, obligation_ids)
    audit = EvidenceConversionAudit(
        source=source,
        matrix_identity=matrix_identity,
        head=head_sha,
        rows=tuple(
            ConversionAuditRow(
                row_id=row_id,
                obligation_identity=converted_evidence_obligation_identity(row_id, matrix_identity),
                obligation_item_id=obligation_ids[row_id],
                owners=tuple(
                    next(
                        item.resolution_owners
                        for item in result
                        if item.item_id == obligation_ids[row_id]
                    )
                ) or tuple(owners_by_row[row_id]),
                converted_item_ids=converted_by_row.get(row_id, ()),
                attempt_record_rounds=tuple(rows[row_id]),
            )
            for row_id in sorted(rows)
        ),
    )
    post_pr_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=_attach_round_metadata(
            _render_evidence_conversion_notice(
                pr_number=pr_number, head_sha=head_sha, audit=audit, explanations=explanations
            ),
            PostedRoundMetadata(
                flow="pr",
                role="summary",
                agent="Orchestrator",
                seat_binding=reviewer_seat_binding(config),
                round_number=round_number,
                subject=head_sha,
                prior_items=tuple(result),
                state="blocking",
                phase=EVIDENCE_CONVERSION_PHASE,
                evidence_conversion=audit,
                conversion_stall_snapshot=(dict(stall_snapshot) if stall_snapshot is not None else None),
                conversion_review_round=(round_number if source == "auto" else None),
                signed_evidence_references=_cumulative_references(signed_evidence_references),
                **_architecture_metadata_fields(config),
            ),
        ),
    )
    log(
        config,
        f"Round {round_number}: PR #{pr_number} converted risk-matrix row(s) "
        f"{', '.join(audit.row_ids)} to the operator evidence freeze at {head_sha} ({source})",
    )
    return EvidenceConversionOutcome(
        ledger=tuple(result), next_item_number=next_item_number, audit=audit
    )


def _prior_stall_snapshots(
    records: Sequence[PostedRoundRecord],
) -> dict[int, Mapping[str, object]]:
    """Review-round snapshots from coder and conversion records; the latest wins."""
    prior: dict[int, Mapping[str, object]] = {}
    for record in records:
        metadata = record.metadata
        if metadata.flow != "pr":
            continue
        if metadata.role == "coder" and metadata.evidence_stall is not None:
            prior[metadata.round_number - 1] = metadata.evidence_stall
        elif (
            metadata.phase == EVIDENCE_CONVERSION_PHASE
            and metadata.conversion_stall_snapshot is not None
        ):
            prior[metadata.round_number] = metadata.conversion_stall_snapshot
    return prior


def _capture_classifications_by_round(
    records: Sequence[PostedRoundRecord],
) -> dict[int, Mapping[str, object]]:
    """Admitted per-attempt classifications keyed by coder record round."""
    result: dict[int, Mapping[str, object]] = {}
    for record in records:
        metadata = record.metadata
        if (
            metadata.flow == "pr"
            and metadata.role == "coder"
            and metadata.risk_test_matrix_capture_classification is not None
        ):
            result[metadata.round_number] = metadata.risk_test_matrix_capture_classification
    return result


def _pr_evidence_stall_decision(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    round_number: int,
    head_sha: str | None,
    approved_plan_context: ApprovedPlanContext | None,
    open_items: Sequence[UnresolvedReviewItem],
    pr_checks: PullRequestChecks | None,
    configured_reviewers: Sequence[str] = (),
    next_item_number: int | None = None,
    signed_evidence_references: Sequence[SignedEvidenceReference] = (),
) -> EvidenceStallOutcome | None:
    """Classify this review round, convert eligible rows, then apply the stall stop.

    Evaluated on the ordinary dispatch path after the check-board snapshot and
    CI-failure reconciliation, before any coder invocation.  State comes only
    from orchestrator-owned records, with #1325 authority admitted from the
    authenticated actor alone.  Order (#1325): the current snapshot (with its
    bound attempt) is computed first; automatic conversion is evaluated and
    persisted next; only when nothing converted is the stall window evaluated,
    so an eligible conversion is never pre-empted by the stall stop.  Returns
    ``None`` when no applicable approved matrix exists or history cannot be
    read.  Raises ``HumanDecisionRequiredError`` (exit 4) when K consecutive
    qualifying rounds share one unsatisfied-row set.
    """
    if not _matrix_applies(approved_plan_context) or not head_sha:
        return None
    assert approved_plan_context is not None
    identity = approved_plan_context.risk_test_matrix_identity
    try:
        records = admitted_pr_records(runner, config, pr_number=pr_number)
    except AgentLoopError as exc:
        log(config, f"PR evidence stall suppressed: round history is unavailable ({exc})")
        return None
    history_references, _audits = evidence_authority_from_records(records)
    references = _cumulative_references((*history_references, *signed_evidence_references))
    already = _evidence_conversion.converted_rows(open_items, matrix_identity=identity)
    bound = _head_bound_coder_record(records, head_sha)
    unsatisfied = _head_bound_unsatisfied_rows(
        records,
        head_sha,
        approved_plan_context,
        excluded_row_ids=already,
        signed_references=references,
    )
    snapshot = _evidence_stall.classify_round(
        review_round=round_number,
        review_head=head_sha,
        approved_identity=identity,
        open_items=open_items,
        unsatisfied=unsatisfied,
        checks=_evidence_stall_check_board(pr_checks),
        freeze_active=any(
            item.is_machine_obligation
            and item.obligation_kind == "human-exact-head-evidence"
            and item.lifecycle == "evidence_frozen"
            for item in open_items
        ),
        bound_attempt_round=(bound.metadata.round_number if bound is not None else None),
    )
    prior = _prior_stall_snapshots(records)
    if (
        next_item_number is not None
        and config.pr_evidence_capture_attempts > 0
        and unsatisfied is not None
        and unsatisfied.reason is None
    ):
        rows = _evidence_conversion.auto_conversion_rows(
            snapshot,
            prior,
            _capture_classifications_by_round(records),
            config.pr_evidence_capture_attempts,
            identity,
            already,
        )
        if rows:
            conversion = _apply_evidence_conversion(
                runner,
                config,
                pr_number=pr_number,
                round_number=round_number,
                head_sha=head_sha,
                matrix_identity=identity,
                ledger=open_items,
                rows=rows,
                explanations=unsatisfied.rows,
                source="auto",
                next_item_number=next_item_number,
                configured_reviewers=configured_reviewers,
                stall_snapshot=snapshot.to_payload(),
                signed_evidence_references=references,
            )
            return EvidenceStallOutcome(snapshot.to_payload(), conversion)
    window = _evidence_stall.stall_window(snapshot, prior, config.pr_evidence_stall_rounds)
    if window.stop and unsatisfied is not None:
        log(
            config,
            f"Round {round_number}: PR #{pr_number} evidence-only stall for "
            f"{window.length} consecutive reviews; stopping for a human decision",
        )
        raise HumanDecisionRequiredError(
            render_evidence_stall_decision(
                pr_number, head_sha, identity, unsatisfied.rows, config.pr_evidence_stall_rounds,
                evidence=(bound.metadata.local_test_evidence if bound is not None else None),
            )
        )
    return EvidenceStallOutcome(snapshot.to_payload())


def preflight_operator_evidence_conversion(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    comments: Sequence[object],
    head_sha: str | None,
    approved_plan_context: ApprovedPlanContext | None,
) -> None:
    """Read-only ``--pr-convert-evidence-rows`` validation before any PR write (#1325).

    Runs at entry, before board-change links, plan-handoff audits, recovery
    handoffs or any other startup write, from admitted history alone: the
    unsatisfied set bound to the live head (with admitted signed references
    overlaid), minus every row an admitted conversion audit already converted,
    and refusal while the latest record at the head is an evidence freeze.
    The round-start conversion re-validates against the reconciled ledger.
    """
    requested = tuple(config.pr_convert_evidence_rows)
    if not requested:
        return
    if not _matrix_applies(approved_plan_context):
        _evidence_conversion.validate_operator_conversion_rows(
            requested, unsatisfied_row_ids=None,
            reason="the approved plan has no applicable risk-test matrix",
        )
    assert approved_plan_context is not None
    if not head_sha:
        _evidence_conversion.validate_operator_conversion_rows(
            requested, unsatisfied_row_ids=None, reason="the live PR head is unknown"
        )
    assert head_sha is not None
    records = admitted_pr_records(runner, config, pr_number=pr_number, comments=comments, resume=True)
    identity = approved_plan_context.risk_test_matrix_identity
    at_head = [record for record in records if record.metadata.subject == head_sha]
    if at_head and at_head[-1].metadata.phase == EVIDENCE_FREEZE_PHASE:
        _evidence_conversion.validate_operator_conversion_rows(
            requested, unsatisfied_row_ids=None,
            reason="an exact-head evidence freeze is active; answer or release it first",
        )
    references, audits = evidence_authority_from_records(records)
    already = {
        row_id
        for audit in audits
        if audit.matrix_identity == identity
        for row_id in audit.row_ids
    }
    unsatisfied = _head_bound_unsatisfied_rows(
        records, head_sha, approved_plan_context,
        excluded_row_ids=already, signed_references=references,
    )
    if unsatisfied is None or unsatisfied.reason is not None:
        _evidence_conversion.validate_operator_conversion_rows(
            requested,
            unsatisfied_row_ids=None,
            reason=(
                "no coder record is bound to the live head"
                if unsatisfied is None
                else f"the head-bound evidence is unusable ({unsatisfied.reason})"
            ),
        )
    assert unsatisfied is not None
    _evidence_conversion.validate_operator_conversion_rows(
        requested, unsatisfied_row_ids=unsatisfied.row_ids, already_converted=already
    )


def _pr_operator_evidence_conversion(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    round_number: int,
    head_sha: str | None,
    approved_plan_context: ApprovedPlanContext | None,
    ledger: Sequence[UnresolvedReviewItem],
    next_item_number: int,
    configured_reviewers: Sequence[str],
    signed_evidence_references: Sequence[SignedEvidenceReference] = (),
) -> EvidenceConversionOutcome:
    """Apply ``--pr-convert-evidence-rows`` before any reviewer or coder (#1325).

    The named rows must all be in the unsatisfied set derived from the coder
    record bound to the live head, minus rows already converted.  Anything
    else raises before any agent runs and before anything is posted.
    """
    requested = tuple(config.pr_convert_evidence_rows)
    if not _matrix_applies(approved_plan_context):
        _evidence_conversion.validate_operator_conversion_rows(
            requested, unsatisfied_row_ids=None,
            reason="the approved plan has no applicable risk-test matrix",
        )
    assert approved_plan_context is not None
    identity = approved_plan_context.risk_test_matrix_identity
    if any(
        _is_evidence_obligation(item) and item.lifecycle == "evidence_frozen"
        for item in ledger
    ):
        _evidence_conversion.validate_operator_conversion_rows(
            requested, unsatisfied_row_ids=None,
            reason="an exact-head evidence freeze is active; answer or release it first",
        )
    if not head_sha:
        _evidence_conversion.validate_operator_conversion_rows(
            requested, unsatisfied_row_ids=None, reason="the live PR head is unknown"
        )
    assert head_sha is not None
    records = admitted_pr_records(runner, config, pr_number=pr_number)
    history_references, _audits = evidence_authority_from_records(records)
    references = _cumulative_references((*history_references, *signed_evidence_references))
    already = _evidence_conversion.converted_rows(ledger, matrix_identity=identity)
    unsatisfied = _head_bound_unsatisfied_rows(
        records,
        head_sha,
        approved_plan_context,
        excluded_row_ids=already,
        signed_references=references,
    )
    if unsatisfied is None or unsatisfied.reason is not None:
        _evidence_conversion.validate_operator_conversion_rows(
            requested,
            unsatisfied_row_ids=None,
            reason=(
                "no coder record is bound to the live head"
                if unsatisfied is None
                else f"the head-bound evidence is unusable ({unsatisfied.reason})"
            ),
        )
    assert unsatisfied is not None
    rows = _evidence_conversion.validate_operator_conversion_rows(
        requested, unsatisfied_row_ids=unsatisfied.row_ids, already_converted=already
    )
    return _apply_evidence_conversion(
        runner,
        config,
        pr_number=pr_number,
        round_number=round_number,
        head_sha=head_sha,
        matrix_identity=identity,
        ledger=ledger,
        rows={row_id: () for row_id in rows},
        explanations=unsatisfied.rows,
        source="operator",
        next_item_number=next_item_number,
        configured_reviewers=configured_reviewers,
        stall_snapshot=None,
        signed_evidence_references=references,
    )


def mint_signed_evidence_references(
    *,
    prior_items: Sequence[UnresolvedReviewItem],
    reconciled_items: Sequence[UnresolvedReviewItem],
    clearances: Sequence[tuple[str, str]],
    dispositions_by_item: Mapping[str, Sequence[object]],
    response_head: str | None,
    surfaced_requirement_ids: Sequence[str],
    freeze_signed_requirement_ids: Sequence[str] | None,
    round_number: int,
) -> tuple[SignedEvidenceReference, ...]:
    """Mint references for row-converted obligations cleared in this pass (#1325).

    Built in the same evidence-response pass, before any later freeze moves
    the signed-input baseline.  The signed-requirement set is the one
    surfaced to this pass minus the freeze record's; with nothing newly
    signed, or no freeze baseline, no reference is minted and the row stays
    unverified.
    """
    if response_head is None or freeze_signed_requirement_ids is None:
        return ()
    signed = sorted(set(surfaced_requirement_ids) - set(freeze_signed_requirement_ids))
    if not signed:
        return ()
    cleared = {identity for identity, head in clearances if head == response_head}
    prior_by_identity = {
        item.obligation_identity: item
        for item in prior_items
        if _is_row_converted_obligation(item)
        and item.lifecycle == "evidence_frozen"
        and item.candidate_head_sha == response_head
    }
    minted: list[SignedEvidenceReference] = []
    for item in reconciled_items:
        if (
            not _is_row_converted_obligation(item)
            or item.lifecycle != "cleared"
            or item.candidate_head_sha != response_head
            or item.obligation_identity not in cleared
            or item.obligation_identity not in prior_by_identity
        ):
            continue
        clearing = sorted(
            {
                getattr(disposition, "reviewer", "")
                for disposition in dispositions_by_item.get(item.item_id, ())
                if getattr(disposition, "disposition", None) == "resolved"
            }
            - {""}
        )
        if not clearing:
            continue
        minted.append(
            SignedEvidenceReference.build(
                matrix_identity=str(item.converted_matrix_identity),
                row_id=str(item.converted_row_id),
                head=response_head,
                obligation_identity=str(item.obligation_identity),
                obligation_item_id=item.item_id,
                owners=item.resolution_owners,
                signed_requirement_ids=signed,
                clearing_reviewers=clearing,
                cleared_round=round_number,
            )
        )
    return tuple(minted)


def freeze_signed_baseline(
    records: Sequence[PostedRoundRecord], head_sha: str | None
) -> tuple[str, ...] | None:
    """Signed-requirement IDs recorded by the newest freeze record at ``head_sha``."""
    if not head_sha:
        return None
    for record in reversed(records):
        metadata = record.metadata
        if (
            metadata.phase == EVIDENCE_FREEZE_PHASE
            and metadata.subject == head_sha
            and metadata.evidence_freeze is not None
            and metadata.evidence_freeze.valid
        ):
            return tuple(metadata.evidence_freeze.signed_requirement_ids_at_freeze)
    return None


def _conversion_repeat_round_number(metadata: PostedRoundMetadata) -> int | None:
    """The round an admitted conversion record reopens, if it names one.

    An auto conversion persists ``conversion_review_round``.  An operator
    conversion deliberately leaves it unset; its admitted audit and its own
    ``round_number`` identify the repeat instead (#1664), so the persisted
    encoding is unchanged.
    """
    if metadata.conversion_review_round is not None:
        return metadata.conversion_review_round
    audit = metadata.evidence_conversion
    if audit is not None and audit.source == "operator":
        return metadata.round_number
    return None


def conversion_repeat_round(
    records: Sequence[PostedRoundRecord], *, round_number: int, head_sha: str | None
) -> bool:
    """Whether a resumed round is an unfinished conversion repeat (#1325).

    True when the latest admitted conversion record names this round as its
    repeat round (``conversion_review_round`` for an auto conversion, the
    record's round for an admitted operator conversion, #1664) at the live
    head and no later admitted freeze, release, or coder record exists at
    that head.
    """
    if not head_sha:
        return False
    conversion = next(
        (
            record
            for record in reversed(records)
            if record.metadata.phase == EVIDENCE_CONVERSION_PHASE
            and _conversion_repeat_round_number(record.metadata) is not None
        ),
        None,
    )
    if (
        conversion is None
        or _conversion_repeat_round_number(conversion.metadata) != round_number
        or conversion.metadata.subject != head_sha
    ):
        return False
    return not any(
        record.index > conversion.index
        and record.metadata.subject == head_sha
        and (
            record.metadata.phase in {EVIDENCE_FREEZE_PHASE, EVIDENCE_RELEASE_PHASE}
            or record.metadata.role == "coder"
        )
        for record in records
    )


def latest_conversion_index(records: Sequence[PostedRoundRecord], *, head_sha: str | None) -> int:
    """Comment index of the newest admitted conversion record at ``head_sha`` (or -1)."""
    for record in reversed(records):
        if record.metadata.phase == EVIDENCE_CONVERSION_PHASE and record.metadata.subject == head_sha:
            return record.index
    return -1


@dataclass(frozen=True)
class ConversionVisibilityBoundary:
    """A trusted conversion record bounding same-round peer visibility (#1664).

    ``index`` is valid only in the comment history it was resolved from;
    ``key`` (author, creation time, body) locates the same comment in another
    history, which must resolve its own index.
    """

    index: int
    key: tuple[str | None, str | None, str | None]


def conversion_boundary_record(
    records: Sequence[PostedRoundRecord], *, round_number: int, head_sha: str | None
) -> PostedRoundRecord | None:
    """The newest admitted conversion record of ``round_number`` at ``head_sha``.

    ``records`` must already be admitted: a foreign-authored or audit-rejected
    conversion has lost its audit and never qualifies.
    """
    if not head_sha:
        return None
    return next(
        (
            record
            for record in reversed(records)
            if record.metadata.flow == "pr"
            and record.metadata.role == "summary"
            and record.metadata.phase == EVIDENCE_CONVERSION_PHASE
            and record.metadata.round_number == round_number
            and record.metadata.subject == head_sha
            and record.metadata.evidence_conversion is not None
        ),
        None,
    )


def comment_identity_key(comment: object) -> tuple[str | None, str | None, str | None]:
    """Stable identity of a comment across GraphQL and REST histories."""
    author = getattr(comment, "author", None)
    return (
        strip_bot_login_suffix(author) if author is not None else None,
        getattr(comment, "created_at", None),
        getattr(comment, "body", None),
    )


def conversion_boundary_candidate_present(
    comments: Sequence[object], *, round_number: int, head_sha: str | None
) -> bool:
    """Whether ``comments`` hold an unadmitted conversion record of this round and head.

    Pure: no identity read.  Lets the guard tell a conversion boundary that
    could not be admitted from a round with no conversion record (#1676).
    """
    try:
        raw = _extract_round_metadata_records(tuple(comments), flow="pr")
    except AgentLoopError:
        return False
    return any(
        record.metadata.phase == EVIDENCE_CONVERSION_PHASE
        and record.metadata.round_number == round_number
        and record.metadata.subject == head_sha
        for record in raw
    )


def conversion_visibility_boundary(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    comments: Sequence[object],
    round_number: int,
    head_sha: str | None,
) -> ConversionVisibilityBoundary | None:
    """Resolve the trusted conversion boundary in exactly this comment history (#1664).

    Publications at or before the boundary belong to the pre-conversion pass,
    which the conversion repeat reads through the converted ledger, so they
    are not same-round peers.  Admission runs over ``comments`` itself; an
    undecodable history, a failed or incomplete identity read, or a
    foreign-authored or audit-rejected record yields ``None`` and the guard
    keeps its ordinary refusal.
    """
    comments = tuple(comments)
    if not conversion_boundary_candidate_present(
        comments, round_number=round_number, head_sha=head_sha
    ):
        # No candidate: nothing to admit, so no identity read is made.
        return None
    try:
        records, _complete = admitted_pr_records_status(
            runner, config, pr_number=pr_number, comments=comments
        )
    except AgentLoopError:
        return None
    record = conversion_boundary_record(records, round_number=round_number, head_sha=head_sha)
    if record is None or not 0 <= record.index < len(comments):
        return None
    return ConversionVisibilityBoundary(
        index=record.index, key=comment_identity_key(comments[record.index])
    )


def round_evidence_authority_records(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    comments: Sequence[object],
    ledger: Sequence[UnresolvedReviewItem],
    force: bool = False,
) -> tuple[tuple[PostedRoundRecord, ...], bool]:
    """Admitted records for round-start reconciliation and the review overlay.

    Also returns whether the authority history was fully available.

    Reads identities only when a converted obligation, a conversion record,
    or a signed-evidence reference exists (or ``force``); otherwise there is
    no #1325 authority to admit and nothing extra is fetched.
    """
    try:
        raw = _extract_round_metadata_records(comments, flow="pr")
    except AgentLoopError:
        return (), False
    if not (
        force
        or any(_is_row_converted_obligation(item) for item in ledger)
        or any(
            record.metadata.phase == EVIDENCE_CONVERSION_PHASE
            or record.metadata.signed_evidence_references
            for record in raw
        )
    ):
        return (), True
    try:
        return admitted_pr_records_status(runner, config, pr_number=pr_number, comments=comments)
    except AgentLoopError as exc:
        log(config, f"Evidence-conversion history not admitted this round ({exc})")
        return (), False


# Plain PR pre-panel stop diagnostic; history_export recognizes posted stops by it.
PR_SCHEDULING_DIAGNOSTIC_TEMPLATE = "PR review scheduling diagnostic ({label}): {message}"


def render_pr_scheduling_diagnostic(label: str, message: str) -> str:
    """The exact body ``stop_pre_panel`` posts."""
    return PR_SCHEDULING_DIAGNOSTIC_TEMPLATE.format(label=label, message=message)


# Public size bound for a posted PR human-decision diagnostic (#1640): well
# under the transport's plain-body limit, so carrier neutralization and host
# footers cannot push it into RoundCommentOverflowError.
PR_HUMAN_DECISION_DIAGNOSTIC_LIMIT = 40_000


def _pr_human_decision_omission_note(omitted: int) -> str:
    return (
        f"… ({omitted} characters of detail omitted from this diagnostic; the full text is "
        "in the CLI output of this run)"
    )


def render_pr_human_decision_diagnostic(label: str, message: str) -> str:
    """The PR scheduling diagnostic for a human-decision stop, bounded in size.

    Within :data:`PR_HUMAN_DECISION_DIAGNOSTIC_LIMIT` this is exactly
    :func:`render_pr_scheduling_diagnostic`.  Otherwise the template, the first
    line and as many following whole lines as fit are kept verbatim, then an
    omission note.  The raised exception and the CLI text are never bounded.
    """
    full = render_pr_scheduling_diagnostic(label, message)
    if len(full) <= PR_HUMAN_DECISION_DIAGNOSTIC_LIMIT:
        return full
    prefix = render_pr_scheduling_diagnostic(label, "")
    # The note's digit count can only shrink from this upper bound.
    reserve = len("\n" + _pr_human_decision_omission_note(len(message)))
    budget = max(PR_HUMAN_DECISION_DIAGNOSTIC_LIMIT - len(prefix) - reserve, 0)
    first, *rest = message.split("\n")
    if len(first) > budget:
        kept = first[:budget]
    else:
        kept = first
        for line in rest:
            if len(kept) + 1 + len(line) > budget:
                break
            kept += "\n" + line
    return prefix + kept + "\n" + _pr_human_decision_omission_note(len(message) - len(kept))




def _managed_v1_dispatch_records(managed_ci) -> tuple:
    """This process's own managed v1 dispatches, for the coder-turn audit (#1414)."""
    if managed_ci is None or managed_ci.protocol_version != 1:
        return ()
    return tuple(managed_ci.v1_dispatch_records)


def unchanged_head_stop_message(
    *, pr, coder_name, previous_head, turns, round_number, evidence, route,
    citation_diagnosis: FollowupCitationDiagnosis | None = None,
) -> str:
    """The unchanged-head stop text; receipts (issue #1182) go before the route.

    When the last follow-up's test_observations were dropped as a non-array
    (#1626), the message names that and the uncited receipt count, so an
    operator can see the blocker may be a tool format problem.
    """
    citation_sentence = ""
    if citation_diagnosis is not None and citation_diagnosis.non_array_drop:
        citation_sentence = (
            " The coder's test_observations were not a JSON array (format repair "
            f"attempted: {'yes' if citation_diagnosis.repair_attempted else 'no'}); "
            f"{citation_diagnosis.uncited_receipts} run-tests receipt(s) from that turn "
            "went uncited, so the blocker may be a tool format problem rather than a "
            "code defect."
        )
    return (
        f"PR #{pr}: {coder_name} left head {previous_head} unchanged in "
        f"{turns} consecutive follow-up rounds, so another "
        "review of the same diff cannot change the verdict. Stopping before round "
        f"{round_number + 1}; human review required."
        f"{unsuperseded_receipts_sentence(evidence)}"
        f"{tool_identity_clause(evidence)}"
        f"{citation_sentence}"
        f"{route}"
    )


def _reconciliation_dispositions_by_item(
    reconciliation: PostedRoundMetadata,
) -> dict[str, list[ReviewItemDisposition]]:
    """Round dispositions keyed by carried item, as the round end built them.

    A disposition naming an item the reconciliation did not carry cannot be
    replayed faithfully, so it refuses instead of silently losing ledger state.
    """
    by_item: dict[str, list[ReviewItemDisposition]] = {
        item.item_id: [] for item in reconciliation.prior_items
    }
    for disposition in reconciliation.dispositions:
        if disposition.item_id not in by_item:
            raise AgentLoopError(
                f"PR round {reconciliation.round_number} reconciliation at "
                f"{reconciliation.subject} records a disposition for {disposition.item_id}, "
                "which its carried ledger does not hold; refusing to settle the closed "
                "round. No agent was invoked."
            )
        by_item[disposition.item_id].append(disposition)
    return by_item


def _stop_for_human_decision(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int,
    exc: HumanDecisionRequiredError, round_number: int, post_comment: Callable,
) -> NoReturn:
    """Post a best-effort PR scheduling diagnostic, then re-raise ``exc`` (#1640).

    Plain audit text only, never state: resume re-derives these stops
    from round records.  A failed post never changes the stop or exit 4.
    """
    message = str(exc)
    log(config, f"Round {round_number}: {message}")
    try:
        post_comment(
            runner,
            config=config,
            pr_number=pr_number,
            body=render_pr_human_decision_diagnostic(f"round {round_number}", message),
        )
    except (AgentLoopError, OSError) as post_error:
        log(
            config,
            "Warning: could not post the human-decision diagnostic to "
            f"PR #{pr_number} ({type(post_error).__name__}: {post_error}); "
            "the stop stands.",
        )
    raise exc
