"""PR/plan panel evidence, board-amendment notes, plan primary streak and plan-growth gates.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1194); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace as dataclasses_replace
from .agents.registry import agent_display_name
from .config import AgentLoopConfig, reviewers
from .board_amendment import (
    ContractLineage,
    added_in_config,
    format_reviewer_board_amendment_comment,
    missing_from_config,
    predates_restoration,
    restoration_rounds_from_comments,
    stale_unread_amendment_locator,
)
from .decomposition import approved_plan_hash
from .errors import AgentLoopError
from .github import IssueContext, HumanReviewRequirement
from .issue_pr_handoff import IssuePrHandoffMetadata
from .logging import log
from .prompts import SupersededPrepanelReview
from .protocol import (
    ParsedReview,
    parse_plan_review,
    parse_plan_revision_patch,
    parse_structured_pr_review,
    review_freeform_summary_text,
)
from .protocol import parse_review
from .round_state import (
    PostedRoundMetadata,
    PostedRoundRecord,
    ResumedReviewRound,
    _extract_round_metadata_records,
    _plan_subject,
)
from .plan_assembly import (
    AssembledPlanSidecar,
    decode_assembled_plan_sidecar,
    hydrate_authenticated_plan_state,
)
from .plan_growth import (
    PlanGrowthApprovalVerdict,
    PlanGrowthAssessment,
    PlanGrowthThresholds,
    assess_plan_growth,
    growth_justification_violation,
    handoff_growth_verdict_violation,
    plan_growth_approval_verdict,
    plan_growth_gate_enforced,
    plan_strategy,
)
from .plan_review_scheduling import (
    PLAN_HISTORY_INTACT,
    PlanCandidateKey,
    PlanCrossCuttingContracts,
    PlanReviewSchedulingContract,
    PlanRevisionDescriptor,
    PlanSchedulerSnapshot,
    PlanSchedulingDecision,
    classify_plan_history,
    classify_plan_transition,
    execution_recommendation_contract_identity,
    execution_recommendation_contract_projection,
    make_plan_contract,
    plan_policy_capabilities,
    select_plan_reviewers,
    surfaced_requirement_id_digest,
)
from .review_scheduling import PANEL_OPENED_PHASES, ReviewObligation
from .response_validation import _surfaced_reviewer_requirement_ids


@dataclass(frozen=True)
class PrPanelEvidence:
    """Qualified panel-opening evidence for ``primary-then-panel`` (#840).

    ``opening_index`` is the comment index of the first qualified panel
    opening.  Every record after it is post-panel state.  Anything that merely
    looks like a panel before it (secondary reviews, full-board or other
    panel-phase checkpoints, unattributed/automatic force-full latches) is an
    unqualified premature-panel artifact and is never panel evidence.
    """

    opening_index: int | None = None
    opening_source: str | None = None
    unqualified_artifacts: tuple[str, ...] = ()
    prepanel_latch: bool = False
    post_opening_automatic_latch: bool = False

    @property
    def opened(self) -> bool:
        return self.opening_index is not None


def _derive_pr_panel_evidence(
    records: Sequence[PostedRoundRecord],
    *,
    primary_reviewer: str | None,
    required_reviewers: Sequence[str],
) -> PrPanelEvidence:
    """Derive the first qualified panel opening from comment-ordered history.

    A qualified opening is either an operator-sourced force-full record, or a
    ``secondary-audit`` scheduler record for subject S that lists the primary
    as approved and is preceded by the primary's own approved review of S.
    """
    secondaries = set(required_reviewers) - {primary_reviewer}
    primary_approved_subjects: set[str] = set()
    artifacts: list[str] = []
    prepanel_latch = False
    opening: PostedRoundRecord | None = None
    opening_source: str | None = None
    for record in sorted(records, key=lambda item: item.index):
        metadata = record.metadata
        valid = metadata.scheduler_metadata_status == "valid"
        if valid and metadata.scheduler_force_full and metadata.scheduler_force_full_source == "operator":
            opening, opening_source = record, "operator"
            break
        if (
            valid
            and primary_reviewer is not None
            and metadata.scheduler_phase == "secondary-audit"
            and primary_reviewer in metadata.scheduler_approved_reviewers
            and metadata.subject in primary_approved_subjects
        ):
            opening, opening_source = record, "primary-approval"
            break
        if (
            metadata.role == "reviewer"
            and metadata.agent == primary_reviewer
            and metadata.state == "approved"
            and metadata.subject
        ):
            primary_approved_subjects.add(metadata.subject)
        if metadata.role == "reviewer" and metadata.agent in secondaries:
            artifacts.append(
                f"{metadata.agent} review (round {metadata.round_number}, head {metadata.subject})"
            )
        elif valid and metadata.scheduler_phase in PANEL_OPENED_PHASES:
            artifacts.append(
                f"{metadata.scheduler_phase} scheduler record (round {metadata.round_number}, "
                f"head {metadata.subject})"
            )
        if valid and metadata.scheduler_force_full:
            prepanel_latch = True
    post_opening_latch = bool(
        opening is not None
        and any(
            record.index > opening.index
            and record.metadata.scheduler_metadata_status == "valid"
            and record.metadata.scheduler_force_full
            and record.metadata.scheduler_force_full_source != "operator"
            for record in records
        )
    )
    return PrPanelEvidence(
        opening_index=opening.index if opening is not None else None,
        opening_source=opening_source,
        unqualified_artifacts=tuple(dict.fromkeys(artifacts)),
        prepanel_latch=prepanel_latch,
        post_opening_automatic_latch=post_opening_latch,
    )


def _pr_record_is_panel_qualified(
    record: PostedRoundRecord,
    evidence: PrPanelEvidence,
    *,
    primary_reviewer: str | None,
    operator_force_full: bool,
) -> bool:
    """Return whether a reviewer record may count under the staged policy.

    Secondary records count only after the first qualified panel opening.  The
    primary is never premature under a primary-approval opening, but an
    operator-sourced opening (recorded, or being established by this round's
    operator flag) qualifies only records written after it.
    """
    if evidence.opening_index is not None and record.index > evidence.opening_index:
        return True
    if record.metadata.agent != primary_reviewer:
        return False
    if evidence.opening_source == "operator":
        return False
    return not (evidence.opening_index is None and operator_force_full)


def _describe_superseded_prepanel_review(record: PostedRoundRecord) -> str:
    metadata = record.metadata
    item_ids = ", ".join(item.item_id for item in metadata.new_items) or "no numbered items"
    return (
        f"{metadata.agent} (round {metadata.round_number}, head {metadata.subject}, "
        f"state {metadata.state or 'unknown'}; items: {item_ids})"
    )


def _superseded_prepanel_review(record: PostedRoundRecord | None) -> SupersededPrepanelReview | None:
    """Build the non-authoritative prompt context for a superseded review."""
    if record is None:
        return None
    metadata = record.metadata
    reviewer = metadata.agent or "reviewer"
    claims: list[str] = [item.text for item in metadata.new_items if item.text]
    source_text = metadata.canonical_reviewer_response or record.body
    if not claims:
        parsed: ParsedReview | None = None
        try:
            if metadata.canonical_reviewer_response is not None:
                parsed = parse_structured_pr_review(source_text, reviewer=reviewer, architecture_status_mode="legacy")
            if parsed is None:
                parsed = parse_review(source_text, reviewer=reviewer)
        except AgentLoopError:
            parsed = None
        if parsed is not None:
            claims.extend(item.text for item in parsed.blocking_items if item.text)
            claims.extend(item.text for item in parsed.followups.same_pr if item.text)
    return SupersededPrepanelReview(
        reviewer=reviewer,
        round_number=metadata.round_number,
        head_sha=metadata.subject or "(unknown)",
        state=metadata.state or "unknown",
        summary=review_freeform_summary_text(record.body),
        claims=tuple(claims),
        item_ids=tuple(item.item_id for item in metadata.new_items),
    )


def _superseded_prepanel_plan_review(
    record: PostedRoundRecord | None,
) -> SupersededPrepanelReview | None:
    """Planning counterpart of ``_superseded_prepanel_review`` (#905).

    Builds the non-authoritative prompt context for a secondary plan review
    recorded before any qualified panel opening, so the operator planning
    force-full override can replay that reviewer's own earlier claims to it
    without their ever entering the ledger, ownership, or approval accounting.
    """
    if record is None:
        return None
    metadata = record.metadata
    reviewer = metadata.agent or "reviewer"
    claims: list[str] = [item.text for item in metadata.new_items if item.text]
    if not claims:
        source_text = metadata.canonical_reviewer_response or record.body
        try:
            parsed = parse_plan_review(source_text, reviewer=reviewer, architecture_status_mode="legacy")
        except AgentLoopError:
            parsed = None
        if parsed is not None:
            claims.extend(item.text for item in parsed.items.blocking if item.text)
            claims.extend(item.text for item in parsed.items.same_plan if item.text)
    return SupersededPrepanelReview(
        reviewer=reviewer,
        round_number=metadata.round_number,
        head_sha=metadata.subject or "(unknown)",
        state=metadata.state or "unknown",
        summary=review_freeform_summary_text(record.body),
        claims=tuple(claims),
        item_ids=tuple(item.item_id for item in metadata.new_items),
    )


def _scheduler_recorded_force_full(*, operator: bool, automatic: bool) -> tuple[bool, str | None]:
    """Return the persisted force-full latch and its audit source (#840)."""
    if operator:
        return True, "operator"
    if automatic:
        return True, "automatic"
    return False, None


# Planning counterpart of ``PANEL_OPENED_PHASES``: phases that only exist after
# the secondary plan panel has opened, and which therefore look like an opening
# without being one on their own.
PLAN_PANEL_OPENED_PHASES = frozenset(
    {"secondary-audit", "remediation", "final-secondary-sweep", "full-board"}
)


@dataclass(frozen=True)
class PlanPanelEvidence:
    """Qualified planning panel-opening evidence (#905, from #841).

    ``opening_index`` is the comment index of the first qualified opening.
    Premature secondary plan reviews, ``full-board``/``remediation``
    checkpoints, and unattributed or automatic latches recorded before it are
    unqualified artifacts and are never panel evidence.
    """

    opening_index: int | None = None
    opening_source: str | None = None
    unqualified_artifacts: tuple[str, ...] = ()
    premature_secondary_reviews: tuple[str, ...] = ()
    prepanel_latch: bool = False
    post_opening_automatic_latch: bool = False

    @property
    def opened(self) -> bool:
        return self.opening_index is not None


def _plan_identity_digest(value: object) -> str | None:
    """Stable short digest of one authenticated plan identity component."""
    if value is None:
        return None
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:32]


def _plan_candidate_key_for(
    *,
    plan_subject: str,
    sidecar: AssembledPlanSidecar | None,
    surfaced_requirement_ids: Sequence[str],
) -> PlanCandidateKey:
    """Build the canonical exact-plan candidate key for one planning round.

    Every component is read from the authenticated assembled sidecar, never
    from rendered Markdown, so the key a reviewer record persists is the same
    key the scheduler, the panel evidence, and resume compare against.
    """
    payload = dict(sidecar.canonical_json) if sidecar is not None else {}
    version = payload.get("execution_strategy_contract_version")
    version = version if isinstance(version, int) and not isinstance(version, bool) else None
    execution_identity = (
        _plan_identity_digest(payload.get("execution_recommendation")) if version == 1 else None
    )
    matrix_identity = None
    if payload.get("risk_test_matrix_contract_version") == 1 and isinstance(
        payload.get("risk_test_matrix"), dict
    ):
        matrix_identity = _plan_identity_digest(
            [payload.get("risk_test_matrix"), payload.get("risk_test_matrix_changes", [])]
        )
    return PlanCandidateKey(
        subject=plan_subject,
        aggregate_plan_identity=(
            sidecar.aggregate_identity if sidecar is not None else None
        ),
        execution_strategy_identity=execution_identity,
        risk_test_matrix_identity=matrix_identity,
        surfaced_requirement_id_digest=surfaced_requirement_id_digest(
            tuple(surfaced_requirement_ids)
        ),
        execution_strategy_contract_version=version,
    )


def _outstanding_plan_phase(
    snapshot: PlanSchedulerSnapshot,
    *,
    decision,
    current_key: PlanCandidateKey | None,
    obligations: Sequence[ReviewObligation],
    qualifying_approvals: Sequence[str],
    panel_evidence: bool,
    force_full: bool,
    force_full_source: str | None,
) -> str:
    """The phase the pending reviewer-only round will run.

    The scheduler decision in hand describes the board that just ran, so it
    still reads `primary` right after the primary's approval and `remediation`
    right after an owner-scoped round.  The phase-advance record and the
    round-budget diagnostic must instead name the outstanding phase, so project
    the scheduler forward over the unchanged candidate key: the plan is
    byte-identical (a `recheck` transition), the ledger is the post-round one,
    the approvals are this round's settled set, and the round just wrote a valid
    scheduler record, which is the next round's recovery boundary and clears the
    readable degraded classes (#905, from #841).
    """
    projected = dataclasses_replace(
        snapshot,
        previous_key=current_key,
        current_key=current_key,
        obligations=tuple(obligations),
        panel_evidence=panel_evidence,
        force_full=force_full,
        force_full_source=force_full_source,
        phase=decision.phase if decision is not None else snapshot.phase,
        degraded_history_class=PLAN_HISTORY_INTACT,
        fallback_reasons=(),
        premature_secondary_reviews=(
            () if panel_evidence else snapshot.premature_secondary_reviews
        ),
    )
    try:
        return select_plan_reviewers(
            projected,
            classify_plan_transition(current_key, current_key),
            qualifying_approvals=tuple(qualifying_approvals),
            phase=projected.phase,
        ).phase
    except AgentLoopError:
        # A projection is never allowed to break the advance itself; the
        # independent panel audit is the conservative outstanding phase.
        return "secondary-audit"


def _board_amendment_template(
    *,
    flow: str,
    issue_number: int | None,
    pr_number: int | None,
    persisted: object,
    removed: Sequence[str],
    start_round_number: int | str,
    restored: Sequence[str] = (),
) -> str:
    """Filled-in signed amendment template printed by fail-closed errors (#943, #984)."""
    return format_reviewer_board_amendment_comment(
        flow=flow,
        issue=issue_number if flow == "plan" else None,
        pr_number=pr_number if flow == "pr" else None,
        original_required_reviewers=tuple(getattr(persisted, "required_reviewers")),
        policy=str(getattr(persisted, "policy")),
        primary_reviewer=getattr(persisted, "primary_reviewer"),
        removed_reviewers=tuple(removed),
        restored_reviewers=tuple(restored),
        effective_from_round=start_round_number,  # type: ignore[arg-type]
    )


_PR_AMENDMENT_RERUN_CLAUSE = (
    " Posting the record is what removes the reviewer. Rerun with the original "
    "reviewer board still configured (recommended); the reduced board is also "
    "accepted, except on an issue-created strict managed-CI PR, where the approved "
    "plan is re-verified against the supplied flags and the original board is "
    "required. Do not drop the reviewer from the command line instead of posting "
    "the record."
)


def _board_amendment_route_clause(
    *,
    flow: str,
    issue_number: int | None,
    pr_number: int | None,
    persisted: object,
    configured: object | None,
    start_round_number: Callable[[], int | None],
    amendments_recognized: bool = False,
) -> str:
    """The amendment-route clause appended to a contract-drift error (#943, #984)."""
    removed = missing_from_config(persisted, configured)
    restored = added_in_config(persisted, configured) if removed is None else None
    if removed is None and restored is None:
        return ""
    try:
        round_number: int | str | None = start_round_number()
    except Exception:  # noqa: BLE001 - the template is advisory text only
        round_number = None
    if round_number is None:
        round_number = "<N: the round this resume re-enters>"
    surface = f"issue #{issue_number}" if flow == "plan" else f"PR #{pr_number}"
    template = _board_amendment_template(
        flow=flow,
        issue_number=issue_number,
        pr_number=pr_number,
        persisted=persisted,
        removed=removed or (),
        restored=restored or (),
        start_round_number=round_number,
    )
    if restored:
        return (
            " If a previously removed reviewer backend has recovered, a human operator may "
            f"restore it with a signed reviewer-board amendment posted on {surface} (only a "
            "reviewer an earlier amendment removed can be restored); replace the rationale "
            "placeholder and keep the effective round printed here:\n\n"
            + template
        )
    if flow == "plan":
        rerun = (
            " Posting the record is what removes the reviewer; a planning rerun then uses "
            "the reduced reviewer board with the same policy and primary."
        )
    else:
        rerun = _PR_AMENDMENT_RERUN_CLAUSE
    not_recognized = (
        ""
        if amendments_recognized
        else " If you already posted a record and still see this drift, it was not "
        "recognized; check the log for an ignored-record diagnostic."
    )
    return (
        " If a reviewer backend is unavailable, a human operator may instead remove it "
        f"with a signed reviewer-board amendment posted on {surface}; replace the "
        "rationale placeholder and keep the effective round printed here."
        + rerun
        + not_recognized
        + "\n\n"
        + template
    )


def _stale_amendment_repost_clause(
    detail: str, start_round_number: Callable[[], int | None]
) -> str:
    """Delete-and-repost instruction for records posted under an unread amendment (#1133)."""
    locator = stale_unread_amendment_locator(detail)
    if locator is None:
        return ""
    try:
        round_number: int | str | None = start_round_number()
    except Exception:  # noqa: BLE001 - advisory text only
        round_number = None
    if round_number is None:
        round_number = "the round the next resume re-enters"
    return (
        f" Delete the amendment comment at {locator} and post a fresh signed record with "
        f"effective_from_round {round_number}; the fresh record is read from the start of "
        "the next run."
    )


def _append_board_amendment_note(body: str, note: str | None) -> str:
    """Insert the reduced-board note before a comment's trailing signature."""
    if not note:
        return body
    lines = body.splitlines()
    index = len(lines)
    while index > 0 and (
        not lines[index - 1].strip()
        or lines[index - 1].startswith("-- ")
        or lines[index - 1].lstrip().startswith("<!--")
    ):
        index -= 1
    return "\n".join([*lines[:index], f"- {note}", *lines[index:]])


def _keep_reused_amendment_round_reviews(
    decision: PlanSchedulingDecision,
    *,
    lineage: ContractLineage,
    round_number: int,
    current_resume: ResumedReviewRound | None,
    eligible: Callable[[str], bool],
) -> PlanSchedulingDecision:
    """Keep remaining reviewers' round-N reviews in the amended selection (#943).

    Re-entering the amendment's activation round reruns scheduler selection
    under the amended contract.  A remaining reviewer that already posted a
    usable review in that round stays selected so its review (and its item
    dispositions) is reused rather than dropped as a carried approval; it is
    never re-invoked.
    """
    active = lineage.active_amendment
    if (
        active is None
        or current_resume is None
        or round_number != active.effective_from_round
    ):
        return decision
    required = getattr(lineage.contracts[-1], "required_reviewers")
    posted = {record.metadata.agent for record in current_resume.completed_reviews}
    keep = [
        name
        for name in required
        if name in posted and name not in decision.selected_reviewers and eligible(name)
    ]
    if not keep:
        return decision
    selected = tuple(
        name for name in required if name in decision.selected_reviewers or name in keep
    )
    return dataclasses_replace(
        decision,
        selected_reviewers=selected,
        paused_reviewers=tuple(
            (name, why) for name, why in decision.paused_reviewers if name not in keep
        ),
        reason=(
            f"{decision.reason}; reviewer board amendment re-entry reuses the round "
            f"{round_number} review(s) already posted by {', '.join(keep)}"
        ),
    )


def _plan_contract_or_none(metadata: PostedRoundMetadata) -> PlanReviewSchedulingContract | None:
    try:
        return _plan_scheduler_contract_from_metadata(metadata)
    except AgentLoopError:
        return None


def _plan_scheduler_contract_from_metadata(
    metadata: PostedRoundMetadata,
) -> PlanReviewSchedulingContract | None:
    if metadata.scheduler_contract is None:
        return None
    return PlanReviewSchedulingContract.from_mapping(metadata.scheduler_contract)


def _plan_key_from_payload(payload: object) -> PlanCandidateKey | None:
    if not isinstance(payload, dict):
        return None
    try:
        return PlanCandidateKey.from_mapping(payload)
    except AgentLoopError:
        return None


def _derive_plan_panel_evidence(
    records: Sequence[PostedRoundRecord],
    *,
    primary_reviewer: str | None,
    required_reviewers: Sequence[str],
) -> PlanPanelEvidence:
    """Derive the first qualified planning panel opening from comment order.

    A qualified opening is either an operator-sourced planning force-full
    record, or a ``secondary-audit`` planning scheduler record for candidate
    key K that lists the primary as approved and is preceded by the primary's
    own approved plan review of K.
    """
    secondaries = set(required_reviewers) - {primary_reviewer}
    primary_approved_keys: set[tuple[object, ...]] = set()
    artifacts: list[str] = []
    premature: list[str] = []
    prepanel_latch = False
    opening: PostedRoundRecord | None = None
    opening_source: str | None = None
    for record in sorted(records, key=lambda item: item.index):
        metadata = record.metadata
        valid = metadata.scheduler_metadata_status == "valid"
        record_key = _plan_key_from_payload(metadata.plan_candidate_key)
        if (
            valid
            and metadata.scheduler_force_full
            and metadata.scheduler_force_full_source == "operator"
            # Defensive: an operator opening is still bound to a complete
            # generation-1 candidate key, so a partial-key record can never
            # establish a qualified panel opening.
            and record_key is not None
            and record_key.complete
        ):
            opening, opening_source = record, "operator"
            break
        if (
            valid
            and primary_reviewer is not None
            and metadata.scheduler_phase == "secondary-audit"
            and primary_reviewer in metadata.scheduler_approved_reviewers
            and record_key is not None
            and record_key.complete
            and record_key.components in primary_approved_keys
        ):
            opening, opening_source = record, "primary-approval"
            break
        if (
            metadata.role == "reviewer"
            and metadata.agent == primary_reviewer
            and metadata.state == "approved"
            and record_key is not None
            and record_key.complete
        ):
            primary_approved_keys.add(record_key.components)
        if metadata.role == "reviewer" and metadata.agent in secondaries:
            artifacts.append(
                f"{metadata.agent} plan review (round {metadata.round_number}, "
                f"plan {metadata.subject})"
            )
            if metadata.state == "blocking":
                premature.append(metadata.agent)
        elif valid and metadata.scheduler_phase in PLAN_PANEL_OPENED_PHASES:
            artifacts.append(
                f"{metadata.scheduler_phase} planning scheduler record "
                f"(round {metadata.round_number}, plan {metadata.subject})"
            )
        elif metadata.scheduler_metadata_status == "invalid":
            # Retained for audit even though it grants no phase authority.
            artifacts.append(
                "invalid planning scheduler record "
                f"(round {metadata.round_number}, plan {metadata.subject})"
            )
        if valid and metadata.scheduler_force_full:
            prepanel_latch = True
    post_opening_latch = bool(
        opening is not None
        and any(
            record.index > opening.index
            and record.metadata.scheduler_metadata_status == "valid"
            and record.metadata.scheduler_force_full
            and record.metadata.scheduler_force_full_source != "operator"
            for record in records
        )
    )
    return PlanPanelEvidence(
        opening_index=opening.index if opening is not None else None,
        opening_source=opening_source,
        unqualified_artifacts=tuple(dict.fromkeys(artifacts)),
        premature_secondary_reviews=tuple(dict.fromkeys(premature)),
        prepanel_latch=prepanel_latch,
        post_opening_automatic_latch=post_opening_latch,
    )


def _carried_plan_approvals(
    records: Sequence[PostedRoundRecord],
    *,
    current_key: PlanCandidateKey,
    required_reviewers: Sequence[str],
    surfaced_requirement_ids: Sequence[str],
    panel_evidence: PlanPanelEvidence,
    primary_reviewer: str | None,
    restoration_rounds: dict[str, int] | None = None,
) -> tuple[str, ...]:
    """Reviewers holding a qualifying exact-key approval carried from history.

    A reviewer restored by a signed board amendment (#984) carries nothing
    from rounds before its restoration round, so it must re-approve.

    A carried approval counts only when the stored record matches every
    component of the current candidate key and itself carried
    ``HUMAN_REQUIREMENTS_RESOLVED`` for exactly the currently surfaced
    planning-requirement ID set.  Plan reviewer records persist the surfaced
    requirement IDs only when the approval actually carried that
    acknowledgement, so an approval without it can never satisfy the signed
    requirement gate vacuously.
    """
    if not current_key.complete:
        return ()
    required = set(required_reviewers)
    surfaced = {str(item) for item in surfaced_requirement_ids}
    carried: dict[str, bool] = {}
    for record in sorted(records, key=lambda item: item.index):
        metadata = record.metadata
        if metadata.role != "reviewer" or metadata.agent not in required:
            continue
        if predates_restoration(
            restoration_rounds or {}, metadata.agent, metadata.round_number
        ):
            continue
        if metadata.agent != primary_reviewer and not (
            panel_evidence.opened
            and panel_evidence.opening_index is not None
            and record.index > panel_evidence.opening_index
        ):
            # Secondary records before the qualified opening are unqualified
            # artifacts: never approvals, never ownership.
            continue
        record_key = _plan_key_from_payload(metadata.plan_candidate_key)
        qualifies = (
            metadata.state == "approved"
            and record_key is not None
            and record_key.matches(current_key)
            and (
                not surfaced
                or set(metadata.surfaced_reviewer_requirement_ids) == surfaced
            )
        )
        # A later record for the same reviewer supersedes an earlier one.
        carried[metadata.agent] = bool(qualifies)
    return tuple(sorted(name for name, ok in carried.items() if ok))


@dataclass(frozen=True)
class _StagedPlanHistory:
    history_class: str
    previous_key: PlanCandidateKey | None
    latest_scheduler_record: PostedRoundRecord | None


def _classify_staged_plan_history(
    plan_records: Sequence[PostedRoundRecord],
    *,
    current_key: PlanCandidateKey,
) -> _StagedPlanHistory:
    """Classify staged planning history against the current candidate key.

    The four-class degraded-history partition.  Exactly one outcome each;
    classes A, B, and C continue the live scheduler under a conservative
    fallback, and only a transport extraction failure stops (handled by the
    caller's record extraction).  Shared by the live planning scheduler and
    managed-CI plan recovery so both read the same history as authoritative.
    """
    latest_scheduler_record = next(
        (
            record
            for record in reversed(plan_records)
            if record.metadata.scheduler_metadata_status == "valid"
            and record.metadata.scheduler_contract is not None
        ),
        None,
    )
    previous_key = (
        _plan_key_from_payload(latest_scheduler_record.metadata.plan_candidate_key)
        if latest_scheduler_record is not None
        else None
    )
    has_planning_history = any(
        record.metadata.role == "reviewer" for record in plan_records
    )
    key_contradiction = bool(
        previous_key is not None
        and previous_key.subject == current_key.subject
        and not previous_key.matches(current_key)
    )
    # Degradation is scoped to the current recoverable boundary: the latest
    # valid planning scheduler checkpoint.  An invalid record written before
    # it is historical audit state that stays listed in the audit but must not
    # pin every later round to the fallback forever, which would suppress each
    # fresh exact-key primary approval until the round budget ran out.
    recovery_boundary_index = (
        latest_scheduler_record.index if latest_scheduler_record is not None else -1
    )
    if any(
        record.index > recovery_boundary_index
        and record.metadata.scheduler_metadata_status == "invalid"
        for record in plan_records
    ):
        history_class = classify_plan_history("invalid")
    elif has_planning_history and latest_scheduler_record is None:
        history_class = classify_plan_history("absent")
    else:
        history_class = classify_plan_history(
            "valid", key_contradiction=key_contradiction
        )
    return _StagedPlanHistory(
        history_class=history_class,
        previous_key=previous_key,
        latest_scheduler_record=latest_scheduler_record,
    )


@dataclass(frozen=True)
class PlanPrimaryStreak:
    """Stall streak plus the newest counted rounds that lack an issue digest."""

    count: int
    # Newest consecutive counted rounds with no recorded issue digest: an
    # issue edit ends the streak at the first digested round, so this is the
    # count an edit leaves behind.
    undigested_prefix: int = 0

    @property
    def legacy_undigested(self) -> bool:
        return self.undigested_prefix > 0

    def edit_cannot_clear(self, threshold: int) -> bool:
        """Whether the undigested rounds alone keep the stop tripped after an edit."""
        return threshold > 0 and self.undigested_prefix >= threshold


def plan_issue_text_digest(issue_context: IssueContext) -> str:
    """16-hex digest of the normalized issue title and body (#1112).

    Comments are deliberately excluded: loop-posted comments share the
    operator's account, so a comment cannot be trusted as operator intent.
    """

    def normalize(text: str | None) -> str:
        return "\n".join(
            line.rstrip() for line in (text or "").replace("\r\n", "\n").split("\n")
        )

    payload = json.dumps(
        [normalize(issue_context.title), normalize(issue_context.body)],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def plan_primary_blocking_streak(
    records: Sequence[PostedRoundRecord],
    *,
    primary: str,
    panel_opening_index: int | None = None,
    current_issue_digest: str | None = None,
) -> int:
    """Integer form of :func:`plan_primary_blocking_streak_detail`."""
    return plan_primary_blocking_streak_detail(
        records,
        primary=primary,
        panel_opening_index=panel_opening_index,
        current_issue_digest=current_issue_digest,
    ).count


def plan_primary_blocking_streak_detail(
    records: Sequence[PostedRoundRecord],
    *,
    primary: str,
    panel_opening_index: int | None = None,
    current_issue_digest: str | None = None,
    current_execution_mode: str | None = None,
) -> PlanPrimaryStreak:
    """Consecutive completed blocking primary plan reviews (#1103).

    A round whose checkpoint recorded a different execution mode than
    ``current_execution_mode`` ends the streak before it counts, exactly like
    an issue-digest mismatch: that review judged another execution contract
    (#1268).  Rounds with no recorded mode count.

    Two further boundaries retire older rounds (#1112): a round whose
    checkpoint recorded a different issue digest than ``current_issue_digest``
    (that review judged other issue text) ends the streak before it counts, and
    a checkpoint carrying the operator reset marker ends it after counting that
    round's own review, or immediately when its review never completed.
    Rounds with no recorded digest count.  ``undigested_prefix`` counts the
    newest consecutive counted rounds with none: an issue edit ends the streak
    at the newest digested round, so those rounds are exactly what an edit
    leaves counting.

    Counts, newest round first, the rounds whose primary review ended
    ``blocking`` and follows a valid ``primary``-phase scheduler checkpoint of
    the same round.  A round with a valid primary-phase checkpoint but no
    primary review (interrupted, or the primary was unavailable) is skipped:
    it neither counts nor resets.  Everything else ends the streak: any
    primary approval (whatever its key), any other review state, a missing or
    invalid checkpoint, a non-primary phase, an invalid scheduler record or a
    phase-advance record positioned after the round's checkpoint, and anything
    at or after a qualified panel opening.  Degraded history can therefore
    only shorten the streak, never lengthen it.
    """
    ordered = sorted(records, key=lambda item: item.index)
    # The newest history boundary: nothing at or before it may count.
    boundary_index = max(
        (
            record.index
            for record in ordered
            if record.metadata.scheduler_metadata_status == "invalid"
            or (
                record.metadata.role == "summary"
                and record.metadata.phase == "plan-phase-advance"
            )
        ),
        default=-1,
    )
    checkpoints: dict[int, list[PostedRoundRecord]] = {}
    reviews: dict[int, PostedRoundRecord] = {}
    for record in ordered:
        metadata = record.metadata
        if metadata.role == "summary" and metadata.phase == "scheduler-prelaunch":
            checkpoints.setdefault(metadata.round_number, []).append(record)
        elif metadata.role == "reviewer" and metadata.agent == primary:
            # A later record for the same round supersedes an earlier one.
            reviews[metadata.round_number] = record

    def usable_primary_checkpoint(checkpoint: PostedRoundRecord | None) -> bool:
        return (
            checkpoint is not None
            and checkpoint.index > boundary_index
            and (panel_opening_index is None or checkpoint.index < panel_opening_index)
            and checkpoint.metadata.scheduler_metadata_status == "valid"
            and checkpoint.metadata.scheduler_phase == "primary"
        )

    def has_reset(round_checkpoints: list[PostedRoundRecord]) -> bool:
        return any(
            usable_primary_checkpoint(record) and record.metadata.scheduler_stall_reset
            for record in round_checkpoints
        )

    streak = 0
    undigested_prefix = 0
    for number in sorted(set(checkpoints) | set(reviews), reverse=True):
        review = reviews.get(number)
        round_checkpoints = checkpoints.get(number, [])
        if review is None:
            if usable_primary_checkpoint(
                round_checkpoints[-1] if round_checkpoints else None
            ):
                if has_reset(round_checkpoints):
                    break
                continue
            break
        if panel_opening_index is not None and review.index >= panel_opening_index:
            break
        checkpoint = next(
            (
                record
                for record in reversed(round_checkpoints)
                if record.index < review.index
            ),
            None,
        )
        if not usable_primary_checkpoint(checkpoint) or review.metadata.state != "blocking":
            break
        recorded_digest = checkpoint.metadata.scheduler_issue_digest
        if (
            recorded_digest is not None
            and current_issue_digest is not None
            and recorded_digest != current_issue_digest
        ):
            break
        recorded_mode = checkpoint.metadata.scheduler_execution_mode
        if (
            recorded_mode is not None
            and current_execution_mode is not None
            and recorded_mode != current_execution_mode
        ):
            break
        # An edit changes the current digest and ends the streak at the newest
        # digested round, so only the undigested *newest* counted rounds are
        # out of an edit's reach.
        if recorded_digest is None and undigested_prefix == streak:
            undigested_prefix += 1
        streak += 1
        if has_reset(round_checkpoints):
            break
    return PlanPrimaryStreak(count=streak, undigested_prefix=undigested_prefix)


def plan_primary_stall_message(
    *,
    streak: int,
    threshold: int,
    plan_chars: int,
    legacy_undigested: bool = False,
    step_back_status: str | None = None,
) -> str:
    """Operator diagnostic for the primary-phase stall stop (#1103, #1112)."""
    step_back_note = (
        f" Plan step-back did not apply: {step_back_status}."
        if step_back_status
        else ""
    )
    legacy_note = (
        " The newest counted rounds predate issue-text tracking and by themselves "
        "reach the threshold, so editing the issue cannot retire them or clear "
        "this stop and --plan-reset-stall-streak is required."
        if legacy_undigested
        else ""
    )
    return (
        f"the primary plan reviewer has blocked {streak} consecutive primary-phase "
        f"planning round(s) with no exact-plan primary approval, reaching "
        f"--plan-primary-stall-rounds {threshold} (current canonical plan: "
        f"{plan_chars} characters). The secondary plan panel was not convened, "
        "because it opens only after an exact-plan primary approval; no reviewer and "
        "no planner turn were invoked. To continue: edit the issue title or body "
        "to narrow it (rounds reviewed against earlier issue text stop counting; a "
        "comment alone does not change the issue text and does not clear this stop); "
        "rerun with --plan-reset-stall-streak to retire the counted rounds, for "
        "example after narrowing the issue by comment; rerun with "
        "--plan-review-force-full to authorize the complete plan board; or raise "
        "--plan-primary-stall-rounds (or pass 0 to disable this stop). Retiring the "
        "streak resumes with the primary review of the current candidate plan, which "
        "the planner then revises against the narrowed issue. If you changed "
        "--plan-execution-mode since these rounds and they predate execution-mode "
        "recording, rerun with --plan-reset-stall-streak to retire them."
        + legacy_note
        + step_back_note
    )


def _planner_candidate_rounds(comments: Sequence[object]) -> set[int]:
    """Round numbers of authenticated planner-authored plan candidates (#886).

    Only coder plan rounds that carry an assembled plan state count, so
    reviewer-only phase-advance rounds never do, and a resumed replay of the
    same round number counts once.  ``round_number`` is never used as a
    count directly.
    """
    try:
        records = _extract_round_metadata_records(comments, flow="plan")
    except AgentLoopError:
        return set()
    return {
        record.metadata.round_number
        for record in records
        if record.metadata.role == "coder"
        and record.metadata.canonical_plan is not None
        and record.metadata.assembled_plan_sidecar is not None
    }


def _plan_growth_candidate_count(candidate_rounds: set[int], round_number: int) -> int:
    """Planner candidates up to and including the one published at ``round_number``.

    ``round_number`` must be the round that published (or, for the
    pre-publication self-check, will publish) the candidate; callers never
    pass a reviewer-only round.
    """
    return len({item for item in candidate_rounds if item < round_number} | {round_number})


def _plan_growth_gate_violation(
    config: AgentLoopConfig,
    *,
    plan_payload: Mapping[str, object] | None,
    plan_text: str,
    revision_count: int,
) -> tuple[PlanGrowthAssessment | None, str | None]:
    """Assessment and gate violation of one authenticated plan candidate.

    ``None`` violation for legacy unversioned plans, free-form plans and when
    ``--plan-growth-gate off``.
    """
    if plan_payload is None or plan_strategy(plan_payload) is None:
        return None, None
    assessment = assess_plan_growth(
        plan_payload,
        rendered_chars=len(plan_text),
        revision_count=revision_count,
        thresholds=PlanGrowthThresholds.from_config(config),
    )
    if not plan_growth_gate_enforced(config):
        return assessment, None
    return assessment, growth_justification_violation(plan_payload, assessment)


def _require_plan_growth_compliance(
    comments: Sequence[object],
    *,
    config: AgentLoopConfig,
    plan_text: str,
    plan_round: ResumedReviewRound,
    error_message: str,
) -> None:
    """Fail closed when carried approvals would approve a non-compliant plan (#886)."""
    coder_metadata = plan_round.coder_metadata
    if coder_metadata is None or coder_metadata.assembled_plan_sidecar is None:
        return
    sidecar = decode_assembled_plan_sidecar(coder_metadata.assembled_plan_sidecar)
    _assessment, violation = _plan_growth_gate_violation(
        config,
        plan_payload=sidecar.canonical_json,
        plan_text=plan_text,
        # The coder round that published the candidate, not the resumed
        # anchor round (which may be a later reviewer-only round).
        revision_count=_plan_growth_candidate_count(
            _planner_candidate_rounds(comments), sidecar.round_number
        ),
    )
    if violation is not None:
        raise AgentLoopError(
            f"{error_message} Plan growth gate: {violation} Re-run planning so a revision "
            "restructures the plan as staged or carries a reviewed justification."
        )


def _plan_growth_verdict_for_hash(
    config: AgentLoopConfig,
    *,
    plan_hash: str,
    comment_sources: Sequence[Sequence[object] | None],
) -> PlanGrowthApprovalVerdict:
    """Approval-time growth verdict of the plan ``plan_hash`` names (#1074).

    Measured exactly as the approval gate measures it: the authenticated
    coder round's stored canonical text and assembled state, and the
    planner-candidate count up to the round that published it.  The first
    comment source holding that candidate wins (the issue, then a staged
    parent).  With no authenticated v1 candidate the plan is legacy or
    free-form, which the gate itself exempts, so the verdict says
    ``not-applicable``.
    """
    for comments in comment_sources:
        if comments is None:
            continue
        for record in reversed(_extract_round_metadata_records(comments, flow="plan")):
            metadata = record.metadata
            if (
                metadata.role != "coder"
                or metadata.canonical_plan is None
                or metadata.assembled_plan_sidecar is None
                or approved_plan_hash(metadata.canonical_plan) != plan_hash
            ):
                continue
            sidecar = decode_assembled_plan_sidecar(metadata.assembled_plan_sidecar)
            assessment, _violation = _plan_growth_gate_violation(
                config,
                plan_payload=sidecar.canonical_json,
                plan_text=metadata.canonical_plan,
                revision_count=_plan_growth_candidate_count(
                    _planner_candidate_rounds(comments), sidecar.round_number
                ),
            )
            return plan_growth_approval_verdict(config, sidecar.canonical_json, assessment)
    return plan_growth_approval_verdict(config, None, None)


def _require_handoff_plan_growth_verdict(
    config: AgentLoopConfig,
    handoff: IssuePrHandoffMetadata,
    *,
    context: str,
) -> None:
    """Re-validate a handoff-backed resume against its recorded verdict (#1074).

    Never against today's thresholds: a plan judged compliant when approved
    stays compliant.  A record with no verdict predates it and is accepted
    as legacy by definition.
    """
    if handoff.flow != "approved-plan-implementation":
        return
    verdict = handoff.plan_growth_verdict
    if verdict is None:
        log(
            config,
            f"{context}: approved-plan handoff for PR #{handoff.pr_number} predates the recorded "
            "plan-growth verdict; accepted as a legacy handoff",
        )
        return
    violation = handoff_growth_verdict_violation(verdict, config)
    if violation is not None:
        raise AgentLoopError(
            f"{context}: approved plan {handoff.plan_hash} handed off to PR "
            f"#{handoff.pr_number} cannot resume: {violation}"
        )


def _require_complete_canonical_plan_approval(
    comments: Sequence[object],
    *,
    config: AgentLoopConfig,
    plan_text: str,
    plan_round: ResumedReviewRound,
    human_requirements: Sequence[HumanReviewRequirement],
    error_message: str,
) -> None:
    """Fail closed unless every configured reviewer approved the resumed plan.

    Shared by the managed-CI fresh-authorization and ordinary-resume paths so
    the two cannot drift.  Under ``all-reviewers`` every reviewer reviews every
    round, so the resumed round must carry the complete approval set.  A
    staged policy splits approvals across rounds by construction (the primary
    approves, then the panel approves while the primary is paused), so the
    union of qualifying exact-key approvals carried from the whole planning
    history is always consulted instead — the same carry the final planning gate
    uses.  A reviewer whose latest record for the exact plan is not an
    approval still leaves the set incomplete (#962).  Carried approvals of a
    plan that fails the plan-growth gate never count (#886).
    """
    _require_plan_growth_compliance(
        comments,
        config=config,
        plan_text=plan_text,
        plan_round=plan_round,
        error_message=error_message,
    )
    configured_names = {agent_display_name(reviewer) for reviewer in reviewers(config)}
    if not plan_policy_capabilities(config.plan_review_policy).scheduler_enabled:
        round_approved = {
            record.metadata.agent
            for record in plan_round.completed_reviews
            if record.metadata.state == "approved"
        }
        if round_approved != configured_names:
            raise AgentLoopError(error_message)
        return
    # Every staged recovery goes through the exact-key gate, even when one
    # full-board round holds the whole set: an approval bound to a stale key
    # or surfaced-requirement set must never satisfy it.
    coder_metadata = plan_round.coder_metadata
    if coder_metadata is None or coder_metadata.assembled_plan_sidecar is None:
        raise AgentLoopError(error_message)
    sidecar = decode_assembled_plan_sidecar(coder_metadata.assembled_plan_sidecar)
    required_names = tuple(agent_display_name(reviewer) for reviewer in reviewers(config))
    contract = make_plan_contract(
        required_names,
        config.plan_review_policy,
        (
            agent_display_name(config.primary_plan_reviewer)
            if config.primary_plan_reviewer is not None
            else None
        ),
    )
    surfaced_ids = _surfaced_reviewer_requirement_ids(
        human_requirements,
        requirement_scope="planning requirements",
    )
    current_key = _plan_candidate_key_for(
        plan_subject=_plan_subject(plan_text),
        sidecar=sidecar,
        surfaced_requirement_ids=surfaced_ids,
    )
    records = _extract_round_metadata_records(comments, flow="plan")
    # Only history the live scheduler itself treats as authoritative for this
    # key may supply approvals; any degraded class would make it re-review.
    if (
        _classify_staged_plan_history(records, current_key=current_key).history_class
        != PLAN_HISTORY_INTACT
    ):
        raise AgentLoopError(error_message)
    carried = _carried_plan_approvals(
        records,
        current_key=current_key,
        required_reviewers=required_names,
        surfaced_requirement_ids=surfaced_ids,
        panel_evidence=_derive_plan_panel_evidence(
            records,
            primary_reviewer=contract.primary_reviewer,
            required_reviewers=required_names,
        ),
        primary_reviewer=contract.primary_reviewer,
        restoration_rounds=restoration_rounds_from_comments(comments, flow="plan"),
    )
    if set(carried) != configured_names:
        raise AgentLoopError(error_message)


def _plan_cross_cutting_contracts(
    sidecar: AssembledPlanSidecar | None,
) -> PlanCrossCuttingContracts:
    """Observe the authenticated cross-cutting plan contracts from a sidecar.

    An unobserved identity stays ``None`` so the classifier reports it rather
    than comparing two defaulted objects equal.
    """
    if sidecar is None:
        return PlanCrossCuttingContracts()
    payload = dict(sidecar.canonical_json)

    def digest(value: object) -> str | None:
        if value is None:
            return None
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()[:32]

    architecture = payload.get("architecture_impact")
    status = (
        architecture.get("status") if isinstance(architecture, dict) else None
    )
    recommendation = payload.get("execution_recommendation")
    return PlanCrossCuttingContracts(
        # The executable contract only: narrative-only edits (rationale,
        # caveats, coupling rationale) keep the identity (#1103).  The
        # candidate key still digests the full content.
        execution_recommendation_identity=execution_recommendation_contract_identity(
            recommendation
        ),
        execution_recommendation_projection=execution_recommendation_contract_projection(
            recommendation
        ),
        human_requirement_disposition_digest=digest(
            payload.get("human_requirement_dispositions", [])
        ),
        additional_closing_issue_ids=tuple(
            str(item) for item in payload.get("additional_closing_issue_ids", []) or ()
        ),
        architecture_impact_status=status if status in {"changed", "unchanged"} else None,
    )


def _resumed_plan_transition_inputs(
    comments: Sequence[object],
    *,
    coder_metadata: PostedRoundMetadata | None,
) -> tuple[object | None, PlanCrossCuttingContracts | None]:
    """Rebuild the authenticated transition inputs a resume would otherwise lose.

    The classifier decides `narrow` from the authenticated `semantic-patch-v1`
    payload and the cross-cutting contracts of the state the patch was bound to.
    Both live in durable records, so a restart immediately after a remediation
    coder turn must reconstruct them instead of classifying the same plan-step
    revision `broad` and latching the complete board (#905, from #841).

    Every binding is re-verified here: the patch must match the record's own
    base round and base state identity, and the base round's sidecar must
    hydrate to exactly that state identity.  Anything unverifiable returns
    ``(None, None)``, which keeps the conservative broad classification.
    """
    if coder_metadata is None or coder_metadata.response_form != "semantic-patch-v1":
        return None, None
    try:
        patch = parse_plan_revision_patch(coder_metadata.raw_patch_provenance or {})
    except (AgentLoopError, TypeError, ValueError, KeyError) as exc:
        del exc
        return None, None
    if (
        patch.base_round_number != coder_metadata.base_round_number
        or patch.base_state_identity != coder_metadata.base_state_identity
        or patch.base_state_identity is None
    ):
        return None, None
    try:
        records = _extract_round_metadata_records(comments, flow="plan")
    except AgentLoopError:
        return None, None
    for record in reversed(records):
        metadata = record.metadata
        if (
            metadata.role != "coder"
            or metadata.round_number != patch.base_round_number
            or metadata.assembled_plan_sidecar is None
        ):
            continue
        try:
            base_sidecar = decode_assembled_plan_sidecar(metadata.assembled_plan_sidecar)
            if (
                hydrate_authenticated_plan_state(base_sidecar).state_identity
                != patch.base_state_identity
            ):
                continue
        except (AgentLoopError, TypeError, ValueError, KeyError) as exc:
            del exc
            continue
        return patch, _plan_cross_cutting_contracts(base_sidecar)
    return None, None


def _plan_revision_descriptor(
    *,
    response_form: str | None,
    sidecar: AssembledPlanSidecar | None,
    patch: object | None,
) -> PlanRevisionDescriptor | None:
    """Describe the authenticated revision that produced the current key."""
    if response_form != "semantic-patch-v1" or patch is None or sidecar is None:
        return None
    operations = getattr(patch, "operations", ())
    fields = tuple(
        str(operation.field)
        for operation in operations
        if getattr(operation, "op", None) == "replace"
        and getattr(operation, "field", None)
    )
    matrix_ops = tuple(
        str(operation.op)
        for operation in operations
        if getattr(operation, "op", None) not in {None, "replace"}
    )
    return PlanRevisionDescriptor(
        response_form="semantic-patch-v1",
        semantic_patch_contract_version=int(
            getattr(patch, "semantic_patch_contract_version", 0) or 0
        ),
        base_state_identity=getattr(patch, "base_state_identity", None),
        sidecar_bound=True,
        operation_fields=fields,
        matrix_operations=matrix_ops,
    )
