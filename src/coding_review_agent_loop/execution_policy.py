"""Execution-policy resolution, child routing/provenance, staged reporting
and fresh-topology preflights.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1197); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass
from .agents.registry import agent_display_name, get_backend
from .config import AgentLoopConfig, phased_delivery_guard_active, reviewers
from .decomposition import (
    _decode_json_payload,
    EXECUTION_DECISION_MARKER_RE,
    issue_has_execution_decision,
    CreatedPhaseIssue,
    PlanDecomposition,
    RecordedPhase,
    approved_plan_hash,
    create_decomposition_child_issues,
    find_existing_decomposition,
    find_existing_one_shot_impl_handoff,
    find_one_shot_impl_handoffs,
    find_latest_one_shot_impl_handoff,
    post_phase_implementation_handoff_comment,
    normalize_execution_recommendation,
    recover_execution_recommendation,
    ExecutionDecision,
    EXECUTION_TOPOLOGY_SOURCE,
    find_existing_execution_decision,
    live_execution_decisions,
    reject_legacy_topology_collision,
    post_execution_decision,
    find_existing_topology_checkpoint,
    find_topology_checkpoints_for_parent,
    find_decompositions_for_parent,
    find_phase_implementation_handoffs_for_parent,
    PHASE_IDENTITY_MARKER_RE,
    phase_identity,
    ChildDispositionOverride,
    PhaseImplementationHandoffMetadata,
    collect_child_disposition_overrides,
    phase_direct_readiness_problems,
    reconcile_handoff_disposition,
    retained_parent_scope_matches,
)
from .protocol import EXECUTION_DISPOSITION_DIRECT, EXECUTION_DISPOSITION_PLANNING
from .child_topology import NeedsHumanDecision, parent_child_search_queries
from .errors import AgentLoopError
from .integration_close import require_child_closed_or_report
from .github import (
    _REST_ISSUE_COMMENT_PAGE_SIZE,
    strip_bot_login_suffix,
    IssueContext,
    get_issue_context,
    parse_strong_issue_reference_evidence,
    get_pr_state,
    post_issue_comment,
    read_rest_issue_comments,
    resolve_authenticated_github_actor,
    search_issues,
)
from .issue_pr_handoff import (
    AGENT_ISSUE_PR_HANDOFF_RE,
    decode_issue_pr_handoff_record,
    resolve_canonical_pr_for_issue,
)
from .issue_pr_provenance import IssuePrProvenanceScope
from .phase_progress import (
    PhaseProgress,
    StagedTopologyOutcome,
    record_staged_completion,
    render_phase_status_line,
    resolve_staged_phase_progress,
    select_current_phase,
)
from .split_materialization import (
    SPLIT_CHILD_MARKER_RE,
    dedupe_split_stage_proposals,
    find_existing_split_materialization,
    has_unfiled_split_warning,
    materialize_split_proposals,
    post_unfiled_split_warning,
    split_stage_proposal_from_deferred_stage,
    split_stage_proposal_from_text,
)
from .logging import log
from .protocol import (
    ChildStage,
    DeferredStage,
    ExecutionStrategyRecommendation,
    validate_structured_plan_state,
    validate_structured_plan_revision,
)
from .runner import Runner
from .workdirs import active_workdir
from .comment_rendering import (
    DEFERRED_STAGES_MARKER_RE,
    EXECUTION_RECOMMENDATION_MARKER_RE,
    decode_deferred_stages_marker,
    decode_execution_recommendation_marker,
    PLAN_EXPECTED_CLOSING_MARKER_RE,
    decode_expected_closing_issue_declaration,
)
from .followups import (
    PLAN_APPROVED_FOLLOWUP_MARKER_RE,
    approved_plan_hashes_for_issue,
)
from .round_state import (
    ApprovedPlanContext,
    ROUND_RESUME_MARKER_RE,
    _extract_round_metadata_records,
    _plan_subject,
    make_approved_plan_context,
    recover_approved_plan_context,
)
from .round_transport import decode_mapping
from .protocol_markers import TrustedBody
from .response_validation import ResolvedExecution
from .discuss_loop import _recover_final_discuss_split_proposals


_DEFERRED_STAGES_SECTION_RE = re.compile(
    r"^###\s*Deferred stages \(not in this plan\)\s*$",
    re.M,
)
_NEXT_HEADING_RE = re.compile(r"^#{1,6}\s", re.M)
_PLAN_NARROWING_PHRASE_RE = re.compile(
    r"\b(stage \d+ of|first stage|out of scope|separate issue|follow-up issue|future issue)\b",
    re.I,
)


def _extract_current_deferred_stages(current_plan: str) -> tuple[DeferredStage, ...]:
    """Recover declared `deferred_stages` from the current plan text (#476).

    `current_plan` is either the raw structured `plan_state` JSON response (the
    initial round, never revised) or the canonical revision markdown (after a
    `plan_revision` round), so both forms are checked.
    """
    try:
        structured = validate_structured_plan_state(current_plan, architecture_status_mode="legacy")
    except AgentLoopError:
        structured = None
    if structured is not None:
        return structured.deferred_stages
    # The canonical markdown carries an AGENT_DEFERRED_STAGES marker with the
    # exact structured title/summary pairs (#492 review): a title containing
    # its own colon (e.g. "Stage 2: API follow-up") would corrupt the
    # human-readable `- {title}: {summary}` bullets if split on the first
    # colon, so the marker is authoritative and the prose is parsed only as a
    # fallback for text that predates it.
    marker_match = DEFERRED_STAGES_MARKER_RE.search(current_plan)
    if marker_match:
        return decode_deferred_stages_marker(marker_match.group("payload"))
    match = _DEFERRED_STAGES_SECTION_RE.search(current_plan)
    if not match:
        return ()
    section = current_plan[match.end():]
    next_heading = _NEXT_HEADING_RE.search(section)
    if next_heading:
        section = section[: next_heading.start()]
    stages: list[DeferredStage] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        title_and_summary = stripped[2:]
        title, _sep, summary = title_and_summary.partition(":")
        stages.append(DeferredStage(title=title.strip(), summary=summary.strip()))
    return tuple(stages)


def _extract_current_expected_closing_issue_ids(
    current_plan: str,
) -> tuple[int, ...] | None:
    """Recover the optional plan declaration from JSON or canonical Markdown."""
    for validator in (validate_structured_plan_state, validate_structured_plan_revision):
        try:
            structured = validator(current_plan)
        except AgentLoopError:
            structured = None
        if structured is not None:
            return structured.additional_closing_issue_ids
    marker = PLAN_EXPECTED_CLOSING_MARKER_RE.search(current_plan)
    if marker is None:
        return None
    return decode_expected_closing_issue_declaration(marker.group("payload"))


def _extract_current_child_stages(current_plan: str) -> tuple[ChildStage, ...]:
    """Return only explicitly typed legacy child stages.

    Fresh generation-1 plans have one reviewed topology. Their enriched
    recommendation stages remain audit-only in Stage 1, and a top-level
    legacy child category is rejected by protocol validation rather than being
    adopted as a second executable topology. Only unversioned historical
    plans can return legacy two-field child stages here.
    """
    try:
        structured = validate_structured_plan_state(current_plan, architecture_status_mode="legacy")
    except AgentLoopError:
        structured = None
    if structured is not None:
        if structured.execution_strategy_contract_version == 1:
            return ()
        return structured.typed_stages.child_stages
    marker = re.search(r"<!--\s*AGENT_TYPED_PLAN_STAGES:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", current_plan, re.I)
    if not marker:
        return ()
    try:
        payload = _decode_json_payload(marker.group("payload"), marker_name="AGENT_TYPED_PLAN_STAGES")
        children = payload.get("child_stages", [])
        if not isinstance(children, list):
            return ()
        return tuple(
            ChildStage(str(item["title"]), str(item["summary"]))
            for item in children
            if isinstance(item, dict)
            and isinstance(item.get("title"), str)
            and isinstance(item.get("summary"), str)
        )
    except AgentLoopError:
        return ()


def _log_typed_plan_stage_dispositions(current_plan: str, *, config: AgentLoopConfig) -> None:
    """Make the record-only typed categories visible in CLI output (#585)."""
    try:
        structured = validate_structured_plan_state(current_plan, architecture_status_mode="legacy")
    except AgentLoopError:
        structured = None
    if structured is not None:
        for entry in structured.deferred_stages:
            log(config, f"Plan scope: recorded-only legacy deferred stage: {entry.title}.")
        categories = (
            ("linked dependency", structured.typed_stages.external_dependencies),
            ("recorded-only deferred work", structured.typed_stages.deferred_work),
            ("recorded-only plan action", structured.typed_stages.plan_actions),
        )
        for disposition, entries in categories:
            for entry in entries:
                log(config, f"Plan scope: {disposition}: {entry.title}.")
        return
    for entry in _extract_current_deferred_stages(current_plan):
        log(config, f"Plan scope: recorded-only legacy deferred stage: {entry.title}.")
    marker = re.search(
        r"<!--\s*AGENT_TYPED_PLAN_STAGES:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
        current_plan,
        re.I,
    )
    if not marker:
        return
    try:
        payload = _decode_json_payload(
            marker.group("payload"), marker_name="AGENT_TYPED_PLAN_STAGES"
        )
    except AgentLoopError:
        return
    for field, disposition in (
        ("external_dependencies", "linked dependency"),
        ("deferred_work", "recorded-only deferred work"),
        ("plan_actions", "recorded-only plan action"),
    ):
        entries = payload.get(field, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("title"), str):
                log(config, f"Plan scope: {disposition}: {entry['title']}.")


def _prior_discuss_split_proposals(
    issue_context: IssueContext, *, config: AgentLoopConfig
) -> list[str]:
    """Recover proposals from a prior discuss `split` consensus on this issue (#476).

    Used by plan-first narrowing so a plan built on top of an earlier discuss
    split still files (or warns about) the stages the plan itself doesn't
    cover, even when discuss-mode materialization was never run.
    """
    records = _extract_round_metadata_records(issue_context.comments, flow="discuss")
    final_summaries = [
        record for record in records if record.metadata.role == "summary" and record.metadata.is_final
    ]
    if not final_summaries:
        return []
    latest = final_summaries[-1]
    if latest.metadata.split_proposals:
        return list(latest.metadata.split_proposals)
    configured_reviewers = reviewers(config)
    reviewer_workdirs = {
        agent_display_name(agent): get_backend(agent).workdir(config) for agent in configured_reviewers
    }
    recovered = _recover_final_discuss_split_proposals(
        issue_context,
        subject=latest.metadata.subject,
        configured_reviewers=configured_reviewers,
        reviewer_workdirs=reviewer_workdirs,
    )
    return list(recovered[0]) if recovered else []


def _plan_text_suggests_narrowing(plan_text: str) -> bool:
    return bool(_PLAN_NARROWING_PHRASE_RE.search(plan_text))


def _plan_first_line(plan_text: str) -> str:
    """A title-like string for `plan_text`, used for split-stage title matching.

    For the initial (never-revised) `plan_state` round, `plan_text` is the raw
    structured JSON response rather than rendered markdown, so its natural
    "title" is the structured `summary` field, not its literal first line.
    """
    try:
        structured = validate_structured_plan_state(plan_text, architecture_status_mode="legacy")
    except AgentLoopError:
        structured = None
    if structured is not None:
        return structured.summary
    for line in plan_text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped
    return plan_text.strip()


def _current_execution_recommendation(
    current_plan: str,
    issue_comments: Sequence[object],
):
    """Recover the complete fresh recommendation from the current plan.

    The visible plan may contain a compact recommendation reference. Hydration
    intentionally uses the complete issue-comment set so the existing bounded
    sidecar transport remains the only spill mechanism.
    """
    marker = EXECUTION_RECOMMENDATION_MARKER_RE.search(current_plan)
    if marker is None:
        try:
            parsed = validate_structured_plan_state(current_plan, architecture_status_mode="legacy")
        except AgentLoopError:
            parsed = None
        if parsed is not None and parsed.execution_recommendation is not None:
            return parsed.execution_recommendation
        return None
    bodies = [current_plan]
    bodies.extend(
        body for comment in issue_comments
        if isinstance((body := getattr(comment, "body", None)), str)
    )
    recommendation = decode_execution_recommendation_marker(
        marker.group("payload"), bodies=bodies
    )
    from .protocol import parse_execution_recommendation_payload

    return parse_execution_recommendation_payload(
        recommendation, context="approved execution_recommendation"
    )


def _normalize_requested_execution_policy(
    config: AgentLoopConfig,
    *,
    requested_policy: str | None = None,
    implement_after_approval: bool = False,
) -> str:
    """Return the one requested policy before any reviewed strategy exists.

    The historical boolean remains accepted at the Python API boundary for
    callers that have not migrated, but it is only an alias for one-shot
    execution.  It cannot become a second routing input alongside ``auto``.
    """
    requested = requested_policy or config.plan_execution_mode or "plan-only"
    if implement_after_approval:
        if requested not in {"plan-only", "implement-one-shot"}:
            raise AgentLoopError(
                "--implement-after-approval is only a one-shot alias and cannot be "
                f"combined with requested policy `{requested}`."
            )
        return "implement-one-shot"
    return requested


def _resolve_execution_policy(
    config: AgentLoopConfig,
    *,
    requested_policy: str | None = None,
    implement_after_approval: bool = False,
    recommendation: ExecutionStrategyRecommendation | None,
) -> ResolvedExecution:
    """Resolve one requested policy against one reviewed recommendation.

    This is the approval-bound routing seam.  Before approval, ``auto`` is
    only metadata in the prompt/configuration; after approval it must resolve
    to one of the concrete actions consumed by every downstream branch.
    """
    requested = _normalize_requested_execution_policy(
        config,
        requested_policy=requested_policy,
        implement_after_approval=implement_after_approval,
    )
    if recommendation is None:
        if requested == "auto":
            raise AgentLoopError(
                "Automatic execution policy requires a fresh reviewed execution "
                "recommendation; the approved plan is legacy-undecided. Re-run plan-first "
                "planning to produce a reviewed revision. No approval-bound work was published."
            )
        return ResolvedExecution(
            requested_policy=requested,
            action=requested,
            strategy=None,
            recommendation=None,
        )
    if requested == "auto":
        action = (
            "implement-one-shot"
            if recommendation.strategy == "one-shot"
            else "implement-by-phase"
        )
        return ResolvedExecution(
            requested_policy=requested,
            action=action,
            strategy=recommendation.strategy,
            recommendation=recommendation,
        )
    if requested == "plan-only":
        # Plan-only remains a non-executing audit mode for both strategies.
        return ResolvedExecution(
            requested_policy=requested,
            action=requested,
            strategy=recommendation.strategy,
            recommendation=recommendation,
        )
    allowed = (
        {"implement-one-shot"}
        if recommendation.strategy == "one-shot"
        else {"decompose-only", "implement-by-phase"}
    )
    if requested not in allowed:
        expected = (
            "implement-one-shot"
            if recommendation.strategy == "one-shot"
            else "decompose-only or implement-by-phase"
        )
        raise AgentLoopError(
            "Fresh execution policy is incompatible with the approved execution strategy: "
            f"strategy `{recommendation.strategy}` cannot run requested policy `{requested}`; "
            f"use `{expected}` or rerun plan-only. No approval-bound work was published."
        )
    return ResolvedExecution(
        requested_policy=requested,
        action=requested,
        strategy=recommendation.strategy,
        recommendation=recommendation,
    )


CHILD_ROUTE_HUMAN = "human"


@dataclass(frozen=True)
class ChildExecutionRoute:
    """The deterministic route selected for one materialized child (#808).

    ``disposition`` is ``direct-implementation``, ``requires-child-planning``,
    or ``human``.  ``override_digest`` is the digest of the signed override that
    selected the route (recorded in the handoff), if any.  ``origin`` is a
    diagnostic only: ``phase``, ``handoff``, ``override``, ``automation``,
    ``legacy-ambiguous``, or ``unsupported-source``.
    """

    disposition: str
    override_digest: str | None = None
    origin: str = "phase"

    @property
    def is_direct(self) -> bool:
        return self.disposition == EXECUTION_DISPOSITION_DIRECT

    @property
    def is_planning(self) -> bool:
        return self.disposition == EXECUTION_DISPOSITION_PLANNING

    @property
    def is_human(self) -> bool:
        return self.disposition == CHILD_ROUTE_HUMAN


def resolve_child_execution_route(
    phase,
    *,
    topology_source: str,
    recorded_handoff: PhaseImplementationHandoffMetadata | None = None,
    overrides: Sequence[ChildDispositionOverride] = (),
) -> ChildExecutionRoute:
    """The one routing seam shared by parent dispatch and direct child entry.

    Effective-disposition algorithm, evaluated in order:

    (a) A non-``agent-pr`` stage routes to ``human``; an override naming it is
        a human-decision error.
    (b) With a recorded handoff, the handoff's disposition (legacy absent =
        direct) is confirmed by the shared reconciliation rule and returned
        without re-validation.  A distinct override that disagrees with an
        already recorded route is an error (override after dispatch).
    (c) Without a handoff, identical override records collapse by digest; more
        than one distinct record for the stage is an error (no latest-wins).
        A single override sets the effective disposition; otherwise the
        persisted phase disposition is used, and ``None`` (legacy-ambiguous)
        or a non-``approved-plan-v1`` source resolves to child planning.
    (d) A direct result is accepted only when the persisted approved parent
        phase fields alone pass the shared direct-readiness validator.
        Override records never supply readiness evidence.

    ``overrides`` must already be the stage-scoped, topology-validated set for
    this phase (see ``collect_child_disposition_overrides``).
    """
    stage_label = getattr(phase, "stage_id", None) or str(getattr(phase, "position", None) or "?")
    phase_disposition = getattr(phase, "execution_disposition", None)
    if phase.automation != "agent-pr":
        if overrides:
            record = overrides[0]
            raise AgentLoopError(
                "Human decision required: signed child-disposition override at "
                f"{record.comment_locator} names stage `{stage_label}`, which is a "
                f"{phase.automation} stage. Human-owned stages keep their human stop and can "
                "never be converted to an agent-pr route; remove the record before rerunning."
            )
        return ChildExecutionRoute(CHILD_ROUTE_HUMAN, None, "automation")
    if recorded_handoff is not None:
        effective = reconcile_handoff_disposition(phase, recorded_handoff, overrides)
        disagreeing = [
            record for record in overrides
            if record.disposition != effective
            and record.digest != recorded_handoff.override_digest
        ]
        if disagreeing:
            raise AgentLoopError(
                f"Human decision required: stage `{stage_label}` was already dispatched with "
                f"disposition `{effective}` (recorded phase handoff), but the signed override at "
                f"{disagreeing[0].comment_locator} now requests `{disagreeing[0].disposition}`. "
                "An override cannot change a recorded route; remove the record or repair the "
                "handoff manually."
            )
        return ChildExecutionRoute(effective, recorded_handoff.override_digest, "handoff")
    distinct: dict[str, ChildDispositionOverride] = {}
    for record in overrides:
        distinct.setdefault(record.digest, record)
    if len(distinct) > 1:
        raise AgentLoopError(
            f"Human decision required: stage `{stage_label}` has {len(distinct)} distinct signed "
            "child-disposition override records ("
            + "; ".join(
                f"{record.comment_locator} -> {record.disposition}" for record in distinct.values()
            )
            + "). Comment order never selects one; remove the superseded record(s) before rerunning."
        )
    override = next(iter(distinct.values()), None)
    if override is not None:
        effective, digest, origin = override.disposition, override.digest, "override"
    elif topology_source != EXECUTION_TOPOLOGY_SOURCE:
        effective, digest, origin = EXECUTION_DISPOSITION_PLANNING, None, "unsupported-source"
    elif phase_disposition is None:
        effective, digest, origin = EXECUTION_DISPOSITION_PLANNING, None, "legacy-ambiguous"
    elif phase_disposition in {EXECUTION_DISPOSITION_DIRECT, EXECUTION_DISPOSITION_PLANNING}:
        effective, digest, origin = phase_disposition, None, "phase"
    else:
        raise AgentLoopError(
            f"Human decision required: stage `{stage_label}` is an agent-pr stage whose persisted "
            f"disposition `{phase_disposition}` is contradictory; repair the approved topology."
        )
    if effective == EXECUTION_DISPOSITION_DIRECT:
        problems = phase_direct_readiness_problems(phase)
        if problems:
            requested_by = (
                f"the signed override at {override.comment_locator}"
                if override is not None else "the persisted approved phase"
            )
            raise AgentLoopError(
                f"Human decision required: {requested_by} selects direct-implementation for stage "
                f"`{stage_label}`, but the reviewed parent stage alone is not direct-ready: "
                + "; ".join(problems)
                + ". Override records never supply readiness evidence; revise the parent plan "
                "or route the child to planning."
            )
    return ChildExecutionRoute(effective, digest, origin)


def _fresh_phase_marker_payload(issue_context: IssueContext) -> dict[str, object] | None:
    """Return the fresh approved-plan-v1 phase identity payload carried by an issue."""
    bodies = [issue_context.body or ""]
    bodies.extend(comment.body or "" for comment in issue_context.comments)
    payloads: list[dict[str, object]] = []
    for body in bodies:
        for match in PHASE_IDENTITY_MARKER_RE.finditer(body):
            payload = _decode_json_payload(
                match.group("payload"), marker_name="AGENT_PLAN_PHASE_IDENTITY"
            )
            if payload.get("source") == EXECUTION_TOPOLOGY_SOURCE:
                payloads.append(payload)
    if not payloads:
        return None
    if any(payload != payloads[0] for payload in payloads[1:]):
        raise AgentLoopError(
            f"Issue #{issue_context.number} carries conflicting fresh phase identities; "
            "repair the child issue provenance before rerunning."
        )
    return payloads[0]


@dataclass(frozen=True)
class _FreshChildProvenance:
    """A materialized fresh decomposition child resolved for direct entry."""

    parent_issue: int
    plan_hash: str
    plan_subject: str
    approved_plan: str
    parent_plan_context: ApprovedPlanContext
    recommendation: ExecutionStrategyRecommendation
    decomposition: PlanDecomposition
    created: CreatedPhaseIssue
    phase_index: int
    stage_id: str
    handoff: PhaseImplementationHandoffMetadata | None
    overrides: tuple[ChildDispositionOverride, ...]
    route: ChildExecutionRoute


def _resolve_fresh_child_provenance(
    *,
    issue_context: IssueContext,
    parent_issue_context: IssueContext | None,
) -> _FreshChildProvenance | None:
    """Bind a directly entered child to its approved parent phase and route it.

    Returns ``None`` for issues without fresh approved-plan-v1 phase provenance
    (split-materialized, hand-created, and legacy children keep today's
    behavior).  Any broken fresh provenance fails closed.
    """
    payload = _fresh_phase_marker_payload(issue_context)
    if payload is None:
        return None
    parent_issue = payload.get("parent_issue")
    plan_hash = payload.get("plan_hash")
    phase_index = payload.get("phase_index")
    stage_id = payload.get("stage_id")
    digest = payload.get("recommendation_digest")
    if (
        not isinstance(parent_issue, int) or isinstance(parent_issue, bool)
        or not isinstance(plan_hash, str) or not plan_hash
        or not isinstance(phase_index, int) or isinstance(phase_index, bool) or phase_index < 1
        or not isinstance(stage_id, str) or not stage_id.strip()
        or not isinstance(digest, str) or not digest
        or payload.get("strategy") != "staged"
        or payload.get("execution_strategy_contract_version") != 1
    ):
        raise AgentLoopError(
            f"Issue #{issue_context.number} carries an invalid fresh phase identity; "
            "repair the child issue provenance before rerunning."
        )
    if parent_issue_context is None or parent_issue_context.number != parent_issue:
        raise AgentLoopError(
            f"Issue #{issue_context.number} names parent #{parent_issue} in its fresh phase "
            "identity, but that parent context could not be resolved."
        )
    parent_plan_context = recover_approved_plan_context(
        parent_issue_context.comments, expected_hash=plan_hash
    )
    if not parent_plan_context.is_available or not parent_plan_context.canonical_text:
        raise AgentLoopError(
            f"Issue #{issue_context.number} is a fresh decomposition child of #{parent_issue}, "
            f"but the approved parent plan {plan_hash} could not be recovered: "
            f"{parent_plan_context.diagnostic or 'no diagnostic available'}"
        )
    approved_plan = parent_plan_context.canonical_text
    plan_subject = parent_plan_context.plan_subject or _plan_subject(approved_plan)
    recommendation = recover_execution_recommendation(
        parent_issue_context.comments, expected_digest=digest
    )
    decomposition, _retained = normalize_execution_recommendation(
        recommendation, approved_plan=approved_plan, plan_subject=plan_subject
    )
    if phase_index > len(decomposition.phases):
        raise AgentLoopError(
            f"Issue #{issue_context.number} references phase {phase_index}, which is outside "
            "the approved parent topology."
        )
    phase = decomposition.phases[phase_index - 1]
    expected_identity = phase_identity(
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        topology_source=EXECUTION_TOPOLOGY_SOURCE,
        phase_index=phase_index,
        phase=phase,
        stage_id=phase.stage_id,
        execution_strategy_contract_version=1,
    )
    if (
        phase.stage_id != stage_id
        or payload.get("identity") != expected_identity
        or digest != decomposition.recommendation_digest
    ):
        raise AgentLoopError(
            f"Issue #{issue_context.number} fresh phase identity does not match the approved "
            "parent topology; repair the child issue provenance before rerunning."
        )
    summary = find_existing_decomposition(
        parent_issue_context.comments,
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        strategy="staged",
        topology_source=EXECUTION_TOPOLOGY_SOURCE,
        recommendation_digest=decomposition.recommendation_digest,
        plan_subject=plan_subject,
    )
    if summary is None or summary.mode not in {"decompose-only", "implement-by-phase"}:
        raise AgentLoopError(
            f"Issue #{issue_context.number} has no matching canonical parent topology summary "
            f"on #{parent_issue}; repair the decomposition record before rerunning."
        )
    handoffs = tuple(
        handoff
        for handoff in find_phase_implementation_handoffs_for_parent(
            parent_issue_context.comments, parent_issue=parent_issue
        )
        if handoff.plan_hash == plan_hash and (
            handoff.phase_index == phase_index
            or handoff.stage_id == stage_id
            or handoff.child_issue_number == issue_context.number
        )
    )
    if len(handoffs) > 1:
        raise AgentLoopError(
            f"Issue #{issue_context.number} has multiple parent phase handoffs; repair the "
            "handoff provenance before rerunning."
        )
    handoff = handoffs[0] if handoffs else None
    if handoff is not None and (
        handoff.child_issue_number != issue_context.number
        or handoff.phase_index != phase_index
        or handoff.stage_id != stage_id
        or handoff.mode != "implement-by-phase"
        or handoff.topology_source != EXECUTION_TOPOLOGY_SOURCE
        or handoff.recommendation_digest != decomposition.recommendation_digest
    ):
        raise AgentLoopError(
            f"Issue #{issue_context.number} phase handoff disagrees with the approved parent "
            "topology; repair the handoff before rerunning."
        )
    overrides = collect_child_disposition_overrides(
        parent_comments=parent_issue_context.comments,
        child_comments=issue_context.comments,
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        topology_stage_ids=tuple(item.stage_id or "" for item in decomposition.phases),
        routed_stage_id=stage_id,
        child_stage_id=stage_id,
        child_issue_number=issue_context.number,
    )
    route = resolve_child_execution_route(
        phase,
        topology_source=EXECUTION_TOPOLOGY_SOURCE,
        recorded_handoff=handoff,
        overrides=overrides,
    )
    return _FreshChildProvenance(
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        plan_subject=plan_subject,
        approved_plan=approved_plan,
        parent_plan_context=parent_plan_context,
        recommendation=recommendation,
        decomposition=decomposition,
        created=CreatedPhaseIssue(
            phase=phase,
            issue_url=issue_context.url,
            issue_number=issue_context.number,
            origin="adopted",
        ),
        phase_index=phase_index,
        stage_id=stage_id,
        handoff=handoff,
        overrides=overrides,
        route=route,
    )


def _child_resume_hint(child_issue_number: int, disposition: str) -> str:
    if disposition == EXECUTION_DISPOSITION_PLANNING:
        return f"agent-loop issue {child_issue_number} --plan-first --plan-execution-mode auto"
    return f"agent-loop issue {child_issue_number}"


def _projection_may_carry_planning_record(comments: Sequence[object]) -> bool:
    """Whether the issue-view projection could hide or show planning state.

    A cheap, unauthenticated pre-check: it only decides whether the complete,
    author-authenticated history must be read.  The ``gh issue view``
    projection is capped, so a full projection may have dropped older
    records and always triggers the authenticated read.
    """
    if len(comments) >= _REST_ISSUE_COMMENT_PAGE_SIZE:
        return True
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        if PLAN_APPROVED_FOLLOWUP_MARKER_RE.search(body) or EXECUTION_DECISION_MARKER_RE.search(body):
            return True
        for match in ROUND_RESUME_MARKER_RE.finditer(body):
            try:
                flow = decode_mapping(match.group("payload")).get("flow")
            except Exception:  # noqa: BLE001 - malformed text only widens the read
                return True
            if flow == "plan":
                return True
        # An approved-plan handoff can be the only planning record left in
        # view; a direct-flow handoff is ordinary resume state and is not.
        for match in AGENT_ISSUE_PR_HANDOFF_RE.finditer(body):
            try:
                handoff = decode_issue_pr_handoff_record(match.group("payload"))
            except Exception:  # noqa: BLE001 - malformed text only widens the read
                return True
            if handoff.flow == "approved-plan-implementation":
                return True
    return False


def _refuse_plain_mode_over_planning(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    projection_comments: Sequence[object],
) -> None:
    """Refuse plain issue mode on an issue that planning already decided (#1088).

    Plain mode implements from the issue text and never consults an approved
    plan, so running it over one would silently discard the reviewed plan.
    This mirrors the decomposition-child guard for a top-level issue.

    Only records authored by the authenticated agent-loop actor count, read
    from the complete REST history so the projection's connection cap cannot
    hide them; a foreign comment can neither fabricate nor mask a decision.
    """
    if not _projection_may_carry_planning_record(projection_comments):
        return
    login, actor_id = resolve_authenticated_github_actor(runner, config=config)
    comments = tuple(
        comment
        for comment in read_rest_issue_comments(
            runner,
            config=config,
            issue_number=issue_number,
            purpose="plain issue mode cannot confirm the issue has no approved plan",
        )
        if comment.author_id == actor_id
        and strip_bot_login_suffix(comment.author) == strip_bot_login_suffix(login)
    )
    evidence: list[str] = []
    try:
        plan_records = _extract_round_metadata_records(comments, flow="plan")
    except AgentLoopError as exc:
        raise AgentLoopError(
            f"Issue #{issue_number} carries planning round records that cannot be read ({exc}); "
            "plain issue mode will not implement over them. Repair the records, or rerun "
            f"`agent-loop issue {issue_number} --plan-first`."
        ) from exc
    if any(
        record.metadata.role == "reviewer" and record.metadata.state == "approved"
        for record in plan_records
    ):
        evidence.append("an approved planning review")
    plan_hashes = approved_plan_hashes_for_issue(comments, issue_number=issue_number)
    if plan_hashes:
        evidence.append(f"an approved plan (hash {plan_hashes[-1]})")
    # Our own decision records on this thread are posted to their parent
    # issue, so an undecodable one still belongs here and fails closed.
    if issue_has_execution_decision(comments, issue_number=issue_number):
        evidence.append("a recorded execution decision")
    # Every handoff counts, not only the latest: a later direct-flow handoff
    # to another PR does not retire the approved plan an earlier one bound.
    approved_handoff_prs: list[int] = []
    unreadable_handoff = False
    for comment in comments:
        for match in AGENT_ISSUE_PR_HANDOFF_RE.finditer(comment.body or ""):
            try:
                handoff = decode_issue_pr_handoff_record(match.group("payload"))
            except AgentLoopError:
                unreadable_handoff = True
                continue
            if (
                handoff.issue_number == issue_number
                and handoff.flow == "approved-plan-implementation"
                and handoff.pr_number not in approved_handoff_prs
            ):
                approved_handoff_prs.append(handoff.pr_number)
    if approved_handoff_prs:
        evidence.append(
            "an approved-plan implementation handoff to "
            + ", ".join(f"PR #{number}" for number in approved_handoff_prs)
        )
    if unreadable_handoff:
        evidence.append("an implementation handoff record that cannot be read")
    if not evidence:
        return
    raise AgentLoopError(
        f"Issue #{issue_number} already carries {', '.join(evidence)}; plain issue mode "
        "implements from the issue text and would bypass the reviewed plan. Rerun "
        f"`agent-loop issue {issue_number} --plan-first` to resume from the approved plan."
    )


def _post_child_planning_handoff(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    plan_hash: str,
    plan_subject: str,
    phase_index: int,
    created: CreatedPhaseIssue,
    recommendation: ExecutionStrategyRecommendation,
    inherited_matrix_row_ids: Sequence[str],
    override_digest: str | None,
) -> None:
    """Record the requires-child-planning handoff before any planning agent runs.

    Both entry paths (parent first-child dispatch and direct child entry) post
    this identical idempotent record with the same identity and override
    digest rules, so a rerun at any later stage finds it and posts nothing.
    """
    post_phase_implementation_handoff_comment(
        runner,
        config=config,
        parent_issue=parent_issue,
        mode="implement-by-phase",
        plan_hash=plan_hash,
        phase_index=phase_index,
        created=created,
        strategy=recommendation.strategy,
        topology_source=EXECUTION_TOPOLOGY_SOURCE,
        execution_strategy_contract_version=1,
        recommendation_digest=str(recommendation.identity()["recommendation_sha256"]),
        plan_subject=plan_subject,
        inherited_matrix_row_ids=inherited_matrix_row_ids,
        execution_disposition=EXECUTION_DISPOSITION_PLANNING,
        override_digest=override_digest,
    )


def _print_staged_phase_progress(
    issue_number: int, progress: Sequence[PhaseProgress]
) -> None:
    """Replace the static remaining-topology line with resolved progress."""
    print(f"Issue #{issue_number} staged child work:")
    for phase in progress:
        print(f"  {phase.phase_index}. {render_phase_status_line(phase)}")


def _print_parent_obligation(label: str, allocation) -> None:
    status = getattr(allocation, "status", None) or "none"
    if status == "none":
        print(f"{label}: none.")
        return
    deliverables = ", ".join(getattr(allocation, "deliverables", ()) or ()) or "none"
    criteria = ", ".join(getattr(allocation, "acceptance_criteria", ()) or ()) or "none"
    print(
        f"{label}: {status}; this is operator-owned parent work. "
        f"Deliverables: {deliverables}; acceptance criteria: {criteria}."
    )


def _print_staged_terminal_report(
    *,
    issue_number: int,
    progress: Sequence[PhaseProgress],
    outcome: StagedTopologyOutcome,
) -> None:
    """Report the terminal state of a fully delivered staged topology.

    Final-integration work is deliberately not implemented here and the parent
    is deliberately not closed: both remain operator decisions.
    """
    print(
        f"Issue #{issue_number}: all {len(progress)} staged phases are delivered; "
        "no child is dispatched and no handoff is recorded."
    )
    for phase in progress:
        if phase.is_human:
            print(
                f"  {phase.phase_index}. {phase.stage_id}: delivered by human work on child "
                f"issue #{phase.child_issue_number}; its closure is the operator attestation."
            )
        else:
            print(
                f"  {phase.phase_index}. {phase.stage_id}: delivered by child issue "
                f"#{phase.child_issue_number} with merged PR #{phase.pr_number}."
            )
    _print_parent_obligation("Retained-parent obligations", outcome.retained_parent_scope)
    _print_parent_obligation("Final-integration obligations", outcome.final_integration_work)
    if outcome.retained_parent_status == "none" and outcome.final_integration_status == "none":
        print(
            f"No parent-side work remains for issue #{issue_number}; it is left open for the "
            "operator to close."
        )
    else:
        print(
            f"Issue #{issue_number} remains open pending that operator-owned parent work; "
            "agent-loop neither implements it nor closes the parent."
        )


def _recorded_staged_outcome_for_child(
    parent_comments: Sequence[object],
    *,
    parent_issue: int,
    child_issue: int,
) -> StagedTopologyOutcome | None:
    """Rebuild a parent's staged topology from its records, seen from one child.

    Returns ``None`` when the parent holds no phase handoff for ``child_issue``
    or no decomposition summary for that handoff's plan identity: the child
    then is not a dispatched phase of a recorded staged topology.
    """
    identities = {
        (handoff.plan_hash, handoff.mode)
        for handoff in find_phase_implementation_handoffs_for_parent(
            parent_comments, parent_issue=parent_issue
        )
        if handoff.child_issue_number == child_issue
    }
    if not identities:
        return None
    if len(identities) > 1:
        raise AgentLoopError(
            f"Issue #{parent_issue} records phase handoffs for child issue #{child_issue} "
            "under more than one plan identity."
        )
    plan_hash, mode = identities.pop()
    existing = find_existing_decomposition(
        parent_comments, parent_issue=parent_issue, plan_hash=plan_hash, mode=mode
    )
    if existing is None:
        return None
    adopted = tuple(
        CreatedPhaseIssue(
            phase=RecordedPhase(title=title, automation=automation),
            issue_url=url,
            issue_number=number,
        )
        for (title, url, number), automation in zip(
            existing.children, existing.automation, strict=False
        )
    )
    return StagedTopologyOutcome(
        created=adopted,
        stage_ids=_resolved_stage_ids(adopted, recorded_stage_ids=existing.stage_ids),
        automations=tuple(item.phase.automation for item in adopted),
        plan_hash=plan_hash,
        mode=mode,
        topology_source=existing.topology_source,
        retained_parent_scope=existing.retained_parent_scope,
        final_integration_work=existing.final_integration_work,
    )


def _record_staged_parent_completion_after_merge(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_context: IssueContext | None,
    pr_number: int,
) -> None:
    """Write the parent's completion record when this merge delivered its last phase.

    The child run that merges the final stage is the one run that knows the
    decomposition just finished (#1018).  It is best effort: the PR is already
    merged, so a parent that cannot be authenticated here is reported with the
    rerun that records it, never raised.
    """
    if issue_context is None or config.dry_run:
        return
    # GitHub closes a child only for a default-branch merge (#1285).
    if not require_child_closed_or_report(
        runner, config=config, issue_context=issue_context, pr_number=pr_number
    ):
        return
    parent_issue: int | None = None
    try:
        parent_issue = _infer_staged_parent_issue(issue_context)
        if parent_issue is None:
            return
        parent_context = get_issue_context(runner, config=config, issue_number=parent_issue)
        outcome = _recorded_staged_outcome_for_child(
            parent_context.comments,
            parent_issue=parent_issue,
            child_issue=issue_context.number,
        )
        if outcome is None:
            return
        progress = resolve_staged_phase_progress(
            runner,
            config=config,
            parent_issue=parent_issue,
            parent_comments=parent_context.comments,
            outcome=outcome,
            just_merged=(issue_context.number, pr_number),
        )
        if select_current_phase(progress) is not None:
            return
        if record_staged_completion(
            runner,
            config=config,
            parent_issue=parent_issue,
            progress=progress,
            outcome=outcome,
        ):
            print(
                f"PR #{pr_number} delivered the last staged phase of issue #{parent_issue}; "
                "recorded the staged completion on the parent."
            )
    except AgentLoopError as exc:
        if parent_issue is None:
            log(config, f"Staged parent completion check skipped: {exc}")
            return
        log(config, f"Staged parent #{parent_issue} completion record not written: {exc}")
        print(
            f"Staged parent issue #{parent_issue}: no completion record was written ({exc}). "
            "Rerunning the parent's staged plan records it once every phase is delivered "
            "and its evidence is readable."
        )


def _print_dry_run_execution_preview(
    *,
    issue_number: int,
    resolved: ResolvedExecution,
    normalized_topology,
) -> None:
    """Render an approval-bound preview without entering a mutation branch."""
    print(
        f"Issue #{issue_number} dry-run preview: requested policy "
        f"`{resolved.requested_policy}` resolves to `{resolved.action}`."
    )
    if resolved.action == "implement-one-shot":
        print(
            "One-shot implementation would be selected; no approval record, follow-up, "
            "handoff, coder, or PR work was performed."
        )
        return
    if resolved.action != "implement-by-phase" or normalized_topology is None:
        print("No implementation dispatch is selected by this preview.")
        return

    decomposition, retained_parent_scope = normalized_topology
    print("Approved child topology (preview only; no child issues will be created):")
    for phase in decomposition.phases:
        dependencies = ", ".join(phase.depends_on_stage_ids) or "none"
        covered = ", ".join(phase.covered_scope_item_ids) or "none"
        print(
            f"{phase.position}. {phase.stage_id}: {phase.title} "
            f"[{phase.automation}]; depends on: {dependencies}; covers: {covered}"
        )
        declared = phase.execution_disposition or "legacy-ambiguous"
        try:
            route_text = resolve_child_execution_route(
                phase, topology_source=EXECUTION_TOPOLOGY_SOURCE
            ).disposition
        except AgentLoopError as exc:
            route_text = f"human decision required ({exc})"
        print(f"   declared disposition: {declared}; resolved route: {route_text}")
    first = decomposition.phases[0] if decomposition.phases else None
    if first is None:
        print("No first phase is available; dispatch is skipped.")
    elif first.automation == "agent-pr":
        print(
            f"First-phase dispatch for `{first.stage_id}` is skipped in dry-run; "
            "the parent remains open."
        )
    else:
        print(
            f"First-phase dispatch is skipped because `{first.stage_id}` requires "
            f"{first.automation}; the parent remains open."
        )
    remaining = [phase.stage_id for phase in decomposition.phases[1:]]
    print(f"Remaining child work: {', '.join(remaining) or 'none'}.")
    print(
        "Retained-parent obligations: "
        f"{retained_parent_scope.status}; final-integration obligations: "
        f"{decomposition.final_integration_work.status}."
    )


def _print_execution_resolution_summary(
    *,
    issue_number: int,
    resolved: ResolvedExecution,
    normalized_topology,
    defer_child_work: bool = False,
) -> None:
    """Report the approval-bound action and the work it leaves behind.

    ``defer_child_work`` suppresses only the static ``Remaining child work:``
    line, and only on the one path that later prints a progress-derived
    replacement: this summary runs before decomposition returns, with no
    runner, config, or child comments, so it cannot resolve progress itself.
    The policy and obligation lines are unaffected and remain true.
    """
    print(
        f"Issue #{issue_number}: requested policy `{resolved.requested_policy}`; "
        f"resolved action `{resolved.action}`."
    )
    if normalized_topology is None:
        if not defer_child_work:
            print("Remaining child work: none.")
        recommendation = resolved.recommendation
        retained_status = (
            recommendation.retained_parent_work.status
            if recommendation is not None
            else "none"
        )
        final_status = (
            recommendation.final_integration_work.status
            if recommendation is not None
            else "none"
        )
        print(
            "Retained-parent obligations: "
            f"{retained_status}; final-integration obligations: {final_status}."
        )
        return
    decomposition, retained_parent_scope = normalized_topology
    if not defer_child_work:
        remaining = [phase.stage_id or str(phase.position) for phase in decomposition.phases]
        print(f"Remaining child work: {', '.join(remaining) or 'none'}.")
    print(
        "Retained-parent obligations: "
        f"{retained_parent_scope.status}; final-integration obligations: "
        f"{decomposition.final_integration_work.status}."
    )


def _persist_execution_decision_if_needed(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    current_plan: str,
    issue_comments: Sequence[object],
    recommendation,
    requested_policy: str,
    resolved_execution: ResolvedExecution | None = None,
    retired_plan_hashes: frozenset[str] = frozenset(),
    retires_plan_hashes: tuple[str, ...] = (),
) -> None:
    if resolved_execution is not None:
        recommendation = resolved_execution.recommendation
        requested_policy = resolved_execution.requested_policy
        current_action = resolved_execution.action
    else:
        current_action = requested_policy
    if recommendation is None or requested_policy == "plan-only" or config.dry_run:
        return
    plan_hash = approved_plan_hash(current_plan)
    plan_subject = _plan_subject(current_plan)
    identity = recommendation.identity()
    decision = ExecutionDecision(
        parent_issue=issue_number,
        plan_hash=plan_hash,
        plan_subject=plan_subject,
        execution_strategy_contract_version=1,
        strategy=recommendation.strategy,
        topology_source=str(identity["topology_source"]),
        recommendation_digest=str(identity["recommendation_sha256"]),
        requested_policy=requested_policy,
        current_action=current_action,
        stage_ids=tuple(stage.stage_id for stage in recommendation.child_stages),
        scope_item_ids=tuple(item.scope_item_id for item in recommendation.scope_items),
        retained_parent_status=recommendation.retained_parent_work.status,
        final_integration_status=recommendation.final_integration_work.status,
        architecture_identity=(
            config.architecture_context.identity()
            if hasattr(config.architecture_context, "identity") else None
        ),
        retires_plan_hashes=retires_plan_hashes,
    )
    existing = find_existing_execution_decision(
        issue_comments,
        parent_issue=issue_number,
        plan_hash=plan_hash,
        plan_subject=plan_subject,
        strategy=recommendation.strategy,
        recommendation_digest=str(identity["recommendation_sha256"]),
        retired_plan_hashes=retired_plan_hashes | frozenset(retires_plan_hashes),
    )
    # A supersession must be recorded even over an equal existing decision
    # (a plan re-approved back to an earlier hash), or the retirement would
    # have to be re-derived on every later run.
    if existing is None or retires_plan_hashes:
        post_execution_decision(runner, config=config, decision=decision)


_REALIZATION_LISTING_LIMIT = 100000


def _strict_gh_listing(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    args: Sequence[str],
) -> list[dict] | None:
    """A complete ``gh ... list --json`` result, or ``None`` when unreadable.

    A failed command, output that is not a JSON list of objects, or a result
    that fills the limit (possibly truncated) is ``None``.
    """
    result = runner.run(
        [config.gh_cmd, *args], cwd=active_workdir(config), check=False
    )
    if result.returncode != 0:
        return None
    try:
        items = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        return None
    if len(items) >= _REALIZATION_LISTING_LIMIT:
        return None
    return items


def _execution_decision_realization_evidence(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    comments: Sequence[object],
) -> list[str]:
    """What, if anything, has acted on an execution decision for this parent (#1087).

    A decision only forks a topology once something depends on it: a PR
    handoff, an implementation or phase handoff, a decomposition record, a
    child issue, a PR in any state closing the parent or on the reserved
    managed branch, or that branch.  Unreadable, incomplete, or unavailable
    state counts as evidence, so the caller fails closed.
    """
    evidence: list[str] = []
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in AGENT_ISSUE_PR_HANDOFF_RE.finditer(body):
            try:
                handoff = decode_issue_pr_handoff_record(match.group("payload"))
            except AgentLoopError:
                evidence.append("an unreadable issue-to-PR handoff record")
                continue
            if handoff.issue_number == issue_number:
                evidence.append(f"an issue-to-PR handoff to PR #{handoff.pr_number}")
        for marker in SPLIT_CHILD_MARKER_RE.finditer(body):
            if int(marker.group("parent")) == issue_number:
                evidence.append("a split child record")
    if find_decompositions_for_parent(comments, parent_issue=issue_number):
        evidence.append("a decomposition summary")
    if find_topology_checkpoints_for_parent(comments, parent_issue=issue_number):
        evidence.append("a topology checkpoint")
    if find_phase_implementation_handoffs_for_parent(comments, parent_issue=issue_number):
        evidence.append("a phase implementation handoff")
    if find_one_shot_impl_handoffs(
        comments, parent_issue=issue_number, mode="implement-one-shot"
    ):
        evidence.append("a one-shot implementation handoff")
    materialization = find_existing_split_materialization(comments, parent_issue=issue_number)
    if materialization is not None and materialization.children:
        evidence.append("a split materialization")
    if config.dry_run:
        # Remote inventory is not read in a dry run, which publishes nothing.
        return list(dict.fromkeys(evidence))
    # Both inventories are read strictly: an unreadable or possibly truncated
    # listing is evidence, never proof of absence (`search_issues` maps bad
    # output to an empty result, which is right for adoption, not here).
    for query in parent_child_search_queries(issue_number):
        children = _strict_gh_listing(
            runner,
            config=config,
            args=[
                "issue", "list", "--repo", config.repo, "--search", query,
                "--state", "all", "--limit", str(_REALIZATION_LISTING_LIMIT),
                "--json", "number,title",
            ],
        )
        if children is None:
            evidence.append(f"child issues that could not be listed ({query})")
            continue
        for candidate in children:
            evidence.append(f"child issue #{candidate.get('number')}")
    managed_branch = f"agent-loop/managed-{issue_number}"
    # Closed PRs count too: a PR opened before its handoff record was posted,
    # then closed with its branch deleted, still acted on the decision.
    pr_items = _strict_gh_listing(
        runner,
        config=config,
        args=[
            "pr", "list", "--repo", config.repo, "--state", "all",
            "--json", "number,body,headRefName", "--limit", str(_REALIZATION_LISTING_LIMIT),
        ],
    )
    if pr_items is None:
        evidence.append("pull requests that could not be listed")
        pr_items = []
    for item in pr_items:
        if item.get("headRefName") == managed_branch or parse_strong_issue_reference_evidence(
            str(item.get("body") or ""), repo=config.repo, issue_number=issue_number
        ):
            evidence.append(f"PR #{item.get('number')}")
    # The branch name contains `/`, which the branches endpoint only accepts
    # encoded; an unencoded path 404s even when the branch exists.
    branch = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/branches/{urllib.parse.quote(managed_branch, safe='')}",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    branch_error = branch.stderr or ""
    if branch.returncode == 0:
        evidence.append(f"branch `{managed_branch}`")
    elif "404" not in branch_error and "Not Found" not in branch_error:
        evidence.append(f"branch `{managed_branch}`, whose existence could not be checked")
    return list(dict.fromkeys(evidence))


def _retirable_execution_decision_hashes(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    comments: Sequence[object],
    plan_hash: str,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Resolve the decisions a plan-first re-approval supersedes (#1087).

    Returns ``(retired, newly_retired)``: the hashes the preflight may skip
    and the ones the next published decision records.  Only *live* records
    count (``live_execution_decisions``): supersession is per record, so a
    plan re-approved back to a retired hash yields a live record that still
    needs the realization check before a later plan replaces it.

    A decision under another hash that something has already acted on still
    fails closed: replacing it would fork the parent's topology.
    """
    stale = tuple(dict.fromkeys(
        decision.plan_hash
        for decision in live_execution_decisions(comments, parent_issue=issue_number)
        if decision.plan_hash != plan_hash
    ))
    if not stale:
        return frozenset(), ()
    evidence = _execution_decision_realization_evidence(
        runner, config=config, issue_number=issue_number, comments=comments
    )
    if evidence:
        raise AgentLoopError(
            "Conflicting execution decision exists for this parent under a different "
            f"approved plan hash ({', '.join(stale)} vs {plan_hash}) and has already been "
            f"acted on ({'; '.join(evidence)}). Resume or close that work before "
            "publishing a decision for the re-approved plan."
        )
    log(
        config,
        f"Issue #{issue_number}: superseding unrealized execution decision(s) under plan "
        f"hash {', '.join(stale)}; nothing has acted on them.",
    )
    return frozenset(stale), stale


def _preflight_fresh_staged_topology(
    runner: Runner,
    *,
    issue_number: int,
    approved_plan: str,
    config: AgentLoopConfig,
    issue_context: IssueContext,
    mode: str,
    normalized_topology,
    retired_plan_hashes: frozenset[str] = frozenset(),
) -> tuple[CreatedPhaseIssue, ...] | NeedsHumanDecision:
    """Validate fresh staged recovery without publishing or creating anything."""
    decomposition, retained_parent_scope = normalized_topology
    plan_hash = approved_plan_hash(approved_plan)
    plan_subject = _plan_subject(approved_plan)
    matrix_context = make_approved_plan_context(
        approved_plan,
        source_locator=f"issue #{issue_number} approved-plan topology",
        expected_hash=plan_hash,
        expected_subject=plan_subject,
    )
    # Reconcile any already-published approval-bound decision before the
    # summary/child/handoff inventory below.  The finder also scans decisions
    # for older plan hashes, preventing a changed approved plan from creating a
    # second canonical decision on the same parent.
    find_existing_execution_decision(
        issue_context.comments,
        parent_issue=issue_number,
        plan_hash=plan_hash,
        plan_subject=plan_subject,
        strategy="staged",
        recommendation_digest=decomposition.recommendation_digest,
        retired_plan_hashes=retired_plan_hashes,
    )
    existing_pr = resolve_canonical_pr_for_issue(
        runner,
        config=config,
        issue_number=issue_number,
        issue_context=issue_context,
        expected_fallback_scope=IssuePrProvenanceScope(
            repository=config.repo,
            issue_number=issue_number,
            flow="approved",
            approved_plan_hash=plan_hash,
        ),
    )
    if existing_pr is not None:
        raise AgentLoopError(
            "Fresh staged execution conflicts with an existing implementation PR "
            f"#{existing_pr.pr_number}; resume that one-shot implementation or repair "
            "the staged topology before rerunning."
        )
    reject_legacy_topology_collision(
        issue_context.comments,
        parent_issue=issue_number,
        plan_hash=plan_hash,
    )
    parent_summaries = find_decompositions_for_parent(
        issue_context.comments, parent_issue=issue_number
    )
    if any(summary.plan_hash != plan_hash for summary in parent_summaries):
        recorded = next(summary for summary in parent_summaries if summary.plan_hash != plan_hash)
        raise AgentLoopError(
            "Fresh staged execution conflicts with an existing decomposition summary for "
            f"plan {recorded.plan_hash}; repair or resume the recorded topology before rerunning."
        )
    if any(summary != parent_summaries[0] for summary in parent_summaries[1:]):
        raise AgentLoopError(
            "Ambiguous staged recovery: multiple divergent decomposition summaries exist for the parent."
        )
    existing_summary = find_existing_decomposition(
        issue_context.comments,
        parent_issue=issue_number,
        plan_hash=plan_hash,
    )
    parent_checkpoints = find_topology_checkpoints_for_parent(
        issue_context.comments, parent_issue=issue_number
    )
    for checkpoint in parent_checkpoints:
        if checkpoint.plan_hash != plan_hash:
            raise AgentLoopError(
                "Fresh staged execution conflicts with an existing topology checkpoint for "
                f"plan {checkpoint.plan_hash}; repair or resume the recorded topology before rerunning."
            )
        if checkpoint.mode not in {"decompose-only", "implement-by-phase"}:
            raise AgentLoopError(
                "Fresh staged execution conflicts with a topology checkpoint using unsupported "
                f"mode `{checkpoint.mode}`; repair the recorded topology before rerunning."
            )
    for checkpoint_mode in ("decompose-only", "implement-by-phase"):
        # Preserve the existing same-plan divergence check while the parent
        # inventory above catches records hidden under older plan hashes.
        find_existing_topology_checkpoint(
            issue_context.comments,
            parent_issue=issue_number,
            plan_hash=plan_hash,
            mode=checkpoint_mode,
        )
    if existing_summary is not None and (
        existing_summary.topology_source != EXECUTION_TOPOLOGY_SOURCE
        or existing_summary.strategy != "staged"
        or existing_summary.recommendation_digest != decomposition.recommendation_digest
        or existing_summary.plan_subject != plan_subject
    ):
        raise AgentLoopError(
            "Fresh staged execution conflicts with an existing decomposition summary; "
            "repair the recorded topology identity before rerunning."
        )
    for checkpoint_mode in ("decompose-only", "implement-by-phase"):
        checkpoint = find_existing_topology_checkpoint(
            issue_context.comments,
            parent_issue=issue_number,
            plan_hash=plan_hash,
            mode=checkpoint_mode,
        )
        if checkpoint is not None and (
            checkpoint.topology_source != EXECUTION_TOPOLOGY_SOURCE
            or checkpoint.strategy != "staged"
            or checkpoint.recommendation_digest != decomposition.recommendation_digest
        ):
            raise AgentLoopError(
                "Fresh staged execution conflicts with an existing topology checkpoint; "
                "repair the recorded topology identity before rerunning."
            )
    if find_latest_one_shot_impl_handoff(
        issue_context.comments,
        parent_issue=issue_number,
        mode="implement-one-shot",
    ) is not None:
        raise AgentLoopError(
            "Fresh staged execution conflicts with an existing one-shot handoff; "
            "resume the recorded implementation or repair the conflicting topology first."
        )
    preflight_children = create_decomposition_child_issues(
        runner,
        config=config,
        parent_issue=issue_number,
        approved_plan=approved_plan,
        decomposition=decomposition,
        topology_source=EXECUTION_TOPOLOGY_SOURCE,
        issue_comments=issue_context.comments,
        mode=mode,
        retained_parent_scope=retained_parent_scope,
        strategy=decomposition.strategy,
        execution_strategy_contract_version=decomposition.execution_strategy_contract_version,
        recommendation_digest=decomposition.recommendation_digest,
        plan_subject=plan_subject,
        preflight_only=True,
        risk_test_matrix=(
            matrix_context.risk_test_matrix_payload
            if matrix_context.matrix_available else None
        ),
    )
    if isinstance(preflight_children, NeedsHumanDecision):
        return preflight_children

    expected_children = tuple(
        (item.phase.title, item.issue_url, item.issue_number)
        for item in preflight_children
    )
    if existing_summary is not None:
        expected_phases = decomposition.phases
        expected_stage_ids = tuple(phase.stage_id or str(index) for index, phase in enumerate(expected_phases, 1))
        expected_phase_identities = tuple(
            phase_identity(
                parent_issue=issue_number,
                plan_hash=plan_hash,
                topology_source=EXECUTION_TOPOLOGY_SOURCE,
                phase_index=index,
                phase=phase,
                stage_id=phase.stage_id,
                execution_strategy_contract_version=decomposition.execution_strategy_contract_version,
            )
            for index, phase in enumerate(expected_phases, 1)
        )
        if (
            existing_summary.phase_count != len(expected_phases)
            or existing_summary.phase_titles != tuple(phase.title for phase in expected_phases)
            or existing_summary.automation != tuple(phase.automation for phase in expected_phases)
            or existing_summary.stage_ids != expected_stage_ids
            or existing_summary.phase_identities != expected_phase_identities
            or existing_summary.children != expected_children
            or existing_summary.execution_strategy_contract_version
            != decomposition.execution_strategy_contract_version
            or existing_summary.final_integration_work
            != decomposition.final_integration_work
            or not retained_parent_scope_matches(
                existing_summary.retained_parent_scope,
                retained_parent_scope,
                parent_issue=issue_number,
            )
        ):
            raise AgentLoopError(
                "Fresh staged execution summary disagrees with the approved normalized topology; "
                "repair the recorded stage allocation, child references, or integration obligations before rerunning."
            )

    handoffs = find_phase_implementation_handoffs_for_parent(
        issue_context.comments, parent_issue=issue_number
    )
    seen_handoff_phases: set[int] = set()
    for handoff in handoffs:
        if handoff.phase_index in seen_handoff_phases:
            raise AgentLoopError(
                "Ambiguous staged recovery: multiple phase implementation handoffs exist for one phase."
            )
        seen_handoff_phases.add(handoff.phase_index)
        if (
            handoff.plan_hash != plan_hash
            or handoff.mode != "implement-by-phase"
            or handoff.strategy != "staged"
            or handoff.topology_source != EXECUTION_TOPOLOGY_SOURCE
            or handoff.execution_strategy_contract_version
            != decomposition.execution_strategy_contract_version
            or handoff.recommendation_digest != decomposition.recommendation_digest
            or handoff.plan_subject != plan_subject
        ):
            raise AgentLoopError(
                "Fresh staged phase implementation handoff disagrees with the approved topology; "
                "repair the handoff before rerunning."
            )
        if not 1 <= handoff.phase_index <= len(decomposition.phases):
            raise AgentLoopError(
                "Fresh staged phase implementation handoff names a phase outside the approved topology."
            )
        phase = decomposition.phases[handoff.phase_index - 1]
        adopted = preflight_children[handoff.phase_index - 1]
        if (
            handoff.stage_id != phase.stage_id
            or handoff.phase_title != phase.title
            or handoff.automation != phase.automation
            or handoff.child_issue_number != adopted.issue_number
            or handoff.child_issue_url != adopted.issue_url
        ):
            raise AgentLoopError(
                "Fresh staged phase implementation handoff does not match its canonical child phase."
            )
        # Shared reconciliation rule (#808). Always discover signed records:
        # an equal phase/handoff disposition still must reject an override
        # added after dispatch, and an override-bound handoff must prove that
        # its exact record remains discoverable and unsuperseded.
        child_comments: Sequence[object] = ()
        if adopted.issue_number is not None and not config.dry_run:
            child_comments = get_issue_context(
                runner, config=config, issue_number=adopted.issue_number
            ).comments
        stage_overrides = collect_child_disposition_overrides(
            parent_comments=issue_context.comments,
            child_comments=child_comments,
            parent_issue=issue_number,
            plan_hash=plan_hash,
            topology_stage_ids=tuple(item.stage_id or "" for item in decomposition.phases),
            routed_stage_id=phase.stage_id or "",
            child_stage_id=phase.stage_id,
            child_issue_number=adopted.issue_number,
        )
        reconcile_handoff_disposition(phase, handoff, stage_overrides)
    return preflight_children


def _preflight_fresh_split_topology(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    issue_context: IssueContext,
    allowed_keys: frozenset[str] = frozenset(),
) -> None:
    """Reject unrelated split state before a fresh one-shot decision is used.

    The parent comment is the normal materialization record, but a crash can
    occur after a split child is filed and before that cumulative record is
    posted.  Search both historical child-title forms as a read-only recovery
    pass so one-shot execution cannot race an orphaned split child into a
    second topology.  A one-shot plan may intentionally retain the historical
    split-materialization option; in that case only children whose normalized
    stage keys are part of this approved request are recoverable.
    """
    materialization = find_existing_split_materialization(
        issue_context.comments,
        parent_issue=issue_number,
    )
    if materialization is not None and materialization.children:
        materialized_keys = tuple(child.key for child in materialization.children)
        if (
            len(set(materialized_keys)) != len(materialized_keys)
            or any(key not in allowed_keys for key in materialized_keys)
        ):
            raise AgentLoopError(
                "Fresh one-shot execution conflicts with an existing split materialization "
                "that is unrelated or ambiguous; resume the split topology or repair it "
                "before rerunning."
            )

    comment_keys: set[str] = set()
    for comment in issue_context.comments:
        body = getattr(comment, "body", None)
        if isinstance(body, str):
            marker = SPLIT_CHILD_MARKER_RE.search(body)
            if marker is not None and int(marker.group("parent")) == issue_number:
                key = marker.group("key")
                if key not in allowed_keys or key in comment_keys:
                    raise AgentLoopError(
                        "Fresh one-shot execution conflicts with an existing split child "
                        "that is unrelated or ambiguous; resume the split topology or repair "
                        "it before rerunning."
                    )
                comment_keys.add(key)

    searches = parent_child_search_queries(issue_number)
    found_keys: dict[str, tuple[int | None, str | None]] = {}
    for search in searches:
        for candidate in search_issues(
            runner,
            config=config,
            search=search,
            state="all",
        ):
            body = candidate.body or ""
            marker = SPLIT_CHILD_MARKER_RE.search(body)
            marker_matches = marker is not None and int(marker.group("parent")) == issue_number
            title_matches = (
                not body
                and bool(candidate.title)
                and candidate.title.startswith(f"[#{issue_number} stage] ")
            )
            if marker_matches or title_matches:
                key = (
                    marker.group("key")
                    if marker_matches
                    else split_stage_proposal_from_text(
                        candidate.title[len(f"[#{issue_number} stage] ") :]
                    ).key
                )
                if key not in allowed_keys:
                    raise AgentLoopError(
                        "Fresh one-shot execution conflicts with an existing split child "
                        "that is unrelated; resume the split topology or repair it before "
                        "rerunning."
                    )
                identity = (candidate.number, candidate.url)
                previous = found_keys.get(key)
                if previous is not None and previous != identity:
                    raise AgentLoopError(
                        "Ambiguous split-child recovery: multiple child issues match an "
                        "approved one-shot split stage."
                    )
                found_keys[key] = identity


def _approved_one_shot_split_keys(
    current_plan: str,
    *,
    issue_context: IssueContext,
    config: AgentLoopConfig,
) -> frozenset[str]:
    """Return the legacy split identities owned by this approved one-shot.

    Fresh execution recommendations intentionally do not use the legacy split
    seam for their own child topology.  They can, however, coexist with the
    explicitly supported ``--materialize-split-issues`` option, which handles
    legacy deferred stages and prior discuss split proposals.  Keep this
    derivation in lockstep with ``_handle_plan_first_split_scope`` so a rerun
    adopts exactly what the first run was allowed to materialize.
    """
    current_child_stages = _extract_current_child_stages(current_plan)
    current_deferred_stages = _extract_current_deferred_stages(current_plan)
    prior_discuss_proposals = _prior_discuss_split_proposals(issue_context, config=config)
    if current_child_stages or current_deferred_stages:
        plan_own_key = split_stage_proposal_from_text(_plan_first_line(current_plan)).key
        prior_discuss_proposals = [
            proposal
            for proposal in prior_discuss_proposals
            if split_stage_proposal_from_text(proposal).key != plan_own_key
        ]
    proposals = dedupe_split_stage_proposals(
        [split_stage_proposal_from_deferred_stage(stage) for stage in current_child_stages]
        + [split_stage_proposal_from_deferred_stage(stage) for stage in current_deferred_stages]
        + [split_stage_proposal_from_text(proposal) for proposal in prior_discuss_proposals]
    )
    return frozenset(proposal.key for proposal in proposals)


def _preflight_fresh_one_shot_recovery(
    runner: Runner,
    *,
    issue_number: int,
    approved_plan: str,
    config: AgentLoopConfig,
    issue_context: IssueContext,
    recommendation=None,
    retired_plan_hashes: frozenset[str] = frozenset(),
) -> None:
    """Validate existing one-shot handoffs before the decision record is posted."""
    plan_hash = approved_plan_hash(approved_plan)
    plan_subject = _plan_subject(approved_plan)
    if recommendation is not None:
        identity = recommendation.identity()
        # Validate durable decision identity as part of the read-only
        # preflight, rather than waiting for the publication helper after
        # other recovery checks have started.  In particular, a changed
        # approved plan must not acquire a second decision record.
        find_existing_execution_decision(
            issue_context.comments,
            parent_issue=issue_number,
            plan_hash=plan_hash,
            plan_subject=plan_subject,
            strategy="one-shot",
            recommendation_digest=str(identity["recommendation_sha256"]),
            retired_plan_hashes=retired_plan_hashes,
        )
    _preflight_fresh_split_topology(
        runner,
        issue_number=issue_number,
        config=config,
        issue_context=issue_context,
        allowed_keys=_approved_one_shot_split_keys(
            approved_plan,
            issue_context=issue_context,
            config=config,
        ),
    )
    parent_summaries = find_decompositions_for_parent(
        issue_context.comments, parent_issue=issue_number
    )
    if parent_summaries:
        recorded = parent_summaries[0]
        raise AgentLoopError(
            "Fresh one-shot execution conflicts with an existing decomposition summary for "
            f"plan {recorded.plan_hash}; resume the staged topology or repair it before rerunning."
        )
    if find_phase_implementation_handoffs_for_parent(
        issue_context.comments, parent_issue=issue_number
    ):
        raise AgentLoopError(
            "Fresh one-shot execution conflicts with an existing phase implementation handoff; "
            "resume the staged topology or repair the conflicting handoff first."
        )
    if find_topology_checkpoints_for_parent(
        issue_context.comments, parent_issue=issue_number
    ):
        raise AgentLoopError(
            "Fresh one-shot execution conflicts with an existing staged topology checkpoint; "
            "resume the staged topology or revise the approved plan before rerunning."
        )
    resolved_pr = resolve_canonical_pr_for_issue(
        runner,
        config=config,
        issue_number=issue_number,
        issue_context=issue_context,
        expected_fallback_scope=IssuePrProvenanceScope(
            repository=config.repo,
            issue_number=issue_number,
            flow="approved",
            approved_plan_hash=plan_hash,
        ),
    )
    if (
        resolved_pr is not None
        and resolved_pr.source == "canonical"
        and resolved_pr.metadata is not None
        and resolved_pr.metadata.flow == "approved-plan-implementation"
        and resolved_pr.metadata.plan_hash != plan_hash
    ):
        raise AgentLoopError(
            f"Canonical approved-plan handoff for issue #{issue_number} points to PR "
            f"#{resolved_pr.pr_number} with plan hash {resolved_pr.metadata.plan_hash}, "
            f"but the current approved plan has hash {plan_hash}. Review the recorded PR "
            f"with `agent-loop pr {resolved_pr.pr_number}` or remove the stale handoff marker."
        )

    existing_handoff = find_existing_one_shot_impl_handoff(
        issue_context.comments,
        parent_issue=issue_number,
        plan_hash=plan_hash,
        mode="implement-one-shot",
    )
    all_handoffs = find_one_shot_impl_handoffs(
        issue_context.comments,
        parent_issue=issue_number,
        mode="implement-one-shot",
    )
    same_plan_handoffs = tuple(item for item in all_handoffs if item.plan_hash == plan_hash)
    if len(same_plan_handoffs) > 1 and len(set(same_plan_handoffs)) != 1:
        raise AgentLoopError(
            "Ambiguous one-shot recovery: multiple divergent handoffs exist for the approved plan."
        )
    any_handoff = find_latest_one_shot_impl_handoff(
        issue_context.comments,
        parent_issue=issue_number,
        mode="implement-one-shot",
    )
    # A same-PR plan rebind (#936) leaves the older one-shot record in place
    # by design: the authoritative canonical handoff already binds that same
    # PR to the current plan, so the older record is history, not a conflict.
    canonical_rebinds_same_pr = bool(
        any_handoff is not None
        and resolved_pr is not None
        and resolved_pr.source == "canonical"
        and resolved_pr.metadata is not None
        and resolved_pr.metadata.plan_hash == plan_hash
        and resolved_pr.pr_number == any_handoff.pr_number
    )
    if (
        existing_handoff is None
        and any_handoff is not None
        and any_handoff.plan_hash != plan_hash
        and not canonical_rebinds_same_pr
    ):
        try:
            older_state = get_pr_state(
                runner,
                config=config,
                pr_number=any_handoff.pr_number,
            )
        except AgentLoopError as exc:
            raise AgentLoopError(
                f"Older one-shot handoff for PR #{any_handoff.pr_number} cannot be validated "
                f"({exc}). Review it directly with `agent-loop pr {any_handoff.pr_number}` "
                "or remove the stale handoff."
            ) from exc
        if older_state == "OPEN":
            raise AgentLoopError(
                f"Open one-shot handoff for PR #{any_handoff.pr_number} has older plan hash "
                f"{any_handoff.plan_hash}, but the current approved plan has hash {plan_hash}. "
                f"Review the recorded PR with `agent-loop pr {any_handoff.pr_number}` or remove "
                "the stale handoff before creating another implementation PR."
            )
    for older_handoff in all_handoffs:
        if older_handoff.plan_hash == plan_hash:
            continue
        if any_handoff is not None and older_handoff.pr_number == any_handoff.pr_number:
            continue
        try:
            older_state = get_pr_state(
                runner,
                config=config,
                pr_number=older_handoff.pr_number,
            )
        except AgentLoopError as exc:
            raise AgentLoopError(
                f"Older one-shot handoff for PR #{older_handoff.pr_number} cannot be validated "
                f"({exc}); review the recorded PR or remove the stale handoff."
            ) from exc
        if older_state == "OPEN":
            raise AgentLoopError(
                f"Open one-shot handoff for PR #{older_handoff.pr_number} has older plan hash "
                f"{older_handoff.plan_hash}; repair the stale handoff before creating another PR."
            )
    if existing_handoff is not None:
        if recommendation is not None:
            identity = recommendation.identity()
            if (
                existing_handoff.strategy != "one-shot"
                or existing_handoff.topology_source != identity["topology_source"]
                or existing_handoff.execution_strategy_contract_version != 1
                or existing_handoff.recommendation_digest != identity["recommendation_sha256"]
                or existing_handoff.plan_subject != plan_subject
            ):
                raise AgentLoopError(
                    "Fresh one-shot handoff disagrees with the approved recommendation; "
                    "repair the handoff before rerunning."
                )
        try:
            get_pr_state(runner, config=config, pr_number=existing_handoff.pr_number)
        except AgentLoopError as exc:
            raise AgentLoopError(
                f"PR #{existing_handoff.pr_number} recorded in the one-shot handoff for issue "
                f"#{issue_number} cannot be found in {config.repo}. Verify the PR exists and "
                f"rerun `agent-loop pr {existing_handoff.pr_number}` directly to continue, or "
                "remove the handoff comment from the issue and rerun to re-implement."
            ) from exc


def _handle_plan_first_split_scope(
    runner: Runner,
    *,
    issue_number: int,
    config: AgentLoopConfig,
    current_plan: str,
    plan_subject: str,
    issue_context: IssueContext,
    execution_mode: str | None = None,
    resolved_execution: ResolvedExecution | None = None,
) -> bool | NeedsHumanDecision:
    """Materialize (or warn about) split/deferred stages before implementation
    handoff (#476), so a plan-first run that narrows scope to one stage cannot
    silently leave the rest unfiled (the #467/#474 gap).

    Returns True when this call may have posted a fresh `AGENT_DISCUSS_SPLIT`
    materialization comment on the parent (#492 review): the caller must then
    refetch `issue_context` before any downstream logic (e.g. implement-one-shot
    selected-stage resolution) reads issue comments, since the in-memory
    `issue_context.comments` snapshot predates this call and would otherwise
    look stale and hide children materialized moments earlier in this same run.
    """
    # Only a fresh staged recommendation owns this seam.  Fresh one-shot plans
    # still need the historical discuss/deferred split guard: those proposals
    # are not represented by the one-shot topology and must not be silently
    # dropped when materialization was explicitly requested.
    recommendation = (
        resolved_execution.recommendation
        if resolved_execution is not None
        else None
    )
    if resolved_execution is None:
        try:
            recommendation = _current_execution_recommendation(
                current_plan, issue_context.comments
            )
            if recommendation is not None and recommendation.strategy == "staged":
                return False
        except AgentLoopError:
            # The approval boundary performs the authoritative recommendation
            # validation.  Preserve the historical split warning behavior if a
            # direct legacy caller supplies malformed, non-fresh text.
            pass
    elif recommendation is not None and recommendation.strategy == "staged":
        return False

    # Decomposition modes have exactly one topology source and are dispatched
    # below.  Keeping split materialization out of this seam prevents typed
    # stages from being filed once here and again by the decomposition path.
    effective_mode = (
        resolved_execution.action
        if resolved_execution is not None
        else execution_mode or config.plan_execution_mode
    )
    if effective_mode in {"decompose-only", "implement-by-phase"}:
        return False

    current_deferred_stages = _extract_current_deferred_stages(current_plan)
    current_child_stages = _extract_current_child_stages(current_plan)
    _log_typed_plan_stage_dispositions(current_plan, config=config)
    prior_discuss_proposals = _prior_discuss_split_proposals(issue_context, config=config)
    if current_child_stages or current_deferred_stages:
        # This plan structurally declares its own deferred_stages, so it keeps
        # a primary scope on the parent (the implement-one-shot branch below
        # never hands that scope off to a child in this case). If a prior
        # discuss split proposal names that same primary scope, it must be
        # excluded here rather than filed as a duplicate child issue for work
        # the parent PR is about to implement and close directly (#492 review).
        plan_own_key = split_stage_proposal_from_text(_plan_first_line(current_plan)).key
        prior_discuss_proposals = [
            proposal
            for proposal in prior_discuss_proposals
            if split_stage_proposal_from_text(proposal).key != plan_own_key
        ]
    remaining_proposals = dedupe_split_stage_proposals(
        [split_stage_proposal_from_deferred_stage(stage) for stage in current_child_stages]
        + [split_stage_proposal_from_text(proposal) for proposal in prior_discuss_proposals]
    )
    if remaining_proposals:
        if config.materialize_split_issues:
            materialized = materialize_split_proposals(
                runner,
                config=config,
                parent_issue=issue_number,
                subject=plan_subject,
                proposals=remaining_proposals,
                issue_comments=issue_context.comments,
            )
            if isinstance(materialized, NeedsHumanDecision):
                return materialized
            return True
        log(
            config,
            f"Planning issue #{issue_number}: split follow-ups remain unfiled; rerun with "
            "--materialize-split-issues or file them manually.",
        )
        if not has_unfiled_split_warning(
            issue_context.comments, issue_number=issue_number, subject=plan_subject
        ):
            post_unfiled_split_warning(
                runner,
                config=config,
                issue_number=issue_number,
                subject=plan_subject,
                proposals=remaining_proposals,
            )
        return False
    if current_deferred_stages or prior_discuss_proposals:
        return False
    if not _plan_text_suggests_narrowing(current_plan):
        return False
    log(
        config,
        f"Planning issue #{issue_number}: approved plan text suggests scope narrowing "
        "but no `deferred_stages` or discuss split proposals were declared or filed.",
    )
    if has_unfiled_split_warning(issue_context.comments, issue_number=issue_number, subject=plan_subject):
        return False
    post_issue_comment(
        runner,
        config=config,
        issue_number=issue_number,
        body=TrustedBody.canonical(
            "\n".join(
                [
                    "### Possible unfiled scope narrowing",
                    "",
                    "The approved plan's text appears to narrow scope (mentions a stage, "
                    "follow-up issue, or out-of-scope work), but no `deferred_stages` were "
                    "structurally declared and no discuss split proposals exist for this issue. "
                    "If this plan intentionally defers work, declare it via `deferred_stages` in a "
                    "future revision or file the follow-up issue(s) manually. This is a heuristic "
                    "warning only; the orchestrator never auto-creates issues from prose.",
                    "",
                    f"<!-- AGENT_SPLIT_UNFILED_WARNING: issue={issue_number} subject={plan_subject} -->",
                    "-- coding-review-agent-loop",
                ]
            ),
            expected_tokens=("AGENT_SPLIT_UNFILED_WARNING",),
        ),
    )
    return False


def _infer_staged_parent_issue(issue_context: IssueContext) -> int | None:
    """Read only generated child-issue markers for direct staged safety checks."""
    candidates: set[int] = set()
    issue_body = issue_context.body or ""
    bodies = [issue_body]
    bodies.extend(comment.body or "" for comment in issue_context.comments)
    for body in bodies:
        for match in SPLIT_CHILD_MARKER_RE.finditer(body):
            candidates.add(int(match.group("parent")))
    first_line = issue_body.splitlines()[0].strip() if issue_body.splitlines() else ""
    decomposition_match = re.fullmatch(
        r"Child phase issue for parent #(?P<parent>\d+)\b.*",
        first_line,
        re.IGNORECASE,
    )
    if decomposition_match:
        candidates.add(int(decomposition_match.group("parent")))
    if len(candidates) > 1:
        joined = ", ".join(f"#{number}" for number in sorted(candidates))
        raise AgentLoopError(
            f"Issue #{issue_context.number} contains conflicting generated staged-parent "
            f"markers ({joined}); resolve the child issue metadata before running issue mode."
        )
    return next(iter(candidates), None)


def _resolved_stage_ids(
    created: Sequence[CreatedPhaseIssue],
    *,
    normalized_topology=None,
    recorded_stage_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    """Resolve one stage identity per phase index, by a single rule.

    The normalized topology is authoritative when the run is fresh; an adopted
    or legacy summary supplies its recorded stage ids when it has them; and the
    remaining case falls back to the 1-based ordinal already used by the
    existing status lines.
    """
    phases = normalized_topology[0].phases if normalized_topology is not None else ()
    resolved: list[str] = []
    for index, item in enumerate(created, start=1):
        if index <= len(phases):
            phase = phases[index - 1]
            resolved.append(phase.stage_id or str(phase.position or index))
            continue
        if len(recorded_stage_ids) == len(created) and recorded_stage_ids[index - 1]:
            resolved.append(recorded_stage_ids[index - 1])
            continue
        stage_id = getattr(item.phase, "stage_id", None)
        resolved.append(stage_id or str(getattr(item.phase, "position", None) or index))
    return tuple(resolved)


def staged_plan_mode_conflict_message(mode: str) -> str:
    """Operator diagnostic for a staged plan under a guard-active mode (#1268)."""
    if mode == "plan-only":
        remedy = (
            f"staged plan under plan-only: approval in this mode creates no tracked child "
            "issues, so the phased-delivery guard cannot be satisfied. Rerun with "
            "--plan-execution-mode decompose-only (creates tracked stage issues, implements "
            "nothing, keeps review before implementation) or implement-by-phase; or rerun "
            "with --plan-narrow-staged to have the planner narrow the plan to one "
            "deliverable."
        )
    else:
        remedy = (
            f"staged plan under {mode}: this mode implements one PR and cannot deliver a "
            "staged plan. Rerun with --plan-execution-mode implement-by-phase, "
            "decompose-only, or auto; or rerun with --plan-narrow-staged to have the "
            "planner narrow the plan to one deliverable."
        )
    return (
        f"{remedy} No reviewer turn and no planner revision turn was invoked for this "
        "candidate."
    )
